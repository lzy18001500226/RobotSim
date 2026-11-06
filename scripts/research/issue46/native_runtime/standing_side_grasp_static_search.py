#!/usr/bin/env python3
"""Search source-preserving upright-bottle side-grasp poses with zero waist roll.

This is a zero-step MuJoCo geometric screen. It varies a free-base placement,
waist yaw/pitch and the right arm, while retaining both frozen STL hands and
the original table/cylinder geometry. It never publishes commands.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from scripts.research.issue46.native_runtime import native_truth_docking_ik as native  # noqa: E402


ARM = native.ARM
LEFT_ARM = tuple("left_" + name[len("right_"):] for name in ARM)
WAIST = native.WAIST
ACTIVE = ("waist_yaw_joint", "waist_pitch_joint") + ARM
DISTALS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
# From the original 58.355 mm physical STL side-grasp trial.  Cylinder is
# upright in that scene; this transforms the hand body, rather than merely
# forcing an ambiguous jaw-centre coincidence.
REFERENCE_HAND_POSITION = np.array((-0.04467184507381172, 0.13716217215236792, 0.09598163959620898))
REFERENCE_HAND_ROTATION = np.array((
    (-0.014915656670289656, -0.9509015020932154, 0.3091340429376232),
    (0.024445240310342287, -0.309422818859578, -0.9506103036445386),
    (0.9995898926120967, -0.006622120948683863, 0.02786026026219679),
))


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted_state(record: dict[str, object]) -> tuple[dict[str, object], str]:
    docked = record.get("native_dock_state_capture")
    gates = record.get("gates")
    if (
        isinstance(docked, dict)
        and isinstance(gates, dict)
        and all(gates.get(name) == "PASS" for name in (
            "NATIVE_TABLE_DOCK_ARRIVAL",
            "NATIVE_TABLE_DOCK_STOP_HOLD",
            "NATIVE_TABLE_DOCK_STATE_HANDOFF",
        ))
    ):
        return docked, "NATIVE_MC_TABLE_DOCK_CAPTURE"
    state = record.get("native_arm_state_capture")
    if isinstance(state, dict):
        return state, "NATIVE_STAND_CAPTURE"
    state = record.get("stabilized_state")
    if isinstance(state, dict) and record.get("gates", {}).get("FULL_CONTACT_STAND") == "PASS":
        return state, "FOUNDATIONWALK_STEPPED_CAPTURE"
    raise ValueError("standing input lacks an accepted complete state")


def set_pose(model, data, state, q, base_offset, yaw_offset, command, cylinder_delta, knee_bend=0.0, left_arm_safe=None):
    upper = np.array(
        [q[0], q[1], float(state["waist_positions_rad"]["waist_roll_joint"])] + q[2:].tolist(),
        dtype=float,
    )
    native.set_native_state(model, data, state, upper, command, base_offset, yaw_offset, None)
    if left_arm_safe is not None:
        for name, value in left_arm_safe.items():
            native.set_joint(model, data, name, float(value))
    if knee_bend:
        feet = [native.bid(model, side + "_ankle_roll_link") for side in ("left", "right")]
        mujoco.mj_forward(model, data)
        reference_height = float(np.mean(data.xpos[feet, 2]))
        for side in ("left", "right"):
            for joint, delta in (("hip_pitch_joint", -0.5 * knee_bend), ("knee_joint", knee_bend), ("ankle_pitch_joint", -0.5 * knee_bend)):
                joint_id = native.jid(model, side + "_" + joint)
                data.qpos[model.jnt_qposadr[joint_id]] += delta
        mujoco.mj_forward(model, data)
        free = native.jid(model, "floating_base_joint")
        data.qpos[model.jnt_qposadr[free] + 2] += reference_height - float(np.mean(data.xpos[feet, 2]))
    cylinder_free = native.jid(model, "test_cylinder_free")
    cylinder_address = model.jnt_qposadr[cylinder_free]
    if 'object_pose_world' in state:
        # Offline planning MjData only, just like the measured base/arm state.
        # Do not confuse an XML default object pose with this episode's truth.
        measured=state['object_pose_world']
        data.qpos[cylinder_address:cylinder_address+3]=measured['xyz_m']
        data.qpos[cylinder_address+3:cylinder_address+7]=measured['quaternion_wxyz']
    data.qpos[cylinder_address:cylinder_address + 3] += cylinder_delta
    data.qvel[model.jnt_dofadr[cylinder_free]:model.jnt_dofadr[cylinder_free] + 6] = 0.0
    mujoco.mj_forward(model, data)


def jac_point(model, data, body, dofs):
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, linear, angular, body)
    return linear[:, dofs]


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(np.cross(current[:, index], target[:, index]) for index in range(3))


def contacts(model, data, cylinder):
    bad, pads, table = [], set(), False
    for contact in data.contact:
        one, two = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        if cylinder not in (one, two):
            continue
        other = two if one == cylinder else one
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or "world"
        if name in DISTALS:
            pads.add(name)
        elif name == "raicom_table_table":
            table = True
        else:
            bad.append(name)
    robot_table = []
    table_body = native.bid(model, "raicom_table_table")
    for contact in data.contact:
        one, two = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        if table_body not in (one, two):
            continue
        other = two if one == table_body else one
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or "world"
        if name != "test_cylinder":
            robot_table.append(name)
    return sorted(pads), sorted(set(bad)), table, sorted(set(robot_table))


def camera_bearing_error(model, data, cylinder) -> float:
    """Angle from the delivered head-forward axis to the cylinder centre."""
    head = native.bid(model, "head_pitch_link")
    forward = data.xmat[head].reshape(3, 3)[:, 0]
    displacement = data.xpos[cylinder] - data.xpos[head]
    horizontal = displacement.copy(); horizontal[2] = 0.0
    if np.linalg.norm(horizontal) <= 1e-8:
        return float("inf")
    return float(np.arccos(np.clip(np.dot(forward[:2], horizontal[:2]) / (np.linalg.norm(forward[:2]) * np.linalg.norm(horizontal[:2])), -1.0, 1.0)))


def solve(model, state, base_offset, yaw_offset, command, seed, cylinder_delta, grasp_azimuth, hand_world_offset, knee_bend, max_waist_deviation, left_arm_safe, object_center_world=None):
    # A zero deviation means the MC-owned waist is deliberately fixed, not
    # that its two coordinates must satisfy a non-zero solver margin.  Keep
    # those joints out of the optimization vector rather than clipping an
    # empty [lower + eps, upper - eps] interval.
    active_indices = np.arange(2, len(ACTIVE), dtype=int) if max_waist_deviation == 0.0 else np.arange(len(ACTIVE), dtype=int)
    active_names = tuple(ACTIVE[index] for index in active_indices)
    dofs = np.array([model.jnt_dofadr[native.jid(model, name)] for name in active_names], dtype=int)
    bounds = np.array([model.jnt_range[native.jid(model, name)] for name in active_names], dtype=float)
    if max_waist_deviation is not None and max_waist_deviation > 0.0:
        for index, name in enumerate(("waist_yaw_joint", "waist_pitch_joint")):
            centre = float(seed[index])
            bounds[index, 0] = max(bounds[index, 0], centre - max_waist_deviation)
            bounds[index, 1] = min(bounds[index, 1], centre + max_waist_deviation)
    q = seed.copy()
    elbow, wrist, narrow, wide, cylinder, hand = (
        native.bid(model, name)
        for name in ("right_elbow_link", "right_wrist_roll_link", "R_hand_narrow3_Link", "R_hand_wide3_Link", "test_cylinder", "R_omnipicker_base_link")
    )
    for iteration in range(120):
        data = mujoco.MjData(model)
        set_pose(model, data, state, q, base_offset, yaw_offset, command, cylinder_delta, knee_bend, left_arm_safe)
        cylinder_rotation = data.xmat[cylinder].reshape(3, 3)
        hand_rotation = data.xmat[hand].reshape(3, 3)
        c, s = math.cos(grasp_azimuth), math.sin(grasp_azimuth)
        around_cylinder = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        object_center = data.xpos[cylinder] if object_center_world is None else object_center_world
        desired_position = object_center + cylinder_rotation @ around_cylinder @ REFERENCE_HAND_POSITION + hand_world_offset
        desired_rotation = cylinder_rotation @ around_cylinder @ REFERENCE_HAND_ROTATION
        centre_error = desired_position - data.xpos[hand]
        angle_error = rotation_error(hand_rotation, desired_rotation)
        active_q = q[active_indices]
        margin = np.min(np.minimum(active_q - bounds[:, 0], bounds[:, 1] - active_q))
        if np.linalg.norm(centre_error) <= 1e-4 and np.linalg.norm(angle_error) <= 1e-3 and margin >= 1e-3:
            break
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, linear, angular, hand)
        error = np.r_[centre_error, angle_error]
        jac = np.vstack((linear[:, dofs], angular[:, dofs]))
        inv = np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), np.eye(6))
        delta = jac.T @ inv @ error - 2e-4 * (active_q - seed[active_indices])
        # Six pose constraints leave three joint-space directions available.
        # Use them to retreat from source range endpoints rather than accepting
        # a visually plausible candidate with no physical tracking margin.
        low_gap, high_gap = active_q - bounds[:, 0], bounds[:, 1] - active_q
        endpoint_bias = 0.001 * (1.0 / (low_gap + 0.02) - 1.0 / (high_gap + 0.02))
        nullspace = np.eye(len(active_indices)) - jac.T @ inv @ jac
        delta += nullspace @ np.clip(endpoint_bias, -0.02, 0.02)
        q[active_indices] += np.clip(delta, -0.03, 0.03)
        q[active_indices] = np.clip(q[active_indices], bounds[:, 0] + 1e-5, bounds[:, 1] - 1e-5)
    data = mujoco.MjData(model)
    set_pose(model, data, state, q, base_offset, yaw_offset, command, cylinder_delta, knee_bend, left_arm_safe)
    cylinder_rotation = data.xmat[cylinder].reshape(3, 3)
    c, s = math.cos(grasp_azimuth), math.sin(grasp_azimuth)
    around_cylinder = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    object_center = data.xpos[cylinder] if object_center_world is None else object_center_world
    desired_position = object_center + cylinder_rotation @ around_cylinder @ REFERENCE_HAND_POSITION + hand_world_offset
    desired_rotation = cylinder_rotation @ around_cylinder @ REFERENCE_HAND_ROTATION
    pads, bad, table, robot_table = contacts(model, data, cylinder)
    return q, {
        "iterations": iteration + 1,
        "hand_position_error_m": float(np.linalg.norm(desired_position - data.xpos[hand])),
        "hand_orientation_error_rad": float(np.linalg.norm(rotation_error(data.xmat[hand].reshape(3, 3), desired_rotation))),
        "minimum_joint_margin_rad": float(np.min(np.minimum(q[active_indices] - bounds[:, 0], bounds[:, 1] - q[active_indices]))),
        "mc_owned_waist_fixed": bool(max_waist_deviation == 0.0),
        "pad_bodies": pads,
        "nonpad_cylinder_bodies": bad,
        "table_support": table,
        "robot_table_bodies": robot_table,
        "camera_bearing_error_rad": camera_bearing_error(model, data, cylinder),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--standing", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--object-center-world", type=float, nargs=3, help="external perception estimate in world metres; object truth remains collision-only")
    parser.add_argument("--perception-rgbd", type=Path, help="accepted known-cylinder RGB-D record; supplies centre and provenance")
    parser.add_argument("--require-camera-facing", action="store_true", help="second-stage visual constraint; disabled for first-stage truth grasp")
    parser.add_argument("--hand-target-world-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="measured dynamic jaw-centre correction applied to the static hand target")
    parser.add_argument("--command-percent", type=int, default=38)
    parser.add_argument("--base-x-offsets", type=float, nargs="+", default=(0.0, 0.10), help="world-X offsets around the measured native dock state")
    parser.add_argument("--base-y-offsets", type=float, nargs="+", default=(-0.30, -0.20), help="world-Y offsets around the measured native dock state")
    parser.add_argument("--base-yaw-offsets", type=float, nargs="+", default=(-0.20, 0.0, 0.20), help="yaw offsets around the measured native dock state")
    parser.add_argument("--knee-bends", type=float, nargs="+", default=(0.0,), help="additional symmetric knee bend; keep 0 to preserve the measured natural stance")
    parser.add_argument("--max-waist-deviation-rad", type=float, default=0.45, help="maximum yaw/pitch deviation from the measured natural waist state")
    parser.add_argument("--grasp-azimuths", type=float, nargs="+", default=(math.pi / 4.0,), help="source-referenced hand placements around the upright cylinder")
    parser.add_argument("--left-arm-safe", type=float, nargs=7, metavar=("SP", "SR", "SY", "E", "WY", "WP", "WR"), help="optional source-recorded left-arm safe target used to keep the non-grasping hand clear")
    args = parser.parse_args()
    table = json.loads(args.table.read_text())
    standing_record = json.loads(args.standing.read_text())
    placement = json.loads(args.placement.read_text())
    hand_world_offset = np.asarray(args.hand_target_world_offset, dtype=float)
    if args.object_center_world is not None and args.perception_rgbd is not None:
        raise ValueError("choose either an explicit object centre or an RGB-D perception record")
    perception = None
    object_center_world = None if args.object_center_world is None else np.asarray(args.object_center_world, dtype=float)
    if args.perception_rgbd is not None:
        perception_path = args.perception_rgbd.resolve()
        perception = json.loads(perception_path.read_text(encoding="utf-8"))
        if perception.get("schema") != "robotsim.issue46.rgbd-known-blue-cylinder.v1":
            raise ValueError("RGB-D record has an unexpected schema")
        gates = perception.get("gates", {})
        proposal_pass = gates.get("YOLO_GEOMETRIC_PROPOSAL") == "PASS" or gates.get("ACTIVE_RGB_GEOMETRIC_PROPOSAL") == "PASS"
        required_perception_gates = ("OBJECT_MASK", "RGBD_TOP_PLANE")
        if not proposal_pass or any(gates.get(name) != "PASS" for name in required_perception_gates):
            raise ValueError("RGB-D record does not pass its geometric proposal gates")
        object_center_world = np.asarray(perception.get("estimated_center_world_m"), dtype=float)
    if table.get("gates", {}).get("TABLE_SCENE_STATIC_COMPILE") != "PASS":
        raise ValueError("table scene does not pass its static compile gate")
    if placement.get("classification") != "HUMAN_SELECTED_STATIC_TABLE_PLACEMENT" or placement.get("table_support_contact") is not True:
        raise ValueError("placement must be a human-selected table-supported record")
    if hand_world_offset.shape != (3,) or not np.isfinite(hand_world_offset).all() or np.linalg.norm(hand_world_offset) > 0.15:
        raise ValueError("hand target offset must be a finite vector no larger than 150 mm")
    if object_center_world is not None and (object_center_world.shape != (3,) or not np.isfinite(object_center_world).all()):
        raise ValueError("external object center must be a finite world three-vector")
    search_values = tuple(args.base_x_offsets) + tuple(args.base_y_offsets) + tuple(args.base_yaw_offsets) + tuple(args.knee_bends) + tuple(args.grasp_azimuths) + (args.max_waist_deviation_rad,)
    if not all(np.isfinite(value) for value in search_values):
        raise ValueError("all local-placement search values must be finite")
    if any(abs(value) > 0.6 for value in args.base_x_offsets + args.base_y_offsets) or any(abs(value) > 0.5 for value in args.base_yaw_offsets):
        raise ValueError("local base placement search exceeds the bounded 0.6 m / 0.5 rad range")
    if any(value < 0.0 or value > 0.35 for value in args.knee_bends):
        raise ValueError("additional knee bend must lie in 0..0.35 rad")
    if not 0.0 <= args.max_waist_deviation_rad <= 0.6:
        raise ValueError("max waist deviation must lie in 0..0.6 rad")
    state, source = accepted_state(standing_record)
    left_arm_safe = None
    if args.left_arm_safe is not None:
        left_arm_safe = {name: float(value) for name, value in zip(LEFT_ARM, args.left_arm_safe)}
    scene = Path(table["generated"]["host_scene"]["path"])
    if sha256(scene) != table["generated"]["host_scene"]["sha256"]:
        raise ValueError("table-scene hash mismatch")
    model = mujoco.MjModel.from_xml_path(str(scene))
    arm = state["arm_positions_rad"]
    waist = state["waist_positions_rad"]
    seed = np.array([waist["waist_yaw_joint"], waist["waist_pitch_joint"]] + [arm[name] for name in ARM], dtype=float)
    rows = []
    # A legacy table overlay contains the source cylinder pose and needs the
    # saved human placement delta.  Newer overlays already embed that exact
    # human-selected world pose, so applying it again would silently solve IK
    # against a bottle translated twice.
    embedded = table.get("cylinder", {}).get("human_selected_placement")
    if embedded is not None:
        if not isinstance(embedded, dict) or embedded.get("sha256") != sha256(args.placement):
            raise ValueError("embedded scene bottle placement does not match the selected placement record")
        cylinder_deltas = ((0.0, 0.0, 0.0),)
        placement_application = "already_embedded_in_scene"
    else:
        cylinder_deltas = (tuple(float(value) for value in placement["cylinder_delta_world_m"]),)
        placement_application = "runtime_delta_from_source_pose"
    for cylinder_delta in cylinder_deltas:
        for knee_bend in args.knee_bends:
            for forward in args.base_x_offsets:
                for lateral in args.base_y_offsets:
                    for yaw in args.base_yaw_offsets:
                        for grasp_azimuth in args.grasp_azimuths:
                            q, metrics = solve(model, state, np.array([forward, lateral, 0.0]), yaw, args.command_percent, seed, np.asarray(cylinder_delta, dtype=float), grasp_azimuth, hand_world_offset, knee_bend, args.max_waist_deviation_rad, left_arm_safe, object_center_world)
                            camera_ok = metrics["camera_bearing_error_rad"] <= np.deg2rad(15.0)
                            passed = metrics["hand_position_error_m"] <= 1e-4 and metrics["hand_orientation_error_rad"] <= 1e-3 and (camera_ok or not args.require_camera_facing) and metrics["minimum_joint_margin_rad"] >= 1e-3 and set(metrics["pad_bodies"]) == DISTALS and not metrics["nonpad_cylinder_bodies"] and not metrics["robot_table_bodies"]
                            rows.append({"cylinder_delta_world_m": list(cylinder_delta), "base_offset_world_m": [forward, lateral, 0.0], "base_yaw_offset_rad": yaw, "grasp_azimuth_rad": grasp_azimuth, "knee_bend_rad": knee_bend, "waist_roll_rad": float(waist["waist_roll_joint"]), "upper_q_rad": {name: float(value) for name, value in zip(ACTIVE, q)}, "metrics": metrics, "pass": bool(passed)})
    # Preserve visual-input provenance separately from source geometry used by
    # MuJoCo only for collision checks.  Camera bearing itself is still a
    # second-stage constraint because the current offline RGB-D fixture does
    # not establish a live vendor optical transform.
    # Rank physically usable joint margin before a cosmetic view-angle score;
    # otherwise the selected candidate can sit milliradians from a source limit.
    best = sorted(rows, key=lambda row: (not row["pass"], -row["metrics"]["minimum_joint_margin_rad"], len(row["metrics"]["nonpad_cylinder_bodies"]), len(row["metrics"]["robot_table_bodies"]), row["metrics"]["hand_position_error_m"], row["metrics"]["hand_orientation_error_rad"], row["metrics"]["camera_bearing_error_rad"]))
    output = pg.create_unique_output_directory(EVIDENCE / "reviews/issue-46", prefix="stl-standing-side-grasp-static-search").native_path
    result = {
        "schema": "robotsim.issue46.stl-standing-side-grasp-static-search.v1",
        "classification": "SIMULATION_ONLY_STATIC_BASE_PLACEMENT_AND_IK_SEARCH",
        "mj_step_calls": 0,
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "standing": {"path": str(args.standing), "sha256": sha256(args.standing), "state_source": source},
        "placement": {"path": str(args.placement), "sha256": sha256(args.placement), "world_xyz_m": placement["cylinder_world_m"]},
        "placement_application": placement_application,
        "object_target_world_m": None if object_center_world is None else object_center_world.tolist(),
        "perception": None if perception is None else {
            "path": str(args.perception_rgbd.resolve()),
            "sha256": sha256(args.perception_rgbd),
            "estimated_center_world_m": object_center_world.tolist(),
            "evaluation_only_center_error_mm": (perception.get("evaluation_only") or {}).get("center_error_mm_evaluation_only"),
        },
        "hand_target_world_offset_m": hand_world_offset.tolist(),
        "local_base_search": {"x_offsets_m": list(args.base_x_offsets), "y_offsets_m": list(args.base_y_offsets), "yaw_offsets_rad": list(args.base_yaw_offsets), "additional_knee_bends_rad": list(args.knee_bends), "grasp_azimuths_rad": list(args.grasp_azimuths), "max_waist_deviation_rad": args.max_waist_deviation_rad},
        "left_arm_safe_target_rad": left_arm_safe,
        "command_percent": args.command_percent,
        "requirement": "waist roll remains the stable standing capture value; search varies robot placement, yaw, waist yaw/pitch, and right arm" + ("; camera horizontal bearing error must be at most 15 degrees" if args.require_camera_facing else "; camera constraint deferred until second-stage visual grasp"),
        "reference_hand_relative_to_upright_cylinder": {"position_m": REFERENCE_HAND_POSITION.tolist(), "rotation_matrix": REFERENCE_HAND_ROTATION.tolist()},
        "rows": rows,
        "best_candidates": best[:12],
        "gates": {
            "STATIC_SEARCH": "PASS",
            "VISUAL_GRASP_CANDIDATE": "PASS" if perception is not None and any(row["pass"] for row in rows) else ("NOT_RUN" if perception is None else "FAIL"),
            "TRUTH_GRASP_CANDIDATE": "PASS" if perception is None and any(row["pass"] for row in rows) else "NOT_RUN",
            "CAMERA_FACING": "PASS" if args.require_camera_facing and any(row["pass"] for row in rows) else "DEFERRED",
            "DYNAMICS": "NOT_RUN",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": result["gates"], "best": best[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
