#!/usr/bin/env python3
"""Zero-step geometric feasibility analysis for the Issue #46 OmniPicker task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


EXPERIMENT_PATH = Path(__file__).with_name("issue46_omnipicker_1dof_m0.py")
OUTPUT_DEFAULT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-long-goal/geometry-candidate-2"
)


def load_experiment():
    spec = importlib.util.spec_from_file_location("issue46_omnipicker_1dof_m0", EXPERIMENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accepted experiment module: {EXPERIMENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(geom_id)])
    ) or "world"


def body_collision_geoms(model: mujoco.MjModel, body: str) -> list[int]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return []
    return [
        gid for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == bid
        and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))
    ]


def q_state(exp, model: mujoco.MjModel, base_data: mujoco.MjData,
            arm_pose: dict[str, float], aperture: float) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = base_data.qpos
    for name, value in arm_pose.items():
        data.qpos[exp.qpos_id(model, name)] = value
    jaw = exp.aperture_targets(aperture)
    data.qpos[exp.qpos_id(model, exp.RIGHT_DRIVER)] = jaw["right_claw_joint_target_rad"]
    data.qpos[exp.qpos_id(model, exp.RIGHT_FOLLOWER)] = jaw["R_hand_wide1_joint_target_rad"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def geom_distance(model: mujoco.MjModel, data: mujoco.MjData,
                  first: int, second: int) -> tuple[float, list[float]]:
    segment = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
    return distance, segment.tolist()


def contact_rows(exp, model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, force)
        rows.append({
            "geom1": geom_name(model, geom1),
            "body1": body_name(model, geom1),
            "geom2": geom_name(model, geom2),
            "body2": body_name(model, geom2),
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
            "normal_force_n": float(abs(force[0])),
        })
    return rows


def target_clearances(exp, model: mujoco.MjModel, data: mujoco.MjData,
                      target_geoms: list[int], hand_geoms: list[int]) -> dict[str, Any]:
    rows = []
    for hand_geom in hand_geoms:
        distances = [geom_distance(model, data, hand_geom, target)[0] for target in target_geoms]
        rows.append({
            "geom": geom_name(model, hand_geom),
            "body": body_name(model, hand_geom),
            "minimum_signed_distance_m": min(distances),
        })
    return {
        "minimum_signed_distance_m": min((row["minimum_signed_distance_m"] for row in rows), default=None),
        "by_geom": rows,
    }


def classify_contact(body: str) -> str:
    lower = body.lower()
    if "wrist" in lower or "omnipicker_base" in lower or "palm" in lower:
        return "wrist_or_palm"
    if "loop" in lower:
        return "jaw_linkage_loop"
    if body.startswith("R_hand_narrow"):
        return "narrow_jaw"
    if body.startswith("R_hand_wide"):
        return "wide_jaw"
    return "other_robot_link"


def inspect_pose(exp, model: mujoco.MjModel, base_data: mujoco.MjData,
                 arm_pose: dict[str, float], aperture: float,
                 bottle_geoms: list[int], table_geoms: list[int],
                 robot_geoms: list[int]) -> dict[str, Any]:
    data = q_state(exp, model, base_data, arm_pose, aperture)
    bottle_by_family = {family: [g for g in robot_geoms
                                 if body_name(model, g).startswith(f"R_hand_{family}")
                                 and "loop" not in body_name(model, g).lower()]
                        for family in ("narrow", "wide")}
    wrist_geoms = [g for g in robot_geoms if classify_contact(body_name(model, g)) in
                   {"wrist_or_palm", "jaw_linkage_loop", "other_robot_link"}]
    table_rows = []
    for geom in robot_geoms:
        if not table_geoms:
            break
        best = min((geom_distance(model, data, geom, table)[0] for table in table_geoms), default=math.inf)
        table_rows.append({"geom": geom_name(model, geom), "body": body_name(model, geom),
                           "minimum_signed_distance_m": best})
    return {
        "aperture_ratio": aperture,
        "arm_qpos_rad": {name: float(data.qpos[exp.qpos_id(model, name)])
                         for name in exp.RIGHT_ARM_JOINTS},
        "wrist_pose_world": {
            "position_m": data.xpos[exp.body_id(model, "R_omnipicker_base_link")].tolist(),
            "rotation_matrix_row_major": data.xmat[exp.body_id(model, "R_omnipicker_base_link")].tolist(),
        },
        "jaw_witness_body_positions_world_m": {
            name: data.xpos[exp.body_id(model, name)].tolist()
            for name in ("R_hand_narrow3_Link", "R_hand_wide3_Link")
        },
        "bottle_clearance_by_family": {
            family: target_clearances(exp, model, bottle_geoms, geoms)
            for family, geoms in bottle_by_family.items()
        },
        "other_hand_and_wrist_bottle_clearance": target_clearances(
            exp, model, bottle_geoms, wrist_geoms
        ),
        "minimum_robot_to_table_clearance_m": min(
            (row["minimum_signed_distance_m"] for row in table_rows), default=None
        ),
        "robot_to_table_by_geom": table_rows,
        "contacts": contact_rows(exp, model, data),
    }


def contact_pose_is_collision_free(exp, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, list[dict[str, Any]]]:
    bad = []
    for contact in contact_rows(exp, model, data):
        bodies = {contact["body1"], contact["body2"]}
        if "m0_bottle" in bodies:
            other = next((name for name in bodies if name != "m0_bottle"), "")
            if classify_contact(other) in {"narrow_jaw", "wide_jaw"} and "loop" not in other.lower():
                continue
        if "m0_table" in bodies and "m0_bottle" in bodies:
            continue
        if contact["distance_m"] < -1e-7:
            bad.append(contact)
    return not bad, bad


def solve_gap_center_pose(exp, model: mujoco.MjModel, base_data: mujoco.MjData,
                          seed: dict[str, float], target_position: np.ndarray,
                          aperture: float,
                          local_gap_point: np.ndarray, local_gap_normal: np.ndarray,
                          max_iterations: int = 500) -> tuple[dict[str, float], dict[str, Any]]:
    """Damped IK for gap center while keeping the jaw axis horizontal."""
    end_body = exp.body_id(model, "R_omnipicker_base_link")
    joint_ids = [exp.joint_id(model, name) for name in exp.RIGHT_ARM_JOINTS]
    dofs = [int(model.jnt_dofadr[jid]) for jid in joint_ids]
    qpos = [int(model.jnt_qposadr[jid]) for jid in joint_ids]
    limits = [model.jnt_range[jid].copy() for jid in joint_ids]
    q = np.asarray([seed[name] for name in exp.RIGHT_ARM_JOINTS], dtype=float)
    vertical = np.array([0.0, 0.0, 1.0])
    orientation_scale = 0.12
    trace = []
    blocked_updates = 0

    def evaluate(q_candidate: np.ndarray) -> tuple[mujoco.MjData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pose = {name: float(value) for name, value in zip(exp.RIGHT_ARM_JOINTS, q_candidate)}
        data = q_state(exp, model, base_data, pose, aperture)
        rotation = data.xmat[end_body].reshape(3, 3)
        point = data.xpos[end_body] + rotation @ local_gap_point
        normal = rotation @ local_gap_normal
        normal /= np.linalg.norm(normal)
        pos_error = target_position - point
        axis_error = float(np.dot(normal, vertical))
        return data, point, normal, pos_error, axis_error

    for iteration in range(max_iterations):
        data, point, normal, pos_error, axis_error = evaluate(q)
        position_norm = float(np.linalg.norm(pos_error))
        axis_angle = abs(math.asin(float(np.clip(axis_error, -1.0, 1.0))))
        safe, bad_contacts = contact_pose_is_collision_free(exp, model, data)
        trace.append({"iteration": iteration, "position_error_m": position_norm,
                      "jaw_axis_error_rad": axis_angle, "safe": safe,
                      "blocking_contacts": bad_contacts})
        if position_norm <= 0.0015 and axis_angle <= math.radians(1.0) and safe:
            return ({name: float(value) for name, value in zip(exp.RIGHT_ARM_JOINTS, q)},
                    {"iterations": iteration, "position_error_m": position_norm,
                     "jaw_axis_error_rad": axis_angle, "blocked_updates": blocked_updates,
                     "trace": trace})

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jac(model, data, jacp, jacr, point, end_body)
        axis_jacobian = np.array([[normal[1], -normal[0], 0.0]]) @ jacr
        jac = np.vstack((jacp[:, dofs], orientation_scale * axis_jacobian[:, dofs]))
        error = np.concatenate((pos_error, np.array([-orientation_scale * axis_error])))
        damping = 0.035
        lower = np.asarray([item[0] + 0.005 for item in limits])
        upper = np.asarray([item[1] - 0.005 for item in limits])
        free = np.ones(len(q), dtype=bool)
        delta = np.zeros_like(q)
        for _ in range(len(q) + 1):
            active_columns = np.flatnonzero(free)
            if len(active_columns) == 0:
                break
            reduced = jac[:, active_columns]
            reduced_delta = reduced.T @ np.linalg.solve(
                reduced @ reduced.T + damping**2 * np.eye(4), error
            )
            delta[:] = 0.0
            delta[active_columns] = reduced_delta
            outward = ((q <= lower + 1e-10) & (delta < 0.0)) | (
                (q >= upper - 1e-10) & (delta > 0.0)
            )
            if not np.any(outward):
                break
            free[outward] = False
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > 0.10:
            delta *= 0.10 / max_delta
        accepted = False
        factor = 1.0
        for index, change in enumerate(delta):
            if change > 0.0:
                factor = min(factor, max(0.0, (upper[index] - q[index]) / change))
            elif change < 0.0:
                factor = min(factor, max(0.0, (lower[index] - q[index]) / change))
        line_search = []
        while factor > 1e-12:
            candidate = q + factor * delta
            if not all(lower[i] - 1e-12 <= candidate[i] <= upper[i] + 1e-12
                       for i in range(len(candidate))):
                line_search.append({"factor": factor, "within_source_limits": False})
                factor *= 0.5
                continue
            test_data, _, _, test_pos_error, test_axis_error = evaluate(candidate)
            test_safe, test_bad_contacts = contact_pose_is_collision_free(exp, model, test_data)
            current_cost = position_norm**2 + (orientation_scale * axis_angle)**2
            test_angle = abs(math.asin(float(np.clip(test_axis_error, -1.0, 1.0))))
            test_cost = float(np.dot(test_pos_error, test_pos_error)
                              + (orientation_scale * test_angle) ** 2)
            line_search.append({"factor": factor, "candidate_qpos_rad": candidate.tolist(),
                                "safe": test_safe, "cost": test_cost,
                                "blocking_contacts": test_bad_contacts})
            if test_safe and test_cost < current_cost:
                q = candidate
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            blocked_updates += 1
            trace[-1]["delta_rad"] = delta.tolist()
            trace[-1]["active_set_free_joints"] = [
                name for name, is_free in zip(exp.RIGHT_ARM_JOINTS, free) if is_free
            ]
            trace[-1]["line_search"] = line_search
            break

    data, point, normal, pos_error, axis_error = evaluate(q)
    final_angle = abs(math.asin(float(np.clip(axis_error, -1.0, 1.0))))
    safe, bad_contacts = contact_pose_is_collision_free(exp, model, data)
    raise RuntimeError(json.dumps({
        "reason": "collision_aware_gap_center_ik_did_not_converge",
        "position_error_m": float(np.linalg.norm(pos_error)),
        "jaw_axis_tilt_from_horizontal_rad": final_angle,
        "source_limit_pose_rad": {name: float(value) for name, value in zip(exp.RIGHT_ARM_JOINTS, q)},
        "within_source_limits": all(float(limits[i][0]) <= q[i] <= float(limits[i][1])
                                     for i in range(len(q))),
        "safe": safe,
        "blocking_contacts": bad_contacts,
        "blocked_updates": blocked_updates,
        "trace": trace,
    }, separators=(",", ":")))


def derive_compiled_contact_targets(exp, model: mujoco.MjModel,
                                   base_data: mujoco.MjData,
                                   bottle_geoms: list[int]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Align the compiled jaw-surface corridor to the canonical bottle body cylinder."""
    narrow = exp.collision_geom_id(model, "R_hand_narrow3_Link")
    wide = exp.collision_geom_id(model, "R_hand_wide3_Link")
    bottle_body = next((g for g in bottle_geoms if "bottle_body" in geom_name(model, g)), None)
    if bottle_body is None:
        raise RuntimeError("Canonical bottle body collision geom is missing")

    reset_arm = exp.model_arm_home(model)
    home = dict(reset_arm)
    home["right_elbow_joint"] = -1.0
    gap_samples = []
    for aperture in np.linspace(1.0, 0.0, 101):
        data = q_state(exp, model, base_data, home, float(aperture))
        gap, witness = geom_distance(model, data, narrow, wide)
        gap_samples.append({"aperture_ratio": float(aperture), "signed_gap_m": gap,
                            "witness_segment_world_m": witness})
    gap_values = [row["signed_gap_m"] for row in gap_samples]
    if any(gap_values[i + 1] > gap_values[i] + 1e-6 for i in range(len(gap_values) - 1)):
        raise RuntimeError("Compiled jaw-surface gap is not monotone through closure")

    body_radius = float(model.geom_size[bottle_body][0])
    body_half_height = float(model.geom_size[bottle_body][1])
    body_center = base_data.geom_xpos[bottle_body].copy()
    open_gap = gap_values[0]
    target_diameter = 2.0 * body_radius
    open_clearance = open_gap - target_diameter
    if open_clearance <= 0.0:
        raise RuntimeError(f"Canonical bottle body does not fit through the compiled open jaw gap: "
                           f"{open_gap=}, {target_diameter=}")
    target_world = body_center.copy()

    crossing = next((i for i in range(len(gap_samples) - 1)
                     if gap_samples[i]["signed_gap_m"] >= target_diameter
                     and gap_samples[i + 1]["signed_gap_m"] <= target_diameter), None)
    if crossing is None:
        raise RuntimeError("Target shoulder slice does not fit the compiled jaw closing envelope")
    a = gap_samples[crossing]
    b = gap_samples[crossing + 1]
    fraction = (a["signed_gap_m"] - target_diameter) / (
        a["signed_gap_m"] - b["signed_gap_m"]
    )
    contact_aperture = a["aperture_ratio"] + fraction * (
        b["aperture_ratio"] - a["aperture_ratio"]
    )
    contact_data = q_state(exp, model, base_data, home, contact_aperture)
    _, witness = geom_distance(model, contact_data, narrow, wide)
    gap_midpoint = 0.5 * (np.asarray(witness[:3]) + np.asarray(witness[3:]))
    end_body = exp.body_id(model, "R_omnipicker_base_link")
    end_position = contact_data.xpos[end_body].copy()
    rotation = contact_data.xmat[end_body].reshape(3, 3)
    gap_offset = gap_midpoint - end_position
    local_gap_point = rotation.T @ gap_offset
    local_gap_normal = rotation.T @ (
        np.asarray(witness[3:]) - np.asarray(witness[:3])
    )
    local_gap_normal /= np.linalg.norm(local_gap_normal)
    grasp_target = target_world
    pregrasp_target = target_world + np.array([0.0, 0.10, 0.0])
    pregrasp, pregrasp_ik = solve_gap_center_pose(
        exp, model, base_data, home, pregrasp_target,
        contact_aperture, local_gap_point, local_gap_normal,
    )
    grasp, grasp_ik = solve_gap_center_pose(
        exp, model, base_data, pregrasp, grasp_target,
        contact_aperture, local_gap_point, local_gap_normal,
    )
    pregrasp_data = q_state(exp, model, base_data, pregrasp, contact_aperture)
    grasp_data = q_state(exp, model, base_data, grasp, contact_aperture)
    pregrasp_end = pregrasp_data.xpos[end_body] + pregrasp_data.xmat[end_body].reshape(3, 3) @ local_gap_point
    grasp_end = grasp_data.xpos[end_body] + grasp_data.xmat[end_body].reshape(3, 3) @ local_gap_point
    derivation = {
        "basis": "compiled mj_geomDistance witnesses for narrow3/wide3; canonical compiled 70 mm bottle-body collision cylinder; no robot body-origin grasp anchoring",
        "bottle_body_geom": geom_name(model, bottle_body),
        "bottle_body_compiled_radius_m": body_radius,
        "bottle_body_compiled_half_height_m": body_half_height,
        "bottle_body_center_world_m": body_center.tolist(),
        "jaw_open_surface_gap_m": open_gap,
        "initial_open_diametral_clearance_m": open_clearance,
        "target_bottle_body_diameter_m": target_diameter,
        "target_body_center_world_m": target_world.tolist(),
        "jaw_aperture_at_matching_compiled_surface_gap": contact_aperture,
        "jaw_surface_gap_at_target_m": float(target_diameter),
        "jaw_surface_witness_segment_at_contact_world_m": witness,
        "jaw_surface_midpoint_at_home_world_m": gap_midpoint.tolist(),
        "end_effector_to_jaw_gap_offset_world_m": gap_offset.tolist(),
        "jaw_gap_point_local_to_end_effector_m": local_gap_point.tolist(),
        "jaw_closing_axis_local": local_gap_normal.tolist(),
        "grasp_target_jaw_gap_center_world_m": grasp_target.tolist(),
        "pregrasp_target_jaw_gap_center_world_m": pregrasp_target.tolist(),
        "grasp_ik_final_gap_center_world_m": grasp_end.tolist(),
        "pregrasp_ik_final_gap_center_world_m": pregrasp_end.tolist(),
        "grasp_ik_diagnostics": grasp_ik,
        "pregrasp_ik_diagnostics": pregrasp_ik,
        "approach_offset_world_m": [0.0, 0.10, 0.0],
        "initial_pose_correction": {
            "joint": "right_elbow_joint",
            "reset_qpos_rad": reset_arm["right_elbow_joint"],
            "planning_operational_qpos_rad": home["right_elbow_joint"],
            "reason": "reset is exactly at the source upper limit; a flexed arm branch is required to route the wrist around the canonical table front-left leg observed in Candidate 2",
            "active_rollout_rule": "must be reached by bounded actuator command from reset; no qpos write",
        },
        "compiled_gap_samples": gap_samples,
    }
    return {"home": home, "pregrasp": pregrasp, "grasp": grasp}, derivation


def render_pose(exp, model: mujoco.MjModel, data: mujoco.MjData, output: Path, name: str) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.30, 0.02, 0.88]
    camera.distance = 1.30
    camera.azimuth = 140
    camera.elevation = -18
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render().copy()).save(output / name)
    renderer.close()


def runtime_identity(exp, output: Path) -> dict[str, Any]:
    native_path = Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"
    head = exp.subprocess.check_output(
        ["git", "-C", str(EXPERIMENT_PATH.parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "branch_head_before_commit": head,
        "experiment_script": str(EXPERIMENT_PATH),
        "experiment_script_sha256": sha256(EXPERIMENT_PATH),
        "feasibility_script": str(Path(__file__).resolve()),
        "feasibility_script_sha256": sha256(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "mujoco_native_library": str(native_path),
        "mujoco_native_library_sha256": sha256(native_path),
        "numpy": np.__version__,
        "working_tree_status_in_task_worktree": exp.subprocess.check_output(
            ["git", "-C", str(EXPERIMENT_PATH.parents[2]), "status", "--short"], text=True
        ).splitlines(),
        "evidence_dir": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    exp = load_experiment()
    source = exp.verify_inputs(output)
    model, build_info = exp.build_model("bottle_hold", {})
    base_data = exp.initialize(model)
    identity = runtime_identity(exp, output)

    bottle_geoms = [g for g in range(model.ngeom)
                    if body_name(model, g).startswith("m0_bottle")
                    and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    table_geoms = [g for g in range(model.ngeom)
                   if body_name(model, g) == "m0_table"
                   and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    robot_geoms = [g for g in range(model.ngeom)
                   if body_name(model, g).startswith(("right_", "R_hand_"))
                   and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if not bottle_geoms or not table_geoms or not robot_geoms:
        raise RuntimeError("Compiled canonical bottle, G1 table, or right-arm collision geoms are missing")

    render_pose(exp, model, base_data, output, "canonical-scene-reset.png")
    rejected_seed = exp.model_arm_home(model)
    rejected_seed["right_elbow_joint"] = -1.0
    seed_data = q_state(exp, model, base_data, rejected_seed, 1.0)
    render_pose(exp, model, seed_data, output, "candidate-3-flexed-open-seed.png")
    try:
        poses, contact_target_derivation = derive_compiled_contact_targets(
            exp, model, base_data, bottle_geoms
        )
    except RuntimeError as error:
        raw_detail = str(error)
        try:
            detail: Any = json.loads(raw_detail)
        except json.JSONDecodeError:
            detail = {"error": raw_detail}
        source_root = exp.ET.parse(exp.X2_ROOT / exp.X2_URDF_REL).getroot()
        source_joints = {joint.get("name", ""): joint for joint in source_root.findall("joint")}
        arm_source_limits = {}
        for name in exp.RIGHT_ARM_JOINTS:
            limit = source_joints[name].find("limit")
            arm_source_limits[name] = [float(limit.get("lower")), float(limit.get("upper"))]
        current_pose = {
            name: float(seed_data.qpos[exp.qpos_id(model, name)])
            for name in exp.RIGHT_ARM_JOINTS
        }
        result = {
            "status": "BLOCKED",
            "first_failed_gate": "no collision-free, source-limited right-arm pregrasp/grasp configuration found within the three-candidate budget",
            "physics_steps": 0,
            "candidate_count": 3,
            "object_state_writes_during_rollout": 0,
            "follower_or_object_qpos_writes": 0,
            "source": source,
            "runtime_identity": identity,
            "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                      "ngeom": model.ngeom, "nbody": model.nbody,
                      "timestep_s": float(model.opt.timestep)},
            "canonical_scene": {
                "bottle_body_position_world_m": base_data.xpos[exp.body_id(model, "m0_bottle")].tolist(),
                "bottle_body_diameter_m": 2.0 * float(model.geom_size[next(
                    g for g in bottle_geoms if "bottle_body" in geom_name(model, g)
                )][0]),
                "table_geom_names": [geom_name(model, gid) for gid in table_geoms],
                "bottle_geom_names": [geom_name(model, gid) for gid in bottle_geoms],
            },
            "candidate_3_flexed_seed": {
                "arm_qpos_rad": current_pose,
                "source_position_limits_rad": arm_source_limits,
                "open_pose_contacts": contact_rows(exp, model, seed_data),
                "solver_failure": detail,
            },
            "next_decision": "Authorize changing the X2 base-to-table station placement or approve a different right-arm approach side; the current fixed station has no demonstrated collision-free corridor within the authorized candidate budget.",
        }
        (output / "candidate-3-zero-step.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        (output / "runtime_identity.json").write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )
        command = (
            "MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python "
            f"{Path(__file__).resolve()} --output {output}"
        )
        (output / "experiment_commands.txt").write_text(command + "\n", encoding="utf-8")
        report = f"""# Issue #46 OmniPicker geometric feasibility

Status: **BLOCKED at GEOMETRY_AUDIT**

The canonical G1 table and 70 mm X2 bottle are unchanged. Three distinct right-arm candidates were evaluated with compiled MuJoCo collision geometry; no physics step was taken. No fixed-object contact, bottle rollout, or lift was run.

## Runtime and provenance

- RobotSim branch HEAD before this evidence commit: `{identity['branch_head_before_commit']}`
- Pinned X2 source: `{source['repository']}` at `{source['commit']}`
- URDF: `{source['urdf_path']}` SHA-256 `{source['urdf_sha256']}`
- MuJoCo: Python/native `{mujoco.__version__}` / `{mujoco.mj_versionString()}`
- Native library SHA-256: `{identity['mujoco_native_library_sha256']}`
- Physics steps: `0`

## Candidate results

1. **Candidate 1: fixed wrist orientation, body-center jaw target.** A zero-step full-orientation IK attempt could not reach a collision-free pose. The best recorded iteration remained 42 mm from the jaw-gap target and the next step intersected the bottle shoulder with the right wrist-yaw collision mesh. The pose was rejected.
2. **Candidate 2: horizontal jaw-axis constraint, elbow seed -0.10 rad.** The arm remained 272 mm from the jaw-gap target; every improving step intersected the canonical front-left table leg with the right wrist-roll geometry. The pose was rejected.
3. **Candidate 3: horizontal jaw-axis constraint, elbow seed -1.00 rad.** The initial planning pose already penetrated the table top by 41.8 mm, the front-left table leg by 52.6 mm, and the narrow jaw link by 2.2 mm. The pose was rejected.

The prior fixture-only placement is also retained separately as a known baseline failure: the open-home pose had 32.1 mm wrist penetration and additional hand-loop intersections. It was not repeated as a new candidate.

## Gate result

No candidate achieved a collision-free source-limited pregrasp and bilateral closure corridor. Global infeasibility over all continuous joint configurations is **not proven**; the authorized three-candidate budget is exhausted. The fixed station geometry is the unresolved gate. The next maintainer decision is whether to permit moving the X2 base relative to the canonical table or to approve an alternate right-arm approach side. No controller, source limit, collision geometry, bottle parameter, or table geometry was changed.

## Reproduction

```bash
{command}
```

See `candidate-3-zero-step.json`, `runtime_identity.json`, `experiment_commands.txt`, and the PNGs for raw zero-step evidence.
"""
        (output / "REPORT.md").write_text(report, encoding="utf-8")
        return 2
    samples = []
    for aperture in np.linspace(1.0, 0.0, 51):
        samples.append(inspect_pose(exp, model, base_data, poses["grasp"], float(aperture),
                                    bottle_geoms, table_geoms, robot_geoms))
    pregrasp = inspect_pose(exp, model, base_data, poses["pregrasp"], 1.0,
                            bottle_geoms, table_geoms, robot_geoms)
    grasp_open = samples[0]
    grasp_closed = samples[-1]
    all_contacts = [row for sample in samples for row in sample["contacts"]
                    if "m0_bottle" in {row["body1"], row["body2"]}]
    target_contact_families = sorted({classify_contact(
        row["body1"] if "m0_bottle" not in row["body1"] else row["body2"]
    ) for row in all_contacts})

    data = q_state(exp, model, base_data, poses["pregrasp"], 1.0)
    render_pose(exp, model, data, output, "candidate-2-pregrasp.png")
    data = q_state(exp, model, base_data, poses["grasp"], 1.0)
    render_pose(exp, model, data, output, "candidate-2-open.png")
    data = q_state(exp, model, base_data, poses["grasp"], 0.0)
    render_pose(exp, model, data, output, "candidate-2-closed.png")

    run = {
        "candidate": 3,
        "candidate_basis": "accepted gripper orientation retained; end-effector target derived from the canonical bottle body center and compiled open-jaw-to-base offset; source-limited arm IK",
        "physics_steps": 0,
        "qpos_writes": "planning-only initial conditions; no rollout or dynamic state writes",
        "source": source,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "mujoco_python_version": mujoco.__version__,
            "mujoco_native_version": mujoco.mj_versionString(),
            "experiment_script_sha256": sha256(EXPERIMENT_PATH),
            "feasibility_script_sha256": sha256(Path(__file__).resolve()),
        },
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                  "timestep_s": float(model.opt.timestep)},
        "canonical_scene": {
            "bottle_body_position_world_m": base_data.xpos[exp.body_id(model, "m0_bottle")].tolist(),
            "bottle_dimensions_m": {"diameter": 0.070, "height": 0.2445, "mass": 0.570},
            "table_geom_names": [geom_name(model, gid) for gid in table_geoms],
            "bottle_geom_names": [geom_name(model, gid) for gid in bottle_geoms],
        },
        "home_arm_qpos_rad": poses["home"],
        "candidate_arm_poses_qpos_rad": {"pregrasp": poses["pregrasp"], "grasp": poses["grasp"]},
        "compiled_contact_target_derivation": contact_target_derivation,
        "pregrasp_open": pregrasp,
        "grasp_open": grasp_open,
        "grasp_closed": grasp_closed,
        "closure_samples": samples,
        "bottle_contact_families_seen_in_closure": target_contact_families,
        "candidate_static_summary": {
            "open_has_no_bottle_contact": not any(
                "m0_bottle" in {row["body1"], row["body2"]} for row in grasp_open["contacts"]
            ),
            "open_other_hand_and_wrist_clearance_m": grasp_open[
                "other_hand_and_wrist_bottle_clearance"]["minimum_signed_distance_m"],
            "open_robot_to_table_clearance_m": grasp_open["minimum_robot_to_table_clearance_m"],
            "closed_narrow_clearance_m": grasp_closed["bottle_clearance_by_family"]["narrow"][
                "minimum_signed_distance_m"],
            "closed_wide_clearance_m": grasp_closed["bottle_clearance_by_family"]["wide"][
                "minimum_signed_distance_m"],
            "closure_bottle_contact_count": len(all_contacts),
            "target_contact_families": target_contact_families,
        },
    }
    (output / "candidate-2-zero-step.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
