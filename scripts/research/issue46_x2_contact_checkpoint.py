from __future__ import annotations

import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import issue46_x2_grasp as task


OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", str(Path(__file__).parent)))
SHORT_LIFT_M = float(os.environ.get("ISSUE46_CHECKPOINT_LIFT_M", "0.030"))
MIMIC_LIMIT_RAD = task.MIMIC_RELATION_TOLERANCE_RAD
STEPS = {
    "approach": 500,
    "close": 500,
    "grasp_approach": 500,
    "hold": 500,
    "short_lift": 1000,
}


def body_family(body_name: str) -> str | None:
    lowered = body_name.lower()
    for family in ("thumb", "index", "middle", "ring", "pinky"):
        if family in lowered:
            return family
    return None


def body_ancestors(model: mujoco.MjModel, body_id: int) -> list[str]:
    names: list[str] = []
    current = body_id
    while current > 0:
        names.append(task.name(model, mujoco.mjtObj.mjOBJ_BODY, current))
        current = int(model.body_parentid[current])
    return names


def quaternion_delta_rad(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = min(1.0, abs(float(np.dot(a, b))))
    return 2.0 * math.acos(dot)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    started_wall = time.time()
    if not 0.0 < SHORT_LIFT_M <= 0.030000001:
        raise ValueError("ISSUE46_CHECKPOINT_LIFT_M must be in (0, 0.03] for this bounded checkpoint")
    if not math.isclose(task.GRASP_PALM_Z_OFFSET_M, 0.007, abs_tol=1e-12):
        raise ValueError("This single bounded checkpoint uses the previously measured +0.007 m grasp-height correction")
    OUT.mkdir(parents=True, exist_ok=True)
    reproduction = "\n".join(
        (
            "set -o pipefail",
            f"mkdir -p {shlex.quote(str(OUT))}",
            f"cd {shlex.quote(str(task.SIM_REPO_ROOT))}",
            "MUJOCO_GL=egl "
            f"ISSUE46_EVIDENCE_DIR={shlex.quote(str(OUT))} "
            "ISSUE46_FINGER_CLOSE_FRACTION=1.0 ISSUE46_GRASP_PALM_Z_OFFSET_M=0.007 "
            f"ISSUE46_CHECKPOINT_LIFT_M={SHORT_LIFT_M:.3f} "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
            f"2>&1 | tee {shlex.quote(str(OUT / 'run.log'))}",
        )
    )
    (OUT / "reproduction_command.txt").write_text(reproduction + "\n", encoding="utf-8")
    print("EXPERIMENT=Issue46 contact-loaded short-lift checkpoint; transfer and later phases are not implemented")
    print(f"COMMAND={reproduction}")

    vendor_head = subprocess.check_output(
        ["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    vendor_dirty = subprocess.check_output(
        ["git", "-C", str(task.ROOT), "status", "--porcelain"], text=True
    ).strip()
    repo_root = task.SIM_REPO_ROOT
    robot_head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    robot_branch = subprocess.check_output(
        ["git", "-C", str(repo_root), "branch", "--show-current"], text=True
    ).strip()
    robot_dirty = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip()
    if vendor_head != task.SOURCE_PIN or vendor_dirty:
        raise RuntimeError(f"Pinned vendor checkout mismatch: head={vendor_head}, dirty={bool(vendor_dirty)}")

    model, details = task.build_model()
    data = details["data"]
    if not details["mimic_constraints_match"] or model.neq != 12:
        raise RuntimeError(f"Expected exactly 12 compiled source mimic equalities, got neq={model.neq}")

    arm_ids, arm_qpos = task.resolve_joints(model, task.ARM)
    finger_joint_names = list(task.FINGERS)
    finger_ids, finger_qpos = task.resolve_joints(model, finger_joint_names)
    source_joints = ET.parse(task.URDF).getroot().findall("joint")
    right_hand_joint_names = [
        joint.get("name")
        for joint in source_joints
        if joint.get("name", "").startswith("R_") and joint.get("type") == "revolute"
    ]
    all_hand_joint_names = [
        joint.get("name")
        for joint in source_joints
        if joint.get("name", "").startswith(("R_", "L_")) and joint.get("type") == "revolute"
    ]
    right_hand_ids, right_hand_qpos = task.resolve_joints(model, right_hand_joint_names)
    hand_ids, hand_qpos = task.resolve_joints(model, all_hand_joint_names)
    actuator_by_joint: dict[int, int] = {}
    for aid in range(model.nu):
        actuator_by_joint[int(model.actuator_trnid[aid, 0])] = aid
    arm_ctrl = [actuator_by_joint[idx] for idx in arm_ids]
    finger_ctrl = [actuator_by_joint[idx] for idx in finger_ids]
    site_id = int(details["site_id"])
    if site_id < 0:
        raise RuntimeError("right_palm_frame site is missing")

    q0 = data.qpos.copy()
    approach_route = task.ik_plan(
        model,
        q0,
        site_id,
        arm_ids,
        [task.APPROACH_PALM_POS.copy(), task.PREGRASP_PALM_POS.copy(), task.GRASP_PALM_POS.copy()],
        task.PALM_TARGET_ROTATION,
    )
    grasp_arm = task.ik_plan(
        model,
        q0,
        site_id,
        arm_ids,
        [task.GRASP_PALM_POS.copy()],
        task.PALM_TARGET_ROTATION,
    )[0]
    lift_seed = q0.copy()
    lift_seed[arm_qpos] = grasp_arm
    lift_arm = task.ik_plan(
        model,
        lift_seed,
        site_id,
        arm_ids,
        [np.array([task.GRASP_PALM_POS[0], task.GRASP_PALM_POS[1], task.GRASP_PALM_POS[2] + SHORT_LIFT_M])],
        task.PALM_TARGET_ROTATION,
    )[0]

    # Initial robot positioning and follower seeding happen before the rollout clock starts.
    data.qpos[arm_qpos] = approach_route[0]
    open_target = q0[finger_qpos].copy()
    for joint_name, value in {
        "R_thumb_roll_joint": 0.80,
        "R_thumb_abad_joint": -1.70,
        "R_index_abad_joint": -0.18,
        "R_ring_abad_joint": 0.15,
        "R_pinky_abad_joint": 0.15,
    }.items():
        open_target[finger_joint_names.index(joint_name)] = value
    data.qpos[finger_qpos] = open_target
    for relation in details["mimic_relations"]:
        driver_qpos = float(data.qpos[relation["driver_qpos_address"]])
        data.qpos[relation["follower_qpos_address"]] = (
            relation["compiled_follower_reference_rad"]
            + relation["mujoco_polycoef"][0]
            + relation["mujoco_polycoef"][1]
            * (driver_qpos - relation["compiled_driver_reference_rad"])
        )
    mujoco.mj_forward(model, data)

    close_target = open_target.copy()
    for i, joint_name in enumerate(finger_joint_names):
        joint_id = finger_ids[i]
        target = task.FINGERS[joint_name]
        if "thumb_mcp" in joint_name:
            target = 0.82
        elif joint_name.endswith("_pip_joint"):
            target = 1.15
        close_target[i] = np.clip(target, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
    close_target = open_target + task.FINGER_CLOSE_FRACTION * (close_target - open_target)
    close_target[finger_joint_names.index("R_thumb_roll_joint")] = 0.65
    close_target[finger_joint_names.index("R_thumb_abad_joint")] = -0.30
    preshape_target = open_target + task.FINGER_PRESHAPE_FRACTION * (close_target - open_target)

    hold_targets = data.ctrl.copy()
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[joint_id])
        hold_targets[aid] = q0[qadr]
    data.ctrl[:] = hold_targets
    data.ctrl[arm_ctrl] = approach_route[0]
    data.ctrl[finger_ctrl] = open_target

    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    bottle_geom_names = ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap")
    bottle_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in bottle_geom_names
    }
    if -1 in bottle_geoms:
        raise RuntimeError("One or more canonical bottle collision geoms are missing")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    bottle_qadr = int(model.jnt_qposadr[bottle_joint])
    bottle_dadr = int(model.jnt_dofadr[bottle_joint])
    bottle_initial_pos = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
    bottle_initial_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
    bottle_initialization_unchanged = np.array_equal(
        q0[bottle_qadr : bottle_qadr + 7], data.qpos[bottle_qadr : bottle_qadr + 7]
    )
    table_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if task.name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith("m0_table")
    }

    def mimic_state() -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for relation in details["mimic_relations"]:
            follower = float(data.qpos[relation["follower_qpos_address"]])
            driver = float(data.qpos[relation["driver_qpos_address"]])
            expected = relation["compiled_follower_reference_rad"] + relation["mujoco_polycoef"][0]
            expected += relation["mujoco_polycoef"][1] * (
                driver - relation["compiled_driver_reference_rad"]
            )
            result[str(relation["follower_joint"])] = {
                "driver_joint": relation["driver_joint"],
                "driver_qpos_rad": driver,
                "follower_qpos_rad": follower,
                "expected_follower_qpos_rad": float(expected),
                "absolute_relation_error_rad": abs(follower - expected),
                "driver_actuator_target_rad": float(data.ctrl[actuator_by_joint[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(relation["driver_joint"]))
                ]]),
                "mujoco_polycoef": relation["mujoco_polycoef"],
            }
        return result

    def bottle_contact_state() -> tuple[set[str], set[str], list[dict[str, object]], float]:
        right_links: set[str] = set()
        left_links: set[str] = set()
        pairs: list[dict[str, object]] = []
        penetration = 0.0
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not bottle_geoms.intersection((g1, g2)):
                continue
            bottle_geom = g1 if g1 in bottle_geoms else g2
            other_geom = g2 if bottle_geom == g1 else g1
            other_body_id = int(model.geom_bodyid[other_geom])
            other_body = task.name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
            ancestors = body_ancestors(model, other_body_id)
            if any(n.startswith(("R_", "right_")) for n in ancestors):
                right_links.add(other_body)
            if any(n.startswith(("L_", "left_")) for n in ancestors):
                left_links.add(other_body)
            force6 = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, ci, force6)
            penetration = max(penetration, max(0.0, -float(contact.dist)))
            pairs.append(
                {
                    "bottle_geom": task.name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom),
                    "other_geom": task.name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom),
                    "other_body": other_body,
                    "other_body_ancestors": ancestors,
                    "right_finger_family": body_family(other_body) if other_body in right_links else None,
                    "distance_m": float(contact.dist),
                    "position_world_m": contact.pos.tolist(),
                    "contact_force_frame_6d": force6.tolist(),
                    "normal_force_n": float(max(0.0, force6[0])),
                }
            )
        return right_links, left_links, pairs, penetration

    def table_contact_links() -> tuple[set[str], set[str]]:
        right_links: set[str] = set()
        left_links: set[str] = set()
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in table_geoms:
                other_geom = g2
            elif g2 in table_geoms:
                other_geom = g1
            else:
                continue
            body_id = int(model.geom_bodyid[other_geom])
            body = task.name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            ancestors = body_ancestors(model, body_id)
            if any(n.startswith(("R_", "right_")) for n in ancestors):
                right_links.add(body)
            if any(n.startswith(("L_", "left_")) for n in ancestors):
                left_links.add(body)
        return right_links, left_links

    def hand_limit_state() -> tuple[list[dict[str, object]], bool, list[str]]:
        violations = []
        hand_q = data.qpos[hand_qpos]
        nan_names: list[str] = []
        for idx, joint_id in enumerate(hand_ids):
            joint_name = task.name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            value = float(hand_q[idx])
            lower, upper = model.jnt_range[joint_id]
            if not math.isfinite(value):
                nan_names.append(joint_name)
            if model.jnt_limited[joint_id] and (value < lower - 1e-8 or value > upper + 1e-8):
                violations.append(
                    {"joint": joint_name, "qpos_rad": value, "range_rad": [float(lower), float(upper)]}
                )
        return violations, bool(nan_names), nan_names

    def render_to(filename: str, render_data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
        renderer.update_scene(render_data, camera=camera, scene_option=render_option)
        pixels = renderer.render()
        if filename:
            import imageio.v2 as imageio

            imageio.imwrite(OUT / filename, pixels)
        return pixels

    import imageio.v2 as imageio

    overview_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(overview_camera)
    overview_camera.lookat[:] = [0.30, 0.10, 1.00]
    overview_camera.distance = 2.65
    overview_camera.azimuth = 135
    overview_camera.elevation = -25
    close_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(close_camera)
    close_camera.lookat[:] = [task.START[0], task.START[1] + 0.10, task.START[2] + 0.08]
    close_camera.distance = 0.82
    close_camera.azimuth = 135
    close_camera.elevation = -12
    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.geomgroup[0] = 0
    render_option.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=720, width=1280)
    video_path = OUT / "contact_loaded_short_lift.mp4"
    video_writer = imageio.get_writer(video_path, fps=25, codec="libx264", quality=8)

    step_count = 0
    qpos_assignments_during_rollout = 0
    first_contact: dict[str, object] | None = None
    first_mimic_exceedance: dict[str, object] | None = None
    first_contact_window_mimic_exceedance: dict[str, object] | None = None
    max_mimic_by_phase = {phase: 0.0 for phase in STEPS}
    max_mimic_error = 0.0
    first_joint_limit_violation: dict[str, object] | None = None
    joint_limit_violation_count = 0
    nan_step_count = 0
    sign_inversion_count = 0
    max_penetration_m = 0.0
    max_bottle_step_m = 0.0
    max_bottle_rotation_step_rad = 0.0
    phase_summaries: list[dict[str, object]] = []
    contact_union_by_phase: dict[str, set[str]] = {phase: set() for phase in STEPS}
    family_union_by_phase: dict[str, set[str]] = {phase: set() for phase in STEPS}
    contact_pair_union_by_phase: dict[str, set[tuple[str, str, str]]] = {phase: set() for phase in STEPS}
    contact_steps_by_phase = {phase: 0 for phase in STEPS}
    opposing_hand_contacts: set[str] = set()
    pregrasp_position: np.ndarray | None = None
    pregrasp_quat: np.ndarray | None = None
    hold_start: dict[str, object] | None = None
    hold_end: dict[str, object] | None = None
    hold_contact_steps = 0
    hold_normal_force_max_n = 0.0
    hold_force_samples: list[float] = []
    short_lift_max_state: tuple[np.ndarray, np.ndarray, float] | None = None
    max_bottle_height_m = float(bottle_initial_pos[2])
    max_short_lift_height_m = float(bottle_initial_pos[2])
    right_table_contact_before_close: set[str] = set()

    (OUT / "checkpoint_contact_trace.jsonl").write_text("", encoding="utf-8")
    (OUT / "hand_mimic_trace.jsonl").write_text("", encoding="utf-8")
    (OUT / "bottle_pose_trace.jsonl").write_text("", encoding="utf-8")
    contact_file = (OUT / "checkpoint_contact_trace.jsonl").open("a", encoding="utf-8")
    hand_file = (OUT / "hand_mimic_trace.jsonl").open("a", encoding="utf-8")
    bottle_file = (OUT / "bottle_pose_trace.jsonl").open("a", encoding="utf-8")

    def snapshot(name: str, render_data: mujoco.MjData | None = None, close: bool = False) -> np.ndarray:
        target = data if render_data is None else render_data
        camera = close_camera if close else overview_camera
        return render_to(name, target, camera)

    snapshot("pregrasp.png")
    snapshot("pregrasp_closeup.png", close=True)
    video_writer.append_data(snapshot("", close=False))

    def run_phase(label: str, steps: int, arm_target: np.ndarray, finger_target: np.ndarray) -> None:
        nonlocal step_count, first_contact, first_mimic_exceedance
        nonlocal first_contact_window_mimic_exceedance, first_joint_limit_violation
        nonlocal joint_limit_violation_count, nan_step_count, sign_inversion_count
        nonlocal max_mimic_error, max_penetration_m, max_bottle_step_m
        nonlocal max_bottle_rotation_step_rad, hold_start, hold_end, hold_contact_steps
        nonlocal hold_normal_force_max_n, short_lift_max_state, max_bottle_height_m, max_short_lift_height_m
        phase_start = float(data.time)
        arm_start = data.ctrl[arm_ctrl].copy()
        fingers_start = data.ctrl[finger_ctrl].copy()
        phase_mimic_max = 0.0
        phase_right_links: set[str] = set()
        phase_families: set[str] = set()
        phase_contact_steps = 0
        phase_force_samples: list[float] = []
        hold_start_pos = data.qpos[bottle_qadr : bottle_qadr + 3].copy() if label == "hold" else None
        hold_start_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy() if label == "hold" else None
        hold_arm_start = data.qpos[arm_qpos].copy() if label == "hold" else None
        hold_wrist_start = data.site_xpos[site_id].copy() if label == "hold" else None

        for k in range(steps):
            alpha = (k + 1) / steps
            data.ctrl[:] = hold_targets
            data.ctrl[arm_ctrl] = arm_start * (1.0 - alpha) + arm_target * alpha
            data.ctrl[finger_ctrl] = fingers_start * (1.0 - alpha) + finger_target * alpha
            old_pos = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
            old_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
            mujoco.mj_step(model, data)
            step_count += 1

            state = mimic_state()
            errors = {key: float(item["absolute_relation_error_rad"]) for key, item in state.items()}
            current_max_mimic = max(errors.values(), default=0.0)
            phase_mimic_max = max(phase_mimic_max, current_max_mimic)
            max_mimic_error = max(max_mimic_error, current_max_mimic)
            max_mimic_by_phase[label] = max(max_mimic_by_phase[label], current_max_mimic)
            if current_max_mimic > MIMIC_LIMIT_RAD and first_mimic_exceedance is None:
                relation = max(state, key=lambda key: errors[key])
                first_mimic_exceedance = {
                    "step": step_count,
                    "simulation_time_s": float(data.time),
                    "phase": label,
                    "relation": relation,
                    "error_rad": errors[relation],
                }
            if label in ("close", "grasp_approach", "hold", "short_lift") and current_max_mimic > MIMIC_LIMIT_RAD:
                if first_contact_window_mimic_exceedance is None:
                    relation = max(state, key=lambda key: errors[key])
                    first_contact_window_mimic_exceedance = {
                        "step": step_count,
                        "simulation_time_s": float(data.time),
                        "phase": label,
                        "relation": relation,
                        "error_rad": errors[relation],
                    }

            hand_values = data.qpos[hand_qpos].copy()
            hand_nan = not np.isfinite(hand_values).all()
            model_nan = not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            if hand_nan or model_nan:
                nan_step_count += 1
            violations, _, nan_joint_names = hand_limit_state()
            if violations:
                joint_limit_violation_count += len(violations)
                if first_joint_limit_violation is None:
                    first_joint_limit_violation = {
                        "step": step_count,
                        "simulation_time_s": float(data.time),
                        "phase": label,
                        "violations": violations,
                    }
            for relation in details["mimic_relations"]:
                follower_q = float(data.qpos[relation["follower_qpos_address"]])
                driver_q = float(data.qpos[relation["driver_qpos_address"]])
                follower_q0 = float(model.qpos0[relation["follower_qpos_address"]])
                expected = state[str(relation["follower_joint"])]["expected_follower_qpos_rad"]
                expected_delta = float(expected) - follower_q0
                actual_delta = follower_q - follower_q0
                if abs(expected_delta) > 1e-7 and actual_delta * expected_delta < -1e-10:
                    sign_inversion_count += 1

            right_links, left_links, pairs, penetration = bottle_contact_state()
            right_table_links, left_table_links = table_contact_links()
            if label == "approach":
                right_table_contact_before_close.update(right_table_links)
            families = {family for family in (body_family(link) for link in right_links) if family}
            phase_right_links.update(right_links)
            phase_families.update(families)
            contact_union_by_phase[label].update(right_links)
            family_union_by_phase[label].update(families)
            for pair in pairs:
                contact_pair_union_by_phase[label].add(
                    (str(pair["bottle_geom"]), str(pair["other_geom"]), str(pair["other_body"]))
                )
                phase_force_samples.append(float(pair["normal_force_n"]))
                max_penetration_m = max(max_penetration_m, float(max(0.0, -float(pair["distance_m"]))))
            max_penetration_m = max(max_penetration_m, penetration)
            if right_links:
                phase_contact_steps += 1
                contact_steps_by_phase[label] += 1
            if left_links:
                opposing_hand_contacts.update(left_links)
            if label == "hold" and right_links:
                hold_contact_steps += 1
            if label == "hold" and pairs:
                hold_force_samples.extend(float(pair["normal_force_n"]) for pair in pairs)
                hold_normal_force_max_n = max(hold_normal_force_max_n, *[float(p["normal_force_n"]) for p in pairs])
            if right_links and first_contact is None:
                first_contact = {
                    "step": step_count,
                    "simulation_time_s": float(data.time),
                    "phase": label,
                    "right_hand_links": sorted(right_links),
                    "finger_families": sorted(families),
                    "contact_pairs": pairs,
                }
                snapshot("first_contact.png")
                snapshot("first_contact_closeup.png", close=True)

            position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
            quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
            linear_velocity = data.qvel[bottle_dadr : bottle_dadr + 3].copy()
            angular_velocity = data.qvel[bottle_dadr + 3 : bottle_dadr + 6].copy()
            step_jump = float(np.linalg.norm(position - old_pos))
            max_bottle_step_m = max(max_bottle_step_m, step_jump)
            max_bottle_rotation_step_rad = max(
                max_bottle_rotation_step_rad, quaternion_delta_rad(old_quat, quat)
            )
            max_bottle_height_m = max(max_bottle_height_m, float(position[2]))
            if label == "short_lift" and float(position[2]) >= max_short_lift_height_m:
                max_short_lift_height_m = float(position[2])
                short_lift_max_state = (data.qpos.copy(), data.qvel.copy(), float(data.time))

            contact_record = {
                "step": step_count,
                "simulation_time_s": float(data.time),
                "phase": label,
                "right_hand_links": sorted(right_links),
                "right_hand_finger_families": sorted(families),
                "opposite_hand_links": sorted(left_links),
                "opposite_hand_contact": bool(left_links),
                "right_hand_table_contact_links": sorted(right_table_links),
                "opposite_hand_table_contact_links": sorted(left_table_links),
                "contact_pairs": pairs,
                "max_penetration_m": penetration,
                "contact_normal_force_sum_n": sum(float(p["normal_force_n"]) for p in pairs),
                "bottle_qpos_writes_this_step": 0,
            }
            contact_file.write(json.dumps(contact_record, separators=(",", ":")) + "\n")

            hand_record = {
                "step": step_count,
                "simulation_time_s": float(data.time),
                "phase": label,
                "right_hand_joint_qpos_rad": {
                    joint_name: float(data.qpos[qadr])
                    for joint_name, qadr in zip(right_hand_joint_names, right_hand_qpos)
                },
                "right_hand_actuator_target_rad": {
                    joint_name: float(data.ctrl[actuator_by_joint[joint_id]])
                    for joint_name, joint_id in zip(finger_joint_names, finger_ids)
                },
                "all_12_mimic_relations": state,
                "max_mimic_relation_error_rad": current_max_mimic,
                "mimic_tolerance_rad": MIMIC_LIMIT_RAD,
                "joint_limit_violations": violations,
                "nan_joint_names": nan_joint_names,
                "model_nan": model_nan,
            }
            hand_file.write(json.dumps(hand_record, separators=(",", ":")) + "\n")

            bottle_record = {
                "step": step_count,
                "simulation_time_s": float(data.time),
                "phase": label,
                "bottle_position_m": position.tolist(),
                "bottle_height_rise_m": float(position[2] - bottle_initial_pos[2]),
                "bottle_quaternion_wxyz": quat.tolist(),
                "bottle_linear_velocity_m_s": linear_velocity.tolist(),
                "bottle_angular_velocity_rad_s": angular_velocity.tolist(),
                "bottle_step_translation_m": step_jump,
                "right_hand_links": sorted(right_links),
                "right_hand_finger_families": sorted(families),
                "bottle_qpos_writes_this_step": 0,
            }
            bottle_file.write(json.dumps(bottle_record, separators=(",", ":")) + "\n")

            if step_count % 20 == 0:
                video_writer.append_data(snapshot(""))
            if label == "short_lift" and step_count % 100 == 0:
                print(
                    f"CHECKPOINT_PROGRESS step={step_count} t={data.time:.3f} "
                    f"rise={position[2] - bottle_initial_pos[2]:.5f} "
                    f"mimic_max={current_max_mimic:.6f} contacts={sorted(families)}"
                )

        phase_end = float(data.time)
        phase_summaries.append(
            {
                "phase": label,
                "start_time_s": phase_start,
                "end_time_s": phase_end,
                "duration_s": phase_end - phase_start,
                "steps": steps,
                "max_mimic_error_rad": phase_mimic_max,
                "right_hand_links": sorted(phase_right_links),
                "finger_families": sorted(phase_families),
                "right_contact_steps": phase_contact_steps,
                "right_contact_persistence": phase_contact_steps / steps,
                "contact_pairs": [list(pair) for pair in sorted(contact_pair_union_by_phase[label])],
                "max_normal_force_n": max(phase_force_samples, default=0.0),
            }
        )
        if label == "hold":
            assert hold_start_pos is not None and hold_start_quat is not None
            assert hold_arm_start is not None and hold_wrist_start is not None
            hold_start = {
                "time_s": phase_start,
                "bottle_position_m": hold_start_pos.tolist(),
                "bottle_quaternion_wxyz": hold_start_quat.tolist(),
            }
            hold_end = {
                "time_s": phase_end,
                "bottle_position_m": data.qpos[bottle_qadr : bottle_qadr + 3].tolist(),
                "bottle_quaternion_wxyz": data.qpos[bottle_qadr + 3 : bottle_qadr + 7].tolist(),
                "arm_joint_change_l2_rad": float(np.linalg.norm(data.qpos[arm_qpos] - hold_arm_start)),
                "wrist_position_change_m": float(np.linalg.norm(data.site_xpos[site_id] - hold_wrist_start)),
            }
            snapshot("hold.png")
            snapshot("hold_closeup.png", close=True)
        if label == "grasp_approach":
            snapshot("closed_grasp.png")
            snapshot("closed_grasp_closeup.png", close=True)
        if label == "short_lift":
            snapshot("maximum_lift.png")
            snapshot("maximum_lift_closeup.png", close=True)

    approach_bottle_start = bottle_initial_pos.copy()
    run_phase("approach", STEPS["approach"], approach_route[1], preshape_target)
    pregrasp_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
    pregrasp_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
    snapshot("pregrasp.png")
    snapshot("pregrasp_closeup.png", close=True)
    video_writer.append_data(snapshot(""))
    run_phase("close", STEPS["close"], approach_route[1], close_target)
    run_phase("grasp_approach", STEPS["grasp_approach"], grasp_arm, close_target)
    run_phase("hold", STEPS["hold"], grasp_arm, close_target)
    closed_contact_right, closed_contact_left, _, _ = bottle_contact_state()
    run_phase("short_lift", STEPS["short_lift"], lift_arm, close_target)
    if short_lift_max_state is not None:
        max_data = mujoco.MjData(model)
        max_data.qpos[:] = short_lift_max_state[0]
        max_data.qvel[:] = short_lift_max_state[1]
        max_data.ctrl[:] = data.ctrl
        mujoco.mj_forward(model, max_data)
        snapshot("maximum_lift.png", max_data)
        snapshot("maximum_lift_closeup.png", max_data, close=True)
    video_writer.append_data(snapshot(""))

    contact_file.close()
    hand_file.close()
    bottle_file.close()
    video_writer.close()

    final_position = data.qpos[bottle_qadr : bottle_qadr + 3].copy()
    final_quat = data.qpos[bottle_qadr + 3 : bottle_qadr + 7].copy()
    final_linear_velocity = data.qvel[bottle_dadr : bottle_dadr + 3].copy()
    final_angular_velocity = data.qvel[bottle_dadr + 3 : bottle_dadr + 6].copy()
    all_checkpoint_links = set().union(
        *(contact_union_by_phase[p] for p in ("close", "grasp_approach", "hold", "short_lift"))
    )
    all_checkpoint_families = set().union(
        *(family_union_by_phase[p] for p in ("close", "grasp_approach", "hold", "short_lift"))
    )
    grasp_families = family_union_by_phase["hold"]
    lift_families = family_union_by_phase["short_lift"]
    has_thumb = "thumb" in grasp_families
    non_thumb_families = sorted(grasp_families - {"thumb"})
    has_opposing_non_thumb = bool(non_thumb_families)
    lift_has_thumb = "thumb" in lift_families
    lift_non_thumb_families = sorted(lift_families - {"thumb"})
    lift_rise_m = max_short_lift_height_m - float(bottle_initial_pos[2])
    hold_slip_m = None
    hold_orientation_change_rad = None
    if hold_start is not None and hold_end is not None:
        hold_slip_m = float(
            np.linalg.norm(
                np.asarray(hold_end["bottle_position_m"], dtype=float)
                - np.asarray(hold_start["bottle_position_m"], dtype=float)
            )
        )
        hold_orientation_change_rad = quaternion_delta_rad(
            np.asarray(hold_start["bottle_quaternion_wxyz"], dtype=float),
            np.asarray(hold_end["bottle_quaternion_wxyz"], dtype=float),
        )
    hold_contact_persistence = hold_contact_steps / STEPS["hold"]
    approach_translation_m = float(np.linalg.norm(pregrasp_position - approach_bottle_start))
    bottle_attachment_equalities = []
    allowed_mimic_followers = {str(item["follower_joint"]) for item in details["mimic_relations"]}
    for equality_id in range(model.neq):
        if int(model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT):
            object_name = task.name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[equality_id]))
        else:
            object_name = f"eq_type_{int(model.eq_type[equality_id])}"
        if object_name not in allowed_mimic_followers:
            bottle_attachment_equalities.append(
                {"id": equality_id, "type": int(model.eq_type[equality_id]), "obj1": object_name}
            )
    mimic_ok = all(
        max_mimic_by_phase[phase] <= MIMIC_LIMIT_RAD
        for phase in ("close", "grasp_approach", "hold", "short_lift")
    )
    max_contact_loaded_mimic_error = max(
        max_mimic_by_phase[phase] for phase in ("close", "grasp_approach", "hold", "short_lift")
    )
    physical_contact_ok = (
        has_thumb
        and has_opposing_non_thumb
        and lift_has_thumb
        and bool(lift_non_thumb_families)
        and not closed_contact_left
        and not opposing_hand_contacts
    )
    bottle_carry_ok = lift_rise_m >= 0.030 and physical_contact_ok
    no_nan = nan_step_count == 0
    no_limit_violation = joint_limit_violation_count == 0
    no_attachment = not bottle_attachment_equalities and model.nmocap == 0
    no_bottle_qpos_write = qpos_assignments_during_rollout == 0
    approach_no_push = not any(contact_union_by_phase["approach"]) and approach_translation_m <= 0.002
    gates = {
        "no_open_hand_approach_push": approach_no_push,
        "no_right_hand_table_contact_before_close": not right_table_contact_before_close,
        "thumb_contact": has_thumb,
        "opposing_non_thumb_contact": has_opposing_non_thumb,
        "multi_digit_contact_persists_through_short_lift": physical_contact_ok,
        "contact_loaded_mimic_error_le_0_003_rad": mimic_ok,
        "no_joint_limit_violations": no_limit_violation,
        "no_nan_or_sign_inversion": no_nan and sign_inversion_count == 0,
        "no_opposite_hand_assistance": not opposing_hand_contacts,
        "no_bottle_weld_equality_mocap_or_follow_attachment": no_attachment,
        "no_bottle_qpos_writes_during_rollout": no_bottle_qpos_write,
        "short_physical_lift_ge_30_mm_with_multi_digit_contact": bottle_carry_ok,
    }
    ordered_checks = [
        ("contact-loaded mimic tolerance exceeded", mimic_ok),
        ("joint limit violation", no_limit_violation),
        ("NaN or mimic sign inversion", no_nan and sign_inversion_count == 0),
        ("opposite-hand contact", not opposing_hand_contacts),
        ("bottle weld/equality/mocap attachment present", no_attachment),
        ("bottle qpos write during rollout", no_bottle_qpos_write),
        ("open-hand approach pushed/contacted bottle", approach_no_push),
        ("right hand contacted the table before close", not right_table_contact_before_close),
        ("thumb contact missing", has_thumb),
        ("opposing non-thumb finger-family contact missing", has_opposing_non_thumb),
        ("multi-digit contact did not persist through short lift", physical_contact_ok),
        ("contact-loaded 30 mm short lift failed", bottle_carry_ok),
    ]
    first_failed_condition = next((message for message, passed in ordered_checks if not passed), None)

    hold_summary = {
        "duration_s": STEPS["hold"] * task.DT,
        "start": hold_start,
        "end": hold_end,
        "bottle_slip_distance_m": hold_slip_m,
        "bottle_orientation_change_rad": hold_orientation_change_rad,
        "right_hand_contact_persistence": hold_contact_persistence,
        "right_hand_contact_steps": hold_contact_steps,
        "right_hand_links": sorted(contact_union_by_phase["hold"]),
        "finger_families": sorted(family_union_by_phase["hold"]),
        "normal_force_mean_n": float(np.mean(hold_force_samples)) if hold_force_samples else 0.0,
        "normal_force_max_n": hold_normal_force_max_n,
        "normal_force_sample_count": len(hold_force_samples),
        "arm_joint_change_l2_rad": None if hold_end is None else hold_end["arm_joint_change_l2_rad"],
        "wrist_position_change_m": None if hold_end is None else hold_end["wrist_position_change_m"],
    }
    result = {
        "status": "READY FOR MAINTAINER CONTACT-LOADED GRASP REVIEW" if all(gates.values()) else "CHECKPOINT FAIL",
        "first_failed_condition": first_failed_condition,
        "reproduction_command": reproduction,
        "identity": {
            "robot_sim_root": str(repo_root),
            "robot_sim_branch": robot_branch,
            "robot_sim_head_at_run": robot_head,
            "robot_sim_worktree_dirty_at_run": bool(robot_dirty),
            "robot_sim_dirty_paths_at_run": robot_dirty.splitlines(),
            "vendor_source_url": "https://github.com/AgibotTech/agibot_x2_urdf",
            "vendor_source_pin": task.SOURCE_PIN,
            "vendor_source_head": vendor_head,
            "vendor_source_dirty": bool(vendor_dirty),
            "vendor_model": str(task.URDF),
            "vendor_urdf_sha256": task.sha256(task.URDF),
            "vendor_mesh_sha256": task.referenced_meshes(),
            "mujoco": mujoco.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "timestep_s": float(model.opt.timestep),
            "integrator": int(model.opt.integrator),
            "mimic_driver_kp": task.RIGHT_MIMIC_DRIVER_SERVO_KP,
            "mimic_driver_kv": task.RIGHT_MIMIC_DRIVER_SERVO_KV,
            "mimic_tolerance_rad": MIMIC_LIMIT_RAD,
            "grasp_palm_z_offset_m": task.GRASP_PALM_Z_OFFSET_M,
            "short_lift_target_m": SHORT_LIFT_M,
            "canonical_bottle_mass_kg": task.CANONICAL_X2_BOTTLE_MASS_KG,
            "canonical_bottle_diameter_m": task.CANONICAL_X2_BOTTLE_DIAMETER_M,
            "canonical_bottle_height_m": task.CANONICAL_X2_BOTTLE_HEIGHT_M,
        },
        "rollout": {
            "phases": phase_summaries,
            "step_count": step_count,
            "simulation_duration_s": float(data.time),
            "wall_duration_s": time.time() - started_wall,
            "exact_phases_run": list(STEPS),
            "later_phases_not_implemented_or_run": ["transfer", "lower", "release", "settle"],
        },
        "physical_integrity": {
            "bottle_is_free_physics_body": True,
            "bottle_qpos_writes_during_active_rollout": qpos_assignments_during_rollout,
            "bottle_attachment_equality_records": bottle_attachment_equalities,
            "model_equality_count": model.neq,
            "all_equalities_are_12_source_mimic_joint_relations": model.neq == 12 and not bottle_attachment_equalities,
            "mocap_body_count": model.nmocap,
            "mocap_attachment": False,
            "follow_hand_logic": False,
            "opposite_hand_contacts": sorted(opposing_hand_contacts),
            "opposite_hand_assistance": bool(opposing_hand_contacts),
            "bottle_qpos_unchanged_during_robot_initialization": bottle_initialization_unchanged,
            "max_bottle_step_translation_m": max_bottle_step_m,
            "max_bottle_step_rotation_rad": max_bottle_rotation_step_rad,
            "max_penetration_m": max_penetration_m,
        },
        "pregrasp_approach": {
            "bottle_initial_position_m": bottle_initial_pos.tolist(),
            "bottle_pregrasp_position_m": pregrasp_position.tolist(),
            "bottle_pregrasp_translation_m": approach_translation_m,
            "bottle_pregrasp_orientation_change_rad": quaternion_delta_rad(bottle_initial_quat, pregrasp_quat),
            "right_hand_links_during_open_hand_approach": sorted(contact_union_by_phase["approach"]),
            "right_hand_table_contact_before_close": sorted(right_table_contact_before_close),
            "no_open_hand_approach_push_gate": approach_no_push,
        },
        "first_contact": first_contact,
        "contact_topology": {
            "exact_right_hand_links_close_through_short_lift": sorted(all_checkpoint_links),
            "finger_families_close_through_short_lift": sorted(all_checkpoint_families),
            "thumb_contact": has_thumb,
            "opposing_non_thumb_finger_families": non_thumb_families,
            "opposing_non_thumb_contact": has_opposing_non_thumb,
            "preferably_multiple_non_thumb_families": len(non_thumb_families) >= 2,
            "by_phase": {
                phase: {
                    "links": sorted(contact_union_by_phase[phase]),
                    "finger_families": sorted(family_union_by_phase[phase]),
                    "contact_persistence": contact_steps_by_phase[phase] / STEPS[phase],
                }
                for phase in STEPS
            },
            "canonical_contact_pair_union": {
                phase: [list(pair) for pair in sorted(contact_pair_union_by_phase[phase])]
                for phase in STEPS
            },
        },
        "hold_stability": hold_summary,
        "mimic_validation": {
            "all_12_relations": True,
            "max_error_all_phases_rad": max_mimic_error,
            "max_error_during_close_grasp_approach_hold_short_lift_rad": max_contact_loaded_mimic_error,
            "max_error_by_phase_rad": max_mimic_by_phase,
            "first_relation_exceeding_0_003_rad_any_phase": first_mimic_exceedance,
            "first_relation_exceeding_0_003_rad_contact_loaded_phases": first_contact_window_mimic_exceedance,
            "tolerance_rad": MIMIC_LIMIT_RAD,
            "joint_limit_violation_count": joint_limit_violation_count,
            "first_joint_limit_violation": first_joint_limit_violation,
            "nan_step_count": nan_step_count,
            "mimic_sign_inversion_count": sign_inversion_count,
        },
        "short_lift": {
            "initial_bottle_height_m": float(bottle_initial_pos[2]),
            "maximum_bottle_height_m": max_short_lift_height_m,
            "maximum_bottle_rise_m": lift_rise_m,
            "target_rise_m": SHORT_LIFT_M,
            "contact_persistence": contact_steps_by_phase["short_lift"] / STEPS["short_lift"],
            "contact_links": sorted(contact_union_by_phase["short_lift"]),
            "contact_finger_families": sorted(family_union_by_phase["short_lift"]),
            "final_bottle_position_m": final_position.tolist(),
            "final_bottle_quaternion_wxyz": final_quat.tolist(),
            "final_linear_velocity_m_s": final_linear_velocity.tolist(),
            "final_angular_velocity_rad_s": final_angular_velocity.tolist(),
        },
        "gates": gates,
    }
    write_json(OUT / "checkpoint_result.json", result)

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "vendor_pin": task.SOURCE_PIN,
        "vendor_head": vendor_head,
        "vendor_dirty": bool(vendor_dirty),
        "urdf_sha256": task.sha256(task.URDF),
        "mesh_sha256": task.referenced_meshes(),
        "robot_sim_head_at_run": robot_head,
        "robot_sim_branch": robot_branch,
        "robot_sim_dirty_at_run": bool(robot_dirty),
        "model_timestep_s": float(model.opt.timestep),
        "model_integrator": int(model.opt.integrator),
        "mimic_driver_kp": task.RIGHT_MIMIC_DRIVER_SERVO_KP,
        "mimic_driver_kv": task.RIGHT_MIMIC_DRIVER_SERVO_KV,
        "reproduction_command": reproduction,
    }
    write_json(OUT / "runtime_identity.json", runtime)
    print(f"CONTACT_TOPOLOGY links={sorted(all_checkpoint_links)} families={sorted(all_checkpoint_families)}")
    print(
        f"HOLD slip_m={hold_slip_m} orientation_change_rad={hold_orientation_change_rad} "
        f"contact_persistence={hold_contact_persistence:.6f} max_normal_force_n={hold_normal_force_max_n:.6f}"
    )
    print(
        f"MIMIC max_close={max_mimic_by_phase['close']:.9f} max_hold={max_mimic_by_phase['hold']:.9f} "
        f"max_short_lift={max_mimic_by_phase['short_lift']:.9f} first={first_mimic_exceedance}"
    )
    print(f"SHORT_LIFT rise_m={lift_rise_m:.9f} target_m={SHORT_LIFT_M:.6f}")
    print(f"RESULT={result['status']} first_failed_condition={first_failed_condition}")
    renderer.close()
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
