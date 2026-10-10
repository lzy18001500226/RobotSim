from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_x2_grasp as task  # noqa: E402

MIMIC_LIMIT_RAD = 0.003
PENETRATION_LIMIT_M = 0.020
DT = 0.002
FINGER_FAMILIES = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded full-body single-hand contact/lift probe for Issue #46.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kinematic-preview-only",
        action="store_true",
        help="render a separate-data hand/bottle pose without stepping physics",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()

    model, details = task.build_model()
    data = details["data"]
    if model.opt.timestep != DT or not details["mimic_constraints_match"]:
        raise RuntimeError("model timestep or compiled source mimic set is invalid")
    if model.neq != len(details["mimic_relations"]):
        raise RuntimeError("unexpected non-mimic equality constraints")

    branch = subprocess.check_output(["git", "-C", str(task.SIM_REPO_ROOT), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(task.SIM_REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(task.SIM_REPO_ROOT), "status", "--porcelain"], text=True).splitlines()
    vendor_head = subprocess.check_output(["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True).strip()
    vendor_dirty = subprocess.check_output(["git", "-C", str(task.ROOT), "status", "--porcelain"], text=True).strip()
    if vendor_head != task.SOURCE_PIN or vendor_dirty:
        raise RuntimeError(f"pinned upstream checkout mismatch or dirty: {vendor_head}; dirty={bool(vendor_dirty)}")

    source_root = ET.parse(task.URDF).getroot()
    source_joints = {node.get("name"): node for node in source_root.findall("joint") if node.get("name")}

    def axis_sign(name: str) -> int:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        source_axis = np.fromstring(source_joints[name].find("axis").get("xyz", "1 0 0"), sep=" ")
        return 1 if float(np.dot(source_axis, model.jnt_axis[jid])) >= 0.0 else -1

    def source_q(name: str, state: mujoco.MjData = data) -> float:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        reference = float(details["joint_refs"].get(name, 0.0))
        return reference + axis_sign(name) * (float(state.qpos[qadr]) - float(model.qpos0[qadr]))

    def set_source_q(name: str, value: float, state: mujoco.MjData = data) -> None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        reference = float(details["joint_refs"].get(name, 0.0))
        state.qpos[qadr] = float(model.qpos0[qadr]) + axis_sign(name) * (value - reference)

    initial_poses = (
        task.LEFT_ARM_NEUTRAL_SOURCE_POSE,
        task.LEFT_HAND_NEUTRAL_SOURCE_POSE,
        task.RIGHT_HAND_COLLISION_CORRECTION_SOURCE_POSE,
    )
    for pose in initial_poses:
        for name, value in pose.items():
            set_source_q(name, value)
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        set_source_q(follower, float(relation["multiplier"]) * source_q(driver) + float(relation["offset"]))
    mujoco.mj_forward(model, data)

    arm_ids, arm_qpos = task.resolve_joints(model, task.ARM)
    finger_names = list(task.FINGERS)
    finger_ids, finger_qpos = task.resolve_joints(model, finger_names)
    actuator_by_joint = {int(model.actuator_trnid[aid, 0]): aid for aid in range(model.nu)}
    arm_ctrl = [actuator_by_joint[jid] for jid in arm_ids]
    finger_ctrl = [actuator_by_joint[jid] for jid in finger_ids]
    site_id = int(details["site_id"])
    q0 = data.qpos.copy()
    bottle_qid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    bottle_qadr = int(model.jnt_qposadr[bottle_qid])
    bottle_dadr = int(model.jnt_dofadr[bottle_qid])
    bottle_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap")
    }
    bottle_initial = data.qpos[bottle_qadr:bottle_qadr + 7].copy()

    route = task.ik_plan(
        model,
        q0,
        site_id,
        arm_ids,
        [task.APPROACH_PALM_POS.copy(), task.PREGRASP_PALM_POS.copy(), task.GRASP_PALM_POS.copy()],
        task.PALM_TARGET_ROTATION.copy(),
    )
    route[-1] = task.ik_plan(
        model, q0, site_id, arm_ids, [task.GRASP_PALM_POS.copy()], task.PALM_TARGET_ROTATION.copy()
    )[0]
    data.qpos[arm_qpos] = route[0]
    open_target = q0[finger_qpos].copy()
    for name, value in {
        "R_thumb_roll_joint": 0.80,
        "R_thumb_abad_joint": -1.70,
        "R_index_abad_joint": -0.18,
        "R_ring_abad_joint": 0.10,
        "R_pinky_abad_joint": 0.15,
    }.items():
        open_target[finger_names.index(name)] = value
    data.qpos[finger_qpos] = open_target
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        set_source_q(follower, float(relation["multiplier"]) * source_q(driver) + float(relation["offset"]))
    mujoco.mj_forward(model, data)

    close_target = open_target.copy()
    for index, name in enumerate(finger_names):
        target = task.FINGERS[name]
        if "thumb_mcp" in name:
            target = 0.82
        elif name.endswith("_pip_joint"):
            target = 1.15
        close_target[index] = np.clip(target, model.jnt_range[finger_ids[index], 0], model.jnt_range[finger_ids[index], 1])
    close_target[finger_names.index("R_thumb_roll_joint")] = 0.65
    close_target[finger_names.index("R_thumb_abad_joint")] = -0.30
    preshape_target = open_target + task.FINGER_PRESHAPE_FRACTION * (close_target - open_target)

    preview_data = mujoco.MjData(model)
    preview_data.qpos[:] = data.qpos
    preview_data.time = data.time
    preview_data.qpos[arm_qpos] = route[-1]
    preview_data.qpos[finger_qpos] = close_target
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        set_source_q(
            follower,
            float(relation["multiplier"]) * source_q(driver, preview_data) + float(relation["offset"]),
            preview_data,
        )
    preview_data.qvel[:] = 0.0
    mujoco.mj_forward(model, preview_data)

    hold_targets = np.asarray([
        float(data.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[aid, 0])])])
        for aid in range(model.nu)
    ])
    data.ctrl[:] = hold_targets
    data.ctrl[arm_ctrl] = route[0]
    data.ctrl[finger_ctrl] = open_target
    mujoco.mj_forward(model, data)
    rollout_bottle_start = data.qpos[bottle_qadr:bottle_qadr + 7].copy()

    hand_joint_names = [
        node.get("name") for node in source_root.findall("joint")
        if node.get("name", "").startswith(("L_", "R_")) and node.get("type") == "revolute"
    ]
    hand_qpos = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in hand_joint_names
    }
    phase_rows: dict[str, dict[str, object]] = {}
    trace_path = out / "physics_contact_trace.jsonl"
    trace = trace_path.open("w", encoding="utf-8")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.30, 0.17, 0.97]
    camera.distance, camera.azimuth, camera.elevation = 1.35, 135, -12
    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.geomgroup[0] = 0
    render_option.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=720, width=1280)
    preview_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(preview_camera)
    preview_camera.lookat[:] = [task.CANONICAL_X2_BOTTLE_START_BODY_POS[0], task.CANONICAL_X2_BOTTLE_START_BODY_POS[1], task.CANONICAL_X2_BOTTLE_START_BODY_POS[2] + 0.06]
    preview_camera.distance, preview_camera.azimuth, preview_camera.elevation = 0.78, 180, -8
    renderer.update_scene(preview_data, camera=preview_camera, scene_option=render_option)
    imageio.imwrite(out / "right_hand_kinematic_bottle_preview.png", renderer.render())
    if args.kinematic_preview_only:
        preview_identity = {
            "robotsim_branch": branch,
            "robotsim_head": head,
            "preview_script_sha256": sha256(Path(__file__).resolve()),
            "vendor_pin": vendor_head,
            "urdf_sha256": sha256(task.URDF),
            "python": subprocess.check_output([sys.executable, "--version"], text=True).strip(),
            "mujoco": mujoco.__version__,
            "physics_stepped": False,
            "contact_or_grasp_evidence": False,
            "bottle_qpos_written": False,
            "follower_qpos_written_in_preview_copy": True,
            "preview_data": "separate MjData copy from the source-faithful single-hand model",
        }
        (out / "preview_identity.json").write_text(json.dumps(preview_identity, indent=2) + "\n", encoding="utf-8")
        renderer.close()
        return 0
    video = imageio.get_writer(out / "single_hand_contact_probe.mp4", fps=25, codec="libx264", quality=8)
    step_count = 0
    max_mimic_error = 0.0
    first_mimic_breach: dict[str, object] | None = None
    max_penetration = 0.0
    max_step_translation = 0.0
    first_failure: dict[str, object] | None = None
    bottle_peak_z = float(rollout_bottle_start[2])
    right_families_seen: set[str] = set()
    left_contact_seen = False
    upward_force_samples: list[float] = []
    active_rollout_bottle_qpos_assignments = 0

    def mimic_state() -> tuple[float, dict[str, float]]:
        errors: dict[str, float] = {}
        for relation in details["mimic_relations"]:
            driver = source_q(str(relation["driver_joint"]))
            follower = source_q(str(relation["follower_joint"]))
            expected = float(relation["multiplier"]) * driver + float(relation["offset"])
            errors[str(relation["follower_joint"])] = abs(follower - expected)
        return max(errors.values(), default=0.0), errors

    def contact_snapshot() -> tuple[list[dict[str, object]], set[str], set[str], float, float]:
        rows: list[dict[str, object]] = []
        right_bodies: set[str] = set()
        left_bodies: set[str] = set()
        upward = 0.0
        penetration = 0.0
        for ci in range(data.ncon):
            contact = data.contact[ci]
            geoms = (int(contact.geom1), int(contact.geom2))
            if not bottle_geom_ids.intersection(geoms):
                continue
            bottle_geom = geoms[0] if geoms[0] in bottle_geom_ids else geoms[1]
            other_geom = geoms[1] if bottle_geom == geoms[0] else geoms[0]
            body_id = int(model.geom_bodyid[other_geom])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
            ancestors = task.body_ancestors(model, body_id) if body_id else ["world"]
            family = next((f for f in FINGER_FAMILIES if any(n.lower().startswith(f"r_{f}_") for n in ancestors)), None)
            if family:
                right_bodies.add(ancestors[0])
            if any(n.startswith("L_") for n in ancestors):
                left_bodies.add(ancestors[0])
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, ci, force)
            normal_world = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
            sign = 1.0 if bottle_geom == geoms[1] else -1.0
            vertical = sign * float(force[0]) * float(normal_world[2])
            if family:
                upward += vertical
            depth = max(0.0, -float(contact.dist))
            penetration = max(penetration, depth)
            rows.append({
                "geom_pair": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms],
                "other_body": body_name,
                "finger_family": family,
                "distance_m": float(contact.dist),
                "penetration_m": depth,
                "force_contact_frame_N": force.tolist(),
                "estimated_upward_hand_force_N": vertical if family else 0.0,
            })
        return rows, right_bodies, left_bodies, upward, penetration

    def advance(label: str, steps: int, arm_target: np.ndarray, finger_target: np.ndarray, finger_steps: int | None = None) -> bool:
        nonlocal step_count, max_mimic_error, first_mimic_breach, max_penetration
        nonlocal max_step_translation, first_failure, bottle_peak_z, left_contact_seen
        arm_start = data.ctrl[arm_ctrl].copy()
        fingers_start = data.ctrl[finger_ctrl].copy()
        finger_steps = steps if finger_steps is None else finger_steps
        stats = {
            "steps": 0,
            "contact_steps": 0,
            "upward_force_N": [],
            "families": set(),
            "bodies": set(),
            "max_hand_target_error_rad": 0.0,
        }
        phase_rows[label] = stats
        for k in range(max(steps, finger_steps)):
            data.ctrl[:] = hold_targets
            data.ctrl[arm_ctrl] = arm_start * (1.0 - min((k + 1) / steps, 1.0)) + arm_target * min((k + 1) / steps, 1.0)
            data.ctrl[finger_ctrl] = fingers_start * (1.0 - min((k + 1) / finger_steps, 1.0)) + finger_target * min((k + 1) / finger_steps, 1.0)
            old_bottle = data.qpos[bottle_qadr:bottle_qadr + 3].copy()
            mujoco.mj_step(model, data)
            step_count += 1
            stats["steps"] += 1
            error, errors = mimic_state()
            max_mimic_error = max(max_mimic_error, error)
            contacts, right_bodies, left_bodies, upward, penetration = contact_snapshot()
            families = {str(row["finger_family"]) for row in contacts if row["finger_family"]}
            right_families_seen.update(families)
            left_contact_seen = left_contact_seen or bool(left_bodies)
            stats["families"].update(families)
            stats["bodies"].update(right_bodies)
            if right_bodies:
                stats["contact_steps"] += 1
            stats["upward_force_N"].append(upward)
            upward_force_samples.append(upward)
            hand_target_error = max(
                (
                    abs(float(data.qpos[qadr]) - float(data.ctrl[actuator_by_joint[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]))
                    for name, qadr in hand_qpos.items()
                    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) in actuator_by_joint
                ),
                default=0.0,
            )
            stats["max_hand_target_error_rad"] = max(stats["max_hand_target_error_rad"], hand_target_error)
            bottle_position = data.qpos[bottle_qadr:bottle_qadr + 3].copy()
            translation_step = float(np.linalg.norm(bottle_position - old_bottle))
            max_step_translation = max(max_step_translation, translation_step)
            max_penetration = max(max_penetration, penetration)
            bottle_peak_z = max(bottle_peak_z, float(bottle_position[2]))
            fail_reasons = []
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.actuator_force).all():
                fail_reasons.append("non_finite_state")
            if error > MIMIC_LIMIT_RAD and first_mimic_breach is None:
                offender = max(errors, key=errors.get)
                first_mimic_breach = {"phase": label, "step": step_count, "time_s": float(data.time), "joint": offender, "error_rad": errors[offender]}
                fail_reasons.append("source_mimic_error_exceeded")
            if penetration > PENETRATION_LIMIT_M:
                fail_reasons.append("bottle_penetration_limit_exceeded")
            if left_bodies:
                fail_reasons.append("opposite_hand_contact")
            if fail_reasons and first_failure is None:
                first_failure = {"phase": label, "step": step_count, "time_s": float(data.time), "reasons": fail_reasons}
            trace.write(json.dumps({
                "step": step_count,
                "time_s": float(data.time),
                "phase": label,
                "mimic_error_max_rad": error,
                "mimic_errors_rad": errors,
                "right_hand_joint_qpos_rad": {name: float(data.qpos[qadr]) for name, qadr in hand_qpos.items()},
                "finger_targets_rad": {name: float(value) for name, value in zip(finger_names, data.ctrl[finger_ctrl])},
                "bottle_position_m": bottle_position.tolist(),
                "bottle_linear_velocity_m_s": data.qvel[bottle_dadr:bottle_dadr + 3].tolist(),
                "bottle_angular_velocity_rad_s": data.qvel[bottle_dadr + 3:bottle_dadr + 6].tolist(),
                "right_contact_bodies": sorted(right_bodies),
                "contact_families": sorted(families),
                "left_contact_bodies": sorted(left_bodies),
                "contacts": contacts,
                "upward_hand_force_estimate_N": upward,
                "bottle_weight_N": task.CANONICAL_X2_BOTTLE_MASS_KG * 9.81,
                "bottle_qpos_assignments_this_step": 0,
            }, separators=(",", ":")) + "\n")
            if step_count % 20 == 0 or step_count == 1:
                renderer.update_scene(data, camera=camera, scene_option=render_option)
                video.append_data(renderer.render())
            if fail_reasons:
                return False
        return True

    reproduction = (
        "MUJOCO_GL=egl "
        f"AGIBOT_X2_VENDOR_ROOT={task.ROOT} "
        f"ISSUE46_EVIDENCE_DIR={out} "
        f"ISSUE46_MIMIC_EQ_SOLREF_SCALE={task.MIMIC_EQ_SOLREF_SCALE} "
        f"ISSUE46_MIMIC_DRIVER_GAIN_SCALE={task.MIMIC_DRIVER_GAIN_SCALE} "
        f"ISSUE46_APPROACH_FINGER_DURATION_STEPS={task.APPROACH_FINGER_DURATION_STEPS} "
        f"{sys.executable} {Path(__file__).resolve()} --output {out}"
    )
    (out / "reproduction_command.txt").write_text(reproduction + "\n", encoding="utf-8")
    identity = {
        "robotsim_branch": branch,
        "robotsim_head": head,
        "robotsim_dirty_paths": dirty,
        "probe_script_sha256": sha256(Path(__file__).resolve()),
        "vendor_pin": vendor_head,
        "vendor_dirty": False,
        "urdf_sha256": sha256(task.URDF),
        "python": subprocess.check_output([sys.executable, "--version"], text=True).strip(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "timestep_s": float(model.opt.timestep),
        "model_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq, "nbody": model.nbody},
        "mimic_mode": "source MuJoCo joint equalities; no virtual-transmission approximation",
        "mimic_solref_scale": task.MIMIC_EQ_SOLREF_SCALE,
        "mimic_driver_gain_scale": task.MIMIC_DRIVER_GAIN_SCALE,
        "finger_ramp_steps": task.APPROACH_FINGER_DURATION_STEPS,
        "base": "official URDF fixed-base topology, placed at the existing Issue #46 right-arm station",
        "bottle_mass_kg": float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")]),
        "bottle_geometry_and_mass_modified": False,
        "opposite_hand_control": "left arm/hand held at the existing neutral pose",
        "runtime_bottle_qpos_writes": 0,
        "runtime_follower_qpos_writes": 0,
        "bottle_hand_preview": {
            "file": "right_hand_kinematic_bottle_preview.png",
            "kind": "static kinematic-only frame from a separate MjData copy",
            "physics_stepped": False,
            "contact_or_grasp_evidence": False,
        },
    }
    (out / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    try:
        renderer.update_scene(data, camera=camera, scene_option=render_option)
        video.append_data(renderer.render())
        phases = [
            ("open_hold", 250, route[0], open_target, 250),
            ("approach_preshape", 500, route[1], preshape_target, task.APPROACH_FINGER_DURATION_STEPS),
            ("finger_close", 500, route[1], close_target, 500),
            ("closed_hand_approach", 500, route[2], close_target, 500),
            ("grasp_hold", 500, route[2], close_target, 500),
        ]
        for label, steps, arm_target, finger_target, finger_steps in phases:
            if not advance(label, steps, np.asarray(arm_target), np.asarray(finger_target), finger_steps):
                break
        trace.flush()
    finally:
        trace.close()
        video.close()
        renderer.close()

    final_contacts, final_right_bodies, final_left_bodies, final_upward, final_penetration = contact_snapshot()
    bottle_final = data.qpos[bottle_qadr:bottle_qadr + 7].copy()
    hold = phase_rows.get("grasp_hold", {})
    hold_force = list(hold.get("upward_force_N", []))
    hold_contact_steps = int(hold.get("contact_steps", 0))
    hold_steps = int(hold.get("steps", 0))
    hold_families = sorted(hold.get("families", set()))
    hold_force_mean = float(np.mean(hold_force)) if hold_force else 0.0
    bottle_weight = task.CANONICAL_X2_BOTTLE_MASS_KG * 9.81
    contact_load_transfer = bool(hold_contact_steps and hold_force_mean > 0.05 * bottle_weight)
    physical_grasp_gate = bool("thumb" in hold_families and len(hold_families) >= 3 and hold_contact_steps / max(hold_steps, 1) >= 0.90)
    lift_precondition = bool(
        physical_grasp_gate
        and hold_force_mean >= 0.80 * bottle_weight
        and not left_contact_seen
        and first_failure is None
    )
    lift_attempted = False
    lift_milestones: list[dict[str, object]] = []
    if lift_precondition:
        lift_attempted = True
        planned_qpos = data.qpos.copy()
        for height in (0.001, 0.005, 0.030, 0.050):
            target_pos = np.array([task.GRASP_PALM_POS[0], task.GRASP_PALM_POS[1], task.GRASP_PALM_POS[2] + height])
            target = task.ik_plan(model, planned_qpos, site_id, arm_ids, [target_pos], task.PALM_TARGET_ROTATION.copy())[0]
            planned_qpos[arm_qpos] = target
            if not advance(f"lift_{int(height * 1000)}mm", 250, target, close_target, 250):
                break
            achieved = float(data.qpos[bottle_qadr + 2] - rollout_bottle_start[2])
            lift_milestones.append({"commanded_palm_lift_m": height, "bottle_lift_from_rollout_start_m": achieved, "right_contact_bodies": sorted(contact_snapshot()[1])})
            if achieved < height * 0.5:
                break

    result = {
        "status": "FAIL" if first_failure else "PASS_TO_CONTACT_CHECKPOINT",
        "first_failure": first_failure,
        "source_mimic_first_breach": first_mimic_breach,
        "max_mimic_error_rad": max_mimic_error,
        "source_mimic_gate_rad": MIMIC_LIMIT_RAD,
        "right_contact_families_seen": sorted(right_families_seen),
        "left_hand_contact_seen": left_contact_seen,
        "grasp_hold": {
            "steps": hold_steps,
            "contact_fraction": hold_contact_steps / max(hold_steps, 1),
            "finger_families": hold_families,
            "upward_support_mean_N": hold_force_mean,
            "upward_support_peak_N": max(hold_force, default=0.0),
            "bottle_weight_N": bottle_weight,
            "support_fraction_mean": hold_force_mean / bottle_weight,
            "contact_load_transfer_gate": contact_load_transfer,
            "physical_grasp_gate": physical_grasp_gate,
            "lift_precondition": lift_precondition,
        },
        "final": {
            "sim_time_s": float(data.time),
            "steps": step_count,
            "bottle_initial_qpos": rollout_bottle_start.tolist(),
            "bottle_final_qpos": bottle_final.tolist(),
            "bottle_peak_z_m": bottle_peak_z,
            "bottle_max_translation_step_m": max_step_translation,
            "max_bottle_penetration_m": max(max_penetration, final_penetration),
            "final_right_bodies": sorted(final_right_bodies),
            "final_left_bodies": sorted(final_left_bodies),
            "final_upward_hand_force_estimate_N": final_upward,
            "final_contact_rows": final_contacts,
            "active_rollout_bottle_qpos_writes": active_rollout_bottle_qpos_assignments,
        },
        "lift_attempted": lift_attempted,
        "lift_milestones": lift_milestones,
        "phases": {
            name: {
                "steps": int(row["steps"]),
                "contact_steps": int(row["contact_steps"]),
                "contact_fraction": int(row["contact_steps"]) / max(int(row["steps"]), 1),
                "finger_families": sorted(row["families"]),
                "right_contact_bodies": sorted(row["bodies"]),
                "max_hand_target_error_rad": float(row["max_hand_target_error_rad"]),
                "upward_force_mean_N": float(np.mean(row["upward_force_N"])) if row["upward_force_N"] else 0.0,
            }
            for name, row in phase_rows.items()
        },
        "reproduction_command": reproduction,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if first_failure is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
