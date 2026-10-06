#!/usr/bin/env python3
"""Gate A initialization and Gate B loaded mimic diagnostics; bottle collision stays off."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_DIR))
import issue46_mimic_runtime_parity as parity
import issue46_x2_grasp as task

TOLERANCE = 0.003
STEPS = int(os.environ.get("ISSUE46_LOADED_MIMIC_STEPS", "500"))
OUT = Path(os.environ.get(
    "ISSUE46_LOADED_MIMIC_EVIDENCE_DIR",
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/runtime-parity-20261006-v16",
))

LEFT_ARM_SOURCE_POSE = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.10,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.15,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
}
LEFT_HAND_SOURCE_POSE = {
    "L_thumb_abad_joint": 0.35,
    "L_thumb_mcp_joint": -0.10,
    "L_index_abad_joint": 0.10,
    "L_index_pip_joint": 0.05,
    "L_middle_pip_joint": 0.05,
    "L_ring_abad_joint": -0.10,
    "L_ring_pip_joint": 0.05,
    "L_pinky_abad_joint": -0.16,
    "L_pinky_pip_joint": 0.05,
}
RIGHT_HAND_SOURCE_POSE = {
    "R_thumb_mcp_joint": 0.05,
    "R_index_pip_joint": 0.05,
    "R_middle_pip_joint": 0.05,
    "R_ring_abad_joint": 0.10,
    "R_ring_pip_joint": 0.05,
    "R_pinky_pip_joint": 0.05,
}


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qid(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"missing joint {name}")
    return jid


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[qid(model, name)])


def axis_sign(model: mujoco.MjModel, name: str, source_joints: dict[str, Any]) -> int:
    source_axis = np.fromstring(source_joints[name].find("axis").get("xyz", "1 0 0"), sep=" ")
    compiled_axis = model.jnt_axis[qid(model, name)]
    return 1 if float(np.dot(source_axis, compiled_axis)) >= 0.0 else -1


def source_coordinate(model: mujoco.MjModel, data: mujoco.MjData, name: str, refs: dict[str, float], source_joints: dict[str, Any]) -> float:
    address = qpos_address(model, name)
    return float(refs.get(name, 0.0)) + axis_sign(model, name, source_joints) * (
        float(data.qpos[address]) - float(model.qpos0[address])
    )


def source_to_qpos(model: mujoco.MjModel, name: str, coordinate: float, refs: dict[str, float], source_joints: dict[str, Any]) -> float:
    address = qpos_address(model, name)
    return float(model.qpos0[address]) + axis_sign(model, name, source_joints) * (
        float(coordinate) - float(refs.get(name, 0.0))
    )


def all_joint_margins(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_joints: dict[str, Any],
    refs: dict[str, float],
) -> list[dict[str, Any]]:
    result = []
    for jid in range(model.njnt):
        if not bool(model.jnt_limited[jid]):
            continue
        name = parity.id_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        node = source_joints.get(name)
        limit = None if node is None else node.find("limit")
        if limit is None:
            continue
        src_lo, src_hi = float(limit.get("lower")), float(limit.get("upper"))
        qadr = int(model.jnt_qposadr[jid])
        raw_qpos = float(data.qpos[qadr])
        src_q = source_coordinate(model, data, name, refs, source_joints)
        model_lo, model_hi = map(float, model.jnt_range[jid])
        result.append({
            "joint": name,
            "source_coordinate_rad": src_q,
            "source_range_rad": [src_lo, src_hi],
            "source_margin_rad": min(src_q - src_lo, src_hi - src_q),
            "mujoco_qpos_rad": raw_qpos,
            "mujoco_range_rad": [model_lo, model_hi],
            "mujoco_margin_rad": min(raw_qpos - model_lo, model_hi - raw_qpos),
            "inside_source_range": src_lo - 1e-12 <= src_q <= src_hi + 1e-12,
        })
    return result


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        bodies = []
        geoms = []
        for gid in geom_ids:
            geoms.append(parity.id_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid))
            bid = int(model.geom_bodyid[gid])
            bodies.append(parity.id_name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
        distance = float(contact.dist)
        rows.append({
            "geom_pair": geoms,
            "body_pair": bodies,
            "signed_distance_m": distance,
            "penetration_depth_m": max(0.0, -distance),
        })
    return sorted(rows, key=lambda item: (item["body_pair"], item["geom_pair"]))


def mimic_records(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return parity.pair_state(
        state["model"], state["data"], state["source_joints"],
        state["source_relations"], state["refs"], state["eq_ids"],
    )


def all_joint_violations(margins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in margins if not row["inside_source_range"]]


def initialize_valid_neutral(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model, data = state["model"], state["data"]
    source_joints = state["source_joints"]
    relations = state["source_relations"]
    refs = state["refs"]
    before = data.qpos.copy()

    for name, value in LEFT_ARM_SOURCE_POSE.items():
        data.qpos[qpos_address(model, name)] = source_to_qpos(model, name, value, refs, source_joints)
    for name, value in LEFT_HAND_SOURCE_POSE.items():
        data.qpos[qpos_address(model, name)] = source_to_qpos(model, name, value, refs, source_joints)
    for name, value in RIGHT_HAND_SOURCE_POSE.items():
        data.qpos[qpos_address(model, name)] = source_to_qpos(model, name, value, refs, source_joints)

    # Seed each follower once before the rollout from the compiled source mimic relation.
    for follower, relation in relations.items():
        driver = str(relation["driver_joint"])
        theta = source_coordinate(model, data, driver, refs, source_joints)
        follower_source = float(relation["offset"]) + float(relation["multiplier"]) * theta
        data.qpos[qpos_address(model, follower)] = source_to_qpos(
            model, follower, follower_source, refs, source_joints
        )

    actuator_map = parity.actuator_by_joint(model)
    changed_names = set(LEFT_ARM_SOURCE_POSE) | set(LEFT_HAND_SOURCE_POSE) | set(RIGHT_HAND_SOURCE_POSE)
    for name in changed_names:
        aid = actuator_map.get(name)
        if aid is not None:
            target = float(data.qpos[qpos_address(model, name)])
            data.ctrl[aid] = target
            state["hold_targets"][aid] = target

    for name in RIGHT_HAND_SOURCE_POSE:
        value = float(data.qpos[qpos_address(model, name)])
        finger_index = state["finger_qpos"].index(qpos_address(model, name))
        state["open_target"][finger_index] = value
        aid = actuator_map.get(name)
        if aid is not None:
            state["hold_targets"][aid] = value
        if name == "R_ring_abad_joint":
            state["preshape_target"][finger_index] = value

    # Keep the manipulation builder's accepted mimic-driver profile, with isolated candidate overrides.
    driver_names = sorted({str(item["driver_joint"]) for item in relations.values()})
    if args.gain_scale != 1.0:
        for name in driver_names:
            parity.set_position_actuator_profile(
                model, name,
                float(task.RIGHT_MIMIC_DRIVER_SERVO_KP) * args.gain_scale,
                float(task.RIGHT_MIMIC_DRIVER_SERVO_KV) * args.gain_scale,
            )
    if args.eq_scale != 1.0:
        for eid in state["eq_ids"].values():
            model.eq_solref[eid, :2] *= args.eq_scale
    if args.solver_iterations is not None:
        model.opt.iterations = args.solver_iterations

    mujoco.mj_forward(model, data)
    margins = all_joint_margins(model, data, source_joints, refs)
    pairs = mimic_records(state)
    contacts = contact_records(model, data)
    baseline_changes = []
    for index, delta in enumerate(data.qpos - before):
        if abs(float(delta)) > 1e-14:
            baseline_changes.append({"qpos_address": index, "delta_rad_or_m": float(delta)})
    return {
        "initial_qpos_changes_before_rollout": baseline_changes,
        "corrected_left_arm_source_pose_rad": LEFT_ARM_SOURCE_POSE,
        "corrected_left_hand_source_pose_rad": LEFT_HAND_SOURCE_POSE,
        "right_hand_collision_correction_source_pose_rad": RIGHT_HAND_SOURCE_POSE,
        "contacts_before_first_step": contacts,
        "max_initial_self_penetration_m": max((item["penetration_depth_m"] for item in contacts), default=0.0),
        "joint_limit_margins_before_first_step": margins,
        "joint_limit_violations_before_first_step": all_joint_violations(margins),
        "mimic_relations_before_first_step": pairs,
        "max_mimic_error_before_first_step_rad": max(abs(float(item["equality_residual_rad"])) for item in pairs.values()),
        "mimic_driver_gain_scale_simulation_derived": args.gain_scale,
        "equality_solref_scale_simulation_derived": args.eq_scale,
        "solver_iterations_simulation_derived": args.solver_iterations,
        "source_parameters_changed": [],
    }


def capture_stage(label: str, state: dict[str, Any]) -> dict[str, Any]:
    model, data = state["model"], state["data"]
    pairs = mimic_records(state)
    margins = all_joint_margins(model, data, state["source_joints"], state["refs"])
    return {
        "stage": label,
        "time_s": float(data.time),
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
        "max_mimic_error_rad": max(abs(float(item["equality_residual_rad"])) for item in pairs.values()),
        "mimic_relations": pairs,
        "joint_limit_margins": margins,
        "joint_limit_violations": all_joint_violations(margins),
        "contacts": contact_records(model, data),
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }


def set_scaled_targets(state: dict[str, Any], index: int, arm_steps: int, finger_steps: int) -> None:
    model, data = state["model"], state["data"]
    arm_start = np.asarray(state["approach_route"][0], dtype=float)
    arm_target = np.asarray(state["approach_route"][1], dtype=float)
    open_target = np.asarray(state["open_target"], dtype=float)
    finger_target = (
        open_target
        if state.get("open_fingers_during_approach")
        else np.asarray(state["preshape_target"], dtype=float)
    )
    arm_alpha = min((index + 1) / arm_steps, 1.0)
    finger_alpha = min((index + 1) / finger_steps, 1.0)
    data.ctrl[:] = state["hold_targets"]
    if state.get("freeze_arm_during_diagnostic"):
        data.ctrl[state["arm_ctrl"]] = arm_start
    else:
        data.ctrl[state["arm_ctrl"]] = arm_start * (1.0 - arm_alpha) + arm_target * arm_alpha
    data.ctrl[state["finger_ctrl"]] = open_target * (1.0 - finger_alpha) + finger_target * finger_alpha


def full_close_target(state: dict[str, Any]) -> np.ndarray:
    model = state["model"]
    target = state["open_target"].copy()
    finger_names = list(task.FINGERS)
    for index, name in enumerate(finger_names):
        value = task.FINGERS[name]
        if "thumb_mcp" in name:
            value = 0.82
        elif name.endswith("_pip_joint"):
            value = 1.15
        jid = qid(model, name)
        target[index] = np.clip(value, model.jnt_range[jid, 0], model.jnt_range[jid, 1])
    target = state["open_target"] + task.FINGER_CLOSE_FRACTION * (target - state["open_target"])
    target[finger_names.index("R_thumb_roll_joint")] = 0.65
    target[finger_names.index("R_thumb_abad_joint")] = -0.30
    return target


def configure_case(state: dict[str, Any], case: str) -> None:
    model = state["model"]
    if case in ("arm_motion_only", "self_contact_only"):
        model.opt.gravity[:] = [0.0, 0.0, 0.0]
    if case == "arm_motion_only":
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        state["freeze_arm_during_diagnostic"] = False
        state["open_fingers_during_approach"] = True
    elif case == "gravity_only":
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        state["freeze_arm_during_diagnostic"] = True
        state["open_fingers_during_approach"] = True
    elif case == "finger_motion_only":
        model.opt.gravity[:] = [0.0, 0.0, 0.0]
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        state["freeze_arm_during_diagnostic"] = True
        state["open_fingers_during_approach"] = False
        state["preshape_target"] = full_close_target(state)
    elif case == "self_contact_only":
        state["freeze_arm_during_diagnostic"] = True
        state["open_fingers_during_approach"] = False
        state["preshape_target"] = full_close_target(state)
        state["diagnostic_steps"] = int(os.environ.get("ISSUE46_LOADED_MIMIC_SELF_CONTACT_STEPS", "7000"))
    elif case == "gravity_plus_arm_motion":
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        state["freeze_arm_during_diagnostic"] = False
        state["open_fingers_during_approach"] = True
    elif case == "full_approach":
        state["freeze_arm_during_diagnostic"] = False
        state["open_fingers_during_approach"] = False
    else:
        raise ValueError(case)


def run_case(state: dict[str, Any], case: str, trace_path: Path) -> dict[str, Any]:
    configure_case(state, case)
    model, data = state["model"], state["data"]
    steps = int(state.get("diagnostic_steps", STEPS))
    finger_steps = (
        int(round(steps * state["finger_duration_scale"]))
        if case in ("full_approach", "finger_motion_only")
        else steps
    )
    execution_steps = max(steps, finger_steps)
    mujoco.mj_forward(model, data)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    first_breach = None
    first_limit_violation = None
    max_error = 0.0
    max_violation_count = 0
    max_penetration = 0.0
    max_contact_count = 0
    nan_seen = False
    sign_inversions: list[dict[str, Any]] = []
    stage_records = []
    self_contact_first_step = None
    self_contact_hold_steps = 0
    self_contact_persistent_steps = 0
    self_contact_hold_ctrl = None
    self_contact_depth_gate_passed = True

    for index in range(execution_steps):
        set_scaled_targets(state, index, steps, finger_steps)
        if case == "self_contact_only" and self_contact_hold_ctrl is not None:
            data.ctrl[state["finger_ctrl"]] = self_contact_hold_ctrl
        if index == 0:
            stage_records.append(capture_stage("D_immediately_before_first_mj_step", state))
        mujoco.mj_step(model, data)
        pairs = mimic_records(state)
        errors = {name: abs(float(item["equality_residual_rad"])) for name, item in pairs.items()}
        current_error = max(errors.values(), default=0.0)
        margins = all_joint_margins(model, data, state["source_joints"], state["refs"])
        violations = all_joint_violations(margins)
        contacts = contact_records(model, data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        max_error = max(max_error, current_error)
        max_violation_count = max(max_violation_count, len(violations))
        max_penetration = max(max_penetration, max((c["penetration_depth_m"] for c in contacts), default=0.0))
        max_contact_count = max(max_contact_count, len(contacts))
        nan_seen = nan_seen or not finite
        if case == "self_contact_only":
            if self_contact_first_step is None and contacts:
                self_contact_first_step = index + 1
                self_contact_hold_ctrl = data.ctrl[state["finger_ctrl"]].copy()
                if max(c["penetration_depth_m"] for c in contacts) > 0.001:
                    self_contact_depth_gate_passed = False
            elif self_contact_hold_ctrl is not None:
                self_contact_hold_steps += 1
                self_contact_persistent_steps += int(bool(contacts))
                if max((c["penetration_depth_m"] for c in contacts), default=0.0) > 0.001:
                    self_contact_depth_gate_passed = False
        if first_breach is None and current_error > TOLERANCE:
            offender = max(errors, key=errors.get)
            first_breach = {"step": index + 1, "time_s": float(data.time), "relation": offender, "error_rad": errors[offender]}
        if first_limit_violation is None and violations:
            first_limit_violation = {"step": index + 1, "time_s": float(data.time), "violations": violations}
        for name, item in pairs.items():
            target_delta = float(item["driver_actuator"]["target_rad"]) - float(item["driver_qpos0"])
            actual_delta = float(item["driver_qpos"]) - float(item["driver_qpos0"])
            if abs(target_delta) > 1e-6 and actual_delta * target_delta < -1e-8:
                sign_inversions.append({"step": index + 1, "relation": name, "actual_delta": actual_delta, "target_delta": target_delta})
        row = {
            "case": case,
            "step": index + 1,
            "time_s": float(data.time),
            "max_mimic_error_rad": current_error,
            "mimic_relations": parity.compact_relation_trace(pairs),
            "joint_limit_violations": violations,
            "contacts": contacts,
            "max_penetration_m": max((c["penetration_depth_m"] for c in contacts), default=0.0),
            "finite_state": finite,
            "active_rollout_follower_qpos_write_count": 0,
            "self_contact_first_step": self_contact_first_step,
            "self_contact_hold_steps": self_contact_hold_steps,
            "self_contact_persistent_steps": self_contact_persistent_steps,
            "self_contact_depth_gate_passed_at_1mm": self_contact_depth_gate_passed,
        }
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps(row, sort_keys=True) + "\n")
        if index == 0:
            stage_records.append(capture_stage("E_immediately_after_first_mj_step", state))
        if case == "self_contact_only" and self_contact_hold_ctrl is not None:
            if not self_contact_depth_gate_passed or self_contact_hold_steps >= 500:
                break

    final_margins = all_joint_margins(model, data, state["source_joints"], state["refs"])
    final_pairs = mimic_records(state)
    executed_steps = index + 1
    result = {
        "case": case,
        "requested_steps": steps,
        "steps": executed_steps,
        "arm_target_duration_s": steps * float(model.opt.timestep),
        "finger_target_duration_s": finger_steps * float(model.opt.timestep),
        "duration_s": float(data.time),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "timestep_s": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "solver_iterations": int(model.opt.iterations),
        "max_mimic_error_rad": max_error,
        "first_breach": first_breach,
        "max_source_joint_limit_violation_count": max_violation_count,
        "first_limit_violation": first_limit_violation,
        "all_source_joint_limits_inside_at_end": not all_joint_violations(final_margins),
        "sign_inversion_count": len(sign_inversions),
        "first_sign_inversions": sign_inversions[:8],
        "nan_seen": nan_seen,
        "max_contact_count": max_contact_count,
        "max_penetration_m": max_penetration,
        "self_contact_first_step": self_contact_first_step,
        "self_contact_hold_steps": self_contact_hold_steps,
        "self_contact_persistent_steps": self_contact_persistent_steps,
        "self_contact_persistence_fraction": (
            self_contact_persistent_steps / self_contact_hold_steps if self_contact_hold_steps else None
        ),
        "self_contact_depth_gate_passed_at_1mm": self_contact_depth_gate_passed,
        "bottle_collision_disabled": all(
            int(model.geom_contype[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]) == 0
            and int(model.geom_conaffinity[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]) == 0
            for name in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap")
        ),
        "active_rollout_follower_qpos_writes": 0,
        "pass_gate": max_error <= TOLERANCE
        and max_violation_count == 0
        and len(sign_inversions) == 0
        and not nan_seen
        and result_bottle_disabled(model),
    }
    dump(trace_path.with_suffix(".summary.json"), result)
    return {"summary": result, "stages": stage_records}


def result_bottle_disabled(model: mujoco.MjModel) -> bool:
    for name in ("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0 or int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0:
            return False
    return True


def prepare_case(case: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    state = parity.manipulation_setup("manipulation_baseline", {})
    state["finger_duration_scale"] = args.finger_duration_scale
    configure_case(state, case)
    gate_a = initialize_valid_neutral(state, args)
    return state, gate_a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain-scale", type=float, default=1.0, help="simulation-derived scale for mimic driver kp and kv")
    parser.add_argument("--eq-scale", type=float, default=1.0, help="simulation-derived scale for direct equality solref coefficients")
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--finger-duration-scale", type=float, default=1.0, help="simulation-derived scale for finger target duration only")
    parser.add_argument("--only-approach", action="store_true", help="run only the bottle-disabled full approach case")
    args = parser.parse_args()
    if not 0.0 < args.gain_scale or not 0.0 < args.eq_scale or not 0.0 < args.finger_duration_scale:
        parser.error("parameter scales must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"evidence directory must be empty: {OUT}")
    source_joints, relations = parity.source_records()
    refs = parity.spec_refs()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "source_pin": task.SOURCE_PIN,
        "urdf": str(task.URDF),
        "urdf_sha256": hashlib.sha256(task.URDF.read_bytes()).hexdigest(),
        "vendor_head": subprocess.check_output(["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "robot_sim_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "robot_sim_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "robot_sim_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip(),
        "tolerance_rad": TOLERANCE,
        "steps_per_case": STEPS,
        "reproduction_command": "MUJOCO_GL=egl ISSUE46_LOADED_MIMIC_EVIDENCE_DIR=<empty-directory> /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python scripts/research/issue46_loaded_mimic_gate.py",
        "source_joint_relation_count": len(relations),
        "source_parameters_changed": [],
        "pose_parameters_are_simulation_derived": True,
        "mimic_driver_gain_scale": args.gain_scale,
        "equality_solref_scale": args.eq_scale,
        "solver_iterations_override": args.solver_iterations,
        "finger_duration_scale_simulation_derived": args.finger_duration_scale,
        "only_approach_candidate": args.only_approach,
    }
    dump(OUT / "runtime_identity.json", runtime)
    initial_state, gate_a = prepare_case("full_approach", args)
    dump(OUT / "gate_a_pre_step.json", gate_a)
    initial_margins = gate_a["joint_limit_margins_before_first_step"]
    init_clean = (
        gate_a["max_initial_self_penetration_m"] <= 1e-12
        and not gate_a["joint_limit_violations_before_first_step"]
        and gate_a["max_mimic_error_before_first_step_rad"] <= TOLERANCE
        and len(gate_a["mimic_relations_before_first_step"]) == 12
    )
    if not init_clean:
        dump(OUT / "result.json", {"status": "BLOCKED_GATE_A_INITIALIZATION", "initialization_clean": False})
        return 3

    gate_a_run = run_case(initial_state, "full_approach", OUT / "gate_a_full_approach.jsonl")
    dump(OUT / "gate_a_stage_a_to_e.json", gate_a_run["stages"])
    if args.only_approach:
        passed = gate_a_run["summary"]["pass_gate"]
        dump(OUT / "result.json", {
            "status": "READY FOR MAINTAINER X2 LOADED-MIMIC REVIEW" if passed else "BLOCKED",
            "gate_a_initialization_clean": True,
            "gate_a_full_approach": gate_a_run["summary"],
            "gate_b_run": False,
            "bottle_work_performed": False,
            "source_parameters_changed": [],
            "simulation_derived_parameters_changed": {
                "left_arm_initial_pose": LEFT_ARM_SOURCE_POSE,
                "left_hand_initial_pose": LEFT_HAND_SOURCE_POSE,
                "right_hand_initial_pose_collision_correction": RIGHT_HAND_SOURCE_POSE,
                "mimic_driver_gain_scale": args.gain_scale,
                "equality_solref_scale": args.eq_scale,
                "finger_duration_scale": args.finger_duration_scale,
                "solver_iterations": args.solver_iterations,
            },
        })
        print(json.dumps(json.loads((OUT / "result.json").read_text()), sort_keys=True))
        return 0 if passed else 2
    if gate_a_run["summary"]["pass_gate"]:
        dump(OUT / "result.json", {
            "status": "READY FOR MAINTAINER X2 LOADED-MIMIC REVIEW",
            "gate_a_initialization_clean": True,
            "gate_a_full_approach": gate_a_run["summary"],
            "gate_b_run": False,
        })
        return 0

    # Gate B is entered only after Gate A's reset pose is measured collision-free and in-range.
    load_results = {"full_approach": gate_a_run["summary"]}
    stage_collection = {"full_approach": gate_a_run["stages"]}
    for case in ("gravity_only", "arm_motion_only", "finger_motion_only", "self_contact_only", "gravity_plus_arm_motion"):
        state, initial = prepare_case(case, args)
        case_result = run_case(state, case, OUT / f"gate_b_{case}.jsonl")
        case_result["summary"]["initialization_clean"] = (
            initial["max_initial_self_penetration_m"] <= 1e-12
            and not initial["joint_limit_violations_before_first_step"]
            and initial["max_mimic_error_before_first_step_rad"] <= TOLERANCE
        )
        dump(OUT / f"gate_b_{case}_pre_step.json", initial)
        load_results[case] = case_result["summary"]
        stage_collection[case] = case_result["stages"]

    dump(OUT / "gate_b_stage_a_to_e.json", stage_collection)
    dump(OUT / "gate_b_load_cases.json", load_results)
    all_pass = all(item["pass_gate"] for item in load_results.values()) and all(
        item.get("initialization_clean", True) for item in load_results.values()
    )
    result = {
        "status": "READY FOR MAINTAINER X2 LOADED-MIMIC REVIEW" if all_pass else "BLOCKED",
        "gate_a_initialization_clean": True,
        "gate_a_full_approach": gate_a_run["summary"],
        "gate_b_load_cases": load_results,
        "gate_b_all_gates_pass": all_pass,
        "bottle_work_performed": False,
        "source_parameters_changed": [],
        "simulation_derived_parameters_changed": {
            "left_arm_initial_pose": LEFT_ARM_SOURCE_POSE,
            "left_hand_initial_pose": LEFT_HAND_SOURCE_POSE,
            "right_hand_initial_pose_collision_correction": RIGHT_HAND_SOURCE_POSE,
            "mimic_driver_gain_scale": args.gain_scale,
            "equality_solref_scale": args.eq_scale,
            "solver_iterations": args.solver_iterations,
        },
    }
    dump(OUT / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
