from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_virtual_transmission_gate as gate  # noqa: E402
import issue46_x2_grasp as task  # noqa: E402

OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-supported-fixed-body"))
PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD = 0.02
HAND_ABORT_RAD = 0.010


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def support_group(joint_name: str) -> str:
    if any(joint_name.startswith(side + suffix) for side in ("left_", "right_") for suffix in ("hip_", "knee_", "ankle_")):
        return "legs"
    if joint_name.startswith("waist_"):
        return "torso_waist"
    if joint_name.startswith("head_"):
        return "head"
    if any(token in joint_name for token in ("shoulder", "elbow", "wrist")):
        return "arms"
    raise ValueError(f"non-hand source joint has no fixed-body support classification: {joint_name}")


def raw_position(runtime: dict[str, object], joint_name: str, source_position: float) -> float:
    model = runtime["model"]
    joint = runtime["joints"][joint_name]
    qpos = int(joint["qpos"])
    return float(model.qpos0[qpos]) + int(joint["axis_sign"]) * (
        float(source_position) - float(joint["reference"])
    )


def source_from_raw(runtime: dict[str, object], joint_name: str, raw_qpos: float) -> float:
    model = runtime["model"]
    joint = runtime["joints"][joint_name]
    qpos = int(joint["qpos"])
    return float(joint["reference"]) + int(joint["axis_sign"]) * (
        float(raw_qpos) - float(model.qpos0[qpos])
    )


def initialize_support(runtime: dict[str, object]) -> dict[str, object]:
    model = runtime["model"]
    data = runtime["data"]
    joints = runtime["joints"]
    hand_names = set(runtime["details"]["active_hand_joint_names"]) | {
        str(relation["follower_joint"]) for relation in runtime["details"]["mimic_relations"]
    }
    support_names = sorted(set(joints) - hand_names)
    if len(support_names) != 31:
        raise ValueError(f"expected 31 non-hand scalar DOFs, found {len(support_names)}")

    base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
    base_fixed = int(model.body_parentid[base_id]) == 0 and int(model.body_jntnum[base_id]) == 0
    if not base_fixed:
        raise ValueError("base_link is not fixed to world in the supported manipulation model")

    references = {name: gate.source_position(model, data, joints[name]) for name in support_names}
    actuator_by_joint = runtime["actuator_by_joint"]
    specs: dict[str, dict[str, object]] = {}
    for joint_name in support_names:
        joint = joints[joint_name]
        jid = int(joint["joint_id"])
        aid = int(actuator_by_joint[jid])
        gear = float(model.actuator_gear[aid, 0])
        kp = float(model.actuator_gainprm[aid, 0])
        kv = -float(model.actuator_biasprm[aid, 2])
        if abs(abs(gear) - 1.0) > 1e-12 or kp <= 0.0 or kv < 0.0:
            raise ValueError(f"unsupported posture actuator model for {joint_name}: gear={gear}, kp={kp}, kv={kv}")
        if not bool(model.actuator_forcelimited[aid]) or not bool(model.actuator_ctrllimited[aid]):
            raise ValueError(f"support actuator is not bounded for {joint_name}")
        if abs(float(model.actuator_biasprm[aid, 1]) + kp) > 1e-9:
            raise ValueError(f"support actuator is not a position servo for {joint_name}")
        effort_bound = min(
            float(joint["effort_limit"]),
            float(np.max(np.abs(model.actuator_forcerange[aid]))) * abs(gear),
        )
        specs[joint_name] = {
            "group": support_group(joint_name),
            "joint_id": jid,
            "dof": int(joint["dof"]),
            "qpos": int(joint["qpos"]),
            "actuator_id": aid,
            "accepted_source_reference_rad": references[joint_name],
            "source_lower_rad": float(joint["lower"]),
            "source_upper_rad": float(joint["upper"]),
            "source_effort_limit_nm": float(joint["effort_limit"]),
            "actuator_effort_bound_nm": effort_bound,
            "actuator_gear": gear,
            "position_kp_nm_per_rad": kp,
            "position_kv_nms_per_rad": kv,
            "support_method": "existing source-bounded position actuator with bias-compensated control target; no qpos state writes",
        }
    runtime["fixed_body_support"] = {
        "base_fixed_to_world": base_fixed,
        "joint_specs": specs,
        "reference_source_positions": references,
        "support_groups": ["legs", "torso_waist", "head", "arms"],
        "intentionally_free_robot_dofs": [],
        "external_free_objects": "omitted from stages 1-4; bottle free joint is present only for stage 5",
    }
    return runtime["fixed_body_support"]


def apply_posture_support(
    runtime: dict[str, object],
    reference_positions: dict[str, float] | None = None,
) -> dict[str, object]:
    model = runtime["model"]
    data = runtime["data"]
    support = runtime["fixed_body_support"]
    specs = support["joint_specs"]
    references = dict(reference_positions or support["reference_source_positions"])
    if set(references) != set(specs):
        raise ValueError("support target set does not cover exactly the non-hand source joints")

    mujoco.mj_forward(model, data)
    rows: dict[str, dict[str, object]] = {}
    for joint_name, spec in specs.items():
        joint = runtime["joints"][joint_name]
        aid = int(spec["actuator_id"])
        dof = int(spec["dof"])
        gear = float(spec["actuator_gear"])
        kp = float(spec["position_kp_nm_per_rad"])
        raw_reference = raw_position(runtime, joint_name, references[joint_name])
        # A MuJoCo position actuator produces gear^2 * kp times q error. Its
        # target offset therefore supplies qfrc_bias without changing gains.
        requested_raw_target = raw_reference + float(data.qfrc_bias[dof]) / (gear * gear * kp)
        requested_source_target = source_from_raw(runtime, joint_name, requested_raw_target)
        requested_source_target = float(np.clip(
            requested_source_target,
            float(spec["source_lower_rad"]),
            float(spec["source_upper_rad"]),
        ))
        requested_ctrl = gear * raw_position(runtime, joint_name, requested_source_target)
        ctrl_low, ctrl_high = map(float, model.actuator_ctrlrange[aid])
        applied_ctrl = float(np.clip(requested_ctrl, ctrl_low, ctrl_high))
        applied_raw_target = applied_ctrl / gear
        applied_source_target = source_from_raw(runtime, joint_name, applied_raw_target)
        low, high = float(spec["source_lower_rad"]), float(spec["source_upper_rad"])
        if applied_source_target < low - 1e-10 or applied_source_target > high + 1e-10:
            raise ValueError(
                f"support control target for {joint_name} maps outside source limits: {applied_source_target} not in [{low}, {high}]"
            )
        data.ctrl[aid] = applied_ctrl
        rows[joint_name] = {
            "group": spec["group"],
            "reference_source_rad": float(references[joint_name]),
            "gravity_compensated_ctrl": applied_ctrl,
            "requested_ctrl_before_range_clamp": requested_ctrl,
            "ctrl_clamped": abs(applied_ctrl - requested_ctrl) > 1e-12,
            "applied_source_target_rad": applied_source_target,
            "target_inside_source_range": low - 1e-10 <= applied_source_target <= high + 1e-10,
            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
            "effort_bound_nm": float(spec["actuator_effort_bound_nm"]),
        }
    mujoco.mj_forward(model, data)
    return rows


def support_diagnostics(
    runtime: dict[str, object],
    reference_positions: dict[str, float] | None = None,
) -> dict[str, object]:
    model, data = runtime["model"], runtime["data"]
    support = runtime["fixed_body_support"]
    refs = dict(reference_positions or support["reference_source_positions"])
    specs = support["joint_specs"]
    by_group: dict[str, dict[str, object]] = {}
    effort_rows = []
    for joint_name, spec in specs.items():
        joint = runtime["joints"][joint_name]
        position = gate.source_position(model, data, joint)
        delta = position - float(refs[joint_name])
        group = str(spec["group"])
        group_row = by_group.setdefault(group, {"max_abs_motion_rad": 0.0, "worst_joint": None, "joint_motion_rad": {}})
        group_row["joint_motion_rad"][joint_name] = delta
        if abs(delta) > float(group_row["max_abs_motion_rad"]):
            group_row["max_abs_motion_rad"] = abs(delta)
            group_row["worst_joint"] = joint_name
        aid = int(spec["actuator_id"])
        generalized_effort = float(data.actuator_force[aid]) * float(model.actuator_gear[aid, 0])
        effort_rows.append({
            "joint": joint_name,
            "group": group,
            "generalized_effort_nm": generalized_effort,
            "bound_nm": float(spec["actuator_effort_bound_nm"]),
            "within_bound": abs(generalized_effort) <= float(spec["actuator_effort_bound_nm"]) + 1e-9,
            "source_position_rad": position,
            "source_lower_rad": float(spec["source_lower_rad"]),
            "source_upper_rad": float(spec["source_upper_rad"]),
        })

    hand_names = sorted(set(runtime["details"]["active_hand_joint_names"]) | set(runtime["transmission"].follower_joint_names))
    hand_targets = runtime["transmission"].active_target_sources()
    for relation in runtime["details"]["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        hand_targets[follower] = float(relation["multiplier"]) * hand_targets[driver] + float(relation["offset"])
    hand_states = {}
    for joint_name in hand_names:
        joint = runtime["joints"][joint_name]
        hand_states[joint_name] = {
            "source_position_rad": gate.source_position(model, data, joint),
            "source_velocity_rad_s": gate.source_velocity(data, joint),
            "source_acceleration_rad_s2": int(joint["axis_sign"]) * float(data.qacc[int(joint["dof"])]),
            "target_position_rad": float(hand_targets[joint_name]),
        }

    hand_dofs = [int(runtime["joints"][name]["dof"]) for name in hand_names]
    support_dofs = [int(spec["dof"]) for spec in specs.values()]
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    coupling_by_hand_joint = {}
    for joint_name in hand_names:
        dof = int(runtime["joints"][joint_name]["dof"])
        coupling_by_hand_joint[joint_name] = float(mass[dof, support_dofs] @ data.qacc[support_dofs]) if support_dofs else 0.0

    gate_state = gate.diagnostics(runtime)
    hand_actuator_efforts = gate_state["follower_efforts"] + gate_state["active_efforts"]
    constraints = {
        joint_name: float(data.qfrc_constraint[int(spec["dof"])])
        for joint_name, spec in specs.items()
    }
    return {
        "time_s": float(data.time),
        "finite": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
            and np.isfinite(data.ctrl).all()
            and np.isfinite(data.actuator_force).all()
        ),
        "source_position_limit_violations": gate_state["position_limit_violations"],
        "source_velocity_limit_violations": gate_state["velocity_limit_violations"],
        "self_contacts": gate_state["robot_self_contacts"],
        "max_self_penetration_m": float(gate_state["max_self_penetration_m"]),
        "source_mimic_error_max_rad": float(gate_state["source_relation_error_max_rad"]),
        "target_tracking_error_max_rad": float(gate_state["target_tracking_error_max_rad"]),
        "support_motion_by_group": by_group,
        "support_efforts": effort_rows,
        "support_max_abs_effort_nm": max((abs(float(row["generalized_effort_nm"])) for row in effort_rows), default=0.0),
        "support_effort_bound_violation": any(not row["within_bound"] for row in effort_rows),
        "support_constraint_reaction_nm": constraints,
        "support_max_abs_constraint_reaction_nm": max((abs(value) for value in constraints.values()), default=0.0),
        "hand_states": hand_states,
        "hand_max_abs_qvel_rad_s": max((abs(row["source_velocity_rad_s"]) for row in hand_states.values()), default=0.0),
        "hand_max_abs_qacc_rad_s2": max((abs(row["source_acceleration_rad_s2"]) for row in hand_states.values()), default=0.0),
        "hand_actuator_efforts": hand_actuator_efforts,
        "hand_max_abs_effort_nm": max((
            abs(float(row.get("actual_actuator_force_nm", row.get("force_nm", 0.0))))
            for row in hand_actuator_efforts
        ), default=0.0),
        "hand_effort_bound_violation": bool(gate_state["actuator_effort_violation"]),
        "whole_body_inertial_coupling_torque_by_hand_joint_nm": coupling_by_hand_joint,
        "whole_body_inertial_coupling_torque_max_abs_nm": max((abs(value) for value in coupling_by_hand_joint.values()), default=0.0),
        "raw_contact_count": int(data.ncon),
        "reported_contacts": gate_state["contacts"],
    }


def support_audit(runtime: dict[str, object]) -> dict[str, object]:
    support = runtime["fixed_body_support"]
    initial_controls = apply_posture_support(runtime)
    state = support_diagnostics(runtime)
    rows = []
    for joint_name, spec in support["joint_specs"].items():
        row = dict(spec)
        row["initial_source_position_rad"] = gate.source_position(runtime["model"], runtime["data"], runtime["joints"][joint_name])
        row["initial_source_velocity_rad_s"] = gate.source_velocity(runtime["data"], runtime["joints"][joint_name])
        row["initial_control"] = initial_controls[joint_name]
        rows.append(row)
    all_targets_valid = all(bool(row["target_inside_source_range"]) for row in initial_controls.values())
    all_bounded = all(bool(row["within_bound"]) for row in state["support_efforts"])
    passed = (
        support["base_fixed_to_world"]
        and len(rows) == 31
        and all_targets_valid
        and all_bounded
        and not state["source_position_limit_violations"]
        and not state["source_velocity_limit_violations"]
        and not state["self_contacts"]
        and state["max_self_penetration_m"] <= 1e-12
        and state["source_mimic_error_max_rad"] <= 1e-9
        and state["finite"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "stage": "1 - support classification and initialized audit",
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "timestep_s": float(runtime["model"].opt.timestep),
            "nq": int(runtime["model"].nq),
            "nv": int(runtime["model"].nv),
            "nu": int(runtime["model"].nu),
            "neq": int(runtime["model"].neq),
            "bottle_included": bool(runtime["details"]["bottle_included"]),
        },
        "classification": {
            "base_link": "fixed to world by model topology; no floating/base generalized coordinates",
            "legs": "12 bounded position actuators hold the accepted neutral pose",
            "torso_waist": "bounded position actuators hold the accepted neutral pose",
            "head": "bounded position actuators hold the accepted neutral pose",
            "both_arms_except_hands": "bounded position actuators hold the accepted manipulation pose",
            "hands": "the existing virtual-transmission controller is the only hand controller",
            "intentionally_free_robot_dofs": [],
            "bottle_free_joint": "omitted in stages 1-4; added only for the conditional bottle stage",
            "floor_and_table": "static world geoms; no generalized coordinates",
        },
        "support_method": {
            "actuator_model": "pre-existing source-bounded MuJoCo position actuators; no support gain changes",
            "gravity_support": "control target is shifted by qfrc_bias / (gear^2 * kp), then clamped to the corresponding source hard range",
            "source_joint_limits_modified": False,
            "per_step_qpos_writes": 0,
            "support_position_drift_gate_rad": PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD,
            "support_targets": rows,
        },
        "initial_state": state,
        "passed": passed,
    }


def stage2_failure_gate(state: dict[str, object]) -> str | None:
    if not state["finite"]:
        return "non-finite simulation state"
    if state["source_position_limit_violations"]:
        return "source hard position limit"
    if state["source_velocity_limit_violations"]:
        return "source velocity limit"
    if state["self_contacts"] or float(state["max_self_penetration_m"]) > 1e-12:
        return "robot self-contact/penetration"
    if state["support_effort_bound_violation"]:
        return "bounded support actuator effort"
    if state["hand_effort_bound_violation"]:
        return "bounded hand actuator effort"
    if float(state["source_mimic_error_max_rad"]) > HAND_ABORT_RAD:
        return "hand mimic manipulation abort ceiling 0.010 rad"
    if float(state["target_tracking_error_max_rad"]) > HAND_ABORT_RAD:
        return "hand target tracking manipulation abort ceiling 0.010 rad"
    largest = max((float(row["max_abs_motion_rad"]) for row in state["support_motion_by_group"].values()), default=0.0)
    if largest > PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD:
        return "supported non-hand posture drift"
    return None


def diagnose_first_failure(runtime: dict[str, object], state: dict[str, object], gate_name: str) -> dict[str, object]:
    model, data = runtime["model"], runtime["data"]
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    worst_hand = max(
        state["hand_states"],
        key=lambda joint: abs(float(state["hand_states"][joint]["source_acceleration_rad_s2"])),
    )
    dof = int(runtime["joints"][worst_hand]["dof"])
    return {
        "diagnosis_type": "single read-only decomposition at the first failing state; no replay or parameter change",
        "failed_gate": gate_name,
        "time_s": float(data.time),
        "worst_hand_acceleration_joint": worst_hand,
        "worst_hand_qacc_rad_s2": float(data.qacc[dof]),
        "hand_qfrc_bias_nm": float(data.qfrc_bias[dof]),
        "hand_qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
        "hand_qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
        "hand_mass_diagonal": float(mass[dof, dof]),
        "whole_body_coupling_torque_by_hand_joint_nm": state["whole_body_inertial_coupling_torque_by_hand_joint_nm"],
        "support_max_abs_effort_nm": state["support_max_abs_effort_nm"],
        "support_max_abs_constraint_reaction_nm": state["support_max_abs_constraint_reaction_nm"],
        "source_position_limit_violations": state["source_position_limit_violations"],
        "source_velocity_limit_violations": state["source_velocity_limit_violations"],
        "self_contacts": state["self_contacts"],
        "source_mimic_error_max_rad": state["source_mimic_error_max_rad"],
    }


def run_stage2(runtime: dict[str, object], trace_path: Path) -> dict[str, object]:
    model, data = runtime["model"], runtime["data"]
    open_targets = dict(runtime["initial_active"])
    trace_rows = 0
    first_failure = None
    maximum = {
        "support_effort_nm": 0.0,
        "support_constraint_reaction_nm": 0.0,
        "hand_effort_nm": 0.0,
        "hand_qvel_rad_s": 0.0,
        "hand_qacc_rad_s2": 0.0,
        "mimic_error_rad": 0.0,
        "target_tracking_error_rad": 0.0,
        "whole_body_inertial_coupling_torque_nm": 0.0,
        "support_motion_by_group_rad": {},
    }
    with trace_path.open("w", encoding="utf-8") as trace:
        for step in range(1, 1001):
            support_controls = apply_posture_support(runtime)
            runtime["transmission"].command_active_sources(open_targets)
            runtime["transmission"].write_internal_controls()
            mujoco.mj_step(model, data)
            state = support_diagnostics(runtime)
            for key, state_key in (
                ("support_effort_nm", "support_max_abs_effort_nm"),
                ("support_constraint_reaction_nm", "support_max_abs_constraint_reaction_nm"),
                ("hand_effort_nm", "hand_max_abs_effort_nm"),
                ("hand_qvel_rad_s", "hand_max_abs_qvel_rad_s"),
                ("hand_qacc_rad_s2", "hand_max_abs_qacc_rad_s2"),
                ("mimic_error_rad", "source_mimic_error_max_rad"),
                ("target_tracking_error_rad", "target_tracking_error_max_rad"),
                ("whole_body_inertial_coupling_torque_nm", "whole_body_inertial_coupling_torque_max_abs_nm"),
            ):
                maximum[key] = max(float(maximum[key]), float(state[state_key]))
            for group, group_state in state["support_motion_by_group"].items():
                maximum["support_motion_by_group_rad"][group] = max(
                    float(maximum["support_motion_by_group_rad"].get(group, 0.0)),
                    float(group_state["max_abs_motion_rad"]),
                )
            failure = stage2_failure_gate(state)
            row = {
                "step": step,
                "simulation_time_s": float(data.time),
                "support_controls": support_controls,
                "state": state,
                "active_rollout_follower_qpos_writes": 0,
            }
            trace.write(json.dumps(row, separators=(",", ":")) + "\n")
            trace.flush()
            trace_rows += 1
            if failure:
                first_failure = {"step": step, "time_s": float(data.time), "gate": failure, "state": state}
                break

    passed = first_failure is None and trace_rows == 1000
    result = {
        "status": "PASS" if passed else "FAIL",
        "stage": "2 - supported full-robot operational OPEN hold",
        "requested_duration_s": 2.0,
        "requested_steps": 1000,
        "completed_steps": trace_rows,
        "actual_duration_s": float(data.time),
        "first_failure": first_failure,
        "maximum_observed": maximum,
        "support_drift_gate_rad": PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD,
        "source_joint_limits_modified": False,
        "accepted_hand_controller_modified": False,
        "active_rollout_qpos_writes": 0,
        "trace_path": str(trace_path),
        "passed": passed,
    }
    if first_failure is not None:
        result["first_failure_diagnosis"] = diagnose_first_failure(runtime, first_failure["state"], first_failure["gate"])
    return result


def run_supported_trial(
    label: str,
    phases: list[tuple[str, int, dict[str, float], dict[str, float]]],
    trace_file,
    *,
    fixed_probe: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime = gate.make_runtime(fixed_probe=fixed_probe, include_bottle=False)
    initialize_support(runtime)
    model, data = runtime["model"], runtime["data"]
    initial_controls = apply_posture_support(runtime)
    initial = support_diagnostics(runtime)
    result: dict[str, object] = {
        "label": label,
        "initial": initial,
        "completed_steps": 0,
        "max_source_mimic_error_rad": 0.0,
        "max_hand_target_error_rad": 0.0,
        "max_abs_hand_qvel_rad_s": 0.0,
        "max_abs_hand_qacc_rad_s2": 0.0,
        "max_abs_hand_effort_nm": 0.0,
        "max_abs_support_effort_nm": 0.0,
        "max_abs_support_reaction_nm": 0.0,
        "max_support_motion_by_group_rad": {},
        "max_whole_body_coupling_torque_nm": 0.0,
        "diagnostic_0p003_samples": 0,
        "abort_0p010_samples": 0,
        "position_limit_violation_samples": 0,
        "velocity_limit_violation_samples": 0,
        "max_self_penetration_m": 0.0,
        "support_effort_bound_violation": False,
        "hand_effort_bound_violation": False,
        "probe_contact_steps": 0,
        "probe_contact_pairs": {},
        "max_probe_force_n": 0.0,
        "phases": {},
        "active_rollout_follower_qpos_writes": 0,
        "fixed_probe": fixed_probe,
        "initial_support_controls": initial_controls,
    }
    failure = None
    if initial["source_position_limit_violations"]:
        failure = "initial source position limit"
    elif initial["source_velocity_limit_violations"]:
        failure = "initial source velocity limit"
    elif initial["self_contacts"] or initial["max_self_penetration_m"] > 1e-12:
        failure = "initial self-contact/penetration"
    elif initial["source_mimic_error_max_rad"] > 1e-9:
        failure = "initial mimic geometry"
    elif initial["support_effort_bound_violation"] or initial["hand_effort_bound_violation"]:
        failure = "initial bounded actuator effort"

    active_targets = runtime["transmission"].active_target_sources()
    support_targets = dict(runtime["fixed_body_support"]["reference_source_positions"])
    step_index = 0
    peak_hand_speed = 0.0
    final_hold_samples: list[dict[str, object]] = []
    for phase, duration, requested_hand_targets, requested_support_targets in phases:
        if failure:
            break
        phase_start_hand = dict(active_targets)
        phase_goal_hand = dict(phase_start_hand)
        phase_goal_hand.update(requested_hand_targets)
        phase_start_support = dict(support_targets)
        phase_goal_support = dict(phase_start_support)
        phase_goal_support.update(requested_support_targets)
        phase_stats = {
            "steps": duration,
            "max_abs_hand_qvel_rad_s": 0.0,
            "max_abs_hand_qacc_rad_s2": 0.0,
            "max_mimic_error_rad": 0.0,
            "max_target_error_rad": 0.0,
            "max_support_effort_nm": 0.0,
            "max_support_motion_by_group_rad": {},
            "final_state": None,
        }
        for k in range(duration):
            alpha = (k + 1) / duration
            active_targets = gate.smooth_blend(phase_start_hand, phase_goal_hand, alpha)
            active_targets = {
                joint: min(
                    runtime["transmission"].operational_limits(joint)[1],
                    max(runtime["transmission"].operational_limits(joint)[0], value),
                )
                for joint, value in active_targets.items()
            }
            support_targets = gate.smooth_blend(phase_start_support, phase_goal_support, alpha)
            support_controls = apply_posture_support(runtime, support_targets)
            runtime["transmission"].command_active_sources(active_targets)
            runtime["transmission"].write_internal_controls()
            mujoco.mj_step(model, data)
            state = support_diagnostics(runtime, support_targets)
            step_index += 1
            hand_speed = float(state["hand_max_abs_qvel_rad_s"])
            peak_hand_speed = max(peak_hand_speed, hand_speed)
            phase_stats["max_abs_hand_qvel_rad_s"] = max(float(phase_stats["max_abs_hand_qvel_rad_s"]), hand_speed)
            phase_stats["max_abs_hand_qacc_rad_s2"] = max(float(phase_stats["max_abs_hand_qacc_rad_s2"]), float(state["hand_max_abs_qacc_rad_s2"]))
            phase_stats["max_mimic_error_rad"] = max(float(phase_stats["max_mimic_error_rad"]), float(state["source_mimic_error_max_rad"]))
            phase_stats["max_target_error_rad"] = max(float(phase_stats["max_target_error_rad"]), float(state["target_tracking_error_max_rad"]))
            phase_stats["max_support_effort_nm"] = max(float(phase_stats["max_support_effort_nm"]), float(state["support_max_abs_effort_nm"]))
            for group, group_state in state["support_motion_by_group"].items():
                phase_stats["max_support_motion_by_group_rad"][group] = max(
                    float(phase_stats["max_support_motion_by_group_rad"].get(group, 0.0)),
                    float(group_state["max_abs_motion_rad"]),
                )
            result["max_source_mimic_error_rad"] = max(float(result["max_source_mimic_error_rad"]), float(state["source_mimic_error_max_rad"]))
            result["max_hand_target_error_rad"] = max(float(result["max_hand_target_error_rad"]), float(state["target_tracking_error_max_rad"]))
            result["max_abs_hand_qvel_rad_s"] = max(float(result["max_abs_hand_qvel_rad_s"]), hand_speed)
            result["max_abs_hand_qacc_rad_s2"] = max(float(result["max_abs_hand_qacc_rad_s2"]), float(state["hand_max_abs_qacc_rad_s2"]))
            result["max_abs_hand_effort_nm"] = max(float(result["max_abs_hand_effort_nm"]), float(state["hand_max_abs_effort_nm"]))
            result["max_abs_support_effort_nm"] = max(float(result["max_abs_support_effort_nm"]), float(state["support_max_abs_effort_nm"]))
            result["max_abs_support_reaction_nm"] = max(float(result["max_abs_support_reaction_nm"]), float(state["support_max_abs_constraint_reaction_nm"]))
            result["max_whole_body_coupling_torque_nm"] = max(float(result["max_whole_body_coupling_torque_nm"]), float(state["whole_body_inertial_coupling_torque_max_abs_nm"]))
            result["position_limit_violation_samples"] += len(state["source_position_limit_violations"])
            result["velocity_limit_violation_samples"] += len(state["source_velocity_limit_violations"])
            result["support_effort_bound_violation"] = bool(result["support_effort_bound_violation"]) or bool(state["support_effort_bound_violation"])
            result["hand_effort_bound_violation"] = bool(result["hand_effort_bound_violation"]) or bool(state["hand_effort_bound_violation"])
            result["max_self_penetration_m"] = max(float(result["max_self_penetration_m"]), float(state["max_self_penetration_m"]))
            if state["source_mimic_error_max_rad"] > task.MIMIC_RELATION_TOLERANCE_RAD:
                result["diagnostic_0p003_samples"] += 1
            if max(float(state["source_mimic_error_max_rad"]), float(state["target_tracking_error_max_rad"])) > HAND_ABORT_RAD:
                result["abort_0p010_samples"] += 1
            probe_contacts = [row for row in state["reported_contacts"] if gate.PROBE_GEOM in row["geom_pair"]]
            if probe_contacts:
                result["probe_contact_steps"] += 1
                for contact in probe_contacts:
                    pair = " + ".join(sorted(contact["body_pair"]))
                    result["probe_contact_pairs"][pair] = int(result["probe_contact_pairs"].get(pair, 0)) + 1
                    result["max_probe_force_n"] = max(
                        float(result["max_probe_force_n"]),
                        float(np.linalg.norm(np.asarray(contact["force_components"][:3]))),
                    )
            phase_stats["final_state"] = {
                "time_s": state["time_s"],
                "max_hand_qvel_rad_s": state["hand_max_abs_qvel_rad_s"],
                "max_hand_qacc_rad_s2": state["hand_max_abs_qacc_rad_s2"],
                "target_error_max_rad": state["target_tracking_error_max_rad"],
                "mimic_error_max_rad": state["source_mimic_error_max_rad"],
            }
            row = {
                "trial": label,
                "step": step_index,
                "phase": phase,
                "simulation_time_s": float(data.time),
                "active_hand_targets_rad": active_targets,
                "support_targets_source_rad": support_targets,
                "support_controls": support_controls,
                "state": state,
                "active_rollout_follower_qpos_writes": 0,
            }
            trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            if not state["finite"]:
                failure = "non-finite simulation state"
            elif state["source_position_limit_violations"]:
                failure = "source hard position limit"
            elif state["source_velocity_limit_violations"]:
                failure = "source velocity limit"
            elif state["self_contacts"] or state["max_self_penetration_m"] > 1e-12:
                failure = "robot self-contact/penetration"
            elif state["support_effort_bound_violation"]:
                failure = "bounded support actuator effort"
            elif state["hand_effort_bound_violation"]:
                failure = "bounded hand actuator effort"
            elif max(float(state["source_mimic_error_max_rad"]), float(state["target_tracking_error_max_rad"])) > HAND_ABORT_RAD:
                failure = "hand tracking exceeded 0.010 rad abort ceiling"
            elif max((float(group["max_abs_motion_rad"]) for group in state["support_motion_by_group"].values()), default=0.0) > PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD:
                failure = "supported non-hand posture tracking"
            if phase == "final_open_hold" and k >= duration - min(25, duration):
                final_hold_samples.append(state)
            if failure:
                result["first_failure"] = {"phase": phase, "step": step_index, "time_s": float(data.time), "gate": failure, "state": state}
                break
        result["phases"][phase] = phase_stats
        result["completed_steps"] = step_index
        trace_file.flush()
        if failure:
            break

    result["passed"] = failure is None
    result["first_failed_gate"] = failure
    result["damped_response"] = None
    if label.startswith("stage3_"):
        final_speed = max((float(row["hand_max_abs_qvel_rad_s"]) for row in final_hold_samples), default=0.0)
        final_error = max((float(row["target_tracking_error_max_rad"]) for row in final_hold_samples), default=0.0)
        response_pass = final_speed <= max(1e-9, 0.10 * peak_hand_speed) and final_error <= HAND_ABORT_RAD
        result["damped_response"] = {
            "passed": response_pass,
            "peak_hand_qvel_rad_s": peak_hand_speed,
            "final_25_step_max_hand_qvel_rad_s": final_speed,
            "final_25_step_max_target_error_rad": final_error,
            "velocity_decay_ratio_limit": 0.10,
            "tracking_error_ceiling_rad": HAND_ABORT_RAD,
        }
        if result["passed"] and not response_pass:
            result["passed"] = False
            result["first_failed_gate"] = "representative hand response did not damp to the open hold"
    if fixed_probe:
        result["stable_external_contact_pass"] = result["probe_contact_steps"] > 0 and result["max_probe_force_n"] > 0.0
        if result["passed"] and not result["stable_external_contact_pass"]:
            result["passed"] = False
            result["first_failed_gate"] = "fixed-object fingertip contact was not established"
    result["first_failure_diagnosis"] = (
        diagnose_first_failure(runtime, result["first_failure"]["state"], str(failure))
        if failure and result.get("first_failure") else None
    )
    result["runtime"] = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "neq": int(model.neq),
        "bottle_included": bool(runtime["details"]["bottle_included"]),
        "initialization_qpos_writes_before_t0": int(runtime["initialization_qpos_write_count"]),
        "active_rollout_follower_qpos_writes": 0,
    }
    return result, runtime


def representative_hand_trials(trace_path: Path) -> list[dict[str, object]]:
    checks = [
        ("R_index_dip_follower", "R_index_pip_joint", 0.01),
        ("R_thumb_pip_follower", "R_thumb_mcp_joint", 0.01),
        ("active_finger_driver", "R_middle_pip_joint", 0.01),
        ("active_thumb_driver", "R_thumb_roll_joint", 0.01),
    ]
    results = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for label, joint_name, delta in checks:
            runtime = gate.make_runtime(include_bottle=False)
            initialize_support(runtime)
            start = dict(runtime["initial_active"])
            low, high = runtime["transmission"].operational_limits(joint_name)
            target = min(high, max(low, start[joint_name] + delta))
            if abs(target - start[joint_name]) < 1e-10:
                target = min(high, max(low, start[joint_name] - delta))
            goal = dict(start)
            goal[joint_name] = target
            one_trial = [
                ("supported_open_hold", 100, start, {}),
                ("source_step", 100, goal, {}),
                ("source_step_hold", 100, goal, {}),
                ("source_return", 100, start, {}),
                ("final_open_hold", 100, start, {}),
            ]
            result, _ = run_supported_trial(f"stage3_{label}", one_trial, trace)
            result["tested_coordinate"] = joint_name
            result["commanded_source_delta_rad"] = target - start[joint_name]
            results.append(result)
            if not result["passed"]:
                break
    return results


def run_stage4(trace_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with trace_path.open("w", encoding="utf-8") as trace:
        runtime = gate.make_runtime(include_bottle=False)
        initialize_support(runtime)
        start = dict(runtime["initial_active"])
        closed = {**start, **gate.feasible_driver_targets(runtime)}
        phases = [
            ("open_hold", 200, start, {}),
            ("close", 600, closed, {}),
            ("partial", 400, gate.smooth_blend(start, closed, 0.5), {}),
            ("closed_hold", 300, closed, {}),
            ("reopen", 600, start, {}),
        ]
        result, _ = run_supported_trial("stage4_full_hand_open_close", phases, trace)
        result["gate_name"] = "OPEN HOLD -> CLOSE -> PARTIAL -> CLOSED HOLD -> REOPEN"
        results.append(result)
        if not result["passed"]:
            return results

        runtime = gate.make_runtime(include_bottle=False)
        initialize_support(runtime)
        arm_names = task.ARM
        arm_ids = [int(runtime["joints"][joint]["joint_id"]) for joint in arm_names]
        site_id = int(runtime["details"]["site_id"])
        arm_route = task.ik_plan(
            runtime["model"],
            runtime["data"].qpos.copy(),
            site_id,
            arm_ids,
            [task.APPROACH_PALM_POS.copy(), task.PREGRASP_PALM_POS.copy()],
            task.PALM_TARGET_ROTATION,
        )
        arm_phases = []
        for index, raw_target in enumerate(arm_route):
            goal = {}
            for i, joint_name in enumerate(arm_names):
                joint = runtime["joints"][joint_name]
                qpos = int(joint["qpos"])
                goal[joint_name] = float(joint["reference"]) + int(joint["axis_sign"]) * (
                    float(raw_target[i]) - float(runtime["model"].qpos0[qpos])
                )
            arm_phases.append((f"right_arm_to_approach_{index + 1}", 600, start, goal))
        result, _ = run_supported_trial("stage4_supported_right_arm_motion", arm_phases, trace)
        result["gate_name"] = "supported right-arm IK motion with the left arm and body support held"
        result["ik_route_raw_joint_targets"] = np.asarray(arm_route).tolist()
        results.append(result)
        if not result["passed"]:
            return results

        relation_drivers = sorted({str(row["driver_joint"]) for row in runtime["details"]["mimic_relations"]})
        for driver in relation_drivers:
            for direction in (-1.0, 1.0):
                runtime = gate.make_runtime(include_bottle=False)
                initialize_support(runtime)
                start = dict(runtime["initial_active"])
                low, high = runtime["transmission"].operational_limits(driver)
                target = min(high, max(low, start[driver] + direction * 0.01))
                goal = dict(start)
                goal[driver] = target
                phases = [
                    ("supported_open_hold", 25, start, {}),
                    ("driver_perturb", 50, goal, {}),
                    ("driver_perturb_hold", 50, goal, {}),
                    ("driver_return", 50, start, {}),
                    ("final_open_hold", 25, start, {}),
                ]
                result, _ = run_supported_trial(f"stage4_driver_perturb_{driver}_{direction:+.0f}x0p01", phases, trace)
                result["gate_name"] = "valid single source-driver perturbation and return"
                result["driver_joint"] = driver
                result["requested_delta_rad"] = target - start[driver]
                results.append(result)
                if not result["passed"]:
                    return results

        runtime = gate.make_runtime(fixed_probe=True, include_bottle=False)
        initialize_support(runtime)
        start = dict(runtime["initial_active"])
        right_close = {**start, **{
            joint: value
            for joint, value in gate.feasible_driver_targets(runtime).items()
            if joint.startswith("R_")
        }}
        phases = [
            ("close_to_fixed_probe", 900, right_close, {}),
            ("fixed_probe_hold", 500, right_close, {}),
            ("open_from_fixed_probe", 700, start, {}),
        ]
        result, _ = run_supported_trial("stage4_fixed_object_fingertip_contact", phases, trace, fixed_probe=True)
        result["gate_name"] = "supported fixed-object fingertip contact"
        results.append(result)
    return results


def runtime_identity() -> dict[str, object]:
    root = task.ROOT
    vendor_head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    urdf_hash = hashlib.sha256(task.URDF.read_bytes()).hexdigest()
    mujoco_dir = Path(mujoco.__file__).resolve().parent
    libraries = sorted(mujoco_dir.glob("libmujoco*"))
    native_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "mujoco_native_library_hashes": native_hashes,
        "robot_sim_head": subprocess.check_output(["git", "-C", str(task.SIM_REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_expected_pin": task.SOURCE_PIN,
        "vendor_observed_head": vendor_head,
        "vendor_pin_matches": vendor_head == task.SOURCE_PIN,
        "urdf_path": str(task.URDF),
        "urdf_sha256": urdf_hash,
        "timestep_s": task.DT,
        "exact_command": (
            "cd /tmp/robotsim-issue46-virtual-transmission-resume-20261007 && "
            "env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 "
            "ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
            "virtual-transmission-20261007/supported-fixed-body-recovery-02 "
            "PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u "
            "scripts/research/issue46_virtual_transmission_supported_recovery.py"
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--continue-after-stage2" in sys.argv:
        identity = json.loads((OUT / "runtime_identity.json").read_text(encoding="utf-8"))
        stage1 = json.loads((OUT / "stage1_support_audit.json").read_text(encoding="utf-8"))
        stage2 = json.loads((OUT / "stage2_supported_open_hold_result.json").read_text(encoding="utf-8"))
        if not stage1.get("passed") or not stage2.get("passed"):
            raise RuntimeError("--continue-after-stage2 requires preserved passing Stage 1 and Stage 2 evidence")
        if not identity.get("vendor_pin_matches") or identity.get("vendor_observed_head") != task.SOURCE_PIN:
            raise RuntimeError("preserved Stage 1/2 evidence does not match the pinned vendor source")
    else:
        identity = runtime_identity()
        write_json(OUT / "runtime_identity.json", identity)
        if not identity["vendor_pin_matches"]:
            write_json(OUT / "stage1_support_audit.json", {"status": "FAIL", "reason": "vendor pin mismatch", "runtime_identity": identity})
            return 2

        runtime = gate.make_runtime(include_bottle=False)
        initialize_support(runtime)
        stage1 = support_audit(runtime)
        stage1["runtime_identity"] = identity
        write_json(OUT / "stage1_support_audit.json", stage1)
        if not stage1["passed"]:
            final = {"status": "FAIL", "stop_stage": "1", "first_failure": stage1}
            write_json(OUT / "supported_recovery_result.json", final)
            print(json.dumps(final, indent=2))
            return 2

        runtime = gate.make_runtime(include_bottle=False)
        initialize_support(runtime)
        stage2 = run_stage2(runtime, OUT / "stage2_supported_open_hold_trace.jsonl")
        stage2["support_audit_path"] = str(OUT / "stage1_support_audit.json")
        write_json(OUT / "stage2_supported_open_hold_result.json", stage2)
        if not stage2["passed"]:
            final = {
                "status": stage2["status"],
                "stop_stage": "2",
                "first_failure": stage2.get("first_failure"),
                "stage1": stage1,
                "stage2": stage2,
                "runtime_identity": identity,
                "conditional_stages_not_run": ["3 representative hand steps", "4 full-hand prevalidation", "5 bottle contact hold"],
            }
            write_json(OUT / "supported_recovery_result.json", final)
            print(json.dumps(final, indent=2))
            return 2

    stage3 = representative_hand_trials(OUT / "stage3_representative_hand_trace.jsonl")
    stage3_passed = len(stage3) == 4 and all(bool(row["passed"]) for row in stage3)
    write_json(OUT / "stage3_representative_hand_result.json", {
        "status": "PASS" if stage3_passed else "FAIL",
        "passed": stage3_passed,
        "required_cases": ["R_index_dip follower", "R_thumb_pip follower", "representative active finger", "active thumb"],
        "results": stage3,
    })
    if not stage3_passed:
        final = {
            "status": "FAIL",
            "stop_stage": "3",
            "first_failure": next((row.get("first_failure") or {"gate": row.get("first_failed_gate"), "label": row["label"]} for row in stage3 if not row["passed"]), None),
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "runtime_identity": identity,
            "conditional_stages_not_run": ["4 full-hand prevalidation", "5 bottle contact hold"],
        }
        write_json(OUT / "supported_recovery_result.json", final)
        print(json.dumps({"status": final["status"], "stop_stage": final["stop_stage"], "first_failure": final["first_failure"], "evidence_dir": str(OUT)}, indent=2))
        return 2

    stage4 = run_stage4(OUT / "stage4_full_hand_prevalidation_trace.jsonl")
    stage4_passed = len(stage4) >= 4 and all(bool(row["passed"]) for row in stage4)
    write_json(OUT / "stage4_full_hand_prevalidation_result.json", {
        "status": "PASS" if stage4_passed else "FAIL",
        "passed": stage4_passed,
        "required_gates": ["full-hand open/close/partial/reopen", "supported right-arm motion", "source-valid driver perturbations", "fixed-object fingertip contact"],
        "results": stage4,
    })
    final = {
        "status": "PASS" if stage4_passed else "FAIL",
        "stop_stage": "4_complete" if stage4_passed else "4",
        "first_failure": next((row.get("first_failure") or {"gate": row.get("first_failed_gate"), "label": row["label"]} for row in stage4 if not row["passed"]), None),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "runtime_identity": identity,
        "conditional_stages_not_run": ["5 bottle contact hold"] if not stage4_passed else [],
    }
    write_json(OUT / "supported_recovery_result.json", final)
    print(json.dumps({
        "status": final["status"],
        "stop_stage": final["stop_stage"],
        "first_failure": final["first_failure"],
        "stage3": [{"label": row["label"], "passed": row["passed"], "max_mimic_error_rad": row["max_source_mimic_error_rad"], "max_qvel": row["max_abs_hand_qvel_rad_s"], "max_qacc": row["max_abs_hand_qacc_rad_s2"], "damped": row["damped_response"]} for row in stage3],
        "stage4": [{"label": row["label"], "passed": row["passed"], "first_failed_gate": row["first_failed_gate"], "steps": row["completed_steps"], "max_mimic_error_rad": row["max_source_mimic_error_rad"]} for row in stage4],
        "evidence_dir": str(OUT),
    }, indent=2))
    return 0 if stage4_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
