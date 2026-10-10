from __future__ import annotations

import hashlib
import json
import math
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
REFERENCE_DURATION_S = 1.0
REFERENCE_DURATION_STEPS = round(REFERENCE_DURATION_S / task.DT)
FINAL_MARGIN_ROUNDING_RAD = 1e-4
FINAL_OPERATIONAL_MARGIN_INPUTS = Path(os.environ.get(
    "ISSUE46_PREVIOUS_EVIDENCE_DIR",
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/smooth-reference-recovery-03",
))


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


def minimum_jerk_blend(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    u = min(1.0, max(0.0, float(alpha)))
    if u == 0.0:
        return dict(start)
    if u == 1.0:
        return dict(end)
    s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return {key: start[key] + s * (end[key] - start[key]) for key in start}


def cubic_smooth_blend_exact(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    u = min(1.0, max(0.0, float(alpha)))
    if u == 0.0:
        return dict(start)
    if u == 1.0:
        return dict(end)
    return gate.smooth_blend(start, end, u)


def controller_hand_records(runtime: dict[str, object]) -> dict[str, dict[str, object]]:
    transmission = runtime["transmission"]
    records: dict[str, dict[str, object]] = {}
    for joint_name, output in transmission.active_measure().items():
        records[joint_name] = {
            "target_position_source_rad": float(output["target_source_rad"]),
            "target_velocity_source_rad_s": float(output["target_velocity_source_rad_s"]),
            "position_error_source_rad": float(output["target_source_rad"]) - float(output["position_source_rad"]),
            "velocity_error_source_rad_s": float(output["target_velocity_source_rad_s"]) - float(output["velocity_source_rad_s"]),
            "actual_velocity_source_rad_s": float(output["velocity_source_rad_s"]),
            "pd_torque_nm": float(output["pd_torque_nm"]),
            "bias_feedforward_torque_nm": float(output["bias_feedforward_nm"]),
            "dynamic_coupling_feedforward_nm": float(output.get("dynamic_coupling_feedforward_nm", 0.0)),
            "dynamic_coupling_by_driver_nm": dict(output.get("dynamic_coupling_by_driver_nm", {})),
            "requested_torque_nm": float(output["requested_torque_nm"]),
            "commanded_torque_after_clipping_nm": float(output["commanded_torque_nm"]),
            "effort_limit_nm": float(output["effort_limit_nm"]),
        }
    for joint_name, output in transmission.measure().items():
        records[joint_name] = {
            "target_position_source_rad": float(output["follower_target_source_rad"]),
            "target_velocity_source_rad_s": float(output["follower_target_velocity_source_rad_s"]),
            "position_error_source_rad": float(output["follower_target_source_rad"]) - float(output["follower_position_source_rad"]),
            "velocity_error_source_rad_s": float(output["follower_target_velocity_source_rad_s"]) - float(output["follower_velocity_source_rad_s"]),
            "actual_velocity_source_rad_s": float(output["follower_velocity_source_rad_s"]),
            "pd_torque_nm": float(output["pd_torque_nm"]),
            "bias_feedforward_torque_nm": float(output["bias_feedforward_nm"]),
            "dynamic_coupling_feedforward_nm": float(output["dynamic_coupling_feedforward_nm"]),
            "dynamic_coupling_by_driver_nm": dict(output["dynamic_coupling_by_driver_nm"]),
            "requested_torque_nm": float(output["requested_torque_nm"]),
            "commanded_torque_after_clipping_nm": float(output["commanded_torque_nm"]),
            "effort_limit_nm": float(output["effort_limit_nm"]),
        }
    return records


def detailed_joint_step(
    runtime: dict[str, object],
    controller_records: dict[str, dict[str, object]],
    joint_names: tuple[str, ...],
    target_acceleration: dict[str, float],
) -> dict[str, dict[str, object]]:
    model, data, transmission = runtime["model"], runtime["data"], runtime["transmission"]
    relation_by_joint = {}
    for relation in runtime["details"]["mimic_relations"]:
        relation_by_joint[str(relation["driver_joint"])] = relation
        relation_by_joint[str(relation["follower_joint"])] = relation
    result: dict[str, dict[str, object]] = {}
    for joint_name in joint_names:
        joint = runtime["joints"][joint_name]
        target = controller_records[joint_name]
        current_position = transmission.source_position(joint_name)
        current_velocity = transmission.source_velocity(joint_name)
        lower = float(joint["lower"])
        upper = float(joint["upper"])
        result[joint_name] = {
            "source_axis_sign": int(joint["axis_sign"]),
            "source_limits_rad": {"lower": lower, "upper": upper},
            "source_velocity_limit_rad_s": float(joint["velocity_limit"]),
            "operational_soft_limits_rad": dict(zip(("lower", "upper"), transmission.operational_limits(joint_name))),
            "target_position_source_rad": float(target["target_position_source_rad"]),
            "target_velocity_source_rad_s": float(target["target_velocity_source_rad_s"]),
            "target_acceleration_source_rad_s2": float(target_acceleration[joint_name]),
            "pre_step": {
                "raw_qpos": float(data.qpos[int(joint["qpos"])]),
                "raw_qvel_rad_s": float(data.qvel[int(joint["dof"])]),
                "source_qpos_rad": current_position,
                "source_qvel_rad_s": current_velocity,
                "source_qacc_rad_s2": int(joint["axis_sign"]) * float(data.qacc[int(joint["dof"])]),
                "position_error_target_minus_actual_rad": float(target["position_error_source_rad"]),
                "velocity_error_target_minus_actual_rad_s": float(target["velocity_error_source_rad_s"]),
                "pd_torque_nm": float(target["pd_torque_nm"]),
                "bias_feedforward_torque_nm": float(target["bias_feedforward_torque_nm"]),
                "requested_torque_before_clipping_nm": float(target["requested_torque_nm"]),
                "commanded_torque_after_clipping_nm": float(target["commanded_torque_after_clipping_nm"]),
                "dynamic_coupling_feedforward_nm": float(target["dynamic_coupling_feedforward_nm"]),
                "dynamic_coupling_by_driver_nm": target["dynamic_coupling_by_driver_nm"],
                "effort_limit_nm": float(target["effort_limit_nm"]),
                "qfrc_bias_raw_nm": float(data.qfrc_bias[int(joint["dof"])]),
                "qfrc_bias_source_nm": int(joint["axis_sign"]) * float(data.qfrc_bias[int(joint["dof"])]),
                "source_lower_limit_margin_rad": current_position - lower,
            },
            "relation": None,
        }
        relation = relation_by_joint.get(joint_name)
        if relation is not None:
            driver = str(relation["driver_joint"])
            follower = str(relation["follower_joint"])
            result[joint_name]["relation"] = {
                "driver_joint": driver,
                "follower_joint": follower,
                "multiplier": float(relation["multiplier"]),
                "offset_rad": float(relation["offset"]),
                "source_residual_rad": (
                    transmission.source_position(follower)
                    - float(relation["multiplier"]) * transmission.source_position(driver)
                    - float(relation["offset"])
                ),
            }
    return result


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
    reference_blend=minimum_jerk_blend,
    detailed_joint_names: tuple[str, ...] = (),
    operational_target_margin_rad: float | None = None,
    controller_profile_override: dict[str, dict[str, float]] | None = None,
    reject_sustained_follower_reverse: bool = False,
    check_final_settle: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime = gate.make_runtime(
        fixed_probe=fixed_probe,
        include_bottle=False,
        operational_target_margin_rad=operational_target_margin_rad,
        controller_profile_override=controller_profile_override,
    )
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
        "max_abs_hand_target_velocity_rad_s": 0.0,
        "max_abs_hand_target_acceleration_rad_s2": 0.0,
        "max_abs_hand_effort_nm": 0.0,
        "max_abs_support_effort_nm": 0.0,
        "max_abs_support_reaction_nm": 0.0,
        "max_support_motion_by_group_rad": {},
        "max_whole_body_coupling_torque_nm": 0.0,
        "max_abs_dynamic_coupling_feedforward_nm": 0.0,
        "max_dynamic_coupling_by_driver_nm": {},
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
        "operational_target_margin_rad": runtime["operational_target_margin_rad"],
        "controller_gain_profile_frozen": controller_profile_override is not None,
        "follower_reverse_samples": 0,
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
    previous_target_velocity = {
        joint_name: 0.0
        for joint_name in runtime["transmission"].active_joint_names + runtime["transmission"].follower_joint_names
    }
    final_hold_samples: list[dict[str, object]] = []
    follower_target_direction: dict[str, int] = {}
    follower_direction_steps: dict[str, int] = {}
    follower_direction_hold_steps = max(1, round(0.05 / float(model.opt.timestep)))
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
            "max_abs_hand_target_velocity_rad_s": 0.0,
            "max_abs_hand_target_acceleration_rad_s2": 0.0,
            "max_mimic_error_rad": 0.0,
            "max_target_error_rad": 0.0,
            "max_support_effort_nm": 0.0,
            "max_abs_dynamic_coupling_feedforward_nm": 0.0,
            "max_support_motion_by_group_rad": {},
            "final_state": None,
        }
        for k in range(duration):
            alpha = (k + 1) / duration
            active_targets = reference_blend(phase_start_hand, phase_goal_hand, alpha)
            support_targets = reference_blend(phase_start_support, phase_goal_support, alpha)
            support_controls = apply_posture_support(runtime, support_targets)
            runtime["transmission"].command_active_sources(active_targets)
            runtime["transmission"].write_internal_controls()
            hand_reference = controller_hand_records(runtime)
            reverse_violations = []
            if reject_sustained_follower_reverse:
                for joint_name in runtime["transmission"].follower_joint_names:
                    target_velocity = float(hand_reference[joint_name]["target_velocity_source_rad_s"])
                    actual_velocity = float(hand_reference[joint_name]["actual_velocity_source_rad_s"])
                    direction = 1 if target_velocity > 0.0 else -1 if target_velocity < 0.0 else 0
                    previous_direction = follower_target_direction.get(joint_name, 0)
                    if direction == 0:
                        follower_direction_steps[joint_name] = 0
                    elif direction != previous_direction:
                        follower_direction_steps[joint_name] = 1
                    else:
                        follower_direction_steps[joint_name] = follower_direction_steps.get(joint_name, 0) + 1
                    follower_target_direction[joint_name] = direction
                    if (
                        direction != 0
                        and follower_direction_steps.get(joint_name, 0) >= follower_direction_hold_steps
                        and actual_velocity * direction < 0.0
                    ):
                        reverse_violations.append({
                            "joint": joint_name,
                            "target_velocity_source_rad_s": target_velocity,
                            "actual_velocity_source_rad_s": actual_velocity,
                            "same_direction_samples": follower_direction_steps[joint_name],
                        })
            hand_target_acceleration = {}
            for joint_name, target in hand_reference.items():
                velocity = float(target["target_velocity_source_rad_s"])
                acceleration = (velocity - previous_target_velocity.get(joint_name, 0.0)) / float(model.opt.timestep)
                previous_target_velocity[joint_name] = velocity
                target["target_acceleration_source_rad_s2"] = acceleration
                hand_target_acceleration[joint_name] = acceleration
            detailed_before = detailed_joint_step(
                runtime, hand_reference, detailed_joint_names, hand_target_acceleration
            ) if detailed_joint_names else None
            mujoco.mj_step(model, data)
            state = support_diagnostics(runtime, support_targets)
            if detailed_before is not None:
                for joint_name, detail in detailed_before.items():
                    joint = runtime["joints"][joint_name]
                    dof = int(joint["dof"])
                    actuator_force = float(data.qfrc_actuator[dof])
                    detail["post_step"] = {
                        "raw_qpos": float(data.qpos[int(joint["qpos"])]),
                        "raw_qvel_rad_s": float(data.qvel[dof]),
                        "source_qpos_rad": gate.source_position(model, data, joint),
                        "source_qvel_rad_s": gate.source_velocity(data, joint),
                        "source_qacc_rad_s2": int(joint["axis_sign"]) * float(data.qacc[dof]),
                        "qfrc_actuator_raw_nm": actuator_force,
                        "qfrc_actuator_source_nm": int(joint["axis_sign"]) * actuator_force,
                        "qfrc_constraint_raw_nm": float(data.qfrc_constraint[dof]),
                        "qfrc_constraint_source_nm": int(joint["axis_sign"]) * float(data.qfrc_constraint[dof]),
                        "source_lower_limit_margin_rad": gate.source_position(model, data, joint) - float(joint["lower"]),
                    }
                    relation = detail["relation"]
                    if relation is not None:
                        driver = str(relation["driver_joint"])
                        follower = str(relation["follower_joint"])
                        relation["source_residual_rad"] = (
                            gate.source_position(model, data, runtime["joints"][follower])
                            - float(relation["multiplier"]) * gate.source_position(model, data, runtime["joints"][driver])
                            - float(relation["offset_rad"])
                        )
            step_index += 1
            hand_speed = float(state["hand_max_abs_qvel_rad_s"])
            peak_hand_speed = max(peak_hand_speed, hand_speed)
            phase_stats["max_abs_hand_qvel_rad_s"] = max(float(phase_stats["max_abs_hand_qvel_rad_s"]), hand_speed)
            phase_stats["max_abs_hand_qacc_rad_s2"] = max(float(phase_stats["max_abs_hand_qacc_rad_s2"]), float(state["hand_max_abs_qacc_rad_s2"]))
            phase_stats["max_abs_hand_target_velocity_rad_s"] = max(
                float(phase_stats["max_abs_hand_target_velocity_rad_s"]),
                max((abs(float(row["target_velocity_source_rad_s"])) for row in hand_reference.values()), default=0.0),
            )
            phase_stats["max_abs_hand_target_acceleration_rad_s2"] = max(
                float(phase_stats["max_abs_hand_target_acceleration_rad_s2"]),
                max((abs(float(value)) for value in hand_target_acceleration.values()), default=0.0),
            )
            phase_stats["max_mimic_error_rad"] = max(float(phase_stats["max_mimic_error_rad"]), float(state["source_mimic_error_max_rad"]))
            phase_stats["max_target_error_rad"] = max(float(phase_stats["max_target_error_rad"]), float(state["target_tracking_error_max_rad"]))
            coupling_peak = max((abs(float(row["dynamic_coupling_feedforward_nm"])) for row in hand_reference.values()), default=0.0)
            phase_stats["max_abs_dynamic_coupling_feedforward_nm"] = max(
                float(phase_stats["max_abs_dynamic_coupling_feedforward_nm"]), coupling_peak
            )
            result["max_abs_dynamic_coupling_feedforward_nm"] = max(
                float(result["max_abs_dynamic_coupling_feedforward_nm"]), coupling_peak
            )
            for hand_joint, record in hand_reference.items():
                for driver_joint, term in dict(record["dynamic_coupling_by_driver_nm"]).items():
                    driver_peaks = result["max_dynamic_coupling_by_driver_nm"]
                    driver_peaks[driver_joint] = max(
                        abs(float(term)), abs(float(driver_peaks.get(driver_joint, 0.0)))
                    )
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
            result["max_abs_hand_target_velocity_rad_s"] = max(
                float(result["max_abs_hand_target_velocity_rad_s"]),
                max((abs(float(row["target_velocity_source_rad_s"])) for row in hand_reference.values()), default=0.0),
            )
            result["max_abs_hand_target_acceleration_rad_s2"] = max(
                float(result["max_abs_hand_target_acceleration_rad_s2"]),
                max((abs(float(value)) for value in hand_target_acceleration.values()), default=0.0),
            )
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
                "hand_reference_dynamics": hand_reference,
                "follower_reverse_violations": reverse_violations,
                "active_rollout_follower_qpos_writes": 0,
            }
            if detailed_before is not None:
                row["detailed_hand"] = detailed_before
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
            elif reverse_violations:
                failure = "follower reversed against a sustained monotonic target"
            elif (
                max(float(state["source_mimic_error_max_rad"]), float(state["target_tracking_error_max_rad"]))
                >= HAND_ABORT_RAD if reject_sustained_follower_reverse
                else max(float(state["source_mimic_error_max_rad"]), float(state["target_tracking_error_max_rad"])) > HAND_ABORT_RAD
            ):
                failure = "hand tracking exceeded 0.010 rad abort ceiling"
            elif max((float(group["max_abs_motion_rad"]) for group in state["support_motion_by_group"].values()), default=0.0) > PREVALIDATION_SUPPORT_DRIFT_LIMIT_RAD:
                failure = "supported non-hand posture tracking"
            if phase == "final_open_hold":
                final_hold_samples.append(state)
            if reverse_violations:
                result["follower_reverse_samples"] = int(result["follower_reverse_samples"]) + len(reverse_violations)
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
        final_window = final_hold_samples[-25:]
        final_speed = max((float(row["hand_max_abs_qvel_rad_s"]) for row in final_window), default=0.0)
        final_error = max((float(row["target_tracking_error_max_rad"]) for row in final_window), default=0.0)
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
    if check_final_settle:
        settle_window = final_hold_samples[-100:]
        joint_names = sorted(final_hold_samples[-1]["hand_states"]) if final_hold_samples else []

        def window_rms(samples: list[dict[str, object]], value_fn) -> float:
            values = [value_fn(sample, joint_name) for sample in samples for joint_name in joint_names]
            return float(np.sqrt(np.mean(np.square(values)))) if values else float("inf")

        def velocity(sample, joint_name):
            return float(sample["hand_states"][joint_name]["source_velocity_rad_s"])

        def acceleration(sample, joint_name):
            return float(sample["hand_states"][joint_name]["source_acceleration_rad_s2"])

        def tracking_error(sample, joint_name):
            state = sample["hand_states"][joint_name]
            return float(state["source_position_rad"]) - float(state["target_position_rad"])

        early_window, late_window = settle_window[:50], settle_window[50:]
        early_rms = {
            "qvel_rad_s": window_rms(early_window, velocity),
            "qacc_rad_s2": window_rms(early_window, acceleration),
            "tracking_error_rad": window_rms(early_window, tracking_error),
        }
        late_rms = {
            "qvel_rad_s": window_rms(late_window, velocity),
            "qacc_rad_s2": window_rms(late_window, acceleration),
            "tracking_error_rad": window_rms(late_window, tracking_error),
        }
        final_window = final_hold_samples[-25:]
        final_speed = max((float(row["hand_max_abs_qvel_rad_s"]) for row in final_window), default=float("inf"))
        final_error = max((float(row["target_tracking_error_max_rad"]) for row in final_window), default=float("inf"))
        no_growth = all(late_rms[key] <= early_rms[key] for key in early_rms)
        settle_pass = (
            len(settle_window) == 100
            and final_speed <= max(1e-9, 0.10 * peak_hand_speed)
            and final_error <= HAND_ABORT_RAD
            and no_growth
        )
        result["final_settling"] = {
            "passed": settle_pass,
            "samples": len(settle_window),
            "window_duration_s": len(settle_window) * float(model.opt.timestep),
            "late_25_step_max_qvel_rad_s": final_speed,
            "late_25_step_max_target_error_rad": final_error,
            "global_peak_qvel_rad_s": peak_hand_speed,
            "velocity_decay_ratio_limit": 0.10,
            "tracking_error_ceiling_rad": HAND_ABORT_RAD,
            "early_half_rms": early_rms,
            "late_half_rms": late_rms,
            "no_growing_oscillation": no_growth,
            "qacc_peak_rad_s2": float(result["max_abs_hand_qacc_rad_s2"]),
            "qacc_finite_throughout": bool(result["max_abs_hand_qacc_rad_s2"] < float("inf")),
        }
        if result["passed"] and not settle_pass:
            failure = "final open hold did not settle without growing oscillation"
            result["passed"] = False
            result["first_failed_gate"] = failure
            result["first_failure"] = {
                "phase": "final_open_hold",
                "step": result["completed_steps"],
                "time_s": float(data.time),
                "gate": failure,
                "state": final_hold_samples[-1] if final_hold_samples else {},
                "settling": result["final_settling"],
            }
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
    result["controller_profile"] = runtime["controller_profile"]
    return result, runtime


def run_stage6_full_hand(
    trace_path: Path,
    *,
    operational_target_margin_rad: float | None = None,
    controller_profile_override: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    runtime_preview = gate.make_runtime(
        include_bottle=False,
        operational_target_margin_rad=operational_target_margin_rad,
        controller_profile_override=controller_profile_override,
    )
    initialize_support(runtime_preview)
    start = dict(runtime_preview["initial_active"])
    closed = {**start, **gate.feasible_driver_targets(runtime_preview)}
    partial = minimum_jerk_blend(start, closed, 0.5)
    phases = [
        ("open_hold", 200, start, {}),
        ("close", REFERENCE_DURATION_STEPS, closed, {}),
        ("partial", REFERENCE_DURATION_STEPS, partial, {}),
        ("closed_return", REFERENCE_DURATION_STEPS, closed, {}),
        ("closed_hold", 300, closed, {}),
        ("reopen", REFERENCE_DURATION_STEPS, start, {}),
        ("final_open_hold", 250, start, {}),
    ]
    with trace_path.open("w", encoding="utf-8") as trace:
        result, runtime = run_supported_trial(
            "stage6_final_full_hand",
            phases,
            trace,
            operational_target_margin_rad=operational_target_margin_rad,
            controller_profile_override=controller_profile_override,
            reject_sustained_follower_reverse=True,
            check_final_settle=True,
        )
    result["stage"] = "6 - final full-hand validation"
    result["required_sequence"] = ["OPEN", "CLOSE", "PARTIAL", "CLOSED HOLD", "REOPEN", "OPEN HOLD"]
    result["executed_phases"] = [phase[0] for phase in phases]
    result["fixed_gates"] = {
        "source_position_limits": result["position_limit_violation_samples"] == 0,
        "source_velocity_limits": result["velocity_limit_violation_samples"] == 0,
        "follower_tracking_below_0p010_rad": float(result["max_hand_target_error_rad"]) < HAND_ABORT_RAD,
        "no_sustained_reverse_against_monotonic_target": int(result["follower_reverse_samples"]) == 0,
        "qvel_and_qacc_finite_with_peaks_recorded": bool(result["initial"]["finite"]) and math.isfinite(float(result["max_abs_hand_qacc_rad_s2"])),
        "bounded_actuator_effort": not bool(result["hand_effort_bound_violation"]) and not bool(result["support_effort_bound_violation"]),
        "final_settle_without_growing_oscillation": bool(result.get("final_settling", {}).get("passed", False)),
        "no_follower_qpos_writes": int(result["active_rollout_follower_qpos_writes"]) == 0,
    }
    result["all_fixed_gates_passed"] = all(result["fixed_gates"].values())
    result["passed"] = bool(result["passed"]) and bool(result["all_fixed_gates_passed"])
    result["status"] = "PASS" if result["passed"] else "FAIL"
    result["trace_path"] = str(trace_path)
    result["runtime"] = {
        **result["runtime"],
        "timestep_s": float(runtime["model"].opt.timestep),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
    }
    return result


def original_return_overshoot_audit(trace_path: Path, previous_trace_path: Path) -> dict[str, object]:
    driver = "R_index_pip_joint"
    follower = "R_index_dip_joint"
    relation = None
    runtime_preview = gate.make_runtime(include_bottle=False)
    initialize_support(runtime_preview)
    for item in runtime_preview["details"]["mimic_relations"]:
        if str(item["driver_joint"]) == driver and str(item["follower_joint"]) == follower:
            relation = item
            break
    if relation is None:
        raise RuntimeError(f"missing source mimic relation {driver} -> {follower}")
    start = dict(runtime_preview["initial_active"])
    goal = dict(start)
    goal[driver] = min(
        runtime_preview["transmission"].operational_limits(driver)[1],
        start[driver] + 0.01,
    )
    phases = [
        ("supported_open_hold", 100, start, {}),
        ("source_step", 100, goal, {}),
        ("source_step_hold", 100, goal, {}),
        ("source_return", 100, start, {}),
        ("final_open_hold", 100, start, {}),
    ]
    with trace_path.open("w", encoding="utf-8") as trace_file:
        result, runtime = run_supported_trial(
            "stage1_original_return_audit",
            phases,
            trace_file,
            reference_blend=cubic_smooth_blend_exact,
            detailed_joint_names=(driver, follower),
        )
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return_rows = [row for row in rows if row["phase"] in ("source_return", "final_open_hold")]
    follower_trace = [row["detailed_hand"][follower] for row in return_rows]
    driver_trace = [row["detailed_hand"][driver] for row in rows]
    actual_positions = [float(row["post_step"]["source_qpos_rad"]) for row in follower_trace]
    follower_target_positions = [float(row["target_position_source_rad"]) for row in follower_trace]
    dynamic_undershoot = max(
        target - actual for target, actual in zip(follower_target_positions, actual_positions)
    )
    driver_open = float(start[driver])
    follower_open = float(relation["multiplier"]) * driver_open + float(relation["offset"])
    lower = float(runtime["joints"][follower]["lower"])
    follower_operational = runtime["transmission"].operational_limits(follower)
    driver_operational = runtime["transmission"].operational_limits(driver)
    target_inside_operational = all(
        driver_operational[0] - 1e-12 <= float(row["detailed_hand"][driver]["target_position_source_rad"]) <= driver_operational[1] + 1e-12
        and follower_operational[0] - 1e-12 <= float(row["detailed_hand"][follower]["target_position_source_rad"]) <= follower_operational[1] + 1e-12
        for row in rows
    )
    baseline_rows = [
        json.loads(line)
        for line in previous_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("trial") == "stage3_R_index_dip_follower"
    ]
    baseline_by_step = {int(row["step"]): row for row in baseline_rows}
    audit_by_step = {int(row["step"]): row for row in rows}
    compared_steps = sorted(set(baseline_by_step) & set(audit_by_step))
    max_position_delta = max((
        abs(
            float(baseline_by_step[step]["state"]["hand_states"][follower]["source_position_rad"])
            - float(audit_by_step[step]["detailed_hand"][follower]["post_step"]["source_qpos_rad"])
        )
        for step in compared_steps
    ), default=float("inf"))
    baseline_step = int((result.get("first_failure") or {}).get("step", -1))
    failure_joint = next((
        item["joint"]
        for item in ((result.get("first_failure") or {}).get("state") or {}).get("source_position_limit_violations", [])
        if item.get("joint") == follower
    ), None)
    return {
        "status": "PASS" if result.get("first_failed_gate") == "source hard position limit" and failure_joint == follower and max_position_delta <= 1e-12 else "FAIL",
        "stage": "1 - detailed replay of saved Stage 3 return transient",
        "classification": "B - operational OPEN target has insufficient dynamic margin",
        "classification_basis": "The cubic smoothstep target stays within its operational range and follows the source relation exactly, but the follower's damped response continues below its OPEN target far enough to cross the unchanged hard lower limit.",
        "original_reference_profile": {
            "shape": "cubic smoothstep",
            "ramp_duration_s": 0.2,
            "source_driver_delta_rad": float(goal[driver] - start[driver]),
            "peak_driver_target_velocity_rad_s": max(abs(float(row["target_velocity_source_rad_s"])) for row in driver_trace),
            "peak_follower_target_velocity_rad_s": max(abs(float(row["detailed_hand"][follower]["target_velocity_source_rad_s"])) for row in rows),
        },
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "timestep_s": float(runtime["model"].opt.timestep),
            "controller_profile": runtime["controller_profile"],
            "soft_limit_margin_rad": gate.SOFT_LIMIT_MARGIN_RAD,
            "source_joint_limits_modified": False,
            "controller_gains_modified": False,
            "active_rollout_follower_qpos_writes": 0,
        },
        "source_relation": {
            "driver": driver,
            "follower": follower,
            "multiplier": float(relation["multiplier"]),
            "offset_rad": float(relation["offset"]),
            "driver_source_limits_rad": [float(runtime["joints"][driver]["lower"]), float(runtime["joints"][driver]["upper"])],
            "follower_source_limits_rad": [lower, float(runtime["joints"][follower]["upper"])],
            "driver_open_target_rad": driver_open,
            "follower_derived_open_target_rad": follower_open,
            "follower_open_target_distance_to_hard_lower_rad": follower_open - lower,
            "target_position_ranges_rad": {
                "driver": [min(float(row["target_position_source_rad"]) for row in driver_trace), max(float(row["target_position_source_rad"]) for row in driver_trace)],
                "follower": [min(float(row["detailed_hand"][follower]["target_position_source_rad"]) for row in rows), max(float(row["detailed_hand"][follower]["target_position_source_rad"]) for row in rows)],
            },
            "target_position_ever_outside_operational_soft_range": not target_inside_operational,
            "dynamic_undershoot_below_follower_target_during_return_rad": dynamic_undershoot,
            "max_hard_lower_limit_crossing_rad": max(max(0.0, lower - position) for position in actual_positions),
            "first_crossing_step": baseline_step,
            "first_crossing_time_s": float((result.get("first_failure") or {}).get("time_s", -1.0)),
            "first_crossing_follower_state": (result.get("first_failure") or {}).get("state", {}).get("hand_states", {}).get(follower),
            "max_abs_driver_target_velocity_rad_s": max(abs(float(row["target_velocity_source_rad_s"])) for row in driver_trace),
            "max_abs_follower_target_velocity_rad_s": max(abs(float(row["detailed_hand"][follower]["target_velocity_source_rad_s"])) for row in rows),
            "max_abs_driver_target_acceleration_rad_s2": max(abs(float(row["target_acceleration_source_rad_s2"])) for row in driver_trace),
            "max_abs_follower_target_acceleration_rad_s2": max(abs(float(row["detailed_hand"][follower]["target_acceleration_source_rad_s2"])) for row in rows),
            "max_abs_follower_actual_velocity_during_return_rad_s": max(abs(float(row["post_step"]["source_qvel_rad_s"])) for row in follower_trace),
            "first_failure_gate": result.get("first_failed_gate"),
            "failure_diagnosis": result.get("first_failure_diagnosis"),
        },
        "saved_trace_comparison": {
            "previous_stage3_trace": str(previous_trace_path),
            "compared_steps": len(compared_steps),
            "max_abs_follower_position_delta_rad": max_position_delta,
            "same_first_crossing_step": baseline_step == 422,
            "matches_saved_failure_within_1e-12": max_position_delta <= 1e-12 and baseline_step == 422,
        },
        "trace_path": str(trace_path),
    }


def representative_hand_trials(
    trace_path: Path,
    *,
    operational_target_margin_rad: float | None = None,
    controller_profile_override: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    checks = [
        ("R_index_dip_follower", "R_index_pip_joint", 0.01),
        ("R_thumb_pip_follower", "R_thumb_mcp_joint", 0.01),
        ("active_finger_driver", "R_middle_pip_joint", 0.01),
        ("active_thumb_driver", "R_thumb_roll_joint", 0.01),
    ]
    runtime_options = {
        "operational_target_margin_rad": operational_target_margin_rad,
        "controller_profile_override": controller_profile_override,
    }
    results = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for label, joint_name, delta in checks:
            runtime = gate.make_runtime(include_bottle=False, **runtime_options)
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
                ("source_step", REFERENCE_DURATION_STEPS, goal, {}),
                ("source_step_hold", 100, goal, {}),
                ("source_return", REFERENCE_DURATION_STEPS, start, {}),
                ("final_open_hold", 250, start, {}),
            ]
            result, _ = run_supported_trial(f"stage3_{label}", one_trial, trace, **runtime_options)
            result["tested_coordinate"] = joint_name
            result["commanded_source_delta_rad"] = target - start[joint_name]
            result["reference_profile"] = {
                "shape": "quintic minimum-jerk",
                "duration_s": REFERENCE_DURATION_S,
                "duration_steps": REFERENCE_DURATION_STEPS,
                "source_position_target_clamping": False,
            }
            results.append(result)
            if not result["passed"]:
                break
    return results


def run_stage4(
    trace_path: Path,
    *,
    operational_target_margin_rad: float | None = None,
    controller_profile_override: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    runtime_options = {
        "operational_target_margin_rad": operational_target_margin_rad,
        "controller_profile_override": controller_profile_override,
    }
    results: list[dict[str, object]] = []
    with trace_path.open("w", encoding="utf-8") as trace:
        runtime = gate.make_runtime(include_bottle=False, **runtime_options)
        initialize_support(runtime)
        start = dict(runtime["initial_active"])
        closed = {**start, **gate.feasible_driver_targets(runtime)}
        phases = [
            ("open_hold", 200, start, {}),
            ("close", REFERENCE_DURATION_STEPS, closed, {}),
            ("partial", REFERENCE_DURATION_STEPS, minimum_jerk_blend(start, closed, 0.5), {}),
            ("closed_hold", 300, closed, {}),
            ("reopen", REFERENCE_DURATION_STEPS, start, {}),
        ]
        result, _ = run_supported_trial("stage4_full_hand_open_close", phases, trace, **runtime_options)
        result["gate_name"] = "OPEN HOLD -> CLOSE -> PARTIAL -> CLOSED HOLD -> REOPEN"
        results.append(result)
        if not result["passed"]:
            return results

        runtime = gate.make_runtime(include_bottle=False, **runtime_options)
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
        result, _ = run_supported_trial("stage4_supported_right_arm_motion", arm_phases, trace, **runtime_options)
        result["gate_name"] = "supported right-arm IK motion with the left arm and body support held"
        result["ik_route_raw_joint_targets"] = np.asarray(arm_route).tolist()
        results.append(result)
        if not result["passed"]:
            return results

        relation_drivers = sorted({str(row["driver_joint"]) for row in runtime["details"]["mimic_relations"]})
        for driver in relation_drivers:
            for direction in (-1.0, 1.0):
                runtime = gate.make_runtime(include_bottle=False, **runtime_options)
                initialize_support(runtime)
                start = dict(runtime["initial_active"])
                low, high = runtime["transmission"].operational_limits(driver)
                target = min(high, max(low, start[driver] + direction * 0.01))
                goal = dict(start)
                goal[driver] = target
                phases = [
                    ("supported_open_hold", 25, start, {}),
                    ("driver_perturb", REFERENCE_DURATION_STEPS, goal, {}),
                    ("driver_perturb_hold", 50, goal, {}),
                    ("driver_return", REFERENCE_DURATION_STEPS, start, {}),
                    ("final_open_hold", 25, start, {}),
                ]
                result, _ = run_supported_trial(
                    f"stage4_driver_perturb_{driver}_{direction:+.0f}x0p01",
                    phases,
                    trace,
                    **runtime_options,
                )
                result["gate_name"] = "valid single source-driver perturbation and return"
                result["driver_joint"] = driver
                result["requested_delta_rad"] = target - start[driver]
                results.append(result)
                if not result["passed"]:
                    return results

        runtime = gate.make_runtime(fixed_probe=True, include_bottle=False, **runtime_options)
        initialize_support(runtime)
        start = dict(runtime["initial_active"])
        right_close = {**start, **{
            joint: value
            for joint, value in gate.feasible_driver_targets(runtime).items()
            if joint.startswith("R_")
        }}
        phases = [
            ("close_to_fixed_probe", REFERENCE_DURATION_STEPS, right_close, {}),
            ("fixed_probe_hold", 500, right_close, {}),
            ("open_from_fixed_probe", REFERENCE_DURATION_STEPS, start, {}),
        ]
        result, _ = run_supported_trial(
            "stage4_fixed_object_fingertip_contact",
            phases,
            trace,
            fixed_probe=True,
            **runtime_options,
        )
        result["gate_name"] = "supported fixed-object fingertip contact"
        results.append(result)
    return results


def derive_final_open_margin(previous_dir: Path) -> dict[str, object]:
    stage3_path = previous_dir / "stage3_representative_hand_result.json"
    trace_path = previous_dir / "stage3_representative_hand_trace.jsonl"
    if not stage3_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError("final margin derivation requires the preserved prior Stage 3 JSON and trace")
    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    trials = list(stage3.get("results", []))
    if stage3.get("status") != "FAIL" or not trials:
        raise ValueError("expected the preserved Stage 3 endpoint-crossing failure packet")
    failed = next((trial for trial in trials if not trial.get("passed")), None)
    if failed is None or failed.get("first_failed_gate") != "source hard position limit":
        raise ValueError("preserved Stage 3 does not contain the expected source hard-limit failure")
    first_failure = failed.get("first_failure") or {}
    failure_state = first_failure.get("state") or {}
    violations = failure_state.get("source_position_limit_violations") or []
    if not violations:
        raise ValueError("preserved first failure lacks a source position-limit crossing")
    crossing = violations[0]
    crossing_joint = str(crossing["joint"])
    crossing_state = failure_state["hand_states"][crossing_joint]
    dynamic_undershoot = abs(
        float(crossing_state["target_position_rad"]) - float(crossing_state["source_position_rad"])
    )

    passing = next((trial for trial in trials if trial.get("passed")), None)
    if passing is None:
        raise ValueError("preserved Stage 3 has no completed smooth-reference case for terminal-velocity measurement")
    passing_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return_rows = [
        row for row in passing_rows
        if row.get("trial") == passing["label"] and row.get("phase") == "source_return"
    ]
    if not return_rows:
        raise ValueError("preserved passing case lacks its smooth-return terminal sample")
    terminal_row = max(return_rows, key=lambda row: int(row["step"]))
    terminal_velocity = max(
        (
            abs(float(record["target_velocity_source_rad_s"]))
            for record in terminal_row["hand_reference_dynamics"].values()
        ),
        default=0.0,
    )
    dt = float(task.DT)
    tracking_error = max(float(trial["max_hand_target_error_rad"]) for trial in trials)
    mimic_residual = max(float(trial["max_source_mimic_error_rad"]) for trial in trials)
    peak_qvel = max(float(trial["max_abs_hand_qvel_rad_s"]) for trial in trials)
    peak_qacc = max(float(trial["max_abs_hand_qacc_rad_s2"]) for trial in trials)
    integration_allowance = dt * peak_qvel + 0.5 * dt * dt * peak_qacc
    tracked_excursion = max(dynamic_undershoot, tracking_error)
    unrounded_margin = tracked_excursion + mimic_residual + terminal_velocity * dt + integration_allowance
    margin = math.ceil(unrounded_margin / FINAL_MARGIN_ROUNDING_RAD - 1e-12) * FINAL_MARGIN_ROUNDING_RAD
    return {
        "status": "DERIVED",
        "method": "max(measured endpoint target-to-state excursion, max target tracking error) + max mimic residual + terminal target travel + one-step integration allowance; rounded upward to 0.0001 rad",
        "source_limits_modified": False,
        "controller_design_margin_rad": gate.SOFT_LIMIT_MARGIN_RAD,
        "timestep_s": dt,
        "inputs": {
            "failed_joint": crossing_joint,
            "measured_endpoint_target_to_state_excursion_rad": dynamic_undershoot,
            "measured_lower_limit_crossing_rad": max(0.0, -float(crossing["lower_margin_rad"])),
            "max_target_tracking_error_rad": tracking_error,
            "max_mimic_residual_rad": mimic_residual,
            "smooth_reference_terminal_target_velocity_rad_s": terminal_velocity,
            "terminal_target_travel_allowance_rad": terminal_velocity * dt,
            "max_actual_hand_speed_rad_s": peak_qvel,
            "max_actual_hand_acceleration_rad_s2": peak_qacc,
            "one_step_integration_allowance_rad": integration_allowance,
        },
        "tracked_excursion_rad": tracked_excursion,
        "unrounded_required_margin_rad": unrounded_margin,
        "rounding_increment_rad": FINAL_MARGIN_ROUNDING_RAD,
        "selected_operational_open_margin_rad": margin,
        "selected_margin_is_negligible_relative_to_finger_range": margin < 0.01,
        "controller_gains_or_design_constants_changed": False,
        "prior_stage3_result": str(stage3_path),
        "prior_stage3_trace": str(trace_path),
    }


def open_target_map(runtime: dict[str, object]) -> dict[str, float]:
    targets = dict(runtime["initial_active"])
    for relation in runtime["details"]["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        targets[follower] = float(relation["multiplier"]) * targets[driver] + float(relation["offset"])
    return targets


def build_endpoint_target_audit(
    old_runtime: dict[str, object],
    new_runtime: dict[str, object],
    margin_rad: float,
) -> dict[str, object]:
    old_targets = open_target_map(old_runtime)
    new_targets = open_target_map(new_runtime)
    followers = set(new_runtime["transmission"].follower_joint_names)
    names = sorted(old_targets)
    rows = []
    invalid = []
    for joint_name in names:
        joint = new_runtime["joints"][joint_name]
        lower, upper = float(joint["lower"]), float(joint["upper"])
        old_value, new_value = old_targets[joint_name], new_targets[joint_name]
        low_new, high_new = new_runtime["transmission"].operational_limits(joint_name)
        endpoint = "lower" if abs(old_value - lower) <= abs(old_value - upper) else "upper"
        row = {
            "joint": joint_name,
            "coordinate_type": "follower" if joint_name in followers else "active",
            "source_hard_limits_rad": {"lower": lower, "upper": upper},
            "source_hard_limits_unchanged": (
                lower == float(old_runtime["joints"][joint_name]["lower"])
                and upper == float(old_runtime["joints"][joint_name]["upper"])
            ),
            "nearest_endpoint_from_old_open": endpoint,
            "old_operational_open_target_rad": old_value,
            "new_operational_open_target_rad": new_value,
            "old_target_distance_to_nearest_endpoint_rad": min(old_value - lower, upper - old_value),
            "new_target_distance_to_nearest_endpoint_rad": min(new_value - lower, upper - new_value),
            "new_operational_target_interval_rad": [low_new, high_new],
            "new_target_inside_operational_interval": low_new <= new_value <= high_new,
            "target_changed_rad": new_value - old_value,
            "endpoint_near_open_posture": min(
                min(old_value - lower, upper - old_value),
                min(new_value - lower, upper - new_value),
            ) <= margin_rad + 1e-12,
        }
        rows.append(row)
        if not row["source_hard_limits_unchanged"] or not row["new_target_inside_operational_interval"]:
            invalid.append(joint_name)
    old_profile = old_runtime["controller_profile"]
    new_profile = new_runtime["controller_profile"]
    gains_equal = all(
        old_profile[name][key] == new_profile[name][key]
        for name in old_profile
        for key in ("kp_nm_per_rad", "kv_nms_per_rad")
    )
    minimum_range = min(
        float(new_runtime["joints"][name]["upper"]) - float(new_runtime["joints"][name]["lower"])
        for name in names
    )
    return {
        "status": "PASS" if not invalid and gains_equal else "FAIL",
        "operational_open_margin_rad": margin_rad,
        "minimum_hand_source_range_rad": minimum_range,
        "margin_fraction_of_minimum_hand_range": margin_rad / minimum_range,
        "source_hard_limits_modified": False,
        "controller_gains_exactly_preserved": gains_equal,
        "controller_design_constants": {
            "wn_dt": gate.CONTROLLER_WN_DT_TARGET,
            "damping_ratio": gate.CONTROLLER_DAMPING_RATIO,
            "effort_bounds_unchanged": True,
            "mimic_mapping_unchanged": True,
        },
        "endpoint_near_rows": [row for row in rows if row["endpoint_near_open_posture"]],
        "all_hand_target_rows": rows,
        "invalid_joints": invalid,
    }


def run_final_endpoint_margin_iteration() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reserved = [
        "final_margin_derivation.json",
        "endpoint_target_audit.json",
        "stage3_representative_hand_result.json",
        "stage3_representative_hand_trace.jsonl",
    ]
    if any((OUT / filename).exists() for filename in reserved):
        raise FileExistsError(f"refusing to overwrite existing final-iteration evidence in {OUT}")
    derivation = derive_final_open_margin(FINAL_OPERATIONAL_MARGIN_INPUTS)
    margin = float(derivation["selected_operational_open_margin_rad"])
    write_json(OUT / "final_margin_derivation.json", derivation)
    identity = runtime_identity()
    identity["robot_sim_worktree_status"] = subprocess.check_output(
        ["git", "-C", str(task.SIM_REPO_ROOT), "status", "--short", "--branch"], text=True
    ).strip()
    write_json(OUT / "runtime_identity.json", identity)

    reference_runtime = gate.make_runtime(include_bottle=False)
    candidate_runtime = gate.make_runtime(
        include_bottle=False,
        operational_target_margin_rad=margin,
        controller_profile_override=reference_runtime["controller_profile"],
    )
    initialize_support(candidate_runtime)
    target_audit = build_endpoint_target_audit(reference_runtime, candidate_runtime, margin)
    initial_support_state = support_diagnostics(candidate_runtime)
    target_audit["candidate_initial_support_state"] = initial_support_state
    target_audit["status"] = "PASS" if (
        target_audit["status"] == "PASS"
        and not initial_support_state["source_position_limit_violations"]
        and not initial_support_state["source_velocity_limit_violations"]
        and not initial_support_state["self_contacts"]
        and float(initial_support_state["max_self_penetration_m"]) <= 1e-12
        and initial_support_state["finite"]
        and float(initial_support_state["source_mimic_error_max_rad"]) <= 1e-9
    ) else "FAIL"
    target_audit["gain_profile_reference"] = "original accepted 0.0015 rad profile; exact per-joint kp/kv copied"
    write_json(OUT / "endpoint_target_audit.json", target_audit)
    write_json(OUT / "controller_profile.json", {
        "design_margin_rad": gate.SOFT_LIMIT_MARGIN_RAD,
        "operational_target_margin_rad": margin,
        "exact_gain_match": target_audit["controller_gains_exactly_preserved"],
        "profile": candidate_runtime["controller_profile"],
    })
    if target_audit["status"] != "PASS":
        write_json(OUT / "final_iteration_result.json", {
            "status": "FAIL",
            "stop_stage": "1_target_preflight",
            "first_failure": target_audit,
            "conditional_stages_not_run": ["2 representative cases", "3 full-hand validation", "4 bottle hold", "5 mm extension"],
        })
        return 2

    stage3 = representative_hand_trials(
        OUT / "stage3_representative_hand_trace.jsonl",
        operational_target_margin_rad=margin,
        controller_profile_override=reference_runtime["controller_profile"],
    )
    stage3_passed = len(stage3) == 4 and all(bool(row["passed"]) for row in stage3)
    stage3_result = {
        "status": "PASS" if stage3_passed else "FAIL",
        "passed": stage3_passed,
        "operational_target_margin_rad": margin,
        "controller_gains_frozen_from_original_profile": True,
        "required_cases": ["R_index_dip follower", "R_thumb_dip / thumb follower", "active finger driver", "active thumb driver"],
        "results": stage3,
    }
    write_json(OUT / "stage3_representative_hand_result.json", stage3_result)
    if not stage3_passed:
        first = next(row for row in stage3 if not row["passed"])
        final = {
            "status": "FAIL",
            "stop_stage": "2_representative_cases",
            "first_failure": first.get("first_failure_diagnosis") or first.get("first_failure") or {"gate": first.get("first_failed_gate"), "label": first["label"]},
            "stage1_target_audit": target_audit,
            "stage2_representative_cases": stage3_result,
            "conditional_stages_not_run": ["3 full-hand prevalidation", "4 bottle close/hold", "5 mm extension"],
            "runtime_identity": identity,
        }
        write_json(OUT / "final_iteration_result.json", final)
        print(json.dumps({"status": "FAIL", "stop_stage": final["stop_stage"], "first_failure": final["first_failure"], "evidence_dir": str(OUT)}, indent=2))
        return 2

    stage4 = run_stage4(
        OUT / "stage4_full_hand_prevalidation_trace.jsonl",
        operational_target_margin_rad=margin,
        controller_profile_override=reference_runtime["controller_profile"],
    )
    stage4_passed = bool(stage4) and all(bool(row["passed"]) for row in stage4)
    stage4_result = {
        "status": "PASS" if stage4_passed else "FAIL",
        "passed": stage4_passed,
        "operational_target_margin_rad": margin,
        "results": stage4,
    }
    write_json(OUT / "stage4_full_hand_prevalidation_result.json", stage4_result)
    final = {
        "status": "PASS" if stage4_passed else "FAIL",
        "stop_stage": "3_full_hand_prevalidation" if not stage4_passed else "3_full_hand_prevalidation_complete",
        "first_failure": next((row.get("first_failure_diagnosis") or row.get("first_failure") or {"gate": row.get("first_failed_gate"), "label": row["label"]} for row in stage4 if not row["passed"]), None),
        "stage1_target_audit": target_audit,
        "stage2_representative_cases": stage3_result,
        "stage3_full_hand_prevalidation": stage4_result,
        "conditional_stages_not_run": ["4 bottle close and hold", "5 mm extension"] if not stage4_passed else ["4 bottle close and hold", "5 mm extension require a virtual-transmission bottle-hold harness"],
        "runtime_identity": identity,
    }
    write_json(OUT / "final_iteration_result.json", final)
    print(json.dumps({
        "status": final["status"],
        "stop_stage": final["stop_stage"],
        "first_failure": final["first_failure"],
        "stage3_representatives": [{"label": row["label"], "passed": row["passed"], "first_failed_gate": row["first_failed_gate"]} for row in stage3],
        "stage4_trials": [{"label": row["label"], "passed": row["passed"], "first_failed_gate": row["first_failed_gate"]} for row in stage4],
        "evidence_dir": str(OUT),
    }, indent=2))
    return 0 if stage4_passed else 2


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
    if "--stage6-final-full-hand" in sys.argv:
        previous_dir = Path(os.environ.get("ISSUE46_PREVIOUS_EVIDENCE_DIR", ""))
        stage3_path = previous_dir / "stage3_damping_corrected_result.json"
        identity_path = previous_dir / "damping_correction_identity.json"
        if not stage3_path.is_file() or not identity_path.is_file():
            raise RuntimeError("Stage 6 requires the preserved Stage 3 PASS result and damping identity")
        stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
        correction = json.loads(identity_path.read_text(encoding="utf-8"))
        trials = list(stage3.get("results", []))
        if stage3.get("status") != "PASS" or len(trials) != 4 or not all(row.get("passed") for row in trials):
            raise RuntimeError("preserved Stage 3 result is not a four-case PASS")
        if any(int(row.get("active_rollout_follower_qpos_writes", -1)) != 0 for row in trials):
            raise RuntimeError("preserved Stage 3 evidence reports a follower qpos write")
        if correction.get("accepted_robot_sim_head") != subprocess.check_output(
            ["git", "-C", str(task.SIM_REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip():
            raise RuntimeError("damping correction identity belongs to a different RobotSim HEAD")
        if abs(float(correction.get("derived_kv_multiplier", 0.0)) - 4.97032338287281) > 1e-12:
            raise RuntimeError("preserved damping correction is not the accepted 4.97032338287281x correction")
        profiles = [row.get("controller_profile") for row in trials]
        if any(profile is None for profile in profiles) or any(profile != profiles[0] for profile in profiles[1:]):
            raise RuntimeError("Stage 3 cases do not share one identical frozen controller profile")
        margins = {float(row["operational_target_margin_rad"]) for row in trials}
        if len(margins) != 1:
            raise RuntimeError("Stage 3 cases do not share one operational target margin")
        margin = margins.pop()
        identity = runtime_identity()
        if not identity["vendor_pin_matches"]:
            raise RuntimeError("current vendor checkout does not match the pinned source")
        result = run_stage6_full_hand(
            OUT / "stage6_final_full_hand_trace.jsonl",
            operational_target_margin_rad=margin,
            controller_profile_override=profiles[0],
        )
        result["runtime_identity"] = identity
        result["stage3_reference_result"] = str(stage3_path)
        result["damping_correction_identity"] = correction
        write_json(OUT / "stage6_final_full_hand_result.json", result)
        print(json.dumps({
            "status": result["status"],
            "passed": result["passed"],
            "first_failed_gate": result["first_failed_gate"],
            "completed_steps": result["completed_steps"],
            "max_hand_target_error_rad": result["max_hand_target_error_rad"],
            "max_hand_qvel_rad_s": result["max_abs_hand_qvel_rad_s"],
            "max_hand_qacc_rad_s2": result["max_abs_hand_qacc_rad_s2"],
            "max_dynamic_coupling_feedforward_nm": result["max_abs_dynamic_coupling_feedforward_nm"],
            "follower_reverse_samples": result["follower_reverse_samples"],
            "fixed_gates": result["fixed_gates"],
            "trace_path": result["trace_path"],
        }, indent=2))
        return 0 if result["passed"] else 2
    if "--final-endpoint-margin" in sys.argv:
        return run_final_endpoint_margin_iteration()
    if "--audit-original-stage3" in sys.argv:
        previous_dir = Path(os.environ.get("ISSUE46_PREVIOUS_EVIDENCE_DIR", ""))
        previous_trace = previous_dir / "stage3_representative_hand_trace.jsonl"
        if not previous_trace.is_file():
            raise RuntimeError("--audit-original-stage3 requires ISSUE46_PREVIOUS_EVIDENCE_DIR with the saved Stage 3 trace")
        audit = original_return_overshoot_audit(OUT / "stage1_return_overshoot_trace.jsonl", previous_trace)
        audit["reproduction_command"] = (
            f"cd {task.SIM_REPO_ROOT} && env "
            f"AGIBOT_X2_VENDOR_ROOT={task.ROOT} "
            f"ISSUE46_PREVIOUS_EVIDENCE_DIR={previous_dir} ISSUE46_EVIDENCE_DIR={OUT} "
            f"PYTHONDONTWRITEBYTECODE=1 {sys.executable} -u {Path(__file__).resolve()} --audit-original-stage3"
        )
        write_json(OUT / "stage1_return_overshoot_audit.json", audit)
        print(json.dumps({
            "status": audit["status"],
            "classification": audit["classification"],
            "crossing_step": audit["source_relation"]["first_crossing_step"],
            "crossing_time_s": audit["source_relation"]["first_crossing_time_s"],
            "open_margin_rad": audit["source_relation"]["follower_open_target_distance_to_hard_lower_rad"],
            "return_undershoot_rad": audit["source_relation"]["dynamic_undershoot_below_follower_target_during_return_rad"],
            "saved_trace_match": audit["saved_trace_comparison"]["matches_saved_failure_within_1e-12"],
            "trace_path": audit["trace_path"],
        }, indent=2))
        return 0 if audit["status"] == "PASS" else 2
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
