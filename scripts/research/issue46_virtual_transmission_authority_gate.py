from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

import issue46_virtual_transmission_gate as gate


OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-vt-authority-gate"))
STATIC_HOLD_STEPS = 1000
TRACKING_ABORT_RAD = 0.010


def hand_force_balance(runtime: dict[str, object]) -> list[dict[str, object]]:
    model = runtime["model"]
    data = runtime["data"]
    joints = runtime["joints"]
    transmission = runtime["transmission"]
    names = set(transmission.active_joint_names) | set(transmission.follower_joint_names)
    rows = []
    for joint_name in sorted(names):
        joint = joints[joint_name]
        dof = int(joint["dof"])
        rows.append({
            "joint": joint_name,
            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
            "ff_sign_residual_nm": float(data.qfrc_actuator[dof] - data.qfrc_bias[dof]),
        })
    return rows


def run_one_step() -> dict[str, object]:
    runtime = gate.make_runtime()
    model, data = runtime["model"], runtime["data"]
    initial = gate.diagnostics(runtime)
    mujoco.mj_forward(model, data)
    force_balance = hand_force_balance(runtime)
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    qacc_before = data.qacc.copy()
    mujoco.mj_step(model, data)
    after = gate.diagnostics(runtime)
    max_ff_residual = max((abs(float(row["ff_sign_residual_nm"])) for row in force_balance), default=0.0)
    passed = (
        not initial["position_limit_violations"]
        and not initial["robot_self_contacts"]
        and initial["source_relation_error_max_rad"] <= 1e-9
        and not after["position_limit_violations"]
        and not after["velocity_limit_violations"]
        and after["finite"]
        and not after["robot_self_contacts"]
        and float(after["max_self_penetration_m"]) <= 1e-12
        and not after["actuator_effort_violation"]
        and max_ff_residual <= 1e-9
        and float(after["source_relation_error_max_rad"]) < TRACKING_ABORT_RAD
        and float(after["target_tracking_error_max_rad"]) < TRACKING_ABORT_RAD
    )
    return {
        "experiment": "bias feedforward sign check and one MuJoCo step from the initialized operational OPEN state",
        "status": "PASS" if passed else "FAIL",
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "equation_used": "M*qacc + qfrc_bias = qfrc_actuator + qfrc_applied + qfrc_passive + qfrc_constraint",
        "feedforward_sign": "tau_bias_ff = +qfrc_bias in MuJoCo generalized-coordinate convention",
        "why_sign_cancels": "At zero velocity and zero target error, qfrc_actuator=qfrc_bias makes the hand generalized residual zero in the equation above.",
        "runtime_initialization_qpos_writes_pre_t0": runtime["initialization_qpos_write_count"],
        "active_rollout_qpos_writes": 0,
        "follower_rollout_qpos_writes": 0,
        "before_step": {
            "qpos_max_abs_delta_from_initialized_state": float(np.max(np.abs(data.qpos - qpos_before))),
            "qvel_max_abs": float(np.max(np.abs(qvel_before))),
            "qacc_max_abs_rad_s2": float(np.max(np.abs(qacc_before))),
            "hand_force_balance": force_balance,
            "max_feedforward_sign_residual_nm": max_ff_residual,
        },
        "after_step": {
            "time_s": float(data.time),
            "position_limit_violations": after["position_limit_violations"],
            "velocity_limit_violations": after["velocity_limit_violations"],
            "finite": after["finite"],
            "self_contact_count": len(after["robot_self_contacts"]),
            "max_self_penetration_m": after["max_self_penetration_m"],
            "mimic_error_max_rad": after["source_relation_error_max_rad"],
            "active_tracking_error_max_rad": after["active_target_tracking_error_max_rad"],
            "follower_tracking_error_max_rad": after["follower_target_tracking_error_max_rad"],
            "max_abs_qvel_rad_s": after["max_abs_qvel_rad_s"],
            "max_abs_qacc_rad_s2": after["max_abs_qacc_rad_s2"],
            "actuator_effort_violation": after["actuator_effort_violation"],
        },
        "controller_profile": runtime["controller_profile"],
    }


def run_static_open_hold() -> dict[str, object]:
    runtime = gate.make_runtime()
    model, data = runtime["model"], runtime["data"]
    transmission = runtime["transmission"]
    target = dict(runtime["initial_active"])
    initial = gate.diagnostics(runtime)
    trace_path = OUT / "static_open_hold_trace.jsonl"
    worst = {
        "active_tracking_error_rad": 0.0,
        "follower_tracking_error_rad": 0.0,
        "mimic_residual_rad": 0.0,
        "bias_feedforward_torque_nm": 0.0,
        "pd_torque_nm": 0.0,
        "total_commanded_torque_nm": 0.0,
        "qvel_rad_s": 0.0,
        "qacc_rad_s2": 0.0,
        "self_penetration_m": 0.0,
    }
    first_failed_gate = None
    first_failure = None
    with trace_path.open("w", encoding="utf-8") as trace:
        if initial["robot_self_contacts"] or float(initial["max_self_penetration_m"]) > 1e-12:
            first_failed_gate = "initial self-contact or penetration"
        elif initial["position_limit_violations"]:
            first_failed_gate = "initial source hard-limit violation"
        elif initial["velocity_limit_violations"]:
            first_failed_gate = "initial source velocity-limit violation"
        elif not initial["finite"]:
            first_failed_gate = "non-finite initialized state"
        elif float(initial["source_relation_error_max_rad"]) > 1e-9:
            first_failed_gate = "initial mimic geometry"

        steps = 0
        for step in range(1, STATIC_HOLD_STEPS + 1):
            if first_failed_gate:
                break
            transmission.command_active_sources(target)
            transmission.write_internal_controls()
            mujoco.mj_step(model, data)
            state = gate.diagnostics(runtime)
            steps = step
            active_values = list(state["active_joint_states"].values())
            follower_values = list(state["mimic_relations"].values())
            torque_rows = active_values + follower_values
            worst["active_tracking_error_rad"] = max(worst["active_tracking_error_rad"], float(state["active_target_tracking_error_max_rad"]))
            worst["follower_tracking_error_rad"] = max(worst["follower_tracking_error_rad"], float(state["follower_target_tracking_error_max_rad"]))
            worst["mimic_residual_rad"] = max(worst["mimic_residual_rad"], float(state["source_relation_error_max_rad"]))
            worst["bias_feedforward_torque_nm"] = max(worst["bias_feedforward_torque_nm"], max((abs(float(row["bias_feedforward_nm"])) for row in torque_rows), default=0.0))
            worst["pd_torque_nm"] = max(worst["pd_torque_nm"], max((abs(float(row["pd_torque_nm"])) for row in torque_rows), default=0.0))
            worst["total_commanded_torque_nm"] = max(worst["total_commanded_torque_nm"], max((abs(float(row["commanded_torque_nm"])) for row in torque_rows), default=0.0))
            worst["qvel_rad_s"] = max(worst["qvel_rad_s"], float(state["max_abs_qvel_rad_s"]))
            worst["qacc_rad_s2"] = max(worst["qacc_rad_s2"], float(state["max_abs_qacc_rad_s2"]))
            worst["self_penetration_m"] = max(worst["self_penetration_m"], float(state["max_self_penetration_m"]))

            trace.write(json.dumps({
                "step": step,
                "time_s": float(data.time),
                "active_joint_states": state["active_joint_states"],
                "mimic_relations": state["mimic_relations"],
                "source_joint_limits": state["source_joint_limits"],
                "position_limit_violations": state["position_limit_violations"],
                "velocity_limit_violations": state["velocity_limit_violations"],
                "active_efforts": state["active_efforts"],
                "follower_efforts": state["follower_efforts"],
                "self_contacts": state["robot_self_contacts"],
                "max_abs_qvel_rad_s": state["max_abs_qvel_rad_s"],
                "max_abs_qacc_rad_s2": state["max_abs_qacc_rad_s2"],
                "finite": state["finite"],
                "active_rollout_qpos_writes": 0,
                "follower_rollout_qpos_writes": 0,
            }, separators=(",", ":")) + "\n")
            trace.flush()

            if state["position_limit_violations"]:
                first_failed_gate = "source hard-limit violation during operational OPEN hold"
            elif state["velocity_limit_violations"]:
                first_failed_gate = "source velocity-limit violation during operational OPEN hold"
            elif not state["finite"]:
                first_failed_gate = "non-finite state during operational OPEN hold"
            elif state["actuator_effort_violation"]:
                first_failed_gate = "actuator torque exceeded source/simulation bound"
            elif float(state["source_relation_error_max_rad"]) >= TRACKING_ABORT_RAD:
                first_failed_gate = "mimic residual reached 0.010 rad manipulation abort ceiling"
            elif float(state["target_tracking_error_max_rad"]) >= TRACKING_ABORT_RAD:
                first_failed_gate = "active/follower target tracking reached 0.010 rad bound"
            elif float(state["max_self_penetration_m"]) > 1e-12:
                first_failed_gate = "self-penetration during operational OPEN hold"
            if first_failed_gate:
                first_failure = {
                    "step": step,
                    "time_s": float(data.time),
                    "first_failed_gate": first_failed_gate,
                    "position_limit_violations": state["position_limit_violations"],
                    "velocity_limit_violations": state["velocity_limit_violations"],
                    "mimic_error_max_rad": state["source_relation_error_max_rad"],
                    "active_tracking_error_max_rad": state["active_target_tracking_error_max_rad"],
                    "follower_tracking_error_max_rad": state["follower_target_tracking_error_max_rad"],
                    "qvel_max_abs_rad_s": state["max_abs_qvel_rad_s"],
                    "qacc_max_abs_rad_s2": state["max_abs_qacc_rad_s2"],
                }
                break

    passed = first_failed_gate is None and steps == STATIC_HOLD_STEPS
    return {
        "experiment": "clean-reset operational OPEN static hold; 2 seconds; no bottle, external object, or arm target motion",
        "status": "PASS" if passed else "FAIL",
        "first_failed_gate": first_failed_gate,
        "first_failure": first_failure,
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "duration_s": float(data.time),
        "steps": steps,
        "operational_open_target_rad": target,
        "source_limits_modified": False,
        "runtime_initialization_qpos_writes_pre_t0": runtime["initialization_qpos_write_count"],
        "active_rollout_qpos_writes": 0,
        "follower_rollout_qpos_writes": 0,
        "initial_state": {
            "hard_limit_violation_count": len(initial["position_limit_violations"]),
            "self_contact_count": len(initial["robot_self_contacts"]),
            "max_self_penetration_m": initial["max_self_penetration_m"],
            "mimic_residual_rad": initial["source_relation_error_max_rad"],
        },
        "worst": worst,
        "diagnostic_0p003_rad_target_met": worst["mimic_residual_rad"] < 0.003,
        "manipulation_0p010_rad_abort_ceiling_met": worst["mimic_residual_rad"] < 0.010,
        "controller_profile": runtime["controller_profile"],
        "trace_path": str(trace_path),
        "exact_command": "AGIBOT_X2_VENDOR_ROOT=<pinned checkout> ISSUE46_EVIDENCE_DIR=<evidence directory> <venv>/bin/python scripts/research/issue46_virtual_transmission_authority_gate.py --static-open-hold",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--static-open-hold" in os.sys.argv:
        result = run_static_open_hold()
        filename = "static_open_hold_result.json"
    else:
        result = run_one_step()
        filename = "bias_aware_one_step.json"
    path = OUT / filename
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(path), **{key: result[key] for key in ("status", "first_failed_gate", "steps", "duration_s") if key in result}}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
