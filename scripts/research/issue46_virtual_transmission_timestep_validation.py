from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_virtual_transmission_gate as gate  # noqa: E402


OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-timestep-aware"))
STEP_RAD = 0.005
OPEN_HOLD_STEPS = 1000
STEP_HOLD_STEPS = 200
RETURN_HOLD_STEPS = 200
TRACKING_ABORT_RAD = 0.010
SETTLE_FRACTION = 0.02
OVERSHOOT_FRACTION = 0.05


def driver_interval(runtime: dict[str, Any], driver_name: str) -> tuple[float, float]:
    transmission = runtime["transmission"]
    low, high = transmission.operational_limits(driver_name)
    for relation in runtime["details"]["mimic_relations"]:
        if str(relation["driver_joint"]) != driver_name:
            continue
        multiplier = float(relation["multiplier"])
        offset = float(relation["offset"])
        if abs(multiplier) < 1e-12:
            raise ValueError(f"zero mimic multiplier for {driver_name}")
        follower_low, follower_high = transmission.operational_limits(str(relation["follower_joint"]))
        mapped = ((follower_low - offset) / multiplier, (follower_high - offset) / multiplier)
        low = max(low, min(mapped))
        high = min(high, max(mapped))
    if low > high:
        raise ValueError(f"no feasible operational target interval for {driver_name}: [{low}, {high}]")
    return float(low), float(high)


def choose_step_target(runtime: dict[str, Any], driver_name: str, start: float) -> tuple[float, float]:
    low, high = driver_interval(runtime, driver_name)
    for delta in (STEP_RAD, -STEP_RAD):
        candidate = start + delta
        if low <= candidate <= high:
            return candidate, delta
    raise ValueError(f"no source-valid {STEP_RAD} rad step fits the operational interval for {driver_name}")


def tested_joint_record(runtime: dict[str, Any], state: dict[str, Any], joint_name: str) -> dict[str, Any]:
    model, data = runtime["model"], runtime["data"]
    joint = runtime["joints"][joint_name]
    if joint_name in state["active_joint_states"]:
        measured = state["active_joint_states"][joint_name]
        target = float(measured["target_source_rad"])
        target_velocity = float(measured.get("target_velocity_source_rad_s", 0.0))
        position_error = target - float(measured["position_source_rad"])
        velocity_error = target_velocity - float(measured["velocity_source_rad_s"])
        driver_name = joint_name
        relation_error = None
    else:
        measured = state["mimic_relations"][joint_name]
        target = float(measured["follower_target_source_rad"])
        target_velocity = float(measured["follower_target_velocity_source_rad_s"])
        position_error = target - float(measured["follower_position_source_rad"])
        velocity_error = target_velocity - float(measured["follower_velocity_source_rad_s"])
        driver_name = str(measured["driver_joint"])
        relation_error = float(measured["source_relation_error_rad"])

    dof = int(joint["dof"])
    aid = runtime["actuator_by_joint"][int(joint["joint_id"])]
    gear = float(model.actuator_gear[aid, 0])
    profile = runtime["controller_profile"][joint_name]
    return {
        "joint": joint_name,
        "role": "follower" if relation_error is not None else "active",
        "driver_joint": driver_name,
        "q_source_rad": float(gate.source_position(model, data, joint)),
        "qvel_source_rad_s": float(gate.source_velocity(data, joint)),
        "qacc_source_rad_s2": int(joint["axis_sign"]) * float(data.qacc[dof]),
        "target_source_rad": target,
        "target_velocity_source_rad_s": target_velocity,
        "position_error_target_minus_q_rad": position_error,
        "velocity_error_target_minus_qvel_rad_s": velocity_error,
        "source_relation_error_rad": relation_error,
        "kp_nm_per_rad": float(profile["kp_nm_per_rad"]),
        "kd_nms_per_rad": float(profile["kv_nms_per_rad"]),
        "pd_torque_nm": float(measured["pd_torque_nm"]),
        "bias_feedforward_torque_nm": float(measured["bias_feedforward_nm"]),
        "requested_torque_nm": float(measured["requested_torque_nm"]),
        "commanded_torque_nm": float(measured["commanded_torque_nm"]),
        "actual_generalized_torque_nm": float(data.actuator_force[aid]) * gear,
        "qfrc_bias_nm": float(data.qfrc_bias[dof]),
        "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
        "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
        "effective_inertia_kg_m2_equiv": float(profile["effective_joint_inertia_kg_m2_equiv"]),
        "source_lower_rad": float(joint["lower"]),
        "source_upper_rad": float(joint["upper"]),
        "source_velocity_limit_rad_s": float(joint["velocity_limit"]),
        "source_effort_limit_nm": float(joint["effort_limit"]),
        "actuator_gear": gear,
    }


def case_definitions(runtime: dict[str, Any]) -> list[dict[str, str]]:
    active = set(runtime["transmission"].active_joint_names)
    followers = set(runtime["transmission"].follower_joint_names)
    relations = {
        str(row["follower_joint"]): row
        for row in runtime["details"]["mimic_relations"]
    }
    requested = [
        ("right_index_dip_follower", "R_index_dip_joint", "follower"),
        ("right_thumb_pip_follower", "R_thumb_pip_joint", "follower"),
        ("right_index_active_driver", "R_index_pip_joint", "active"),
        ("right_thumb_active_driver", "R_thumb_abad_joint", "active"),
    ]
    result = []
    for label, joint_name, expected_role in requested:
        if expected_role == "follower":
            if joint_name not in followers or joint_name not in relations:
                raise ValueError(f"representative follower missing from compiled model: {joint_name}")
            driver = str(relations[joint_name]["driver_joint"])
        else:
            if joint_name not in active:
                raise ValueError(f"representative active coordinate missing from compiled model: {joint_name}")
            driver = joint_name
        result.append({"label": label, "joint": joint_name, "driver": driver, "role": expected_role})
    return result


def run_case(case: dict[str, str], trace_path: Path) -> dict[str, Any]:
    runtime = gate.make_runtime()
    model, data, transmission = runtime["model"], runtime["data"], runtime["transmission"]
    initial = gate.diagnostics(runtime)
    if initial["robot_self_contacts"] or float(initial["max_self_penetration_m"]) > 1e-12:
        return {"label": case["label"], "passed": False, "first_failed_gate": "initial self-penetration", "initial": initial, "steps": 0}
    if initial["contacts"]:
        return {"label": case["label"], "passed": False, "first_failed_gate": "initial contact in no-contact test", "initial": initial, "steps": 0}
    if initial["position_limit_violations"] or initial["velocity_limit_violations"] or not initial["finite"]:
        return {"label": case["label"], "passed": False, "first_failed_gate": "invalid initial state", "initial": initial, "steps": 0}

    open_targets = dict(runtime["initial_active"])
    start = float(open_targets[case["driver"]])
    step_target, delta = choose_step_target(runtime, case["driver"], start)
    step_targets = dict(open_targets)
    step_targets[case["driver"]] = step_target
    phases = [
        ("open_hold_2s", OPEN_HOLD_STEPS, open_targets),
        ("source_step_hold_0p4s", STEP_HOLD_STEPS, step_targets),
        ("return_hold_0p4s", RETURN_HOLD_STEPS, open_targets),
    ]
    relation_target_changes = {
        str(row["follower_joint"]): float(row["multiplier"]) * delta
        for row in runtime["details"]["mimic_relations"]
        if str(row["driver_joint"]) == case["driver"]
    }
    summary: dict[str, Any] = {
        "label": case["label"],
        "tested_joint": case["joint"],
        "tested_role": case["role"],
        "only_commanded_source_coordinate": case["driver"],
        "follower_commanded_independently": False,
        "operational_open_target_rad": start,
        "source_valid_step_delta_rad": delta,
        "step_target_rad": step_target,
        "derived_follower_target_deltas_rad": relation_target_changes,
        "initial_self_contact_count": len(initial["robot_self_contacts"]),
        "initial_contact_count": len(initial["contacts"]),
        "max_source_relation_error_rad": 0.0,
        "max_active_or_follower_target_error_rad": 0.0,
        "max_abs_qvel_rad_s": 0.0,
        "max_abs_qacc_rad_s2": 0.0,
        "max_tested_joint_abs_qvel_rad_s": 0.0,
        "max_tested_joint_abs_qacc_rad_s2": 0.0,
        "max_tested_joint_abs_torque_nm": 0.0,
        "max_tested_joint_effort_fraction": 0.0,
        "limit_violation_count": 0,
        "velocity_violation_count": 0,
        "nonfinite_seen": False,
        "contact_seen": False,
        "first_failed_gate": None,
        "steps": 0,
        "duration_s": 0.0,
        "active_rollout_qpos_writes": 0,
        "follower_rollout_qpos_writes": 0,
        "initialization_qpos_writes_before_t0": runtime["initialization_qpos_write_count"],
    }
    response_rows: dict[str, list[dict[str, float]]] = {phase: [] for phase, _, _ in phases}
    first_failure = None
    step_index = 0

    with trace_path.open("w", encoding="utf-8") as trace_file:
        for phase, duration, command_targets in phases:
            for _ in range(duration):
                transmission.command_active_sources(command_targets)
                transmission.write_internal_controls()
                mujoco.mj_step(model, data)
                step_index += 1
                state = gate.diagnostics(runtime)
                tested = tested_joint_record(runtime, state, case["joint"])
                row = {
                    "trial": case["label"],
                    "phase": phase,
                    "step": step_index,
                    "simulation_time_s": float(data.time),
                    "tested_joint": tested,
                    "all_hand_max_abs_qvel_rad_s": float(state["max_abs_qvel_rad_s"]),
                    "all_model_max_abs_qacc_rad_s2": float(state["max_abs_qacc_rad_s2"]),
                    "active_target_tracking_error_max_rad": float(state["active_target_tracking_error_max_rad"]),
                    "follower_target_tracking_error_max_rad": float(state["follower_target_tracking_error_max_rad"]),
                    "source_relation_error_max_rad": float(state["source_relation_error_max_rad"]),
                    "position_limit_violations": state["position_limit_violations"],
                    "velocity_limit_violations": state["velocity_limit_violations"],
                    "contacts": state["contacts"],
                    "robot_self_contacts": state["robot_self_contacts"],
                    "actuator_effort_violation": bool(state["actuator_effort_violation"]),
                    "finite": bool(state["finite"]),
                    "active_rollout_qpos_writes": 0,
                    "follower_rollout_qpos_writes": 0,
                }
                trace_file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                trace_file.flush()

                summary["steps"] = step_index
                summary["duration_s"] = float(data.time)
                summary["max_source_relation_error_rad"] = max(summary["max_source_relation_error_rad"], float(state["source_relation_error_max_rad"]))
                summary["max_active_or_follower_target_error_rad"] = max(summary["max_active_or_follower_target_error_rad"], float(state["target_tracking_error_max_rad"]))
                summary["max_abs_qvel_rad_s"] = max(summary["max_abs_qvel_rad_s"], float(state["max_abs_qvel_rad_s"]))
                summary["max_abs_qacc_rad_s2"] = max(summary["max_abs_qacc_rad_s2"], float(state["max_abs_qacc_rad_s2"]))
                summary["max_tested_joint_abs_qvel_rad_s"] = max(summary["max_tested_joint_abs_qvel_rad_s"], abs(float(tested["qvel_source_rad_s"])))
                summary["max_tested_joint_abs_qacc_rad_s2"] = max(summary["max_tested_joint_abs_qacc_rad_s2"], abs(float(tested["qacc_source_rad_s2"])))
                summary["max_tested_joint_abs_torque_nm"] = max(summary["max_tested_joint_abs_torque_nm"], abs(float(tested["actual_generalized_torque_nm"])))
                effort = float(tested["source_effort_limit_nm"])
                summary["max_tested_joint_effort_fraction"] = max(summary["max_tested_joint_effort_fraction"], abs(float(tested["actual_generalized_torque_nm"])) / effort if effort > 0 else float("inf"))
                summary["limit_violation_count"] += len(state["position_limit_violations"])
                summary["velocity_violation_count"] += len(state["velocity_limit_violations"])
                summary["nonfinite_seen"] = bool(summary["nonfinite_seen"]) or not bool(state["finite"])
                summary["contact_seen"] = bool(summary["contact_seen"]) or bool(state["contacts"])
                response_rows[phase].append({
                    "q": float(tested["q_source_rad"]),
                    "target": float(tested["target_source_rad"]),
                    "error": float(tested["target_source_rad"]) - float(tested["q_source_rad"]),
                    "qvel": float(tested["qvel_source_rad_s"]),
                })

                failed_gate = None
                if state["position_limit_violations"]:
                    failed_gate = "source position limit"
                elif state["velocity_limit_violations"]:
                    failed_gate = "source velocity limit"
                elif not state["finite"]:
                    failed_gate = "nonfinite state"
                elif state["contacts"]:
                    failed_gate = "contact in no-contact validation"
                elif state["actuator_effort_violation"]:
                    failed_gate = "actuator effort exceeded source/simulation bound"
                elif float(state["source_relation_error_max_rad"]) >= TRACKING_ABORT_RAD:
                    failed_gate = "source mimic residual reached 0.010 rad bound"
                elif float(state["target_tracking_error_max_rad"]) >= TRACKING_ABORT_RAD:
                    failed_gate = "active/follower tracking reached 0.010 rad bound"
                elif float(state["max_self_penetration_m"]) > 1e-12:
                    failed_gate = "self-penetration"
                if failed_gate:
                    summary["first_failed_gate"] = failed_gate
                    first_failure = {"phase": phase, "step": step_index, "time_s": float(data.time), "gate": failed_gate, "state": row}
                    break
            if summary["first_failed_gate"]:
                break

    step_response = response_rows["source_step_hold_0p4s"]
    return_response = response_rows["return_hold_0p4s"]
    signed_overshoot = [
        (float(sample["q"]) - float(sample["target"])) * np.sign(delta)
        for sample in step_response
    ]
    overshoot_rad = max(0.0, max(signed_overshoot, default=0.0))
    final_step_error = abs(float(step_response[-1]["error"])) if step_response else None
    final_return_error = abs(float(return_response[-1]["error"])) if return_response else None
    step_error_signs = [np.sign(float(sample["error"])) for sample in step_response if abs(float(sample["error"])) > 1e-6]
    crossings = sum(1 for left, right in zip(step_error_signs, step_error_signs[1:]) if left != right)
    tolerance = SETTLE_FRACTION * abs(delta)
    summary["response"] = {
        "overshoot_rad": overshoot_rad,
        "overshoot_fraction_of_step": overshoot_rad / abs(delta),
        "step_hold_final_abs_error_rad": final_step_error,
        "return_hold_final_abs_error_rad": final_return_error,
        "step_response_error_sign_crossings": crossings,
        "settle_tolerance_rad": tolerance,
    }
    if summary["first_failed_gate"] is None:
        if final_step_error is None or final_step_error > tolerance:
            summary["first_failed_gate"] = "step response did not settle within 2% in 0.4 s"
        elif final_return_error is None or final_return_error > tolerance:
            summary["first_failed_gate"] = "return response did not settle within 2% in 0.4 s"
        elif overshoot_rad > OVERSHOOT_FRACTION * abs(delta):
            summary["first_failed_gate"] = "step response overshoot exceeded 5%"
        elif crossings > 1:
            summary["first_failed_gate"] = "step response showed repeated error sign reversals"
    summary["passed"] = summary["first_failed_gate"] is None and summary["steps"] == OPEN_HOLD_STEPS + STEP_HOLD_STEPS + RETURN_HOLD_STEPS
    summary["first_failure"] = first_failure
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = gate.make_runtime()
    controller_profile = runtime["controller_profile"]
    native_library = Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"
    vendor_head = subprocess.run(
        ["git", "-C", str(gate.task.ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    robotsim_head = subprocess.run(
        ["git", "-C", str(gate.task.SIM_REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    identity = {
        "python_version": sys.version,
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_library": str(native_library),
        "mujoco_native_library_sha256": hashlib.sha256(native_library.read_bytes()).hexdigest(),
        "vendor_checkout": str(gate.task.ROOT),
        "vendor_head": vendor_head,
        "vendor_source_pin": gate.task.SOURCE_PIN,
        "pinned_urdf": str(gate.task.URDF),
        "pinned_urdf_sha256": hashlib.sha256(gate.task.URDF.read_bytes()).hexdigest(),
        "robotsim_checkout": str(gate.task.SIM_REPO_ROOT),
        "robotsim_head": robotsim_head,
    }
    (OUT / "runtime_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    design = {
        "model_label": runtime["details"]["mimic_implementation"],
        "source_pin": gate.task.SOURCE_PIN,
        "runtime_identity": identity,
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(runtime["model"].opt.timestep),
        "sample_rate_hz": 1.0 / float(runtime["model"].opt.timestep),
        "controller_design": {
            "nominal_wn_dt": gate.CONTROLLER_WN_DT_TARGET,
            "nominal_wn_rad_s": gate.CONTROLLER_WN_DT_TARGET / float(runtime["model"].opt.timestep),
            "damping_ratio": gate.CONTROLLER_DAMPING_RATIO,
            "inertia_method": "I_eff = 1/(M^-1)[dof,dof] from the full compiled MuJoCo articulated mass matrix at clean operational OPEN; this is the diagonal inverse-mass effective inertia with all other coordinates free to respond.",
            "gain_rule": "kp = I_eff*wn^2; kd = 2*zeta*I_eff*wn.",
            "effort_cap_rule": "wn is reduced only if needed so kp*soft_limit_margin <= min(source effort, existing simulation torque bound)-abs(open qfrc_bias). No source limits or actuator bounds are raised.",
            "discrete_validation_model": "semi-implicit sampled double integrator with a zero-order-held PD command; its poles are recorded per DOF, then the actual MuJoCo 3.3.6 dt=0.002 step response is tested.",
            "active_rollout_follower_qpos_writes": 0,
        },
        "profile_by_joint": controller_profile,
    }
    (OUT / "controller_design.json").write_text(json.dumps(design, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    cases = case_definitions(runtime)
    trial_results = []
    for case in cases:
        result = run_case(case, OUT / f"{case['label']}.jsonl")
        trial_results.append(result)
        if not result["passed"]:
            break

    all_pass = len(trial_results) == len(cases) and all(row["passed"] for row in trial_results)
    result = {
        "status": "PASS" if all_pass else "FAIL",
        "first_failed_gate": next((row["first_failed_gate"] for row in trial_results if not row["passed"]), None),
        "source_pin": gate.task.SOURCE_PIN,
        "mujoco_version": mujoco.__version__,
        "python_version": sys.version,
        "timestep_s": float(runtime["model"].opt.timestep),
        "stage": "2: representative source-coordinate step response, clean reset per case",
        "case_results": trial_results,
        "exact_command_template": "AGIBOT_X2_VENDOR_ROOT=<pinned checkout> ISSUE46_EVIDENCE_DIR=<evidence directory> <venv>/bin/python scripts/research/issue46_virtual_transmission_timestep_validation.py",
        "no_bottle_or_external_contact": True,
        "active_rollout_follower_qpos_writes": 0,
    }
    result_path = OUT / "stage2_single_dof_validation.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result_path": str(result_path), **{key: result[key] for key in ("status", "first_failed_gate", "timestep_s")}}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
