#!/usr/bin/env python3
"""Bounded X2 OmniPicker endpoint and constrained-controller recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
RECOVERY_PATH = HERE / "issue46_omnipicker_recovery.py"
STATIC_ROOT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready")
DEFAULT_OUTPUT = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-dynamic/20261007-constrained-open-hold-v3")

PREVIOUS_EXCURSION_RAD = 0.0001543257750263
PREVIOUS_MIMIC_RESIDUAL_RAD = 0.00020590330192338246
MARGIN_HEADROOM_FACTOR = 2.0
MARGIN_QUANTUM_RAD = 0.001
OPEN_MARGIN_RAD = math.ceil(
    MARGIN_HEADROOM_FACTOR * (PREVIOUS_EXCURSION_RAD + PREVIOUS_MIMIC_RESIDUAL_RAD) / MARGIN_QUANTUM_RAD
) * MARGIN_QUANTUM_RAD
SOURCE_OPEN_LIMIT_RAD = -1.0
OPERATIONAL_OPEN_Q_RAD = SOURCE_OPEN_LIMIT_RAD + OPEN_MARGIN_RAD
LOCAL_TORQUE_DELTA_NM = 0.004
NATURAL_FREQUENCY_RAD_S = 5.0
DAMPING_RATIO = 1.0
EFFORT_CAP_NM = 0.08
TRAJECTORY_DURATION_S = 2.5
OPEN_HOLD_DURATION_S = 2.0
SETTLE_WINDOW_S = 0.25
SETTLE_POSITION_ERROR_RAD = OPEN_MARGIN_RAD / 2.0
SETTLE_QVEL_RAD_S = 0.01


def load_recovery():
    spec = importlib.util.spec_from_file_location("issue46_recovery_helper", RECOVERY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load recovery helper: {RECOVERY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def efc_rows(model, data, equality_name: str) -> list[dict[str, float | int]]:
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
    rows = []
    for i in range(data.nefc):
        if int(data.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) and int(data.efc_id[i]) == eq_id:
            rows.append({"row": i, "position_error": float(data.efc_pos[i]), "force": float(data.efc_force[i])})
    return rows


def pair_snapshot(model, data, helper, actuator: int) -> dict[str, object]:
    driver = "right_claw_joint"
    follower = "R_hand_wide1_joint"
    dd, fd = helper.dof_id(model, driver), helper.dof_id(model, follower)
    dq, fq = helper.qpos_id(model, driver), helper.qpos_id(model, follower)
    return {
        "time_s": float(data.time),
        "driver": {
            "qpos_rad": float(data.qpos[dq]), "qvel_rad_s": float(data.qvel[dd]), "qacc_rad_s2": float(data.qacc[dd]),
            "qfrc_bias_nm": float(data.qfrc_bias[dd]), "qfrc_passive_nm": float(data.qfrc_passive[dd]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dd]), "qfrc_constraint_nm": float(data.qfrc_constraint[dd]),
        },
        "follower": {
            "qpos_rad": float(data.qpos[fq]), "qvel_rad_s": float(data.qvel[fd]), "qacc_rad_s2": float(data.qacc[fd]),
            "qfrc_bias_nm": float(data.qfrc_bias[fd]), "qfrc_passive_nm": float(data.qfrc_passive[fd]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[fd]), "qfrc_constraint_nm": float(data.qfrc_constraint[fd]),
        },
        "mimic": {
            "equality_active": bool(data.eq_active[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_picker_mimic")]),
            "residual_rad": float(data.qpos[fq] + data.qpos[dq]),
            "equality_rows": efc_rows(model, data, "right_picker_mimic"),
        },
        "actuator": {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
            "ctrl_nm": float(data.ctrl[actuator]), "actuator_force_nm": float(data.actuator_force[actuator]),
        },
    }


def new_data(model, helper, driver_q: float, ctrl_nm: float = 0.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_picker_internal_motor")
    if actuator < 0 or model.nu != 1:
        raise RuntimeError(f"Expected the sole driver actuator, got nu={model.nu}, actuator={actuator}")
    initial = helper.initial_values(model)
    initial["right_claw_joint"] = driver_q
    initial["R_hand_wide1_joint"] = -driver_q
    for name, value in initial.items():
        data.qpos[helper.qpos_id(model, name)] = value
    data.qvel[:] = 0.0
    data.ctrl[actuator] = ctrl_nm
    mujoco.mj_forward(model, data)
    return data, int(actuator)


def limited_joint_state(model, data, helper, source_limits: dict[str, tuple[float, float]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for name, (lower, upper) in source_limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or int(model.jnt_type[jid]) not in {
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        }:
            continue
        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        q = float(data.qpos[qadr])
        out[name] = {
            "qpos_rad_or_m": q,
            "qvel_rad_s_or_m_s": float(data.qvel[dadr]),
            "qacc_rad_s2_or_m_s2": float(data.qacc[dadr]),
            "source_range": [lower, upper],
            "lower_margin": q - lower,
            "upper_margin": upper - q,
            "within_source_range": lower - 1e-10 <= q <= upper + 1e-10,
        }
    return out


def capture_initial_audit(model, helper, actuator: int) -> dict[str, object]:
    data, actuator = new_data(model, helper, SOURCE_OPEN_LIMIT_RAD, 0.0)
    zero_ctrl = pair_snapshot(model, data, helper, actuator)
    old_inertia, old_bias = helper.effective_terms(model, data)
    old_profile_torque = float(old_bias)
    data.ctrl[actuator] = old_profile_torque
    mujoco.mj_forward(model, data)
    old_command = pair_snapshot(model, data, helper, actuator)
    state = limited_joint_state(model, data, helper, helper.source_limit_map(
        ET.parse(helper.URDF_ROOT / helper.X2_URDF).getroot()
    ))
    return {
        "mj_step_calls": 0,
        "exact_previous_open_state": {"driver_target_rad": -1.0, "follower_target_rad": 1.0},
        "source_limited_joint_positions_after_forward": state,
        "zero_velocity_after_reset": True,
        "state_source_valid": all(row["within_source_range"] for row in state.values()),
        "mimic_state_consistent": abs(float(zero_ctrl["mimic"]["residual_rad"])) <= 1e-12,
        "dynamically_supported_with_reset_command": max(
            abs(float(zero_ctrl["driver"]["qacc_rad_s2"])), abs(float(zero_ctrl["follower"]["qacc_rad_s2"]))
        ) < 1e-6,
        "reset_controller_state": zero_ctrl,
        "previous_profile_first_command": {
            "effective_inertia_projection_kg_m2": old_inertia,
            "bias_projection_nm": old_bias,
            "commanded_torque_nm": old_profile_torque,
            "forward_state": old_command,
            "outward_driver_acceleration_rad_s2": float(old_command["driver"]["qacc_rad_s2"]),
        },
        "conclusion": "The exact source-open reset is position-valid, mimic-consistent, and zero-velocity, but the prior controller reset does not dynamically support it. Its projected bias feedforward produces an outward driver acceleration before the first integration step.",
    }


def jaw_gap(model, helper, driver_q: float) -> float:
    data, _ = new_data(model, helper, driver_q)
    a = data.xpos[helper.body_id(model, "R_hand_narrow3_Link")]
    b = data.xpos[helper.body_id(model, "R_hand_wide3_Link")]
    return float(np.linalg.norm(a - b))


def constrained_sample(model, helper, driver_q: float, ctrl_nm: float) -> dict[str, object]:
    data, actuator = new_data(model, helper, driver_q, ctrl_nm)
    return pair_snapshot(model, data, helper, actuator)


def identify_constrained_dynamics(model, helper) -> dict[str, object]:
    q = OPERATIONAL_OPEN_Q_RAD
    base = constrained_sample(model, helper, q, 0.0)
    plus = constrained_sample(model, helper, q, LOCAL_TORQUE_DELTA_NM)
    minus = constrained_sample(model, helper, q, -LOCAL_TORQUE_DELTA_NM)
    a0 = float(base["driver"]["qacc_rad_s2"])
    af0 = float(base["follower"]["qacc_rad_s2"])
    b_driver = (float(plus["driver"]["qacc_rad_s2"]) - float(minus["driver"]["qacc_rad_s2"])) / (2.0 * LOCAL_TORQUE_DELTA_NM)
    b_follower = (float(plus["follower"]["qacc_rad_s2"]) - float(minus["follower"]["qacc_rad_s2"])) / (2.0 * LOCAL_TORQUE_DELTA_NM)
    if not math.isfinite(b_driver) or b_driver <= 0.0:
        raise RuntimeError(f"Invalid driver local command response: {b_driver}")
    effective_inertia = 1.0 / b_driver
    hold_torque = -a0 / b_driver
    hold = constrained_sample(model, helper, q, hold_torque)
    return {
        "mj_step_calls": 0,
        "model_constraint_state": "native source mimic equality active; all other robot joints retain the accepted fixed-pose equalities",
        "operational_open_qpos_rad": q,
        "small_signal_delta_nm": LOCAL_TORQUE_DELTA_NM,
        "local_command_to_acceleration": {
            "driver_acceleration_intercept_rad_s2": a0,
            "follower_acceleration_intercept_rad_s2": af0,
            "driver_daccel_dtorque_rad_s2_per_nm": b_driver,
            "follower_daccel_dtorque_rad_s2_per_nm": b_follower,
            "driver_effective_constrained_inertia_kg_m2": effective_inertia,
            "hold_torque_for_zero_driver_acceleration_nm": hold_torque,
            "driver_acceleration_at_hold_command_rad_s2": float(hold["driver"]["qacc_rad_s2"]),
            "follower_acceleration_at_hold_command_rad_s2": float(hold["follower"]["qacc_rad_s2"]),
        },
        "samples": {"zero_torque": base, "positive_delta": plus, "negative_delta": minus, "hold_command": hold},
        "interpretation": "Effective inertia is the inverse local driver acceleration response with the compiled equality and pose constraints active. It is not v^T M v and is valid only as a local SIMULATION_ONLY_M0 control model.",
    }


def derive_profile(local: dict[str, object]) -> dict[str, object]:
    local_terms = local["local_command_to_acceleration"]
    inertia = float(local_terms["driver_effective_constrained_inertia_kg_m2"])
    stroke = abs(OPERATIONAL_OPEN_Q_RAD)
    peak_v = 1.875 * stroke / TRAJECTORY_DURATION_S
    peak_a = 5.773502691896258 * stroke / TRAJECTORY_DURATION_S**2
    return {
        "classification": "SIMULATION_ONLY_M0; not AgiBot hardware ratings",
        "trajectory": {"shape": "minimum jerk", "duration_s": TRAJECTORY_DURATION_S,
                       "stroke_rad": stroke, "max_target_velocity_rad_s": peak_v,
                       "max_target_acceleration_rad_s2": peak_a},
        "local_model": "finite-difference command-to-acceleration response at operational OPEN with native equality active",
        "natural_frequency_rad_s": NATURAL_FREQUENCY_RAD_S,
        "damping_ratio": DAMPING_RATIO,
        "kp_nm_per_rad": inertia * NATURAL_FREQUENCY_RAD_S**2,
        "kd_nm_s_per_rad": 2.0 * DAMPING_RATIO * NATURAL_FREQUENCY_RAD_S * inertia,
        "equilibrium_open_feedforward_nm": float(local_terms["hold_torque_for_zero_driver_acceleration_nm"]),
        "effort_cap_nm": EFFORT_CAP_NM,
        "qvel_bound_rad_s": 1.5 * peak_v,
        "qacc_bound_rad_s2": 2.0 * peak_a,
        "law": "tau = tau_open_eq + J_eff * (target_acc + 2*zeta*wn*(target_vel-qvel) + wn^2*(target_q-qpos))",
        "reset": "qpos at operational OPEN, qvel zero, equality active, ctrl initialized to constrained open equilibrium torque before mj_forward and first mj_step",
        "source_hard_limits_unchanged": True,
    }


def run_open_hold(model, helper, profile: dict[str, object], source_limits: dict[str, tuple[float, float]], out: Path) -> dict[str, object]:
    # Recover the scalar local inertia directly from the documented critical-damping gains.
    inertia = float(profile["kd_nm_s_per_rad"]) / (2.0 * NATURAL_FREQUENCY_RAD_S)
    feedforward = float(profile["equilibrium_open_feedforward_nm"])
    model_data, actuator = new_data(model, helper, OPERATIONAL_OPEN_Q_RAD, feedforward)
    initial_contacts = helper.contact_record(model, model_data)
    initial_penetrations = [c for c in initial_contacts if float(c["penetration_m"]) > 1e-8]
    init_limits = limited_joint_state(model, model_data, helper, source_limits)
    init_failures = [name for name, row in init_limits.items() if not row["within_source_range"]]
    trace_path = out / "open_hold_trace.jsonl"
    trace: list[dict[str, object]] = []
    first_failure: dict[str, object] | None = None
    steps = round(OPEN_HOLD_DURATION_S / helper.TIMESTEP)
    max_qvel = 0.0
    max_qacc = 0.0
    max_effort = 0.0
    max_residual = 0.0
    violations: list[dict[str, object]] = []
    if initial_penetrations:
        first_failure = {"gate": "initial penetration", "contacts": initial_penetrations}
    elif init_failures:
        first_failure = {"gate": "initial source position range", "joints": init_failures}
    elif abs(feedforward) > EFFORT_CAP_NM:
        first_failure = {"gate": "open equilibrium torque exceeds effort cap", "torque_nm": feedforward}

    with trace_path.open("w", encoding="utf-8") as stream:
        if first_failure is None:
            for i in range(steps):
                qd = float(model_data.qpos[helper.qpos_id(model, "right_claw_joint")])
                vd = float(model_data.qvel[helper.dof_id(model, "right_claw_joint")])
                # The target is stationary at the operational OPEN endpoint.
                tau_raw = feedforward + inertia * NATURAL_FREQUENCY_RAD_S**2 * (OPERATIONAL_OPEN_Q_RAD - qd)
                tau = float(np.clip(tau_raw, -EFFORT_CAP_NM, EFFORT_CAP_NM))
                model_data.ctrl[actuator] = tau
                mujoco.mj_step(model, model_data)
                state = limited_joint_state(model, model_data, helper, source_limits)
                out_of_range = [
                    {"joint": name, **row} for name, row in state.items() if not row["within_source_range"]
                ]
                if out_of_range:
                    violations.extend(out_of_range)
                pair = pair_snapshot(model, model_data, helper, actuator)
                contacts = helper.contact_record(model, model_data)
                qvel_values = [abs(float(pair[k]["qvel_rad_s"])) for k in ("driver", "follower")]
                qacc_values = [abs(float(pair[k]["qacc_rad_s2"])) for k in ("driver", "follower")]
                max_qvel = max(max_qvel, *qvel_values)
                max_qacc = max(max_qacc, *qacc_values)
                max_effort = max(max_effort, abs(float(model_data.actuator_force[actuator])))
                residual = abs(float(pair["mimic"]["residual_rad"]))
                max_residual = max(max_residual, residual)
                row = {
                    "step": i + 1, "time_s": float(model_data.time), "phase": "OPEN_HOLD",
                    "target_aperture_ratio": 1.0, "driver_target_rad": OPERATIONAL_OPEN_Q_RAD,
                    "follower_target_rad": -OPERATIONAL_OPEN_Q_RAD,
                    "torque_unclipped_nm": tau_raw, "torque_command_nm": tau,
                    "max_abs_picker_qvel_rad_s": max(qvel_values), "max_abs_picker_qacc_rad_s2": max(qacc_values),
                    "max_abs_mimic_residual_rad": residual, "contacts": contacts,
                    "source_limited_joints": state, "pair_state": pair,
                }
                trace.append(row)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if out_of_range:
                    first_failure = {"gate": "source hard position limit", "step": i + 1, "joints": out_of_range}
                elif not np.isfinite([*qvel_values, *qacc_values, tau, residual]).all():
                    first_failure = {"gate": "finite state", "step": i + 1}
                elif max_qvel > float(profile["qvel_bound_rad_s"]):
                    first_failure = {"gate": "bounded qvel", "step": i + 1, "observed_rad_s": max_qvel,
                                     "bound_rad_s": profile["qvel_bound_rad_s"]}
                elif max_qacc > float(profile["qacc_bound_rad_s2"]):
                    first_failure = {"gate": "bounded qacc", "step": i + 1, "observed_rad_s2": max_qacc,
                                     "bound_rad_s2": profile["qacc_bound_rad_s2"]}
                elif max_effort > EFFORT_CAP_NM + 1e-10:
                    first_failure = {"gate": "bounded actuator effort", "step": i + 1,
                                     "observed_nm": max_effort, "bound_nm": EFFORT_CAP_NM}
                elif contacts:
                    first_failure = {"gate": "no-contact open hold", "step": i + 1, "contacts": contacts}
                if first_failure:
                    break

    window_steps = round(SETTLE_WINDOW_S / helper.TIMESTEP)
    errors = np.array([float(row["pair_state"]["driver"]["qpos_rad"]) - OPERATIONAL_OPEN_Q_RAD for row in trace])
    velocities = np.array([float(row["pair_state"]["driver"]["qvel_rad_s"]) for row in trace])
    mimic = np.array([float(row["pair_state"]["mimic"]["residual_rad"]) for row in trace])
    first_window = trace[:window_steps]
    last_window = trace[-window_steps:]
    first_err_rms = float(np.sqrt(np.mean(errors[:window_steps] ** 2))) if len(errors) >= window_steps else None
    last_err_rms = float(np.sqrt(np.mean(errors[-window_steps:] ** 2))) if len(errors) >= window_steps else None
    first_amp = float(np.max(np.abs(errors[:window_steps]))) if len(errors) >= window_steps else None
    last_amp = float(np.max(np.abs(errors[-window_steps:]))) if len(errors) >= window_steps else None
    final_error = float(abs(errors[-1])) if len(errors) else None
    final_velocity = float(abs(velocities[-1])) if len(velocities) else None
    final_mimic = float(abs(mimic[-1])) if len(mimic) else None
    no_growth = bool(last_err_rms is not None and first_err_rms is not None and last_err_rms <= first_err_rms + 1e-12)
    settled = bool(final_error is not None and final_velocity is not None and
                   float(np.max(np.abs(errors[-window_steps:]))) <= SETTLE_POSITION_ERROR_RAD and
                   float(np.max(np.abs(velocities[-window_steps:]))) <= SETTLE_QVEL_RAD_S) if len(trace) >= window_steps else False
    mimic_stable = bool(final_mimic is not None and len(mimic) >= window_steps and
                        float(np.max(np.abs(mimic[-window_steps:]))) <=
                        float(np.max(np.abs(mimic[:window_steps]))) + 1e-6)
    if first_failure is None and not no_growth:
        first_failure = {"gate": "no growing oscillation", "first_window_error_rms_rad": first_err_rms,
                         "final_window_error_rms_rad": last_err_rms}
    if first_failure is None and not settled:
        first_failure = {"gate": "final settling", "position_error_rad": final_error,
                         "velocity_rad_s": final_velocity, "position_bound_rad": SETTLE_POSITION_ERROR_RAD,
                         "velocity_bound_rad_s": SETTLE_QVEL_RAD_S}
    if first_failure is None and not mimic_stable:
        first_failure = {"gate": "mimic residual stability", "first_window_max_rad": float(np.max(np.abs(mimic[:window_steps]))) if len(mimic) else None,
                         "final_window_max_rad": float(np.max(np.abs(mimic[-window_steps:]))) if len(mimic) else None}
    failure_diagnostic = None
    if first_failure and trace:
        last = trace[-1]
        qpos = model_data.qpos.copy()
        qvel = model_data.qvel.copy()
        actual_ctrl = float(model_data.ctrl[actuator])
        cases = {}
        for label, ctrl in (("zero_torque", 0.0), ("last_controller_command", actual_ctrl)):
            diagnostic = mujoco.MjData(model)
            mujoco.mj_resetData(model, diagnostic)
            diagnostic.qpos[:] = qpos
            diagnostic.qvel[:] = qvel
            diagnostic.time = float(model_data.time)
            diagnostic.eq_active[:] = model_data.eq_active
            diagnostic.ctrl[actuator] = ctrl
            mujoco.mj_forward(model, diagnostic)
            cases[label] = pair_snapshot(model, diagnostic, helper, actuator)
        failure_diagnostic = {
            "classification": "one bounded no-step force/acceleration decomposition at the first failed state",
            "mj_step_calls": 0,
            "failure_time_s": float(model_data.time),
            "failure_step": int(last["step"]),
            "failure_gate": first_failure,
            "cases": cases,
        }
        dump_json(out / "open_hold_failure_diagnostic.json", failure_diagnostic)

    return {
        "status": "PASS" if first_failure is None and len(trace) == steps else "FAIL",
        "first_failed_gate": first_failure,
        "duration_s": OPEN_HOLD_DURATION_S,
        "steps_expected": steps,
        "steps_completed": len(trace),
        "max_abs_picker_qvel_rad_s": max_qvel,
        "max_abs_picker_qacc_rad_s2": max_qacc,
        "max_applied_effort_nm": max_effort,
        "max_mimic_residual_rad": max_residual,
        "source_position_violations": violations,
        "finite_trace": bool(all(np.isfinite([
            float(row["max_abs_picker_qvel_rad_s"]), float(row["max_abs_picker_qacc_rad_s2"]),
            float(row["max_abs_mimic_residual_rad"]),
        ]).all() for row in trace)),
        "initial_contacts": initial_contacts,
        "no_growing_oscillation": no_growth,
        "settled": settled,
        "mimic_residual_stable": mimic_stable,
        "settling_metrics": {"window_s": SETTLE_WINDOW_S, "position_tolerance_rad": SETTLE_POSITION_ERROR_RAD,
                             "velocity_tolerance_rad_s": SETTLE_QVEL_RAD_S,
                             "first_window_error_rms_rad": first_err_rms, "final_window_error_rms_rad": last_err_rms,
                             "first_window_peak_error_rad": first_amp, "final_window_peak_error_rad": last_amp,
                             "final_error_rad": final_error, "final_velocity_rad_s": final_velocity,
                             "final_mimic_residual_rad": final_mimic},
        "qvel_bound_rad_s": profile["qvel_bound_rad_s"],
        "qacc_bound_rad_s2": profile["qacc_bound_rad_s2"],
        "effort_cap_nm": EFFORT_CAP_NM,
        "active_rollout_follower_qpos_writes": 0,
        "trace": trace_path.name,
        "failure_diagnostic": failure_diagnostic,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--x1-infer-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    helper = load_recovery()
    if args.source_root:
        helper.URDF_ROOT = args.source_root.resolve()
    if args.sdk_root:
        helper.SDK_ROOT = args.sdk_root.resolve()
    if args.x1_infer_root:
        helper.X1_ROOT = args.x1_infer_root.resolve()
    urdf_path = helper.URDF_ROOT / helper.X2_URDF
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source, _ = helper.source_joint_audit(urdf_path)
    x2_identity = {"repository": "https://github.com/AgibotTech/agibot_x2_urdf", "commit": helper.git_head(helper.URDF_ROOT),
                   "expected_commit": helper.X2_PIN, "working_tree_clean": helper.git_clean(helper.URDF_ROOT),
                   "urdf_path": str(helper.X2_URDF), "urdf_sha256": helper.sha256(urdf_path),
                   "expected_urdf_sha256": helper.X2_URDF_SHA}
    sdk_identity = helper.verify_pin(helper.SDK_ROOT, helper.SDK_PIN, helper.SDK_HEADER, helper.SDK_HEADER_SHA)
    x1_identity = helper.verify_pin(helper.X1_ROOT, helper.X1_PIN, helper.X1_HEADER, helper.X1_HEADER_SHA)
    for identity in (x2_identity, sdk_identity, x1_identity):
        if identity["commit"] != identity["expected_commit"] or not identity["working_tree_clean"]:
            raise RuntimeError(f"Pinned source checkout mismatch or dirty tree: {identity}")
    if x2_identity["urdf_sha256"] != helper.X2_URDF_SHA:
        raise RuntimeError(f"Pinned X2 URDF hash mismatch: {x2_identity}")
    dump_json(out / "source_audit.json", {"X2": x2_identity, "OP1_SDK": sdk_identity, "X1_inference_API": x1_identity, "source": source})

    model = helper.build_spec().compile()
    if model.nu != 1 or model.neq < 3:
        raise RuntimeError(f"Expected one driver actuator and active source/rest equalities; got nu={model.nu} neq={model.neq}")
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_picker_internal_motor")
    source_limits = helper.source_limit_map(__import__("xml.etree.ElementTree", fromlist=["ElementTree"]).parse(urdf_path).getroot())

    initial_audit = capture_initial_audit(model, helper, int(actuator))
    dump_json(out / "initial_state_audit.json", initial_audit)
    margin = {
        "source_hard_open_limit_rad": SOURCE_OPEN_LIMIT_RAD,
        "measured_previous_first_step_excursion_rad": PREVIOUS_EXCURSION_RAD,
        "measured_previous_mimic_residual_rad": PREVIOUS_MIMIC_RESIDUAL_RAD,
        "combined_observed_envelope_rad": PREVIOUS_EXCURSION_RAD + PREVIOUS_MIMIC_RESIDUAL_RAD,
        "headroom_factor": MARGIN_HEADROOM_FACTOR,
        "deterministic_rounding_quantum_rad": MARGIN_QUANTUM_RAD,
        "derived_open_margin_rad": OPEN_MARGIN_RAD,
        "margin_over_combined_envelope_rad": OPEN_MARGIN_RAD - (PREVIOUS_EXCURSION_RAD + PREVIOUS_MIMIC_RESIDUAL_RAD),
        "margin_headroom_above_factor_two_envelope_rad": OPEN_MARGIN_RAD - MARGIN_HEADROOM_FACTOR * (PREVIOUS_EXCURSION_RAD + PREVIOUS_MIMIC_RESIDUAL_RAD),
        "operational_open_target_rad": OPERATIONAL_OPEN_Q_RAD,
        "follower_target_rad": -OPERATIONAL_OPEN_Q_RAD,
        "aperture_api": {"closed_ratio": 0.0, "open_ratio": 1.0,
                          "driver_target_equation": f"q_driver = {OPERATIONAL_OPEN_Q_RAD} * aperture_ratio",
                          "follower_target_equation": "q_follower = -q_driver from source mimic equality"},
        "source_limits_unchanged": True,
        "classification": "SIMULATION_ONLY_M0 operational target margin; not a source or hardware limit",
        "visual_jaw_gap": {
            "source_open_link_origin_separation_m": jaw_gap(model, helper, SOURCE_OPEN_LIMIT_RAD),
            "operational_open_link_origin_separation_m": jaw_gap(model, helper, OPERATIONAL_OPEN_Q_RAD),
            "separation_change_mm": 1000.0 * (jaw_gap(model, helper, OPERATIONAL_OPEN_Q_RAD) - jaw_gap(model, helper, SOURCE_OPEN_LIMIT_RAD)),
            "metric_note": "jaw link-origin separation, not fingertip surface aperture",
        },
    }
    dump_json(out / "operational_open_margin.json", margin)

    local = identify_constrained_dynamics(model, helper)
    dump_json(out / "constrained_local_dynamics.json", local)
    profile = derive_profile(local)
    dump_json(out / "controller_profile.json", profile)
    stage5 = run_open_hold(model, helper, profile, source_limits, out)
    stage_status = {"stage_1_initial_state_consistency": "PASS" if initial_audit["state_source_valid"] and
                    initial_audit["mimic_state_consistent"] and initial_audit["zero_velocity_after_reset"] else "FAIL",
                    "stage_2_operational_open_margin": "PASS",
                    "stage_3_constrained_local_dynamics": "PASS" if abs(float(profile["equilibrium_open_feedforward_nm"])) <= EFFORT_CAP_NM else "FAIL",
                    "stage_4_controller_derivation": "PASS",
                    "stage_5_open_hold_2s": stage5["status"],
                    "stage_6_no_contact_close_reopen": "NOT_RUN", "stage_7_fixed_object_contact": "NOT_RUN",
                    "stage_8_bottle_hold_and_lifts": "NOT_RUN"}
    result = {
        "result": "PASS_OPEN_HOLD" if stage5["status"] == "PASS" else "FAIL_OPEN_HOLD",
        "task": "Issue #46 X2 OmniPicker endpoint/constrained-controller recovery",
        "source_identity": {"X2": x2_identity, "OP1_SDK": sdk_identity, "X1_inference_API": x1_identity},
        "runtime": {"python": platform.python_version(), "mujoco_python": mujoco.__version__,
                    "mujoco_native": mujoco.mj_versionString(),
                    "native_library": str(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "native_library_sha256": helper.sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
                    "timestep_s": helper.TIMESTEP,
                    "model_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq}},
        "metadata": {"source_velocity_effort_values_preserved": True,
                     "interpretation": "UNSPECIFIED_DYNAMIC_METADATA; not zero-speed/zero-effort hardware bounds"},
        "operational_open_margin": margin,
        "constrained_local_dynamics": local,
        "controller_profile": profile,
        "stages": stage_status,
        "open_hold": stage5,
        "forbidden_operations": {"active_rollout_follower_qpos_writes": 0, "object_qpos_writes": 0,
                                  "bottle_weld_or_equality": False, "mocap_or_follow_hand": False,
                                  "opposite_hand_assistance": False},
        "conditional_stop": "If OPEN HOLD fails, exactly one no-step diagnosis was captured and no CLOSE/contact stage ran." if stage5["status"] != "PASS" else "OPEN HOLD passed; next stages require continuation without repeating it.",
        "claim_boundary": "The X2 OmniPicker position topology and mimic relation are source-derived. The velocity/effort bounds used here are SIMULATION_ONLY_M0 engineering parameters because the pinned X2 URDF provides zero placeholders and an exact X2 hardware dynamic calibration has not been established.",
    }
    dump_json(out / "result.json", result)
    for name in ("omnipicker_overview.png", "omnipicker_front.png", "omnipicker_side.png", "right_omnipicker_closeup.png",
                 "right_omnipicker_open.png", "right_omnipicker_closed.png"):
        shutil.copy2(STATIC_ROOT / name, out / name)
    command = (f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} --source-root {helper.URDF_ROOT} "
               f"--sdk-root {helper.SDK_ROOT} --x1-infer-root {helper.X1_ROOT} --output {out}")
    (out / "experiment_command.txt").write_text(command + "\n", encoding="utf-8")
    log = [f"COMMAND={command}", f"RESULT={result['result']}", f"OPEN_MARGIN_RAD={OPEN_MARGIN_RAD:.12g}",
           f"OPEN_TARGET_RAD={OPERATIONAL_OPEN_Q_RAD:.12g}",
           f"J_EFFECTIVE_CONSTRAINED={profile['local_model']}; {local['local_command_to_acceleration']['driver_effective_constrained_inertia_kg_m2']:.12g} kg*m^2",
           f"OPEN_EQUILIBRIUM_TORQUE_NM={profile['equilibrium_open_feedforward_nm']:.12g}",
           f"OPEN_HOLD={stage5['status']} steps={stage5['steps_completed']}/{stage5['steps_expected']}",
           f"FIRST_FAILURE={stage5['first_failed_gate']}"]
    (out / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0 if stage5["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
