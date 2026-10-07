from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_virtual_transmission_gate as gate  # noqa: E402


OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-open-startup-diagnostic"))
DIAGNOSTIC_STEPS = 3


def snapshot(runtime: dict[str, Any], label: str, step: int) -> dict[str, Any]:
    model, data = runtime["model"], runtime["data"]
    transmission = runtime["transmission"]
    active = transmission.active_measure()
    followers = transmission.measure()
    names = sorted(set(transmission.active_joint_names) | set(transmission.follower_joint_names))
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    dynamics_residual = mass @ data.qacc + data.qfrc_bias - (
        data.qfrc_actuator + data.qfrc_passive + data.qfrc_applied + data.qfrc_constraint
    )
    hand_rows = []
    hand_dofs = set()
    for joint_name in names:
        joint = runtime["joints"][joint_name]
        dof = int(joint["dof"])
        qpos = int(joint["qpos"])
        hand_dofs.add(dof)
        actuator_id = runtime["actuator_by_joint"][int(joint["joint_id"])]
        gear = float(model.actuator_gear[actuator_id, 0])
        measured = active[joint_name] if joint_name in active else followers[joint_name]
        profile = runtime["controller_profile"][joint_name]
        if joint_name in active:
            target = float(measured["target_source_rad"])
            target_velocity = float(measured["target_velocity_source_rad_s"])
            source_q = float(measured["position_source_rad"])
            source_qvel = float(measured["velocity_source_rad_s"])
            tracking_error = target - source_q
            relation_error = None
        else:
            target = float(measured["follower_target_source_rad"])
            target_velocity = float(measured["follower_target_velocity_source_rad_s"])
            source_q = float(measured["follower_position_source_rad"])
            source_qvel = float(measured["follower_velocity_source_rad_s"])
            tracking_error = target - source_q
            relation_error = float(measured["source_relation_error_rad"])
        hand_rows.append({
            "joint": joint_name,
            "role": "active" if joint_name in active else "follower",
            "q_source_rad": source_q,
            "qvel_source_rad_s": source_qvel,
            "qacc_source_rad_s2": int(joint["axis_sign"]) * float(data.qacc[dof]),
            "qpos_raw_rad": float(data.qpos[qpos]),
            "qvel_raw_rad_s": float(data.qvel[dof]),
            "qacc_raw_rad_s2": float(data.qacc[dof]),
            "target_source_rad": target,
            "target_velocity_source_rad_s": target_velocity,
            "target_tracking_error_rad": tracking_error,
            "source_relation_error_rad": relation_error,
            "pd_torque_nm": float(measured["pd_torque_nm"]),
            "bias_feedforward_nm": float(measured["bias_feedforward_nm"]),
            "requested_torque_nm": float(measured["requested_torque_nm"]),
            "commanded_torque_nm": float(measured["commanded_torque_nm"]),
            "actual_actuator_generalized_torque_nm": float(data.actuator_force[actuator_id]) * gear,
            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
            "qfrc_passive_nm": float(data.qfrc_passive[dof]),
            "qfrc_applied_nm": float(data.qfrc_applied[dof]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
            "M_ii": float(mass[dof, dof]),
            "effective_inertia_kg_m2_equiv": float(profile["effective_joint_inertia_kg_m2_equiv"]),
            "source_lower_rad": float(joint["lower"]),
            "source_upper_rad": float(joint["upper"]),
            "source_velocity_limit_rad_s": float(joint["velocity_limit"]),
            "source_effort_limit_nm": float(joint["effort_limit"]),
            "actuator_gear": gear,
        })

    non_hand_acceleration = []
    for dof in range(model.nv):
        if dof in hand_dofs:
            continue
        joint_id = next((jid for jid in range(model.njnt) if int(model.jnt_dofadr[jid]) <= dof < int(model.jnt_dofadr[jid]) + (6 if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE) else 1)), None)
        joint_name = None if joint_id is None else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        non_hand_acceleration.append({
            "joint": joint_name,
            "dof": dof,
            "qacc": float(data.qacc[dof]),
            "qfrc_bias": float(data.qfrc_bias[dof]),
            "qfrc_actuator": float(data.qfrc_actuator[dof]),
            "qfrc_passive": float(data.qfrc_passive[dof]),
            "qfrc_constraint": float(data.qfrc_constraint[dof]),
        })
    non_hand_acceleration.sort(key=lambda row: abs(float(row["qacc"])), reverse=True)
    return {
        "stage": label,
        "step": step,
        "time_s": float(data.time),
        "contacts": gate.contact_rows(model, data),
        "max_abs_qacc_rad_s2": float(np.max(np.abs(data.qacc))) if data.qacc.size else 0.0,
        "max_abs_qvel_rad_s": float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0,
        "dynamics_equation_residual_max_nm": float(np.max(np.abs(dynamics_residual))) if dynamics_residual.size else 0.0,
        "hand_dofs": hand_rows,
        "largest_non_hand_accelerations": non_hand_acceleration[:12],
        "active_rollout_qpos_writes": 0,
        "follower_rollout_qpos_writes": 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = gate.make_runtime()
    transmission = runtime["transmission"]
    open_targets = dict(runtime["initial_active"])
    rows = []
    rows.append(snapshot(runtime, "initialized_after_controller_setup", 0))
    for step in range(1, DIAGNOSTIC_STEPS + 1):
        transmission.command_active_sources(open_targets)
        transmission.write_internal_controls()
        mujoco.mj_forward(runtime["model"], runtime["data"])
        rows.append(snapshot(runtime, "pre_step", step - 1))
        mujoco.mj_step(runtime["model"], runtime["data"])
        rows.append(snapshot(runtime, "post_step", step))

    path = OUT / "open_startup_diagnostic.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
    limits = gate.diagnostics(runtime)
    result = {
        "status": "DIAGNOSTIC COMPLETE",
        "purpose": "one bounded three-step diagnosis after the Stage 2 open-hold source position-limit failure; no retuning",
        "timestep_s": float(runtime["model"].opt.timestep),
        "steps": DIAGNOSTIC_STEPS,
        "no_bottle_or_external_contact": True,
        "final_position_limit_violations": limits["position_limit_violations"],
        "final_velocity_limit_violations": limits["velocity_limit_violations"],
        "final_source_relation_error_max_rad": limits["source_relation_error_max_rad"],
        "final_target_tracking_error_max_rad": limits["target_tracking_error_max_rad"],
        "active_rollout_qpos_writes": 0,
        "follower_rollout_qpos_writes": 0,
        "trace_path": str(path),
    }
    (OUT / "open_startup_diagnostic_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
