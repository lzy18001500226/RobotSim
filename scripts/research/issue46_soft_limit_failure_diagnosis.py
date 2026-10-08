from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco

import issue46_virtual_transmission_gate as gate


def main() -> int:
    out = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-soft-limit-diagnosis"))
    out.mkdir(parents=True, exist_ok=True)
    runtime = gate.make_runtime()
    model, data = runtime["model"], runtime["data"]
    transmission = runtime["transmission"]
    joints = runtime["joints"]
    start = dict(runtime["initial_active"])
    initial = gate.diagnostics(runtime)
    transmission.command_active_sources(start)
    follower_outputs = transmission.write_internal_controls()
    commands = {}
    for joint_name, joint in joints.items():
        if joint_name in start:
            target = float(start[joint_name])
            role = "public_active_position_target"
            requested_torque = None
            raw_target = transmission._raw_target(joint_name, target)
        elif joint_name in follower_outputs:
            target = float(follower_outputs[joint_name]["follower_target_source_rad"])
            role = "internal_follower_position_target"
            requested_torque = float(follower_outputs[joint_name]["requested_torque_nm"])
            raw_target = None
        else:
            continue
        aid = runtime["actuator_by_joint"][int(joint["joint_id"])]
        q, dof = int(joint["qpos"]), int(joint["dof"])
        commands[joint_name] = {
            "role": role,
            "source_target_rad": target,
            "raw_qpos_target": raw_target,
            "source_axis_sign": int(joint["axis_sign"]),
            "initial_source_qpos_rad": gate.source_position(model, data, joint),
            "initial_raw_qpos": float(data.qpos[q]),
            "initial_source_qvel_rad_s": gate.source_velocity(data, joint),
            "actuator": gate.name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "actuator_type": int(model.actuator_dyntype[aid]),
            "actuator_transmission_type": int(model.actuator_trntype[aid]),
            "actuator_gear": model.actuator_gear[aid].tolist(),
            "actuator_ctrllimited": bool(model.actuator_ctrllimited[aid]),
            "actuator_ctrlrange": model.actuator_ctrlrange[aid].tolist(),
            "actuator_ctrl_command": float(data.ctrl[aid]),
            "actuator_forcelimited": bool(model.actuator_forcelimited[aid]),
            "actuator_forcerange": model.actuator_forcerange[aid].tolist(),
            "actuator_gainprm": model.actuator_gainprm[aid].tolist(),
            "actuator_biasprm": model.actuator_biasprm[aid].tolist(),
            "follower_requested_torque_nm": requested_torque,
            "dof_address": dof,
        }

    mujoco.mj_step(model, data)
    after = gate.diagnostics(runtime)
    violation_rows = []
    after_rows = {row["joint"]: row for row in after["source_joint_limits"]}
    for violation in after["position_limit_violations"]:
        joint_name = str(violation["joint"])
        joint = joints[joint_name]
        command = commands[joint_name]
        aid = runtime["actuator_by_joint"][int(joint["joint_id"])]
        dof = int(joint["dof"])
        violation_rows.append({
            **command,
            "source_lower_rad": float(joint["lower"]),
            "source_upper_rad": float(joint["upper"]),
            "first_step_source_qpos_rad": float(after_rows[joint_name]["position_rad"]),
            "first_step_raw_qpos": float(data.qpos[int(joint["qpos"])]),
            "first_step_source_qvel_rad_s": gate.source_velocity(data, joint),
            "actuator_force_nm": float(data.actuator_force[aid]) * float(model.actuator_gear[aid, 0]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
            "qfrc_passive_nm": float(data.qfrc_passive[dof]),
            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
            "qfrc_applied_nm": float(data.qfrc_applied[dof]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
            "qacc_rad_s2": float(data.qacc[dof]),
            "overshoot_rad": max(float(joint["lower"]) - float(after_rows[joint_name]["position_rad"]), float(after_rows[joint_name]["position_rad"]) - float(joint["upper"]), 0.0),
        })

    result = {
        "experiment": "single clean reset and one step, exact Stage 3 OPEN-hold command/control order; bounded failure diagnosis",
        "status": "FAIL" if violation_rows else "PASS",
        "time_s": float(data.time),
        "initial_self_contact_count": len(initial["robot_self_contacts"]),
        "initial_max_self_penetration_m": float(initial["max_self_penetration_m"]),
        "initial_position_limit_violation_count": len(initial["position_limit_violations"]),
        "initial_mimic_error_max_rad": float(initial["source_relation_error_max_rad"]),
        "first_step_contact_count": len(after["contacts"]),
        "first_step_mimic_error_max_rad": float(after["source_relation_error_max_rad"]),
        "first_step_max_abs_qacc_rad_s2": float(after["max_abs_qacc_rad_s2"]),
        "soft_limit_margin_rad": transmission.soft_limit_margin_rad,
        "source_limits_modified": False,
        "qpos_forcing_used": False,
        "violation_count": len(violation_rows),
        "violations": violation_rows,
    }
    path = out / "soft_limit_failure_diagnosis.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnosis_path": str(path),
        "status": result["status"],
        "violation_count": len(violation_rows),
        "first_step_contact_count": result["first_step_contact_count"],
        "max_abs_qacc_rad_s2": result["first_step_max_abs_qacc_rad_s2"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
