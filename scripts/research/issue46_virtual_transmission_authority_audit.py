from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

import issue46_virtual_transmission_gate as gate


OUT = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-vt-authority-audit"))
PROBE_ERRORS_RAD = (0.0, 0.0015, 0.003)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = gate.make_runtime()
    model = runtime["model"]
    data = runtime["data"]
    transmission = runtime["transmission"]
    joints = runtime["joints"]
    details = runtime["details"]

    active_targets = transmission.active_target_sources()
    for joint_name, source_target in active_targets.items():
        gate.set_initial_source_position(model, data, joints[joint_name], source_target)

    follower_targets: dict[str, float] = {}
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        target = float(relation["multiplier"]) * active_targets[driver] + float(relation["offset"])
        follower_targets[follower] = target
        gate.set_initial_source_position(model, data, joints[follower], target)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    transmission.write_internal_controls()

    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    rows: list[dict[str, object]] = []
    source_bounds_pass = True
    bounded_available_authority_pass = True
    for joint_name in sorted(set(transmission.active_joint_names) | set(transmission.follower_joint_names)):
        joint = joints[joint_name]
        jid = int(joint["joint_id"])
        dof = int(joint["dof"])
        aid = runtime["actuator_by_joint"][jid]
        is_follower = joint_name in follower_targets
        target = follower_targets[joint_name] if is_follower else active_targets[joint_name]
        lower, upper = float(joint["lower"]), float(joint["upper"])
        effort_limit = float(joint["effort_limit"])
        if is_follower:
            kp = float(transmission.kp)
            kv = float(transmission.kv)
            role = "internal_follower"
        else:
            kp = max(0.0, -float(model.actuator_biasprm[aid, 1]))
            kv = max(0.0, -float(model.actuator_biasprm[aid, 2]))
            role = "public_active"
        gear = abs(float(model.actuator_gear[aid, 0]))
        if bool(model.actuator_forcelimited[aid]):
            force_cap = float(max(abs(model.actuator_forcerange[aid, 0]), abs(model.actuator_forcerange[aid, 1]))) * gear
        else:
            force_cap = float("inf")
        available_bound = min(force_cap, effort_limit)
        bias = float(data.qfrc_bias[dof])
        required_kp = abs(bias) / 0.0015
        required_pass_source = abs(bias) <= effort_limit + 1e-12
        required_pass_available = abs(bias) <= available_bound + 1e-12
        source_bounds_pass &= required_pass_source
        bounded_available_authority_pass &= required_pass_available

        full_inverse_column = np.linalg.solve(mass, np.eye(model.nv, dtype=float)[:, dof])
        effective_inertia = 1.0 / float(full_inverse_column[dof])
        current_q = gate.source_position(model, data, joint)
        current_qvel = gate.source_velocity(data, joint)
        torque_at_error = {
            f"{error:.4f}_rad": min(kp * error, available_bound)
            for error in PROBE_ERRORS_RAD
        }
        rows.append({
            "joint": joint_name,
            "role": role,
            "source_lower_rad": lower,
            "source_upper_rad": upper,
            "source_effort_limit_nm": effort_limit,
            "operational_open_target_rad": float(target),
            "open_qpos_rad": current_q,
            "open_qvel_rad_s": current_qvel,
            "qfrc_bias_nm": bias,
            "current_actuator_torque_bound_nm": force_cap,
            "effective_torque_bound_nm": available_bound,
            "current_kp_nm_per_rad": kp,
            "current_kv_nms_per_rad": kv,
            "current_pd_torque_nm_at_error": torque_at_error,
            "minimum_kp_to_counter_bias_at_0p0015_nm_per_rad": required_kp,
            "bias_torque_within_source_effort": required_pass_source,
            "bias_torque_within_effective_actuator_bound": required_pass_available,
            "required_bias_torque_nm": abs(bias),
            "mass_matrix_diagonal_kg_m2_or_kg_m2_equiv": float(mass[dof, dof]),
            "effective_joint_inertia_all_other_dofs_free_kg_m2_or_kg_m2_equiv": effective_inertia,
            "dof_address": dof,
            "qpos_address": int(joint["qpos"]),
            "actuator": gate.name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "motor_gear": float(model.actuator_gear[aid, 0]),
        })

    result = {
        "experiment": "static operational OPEN authority audit; initialized qpos and mj_forward only; no mj_step",
        "status": "PASS" if bounded_available_authority_pass else "FAIL",
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "soft_limit_margin_rad": float(gate.SOFT_LIMIT_MARGIN_RAD),
        "source_limits_modified": False,
        "rollout_steps": 0,
        "follower_qpos_writes_after_rollout_start": 0,
        "active_joint_count": len(transmission.active_joint_names),
        "follower_joint_count": len(transmission.follower_joint_names),
        "all_required_bias_torques_within_source_effort": source_bounds_pass,
        "all_required_bias_torques_within_effective_actuator_bounds": bounded_available_authority_pass,
        "required_kp_error_reference_rad": 0.0015,
        "rows": rows,
    }
    path = OUT / "controller_authority_audit.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit_path": str(path),
        "status": result["status"],
        "active_joint_count": len(transmission.active_joint_names),
        "follower_joint_count": len(transmission.follower_joint_names),
        "source_effort_pass": source_bounds_pass,
        "effective_actuator_bound_pass": bounded_available_authority_pass,
        "maximum_abs_bias_nm": max(abs(float(row["qfrc_bias_nm"])) for row in rows),
    }, indent=2))
    return 0 if bounded_available_authority_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
