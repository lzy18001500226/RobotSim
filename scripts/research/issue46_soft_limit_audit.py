from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import mujoco

import issue46_virtual_transmission_gate as gate


def main() -> int:
    out = Path(os.environ.get("ISSUE46_EVIDENCE_DIR", "/tmp/issue46-soft-limit-audit"))
    out.mkdir(parents=True, exist_ok=True)

    runtime = gate.make_runtime()
    model, data = runtime["model"], runtime["data"]
    transmission = runtime["transmission"]
    joints = runtime["joints"]
    initial = gate.diagnostics(runtime)
    initial_rows = {row["joint"]: row for row in initial["source_joint_limits"]}
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    output = transmission.write_internal_controls()
    requested_targets: dict[str, float] = {}
    for joint_name, joint in joints.items():
        if joint_name in transmission.active_joint_names:
            aid = runtime["actuator_by_joint"][int(joint["joint_id"])]
            requested_targets[joint_name] = transmission.source_position(joint_name, float(data.ctrl[aid]))
        elif joint_name in output:
            requested_targets[joint_name] = float(output[joint_name]["follower_target_source_rad"])

    controls_before_step = data.ctrl.copy()
    mujoco.mj_step(model, data)
    first = gate.diagnostics(runtime)
    first_rows = {row["joint"]: row for row in first["source_joint_limits"]}
    violations = {row["joint"] for row in first["position_limit_violations"]}

    rows = []
    for joint_name in sorted(violations):
        joint = joints[joint_name]
        aid = runtime["actuator_by_joint"][int(joint["joint_id"])]
        raw_qpos_index = int(joint["qpos"])
        dof_index = int(joint["dof"])
        low, high = float(joint["lower"]), float(joint["upper"])
        first_source = float(first_rows[joint_name]["position_rad"])
        overshoot = max(low - first_source, first_source - high, 0.0)
        target = float(requested_targets[joint_name])
        if abs(target - low) <= 1e-12:
            target_at_hard_limit = "lower"
        elif abs(target - high) <= 1e-12:
            target_at_hard_limit = "upper"
        else:
            target_at_hard_limit = None
        follower_output = output.get(joint_name, {})
        rows.append({
            "joint": joint_name,
            "source_lower_rad": low,
            "source_upper_rad": high,
            "requested_open_target_source_rad": target,
            "initial_source_qpos_rad": float(initial_rows[joint_name]["position_rad"]),
            "initial_raw_mujoco_qpos": float(initial_qpos[raw_qpos_index]),
            "first_step_source_qpos_rad": first_source,
            "first_step_raw_mujoco_qpos": float(data.qpos[raw_qpos_index]),
            "first_step_qvel_source_rad_s": gate.source_velocity(data, joint),
            "first_step_raw_mujoco_qvel": float(data.qvel[dof_index]),
            "actuator_name": gate.name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
            "actuator_role": "internal_follower" if joint_name in output else "public_active",
            "actuator_ctrl_before_step": float(controls_before_step[aid]),
            "actuator_gear": float(model.actuator_gear[aid, 0]),
            "requested_torque_nm": follower_output.get("requested_torque_nm"),
            "commanded_torque_nm": follower_output.get("commanded_torque_nm"),
            "first_step_actuator_force_nm": float(data.actuator_force[aid]) * float(model.actuator_gear[aid, 0]),
            "target_equals_hard_limit": target_at_hard_limit is not None,
            "target_hard_limit_side": target_at_hard_limit,
            "overshoot_rad": overshoot,
        })

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gate.HERE.parents[1], text=True).strip()
    except Exception:
        head = "unavailable"
    result = {
        "experiment": "single clean reset and exactly one MuJoCo step; no object contact or rollout continuation",
        "model_label": "CONTROL-LEVEL / ACTUATED MIMIC EMULATION",
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "worktree_head": head,
        "initial_time_s": float(initial["time_s"]),
        "first_step_time_s": float(first["time_s"]),
        "initial_self_contacts": initial["robot_self_contacts"],
        "initial_max_self_penetration_m": float(initial["max_self_penetration_m"]),
        "initial_mimic_error_max_rad": float(initial["source_relation_error_max_rad"]),
        "first_step_mimic_error_max_rad": float(first["source_relation_error_max_rad"]),
        "first_step_contact_count": len(first["contacts"]),
        "first_step_nan": not bool(first["finite"]),
        "first_step_velocity_limit_violations": first["velocity_limit_violations"],
        "first_step_max_abs_qacc_rad_s2": float(first["max_abs_qacc_rad_s2"]),
        "violating_joint_count": len(rows),
        "violations": rows,
    }
    path = out / "hard_limit_target_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_path": str(path), "violating_joint_count": len(rows), "max_overshoot_rad": max((row["overshoot_rad"] for row in rows), default=0.0), "first_step_contact_count": len(first["contacts"]), "mimic_error_max_rad": result["first_step_mimic_error_max_rad"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
