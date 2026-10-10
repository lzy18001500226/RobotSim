from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_dual_omnihand_transmission_probe as probe  # noqa: E402
import issue46_x2_grasp as task  # noqa: E402

DRIVER = "L_thumb_mcp_joint"
FOLLOWERS = {
    "L_thumb_pip_joint": 1.33,
    "L_thumb_dip_joint": 1.30,
}
OPEN_Q = -0.10
CLOSE_Q = -0.45
NATURAL_FREQUENCY_RAD_S = 8.0
MIMIC_LIMIT_RAD = 0.003
LIMIT_TOLERANCE_RAD = 1e-5
MAX_CONTACT_PENETRATION_M = 0.002
PHASES = (
    ("OPEN_HOLD", 500, OPEN_Q, OPEN_Q),
    ("CLOSE", 750, OPEN_Q, CLOSE_Q),
    ("CLOSE_HOLD", 250, CLOSE_Q, CLOSE_Q),
    ("OPEN", 750, CLOSE_Q, OPEN_Q),
    ("OPEN_SETTLE", 500, OPEN_Q, OPEN_Q),
)


def smoothstep5(fraction: float) -> float:
    x = float(np.clip(fraction, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x * x)


def source_joint_map() -> dict[str, ET.Element]:
    root = ET.parse(task.URDF).getroot()
    return {str(joint.get("name")): joint for joint in root.findall("joint") if joint.get("name")}


def reduced_column(
    nv: int,
    driver_dof: int,
    driver_axis_sign: int,
    followers: list[tuple[int, float, int]],
) -> np.ndarray:
    column = np.zeros(nv, dtype=float)
    column[driver_dof] = float(driver_axis_sign)
    for dof, multiplier, axis_sign in followers:
        column[dof] = float(axis_sign) * float(multiplier)
    return column


def state_expansion(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    base_qpos: np.ndarray,
    driver_source_q: float,
    driver_source_v: float,
    data: mujoco.MjData,
) -> int:
    """Expand one reduced state into MuJoCo coordinates for FK/dynamics."""
    data.qpos[:] = base_qpos
    probe.set_source_q(model, details, joints, DRIVER, driver_source_q, data)
    driver = probe.joint_id(model, DRIVER)
    driver_dof = int(model.jnt_dofadr[driver])
    data.qvel[:] = 0.0
    data.qvel[driver_dof] = probe.axis_sign(model, joints, DRIVER) * driver_source_v
    writes = 0
    for follower, multiplier in FOLLOWERS.items():
        probe.set_source_q(model, details, joints, follower, multiplier * driver_source_q, data)
        dof = int(model.jnt_dofadr[probe.joint_id(model, follower)])
        data.qvel[dof] = probe.axis_sign(model, joints, follower) * multiplier * driver_source_v
        writes += 1
    return writes


def set_driver_motor(
    model: mujoco.MjModel,
    joints: dict[str, ET.Element],
) -> tuple[int, float]:
    actuator_id = -1
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == DRIVER:
            actuator_id = aid
            break
    if actuator_id < 0:
        raise RuntimeError(f"missing source driver actuator: {DRIVER}")
    limit = float(joints[DRIVER].find("limit").get("effort"))
    model.actuator_gaintype[actuator_id] = int(mujoco.mjtGain.mjGAIN_FIXED)
    model.actuator_gainprm[actuator_id, :] = 0.0
    model.actuator_gainprm[actuator_id, 0] = 1.0
    model.actuator_biastype[actuator_id] = int(mujoco.mjtBias.mjBIAS_NONE)
    model.actuator_biasprm[actuator_id, :] = 0.0
    model.actuator_gear[actuator_id, :] = 0.0
    model.actuator_gear[actuator_id, 0] = 1.0
    model.actuator_ctrllimited[actuator_id] = 1
    model.actuator_ctrlrange[actuator_id, :] = [-limit, limit]
    model.actuator_forcelimited[actuator_id] = 1
    model.actuator_forcerange[actuator_id, :] = [-limit, limit]
    return actuator_id, limit


def relation_id(model: mujoco.MjModel, follower: str) -> int:
    expected = f"urdf_mimic_{follower}"
    for eqid in range(model.neq):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eqid) == expected:
            return eqid
    raise KeyError(expected)


def source_limits_ok(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    data: mujoco.MjData,
) -> tuple[float, dict[str, float]]:
    maximum = 0.0
    source_q = {}
    for name, joint in joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limit = joint.find("limit")
        if jid < 0 or limit is None:
            continue
        value = probe.source_q(model, details, joints, name, data)
        source_q[name] = value
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        maximum = max(maximum, lower - value, value - upper, 0.0)
    return maximum, source_q


def max_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[float, list[dict[str, object]]]:
    contacts = []
    maximum = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        penetration = max(0.0, -float(contact.dist))
        maximum = max(maximum, penetration)
        body_names = []
        for geom in (int(contact.geom1), int(contact.geom2)):
            body_id = int(model.geom_bodyid[geom])
            body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        contacts.append({
            "geom1_id": int(contact.geom1),
            "geom2_id": int(contact.geom2),
            "body_names": body_names,
            "penetration_m": penetration,
        })
    return maximum, contacts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    model, details = task.build_model(include_bottle=False)
    data = details["data"]
    joints = source_joint_map()
    open_targets = probe.initialize_open_pose(model, details, joints, data)
    if any(not math.isclose(open_targets[name], OPEN_Q, abs_tol=1e-12) for name in (DRIVER,)):
        raise RuntimeError("the source OPEN pose changed; review the fixture target")

    follower_names = set(FOLLOWERS)
    actuator_joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[aid, 0]))
        for aid in range(model.nu)
    }
    if follower_names & actuator_joint_names:
        raise RuntimeError("a mimic follower has an actuator")

    for follower in FOLLOWERS:
        data.eq_active[relation_id(model, follower)] = 0
    actuator_id, motor_limit = set_driver_motor(model, joints)
    held_ctrl = np.zeros(model.nu, dtype=float)
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        held_ctrl[aid] = float(data.qpos[qpos_address])
    held_ctrl[actuator_id] = 0.0
    base_qpos = data.qpos.copy()
    driver_jid = probe.joint_id(model, DRIVER)
    driver_dof = int(model.jnt_dofadr[driver_jid])
    driver_sign = probe.axis_sign(model, joints, DRIVER)
    follower_columns = [
        (
            int(model.jnt_dofadr[probe.joint_id(model, follower)]),
            multiplier,
            probe.axis_sign(model, joints, follower),
        )
        for follower, multiplier in FOLLOWERS.items()
    ]
    column = reduced_column(model.nv, driver_dof, driver_sign, follower_columns)
    q = OPEN_Q
    velocity = 0.0
    dt = float(model.opt.timestep)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    effective_inertia = float(column @ mass @ column)
    kp = effective_inertia * NATURAL_FREQUENCY_RAD_S**2
    kd = 2.0 * effective_inertia * NATURAL_FREQUENCY_RAD_S

    trace_path = output / "reduced_coordinate_trace.jsonl"
    video_path = output / "reduced_coordinate_thumb.mp4"
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.site_xpos[int(details["site_id"])].copy()
    camera.distance = 0.72
    camera.azimuth = 200.0
    camera.elevation = -18.0

    max_mimic_error = 0.0
    max_target_error = 0.0
    max_limit_violation = 0.0
    max_penetration_seen = 0.0
    max_speed = 0.0
    max_effort = 0.0
    expansion_writes = 0
    first_failure = None
    phase_stats: dict[str, dict[str, object]] = {}
    step = 0
    with trace_path.open("w", encoding="utf-8") as trace, imageio.get_writer(
        video_path, fps=30, codec="libx264", quality=8, macro_block_size=None
    ) as video:
        for phase, steps, start_target, end_target in PHASES:
            phase_rows = []
            for local_step in range(steps):
                fraction = (local_step + 1) / steps
                target = start_target + smoothstep5(fraction) * (end_target - start_target)
                expansion_writes += state_expansion(
                    model, details, joints, base_qpos, q, velocity, data
                )
                data.ctrl[:] = held_ctrl
                mujoco.mj_forward(model, data)
                feedforward = float(column @ (
                    data.qfrc_bias - data.qfrc_passive - data.qfrc_constraint - data.qfrc_applied
                ))
                torque_source = feedforward + kp * (target - q) - kd * velocity
                torque_source = float(np.clip(torque_source, -motor_limit, motor_limit))
                data.ctrl[actuator_id] = driver_sign * torque_source
                mujoco.mj_forward(model, data)
                mujoco.mj_fullM(model, mass, data.qM)
                reduced_mass = float(column @ mass @ column)
                generalized_force = float(column @ (
                    data.qfrc_actuator + data.qfrc_passive + data.qfrc_applied
                    + data.qfrc_constraint - data.qfrc_bias
                ))
                acceleration = generalized_force / reduced_mass
                velocity += dt * acceleration
                q += dt * velocity
                data.time += dt
                expansion_writes += state_expansion(
                    model, details, joints, base_qpos, q, velocity, data
                )
                mujoco.mj_forward(model, data)
                step += 1

                errors = probe.mimic_errors(model, details, joints, data)
                mimic_max = max(errors.values(), default=0.0)
                limit_violation, positions = source_limits_ok(model, details, joints, data)
                penetration, contacts = max_penetration(model, data)
                target_error = abs(q - target)
                max_mimic_error = max(max_mimic_error, mimic_max)
                max_target_error = max(max_target_error, target_error)
                max_limit_violation = max(max_limit_violation, limit_violation)
                max_penetration_seen = max(max_penetration_seen, penetration)
                max_speed = max(max_speed, abs(velocity))
                max_effort = max(max_effort, abs(torque_source))
                row = {
                    "step": step,
                    "time_s": float(data.time),
                    "phase": phase,
                    "driver_target_source_rad": target,
                    "driver_source_rad": q,
                    "driver_velocity_source_rad_s": velocity,
                    "driver_torque_source_nm": torque_source,
                    "projected_gravity_feedforward_nm": feedforward,
                    "mimic_residuals_rad": errors,
                    "max_mimic_residual_rad": mimic_max,
                    "source_joint_limit_violation_rad": limit_violation,
                    "max_contact_penetration_m": penetration,
                    "contacts": contacts,
                    "selected_driver_and_follower_positions_rad": {
                        name: positions[name] for name in (DRIVER, *FOLLOWERS.keys())
                    },
                    "follower_qpos_expansion_writes": len(FOLLOWERS) * 2,
                }
                trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                phase_rows.append(row)
                if step == 1 or step % 20 == 0:
                    renderer.update_scene(data, camera=camera)
                    video.append_data(renderer.render())
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    first_failure = {"step": step, "phase": phase, "reason": "non_finite_state"}
                elif mimic_max >= MIMIC_LIMIT_RAD:
                    first_failure = {
                        "step": step, "phase": phase, "reason": "mimic_gate",
                        "residual_rad": mimic_max,
                    }
                elif limit_violation > LIMIT_TOLERANCE_RAD:
                    first_failure = {
                        "step": step, "phase": phase, "reason": "source_joint_limit",
                        "violation_rad": limit_violation,
                    }
                elif penetration > MAX_CONTACT_PENETRATION_M:
                    first_failure = {
                        "step": step, "phase": phase, "reason": "contact_penetration",
                        "penetration_m": penetration,
                    }
                if first_failure:
                    break
            tail = phase_rows[-100:]
            phase_stats[phase] = {
                "steps": len(phase_rows),
                "max_abs_target_error_rad": max((abs(r["driver_target_source_rad"] - r["driver_source_rad"]) for r in phase_rows), default=0.0),
                "max_mimic_residual_rad": max((r["max_mimic_residual_rad"] for r in phase_rows), default=0.0),
                "tail_100_max_abs_error_rad": max((abs(r["driver_target_source_rad"] - r["driver_source_rad"]) for r in tail), default=0.0),
                "tail_100_max_abs_velocity_rad_s": max((abs(r["driver_velocity_source_rad_s"]) for r in tail), default=0.0),
            }
            if phase in ("OPEN_HOLD", "CLOSE_HOLD", "OPEN_SETTLE") and len(tail) == 100:
                if phase_stats[phase]["tail_100_max_abs_error_rad"] > 0.010:
                    first_failure = {"phase": phase, "reason": "not_settled_position"}
                elif phase_stats[phase]["tail_100_max_abs_velocity_rad_s"] > 0.020:
                    first_failure = {"phase": phase, "reason": "not_settled_velocity"}
            if first_failure:
                break

    result = {
        "status": "PASS" if first_failure is None else "FAIL",
        "first_failure": first_failure,
        "classification": "SIMULATION_ONLY custom reduced-coordinate integrator; not hardware transmission validation",
        "source": {
            "urdf_sha256": probe.sha256(task.URDF),
            "mujoco_version": mujoco.__version__,
            "timestep_s": dt,
            "gravity_m_s2": model.opt.gravity.tolist(),
            "compiled_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq},
            "active_mimic_equalities": int(np.sum(data.eq_active)),
            "inactive_selected_equalities": [
                f"urdf_mimic_{name}" for name in FOLLOWERS
            ],
            "driver_joint": DRIVER,
            "follower_relations": FOLLOWERS,
            "source_open_rad": OPEN_Q,
            "source_close_rad": CLOSE_Q,
            "source_effort_limit_nm": motor_limit,
        },
        "reduced_state": {
            "dimension": 1,
            "mass_projected_from_all_source_link_inertias": True,
            "effective_inertia_kg_m2_at_open": effective_inertia,
            "kp_nm_per_rad": kp,
            "kd_nms_per_rad": kd,
            "follower_actuators": 0,
            "follower_qpos_expansion_writes": expansion_writes,
            "expansion_is_explicit_state_map": True,
            "uses_mujoco_mj_step": False,
            "integrator": "semi-implicit Euler on reduced coordinate; MuJoCo mj_forward/mass/bias/constraint forces",
        },
        "gates": {
            "gravity_loaded_open_hold": phase_stats.get("OPEN_HOLD"),
            "open_close_open": all(name in phase_stats for name in ("CLOSE", "OPEN", "OPEN_SETTLE")),
            "mimic_limit_rad": MIMIC_LIMIT_RAD,
            "max_mimic_residual_rad": max_mimic_error,
            "max_target_error_rad": max_target_error,
            "max_speed_rad_s": max_speed,
            "max_motor_effort_nm": max_effort,
            "max_source_limit_violation_rad": max_limit_violation,
            "max_contact_penetration_m": max_penetration_seen,
            "phase_stats": phase_stats,
        },
        "evidence": {
            "trace": str(trace_path),
            "trace_sha256": sha256(trace_path),
            "video": str(video_path),
            "video_sha256": sha256(video_path),
        },
        "command": shlex.join(sys.argv),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "reproduction_command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue 46 reduced-coordinate thumb transmission diagnostic.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
