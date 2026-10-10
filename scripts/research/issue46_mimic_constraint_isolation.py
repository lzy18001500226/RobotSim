from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import issue46_dual_omnihand_transmission_probe as probe  # noqa: E402
import issue46_x2_grasp as task  # noqa: E402

STEPS = 5
CONFIGURATIONS = ("baseline", "gravity_disabled", "hand_self_contact_disabled", "driver_actuation_disabled")


def source_velocity(model, joints, data, name):
    jid = probe.joint_id(model, name)
    return probe.axis_sign(model, joints, name) * float(data.qvel[int(model.jnt_dofadr[jid])])


def solver_settings(model, data):
    return {
        "timestep_s": float(model.opt.timestep),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "integrator": str(mujoco.mjtIntegrator(int(model.opt.integrator))),
        "solver": str(mujoco.mjtSolver(int(model.opt.solver))),
        "cone": str(mujoco.mjtCone(int(model.opt.cone))),
        "jacobian": str(mujoco.mjtJacobian(int(model.opt.jacobian))),
        "iterations": int(model.opt.iterations),
        "tolerance": float(model.opt.tolerance),
        "ls_iterations": int(model.opt.ls_iterations),
        "ls_tolerance": float(model.opt.ls_tolerance),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "impratio": float(model.opt.impratio),
        "nefc": int(data.nefc),
        "ncon": int(data.ncon),
    }


def mimic_state(model, details, joints, data):
    ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i): i
        for i in range(model.neq)
    }
    result = {}
    for rel in details["mimic_relations"]:
        follower, driver = str(rel["follower_joint"]), str(rel["driver_joint"])
        ratio, offset = float(rel["multiplier"]), float(rel["offset"])
        position_residual = (
            probe.source_q(model, details, joints, follower, data)
            - ratio * probe.source_q(model, details, joints, driver, data) - offset
        )
        velocity_residual = source_velocity(model, joints, data, follower) - ratio * source_velocity(
            model, joints, data, driver
        )
        eqid = ids[f"urdf_mimic_{follower}"]
        rows = [
            {
                "row": i,
                "position_residual_native": float(data.efc_pos[i]),
                "velocity_residual_native": float(data.efc_vel[i]),
                "constraint_force": float(data.efc_force[i]),
            }
            for i in range(int(data.nefc))
            if int(data.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            and int(data.efc_id[i]) == eqid
        ]
        result[follower] = {
            "driver": driver,
            "position_residual_source_rad": float(position_residual),
            "velocity_residual_source_rad_s": float(velocity_residual),
            "equality_id": int(eqid),
            "equality_rows": rows,
        }
    return result


def disable_hand_self_contact(model, _data, geom1, geom2):
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        for g in (geom1, geom2)
    ]
    return int(all(name.startswith(("L_", "R_")) for name in names))


def contacts(model, data):
    result = []
    for i in range(data.ncon):
        c = data.contact[i]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        result.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
            "distance_m": float(c.dist),
            "penetration_m": max(0.0, -float(c.dist)),
            "contact_force_local_6d": wrench.tolist(),
        })
    return result


def controller(model, details, joints, data, torque_limit_override=None):
    active = [str(n) for n in details["active_hand_joint_names"]]
    columns = probe.driver_columns(model, details, joints)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, mass, data.qM)
    gains, bias_loads = {}, {}
    for name in active:
        inertia = float(columns[name] @ mass @ columns[name])
        kp, kd = probe.critical_damping(inertia, probe.NATURAL_FREQUENCY_RAD_S)
        gains[name] = {"kp": kp, "kd": kd}
        bias_loads[name] = abs(probe.projected_feedforward(data, columns[name]))
    limits = {n: float(joints[n].find("limit").get("effort")) for n in active}
    limit = torque_limit_override or min(
        min(limits.values()), probe.TORQUE_LIMIT_GRAVITY_MULTIPLIER * max(bias_loads.values())
    )
    if limit <= 0:
        raise RuntimeError("non-positive diagnostic torque limit")
    actuator = probe.replace_hand_actuators_with_motors(model, details, limit)
    held = np.zeros(model.nu)
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        held[aid] = 0.0 if name in actuator else float(data.qpos[int(model.jnt_qposadr[jid])])
    return active, columns, gains, limit, actuator, held, bias_loads


def run(configuration, torque_limit, baseline_gains):
    model, details, joints = probe.load_source_context()
    data = details["data"]
    if configuration == "gravity_disabled":
        model.opt.gravity[:] = 0.0
    open_targets = probe.initialize_open_pose(model, details, joints, data)
    active, columns, gains, _, actuators, held, bias_loads = controller(
        model, details, joints, data, torque_limit_override=torque_limit
    )
    gains = baseline_gains
    previous_filter = mujoco.get_mjcb_contactfilter()
    if configuration == "hand_self_contact_disabled":
        mujoco.set_mjcb_contactfilter(disable_hand_self_contact)
    try:
        data.ctrl[:] = held
        mujoco.mj_forward(model, data)
        initial = mimic_state(model, details, joints, data)
        rows = []
        for step in range(1, STEPS + 1):
            data.ctrl[:] = held
            if configuration != "driver_actuation_disabled":
                for name in active:
                    jid = probe.joint_id(model, name)
                    dof, sign = int(model.jnt_dofadr[jid]), probe.axis_sign(model, joints, name)
                    q = probe.source_q(model, details, joints, name, data)
                    vel = sign * float(data.qvel[dof])
                    ff = probe.projected_feedforward(data, columns[name])
                    torque = ff + gains[name]["kp"] * (open_targets[name] - q) - gains[name]["kd"] * vel
                    data.ctrl[actuators[name]] = float(np.clip(sign * torque, -torque_limit, torque_limit))
            mujoco.mj_forward(model, data)
            pair_state = mimic_state(model, details, joints, data)
            dynamics = {}
            for rel in details["mimic_relations"]:
                for name in (str(rel["driver_joint"]), str(rel["follower_joint"])):
                    jid = probe.joint_id(model, name)
                    dof, sign = int(model.jnt_dofadr[jid]), probe.axis_sign(model, joints, name)
                    vel = source_velocity(model, joints, data, name)
                    dynamics[name] = {
                        "source_position_rad": probe.source_q(model, details, joints, name, data),
                        "source_velocity_rad_s": vel,
                        "qacc_source_rad_s2": sign * float(data.qacc[dof]),
                        "qfrc_bias_native": float(data.qfrc_bias[dof]),
                        "qfrc_passive_native": float(data.qfrc_passive[dof]),
                        "qfrc_actuator_native": float(data.qfrc_actuator[dof]),
                    }
            motors = {}
            for name, aid in actuators.items():
                ctrl = float(data.ctrl[aid])
                motors[name] = {
                    "ctrl_nm": ctrl,
                    "actuator_force_nm": float(data.actuator_force[aid]),
                    "at_torque_limit": abs(ctrl) >= torque_limit - 1e-10,
                    "torque_limit_nm": torque_limit,
                }
            row = {
                "step": step,
                "time_before_s": float(data.time),
                "solver": solver_settings(model, data),
                "mimic_constraints": pair_state,
                "joint_dynamics": dynamics,
                "driver_motors": motors,
                "contacts": contacts(model, data),
            }
            mujoco.mj_step(model, data)
            row["time_after_s"] = float(data.time)
            for name, state in dynamics.items():
                velocity_after = source_velocity(model, joints, data, name)
                state["qacc_integrated_rad_s2"] = (
                    velocity_after - float(state["source_velocity_rad_s"])
                ) / float(model.opt.timestep)
            row["mimic_after_step"] = mimic_state(model, details, joints, data)
            rows.append(row)
        return {
            "configuration": configuration,
            "steps": STEPS,
            "duration_s": STEPS * float(model.opt.timestep),
            "initial_mimic_constraints": initial,
            "mimic_gate_rad": probe.MIMIC_LIMIT_RAD,
            "controller_torque_limit_nm": torque_limit,
            "gravity_projected_loads_nm": bias_loads,
            "solver_settings_final": solver_settings(model, data),
            "rows": rows,
        }
    finally:
        mujoco.set_mjcb_contactfilter(previous_filter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model, details, joints = probe.load_source_context()
    data = details["data"]
    probe.initialize_open_pose(model, details, joints, data)
    active, _, gains, limit, _, _, _ = controller(model, details, joints, data)
    results = [run(name, limit, gains) for name in CONFIGURATIONS]
    result = {
        "robotsim_head": subprocess.check_output(
            ["git", "-C", str(probe.REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "robotsim_dirty_paths": subprocess.check_output(
            ["git", "-C", str(probe.REPO), "status", "--porcelain"], text=True
        ).splitlines(),
        "vendor_head": subprocess.check_output(
            ["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "vendor_dirty_paths": subprocess.check_output(
            ["git", "-C", str(task.ROOT), "status", "--porcelain"], text=True
        ).splitlines(),
        "urdf_sha256": probe.sha256(task.URDF),
        "diagnostic_script_sha256": probe.sha256(Path(__file__).resolve()),
        "mujoco": mujoco.__version__,
        "python": sys.version.split()[0],
        "timestep_s": task.DT,
        "steps_per_configuration": STEPS,
        "fixed_mimic_gate_rad": probe.MIMIC_LIMIT_RAD,
        "source_mimic_equality_count": model.neq,
        "driver_motor_joint_count": len(active),
        "follower_actuator_count": 0,
        "mimic_equality_solref_direct": sorted(
            {tuple(float(x) for x in model.eq_solref[i]) for i in range(model.neq)}
        ),
        "configuration_results": results,
    }
    out = args.output.resolve() / "constraint_isolation.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "configurations": [
            {
                "name": r["configuration"],
                "step4_max_mimic_error_rad": max(
                    abs(v["position_residual_source_rad"])
                    for v in r["rows"][3]["mimic_after_step"].values()
                ),
                "step4_max_penetration_m": max(
                    (c["penetration_m"] for c in r["rows"][3]["contacts"]), default=0.0
                ),
            }
            for r in results
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
