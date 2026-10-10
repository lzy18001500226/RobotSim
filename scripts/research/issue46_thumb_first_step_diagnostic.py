from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.issue46_mjstep_thumb_transmission import (
    DEFAULT_VENDOR_ROOT,
    DRIVER,
    DRIVER_KD,
    DRIVER_KP,
    DRIVER_TORQUE_CAP_NM,
    DT,
    FIXTURE_LOCKS,
    FOLLOWERS,
    OPEN_Q_SOURCE,
    build_model,
    controller_torque,
    equality_residuals,
    fixture_lock_residuals,
    source_position,
    source_velocity,
    verify_source,
)


def _enum_names(enum_type: Any, prefix: str) -> dict[int, str]:
    return {
        int(value): name
        for name in dir(enum_type)
        if name.startswith(prefix) and isinstance((value := getattr(enum_type, name)), enum_type)
    }


def _capture(model: mujoco.MjModel, data: mujoco.MjData, info: dict[str, Any], label: str,
             step: int | None, command_nm: float | None, target_source_rad: float) -> dict[str, Any]:
    mappings = info["source_joint_mappings"]
    joint_names = ("R_thumb_roll_joint", "R_thumb_abad_joint", DRIVER, *FOLLOWERS)
    joints: dict[str, Any] = {}
    for name in joint_names:
        mapping = mappings[name]
        q = source_position(data, mapping)
        low, high = mapping["source_range"]
        joints[name] = {
            "source_position_rad": q,
            "source_velocity_rad_s": source_velocity(data, mapping),
            "source_acceleration_rad_s2": float(mapping["sign"] * data.qacc[mapping["dadr"]]),
            "source_range_rad": [low, high],
            "distance_to_lower_rad": q - low,
            "distance_to_upper_rad": high - q,
            "qfrc_bias_nm": float(data.qfrc_bias[mapping["dadr"]]),
            "qfrc_actuator_nm": float(data.qfrc_actuator[mapping["dadr"]]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[mapping["dadr"]]),
        }

    equality_names = {
        i: str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i))
        for i in range(model.neq)
    }
    constraint_names = _enum_names(mujoco.mjtConstraint, "mjCNSTR_")
    constraint_force_by_type: dict[str, np.ndarray] = {}
    for kind, kind_name in constraint_names.items():
        mask = data.efc_type == kind
        if np.any(mask):
            efc_force = np.zeros(data.nefc)
            efc_force[mask] = data.efc_force[mask]
            generalized_force = np.zeros(model.nv)
            mujoco.mj_mulJacTVec(model, data, generalized_force, efc_force)
            constraint_force_by_type[kind_name] = generalized_force
    for name, mapping in ((name, mappings[name]) for name in joint_names):
        joints[name]["qfrc_constraint_by_type_nm"] = {
            kind: float(force[mapping["dadr"]]) for kind, force in constraint_force_by_type.items()
        }

    constraints = []
    joint_limit_rows = []
    for row in range(data.nefc):
        kind = int(data.efc_type[row])
        item = {
            "row": row,
            "type": constraint_names.get(kind, str(kind)),
            "id": int(data.efc_id[row]),
            "equality_name": equality_names.get(int(data.efc_id[row])) if kind == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) else None,
            "position_residual": float(data.efc_pos[row]),
            "force": float(data.efc_force[row]),
            "reference_acceleration": float(data.efc_aref[row]),
        }
        constraints.append(item)
        if kind == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
            joint_limit_rows.append(item)

    contacts = []
    contact_force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, contact_force)
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        body1_id, body2_id = int(model.geom_bodyid[geom1_id]), int(model.geom_bodyid[geom2_id])
        contacts.append({
            "geom1_id": geom1_id,
            "geom1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id),
            "geom1_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id),
            "geom2_id": geom2_id,
            "geom2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id),
            "geom2_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id),
            "distance_m": float(contact.dist),
            "position_world_m": contact.pos.tolist(),
            "force_contact_frame": contact_force.tolist(),
        })

    actuator_id = int(info["driver_actuator_id"])
    target_by_joint = {DRIVER: target_source_rad}
    for follower in FOLLOWERS:
        target_by_joint[follower] = next(
            float(relation["offset"]) + float(relation["multiplier"]) * target_source_rad
            for relation in info["mimic_relations"] if relation["follower"] == follower
        )
    for name in FIXTURE_LOCKS:
        target_by_joint[name] = float(mappings[name]["source_ref"])
    return {
        "label": label,
        "step_before_integration": step,
        "time_s": float(data.time),
        "target_source_rad": target_source_rad,
        "target_by_joint_source_rad": target_by_joint,
        "command_torque_nm": command_nm,
        "actuator": {
            "name": "thumb_driver_motor",
            "target_joint_id": int(model.actuator_trnid[actuator_id, 0]),
            "gear": model.actuator_gear[actuator_id].tolist(),
            "ctrl": float(data.ctrl[actuator_id]),
            "force": float(data.actuator_force[actuator_id]),
            "ctrlrange": model.actuator_ctrlrange[actuator_id].tolist(),
            "forcerange": model.actuator_forcerange[actuator_id].tolist(),
        },
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "qacc": data.qacc.tolist(),
        "qfrc_bias": data.qfrc_bias.tolist(),
        "qfrc_actuator": data.qfrc_actuator.tolist(),
        "qfrc_constraint": data.qfrc_constraint.tolist(),
        "joints": joints,
        "mimic_residual_source_rad": equality_residuals(data, info),
        "fixture_lock_residual_source_rad": fixture_lock_residuals(data, mappings),
        "joint_limit_constraint_rows": joint_limit_rows,
        "constraints": constraints,
        "contacts": contacts,
    }


def run_trial(source_urdf: Path, out_dir: Path, gravity_enabled: bool) -> dict[str, Any]:
    trial_dir = out_dir / ("gravity-on" if gravity_enabled else "gravity-off")
    trial_dir.mkdir(parents=True, exist_ok=True)
    model, info = build_model(source_urdf, trial_dir, timestep_s=DT,
                              exclude_fixture_root_contact=False)
    if not gravity_enabled:
        model.opt.gravity[:] = [0.0, 0.0, 0.0]

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pre_command = _capture(model, data, info, "t0_after_mj_forward_before_command", None, None, OPEN_Q_SOURCE)

    trace: list[dict[str, Any]] = [pre_command]
    commands: list[float] = []
    for step in range(10):
        mujoco.mj_forward(model, data)
        command = controller_torque(model, data, info, OPEN_Q_SOURCE)
        mujoco.mj_forward(model, data)
        commands.append(command)
        trace.append(_capture(model, data, info, "pre_mj_step", step, command, OPEN_Q_SOURCE))
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    final = _capture(model, data, info, "after_ten_mj_steps", 10, commands[-1], OPEN_Q_SOURCE)
    trace.append(final)
    driver_map = info["source_joint_mappings"][DRIVER]
    initial_q = float(pre_command["joints"][DRIVER]["source_position_rad"])
    final_q = float(final["joints"][DRIVER]["source_position_rad"])
    result = {
        "gravity_enabled": gravity_enabled,
        "gravity_m_s2": model.opt.gravity.tolist(),
        "mujoco_version": mujoco.__version__,
        "timestep_s": float(model.opt.timestep),
        "dimensions": info["dimensions"],
        "initial_state_source_valid": all(
            joint["distance_to_lower_rad"] >= -1e-12 and joint["distance_to_upper_rad"] >= -1e-12
            for joint in pre_command["joints"].values()
        ),
        "initial_driver_position_rad": initial_q,
        "target_open_rad": OPEN_Q_SOURCE,
        "initial_command_torque_nm": commands[0],
        "first_step_driver_delta_rad": float(trace[2]["joints"][DRIVER]["source_position_rad"] - initial_q),
        "position_after_ten_steps_rad": final_q,
        "driver_delta_after_ten_steps_rad": final_q - initial_q,
        "actuator_gear": model.actuator_gear[info["driver_actuator_id"]].tolist(),
        "source_driver_sign": int(driver_map["sign"]),
        "source_joint_mappings": info["source_joint_mappings"],
        "actuator_transmission": {
            "trnid": model.actuator_trnid[info["driver_actuator_id"]].tolist(),
            "gear": model.actuator_gear[info["driver_actuator_id"]].tolist(),
            "ctrlrange": model.actuator_ctrlrange[info["driver_actuator_id"]].tolist(),
            "forcerange": model.actuator_forcerange[info["driver_actuator_id"]].tolist(),
            "ctrllimited": bool(model.actuator_ctrllimited[info["driver_actuator_id"]]),
            "forcelimited": bool(model.actuator_forcelimited[info["driver_actuator_id"]]),
        },
        "controller": {
            "type": "projected gravity bias plus PD",
            "kp_nm_per_rad": DRIVER_KP,
            "kd_nm_s_per_rad": DRIVER_KD,
            "torque_cap_nm": DRIVER_TORQUE_CAP_NM,
            "source_effort_limit_nm": driver_map["source_effort_limit_nm"],
        },
        "source_mimic_coefficients": info["mimic_relations"],
        "eq_data": {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)): model.eq_data[index].tolist()
            for index in range(model.neq)
        },
        "equality_objects": {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)): {
                "obj1_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[index])),
                "obj2_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[index]))
                if int(model.eq_obj2id[index]) >= 0 else None,
                "active": bool(data.eq_active[index]),
            }
            for index in range(model.neq)
        },
        "trace": trace,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture first-step X2 thumb drive/constraint diagnostics without changing the model")
    parser.add_argument("--vendor-root", type=Path, default=Path(os.environ.get("AGIBOT_X2_VENDOR_ROOT", DEFAULT_VENDOR_ROOT)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_urdf = verify_source(args.vendor_root)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "source_pin": "575cc6b988f976c23550e0db85aa1e5475d3652d",
        "source_urdf_sha256": "344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259",
        "source_urdf": str(source_urdf),
        "trial_state": "mj_resetData source reference state; no qpos/qvel writes",
        "command": "OPEN hold at +0.08 rad for ten mj_step calls",
        "trials": [
            run_trial(source_urdf, args.output, gravity_enabled=False),
            run_trial(source_urdf, args.output, gravity_enabled=True),
        ],
    }
    output = args.output / "first_step_diagnostics.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "trials": [{
            "gravity_enabled": trial["gravity_enabled"],
            "initial_command_torque_nm": trial["initial_command_torque_nm"],
            "first_step_driver_delta_rad": trial["first_step_driver_delta_rad"],
            "driver_delta_after_ten_steps_rad": trial["driver_delta_after_ten_steps_rad"],
            "initial_mimic_residual_rad": trial["trace"][0]["mimic_residual_source_rad"],
            "final_mimic_residual_rad": trial["trace"][-1]["mimic_residual_source_rad"],
        } for trial in results["trials"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
