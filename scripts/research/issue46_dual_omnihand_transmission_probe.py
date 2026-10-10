from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import issue46_x2_grasp as task  # noqa: E402

DT = task.DT
MIMIC_LIMIT_RAD = task.MIMIC_RELATION_TOLERANCE_RAD
SETTLED_POSITION_ERROR_RAD = 0.010
SETTLED_VELOCITY_RAD_S = 0.020
LIMIT_TOLERANCE_RAD = 1e-5
MAX_CONTACT_PENETRATION_M = 0.002
NATURAL_FREQUENCY_RAD_S = 8.0
TORQUE_LIMIT_GRAVITY_MULTIPLIER = 2.0
SIM_ONLY_MIMIC_EQ_SOLREF_SCALE = 2.0
MOTION_JOINTS = {
    "R_thumb_roll_joint": 0.65,
    "R_thumb_abad_joint": -0.30,
    "R_thumb_mcp_joint": 0.82,
    "R_middle_pip_joint": 1.15,
}
OPEN_OVERRIDES = {
    "R_thumb_roll_joint": 0.80,
    "R_thumb_abad_joint": -1.70,
    "R_ring_abad_joint": 0.0,
}
PHASES = (
    ("OPEN_HOLD", 400, "open"),
    ("CLOSE", 750, "close"),
    ("CLOSE_HOLD", 250, "close"),
    ("OPEN", 750, "open"),
    ("OPEN_SETTLE", 400, "open"),
)


def smoothstep5(fraction: float) -> float:
    x = float(np.clip(fraction, 0.0, 1.0))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def critical_damping(inertia: float, natural_frequency: float) -> tuple[float, float]:
    if not math.isfinite(inertia) or inertia <= 0.0 or natural_frequency <= 0.0:
        raise ValueError("inertia and natural frequency must be positive and finite")
    kp = inertia * natural_frequency**2
    kd = 2.0 * inertia * natural_frequency
    return kp, kd


def transmission_column(
    nv: int,
    driver_dof: int,
    driver_axis_sign: int,
    followers: list[tuple[int, float, int]],
) -> np.ndarray:
    column = np.zeros(nv, dtype=float)
    column[driver_dof] = float(driver_axis_sign)
    for follower_dof, multiplier, follower_axis_sign in followers:
        column[follower_dof] = float(follower_axis_sign) * float(multiplier)
    return column


def expand_transmission_targets(
    driver_targets: dict[str, float],
    relations: list[dict[str, object]],
) -> dict[str, float]:
    targets = dict(driver_targets)
    for relation in relations:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        if driver in targets:
            targets[follower] = float(relation["multiplier"]) * targets[driver] + float(relation["offset"])
    return targets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_context() -> tuple[mujoco.MjModel, dict[str, object], dict[str, ET.Element]]:
    model, details = task.build_model(include_bottle=False)
    if model.neq != 12 or not details["mimic_constraints_match"]:
        raise RuntimeError("the complete 12-relation official mimic set is not compiled")
    root = ET.parse(task.URDF).getroot()
    joints = {str(node.get("name")): node for node in root.findall("joint") if node.get("name")}
    return model, details, joints


def apply_sim_only_mimic_constraint_correction(model: mujoco.MjModel) -> dict[str, object]:
    if model.neq != 12:
        raise RuntimeError(f"expected 12 source mimic equalities, found {model.neq}")
    original = model.eq_solref.copy()
    model.eq_solref[:] = original * SIM_ONLY_MIMIC_EQ_SOLREF_SCALE
    return {
        "classification": "SIMULATION_ONLY constraint regularization; not hardware-equivalent",
        "scale": SIM_ONLY_MIMIC_EQ_SOLREF_SCALE,
        "original_solref_direct": original.tolist(),
        "applied_solref_direct": model.eq_solref.tolist(),
    }


def joint_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise KeyError(name)
    return int(result)


def axis_sign(model: mujoco.MjModel, joints: dict[str, ET.Element], name: str) -> int:
    jid = joint_id(model, name)
    source_axis = np.fromstring(joints[name].find("axis").get("xyz", "1 0 0"), sep=" ")
    return 1 if float(np.dot(source_axis, model.jnt_axis[jid])) >= 0.0 else -1


def source_q(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    name: str,
    state: mujoco.MjData,
) -> float:
    jid = joint_id(model, name)
    qadr = int(model.jnt_qposadr[jid])
    reference = float(details["joint_refs"].get(name, 0.0))
    return reference + axis_sign(model, joints, name) * (float(state.qpos[qadr]) - float(model.qpos0[qadr]))


def set_source_q(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    name: str,
    value: float,
    state: mujoco.MjData,
) -> None:
    jid = joint_id(model, name)
    qadr = int(model.jnt_qposadr[jid])
    reference = float(details["joint_refs"].get(name, 0.0))
    state.qpos[qadr] = float(model.qpos0[qadr]) + axis_sign(model, joints, name) * (value - reference)


def initialize_open_pose(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    data: mujoco.MjData,
) -> dict[str, float]:
    for pose in (
        task.LEFT_ARM_NEUTRAL_SOURCE_POSE,
        task.LEFT_HAND_NEUTRAL_SOURCE_POSE,
        task.RIGHT_HAND_COLLISION_CORRECTION_SOURCE_POSE,
    ):
        for name, value in pose.items():
            set_source_q(model, details, joints, name, value, data)
    for name, value in OPEN_OVERRIDES.items():
        set_source_q(model, details, joints, name, value, data)
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        expected = float(relation["multiplier"]) * source_q(model, details, joints, driver, data)
        expected += float(relation["offset"])
        set_source_q(model, details, joints, follower, expected, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return {
        name: source_q(model, details, joints, name, data)
        for name in details["active_hand_joint_names"]
    }


def validate_targets(
    model: mujoco.MjModel,
    joints: dict[str, ET.Element],
    targets: dict[str, float],
) -> None:
    for name, value in targets.items():
        limit = joints[name].find("limit")
        if limit is None:
            raise RuntimeError(f"source joint has no range: {name}")
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        if not lower <= value <= upper:
            raise ValueError(f"target outside source limit: {name}={value} not in [{lower}, {upper}]")
        if not math.isfinite(value):
            raise ValueError(f"non-finite target: {name}")


def driver_columns(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
) -> dict[str, np.ndarray]:
    followers_by_driver: dict[str, list[tuple[int, float, int]]] = {}
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        followers_by_driver.setdefault(driver, []).append(
            (
                int(model.jnt_dofadr[joint_id(model, follower)]),
                float(relation["multiplier"]),
                axis_sign(model, joints, follower),
            )
        )
    columns = {}
    for name in details["active_hand_joint_names"]:
        jid = joint_id(model, str(name))
        columns[str(name)] = transmission_column(
            model.nv,
            int(model.jnt_dofadr[jid]),
            axis_sign(model, joints, str(name)),
            followers_by_driver.get(str(name), []),
        )
    return columns


def replace_hand_actuators_with_motors(
    model: mujoco.MjModel,
    details: dict[str, object],
    torque_limit: float,
) -> dict[str, int]:
    by_joint = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            by_joint[name] = aid
    active = [str(name) for name in details["active_hand_joint_names"]]
    if any(name not in by_joint for name in active):
        missing = sorted(set(active) - set(by_joint))
        raise RuntimeError(f"source-independent hand joint actuator missing: {missing}")
    for name in active:
        aid = by_joint[name]
        source_limit = float(np.max(np.abs(model.actuator_forcerange[aid])))
        bound = min(source_limit, torque_limit)
        model.actuator_gaintype[aid] = int(mujoco.mjtGain.mjGAIN_FIXED)
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = 1.0
        model.actuator_biastype[aid] = int(mujoco.mjtBias.mjBIAS_NONE)
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_ctrllimited[aid] = 1
        model.actuator_ctrlrange[aid, :] = [-bound, bound]
        model.actuator_forcelimited[aid] = 1
        model.actuator_forcerange[aid, :] = [-bound, bound]
    return {name: by_joint[name] for name in active}


def projected_feedforward(
    data: mujoco.MjData,
    column: np.ndarray,
) -> float:
    return float(np.dot(column, data.qfrc_bias - data.qfrc_passive))


def mimic_errors(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    data: mujoco.MjData,
) -> dict[str, float]:
    result = {}
    for relation in details["mimic_relations"]:
        driver = str(relation["driver_joint"])
        follower = str(relation["follower_joint"])
        expected = float(relation["multiplier"]) * source_q(model, details, joints, driver, data)
        expected += float(relation["offset"])
        result[follower] = abs(source_q(model, details, joints, follower, data) - expected)
    return result


def check_limits_and_contacts(
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    data: mujoco.MjData,
) -> tuple[float, float, list[dict[str, object]]]:
    max_limit_violation = 0.0
    for name, joint in joints.items():
        if not name.startswith(("L_", "R_")) or joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        jid = joint_id(model, name)
        qadr = int(model.jnt_qposadr[jid])
        value = source_q(model, details, joints, name, data)
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        max_limit_violation = max(max_limit_violation, lower - value, value - upper, 0.0)
    contacts = []
    max_penetration = 0.0
    for cid in range(data.ncon):
        contact = data.contact[cid]
        penetration = max(0.0, -float(contact.dist))
        max_penetration = max(max_penetration, penetration)
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        contacts.append(
            {
                "geom1_id": geom1,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "body1_id": body1,
                "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                "geom2_id": geom2,
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "body2_id": body2,
                "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
                "penetration_m": penetration,
            }
        )
    return max_limit_violation, max_penetration, contacts


def analyze_baseline_trace(
    path: Path,
    output: Path,
    model: mujoco.MjModel,
    details: dict[str, object],
    joints: dict[str, ET.Element],
    initial_qpos: np.ndarray,
) -> dict[str, object]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("baseline trace is empty")
    dt = float(rows[1]["time_s"] - rows[0]["time_s"]) if len(rows) > 1 else DT
    all_hand = rows[0]["right_hand_joint_qpos_rad"]
    initial_source = {}
    for name in all_hand:
        jid = joint_id(model, name)
        qadr = int(model.jnt_qposadr[jid])
        initial_source[name] = float(model.qpos0[qadr]) + axis_sign(model, joints, name) * (
            float(initial_qpos[qadr]) - float(model.qpos0[qadr])
        )
    follower_to_relation = {str(r["follower_joint"]): r for r in details["mimic_relations"]}
    peak_open: tuple[float, int, str, float, float] | None = None
    for row in rows:
        if row["phase"] != "open_hold":
            continue
        targets = dict(initial_source)
        for name, qtarget in row["finger_targets_rad"].items():
            jid = joint_id(model, name)
            qadr = int(model.jnt_qposadr[jid])
            targets[name] = float(details["joint_refs"].get(name, 0.0)) + axis_sign(model, joints, name) * (
                float(qtarget) - float(model.qpos0[qadr])
            )
        for follower, relation in follower_to_relation.items():
            driver = str(relation["driver_joint"])
            targets[follower] = float(relation["multiplier"]) * targets[driver] + float(relation["offset"])
        for name in details["active_hand_joint_names"]:
            jid = joint_id(model, str(name))
            qadr = int(model.jnt_qposadr[jid])
            actual = float(details["joint_refs"].get(str(name), 0.0)) + axis_sign(model, joints, str(name)) * (
                float(row["right_hand_joint_qpos_rad"][str(name)]) - float(model.qpos0[qadr])
            )
            error = abs(actual - targets[str(name)])
            if peak_open is None or error > peak_open[0]:
                peak_open = (error, int(row["step"]), str(name), targets[str(name)], actual)
    if peak_open is None:
        raise ValueError("baseline trace has no OPEN hold rows")
    contact_row = next((row for row in rows if row["right_contact_bodies"]), None)
    mimic_breach = next(
        (row for row in rows if max(row["mimic_errors_rad"].values(), default=0.0) > MIMIC_LIMIT_RAD),
        None,
    )
    if contact_row is None or mimic_breach is None:
        raise ValueError("baseline trace lacks the expected first contact or mimic breach")
    approach_start = next(row for row in rows if row["phase"] == "approach_preshape")
    samples: dict[str, int] = {
        "open_first_step": 1,
        "open_peak_target_error": peak_open[1],
        "open_hold_end": 250,
        "approach_start": int(approach_start["step"]),
        "first_bottle_contact": int(contact_row["step"]),
        "first_mimic_violation": int(mimic_breach["step"]),
    }
    with (output / "baseline_joint_events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample", "step", "phase", "joint", "target_source_rad", "actual_source_rad", "velocity_source_rad_s", "mimic_residual_rad"),
        )
        writer.writeheader()
        for sample, step in samples.items():
            row = rows[step - 1]
            previous = initial_qpos if step == 1 else None
            previous_row = rows[step - 2] if step > 1 else None
            targets = dict(initial_source)
            for name, qtarget in row["finger_targets_rad"].items():
                jid = joint_id(model, name)
                qadr = int(model.jnt_qposadr[jid])
                targets[name] = float(details["joint_refs"].get(name, 0.0)) + axis_sign(model, joints, name) * (
                    float(qtarget) - float(model.qpos0[qadr])
                )
            for follower, relation in follower_to_relation.items():
                driver = str(relation["driver_joint"])
                if driver in targets:
                    targets[follower] = float(relation["multiplier"]) * targets[driver] + float(relation["offset"])
            for name, actual_raw in row["right_hand_joint_qpos_rad"].items():
                jid = joint_id(model, name)
                qadr = int(model.jnt_qposadr[jid])
                actual = float(model.qpos0[qadr]) + axis_sign(model, joints, name) * (
                    float(actual_raw) - float(model.qpos0[qadr])
                )
                if step == 1:
                    prior = float(model.qpos0[qadr]) + axis_sign(model, joints, name) * (
                        float(previous[qadr]) - float(model.qpos0[qadr])
                    )
                else:
                    prior_raw = float(previous_row["right_hand_joint_qpos_rad"][name])
                    prior = float(model.qpos0[qadr]) + axis_sign(model, joints, name) * (
                        prior_raw - float(model.qpos0[qadr])
                    )
                writer.writerow(
                    {
                        "sample": sample,
                        "step": step,
                        "phase": row["phase"],
                        "joint": name,
                        "target_source_rad": f"{targets[name]:.9f}" if name in targets else "",
                        "actual_source_rad": f"{actual:.9f}",
                        "velocity_source_rad_s": f"{(actual-prior)/dt:.9f}",
                        "mimic_residual_rad": f"{row['mimic_errors_rad'][name]:.9f}" if name in row["mimic_errors_rad"] else "",
                    }
                )
    return {
        "source_trace": str(path),
        "source_trace_sha256": sha256(path),
        "rows": len(rows),
        "event_steps": samples,
        "peak_open_hold_target_error_rad": peak_open[0],
        "peak_open_hold_target_error_joint": peak_open[2],
        "peak_open_hold_target_error_step": peak_open[1],
        "peak_open_hold_target_rad": peak_open[3],
        "peak_open_hold_actual_rad": peak_open[4],
        "mimic_threshold_rad": MIMIC_LIMIT_RAD,
        "first_mimic_violation": {
            "joint": "R_middle_dip_joint",
            "step": int(mimic_breach["step"]),
            "time_s": float(mimic_breach["time_s"]),
            "residual_rad": float(mimic_breach["mimic_error_max_rad"]),
            "joint": max(mimic_breach["mimic_errors_rad"], key=mimic_breach["mimic_errors_rad"].get),
            "first_bottle_contact_step": int(contact_row["step"]),
        },
        "axis_signs_all_revolute_hand_joints": {
            name: axis_sign(model, joints, name)
            for name, joint in joints.items()
            if name.startswith(("L_", "R_")) and joint.get("type") == "revolute"
        },
        "joint_event_csv": str(output / "baseline_joint_events.csv"),
        "joint_event_csv_sha256": sha256(output / "baseline_joint_events.csv"),
    }


def run_motion(output: Path, baseline_trace: Path | None) -> dict[str, object]:
    model, details, joints = load_source_context()
    mimic_constraint_correction = apply_sim_only_mimic_constraint_correction(model)
    data = details["data"]
    active_names = [str(name) for name in details["active_hand_joint_names"]]
    all_hand_names = active_names + sorted(
        str(relation["follower_joint"]) for relation in details["mimic_relations"]
    )
    open_targets = initialize_open_pose(model, details, joints, data)
    initial_qpos = data.qpos.copy()
    close_targets = dict(open_targets)
    close_targets.update(MOTION_JOINTS)
    validate_targets(model, joints, close_targets)
    columns = driver_columns(model, details, joints)
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, mass, data.qM)
    gains = {}
    bias_loads = {}
    for name in active_names:
        column = columns[name]
        effective_inertia = float(column @ mass @ column)
        kp, kd = critical_damping(effective_inertia, NATURAL_FREQUENCY_RAD_S)
        gains[name] = {"effective_inertia_kg_m2": effective_inertia, "kp_nm_per_rad": kp, "kd_nms_per_rad": kd}
        bias_loads[name] = abs(projected_feedforward(data, column))
    source_effort_limits = {
        str(name): float(joints[str(name)].find("limit").get("effort"))
        for name in active_names
    }
    torque_limit = min(
        min(source_effort_limits.values()),
        TORQUE_LIMIT_GRAVITY_MULTIPLIER * max(bias_loads.values()),
    )
    if not math.isfinite(torque_limit) or torque_limit <= 0.0:
        raise RuntimeError("could not derive a positive bounded SIMULATION_ONLY hand torque limit")
    hand_actuator = replace_hand_actuators_with_motors(model, details, torque_limit)
    actuator_by_joint = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            actuator_by_joint[name] = aid
    held_ctrl = np.zeros(model.nu, dtype=float)
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        qadr = int(model.jnt_qposadr[jid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        held_ctrl[aid] = 0.0 if name in hand_actuator else float(data.qpos[qadr])
    mimic_drivers = {str(r["driver_joint"]) for r in details["mimic_relations"]}

    if baseline_trace is not None and baseline_trace.is_file():
        baseline = analyze_baseline_trace(baseline_trace, output, model, details, joints, initial_qpos)
    else:
        baseline = {
            "status": "RAW_TRACE_UNAVAILABLE",
            "expected_path": str(baseline_trace) if baseline_trace is not None else None,
            "previously_recorded_sha256": "6ba5917abbe873426bada5336f44ead561277605f3d611162251e78ddc0a28d7",
            "note": "The prior run's extracted event values are summarized in the report; the original file was absent when this probe ran.",
        }
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.site_xpos[int(details["site_id"])].copy()
    camera.distance = 0.72
    camera.azimuth = 200.0
    camera.elevation = -18.0
    video_path = output / "open_close_open.mp4"
    trace_path = output / "dynamic_trace.jsonl"
    phase_stats: dict[str, dict[str, object]] = {}
    maximum_mimic_error = 0.0
    maximum_limit_violation = 0.0
    maximum_contact_penetration = 0.0
    first_failure: dict[str, object] | None = None
    qpos_assignment_count_during_rollout = 0
    step = 0
    with trace_path.open("w", encoding="utf-8") as trace, imageio.get_writer(
        video_path, fps=30, codec="libx264", quality=8, macro_block_size=None
    ) as video:
        for phase_name, duration_steps, target_key in PHASES:
            if phase_name in ("OPEN_HOLD", "OPEN_SETTLE"):
                start_targets = dict(open_targets)
                end_targets = open_targets
            elif phase_name == "CLOSE":
                start_targets = dict(open_targets)
                end_targets = close_targets
            elif phase_name == "CLOSE_HOLD":
                start_targets = dict(close_targets)
                end_targets = close_targets
            elif phase_name == "OPEN":
                start_targets = dict(close_targets)
                end_targets = open_targets
            else:
                raise RuntimeError(f"unhandled motion phase {phase_name}")
            phase_rows = []
            phase_stats[phase_name] = {}
            for local_step in range(duration_steps):
                fraction = 1.0 if phase_name.endswith("HOLD") else (local_step + 1) / duration_steps
                blend = smoothstep5(fraction)
                targets = {
                    name: float(start_targets[name] + blend * (end_targets[name] - start_targets[name]))
                    for name in active_names
                }
                all_targets = expand_transmission_targets(targets, details["mimic_relations"])
                data.ctrl[:] = held_ctrl
                mujoco.mj_forward(model, data)
                torque_by_joint = {}
                for name in active_names:
                    jid = joint_id(model, name)
                    dof = int(model.jnt_dofadr[jid])
                    sign = axis_sign(model, joints, name)
                    q = source_q(model, details, joints, name, data)
                    velocity = sign * float(data.qvel[dof])
                    kp = gains[name]["kp_nm_per_rad"]
                    kd = gains[name]["kd_nms_per_rad"]
                    feedforward = projected_feedforward(data, columns[name])
                    torque_source = feedforward + kp * (targets[name] - q) - kd * velocity
                    torque_mj = sign * torque_source
                    torque_mj = float(np.clip(torque_mj, -torque_limit, torque_limit))
                    data.ctrl[hand_actuator[name]] = torque_mj
                    torque_by_joint[name] = {
                        "commanded_nm": torque_mj,
                        "projected_bias_compensation_nm": feedforward,
                    }
                mujoco.mj_step(model, data)
                step += 1
                errors = mimic_errors(model, details, joints, data)
                mimic_max = max(errors.values(), default=0.0)
                maximum_mimic_error = max(maximum_mimic_error, mimic_max)
                limit_violation, penetration, contacts = check_limits_and_contacts(model, details, joints, data)
                maximum_limit_violation = max(maximum_limit_violation, limit_violation)
                maximum_contact_penetration = max(maximum_contact_penetration, penetration)
                actuals = {name: source_q(model, details, joints, name, data) for name in all_hand_names}
                velocities = {
                    name: axis_sign(model, joints, name) * float(data.qvel[int(model.jnt_dofadr[joint_id(model, name)])])
                    for name in all_hand_names
                }
                target_errors = {name: abs(actuals[name] - all_targets[name]) for name in all_hand_names}
                row = {
                    "step": step,
                    "time_s": float(data.time),
                    "phase": phase_name,
                    "targets_source_rad": all_targets,
                    "actuals_source_rad": actuals,
                    "velocities_source_rad_s": velocities,
                    "target_errors_rad": target_errors,
                    "mimic_errors_rad": errors,
                    "torque_by_joint_nm": torque_by_joint,
                    "mimic_driver_names": sorted(mimic_drivers),
                    "max_joint_limit_violation_rad": limit_violation,
                    "contacts": contacts,
                    "max_contact_penetration_m": penetration,
                    "qpos_write_count_this_step": 0,
                }
                trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                phase_rows.append(row)
                if step % 20 == 0 or step == 1:
                    renderer.update_scene(data, camera=camera)
                    video.append_data(renderer.render())
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    first_failure = {"step": step, "phase": phase_name, "reason": "non_finite_state"}
                elif mimic_max >= MIMIC_LIMIT_RAD:
                    first_failure = {"step": step, "phase": phase_name, "reason": "source_mimic_limit", "max_error_rad": mimic_max}
                elif limit_violation > LIMIT_TOLERANCE_RAD:
                    first_failure = {"step": step, "phase": phase_name, "reason": "joint_limit_violation", "violation_rad": limit_violation}
                elif penetration > MAX_CONTACT_PENETRATION_M:
                    first_failure = {"step": step, "phase": phase_name, "reason": "collision_penetration", "penetration_m": penetration}
                if first_failure:
                    break
            phase_stats[phase_name] = {
                "steps": len(phase_rows),
                "max_target_error_rad": max((max(r["target_errors_rad"].values()) for r in phase_rows), default=0.0),
                "max_mimic_error_rad": max((max(r["mimic_errors_rad"].values()) for r in phase_rows), default=0.0),
                "max_contact_penetration_m": max((r["max_contact_penetration_m"] for r in phase_rows), default=0.0),
                "final_target_error_rad": max(phase_rows[-1]["target_errors_rad"].values()) if phase_rows else None,
                "final_velocity_rad_s": max(abs(v) for v in phase_rows[-1]["velocities_source_rad_s"].values()) if phase_rows else None,
            }
            if phase_name in ("OPEN_HOLD", "CLOSE_HOLD", "OPEN_SETTLE") and len(phase_rows) >= 100:
                window = phase_rows[-100:]
                window_error = max(max(row["target_errors_rad"].values()) for row in window)
                window_velocity = max(
                    max(abs(value) for value in row["velocities_source_rad_s"].values())
                    for row in window
                )
                phase_stats[phase_name]["settled_window_max_error_rad"] = window_error
                phase_stats[phase_name]["settled_window_max_velocity_rad_s"] = window_velocity
                if window_error > SETTLED_POSITION_ERROR_RAD or window_velocity > SETTLED_VELOCITY_RAD_S:
                    first_failure = {
                        "phase": phase_name,
                        "reason": "not_settled",
                        "max_position_error_rad_last_100": window_error,
                        "max_velocity_rad_s_last_100": window_velocity,
                    }
                    break
            if first_failure:
                break
    stable_rows = []
    if not first_failure:
        final_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()[-100:]]
        stable_rows = final_rows
        stable_error = max(max(r["target_errors_rad"].values()) for r in stable_rows)
        stable_velocity = max(max(abs(v) for v in r["velocities_source_rad_s"].values()) for r in stable_rows)
        if stable_error > SETTLED_POSITION_ERROR_RAD or stable_velocity > SETTLED_VELOCITY_RAD_S:
            first_failure = {
                "phase": "OPEN_SETTLE",
                "reason": "not_settled",
                "max_position_error_rad_last_100": stable_error,
                "max_velocity_rad_s_last_100": stable_velocity,
            }
    motion = {
        "status": "PASS" if not first_failure else "FAIL",
        "first_failure": first_failure,
        "phases": phase_stats,
        "max_mimic_error_rad": maximum_mimic_error,
        "mimic_limit_rad": MIMIC_LIMIT_RAD,
        "max_joint_limit_violation_rad": maximum_limit_violation,
        "max_contact_penetration_m": maximum_contact_penetration,
        "contact_penetration_abort_m": MAX_CONTACT_PENETRATION_M,
        "qpos_writes_during_rollout": qpos_assignment_count_during_rollout,
        "settle_window_samples": len(stable_rows),
        "settled_position_error_limit_rad": SETTLED_POSITION_ERROR_RAD,
        "settled_velocity_limit_rad_s": SETTLED_VELOCITY_RAD_S,
        "motion_joints": MOTION_JOINTS,
        "motion_source_targets_in_limits": True,
        "actuation": "SIMULATION_ONLY torque-limited independent driver motors; official source joint-equality mimic constraints retained; no follower actuators",
        "actuator_joint_count": len(hand_actuator),
        "follower_actuator_count": 0,
        "mimic_constraint_correction": mimic_constraint_correction,
        "natural_frequency_rad_s": NATURAL_FREQUENCY_RAD_S,
        "damping_ratio": 1.0,
        "torque_limit_nm": torque_limit,
        "gains": gains,
        "gravity_projected_loads_nm_at_open": bias_loads,
        "baseline_trace_analysis": baseline,
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "video": str(video_path),
        "video_sha256": sha256(video_path),
    }
    return motion


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #46 source-mimic SIMULATION_ONLY transmission motion probe.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-trace",
        type=Path,
        default=Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/physical-probe-02/physics_contact_trace.jsonl"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    baseline_path = args.baseline_trace.resolve() if args.baseline_trace else None
    motion = run_motion(output, baseline_path)
    runtime = {
        "robotsim_branch": subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip(),
        "robotsim_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "robotsim_dirty_paths": subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).splitlines(),
        "probe_sha256": sha256(Path(__file__).resolve()),
        "vendor_head": subprocess.check_output(["git", "-C", str(task.ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_dirty_paths": subprocess.check_output(["git", "-C", str(task.ROOT), "status", "--porcelain"], text=True).splitlines(),
        "urdf_sha256": sha256(task.URDF),
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "timestep_s": DT,
        "model": "full pinned X2 with both official OmniHands; no bottle during motion gate",
        "mimic_relations": 12,
        "follower_qpos_writes_during_rollout": 0,
    }
    result = {"runtime": runtime, "motion_gate": motion}
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = f"AGIBOT_X2_VENDOR_ROOT={shlex.quote(str(task.ROOT))} {shlex.join(sys.argv)}"
    (output / "reproduction_command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps({"status": motion["status"], "first_failure": motion["first_failure"], "output": str(output)}, indent=2))
    return 0 if motion["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
