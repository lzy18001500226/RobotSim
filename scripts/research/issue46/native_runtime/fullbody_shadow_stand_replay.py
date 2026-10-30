#!/usr/bin/env python3
"""Replay source MC standing commands in a full-contact X2 MuJoCo shadow scene.

The delivered RAICOM ``sim_mujoco`` binary loses its complete joint-state
mapping whenever an extra world geom is added.  This research-only bridge keeps
the native MC scene unchanged, takes a hash-validated native STAND command/state
capture, and applies those *recorded* position/stiffness commands as torques to
a separate full-contact MuJoCo model containing the source RAICOM table and
true-pose cylinder.  It never writes qpos/qvel after the first ``mj_step``.

It proves neither online MC state feedback nor a grasp.  Its sole purpose is a
bounded physical standing preflight before arm planning and contact work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from scripts.research.issue46.native_runtime.stl_hand_startup import (  # noqa: E402
    load_source_arm_pd_parameters,
)
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    load_registered_trajectory,
)


LEG_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
WAIST_ORDER = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
ARM_ORDER = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
RIGHT_ARM_ORDER = ARM_ORDER[7:]
ROOTS = {"left": "L_claw_joint", "right": "R_claw_joint"}
MAX_SECONDS = 12.0
MAX_STEPS = 12_000
TILT_GATE_RAD = math.radians(15.0)
DRIFT_GATE_M = 0.10
QVEL_GATE_RAD_S = 30.0
BASE_LINEAR_SPEED_GATE_M_S = 3.0
BASE_ANGULAR_SPEED_GATE_RAD_S = 30.0
BASE_HEIGHT_MIN_M = 0.20
BASE_HEIGHT_MAX_M = 1.20
LIMIT_TOLERANCE_RAD = 1e-5
INITIALIZATION_LIMIT_MARGIN_RAD = 1e-5
MAX_INITIALIZATION_CLAMP_RAD = 1e-3
INITIAL_STL_COMMAND_PERCENT = 99.0
COMMUNITY_READY_ARM_POSITIONS = {
    "left": (-0.35, 0.45, 0.0, -1.0, 0.0, 0.15, 0.0),
    "right": (-0.35, -0.45, 0.0, -1.0, 0.0, 0.15, 0.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_result(path: Path, expected_sha256: str, *, label: str) -> dict[str, object]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    if sha256(path) != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def generated_file(result: dict[str, object], name: str) -> Path:
    item = result.get("generated", {}).get(name) if isinstance(result.get("generated"), dict) else None
    if not isinstance(item, dict):
        raise ValueError(f"table overlay lacks generated {name}")
    path = Path(str(item.get("path", "")))
    expected = str(item.get("sha256", ""))
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"generated {name} is unavailable or hash-mismatched")
    return path


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0 or int(model.jnt_qposadr[joint]) < 0:
        raise ValueError(f"missing scalar joint: {name}")
    data.qpos[int(model.jnt_qposadr[joint])] = value


def set_initial_joint_inside_limits(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    value: float,
) -> dict[str, float] | None:
    """Use an interior initial state without relaxing a source joint range."""
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"missing joint: {name}")
    initialized = float(value)
    if bool(model.jnt_limited[joint]):
        low, high = (float(item) for item in model.jnt_range[joint])
        initialized = min(max(initialized, low + INITIALIZATION_LIMIT_MARGIN_RAD), high - INITIALIZATION_LIMIT_MARGIN_RAD)
        if abs(initialized - value) > MAX_INITIALIZATION_CLAMP_RAD:
            raise ValueError(f"captured {name} exceeds its source limit by more than the permitted initialization margin")
    set_joint(model, data, name, initialized)
    if initialized != value:
        return {"source_rad": float(value), "initialized_rad": initialized, "delta_rad": initialized - float(value)}
    return None


def joint_state(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[float, float]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"missing joint: {name}")
    return float(data.qpos[model.jnt_qposadr[joint]]), float(data.qvel[model.jnt_dofadr[joint]])


def actuator_for_joint(model: mujoco.MjModel, joint_name: str) -> int:
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + joint_name)
    if actuator < 0:
        raise ValueError(f"missing torque actuator for {joint_name}")
    return actuator


def set_free_pelvis_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, object]) -> dict[str, float]:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ValueError("full-contact scene lacks pelvis")
    free = next((joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) == pelvis and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)), -1)
    if free < 0:
        raise ValueError("pelvis has no free joint")
    xyz = np.asarray(pose.get("xyz_m"), dtype=float)
    quat = np.asarray(pose.get("quaternion_wxyz"), dtype=float)
    if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) <= 1e-8:
        raise ValueError("captured base pose is invalid")
    qadr = int(model.jnt_qposadr[free])
    compiled_z = float(data.qpos[qadr + 2])
    data.qpos[qadr:qadr + 3] = (xyz[0], xyz[1], compiled_z)
    data.qpos[qadr + 3:qadr + 7] = quat / np.linalg.norm(quat)
    return {"captured_odom_z_m": float(xyz[2]), "initialized_source_scene_z_m": compiled_z}


def roll_pitch_from_quaternion(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def limited_joint_violations(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    violations: dict[str, float] = {}
    for joint in range(model.njnt):
        if not bool(model.jnt_limited[joint]) or int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        value = float(data.qpos[model.jnt_qposadr[joint]])
        low, high = model.jnt_range[joint]
        excess = max(float(low - value), float(value - high), 0.0)
        if excess > LIMIT_TOLERANCE_RAD:
            violations[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint#{joint}"] = excess
    return violations


def maximum_named_joint_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[str, float]:
    maximum_name, maximum_value = "", -1.0
    for joint in range(model.njnt):
        start = int(model.jnt_dofadr[joint])
        end = int(model.jnt_dofadr[joint + 1]) if joint + 1 < model.njnt else model.nv
        if end <= start:
            continue
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        value = float(np.max(np.abs(data.qvel[start:end])))
        if value > maximum_value:
            maximum_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or f"joint#{joint}"
            maximum_value = value
    return maximum_name, maximum_value


def pelvis_free_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    free = next((joint for joint in range(model.njnt) if int(model.jnt_bodyid[joint]) == pelvis and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)), -1)
    if free < 0:
        raise ValueError("pelvis has no free joint")
    dadr = int(model.jnt_dofadr[free])
    return data.qvel[dadr:dadr + 3].copy(), data.qvel[dadr + 3:dadr + 6].copy()


def table_robot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    contacts = []
    for contact in data.contact:
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1])) or ""
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2])) or ""
        if "raicom_table_" in first or "raicom_table_" in second:
            other = second if "raicom_table_" in first else first
            if other != "test_cylinder":
                contacts.append({"body_names": [first, second], "distance_m": float(contact.dist)})
    return contacts


def apply_pd_torques(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    commands: dict[str, dict[str, float]],
    *,
    gravity_compensation: bool,
) -> None:
    """Translate delivered JointCommand PD fields into source-range motor torques."""
    for name, command in commands.items():
        q, qvel = joint_state(model, data, name)
        torque = (
            float(command["effort"])
            + float(command["stiffness"]) * (float(command["position"]) - q)
            + float(command["damping"]) * (float(command["velocity"]) - qvel)
        )
        if gravity_compensation:
            torque += float(data.qfrc_bias[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
        actuator = actuator_for_joint(model, name)
        low, high = model.actuator_ctrlrange[actuator]
        data.ctrl[actuator] = np.clip(torque, low, high)


def apply_native_stl_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: object,
    *,
    side: str,
    command_percent: float,
) -> dict[str, float]:
    """Apply one reviewed STL row to the delivered L_/R_ runtime aliases."""
    pose = trajectory.pose(command_percent)
    mapping = {
        "right_claw_joint": ROOTS[side],
        "R_hand_wide1_joint": f"{'L' if side == 'left' else 'R'}_hand_wide1_joint",
        "R_hand_narrow3_joint": f"{'L' if side == 'left' else 'R'}_hand_narrow3_joint",
        "R_hand_narrow_loop_joint": f"{'L' if side == 'left' else 'R'}_hand_narrow_loop_joint",
        "R_hand_wide3_joint": f"{'L' if side == 'left' else 'R'}_hand_wide3_joint",
        "R_hand_wide_loop_joint": f"{'L' if side == 'left' else 'R'}_hand_wide_loop_joint",
    }
    resolved: dict[str, float] = {}
    for source_name, value in pose.positions_rad.items():
        target = mapping[source_name]
        set_joint(model, data, target, value)
        resolved[target] = value
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-overlay-result", required=True, type=Path)
    parser.add_argument("--table-overlay-result-sha256", required=True)
    parser.add_argument("--native-stand-result", required=True, type=Path)
    parser.add_argument("--native-stand-result-sha256", required=True)
    parser.add_argument("--ik-validation-result", type=Path, help="hash-validated static right-arm pregrasp target")
    parser.add_argument("--ik-validation-result-sha256", help="hash paired with --ik-validation-result")
    parser.add_argument("--arm-ramp-seconds", type=float, default=8.0)
    parser.add_argument("--arm-hold-seconds", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--gravity-compensation", action="store_true", help="SIMULATION_ONLY: add MuJoCo qfrc_bias feedforward to the unchanged recorded MC PD command")
    parser.add_argument("--community-ready-arm-pose", action="store_true", help="SIMULATION_ONLY: initialize arms at the documented community X2 IK ready pose, inside source joint ranges")
    parser.add_argument("--execute-dynamics", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.hold_seconds <= MAX_SECONDS:
        raise SystemExit(f"hold seconds must be in [0.1,{MAX_SECONDS:g}]")
    if (args.ik_validation_result is None) != (args.ik_validation_result_sha256 is None):
        raise SystemExit("IK result path and SHA-256 must be supplied together")
    if args.ik_validation_result is not None and (not 4.0 <= args.arm_ramp_seconds <= 10.0 or not 0.2 <= args.arm_hold_seconds <= 2.0 or args.hold_seconds < args.arm_ramp_seconds + args.arm_hold_seconds):
        raise SystemExit("bounded arm-ramp timing is invalid")
    table = require_result(args.table_overlay_result, args.table_overlay_result_sha256, label="table overlay result")
    native = require_result(args.native_stand_result, args.native_stand_result_sha256, label="native stand result")
    ik_result = None if args.ik_validation_result is None else require_result(args.ik_validation_result, args.ik_validation_result_sha256 or "", label="IK validation result")
    if table.get("classification") != "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED":
        raise SystemExit("table overlay has the wrong classification")
    table_gates = table.get("gates", {})
    if not isinstance(table_gates, dict) or any(table_gates.get(name) != "PASS" for name in ("FROZEN_STL_MOUNT_INPUT", "HOST_STATIC_MODEL_COMPILE", "TABLE_SCENE_STATIC_COMPILE", "TRUE_POSE_CYLINDER_PRESENT")):
        raise SystemExit("table overlay did not pass static true-pose preflight")
    native_gates = native.get("gates", {})
    if not isinstance(native_gates, dict) or any(native_gates.get(name) != "PASS" for name in ("NATIVE_STAND_REQUEST", "NATIVE_STAND_READBACK", "NATIVE_STAND_STABILITY", "MC_STAND_COMMAND_CAPTURE")):
        raise SystemExit("native stand capture did not pass")
    captured_state = native.get("native_arm_state_capture", {}).get("result") if isinstance(native.get("native_arm_state_capture"), dict) else None
    captured_commands = native.get("native_stand_command_capture", {}).get("result") if isinstance(native.get("native_stand_command_capture"), dict) else None
    if not isinstance(captured_state, dict) or not isinstance(captured_commands, dict):
        raise SystemExit("native result lacks captured state or command stream")
    if ik_result is not None:
        ik_gates = ik_result.get("gates", {})
        right_target = ik_result.get("ik", {}).get("right_arm_q_rad") if isinstance(ik_result.get("ik"), dict) else None
        if ik_result.get("classification") != "SIMULATION_ONLY_STATIC_EXTERNAL_IK_REFERENCE" or not isinstance(ik_gates, dict) or ik_gates.get("IK_CONVERGED") != "PASS" or ik_gates.get("POST_IK_TABLE_CLEARANCE") != "PASS" or not isinstance(right_target, list) or len(right_target) != len(RIGHT_ARM_ORDER) or not all(math.isfinite(float(value)) for value in right_target):
            raise SystemExit("IK validation result is not a passing right-arm pregrasp reference")
    scene = generated_file(table, "host_scene")
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-fullbody-shadow-stand").native_path
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    base_initialization = set_free_pelvis_pose(model, data, dict(captured_state.get("base_pose_world", {})))
    initialization_clamps: dict[str, dict[str, float]] = {}
    for group, names in (("leg_positions_rad", LEG_ORDER), ("waist_positions_rad", WAIST_ORDER)):
        values = captured_state.get(group)
        if not isinstance(values, dict) or any(name not in values for name in names):
            raise SystemExit(f"native stand state lacks {group}")
        for name in names:
            clamp = set_initial_joint_inside_limits(model, data, name, float(values[name]))
            if clamp is not None:
                initialization_clamps[name] = clamp
    arm_commands = captured_commands.get("arm_commands")
    if not isinstance(arm_commands, dict) or any(name not in arm_commands or not isinstance(arm_commands[name], dict) for name in ARM_ORDER):
        raise SystemExit("native stand capture lacks arm commands")
    arm_initialization: dict[str, dict[str, float]] = {}
    selected_arm_targets = dict(zip(ARM_ORDER[:7], COMMUNITY_READY_ARM_POSITIONS["left"])) | dict(zip(RIGHT_ARM_ORDER, COMMUNITY_READY_ARM_POSITIONS["right"])) if args.community_ready_arm_pose else {name: float(arm_commands[name]["position"]) for name in ARM_ORDER}
    for name in ARM_ORDER:
        target = selected_arm_targets[name]
        source = float(captured_state["arm_positions_rad"][name])
        clamp = set_initial_joint_inside_limits(model, data, name, target)
        if clamp is not None:
            initialization_clamps[name] = clamp
            target = clamp["initialized_rad"]
        arm_initialization[name] = {"native_state_rad": source, "mc_command_position_rad": float(arm_commands[name]["position"]), "initialized_rad": target}
    trajectory = load_registered_trajectory(REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic")
    for side in ("left", "right"):
        pose = apply_native_stl_pose(model, data, trajectory, side=side, command_percent=INITIAL_STL_COMMAND_PERCENT)
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{'L' if side == 'left' else 'R'}_claw_joint")
        if actuator < 0:
            raise SystemExit(f"STL root actuator absent: {side}")
        # The delivered position actuator uses gear=-1 and ctrlrange [0,1]:
        # the reviewed command percentage, rather than the negative hinge
        # angle, is its physical command coordinate.
        data.ctrl[actuator] = INITIAL_STL_COMMAND_PERCENT / 100.0
    leg_commands = captured_commands.get("leg_commands")
    waist_commands = captured_commands.get("waist_commands")
    if not isinstance(leg_commands, dict) or not isinstance(waist_commands, dict):
        raise SystemExit("native stand capture lacks leg/waist commands")
    commands: dict[str, dict[str, float]] = {}
    for name in (*LEG_ORDER, *WAIST_ORDER):
        raw = (leg_commands if name in LEG_ORDER else waist_commands).get(name)
        if not isinstance(raw, dict) or any(key not in raw for key in ("position", "velocity", "effort", "stiffness", "damping")):
            raise SystemExit(f"native stand command lacks {name}")
        commands[name] = {key: float(raw[key]) for key in ("position", "velocity", "effort", "stiffness", "damping")}
    arm_gains = load_source_arm_pd_parameters()
    for name in ARM_ORDER:
        commands[name] = {
            "position": arm_initialization[name]["initialized_rad"],
            "velocity": 0.0,
            "effort": 0.0,
            "stiffness": arm_gains[name][0],
            "damping": arm_gains[name][1],
        }
    initial_arm_targets = {name: commands[name]["position"] for name in ARM_ORDER}
    right_pregrasp_target = None if ik_result is None else {name: float(value) for name, value in zip(RIGHT_ARM_ORDER, ik_result["ik"]["right_arm_q_rad"])}
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    initial_pelvis = data.xpos[pelvis].copy()
    initial_contacts = table_robot_contacts(model, data)
    initial_limits = limited_joint_violations(model, data)
    pregrasp_static: dict[str, object] | None = None
    if right_pregrasp_target is not None:
        for name, value in right_pregrasp_target.items():
            set_joint(model, data, name, value)
        mujoco.mj_forward(model, data)
        pregrasp_static = {
            "right_arm_target_rad": right_pregrasp_target,
            "finite_qpos": bool(np.isfinite(data.qpos).all()),
            "joint_limit_violations": limited_joint_violations(model, data),
            "table_robot_contacts": table_robot_contacts(model, data),
        }
        for name, value in initial_arm_targets.items():
            set_joint(model, data, name, value)
        mujoco.mj_forward(model, data)
    preflight = {
        "scene_nq": int(model.nq), "scene_nv": int(model.nv), "scene_nu": int(model.nu), "scene_neq": int(model.neq),
        "finite_qpos": bool(np.isfinite(data.qpos).all()), "finite_qvel": bool(np.isfinite(data.qvel).all()),
        "initial_table_robot_contacts": initial_contacts, "initial_joint_limit_violations": initial_limits,
        "recorded_mc_joint_count": len(LEG_ORDER) + len(WAIST_ORDER), "source_arm_hold_joint_count": len(ARM_ORDER),
        "base_initialization": base_initialization,
        "initial_stl_command_percent": INITIAL_STL_COMMAND_PERCENT,
        "initialization_limit_clamps": initialization_clamps,
        "arm_initialization": arm_initialization,
    }
    pregrasp_static_pass = pregrasp_static is None or (pregrasp_static["finite_qpos"] and not pregrasp_static["joint_limit_violations"] and not pregrasp_static["table_robot_contacts"])
    preflight_pass = preflight["finite_qpos"] and preflight["finite_qvel"] and not initial_contacts and not initial_limits and pregrasp_static_pass
    trace: list[dict[str, object]] = []
    failure: dict[str, object] | None = None
    steps = min(MAX_STEPS, int(round(args.hold_seconds / model.opt.timestep)))
    if args.execute_dynamics and preflight_pass:
        for step in range(steps):
            active_commands = {name: dict(command) for name, command in commands.items()}
            if right_pregrasp_target is not None:
                phase = min(1.0, float(data.time) / args.arm_ramp_seconds)
                blend = 3.0 * phase * phase - 2.0 * phase * phase * phase
                for name in RIGHT_ARM_ORDER:
                    active_commands[name]["position"] = initial_arm_targets[name] + blend * (right_pregrasp_target[name] - initial_arm_targets[name])
            apply_pd_torques(model, data, active_commands, gravity_compensation=args.gravity_compensation)
            mujoco.mj_step(model, data)
            roll, pitch = roll_pitch_from_quaternion(data.xquat[pelvis])
            max_qvel_name, max_qvel_value = maximum_named_joint_velocity(model, data)
            base_linear_velocity, base_angular_velocity = pelvis_free_velocity(model, data)
            record = {
                "step": step + 1, "time_s": float(data.time),
                "pelvis_xy_drift_m": float(np.linalg.norm(data.xpos[pelvis, :2] - initial_pelvis[:2])),
                "pelvis_tilt_rad": max(abs(roll), abs(pitch)),
                "pelvis_height_m": float(data.xpos[pelvis, 2]),
                "base_linear_velocity_m_s": base_linear_velocity.tolist(),
                "base_angular_velocity_rad_s": base_angular_velocity.tolist(),
                "base_linear_speed_m_s": float(np.linalg.norm(base_linear_velocity)),
                "base_angular_speed_rad_s": float(np.linalg.norm(base_angular_velocity)),
                "max_abs_qvel_rad_s": max_qvel_value,
                "max_abs_qvel_joint": max_qvel_name,
                "joint_limit_violations": limited_joint_violations(model, data),
                "table_robot_contacts": table_robot_contacts(model, data),
                "max_right_arm_tracking_error_rad": max(abs(joint_state(model, data, name)[0] - active_commands[name]["position"]) for name in RIGHT_ARM_ORDER),
            }
            if step % 10 == 0 or step + 1 == steps:
                trace.append(record)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = {"gate": "NONFINITE_STATE", **record}
            elif record["pelvis_tilt_rad"] > TILT_GATE_RAD:
                failure = {"gate": "PELVIS_TILT", **record}
            elif record["pelvis_xy_drift_m"] > DRIFT_GATE_M:
                failure = {"gate": "PELVIS_DRIFT", **record}
            elif record["max_abs_qvel_rad_s"] > QVEL_GATE_RAD_S:
                failure = {"gate": "EXCESSIVE_QVEL", **record}
            elif record["base_linear_speed_m_s"] > BASE_LINEAR_SPEED_GATE_M_S:
                failure = {"gate": "EXCESSIVE_BASE_LINEAR_SPEED", **record}
            elif record["base_angular_speed_rad_s"] > BASE_ANGULAR_SPEED_GATE_RAD_S:
                failure = {"gate": "EXCESSIVE_BASE_ANGULAR_SPEED", **record}
            elif not BASE_HEIGHT_MIN_M <= record["pelvis_height_m"] <= BASE_HEIGHT_MAX_M:
                failure = {"gate": "BASE_HEIGHT", **record}
            elif record["joint_limit_violations"]:
                failure = {"gate": "JOINT_LIMIT", **record}
            elif record["table_robot_contacts"]:
                failure = {"gate": "TABLE_ROBOT_CONTACT", **record}
            elif right_pregrasp_target is not None and record["max_right_arm_tracking_error_rad"] > 0.15:
                failure = {"gate": "ARM_TRACKING", **record}
            if failure:
                break
    status = "NOT_RUN" if not args.execute_dynamics else ("PASS" if preflight_pass and failure is None else "FAIL")
    result = {
        "schema": "robotsim.issue46.raicom-stl-fullbody-shadow-stand.v1",
        "classification": "SIMULATION_ONLY_FULL_CONTACT_SHADOW_STAND_REPLAY",
        "mj_step_calls": int(round(data.time / model.opt.timestep)) if args.execute_dynamics else 0,
        "table_overlay": {"path": str(args.table_overlay_result), "sha256": sha256(args.table_overlay_result)},
        "native_stand_capture": {"path": str(args.native_stand_result), "sha256": sha256(args.native_stand_result)},
        "ik_validation": None if args.ik_validation_result is None else {"path": str(args.ik_validation_result), "sha256": sha256(args.ik_validation_result)},
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "commands": {"native_mc_position_pd": commands, "arm_hold_source": "AgiBot-Community raicom2026-agibot-X2 X2IKConfig.ready_arm_pos" if args.community_ready_arm_pose else "delivered remote_controller_config/model_config.yaml", "mujoco_gravity_compensation": args.gravity_compensation, "community_ready_arm_pose": args.community_ready_arm_pose},
        "preflight": preflight,
        "pregrasp_static": pregrasp_static,
        "trace": trace,
        "failure": failure,
        "gates": {"HASH_VALIDATED_INPUTS": "PASS", "STATIC_FULL_CONTACT_STATE": "PASS" if preflight_pass else "FAIL", "PREGRASP_STATIC_CLEARANCE": "NOT_RUN" if pregrasp_static is None else ("PASS" if pregrasp_static_pass else "FAIL"), "FULLBODY_SHADOW_STAND_REPLAY": status, "PREGRASP_ARM_RAMP": "NOT_RUN" if right_pregrasp_target is None else status, "ONLINE_MC_STATE_FEEDBACK": "NOT_RUN", "OBJECT_CONTACT": "NOT_RUN", "GRASP": "NOT_RUN"},
        "limitations": [
            "The vendor sim_mujoco adapter cannot retain a state map after any added world geom; native MC therefore runs only in its immutable source scene.",
            "This full-contact shadow uses recorded native MC leg/waist commands and delivered arm PD values, not a live state-feedback bridge.",
            "When enabled, qfrc_bias feedforward is a SIMULATION_ONLY MuJoCo control overlay, not a claim about vendor actuator firmware.",
            "No qpos/qvel is written after the first mj_step. No object motion, grasp, or lift is claimed.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "mj_step_calls": result["mj_step_calls"], "failure": failure}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
