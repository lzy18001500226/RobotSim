#!/usr/bin/env python3
"""One bounded, contact-driven full-X2 horizontal cylinder grasp episode.

The runner consumes a hash-validated static preflight.  It never writes a
follower joint or cylinder qpos after initialization, and it stops at the first
contact, stability, range, tracking, or numerical safety failure.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import imageio_ffmpeg
import mujoco
import numpy as np


WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
from agent import path_governance as pg


EXPECTED_FROZEN_SHA = "5e33d30e212ee72271efc3f0a790ebfb46cd31e775c578a10f1a2fb1f33c93e7"
MAX_STEPS = 20_000
MAX_TIME_S = 20.0
ROOT_TRACKING_GATE_RAD = 0.020
EQUALITY_JOINT_GATE_RAD = 0.003
EQUALITY_CONNECT_GATE_M = 0.001
# MuJoCo's implicit integrator can report an endpoint displacement of roughly
# 1e-9 rad while the target, velocity and actuator effort remain at the source
# limit.  This comparison tolerance does not change the model range.
JOINT_LIMIT_NUMERICAL_TOL_RAD = 1e-8
SETTLE_MAX_STEPS = 2_000
SETTLE_REQUIRED_STABLE_STEPS = 500
PREGRASP_HOLD_STEPS = 250
APPROACH_STEPS = 2_000
CONTACT_POSE_STEPS = 1_500
CLOSE_STEPS = 2_000
CONTACT_HOLD_STEPS = 500
LIFT_STEPS = 3_000
FINAL_HOLD_STEPS = 500
VIDEO_EVERY = 20
VIDEO_FPS = 25
SETTLE_LINEAR_SPEED_M_S = 0.020
SETTLE_ANGULAR_SPEED_RAD_S = 0.050
SETTLE_SUPPORT_RATIO = (0.80, 1.20)
MIN_CONTACT_FRAMES = 100
MIN_LOAD_TRANSFER_RATIO = 0.20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(filename: str, module_name: str):
    path = WORKTREE / "scripts/research" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def name(model: mujoco.MjModel, object_type, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(index)) or f"unnamed_{int(index)}"


def source_limits(path: Path) -> dict[str, dict[str, float | None]]:
    root = ET.parse(path).getroot()
    result = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            continue
        def value(key: str) -> float | None:
            raw = limit.get(key)
            return None if raw is None else float(raw)
        velocity = value("velocity")
        result[joint.get("name", "")] = {
            "lower": value("lower"), "upper": value("upper"),
            "velocity": velocity if velocity is not None and velocity > 0.0 else None,
            "velocity_source_value": velocity,
        }
    return result


def quintic(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha ** 3 - 15.0 * alpha ** 4 + 6.0 * alpha ** 5


def interpolate(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - quintic(alpha)) * first + quintic(alpha) * second


def outside_position_range(qpos: float, lower: float, upper: float, tolerance_rad: float = JOINT_LIMIT_NUMERICAL_TOL_RAD) -> bool:
    """Keep source limits intact while ignoring IEEE round-off at an endpoint."""
    return qpos < lower - tolerance_rad or qpos > upper + tolerance_rad


def all_equalities(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict[str, float], list[dict[str, float]]]:
    joint = {}
    connect = []
    for equality in range(model.neq):
        eq_name = name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality)
        if int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_JOINT):
            first, second = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            q_first = float(data.qpos[model.jnt_qposadr[first]])
            q_second = float(data.qpos[model.jnt_qposadr[second]])
            coefficient = model.eq_data[equality, :5]
            joint[eq_name] = q_first - sum(float(coefficient[index]) * q_second ** index for index in range(5))
    for row in range(data.nefc):
        if int(data.efc_type[row]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        equality = int(data.efc_id[row])
        if int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_CONNECT):
            connect.append({"name": name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality), "residual_m": float(data.efc_pos[row])})
    return joint, connect


def object_state(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> dict[str, object]:
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    return {
        "com_world_m": data.xipos[body].tolist(),
        "linear_velocity_world_m_s": velocity[3:].tolist(),
        "angular_velocity_world_rad_s": velocity[:3].tolist(),
        "linear_speed_m_s": float(np.linalg.norm(velocity[3:])),
        "angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
    }


def collect_contacts(model: mujoco.MjModel, data: mujoco.MjData, pads: set[str]) -> dict[str, object]:
    cylinder = model.body("test_cylinder").id
    records = []
    table_force = np.zeros(3)
    pad_forces: dict[str, np.ndarray] = {pad: np.zeros(3) for pad in pads}
    table_count = 0
    pad_bodies = set()
    forbidden = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_a, geom_b = int(contact.geom1), int(contact.geom2)
        body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
        body_name_a = name(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
        body_name_b = name(model, mujoco.mjtObj.mjOBJ_BODY, body_b)
        geom_name_a = name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
        geom_name_b = name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        frame = np.asarray(contact.frame).reshape(3, 3)
        force_on_geom_b = frame.T @ wrench[:3]
        force_on_cylinder = None
        if body_a == cylinder:
            force_on_cylinder = -force_on_geom_b
        elif body_b == cylinder:
            force_on_cylinder = force_on_geom_b
        pair = {body_name_a, body_name_b}
        category = "other"
        if pair == {"test_cylinder", "raicom_table"}:
            category = "table_support"
            table_count += 1
            if force_on_cylinder is not None:
                table_force += force_on_cylinder
        elif "test_cylinder" in pair and pair & pads:
            category = "distal_pad"
            pad = next(item for item in pair if item in pads)
            pad_bodies.add(pad)
            if force_on_cylinder is not None:
                pad_forces[pad] += force_on_cylinder
        elif "head_pitch_link" in pair and "torso_link" in pair:
            category = "forbidden_head_torso"
            forbidden.append(index)
        elif "test_cylinder" in pair:
            category = "forbidden_cylinder_robot"
            forbidden.append(index)
        elif "raicom_table" in pair:
            category = "forbidden_robot_table"
            forbidden.append(index)
        elif "world" not in pair:
            category = "forbidden_robot_self"
            forbidden.append(index)
        records.append({
            "index": index, "category": category, "body_names": [body_name_a, body_name_b],
            "geom_names": [geom_name_a, geom_name_b], "distance_m": float(contact.dist),
            "force_contact_frame_N_Nm": wrench.tolist(),
            "force_on_cylinder_world_N": None if force_on_cylinder is None else force_on_cylinder.tolist(),
        })
    return {
        "records": records, "forbidden_indices": forbidden, "table_contact_count": table_count,
        "table_force_on_cylinder_world_N": table_force.tolist(),
        "pad_forces_on_cylinder_world_N": {key: value.tolist() for key, value in pad_forces.items()},
        "pad_bodies": sorted(pad_bodies),
    }


def full_joint_row(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    output = []
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        nq = 7 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else (4 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
        nv = 6 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else (3 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
        output.extend(data.qpos[int(model.jnt_qposadr[joint]):int(model.jnt_qposadr[joint]) + nq])
        output.extend(data.qvel[int(model.jnt_dofadr[joint]):int(model.jnt_dofadr[joint]) + nv])
    return [float(item) for item in output]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-result", required=True, type=Path)
    parser.add_argument("--frozen-model", required=True, type=Path)
    parser.add_argument("--full-urdf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute-dynamics", action="store_true")
    args = parser.parse_args()
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite trial evidence")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "STATIC_READY_FOR_ONE_DYNAMICS_EPISODE" or any(value != "PASS" for key, value in config.get("gates", {}).items() if key != "DYNAMICS"):
        raise SystemExit("static preflight is not fully PASS")
    scene = Path(config["scene"]).resolve()
    if sha256(scene) != config["scene_sha256"] or sha256(args.frozen_model) != EXPECTED_FROZEN_SHA:
        raise SystemExit("scene or frozen model hash mismatch")
    if not args.execute_dynamics:
        print(json.dumps({"preflight_only": True, "mj_step_calls": 0, "scene_sha256": config["scene_sha256"], "controller_overlay": config["controller_overlay"]}, indent=2))
        return
    if mujoco.__version__ != "3.3.6":
        raise SystemExit(f"expected MuJoCo 3.3.6, got {mujoco.__version__}")
    preflight = load_module("issue46_full_x2_horizontal_cylinder_preflight.py", "full_x2_preflight")
    controller = load_module("issue46_full_x2_cylinder_controller_preflight.py", "full_x2_controller")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if abs(float(model.opt.timestep) - 0.001) > 1e-12:
        raise SystemExit("unexpected timestep")
    data = mujoco.MjData(model)
    initial = next(item for item in config["static_states"] if item["phase"] == "INITIAL_SETTLE")
    pregrasp = next(item for item in config["static_states"] if item["phase"] == "HORIZONTAL_PREGRASP")
    approach = next(item for item in config["static_states"] if item["phase"] == "HORIZONTAL_APPROACH")
    contact = next(item for item in config["static_states"] if item["phase"] == "CLOSE_75_PERCENT")
    lift = next(item for item in config["static_states"] if item["phase"] == "VERTICAL_LIFT_50MM")
    original = json.loads(args.preflight_result.read_text(encoding="utf-8"))
    waypoints = original["waypoints"]
    arm_initial = np.asarray(waypoints["HORIZONTAL_PREGRASP"]["arm_q_rad"], dtype=float)
    arm_approach = np.asarray(waypoints["HORIZONTAL_APPROACH"]["arm_q_rad"], dtype=float)
    arm_contact = np.asarray(waypoints["HORIZONTAL_CONTACT"]["arm_q_rad"], dtype=float)
    arm_lift = np.asarray(waypoints["VERTICAL_LIFT_50MM"]["arm_q_rad"], dtype=float)
    preflight.set_static_state(model, data, arm_initial, initial["right_hand_qpos_rad"], initial["left_hand_qpos_rad"])
    motor_actuators = controller.all_motor_actuators(model)
    motor_targets = dict(initial["motor_targets_rad"])
    # Static preflight may intentionally move idle torque-motor joints one
    # numerical margin inside a source endpoint.  Reapply that exact initial
    # state before physics begins; no qpos is written after this block.
    for joint_name, value in motor_targets.items():
        joint = model.joint(joint_name).id
        data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    arm_names = list(preflight.ARM)
    right_root = float(config["right_root_open_target_rad"])
    left_root = float(config["left_root_idle_target_rad"])
    pads = set(preflight.PAD_BODIES)
    cylinder = model.body("test_cylinder").id
    cylinder_joint = model.joint("test_cylinder_free").id
    cylinder_dofs = set(range(int(model.jnt_dofadr[cylinder_joint]), int(model.jnt_dofadr[cylinder_joint]) + 6))
    source = source_limits(args.full_urdf)
    root_actuator = model.actuator("servo_right_claw_joint").id
    left_actuator = model.actuator("servo_left_claw_joint").id
    initial_com = data.xipos[cylinder].copy()
    weight = float(model.body_mass[cylinder]) * abs(float(model.opt.gravity[2]))
    if abs(float(model.body_mass[cylinder]) - 0.20) > 1e-12:
        raise SystemExit("cylinder mass changed")

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [1.43, -0.92, 0.80]
    camera.distance = 1.45
    camera.azimuth = 165
    camera.elevation = -18
    video = output / "full_x2_horizontal_cylinder_actual_mujoco.mp4"
    writer = imageio_ffmpeg.write_frames(str(video), (640, 480), fps=VIDEO_FPS, quality=7, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", output_params=["-preset", "ultrafast", "-movflags", "+faststart"])
    writer.send(None)
    trace = gzip.open(output / "joint_actuator_trace.csv.gz", "wt", newline="", encoding="utf-8")
    joint_header = []
    for joint in range(model.njnt):
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        kind = int(model.jnt_type[joint])
        nq = 7 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else (4 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
        nv = 6 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else (3 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
        joint_header.extend([f"qpos_{joint_name}_{index}" for index in range(nq)])
        joint_header.extend([f"qvel_{joint_name}_{index}" for index in range(nv)])
    trace_writer = csv.writer(trace)
    actuator_names = [name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) for actuator in range(model.nu)]
    trace_writer.writerow(["step", "time_s", "stage"] + joint_header + [f"ctrl_{item}" for item in actuator_names] + [f"force_{item}" for item in actuator_names])
    contact_trace = gzip.open(output / "contact_force_trace.jsonl.gz", "wt", encoding="utf-8")
    summary_trace = gzip.open(output / "phase_summary_trace.csv.gz", "wt", newline="", encoding="utf-8")
    summary_writer = csv.writer(summary_trace)
    summary_writer.writerow(["step", "time_s", "stage", "cylinder_com_z_m", "cylinder_lift_m", "linear_speed_m_s", "angular_speed_rad_s", "table_force_z_N", "right_pad_force_norm_N", "left_pad_force_norm_N", "root_tracking_error_rad", "max_joint_equality_rad", "max_connect_equality_m", "forbidden_contacts"])

    metrics = {
        "max_cylinder_com_rise_m": 0.0, "max_root_tracking_error_rad": 0.0,
        "max_joint_equality_rad": 0.0, "max_connect_equality_m": 0.0,
        "max_actuator_force": {}, "max_requested_motor_ctrl": {},
        "first_bilateral_step": None, "first_load_transfer_step": None,
        "head_torso_contact_count": 0, "controller_clipping_events": 0,
    }
    first_failure = None
    stage_results = {stage: "NOT RUN" for stage in ["INITIAL_SETTLE", "HORIZONTAL_PREGRASP", "APPROACH", "CLOSE", "BILATERAL_DISTAL_PAD_CONTACT", "LOAD_TRANSFER", "LIFT_50MM", "HOLD_0_5S"]}
    bilateral_streak = 0
    transfer_streak = 0
    settle_streak = 0
    physics_steps = 0
    post_start_cylinder_qpos_writes = 0
    initial_warnings = np.asarray(data.warning.number, dtype=int).copy()

    def render_frame() -> None:
        renderer.update_scene(data, camera=camera)
        writer.send(np.ascontiguousarray(renderer.render(), dtype=np.uint8))

    def current_inverse() -> mujoco.MjData:
        inverse = mujoco.MjData(model)
        inverse.qpos[:] = data.qpos
        inverse.qvel[:] = data.qvel
        inverse.time = data.time
        inverse.ctrl[:] = data.ctrl
        mujoco.mj_forward(model, inverse)
        inverse.qacc[:] = 0.0
        mujoco.mj_inverse(model, inverse)
        return inverse

    def apply_controller(right_arm_target: np.ndarray, right_root_target: float) -> list[dict[str, object]]:
        inverse = current_inverse()
        for joint_name, value in zip(arm_names, right_arm_target):
            motor_targets[joint_name] = float(value)
        clipping = []
        data.ctrl[:] = 0.0
        for actuator in motor_actuators:
            actuator_name = actuator_names[actuator]
            joint, dof = controller.actuator_joint(model, actuator)
            joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            gear = float(model.actuator_gear[actuator, 0])
            q = float(data.qpos[model.jnt_qposadr[joint]])
            qvel = float(data.qvel[dof])
            kp, kd = controller.motor_posture_gains(joint_name)
            requested = float(inverse.qfrc_inverse[dof] / gear + kp * (motor_targets[joint_name] - q) - kd * qvel)
            lo, hi = map(float, model.actuator_ctrlrange[actuator])
            if requested < lo - 1e-9 or requested > hi + 1e-9:
                clipping.append({"actuator": actuator_name, "joint": joint_name, "requested_ctrl": requested, "range": [lo, hi]})
            data.ctrl[actuator] = float(np.clip(requested, lo, hi))
            metrics["max_requested_motor_ctrl"][actuator_name] = max(abs(requested), metrics["max_requested_motor_ctrl"].get(actuator_name, 0.0))
        data.ctrl[root_actuator] = float(np.clip(right_root_target, *model.actuator_ctrlrange[root_actuator]))
        data.ctrl[left_actuator] = float(np.clip(left_root, *model.actuator_ctrlrange[left_actuator]))
        return clipping

    def evaluate(step: int, stage: str, clipping: list[dict[str, object]]) -> dict[str, object]:
        nonlocal first_failure, bilateral_streak, transfer_streak, settle_streak
        contacts = collect_contacts(model, data, pads)
        cylinder_state = object_state(model, data, cylinder)
        lift = float(cylinder_state["com_world_m"][2] - initial_com[2])
        metrics["max_cylinder_com_rise_m"] = max(metrics["max_cylinder_com_rise_m"], lift)
        joint_eq, connect_eq = all_equalities(model, data)
        max_joint_eq = max((abs(value) for value in joint_eq.values()), default=0.0)
        max_connect_eq = max((abs(value["residual_m"]) for value in connect_eq), default=0.0)
        metrics["max_joint_equality_rad"] = max(metrics["max_joint_equality_rad"], max_joint_eq)
        metrics["max_connect_equality_m"] = max(metrics["max_connect_equality_m"], max_connect_eq)
        right_joint = model.joint("right_claw_joint").id
        right_error = abs(float(data.qpos[model.jnt_qposadr[right_joint]]) - float(data.ctrl[root_actuator]))
        metrics["max_root_tracking_error_rad"] = max(metrics["max_root_tracking_error_rad"], right_error)
        table_force_z = float(contacts["table_force_on_cylinder_world_N"][2])
        pad_norms = {key: float(np.linalg.norm(value)) for key, value in contacts["pad_forces_on_cylinder_world_N"].items()}
        bilateral = set(contacts["pad_bodies"]) == pads
        bilateral_streak = bilateral_streak + 1 if bilateral else 0
        load_transferred = bilateral and table_force_z < MIN_LOAD_TRANSFER_RATIO * weight
        transfer_streak = transfer_streak + 1 if load_transferred else 0
        if bilateral and metrics["first_bilateral_step"] is None:
            metrics["first_bilateral_step"] = step
        if load_transferred and metrics["first_load_transfer_step"] is None:
            metrics["first_load_transfer_step"] = step
        stable_table_support = bool(
            contacts["table_contact_count"] > 0
            and SETTLE_SUPPORT_RATIO[0] * weight <= table_force_z <= SETTLE_SUPPORT_RATIO[1] * weight
            and cylinder_state["linear_speed_m_s"] <= SETTLE_LINEAR_SPEED_M_S
            and cylinder_state["angular_speed_rad_s"] <= SETTLE_ANGULAR_SPEED_RAD_S
        )
        if stage == "INITIAL_SETTLE":
            settle_streak = settle_streak + 1 if stable_table_support else 0
        if clipping:
            metrics["controller_clipping_events"] += len(clipping)
        for actuator, actuator_name in enumerate(actuator_names):
            metrics["max_actuator_force"][actuator_name] = max(abs(float(data.actuator_force[actuator])), metrics["max_actuator_force"].get(actuator_name, 0.0))
        if first_failure is None:
            if contacts["forbidden_indices"]:
                first_failure = {"gate": "UNEXPECTED_CONTACT", "step": step, "stage": stage, "contacts": [contacts["records"][index] for index in contacts["forbidden_indices"]]}
            elif clipping:
                first_failure = {"gate": "MOTOR_CONTROL_RANGE", "step": step, "stage": stage, "clipping": clipping}
            elif right_error > ROOT_TRACKING_GATE_RAD:
                first_failure = {"gate": "ROOT_TRACKING_ERROR", "step": step, "stage": stage, "error_rad": right_error, "limit_rad": ROOT_TRACKING_GATE_RAD}
            elif max_joint_eq > EQUALITY_JOINT_GATE_RAD or max_connect_eq > EQUALITY_CONNECT_GATE_M:
                first_failure = {"gate": "EQUALITY_RESIDUAL", "step": step, "stage": stage, "joint_rad": max_joint_eq, "connect_m": max_connect_eq}
            elif not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)) and np.all(np.isfinite(data.qacc)) and np.all(np.isfinite(data.ctrl))):
                first_failure = {"gate": "NONFINITE_STATE", "step": step, "stage": stage}
            elif np.any(np.asarray(data.warning.number, dtype=int) > initial_warnings):
                first_failure = {"gate": "MUJOCO_WARNING", "step": step, "stage": stage, "warnings": np.asarray(data.warning.number, dtype=int).tolist()}
            else:
                for joint in range(model.njnt):
                    if int(model.jnt_dofadr[joint]) in cylinder_dofs:
                        continue
                    joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    if bool(model.jnt_limited[joint]):
                        q = float(data.qpos[model.jnt_qposadr[joint]])
                        lo, hi = map(float, model.jnt_range[joint])
                        if outside_position_range(q, lo, hi):
                            first_failure = {"gate": "JOINT_POSITION_LIMIT", "step": step, "stage": stage, "joint": joint_name, "qpos": q, "range": [lo, hi]}
                            break
                    limit = source.get(joint_name, {}).get("velocity")
                    if limit is not None and abs(float(data.qvel[model.jnt_dofadr[joint]])) > float(limit):
                        first_failure = {"gate": "SOURCE_VELOCITY_LIMIT", "step": step, "stage": stage, "joint": joint_name, "qvel": float(data.qvel[model.jnt_dofadr[joint]]), "limit": limit}
                        break
        trace_writer.writerow([step, float(data.time), stage] + full_joint_row(model, data) + list(data.ctrl) + list(data.actuator_force))
        contact_trace.write(json.dumps({"step": step, "time_s": float(data.time), "stage": stage, "contacts": contacts, "joint_equality_rad": joint_eq, "connect_equality_m": connect_eq}, allow_nan=False) + "\n")
        summary_writer.writerow([step, float(data.time), stage, cylinder_state["com_world_m"][2], lift, cylinder_state["linear_speed_m_s"], cylinder_state["angular_speed_rad_s"], table_force_z, pad_norms.get("R_hand_narrow3_Link", 0.0), pad_norms.get("R_hand_wide3_Link", 0.0), right_error, max_joint_eq, max_connect_eq, len(contacts["forbidden_indices"])])
        if step % VIDEO_EVERY == 0:
            render_frame()
        return {"cylinder": cylinder_state, "lift_m": lift, "table_force_z": table_force_z, "bilateral": bilateral, "bilateral_streak": bilateral_streak, "transfer": load_transferred, "transfer_streak": transfer_streak, "settle_streak": settle_streak}

    def step(stage: str, arm_target: np.ndarray, root_target: float) -> dict[str, object] | None:
        nonlocal physics_steps
        if physics_steps >= MAX_STEPS or data.time >= MAX_TIME_S:
            return None
        clipping = apply_controller(arm_target, root_target)
        mujoco.mj_step(model, data)
        physics_steps += 1
        return evaluate(physics_steps, stage, clipping)

    def run_phase(label: str, count: int, arm_first: np.ndarray, arm_second: np.ndarray, root_first: float, root_second: float, require: str | None = None) -> bool:
        for index in range(1, count + 1):
            fraction = index / count
            record = step(label, interpolate(arm_first, arm_second, fraction), (1.0 - quintic(fraction)) * root_first + quintic(fraction) * root_second)
            if record is None or first_failure is not None:
                return False
            if require == "settle" and record["settle_streak"] >= SETTLE_REQUIRED_STABLE_STEPS:
                return True
        if require == "settle":
            return False
        if require == "bilateral":
            return record["bilateral_streak"] >= MIN_CONTACT_FRAMES
        if require == "transfer":
            return record["transfer_streak"] >= MIN_CONTACT_FRAMES
        return True

    # Set the validated existing-actuator controls before the zero-step audit.
    # The initial passive qpos is an approved static equilibrium, but the
    # frozen root position servo otherwise starts with ctrl=0 and would make a
    # false ROOT_TRACKING_ERROR before any physics integration occurs.
    initial_clipping = apply_controller(arm_initial, right_root)
    render_frame()
    initial_record = evaluate(0, "INITIAL", initial_clipping)
    if initial_record["table_force_z"] <= 0.0 or first_failure is not None:
        first_failure = first_failure or {"gate": "INITIAL_TABLE_SUPPORT", "step": 0}
    if first_failure is None and run_phase("INITIAL_SETTLE", SETTLE_MAX_STEPS, arm_initial, arm_initial, right_root, right_root, "settle"):
        stage_results["INITIAL_SETTLE"] = "PASS"
    elif first_failure is None:
        stage_results["INITIAL_SETTLE"] = "FAIL"
        first_failure = {"gate": "INITIAL_SETTLE_UNSTABLE", "step": physics_steps, "stage": "INITIAL_SETTLE"}
    else:
        stage_results["INITIAL_SETTLE"] = "FAIL"
    if first_failure is None:
        stage_results["HORIZONTAL_PREGRASP"] = "PASS" if run_phase("HORIZONTAL_PREGRASP", PREGRASP_HOLD_STEPS, arm_initial, arm_initial, right_root, right_root) else "FAIL"
    if first_failure is None:
        stage_results["APPROACH"] = "PASS" if run_phase("APPROACH", APPROACH_STEPS, arm_initial, arm_contact, right_root, right_root) else "FAIL"
    if first_failure is None:
        stage_results["CLOSE"] = "PASS" if run_phase("CLOSE", CLOSE_STEPS, arm_contact, arm_contact, right_root, float(config["right_root_close_target_rad"])) else "FAIL"
    if first_failure is None:
        stage_results["BILATERAL_DISTAL_PAD_CONTACT"] = "PASS" if run_phase("BILATERAL_DISTAL_PAD_CONTACT", CONTACT_HOLD_STEPS, arm_contact, arm_contact, float(config["right_root_close_target_rad"]), float(config["right_root_close_target_rad"]), "bilateral") else "FAIL"
        if first_failure is None and stage_results["BILATERAL_DISTAL_PAD_CONTACT"] != "PASS":
            first_failure = {"gate": "NO_SUSTAINED_BILATERAL_PAD_CONTACT", "step": physics_steps, "stage": "BILATERAL_DISTAL_PAD_CONTACT"}
    if first_failure is None:
        stage_results["LOAD_TRANSFER"] = "PASS" if run_phase("LOAD_TRANSFER", CONTACT_HOLD_STEPS, arm_contact, arm_contact, float(config["right_root_close_target_rad"]), float(config["right_root_close_target_rad"]), "transfer") else "FAIL"
        if first_failure is None and stage_results["LOAD_TRANSFER"] != "PASS":
            first_failure = {"gate": "NO_TABLE_TO_GRIPPER_LOAD_TRANSFER", "step": physics_steps, "stage": "LOAD_TRANSFER"}
    if first_failure is None:
        stage_results["LIFT_50MM"] = "PASS" if run_phase("LIFT_50MM", LIFT_STEPS, arm_contact, arm_lift, float(config["right_root_close_target_rad"]), float(config["right_root_close_target_rad"])) else "FAIL"
    if first_failure is None and metrics["max_cylinder_com_rise_m"] < 0.050:
        first_failure = {"gate": "LIFT_50MM_NOT_REACHED", "step": physics_steps, "max_lift_m": metrics["max_cylinder_com_rise_m"]}
    if first_failure is None:
        stage_results["HOLD_0_5S"] = "PASS" if run_phase("HOLD_0_5S", FINAL_HOLD_STEPS, arm_lift, arm_lift, float(config["right_root_close_target_rad"]), float(config["right_root_close_target_rad"])) else "FAIL"
    if first_failure is None:
        final_contacts = collect_contacts(model, data, pads)
        if metrics["max_cylinder_com_rise_m"] >= 0.050 and set(final_contacts["pad_bodies"]) == pads:
            result_status = "PASS"
        else:
            result_status = "FAIL"
            first_failure = {"gate": "FINAL_GRASP_HOLD", "step": physics_steps}
    else:
        result_status = "FAIL"
    render_frame()
    writer.close()
    renderer.close()
    trace.close()
    contact_trace.close()
    summary_trace.close()
    result = {
        "schema": "robotsim.issue46.full_x2_horizontal_cylinder_dynamics.v1",
        "status": result_status, "mj_step_calls": physics_steps, "simulated_time_s": float(data.time),
        "scene": str(scene), "scene_sha256": sha256(scene), "frozen_model_sha256": sha256(args.frozen_model),
        "controller_overlay": config["controller_overlay"], "static_preflight_config": str(args.config.resolve()), "static_preflight_config_sha256": sha256(args.config.resolve()),
        "stages": stage_results, "first_failure": first_failure, "metrics": metrics,
        "final_cylinder": object_state(model, data, cylinder),
        "initial_cylinder_com_world_m": initial_com.tolist(), "weight_N": weight,
        "safety": {"post_start_cylinder_qpos_writes": post_start_cylinder_qpos_writes, "weld_or_attachment": False, "mocap": False, "follower_qpos_writes": 0, "exact_pair_exception_only": True},
        "video": {"path": str(video), "sha256": sha256(video) if video.is_file() else None, "bytes": video.stat().st_size if video.is_file() else 0, "fps": VIDEO_FPS},
        "traces": ["joint_actuator_trace.csv.gz", "contact_force_trace.jsonl.gz", "phase_summary_trace.csv.gz"],
        "reproduction_command": f"MUJOCO_GL=egl python {Path(__file__).resolve()} --config {args.config.resolve()} --preflight-result {args.preflight_result.resolve()} --frozen-model {args.frozen_model.resolve()} --full-urdf {args.full_urdf.resolve()} --output-dir {output} --execute-dynamics",
    }
    pg.write_new_text(output / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": result_status, "steps": physics_steps, "first_failure": first_failure, "lift_mm": metrics["max_cylinder_com_rise_m"] * 1000.0, "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
