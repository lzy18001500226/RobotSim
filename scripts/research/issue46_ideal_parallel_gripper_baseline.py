#!/usr/bin/env python3
"""SIMULATION_ONLY ideal parallel-gripper contact baseline for Issue #46.

The independent ideal gripper and powered vertical carriage are diagnostic
fixtures. They are not X2 hardware models and apply no force to the bottle
except through MuJoCo contacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


DT = 0.002
GRAVITY = 9.81
TABLE_CENTER = (0.300, -0.100)
TABLE_TOP = 0.800
TABLE_HALF = 0.200
CYLINDER_CENTER = (0.300, -0.175, TABLE_TOP + 0.2445 / 2.0)
CYLINDER_RADIUS = 0.035
CYLINDER_HALF_HEIGHT = 0.12225
CYLINDER_MASS = 0.570
FRICTION = (1.4, 0.02, 0.001)

ROOT_BODY = "ideal_gripper_carriage"
CYLINDER_BODY = "canonical_test_cylinder"
CYLINDER_GEOM = "canonical_test_cylinder_geom"
LIFT_JOINT = "ideal_lift_slide"
LEFT_JOINT = "left_pad_slide"
RIGHT_JOINT = "right_pad_slide"
LEFT_PAD_BODY = "left_pad_body"
RIGHT_PAD_BODY = "right_pad_body"
PAD_HALF_X = 0.010
PAD_HALF_Y = 0.030
PAD_HALF_Z = 0.040
PAD_ANCHOR_X = 0.060
PAD_MAX_TRAVEL = 0.030
LIFT_START_Q = 0.010
OPEN_TARGET = 0.0
CLOSE_TARGET = 0.022
LIFT_TARGET_M = 0.050
PAD_FORCE_CAP_N = 8.0
LIFT_FORCE_CAP_N = 25.0
PAD_KP_N_M = 1000.0
PAD_KV_N_S_M = 20.0
LIFT_KP_N_M = 800.0
LIFT_KV_N_S_M = 60.0
PAD_MASS_KG = 0.060
PAD_DAMPING = 1.0
LIFT_DAMPING = 4.0

PHASES = (
    ("OPEN_SETTLE", 0.50),
    ("CLOSE", 0.70),
    ("BILATERAL_HOLD_ON_TABLE", 1.00),
    ("LIFT_50MM", 2.00),
    ("SUSPENDED_HOLD", 1.00),
)
TOTAL_STEPS = round(sum(duration for _, duration in PHASES) / DT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def xml_scene() -> str:
    # Table dimensions, placement, colors, and friction match the accepted G1 M0 table.
    return f'''<mujoco model="issue46_ideal_parallel_gripper_simulation_only">
  <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>
  <option timestep="{DT}" gravity="0 0 {-GRAVITY}" integrator="implicitfast" solver="Newton" iterations="80" tolerance="1e-10"/>
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65" specular="0.2 0.2 0.2"/>
    <rgba haze="0.25 0.35 0.45 1"/>
  </visual>
  <default>
    <geom condim="4" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <joint limited="true" armature="0.0001"/>
  </default>
  <worldbody>
    <light name="key_light" pos="0.2 -1.0 2.0" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="0 0 0.05" friction="0.9 0.01 0.001" rgba="0.55 0.60 0.64 1"/>
    <body name="m0_table" pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} 0">
      <geom name="m0_table_top" type="box" pos="0 0 0.775" size="0.2 0.2 0.025" rgba="0.6 0.4 0.2 1" friction="1.2 0.01 0.001" group="1"/>
      <geom name="m0_table_leg_front_left" type="box" pos="-0.175 0.175 0.375" size="0.025 0.025 0.375" rgba="0.6 0.4 0.2 1" friction="0.9 0.01 0.001" group="1"/>
      <geom name="m0_table_leg_front_right" type="box" pos="0.175 0.175 0.375" size="0.025 0.025 0.375" rgba="0.6 0.4 0.2 1" friction="0.9 0.01 0.001" group="1"/>
      <geom name="m0_table_leg_back_left" type="box" pos="-0.175 -0.175 0.375" size="0.025 0.025 0.375" rgba="0.6 0.4 0.2 1" friction="0.9 0.01 0.001" group="1"/>
      <geom name="m0_table_leg_back_right" type="box" pos="0.175 -0.175 0.375" size="0.025 0.025 0.375" rgba="0.6 0.4 0.2 1" friction="0.9 0.01 0.001" group="1"/>
    </body>
    <body name="{CYLINDER_BODY}" pos="{' '.join(str(x) for x in CYLINDER_CENTER)}">
      <freejoint name="canonical_test_cylinder_free"/>
      <geom name="{CYLINDER_GEOM}" type="cylinder" size="{CYLINDER_RADIUS} {CYLINDER_HALF_HEIGHT}" mass="{CYLINDER_MASS}" friction="{' '.join(str(x) for x in FRICTION)}" rgba="0.08 0.48 0.82 1"/>
    </body>
    <body name="{ROOT_BODY}" pos="{CYLINDER_CENTER[0]} {CYLINDER_CENTER[1]} {CYLINDER_CENTER[2]}">
      <joint name="{LIFT_JOINT}" type="slide" axis="0 0 1" range="0 0.08" ref="{LIFT_START_Q}" damping="{LIFT_DAMPING}"/>
      <inertial pos="0 0 0" mass="0.20" diaginertia="0.0005 0.0005 0.0005"/>
      <geom name="carriage_visual" type="box" pos="0 0 0.054" size="0.022 0.045 0.010" mass="0" contype="0" conaffinity="0" rgba="0.32 0.34 0.36 1"/>
      <body name="{LEFT_PAD_BODY}" pos="{-PAD_ANCHOR_X} 0 0">
        <joint name="{LEFT_JOINT}" type="slide" axis="1 0 0" range="0 {PAD_MAX_TRAVEL}" damping="{PAD_DAMPING}"/>
        <geom name="left_contact_pad" type="box" size="{PAD_HALF_X} {PAD_HALF_Y} {PAD_HALF_Z}" mass="{PAD_MASS_KG}" friction="{' '.join(str(x) for x in FRICTION)}" rgba="0.12 0.14 0.16 1"/>
      </body>
      <body name="{RIGHT_PAD_BODY}" pos="{PAD_ANCHOR_X} 0 0">
        <joint name="{RIGHT_JOINT}" type="slide" axis="-1 0 0" range="0 {PAD_MAX_TRAVEL}" damping="{PAD_DAMPING}"/>
        <geom name="right_contact_pad" type="box" size="{PAD_HALF_X} {PAD_HALF_Y} {PAD_HALF_Z}" mass="{PAD_MASS_KG}" friction="{' '.join(str(x) for x in FRICTION)}" rgba="0.12 0.14 0.16 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="left_pad_bounded_servo" joint="{LEFT_JOINT}" kp="{PAD_KP_N_M}" kv="{PAD_KV_N_S_M}" ctrlrange="0 {PAD_MAX_TRAVEL}" forcelimited="true" forcerange="{-PAD_FORCE_CAP_N} {PAD_FORCE_CAP_N}"/>
    <position name="right_pad_bounded_servo" joint="{RIGHT_JOINT}" kp="{PAD_KP_N_M}" kv="{PAD_KV_N_S_M}" ctrlrange="0 {PAD_MAX_TRAVEL}" forcelimited="true" forcerange="{-PAD_FORCE_CAP_N} {PAD_FORCE_CAP_N}"/>
    <position name="bounded_vertical_carriage_servo" joint="{LIFT_JOINT}" kp="{LIFT_KP_N_M}" kv="{LIFT_KV_N_S_M}" ctrlrange="0 0.08" forcelimited="true" forcerange="{-LIFT_FORCE_CAP_N} {LIFT_FORCE_CAP_N}"/>
  </actuator>
</mujoco>
'''


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"Missing body {name}")
    return int(result)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise RuntimeError(f"Missing joint {name}")
    return int(result)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if result < 0:
        raise RuntimeError(f"Missing actuator {name}")
    return int(result)


def qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    return float(data.qpos[int(model.jnt_qposadr[joint_id(model, name)])])


def qvel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    return float(data.qvel[int(model.jnt_dofadr[joint_id(model, name)])])


def body_state(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, Any]:
    bid = body_id(model, name)
    linear = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, linear, 0)
    return {
        "position_world_m": data.xpos[bid].tolist(),
        "quaternion_wxyz": data.xquat[bid].tolist(),
        "linear_velocity_world_m_s": linear[3:].tolist(),
        "angular_velocity_world_rad_s": linear[:3].tolist(),
    }


def upright_error_deg(quaternion_wxyz: list[float]) -> float:
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion_wxyz, dtype=np.float64))
    return math.degrees(math.acos(float(np.clip(rotation.reshape(3, 3)[2, 2], -1.0, 1.0))))


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CYLINDER_GEOM)
    bottle_body = body_id(model, CYLINDER_BODY)
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if bottle_geom not in (g1, g2):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, local)
        frame = np.asarray(contact.frame).reshape(3, 3)
        world_on_geom2 = frame.T @ local[:3]
        if int(model.geom_bodyid[g2]) == bottle_body:
            force_on_bottle = world_on_geom2
        else:
            force_on_bottle = -world_on_geom2
        other_geom = g1 if g2 == bottle_geom else g2
        other_body_id = int(model.geom_bodyid[other_geom])
        other_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id) or "world"
        other_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or "world_geom"
        if other_body == "m0_table":
            support_class = "table"
        elif other_body in (LEFT_PAD_BODY, RIGHT_PAD_BODY):
            support_class = "hand_pad"
        else:
            support_class = "other"
        result.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])) or "world",
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])) or "world",
            "other_body": other_body,
            "other_geom": other_geom_name,
            "support_class": support_class,
            "position_world_m": np.asarray(contact.pos).tolist(),
            "normal_world": frame[0].tolist(),
            "force_on_bottle_world_N": force_on_bottle.tolist(),
            "normal_force_N": float(max(0.0, local[0])),
            "tangential_force_N": float(np.linalg.norm(local[1:3])),
            "friction_coefficients": np.asarray(contact.friction).tolist(),
            "distance_m": float(contact.dist),
            "penetration_m": float(max(0.0, -contact.dist)),
            "contact_force_local_N_Nm": local.tolist(),
        })
    return result


def signed_vertical_force_summary(contacts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Sum signed bottle-side world-Z contact forces without hiding cancellation."""
    summaries: dict[str, dict[str, float]] = {}
    for support_class in ("table", "hand_pad"):
        forces = [
            float(row["force_on_bottle_world_N"][2])
            for row in contacts
            if row["support_class"] == support_class
        ]
        summaries[support_class] = {
            "net_N": math.fsum(forces),
            "upward_sum_N": math.fsum(force for force in forces if force > 0.0),
            "downward_sum_N": math.fsum(force for force in forces if force < 0.0),
        }
    return summaries


def phase_reference(phase: str, phase_t: float, duration: float) -> tuple[float, float]:
    if phase in ("OPEN_SETTLE",):
        return OPEN_TARGET, 0.0
    if phase == "CLOSE":
        s = min(1.0, max(0.0, phase_t / duration))
        smooth = s * s * (3.0 - 2.0 * s)
        return OPEN_TARGET + smooth * (CLOSE_TARGET - OPEN_TARGET), 0.0
    return CLOSE_TARGET, 0.0


def lift_reference(phase: str, phase_t: float, duration: float) -> tuple[float, float]:
    if phase != "LIFT_50MM":
        return LIFT_START_Q + (LIFT_TARGET_M if phase == "SUSPENDED_HOLD" else 0.0), 0.0
    s = min(1.0, max(0.0, phase_t / duration))
    smooth = s * s * (3.0 - 2.0 * s)
    return LIFT_START_Q + LIFT_TARGET_M * smooth, LIFT_TARGET_M * (6.0 * s * (1.0 - s) / duration)


def make_renderer(model: mujoco.MjModel) -> mujoco.Renderer:
    renderer = mujoco.Renderer(model, height=480, width=640)
    return renderer


def render(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
           view: str) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    bottle = np.asarray(CYLINDER_CENTER)
    if view == "side":
        camera.azimuth, camera.elevation, camera.distance = 90.0, -8.0, 0.66
    elif view == "front":
        camera.azimuth, camera.elevation, camera.distance = 180.0, -8.0, 0.66
    elif view == "close":
        camera.azimuth, camera.elevation, camera.distance = 165.0, -9.0, 0.40
    else:
        camera.azimuth, camera.elevation, camera.distance = 145.0, -18.0, 0.95
    camera.lookat[:] = [bottle[0], bottle[1], 0.88 + max(0.0, qpos(model, data, LIFT_JOINT)) / 2.0]
    renderer.update_scene(data, camera)
    return renderer.render().copy()


def native_library_identity() -> list[dict[str, Any]]:
    package_dir = Path(mujoco.__file__).resolve().parent
    candidates = list(package_dir.glob("libmujoco.so*")) + list(package_dir.glob("**/libmujoco.so*"))
    found = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        found.append({"path": str(resolved), "sha256": sha256(resolved)})
    return found


def run(output: Path, script_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[2]
    model_xml = xml_scene()
    (output / "ideal_parallel_gripper.xml").write_text(model_xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if model.neq != 0 or model.nmocap != 0:
        raise RuntimeError("Unexpected equality or mocap object in diagnostic model")

    renderer = make_renderer(model)
    named_views = {"overview": "overview.png", "front": "front.png", "side": "side.png", "close": "gripper_closeup.png"}
    for view, filename in named_views.items():
        Image.fromarray(render(renderer, model, data, view)).save(output / filename)
    frames: list[np.ndarray] = []
    trace_path = output / "physics_trace.jsonl"
    phase_summaries: dict[str, Any] = {}
    stage_start = 0.0
    total_step = 0
    all_finite = True
    max_penetration = 0.0
    max_joint_speed = {LEFT_JOINT: 0.0, RIGHT_JOINT: 0.0, LIFT_JOINT: 0.0}
    max_pad_tracking_error = {LEFT_JOINT: 0.0, RIGHT_JOINT: 0.0}
    max_actuator_force = {"left": 0.0, "right": 0.0, "lift": 0.0}
    sat_steps = {"left": 0, "right": 0, "lift": 0}
    limit_violations: list[dict[str, Any]] = []
    unexpected_contacts: set[str] = set()
    table_support_peak = 0.0
    hand_support_peak = 0.0
    max_lift_m = 0.0
    suspended_samples = 0
    suspended_bilateral_samples = 0
    suspended_table_contact_samples = 0
    max_penetration_bottle_pad = 0.0
    relative_pose_at_lift_start: np.ndarray | None = None
    max_relative_slip = 0.0
    with trace_path.open("w", encoding="utf-8") as trace_stream:
        for phase, duration in PHASES:
            phase_start = float(data.time)
            nsteps = round(duration / DT)
            contact_samples = 0
            bilateral_samples = 0
            phase_table_sum = 0.0
            phase_hand_sum = 0.0
            phase_table_upward_sum = 0.0
            phase_table_downward_sum = 0.0
            phase_hand_upward_sum = 0.0
            phase_hand_downward_sum = 0.0
            phase_bottle_z_min = math.inf
            phase_bottle_z_max = -math.inf
            phase_joint_speed_max = 0.0
            phase_penetration_max = 0.0
            for step_in_phase in range(nsteps):
                local_t = step_in_phase * DT
                contact_force_evaluation_time = float(data.time)
                pad_target, _ = phase_reference(phase, local_t, duration)
                lift_target, lift_target_velocity = lift_reference(phase, local_t, duration)
                data.ctrl[actuator_id(model, "left_pad_bounded_servo")] = pad_target
                data.ctrl[actuator_id(model, "right_pad_bounded_servo")] = pad_target
                data.ctrl[actuator_id(model, "bounded_vertical_carriage_servo")] = lift_target
                mujoco.mj_step(model, data)
                total_step += 1

                bottle = body_state(model, data, CYLINDER_BODY)
                carriage = body_state(model, data, ROOT_BODY)
                bottom_z = float(bottle["position_world_m"][2] - CYLINDER_HALF_HEIGHT)
                lift_height = float(bottle["position_world_m"][2] - CYLINDER_CENTER[2])
                max_lift_m = max(max_lift_m, lift_height)
                phase_bottle_z_min = min(phase_bottle_z_min, float(bottle["position_world_m"][2]))
                phase_bottle_z_max = max(phase_bottle_z_max, float(bottle["position_world_m"][2]))
                contacts = contact_rows(model, data)
                vertical_forces = signed_vertical_force_summary(contacts)
                pad_normals = {LEFT_PAD_BODY: 0.0, RIGHT_PAD_BODY: 0.0}
                table_vertical = vertical_forces["table"]["net_N"]
                hand_vertical = vertical_forces["hand_pad"]["net_N"]
                for row in contacts:
                    if row["support_class"] == "hand_pad":
                        pad_normals[row["other_body"]] += row["normal_force_N"]
                    elif row["support_class"] != "table":
                        unexpected_contacts.add(f"{row['other_body']}|{row['other_geom']}")
                    max_penetration = max(max_penetration, row["penetration_m"])
                    phase_penetration_max = max(phase_penetration_max, row["penetration_m"])
                    if row["support_class"] == "hand_pad":
                        max_penetration_bottle_pad = max(max_penetration_bottle_pad, row["penetration_m"])
                if contacts:
                    contact_samples += 1
                bilateral = pad_normals[LEFT_PAD_BODY] > 1e-5 and pad_normals[RIGHT_PAD_BODY] > 1e-5
                if bilateral:
                    bilateral_samples += 1
                table_support_peak = max(table_support_peak, table_vertical)
                hand_support_peak = max(hand_support_peak, hand_vertical)
                phase_table_sum += table_vertical
                phase_hand_sum += hand_vertical
                phase_table_upward_sum += vertical_forces["table"]["upward_sum_N"]
                phase_table_downward_sum += vertical_forces["table"]["downward_sum_N"]
                phase_hand_upward_sum += vertical_forces["hand_pad"]["upward_sum_N"]
                phase_hand_downward_sum += vertical_forces["hand_pad"]["downward_sum_N"]

                actuator_info: dict[str, Any] = {}
                for key, joint, actuator, ref, cap in (
                    ("left", LEFT_JOINT, "left_pad_bounded_servo", pad_target, PAD_FORCE_CAP_N),
                    ("right", RIGHT_JOINT, "right_pad_bounded_servo", pad_target, PAD_FORCE_CAP_N),
                    ("lift", LIFT_JOINT, "bounded_vertical_carriage_servo", lift_target, LIFT_FORCE_CAP_N),
                ):
                    aid = actuator_id(model, actuator)
                    actual = float(qpos(model, data, joint))
                    velocity = float(qvel(model, data, joint))
                    force = float(data.actuator_force[aid])
                    max_actuator_force[key] = max(max_actuator_force[key], abs(force))
                    if abs(force) >= cap * 0.999:
                        sat_steps[key] += 1
                    max_joint_speed[joint] = max(max_joint_speed[joint], abs(velocity))
                    phase_joint_speed_max = max(phase_joint_speed_max, abs(velocity))
                    if joint != LIFT_JOINT:
                        max_pad_tracking_error[joint] = max(max_pad_tracking_error[joint], abs(actual - ref))
                    lo, hi = model.jnt_range[joint_id(model, joint)]
                    margin = min(actual - float(lo), float(hi) - actual)
                    if margin < -1e-9:
                        limit_violations.append({"step": total_step, "time_s": data.time, "joint": joint,
                                                 "qpos": actual, "range": [float(lo), float(hi)], "margin": margin})
                    actuator_info[key] = {"joint": joint, "qpos": actual, "qvel": velocity,
                                          "target_q": ref, "tracking_error": actual - ref,
                                          "actuator_force_N": force, "force_cap_N": cap,
                                          "saturated": abs(force) >= cap * 0.999,
                                          "limit_margin": margin}

                if phase == "LIFT_50MM" and relative_pose_at_lift_start is None:
                    relative_pose_at_lift_start = np.asarray(bottle["position_world_m"]) - np.asarray(carriage["position_world_m"])
                if phase in ("LIFT_50MM", "SUSPENDED_HOLD") and relative_pose_at_lift_start is not None:
                    relative = np.asarray(bottle["position_world_m"]) - np.asarray(carriage["position_world_m"])
                    max_relative_slip = max(max_relative_slip, float(np.linalg.norm(relative - relative_pose_at_lift_start)))
                if phase == "SUSPENDED_HOLD":
                    suspended_samples += 1
                    suspended_bilateral_samples += int(bilateral)
                    suspended_table_contact_samples += int(any(row["support_class"] == "table" for row in contacts))

                finite = bool(np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc, data.actuator_force))).all())
                all_finite = all_finite and finite
                record = {
                    "step": total_step, "time_s": float(data.time), "phase": phase,
                    "contact_force_evaluation_time_s": contact_force_evaluation_time,
                    "phase_time_s": local_t,
                    "command": {"pad_target_q_m": pad_target, "lift_target_q_m": lift_target,
                                "lift_target_velocity_m_s": lift_target_velocity},
                    "joints_actuators": actuator_info,
                    "bottle": {"position_world_m": bottle["position_world_m"],
                               "quaternion_wxyz": bottle["quaternion_wxyz"],
                               "linear_velocity_world_m_s": bottle["linear_velocity_world_m_s"],
                               "angular_velocity_world_rad_s": bottle["angular_velocity_world_rad_s"],
                               "bottom_separation_from_table_m": bottom_z - TABLE_TOP,
                               "height_change_from_reset_m": lift_height,
                               "upright_error_deg": upright_error_deg(bottle["quaternion_wxyz"])},
                    "carriage": {"position_world_m": carriage["position_world_m"],
                                 "q_m": qpos(model, data, LIFT_JOINT),
                                 "qvel_m_s": qvel(model, data, LIFT_JOINT)},
                    "contact_forces_on_bottle": contacts,
                    "measured_contact_force_world_z_N": {
                        "table": vertical_forces["table"],
                        "hand_pads": vertical_forces["hand_pad"],
                    },
                    "pad_normal_force_N": pad_normals,
                    "bilateral_contact": bilateral,
                    "finite_state": finite,
                    "bottle_qpos_write_count": 0,
                    "equalities": 0,
                    "mocap_carry": False,
                }
                trace_stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                if total_step % 8 == 0:
                    frames.append(render(renderer, model, data, "overview"))
            phase_summaries[phase] = {
                "start_time_s": phase_start, "end_time_s": float(data.time), "steps": nsteps,
                "contact_fraction": contact_samples / nsteps,
                "bilateral_contact_fraction": bilateral_samples / nsteps,
                "mean_table_net_force_world_z_N": phase_table_sum / nsteps,
                "mean_hand_net_force_world_z_N": phase_hand_sum / nsteps,
                "mean_table_upward_force_world_z_N": phase_table_upward_sum / nsteps,
                "mean_table_downward_force_world_z_N": phase_table_downward_sum / nsteps,
                "mean_hand_upward_force_world_z_N": phase_hand_upward_sum / nsteps,
                "mean_hand_downward_force_world_z_N": phase_hand_downward_sum / nsteps,
                "bottle_center_z_min_m": phase_bottle_z_min,
                "bottle_center_z_max_m": phase_bottle_z_max,
                "maximum_joint_speed": phase_joint_speed_max,
                "maximum_bottle_contact_penetration_m": phase_penetration_max,
            }
    for view, filename in named_views.items():
        Image.fromarray(render(renderer, model, data, view)).save(output / filename.replace(".png", "_final.png"))
    imageio.mimsave(output / "ideal_gripper_lift.mp4", frames, fps=25, macro_block_size=1)
    renderer.close()

    hold = phase_summaries["SUSPENDED_HOLD"]
    final_bottom_separation = float(data.xpos[body_id(model, CYLINDER_BODY)][2] - CYLINDER_HALF_HEIGHT - TABLE_TOP)
    lift_ok = max_lift_m >= LIFT_TARGET_M - 0.005
    suspended_ok = (hold["end_time_s"] - hold["start_time_s"] >= 1.0 - 1e-9
                    and hold["bilateral_contact_fraction"] >= 0.95
                    and hold["mean_table_net_force_world_z_N"] <= 0.05
                    and final_bottom_separation >= 0.045)
    passed = (lift_ok and suspended_ok and all_finite and not limit_violations
              and not unexpected_contacts and phase_summaries["BILATERAL_HOLD_ON_TABLE"]["bilateral_contact_fraction"] >= 0.95)

    identity = {
        "robot_sim_head": subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                          check=False, capture_output=True, text=True).stdout.strip(),
        "branch": subprocess.run(["git", "-C", str(source_root), "branch", "--show-current"],
                                 check=False, capture_output=True, text=True).stdout.strip(),
        "script_sha256": sha256(script_path),
        "model_xml_sha256": sha256(output / "ideal_parallel_gripper.xml"),
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_python_module": str(Path(mujoco.__file__).resolve()),
        "native_libraries": native_library_identity(),
        "timestep_s": DT,
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "solver_iterations": int(model.opt.iterations),
        "gravity_m_s2": model.opt.gravity.tolist(),
    }
    result = {
        "status": "PASS" if passed else "FAIL",
        "classification": "SIMULATION_ONLY_IDEAL_PARALLEL_GRIPPER",
        "claim_scope": "Independent contact-and-friction capacity baseline only; not X2 hardware or OmniPicker acceptance.",
        "scene": {"table": {"source": "accepted G1 M0 canonical table dimensions copied exactly from canonical_manipulation_assets.py",
                             "center_xy_m": list(TABLE_CENTER), "top_z_m": TABLE_TOP,
                             "half_extents_m": [TABLE_HALF, TABLE_HALF], "leg_count": 4},
                  "cylinder": {"diameter_m": 2 * CYLINDER_RADIUS, "height_m": 2 * CYLINDER_HALF_HEIGHT,
                               "mass_kg": CYLINDER_MASS, "friction": list(FRICTION),
                               "freejoint": True, "initial_center_world_m": list(CYLINDER_CENTER)},
                  "gravity_m_s2": GRAVITY},
        "gripper": {"pad_geometry": "opposed box pads; no robot/vendor geometry", "pad_half_extents_m": [PAD_HALF_X, PAD_HALF_Y, PAD_HALF_Z],
                    "pad_mass_each_kg": PAD_MASS_KG, "open_inner_gap_m": 2 * (PAD_ANCHOR_X - PAD_HALF_X),
                    "open_clearance_each_side_m": (2 * (PAD_ANCHOR_X - PAD_HALF_X) - 2 * CYLINDER_RADIUS) / 2,
                    "target_closure_travel_m": CLOSE_TARGET, "target_closed_inner_gap_m": 2 * (PAD_ANCHOR_X - CLOSE_TARGET - PAD_HALF_X),
                    "pad_kp_N_m": PAD_KP_N_M, "pad_kv_N_s_m": PAD_KV_N_S_M,
                    "pad_effort_cap_N": PAD_FORCE_CAP_N,
                    "lift_operational_start_q_m": LIFT_START_Q, "lift_target_displacement_m": LIFT_TARGET_M,
                    "lift_effort_cap_N": LIFT_FORCE_CAP_N,
                    "lift_kp_N_m": LIFT_KP_N_M, "lift_kv_N_s_m": LIFT_KV_N_S_M,
                    "classification": "SIMULATION_ONLY idealized parallel gripper and powered carriage"},
        "phases": phase_summaries,
        "metrics": {"maximum_bottle_lift_m": max_lift_m,
                    "final_bottle_bottom_separation_from_table_m": final_bottom_separation,
                    "suspended_hold_duration_s": hold["end_time_s"] - hold["start_time_s"],
                    "suspended_hold_bilateral_contact_fraction": hold["bilateral_contact_fraction"],
                    "suspended_hold_mean_hand_net_force_world_z_N": hold["mean_hand_net_force_world_z_N"],
                    "suspended_hold_mean_table_net_force_world_z_N": hold["mean_table_net_force_world_z_N"],
                    "suspended_hold_mean_hand_upward_force_world_z_N": hold["mean_hand_upward_force_world_z_N"],
                    "suspended_hold_mean_hand_downward_force_world_z_N": hold["mean_hand_downward_force_world_z_N"],
                    "suspended_hold_table_contact_samples": suspended_table_contact_samples,
                    "required_static_weight_N": CYLINDER_MASS * GRAVITY,
                    "maximum_hand_net_force_world_z_N": hand_support_peak,
                    "maximum_table_net_force_world_z_N": table_support_peak,
                    "maximum_relative_slip_bottle_to_carriage_m": max_relative_slip,
                    "maximum_bottle_contact_penetration_m": max_penetration,
                    "maximum_bottle_pad_penetration_m": max_penetration_bottle_pad,
                    "maximum_pad_tracking_error_m": max_pad_tracking_error,
                    "maximum_joint_speed": max_joint_speed,
                    "maximum_actuator_force_N": max_actuator_force,
                    "actuator_saturated_steps": sat_steps,
                    "source_limit_violations": limit_violations,
                    "all_finite": all_finite,
                    "unexpected_bottle_contact_pairs": sorted(unexpected_contacts),
                    "object_state_writes": 0,
                    "welds_or_equalities": [],
                    "mocap_bodies": 0,
                    "hidden_support": False},
        "runtime_identity": identity,
        "reproduction_command": "PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 MUJOCO_GL=egl /usr/bin/python3 scripts/research/issue46_ideal_parallel_gripper_baseline.py --output <new-empty-output-dir>",
        "evidence": {"trace": "physics_trace.jsonl", "video": "ideal_gripper_lift.mp4",
                     "model": "ideal_parallel_gripper.xml", "images": list(named_views.values())},
    }
    write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    started = time.time()
    result = run(args.output.resolve(), script_path)
    result["wall_runtime_s"] = time.time() - started
    write_json(args.output.resolve() / "result.json", result)
    print(json.dumps({"status": result["status"], "classification": result["classification"],
                      "metrics": result["metrics"], "output": str(args.output.resolve())}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
