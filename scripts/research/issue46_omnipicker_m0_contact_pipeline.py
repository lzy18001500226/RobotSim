#!/usr/bin/env python3
"""Bounded SIMULATION_ONLY_M0 contact-pipeline check on an isolated X2 OmniPicker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


X2_REPO = "https://github.com/AgibotTech/agibot_x2_urdf"
X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_ROOT = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0")
X2_URDF = X2_ROOT / "X2-Ultra_omnipicker.urdf"
X2_MESHES = X2_ROOT / "meshes"
X2_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
M0_BASELINE_BRANCH = "codex/issue46-omnipicker-1dof-m0-20261007"
M0_BASELINE_COMMIT = "fea64b09c8b159c4597fdc58de197b801136e144"
M0_BASELINE_SCRIPT_SHA256 = "9ed3523b397d8cc009d437e445aeea8e027451dfd4846ead06cbb2a43629b28f"
M0_BASELINE_REPORT_SHA256 = "d9864ad27207161eacfa611856b67bc1c5bc53074b27a0efd829ce3d1a883bfa"

RIGHT_DRIVER = "right_claw_joint"
RIGHT_FOLLOWER = "R_hand_wide1_joint"
NARROW_PAD = "R_hand_narrow3_Link"
WIDE_PAD = "R_hand_wide3_Link"
OBJECT_BODY = "diag_object"
OBJECT_GEOM = "diag_cylinder"
TABLE_GEOM = "diag_table"
LIFT_JOINT = "m0_lift_slide"

TIMESTEP_S = 0.002
GRAVITY_M_S2 = 9.81
OPEN_Q_DRIVER_RAD = -0.999
OPEN_Q_FOLLOWER_RAD = 0.999
CONTACT_APERTURE_RATIO = 0.052  # Geometry-derived from run04's 0.474 mm narrow-pad gap at 0.060.
CONTACT_PRELOAD_GAP_M = 0.0004
ENDPOINT_MARGIN_RAD = 0.001
MOTION_DURATION_S = 2.5
M0_KP_NM_PER_RAD = {RIGHT_DRIVER: 0.005311594119130363, RIGHT_FOLLOWER: 0.010911462396815307}
M0_KD_NMS_PER_RAD = {RIGHT_DRIVER: 0.0013278985297825907, RIGHT_FOLLOWER: 0.0027278655992038266}
M0_EFFORT_CAP_NM = {RIGHT_DRIVER: 0.2380705710105672, RIGHT_FOLLOWER: 0.2538932928038361}
M0_MAX_JAW_SPEED_RAD_S = 1.173
M0_MAX_JAW_ACCEL_RAD_S2 = 151.325
M0_MAX_PENETRATION_M = 0.006
M0_PRECONTACT_RELATION_DIAGNOSTIC_RAD = 0.020
FORCE_BEARING_EPS_N = 1e-6

# Geometry-only insertion from the compiled OPEN pad witness. This is an isolated
# fixed-rig diagnostic, not an X2 arm target or a production grasp pose.
# One geometry-derived contact-pipeline candidate: raise the fixed fixture 25 mm,
# keep the 150 mm diagnostic rod standing on the same table, and grasp its upper
# section. A 50 mm raise would put the open-pad witness above the rod tip.
FIXTURE_RAISE_M = 0.025
OBJECT_RELATIVE_TO_BASE_WORLD_M = np.array([-0.0001996246106712, 0.0127422484175874, -0.150], dtype=float)
OBJECT_RADIUS_M = 0.005
OBJECT_HALF_HEIGHT_M = 0.075
OBJECT_DENSITY_ASSUMPTION_KG_M3 = 7800.0
OBJECT_MASS_KG = math.pi * OBJECT_RADIUS_M**2 * (2.0 * OBJECT_HALF_HEIGHT_M) * OBJECT_DENSITY_ASSUMPTION_KG_M3
OBJECT_FRICTION = [1.4, 0.02, 0.001]
TABLE_HALF_SIZE_M = [0.006, 0.006, 0.005]
TABLE_TOP_Z_M = 0.0
CARriage_WORLD_POS_M = [0.0, 0.0, 0.287]
CARriage_WORLD_YAW_RAD = -math.pi / 2.0
CARriage_MASS_KG = 0.5
CARriage_DIAG_INERTIA_KG_M2 = [0.001, 0.001, 0.001]
CARriage_FORCE_CAP_N = 30.0
CARriage_KP_N_M = 10000.0
CARriage_KV_NS_M = 200.0
MAX_SHORT_LIFT_M = 0.010
FIXTURE_LIMIT_NUMERICAL_TOLERANCE_M = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl_line(stream, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def body_parent(root: ET.Element, name: str) -> tuple[ET.Element, ET.Element] | None:
    world = root.find("worldbody")
    if world is None:
        return None

    def walk(parent: ET.Element):
        for body in parent.findall("body"):
            if body.get("name") == name:
                return parent, body
            found = walk(body)
            if found is not None:
                return found
        return None

    return walk(world)


def m0_targets(aperture: float, aperture_velocity: float = 0.0,
               aperture_acceleration: float = 0.0) -> dict[str, float]:
    span = 1.0 - 2.0 * ENDPOINT_MARGIN_RAD
    driver = -(ENDPOINT_MARGIN_RAD + span * aperture)
    driver_v = -span * aperture_velocity
    driver_a = -span * aperture_acceleration
    return {
        "driver_q": driver,
        "driver_v": driver_v,
        "driver_a": driver_a,
        "follower_q": -driver,
        "follower_v": -driver_v,
        "follower_a": -driver_a,
    }


def minimum_jerk(time_s: float, duration_s: float, start: float, end: float) -> tuple[float, float, float]:
    if time_s >= duration_s:
        return end, 0.0, 0.0
    u = max(0.0, time_s / duration_s)
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration_s
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration_s**2
    delta = end - start
    return start + delta * s, delta * ds, delta * dds


def quaternion_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)


def quat_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def compose_quaternions(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(result, first, second)
    return result


def build_xml(include_object: bool) -> tuple[str, dict[str, Any]]:
    if not X2_URDF.is_file() or not X2_MESHES.is_dir():
        raise FileNotFoundError("Pinned AgiBot X2 URDF or mesh directory is unavailable")
    if sha256(X2_URDF) != X2_URDF_SHA256:
        raise RuntimeError("Pinned X2 URDF SHA-256 mismatch")

    source_spec = mujoco.MjSpec.from_file(str(X2_URDF))
    source_spec.compiler.fusestatic = False
    root = ET.fromstring(source_spec.to_xml())
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(X2_MESHES))

    found = body_parent(root, "R_omnipicker_base_link")
    if found is None:
        raise RuntimeError("Pinned URDF import did not contain R_omnipicker_base_link")
    old_parent, base = found
    base_mount_pos = np.fromstring(base.get("pos", "0 0 0"), sep=" ")
    base_mount_quat = np.fromstring(base.get("quat", "1 0 0 0"), sep=" ")
    if base_mount_pos.shape != (3,) or base_mount_quat.shape != (4,):
        raise RuntimeError("Unexpected compiled URDF mount transform format")
    old_parent.remove(base)

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Generated X2 MJCF has no worldbody")
    for child in list(world):
        if child.tag == "body":
            world.remove(child)

    yaw = quaternion_yaw(CARriage_WORLD_YAW_RAD)
    carriage = ET.Element("body", {
        "name": "m0_lift_carriage",
        "pos": " ".join(f"{v:.12g}" for v in CARriage_WORLD_POS_M),
        "quat": " ".join(f"{v:.12g}" for v in yaw),
    })
    ET.SubElement(carriage, "joint", {
        "name": LIFT_JOINT,
        "type": "slide",
        "axis": "0 0 1",
        "limited": "true",
        "range": f"0 {MAX_SHORT_LIFT_M:.12g}",
    })
    ET.SubElement(carriage, "inertial", {
        "pos": "0 0 0",
        "mass": f"{CARriage_MASS_KG:.12g}",
        "diaginertia": " ".join(f"{v:.12g}" for v in CARriage_DIAG_INERTIA_KG_M2),
    })
    carriage.append(base)
    world.append(carriage)

    base_world_quat = compose_quaternions(yaw, base_mount_quat)
    base_world_pos = np.array(CARriage_WORLD_POS_M, dtype=float) + quat_matrix(yaw) @ base_mount_pos
    object_world_pos = base_world_pos + OBJECT_RELATIVE_TO_BASE_WORLD_M
    table_center = [float(object_world_pos[0]), float(object_world_pos[1]),
                    TABLE_TOP_Z_M - TABLE_HALF_SIZE_M[2]]

    if include_object:
        ET.SubElement(world, "geom", {
            "name": TABLE_GEOM,
            "type": "box",
            "pos": " ".join(f"{v:.12g}" for v in table_center),
            "size": " ".join(f"{v:.12g}" for v in TABLE_HALF_SIZE_M),
            "friction": " ".join(f"{v:.12g}" for v in OBJECT_FRICTION),
            "condim": "4",
            "rgba": "0.38 0.27 0.17 1",
        })
        object_body = ET.SubElement(world, "body", {
            "name": OBJECT_BODY,
            "pos": " ".join(f"{v:.12g}" for v in object_world_pos),
        })
        ET.SubElement(object_body, "freejoint", {"name": "diag_object_free"})
        ET.SubElement(object_body, "geom", {
            "name": OBJECT_GEOM,
            "type": "cylinder",
            "size": f"{OBJECT_RADIUS_M:.12g} {OBJECT_HALF_HEIGHT_M:.12g}",
            "mass": f"{OBJECT_MASS_KG:.12g}",
            "friction": " ".join(f"{v:.12g}" for v in OBJECT_FRICTION),
            "condim": "4",
            "rgba": "0.13 0.52 0.76 1",
        })

    for element in list(root.findall("actuator")):
        root.remove(element)
    actuator = ET.SubElement(root, "actuator")
    for name, joint, cap in (
        ("m0_driver_servo", RIGHT_DRIVER, M0_EFFORT_CAP_NM[RIGHT_DRIVER]),
        ("m0_follower_servo", RIGHT_FOLLOWER, M0_EFFORT_CAP_NM[RIGHT_FOLLOWER]),
    ):
        ET.SubElement(actuator, "motor", {
            "name": name,
            "joint": joint,
            "gear": "1",
            "ctrllimited": "true",
            "ctrlrange": f"{-cap:.12g} {cap:.12g}",
            "forcelimited": "true",
            "forcerange": f"{-cap:.12g} {cap:.12g}",
        })
    ET.SubElement(actuator, "motor", {
        "name": "m0_lift_carriage_servo",
        "joint": LIFT_JOINT,
        "gear": "1",
        "ctrllimited": "true",
        "ctrlrange": f"{-CARriage_FORCE_CAP_N:.12g} {CARriage_FORCE_CAP_N:.12g}",
        "forcelimited": "true",
        "forcerange": f"{-CARriage_FORCE_CAP_N:.12g} {CARriage_FORCE_CAP_N:.12g}",
    })

    # The public contact-pipeline abstraction owns both jaw servos from one aperture command.
    for equality in list(root.findall("equality")):
        root.remove(equality)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{TIMESTEP_S:.12g}")
    option.set("gravity", f"0 0 {-GRAVITY_M_S2:.12g}")
    option.set("integrator", "Euler")
    option.set("solver", "Newton")
    option.set("iterations", "100")

    base_world = {
        "pos_m": base_world_pos.tolist(),
        "quat_wxyz": base_world_quat.tolist(),
        "source_mount_pos_m": base_mount_pos.tolist(),
        "source_mount_quat_wxyz": base_mount_quat.tolist(),
        "source_mount_parent_role": "right_wrist_roll_link; replaced by SIMULATION_ONLY carriage",
    }
    geometry = {
        "object_center_world_m": object_world_pos.tolist() if include_object else None,
        "fixture_raise_from_previous_candidate_m": FIXTURE_RAISE_M,
        "object_relative_to_base_world_m": OBJECT_RELATIVE_TO_BASE_WORLD_M.tolist(),
        "pad_midpoint_at_open_relative_to_base_world_m": [-0.0001996246106712, 0.0127422484175874, -0.0886694901951736],
        "distal_insertion_from_open_pad_midpoint_m": [0.0, 0.0, -0.0363305098048264],
        "object_radius_m": OBJECT_RADIUS_M if include_object else None,
        "object_height_m": 2.0 * OBJECT_HALF_HEIGHT_M if include_object else None,
        "object_density_assumption_kg_m3": OBJECT_DENSITY_ASSUMPTION_KG_M3 if include_object else None,
        "object_mass_kg": OBJECT_MASS_KG if include_object else None,
        "object_shape_classification": "SIMULATION_ONLY solid-cylinder contact fixture; not the canonical bottle",
        "table_top_z_m": TABLE_TOP_Z_M if include_object else None,
        "table_half_size_m": TABLE_HALF_SIZE_M if include_object else None,
        "table_material_friction": OBJECT_FRICTION if include_object else None,
    }
    return ET.tostring(root, encoding="unicode"), {"base_world": base_world, "fixture_geometry": geometry}


def compile_model(include_object: bool, output: Path) -> tuple[mujoco.MjModel, dict[str, Any], str]:
    xml, info = build_xml(include_object)
    model = mujoco.MjSpec.from_string(xml).compile()
    if model.neq != 0:
        raise RuntimeError("M0 model must not use a right-jaw equality constraint")
    for joint_name, expected in ((RIGHT_DRIVER, [-1.0, 0.0]), (RIGHT_FOLLOWER, [0.0, 1.0])):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0 or not np.allclose(model.jnt_range[jid], expected, atol=0.0, rtol=0.0):
            raise RuntimeError(f"Source joint range changed or missing: {joint_name}")
    model_xml_path = output / ("m0_isolated_contact.xml" if include_object else "m0_isolated_no_object.xml")
    with model_xml_path.open("x", encoding="utf-8") as stream:
        stream.write(xml)
        stream.write("\n")
    return model, info, xml


def id_by_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, obj_type, name))
    if value < 0:
        raise RuntimeError(f"Missing compiled object {name}")
    return value


def geom_body_name(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"


def active_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [g for g in range(model.ngeom)
            if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])]


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[model.jnt_qposadr[jid]])


def joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qvel[model.jnt_dofadr[jid]])


def joint_qacc(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qacc[model.jnt_dofadr[jid]])


def source_limits(urdf_path: Path) -> dict[str, list[float]]:
    root = ET.parse(urdf_path).getroot()
    out = {}
    for name in (RIGHT_DRIVER, RIGHT_FOLLOWER):
        joint = next(j for j in root.findall("joint") if j.get("name") == name)
        limit = joint.find("limit")
        out[name] = [float(limit.get("lower")), float(limit.get("upper"))]
    return out


def initialize(model: mujoco.MjModel) -> tuple[mujoco.MjData, int]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    open_target = m0_targets(1.0)
    data.qpos[model.jnt_qposadr[id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_DRIVER)]] = open_target["driver_q"]
    data.qpos[model.jnt_qposadr[id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_FOLLOWER)]] = open_target["follower_q"]
    # Only the two jaw coordinates are initialized. The free object remains at the MJCF qpos0.
    mujoco.mj_forward(model, data)
    return data, 2


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, full, data.qM)
    return full


def actuate(model: mujoco.MjModel, data: mujoco.MjData, aperture: float,
            aperture_v: float, aperture_a: float, lift_target_m: float,
            lift_target_velocity_m_s: float = 0.0,
            lift_target_acceleration_m_s2: float = 0.0) -> dict[str, Any]:
    targets = m0_targets(aperture, aperture_v, aperture_a)
    full = mass_matrix(model, data)
    out = {}
    for name, aid_name, q_key, v_key, a_key in (
        (RIGHT_DRIVER, "m0_driver_servo", "driver_q", "driver_v", "driver_a"),
        (RIGHT_FOLLOWER, "m0_follower_servo", "follower_q", "follower_v", "follower_a"),
    ):
        jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        q, v = float(data.qpos[qadr]), float(data.qvel[dof])
        qerr = targets[q_key] - q
        verr = targets[v_key] - v
        pd_tau = M0_KP_NM_PER_RAD[name] * qerr + M0_KD_NMS_PER_RAD[name] * verr
        bias_tau = float(data.qfrc_bias[dof])
        accel_tau = float(full[dof, dof]) * targets[a_key]
        raw_tau = bias_tau + pd_tau + accel_tau
        cap = M0_EFFORT_CAP_NM[name]
        applied_tau = float(np.clip(raw_tau, -cap, cap))
        aid = id_by_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid_name)
        data.ctrl[aid] = applied_tau
        out[name] = {
            "target_q_rad": float(targets[q_key]),
            "target_qvel_rad_s": float(targets[v_key]),
            "target_qacc_rad_s2": float(targets[a_key]),
            "qpos_rad": q,
            "qvel_rad_s": v,
            "qacc_rad_s2": float(data.qacc[dof]),
            "position_error_rad": qerr,
            "velocity_error_rad_s": verr,
            "pd_torque_nm": pd_tau,
            "bias_feedforward_nm": bias_tau,
            "reference_acceleration_torque_nm": accel_tau,
            "requested_torque_nm": raw_tau,
            "applied_torque_nm": applied_tau,
            "actuator_force_nm_pre_step": float(data.actuator_force[aid]),
            "qfrc_actuator_nm_pre_step": float(data.qfrc_actuator[dof]),
            "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
            "mass_diagonal_kg_m2": float(full[dof, dof]),
            "effort_cap_nm": cap,
        }
    lift_aid = id_by_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m0_lift_carriage_servo")
    lift_jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)
    lift_qadr, lift_dof = int(model.jnt_qposadr[lift_jid]), int(model.jnt_dofadr[lift_jid])
    lift_q, lift_v = float(data.qpos[lift_qadr]), float(data.qvel[lift_dof])
    lift_pd_force = CARriage_KP_N_M * (lift_target_m - lift_q) + CARriage_KV_NS_M * (lift_target_velocity_m_s - lift_v)
    lift_bias_force = float(data.qfrc_bias[lift_dof])
    lift_acceleration_force = float(full[lift_dof, lift_dof]) * lift_target_acceleration_m_s2
    lift_requested_force = lift_bias_force + lift_pd_force + lift_acceleration_force
    lift_applied_force = float(np.clip(lift_requested_force, -CARriage_FORCE_CAP_N, CARriage_FORCE_CAP_N))
    data.ctrl[lift_aid] = lift_applied_force
    out["m0_lift_carriage"] = {
        "target_m": float(lift_target_m),
        "target_velocity_m_s": float(lift_target_velocity_m_s),
        "target_acceleration_m_s2": float(lift_target_acceleration_m_s2),
        "position_m": lift_q,
        "velocity_m_s": lift_v,
        "pd_force_n": float(lift_pd_force),
        "bias_feedforward_force_n": lift_bias_force,
        "reference_acceleration_force_n": float(lift_acceleration_force),
        "requested_force_n": float(lift_requested_force),
        "applied_force_n": lift_applied_force,
        "actuator_force_n_pre_step": float(data.actuator_force[lift_aid]),
        "force_cap_n": CARriage_FORCE_CAP_N,
    }
    return out


def point_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point: np.ndarray) -> np.ndarray:
    if body_id == 0:
        return np.zeros(3, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
    return jacp @ data.qvel


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    object_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM))
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        force_local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, force_local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        normal_1_to_2_world = frame[0].copy()
        force_on_second_world = frame.T @ force_local[:3]
        object_in_contact = object_geom in (geom1, geom2)
        if geom2 == object_geom:
            force_on_object_world = force_on_second_world
            normal_on_object_world = normal_1_to_2_world
            other_geom = geom1
            other_body = body1
        elif geom1 == object_geom:
            force_on_object_world = -force_on_second_world
            normal_on_object_world = -normal_1_to_2_world
            other_geom = geom2
            other_body = body2
        else:
            force_on_object_world = np.zeros(3, dtype=float)
            normal_on_object_world = np.zeros(3, dtype=float)
            other_geom = -1
            other_body = -1
        normal_component = float(np.dot(force_on_object_world, normal_on_object_world)) if object_in_contact else 0.0
        tangent_force = (force_on_object_world - normal_component * normal_on_object_world
                         if object_in_contact else np.zeros(3, dtype=float))
        rel_tangent_speed = 0.0
        if object_in_contact:
            object_body = int(model.geom_bodyid[object_geom])
            v_obj = point_velocity(model, data, object_body, np.asarray(contact.pos, dtype=float))
            v_other = point_velocity(model, data, other_body, np.asarray(contact.pos, dtype=float))
            relative = v_obj - v_other
            rel_tangent = relative - float(np.dot(relative, normal_on_object_world)) * normal_on_object_world
            rel_tangent_speed = float(np.linalg.norm(rel_tangent))
        rows.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world",
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world",
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
            "contact_position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_frame_rows_world": frame.tolist(),
            "force_torque_contact_frame": force_local.tolist(),
            "force_on_object_world_n": force_on_object_world.tolist() if object_in_contact else None,
            "normal_on_object_world": normal_on_object_world.tolist() if object_in_contact else None,
            "normal_force_on_object_n": max(0.0, normal_component),
            "tangent_force_on_object_world_n": tangent_force.tolist() if object_in_contact else None,
            "tangent_force_magnitude_n": float(np.linalg.norm(tangent_force)) if object_in_contact else 0.0,
            "object_tangential_slip_speed_m_s": rel_tangent_speed,
            "object_contact": bool(object_in_contact),
            "object_other_body": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or "world") if object_in_contact else None,
        })
    return rows


def geom_distance(model: mujoco.MjModel, data: mujoco.MjData, geom1: int, geom2: int) -> tuple[float, list[float]]:
    segment = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, geom1, geom2, 1.0, segment))
    return distance, segment.tolist()


def static_preflight(model: mujoco.MjModel, output: Path) -> dict[str, Any]:
    data, init_jaw_writes = initialize(model)
    pad_body_id = id_by_name(model, mujoco.mjtObj.mjOBJ_BODY, NARROW_PAD)
    object_body_id = id_by_name(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    open_pad_body_z = float(data.xpos[pad_body_id][2])
    object_center_z = float(data.xpos[object_body_id][2])
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = camera(model)
    Image.fromarray(render_frame(renderer, model, data, cam)).save(output / "static_preflight_open.png")
    open_contacts = contact_rows(model, data)
    object_geom = id_by_name(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)
    table_geom = id_by_name(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    target_bodies = {NARROW_PAD, WIDE_PAD}
    collision_ids = active_geom_ids(model)
    open_rows = []
    for gid in collision_ids:
        if gid in (object_geom, table_geom):
            continue
        distance, segment = geom_distance(model, data, object_geom, gid)
        open_rows.append({
            "body": geom_body_name(model, gid),
            "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "distance_m": distance,
            "witness_segment_world_m": segment,
        })
    open_robot = [row for row in open_rows if row["body"] != "world"]
    nonpad = [row for row in open_robot if row["body"] not in target_bodies]
    table_robot = []
    for gid in collision_ids:
        if gid in (object_geom, table_geom) or geom_body_name(model, gid) == "world":
            continue
        d, segment = geom_distance(model, data, table_geom, gid)
        table_robot.append({"body": geom_body_name(model, gid), "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid), "distance_m": d, "witness_segment_world_m": segment})

    pads = {}
    for body_name in (NARROW_PAD, WIDE_PAD):
        bodyid = id_by_name(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        gids = [g for g in collision_ids if int(model.geom_bodyid[g]) == bodyid]
        if not gids:
            raise RuntimeError(f"No active collision geom on target body {body_name}")
        vals = [(geom_distance(model, data, object_geom, g)[0], g) for g in gids]
        pads[body_name] = min(vals, key=lambda x: x[0])[0]

    sweep = []
    for aperture in np.linspace(1.0, CONTACT_APERTURE_RATIO, 61):
        targets = m0_targets(float(aperture))
        for name, value in ((RIGHT_DRIVER, targets["driver_q"]), (RIGHT_FOLLOWER, targets["follower_q"])):
            jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        pad_clear = {}
        nonpad_clear = []
        for gid in collision_ids:
            if gid in (object_geom, table_geom) or geom_body_name(model, gid) == "world":
                continue
            distance = geom_distance(model, data, object_geom, gid)[0]
            body = geom_body_name(model, gid)
            if body in target_bodies:
                pad_clear[body] = min(pad_clear.get(body, float("inf")), distance)
            else:
                nonpad_clear.append((distance, body))
        table_clear = [geom_distance(model, data, table_geom, gid)[0]
                       for gid in collision_ids if gid not in (object_geom, table_geom) and geom_body_name(model, gid) != "world"]
        sweep.append({
            "aperture_ratio": float(aperture),
            "driver_q_rad": float(targets["driver_q"]),
            "follower_q_rad": float(targets["follower_q"]),
            "pad_clearance_m": pad_clear,
            "minimum_nonpad_object_clearance_m": min((x[0] for x in nonpad_clear), default=None),
            "minimum_table_robot_clearance_m": min(table_clear, default=None),
        })

    Image.fromarray(render_frame(renderer, model, data, cam)).save(output / "static_preflight_contact_endpoint.png")
    renderer.close()

    def decreases(family: str) -> bool:
        series = [row["pad_clearance_m"][family] for row in sweep]
        return all(series[i + 1] <= series[i] + 1e-6 for i in range(len(series) - 1))

    initial_object_contacts = [c for c in open_contacts if c["object_contact"] and c["object_other_body"] != "world"]
    source_self_contacts = [c for c in open_contacts if not c["object_contact"] and c["distance_m"] < 0]
    open_clear = min((row["distance_m"] for row in open_robot), default=float("inf")) > 0.0
    nonpad_clear = all(row["distance_m"] > 0.0 for row in nonpad)
    table_clear = all(row["distance_m"] > 0.0 for row in table_robot)
    contact_reached = all(sweep[-1]["pad_clearance_m"][body] <= 0 for body in target_bodies)
    nonpad_sweep_clear = all(row["minimum_nonpad_object_clearance_m"] is None or row["minimum_nonpad_object_clearance_m"] > 0 for row in sweep)
    table_sweep_clear = all(row["minimum_table_robot_clearance_m"] is None or row["minimum_table_robot_clearance_m"] > 0 for row in sweep)
    result = {
        "status": "PASS" if (open_clear and nonpad_clear and table_clear and not initial_object_contacts and contact_reached and nonpad_sweep_clear and table_sweep_clear and decreases(NARROW_PAD) and decreases(WIDE_PAD)) else "FAIL",
        "mj_step_calls": 0,
        "object_qpos_writes": 0,
        "initial_jaw_initialization_writes": init_jaw_writes,
        "initial_object_contact_with_robot": initial_object_contacts,
        "initial_source_tree_self_contacts": source_self_contacts,
        "open_object_clearances": open_rows,
        "open_table_to_robot_clearances": table_robot,
        "open_target_pad_clearance_m": pads,
        "minimum_nonpad_object_clearance_m_over_sweep": min((row["minimum_nonpad_object_clearance_m"] for row in sweep if row["minimum_nonpad_object_clearance_m"] is not None), default=None),
        "minimum_table_robot_clearance_m_over_sweep": min((row["minimum_table_robot_clearance_m"] for row in sweep if row["minimum_table_robot_clearance_m"] is not None), default=None),
        "target_pad_clearance_monotonic": {body: decreases(body) for body in target_bodies},
        "both_target_pad_geoms_reach_object_by_endpoint": contact_reached,
        "nonpad_object_clear_over_contact_sweep": nonpad_sweep_clear,
        "table_clear_of_robot_over_contact_sweep": table_sweep_clear,
        "sweep_samples": sweep,
        "interpretation": "SIMULATION_ONLY M0 source-tree contact geometry. Source loop branches remain fixed; this is not a physical loop validation.",
        "fixture_raise_from_previous_candidate_m": FIXTURE_RAISE_M,
        "open_narrow_pad_body_origin_world_z_m": open_pad_body_z,
        "rod_bottom_clearance_from_table_top_m": object_center_z - OBJECT_HALF_HEIGHT_M - TABLE_TOP_Z_M,
        "rod_top_world_z_m": object_center_z + OBJECT_HALF_HEIGHT_M,
    }
    return result


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return id_by_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def object_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any] | None:
    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY))
    if bid < 0:
        return None
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "diag_object_free"))
    qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    qvel = data.qvel[dadr:dadr + 6].copy()
    return {
        "position_world_m": data.xpos[bid].astype(float).tolist(),
        "quaternion_wxyz": data.xquat[bid].astype(float).tolist(),
        "qpos_freejoint": data.qpos[qadr:qadr + 7].astype(float).tolist(),
        "qvel_freejoint": qvel.astype(float).tolist(),
        "linear_velocity_bound_m_s": float(np.linalg.norm(qvel[3:6])),
    }


def contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    rows = contact_rows(model, data)
    object_rows = [row for row in rows if row["object_contact"]]
    by_family = {NARROW_PAD: [], WIDE_PAD: []}
    table_forces = []
    hand_forces = []
    nonpad_rows = []
    for row in object_rows:
        body = str(row["object_other_body"])
        if body in by_family:
            by_family[body].append(row)
            hand_forces.append(row["force_on_object_world_n"])
        elif body == "world":
            table_forces.append(row["force_on_object_world_n"])
        else:
            nonpad_rows.append(row)
    def sum_forces(records: list[list[float]]) -> list[float]:
        if not records:
            return [0.0, 0.0, 0.0]
        return np.sum(np.asarray(records, dtype=float), axis=0).tolist()
    family_force = {
        family: {
            "contact_count": len(rows_f),
            "normal_force_sum_n": float(sum(max(0.0, row["normal_force_on_object_n"]) for row in rows_f)),
            "world_force_sum_on_object_n": sum_forces([row["force_on_object_world_n"] for row in rows_f]),
            "max_penetration_m": max((row["penetration_m"] for row in rows_f), default=0.0),
            "mean_tangential_slip_speed_m_s": float(np.mean([row["object_tangential_slip_speed_m_s"] for row in rows_f])) if rows_f else 0.0,
        }
        for family, rows_f in by_family.items()
    }
    return {
        "contacts": rows,
        "object_contact_count": len(object_rows),
        "force_on_object_world_sum_n": sum_forces([row["force_on_object_world_n"] for row in object_rows]),
        "table_support_force_world_sum_n": sum_forces(table_forces),
        "hand_support_force_world_sum_n": sum_forces(hand_forces),
        "family": family_force,
        "nonpad_object_contacts": nonpad_rows,
    }


def camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    base_pos = np.array(CARriage_WORLD_POS_M, dtype=float)
    base_q = compose_quaternions(quaternion_yaw(CARriage_WORLD_YAW_RAD),
                                 np.array([0.000796327, 1.0, 0.0, 0.0], dtype=float))
    base_pos += quat_matrix(quaternion_yaw(CARriage_WORLD_YAW_RAD)) @ np.array([0.00196, 0.00035, -0.062])
    obj_pos = base_pos + OBJECT_RELATIVE_TO_BASE_WORLD_M
    cam.lookat[:] = obj_pos
    cam.distance = 0.42
    cam.azimuth = 120.0
    cam.elevation = -16.0
    return cam


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
                 cam: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def phase_schedule(include_object: bool) -> list[tuple[str, float, float, float, float]]:
    if not include_object:
        return [
            ("OPEN_HOLD", 0.25, 1.0, 1.0, 0.0),
            ("CLOSE_TO_OBJECT_APERTURE", MOTION_DURATION_S, 1.0, CONTACT_APERTURE_RATIO, 0.0),
            ("CLOSED_AT_OBJECT_WIDTH_HOLD", 0.25, CONTACT_APERTURE_RATIO, CONTACT_APERTURE_RATIO, 0.0),
            ("REOPEN", MOTION_DURATION_S, CONTACT_APERTURE_RATIO, 1.0, 0.0),
            ("OPEN_HOLD_FINAL", 0.25, 1.0, 1.0, 0.0),
        ]
    return [
        ("SETTLE_OBJECT_ON_TABLE", 0.5, 1.0, 1.0, 0.0),
        ("OPEN_HOLD", 0.25, 1.0, 1.0, 0.0),
        ("CLOSE_TO_CONTACT", MOTION_DURATION_S, 1.0, CONTACT_APERTURE_RATIO, 0.0),
        ("BILATERAL_CONTACT_HOLD", 1.0, CONTACT_APERTURE_RATIO, CONTACT_APERTURE_RATIO, 0.0),
    ]


def check_safety(model: mujoco.MjModel, data: mujoco.MjData,
                 object_enabled: bool, contact_seen: bool) -> list[str]:
    failures = []
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all()):
        failures.append("nonfinite_state")
        return failures
    for name, lower, upper in ((RIGHT_DRIVER, -1.0, 0.0), (RIGHT_FOLLOWER, 0.0, 1.0)):
        q = joint_qpos(model, data, name)
        if q < lower - 1e-6 or q > upper + 1e-6:
            failures.append(f"source_limit_violation:{name}")
        if abs(joint_qvel(model, data, name)) > M0_MAX_JAW_SPEED_RAD_S:
            failures.append(f"jaw_speed_bound:{name}")
        if abs(joint_qacc(model, data, name)) > M0_MAX_JAW_ACCEL_RAD_S2:
            failures.append(f"jaw_acceleration_bound:{name}")
    carriage_q = joint_qpos(model, data, LIFT_JOINT)
    if (carriage_q < -FIXTURE_LIMIT_NUMERICAL_TOLERANCE_M
            or carriage_q > MAX_SHORT_LIFT_M + FIXTURE_LIMIT_NUMERICAL_TOLERANCE_M):
        failures.append("simulation_fixture_carriage_limit_violation")
    if object_enabled:
        rows = contact_metrics(model, data)
        max_pen = max((row["penetration_m"] for row in rows["contacts"]), default=0.0)
        if max_pen > M0_MAX_PENETRATION_M:
            failures.append("contact_penetration_bound")
        if rows["nonpad_object_contacts"]:
            failures.append("unexpected_nonpad_object_contact")
        for row in rows["contacts"]:
            if TABLE_GEOM in (row.get("geom1"), row.get("geom2")) and row["penetration_m"] > 1e-6:
                other_geom = row.get("geom2") if row.get("geom1") == TABLE_GEOM else row.get("geom1")
                if other_geom != OBJECT_GEOM:
                    failures.append("table_robot_collision")
                    break
    return failures


def step_sequence(model: mujoco.MjModel, data: mujoco.MjData, *, include_object: bool,
                  output: Path, prefix: str, stop_on_bilateral_loss: bool = False,
                  initial_jaw_writes: int = 0) -> dict[str, Any]:
    trace_path = output / f"{prefix}_trace.jsonl"
    pose_path = output / f"{prefix}_object_pose_trace.jsonl" if include_object else None
    contact_path = output / f"{prefix}_contact_trace.jsonl" if include_object else None
    for path in (trace_path, pose_path, contact_path):
        if path is not None and path.exists():
            raise FileExistsError(path)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = camera(model)
    video_path = output / f"{prefix}.mp4"
    writer = None
    video_error = None
    try:
        writer = imageio.get_writer(video_path, fps=25, codec="libx264", macro_block_size=1)
    except Exception as exc:
        video_error = repr(exc)
        video_path = output / f"{prefix}.gif"
        writer = imageio.get_writer(video_path, mode="I", duration=0.04)

    dt = float(model.opt.timestep)
    start_step = int(round(float(data.time) / dt))
    step = start_step
    qpos_writes_after_start = 0
    first_object_contact_step = None
    phase_stats: dict[str, dict[str, Any]] = {}
    safety_failure = None
    hold_samples: list[dict[str, Any]] = []
    all_stage_metrics: list[dict[str, Any]] = []
    max_object_z = -float("inf")
    max_object_frame = None
    image_paths = {
        "open": output / f"{prefix}_open.png",
        "first_contact": output / f"{prefix}_first_contact.png",
        "closed_hold": output / f"{prefix}_closed_hold.png",
        "maximum_lift": output / f"{prefix}_maximum_lift.png",
    }
    for image_path in image_paths.values():
        if image_path.exists():
            raise FileExistsError(image_path)
    object_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)) if include_object else -1
    table_object_static_samples: list[float] = []
    try:
        with trace_path.open("x", encoding="utf-8") as trace_stream:
            pose_stream = pose_path.open("x", encoding="utf-8") if pose_path is not None else None
            contact_stream = contact_path.open("x", encoding="utf-8") if contact_path is not None else None
            try:
                if include_object:
                    Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["open"])
                for phase, duration, start, end, lift_end in phase_schedule(include_object):
                    phase_start_step = step
                    phase_samples = []
                    phase_frames = max(1, round(duration / dt))
                    for local in range(phase_frames):
                        t = local * dt
                        if phase in {"CLOSE_TO_OBJECT_APERTURE", "CLOSE_TO_CONTACT", "REOPEN"}:
                            aperture, aperture_v, aperture_a = minimum_jerk(t, duration, start, end)
                        else:
                            aperture, aperture_v, aperture_a = start, 0.0, 0.0
                        lift_target = 0.0
                        if phase == "LOAD_TRANSFER_PROBE_1MM":
                            lift_target = minimum_jerk(t, duration, 0.0, 0.001)[0]
                        elif phase == "SHORT_LIFT_10MM":
                            lift_target = minimum_jerk(t, duration, 0.001, 0.010)[0]
                        elif phase == "SUSPENDED_HOLD":
                            lift_target = 0.010

                        mujoco.mj_forward(model, data)
                        control = actuate(model, data, aperture, aperture_v, aperture_a, lift_target)
                        mujoco.mj_step(model, data)
                        step += 1
                        fails = check_safety(model, data, include_object, first_object_contact_step is not None)
                        contacts = contact_metrics(model, data) if include_object else None
                        if include_object and contacts["object_contact_count"]:
                            if first_object_contact_step is None:
                                first_object_contact_step = step
                                Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["first_contact"])
                        driver_q = joint_qpos(model, data, RIGHT_DRIVER)
                        follower_q = joint_qpos(model, data, RIGHT_FOLLOWER)
                        contact_normals = contacts["family"] if contacts else None
                        row = {
                            "step": step,
                            "time_s": float(data.time),
                            "phase": phase,
                            "aperture_target": float(aperture),
                            "aperture_velocity_target_s": float(aperture_v),
                            "object_qpos_writes_after_start": qpos_writes_after_start,
                            "driver_qpos_rad": driver_q,
                            "follower_qpos_rad": follower_q,
                            "mimic_residual_rad": driver_q + follower_q,
                            "driver_qvel_rad_s": joint_qvel(model, data, RIGHT_DRIVER),
                            "follower_qvel_rad_s": joint_qvel(model, data, RIGHT_FOLLOWER),
                            "driver_qacc_rad_s2": joint_qacc(model, data, RIGHT_DRIVER),
                            "follower_qacc_rad_s2": joint_qacc(model, data, RIGHT_FOLLOWER),
                            "control": control,
                            "contact_family_metrics": contact_normals,
                            "contacts": contacts["contacts"] if contacts else [],
                            "safety_failures": fails,
                        }
                        if include_object:
                            obj_state = object_summary(model, data)
                            row["object_position_world_m"] = obj_state["position_world_m"]
                            row["object_linear_speed_m_s"] = obj_state["linear_velocity_bound_m_s"]
                            if obj_state["position_world_m"][2] > max_object_z:
                                max_object_z = float(obj_state["position_world_m"][2])
                                max_object_frame = render_frame(renderer, model, data, cam)
                        write_jsonl_line(trace_stream, row)
                        if pose_stream is not None:
                            obj = object_summary(model, data)
                            pose_row = {"step": step, "time_s": float(data.time), "phase": phase, **(obj or {})}
                            write_jsonl_line(pose_stream, pose_row)
                        if contact_stream is not None and contacts is not None:
                            write_jsonl_line(contact_stream, {"step": step, "time_s": float(data.time), "phase": phase, **contacts})
                        phase_samples.append(row)
                        if phase == "BILATERAL_CONTACT_HOLD":
                            hold_samples.append(row)
                        if include_object and phase == "SETTLE_OBJECT_ON_TABLE":
                            table_z = float(contacts["table_support_force_world_sum_n"][2])
                            table_object_static_samples.append(table_z)
                        if step % 25 == 0 or step == 1:
                            writer.append_data(render_frame(renderer, model, data, cam))
                        if fails:
                            safety_failure = {"step": step, "time_s": float(data.time), "phase": phase, "failures": fails}
                            break
                        if stop_on_bilateral_loss and phase == "BILATERAL_CONTACT_HOLD":
                            narrow = contact_normals[NARROW_PAD]["normal_force_sum_n"] > FORCE_BEARING_EPS_N
                            wide = contact_normals[WIDE_PAD]["normal_force_sum_n"] > FORCE_BEARING_EPS_N
                            if not (narrow and wide):
                                # Continue collecting the one-second persistence trace; this is a measured FAIL, not a forced state.
                                pass
                    phase_stats[phase] = summarize_phase(phase_samples, include_object)
                    all_stage_metrics.append({"phase": phase, "step_start": phase_start_step + 1,
                                              "step_end": step, "summary": phase_stats[phase]})
                    if include_object and phase == "BILATERAL_CONTACT_HOLD" and phase_samples:
                        Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["closed_hold"])
                    if safety_failure:
                        break
                # load transfer is attempted only after a complete force-bearing bilateral hold
                hold_gate = bilateral_hold_gate(hold_samples)
                if include_object and not safety_failure and hold_gate["pass"]:
                    for phase, duration, start, end, lift_end in (
                        ("LOAD_TRANSFER_PROBE_1MM", 0.5, 0.0, 0.0, 0.001),
                    ):
                        phase_start_step = step
                        phase_samples = []
                        for local in range(max(1, round(duration / dt))):
                            t = local * dt
                            aperture = CONTACT_APERTURE_RATIO
                            aperture_v = aperture_a = 0.0
                            lift_target = minimum_jerk(t, duration, 0.0, lift_end)[0]
                            mujoco.mj_forward(model, data)
                            control = actuate(model, data, aperture, aperture_v, aperture_a, lift_target)
                            mujoco.mj_step(model, data)
                            step += 1
                            fails = check_safety(model, data, True, first_object_contact_step is not None)
                            contacts = contact_metrics(model, data)
                            row = {
                                "step": step, "time_s": float(data.time), "phase": phase,
                                "aperture_target": aperture,
                                "object_qpos_writes_after_start": qpos_writes_after_start,
                                "driver_qpos_rad": joint_qpos(model, data, RIGHT_DRIVER),
                                "follower_qpos_rad": joint_qpos(model, data, RIGHT_FOLLOWER),
                                "mimic_residual_rad": joint_qpos(model, data, RIGHT_DRIVER) + joint_qpos(model, data, RIGHT_FOLLOWER),
                                "driver_qvel_rad_s": joint_qvel(model, data, RIGHT_DRIVER),
                                "follower_qvel_rad_s": joint_qvel(model, data, RIGHT_FOLLOWER),
                                "driver_qacc_rad_s2": joint_qacc(model, data, RIGHT_DRIVER),
                                "follower_qacc_rad_s2": joint_qacc(model, data, RIGHT_FOLLOWER),
                                "control": control,
                                "contact_family_metrics": contacts["family"],
                                "contacts": contacts["contacts"],
                                "safety_failures": fails,
                            }
                            obj_state = object_summary(model, data)
                            row["object_position_world_m"] = obj_state["position_world_m"]
                            row["object_linear_speed_m_s"] = obj_state["linear_velocity_bound_m_s"]
                            if obj_state["position_world_m"][2] > max_object_z:
                                max_object_z = float(obj_state["position_world_m"][2])
                                max_object_frame = render_frame(renderer, model, data, cam)
                            write_jsonl_line(trace_stream, row)
                            write_jsonl_line(pose_stream, {"step": step, "time_s": float(data.time), "phase": phase, **obj_state})
                            write_jsonl_line(contact_stream, {"step": step, "time_s": float(data.time), "phase": phase, **contacts})
                            phase_samples.append(row)
                            if step % 25 == 0:
                                writer.append_data(render_frame(renderer, model, data, cam))
                            if fails:
                                safety_failure = {"step": step, "time_s": float(data.time), "phase": phase, "failures": fails}
                                break
                        phase_stats[phase] = summarize_phase(phase_samples, True)
                        all_stage_metrics.append({"phase": phase, "step_start": phase_start_step + 1,
                                                  "step_end": step, "summary": phase_stats[phase]})
                        if not safety_failure:
                            transfer = load_transfer_gate(phase_samples, OBJECT_MASS_KG, GRAVITY_M_S2)
                            if transfer["pass"]:
                                for phase2, duration2, lift_end2 in (("SHORT_LIFT_10MM", 1.0, 0.010),
                                                                     ("SUSPENDED_HOLD", 1.0, 0.010)):
                                    phase_start_step = step
                                    phase_samples = []
                                    for local in range(max(1, round(duration2 / dt))):
                                        t2 = local * dt
                                        lift_target = (minimum_jerk(t2, duration2, 0.001, lift_end2)[0]
                                                       if phase2 == "SHORT_LIFT_10MM" else 0.010)
                                        aperture = CONTACT_APERTURE_RATIO
                                        mujoco.mj_forward(model, data)
                                        control = actuate(model, data, aperture, 0.0, 0.0, lift_target)
                                        mujoco.mj_step(model, data)
                                        step += 1
                                        fails = check_safety(model, data, True, first_object_contact_step is not None)
                                        contacts = contact_metrics(model, data)
                                        row = {
                                            "step": step, "time_s": float(data.time), "phase": phase2,
                                            "aperture_target": aperture,
                                            "object_qpos_writes_after_start": qpos_writes_after_start,
                                            "driver_qpos_rad": joint_qpos(model, data, RIGHT_DRIVER),
                                            "follower_qpos_rad": joint_qpos(model, data, RIGHT_FOLLOWER),
                                            "mimic_residual_rad": joint_qpos(model, data, RIGHT_DRIVER) + joint_qpos(model, data, RIGHT_FOLLOWER),
                                            "driver_qvel_rad_s": joint_qvel(model, data, RIGHT_DRIVER),
                                            "follower_qvel_rad_s": joint_qvel(model, data, RIGHT_FOLLOWER),
                                            "driver_qacc_rad_s2": joint_qacc(model, data, RIGHT_DRIVER),
                                            "follower_qacc_rad_s2": joint_qacc(model, data, RIGHT_FOLLOWER),
                                            "control": control,
                                            "contact_family_metrics": contacts["family"],
                                            "contacts": contacts["contacts"],
                                            "safety_failures": fails,
                                        }
                                        obj_state = object_summary(model, data)
                                        row["object_position_world_m"] = obj_state["position_world_m"]
                                        row["object_linear_speed_m_s"] = obj_state["linear_velocity_bound_m_s"]
                                        if obj_state["position_world_m"][2] > max_object_z:
                                            max_object_z = float(obj_state["position_world_m"][2])
                                            max_object_frame = render_frame(renderer, model, data, cam)
                                        write_jsonl_line(trace_stream, row)
                                        write_jsonl_line(pose_stream, {"step": step, "time_s": float(data.time), "phase": phase2, **obj_state})
                                        write_jsonl_line(contact_stream, {"step": step, "time_s": float(data.time), "phase": phase2, **contacts})
                                        phase_samples.append(row)
                                        if step % 25 == 0:
                                            writer.append_data(render_frame(renderer, model, data, cam))
                                        if fails:
                                            safety_failure = {"step": step, "time_s": float(data.time), "phase": phase2, "failures": fails}
                                            break
                                    phase_stats[phase2] = summarize_phase(phase_samples, True)
                                    all_stage_metrics.append({"phase": phase2, "step_start": phase_start_step + 1,
                                                              "step_end": step, "summary": phase_stats[phase2]})
                                    if safety_failure:
                                        break
                            else:
                                phase_stats["load_transfer_gate"] = transfer
                elif include_object:
                    phase_stats["load_transfer_gate"] = {
                        "pass": False,
                        "status": "NOT_RUN",
                        "reason": "One-second bilateral force-bearing hold gate did not pass; no lift command was issued.",
                    }
            finally:
                if pose_stream is not None:
                    pose_stream.close()
                if contact_stream is not None:
                    contact_stream.close()
    finally:
        renderer.close()
        writer.close()
    if max_object_frame is not None and include_object:
        Image.fromarray(max_object_frame).save(image_paths["maximum_lift"])
    if video_error:
        phase_stats["video_fallback_reason"] = video_error

    contact_rows_all = []
    if contact_path is not None:
        with contact_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                contact_rows_all.append(record)
    object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)) if include_object else -1
    table_geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)) if include_object else -1
    table_contact_steps = sum(
        1 for rec in contact_rows_all
        if any(TABLE_GEOM == c["geom1"] or TABLE_GEOM == c["geom2"] for c in rec.get("contacts", []))
    )
    final_object = object_summary(model, data) if include_object else None
    return {
        "status": "FAIL" if safety_failure else "FINISHED_FINITE",
        "simulation_only": True,
        "mj_step_calls": step - start_step,
        "active_rollout_object_qpos_writes": qpos_writes_after_start,
        "initial_jaw_qpos_assignments_before_rollout": initial_jaw_writes,
        "first_object_contact_step": first_object_contact_step,
        "first_object_contact_time_s": first_object_contact_step * dt if first_object_contact_step else None,
        "phase_summaries": phase_stats,
        "ordered_phase_summaries": all_stage_metrics,
        "safety_failure": safety_failure,
        "hold_gate": bilateral_hold_gate(hold_samples) if include_object else None,
        "table_contact_steps_total": table_contact_steps,
        "final_object": final_object,
        "trace": trace_path.name,
        "contact_trace": contact_path.name if contact_path else None,
        "object_pose_trace": pose_path.name if pose_path else None,
        "video": video_path.name,
        "video_fallback_reason": video_error,
        "images": {name: path.name for name, path in image_paths.items() if path.is_file()},
    }


def run_sequence(model: mujoco.MjModel, data: mujoco.MjData, *, include_object: bool,
                 output: Path, prefix: str,
                 sequence: list[tuple[str, float, float, float, float]],
                 initial_jaw_writes: int = 0,
                 continue_data: bool = False) -> dict[str, Any]:
    trace_path = output / f"{prefix}_trace.jsonl"
    pose_path = output / f"{prefix}_object_pose_trace.jsonl" if include_object else None
    contact_path = output / f"{prefix}_contact_trace.jsonl" if include_object else None
    image_paths = {
        "open": output / f"{prefix}_open.png",
        "first_contact": output / f"{prefix}_first_contact.png",
        "hold_end": output / f"{prefix}_hold_end.png",
        "max_lift": output / f"{prefix}_max_lift.png",
    }
    for path in (trace_path, pose_path, contact_path, *image_paths.values()):
        if path is not None and path.exists():
            raise FileExistsError(path)

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = camera(model)
    video_path = output / f"{prefix}.mp4"
    video_error = None
    try:
        writer = imageio.get_writer(video_path, fps=25, codec="libx264", macro_block_size=1)
    except Exception as exc:
        video_error = repr(exc)
        video_path = output / f"{prefix}.gif"
        writer = imageio.get_writer(video_path, mode="I", duration=0.04)

    start_step = int(round(float(data.time) / float(model.opt.timestep)))
    step = start_step
    first_object_contact_step = None
    safety_failure = None
    phase_summaries = []
    hold_samples: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    max_object_z = -float("inf")
    max_object_z_frame = None
    object_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM)) if include_object else -1
    dt = float(model.opt.timestep)

    try:
        with trace_path.open("x", encoding="utf-8") as trace_stream:
            pose_stream = pose_path.open("x", encoding="utf-8") if pose_path is not None else None
            contact_stream = contact_path.open("x", encoding="utf-8") if contact_path is not None else None
            try:
                for phase, duration, aperture_start, aperture_end, lift_end in sequence:
                    phase_start_step = step
                    phase_rows = []
                    phase_steps = max(1, int(round(duration / dt)))
                    if phase == "OPEN_HOLD" and include_object and not (output / f"{prefix}_open.png").exists():
                        Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["open"])
                    for local_step in range(phase_steps):
                        t = local_step * dt
                        if phase in {"CLOSE_TO_OBJECT_APERTURE", "CLOSE_TO_CONTACT", "REOPEN"}:
                            aperture, aperture_v, aperture_a = minimum_jerk(t, duration, aperture_start, aperture_end)
                        else:
                            aperture, aperture_v, aperture_a = aperture_start, 0.0, 0.0
                        lift_target = 0.0
                        lift_target_velocity = 0.0
                        lift_target_acceleration = 0.0
                        if phase == "LOAD_TRANSFER_PROBE_1MM":
                            lift_target, lift_target_velocity, lift_target_acceleration = minimum_jerk(t, duration, 0.0, lift_end)
                        elif phase == "SHORT_LIFT_10MM":
                            lift_target, lift_target_velocity, lift_target_acceleration = minimum_jerk(t, duration, 0.001, lift_end)
                        elif phase == "SUSPENDED_HOLD_1S":
                            lift_target = lift_end

                        mujoco.mj_forward(model, data)
                        control = actuate(model, data, aperture, aperture_v, aperture_a, lift_target,
                                          lift_target_velocity, lift_target_acceleration)
                        mujoco.mj_step(model, data)
                        step += 1
                        mass_after = mass_matrix(model, data)
                        rhs_without_bias = (data.qfrc_actuator + data.qfrc_passive
                                            + data.qfrc_applied + data.qfrc_constraint)
                        equation_residual_minus_bias = mass_after @ data.qacc - (rhs_without_bias - data.qfrc_bias)
                        equation_residual_plus_bias = mass_after @ data.qacc - (rhs_without_bias + data.qfrc_bias)
                        joint_dynamics = {}
                        for name, aid_name in ((RIGHT_DRIVER, "m0_driver_servo"),
                                               (RIGHT_FOLLOWER, "m0_follower_servo")):
                            jid = id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                            dof = int(model.jnt_dofadr[jid])
                            aid = id_by_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid_name)
                            control[name]["actuator_force_nm_post_step"] = float(data.actuator_force[aid])
                            control[name]["qfrc_actuator_nm_post_step"] = float(data.qfrc_actuator[dof])
                            joint_dynamics[name] = {
                                "qacc_rad_s2": float(data.qacc[dof]),
                                "qfrc_actuator_nm": float(data.qfrc_actuator[dof]),
                                "qfrc_bias_nm": float(data.qfrc_bias[dof]),
                                "qfrc_constraint_nm": float(data.qfrc_constraint[dof]),
                                "qfrc_passive_nm": float(data.qfrc_passive[dof]),
                                "qfrc_applied_nm": float(data.qfrc_applied[dof]),
                                "actuator_force_nm": float(data.actuator_force[aid]),
                                "ctrl_nm": float(data.ctrl[aid]),
                                "gear": float(model.actuator_gear[aid, 0]),
                                "mass_matrix_row_kg_m2": mass_after[dof].astype(float).tolist(),
                            }
                        lift_dof = int(model.jnt_dofadr[id_by_name(model, mujoco.mjtObj.mjOBJ_JOINT, LIFT_JOINT)])
                        lift_aid = id_by_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m0_lift_carriage_servo")
                        control["m0_lift_carriage"]["actuator_force_n_post_step"] = float(data.actuator_force[lift_aid])
                        control["m0_lift_carriage"]["qfrc_actuator_n_post_step"] = float(data.qfrc_actuator[lift_dof])
                        control["m0_lift_carriage"]["qacc_m_s2_post_step"] = float(data.qacc[lift_dof])
                        control["m0_lift_carriage"]["qfrc_bias_n_post_step"] = float(data.qfrc_bias[lift_dof])
                        dynamics_audit = {
                            "joint_terms": joint_dynamics,
                            "qacc_vector": data.qacc.astype(float).tolist(),
                            "qfrc_actuator_vector": data.qfrc_actuator.astype(float).tolist(),
                            "qfrc_bias_vector": data.qfrc_bias.astype(float).tolist(),
                            "qfrc_constraint_vector": data.qfrc_constraint.astype(float).tolist(),
                            "qfrc_passive_vector": data.qfrc_passive.astype(float).tolist(),
                            "qfrc_applied_vector": data.qfrc_applied.astype(float).tolist(),
                            "mass_matrix": mass_after.astype(float).tolist(),
                            "equation_residual_assuming_minus_bias_l2": float(np.linalg.norm(equation_residual_minus_bias)),
                            "equation_residual_assuming_plus_bias_l2": float(np.linalg.norm(equation_residual_plus_bias)),
                        }
                        contacts = contact_metrics(model, data) if include_object else None
                        hand_contact = bool(contacts and any(
                            c["object_contact"] and c["object_other_body"] in (NARROW_PAD, WIDE_PAD)
                            and c["normal_force_on_object_n"] > FORCE_BEARING_EPS_N
                            for c in contacts["contacts"]
                        ))
                        if hand_contact and first_object_contact_step is None:
                            first_object_contact_step = step
                            Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["first_contact"])
                        failures = check_safety(model, data, include_object, first_object_contact_step is not None)
                        obj_state = object_summary(model, data) if include_object else None
                        driver_q = joint_qpos(model, data, RIGHT_DRIVER)
                        follower_q = joint_qpos(model, data, RIGHT_FOLLOWER)
                        row = {
                            "step": step,
                            "time_s": float(data.time),
                            "phase": phase,
                            "aperture_target_ratio": float(aperture),
                            "driver_qpos_rad": driver_q,
                            "follower_qpos_rad": follower_q,
                            "mimic_residual_rad": driver_q + follower_q,
                            "driver_qvel_rad_s": joint_qvel(model, data, RIGHT_DRIVER),
                            "follower_qvel_rad_s": joint_qvel(model, data, RIGHT_FOLLOWER),
                            "driver_qacc_rad_s2": joint_qacc(model, data, RIGHT_DRIVER),
                            "follower_qacc_rad_s2": joint_qacc(model, data, RIGHT_FOLLOWER),
                            "carriage_qpos_m": joint_qpos(model, data, LIFT_JOINT),
                            "carriage_qvel_m_s": joint_qvel(model, data, LIFT_JOINT),
                            "control": control,
                            "post_step_dynamics_audit": dynamics_audit,
                            "object_position_world_m": obj_state["position_world_m"] if obj_state else None,
                            "object_linear_speed_m_s": obj_state["linear_velocity_bound_m_s"] if obj_state else None,
                            "contact_family_metrics": contacts["family"] if contacts else None,
                            "table_support_force_world_n": contacts["table_support_force_world_sum_n"] if contacts else None,
                            "hand_support_force_world_n": contacts["hand_support_force_world_sum_n"] if contacts else None,
                            "contacts": contacts["contacts"] if contacts else [],
                            "safety_failures": failures,
                            "active_rollout_object_qpos_writes": 0,
                        }
                        write_jsonl_line(trace_stream, row)
                        if pose_stream is not None and obj_state is not None:
                            write_jsonl_line(pose_stream, {"step": step, "time_s": float(data.time), "phase": phase,
                                                           **obj_state, "object_qpos_write": False})
                        if contact_stream is not None and contacts is not None:
                            write_jsonl_line(contact_stream, {"step": step, "time_s": float(data.time), "phase": phase,
                                                             **contacts})
                        phase_rows.append(row)
                        if phase == "BILATERAL_CONTACT_HOLD":
                            hold_samples.append(row)
                        if phase == "LOAD_TRANSFER_PROBE_1MM":
                            probe_rows.append(row)
                        if obj_state is not None and obj_state["position_world_m"][2] > max_object_z:
                            max_object_z = float(obj_state["position_world_m"][2])
                            max_object_z_frame = render_frame(renderer, model, data, cam)
                        if step == phase_start_step + 1 or step % 25 == 0:
                            writer.append_data(render_frame(renderer, model, data, cam))
                        if failures:
                            safety_failure = {"step": step, "time_s": float(data.time), "phase": phase,
                                              "failures": failures}
                            break
                    summary = summarize_phase(phase_rows, include_object)
                    phase_summaries.append({"phase": phase, "step_start": phase_start_step + 1,
                                            "step_end": step, "summary": summary})
                    if phase == "BILATERAL_CONTACT_HOLD" and phase_rows:
                        Image.fromarray(render_frame(renderer, model, data, cam)).save(image_paths["hold_end"])
                    if safety_failure:
                        break
                if max_object_z_frame is not None and include_object:
                    Image.fromarray(max_object_z_frame).save(image_paths["max_lift"])
            finally:
                if pose_stream is not None:
                    pose_stream.close()
                if contact_stream is not None:
                    contact_stream.close()
    finally:
        renderer.close()
        writer.close()

    hold_gate = bilateral_hold_gate(hold_samples) if include_object else None
    return {
        "status": "FAIL_SAFETY_GATE" if safety_failure else "FINISHED_FINITE",
        "mj_step_calls": step - start_step,
        "simulation_time_s": float(data.time),
        "initial_jaw_initialization_writes_before_rollout": initial_jaw_writes,
        "active_rollout_object_qpos_writes": 0,
        "first_object_contact_step": first_object_contact_step,
        "first_object_contact_time_s": first_object_contact_step * dt if first_object_contact_step else None,
        "phase_summaries": phase_summaries,
        "hold_gate": hold_gate,
        "safety_failure": safety_failure,
        "probe_rows": probe_rows,
        "video": video_path.name,
        "video_fallback_reason": video_error,
        "object_summary_final": object_summary(model, data) if include_object else None,
    }


def summarize_phase(rows: list[dict[str, Any]], include_object: bool) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    result = {
        "samples": len(rows),
        "time_start_s": rows[0]["time_s"],
        "time_end_s": rows[-1]["time_s"],
        "max_abs_mimic_residual_rad": max(abs(float(row["mimic_residual_rad"])) for row in rows),
        "max_abs_jaw_qvel_rad_s": max(max(abs(float(row["driver_qvel_rad_s"])), abs(float(row["follower_qvel_rad_s"]))) for row in rows),
        "max_abs_jaw_qacc_rad_s2": max(max(abs(float(row["driver_qacc_rad_s2"])), abs(float(row["follower_qacc_rad_s2"]))) for row in rows),
        "driver_force_cap_hit_steps": sum(abs(float(row["control"][RIGHT_DRIVER]["applied_torque_nm"])) >= M0_EFFORT_CAP_NM[RIGHT_DRIVER] * 0.999 for row in rows),
        "follower_force_cap_hit_steps": sum(abs(float(row["control"][RIGHT_FOLLOWER]["applied_torque_nm"])) >= M0_EFFORT_CAP_NM[RIGHT_FOLLOWER] * 0.999 for row in rows),
        "safety_failure_count": sum(bool(row["safety_failures"]) for row in rows),
    }
    if include_object:
        bilateral = []
        table_fz = []
        hand_fz = []
        max_pen = 0.0
        for row in rows:
            families = row.get("contact_family_metrics") or {}
            bilateral.append(
                families.get(NARROW_PAD, {}).get("normal_force_sum_n", 0.0) > FORCE_BEARING_EPS_N
                and families.get(WIDE_PAD, {}).get("normal_force_sum_n", 0.0) > FORCE_BEARING_EPS_N
            )
            contacts = row.get("contacts", [])
            for c in contacts:
                if not c.get("object_contact"):
                    continue
                if c.get("object_other_body") == "world":
                    table_fz.append(c["force_on_object_world_n"][2])
                elif str(c.get("object_other_body", "")).startswith("R_hand_") and "loop" not in str(c.get("object_other_body", "")).lower():
                    hand_fz.append(c["force_on_object_world_n"][2])
                max_pen = max(max_pen, float(c.get("penetration_m", 0.0)))
        result.update({
            "bilateral_force_bearing_fraction": float(np.mean(bilateral)) if bilateral else 0.0,
            "mean_table_vertical_support_n": float(np.mean(table_fz)) if table_fz else 0.0,
            "mean_hand_vertical_support_n": float(np.mean(hand_fz)) if hand_fz else 0.0,
            "max_object_contact_penetration_m": max_pen,
            "object_z_start_m": rows[0].get("object_position_world_m", [None, None, None])[2]
            if "object_position_world_m" in rows[0] else None,
            "object_z_end_m": rows[-1].get("object_position_world_m", [None, None, None])[2]
            if "object_position_world_m" in rows[-1] else None,
        })
    return result


def bilateral_hold_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pass": False, "status": "NOT_RUN", "samples": 0}
    bilateral = []
    for row in rows:
        families = row.get("contact_family_metrics") or {}
        bilateral.append(
            families.get(NARROW_PAD, {}).get("normal_force_sum_n", 0.0) > FORCE_BEARING_EPS_N
            and families.get(WIDE_PAD, {}).get("normal_force_sum_n", 0.0) > FORCE_BEARING_EPS_N
        )
    fraction = float(np.mean(bilateral))
    return {"pass": bool(fraction >= 0.95), "status": "PASS" if fraction >= 0.95 else "FAIL",
            "samples": len(rows), "duration_s": len(rows) * TIMESTEP_S,
            "bilateral_force_bearing_fraction": fraction,
            "required_fraction": 0.95,
            "threshold_n_per_pad_family": FORCE_BEARING_EPS_N}


def load_transfer_gate(rows: list[dict[str, Any]], mass_kg: float, gravity: float) -> dict[str, Any]:
    if not rows:
        return {"pass": False, "status": "NOT_RUN", "samples": 0}
    metrics = summarize_phase(rows, True)
    weight = mass_kg * gravity
    table_z = float(metrics["mean_table_vertical_support_n"])
    hand_z = float(metrics["mean_hand_vertical_support_n"])
    final_pose = rows[-1].get("object_position_world_m")
    first_pose = rows[0].get("object_position_world_m")
    rise = (float(final_pose[2] - first_pose[2]) if final_pose and first_pose else 0.0)
    separated_frames = sum(
        not any(c.get("object_contact") and c.get("object_other_body") == "world" for c in row.get("contacts", []))
        for row in rows
    )
    bilateral = bilateral_hold_gate(rows)
    passed = (table_z <= 0.10 * weight and hand_z >= 0.90 * weight and
              separated_frames >= 0.90 * len(rows) and rise >= 0.0005 and bilateral["pass"])
    return {
        "pass": passed,
        "status": "PASS" if passed else "FAIL",
        "object_weight_n": weight,
        "mean_table_support_n": table_z,
        "mean_hand_vertical_support_n": hand_z,
        "table_separated_fraction": separated_frames / len(rows),
        "object_rise_m": rise,
        "bilateral_contact": bilateral,
        "diagnostic_screen": "table support <=10% weight, hand upward support >=90% weight, >=90% table-separated frames, >=0.5mm rise",
    }


def contact_object_z(model: mujoco.MjModel, data: mujoco.MjData) -> float | None:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY))
    return float(data.xpos[body_id][2]) if body_id >= 0 else None


def first_static_gate_failure(preflight: dict[str, Any]) -> str | None:
    if preflight["status"] != "PASS":
        return "M0 static fixture preflight failed"
    return None


def write_report(output: Path, result: dict[str, Any]) -> None:
    status = str(result.get("status", "UNKNOWN"))
    gates = result.get("gates", {})
    lines = [
        "# X2 OmniPicker SIMULATION_ONLY M0 接触管线诊断",
        "",
        f"状态：**{status}**",
        "",
        "本报告复用 P1 FreeCAD STEP 测量、P2 销轴/轴承候选和 P3 已失败的闭环假设。P3 不重跑。本轮只验证保留官方右侧 OmniPicker 树几何、单一公开 aperture 命令和已接受 M0 执行器参数的隔离接触管线。",
        "",
        "## 证据边界",
        "",
        "- 机制标签：`SIMULATION_ONLY_M0_1DOF_CONTACT_PIPELINE_DIAGNOSTIC`。窄/宽中间连杆和 loop 仍采用官方 URDF 的 fixed/tree 表示。",
        "- 右夹爪使用内部 driver/follower 伺服，follower target 从 driver 的源 mimic 比率推导；它不是物理被动闭环。",
        "- 固定夹爪安装、垂直 carriage 和钢质小圆柱均为本诊断夹具；圆柱不是 canonical multipart bottle。",
        "- 不使用 weld、mocap、物体 qpos 写入、额外碰撞排除、左手辅助或摩擦调参。",
        "",
        "## 分阶段结果",
        "",
        f"- P1 CAD：`{result.get('stage_p1', {}).get('status', 'NOT_RECORDED')}`。",
        f"- P2 机构：`{result.get('stage_p2', {}).get('status', 'NOT_RECORDED')}`；唯一闭环拓扑仍缺制造商语义。",
        f"- P3 源忠实闭环：`{result.get('stage_p3', {}).get('status', 'NOT_RECORDED')}`；复用既有报告，未重跑。",
        f"- P4 静态几何：`{gates.get('static_geometry_preflight', {}).get('status', 'NOT_RUN')}`。",
        f"- P4 空载孤立 M0 开合：`{gates.get('no_contact_open_close', {}).get('status', 'NOT_RUN')}`。",
        f"- P4 自由圆柱双侧接触/保持：`{gates.get('bilateral_contact_hold', {}).get('status', 'NOT_RUN')}`。",
        f"- P4 1 mm 负载转移探测：`{gates.get('load_transfer_or_short_lift', {}).get('status', 'NOT_RUN')}`。",
        "- P5 报告/JSON/迹线/图片/视频：本目录中的实际文件；未声称生产集成。",
        "",
        "## 结果摘要",
        "",
        f"- 结果 JSON：`result.json`。",
        f"- 运行身份：`runtime_identity.json`。",
        f"- 编译模型审计：`compiled_model_audit.json`。",
        f"- 静态预检：`static_preflight.json`。",
        f"- 复现命令：`experiment_commands.txt`。",
        "- `mj_step` 迹线使用 JSONL；对象 qpos 在 rollout 开始后写入计数固定为 0。",
        "- 若双侧保持或负载转移门失败，后续更大升程不会执行。",
        "",
        "## 复现",
        "",
        "```bash",
        (output / "experiment_commands.txt").read_text(encoding="utf-8").strip() if (output / "experiment_commands.txt").exists() else "尚未写入命令。",
        "```",
        "",
        "该诊断结果不能证明 X2 OmniPicker 被动机构或真实硬件抓取通过。下一步源忠实建模仍需 AgiBot 确认实际销孔运动副、loop 分支、齿轮/弹簧关系及活动范围。",
    ]
    path = output / "REPORT.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    if not output.is_dir() or not (output / ".robotsim-output-reservation.json").is_file():
        raise RuntimeError("--output must be a fresh directory created by path_governance")
    remaining = [p for p in output.iterdir() if p.name != ".robotsim-output-reservation.json"]
    if remaining:
        raise FileExistsError(f"Refusing to overwrite evidence in {output}: {remaining[0]}")

    m0_source_sha = M0_BASELINE_REPORT_SHA256
    identity = {
        "classification": "SIMULATION_ONLY_M0_1DOF_CONTACT_PIPELINE_DIAGNOSTIC",
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(),
        "numpy": np.__version__,
        "imageio": importlib.metadata.version("imageio"),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "egl"),
        "timestep_s": TIMESTEP_S,
        "integrator": "Euler",
        "solver": "Newton",
        "solver_iterations": 100,
        "source_repo": X2_REPO,
        "source_commit": X2_PIN,
        "source_urdf": str(X2_URDF),
        "source_urdf_sha256": sha256(X2_URDF),
        "runner_script_sha256": sha256(Path(__file__).resolve()),
        "m0_reference_branch": M0_BASELINE_BRANCH,
        "m0_reference_commit": M0_BASELINE_COMMIT,
        "m0_reference_report_sha256": m0_source_sha,
        "new_collision_exclusions": 0,
        "right_jaw_equality": False,
        "mimic_follower_user_command": False,
        "object_qpos_writes_after_rollout_start": 0,
        "object_weld_or_mocap": False,
        "left_hand_assistance": False,
    }
    identity_path = output / "runtime_identity.json"
    write_json(identity_path, identity)
    command_path = output / "experiment_commands.txt"
    reproduction = (
        f"MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python "
        f"{Path(__file__).resolve()} --output {output.resolve()}"
    )
    with command_path.open("x", encoding="utf-8") as stream:
        stream.write(reproduction + "\n")

    result: dict[str, Any] = {
        "attempt_id": "issue46-m0-contact-pipeline-20261018-run10",
        "status": "RUNNING",
        "stage_p1": {"status": "PASS_REUSED", "evidence": "omnipicker-step-cad-review report and source hashes"},
        "stage_p2": {"status": "PARTIAL_TOPOLOGY_UNDERDETERMINED", "evidence": "STEP pin/bearing report; exact physical joint semantics absent"},
        "stage_p3": {"status": "FAIL_REUSED_NO_RERUN", "evidence": "x2-omnipicker-closed-loop-dynamics-20261017-run01/REPORT.md; source-mesh hypothesis failed continuity, collisions, tracking, and limits"},
        "sim_only_m0_identity": identity,
        "object": {
        "kind": "10 mm diameter x 150 mm steel-density free-jointed diagnostic rod; SIMULATION_ONLY, not canonical multipart bottle",
            "radius_m": OBJECT_RADIUS_M,
            "height_m": 2.0 * OBJECT_HALF_HEIGHT_M,
            "density_assumption_kg_m3": OBJECT_DENSITY_ASSUMPTION_KG_M3,
            "mass_kg": OBJECT_MASS_KG,
            "friction": OBJECT_FRICTION,
            "initial_pose_from_mjcf": True,
            "qpos_writes_after_rollout_start": 0,
        },
        "fixture": {
            "worldbody": "isolated right OmniPicker with source R_omnipicker_joint transform nested under a SIMULATION_ONLY vertical carriage",
            "carriage_mass_kg": CARriage_MASS_KG,
            "carriage_diaginertia_kg_m2": CARriage_DIAG_INERTIA_KG_M2,
            "carriage_force_pd_kp_n_per_m": CARriage_KP_N_M,
            "carriage_force_pd_kv_ns_per_m": CARriage_KV_NS_M,
            "carriage_force_cap_n": CARriage_FORCE_CAP_N,
            "carriage_max_travel_m": MAX_SHORT_LIFT_M,
            "carriage_numeric_limit_tolerance_m": FIXTURE_LIMIT_NUMERICAL_TOLERANCE_M,
            "object_relative_to_base_world_m": OBJECT_RELATIVE_TO_BASE_WORLD_M.tolist(),
            "fixture_raise_from_previous_candidate_m": FIXTURE_RAISE_M,
            "rod_bottom_world_z_m": TABLE_TOP_Z_M,
            "rod_top_world_z_m": TABLE_TOP_Z_M + 2.0 * OBJECT_HALF_HEIGHT_M,
            "table_top_z_world_m": TABLE_TOP_Z_M,
            "table_half_size_m": TABLE_HALF_SIZE_M,
            "new_collision_exclusions": 0,
        },
        "controller": {
            "public_command": "aperture_ratio in [0,1]",
            "aperture_map": "q_driver=-(0.001+0.998*a); q_follower=-q_driver",
            "object_contact_aperture_ratio": CONTACT_APERTURE_RATIO,
            "endpoint_margin_rad": ENDPOINT_MARGIN_RAD,
            "motion_reference": "minimum jerk",
            "duration_s": MOTION_DURATION_S,
            "driver_kp_kd_effort_cap": [M0_KP_NM_PER_RAD[RIGHT_DRIVER], M0_KD_NMS_PER_RAD[RIGHT_DRIVER], M0_EFFORT_CAP_NM[RIGHT_DRIVER]],
            "follower_kp_kd_effort_cap": [M0_KP_NM_PER_RAD[RIGHT_FOLLOWER], M0_KD_NMS_PER_RAD[RIGHT_FOLLOWER], M0_EFFORT_CAP_NM[RIGHT_FOLLOWER]],
            "gain_effort_provenance": "reused exact accepted M0 controller parameters; no tuning",
            "carriage_controller": "SIMULATION_ONLY bounded motor force; PD + qfrc_bias + reference-acceleration feedforward",
        },
        "gates": {},
        "outputs": {},
        "claim_boundary": "A result here validates only the SIMULATION_ONLY M0 source-tree contact pipeline. It does not validate the OmniPicker's passive mechanical loop or X2 hardware grasp capability.",
    }

    no_object_model, no_object_info, no_object_xml = compile_model(False, output)
    contact_model, contact_info, contact_xml = compile_model(True, output)
    for name, xml in (("m0_isolated_no_object.xml", no_object_xml), ("m0_isolated_contact.xml", contact_xml)):
        if name in {"m0_isolated_no_object.xml", "m0_isolated_contact.xml"}:
            result.setdefault("model_xml_hashes", {})[name] = hashlib.sha256(xml.encode("utf-8")).hexdigest()

    model_metadata = {
        "no_object": {"nq": no_object_model.nq, "nv": no_object_model.nv, "nbody": no_object_model.nbody,
                      "ngeom": no_object_model.ngeom, "njnt": no_object_model.njnt, "nu": no_object_model.nu, "neq": no_object_model.neq},
        "contact": {"nq": contact_model.nq, "nv": contact_model.nv, "nbody": contact_model.nbody,
                    "ngeom": contact_model.ngeom, "njnt": contact_model.njnt, "nu": contact_model.nu, "neq": contact_model.neq},
        "source_joint_ranges_rad": source_limits(X2_URDF),
        "mount_transform": contact_info["base_world"],
        "geometry": contact_info["fixture_geometry"],
        "collision_filters_added": 0,
    }
    write_json(output / "compiled_model_audit.json", model_metadata)

    preflight = static_preflight(contact_model, output)
    write_json(output / "static_preflight.json", preflight)
    result["gates"]["static_geometry_preflight"] = {k: v for k, v in preflight.items() if k != "sweep_samples"}
    if args.preflight_only or first_static_gate_failure(preflight):
        result["status"] = "FAIL_STATIC_PREFLIGHT" if preflight["status"] != "PASS" else "STATIC_PREFLIGHT_PASS_ONLY"
        result["gates"]["no_contact_open_close"] = {"status": "NOT_RUN" if args.preflight_only else "NOT_RUN_STATIC_GATE_FAILED"}
        result["gates"]["bilateral_contact_hold"] = {"status": "NOT_RUN"}
        result["gates"]["load_transfer_or_short_lift"] = {"status": "NOT_RUN"}
        write_json(output / "result.json", result)
        write_report(output, result)
        return 0 if preflight["status"] == "PASS" else 2

    no_contact_data, no_contact_assignments = initialize(no_object_model)
    no_contact_sequence = [
        ("OPEN_HOLD", 0.25, 1.0, 1.0, 0.0),
        ("CLOSE_TO_OBJECT_APERTURE", MOTION_DURATION_S, 1.0, CONTACT_APERTURE_RATIO, 0.0),
        ("HOLD_AT_OBJECT_WIDTH", 0.25, CONTACT_APERTURE_RATIO, CONTACT_APERTURE_RATIO, 0.0),
        ("REOPEN", MOTION_DURATION_S, CONTACT_APERTURE_RATIO, 1.0, 0.0),
        ("OPEN_HOLD_FINAL", 0.25, 1.0, 1.0, 0.0),
    ]
    no_contact_result = run_sequence(no_object_model, no_contact_data, include_object=False,
                                     output=output, prefix="isolated_m0_open_close", sequence=no_contact_sequence,
                                     initial_jaw_writes=no_contact_assignments)
    result["gates"]["no_contact_open_close"] = no_contact_result
    write_json(output / "no_contact_result.json", no_contact_result)
    if no_contact_result["status"] != "FINISHED_FINITE" or no_contact_result.get("safety_failure"):
        result["status"] = "FAIL_NO_CONTACT_MOTION"
        result["gates"]["bilateral_contact_hold"] = {"status": "NOT_RUN_NO_CONTACT_GATE_FAILED"}
        result["gates"]["load_transfer_or_short_lift"] = {"status": "NOT_RUN"}
        write_json(output / "result.json", result)
        write_report(output, result)
        return 2

    contact_data, contact_assignments = initialize(contact_model)
    contact_sequence = [
        ("SETTLE_OBJECT_ON_TABLE", 0.5, 1.0, 1.0, 0.0),
        ("OPEN_HOLD", 0.25, 1.0, 1.0, 0.0),
        ("CLOSE_TO_CONTACT", MOTION_DURATION_S, 1.0, CONTACT_APERTURE_RATIO, 0.0),
        ("BILATERAL_CONTACT_HOLD", 1.0, CONTACT_APERTURE_RATIO, CONTACT_APERTURE_RATIO, 0.0),
    ]
    contact_result = run_sequence(contact_model, contact_data, include_object=True,
                                  output=output, prefix="isolated_m0_contact_hold", sequence=contact_sequence,
                                  initial_jaw_writes=contact_assignments)
    result["gates"]["bilateral_contact_hold"] = contact_result
    write_json(output / "contact_hold_result.json", contact_result)
    hold_gate = contact_result.get("hold_gate") or {"pass": False, "status": "NOT_RUN"}
    if contact_result["status"] != "FINISHED_FINITE" or not hold_gate.get("pass", False):
        result["status"] = "FAIL_BILATERAL_CONTACT_OR_HOLD"
        result["gates"]["load_transfer_or_short_lift"] = {"status": "NOT_RUN_CONTACT_GATE_FAILED"}
        write_json(output / "result.json", result)
        write_report(output, result)
        return 2

    # The first 1 mm carriage movement is the load-transfer probe. A 10 mm lift
    # is only attempted if table support is genuinely replaced by hand contact.
    transfer_model = contact_model
    transfer_data = contact_data
    transfer_result = run_sequence(transfer_model, transfer_data, include_object=True,
                                    output=output, prefix="isolated_m0_load_transfer",
                                    sequence=[("LOAD_TRANSFER_PROBE_1MM", 0.5, CONTACT_APERTURE_RATIO,
                                               CONTACT_APERTURE_RATIO, 0.001)],
                                    initial_jaw_writes=0, continue_data=True)
    transfer_gate = load_transfer_gate(transfer_result.get("probe_rows", []), OBJECT_MASS_KG, GRAVITY_M_S2)
    transfer_result["load_transfer_gate"] = transfer_gate
    result["gates"]["load_transfer_or_short_lift"] = transfer_result
    write_json(output / "load_transfer_result.json", transfer_result)
    if not transfer_gate["pass"]:
        result["status"] = "FAIL_SIMULATION_ONLY_M0_LOAD_TRANSFER"
        write_json(output / "result.json", result)
        write_report(output, result)
        return 2

    lift_result = run_sequence(transfer_model, transfer_data, include_object=True,
                               output=output, prefix="isolated_m0_short_lift",
                               sequence=[("SHORT_LIFT_10MM", 1.0, CONTACT_APERTURE_RATIO,
                                          CONTACT_APERTURE_RATIO, 0.010),
                                         ("SUSPENDED_HOLD_1S", 1.0, CONTACT_APERTURE_RATIO,
                                          CONTACT_APERTURE_RATIO, 0.010)],
                               initial_jaw_writes=0, continue_data=True)
    result["gates"]["short_lift_10mm_and_hold"] = lift_result
    write_json(output / "short_lift_result.json", lift_result)
    result["status"] = "PASS_SIMULATION_ONLY_M0_SHORT_LIFT" if lift_result["status"] == "FINISHED_FINITE" else "FAIL_SIMULATION_ONLY_M0_SHORT_LIFT"
    write_json(output / "result.json", result)
    write_report(output, result)
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        output_arg = next((sys.argv[i + 1] for i, value in enumerate(sys.argv[:-1]) if value == "--output"), None)
        if output_arg:
            output = Path(output_arg)
            if output.is_dir() and not (output / "unhandled_error.json").exists():
                write_json(output / "unhandled_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise
