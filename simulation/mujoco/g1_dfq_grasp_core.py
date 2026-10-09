#!/usr/bin/env python3
"""One-shot MuJoCo 3.3.6 DFQ right-hand short-grasp checkpoint."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "simulation/mujoco/fixtures/g1_dfq_m0_c2"
ROOT = Path(os.environ.get(
    "ROBOTSIM_EVIDENCE_ROOT",
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-physical-grasp-priority-p0-20261009",
))
SCRATCH = Path(__file__).resolve().parent
REPO = PROJECT_ROOT
TARGET_BRANCH = "issue/43-mujoco-bottle-pick-place"
SCENE = Path(os.environ.get("ROBOTSIM_G1_DFQ_SCENE", FIXTURE / "g1_inspire_dfq_accepted_scene.xml"))
ROBOT_XML = Path(os.environ.get(
    "ROBOTSIM_G1_DFQ_ROBOT_XML", FIXTURE / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
))
UPSTREAM = Path(os.environ.get("ROBOTSIM_HUMANOID_VLA_DIR", "/tmp/robotsim-g1-dfq/upstream/humanoid_vla"))
UNITREE_ROS = Path(os.environ.get("ROBOTSIM_UNITREE_ROS_DIR", "/tmp/robotsim-g1-dfq/upstream/unitree_ros"))
UNITREE_MUJOCO = Path(os.environ.get("ROBOTSIM_UNITREE_MUJOCO_DIR", "/tmp/robotsim-g1-dfq/upstream/unitree_mujoco"))
URDF = UNITREE_ROS / "robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
WRENCH_AUDIT_JSON = FIXTURE / "candidate_c2_robust_wrench.json"
WRENCH_SOURCE_INPUT_JSON = FIXTURE / "canonical_ready_wrench_input.json"
EXPECTED_WRENCH_MODEL_SHA256 = "ebdf74fc4ac1fb9bf0e9c5d7be62654f03fd5ca51c2968085b08356ae96c2d67"

UNITREE_ROS_SHA = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
UNITREE_MUJOCO_SHA = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
HUMANOID_VLA_SHA = "3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
MUJOCO_VERSION = "3.3.6"
DT = 0.00025
LOCK_SOLREF = "0.0005 1"
LOCK_SOLIMP = "0.99 0.999 0.001"
LIMIT_SOLREF = "0.00025 1"
LIMIT_SOLIMP = "0.999 0.9999 0.001"
HAND_KP, HAND_KV, HAND_MAX_TORQUE, HAND_DAMPING = 1.0, 0.1, 0.20, 0.02
OPEN_TARGET_SLEW_RAD_S = 0.075
CLOSE_TARGET_SLEW_RAD_S = 0.060
MAX_HAND_QVEL_RAD_S = 0.5
MIMIC_DIAGNOSTIC_TOLERANCE_RAD = 0.003
MIMIC_MANIPULATION_LIMIT_RAD = 0.010
JOINT_LIMIT_TOLERANCE_RAD = 0.0001
MAX_CONTACT_FORCE_N = 30.0
MIN_LIFT_M, LIFT_TARGET_M = 0.030, 0.040
HOLD_SECONDS, LIFT_TIMEOUT_SECONDS = 1.0, 2.5
RELEASE_TIMEOUT_SECONDS = 3.0
RELEASE_SETTLE_DWELL_SECONDS = 0.25
RELEASE_OPEN_TARGET_TOLERANCE_RAD = 0.03
APPROACH_SECONDS = 1.0
PREGRASP_SECONDS = 0.25
SETTLE_MAX_SECONDS = 4.0
SETTLE_DWELL_SECONDS = 0.25
SETTLE_MAX_LINEAR_SPEED_M_S = 0.010
SETTLE_MAX_ANGULAR_SPEED_RAD_S = 0.050
CONTACT_HOLD_FRACTION = 0.80
CONTACT_TRIGGER_FORCE_N = 0.01
CONTACT_CONFIRM_SAMPLES = 20
CONTACT_PRELOAD_MAX_RAD = 0.004
CONTACT_PRELOAD_GATE_TOLERANCE_RAD = 1e-5
CONTACT_LOSS_CONFIRM_SAMPLES = 20
LIFT_ARMING_DWELL_SECONDS = 0.020
LIFT_ARMING_TIMEOUT_SECONDS = 2.0
CONTACT_VALIDITY_WINDOW_S = 0.005
CONTACT_VALIDITY_WINDOW_SAMPLES = int(round(CONTACT_VALIDITY_WINDOW_S / DT))
CONTACT_VALIDITY_MAX_ZERO_S = 0.00125
CONTACT_VALIDITY_MAX_ZERO_SAMPLES = int(round(CONTACT_VALIDITY_MAX_ZERO_S / DT))
CONTACT_VALIDITY_MIN_DUTY = 0.75
CONTACT_VALIDITY_MIN_IMPULSE_NS = (
    CONTACT_TRIGGER_FORCE_N * DT
    * (CONTACT_VALIDITY_WINDOW_SAMPLES - CONTACT_VALIDITY_MAX_ZERO_SAMPLES)
)
SIM_ONLY_MAX_DRIVER_TORQUE_NM = 0.531
SIM_ONLY_FEEDFORWARD_RAMP_S = 1.0
LOAD_CONTACT_REALLOCATION = {
    "status": "ONE_EVIDENCE_DERIVED_CORRECTION",
    "source_run_id": "g1-dfq-physical-1791547522380405273-28009",
    "source_snapshot_time_s": 23.66000000002196,
    "activation": "load transfer only; C2 command trajectory and close phase unchanged",
    "contact_bodies": [
        "R_thumb_distal", "R_index_proximal", "R_index_intermediate",
        "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate",
    ],
    "friction_coefficient": 1.4,
    "friction_cone_utilization_cap": 0.9,
    "total_normal_force_cap_n": 6.0,
    "single_contact_resultant_cap_n": 5.0,
    "predicted_total_normal_force_n": 4.544179527087802,
    "predicted_peak_contact_resultant_n": 3.4420974791736514,
    "predicted_peak_required_channel_effort_including_bias_nm": 0.3661180049963703,
    "driver_feedforward_nm": {
        "pinky_proximal": 0.0,
        "ring_proximal": 0.0,
        "middle_proximal": 0.028692187841682,
        "index_proximal": 0.05153331018634697,
        "thumb_proximal_pitch": 0.3670486752117906,
        "thumb_proximal_yaw": -0.18615235731326174,
    },
    "allocation_by_contact": {
        "R_thumb_distal": {
            "normal_n": 2.1450109657652026,
            "tangent_local_n": [-0.1117119188793386, 2.689699511109175],
        },
        "R_index_proximal": {
            "normal_n": 0.6513595815530454,
            "tangent_local_n": [-0.3868814643681141, -0.7238043107932789],
        },
        "R_index_intermediate": {
            "normal_n": 1.229539980428406,
            "tangent_local_n": [0.15185015093054433, -1.541760455784016],
        },
        "R_middle_intermediate": {
            "normal_n": 0.5182689993411479,
            "tangent_local_n": [0.1315699604962905, -0.6364353758157268],
        },
        "R_ring_intermediate": {"normal_n": 0.0, "tangent_local_n": [0.0, 0.0]},
        "R_pinky_intermediate": {"normal_n": 0.0, "tangent_local_n": [0.0, 0.0]},
    },
    "physics_steps": 0,
    "bottle_qpos_modified": False,
}
SIM_ONLY_VELOCITY_GUARD_START_RAD_S = 0.45
SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S = 0.48
SIM_ONLY_VELOCITY_GUARD_BISECTION_STEPS = 12
SIM_ONLY_LOAD_DWELL_S = 0.050
SIM_ONLY_SUPPORT_FRACTION_MIN = 0.85
SIM_ONLY_TABLE_REMAINDER_FRACTION_MIN = 0.02
SIM_ONLY_FORCE_BALANCE_TOLERANCE_FRACTION = 0.05
SIM_ONLY_HORIZONTAL_FORCE_TOLERANCE_FRACTION = 0.10
SIM_ONLY_WRENCH_TORQUE_TOLERANCE_NM = 0.08
SIM_ONLY_MAX_FRICTION_UTILIZATION = 0.90
LOAD_BUILD_SAFETY_FACTOR = 1.0
LOAD_BUILD_PRELOAD_MAX_RAD = 0.10
LOAD_BUILD_PRELOAD_CAP_RATE_RAD_S = 0.024
LOAD_BUILD_MAX_SECONDS = 4.0
LOAD_BUILD_STABLE_DWELL_S = 0.050
LOAD_BUILD_MAX_ACTUATOR_EFFORT_NM = SIM_ONLY_MAX_DRIVER_TORQUE_NM
LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N = 5.0
LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N = 10.0
LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N = 6.0
LOAD_BUILD_WRIST_DRIFT_GATE_TOLERANCE_M = 0.000001
LOAD_BUILD_MAX_BOTTLE_RISE_M = 0.0005
LIVE_ALLOCATION_UPDATE_STEPS = 20
LIVE_ALLOCATION_EFFORT_HEADROOM_FRACTION = 0.95
LIVE_ALLOCATION_NORMAL_HEADROOM_FRACTION = 0.90
LIVE_ALLOCATION_CONTACT_HEADROOM_FRACTION = 0.90
LIVE_ALLOCATION_TOTAL_CONTACT_HEADROOM_FRACTION = 0.90
LIVE_ALLOCATION_FRICTION_UTILIZATION = 0.85
LIVE_SUPPORT_TARGET_RATE_N_S = 12.0
LIVE_ALLOCATION_HANDOFF_STEPS = 1
LIVE_TRANSFER_STEP_HEIGHTS_M = (0.0001, 0.00025, 0.0005)
LIVE_TRANSFER_STEP_DWELL_S = 0.10
WRIST_YAW_CORRECTION_DEG = 60.0
PREGRASP_SITE_OFFSET_M = np.array([0.0, -0.160, 0.0])
APPROACH_SITE_OFFSET_M = np.array([0.0, -0.120, 0.0])
TRACE_PERIOD_S, VIDEO_PERIOD_S = 0.01, 0.05

sys.path.insert(0, str(SCRATCH))
import g1_inspire_hand as hand  # noqa: E402
import g1_dfq_grasp_gates as gates  # noqa: E402


class GateFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def element_id(model, kind, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise GateFailure(f"Missing model element: {name}")
    return value


def joint_qpos(model, data, name: str) -> float:
    jid = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def joint_qvel(model, data, name: str) -> float:
    jid = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qvel[int(model.jnt_dofadr[jid])])


def bottle_qpos_addresses(model) -> list[int]:
    jid = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    start = int(model.jnt_qposadr[jid])
    return list(range(start, start + 7))


ARM_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def arm_qpos_addresses(model) -> np.ndarray:
    return np.asarray([
        int(model.jnt_qposadr[element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in ARM_NAMES
    ], dtype=int)


def build_scene(output_xml: Path, ranges: dict, limits: dict, mimics: dict) -> dict:
    scene = ET.parse(SCENE).getroot()
    root = ET.parse(ROBOT_XML).getroot()
    base_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    robot_world = root.find("worldbody")
    if robot_world is None:
        raise GateFailure("Approved G1 XML has no worldbody")

    for section in ("statistic", "visual"):
        old = root.find(section)
        if old is not None:
            root.remove(old)
        new = scene.find(section)
        if new is not None:
            root.append(deepcopy(new))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    scene_asset = scene.find("asset")
    if scene_asset is not None:
        asset.extend(deepcopy(list(scene_asset)))
    scene_world = scene.find("worldbody")
    if scene_world is None:
        raise GateFailure("Accepted scene has no worldbody")
    robot_world.extend(deepcopy(list(scene_world)))
    root.set("model", "g1_inspire_dfq_short_grasp_mj336")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(DT))
    option.set("integrator", "implicitfast")
    option.set("iterations", "200")
    option.set("tolerance", "1e-10")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    mimic_count = 0
    follower_names = set(mimics)
    for eq in equality.findall("joint"):
        if eq.get("name", "").startswith("mimic_"):
            eq.set("solref", LOCK_SOLREF)
            eq.set("solimp", LOCK_SOLIMP)
            mimic_count += 1
    if mimic_count != 12:
        raise GateFailure(f"Expected 12 accepted mimic constraints, got {mimic_count}")
    ET.SubElement(equality, "weld", {
        "name": "scratch_fixed_pelvis", "body1": "pelvis",
        "solref": LOCK_SOLREF, "solimp": LOCK_SOLIMP,
    })

    active_names = set(ARM_NAMES)
    active_names |= {hand.channel_joint("R", suffix) for _, suffix in hand.CHANNELS}
    locked = []
    for jid in range(1, base_model.njnt):
        name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if not name or name == "floating_base_joint" or name in follower_names or name in active_names:
            continue
        q0 = float(base_model.qpos0[int(base_model.jnt_qposadr[jid])])
        ET.SubElement(equality, "joint", {
            "name": f"scratch_lock_{name}", "joint1": name,
            "polycoef": f"{q0:.17g} 0 0 0 0", "solref": LOCK_SOLREF, "solimp": LOCK_SOLIMP,
        })
        locked.append(name)

    for body in root.findall(".//body"):
        for joint in body.findall("joint"):
            name = joint.get("name", "")
            if name.startswith(("L_", "R_")) and any(
                digit in name for digit in ("thumb", "index", "middle", "ring", "pinky")
            ):
                joint.set("damping", str(HAND_DAMPING))
                joint.set("solreflimit", LIMIT_SOLREF)
                joint.set("solimplimit", LIMIT_SOLIMP)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for idx, pair in enumerate((
        ("L_hand_base_link", "L_thumb_proximal"),
        ("R_hand_base_link", "R_thumb_proximal"),
    )):
        ET.SubElement(contact, "exclude", {
            "name": f"scratch_initial_thumb_overlap_{idx}", "body1": pair[0], "body2": pair[1],
        })

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    if len(actuator) != 29:
        raise GateFailure(f"Expected existing 29 G1 motor actuators, got {len(actuator)}")
    hand_actuators = {}
    for channel, suffix in hand.CHANNELS:
        joint = hand.channel_joint("R", suffix)
        lo, hi = ranges[joint]["simulation_feasible_range_rad"]
        name = f"R_{channel}"
        ET.SubElement(actuator, "position", {
            "name": name, "joint": joint, "kp": str(HAND_KP), "kv": str(HAND_KV),
            "ctrlrange": f"{lo:.12g} {hi:.12g}",
            "forcerange": f"{-HAND_MAX_TORQUE} {HAND_MAX_TORQUE}", "forcelimited": "true",
        })
        hand_actuators[channel] = name

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    if model.nu != 35 or model.nmocap != 0:
        raise GateFailure(f"Unexpected control/mocap counts: nu={model.nu}, nmocap={model.nmocap}")
    object_body = element_id(model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
    object_joint = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
    weld_count = int(np.count_nonzero(model.eq_type == int(mujoco.mjtEq.mjEQ_WELD)))
    if weld_count != 1:
        raise GateFailure(f"Expected only fixed-pelvis weld, got {weld_count} welds")
    for eqid in range(model.neq):
        typ = int(model.eq_type[eqid])
        if typ == int(mujoco.mjtEq.mjEQ_WELD):
            if object_body in (int(model.eq_obj1id[eqid]), int(model.eq_obj2id[eqid])):
                raise GateFailure("Bottle is referenced by a weld")
        if typ == int(mujoco.mjtEq.mjEQ_JOINT):
            if object_joint in (int(model.eq_obj1id[eqid]), int(model.eq_obj2id[eqid])):
                raise GateFailure("Bottle free joint is referenced by an equality")
    return {
        "model": model, "right_hand_actuators": hand_actuators,
        "right_arm_joint_names": ARM_NAMES, "locked_joint_names": locked,
        "mimic_constraint_count": mimic_count, "fixed_base_weld_count": weld_count,
        "bottle_equality_reference_count": 0, "mocap_body_count": int(model.nmocap),
        "model_xml": output_xml,
    }


def plan_arm(model, data, site_id: int, target: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, dict]:
    qadr = arm_qpos_addresses(model)
    joint_ids = np.asarray([
        element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_NAMES
    ], dtype=int)
    dofs = np.asarray(model.jnt_dofadr[joint_ids], dtype=int)
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    saved = data.qpos[qadr].copy()
    data.qpos[qadr] = np.clip(np.zeros(7), lower, upper)
    pos_error = np.zeros(3)
    rot_error = np.zeros(3)
    converged = False
    for iteration in range(1200):
        mujoco.mj_forward(model, data)
        pos_error = target - data.site_xpos[site_id]
        current = data.site_xmat[site_id].reshape(3, 3)
        rot_error = 0.5 * sum(np.cross(current[:, axis], rotation[:, axis]) for axis in range(3))
        if np.linalg.norm(pos_error) <= 0.003 and np.linalg.norm(rot_error) <= 0.06:
            converged = True
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        weight, damping = 0.25, 0.05
        jac = np.vstack((jacp[:, dofs], jacr[:, dofs] * weight))
        error = np.r_[pos_error, rot_error * weight]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
        norm = float(np.linalg.norm(dq))
        if norm > 0.02:
            dq *= 0.02 / norm
        data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lower, upper)
    planned = data.qpos[qadr].copy()
    mujoco.mj_forward(model, data)
    achieved = data.site_xpos[site_id].copy()
    achieved_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    data.qpos[qadr] = saved
    mujoco.mj_forward(model, data)
    return planned, {
        "converged": converged, "iterations": iteration + 1,
        "requested_site_m": target.tolist(), "achieved_site_m": achieved.tolist(),
        "position_error_m": float(np.linalg.norm(target - achieved)),
        "rotation_error_rad": float(np.linalg.norm(rot_error)),
        "achieved_rotation": achieved_rotation.tolist(),
    }


def body_name_for_geom(model, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def bottle_contacts(model, data) -> list[dict]:
    bottle_geoms = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}
    result = []
    for cid in range(data.ncon):
        c = data.contact[cid]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if (n1 in bottle_geoms) == (n2 in bottle_geoms):
            continue
        bottle_geom = n1 if n1 in bottle_geoms else n2
        other_geom = n2 if n1 in bottle_geoms else n1
        body = body_name_for_geom(model, g2 if n1 in bottle_geoms else g1)
        side = ("right_hand" if body.startswith("R_") else "left_hand" if body.startswith("L_")
                else "right_arm" if body.startswith("right_") else "left_arm" if body.startswith("left_") else "other")
        digit = next((d for d in ("thumb", "index", "middle", "ring", "pinky") if d in body.lower()), None)
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, cid, wrench)
        frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
        force_world_in_geom2_direction = frame.T @ wrench[:3]
        force_on_bottle = (
            -force_world_in_geom2_direction if n1 in bottle_geoms
            else force_world_in_geom2_direction
        )
        contact_position = np.asarray(c.pos, dtype=float).tolist()
        contact_normal_basis = frame[0].tolist()
        result.append({
            "bottle_geom": bottle_geom, "other_geom": other_geom, "other_body": body,
            "side": side, "digit": digit, "distance_m": float(c.dist),
            "normal_force_n": float(np.linalg.norm(wrench[:3])),
            "true_normal_force_n": float(abs(wrench[0])),
            "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
            "force_on_bottle_world_n": force_on_bottle.tolist(),
            "contact_position_world_m": contact_position,
            "contact_normal_basis_world": contact_normal_basis,
            "mujoco_contact_index": cid,
        })
    return result


def active_contact_sets(contacts: list[dict]) -> tuple[bool, bool, bool, float]:
    active = [c for c in contacts if c["distance_m"] <= 0.0
              and c["normal_force_n"] > CONTACT_TRIGGER_FORCE_N]
    thumb = any(c["side"] == "right_hand" and c["digit"] == "thumb" for c in active)
    opposing = any(c["side"] == "right_hand" and c["digit"] in {"index", "middle", "ring", "pinky"} for c in active)
    left = any(c["side"] in {"left_hand", "left_arm"} for c in active)
    force = max((c["normal_force_n"] for c in active), default=0.0)
    return thumb, opposing, left, force


def right_digit_contact_forces(contacts: list[dict]) -> dict[str, float]:
    return {
        digit: sum(
            c["normal_force_n"] for c in contacts
            if c["side"] == "right_hand" and c["digit"] == digit
            and c["distance_m"] <= 0.0
        )
        for digit in ("thumb", "index", "middle", "ring", "pinky")
    }


def right_digit_true_normal_forces(contacts: list[dict]) -> dict[str, float]:
    return {
        digit: sum(
            c["true_normal_force_n"] for c in contacts
            if c["side"] == "right_hand" and c["digit"] == digit
            and c["distance_m"] <= 0.0
        )
        for digit in ("thumb", "index", "middle", "ring", "pinky")
    }


def bottle_hand_wrench(contacts: list[dict], bottle_com_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand = [c for c in contacts if c["side"] == "right_hand"]
    force = np.zeros(3)
    torque = np.zeros(3)
    for contact in hand:
        contact_force = np.asarray(contact["force_on_bottle_world_n"], dtype=float)
        contact_position = np.asarray(contact["contact_position_world_m"], dtype=float)
        force += contact_force
        torque += np.cross(contact_position - bottle_com_world, contact_force)
    return force, torque


def measured_friction_utilization(model, data, contacts: list[dict]) -> tuple[float, dict[str, float]]:
    per_contact = {}
    maximum = 0.0
    for contact in contacts:
        if contact["side"] != "right_hand" or contact["true_normal_force_n"] <= 1e-9:
            continue
        index = int(contact["mujoco_contact_index"])
        mu = float(data.contact[index].friction[0])
        if not math.isfinite(mu) or mu <= 0.0:
            raise GateFailure(f"Invalid effective friction at contact {index}: {mu}")
        utilization = float(contact["tangential_force_n"]) / (
            mu * float(contact["true_normal_force_n"])
        )
        key = f"{contact['other_body']}|{contact['other_geom']}"
        per_contact[key] = utilization
        maximum = max(maximum, utilization)
    return maximum, per_contact


def right_digit_contact_geometry(contacts: list[dict]) -> dict[str, list[dict]]:
    return {
        digit: [
            {
                "bottle_geom": c["bottle_geom"], "hand_geom": c["other_geom"],
                "hand_body": c["other_body"], "signed_distance_m": c["distance_m"],
                "normal_force_n": c["normal_force_n"],
                "true_normal_force_n": c["true_normal_force_n"],
                "tangential_force_n": c["tangential_force_n"],
                "contact_position_world_m": c["contact_position_world_m"],
                "contact_normal_basis_world": c["contact_normal_basis_world"],
                "force_on_bottle_world_n": c["force_on_bottle_world_n"],
            }
            for c in contacts
            if c["side"] == "right_hand" and c["digit"] == digit
        ]
        for digit in ("thumb", "index", "middle", "ring", "pinky")
    }


def longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def contact_window_summary(history: list[dict]) -> dict:
    sample_count = len(history)
    result = {
        "sample_count": sample_count,
        "window_s": sample_count * DT,
        "required_samples": CONTACT_VALIDITY_WINDOW_SAMPLES,
        "valid": False,
        "valid_opposing_digits": [],
        "per_digit": {},
        "left_contact_seen": any(item["left_contact"] for item in history),
    }
    if sample_count == 0:
        result["failure_reason"] = "empty_contact_window"
        return result

    for digit in ("thumb", "index", "middle", "ring", "pinky"):
        forces = [float(item["forces_n"][digit]) for item in history]
        bearing = [force > CONTACT_TRIGGER_FORCE_N for force in forces]
        impulse = sum(forces) * DT
        max_zero_samples = longest_false_run(bearing)
        controller_preload = all(
            gates.is_contact_control_state(item["controller_states"].get(digit, ""))
            for item in history
        )
        result["per_digit"][digit] = {
            "force_bearing_duty": sum(bearing) / sample_count,
            "normal_impulse_ns": impulse,
            "mean_normal_force_n": sum(forces) / sample_count,
            "minimum_normal_force_n": min(forces),
            "maximum_normal_force_n": max(forces),
            "maximum_zero_run_samples": max_zero_samples,
            "maximum_zero_run_s": max_zero_samples * DT,
            "controller_contact_preload_through_window": controller_preload,
            "qualifies": (
                sample_count == CONTACT_VALIDITY_WINDOW_SAMPLES
                and sum(bearing) / sample_count >= CONTACT_VALIDITY_MIN_DUTY
                and impulse >= CONTACT_VALIDITY_MIN_IMPULSE_NS
                and max_zero_samples <= CONTACT_VALIDITY_MAX_ZERO_SAMPLES
                and controller_preload
            ),
        }

    result["thumb_qualifies"] = result["per_digit"]["thumb"]["qualifies"]
    result["valid_opposing_digits"] = [
        digit for digit in ("index", "middle", "ring", "pinky")
        if result["per_digit"][digit]["qualifies"]
    ]
    result["valid"] = (
        sample_count == CONTACT_VALIDITY_WINDOW_SAMPLES
        and result["thumb_qualifies"]
        and bool(result["valid_opposing_digits"])
        and not result["left_contact_seen"]
    )
    if not result["valid"]:
        result["failure_reason"] = (
            "left_hand_contact" if result["left_contact_seen"]
            else "thumb_window_invalid" if not result["thumb_qualifies"]
            else "no_opposing_digit_window_valid" if not result["valid_opposing_digits"]
            else "window_not_full"
        )
    return result


def fixed_bin_means(rows: list[dict], field: str, samples_per_bin: int = 40) -> list[dict]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    bins = []
    for start in range(0, len(ordered) - samples_per_bin + 1, samples_per_bin):
        chunk = ordered[start:start + samples_per_bin]
        values = [float(row[field]) for row in chunk]
        bins.append({
            "start_time_s": float(chunk[0]["time_s"]),
            "end_time_s": float(chunk[-1]["time_s"]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "rows": len(chunk),
            "wrist_actual_z_mean_m": float(np.mean([
                float(row["right_wrist_actual_z_m"]) for row in chunk
            ])) if "right_wrist_actual_z_m" in chunk[0] else None,
        })
    return bins


def table_load_transfer_summary(arming_rows: list[dict], load_rows: list[dict],
                                ready_time_s: float) -> dict:
    start = ready_time_s - 0.0500001
    baseline_rows = [
        row for row in arming_rows
        if start <= float(row["time_s"]) <= ready_time_s + 1e-10
    ]
    baseline_bins = fixed_bin_means(baseline_rows, "table_normal_force_n")
    transfer_bins = fixed_bin_means(load_rows, "table_normal_force_n")
    if not baseline_rows or not load_rows:
        return {
            "passed": False,
            "failure_reason": "missing_50ms_preload_or_load_transfer_samples",
            "baseline_sample_count": len(baseline_rows),
            "load_transfer_sample_count": len(load_rows),
        }
    baseline_force = [float(row["table_normal_force_n"]) for row in baseline_rows]
    final_rows = sorted(load_rows, key=lambda row: float(row["time_s"]))[-200:]
    final_force = [float(row["table_normal_force_n"]) for row in final_rows]
    weight = float(load_rows[0]["bottle_weight_n"])
    baseline_mean = float(np.mean(baseline_force))
    baseline_noise = float(np.std([item["mean"] for item in baseline_bins], ddof=1)) \
        if len(baseline_bins) > 1 else 0.0
    detectable_drop = max(3.0 * baseline_noise, 1e-6)
    final_mean = float(np.mean(final_force))
    drop = baseline_mean - final_mean
    slope_n_per_m = None
    correlation = None
    valid_bins = [item for item in transfer_bins if item["wrist_actual_z_mean_m"] is not None]
    if len(valid_bins) >= 2 and np.ptp([item["wrist_actual_z_mean_m"] for item in valid_bins]) > 1e-9:
        x = np.asarray([item["wrist_actual_z_mean_m"] for item in valid_bins])
        y = np.asarray([item["mean"] for item in valid_bins])
        slope_n_per_m = float(np.polyfit(x, y, 1)[0])
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0.0 else 0.0
    unloading = None
    ordered_load = sorted(load_rows, key=lambda row: float(row["time_s"]))
    for index in range(0, len(ordered_load) - 39):
        chunk = ordered_load[index:index + 40]
        mean_force = float(np.mean([float(row["table_normal_force_n"]) for row in chunk]))
        if mean_force < baseline_mean - detectable_drop:
            unloading = {
                "time_s": float(chunk[0]["time_s"]),
                "table_normal_force_at_first_sample_n": float(chunk[0]["table_normal_force_n"]),
                "following_10ms_mean_n": mean_force,
                "threshold_n": baseline_mean - detectable_drop,
            }
            break
    passed = bool(drop > detectable_drop and slope_n_per_m is not None and slope_n_per_m < 0.0)
    return {
        "passed": passed,
        "criterion": "10ms table-normal bins; final 50ms reduction exceeds 3x pre-lift 10ms-bin standard deviation and fitted table-force slope versus actual wrist z is negative",
        "bin_width_s": 0.010,
        "pre_lift_window_s": 0.050,
        "final_window_s": len(final_rows) * DT,
        "baseline_sample_count": len(baseline_rows),
        "baseline_table_normal_mean_n": baseline_mean,
        "baseline_10ms_bin_means": baseline_bins,
        "baseline_10ms_bin_mean_std_n": baseline_noise,
        "minimum_detectable_force_drop_n": detectable_drop,
        "final_table_normal_mean_n": final_mean,
        "table_normal_force_drop_n": drop,
        "bottle_weight_n": weight,
        "baseline_table_weight_fraction": baseline_mean / max(weight, 1e-12),
        "final_table_weight_fraction": final_mean / max(weight, 1e-12),
        "table_weight_fraction_reduction": drop / max(weight, 1e-12),
        "table_force_slope_vs_actual_wrist_z_n_per_m": slope_n_per_m,
        "table_force_wrist_z_correlation": correlation,
        "first_table_unloading_timestep": unloading,
        "load_transfer_10ms_bins": transfer_bins,
        "load_transfer_sample_count": len(load_rows),
    }


def stage3_table_unload_summary(load_build_rows: list[dict], load_rows: list[dict],
                                bottle_weight_n: float, wrist_rise_m: float) -> dict:
    if len(load_build_rows) < 200 or len(load_rows) < 40:
        return {
            "passed": False,
            "failure_reason": "insufficient_load_ready_baseline_or_stage3_samples",
            "load_build_rows": len(load_build_rows), "stage3_rows": len(load_rows),
        }
    load_build_rows = sorted(load_build_rows, key=lambda row: float(row["time_s"]))
    load_rows = sorted(load_rows, key=lambda row: float(row["time_s"]))
    ready_time = float(load_rows[0]["time_s"])
    baseline_rows = [
        row for row in load_build_rows
        if ready_time - 0.0500001 <= float(row["time_s"]) <= ready_time + 1e-10
    ]
    if len(baseline_rows) < 200:
        return {
            "passed": False,
            "failure_reason": "less_than_50ms_stable_load_ready_baseline",
            "baseline_sample_count": len(baseline_rows), "stage3_rows": len(load_rows),
        }
    baseline_table = np.asarray(
        [float(row["table_normal_force_n"]) for row in baseline_rows], dtype=float
    )
    baseline_bins = fixed_bin_means(baseline_rows, "table_normal_force_n")
    transfer_bins = fixed_bin_means(load_rows, "table_normal_force_n")
    baseline_bin_means = [item["mean"] for item in baseline_bins]
    baseline_noise = float(np.std(baseline_bin_means, ddof=1)) if len(baseline_bin_means) > 1 else 0.0
    final_rows = load_rows[-200:]
    baseline_mean = float(np.mean(baseline_table))
    final_table_mean = float(np.mean([float(row["table_normal_force_n"]) for row in final_rows]))
    baseline_hand_vertical = float(np.mean([
        float(row["hand_vertical_support_force_n"]) for row in baseline_rows
    ]))
    final_hand_vertical = float(np.mean([
        float(row["hand_vertical_support_force_n"]) for row in final_rows
    ]))
    table_drop = baseline_mean - final_table_mean
    hand_gain = final_hand_vertical - baseline_hand_vertical
    material_threshold = max(3.0 * baseline_noise, 0.05 * bottle_weight_n)
    monotonic_tolerance = max(3.0 * baseline_noise, 0.005 * bottle_weight_n)
    binned_table = [item["mean"] for item in transfer_bins]
    monotonic_increases = [
        max(0.0, binned_table[i + 1] - binned_table[i])
        for i in range(len(binned_table) - 1)
    ]
    maximum_rebound = max(monotonic_increases, default=0.0)
    wrist_z_bins = [item for item in transfer_bins if item.get("wrist_actual_z_mean_m") is not None]
    slope = None
    if len(wrist_z_bins) >= 2 and np.ptp([item["wrist_actual_z_mean_m"] for item in wrist_z_bins]) > 1e-9:
        slope = float(np.polyfit(
            [item["wrist_actual_z_mean_m"] for item in wrist_z_bins],
            [item["mean"] for item in wrist_z_bins], 1
        )[0])
    correspondence_error = abs(hand_gain - table_drop)
    balance_tolerance = max(3.0 * baseline_noise, 0.10 * bottle_weight_n)
    passed = bool(
        table_drop >= material_threshold
        and hand_gain > 0.0
        and correspondence_error <= balance_tolerance
        and maximum_rebound <= monotonic_tolerance
        and slope is not None and slope < 0.0
        and 0.0 < wrist_rise_m <= 0.0005 + LOAD_BUILD_WRIST_DRIFT_GATE_TOLERANCE_M
    )
    return {
        "passed": passed,
        "criterion": "load-ready 50ms baseline; table force reduction >= max(3x baseline-bin noise, 5% bottle weight); hand vertical support gain balances table loss within max(3x noise, 10% weight); 10ms table-force rebounds stay within max(3x noise, 0.5% weight); negative table-force slope versus actual wrist z; measured wrist rise <=0.500mm plus 0.001mm comparison tolerance",
        "baseline_window_s": 0.05,
        "bin_width_s": 0.01,
        "baseline_sample_count": len(baseline_rows),
        "stage3_sample_count": len(load_rows),
        "baseline_table_normal_mean_n": baseline_mean,
        "baseline_10ms_bin_means": baseline_bins,
        "baseline_10ms_bin_mean_std_n": baseline_noise,
        "material_table_drop_threshold_n": material_threshold,
        "monotonic_rebound_tolerance_n": monotonic_tolerance,
        "maximum_10ms_table_force_rebound_n": maximum_rebound,
        "stage3_final_table_normal_mean_n": final_table_mean,
        "table_normal_drop_n": table_drop,
        "table_normal_drop_fraction_of_weight": table_drop / max(bottle_weight_n, 1e-12),
        "absolute_hand_supported_fraction_at_end": 1.0 - final_table_mean / max(bottle_weight_n, 1e-12),
        "baseline_hand_vertical_support_mean_n": baseline_hand_vertical,
        "final_hand_vertical_support_mean_n": final_hand_vertical,
        "hand_vertical_support_gain_n": hand_gain,
        "force_balance_correspondence_error_n": correspondence_error,
        "force_balance_tolerance_n": balance_tolerance,
        "bottle_weight_n": bottle_weight_n,
        "table_force_slope_vs_actual_wrist_z_n_per_m": slope,
        "measured_wrist_rise_m": wrist_rise_m,
        "table_normal_load_transfer_history": [
            {
                "time_s": float(row["time_s"]),
                "table_normal_force_n": float(row["table_normal_force_n"]),
                "hand_supported_fraction": 1.0 - float(row["table_normal_force_n"]) / max(bottle_weight_n, 1e-12),
                "hand_vertical_support_force_n": float(row["hand_vertical_support_force_n"]),
                "bottle_z_m": float(row["bottle_z_m"]),
                "bottle_z_velocity_m_s": float(row["bottle_z_velocity_m_s"]),
                "right_wrist_actual_z_m": float(row["right_wrist_actual_z_m"]),
            }
            for row in load_rows
        ],
    }


def force_bearing_contact_details(model, data, contacts: list[dict]) -> list[dict]:
    details = []
    for item in contacts:
        if (item["side"] != "right_hand" or item["distance_m"] > 0.0
                or item["normal_force_n"] <= CONTACT_TRIGGER_FORCE_N):
            continue
        contact = data.contact[int(item["mujoco_contact_index"])]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, int(item["mujoco_contact_index"]), wrench)
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        details.append({
            "digit": item["digit"], "bottle_geom": item["bottle_geom"],
            "other_geom": item["other_geom"], "other_body": item["other_body"],
            "geom1": geom1, "geom2": geom2,
            "distance_m": item["distance_m"], "normal_force_n": item["normal_force_n"],
            "true_normal_force_n": abs(float(wrench[0])),
            "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
            "contact_position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_frame_world": frame.tolist(),
            "contact_frame_normal_basis_world": frame[0].tolist(),
            "contact_wrench_local": wrench.tolist(),
            "force_on_bottle_world_n": item["force_on_bottle_world_n"],
        })
    return details


def _mimic_velocity_factor(joint: str, driver: str, mimics: dict,
                           visited: frozenset[str] = frozenset()) -> float:
    if joint == driver:
        return 1.0
    if joint in visited or joint not in mimics:
        return 0.0
    relation = mimics[joint]
    return float(relation["multiplier"]) * _mimic_velocity_factor(
        relation["parent"], driver, mimics, visited | {joint}
    )


def live_contact_allocation_geometry(model, data, contacts: list[dict],
                                     channel_joint_names: dict[str, str],
                                     mimics: dict) -> dict:
    """Build current contact bases and driver-reduced point Jacobians."""
    bottle_geoms = {"bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"}
    channels = [channel for channel, _ in hand.CHANNELS if channel in channel_joint_names]
    bases, positions, frictions, torque_maps, metadata = [], [], [], [], []
    candidates = [item for item in contacts
                  if item["side"] == "right_hand" and item["digit"] is not None
                  and item["distance_m"] <= 0.0
                  and item["true_normal_force_n"] > CONTACT_TRIGGER_FORCE_N]
    if len(candidates) > 24:
        return {"success": False, "reason": "live_contact_count_exceeds_bounded_solver",
                "contact_count": len(candidates), "maximum_contacts": 24}

    for item in candidates:
        contact_id = int(item["mujoco_contact_index"])
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        bottle_is_geom1 = name1 in bottle_geoms
        other_geom_id = geom2 if bottle_is_geom1 else geom1
        other_body_id = int(model.geom_bodyid[other_geom_id])
        sign = -1.0 if bottle_is_geom1 else 1.0
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        basis_world = sign * frame.T
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jac(model, data, jacp, jacr, np.asarray(contact.pos), other_body_id)
        effective = np.zeros((3, len(channels)), dtype=float)
        for channel_index, channel in enumerate(channels):
            driver = channel_joint_names[channel]
            for joint in (driver, *mimics.keys()):
                factor = _mimic_velocity_factor(joint, driver, mimics)
                if abs(factor) <= 1e-12:
                    continue
                joint_id = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                dof = int(model.jnt_dofadr[joint_id])
                effective[:, channel_index] += jacp[:, dof] * factor
        bases.append(basis_world)
        positions.append(np.asarray(contact.pos, dtype=float).copy())
        friction = np.asarray(contact.friction[:2], dtype=float)
        frictions.append((float(friction[0]), float(friction[1])))
        torque_maps.append(effective.T @ basis_world)
        metadata.append({
            "contact_index": contact_id,
            "digit": item["digit"],
            "hand_geom": item["other_geom"],
            "hand_body": item["other_body"],
            "bottle_geom": item["bottle_geom"],
            "distance_m": float(contact.dist),
            "normal_force_n": float(item["true_normal_force_n"]),
            "contact_position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_basis_world_columns": basis_world.tolist(),
            "friction_axes": friction.tolist(),
        })
    return {
        "success": bool(bases), "reason": None if bases else "no_current_right_hand_bottle_contacts",
        "channels": channels,
        "contact_bases_world": [item.tolist() for item in bases],
        "contact_positions_world": [item.tolist() for item in positions],
        "contact_friction": frictions,
        "contact_driver_torque_maps": [item.tolist() for item in torque_maps],
        "contacts": metadata,
    }


def current_joint_state(model, data, mimics: dict) -> dict:
    drivers = {}
    followers = {}
    max_mimic = 0.0
    for side in ("L", "R"):
        for _, suffix in hand.CHANNELS:
            name = hand.channel_joint(side, suffix)
            drivers[name] = {"q": joint_qpos(model, data, name), "qd": joint_qvel(model, data, name)}
    for follower, relation in mimics.items():
        qpos = joint_qpos(model, data, follower)
        qvel = joint_qvel(model, data, follower)
        error = qpos - (
            relation["multiplier"] * joint_qpos(model, data, relation["parent"]) + relation["offset"]
        )
        followers[follower] = {"parent": relation["parent"], "q": qpos, "qd": qvel, "error": error}
        max_mimic = max(max_mimic, abs(error))
    return {"drivers": drivers, "followers": followers, "max_mimic": max_mimic}


def max_hand_limit_violation(model, data, limits: dict) -> tuple[float, dict | None]:
    maximum = 0.0
    offender = None
    for name, (lower, upper) in limits.items():
        if name.startswith(("L_", "R_")):
            value = joint_qpos(model, data, name)
            violation = max(lower - value, value - upper, 0.0)
            if violation > maximum:
                maximum = violation
                offender = {"joint": name, "qpos_rad": value,
                            "lower_limit_rad": lower, "upper_limit_rad": upper,
                            "violation_rad": violation}
    return maximum, offender


def corrected_hand_rotation(model, data, site_id: int) -> np.ndarray:
    base = data.site_xmat[site_id].reshape(3, 3).copy()
    yaw = math.radians(WRIST_YAW_CORRECTION_DEG)
    correction = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return correction @ base


def pregrasp_ik(model, data, site_id: int, bottle_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    rotation = corrected_hand_rotation(model, data, site_id)
    qadr = arm_qpos_addresses(model)
    data.qpos[qadr] = 0.0
    mujoco.mj_forward(model, data)
    target = bottle_position + PREGRASP_SITE_OFFSET_M
    arm_q, ik = plan_arm(model, data, site_id, target, rotation)
    data.qpos[qadr] = arm_q
    mujoco.mj_forward(model, data)
    robot_contacts = [c for c in bottle_contacts(model, data) if c["side"] != "other"]
    if not ik["converged"] or robot_contacts:
        raise GateFailure(
            "Corrected pregrasp is unreachable or not contact-free: "
            + json.dumps({"ik": ik, "robot_bottle_contacts": robot_contacts}, separators=(",", ":"))
        )
    return arm_q, rotation, {
        "ik": ik,
        "robot_bottle_contacts": robot_contacts,
        "site_offset_m": PREGRASP_SITE_OFFSET_M.tolist(),
        "wrist_yaw_correction_deg": WRIST_YAW_CORRECTION_DEG,
    }


def plan_lift(model, data, site_id: int, approach_q: np.ndarray,
              position: np.ndarray, rotation: np.ndarray,
              lift_height_m: float = LIFT_TARGET_M) -> tuple[np.ndarray, dict]:
    qadr = arm_qpos_addresses(model)
    data.qpos[qadr] = approach_q
    mujoco.mj_forward(model, data)
    target = position + np.array([0.0, 0.0, lift_height_m])
    lift_q, detail = plan_arm(model, data, site_id, target, rotation)
    data.qpos[qadr] = approach_q
    mujoco.mj_forward(model, data)
    if not detail["converged"]:
        raise GateFailure(f"Lift IK failed: {detail}")
    return lift_q, detail


def ast_rollout_write_check() -> dict:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_rollout")
    lines = []
    for node in ast.walk(function):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Attribute) and sub.attr == "qpos" and isinstance(sub.ctx, ast.Store):
                    lines.append(node.lineno)
                if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store) and isinstance(sub.value, ast.Attribute) and sub.value.attr == "qpos":
                    lines.append(node.lineno)
    if lines:
        raise GateFailure(f"Active rollout contains qpos writes at lines {sorted(set(lines))}")
    return {"active_rollout_qpos_write_count": 0, "lines": []}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def forward_wrist_target(model, fk_data, arm_qadr: np.ndarray, arm_target: np.ndarray,
                         wrist_id: int, site_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fk_data.qpos[arm_qadr] = arm_target
    mujoco.mj_forward(model, fk_data)
    return (fk_data.xpos[wrist_id].copy(), fk_data.xquat[wrist_id].copy(),
            fk_data.site_xpos[site_id].copy())


def make_renderer(model):
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.300, -0.100, 0.900]
    camera.distance, camera.azimuth, camera.elevation = 0.52, -130, -12
    return renderer, camera


def render_frame(renderer, model, data, camera, title: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    image = cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (960, 48), (22, 30, 36), -1)
    cv2.putText(image, f"{title} | t={data.time:.3f}s", (16, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def save_contact_views(renderer, model, data, camera, site_id: int, bottle_id: int,
                       output: Path, phase: str) -> None:
    saved = (camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation)
    site = data.site_xpos[site_id].copy()
    bottle = data.xpos[bottle_id].copy()
    configs = {
        "front": ((site + bottle) * 0.5, 90.0, -3.0, 0.34),
        "side": ((site + bottle) * 0.5, 0.0, -3.0, 0.34),
        "palm": (site * 0.65 + bottle * 0.35, -130.0, -8.0, 0.26),
    }
    for view, (lookat, azimuth, elevation, distance) in configs.items():
        camera.lookat[:] = lookat
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        image = render_frame(renderer, model, data, camera, f"{phase} / {view.upper()}")
        cv2.imwrite(str(output / f"{phase.lower()}_{view}.png"), image)
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = saved


def run_rollout(model, data, meta, output: Path, result: dict, settle_clear_q: np.ndarray,
                pregrasp_q: np.ndarray, approach_q: np.ndarray, lift_q: np.ndarray,
                ranges: dict, limits: dict,
                mimics: dict, diagnostic_only: bool, lift_height_m: float,
                post_lift_hold_seconds: float, load_transfer_only: bool,
                load_capacity_targets: dict, stop_after_load_build: bool = False,
                direct_arm_lift: bool = False) -> dict:
    """Active physics loop; no direct qpos writes, object constraints, or mocap."""
    qpos_check = ast_rollout_write_check()
    body_kp = np.asarray([44,44,44,70,25,25,44,44,44,70,25,25,44,25,25,
                          12,12,12,12,3,3,3,12,12,12,12,3,3,3], dtype=float)
    body_kd = np.asarray([4,4,4,7,2.5,2.5,4,4,4,7,2.5,2.5,4,2.5,2.5,
                          1.2,1.2,1.2,1.2,0.3,0.3,0.3,1.2,1.2,1.2,1.2,0.3,0.3,0.3], dtype=float)
    body_count = 29
    if len(body_kp) != body_count or len(body_kd) != body_count:
        raise GateFailure("Pinned upstream arm/body gain map is malformed")
    body_joint_ids = model.actuator_trnid[:body_count, 0].astype(int)
    body_dofs = model.jnt_dofadr[body_joint_ids].astype(int)
    body_ctrlrange = model.actuator_ctrlrange[:body_count].copy()
    body_targets = data.actuator_length[:body_count].copy()
    arm_qadr = arm_qpos_addresses(model)
    if not np.allclose(data.qpos[arm_qadr], settle_clear_q, atol=1e-12):
        raise GateFailure("Bottle-settle arm state changed before rollout")
    hand_ids = {ch: element_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for ch, name in meta["right_hand_actuators"].items()}
    hand_driver_dofs = {
        channel: int(model.jnt_dofadr[int(model.actuator_trnid[aid, 0])])
        for channel, aid in hand_ids.items()
    }
    hand_position_gainprm = {aid: model.actuator_gainprm[aid].copy() for aid in hand_ids.values()}
    hand_position_biasprm = {aid: model.actuator_biasprm[aid].copy() for aid in hand_ids.values()}
    hand_effort_allocation = load_capacity_targets["effort_balanced_10pct_margin_allocation"]
    static_effort_by_channel = {
        channel: float(hand_effort_allocation["driver_efforts"][channel]["estimated_required_effort_nm"])
        for channel in hand_ids
    }
    load_effort_by_channel = LOAD_CONTACT_REALLOCATION["driver_feedforward_nm"]
    channel_joint_names = {
        channel: hand.channel_joint("R", suffix) for channel, suffix in hand.CHANNELS
    }
    hand_velocity_chain_names = {}
    for channel in hand_ids:
        chain = {channel_joint_names[channel]}
        changed = True
        while changed:
            changed = False
            for follower, relation in mimics.items():
                if relation["parent"] in chain and follower not in chain:
                    chain.add(follower)
                    changed = True
        hand_velocity_chain_names[channel] = tuple(sorted(chain))
    hand_velocity_chain_dofs = {
        channel: {
            name: int(model.jnt_dofadr[element_id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )])
            for name in names
        }
        for channel, names in hand_velocity_chain_names.items()
    }
    channel_digits = {
        channel: next(digit for digit in ("thumb", "index", "middle", "ring", "pinky")
                      if digit in channel)
        for channel in hand_ids
    }
    contact_controller_state = {
        digit: "FREE_CLOSE" for digit in ("thumb", "index", "middle", "ring", "pinky")
    }
    current_preload_cap_rad = CONTACT_PRELOAD_MAX_RAD
    contact_hold_targets: dict[str, float] = {}
    contact_preload_offsets: dict[str, float] = {}
    contact_commanded_errors: dict[str, float] = {}
    contact_hold_transition_times: dict[str, float | None] = {
        channel: None for channel in hand_ids
    }
    digit_contact_run = {digit: 0 for digit in ("thumb", "index", "middle", "ring", "pinky")}
    digit_contact_lost_run = {digit: 0 for digit in digit_contact_run}
    last_close_command_u = 1.0
    # Start from the measured pre-rollout driver state. Reset commands derive
    # from model.qpos0 and can differ from a caller's already-open state,
    # producing an avoidable first-step position jump before SETTLE.
    initial_commands = {}
    for channel, suffix in hand.CHANNELS:
        joint_name = hand.channel_joint("R", suffix)
        jid = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        q_actual = float(data.qpos[int(model.jnt_qposadr[jid])])
        lo, hi = ranges[joint_name]["simulation_feasible_range_rad"]
        initial_commands[channel] = float(np.clip((q_actual - hi) / (lo - hi), 0.0, 1.0))
    initial_targets = {ch: hand.normalized_target("R", ch, initial_commands[ch], ranges)
                       for ch, _ in hand.CHANNELS}
    open_targets = {ch: hand.normalized_target("R", ch, 1.0, ranges) for ch, _ in hand.CHANNELS}
    close_targets = {
        ch: float(meta.get("close_target_rad_by_channel", {}).get(
            ch, hand.normalized_target("R", ch, 0.0, ranges)
        ))
        for ch, _ in hand.CHANNELS
    }
    closure_directions = {
        channel: (1.0 if close_targets[channel] >= open_targets[channel] else -1.0)
        for channel in hand_ids
    }
    channel_close_targets = initial_targets.copy()
    channel_close_rates = {channel: 0.0 for channel in hand_ids}
    maximum_contact_target_error_rad = 0.0
    minimum_contact_target_error_rad = math.inf
    maximum_contact_commanded_error_rad = 0.0
    maximum_hand_actuator_effort_nm = 0.0
    last_applied_targets = {channel: float(data.ctrl[aid]) for channel, aid in hand_ids.items()}
    for channel, aid in hand_ids.items():
        data.ctrl[aid] = initial_targets[channel]

    renderer, camera = make_renderer(model)
    writer = cv2.VideoWriter(str(output / "short_grasp.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (960, 720))
    if not writer.isOpened():
        renderer.close()
        raise GateFailure("Could not open short-grasp MP4 writer")
    bottle_id = element_id(model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
    site_id = element_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
    left_site = element_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
    wrist_id = element_id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    fk_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, fk_data)
    mujoco.mj_forward(model, fk_data)
    right_joint_meta = {
        name: (
            int(model.jnt_qposadr[element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
            int(model.jnt_dofadr[element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
        )
        for name in limits if name.startswith("R_")
    }
    bottle_qadr = bottle_qpos_addresses(model)
    spawn_bottle_qpos = data.qpos[bottle_qadr].copy()
    spawn_bottle_pos = spawn_bottle_qpos[:3].copy()
    initial_bottle_qpos = spawn_bottle_qpos.copy()
    initial_bottle_pos = spawn_bottle_pos.copy()
    settled_bottle_pos: np.ndarray | None = None

    (hand_rows, contact_rows, bottle_rows, arm_rows, digit_force_rows, mimic_rows,
     settle_rows, contact_aware_rows, events) = (
        [], [], [], [], [], [], [], [], []
    )
    transition_trace_rows: list[dict] = []
    lift_arming_trace_rows: list[dict] = []
    contact_window_rows: list[dict] = []
    load_transfer_rows: list[dict] = []
    load_build_trace_rows: list[dict] = []
    effort_control_trace_rows: list[dict] = []
    live_contact_allocation_trace: list[dict] = []
    lift_stage_trace_rows: list[dict] = []
    release_trace_rows: list[dict] = []
    current_load_stage_name = ""
    hand_motion_by_phase: dict[str, dict] = {}
    load_build_stage3_wrist_start: np.ndarray | None = None
    load_build_stage3_bottle_start: np.ndarray | None = None
    effort_q_reference: dict[str, float] = {}
    effort_mode_active = False
    live_allocation_state = {
        "last_update_step": -LIVE_ALLOCATION_UPDATE_STEPS,
        "last_target_force_n": None,
        "support_target_n": 0.0,
        "last_support_update_time_s": None,
        "feedforward_by_channel": {channel: 0.0 for channel in hand_ids},
        "allocation": None,
        "base_effort_by_channel": {},
    }
    contact_window_history = deque(maxlen=CONTACT_VALIDITY_WINDOW_SAMPLES)
    transition_hold_start: float | None = None
    transition_lift_start: float | None = None
    steps = 0
    last_trace, last_video = -1.0, -1.0
    hold_contact_steps = 0
    hold_table_support_steps = 0
    table_support_misses = 0
    diagnostic_sample_steps = 0
    max_hold_bottle_lift = 0.0
    max_force = max_penetration = max_mimic = max_limit = max_qvel = 0.0
    max_mimic_diagnostic = 0.0
    mimic_diagnostic_exceed_steps = 0
    first_mimic_diagnostic_exceed = None
    max_limit_detail = None
    first_limit_breach = None
    max_qvel_joint = None
    first_velocity_breach = None
    left_contact_seen = thumb_seen = opposing_seen = False

    def arm_motor_targets(q: np.ndarray) -> np.ndarray:
        targets = body_targets.copy()
        requested = dict(zip(ARM_NAMES, q.tolist()))
        for aid in range(body_count):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(body_joint_ids[aid])) or ""
            if name in requested:
                targets[aid] = requested[name]
        return targets

    def set_body_ctrl(targets: np.ndarray) -> None:
        tau = body_kp * (targets - data.actuator_length[:body_count])
        tau -= body_kd * data.actuator_velocity[:body_count]
        tau += data.qfrc_bias[body_dofs]
        np.copyto(data.ctrl[:body_count],
                  np.clip(tau, body_ctrlrange[:, 0], body_ctrlrange[:, 1]))

    def begin_simulation_only_effort_mode() -> None:
        nonlocal effort_mode_active
        if effort_mode_active:
            raise GateFailure("Simulation-only effort mode was activated more than once")
        handoff_references = gates.effort_handoff_references(channel_close_targets)
        handoff_channels = {}
        for channel, aid in hand_ids.items():
            actual_qpos = joint_qpos(model, data, channel_joint_names[channel])
            effort_q_reference[channel] = handoff_references[channel]
            handoff_channels[channel] = {
                "joint": channel_joint_names[channel],
                "commanded_position_target_rad": handoff_references[channel],
                "qpos_at_handoff_rad": actual_qpos,
                "closure_direction_error_rad": closure_directions[channel]
                * (handoff_references[channel] - actual_qpos),
            }
            model.actuator_gainprm[aid, :] = 0.0
            model.actuator_biasprm[aid, :] = 0.0
            data.ctrl[aid] = 0.0
        result["simulation_effort_mode_handoff"] = {
            "reference_source": "last bounded CLOSE/HOLD commanded position target",
            "position_targets_preserved": True,
            "qpos_written": False,
            "channels": handoff_channels,
        }
        for digit in contact_controller_state:
            contact_controller_state[digit] = "SIMULATION_ONLY_TORQUE_IMPEDANCE"
        effort_mode_active = True

    def apply_simulation_only_effort(elapsed_s: float, *, load_build_schedule: bool = False,
                                     live_support_force_n: float | None = None) -> dict:
        if not effort_mode_active:
            raise GateFailure("Effort command requested before effort mode activation")
        ramp_phase = min(max(elapsed_s / SIM_ONLY_FEEDFORWARD_RAMP_S, 0.0), 1.0)
        feedforward_ramp = 10.0 * ramp_phase**3 - 15.0 * ramp_phase**4 + 6.0 * ramp_phase**5
        data.qfrc_applied[:] = 0.0
        commanded = {}
        live_allocation = None
        live_feedforward = None
        if load_build_schedule and live_support_force_n is not None:
            allocation_channels = [channel for channel, _ in hand.CHANNELS if channel in hand_ids]
            base_by_channel = {}
            for channel in allocation_channels:
                joint = channel_joint_names[channel]
                dof = hand_driver_dofs[channel]
                q = joint_qpos(model, data, joint)
                qd = joint_qvel(model, data, joint)
                position_impedance = HAND_KP * (effort_q_reference[channel] - q)
                if channel == "thumb_proximal_yaw":
                    position_impedance = 0.0
                base_by_channel[channel] = (
                    position_impedance - HAND_KV * qd + float(data.qfrc_bias[dof])
                )
            desired_support = float(np.clip(live_support_force_n, 0.0, bottle_weight_n))
            previous_support_time = live_allocation_state["last_support_update_time_s"]
            support_dt = DT if previous_support_time is None else max(
                0.0, min(float(data.time) - float(previous_support_time), DT * 2.0)
            )
            support_delta = LIVE_SUPPORT_TARGET_RATE_N_S * support_dt
            target_support = float(live_allocation_state["support_target_n"] + np.clip(
                desired_support - live_allocation_state["support_target_n"],
                -support_delta,
                support_delta,
            ))
            live_allocation_state["support_target_n"] = target_support
            live_allocation_state["last_support_update_time_s"] = float(data.time)
            refresh = (
                live_allocation_state["allocation"] is None
                or steps - int(live_allocation_state["last_update_step"])
                    >= LIVE_ALLOCATION_UPDATE_STEPS
                or abs(target_support - float(live_allocation_state["last_target_force_n"] or 0.0))
                    >= 0.05
            )
            if refresh:
                if target_support <= 1e-6:
                    allocation = {
                        "success": True,
                        "target_vertical_force_n": target_support,
                        "predicted_wrench_on_bottle": {
                            "force_n": [0.0, 0.0, 0.0], "torque_about_com_nm": [0.0, 0.0, 0.0]
                        },
                        "driver_effort_nm": [base_by_channel[channel]
                                             for channel in allocation_channels],
                        "driver_effort_headroom_nm": [], "per_contact": [],
                        "total_normal_force_n": 0.0,
                        "total_conservative_resultant_bound_n": 0.0,
                        "maximum_friction_utilization": 0.0,
                        "solver_message": "zero support requested while table carries the bottle",
                    }
                    geometry = {"success": True, "channels": allocation_channels,
                                "contacts": [], "contact_bases_world": [],
                                "contact_positions_world": [], "contact_friction": [],
                                "contact_driver_torque_maps": []}
                else:
                    geometry = live_contact_allocation_geometry(
                        model, data, bottle_contacts(model, data), channel_joint_names, mimics
                    )
                    if not geometry.get("success"):
                        failure = {
                            "time_s": float(data.time),
                            "target_vertical_force_n": target_support,
                            "contact_geometry": geometry,
                        }
                        result["live_contact_allocation_failure"] = failure
                        raise GateFailure("Live contact allocation has no current usable contacts: "
                                          + json.dumps(failure, sort_keys=True))
                    allocation = gates.solve_live_contact_wrench_allocation(
                        geometry["contact_bases_world"],
                        geometry["contact_positions_world"],
                        geometry["contact_friction"],
                        geometry["contact_driver_torque_maps"],
                        np.asarray(data.xipos[bottle_id], dtype=float),
                        target_support,
                        np.asarray([base_by_channel[channel] for channel in allocation_channels]),
                        driver_effort_limit_nm=(SIM_ONLY_MAX_DRIVER_TORQUE_NM
                                                * LIVE_ALLOCATION_EFFORT_HEADROOM_FRACTION),
                        total_normal_limit_n=(LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N
                                              * LIVE_ALLOCATION_NORMAL_HEADROOM_FRACTION),
                        single_contact_limit_n=(LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N
                                                * LIVE_ALLOCATION_CONTACT_HEADROOM_FRACTION),
                        total_contact_limit_n=(LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N
                                               * LIVE_ALLOCATION_TOTAL_CONTACT_HEADROOM_FRACTION),
                        horizontal_force_tolerance_n=(
                            SIM_ONLY_HORIZONTAL_FORCE_TOLERANCE_FRACTION * bottle_weight_n
                        ),
                        torque_tolerance_nm=SIM_ONLY_WRENCH_TORQUE_TOLERANCE_NM,
                        friction_utilization_limit=LIVE_ALLOCATION_FRICTION_UTILIZATION,
                    )
                    if not allocation.get("success"):
                        failure = {
                            "time_s": float(data.time),
                            "target_vertical_force_n": target_support,
                            "base_driver_effort_nm": base_by_channel,
                            "contact_geometry": geometry,
                            "solver": allocation,
                        }
                        result["live_contact_allocation_failure"] = failure
                        raise GateFailure("Live contact wrench is infeasible under existing caps: "
                                          + json.dumps(failure, sort_keys=True))
                feedforward_values = allocation["driver_effort_nm"]
                live_feedforward = {
                    channel: float(feedforward_values[index] - base_by_channel[channel])
                    for index, channel in enumerate(allocation_channels)
                }
                live_allocation_state.update({
                    "last_update_step": steps,
                    "last_target_force_n": target_support,
                    "feedforward_by_channel": live_feedforward,
                    "allocation": allocation,
                    "base_effort_by_channel": base_by_channel,
                    "contact_geometry": geometry,
                })
                trace_entry = {
                    "time_s": float(data.time),
                    "physics_step": steps,
                    "target_vertical_force_n": target_support,
                    "base_driver_effort_nm": base_by_channel,
                    "allocation": allocation,
                    "contact_geometry": geometry.get("contacts", []),
                    "contact_driver_torque_maps": [np.asarray(item).tolist()
                                                    for item in geometry.get("contact_driver_torque_maps", [])],
                }
                live_contact_allocation_trace.append(trace_entry)
            else:
                live_feedforward = live_allocation_state["feedforward_by_channel"]
                allocation = live_allocation_state["allocation"]
            live_allocation = {
                "target_vertical_force_n": target_support,
                "last_solved_target_vertical_force_n": live_allocation_state["last_target_force_n"],
                "last_solved_time_s": float(data.time),
                "allocation": allocation,
                "contact_geometry": live_allocation_state.get("contact_geometry", {}).get("contacts", []),
            }
        else:
            live_feedforward = None
        for channel, dof in hand_driver_dofs.items():
            joint = channel_joint_names[channel]
            q = joint_qpos(model, data, joint)
            qd = joint_qvel(model, data, joint)
            q_error = effort_q_reference[channel] - q
            position_impedance = HAND_KP * q_error
            impedance = position_impedance - HAND_KV * qd
            if load_build_schedule and channel == "thumb_proximal_yaw":
                # The prior trace showed the position spring canceling the
                # statically derived thumb-yaw support torque during LOAD_BUILD.
                impedance -= position_impedance
            bias = float(data.qfrc_bias[dof])
            if live_feedforward is not None:
                feedforward = float(live_feedforward.get(channel, 0.0))
            else:
                feedforward_source = gates.select_effort_feedforward(
                    channel, static_effort_by_channel, load_effort_by_channel,
                    load_build_schedule,
                )
                feedforward = feedforward_ramp * feedforward_source
            requested = feedforward + impedance + bias
            if live_feedforward is not None:
                live_effort_limit = (SIM_ONLY_MAX_DRIVER_TORQUE_NM
                                     * LIVE_ALLOCATION_EFFORT_HEADROOM_FRACTION)
                if abs(requested) > live_effort_limit + 1e-8:
                    raise GateFailure(
                        f"Live allocation effort headroom failed for {channel}: "
                        f"{requested:.9g} Nm > {live_effort_limit:.6f} Nm"
                    )
                applied = float(requested)
            else:
                applied = float(np.clip(
                    requested,
                    -SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                    SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                ))
            data.qfrc_applied[dof] = applied
            commanded[channel] = {
                "joint": joint,
                "dof": dof,
                "q_reference_rad": effort_q_reference[channel],
                "q_command_state_rad": q,
                "qd_command_state_rad_s": qd,
                "feedforward_nm": feedforward,
                "impedance_nm": impedance,
                "bias_compensation_nm": bias,
                "requested_nm": requested,
                "applied_nm": applied,
                "saturated": abs(requested - applied) > 1e-12,
                "velocity_guard_brake_nm": 0.0,
                "velocity_guard_minimum_safe_scale": None,
                "velocity_guard_minimum_peak_rad_s": None,
                "predicted_qvel_before_guard_rad_s": None,
                "predicted_qvel_after_guard_rad_s": None,
            }
        mujoco.mj_forward(model, data)
        for _ in range(2):
            for channel, dof in hand_driver_dofs.items():
                chain_dofs = hand_velocity_chain_dofs[channel]
                current_velocities = {
                    name: float(data.qvel[chain_dof])
                    for name, chain_dof in chain_dofs.items()
                }
                predicted_velocities = {
                    name: current_velocities[name] + DT * float(data.qacc[chain_dof])
                    for name, chain_dof in chain_dofs.items()
                }
                peak_current = max(map(abs, current_velocities.values()), default=0.0)
                peak_predicted = max(map(abs, predicted_velocities.values()), default=0.0)
                driver_name = channel_joint_names[channel]
                commanded[channel]["predicted_qvel_before_guard_rad_s"] = float(
                    predicted_velocities[driver_name]
                )
                commanded[channel]["predicted_qvel_chain_before_guard_rad_s"] = predicted_velocities.copy()
                if peak_predicted <= SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S:
                    commanded[channel]["predicted_qvel_after_guard_rad_s"] = float(
                        predicted_velocities[driver_name]
                    )
                    commanded[channel]["predicted_qvel_chain_after_guard_rad_s"] = predicted_velocities.copy()
                    continue

                driver_qd = current_velocities[driver_name]
                driver_prediction = predicted_velocities[driver_name]
                direction_source = driver_qd if abs(driver_qd) > 1e-12 else driver_prediction
                direction = math.copysign(1.0, direction_source)
                base = float(data.qfrc_applied[dof])
                braking_endpoint = -direction * SIM_ONLY_MAX_DRIVER_TORQUE_NM

                def predict_with_torque(torque: float) -> dict[str, float]:
                    data.qfrc_applied[dof] = torque
                    mujoco.mj_forward(model, data)
                    return {
                        name: current_velocities[name] + DT * float(data.qacc[chain_dof])
                        for name, chain_dof in chain_dofs.items()
                    }

                # Minimize predicted chain speed, then use only the least braking
                # needed to meet the existing prediction ceiling.
                low, high = 0.0, 1.0
                for _ in range(SIM_ONLY_VELOCITY_GUARD_BISECTION_STEPS):
                    left = (2.0 * low + high) / 3.0
                    right = (low + 2.0 * high) / 3.0
                    left_torque = base + left * (braking_endpoint - base)
                    right_torque = base + right * (braking_endpoint - base)
                    left_prediction = predict_with_torque(left_torque)
                    right_prediction = predict_with_torque(right_torque)
                    left_peak = max(map(abs, left_prediction.values()), default=0.0)
                    right_peak = max(map(abs, right_prediction.values()), default=0.0)
                    if left_peak <= right_peak:
                        high = right
                    else:
                        low = left
                candidate_scales = (0.0, 1.0, 0.5 * (low + high))
                minimum_scale = min(
                    candidate_scales,
                    key=lambda scale: max(map(abs, predict_with_torque(
                        base + scale * (braking_endpoint - base)
                    ).values()), default=0.0),
                )
                minimum_peak = max(map(abs, predict_with_torque(
                    base + minimum_scale * (braking_endpoint - base)
                ).values()), default=0.0)
                if minimum_peak > SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S:
                    data.qfrc_applied[dof] = base
                    mujoco.mj_forward(model, data)
                    raise GateFailure(
                        f"Velocity guard has no safe command for {driver_name}: "
                        f"minimum predicted chain speed {minimum_peak:.9g} rad/s exceeds "
                        f"{SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S:.3f} rad/s"
                    )
                scale = gates.minimum_safe_brake_scale(
                    lambda fraction: max(map(abs, predict_with_torque(
                        base + fraction * minimum_scale * (braking_endpoint - base)
                    ).values()), default=0.0),
                    SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S,
                    iterations=SIM_ONLY_VELOCITY_GUARD_BISECTION_STEPS,
                )
                if scale is None:
                    raise GateFailure(
                        f"Velocity guard failed to find a bounded command for {driver_name}"
                    )
                applied = base + scale * minimum_scale * (braking_endpoint - base)
                final_prediction = predict_with_torque(applied)
                commanded[channel]["applied_nm"] = float(applied)
                commanded[channel]["velocity_guard_brake_nm"] = float(base - applied)
                commanded[channel]["velocity_guard_minimum_safe_scale"] = float(scale)
                commanded[channel]["velocity_guard_minimum_peak_rad_s"] = float(minimum_peak)
                commanded[channel]["predicted_qvel_after_guard_rad_s"] = float(
                    final_prediction[driver_name]
                )
                commanded[channel]["predicted_qvel_chain_after_guard_rad_s"] = final_prediction.copy()
                commanded[channel]["saturated"] |= abs(
                    commanded[channel]["requested_nm"] - applied
                ) > 1e-12
        applied_peak = max(
            (abs(float(data.qfrc_applied[dof])) for dof in hand_driver_dofs.values()),
            default=0.0,
        )
        if applied_peak > SIM_ONLY_MAX_DRIVER_TORQUE_NM + 1e-9:
            raise GateFailure(
                f"Applied simulation-only effort exceeded hard cap: {applied_peak:.9g} Nm "
                f"> {SIM_ONLY_MAX_DRIVER_TORQUE_NM:.3f} Nm"
            )
        return {
            "command_time_s": float(data.time),
            "elapsed_since_load_start_s": float(elapsed_s),
            "feedforward_ramp_fraction": float(feedforward_ramp),
            "live_contact_allocation": live_allocation,
            "live_support_demand_n": (
                None if live_support_force_n is None else float(live_support_force_n)
            ),
            "live_support_target_rate_n_s": LIVE_SUPPORT_TARGET_RATE_N_S,
            "velocity_guard": {
                "mode": "forward-dynamics predicted torque braking; qvel is never directly clamped",
                "start_rad_s": SIM_ONLY_VELOCITY_GUARD_START_RAD_S,
                "predicted_max_rad_s": SIM_ONLY_VELOCITY_GUARD_PREDICTED_MAX_RAD_S,
                "active_channels": [
                    channel for channel, values in commanded.items()
                    if values["velocity_guard_brake_nm"] != 0.0
                ],
            },
            "channels": commanded,
        }

    def record_simulation_only_effort(command: dict, phase: str) -> None:
        contacts = bottle_contacts(model, data)
        force, torque = bottle_hand_wrench(contacts, np.asarray(data.xipos[bottle_id], dtype=float))
        friction_util, friction_by_contact = measured_friction_utilization(model, data, contacts)
        digit_forces = right_digit_true_normal_forces(contacts)
        table_normal = sum(
            c["true_normal_force_n"] for c in contacts
            if c["side"] == "other" and c["other_body"] == "m0_table"
        )
        row = {
            "time_s": float(data.time),
            "command_time_s": command["command_time_s"],
            "phase": phase,
            "elapsed_since_load_start_s": command["elapsed_since_load_start_s"],
            "feedforward_ramp_fraction": command["feedforward_ramp_fraction"],
            "hand_force_on_bottle_world_n": force.tolist(),
            "hand_torque_about_com_world_nm": torque.tolist(),
            "hand_vertical_support_force_n": float(force[2]),
            "table_normal_force_n": table_normal,
            "right_digit_true_normal_forces_n": digit_forces,
            "maximum_measured_friction_utilization": friction_util,
            "friction_utilization_by_contact": friction_by_contact,
            "live_support_demand_n": command.get("live_support_demand_n"),
            "live_support_target_rate_n_s": command.get("live_support_target_rate_n_s"),
            "live_contact_allocation_json": json.dumps(
                command.get("live_contact_allocation"), sort_keys=True
            ),
            "velocity_guard_active_channels": json.dumps(
                command["velocity_guard"]["active_channels"]
            ),
        }
        for channel, values in command["channels"].items():
            dof = int(values["dof"])
            row.update({
                f"{channel}_joint": values["joint"],
                f"{channel}_q_reference_rad": values["q_reference_rad"],
                f"{channel}_qpos_rad": joint_qpos(model, data, values["joint"]),
                f"{channel}_qvel_rad_s": joint_qvel(model, data, values["joint"]),
                f"{channel}_qacc_rad_s2": float(data.qacc[dof]),
                f"{channel}_feedforward_nm": values["feedforward_nm"],
                f"{channel}_impedance_nm": values["impedance_nm"],
                f"{channel}_bias_compensation_nm": values["bias_compensation_nm"],
                f"{channel}_requested_nm": values["requested_nm"],
                f"{channel}_applied_nm": float(data.qfrc_applied[dof]),
                f"{channel}_velocity_guard_brake_nm": values["velocity_guard_brake_nm"],
                f"{channel}_velocity_guard_minimum_safe_scale": values[
                    "velocity_guard_minimum_safe_scale"
                ],
                f"{channel}_predicted_qvel_before_guard_rad_s": values["predicted_qvel_before_guard_rad_s"],
                f"{channel}_predicted_qvel_after_guard_rad_s": values["predicted_qvel_after_guard_rad_s"],
                f"{channel}_predicted_qvel_chain_before_guard_json": json.dumps(
                    values.get("predicted_qvel_chain_before_guard_rad_s", {}), sort_keys=True
                ),
                f"{channel}_predicted_qvel_chain_after_guard_json": json.dumps(
                    values.get("predicted_qvel_chain_after_guard_rad_s", {}), sort_keys=True
                ),
                f"{channel}_saturated": int(values["saturated"]),
                f"{channel}_actuator_force_nm": float(data.actuator_force[hand_ids[channel]]),
            })
        effort_control_trace_rows.append(row)

    def sample(phase: str, command_u: float | None, arm_target: np.ndarray) -> None:
        nonlocal last_trace, last_video, hold_contact_steps
        nonlocal hold_table_support_steps, table_support_misses, diagnostic_sample_steps
        nonlocal max_hold_bottle_lift
        nonlocal max_force, max_penetration, max_mimic, max_mimic_diagnostic
        nonlocal mimic_diagnostic_exceed_steps, first_mimic_diagnostic_exceed
        nonlocal max_limit_detail, first_limit_breach
        nonlocal max_limit, max_qvel, max_qvel_joint
        nonlocal first_velocity_breach, left_contact_seen, thumb_seen, opposing_seen
        nonlocal maximum_contact_target_error_rad, maximum_hand_actuator_effort_nm
        nonlocal minimum_contact_target_error_rad, maximum_contact_commanded_error_rad
        contacts = bottle_contacts(model, data)
        friction_utilization, _ = measured_friction_utilization(model, data, contacts)
        if phase in {"OPEN", "CLOSE", "RELEASE"}:
            phase_motion = hand_motion_by_phase.setdefault(phase, {
                "start_time_s": float(data.time),
                "end_time_s": float(data.time),
                "channels": {},
            })
            phase_motion["end_time_s"] = float(data.time)
            for channel, joint in channel_joint_names.items():
                q = joint_qpos(model, data, joint)
                channel_motion = phase_motion["channels"].setdefault(channel, {
                    "joint": joint,
                    "first_qpos_rad": q,
                    "minimum_qpos_rad": q,
                    "maximum_qpos_rad": q,
                    "last_qpos_rad": q,
                    "sample_count": 0,
                })
                channel_motion["minimum_qpos_rad"] = min(
                    channel_motion["minimum_qpos_rad"], q
                )
                channel_motion["maximum_qpos_rad"] = max(
                    channel_motion["maximum_qpos_rad"], q
                )
                channel_motion["last_qpos_rad"] = q
                channel_motion["sample_count"] += 1
        thumb, opposing, left_contact, active_force = active_contact_sets(contacts)
        left_contact = left_contact or any(
            c["side"] in {"left_hand", "left_arm"} and c["distance_m"] <= 0.0
            for c in contacts
        )
        table_normal_force = sum(
            c["true_normal_force_n"] for c in contacts
            if c["side"] == "other" and c["other_body"] == "m0_table"
        )
        hand_bottle_normal_force = sum(
            c["true_normal_force_n"] for c in contacts if c["side"] == "right_hand"
        )
        hand_force_world, hand_torque_world = bottle_hand_wrench(
            contacts, np.asarray(data.xipos[bottle_id], dtype=float)
        )
        table_support = any(
            c["side"] == "other" and c["other_body"] == "m0_table"
            and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
            for c in contacts
        )
        thumb_seen |= thumb
        opposing_seen |= opposing
        left_contact_seen |= left_contact
        max_force = max(max_force, active_force)
        penetration = max((max(0.0, -c["distance_m"]) for c in contacts if c["side"] != "other"), default=0.0)
        max_penetration = max(max_penetration, penetration)
        state = current_joint_state(model, data, mimics)
        max_mimic = max(max_mimic, state["max_mimic"])
        max_mimic_diagnostic = max(max_mimic_diagnostic, state["max_mimic"])
        if state["max_mimic"] > MIMIC_DIAGNOSTIC_TOLERANCE_RAD:
            mimic_diagnostic_exceed_steps += 1
            if first_mimic_diagnostic_exceed is None:
                first_mimic_diagnostic_exceed = {
                    "time_s": float(data.time), "max_mimic_error_rad": state["max_mimic"]
                }
        current_limit_violation, current_limit_detail = max_hand_limit_violation(model, data, limits)
        if current_limit_violation > max_limit:
            max_limit = current_limit_violation
            max_limit_detail = current_limit_detail
        if current_limit_violation > JOINT_LIMIT_TOLERANCE_RAD and first_limit_breach is None:
            first_limit_breach = {"time_s": float(data.time), **(current_limit_detail or {})}
        for name in limits:
            if not name.startswith(("L_", "R_")):
                continue
            qd = joint_qvel(model, data, name)
            speed = abs(qd)
            if speed > max_qvel:
                max_qvel, max_qvel_joint = speed, name
            if speed > MAX_HAND_QVEL_RAD_S and first_velocity_breach is None:
                jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
                dof = int(model.jnt_dofadr[jid])
                channel = next(
                    (key for key, joint in channel_joint_names.items() if joint == name), None
                )
                chain = (
                    hand_velocity_chain_dofs.get(channel, {}) if channel is not None else {}
                )
                first_velocity_breach = {
                    "time_s": float(data.time),
                    "phase": phase,
                    "joint": name,
                    "channel": channel,
                    "qpos_rad": joint_qpos(model, data, name),
                    "qvel_rad_s": qd,
                    "qacc_rad_s2": float(data.qacc[dof]),
                    "qfrc_applied_nm": float(data.qfrc_applied[dof]),
                    "actuator_force_nm": (
                        float(data.actuator_force[hand_ids[channel]])
                        if channel is not None else None
                    ),
                    "velocity_chain": {
                        joint_name: {
                            "qpos_rad": joint_qpos(model, data, joint_name),
                            "qvel_rad_s": float(data.qvel[chain_dof]),
                            "qacc_rad_s2": float(data.qacc[chain_dof]),
                            "qfrc_applied_nm": float(data.qfrc_applied[chain_dof]),
                        }
                        for joint_name, chain_dof in chain.items()
                    },
                    "contacts": contacts,
                    "load_build_command_pre_step": (
                        result.get("last_load_build_command_pre_step")
                        if phase == "LOAD_BUILD" else None
                    ),
                    "constraint_force_count": int(data.nefc),
                    "maximum_absolute_constraint_force": (
                        float(np.max(np.abs(data.efc_force[:data.nefc])))
                        if data.nefc else 0.0
                    ),
                }
                result["first_velocity_breach"] = first_velocity_breach
        for channel, aid in hand_ids.items():
            effort = abs(float(data.actuator_force[aid]))
            maximum_hand_actuator_effort_nm = max(maximum_hand_actuator_effort_nm, effort)
            if effort > HAND_MAX_TORQUE + 1e-6:
                raise GateFailure(
                    f"Hand actuator effort exceeded its existing bound in {phase}: "
                    f"{channel}={effort:.9g} Nm > {HAND_MAX_TORQUE:.3f} Nm"
                )
        if phase in {"LOAD_BUILD", "LIFT"} and result.get("load_capacity_audit"):
            right_hand_contacts = [c for c in contacts if c["side"] == "right_hand"]
            total_contact_resultant = sum(c["normal_force_n"] for c in right_hand_contacts)
            max_contact_resultant = max(
                (c["normal_force_n"] for c in right_hand_contacts), default=0.0
            )
            current_hand_effort_nm = max(
                (abs(float(data.actuator_force[aid])) for aid in hand_ids.values()), default=0.0
            )
            bound_failure = None
            if current_hand_effort_nm > LOAD_BUILD_MAX_ACTUATOR_EFFORT_NM + 1e-6:
                bound_failure = f"driver effort {current_hand_effort_nm:.9g} Nm exceeded {LOAD_BUILD_MAX_ACTUATOR_EFFORT_NM:.3f} Nm"
            elif max_contact_resultant > LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:
                bound_failure = f"single contact resultant {max_contact_resultant:.9g} N exceeded {LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:.3f} N"
            elif total_contact_resultant > LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:
                bound_failure = f"total hand contact resultant {total_contact_resultant:.9g} N exceeded {LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:.3f} N"
            elif hand_bottle_normal_force > LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:
                bound_failure = f"total hand normal force {hand_bottle_normal_force:.9g} N exceeded {LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:.3f} N"
            if bound_failure:
                result["load_capacity_first_bound_failure"] = {
                    "time_s": float(data.time), "phase": phase,
                    "reason": bound_failure,
                    "maximum_hand_actuator_effort_nm": current_hand_effort_nm,
                    "maximum_single_contact_resultant_n": max_contact_resultant,
                    "total_hand_contact_resultant_n": total_contact_resultant,
                    "total_hand_true_normal_force_n": hand_bottle_normal_force,
                    "contacts": right_hand_contacts,
                }
                raise GateFailure(f"Load-capacity safe bound failed in {phase}: {bound_failure}")
        if phase in {"CLOSE", "HOLD", "LIFT_ARMING", "LIVE_CONTACT_ALLOCATION_ARMING",
                     "LOAD_BUILD", "LIFT", "POST_LIFT_HOLD"}:
            right_hand_contacts = [c for c in contacts if c["side"] == "right_hand"]
            total_normal = float(sum(c["true_normal_force_n"] for c in right_hand_contacts))
            total_resultant = float(sum(c["normal_force_n"] for c in right_hand_contacts))
            single_resultant = float(max(
                (c["normal_force_n"] for c in right_hand_contacts), default=0.0
            ))
            applied_effort = float(max(
                (abs(float(data.qfrc_applied[dof])) for dof in hand_driver_dofs.values()),
                default=0.0,
            ))
            actuator_effort = float(max(
                (abs(float(data.actuator_force[aid])) for aid in hand_ids.values()),
                default=0.0,
            ))
            effort_peak = max(applied_effort, actuator_effort)
            cap_state = result.setdefault("contact_force_cap_gate", {
                "passed": True,
                "samples": 0,
                "thresholds": {
                    "total_normal_force_n": LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N,
                    "single_contact_resultant_n": LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N,
                    "total_contact_resultant_n": LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N,
                    "driver_effort_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                },
                "maxima": {},
                "first_failure": None,
            })
            cap_state["samples"] += 1
            measured_maxima = {
                "total_normal_force_n": total_normal,
                "single_contact_resultant_n": single_resultant,
                "total_contact_resultant_n": total_resultant,
                "driver_effort_nm": effort_peak,
            }
            for name, value in measured_maxima.items():
                cap_state["maxima"][name] = max(
                    float(cap_state["maxima"].get(name, 0.0)), value
                )
            cap_failure = None
            if total_normal > LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N + 1e-9:
                cap_failure = "total_hand_normal_force_exceeded"
            elif single_resultant > LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N + 1e-9:
                cap_failure = "single_contact_resultant_exceeded"
            elif total_resultant > LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N + 1e-9:
                cap_failure = "total_contact_resultant_exceeded"
            elif effort_peak > SIM_ONLY_MAX_DRIVER_TORQUE_NM + 1e-9:
                cap_failure = "simulation_only_driver_effort_exceeded"
            if cap_failure:
                cap_state["passed"] = False
                cap_state["first_failure"] = {
                    "time_s": float(data.time),
                    "phase": phase,
                    "reason": cap_failure,
                    "measurements": measured_maxima,
                    "contacts": right_hand_contacts,
                }
                events.append({"event": "CONTACT_FORCE_CAP_FAILURE", **cap_state["first_failure"]})
                raise GateFailure(
                    f"Contact/effort hard cap failed in {phase}: "
                    + json.dumps(cap_state["first_failure"], sort_keys=True)
                )
        if phase in {"PREGRASP", "OPEN", "APPROACH", "CLOSE", "HOLD"}:
            diagnostic_sample_steps += 1
            table_support_misses += int(not table_support)
        if phase == "HOLD":
            hold_contact_steps += int(thumb and opposing)
            hold_table_support_steps += int(table_support)
            max_hold_bottle_lift = max(
                max_hold_bottle_lift,
                float(data.xpos[bottle_id][2] - initial_bottle_pos[2]),
            )
        result.update({
            "physics_steps": steps, "maximum_hand_joint_qvel_rad_s": max_qvel,
            "maximum_hand_joint_qvel_joint": max_qvel_joint,
            "first_velocity_breach": first_velocity_breach,
            "maximum_mimic_error_rad": max_mimic, "maximum_joint_limit_violation_rad": max_limit,
            "maximum_joint_limit_violation": max_limit_detail,
            "first_joint_limit_breach": first_limit_breach,
            "maximum_mimic_diagnostic_error_rad": max_mimic_diagnostic,
            "mimic_diagnostic_threshold_exceed_steps": mimic_diagnostic_exceed_steps,
            "first_mimic_diagnostic_threshold_exceed": first_mimic_diagnostic_exceed,
            "maximum_contact_preload_target_error_rad": maximum_contact_target_error_rad,
            "minimum_contact_preload_target_error_rad": (
                None if not math.isfinite(minimum_contact_target_error_rad)
                else minimum_contact_target_error_rad
            ),
            "maximum_contact_commanded_error_rad": maximum_contact_commanded_error_rad,
            "maximum_hand_actuator_effort_nm": maximum_hand_actuator_effort_nm,
            "maximum_object_contact_force_n": max_force, "maximum_object_penetration_m": max_penetration,
            "current_table_normal_force_n": table_normal_force,
            "current_hand_normal_force_sum_n": hand_bottle_normal_force,
            "current_hand_force_on_bottle_world_n": hand_force_world.tolist(),
            "current_hand_vertical_support_force_n": float(hand_force_world[2]),
            "current_hand_torque_about_com_world_nm": hand_torque_world.tolist(),
            "right_thumb_contact_seen": thumb_seen, "right_opposing_digit_contact_seen": opposing_seen,
            "left_hand_bottle_contact": left_contact_seen,
            "hold_valid_contact_steps": hold_contact_steps,
            "hold_table_support_steps": hold_table_support_steps,
            "diagnostic_table_support_misses": table_support_misses,
        })
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all():
            raise GateFailure(f"Non-finite state during {phase}")
        if max_mimic > MIMIC_MANIPULATION_LIMIT_RAD:
            raise GateFailure(
                f"Mimic manipulation gate exceeded in {phase}: {max_mimic:.9g} rad "
                f"> {MIMIC_MANIPULATION_LIMIT_RAD:.3f} rad"
            )
        if max_mimic_diagnostic > MIMIC_DIAGNOSTIC_TOLERANCE_RAD:
            raise GateFailure(
                f"Mimic diagnostic gate exceeded in {phase}: {max_mimic_diagnostic:.9g} rad "
                f"> {MIMIC_DIAGNOSTIC_TOLERANCE_RAD:.3f} rad"
            )
        if max_limit > JOINT_LIMIT_TOLERANCE_RAD:
            raise GateFailure(
                f"Hand limit gate exceeded in {phase}: {max_limit:.9g} rad; "
                f"first={first_limit_breach}, maximum={max_limit_detail}"
            )
        if first_velocity_breach:
            raise GateFailure(f"Source velocity gate exceeded in {phase}: {first_velocity_breach}")
        if active_force > MAX_CONTACT_FORCE_N:
            raise GateFailure(f"Object contact force gate exceeded: {active_force:.6g} N")
        if left_contact:
            raise GateFailure("Left hand/arm contacted the bottle")
        if phase in {"SETTLE", "PREGRASP", "OPEN"}:
            robot_contacts = [c for c in contacts if c["side"] != "other"]
            if robot_contacts:
                raise GateFailure(f"Robot contacted bottle during {phase}: {robot_contacts}")

        for c in contacts:
            contact_rows.append({
                "time_s": f"{data.time:.8f}", "phase": phase, "bottle_geom": c["bottle_geom"],
                "other_geom": c["other_geom"], "other_body": c["other_body"],
                "side": c["side"], "digit": c["digit"] or "",
                "distance_m": f"{c['distance_m']:.9f}", "normal_force_n": f"{c['normal_force_n']:.8f}",
            })
        mimic_rows.append({
            "time_s": f"{data.time:.8f}", "phase": phase,
            "max_mimic_error_rad": f"{state['max_mimic']:.9f}",
            "diagnostic_threshold_rad": f"{MIMIC_DIAGNOSTIC_TOLERANCE_RAD:.6f}",
            "manipulation_limit_rad": f"{MIMIC_MANIPULATION_LIMIT_RAD:.6f}",
            "diagnostic_threshold_exceeded": int(state["max_mimic"] > MIMIC_DIAGNOSTIC_TOLERANCE_RAD),
            **{f"{name}_mimic_error_rad": f"{item['error']:.9f}"
               for name, item in state["followers"].items() if name.startswith("R_")},
        })
        digit_forces = right_digit_contact_forces(contacts)
        digit_normal_forces = right_digit_true_normal_forces(contacts)
        digit_geometry = right_digit_contact_geometry(contacts)
        if phase in {"HOLD", "LIFT_ARMING", "LIFT", "POST_LIFT_HOLD"}:
            contact_window_history.append({
                "time_s": float(data.time), "forces_n": digit_forces,
                "controller_states": contact_controller_state.copy(),
                "left_contact": bool(left_contact),
            })
        window_metric = contact_window_summary(list(contact_window_history))
        if phase in {"HOLD", "LIFT_ARMING", "LIFT", "POST_LIFT_HOLD"}:
            contact_window_rows.append({
                "time_s": float(data.time), "phase": phase,
                "window_valid": int(window_metric["valid"]),
                "window_failure_reason": window_metric.get("failure_reason", ""),
                "window_metrics_json": json.dumps(window_metric, sort_keys=True),
            })
        digit_force_rows.append({
            "time_s": f"{data.time:.8f}", "phase": phase,
            "normalized_close_command": "" if command_u is None else f"{command_u:.8f}",
            "table_supported": int(table_support),
            **{f"{digit}_normal_force_n": f"{force:.8f}"
               for digit, force in digit_forces.items()},
            **{f"{digit}_true_normal_force_n": f"{force:.8f}"
               for digit, force in digit_normal_forces.items()},
        })
        in_hold_window = (
            phase == "HOLD" and transition_hold_start is not None
            and data.time >= transition_hold_start + HOLD_SECONDS - 0.0500001
        )
        in_lift_window = (
            phase == "LIFT" and transition_lift_start is not None
            and data.time <= transition_lift_start + 0.1000001
        )
        if in_hold_window or in_lift_window:
            target_wrist_pos, target_wrist_quat, target_hand_site = forward_wrist_target(
                model, fk_data, arm_qadr, arm_target, wrist_id, site_id
            )
            object_velocity = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, object_velocity, 0
            )
            transition_row = {
                "time_s": float(data.time), "phase": phase,
                "window": "last_50ms_hold" if in_hold_window else "first_100ms_lift",
                "normalized_close_command": "" if command_u is None else float(command_u),
                "controller_states": json.dumps(contact_controller_state, sort_keys=True),
                "right_thumb_active_force_contact": int(digit_forces["thumb"] > 0.01),
                "right_thumb_geom_overlap": int(any(
                    c["side"] == "right_hand" and c["digit"] == "thumb"
                    and c["distance_m"] <= 0.0 for c in contacts
                )),
                "right_opposing_active_force_contact": int(any(
                    digit_forces[digit] > 0.01 for digit in ("index", "middle", "ring", "pinky")
                )),
                "left_bottle_contact": int(left_contact),
                "bottle_table_supported": int(table_support),
                "table_normal_force_n": table_normal_force,
                "total_hand_bottle_normal_force_n": hand_bottle_normal_force,
                "hand_vertical_support_force_n": float(hand_force_world[2]),
                "hand_force_on_bottle_world_n": hand_force_world.tolist(),
                "bottle_weight_n": float(model.body_mass[bottle_id] * abs(model.opt.gravity[2])),
                "table_weight_fraction": table_normal_force / max(
                    float(model.body_mass[bottle_id] * abs(model.opt.gravity[2])), 1e-12
                ),
                "right_digit_contact_forces_n": json.dumps(digit_forces, sort_keys=True),
                "right_digit_contact_geometry_json": json.dumps(digit_geometry, sort_keys=True),
                "bottle_x_m": float(data.xpos[bottle_id][0]),
                "bottle_y_m": float(data.xpos[bottle_id][1]),
                "bottle_z_m": float(data.xpos[bottle_id][2]),
                "bottle_quat_w": float(data.xquat[bottle_id][0]),
                "bottle_quat_x": float(data.xquat[bottle_id][1]),
                "bottle_quat_y": float(data.xquat[bottle_id][2]),
                "bottle_quat_z": float(data.xquat[bottle_id][3]),
                "bottle_angular_velocity_world_rad_s": object_velocity[:3].tolist(),
                "bottle_linear_velocity_world_m_s": object_velocity[3:].tolist(),
                "right_wrist_target_x_m": float(target_wrist_pos[0]),
                "right_wrist_target_y_m": float(target_wrist_pos[1]),
                "right_wrist_target_z_m": float(target_wrist_pos[2]),
                "right_wrist_target_quat_w": float(target_wrist_quat[0]),
                "right_wrist_target_quat_x": float(target_wrist_quat[1]),
                "right_wrist_target_quat_y": float(target_wrist_quat[2]),
                "right_wrist_target_quat_z": float(target_wrist_quat[3]),
                "right_wrist_actual_x_m": float(data.xpos[wrist_id][0]),
                "right_wrist_actual_y_m": float(data.xpos[wrist_id][1]),
                "right_wrist_actual_z_m": float(data.xpos[wrist_id][2]),
                "right_wrist_actual_quat_w": float(data.xquat[wrist_id][0]),
                "right_wrist_actual_quat_x": float(data.xquat[wrist_id][1]),
                "right_wrist_actual_quat_y": float(data.xquat[wrist_id][2]),
                "right_wrist_actual_quat_z": float(data.xquat[wrist_id][3]),
                "right_wrist_position_error_m": float(np.linalg.norm(
                    target_wrist_pos - data.xpos[wrist_id]
                )),
                "right_hand_target_site_x_m": float(target_hand_site[0]),
                "right_hand_target_site_y_m": float(target_hand_site[1]),
                "right_hand_target_site_z_m": float(target_hand_site[2]),
                "right_hand_actual_site_x_m": float(data.site_xpos[site_id][0]),
                "right_hand_actual_site_y_m": float(data.site_xpos[site_id][1]),
                "right_hand_actual_site_z_m": float(data.site_xpos[site_id][2]),
                "contact_window_valid": int(window_metric["valid"]),
                "contact_window_metrics_json": json.dumps(window_metric, sort_keys=True),
            }
            target_quat = np.asarray(target_wrist_quat, dtype=float)
            actual_quat = np.asarray(data.xquat[wrist_id], dtype=float)
            transition_row["right_wrist_rotation_error_rad"] = float(
                2.0 * math.acos(min(1.0, abs(float(np.dot(target_quat, actual_quat)))))
            )
            for index, name in enumerate(ARM_NAMES):
                jid = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
                transition_row[f"{name}_target_rad"] = float(arm_target[index])
                transition_row[f"{name}_actual_rad"] = float(data.qpos[qadr])
                transition_row[f"{name}_qvel_rad_s"] = float(data.qvel[dof])
                transition_row[f"{name}_qacc_rad_s2"] = float(data.qacc[dof])
            for name, (qadr, dof) in right_joint_meta.items():
                transition_row[f"{name}_qpos_rad"] = float(data.qpos[qadr])
                transition_row[f"{name}_qvel_rad_s"] = float(data.qvel[dof])
                transition_row[f"{name}_qacc_rad_s2"] = float(data.qacc[dof])
            for name, item in state["followers"].items():
                if name.startswith("R_"):
                    transition_row[f"{name}_mimic_error_rad"] = float(item["error"])
            for channel, aid in hand_ids.items():
                digit = channel_digits[channel]
                joint = channel_joint_names[channel]
                qadr, dof = right_joint_meta[joint]
                target = float(data.ctrl[aid])
                actual = float(data.qpos[qadr])
                direction = closure_directions[channel]
                transition_row.update({
                    f"{channel}_mode": contact_controller_state[digit],
                    f"{channel}_command_target_rad": target,
                    f"{channel}_last_applied_target_rad": float(last_applied_targets[channel]),
                    f"{channel}_actual_rad": actual,
                    f"{channel}_closure_direction_error_rad": direction * (target - actual),
                    f"{channel}_qvel_rad_s": float(data.qvel[dof]),
                    f"{channel}_qacc_rad_s2": float(data.qacc[dof]),
                    f"{channel}_actuator_force_nm": float(data.actuator_force[aid]),
                    f"{channel}_contact_force_n": float(digit_forces[digit]),
                    f"{channel}_minimum_contact_distance_m": min((
                        c["distance_m"] for c in contacts
                        if c["side"] == "right_hand" and c["digit"] == digit
                    ), default=""),
                    f"{channel}_contact_transition_time_s": contact_hold_transition_times[channel] or "",
                })
            previous_row = transition_trace_rows[-1] if transition_trace_rows else None
            for name in ARM_NAMES:
                target = transition_row[f"{name}_target_rad"]
                if previous_row is None:
                    target_velocity = 0.0
                    target_acceleration = 0.0
                else:
                    target_velocity = (
                        target - previous_row[f"{name}_target_rad"]
                    ) / DT
                    target_acceleration = (
                        target_velocity - previous_row[f"{name}_target_velocity_rad_s"]
                    ) / DT
                transition_row[f"{name}_target_velocity_rad_s"] = target_velocity
                transition_row[f"{name}_target_acceleration_rad_s2"] = target_acceleration
            for axis in "xyz":
                key = f"right_wrist_target_{axis}_m"
                target = transition_row[key]
                if previous_row is None:
                    target_velocity = 0.0
                    target_acceleration = 0.0
                else:
                    target_velocity = (target - previous_row[key]) / DT
                    target_acceleration = (
                        target_velocity - previous_row[f"right_wrist_target_{axis}_velocity_m_s"]
                    ) / DT
                transition_row[f"right_wrist_target_{axis}_velocity_m_s"] = target_velocity
                transition_row[f"right_wrist_target_{axis}_acceleration_m_s2"] = target_acceleration
            transition_trace_rows.append(transition_row)
        if phase in {"LIFT", "POST_LIFT_HOLD"}:
            target_wrist_pos, target_wrist_quat, target_hand_site = forward_wrist_target(
                model, fk_data, arm_qadr, arm_target, wrist_id, site_id
            )
            object_velocity = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, object_velocity, 0
            )
            bottle_weight = float(model.body_mass[bottle_id] * abs(model.opt.gravity[2]))
            load_transfer_rows.append({
                "time_s": float(data.time), "phase": phase,
                "lift_target_height_m": lift_height_m,
                "table_normal_force_n": table_normal_force,
                "table_weight_fraction": table_normal_force / max(bottle_weight, 1e-12),
                "bottle_weight_n": bottle_weight,
                "total_hand_bottle_normal_force_n": hand_bottle_normal_force,
                "hand_force_on_bottle_world_n": hand_force_world.tolist(),
                "hand_vertical_support_force_n": float(hand_force_world[2]),
                "hand_torque_about_com_world_nm": hand_torque_world.tolist(),
                "hand_supported_fraction": 1.0 - table_normal_force / max(bottle_weight, 1e-12),
                "right_thumb_true_normal_force_n": digit_normal_forces["thumb"],
                "right_opposing_true_normal_force_n": sum(
                    digit_normal_forces[digit] for digit in ("index", "middle", "ring", "pinky")
                ),
                "stage": current_load_stage_name,
                "table_supported": int(table_support),
                "left_hand_contact": int(left_contact),
                "horizontal_hand_force_n": float(np.linalg.norm(hand_force_world[:2])),
                "hand_torque_norm_nm": float(np.linalg.norm(hand_torque_world)),
                "maximum_friction_utilization": friction_utilization,
                "maximum_hand_qvel_rad_s": max_qvel,
                "maximum_applied_driver_effort_nm": float(max(
                    (abs(float(data.qfrc_applied[dof])) for dof in hand_driver_dofs.values()),
                    default=0.0,
                )),
                "maximum_mimic_error_rad": state["max_mimic"],
                "contact_window_valid": int(window_metric["valid"]),
                "bottle_x_m": float(data.xpos[bottle_id][0]),
                "bottle_y_m": float(data.xpos[bottle_id][1]),
                "bottle_z_m": float(data.xpos[bottle_id][2]),
                "bottle_quaternion_wxyz": np.asarray(data.xquat[bottle_id], dtype=float).tolist(),
                "bottle_z_velocity_m_s": float(object_velocity[5]),
                "bottle_linear_velocity_world_m_s": json.dumps(object_velocity[3:].tolist()),
                "bottle_angular_velocity_world_rad_s": json.dumps(object_velocity[:3].tolist()),
                "right_wrist_target_z_m": float(target_wrist_pos[2]),
                "right_wrist_actual_z_m": float(data.xpos[wrist_id][2]),
                "right_wrist_rise_from_load_ready_m": (
                    float(data.xpos[wrist_id][2] - load_build_stage3_wrist_start[2])
                    if load_build_stage3_wrist_start is not None else 0.0
                ),
                "right_hand_target_site_z_m": float(target_hand_site[2]),
                "right_hand_actual_site_z_m": float(data.site_xpos[site_id][2]),
                "right_digit_contact_forces_n": json.dumps(digit_forces, sort_keys=True),
                "right_digit_true_normal_forces_n": json.dumps(digit_normal_forces, sort_keys=True),
                "right_digit_contact_geometry_json": json.dumps(digit_geometry, sort_keys=True),
                "force_bearing_contact_details_json": json.dumps(
                    force_bearing_contact_details(model, data, contacts), sort_keys=True
                ),
                "controller_states_json": json.dumps(contact_controller_state, sort_keys=True),
                "contact_window_valid": int(window_metric["valid"]),
                "contact_window_failure_reason": window_metric.get("failure_reason", ""),
                "contact_window_metrics_json": json.dumps(window_metric, sort_keys=True),
                "left_bottle_contact": int(left_contact),
                "bottle_table_supported": int(table_support),
            })
        if phase in {"CLOSE", "HOLD", "LIFT_ARMING"}:
            aware_row = {
                "time_s": f"{data.time:.8f}", "phase": phase,
                "normalized_close_command": f"{command_u:.8f}" if command_u is not None else "",
                "contact_trigger_force_n": f"{CONTACT_TRIGGER_FORCE_N:.6f}",
                "contact_confirm_samples": CONTACT_CONFIRM_SAMPLES,
            }
            for channel, aid in hand_ids.items():
                joint = channel_joint_names[channel]
                actual = joint_qpos(model, data, joint)
                target = float(data.ctrl[aid])
                direction = closure_directions[channel]
                target_error = direction * (target - actual)
                applied_target = last_applied_targets[channel]
                applied_target_error = direction * (applied_target - actual)
                if contact_controller_state[channel_digits[channel]] == "CONTACT_PRELOAD":
                    maximum_contact_target_error_rad = max(
                        maximum_contact_target_error_rad, target_error
                    )
                    minimum_contact_target_error_rad = min(
                        minimum_contact_target_error_rad, target_error
                    )
                    maximum_contact_commanded_error_rad = max(
                        maximum_contact_commanded_error_rad,
                        contact_commanded_errors[channel],
                    )
                    if (
                        target_error < -CONTACT_PRELOAD_GATE_TOLERANCE_RAD
                        or target_error > current_preload_cap_rad
                        + CONTACT_PRELOAD_GATE_TOLERANCE_RAD
                    ):
                        digit = channel_digits[channel]
                        result["first_contact_preload_cap_breach"] = {
                            "time_s": float(data.time),
                            "channel": channel,
                            "joint": joint,
                            "controller_state": contact_controller_state[digit],
                            "target_rad": target,
                            "applied_target_rad": applied_target,
                            "qpos_rad": actual,
                            "closure_direction": direction,
                            "closure_direction_error_rad": target_error,
                            "applied_closure_direction_error_rad": applied_target_error,
                            "cap_rad": current_preload_cap_rad,
                            "gate_tolerance_rad": CONTACT_PRELOAD_GATE_TOLERANCE_RAD,
                            "actuator_effort_nm": float(data.actuator_force[aid]),
                            "digit_contact_force_n": right_digit_contact_forces(contacts)[digit],
                            "digit_contacts": [
                                c for c in contacts
                                if c["side"] == "right_hand" and c["digit"] == digit
                            ],
                            "bottle_position_world_m": data.xpos[bottle_id].tolist(),
                            "bottle_table_supported": table_support,
                        }
                        raise GateFailure(
                            f"Contact preload measured error gate failed: {channel} "
                            f"target={target:.9g} rad, qpos={actual:.9g} rad, "
                            f"directional_error={target_error:.9g} rad outside "
                            f"[-{CONTACT_PRELOAD_GATE_TOLERANCE_RAD:.1g}, "
                            f"{current_preload_cap_rad + CONTACT_PRELOAD_GATE_TOLERANCE_RAD:.9g}] rad"
                        )
                aware_row.update({
                    f"{channel}_mode": contact_controller_state[channel_digits[channel]],
                    f"{channel}_target_rad": f"{target:.9f}",
                    f"{channel}_applied_target_rad": f"{applied_target:.9f}",
                    f"{channel}_actual_rad": f"{actual:.9f}",
                    f"{channel}_target_error_rad": f"{target_error:.9f}",
                    f"{channel}_applied_target_error_rad": f"{applied_target_error:.9f}",
                    f"{channel}_qvel_rad_s": f"{joint_qvel(model, data, joint):.9f}",
                    f"{channel}_actuator_force_nm": f"{data.actuator_force[aid]:.9f}",
                    f"{channel}_bottle_contact_force_n": f"{digit_forces[channel_digits[channel]]:.9f}",
                    f"{channel}_transition_time_s": (
                        "" if contact_hold_transition_times[channel] is None
                        else f"{contact_hold_transition_times[channel]:.8f}"
                    ),
                    f"{channel}_preload_error_cap_rad": current_preload_cap_rad,
                })
            aware_row.update({
                "bottle_x_m": f"{data.xpos[bottle_id][0]:.9f}",
                "bottle_y_m": f"{data.xpos[bottle_id][1]:.9f}",
                "bottle_z_m": f"{data.xpos[bottle_id][2]:.9f}",
                "bottle_quat_w": f"{data.xquat[bottle_id][0]:.9f}",
                "bottle_quat_x": f"{data.xquat[bottle_id][1]:.9f}",
                "bottle_quat_y": f"{data.xquat[bottle_id][2]:.9f}",
                "bottle_quat_z": f"{data.xquat[bottle_id][3]:.9f}",
                "maximum_contact_target_error_rad": f"{maximum_contact_target_error_rad:.9f}",
                "minimum_contact_target_error_rad": (
                    "" if not math.isfinite(minimum_contact_target_error_rad)
                    else f"{minimum_contact_target_error_rad:.9f}"
                ),
                "maximum_contact_commanded_error_rad": f"{maximum_contact_commanded_error_rad:.9f}",
                "maximum_hand_actuator_effort_nm": f"{maximum_hand_actuator_effort_nm:.9f}",
            })
            contact_aware_rows.append(aware_row)
        bottle_pos = data.xpos[bottle_id].copy()
        if data.time - last_trace >= TRACE_PERIOD_S:
            last_trace = float(data.time)
            row = {"time_s": f"{data.time:.8f}", "phase": phase,
                   "normalized_close_command": "" if command_u is None else f"{command_u:.8f}",
                   "right_thumb_contact": int(thumb), "right_opposing_contact": int(opposing),
                   "left_object_contact": int(left_contact),
                   "bottle_table_support": int(table_support)}
            for channel, aid in hand_ids.items():
                row[f"{channel}_target_rad"] = f"{data.ctrl[aid]:.9f}"
                row[f"{channel}_force_nm"] = f"{data.actuator_force[aid]:.9f}"
            for name, item in state["drivers"].items():
                row[f"{name}_qpos_rad"] = f"{item['q']:.9f}"
                row[f"{name}_qvel_rad_s"] = f"{item['qd']:.9f}"
            for name, item in state["followers"].items():
                if name.startswith("R_"):
                    row[f"{name}_qpos_rad"] = f"{item['q']:.9f}"
                    row[f"{name}_qvel_rad_s"] = f"{item['qd']:.9f}"
                    row[f"{name}_mimic_error_rad"] = f"{item['error']:.9f}"
            for i, name in enumerate(ARM_NAMES):
                row[f"{name}_qpos_rad"] = f"{joint_qpos(model, data, name):.9f}"
                row[f"{name}_qvel_rad_s"] = f"{joint_qvel(model, data, name):.9f}"
                row[f"{name}_target_rad"] = f"{arm_target[i]:.9f}"
            row["max_mimic_error_rad"] = f"{state['max_mimic']:.9f}"
            hand_rows.append(row)
            bottle_rows.append({
                "time_s": f"{data.time:.8f}", "phase": phase,
                "x_m": f"{bottle_pos[0]:.9f}", "y_m": f"{bottle_pos[1]:.9f}",
                "z_m": f"{bottle_pos[2]:.9f}",
                "bottle_z_from_spawn_m": f"{bottle_pos[2]-spawn_bottle_pos[2]:.9f}",
                "lift_from_settled_m": ("" if settled_bottle_pos is None else
                                         f"{bottle_pos[2]-settled_bottle_pos[2]:.9f}"),
                "quat_w": f"{data.xquat[bottle_id][0]:.9f}",
                "quat_x": f"{data.xquat[bottle_id][1]:.9f}",
                "quat_y": f"{data.xquat[bottle_id][2]:.9f}",
                "quat_z": f"{data.xquat[bottle_id][3]:.9f}",
                "right_hand_site_xyz_m": ",".join(f"{x:.8f}" for x in data.site_xpos[site_id]),
                "left_hand_site_xyz_m": ",".join(f"{x:.8f}" for x in data.site_xpos[left_site]),
            })
            arm_rows.append({
                "time_s": f"{data.time:.8f}", "phase": phase,
                **{f"{name}_qpos_rad": f"{joint_qpos(model, data, name):.9f}" for name in ARM_NAMES},
                **{f"{name}_target_rad": f"{arm_target[i]:.9f}" for i, name in enumerate(ARM_NAMES)},
            })
        if data.time - last_video >= VIDEO_PERIOD_S:
            last_video = float(data.time)
            writer.write(render_frame(renderer, model, data, camera, phase))

    try:
        output.mkdir(parents=True, exist_ok=True)
        bottle_joint = element_id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
        bottle_dof = int(model.jnt_dofadr[bottle_joint])
        settle_start = float(data.time)
        settle_dwell_start = None
        settle_dwell_max_linear = 0.0
        settle_dwell_max_angular = 0.0
        settle_max_linear = 0.0
        settle_max_angular = 0.0
        settle_supported_steps = 0
        events.append({"event": "SETTLE_START", "time_s": settle_start,
                       "arm_clear_target_rad": settle_clear_q.tolist(),
                       "maximum_duration_s": SETTLE_MAX_SECONDS,
                       "stable_dwell_s": SETTLE_DWELL_SECONDS,
                       "linear_speed_limit_m_s": SETTLE_MAX_LINEAR_SPEED_M_S,
                       "angular_speed_limit_rad_s": SETTLE_MAX_ANGULAR_SPEED_RAD_S})
        settled = False
        settle_steps = int(round(SETTLE_MAX_SECONDS / DT))
        for _ in range(settle_steps):
            set_body_ctrl(arm_motor_targets(settle_clear_q))
            mujoco.mj_step(model, data)
            steps += 1
            contacts = bottle_contacts(model, data)
            robot_contacts = [c for c in contacts if c["side"] != "other"]
            if robot_contacts:
                raise GateFailure(f"Robot contacted bottle during SETTLE: {robot_contacts}")
            sample("SETTLE", None, settle_clear_q)
            table_supported = any(
                c["side"] == "other" and c["other_body"] == "m0_table"
                and c["distance_m"] <= 0.0 and c["normal_force_n"] > 0.01
                for c in contacts
            )
            linear_speed = float(np.linalg.norm(data.qvel[bottle_dof:bottle_dof + 3]))
            angular_speed = float(np.linalg.norm(data.qvel[bottle_dof + 3:bottle_dof + 6]))
            settle_max_linear = max(settle_max_linear, linear_speed)
            settle_max_angular = max(settle_max_angular, angular_speed)
            settle_supported_steps += int(table_supported)
            stable_now = (
                table_supported
                and linear_speed <= SETTLE_MAX_LINEAR_SPEED_M_S
                and angular_speed <= SETTLE_MAX_ANGULAR_SPEED_RAD_S
            )
            if stable_now:
                if settle_dwell_start is None:
                    settle_dwell_start = float(data.time)
                    settle_dwell_max_linear = 0.0
                    settle_dwell_max_angular = 0.0
                settle_dwell_max_linear = max(settle_dwell_max_linear, linear_speed)
                settle_dwell_max_angular = max(settle_dwell_max_angular, angular_speed)
                dwell_elapsed = float(data.time - settle_dwell_start)
            else:
                settle_dwell_start = None
                settle_dwell_max_linear = 0.0
                settle_dwell_max_angular = 0.0
                dwell_elapsed = 0.0
            settle_rows.append({
                "time_s": f"{data.time:.8f}", "table_supported": int(table_supported),
                "linear_speed_m_s": f"{linear_speed:.9f}",
                "angular_speed_rad_s": f"{angular_speed:.9f}",
                "stable_dwell_elapsed_s": f"{dwell_elapsed:.8f}",
                "bottle_x_m": f"{data.xpos[bottle_id][0]:.9f}",
                "bottle_y_m": f"{data.xpos[bottle_id][1]:.9f}",
                "bottle_z_m": f"{data.xpos[bottle_id][2]:.9f}",
                "robot_bottle_contact_count": len(robot_contacts),
            })
            result["bottle_settle"] = {
                "status": "RUNNING", "elapsed_s": float(data.time - settle_start),
                "stable_dwell_elapsed_s": dwell_elapsed,
                "table_supported_steps": settle_supported_steps,
                "maximum_linear_speed_m_s": settle_max_linear,
                "maximum_angular_speed_rad_s": settle_max_angular,
                "current_linear_speed_m_s": linear_speed,
                "current_angular_speed_rad_s": angular_speed,
            }
            if dwell_elapsed >= SETTLE_DWELL_SECONDS:
                settled = True
                break
        if not settled:
            raise GateFailure(
                f"Bottle did not satisfy continuous table-support/low-velocity settle "
                f"within {SETTLE_MAX_SECONDS:.2f} s; last stable dwell="
                f"{dwell_elapsed:.6f} s, table_supported={table_supported}, "
                f"linear={linear_speed:.6f} m/s, angular={angular_speed:.6f} rad/s"
            )
        settled_bottle_pos = data.xpos[bottle_id].copy()
        initial_bottle_qpos = data.qpos[bottle_qadr].copy()
        initial_bottle_pos = settled_bottle_pos.copy()
        settle_result = {
            "status": "PASS", "elapsed_s": float(data.time - settle_start),
            "stable_dwell_s": float(dwell_elapsed),
            "table_supported_steps": settle_supported_steps,
            "maximum_linear_speed_m_s": settle_max_linear,
            "maximum_angular_speed_rad_s": settle_max_angular,
            "maximum_linear_speed_during_dwell_m_s": settle_dwell_max_linear,
            "maximum_angular_speed_during_dwell_rad_s": settle_dwell_max_angular,
            "settled_pose_qpos": initial_bottle_qpos.tolist(),
            "settled_position_world_m": settled_bottle_pos.tolist(),
            "robot_bottle_contacts": 0,
        }
        result["bottle_settle"] = settle_result
        events.append({"event": "BOTTLE_SETTLED", "time_s": float(data.time), **settle_result})

        events.append({"event": "PREGRASP_START", "time_s": float(data.time),
                       "duration_s": PREGRASP_SECONDS,
                       "site_position_m": data.site_xpos[site_id].tolist(),
                       "bottle_settled_qpos": initial_bottle_qpos.tolist(),
                       "target_arm_q_rad": pregrasp_q.tolist()})
        pregrasp_steps = int(round(PREGRASP_SECONDS / DT))
        for step in range(1, pregrasp_steps + 1):
            alpha = step / pregrasp_steps
            arm_target = settle_clear_q + alpha * (pregrasp_q - settle_clear_q)
            set_body_ctrl(arm_motor_targets(arm_target))
            mujoco.mj_step(model, data)
            steps += 1
            sample("PREGRASP", initial_commands["thumb_proximal_yaw"], arm_target)
        events.append({"event": "PREGRASP_COMPLETE", "time_s": float(data.time),
                       "site_position_m": data.site_xpos[site_id].tolist()})

        open_delta = max(abs(initial_targets[ch] - open_targets[ch]) for ch in hand_ids)
        open_duration = math.ceil(open_delta / OPEN_TARGET_SLEW_RAD_S / DT) * DT
        open_steps = int(round(open_duration / DT))
        events.append({"event": "OPEN_START", "time_s": float(data.time),
                       "duration_s": open_duration, "initial_normalized_commands": initial_commands})
        for step in range(1, open_steps + 1):
            alpha = step / open_steps
            for channel, aid in hand_ids.items():
                u = initial_commands[channel] + alpha * (1.0 - initial_commands[channel])
                data.ctrl[aid] = hand.normalized_target("R", channel, u, ranges)
            set_body_ctrl(arm_motor_targets(pregrasp_q))
            mujoco.mj_step(model, data)
            steps += 1
            sample("OPEN", 1.0, pregrasp_q)
        events.append({"event": "OPEN_COMPLETE", "time_s": float(data.time),
                       "normalized_command": 1.0})
        cv2.imwrite(str(output / "pregrasp.png"),
                    render_frame(renderer, model, data, camera, "PREGRASP / OPEN"))

        events.append({"event": "APPROACH_START", "time_s": float(data.time),
                       "duration_s": APPROACH_SECONDS,
                       "target_arm_q_rad": approach_q.tolist()})
        approach_steps = int(round(APPROACH_SECONDS / DT))
        for step in range(1, approach_steps + 1):
            alpha = step / approach_steps
            arm_target = pregrasp_q + alpha * (approach_q - pregrasp_q)
            set_body_ctrl(arm_motor_targets(arm_target))
            mujoco.mj_step(model, data)
            steps += 1
            sample("APPROACH", 1.0, arm_target)
        events.append({"event": "APPROACH_COMPLETE", "time_s": float(data.time),
                       "site_position_m": data.site_xpos[site_id].tolist()})
        cv2.imwrite(str(output / "approach.png"),
                    render_frame(renderer, model, data, camera, "APPROACH / OPEN"))

        max_delta = max(abs(open_targets[ch] - close_targets[ch]) for ch in hand_ids)
        close_duration = math.ceil(max_delta / CLOSE_TARGET_SLEW_RAD_S / DT) * DT
        close_steps = int(round(close_duration / DT))
        channel_close_targets = open_targets.copy()
        channel_close_rates = {
            channel: (close_targets[channel] - open_targets[channel]) / close_duration
            for channel in hand_ids
        }
        for channel, aid in hand_ids.items():
            data.ctrl[aid] = channel_close_targets[channel]
        result["contact_aware_close"] = {
            "enabled": True,
            "trigger_condition": {
                "right_hand_bottle_contact_distance_m_lte": 0.0,
                "single_contact_normal_force_n_gt": CONTACT_TRIGGER_FORCE_N,
                "consecutive_physics_samples": CONTACT_CONFIRM_SAMPLES,
                "persistence_s": CONTACT_CONFIRM_SAMPLES * DT,
                "force_threshold_basis": "existing active_contact_sets threshold",
            },
            "contact_loss_confirmation_samples": CONTACT_LOSS_CONFIRM_SAMPLES,
            "preload_rule": {
                "rule": "project candidate target by joint closure sign so 0 <= direction * (target - q_actual) <= cap each physics step",
                "cap_rad": CONTACT_PRELOAD_MAX_RAD,
                "measurement_gate_tolerance_rad": CONTACT_PRELOAD_GATE_TOLERANCE_RAD,
                "basis": "half of the smallest measured first-contact target error (0.008283 rad)",
                "target_projection_uses_gate_tolerance": False,
                "candidate_advance": "existing per-channel slow close trajectory, capped at the full-close endpoint",
                "actuator_effort_bound_nm": HAND_MAX_TORQUE,
                "contact_force_bound_n": MAX_CONTACT_FORCE_N,
            },
            "mode": "per-digit FREE_CLOSE -> CONTACT_PRELOAD; each post-step command is reprojected against current qpos; after 20 absent samples CONTACT_LOST -> REACQUIRE at the original per-channel close rate",
            "channel_digits": channel_digits,
            "closure_directions": closure_directions,
            "channel_transitions": contact_hold_transition_times,
        }
        events.append({"event": "CLOSE_START", "time_s": float(data.time),
                       "duration_s": close_duration,
                       "target_slew_ceiling_rad_s": CLOSE_TARGET_SLEW_RAD_S,
                       "contact_aware": True,
                       "contact_trigger_force_n": CONTACT_TRIGGER_FORCE_N,
                       "contact_confirm_samples": CONTACT_CONFIRM_SAMPLES})

        def project_contact_target(channel: str, q_actual: float,
                                   candidate_target: float) -> tuple[float, float]:
            direction = closure_directions[channel]
            requested_error = direction * (candidate_target - q_actual)
            bounded_error = min(max(requested_error, 0.0), current_preload_cap_rad)
            target = q_actual + direction * bounded_error
            realized_error = direction * (target - q_actual)
            if realized_error < 0.0:
                target = q_actual
                realized_error = 0.0
            while realized_error > current_preload_cap_rad:
                target = math.nextafter(target, q_actual)
                realized_error = direction * (target - q_actual)
            if not 0.0 <= realized_error <= current_preload_cap_rad:
                raise GateFailure(
                    f"Dynamic preload projection failed for {channel}: "
                    f"directional error {realized_error:.17g} rad"
                )
            return target, realized_error

        def update_contact_close_targets(u: float) -> None:
            nonlocal maximum_contact_commanded_error_rad
            prior_contacts = bottle_contacts(model, data)
            active_digits = {
                digit: any(
                    c["side"] == "right_hand" and c["digit"] == digit
                    and c["distance_m"] <= 0.0
                    and c["normal_force_n"] > CONTACT_TRIGGER_FORCE_N
                    for c in prior_contacts
                )
                for digit in digit_contact_run
            }
            digit_forces = right_digit_contact_forces(prior_contacts)
            for digit, active in active_digits.items():
                mode = contact_controller_state[digit]
                if mode in {"FREE_CLOSE", "REACQUIRE"}:
                    digit_contact_run[digit] = digit_contact_run[digit] + 1 if active else 0
                    if digit_contact_run[digit] < CONTACT_CONFIRM_SAMPLES:
                        continue
                    transition_time = float(data.time)
                    surfaces = [
                        {"body": c["other_body"], "geom": c["other_geom"],
                         "bottle_geom": c["bottle_geom"], "distance_m": c["distance_m"],
                         "normal_force_n": c["normal_force_n"]}
                        for c in prior_contacts
                        if c["side"] == "right_hand" and c["digit"] == digit
                        and c["distance_m"] <= 0.0
                        and c["normal_force_n"] > CONTACT_TRIGGER_FORCE_N
                    ]
                    channel_entries = {}
                    for channel, channel_digit in channel_digits.items():
                        if channel_digit != digit:
                            continue
                        joint = channel_joint_names[channel]
                        q_contact = joint_qpos(model, data, joint)
                        advancing_target = channel_close_targets[channel]
                        direction = closure_directions[channel]
                        existing_error = direction * (advancing_target - q_contact)
                        hold_target, preload = project_contact_target(
                            channel, q_contact, advancing_target
                        )
                        channel_close_targets[channel] = hold_target
                        contact_hold_targets[channel] = hold_target
                        contact_preload_offsets[channel] = preload
                        contact_commanded_errors[channel] = preload
                        maximum_contact_commanded_error_rad = max(
                            maximum_contact_commanded_error_rad, preload
                        )
                        contact_hold_transition_times[channel] = transition_time
                        channel_entries[channel] = {
                            "joint": joint,
                            "q_contact_rad": q_contact,
                            "advancing_target_rad": advancing_target,
                            "closure_direction": direction,
                            "pre_contact_target_error_rad": existing_error,
                            "preload_offset_rad": preload,
                            "contact_hold_target_rad": hold_target,
                        }
                    contact_controller_state[digit] = "CONTACT_PRELOAD"
                    digit_contact_lost_run[digit] = 0
                    event = {
                        "event": "CONTACT_PRELOAD_ENTER", "time_s": transition_time,
                        "digit": digit, "from_state": mode, "to_state": "CONTACT_PRELOAD",
                        "trigger": {
                            "distance_m_lte": 0.0,
                            "single_contact_normal_force_n_gt": CONTACT_TRIGGER_FORCE_N,
                            "consecutive_physics_samples": CONTACT_CONFIRM_SAMPLES,
                            "persistence_s": CONTACT_CONFIRM_SAMPLES * DT,
                            "digit_total_normal_force_n": digit_forces[digit],
                            "surfaces": surfaces,
                        },
                        "channels": channel_entries,
                        "normalized_close_command": u,
                    }
                    events.append(event)
                    result["contact_aware_close"].setdefault("transition_events", []).append(event)
                elif mode == "CONTACT_PRELOAD":
                    digit_contact_lost_run[digit] = 0 if active else digit_contact_lost_run[digit] + 1
                    if digit_contact_lost_run[digit] >= CONTACT_LOSS_CONFIRM_SAMPLES:
                        transition_time = float(data.time)
                        lost_event = {
                            "event": "CONTACT_LOST", "time_s": transition_time,
                            "digit": digit, "from_state": "CONTACT_PRELOAD",
                            "to_state": "REACQUIRE",
                            "absent_samples": digit_contact_lost_run[digit],
                        }
                        events.append(lost_event)
                        result["contact_aware_close"].setdefault("transition_events", []).append(lost_event)
                        contact_controller_state[digit] = "REACQUIRE"
                        digit_contact_run[digit] = 0
                        digit_contact_lost_run[digit] = 0
                        reacquire_event = {
                            "event": "REACQUIRE", "time_s": transition_time,
                            "digit": digit, "from_state": "CONTACT_LOST",
                            "to_state": "REACQUIRE",
                        }
                        events.append(reacquire_event)
                        result["contact_aware_close"].setdefault("transition_events", []).append(reacquire_event)

            for channel, aid in hand_ids.items():
                digit = channel_digits[channel]
                if contact_controller_state[digit] != "CONTACT_PRELOAD":
                    next_target = channel_close_targets[channel] + channel_close_rates[channel] * DT
                    if channel_close_rates[channel] >= 0.0:
                        channel_close_targets[channel] = min(close_targets[channel], next_target)
                    else:
                        channel_close_targets[channel] = max(close_targets[channel], next_target)
                else:
                    q_actual = joint_qpos(model, data, channel_joint_names[channel])
                    direction = closure_directions[channel]
                    candidate = channel_close_targets[channel] + channel_close_rates[channel] * DT
                    if direction * (candidate - close_targets[channel]) > 0.0:
                        candidate = close_targets[channel]
                    target, commanded_error = project_contact_target(channel, q_actual, candidate)
                    channel_close_targets[channel] = target
                    contact_hold_targets[channel] = target
                    contact_commanded_errors[channel] = commanded_error
                    maximum_contact_commanded_error_rad = max(
                        maximum_contact_commanded_error_rad, commanded_error
                    )
                data.ctrl[aid] = channel_close_targets[channel]

        for step in range(1, close_steps + 1):
            u = 1.0 - step / close_steps
            last_close_command_u = u
            set_body_ctrl(arm_motor_targets(approach_q))
            last_applied_targets.update({channel: float(data.ctrl[aid])
                                         for channel, aid in hand_ids.items()})
            mujoco.mj_step(model, data)
            steps += 1
            update_contact_close_targets(u)
            sample("CLOSE", u, approach_q)
        events.append({"event": "CLOSE_COMPLETE", "time_s": float(data.time),
                       "normalized_command": 0.0,
                       "contact_preload_channels": sorted(contact_hold_targets),
                       "contact_hold_targets_rad": contact_hold_targets.copy()})
        cv2.imwrite(str(output / "grasp_hold.png"),
                    render_frame(renderer, model, data, camera, "CLOSE / HOLD"))
        save_contact_views(renderer, model, data, camera, site_id, bottle_id, output, "close")

        hold_steps = int(round(HOLD_SECONDS / DT))
        hold_start = float(data.time)
        transition_hold_start = hold_start
        events.append({"event": "CONTACT_PRELOAD_HOLD_START", "time_s": hold_start,
                       "channel_targets_rad": channel_close_targets.copy(),
                       "contact_preload_channels": sorted(contact_hold_targets)})
        for _ in range(hold_steps):
            set_body_ctrl(arm_motor_targets(approach_q))
            last_applied_targets.update({channel: float(data.ctrl[aid])
                                         for channel, aid in hand_ids.items()})
            mujoco.mj_step(model, data)
            steps += 1
            update_contact_close_targets(0.0)
            sample("HOLD", last_close_command_u, approach_q)
        save_contact_views(renderer, model, data, camera, site_id, bottle_id, output, "hold")
        contact_fraction = hold_contact_steps / hold_steps
        hold_table_fraction = hold_table_support_steps / hold_steps
        result.update({
            "hold_valid_contact_steps": hold_contact_steps,
            "hold_steps": hold_steps,
            "hold_valid_contact_fraction": contact_fraction,
            "hold_table_support_steps": hold_table_support_steps,
            "hold_table_support_fraction": hold_table_fraction,
            "diagnostic_table_support_misses": table_support_misses,
            "diagnostic_sample_steps": diagnostic_sample_steps,
            "max_hold_bottle_lift_m": max_hold_bottle_lift,
        })
        if contact_fraction < CONTACT_HOLD_FRACTION:
            raise GateFailure(
                f"Opposing-digit topology failed: simultaneous thumb/opposing contact "
                f"for {hold_contact_steps}/{hold_steps} HOLD steps ({contact_fraction:.1%})"
            )
        if hold_table_support_steps != hold_steps or table_support_misses:
            raise GateFailure(
                f"Bottle lost table support during diagnostic: {hold_table_support_steps}/{hold_steps} "
                f"HOLD steps; {table_support_misses}/{diagnostic_sample_steps} diagnostic samples missed"
            )
        if max_hold_bottle_lift > 0.002:
            raise GateFailure(
                f"Bottle left table support during contact diagnostic: {max_hold_bottle_lift:.6f} m lift"
            )
        events.append({"event": "HOLD_VALIDATED", "time_s": float(data.time),
                       "duration_s": float(data.time-hold_start),
                       "simultaneous_contact_steps": hold_contact_steps,
                       "contact_fraction": contact_fraction,
                       "table_support_steps": hold_table_support_steps})

        if diagnostic_only:
            result.update({
                "status": "CONTACT_DIAGNOSTIC_PASS", "passed": True,
                "completed_stage": "CONTACT_HOLD_1S_STOP", "physics_steps": steps,
                "physics_time_s": float(data.time),
                "initial_bottle_pose_qpos": initial_bottle_qpos.tolist(),
                "final_bottle_position_world_m": data.xpos[bottle_id].tolist(),
                "right_thumb_contact_seen": thumb_seen,
                "right_opposing_digit_contact_seen": opposing_seen,
                "left_hand_bottle_contact": left_contact_seen,
                "maximum_hand_joint_qvel_rad_s": max_qvel,
                "maximum_hand_joint_qvel_joint": max_qvel_joint,
                "maximum_mimic_error_rad": max_mimic,
                "maximum_mimic_diagnostic_error_rad": max_mimic_diagnostic,
                "mimic_diagnostic_threshold_exceed_steps": mimic_diagnostic_exceed_steps,
                "first_mimic_diagnostic_threshold_exceed": first_mimic_diagnostic_exceed,
                "maximum_contact_preload_target_error_rad": maximum_contact_target_error_rad,
                "minimum_contact_preload_target_error_rad": (
                    None if not math.isfinite(minimum_contact_target_error_rad)
                    else minimum_contact_target_error_rad
                ),
                "maximum_contact_commanded_error_rad": maximum_contact_commanded_error_rad,
                "maximum_hand_actuator_effort_nm": maximum_hand_actuator_effort_nm,
                "maximum_joint_limit_violation_rad": max_limit,
                "maximum_object_contact_force_n": max_force,
                "maximum_object_penetration_m": max_penetration,
                "no_bottle_qpos_writes": True,
                "active_rollout_qpos_write_scan": qpos_check,
                "events": events,
            })
            return {"result": result, "physics_steps": steps}

        arming_start = float(data.time)
        required_arming_samples = int(math.ceil(LIFT_ARMING_DWELL_SECONDS / DT))
        arming_timeout_steps = int(math.ceil(LIFT_ARMING_TIMEOUT_SECONDS / DT))
        arming_consecutive_samples = 0
        previous_arming_conditions: tuple[str, ...] = ()
        result["lift_arming"].update({
            "status": "RUNNING", "state": "LIFT_ARMING",
            "start_time_s": arming_start,
            "required_consecutive_samples": required_arming_samples,
            "timeout_steps": arming_timeout_steps,
        })
        events.append({
            "event": "LIFT_ARMING_START", "time_s": arming_start,
            "state": "LIFT_ARMING", "held_arm_target_rad": approach_q.tolist(),
            "contact_force_threshold_n": CONTACT_TRIGGER_FORCE_N,
            "dwell_s": LIFT_ARMING_DWELL_SECONDS,
            "required_consecutive_samples": required_arming_samples,
            "timeout_s": LIFT_ARMING_TIMEOUT_SECONDS,
            "controller": "existing dynamic contact-preload/reacquire controller",
        })
        lift_ready = False
        ready_snapshot = None
        for _ in range(arming_timeout_steps):
            set_body_ctrl(arm_motor_targets(approach_q))
            last_applied_targets.update({channel: float(data.ctrl[aid])
                                         for channel, aid in hand_ids.items()})
            mujoco.mj_step(model, data)
            steps += 1
            update_contact_close_targets(0.0)
            sample("LIFT_ARMING", 0.0, approach_q)

            arming_contacts = bottle_contacts(model, data)
            thumb, opposing, left_contact, _ = active_contact_sets(arming_contacts)
            digit_forces = right_digit_contact_forces(arming_contacts)
            table_supported = any(
                c["side"] == "other" and c["other_body"] == "m0_table"
                and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > CONTACT_TRIGGER_FORCE_N
                for c in arming_contacts
            )
            object_velocity = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, object_velocity, 0
            )
            angular_speed = float(np.linalg.norm(object_velocity[:3]))
            linear_speed = float(np.linalg.norm(object_velocity[3:]))
            joint_state = current_joint_state(model, data, mimics)
            current_limit_violation, current_limit_detail = max_hand_limit_violation(
                model, data, limits
            )
            hand_qvel = {
                name: joint_qvel(model, data, name)
                for name in limits if name.startswith(("L_", "R_"))
            }
            current_max_qvel = max((abs(value) for value in hand_qvel.values()), default=0.0)
            failures = []
            if not thumb:
                failures.append("right_thumb_not_force_bearing")
            if not opposing:
                failures.append("opposing_digit_not_force_bearing")
            if left_contact:
                failures.append("left_hand_not_clear")
            if not table_supported:
                failures.append("bottle_not_table_supported")
            if linear_speed > SETTLE_MAX_LINEAR_SPEED_M_S:
                failures.append("bottle_linear_speed_above_settle_bound")
            if angular_speed > SETTLE_MAX_ANGULAR_SPEED_RAD_S:
                failures.append("bottle_angular_speed_above_settle_bound")
            if current_max_qvel > MAX_HAND_QVEL_RAD_S:
                failures.append("hand_source_velocity_gate")
            if joint_state["max_mimic"] > MIMIC_DIAGNOSTIC_TOLERANCE_RAD:
                failures.append("mimic_diagnostic_gate")
            if current_limit_violation > JOINT_LIMIT_TOLERANCE_RAD:
                failures.append("joint_limit_gate")
            conditions = tuple(failures)
            if conditions:
                if arming_consecutive_samples:
                    events.append({
                        "event": "LIFT_ARMING_DWELL_RESET", "time_s": float(data.time),
                        "previous_dwell_samples": arming_consecutive_samples,
                        "previous_dwell_s": arming_consecutive_samples * DT,
                        "failed_conditions": list(conditions),
                    })
                arming_consecutive_samples = 0
                previous_arming_conditions = conditions
            else:
                arming_consecutive_samples += 1
                previous_arming_conditions = ()

            dwell_elapsed = arming_consecutive_samples * DT
            contacts_detail = force_bearing_contact_details(model, data, arming_contacts)
            arming_row = {
                "time_s": float(data.time),
                "state": "READY_TO_LIFT" if arming_consecutive_samples >= required_arming_samples
                else "LIFT_ARMING",
                "arming_elapsed_s": float(data.time - arming_start),
                "continuous_force_bearing_samples": arming_consecutive_samples,
                "continuous_force_bearing_dwell_s": dwell_elapsed,
                "ready_conditions_pass": int(not conditions),
                "failed_conditions": json.dumps(list(conditions)),
                "contact_force_threshold_n": CONTACT_TRIGGER_FORCE_N,
                "thumb_force_bearing": int(thumb),
                "opposing_force_bearing": int(opposing),
                "left_hand_bottle_contact": int(left_contact),
                "bottle_table_supported": int(table_supported),
                "table_normal_force_n": sum(
                    c["true_normal_force_n"] for c in arming_contacts
                    if c["side"] == "other" and c["other_body"] == "m0_table"
                ),
                "total_hand_bottle_normal_force_n": sum(
                    c["true_normal_force_n"] for c in arming_contacts if c["side"] == "right_hand"
                ),
                "bottle_weight_n": float(model.body_mass[bottle_id] * abs(model.opt.gravity[2])),
                "table_weight_fraction": sum(
                    c["true_normal_force_n"] for c in arming_contacts
                    if c["side"] == "other" and c["other_body"] == "m0_table"
                ) / max(float(model.body_mass[bottle_id] * abs(model.opt.gravity[2])), 1e-12),
                "bottle_position_world_m": json.dumps(data.xpos[bottle_id].tolist()),
                "bottle_quaternion_wxyz": json.dumps(data.xquat[bottle_id].tolist()),
                "bottle_linear_velocity_world_m_s": json.dumps(object_velocity[3:].tolist()),
                "bottle_angular_velocity_world_rad_s": json.dumps(object_velocity[:3].tolist()),
                "bottle_linear_speed_m_s": linear_speed,
                "bottle_angular_speed_rad_s": angular_speed,
                "bottle_settle_linear_speed_bound_m_s": SETTLE_MAX_LINEAR_SPEED_M_S,
                "bottle_settle_angular_speed_bound_rad_s": SETTLE_MAX_ANGULAR_SPEED_RAD_S,
                "right_digit_contact_forces_n": json.dumps(digit_forces, sort_keys=True),
                "right_digit_contact_geometry_json": json.dumps(
                    right_digit_contact_geometry(arming_contacts), sort_keys=True
                ),
                "force_bearing_contact_geometry_json": json.dumps(contacts_detail, sort_keys=True),
                "controller_states_json": json.dumps(contact_controller_state, sort_keys=True),
                "maximum_hand_qvel_rad_s": current_max_qvel,
                "maximum_hand_qvel_joint": max(hand_qvel, key=lambda name: abs(hand_qvel[name]))
                if hand_qvel else "",
                "maximum_mimic_error_rad": joint_state["max_mimic"],
                "maximum_joint_limit_violation_rad": current_limit_violation,
                "joint_limit_violation_detail": json.dumps(current_limit_detail, sort_keys=True),
            }
            for channel, aid in hand_ids.items():
                digit = channel_digits[channel]
                joint = channel_joint_names[channel]
                target = float(data.ctrl[aid])
                actual = joint_qpos(model, data, joint)
                error = closure_directions[channel] * (target - actual)
                arming_row.update({
                    f"{channel}_target_rad": target,
                    f"{channel}_actual_rad": actual,
                    f"{channel}_closure_direction_error_rad": error,
                    f"{channel}_qvel_rad_s": joint_qvel(model, data, joint),
                    f"{channel}_actuator_force_nm": float(data.actuator_force[aid]),
                    f"{channel}_contact_force_n": digit_forces[digit],
                })
            lift_arming_trace_rows.append(arming_row)

            if arming_consecutive_samples >= required_arming_samples:
                lift_ready = True
                ready_snapshot = {
                    "time_s": float(data.time), "dwell_s": dwell_elapsed,
                    "consecutive_samples": arming_consecutive_samples,
                    "thumb_contact_force_n": digit_forces["thumb"],
                    "opposing_digit_contact_forces_n": {
                        digit: digit_forces[digit]
                        for digit in ("index", "middle", "ring", "pinky")
                    },
                    "bottle_table_supported": table_supported,
                    "table_normal_force_n": arming_row["table_normal_force_n"],
                    "bottle_position_world_m": data.xpos[bottle_id].tolist(),
                    "bottle_linear_velocity_world_m_s": object_velocity[3:].tolist(),
                    "bottle_angular_velocity_world_rad_s": object_velocity[:3].tolist(),
                    "force_bearing_contacts": contacts_detail,
                    "controller_states": contact_controller_state.copy(),
                    "arm_target_rad": approach_q.tolist(),
                }
                result["lift_arming"].update({
                    "status": "PASS", "state": "READY_TO_LIFT",
                    "ready_time_s": float(data.time),
                    "achieved_dwell_s": dwell_elapsed,
                    "achieved_consecutive_samples": arming_consecutive_samples,
                    "ready_snapshot": ready_snapshot,
                })
                events.append({
                    "event": "READY_TO_LIFT", "time_s": float(data.time),
                    "dwell_s": dwell_elapsed,
                    "consecutive_samples": arming_consecutive_samples,
                    "force_bearing_threshold_n": CONTACT_TRIGGER_FORCE_N,
                    "snapshot": ready_snapshot,
                })
                break

            if data.time - arming_start >= LIFT_ARMING_TIMEOUT_SECONDS - 1e-12:
                result["lift_arming"].update({
                    "status": "FAIL", "state": "LIFT_ARMING_TIMEOUT",
                    "timeout_time_s": float(data.time),
                    "last_failed_conditions": list(previous_arming_conditions),
                    "last_continuous_dwell_s": dwell_elapsed,
                })
                events.append({
                    "event": "LIFT_ARMING_TIMEOUT", "time_s": float(data.time),
                    "last_failed_conditions": list(previous_arming_conditions),
                    "last_continuous_dwell_s": dwell_elapsed,
                })
                break

        if not lift_ready:
            raise GateFailure(
                "Stable force-bearing lift-entry state cannot be armed within "
                f"{LIFT_ARMING_TIMEOUT_SECONDS:.2f} s; last failed conditions="
                f"{previous_arming_conditions}, last continuous dwell={dwell_elapsed:.6f} s"
            )

        stage2_wrench = load_capacity_targets["effort_balanced_10pct_margin_allocation"]
        if not stage2_wrench.get("feasible_at_mu_1_4"):
            raise GateFailure("Canonical effort-balanced support allocation is infeasible; LOAD_BUILD is prohibited")
        bottle_weight_n = float(model.body_mass[bottle_id] * abs(model.opt.gravity[2]))
        lp_weight_n = float(stage2_wrench["achieved_wrench_about_com"]["force_n"][2])
        if abs(bottle_weight_n - lp_weight_n) > 0.005:
            raise GateFailure(
                f"Stage 1 wrench target weight {lp_weight_n:.9g} N does not match runtime "
                f"bottle weight {bottle_weight_n:.9g} N"
            )
        lp_solution = {}
        for item in stage2_wrench["per_contact"]:
            group = lp_solution.setdefault(item["digit"], {"normal_force_n": 0.0})
            group["normal_force_n"] += float(item["normal_force_n"])
        required_total_normal_n = float(stage2_wrench["total_normal_force_n"])
        required_thumb_normal_n = float(lp_solution["thumb"]["normal_force_n"])
        required_opposing_normal_n = sum(
            float(lp_solution[digit]["normal_force_n"])
            for digit in ("index", "middle", "ring", "pinky")
        )
        load_build_targets = {
            "source_static_lp": "candidate_c2_robust_wrench.json effort_balanced_10pct_margin_allocation",
            "friction_model": load_capacity_targets["friction_model"],
            "measured_force_gate": "physical contact wrench and table normal must balance bottle weight; target torque alone is insufficient",
            "safety_factor": LOAD_BUILD_SAFETY_FACTOR,
            "minimum_total_true_normal_force_n": required_total_normal_n,
            "target_total_true_normal_force_n": required_total_normal_n,
            "minimum_thumb_true_normal_force_n": required_thumb_normal_n,
            "target_thumb_true_normal_force_n": required_thumb_normal_n,
            "minimum_opposing_true_normal_force_n": required_opposing_normal_n,
            "bottle_weight_n": bottle_weight_n,
            "control_mode": "simulation-only generalized-force impedance with saved-state contact-topology reallocation during load transfer",
            "source_lp_feedforward_driver_torques_nm": static_effort_by_channel,
            "feedforward_driver_torques_nm": load_effort_by_channel.copy(),
            "contact_topology_reallocation": {
                **LOAD_CONTACT_REALLOCATION,
                "driver_efforts_nm": load_effort_by_channel.copy(),
                "source_static_c2_driver_efforts_nm": static_effort_by_channel.copy(),
            },
            "index_feedforward_correction": {
                "status": "NOT APPLIED",
                "historical_value_nm": 0.085,
                "source_lp_value_nm": static_effort_by_channel["index_proximal"],
                "reason": "the prior correction exceeded the unchanged 6 N total normal-force bound",
            },
            "bounded_correction": {
                "status": "REJECTED_NOT_APPLIED",
                "classification": "historical contact/solver active-set hypothesis",
                "changed_channel": "index_proximal",
                "source_lp_feedforward_nm": static_effort_by_channel["index_proximal"],
                "historical_test_feedforward_nm": 0.085,
                "scope": "historical LOAD_BUILD-only proposal; not used in this run",
                "reason": "the prior trial exceeded the unchanged 6 N total normal-force bound",
            },
            "trace_derived_command_realization_correction": {
                "classification": "single LOAD_BUILD control-allocation realization correction",
                "reference_run_id": "dfq-long-c2-dynamic-1791446357-38919",
                "changed_channel": "thumb_proximal_yaw",
                "static_lp_feedforward_nm": static_effort_by_channel["thumb_proximal_yaw"],
                "observed_prior_position_impedance_nm": 0.16837492809141844,
                "observed_prior_applied_generalized_effort_nm": -0.0121994727704315,
                "method": "cancel only the Kp position-error term opposing static LP feedforward during LOAD_BUILD; retain the existing Kv damping, feedforward ramp, MuJoCo bias compensation, and 0.531 Nm cap",
                "basis": "the prior static allocation required -0.1741287755 Nm thumb-yaw torque, but measured +0.1683749281 Nm position impedance reduced the actual applied generalized effort to -0.0121994728 Nm",
            },
            "feedforward_ramp_seconds": SIM_ONLY_FEEDFORWARD_RAMP_S,
            "impedance_kp_nm_per_rad": HAND_KP,
            "impedance_kv_nms_per_rad": HAND_KV,
            "bias_compensation": "MuJoCo qfrc_bias on each active driver DoF",
            "max_driver_effort_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
            "effort_cap_basis": (
                "0.531 Nm is 10.20% above the 0.4818446744 Nm worst effort-balanced "
                "bounded-uncertainty allocation and 53.1% of each 1.0 Nm URDF effort limit"
            ),
            "position_hand_actuators_disabled_at_load_build": True,
            "direct_arm_lift_after_contact": direct_arm_lift,
            "follower_actuators": 0,
            "follower_qpos_writes": 0,
            "max_single_contact_resultant_n": LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N,
            "max_total_hand_contact_resultant_n": LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N,
            "max_total_hand_true_normal_force_n": LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N,
            "actual_wrist_drift": (
                "not applicable; fixed-wrist LOAD_BUILD is skipped"
                if direct_arm_lift else
                "diagnostic-only; arm joint target remains fixed at approach_q throughout LOAD_BUILD"
            ),
            "max_bottle_rise_m": LOAD_BUILD_MAX_BOTTLE_RISE_M,
            "source_hand_velocity_limit_rad_s": MAX_HAND_QVEL_RAD_S,
            "mimic_diagnostic_tolerance_rad": MIMIC_DIAGNOSTIC_TOLERANCE_RAD,
            "joint_limit_tolerance_rad": JOINT_LIMIT_TOLERANCE_RAD,
            "wrist_target": "approach_q held constant throughout LOAD_BUILD",
            "bottle_state_writes": 0,
        }
        result["load_build"] = {
            "status": "RUNNING", "state": "LOAD_BUILD",
            "derived_targets_and_bounds": load_build_targets,
            "load_ready": False, "stable_samples_required": int(math.ceil(
                SIM_ONLY_LOAD_DWELL_S / DT
            )),
        }
        if not direct_arm_lift:
            events.append({
                "event": "LOAD_BUILD_START", "time_s": float(data.time),
                "control_mode": "simulation-only torque impedance plus static feedforward",
                "feedforward_ramp_seconds": SIM_ONLY_FEEDFORWARD_RAMP_S,
                "effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                "arm_target_rad": approach_q.tolist(),
                "derived_targets": load_build_targets,
            })
        load_build_start_time = float(data.time)
        load_build_wrist_start = np.asarray(data.xpos[wrist_id], dtype=float).copy()
        load_build_bottle_start = np.asarray(data.xpos[bottle_id], dtype=float).copy()
        if direct_arm_lift and stop_after_load_build:
            raise GateFailure("stop_after_load_build is incompatible with direct_arm_lift")
        if direct_arm_lift:
            data.qfrc_applied[:] = 0.0
        begin_simulation_only_effort_mode()
        load_build_consecutive = 0
        load_build_ready_snapshot = None
        load_build_max_steps = 0 if direct_arm_lift else int(math.ceil(
            (LOAD_BUILD_MAX_SECONDS + SIM_ONLY_LOAD_DWELL_S) / DT
        ))
        required_load_normal_n = required_total_normal_n

        for load_step in range(1, load_build_max_steps + 1):
            elapsed = float(data.time - load_build_start_time)
            set_body_ctrl(arm_motor_targets(approach_q))
            effort_command = apply_simulation_only_effort(elapsed, load_build_schedule=True)
            result["last_load_build_command_pre_step"] = {
                "time_s": float(data.time),
                "velocity_guard": effort_command["velocity_guard"],
                "channels": effort_command["channels"],
            }
            mujoco.mj_step(model, data)
            steps += 1
            record_simulation_only_effort(effort_command, "LOAD_BUILD")
            sample("LOAD_BUILD", last_close_command_u, approach_q)

            contacts = bottle_contacts(model, data)
            digit_normals = right_digit_true_normal_forces(contacts)
            thumb_normal = digit_normals["thumb"]
            opposing_normal = sum(
                digit_normals[digit] for digit in ("index", "middle", "ring", "pinky")
            )
            total_hand_normal = sum(digit_normals.values())
            total_hand_contact_resultant = sum(
                c["normal_force_n"] for c in contacts if c["side"] == "right_hand"
            )
            max_single_contact_resultant = max((
                c["normal_force_n"] for c in contacts if c["side"] == "right_hand"
            ), default=0.0)
            table_normal = sum(
                c["true_normal_force_n"] for c in contacts
                if c["side"] == "other" and c["other_body"] == "m0_table"
            )
            table_supported = any(
                c["side"] == "other" and c["other_body"] == "m0_table"
                and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                for c in contacts
            )
            hand_force, hand_torque = bottle_hand_wrench(contacts, data.xipos[bottle_id])
            friction_utilization, friction_by_contact = measured_friction_utilization(
                model, data, contacts
            )
            object_velocity = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, object_velocity, 0
            )
            wrist_drift = float(np.linalg.norm(data.xpos[wrist_id] - load_build_wrist_start))
            bottle_rise = float(data.xpos[bottle_id][2] - load_build_bottle_start[2])
            max_applied_effort = max(
                abs(float(values["applied_nm"]))
                for values in effort_command["channels"].values()
            )
            max_qvel_this_step = max((
                abs(joint_qvel(model, data, name)) for name in limits
                if name.startswith(("L_", "R_"))
            ), default=0.0)
            current_joint_state_data = current_joint_state(model, data, mimics)
            limit_violation, limit_detail = max_hand_limit_violation(model, data, limits)
            load_row = {
                "time_s": float(data.time),
                "elapsed_s": float(data.time - load_build_start_time),
                "phase": "LOAD_BUILD",
                "controller_mode": "SIMULATION_ONLY_TORQUE_IMPEDANCE_FEEDFORWARD",
                "stable_dwell_samples": load_build_consecutive,
                "required_total_true_normal_force_n": required_load_normal_n,
                "required_thumb_true_normal_force_n": required_thumb_normal_n,
                "required_opposing_true_normal_force_n": required_opposing_normal_n,
                "total_hand_true_normal_force_n": total_hand_normal,
                "thumb_true_normal_force_n": thumb_normal,
                "opposing_true_normal_force_n": opposing_normal,
                "table_normal_force_n": table_normal,
                "table_supported": int(table_supported),
                "hand_force_on_bottle_world_n": hand_force.tolist(),
                "hand_vertical_support_force_n": float(hand_force[2]),
                "hand_torque_about_com_world_nm": hand_torque.tolist(),
                "hand_supported_fraction": float(hand_force[2]) / max(bottle_weight_n, 1e-12),
                "vertical_force_balance_error_n": abs(
                    float(hand_force[2]) + table_normal - bottle_weight_n
                ),
                "horizontal_hand_force_n": float(np.linalg.norm(hand_force[:2])),
                "hand_wrench_torque_norm_nm": float(np.linalg.norm(hand_torque)),
                "maximum_measured_friction_utilization": friction_utilization,
                "friction_utilization_by_contact": friction_by_contact,
                "total_hand_contact_resultant_n": total_hand_contact_resultant,
                "maximum_single_contact_resultant_n": max_single_contact_resultant,
                "maximum_native_hand_actuator_force_nm": 0.0,
                "maximum_applied_driver_effort_nm": max_applied_effort,
                "driver_effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                "maximum_hand_qvel_rad_s": max_qvel_this_step,
                "maximum_mimic_error_rad": current_joint_state_data["max_mimic"],
                "maximum_joint_limit_violation_rad": limit_violation,
                "joint_limit_violation_detail": json.dumps(limit_detail, sort_keys=True),
                "wrist_drift_m": wrist_drift,
                "right_wrist_actual_z_m": float(data.xpos[wrist_id][2]),
                "bottle_rise_from_load_build_start_m": bottle_rise,
                "bottle_position_world_m": data.xpos[bottle_id].tolist(),
                "bottle_linear_velocity_world_m_s": object_velocity[3:].tolist(),
                "bottle_angular_velocity_world_rad_s": object_velocity[:3].tolist(),
                "right_digit_true_normal_forces_n": digit_normals,
                "right_digit_contact_geometry_json": right_digit_contact_geometry(contacts),
                "force_bearing_contact_details_json": force_bearing_contact_details(
                    model, data, contacts
                ),
                "controller_states": {digit: "SIMULATION_ONLY_TORQUE_IMPEDANCE"
                                       for digit in contact_controller_state},
                "channel_targets_actual_errors_efforts": {},
            }
            for channel, aid in hand_ids.items():
                joint = channel_joint_names[channel]
                actual = joint_qpos(model, data, joint)
                command_values = effort_command["channels"][channel]
                target = float(command_values["q_reference_rad"])
                error = target - actual
                load_row["channel_targets_actual_errors_efforts"][channel] = {
                    "joint": joint, "target_rad": target, "actual_rad": actual,
                    "directional_error_rad": error,
                    "qvel_rad_s": joint_qvel(model, data, joint),
                    "actuator_effort_nm": float(data.actuator_force[aid]),
                    "feedforward_effort_nm": command_values["feedforward_nm"],
                    "impedance_effort_nm": command_values["impedance_nm"],
                    "bias_compensation_nm": command_values["bias_compensation_nm"],
                    "applied_generalized_effort_nm": float(data.qfrc_applied[hand_driver_dofs[channel]]),
                }
                load_row[f"{channel}_target_rad"] = target
                load_row[f"{channel}_actual_rad"] = actual
                load_row[f"{channel}_directional_error_rad"] = error
                load_row[f"{channel}_qvel_rad_s"] = joint_qvel(model, data, joint)
                load_row[f"{channel}_actuator_effort_nm"] = float(data.actuator_force[aid])
                load_row[f"{channel}_feedforward_nm"] = command_values["feedforward_nm"]
                load_row[f"{channel}_impedance_nm"] = command_values["impedance_nm"]
                load_row[f"{channel}_applied_generalized_effort_nm"] = float(
                    data.qfrc_applied[hand_driver_dofs[channel]]
                )
            load_build_trace_rows.append(load_row)

            hard_failure = None
            if max_single_contact_resultant > LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:
                hard_failure = f"single contact resultant {max_single_contact_resultant:.9g} N exceeded {LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:.3f} N"
            elif total_hand_contact_resultant > LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:
                hard_failure = f"total hand contact resultant {total_hand_contact_resultant:.9g} N exceeded {LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:.3f} N"
            elif total_hand_normal > LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:
                hard_failure = f"total hand normal force {total_hand_normal:.9g} N exceeded {LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:.3f} N"
            elif bottle_rise > LOAD_BUILD_MAX_BOTTLE_RISE_M:
                hard_failure = f"bottle rose {bottle_rise:.9g} m during LOAD_BUILD before Stage 3"
            elif not table_supported:
                hard_failure = "bottle lost table support during stationary LOAD_BUILD"
            if hard_failure:
                result["load_build"].update({
                    "status": "FAIL", "first_bound_failure": {
                        "time_s": float(data.time), "reason": hard_failure,
                        "sample": load_row,
                    },
                })
                raise GateFailure("LOAD_BUILD safe-bound failure: " + hard_failure)

            thresholds_met = (
                thumb_normal > CONTACT_TRIGGER_FORCE_N
                and opposing_normal > CONTACT_TRIGGER_FORCE_N
                and float(hand_force[2]) >= SIM_ONLY_SUPPORT_FRACTION_MIN * bottle_weight_n
                and table_normal >= SIM_ONLY_TABLE_REMAINDER_FRACTION_MIN * bottle_weight_n
                and abs(float(hand_force[2]) + table_normal - bottle_weight_n)
                    <= SIM_ONLY_FORCE_BALANCE_TOLERANCE_FRACTION * bottle_weight_n
                and float(np.linalg.norm(hand_force[:2]))
                    <= SIM_ONLY_HORIZONTAL_FORCE_TOLERANCE_FRACTION * bottle_weight_n
                and float(np.linalg.norm(hand_torque)) <= SIM_ONLY_WRENCH_TORQUE_TOLERANCE_NM
                and friction_utilization <= SIM_ONLY_MAX_FRICTION_UTILIZATION
                and table_supported
                and bottle_rise <= LOAD_BUILD_MAX_BOTTLE_RISE_M
                and not left_contact_seen
                and float(np.linalg.norm(object_velocity[3:])) <= SETTLE_MAX_LINEAR_SPEED_M_S
                and float(np.linalg.norm(object_velocity[:3])) <= SETTLE_MAX_ANGULAR_SPEED_RAD_S
            )
            if thresholds_met and load_build_consecutive == 0:
                events.append({
                    "event": "LOAD_BUILD_TARGET_REACHED", "time_s": float(data.time),
                    "total_hand_true_normal_force_n": total_hand_normal,
                    "thumb_true_normal_force_n": thumb_normal,
                    "opposing_true_normal_force_n": opposing_normal,
                    "table_normal_force_n": table_normal,
                    "hand_vertical_support_force_n": float(hand_force[2]),
                    "maximum_friction_utilization": friction_utilization,
                    "channel_generalized_effort_nm": {
                        channel: float(data.qfrc_applied[hand_driver_dofs[channel]])
                        for channel in hand_ids
                    },
                })

            if thresholds_met:
                load_build_consecutive += 1
                if load_build_consecutive * DT >= SIM_ONLY_LOAD_DWELL_S:
                    load_build_ready_snapshot = load_row.copy()
                    break
            else:
                load_build_consecutive = 0

            if elapsed >= LOAD_BUILD_MAX_SECONDS - DT:
                break

        if direct_arm_lift:
            result["live_contact_allocation_arming"] = {
                "status": "RUNNING",
                "control_mode": "SIMULATION_ONLY torque impedance plus live measured-contact wrench allocation",
                "duration_s": LIVE_ALLOCATION_HANDOFF_STEPS * DT,
                "fixed_wrist": True,
                "load_ready_claimed": False,
                "purpose": "single-step controller handoff; progressive wrist transfer starts immediately afterward",
            }
            preload_start = float(data.time)
            preload_bottle_z = float(data.xpos[bottle_id][2])
            preload_max_total_normal = 0.0
            preload_max_single_contact = 0.0
            preload_max_total_resultant = 0.0

            def fail_live_contact_arming(reason: str, sample_data: dict) -> None:
                result["live_contact_allocation_arming"].update({
                    "status": "FAIL",
                    "end_time_s": float(data.time),
                    "first_bound_failure": {"reason": reason, **sample_data},
                })
                events.append({
                    "event": "LIVE_CONTACT_ALLOCATION_ARMING_FAIL",
                    "time_s": float(data.time),
                    "reason": reason,
                    "sample": sample_data,
                })
                raise GateFailure(reason)

            for _ in range(LIVE_ALLOCATION_HANDOFF_STEPS):
                elapsed = float(data.time - preload_start)
                set_body_ctrl(arm_motor_targets(approach_q))
                before_step_contacts = bottle_contacts(model, data)
                before_step_table_normal = sum(
                    c["true_normal_force_n"] for c in before_step_contacts
                    if c["side"] == "other" and c["other_body"] == "m0_table"
                )
                effort_command = apply_simulation_only_effort(
                    elapsed,
                    load_build_schedule=True,
                    live_support_force_n=float(np.clip(
                        bottle_weight_n - before_step_table_normal, 0.0, bottle_weight_n
                    )),
                )
                mujoco.mj_step(model, data)
                steps += 1
                record_simulation_only_effort(effort_command, "LIVE_CONTACT_ALLOCATION_ARMING")
                sample("LIVE_CONTACT_ALLOCATION_ARMING", 0.0, approach_q)
                contacts = bottle_contacts(model, data)
                _, _, left_contact, _ = active_contact_sets(contacts)
                normal_forces = [
                    c["true_normal_force_n"] for c in contacts if c["side"] == "right_hand"
                ]
                total_normal = sum(normal_forces)
                total_resultant = sum(
                    c["normal_force_n"] for c in contacts if c["side"] == "right_hand"
                )
                max_single = max((
                    c["normal_force_n"] for c in contacts if c["side"] == "right_hand"
                ), default=0.0)
                table_supported = any(
                    c["side"] == "other" and c["other_body"] == "m0_table"
                    and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                    for c in contacts
                )
                table_normal = sum(
                    c["true_normal_force_n"] for c in contacts
                    if c["side"] == "other" and c["other_body"] == "m0_table"
                )
                bottle_rise = float(data.xpos[bottle_id][2] - preload_bottle_z)
                preload_max_total_normal = max(preload_max_total_normal, total_normal)
                preload_max_single_contact = max(preload_max_single_contact, max_single)
                preload_max_total_resultant = max(preload_max_total_resultant, total_resultant)
                preload_sample = {
                    "elapsed_s": float(data.time - preload_start),
                    "maximum_single_contact_resultant_n": max_single,
                    "total_contact_resultant_n": total_resultant,
                    "total_true_normal_force_n": total_normal,
                    "per_digit_true_normal_force_n": right_digit_true_normal_forces(contacts),
                    "table_normal_force_n": table_normal,
                    "table_supported": table_supported,
                    "bottle_rise_m": bottle_rise,
                    "bottle_position_world_m": np.asarray(data.xpos[bottle_id]).tolist(),
                    "left_contact": left_contact,
                    "source_velocity_guard": effort_command["velocity_guard"],
                    "driver_commands": {
                        channel: {
                            "joint": values["joint"],
                            "qpos_rad": joint_qpos(model, data, values["joint"]),
                            "qvel_rad_s": joint_qvel(model, data, values["joint"]),
                            "requested_nm": values["requested_nm"],
                            "applied_nm": values["applied_nm"],
                            "velocity_guard_brake_nm": values["velocity_guard_brake_nm"],
                            "predicted_qvel_chain_after_guard_rad_s": values.get(
                                "predicted_qvel_chain_after_guard_rad_s", {}
                            ),
                        }
                        for channel, values in effort_command["channels"].items()
                    },
                    "contact_details": force_bearing_contact_details(
                        model, data, contacts
                    ),
                }
                result["live_contact_allocation_arming"]["last_sample"] = preload_sample
                if left_contact:
                    fail_live_contact_arming(
                        "Left hand/arm contacted the bottle during contact-force preload",
                        preload_sample,
                    )
                if max_single > LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:
                    fail_live_contact_arming(
                        "Contact-force preload exceeded per-contact bound: "
                        f"{max_single:.9g} N > {LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N:.3f} N",
                        preload_sample,
                    )
                if total_resultant > LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:
                    fail_live_contact_arming(
                        "Contact-force preload exceeded total-resultant bound: "
                        f"{total_resultant:.9g} N > {LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N:.3f} N",
                        preload_sample,
                    )
                if total_normal > LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:
                    fail_live_contact_arming(
                        "Contact-force preload exceeded total-normal bound: "
                        f"{total_normal:.9g} N > {LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N:.3f} N",
                        preload_sample,
                    )
                if bottle_rise > LOAD_BUILD_MAX_BOTTLE_RISE_M or not table_supported:
                    fail_live_contact_arming(
                        "Bottle lost table support during contact-force preload: "
                        f"rise={bottle_rise:.9g} m, table_supported={table_supported}",
                        preload_sample,
                    )
            preload_contacts = bottle_contacts(model, data)
            preload_window = contact_window_summary(list(contact_window_history))
            preload_thumb, preload_opposing, preload_left, _ = active_contact_sets(preload_contacts)
            if preload_left or not preload_thumb or not preload_opposing or not preload_window["valid"]:
                raise GateFailure(
                    "Contact-force preload did not retain force-bearing thumb/opposing contact: "
                    + json.dumps(preload_window, sort_keys=True)
                )
            result["live_contact_allocation_arming"].update({
                "status": "PASS",
                "end_time_s": float(data.time),
                "maximum_total_true_normal_force_n": preload_max_total_normal,
                "maximum_single_contact_resultant_n": preload_max_single_contact,
                "maximum_total_contact_resultant_n": preload_max_total_resultant,
                "final_contact_window": preload_window,
                "final_table_supported": True,
                "bottle_qpos_writes": 0,
            })
            events.append({
                "event": "LIVE_CONTACT_ALLOCATION_ARMING_COMPLETE",
                "time_s": float(data.time),
                "maximum_total_true_normal_force_n": preload_max_total_normal,
                "final_contact_window": preload_window,
                "load_ready_claimed": False,
            })

        if direct_arm_lift:
            result["load_build"].update({
                "status": "RUNNING",
                "state": "PROGRESSIVE_LIVE_CONTACT_LOAD_TRANSFER",
                "load_ready": False,
                "reason": (
                    "the previous fixed-wrist saved-force path is replaced by measured-contact "
                    "allocation during incremental wrist motion; LOAD_READY requires measured airborne support"
                ),
                "physics_steps": steps,
            })
            events.append({
                "event": "PROGRESSIVE_LIVE_LOAD_BUILD_STARTED",
                "time_s": float(data.time),
                "reason": result["load_build"]["reason"],
            })
        elif load_build_ready_snapshot is None:
            result["load_build"].update({
                "status": "FAIL", "state": "SAFE_CAP_OR_DWELL_EXHAUSTED",
                "last_sample": load_build_trace_rows[-1] if load_build_trace_rows else None,
                "load_ready": False,
            })
            raise GateFailure(
                "LOAD_BUILD did not reach a stable measured physical wrench/support state within "
                f"{LOAD_BUILD_MAX_SECONDS:.2f} s under the fixed {SIM_ONLY_MAX_DRIVER_TORQUE_NM:.3f} Nm cap"
            )

        if not direct_arm_lift:
            result["load_build"].update({
                "status": "PASS", "state": "LOAD_READY", "load_ready": True,
                "ready_time_s": float(data.time),
                "achieved_stable_dwell_s": load_build_consecutive * DT,
                "ready_snapshot": load_build_ready_snapshot,
                "simulation_only_effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                "effort_mode_active": effort_mode_active,
            })
            events.append({
                "event": "LOAD_READY", "time_s": float(data.time),
                "stable_dwell_s": load_build_consecutive * DT,
                "snapshot": load_build_ready_snapshot,
            })
        else:
            result["load_build"]["effort_mode_active"] = False
        if stop_after_load_build:
            final_window = contact_window_summary(list(contact_window_history))
            result.update({
                "status": "PASS", "passed": True, "completed_stage": "LOAD_READY_STOP",
                "physics_steps": steps, "physics_time_s": float(data.time),
                "initial_bottle_pose_qpos": initial_bottle_qpos.tolist(),
                "final_bottle_position_world_m": data.xpos[bottle_id].tolist(),
                "lift_height_m": 0.0, "lift_target_height_m": 0.0,
                "final_right_thumb_contact": bool(final_window.get("thumb_qualifies", False)),
                "final_right_opposing_digit_contact": bool(final_window.get("valid_opposing_digits", [])),
                "final_contact_window": final_window,
                "left_hand_bottle_contact": left_contact_seen,
                "maximum_hand_joint_qvel_rad_s": max_qvel,
                "maximum_hand_joint_qvel_joint": max_qvel_joint,
                "first_velocity_breach": first_velocity_breach,
                "maximum_mimic_error_rad": max_mimic,
                "maximum_mimic_diagnostic_error_rad": max_mimic_diagnostic,
                "maximum_joint_limit_violation_rad": max_limit,
                "maximum_object_contact_force_n": max_force,
                "maximum_object_penetration_m": max_penetration,
                "no_bottle_qpos_writes": True,
                "active_rollout_qpos_write_scan": qpos_check,
                "events": [*events, {"event": "STOP_AFTER_LOAD_BUILD", "time_s": float(data.time)}],
                "final_contacts": bottle_contacts(model, data),
                "stop_after_load_build": True,
                "stage3_and_later_executed": False,
            })
            return {"result": result, "physics_steps": steps}
        result["stage3_and_later_executed"] = True
        load_build_stage3_wrist_start = np.asarray(data.xpos[wrist_id], dtype=float).copy()
        load_build_stage3_bottle_start = np.asarray(data.xpos[bottle_id], dtype=float).copy()
        for channel in hand_ids:
            channel_close_rates[channel] = 0.0

        transition_lift_start = float(data.time)
        reached = False
        lift_origin_z = float(
            settled_bottle_pos[2] if settled_bottle_pos is not None else initial_bottle_pos[2]
        )
        transfer_baseline_rows = [
            row for row in transition_trace_rows
            if row.get("phase") == "HOLD" and row.get("window") == "last_50ms_hold"
        ]
        if direct_arm_lift:
            if len(transfer_baseline_rows) < 20:
                raise GateFailure(
                    "Cannot establish a measured 50 ms HOLD load baseline before wrist transfer"
                )
            baseline_table_values = np.asarray([
                float(row["table_normal_force_n"]) for row in transfer_baseline_rows
            ], dtype=float)
            baseline_hand_values = np.asarray([
                float(row["hand_vertical_support_force_n"]) for row in transfer_baseline_rows
            ], dtype=float)
            transfer_baseline_table_n = float(np.mean(baseline_table_values))
            transfer_baseline_table_sigma_n = float(np.std(baseline_table_values, ddof=1))
            transfer_baseline_hand_n = float(np.mean(baseline_hand_values))
            initial_support_target_n = gates.progressive_support_demand_n(
                transfer_baseline_table_n, bottle_weight_n
            )
            live_allocation_state["support_target_n"] = initial_support_target_n
            result["progressive_load_transfer"] = {
                "status": "RUNNING",
                "passed": False,
                "measurement_window_s": 0.050,
                "baseline_source": "last 50 ms of validated HOLD before wrist motion",
                "baseline_sample_count": int(len(transfer_baseline_rows)),
                "baseline_table_normal_force_n": transfer_baseline_table_n,
                "baseline_table_normal_sigma_n": transfer_baseline_table_sigma_n,
                "baseline_hand_vertical_support_force_n": transfer_baseline_hand_n,
                "minimum_detectable_change_rule": "max(3 * baseline sigma, 1% bottle weight)",
                "initial_support_target_n": initial_support_target_n,
                "initial_support_target_rule": (
                    "max(measured table load deficit, 1% bottle weight); live contact allocation "
                    "is recomputed from current contacts and Jacobians"
                ),
                "stages": [],
                "load_ready": None,
            }
        if not load_transfer_only:
            if lift_height_m < 0.030 - 1e-9:
                raise GateFailure("Simulation-only pickup is bounded to the requested 30 mm lift")
            stage_targets = [
                (0.0001, "TRANSFER_0P1MM"),
                (0.00025, "TRANSFER_0P25MM"),
                (0.0005, "TABLE_UNLOAD_0P5MM"),
                (0.0010, "LIFT_1MM"),
                (0.0050, "LIFT_5MM"),
                (lift_height_m, f"LIFT_{int(round(lift_height_m * 1000))}MM"),
            ]
            previous_stage_height = 0.0
            previous_arm_target = approach_q.copy()
            live_load_ready = False
            for stage_height, stage_name in stage_targets:
                current_load_stage_name = stage_name
                segment_fraction = max((stage_height - previous_stage_height) / lift_height_m, 0.0)
                stage_duration = max(
                    0.100 if stage_height <= 0.0005 else 0.050,
                    LIFT_TIMEOUT_SECONDS * segment_fraction,
                )
                stage_steps = max(1, int(math.ceil(stage_duration / DT)))
                stage_arm_target = approach_q + (stage_height / lift_height_m) * (lift_q - approach_q)

                def current_support_demand_n() -> float:
                    current_contacts = bottle_contacts(model, data)
                    current_table_normal = sum(
                        c["true_normal_force_n"] for c in current_contacts
                        if c["side"] == "other" and c["other_body"] == "m0_table"
                    )
                    return gates.progressive_support_demand_n(
                        current_table_normal, bottle_weight_n
                    )

                for stage_step in range(1, stage_steps + 1):
                    alpha = stage_step / stage_steps
                    smooth_alpha = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
                    arm_target = previous_arm_target + smooth_alpha * (
                        stage_arm_target - previous_arm_target
                    )
                    set_body_ctrl(arm_motor_targets(arm_target))
                    effort_command = apply_simulation_only_effort(
                        float(data.time - load_build_start_time),
                        load_build_schedule=direct_arm_lift,
                        live_support_force_n=current_support_demand_n(),
                    )
                    mujoco.mj_step(model, data)
                    steps += 1
                    record_simulation_only_effort(effort_command, "LIFT")
                    sample("LIFT", 0.0, arm_target)
                    lift_contacts = bottle_contacts(model, data)
                    thumb, opposing, left_contact, _ = active_contact_sets(lift_contacts)
                    lift_window = contact_window_summary(list(contact_window_history))
                    bottle_table_supported = any(
                        c["side"] == "other" and c["other_body"] == "m0_table"
                        and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                        for c in lift_contacts
                    )
                    bottle_lift = float(data.xpos[bottle_id][2] - lift_origin_z)
                    table_normal = sum(
                        c["true_normal_force_n"] for c in lift_contacts
                        if c["side"] == "other" and c["other_body"] == "m0_table"
                    )
                    hand_force, hand_torque = bottle_hand_wrench(
                        lift_contacts, np.asarray(data.xipos[bottle_id], dtype=float)
                    )
                    lift_stage_trace_rows.append({
                        "time_s": float(data.time),
                        "stage": stage_name,
                        "stage_target_height_m": stage_height,
                        "bottle_lift_from_settled_m": bottle_lift,
                        "right_wrist_z_m": float(data.xpos[wrist_id][2]),
                        "right_thumb_contact": int(thumb),
                        "right_opposing_digit_contact": int(opposing),
                        "left_hand_contact": int(left_contact),
                        "contact_window_valid": int(lift_window["valid"]),
                        "table_supported": int(bottle_table_supported),
                        "table_normal_force_n": table_normal,
                        "hand_vertical_support_force_n": float(hand_force[2]),
                        "hand_torque_about_com_world_nm": hand_torque.tolist(),
                    })
                    if left_contact:
                        raise GateFailure(f"Left hand/arm contacted the bottle during {stage_name}")
                    if direct_arm_lift and not live_load_ready \
                            and bottle_lift > LOAD_BUILD_MAX_BOTTLE_RISE_M + 1e-6:
                        failure = {
                            "time_s": float(data.time), "stage": stage_name,
                            "bottle_rise_m": bottle_lift,
                            "maximum_pre_load_ready_rise_m": LOAD_BUILD_MAX_BOTTLE_RISE_M,
                            "table_supported": bottle_table_supported,
                            "table_normal_force_n": table_normal,
                            "hand_vertical_support_force_n": float(hand_force[2]),
                        }
                        result["progressive_load_transfer"]["first_failure"] = failure
                        raise GateFailure(
                            "Bottle rise exceeded the unchanged pre-LOAD_READY 0.5 mm gate: "
                            + json.dumps(failure, sort_keys=True)
                        )
                    if not gates.contact_window_allows_transfer(lift_window, left_contact):
                        raise GateFailure(
                            f"Right-hand thumb/opposing contact gate failed during {stage_name}: "
                            + json.dumps(lift_window, sort_keys=True)
                        )
                    if stage_name in {"LIFT_1MM", "LIFT_5MM",
                                      f"LIFT_{int(round(lift_height_m * 1000))}MM"} \
                            and bottle_table_supported:
                        raise GateFailure(
                            f"Bottle remained table-supported at {stage_name} "
                            f"({bottle_lift:.6f} m lift)"
                        )

                for _ in range(int(round(LIVE_TRANSFER_STEP_DWELL_S / DT))):
                    set_body_ctrl(arm_motor_targets(stage_arm_target))
                    effort_command = apply_simulation_only_effort(
                        float(data.time - load_build_start_time),
                        load_build_schedule=direct_arm_lift,
                        live_support_force_n=current_support_demand_n(),
                    )
                    mujoco.mj_step(model, data)
                    steps += 1
                    record_simulation_only_effort(effort_command, "LIFT")
                    sample("LIFT", 0.0, stage_arm_target)

                stage_contacts = bottle_contacts(model, data)
                thumb, opposing, left_contact, _ = active_contact_sets(stage_contacts)
                lift_window = contact_window_summary(list(contact_window_history))
                bottle_table_supported = any(
                    c["side"] == "other" and c["other_body"] == "m0_table"
                    and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                    for c in stage_contacts
                )
                bottle_lift = float(data.xpos[bottle_id][2] - lift_origin_z)
                if not gates.contact_window_allows_transfer(lift_window, left_contact):
                    raise GateFailure(
                        f"Right-hand force-bearing contact was lost during {stage_name} dwell: "
                        + json.dumps({
                            "left_hand_contact": left_contact,
                            "thumb_contact": thumb,
                            "opposing_contact": opposing,
                            "contact_window": lift_window,
                        }, sort_keys=True)
                    )
                if direct_arm_lift and not live_load_ready \
                        and bottle_lift > LOAD_BUILD_MAX_BOTTLE_RISE_M + 1e-6:
                    failure = {
                        "time_s": float(data.time), "stage": stage_name,
                        "bottle_rise_m": bottle_lift,
                        "maximum_pre_load_ready_rise_m": LOAD_BUILD_MAX_BOTTLE_RISE_M,
                        "table_supported": bottle_table_supported,
                    }
                    result["progressive_load_transfer"]["first_failure"] = failure
                    raise GateFailure(
                        "Bottle rise exceeded the unchanged pre-LOAD_READY 0.5 mm gate "
                        "during stage dwell: " + json.dumps(failure, sort_keys=True)
                    )
                if stage_name in {"LIFT_1MM", "LIFT_5MM",
                                  f"LIFT_{int(round(lift_height_m * 1000))}MM"} \
                        and bottle_table_supported:
                    raise GateFailure(
                        f"Bottle regained table support during {stage_name} dwell "
                        f"({bottle_lift:.6f} m lift)"
                    )

                stage_window_start = float(data.time - LIVE_TRANSFER_STEP_DWELL_S / 2.0)
                stage_window_rows = [
                    row for row in load_transfer_rows
                    if row.get("stage") == stage_name
                    and float(row["time_s"]) >= stage_window_start
                ]
                transfer_stage_summary = None
                if direct_arm_lift and stage_height <= 0.0005:
                    transfer_stage_summary = gates.progressive_load_transfer_summary(
                        transfer_baseline_table_n,
                        transfer_baseline_table_sigma_n,
                        transfer_baseline_hand_n,
                        stage_window_rows,
                        bottle_weight_n,
                    )
                    transfer_stage_summary.update({
                        "stage": stage_name,
                        "stage_target_height_m": stage_height,
                        "time_s": float(data.time),
                    })
                    result["progressive_load_transfer"]["stages"].append(
                        transfer_stage_summary
                    )
                    if not transfer_stage_summary["passed"]:
                        result["progressive_load_transfer"].update({
                            "status": "FAIL",
                            "first_failure": transfer_stage_summary,
                        })
                        raise GateFailure(
                            f"Measured progressive transfer failed at {stage_name}: "
                            + json.dumps(transfer_stage_summary, sort_keys=True)
                        )
                    events.append({
                        "event": "PROGRESSIVE_TRANSFER_STAGE_PASS",
                        **transfer_stage_summary,
                    })
                    if ("UPWARD_HAND_SUPPORT_ESTABLISHED" not in {
                        event.get("event") for event in events
                    } and transfer_stage_summary["stage_mean_hand_vertical_support_force_n"] > 0.0):
                        events.append({
                            "event": "UPWARD_HAND_SUPPORT_ESTABLISHED",
                            "time_s": float(data.time),
                            "stage": stage_name,
                            "measured_hand_support_n": transfer_stage_summary[
                                "stage_mean_hand_vertical_support_force_n"
                            ],
                            "increase_over_previous_n": transfer_stage_summary[
                                "measured_hand_support_increase_n"
                            ],
                        })
                    baseline_table_values = np.asarray([
                        float(row["table_normal_force_n"]) for row in stage_window_rows
                    ], dtype=float)
                    baseline_hand_values = np.asarray([
                        float(row["hand_vertical_support_force_n"]) for row in stage_window_rows
                    ], dtype=float)
                    transfer_baseline_table_n = float(np.mean(baseline_table_values))
                    transfer_baseline_table_sigma_n = float(np.std(
                        baseline_table_values, ddof=1
                    )) if len(baseline_table_values) > 1 else 0.0
                    transfer_baseline_hand_n = float(np.mean(baseline_hand_values))
                    if stage_name == "TABLE_UNLOAD_0P5MM":
                        load_ready = gates.load_ready_summary(
                            stage_window_rows, bottle_weight_n
                        )
                        result["progressive_load_transfer"]["load_ready"] = load_ready
                        if not load_ready["passed"]:
                            result["progressive_load_transfer"].update({
                                "status": "FAIL",
                                "first_failure": {
                                    "stage": stage_name,
                                    "time_s": float(data.time),
                                    "load_ready": load_ready,
                                },
                            })
                            raise GateFailure(
                                "Measured LOAD_READY failed at the 0.5 mm transfer checkpoint: "
                                + json.dumps(load_ready, sort_keys=True)
                            )
                        live_load_ready = True
                        result["progressive_load_transfer"].update({
                            "status": "PASS",
                            "passed": True,
                            "load_ready_time_s": float(data.time),
                        })
                        result["load_build"].update({
                            "status": "PASS",
                            "state": "LOAD_READY",
                            "load_ready": True,
                            "ready_time_s": float(data.time),
                            "achieved_stable_dwell_s": LIVE_TRANSFER_STEP_DWELL_S / 2.0,
                            "ready_snapshot": load_ready,
                            "control_mode": "live measured-contact force allocation during progressive wrist motion",
                        })
                        result["direct_arm_lift_table_unload"] = {
                            "status": "PASS",
                            "first_stage": transfer_stage_summary,
                            "load_ready": load_ready,
                            "measurement": "measured table load fell as right-hand vertical support rose; airborne geometry confirmed",
                        }
                        events.append({
                            "event": "LOAD_READY",
                            "time_s": float(data.time),
                            "stable_dwell_s": LIVE_TRANSFER_STEP_DWELL_S / 2.0,
                            "snapshot": load_ready,
                        })

                achieved_lift = float(data.xpos[bottle_id][2] - lift_origin_z)
                stage_tolerance = 0.0001 if stage_height < 0.005 else 0.0002
                requires_bottle_height = stage_name in {
                    "LIFT_1MM", "LIFT_5MM", f"LIFT_{int(round(lift_height_m * 1000))}MM"
                }
                if requires_bottle_height and achieved_lift < stage_height - stage_tolerance:
                    raise GateFailure(
                        f"{stage_name} gate failed: achieved {achieved_lift:.6f} m, "
                        f"required {stage_height:.6f} m"
                    )
                stage_table_normal = sum(
                    c["true_normal_force_n"] for c in stage_contacts
                    if c["side"] == "other" and c["other_body"] == "m0_table"
                )
                stage_record = {
                    "stage": stage_name,
                    "time_s": float(data.time),
                    "target_height_m": stage_height,
                    "achieved_bottle_lift_m": achieved_lift,
                    "right_wrist_rise_m": float(
                        data.xpos[wrist_id][2] - load_build_stage3_wrist_start[2]
                    ),
                    "table_supported": bottle_table_supported,
                    "table_normal_force_n": stage_table_normal,
                    "right_thumb_contact": thumb,
                    "right_opposing_digit_contact": opposing,
                    "contact_window": lift_window,
                }
                lift_stage_trace_rows.append(stage_record)
                events.append({"event": stage_name + "_PASS", **stage_record})
                if stage_name == "LIFT_1MM":
                    events.append({
                        "event": "AIRBORNE_1MM_REACHED",
                        "time_s": float(data.time), "lift_m": achieved_lift,
                        "table_supported": bottle_table_supported,
                        "stage_record": stage_record,
                    })
                elif stage_name == "LIFT_5MM":
                    events.append({
                        "event": "AIRBORNE_5MM_REACHED",
                        "time_s": float(data.time), "lift_m": achieved_lift,
                        "table_supported": bottle_table_supported,
                        "stage_record": stage_record,
                    })
                elif stage_name == f"LIFT_{int(round(lift_height_m * 1000))}MM":
                    events.append({
                        "event": "SHORT_LIFT_REACHED",
                        "time_s": float(data.time), "lift_m": achieved_lift,
                        "table_supported": bottle_table_supported,
                        "stage_record": stage_record,
                    })
                image_label = {
                    "TRANSFER_0P1MM": "transfer_0p1mm",
                    "TRANSFER_0P25MM": "transfer_0p25mm",
                    "TABLE_UNLOAD_0P5MM": "table_unload_0p5mm",
                    "LIFT_1MM": "lift_1mm",
                    "LIFT_5MM": "lift_5mm",
                }.get(stage_name, "lift_30mm")
                cv2.imwrite(
                    str(output / f"{image_label}.png"),
                    render_frame(renderer, model, data, camera, stage_name.replace("_", " ")),
                )
                save_contact_views(renderer, model, data, camera, site_id, bottle_id,
                                   output, image_label)
                if stage_name == "TABLE_UNLOAD_0P5MM":
                    if direct_arm_lift and result.get("direct_arm_lift_table_unload", {}).get(
                        "status"
                    ) != "PASS":
                        result["direct_arm_lift_table_unload"] = {
                            "status": "IN PROGRESS",
                            "first_stage": stage_record,
                            "measurement": (
                                "bottle height and table support are sampled during commanded "
                                "arm motion; no stationary hand-only LOAD_BUILD is claimed"
                            ),
                        }
                    else:
                        load_summary = stage3_table_unload_summary(
                            load_build_trace_rows,
                            load_transfer_rows,
                            bottle_weight_n,
                            stage_record["right_wrist_rise_m"],
                        )
                        result["stage3_table_unload"] = load_summary
                        result["table_load_transfer"] = load_summary
                        if not load_summary["passed"]:
                            raise GateFailure(
                                "0.5 mm table-unload gate failed: "
                                + json.dumps(load_summary, sort_keys=True)
                            )
                previous_stage_height = stage_height
                previous_arm_target = stage_arm_target

            if direct_arm_lift:
                five_mm_rows = [
                    row for row in lift_stage_trace_rows
                    if row.get("stage") == "LIFT_5MM"
                ]
                result["direct_arm_lift_table_unload"].update({
                    "status": "PASS",
                    "five_mm_stage": five_mm_rows[-1] if five_mm_rows else None,
                    "thirty_mm_stage": stage_record,
                    "table_support_lost_by_5mm": bool(
                        five_mm_rows and not five_mm_rows[-1]["table_supported"]
                    ),
                    "measurement": (
                        "bottle separation and support loss were measured during commanded "
                        "arm motion; no stationary hand-only LOAD_BUILD is claimed"
                    ),
                })

            lift_label = int(round(lift_height_m * 1000))
            events.append({
                "event": "POST_LIFT_HOLD_START", "time_s": float(data.time),
                "duration_s": post_lift_hold_seconds, "arm_target_rad": lift_q.tolist(),
            })
            post_hold_steps = int(round(post_lift_hold_seconds / DT))
            for _ in range(post_hold_steps):
                set_body_ctrl(arm_motor_targets(lift_q))
                effort_command = apply_simulation_only_effort(
                    float(data.time - load_build_start_time),
                    load_build_schedule=direct_arm_lift,
                    live_support_force_n=(current_support_demand_n()
                                          if direct_arm_lift else None),
                )
                mujoco.mj_step(model, data)
                steps += 1
                record_simulation_only_effort(effort_command, "POST_LIFT_HOLD")
                sample("POST_LIFT_HOLD", 0.0, lift_q)
                post_contacts = bottle_contacts(model, data)
                thumb, opposing, left_contact, _ = active_contact_sets(post_contacts)
                post_window = contact_window_summary(list(contact_window_history))
                post_table_supported = any(
                    c["side"] == "other" and c["other_body"] == "m0_table"
                    and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                    for c in post_contacts
                )
                current_height = float(data.xpos[bottle_id][2] - lift_origin_z)
                if left_contact or post_table_supported or not thumb or not opposing:
                    raise GateFailure(
                        "Contact/table-support gate failed during 30 mm post-lift hold: "
                        f"left={left_contact}, table={post_table_supported}, "
                        f"thumb={thumb}, opposing={opposing}"
                    )
                if not post_window["valid"]:
                    raise GateFailure(
                        "5 ms contact-validity window failed during 30 mm post-lift hold: "
                        + json.dumps(post_window, sort_keys=True)
                    )
                if current_height < lift_height_m - 0.0002:
                    raise GateFailure(
                        f"Bottle dropped below 30 mm during post-lift hold at {current_height:.6f} m"
                    )
            events.append({
                "event": "POST_LIFT_HOLD_COMPLETE", "time_s": float(data.time),
                "duration_s": post_lift_hold_seconds,
                "lift_m": float(data.xpos[bottle_id][2] - lift_origin_z),
            })
            result["lift_stages"] = lift_stage_trace_rows
            reached = True
            lift_steps = 0
        else:
            lift_steps = int(round(LIFT_TIMEOUT_SECONDS / DT))
        for step in range(1, lift_steps + 1):
            alpha = step / lift_steps
            smooth_alpha = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
            arm_target = approach_q + smooth_alpha * (lift_q - approach_q)
            set_body_ctrl(arm_motor_targets(arm_target))
            last_applied_targets.update({channel: float(data.ctrl[aid])
                                         for channel, aid in hand_ids.items()})
            mujoco.mj_step(model, data)
            steps += 1
            sample("LIFT", 0.0, arm_target)
            stage3_wrist_rise = float(
                data.xpos[wrist_id][2] - load_build_stage3_wrist_start[2]
            )
            if stage3_wrist_rise > 0.0005 + LOAD_BUILD_WRIST_DRIFT_GATE_TOLERANCE_M:
                result["stage3_first_bound_failure"] = {
                    "time_s": float(data.time),
                    "actual_wrist_rise_m": stage3_wrist_rise,
                    "maximum_wrist_rise_m": 0.0005,
                    "measurement_tolerance_m": LOAD_BUILD_WRIST_DRIFT_GATE_TOLERANCE_M,
                }
                raise GateFailure(
                    f"Stage 3 wrist rise exceeded 0.5 mm plus measurement tolerance: {stage3_wrist_rise:.9g} m"
                )
            lift_height = float(data.xpos[bottle_id][2] - lift_origin_z)
            lift_window = contact_window_summary(list(contact_window_history))
            thumb, opposing, left_contact, _ = active_contact_sets(bottle_contacts(model, data))
            lift_contacts = bottle_contacts(model, data)
            bottle_table_supported = any(
                c["side"] == "other" and c["other_body"] == "m0_table"
                and c["distance_m"] <= 0.0 and c["true_normal_force_n"] > 0.01
                for c in lift_contacts
            )
            if left_contact:
                raise GateFailure("Left hand/arm contacted the bottle during lift")
            if not lift_window["valid"]:
                result["first_invalid_contact_window"] = {
                    "time_s": float(data.time), "phase": "LIFT", "lift_height_m": lift_height,
                    **lift_window,
                }
                raise GateFailure(
                    "5 ms contact-validity window failed during lift: "
                    + json.dumps(result["first_invalid_contact_window"], sort_keys=True)
                )
            if not load_transfer_only and lift_height >= 0.005 and bottle_table_supported:
                raise GateFailure(
                    f"Bottle remained table-supported after lift began at {lift_height:.6f} m"
                )
            load_target_complete = load_transfer_only and step == lift_steps
            bottle_lift_complete = not load_transfer_only and lift_height >= lift_height_m
            if load_target_complete or bottle_lift_complete:
                if not lift_window["valid"]:
                    raise GateFailure(
                        "Selected 5 ms contact-validity window was not valid at lift target"
                    )
                if not load_transfer_only and bottle_table_supported:
                    raise GateFailure(
                        f"Bottle reached {lift_height_m * 1000:.0f} mm while still table-supported"
                    )
                reached = True
                lift_label = int(round(lift_height_m * 1000))
                events.append({
                    "event": "LOAD_TRANSFER_ARM_TARGET_REACHED" if load_transfer_only
                    else "SHORT_LIFT_REACHED",
                    "time_s": float(data.time), "target_lift_m": lift_height_m,
                    "lift_m": lift_height, "right_thumb_contact": thumb,
                    "right_opposing_digit_contact": opposing,
                    "contact_window": lift_window,
                    "table_supported": bottle_table_supported,
                    "table_normal_force_n": sum(
                        c["true_normal_force_n"] for c in lift_contacts
                        if c["side"] == "other" and c["other_body"] == "m0_table"
                    ),
                })
                image_label = f"load_transfer_{lift_label}mm" if load_transfer_only else f"lift_{lift_label}mm"
                cv2.imwrite(str(output / f"{image_label}.png"),
                            render_frame(renderer, model, data, camera,
                                         f"{lift_label} mm LOAD TRANSFER" if load_transfer_only
                                         else f"SMOOTH LIFT {lift_label} mm"))
                save_contact_views(renderer, model, data, camera, site_id, bottle_id,
                                   output, image_label)
                if post_lift_hold_seconds > 0.0:
                    events.append({"event": "POST_LIFT_HOLD_START", "time_s": float(data.time),
                                   "duration_s": post_lift_hold_seconds,
                                   "arm_target_rad": lift_q.tolist()})
                    post_hold_steps = int(round(post_lift_hold_seconds / DT))
                    for _ in range(post_hold_steps):
                        set_body_ctrl(arm_motor_targets(lift_q))
                        last_applied_targets.update({channel: float(data.ctrl[aid])
                                                     for channel, aid in hand_ids.items()})
                        mujoco.mj_step(model, data)
                        steps += 1
                        sample("POST_LIFT_HOLD", 0.0, lift_q)
                        post_contacts = bottle_contacts(model, data)
                        thumb, opposing, left_contact, _ = active_contact_sets(post_contacts)
                        post_window = contact_window_summary(list(contact_window_history))
                        post_table_supported = any(
                            c["side"] == "other" and c["other_body"] == "m0_table"
                            and c["distance_m"] <= 0.0 and c["normal_force_n"] > 0.01
                            for c in post_contacts
                        )
                        current_height = float(data.xpos[bottle_id][2] - lift_origin_z)
                        if left_contact:
                            raise GateFailure("Left hand/arm contacted the bottle during post-lift hold")
                        if not post_window["valid"]:
                            result["first_invalid_contact_window"] = {
                                "time_s": float(data.time), "phase": "POST_LIFT_HOLD",
                                "lift_height_m": current_height, **post_window,
                            }
                            raise GateFailure(
                                "5 ms contact-validity window failed during post-lift hold: "
                                + json.dumps(result["first_invalid_contact_window"], sort_keys=True)
                            )
                        if not load_transfer_only and post_table_supported:
                            raise GateFailure(
                                f"Bottle regained table support during {lift_label} mm post-lift hold"
                            )
                        if not load_transfer_only and current_height < lift_height_m:
                            raise GateFailure(
                                f"Bottle dropped below {lift_label} mm during post-lift hold "
                                f"at {current_height:.6f} m"
                            )
                    events.append({"event": "POST_LIFT_HOLD_COMPLETE", "time_s": float(data.time),
                                   "duration_s": post_lift_hold_seconds,
                                   "lift_m": float(data.xpos[bottle_id][2] - lift_origin_z)})
                if load_transfer_only:
                    load_summary = stage3_table_unload_summary(
                        load_build_trace_rows,
                        load_transfer_rows,
                        bottle_weight_n,
                        float(data.xpos[wrist_id][2] - load_build_stage3_wrist_start[2]),
                    )
                    result["stage3_table_unload"] = load_summary
                    result["table_load_transfer"] = load_summary
                    if not load_summary["passed"]:
                        raise GateFailure(
                            "Stage 3 table-unload gate failed: "
                            + json.dumps(load_summary, sort_keys=True)
                        )
                break
        if not reached:
            value = float(data.xpos[bottle_id][2] - lift_origin_z)
            raise GateFailure(f"Lift gate failed: achieved {value:.6f} m within {LIFT_TIMEOUT_SECONDS:.2f}s")

        release_result = {"status": "NOT RUN"}
        peak_lift_before_release = float(data.xpos[bottle_id][2] - lift_origin_z)
        if not load_transfer_only:
            release_start_time = float(data.time)
            release_result = {
                "status": "RUNNING",
                "start_time_s": release_start_time,
                "open_target_slew_rad_s": OPEN_TARGET_SLEW_RAD_S,
                "open_target_tolerance_rad": RELEASE_OPEN_TARGET_TOLERANCE_RAD,
                "table_settle_dwell_s": RELEASE_SETTLE_DWELL_SECONDS,
                "timeout_s": RELEASE_TIMEOUT_SECONDS,
                "bottle_qpos_writes": 0,
                "bottle_attachment": False,
            }
            events.append({"event": "RELEASE_START", "time_s": release_start_time})
            release_targets = {}
            for channel, aid in hand_ids.items():
                joint = channel_joint_names[channel]
                current = joint_qpos(model, data, joint)
                release_targets[channel] = current
                model.actuator_gainprm[aid, :] = hand_position_gainprm[aid]
                model.actuator_biasprm[aid, :] = hand_position_biasprm[aid]
                data.ctrl[aid] = current
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            maximum_release_steps = int(math.ceil(RELEASE_TIMEOUT_SECONDS / DT))
            clear_since: float | None = None
            stable_since: float | None = None
            fingers_opened_time: float | None = None
            contact_cleared_time: float | None = None
            release_completed = False
            for release_step in range(maximum_release_steps):
                for channel, aid in hand_ids.items():
                    target = open_targets[channel]
                    current_target = release_targets[channel]
                    delta = target - current_target
                    step_limit = OPEN_TARGET_SLEW_RAD_S * DT
                    release_targets[channel] = current_target + float(
                        np.clip(delta, -step_limit, step_limit)
                    )
                    data.ctrl[aid] = release_targets[channel]
                set_body_ctrl(arm_motor_targets(lift_q))
                data.qfrc_applied[:] = 0.0
                mujoco.mj_step(model, data)
                steps += 1
                sample("RELEASE", None, lift_q)

                contacts = bottle_contacts(model, data)
                right_hand_contact = any(
                    contact["side"] == "right_hand"
                    and contact["distance_m"] <= 0.0
                    and contact["true_normal_force_n"] > 0.01
                    for contact in contacts
                )
                table_supported = any(
                    contact["side"] == "other"
                    and contact["other_body"] == "m0_table"
                    and contact["distance_m"] <= 0.0
                    and contact["true_normal_force_n"] > 0.01
                    for contact in contacts
                )
                open_errors = {
                    channel: abs(joint_qpos(model, data, channel_joint_names[channel])
                                 - open_targets[channel])
                    for channel in hand_ids
                }
                all_fingers_open = max(open_errors.values(), default=0.0) \
                    <= RELEASE_OPEN_TARGET_TOLERANCE_RAD
                if all_fingers_open and fingers_opened_time is None:
                    fingers_opened_time = float(data.time)
                    events.append({
                        "event": "FINGERS_OPENED",
                        "time_s": fingers_opened_time,
                        "maximum_driver_open_error_rad": max(open_errors.values(), default=0.0),
                    })
                if not right_hand_contact:
                    if clear_since is None:
                        clear_since = float(data.time)
                        contact_cleared_time = clear_since
                        events.append({"event": "BOTTLE_CONTACT_CLEARED", "time_s": clear_since})
                else:
                    clear_since = None
                    contact_cleared_time = None

                velocity = np.zeros(6, dtype=float)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, velocity, 0
                )
                linear_speed = float(np.linalg.norm(velocity[3:]))
                angular_speed = float(np.linalg.norm(velocity[:3]))
                stable = (
                    all_fingers_open
                    and not right_hand_contact
                    and table_supported
                    and linear_speed <= SETTLE_MAX_LINEAR_SPEED_M_S
                    and angular_speed <= SETTLE_MAX_ANGULAR_SPEED_RAD_S
                )
                if stable:
                    if stable_since is None:
                        stable_since = float(data.time)
                else:
                    stable_since = None

                if release_step % 10 == 0 or stable:
                    release_trace_rows.append({
                        "time_s": float(data.time),
                        "phase": "RELEASE",
                        "right_hand_bottle_contact": int(right_hand_contact),
                        "table_supported": int(table_supported),
                        "all_fingers_open": int(all_fingers_open),
                        "maximum_open_target_error_rad": max(open_errors.values(), default=0.0),
                        "bottle_x_m": float(data.xpos[bottle_id][0]),
                        "bottle_y_m": float(data.xpos[bottle_id][1]),
                        "bottle_z_m": float(data.xpos[bottle_id][2]),
                        "bottle_lift_from_settled_m": float(
                            data.xpos[bottle_id][2] - lift_origin_z
                        ),
                        "linear_speed_m_s": linear_speed,
                        "angular_speed_rad_s": angular_speed,
                        "stable_dwell_s": 0.0 if stable_since is None
                        else float(data.time - stable_since),
                    })
                if stable_since is not None and data.time - stable_since >= RELEASE_SETTLE_DWELL_SECONDS:
                    release_completed = True
                    events.append({
                        "event": "RELEASE_SETTLED",
                        "time_s": float(data.time),
                        "stable_dwell_s": float(data.time - stable_since),
                        "bottle_position_world_m": data.xpos[bottle_id].tolist(),
                    })
                    break

            release_result.update({
                "status": "PASS" if release_completed else "FAIL",
                "passed": release_completed,
                "fingers_opened_time_s": fingers_opened_time,
                "contact_cleared_time_s": contact_cleared_time,
                "time_s": float(data.time),
                "duration_s": float(data.time - release_start_time),
                "maximum_open_target_error_rad": max(
                    (abs(joint_qpos(model, data, channel_joint_names[channel])
                         - open_targets[channel]) for channel in hand_ids), default=0.0
                ),
                "final_right_hand_bottle_contact": any(
                    contact["side"] == "right_hand"
                    and contact["distance_m"] <= 0.0
                    and contact["true_normal_force_n"] > 0.01
                    for contact in bottle_contacts(model, data)
                ),
                "final_table_supported": any(
                    contact["side"] == "other"
                    and contact["other_body"] == "m0_table"
                    and contact["distance_m"] <= 0.0
                    and contact["true_normal_force_n"] > 0.01
                    for contact in bottle_contacts(model, data)
                ),
                "final_bottle_position_world_m": data.xpos[bottle_id].tolist(),
                "peak_lift_before_release_m": peak_lift_before_release,
            })
            if not release_completed:
                raise GateFailure(
                    "Physical release did not produce open fingers, contact clearance, and a "
                    "table-supported stable bottle within the bounded release window: "
                    + json.dumps(release_result, sort_keys=True)
                )
            events.append({"event": "RELEASE_COMPLETE", **release_result})

        lift_label = int(round(lift_height_m * 1000))
        final_window = contact_window_summary(list(contact_window_history))
        thumb = bool(final_window.get("thumb_qualifies", False))
        opposing = bool(final_window.get("valid_opposing_digits", []))
        left_contact = left_contact_seen
        final_force = active_contact_sets(bottle_contacts(model, data))[3]
        stage_gate_passed = (
            bool(result.get("table_load_transfer", {}).get("passed"))
            if load_transfer_only else bool(final_window["valid"])
        )
        result.update({
            "status": "PASS" if stage_gate_passed and not left_contact and release_result.get("passed", False) else "FAIL",
            "passed": bool(stage_gate_passed and not left_contact and release_result.get("passed", False)),
            "completed_stage": (
                "TABLE_UNLOAD_0P5MM_STOP" if load_transfer_only and lift_height_m <= 0.0005001
                else "LOAD_TRANSFER_1MM_HOLD_STOP" if load_transfer_only
                else f"SHORT_LIFT_{lift_label}MM_HOLD_STOP"
            ), "physics_steps": steps,
            "physics_time_s": float(data.time), "initial_bottle_pose_qpos": initial_bottle_qpos.tolist(),
            "final_bottle_position_world_m": data.xpos[bottle_id].tolist(),
            "lift_height_m": peak_lift_before_release,
            "peak_lift_before_release_m": peak_lift_before_release,
            "release": release_result,
            "lift_target_height_m": lift_height_m,
            "post_lift_hold_seconds": post_lift_hold_seconds,
            "final_right_thumb_contact": thumb, "final_right_opposing_digit_contact": opposing,
            "final_contact_window": final_window,
            "load_transfer_only": load_transfer_only,
            "left_hand_bottle_contact": left_contact_seen, "final_contact_force_n": final_force,
            "maximum_hand_joint_qvel_rad_s": max_qvel, "maximum_hand_joint_qvel_joint": max_qvel_joint,
            "first_velocity_breach": first_velocity_breach, "maximum_mimic_error_rad": max_mimic,
            "maximum_mimic_diagnostic_error_rad": max_mimic_diagnostic,
            "mimic_diagnostic_threshold_exceed_steps": mimic_diagnostic_exceed_steps,
            "first_mimic_diagnostic_threshold_exceed": first_mimic_diagnostic_exceed,
            "maximum_joint_limit_violation_rad": max_limit, "maximum_object_contact_force_n": max_force,
            "maximum_object_penetration_m": max_penetration, "no_bottle_qpos_writes": True,
            "active_rollout_qpos_write_scan": qpos_check, "events": events,
            "final_contacts": bottle_contacts(model, data),
        })
    finally:
        result["events"] = events
        result["first_velocity_breach"] = first_velocity_breach
        result["measured_hand_motion"] = hand_motion_by_phase
        result["contact_aware_close"] = {
            **result.get("contact_aware_close", {}),
            "enabled": True,
            "contact_trigger_force_n": CONTACT_TRIGGER_FORCE_N,
            "contact_confirm_samples": CONTACT_CONFIRM_SAMPLES,
            "contact_loss_confirm_samples": CONTACT_LOSS_CONFIRM_SAMPLES,
            "channel_transitions": contact_hold_transition_times,
            "channel_preload_targets_rad": contact_hold_targets,
            "channel_preload_offsets_rad": contact_preload_offsets,
            "digit_states_at_stop": contact_controller_state,
            "maximum_contact_preload_target_error_rad": maximum_contact_target_error_rad,
            "minimum_contact_preload_target_error_rad": (
                None if not math.isfinite(minimum_contact_target_error_rad)
                else minimum_contact_target_error_rad
            ),
            "maximum_contact_commanded_error_rad": maximum_contact_commanded_error_rad,
            "maximum_hand_actuator_effort_nm": maximum_hand_actuator_effort_nm,
            "close_command_at_stop": last_close_command_u,
        }
        (output / "live_contact_allocation_trace.json").write_text(
            json.dumps(live_contact_allocation_trace, indent=2), encoding="utf-8"
        )
        result["live_contact_allocation_trace"] = {
            "path": "live_contact_allocation_trace.json",
            "sample_count": len(live_contact_allocation_trace),
        }
        writer.release()
        renderer.close()
        write_csv(output / "hand_command_joint_mimic_trace.csv", hand_rows)
        write_csv(output / "contact_trace.csv", contact_rows)
        write_csv(output / "per_digit_contact_force.csv", digit_force_rows)
        write_csv(output / "contact_aware_close_trace.csv", contact_aware_rows)
        write_csv(output / "mimic_error_trace.csv", mimic_rows)
        write_csv(output / "settle_trace.csv", settle_rows)
        write_csv(output / "bottle_pose_trace.csv", bottle_rows)
        write_csv(output / "right_arm_trace.csv", arm_rows)
        write_csv(output / "hold_lift_transition_trace.csv", transition_trace_rows)
        write_csv(output / "lift_arming_trace.csv", lift_arming_trace_rows)
        write_csv(output / "contact_window_trace.csv", contact_window_rows)
        write_csv(output / "load_transfer_trace.csv", load_transfer_rows)
        write_csv(output / "load_build_trace.csv", load_build_trace_rows)
        write_csv(output / "effort_control_trace.csv", effort_control_trace_rows)
        write_csv(output / "lift_stage_trace.csv", lift_stage_trace_rows)
        write_csv(output / "release_trace.csv", release_trace_rows)
        if effort_control_trace_rows:
            result["maximum_simulation_only_driver_effort_nm"] = max(
                abs(float(row[f"{channel}_applied_nm"]))
                for row in effort_control_trace_rows for channel in hand_ids
            )
            result["maximum_simulation_only_requested_effort_nm"] = max(
                abs(float(row[f"{channel}_requested_nm"]))
                for row in effort_control_trace_rows for channel in hand_ids
            )
            result["simulation_only_effort_cap_saturation_samples"] = sum(
                int(row[f"{channel}_saturated"])
                for row in effort_control_trace_rows for channel in hand_ids
            )
        result["simulation_only_effort_cap_nm"] = SIM_ONLY_MAX_DRIVER_TORQUE_NM
        result["position_hand_actuators_disabled_after_load_start"] = effort_mode_active
        result["no_follower_actuators_or_qpos_writes"] = True
    return {"result": result, "physics_steps": steps}


def write_legacy_report(output: Path, result: dict, command: str) -> None:
    scope = (
        "This bounded audit covers SETTLE -> PREGRASP -> APPROACH -> CLOSE -> HOLD -> "
        "LIFT_ARMING -> LOAD_BUILD -> LOAD_READY -> STOP. No wrist rise, lift, transfer, release, or placement was run."
        if result.get("stop_after_load_build") else (
        "This bounded audit covers SETTLE -> PREGRASP -> APPROACH -> CLOSE -> HOLD -> "
        "LIFT_ARMING -> LOAD_BUILD -> LOAD_READY -> a maximum 0.5 mm quasi-static wrist rise -> STOP. "
        "No transfer, release, or placement was run."
        if result.get("load_capacity_audit")
        else (
            "This contact-only packet covers SETTLE -> PREGRASP -> OPEN -> APPROACH -> CLOSE -> HOLD 1 s -> STOP. "
            "No lift or later manipulation stage was run."
            if result.get("contact_only_diagnostic") else (
                "This packet covers the existing bounded short-grasp sequence; see the run metadata for its stage limit."
            )
        )
        )
    )
    hold_contact = (
        f"{result['hold_valid_contact_steps']}/{result['hold_steps']} steps "
        f"({result['hold_valid_contact_fraction']:.1%})"
        if "hold_steps" in result else "not reached; run stopped before HOLD"
    )
    hold_support = (
        f"{result['hold_table_support_steps']}/{result['hold_steps']} steps"
        if "hold_steps" in result else "not reached; run stopped before HOLD"
    )
    view_files = sorted(path.name for pattern in ("close_*.png", "hold_*.png")
                        for path in output.glob(pattern))
    view_files.extend(sorted(path.name for path in output.glob("lift_*.png")))
    view_summary = ", ".join(view_files) if view_files else (
        "not produced before the first gate stopped the run"
    )
    settle = result.get("bottle_settle", {})
    video_status = (
        "MP4 ends at the bounded 0.5 mm table-unload stop."
        if result.get("completed_stage") == "TABLE_UNLOAD_0P5MM_STOP"
        else "MP4 ends at the first gate failure or requested stage stop; no later lift is claimed."
    )
    lines = [
        "# DFQ Short-Grasp Checkpoint", "",
        f"Status: {result['status']}", "",
        "## Scope", "",
        scope,
        "The accepted G1+DFQ scene, hand meshes, scale, wrist transforms, converter output, table, and bottle were not edited. The bottle remains free-jointed; no equality or mocap attachment references it. The sole weld fixes the pelvis.",
        "The accepted six-channel right-hand mapping, servo parameters, mimic constraints, and OPEN/FULL targets are reused. OPEN keeps its 0.075 rad/s simulation-derived slew; CLOSE uses the requested 0.060 rad/s target slew without changing its final target. Right-arm control uses the pinned humanoid_vla G1 motor-controller gains and bias compensation.",
        "The bottle starts only after a free-physics SETTLE phase with the arm in a statically verified clear pose. No bottle qpos writes or teleporting occur.", "",
        "## Results", "",
        f"- Settle: {settle.get('status', 'not reached')}; elapsed {settle.get('elapsed_s', 'n/a')} s; stable dwell {settle.get('stable_dwell_s', 'n/a')} s.",
        f"- Settle criterion: continuous table support, linear speed <= {SETTLE_MAX_LINEAR_SPEED_M_S:.3f} m/s, angular speed <= {SETTLE_MAX_ANGULAR_SPEED_RAD_S:.3f} rad/s for {SETTLE_DWELL_SECONDS:.2f} s (timeout {SETTLE_MAX_SECONDS:.2f} s).",
        f"- Maximum settle linear/angular speeds: {settle.get('maximum_linear_speed_m_s', 'n/a')} m/s / {settle.get('maximum_angular_speed_rad_s', 'n/a')} rad/s.",
        f"- Wrist yaw correction: {WRIST_YAW_CORRECTION_DEG:.1f} degrees; frozen pregrasp/approach targets retained; PREGRASP dwell after settling {PREGRASP_SECONDS:.2f} s, approach duration {APPROACH_SECONDS:.2f} s.",
        f"- CLOSE target slew: {CLOSE_TARGET_SLEW_RAD_S:.3f} rad/s; OPEN target slew: {OPEN_TARGET_SLEW_RAD_S:.3f} rad/s.",
        f"- Contact trigger: right-hand digit/bottle distance <= 0 m and single-contact normal force > {CONTACT_TRIGGER_FORCE_N:.3f} N for {CONTACT_CONFIRM_SAMPLES} consecutive physics samples ({CONTACT_CONFIRM_SAMPLES * DT:.4f} s). Contact loss requires {CONTACT_LOSS_CONFIRM_SAMPLES} absent samples.",
        f"- Preload rule: after each physics step, project the next target by joint closure sign so the directional target error is in [0, {CONTACT_PRELOAD_MAX_RAD:.3f}] rad; the projection itself uses no gate tolerance.",
        f"- Lift arming: continuous force-bearing thumb plus at least one opposing digit above {CONTACT_TRIGGER_FORCE_N:.3f} N, bottle table-supported, bottle speed within settle bounds, and hand/mimic/limit gates passing for {LIFT_ARMING_DWELL_SECONDS:.3f} s ({int(math.ceil(LIFT_ARMING_DWELL_SECONDS / DT))} consecutive physics samples); timeout {LIFT_ARMING_TIMEOUT_SECONDS:.2f} s.",
        f"- Lift arming result: {json.dumps(result.get('lift_arming', {}), sort_keys=True)}.",
        f"- Stage 1 wrench basis: {json.dumps(result.get('stage1_wrench_audit', {}), sort_keys=True)}.",
        f"- LOAD_BUILD result: {json.dumps(result.get('load_build', {}), sort_keys=True)}.",
        f"- Stage 3 table-unload result: {json.dumps(result.get('stage3_table_unload', {}), sort_keys=True)}.",
        "- `load_build_trace.csv` records the bounded preload ramp, per-digit true normal force, hand force/torque, wrist drift, object pose, and all hand limits.",
        "- `load_transfer_trace.csv` records the table normal load, absolute hand-supported fraction, hand vertical force, bottle motion, grasp contacts, and wrist motion throughout the 0.5 mm test.",
        "- Per-physics-step force-bearing arming state, bottle/table state, hand state, per-channel targets/errors/velocities/efforts, and available MuJoCo contact points/frame axes are in `lift_arming_trace.csv`.",
        f"- Measured preload gate: [-{CONTACT_PRELOAD_GATE_TOLERANCE_RAD:.1g}, {CONTACT_PRELOAD_MAX_RAD + CONTACT_PRELOAD_GATE_TOLERANCE_RAD:.9g}] rad; the {CONTACT_PRELOAD_GATE_TOLERANCE_RAD:.1g} rad tolerance is used only for measurement comparison.",
        f"- Controller transitions: {json.dumps(result.get('contact_aware_close', {}).get('transition_events', []), sort_keys=True)}.",
        f"- Commanded preload error maximum: {result.get('maximum_contact_commanded_error_rad', 'not reached')} rad; measured directional error min/max: {result.get('minimum_contact_preload_target_error_rad', 'not reached')} / {result.get('maximum_contact_preload_target_error_rad', 'not reached')} rad; peak hand actuator effort: {result.get('maximum_hand_actuator_effort_nm', 'not reached')} Nm (existing limit {HAND_MAX_TORQUE:.3f} Nm).",
        "- Per-physics-step next target, target applied during the completed step, actual position, directional errors, qvel, actuator effort, digit bottle-contact force, bottle pose, state, and transition time are in `contact_aware_close_trace.csv`; the same states and exact transition times are recorded in result JSON events.",
        f"- Simultaneous thumb/opposing contact during HOLD: {hold_contact}.",
        f"- Bottle table support during HOLD: {hold_support}.",
        f"- Lift target: {result.get('lift_target_height_m', 'not reached')} m; achieved {result.get('lift_height_m', 'not reached')} m.",
        f"- Load-transfer-only mode: {result.get('load_transfer_only', False)}; table-load summary: {json.dumps(result.get('table_load_transfer', {}), sort_keys=True)}.",
        f"- Contact-validity window: {json.dumps(result.get('final_contact_window', result.get('first_invalid_contact_window', {})), sort_keys=True)}.",
        f"- Lift profile: {json.dumps(result.get('lift_trajectory', {}), sort_keys=True)}.",
        f"- Right thumb contact at stop: {result.get('final_right_thumb_contact', False)}.",
        f"- Right opposing digit contact at stop: {result.get('final_right_opposing_digit_contact', False)}.",
        f"- Left bottle contact: {result.get('left_hand_bottle_contact', False)}.",
        f"- Peak hand speed: {result.get('maximum_hand_joint_qvel_rad_s', 'not reached')} rad/s, {result.get('maximum_hand_joint_qvel_joint', 'n/a')}.",
        f"- Peak mimic error: {result.get('maximum_mimic_error_rad', 'not reached')} rad; peak contact force: {result.get('maximum_object_contact_force_n', 'not reached')} N.",
        f"- Mimic limits: {MIMIC_DIAGNOSTIC_TOLERANCE_RAD:.3f} rad diagnostic trace; {MIMIC_MANIPULATION_LIMIT_RAD:.3f} rad manipulation gate; diagnostic exceed steps: {result.get('mimic_diagnostic_threshold_exceed_steps', 'not reached')}.",
        f"- First gate failure: {result.get('error', 'none')}.",
        f"- CLOSE/HOLD view PNGs: {view_summary}.", "",
        "## Reproduction", "", "Exact command:", "", command, "",
        "## Files", "",
        f"Generated packet files are enumerated in the output directory. {video_status}",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output: Path, result: dict, command: str) -> None:
    load_build = result.get("load_build", {})
    settle = result.get("bottle_settle", {})
    stage_names = [row.get("stage") for row in result.get("lift_stages", [])
                   if isinstance(row, dict) and "target_height_m" in row]
    lines = [
        "# G1 Inspire DFQ Simulation-Only M0 Pickup", "",
        f"Status: {result.get('status', 'UNKNOWN')}", "",
        "This is a SIMULATION_ONLY_M0 actuator abstraction. The real Inspire DFQ hardware command interface was not established by the inspected sources; this result makes no Sim2Real claim.",
        "The accepted DFQ meshes, scale, converter, wrist mounts, grasp geometry, bottle, table, friction, mimic topology, source joint limits, and source velocity gate were not changed.",
        "The bottle remains a free body. The active rollout contains no bottle/follower qpos writes, no bottle equality/weld/mocap/follow-hand attachment, and no left-hand assistance.", "",
        "## Controller", "",
        f"- Free-space approach and closure: existing six right-driver position actuators, kp={HAND_KP:g} Nm/rad, kv={HAND_KV:g} Nms/rad, +/-{HAND_MAX_TORQUE:.3f} Nm.",
        f"- Load, hold, and lift: position actuators disabled; qfrc_applied impedance with the same kp/kv, MuJoCo qfrc_bias compensation, and one smooth {SIM_ONLY_FEEDFORWARD_RAMP_S:.3f} s feedforward ramp.",
        f"- Static feedforward: effort-balanced wrench allocation with 10% friction-cone slack; selected cap {SIM_ONLY_MAX_DRIVER_TORQUE_NM:.3f} Nm, 10.20% above the worst saved bounded-uncertainty effort {result.get('stage1_wrench_audit', {}).get('robustness_summary', {}).get('effort_balanced_worst_peak_effort_nm', 'n/a')} Nm.",
        "- Only six right-hand proximal driver DoFs receive the generalized effort. Mimic followers remain equality-coupled and are never actuated or written.", "",
        "## Results", "",
        f"- Settle: {settle.get('status', 'not reached')}; stable dwell {settle.get('stable_dwell_s', 'n/a')} s.",
        f"- Load build: {load_build.get('status', 'not reached')}; measured physical support snapshot: {json.dumps(load_build.get('ready_snapshot', load_build.get('last_sample', {})), sort_keys=True)}.",
        f"- Lift milestones: {', '.join(stage_names) if stage_names else 'not reached'}.",
        f"- Final bottle lift: {result.get('lift_height_m', 'not reached')} m; 30 mm hold requested: {result.get('post_lift_hold_seconds', 'not reached')} s.",
        f"- Peak applied driver effort: {result.get('maximum_simulation_only_driver_effort_nm', 'not reached')} Nm; requested peak {result.get('maximum_simulation_only_requested_effort_nm', 'not reached')} Nm; cap saturation samples {result.get('simulation_only_effort_cap_saturation_samples', 'not reached')}.",
        f"- Peak hand speed: {result.get('maximum_hand_joint_qvel_rad_s', 'not reached')} rad/s ({result.get('maximum_hand_joint_qvel_joint', 'n/a')}); peak mimic error {result.get('maximum_mimic_error_rad', 'not reached')} rad; peak penetration {result.get('maximum_object_penetration_m', 'not reached')} m.",
        f"- Peak contact force: {result.get('maximum_object_contact_force_n', 'not reached')} N; left-hand contact: {result.get('left_hand_bottle_contact', 'not reached')}.",
        f"- First failed gate: {result.get('error', 'none')}.", "",
        "## Evidence", "",
        "`m0_result.json` contains the run status, exact reproduction command, provenance, thresholds, ordered events, and gate summaries. `actuator_control_mapping.json` distinguishes URDF/source facts from simulation-derived parameters.",
        "`effort_control_trace.csv` records per-physics-step feedforward, impedance, bias compensation, requested/applied driver efforts, qpos/qvel/qacc, and measured bottle-contact wrench/friction. `load_build_trace.csv`, `lift_stage_trace.csv`, `contact_trace.csv`, `per_digit_contact_force.csv`, `bottle_pose_trace.csv`, and `mimic_error_trace.csv` preserve stage evidence.",
        "The MP4 and PNGs show the scene through the last reached gate. Failed runs do not claim later milestones.", "",
        "## Reproduction", "", command, "",
        "## Scope Boundary", "",
        "No transfer, lowering, release, placement, or settle-after-release was run.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--contact-only", action="store_true")
    parser.add_argument("--lift-height-m", type=float, default=0.030)
    parser.add_argument("--post-lift-hold-s", type=float, default=1.0)
    parser.add_argument("--load-transfer-only", action="store_true")
    parser.add_argument("--stop-after-load-build", action="store_true")
    parser.add_argument("--wrench-audit-json", type=Path, required=True)
    args = parser.parse_args()
    if args.stop_after_load_build and args.load_transfer_only:
        parser.error("--stop-after-load-build cannot be combined with --load-transfer-only")
    if not 0.0 < args.lift_height_m <= LIFT_TARGET_M:
        parser.error(f"--lift-height-m must be in (0, {LIFT_TARGET_M}] m")
    if args.post_lift_hold_s < 0.0:
        parser.error("--post-lift-hold-s must be nonnegative")
    if not math.isclose(args.lift_height_m, 0.030, abs_tol=1e-12):
        parser.error("this simulation-only gate is fixed to a 30 mm lift; parameter sweeps are out of scope")
    if args.post_lift_hold_s < 1.0:
        parser.error("this simulation-only gate requires at least a 1 s hold at 30 mm")
    if args.load_transfer_only and args.lift_height_m > 0.0010001:
        parser.error("--load-transfer-only is bounded to a 1 mm arm-target lift")
    if args.load_transfer_only:
        parser.error("the simulation-only task requires the full bounded 30 mm staged pickup")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = args.output.resolve()
    evidence_root = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-dfq-simulation-only").resolve()
    if not output.is_relative_to(evidence_root) or output == evidence_root:
        parser.error(f"--output must be a run subdirectory of {evidence_root}")
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"dfq-simonly-{int(time.time())}-{os.getpid()}"
    command = (
        "wsl.exe -d Ubuntu-22.04 --exec env MUJOCO_GL=egl "
        "/tmp/robotsim-issue43-m0/venv-mj336-uv/bin/python "
        "/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-dfq-simulation-only/implementation/"
        "run_dfq_simulation_only_m0_contact_schedule.py "
        f"--output {output.as_posix()} --wrench-audit-json "
        "/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-dfq-canonical-robust-wrench-20261007/"
        "canonical_robust_wrench.json --lift-height-m 0.030000 --post-lift-hold-s 1.000 --stop-after-load-build"
    )
    wrench_audit_path = args.wrench_audit_json.resolve()
    if wrench_audit_path != WRENCH_AUDIT_JSON.resolve():
        parser.error(f"--wrench-audit-json must name the accepted canonical audit: {WRENCH_AUDIT_JSON}")
    repo_head = git_value(REPO, "rev-parse", "HEAD")
    repo_branch = git_value(REPO, "branch", "--show-current")
    repo_dirty = bool(git_value(REPO, "status", "--porcelain"))
    _, urdf_limits, mimics, _ = hand.parse_urdf()
    ranges = hand.derive_channel_ranges(urdf_limits, mimics)
    result = {
        "run_id": run_id, "status": "RUNNING", "passed": False,
        "reproduction_command": command, "RobotSim_sha": repo_head,
        "RobotSim_branch": repo_branch, "RobotSim_dirty": repo_dirty,
        "RobotSim_worktree_status": "isolated temporary clone; status checked at run start",
        "upstream": {"humanoid_vla_sha": HUMANOID_VLA_SHA, "unitree_ros_sha": UNITREE_ROS_SHA,
                     "unitree_mujoco_sha": UNITREE_MUJOCO_SHA},
        "python_version": platform.python_version(), "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
        "accepted_scene_sha256": sha256(SCENE), "approved_robot_mjcf_sha256": sha256(ROBOT_XML),
        "official_dfq_urdf_sha256": sha256(URDF), "seed": 0,
        "simulation_derived_driver_effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
        "simulation_only_controller": {
            "classification": "SIMULATION_ONLY_M0 actuator abstraction",
            "hardware_command_interface": "not established by the inspected DFQ source; no Sim2Real claim",
            "position_control_until_contact": True,
            "post_contact_mode": (
                "SIMULATION_ONLY torque impedance plus the saved-state contact-topology allocation "
                "remain active during commanded arm lift; no fixed-wrist LOAD_READY is claimed"
                if direct_arm_lift else
                "six active-driver generalized-force impedance plus LP feedforward"
            ),
            "feedforward_ramp_seconds": SIM_ONLY_FEEDFORWARD_RAMP_S,
            "kp_nm_per_rad": HAND_KP,
            "kv_nms_per_rad": HAND_KV,
            "bias_compensation": "MuJoCo qfrc_bias",
            "driver_effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
            "cap_derivation": "10.20% headroom over the 0.4818446744 Nm robust effort-balanced static allocation",
            "position_actuators_disabled_after_contact": True,
            "fixed_wrist_load_build_executed": not direct_arm_lift,
            "contact_force_preload_before_direct_lift": bool(direct_arm_lift),
            "mimic_follower_actuators": 0,
            "mimic_follower_qpos_writes": 0,
        },
        "stop_after_load_build": args.stop_after_load_build,
        "stage3_and_later_executed": False,
        "thresholds": {"source_hand_velocity_rad_s": MAX_HAND_QVEL_RAD_S,
                       "mimic_diagnostic_rad": MIMIC_DIAGNOSTIC_TOLERANCE_RAD,
                       "mimic_manipulation_limit_rad": MIMIC_MANIPULATION_LIMIT_RAD,
                       "joint_limit_violation_rad": JOINT_LIMIT_TOLERANCE_RAD,
                       "max_object_contact_force_n": MAX_CONTACT_FORCE_N,
                       "minimum_lift_m": args.lift_height_m,
                       "hold_simultaneous_contact_fraction": CONTACT_HOLD_FRACTION,
                       "hold_duration_s": HOLD_SECONDS,
                       "contact_hold_trigger_force_n": CONTACT_TRIGGER_FORCE_N,
                       "contact_hold_confirm_samples": CONTACT_CONFIRM_SAMPLES,
                       "contact_hold_confirm_duration_s": CONTACT_CONFIRM_SAMPLES * DT,
                       "contact_loss_confirm_samples": CONTACT_LOSS_CONFIRM_SAMPLES,
                       "contact_preload_target_error_cap_rad": CONTACT_PRELOAD_MAX_RAD,
                       "contact_preload_gate_tolerance_rad": CONTACT_PRELOAD_GATE_TOLERANCE_RAD,
                       "contact_preload_target_error_min_rad": 0.0,
                       "load_build_safety_factor": LOAD_BUILD_SAFETY_FACTOR,
                       "load_build_max_applied_driver_effort_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                       "load_build_stable_dwell_s": SIM_ONLY_LOAD_DWELL_S,
                       "load_build_min_hand_vertical_support_fraction": SIM_ONLY_SUPPORT_FRACTION_MIN,
                       "load_build_min_remaining_table_support_fraction": SIM_ONLY_TABLE_REMAINDER_FRACTION_MIN,
                       "load_build_force_balance_tolerance_fraction": SIM_ONLY_FORCE_BALANCE_TOLERANCE_FRACTION,
                       "load_build_horizontal_force_tolerance_fraction": SIM_ONLY_HORIZONTAL_FORCE_TOLERANCE_FRACTION,
                       "load_build_hand_torque_tolerance_nm": SIM_ONLY_WRENCH_TORQUE_TOLERANCE_NM,
                       "load_build_max_measured_friction_utilization": SIM_ONLY_MAX_FRICTION_UTILIZATION,
                       "hand_actuator_forcerange_cap_nm": HAND_MAX_TORQUE,
                       "load_build_max_single_contact_resultant_n": LOAD_BUILD_MAX_SINGLE_CONTACT_FORCE_N,
                       "load_build_max_total_contact_resultant_n": LOAD_BUILD_MAX_TOTAL_CONTACT_FORCE_N,
                       "load_build_max_total_normal_force_n": LOAD_BUILD_MAX_TOTAL_NORMAL_FORCE_N,
                       "load_build_arm_target_fixed_during_ramp": True,
                       "load_build_wrist_drift_gate_comparison_tolerance_m": LOAD_BUILD_WRIST_DRIFT_GATE_TOLERANCE_M,
                       "load_build_max_bottle_rise_m": LOAD_BUILD_MAX_BOTTLE_RISE_M,
                       "contact_validity_window_s": CONTACT_VALIDITY_WINDOW_S,
                       "contact_validity_window_samples": CONTACT_VALIDITY_WINDOW_SAMPLES,
                       "contact_validity_min_force_bearing_duty": CONTACT_VALIDITY_MIN_DUTY,
                       "contact_validity_min_impulse_ns": CONTACT_VALIDITY_MIN_IMPULSE_NS,
                       "contact_validity_max_zero_run_s": CONTACT_VALIDITY_MAX_ZERO_S,
                       "open_target_slew_rad_s": OPEN_TARGET_SLEW_RAD_S,
                       "close_target_slew_rad_s": CLOSE_TARGET_SLEW_RAD_S,
                       "pregrasp_dwell_after_settle_s": PREGRASP_SECONDS,
                       "settle_max_duration_s": SETTLE_MAX_SECONDS,
                       "settle_stable_dwell_s": SETTLE_DWELL_SECONDS,
                       "settle_max_linear_speed_m_s": SETTLE_MAX_LINEAR_SPEED_M_S,
                       "settle_max_angular_speed_rad_s": SETTLE_MAX_ANGULAR_SPEED_RAD_S},
        "grasp_pose_correction": {
            "world_wrist_yaw_deg": WRIST_YAW_CORRECTION_DEG,
            "pregrasp_site_offset_m": PREGRASP_SITE_OFFSET_M.tolist(),
            "approach_site_offset_m": APPROACH_SITE_OFFSET_M.tolist(),
            "approach_duration_s": APPROACH_SECONDS,
        },
        "contact_only_diagnostic": args.contact_only,
        "lift_target_height_m": args.lift_height_m,
        "post_lift_hold_seconds": args.post_lift_hold_s,
        "load_transfer_only": args.load_transfer_only,
        "load_capacity_audit": True,
        "stage1_wrench_audit_json": wrench_audit_path.as_posix(),
        "lift_trajectory": {
            "profile": "quintic smoothstep in joint target space",
            "duration_s": LIFT_TIMEOUT_SECONDS,
            "initial_target_velocity": "zero by construction",
            "initial_target_acceleration": "zero by construction",
            "hand_control": (
                "SIMULATION_ONLY torque impedance and the saved-state contact-topology allocation "
                "remain active; fixed-wrist LOAD_READY is not claimed"
                if direct_arm_lift else
                "simulation-only torque impedance and the same contact-topology feedforward remain active"
            ),
        },
        "lift_arming": {
            "state_machine": (
                "CONTACT_PRELOAD -> HOLD -> LIFT_ARMING -> CONTACT_FORCE_PRELOAD -> DIRECT_ARM_LIFT"
                if direct_arm_lift else
                "CONTACT_PRELOAD -> HOLD -> LIFT_ARMING -> READY_TO_LIFT -> LOAD_BUILD -> LOAD_READY -> LIFT"
            ),
            "dwell_s": LIFT_ARMING_DWELL_SECONDS,
            "required_consecutive_physics_samples": int(math.ceil(LIFT_ARMING_DWELL_SECONDS / DT)),
            "timeout_s": LIFT_ARMING_TIMEOUT_SECONDS,
            "force_threshold_n": CONTACT_TRIGGER_FORCE_N,
            "requires_force_bearing_thumb": True,
            "requires_force_bearing_opposing_digit": True,
            "requires_table_supported_bottle": True,
            "bottle_linear_speed_bound_m_s": SETTLE_MAX_LINEAR_SPEED_M_S,
            "bottle_angular_speed_bound_rad_s": SETTLE_MAX_ANGULAR_SPEED_RAD_S,
            "requires_source_velocity_mimic_and_joint_limit_gates": True,
            "contact_validity_rule": {
                "window_s": CONTACT_VALIDITY_WINDOW_S,
                "minimum_bearing_duty": CONTACT_VALIDITY_MIN_DUTY,
                "minimum_impulse_ns": CONTACT_VALIDITY_MIN_IMPULSE_NS,
                "maximum_zero_run_s": CONTACT_VALIDITY_MAX_ZERO_S,
                "threshold_force_n": CONTACT_TRIGGER_FORCE_N,
                "controller_must_remain_contact_preload": True,
            },
        },
        "output_directory": output.as_posix(), "physics_steps": 0,
    }
    (output / "m0_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "run.log").write_text(
        f"run_id={run_id}\nstatus=RUNNING\nreproduction_command={command}\n", encoding="utf-8"
    )
    try:
        if mujoco.__version__ != MUJOCO_VERSION:
            raise GateFailure(f"Expected MuJoCo {MUJOCO_VERSION}, got {mujoco.__version__}")
        load_capacity_targets = json.loads(wrench_audit_path.read_text(encoding="utf-8"))
        wrench_source_input = json.loads(WRENCH_SOURCE_INPUT_JSON.read_text(encoding="utf-8"))
        normalized_source_input = str(load_capacity_targets.get("input", "")).replace("\\", "/")
        expected_source_suffix = "/issue43-g1-dfq-canonical-robust-wrench-20261007/canonical_ready_wrench_input.json"
        if not normalized_source_input.endswith(expected_source_suffix):
            raise GateFailure("Canonical robust-wrench audit does not reference its accepted synchronized input")
        if wrench_source_input.get("run_id") != load_capacity_targets.get("run_id"):
            raise GateFailure("Canonical robust-wrench audit and synchronized source input have different run IDs")
        if wrench_source_input.get("model_sha256") != EXPECTED_WRENCH_MODEL_SHA256:
            raise GateFailure("Accepted synchronized wrench input has an unexpected compiled-model hash")
        effort_allocation = load_capacity_targets.get("effort_balanced_10pct_margin_allocation", {})
        robustness = load_capacity_targets.get("robustness_summary", {})
        if not effort_allocation.get("feasible_at_mu_1_4"):
            raise GateFailure("Canonical 10%-slack static support allocation is missing or infeasible")
        if effort_allocation.get("max_inscribed_64_cone_utilization", 1.0) > 0.900001:
            raise GateFailure("Canonical static allocation no longer meets its 10% friction-cone slack")
        robust_peak_effort = float(robustness.get("effort_balanced_worst_peak_effort_nm", math.inf))
        if not math.isfinite(robust_peak_effort) or robust_peak_effort >= SIM_ONLY_MAX_DRIVER_TORQUE_NM:
            raise GateFailure(
                "The single selected effort cap no longer exceeds the saved bounded-uncertainty allocation"
            )
        if not robustness.get("all_effort_balanced_scenarios_meet_10pct_cone_slack_and_20pct_effort_headroom"):
            raise GateFailure("Bounded-uncertainty static support evidence does not satisfy its saved criteria")
        if not math.isclose(float(load_capacity_targets.get("bottle_mass_kg", math.nan)), 0.57,
                            rel_tol=0.0, abs_tol=1e-9):
            raise GateFailure("Saved canonical wrench allocation is for a different bottle mass")
        if not math.isclose(float(load_capacity_targets.get("configured_mu", math.nan)), 1.4,
                            rel_tol=0.0, abs_tol=1e-9):
            raise GateFailure("Saved canonical wrench allocation is for a different friction coefficient")
        expected_channels = {channel for channel, _ in hand.CHANNELS}
        if set(effort_allocation.get("driver_efforts", {})) != expected_channels:
            raise GateFailure("Saved canonical effort allocation does not map all six driver channels")
        result["stage1_wrench_audit"] = {
            "path": wrench_audit_path.as_posix(),
            "sha256": sha256(wrench_audit_path),
            "source_run_id": load_capacity_targets.get("run_id"),
            "physics_steps": load_capacity_targets.get("physics_steps"),
            "bottle_mass_kg": load_capacity_targets["bottle_mass_kg"],
            "bottle_weight_n": load_capacity_targets["bottle_weight_n"],
            "configured_mu": load_capacity_targets["configured_mu"],
            "allocation": effort_allocation,
            "robustness_summary": robustness,
        }
        for path, expected in ((UPSTREAM, HUMANOID_VLA_SHA), (UNITREE_ROS, UNITREE_ROS_SHA),
                               (UNITREE_MUJOCO, UNITREE_MUJOCO_SHA)):
            if git_value(path, "rev-parse", "HEAD") != expected:
                raise GateFailure(f"Wrong pinned upstream revision: {path}")
            if git_value(path, "status", "--porcelain"):
                raise GateFailure(f"Dirty pinned upstream checkout: {path}")

        build = build_scene(output / "compiled/dfq_short_grasp_mj336.xml",
                            ranges, urdf_limits, mimics)
        compiled_model_sha256 = sha256(output / "compiled/dfq_short_grasp_mj336.xml")
        if compiled_model_sha256 != wrench_source_input["model_sha256"]:
            raise GateFailure(
                "Compiled pickup model does not match the static wrench source model: "
                f"{compiled_model_sha256} != {wrench_source_input['model_sha256']}"
            )
        result["compiled_model_sha256"] = compiled_model_sha256
        result["canonical_wrench_source_input"] = {
            "path": WRENCH_SOURCE_INPUT_JSON.as_posix(),
            "sha256": sha256(WRENCH_SOURCE_INPUT_JSON),
            "source_run_id": wrench_source_input["run_id"],
            "model_sha256": wrench_source_input["model_sha256"],
        }
        model = build["model"]
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        bottle_id = element_id(model, mujoco.mjtObj.mjOBJ_BODY, "green_box")
        site_id = element_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")
        bottle_position = data.xpos[bottle_id].copy()
        arm_q, hand_rotation, pregrasp_info = pregrasp_ik(
            model, data, site_id, bottle_position
        )
        arm_qadr = arm_qpos_addresses(model)
        data.qpos[arm_qadr] = arm_q
        mujoco.mj_forward(model, data)
        pregrasp_position = data.site_xpos[site_id].copy()
        approach_target = bottle_position + APPROACH_SITE_OFFSET_M
        approach_q, approach_ik = plan_arm(
            model, data, site_id, approach_target, hand_rotation
        )
        if not approach_ik["converged"]:
            raise GateFailure(f"Corrected approach IK failed: {approach_ik}")
        data.qpos[arm_qadr] = approach_q
        mujoco.mj_forward(model, data)
        approach_position = data.site_xpos[site_id].copy()
        lift_q, lift_info = plan_lift(
            model, data, site_id, approach_q, approach_position, hand_rotation,
            args.lift_height_m,
        )
        data.qpos[arm_qadr] = arm_q
        mujoco.mj_forward(model, data)
        initial_bottle = data.qpos[bottle_qpos_addresses(model)].copy()
        robot_contacts = [c for c in bottle_contacts(model, data) if c["side"] != "other"]
        if robot_contacts:
            raise GateFailure(f"Pregrasp overlaps robot and bottle: {robot_contacts}")
        settle_clear_q = arm_q.copy()
        data.qpos[arm_qadr] = settle_clear_q
        mujoco.mj_forward(model, data)
        settle_clear_contacts = [
            c for c in bottle_contacts(model, data) if c["side"] != "other"
        ]
        if settle_clear_contacts:
            raise GateFailure(f"Bottle-settle park pose is not contact-free: {settle_clear_contacts}")
        plan = {
            "physics_steps": 0, "pregrasp_arm_joint_names": ARM_NAMES,
            "pregrasp_arm_target_rad": arm_q.tolist(),
            "pregrasp_site_position_m": pregrasp_position.tolist(),
            "pregrasp": pregrasp_info,
            "corrected_hand_rotation_matrix": hand_rotation.tolist(),
            "wrist_yaw_correction_deg": WRIST_YAW_CORRECTION_DEG,
            "approach_site_target_m": approach_target.tolist(),
            "approach_site_position_m": approach_position.tolist(),
            "approach_arm_target_rad": approach_q.tolist(),
            "approach_ik": approach_ik,
            "finger_targets_normalized": {"OPEN": 1.0, "FULL_CLOSE": 0.0},
            "finger_targets_rad": {
                "OPEN": {ch: hand.normalized_target("R", ch, 1.0, ranges)
                         for ch, _ in hand.CHANNELS},
                "FULL_CLOSE": {ch: hand.normalized_target("R", ch, 0.0, ranges)
                               for ch, _ in hand.CHANNELS},
            },
            "lift_arm_target_rad": lift_q.tolist(),
            "lift_target_height_m": args.lift_height_m,
            "load_transfer_only": args.load_transfer_only,
            "load_capacity_audit": True,
            "stage1_wrench_audit_json": wrench_audit_path.as_posix(),
            "lift_profile": result["lift_trajectory"],
            "post_lift_hold_seconds": args.post_lift_hold_s,
            "lift_ik": lift_info, "bottle_initial_qpos": initial_bottle.tolist(),
            "settle_clear_arm_target_rad": settle_clear_q.tolist(),
            "pregrasp_dwell_after_settle_s": PREGRASP_SECONDS,
            "settle_clear_right_hand_site_position_m": data.site_xpos[site_id].tolist(),
            "settle_clear_robot_bottle_contacts": settle_clear_contacts,
            "settle_criterion": {
                "table_contact_continuous": True,
                "linear_speed_max_m_s": SETTLE_MAX_LINEAR_SPEED_M_S,
                "angular_speed_max_rad_s": SETTLE_MAX_ANGULAR_SPEED_RAD_S,
                "stable_dwell_s": SETTLE_DWELL_SECONDS,
                "timeout_s": SETTLE_MAX_SECONDS,
            },
            "initial_robot_bottle_contacts": robot_contacts,
            "pre_rollout_robot_qpos_write_joint_names": ARM_NAMES,
            "bottle_qpos_write_count": 0, "reproduction_command": command,
        }
        (output / "preflight_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if args.plan_only:
            result.update({"status": "PLAN_ONLY", "passed": False, "plan": plan})
            (output / "m0_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (output / "run.log").write_text(
                f"run_id={run_id}\nstatus=PLAN_ONLY\nphysics_steps=0\nreproduction_command={command}\n",
                encoding="utf-8",
            )
            return 0

        model_integrity = {
            "actuator_count": model.nu,
            "actuator_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                               for i in range(model.nu)],
            "right_hand_actuators": build["right_hand_actuators"], "left_hand_actuators": [],
            "mimic_constraint_count": build["mimic_constraint_count"],
            "mimic_follower_actuators": 0, "fixed_base_weld_count": build["fixed_base_weld_count"],
            "bottle_equality_reference_count": build["bottle_equality_reference_count"],
            "mocap_body_count": build["mocap_body_count"], "bottle_free_joint": "bottle_free",
            "bottle_geometry": ["bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"],
            "active_rollout_qpos_write_scan": ast_rollout_write_check(),
        }
        (output / "model_integrity.json").write_text(json.dumps(model_integrity, indent=2), encoding="utf-8")
        actuator_mapping = {
            "classification": "SIMULATION_ONLY_M0; not validated against the real Inspire DFQ command interface",
            "source": {
                "official_urdf": URDF.as_posix(),
                "unitree_ros_revision": UNITREE_ROS_SHA,
                "urdf_sha256": sha256(URDF),
                "driver_joints": {
                    channel: {
                        "joint": hand.channel_joint("R", suffix),
                        "limits": urdf_limits[hand.channel_joint("R", suffix)],
                        "command_interpretation": "URDF joint-space coordinate used only by this simulation model",
                    }
                    for channel, suffix in hand.CHANNELS
                },
                "right_mimic_followers": mimics,
                "hardware_interface_status": "not established by the inspected DFQ sources; Sim2Real blocked",
            },
            "derived_for_simulation": {
                "free_space_actuators": {
                    "type": "MuJoCo position actuator",
                    "kp_nm_per_rad": HAND_KP,
                    "kv_nms_per_rad": HAND_KV,
                    "force_cap_nm": HAND_MAX_TORQUE,
                },
                "load_hold_lift": {
                    "type": "qfrc_applied generalized-force impedance with static feedforward",
                    "active_driver_channels": list(build["right_hand_actuators"]),
                    "follower_actuators": 0,
                    "follower_qpos_writes": 0,
                    "feedforward_torques_nm": {
                        channel: float(effort_allocation["driver_efforts"][channel]["estimated_required_effort_nm"])
                        for channel in build["right_hand_actuators"]
                    },
                    "feedforward_ramp_s": SIM_ONLY_FEEDFORWARD_RAMP_S,
                    "impedance_kp_nm_per_rad": HAND_KP,
                    "impedance_kv_nms_per_rad": HAND_KV,
                    "gravity_bias_compensation": "MuJoCo qfrc_bias on active driver DoFs",
                    "total_applied_driver_effort_cap_nm": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                    "cap_derivation": {
                        "worst_bounded_uncertainty_allocation_nm": robust_peak_effort,
                        "headroom_fraction": SIM_ONLY_MAX_DRIVER_TORQUE_NM / robust_peak_effort - 1.0,
                        "source_effort_limit_nm": 1.0,
                        "fraction_of_source_limit": SIM_ONLY_MAX_DRIVER_TORQUE_NM,
                    },
                },
                "input_wrench_file_sha256": sha256(wrench_audit_path),
            },
        }
        (output / "actuator_control_mapping.json").write_text(
            json.dumps(actuator_mapping, indent=2), encoding="utf-8"
        )
        (output / "source_vs_derived_parameters.json").write_text(
            json.dumps({
                "source_parameters": actuator_mapping["source"],
                "simulation_derived_parameters": actuator_mapping["derived_for_simulation"],
                "classification": actuator_mapping["classification"],
            }, indent=2), encoding="utf-8"
        )
        result["plan"], result["model_integrity"] = plan, model_integrity
        outcome = run_rollout(model, data, build, output, result, settle_clear_q,
                              arm_q, approach_q, lift_q, ranges, urdf_limits, mimics,
                              args.contact_only, args.lift_height_m,
                              args.post_lift_hold_s, args.load_transfer_only,
                              load_capacity_targets, args.stop_after_load_build)
        result = outcome["result"]
        result["plan"], result["model_integrity"] = plan, model_integrity
        (output / "m0_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "run.log").write_text(
            f"run_id={run_id}\nstatus={result['status']}\nRobotSim_sha={repo_head}\n"
            f"RobotSim_branch={repo_branch}\nRobotSim_dirty={repo_dirty}\n"
            "RobotSim_worktree_status=isolated temporary clone\n"
            f"MuJoCo={mujoco.__version__}\n"
            f"simulation_only_effort_cap_nm={SIM_ONLY_MAX_DRIVER_TORQUE_NM}\n"
            f"maximum_simulation_only_driver_effort_nm={result.get('maximum_simulation_only_driver_effort_nm')}\n"
            f"maximum_simulation_only_requested_effort_nm={result.get('maximum_simulation_only_requested_effort_nm')}\n"
            f"simulation_only_effort_cap_saturation_samples={result.get('simulation_only_effort_cap_saturation_samples')}\n"
            f"physics_time_s={data.time:.9f}\nphysics_steps={result['physics_steps']}\n"
            f"max_hand_qvel_rad_s={result.get('maximum_hand_joint_qvel_rad_s')}\n"
            f"max_mimic_error_rad={result.get('maximum_mimic_error_rad')}\n"
            f"mimic_diagnostic_threshold_rad={MIMIC_DIAGNOSTIC_TOLERANCE_RAD}\n"
            f"mimic_manipulation_limit_rad={MIMIC_MANIPULATION_LIMIT_RAD}\n"
            f"mimic_diagnostic_threshold_exceed_steps={result.get('mimic_diagnostic_threshold_exceed_steps')}\n"
            f"close_target_slew_rad_s={CLOSE_TARGET_SLEW_RAD_S}\n"
            f"open_target_slew_rad_s={OPEN_TARGET_SLEW_RAD_S}\n"
            f"contact_hold_trigger_force_n={CONTACT_TRIGGER_FORCE_N}\n"
            f"contact_hold_confirm_samples={CONTACT_CONFIRM_SAMPLES}\n"
            f"contact_hold_transitions={json.dumps(result.get('contact_aware_close', {}).get('channel_transitions', {}), separators=(',', ':'))}\n"
            f"contact_preload_cap_rad={CONTACT_PRELOAD_MAX_RAD}\n"
            f"contact_preload_gate_tolerance_rad={CONTACT_PRELOAD_GATE_TOLERANCE_RAD}\n"
            "contact_preload_controller=post-step sign-aware target projection; gate tolerance not used by controller\n"
            f"contact_preload_offsets={json.dumps(result.get('contact_aware_close', {}).get('channel_preload_offsets_rad', {}), separators=(',', ':'))}\n"
            f"contact_controller_events={json.dumps(result.get('contact_aware_close', {}).get('transition_events', []), separators=(',', ':'))}\n"
            f"max_contact_preload_target_error_rad={result.get('maximum_contact_preload_target_error_rad')}\n"
            f"min_contact_preload_target_error_rad={result.get('minimum_contact_preload_target_error_rad')}\n"
            f"max_contact_commanded_error_rad={result.get('maximum_contact_commanded_error_rad')}\n"
            f"max_hand_actuator_effort_nm={result.get('maximum_hand_actuator_effort_nm')}\n"
            f"bottle_settle={json.dumps(result.get('bottle_settle', {}), separators=(',', ':'))}\n"
            f"lift_height_m={result.get('lift_height_m')}\n"
            f"lift_target_height_m={result.get('lift_target_height_m')}\n"
            f"post_lift_hold_seconds={result.get('post_lift_hold_seconds')}\n"
            f"lift_trajectory={json.dumps(result.get('lift_trajectory', {}), separators=(',', ':'))}\n"
            f"right_thumb_contact={result.get('final_right_thumb_contact')}\n"
            f"right_opposing_contact={result.get('final_right_opposing_digit_contact')}\n"
            f"left_hand_bottle_contact={result.get('left_hand_bottle_contact')}\n"
            f"reproduction_command={command}\n", encoding="utf-8",
        )
        write_report(output, result, command)
        return 0 if result["passed"] else 1
    except Exception as exc:
        result.update({"status": "FAIL", "passed": False, "error": str(exc),
                       "traceback": traceback.format_exc()})
        (output / "m0_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output / "run.log").write_text(
            f"run_id={run_id}\nstatus=FAIL\nerror={exc}\nreproduction_command={command}\n",
            encoding="utf-8",
        )
        try:
            write_report(output, result, command)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
