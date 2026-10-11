#!/usr/bin/env python3
"""Bounded Issue #46 source-faithful OmniPicker assembly and cylinder audit.

Generated evidence is written outside the repository. The paired-jaw servo used
for the empty-space motion gate is explicitly SIMULATION_ONLY.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin
from PIL import Image
from scipy.spatial import cKDTree


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_REPO = "https://github.com/AgibotTech/agibot_x2_urdf"
X2_URDF_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
X2_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
CANONICAL_HELPER_SHA256 = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"

TIMESTEP = 0.002
GRAVITY = 9.81
BASE_WORLD = np.array([0.38, 0.32, 0.68], dtype=float)
BASE_YAW = -math.pi / 2.0
ENDPOINT_MARGIN = 0.001
MOTION_DURATION = 2.5
HOLD_DURATION = 0.5
FINGER_HOLD_DURATION = 0.5
FREQUENCY = 8.0
DAMPING_RATIO = 1.0
CONTACT_FRICTION_ASSUMPTION = 1.4
NORMAL_DESIGN_FACTOR = 2.0
MIMIC_TRACKING_DIAGNOSTIC_RAD = 0.02
CYLINDER_RADIUS = 0.035
CYLINDER_HALF_HEIGHT = 0.12225
CYLINDER_MASS = 0.570
CYLINDER_FRICTION = [1.4, 0.02, 0.001]

DRIVER_FOLLOWER = {
    "left": ("left_claw_joint", "L_hand_wide1_joint", "L_hand_narrow3_Link", "L_hand_wide3_Link"),
    "right": ("right_claw_joint", "R_hand_wide1_joint", "R_hand_narrow3_Link", "R_hand_wide3_Link"),
}

# Exact exclusions already used by the accepted Issue #46 paired-jaw research
# runner. They are surfaced in every diagnostic result and are not source data.
HISTORICAL_EXCLUSIONS = (
    ("pelvis", "left_hip_pitch_link"),
    ("pelvis", "left_hip_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("pelvis", "right_hip_roll_link"),
    ("pelvis", "waist_yaw_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("waist_yaw_link", "torso_link"),
    ("torso_link", "head_pitch_link"),
    ("left_wrist_yaw_link", "left_wrist_roll_link"),
    ("right_wrist_yaw_link", "right_wrist_roll_link"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def load_canonical_module(path: Path):
    spec = importlib.util.spec_from_file_location("robotsim_canonical_assets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load canonical scene helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise RuntimeError(f"Missing compiled joint: {name}")
    return int(result)


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"Missing compiled body: {name}")
    return int(result)


def geom_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if result < 0:
        raise RuntimeError(f"Missing compiled geom: {name}")
    return int(result)


def qpos_addr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, name)])


def dof_addr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, name)])


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if result < 0:
        raise RuntimeError(f"Missing compiled actuator: {name}")
    return int(result)


def collision_geoms(model: mujoco.MjModel) -> list[int]:
    return [i for i in range(model.ngeom)
            if int(model.geom_contype[i]) != 0 or int(model.geom_conaffinity[i]) != 0]


def collision_geom_for_body(model: mujoco.MjModel, name: str) -> int:
    bid = body_id(model, name)
    matches = [gid for gid in collision_geoms(model) if int(model.geom_bodyid[gid]) == bid]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one collision geom on {name}, found {matches}")
    return matches[0]


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or "world"


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(model.njnt)]


def quaternion_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = a @ b.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def rpy_matrix(rpy: list[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def mesh_vertices_world(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray:
    mid = int(model.geom_dataid[gid])
    start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return verts @ rotation.T + data.geom_xpos[gid]


def source_file_for_mesh(root: Path, urdf_dir: Path, filename: str) -> Path | None:
    if filename.startswith("package://"):
        candidate = root / filename[len("package://"):]
        if candidate.is_file():
            return candidate
    candidate = (urdf_dir / filename).resolve()
    if candidate.is_file():
        return candidate
    basename = Path(filename).name
    hits = list(root.rglob(basename))
    return hits[0] if len(hits) == 1 else None


def source_records(urdf: Path, model: mujoco.MjModel, root: Path) -> dict[str, Any]:
    robot = ET.parse(urdf).getroot()
    links = {element.get("name", ""): element for element in robot.findall("link")}
    joints_xml = {element.get("name", ""): element for element in robot.findall("joint")}
    parent_by_child: dict[str, str] = {}
    hierarchy = []
    joint_rows = []
    for name, joint in joints_xml.items():
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        parent_by_child[child] = parent
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
        axis_element = joint.find("axis")
        axis = [float(v) for v in (axis_element.get("xyz", "0 0 1").split() if axis_element is not None else [0, 0, 1])]
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        compiled_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        compiled = None
        if compiled_jid >= 0:
            jid = int(compiled_jid)
            compiled = {
                "type": int(model.jnt_type[jid]),
                "axis": model.jnt_axis[jid].tolist(),
                "range_rad": model.jnt_range[jid].tolist() if model.jnt_limited[jid] else None,
                "limited": bool(model.jnt_limited[jid]),
                "ref": float(model.qpos0[int(model.jnt_qposadr[jid])]),
                "qpos_address": int(model.jnt_qposadr[jid]),
                "dof_address": int(model.jnt_dofadr[jid]),
            }
        source_range = None if limit is None or limit.get("lower") is None else [
            float(limit.get("lower")), float(limit.get("upper"))]
        mimic_row = None if mimic is None else {
            "joint": mimic.get("joint"),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        }
        joint_rows.append({
            "name": name, "type": joint.get("type"), "parent": parent, "child": child,
            "origin_xyz_m": xyz, "origin_rpy_rad": rpy, "source_axis": axis,
            "source_position_range_rad": source_range,
            "source_velocity_limit_rad_s": None if limit is None else float(limit.get("velocity", "0")),
            "source_effort_metadata": None if limit is None else float(limit.get("effort", "0")),
            "mimic": mimic_row, "compiled": compiled,
        })
        hierarchy.append({"joint": name, "type": joint.get("type"), "parent": parent,
                          "child": child, "origin_xyz_m": xyz, "origin_rpy_rad": rpy})

    links_rows: list[dict[str, Any]] = []
    missing_links = []
    source_mesh_hashes: dict[str, str] = {}
    for name, link in links.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            missing_links.append(name)
        geom_rows = []
        compiled_geoms = []
        if bid >= 0:
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) != int(bid):
                    continue
                mid = int(model.geom_dataid[gid])
                compiled_geoms.append({
                    "geom_id": gid, "geom_name": geom_name(model, gid),
                    "group": int(model.geom_group[gid]), "type": int(model.geom_type[gid]),
                    "collision_enabled": bool(model.geom_contype[gid] or model.geom_conaffinity[gid]),
                    "contype": int(model.geom_contype[gid]), "conaffinity": int(model.geom_conaffinity[gid]),
                    "geom_pos_m": model.geom_pos[gid].tolist(), "geom_quat_wxyz": model.geom_quat[gid].tolist(),
                    "mesh_name": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) if mid >= 0 else None),
                    "mesh_vertices": (int(model.mesh_vertnum[mid]) if mid >= 0 else None),
                    "mesh_faces": (int(model.mesh_facenum[mid]) if mid >= 0 else None),
                    "mesh_scale": (model.mesh_scale[mid].tolist() if mid >= 0 else None),
                })
        for kind in ("visual", "collision"):
            for geom in link.findall(kind):
                origin = geom.find("origin")
                xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])]
                rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
                mesh = geom.find("geometry/mesh")
                mesh_ref = None if mesh is None else mesh.get("filename")
                mesh_scale = None if mesh is None else [float(v) for v in mesh.get("scale", "1 1 1").split()]
                resolved = source_file_for_mesh(root, urdf.parent, mesh_ref) if mesh_ref else None
                if resolved:
                    source_mesh_hashes[str(resolved.relative_to(root))] = sha256(resolved)
                geom_rows.append({
                    "kind": kind, "name": geom.get("name"), "origin_xyz_m": xyz,
                    "origin_rpy_rad": rpy, "mesh_ref": mesh_ref, "mesh_scale": mesh_scale,
                    "resolved_mesh": None if resolved is None else str(resolved),
                    "mesh_sha256": None if resolved is None else sha256(resolved),
                })
        links_rows.append({"name": name, "parent": parent_by_child.get(name),
                           "compiled_body_id": None if bid < 0 else int(bid),
                           "source_geometries": geom_rows, "compiled_geometries": compiled_geoms})

    source_movable_joints = {name for name, j in joints_xml.items() if j.get("type") != "fixed"}
    compiled_names = set(joint_names(model))
    compiled_bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(1, model.nbody)}
    mimic_checks = []
    for side, (driver, follower, _, _) in DRIVER_FOLLOWER.items():
        source_mimic = joints_xml[follower].find("mimic")
        source_limit_d = joints_xml[driver].find("limit")
        source_limit_f = joints_xml[follower].find("limit")
        multiplier = float(source_mimic.get("multiplier", "1")) if source_mimic is not None else None
        offset = float(source_mimic.get("offset", "0")) if source_mimic is not None else None
        expected_endpoints = None
        if source_limit_d is not None and source_limit_f is not None and multiplier is not None:
            dlow, dhigh = float(source_limit_d.get("lower")), float(source_limit_d.get("upper"))
            flow, fhigh = float(source_limit_f.get("lower")), float(source_limit_f.get("upper"))
            expected_endpoints = sorted([multiplier * dlow + offset, multiplier * dhigh + offset])
            follower_range = sorted([flow, fhigh])
        else:
            follower_range = None
        mimic_checks.append({"side": side, "driver": driver, "follower": follower,
                             "source_relation": {"multiplier": multiplier, "offset": offset,
                                                 "source_joint": None if source_mimic is None else source_mimic.get("joint")},
                             "driver_source_range_rad": None if source_limit_d is None else [float(source_limit_d.get("lower")), float(source_limit_d.get("upper"))],
                             "follower_source_range_rad": None if source_limit_f is None else [float(source_limit_f.get("lower")), float(source_limit_f.get("upper"))],
                             "driver_range_mapped_to_follower_rad": expected_endpoints,
                             "range_mapping_matches": expected_endpoints == follower_range,
                             "compiled_driver_present": driver in compiled_names,
                             "compiled_follower_present": follower in compiled_names})
    extra_bodies = sorted(name for name in compiled_bodies if name not in links)
    return {"source_link_count": len(links), "compiled_named_link_count": len(links) - len(missing_links),
            "missing_compiled_links": missing_links, "source_joint_count": len(joints_xml),
            "source_movable_joint_count": len(source_movable_joints), "compiled_joint_count": model.njnt,
            "missing_compiled_movable_joints": sorted(source_movable_joints - compiled_names),
            "extra_compiled_joints": sorted(compiled_names - source_movable_joints),
            "extra_compiled_bodies": extra_bodies, "mimic_checks": mimic_checks,
            "joint_hierarchy": hierarchy,
            "joints": joint_rows, "links": links_rows, "source_mesh_hashes": source_mesh_hashes}


def qpos_from_config(model: mujoco.MjModel, q: dict[str, float]) -> np.ndarray:
    result = model.qpos0.copy()
    for name, value in q.items():
        result[qpos_addr(model, name)] = value
    return result


def pose_configurations(source_joints: dict[str, ET.Element]) -> list[dict[str, Any]]:
    limits = {}
    for name, joint in source_joints.items():
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and joint.get("type") != "fixed":
            limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    pair_names = {j for pair in DRIVER_FOLLOWER.values() for j in pair[:2]}
    configurations = []
    for label, fraction in (("zero", None), ("range_35_percent", 0.35), ("range_65_percent", 0.65)):
        values: dict[str, float] = {}
        for name, (low, high) in limits.items():
            if name in pair_names:
                continue
            if fraction is None:
                values[name] = min(max(0.0, low), high)
            else:
                values[name] = low + fraction * (high - low)
        aperture = {"zero": 0.5, "range_35_percent": 1.0, "range_65_percent": 0.0}[label]
        for side, (driver, follower, _, _) in DRIVER_FOLLOWER.items():
            driver_q = -(ENDPOINT_MARGIN + (1.0 - 2.0 * ENDPOINT_MARGIN) * aperture)
            values[driver] = driver_q
            values[follower] = -driver_q
        configurations.append({"name": label, "qpos": values,
                               "source_limit_valid_by_construction": True,
                               "mimic_consistent_by_construction": True})
    return configurations


def run_fk_audit(model: mujoco.MjModel, urdf: Path, source_joints: dict[str, ET.Element]) -> dict[str, Any]:
    pin_model = pin.buildModelFromUrdf(str(urdf))
    pin_data = pin_model.createData()
    frames = {frame.name: index for index, frame in enumerate(pin_model.frames)}
    results = []
    for config in pose_configurations(source_joints):
        qmap = config["qpos"]
        pin_q = pin.neutral(pin_model)
        mujoco_data = mujoco.MjData(model)
        mujoco_data.qpos[:] = qpos_from_config(model, qmap)
        mujoco.mj_forward(model, mujoco_data)
        for name, value in qmap.items():
            pid = pin_model.getJointId(name)
            if pid >= len(pin_model.names):
                raise RuntimeError(f"Pinocchio source joint is missing: {name}")
            pin_q[int(pin_model.idx_qs[pid])] = value
        pin.forwardKinematics(pin_model, pin_data, pin_q)
        pin.updateFramePlacements(pin_model, pin_data)
        errors = []
        missing = []
        for bid in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or name not in frames:
                missing.append(name)
                continue
            placement = pin_data.oMf[frames[name]]
            position_error = float(np.linalg.norm(mujoco_data.xpos[bid] - placement.translation))
            rotation_error = quaternion_distance(mujoco_data.xmat[bid].reshape(3, 3), placement.rotation)
            errors.append({"body": name, "translation_error_m": position_error,
                           "rotation_error_rad": rotation_error})
        results.append({"configuration": config["name"], "compared_body_frames": len(errors),
                        "missing_pinocchio_frames": missing,
                        "maximum_translation_error_m": max((r["translation_error_m"] for r in errors), default=0.0),
                        "maximum_rotation_error_rad": max((r["rotation_error_rad"] for r in errors), default=0.0),
                        "frame_errors": errors})
    joint_checks = []
    for name, joint in source_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or joint.get("type") == "fixed":
            continue
        axis = [float(v) for v in joint.find("axis").get("xyz", "0 0 1").split()]
        limit = joint.find("limit")
        source_range = [float(limit.get("lower")), float(limit.get("upper"))]
        compiled_range = model.jnt_range[jid].tolist() if model.jnt_limited[jid] else None
        axis_error = float(np.linalg.norm(model.jnt_axis[jid] - np.asarray(axis)))
        range_error = None if compiled_range is None else max(abs(compiled_range[i] - source_range[i]) for i in range(2))
        joint_checks.append({"joint": name, "source_axis": axis, "compiled_axis": model.jnt_axis[jid].tolist(),
                             "axis_error": axis_error, "source_range_rad": source_range,
                             "compiled_range_rad": compiled_range, "range_error_rad": range_error,
                             "compiled_ref_rad": float(model.qpos0[int(model.jnt_qposadr[jid])])})
    return {"pinocchio_version": getattr(pin, "__version__", "unknown"),
            "pinocchio_nq": int(pin_model.nq), "mujoco_nq": int(model.nq),
            "configurations": results, "joint_axis_limit_checks": joint_checks,
            "translation_tolerance_m": 1e-8, "rotation_tolerance_rad": 1e-7,
            "status": "PASS" if all(not r["missing_pinocchio_frames"]
                                    and r["maximum_translation_error_m"] <= 1e-8
                                    and r["maximum_rotation_error_rad"] <= 1e-7 for r in results)
            and all(r["axis_error"] <= 1e-12 and (r["range_error_rad"] is None or r["range_error_rad"] <= 1e-12)
                    for r in joint_checks) else "FAIL"}


def collision_mask(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool((int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first])))


def active_contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        rows.append({"geom1": geom_name(model, g1), "geom2": geom_name(model, g2),
                     "body1": body_name(model, b1), "body2": body_name(model, b2),
                     "signed_distance_m": float(contact.dist), "penetration_m": max(0.0, -float(contact.dist)),
                     "position_world_m": contact.pos.tolist(), "normal_world": contact.frame[:3].tolist(),
                     "contact_force_local": force.tolist(), "normal_force_n": float(force[0])})
    return rows


def source_collision_sweep(model: mujoco.MjModel, source_joints: dict[str, ET.Element]) -> dict[str, Any]:
    data = mujoco.MjData(model)
    collision_ids = collision_geoms(model)
    parent_pairs = set()
    for joint in source_joints.values():
        if joint.find("parent") is not None:
            pair = tuple(sorted((joint.find("parent").get("link"), joint.find("child").get("link"))))
            parent_pairs.add(pair)
    samples = []
    pair_min: dict[str, dict[str, Any]] = {}
    for index, aperture in enumerate(np.linspace(1.0, 0.0, 41)):
        qmap = {}
        for side, (driver, follower, _, _) in DRIVER_FOLLOWER.items():
            driver_q = -(ENDPOINT_MARGIN + (1.0 - 2.0 * ENDPOINT_MARGIN) * float(aperture))
            qmap[driver] = driver_q
            qmap[follower] = -driver_q
        data.qpos[:] = qpos_from_config(model, qmap)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        contacts = active_contact_rows(model, data)
        samples.append({"sample": index, "aperture_ratio": float(aperture),
                        "active_contact_count": len(contacts), "contacts": contacts})
        for row in contacts:
            key = "|".join(sorted((row["body1"], row["body2"])))
            adjacent = tuple(sorted((row["body1"], row["body2"]))) in parent_pairs
            candidate = {**row, "aperture_ratio": float(aperture), "source_parent_child_pair": adjacent}
            if key not in pair_min or candidate["signed_distance_m"] < pair_min[key]["signed_distance_m"]:
                pair_min[key] = candidate
    historical = {tuple(sorted(pair)) for pair in HISTORICAL_EXCLUSIONS}
    for row in pair_min.values():
        pair = tuple(sorted((row["body1"], row["body2"])))
        row["historical_research_exception_pair"] = pair in historical
        row["collision_class"] = ("HISTORICAL_SIMULATION_ONLY_EXCEPTION" if pair in historical else
                                  "SOURCE_ADJACENT_LINK_CONTACT" if row["source_parent_child_pair"] else
                                  "UNEXPLAINED_NONADJACENT_CONTACT")
    unexpected = [row for row in pair_min.values() if row["collision_class"] == "UNEXPLAINED_NONADJACENT_CONTACT"]
    return {"physics_steps": 0, "samples": 41, "sides_swept_together": ["left", "right"],
            "exclusions_added": [], "pair_minimums": list(pair_min.values()), "samples_detail": samples,
            "non_parent_child_overlap_count": len(unexpected),
            "maximum_observed_penetration_m": max((row["penetration_m"] for row in pair_min.values()), default=0.0),
            "status": "PASS" if not unexpected else "FAIL",
            "unexplained_nonadjacent_pairs": unexpected,
            "interpretation": "This sweep uses the source MuJoCo collision masks with no exclusions. Source-adjacent contacts and the exact previously documented Issue #46 exception pairs are classified separately; any other non-adjacent contact fails this audit."}


def check_mesh_pair(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> dict[str, Any]:
    bid = body_id(model, body)
    rows = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid or int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[gid])
        verts = mesh_vertices_world(model, data, gid)
        rows.append({"geom_id": gid, "geom_name": geom_name(model, gid),
                     "mesh_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid),
                     "role": "collision" if collision_mask(model, gid, gid) else "visual_or_noncolliding",
                     "group": int(model.geom_group[gid]), "contype": int(model.geom_contype[gid]),
                     "conaffinity": int(model.geom_conaffinity[gid]),
                     "vertices": int(model.mesh_vertnum[mid]), "faces": int(model.mesh_facenum[mid]),
                     "bbox_min_world_m": verts.min(axis=0).tolist(), "bbox_max_world_m": verts.max(axis=0).tolist(),
                     "vertices_world": verts})
    visual = next((r for r in rows if r["mesh_name"] == "right_wrist_roll_extend_link"), None)
    collision = next((r for r in rows if r["mesh_name"] == "right_wrist_roll_link"), None)
    if visual is None or collision is None:
        return {"status": "FAIL", "rows": [{k: v for k, v in r.items() if k != "vertices_world"} for r in rows],
                "reason": "Expected distinct official visual and collision mesh names were not both found."}
    visual_points = visual["vertices_world"]
    collision_points = collision["vertices_world"]
    v_to_c = cKDTree(collision_points).query(visual_points, k=1)[0]
    c_to_v = cKDTree(visual_points).query(collision_points, k=1)[0]
    report = {"visual": {k: v for k, v in visual.items() if k != "vertices_world"},
              "collision": {k: v for k, v in collision.items() if k != "vertices_world"},
              "vertex_cloud_symmetric_nearest_distance_m": {
                  "visual_to_collision_max": float(v_to_c.max()), "collision_to_visual_max": float(c_to_v.max()),
                  "visual_to_collision_p95": float(np.quantile(v_to_c, 0.95)),
                  "collision_to_visual_p95": float(np.quantile(c_to_v, 0.95)),
                  "metric_note": "symmetric nearest-vertex cloud distance; not a surface Hausdorff distance"},
              "source_usage": "URDF visual references right_wrist_roll_extend_link.stl; URDF collision references right_wrist_roll_link.stl; both are retained unchanged."}
    return report


def model_identity(model: mujoco.MjModel, script: Path, urdf: Path, vendor_root: Path) -> dict[str, Any]:
    package_dir = Path(mujoco.__file__).resolve().parent
    lib_candidates = sorted(package_dir.glob("libmujoco.so*"))
    return {
        "python": sys.version, "platform": platform.platform(), "mujoco_python": mujoco.__version__,
        "mujoco_native": mujoco.mj_versionString(), "numpy": np.__version__, "pinocchio": pin.__version__,
        "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
        "gravity_m_s2": model.opt.gravity.tolist(), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "native_libraries": [{"path": str(p), "sha256": sha256(p)} for p in lib_candidates],
        "source_checkout": {"path": str(vendor_root), "commit": subprocess.check_output(
            ["git", "-C", str(vendor_root), "rev-parse", "HEAD"], text=True).strip(),
            "status_porcelain": subprocess.check_output(["git", "-C", str(vendor_root), "status", "--porcelain"], text=True).strip()},
        "urdf": {"path": str(urdf), "sha256": sha256(urdf)},
        "runner": {"path": str(script), "sha256": sha256(script)},
    }


def source_model(urdf: Path) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.fusestatic = False
    model = spec.compile()
    return spec, model


def pair_contact_at(model: mujoco.MjModel, data: mujoco.MjData, geom_a: int, geom_b: int) -> dict[str, Any]:
    segment = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, 1.0, segment))
    return {"distance_m": distance, "witness_a_world_m": segment[:3].tolist(),
            "witness_b_world_m": segment[3:].tolist()}


def set_aperture_qpos(model: mujoco.MjModel, data: mujoco.MjData, side: str, aperture: float) -> None:
    driver, follower, _, _ = DRIVER_FOLLOWER[side]
    driver_q = -(ENDPOINT_MARGIN + (1.0 - 2.0 * ENDPOINT_MARGIN) * aperture)
    data.qpos[qpos_addr(model, driver)] = driver_q
    data.qpos[qpos_addr(model, follower)] = -driver_q


def aperture_reference(aperture: float, velocity: float, acceleration: float) -> dict[str, float]:
    span = 1.0 - 2.0 * ENDPOINT_MARGIN
    driver = -(ENDPOINT_MARGIN + span * aperture)
    driver_v = -span * velocity
    driver_a = -span * acceleration
    return {"driver_q": driver, "follower_q": -driver,
            "driver_v": driver_v, "follower_v": -driver_v,
            "driver_a": driver_a, "follower_a": -driver_a}


def minimum_jerk(t: float, duration: float, start: float, end: float) -> tuple[float, float, float]:
    if t <= 0.0:
        return start, 0.0, 0.0
    if t >= duration:
        return end, 0.0, 0.0
    s = t / duration
    p = 10 * s**3 - 15 * s**4 + 6 * s**5
    dp = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    ddp = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
    return start + (end - start) * p, (end - start) * dp, (end - start) * ddp


def full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    result = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, result, data.qM)
    return result


def derive_controller(model: mujoco.MjModel, bottle_mass: float) -> dict[str, Any]:
    data = mujoco.MjData(model)
    result: dict[str, Any] = {"label": "SIMULATION_ONLY paired-jaw actuator emulation",
                              "not_hardware_calibrated": True, "endpoint_margin_rad": ENDPOINT_MARGIN,
                              "motion_duration_s": MOTION_DURATION, "source_mapping": "follower=-driver",
                              "frequency_rad_s": FREQUENCY, "damping_ratio": DAMPING_RATIO,
                              "friction_assumption": CONTACT_FRICTION_ASSUMPTION,
                              "actuator_effort_basis": "inertia-scaled gains plus bounded force estimate, reused from the accepted Issue #46 research controller",
                              "per_side": {}}
    design_normal = NORMAL_DESIGN_FACTOR * bottle_mass * GRAVITY / (2.0 * CONTACT_FRICTION_ASSUMPTION)
    peak_accel = (1.0 - 2.0 * ENDPOINT_MARGIN) * 5.773502691896258 / MOTION_DURATION**2
    for side, (driver, follower, narrow, wide) in DRIVER_FOLLOWER.items():
        sampled: dict[str, dict[str, float]] = {driver: {"inertia": 0.0, "bias": 0.0, "lever": 0.0},
                                                follower: {"inertia": 0.0, "bias": 0.0, "lever": 0.0}}
        for aperture in np.linspace(0.0, 1.0, 9):
            data.qpos[:] = model.qpos0
            for other in DRIVER_FOLLOWER:
                set_aperture_qpos(model, data, other, 1.0)
            set_aperture_qpos(model, data, side, float(aperture))
            mujoco.mj_forward(model, data)
            M = full_mass_matrix(model, data)
            g1, g2 = body_id(model, narrow), body_id(model, wide)
            normal = data.xpos[g1] - data.xpos[g2]
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            for name, body in ((driver, g1), (follower, g2)):
                dof = dof_addr(model, name)
                sampled[name]["inertia"] = max(sampled[name]["inertia"], float(M[dof, dof]))
                sampled[name]["bias"] = max(sampled[name]["bias"], abs(float(data.qfrc_bias[dof])))
                jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, body)
                sampled[name]["lever"] = max(sampled[name]["lever"], abs(float(np.dot(jacp[:, dof], normal))))
        joints = {}
        for name, row in sampled.items():
            kp = row["inertia"] * FREQUENCY**2
            kd = 2.0 * DAMPING_RATIO * FREQUENCY * row["inertia"]
            free_demand = row["bias"] + row["inertia"] * peak_accel
            contact_demand = design_normal * row["lever"] + row["bias"] + row["inertia"] * peak_accel
            cap = max(2.0 * free_demand, 1.15 * contact_demand)
            joints[name] = {"kp_nm_rad": kp, "kd_nm_s_rad": kd,
                            "effort_cap_nm": cap, "compiled_inertia_kg_m2": row["inertia"],
                            "sampled_bias_nm": row["bias"], "lever_arm_m": row["lever"],
                            "cap_derivation": "max(2*(bias+inertia*reference_acceleration), 1.15*(design_normal*lever+bias+inertia*reference_acceleration))"}
        result["per_side"][side] = {"driver": driver, "follower": follower,
                                    "design_normal_force_per_jaw_n": design_normal,
                                    "peak_reference_acceleration_rad_s2": peak_accel, "joints": joints}
    return result


def build_dynamic_model(urdf: Path, active_sides: set[str], base_pose: bool = True) -> tuple[mujoco.MjModel, dict[str, Any]]:
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.fusestatic = False
    spec.option.timestep = TIMESTEP
    spec.option.gravity = [0.0, 0.0, -GRAVITY]
    pelvis = next((body for body in spec.bodies if body.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("URDF root body pelvis not found")
    if base_pose:
        pelvis.pos = BASE_WORLD.tolist()
        pelvis.quat = [math.cos(BASE_YAW / 2), 0.0, 0.0, math.sin(BASE_YAW / 2)]
    for index, pair in enumerate(HISTORICAL_EXCLUSIONS):
        spec.add_exclude(name=f"historical_accepted_overlap_{index}", bodyname1=pair[0], bodyname2=pair[1])
    active_joints = {joint for side in active_sides for joint in DRIVER_FOLLOWER[side][:2]}
    equality_targets = []
    for joint in spec.joints:
        name = str(joint.name)
        if not name or name in active_joints:
            continue
        if any(name in pair[:2] for pair in DRIVER_FOLLOWER.values()):
            target = -(ENDPOINT_MARGIN + (1.0 - 2.0 * ENDPOINT_MARGIN) * 1.0) if name.endswith("claw_joint") else ENDPOINT_MARGIN + (1.0 - 2.0 * ENDPOINT_MARGIN)
        else:
            target = float(joint.ref)
        spec.add_equality(name=f"diag_pose_lock_{name}", type=mujoco.mjtEq.mjEQ_JOINT,
                          name1=name, data=[target - float(joint.ref), 0.0, 0.0, 0.0, 0.0] + [0.0] * 6,
                          solref=[0.002, 1.0])
        equality_targets.append({"joint": name, "target_rad": target})
    actuator_names = []
    for side in active_sides:
        for role, joint_name in zip(("driver", "follower"), DRIVER_FOLLOWER[side][:2]):
            name = f"diag_{side}_{role}_servo"
            actuator_names.append(name)
            spec.add_actuator(name=name, target=joint_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                              biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
                              ctrllimited=True, ctrlrange=[-1.0, 1.0], forcelimited=True,
                              forcerange=[-1.0, 1.0])
    # A non-colliding floor supplies a familiar render reference but cannot
    # support or contact the robot in the empty-space motion test.
    spec.worldbody.add_geom(name="diagnostic_floor_visual", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0.0, 0.0, 0.0], size=[0.0, 0.0, 0.05],
                            contype=0, conaffinity=0, rgba=[0.28, 0.34, 0.40, 1.0])
    model = spec.compile()
    info = {"active_sides": sorted(active_sides), "historical_exclusions": [list(p) for p in HISTORICAL_EXCLUSIONS],
            "new_exclusions": [], "pose_lock_equalities": equality_targets,
            "actuator_names": actuator_names, "timestep_s": TIMESTEP,
            "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
            "solver_iterations": int(model.opt.iterations), "gravity_m_s2": model.opt.gravity.tolist()}
    return model, info


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
                 lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera)
    return renderer.render().copy()


def controller_step(model: mujoco.MjModel, data: mujoco.MjData, controllers: dict[str, Any],
                    references: dict[str, dict[str, float]]) -> dict[str, Any]:
    M = full_mass_matrix(model, data)
    rows = {}
    for side, reference in references.items():
        driver, follower, _, _ = DRIVER_FOLLOWER[side]
        side_gains = controllers["per_side"][side]["joints"]
        for role, name, q_ref, v_ref, a_ref in (
            ("driver", driver, reference["driver_q"], reference["driver_v"], reference["driver_a"]),
            ("follower", follower, reference["follower_q"], reference["follower_v"], reference["follower_a"]),
        ):
            dof = dof_addr(model, name)
            q = float(data.qpos[qpos_addr(model, name)])
            v = float(data.qvel[dof])
            gains = side_gains[name]
            pd = gains["kp_nm_rad"] * (q_ref - q) + gains["kd_nm_s_rad"] * (v_ref - v)
            bias = float(data.qfrc_bias[dof])
            ff = float(M[dof, dof]) * a_ref
            requested = bias + pd + ff
            cap = float(gains["effort_cap_nm"])
            applied = float(np.clip(requested, -cap, cap))
            actuator = actuator_id(model, f"diag_{side}_{role}_servo")
            data.ctrl[actuator] = applied
            rows[name] = {"qpos_rad": q, "qvel_rad_s": v, "target_q_rad": q_ref,
                          "target_qvel_rad_s": v_ref, "position_error_rad": q_ref - q,
                          "velocity_error_rad_s": v_ref - v, "kp_nm_rad": gains["kp_nm_rad"],
                          "kd_nm_s_rad": gains["kd_nm_s_rad"], "pd_torque_nm": pd,
                          "bias_feedforward_nm": bias, "reference_acceleration_torque_nm": ff,
                          "requested_torque_nm": requested, "applied_torque_nm": applied,
                          "effort_cap_nm": cap, "saturated": abs(requested) > cap}
    return rows


def valid_joint_state(model: mujoco.MjModel, data: mujoco.MjData, source_joints: dict[str, ET.Element]) -> list[dict[str, Any]]:
    violations = []
    for name, joint in source_joints.items():
        limit = joint.find("limit")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or limit is None or limit.get("lower") is None or joint.get("type") == "fixed":
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        low, high = float(limit.get("lower")), float(limit.get("upper"))
        if q < low - 1e-10 or q > high + 1e-10:
            violations.append({"joint": name, "qpos_rad": q, "range_rad": [low, high]})
    return violations


def dynamic_open_close(urdf: Path, canonical: Any, source_joints: dict[str, ET.Element],
                       output: Path) -> dict[str, Any]:
    runs = []
    for label, active in (("left_only", {"left"}), ("right_only", {"right"}), ("both", {"left", "right"})):
        model, build = build_dynamic_model(urdf, active)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        for side in DRIVER_FOLLOWER:
            set_aperture_qpos(model, data, side, 1.0)
        mujoco.mj_forward(model, data)
        controller = derive_controller(model, float(canonical.CANONICAL_X2_BOTTLE_MASS_KG))
        for side in active:
            for name, role in zip(DRIVER_FOLLOWER[side][:2], ("driver", "follower")):
                aid = actuator_id(model, f"diag_{side}_{role}_servo")
                cap = float(controller["per_side"][side]["joints"][name]["effort_cap_nm"])
                model.actuator_ctrlrange[aid] = [-cap, cap]
                model.actuator_forcerange[aid] = [-cap, cap]
        renderer = mujoco.Renderer(model, height=480, width=640)
        trace_path = output / f"open_close_{label}_trace.jsonl"
        video_path = output / f"open_close_{label}.mp4"
        image_path = output / f"open_close_{label}_overview.png"
        phase_schedule = [("OPEN_HOLD", HOLD_DURATION, 1.0, 1.0),
                          ("CLOSE", MOTION_DURATION, 1.0, 0.0),
                          ("CLOSED_HOLD", FINGER_HOLD_DURATION, 0.0, 0.0),
                          ("OPEN", MOTION_DURATION, 0.0, 1.0),
                          ("OPEN_HOLD_FINAL", HOLD_DURATION, 1.0, 1.0)]
        start_contacts = active_contact_rows(model, data)
        first_frame = render_frame(renderer, model, data, np.array([0.38, 0.32, 1.12]), 1.85, 145, -12)
        Image.fromarray(first_frame).save(image_path)
        frames = [first_frame]
        trace = []
        maximum_relation = {side: 0.0 for side in active}
        maximum_effort = {name: 0.0 for side in active for name in DRIVER_FOLLOWER[side][:2]}
        saturation = {name: 0 for name in maximum_effort}
        max_velocity = {name: 0.0 for name in maximum_effort}
        max_penetration = 0.0
        unexpected_contact_keys: set[str] = set()
        limit_violations: list[dict[str, Any]] = []
        nonfinite_state = False
        velocity_bound = (1.0 - 2.0 * ENDPOINT_MARGIN) * 1.875 / MOTION_DURATION * 1.5 + 0.05
        step = 0
        for phase, duration, start, end in phase_schedule:
            nsteps = int(round(duration / TIMESTEP))
            for local in range(nsteps):
                phase_t = local * TIMESTEP
                if phase in {"CLOSE", "OPEN"}:
                    ap, av, aa = minimum_jerk(phase_t, duration, start, end)
                else:
                    ap, av, aa = start, 0.0, 0.0
                reference = aperture_reference(ap, av, aa)
                refs = {side: reference for side in active}
                mujoco.mj_forward(model, data)
                control = controller_step(model, data, controller, refs)
                mujoco.mj_step(model, data)
                step += 1
                contacts = active_contact_rows(model, data)
                for contact in contacts:
                    max_penetration = max(max_penetration, contact["penetration_m"])
                    unexpected_contact_keys.add("|".join(sorted([contact["body1"], contact["body2"]])))
                violations = valid_joint_state(model, data, source_joints)
                limit_violations.extend({"step": step, **row} for row in violations)
                finite = bool(np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc))).all())
                nonfinite_state = nonfinite_state or not finite
                hand_state = {}
                for side in active:
                    driver, follower, narrow, wide = DRIVER_FOLLOWER[side]
                    rel = float(data.qpos[qpos_addr(model, follower)] + data.qpos[qpos_addr(model, driver)])
                    maximum_relation[side] = max(maximum_relation[side], abs(rel))
                    a, b = collision_geom_for_body(model, narrow), collision_geom_for_body(model, wide)
                    width = pair_contact_at(model, data, a, b)["distance_m"]
                    hand_state[side] = {"aperture_command": ap, "driver_qpos_rad": float(data.qpos[qpos_addr(model, driver)]),
                                        "follower_qpos_rad": float(data.qpos[qpos_addr(model, follower)]),
                                        "driver_qvel_rad_s": float(data.qvel[dof_addr(model, driver)]),
                                        "follower_qvel_rad_s": float(data.qvel[dof_addr(model, follower)]),
                                        "driver_qacc_rad_s2": float(data.qacc[dof_addr(model, driver)]),
                                        "follower_qacc_rad_s2": float(data.qacc[dof_addr(model, follower)]),
                                        "mimic_residual_rad": rel, "jaw_collision_geom_gap_m": width}
                for name, row in control.items():
                    maximum_effort[name] = max(maximum_effort[name], abs(row["applied_torque_nm"]))
                    max_velocity[name] = max(max_velocity[name], abs(float(data.qvel[dof_addr(model, name)])))
                    saturation[name] += int(row["saturated"])
                trace_row = {"step": step, "time_s": float(data.time), "phase": phase,
                             "hands": hand_state, "controller": control, "contacts": contacts,
                             "source_limit_violations": violations, "finite_state": finite,
                             "qpos_writes_during_rollout": 0, "bottle_qpos_writes": 0,
                             "model_exceptions": build["historical_exclusions"]}
                trace.append(trace_row)
                if step % 25 == 0:
                    frames.append(render_frame(renderer, model, data, np.array([0.38, 0.32, 1.12]), 1.85, 145, -12))
                if nonfinite_state or limit_violations or max(max_velocity.values(), default=0.0) > velocity_bound:
                    break
            if nonfinite_state or limit_violations or max(max_velocity.values(), default=0.0) > velocity_bound:
                break
        frames.append(render_frame(renderer, model, data, np.array([0.38, 0.32, 1.12]), 1.85, 145, -12))
        with trace_path.open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row, separators=(",", ":"), default=json_default) + "\n")
        imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
        end_contacts = active_contact_rows(model, data)
        velocity_exceeded = max(max_velocity.values(), default=0.0) > velocity_bound
        result = {"stage": label, "status": "PASS" if not start_contacts and not unexpected_contact_keys
                  and not end_contacts and not limit_violations and not nonfinite_state and not velocity_exceeded
                  and all(v <= MIMIC_TRACKING_DIAGNOSTIC_RAD for v in maximum_relation.values()) else "FAIL",
                  "physics_steps": step, "timestep_s": TIMESTEP, "initial_contacts": start_contacts,
                  "unexpected_contact_pairs_during_run": sorted(unexpected_contact_keys),
                  "final_contacts": end_contacts, "max_penetration_m": max_penetration,
                  "maximum_mimic_residual_rad": maximum_relation, "maximum_joint_velocity_rad_s": max_velocity,
                  "mimic_tracking_diagnostic_bound_rad": MIMIC_TRACKING_DIAGNOSTIC_RAD,
                  "maximum_actuator_effort_nm": maximum_effort, "saturated_steps": saturation,
                  "source_position_limit_violations": limit_violations, "nonfinite_state": nonfinite_state,
                  "velocity_bound_rad_s": velocity_bound, "velocity_bound_exceeded": velocity_exceeded,
                  "active_rollout_qpos_writes": 0, "bottle_qpos_writes": 0,
                  "controller": controller, "build": build,
                  "trace": trace_path.name, "video": video_path.name, "overview_png": image_path.name,
                  "interpretation": "Empty-space actuator-driven motion only; this is not a bottle-grasp or hardware result."}
        write_json(output / f"open_close_{label}_result.json", result)
        runs.append(result)
        renderer.close()
    status = "PASS" if all(run["status"] == "PASS" for run in runs) else "FAIL"
    return {"status": status, "runs": runs,
            "controller_classification": "SIMULATION_ONLY actuated mimic emulation; not a physical model of source transmission",
            "historical_collision_exclusions": [list(row) for row in HISTORICAL_EXCLUSIONS]}


def set_root_world(spec: mujoco.MjSpec, base_position: np.ndarray, yaw: float) -> None:
    pelvis = next((body for body in spec.bodies if body.name == "pelvis"), None)
    if pelvis is None:
        raise RuntimeError("pelvis root missing")
    pelvis.pos = base_position.tolist()
    pelvis.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def cylinder_geometry_gate(urdf: Path, canonical: Any, source_joints: dict[str, ET.Element],
                           output: Path) -> dict[str, Any]:
    # First compile exactly the pinned source geometry and obtain the intended
    # distal jaw surface witnesses from the actual collision meshes.
    spec = mujoco.MjSpec.from_file(str(urdf)); spec.compiler.fusestatic = False
    set_root_world(spec, BASE_WORLD, BASE_YAW)
    spec.option.timestep = TIMESTEP
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_aperture_qpos(model, data, "right", 1.0)
    set_aperture_qpos(model, data, "left", 1.0)
    mujoco.mj_forward(model, data)
    narrow_gid = next(g for g in collision_geoms(model)
                      if int(model.geom_bodyid[g]) == body_id(model, DRIVER_FOLLOWER["right"][2]))
    wide_gid = next(g for g in collision_geoms(model)
                    if int(model.geom_bodyid[g]) == body_id(model, DRIVER_FOLLOWER["right"][3]))
    pair = pair_contact_at(model, data, narrow_gid, wide_gid)
    jaw_mid = 0.5 * (np.asarray(pair["witness_a_world_m"]) + np.asarray(pair["witness_b_world_m"]))
    jaw_axis = np.asarray(pair["witness_b_world_m"]) - np.asarray(pair["witness_a_world_m"])
    jaw_axis /= max(float(np.linalg.norm(jaw_axis)), 1e-12)
    # Bottle uses its required vertical world axis. This one mechanically
    # defined candidate uses the exact midpoint of the compiled distal-pad
    # closest-surface witnesses, not the wrist or body origin.
    cylinder_center = jaw_mid
    spec2 = mujoco.MjSpec.from_file(str(urdf)); spec2.compiler.fusestatic = False
    spec2.option.timestep = TIMESTEP
    set_root_world(spec2, BASE_WORLD, BASE_YAW)
    for index, pair_exclusion in enumerate(HISTORICAL_EXCLUSIONS):
        spec2.add_exclude(name=f"historical_accepted_overlap_{index}", bodyname1=pair_exclusion[0], bodyname2=pair_exclusion[1])
    canonical.add_g1_canonical_table(spec2)
    bottle = spec2.worldbody.add_body(name="p0_test_cylinder", pos=cylinder_center.tolist())
    bottle.add_freejoint(name="p0_test_cylinder_free")
    bottle.add_geom(name="p0_test_cylinder_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT, 0.0], mass=CYLINDER_MASS,
                    friction=CYLINDER_FRICTION, condim=4, rgba=[0.12, 0.52, 0.82, 1.0])
    model2 = spec2.compile()
    data2 = mujoco.MjData(model2); mujoco.mj_resetData(model2, data2)
    set_aperture_qpos(model2, data2, "left", 1.0); set_aperture_qpos(model2, data2, "right", 1.0)
    mujoco.mj_forward(model2, data2)
    cylinder_gid = geom_id(model2, "p0_test_cylinder_geom")
    distances = []
    for gid in collision_geoms(model2):
        if gid == cylinder_gid or int(model2.geom_bodyid[gid]) == int(model2.geom_bodyid[cylinder_gid]):
            continue
        witness = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model2, data2, cylinder_gid, gid, 1.0, witness))
        bname = body_name(model2, int(model2.geom_bodyid[gid]))
        distances.append({"geom_id": gid, "geom": geom_name(model2, gid), "body": bname,
                          "signed_distance_m": distance, "penetration_m": max(0.0, -distance),
                          "witness_segment_world_m": witness.tolist(),
                          "intended_contact_body": bname in {DRIVER_FOLLOWER["right"][2], DRIVER_FOLLOWER["right"][3]}})
    by_aperture = []
    for aperture in np.linspace(1.0, 0.0, 41):
        set_aperture_qpos(model2, data2, "right", float(aperture))
        mujoco.mj_forward(model2, data2)
        row_distances = []
        for gid in collision_geoms(model2):
            if gid == cylinder_gid or int(model2.geom_bodyid[gid]) == int(model2.geom_bodyid[cylinder_gid]):
                continue
            witness = np.zeros(6)
            d = float(mujoco.mj_geomDistance(model2, data2, cylinder_gid, gid, 1.0, witness))
            row_distances.append({"geom_id": gid, "geom": geom_name(model2, gid),
                                  "body": body_name(model2, int(model2.geom_bodyid[gid])), "distance_m": d})
        jaw_dists = {name: min((r["distance_m"] for r in row_distances if r["body"] == name), default=None)
                     for name in DRIVER_FOLLOWER["right"][2:4]}
        min_other = min((r["distance_m"] for r in row_distances if r["body"] not in DRIVER_FOLLOWER["right"][2:4]), default=None)
        by_aperture.append({"aperture_ratio": float(aperture), "intended_jaw_distances_m": jaw_dists,
                            "minimum_non_gripping_geom_distance_m": min_other,
                            "minimum_all_robot_geom_distance_m": min((r["distance_m"] for r in row_distances), default=None)})
    failures = [r for r in distances if r["signed_distance_m"] <= 0.0 and not r["intended_contact_body"]]
    non_target_touch = [r for r in distances if r["signed_distance_m"] <= 0.0]
    intended = [r for r in distances if r["intended_contact_body"]]
    target_start_clear = all(r["signed_distance_m"] > 0.0 for r in intended)
    minimum = min(distances, key=lambda r: r["signed_distance_m"])
    table_bottom_z = canonical.G1_CANONICAL_TABLE_TOP_Z - 0.050
    cylinder_bottom_z = float(cylinder_center[2] - CYLINDER_HALF_HEIGHT)
    cylinder_top_z = float(cylinder_center[2] + CYLINDER_HALF_HEIGHT)
    if abs(cylinder_bottom_z - canonical.G1_CANONICAL_TABLE_TOP_Z) <= 1e-4:
        support_relation = "RESTING_ON_TABLE_BY_GEOMETRY"
    elif cylinder_top_z < table_bottom_z:
        support_relation = "BELOW_TABLETOP"
    elif cylinder_bottom_z < canonical.G1_CANONICAL_TABLE_TOP_Z:
        support_relation = "INTERSECTS_TABLE_HEIGHT_BAND"
    else:
        support_relation = "UNSUPPORTED_ABOVE_TABLE"
    result = {
        "status": "PASS" if not failures and target_start_clear else "FAIL",
        "physics_steps": 0,
        "candidate_count": 1,
        "candidate_rule": "right distal jaw collision-geom closest-witness midpoint at OPEN, cylinder vertical world axis, no hand/body origin used",
        "right_open_jaw_surface_gap_m": pair["distance_m"],
        "jaw_witness_a_world_m": pair["witness_a_world_m"], "jaw_witness_b_world_m": pair["witness_b_world_m"],
        "jaw_axis_world": jaw_axis.tolist(), "candidate_cylinder_center_world_m": cylinder_center.tolist(),
        "candidate_cylinder": {"diameter_m": 2 * CYLINDER_RADIUS, "height_m": 2 * CYLINDER_HALF_HEIGHT,
                               "mass_kg": CYLINDER_MASS, "friction": CYLINDER_FRICTION},
        "static_table_top_z_m": canonical.G1_CANONICAL_TABLE_TOP_Z,
        "cylinder_bottom_z_m": cylinder_bottom_z,
        "cylinder_top_z_m": cylinder_top_z,
        "table_underside_z_m": table_bottom_z,
        "table_support_relation": support_relation,
        "opening_surface_clearance_to_intended_jaws_m": [r["signed_distance_m"] for r in intended],
        "non_gripping_or_non_target_collisions_at_open": failures,
        "all_robot_distances_at_open": distances,
        "full_closure_samples": by_aperture,
        "minimum_clearance_record": minimum,
        "minimum_non_gripping_distance_over_closure_m": min((r["minimum_non_gripping_geom_distance_m"] for r in by_aperture if r["minimum_non_gripping_geom_distance_m"] is not None), default=None),
        "all_geometry_distance_trace": output / "cylinder_closure_geometry.jsonl",
        "historical_collision_exclusions_for_self_contacts_only": [list(x) for x in HISTORICAL_EXCLUSIONS],
        "note": "The exact single candidate is a geometry diagnostic. It does not validate a table-supported station or arm approach path.",
    }
    with (output / "cylinder_closure_geometry.jsonl").open("w", encoding="utf-8") as stream:
        for row in by_aperture:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    renderer = mujoco.Renderer(model2, height=480, width=640)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = cylinder_center
    camera.distance = 0.55; camera.azimuth = 145; camera.elevation = -12
    renderer.update_scene(data2, camera)
    Image.fromarray(renderer.render()).save(output / "cylinder_geometry_overview.png")
    renderer.close()
    write_json(output / "cylinder_geometry_gate.json", result)
    return result


def runtime_identity(model: mujoco.MjModel, script: Path, urdf: Path, vendor_root: Path) -> dict[str, Any]:
    package = Path(mujoco.__file__).resolve().parent
    libraries = sorted(package.glob("libmujoco.so*"))
    return {"python": sys.version, "platform": platform.platform(), "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(), "numpy": np.__version__, "pinocchio": pin.__version__,
            "timestep_s": float(model.opt.timestep), "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations),
            "gravity": model.opt.gravity.tolist(), "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "native_library_hashes": [{"path": str(p), "sha256": sha256(p)} for p in libraries],
            "source_repo": X2_REPO, "source_commit": subprocess.check_output(
                ["git", "-C", str(vendor_root), "rev-parse", "HEAD"], text=True).strip(),
            "source_worktree_clean": not bool(subprocess.check_output(
                ["git", "-C", str(vendor_root), "status", "--porcelain"], text=True).strip()),
            "urdf_path": str(urdf), "urdf_sha256": sha256(urdf),
            "runner_path": str(script), "runner_sha256": sha256(script)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--canonical-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-only", action="store_true",
                        help="run only the interrupted static cylinder-geometry gate; does not repeat open/close dynamics")
    args = parser.parse_args()
    root = args.vendor_root.resolve()
    urdf = root / X2_URDF_REL
    helper = args.canonical_helper.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Evidence output already exists; refusing overwrite: {output}")
    if not urdf.is_file() or not helper.is_file():
        raise FileNotFoundError("Pinned URDF or canonical table helper is missing")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if commit != X2_PIN or dirty or sha256(urdf) != X2_URDF_SHA256:
        raise RuntimeError(f"Source identity mismatch: commit={commit}; dirty={bool(dirty)}; sha={sha256(urdf)}")
    if sha256(helper) != CANONICAL_HELPER_SHA256:
        raise RuntimeError(f"Canonical helper identity mismatch: {sha256(helper)}")
    output.mkdir(parents=True)
    canonical = load_canonical_module(helper)
    robot = ET.parse(urdf).getroot()
    source_joints = {j.get("name", ""): j for j in robot.findall("joint")}
    if args.geometry_only:
        _, geometry_model = source_model(urdf)
        identity = runtime_identity(geometry_model, Path(__file__).resolve(), urdf, root)
        write_json(output / "runtime_identity.json", identity)
        geometry = cylinder_geometry_gate(urdf, canonical, source_joints, output)
        write_json(output / "CYLINDER_GEOMETRY_GATE.json", geometry)
        command = ("PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:"
                   "/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages "
                   "LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:"
                   "/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 "
                   f"python3 {Path(__file__).resolve()} --vendor-root {root} --canonical-helper {helper} "
                   f"--output {output} --geometry-only")
        (output / "REPRODUCE.md").write_text("# Geometry-only reproduction\n\n```bash\n" + command + "\n```\n", encoding="utf-8")
        (output / "REPORT.md").write_text(
            "# Issue #46 cylinder geometry gate\n\n"
            f"Status: **{geometry['status']}**\n\n"
            "Static compiled-geom check only; no physics steps, contact test, or lift were run.\n",
            encoding="utf-8")
        print(json.dumps({"status": geometry["status"], "output": str(output)}, indent=2))
        return 0 if geometry["status"] == "PASS" else 2
    _, source = source_model(urdf)
    identity = runtime_identity(source, Path(__file__).resolve(), urdf, root)
    write_json(output / "runtime_identity.json", identity)
    inventory = source_records(urdf, source, root)
    write_json(output / "source_model_inventory.json", inventory)
    mesh_audit_data = mujoco.MjData(source); mujoco.mj_forward(source, mesh_audit_data)
    wrist_mesh = check_mesh_pair(source, mesh_audit_data, "right_wrist_roll_link")
    write_json(output / "wrist_visual_collision_comparison.json", wrist_mesh)
    fk = run_fk_audit(source, urdf, source_joints)
    write_json(output / "independent_fk_comparison.json", fk)
    collision = source_collision_sweep(source, source_joints)
    with (output / "source_collision_sweep.jsonl").open("w", encoding="utf-8") as stream:
        for row in collision["samples_detail"]:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary_collision = {key: value for key, value in collision.items() if key != "samples_detail"}
    write_json(output / "source_collision_sweep_summary.json", summary_collision)
    inventory_consistent = (not inventory["missing_compiled_links"]
                            and not inventory["missing_compiled_movable_joints"]
                            and not inventory["extra_compiled_joints"]
                            and not inventory["extra_compiled_bodies"]
                            and all(row["range_mapping_matches"] and row["compiled_driver_present"]
                                    and row["compiled_follower_present"] for row in inventory["mimic_checks"]))
    assembly_status = "PASS" if fk["status"] == "PASS" and inventory_consistent and collision["status"] == "PASS" else "FAIL"
    assembly = {"status": assembly_status,
                "source": {"repository": X2_REPO, "commit": commit, "license": "Mulan PSL v2",
                           "urdf": str(X2_URDF_REL), "urdf_sha256": sha256(urdf),
                           "source_worktree_clean": True},
                "compiled_model": {"nq": source.nq, "nv": source.nv, "njnt": source.njnt,
                                   "nbody": source.nbody, "ngeom": source.ngeom, "nmesh": source.nmesh,
                                   "neq": source.neq, "nu": source.nu},
                "inventory_path": "source_model_inventory.json",
                "inventory_consistent": inventory_consistent,
                "fk_comparison_path": "independent_fk_comparison.json",
                "fk_status": fk["status"], "collision_sweep_status": collision["status"],
                "collision_sweep_path": "source_collision_sweep_summary.json",
                "wrist_mesh_comparison_path": "wrist_visual_collision_comparison.json",
                "existing_simulation_only_collision_exclusions": [list(p) for p in HISTORICAL_EXCLUSIONS],
                "added_to_source_fidelity_model": [],
                "assembly_gate_note": "Assembly/fk consistency is assessed separately from original-source collision overlaps. The listed overlaps are not silently excluded from the source collision audit."}
    write_json(output / "X2_OMNIPICKER_ASSEMBLY_GATE.json", assembly)

    if assembly_status == "PASS":
        dynamic = dynamic_open_close(urdf, canonical, source_joints, output)
    else:
        dynamic = {"status": "BLOCKED", "reason": "ASSEMBLY_GATE did not pass; serial gate prevents dynamics."}
    write_json(output / "OPEN_CLOSE_GATE.json", dynamic)
    if dynamic["status"] == "PASS":
        geometry = cylinder_geometry_gate(urdf, canonical, source_joints, output)
    else:
        geometry = {"status": "BLOCKED", "reason": "OPEN_CLOSE_GATE did not pass; serial gate prevents cylinder geometry stage."}
    stage_status = {
        "ASSEMBLY_GATE": assembly_status,
        "OPEN_CLOSE_GATE": dynamic["status"],
        "CYLINDER_GEOMETRY_GATE": geometry["status"],
        "ISOLATED_CONTACT_LIFT_50MM": "NOT_RUN",
        "FULL_X2_CONTACT_LIFT_50MM": "NOT_RUN",
        "FULL_X2_RELEASE_SETTLE": "NOT_RUN",
    }
    result = {"overall_status": "PASS" if all(v == "PASS" for v in stage_status.values()) else "BLOCKED",
              "stages": stage_status, "runtime": identity,
              "assembly_gate": assembly, "open_close_gate": dynamic, "cylinder_geometry_gate": geometry,
              "prohibited_mechanisms": {"object_qpos_writes": 0, "weld_or_equality_attachment": False,
                                         "mocap_attachment": False, "artificial_object_force": False,
                                         "hidden_support": False, "friction_changed": False,
                                         "vendor_mesh_changed": False},
              "unresolved": ["No contact or lift was run unless assembly, open/close, and cylinder geometry gates all pass."]}
    write_json(output / "result.json", result)
    command = ("PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:"
               "/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages "
               "LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:"
               "/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 "
               f"python3 {Path(__file__).resolve()} --vendor-root {root} --canonical-helper {helper} --output {output}")
    (output / "REPRODUCE.md").write_text("# Reproduction\n\n```bash\n" + command + "\n```\n", encoding="utf-8")
    lines = [
        "# Issue #46: X2 OmniPicker P0 acceptance", "",
        f"Overall status: **{result['overall_status']}**", "",
        "## Stage results", "",
    ]
    for name, status in stage_status.items():
        lines.append(f"- `{name}`: **{status}**")
    lines += ["", "## Findings", "",
              f"- Official URDF import: {source.nq} scalar coordinates, {source.ngeom} geoms, {source.nmesh} meshes, {source.nu} actuators, {source.neq} equalities before diagnostic changes.",
              f"- Independent Pinocchio/MuJoCo FK: **{fk['status']}**, worst translation error {max(r['maximum_translation_error_m'] for r in fk['configurations']):.3e} m and worst rotation error {max(r['maximum_rotation_error_rad'] for r in fk['configurations']):.3e} rad.",
              f"- Source collision sweep with no added exclusions: **{collision['status']}**, {len(collision['pair_minimums'])} contacting body pairs over 41 static aperture samples.",
              f"- Empty-space jaw motion: **{dynamic['status']}** using an explicitly simulation-only paired actuator model and only the already documented 11 collision exclusions.",
              (f"- 70 mm cylinder static geometry: **{geometry['status']}**, minimum open clearance "
               f"{geometry['minimum_clearance_record']['signed_distance_m'] * 1000:.2f} mm against "
               f"{geometry['minimum_clearance_record']['body']}." if "minimum_clearance_record" in geometry else
               f"- 70 mm cylinder static geometry: **{geometry['status']}**; {geometry.get('reason', 'see gate JSON')}.") ,
              "", "## Claim boundary", "",
              "No object contact, load transfer, or lift is claimed unless its gate ran. The dynamic paired-jaw actuator is SIMULATION_ONLY and is not a hardware transmission model. The canonical blue multipart bottle was not modified.",
              "", "## Evidence", "",
              "Machine-readable results and raw per-step/per-aperture traces are in this evidence folder. `X2_OMNIPICKER_ASSEMBLY_GATE.json` records source/model identity and FK/collision audit.",
              ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall_status": result["overall_status"], "stages": stage_status,
                      "output": str(output)}, indent=2))
    return 0 if result["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
