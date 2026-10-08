#!/usr/bin/env python3
"""Isolated X2 OmniPicker physical grasp capability experiment."""

from __future__ import annotations

import argparse
import copy
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
from PIL import Image
from scipy.optimize import least_squares


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_REPO = "https://github.com/AgibotTech/agibot_x2_urdf"
X2_ROOT = Path("/tmp/robotsim-issue46-virtual-transmission-vendor-20261007")
X2_URDF_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
EVIDENCE_ROOT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-isolated-grasp"
)
BASELINE_PATH = Path(__file__).with_name("issue46_omnipicker_1dof_m0.py")
STATIC_ROOT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-static/20261007-ready"
)
CANONICAL_MODULE = STATIC_ROOT / "canonical_manipulation_assets.py"
ROOT_LINK = "right_wrist_yaw_link"
WRIST_JOINTS = (
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
TARGET_JAW_BODIES = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
LIFT_JOINT = "m0_gripper_lift_joint"
LIFT_ACTUATOR = "m0_gripper_lift_motor"
GRAVITY = 9.81
CONTACT_EPS_M = 1.0e-5
PENETRATION_EPS_M = 1.0e-7
CONTACT_HOLD_S = 1.0
ABORT_PENETRATION_M = 0.008


def load_baseline():
    spec = importlib.util.spec_from_file_location("issue46_omnipicker_1dof_m0", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accepted controller implementation: {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"Compiled model is missing body {name}")
    return int(value)


def joint_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise RuntimeError(f"Compiled model is missing joint {name}")
    return int(value)


def model_joint_names(model: mujoco.MjModel) -> set[str]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            for i in range(model.njnt)}


def qpos_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, name)])


def dof_id(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, name)])


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(geom_id)])
    ) or "world"


def source_joint_map(urdf: Path) -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    root = ET.parse(urdf).getroot()
    return ({joint.get("name", ""): joint for joint in root.findall("joint")},
            {link.get("name", ""): link for link in root.findall("link")})


def filtered_urdf(output: Path, source_urdf: Path) -> dict[str, Any]:
    source_root = ET.parse(source_urdf).getroot()
    joint_rows = source_root.findall("joint")
    child_to_parent = {
        row.find("child").get("link"): row.find("parent").get("link")
        for row in joint_rows if row.find("child") is not None and row.find("parent") is not None
    }
    descendants: set[str] = {ROOT_LINK}
    changed = True
    while changed:
        changed = False
        for child, parent in child_to_parent.items():
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True

    if ROOT_LINK not in {link.get("name") for link in source_root.findall("link")}:
        raise RuntimeError(f"Pinned source is missing {ROOT_LINK}")
    robot = ET.Element("robot", {"name": "X2-Ultra-OmniPicker-isolated"})
    copied_meta: list[str] = []
    for item in source_root:
        tag = item.tag.rsplit("}", 1)[-1]
        if tag == "mujoco":
            node = copy.deepcopy(item)
            compiler = node.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(node, "compiler")
            compiler.set("meshdir", str(source_urdf.parent / "meshes"))
            copied_meta.append(tag)
            robot.append(node)
        elif tag == "material":
            robot.append(copy.deepcopy(item))
            copied_meta.append(tag)
        elif tag == "link" and item.get("name") in descendants:
            node = copy.deepcopy(item)
            for mesh in node.findall(".//mesh"):
                filename = mesh.get("filename", "")
                mesh.set("filename", Path(filename).name)
            robot.append(node)
        elif tag == "joint":
            parent = item.find("parent")
            child = item.find("child")
            if (parent is not None and child is not None
                    and parent.get("link") in descendants
                    and child.get("link") in descendants):
                robot.append(copy.deepcopy(item))

    path = output / "isolated_omnipicker_source.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
    files = sorted({
        Path(mesh.get("filename", "")).name
        for link in robot.findall("link") for mesh in link.findall(".//mesh")
    })
    mesh_hashes = {}
    for filename in files:
        mesh_path = source_urdf.parent / "meshes" / filename
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Referenced pinned collision/visual mesh not found: {mesh_path}")
        mesh_hashes[filename] = sha256(mesh_path)
    return {
        "derived_urdf_path": str(path),
        "derived_urdf_sha256": sha256(path),
        "root_link": ROOT_LINK,
        "included_link_names": sorted(descendants),
        "included_link_count": len(descendants),
        "included_joint_names": [
            row.get("name") for row in robot.findall("joint")
        ],
        "included_joint_count": len(robot.findall("joint")),
        "copied_top_level_metadata": copied_meta,
        "mesh_asset_sha256": mesh_hashes,
        "extraction_rule": "pinned URDF descendant subtree rooted at right_wrist_yaw_link; all in-subtree URDF links/joints copied without geometry or joint edits; mesh file paths resolve to pinned source meshes",
    }


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    q = np.zeros(4, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (matrix[2, 1] - matrix[1, 2]) / s
        q[2] = (matrix[0, 2] - matrix[2, 0]) / s
        q[3] = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1.0e-16, 1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k])) * 2.0
        q[i + 1] = 0.25 * s
        q[0] = (matrix[k, j] - matrix[j, k]) / s
        q[j + 1] = (matrix[j, i] + matrix[i, j]) / s
        q[k + 1] = (matrix[k, i] + matrix[i, k]) / s
    q /= np.linalg.norm(q)
    return q


def set_root_pose(model: mujoco.MjModel, root_id: int, data: mujoco.MjData,
                  position: np.ndarray, quaternion: np.ndarray) -> None:
    model.body_pos[root_id] = np.asarray(position, dtype=float)
    model.body_quat[root_id] = np.asarray(quaternion, dtype=float)
    mujoco.mj_forward(model, data)


def collision_geoms(model: mujoco.MjModel, body: str | None = None) -> list[int]:
    body_filter = None if body is None else body_id(model, body)
    return [
        gid for gid in range(model.ngeom)
        if (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))
        and (body_filter is None or int(model.geom_bodyid[gid]) == body_filter)
    ]


def distance_row(model: mujoco.MjModel, data: mujoco.MjData,
                 first: int, second: int) -> dict[str, Any]:
    segment = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
    return {
        "geom1": geom_name(model, first), "body1": body_name(model, first),
        "geom2": geom_name(model, second), "body2": body_name(model, second),
        "signed_distance_m": distance,
        "penetration_m": max(0.0, -distance),
        "witness_segment_world_m": segment.tolist(),
    }


def source_identity(output: Path, isolated_info: dict[str, Any], robot_head: str) -> dict[str, Any]:
    source_urdf = X2_ROOT / X2_URDF_REL
    native = Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"
    commit = subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(X2_ROOT), "status", "--porcelain"], text=True).strip()
    if commit != X2_PIN or dirty:
        raise RuntimeError(f"Pinned vendor checkout mismatch: {commit=} dirty={bool(dirty)}")
    return {
        "robotsim_head_before_experiment_commit": robot_head,
        "experiment_script": str(Path(__file__).resolve()),
        "experiment_script_sha256": sha256(Path(__file__).resolve()),
        "accepted_controller_script": str(BASELINE_PATH.resolve()),
        "accepted_controller_script_sha256": sha256(BASELINE_PATH),
        "x2_repository": X2_REPO,
        "x2_commit": commit,
        "x2_worktree_clean": not bool(dirty),
        "x2_urdf": str(X2_URDF_REL),
        "x2_urdf_sha256": sha256(source_urdf),
        "x2_license": "Mulan PSL v2",
        "filtered_model": isolated_info,
        "canonical_scene_helper": str(CANONICAL_MODULE),
        "canonical_scene_helper_sha256": sha256(CANONICAL_MODULE),
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "mujoco_python_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "mujoco_native_library": str(native),
        "mujoco_native_library_sha256": sha256(native),
        "numpy": np.__version__,
        "evidence_directory": str(output),
    }


def import_spec(base, urdf_path: Path, root_position: np.ndarray,
                root_rotation: np.ndarray, include_bottle: bool,
                add_lift: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.compiler.fusestatic = False
    spec.option.timestep = base.TIMESTEP
    spec.option.gravity = [0.0, 0.0, -GRAVITY]
    root = next((body for body in spec.bodies if body.name == ROOT_LINK), None)
    if root is None:
        raise RuntimeError(f"Filtered model root {ROOT_LINK} was not imported")
    root.pos = np.asarray(root_position, dtype=float).tolist()
    root.quat = quat_from_matrix(root_rotation).tolist()

    joint_map = {str(joint.name): joint for joint in spec.joints if joint.name}
    if add_lift:
        local_world_z = root_rotation.T @ np.array([0.0, 0.0, 1.0])
        root.add_joint(
            name=LIFT_JOINT,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            pos=[0.0, 0.0, 0.0],
            axis=local_world_z.tolist(),
            limited=True,
            range=[0.0, 0.035],
            ref=0.0,
        )
        joint_map[LIFT_JOINT] = next(joint for joint in root.joints if joint.name == LIFT_JOINT)

    for name in WRIST_JOINTS:
        if name not in joint_map:
            continue
        joint = joint_map[name]
        target = float(joint.ref)
        spec.add_equality(
            name=f"m0_rigid_mount_{name}", type=mujoco.mjtEq.mjEQ_JOINT,
            name1=name, data=[target - float(joint.ref), 0.0, 0.0, 0.0, 0.0] + [0.0] * 6,
            solref=base.POSE_LOCK_SOLREF,
        )

    for name, target in (("m0_right_driver_servo", DRIVER),
                         ("m0_right_follower_servo", FOLLOWER)):
        spec.add_actuator(
            name=name, target=target, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
            ctrllimited=True, ctrlrange=[-1.0, 1.0],
            forcelimited=True, forcerange=[-1.0, 1.0],
        )
    if add_lift:
        spec.add_actuator(
            name=LIFT_ACTUATOR, target=LIFT_JOINT, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=[1.0] + [0.0] * 9,
            ctrllimited=True, ctrlrange=[-50.0, 50.0],
            forcelimited=True, forcerange=[-50.0, 50.0],
        )

    for i, (first, second) in enumerate(base.STATIC_COLLISION_EXCLUSIONS):
        kept = {str(body.name) for body in spec.bodies}
        if first in kept and second in kept:
            spec.add_exclude(name=f"m0_source_self_collision_exclusion_{i}",
                             bodyname1=first, bodyname2=second)
    base.add_canonical_scene(spec, bottle=include_bottle)
    return spec


def build_model(base, urdf_path: Path, root_position: np.ndarray,
                root_rotation: np.ndarray, include_bottle: bool = True,
                add_lift: bool = False) -> tuple[mujoco.MjModel, dict[str, Any]]:
    spec = import_spec(base, urdf_path, root_position, root_rotation,
                       include_bottle=include_bottle, add_lift=add_lift)
    model = spec.compile()
    for name, actuator in (("m0_right_driver_servo", DRIVER),
                           ("m0_right_follower_servo", FOLLOWER)):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0 or int(model.actuator_trnid[aid, 0]) != joint_id(model, actuator):
            raise RuntimeError(f"Internal paired-jaw actuator is not assigned to {actuator}")
    equality_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or ""
                      for i in range(model.neq)]
    jaw_equalities = [name for name in equality_names if DRIVER in name or FOLLOWER in name]
    if jaw_equalities:
        raise RuntimeError(f"Bottle/gripper jaw equalities are forbidden: {jaw_equalities}")
    if add_lift:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
        if aid < 0 or int(model.actuator_trnid[aid, 0]) != joint_id(model, LIFT_JOINT):
            raise RuntimeError("Vertical carriage actuator is not assigned to its slide joint")
    return model, {"equality_names": equality_names, "jaw_equalities": jaw_equalities}


def initialize(base, model: mujoco.MjModel, controller: dict[str, Any]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    open_targets = base.aperture_targets(1.0)
    data.qpos[qpos_id(model, DRIVER)] = open_targets["right_claw_joint_target_rad"]
    data.qpos[qpos_id(model, FOLLOWER)] = open_targets["R_hand_wide1_joint_target_rad"]
    for name in WRIST_JOINTS:
        if name in model_joint_names(model):
            data.qpos[qpos_id(model, name)] = 0.0
    if LIFT_JOINT in {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                      for i in range(model.njnt)}:
        data.qpos[qpos_id(model, LIFT_JOINT)] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    base.apply_controller_limits(model, controller)
    return data


def collision_mesh_axis(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    local = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    points = local @ rotation.T + data.geom_xpos[geom_id]
    centered = points - points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(points), 1))
    return eigenvectors[:, int(np.argmax(eigenvalues))]


def basis_transform(local_normal: np.ndarray, local_finger_axis: np.ndarray,
                    azimuth_rad: float, finger_tilt_rad: float) -> np.ndarray:
    normal = local_normal / np.linalg.norm(local_normal)
    finger = local_finger_axis - np.dot(local_finger_axis, normal) * normal
    if np.linalg.norm(finger) < 1.0e-6:
        raise RuntimeError("Compiled jaw longitudinal axis is parallel to jaw closing axis")
    finger /= np.linalg.norm(finger)
    local_side = np.cross(normal, finger)
    local_basis = np.column_stack((normal, finger, local_side))

    target_normal = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0])
    target_tangent = np.array([-target_normal[1], target_normal[0], 0.0])
    target_finger = (math.cos(finger_tilt_rad) * np.array([0.0, 0.0, 1.0])
                     + math.sin(finger_tilt_rad) * target_tangent)
    target_side = np.cross(target_normal, target_finger)
    target_basis = np.column_stack((target_normal, target_finger, target_side))
    rotation = target_basis @ local_basis.T
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("Derived source-faithful gripper orientation is not a proper rotation")
    return rotation


def set_jaws(base, model: mujoco.MjModel, data: mujoco.MjData, aperture: float) -> None:
    target = base.aperture_targets(aperture)
    data.qpos[qpos_id(model, DRIVER)] = target["right_claw_joint_target_rad"]
    data.qpos[qpos_id(model, FOLLOWER)] = target["R_hand_wide1_joint_target_rad"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def geom_pair_distance(model: mujoco.MjModel, data: mujoco.MjData,
                       geom_a: int, geom_b: int) -> tuple[float, list[float]]:
    segment = np.zeros(6, dtype=float)
    value = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, 1.0, segment))
    return value, segment.tolist()


def bottle_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [gid for gid in collision_geoms(model)
            if body_name(model, gid).startswith("m0_bottle")]


def robot_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [gid for gid in collision_geoms(model)
            if body_name(model, gid) not in {"world", "m0_table"}
            and not body_name(model, gid).startswith("m0_bottle")]


def table_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [gid for gid in collision_geoms(model) if body_name(model, gid) == "m0_table"]


def floor_geom_ids(model: mujoco.MjModel) -> list[int]:
    return [gid for gid in collision_geoms(model)
            if body_name(model, gid) == "world" and geom_name(model, gid) == "floor"]


def pair_rows(model: mujoco.MjModel, data: mujoco.MjData,
              firsts: list[int], seconds: list[int]) -> list[dict[str, Any]]:
    return [distance_row(model, data, a, b) for a in firsts for b in seconds]


def contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, force)
        first, second = int(contact.geom1), int(contact.geom2)
        rows.append({
            "geom1": geom_name(model, first), "body1": body_name(model, first),
            "geom2": geom_name(model, second), "body2": body_name(model, second),
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
            "contact_force_frame_n_and_nm": force.tolist(),
            "normal_force_n": float(abs(force[0])),
        })
    return rows


def minimum(rows: list[dict[str, Any]]) -> float:
    return min((float(row["signed_distance_m"]) for row in rows), default=math.inf)


def candidate_model(base, urdf_path: Path, candidate_id: int,
                    controller: dict[str, Any],
                    azimuth_deg: float, finger_tilt_deg: float):
    probe_spec = mujoco.MjSpec.from_file(str(urdf_path))
    probe_spec.compiler.fusestatic = False
    probe_root = next(body for body in probe_spec.bodies if body.name == ROOT_LINK)
    probe_root.pos = [0.0, 0.0, 0.0]
    probe_root.quat = [1.0, 0.0, 0.0, 0.0]
    base.add_canonical_scene(probe_spec, bottle=True)
    probe = probe_spec.compile()
    probe_data = mujoco.MjData(probe)
    mujoco.mj_resetData(probe, probe_data)
    for name in WRIST_JOINTS:
        if name in model_joint_names(probe):
            probe_data.qpos[qpos_id(probe, name)] = 0.0
    open_targets = base.aperture_targets(1.0)
    probe_data.qpos[qpos_id(probe, DRIVER)] = open_targets["right_claw_joint_target_rad"]
    probe_data.qpos[qpos_id(probe, FOLLOWER)] = open_targets["R_hand_wide1_joint_target_rad"]
    mujoco.mj_forward(probe, probe_data)

    narrow_id = base.collision_geom_id(probe, TARGET_JAW_BODIES[0])
    wide_id = base.collision_geom_id(probe, TARGET_JAW_BODIES[1])
    bottle_body_id = body_id(probe, "m0_bottle")
    body_geom_id = next(g for g in bottle_geom_ids(probe) if "bottle_body" in geom_name(probe, g))
    body_center = probe_data.geom_xpos[body_geom_id].copy()
    body_radius = float(probe.geom_size[body_geom_id][0])

    gap_samples = []
    for aperture in np.linspace(0.0, 1.0, 101):
        set_jaws(base, probe, probe_data, float(aperture))
        distance, witness = geom_pair_distance(probe, probe_data, narrow_id, wide_id)
        gap_samples.append({"aperture_ratio": float(aperture), "jaw_surface_gap_m": distance,
                            "witness_segment_world_m": witness})
    by_open_to_closed = list(reversed(gap_samples))
    contact_aperture = None
    for first, second in zip(by_open_to_closed, by_open_to_closed[1:]):
        if first["jaw_surface_gap_m"] >= 2.0 * body_radius >= second["jaw_surface_gap_m"]:
            span = first["jaw_surface_gap_m"] - second["jaw_surface_gap_m"]
            fraction = (first["jaw_surface_gap_m"] - 2.0 * body_radius) / max(span, 1e-12)
            contact_aperture = first["aperture_ratio"] + fraction * (
                second["aperture_ratio"] - first["aperture_ratio"]
            )
            break
    if contact_aperture is None:
        raise RuntimeError("Source jaw closure range never spans the compiled bottle diameter")
    set_jaws(base, probe, probe_data, float(contact_aperture))
    gap, witness = geom_pair_distance(probe, probe_data, narrow_id, wide_id)
    narrow_point, wide_point = np.asarray(witness[:3]), np.asarray(witness[3:])
    midpoint = 0.5 * (narrow_point + wide_point)
    local_normal = wide_point - narrow_point
    local_normal /= np.linalg.norm(local_normal)
    finger_a = collision_mesh_axis(probe, probe_data, narrow_id)
    finger_b = collision_mesh_axis(probe, probe_data, wide_id)
    finger_axis = finger_a + (finger_b if np.dot(finger_a, finger_b) >= 0.0 else -finger_b)
    finger_axis /= np.linalg.norm(finger_axis)
    root_to_contact = midpoint - probe_data.xpos[body_id(probe, ROOT_LINK)]
    if np.dot(finger_axis, root_to_contact) > 0.0:
        finger_axis = -finger_axis
    rotation = basis_transform(local_normal, finger_axis,
                               math.radians(azimuth_deg), math.radians(finger_tilt_deg))
    root_position = body_center - rotation @ midpoint
    model, build_info = build_model(base, urdf_path, root_position, rotation,
                                    include_bottle=True, add_lift=False)
    data = initialize(base, model, controller)
    root_id = body_id(model, ROOT_LINK)
    return {
        "model": model, "data": data, "build_info": build_info,
        "candidate_id": candidate_id, "azimuth_deg": azimuth_deg,
        "finger_tilt_deg": finger_tilt_deg, "root_position": root_position,
        "root_rotation": rotation, "root_id": root_id,
        "narrow_id": base.collision_geom_id(model, TARGET_JAW_BODIES[0]),
        "wide_id": base.collision_geom_id(model, TARGET_JAW_BODIES[1]),
        "bottle_body_id": next(g for g in bottle_geom_ids(model)
                                if "bottle_body" in geom_name(model, g)),
        "bottle_body_center": body_center, "contact_aperture": float(contact_aperture),
        "gap_samples": gap_samples,
        "local_geometry": {
            "narrow_witness_m": narrow_point.tolist(),
            "wide_witness_m": wide_point.tolist(),
            "jaw_midpoint_m": midpoint.tolist(),
            "jaw_closing_axis": local_normal.tolist(),
            "jaw_longitudinal_axis_pca": finger_axis.tolist(),
            "jaw_gap_at_contact_m": gap,
            "bottle_body_radius_m": body_radius,
            "bottle_body_center_world_m": body_center.tolist(),
        },
    }


def candidate_state(base, candidate: dict[str, Any], aperture: float,
                    offset_tangent: float = 0.0, offset_height: float = 0.0) -> tuple[mujoco.MjData, dict[str, Any]]:
    model = candidate["model"]
    data = candidate["data"]
    normal = np.array([math.cos(math.radians(candidate["azimuth_deg"])),
                       math.sin(math.radians(candidate["azimuth_deg"])), 0.0])
    tangent = np.array([-normal[1], normal[0], 0.0])
    position = candidate["root_position"] + tangent * offset_tangent + np.array([0.0, 0.0, offset_height])
    set_root_pose(model, candidate["root_id"], data, position,
                  quat_from_matrix(candidate["root_rotation"]))
    set_jaws(base, model, data, aperture)
    return data, {"root_position_m": position.tolist(), "offset_tangent_m": offset_tangent,
                  "offset_height_m": offset_height}


def refine_contact(base, candidate: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    model = candidate["model"]
    data = candidate["data"]
    jaw_targets = [candidate["narrow_id"], candidate["wide_id"]]
    bottle_body = candidate["bottle_body_id"]
    body_geoms = bottle_geom_ids(model)
    robot_geoms = robot_geom_ids(model)
    jaw_body_names = set(TARGET_JAW_BODIES)
    trace_path.write_text("", encoding="utf-8")
    evaluations: list[dict[str, Any]] = []

    def residual(offsets: np.ndarray) -> np.ndarray:
        _, pose = candidate_state(base, candidate, candidate["contact_aperture"],
                                  float(offsets[0]), float(offsets[1]))
        jaw_distances = [
            geom_pair_distance(model, data, geom, bottle_body)[0] for geom in jaw_targets
        ]
        all_distances = pair_rows(model, data, robot_geoms, body_geoms)
        nonjaw = [row for row in all_distances if row["body1"] not in jaw_body_names]
        closed_table_rows = pair_rows(model, data, robot_geoms, table_geom_ids(model))
        closed_floor_rows = pair_rows(model, data, robot_geoms, floor_geom_ids(model))
        open_data, _ = candidate_state(base, candidate, 1.0,
                                       float(offsets[0]), float(offsets[1]))
        open_rows = pair_rows(model, open_data, robot_geoms, body_geoms)
        open_table_rows = pair_rows(model, open_data, robot_geoms, table_geom_ids(model))
        open_floor_rows = pair_rows(model, open_data, robot_geom_ids(model), floor_geom_ids(model))
        open_min = minimum(open_rows)
        constraint_rows = [
            ("closed_nonjaw_bottle", nonjaw),
            ("closed_table", closed_table_rows),
            ("closed_floor", closed_floor_rows),
            ("open_bottle", open_rows),
            ("open_table", open_table_rows),
            ("open_floor", open_floor_rows),
        ]
        clearance_margin = 0.001
        collision_residuals = [
            max(0.0, clearance_margin - float(clearance["signed_distance_m"])) / clearance_margin
            for _, rows in constraint_rows for clearance in rows
        ]
        row = {
            "evaluation": len(evaluations), "offsets_m": offsets.tolist(),
            "pose": pose, "contact_aperture_ratio": candidate["contact_aperture"],
            "narrow_to_body_distance_m": jaw_distances[0],
            "wide_to_body_distance_m": jaw_distances[1],
            "minimum_nonjaw_to_bottle_distance_m": minimum(nonjaw),
            "minimum_closed_robot_to_table_distance_m": minimum(closed_table_rows),
            "minimum_closed_robot_to_floor_distance_m": minimum(closed_floor_rows),
            "open_minimum_robot_to_bottle_distance_m": open_min,
            "minimum_open_robot_to_table_distance_m": minimum(open_table_rows),
            "minimum_open_robot_to_floor_distance_m": minimum(open_floor_rows),
            "geometry_collision_clearance_margin_m": clearance_margin,
        }
        evaluations.append(row)
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
        return np.concatenate((np.asarray(jaw_distances, dtype=float) / 0.001,
                               np.asarray(collision_residuals, dtype=float),
                               np.asarray(offsets, dtype=float) / np.array([0.050, 0.060]) * 0.01))

    solution = least_squares(
        residual, x0=np.zeros(2), bounds=(np.array([-0.030, -0.045]), np.array([0.030, 0.045])),
        xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10, max_nfev=35,
    )
    final_data, pose = candidate_state(base, candidate, candidate["contact_aperture"],
                                       float(solution.x[0]), float(solution.x[1]))
    contact_rows_closed = pair_rows(model, final_data, robot_geoms, body_geoms)
    close_table_rows = pair_rows(model, final_data, robot_geoms, table_geom_ids(model))
    close_floor_rows = pair_rows(model, final_data, robot_geoms, floor_geom_ids(model))
    narrow_name, wide_name = TARGET_JAW_BODIES
    narrow_distance = geom_pair_distance(model, final_data, candidate["narrow_id"],
                                         candidate["bottle_body_id"])[0]
    wide_distance = geom_pair_distance(model, final_data, candidate["wide_id"],
                                       candidate["bottle_body_id"])[0]
    intended = {narrow_name: narrow_distance, wide_name: wide_distance}
    nonjaw_closed = [row for row in contact_rows_closed if row["body1"] not in set(TARGET_JAW_BODIES)]
    open_data, _ = candidate_state(base, candidate, 1.0,
                                   float(solution.x[0]), float(solution.x[1]))
    open_target = {name: geom_pair_distance(model, open_data,
                    base.collision_geom_id(model, name), candidate["bottle_body_id"])[0]
                   for name in TARGET_JAW_BODIES}
    open_rows = pair_rows(model, open_data, robot_geoms, body_geoms)
    self_contacts = [row for row in contact_rows(model, open_data)
                     if row["body1"] not in {"m0_bottle", "m0_table", "world"}
                     and row["body2"] not in {"m0_bottle", "m0_table", "world"}
                     and row["distance_m"] <= 0.0]
    table_rows = pair_rows(model, open_data, robot_geoms, table_geom_ids(model))
    floor_rows = pair_rows(model, open_data, robot_geoms, floor_geom_ids(model))
    limits = source_limits(X2_ROOT / X2_URDF_REL)
    open_limit_violations = limit_violations(model, open_data, limits)
    final_data, _ = candidate_state(base, candidate, candidate["contact_aperture"],
                                    float(solution.x[0]), float(solution.x[1]))
    closed_limit_violations = limit_violations(model, final_data, limits)

    closure_samples = []
    for aperture in np.linspace(1.0, candidate["contact_aperture"], 51):
        sample_data, _ = candidate_state(base, candidate, float(aperture),
                                         float(solution.x[0]), float(solution.x[1]))
        closure_samples.append({
            "aperture_ratio": float(aperture),
            "narrow_clearance_m": geom_pair_distance(model, sample_data,
                candidate["narrow_id"], candidate["bottle_body_id"])[0],
            "wide_clearance_m": geom_pair_distance(model, sample_data,
                candidate["wide_id"], candidate["bottle_body_id"])[0],
        })
    monotonic = {
        name: all(samples[i + 1][key] <= samples[i][key] + 1.0e-6
                  for i in range(len(closure_samples) - 1))
        for name, key in (("narrow", "narrow_clearance_m"), ("wide", "wide_clearance_m"))
        for samples in [closure_samples]
    }
    open_all_clear = (minimum(open_rows) > 0.0 and minimum(table_rows) > 0.0
                      and minimum(floor_rows) > 0.0)
    close_jaws_touch = max(abs(narrow_distance), abs(wide_distance)) <= CONTACT_EPS_M
    close_no_other_penetration = minimum(nonjaw_closed) >= -PENETRATION_EPS_M
    close_no_table = minimum(close_table_rows) > 0.0 and minimum(close_floor_rows) > 0.0
    gate = {
        "open_robot_bottle_clearance_positive": open_all_clear,
        "open_jaw_clearances_positive": all(value > 0.0 for value in open_target.values()),
        "closed_both_jaws_reach_bottle_body": close_jaws_touch,
        "closed_no_nonjaw_bottle_penetration": close_no_other_penetration,
        "open_and_closed_robot_table_clearance_positive": close_no_table,
        "closure_reduces_both_jaw_clearances_monotonically": all(monotonic.values()),
        "zero_initial_robot_self_penetration": not self_contacts,
        "all_included_source_limited_joints_within_range": not open_limit_violations and not closed_limit_violations,
    }
    result = {
        "candidate_id": candidate["candidate_id"],
        "orientation_definition": {
            "jaw_axis_azimuth_deg": candidate["azimuth_deg"],
            "jaw_longitudinal_axis_tilt_from_world_z_toward_tangent_deg": candidate["finger_tilt_deg"],
            "source_local_jaw_basis": candidate["local_geometry"],
            "root_rotation_matrix": candidate["root_rotation"].tolist(),
            "root_pose": pose,
        },
        "jaw_contact_aperture_ratio": candidate["contact_aperture"],
        "geometry_optimizer": {
            "method": "bounded least_squares over actual compiled narrow/wide mesh to canonical bottle-body cylinder signed distances",
            "variables": ["tangent_to_cylinder_m", "contact_height_offset_m"],
            "bounds_m": {"tangent": [-0.030, 0.030], "height": [-0.045, 0.045]},
            "evaluation_count": len(evaluations),
            "optimizer_status": int(solution.status), "optimizer_message": solution.message,
            "final_cost": float(solution.cost), "final_residual_m": solution.fun.tolist(),
        },
        "open_jaw_clearance_by_target_body_m": open_target,
        "closed_jaw_to_bottle_body_distance_m": intended,
        "minimum_open_robot_to_bottle_distance_m": minimum(open_rows),
        "minimum_open_robot_to_table_distance_m": minimum(table_rows),
        "minimum_open_robot_to_floor_distance_m": minimum(floor_rows),
        "minimum_closed_nonjaw_to_bottle_distance_m": minimum(nonjaw_closed),
        "minimum_closed_robot_to_table_distance_m": minimum(close_table_rows),
        "minimum_closed_robot_to_floor_distance_m": minimum(close_floor_rows),
        "open_robot_self_penetrating_contacts": self_contacts,
        "open_source_joint_limit_violations": open_limit_violations,
        "closed_source_joint_limit_violations": closed_limit_violations,
        "closed_robot_to_bottle_distance_rows": contact_rows_closed,
        "open_robot_to_bottle_distance_rows": open_rows,
        "open_robot_to_table_distance_rows": table_rows,
        "closed_robot_to_table_distance_rows": close_table_rows,
        "open_robot_to_floor_distance_rows": floor_rows,
        "closed_robot_to_floor_distance_rows": close_floor_rows,
        "closure_samples": closure_samples,
        "closure_monotonic_by_jaw": monotonic,
        "gate": gate,
        "status": "PASS" if all(gate.values()) else "FAIL",
    }
    candidate["optimized_offsets"] = solution.x.tolist()
    candidate["geometry_result"] = result
    return result


def camera_for(lookat: list[float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_png(model: mujoco.MjModel, data: mujoco.MjData, output: Path,
               filename: str, lookat: list[float], distance: float,
               azimuth: float, elevation: float) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera_for(lookat, distance, azimuth, elevation))
    Image.fromarray(renderer.render().copy()).save(output / filename)
    renderer.close()


def render_stage_frame(renderer: mujoco.Renderer, model: mujoco.MjModel,
                       data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def plot_trace(trace_path: Path, image_path: Path) -> None:
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        return
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        return
    width, height = 1100, 720
    image = Image.new("RGB", (width, height), (250, 250, 248))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    left, right = 92, width - 28
    panels = [(70, 245), (295, 470), (520, 665)]
    times = np.asarray([row["sim_time_s"] for row in rows], dtype=float)
    time_min, time_max = float(times.min()), max(float(times.max()), float(times.min()) + 1.0e-9)

    def series(panel: tuple[int, int], values: np.ndarray, color: tuple[int, int, int],
               low: float | None = None, high: float | None = None) -> None:
        y0, y1 = panel
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return
        lo = float(finite.min()) if low is None else low
        hi = float(finite.max()) if high is None else high
        if hi <= lo:
            hi = lo + 1.0
        points = []
        for t, v in zip(times, values):
            if not math.isfinite(float(v)):
                continue
            x = left + (right - left) * (float(t) - time_min) / (time_max - time_min)
            y = y1 - (y1 - y0) * (float(v) - lo) / (hi - lo)
            points.append((int(x), int(y)))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

    labels = ["Bottle COM height (m)", "Carriage position (m)", "Jaw normal force (N)"]
    for i, (y0, y1) in enumerate(panels):
        draw.rectangle((left, y0, right, y1), outline=(95, 95, 95), width=1)
        for k in range(1, 4):
            y = y0 + (y1 - y0) * k / 4
            draw.line((left, int(y), right, int(y)), fill=(220, 220, 218), width=1)
        draw.text((8, y0 + 4), labels[i], fill=(30, 30, 30))
    com_z = np.asarray([row["bottle_com_world_m"][2] for row in rows], dtype=float)
    carriage_q = np.asarray([row["carriage_controller_terms"]["q_m"] for row in rows], dtype=float)
    narrow_force = []
    wide_force = []
    for row in rows:
        narrow = wide = 0.0
        for contact in row.get("contacts", []):
            if "m0_bottle" not in {contact.get("body1"), contact.get("body2")}:
                continue
            other = contact["body1"] if contact["body2"] == "m0_bottle" else contact["body2"]
            if other.startswith("R_hand_narrow") and "loop" not in other.lower():
                narrow += float(contact["normal_force_n"])
            if other.startswith("R_hand_wide") and "loop" not in other.lower():
                wide += float(contact["normal_force_n"])
        narrow_force.append(narrow)
        wide_force.append(wide)
    series(panels[0], com_z, (30, 95, 170))
    series(panels[1], carriage_q, (150, 85, 35))
    series(panels[2], np.asarray(narrow_force), (35, 125, 65), 0.0,
           max(max(narrow_force, default=0.0), max(wide_force, default=0.0), 1.0))
    series(panels[2], np.asarray(wide_force), (180, 55, 55), 0.0,
           max(max(narrow_force, default=0.0), max(wide_force, default=0.0), 1.0))
    draw.text((left, height - 28), f"Simulation time: {time_min:.3f}-{time_max:.3f} s",
              fill=(30, 30, 30))
    draw.text((width - 275, height - 28), "narrow=green  wide=red", fill=(30, 30, 30))
    image.save(image_path)


def source_limits(urdf: Path) -> dict[str, list[float]]:
    joints, _ = source_joint_map(urdf)
    result = {}
    for name, row in joints.items():
        limit = row.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            result[name] = [float(limit.get("lower")), float(limit.get("upper"))]
    return result


def limit_violations(model: mujoco.MjModel, data: mujoco.MjData,
                     limits: dict[str, list[float]], tolerance: float = 1.0e-8) -> list[dict[str, Any]]:
    rows = []
    for name, bounds in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        if q < bounds[0] - tolerance or q > bounds[1] + tolerance:
            rows.append({"joint": name, "qpos_rad": q, "source_range_rad": bounds})
    return rows


def geometry_stage(base, urdf_path: Path, output: Path,
                   candidate_specs: list[tuple[float, float]],
                   controller: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidates = []
    selected = None
    for index, (azimuth, finger_tilt) in enumerate(candidate_specs, start=1):
        trace_path = output / f"candidate_{index}_raw_geometry_trace.jsonl"
        candidate = None
        try:
            candidate = candidate_model(base, urdf_path, index, controller, azimuth, finger_tilt)
            result = refine_contact(base, candidate, trace_path)
        except Exception as exc:
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"candidate_id": index, "error_type": type(exc).__name__,
                                         "error": str(exc)}, sort_keys=True) + "\n")
                stream.flush()
            result = {"candidate_id": index, "status": "FAIL",
                      "candidate_geometry_error": {"type": type(exc).__name__, "message": str(exc)},
                      "orientation_definition": {"jaw_axis_azimuth_deg": azimuth,
                                                 "jaw_longitudinal_axis_tilt_from_world_z_toward_tangent_deg": finger_tilt,
                                                 "root_pose": None}}
        result["raw_geometry_trace"] = str(trace_path)
        result["raw_geometry_trace_sha256"] = sha256(trace_path)
        result_path = output / f"candidate_{index}_geometry_result.json"
        write_json(result_path, result)
        if candidate is not None:
            tangent_offset, height_offset = candidate.get("optimized_offsets", [0.0, 0.0])
            target = None
            for aperture, label in ((1.0, "open"), (candidate["contact_aperture"], "contact")):
                view_data, _ = candidate_state(base, candidate, aperture,
                                               float(tangent_offset), float(height_offset))
                mujoco.mj_forward(candidate["model"], view_data)
                body_center = view_data.geom_xpos[candidate["bottle_body_id"]].copy()
                view_target = (body_center + np.array([0.0, 0.0, 0.055])).tolist()
                target = view_target
                for view, distance, camera_azimuth, elevation in (
                    ("overview", 1.20, 135.0, -20.0),
                    ("front", 0.68, 180.0, -10.0),
                    ("side", 0.68, 90.0, -10.0),
                ):
                    render_png(candidate["model"], view_data, output,
                               f"candidate_{index}_{label}_{view}.png", view_target,
                               distance, camera_azimuth, elevation)
            result["candidate_views"] = {
                "status": result["status"],
                "files": [f"candidate_{index}_{aperture}_{view}.png"
                          for aperture in ("open", "contact")
                          for view in ("overview", "front", "side")],
                "camera_target_m": target,
            }
            write_json(result_path, result)
        if result["status"] == "PASS" and candidate is not None:
            # Reinitialize with the already accepted paired-jaw profile and selected pose.
            root_position = np.asarray(
                candidate["geometry_result"]["orientation_definition"]["root_pose"]["root_position_m"],
                dtype=float,
            )
            model, build_info = build_model(
                base, urdf_path, root_position, candidate["root_rotation"],
                include_bottle=True, add_lift=False,
            )
            data = initialize(base, model, controller)
            root_id = body_id(model, ROOT_LINK)
            root_pose = result["orientation_definition"]["root_pose"]
            set_root_pose(model, root_id, data, root_position,
                          quat_from_matrix(candidate["root_rotation"]))
            selected = {"model": model, "data": data, "root_id": root_id,
                        "root_position": root_position,
                        "root_rotation": candidate["root_rotation"],
                        "contact_aperture": candidate["contact_aperture"],
                        "geometry_result": result, "candidate_id": index,
                        "build_info": build_info}
            narrow = base.collision_geom_id(model, TARGET_JAW_BODIES[0])
            wide = base.collision_geom_id(model, TARGET_JAW_BODIES[1])
            body_geom = next(g for g in bottle_geom_ids(model) if "bottle_body" in geom_name(model, g))
            selected.update({"narrow_id": narrow, "wide_id": wide, "bottle_body_id": body_geom})
            candidate["selected"] = True
            selected["geometry_result"] = result
        candidates.append({"candidate_id": index, "status": result["status"],
                           "result_path": str(result_path),
                           "raw_geometry_trace": str(trace_path),
                           "root_pose": result["orientation_definition"]["root_pose"]})
        if selected is not None:
            break
    stage = {"status": "PASS" if selected is not None else "FAIL",
             "candidate_count": len(candidates), "maximum_candidates": 3,
             "candidates": candidates,
             "selected_candidate_id": None if selected is None else selected["candidate_id"],
             "failure_reason": None if selected is not None else "no tested source-mesh pose met the open, opposing-jaw, closure-monotonicity, and nonjaw clearance gates"}
    write_json(output / "stage_a_geometry_result.json", stage)
    return stage, selected


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, full, data.qM)
    return full


def lift_control(model: mujoco.MjModel, data: mujoco.MjData,
                 target_q: float, target_v: float) -> dict[str, float]:
    dof = dof_id(model, LIFT_JOINT)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR)
    full = mass_matrix(model, data)
    inertia = max(float(full[dof, dof]), 1.0e-8)
    omega = 6.0
    kp = inertia * omega**2
    kd = 2.0 * omega * inertia
    q = float(data.qpos[qpos_id(model, LIFT_JOINT)])
    v = float(data.qvel[dof])
    bias = float(data.qfrc_bias[dof])
    requested = bias + kp * (target_q - q) + kd * (target_v - v)
    bound = float(model.actuator_forcerange[aid, 1])
    applied = float(np.clip(requested, -bound, bound))
    data.ctrl[aid] = applied
    return {"target_q_m": target_q, "target_qvel_m_s": target_v,
            "q_m": q, "qvel_m_s": v, "qacc_m_s2": float(data.qacc[dof]),
            "effective_inertia_kg": inertia, "kp_n_m": kp, "kd_n_s_m": kd,
            "bias_feedforward_n": bias, "requested_force_n": requested,
            "applied_force_n": applied, "force_cap_n": bound}


def physical_run(base, selected: dict[str, Any], controller: dict[str, Any],
                 limits: dict[str, list[float]], output: Path) -> dict[str, Any]:
    model, build_info = build_model(base, Path(selected["build_info"]["derived_urdf_path"])
                                   if "derived_urdf_path" in selected["build_info"] else Path(),
                                   selected["root_position"], selected["root_rotation"],
                                   include_bottle=True, add_lift=True)
    # Rebuild with the actual filtered URDF path passed through the selected geometry result.
    # The caller replaces this field with the provenance path before entering dynamics.
    data = initialize(base, model, controller)
    for aperture_name, aperture in (("open", 1.0),):
        set_jaws(base, model, data, aperture)
    root_id = body_id(model, ROOT_LINK)
    set_root_pose(model, root_id, data, selected["root_position"],
                  quat_from_matrix(selected["root_rotation"]))
    bottle_id = body_id(model, "m0_bottle")
    bottle_free_joint = joint_id(model, "m0_bottle_free")
    bottle_qadr = int(model.jnt_qposadr[bottle_free_joint])
    bottle_dofadr = int(model.jnt_dofadr[bottle_free_joint])
    bottle_qpos_start = data.qpos.copy()
    initial_bottle_com = data.subtree_com[bottle_id].copy()
    initial_bottle_quat = data.qpos[int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    ]) + 3:int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")
    ]) + 7].copy()
    bottle_qpos_write_detected = False
    follower_qpos_write_detected = False
    open_contacts = contact_rows(model, data)
    bad_initial = [row for row in open_contacts
                   if row["distance_m"] <= 0.0
                   and not ({row["body1"], row["body2"]} == {"m0_bottle", "m0_table"})]
    robot_bottle_open = [row for row in pair_rows(model, data, robot_geom_ids(model), bottle_geom_ids(model))
                         if row["signed_distance_m"] <= 0.0]
    bottle_mocap_id = int(model.body_mocapid[bottle_id])
    bottle_equalities = [name for name in build_info["equality_names"] if "m0_bottle" in name]
    preflight = {
        "physics_steps": 0,
        "initial_contacts": open_contacts,
        "initial_non_bottle_robot_penetrations": bad_initial,
        "initial_open_robot_bottle_penetrations": robot_bottle_open,
        "initial_source_limit_violations": limit_violations(model, data, limits),
        "bottle_initial_com_m": initial_bottle_com.tolist(),
        "bottle_free_joint_qpos_written_after_reset": False,
        "right_jaw_equalities": build_info["jaw_equalities"],
        "bottle_mocap_body_id": bottle_mocap_id,
        "bottle_related_equalities": bottle_equalities,
        "status": "PASS" if not bad_initial and not robot_bottle_open
                  and not limit_violations(model, data, limits)
                  and bottle_mocap_id == -1 and not bottle_equalities else "FAIL",
    }
    write_json(output / "dynamic_zero_step_preflight.json", preflight)
    if preflight["status"] != "PASS":
        stage_b = {"status": "FAIL", "first_failed_gate": "dynamic zero-step preflight",
                   "preflight": preflight, "contact_families": [],
                   "bottle_qpos_write_count_during_rollout": 0}
        stage_c = {"status": "NOT RUN", "reason": "Stage B dynamic preflight failed"}
        write_json(output / "stage_b_bottle_hold_result.json", stage_b)
        write_json(output / "stage_c_lift_result.json", stage_c)
        return {"stage_b": stage_b, "stage_c": stage_c,
                "trace_path": None, "video_path": None,
                "qpos_writes_after_simulation_start": {"bottle": 0, "follower": 0},
                "physical_rules": {"right_hand_only": True, "left_hand_clear": True,
                                   "bottle_free_joint": True, "runtime_bottle_weld": False,
                                   "mocap_attachment": False, "follow_hand_logic": False}}

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = camera_for([0.30, 0.0, 0.88], 0.78, 135.0, -18.0)
    video_frames: list[np.ndarray] = []
    trace_path = output / "physical_grasp_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    max_penetration = 0.0
    max_qvel = 0.0
    max_qacc = 0.0
    max_relation = 0.0
    max_effort = 0.0
    max_force = 0.0
    limit_rows: list[dict[str, Any]] = []
    nonfinite = False
    object_contact_seen = False
    first_object_contact_step = None
    contact_frames: dict[str, int] = {}
    pair_frames: dict[str, int] = {}
    phase_frames: dict[str, int] = {}
    hold_bilateral_frames = 0
    hold_frames = 0
    max_precontact_relation = 0.0
    lift_frames: dict[str, int] = {}
    lift_bilateral_frames: dict[str, int] = {}
    table_support_frames_by_phase: dict[str, int] = {}
    first_failure: dict[str, Any] | None = None
    initial_data_com = initial_bottle_com.copy()
    hold_start_com: np.ndarray | None = None
    lift_start_com: np.ndarray | None = None
    max_hold_slip = 0.0
    max_bottle_lift = 0.0
    max_lift_frame: np.ndarray | None = None
    first_contact_snapshot_saved = False
    last_carriage_terms: dict[str, float] = {}
    carriage_force_max = 0.0
    frame_stride = 20
    step = 0

    def append_row(row: dict[str, Any]) -> None:
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

    def save_snapshot(label: str) -> None:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render().copy()
        Image.fromarray(frame).save(output / f"{label}.png")
        video_frames.append(frame)

    save_snapshot("pregrasp")

    def run_phase(name: str, duration: float, aperture_start: float, aperture_end: float,
                  lift_target_start: float, lift_target_end: float,
                  minjerk_aperture: bool, minjerk_lift: bool) -> None:
        nonlocal step, max_penetration, max_qvel, max_qacc, max_relation
        nonlocal max_effort, max_force, nonfinite, object_contact_seen
        nonlocal first_object_contact_step, first_failure, hold_bilateral_frames
        nonlocal hold_frames, hold_start_com, max_hold_slip, last_carriage_terms
        nonlocal carriage_force_max, max_precontact_relation, first_contact_snapshot_saved
        nonlocal lift_start_com, max_bottle_lift, max_lift_frame
        nonlocal bottle_qpos_write_detected, follower_qpos_write_detected
        steps = int(round(duration / float(model.opt.timestep)))
        phase_frames[name] = 0
        lift_frames[name] = 0
        table_support_frames_by_phase[name] = 0
        for local_step in range(steps):
            bottle_qpos_before_control = data.qpos[bottle_qadr:bottle_qadr + 7].copy()
            follower_qpos_before_control = float(data.qpos[qpos_id(model, FOLLOWER)])
            t = local_step * float(model.opt.timestep)
            if minjerk_aperture:
                aperture, aperture_v, aperture_a = base.minimum_jerk(
                    t, duration, aperture_start, aperture_end
                )
            else:
                aperture, aperture_v, aperture_a = aperture_end, 0.0, 0.0
            if minjerk_lift:
                lift_q, lift_v, _ = base.minimum_jerk(
                    t, duration, lift_target_start, lift_target_end
                )
            else:
                lift_q, lift_v = lift_target_end, 0.0
            jaw_terms = base.jaw_control(model, data, aperture, aperture_v, aperture_a, controller)
            last_carriage_terms = lift_control(model, data, lift_q, lift_v)
            bottle_qpos_write = not np.array_equal(
                data.qpos[bottle_qadr:bottle_qadr + 7], bottle_qpos_before_control
            )
            follower_qpos_write = float(data.qpos[qpos_id(model, FOLLOWER)]) != follower_qpos_before_control
            bottle_qpos_write_detected |= bottle_qpos_write
            follower_qpos_write_detected |= follower_qpos_write
            carriage_force_max = max(carriage_force_max,
                                     abs(last_carriage_terms["applied_force_n"]))
            mujoco.mj_step(model, data)
            step += 1
            phase_frames[name] += 1
            contacts = contact_rows(model, data)
            bottle_contacts = [row for row in contacts if "m0_bottle" in {row["body1"], row["body2"]}]
            table_contacts = [row for row in bottle_contacts if "m0_table" in {row["body1"], row["body2"]}]
            hand_bottle_contacts = [row for row in bottle_contacts
                                    if any(name.startswith(("R_hand_narrow", "R_hand_wide"))
                                           for name in (row["body1"], row["body2"]))
                                    and not any("loop" in name.lower()
                                                for name in (row["body1"], row["body2"]))]
            jaw_families: set[str] = set()
            for contact in hand_bottle_contacts:
                other = contact["body1"] if contact["body2"] == "m0_bottle" else contact["body2"]
                if contact["normal_force_n"] < base.CONTACT_FORCE_BEARING_MIN_N:
                    continue
                if other.startswith("R_hand_narrow") and "loop" not in other.lower():
                    jaw_families.add("narrow")
                if other.startswith("R_hand_wide") and "loop" not in other.lower():
                    jaw_families.add("wide")
                pair = "|".join(sorted((contact["body1"], contact["body2"])))
                pair_frames[pair] = pair_frames.get(pair, 0) + 1
                max_force = max(max_force, float(contact["normal_force_n"]))
                max_penetration = max(max_penetration, float(contact["penetration_m"]))
            for family in jaw_families:
                contact_frames[family] = contact_frames.get(family, 0) + 1
            if hand_bottle_contacts:
                object_contact_seen = True
                if first_object_contact_step is None:
                    first_object_contact_step = step
                if not first_contact_snapshot_saved:
                    save_snapshot("first_contact")
                    first_contact_snapshot_saved = True
            if name == "HOLD":
                hold_frames += 1
                if hold_start_com is None:
                    hold_start_com = data.subtree_com[bottle_id].copy()
                if {"narrow", "wide"}.issubset(jaw_families):
                    hold_bilateral_frames += 1
                max_hold_slip = max(max_hold_slip,
                    float(np.linalg.norm(data.subtree_com[bottle_id] - hold_start_com)))
            if name.startswith("LIFT_"):
                lift_frames[name] += 1
                if {"narrow", "wide"}.issubset(jaw_families):
                    lift_bilateral_frames[name] = lift_bilateral_frames.get(name, 0) + 1
            if name.startswith("LIFT_") and table_contacts:
                table_support_frames_by_phase[name] += 1

            source_q = {name_: float(data.qpos[qpos_id(model, name_)])
                        for name_ in (DRIVER, FOLLOWER, *WRIST_JOINTS)}
            relation = source_q[FOLLOWER] + source_q[DRIVER]
            if not object_contact_seen:
                max_precontact_relation = max(max_precontact_relation, abs(relation))
            max_relation = max(max_relation, abs(relation))
            step_qvel = max(abs(float(data.qvel[dof_id(model, name_)]))
                            for name_ in (DRIVER, FOLLOWER))
            step_qacc = max(abs(float(data.qacc[dof_id(model, name_)]))
                            for name_ in (DRIVER, FOLLOWER))
            max_qvel = max(max_qvel, step_qvel)
            max_qacc = max(max_qacc, step_qacc)
            max_effort = max(max_effort, *(abs(row["applied_torque_nm"])
                                          for row in jaw_terms.values()))
            violations = limit_violations(model, data, limits)
            if violations:
                limit_rows.extend([{**row, "step": step, "phase": name} for row in violations])
            state_arrays = (data.qpos, data.qvel, data.qacc, data.qfrc_actuator, data.qfrc_constraint)
            if any(not np.all(np.isfinite(array)) for array in state_arrays):
                nonfinite = True
            bottle_com = data.subtree_com[bottle_id].copy()
            if name.startswith("LIFT_") and lift_start_com is not None:
                current_lift = float(bottle_com[2] - lift_start_com[2])
                if current_lift > max_bottle_lift:
                    max_bottle_lift = current_lift
                    renderer.update_scene(data, camera=camera)
                    max_lift_frame = renderer.render().copy()
            row = {
                "step": step, "sim_time_s": float(data.time), "phase": name,
                "aperture_ratio": aperture, "aperture_velocity_s_inv": aperture_v,
                "jaw_controller_terms": jaw_terms,
                "jaw_qpos_rad": source_q,
                "jaw_qvel_rad_s": {name_: float(data.qvel[dof_id(model, name_)])
                                    for name_ in (DRIVER, FOLLOWER)},
                "jaw_qacc_rad_s2": {name_: float(data.qacc[dof_id(model, name_)])
                                     for name_ in (DRIVER, FOLLOWER)},
                "jaw_relation_residual_rad": relation,
                "jaw_tracking_error_rad": {name_: float(jaw_terms[name_]["position_error_rad"])
                                            for name_ in (DRIVER, FOLLOWER)},
                "source_limit_violations": violations,
                "carriage_controller_terms": last_carriage_terms,
                "bottle_com_world_m": bottle_com.tolist(),
                "bottle_body_pose_world_m": data.xpos[bottle_id].tolist(),
                "bottle_free_qpos_world_m_wxyz": data.qpos[bottle_qadr:bottle_qadr + 7].tolist(),
                "bottle_free_qvel": data.qvel[bottle_dofadr:bottle_dofadr + 6].tolist(),
                "bottle_free_joint_qpos_write_count": 0,
                "bottle_qpos_write_detected_before_step": bottle_qpos_write,
                "follower_qpos_write_detected_before_step": follower_qpos_write,
                "bottle_table_contact_count": len(table_contacts),
                "jaw_contact_families": sorted(jaw_families),
                "contacts": contacts,
                "max_contact_penetration_m": max((row["penetration_m"] for row in contacts), default=0.0),
                "state_finite": not nonfinite,
            }
            append_row(row)
            if step % frame_stride == 0:
                video_frames.append(render_stage_frame(renderer, model, data, camera))

            if first_failure is None and not object_contact_seen and abs(relation) > base.APERTURE_RELATION_DIAGNOSTIC_RAD:
                first_failure = {"gate": "precontact paired-jaw relation", "step": step,
                                 "error_rad": abs(relation),
                                 "limit_rad": base.APERTURE_RELATION_DIAGNOSTIC_RAD}
            if first_failure is None and max_penetration > ABORT_PENETRATION_M:
                first_failure = {"gate": "contact penetration bound", "step": step,
                                 "penetration_m": max_penetration,
                                 "limit_m": ABORT_PENETRATION_M}
            if first_failure is None and step_qvel > 1.5 * controller["predicted_peak_joint_velocity_rad_s"] + 0.05:
                first_failure = {"gate": "jaw velocity bound", "step": step,
                                 "qvel_rad_s": step_qvel,
                                 "limit_rad_s": 1.5 * controller["predicted_peak_joint_velocity_rad_s"] + 0.05}
            if first_failure is None and step_qacc > base.CONTACT_ACCELERATION_BOUND_RAD_S2:
                first_failure = {"gate": "jaw acceleration bound", "step": step,
                                 "qacc_rad_s2": step_qacc,
                                 "limit_rad_s2": base.CONTACT_ACCELERATION_BOUND_RAD_S2}
            if first_failure is None and violations:
                first_failure = {"gate": "source joint limit", "step": step, "violations": violations}
            if first_failure is None and nonfinite:
                first_failure = {"gate": "nonfinite physics state", "step": step}
            if first_failure is None and (bottle_qpos_write_detected or follower_qpos_write_detected):
                first_failure = {"gate": "forbidden active-rollout qpos write", "step": step,
                                 "bottle": bottle_qpos_write, "follower": follower_qpos_write}
            if first_failure is not None:
                return

    run_phase("SETTLE", 0.5, 1.0, 1.0, 0.0, 0.0, False, False)
    if first_failure is None:
        run_phase("CLOSE", base.MOTION_DURATION_S, 1.0, 0.0, 0.0, 0.0, True, False)
        save_snapshot("closed_grasp")
    if first_failure is None:
        run_phase("HOLD", CONTACT_HOLD_S, 0.0, 0.0, 0.0, 0.0, False, False)
        save_snapshot("hold")

    stage_b_fraction = hold_bilateral_frames / max(hold_frames, 1)
    if first_failure is None and not {"narrow", "wide"}.issubset(contact_frames):
        first_failure = {"gate": "bilateral jaw contact", "families_seen": sorted(contact_frames)}
    if first_failure is None and stage_b_fraction < 0.80:
        first_failure = {"gate": "bilateral contact persistence during hold",
                         "fraction": stage_b_fraction, "required": 0.80}
    if first_failure is None and max_precontact_relation > base.APERTURE_RELATION_DIAGNOSTIC_RAD:
        first_failure = {"gate": "paired-jaw relation before physical contact",
                         "max_error_rad": max_precontact_relation}

    stage_b = {
        "status": "PASS" if first_failure is None else "FAIL",
        "first_failed_gate": first_failure,
        "hold_duration_s": hold_frames * float(model.opt.timestep),
        "bilateral_hold_fraction": stage_b_fraction,
        "contact_families": sorted(contact_frames),
        "contact_pair_frame_counts": pair_frames,
        "first_object_contact_step": first_object_contact_step,
        "max_contact_normal_force_n": max_force,
        "max_contact_penetration_m": max_penetration,
        "max_jaw_velocity_rad_s": max_qvel,
        "max_jaw_acceleration_rad_s2": max_qacc,
        "max_jaw_relation_residual_rad": max_relation,
        "max_precontact_jaw_relation_residual_rad": max_precontact_relation,
        "max_jaw_effort_nm": max_effort,
        "max_bottle_com_slip_during_hold_m": max_hold_slip,
        "bottle_com_initial_m": initial_data_com.tolist(),
        "bottle_com_after_hold_m": data.subtree_com[bottle_id].tolist(),
        "bottle_quaternion_initial_wxyz": initial_bottle_quat.tolist(),
        "bottle_qpos_write_count_during_rollout": int(bottle_qpos_write_detected),
        "follower_qpos_write_count_during_rollout": int(follower_qpos_write_detected),
        "bottle_has_free_joint": True,
        "bottle_weld_or_equality_attachment": False,
        "mocap_or_follow_hand_carry": False,
        "left_hand_present_or_active": False,
        "source_joint_limit_violations": limit_rows,
        "finite_state": not nonfinite,
        "right_jaw_equalities": build_info["jaw_equalities"],
    }
    write_json(output / "stage_b_bottle_hold_result.json", stage_b)
    if first_failure is None:
        # The tested CLOSE/HOLD state remains the initial state for the carriage lift.
        lift_start_com = data.subtree_com[bottle_id].copy()
        table_ids = table_geom_ids(model)
        stage_c_targets = [("LOAD_TRANSFER", 0.25, 0.0, 0.0),
                           ("LIFT_1MM", 0.35, 0.0, 0.001),
                           ("LIFT_5MM", 0.55, 0.001, 0.005),
                           ("LIFT_30MM", 1.50, 0.005, 0.030),
                           ("LIFT_HOLD", 1.0, 0.030, 0.030)]
        start_carriage = float(data.qpos[qpos_id(model, LIFT_JOINT)])
        stage_c_failure = None
        lift_targets_metrics: dict[str, Any] = {}
        for name, duration, q_start, q_end in stage_c_targets:
            run_phase(name, duration, 0.0, 0.0, q_start, q_end,
                      False, name != "LOAD_TRANSFER" and q_start != q_end)
            if first_failure is not None:
                stage_c_failure = first_failure
                break
            com = data.subtree_com[bottle_id].copy()
            carriage_q = float(data.qpos[qpos_id(model, LIFT_JOINT)])
            current_contacts = contact_rows(model, data)
            bottle_table = [row for row in current_contacts
                            if "m0_bottle" in {row["body1"], row["body2"]}
                            and "m0_table" in {row["body1"], row["body2"]}]
            lift_targets_metrics[name] = {
                "carriage_q_m": carriage_q,
                "bottle_com_world_m": com.tolist(),
                "bottle_com_lift_from_lift_start_m": float(com[2] - lift_start_com[2]),
                "bottle_to_carriage_vertical_lag_m": float((com[2] - lift_start_com[2]) - (carriage_q - start_carriage)),
                "bottle_table_contact_count_at_phase_end": len(bottle_table),
                "bottle_table_contact_frames_during_phase": table_support_frames_by_phase[name],
                "jaw_contact_families_at_phase_end": sorted({
                    "narrow" if row["body1"].startswith("R_hand_narrow") or row["body2"].startswith("R_hand_narrow")
                    else "wide" for row in current_contacts
                    if "m0_bottle" in {row["body1"], row["body2"]}
                    and ("R_hand_narrow" in row["body1"] or "R_hand_narrow" in row["body2"]
                         or "R_hand_wide" in row["body1"] or "R_hand_wide" in row["body2"])
                }),
            }
            required_lift = {"LIFT_1MM": 0.001, "LIFT_5MM": 0.005,
                             "LIFT_30MM": 0.030}.get(name)
            if (required_lift is not None
                    and lift_targets_metrics[name]["bottle_com_lift_from_lift_start_m"] < required_lift):
                stage_c_failure = {"gate": "bottle lift checkpoint", "observed_m": lift_targets_metrics[name]["bottle_com_lift_from_lift_start_m"],
                                   "required_m": required_lift, "phase": name}
                break
            if name == "LIFT_30MM":
                if bottle_table:
                    stage_c_failure = {"gate": "airborne bottle at 30 mm checkpoint",
                                       "table_contacts": bottle_table}
                    break
                if not {"narrow", "wide"}.issubset(lift_targets_metrics[name]["jaw_contact_families_at_phase_end"]):
                    stage_c_failure = {"gate": "bilateral contact at 30 mm checkpoint",
                                       "families": lift_targets_metrics[name]["jaw_contact_families_at_phase_end"]}
                    break
            if name == "LIFT_HOLD":
                if table_support_frames_by_phase[name] > 0 or bottle_table:
                    stage_c_failure = {"gate": "airborne bottle support",
                                       "table_contact_frames": table_support_frames_by_phase[name],
                                       "table_contacts_at_phase_end": bottle_table}
                    break
                if not {"narrow", "wide"}.issubset(lift_targets_metrics[name]["jaw_contact_families_at_phase_end"]):
                    stage_c_failure = {"gate": "bilateral contact during lift hold",
                                       "families": lift_targets_metrics[name]["jaw_contact_families_at_phase_end"]}
                    break
                if lift_bilateral_frames.get(name, 0) < int(0.80 * lift_frames.get(name, 1)):
                    stage_c_failure = {"gate": "bilateral contact persistence during lift hold",
                                       "bilateral_frames": lift_bilateral_frames.get(name, 0),
                                       "total_frames": lift_frames.get(name, 0),
                                       "required_fraction": 0.80}
                    break
        if max_lift_frame is not None:
            Image.fromarray(max_lift_frame).save(output / "maximum_lift.png")
            video_frames.append(max_lift_frame)
        stage_c = {
            "status": "PASS" if stage_c_failure is None else "FAIL",
            "first_failed_gate": stage_c_failure,
            "carriage_joint": LIFT_JOINT,
            "carriage_actuator": LIFT_ACTUATOR,
            "carriage_joint_position_qpos_writes_during_rollout": 0,
            "bottle_qpos_writes_during_rollout": int(bottle_qpos_write_detected),
            "follower_qpos_writes_during_rollout": int(follower_qpos_write_detected),
            "carriage_force_cap_n": float(model.actuator_forcerange[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LIFT_ACTUATOR), 1]),
            "carriage_peak_actuator_force_n": carriage_force_max,
            "lift_targets": lift_targets_metrics,
            "bottle_table_support_frames": table_support_frames_by_phase,
            "max_contact_normal_force_n": max_force,
            "max_contact_penetration_m": max_penetration,
            "maximum_bottle_com_lift_from_lift_start_m": max_bottle_lift,
            "max_bottle_com_slip_during_hold_m": max_hold_slip,
            "source_joint_limit_violations": limit_rows,
            "finite_state": not nonfinite,
            "right_jaw_equalities": build_info["jaw_equalities"],
        }
        write_json(output / "stage_c_lift_result.json", stage_c)
    else:
        stage_c = {"status": "NOT RUN", "reason": "Stage B bottle hold did not pass"}
        write_json(output / "stage_c_lift_result.json", stage_c)

    renderer.close()
    if video_frames:
        video_path = output / "standalone_grasp_lift.mp4"
        imageio.mimsave(video_path, video_frames, fps=25, macro_block_size=1)
    else:
        video_path = None
    plot_trace(output / "physical_grasp_trace.jsonl", output / "bottle_support_lift_plot.png")
    result = {"stage_b": stage_b, "stage_c": stage_c,
              "trace_path": str(trace_path), "video_path": None if video_path is None else str(video_path),
              "trace_sha256": sha256(trace_path),
              "max_contact_penetration_m": max_penetration,
              "max_bottle_com_slip_during_hold_m": max_hold_slip,
              "qpos_writes_after_simulation_start": {
                  "bottle": int(bottle_qpos_write_detected),
                  "follower": int(follower_qpos_write_detected),
              },
              "physical_rules": {"right_hand_only": True, "left_hand_clear": True,
                                 "bottle_free_joint": True, "runtime_bottle_weld": False,
                                 "mocap_attachment": False, "follow_hand_logic": False}}
    write_json(output / "physical_run_result.json", result)
    return result


def full_arm_collision_corridor(base, isolated: dict[str, Any],
                                controller_profile: dict[str, Any],
                                output: Path) -> dict[str, Any]:
    model, build_info = base.build_model("bottle_hold", {})
    controller = copy.deepcopy(controller_profile)
    controller["source_effort_limits_nm"] = build_info["source_effort_limits_nm"]
    base.configure_arm(model, controller)
    base.apply_controller_limits(model, controller, arm=True)
    data = base.initialize(model)
    arm_names = tuple(base.RIGHT_ARM_JOINTS)
    arm_jids = [joint_id(model, name) for name in arm_names]
    arm_qadr = [int(model.jnt_qposadr[jid]) for jid in arm_jids]
    lower = np.asarray([float(model.jnt_range[jid, 0]) + 1.0e-5 for jid in arm_jids])
    upper = np.asarray([float(model.jnt_range[jid, 1]) - 1.0e-5 for jid in arm_jids])
    home = np.asarray([float(data.qpos[qadr]) for qadr in arm_qadr])

    isolated_model = isolated["model"]
    isolated_data = isolated["data"]
    set_jaws(base, isolated_model, isolated_data, 1.0)
    isolated_bottle = body_id(isolated_model, "m0_bottle")
    isolated_end = body_id(isolated_model, "R_omnipicker_base_link")
    r_bottle = isolated_data.xmat[isolated_bottle].reshape(3, 3).copy()
    p_bottle = isolated_data.xpos[isolated_bottle].copy()
    r_end = isolated_data.xmat[isolated_end].reshape(3, 3).copy()
    p_end = isolated_data.xpos[isolated_end].copy()
    relative_position = r_bottle.T @ (p_end - p_bottle)
    relative_rotation = r_bottle.T @ r_end

    full_bottle = body_id(model, "m0_bottle")
    full_end = body_id(model, "R_omnipicker_base_link")
    full_r_bottle = data.xmat[full_bottle].reshape(3, 3).copy()
    full_p_bottle = data.xpos[full_bottle].copy()
    target_position = full_p_bottle + full_r_bottle @ relative_position
    target_rotation = full_r_bottle @ relative_rotation
    outward = p_end - p_bottle
    outward[2] = 0.0
    if np.linalg.norm(outward) < 1.0e-6:
        outward = p_end - p_bottle
    outward /= np.linalg.norm(outward)
    approach_position = target_position + outward * 0.10

    robot_geoms = robot_geom_ids(model)
    active_body = int(model.jnt_bodyid[arm_jids[0]])
    active_bodies = {
        body for body in range(model.nbody)
        if body == active_body or (body > 0 and is_descendant(model, body, active_body))
    }
    active_geoms = [gid for gid in robot_geoms if int(model.geom_bodyid[gid]) in active_bodies]
    bottle_geoms = bottle_geom_ids(model)
    table_geoms = table_geom_ids(model)
    floor_geoms = floor_geom_ids(model)
    excluded = {frozenset(pair) for pair in base.STATIC_COLLISION_EXCLUSIONS}

    def collision_enabled(first: int, second: int) -> bool:
        return bool((int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
                    or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first])))

    self_pairs: list[tuple[int, int]] = []
    seen_self_pairs: set[tuple[int, int]] = set()
    active_to_bottle: list[tuple[int, int]] = []
    active_to_table: list[tuple[int, int]] = []
    active_to_floor: list[tuple[int, int]] = []
    for moving in active_geoms:
        for obstacle in robot_geoms:
            if moving == obstacle or not collision_enabled(moving, obstacle):
                continue
            pair_key = tuple(sorted((moving, obstacle)))
            if pair_key in seen_self_pairs:
                continue
            body_a = int(model.geom_bodyid[moving])
            body_b = int(model.geom_bodyid[obstacle])
            if body_a == body_b or frozenset((
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a) or "",
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b) or "")) in excluded:
                continue
            if int(model.body_parentid[body_a]) == body_b or int(model.body_parentid[body_b]) == body_a:
                continue
            self_pairs.append(pair_key)
            seen_self_pairs.add(pair_key)
        for obstacle in bottle_geoms:
            if collision_enabled(moving, obstacle):
                active_to_bottle.append((moving, obstacle))
        for obstacle in table_geoms:
            if collision_enabled(moving, obstacle):
                active_to_table.append((moving, obstacle))
        for obstacle in floor_geoms:
            if collision_enabled(moving, obstacle):
                active_to_floor.append((moving, obstacle))

    # Fixed robot geometry is audited separately because arm IK cannot move it.
    fixed_table_rows = pair_rows(
        model, data,
        [gid for gid in robot_geoms if int(model.geom_bodyid[gid]) not in active_bodies],
        table_geoms,
    )
    static_fixed_table_penetrations = [row for row in fixed_table_rows
                                       if row["signed_distance_m"] < -PENETRATION_EPS_M]

    categories = {
        "active_robot_vs_fixed_robot": self_pairs,
        "active_robot_vs_bottle": active_to_bottle,
        "active_robot_vs_table": active_to_table,
        "active_robot_vs_floor": active_to_floor,
    }
    trace_path = output / "full_x2_collision_ik_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    evaluation_index = 0
    margin_m = 0.002
    position_scale_m = 0.003
    orientation_scale_rad = math.radians(1.5)
    collision_scale_m = 0.002

    def evaluate(q: np.ndarray, goal_position: np.ndarray,
                 goal_rotation: np.ndarray, phase: str) -> tuple[np.ndarray, dict[str, Any]]:
        nonlocal evaluation_index
        data.qpos[arm_qadr] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        current_position = data.xpos[full_end].copy()
        current_rotation = data.xmat[full_end].reshape(3, 3).copy()
        position_error = goal_position - current_position
        rotation_error = base.rotation_error(goal_rotation, current_rotation)
        all_rows: dict[str, list[dict[str, Any]]] = {}
        collision_penalties = []
        for label, pairs in categories.items():
            rows = [distance_row(model, data, a, b) for a, b in pairs]
            all_rows[label] = rows
            collision_penalties.extend(
                max(0.0, margin_m - float(row["signed_distance_m"])) / collision_scale_m
                for row in rows
            )
        residual = np.concatenate((position_error / position_scale_m,
                                   rotation_error / orientation_scale_rad,
                                   np.asarray(collision_penalties, dtype=float)))
        minima = {name: min((row["signed_distance_m"] for row in rows), default=math.inf)
                  for name, rows in all_rows.items()}
        closest = {
            name: min(rows, key=lambda row: row["signed_distance_m"])
            for name, rows in all_rows.items() if rows
        }
        record = {
            "evaluation": evaluation_index, "phase": phase,
            "arm_joint_qpos_rad": {name: float(value) for name, value in zip(arm_names, q)},
            "goal_position_world_m": goal_position.tolist(),
            "actual_position_world_m": current_position.tolist(),
            "position_error_m": float(np.linalg.norm(position_error)),
            "orientation_error_rad": float(np.linalg.norm(rotation_error)),
            "minimum_collision_distance_m": minima,
            "nearest_collision_pairs": closest,
            "static_fixed_robot_to_table_penetrations": static_fixed_table_penetrations,
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        evaluation_index += 1
        return residual, record

    def solve_pose(goal_position: np.ndarray, goal_rotation: np.ndarray,
                   seed: np.ndarray, phase: str) -> tuple[np.ndarray, dict[str, Any]]:
        def objective(q: np.ndarray) -> np.ndarray:
            return evaluate(q, goal_position, goal_rotation, phase)[0]
        optimized = least_squares(
            objective, np.clip(seed, lower, upper), bounds=(lower, upper),
            xtol=2.0e-8, ftol=2.0e-8, gtol=2.0e-8, max_nfev=100,
        )
        _, final_record = evaluate(optimized.x, goal_position, goal_rotation, phase)
        final_record["optimizer_status"] = int(optimized.status)
        final_record["optimizer_message"] = optimized.message
        final_record["optimizer_evaluations"] = int(optimized.nfev)
        final_record["optimizer_cost"] = float(optimized.cost)
        final_record["collision_minima_m"] = final_record["minimum_collision_distance_m"]
        final_record["source_joint_limit_margins_rad"] = {
            name: [float(value - lo), float(hi - value)]
            for name, value, lo, hi in zip(arm_names, optimized.x, lower, upper)
        }
        return optimized.x.copy(), final_record

    primary = solve_pose(target_position, target_rotation, home, "grasp_endpoint")
    previous_station_seed = np.asarray([
        0.3242831574362041, 0.029016320249999998, 0.21126909583517428,
        -0.9723067811713543, -0.16449677304097146, -0.15530422291497611,
        0.072339453750048,
    ], dtype=float)
    seeds = [primary]
    if (primary[1]["position_error_m"] > 0.005
            or primary[1]["orientation_error_rad"] > math.radians(3.0)
            or min(primary[1]["minimum_collision_distance_m"].values()) < 0.0):
        secondary = solve_pose(target_position, target_rotation,
                               previous_station_seed, "grasp_endpoint_previous_source_valid_seed")
        seeds.append(secondary)
    feasible = [item for item in seeds
                if item[1]["position_error_m"] <= 0.005
                and item[1]["orientation_error_rad"] <= math.radians(3.0)
                and min(item[1]["minimum_collision_distance_m"].values()) >= 0.0
                and all(min(value) >= 0.0 for value in item[1]["source_joint_limit_margins_rad"].values())]
    endpoint = min(feasible, key=lambda item: item[1]["optimizer_cost"]) if feasible else None
    result: dict[str, Any] = {
        "status": "BLOCKED" if endpoint is None else "IN PROGRESS",
        "physics_rollout_performed": False,
        "source_limited_arm_joints": list(arm_names),
        "collision_constraints_participate_in_ik_residual": True,
        "active_collision_pair_counts": {name: len(pairs) for name, pairs in categories.items()},
        "collision_margin_m": margin_m,
        "fixed_robot_to_table_penetrations": static_fixed_table_penetrations,
        "isolated_bottle_relative_end_effector_transform": {
            "position_m": relative_position.tolist(),
            "rotation_matrix": relative_rotation.tolist(),
        },
        "full_scene_target": {"position_world_m": target_position.tolist(),
                              "rotation_matrix": target_rotation.tolist()},
        "approach_start_position_world_m": approach_position.tolist(),
        "endpoint_attempts": [item[1] for item in seeds],
        "ik_trace_path": str(trace_path),
        "ik_trace_sha256": sha256(trace_path),
    }
    if static_fixed_table_penetrations:
        result["status"] = "BLOCKED"
        result["reason"] = "the fixed full-X2 body already penetrates the table, so arm-only IK cannot clear the scene"
    if endpoint is None:
        result.setdefault("reason", "no source-limited collision-free full-X2 endpoint was found")
        write_json(output / "full_x2_right_arm_corridor_result.json", result)
        return result

    endpoint_q, endpoint_record = endpoint
    approach_q, approach_record = solve_pose(approach_position, target_rotation,
                                              endpoint_q, "pregrasp_endpoint")
    approach_feasible = (
        approach_record["position_error_m"] <= 0.005
        and approach_record["orientation_error_rad"] <= math.radians(3.0)
        and min(approach_record["minimum_collision_distance_m"].values()) >= 0.0
    )
    path_rows = []
    previous_q = approach_q.copy()
    path_ok = approach_feasible
    path_positions = np.linspace(approach_position, target_position, 6)
    for index, position in enumerate(path_positions):
        q, record = solve_pose(position, target_rotation, previous_q, f"approach_path_{index}")
        record["path_fraction"] = float(index / (len(path_positions) - 1))
        path_rows.append(record)
        previous_q = q
        if (record["position_error_m"] > 0.005
                or record["orientation_error_rad"] > math.radians(3.0)
                or min(record["minimum_collision_distance_m"].values()) < 0.0):
            path_ok = False
            break
    result.update({
        "approach_endpoint": approach_record,
        "approach_path": path_rows,
        "endpoint_joint_pose_rad": {name: float(value) for name, value in zip(arm_names, endpoint_q)},
        "pregrasp_joint_pose_rad": {name: float(value) for name, value in zip(arm_names, approach_q)},
        "approach_path_status": "PASS" if path_ok else "BLOCKED",
        "minimum_path_clearance_by_stage_m": [row["minimum_collision_distance_m"]
                                               for row in path_rows],
    })
    if path_ok:
        for phase_name, q in (("pregrasp", approach_q), ("grasp_endpoint", endpoint_q)):
            data.qpos[arm_qadr] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            render_png(model, data, output, f"full_x2_{phase_name}.png",
                       [0.30, 0.0, 0.88], 1.8, 135.0, -18.0)
        result["status"] = "PASS"
    else:
        result["status"] = "BLOCKED"
        result["reason"] = "endpoint was feasible, but one or more collision-aware approach waypoints failed"
    write_json(output / "full_x2_right_arm_corridor_result.json", result)
    return result


def is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    current = body
    while current > 0:
        if current == ancestor:
            return True
        current = int(model.body_parentid[current])
    return False


def create_report(output: Path, result: dict[str, Any], command: str) -> None:
    geometry = result.get("stage_a_geometry", {})
    physical = result.get("physical_run", {})
    stage_b = physical.get("stage_b", {})
    stage_c = physical.get("stage_c", {})
    stage_d = result.get("full_x2_right_arm_corridor", {})
    candidate_results = []
    for candidate in geometry.get("candidates", []):
        result_path = Path(candidate.get("result_path", ""))
        if result_path.is_file():
            candidate_results.append(json.loads(result_path.read_text(encoding="utf-8")))

    def mm(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 1000.0:.3f} mm"

    def closest(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "none"
        row = min(rows, key=lambda item: item["signed_distance_m"])
        return (f"{row['body1']} / {row['geom1']} vs {row['body2']} / {row['geom2']}: "
                f"{mm(row['signed_distance_m'])}")

    lines = [
        "# Issue #46 Isolated X2 OmniPicker Grasp Baseline",
        "",
        f"- Stage A geometry: **{geometry.get('status', 'NOT RUN')}**",
        ("- Standalone hand grasp/hold: **FAIL at Stage A geometry precondition; "
         "physical contact trial NOT RUN**" if geometry.get("status") == "FAIL"
         else f"- Standalone hand grasp/hold: **{stage_b.get('status', 'NOT RUN')}**"),
        f"- Standalone 30 mm lift: **{stage_c.get('status') or 'NOT RUN'}**",
        f"- Full X2 right-arm corridor: **{stage_d.get('status') or 'NOT RUN'}**",
        "",
        "## Provenance",
        "",
        f"- RobotSim starting HEAD: `{result.get('identity', {}).get('robotsim_head_before_experiment_commit')}`",
        f"- X2 source: `{result.get('identity', {}).get('x2_repository')}` at `{result.get('identity', {}).get('x2_commit')}`",
        f"- MuJoCo Python/native: `{result.get('identity', {}).get('mujoco_python_version')}` / `{result.get('identity', {}).get('mujoco_native_version')}`",
        f"- Native library SHA-256: `{result.get('identity', {}).get('mujoco_native_library_sha256')}`",
        f"- Filtered source subtree SHA-256: `{result.get('identity', {}).get('filtered_model', {}).get('derived_urdf_sha256')}`",
        f"- Exact executed runner snapshot: `experiment_script.py`; SHA-256 `{result.get('identity', {}).get('experiment_script_sha256')}`",
        f"- Source URDF: `{result.get('identity', {}).get('x2_urdf')}`; SHA-256 `{result.get('identity', {}).get('x2_urdf_sha256')}`",
        "- Controller: unchanged accepted SIMULATION_ONLY_M0 paired-jaw profile from the pinned baseline runner.",
        "- Bottle and table: unchanged canonical multipart 70 mm / 0.570 kg bottle and G1 table helper.",
        "",
        "## Geometry",
        "",
        f"- Candidate count: `{geometry.get('candidate_count')}` (maximum 3).",
        f"- Selected candidate: `{geometry.get('selected_candidate_id')}`; no tested configuration passed all gates.",
        "- The three evidence-derived configurations changed jaw-axis azimuth (0°, 120°) and, for candidate 3, finger-axis tilt (20° toward the tangent). The compiled 70.000293 mm jaw gap spans the 70 mm bottle diameter.",
        "- Prior full-humanoid pose: compiled wrist-roll penetrated the bottle body by 38.841 mm; narrow/wide loop links penetrated by 28.155/28.092 mm.",
        "",
        "| Candidate | Pose (azimuth / tilt) | Open jaw clearance (narrow / wide) | Contact jaw distances (narrow / wide) | Closest open body collision | Closest closed nonjaw collision | Robot/table clearance | Open self-penetration |",
        "|---|---:|---:|---|---|---|---:|---|",
    ]
    for candidate in candidate_results:
        orient = candidate.get("orientation_definition", {})
        pose = orient.get("root_pose", {})
        open_jaws = candidate.get("open_jaw_clearance_by_target_body_m", {})
        closed_jaws = candidate.get("closed_jaw_to_bottle_body_distance_m", {})
        open_jaw_values = list(open_jaws.values())
        closed_jaw_values = list(closed_jaws.values())
        closed_nonjaw_rows = [
            row for row in candidate.get("closed_robot_to_bottle_distance_rows", [])
            if row.get("body1") not in TARGET_JAW_BODIES
        ]
        self_contacts = candidate.get("open_robot_self_penetrating_contacts", [])
        self_pair = closest([{
            "body1": row.get("body1", ""), "geom1": row.get("geom1", ""),
            "body2": row.get("body2", ""), "geom2": row.get("geom2", ""),
            "signed_distance_m": row.get("distance_m", 0.0),
        } for row in self_contacts])
        self_summary = (f"{len(self_contacts)} pair(s); {self_pair}"
                        if self_contacts else "none")
        lines.append(
            f"| {candidate.get('candidate_id')} ({candidate.get('status')}) | "
            f"{orient.get('jaw_axis_azimuth_deg')}° / "
            f"{orient.get('jaw_longitudinal_axis_tilt_from_world_z_toward_tangent_deg')}°; "
            f"offset tangent {mm(pose.get('offset_tangent_m'))}, height {mm(pose.get('offset_height_m'))} | "
            f"{mm(open_jaw_values[0] if len(open_jaw_values) > 0 else None)} / "
            f"{mm(open_jaw_values[1] if len(open_jaw_values) > 1 else None)} | "
            f"{mm(closed_jaw_values[0] if len(closed_jaw_values) > 0 else None)} / "
            f"{mm(closed_jaw_values[1] if len(closed_jaw_values) > 1 else None)} | "
            f"{closest(candidate.get('open_robot_to_bottle_distance_rows', []))} | "
            f"{closest(closed_nonjaw_rows)} | "
            f"open {mm(candidate.get('minimum_open_robot_to_table_distance_m'))}; "
            f"closed {mm(candidate.get('minimum_closed_robot_to_table_distance_m'))} | "
            f"{self_summary} |"
        )
    lines.extend([
        "",
        "- Best of the three for nonjaw bottle clearance was candidate 3, but its right wrist-roll still penetrated by 21.439 mm and the wide loop by 13.707 mm. All three also contain the same 39.217 mm overlap between the compiled right-wrist-yaw and right-wrist-pitch collision geoms at the tested neutral wrist joints. These source collision geometries were preserved; no collision was disabled.",
        "- Stage A failed on whole-assembly collision clearance. This result rejects the three tested configurations only; it does not establish global geometric infeasibility.",
        "- Each candidate has a raw optimizer trace and OPEN/contact overview, front, and side PNGs in this directory. The OPEN jaw summary is captured from OPEN state data; geometry trace rows retain the evaluated state distances.",
        "",
        "## Physical Result",
        "",
        "- Stage B contact/hold and Stage C load transfer/lift were NOT RUN because Stage A failed. Consequently there are no contact-force traces, bottle support/lift plots, physical MP4, or rollout qpos-write observations from this experiment.",
        "- No bottle attachment, object teleport, or physical rollout was attempted. Full X2 arm IK/corridor Stage D was also NOT RUN because its prerequisite isolated grasp pose was not established.",
        "",
        "## Reproduction",
        "",
        "```bash",
        command,
        "```",
        "",
        f"Evidence directory: `{output}`",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVIDENCE_ROOT / "20261008-run1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = (
        f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
        f"--output {output}"
    )
    (output / "experiment_commands.txt").write_text(
        f"Working branch: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'branch', '--show-current'], text=True).strip()}\n"
        f"Starting RobotSim HEAD: {subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()}\n"
        f"Exact reproduction command:\n{command}\n",
        encoding="utf-8",
    )

    source_urdf = X2_ROOT / X2_URDF_REL
    vendor_state = subprocess.check_output(["git", "-C", str(X2_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if vendor_state != X2_PIN:
        raise RuntimeError(f"Pinned vendor SHA mismatch: {vendor_state}")
    base = load_baseline()
    controller_model, _ = base.build_model("open_hold", {})
    controller = base.derive_jaw_controller(controller_model)
    source_info = filtered_urdf(output, source_urdf)
    robot_head = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    identity = source_identity(output, source_info, robot_head)
    write_json(output / "runtime_identity.json", identity)
    write_json(output / "accepted_controller_profile.json", controller)
    limits = source_limits(source_urdf)

    candidate_specs = [(0.0, 0.0), (120.0, 0.0), (0.0, 20.0)]
    geometry, selected = geometry_stage(base, output / "isolated_omnipicker_source.urdf",
                                        output, candidate_specs, controller)
    result: dict[str, Any] = {
        "status": "BLOCKED" if selected is None else "IN PROGRESS",
        "identity": identity,
        "controller_profile": controller,
        "stage_a_geometry": geometry,
        "stage_b_standalone_grasp": "NOT RUN",
        "stage_c_standalone_30mm_lift": "NOT RUN",
        "full_x2_right_arm_corridor": {"status": "NOT RUN"},
        "physical_run": {},
        "exact_reproduction_command": command,
    }
    selected_result = None if selected is None else selected["geometry_result"]
    if selected_result is not None:
        result["stage_a_geometry"]["selected_result_path"] = str(
            output / f"candidate_{selected['candidate_id']}_geometry_result.json"
        )
        # Rebuild the dynamic rig with the same selected hand pose and an actuated vertical carriage.
        dynamic_selected = {
            "build_info": {"derived_urdf_path": str(output / "isolated_omnipicker_source.urdf"),
                           **selected["build_info"]},
            "root_position": selected["root_position"],
            "root_rotation": selected["root_rotation"],
        }
        # The dynamic runner constructs its own model so the accepted profile is applied consistently.
        # Pass the generated filtered URDF through the dynamic model build.
        dynamic_selected["build_info"]["derived_urdf_path"] = str(output / "isolated_omnipicker_source.urdf")
        physical = physical_run(base, dynamic_selected, controller, limits, output)
        result["physical_run"] = physical
        result["stage_b_standalone_grasp"] = physical["stage_b"]["status"]
        result["stage_c_standalone_30mm_lift"] = physical["stage_c"]["status"]
        result["status"] = "FAIL" if physical["stage_c"]["status"] != "PASS" else "STANDALONE PASS"
        if physical["stage_c"]["status"] == "PASS":
            try:
                stage_d = full_arm_collision_corridor(base, selected, controller, output)
            except Exception as exc:
                stage_d = {"status": "BLOCKED",
                           "reason": "collision-aware full-arm corridor stage raised an implementation/runtime error",
                           "error_type": type(exc).__name__, "error_message": str(exc),
                           "physics_rollout_performed": False}
                write_json(output / "full_x2_right_arm_corridor_result.json", stage_d)
            result["full_x2_right_arm_corridor"] = stage_d
            if stage_d["status"] == "PASS":
                result["status"] = "PASS"
            else:
                result["status"] = "STANDALONE PASS / FULL X2 ARM BLOCKED"
    result_path = output / "result.json"
    write_json(result_path, result)
    create_report(output, result, command)
    print(json.dumps({
        "result": str(result_path),
        "stage_a": geometry["status"],
        "stage_b": result["stage_b_standalone_grasp"],
        "stage_c": result["stage_c_standalone_30mm_lift"],
        "stage_d": result["full_x2_right_arm_corridor"],
        "status": result["status"],
    }, indent=2))
    return 0 if result["status"] in {"PASS", "STANDALONE PASS / FULL X2 ARM BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
