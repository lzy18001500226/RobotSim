#!/usr/bin/env python3
"""Run the RobotSim M0 bottle task with a pinned G1 MuJoCo controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import uuid

import numpy as np


MUJOCO_VERSION = "3.2.6"
UPSTREAM_COMMIT = "3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
UNITREE_COMMIT = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
DEX3_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
DEX3_MODEL_RELATIVE_PATH = Path("robots/g1_description/g1_29dof_with_hand_rev_1_0.xml")
DEX3_MESH_RELATIVE_PATH = Path("robots/g1_description/meshes")
RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
RIGHT_HAND_GROUPS = ("thumb", "index", "middle")
RIGHT_HAND_OPEN_POS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RIGHT_HAND_CLOSED_POS = (-0.50, -0.70, -0.80, 1.05, 0.75, 1.05, 0.75)
HAND_COLLISION_FRICTION = "2.0 0.01 0.001"
HAND_POSITION_KP = 15.0
HAND_VELOCITY_KD = 0.5
CARRY_CONTACT_FRACTION_MIN = 0.90
CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS = 25
DEFAULT_HAND_ROLL_DEG = 20.0
DEFAULT_GRASP_HOLD_FRAMES = 2
DEFAULT_LIFT_HEIGHT_M = 0.16
DEFAULT_LIFT_FRAMES = 15
DEFAULT_TRANSFER_FRAMES = 30
DEFAULT_LOWER_FRAMES = 15
GRASP_PRELOAD_RAD = 0.30
GRASP_SITE_OFFSET_M = (-0.035, 0.0, 0.0)
APPROACH_SITE_OFFSET_M = (-0.105, 0.0, 0.140)

SOURCE_TABLE_TOP_Z = 0.8
SOURCE_TABLE_XY = np.array([0.3, 0.0], dtype=float)
SOURCE_TABLE_HALF_EXTENTS = np.array([0.2, 0.12], dtype=float)
TARGET_TABLE_HALF_EXTENTS = np.array([0.20, 0.10], dtype=float)
TABLE_GAP_M = 0.02
TARGET_TABLE_BODY_XY = np.array(
    [
        SOURCE_TABLE_XY[0],
        SOURCE_TABLE_XY[1]
        - SOURCE_TABLE_HALF_EXTENTS[1]
        - TARGET_TABLE_HALF_EXTENTS[1]
        - TABLE_GAP_M,
    ],
    dtype=float,
)
TARGET_TABLE_SITE_OFFSET_Y_M = 0.04
TARGET_TABLE_XY = TARGET_TABLE_BODY_XY + np.array(
    [0.0, TARGET_TABLE_SITE_OFFSET_Y_M], dtype=float
)
TARGET_TABLE_BODY_Z = 0.0
TARGET_TABLE_TOP_Z = TARGET_TABLE_BODY_Z + 0.8
TARGET_MARGIN_M = 0.03
OBJECT_BODY_NAME = "green_box"
OBJECT_JOINT_NAME = "box_joint"
OBJECT_MAIN_GEOM = "m0_bottle_body"
OBJECT_COLLISION_GEOMS = (
    OBJECT_MAIN_GEOM,
    "m0_bottle_shoulder",
    "m0_bottle_neck",
    "m0_bottle_cap",
)
OBJECT_GEOM_SPECS = (
    {
        "name": OBJECT_MAIN_GEOM,
        "type": "cylinder",
        "pos": "0 0 0",
        "size": "0.0325 0.078",
        "mass": "0.470",
        "rgba": "0.64 0.84 0.90 0.78",
    },
    {
        "name": "m0_bottle_shoulder",
        "type": "cylinder",
        "pos": "0 0 0.084",
        "size": "0.029 0.020",
        "mass": "0.015",
        "rgba": "0.64 0.84 0.90 0.78",
    },
    {
        "name": "m0_bottle_neck",
        "type": "cylinder",
        "pos": "0 0 0.118",
        "size": "0.015 0.019",
        "mass": "0.010",
        "rgba": "0.68 0.86 0.91 0.85",
    },
    {
        "name": "m0_bottle_cap",
        "type": "cylinder",
        "pos": "0 0 0.145",
        "size": "0.018 0.008",
        "mass": "0.005",
        "rgba": "0.94 0.95 0.91 1",
    },
)


def _git_value(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(distribution: str, module_name: str | None = None) -> str | None:
    module = sys.modules.get(module_name or distribution.replace("-", "_"))
    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        return str(module_version)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mesh_provenance(unitree_root: Path, mesh_dir: Path) -> dict:
    try:
        unitree_resolved = unitree_root.resolve(strict=True)
        g1_root = (unitree_resolved / "unitree_robots/g1").resolve(strict=True)
        g1_root.relative_to(unitree_resolved)
        mesh_root = (g1_root / "meshes").resolve(strict=True)
        mesh_root.relative_to(g1_root / "meshes")
        mesh_resolved = mesh_dir.resolve(strict=True)
        mesh_resolved.relative_to(mesh_root)
    except (OSError, ValueError):
        return {
            "path_class": "external_rejected",
            "path_relative_to_unitree": "outside_verified_unitree_g1_mesh_tree",
        }
    return {
        "path_class": "pinned_unitree_g1_mesh_tree",
        "path_relative_to_unitree": mesh_resolved.relative_to(unitree_resolved).as_posix(),
    }


def _acceptance_thresholds(monitor: "AcceptanceMonitor | None" = None) -> dict:
    if monitor is None:
        return {
            "target_table_margin_m": TARGET_MARGIN_M,
            "minimum_thumb_contact_force_n": 0.2,
            "minimum_opposing_finger_force_n": 0.2,
            "grasp_contact_frames": 5,
            "carry_contact_fraction_min": CARRY_CONTACT_FRACTION_MIN,
            "carry_contact_max_gap_physics_steps": CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS,
            "minimum_lift_height_m": 0.05,
            "contact_free_release_frames": 5,
            "stable_duration_s": 1.0,
            "linear_speed_m_s": 0.03,
            "angular_speed_rad_s": 0.20,
            "stable_position_radius_m": 0.02,
            "control_step_translation_m": 0.20,
            "physics_step_translation_m": 0.005,
            "physics_step_angular_jump_rad": 0.025,
            "maximum_penetration_m": 0.025,
            "drop_height_m": 0.50,
            "left_arm_object_contact_allowed": False,
            "runtime_object_qpos_write_allowed": False,
            "runtime_equality_carry_allowed": False,
        }
    return {
        "target_table_margin_m": monitor.target_margin,
        "minimum_thumb_contact_force_n": monitor.minimum_thumb_force_n,
        "minimum_opposing_finger_force_n": monitor.minimum_opposing_force_n,
        "grasp_contact_frames": monitor.grasp_frames_required,
        "carry_contact_fraction_min": CARRY_CONTACT_FRACTION_MIN,
        "carry_contact_max_gap_physics_steps": CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS,
        "minimum_lift_height_m": monitor.lift_height,
        "contact_free_release_frames": monitor.release_frames_required,
        "stable_duration_s": monitor.minimum_stable_duration_s,
        "linear_speed_m_s": monitor.linear_speed_limit,
        "angular_speed_rad_s": monitor.angular_speed_limit,
        "stable_position_radius_m": monitor.stable_position_radius,
        "control_step_translation_m": monitor.teleport_step_limit,
        "physics_step_translation_m": monitor.physics_step_translation_limit,
        "physics_step_angular_jump_rad": monitor.physics_step_angular_jump_limit,
        "maximum_penetration_m": monitor.penetration_limit,
        "drop_height_m": monitor.drop_z_limit,
        "left_arm_object_contact_allowed": False,
        "runtime_object_qpos_write_allowed": False,
        "runtime_equality_carry_allowed": False,
    }


def _runtime_identity(args: argparse.Namespace, monitor: "AcceptanceMonitor | None" = None) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    unitree_root = Path(args.unitree_root)
    candidate_root = Path(args.candidate_root)
    dex3_root = Path(args.dex3_root)
    unitree_sha = _git_value(unitree_root, "rev-parse", "HEAD")
    candidate_sha = _git_value(candidate_root, "rev-parse", "HEAD")
    dex3_sha = _git_value(dex3_root, "rev-parse", "HEAD")
    unitree_dirty = _git_value(unitree_root, "status", "--porcelain", "--untracked-files=all")
    candidate_dirty = _git_value(candidate_root, "status", "--porcelain", "--untracked-files=all")
    dex3_dirty = _git_value(dex3_root, "status", "--porcelain", "--untracked-files=all")
    mujoco_module = sys.modules.get("mujoco")
    robotsim_dirty = _git_value(repo_root, "status", "--porcelain", "--untracked-files=all")
    return {
        "run_id": args.run_id or uuid.uuid4().hex,
        "robotsim_sha": _git_value(repo_root, "rev-parse", "HEAD"),
        "robotsim_dirty": None if robotsim_dirty is None else bool(robotsim_dirty),
        "python_version": platform.python_version(),
        "mujoco_version": getattr(mujoco_module, "__version__", None)
        or _package_version("mujoco"),
        "mujoco_native_version": (
            mujoco_module.mj_versionString() if mujoco_module is not None else None
        ),
        "numpy_version": _package_version("numpy"),
        "h5py_version": _package_version("h5py"),
        "opencv_version": _package_version("opencv-python-headless", "cv2"),
        "upstream_shas": {
            "humanoid_vla": candidate_sha,
            "unitree_mujoco": unitree_sha,
            "unitree_ros_dex3": dex3_sha,
            "grasp_reference": "ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff",
        },
        "upstream_dirty": {
            "humanoid_vla": None if candidate_dirty is None else bool(candidate_dirty),
            "unitree_mujoco": None if unitree_dirty is None else bool(unitree_dirty),
            "unitree_ros_dex3": None if dex3_dirty is None else bool(dex3_dirty),
        },
        "mesh_provenance": _mesh_provenance(unitree_root, Path(args.mesh_dir)),
        "dex3_model_provenance": {
            "repository": "https://github.com/unitreerobotics/unitree_ros",
            "revision": dex3_sha,
            "model": DEX3_MODEL_RELATIVE_PATH.as_posix(),
            "mesh_path_class": "pinned_unitree_ros_g1_description_meshes",
            "mesh_path_relative_to_repository": DEX3_MESH_RELATIVE_PATH.as_posix(),
        },
        "seed": args.seed,
        "acceptance_thresholds": _acceptance_thresholds(monitor),
        "output_directory": str(Path(args.output_json).parent.resolve()),
    }


def _verify_unitree_mesh_inputs(args: argparse.Namespace) -> None:
    candidate_root = Path(args.candidate_root)
    candidate_sha = _git_value(candidate_root, "rev-parse", "HEAD")
    if candidate_sha != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected Humanoid VLA {UPSTREAM_COMMIT}; found {candidate_sha or 'unavailable'}"
        )
    candidate_dirty = _git_value(
        candidate_root, "status", "--porcelain", "--untracked-files=all"
    )
    if candidate_dirty is None or candidate_dirty:
        raise RuntimeError("Humanoid VLA checkout must be clean")
    actual_sha = _git_value(Path(args.unitree_root), "rev-parse", "HEAD")
    if actual_sha != UNITREE_COMMIT:
        raise RuntimeError(f"Expected Unitree MuJoCo {UNITREE_COMMIT}; found {actual_sha or 'unavailable'}")
    dirty = _git_value(Path(args.unitree_root), "status", "--porcelain", "--untracked-files=all")
    if dirty is None or dirty:
        raise RuntimeError("Unitree MuJoCo mesh checkout must be clean")
    provenance = _mesh_provenance(Path(args.unitree_root), Path(args.mesh_dir))
    if provenance["path_class"] != "pinned_unitree_g1_mesh_tree":
        raise RuntimeError("mesh directory must resolve inside the pinned Unitree G1 mesh tree")

    dex3_root = Path(args.dex3_root)
    actual_dex3 = _git_value(dex3_root, "rev-parse", "HEAD")
    if actual_dex3 != DEX3_COMMIT:
        raise RuntimeError(f"Expected Unitree Dex3 model {DEX3_COMMIT}; found {actual_dex3 or 'unavailable'}")
    dex3_dirty = _git_value(dex3_root, "status", "--porcelain", "--untracked-files=all")
    if dex3_dirty is None or dex3_dirty:
        raise RuntimeError("Unitree Dex3 model checkout must be clean")
    dex3_resolved = dex3_root.resolve(strict=True)
    model_path = (dex3_resolved / DEX3_MODEL_RELATIVE_PATH).resolve(strict=True)
    mesh_root = (dex3_resolved / DEX3_MESH_RELATIVE_PATH).resolve(strict=True)
    if not model_path.is_file() or not mesh_root.is_dir():
        raise RuntimeError("Pinned Unitree Dex3 model or mesh directory is missing")
    model_path.relative_to(dex3_resolved)
    mesh_root.relative_to(dex3_resolved)


def _safe_error(exc: Exception, args: argparse.Namespace) -> str:
    message = f"{type(exc).__name__}: {exc}"
    replacements = (
        (Path(args.candidate_root), "<candidate-root>"),
        (Path(args.unitree_root), "<unitree-root>"),
        (Path(args.dex3_root), "<dex3-root>"),
        (Path(args.mesh_dir), "<mesh-dir>"),
        (Path(args.output_json).parent, "<output-directory>"),
    )
    for path, label in replacements:
        message = message.replace(str(path), label)
        try:
            message = message.replace(str(path.resolve()), label)
        except OSError:
            pass
    return message


def make_single_right_dex3_model_xml(
    model_xml: str, dex3_model_xml: str, dex3_mesh_dir: Path
) -> str:
    """Graft the pinned right Dex3 subtree onto the unchanged 29-DoF G1 model."""
    root = ET.fromstring(model_xml)
    dex3_root = ET.fromstring(dex3_model_xml)
    asset = root.find("asset")
    source_asset = dex3_root.find("asset")
    actuator = root.find("actuator")
    source_actuator = dex3_root.find("actuator")
    wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    source_wrist = dex3_root.find(".//body[@name='right_wrist_yaw_link']")
    left_wrist = root.find(".//body[@name='left_wrist_yaw_link']")
    if any(item is None for item in (asset, source_asset, actuator, source_actuator, wrist, source_wrist, left_wrist)):
        raise ValueError("G1 or pinned Dex3 model is missing required hand integration elements")

    base_actuator_names = [item.attrib.get("name") for item in actuator]
    if len(base_actuator_names) != 29:
        raise ValueError(f"Expected the pinned G1 body to retain 29 actuators, found {len(base_actuator_names)}")
    if any(item.tag == "body" and item.attrib.get("name", "").startswith("left_hand_") for item in left_wrist):
        raise ValueError("The G1 model unexpectedly contains a left articulated hand")

    for parent, obsolete in (
        (wrist, {"right_palm_pad", "right_rubber_hand"}),
        (left_wrist, {"left_palm_pad"}),
    ):
        for child in list(parent):
            if child.tag == "geom" and (
                child.attrib.get("name") in obsolete
                or child.attrib.get("mesh") in obsolete
            ):
                parent.remove(child)

    mesh_root = dex3_mesh_dir.resolve(strict=True)
    existing_meshes = {item.attrib.get("name") for item in asset.findall("mesh")}
    for mesh in source_asset.findall("mesh"):
        name = mesh.attrib.get("name", "")
        if not name.startswith("right_hand_"):
            continue
        if name in existing_meshes:
            raise ValueError(f"G1 model already defines Dex3 mesh {name!r}")
        source_path = (mesh_root / mesh.attrib["file"]).resolve(strict=True)
        source_path.relative_to(mesh_root)
        if not source_path.is_file():
            raise FileNotFoundError(f"Pinned Dex3 mesh is missing: {name}")
        copied_mesh = copy.deepcopy(mesh)
        copied_mesh.set("file", str(source_path))
        asset.append(copied_mesh)
        existing_meshes.add(name)

    if not {"right_hand_palm_link", *[name.replace("_joint", "_link") for name in RIGHT_HAND_JOINT_NAMES]} <= existing_meshes:
        raise ValueError("Pinned Dex3 model is missing one or more right-hand meshes")

    imported = []
    for child in source_wrist:
        is_palm_geom = child.tag == "geom" and child.attrib.get("mesh") == "right_hand_palm_link"
        is_right_digit = child.tag == "body" and child.attrib.get("name", "").startswith("right_hand_")
        if is_palm_geom or is_right_digit:
            imported.append(copy.deepcopy(child))
    if not imported:
        raise ValueError("Pinned Dex3 model has no right palm or finger subtree")

    joints_found = set()
    def configure_node(node: ET.Element) -> None:
        if node.tag == "joint":
            name = node.attrib.get("name")
            if name in RIGHT_HAND_JOINT_NAMES:
                joints_found.add(name)
                node.set("damping", "0.08")
                node.set("armature", "0.005")
                node.set("frictionloss", "0.02")
        if node.tag == "geom" and node.attrib.get("contype") != "0":
            node.set("friction", HAND_COLLISION_FRICTION)
            node.set("condim", "4")
            node.set("solimp", "0.95 0.95 0.001")
            node.set("solref", "0.01 1")
        for descendant in node:
            configure_node(descendant)

    for child in imported:
        configure_node(child)
        wrist.append(child)
    if joints_found != set(RIGHT_HAND_JOINT_NAMES):
        raise ValueError(f"Pinned Dex3 right hand joint set is incomplete: {sorted(joints_found)}")

    source_motors = {
        item.attrib.get("name"): item
        for item in source_actuator
        if item.tag == "motor"
    }
    source_joints = {
        item.attrib.get("name"): item
        for item in dex3_root.iter("joint")
    }
    for name in RIGHT_HAND_JOINT_NAMES:
        if name in {item.attrib.get("name") for item in actuator}:
            raise ValueError(f"G1 model already defines actuator {name!r}")
        motor = source_motors.get(name)
        joint = source_joints.get(name)
        if motor is None or joint is None or "actuatorfrcrange" not in joint.attrib:
            raise ValueError(f"Pinned Dex3 model is missing actuator or torque limits for {name}")
        copied_motor = copy.deepcopy(motor)
        copied_motor.set("ctrlrange", joint.attrib["actuatorfrcrange"])
        copied_motor.set("ctrllimited", "true")
        actuator.append(copied_motor)

    final_actuator_names = [item.attrib.get("name") for item in actuator]
    if final_actuator_names[:29] != base_actuator_names:
        raise AssertionError("Appending Dex3 actuators changed the pinned G1 actuator mapping")
    if len(final_actuator_names) != 36:
        raise AssertionError("Expected 29 G1 actuators followed by seven right Dex3 actuators")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def make_bottle_scene_xml(scene_xml: str, robot_model_path: str) -> str:
    """Adapt the candidate scene while preserving its G1 model and freejoint."""
    root = ET.fromstring(scene_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Candidate scene has no worldbody")

    include = root.find("include")
    if include is None:
        raise ValueError("Candidate scene has no G1 model include")
    include.set("file", str(Path(robot_model_path).resolve()))

    bottle = worldbody.find(f"body[@name='{OBJECT_BODY_NAME}']")
    if bottle is None:
        raise ValueError(f"Candidate scene has no {OBJECT_BODY_NAME} body")
    freejoint = bottle.find(f"freejoint[@name='{OBJECT_JOINT_NAME}']")
    if freejoint is None:
        raise ValueError(f"{OBJECT_BODY_NAME} must retain its freejoint")
    site = bottle.find("site[@name='box_site']")
    bottle.set("pos", "0.3 0 0.878")

    for child in list(bottle):
        if child is not freejoint and child is not site:
            bottle.remove(child)
    for spec in OBJECT_GEOM_SPECS:
        ET.SubElement(
            bottle,
            "geom",
            {
                **spec,
                "friction": "1.5 0.01 0.001",
                "condim": "4",
                "solimp": "0.95 0.95 0.001",
                "solref": "0.01 1",
            },
        )

    for body in list(worldbody.findall("body")):
        name = body.attrib.get("name", "")
        if name == "red_cube" or name.startswith("distractor_"):
            worldbody.remove(body)

    source_table = worldbody.find("body[@name='table']")
    if source_table is None:
        raise ValueError("Candidate scene has no source table")
    source_top = next(
        (geom for geom in source_table.findall("geom") if geom.attrib.get("type") == "box"),
        None,
    )
    if source_top is None:
        raise ValueError("Candidate source table has no box top")
    source_top.set("name", "m0_source_table_top")
    source_table.set("pos", f"{SOURCE_TABLE_XY[0]:.3f} {SOURCE_TABLE_XY[1]:.3f} 0")
    source_top.set(
        "size",
        f"{SOURCE_TABLE_HALF_EXTENTS[0]:.3f} {SOURCE_TABLE_HALF_EXTENTS[1]:.3f} 0.4",
    )
    for geom in list(source_table.findall("geom")):
        if geom.attrib.get("name") == "place_marker":
            source_table.remove(geom)

    target_table = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "m0_target_table",
            "pos": (
                f"{TARGET_TABLE_BODY_XY[0]:.3f} {TARGET_TABLE_BODY_XY[1]:.3f} "
                f"{TARGET_TABLE_BODY_Z:.3f}"
            ),
        },
    )
    ET.SubElement(
        target_table,
        "geom",
        {
            "name": "m0_target_table_top",
            "type": "box",
            "pos": "0 0 0.4",
            "size": (
                f"{TARGET_TABLE_HALF_EXTENTS[0]:.3f} "
                f"{TARGET_TABLE_HALF_EXTENTS[1]:.3f} 0.4"
            ),
            "rgba": "0.24 0.34 0.42 1",
            "friction": "1.2 0.01 0.001",
        },
    )
    ET.SubElement(
        target_table,
        "geom",
        {
            "name": "m0_target_marker",
            "type": "cylinder",
            "pos": f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.803",
            "size": "0.055 0.002",
            "rgba": "0.90 0.66 0.16 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        target_table,
        "site",
        {
            "name": "m0_target_site",
            "pos": f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.805",
            "size": "0.005",
        },
    )

    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)

    camera = worldbody.find("camera[@name='scene_camera']")
    if camera is not None:
        camera.set("pos", "0.4 -1.1 1.7")
        camera.set("xyaxes", "1 0 0 0 0.55 1")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def prepare_scene(candidate_root: Path, mesh_dir: Path, dex3_root: Path, output_dir: Path) -> Path:
    source_model_path = candidate_root / "sim" / "models" / "g1_29dof.xml"
    source_scene_path = candidate_root / "sim" / "g1_with_camera.xml"
    if not source_model_path.is_file() or not source_scene_path.is_file():
        raise FileNotFoundError("Pinned candidate is missing its G1 scene assets")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Pinned Unitree G1 meshes are missing: {mesh_dir}")

    model_copy_path = output_dir / "g1_29dof.xml"
    scene_path = output_dir / "g1_m0_bottle.xml"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_root = ET.parse(source_model_path).getroot()
    compiler = model_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        model_root.insert(0, compiler)
    compiler.set("meshdir", str(mesh_dir.resolve()))
    model_xml = ET.tostring(model_root, encoding="unicode")
    dex3_model_path = dex3_root / DEX3_MODEL_RELATIVE_PATH
    dex3_mesh_dir = dex3_root / DEX3_MESH_RELATIVE_PATH
    dex3_model_xml = dex3_model_path.read_text(encoding="utf-8")
    integrated_model_xml = make_single_right_dex3_model_xml(
        model_xml, dex3_model_xml, dex3_mesh_dir
    )
    model_root = ET.fromstring(integrated_model_xml)
    ET.indent(model_root, space="  ")
    ET.ElementTree(model_root).write(model_copy_path, encoding="unicode", xml_declaration=True)

    scene_xml = source_scene_path.read_text(encoding="utf-8")
    adapted = make_bottle_scene_xml(scene_xml, str(model_copy_path))
    scene_path.write_text(adapted + "\n", encoding="utf-8")
    return scene_path


def _projected_geom_half_extents_xy(
    shape: str, rotation: np.ndarray, size: np.ndarray
) -> np.ndarray:
    """Return conservative world-XY half extents for a supported MuJoCo geom."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    size = np.asarray(size, dtype=float)
    if shape == "box":
        return np.abs(rotation[:2, :]) @ size[:3]
    if shape == "sphere":
        return np.full(2, float(size[0]))
    if shape in ("cylinder", "capsule"):
        axis = rotation[:, 2]
        radial_size = float(size[0])
        half_length = float(size[1])
        axial_xy = np.abs(axis[:2])
        if shape == "cylinder":
            radial_xy = radial_size * np.sqrt(np.maximum(0.0, 1.0 - axis[:2] ** 2))
        else:
            radial_xy = np.full(2, radial_size)
        return radial_xy + half_length * axial_xy
    raise ValueError(f"unsupported bottle collision geometry: {shape}")


def check_projected_footprint(
    projected_geoms: list[dict],
    target_xy: np.ndarray,
    target_half_extents: np.ndarray,
    margin: float,
) -> dict:
    """Require every projected object geom to clear all four table edges by margin."""
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    target_half_extents = np.asarray(target_half_extents, dtype=float).reshape(2)
    inner_half_extents = target_half_extents - float(margin)
    valid_table = bool(
        np.all(np.isfinite(target_xy))
        and np.all(np.isfinite(inner_half_extents))
        and np.all(inner_half_extents > 0.0)
        and math.isfinite(margin)
        and margin >= 0.0
    )
    geometries = []
    for geom in projected_geoms:
        center_xy = np.asarray(geom["center_xy_m"], dtype=float).reshape(2)
        half_extents_xy = np.asarray(geom["half_extents_xy_m"], dtype=float).reshape(2)
        clearances = inner_half_extents - np.abs(center_xy - target_xy) - half_extents_xy
        finite = bool(np.all(np.isfinite(center_xy)) and np.all(np.isfinite(half_extents_xy)))
        inside = bool(valid_table and finite and np.all(clearances >= -1e-9))
        geometries.append(
            {
                "name": geom["name"],
                "center_xy_m": center_xy.tolist(),
                "half_extents_xy_m": half_extents_xy.tolist(),
                "edge_clearance_xy_m": clearances.tolist(),
                "inside_with_margin": inside,
            }
        )
    passed = bool(
        valid_table and geometries and all(item["inside_with_margin"] for item in geometries)
    )
    return {
        "passed": passed,
        "configured_margin_m": float(margin),
        "target_center_xy_m": target_xy.tolist(),
        "target_half_extents_xy_m": target_half_extents.tolist(),
        "inner_half_extents_xy_m": inner_half_extents.tolist(),
        "geometries": geometries,
    }


def _quaternion_angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).reshape(4)
    second = np.asarray(second, dtype=float).reshape(4)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if not math.isfinite(first_norm + second_norm) or min(first_norm, second_norm) <= 1e-12:
        return math.inf
    dot = float(np.clip(abs(np.dot(first / first_norm, second / second_norm)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _grasp_hold_target(hand_joint_positions: dict[str, float]) -> np.ndarray:
    measured = np.asarray(
        [hand_joint_positions[name] for name in RIGHT_HAND_JOINT_NAMES], dtype=float
    )
    open_target = np.asarray(RIGHT_HAND_OPEN_POS, dtype=float)
    closed_target = np.asarray(RIGHT_HAND_CLOSED_POS, dtype=float)
    closure_direction = np.sign(closed_target - open_target)
    return np.clip(
        measured + closure_direction * GRASP_PRELOAD_RAD,
        np.minimum(open_target, closed_target),
        np.maximum(open_target, closed_target),
    )


@dataclass
class AcceptanceMonitor:
    initial_position: np.ndarray
    target_xy: np.ndarray
    target_half_extents: np.ndarray
    dt: float
    lift_height: float = 0.05
    target_margin: float = TARGET_MARGIN_M
    minimum_thumb_force_n: float = 0.2
    minimum_opposing_force_n: float = 0.2
    grasp_frames_required: int = 5
    release_frames_required: int = 5
    minimum_stable_duration_s: float = 1.0
    settling_frames_required: int = field(init=False)
    linear_speed_limit: float = 0.03
    angular_speed_limit: float = 0.20
    stable_position_radius: float = 0.02
    teleport_step_limit: float = 0.20
    physics_step_translation_limit: float = 0.005
    physics_step_angular_jump_limit: float = 0.025
    penetration_limit: float = 0.025
    drop_z_limit: float = 0.50
    runtime_equality_count: int = 0

    def __post_init__(self) -> None:
        self.initial_position = np.asarray(self.initial_position, dtype=float).copy()
        self.target_xy = np.asarray(self.target_xy, dtype=float).copy()
        self.target_half_extents = np.asarray(self.target_half_extents, dtype=float).copy()
        self.settling_frames_required = math.ceil(
            self.minimum_stable_duration_s / self.dt
        ) + 1
        self.stages: dict[str, dict[str, float | int] | None] = {
            name: None
            for name in ("GRASP", "LIFT", "TRANSFER", "RELEASE", "PLACE")
        }
        self.failures: list[str] = []
        self.frame_count = 0
        self.grasp_contact_frames = 0
        self.max_grasp_contact_frames = 0
        self.release_frames = 0
        self.settling_frames = 0
        self.max_settling_frames = 0
        self.max_left_arm_force_n = 0.0
        self.max_right_hand_force_n = 0.0
        self.max_digit_force_n = {name: 0.0 for name in RIGHT_HAND_GROUPS}
        self.max_post_release_linear_speed = 0.0
        self.max_post_release_angular_speed = 0.0
        self.max_object_step = 0.0
        self.physics_step_count = 0
        self.max_physics_step_translation = 0.0
        self.max_physics_step_angular_jump = 0.0
        self.max_physics_step_penetration = 0.0
        self.previous_physics_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.max_penetration = 0.0
        self.last_position: np.ndarray | None = None
        self.stable_anchor: np.ndarray | None = None
        self.last_footprint_check: dict | None = None
        self.placement_footprint_check: dict | None = None
        self.finite_state = True
        self.bounded_state = True
        self.dropped = False
        self.left_arm_contact_seen = False
        self.carry_contact_lost = False
        self.carry_contact_frames = 0
        self.carry_contact_required = False
        self.carry_physics_steps = 0
        self.carry_contact_gap_physics_steps = 0
        self.max_carry_contact_gap_physics_steps = 0

    def _stage(self, name: str, sim_time: float) -> None:
        if self.stages[name] is None:
            self.stages[name] = {"sim_time_s": float(sim_time), "control_frame": self.frame_count}

    def update(
        self,
        *,
        sim_time: float,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        upright: bool,
        left_contact: bool,
        right_contact: bool,
        digit_contacts: dict[str, bool],
        digit_forces_n: dict[str, float],
        left_force_n: float,
        right_force_n: float,
        fingers_open: bool,
        target_contact: bool,
        linear_speed: float,
        angular_speed: float,
        minimum_contact_distance: float,
        state_finite: bool,
        state_extent: float,
        footprint_check: dict,
    ) -> None:
        position = np.asarray(position, dtype=float)
        self.frame_count += 1
        self.last_footprint_check = footprint_check
        self.finite_state &= bool(
            state_finite
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(quaternion_wxyz))
            and math.isfinite(sim_time)
        )
        self.bounded_state &= bool(math.isfinite(state_extent) and state_extent <= 1000.0)
        self.max_left_arm_force_n = max(self.max_left_arm_force_n, float(left_force_n))
        self.max_right_hand_force_n = max(self.max_right_hand_force_n, float(right_force_n))
        for name in RIGHT_HAND_GROUPS:
            self.max_digit_force_n[name] = max(
                self.max_digit_force_n[name], float(digit_forces_n.get(name, 0.0))
            )
        if left_contact:
            self.left_arm_contact_seen = True
            self._fail("left arm contacted the bottle")

        if self.last_position is not None and np.all(np.isfinite(position)):
            step_distance = float(np.linalg.norm(position - self.last_position))
            self.max_object_step = max(self.max_object_step, step_distance)
            if step_distance > self.teleport_step_limit:
                self._fail("object teleport exceeded the per-control-step limit")
        if np.all(np.isfinite(position)):
            self.last_position = position.copy()
            if position[2] < self.drop_z_limit:
                self.dropped = True
                self._fail("object fell below the configured drop-height limit")

        self._record_penetration(minimum_contact_distance)

        opposing_contact = any(
            digit_contacts.get(name, False)
            and digit_forces_n.get(name, 0.0) >= self.minimum_opposing_force_n
            for name in ("index", "middle")
        )
        credible_grasp = bool(
            digit_contacts.get("thumb", False)
            and digit_forces_n.get("thumb", 0.0) >= self.minimum_thumb_force_n
            and opposing_contact
        )
        self.grasp_contact_frames = self.grasp_contact_frames + 1 if credible_grasp else 0
        self.max_grasp_contact_frames = max(
            self.max_grasp_contact_frames, self.grasp_contact_frames
        )
        if self.grasp_contact_frames >= self.grasp_frames_required:
            self._stage("GRASP", sim_time)

        lifted = position[2] >= self.initial_position[2] + self.lift_height
        if self.stages["GRASP"] is not None and lifted:
            self._stage("LIFT", sim_time)

        target_inner_extents = self.target_half_extents - self.target_margin
        inside_target = bool(
            np.all(target_inner_extents > 0)
            and np.all(np.abs(position[:2] - self.target_xy) <= target_inner_extents)
        )
        if self.stages["LIFT"] is not None and inside_target:
            self._stage("TRANSFER", sim_time)

        if (
            self.stages["TRANSFER"] is not None
            and fingers_open
            and not right_contact
            and not left_contact
        ):
            self.release_frames += 1
            if self.release_frames >= self.release_frames_required:
                self._stage("RELEASE", sim_time)
        else:
            self.release_frames = 0

        if self.stages["RELEASE"] is not None:
            self.placement_footprint_check = footprint_check
            self.max_post_release_linear_speed = max(
                self.max_post_release_linear_speed, float(linear_speed)
            )
            self.max_post_release_angular_speed = max(
                self.max_post_release_angular_speed, float(angular_speed)
            )

        stable = bool(
            self.stages["RELEASE"] is not None
            and target_contact
            and inside_target
            and bool(footprint_check.get("passed", False))
            and linear_speed <= self.linear_speed_limit
            and angular_speed <= self.angular_speed_limit
        )
        if stable:
            if (
                self.stable_anchor is None
                or np.linalg.norm(position - self.stable_anchor) > self.stable_position_radius
            ):
                self.stable_anchor = position.copy()
                self.settling_frames = 1
            else:
                self.settling_frames += 1
            self.max_settling_frames = max(self.max_settling_frames, self.settling_frames)
            if self.settling_frames >= self.settling_frames_required:
                self._stage("PLACE", sim_time)
        else:
            self.stable_anchor = None
            self.settling_frames = 0

    def _record_penetration(self, minimum_contact_distance: float) -> None:
        if not math.isfinite(minimum_contact_distance):
            return
        penetration = max(0.0, -float(minimum_contact_distance))
        self.max_penetration = max(self.max_penetration, penetration)
        if penetration > self.penetration_limit:
            self._fail("object penetration exceeded the configured limit")

    def record_physics_step(
        self,
        *,
        sim_time: float,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        minimum_contact_distance: float,
        left_arm_contact: bool = False,
        digit_contacts: dict[str, bool] | None = None,
        digit_forces_n: dict[str, float] | None = None,
    ) -> None:
        position = np.asarray(position, dtype=float).reshape(3)
        quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
        self.physics_step_count += 1
        if left_arm_contact:
            self.left_arm_contact_seen = True
            self._fail("left arm contacted the bottle during a physics step")
        if not (
            math.isfinite(sim_time)
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(quaternion_wxyz))
        ):
            self.finite_state = False
            self._fail("non-finite object pose during a physics step")
        previous = self.previous_physics_pose
        if previous is not None and np.all(np.isfinite(position)):
            translation = float(np.linalg.norm(position - previous[0]))
            angular_jump = _quaternion_angular_distance(previous[1], quaternion_wxyz)
            self.max_physics_step_translation = max(
                self.max_physics_step_translation, translation
            )
            self.max_physics_step_angular_jump = max(
                self.max_physics_step_angular_jump, angular_jump
            )
            if translation > self.physics_step_translation_limit:
                self._fail("object translation exceeded the per-physics-step limit")
            if angular_jump > self.physics_step_angular_jump_limit:
                self._fail("object rotation exceeded the per-physics-step limit")
        if np.all(np.isfinite(position)) and np.all(np.isfinite(quaternion_wxyz)):
            self.previous_physics_pose = (position.copy(), quaternion_wxyz.copy())
        if math.isfinite(minimum_contact_distance):
            penetration = max(0.0, -float(minimum_contact_distance))
            self.max_physics_step_penetration = max(
                self.max_physics_step_penetration, penetration
            )
            self._record_penetration(minimum_contact_distance)

        if self.carry_contact_required and self.stages["RELEASE"] is None:
            self.carry_physics_steps += 1
            contacts = digit_contacts or {}
            forces = digit_forces_n or {}
            opposing_contact = any(
                contacts.get(name, False)
                and forces.get(name, 0.0) >= self.minimum_opposing_force_n
                for name in ("index", "middle")
            )
            credible_grasp = bool(
                contacts.get("thumb", False)
                and forces.get("thumb", 0.0) >= self.minimum_thumb_force_n
                and opposing_contact
            )
            if credible_grasp:
                self.carry_contact_frames += 1
                self.carry_contact_gap_physics_steps = 0
            else:
                self.carry_contact_gap_physics_steps += 1
                self.max_carry_contact_gap_physics_steps = max(
                    self.max_carry_contact_gap_physics_steps,
                    self.carry_contact_gap_physics_steps,
                )
                if self.carry_contact_gap_physics_steps > CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS:
                    self.carry_contact_lost = True
                    self._fail("right Dex3 multi-finger contact was lost during carry")

    def _fail(self, reason: str) -> None:
        if reason not in self.failures:
            self.failures.append(reason)

    def result(self, final_sample: dict | None) -> dict:
        task_stages = {name: sample is not None for name, sample in self.stages.items()}
        stage_times = [self.stages[name]["sim_time_s"] for name in ("GRASP", "LIFT", "TRANSFER", "RELEASE", "PLACE") if self.stages[name] is not None]
        ordered_stages = bool(
            len(stage_times) == len(self.stages)
            and all(first < second for first, second in zip(stage_times, stage_times[1:]))
        )
        safety = {
            "FINITE_STATE": self.finite_state,
            "ROBOT_STATE_BOUNDED": self.bounded_state,
            "NO_TELEPORT": bool(
                self.max_object_step <= self.teleport_step_limit
                and self.max_physics_step_translation <= self.physics_step_translation_limit
                and self.max_physics_step_angular_jump <= self.physics_step_angular_jump_limit
            ),
            "NO_EXCESSIVE_PENETRATION": self.max_penetration <= self.penetration_limit,
            "NO_DROP": not self.dropped,
            "RIGHT_HAND_MULTI_FINGER_CONTACT": self.stages["GRASP"] is not None,
            "RIGHT_HAND_CONTACT_DURING_CARRY": bool(
                self.carry_contact_frames > 0
                and not self.carry_contact_lost
                and self.carry_contact_frames / max(1, self.carry_physics_steps)
                >= CARRY_CONTACT_FRACTION_MIN
            ),
            "ORDERED_TASK_STAGES": ordered_stages,
            "LEFT_ARM_CLEAR": not self.left_arm_contact_seen,
            "NO_RUNTIME_EQUALITY_CARRY": self.runtime_equality_count == 0,
            "TRUE_FINGER_OPEN_RELEASE": bool(
                self.stages["RELEASE"] is not None
                and final_sample
                and final_sample.get("fingers_open")
            ),
            "TARGET_TABLE_CONTACT": bool(final_sample and final_sample.get("target_contact")),
            "WHOLE_OBJECT_FOOTPRINT_INSIDE_TARGET": bool(
                (self.placement_footprint_check or self.last_footprint_check or {}).get(
                    "passed", False
                )
            ),
        }
        final = final_sample or {}
        final_position = final.get("position")
        return {
            "passed": bool(all(task_stages.values()) and all(safety.values())),
            "stage_order": [
                {"name": name, **(self.stages[name] or {})}
                for name in ("GRASP", "LIFT", "TRANSFER", "RELEASE", "PLACE")
            ],
            "task_stages": {
                name: {"passed": task_stages[name], **(sample or {})}
                for name, sample in self.stages.items()
            },
            "safety_checks": safety,
            "whole_object_footprint": self.placement_footprint_check
            or self.last_footprint_check,
            "metrics": {
                "control_frames": self.frame_count,
                "multi_finger_grasp_frames_max": self.max_grasp_contact_frames,
                "grasp_contact_frames_required": self.grasp_frames_required,
                "minimum_thumb_contact_force_n": self.minimum_thumb_force_n,
                "minimum_opposing_finger_force_n": self.minimum_opposing_force_n,
                "max_left_arm_force_n": self.max_left_arm_force_n,
                "max_right_hand_force_n": self.max_right_hand_force_n,
                "max_digit_force_n": dict(self.max_digit_force_n),
                "carry_contact_frames": self.carry_contact_frames,
                "carry_physics_steps": self.carry_physics_steps,
                "carry_contact_fraction": (
                    self.carry_contact_frames / self.carry_physics_steps
                    if self.carry_physics_steps
                    else 0.0
                ),
                "max_carry_contact_gap_physics_steps": self.max_carry_contact_gap_physics_steps,
                "carry_contact_lost": self.carry_contact_lost,
                "lift_height_required_m": self.lift_height,
                "minimum_stable_duration_s": self.minimum_stable_duration_s,
                "settling_frames_required": self.settling_frames_required,
                "settling_duration_required_s": (
                    self.settling_frames_required - 1
                ) * self.dt,
                "settling_frames_observed_max": self.max_settling_frames,
                "max_object_step_m": self.max_object_step,
                "teleport_step_limit_m": self.teleport_step_limit,
                "physics_steps_observed": self.physics_step_count,
                "max_physics_step_translation_m": self.max_physics_step_translation,
                "physics_step_translation_limit_m": self.physics_step_translation_limit,
                "max_physics_step_angular_jump_rad": self.max_physics_step_angular_jump,
                "physics_step_angular_jump_limit_rad": self.physics_step_angular_jump_limit,
                "max_penetration_m": self.max_penetration,
                "max_physics_step_penetration_m": self.max_physics_step_penetration,
                "penetration_limit_m": self.penetration_limit,
                "target_footprint_margin_m": self.target_margin,
                "linear_speed_limit_m_s": self.linear_speed_limit,
                "angular_speed_limit_rad_s": self.angular_speed_limit,
                "max_post_release_linear_speed_m_s": self.max_post_release_linear_speed,
                "max_post_release_angular_speed_rad_s": self.max_post_release_angular_speed,
                "final_position_m": None
                if final_position is None
                else np.asarray(final_position, dtype=float).tolist(),
                "final_quaternion_wxyz": None
                if final.get("quaternion_wxyz") is None
                else np.asarray(final["quaternion_wxyz"], dtype=float).tolist(),
                "final_target_xy_error_m": None
                if final_position is None
                else float(np.linalg.norm(np.asarray(final_position)[:2] - self.target_xy)),
                "final_sim_time_s": final.get("sim_time_s"),
                "final_linear_speed_m_s": final.get("linear_speed_m_s"),
                "final_angular_speed_rad_s": final.get("angular_speed_rad_s"),
                "final_upright": final.get("upright"),
                "final_target_contact": final.get("target_contact"),
                "final_left_arm_contact": final.get("left_contact"),
                "final_right_hand_contact": final.get("right_contact"),
                "final_digit_contacts": final.get("digit_contacts"),
                "final_fingers_open": final.get("fingers_open"),
                "failure_reasons": list(self.failures),
            },
        }


def _name_id(mujoco, model, kind, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, kind, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model has no named object {name!r}")
    return object_id


def load_upstream(candidate_root: Path):
    if not (candidate_root / ".git").exists():
        raise FileNotFoundError(f"Pinned Humanoid VLA checkout not found: {candidate_root}")
    import mujoco

    if mujoco.__version__ != MUJOCO_VERSION or mujoco.mj_versionString() != MUJOCO_VERSION:
        raise RuntimeError(
            f"Expected MuJoCo {MUJOCO_VERSION}; found Python {mujoco.__version__} "
            f"and native {mujoco.mj_versionString()}"
        )
    head = subprocess.run(
        ["git", "-C", str(candidate_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected Humanoid VLA {UPSTREAM_COMMIT}; found {head}")

    sys.path.insert(0, str(candidate_root / "scripts"))
    from physics_sim import LEFT_ARM_CTRL, RIGHT_ARM_CTRL, PhysicsSim

    return mujoco, PhysicsSim, LEFT_ARM_CTRL.copy(), RIGHT_ARM_CTRL.copy()


def _configure_hand_controller(mujoco, PhysicsSim, model) -> dict[str, int]:
    controller_globals = PhysicsSim._compute_pd_torques.__globals__
    old_kp = np.asarray(controller_globals["_KP"], dtype=float)
    old_kd = np.asarray(controller_globals["_KD"], dtype=float)
    if old_kp.shape != (29,) or old_kd.shape != (29,):
        raise RuntimeError("Pinned G1 controller no longer has the reviewed 29-actuator mapping")

    base_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(29)
    ]
    expected_body_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
        "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
        "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
        "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
        "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
        "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
        "right_wrist_pitch", "right_wrist_yaw",
    ]
    if base_names != expected_body_names:
        raise RuntimeError("Dex3 integration changed one or more G1 body actuator indices")

    hand_actuators = {}
    for name in RIGHT_HAND_JOINT_NAMES:
        actuator_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name != name:
            raise RuntimeError(f"Dex3 actuator {name!r} resolves to unexpected joint {joint_name!r}")
        hand_actuators[name] = actuator_id
    if model.nu != 36 or sorted(hand_actuators.values()) != list(range(29, 36)):
        raise RuntimeError("Model must contain 29 preserved G1 actuators plus seven appended Dex3 actuators")

    controller_globals["NUM_ACTUATORS"] = 36
    controller_globals["_ACTUATED_DOF_END"] = 42
    controller_globals["_KP"] = np.concatenate((old_kp, np.full(7, HAND_POSITION_KP)))
    controller_globals["_KD"] = np.concatenate((old_kd, np.full(7, HAND_VELOCITY_KD)))
    return hand_actuators


def _contact_sample(mujoco, sim, geometry: dict) -> dict:
    data = sim.data
    object_position = np.asarray(data.xpos[geometry["object_body"]], dtype=float).copy()
    qvel = np.asarray(data.qvel[geometry["object_qvel_adr"] : geometry["object_qvel_adr"] + 6])
    qpos_start = geometry["object_qpos_adr"]
    quaternion = np.asarray(data.qpos[qpos_start + 3 : qpos_start + 7], dtype=float).copy()
    object_rotation = np.asarray(data.xmat[geometry["object_body"]]).reshape(3, 3)
    left_contact = False
    right_contact = False
    left_force = 0.0
    right_force = 0.0
    digit_contacts = {name: False for name in RIGHT_HAND_GROUPS}
    digit_forces = {name: 0.0 for name in RIGHT_HAND_GROUPS}
    digit_contact_geometry = {name: [] for name in RIGHT_HAND_GROUPS}
    target_contact = False
    contact_distances = []

    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geometry["object_geoms"] or second in geometry["object_geoms"]:
            contact_distances.append(float(contact.dist))
            other = second if first in geometry["object_geoms"] else first
            if other == geometry["target_geom"]:
                target_contact = True
            digit_group = next(
                (name for name, geom_ids in geometry["right_digit_geoms"].items() if other in geom_ids),
                None,
            )
            if other in geometry["left_arm_geoms"] or digit_group is not None or other in geometry["right_arm_geoms"]:
                force = np.zeros(6)
                mujoco.mj_contactForce(sim.model, data, index, force)
                force_n = float(np.linalg.norm(force[:3]))
                if other in geometry["left_arm_geoms"]:
                    left_contact = True
                    left_force += force_n
                elif other in geometry["right_arm_geoms"]:
                    right_contact = True
                    right_force += force_n
                    if digit_group is not None:
                        digit_contacts[digit_group] = True
                        digit_forces[digit_group] += force_n
                        radial_xy = np.asarray(contact.pos[:2], dtype=float) - object_position[:2]
                        radial_norm = float(np.linalg.norm(radial_xy))
                        digit_contact_geometry[digit_group].append(
                            {
                                "position_m": np.asarray(contact.pos, dtype=float).tolist(),
                                "radial_xy_unit": (
                                    (radial_xy / radial_norm).tolist()
                                    if radial_norm > 1e-9
                                    else None
                                ),
                            }
                        )

    hand_joint_positions = {
        name: float(data.qpos[geometry["hand_qpos_addresses"][name]])
        for name in RIGHT_HAND_JOINT_NAMES
    }
    hand_qpos = np.asarray(list(hand_joint_positions.values()), dtype=float)
    fingers_open = bool(
        np.all(np.isfinite(hand_qpos))
        and np.allclose(hand_qpos, RIGHT_HAND_OPEN_POS, atol=0.12, rtol=0.0)
    )

    state = np.concatenate((data.qpos, data.qvel, data.qacc, object_position))
    state_extent = float(np.max(np.abs(state))) if state.size else 0.0
    hand_distance = float(np.linalg.norm(sim.right_hand_pos - object_position))
    return {
        "sim_time_s": float(data.time),
        "position": object_position,
        "quaternion_wxyz": quaternion,
        "upright": bool(float(object_rotation[2, 2]) >= 0.95),
        "linear_speed_m_s": float(np.linalg.norm(qvel[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
        "left_contact": left_contact,
        "right_contact": right_contact,
        "digit_contacts": digit_contacts,
        "digit_forces_n": digit_forces,
        "digit_contact_geometry": digit_contact_geometry,
        "fingers_open": fingers_open,
        "hand_joint_positions_rad": hand_joint_positions,
        "right_hand_site_position_m": np.asarray(sim.right_hand_pos, dtype=float).copy(),
        "right_hand_site_rotation": np.asarray(
            data.site_xmat[sim.right_hand_site_id], dtype=float
        ).reshape(3, 3).copy(),
        "left_force_n": left_force,
        "right_force_n": right_force,
        "hand_distance_m": hand_distance,
        "target_contact": target_contact,
        "minimum_contact_distance_m": min(contact_distances) if contact_distances else math.inf,
        "footprint_check": _object_footprint(mujoco, sim, geometry),
        "state_finite": bool(np.all(np.isfinite(state))),
        "state_extent": state_extent,
    }


def _geometry(mujoco, sim) -> dict:
    model = sim.model
    object_body = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    joint = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT_NAME)
    target_geom = _name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "m0_target_table_top"
    )
    target_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "m0_target_site")
    geom_shapes = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    }
    object_geoms = {
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name): name
        for name in OBJECT_COLLISION_GEOMS
    }
    right_digit_geoms = {name: set() for name in RIGHT_HAND_GROUPS}
    left_arm_geoms = set()
    right_arm_geoms = set()
    body_names = {}
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = body_names.setdefault(
            body_id,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "",
        )
        if body_name.startswith(("left_shoulder_", "left_elbow_", "left_wrist_")):
            left_arm_geoms.add(geom_id)
        if body_name.startswith(("right_shoulder_", "right_elbow_", "right_wrist_", "right_hand_")):
            right_arm_geoms.add(geom_id)
        for group in RIGHT_HAND_GROUPS:
            if body_name.startswith(f"right_hand_{group}_"):
                right_digit_geoms[group].add(geom_id)
    hand_qpos_addresses = {}
    for name in RIGHT_HAND_JOINT_NAMES:
        joint_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        hand_qpos_addresses[name] = int(model.jnt_qposadr[joint_id])
    return {
        "object_body": object_body,
        "object_qpos_adr": int(model.jnt_qposadr[joint]),
        "object_qvel_adr": int(model.jnt_dofadr[joint]),
        "object_geoms": object_geoms,
        "geom_shapes": geom_shapes,
        "target_geom": target_geom,
        "target_center": np.asarray(sim.data.site_xpos[target_site][:2], dtype=float).copy(),
        "target_half_extents": np.asarray(model.geom_size[target_geom][:2], dtype=float).copy(),
        "right_digit_geoms": right_digit_geoms,
        "left_arm_geoms": left_arm_geoms,
        "right_arm_geoms": right_arm_geoms,
        "hand_qpos_addresses": hand_qpos_addresses,
        "runtime_equality_count": int(model.neq),
    }


def _object_footprint(mujoco, sim, geometry: dict) -> dict:
    projected_geoms = []
    for geom_id, name in geometry["object_geoms"].items():
        shape = geometry["geom_shapes"].get(int(sim.model.geom_type[geom_id]))
        if shape is None:
            raise ValueError(f"unsupported bottle geom type for {name}")
        rotation = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        projected_geoms.append(
            {
                "name": name,
                "center_xy_m": np.asarray(sim.data.geom_xpos[geom_id][:2], dtype=float),
                "half_extents_xy_m": _projected_geom_half_extents_xy(
                    shape, rotation, sim.model.geom_size[geom_id]
                ),
            }
        )
    return check_projected_footprint(
        projected_geoms,
        geometry["target_center"],
        geometry["target_half_extents"],
        TARGET_MARGIN_M,
    )


def _install_physics_step_recorder(
    mujoco, sim, geometry: dict, monitor: AcceptanceMonitor, trace_path: Path
):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = trace_path.open("w", encoding="utf-8")
    original_mj_step = mujoco.mj_step

    def recorded_mj_step(model, data, *args, **kwargs):
        result = original_mj_step(model, data, *args, **kwargs)
        if model is sim.model and data is sim.data:
            object_id = geometry["object_body"]
            position = np.asarray(data.xpos[object_id], dtype=float).copy()
            quaternion = np.asarray(data.xquat[object_id], dtype=float).copy()
            contact_distances = []
            digit_contacts = {name: False for name in RIGHT_HAND_GROUPS}
            digit_forces_n = {name: 0.0 for name in RIGHT_HAND_GROUPS}
            left_arm_contact = False
            for index in range(data.ncon):
                contact = data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first not in geometry["object_geoms"] and second not in geometry["object_geoms"]:
                    continue
                contact_distances.append(float(contact.dist))
                other = second if first in geometry["object_geoms"] else first
                left_arm_contact |= other in geometry["left_arm_geoms"]
                for group, geom_ids in geometry["right_digit_geoms"].items():
                    if other in geom_ids:
                        digit_contacts[group] = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        digit_forces_n[group] += float(np.linalg.norm(force[:3]))
            minimum_distance = min(contact_distances) if contact_distances else math.inf
            monitor.record_physics_step(
                sim_time=float(data.time),
                position=position,
                quaternion_wxyz=quaternion,
                minimum_contact_distance=minimum_distance,
                left_arm_contact=left_arm_contact,
                digit_contacts=digit_contacts,
                digit_forces_n=digit_forces_n,
            )
            trace.write(
                json.dumps(
                    {
                        "sim_time_s": float(data.time),
                        "position_m": position.tolist()
                        if np.all(np.isfinite(position))
                        else None,
                        "quaternion_wxyz": quaternion.tolist()
                        if np.all(np.isfinite(quaternion))
                        else None,
                        "minimum_contact_distance_m": minimum_distance
                        if math.isfinite(minimum_distance)
                        else None,
                        "object_penetration_m": max(0.0, -minimum_distance)
                        if math.isfinite(minimum_distance)
                        else 0.0,
                        "left_arm_contact": left_arm_contact,
                        "right_digit_contacts": digit_contacts,
                        "right_digit_forces_n": digit_forces_n,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        return result

    mujoco.mj_step = recorded_mj_step

    def close() -> None:
        mujoco.mj_step = original_mj_step
        trace.close()

    return close


def _plan_right_hand_target(
    sim,
    target_hand_position: np.ndarray,
    target_hand_rotation: np.ndarray,
) -> np.ndarray:
    import mujoco

    saved_right_q = sim.right_arm_q
    target_position = np.asarray(target_hand_position, dtype=float)
    target_rotation = np.asarray(target_hand_rotation, dtype=float).reshape(3, 3)
    joint_ids = [
        int(np.flatnonzero(sim.model.jnt_qposadr == address)[0])
        for address in sim.right_arm_qpos_adr
    ]
    dof_ids = np.asarray(sim.model.jnt_dofadr[joint_ids], dtype=int)
    site_id = sim.right_hand_site_id
    orientation_weight = 0.25
    position_error = np.zeros(3)
    rotation_error = np.zeros(3)

    for _ in range(700):
        mujoco.mj_forward(sim.model, sim.data)
        position_error = target_position - sim.data.site_xpos[site_id]
        current_rotation = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(current_rotation[:, axis], target_rotation[:, axis])
            for axis in range(3)
        )
        if np.linalg.norm(position_error) <= 0.02 and np.linalg.norm(rotation_error) <= 0.06:
            target = sim.right_arm_q
            break

        jacobian_position = np.zeros((3, sim.model.nv))
        jacobian_rotation = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(
            sim.model, sim.data, jacobian_position, jacobian_rotation, site_id
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, dof_ids],
                jacobian_rotation[:, dof_ids] * orientation_weight,
            )
        )
        error = np.concatenate((position_error, rotation_error * orientation_weight))
        damping = 0.05
        delta_q = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        delta_norm = float(np.linalg.norm(delta_q))
        if delta_norm > 0.02:
            delta_q *= 0.02 / delta_norm
        sim.data.qpos[sim.right_arm_qpos_adr] = np.clip(
            sim.data.qpos[sim.right_arm_qpos_adr] + delta_q,
            sim.right_arm_lo,
            sim.right_arm_hi,
        )
    else:
        sim.data.qpos[sim.right_arm_qpos_adr] = saved_right_q
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            "Right arm IK cannot reach pose "
            f"{target_position.tolist()} (position error {np.linalg.norm(position_error):.3f} m, "
            f"orientation error {np.linalg.norm(rotation_error):.3f} rad)"
        )

    sim.data.qpos[sim.right_arm_qpos_adr] = saved_right_q
    mujoco.mj_forward(sim.model, sim.data)
    return target


def _plan_left_hand_target(sim, target_hand_position: np.ndarray) -> np.ndarray:
    import mujoco

    saved_left_q = sim.left_arm_q
    if not sim.solve_ik_left(np.asarray(target_hand_position, dtype=float)):
        hand_position = sim.left_hand_pos.tolist()
        sim.data.qpos[sim.left_arm_qpos_adr] = saved_left_q
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            f"Left arm IK cannot reach hand waypoint {target_hand_position}; "
            f"current hand site is {hand_position}"
        )
    target = sim.left_arm_q
    sim.data.qpos[sim.left_arm_qpos_adr] = saved_left_q
    mujoco.mj_forward(sim.model, sim.data)
    return target


def _interpolate(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    return np.linspace(np.asarray(start), np.asarray(end), frames + 1)[1:]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def run_demo(args: argparse.Namespace) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2

    _verify_unitree_mesh_inputs(args)
    mujoco, PhysicsSim, left_ctrl, right_ctrl = load_upstream(args.candidate_root)
    scene_path = prepare_scene(
        args.candidate_root,
        args.mesh_dir,
        args.dex3_root,
        args.output_dir / "model",
    )
    compiled_model = mujoco.MjModel.from_xml_path(str(scene_path))
    hand_actuators = _configure_hand_controller(mujoco, PhysicsSim, compiled_model)
    sim = PhysicsSim(model_path=str(scene_path))
    geometry = _geometry(mujoco, sim)
    frame_dt = float(sim.model.opt.timestep * (500 // 30))
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    if geometry["runtime_equality_count"] != 0:
        raise RuntimeError("M0 single-hand rollout must not contain runtime equality constraints")

    initial_position = np.asarray(sim.box_pos, dtype=float).copy()
    object_qpos = geometry["object_qpos_adr"]
    initial_quaternion = np.asarray(
        sim.data.qpos[object_qpos + 3 : object_qpos + 7], dtype=float
    ).copy()
    monitor = AcceptanceMonitor(
        initial_position=initial_position,
        target_xy=geometry["target_center"],
        target_half_extents=geometry["target_half_extents"],
        dt=frame_dt,
        target_margin=TARGET_MARGIN_M,
        runtime_equality_count=geometry["runtime_equality_count"],
    )
    run_identity = _runtime_identity(args, monitor)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / frame_dt, (640, 480)
    )
    if not video.isOpened():
        sim.renderer.close()
        raise RuntimeError(f"Could not create demo video: {args.video}")

    hand_ctrl = np.asarray([hand_actuators[name] for name in RIGHT_HAND_JOINT_NAMES], dtype=int)
    open_hand = np.asarray(RIGHT_HAND_OPEN_POS, dtype=float)
    closed_hand = np.asarray(RIGHT_HAND_CLOSED_POS, dtype=float)
    sim.target_pos[:] = sim.data.actuator_length.copy()
    sim.target_pos[hand_ctrl] = open_hand
    object_center = initial_position.copy()
    hand_roll = math.radians(args.hand_roll_deg)
    hand_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(hand_roll), -math.sin(hand_roll)],
            [0.0, math.sin(hand_roll), math.cos(hand_roll)],
        ],
        dtype=float,
    )
    approach_site = object_center + np.asarray(APPROACH_SITE_OFFSET_M, dtype=float)
    # Keep thumb and opposing fingers on the straight cylindrical body.
    grasp_site = object_center + np.asarray(GRASP_SITE_OFFSET_M, dtype=float)
    clear_left = _plan_left_hand_target(sim, np.array([0.23, 0.34, 0.97]))
    right_approach = _plan_right_hand_target(sim, approach_site, hand_rotation)

    controller_stages: dict[str, dict[str, float | int]] = {}
    post_release_trace: list[dict[str, object]] = []
    waypoint_checks: dict[str, dict[str, object]] = {}
    trace_start_frame: int | None = None
    final_sample = None
    frame_index = 0

    def record_post_release_sample(sample: dict) -> None:
        if trace_start_frame is None or frame_index - trace_start_frame > 90:
            return
        post_release_trace.append(
            {
                "control_frame": frame_index,
                "sim_time_s": sample["sim_time_s"],
                "position_m": sample["position"].tolist(),
                "quaternion_wxyz": sample["quaternion_wxyz"].tolist(),
                "linear_speed_m_s": sample["linear_speed_m_s"],
                "angular_speed_rad_s": sample["angular_speed_rad_s"],
                "upright": sample["upright"],
                "target_contact": sample["target_contact"],
                "left_contact": sample["left_contact"],
                "right_contact": sample["right_contact"],
                "digit_contacts": sample["digit_contacts"],
                "fingers_open": sample["fingers_open"],
            }
        )

    def observe(label: str | None = None) -> dict:
        nonlocal frame_index, final_sample, trace_start_frame
        sample = _contact_sample(mujoco, sim, geometry)
        frame_index += 1
        release_before = monitor.stages["RELEASE"]
        monitor.update(
            sim_time=sample["sim_time_s"],
            position=sample["position"],
            quaternion_wxyz=sample["quaternion_wxyz"],
            upright=sample["upright"],
            left_contact=sample["left_contact"],
            right_contact=sample["right_contact"],
            digit_contacts=sample["digit_contacts"],
            digit_forces_n=sample["digit_forces_n"],
            left_force_n=sample["left_force_n"],
            right_force_n=sample["right_force_n"],
            fingers_open=sample["fingers_open"],
            target_contact=sample["target_contact"],
            linear_speed=sample["linear_speed_m_s"],
            angular_speed=sample["angular_speed_rad_s"],
            minimum_contact_distance=sample["minimum_contact_distance_m"],
            state_finite=sample["state_finite"],
            state_extent=sample["state_extent"],
            footprint_check=sample["footprint_check"],
        )
        if release_before is None and monitor.stages["RELEASE"] is not None:
            trace_start_frame = frame_index
            controller_stages["RELEASE"] = {
                "sim_time_s": sample["sim_time_s"],
                "control_frame": frame_index,
            }
        record_post_release_sample(sample)
        sim.renderer.update_scene(sim.data, camera="scene_camera")
        video.write(cv2.cvtColor(sim.renderer.render().copy(), cv2.COLOR_RGB2BGR))
        final_sample = sample
        if label:
            controller_stages[label] = {
                "sim_time_s": sample["sim_time_s"],
                "control_frame": frame_index,
            }
            print(
                f"{label}: sim_time={sample['sim_time_s']:.3f}s "
                f"bottle_xyz={np.round(sample['position'], 3).tolist()} "
                f"digits={sample['digit_contacts']} left={sample['left_contact']}",
                flush=True,
            )
        for name, stage in monitor.stages.items():
            if stage is not None and name not in getattr(observe, "announced", set()):
                announced = getattr(observe, "announced", set())
                announced.add(name)
                setattr(observe, "announced", announced)
                print(
                    f"ACCEPT_{name}: sim_time={stage['sim_time_s']:.3f}s "
                    f"digits={sample['digit_contacts']} forces={sample['digit_forces_n']} "
                    f"hand_q={sample['hand_joint_positions_rad']}",
                    flush=True,
                )
        if frame_index % 50 == 0:
            print(
                f"frame={frame_index} sim_time={sample['sim_time_s']:.3f}s "
                f"digits={sample['digit_contacts']} "
                f"forces={np.round([sample['digit_forces_n'][key] for key in RIGHT_HAND_GROUPS], 2).tolist()} "
                f"hand_q={np.round(list(sample['hand_joint_positions_rad'].values()), 2).tolist()} "
                f"left={sample['left_contact']} target={sample['target_contact']}",
                flush=True,
            )
        if frame_index % 25 == 0:
            checkpoint = monitor.result(sample)
            checkpoint.update(run_identity)
            checkpoint.update(
                {
                    "partial_run": True,
                    "state": "RUNNING",
                    "stage": "SIMULATION",
                    "complete": False,
                    "controller_stages": controller_stages,
                    "waypoint_checks": waypoint_checks,
                    "controller_parameters": {
                        "hand_roll_deg": args.hand_roll_deg,
                        "grasp_site_offset_m": list(GRASP_SITE_OFFSET_M),
                        "grasp_hold_frames": args.grasp_hold_frames,
                        "grasp_preload_rad": GRASP_PRELOAD_RAD,
                        "lift_frames": args.lift_frames,
                        "commanded_lift_height_m": args.lift_height_m,
                        "transfer_frames": args.transfer_frames,
                        "lower_frames": args.lower_frames,
                    },
                    "last_sample": _json_value(sample),
                }
            )
            args.output_json.write_text(
                json.dumps(checkpoint, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        return sample

    def move_group(
        indices: np.ndarray,
        target: np.ndarray,
        frames: int,
        *,
        stop_on_grasp: bool = False,
    ) -> None:
        indices = np.asarray(indices, dtype=int)
        current = sim.target_pos[indices].copy()
        for values in _interpolate(current, np.asarray(target, dtype=float), frames):
            sim.target_pos[indices] = values
            sim.step_frame()
            observe()
            if stop_on_grasp and monitor.stages["GRASP"] is not None:
                break
            if monitor.carry_contact_lost:
                raise RuntimeError("Right Dex3 multi-finger contact was lost during lift/carry")

    def hold_group(indices: np.ndarray, target: np.ndarray, frames: int) -> None:
        for _ in range(frames):
            sim.target_pos[indices] = target
            sim.step_frame()
            observe()
            if monitor.carry_contact_required and monitor.carry_contact_lost:
                raise RuntimeError("Right Dex3 multi-finger contact was lost during lift/carry")

    start_wall = time.monotonic()
    physics_trace_path = args.output_json.parent / "m0_physics_trace.jsonl"
    close_physics_recorder = None
    try:
        close_physics_recorder = _install_physics_step_recorder(
            mujoco, sim, geometry, monitor, physics_trace_path
        )
        observe("START")
        move_group(left_ctrl, clear_left, max(35, args.approach_frames))
        controller_stages["LEFT_ARM_CLEAR"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        move_group(right_ctrl, right_approach, args.approach_frames)
        controller_stages["APPROACH"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        right_grasp = _plan_right_hand_target(sim, grasp_site, hand_rotation)
        move_group(right_ctrl, right_grasp, max(25, args.close_frames // 2))
        controller_stages["HAND_AROUND_BOTTLE"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        move_group(hand_ctrl, closed_hand, args.close_frames, stop_on_grasp=True)
        grasp_sample = final_sample
        if monitor.stages["GRASP"] is None or grasp_sample is None:
            raise RuntimeError(
                "Dex3 did not establish sustained thumb-plus-opposing-finger contact; "
                f"contacts={None if grasp_sample is None else grasp_sample['digit_contacts']} "
                f"forces={None if grasp_sample is None else grasp_sample['digit_forces_n']}"
            )
        thumb_holding = bool(
            grasp_sample["digit_contacts"].get("thumb", False)
            and grasp_sample["digit_forces_n"].get("thumb", 0.0)
            >= monitor.minimum_thumb_force_n
        )
        opposing_holding = any(
            grasp_sample["digit_contacts"].get(group, False)
            and grasp_sample["digit_forces_n"].get(group, 0.0)
            >= monitor.minimum_opposing_force_n
            for group in ("index", "middle")
        )
        if not thumb_holding or not opposing_holding:
            raise RuntimeError(
                "Dex3 multi-finger contact was not present at lift start; "
                f"contacts={grasp_sample['digit_contacts']} "
                f"forces={grasp_sample['digit_forces_n']}"
            )
        monitor.carry_contact_required = True
        grasp_hold_target = _grasp_hold_target(
            grasp_sample["hand_joint_positions_rad"]
        )
        hold_group(hand_ctrl, grasp_hold_target, args.grasp_hold_frames)
        grasp_sample = final_sample
        if grasp_sample is None:
            raise RuntimeError("Dex3 grasp hold produced no physics sample")
        thumb_holding = bool(
            grasp_sample["digit_contacts"].get("thumb", False)
            and grasp_sample["digit_forces_n"].get("thumb", 0.0)
            >= monitor.minimum_thumb_force_n
        )
        opposing_holding = any(
            grasp_sample["digit_contacts"].get(group, False)
            and grasp_sample["digit_forces_n"].get(group, 0.0)
            >= monitor.minimum_opposing_force_n
            for group in ("index", "middle")
        )
        if not thumb_holding or not opposing_holding:
            raise RuntimeError(
                "Dex3 lost multi-finger contact during the pre-lift hold; "
                f"contacts={grasp_sample['digit_contacts']} "
                f"forces={grasp_sample['digit_forces_n']}"
            )
        waypoint_checks["GRASP_END"] = {
            "object_position_m": grasp_sample["position"].tolist(),
            "right_hand_site_position_m": grasp_sample["right_hand_site_position_m"].tolist(),
            "right_hand_site_rotation": grasp_sample["right_hand_site_rotation"].tolist(),
            "hand_joint_positions_rad": grasp_sample["hand_joint_positions_rad"],
            "right_digit_contacts": grasp_sample["digit_contacts"],
            "right_digit_forces_n": grasp_sample["digit_forces_n"],
            "right_digit_contact_geometry": grasp_sample["digit_contact_geometry"],
            "grasp_hold_target_rad": grasp_hold_target.tolist(),
            "grasp_preload_rad": GRASP_PRELOAD_RAD,
        }
        hand_rotation = grasp_sample["right_hand_site_rotation"]
        controller_stages["GRASP"] = {
            "sim_time_s": monitor.stages["GRASP"]["sim_time_s"],
            "control_frame": monitor.stages["GRASP"]["control_frame"],
        }

        # Contact force can deflect the arm from its previous position target.
        # Begin the lift interpolation from the measured pose to avoid pulling
        # the fingers away from the established grasp on the first control step.
        sim.target_pos[right_ctrl] = sim.right_arm_q
        current_site = np.asarray(sim.right_hand_pos, dtype=float)
        lift_target = _plan_right_hand_target(
            sim, current_site + np.array([0.0, 0.0, args.lift_height_m]), hand_rotation
        )
        move_group(right_ctrl, lift_target, args.lift_frames)
        lift_sample = _contact_sample(mujoco, sim, geometry)
        if monitor.carry_contact_lost:
            raise RuntimeError(
                "Right Dex3 thumb-plus-opposing-finger contact was lost during lift; "
                f"contacts={lift_sample['digit_contacts']} forces={lift_sample['digit_forces_n']}"
            )
        waypoint_checks["LIFT_END"] = {
            "object_position_m": lift_sample["position"].tolist(),
            "right_hand_site_position_m": lift_sample["right_hand_site_position_m"].tolist(),
            "right_hand_site_rotation": lift_sample["right_hand_site_rotation"].tolist(),
            "hand_joint_positions_rad": lift_sample["hand_joint_positions_rad"],
            "right_digit_contacts": lift_sample["digit_contacts"],
            "right_digit_forces_n": lift_sample["digit_forces_n"],
            "lift_height_m": float(lift_sample["position"][2] - initial_position[2]),
        }
        if monitor.stages["LIFT"] is None:
            raise RuntimeError("Bottle did not rise at least 0.05 m while held by right Dex3 contact")

        current_object = lift_sample["position"]
        target_delta = np.array(
            [geometry["target_center"][0] - current_object[0],
             geometry["target_center"][1] - current_object[1], 0.0],
            dtype=float,
        )
        sim.target_pos[right_ctrl] = sim.right_arm_q
        transfer_target = _plan_right_hand_target(
            sim, np.asarray(sim.right_hand_pos, dtype=float) + target_delta, hand_rotation
        )
        move_group(right_ctrl, transfer_target, args.transfer_frames)
        transfer_sample = _contact_sample(mujoco, sim, geometry)
        controller_stages["TRANSFER"] = {
            "sim_time_s": transfer_sample["sim_time_s"],
            "control_frame": frame_index,
        }
        waypoint_checks["TRANSFER_END"] = {
            "object_position_m": transfer_sample["position"].tolist(),
            "right_digit_contacts": transfer_sample["digit_contacts"],
            "right_digit_forces_n": transfer_sample["digit_forces_n"],
            "footprint": transfer_sample["footprint_check"],
        }
        if monitor.stages["TRANSFER"] is None:
            raise RuntimeError("Right Dex3 contact did not transfer the bottle into the target table region")

        place_center_z = TARGET_TABLE_TOP_Z + 0.078
        lower_delta_z = place_center_z - float(transfer_sample["position"][2])
        sim.target_pos[right_ctrl] = sim.right_arm_q
        lower_target = _plan_right_hand_target(
            sim,
            np.asarray(sim.right_hand_pos, dtype=float) + np.array([0.0, 0.0, lower_delta_z]),
            hand_rotation,
        )
        move_group(right_ctrl, lower_target, args.lower_frames)
        hold_group(right_ctrl, lower_target, 5)
        lower_sample = _contact_sample(mujoco, sim, geometry)
        waypoint_checks["LOWER_END"] = {
            "object_position_m": lower_sample["position"].tolist(),
            "target_table_contact": lower_sample["target_contact"],
            "footprint": lower_sample["footprint_check"],
        }

        monitor.carry_contact_required = False
        move_group(hand_ctrl, open_hand, args.open_frames)
        controller_stages["FINGER_OPEN"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        retreat_target = _plan_right_hand_target(
            sim,
            np.asarray(sim.right_hand_pos, dtype=float) + np.array([-0.12, 0.0, 0.10]),
            hand_rotation,
        )
        move_group(right_ctrl, retreat_target, args.retreat_frames)
        controller_stages["RETREAT"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        for _ in range(args.settle_frames):
            sim.target_pos[right_ctrl] = retreat_target
            sim.target_pos[hand_ctrl] = open_hand
            sim.step_frame()
            observe()
            if monitor.stages["PLACE"] is not None:
                break
    finally:
        if close_physics_recorder is not None:
            close_physics_recorder()
        video.release()
        if final_sample is not None:
            sim.renderer.update_scene(sim.data, camera="scene_camera")
            final_rgb = sim.renderer.render().copy()
            cv2.imwrite(str(args.screenshot), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        sim.renderer.close()

    result = monitor.result(final_sample)
    result.update(
        {
            "issue": 43,
            "demo": "M0 fixed-base G1, one right Dex3 hand, normal water bottle",
            **_runtime_identity(args, monitor),
            "object_geometry_parameters": [
                {
                    "name": spec["name"],
                    "shape": spec["type"],
                    "position_m": [float(value) for value in spec["pos"].split()],
                    "size": [float(value) for value in spec["size"].split()],
                    "mass_kg": float(spec["mass"]),
                }
                for spec in OBJECT_GEOM_SPECS
            ],
            "robot_hand": {
                "model": "Unitree Dex3-1 Rev 1.0 right hand",
                "parent_body": "right_wrist_yaw_link",
                "joint_names": list(RIGHT_HAND_JOINT_NAMES),
                "actuator_ids_by_joint_name": hand_actuators,
                "g1_body_actuator_count": 29,
                "total_actuator_count": 36,
                "left_articulated_hand": False,
                "palm_roll_deg": args.hand_roll_deg,
                "grasp_site_offset_m": list(GRASP_SITE_OFFSET_M),
                "grasp_hold_frames": args.grasp_hold_frames,
                "grasp_preload_rad": GRASP_PRELOAD_RAD,
                "position_kp": HAND_POSITION_KP,
                "velocity_kd": HAND_VELOCITY_KD,
                "grasp_mode": "measured multi-finger contact and friction only",
            },
            "acceptance_thresholds": _acceptance_thresholds(monitor),
            "candidate": {
                "repository": "https://github.com/ozkannceylan/humanoid_vla",
                "commit": UPSTREAM_COMMIT,
                "license": "MIT",
                "unitree_mesh_repository": "https://github.com/unitreerobotics/unitree_mujoco",
                "unitree_mesh_commit": UNITREE_COMMIT,
                "dex3_repository": "https://github.com/unitreerobotics/unitree_ros",
                "dex3_commit": DEX3_COMMIT,
                "dex3_license": "BSD-3-Clause",
                "mujoco_python_version": MUJOCO_VERSION,
                "controller": "pinned G1 PD torque controller with name-resolved right Dex3 joint targets",
                "physics": "MuJoCo mj_step, 500 Hz physics substeps, fixed base",
                "grasp": "right Dex3 thumb plus index and/or middle physical friction contact; no runtime equality and no object qpos writes",
            },
            "run": {
                "seed": args.seed,
                "simulation_step_s": float(sim.model.opt.timestep),
                "control_substeps": 500 // 30,
                "control_step_s": frame_dt,
                "wall_duration_s": time.monotonic() - start_wall,
                "initial_position_m": initial_position.tolist(),
                "initial_quaternion_wxyz": initial_quaternion.tolist(),
                "controller_stages": controller_stages,
                "post_release_trace": post_release_trace,
                "waypoint_checks": waypoint_checks,
                "source_table_top_z_m": SOURCE_TABLE_TOP_Z,
                "target_table_top_z_m": TARGET_TABLE_TOP_Z,
                "target_xy_m": geometry["target_center"].tolist(),
                "target_table_body_xy_m": TARGET_TABLE_BODY_XY.tolist(),
                "target_table_gap_m": TABLE_GAP_M,
                "hand_roll_deg": args.hand_roll_deg,
                "commanded_lift_height_m": args.lift_height_m,
                "transfer_frames": args.transfer_frames,
                "lower_frames": args.lower_frames,
                "target_half_extents_m": geometry["target_half_extents"].tolist(),
                "bottle_collision_geoms": list(OBJECT_COLLISION_GEOMS),
                "bottle_total_mass_kg": sum(float(spec["mass"]) for spec in OBJECT_GEOM_SPECS),
                "bottle_body_diameter_m": 2.0 * float(OBJECT_GEOM_SPECS[0]["size"].split()[0]),
                "bottle_total_height_m": 0.231,
                "runtime_equality_count": geometry["runtime_equality_count"],
                "object_qpos_address": geometry["object_qpos_adr"],
                "object_qpos_written_after_reset": False,
            },
            "artifacts": {
                "result_json": str(args.output_json),
                "log": str(args.output_json.parent / "run.log"),
                "scene_xml": str(scene_path),
                "video": str(args.video),
                "screenshot": str(args.screenshot),
                "physics_step_trace": str(physics_trace_path),
            },
        }
    )
    args.output_json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--unitree-root", type=Path, required=True)
    parser.add_argument("--dex3-root", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--approach-frames", type=int, default=60)
    parser.add_argument("--close-frames", type=int, default=60)
    parser.add_argument("--lift-height-m", type=float, default=DEFAULT_LIFT_HEIGHT_M)
    parser.add_argument("--hand-roll-deg", type=float, default=DEFAULT_HAND_ROLL_DEG)
    parser.add_argument("--grasp-hold-frames", type=int, default=DEFAULT_GRASP_HOLD_FRAMES)
    parser.add_argument("--lift-frames", type=int, default=DEFAULT_LIFT_FRAMES)
    parser.add_argument("--transfer-frames", type=int, default=DEFAULT_TRANSFER_FRAMES)
    parser.add_argument("--lower-frames", type=int, default=DEFAULT_LOWER_FRAMES)
    parser.add_argument("--open-frames", type=int, default=40)
    parser.add_argument("--retreat-frames", type=int, default=35)
    parser.add_argument("--settle-frames", type=int, default=200)
    args = parser.parse_args(argv)
    if abs(args.hand_roll_deg) > 90.0:
        parser.error("hand roll must be between -90 and 90 degrees")
    if min(
        args.approach_frames,
        args.close_frames,
        args.grasp_hold_frames,
        args.lift_height_m,
        args.lift_frames,
        args.transfer_frames,
        args.lower_frames,
        args.open_frames,
        args.retreat_frames,
        args.settle_frames,
    ) <= 0:
        parser.error("phase frame counts and lift height must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_id = args.run_id or uuid.uuid4().hex
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    identity = _runtime_identity(args)
    initial_result = {
        "issue": 43,
        "demo": "M0 G1 bottle pick-and-place",
        **identity,
        "passed": False,
        "complete": False,
        "state": "RUNNING",
        "stage": "RUNTIME_START",
        "controller_parameters": {
            "hand_roll_deg": args.hand_roll_deg,
            "grasp_site_offset_m": list(GRASP_SITE_OFFSET_M),
            "grasp_hold_frames": args.grasp_hold_frames,
            "grasp_preload_rad": GRASP_PRELOAD_RAD,
            "lift_frames": args.lift_frames,
            "commanded_lift_height_m": args.lift_height_m,
            "transfer_frames": args.transfer_frames,
            "lower_frames": args.lower_frames,
        },
    }
    args.output_json.write_text(
        json.dumps(initial_result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    try:
        result = run_demo(args)
    except Exception as exc:
        try:
            failure = json.loads(args.output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failure = initial_result
        failure.update(_runtime_identity(args))
        failure.update(
            {
                "issue": 43,
                "demo": "M0 G1 bottle pick-and-place",
                "passed": False,
                "complete": False,
                "state": "FAILED",
                "stage": "RUNTIME",
                "error": _safe_error(exc, args),
            }
        )
        args.output_json.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {failure['error']}", file=sys.stderr, flush=True)
        return 1
    result["state"] = "COMPLETE" if result["passed"] else "FAILED"
    result["stage"] = "ACCEPTANCE"
    result["complete"] = True
    result.setdefault("error", None if result["passed"] else "acceptance checks did not pass")
    args.output_json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.output_json}", flush=True)
    print(json.dumps(result["task_stages"], indent=2), flush=True)
    print(json.dumps(result["safety_checks"], indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
