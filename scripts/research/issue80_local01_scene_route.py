#!/usr/bin/env python3
"""Local01 RAICOM scene-pose adapter and static short-route preflight.

This module converts the pinned Local01 scene-world pose file into the
existing Issue #80 odom-route input. It does not publish commands or step
MuJoCo. The route targets are a 0.20 m + 0.20 m diagnostic straight ahead
from the configured START pose; they are not the table-docking target.
"""
from __future__ import annotations

import hashlib
import copy
from itertools import product
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from issue80_waypoint_route import (
    LOCAL01_SHORT_ROUTE_STEP_M,
    LOCAL01_SHORT_ROUTE_MAX_WINDOWS,
    LOCAL01_SHORT_ROUTE_MAX_WINDOW_S,
    LOCAL01_SHORT_ROUTE_MAX_AGGREGATE_S,
    LOCAL01_SHORT_ROUTE_FORWARD_M_S,
    LOCAL01_SHORT_ROUTE_ANGULAR_RAD_S,
    LOCAL01_SHORT_ROUTE_MAX_CROSS_TRACK_M,
)


LOCAL01_COMMIT = "95d76d82ea7def0dd719da4ebe07f9654044415e"
LOCAL01_POSES_RELATIVE = Path(
    "simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json"
)
LOCAL01_ROUTE_RELATIVE = Path(
    "simulation/mujoco/scenes/raicom2026/config/route_waypoints.json"
)
LOCAL01_SCENE_RELATIVE = Path(
    "simulation/mujoco/scenes/raicom2026/scene_workspace.xml"
)
LOCAL01_MUG_RELATIVE = Path(
    "simulation/mujoco/scenes/raicom2026/mug_portable.xml"
)
LOCAL01_POSES_SHA256 = "14640437e49953c6ba1b89f859a4d23f4204a9b665fcc3427e8f29c6961f1a19"
LOCAL01_ROUTE_SHA256 = "7575e61eaeff2e835f19b4e3b1a1a2d046765fcc4df0891f02fcaced0aa16c9e"
LOCAL01_SCENE_SHA256 = "a79504ed3f2fdc6fc4da5c2bad5dcfa947361625c7a79d1f6aab7c6258ab5ffe"
LOCAL01_MUG_SHA256 = "63e36f7733285ab3fe80c0149f2572f2d862c7b93c6dd49eb4a22dff4b1c1642"
LOCAL01_RUNTIME_X2_SHA256 = "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3"
LOCAL01_SCENE_REPO_RELATIVE = Path("reference/AgiBot/raicom2026-x2-mujoco-scene")
LOCAL01_ROOM_XML_SHA256 = "d72618e8d88f1c6417cc239bc3ae690068845665a5db01775ed24dd81edc6c0a"
LOCAL01_TABLE_MESH_SHA256 = "1467760febf4af9898e55ad3de30c45577d1f5d0e26ca525e85ced5f07035376"
LOCAL01_CHAIR_MESH_SHA256 = "e3b08cb3ed1bdda668a46c14ec241ecbbca3c6829869905bda61c56e891f519f"
LOCAL01_MUG_OBJ_SHA256 = "656bfc987a038c90417ec98c671ab481e6a3ebe35a17d3b92e2ec0ee9593eb19"
LOCAL01_MUG_PNG_SHA256 = "8bee0ad778cda4459b486a3fe97d49196d3583abf041020eb1a54ef840e95318"

SHORT_ROUTE_SCHEMA = "robotsim.issue80.local01-short-scene-route.v1"
TABLE_APPROACH_MISSION_SCHEMA = "robotsim.issue80.local01-table-approach-mission.v1"
ALIGNED_START_TABLE_APPROACH_MISSION_SCHEMA = (
    "robotsim.issue80.local01-aligned-start-table-approach-mission.v1"
)
ALIGNED_START_TABLE_APPROACH_CLASSIFICATION = (
    "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION"
)
ALIGNED_START_SCENE_VERSION = "aligned-start-v1"
NATIVE_X2_TIMESTEP_SCENE_VERSION = "native-x2-timestep-v1"
NATIVE_X2_TIMESTEP_S = 0.001
COMBINED_SCENE_DEFAULT_TIMESTEP_S = 0.002
SYNCHRONIZED_NOMINAL_KEYFRAME_VERSION = "nominal-keyframe-v1"
SYNCHRONIZED_NOMINAL_KEYFRAME_NAME = "issue80_nominal_stand_v1"
STATIC_NOMINAL_START_VERSION = "static-nominal-start-v1"
X2_START_FRAME_NAME = "x2_start_frame"
X2_SOURCE_START_POS_M = (-1.5, -1.5, 0.0)
X2_SOURCE_START_QUAT_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865476,
)
X2_ALIGNED_START_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
SHORT_ROUTE_STEP_M = LOCAL01_SHORT_ROUTE_STEP_M
SHORT_ROUTE_TARGETS_M = (SHORT_ROUTE_STEP_M, 2.0 * SHORT_ROUTE_STEP_M)
SHORT_ROUTE_MAX_WINDOWS = LOCAL01_SHORT_ROUTE_MAX_WINDOWS
SHORT_ROUTE_MAX_WINDOW_S = LOCAL01_SHORT_ROUTE_MAX_WINDOW_S
SHORT_ROUTE_MAX_AGGREGATE_S = LOCAL01_SHORT_ROUTE_MAX_AGGREGATE_S
SHORT_ROUTE_FORWARD_M_S = LOCAL01_SHORT_ROUTE_FORWARD_M_S
SHORT_ROUTE_ANGULAR_RAD_S = LOCAL01_SHORT_ROUTE_ANGULAR_RAD_S
SHORT_ROUTE_MAX_CROSS_TRACK_M = LOCAL01_SHORT_ROUTE_MAX_CROSS_TRACK_M
RUNTIME_RAW_ODOM_START_POSITION_ENVELOPE_M = 0.025
# Qualified RAICOM SD traces retained raw scene-world yaw offsets of
# 0.000045, 0.030419, 0.031349, and 0.068198 rad from the source START.
# The 0.075 rad bound is the smallest reviewed rounded-up envelope above that
# observed maximum; its complete short-path collision sweep remains required.
RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD = 0.075
RUNTIME_RAW_ODOM_START_Z_MIN_M = 0.64
RUNTIME_RAW_ODOM_START_Z_MAX_M = 0.69

# This is a separately named navigation objective.  It intentionally stops
# 100 mm before Local01's immutable TABLE_DOCKING pose; it is not a docking or
# manipulation acceptance pose.  The window budget follows the measured
# 0.171312 m maximum-progress evidence from the existing 5 s M1 transit
# profile: 2.40 m needs 15 such bounded windows in the conservative case.
TABLE_APPROACH_X_M = 0.90
TABLE_APPROACH_Y_M = -1.50
TABLE_APPROACH_YAW_RAD = 0.0
TABLE_APPROACH_MAX_WINDOWS = 15
TABLE_APPROACH_MAX_WINDOW_S = 5.0
TABLE_APPROACH_MAX_AGGREGATE_NONZERO_S = (
    TABLE_APPROACH_MAX_WINDOWS * TABLE_APPROACH_MAX_WINDOW_S
)
TABLE_APPROACH_HISTORICAL_PROGRESS_PER_WINDOW_M = 0.171312
TABLE_APPROACH_ALIGNMENT_MAX_WINDOWS = 8
TABLE_APPROACH_ALIGNMENT_MAX_DISPLACEMENT_M = 0.05


class Local01SceneRouteError(ValueError):
    """Local01 scene provenance, frame, or route validation failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _parse_xml_vector(value: str | None, count: int, label: str) -> tuple[float, ...]:
    """Parse one finite, exact-length MJCF vector without accepting omissions."""
    if not isinstance(value, str):
        raise Local01SceneRouteError(f"{label} is missing")
    try:
        parsed = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise Local01SceneRouteError(f"{label} is not numeric") from exc
    if len(parsed) != count or not all(math.isfinite(item) for item in parsed):
        raise Local01SceneRouteError(f"{label} is not a finite {count}-vector")
    return parsed


def _x2_start_frame(root: ET.Element) -> ET.Element:
    frames = root.findall(f"./worldbody/frame[@name='{X2_START_FRAME_NAME}']")
    if len(frames) != 1:
        raise Local01SceneRouteError("expected exactly one X2 start frame")
    return frames[0]


def _validate_source_x2_start_frame(frame: ET.Element) -> None:
    pos = _parse_xml_vector(frame.get("pos"), 3, "x2_start_frame.pos")
    quat = _parse_xml_vector(frame.get("quat"), 4, "x2_start_frame.quat")
    if any(abs(actual - expected) > 1e-12 for actual, expected in zip(pos, X2_SOURCE_START_POS_M)):
        raise Local01SceneRouteError("source X2 start-frame position differs from Local01 provenance")
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(quat, X2_SOURCE_START_QUAT_WXYZ)
    ):
        raise Local01SceneRouteError("source X2 start-frame quaternion differs from Local01 provenance")


def _format_xml_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.16g}" for value in values)


def derive_synchronized_nominal_keyframe(model: Any, runtime_config_path: str | Path) -> dict[str, Any]:
    """Derive a contact-valid X2 standing keyframe from the installed MC config.

    This is a simulation-only initialization artifact.  It uses the vendor
    ``logical.nominal_configuration`` verbatim, retains every non-robot qpos0
    entry, sets all qvel to zero, and derives the root height from the compiled
    selected-foot collision geometry.  It performs only ``mj_forward``.
    """
    import mujoco
    import yaml

    config_path = Path(runtime_config_path)
    if config_path.is_symlink() or not config_path.is_file():
        raise Local01SceneRouteError("runtime default.yaml is missing or unsafe")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        joint_groups = config["actual"]["active_joint_name"]
        nominal_groups = config["logical"]["nominal_configuration"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise Local01SceneRouteError("installed MC nominal configuration is invalid") from exc
    joint_names = [str(name) for group in joint_groups for name in group]
    nominal = [float(value) for group in nominal_groups for value in group]
    if len(joint_names) != len(nominal) or not joint_names:
        raise Local01SceneRouteError("MC nominal configuration does not match active joint names")
    if not all(math.isfinite(value) for value in nominal):
        raise Local01SceneRouteError("MC nominal configuration contains a non-finite joint value")

    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise Local01SceneRouteError("compiled scene lacks the native X2 floating base")
    root_qadr = int(model.jnt_qposadr[root_joint])
    qpos = model.qpos0.copy()
    for name, value in zip(joint_names, nominal):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise Local01SceneRouteError(f"MC nominal joint is not a compiled X2 hinge: {name}")
        lower, upper = model.jnt_range[joint_id]
        if model.jnt_limited[joint_id] and not (float(lower) <= value <= float(upper)):
            raise Local01SceneRouteError(f"MC nominal joint violates source limit: {name}")
        qpos[int(model.jnt_qposadr[joint_id])] = value

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise Local01SceneRouteError("compiled RAICOM scene lacks the canonical floor geom")
    selected_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
    }
    if min(selected_bodies) < 0:
        raise Local01SceneRouteError("compiled scene lacks the selected X2 ankle-roll bodies")
    foot_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in selected_bodies and int(model.geom_contype[geom_id])
    ]
    if not foot_geoms:
        raise Local01SceneRouteError("selected X2 feet have no compiled collision geoms")
    lower_bounds = [
        float(data.geom_xpos[geom_id][2] - model.geom_rbound[geom_id])
        for geom_id in foot_geoms
    ]
    minimum_before = min(lower_bounds)
    qpos[root_qadr + 2] = float(qpos[root_qadr + 2] - minimum_before)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    foot_bounds_after = {
        str(geom_id): float(data.geom_xpos[geom_id][2] - model.geom_rbound[geom_id])
        for geom_id in foot_geoms
    }
    permitted_contact_pairs: list[dict[str, Any]] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if int(floor_id) not in (contact.geom1, contact.geom2) or not ({body1, body2} & selected_bodies):
            raise Local01SceneRouteError("nominal initialization has a non-foot/environment contact")
        if abs(float(contact.dist)) > 1e-8:
            raise Local01SceneRouteError("nominal initialization contact distance is outside static tolerance")
        permitted_contact_pairs.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            "distance_m": float(contact.dist),
        })
    if len(permitted_contact_pairs) < 2:
        raise Local01SceneRouteError("nominal initialization lacks bilateral floor contacts")
    if any(abs(value) > 1e-8 for value in foot_bounds_after.values() if abs(value) > 0.0):
        # Non-lowest collision pads are intentionally above the floor.  Each
        # selected foot must nevertheless own at least one tangent pad.
        sides = {
            "left": any(
                int(model.geom_bodyid[geom_id]) == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
                and abs(foot_bounds_after[str(geom_id)]) <= 1e-8
                for geom_id in foot_geoms
            ),
            "right": any(
                int(model.geom_bodyid[geom_id]) == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
                and abs(foot_bounds_after[str(geom_id)]) <= 1e-8
                for geom_id in foot_geoms
            ),
        }
    else:
        sides = {"left": True, "right": True}
    if not all(sides.values()):
        raise Local01SceneRouteError("nominal initialization lacks a tangent collision pad on one foot")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return {
        "schema": "robotsim.issue80.synchronized-nominal-keyframe.v1",
        "classification": "SIMULATION_ONLY_SOURCE_MC_NOMINAL_CONFIGURATION_KEYFRAME",
        "key_name": SYNCHRONIZED_NOMINAL_KEYFRAME_NAME,
        "version": SYNCHRONIZED_NOMINAL_KEYFRAME_VERSION,
        "qpos": [float(value) for value in qpos],
        "qvel": [0.0] * int(model.nv),
        "source_runtime_config": str(config_path),
        "source_nominal_joint_names": joint_names,
        "source_nominal_joint_values_rad": nominal,
        "source_qpos0_root": [float(value) for value in model.qpos0[root_qadr:root_qadr + 7]],
        "derived_root_qpos": [float(value) for value in qpos[root_qadr:root_qadr + 7]],
        "root_height_shift_m": float(-minimum_before),
        "lowest_selected_foot_bound_before_shift_m": minimum_before,
        "selected_foot_lower_bounds_after_shift_m": foot_bounds_after,
        "selected_foot_tangent_contact": sides,
        "contact_pairs": permitted_contact_pairs,
        "pelvis_world_position_m": [float(value) for value in data.xpos[pelvis_id]],
        "qvel_zero": bool(not data.qvel.any()),
        "mj_step_calls": 0,
        "limitations": [
            "This is a static mj_forward collision/contact validation, not an active standing result.",
            "The keyframe derives joint values only from the installed MC nominal_configuration and retains non-robot qpos0 entries.",
        ],
    }


def _pose_xyz_yaw(value: Mapping[str, Any], label: str) -> tuple[float, float, float, float]:
    try:
        if "xyz_m" in value:
            xyz = value["xyz_m"]
            if not isinstance(xyz, list) or len(xyz) != 3:
                raise Local01SceneRouteError(f"{label}.xyz_m must contain three coordinates")
            result = tuple(float(v) for v in xyz) + (float(value["yaw_rad"]),)
        else:
            result = tuple(float(value[key]) for key in ("x_m", "y_m", "z_m", "yaw_rad"))
    except (KeyError, TypeError, ValueError) as exc:
        raise Local01SceneRouteError(f"{label} is missing a finite xyz/yaw pose") from exc
    if not all(math.isfinite(v) for v in result):
        raise Local01SceneRouteError(f"{label} contains a non-finite coordinate")
    return result  # type: ignore[return-value]


def _transform_pose(
    pose: Mapping[str, Any], transform: Mapping[str, Any], *, label: str
) -> dict[str, float]:
    x, y, z, yaw = _pose_xyz_yaw(pose, label)
    try:
        tx, ty, tz = (float(v) for v in transform["translation_xyz_m"])
        theta = float(transform["yaw_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Local01SceneRouteError("scene-to-odom transform is malformed") from exc
    if not all(math.isfinite(v) for v in (tx, ty, tz, theta)):
        raise Local01SceneRouteError("scene-to-odom transform contains non-finite values")
    c, s = math.cos(theta), math.sin(theta)
    result = {
        "x_m": tx + c * x - s * y,
        "y_m": ty + s * x + c * y,
        "z_m": tz + z,
        "yaw_rad": _wrap(theta + yaw),
    }
    return {key: (0.0 if abs(value) < 1e-12 else value) for key, value in result.items()}


def _scene_pose(x: float, y: float, z: float, yaw: float) -> dict[str, Any]:
    return {"xyz_m": [float(x), float(y), float(z)], "yaw_rad": float(yaw)}


def _find_repository_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists() and (parent / "README.md").is_file():
            return parent
    raise Local01SceneRouteError("Local01 input is not inside a RobotSim Git worktree")


def load_local01_scene_inputs(
    poses_path: str | Path,
    *,
    require_clean_commit: bool = True,
) -> dict[str, Any]:
    """Read exact Local01 commit 95d76d8 inputs without modifying its worktree."""
    poses_file = Path(poses_path)
    if poses_file.is_symlink() or not poses_file.is_file():
        raise Local01SceneRouteError("Local01 scene pose input must be a regular file")
    root = _find_repository_root(poses_file)
    expected_pose_path = root / LOCAL01_POSES_RELATIVE
    if poses_file.resolve(strict=True) != expected_pose_path.resolve(strict=True):
        raise Local01SceneRouteError("pose input is not the pinned Local01 scene_world_poses.json path")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise Local01SceneRouteError("unable to verify Local01 worktree provenance") from exc
    if head != LOCAL01_COMMIT:
        raise Local01SceneRouteError(
            f"Local01 commit mismatch: expected {LOCAL01_COMMIT}, got {head}"
        )
    if require_clean_commit and status:
        raise Local01SceneRouteError("Local01 worktree is dirty; refuse uncommitted scene inputs")

    files = {
        "scene_world_poses": (expected_pose_path, LOCAL01_POSES_SHA256),
        "route_waypoints": (root / LOCAL01_ROUTE_RELATIVE, LOCAL01_ROUTE_SHA256),
        "scene_workspace": (root / LOCAL01_SCENE_RELATIVE, LOCAL01_SCENE_SHA256),
        "mug_portable": (root / LOCAL01_MUG_RELATIVE, LOCAL01_MUG_SHA256),
    }
    hashes: dict[str, str] = {}
    for label, (path, expected) in files.items():
        if path.is_symlink() or not path.is_file():
            raise Local01SceneRouteError(f"Local01 {label} source is missing or is not a regular file")
        actual = _sha256(path)
        if actual != expected:
            raise Local01SceneRouteError(
                f"Local01 {label} hash mismatch: expected {expected}, got {actual}"
            )
        hashes[label] = actual

    reference_root = (root / LOCAL01_SCENE_REPO_RELATIVE).resolve(strict=True)
    if not reference_root.is_dir():
        raise Local01SceneRouteError("resolved RAICOM scene reference root is not a directory")
    required_reference_files = {
        "room": reference_root / "robot/lx2501_3_t2d5/model_info/scene/room/room_with_chair_review.xml",
        "table": reference_root / "robot/lx2501_3_t2d5/model_info/object/mesh/table.stl",
        "chair": reference_root / "robot/lx2501_3_t2d5/model_info/object/mesh/chair.stl",
        "mug_obj": reference_root / "robot/lx2501_3_t2d5/model_info/object/model/mug/mug.obj",
        "mug_png": reference_root / "robot/lx2501_3_t2d5/model_info/object/model/mug/mug.png",
    }
    for path in required_reference_files.values():
        if path.is_symlink() or not path.is_file():
            raise Local01SceneRouteError(f"pinned RAICOM scene reference file is missing: {path.name}")
    expected_reference_hashes = {
        "room": LOCAL01_ROOM_XML_SHA256,
        "table": LOCAL01_TABLE_MESH_SHA256,
        "chair": LOCAL01_CHAIR_MESH_SHA256,
        "mug_obj": LOCAL01_MUG_OBJ_SHA256,
        "mug_png": LOCAL01_MUG_PNG_SHA256,
    }
    for name, expected in expected_reference_hashes.items():
        if _sha256(required_reference_files[name]) != expected:
            raise Local01SceneRouteError(f"RAICOM {name} source hash does not match Local01 static evidence")

    runtime_x2 = root / "simulation/mujoco/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
    if runtime_x2.is_symlink() or not runtime_x2.is_file() or _sha256(runtime_x2) != LOCAL01_RUNTIME_X2_SHA256:
        raise Local01SceneRouteError("Local01 scene does not target the pinned MC/runtime X2 XML")

    try:
        config = json.loads(expected_pose_path.read_text(encoding="utf-8"))
        route = json.loads(files["route_waypoints"][0].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Local01SceneRouteError("Local01 coordinate input is invalid JSON") from exc
    if not isinstance(config, Mapping) or config.get("schema") != "robotsim.raicom_scene_world_poses.v1":
        raise Local01SceneRouteError("unsupported Local01 scene-world pose schema")
    if not isinstance(route, Mapping) or route.get("schema") != "robotsim.raicom_scene_route.v1":
        raise Local01SceneRouteError("unsupported Local01 route schema")
    if route.get("poses_file") != LOCAL01_POSES_RELATIVE.as_posix():
        raise Local01SceneRouteError("Local01 route does not reference the authoritative pose file")
    transforms = config.get("transforms")
    poses = config.get("poses")
    if not isinstance(transforms, Mapping) or not isinstance(poses, Mapping):
        raise Local01SceneRouteError("Local01 poses or transform block is missing")
    scene_from_odom = transforms.get("scene_world_from_odom")
    odom_from_scene = transforms.get("odom_from_scene_world")
    reset = transforms.get("odom_reset_anchor")
    if not all(isinstance(v, Mapping) for v in (scene_from_odom, odom_from_scene, reset)):
        raise Local01SceneRouteError("Local01 scene/odom transform pair or reset anchor is missing")
    if (
        scene_from_odom.get("parent_frame") != "raicom_scene_world"
        or scene_from_odom.get("child_frame") != "odom"
        or odom_from_scene.get("parent_frame") != "odom"
        or odom_from_scene.get("child_frame") != "raicom_scene_world"
    ):
        raise Local01SceneRouteError("Local01 frame transform parent/child definitions are invalid")
    probe_scene = _scene_pose(0.73, -0.41, 0.22, 0.37)
    probe_odom = _transform_pose(probe_scene, odom_from_scene, label="transform inverse probe")
    probe_roundtrip = _transform_pose(probe_odom, scene_from_odom, label="transform roundtrip probe")
    probe_xyz_yaw = _pose_xyz_yaw(probe_scene, "transform inverse probe")
    roundtrip_xyz_yaw = (
        probe_roundtrip["x_m"], probe_roundtrip["y_m"],
        probe_roundtrip["z_m"], probe_roundtrip["yaw_rad"],
    )
    if (
        math.dist(probe_xyz_yaw[:3], roundtrip_xyz_yaw[:3]) > 1e-9
        or abs(_wrap(probe_xyz_yaw[3] - roundtrip_xyz_yaw[3])) > 1e-9
    ):
        raise Local01SceneRouteError("Local01 scene-world/odom transforms are not inverses")
    start_scene = poses.get("START")
    start_expected_odom = reset.get("base_pose_in_odom_after_reset")
    if not isinstance(start_scene, Mapping) or not isinstance(start_expected_odom, Mapping):
        raise Local01SceneRouteError("Local01 START pose or expected reset odom pose is missing")
    start_odom = _transform_pose(start_scene, odom_from_scene, label="START")
    ex, ey, ez, eyaw = _pose_xyz_yaw(start_expected_odom, "odom_reset_anchor.base_pose_in_odom_after_reset")
    if (
        math.hypot(start_odom["x_m"] - ex, start_odom["y_m"] - ey) > 1e-9
        or abs(start_odom["z_m"] - ez) > 1e-9
        or abs(_wrap(start_odom["yaw_rad"] - eyaw)) > 1e-9
    ):
        raise Local01SceneRouteError("Local01 transform does not map START to its declared reset anchor")
    route_order = route.get("route_order")
    if not isinstance(route_order, list):
        raise Local01SceneRouteError("Local01 route order is missing")
    expected_route_odom = {
        "START": (0.0, 0.0, 0.68, 0.0),
        "START_ALIGN": (0.0, 0.0, 0.68, -math.pi / 2.0),
        "WAYPOINT": (0.0, -1.25, 0.68, -math.pi / 2.0),
        "TABLE_DOCKING": (0.0, -2.5, 0.68, -math.pi / 2.0),
    }
    for name, expected_pose in expected_route_odom.items():
        scene_pose = poses.get(name)
        if name not in route_order or not isinstance(scene_pose, Mapping):
            raise Local01SceneRouteError(f"Local01 route is missing authoritative pose {name}")
        transformed = _transform_pose(scene_pose, odom_from_scene, label=name)
        actual = (transformed["x_m"], transformed["y_m"], transformed["z_m"], transformed["yaw_rad"])
        if (
            math.dist(actual[:3], expected_pose[:3]) > 1e-9
            or abs(_wrap(actual[3] - expected_pose[3])) > 1e-9
        ):
            raise Local01SceneRouteError(f"Local01 {name} transform differs from the pinned scene/odom mapping")
    verification = reset.get("verification")
    if not isinstance(verification, Mapping):
        raise Local01SceneRouteError("Local01 reset/odom verification metadata is missing")
    if verification.get("live_reset_or_ros_odom_sample") != "NOT_RUN; no physics or navigation runtime was started in this task.":
        raise Local01SceneRouteError("unexpected Local01 live Reset/odom evidence status")

    return {
        "repository_root": str(root),
        "source_commit": head,
        "source_hashes": hashes,
        "reference_root": str(reference_root),
        "reference_files": {name: str(path) for name, path in required_reference_files.items()},
        "runtime_x2_path": str(runtime_x2),
        "scene_world_poses": dict(config),
        "route_waypoints": dict(route),
        "scene_from_odom": dict(scene_from_odom),
        "odom_from_scene": dict(odom_from_scene),
        "reset_anchor": dict(reset),
        "start_odom_expected": start_odom,
        "route_poses_odom_expected": expected_route_odom,
        "runtime_reset_odom_observed": False,
    }


def transform_scene_pose_to_odom(
    pose: Mapping[str, Any], transform: Mapping[str, Any], *, label: str = "pose"
) -> dict[str, float]:
    """Convert one `raicom_scene_world` XYZ/yaw pose using Local01's inverse transform."""
    return _transform_pose(pose, transform, label=label)


def build_short_route_config(source: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a 0.40 m, two-target diagnostic route along the Reset heading.

    The full TABLE_DOCKING target is retained as a reference and is never
    substituted by the short route endpoint.
    """
    config = source.get("scene_world_poses")
    transform = source.get("odom_from_scene")
    if not isinstance(config, Mapping) or not isinstance(transform, Mapping):
        raise Local01SceneRouteError("verified Local01 source mapping is required")
    poses = config.get("poses")
    if not isinstance(poses, Mapping):
        raise Local01SceneRouteError("Local01 authoritative pose table is missing")
    start_scene = poses.get("START")
    full_dock_scene = poses.get("TABLE_DOCKING")
    if not isinstance(start_scene, Mapping) or not isinstance(full_dock_scene, Mapping):
        raise Local01SceneRouteError("Local01 START or safe TABLE_DOCKING pose is missing")
    start = _transform_pose(start_scene, transform, label="START")
    _, _, _, start_yaw = _pose_xyz_yaw(start_scene, "START")

    def target(distance: float) -> tuple[dict[str, Any], dict[str, float]]:
        scene_pose = _scene_pose(
            float(start_scene["xyz_m"][0]) + distance * math.cos(start_yaw),
            float(start_scene["xyz_m"][1]) + distance * math.sin(start_yaw),
            float(start_scene["xyz_m"][2]),
            start_yaw,
        )
        return scene_pose, _transform_pose(scene_pose, transform, label=f"short target {distance:.2f} m")

    first_scene, first_odom = target(SHORT_ROUTE_STEP_M)
    end_scene, end_odom = target(SHORT_ROUTE_TARGETS_M[-1])
    full_dock = _transform_pose(full_dock_scene, transform, label="TABLE_DOCKING")
    full_distance = math.hypot(
        float(full_dock_scene["xyz_m"][0]) - float(start_scene["xyz_m"][0]),
        float(full_dock_scene["xyz_m"][1]) - float(start_scene["xyz_m"][1]),
    )
    raw_reset_anchor = source["reset_anchor"]["base_pose_in_odom_after_reset"]
    reset_xyz = raw_reset_anchor["xyz_m"]
    reset_anchor = {
        "x_m": float(reset_xyz[0]),
        "y_m": float(reset_xyz[1]),
        "z_m": float(reset_xyz[2]),
        "yaw_rad": float(raw_reset_anchor["yaw_rad"]),
    }
    theoretical_budget_distance = SHORT_ROUTE_MAX_AGGREGATE_S * SHORT_ROUTE_FORWARD_M_S
    return {
        "schema_version": SHORT_ROUTE_SCHEMA,
        "route_id": "local01-reset-heading-short-two-target-diagnostic",
        "route_kind": "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING",
        "frame_id": "odom",
        "source_frame_id": "raicom_scene_world",
        "source_local01_commit": source.get("source_commit"),
        "source_pose_sha256": source.get("source_hashes", {}).get("scene_world_poses"),
        "odom_from_scene_transform": dict(transform),
        "reset_anchor_expected_odom": reset_anchor,
        "runtime_reset_odom_observed": False,
        "start": start,
        "waypoint": first_odom,
        "short_route_end": end_odom,
        "scene_world_poses": {
            "start": dict(start_scene),
            "waypoint": first_scene,
            "short_route_end": end_scene,
        },
        "safe_table_docking_reference": {
            "pose_scene_world": dict(full_dock_scene),
            "pose_odom": full_dock,
            "distance_from_start_m": full_distance,
            "status": "STATIC_BASELINE_ONLY_NOT_TARGETED_BY_SHORT_ROUTE",
        },
        "limits": {
            "forward_m_s": SHORT_ROUTE_FORWARD_M_S,
            "angular_rad_s": SHORT_ROUTE_ANGULAR_RAD_S,
            "max_cross_track_m": SHORT_ROUTE_MAX_CROSS_TRACK_M,
            "max_windows": SHORT_ROUTE_MAX_WINDOWS,
            "max_window_s": SHORT_ROUTE_MAX_WINDOW_S,
            "max_aggregate_nonzero_s": SHORT_ROUTE_MAX_AGGREGATE_S,
            "max_distance_at_speed_cap_m": theoretical_budget_distance,
        },
        "full_table_route_budget": {
            "distance_m": full_distance,
            "minimum_translation_time_at_0p12_m_s": full_distance / SHORT_ROUTE_FORWARD_M_S,
            "max_aggregate_nonzero_s": 15.0,
            "maximum_translation_distance_m": 15.0 * SHORT_ROUTE_FORWARD_M_S,
            "status": "CANNOT_REACH_ENDPOINT_WITH_EXISTING_BUDGET",
        },
    }


def build_runtime_raw_odom_short_route_config(source: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the two short targets in the numeric convention emitted at runtime.

    The vendor simulator labels the published base pose frame ``odom``.  The
    first real RAICOM Reset/standing trace showed that its numeric values match
    the RAICOM scene world directly, rather than Local01's static
    ``odom_from_scene_world`` algebra.  Keep the source transform as a static
    reference, but do not apply it to a live control target until a fresh
    runtime Reset sample has proven that convention again.
    """
    static = build_short_route_config(source)
    config = source.get("scene_world_poses")
    if not isinstance(config, Mapping) or not isinstance(config.get("poses"), Mapping):
        raise Local01SceneRouteError("Local01 authoritative pose table is missing")
    poses = config["poses"]
    start_scene = poses.get("START")
    full_dock_scene = poses.get("TABLE_DOCKING")
    if not isinstance(start_scene, Mapping) or not isinstance(full_dock_scene, Mapping):
        raise Local01SceneRouteError("Local01 START or safe TABLE_DOCKING pose is missing")
    start_x, start_y, start_z, start_yaw = _pose_xyz_yaw(start_scene, "START")

    def target(distance: float) -> dict[str, float]:
        return {
            "x_m": start_x + distance * math.cos(start_yaw),
            "y_m": start_y + distance * math.sin(start_yaw),
            "z_m": start_z,
            "yaw_rad": start_yaw,
        }

    first = target(SHORT_ROUTE_STEP_M)
    end = target(SHORT_ROUTE_TARGETS_M[-1])
    dock_x, dock_y, dock_z, dock_yaw = _pose_xyz_yaw(full_dock_scene, "TABLE_DOCKING")
    raw = dict(static)
    raw.update({
        "route_id": "local01-runtime-raw-odom-forward-two-target-diagnostic",
        "route_kind": "SIMULATION_ONLY_SHORT_ROUTE_RAW_ODOM_SCENE_WORLD_DIAGNOSTIC_NOT_TABLE_DOCKING",
        "start": {"x_m": start_x, "y_m": start_y, "z_m": start_z, "yaw_rad": start_yaw},
        "waypoint": first,
        "short_route_end": end,
        "reset_anchor_expected_odom": {
            "x_m": start_x, "y_m": start_y, "z_m": start_z, "yaw_rad": start_yaw,
        },
        "runtime_coordinate_contract": {
            "published_frame_id": "odom",
            "numeric_coordinate_convention": "RAICOM_SCENE_WORLD_AFTER_LIVE_RESET_GATE",
            "static_scene_start": {"x_m": start_x, "y_m": start_y, "z_m": start_z, "yaw_rad": start_yaw},
            "maximum_start_position_deviation_m": RUNTIME_RAW_ODOM_START_POSITION_ENVELOPE_M,
            "maximum_start_yaw_deviation_rad": RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,
            "minimum_start_z_m": RUNTIME_RAW_ODOM_START_Z_MIN_M,
            "maximum_start_z_m": RUNTIME_RAW_ODOM_START_Z_MAX_M,
            "runtime_target_policy": "derive each 0.20 m target from the fresh post-STAND measured pose and yaw",
            "static_transform_retained_as_reference_only": static["odom_from_scene_transform"],
        },
        "scene_world_poses": {
            "start": dict(start_scene),
            "waypoint": _scene_pose(first["x_m"], first["y_m"], first["z_m"], first["yaw_rad"]),
            "short_route_end": _scene_pose(end["x_m"], end["y_m"], end["z_m"], end["yaw_rad"]),
        },
        "safe_table_docking_reference": {
            "pose_scene_world": dict(full_dock_scene),
            "published_odom_numeric_pose": {"x_m": dock_x, "y_m": dock_y, "z_m": dock_z, "yaw_rad": dock_yaw},
            "distance_from_start_m": math.hypot(dock_x - start_x, dock_y - start_y),
            "status": "STATIC_BASELINE_ONLY_NOT_TARGETED_BY_SHORT_ROUTE",
        },
    })
    return raw


def build_runtime_raw_odom_table_approach_mission(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build the source-pinned RAICOM table-approach mission definition.

    This is an offline mission contract, not a claim that the current MC can
    perform the required initial yaw-only alignment.  The native simulator's
    observed numeric ``odom`` convention is scene-world after Reset; retain
    Local01's planned-local transform separately as source documentation.
    """
    config = source.get("scene_world_poses")
    if not isinstance(config, Mapping) or not isinstance(config.get("poses"), Mapping):
        raise Local01SceneRouteError("Local01 authoritative pose table is missing")
    poses = config["poses"]
    required = ("START", "START_ALIGN", "WAYPOINT", "TABLE_DOCKING")
    if any(not isinstance(poses.get(name), Mapping) for name in required):
        raise Local01SceneRouteError("Local01 table mission is missing a required source pose")

    start_x, start_y, start_z, start_yaw = _pose_xyz_yaw(poses["START"], "START")
    align_x, align_y, align_z, align_yaw = _pose_xyz_yaw(poses["START_ALIGN"], "START_ALIGN")
    waypoint_x, waypoint_y, waypoint_z, waypoint_yaw = _pose_xyz_yaw(
        poses["WAYPOINT"], "WAYPOINT"
    )
    dock_x, dock_y, dock_z, dock_yaw = _pose_xyz_yaw(poses["TABLE_DOCKING"], "TABLE_DOCKING")
    if (
        math.dist((start_x, start_y, start_z), (align_x, align_y, align_z)) > 1e-9
        or abs(_wrap(align_yaw - TABLE_APPROACH_YAW_RAD)) > 1e-9
    ):
        raise Local01SceneRouteError("START_ALIGN must preserve START xyz and set the source table heading")
    if (
        math.dist((waypoint_x, waypoint_y, waypoint_z), (-0.25, -1.50, start_z)) > 1e-9
        or abs(_wrap(waypoint_yaw - TABLE_APPROACH_YAW_RAD)) > 1e-9
    ):
        raise Local01SceneRouteError("Local01 WAYPOINT does not match the pinned table corridor")
    if (
        math.dist((dock_x, dock_y, dock_z), (1.00, -1.50, start_z)) > 1e-9
        or abs(_wrap(dock_yaw - TABLE_APPROACH_YAW_RAD)) > 1e-9
    ):
        raise Local01SceneRouteError("Local01 TABLE_DOCKING differs from the pinned source pose")

    approach = {
        "x_m": TABLE_APPROACH_X_M,
        "y_m": TABLE_APPROACH_Y_M,
        "z_m": start_z,
        "yaw_rad": TABLE_APPROACH_YAW_RAD,
    }
    start = {"x_m": start_x, "y_m": start_y, "z_m": start_z, "yaw_rad": start_yaw}
    start_align = {"x_m": align_x, "y_m": align_y, "z_m": align_z, "yaw_rad": align_yaw}
    waypoint = {"x_m": waypoint_x, "y_m": waypoint_y, "z_m": waypoint_z, "yaw_rad": waypoint_yaw}
    docking = {"x_m": dock_x, "y_m": dock_y, "z_m": dock_z, "yaw_rad": dock_yaw}
    translation_distance = math.hypot(waypoint_x - align_x, waypoint_y - align_y) + math.hypot(
        TABLE_APPROACH_X_M - waypoint_x, TABLE_APPROACH_Y_M - waypoint_y
    )
    expected_windows = math.ceil(
        translation_distance / TABLE_APPROACH_HISTORICAL_PROGRESS_PER_WINDOW_M
    )
    return {
        "schema_version": TABLE_APPROACH_MISSION_SCHEMA,
        "mission_id": "local01-raicom-safe-table-approach-raw-world-odom",
        "classification": "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION",
        "frame_id": "odom",
        "source_frame_id": "raicom_scene_world",
        "source_local01_commit": source.get("source_commit"),
        "source_pose_sha256": source.get("source_hashes", {}).get("scene_world_poses"),
        "runtime_reset_odom_observed": False,
        "runtime_coordinate_contract": {
            "published_frame_id": "odom",
            "numeric_coordinate_convention": "RAICOM_SCENE_WORLD_AFTER_LIVE_RESET_GATE",
            "static_scene_start": start,
            "maximum_start_position_deviation_m": RUNTIME_RAW_ODOM_START_POSITION_ENVELOPE_M,
            "maximum_start_yaw_deviation_rad": RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,
            "minimum_start_z_m": RUNTIME_RAW_ODOM_START_Z_MIN_M,
            "maximum_start_z_m": RUNTIME_RAW_ODOM_START_Z_MAX_M,
            "static_transform_retained_as_reference_only": dict(source["odom_from_scene"]),
        },
        "poses": {
            "START": start,
            "START_ALIGN": start_align,
            "WAYPOINT": waypoint,
            "TABLE_APPROACH": approach,
            "TABLE_DOCKING_REFERENCE_UNCHANGED": docking,
        },
        "execution_order": ["START", "START_ALIGN", "WAYPOINT", "TABLE_APPROACH"],
        "initial_alignment": {
            "requested_signed_yaw_change_rad": _wrap(TABLE_APPROACH_YAW_RAD - start_yaw),
            "target_yaw_rad": TABLE_APPROACH_YAW_RAD,
            "target_yaw_tolerance_rad": RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD,
            "max_turn_windows": TABLE_APPROACH_ALIGNMENT_MAX_WINDOWS,
            "max_total_planar_displacement_m": TABLE_APPROACH_ALIGNMENT_MAX_DISPLACEMENT_M,
            "static_status": "PASS_ONLY_IF_FULL_BODY_SWEEP_PASSES_SEPARATELY",
            "zero_forward_angular_mc_status": "NOT_QUALIFIED_REQUIRES_BOUNDED_LIVE_EVIDENCE",
            "moving_turn_status": "NOT_SUBSTITUTED_FOR_START_ALIGN; must remain in its own collision-checked corridor",
        },
        "limits": {
            "forward_m_s": SHORT_ROUTE_FORWARD_M_S,
            "angular_rad_s": SHORT_ROUTE_ANGULAR_RAD_S,
            "max_cross_track_m": SHORT_ROUTE_MAX_CROSS_TRACK_M,
            "max_window_s": TABLE_APPROACH_MAX_WINDOW_S,
            "max_windows": TABLE_APPROACH_MAX_WINDOWS,
            "max_aggregate_nonzero_s": TABLE_APPROACH_MAX_AGGREGATE_NONZERO_S,
        },
        "bounded_runtime_budget": {
            "translation_distance_m": translation_distance,
            "ideal_translation_minimum_s_at_speed_cap": translation_distance / SHORT_ROUTE_FORWARD_M_S,
            "historical_progress_per_5_s_window_m": TABLE_APPROACH_HISTORICAL_PROGRESS_PER_WINDOW_M,
            "historical_windows_required": expected_windows,
            "configured_windows_available": TABLE_APPROACH_MAX_WINDOWS,
            "translation_nonzero_budget_s": TABLE_APPROACH_MAX_AGGREGATE_NONZERO_S,
            "post_window_physical_stop": "unchanged full STAND/readback/composite physical-stop gate",
            "mission_runtime_requires_explicit_parent_watchdog_budget": True,
        },
        "table_docking_reference": {
            "pose": docking,
            "distance_from_table_approach_m": math.hypot(dock_x - TABLE_APPROACH_X_M, dock_y - TABLE_APPROACH_Y_M),
            "source_pose_unchanged": True,
            "dynamic_docking_status": "NOT_AUTHORIZED_BY_TABLE_APPROACH_MISSION",
            "manipulation_handoff": "PENDING_LOCAL04_REACHABILITY_AND_SOURCE_FAITHFUL_GRASP_VALIDATION",
        },
    }


def validate_runtime_raw_odom_table_approach_mission(mission: Mapping[str, Any]) -> None:
    """Validate the explicit non-docking table-approach mission contract."""
    if mission.get("schema_version") != TABLE_APPROACH_MISSION_SCHEMA:
        raise Local01SceneRouteError("unsupported table-approach mission schema")
    if mission.get("classification") != "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION":
        raise Local01SceneRouteError("table approach must remain explicitly non-docking")
    if mission.get("frame_id") != "odom" or mission.get("source_frame_id") != "raicom_scene_world":
        raise Local01SceneRouteError("table-approach mission frame contract is invalid")
    if mission.get("runtime_reset_odom_observed") is not False:
        raise Local01SceneRouteError("static table-approach mission must not claim live Reset evidence")
    if mission.get("execution_order") != ["START", "START_ALIGN", "WAYPOINT", "TABLE_APPROACH"]:
        raise Local01SceneRouteError("table-approach mission step order is invalid")
    poses = mission.get("poses")
    if not isinstance(poses, Mapping):
        raise Local01SceneRouteError("table-approach mission poses are missing")
    required = ("START", "START_ALIGN", "WAYPOINT", "TABLE_APPROACH", "TABLE_DOCKING_REFERENCE_UNCHANGED")
    if any(not isinstance(poses.get(name), Mapping) for name in required):
        raise Local01SceneRouteError("table-approach mission has a missing pose")
    parsed = {name: _pose_xyz_yaw(poses[name], name) for name in required}
    start, align, waypoint, approach, docking = (parsed[name] for name in required)
    if math.dist(start[:3], align[:3]) > 1e-9 or abs(_wrap(align[3])) > 1e-9:
        raise Local01SceneRouteError("START_ALIGN must be the source start position at yaw zero")
    if (
        math.dist(approach[:3], (TABLE_APPROACH_X_M, TABLE_APPROACH_Y_M, start[2])) > 1e-9
        or abs(_wrap(approach[3] - TABLE_APPROACH_YAW_RAD)) > 1e-9
    ):
        raise Local01SceneRouteError("TABLE_APPROACH differs from the reviewed source-derived standoff")
    if (
        math.dist(docking[:3], (1.00, -1.50, start[2])) > 1e-9
        or abs(_wrap(docking[3] - TABLE_APPROACH_YAW_RAD)) > 1e-9
    ):
        raise Local01SceneRouteError("TABLE_DOCKING reference must remain unchanged")
    if not math.isclose(math.hypot(docking[0] - approach[0], docking[1] - approach[1]), 0.10, abs_tol=1e-9):
        raise Local01SceneRouteError("table approach must remain exactly 100 mm short of TABLE_DOCKING")
    if not math.isclose(waypoint[1], start[1], abs_tol=1e-9) or not math.isclose(approach[1], start[1], abs_tol=1e-9):
        raise Local01SceneRouteError("table-approach translation corridor must retain the pinned y coordinate")
    alignment = mission.get("initial_alignment")
    if not isinstance(alignment, Mapping) or alignment.get("zero_forward_angular_mc_status") != "NOT_QUALIFIED_REQUIRES_BOUNDED_LIVE_EVIDENCE":
        raise Local01SceneRouteError("initial yaw alignment must remain dynamically unqualified before live evidence")
    if (
        alignment.get("max_turn_windows") != TABLE_APPROACH_ALIGNMENT_MAX_WINDOWS
        or abs(float(alignment.get("max_total_planar_displacement_m", float("nan"))) - TABLE_APPROACH_ALIGNMENT_MAX_DISPLACEMENT_M) > 1e-12
        or abs(float(alignment.get("target_yaw_tolerance_rad", float("nan"))) - RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD) > 1e-12
    ):
        raise Local01SceneRouteError("table-approach initial alignment exceeds the reviewed bounded envelope")
    limits = mission.get("limits")
    expected_limits = {
        "forward_m_s": SHORT_ROUTE_FORWARD_M_S,
        "angular_rad_s": SHORT_ROUTE_ANGULAR_RAD_S,
        "max_cross_track_m": SHORT_ROUTE_MAX_CROSS_TRACK_M,
        "max_window_s": TABLE_APPROACH_MAX_WINDOW_S,
        "max_windows": TABLE_APPROACH_MAX_WINDOWS,
        "max_aggregate_nonzero_s": TABLE_APPROACH_MAX_AGGREGATE_NONZERO_S,
    }
    if not isinstance(limits, Mapping) or any(limits.get(key) != value for key, value in expected_limits.items()):
        raise Local01SceneRouteError("table-approach mission exceeds the approved bounded motion envelope")
    reference = mission.get("table_docking_reference")
    if not isinstance(reference, Mapping) or reference.get("source_pose_unchanged") is not True:
        raise Local01SceneRouteError("table-approach mission must preserve the source docking reference")


def build_runtime_raw_odom_aligned_start_table_approach_mission(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a separately named, pre-physics yaw-zero table-approach mission.

    This is intentionally a different simulation integration from the pinned
    RAICOM start.  It preserves source coordinates, robot and room geometry,
    limits, actuators, and controller configuration, while recording the one
    permitted initial-frame delta explicitly.  It never relabels the original
    +pi/2 start or claims that the original yaw-alignment problem is solved.
    """
    original = build_runtime_raw_odom_table_approach_mission(source)
    aligned = copy.deepcopy(original)
    source_start = dict(original["poses"]["START"])
    source_align = dict(original["poses"]["START_ALIGN"])
    if (
        math.dist(
            (float(source_start["x_m"]), float(source_start["y_m"]), float(source_start["z_m"])),
            (float(source_align["x_m"]), float(source_align["y_m"]), float(source_align["z_m"])),
        )
        > 1e-12
        or abs(_wrap(float(source_align["yaw_rad"]))) > 1e-12
    ):
        raise Local01SceneRouteError("source START_ALIGN is not the required same-position yaw-zero reference")

    aligned.update({
        "schema_version": ALIGNED_START_TABLE_APPROACH_MISSION_SCHEMA,
        "mission_id": "local01-raicom-aligned-start-v1-table-approach-raw-world-odom",
        "classification": ALIGNED_START_TABLE_APPROACH_CLASSIFICATION,
        "execution_order": ["START", "WAYPOINT", "TABLE_APPROACH"],
        "runtime_reset_odom_observed": False,
        "derived_scene_initialization": {
            "schema": "robotsim.issue80.aligned-start-scene-delta.v1",
            "version": ALIGNED_START_SCENE_VERSION,
            "applies_before_physics": True,
            "active_simulation_qpos_write": False,
            "source_x2_start_frame": {
                "name": X2_START_FRAME_NAME,
                "pos_m": list(X2_SOURCE_START_POS_M),
                "quat_wxyz": list(X2_SOURCE_START_QUAT_WXYZ),
            },
            "derived_x2_start_frame": {
                "name": X2_START_FRAME_NAME,
                "pos_m": list(X2_SOURCE_START_POS_M),
                "quat_wxyz": list(X2_ALIGNED_START_QUAT_WXYZ),
            },
            "original_raicom_start_yaw_rad": float(source_start["yaw_rad"]),
            "derived_start_yaw_rad": 0.0,
            "original_start_turn_status": "UNQUALIFIED_RETAINED_SEPARATELY",
        },
        "original_raicom_start_reference": source_start,
        "poses": {
            "START": source_align,
            "WAYPOINT": dict(original["poses"]["WAYPOINT"]),
            "TABLE_APPROACH": dict(original["poses"]["TABLE_APPROACH"]),
            "TABLE_DOCKING_REFERENCE_UNCHANGED": dict(
                original["poses"]["TABLE_DOCKING_REFERENCE_UNCHANGED"]
            ),
        },
        "initial_alignment": {
            "status": "NOT_APPLICABLE_DERIVED_INITIAL_YAW_ZERO_BEFORE_PHYSICS",
            "nonzero_command_authorized_for_alignment": False,
            "source_initial_yaw_rad": float(source_start["yaw_rad"]),
            "derived_initial_yaw_rad": 0.0,
            "original_yaw_only_turn_status": "UNQUALIFIED_RETAINED_SEPARATELY",
            "moving_turn_substitution": "NOT_PERMITTED_BY_THIS_DERIVED_ROUTE",
        },
    })
    contract = dict(aligned["runtime_coordinate_contract"])
    contract["static_scene_start"] = dict(source_align)
    contract["numeric_coordinate_convention"] = (
        "DERIVED_ALIGNED_SCENE_WORLD_RAW_ODOM_AFTER_LIVE_RESET_GATE"
    )
    contract["original_source_scene_start_reference"] = source_start
    aligned["runtime_coordinate_contract"] = contract
    return aligned


def validate_runtime_raw_odom_aligned_start_table_approach_mission(
    mission: Mapping[str, Any],
) -> None:
    """Fail closed unless the aligned-start contract changes only initialization yaw."""
    if mission.get("schema_version") != ALIGNED_START_TABLE_APPROACH_MISSION_SCHEMA:
        raise Local01SceneRouteError("unsupported aligned-start table-approach mission schema")
    if mission.get("classification") != ALIGNED_START_TABLE_APPROACH_CLASSIFICATION:
        raise Local01SceneRouteError("aligned-start table-approach classification is invalid")
    if mission.get("frame_id") != "odom" or mission.get("source_frame_id") != "raicom_scene_world":
        raise Local01SceneRouteError("aligned-start mission frame contract is invalid")
    if mission.get("execution_order") != ["START", "WAYPOINT", "TABLE_APPROACH"]:
        raise Local01SceneRouteError("aligned-start mission must omit dynamic START_ALIGN")
    if mission.get("runtime_reset_odom_observed") is not False:
        raise Local01SceneRouteError("aligned-start static mission must not claim live Reset evidence")
    poses = mission.get("poses")
    if not isinstance(poses, Mapping):
        raise Local01SceneRouteError("aligned-start mission poses are missing")
    required = ("START", "WAYPOINT", "TABLE_APPROACH", "TABLE_DOCKING_REFERENCE_UNCHANGED")
    if any(not isinstance(poses.get(name), Mapping) for name in required):
        raise Local01SceneRouteError("aligned-start mission has a missing pose")
    start, waypoint, approach, docking = (_pose_xyz_yaw(poses[name], name) for name in required)
    original_start = mission.get("original_raicom_start_reference")
    if not isinstance(original_start, Mapping):
        raise Local01SceneRouteError("aligned-start mission lost its original START reference")
    original_xyz_yaw = _pose_xyz_yaw(original_start, "original_raicom_start_reference")
    if (
        math.dist(start[:3], original_xyz_yaw[:3]) > 1e-12
        or abs(_wrap(start[3])) > 1e-12
        or abs(_wrap(original_xyz_yaw[3] - math.pi / 2.0)) > 1e-12
    ):
        raise Local01SceneRouteError("aligned-start must keep source START xyz and change only yaw to zero")
    if (
        math.dist(waypoint[:3], (-0.25, -1.50, start[2])) > 1e-12
        or abs(_wrap(waypoint[3])) > 1e-12
        or math.dist(approach[:3], (TABLE_APPROACH_X_M, TABLE_APPROACH_Y_M, start[2])) > 1e-12
        or abs(_wrap(approach[3])) > 1e-12
        or math.dist(docking[:3], (1.00, -1.50, start[2])) > 1e-12
        or abs(_wrap(docking[3])) > 1e-12
    ):
        raise Local01SceneRouteError("aligned-start mission altered the pinned table corridor or docking reference")
    if not math.isclose(math.hypot(docking[0] - approach[0], docking[1] - approach[1]), 0.10, abs_tol=1e-12):
        raise Local01SceneRouteError("aligned-start table approach must remain 100 mm short of docking")
    limits = mission.get("limits")
    expected_limits = {
        "forward_m_s": SHORT_ROUTE_FORWARD_M_S,
        "angular_rad_s": SHORT_ROUTE_ANGULAR_RAD_S,
        "max_cross_track_m": SHORT_ROUTE_MAX_CROSS_TRACK_M,
        "max_window_s": TABLE_APPROACH_MAX_WINDOW_S,
        "max_windows": TABLE_APPROACH_MAX_WINDOWS,
        "max_aggregate_nonzero_s": TABLE_APPROACH_MAX_AGGREGATE_NONZERO_S,
    }
    if not isinstance(limits, Mapping) or any(limits.get(key) != value for key, value in expected_limits.items()):
        raise Local01SceneRouteError("aligned-start mission exceeds the approved bounded motion envelope")
    initialization = mission.get("derived_scene_initialization")
    if not isinstance(initialization, Mapping) or (
        initialization.get("version") != ALIGNED_START_SCENE_VERSION
        or initialization.get("applies_before_physics") is not True
        or initialization.get("active_simulation_qpos_write") is not False
        or initialization.get("derived_start_yaw_rad") != 0.0
    ):
        raise Local01SceneRouteError("aligned-start initialization provenance is incomplete or unsafe")
    if initialization.get("source_x2_start_frame", {}).get("pos_m") != list(X2_SOURCE_START_POS_M):
        raise Local01SceneRouteError("aligned-start source frame position changed")
    if initialization.get("derived_x2_start_frame", {}).get("quat_wxyz") != list(X2_ALIGNED_START_QUAT_WXYZ):
        raise Local01SceneRouteError("aligned-start derived frame quaternion is invalid")
    alignment = mission.get("initial_alignment")
    if not isinstance(alignment, Mapping) or alignment.get("status") != "NOT_APPLICABLE_DERIVED_INITIAL_YAW_ZERO_BEFORE_PHYSICS":
        raise Local01SceneRouteError("aligned-start mission must not schedule an unqualified yaw-only turn")
    contract = mission.get("runtime_coordinate_contract")
    if not isinstance(contract, Mapping) or contract.get("static_scene_start") != dict(poses["START"]):
        raise Local01SceneRouteError("aligned-start runtime start contract does not match its derived scene pose")
    reference = mission.get("table_docking_reference")
    if not isinstance(reference, Mapping) or reference.get("source_pose_unchanged") is not True:
        raise Local01SceneRouteError("aligned-start mission must preserve the source docking reference")


def validate_runtime_raw_odom_table_approach_start(
    mission: Mapping[str, Any], observed_pose: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the fresh post-SD pose to remain inside the swept start envelope.

    The Reset anchor alone is insufficient: active standing can move the root
    before source-grounded locomotion begins.  This gate is deliberately
    evaluated immediately before the first yaw-only command.
    """
    validate_runtime_raw_odom_table_approach_mission(mission)
    contract = mission.get("runtime_coordinate_contract")
    if not isinstance(contract, Mapping):
        raise Local01SceneRouteError("table-approach mission lacks a runtime coordinate contract")
    expected = contract.get("static_scene_start")
    if not isinstance(expected, Mapping):
        raise Local01SceneRouteError("table-approach start anchor is missing")
    try:
        ex, ey, eyaw = (float(expected[name]) for name in ("x_m", "y_m", "yaw_rad"))
        ox, oy, oz, oyaw = (float(observed_pose[name]) for name in ("x_m", "y_m", "z_m", "yaw_rad"))
        position_limit = float(contract["maximum_start_position_deviation_m"])
        yaw_limit = float(contract["maximum_start_yaw_deviation_rad"])
        z_min = float(contract["minimum_start_z_m"])
        z_max = float(contract["maximum_start_z_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Local01SceneRouteError("table-approach runtime start pose is malformed") from exc
    values = (ex, ey, eyaw, ox, oy, oz, oyaw, position_limit, yaw_limit, z_min, z_max)
    if not all(math.isfinite(value) for value in values) or position_limit <= 0.0 or yaw_limit <= 0.0:
        raise Local01SceneRouteError("table-approach runtime start contains non-finite values or invalid limits")
    planar_error = math.hypot(ox - ex, oy - ey)
    yaw_error = abs(_wrap(oyaw - eyaw))
    passed = planar_error <= position_limit and yaw_error <= yaw_limit and z_min <= oz <= z_max
    return {
        "schema": "robotsim.issue80.table-approach-runtime-start-gate.v1",
        "status": "PASS" if passed else "FAIL",
        "expected_static_scene_start": dict(expected),
        "observed_post_sd_pose": dict(observed_pose),
        "planar_position_error_m": planar_error,
        "yaw_error_rad": yaw_error,
        "observed_z_m": oz,
        "maximum_position_error_m": position_limit,
        "maximum_yaw_error_rad": yaw_limit,
        "minimum_z_m": z_min,
        "maximum_z_m": z_max,
        "permission_to_publish_first_nonzero": passed,
        "basis": "fresh post-SD odometry, not the earlier Reset anchor sample",
    }


def validate_runtime_raw_odom_aligned_start_table_approach_start(
    mission: Mapping[str, Any], observed_pose: Mapping[str, Any],
) -> dict[str, Any]:
    """Require fresh STAND to retain the derived pre-physics start pose."""
    validate_runtime_raw_odom_aligned_start_table_approach_mission(mission)
    contract = mission.get("runtime_coordinate_contract")
    if not isinstance(contract, Mapping):
        raise Local01SceneRouteError("aligned-start mission lacks a runtime coordinate contract")
    expected = contract.get("static_scene_start")
    if not isinstance(expected, Mapping):
        raise Local01SceneRouteError("aligned-start mission lacks a static start pose")
    try:
        ex, ey, eyaw = (float(expected[name]) for name in ("x_m", "y_m", "yaw_rad"))
        ox, oy, oz, oyaw = (float(observed_pose[name]) for name in ("x_m", "y_m", "z_m", "yaw_rad"))
        position_limit = float(contract["maximum_start_position_deviation_m"])
        yaw_limit = float(contract["maximum_start_yaw_deviation_rad"])
        z_min = float(contract["minimum_start_z_m"])
        z_max = float(contract["maximum_start_z_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Local01SceneRouteError("aligned-start runtime pose is malformed") from exc
    values = (ex, ey, eyaw, ox, oy, oz, oyaw, position_limit, yaw_limit, z_min, z_max)
    if not all(math.isfinite(value) for value in values) or position_limit <= 0.0 or yaw_limit <= 0.0:
        raise Local01SceneRouteError("aligned-start runtime pose contains non-finite values or invalid limits")
    planar_error = math.hypot(ox - ex, oy - ey)
    yaw_error = abs(_wrap(oyaw - eyaw))
    passed = planar_error <= position_limit and yaw_error <= yaw_limit and z_min <= oz <= z_max
    return {
        "schema": "robotsim.issue80.aligned-start-table-approach-runtime-start-gate.v1",
        "status": "PASS" if passed else "FAIL",
        "expected_static_scene_start": dict(expected),
        "observed_post_sd_pose": dict(observed_pose),
        "planar_position_error_m": planar_error,
        "yaw_error_rad": yaw_error,
        "observed_z_m": oz,
        "maximum_position_error_m": position_limit,
        "maximum_yaw_error_rad": yaw_limit,
        "minimum_z_m": z_min,
        "maximum_z_m": z_max,
        "permission_to_publish_first_nonzero": passed,
        "basis": "fresh post-SD odometry in the pre-physics derived aligned scene, not active qpos mutation",
    }


def validate_short_route_config(route: Mapping[str, Any]) -> None:
    if route.get("schema_version") != SHORT_ROUTE_SCHEMA:
        raise Local01SceneRouteError("unsupported Local01 short-route schema")
    if route.get("route_kind") not in {
        "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING",
        "SIMULATION_ONLY_SHORT_ROUTE_RAW_ODOM_SCENE_WORLD_DIAGNOSTIC_NOT_TABLE_DOCKING",
    }:
        raise Local01SceneRouteError("short route must remain explicitly diagnostic and non-docking")
    if route.get("frame_id") != "odom" or route.get("source_frame_id") != "raicom_scene_world":
        raise Local01SceneRouteError("short route frame contract is invalid")
    for key in ("start", "waypoint", "short_route_end"):
        pose = route.get(key)
        if not isinstance(pose, Mapping):
            raise Local01SceneRouteError(f"short route is missing pose {key}")
        if not all(math.isfinite(float(pose.get(field, float("nan")))) for field in ("x_m", "y_m", "z_m", "yaw_rad")):
            raise Local01SceneRouteError(f"short route pose {key} is non-finite")
    start = route["start"]
    waypoint = route["waypoint"]
    end = route["short_route_end"]
    d1 = math.hypot(waypoint["x_m"] - start["x_m"], waypoint["y_m"] - start["y_m"])
    d2 = math.hypot(end["x_m"] - waypoint["x_m"], end["y_m"] - waypoint["y_m"])
    if abs(d1 - SHORT_ROUTE_STEP_M) > 1e-9 or abs(d2 - SHORT_ROUTE_STEP_M) > 1e-9:
        raise Local01SceneRouteError("short diagnostic route must have two 0.20 m target segments")
    if abs(_wrap(waypoint["yaw_rad"] - start["yaw_rad"])) > 1e-9 or abs(_wrap(end["yaw_rad"] - start["yaw_rad"])) > 1e-9:
        raise Local01SceneRouteError("short diagnostic route must preserve the Reset start heading")
    if route.get("runtime_reset_odom_observed") is not False:
        raise Local01SceneRouteError("static config must not claim a live Reset/odom observation")
    contract = route.get("runtime_coordinate_contract")
    if contract is not None:
        if not isinstance(contract, Mapping):
            raise Local01SceneRouteError("runtime raw-odom contract is malformed")
        required = {
            "published_frame_id": "odom",
            "numeric_coordinate_convention": "RAICOM_SCENE_WORLD_AFTER_LIVE_RESET_GATE",
            "runtime_target_policy": "derive each 0.20 m target from the fresh post-STAND measured pose and yaw",
        }
        if any(contract.get(key) != value for key, value in required.items()):
            raise Local01SceneRouteError("runtime raw-odom contract differs from the reviewed policy")
        for key, expected in (
            ("maximum_start_position_deviation_m", RUNTIME_RAW_ODOM_START_POSITION_ENVELOPE_M),
            ("maximum_start_yaw_deviation_rad", RUNTIME_RAW_ODOM_START_YAW_ENVELOPE_RAD),
            ("minimum_start_z_m", RUNTIME_RAW_ODOM_START_Z_MIN_M),
            ("maximum_start_z_m", RUNTIME_RAW_ODOM_START_Z_MAX_M),
        ):
            if abs(float(contract.get(key, float("nan"))) - expected) > 1e-12:
                raise Local01SceneRouteError("runtime raw-odom start envelope differs from the reviewed policy")
    limits = route.get("limits")
    if not isinstance(limits, Mapping) or (
        limits.get("forward_m_s") != SHORT_ROUTE_FORWARD_M_S
        or limits.get("angular_rad_s") != SHORT_ROUTE_ANGULAR_RAD_S
        or limits.get("max_cross_track_m") != SHORT_ROUTE_MAX_CROSS_TRACK_M
        or limits.get("max_windows") != SHORT_ROUTE_MAX_WINDOWS
        or limits.get("max_window_s") != SHORT_ROUTE_MAX_WINDOW_S
        or limits.get("max_aggregate_nonzero_s") != SHORT_ROUTE_MAX_AGGREGATE_S
    ):
        raise Local01SceneRouteError("short diagnostic route exceeds the existing bounded motion profile")


def _world_geom_aabb(model: Any, data: Any, geom_id: int, np: Any) -> tuple[Any, Any]:
    local = model.geom_aabb[geom_id]
    center_local = local[:3]
    half_local = local[3:6]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    center_world = data.geom_xpos[geom_id] + rotation @ center_local
    half_world = np.abs(rotation) @ half_local
    return center_world - half_world, center_world + half_world


def native_name_scene_tree(scene_path: str | Path) -> ET.Element:
    """Resolve include paths and change exactly the X2 attach namespace.

    The vendor X2 file and Local01 input remain immutable. Environment prefixes
    and every frame, joint, geom, and physical setting are left untouched.
    """
    path = Path(scene_path).resolve(strict=True)
    tree = ET.parse(path).getroot()
    for item in tree.findall("./asset/model"):
        item.set("file", str((path.parent / item.attrib["file"]).resolve(strict=True)))
    attach = tree.findall("./worldbody/frame/attach[@model='x2_mc_runtime']")
    if len(attach) != 1 or attach[0].get("prefix") != "x2_":
        raise Local01SceneRouteError("expected exactly one source X2 attach with prefix x2_")
    attach[0].set("prefix", "")
    return tree


def aligned_start_scene_tree(scene_path: str | Path) -> ET.Element:
    """Build the explicit, evidence-only zero-yaw RAICOM start wrapper.

    The immutable Local01 scene remains the input.  The only geometric delta
    is the X2 attachment frame quaternion, set before MuJoCo creates its first
    ``MjData``.  It is deliberately not a runtime qpos write, a reset-time
    teleport, or a substitute for the source scene's unsupported initial turn.
    """
    tree = native_name_scene_tree(scene_path)
    frame = _x2_start_frame(tree)
    _validate_source_x2_start_frame(frame)
    frame.set("quat", _format_xml_vector(X2_ALIGNED_START_QUAT_WXYZ))
    return tree


def set_native_x2_timestep(scene_root: ET.Element) -> dict[str, Any]:
    """Set only the top-level scene timestep to the pinned X2 source rate.

    Attached MuJoCo models do not propagate their ``<option>`` into the parent
    scene. The Local01 workspace therefore compiles at MuJoCo's 0.002 s default
    although the pinned runtime X2 source declares 0.001 s. This wrapper delta
    changes no robot or environment geometry, contact, actuator, or controller
    value; it makes the parent scene inherit the source robot's physics rate.
    """
    option = scene_root.find("./option")
    if option is not None:
        raise Local01SceneRouteError(
            "Local01 scene unexpectedly declares a top-level option; native timestep delta is ambiguous"
        )
    option = ET.Element("option", {"timestep": f"{NATIVE_X2_TIMESTEP_S:.3f}"})
    # Keep the stable visual/asset ordering intact and insert after optional
    # statistic/visual blocks, before the asset section.
    insert_at = next(
        (index for index, child in enumerate(list(scene_root)) if child.tag == "asset"),
        len(scene_root),
    )
    scene_root.insert(insert_at, option)
    return {
        "enabled": True,
        "version": NATIVE_X2_TIMESTEP_SCENE_VERSION,
        "parent_scene_default_timestep_s": COMBINED_SCENE_DEFAULT_TIMESTEP_S,
        "native_x2_source_timestep_s": NATIVE_X2_TIMESTEP_S,
        "applies_before_physics": True,
        "active_simulation_qpos_write": False,
    }


def audit_runtime_name_contract(native: Any, candidate: Any, config_path: Path) -> dict[str, Any]:
    """Enumerate the full native inventory plus actual model-manager YAML lookups."""
    import mujoco
    import yaml

    groups = {"body": "nbody", "joint": "njnt", "actuator": "nu", "sensor": "nsensor"}
    inventory: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for kind, count in groups.items():
        obj = getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper())
        names = [mujoco.mj_id2name(native, obj, i) for i in range(getattr(native, count))]
        names = [name for name in names if name and name != "world"]
        inventory[kind] = names
        missing[kind] = [name for name in names if mujoco.mj_name2id(candidate, obj, name) < 0]
        actual = [mujoco.mj_id2name(candidate, obj, i) for i in range(getattr(candidate, count))]
        named = [name for name in actual if name]
        if len(named) != len(set(named)):
            raise Local01SceneRouteError(f"duplicate compiled names in {kind} namespace")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_joints = [name for group in cfg["actual"]["active_joint_name"] for name in group]
    configured_sensor_groups = dict(zip(cfg["sensor"]["sensor_frame_type"], cfg["sensor"]["sensor_name"]))
    baseline_missing_sensors = [
        name for frame, names in configured_sensor_groups.items() if frame != "odom"
        for name in names if mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_SENSOR, name) < 0
    ]
    missing_configured = {
        "joint": [name for name in configured_joints
                  if mujoco.mj_name2id(candidate, mujoco.mjtObj.mjOBJ_JOINT, name) < 0],
        "odom_body": [name for name in configured_sensor_groups.get("odom", [])
                      if mujoco.mj_name2id(candidate, mujoco.mjtObj.mjOBJ_BODY, name) < 0],
        "sensor_present_in_native": [
            name for frame, names in configured_sensor_groups.items() if frame != "odom"
            for name in names if name not in baseline_missing_sensors
            and mujoco.mj_name2id(candidate, mujoco.mjtObj.mjOBJ_SENSOR, name) < 0
        ],
    }
    free_joints = [
        mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(native.njnt) if native.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    return {
        "status": ("PASS_NATIVE_RUNTIME_NAMES" if not any(missing.values())
                   and not any(missing_configured.values()) else "FAIL_NATIVE_RUNTIME_NAMES"),
        "native_inventory": inventory,
        "original_to_corrected_names": {
            kind: [{"original": "x2_" + name, "corrected": name} for name in names]
            for kind, names in inventory.items()
        },
        "missing_native_names": missing,
        "native_freejoint_names": free_joints,
        "configured_active_joints": configured_joints,
        "configured_sensor_groups": configured_sensor_groups,
        "missing_configured_native_lookups": missing_configured,
        "preexisting_configured_sensors_absent_from_native_model": baseline_missing_sensors,
        "duplicate_names_by_namespace": [],
        "scope": "all compiled native bodies/joints/actuators/sensors; YAML is the documented model-manager lookup contract, not proprietary implementation source",
    }


def assert_scene_name_only_equivalence(original: Any, corrected: Any) -> dict[str, Any]:
    """Compare every public numeric compiled-model array, excluding name addresses.

    Object ordering must remain identical. This includes robot and environment
    meshes, inertias, limits, actuators, sensors, contacts, qpos0 and solver data.
    Forward kinematics is sampled without any integration.
    """
    import mujoco
    import numpy as np

    checked = []
    for field in dir(original):
        if (field.startswith("_") or field.startswith("name_")
                or field == "names_map" or field.endswith("_pathadr")):
            continue
        value = getattr(original, field)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            other = getattr(corrected, field)
            if not np.array_equal(value, other, equal_nan=True):
                raise Local01SceneRouteError(f"name-only wrapper changed compiled physical array {field}")
            checked.append(field)
    for count in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nsensor", "nsite", "neq"):
        if getattr(original, count) != getattr(corrected, count):
            raise Local01SceneRouteError(f"name-only wrapper changed {count}")
    for field in dir(original.opt):
        if field.startswith("_"):
            continue
        value, other = getattr(original.opt, field), getattr(corrected.opt, field)
        if isinstance(value, (int, float, np.ndarray)) and not np.array_equal(value, other):
            raise Local01SceneRouteError(f"name-only wrapper changed solver option {field}")
    a, b = mujoco.MjData(original), mujoco.MjData(corrected)
    max_residual = 0.0
    for fraction in (None, 0.25, 0.50, 0.75):
        mujoco.mj_resetData(original, a)
        mujoco.mj_resetData(corrected, b)
        if fraction is not None:
            for j in range(original.njnt):
                if original.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and original.jnt_limited[j]:
                    lo, hi = original.jnt_range[j]
                    q = lo + fraction * (hi - lo)
                    a.qpos[original.jnt_qposadr[j]] = q
                    b.qpos[corrected.jnt_qposadr[j]] = q
        mujoco.mj_forward(original, a)
        mujoco.mj_forward(corrected, b)
        for field in ("xpos", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat"):
            if not getattr(a, field).size:
                continue
            residual = float(np.max(np.abs(getattr(a, field) - getattr(b, field))))
            max_residual = max(max_residual, residual)
        if max_residual > 1e-12:
            raise Local01SceneRouteError("name-only wrapper changed forward kinematics")
    return {
        "status": "PASS_COMPILED_PHYSICS_EQUIVALENT",
        "numeric_arrays_equal": checked,
        "numeric_array_count": len(checked),
        "options_equal": True,
        "static_fk_samples": 4,
        "maximum_fk_residual": max_residual,
        "initial_qpos_equal": True,
        "environment_geometry_and_placements_equal": True,
        "mj_step_calls": 0,
    }


def assert_aligned_start_scene_equivalence(
    original: Any,
    aligned: Any,
    *,
    native_x2_timestep: bool = False,
) -> dict[str, Any]:
    """Verify the documented pre-physics frame rotation changed nothing else.

    A MuJoCo frame rotation necessarily changes pose-derived compiled arrays
    such as ``qpos0`` and world-frame inverse weights.  Every remaining public
    numeric model array must still match exactly.  Robot inertias, limits,
    actuator mappings, contact settings, environment transforms, and solver
    options are then checked independently below.
    """
    import mujoco
    import numpy as np

    expected_pose_derived_deltas = {
        "actuator_acc0",
        "body_invweight0",
        "body_pos",
        "body_quat",
        "dof_M0",
        "dof_invweight0",
        "light_pos0",
        "light_poscom0",
        "qpos0",
        "qpos_spring",
    }
    changed: list[str] = []
    for field in dir(original):
        if (
            field.startswith("_")
            or field.startswith("name_")
            or field == "names_map"
            or field.endswith("_pathadr")
        ):
            continue
        value = getattr(original, field)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            other = getattr(aligned, field)
            if not np.array_equal(value, other, equal_nan=True):
                changed.append(field)
                if field not in expected_pose_derived_deltas:
                    raise Local01SceneRouteError(
                        f"aligned-start wrapper changed unsupported compiled physical array {field}"
                    )
    if set(changed) != expected_pose_derived_deltas:
        raise Local01SceneRouteError(
            "aligned-start wrapper changed an incomplete or unexpected set of pose-derived arrays"
        )
    for count in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nsensor", "nsite", "neq"):
        if getattr(original, count) != getattr(aligned, count):
            raise Local01SceneRouteError(f"aligned-start wrapper changed {count}")
    changed_options: dict[str, tuple[float, float]] = {}
    for field in dir(original.opt):
        if field.startswith("_"):
            continue
        value, other = getattr(original.opt, field), getattr(aligned.opt, field)
        if isinstance(value, (int, float, np.ndarray)) and not np.array_equal(value, other):
            if (
                native_x2_timestep
                and field == "timestep"
                and math.isclose(float(value), COMBINED_SCENE_DEFAULT_TIMESTEP_S, abs_tol=1e-12)
                and math.isclose(float(other), NATIVE_X2_TIMESTEP_S, abs_tol=1e-12)
            ):
                changed_options[field] = (float(value), float(other))
                continue
            raise Local01SceneRouteError(f"aligned-start wrapper changed solver option {field}")
    if native_x2_timestep:
        if changed_options != {"timestep": (COMBINED_SCENE_DEFAULT_TIMESTEP_S, NATIVE_X2_TIMESTEP_S)}:
            raise Local01SceneRouteError("native timestep wrapper changed an unexpected solver option set")
    elif changed_options:
        raise Local01SceneRouteError("aligned-start wrapper changed solver options")

    root_a = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    root_b = mujoco.mj_name2id(aligned, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_a < 0 or root_b < 0 or original.jnt_type[root_a] != mujoco.mjtJoint.mjJNT_FREE:
        raise Local01SceneRouteError("aligned-start wrapper lacks the native floating base")
    qadr_a, qadr_b = int(original.jnt_qposadr[root_a]), int(aligned.jnt_qposadr[root_b])
    if qadr_a != qadr_b:
        raise Local01SceneRouteError("aligned-start wrapper changed freejoint qpos addressing")
    if not np.allclose(
        original.qpos0[:qadr_a + 3], aligned.qpos0[:qadr_b + 3], rtol=0, atol=1e-12
    ):
        raise Local01SceneRouteError("aligned-start wrapper changed root position or prior qpos values")
    if not np.allclose(aligned.qpos0[qadr_b + 3:qadr_b + 7], X2_ALIGNED_START_QUAT_WXYZ, rtol=0, atol=1e-12):
        raise Local01SceneRouteError("aligned-start wrapper did not set the root yaw-zero quaternion")
    if not np.allclose(
        original.qpos0[qadr_a + 7:], aligned.qpos0[qadr_b + 7:], rtol=0, atol=1e-12
    ):
        raise Local01SceneRouteError("aligned-start wrapper changed passive joints or environment qpos")

    # Check contact/collision material fields explicitly instead of inferring
    # them from the tolerated pose-dependent model arrays above.
    invariant_fields = (
        "body_mass", "body_inertia", "body_ipos", "body_iquat",
        "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
        "jnt_actfrcrange", "geom_type", "geom_size", "geom_dataid",
        "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
        "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
        "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gainprm",
        "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange",
        "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited",
        "sensor_type", "sensor_objtype", "sensor_objid", "sensor_dim",
        "sensor_noise", "sensor_cutoff", "sensor_datatype",
    )
    for field in invariant_fields:
        if not np.array_equal(getattr(original, field), getattr(aligned, field), equal_nan=True):
            raise Local01SceneRouteError(f"aligned-start wrapper changed invariant field {field}")

    data_original, data_aligned = mujoco.MjData(original), mujoco.MjData(aligned)
    mujoco.mj_forward(original, data_original)
    mujoco.mj_forward(aligned, data_aligned)
    pelvis = mujoco.mj_name2id(aligned, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_bodies = {pelvis}
    for body_id in range(pelvis + 1, aligned.nbody):
        if int(aligned.body_parentid[body_id]) in robot_bodies:
            robot_bodies.add(body_id)
    environment_geoms = [
        geom_id for geom_id in range(aligned.ngeom)
        if int(aligned.geom_bodyid[geom_id]) not in robot_bodies
    ]
    environment_bodies = sorted({int(aligned.geom_bodyid[geom_id]) for geom_id in environment_geoms})
    for body_id in environment_bodies:
        if not np.array_equal(data_original.xpos[body_id], data_aligned.xpos[body_id]):
            raise Local01SceneRouteError("aligned-start wrapper moved an environment body")
        if not np.array_equal(data_original.xmat[body_id], data_aligned.xmat[body_id]):
            raise Local01SceneRouteError("aligned-start wrapper rotated an environment body")
    for geom_id in environment_geoms:
        if not np.array_equal(data_original.geom_xpos[geom_id], data_aligned.geom_xpos[geom_id]):
            raise Local01SceneRouteError("aligned-start wrapper moved an environment geom")
        if not np.array_equal(data_original.geom_xmat[geom_id], data_aligned.geom_xmat[geom_id]):
            raise Local01SceneRouteError("aligned-start wrapper rotated an environment geom")
    return {
        "status": (
            "PASS_ALIGNED_START_NATIVE_TIMESTEP_PHYSICAL_INVARIANTS"
            if native_x2_timestep else "PASS_ALIGNED_START_PHYSICAL_INVARIANTS"
        ),
        "expected_pose_derived_changed_arrays": sorted(expected_pose_derived_deltas),
        "observed_changed_arrays": sorted(changed),
        "solver_options_equal": not native_x2_timestep,
        "solver_options_equal_except_native_timestep": native_x2_timestep,
        "timestep_before_s": float(original.opt.timestep),
        "timestep_after_s": float(aligned.opt.timestep),
        "root_position_unchanged": True,
        "root_quaternion_before": original.qpos0[qadr_a + 3:qadr_a + 7].tolist(),
        "root_quaternion_after": aligned.qpos0[qadr_b + 3:qadr_b + 7].tolist(),
        "passive_joint_and_environment_qpos_unchanged": True,
        "robot_geometry_limits_actuators_and_sensors_unchanged": True,
        "environment_body_count": len(environment_bodies),
        "environment_geom_count": len(environment_geoms),
        "environment_world_placements_unchanged": True,
        "mj_step_calls": 0,
    }


def assert_synchronized_keyframe_addition(
    aligned: Any, keyed: Any, expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a runtime-only keyframe changed no physical model field."""
    import mujoco
    import numpy as np

    if keyed.nkey != aligned.nkey + 1:
        raise Local01SceneRouteError("synchronized initialization must add exactly one keyframe")
    for count in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nsensor", "nsite", "neq"):
        if getattr(aligned, count) != getattr(keyed, count):
            raise Local01SceneRouteError(f"keyframe wrapper changed {count}")
    changed: list[str] = []
    for field in dir(aligned):
        if (
            field.startswith("_")
            or field.startswith("name_")
            or field == "names_map"
            or field.endswith("_pathadr")
            or field.startswith("key_")
        ):
            continue
        value = getattr(aligned, field)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            other = getattr(keyed, field)
            if not np.array_equal(value, other, equal_nan=True):
                changed.append(field)
    if changed:
        raise Local01SceneRouteError(
            "synchronized keyframe changed non-keyframe model fields: " + ", ".join(changed)
        )
    for field in dir(aligned.opt):
        if field.startswith("_"):
            continue
        value, other = getattr(aligned.opt, field), getattr(keyed.opt, field)
        if isinstance(value, (int, float, np.ndarray)) and not np.array_equal(value, other):
            raise Local01SceneRouteError(f"keyframe wrapper changed solver option {field}")
    key_id = mujoco.mj_name2id(keyed, mujoco.mjtObj.mjOBJ_KEY, str(expected["key_name"]))
    if key_id < 0:
        raise Local01SceneRouteError("compiled synchronized keyframe name is missing")
    qpos = np.asarray(expected["qpos"], dtype=float)
    qvel = np.asarray(expected["qvel"], dtype=float)
    if qpos.shape != (keyed.nq,) or qvel.shape != (keyed.nv,):
        raise Local01SceneRouteError("synchronized keyframe dimensions are invalid")
    if not np.allclose(keyed.key_qpos[key_id], qpos, rtol=0, atol=1e-12):
        raise Local01SceneRouteError("compiled synchronized keyframe qpos differs from static derivation")
    if not np.allclose(keyed.key_qvel[key_id], qvel, rtol=0, atol=1e-12):
        raise Local01SceneRouteError("compiled synchronized keyframe qvel differs from static derivation")
    a, b = mujoco.MjData(aligned), mujoco.MjData(keyed)
    mujoco.mj_forward(aligned, a)
    mujoco.mj_forward(keyed, b)
    for field in ("xpos", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat"):
        if not np.array_equal(getattr(a, field), getattr(b, field), equal_nan=True):
            raise Local01SceneRouteError("keyframe wrapper changed qpos0 forward kinematics")
    mujoco.mj_resetDataKeyframe(keyed, b, key_id)
    mujoco.mj_forward(keyed, b)
    if not np.allclose(b.qpos, qpos, rtol=0, atol=1e-12) or not np.allclose(b.qvel, qvel, rtol=0, atol=1e-12):
        raise Local01SceneRouteError("compiled keyframe did not restore the validated state")
    return {
        "status": "PASS_SYNCHRONIZED_KEYFRAME_PHYSICAL_INVARIANTS",
        "key_name": str(expected["key_name"]),
        "key_id": int(key_id),
        "solver_options_equal": True,
        "non_keyframe_numeric_fields_changed": [],
        "qpos0_forward_kinematics_equal": True,
        "keyframe_restore_matches_static_derivation": True,
        "mj_step_calls": 0,
    }


def assert_static_nominal_start_equivalence(
    aligned: Any, static_start: Any, expected_nominal: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a pre-load frame-Z wrapper equals the old nominal keyframe state.

    The vendor ModelManager already applies ``nominal_configuration`` before
    normal stepping.  This check establishes that moving only the scene's X2
    start-frame Z coordinate makes that vendor-owned nominal state physically
    identical to the collision-checked target formerly held in a GUI keyframe.
    No ``mj_step`` is called.
    """
    import mujoco
    import numpy as np

    if aligned.nkey != static_start.nkey:
        raise Local01SceneRouteError("static nominal-start wrapper changed keyframe count")
    for count in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nsensor", "nsite", "neq"):
        if getattr(aligned, count) != getattr(static_start, count):
            raise Local01SceneRouteError(f"static nominal-start wrapper changed {count}")
    invariant_fields = (
        "body_mass", "body_inertia", "body_ipos", "body_iquat",
        "jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
        "jnt_actfrcrange", "geom_type", "geom_size", "geom_dataid",
        "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction",
        "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
        "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_gainprm",
        "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange",
        "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited",
        "sensor_type", "sensor_objtype", "sensor_objid", "sensor_dim",
        "sensor_noise", "sensor_cutoff", "sensor_datatype",
    )
    for field in invariant_fields:
        if not np.array_equal(
            getattr(aligned, field), getattr(static_start, field), equal_nan=True,
        ):
            raise Local01SceneRouteError(
                f"static nominal-start wrapper changed physical field {field}"
            )
    for field in dir(aligned.opt):
        if field.startswith("_"):
            continue
        value, other = getattr(aligned.opt, field), getattr(static_start.opt, field)
        if isinstance(value, (int, float, np.ndarray)) and not np.array_equal(value, other):
            raise Local01SceneRouteError(
                f"static nominal-start wrapper changed solver option {field}"
            )
    root_a = mujoco.mj_name2id(aligned, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    root_b = mujoco.mj_name2id(static_start, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_a < 0 or root_b < 0:
        raise Local01SceneRouteError("static nominal-start wrapper lacks native floating base")
    qadr_a, qadr_b = int(aligned.jnt_qposadr[root_a]), int(static_start.jnt_qposadr[root_b])
    expected_qpos = np.asarray(expected_nominal["qpos"], dtype=float)
    expected_qvel = np.asarray(expected_nominal["qvel"], dtype=float)
    if expected_qpos.shape != (aligned.nq,) or expected_qvel.shape != (aligned.nv,):
        raise Local01SceneRouteError("static nominal-start target state dimensions are invalid")
    if not np.allclose(
        aligned.qpos0[:qadr_a + 2], static_start.qpos0[:qadr_b + 2], rtol=0, atol=1e-12,
    ) or not np.allclose(
        aligned.qpos0[qadr_a + 3:], static_start.qpos0[qadr_b + 3:], rtol=0, atol=1e-12,
    ):
        raise Local01SceneRouteError("static nominal-start wrapper changed qpos0 beyond root Z")
    if not math.isclose(
        float(static_start.qpos0[qadr_b + 2]), float(expected_qpos[qadr_a + 2]), abs_tol=1e-12,
    ):
        raise Local01SceneRouteError("static nominal-start root Z does not equal collision-checked target")

    a, b = mujoco.MjData(aligned), mujoco.MjData(static_start)
    a.qpos[:] = expected_qpos
    b.qpos[:] = expected_qpos
    a.qvel[:] = expected_qvel
    b.qvel[:] = expected_qvel
    mujoco.mj_forward(aligned, a)
    mujoco.mj_forward(static_start, b)
    max_residual = 0.0
    for field in ("xpos", "xmat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat"):
        if getattr(a, field).size:
            max_residual = max(
                max_residual,
                float(np.max(np.abs(getattr(a, field) - getattr(b, field)))),
            )
    if max_residual > 1e-12 or a.ncon != b.ncon:
        raise Local01SceneRouteError(
            "static nominal-start wrapper does not reproduce the collision-checked nominal state"
        )
    return {
        "status": "PASS_STATIC_NOMINAL_START_PHYSICAL_INVARIANTS",
        "root_z_before_m": float(aligned.qpos0[qadr_a + 2]),
        "root_z_after_m": float(static_start.qpos0[qadr_b + 2]),
        "root_z_offset_m": float(static_start.qpos0[qadr_b + 2] - aligned.qpos0[qadr_a + 2]),
        "nominal_state_fk_maximum_residual": max_residual,
        "nominal_state_contact_count": int(a.ncon),
        "robot_geometry_limits_actuators_sensors_and_solver_unchanged": True,
        "mj_step_calls": 0,
    }


def assert_native_robot_equivalence(native: Any, combined: Any) -> dict[str, Any]:
    """Check native robot inertias/limits/actuators and FK under the START frame."""
    import mujoco
    import numpy as np

    groups = {
        "body": ("nbody", ("body_mass", "body_inertia", "body_ipos", "body_iquat")),
        "joint": ("njnt", ("jnt_type", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "jnt_actfrcrange")),
        "actuator": ("nu", ("actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited", "actuator_trntype")),
        "sensor": ("nsensor", ("sensor_type", "sensor_dim", "sensor_noise", "sensor_cutoff", "sensor_datatype")),
    }
    mappings: dict[str, list[tuple[int, int]]] = {}
    for kind, (count, fields) in groups.items():
        obj = getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper())
        mappings[kind] = []
        for i in range(getattr(native, count)):
            name = mujoco.mj_id2name(native, obj, i)
            if not name or name == "world":
                continue
            j = mujoco.mj_name2id(combined, obj, name)
            if j < 0:
                raise Local01SceneRouteError(f"missing native {kind}: {name}")
            mappings[kind].append((i, j))
            for field in fields:
                if not np.allclose(getattr(native, field)[i], getattr(combined, field)[j], rtol=0, atol=1e-12):
                    raise Local01SceneRouteError(f"native robot changed {name}: {field}")
    for i, j in mappings["actuator"]:
        ni, cj = int(native.actuator_trnid[i, 0]), int(combined.actuator_trnid[j, 0])
        if mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_JOINT, ni) != mujoco.mj_id2name(combined, mujoco.mjtObj.mjOBJ_JOINT, cj):
            raise Local01SceneRouteError("native actuator joint transmission changed")
    a, b = mujoco.MjData(native), mujoco.MjData(combined)
    mujoco.mj_forward(native, a)
    mujoco.mj_forward(combined, b)
    pelvis_a = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_b = mujoco.mj_name2id(combined, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rotation = b.xmat[pelvis_b].reshape(3, 3) @ a.xmat[pelvis_a].reshape(3, 3).T
    translation = b.xpos[pelvis_b] - rotation @ a.xpos[pelvis_a]
    max_position, max_orientation = 0.0, 0.0
    for fraction in (None, 0.25, 0.5, 0.75):
        mujoco.mj_resetData(native, a)
        mujoco.mj_resetData(combined, b)
        if fraction is not None:
            for i, j in mappings["joint"]:
                if native.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE and native.jnt_limited[i]:
                    lo, hi = native.jnt_range[i]
                    a.qpos[native.jnt_qposadr[i]] = b.qpos[combined.jnt_qposadr[j]] = lo + fraction * (hi - lo)
        mujoco.mj_forward(native, a)
        mujoco.mj_forward(combined, b)
        for i, j in mappings["body"]:
            max_position = max(max_position, float(np.max(np.abs(b.xpos[j] - (rotation @ a.xpos[i] + translation)))))
            max_orientation = max(max_orientation, float(np.max(np.abs(b.xmat[j].reshape(3, 3) - rotation @ a.xmat[i].reshape(3, 3)))))
    if max_position > 1e-12 or max_orientation > 1e-12:
        raise Local01SceneRouteError("native X2 kinematics differ beyond the existing scene START transform")
    return {
        "status": "PASS_NATIVE_ROBOT_UNDER_SCENE_START_TRANSFORM",
        "checked_object_counts": {kind: len(pairs) for kind, pairs in mappings.items()},
        "physical_fields": {kind: list(fields) for kind, (_, fields) in groups.items()},
        "static_fk_samples": 4,
        "maximum_body_position_residual_m": max_position,
        "maximum_body_rotation_matrix_residual": max_orientation,
        "mj_step_calls": 0,
    }


def validate_short_route_static_scene(
    scene_root: str | Path,
    route: Mapping[str, Any],
    *,
    minimum_clearance_m: float = 0.05,
    maximum_translation_sample_m: float = 0.05,
    maximum_heading_sample_rad: float = math.radians(5.0),
) -> dict[str, Any]:
    """Static MuJoCo scene/collision preflight; uses mj_resetData/mj_forward only."""
    validate_short_route_config(route)
    root = Path(scene_root)
    scene_path = root / LOCAL01_SCENE_RELATIVE.relative_to("simulation/mujoco/scenes/raicom2026")
    if scene_path.is_symlink() or not scene_path.is_file():
        raise Local01SceneRouteError("portable RAICOM scene_workspace.xml is missing")
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise Local01SceneRouteError("MuJoCo 3.3.6 and NumPy are required for static scene preflight") from exc
    if mujoco.__version__ != "3.3.6":
        raise Local01SceneRouteError(f"static scene preflight requires MuJoCo 3.3.6, got {mujoco.__version__}")
    try:
        original = mujoco.MjModel.from_xml_path(str(scene_path))
        scene_tree = native_name_scene_tree(scene_path)
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene_tree, encoding="unicode"))
        x2_source = Path(scene_tree.find("./asset/model[@name='x2_mc_runtime']").attrib["file"])
        native = mujoco.MjModel.from_xml_path(str(x2_source.with_name("scene.xml")))
        name_contract = audit_runtime_name_contract(native, model, x2_source.with_name("default.yaml"))
        original_contract = audit_runtime_name_contract(native, original, x2_source.with_name("default.yaml"))
        if name_contract["status"] != "PASS_NATIVE_RUNTIME_NAMES":
            raise Local01SceneRouteError("compiled RAICOM scene does not satisfy native X2 name lookups")
        equivalence = assert_scene_name_only_equivalence(original, model)
        native_equivalence = assert_native_robot_equivalence(native, model)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        raise Local01SceneRouteError(f"RAICOM scene failed MuJoCo static load: {type(exc).__name__}: {exc}") from exc
    if int(model.nq) < 7:
        raise Local01SceneRouteError("compiled RAICOM scene does not contain a free root pose")
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_body < 0 or root_joint < 0:
        raise Local01SceneRouteError("compiled scene is missing the expected X2 runtime base/freejoint")
    source_start = route["scene_world_poses"]["start"]
    expected = _pose_xyz_yaw(source_start, "scene-world START")
    actual_xyz = data.xpos[root_body]
    actual_quat = data.xquat[root_body]
    actual_yaw = math.atan2(
        2.0 * (actual_quat[0] * actual_quat[3] + actual_quat[1] * actual_quat[2]),
        1.0 - 2.0 * (actual_quat[2] ** 2 + actual_quat[3] ** 2),
    )
    if (
        math.dist(actual_xyz.tolist(), expected[:3]) > 1e-8
        or abs(_wrap(actual_yaw - expected[3])) > 1e-8
    ):
        raise Local01SceneRouteError("compiled MuJoCo qpos0 root pose does not match Local01 scene-world START")

    qposadr = int(model.jnt_qposadr[root_joint])
    route_scene_poses = [
        route["scene_world_poses"][key]
        for key in ("start", "waypoint", "short_route_end")
    ]
    cross_track_inflation = float(route["limits"]["max_cross_track_m"])
    robot_bodies = {root_body}
    for body_id in range(root_body + 1, model.nbody):
        if int(model.body_parentid[body_id]) in robot_bodies:
            robot_bodies.add(body_id)
    robot_geoms: list[int] = []
    scene_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_id in robot_bodies:
            robot_geoms.append(geom_id)
        elif body_name.startswith(("raicom_", "mug_")):
            scene_geoms.append(geom_id)
    root_xy = data.xpos[root_body][:2]
    footprint_bound = max(
        float(np.linalg.norm(data.geom_xpos[geom_id][:2] - root_xy))
        + float(model.geom_rbound[geom_id])
        for geom_id in robot_geoms
    )

    segment_results = []
    all_external_contacts: list[dict[str, Any]] = []
    all_self_contacts: list[dict[str, Any]] = []
    sample_total = 0
    runtime_contract = route.get("runtime_coordinate_contract")
    route_scenarios: list[tuple[str, list[Mapping[str, Any]]]] = [
        ("STATIC_NOMINAL_START", route_scene_poses)
    ]
    if isinstance(runtime_contract, Mapping):
        position_limit = float(runtime_contract["maximum_start_position_deviation_m"])
        yaw_limit = float(runtime_contract["maximum_start_yaw_deviation_rad"])
        z_values = (
            float(runtime_contract["minimum_start_z_m"]),
            float(runtime_contract["maximum_start_z_m"]),
        )
        start_x, start_y, _, start_yaw = _pose_xyz_yaw(route_scene_poses[0], "runtime raw-odom static start")
        for x_offset, y_offset, z, yaw_offset in product(
            (-position_limit, 0.0, position_limit),
            (-position_limit, 0.0, position_limit),
            z_values,
            (-yaw_limit, 0.0, yaw_limit),
        ):
            yaw = _wrap(start_yaw + yaw_offset)
            dynamic_start = _scene_pose(start_x + x_offset, start_y + y_offset, z, yaw)
            dynamic_waypoint = _scene_pose(
                start_x + x_offset + SHORT_ROUTE_STEP_M * math.cos(yaw),
                start_y + y_offset + SHORT_ROUTE_STEP_M * math.sin(yaw), z, yaw,
            )
            dynamic_end = _scene_pose(
                start_x + x_offset + SHORT_ROUTE_TARGETS_M[-1] * math.cos(yaw),
                start_y + y_offset + SHORT_ROUTE_TARGETS_M[-1] * math.sin(yaw), z, yaw,
            )
            route_scenarios.append((
                "RAW_ODOM_ENVELOPE_"
                f"dx{x_offset:+.3f}_dy{y_offset:+.3f}_z{z:.3f}_dyaw{yaw_offset:+.3f}",
                [dynamic_start, dynamic_waypoint, dynamic_end],
            ))
    for scenario_id, scenario_poses in route_scenarios:
      for segment_index in range(2):
        start_pose = _pose_xyz_yaw(scenario_poses[segment_index], "short route segment start")
        end_pose = _pose_xyz_yaw(scenario_poses[segment_index + 1], "short route segment end")
        length = math.hypot(end_pose[0] - start_pose[0], end_pose[1] - start_pose[1])
        yaw_delta = abs(_wrap(end_pose[3] - start_pose[3]))
        sample_count = max(
            2,
            math.ceil(length / maximum_translation_sample_m) + 1,
            math.ceil(yaw_delta / maximum_heading_sample_rad) + 1,
        )
        robot_boxes: dict[int, list[tuple[Any, Any]]] = {g: [] for g in robot_geoms}
        scene_boxes: dict[int, tuple[Any, Any]] = {}
        for sample_index in range(sample_count):
            fraction = sample_index / (sample_count - 1)
            x = start_pose[0] + fraction * (end_pose[0] - start_pose[0])
            y = start_pose[1] + fraction * (end_pose[1] - start_pose[1])
            z = start_pose[2] + fraction * (end_pose[2] - start_pose[2])
            yaw = _wrap(start_pose[3] + fraction * _wrap(end_pose[3] - start_pose[3]))
            mujoco.mj_resetData(model, data)
            data.qpos[qposadr:qposadr + 3] = [x, y, z]
            data.qpos[qposadr + 3:qposadr + 7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            mujoco.mj_forward(model, data)
            sample_total += 1
            for geom_id in robot_geoms:
                robot_boxes[geom_id].append(_world_geom_aabb(model, data, geom_id, np))
            if sample_index == 0:
                scene_boxes = {geom_id: _world_geom_aabb(model, data, geom_id, np) for geom_id in scene_geoms}
            for contact in data.contact[:data.ncon]:
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])) or ""
                body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])) or ""
                robot1 = int(model.geom_bodyid[geom1]) in robot_bodies
                robot2 = int(model.geom_bodyid[geom2]) in robot_bodies
                if (robot1 and body2.startswith(("raicom_", "mug_"))) or (
                    robot2 and body1.startswith(("raicom_", "mug_"))
                ):
                    all_external_contacts.append({
                        "scenario": scenario_id,
                        "segment_index": segment_index,
                        "distance_m": float(contact.dist),
                        "body1": body1,
                        "body2": body2,
                        "sample_index": sample_index,
                    })
                elif robot1 and robot2:
                    all_self_contacts.append({
                        "scenario": scenario_id,
                        "segment_index": segment_index,
                        "distance_m": float(contact.dist),
                        "body1": body1,
                        "body2": body2,
                        "sample_index": sample_index,
                    })
        min_clearance = math.inf
        limiting_pair: dict[str, Any] | None = None
        for robot_geom, boxes in robot_boxes.items():
            robot_min = np.min(np.stack([box[0] for box in boxes]), axis=0)
            robot_max = np.max(np.stack([box[1] for box in boxes]), axis=0)
            # Include the full allowed lateral tracking error in the static
            # scene corridor. Inflating both world XY axes by the bound is a
            # conservative superset of every displacement within the 50 mm
            # cross-track guard for this fixed-heading route.
            robot_min[:2] -= cross_track_inflation
            robot_max[:2] += cross_track_inflation
            for scene_geom, (scene_min, scene_max) in scene_boxes.items():
                if not (
                    (int(model.geom_contype[robot_geom]) & int(model.geom_conaffinity[scene_geom]))
                    or (int(model.geom_contype[scene_geom]) & int(model.geom_conaffinity[robot_geom]))
                ):
                    continue
                gap = np.maximum(0.0, np.maximum(scene_min - robot_max, robot_min - scene_max))
                clearance = float(np.linalg.norm(gap))
                if clearance < min_clearance:
                    min_clearance = clearance
                    limiting_pair = {
                        "robot_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])),
                        "robot_geom_id": int(robot_geom),
                        "scene_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[scene_geom])),
                        "scene_geom_id": int(scene_geom),
                    }
        segment_results.append({
            "scenario": scenario_id,
            "index": segment_index,
            "from": ("START", "WAYPOINT")[segment_index],
            "to": ("WAYPOINT", "SHORT_ROUTE_END")[segment_index],
            "static_samples": sample_count,
            "minimum_swept_aabb_clearance_lower_bound_m": min_clearance,
            "limiting_pair": limiting_pair,
        })
    if all_external_contacts or all_self_contacts:
        status = "FAIL_ACTIVE_ROBOT_OR_SELF_CONTACT"
    elif any(
        item["minimum_swept_aabb_clearance_lower_bound_m"] < minimum_clearance_m
        for item in segment_results
    ):
        status = "FAIL_STATIC_CLEARANCE_MARGIN"
    else:
        status = "PASS_STATIC_SCENE_CLEARANCE"
    return {
        "schema": "robotsim.issue80.local01-short-route-static-preflight.v1",
        "status": status,
        "classification": "STATIC_MUJOCO_FORWARD_ONLY_NO_DYNAMICS_NOT_TABLE_DOCKING_VALIDATION",
        "mujoco_version": mujoco.__version__,
        "mj_step_calls": 0,
        "source_local01_commit": route.get("source_local01_commit"),
        "source_scene_world_poses_sha256": route.get("source_pose_sha256"),
        "scene_workspace_sha256": _sha256(scene_path),
        "scene_wrapper_change_classification": "X2_ATTACH_PREFIX_ONLY_AND_MODEL_FILE_PATHS",
        "native_runtime_name_contract": name_contract,
        "original_prefixed_name_contract": original_contract,
        "compiled_name_only_equivalence": equivalence,
        "native_robot_equivalence": native_equivalence,
        "native_timestep_s": float(native.opt.timestep),
        "combined_scene_timestep_s": float(model.opt.timestep),
        "native_timestep_matches_combined_scene": bool(native.opt.timestep == model.opt.timestep),
        "compiled_root_pose_matches_scene_world_START": True,
        "robot_collision_geom_count": len(robot_geoms),
        "scene_collision_geom_count": len(scene_geoms),
        "robot_collision_footprint_radial_upper_bound_m": footprint_bound,
        "cross_track_guard_inflation_m": cross_track_inflation,
        "runtime_raw_odom_start_envelope": dict(runtime_contract) if isinstance(runtime_contract, Mapping) else None,
        "static_route_scenario_count": len(route_scenarios),
        "verified_local_clear_radius_m": 2.0,
        "static_sample_count": sample_total,
        "minimum_required_aabb_margin_m": float(minimum_clearance_m),
        "minimum_swept_aabb_clearance_lower_bound_m": min(
            item["minimum_swept_aabb_clearance_lower_bound_m"] for item in segment_results
        ),
        "segments": segment_results,
        "active_robot_scene_contacts": all_external_contacts,
        "active_robot_self_contacts": all_self_contacts,
        "limitations": [
            "The combined scene's unchanged 0.002 s timestep differs from the native runtime scene's 0.001 s; namespace compatibility is not complete runtime qualification.",
            "Five configured imu_1 sensors are already absent in the untouched native model; no sensors were fabricated.",
            "Static collision geometry only; does not validate dynamic tracking or post-Reset ROS odometry.",
            "The short route ends before the table and is not TABLE_DOCKING; full-distance docking remains NOT RUN.",
            "The configured odom reset anchor is statically derived; a live Reset/odom sample is required before motion starts.",
        ],
    }


def prepare_local01_scene_runtime_bundle(
    source: Mapping[str, Any],
    evidence_directory: str | Path,
    *,
    runtime_root: str | Path,
    aligned_start: bool = False,
    native_x2_timestep: bool = False,
    synchronized_nominal_initialization: bool = False,
    static_nominal_start_initialization: bool = False,
) -> dict[str, Any]:
    """Copy immutable inputs and generate a provenance-tracked runtime wrapper.

    ``aligned_start`` is limited to the separately versioned derived scene. It
    changes the X2 start-frame quaternion before first physics only; all source
    and runtime inputs remain read-only and the original scene is copied as an
    immutable evidence input beside the generated wrapper.  The optional
    synchronized initialization adds one source-MC-derived keyframe; it does
    not alter qpos0, geometry, contacts, gains, or source assets.  The static
    nominal-start variant instead adjusts only the scene attachment Z before
    model load, so the installed simulator's existing nominal-configuration
    initialization starts at the collision-checked root height.
    """
    evidence = Path(evidence_directory)
    if synchronized_nominal_initialization and not aligned_start:
        raise Local01SceneRouteError(
            "synchronized nominal initialization requires the separately reviewed aligned-start scene"
        )
    if static_nominal_start_initialization and not aligned_start:
        raise Local01SceneRouteError(
            "static nominal-start initialization requires the separately reviewed aligned-start scene"
        )
    if synchronized_nominal_initialization and static_nominal_start_initialization:
        raise Local01SceneRouteError(
            "keyframe and static nominal-start initialization are mutually exclusive"
        )
    if native_x2_timestep and not aligned_start:
        raise Local01SceneRouteError(
            "native X2 timestep synchronization requires the separately reviewed aligned-start scene"
        )
    if native_x2_timestep and (
        synchronized_nominal_initialization or static_nominal_start_initialization
    ):
        raise Local01SceneRouteError(
            "native X2 timestep synchronization is a separate direct-stand experiment and cannot combine with keyframe or static-nominal initialization"
        )
    runtime_x2_path = (
        Path(runtime_root)
        / "sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
    )
    if (
        runtime_x2_path.is_symlink()
        or not runtime_x2_path.is_file()
        or _sha256(runtime_x2_path) != LOCAL01_RUNTIME_X2_SHA256
    ):
        raise Local01SceneRouteError(
            "runtime X2 model differs from the source model used by Local01's static scene preflight"
        )
    bundle = evidence / "local01_scene"
    bundle.mkdir(mode=0o755, parents=False, exist_ok=False)
    source_root = Path(str(source["repository_root"]))
    reference_root = Path(str(source["reference_root"]))
    source_files = {
        "scene_world_poses.json": source_root / LOCAL01_POSES_RELATIVE,
        "route_waypoints.json": source_root / LOCAL01_ROUTE_RELATIVE,
        "scene_workspace.xml": source_root / LOCAL01_SCENE_RELATIVE,
        "mug_portable.xml": source_root / LOCAL01_MUG_RELATIVE,
    }
    copied: dict[str, str] = {}
    for name, path in source_files.items():
        target = bundle / name
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        copied[name] = _sha256(target)

    runtime_x2 = "/workspace/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
    room_xml = "/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/scene/room/room_with_chair_review.xml"
    mug_xml = "/workspace/evidence/local01_scene/mug_portable_runtime.xml"
    source_dir = "/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/object/model/mug"

    source_scene_path = source_root / LOCAL01_SCENE_RELATIVE
    scene_root = (
        aligned_start_scene_tree(source_scene_path)
        if aligned_start else native_name_scene_tree(source_scene_path)
    )
    source_scene_root = ET.parse(bundle / "scene_workspace.xml").getroot()
    source_start_frame = _x2_start_frame(source_scene_root)
    _validate_source_x2_start_frame(source_start_frame)
    native_timestep = set_native_x2_timestep(scene_root) if native_x2_timestep else {
        "enabled": False,
        "version": None,
        "parent_scene_default_timestep_s": COMBINED_SCENE_DEFAULT_TIMESTEP_S,
        "native_x2_source_timestep_s": NATIVE_X2_TIMESTEP_S,
        "applies_before_physics": False,
        "active_simulation_qpos_write": False,
    }
    model_paths = {
        "x2_mc_runtime": runtime_x2,
        "raicom_room": room_xml,
        "raicom_mug": mug_xml,
    }
    for model_elem in scene_root.findall("./asset/model"):
        name = model_elem.get("name")
        if name not in model_paths:
            raise Local01SceneRouteError(f"unrecognized Local01 scene model include: {name}")
        model_elem.set("file", model_paths[name])
    attaches = scene_root.findall("./worldbody/frame/attach[@model='x2_mc_runtime']")
    if len(attaches) != 1 or attaches[0].get("prefix") != "":
        raise Local01SceneRouteError("native-name scene wrapper must contain exactly one unprefixed X2 attachment")
    # Restore the allowed deltas and compare the entire parsed XML tree.
    # Environment names, placements and every physical setting must be equal.
    source_models = {item.get("name"): item.get("file") for item in source_scene_root.findall("./asset/model")}
    runtime_models = {item.get("name"): item.get("file") for item in scene_root.findall("./asset/model")}
    if set(source_models) != set(model_paths) or runtime_models != model_paths:
        raise Local01SceneRouteError("Local01 runtime scene wrapper did not resolve exactly the three scene model files")
    for item in scene_root.findall("./asset/model"):
        item.set("file", source_models[item.get("name")])
    attaches[0].set("prefix", "x2_")
    if aligned_start:
        _x2_start_frame(scene_root).set("quat", source_start_frame.get("quat", ""))
    if native_x2_timestep:
        option = scene_root.find("./option")
        if option is None:
            raise Local01SceneRouteError("native timestep wrapper lost its top-level option before source comparison")
        scene_root.remove(option)
    if ET.tostring(scene_root) != ET.tostring(source_scene_root):
        raise Local01SceneRouteError(
            "Local01 runtime scene wrapper changed content beyond allowed model paths, namespace, or aligned start quaternion"
        )
    for item in scene_root.findall("./asset/model"):
        item.set("file", model_paths[item.get("name")])
    attaches[0].set("prefix", "")
    if aligned_start:
        _x2_start_frame(scene_root).set("quat", _format_xml_vector(X2_ALIGNED_START_QUAT_WXYZ))
    if native_x2_timestep:
        set_native_x2_timestep(scene_root)
    scene_runtime_name = (
        "scene_workspace_aligned_start_v3_static_nominal_runtime.xml"
        if static_nominal_start_initialization else
        "scene_workspace_aligned_start_v2_nominal_keyframe_runtime.xml"
        if synchronized_nominal_initialization else
        "scene_workspace_aligned_start_v1_native_x2_timestep_runtime.xml"
        if native_x2_timestep else
        "scene_workspace_aligned_start_v1_runtime.xml"
        if aligned_start else "scene_workspace_runtime.xml"
    )
    scene_runtime = bundle / scene_runtime_name
    ET.ElementTree(scene_root).write(scene_runtime, encoding="utf-8", xml_declaration=True)

    mug_root = ET.parse(bundle / "mug_portable.xml").getroot()
    source_mug_root = ET.parse(bundle / "mug_portable.xml").getroot()
    compiler = mug_root.find("compiler")
    if compiler is None:
        raise Local01SceneRouteError("Local01 portable mug scene has no compiler asset path declaration")
    compiler.set("meshdir", source_dir)
    compiler.set("texturedir", source_dir)
    source_compiler = source_mug_root.find("compiler")
    assert source_compiler is not None
    compiler.set("meshdir", source_compiler.get("meshdir", ""))
    compiler.set("texturedir", source_compiler.get("texturedir", ""))
    if ET.tostring(mug_root) != ET.tostring(source_mug_root):
        raise Local01SceneRouteError("Local01 mug runtime overlay changed collision or visual content beyond compiler asset paths")
    compiler.set("meshdir", source_dir)
    compiler.set("texturedir", source_dir)
    mug_runtime = bundle / "mug_portable_runtime.xml"
    ET.ElementTree(mug_root).write(mug_runtime, encoding="utf-8", xml_declaration=True)

    default_config = Path(runtime_root) / "sim_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml"
    if default_config.is_symlink() or not default_config.is_file():
        raise Local01SceneRouteError("installed simulator default.yaml is missing or unsafe")
    default_text = default_config.read_text(encoding="utf-8")
    old_model_line = "simulation_model: ${SIM_ROBOT_PATH}/model_info/scene.xml"
    if default_text.count(old_model_line) != 1:
        raise Local01SceneRouteError("installed simulator config has an unexpected simulation_model declaration")
    runtime_config_text = default_text.replace(
        old_model_line,
        "simulation_model: /workspace/evidence/local01_scene/scene_workspace_runtime.xml",
        1,
    )
    simulator_config_name = (
        "simulator_default_local01_aligned_start_v3_static_nominal.yaml"
        if static_nominal_start_initialization else
        "simulator_default_local01_aligned_start_v2_nominal_keyframe.yaml"
        if synchronized_nominal_initialization else
        "simulator_default_local01_aligned_start_v1_native_x2_timestep.yaml"
        if native_x2_timestep else
        "simulator_default_local01_aligned_start_v1.yaml"
        if aligned_start else "simulator_default_local01_scene.yaml"
    )
    runtime_scene_container_path = "/workspace/evidence/local01_scene/" + scene_runtime_name
    runtime_simulator_config_container_path = "/workspace/evidence/local01_scene/" + simulator_config_name
    runtime_config_text = runtime_config_text.replace(
        "/workspace/evidence/local01_scene/scene_workspace_runtime.xml",
        runtime_scene_container_path,
    )
    simulator_config = bundle / simulator_config_name
    with simulator_config.open("x", encoding="utf-8") as stream:
        stream.write(runtime_config_text)
    if runtime_config_text.replace(
        "simulation_model: " + runtime_scene_container_path,
        old_model_line,
        1,
    ) != default_text:
        raise Local01SceneRouteError("simulator config overlay changed fields beyond simulation_model path")

    def resolve_container_path(container_path: str) -> Path:
        mappings = (
            ("/workspace/raicom2026", Path(runtime_root)),
            ("/workspace/raicom_scene_source", reference_root),
            ("/workspace/evidence", evidence),
        )
        for container_root, host_root in mappings:
            if container_path == container_root or container_path.startswith(container_root + "/"):
                relative = container_path[len(container_root):].lstrip("/")
                return host_root / relative
        raise Local01SceneRouteError(f"container scene asset path is outside the three narrow mounts: {container_path}")

    runtime_scene_root = ET.parse(scene_runtime).getroot()
    resolved_scene_models: dict[str, str] = {}
    for model_elem in runtime_scene_root.findall("./asset/model"):
        model_name = str(model_elem.get("name", ""))
        model_path = str(model_elem.get("file", ""))
        host_path = resolve_container_path(model_path)
        if host_path.is_symlink() or not host_path.is_file():
            raise Local01SceneRouteError(f"runtime scene model path does not resolve from a mounted source: {model_name}")
        resolved_scene_models[model_name] = str(host_path)
    if set(resolved_scene_models) != {"x2_mc_runtime", "raicom_room", "raicom_mug"}:
        raise Local01SceneRouteError("runtime scene wrapper must resolve exactly X2, room, and mug model sources")
    runtime_mug_root = ET.parse(mug_runtime).getroot()
    runtime_mug_compiler = runtime_mug_root.find("compiler")
    if runtime_mug_compiler is None:
        raise Local01SceneRouteError("runtime mug wrapper lost its compiler asset paths")
    for path_attr, asset_name in (("meshdir", "mug.obj"), ("texturedir", "mug.png")):
        directory = resolve_container_path(str(runtime_mug_compiler.get(path_attr, "")))
        if directory.is_symlink() or not directory.is_dir() or not (directory / asset_name).is_file():
            raise Local01SceneRouteError(f"runtime mug {path_attr} does not resolve from the read-only reference mount")
    simulation_model_lines = [
        line.strip().split(":", 1)[1].strip()
        for line in runtime_config_text.splitlines()
        if line.strip().startswith("simulation_model:")
    ]
    if len(simulation_model_lines) != 1:
        raise Local01SceneRouteError("runtime simulator overlay must have one simulation_model path")
    runtime_simulation_model = resolve_container_path(simulation_model_lines[0])
    if runtime_simulation_model != scene_runtime.resolve():
        raise Local01SceneRouteError("runtime simulator config scene path does not resolve to the staged scene wrapper")

    # Compile the actual generated wrapper with the three declared mounts
    # resolved on the host. The staged mug XML is passed through MuJoCo's VFS,
    # changing only its asset search paths for this offline compilation.
    import mujoco
    host_scene = ET.fromstring(ET.tostring(runtime_scene_root))
    host_mug = ET.fromstring(ET.tostring(runtime_mug_root))
    for attr in ("meshdir", "texturedir"):
        host_mug.find("compiler").set(attr, str(resolve_container_path(host_mug.find("compiler").attrib[attr])))
    for item in host_scene.findall("./asset/model"):
        item.set("file", "offline_mug.xml" if item.get("name") == "raicom_mug"
                 else str(resolve_container_path(item.attrib["file"])))
    generated_model = mujoco.MjModel.from_xml_string(
        ET.tostring(host_scene, encoding="unicode"),
        assets={"offline_mug.xml": ET.tostring(host_mug)},
    )
    corrected_source_model = mujoco.MjModel.from_xml_string(
        ET.tostring(native_name_scene_tree(source_root / LOCAL01_SCENE_RELATIVE), encoding="unicode")
    )
    native_model = mujoco.MjModel.from_xml_path(str(runtime_x2_path.with_name("scene.xml")))
    base_equivalence = (
        assert_aligned_start_scene_equivalence(
            corrected_source_model,
            generated_model,
            native_x2_timestep=native_x2_timestep,
        )
        if aligned_start else assert_scene_name_only_equivalence(corrected_source_model, generated_model)
    )
    static_nominal_start: dict[str, Any] | None = None
    static_nominal_start_equivalence: dict[str, Any] | None = None
    if static_nominal_start_initialization:
        nominal_target = derive_synchronized_nominal_keyframe(
            generated_model, runtime_x2_path.with_name("default.yaml")
        )
        frame = _x2_start_frame(scene_root)
        source_pos = _parse_xml_vector(frame.get("pos"), 3, "aligned x2_start_frame.pos")
        if not (
            math.isclose(source_pos[0], X2_SOURCE_START_POS_M[0], abs_tol=1e-12)
            and math.isclose(source_pos[1], X2_SOURCE_START_POS_M[1], abs_tol=1e-12)
            and math.isclose(source_pos[2], X2_SOURCE_START_POS_M[2], abs_tol=1e-12)
        ):
            raise Local01SceneRouteError(
                "static nominal-start may alter only the source X2 frame Z coordinate"
            )
        frame.set(
            "pos",
            _format_xml_vector((
                source_pos[0],
                source_pos[1],
                source_pos[2] + float(nominal_target["root_height_shift_m"]),
            )),
        )
        ET.ElementTree(scene_root).write(scene_runtime, encoding="utf-8", xml_declaration=True)
        runtime_scene_root = ET.parse(scene_runtime).getroot()
        host_scene = ET.fromstring(ET.tostring(runtime_scene_root))
        for item in host_scene.findall("./asset/model"):
            item.set(
                "file",
                "offline_mug.xml" if item.get("name") == "raicom_mug"
                else str(resolve_container_path(item.attrib["file"])),
            )
        static_model = mujoco.MjModel.from_xml_string(
            ET.tostring(host_scene, encoding="unicode"),
            assets={"offline_mug.xml": ET.tostring(host_mug)},
        )
        static_nominal_start_equivalence = assert_static_nominal_start_equivalence(
            generated_model, static_model, nominal_target,
        )
        post_static_nominal = derive_synchronized_nominal_keyframe(
            static_model, runtime_x2_path.with_name("default.yaml")
        )
        if abs(float(post_static_nominal["root_height_shift_m"])) > 1e-8:
            raise Local01SceneRouteError(
                "static nominal-start frame Z does not produce tangent selected-foot contacts"
            )
        generated_model = static_model
        static_nominal_start = {
            "schema": "robotsim.issue80.static-nominal-start.v1",
            "classification": "SIMULATION_ONLY_SOURCE_MC_NOMINAL_CONFIGURATION_STATIC_FRAME_Z",
            "version": STATIC_NOMINAL_START_VERSION,
            "source_nominal_target": nominal_target,
            "derived_frame_z_offset_m": float(nominal_target["root_height_shift_m"]),
            "derived_start_frame_pos_m": list(_parse_xml_vector(
                _x2_start_frame(runtime_scene_root).get("pos"), 3, "static x2_start_frame.pos",
            )),
            "post_static_contact_validation": post_static_nominal,
            "equivalence": static_nominal_start_equivalence,
            "applies_before_vendor_model_load": True,
            "post_start_qpos_write": False,
            "mj_step_calls": 0,
        }
        with (bundle / "static_nominal_start.json").open("x", encoding="utf-8") as stream:
            json.dump(static_nominal_start, stream, indent=2, sort_keys=True)
            stream.write("\n")
    synchronized_keyframe: dict[str, Any] | None = None
    keyframe_equivalence: dict[str, Any] | None = None
    if synchronized_nominal_initialization:
        synchronized_keyframe = derive_synchronized_nominal_keyframe(
            generated_model, runtime_x2_path.with_name("default.yaml")
        )
        if scene_root.find("./keyframe") is not None:
            raise Local01SceneRouteError("source-aligned scene unexpectedly already contains a keyframe section")
        keyframes = ET.SubElement(scene_root, "keyframe")
        ET.SubElement(
            keyframes,
            "key",
            name=str(synchronized_keyframe["key_name"]),
            qpos=_format_xml_vector(tuple(synchronized_keyframe["qpos"])),
            qvel=_format_xml_vector(tuple(synchronized_keyframe["qvel"])),
        )
        ET.ElementTree(scene_root).write(scene_runtime, encoding="utf-8", xml_declaration=True)
        runtime_scene_root = ET.parse(scene_runtime).getroot()
        host_scene = ET.fromstring(ET.tostring(runtime_scene_root))
        for item in host_scene.findall("./asset/model"):
            item.set("file", "offline_mug.xml" if item.get("name") == "raicom_mug"
                     else str(resolve_container_path(item.attrib["file"])))
        generated_model = mujoco.MjModel.from_xml_string(
            ET.tostring(host_scene, encoding="unicode"),
            assets={"offline_mug.xml": ET.tostring(host_mug)},
        )
        # Rebuild the unkeyed scene from the staged wrapper and compare it to
        # the keyed model explicitly.  The source/vendor files remain read-only.
        unkeyed_scene = ET.parse(scene_runtime).getroot()
        unkeyed_scene.remove(unkeyed_scene.find("./keyframe"))
        host_unkeyed = ET.fromstring(ET.tostring(unkeyed_scene))
        for item in host_unkeyed.findall("./asset/model"):
            item.set("file", "offline_mug.xml" if item.get("name") == "raicom_mug"
                     else str(resolve_container_path(item.attrib["file"])))
        unkeyed_model = mujoco.MjModel.from_xml_string(
            ET.tostring(host_unkeyed, encoding="unicode"),
            assets={"offline_mug.xml": ET.tostring(host_mug)},
        )
        keyframe_equivalence = assert_synchronized_keyframe_addition(
            unkeyed_model, generated_model, synchronized_keyframe
        )
        with (bundle / "synchronized_nominal_keyframe.json").open("x", encoding="utf-8") as stream:
            json.dump(synchronized_keyframe, stream, indent=2, sort_keys=True)
            stream.write("\n")
    generated_equivalence = (
        {
            "status": "PASS_ALIGNED_START_STATIC_NOMINAL_INITIALIZATION_INVARIANTS",
            "aligned_start_equivalence": base_equivalence,
            "static_nominal_start_equivalence": static_nominal_start_equivalence,
        }
        if static_nominal_start_initialization else
        {
            "status": "PASS_ALIGNED_START_KEYFRAME_INITIALIZATION_INVARIANTS",
            "aligned_start_equivalence": base_equivalence,
            "synchronized_keyframe_equivalence": keyframe_equivalence,
        }
        if synchronized_nominal_initialization else base_equivalence
    )
    native_robot_equivalence = assert_native_robot_equivalence(native_model, generated_model)
    generated_contract = audit_runtime_name_contract(native_model, generated_model, runtime_x2_path.with_name("default.yaml"))
    if generated_contract["status"] != "PASS_NATIVE_RUNTIME_NAMES":
        raise Local01SceneRouteError("generated runtime wrapper lost native X2 names")

    records = {
        "schema": "robotsim.issue80.local01-scene-runtime-bundle.v1",
        "classification": (
            "SIMULATION_ONLY_ALIGNED_START_STATIC_NOMINAL_INITIALIZATION_V1"
            if static_nominal_start_initialization else
            "SIMULATION_ONLY_ALIGNED_START_KEYFRAME_SYNCHRONIZED_INITIALIZATION_V1"
            if synchronized_nominal_initialization else
            "SIMULATION_ONLY_ALIGNED_START_NATIVE_X2_TIMESTEP_SCENE_WRAPPER_V1"
            if native_x2_timestep else
            "SIMULATION_ONLY_ALIGNED_START_SCENE_WRAPPER_V1"
            if aligned_start else "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER"
        ),
        "source_local01_commit": source["source_commit"],
        "source_files_sha256": copied,
        "runtime_files_sha256": {
            scene_runtime_name: _sha256(scene_runtime),
            "mug_portable_runtime.xml": _sha256(mug_runtime),
            simulator_config_name: _sha256(simulator_config),
            **({"synchronized_nominal_keyframe.json": _sha256(bundle / "synchronized_nominal_keyframe.json")}
               if synchronized_nominal_initialization else {}),
            **({"static_nominal_start.json": _sha256(bundle / "static_nominal_start.json")}
               if static_nominal_start_initialization else {}),
        },
        "runtime_x2_sha256": _sha256(
            runtime_x2_path
        ),
        "simulator_config_only_change": {
            "field": "simulation_model",
            "from": old_model_line.split(": ", 1)[1],
            "to": runtime_scene_container_path,
        },
        "scene_wrapper_change_classification": (
            "X2_ATTACH_PREFIX_ALIGNED_START_QUATERNION_AND_SOURCE_MC_NOMINAL_STATIC_FRAME_Z_V1"
            if static_nominal_start_initialization else
            "X2_ATTACH_PREFIX_ALIGNED_START_QUATERNION_AND_SOURCE_MC_NOMINAL_KEYFRAME_V1"
            if synchronized_nominal_initialization else
            "X2_ATTACH_PREFIX_ALIGNED_START_QUATERNION_NATIVE_X2_TIMESTEP_V1_AND_MODEL_FILE_PATHS"
            if native_x2_timestep else
            "X2_ATTACH_PREFIX_AND_ALIGNED_START_QUATERNION_V1_AND_MODEL_FILE_PATHS"
            if aligned_start else "X2_ATTACH_PREFIX_ONLY_AND_MODEL_FILE_PATHS"
        ),
        "x2_attach_prefix_change": {"source": "x2_", "runtime": ""},
        "aligned_start": {
            "enabled": aligned_start,
            "version": ALIGNED_START_SCENE_VERSION if aligned_start else None,
            "frame": X2_START_FRAME_NAME,
            "source_pos_m": list(X2_SOURCE_START_POS_M),
            "source_quat_wxyz": list(X2_SOURCE_START_QUAT_WXYZ),
            "derived_quat_wxyz": list(X2_ALIGNED_START_QUAT_WXYZ) if aligned_start else None,
            "applies_before_physics": aligned_start,
            "active_simulation_qpos_write": False,
        },
        "native_x2_timestep": native_timestep,
        "synchronized_nominal_initialization": {
            "enabled": synchronized_nominal_initialization,
            "keyframe": synchronized_keyframe,
            "keyframe_equivalence": keyframe_equivalence,
            "applies_only_via_native_MuJoCo_Load_key_before_physics_release": (
                synchronized_nominal_initialization
            ),
            "post_start_qpos_write": False,
        },
        "static_nominal_start_initialization": {
            "enabled": static_nominal_start_initialization,
            "initialization": static_nominal_start,
            "applies_only_before_vendor_model_load": static_nominal_start_initialization,
            "post_start_qpos_write": False,
        },
        "generated_runtime_name_contract": generated_contract,
        "generated_wrapper_physics_equivalence": generated_equivalence,
        "generated_native_robot_equivalence": native_robot_equivalence,
        "mug_wrapper_change_classification": "COMPILER_ASSET_SEARCH_PATHS_ONLY",
        "reference_scene_mount_target": "/workspace/raicom_scene_source",
        "runtime_simulator_config_path": runtime_simulator_config_container_path,
        "runtime_scene_path": runtime_scene_container_path,
        "runtime_scene_filename": scene_runtime_name,
        "runtime_simulator_config_filename": simulator_config_name,
        "container_path_resolution_status": "PASS_NARROW_READ_ONLY_RUNTIME_REFERENCE_AND_EVIDENCE_MOUNTS",
        "resolved_scene_model_sources": resolved_scene_models,
        "runtime_robot_model_path": "unchanged /workspace/raicom2026/.../model_info/default.yaml",
        "no_vendor_model_or_collision_geometry_changes": True,
        "mj_step_calls": 0,
    }
    manifest = bundle / "runtime_bundle_manifest.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return records


def main() -> int:
    """Generate offline namespace/physics evidence, never start a runtime."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Offline Local01 namespace and compiled-physics audit; no mj_step/runtime")
    parser.add_argument("--poses", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-parent", required=True)
    args = parser.parse_args()
    source = load_local01_scene_inputs(args.poses)
    route = build_short_route_config(source)
    static = validate_short_route_static_scene(
        Path(source["repository_root"]) / "simulation/mujoco/scenes/raicom2026", route,
    )
    if static["status"] != "PASS_STATIC_SCENE_CLEARANCE":
        raise Local01SceneRouteError("corrected static corridor did not pass")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from path_governance import create_unique_output_directory, write_artifact_manifest

    output = create_unique_output_directory(args.output_parent, prefix="local01-native-names").native_path
    bundle = prepare_local01_scene_runtime_bundle(source, output, runtime_root=args.runtime_root)
    # Reuse the established pure overlay builder; inspect bytes only. Do not
    # install an overlay or alter any vendor/runtime configuration.
    from issue80_m1_auto_reset_run import (
        FOUNDATIONWALK_CONFIG_RELATIVE, RUN38_FOUNDATIONWALK_CONFIG_SHA256,
        build_run41_foundationwalk_min_run_overlay,
    )
    foundationwalk = Path(args.runtime_root) / FOUNDATIONWALK_CONFIG_RELATIVE
    if _sha256(foundationwalk) != RUN38_FOUNDATIONWALK_CONFIG_SHA256:
        raise Local01SceneRouteError("FoundationWalk source configuration hash changed")
    _, overlay_audit = build_run41_foundationwalk_min_run_overlay(foundationwalk.read_bytes())
    overlay_audit["builder_default_command_speed_m_s"] = overlay_audit.pop("command_speed_m_s")
    overlay_audit["selected_route_command_speed_m_s"] = route["limits"]["forward_m_s"]
    report = {
        "schema": "robotsim.issue80.native-scene-name-audit.v1",
        "status": "PASS_OFFLINE_NAME_AND_PHYSICS_EQUIVALENCE_NOT_RUNTIME_QUALIFICATION",
        "source_commit": source["source_commit"],
        "source_hashes": source["source_hashes"],
        "static_preflight": static,
        "runtime_bundle": bundle,
        "foundationwalk_overlay_audit": overlay_audit,
        "foundationwalk_profile_conclusion": "existing SIMULATION_ONLY profile unchanged; not unmodified vendor parameter acceptance",
        "runtime_or_reset_attempted": False,
        "mj_step_calls": 0,
    }
    (output / "name_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after = load_local01_scene_inputs(args.poses)
    if after["source_hashes"] != source["source_hashes"]:
        raise Local01SceneRouteError("immutable Local01 sources changed during audit")
    manifest = write_artifact_manifest(output)
    print(json.dumps({
        "status": report["status"], "output_directory": str(output),
        "manifest": manifest["manifest"], "static_clearance_m": static["minimum_swept_aabb_clearance_lower_bound_m"],
        "runtime_validation": "NOT_RUN",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
