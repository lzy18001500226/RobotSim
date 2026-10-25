#!/usr/bin/env python3
"""Local01 RAICOM scene-pose adapter and static short-route preflight.

This module converts the pinned Local01 scene-world pose file into the
existing Issue #80 odom-route input. It does not publish commands or step
MuJoCo. The route targets are a 0.20 m + 0.20 m diagnostic straight ahead
from the configured START pose; they are not the table-docking target.
"""
from __future__ import annotations

import hashlib
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
SHORT_ROUTE_STEP_M = LOCAL01_SHORT_ROUTE_STEP_M
SHORT_ROUTE_TARGETS_M = (SHORT_ROUTE_STEP_M, 2.0 * SHORT_ROUTE_STEP_M)
SHORT_ROUTE_MAX_WINDOWS = LOCAL01_SHORT_ROUTE_MAX_WINDOWS
SHORT_ROUTE_MAX_WINDOW_S = LOCAL01_SHORT_ROUTE_MAX_WINDOW_S
SHORT_ROUTE_MAX_AGGREGATE_S = LOCAL01_SHORT_ROUTE_MAX_AGGREGATE_S
SHORT_ROUTE_FORWARD_M_S = LOCAL01_SHORT_ROUTE_FORWARD_M_S
SHORT_ROUTE_ANGULAR_RAD_S = LOCAL01_SHORT_ROUTE_ANGULAR_RAD_S
SHORT_ROUTE_MAX_CROSS_TRACK_M = LOCAL01_SHORT_ROUTE_MAX_CROSS_TRACK_M


class Local01SceneRouteError(ValueError):
    """Local01 scene provenance, frame, or route validation failed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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


def validate_short_route_config(route: Mapping[str, Any]) -> None:
    if route.get("schema_version") != SHORT_ROUTE_SCHEMA:
        raise Local01SceneRouteError("unsupported Local01 short-route schema")
    if route.get("route_kind") != "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING":
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
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        raise Local01SceneRouteError(f"RAICOM scene failed MuJoCo static load: {type(exc).__name__}: {exc}") from exc
    if int(model.nq) < 7:
        raise Local01SceneRouteError("compiled RAICOM scene does not contain a free root pose")
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2_pelvis")
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "x2_floating_base_joint")
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
    robot_geoms: list[int] = []
    scene_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("x2_"):
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
    for segment_index in range(2):
        start_pose = _pose_xyz_yaw(route_scene_poses[segment_index], "short route segment start")
        end_pose = _pose_xyz_yaw(route_scene_poses[segment_index + 1], "short route segment end")
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
                if (body1.startswith("x2_") and body2.startswith(("raicom_", "mug_"))) or (
                    body2.startswith("x2_") and body1.startswith(("raicom_", "mug_"))
                ):
                    all_external_contacts.append({
                        "segment_index": segment_index,
                        "distance_m": float(contact.dist),
                        "body1": body1,
                        "body2": body2,
                        "sample_index": sample_index,
                    })
                elif body1.startswith("x2_") and body2.startswith("x2_"):
                    all_self_contacts.append({
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
        "compiled_root_pose_matches_scene_world_START": True,
        "robot_collision_geom_count": len(robot_geoms),
        "scene_collision_geom_count": len(scene_geoms),
        "robot_collision_footprint_radial_upper_bound_m": footprint_bound,
        "cross_track_guard_inflation_m": cross_track_inflation,
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
) -> dict[str, Any]:
    """Copy the source scene inputs and derive path-only container wrappers."""
    evidence = Path(evidence_directory)
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

    scene_root = ET.parse(bundle / "scene_workspace.xml").getroot()
    source_scene_root = ET.parse(bundle / "scene_workspace.xml").getroot()
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
    # The runtime wrapper is allowed to change only the three resolved file
    # paths. All model/geom/joint transforms and collision attributes remain
    # byte-for-byte equivalent at the parsed XML element level.
    source_models = {item.get("name"): item.get("file") for item in source_scene_root.findall("./asset/model")}
    runtime_models = {item.get("name"): item.get("file") for item in scene_root.findall("./asset/model")}
    if set(source_models) != set(model_paths) or runtime_models != model_paths:
        raise Local01SceneRouteError("Local01 runtime scene wrapper did not resolve exactly the three scene model files")
    for item in scene_root.findall("./asset/model"):
        item.set("file", source_models[item.get("name")])
    if ET.tostring(scene_root) != ET.tostring(source_scene_root):
        raise Local01SceneRouteError("Local01 runtime scene wrapper changed scene content beyond model file paths")
    for item in scene_root.findall("./asset/model"):
        item.set("file", model_paths[item.get("name")])
    scene_runtime = bundle / "scene_workspace_runtime.xml"
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
    simulator_config = bundle / "simulator_default_local01_scene.yaml"
    with simulator_config.open("x", encoding="utf-8") as stream:
        stream.write(runtime_config_text)
    if runtime_config_text.replace(
        "simulation_model: /workspace/evidence/local01_scene/scene_workspace_runtime.xml",
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

    records = {
        "schema": "robotsim.issue80.local01-scene-runtime-bundle.v1",
        "classification": "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER",
        "source_local01_commit": source["source_commit"],
        "source_files_sha256": copied,
        "runtime_files_sha256": {
            "scene_workspace_runtime.xml": _sha256(scene_runtime),
            "mug_portable_runtime.xml": _sha256(mug_runtime),
            "simulator_default_local01_scene.yaml": _sha256(simulator_config),
        },
        "runtime_x2_sha256": _sha256(
            runtime_x2_path
        ),
        "simulator_config_only_change": {
            "field": "simulation_model",
            "from": old_model_line.split(": ", 1)[1],
            "to": "/workspace/evidence/local01_scene/scene_workspace_runtime.xml",
        },
        "scene_wrapper_change_classification": "MODEL_FILE_PATHS_ONLY",
        "mug_wrapper_change_classification": "COMPILER_ASSET_SEARCH_PATHS_ONLY",
        "reference_scene_mount_target": "/workspace/raicom_scene_source",
        "runtime_simulator_config_path": "/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml",
        "runtime_scene_path": "/workspace/evidence/local01_scene/scene_workspace_runtime.xml",
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
