#!/usr/bin/env python3
"""Audit a LiDAR-observed detour against the complete generated RAICOM scene.

This is a source-preserving, ``mj_forward``-only gate between the read-only
LiDAR planner and the existing gait executor.  It deliberately consumes the
saved observed path rather than obstacle-generator truth, while compiling the
actual runtime scene which contains the generated unknown obstacles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco

from issue80_local01_scene_route import _resolve_runtime_x2_path, assert_native_robot_equivalence
from issue80_raicom_chair_detour_static import (
    ChairDetourError,
    heading_points,
    robot_body_ids,
    static_clearance_audit,
)


class ObservedDetourError(ValueError):
    """A retained observation or source scene cannot safely be audited."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_route(path: Path) -> tuple[tuple[float, float], ...]:
    if path.is_symlink() or not path.is_file():
        raise ObservedDetourError("observed LiDAR replay must be a regular file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "robotsim.issue80.lidar-map-background-replay.v1":
        raise ObservedDetourError("observed replay has the wrong schema")
    if raw.get("status") != "PASS_CAPTURE_REPLAYED" or raw.get("second_decision") != "DETOUR_READY_STATIC_ONLY":
        raise ObservedDetourError("observed replay did not qualify a detour")
    points = raw.get("detour_waypoints")
    if not isinstance(points, list) or len(points) < 3:
        raise ObservedDetourError("observed detour needs at least start, bend, and goal")
    result: list[tuple[float, float]] = []
    for index, item in enumerate(points):
        if not isinstance(item, dict):
            raise ObservedDetourError(f"detour waypoint {index} is malformed")
        try:
            point = (float(item["x_m"]), float(item["y_m"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservedDetourError(f"detour waypoint {index} lacks finite planar coordinates") from exc
        if not all(math.isfinite(value) for value in point):
            raise ObservedDetourError(f"detour waypoint {index} is non-finite")
        result.append(point)
    return tuple(result)


def collision_sets(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ObservedDetourError("compiled scene lacks native pelvis")
    robot_bodies = robot_body_ids(model)
    robot: list[int] = []
    environment: list[int] = []
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        if int(model.geom_bodyid[geom_id]) in robot_bodies:
            robot.append(geom_id)
            continue
        # Feet legitimately touch this ground geom.  Every other collidable
        # scene geom, including generated unknown obstacles, remains in the
        # full-body route clearance set.
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name != "floor":
            environment.append(geom_id)
    if not robot or not environment:
        raise ObservedDetourError("scene lacks robot or non-floor environment collision geoms")
    return robot, environment


def audit(scene: Path, replay: Path, *, final_yaw_rad: float) -> dict[str, Any]:
    if scene.is_symlink() or not scene.is_file():
        raise ObservedDetourError("runtime scene must be a regular file")
    if mujoco.__version__ != "3.3.6":
        raise ObservedDetourError(f"requires MuJoCo 3.3.6, got {mujoco.__version__}")
    points = observed_route(replay)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    native_scene = _resolve_runtime_x2_path(Path(__file__).resolve().parents[2]).with_name("scene.xml")
    native = mujoco.MjModel.from_xml_path(str(native_scene))
    equivalence = assert_native_robot_equivalence(native, model)
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint < 0:
        raise ObservedDetourError("compiled scene lacks native floating_base_joint")
    robot, environment = collision_sets(model)
    clearance = static_clearance_audit(
        model,
        data,
        poses=heading_points(points, final_yaw_rad),
        robot_geoms=robot,
        environment_geoms=environment,
        root_joint=root_joint,
    )
    return {
        "schema": "robotsim.issue80.raicom-observed-detour-static.v1",
        "classification": "SIMULATION_ONLY_LIDAR_OBSERVED_DETOUR_FULLBODY_STATIC_AUDIT",
        "status": clearance["status"],
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "observed_replay": {"path": str(replay), "sha256": sha256(replay)},
        "robot_equivalence": equivalence,
        "planner_input": {
            "route_source": "OBSERVED_LIDAR_BACKGROUND_REPLAY",
            "map_waypoints": [
                {"x_m": x, "y_m": y, "yaw_rad": yaw}
                for x, y, yaw in heading_points(points, final_yaw_rad)
            ],
        },
        "static_fullbody_clearance": clearance,
        "mj_step_calls": 0,
        "velocity_publisher_created": False,
        "mc_action_or_source_client_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-scene", type=Path, required=True)
    parser.add_argument("--observed-detour-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-yaw-rad", type=float, default=0.0)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=True)
    if output.is_symlink() or any(item.name != ".robotsim-output-reservation.json" for item in output.iterdir()):
        parser.error("output directory must contain only its governed reservation")
    if not math.isfinite(args.final_yaw_rad):
        parser.error("final yaw must be finite")
    try:
        result = audit(
            args.runtime_scene.resolve(strict=True),
            args.observed_detour_replay.resolve(strict=True),
            final_yaw_rad=args.final_yaw_rad,
        )
    except (OSError, ValueError, ChairDetourError, mujoco.FatalError) as exc:
        parser.error(f"observed-detour static audit failed: {type(exc).__name__}: {exc}")
    (output / "observed_detour_static_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
