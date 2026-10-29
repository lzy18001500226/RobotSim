#!/usr/bin/env python3
"""Build a known-static 2-D route occupancy layer from a pinned RAICOM scene.

The input scene intentionally excludes generated runtime obstacles.  This is
the map's known wall/table/chair layer, not a runtime truth feed.  Unknown
objects must still enter through the separate LiDAR novelty pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from issue80_lidar_map_localizer import load_database
from issue80_lidar_obstacle_layer import GridCell
from issue80_lidar_obstacle_replay import grid_from_candidates
from issue80_local01_scene_route import _world_geom_aabb
from issue80_raicom_chair_detour_static import collision_geom_ids, footprint_radius, robot_body_ids


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(scene: Path, static_database: Path, *, planner_envelope_m: float) -> dict:
    if scene.is_symlink() or static_database.is_symlink() or not scene.is_file() or not static_database.is_file():
        raise ValueError("scene and static database must be regular files")
    if mujoco.__version__ != "3.3.6":
        raise ValueError(f"requires MuJoCo 3.3.6, got {mujoco.__version__}")
    candidates, _ = load_database(static_database)
    grid = grid_from_candidates(candidates, resolution_m=0.10)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ValueError("compiled scene lacks native pelvis")
    robot, environment = collision_geom_ids(model, robot_body_ids(model))
    measured_radial_robot_bound = footprint_radius(model, data, robot, pelvis) + 0.05
    # A radial full-body bound would reject the already physically validated
    # TABLE_APPROACH cell because the robot's arms make that bound overly
    # conservative for all headings.  This is deliberately only a known-map
    # planner exclusion, not a collision acceptance condition: every selected
    # detour remains subject to the independent full-body static audit with
    # the measured radial bound before runtime execution.
    envelope = planner_envelope_m
    boxes = {geom: _world_geom_aabb(model, data, geom, np) for geom in environment}
    blocked: set[GridCell] = set()
    for ix in range(grid.width):
        for iy in range(grid.height):
            cell = GridCell(ix, iy)
            x, y = grid.cell_center(cell)
            if any(
                lower[0] - envelope <= x <= upper[0] + envelope
                and lower[1] - envelope <= y <= upper[1] + envelope
                for lower, upper in boxes.values()
            ):
                blocked.add(cell)
    start = grid.world_to_cell(-1.5, -1.5)
    goal = grid.world_to_cell(0.9, -1.5)
    return {
        "schema": "robotsim.issue80.raicom-static-occupancy.v1",
        "classification": "SIMULATION_ONLY_PREBUILT_KNOWN_STATIC_OCCUPANCY",
        "status": "PASS_STATIC_OCCUPANCY_BUILT" if start not in blocked and goal not in blocked else "BLOCKED_ENDPOINT_OCCUPIED",
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "static_database": {"path": str(static_database), "sha256": sha256(static_database)},
        "grid": {
            "origin_x_m": grid.origin_x_m, "origin_y_m": grid.origin_y_m,
            "resolution_m": grid.resolution_m, "width": grid.width, "height": grid.height,
        },
        "planner_envelope_m": envelope,
        "measured_fullbody_radial_envelope_m": measured_radial_robot_bound,
        "planner_limitations": [
            "The 2-D preplanner uses its explicit source-map exclusion envelope.",
            "Full-body static clearance with the measured radial envelope remains mandatory before a route may execute.",
        ],
        "environment_geom_ids": sorted(boxes),
        "blocked_cells": [{"x": item.x, "y": item.y} for item in sorted(blocked)],
        "start_cell": {"x": start.x, "y": start.y},
        "goal_cell": {"x": goal.x, "y": goal.y},
        "mj_step_calls": 0,
        "velocity_publisher_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--planner-envelope-m", type=float, default=0.20)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("result must be new")
    if not 0.0 < args.planner_envelope_m <= 0.30:
        parser.error("planner envelope must be in (0, 0.30]")
    try:
        result = build(
            args.scene.resolve(strict=True), args.static_database.resolve(strict=True),
            planner_envelope_m=args.planner_envelope_m,
        )
    except (OSError, ValueError, mujoco.FatalError) as exc:
        parser.error(f"static occupancy build failed: {type(exc).__name__}: {exc}")
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS_STATIC_OCCUPANCY_BUILT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
