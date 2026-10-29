#!/usr/bin/env python3
"""Offline distance audit of one captured map-frame LiDAR point cloud."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from issue80_lidar_map_obstacle_field import (
    MapPoint,
    static_surface_cells_from_database,
    static_surface_distances_m,
)
from issue80_lidar_map_localizer import load_database
from issue80_lidar_static_surface_audit import calibration_from_localization_trace
from issue80_lidar_obstacle_replay import grid_from_candidates


def replay(capture: Path, database: Path, localization_trace: Path, *, grid_resolution_m: float) -> dict[str, Any]:
    payload = json.loads(capture.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_NONEMPTY_POINTCLOUD":
        raise ValueError("captured PointCloud2 did not pass nonempty/frame validation")
    message = payload.get("last_message")
    if not isinstance(message, dict) or message.get("frame_id") != "map":
        raise ValueError("captured PointCloud2 is not a map-frame message")
    raw_points = message.get("all_finite_xyz")
    if not isinstance(raw_points, list) or message.get("all_finite_xyz_retention_blocked") is not False:
        raise ValueError("captured PointCloud2 lacks a complete retained finite point array")
    points = tuple(MapPoint(*(float(value) for value in item)) for item in raw_points if isinstance(item, list) and len(item) == 3)
    if not points:
        raise ValueError("captured PointCloud2 has no finite XYZ points")
    candidates, ranges = load_database(database)
    transform = calibration_from_localization_trace(localization_trace)
    grid = grid_from_candidates(candidates, resolution_m=grid_resolution_m)
    surface = static_surface_cells_from_database(candidates, ranges, static_to_map=transform, grid=grid)
    distances = np.asarray(static_surface_distances_m(points, static_surface=surface, grid=grid), dtype=float)
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        raise ValueError("all map-frame points lie outside the static surface field")
    thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
    return {
        "schema": "robotsim.issue80.lidar-map-frame-static-replay.v1",
        "classification": "SIMULATION_ONLY_OFFLINE_MAP_FRAME_SURFACE_DISTANCE_AUDIT",
        "status": "PASS_CAPTURED_MAP_FRAME_POINTS_ANALYZED",
        "inputs": {
            "capture": str(capture), "static_database": str(database), "localization_trace": str(localization_trace),
            "frame_id": "map", "point_count": len(points),
            "static_signature_to_map": {"x_m": transform.x_m, "y_m": transform.y_m, "yaw_rad": transform.yaw_rad},
        },
        "static_surface_cell_count": len(surface),
        "grid_resolution_m": grid.resolution_m,
        "distance_m": {
            "minimum": float(np.min(finite)), "median": float(np.median(finite)),
            "p90": float(np.quantile(finite, 0.90)), "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99)), "maximum": float(np.max(finite)),
            "outside_static_field_point_count": int(np.count_nonzero(~np.isfinite(distances))),
            "matching_fraction_by_threshold_m": {str(value): float(np.mean(distances <= value)) for value in thresholds},
        },
        "threshold_decision": "DESCRIPTIVE_ONLY_NOT_YET_AN_OBSTACLE_ACCEPTANCE_THRESHOLD",
        "mj_step_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--localization-trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--grid-resolution-m", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be new")
    try:
        result = replay(args.capture, args.static_database, args.localization_trace, grid_resolution_m=args.grid_resolution_m)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
