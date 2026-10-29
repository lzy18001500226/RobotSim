#!/usr/bin/env python3
"""Build and audit a map-frame static LiDAR surface field without physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from issue80_lidar_map_localizer import load_database
from issue80_lidar_map_obstacle_field import static_surface_cells_from_database
from issue80_lidar_obstacle_layer import Pose2D
from issue80_lidar_obstacle_replay import grid_from_candidates


def calibration_from_localization_trace(trace: Path) -> Pose2D:
    if trace.is_symlink() or not trace.is_file():
        raise ValueError("localization trace must be a regular file")
    for line_number, line in enumerate(trace.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"localization trace line {line_number} is not JSON") from exc
        calibration = item.get("calibration") if isinstance(item, dict) else None
        if isinstance(calibration, list) and len(calibration) == 3:
            try:
                return Pose2D(*(float(value) for value in calibration))
            except (TypeError, ValueError) as exc:
                raise ValueError("localization calibration is not numeric") from exc
    raise ValueError("localization trace contains no static-signature-to-map calibration")


def audit(database: Path, localization_trace: Path, *, grid_resolution_m: float) -> dict[str, Any]:
    candidates, ranges = load_database(database)
    static_to_map = calibration_from_localization_trace(localization_trace)
    grid = grid_from_candidates(candidates, resolution_m=grid_resolution_m)
    surface = static_surface_cells_from_database(
        candidates, ranges, static_to_map=static_to_map, grid=grid,
    )
    if not surface:
        raise ValueError("static database produced no map-frame surface cells")
    return {
        "schema": "robotsim.issue80.lidar-static-surface-audit.v1",
        "classification": "SIMULATION_ONLY_OFFLINE_STATIC_SURFACE_FIELD",
        "status": "PASS_STATIC_SURFACE_FIELD_BUILT",
        "inputs": {
            "static_database": str(database),
            "localization_trace": str(localization_trace),
            "static_signature_to_map": {
                "x_m": static_to_map.x_m,
                "y_m": static_to_map.y_m,
                "yaw_rad": static_to_map.yaw_rad,
            },
        },
        "database": {"candidate_count": int(len(candidates)), "ray_count": int(ranges.shape[1])},
        "grid": {
            "origin_x_m": grid.origin_x_m,
            "origin_y_m": grid.origin_y_m,
            "resolution_m": grid.resolution_m,
            "width": grid.width,
            "height": grid.height,
        },
        "static_surface_cell_count": len(surface),
        "mj_step_calls": 0,
        "pointcloud_map_frame_verified": False,
        "next_required_gate": "READ_ONLY_CAPTURE_OF_AIMA_SIM_LIDAR_POINTCLOUD_FRAME_AND_TF",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--localization-trace", type=Path, required=True)
    parser.add_argument("--grid-resolution-m", type=float, default=0.10)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be a new regular path")
    try:
        result = audit(args.static_database, args.localization_trace, grid_resolution_m=args.grid_resolution_m)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
