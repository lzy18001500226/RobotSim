#!/usr/bin/env python3
"""Replay one retained map-frame capture against a measured stable background.

This is a no-ROS, no-MuJoCo-step verifier for the same temporal/spatial
obstacle decision used by the read-only runtime diagnostic.  It never emits a
velocity or executes the returned detour.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from issue80_lidar_map_localizer import load_database
from issue80_lidar_map_obstacle_field import load_map_background_capture
from issue80_lidar_obstacle_layer import GridCell, GridSpec
from issue80_lidar_obstacle_pipeline import MapFrameObstaclePipeline, ObstaclePipelineConfig
from issue80_lidar_obstacle_replay import grid_from_candidates


def point_bytes(points: tuple) -> bytes:
    return b"".join(struct.pack("<fff", point.x_m, point.y_m, point.z_m) for point in points)


def load_static_route_blocked(path: Path | None, *, grid: GridSpec) -> frozenset[GridCell]:
    if path is None:
        return frozenset()
    if path.is_symlink() or not path.is_file():
        raise ValueError("static route occupancy must be a regular file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_grid = raw.get("grid") if isinstance(raw, dict) else None
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "robotsim.issue80.raicom-static-occupancy.v1"
        or not isinstance(expected_grid, dict)
        or float(expected_grid.get("origin_x_m")) != grid.origin_x_m
        or float(expected_grid.get("origin_y_m")) != grid.origin_y_m
        or float(expected_grid.get("resolution_m")) != grid.resolution_m
        or int(expected_grid.get("width")) != grid.width
        or int(expected_grid.get("height")) != grid.height
    ):
        raise ValueError("static route occupancy does not match the LiDAR static-map grid")
    cells: set[GridCell] = set()
    for item in raw.get("blocked_cells", []):
        if not isinstance(item, dict):
            raise ValueError("static route occupancy cell is malformed")
        cell = GridCell(int(item["x"]), int(item["y"]))
        if not grid.contains(cell):
            raise ValueError("static route occupancy contains an out-of-grid cell")
        cells.add(cell)
    return frozenset(cells)


def replay(
    *, background_capture: Path, observed_capture: Path, static_database: Path,
    maximum_match_distance_m: float, minimum_cluster_cells: int,
    minimum_obstacle_span_x_m: float = 0.0, minimum_obstacle_span_y_m: float = 0.0,
    static_route_blocked_path: Path | None = None,
) -> dict:
    background = load_map_background_capture(background_capture)
    observed = load_map_background_capture(observed_capture)
    candidates, _ = load_database(static_database)
    grid = grid_from_candidates(candidates, resolution_m=0.10)
    static_route_blocked = load_static_route_blocked(static_route_blocked_path, grid=grid)
    pipeline = MapFrameObstaclePipeline(ObstaclePipelineConfig(
        grid=grid,
        static_surface=frozenset(),
        static_route_blocked=static_route_blocked,
        start=grid.world_to_cell(-1.5, -1.5),
        goal=grid.world_to_cell(0.9, -1.5),
        static_match_distance_m=maximum_match_distance_m,
        obstacle_inflation_m=0.46565748847422734,
        confirmation_frames=2,
        minimum_confirmed_cluster_cells=minimum_cluster_cells,
        map_background_points=background,
        minimum_obstacle_span_x_m=minimum_obstacle_span_x_m,
        minimum_obstacle_span_y_m=minimum_obstacle_span_y_m,
    ))
    kwargs = {
        "frame_id": "map",
        "data": point_bytes(observed),
        "width": len(observed),
        "height": 1,
        "point_step": 12,
        "field_offsets": {"x": 0, "y": 4, "z": 8},
    }
    first = pipeline.observe(sequence=0, **kwargs)
    second = pipeline.observe(sequence=1, **kwargs)
    return {
        "schema": "robotsim.issue80.lidar-map-background-replay.v1",
        "classification": "SIMULATION_ONLY_OFFLINE_MEASURED_MAP_BACKGROUND_REPLAY",
        "status": "PASS_CAPTURE_REPLAYED",
        "inputs": {
            "background_capture": str(background_capture),
            "observed_capture": str(observed_capture),
            "static_database": str(static_database),
            "background_point_count": len(background),
            "observed_point_count": len(observed),
            "maximum_match_distance_m": maximum_match_distance_m,
            "minimum_cluster_cells": minimum_cluster_cells,
            "minimum_obstacle_span_m": {"x": minimum_obstacle_span_x_m, "y": minimum_obstacle_span_y_m},
            "static_route_blocked_source": None if static_route_blocked_path is None else str(static_route_blocked_path),
            "static_route_blocked_cell_count": len(static_route_blocked),
        },
        "first_decision": first.status.value,
        "second_decision": second.status.value,
        "novel_point_count": second.novel_point_count,
        "confirmed_cells": [{"x": cell.x, "y": cell.y} for cell in second.confirmed_cells],
        "obstacle_envelope_cells": [{"x": cell.x, "y": cell.y} for cell in second.obstacle_envelope_cells],
        "detour_waypoints": [{"x_m": x, "y_m": y} for x, y in second.detour_map_waypoints],
        "read_only": True,
        "mj_step_calls": 0,
        "velocity_publisher_created": False,
        "mc_action_or_source_client_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-capture", type=Path, required=True)
    parser.add_argument("--observed-capture", type=Path, required=True)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--maximum-match-distance-m", type=float, default=0.02)
    parser.add_argument("--minimum-cluster-cells", type=int, default=3)
    parser.add_argument("--minimum-obstacle-span-x-m", type=float, default=0.0)
    parser.add_argument("--minimum-obstacle-span-y-m", type=float, default=0.0)
    parser.add_argument("--static-route-blocked", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be new")
    if args.maximum_match_distance_m <= 0.0 or args.minimum_cluster_cells < 1 or args.minimum_obstacle_span_x_m < 0.0 or args.minimum_obstacle_span_y_m < 0.0:
        parser.error("invalid replay thresholds")
    try:
        result = replay(
            background_capture=args.background_capture,
            observed_capture=args.observed_capture,
            static_database=args.static_database,
            maximum_match_distance_m=args.maximum_match_distance_m,
            minimum_cluster_cells=args.minimum_cluster_cells,
            minimum_obstacle_span_x_m=args.minimum_obstacle_span_x_m,
            minimum_obstacle_span_y_m=args.minimum_obstacle_span_y_m,
            static_route_blocked_path=args.static_route_blocked,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
