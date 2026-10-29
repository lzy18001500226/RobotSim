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
from issue80_lidar_obstacle_layer import GridCell
from issue80_lidar_obstacle_pipeline import MapFrameObstaclePipeline, ObstaclePipelineConfig
from issue80_lidar_obstacle_replay import grid_from_candidates


def point_bytes(points: tuple) -> bytes:
    return b"".join(struct.pack("<fff", point.x_m, point.y_m, point.z_m) for point in points)


def replay(
    *, background_capture: Path, observed_capture: Path, static_database: Path,
    maximum_match_distance_m: float, minimum_cluster_cells: int,
) -> dict:
    background = load_map_background_capture(background_capture)
    observed = load_map_background_capture(observed_capture)
    candidates, _ = load_database(static_database)
    grid = grid_from_candidates(candidates, resolution_m=0.10)
    pipeline = MapFrameObstaclePipeline(ObstaclePipelineConfig(
        grid=grid,
        static_surface=frozenset(),
        static_route_blocked=frozenset(),
        start=grid.world_to_cell(-1.5, -1.5),
        goal=grid.world_to_cell(0.9, -1.5),
        static_match_distance_m=maximum_match_distance_m,
        obstacle_inflation_m=0.46565748847422734,
        confirmation_frames=2,
        minimum_confirmed_cluster_cells=minimum_cluster_cells,
        map_background_points=background,
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
        },
        "first_decision": first.status.value,
        "second_decision": second.status.value,
        "novel_point_count": second.novel_point_count,
        "confirmed_cells": [{"x": cell.x, "y": cell.y} for cell in second.confirmed_cells],
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
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be new")
    if args.maximum_match_distance_m <= 0.0 or args.minimum_cluster_cells < 1:
        parser.error("invalid replay thresholds")
    try:
        result = replay(
            background_capture=args.background_capture,
            observed_capture=args.observed_capture,
            static_database=args.static_database,
            maximum_match_distance_m=args.maximum_match_distance_m,
            minimum_cluster_cells=args.minimum_cluster_cells,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
