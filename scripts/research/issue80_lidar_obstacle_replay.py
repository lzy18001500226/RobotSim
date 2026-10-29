#!/usr/bin/env python3
"""Replay raw RAICOM LiDAR frames through the offline obstacle layer.

No ROS graph, MuJoCo stepping, MC service, velocity publisher, or command
source is created.  The tool is deliberately useful for two cases only:
checking that a known-static capture does not fabricate a persistent obstacle,
and preserving the exact decision inputs before a separately authorized live
obstacle episode.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from issue80_lidar_map_localizer import best_candidate, load_database, raw_ranges
from issue80_lidar_obstacle_layer import (
    GridSpec,
    NovelObstacleTracker,
    Pose2D,
    RAICOM_CHEST_RAYCASTER,
    novel_returns_from_ranges,
)
from issue80_lidar_raw_bridge import RAW_SCHEMA


# Match the accepted static-map localizer contract.  A finite nearest-signature
# score alone is not a valid pose correspondence and must never drive obstacle
# extraction or a future collision decision.
MAX_STATIC_SIGNATURE_SCORE_M = 0.16
MIN_STATIC_SIGNATURE_MATCHES = 80


def _optional_ranges(values: np.ndarray) -> tuple[float | None, ...]:
    return tuple(float(value) if math.isfinite(float(value)) else None for value in values)


def grid_from_candidates(
    candidates: np.ndarray,
    *,
    resolution_m: float,
    sensor_range_margin_m: float = RAICOM_CHEST_RAYCASTER.maximum_range_m,
) -> GridSpec:
    if candidates.ndim != 2 or candidates.shape[1] != 3 or not len(candidates):
        raise ValueError("static LiDAR database has no pose candidates")
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("grid resolution must be finite and positive")
    if not math.isfinite(sensor_range_margin_m) or sensor_range_margin_m <= 0.0:
        raise ValueError("sensor range margin must be finite and positive")
    minimum = np.min(candidates[:, :2], axis=0)
    maximum = np.max(candidates[:, :2], axis=0)
    # Candidate poses are not obstacle coordinates: a valid new return can be
    # up to the configured sensor range away from its matched pose.  Keep the
    # complete observed horizon representable rather than silently discarding
    # a chair at the edge of the static-signature lattice.
    origin_x = math.floor((float(minimum[0]) - sensor_range_margin_m) / resolution_m - 1.0) * resolution_m
    origin_y = math.floor((float(minimum[1]) - sensor_range_margin_m) / resolution_m - 1.0) * resolution_m
    width = math.ceil((float(maximum[0]) + sensor_range_margin_m - origin_x) / resolution_m) + 2
    height = math.ceil((float(maximum[1]) + sensor_range_margin_m - origin_y) / resolution_m) + 2
    return GridSpec(origin_x, origin_y, resolution_m, width, height)


def replay(
    raw_trace: Path,
    database: Path,
    *,
    novelty_margin_m: float,
    confirmation_frames: int,
    grid_resolution_m: float,
) -> dict[str, Any]:
    if raw_trace.is_symlink() or not raw_trace.is_file():
        raise ValueError("raw trace must be a regular file")
    candidates, expected_ranges = load_database(database)
    if expected_ranges.shape[1] != RAICOM_CHEST_RAYCASTER.ray_count:
        raise ValueError("static database ray count differs from the provenance-pinned RayCaster profile")
    grid = grid_from_candidates(candidates, resolution_m=grid_resolution_m)
    tracker = NovelObstacleTracker(grid, confirmation_frames=confirmation_frames)
    frames_total = 0
    candidate_qualified = 0
    candidate_unqualified = 0
    novel_returns_total = 0
    confirmed: set[tuple[int, int]] = set()
    maximum_score = 0.0
    previous_sequence: int | None = None
    lines = raw_trace.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"raw trace line {line_number} is not JSON") from exc
        if not isinstance(raw, dict) or raw.get("schema") != RAW_SCHEMA:
            raise ValueError(f"raw trace line {line_number} has an unexpected schema")
        sequence = raw.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError(f"raw trace line {line_number} has invalid sequence")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError(f"raw trace sequence is not strictly increasing at line {line_number}")
        previous_sequence = sequence
        xyz = raw.get("xyz")
        if not isinstance(xyz, list):
            raise ValueError(f"raw trace line {line_number} has no xyz array")
        observed = raw_ranges(raw, expected_ranges.shape[1])
        index, score, matches = best_candidate(candidates, expected_ranges, observed)
        frames_total += 1
        if not (math.isfinite(score) and score <= MAX_STATIC_SIGNATURE_SCORE_M and matches >= MIN_STATIC_SIGNATURE_MATCHES):
            candidate_unqualified += 1
            continue
        candidate_qualified += 1
        maximum_score = max(maximum_score, score)
        pose = Pose2D(*[float(value) for value in candidates[index]])
        returns = novel_returns_from_ranges(
            _optional_ranges(observed), _optional_ranges(expected_ranges[index]),
            robot_pose=pose, novelty_margin_m=novelty_margin_m,
        )
        novel_returns_total += len(returns)
        cells = tracker.update(sequence, returns)
        confirmed.update((cell.x, cell.y) for cell in cells)
    if frames_total == 0:
        raise ValueError("raw trace contains no frames")
    if candidate_unqualified:
        status = "BLOCKED_STATIC_SIGNATURE_UNQUALIFIED_FRAMES"
    elif not confirmed:
        status = "PASS_STATIC_REPLAY_NO_CONFIRMED_NOVEL_OBSTACLE"
    else:
        status = "PASS_STATIC_REPLAY_CONFIRMED_NOVEL_OBSTACLES"
    return {
        "schema": "robotsim.issue80.lidar-obstacle-replay.v1",
        "classification": "SIMULATION_ONLY_OFFLINE_LIDAR_OBSTACLE_REPLAY",
        "status": status,
        "input": {
            "raw_trace": str(raw_trace),
            "static_database": str(database),
            "ray_count": RAICOM_CHEST_RAYCASTER.ray_count,
            "novelty_margin_m": novelty_margin_m,
            "confirmation_frames": confirmation_frames,
            "grid": {
                "origin_x_m": grid.origin_x_m,
                "origin_y_m": grid.origin_y_m,
                "resolution_m": grid.resolution_m,
                "width": grid.width,
                "height": grid.height,
            },
        },
        "frames_total": frames_total,
        "candidate_qualified_frames": candidate_qualified,
        "candidate_unqualified_frames": candidate_unqualified,
        "static_signature_contract": {
            "maximum_score_m": MAX_STATIC_SIGNATURE_SCORE_M,
            "minimum_matches": MIN_STATIC_SIGNATURE_MATCHES,
        },
        "maximum_static_signature_score_m": maximum_score,
        "novel_returns_total": novel_returns_total,
        "confirmed_obstacle_cells": [{"x": x, "y": y} for x, y in sorted(confirmed)],
        "read_only": True,
        "mj_step_calls": 0,
        "velocity_publisher_created": False,
        "mc_action_or_source_client_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trace", type=Path, required=True)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--novelty-margin-m", type=float, default=0.20)
    parser.add_argument("--confirmation-frames", type=int, default=2)
    parser.add_argument("--grid-resolution-m", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be a new regular path")
    if not math.isfinite(args.novelty_margin_m) or args.novelty_margin_m <= 0.0:
        parser.error("--novelty-margin-m must be finite and positive")
    if args.confirmation_frames < 1:
        parser.error("--confirmation-frames must be at least one")
    try:
        result = replay(
            args.raw_trace, args.static_database,
            novelty_margin_m=args.novelty_margin_m,
            confirmation_frames=args.confirmation_frames,
            grid_resolution_m=args.grid_resolution_m,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
