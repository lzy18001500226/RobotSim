#!/usr/bin/env python3
"""Convert one confirmed LiDAR detour into the existing RouteSpec JSON format.

This is only an offline adapter.  It accepts no obstacle-generator positions:
the path must come from a measured map-background replay and must have already
passed an independent full-body static audit of the matching runtime scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from issue80_waypoint_route import (
    SCHEMA_VERSION,
    RouteSpec,
    _corridor_polygon,
)
from issue80_m1_mode_logic import VENDOR_EXAMPLE_FORWARD_M_S


class LidarRouteConfigError(ValueError):
    """The measured detour or its static audit is not safe to adapt."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_initial_heading(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    """Insert a source-frame forward-tangent entry into a LiDAR A* polyline.

    Grid simplification can turn a collision-free path into an initial corner
    of roughly one radian.  FoundationWalk has a bounded yaw-rate channel, so
    the route manager must not ask it to instantaneously follow that corner.
    This deterministic Bézier sampling uses only the observed path's first
    two tangents and the known Reset heading (+map X); it neither uses
    obstacle-generator geometry nor modifies the A* goal.
    """
    if len(points) < 3:
        raise LidarRouteConfigError("LiDAR path needs at least three points for entry smoothing")
    start, first, second = points[:3]
    next_dx, next_dy = second[0] - first[0], second[1] - first[1]
    next_norm = math.hypot(next_dx, next_dy)
    if next_norm <= 0.05:
        raise LidarRouteConfigError("LiDAR path first outgoing segment is degenerate")
    # The source map's fixed start heading is +X.  The endpoint tangent uses
    # the following A* segment to preserve continuity at the first corner.
    entry = (start[0] + 0.35, start[1])
    control_one = (entry[0] + 0.90, entry[1])
    control_two = (first[0] - 0.90 * next_dx / next_norm, first[1] - 0.90 * next_dy / next_norm)
    samples: list[tuple[float, float]] = [start, entry]
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        complement = 1.0 - fraction
        samples.append((
            complement ** 3 * entry[0] + 3.0 * complement ** 2 * fraction * control_one[0]
            + 3.0 * complement * fraction ** 2 * control_two[0] + fraction ** 3 * first[0],
            complement ** 3 * entry[1] + 3.0 * complement ** 2 * fraction * control_one[1]
            + 3.0 * complement * fraction ** 2 * control_two[1] + fraction ** 3 * first[1],
        ))
    return tuple(samples + list(points[2:]))


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LidarRouteConfigError("route input must be a regular file")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LidarRouteConfigError("route input JSON must be an object")
    return raw


def build(*, replay_path: Path, static_audit_path: Path, occupancy_path: Path) -> dict[str, Any]:
    replay = _read(replay_path)
    audit = _read(static_audit_path)
    occupancy = _read(occupancy_path)
    if (
        replay.get("schema") != "robotsim.issue80.lidar-map-background-replay.v1"
        or replay.get("status") != "PASS_CAPTURE_REPLAYED"
        or replay.get("second_decision") != "DETOUR_READY_STATIC_ONLY"
    ):
        raise LidarRouteConfigError("replay is not a confirmed LiDAR detour")
    if (
        audit.get("schema") != "robotsim.issue80.raicom-observed-detour-static.v1"
        or audit.get("status") != "PASS_STATIC_FULLBODY_DETOUR_CLEARANCE"
        or audit.get("observed_replay", {}).get("sha256") != sha256(replay_path)
    ):
        raise LidarRouteConfigError("matching full-body static detour audit did not pass")
    if (
        occupancy.get("schema") != "robotsim.issue80.raicom-static-occupancy.v1"
        or occupancy.get("status") != "PASS_STATIC_OCCUPANCY_BUILT"
    ):
        raise LidarRouteConfigError("known static occupancy did not pass")
    raw_points = replay.get("detour_waypoints")
    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise LidarRouteConfigError("confirmed LiDAR detour must have a bend")
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            raise LidarRouteConfigError("LiDAR detour waypoint is malformed")
        try:
            point = (float(item["x_m"]), float(item["y_m"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LidarRouteConfigError("LiDAR detour waypoint lacks x/y") from exc
        if not all(math.isfinite(value) for value in point):
            raise LidarRouteConfigError("LiDAR detour waypoint is non-finite")
        points.append(point)
    audit_points = audit.get("planner_input", {}).get("map_waypoints")
    if not isinstance(audit_points, list) or len(audit_points) != len(points):
        raise LidarRouteConfigError("static audit does not contain the same number of route waypoints")
    for observed, audited in zip(points, audit_points):
        if not isinstance(audited, dict) or math.hypot(
            observed[0] - float(audited.get("x_m", math.inf)),
            observed[1] - float(audited.get("y_m", math.inf)),
        ) > 1e-9:
            raise LidarRouteConfigError("static audit route differs from the observed LiDAR route")
    measured_envelope = float(occupancy.get("measured_fullbody_radial_envelope_m"))
    # The occupancy artifact's radial value already includes the static
    # audit's 50 mm tracking envelope.  RouteSpec instead stores the bare
    # collision footprint plus its own boundary margin.  Keep exactly the
    # same total lateral envelope (bare radius + 50 mm route margin + 50 mm
    # permitted tracking) rather than double-counting it and rejecting the
    # centreline before a command can be proposed.
    footprint = measured_envelope - 0.05
    clearance = float(audit["static_fullbody_clearance"]["minimum_tracking_inflated_aabb_clearance_m"])
    if not all(math.isfinite(value) and value > 0.0 for value in (footprint, clearance)):
        raise LidarRouteConfigError("static audit has no positive footprint/clearance")
    # The static audit samples +/- 0.075 rad and carries a 50 mm tracking
    # envelope.  Preserve that envelope in the route corridor; never expand
    # a LiDAR route into unexplored free space at configuration time.
    boundary_margin = 0.05
    permitted_tracking = 0.05
    half_width = footprint + boundary_margin + permitted_tracking
    steps = []
    for index, (start, target) in enumerate(zip(points, points[1:]), start=1):
        final = index == len(points) - 1
        steps.append({
            "id": f"LIDAR_DETOUR_{index:02d}",
            "profile": "TRANSIT",
            "pose": {"x_m": target[0], "y_m": target[1], "yaw_rad": None},
            "acceptance": {
                "position_tolerance_m": 0.15,
                "yaw_policy": "IGNORE",
                "yaw_tolerance_rad": None,
                "max_position_std_m": 0.075,
                "max_yaw_std_rad": 0.25,
            },
            "stop_behavior": "VERIFY_PHYSICAL_STOP" if final else "ROLLING",
            "segment_timeout_s": 60.0,
            "recovery_limit": 0,
            "corridor": {
                # The route manager measures clearance to every rectangle
                # edge, including the start/end caps.  The LiDAR localizer's
                # accepted 75 mm initial correspondence residual plus the
                # 0.466 m footprint/boundary margin needs a longitudinal cap
                # extension larger than the 0.15 m arrival tolerance.
                "polygon": [list(item) for item in _corridor_polygon(start, target, half_width_m=half_width, extension_m=0.75)],
                "minimum_obstacle_clearance_m": clearance,
            },
            "command_bounds": {
                "target_forward_m_s": VENDOR_EXAMPLE_FORWARD_M_S,
                "max_forward_m_s": VENDOR_EXAMPLE_FORWARD_M_S,
                "max_angular_rad_s": 0.075,
                "max_command_window_s": 5.0,
            },
        })
    raw = {
        "schema_version": SCHEMA_VERSION,
        "route_id": "raicom-lidar-observed-static-detour",
        "frame_id": "map",
        "workspace_clearance_verified": True,
        "vendor_gait_profile": True,
        "turn_capabilities": {
            "zero_forward_turn_qualified": False,
            "bounded_moving_turn_qualified": False,
            "evidence_classification": "NO_SEPARATE_CORNER_DIRECTIVE; fresh bearing steering remains bounded by the existing translational route envelope",
            "evidence_attempt_id": "LIDAR_BACKGROUND_REPLAY_AND_STATIC_FULLBODY_AUDIT",
        },
        "start_anchor": {
            "x_m": points[0][0], "y_m": points[0][1], "yaw_rad": 0.0,
            "position_tolerance_m": 0.15, "yaw_tolerance_rad": 0.25,
        },
        "localization": {
            "max_pose_age_s": 0.40, "max_transform_age_s": 0.50,
            "max_position_std_m": 0.075, "max_yaw_std_rad": 0.25,
        },
        "safety": {
            "robot_footprint_radius_m": footprint,
            "boundary_margin_m": boundary_margin,
            "max_observation_age_s": 0.50,
        },
        "steps": steps,
        "provenance": {
            "classification": "SIMULATION_ONLY_LIDAR_LOCALIZED_KNOWN_STATIC_MAP_UNKNOWN_RUNTIME_OBSTACLE_ROUTE",
            "measured_replay_sha256": sha256(replay_path),
            "static_audit_sha256": sha256(static_audit_path),
            "static_occupancy_sha256": sha256(occupancy_path),
            "unknown_obstacles_are_not_encoded_as_generator_truth": True,
            "execution_requires_fresh_live_lidar_confirmation_and_existing_MC_source_mode_stop_gates": True,
            "measured_static_fullbody_envelope_m": measured_envelope,
            "route_corridor_half_width_m": half_width,
            "permitted_tracking_m": permitted_tracking,
            "route_forward_m_s": VENDOR_EXAMPLE_FORWARD_M_S,
            "foundationwalk_config": "UNMODIFIED_PINNED_VENDOR_CONFIG",
            "raw_lidar_a_star_waypoint_count": len(replay.get("raw_lidar_a_star_waypoints", points)),
            "smoothed_route_waypoint_count": len(points),
            "initial_heading_smoothing": "deterministic_bezier_from_observed_ASTAR_first_corner_and_known_+mapX_Reset_heading",
        },
    }
    RouteSpec.from_mapping(raw)
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--static-occupancy", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("result must be new")
    try:
        route = build(
            replay_path=args.replay.resolve(strict=True),
            static_audit_path=args.static_audit.resolve(strict=True),
            occupancy_path=args.static_occupancy.resolve(strict=True),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"LiDAR route config rejected: {type(exc).__name__}: {exc}")
    args.result.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_ROUTE_CONFIG_BUILT", "route_id": route["route_id"], "step_count": len(route["steps"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
