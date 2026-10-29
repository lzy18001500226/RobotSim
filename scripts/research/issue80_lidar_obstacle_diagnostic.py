#!/usr/bin/env python3
"""Read-only runtime diagnostic for map-frame LiDAR obstacle decisions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from issue80_lidar_map_obstacle_field import load_map_background_capture, static_surface_cells_from_database
from issue80_lidar_map_localizer import load_database
from issue80_lidar_static_surface_audit import calibration_from_localization_trace
from issue80_lidar_obstacle_layer import GridSpec
from issue80_lidar_obstacle_layer import GridCell
from issue80_lidar_obstacle_pipeline import MapFrameObstaclePipeline, ObstaclePipelineConfig
from issue80_lidar_obstacle_replay import grid_from_candidates


DEFAULT_TOPIC = "/aima/sim/lidar/points"


class Diagnostic(Node):
    def __init__(self, pipeline: MapFrameObstaclePipeline, trace: Path, *, topic: str) -> None:
        super().__init__("issue80_lidar_obstacle_diagnostic", enable_rosout=False, start_parameter_services=False)
        parameter_events = getattr(self, "_parameter_event_publisher", None)
        if parameter_events is not None:
            self.destroy_publisher(parameter_events)
            self._parameter_event_publisher = None
        self.pipeline = pipeline
        self.sequence = 0
        self.decisions = []
        self.trace = trace.open("x", encoding="utf-8", buffering=1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.subscription_handles = [self.create_subscription(PointCloud2, topic, self.callback, qos)]

    def callback(self, message: PointCloud2) -> None:
        offsets = {str(field.name): int(field.offset) for field in message.fields}
        decision = self.pipeline.observe(
            sequence=self.sequence,
            frame_id=str(message.header.frame_id),
            data=bytes(message.data),
            width=int(message.width), height=int(message.height), point_step=int(message.point_step), field_offsets=offsets,
        )
        payload = {
            "sequence": self.sequence, "stamp": {"sec": int(message.header.stamp.sec), "nanosec": int(message.header.stamp.nanosec)},
            "status": decision.status.value, "observed_point_count": decision.observed_point_count,
            "novel_point_count": decision.novel_point_count,
            "confirmed_cells": [{"x": cell.x, "y": cell.y} for cell in decision.confirmed_cells],
            "obstacle_envelope_cells": [{"x": cell.x, "y": cell.y} for cell in decision.obstacle_envelope_cells],
            "detour_waypoint_count": len(decision.detour_map_waypoints),
            "detour_map_waypoints": [{"x_m": x, "y_m": y} for x, y in decision.detour_map_waypoints],
            "reason": decision.reason,
        }
        self.trace.write(json.dumps(payload) + "\n")
        self.decisions.append(payload)
        self.sequence += 1

    def close(self) -> None:
        self.trace.close()


def load_static_route_blocked(path: Path | None, *, grid: GridSpec) -> frozenset[GridCell]:
    """Load a provenance-pinned known-static occupancy layer, if supplied."""
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
    raw_cells = raw.get("blocked_cells")
    if not isinstance(raw_cells, list):
        raise ValueError("static route occupancy has no blocked cells")
    cells: set[GridCell] = set()
    for item in raw_cells:
        if not isinstance(item, dict):
            raise ValueError("static route occupancy cell is malformed")
        cell = GridCell(int(item["x"]), int(item["y"]))
        if not grid.contains(cell):
            raise ValueError("static route occupancy contains an out-of-grid cell")
        cells.add(cell)
    return frozenset(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--localization-trace", type=Path, default=None)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--background-capture", type=Path, default=None)
    parser.add_argument("--static-route-blocked", type=Path, default=None)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--static-match-distance-m", type=float, default=0.40)
    parser.add_argument("--minimum-cluster-cells", type=int, default=3)
    parser.add_argument("--minimum-obstacle-span-x-m", type=float, default=0.0)
    parser.add_argument("--minimum-obstacle-span-y-m", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.trace.exists() or args.result.exists():
        parser.error("trace/result must be new")
    if not 1.0 <= args.duration_s <= 60.0 or args.static_match_distance_m <= 0.0 or args.minimum_cluster_cells < 1 or args.minimum_obstacle_span_x_m < 0.0 or args.minimum_obstacle_span_y_m < 0.0:
        parser.error("invalid bounded diagnostic arguments")
    candidates, ranges = load_database(args.static_database)
    grid = grid_from_candidates(candidates, resolution_m=0.10)
    static_route_blocked = load_static_route_blocked(args.static_route_blocked, grid=grid)
    background = () if args.background_capture is None else load_map_background_capture(args.background_capture)
    # A measured map-frame background deliberately supersedes the coarse
    # static-signature surface for novelty matching.  Avoid spending the
    # diagnostic's bounded capture window rasterizing a field it will not use.
    if background:
        static_surface = frozenset()
    else:
        if args.localization_trace is None:
            parser.error("--localization-trace is required without --background-capture")
        static_surface = static_surface_cells_from_database(
            candidates, ranges,
            static_to_map=calibration_from_localization_trace(args.localization_trace),
            grid=grid,
        )
    config = ObstaclePipelineConfig(
        grid=grid, static_surface=static_surface, static_route_blocked=static_route_blocked,
        start=grid.world_to_cell(-1.5, -1.5), goal=grid.world_to_cell(0.9, -1.5),
        static_match_distance_m=args.static_match_distance_m,
        obstacle_inflation_m=0.46565748847422734,
        confirmation_frames=2, minimum_confirmed_cluster_cells=args.minimum_cluster_cells,
        map_background_points=background,
        minimum_obstacle_span_x_m=args.minimum_obstacle_span_x_m,
        minimum_obstacle_span_y_m=args.minimum_obstacle_span_y_m,
    )
    rclpy.init()
    node = Diagnostic(MapFrameObstaclePipeline(config), args.trace, topic=args.topic)
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        result = {
            "schema": "robotsim.issue80.lidar-obstacle-diagnostic.v1",
            "classification": "SIMULATION_ONLY_READ_ONLY_OBSTACLE_DIAGNOSTIC",
            "status": "PASS_DIAGNOSTIC_CAPTURED" if node.decisions else "BLOCKED_NO_POINTCLOUD",
            "decision_count": len(node.decisions),
            "final_decision": node.decisions[-1] if node.decisions else None,
            "novelty_reference": "MAP_BACKGROUND_CAPTURE" if background else "STATIC_SURFACE_FIELD",
            "background_point_count": len(background),
            "static_surface_cell_count": len(static_surface),
            "static_route_blocked_cell_count": len(static_route_blocked),
            "static_route_blocked_source": None if args.static_route_blocked is None else str(args.static_route_blocked),
            "topic": args.topic,
            "minimum_obstacle_span_m": {"x": args.minimum_obstacle_span_x_m, "y": args.minimum_obstacle_span_y_m},
            "velocity_publisher_created": False, "service_clients_created": 0,
        }
        args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if node.decisions else 2
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
