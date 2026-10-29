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

from issue80_lidar_map_obstacle_field import static_surface_cells_from_database
from issue80_lidar_map_localizer import load_database
from issue80_lidar_static_surface_audit import calibration_from_localization_trace
from issue80_lidar_obstacle_layer import GridSpec
from issue80_lidar_obstacle_pipeline import MapFrameObstaclePipeline, ObstaclePipelineConfig
from issue80_lidar_obstacle_replay import grid_from_candidates


TOPIC = "/aima/sim/lidar/points"


class Diagnostic(Node):
    def __init__(self, pipeline: MapFrameObstaclePipeline, trace: Path) -> None:
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
        self.subscription_handles = [self.create_subscription(PointCloud2, TOPIC, self.callback, qos)]

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
            "detour_waypoint_count": len(decision.detour_map_waypoints), "reason": decision.reason,
        }
        self.trace.write(json.dumps(payload) + "\n")
        self.decisions.append(payload)
        self.sequence += 1

    def close(self) -> None:
        self.trace.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-database", type=Path, required=True)
    parser.add_argument("--localization-trace", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--static-match-distance-m", type=float, default=0.40)
    parser.add_argument("--minimum-cluster-cells", type=int, default=3)
    args = parser.parse_args(argv)
    if args.trace.exists() or args.result.exists():
        parser.error("trace/result must be new")
    if not 1.0 <= args.duration_s <= 60.0 or args.static_match_distance_m <= 0.0 or args.minimum_cluster_cells < 1:
        parser.error("invalid bounded diagnostic arguments")
    candidates, ranges = load_database(args.static_database)
    grid = grid_from_candidates(candidates, resolution_m=0.10)
    static_surface = static_surface_cells_from_database(
        candidates, ranges, static_to_map=calibration_from_localization_trace(args.localization_trace), grid=grid,
    )
    config = ObstaclePipelineConfig(
        grid=grid, static_surface=static_surface, static_route_blocked=frozenset(),
        start=grid.world_to_cell(-1.5, -1.5), goal=grid.world_to_cell(0.9, -1.5),
        static_match_distance_m=args.static_match_distance_m,
        obstacle_inflation_m=0.46565748847422734,
        confirmation_frames=2, minimum_confirmed_cluster_cells=args.minimum_cluster_cells,
    )
    rclpy.init()
    node = Diagnostic(MapFrameObstaclePipeline(config), args.trace)
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
