#!/usr/bin/env python3
"""Read-only bounded capture of the SIMULATION_ONLY RAICOM LiDAR topic."""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from issue80_lidar_map_obstacle_field import classify_map_frame_capture


TOPIC = "/aima/sim/lidar/points"
MAX_RETAINED_FINITE_POINTS = 4_096


def summarize(message: PointCloud2, *, retain_all_finite_points: bool = False) -> dict[str, Any]:
    fields = [{"name": field.name, "offset": int(field.offset), "datatype": int(field.datatype)} for field in message.fields]
    offsets = {field["name"]: field["offset"] for field in fields}
    valid_xyz = 0
    sample_xyz: list[list[float]] = []
    xyz_min: list[float] | None = None
    xyz_max: list[float] | None = None
    retained_xyz: list[list[float]] | None = [] if retain_all_finite_points else None
    declared_count = int(message.width) * int(message.height)
    retention_blocked = retain_all_finite_points and declared_count > MAX_RETAINED_FINITE_POINTS
    if all(axis in offsets for axis in ("x", "y", "z")) and message.point_step >= 12:
        count = min(declared_count, MAX_RETAINED_FINITE_POINTS if retain_all_finite_points else 1_000)
        for index in range(count):
            start = index * int(message.point_step)
            if start + max(offsets[axis] for axis in ("x", "y", "z")) + 4 > len(message.data):
                break
            xyz = [struct.unpack_from("<f", message.data, start + offsets[axis])[0] for axis in ("x", "y", "z")]
            if all(math.isfinite(value) for value in xyz):
                valid_xyz += 1
                if len(sample_xyz) < 8:
                    sample_xyz.append(xyz)
                if retained_xyz is not None:
                    retained_xyz.append(xyz)
                if xyz_min is None:
                    xyz_min, xyz_max = list(xyz), list(xyz)
                else:
                    xyz_min = [min(lower, value) for lower, value in zip(xyz_min, xyz)]
                    xyz_max = [max(upper, value) for upper, value in zip(xyz_max, xyz)]
    return {
        "frame_id": str(message.header.frame_id),
        "stamp_sec": int(message.header.stamp.sec),
        "stamp_nanosec": int(message.header.stamp.nanosec),
        "height": int(message.height),
        "width": int(message.width),
        "point_step": int(message.point_step),
        "row_step": int(message.row_step),
        "data_bytes": len(message.data),
        "fields": fields,
        "valid_xyz_sample_count": valid_xyz,
        "valid_xyz_examples": sample_xyz,
        "all_finite_xyz": retained_xyz,
        "all_finite_xyz_retention_requested": retain_all_finite_points,
        "all_finite_xyz_retention_blocked": retention_blocked,
        "all_finite_xyz_maximum_points": MAX_RETAINED_FINITE_POINTS,
        "valid_xyz_bounds": {"min": xyz_min, "max": xyz_max},
    }


class Capture(Node):
    def __init__(self, *, retain_all_finite_points: bool = False) -> None:
        super().__init__("issue80_lidar_capture")
        self.count = 0
        self.nonempty_count = 0
        self.last: dict[str, Any] | None = None
        self.retain_all_finite_points = retain_all_finite_points
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PointCloud2, TOPIC, self._callback, qos)

    def _callback(self, message: PointCloud2) -> None:
        self.count += 1
        self.last = summarize(message, retain_all_finite_points=self.retain_all_finite_points)
        if self.last["width"] > 0 and self.last["data_bytes"] > 0 and self.last["valid_xyz_sample_count"] > 0:
            self.nonempty_count += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument(
        "--expected-frame-id", default=None,
        help="optional exact PointCloud2 header.frame_id contract, e.g. map",
    )
    parser.add_argument(
        "--save-all-finite-points", action="store_true",
        help=f"retain every finite XYZ point in the final frame (bounded to {MAX_RETAINED_FINITE_POINTS})",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if not 1.0 <= args.duration_s <= 20.0:
        parser.error("duration must be within [1, 20] seconds")
    rclpy.init()
    node = Capture(retain_all_finite_points=args.save_all_finite_points)
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        status = classify_map_frame_capture(
            message_count=node.count,
            nonempty_message_count=node.nonempty_count,
            last_frame_id=None if node.last is None else str(node.last.get("frame_id", "")),
            expected_frame_id=args.expected_frame_id,
        )
        if node.last is not None and node.last["all_finite_xyz_retention_blocked"]:
            status = "BLOCKED_POINTCLOUD_CAPTURE_LIMIT_EXCEEDED"
        result = {
            "schema": "robotsim.issue80.raicom-lidar-capture.v1",
            "classification": "SIMULATION_ONLY_RAYCASTER_POINTCLOUD_CAPTURE",
            "status": status,
            "topic": TOPIC,
            "expected_frame_id": args.expected_frame_id,
            "save_all_finite_points": args.save_all_finite_points,
            "duration_s": args.duration_s,
            "message_count": node.count,
            "nonempty_message_count": node.nonempty_count,
            "last_message": node.last,
            "read_only": True,
            "velocity_publisher_created": False,
            "mc_action_or_source_client_created": False,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if status.startswith("PASS") else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
