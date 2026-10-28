#!/usr/bin/env python3
"""SIMULATION_ONLY scan-to-scan LiDAR localization for the RAICOM X2 room.

The input is the evidence-local ray-caster point cloud.  This node deliberately
does not subscribe to `/aima/hal/odom/state`; its map pose evolves only from
the configured source START prior and LiDAR scan registration.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from issue80_lidar_scan_math import Pose2, scan_delta, wrap


LIDAR_TOPIC = "/aima/sim/lidar/points"
LOCALIZATION_TOPIC = "/robotsim/lidar_localization_pose"
SOURCE_START_XY_YAW = (-1.5, -1.5, 0.0)
CHEST_LIDAR_X_OFFSET_M = 0.102632788961867
MIN_TRANSLATION_UPDATE_M = 0.01
MIN_YAW_UPDATE_RAD = 0.01


def xy_from_cloud(message: PointCloud2) -> np.ndarray:
    """Project source-derived LiDAR-frame points onto the horizontal plane."""
    offsets = {field.name: int(field.offset) for field in message.fields}
    if not all(axis in offsets for axis in ("x", "y", "z")) or message.point_step < 12:
        return np.empty((0, 2), dtype=np.float64)
    result: list[tuple[float, float]] = []
    count = int(message.width) * int(message.height)
    for index in range(count):
        base = index * int(message.point_step)
        if base + max(offsets[axis] for axis in ("x", "y", "z")) + 4 > len(message.data):
            break
        x, y, z = (struct.unpack_from("<f", message.data, base + offsets[axis])[0]
                   for axis in ("x", "y", "z"))
        # Plugin convention: camera -Z points forward; X is lateral; Y is up.
        forward, lateral = -z, x
        if math.isfinite(forward) and math.isfinite(lateral) and 0.25 <= math.hypot(forward, lateral) <= 7.8:
            result.append((forward, lateral))
    return np.asarray(result, dtype=np.float64)


class LidarLocalizer(Node):
    def __init__(self, start: Pose2, trace_path: Path | None = None, topic: str = LIDAR_TOPIC) -> None:
        super().__init__("issue80_lidar_localizer")
        self.pose = start
        self.previous: np.ndarray | None = None
        self.last_result: dict[str, float | int | str] | None = None
        self.trace = trace_path.open("x", encoding="utf-8", buffering=1) if trace_path else None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PointCloud2, topic, self._scan, qos)
        self.publisher = self.create_publisher(PoseStamped, LOCALIZATION_TOPIC, 10)

    def _publish(self, stamp) -> None:
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        c, s = math.cos(self.pose.yaw / 2.0), math.sin(self.pose.yaw / 2.0)
        message.pose.position.x = self.pose.x - CHEST_LIDAR_X_OFFSET_M * math.cos(self.pose.yaw)
        message.pose.position.y = self.pose.y - CHEST_LIDAR_X_OFFSET_M * math.sin(self.pose.yaw)
        message.pose.orientation.z = s
        message.pose.orientation.w = c
        self.publisher.publish(message)

    def _scan(self, message: PointCloud2) -> None:
        current = xy_from_cloud(message)
        if len(current) < 16:
            return
        if self.previous is not None:
            try:
                forward, lateral, yaw, residual = scan_delta(self.previous, current)
            except ValueError:
                return
            displacement = math.hypot(forward, lateral)
            if displacement < MIN_TRANSLATION_UPDATE_M and abs(yaw) < MIN_YAW_UPDATE_RAD:
                self.last_result = {
                    "status": "SCAN_STATIC_HOLD",
                    "forward_m": forward,
                    "lateral_m": lateral,
                    "yaw_rad": yaw,
                    "residual_m": residual,
                    "point_count": int(len(current)),
                }
            elif residual <= 0.20 and displacement <= 0.25 and abs(yaw) <= 0.20:
                self.pose = self.pose.apply_delta(forward, lateral, yaw)
                self.last_result = {
                    "status": "SCAN_DELTA_ACCEPTED",
                    "forward_m": forward,
                    "lateral_m": lateral,
                    "yaw_rad": yaw,
                    "residual_m": residual,
                    "point_count": int(len(current)),
                }
            else:
                self.last_result = {
                    "status": "SCAN_DELTA_REJECTED",
                    "residual_m": residual,
                    "forward_m": forward,
                    "lateral_m": lateral,
                    "yaw_rad": yaw,
                    "point_count": int(len(current)),
                }
        self.previous = current
        self._publish(message.header.stamp)
        if self.trace is not None:
            self.trace.write(json.dumps({
                "host_monotonic_s": time.monotonic(),
                "stamp_sec": int(message.header.stamp.sec),
                "stamp_nanosec": int(message.header.stamp.nanosec),
                "pose": self.pose.__dict__,
                "scan": self.last_result,
            }, allow_nan=False) + "\n")

    def close(self) -> None:
        if self.trace is not None:
            self.trace.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-x", type=float, default=SOURCE_START_XY_YAW[0])
    parser.add_argument("--start-y", type=float, default=SOURCE_START_XY_YAW[1])
    parser.add_argument("--start-yaw", type=float, default=SOURCE_START_XY_YAW[2])
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--topic", default=LIDAR_TOPIC)
    args = parser.parse_args()
    if not 1.0 <= args.duration_s <= 120.0:
        parser.error("duration must be in [1, 120]")
    if args.trace is not None and args.trace.exists():
        parser.error("--trace must be new")
    if args.result is not None and args.result.exists():
        parser.error("--result must be new")
    rclpy.init()
    node = LidarLocalizer(Pose2(args.start_x, args.start_y, args.start_yaw), args.trace, args.topic)
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        result: dict[str, Any] = {"pose": node.pose.__dict__, "last_scan": node.last_result}
        if args.result is not None:
            args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
