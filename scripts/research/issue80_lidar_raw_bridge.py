#!/usr/bin/env python3
"""Publish source-local LiDAR frames traced by the SIMULATION_ONLY plugin.

This bridge has one input: the evidence-local append-only JSONL trace written
by ``robotsim.issue80.ray_caster_lidar``.  It has no MC service client, odom
subscriber, velocity publisher, or robot-state write path.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Iterable


RAW_SCHEMA = "robotsim.issue80.lidar-raw.v1"
DEFAULT_TOPIC = "/robotsim/lidar_chest_front/points"
DEFAULT_FRAME = "lidar_chest_front"


def decode_frame(payload: Any) -> list[tuple[float, float, float]]:
    if not isinstance(payload, dict) or payload.get("schema") != RAW_SCHEMA:
        raise ValueError("unexpected raw LiDAR frame schema")
    xyz = payload.get("xyz")
    if not isinstance(xyz, list):
        raise ValueError("raw LiDAR frame lacks xyz list")
    points: list[tuple[float, float, float]] = []
    for point in xyz:
        if not isinstance(point, list) or len(point) != 3:
            continue
        try:
            parsed = tuple(float(value) for value in point)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in parsed):
            points.append(parsed)
    return points


def read_new_frames(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not path.exists():
        return offset, []
    if path.is_symlink() or not path.is_file():
        raise ValueError("raw LiDAR trace is unsafe")
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read()
        next_offset = stream.tell()
    frames: list[dict[str, Any]] = []
    for raw in payload.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            next_offset -= len(raw)
            break
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            frames.append(decoded)
    return next_offset, frames


def point_bytes(points: Iterable[tuple[float, float, float]]) -> bytes:
    return b"".join(struct.pack("<fff", *point) for point in points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists() or args.result.is_symlink():
        parser.error("result must be new")
    if not 1.0 <= args.duration_s <= 120.0:
        parser.error("duration must be in [1, 120]")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField

    rclpy.init()
    node = Node("issue80_lidar_raw_bridge")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    publisher = node.create_publisher(PointCloud2, args.topic, qos)
    offset = 0
    published = 0
    points_last = 0
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            offset, frames = read_new_frames(args.trace, offset)
            for raw in frames:
                points = decode_frame(raw)
                if not points:
                    continue
                message = PointCloud2()
                message.header.stamp = node.get_clock().now().to_msg()
                message.header.frame_id = args.frame
                message.height = 1
                message.width = len(points)
                message.fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                message.is_bigendian = False
                message.point_step = 12
                message.row_step = message.width * message.point_step
                message.is_dense = True
                message.data = point_bytes(points)
                publisher.publish(message)
                published += 1
                points_last = len(points)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)
        result = {"schema": "robotsim.issue80.lidar-raw-bridge.v1", "status": "PASS" if published else "BLOCKED_NO_RAW_FRAMES", "topic": args.topic, "frame": args.frame, "published_frames": published, "last_point_count": points_last, "velocity_publisher_created": False, "mc_client_created": False}
        args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if published else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
