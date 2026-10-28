#!/usr/bin/env python3
"""Known-map LiDAR localization from raw local chest scans.

The static signature database is built from the pinned RAICOM scene.  Online
pose selection reads only raw local sensor frames, never live odometry.  Odom
is separately consumed only to publish the derived map->odom transform.
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

from issue80_lidar_raw_bridge import RAW_SCHEMA, read_new_frames

MAGIC = b"RSLMAP1\0"


def load_database(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("invalid static LiDAR map magic")
        count, rays = struct.unpack("<II", stream.read(8))
    if count <= 0 or rays <= 0 or rays > 4096:
        raise ValueError("invalid static LiDAR map dimensions")
    values = np.memmap(path, dtype="<f4", mode="r", offset=16, shape=(count, 3 + rays))
    return np.asarray(values[:, :3]), np.asarray(values[:, 3:])


def raw_ranges(frame: dict[str, Any], rays: int) -> np.ndarray:
    if frame.get("schema") != RAW_SCHEMA:
        raise ValueError("unexpected raw LiDAR schema")
    xyz = frame.get("xyz")
    if not isinstance(xyz, list) or len(xyz) != rays:
        raise ValueError("raw LiDAR ray count does not match static map")
    result = np.full(rays, np.nan, dtype=np.float32)
    for index, point in enumerate(xyz):
        if not isinstance(point, list) or len(point) != 3:
            continue
        try:
            values = tuple(float(value) for value in point)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            result[index] = math.sqrt(sum(value * value for value in values))
    return result


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def best_candidate(candidates: np.ndarray, ranges: np.ndarray, scan: np.ndarray, *, minimum_matches: int = 80, center: np.ndarray | None = None, max_translation_m: float = math.inf, max_yaw_rad: float = math.inf) -> tuple[int, float, int]:
    valid = np.isfinite(ranges) & np.isfinite(scan)[None, :]
    count = valid.sum(axis=1)
    difference = np.where(valid, np.abs(ranges - scan[None, :]), np.nan)
    score = np.nanmedian(difference, axis=1)
    score[count < minimum_matches] = np.inf
    if center is not None:
        translation = np.linalg.norm(candidates[:, :2] - center[:2], axis=1)
        yaw = np.abs(np.arctan2(np.sin(candidates[:, 2] - center[2]), np.cos(candidates[:, 2] - center[2])))
        score[(translation > max_translation_m) | (yaw > max_yaw_rad)] = np.inf
    index = int(np.argmin(score))
    return index, float(score[index]), int(count[index])


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)), 1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--pose-topic", default="/map_tf_distribution/localization_pose")
    parser.add_argument("--trace-output", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists() or args.trace_output.exists():
        parser.error("outputs must be new")
    if not 1.0 <= args.duration_s <= 120.0:
        parser.error("duration must be in [1, 120]")
    candidates, ranges = load_database(args.database)

    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = Node("issue80_lidar_map_localizer")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    pose_pub = node.create_publisher(PoseStamped, args.pose_topic, 10)
    tf_pub = TransformBroadcaster(node)
    latest_odom: Odometry | None = None
    def odom_callback(message: Odometry) -> None:
        nonlocal latest_odom
        latest_odom = message
    node.create_subscription(Odometry, "/aima/hal/odom/state", odom_callback, qos)
    offset = 0
    accepted = 0
    last_pose: tuple[float, float, float] | None = None
    source_start = np.array([-1.5, -1.5, 0.0], dtype=np.float32)
    calibration: np.ndarray | None = None
    trace = args.trace_output.open("x", encoding="utf-8", buffering=1)
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            offset, frames = read_new_frames(args.trace, offset)
            for raw in frames:
                scan = raw_ranges(raw, ranges.shape[1])
                raw_center = None if calibration is None or last_pose is None else np.array(last_pose, dtype=np.float32) - calibration
                index, score, matches = best_candidate(
                    candidates, ranges, scan, center=raw_center,
                    max_translation_m=0.35, max_yaw_rad=0.25,
                )
                raw_pose = np.array(candidates[index], dtype=np.float32)
                if calibration is None:
                    calibration = source_start - raw_pose
                corrected = raw_pose + calibration
                corrected[2] = wrap(float(corrected[2]))
                pose = tuple(float(value) for value in corrected)
                accepted_now = math.isfinite(score) and score <= 0.16 and matches >= 80
                if accepted_now:
                    last_pose = pose
                    accepted += 1
                    message = PoseStamped()
                    message.header.stamp = node.get_clock().now().to_msg()
                    message.header.frame_id = "map"
                    message.pose.position.x, message.pose.position.y = pose[0], pose[1]
                    message.pose.orientation.z = math.sin(pose[2] / 2.0)
                    message.pose.orientation.w = math.cos(pose[2] / 2.0)
                    pose_pub.publish(message)
                    if latest_odom is not None:
                        op = latest_odom.pose.pose.position
                        oyaw = yaw_from_quaternion(latest_odom.pose.pose.orientation)
                        dyaw = wrap(pose[2] - oyaw)
                        c, s = math.cos(dyaw), math.sin(dyaw)
                        dx = pose[0] - (c * float(op.x) - s * float(op.y))
                        dy = pose[1] - (s * float(op.x) + c * float(op.y))
                        transform = TransformStamped()
                        transform.header.stamp = message.header.stamp
                        transform.header.frame_id = "map"
                        transform.child_frame_id = "odom"
                        transform.transform.translation.x = dx
                        transform.transform.translation.y = dy
                        transform.transform.rotation.z = math.sin(dyaw / 2.0)
                        transform.transform.rotation.w = math.cos(dyaw / 2.0)
                        tf_pub.sendTransform(transform)
                trace.write(json.dumps({"host_monotonic_s": time.monotonic(), "sequence": raw.get("sequence"), "candidate": {"x": pose[0], "y": pose[1], "yaw": pose[2]}, "raw_candidate": {"x": float(raw_pose[0]), "y": float(raw_pose[1]), "yaw": float(raw_pose[2])}, "calibration": calibration.tolist() if calibration is not None else None, "score": score, "matching_rays": matches, "accepted": accepted_now}, allow_nan=False) + "\n")
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)
        result = {"schema": "robotsim.issue80.lidar-map-localizer.v1", "status": "PASS" if accepted else "BLOCKED_NO_ACCEPTED_MAP_MATCH", "accepted_frames": accepted, "last_pose_map": last_pose, "localization_uses_odom": False, "odom_usage": "map_to_odom_tf_only", "velocity_publisher_created": False, "mc_client_created": False}
        args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if accepted else 2
    finally:
        trace.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
