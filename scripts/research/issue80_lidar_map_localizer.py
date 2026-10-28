#!/usr/bin/env python3
"""Known-map LiDAR/odom localization from raw local chest scans.

The static signature database is built from the pinned RAICOM scene.  A raw
LiDAR scan supplies bounded, low-rate global corrections while MC odometry
propagates the map pose between scans.  This is deliberately explicit: the
published pose is a SIMULATION_ONLY LiDAR/odom *estimate*, not MuJoCo ground
truth.  In particular, a coarse static-map candidate must never make the
route controller jump by one grid cell at every scan.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np

from issue80_lidar_raw_bridge import RAW_SCHEMA, read_new_frames

MAGIC = b"RSLMAP1\0"
LIDAR_CORRECTION_TRANSLATION_M = 0.30
LIDAR_CORRECTION_YAW_RAD = 0.25
LIDAR_CORRECTION_BLEND = 0.15


@dataclass(frozen=True)
class Pose2D:
    """A planar pose in the REP-105 map or odom frame."""

    x_m: float
    y_m: float
    yaw_rad: float


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


def map_to_odom_from_correspondence(map_pose: Pose2D, odom_pose: Pose2D) -> Pose2D:
    """Return ``map <- odom`` that maps ``odom_pose`` onto ``map_pose``."""
    yaw = wrap(map_pose.yaw_rad - odom_pose.yaw_rad)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return Pose2D(
        map_pose.x_m - (cosine * odom_pose.x_m - sine * odom_pose.y_m),
        map_pose.y_m - (sine * odom_pose.x_m + cosine * odom_pose.y_m),
        yaw,
    )


def apply_map_to_odom(map_to_odom: Pose2D, odom_pose: Pose2D) -> Pose2D:
    """Apply a ``map <- odom`` transform to one odometry pose."""
    cosine, sine = math.cos(map_to_odom.yaw_rad), math.sin(map_to_odom.yaw_rad)
    return Pose2D(
        map_to_odom.x_m + cosine * odom_pose.x_m - sine * odom_pose.y_m,
        map_to_odom.y_m + sine * odom_pose.x_m + cosine * odom_pose.y_m,
        wrap(map_to_odom.yaw_rad + odom_pose.yaw_rad),
    )


def blend_map_to_odom(current: Pose2D, observed: Pose2D, weight: float) -> Pose2D:
    """Blend a bounded scan correction into an existing map/odom transform."""
    if not 0.0 < weight <= 1.0:
        raise ValueError("correction blend weight must be in (0, 1]")
    return Pose2D(
        current.x_m + weight * (observed.x_m - current.x_m),
        current.y_m + weight * (observed.y_m - current.y_m),
        wrap(current.yaw_rad + weight * wrap(observed.yaw_rad - current.yaw_rad)),
    )


def pose_residual(reference: Pose2D, observed: Pose2D) -> tuple[float, float]:
    """Return translation and signed yaw from a propagated to a scan pose."""
    return (
        math.hypot(observed.x_m - reference.x_m, observed.y_m - reference.y_m),
        wrap(observed.yaw_rad - reference.yaw_rad),
    )


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
    map_to_odom: Pose2D | None = None
    last_published_pose: Pose2D | None = None

    def odom_pose() -> Pose2D | None:
        if latest_odom is None:
            return None
        position = latest_odom.pose.pose.position
        return Pose2D(float(position.x), float(position.y), yaw_from_quaternion(latest_odom.pose.pose.orientation))

    def publish_estimate(pose: Pose2D) -> None:
        nonlocal last_published_pose
        message = PoseStamped()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.position.x, message.pose.position.y = pose.x_m, pose.y_m
        message.pose.orientation.z = math.sin(pose.yaw_rad / 2.0)
        message.pose.orientation.w = math.cos(pose.yaw_rad / 2.0)
        pose_pub.publish(message)
        last_published_pose = pose
        if latest_odom is not None and map_to_odom is not None:
            transform = TransformStamped()
            transform.header.stamp = message.header.stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = "odom"
            transform.transform.translation.x = map_to_odom.x_m
            transform.transform.translation.y = map_to_odom.y_m
            transform.transform.rotation.z = math.sin(map_to_odom.yaw_rad / 2.0)
            transform.transform.rotation.w = math.cos(map_to_odom.yaw_rad / 2.0)
            tf_pub.sendTransform(transform)

    def odom_callback(message: Odometry) -> None:
        nonlocal latest_odom
        latest_odom = message
        # Preserve a continuous control pose between the lower-rate raw scans.
        # No pose is published until a LiDAR map correspondence initialized the
        # map->odom transform.
        if map_to_odom is not None:
            propagated = apply_map_to_odom(map_to_odom, odom_pose())
            publish_estimate(propagated)
    node.create_subscription(Odometry, "/aima/hal/odom/state", odom_callback, qos)
    offset = 0
    accepted = 0
    last_scan_pose: Pose2D | None = None
    source_start = np.array([-1.5, -1.5, 0.0], dtype=np.float32)
    calibration: np.ndarray | None = None
    trace = args.trace_output.open("x", encoding="utf-8", buffering=1)
    try:
        deadline = time.monotonic() + args.duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            offset, frames = read_new_frames(args.trace, offset)
            for raw in frames:
                scan = raw_ranges(raw, ranges.shape[1])
                propagated = apply_map_to_odom(map_to_odom, odom_pose()) if map_to_odom is not None and odom_pose() is not None else None
                raw_center = None if calibration is None or propagated is None else np.array(
                    [propagated.x_m, propagated.y_m, propagated.yaw_rad], dtype=np.float32
                ) - calibration
                index, score, matches = best_candidate(
                    candidates, ranges, scan, center=raw_center,
                    max_translation_m=0.35, max_yaw_rad=0.25,
                )
                raw_pose = np.array(candidates[index], dtype=np.float32)
                if calibration is None:
                    calibration = source_start - raw_pose
                corrected = raw_pose + calibration
                corrected[2] = wrap(float(corrected[2]))
                scan_pose = Pose2D(*[float(value) for value in corrected])
                scan_qualified = math.isfinite(score) and score <= 0.16 and matches >= 80
                residual_translation_m: float | None = None
                residual_yaw_rad: float | None = None
                correction_accepted = False
                current_odom = odom_pose()
                if scan_qualified and current_odom is not None:
                    if map_to_odom is None:
                        map_to_odom = map_to_odom_from_correspondence(scan_pose, current_odom)
                        correction_accepted = True
                    else:
                        assert propagated is not None
                        residual_translation_m, residual_yaw_rad = pose_residual(propagated, scan_pose)
                        if residual_translation_m <= LIDAR_CORRECTION_TRANSLATION_M and abs(residual_yaw_rad) <= LIDAR_CORRECTION_YAW_RAD:
                            observed = map_to_odom_from_correspondence(scan_pose, current_odom)
                            map_to_odom = blend_map_to_odom(map_to_odom, observed, LIDAR_CORRECTION_BLEND)
                            correction_accepted = True
                if correction_accepted:
                    assert map_to_odom is not None and current_odom is not None
                    pose = apply_map_to_odom(map_to_odom, current_odom)
                    last_scan_pose = scan_pose
                    accepted += 1
                    publish_estimate(pose)
                trace.write(json.dumps({
                    "host_monotonic_s": time.monotonic(), "sequence": raw.get("sequence"),
                    "scan_candidate": {"x": scan_pose.x_m, "y": scan_pose.y_m, "yaw": scan_pose.yaw_rad},
                    "propagated_pose": None if propagated is None else {"x": propagated.x_m, "y": propagated.y_m, "yaw": propagated.yaw_rad},
                    "map_to_odom": None if map_to_odom is None else {"x": map_to_odom.x_m, "y": map_to_odom.y_m, "yaw": map_to_odom.yaw_rad},
                    "calibration": calibration.tolist() if calibration is not None else None,
                    "score": score, "matching_rays": matches, "scan_qualified": scan_qualified,
                    "residual_translation_m": residual_translation_m, "residual_yaw_rad": residual_yaw_rad,
                    "lidar_correction_accepted": correction_accepted,
                }, allow_nan=False) + "\n")
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)
        result = {
            "schema": "robotsim.issue80.lidar-map-localizer.v2",
            "status": "PASS" if accepted else "BLOCKED_NO_ACCEPTED_MAP_MATCH",
            "accepted_lidar_corrections": accepted,
            "last_scan_pose_map": None if last_scan_pose is None else last_scan_pose.__dict__,
            "last_published_pose_map": None if last_published_pose is None else last_published_pose.__dict__,
            "localization_uses_odom": True,
            "odom_usage": "continuous_map_pose_propagation_and_map_to_odom_transform",
            "lidar_usage": "bounded_static_map_correction_only",
            "velocity_publisher_created": False,
            "mc_client_created": False,
        }
        args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if accepted else 2
    finally:
        trace.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
