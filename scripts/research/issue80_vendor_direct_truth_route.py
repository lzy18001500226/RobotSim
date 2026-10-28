#!/usr/bin/env python3
"""Minimal official-style X2 velocity publisher with MuJoCo-odom stop.

This deliberately follows the delivered ``mc_locomotion_velocity.py`` contract:
``node`` is ADD-registered at priority 40 and publishes one 0.20 m/s command
stream at 50 Hz.  It is a bounded SIMULATION_ONLY gait baseline, not a route
planner and not a replacement MC controller.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


PUBLISH_PERIOD_S = 0.02
ZERO_HOLD_S = 2.0
ODOM_FRESHNESS_S = 0.5
MIN_ROOT_Z_M = 0.52
MAX_TILT_RAD = math.radians(15.0)
HEADING_HOLD_COMMAND_RAD_S = 0.075
HEADING_HOLD_DEADBAND_RAD = 0.04


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def roll_pitch_yaw(q: Any) -> tuple[float, float, float]:
    """Return REP-103 roll/pitch/yaw for a ROS quaternion."""
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


@dataclass(frozen=True)
class PoseSample:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    linear_x_m_s: float
    linear_y_m_s: float
    source_stamp_ns: int
    received_monotonic_s: float


class DirectTruthRoute(Node):
    """Reuse the vendor ADD + 50 Hz contract; truth only chooses the stop time."""

    def __init__(self, target_distance_m: float, trace_path: Path, localization_topic: str | None = None) -> None:
        super().__init__("issue80_vendor_direct_truth_route")
        self.target_distance_m = target_distance_m
        self.trace_path = trace_path
        self.trace = trace_path.open("x", encoding="utf-8", buffering=1)
        self.publisher = self.create_publisher(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10
        )
        self.client = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )
        # Match the known-good M1 observation contract.  The simulator offers
        # odom as BEST_EFFORT, so a default RELIABLE subscription receives no
        # samples in Fast DDS and must never be mistaken for a stationary robot.
        self.odom_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.start: PoseSample | None = None
        self.last: PoseSample | None = None
        self.last_localization: PoseSample | None = None
        self.publish_times: list[float] = []
        self.heading_hold_commands: list[dict[str, float]] = []
        self.post_stop_samples: list[PoseSample] = []
        self.post_stop_started_monotonic_s: float | None = None
        self.nonzero_publish_calls = 0
        self.zero_publish_calls = 0
        self.create_subscription(Odometry, "/aima/hal/odom/state", self._odom, self.odom_qos)
        if localization_topic:
            self.create_subscription(PoseStamped, localization_topic, self._localization, self.odom_qos)

    def _odom(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        roll, pitch, yaw = roll_pitch_yaw(msg.pose.pose.orientation)
        sample = PoseSample(
            x_m=float(position.x),
            y_m=float(position.y),
            z_m=float(position.z),
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            linear_x_m_s=float(msg.twist.twist.linear.x),
            linear_y_m_s=float(msg.twist.twist.linear.y),
            source_stamp_ns=(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)),
            received_monotonic_s=time.monotonic(),
        )
        if self.start is None:
            self.start = sample
        self.last = sample
        if self.post_stop_started_monotonic_s is not None and sample.received_monotonic_s >= self.post_stop_started_monotonic_s:
            self.post_stop_samples.append(sample)
        self._trace({"event": "odom", **sample.__dict__})

    def _localization(self, msg: PoseStamped) -> None:
        roll, pitch, yaw = roll_pitch_yaw(msg.pose.orientation)
        self.last_localization = PoseSample(
            x_m=float(msg.pose.position.x), y_m=float(msg.pose.position.y), z_m=float(msg.pose.position.z),
            roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
            linear_x_m_s=0.0, linear_y_m_s=0.0,
            source_stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
            received_monotonic_s=time.monotonic(),
        )
        self._trace({"event": "localization", **self.last_localization.__dict__})

    def _trace(self, row: dict[str, Any]) -> None:
        self.trace.write(json.dumps({"monotonic_s": time.monotonic(), **row}, allow_nan=False) + "\n")

    def register(self) -> dict[str, Any]:
        if not self.client.wait_for_service(timeout_sec=5.0):
            return {"success": False, "fault": "SetMcInputSource service unavailable"}
        request = SetMcInputSource.Request()
        request.action.value = 1001
        request.input_source.name = "node"
        request.input_source.priority = 40
        request.input_source.timeout = 1000
        request.request.header.stamp = self.get_clock().now().to_msg()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            return {"success": False, "fault": "SetMcInputSource call timed out"}
        response = future.result()
        result = {
            "success": True,
            "source": "node",
            "action": "ADD",
            "priority": 40,
            "timeout_ms": 1000,
            "response_header_code": int(response.response.header.code),
            "task_id": int(response.response.task_id),
            "state_value": int(response.response.state.value),
        }
        self._trace({"event": "source_registration", **result})
        return result

    def publish_velocity(self, forward_m_s: float, angular_rad_s: float = 0.0) -> None:
        message = McLocomotionVelocity()
        message.header = MessageHeader()
        message.header.stamp = self.get_clock().now().to_msg()
        message.source = "node"
        message.forward_velocity = forward_m_s
        message.lateral_velocity = 0.0
        message.angular_velocity = angular_rad_s
        self.publisher.publish(message)
        now = time.monotonic()
        if forward_m_s == 0.0:
            self.zero_publish_calls += 1
        else:
            self.nonzero_publish_calls += 1
            self.publish_times.append(now)
        self._trace({
            "event": "velocity_command",
            "source": "node",
            "forward_velocity_m_s": forward_m_s,
            "lateral_velocity_m_s": 0.0,
            "angular_velocity_rad_s": angular_rad_s,
        })

    def distance_m(self) -> float | None:
        if self.start is None or self.last is None:
            return None
        return math.hypot(self.last.x_m - self.start.x_m, self.last.y_m - self.start.y_m)

    def motion_fault(self) -> str | None:
        if self.last is None:
            return "missing_odometry"
        if time.monotonic() - self.last.received_monotonic_s > ODOM_FRESHNESS_S:
            return "stale_odometry"
        if not all(math.isfinite(value) for value in self.last.__dict__.values() if isinstance(value, float)):
            return "nonfinite_odometry"
        if self.last.z_m < MIN_ROOT_Z_M:
            return "fall_root_height"
        if max(abs(self.last.roll_rad), abs(self.last.pitch_rad)) > MAX_TILT_RAD:
            return "fall_tilt"
        return None

    def close(self) -> None:
        self.trace.close()

    def physical_stop(self) -> dict[str, Any]:
        samples = self.post_stop_samples
        if len(samples) < 20:
            return {"status": "FAIL_INSUFFICIENT_FRESH_ODOM", "sample_count": len(samples)}
        duration = samples[-1].received_monotonic_s - samples[0].received_monotonic_s
        origin = samples[0]
        drift = max(math.hypot(item.x_m - origin.x_m, item.y_m - origin.y_m) for item in samples)
        speed = max(math.hypot(item.linear_x_m_s, item.linear_y_m_s) for item in samples)
        safe = all(item.z_m >= MIN_ROOT_Z_M and max(abs(item.roll_rad), abs(item.pitch_rad)) <= MAX_TILT_RAD for item in samples)
        return {
            "status": "PASS" if duration >= 1.9 and drift <= 0.05 and speed <= 0.05 and safe else "FAIL_PHYSICAL_STOP",
            "sample_count": len(samples), "duration_s": duration, "max_planar_drift_m": drift,
            "max_planar_speed_m_s": speed, "posture_safe": safe,
            "criteria": {"minimum_duration_s": 1.9, "maximum_planar_drift_m": 0.05, "maximum_planar_speed_m_s": 0.05},
        }


def write_result(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prior result: {path}")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rate_summary(times: list[float]) -> dict[str, float | int | None]:
    if len(times) < 2:
        return {"count": len(times), "mean_hz": None, "max_gap_s": None}
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    return {
        "count": len(times),
        "mean_hz": (len(gaps) / sum(gaps)) if sum(gaps) > 0.0 else None,
        "max_gap_s": max(gaps),
    }


def load_truth_target(path: Path, name: str) -> dict[str, float | str]:
    """Load one explicitly named raw-odom target from the staged mission JSON."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        target = document["poses"][name]
        x_m, y_m, yaw_rad = float(target["x_m"]), float(target["y_m"]), float(target["yaw_rad"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid truth target {name!r}: {exc}") from exc
    if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
        raise ValueError(f"truth target {name!r} has nonfinite pose")
    return {"name": name, "x_m": x_m, "y_m": y_m, "yaw_rad": yaw_rad}


def heading_hold_command(current_yaw_rad: float, target_yaw_rad: float) -> tuple[float, float]:
    """Use only the measured, symmetric Run65 angular response and a deadband."""
    error = wrap_angle(target_yaw_rad - current_yaw_rad)
    if error > HEADING_HOLD_DEADBAND_RAD:
        return error, HEADING_HOLD_COMMAND_RAD_S
    if error < -HEADING_HOLD_DEADBAND_RAD:
        return error, -HEADING_HOLD_COMMAND_RAD_S
    return error, 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--target-route-config", type=Path, default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--target-tolerance-m", type=float, default=None)
    parser.add_argument("--heading-hold", action="store_true")
    parser.add_argument("--localization-topic", default=None)
    args = parser.parse_args()
    if not 1.0 < args.distance_m <= 2.5:
        parser.error("--distance-m must be in (1.0, 2.5] for the bounded direct-gait baseline")
    if not 1.0 <= args.timeout_s <= 30.0:
        parser.error("--timeout-s must be in [1, 30]")
    if args.output.exists() or args.trace.exists():
        parser.error("output paths must be new")
    target_requested = args.target_route_config is not None
    if target_requested != (args.target_name is not None) or target_requested != (args.target_tolerance_m is not None):
        parser.error("target route, target name, and target tolerance must be supplied together")
    if target_requested and not 0.0 < float(args.target_tolerance_m) <= 0.15:
        parser.error("--target-tolerance-m must be in (0, 0.15]")
    if args.heading_hold and not target_requested:
        parser.error("--heading-hold requires an explicit truth target")
    target = load_truth_target(args.target_route_config, str(args.target_name)) if target_requested else None
    rclpy.init()
    node = DirectTruthRoute(args.distance_m, args.trace, args.localization_topic)
    result: dict[str, Any] = {
        "schema": "robotsim.issue80.vendor-direct-truth-route.v1",
        "classification": (
            "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET"
            if target is not None else "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_DISTANCE_BASELINE"
        ),
        "status": "BLOCKED",
        "target_distance_m": args.distance_m,
        "target_pose_odom": target,
        "target_pose_source": "map_localization" if args.localization_topic else "odom",
        "target_tolerance_m": args.target_tolerance_m,
        "final_target_error_m": None,
        "command": {"source": "node", "forward_m_s": 0.2, "publish_hz": 50.0},
        "heading_hold": {
            "enabled": bool(args.heading_hold),
            "target_yaw_rad": target.get("yaw_rad") if target else None,
            "deadband_rad": HEADING_HOLD_DEADBAND_RAD if args.heading_hold else None,
            "command_magnitude_rad_s": HEADING_HOLD_COMMAND_RAD_S if args.heading_hold else None,
        },
        "registration": None,
        "start_pose_odom": None,
        "final_pose_odom": None,
        "measured_distance_m": None,
        "nonzero_publish_calls_returned": 0,
        "zero_publish_calls_returned": 0,
        "publish_rate": None,
        "fault": None,
    }
    try:
        registration = node.register()
        result["registration"] = registration
        if not registration["success"]:
            result["fault"] = registration["fault"]
            return 2
        deadline = time.monotonic() + args.timeout_s
        # The parent has already qualified STAND, but this independently waits
        # for the direct node's first fresh odom callback.  A subscription that
        # has not yet received a sample is not a reason to publish motion.
        initial_odom_deadline = min(deadline, time.monotonic() + 4.0)
        while rclpy.ok() and (node.last is None or (target is not None and args.localization_topic and node.last_localization is None)) and time.monotonic() < initial_odom_deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
        if node.last is None or (target is not None and args.localization_topic and node.last_localization is None):
            result["fault"] = "initial_odometry_or_localization_unavailable"
            return 2
        next_publish = time.monotonic()
        terminal_reason = "timeout"
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.002)
            fault = node.motion_fault()
            if fault is not None:
                terminal_reason = fault
                break
            assert node.start is not None and node.last is not None
            if target is not None:
                target_pose = node.last_localization if args.localization_topic else node.last
                assert target_pose is not None
                target_error = math.hypot(
                    target_pose.x_m - float(target["x_m"]),
                    target_pose.y_m - float(target["y_m"]),
                )
                if target_error <= float(args.target_tolerance_m):
                    terminal_reason = "target_reached"
                    break
            distance = node.distance_m()
            if distance is not None and distance >= args.distance_m:
                terminal_reason = "distance_reached"
                break
            now = time.monotonic()
            if now >= next_publish:
                angular = 0.0
                if target is not None and args.heading_hold:
                    heading_pose = node.last_localization if args.localization_topic else node.last
                    assert heading_pose is not None
                    error, angular = heading_hold_command(
                        heading_pose.yaw_rad, float(target["yaw_rad"])
                    )
                    node.heading_hold_commands.append({
                        "monotonic_s": now,
                        "yaw_error_rad": error,
                        "angular_velocity_rad_s": angular,
                    })
                node.publish_velocity(0.2, angular)
                while next_publish <= now:
                    next_publish += PUBLISH_PERIOD_S
        node.post_stop_started_monotonic_s = time.monotonic()
        zero_deadline = node.post_stop_started_monotonic_s + ZERO_HOLD_S
        while rclpy.ok() and time.monotonic() < zero_deadline:
            node.publish_velocity(0.0)
            rclpy.spin_once(node, timeout_sec=PUBLISH_PERIOD_S)
        result.update({
            "start_pose_odom": node.start.__dict__ if node.start else None,
            "final_pose_odom": node.last.__dict__ if node.last else None,
            "measured_distance_m": node.distance_m(),
            "nonzero_publish_calls_returned": node.nonzero_publish_calls,
            "zero_publish_calls_returned": node.zero_publish_calls,
            "publish_rate": rate_summary(node.publish_times),
            "terminal_reason": terminal_reason,
            "physical_stop": node.physical_stop(),
            "final_target_error_m": (
                math.hypot(
                    (node.last_localization if args.localization_topic else node.last).x_m - float(target["x_m"]),
                    (node.last_localization if args.localization_topic else node.last).y_m - float(target["y_m"]),
                ) if target is not None and node.last is not None else None
            ),
            "final_target_yaw_error_rad": (
                wrap_angle(float(target["yaw_rad"]) - (node.last_localization if args.localization_topic else node.last).yaw_rad)
                if target is not None and node.last is not None else None
            ),
            "heading_hold_command_count": len(node.heading_hold_commands),
            "heading_hold_nonzero_command_count": sum(
                1 for item in node.heading_hold_commands if item["angular_velocity_rad_s"] != 0.0
            ),
        })
        reached = (
            terminal_reason == ("target_reached" if target is not None else "distance_reached")
            and node.distance_m() is not None
            and (
                result["final_target_error_m"] is not None
                and float(result["final_target_error_m"]) <= float(args.target_tolerance_m)
                if target is not None else node.distance_m() >= args.distance_m
            )
        )
        result["status"] = "PASS" if reached and result["physical_stop"]["status"] == "PASS" else "BLOCKED"
        if not reached:
            result["fault"] = terminal_reason
        elif result["physical_stop"]["status"] != "PASS":
            result["fault"] = "physical_stop_gate_failed"
        return 0 if reached else 2
    except Exception as exc:
        result["fault"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        result["nonzero_publish_calls_returned"] = node.nonzero_publish_calls
        result["zero_publish_calls_returned"] = node.zero_publish_calls
        result["publish_rate"] = rate_summary(node.publish_times)
        result["start_pose_odom"] = node.start.__dict__ if node.start else None
        result["final_pose_odom"] = node.last.__dict__ if node.last else None
        result["measured_distance_m"] = node.distance_m()
        result["final_target_error_m"] = (
            math.hypot(
                    (node.last_localization if args.localization_topic else node.last).x_m - float(target["x_m"]),
                    (node.last_localization if args.localization_topic else node.last).y_m - float(target["y_m"]),
            ) if target is not None and node.last is not None else None
        )
        result["final_target_yaw_error_rad"] = (
            wrap_angle(float(target["yaw_rad"]) - (node.last_localization if args.localization_topic else node.last).yaw_rad)
            if target is not None and node.last is not None else None
        )
        result["heading_hold_command_count"] = len(node.heading_hold_commands)
        result["heading_hold_nonzero_command_count"] = sum(
            1 for item in node.heading_hold_commands if item["angular_velocity_rad_s"] != 0.0
        )
        try:
            write_result(args.output, result)
        finally:
            node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
