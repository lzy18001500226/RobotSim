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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


PUBLISH_PERIOD_S = 0.02
ZERO_HOLD_S = 1.0
ODOM_FRESHNESS_S = 0.5
MIN_ROOT_Z_M = 0.52
MAX_TILT_RAD = math.radians(15.0)


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
    source_stamp_ns: int
    received_monotonic_s: float


class DirectTruthRoute(Node):
    """Reuse the vendor ADD + 50 Hz contract; truth only chooses the stop time."""

    def __init__(self, target_distance_m: float, trace_path: Path) -> None:
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
        self.publish_times: list[float] = []
        self.nonzero_publish_calls = 0
        self.zero_publish_calls = 0
        self.create_subscription(Odometry, "/aima/hal/odom/state", self._odom, self.odom_qos)

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
            source_stamp_ns=(int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)),
            received_monotonic_s=time.monotonic(),
        )
        if self.start is None:
            self.start = sample
        self.last = sample
        self._trace({"event": "odom", **sample.__dict__})

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

    def publish_velocity(self, forward_m_s: float) -> None:
        message = McLocomotionVelocity()
        message.header = MessageHeader()
        message.header.stamp = self.get_clock().now().to_msg()
        message.source = "node"
        message.forward_velocity = forward_m_s
        message.lateral_velocity = 0.0
        message.angular_velocity = 0.0
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
            "angular_velocity_rad_s": 0.0,
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
        x_m, y_m = float(target["x_m"]), float(target["y_m"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid truth target {name!r}: {exc}") from exc
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise ValueError(f"truth target {name!r} has nonfinite XY")
    return {"name": name, "x_m": x_m, "y_m": y_m}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--target-route-config", type=Path, default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--target-tolerance-m", type=float, default=None)
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
    target = load_truth_target(args.target_route_config, str(args.target_name)) if target_requested else None
    rclpy.init()
    node = DirectTruthRoute(args.distance_m, args.trace)
    result: dict[str, Any] = {
        "schema": "robotsim.issue80.vendor-direct-truth-route.v1",
        "classification": (
            "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET"
            if target is not None else "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_DISTANCE_BASELINE"
        ),
        "status": "BLOCKED",
        "target_distance_m": args.distance_m,
        "target_pose_odom": target,
        "target_tolerance_m": args.target_tolerance_m,
        "final_target_error_m": None,
        "command": {"source": "node", "forward_m_s": 0.2, "publish_hz": 50.0},
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
        while rclpy.ok() and node.last is None and time.monotonic() < initial_odom_deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
        if node.last is None:
            result["fault"] = "initial_odometry_unavailable"
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
                target_error = math.hypot(
                    node.last.x_m - float(target["x_m"]),
                    node.last.y_m - float(target["y_m"]),
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
                node.publish_velocity(0.2)
                while next_publish <= now:
                    next_publish += PUBLISH_PERIOD_S
        zero_deadline = time.monotonic() + ZERO_HOLD_S
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
            "final_target_error_m": (
                math.hypot(
                    node.last.x_m - float(target["x_m"]),
                    node.last.y_m - float(target["y_m"]),
                ) if target is not None and node.last is not None else None
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
        result["status"] = "PASS" if reached else "BLOCKED"
        if not reached:
            result["fault"] = terminal_reason
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
                node.last.x_m - float(target["x_m"]),
                node.last.y_m - float(target["y_m"]),
            ) if target is not None and node.last is not None else None
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
