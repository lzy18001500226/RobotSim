#!/usr/bin/env python3
"""Send exactly one AimDK SetMcAction request and preserve its complete response."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from typing import Any


def json_safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {name: json_safe(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def set_action_record(request: Any, response: Any, *, success_status: int,
                      submitted_utc: str, submitted_monotonic_ns: int,
                      returned_utc: str, returned_monotonic_ns: int) -> dict[str, Any]:
    """Keep the complete nested CommonResponse plus explicit decision fields."""
    common = response.response
    status_value = int(common.status.value)
    header_code = int(common.header.code)
    return {
        "transport_state": "RESPONSE_RECEIVED",
        "wall_utc_submitted": submitted_utc,
        "wall_monotonic_ns_submitted": submitted_monotonic_ns,
        "wall_utc_returned": returned_utc,
        "wall_monotonic_ns_returned": returned_monotonic_ns,
        "request": json_safe(request),
        "response": json_safe(common),
        "response_status_value": status_value,
        "response_header_code": header_code,
        "response_message": str(common.message),
        "service_success": status_value == int(success_status),
        "action_result_success": header_code == 0,
    }



def odom_pose(msg: Any) -> dict[str, float]:
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return {
        "x": float(p.x), "y": float(p.y), "z": float(p.z),
        "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),
        "linear_velocity": json_safe(msg.twist.twist.linear),
        "angular_velocity": json_safe(msg.twist.twist.angular),
    }


def assess_pre_action_state(
    latest: dict[str, dict[str, Any]],
    *,
    now_monotonic_ns: int,
    upright_odom_samples: int,
    service_ready: bool,
) -> dict[str, Any]:
    required = {"odom", "imu", "common", "leg_state", "leg_command"}
    ages = {
        key: (now_monotonic_ns - int(row["wall_monotonic_ns"])) / 1e9
        for key, row in latest.items()
    }
    fresh = required.issubset(latest) and all(0.0 <= ages[key] <= 0.5 for key in required)
    odom = latest.get("odom", {}).get("data", {})
    try:
        tilt = max(abs(float(odom["roll_deg"])), abs(float(odom["pitch_deg"])))
        pose_ok = float(odom["z"]) >= 0.52 and tilt < 15.0
    except (KeyError, TypeError, ValueError):
        pose_ok = False
    if not fresh:
        reason = "required_state_missing_or_stale"
    elif not pose_ok:
        reason = "live_posture_outside_existing_gate"
    elif upright_odom_samples < 3:
        reason = "upright_odom_samples_insufficient"
    elif not service_ready:
        reason = "SetMcAction_service_not_ready"
    else:
        reason = None
    return {
        "eligible": reason is None,
        "failure_reason": reason,
        "state_ages_s": ages,
        "fresh_required_topics": fresh,
        "posture_gate_pass": pose_ok,
        "upright_odom_samples": upright_odom_samples,
        "required_upright_odom_samples": 3,
        "service_ready": service_ready,
        "pose": odom if odom else None,
        "raw_common_state": latest.get("common", {}).get("data"),
        "imu": latest.get("imu", {}).get("data"),
        "leg_state": latest.get("leg_state", {}).get("data"),
        "leg_command": latest.get("leg_command", {}).get("data"),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True)
    parser.add_argument("--source", default="node.set_mc_action")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--precondition-timeout-s", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout_s <= 0 or args.timeout_s > 4.0:
        parser.error("--timeout-s must be in (0, 4]")
    if args.precondition_timeout_s <= 0 or args.precondition_timeout_s > 3.0:
        parser.error("--precondition-timeout-s must be in (0, 3]")

    import rclpy
    from rclpy.node import Node
    from aimdk_msgs.msg import (
        CommonState, JointCommandArray, JointStateArray, McActionCommand,
        McCommonState, RequestHeader,
    )
    from aimdk_msgs.srv import SetMcAction
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu

    class SetActionOnce(Node):
        def __init__(self) -> None:
            super().__init__("issue80_m1_set_action_once")
            self.client = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")
            self.latest: dict[str, dict[str, Any]] = {}
            self.upright_odom_samples = 0
            self.qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            subscriptions = (
                ("odom", "/aima/hal/odom/state", Odometry),
                ("imu", "/aima/hal/imu/torso/state", Imu),
                ("common", "/aima/mc/common/state", McCommonState),
                ("leg_state", "/aima/hal/joint/leg/state", JointStateArray),
                ("leg_command", "/aima/hal/joint/leg/command", JointCommandArray),
            )
            for key, topic, msg_type in subscriptions:
                self.create_subscription(
                    msg_type, topic, lambda msg, name=key: self.store(name, msg), self.qos
                )

        def store(self, key: str, msg: Any) -> None:
            now_ns = time.monotonic_ns()
            data = odom_pose(msg) if key == "odom" else json_safe(msg)
            self.latest[key] = {
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": now_ns,
                "data": data,
            }
            if key == "odom":
                try:
                    ok = float(data["z"]) >= 0.52 and max(
                        abs(float(data["roll_deg"])), abs(float(data["pitch_deg"]))
                    ) < 15.0
                except (KeyError, TypeError, ValueError):
                    ok = False
                self.upright_odom_samples = self.upright_odom_samples + 1 if ok else 0

        def pre_action_snapshot(self) -> dict[str, Any]:
            return assess_pre_action_state(
                self.latest,
                now_monotonic_ns=time.monotonic_ns(),
                upright_odom_samples=self.upright_odom_samples,
                service_ready=self.client.service_is_ready(),
            )

    rclpy.init()
    node = SetActionOnce()
    try:
        precondition_deadline = time.monotonic() + args.precondition_timeout_s
        precondition = node.pre_action_snapshot()
        while rclpy.ok() and time.monotonic() < precondition_deadline and not precondition["eligible"]:
            rclpy.spin_once(node, timeout_sec=0.005)
            precondition = node.pre_action_snapshot()
        if not precondition["eligible"]:
            state = (
                "PRECONDITION_UNSUITABLE_POSTURE"
                if precondition["fresh_required_topics"] and not precondition["posture_gate_pass"]
                else "PRECONDITION_STATE_OR_SERVICE_UNAVAILABLE"
            )
            print(json.dumps({
                "transport_state": state,
                "requested_action_desc": args.action,
                "request_source": args.source,
                "set_mc_action_sent": False,
                "pre_action_state": precondition,
                "response": None,
                "response_status_value": None,
                "response_header_code": None,
                "response_message": "SD was not sent because current live posture/telemetry was not eligible",
                "service_success": False,
                "action_result_success": False,
            }, separators=(",", ":")))
            return 2

        request = SetMcAction.Request()

        request.header = RequestHeader()
        request.header.stamp = node.get_clock().now().to_msg()
        request.source = args.source
        request.command = McActionCommand()
        request.command.action_desc = args.action
        submitted_utc = datetime.now(timezone.utc).isoformat()
        submitted_ns = time.monotonic_ns()
        future = node.client.call_async(request)  # Exactly one mutation request; never retry.
        deadline = time.monotonic() + args.timeout_s
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))
        returned_utc = datetime.now(timezone.utc).isoformat()
        returned_ns = time.monotonic_ns()
        if not future.done():
            print(json.dumps({
                "transport_state": "REQUEST_OUTCOME_UNKNOWN_TIMEOUT",
                "wall_utc_submitted": submitted_utc,
                "wall_monotonic_ns_submitted": submitted_ns,
                "wall_utc_returned": returned_utc,
                "wall_monotonic_ns_returned": returned_ns,
                "request": json_safe(request),
                "pre_action_state": precondition,
                "response": None,
                "response_status_value": None,
                "response_header_code": None,
                "response_message": "request was sent once but no service response arrived before timeout; do not retry",
                "service_success": False,
                "action_result_success": False,
            }, separators=(",", ":")))
            return 2

        response = future.result()
        if response is None:
            print(json.dumps({
                "transport_state": "EMPTY_RESPONSE",
                "wall_utc_submitted": submitted_utc,
                "wall_monotonic_ns_submitted": submitted_ns,
                "wall_utc_returned": returned_utc,
                "wall_monotonic_ns_returned": returned_ns,
                "request": json_safe(request),
                "pre_action_state": precondition,
                "response": None,
                "response_status_value": None,
                "response_header_code": None,
                "response_message": "service future completed without a response",
                "service_success": False,
                "action_result_success": False,
            }, separators=(",", ":")))
            return 2

        result = set_action_record(
            request,
            response,
            success_status=CommonState.SUCCESS,
            submitted_utc=submitted_utc,
            submitted_monotonic_ns=submitted_ns,
            returned_utc=returned_utc,
            returned_monotonic_ns=returned_ns,
        )
        result["pre_action_state"] = precondition
        result["set_mc_action_sent"] = True
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
        return 0 if result["service_success"] and result["action_result_success"] else 2
    except Exception as exc:
        print(json.dumps({
            "transport_state": "CLIENT_EXCEPTION",
            "requested_action_desc": args.action,
            "request_source": args.source,
            "response": None,
            "response_status_value": None,
            "response_header_code": None,
            "response_message": f"{type(exc).__name__}: {exc}",
            "service_success": False,
            "action_result_success": False,
        }, separators=(",", ":")))
        return 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
