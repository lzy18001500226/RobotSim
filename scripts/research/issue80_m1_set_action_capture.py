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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True)
    parser.add_argument("--source", default="node.set_mc_action")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout_s <= 0 or args.timeout_s > 4.0:
        parser.error("--timeout-s must be in (0, 4]")

    import rclpy
    from rclpy.node import Node
    from aimdk_msgs.msg import CommonState, McActionCommand, RequestHeader
    from aimdk_msgs.srv import SetMcAction

    class SetActionOnce(Node):
        def __init__(self) -> None:
            super().__init__("issue80_m1_set_action_once")
            self.client = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")

    rclpy.init()
    node = SetActionOnce()
    try:
        if not node.client.wait_for_service(timeout_sec=min(args.timeout_s, 2.0)):
            print(json.dumps({
                "transport_state": "SERVICE_UNAVAILABLE",
                "requested_action_desc": args.action,
                "request_source": args.source,
                "response": None,
                "response_status_value": None,
                "response_header_code": None,
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
