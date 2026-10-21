#!/usr/bin/env python3
"""Pre-arm one ROS 2 observer/client and run the source-supported X2 stand handoff.

The process starts before the GUI Reset, subscribes to the same live state used
by the standing gate, and creates SetMcAction/GetMcAction clients before Reset.
After a verified Reset event it performs one PD -> JD -> SD sequence:

* request JOINT_DEFAULT once after the existing fresh upright pose gate;
* request STAND_DEFAULT once only after the MC reports JOINT_DEFAULT and its
  installed McBodyPoseStatus.STAND value with fresh, safe odometry;
* never retry a mutating request and never infer success from the service reply
  alone.

This script is simulation-only. It does not write robot state or alter models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TOPICS = frozenset({"odom", "imu", "common", "leg_state", "leg_command"})
ROOT_Z_MIN_M = 0.52
MAX_TILT_DEG = 15.0
MAX_STATE_AGE_S = 0.5
REQUIRED_UPRIGHT_ODOM_SAMPLES = 3
REQUIRED_STAND_COMMON_SAMPLES = 3


def write_complete_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one complete result atomically, without replacing prior evidence."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def odom_pose(msg: Any) -> dict[str, Any]:
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


def pose_is_upright(data: dict[str, Any]) -> bool:
    try:
        return (
            float(data["z"]) >= ROOT_Z_MIN_M
            and max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"]))) < MAX_TILT_DEG
        )
    except (KeyError, TypeError, ValueError):
        return False


def command_is_effective(data: dict[str, Any]) -> bool:
    """Whether any leg-command target/gain field is nonzero."""
    joints = data.get("joints", [])
    if not isinstance(joints, list):
        return False
    fields = ("position", "velocity", "effort", "stiffness", "damping")
    for joint in joints:
        if not isinstance(joint, dict):
            continue
        for field in fields:
            try:
                if abs(float(joint.get(field, 0.0) or 0.0)) > 1e-12:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def installed_body_stand_value(message_type: Any) -> int:
    """Read the STAND constant from the installed generated ROS message type."""
    instance = message_type()
    value = getattr(instance, "MC_BODY_POSE_STAND", None)
    if value is None:
        raise ValueError("installed McBodyPoseStatus has no MC_BODY_POSE_STAND constant")
    return int(value)


def assess_pre_action_state(
    latest: dict[str, dict[str, Any]],
    *,
    now_monotonic_ns: int,
    upright_odom_samples: int,
    service_ready: bool,
    reset_dispatch_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Apply the existing freshness/upright gate, optionally after Reset only."""
    post_reset = {
        key: row for key, row in latest.items()
        if reset_dispatch_monotonic_ns is None
        or int(row.get("wall_monotonic_ns", 0)) > reset_dispatch_monotonic_ns
    }
    ages = {
        key: (now_monotonic_ns - int(row["wall_monotonic_ns"])) / 1e9
        for key, row in post_reset.items()
    }
    fresh = REQUIRED_TOPICS.issubset(post_reset) and all(
        0.0 <= ages[key] <= MAX_STATE_AGE_S for key in REQUIRED_TOPICS
    )
    odom = post_reset.get("odom", {}).get("data", {})
    pose_ok = pose_is_upright(odom)
    if not fresh:
        reason = "required_post_reset_state_missing_or_stale"
    elif not pose_ok:
        reason = "live_posture_outside_existing_gate"
    elif upright_odom_samples < REQUIRED_UPRIGHT_ODOM_SAMPLES:
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
        "post_reset_required_topics": sorted(post_reset),
        "posture_gate_pass": pose_ok,
        "upright_odom_samples": upright_odom_samples,
        "required_upright_odom_samples": REQUIRED_UPRIGHT_ODOM_SAMPLES,
        "service_ready": service_ready,
        "pose": odom if odom else None,
        "raw_common_state": post_reset.get("common", {}).get("data"),
        "imu": post_reset.get("imu", {}).get("data"),
        "leg_state": post_reset.get("leg_state", {}).get("data"),
        "leg_command": post_reset.get("leg_command", {}).get("data"),
    }


def assess_joint_to_stand_state(
    latest: dict[str, dict[str, Any]],
    *,
    now_monotonic_ns: int,
    reset_dispatch_monotonic_ns: int,
    upright_odom_samples: int,
    stand_common_samples: int,
    set_service_ready: bool,
    get_service_ready: bool,
    get_mode_desc: str | None,
    get_mode_status: int | None,
    body_stand_value: int,
) -> dict[str, Any]:
    """Gate SD on fresh physical posture and the installed MC stand classifier."""
    base = assess_pre_action_state(
        latest,
        now_monotonic_ns=now_monotonic_ns,
        upright_odom_samples=upright_odom_samples,
        service_ready=set_service_ready and get_service_ready,
        reset_dispatch_monotonic_ns=reset_dispatch_monotonic_ns,
    )
    common = base.get("raw_common_state") or {}
    action = common.get("action_info", {}) if isinstance(common, dict) else {}
    body = common.get("body_status", {}) if isinstance(common, dict) else {}
    action_desc = action.get("action_desc")
    action_status = action.get("status", {}).get("value")
    body_status = body.get("value")
    body_is_stand = body_status == body_stand_value
    mode_is_joint = action_desc == "JOINT_DEFAULT" and action_status == 100
    get_readback_is_joint = get_mode_desc == "JOINT_DEFAULT" and get_mode_status == 100
    if not base["fresh_required_topics"]:
        reason = base["failure_reason"]
    elif not base["posture_gate_pass"]:
        reason = base["failure_reason"]
    elif not mode_is_joint or not get_readback_is_joint:
        reason = "JOINT_DEFAULT_not_confirmed_by_common_state_and_GetMcAction"
    elif not body_is_stand:
        reason = "MC_body_status_not_STAND"
    elif stand_common_samples < REQUIRED_STAND_COMMON_SAMPLES:
        reason = "fresh_STAND_body_status_samples_insufficient"
    elif not set_service_ready or not get_service_ready:
        reason = "required_MC_service_not_ready"
    else:
        reason = None
    return {
        "eligible": reason is None,
        "failure_reason": reason,
        "state_ages_s": base["state_ages_s"],
        "fresh_required_topics": base["fresh_required_topics"],
        "posture_gate_pass": base["posture_gate_pass"],
        "pose": base["pose"],
        "upright_odom_samples": upright_odom_samples,
        "stand_common_samples": stand_common_samples,
        "required_stand_common_samples": REQUIRED_STAND_COMMON_SAMPLES,
        "body_stand_value_from_installed_aimdk_msgs": body_stand_value,
        "raw_body_status": body_status,
        "raw_action_desc": action_desc,
        "raw_action_status": action_status,
        "GetMcAction_action_desc": get_mode_desc,
        "GetMcAction_status": get_mode_status,
        "set_service_ready": set_service_ready,
        "get_service_ready": get_service_ready,
        "raw_common_state": common,
        "imu": base["imu"],
        "leg_state": base["leg_state"],
        "leg_command": base["leg_command"],
    }


def set_action_record(
    request: Any,
    response: Any,
    *,
    success_status: int,
    submitted_utc: str,
    submitted_monotonic_ns: int,
    returned_utc: str,
    returned_monotonic_ns: int,
) -> dict[str, Any]:
    """Keep full nested CommonResponse and explicit status/code fields."""
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


def get_action_record(
    request: Any,
    response: Any,
    *,
    submitted_utc: str,
    submitted_monotonic_ns: int,
    returned_utc: str,
    returned_monotonic_ns: int,
) -> dict[str, Any]:
    info = response.info
    return {
        "transport_state": "RESPONSE_RECEIVED",
        "wall_utc_submitted": submitted_utc,
        "wall_monotonic_ns_submitted": submitted_monotonic_ns,
        "wall_utc_returned": returned_utc,
        "wall_monotonic_ns_returned": returned_monotonic_ns,
        "request": json_safe(request),
        "response": json_safe(response),
        "response_header_code": int(response.header.code),
        "action_desc": str(info.action_desc),
        "action_status_value": int(info.status.value),
    }


def elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return (int(end_ns) - int(start_ns)) / 1e6


def summarize_takeover_timing(events: dict[str, Any]) -> dict[str, Any]:
    """Compute wall-monotonic takeover intervals without mixing ROS time."""
    reset_ns = events.get("reset_dispatch_monotonic_ns")
    first_valid_ns = events.get("first_valid_post_reset_state_monotonic_ns")
    first_upright_ns = events.get("first_upright_post_reset_monotonic_ns")
    jd = events.get("joint_default", {}) or {}
    sd = events.get("stand_default", {}) or {}
    first_cmd_jd = events.get("first_effective_leg_command_after_jd_monotonic_ns")
    first_cmd_sd = events.get("first_effective_leg_command_after_sd_monotonic_ns")
    unsafe = events.get("first_unsafe_post_reset_monotonic_ns")
    return {
        "reset_to_first_valid_state_ms": elapsed_ms(reset_ns, first_valid_ns),
        "reset_to_first_upright_sample_ms": elapsed_ms(reset_ns, first_upright_ns),
        "reset_to_JD_request_ms": elapsed_ms(reset_ns, jd.get("wall_monotonic_ns_submitted")),
        "JD_request_to_response_ms": elapsed_ms(jd.get("wall_monotonic_ns_submitted"), jd.get("wall_monotonic_ns_returned")),
        "JD_response_to_first_effective_leg_command_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), first_cmd_jd),
        "JD_response_to_MC_STAND_classification_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), events.get("body_stand_ready_monotonic_ns")),
        "reset_to_SD_request_ms": elapsed_ms(reset_ns, sd.get("wall_monotonic_ns_submitted")),
        "SD_request_to_response_ms": elapsed_ms(sd.get("wall_monotonic_ns_submitted"), sd.get("wall_monotonic_ns_returned")),
        "SD_response_to_first_effective_leg_command_ms": elapsed_ms(sd.get("wall_monotonic_ns_returned"), first_cmd_sd),
        "reset_to_first_unsafe_posture_ms": elapsed_ms(reset_ns, unsafe),
        "JD_response_to_first_unsafe_posture_ms": elapsed_ms(jd.get("wall_monotonic_ns_returned"), unsafe),
        "SD_response_to_first_unsafe_posture_ms": elapsed_ms(sd.get("wall_monotonic_ns_returned"), unsafe),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="STAND_DEFAULT")
    parser.add_argument("--source", default="node.set_mc_action")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reset-event-file", required=True, type=Path)
    parser.add_argument("--set-timeout-s", type=float, default=2.0)
    parser.add_argument("--get-timeout-s", type=float, default=0.75)
    parser.add_argument("--ready-timeout-s", type=float, default=3.0)
    parser.add_argument("--reset-wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--upright-timeout-s", type=float, default=2.0)
    parser.add_argument("--joint-stand-timeout-s", type=float, default=4.0)
    parser.add_argument("--duration-s", type=float, default=8.0)
    args = parser.parse_args()
    limits = (
        (args.set_timeout_s, 0.0, 4.0, "--set-timeout-s"),
        (args.get_timeout_s, 0.0, 2.0, "--get-timeout-s"),
        (args.ready_timeout_s, 0.0, 5.0, "--ready-timeout-s"),
        (args.reset_wait_timeout_s, 0.0, 60.0, "--reset-wait-timeout-s"),
        (args.upright_timeout_s, 0.0, 5.0, "--upright-timeout-s"),
        (args.joint_stand_timeout_s, 0.0, 6.0, "--joint-stand-timeout-s"),
        (args.duration_s, 0.0, 30.0, "--duration-s"),
    )
    for value, lower, upper, name in limits:
        if not math.isfinite(value) or value <= lower or value > upper:
            parser.error(f"{name} must be in ({lower}, {upper}]")
    if args.action != "STAND_DEFAULT":
        parser.error("this capture only supports the reviewed PD -> JD -> SD standing handoff")

    import rclpy
    from rclpy.node import Node
    from aimdk_msgs.msg import (
        CommonRequest, CommonState, JointCommandArray, JointStateArray,
        McActionCommand, McBodyPoseStatus, McCommonState, RequestHeader,
    )
    from aimdk_msgs.srv import GetMcAction, SetMcAction
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Imu

    try:
        body_stand_value = installed_body_stand_value(McBodyPoseStatus)
    except (TypeError, ValueError, AttributeError) as exc:
        print(json.dumps({
            "status": "BLOCKED_INSTALLED_BODY_STATUS_CONSTANT_MISSING",
            "expected_constant": "aimdk_msgs.msg.McBodyPoseStatus.MC_BODY_POSE_STAND",
            "error": f"{type(exc).__name__}: {exc}",
        }, separators=(",", ":")), flush=True)
        return 2

    class TakeoverCapture(Node):
        def __init__(self) -> None:
            super().__init__("issue80_m1_standing_takeover_capture")
            self.output_dir = args.output_dir.resolve(strict=True)
            self.reset_event_file = args.reset_event_file.resolve(strict=False)
            self.trace_path = self.output_dir / "reset_state_trace.jsonl"
            self.ready_path = self.output_dir / "reset_watch_ready.json"
            self.summary_path = self.output_dir / "reset_state_summary.json"
            self.result_path = self.output_dir / "set_action_capture_result.json"
            self.trace = self.trace_path.open("x", encoding="utf-8")
            self.started_utc = datetime.now(timezone.utc).isoformat()
            self.started_ns = time.monotonic_ns()
            self.latest: dict[str, dict[str, Any]] = {}
            self.counts: Counter[str] = Counter()
            self.clocks: list[dict[str, Any]] = []
            self.reset_dispatch: dict[str, Any] | None = None
            self.reset_ns: int | None = None
            self.first_valid_state_ns: int | None = None
            self.first_upright_ns: int | None = None
            self.first_unsafe_ns: int | None = None
            self.first_unsafe_row: dict[str, Any] | None = None
            self.first_effective_after_jd: int | None = None
            self.first_effective_after_sd: int | None = None
            self.upright_samples = 0
            self.stand_samples = 0
            self.set_attempted: set[str] = set()
            self.set_records: dict[str, dict[str, Any]] = {}
            self.get_records: list[dict[str, Any]] = []
            self.get_mode_desc: str | None = None
            self.get_mode_status: int | None = None
            self.result: dict[str, Any] | None = None
            self.error: str | None = None
            self.observer_ready = False
            self.qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.set_client = self.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")
            self.get_client = self.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
            self.topic_specs = (
                ("odom", "/aima/hal/odom/state", Odometry),
                ("imu", "/aima/hal/imu/torso/state", Imu),
                ("common", "/aima/mc/common/state", McCommonState),
                ("leg_state", "/aima/hal/joint/leg/state", JointStateArray),
                ("leg_command", "/aima/hal/joint/leg/command", JointCommandArray),
            )
            for key, topic, msg_type in self.topic_specs:
                self.create_subscription(
                    msg_type, topic, lambda msg, name=key: self.store(name, msg), self.qos
                )
            self.create_subscription(Clock, "/clock", self.store_clock, self.qos)
            self.subscriptions_created_utc = datetime.now(timezone.utc).isoformat()
            self.subscriptions_created_ns = time.monotonic_ns()

        def write_trace(self, row: dict[str, Any]) -> None:
            self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            self.trace.flush()

        def store(self, key: str, msg: Any) -> None:
            now_utc = datetime.now(timezone.utc).isoformat()
            now_ns = time.monotonic_ns()
            data = odom_pose(msg) if key == "odom" else json_safe(msg)
            row = {
                "event": "state",
                "topic_key": key,
                "wall_utc": now_utc,
                "wall_monotonic_ns": now_ns,
                "header_stamp": json_safe(msg.header.stamp) if hasattr(msg, "header") else None,
                "data": data,
            }
            self.latest[key] = row
            self.counts[key] += 1
            self.write_trace(row)
            if self.reset_ns is not None and now_ns > self.reset_ns and key == "odom":
                if pose_is_upright(data):
                    self.upright_samples += 1
                    if self.first_upright_ns is None:
                        self.first_upright_ns = now_ns
                else:
                    self.upright_samples = 0
                    if self.first_upright_ns is not None and self.first_unsafe_ns is None:
                        self.first_unsafe_ns = now_ns
                        self.first_unsafe_row = row
                if self.first_valid_state_ns is None and self.fresh_topics_after_reset(now_ns):
                    self.first_valid_state_ns = now_ns
            if self.reset_ns is not None and now_ns > self.reset_ns and key == "common":
                common = data
                if (
                    common.get("action_info", {}).get("action_desc") == "JOINT_DEFAULT"
                    and int(common.get("action_info", {}).get("status", {}).get("value", -1)) == 100
                    and int(common.get("body_status", {}).get("value", -1)) == body_stand_value
                ):
                    self.stand_samples += 1
                else:
                    self.stand_samples = 0
            if key == "leg_command" and command_is_effective(data):
                if self.first_effective_after_jd is None:
                    response_ns = self.set_records.get("JOINT_DEFAULT", {}).get("wall_monotonic_ns_returned")
                    if response_ns is not None and now_ns > int(response_ns):
                        self.first_effective_after_jd = now_ns
                if self.first_effective_after_sd is None:
                    response_ns = self.set_records.get("STAND_DEFAULT", {}).get("wall_monotonic_ns_returned")
                    if response_ns is not None and now_ns > int(response_ns):
                        self.first_effective_after_sd = now_ns

        def store_clock(self, msg: Any) -> None:
            row = {
                "event": "clock",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                "clock": json_safe(msg.clock),
            }
            self.clocks.append(row)
            self.write_trace(row)

        def _post_reset_latest(self) -> dict[str, dict[str, Any]]:
            if self.reset_ns is None:
                return {}
            return {
                key: row for key, row in self.latest.items()
                if int(row.get("wall_monotonic_ns", 0)) > self.reset_ns
            }

        def fresh_topics_after_reset(self, now_ns: int) -> bool:
            current = self._post_reset_latest()
            return REQUIRED_TOPICS.issubset(current) and all(
                0.0 <= (now_ns - int(current[key]["wall_monotonic_ns"])) / 1e9 <= MAX_STATE_AGE_S
                for key in REQUIRED_TOPICS
            )

        def record_event(self, event_name: str, **payload: Any) -> None:
            self.write_trace({
                "event": event_name,
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                **payload,
            })

        def write_json_once(self, path: Path, value: dict[str, Any]) -> None:
            write_complete_json(path, value)

        def spin_for(self, timeout_s: float) -> None:
            rclpy.spin_once(self, timeout_sec=max(0.0, timeout_s))

        def mode_readback(self, timeout_s: float) -> dict[str, Any]:
            request = GetMcAction.Request()
            request.request = CommonRequest()
            request.request.header.stamp = self.get_clock().now().to_msg()
            submitted_utc = datetime.now(timezone.utc).isoformat()
            submitted_ns = time.monotonic_ns()
            future = self.get_client.call_async(request)
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                self.spin_for(min(0.01, max(0.0, deadline - time.monotonic())))
            returned_utc = datetime.now(timezone.utc).isoformat()
            returned_ns = time.monotonic_ns()
            if not future.done() or future.result() is None:
                record = {
                    "transport_state": "GET_MC_ACTION_TIMEOUT",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "response": None,
                    "action_desc": None,
                    "action_status_value": None,
                }
            else:
                record = get_action_record(
                    request,
                    future.result(),
                    submitted_utc=submitted_utc,
                    submitted_monotonic_ns=submitted_ns,
                    returned_utc=returned_utc,
                    returned_monotonic_ns=returned_ns,
                )
                self.get_mode_desc = record["action_desc"]
                self.get_mode_status = record["action_status_value"]
            self.get_records.append(record)
            self.record_event("get_mc_action_readback", record=record)
            return record

        def set_action_once(self, action_desc: str, timeout_s: float, pre_action: dict[str, Any]) -> dict[str, Any]:
            if action_desc in self.set_attempted:
                raise RuntimeError(f"mutating SetMcAction retry blocked for {action_desc}")
            if action_desc not in {"JOINT_DEFAULT", "STAND_DEFAULT"}:
                raise ValueError(f"unsupported standing takeover stage: {action_desc}")
            self.set_attempted.add(action_desc)
            request = SetMcAction.Request()
            request.header = RequestHeader()
            request.header.stamp = self.get_clock().now().to_msg()
            request.source = args.source
            request.command = McActionCommand()
            request.command.action_desc = action_desc
            submitted_utc = datetime.now(timezone.utc).isoformat()
            submitted_ns = time.monotonic_ns()
            self.record_event(
                "set_mc_action_request",
                action_desc=action_desc,
                submitted_wall_utc=submitted_utc,
                submitted_monotonic_ns=submitted_ns,
                pre_action_state=pre_action,
            )
            future = self.set_client.call_async(request)
            deadline = time.monotonic() + timeout_s
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                self.spin_for(min(0.01, max(0.0, deadline - time.monotonic())))
                if self.first_unsafe_ns is not None and self.first_unsafe_ns > submitted_ns:
                    record = {
                        "transport_state": "REQUEST_OUTCOME_UNKNOWN_POSTURE_UNSAFE",
                        "wall_utc_submitted": submitted_utc,
                        "wall_monotonic_ns_submitted": submitted_ns,
                        "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns_returned": time.monotonic_ns(),
                        "request": json_safe(request),
                        "pre_action_state": pre_action,
                        "response": None,
                        "response_status_value": None,
                        "response_header_code": None,
                        "response_message": "posture crossed the existing gate while waiting for this single request; outcome is unknown and it will not be retried",
                        "service_success": False,
                        "action_result_success": False,
                        "requested_action_desc": action_desc,
                    }
                    self.set_records[action_desc] = record
                    self.record_event("set_mc_action_response", action_desc=action_desc, record=record)
                    return record
            returned_utc = datetime.now(timezone.utc).isoformat()
            returned_ns = time.monotonic_ns()
            if not future.done():
                record = {
                    "transport_state": "REQUEST_OUTCOME_UNKNOWN_TIMEOUT",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "pre_action_state": pre_action,
                    "response": None,
                    "response_status_value": None,
                    "response_header_code": None,
                    "response_message": "request was sent once; no retry is permitted",
                    "service_success": False,
                    "action_result_success": False,
                }
            elif future.result() is None:
                record = {
                    "transport_state": "EMPTY_RESPONSE",
                    "wall_utc_submitted": submitted_utc,
                    "wall_monotonic_ns_submitted": submitted_ns,
                    "wall_utc_returned": returned_utc,
                    "wall_monotonic_ns_returned": returned_ns,
                    "request": json_safe(request),
                    "pre_action_state": pre_action,
                    "response": None,
                    "response_status_value": None,
                    "response_header_code": None,
                    "response_message": "service future completed without a response",
                    "service_success": False,
                    "action_result_success": False,
                }
            else:
                record = set_action_record(
                    request,
                    future.result(),
                    success_status=CommonState.SUCCESS,
                    submitted_utc=submitted_utc,
                    submitted_monotonic_ns=submitted_ns,
                    returned_utc=returned_utc,
                    returned_monotonic_ns=returned_ns,
                )
                record["pre_action_state"] = pre_action
            record["requested_action_desc"] = action_desc
            self.set_records[action_desc] = record
            self.record_event("set_mc_action_response", action_desc=action_desc, record=record)
            return record

        def wait_reset_event(self) -> dict[str, Any] | None:
            deadline = time.monotonic() + args.reset_wait_timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                if not self.reset_event_file.exists():
                    continue
                try:
                    payload = json.loads(self.reset_event_file.read_text(encoding="utf-8"))
                    event = payload.get("event", {})
                    dispatch_ns = int(event.get("dispatch_monotonic_ns", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if event.get("status") != "KEY_DISPATCHED" or dispatch_ns <= 0:
                    self.error = "Reset event file did not contain a verified KEY_DISPATCHED event"
                    return None
                self.reset_dispatch = payload
                self.reset_ns = dispatch_ns
                self.record_event("reset_barrier_observed", reset_dispatch=payload)
                return payload
            self.error = "verified Reset event file was not observed before timeout"
            return None

        def wait_pre_reset_ready(self) -> bool:
            deadline = time.monotonic() + args.ready_timeout_s
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                now_ns = time.monotonic_ns()
                fresh = [
                    key for key, row in self.latest.items()
                    if key in REQUIRED_TOPICS and 0.0 <= (now_ns - int(row["wall_monotonic_ns"])) / 1e9 <= MAX_STATE_AGE_S
                ]
                if len(fresh) != len(REQUIRED_TOPICS):
                    continue
                set_ready = bool(self.set_client.service_is_ready())
                get_ready = bool(self.get_client.service_is_ready())
                if not set_ready or not get_ready:
                    continue
                pre_mode = self.mode_readback(args.get_timeout_s)
                if pre_mode.get("transport_state") != "RESPONSE_RECEIVED":
                    self.error = "GetMcAction pre-Reset readback failed"
                elif pre_mode.get("action_desc") != "PASSIVE_DEFAULT" or pre_mode.get("action_status_value") != 100:
                    self.error = (
                        "unexpected pre-Reset mode/readiness: "
                        f"action={pre_mode.get('action_desc')!r} status={pre_mode.get('action_status_value')!r}"
                    )
                else:
                    self.observer_ready = True
                ready = {
                    "schema": "robotsim.issue80.takeover-watch-ready.v1",
                    "status": "READY" if self.observer_ready else "BLOCKED_PRE_RESET_MODE_OR_SERVICE",
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_monotonic_ns": time.monotonic_ns(),
                    "node_started_wall_utc": self.started_utc,
                    "node_started_monotonic_ns": self.started_ns,
                    "subscriptions_created_wall_utc": self.subscriptions_created_utc,
                    "subscriptions_created_monotonic_ns": self.subscriptions_created_ns,
                    "fresh_topics": sorted(fresh),
                    "set_mc_action_service_ready": set_ready,
                    "get_mc_action_service_ready": get_ready,
                    "pre_reset_get_mc_action": pre_mode,
                    "raw_common_state": self.latest.get("common", {}).get("data"),
                    "error": self.error,
                }
                self.write_json_once(self.ready_path, ready)
                self.record_event("takeover_client_ready", readiness=ready)
                return self.observer_ready
            ready = {
                "schema": "robotsim.issue80.takeover-watch-ready.v1",
                "status": "BLOCKED_PRE_RESET_READINESS_TIMEOUT",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_ns": time.monotonic_ns(),
                "fresh_topics": sorted(self.latest),
                "set_mc_action_service_ready": bool(self.set_client.service_is_ready()),
                "get_mc_action_service_ready": bool(self.get_client.service_is_ready()),
                "error": "required state streams and both MC clients did not become ready before Reset",
            }
            self.write_json_once(self.ready_path, ready)
            self.error = ready["error"]
            return False

        def wait_for_post_reset_upright(self) -> dict[str, Any] | None:
            assert self.reset_ns is not None
            deadline = min(
                self.reset_ns / 1e9 + args.upright_timeout_s,
                time.monotonic() + args.upright_timeout_s,
            )
            last: dict[str, Any] = {"status": "POST_RESET_STATE_NOT_READY"}
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                now_ns = time.monotonic_ns()
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed the existing gate after a transient upright pose and before JD"
                    return {
                        "status": "POST_RESET_UNSAFE_BEFORE_JD",
                        "reset_dispatch_monotonic_ns": self.reset_ns,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                        "first_unsafe_monotonic_ns": self.first_unsafe_ns,
                        "error": self.error,
                    }
                latest = self._post_reset_latest()
                gate = assess_pre_action_state(
                    latest,
                    now_monotonic_ns=now_ns,
                    upright_odom_samples=self.upright_samples,
                    service_ready=self.set_client.service_is_ready() and self.get_client.service_is_ready(),
                    reset_dispatch_monotonic_ns=self.reset_ns,
                )
                common = latest.get("common", {}).get("data", {})
                action = common.get("action_info", {})
                mode = action.get("action_desc")
                last = {
                    "status": "POST_RESET_UPRIGHT_NOT_YET_CONFIRMED",
                    "reset_dispatch_monotonic_ns": self.reset_ns,
                    "snapshot": gate,
                    "raw_action_desc": mode,
                    "raw_action_status": action.get("status", {}).get("value"),
                }
                if gate["eligible"] and mode == "PASSIVE_DEFAULT" and action.get("status", {}).get("value") == 100:
                    last.update({
                        "status": "POST_RESET_UPRIGHT_OBSERVED",
                        "wall_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_monotonic_ns": now_ns,
                        "reset_to_upright_gate_ms": elapsed_ms(self.reset_ns, now_ns),
                        "raw_common_state": common,
                    })
                    self.record_event("post_reset_upright_gate", snapshot=last)
                    return last
            self.error = "no eligible post-Reset upright state under the unchanged height/tilt/freshness gate"
            last.update({"status": "POST_RESET_UPRIGHT_TIMEOUT", "timeout_s": args.upright_timeout_s})
            return last

        def confirm_mode(self, expected: str, timeout_s: float) -> dict[str, Any] | None:
            deadline = time.monotonic() + timeout_s
            last: dict[str, Any] | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                if self.first_unsafe_ns is not None:
                    return last
                last = self.mode_readback(args.get_timeout_s)
                if last.get("transport_state") == "RESPONSE_RECEIVED" and last.get("action_desc") == expected and last.get("action_status_value") == 100:
                    return last
                if self.first_unsafe_ns is not None:
                    return last
                self.spin_for(0.05)
            return last

        def wait_for_joint_stand(self, timeout_s: float) -> dict[str, Any] | None:
            assert self.reset_ns is not None
            deadline = time.monotonic() + timeout_s
            last: dict[str, Any] | None = None
            while rclpy.ok() and time.monotonic() < deadline:
                self.spin_for(0.005)
                now_ns = time.monotonic_ns()
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed existing height/tilt gate after the first upright post-Reset pose"
                    return {
                        "eligible": False,
                        "failure_reason": self.error,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                    }
                latest = self._post_reset_latest()
                last_common = latest.get("common", {}).get("data", {})
                common_action = last_common.get("action_info", {}).get("action_desc")
                common_status = last_common.get("action_info", {}).get("status", {}).get("value")
                readback_desc = self.get_mode_desc
                readback_status = self.get_mode_status
                if self.stand_samples >= REQUIRED_STAND_COMMON_SAMPLES:
                    last_get_ns = int(self.get_records[-1].get("wall_monotonic_ns_returned", 0)) if self.get_records else 0
                    if now_ns - last_get_ns > 150_000_000:
                        readback = self.mode_readback(args.get_timeout_s)
                        readback_desc = readback.get("action_desc")
                        readback_status = readback.get("action_status_value")
                if self.first_unsafe_ns is not None:
                    self.error = "posture crossed the existing gate while confirming the JOINT_DEFAULT stand state"
                    return {
                        "eligible": False,
                        "failure_reason": self.error,
                        "first_unsafe_wall_utc": self.first_unsafe_row.get("wall_utc") if self.first_unsafe_row else None,
                    }
                last = assess_joint_to_stand_state(
                    latest,
                    now_monotonic_ns=now_ns,
                    reset_dispatch_monotonic_ns=self.reset_ns,
                    upright_odom_samples=self.upright_samples,
                    stand_common_samples=self.stand_samples,
                    set_service_ready=self.set_client.service_is_ready(),
                    get_service_ready=self.get_client.service_is_ready(),
                    get_mode_desc=readback_desc,
                    get_mode_status=readback_status,
                    body_stand_value=body_stand_value,
                )
                if last["eligible"]:
                    last["wall_utc"] = datetime.now(timezone.utc).isoformat()
                    last["wall_monotonic_ns"] = now_ns
                    return last
            self.error = "JOINT_DEFAULT did not produce fresh upright MC STAND status within its bounded window"
            if last is None:
                last = {"eligible": False, "failure_reason": self.error}
            else:
                last["eligible"] = False
                last["failure_reason"] = self.error
            return last

        def _write_final_result(self, status: str, *, block_stage: str | None = None, error: str | None = None,
                                pre_reset: dict[str, Any] | None = None,
                                upright_gate: dict[str, Any] | None = None,
                                stand_gate: dict[str, Any] | None = None) -> None:
            if self.result is not None:
                return
            jd = self.set_records.get("JOINT_DEFAULT")
            sd = self.set_records.get("STAND_DEFAULT")
            timing_events = {
                "reset_dispatch_monotonic_ns": self.reset_ns,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "joint_default": jd,
                "stand_default": sd,
                "body_stand_ready_monotonic_ns": (stand_gate or {}).get("wall_monotonic_ns"),
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "first_unsafe_post_reset_monotonic_ns": self.first_unsafe_ns,
            }
            self.result = {
                "schema": "robotsim.issue80.standing-takeover-capture.v1",
                "status": status,
                "blocking_stage": block_stage,
                "error": error,
                "source": args.source,
                "pre_reset_get_mc_action": pre_reset,
                "reset_dispatch": self.reset_dispatch,
                "post_reset_upright_gate": upright_gate,
                "joint_default_set_mc_action": jd,
                "joint_default_get_mc_action_readbacks": [r for r in self.get_records if r.get("wall_monotonic_ns_submitted", 0) > (self.reset_ns or 0) and r.get("action_desc") == "JOINT_DEFAULT"],
                "joint_to_stand_gate": stand_gate,
                "stand_default_set_mc_action": sd,
                "get_mc_action_readbacks": list(self.get_records),
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "first_unsafe_post_reset": self.first_unsafe_row,
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "timing_ms": summarize_takeover_timing(timing_events),
                "limitations": [
                    "ROS graph exposes no foot-contact or ground-reaction-force topic in the captured runtime inventory.",
                    "MC body_status=STAND is retained as the installed AimDK classifier; it is not a direct force-plate measurement.",
                    "Header stamps are kept raw; all elapsed timing uses host monotonic receipt/submission timestamps.",
                ],
            }
            self.write_json_once(self.result_path, self.result)
            self.record_event("takeover_result", result=self.result)

        def run_takeover(self) -> int:
            pre_reset_mode: dict[str, Any] | None = None
            if not self.wait_pre_reset_ready():
                self._write_final_result("BLOCKED_PRE_RESET_READINESS", block_stage="PRE_RESET", error=self.error)
                return 2
            readiness = json.loads(self.ready_path.read_text(encoding="utf-8"))
            pre_reset_mode = readiness.get("pre_reset_get_mc_action")
            event_doc = self.wait_reset_event()
            if event_doc is None:
                self._write_final_result("BLOCKED_RESET_EVENT", block_stage="RESET", error=self.error, pre_reset=pre_reset_mode)
                return 2
            upright_gate = self.wait_for_post_reset_upright()
            if upright_gate is None or upright_gate.get("status") != "POST_RESET_UPRIGHT_OBSERVED":
                self._write_final_result("BLOCKED_POST_RESET_UPRIGHT", block_stage="POST_RESET_UPRIGHT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            jd_pre = upright_gate["snapshot"]
            jd = self.set_action_once("JOINT_DEFAULT", args.set_timeout_s, jd_pre)
            if jd.get("transport_state") != "RESPONSE_RECEIVED" or jd.get("service_success") is not True or jd.get("action_result_success") is not True:
                self.error = "JOINT_DEFAULT SetMcAction did not return a successful status/code pair"
                self._write_final_result("BLOCKED_JOINT_DEFAULT_REJECTED", block_stage="JOINT_DEFAULT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            jd_readback = self.confirm_mode("JOINT_DEFAULT", 1.5)
            if jd_readback is None or jd_readback.get("action_desc") != "JOINT_DEFAULT" or jd_readback.get("action_status_value") != 100:
                self.error = "GetMcAction did not confirm JOINT_DEFAULT/status=100 after the single JD request"
                self._write_final_result("BLOCKED_JOINT_DEFAULT_READBACK", block_stage="JOINT_DEFAULT_READBACK", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate)
                return 2
            stand_gate = self.wait_for_joint_stand(args.joint_stand_timeout_s)
            if not stand_gate.get("eligible"):
                self._write_final_result("BLOCKED_STAND_PRECONDITION", block_stage="BODY_STAND_GATE", error=str(stand_gate.get("failure_reason")), pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            sd = self.set_action_once("STAND_DEFAULT", args.set_timeout_s, stand_gate)
            if sd.get("transport_state") != "RESPONSE_RECEIVED" or sd.get("service_success") is not True or sd.get("action_result_success") is not True:
                self.error = "STAND_DEFAULT SetMcAction did not return a successful status/code pair"
                self._write_final_result("BLOCKED_STAND_DEFAULT_REJECTED", block_stage="STAND_DEFAULT", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            sd_readback = self.confirm_mode("STAND_DEFAULT", 2.0)
            if sd_readback is None or sd_readback.get("action_desc") != "STAND_DEFAULT" or sd_readback.get("action_status_value") != 100:
                self.error = "GetMcAction did not confirm STAND_DEFAULT/status=100 after the single SD request"
                self._write_final_result("BLOCKED_STAND_DEFAULT_READBACK", block_stage="STAND_DEFAULT_READBACK", error=self.error, pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
                return 2
            self._write_final_result("TAKEOVER_SEQUENCE_COMPLETE", pre_reset=pre_reset_mode, upright_gate=upright_gate, stand_gate=stand_gate)
            return 0

        def capture(self) -> dict[str, Any]:
            status = "TRACE_CAPTURED"
            start = time.monotonic()
            # Always continue the read-only trace through the bounded Reset window.
            # The parent independently stops/cleans the whole container at first fall.
            while rclpy.ok() and self.reset_ns is not None and time.monotonic_ns() < self.reset_ns + int(args.duration_s * 1e9):
                self.spin_for(0.005)
            if self.reset_ns is None:
                status = "BLOCKED_NO_RESET_EVENT"
            self.trace.flush()
            odom_rows = []
            for line in self.trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "state" and row.get("topic_key") == "odom":
                    odom_rows.append(row)
            post_odom = [r for r in odom_rows if self.reset_ns is not None and int(r.get("wall_monotonic_ns", 0)) > self.reset_ns]
            z = [float(r.get("data", {}).get("z")) for r in post_odom if isinstance(r.get("data", {}).get("z"), (int, float))]
            jd = self.set_records.get("JOINT_DEFAULT", {})
            sd = self.set_records.get("STAND_DEFAULT", {})
            stand_gate = (self.result or {}).get("joint_to_stand_gate") or {}
            timing_events = {
                "reset_dispatch_monotonic_ns": self.reset_ns,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "joint_default": jd,
                "stand_default": sd,
                "body_stand_ready_monotonic_ns": stand_gate.get("wall_monotonic_ns"),
                "first_effective_leg_command_after_jd_monotonic_ns": self.first_effective_after_jd,
                "first_effective_leg_command_after_sd_monotonic_ns": self.first_effective_after_sd,
                "first_unsafe_post_reset_monotonic_ns": self.first_unsafe_ns,
            }
            summary = {
                "schema": "robotsim.issue80.reset-state-trace.v1",
                "observer_status": status,
                "takeover_status": (self.result or {}).get("status", "NOT_COMPLETED"),
                "duration_limit_s": args.duration_s,
                "wall_elapsed_s": time.monotonic() - start,
                "sample_counts": dict(self.counts),
                "clock_messages": len(self.clocks),
                "clock_topic_exposed": bool(self.clocks),
                "odom_sample_count": len(post_odom),
                "odom_z_min_m": min(z) if z else None,
                "odom_z_max_m": max(z) if z else None,
                "odom_first": post_odom[0].get("data") if post_odom else None,
                "odom_last": post_odom[-1].get("data") if post_odom else None,
                "first_odom_wall_utc": post_odom[0].get("wall_utc") if post_odom else None,
                "last_odom_wall_utc": post_odom[-1].get("wall_utc") if post_odom else None,
                "first_valid_post_reset_state_monotonic_ns": self.first_valid_state_ns,
                "first_upright_post_reset_monotonic_ns": self.first_upright_ns,
                "first_unsafe_post_reset": self.first_unsafe_row,
                "takeover_result": self.result,
                "timing_ms": summarize_takeover_timing(timing_events),
                "limitations": [
                    "Header timestamps are recorded as fields; elapsed intervals use host monotonic time.",
                    "This observer does not write qpos, joint commands, or contact forces.",
                    "The captured ROS graph has no dedicated foot-contact/ground-reaction-force topic.",
                ],
            }
            self.write_json_once(self.summary_path, summary)
            self.record_event("takeover_trace_complete", summary=summary)
            return summary

    rclpy.init()
    node = TakeoverCapture()
    try:
        action_rc = node.run_takeover()
        node.capture()
        print(json.dumps(node.result or {"status": "NO_TAKEOVER_RESULT", "error": node.error}, separators=(",", ":"), allow_nan=False), flush=True)
        return action_rc
    except Exception as exc:
        node.error = f"{type(exc).__name__}: {exc}"
        if node.result is None:
            node._write_final_result("BLOCKED_CLIENT_EXCEPTION", block_stage="CLIENT", error=node.error)
        try:
            if not node.summary_path.exists():
                node.capture()
        except Exception:
            pass
        print(json.dumps({"status": "BLOCKED_CLIENT_EXCEPTION", "error": node.error}, separators=(",", ":"), allow_nan=False), flush=True)
        return 2
    finally:
        node.trace.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
