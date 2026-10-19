#!/usr/bin/env python3
"""Bounded, simulation-only X2 Issue #80 source and LD diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from aimdk_msgs.msg import (
    CommonRequest,
    JointCommandArray,
    JointStateArray,
    McInputAction,
    McActionCommand,
    McCommonState,
    McLocomotionVelocity,
    MessageHeader,
    RequestHeader,
)
from aimdk_msgs.srv import (
    GetCurrentInputSource,
    GetMcAction,
    SetMcAction,
    SetMcInputSource,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from issue80_m1_mode_logic import classify_ld_transition


VELOCITY_TOPIC = "/aima/mc/locomotion/velocity"
INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
CURRENT_INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/GetCurrentInputSource"
GET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"
MODE_LD = "LOCOMOTION_DEFAULT"
REQUIRED_TOPICS = ("odom", "imu", "common", "leg_state", "leg_command")


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def roll_pitch_from_quaternion(q: Any) -> tuple[float, float]:
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    roll = math.atan2(sinr, cosr)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _json_safe(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.get_fields_and_field_types()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


class FocusedM1Diagnostic(Node):
    """One-ADD, zero-only LD diagnosis with an optional guarded 150 ms pulse."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("issue80_m1_focused_diagnostic")
        if not re.fullmatch(r"issue80_m1_[A-Za-z0-9_]{8,40}", args.source):
            raise ValueError("source must be a fresh issue80_m1_<unique-suffix> name")
        if not 0.1 <= args.zero_duration <= 1.0:
            raise ValueError("zero-only interval must be within 0.1..1.0 s")
        if not 0.1 <= args.mode_observe_duration <= 5.0:
            raise ValueError("mode observation interval is capped at 5 seconds")
        if not 0.1 <= args.state_timeout <= 0.5:
            raise ValueError("state freshness timeout may not exceed the established 0.5 s gate")
        if args.max_tilt_deg > 15.0 or args.min_root_z < 0.52:
            raise ValueError("safety guards may be tightened but not weakened")
        self.args = args
        self.source = args.source
        self.output_dir = Path(args.output_dir).resolve(strict=True)
        self.trace_path = self.output_dir / "m1_diagnostic_trace.jsonl"
        self.result_path = self.output_dir / "m1_diagnostic_result.json"
        if self.trace_path.exists() or self.result_path.exists():
            raise FileExistsError("refusing to overwrite prior M1 diagnostic evidence")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        self.latest: dict[str, Any] = {}
        self.received_at: dict[str, float] = {}
        self.interrupted = False
        self.fault: str | None = None
        self.nonzero_messages = 0
        self.source_observations: list[dict[str, Any]] = []
        self.mode_observations: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            McLocomotionVelocity, VELOCITY_TOPIC, self.state_qos
        )
        self.source_client = self.create_client(
            GetCurrentInputSource, CURRENT_INPUT_SOURCE_SERVICE
        )
        self.register_client = self.create_client(SetMcInputSource, INPUT_SOURCE_SERVICE)
        self.set_mode_client = self.create_client(
            SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction"
        )
        self.get_mode_client = self.create_client(GetMcAction, GET_MC_ACTION_SERVICE)
        self.create_subscription(
            Odometry, "/aima/hal/odom/state", lambda msg: self._store("odom", msg), self.state_qos
        )
        self.create_subscription(
            Imu, "/aima/hal/imu/torso/state", lambda msg: self._store("imu", msg), self.state_qos
        )
        self.create_subscription(
            McCommonState,
            "/aima/mc/common/state",
            lambda msg: self._store("common", msg),
            self.state_qos,
        )
        self.create_subscription(
            JointStateArray,
            "/aima/hal/joint/leg/state",
            lambda msg: self._store("leg_state", msg),
            self.state_qos,
        )
        self.create_subscription(
            JointCommandArray,
            "/aima/hal/joint/leg/command",
            lambda msg: self._store("leg_command", msg),
            self.state_qos,
        )

    def _store(self, key: str, msg: Any) -> None:
        self.latest[key] = msg
        self.received_at[key] = time.monotonic()

    def _mode(self) -> str:
        common = self.latest.get("common")
        return common.action_info.action_desc if common is not None else ""

    def _pose(self) -> dict[str, float] | None:
        odom = self.latest.get("odom")
        if odom is None:
            return None
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        roll, pitch = roll_pitch_from_quaternion(q)
        return {
            "x": float(p.x), "y": float(p.y), "z": float(p.z),
            "roll": roll, "pitch": pitch, "yaw": yaw_from_quaternion(q),
            "vx": float(odom.twist.twist.linear.x),
            "vy": float(odom.twist.twist.linear.y),
            "vz": float(odom.twist.twist.linear.z),
            "stamp_sec": int(odom.header.stamp.sec),
            "stamp_nanosec": int(odom.header.stamp.nanosec),
        }

    def _state_ages(self) -> dict[str, float | None]:
        now = time.monotonic()
        return {
            key: now - self.received_at[key] if key in self.received_at else None
            for key in REQUIRED_TOPICS
        }

    def _safety_fault(self, *, expected_mode: str | tuple[str, ...]) -> str | None:
        ages = self._state_ages()
        stale = [k for k, age in ages.items() if age is None or age > self.args.state_timeout]
        if stale:
            return "stale_or_missing_state:" + ",".join(stale)
        allowed_modes = (expected_mode,) if isinstance(expected_mode, str) else expected_mode
        if self._mode() not in allowed_modes:
            return f"MC_mode_regressed_or_changed:{self._mode()}"
        pose = self._pose()
        if pose is None or not all(
            math.isfinite(pose[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ):
            return "missing_or_nonfinite_odom"
        if pose["z"] < self.args.min_root_z:
            return "root_height_below_safety_floor"
        if max(abs(pose["roll"]), abs(pose["pitch"])) > math.radians(self.args.max_tilt_deg):
            return "tilt_safety_stop"
        for joint in self.latest["leg_state"].joints:
            if not all(math.isfinite(float(v)) for v in (joint.position, joint.velocity)):
                return "nonfinite_leg_state"
        return None

    def _snapshot(self) -> dict[str, Any]:
        common = self.latest.get("common")
        return {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_sec": time.monotonic(),
            "mode": self._mode(),
            "pose_odom": self._pose(),
            "state_age_sec": self._state_ages(),
            "active_source_common_state": _json_safe(common.input_source) if common else None,
            "common_action_info": _json_safe(common.action_info) if common else None,
            "fsm_state": int(common.fsm_state.current_state) if common else None,
            "body_status": int(common.body_status.value) if common else None,
            "leg_state": _json_safe(self.latest.get("leg_state")),
            "leg_command": _json_safe(self.latest.get("leg_command")),
            "imu": _json_safe(self.latest.get("imu")),
        }

    def _trace(self, phase: str, command: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        row = {
            "phase": phase,
            "source": self.source,
            "command": {
                "forward_m_s": command[0],
                "lateral_m_s": command[1],
                "yaw_rate_rad_s": command[2],
            },
            **self._snapshot(),
            "fault": self.fault,
        }
        self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def _publish(self, forward: float = 0.0) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source
        msg.forward_velocity = float(forward)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = 0.0
        self.publisher.publish(msg)
        if forward != 0.0:
            self.nonzero_messages += 1

    def _wait_service(self, client: Any, label: str, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if client.wait_for_service(timeout_sec=0.2):
                return
            rclpy.spin_once(self, timeout_sec=0.01)
        raise TimeoutError(f"{label} service unavailable")

    def _call(self, client: Any, request: Any, label: str, timeout: float = 1.0,
              heartbeat_mode: str | tuple[str, ...] | None = None) -> Any:
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        next_heartbeat = time.monotonic()
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self.interrupted:
                self._publish()
                self._trace(f"INTERRUPTED_ZERO:{label}")
                raise RuntimeError("operator_signal_stop")
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.monotonic()
            if heartbeat_mode is not None and now >= next_heartbeat:
                fault = self._safety_fault(expected_mode=heartbeat_mode)
                if fault:
                    self.fault = fault
                    self._publish()
                else:
                    self._publish()
                self._trace(f"RPC_HEARTBEAT:{label}")
                next_heartbeat = now + 0.05
        if not future.done() or future.result() is None:
            raise TimeoutError(f"{label} service call timed out")
        return future.result()

    def read_input_source(self, label: str, *, heartbeat_mode: str | None = None) -> dict[str, Any]:
        self._wait_service(self.source_client, CURRENT_INPUT_SOURCE_SERVICE)
        request = GetCurrentInputSource.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        response = self._call(
            self.source_client, request, f"GetCurrentInputSource:{label}",
            heartbeat_mode=heartbeat_mode,
        )
        result = {
            "label": label,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "response": _json_safe(response),
            "response_header_code": int(response.response.header.code),
            "response_state": int(response.response.state.value),
            "task_id": int(response.response.task_id),
            "input_source": _json_safe(response.input_source),
        }
        self.source_observations.append(result)
        self.events.append({"event": "get_current_input_source", **result})
        return result

    def read_mode(self, label: str, *, heartbeat_mode: str | None = None) -> dict[str, Any]:
        self._wait_service(self.get_mode_client, GET_MC_ACTION_SERVICE)
        request = GetMcAction.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        response = self._call(
            self.get_mode_client, request, f"GetMcAction:{label}",
            heartbeat_mode=heartbeat_mode,
        )
        result = {
            "label": label,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "response": _json_safe(response),
            "response_header_code": int(response.header.code),
            "current_action_value": int(response.info.current_action.value),
            "action_desc": response.info.action_desc,
            "status_value": int(response.info.status.value),
        }
        self.mode_observations.append(result)
        self.events.append({"event": "get_mc_action", **result})
        return result

    def register_add_once(self) -> dict[str, Any]:
        self._wait_service(self.register_client, INPUT_SOURCE_SERVICE)
        request = SetMcInputSource.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        request.action = McInputAction()
        request.action.value = 1001  # ADD. ENABLE is intentionally not issued.
        request.input_source.name = self.source
        request.input_source.priority = self.args.priority
        request.input_source.timeout = self.args.source_timeout_ms
        response = self._call(self.register_client, request, "SetMcInputSource:ADD")
        result = {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "response": _json_safe(response),
            "response_header_code": int(response.response.header.code),
            "response_state": int(response.response.state.value),
            "task_id": int(response.response.task_id),
            "accepted": int(response.response.header.code) == 0,
        }
        self.events.append({"event": "set_mc_input_source_add", **result})
        return result

    def wait_for_standing(self) -> None:
        deadline = time.monotonic() + self.args.ready_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.interrupted:
                raise RuntimeError("operator_signal_stop")
            rclpy.spin_once(self, timeout_sec=0.05)
            fault = self._safety_fault(expected_mode="STAND_DEFAULT")
            if fault is None:
                common = self.latest["common"]
                if int(common.body_status.value) == 1:
                    return
        raise TimeoutError("fresh upright STAND_DEFAULT telemetry not observed; reset is required")

    def check_single_publisher(self) -> list[str]:
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and time.monotonic() < deadline:
            names = [i.node_name for i in self.get_publishers_info_by_topic(VELOCITY_TOPIC)]
            if names == [self.get_name()]:
                return names
            rclpy.spin_once(self, timeout_sec=0.05)
        names = [i.node_name for i in self.get_publishers_info_by_topic(VELOCITY_TOPIC)]
        raise RuntimeError(f"velocity topic must have exactly this one publisher; found {names}")

    def zero_phase(self, duration: float, *, expected_mode: str) -> None:
        end = time.monotonic() + duration
        next_publish = time.monotonic()
        while rclpy.ok() and time.monotonic() < end:
            if self.interrupted:
                self.fault = "operator_signal_stop"
            fault = self.fault or self._safety_fault(expected_mode=expected_mode)
            if fault:
                self.fault = fault
                self._publish()
                self._trace("SAFE_STOP_ZERO")
                raise RuntimeError(fault)
            now = time.monotonic()
            if now >= next_publish:
                self._publish()
                self._trace(f"ZERO_ONLY:{expected_mode}")
                next_publish = now + 0.05
            rclpy.spin_once(self, timeout_sec=0.01)

    def request_ld_once(self) -> dict[str, Any]:
        self._wait_service(self.set_mode_client, "/aimdk_5Fmsgs/srv/SetMcAction")
        before = self._mode()
        request = SetMcAction.Request()
        request.header = RequestHeader()
        request.header.stamp = self.get_clock().now().to_msg()
        # The first Issue 80 attempt matched the pinned Nova-Valley example's
        # empty source and the installed MC explicitly rejected it. This
        # bounded follow-up uses only the source just registered by this
        # diagnostic; it still publishes zero until a live LD readback.
        request.source = self.args.set_action_source
        request.command = McActionCommand()
        request.command.action_desc = MODE_LD
        submitted_at = datetime.now(timezone.utc).isoformat()
        response = self._call(
            self.set_mode_client, request, "SetMcAction:LOCOMOTION_DEFAULT",
            timeout=3.0, heartbeat_mode=("STAND_DEFAULT", MODE_LD),
        )
        result = {
            "wall_utc_submitted": submitted_at,
            "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "request_source": request.source,
            "requested_action_desc": request.command.action_desc,
            "response": _json_safe(response.response),
            "response_status_value": int(response.response.status.value),
            "response_message": response.response.message,
            "mode_before": before,
        }
        self.events.append({"event": "set_mc_action_ld", **result})
        return result

    def poll_mode_readback(self, duration: float) -> None:
        end = time.monotonic() + duration
        next_query = time.monotonic()
        next_zero = time.monotonic()
        expected_modes = {"STAND_DEFAULT", MODE_LD}
        while rclpy.ok() and time.monotonic() < end:
            if self.interrupted:
                self.fault = "operator_signal_stop"
            common_mode = self._mode()
            fault = self.fault or self._safety_fault(expected_mode=common_mode)
            if fault or common_mode not in expected_modes:
                self.fault = fault or f"unexpected_mode_during_diagnosis:{common_mode}"
                self._publish()
                self._trace("SAFE_STOP_ZERO")
                break
            now = time.monotonic()
            if now >= next_zero:
                self._publish()
                self._trace("MODE_DIAG_ZERO_ONLY")
                next_zero = now + 0.05
            if now >= next_query:
                try:
                    self.read_mode(
                        f"ld_poll_{len(self.mode_observations)}",
                        heartbeat_mode=("STAND_DEFAULT", MODE_LD),
                    )
                except Exception as exc:
                    self.events.append({"event": "get_mc_action_error", "error": str(exc)})
                    break
                if self._mode() == MODE_LD:
                    break
                next_query = time.monotonic() + 0.5
            rclpy.spin_once(self, timeout_sec=0.01)

    def _classify(self, set_result: dict[str, Any] | None) -> str:
        return classify_ld_transition(
            set_result,
            self.mode_observations,
            self._mode(),
            readback_error=any(event["event"] == "get_mc_action_error" for event in self.events),
        )

    def activation_pulse(self) -> dict[str, Any]:
        # Pulse authority is deliberately fixed and cannot be increased by CLI input.
        if self._mode() != MODE_LD:
            return {"status": "NOT_RUN", "reason": "GetMcAction/McCommonState did not confirm LD"}
        if self.fault:
            return {"status": "BLOCKED", "reason": f"prior safety fault: {self.fault}"}
        self.check_single_publisher()
        fault = self._safety_fault(expected_mode=MODE_LD)
        if fault:
            return {"status": "BLOCKED", "reason": fault}
        bounds = self.latest["common"].speed_status.forward_bounds
        if not float(bounds.min_value) <= 0.2 <= float(bounds.max_value):
            return {
                "status": "BLOCKED",
                "reason": "0.2 m/s activation speed is outside live MC bounds",
                "live_forward_bounds": _json_safe(bounds),
            }
        pre_source = self.read_input_source("immediately_before_activation_pulse")
        pre_name = pre_source["input_source"].get("name", "")
        common_name = self.latest["common"].input_source.name
        if pre_source["response_header_code"] != 0:
            return {"status": "BLOCKED", "reason": "GetCurrentInputSource pre-pulse read failed", "readback": pre_source}
        if pre_name not in ("", self.source) or common_name not in ("", self.source):
            return {
                "status": "BLOCKED", "reason": "another source currently owns arbitration",
                "get_current_input_source_before": pre_source,
                "common_source_before": common_name,
            }
        start = self._pose()
        start_yaw = start["yaw"] if start else None
        started = time.monotonic()
        end = started + 0.15
        next_publish = started
        own_source_seen = False
        first_command = True
        pulse_fault = None
        while rclpy.ok() and time.monotonic() < end:
            if self.interrupted:
                pulse_fault = "operator_signal_stop"
            pulse_fault = pulse_fault or self._safety_fault(expected_mode=MODE_LD)
            common_name = self.latest["common"].input_source.name if "common" in self.latest else ""
            if common_name not in ("", self.source):
                pulse_fault = pulse_fault or f"active_source_mismatch:{common_name!r}"
            elapsed = time.monotonic() - started
            if not first_command and not own_source_seen and elapsed >= 0.05:
                if common_name == self.source:
                    own_source_seen = True
                else:
                    pulse_fault = pulse_fault or f"source_not_selected_after_first_activation_tick:{common_name!r}"
            if own_source_seen and common_name != self.source:
                pulse_fault = pulse_fault or f"active_source_mismatch:{common_name!r}"
            if pulse_fault:
                self.fault = pulse_fault
                self._publish()
                self._trace("PULSE_ABORT_ZERO")
                break
            now = time.monotonic()
            if now >= next_publish:
                self._publish(0.2)
                if not first_command and common_name == self.source:
                    own_source_seen = True
                self._trace("ACTIVATION_PULSE", (0.2, 0.0, 0.0))
                first_command = False
                next_publish = now + 0.05
            rclpy.spin_once(self, timeout_sec=0.01)
            # Allow the first command one 50 ms arbitration interval, then fail closed.
            if not first_command and not own_source_seen:
                common_name = self.latest["common"].input_source.name if "common" in self.latest else ""
                if common_name == self.source:
                    own_source_seen = True
                elif time.monotonic() - started >= 0.05:
                    pulse_fault = f"source_not_selected_after_first_activation_tick:{common_name!r}"
                    self.fault = pulse_fault
                    self._publish()
                    self._trace("PULSE_ABORT_ZERO")
                    break
        actual_duration = min(time.monotonic() - started, 0.15)
        end_pose = self._pose()
        projection = None
        if start and end_pose:
            dx, dy = end_pose["x"] - start["x"], end_pose["y"] - start["y"]
            projection = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
        common_during_pulse = _json_safe(self.latest.get("common"))
        self._repeat_zero()
        post_source = self.read_input_source("after_activation_pulse")
        post_common = self.latest.get("common")
        if post_source["input_source"].get("name", "") not in ("", self.source):
            pulse_fault = pulse_fault or "GetCurrentInputSource reports another winner after pulse"
        if post_source["response_header_code"] != 0:
            pulse_fault = pulse_fault or "GetCurrentInputSource post-pulse read failed"
        if post_common and post_common.input_source.name not in ("", self.source):
            pulse_fault = pulse_fault or "McCommonState reports another winner after pulse"
        return {
            "status": "ABORTED" if pulse_fault else "EXECUTED",
            "fault": pulse_fault,
            "command_forward_m_s": 0.2,
            "command_yaw_rate_rad_s": 0.0,
            "command_duration_limit_s": 0.15,
            "actual_duration_s": actual_duration,
            "nonzero_messages": self.nonzero_messages,
            "get_current_input_source_before": pre_source,
            "get_current_input_source_after": post_source,
            "common_state_during_pulse": common_during_pulse,
            "common_source_after": _json_safe(post_common.input_source) if post_common else None,
            "start_pose_odom": start,
            "end_pose_odom": end_pose,
            "forward_projection_m": projection,
            "source_selected_during_pulse": own_source_seen,
        }

    def _repeat_zero(self) -> None:
        for _ in range(10):
            self._publish()
            self._trace("REPEATED_SAFE_ZERO")
            time.sleep(0.01)

    def execute(self) -> dict[str, Any]:
        before = after_add = after_zero = before_mode = set_ld = None
        registration = None
        pulse: dict[str, Any] = {"status": "NOT_RUN"}
        publishers: list[str] = []
        try:
            self.wait_for_standing()
            publishers = self.check_single_publisher()
            before = self.read_input_source("before_registration")
            if before["response_header_code"] != 0:
                raise RuntimeError("GetCurrentInputSource before ADD returned a nonzero header.code")
            before_mode = self.read_mode("before_registration")
            if before_mode["response_header_code"] != 0 or before_mode["action_desc"] != "STAND_DEFAULT":
                raise RuntimeError("GetMcAction did not confirm STAND_DEFAULT before ADD")
            registration = self.register_add_once()
            after_add = self.read_input_source("after_add")
            if after_add["response_header_code"] != 0:
                raise RuntimeError("GetCurrentInputSource after ADD returned a nonzero header.code")
            if not registration["accepted"]:
                raise RuntimeError("SetMcInputSource ADD response header.code != 0")
            self.zero_phase(self.args.zero_duration, expected_mode="STAND_DEFAULT")
            after_zero = self.read_input_source("after_zero_only", heartbeat_mode="STAND_DEFAULT")
            if after_zero["response_header_code"] != 0:
                raise RuntimeError("GetCurrentInputSource after zero-only returned a nonzero header.code")
            after_zero_mode = self.read_mode("after_zero_only", heartbeat_mode="STAND_DEFAULT")
            if after_zero_mode["response_header_code"] != 0:
                raise RuntimeError("GetMcAction after zero-only returned a nonzero header.code")
            set_ld = self.request_ld_once()
            self.poll_mode_readback(self.args.mode_observe_duration)
            after_ld_source = self.read_input_source("after_ld_readback", heartbeat_mode=self._mode())
            if after_ld_source["response_header_code"] != 0:
                raise RuntimeError("GetCurrentInputSource after LD readback returned a nonzero header.code")
            classification = self._classify(set_ld)
            if classification == "LD_CONFIRMED" and self.args.activation_pulse:
                pulse = self.activation_pulse()
            self._repeat_zero()
        except Exception as exc:
            self.fault = str(exc)
            self._repeat_zero()
            classification = "DIAGNOSTIC_ABORTED"
        finally:
            if self.result_path.exists():
                raise FileExistsError("refusing to overwrite diagnostic result")
            result = {
                "schema": "robotsim.issue80.m1.focused-diagnostic.v1",
                "simulation_only": True,
                "status": "PASS" if classification == "LD_CONFIRMED" and pulse.get("status") == "EXECUTED" and not pulse.get("fault") and not self.fault else "BLOCKED",
                "classification": classification,
                "source": self.source,
                "set_action_source": self.args.set_action_source,
                "source_registration": registration,
                "get_current_input_source": self.source_observations,
                "get_mc_action": self.mode_observations,
                "set_mc_action_ld": set_ld,
                "velocity_publishers": publishers,
                "zero_only_duration_s": self.args.zero_duration,
                "activation_pulse": pulse,
                "nonzero_velocity_messages_total": self.nonzero_messages,
                "final_common_state": _json_safe(self.latest.get("common")),
                "final_pose_odom": self._pose(),
                "fault": self.fault,
                "limitations": [
                    "ADD registration, source arbitration, mode transition, and physical motion are separate gates.",
                    "An empty active source after zero-only messages is recorded but is not interpreted as registration failure.",
                    "No nonzero velocity is published unless both mode readbacks confirm LOCOMOTION_DEFAULT.",
                    "The only permitted motion is one forward-only 0.2 m/s pulse capped at 150 ms; no yaw command is sent.",
                    "Simulation-only gate; no real-robot operation or M2/M3 navigation was attempted.",
                ],
            }
            with self.result_path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        return result

    def close(self) -> None:
        try:
            self._repeat_zero()
        finally:
            self.trace.close()


def _focused_diagnostic_command(args: argparse.Namespace) -> int:
    rclpy.init()
    node = FocusedM1Diagnostic(args)

    def request_stop(signum, frame):
        node.interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = node.execute()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    diagnose = subparsers.add_parser(
        "diagnose", help="one ADD, read-only arbitration checks, zero-only LD diagnosis, optional 150 ms pulse"
    )
    diagnose.add_argument("--source", required=True)
    diagnose.add_argument(
        "--set-action-source",
        default="",
        help="SetMcAction.source value; empty matches the pinned example, while a registered source is a bounded runtime-parity diagnostic",
    )
    diagnose.add_argument("--output-dir", required=True)
    diagnose.add_argument("--priority", type=int, default=40)
    diagnose.add_argument("--source-timeout-ms", type=int, default=1000)
    diagnose.add_argument("--state-timeout", type=float, default=0.5)
    diagnose.add_argument("--ready-timeout", type=float, default=10.0)
    diagnose.add_argument("--mode-observe-duration", type=float, default=5.0)
    diagnose.add_argument("--max-tilt-deg", type=float, default=15.0)
    diagnose.add_argument("--min-root-z", type=float, default=0.52)
    diagnose.add_argument("--zero-duration", type=float, default=0.8)
    diagnose.add_argument("--activation-pulse", action="store_true")
    args = parser.parse_args()
    try:
        return _focused_diagnostic_command(args)
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
