#!/usr/bin/env python3
"""Bounded, simulation-only X2 M1 first-motion probe. Requires explicit maintainer authorization."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import rclpy
from aimdk_msgs.msg import (
    CommonRequest, JointCommandArray, JointStateArray, McActionCommand,
    McCommonState, McInputAction, McLocomotionVelocity, MessageHeader, RequestHeader,
)
from aimdk_msgs.srv import GetCurrentInputSource, GetMcAction, SetMcAction, SetMcInputSource
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME, COMMAND_PUBLISHER_TOPIC, COMMAND_QOS_DEPTH, COMMAND_QOS_DURABILITY,
    BASELINE_DURATION_S, BASELINE_SAMPLE_PERIOD_S, BASELINE_WINDOW_COUNT,
    COMMAND_QOS_RELIABILITY, MC_SIM_CONFIG_PATH, MODE_LOCOMOTION, MODE_STAND, PUBLISH_HZ,
    MC_FSM_ACTIVE_STATES, MC_FSM_SAFE, MC_FSM_STABLE, verify_mc_fsm_runtime_contract,
    SOURCE_PRIORITY, SOURCE_TIMEOUT_MS, VELOCITY_SOURCE_NAME, classify_source_registration,
    capture_dds_qos_override_environment, derive_stationary_odom_baseline,
    load_leg_joint_limits, load_mc_input_source_timeouts,
    VerifiedMcRuntimeIdentity, load_verified_mc_runtime_identity,
    load_verified_mc_velocity_subscriber_provenance, mc_subscriber_history_depth_evidence,
    run_first_motion_window, snapshot_fault,
    validate_local_sim_publishers, validate_first_motion_authorization,
    validate_runtime_scope,
)
from issue80_m1_mode_logic import classify_set_mc_action_response

VELOCITY_TOPIC = COMMAND_PUBLISHER_TOPIC
INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
CURRENT_INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/GetCurrentInputSource"
GET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"
SET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/SetMcAction"
REQUIRED_TOPICS = ("odom", "imu", "common", "leg_state", "leg_command")
ROBOT_MODEL_PATH = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/"
    "lx2501_3_t2d5/sim/robot_model.yaml"
)
MC_SIM_CONFIG_PATH = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/"
    "lx2501_3_t2d5/mc_sim.yaml"
)


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
        return {name: _json_safe(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


class FocusedM1Diagnostic(Node):
    """Run one authorized 0.2 m/s first-motion test with fail-closed monitoring."""

    def __init__(self, args: argparse.Namespace):
        verified_network_mode = os.environ.get("ROBOTSIM_VERIFIED_NETWORK_MODE")
        validate_first_motion_authorization(
            reset_gate_verified=args.reset_gate_verified,
            maintainer_authorized=args.maintainer_authorized,
            network_mode=verified_network_mode,
        )
        if not 0.1 <= args.state_timeout <= 0.5:
            raise ValueError("state freshness limit may not be weakened beyond the existing 0.5 s gate")
        if args.max_tilt_deg <= 0.0 or args.max_tilt_deg > 15.0:
            raise ValueError("tilt safety threshold may be tightened but not weakened beyond 15 degrees")
        if args.min_root_z < 0.52:
            raise ValueError("root-height safety threshold may be tightened but not weakened below 0.52 m")
        self.fsm_runtime_contract = verify_mc_fsm_runtime_contract(
            "/workspace/raicom2026/mc/lib"
        )
        super().__init__("issue80_m1_first_motion")
        self.args = args
        self.verified_network_mode = verified_network_mode
        self.source = VELOCITY_SOURCE_NAME
        self.output_dir = Path(args.output_dir).resolve(strict=True)
        self.trace_path = self.output_dir / "m1_first_motion_trace.jsonl"
        self.result_path = self.output_dir / "m1_first_motion_result.json"
        if self.trace_path.exists() or self.result_path.exists():
            raise FileExistsError("refusing to overwrite prior M1 evidence")
        self.trace = self.trace_path.open("x", encoding="utf-8", buffering=1)
        self.joint_limits, self.joint_limits_sha256 = load_leg_joint_limits(args.robot_model_path)
        self.mc_input_source_timeouts, self.mc_sim_config_sha256 = load_mc_input_source_timeouts(
            args.mc_sim_config_path
        )
        self.latest: dict[str, Any] = {}
        self.received_at: dict[str, float] = {}
        self.last_mode: dict[str, Any] | None = None
        self.last_source: dict[str, Any] | None = None
        self.readback_at: dict[str, float] = {}
        self.mode_future = None
        self.source_future = None
        self.mode_future_request = None
        self.source_future_request = None
        self.mode_future_started = 0.0
        self.source_future_started = 0.0
        self.next_mode_query = 0.0
        self.next_source_query = 0.0
        self.async_readback_fault: str | None = None
        self.interrupted = threading.Event()
        self.publisher_lock = threading.RLock()
        self.trace_lock = threading.RLock()
        self.ros_spin_lock = threading.RLock()
        self.nonzero_publish_calls_returned = 0
        self.zero_publish_calls_returned = 0
        self.command_trace: list[dict[str, Any]] = []
        self.current_phase = "preflight"
        self.events: list[dict[str, Any]] = []
        self.mode_observations: list[dict[str, Any]] = []
        self.source_observations: list[dict[str, Any]] = []
        self.mc_runtime_identity: VerifiedMcRuntimeIdentity | None = None
        self.scope_evidence: dict[str, Any] | None = None
        self.pulse_result: dict[str, Any] | None = None

        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=COMMAND_QOS_DEPTH,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Snapshot the requested profile before creation, then read it back
        # through the created rclpy Publisher API during the transport gate.
        self.publisher_configured_qos = self._qos_profile(self.command_qos)
        self.publisher = self.create_publisher(McLocomotionVelocity, VELOCITY_TOPIC, self.command_qos)
        self.source_client = self.create_client(GetCurrentInputSource, CURRENT_INPUT_SOURCE_SERVICE)
        self.register_client = self.create_client(SetMcInputSource, INPUT_SOURCE_SERVICE)
        self.set_mode_client = self.create_client(SetMcAction, SET_MC_ACTION_SERVICE)
        self.get_mode_client = self.create_client(GetMcAction, GET_MC_ACTION_SERVICE)
        self.create_subscription(
            Odometry, "/aima/hal/odom/state", lambda msg: self._store("odom", msg), self.state_qos
        )
        self.create_subscription(
            Imu, "/aima/hal/imu/torso/state", lambda msg: self._store("imu", msg), self.state_qos
        )
        self.create_subscription(
            McCommonState, "/aima/mc/common/state", lambda msg: self._store("common", msg), self.state_qos
        )
        self.create_subscription(
            JointStateArray, "/aima/hal/joint/leg/state", lambda msg: self._store("leg_state", msg), self.state_qos
        )
        self.create_subscription(
            JointCommandArray, "/aima/hal/joint/leg/command", lambda msg: self._store("leg_command", msg), self.state_qos
        )

    def _store(self, key: str, msg: Any) -> None:
        self.latest[key] = msg
        self.received_at[key] = time.monotonic()

    def _mode(self) -> str:
        common = self.latest.get("common")
        return str(common.action_info.action_desc) if common is not None else ""

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
        ages = {
            key: now - self.received_at[key] if key in self.received_at else None
            for key in REQUIRED_TOPICS
        }
        for key in ("get_mode", "get_source"):
            ages[key] = now - self.readback_at[key] if key in self.readback_at else None
        return ages

    def _joint_states_as_dicts(self) -> list[dict[str, Any]]:
        state = self.latest.get("leg_state")
        if state is None:
            return []
        return [
            {
                "name": str(joint.name), "position": float(joint.position),
                "velocity": float(joint.velocity), "effort": float(joint.effort),
                "error_code": int(joint.error_code),
            }
            for joint in state.joints
        ]

    def _joint_state_fault(self) -> str | None:
        joints = self._joint_states_as_dicts()
        if len(joints) != len(self.joint_limits):
            return "leg_joint_limit_mapping_incomplete"
        by_name = {joint["name"]: joint for joint in joints}
        if len(by_name) != len(joints) or set(by_name) != set(self.joint_limits):
            return "leg_joint_limit_mapping_incomplete"
        for name, joint in by_name.items():
            limits = self.joint_limits[name]
            if not all(math.isfinite(float(joint[k])) for k in ("position", "velocity")):
                return f"nonfinite_leg_joint_state:{name}"
            if not float(limits["lower_position"]) <= joint["position"] <= float(limits["upper_position"]):
                return f"source_joint_position_limit:{name}"
            if abs(joint["velocity"]) > float(limits["max_velocity"]):
                return f"source_joint_velocity_limit:{name}"
        return None

    def _safety_fault(self, *, expected_mode: str | tuple[str, ...]) -> str | None:
        ages = self._state_ages()
        stale = [key for key in REQUIRED_TOPICS if ages.get(key) is None or ages[key] > self.args.state_timeout]
        if stale:
            return "stale_or_missing_state:" + ",".join(stale)
        allowed = (expected_mode,) if isinstance(expected_mode, str) else expected_mode
        if self._mode() not in allowed:
            return f"MC_mode_changed:{self._mode()}"
        common = self.latest.get("common")
        if common is None or int(common.body_status.value) != 1:
            return "non_standing_body_status"
        fsm = int(common.fsm_state.current_state)
        if fsm == MC_FSM_SAFE:
            return "mc_safety_protection_state"
        if fsm not in MC_FSM_ACTIVE_STATES:
            return f"unknown_or_unrelated_fsm_state:{fsm}"
        pose = self._pose()
        if pose is None or not all(math.isfinite(pose[key]) for key in ("x", "y", "z", "roll", "pitch", "yaw")):
            return "missing_or_nonfinite_odom"
        if pose["z"] < self.args.min_root_z:
            return "fall_root_height"
        if max(abs(pose["roll"]), abs(pose["pitch"])) > math.radians(self.args.max_tilt_deg):
            return "fall_tilt"
        return self._joint_state_fault()

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

    def _trace(self, phase: str, snapshot: Mapping[str, Any] | None = None) -> None:
        row = {
            "event": "state_sample",
            "phase": phase,
            "source": self.source,
            "snapshot": _json_safe(snapshot if snapshot is not None else self._snapshot()),
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_sec": time.monotonic(),
        }
        self._write_trace_row(row)

    def _write_trace_row(self, row: Mapping[str, Any]) -> None:
        with self.trace_lock:
            self.trace.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def _publish(self, forward: float = 0.0) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source
        msg.forward_velocity = float(forward)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = 0.0
        with self.publisher_lock:
            command_row = {
                "event": "velocity_command",
                "command_kind": "zero_stop" if forward == 0.0 else "nonzero_activation",
                "publish_outcome": "CALL_STARTED",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_sec": time.monotonic(),
                "ros_stamp_sec": int(msg.header.stamp.sec),
                "ros_stamp_nanosec": int(msg.header.stamp.nanosec),
                "phase": self.current_phase,
                "source": str(msg.source),
                "forward_velocity_m_s": float(msg.forward_velocity),
                "lateral_velocity_m_s": float(msg.lateral_velocity),
                "angular_velocity_rad_s": float(msg.angular_velocity),
            }
            self.command_trace.append(command_row)
            self._write_trace_row(command_row)
            try:
                self.publisher.publish(msg)
            except Exception as exc:
                command_row["publish_outcome"] = "RAISED"
                command_row["publish_error"] = f"{type(exc).__name__}: {exc}"
                self._write_trace_row({
                    "event": "velocity_publish_result",
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_monotonic_sec": time.monotonic(),
                    "source": str(msg.source),
                    "publish_outcome": "RAISED",
                    "error": command_row["publish_error"],
                })
                raise
            command_row["publish_outcome"] = "CALL_RETURNED"
            self._write_trace_row({
                "event": "velocity_publish_result",
                "wall_utc": datetime.now(timezone.utc).isoformat(),
                "wall_monotonic_sec": time.monotonic(),
                "source": str(msg.source),
                "publish_outcome": "CALL_RETURNED",
            })
            if forward != 0.0:
                self.nonzero_publish_calls_returned += 1
            else:
                self.zero_publish_calls_returned += 1

    def _wait_service(self, client: Any, label: str, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if client.wait_for_service(timeout_sec=0.1):
                return
            rclpy.spin_once(self, timeout_sec=0.005)
        raise TimeoutError(f"{label} service unavailable")

    def _call(self, client: Any, request: Any, label: str, timeout: float = 1.0) -> Any:
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self.interrupted.is_set():
                raise RuntimeError("operator_shutdown")
            rclpy.spin_once(self, timeout_sec=0.005)
        if not future.done() or future.result() is None:
            raise TimeoutError(f"{label} timed out")
        return future.result()

    def read_input_source(self, label: str) -> dict[str, Any]:
        self._wait_service(self.source_client, CURRENT_INPUT_SOURCE_SERVICE)
        request = GetCurrentInputSource.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        response = self._call(self.source_client, request, f"GetCurrentInputSource:{label}")
        return self._record_source_response(request, response, label)

    def _record_source_response(self, request: Any, response: Any, label: str) -> dict[str, Any]:
        observed_monotonic = time.monotonic()
        result = {
            "label": label, "wall_utc": datetime.now(timezone.utc).isoformat(),
            "observed_monotonic_s": observed_monotonic,
            "request": _json_safe(request), "response": _json_safe(response),
            "response_header_code": int(response.response.header.code),
            "response_state": int(response.response.state.value),
            "task_id": int(response.response.task_id),
            "input_source": _json_safe(response.input_source),
        }
        self.last_source = result
        self.readback_at["get_source"] = observed_monotonic
        self.source_observations.append(result)
        self.events.append({"event": "get_current_input_source", **result})
        return result

    def read_mode(self, label: str) -> dict[str, Any]:
        self._wait_service(self.get_mode_client, GET_MC_ACTION_SERVICE)
        request = GetMcAction.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        response = self._call(self.get_mode_client, request, f"GetMcAction:{label}")
        return self._record_mode_response(request, response, label)

    def _record_mode_response(self, request: Any, response: Any, label: str) -> dict[str, Any]:
        observed_monotonic = time.monotonic()
        result = {
            "label": label, "wall_utc": datetime.now(timezone.utc).isoformat(),
            "observed_monotonic_s": observed_monotonic,
            "request": _json_safe(request), "response": _json_safe(response),
            "response_header_code": int(response.header.code),
            "current_action_value": int(response.info.current_action.value),
            "action_desc": str(response.info.action_desc),
            "status_value": int(response.info.status.value),
        }
        self.last_mode = result
        self.readback_at["get_mode"] = observed_monotonic
        self.mode_observations.append(result)
        self.events.append({"event": "get_mc_action", **result})
        return result

    def _make_source_request(self, action_value: int) -> Any:
        request = SetMcInputSource.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        request.action = McInputAction()
        request.action.value = int(action_value)
        request.input_source.name = self.source
        request.input_source.priority = SOURCE_PRIORITY
        request.input_source.timeout = SOURCE_TIMEOUT_MS
        return request

    def _source_action_once(self, action_value: int, action_name: str) -> dict[str, Any]:
        """Submit one mutating source action; never replay an ambiguous timeout."""
        self._wait_service(self.register_client, INPUT_SOURCE_SERVICE, timeout=8.0)
        request = self._make_source_request(action_value)
        try:
            response = self._call(
                self.register_client, request, f"SetMcInputSource:{action_name}", timeout=1.0,
            )
        except TimeoutError as exc:
            attempt = {
                "attempt": 1,
                "transport_timeout": True,
                "transport_outcome": "UNKNOWN_REQUEST_MAY_HAVE_BEEN_ACCEPTED",
                "error": str(exc),
                "request": _json_safe(request),
            }
            return {
                "action": action_name,
                "transport_state": "TIMEOUT_UNKNOWN",
                "attempts": [attempt],
                "final": None,
                "enable_fallback_allowed": False,
            }
        header_code = int(response.response.header.code)
        state_value = int(response.response.state.value)
        attempt = {
            "attempt": 1,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "response": _json_safe(response),
            "response_header_code": header_code,
            "response_state": state_value,
            "task_id": int(response.response.task_id),
            "operation_accepted_by_service": header_code == 0,
            "task_completion_confirmed": state_value == 1,
        }
        return {
            "action": action_name,
            "transport_state": "RESPONSE_RECEIVED",
            "attempts": [attempt],
            "final": attempt,
            "enable_fallback_allowed": header_code != 0,
        }

    def register_source(self) -> dict[str, Any]:
        add = self._source_action_once(1001, "ADD")
        add_final = add.get("final")
        enable = None
        # ENABLE is a bounded recovery only after an explicit ADD rejection.
        # A timeout is ambiguous and must not be replayed as another mutation.
        add_assessment = classify_source_registration(add.get("final"))
        if add_assessment["enable_fallback_allowed"]:
            enable = self._source_action_once(2001, "ENABLE")
        registration = classify_source_registration(
            add_final,
            enable.get("final") if enable else None,
        )
        registration.update({
            "source": self.source,
            "priority": SOURCE_PRIORITY,
            "timeout_ms": SOURCE_TIMEOUT_MS,
            "add": add,
            "enable": enable,
            "transport_ambiguity_blocks_motion": add.get("transport_state") == "TIMEOUT_UNKNOWN",
        })
        self.events.append({"event": "register_input_source", **registration})
        return registration

    def request_ld_once(self) -> dict[str, Any]:
        self._wait_service(self.set_mode_client, SET_MC_ACTION_SERVICE)
        request = SetMcAction.Request()
        request.header = RequestHeader()
        request.header.stamp = self.get_clock().now().to_msg()
        request.source = ACTION_SOURCE_NAME
        request.command = McActionCommand()
        request.command.action_desc = MODE_LOCOMOTION
        submitted = datetime.now(timezone.utc).isoformat()
        response = self._call(self.set_mode_client, request, "SetMcAction:LOCOMOTION_DEFAULT", timeout=1.0)
        response_payload = _json_safe(response.response)
        diagnostics = classify_set_mc_action_response(
            int(response.response.header.code),
            int(response.response.status.value),
            str(response.response.message),
        )
        result = {
            "wall_utc_submitted": submitted,
            "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "request_source": str(request.source),
            "requested_action_desc": str(request.command.action_desc),
            "response": response_payload,
            **diagnostics,
        }
        self.events.append({"event": "set_mc_action_ld", **result})
        return result

    def wait_for_standing(self) -> None:
        deadline = time.monotonic() + self.args.ready_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.interrupted.is_set():
                raise RuntimeError("operator_shutdown")
            rclpy.spin_once(self, timeout_sec=0.02)
            fault = self._safety_fault(expected_mode=MODE_STAND)
            if (
                fault is None
                and int(self.latest["common"].body_status.value) == 1
                and int(self.latest["common"].fsm_state.current_state) == MC_FSM_STABLE
            ):
                return
        raise TimeoutError("fresh upright STAND_DEFAULT telemetry not observed; verified Reset is required")

    def check_transport_scope(self) -> dict[str, Any]:
        try:
            mc_identity = load_verified_mc_runtime_identity(
                self.args.mc_sim_config_path,
                self.output_dir / "mc-console.log",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"MC runtime identity could not be verified: {exc}") from exc
        if mc_identity.mc_config_sha256 != self.mc_sim_config_sha256:
            raise RuntimeError("MC runtime identity config hash differs from the loaded MC safety config")
        try:
            mc_subscriber_provenance = load_verified_mc_velocity_subscriber_provenance(
                self.args.mc_sim_config_path,
                self.output_dir / "mc-console.log",
                mc_identity,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"MC velocity subscriber provenance could not be verified: {exc}") from exc
        dds_qos_override_environment = capture_dds_qos_override_environment()
        self.mc_runtime_identity = mc_identity
        deadline = time.monotonic() + 2.0
        last_publishers: list[str] = []
        last_subscribers: list[str] = []
        state_topic_publishers: dict[str, list[str]] = {}
        last_publisher_qos: dict[str, Any] | None = None
        last_subscriber_qos: list[dict[str, Any]] = []
        last_subscriber_endpoint_metadata: list[dict[str, Any]] = []
        fault = None
        while rclpy.ok() and time.monotonic() < deadline:
            publisher_info = list(self.get_publishers_info_by_topic(VELOCITY_TOPIC))
            subscriber_info = list(self.get_subscriptions_info_by_topic(VELOCITY_TOPIC))
            last_publishers = [str(info.node_name) for info in publisher_info]
            last_subscribers = [str(info.node_name) for info in subscriber_info]
            last_subscriber_endpoint_metadata = [
                {
                    "node_name": str(info.node_name),
                    "node_namespace": str(info.node_namespace),
                    "queried_topic_name": VELOCITY_TOPIC,
                    "topic_type": str(info.topic_type),
                }
                for info in subscriber_info
            ]
            state_topic_publishers = {
                topic: [
                    str(info.node_name)
                    for info in self.get_publishers_info_by_topic(topic)
                ]
                for topic in (
                    "/aima/hal/odom/state",
                    "/aima/hal/imu/torso/state",
                    "/aima/hal/joint/leg/state",
                    "/aima/hal/joint/leg/command",
                )
            }
            last_publisher_qos = self._endpoint_qos(publisher_info[0]) if len(publisher_info) == 1 else None
            last_subscriber_qos = [self._endpoint_qos(info) for info in subscriber_info]
            publisher_qos = last_publisher_qos or {}
            node_velocity_publishers = [
                publisher for publisher in self.publishers
                if str(publisher.topic_name) == VELOCITY_TOPIC
            ]
            local_publisher_identity = {
                "node_name": self.get_name(),
                "topic_name": str(self.publisher.topic_name),
                "message_type": str(self.publisher.msg_type.__name__),
                "is_expected_message_type": self.publisher.msg_type is McLocomotionVelocity,
                "is_owned_publisher": any(publisher is self.publisher for publisher in self.publishers),
                "node_topic_publisher_count": len(node_velocity_publishers),
            }
            publisher_api_qos = self._qos_profile(self.publisher.qos_profile)
            fault = validate_runtime_scope(
                rmw_implementation=os.environ.get("RMW_IMPLEMENTATION"),
                ros_distro=os.environ.get("ROS_DISTRO"),
                ros_domain_id=os.environ.get("ROS_DOMAIN_ID"),
                docker_network_mode=os.environ.get("ROBOTSIM_VERIFIED_NETWORK_MODE"),
                publisher_names=last_publishers,
                subscriber_names=last_subscribers,
                own_node_name=self.get_name(),
                publisher_reliability=str(publisher_qos.get("reliability", "unknown")),
                subscriber_reliabilities=[str(qos["reliability"]) for qos in last_subscriber_qos],
                publisher_durability=str(publisher_qos.get("durability", "unknown")),
                subscriber_durabilities=[str(qos["durability"]) for qos in last_subscriber_qos],
                publisher_history=str(publisher_qos.get("history", "unknown")),
                publisher_depth=int(publisher_qos.get("depth", 0)),
                publisher_local_identity=local_publisher_identity,
                publisher_configured_qos=self.publisher_configured_qos,
                publisher_api_qos=publisher_api_qos,
                subscriber_histories=[str(qos["history"]) for qos in last_subscriber_qos],
                subscriber_depths=[int(qos["depth"]) for qos in last_subscriber_qos],
                subscriber_endpoint_metadata=last_subscriber_endpoint_metadata,
                mc_subscriber_provenance=mc_subscriber_provenance,
                dds_qos_override_environment=dds_qos_override_environment,
                mc_identity=mc_identity,
            )
            if fault is None:
                fault = validate_local_sim_publishers(
                    state_topic_publishers,
                    mc_identity=mc_identity,
                )
            if fault is None:
                result = {
                    "status": "PASS",
                    "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
                    "ros_distro": os.environ.get("ROS_DISTRO"),
                    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
                    "docker_network_mode": os.environ.get("ROBOTSIM_VERIFIED_NETWORK_MODE"),
                    "velocity_publishers": last_publishers,
                    "velocity_subscribers": last_subscribers,
                    "verified_mc_runtime_identity": mc_identity.as_dict(),
                    "publisher_qos": last_publisher_qos,
                    "publisher_local_identity": local_publisher_identity,
                    "publisher_configured_qos": self.publisher_configured_qos,
                    "publisher_api_qos": publisher_api_qos,
                    "publisher_qos_profile_api": {
                        "name": "rclpy.Publisher.qos_profile",
                        "separate_middleware_actual_qos_query_available": False,
                    },
                    "publisher_qos_evidence_note": (
                        "The Humble rclpy Publisher exposes the profile attached at create_publisher as "
                        "qos_profile; it does not expose a separate effective-DDS QoS query here. The graph "
                        "independently verifies reliability/durability while history/depth may be unknown/0. "
                        "This does not prove that MC received a command."
                    ),
                    "subscriber_qos": last_subscriber_qos,
                    "subscriber_endpoint_metadata": last_subscriber_endpoint_metadata,
                    "mc_velocity_subscriber_configuration_provenance": mc_subscriber_provenance,
                    "dds_qos_override_environment": dds_qos_override_environment,
                    "subscriber_history_depth_discovery": {
                        **mc_subscriber_history_depth_evidence(
                            str(last_subscriber_qos[0]["history"]),
                            int(last_subscriber_qos[0]["depth"]),
                            rmw_implementation=str(os.environ.get("RMW_IMPLEMENTATION")),
                            ros_distro=str(os.environ.get("ROS_DISTRO")),
                        ),
                        "validation_exception_applied": (
                            len(last_subscriber_qos) == 1
                            and last_subscriber_qos[0]["history"] == "unknown"
                            and last_subscriber_qos[0]["depth"] == 0
                        ),
                    },
                    "subscriber_qos_evidence_note": (
                        "Reliability and durability are graph-observed and compatibility-checked. The exact "
                        "Humble/rmw_fastrtps_cpp unknown/0 history-depth discovery pair is accepted only "
                        "after MC process, config route, startup type registration, endpoint namespace/type, "
                        "and absence of known XML QoS overrides are verified. The actual MC DataReader "
                        "history/depth remains UNVERIFIED; this exception does not prove command delivery."
                    ),
                    "local_state_and_command_publishers": state_topic_publishers,
                    "command_qos": {
                        "reliability": COMMAND_QOS_RELIABILITY,
                        "depth": COMMAND_QOS_DEPTH,
                        "durability": COMMAND_QOS_DURABILITY,
                    },
                }
                self.scope_evidence = result
                return result
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f"simulation-only transport isolation failed: {fault}; "
            f"publishers={last_publishers!r} ({last_publisher_qos!r}); "
            f"subscribers={last_subscribers!r} ({last_subscriber_qos!r}); "
            f"local_state_and_command_publishers={state_topic_publishers!r}"
        )

    @staticmethod
    def _endpoint_qos(info: Any) -> dict[str, Any]:
        return FocusedM1Diagnostic._qos_profile(info.qos_profile)

    @staticmethod
    def _qos_profile(profile: Any) -> dict[str, Any]:
        reliability = (
            "reliable" if profile.reliability == ReliabilityPolicy.RELIABLE
            else "best_effort" if profile.reliability == ReliabilityPolicy.BEST_EFFORT
            else "unknown"
        )
        durability = (
            "volatile" if profile.durability == DurabilityPolicy.VOLATILE
            else "transient_local" if profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
            else "unknown"
        )
        history = (
            "keep_last" if profile.history == HistoryPolicy.KEEP_LAST
            else "keep_all" if profile.history == HistoryPolicy.KEEP_ALL
            else "unknown"
        )
        return {
            "reliability": reliability,
            "durability": durability,
            "history": history,
            "depth": int(profile.depth),
        }

    def _start_async_mode_query(self) -> None:
        request = GetMcAction.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        self.mode_future = self.get_mode_client.call_async(request)
        self.mode_future_request = request
        self.mode_future_started = time.monotonic()

    def _start_async_source_query(self) -> None:
        request = GetCurrentInputSource.Request()
        request.request = CommonRequest()
        request.request.header.stamp = self.get_clock().now().to_msg()
        self.source_future = self.source_client.call_async(request)
        self.source_future_request = request
        self.source_future_started = time.monotonic()

    def _poll_async_readbacks(self) -> None:
        now = time.monotonic()
        if self.mode_future is not None and self.mode_future.done():
            try:
                response = self.mode_future.result()
                if response is None:
                    self.async_readback_fault = "GetMcAction returned no response"
                else:
                    self._record_mode_response(self.mode_future_request, response, "activation_poll")
            except Exception as exc:
                self.async_readback_fault = f"GetMcAction async error:{type(exc).__name__}:{exc}"
            self.mode_future = None
            self.mode_future_request = None
        elif self.mode_future is not None and now - self.mode_future_started > 0.4:
            self.async_readback_fault = "GetMcAction async response timeout"
            self.mode_future = None
            self.mode_future_request = None

        if self.source_future is not None and self.source_future.done():
            try:
                response = self.source_future.result()
                if response is None:
                    self.async_readback_fault = "GetCurrentInputSource returned no response"
                else:
                    self._record_source_response(self.source_future_request, response, "activation_poll")
            except Exception as exc:
                self.async_readback_fault = f"GetCurrentInputSource async error:{type(exc).__name__}:{exc}"
            self.source_future = None
            self.source_future_request = None
        elif self.source_future is not None and now - self.source_future_started > 0.4:
            self.async_readback_fault = "GetCurrentInputSource async response timeout"
            self.source_future = None
            self.source_future_request = None

        if self.mode_future is None and now >= self.next_mode_query:
            self._start_async_mode_query()
            self.next_mode_query = now + 0.2
        if self.source_future is None and now >= self.next_source_query:
            self._start_async_source_query()
            self.next_source_query = now + 0.2

    def _first_motion_snapshot(self) -> dict[str, Any]:
        common = self.latest.get("common")
        now = time.monotonic()
        ages = {
            key: now - self.received_at[key] if key in self.received_at else None
            for key in REQUIRED_TOPICS
        }
        ages["get_mode"] = now - self.readback_at["get_mode"] if "get_mode" in self.readback_at else None
        ages["get_source"] = now - self.readback_at["get_source"] if "get_source" in self.readback_at else None
        pose = self._pose()
        bounds = None
        if common is not None:
            try:
                bounds = {
                    "min_value": float(common.speed_status.forward_bounds.min_value),
                    "max_value": float(common.speed_status.forward_bounds.max_value),
                }
            except (AttributeError, TypeError, ValueError):
                bounds = None
        return {
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "sample_monotonic_s": now,
            "state_age_s": ages,
            "readback_monotonic_s": {
                "get_mode": self.readback_at.get("get_mode"),
                "get_source": self.readback_at.get("get_source"),
            },
            "pose": pose,
            "get_mode_desc": self.last_mode.get("action_desc") if self.last_mode else None,
            "get_mode_header_code": self.last_mode.get("response_header_code") if self.last_mode else None,
            "get_mode_status_value": self.last_mode.get("status_value") if self.last_mode else None,
            "common_mode_desc": str(common.action_info.action_desc) if common else None,
            "common_mode_status_value": int(common.action_info.status.value) if common else None,
            "fsm_state": int(common.fsm_state.current_state) if common else None,
            "body_status": int(common.body_status.value) if common else None,
            "get_source_header_code": self.last_source.get("response_header_code") if self.last_source else None,
            "get_source_name": (
                self.last_source.get("input_source", {}).get("name", "") if self.last_source else ""
            ),
            "common_source_name": str(common.input_source.name) if common else "",
            "forward_bounds": bounds,
            "leg_states": self._joint_states_as_dicts(),
            "readback_fault": self.async_readback_fault,
        }

    def collect_stationary_odom_baseline(self) -> dict[str, Any]:
        """Measure same-run no-command odometry noise before registering or sending velocity."""
        self.read_mode("stationary_baseline_start")
        source_start = self.read_input_source("stationary_baseline_start")
        if source_start["response_header_code"] != 0:
            raise RuntimeError("GetCurrentInputSource failed before odometry baseline")
        source_name = str(source_start["input_source"].get("name", ""))
        if source_name not in ("", self.source):
            raise RuntimeError(f"source conflict before odometry baseline: {source_name!r}")

        samples: list[dict[str, Any]] = []
        baseline_start_s: float | None = None
        next_sample_s: float | None = None
        baseline_end_s: float | None = None
        while rclpy.ok():
            if self.interrupted.is_set():
                raise RuntimeError("operator_shutdown during stationary odometry baseline")
            if next_sample_s is not None:
                delay = next_sample_s - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
            snapshot = self.sample_for_first_motion()
            fault = snapshot_fault(
                snapshot,
                self.joint_limits,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
            )
            if fault:
                raise RuntimeError(f"stationary odometry baseline safety fault: {fault}")
            if (
                snapshot.get("get_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
                or snapshot.get("common_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
                or snapshot.get("fsm_state") != MC_FSM_STABLE
            ):
                raise RuntimeError("stationary odometry baseline requires SD/LD with stable FSM state")
            if snapshot.get("get_source_name", "") or snapshot.get("common_source_name", ""):
                raise RuntimeError("input source is already active during the no-command odometry baseline")
            sample_time = float(snapshot["sample_monotonic_s"])
            if baseline_start_s is None:
                baseline_start_s = sample_time
                baseline_end_s = baseline_start_s + BASELINE_DURATION_S
            samples.append(snapshot)
            self._trace_first_motion_sample("stationary-baseline", snapshot)
            if sample_time >= float(baseline_end_s):
                break
            next_sample_s = baseline_start_s + len(samples) * BASELINE_SAMPLE_PERIOD_S
        else:
            raise RuntimeError("ROS shutdown during stationary odometry baseline")

        baseline = derive_stationary_odom_baseline(samples)
        baseline.update({
            "requested_duration_s": BASELINE_DURATION_S,
            "start_monotonic_s": baseline_start_s,
            "end_monotonic_s": samples[-1]["sample_monotonic_s"],
            "commands_sent": 0,
        })
        return baseline

    def _spin_for(self, duration_s: float) -> None:
        """Service telemetry/readbacks while waiting, including the zero-stop window.

        Two spin_once calls per sample cannot consume all five state streams.
        A plain sleep during zero-stop prevents any callbacks for >1 second
        and manufactures stale telemetry. Keep the existing deadline and
        watchdog; pumping begins only after each command publication.
        """
        deadline = time.monotonic() + max(0.0, duration_s)
        while rclpy.ok() and time.monotonic() < deadline:
            # The watchdog may service zero-stop waits on its own thread.
            # Never spin the same executor concurrently; zero publication
            # already happened before waiting for this short lock.
            with self.ros_spin_lock:
                rclpy.spin_once(self, timeout_sec=min(0.001, max(0.0, deadline - time.monotonic())))
                self._poll_async_readbacks()

    def sample_for_first_motion(self) -> dict[str, Any]:
        self._spin_for(0.005)
        snapshot = self._first_motion_snapshot()
        if self.async_readback_fault:
            snapshot["get_mode_header_code"] = -1
            snapshot["get_source_header_code"] = -1
        return snapshot

    def _trace_first_motion_sample(self, phase: str, snapshot: dict[str, Any]) -> None:
        self.current_phase = phase
        self._trace(phase, snapshot=snapshot)

    def _source_response_before_motion(self) -> dict[str, Any]:
        result = self.read_input_source("before_first_motion")
        if result["response_header_code"] != 0:
            raise RuntimeError("GetCurrentInputSource preflight failed")
        current = str(result["input_source"].get("name", ""))
        common = self.latest.get("common")
        common_name = str(common.input_source.name) if common else ""
        if current not in ("", self.source) or common_name not in ("", self.source):
            raise RuntimeError(f"source conflict before first motion: service={current!r}, common={common_name!r}")
        return result

    def execute(self) -> dict[str, Any]:
        stationary_baseline: dict[str, Any] | None = None
        result: dict[str, Any] = {
            "schema": "robotsim.issue80.m1.first-motion.v2",
            "simulation_only": True,
            "acceptance_scope": "single first-motion diagnostic only; not a locomotion or navigation pass",
            "attempt_id": self.args.attempt_id,
            "maintainer_authorized": bool(self.args.maintainer_authorized),
            "reset_gate_verified_by_parent": bool(self.args.reset_gate_verified),
            "status": "NOT_RUN",
            "source": self.source,
            "action_source": ACTION_SOURCE_NAME,
            "source_timeout_ms": SOURCE_TIMEOUT_MS,
            "installed_mc_sim_config_path": self.args.mc_sim_config_path,
            "installed_mc_sim_config_sha256": self.mc_sim_config_sha256,
            "installed_mc_default_source_timeouts": self.mc_input_source_timeouts,
            "verified_fsm_runtime_library_sha256": self.fsm_runtime_contract,
            "source_timeout_semantics": {
                "configured_registration_timeout_ms": SOURCE_TIMEOUT_MS,
                "documented_timeout_effect": "timeout is measured from the last nonzero command; zero commands do not refresh it; after expiry command data is cleared while the selected source remains",
                "installed_mc_effect_dynamically_verified": False,
                "mc_receipt_or_stop_ack_available": False,
            },
            "publish_hz": PUBLISH_HZ,
            "forward_command_m_s": self.args.forward_speed,
            "pulse_limit_s": self.args.pulse_duration,
            "zero_stop_min_s": 1.0,
            "post_stop_observe_s": 2.5,
            "leg_limit_source": self.args.robot_model_path,
            "leg_limit_source_sha256": self.joint_limits_sha256,
            "leg_joint_limits": self.joint_limits,
            "safety_limits": {
                "state_timeout_s": self.args.state_timeout,
                "min_root_z_m": self.args.min_root_z,
                "max_tilt_deg": self.args.max_tilt_deg,
            },
            "fault": None,
        }
        try:
            result["transport_scope"] = self.check_transport_scope()
            self.wait_for_standing()
            result["standing_preflight"] = "FRESH_STAND_DEFAULT_BODY_STAND"
            result["source_before_registration"] = self._source_response_before_motion()
            registration = self.register_source()
            result["source_registration"] = registration
            if registration["success"]:
                self.read_mode("before_set_mc_action_ld")
                set_action = self.request_ld_once()
            else:
                set_action = {
                    "accepted_by_service": False,
                    "header_code": None,
                    "status_value": None,
                    "message": "source registration failed; SetMcAction was not sent",
                }
                self.events.append({"event": "set_mc_action_ld_not_sent", **set_action})
            result["set_mc_action_ld"] = set_action
            result["mode_after_ld_request"] = self.read_mode("after_set_mc_action_ld")
            result["source_after_ld_request"] = self.read_input_source("after_set_mc_action_ld")
            if registration["success"] and set_action.get("accepted_by_service"):
                stationary_baseline = self.collect_stationary_odom_baseline()
            result["stationary_odom_baseline"] = stationary_baseline

            # The first nonzero request is permitted after the standing, transport,
            # registration and SetMcAction response gates. It does not require LD or
            # an active source beforehand; these are measured during activation.
            pulse = run_first_motion_window(
                sample=self.sample_for_first_motion,
                publish=self._publish,
                sleep=self._spin_for,
                registration=registration,
                set_action=set_action,
                joint_limits=self.joint_limits,
                baseline_forward_displacements_m=(
                    [window["forward_displacement_m"] for window in stationary_baseline["windows"]]
                    if stationary_baseline else None
                ),
                authorized=True,
                forward_command_m_s=self.args.forward_speed,
                pulse_duration_s=self.args.pulse_duration,
                extra_post_stop_observe_s=self.args.extra_stop_observation,
                shutdown_event=self.interrupted,
                on_sample=self._trace_first_motion_sample,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
            )
            self.pulse_result = pulse
            result["activation"] = pulse
            result["nonzero_publish_calls_returned"] = pulse["nonzero_publish_calls_returned"]
            result["zero_publish_calls_returned"] = self.zero_publish_calls_returned
            result["mc_velocity_receipt_confirmed"] = False
            result["mc_stop_confirmed"] = False
            result["get_mc_action_final"] = self.read_mode("final_post_stop")
            result["get_current_input_source_final"] = self.read_input_source("final_post_stop")
            result["final_common_state"] = _json_safe(self.latest.get("common"))
            result["final_odometry"] = self._pose()
            result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
            result["status"] = (
                "PASS" if pulse["status"] == "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED"
                else "BLOCKED"
            )
        except Exception as exc:
            result["fault"] = f"{type(exc).__name__}: {exc}"
            result["status"] = "BLOCKED"
            if self.scope_evidence is not None and self.pulse_result is None:
                # After graph isolation, an exception before the pulse still takes
                # the complete source-timeout zero window and post-stop observation.
                try:
                    result["exception_safe_activation"] = run_first_motion_window(
                        sample=self.sample_for_first_motion,
                        publish=self._publish,
                        sleep=self._spin_for,
                        registration={"success": False, "selected_action": None},
                        set_action={"accepted_by_service": False},
                        joint_limits=self.joint_limits,
                        baseline_forward_displacements_m=(
                            [window["forward_displacement_m"] for window in stationary_baseline["windows"]]
                            if stationary_baseline else None
                        ),
                        authorized=True,
                        forward_command_m_s=self.args.forward_speed,
                        pulse_duration_s=self.args.pulse_duration,
                        extra_post_stop_observe_s=self.args.extra_stop_observation,
                        shutdown_event=self.interrupted,
                        on_sample=self._trace_first_motion_sample,
                        state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z,
                        max_tilt_deg=self.args.max_tilt_deg,
                    )
                except Exception as stop_exc:
                    result["exception_safe_stop_fault"] = f"{type(stop_exc).__name__}: {stop_exc}"
            result["final_common_state"] = _json_safe(self.latest.get("common"))
            result["final_odometry"] = self._pose()
            result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
        finally:
            result["events"] = self.events
            result["get_mc_action_history"] = self.mode_observations
            result["get_current_input_source_history"] = self.source_observations
            result["velocity_command_trace"] = self.command_trace
            result["velocity_publisher_count"] = 1
            result["limitations"] = [
                "Nonzero motion requires the maintainer authorization gate in the parent Reset runner and this child command.",
                "Control handshake requires contemporaneous source and LD evidence; physical motion is separately classified from forward-projected odometry against this run's stationary baseline.",
                "The in-process threading.Timer is not a process-external watchdog and cannot publish after this process is killed.",
                "Publisher.publish returning records only a local call outcome; MC command receipt and zero-command effect have no direct acknowledgement and remain unconfirmed.",
                "The installed MC config shows a 1000 ms input source timeout and official docs describe command clearing after timeout; exact installed binary stop behavior is not live-verified.",
                "Leg position and velocity cutoffs are loaded from the installed MC simulation robot_model.yaml and its SHA-256 is recorded.",
                "No vendor model, binary, gain, safety threshold, or original reset/standing gate is modified.",
            ]
            if self.result_path.exists():
                raise FileExistsError("refusing to overwrite first-motion result")
            with self.result_path.open("x", encoding="utf-8") as stream:
                json.dump(_json_safe(result), stream, indent=2, allow_nan=False)
                stream.write("\n")
        return result

    def close(self) -> None:
        self.trace.close()


def _focused_diagnostic_command(args: argparse.Namespace) -> int:
    validate_first_motion_authorization(
        reset_gate_verified=args.reset_gate_verified,
        maintainer_authorized=args.maintainer_authorized,
        network_mode=os.environ.get("ROBOTSIM_VERIFIED_NETWORK_MODE"),
    )
    rclpy.init()
    node = FocusedM1Diagnostic(args)

    def request_stop(signum, frame):
        node.interrupted.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = node.execute()
        print(json.dumps(_json_safe(result), indent=2))
        return 0 if result["status"] == "PASS" else 2
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.close()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    first = subparsers.add_parser(
        "first-motion",
        help="one authorized SIMULATION_ONLY first-motion run following the verified Reset/standing gates",
    )
    first.add_argument("--maintainer-authorized", action="store_true")
    first.add_argument("--reset-gate-verified", action="store_true")
    first.add_argument("--attempt-id", required=True)
    first.add_argument("--output-dir", required=True)
    first.add_argument("--robot-model-path", default=ROBOT_MODEL_PATH)
    first.add_argument("--mc-sim-config-path", default=MC_SIM_CONFIG_PATH)
    first.add_argument("--extra-stop-observation", type=float, default=0.0)
    first.add_argument("--pulse-duration", type=float, default=1.0)
    first.add_argument("--forward-speed", type=float, default=0.2)
    first.add_argument("--state-timeout", type=float, default=0.5)
    first.add_argument("--ready-timeout", type=float, default=4.0)
    first.add_argument("--max-tilt-deg", type=float, default=15.0)
    first.add_argument("--min-root-z", type=float, default=0.52)
    args = parser.parse_args()
    try:
        return _focused_diagnostic_command(args)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
