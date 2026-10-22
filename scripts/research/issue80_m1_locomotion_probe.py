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
from std_msgs.msg import Float64MultiArray

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME, ActivationTracker, COMMAND_PUBLISHER_TOPIC, COMMAND_QOS_DEPTH, COMMAND_QOS_DURABILITY,
    BASELINE_DURATION_S, BASELINE_SAMPLE_PERIOD_S, BASELINE_WINDOW_COUNT,
    COMMAND_QOS_RELIABILITY, MC_SIM_CONFIG_PATH, MODE_LOCOMOTION, MODE_STAND, PUBLISH_HZ,
    MC_FSM_ACTIVE_STATES, MC_FSM_SAFE, MC_FSM_STABLE, MAX_AUTHORIZED_PULSE_S, verify_mc_fsm_runtime_contract,
    SOURCE_PRIORITY, SOURCE_TIMEOUT_MS, VELOCITY_SOURCE_NAME, classify_source_registration,
    TURN_COMMAND_RAD_S, TURN_FORWARD_COMMAND_M_S, TURN_FORWARD_WARMUP_S,
    TURN_PULSE_S, ZERO_STOP_MIN_MESSAGES, SustainedYawRateWindow, run_angular_turn_window,
    capture_dds_qos_override_environment, derive_stationary_odom_baseline,
    float64_multi_array_snapshot, load_leg_joint_limits, load_mc_input_source_timeouts,
    joint_command_array_snapshot,
    VerifiedMcRuntimeIdentity, load_verified_mc_runtime_identity,
    load_verified_mc_velocity_subscriber_provenance, mc_subscriber_history_depth_evidence,
    build_forward_point_goal, proximity_smoothstep_speed, run_first_motion_window, snapshot_fault,
    stand_mode_readback_confirmed,
    summarize_stationary_physical_state,
    validate_local_sim_publishers, validate_first_motion_authorization,
    validate_runtime_scope,
)
from issue80_m1_mode_logic import classify_set_mc_action_response

VELOCITY_TOPIC = COMMAND_PUBLISHER_TOPIC
MC_VELOCITY_ENDPOINT_DISCOVERY_TIMEOUT_S = 5.0
POINT_GOAL_DISTANCE_M = 0.30
POINT_GOAL_STOP_COMPENSATION_M = 0.10
POINT_GOAL_STOP_TRIGGER_M = 0.04
POINT_GOAL_FINAL_TOLERANCE_M = 0.05
POINT_GOAL_COMMAND_M_S = 0.14
POINT_GOAL_TERMINAL_COMMAND_M_S = 0.112
POINT_GOAL_DECEL_START_DISTANCE_M = 0.12
POINT_GOAL_DECEL_DURATION_S = 1.0
POINT_GOAL_MAX_DURATION_S = 10.0
POINT_GOAL_SETTLED_SAMPLES = 3
INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
CURRENT_INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/GetCurrentInputSource"
GET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"
SET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/SetMcAction"
MC_DEBUG_F64_TOPIC = "/aima/mc_debug_f64"
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
        if not 0.0 < args.pulse_duration <= MAX_AUTHORIZED_PULSE_S:
            raise ValueError("pulse duration exceeds the maintainer-reviewed 10 s ceiling")
        if not 0.0 <= args.ramp_duration_s <= args.pulse_duration:
            raise ValueError("velocity ramp duration must fit inside the bounded pulse")
        if args.ramp_duration_s > 0.0 and (
            args.ramp_start_speed is None
            or not 0.0 < args.ramp_start_speed <= args.forward_speed
        ):
            raise ValueError("smooth ramp start must be positive and no faster than target")
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
        self.mc_debug_f64_message_count = 0
        self.mc_debug_f64_first_layout: dict[str, Any] | None = None
        self.latest_mc_debug_f64_snapshot: dict[str, Any] | None = None
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
        # Optional raw diagnostic only: its values are not safety gates and are
        # never assigned semantics from the generic Float64MultiArray type.
        self.create_subscription(
            Float64MultiArray, MC_DEBUG_F64_TOPIC, self._store_mc_debug_f64, self.state_qos
        )

    def _store(self, key: str, msg: Any) -> None:
        self.latest[key] = msg
        self.received_at[key] = time.monotonic()

    def _store_mc_debug_f64(self, msg: Float64MultiArray) -> None:
        received = time.monotonic()
        payload = float64_multi_array_snapshot(msg, received)
        if payload is None:
            return
        self.mc_debug_f64_message_count += 1
        if self.mc_debug_f64_first_layout is None:
            self.mc_debug_f64_first_layout = payload["layout"]
        self.latest_mc_debug_f64_snapshot = payload
        self._write_trace_row({
            "event": "mc_debug_f64",
            "phase": self.current_phase,
            "source": self.source,
            "topic": MC_DEBUG_F64_TOPIC,
            "payload": payload,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "wall_monotonic_sec": received,
        })

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
            "wx": float(odom.twist.twist.angular.x),
            "wy": float(odom.twist.twist.angular.y),
            "wz": float(odom.twist.twist.angular.z),
            "frame_id": str(odom.header.frame_id),
            "child_frame_id": str(odom.child_frame_id),
            "twist_frame_id": str(odom.child_frame_id),
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

    def _publish(
        self, forward: float = 0.0, lateral: float = 0.0, angular: float = 0.0,
    ) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source
        msg.forward_velocity = float(forward)
        msg.lateral_velocity = float(lateral)
        msg.angular_velocity = float(angular)
        with self.publisher_lock:
            command_row = {
                "event": "velocity_command",
                "command_kind": (
                    "zero_stop" if forward == 0.0 and lateral == 0.0 and angular == 0.0
                    else "turning_activation" if forward == 0.0 and lateral == 0.0
                    else "nonzero_activation"
                ),
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
            if forward != 0.0 or lateral != 0.0 or angular != 0.0:
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

    def request_postwalk_stand_once(self) -> dict[str, Any]:
        """Reassert the vendor stand action once after a confirmed stop window."""
        snapshot = self._first_motion_snapshot()
        fault = snapshot_fault(
            snapshot, self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=getattr(self.args, "max_observed_planar_speed_m_s", None),
        )
        if fault:
            raise RuntimeError(f"post-stop stand request safety gate failed: {fault}")
        if (
            snapshot.get("get_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
            or snapshot.get("common_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
        ):
            raise RuntimeError("post-stop stand request observed unrelated MC action")
        if snapshot.get("common_source_name") not in ("", self.source):
            raise RuntimeError("post-stop stand request observed conflicting active source")
        request = SetMcAction.Request()
        request.header = RequestHeader()
        request.header.stamp = self.get_clock().now().to_msg()
        request.source = ACTION_SOURCE_NAME
        request.command = McActionCommand()
        request.command.action_desc = MODE_STAND
        submitted = datetime.now(timezone.utc).isoformat()
        response = self._call(
            self.set_mode_client, request, "SetMcAction:post-stop-STAND_DEFAULT", timeout=1.0
        )
        payload = _json_safe(response.response)
        diagnostics = classify_set_mc_action_response(
            int(response.response.header.code),
            int(response.response.status.value),
            str(response.response.message),
        )
        result = {
            "wall_utc_submitted": submitted,
            "wall_utc_returned": datetime.now(timezone.utc).isoformat(),
            "request": _json_safe(request),
            "response": payload,
            **diagnostics,
        }
        result["success"] = bool(diagnostics.get("accepted_by_service"))
        self.events.append({"event": "set_mc_action_post_stop_stand", **result})
        result["mode_readback_after_request"] = self.read_mode("post_stop_stand_reassert")
        return result

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
        # Freeze one byte-identical evidence input for both provenance checks.
        # The live MC log keeps appending warnings between reads; that is not
        # an executable/identity change. Live /proc verification remains strict.
        startup_snapshot = self.output_dir / "mc-startup-provenance.snapshot.log"
        with startup_snapshot.open("xb") as stream:
            stream.write((self.output_dir / "mc-console.log").read_bytes())
        try:
            mc_identity = load_verified_mc_runtime_identity(
                self.args.mc_sim_config_path,
                startup_snapshot,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"MC runtime identity could not be verified: {exc}") from exc
        if mc_identity.mc_config_sha256 != self.mc_sim_config_sha256:
            raise RuntimeError("MC runtime identity config hash differs from the loaded MC safety config")
        try:
            mc_subscriber_provenance = load_verified_mc_velocity_subscriber_provenance(
                self.args.mc_sim_config_path,
                startup_snapshot,
                mc_identity,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"MC velocity subscriber provenance could not be verified: {exc}") from exc
        dds_qos_override_environment = capture_dds_qos_override_environment()
        self.mc_runtime_identity = mc_identity
        discovery_started = time.monotonic()
        deadline = discovery_started + MC_VELOCITY_ENDPOINT_DISCOVERY_TIMEOUT_S
        graph_poll_count = 0
        last_publishers: list[str] = []
        last_subscribers: list[str] = []
        state_topic_publishers: dict[str, list[str]] = {}
        last_publisher_qos: dict[str, Any] | None = None
        last_subscriber_qos: list[dict[str, Any]] = []
        last_subscriber_endpoint_metadata: list[dict[str, Any]] = []
        fault = None
        while rclpy.ok() and time.monotonic() < deadline:
            graph_poll_count += 1
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
                    "graph_discovery_poll_count": graph_poll_count,
                    "graph_discovery_wait_s": time.monotonic() - discovery_started,
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
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            f"simulation-only transport isolation failed after "
            f"{time.monotonic() - discovery_started:.3f}s/{graph_poll_count} graph polls: {fault}; "
            f"MC startup provenance={mc_subscriber_provenance!r}; "
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
        speed_status = None
        if common is not None:
            try:
                bounds = {
                    "min_value": float(common.speed_status.forward_bounds.min_value),
                    "max_value": float(common.speed_status.forward_bounds.max_value),
                }
            except (AttributeError, TypeError, ValueError):
                bounds = None
            try:
                status = common.speed_status
                def _bounds(value: Any) -> dict[str, float]:
                    return {
                        "min_value": float(value.min_value),
                        "max_value": float(value.max_value),
                    }
                speed_status = {
                    "speed_mode": int(status.speed_mode.value),
                    "forward_bounds": _bounds(status.forward_bounds),
                    "lateral_bounds": _bounds(status.lateral_bounds),
                    "angular_bounds": _bounds(status.angular_bounds),
                }
            except (AttributeError, TypeError, ValueError):
                speed_status = None
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
            "speed_status": speed_status,
            "leg_states": self._joint_states_as_dicts(),
            "mc_leg_command": joint_command_array_snapshot(
                self.latest.get("leg_command"), self.received_at.get("leg_command")
            ),
            "mc_debug_f64_latest": self.latest_mc_debug_f64_snapshot,
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
                max_observed_planar_speed_m_s=getattr(self.args, "max_observed_planar_speed_m_s", None),
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
            "physical_state_summary": summarize_stationary_physical_state(samples),
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

    def run_ground_truth_point_goal(
        self,
        *,
        registration: Mapping[str, Any],
        set_action: Mapping[str, Any],
        stationary_baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Drive one short odom-frame point goal from MuJoCo ground truth with MC safety gates."""
        preflight = self.sample_for_first_motion()
        preflight_fault = snapshot_fault(
            preflight,
            self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            required_forward_mps=POINT_GOAL_COMMAND_M_S,
            max_observed_planar_speed_m_s=0.2,
        )
        if preflight_fault:
            return {"status": "BLOCKED", "fault": f"point_goal_preflight:{preflight_fault}"}
        if not stand_mode_readback_confirmed(preflight):
            return {"status": "BLOCKED", "fault": "point_goal_requires_fresh_stand_preflight"}
        start_pose = preflight.get("pose")
        if not isinstance(start_pose, Mapping):
            return {"status": "BLOCKED", "fault": "point_goal_preflight_pose_missing"}
        if start_pose.get("frame_id") != "odom" or start_pose.get("child_frame_id") != "base_link":
            return {
                "status": "BLOCKED",
                "fault": "point_goal_requires_odom_to_base_link_frame_pair",
                "observed_frame_pair": [start_pose.get("frame_id"), start_pose.get("child_frame_id")],
            }
        try:
            start_yaw = float(start_pose["yaw"])
            goal, stop_target = build_forward_point_goal(
                start_pose,
                distance_m=POINT_GOAL_DISTANCE_M,
                stop_compensation_m=POINT_GOAL_STOP_COMPENSATION_M,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "BLOCKED", "fault": f"point_goal_target:{type(exc).__name__}:{exc}"}
        latest: dict[str, Any] = {"snapshot": dict(preflight)}
        activation_tracker = ActivationTracker(expected_source=self.source)
        odom_stamp_monitor = SustainedYawRateWindow()
        goal_trace: list[dict[str, Any]] = []
        state: dict[str, Any] = {
            "inside_trigger_samples": 0,
            "goal_triggered": False,
            "fault": None,
            "last_metrics": None,
            "forward_bounds": None,
            "deceleration_start_monotonic_s": None,
            "deceleration_phase": "cruise",
        }
        command_trace: list[dict[str, Any]] = []

        def metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
            pose = snapshot.get("pose")
            if not isinstance(pose, Mapping):
                raise ValueError("point_goal_odometry_missing")
            if pose.get("frame_id") != "odom" or pose.get("child_frame_id") != "base_link":
                raise ValueError("point_goal_frame_pair_changed")
            dx = goal["x"] - float(pose["x"])
            dy = goal["y"] - float(pose["y"])
            distance = math.hypot(dx, dy)
            stop_dx = stop_target["x"] - float(pose["x"])
            stop_dy = stop_target["y"] - float(pose["y"])
            stop_distance = math.hypot(stop_dx, stop_dy)
            bearing = math.atan2(stop_dy, stop_dx)
            yaw = float(pose["yaw"])
            heading_error = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
            return {
                "distance_to_goal_m": distance,
                "distance_to_stop_target_m": stop_distance,
                "forward_error_m": dx * math.cos(start_yaw) + dy * math.sin(start_yaw),
                "lateral_error_m": -dx * math.sin(start_yaw) + dy * math.cos(start_yaw),
                "heading_error_rad": heading_error,
                "pose": dict(pose),
            }

        def on_goal_sample(phase: str, snapshot: Mapping[str, Any]) -> None:
            current = dict(snapshot)
            latest["snapshot"] = current
            activation_tracker.observe(current, phase="activation" if phase == "activation" else phase)
            stamp_rate = odom_stamp_monitor.update(current)
            try:
                current_metrics = metrics(current)
                state["last_metrics"] = current_metrics
                trace_row = {
                    "phase": phase,
                    "sample_monotonic_s": current.get("sample_monotonic_s"),
                    **current_metrics,
                    "source_stamp_diagnostic": {
                        "status": stamp_rate.get("status"),
                        "source_stamp_s": stamp_rate.get("source_stamp_s"),
                        "regression_count": odom_stamp_monitor.source_stamp_regression_count,
                        "duplicate_count": odom_stamp_monitor.source_stamp_duplicate_count,
                    },
                }
            except (KeyError, TypeError, ValueError) as exc:
                state["fault"] = f"point_goal_state:{type(exc).__name__}:{exc}"
                trace_row = {
                    "phase": phase,
                    "sample_monotonic_s": current.get("sample_monotonic_s"),
                    "fault": state["fault"],
                }
            goal_trace.append(trace_row)
            current["point_goal"] = trace_row
            self._trace_first_motion_sample(f"point-goal-{phase}", current)

        def publish_goal(forward_velocity_m_s: float) -> None:
            if forward_velocity_m_s <= 0.0:
                self._publish(0.0, 0.0, 0.0)
                return
            snapshot = latest.get("snapshot")
            if not isinstance(snapshot, Mapping):
                raise RuntimeError("point_goal_has_no_current_ground_truth_snapshot")
            point_metrics = metrics(snapshot)
            error = float(point_metrics["heading_error_rad"])
            sample_monotonic_s = float(snapshot["sample_monotonic_s"])
            forward_command, state["deceleration_start_monotonic_s"], state["deceleration_phase"] = (
                proximity_smoothstep_speed(
                    point_metrics["distance_to_stop_target_m"],
                    sample_monotonic_s,
                    state["deceleration_start_monotonic_s"],
                    deceleration_start_distance_m=POINT_GOAL_DECEL_START_DISTANCE_M,
                    initial_speed_m_s=POINT_GOAL_COMMAND_M_S,
                    terminal_speed_m_s=POINT_GOAL_TERMINAL_COMMAND_M_S,
                    deceleration_duration_s=POINT_GOAL_DECEL_DURATION_S,
                )
            )
            angular = 0.0
            speed_status = snapshot.get("speed_status")
            if not isinstance(speed_status, Mapping):
                raise RuntimeError("point_goal_live_speed_bounds_missing")
            forward_bounds = speed_status.get("forward_bounds")
            angular_bounds = speed_status.get("angular_bounds")
            if not isinstance(forward_bounds, Mapping) or not isinstance(angular_bounds, Mapping):
                raise RuntimeError("point_goal_live_speed_bounds_malformed")
            if not float(forward_bounds["min_value"]) <= forward_command <= float(forward_bounds["max_value"]):
                raise RuntimeError("point_goal_forward_command_outside_live_bounds")
            if not float(angular_bounds["min_value"]) <= angular <= float(angular_bounds["max_value"]):
                raise RuntimeError("point_goal_angular_command_outside_live_bounds")
            self._publish(forward_command, 0.0, angular)
            command_trace.append({
                "sample_monotonic_s": sample_monotonic_s,
                "forward_velocity_m_s": forward_command,
                "lateral_velocity_m_s": 0.0,
                "angular_velocity_rad_s": angular,
                "goal_distance_m": point_metrics["distance_to_goal_m"],
                "stop_target_distance_m": point_metrics["distance_to_stop_target_m"],
                "heading_error_rad": error,
                "profile_phase": state["deceleration_phase"],
                "deceleration_start_monotonic_s": state["deceleration_start_monotonic_s"],
                "control_handshake_confirmed": activation_tracker.control_handshake_confirmed,
            })

        def goal_reached(snapshot: Mapping[str, Any]) -> bool:
            if state.get("fault"):
                return True
            point_metrics = metrics(snapshot)
            if point_metrics["distance_to_stop_target_m"] <= POINT_GOAL_STOP_TRIGGER_M:
                state["inside_trigger_samples"] += 1
            else:
                state["inside_trigger_samples"] = 0
            if state["inside_trigger_samples"] >= POINT_GOAL_SETTLED_SAMPLES:
                state["goal_triggered"] = True
                return True
            return False

        activation = run_first_motion_window(
            sample=self.sample_for_first_motion,
            publish=publish_goal,
            registration=registration,
            set_action=set_action,
            joint_limits=self.joint_limits,
            baseline_forward_displacements_m=[
                float(window["forward_displacement_m"])
                for window in stationary_baseline.get("windows", [])
            ],
            baseline_state_summary=stationary_baseline.get("physical_state_summary"),
            authorized=True,
            forward_command_m_s=POINT_GOAL_COMMAND_M_S,
            pulse_duration_s=POINT_GOAL_MAX_DURATION_S,
            max_pulse_duration_s=POINT_GOAL_MAX_DURATION_S,
            extra_post_stop_observe_s=5.0,
            after_zero_stop=self.request_postwalk_stand_once,
            shutdown_event=self.interrupted,
            on_sample=on_goal_sample,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=0.2,
            zero_until_stand_timeout_s=3.5,
            termination_condition=goal_reached,
            termination_condition_name="ground_truth_goal_within_stop_trigger_for_three_samples",
        )
        final_metrics = state.get("last_metrics")
        final_distance = (
            float(final_metrics["distance_to_goal_m"])
            if isinstance(final_metrics, Mapping) else None
        )
        final_stop_target_distance = (
            float(final_metrics["distance_to_stop_target_m"])
            if isinstance(final_metrics, Mapping) else None
        )
        physical_stop = activation.get("physical_stop_gate", {})
        zero_stop = activation.get("zero_stop", {})
        acceptance_gates = {
            "compensated_stop_target_triggered_inside_radius": state.get("goal_triggered") is True,
            "final_distance_within_50mm": (
                final_distance is not None and final_distance <= POINT_GOAL_FINAL_TOLERANCE_M
            ),
            "termination_condition_triggered_stop": activation.get("termination_condition_met") is True,
            "measured_physical_motion": activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED",
            "same_sample_source_locomotion_handshake": activation.get("control_handshake_status") == "CONTROL_HANDSHAKE_CONFIRMED",
            "independent_physical_stop_gate": (
                isinstance(physical_stop, Mapping) and physical_stop.get("status") == "PASS"
            ),
            "zero_stop_stream_minimum": (
                isinstance(zero_stop, Mapping)
                and int(zero_stop.get("zero_publish_calls_returned", 0) or 0) >= ZERO_STOP_MIN_MESSAGES
            ),
            "no_controller_or_post_stop_fault": (
                activation.get("fault") is None and activation.get("post_stop_fault") is None
            ),
        }
        goal_pass = (
            all(acceptance_gates.values())
            and state.get("fault") is None
        )
        result = {
            "schema": "robotsim.issue80.point-goal-ground-truth.v1",
            "classification": "SIMULATION_ONLY_DIRECT_MUJOCO_GROUND_TRUTH_POINT_GOAL",
            "status": "PASS" if goal_pass else "BLOCKED",
            "goal_frame": "odom",
            "start_pose": dict(start_pose),
            "target_pose": goal,
            "compensated_stop_target_pose": stop_target,
            "distance_m": POINT_GOAL_DISTANCE_M,
            "stop_compensation_m": POINT_GOAL_STOP_COMPENSATION_M,
            "stop_trigger_radius_m": POINT_GOAL_STOP_TRIGGER_M,
            "final_tolerance_m": POINT_GOAL_FINAL_TOLERANCE_M,
            "required_consecutive_goal_samples": POINT_GOAL_SETTLED_SAMPLES,
            "forward_command_m_s": POINT_GOAL_COMMAND_M_S,
            "maximum_active_command_s": POINT_GOAL_MAX_DURATION_S,
            "command_profile": "straight-ahead point goal from measured initial heading; 0.14 m/s cruise, 1-second smoothstep to 0.112 m/s when within 0.12 m of the stop waypoint, then zero-only stop; no angular command",
            "goal_triggered": state.get("goal_triggered"),
            "stop_target_triggered": state.get("goal_triggered"),
            "final_distance_to_goal_m": final_distance,
            "final_distance_to_compensated_stop_target_m": final_stop_target_distance,
            "measured_stop_compensation_basis": {
                "source_attempt_id": "ceadcc39-350d-4318-ab91-2ae2d66f9c2f",
                "observed_backward_forward_projection_m": -0.0589638706430671,
                "command_end_speed_m_s": 0.112,
                "physical_stop_gate_status": "PASS",
                "status": "SIMULATION_ONLY_DERIVED_FROM_PRIOR_PHYSICAL_STOP_PASS",
            },
            "postwalk_stand_action": (
                "NOT_REQUESTED_ZERO_ONLY_STOP; wait for measured automatic transition and physical stop gate"
            ),
            "command_profile_parameters": {
                "cruise_speed_m_s": POINT_GOAL_COMMAND_M_S,
                "terminal_speed_m_s": POINT_GOAL_TERMINAL_COMMAND_M_S,
                "deceleration_start_distance_m": POINT_GOAL_DECEL_START_DISTANCE_M,
                "deceleration_duration_s": POINT_GOAL_DECEL_DURATION_S,
                "angular_velocity_rad_s": 0.0,
            },
            "activation_fault": activation.get("fault"),
            "post_stop_fault": activation.get("post_stop_fault"),
            "acceptance_gates": acceptance_gates,
            "controller_fault": state.get("fault"),
            "source_odometry_stamp_integrity": {
                "status": (
                    "UNAVAILABLE" if odom_stamp_monitor.source_stamp_sample_count == 0
                    else "NONMONOTONIC"
                    if odom_stamp_monitor.source_stamp_regression_count or odom_stamp_monitor.source_stamp_duplicate_count
                    else "MONOTONIC"
                ),
                "sample_count": odom_stamp_monitor.source_stamp_sample_count,
                "regression_count": odom_stamp_monitor.source_stamp_regression_count,
                "duplicate_count": odom_stamp_monitor.source_stamp_duplicate_count,
                "first_regression": odom_stamp_monitor.first_source_stamp_regression,
                "first_duplicate": odom_stamp_monitor.first_source_stamp_duplicate,
                "control_time_basis": "host_monotonic_receive_time",
            },
            "activation": activation,
            "goal_trace": goal_trace,
            "command_trace": command_trace,
            "limitations": [
                "SIMULATION_ONLY direct MC command path; no Nav2, localization, LiDAR, obstacle avoidance, or physical robot.",
                "Ground truth is the simulator-owned odometry stream; source header stamp integrity is reported separately.",
            ],
        }
        return result

    def run_turning_pair(
        self,
        *,
        registration: Mapping[str, Any],
        set_action: Mapping[str, Any],
        stationary_baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        baseline_displacements = [
            float(window["forward_displacement_m"])
            for window in stationary_baseline.get("windows", [])
        ]
        baseline_state_summary = stationary_baseline.get("physical_state_summary")
        directions = (("left", TURN_COMMAND_RAD_S), ("right", -TURN_COMMAND_RAD_S))
        forward_velocity = TURN_FORWARD_COMMAND_M_S
        turns = []
        for name, angular in directions:
            turn = run_angular_turn_window(
                sample=self.sample_for_first_motion,
                publish_turn=lambda forward, angular: self._publish(forward, 0.0, angular),
                publish_zero=lambda: self._publish(0.0, 0.0, 0.0),
                registration=registration,
                set_action=set_action,
                joint_limits=self.joint_limits,
                baseline_forward_displacements_m=baseline_displacements,
                baseline_state_summary=baseline_state_summary,
                authorized=True,
                angular_velocity_rad_s=angular,
                forward_velocity_m_s=forward_velocity,
                forward_warmup_s=TURN_FORWARD_WARMUP_S,
                pulse_duration_s=TURN_PULSE_S,
                max_observed_planar_speed_m_s=0.2,
                zero_until_stand_timeout_s=3.5,
                extra_post_stop_observe_s=5.0,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                shutdown_event=self.interrupted,
                on_sample=lambda phase, snapshot, side=name: self._trace_first_motion_sample(
                    f"turn-{side}-{phase}", snapshot,
                ),
            )
            turn["direction"] = name
            turns.append(turn)
            if turn.get("status") != "PHYSICAL_TURN_AND_STOP_PASS":
                break
        passed = len(turns) == len(directions) and all(
            turn.get("status") == "PHYSICAL_TURN_AND_STOP_PASS" for turn in turns
        )
        return {
            "status": "PASS" if passed else "BLOCKED",
            "classification": "SIMULATION_ONLY_DERIVED_FOUNDATIONWALK_TURNING",
            "directions": [turn.get("direction") for turn in turns],
            "forward_velocity_m_s": forward_velocity,
            "forward_warmup_s": TURN_FORWARD_WARMUP_S,
            "angular_command_rad_s": TURN_COMMAND_RAD_S,
            "pulse_duration_each_s": TURN_PULSE_S,
            "total_active_command_s": sum(
                float(turn.get("pulse_actual_duration_s", 0.0)) for turn in turns
            ),
            "turns": turns,
            "not_run_directions": [name for name, _ in directions[len(turns):]],
            "limitations": [
                "No foot contact force or MC stop acknowledgement is published by the installed runtime.",
                "Acceptance requires measured simulator yaw in both commanded directions and a passing PHYSICAL_STOP gate after each turn.",
            ],
        }

    def execute(self) -> dict[str, Any]:
        stationary_baseline: dict[str, Any] | None = None
        result: dict[str, Any] = {
            "schema": "robotsim.issue80.m1.first-motion.v2",
            "simulation_only": True,
            "acceptance_scope": (
                "SIMULATION_ONLY direct MuJoCo ground-truth point-goal; not Nav2, localization, LiDAR, or manufacturer-original acceptance"
                if getattr(self.args, "point_goal_diagnostic", False)
                else "SIMULATION_ONLY derived-configuration left/right turning and physical-stop validation; "
                "not manufacturer-original acceptance or point-goal navigation"
                if getattr(self.args, "turning_diagnostic", False)
                else "SIMULATION_ONLY derived-configuration locomotion and physical-stop validation; not manufacturer-original acceptance, turning, or point-goal navigation"
            ),
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
                "max_observed_planar_speed_m_s": getattr(self.args, "max_observed_planar_speed_m_s", None),
                "zero_until_stand_timeout_s": getattr(self.args, "zero_until_stand_timeout_s", 0.0),
            },
            "fault": None,
        }
        if getattr(self.args, "point_goal_diagnostic", False):
            result["forward_command_m_s"] = POINT_GOAL_COMMAND_M_S
            result["pulse_limit_s"] = POINT_GOAL_MAX_DURATION_S
        if getattr(self.args, "turning_diagnostic", False):
            result["forward_command_m_s"] = TURN_FORWARD_COMMAND_M_S
            result["forward_warmup_s"] = TURN_FORWARD_WARMUP_S
            result["pulse_limit_s"] = TURN_PULSE_S
            result["total_turn_command_window_s"] = TURN_FORWARD_WARMUP_S + TURN_PULSE_S
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

            if getattr(self.args, "point_goal_diagnostic", False):
                point_goal = self.run_ground_truth_point_goal(
                    registration=registration,
                    set_action=set_action,
                    stationary_baseline=stationary_baseline or {},
                )
                activation = point_goal.get("activation")
                result["point_goal_diagnostic"] = point_goal
                result["activation"] = activation
                result["nonzero_publish_calls_returned"] = (
                    int(activation.get("nonzero_publish_calls_returned", 0) or 0)
                    if isinstance(activation, Mapping) else 0
                )
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["physical_stop_gate"] = (
                    activation.get("physical_stop_gate") if isinstance(activation, Mapping) else None
                )
                result["legacy_baseline_stop_gate"] = (
                    activation.get("legacy_baseline_stop_gate") if isinstance(activation, Mapping) else None
                )
                result["get_mc_action_final"] = self.read_mode("final_point_goal_post_stop")
                result["get_current_input_source_final"] = self.read_input_source("final_point_goal_post_stop")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["fault"] = point_goal.get("controller_fault")
                result["status"] = "PASS" if point_goal.get("status") == "PASS" else "BLOCKED"
                return result

            if getattr(self.args, "turning_diagnostic", False):
                turning = self.run_turning_pair(
                    registration=registration,
                    set_action=set_action,
                    stationary_baseline=stationary_baseline or {},
                )
                self.pulse_result = turning
                result["turning_diagnostic"] = turning
                result["nonzero_publish_calls_returned"] = sum(
                    int(turn.get("nonzero_publish_calls_returned", 0) or 0)
                    for turn in turning.get("turns", [])
                )
                result["zero_publish_calls_returned"] = self.zero_publish_calls_returned
                result["mc_velocity_receipt_confirmed"] = False
                result["mc_stop_confirmed"] = False
                result["legacy_baseline_stop_gates"] = [
                    turn.get("legacy_baseline_stop_gate")
                    for turn in turning.get("turns", [])
                ]
                result["physical_stop_gates"] = [
                    turn.get("physical_stop_gate")
                    for turn in turning.get("turns", [])
                ]
                result["fault"] = next(
                    (turn.get("fault") for turn in turning.get("turns", []) if turn.get("fault")),
                    None,
                )
                result["get_mc_action_final"] = self.read_mode("final_turning_post_stop")
                result["get_current_input_source_final"] = self.read_input_source("final_turning_post_stop")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["status"] = "PASS" if turning.get("status") == "PASS" else "BLOCKED"
                return result

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
                baseline_state_summary=(
                    stationary_baseline.get("physical_state_summary")
                    if stationary_baseline else None
                ),
                authorized=True,
                forward_command_m_s=self.args.forward_speed,
                pulse_duration_s=self.args.pulse_duration,
                max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
                ramp_start_speed_m_s=self.args.ramp_start_speed,
                ramp_duration_s=self.args.ramp_duration_s,
                deceleration_duration_s=self.args.deceleration_duration_s,
                deceleration_end_speed_m_s=self.args.deceleration_end_speed_m_s,
                startup_speed_m_s=self.args.startup_speed,
                startup_transition_displacement_m=self.args.startup_transition_displacement,
                startup_transition_timeout_s=self.args.startup_transition_timeout_s,
                extra_post_stop_observe_s=self.args.extra_stop_observation,
                after_zero_stop=(
                    None if getattr(self.args, "zero_only_stop", False)
                    else self.request_postwalk_stand_once
                ),
                shutdown_event=self.interrupted,
                on_sample=self._trace_first_motion_sample,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=getattr(self.args, "max_observed_planar_speed_m_s", None),
                zero_until_stand_timeout_s=getattr(self.args, "zero_until_stand_timeout_s", 0.0),
            )
            self.pulse_result = pulse
            result["activation"] = pulse
            result["legacy_baseline_stop_gate"] = pulse.get("legacy_baseline_stop_gate")
            result["physical_stop_gate"] = pulse.get("physical_stop_gate")
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
                "PASS"
                if pulse.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
                and (
                    pulse.get("physical_stop_gate", {}).get("status") == "PASS"
                    or pulse.get("legacy_baseline_stop_gate", {}).get("status") == "WITHIN_BASELINE_ENVELOPE"
                )
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
                        baseline_state_summary=(
                            stationary_baseline.get("physical_state_summary")
                            if stationary_baseline else None
                        ),
                        authorized=True,
                        forward_command_m_s=self.args.forward_speed,
                        pulse_duration_s=self.args.pulse_duration,
                        max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
                        extra_post_stop_observe_s=self.args.extra_stop_observation,
                        after_zero_stop=(
                            None if getattr(self.args, "zero_only_stop", False)
                            else self.request_postwalk_stand_once
                        ),
                        shutdown_event=self.interrupted,
                        on_sample=self._trace_first_motion_sample,
                        state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z,
                        max_tilt_deg=self.args.max_tilt_deg,
                        max_observed_planar_speed_m_s=getattr(self.args, "max_observed_planar_speed_m_s", None),
                        zero_until_stand_timeout_s=getattr(self.args, "zero_until_stand_timeout_s", 0.0),
                    )
                except Exception as stop_exc:
                    result["exception_safe_stop_fault"] = f"{type(stop_exc).__name__}: {stop_exc}"
            result["final_common_state"] = _json_safe(self.latest.get("common"))
            result["final_odometry"] = self._pose()
            result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
        finally:
            result["events"] = self.events
            result["optional_debug_capture"] = {
                "mc_debug_f64_topic": MC_DEBUG_F64_TOPIC,
                "mc_debug_f64_type": "std_msgs/msg/Float64MultiArray",
                "mc_debug_f64_message_count": self.mc_debug_f64_message_count,
                "mc_debug_f64_first_layout": self.mc_debug_f64_first_layout,
                "mc_debug_f64_last_layout": (
                    self.latest_mc_debug_f64_snapshot.get("layout")
                    if self.latest_mc_debug_f64_snapshot else None
                ),
                "mc_rl_debug_topic": "/aima/mc/rl/debug",
                "mc_rl_debug_schema": (
                    "Available from the installed MC and simulator introspection libraries as "
                    "header + float32[] obs_vec + float32[] action_vec; this M1 node does not subscribe, "
                    "and the optional isolated sidecar records it without assigning vector-index semantics."
                ),
            }
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
    first.add_argument("--ramp-start-speed", type=float, default=None)
    first.add_argument("--ramp-duration-s", type=float, default=0.0)
    first.add_argument("--deceleration-duration-s", type=float, default=0.0)
    first.add_argument("--deceleration-end-speed-m-s", type=float, default=0.0)
    first.add_argument("--startup-speed", type=float, default=None)
    first.add_argument("--startup-transition-displacement", type=float, default=None)
    first.add_argument("--startup-transition-timeout-s", type=float, default=0.0)
    first.add_argument("--state-timeout", type=float, default=0.5)
    first.add_argument("--ready-timeout", type=float, default=4.0)
    first.add_argument("--max-tilt-deg", type=float, default=15.0)
    first.add_argument("--min-root-z", type=float, default=0.52)
    first.add_argument("--max-observed-planar-speed-m-s", type=float, default=None)
    first.add_argument("--zero-until-stand-timeout-s", type=float, default=0.0)
    first.add_argument("--zero-only-stop", action="store_true")
    first.add_argument("--turning-diagnostic", action="store_true")
    first.add_argument("--point-goal-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.turning_diagnostic and args.point_goal_diagnostic:
        parser.error("select only one of turning or point-goal diagnostics")
    if args.point_goal_diagnostic and (
        args.forward_speed != POINT_GOAL_COMMAND_M_S
        or args.pulse_duration != POINT_GOAL_MAX_DURATION_S
        or args.ramp_duration_s != 0.0
        or args.deceleration_duration_s != 0.0
        or args.startup_speed is not None
        or args.extra_stop_observation != 5.0
        or args.max_observed_planar_speed_m_s != 0.2
        or args.zero_until_stand_timeout_s != 3.5
        or not args.zero_only_stop
    ):
        parser.error(
            "point-goal diagnostics use the fixed 0.14 m/s, 10-second, no-ramp profile, "
            "zero-only stop, 0.2 m/s measured-speed guard, 3.5-second stand window, "
            "and 5-second observation"
        )
    if args.max_observed_planar_speed_m_s is not None and (
        not math.isfinite(args.max_observed_planar_speed_m_s)
        or not 0.0 < args.max_observed_planar_speed_m_s <= 0.2
    ):
        parser.error("observed planar speed gate must be positive and no greater than 0.2 m/s")
    if not math.isfinite(args.zero_until_stand_timeout_s) or not 0.0 <= args.zero_until_stand_timeout_s <= 3.5:
        parser.error("zero-until-stand timeout must remain within 0..3.5 seconds")
    try:
        return _focused_diagnostic_command(args)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
