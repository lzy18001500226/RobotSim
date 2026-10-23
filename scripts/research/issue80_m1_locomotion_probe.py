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
    MC_FSM_ACTIVE_STATES, MC_FSM_SAFE, MC_FSM_STABLE, MAX_AUTHORIZED_PULSE_S,
    MAX_EXTRA_POST_STOP_OBSERVE_S, PHYSICAL_STOP_MAX_YAW_RATE_RAD_S, verify_mc_fsm_runtime_contract,
    SOURCE_PRIORITY, SOURCE_TIMEOUT_MS, VELOCITY_SOURCE_NAME, classify_source_registration,
    TURN_COMMAND_RAD_S, TURN_FORWARD_COMMAND_M_S, TURN_FORWARD_WARMUP_S,
    TURN_MIN_HEADING_CHANGE_RAD, TURN_PULSE_S, ZERO_STOP_MIN_MESSAGES,
    SustainedYawRateWindow, run_angular_turn_window,
    capture_dds_qos_override_environment, derive_stationary_odom_baseline,
    float64_multi_array_snapshot, load_leg_joint_limits, load_mc_input_source_timeouts,
    joint_command_array_snapshot,
    VerifiedMcRuntimeIdentity, load_verified_mc_runtime_identity,
    load_verified_mc_velocity_subscriber_provenance, mc_subscriber_history_depth_evidence,
    build_forward_point_goal, classify_heading_goal_measurement,
    point_goal_command_from_bearing_error,
    point_goal_forward_speed_m_s,
    point_goal_heading_error_rad, point_goal_pose_angular_taper_factor,
    point_goal_pose_heading_error_rad,
    point_goal_stop_reason,
    speed_scaled_point_goal_stop_advance_m, run_first_motion_window,
    readback_refresh_needed,
    snapshot_fault, stale_readback_keys,
    stand_mode_readback_confirmed,
    summarize_stationary_physical_state,
    validate_local_sim_publishers, validate_first_motion_authorization,
    validate_runtime_scope,
)
from issue80_coordinated_pose_goal import (
    APPROACH,
    FAILED as POSE_GOAL_FAILED,
    TERMINAL_POSE_CONTROL,
    ZERO_STOP as POSE_GOAL_ZERO_STOP,
    PoseGoalPhaseController,
    POSE_GOAL_APPROACH_WARMUP_S,
    POSE_GOAL_CURVE_ANGULAR_UTILIZATION,
    POSE_GOAL_CURVE_FORWARD_M_S,
    POSE_GOAL_PATH_ENDPOINT_OFFSET_LIMIT_M,
    POSE_GOAL_WARMUP_FORWARD_M_S,
    POSE_GOAL_LOOKAHEAD_FRACTION,
    POSE_GOAL_TERMINAL_CONTROL_MAX_S,
    POSE_GOAL_TERMINAL_HEADING_GAIN,
    POSE_GOAL_TERMINAL_MIN_ANGULAR_RAD_S,
    POSE_GOAL_TERMINAL_RATE_DAMPING,
    POSE_GOAL_TERMINAL_READY_DWELL_S,
    POSE_GOAL_PRESTOP_YAW_MARGIN_RAD,
    POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD,
    POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD,
    POSE_GOAL_RUN95_POST_STOP_YAW_DRIFT_RAD,
    POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD,
    POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD,
    active_locomotion_readback_confirmed,
    build_warmup_aware_pose_goal_path,
    compensated_terminal_yaw_target_rad,
    fresh_yaw_rate_feedback,
    pose_goal_approach_path_reference,
    pose_path_tracking_yaw_rate_command,
    terminal_yaw_rate_command,
)
from issue80_m1_mode_logic import classify_set_mc_action_response

VELOCITY_TOPIC = COMMAND_PUBLISHER_TOPIC
MC_VELOCITY_ENDPOINT_DISCOVERY_TIMEOUT_S = 5.0
POINT_GOAL_DISTANCE_M = 0.30
POINT_GOAL_FINAL_TOLERANCE_M = 0.05
POINT_GOAL_COMMAND_M_S = 0.14
POINT_GOAL_POSE_CURVE_COMMAND_M_S = POSE_GOAL_CURVE_FORWARD_M_S
# Run77's 0.12 m/s steering profile left the measured gait targets nearly
# static after the first step sequence. Use the already validated 0.14 m/s
# profile during heading correction to test command-floor margin, still within
# the original M1 forward-command envelope.
POINT_GOAL_STEERING_COMMAND_M_S = POINT_GOAL_COMMAND_M_S
POINT_GOAL_MAX_DURATION_S = 10.0
POINT_GOAL_MAX_PLANAR_SPEED_M_S = 0.2
POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S = 0.075
POINT_GOAL_HEADING_GAIN = 0.8
# Run80's 0.15 m taper still missed the yaw stop gate by 0.13 deg. A saved-pose
# command-only replay showed 0.25 m start reduces angular command integral
# further; this is one bounded terminal-settling diagnostic, not a new goal
# tolerance or physical-stability threshold.
POINT_GOAL_ANGULAR_TAPER_START_ERROR_M = 0.25
POINT_GOAL_MAX_LATERAL_ERROR_M = 0.05
POINT_GOAL_MAX_OVERSHOOT_M = 0.05
POINT_GOAL_MAX_HEADING_CHANGE_RAD = 0.20
POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S = 5.0
POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S = 7.0
# The pinned community Task1 pose-goal example uses a 0.17 rad default yaw
# tolerance; this is an intermediate SIMULATION_ONLY heading-goal criterion.
POINT_GOAL_FINAL_YAW_TOLERANCE_RAD = 0.17
POINT_GOAL_HEADING_CHANGE_WARMUP_S = POSE_GOAL_APPROACH_WARMUP_S
POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S = POSE_GOAL_WARMUP_FORWARD_M_S
# Bounded two-run SIMULATION_ONLY fit: Run69 and Run73 endpoint shifts versus
# measured forward speed. It is only a stop predictor for this fixture, not a
# general or vendor FoundationWalk braking model.
POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M = 0.00347654
POINT_GOAL_ENDPOINT_RESPONSE_TIME_S = 0.09162162
# Run92 entered terminal control at 43.4 mm goal error with 84 mm/s forward
# speed, then crossed the unchanged 50 mm position gate before the 0.4 s pose
# dwell. Begin terminal control at the goal line to preserve the existing
# positional margin while the measured yaw response settles.
POINT_GOAL_HEADING_STOP_ADVANCE_M = 0.0
POINT_GOAL_HEADING_PREDICTED_STOP_TOLERANCE_M = 0.005
POINT_GOAL_MAX_YAW_RATE_FEEDBACK_AGE_S = 0.15
POINT_GOAL_POSE_LOOKAHEAD_FRACTION = POSE_GOAL_LOOKAHEAD_FRACTION
POINT_GOAL_POSE_TERMINAL_CONTROL_MAX_S = POSE_GOAL_TERMINAL_CONTROL_MAX_S
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
        # Keep mode and source service failures independent. A source query
        # timeout may occur while simulator-owned safety state is still fresh;
        # it must be auditable without poisoning every later readback forever.
        self.async_mode_readback_fault: str | None = None
        self.async_source_readback_fault: str | None = None
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
        # The one-second zero burst services its own ROS callbacks; refresh the
        # snapshot again before the safety gate so its cached odometry is not
        # falsely rejected as stale at the end of that bounded burst.
        snapshot = self.sample_for_first_motion()
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
                    self.async_mode_readback_fault = "GetMcAction returned no response"
                else:
                    recorded = self._record_mode_response(
                        self.mode_future_request, response, "activation_poll"
                    )
                    self.async_mode_readback_fault = (
                        None if recorded["response_header_code"] == 0
                        else f"GetMcAction header code={recorded['response_header_code']}"
                    )
            except Exception as exc:
                self.async_mode_readback_fault = f"GetMcAction async error:{type(exc).__name__}:{exc}"
            self.mode_future = None
            self.mode_future_request = None
        elif self.mode_future is not None and now - self.mode_future_started > 0.4:
            self.async_mode_readback_fault = "GetMcAction async response timeout"
            self.mode_future = None
            self.mode_future_request = None

        if self.source_future is not None and self.source_future.done():
            try:
                response = self.source_future.result()
                if response is None:
                    self.async_source_readback_fault = "GetCurrentInputSource returned no response"
                else:
                    recorded = self._record_source_response(
                        self.source_future_request, response, "activation_poll"
                    )
                    self.async_source_readback_fault = (
                        None if recorded["response_header_code"] == 0
                        else f"GetCurrentInputSource header code={recorded['response_header_code']}"
                    )
            except Exception as exc:
                self.async_source_readback_fault = f"GetCurrentInputSource async error:{type(exc).__name__}:{exc}"
            self.source_future = None
            self.source_future_request = None
        elif self.source_future is not None and now - self.source_future_started > 0.4:
            self.async_source_readback_fault = "GetCurrentInputSource async response timeout"
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
            "get_mode_readback_fault": self.async_mode_readback_fault,
            "get_source_readback_fault": self.async_source_readback_fault,
            "readback_fault": "; ".join(
                fault for fault in (
                    self.async_mode_readback_fault,
                    self.async_source_readback_fault,
                ) if fault
            ) or None,
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
        started = time.monotonic()
        self._spin_for(0.005)
        snapshot = self._first_motion_snapshot()
        stale = stale_readback_keys(snapshot, float(self.args.state_timeout))
        refresh_faults: list[dict[str, Any]] = []
        refresh_deadline = time.monotonic() + 0.12
        while readback_refresh_needed(snapshot, float(self.args.state_timeout)) and time.monotonic() < refresh_deadline:
            if snapshot.get("readback_fault"):
                refresh_faults.append({
                    "sample_monotonic_s": float(snapshot.get("sample_monotonic_s", time.monotonic())),
                    "readback_fault": str(snapshot["readback_fault"]),
                    "stale_keys": list(stale),
                })
            self._spin_for(0.01)
            snapshot = self._first_motion_snapshot()
            stale = stale_readback_keys(snapshot, float(self.args.state_timeout))
        snapshot["readback_refresh_wait_s"] = time.monotonic() - started
        snapshot["readback_refresh_stale_keys"] = list(stale)
        snapshot["readback_refresh_faults_seen"] = refresh_faults
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
        heading_change_rad: float = 0.0,
    ) -> dict[str, Any]:
        """Drive one short odom-frame point goal from MuJoCo ground truth with MC safety gates."""
        heading_control_policy = self.args.heading_control_policy
        final_yaw_tolerance_rad = float(self.args.final_yaw_tolerance_rad)
        heading_goal_stop_advance_m = self.args.heading_goal_stop_advance_m
        heading_goal_predicted_stop_tolerance_m = self.args.heading_goal_predicted_stop_tolerance_m
        has_heading_target = abs(float(heading_change_rad)) > 1e-9
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
            heading_change_rad = float(heading_change_rad)
            if (
                not math.isfinite(heading_change_rad)
                or abs(heading_change_rad) > POINT_GOAL_MAX_HEADING_CHANGE_RAD
            ):
                raise ValueError(
                    f"heading change exceeds +/-{POINT_GOAL_MAX_HEADING_CHANGE_RAD:.3f} rad"
                )
            goal = build_forward_point_goal(
                start_pose,
                distance_m=POINT_GOAL_DISTANCE_M,
                heading_change_rad=heading_change_rad,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "BLOCKED", "fault": f"point_goal_target:{type(exc).__name__}:{exc}"}
        path_yaw = float(goal["yaw"])
        pose_goal_approach_path = None
        pose_goal_path_feasibility = None
        if has_heading_target:
            try:
                stop_advance = float(heading_goal_stop_advance_m)
                stop_pose = {
                    "frame_id": "odom",
                    "x": float(goal["x"]) + stop_advance * math.cos(path_yaw),
                    "y": float(goal["y"]) + stop_advance * math.sin(path_yaw),
                }
                pose_goal_approach_path = build_warmup_aware_pose_goal_path(
                    start_pose,
                    stop_pose,
                    path_yaw,
                    POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD,
                    terminal_pre_stop_yaw_bias_rad=POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD,
                    warmup_duration_s=POINT_GOAL_HEADING_CHANGE_WARMUP_S,
                    warmup_forward_m_s=POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S,
                    curve_forward_m_s=POINT_GOAL_POSE_CURVE_COMMAND_M_S,
                    maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    maximum_endpoint_offset_m=POSE_GOAL_PATH_ENDPOINT_OFFSET_LIMIT_M,
                    angular_utilization_limit=POSE_GOAL_CURVE_ANGULAR_UTILIZATION,
                    lookahead_fraction=POINT_GOAL_POSE_LOOKAHEAD_FRACTION,
                )
                pose_goal_path_feasibility = pose_goal_approach_path["feasibility"]
                if (
                    pose_goal_path_feasibility["endpoint_position_error_m"] > POINT_GOAL_FINAL_TOLERANCE_M
                    or pose_goal_path_feasibility["maximum_lateral_error_to_requested_stop_line_m"]
                    > POINT_GOAL_MAX_LATERAL_ERROR_M
                    or not pose_goal_path_feasibility["terminal_residual_and_dwell_fit"]
                ):
                    return {
                        "status": "BLOCKED",
                        "fault": "pose_goal_warmup_aware_path_exceeds_existing_geometry_or_terminal_limits",
                        "pose_goal_geometry_preflight": pose_goal_path_feasibility,
                        "pose_goal_approach_path": pose_goal_approach_path,
                    }
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "status": "BLOCKED",
                    "fault": f"pose_goal_path_preflight:{type(exc).__name__}:{exc}",
                }
        latest: dict[str, Any] = {"snapshot": dict(preflight)}
        activation_tracker = ActivationTracker(expected_source=self.source)
        odom_stamp_monitor = SustainedYawRateWindow()
        pose_controller = (
            PoseGoalPhaseController(
                goal_distance_m=POINT_GOAL_DISTANCE_M,
                final_position_tolerance_m=POINT_GOAL_FINAL_TOLERANCE_M,
                maximum_lateral_error_m=POINT_GOAL_MAX_LATERAL_ERROR_M,
                maximum_overshoot_m=POINT_GOAL_MAX_OVERSHOOT_M,
                final_yaw_tolerance_rad=final_yaw_tolerance_rad,
                terminal_yaw_setpoint_tolerance_rad=POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD,
            ) if has_heading_target else None
        )
        goal_trace: list[dict[str, Any]] = []
        state: dict[str, Any] = {
            "goal_triggered": False,
            "target_crossing_detected": False,
            "stop_reason": None,
            "stop_advance_at_trigger_m": None,
            "measured_forward_speed_at_trigger_m_s": None,
            "stop_progress_at_trigger_m": None,
            "stop_lateral_error_at_trigger_m": None,
            "metrics_at_trigger": None,
            "fault": None,
            "last_metrics": None,
            "activation_started_monotonic_s": None,
            "yaw_rate_feedback": None,
            "terminal_pose_ready": not has_heading_target,
            "pose_goal_phase_history": [],
            "pose_goal_position_triggered": False,
            "pose_goal_terminal_control_samples": 0,
            "pose_goal_approach_path": pose_goal_approach_path,
            "pose_goal_geometry_preflight": pose_goal_path_feasibility,
        }
        command_trace: list[dict[str, Any]] = []

        def metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
            pose = snapshot.get("pose")
            if not isinstance(pose, Mapping):
                raise ValueError("point_goal_odometry_missing")
            if pose.get("frame_id") != "odom" or pose.get("child_frame_id") != "base_link":
                raise ValueError("point_goal_frame_pair_changed")
            x0, y0 = float(start_pose["x"]), float(start_pose["y"])
            x, y = float(pose["x"]), float(pose["y"])
            heading_x, heading_y = math.cos(path_yaw), math.sin(path_yaw)
            dx, dy = goal["x"] - x, goal["y"] - y
            distance = math.hypot(dx, dy)
            progress = (x - x0) * heading_x + (y - y0) * heading_y
            lateral = -(x - x0) * heading_y + (y - y0) * heading_x
            measured_forward_speed = point_goal_forward_speed_m_s(
                pose, start_yaw_rad=path_yaw,
            )
            if has_heading_target:
                stop_advance = float(heading_goal_stop_advance_m)
                stop_model = "run92_measured_terminal_entry_position_margin"
            else:
                stop_advance = speed_scaled_point_goal_stop_advance_m(
                    measured_forward_speed,
                    endpoint_response_intercept_m=POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M,
                    endpoint_response_time_s=POINT_GOAL_ENDPOINT_RESPONSE_TIME_S,
                    maximum_forward_speed_m_s=POINT_GOAL_MAX_PLANAR_SPEED_M_S,
                )
                stop_model = "speed_scaled_endpoint_response"
            stop_progress = POINT_GOAL_DISTANCE_M + stop_advance
            stop_target = {
                "frame_id": "odom",
                "x": goal["x"] + stop_advance * heading_x,
                "y": goal["y"] + stop_advance * heading_y,
            }
            stop_dx = stop_target["x"] - x
            stop_dy = stop_target["y"] - y
            stop_distance = math.hypot(stop_dx, stop_dy)
            predicted_settled_error = math.hypot(stop_progress - progress, lateral)
            yaw = float(pose["yaw"])
            # Restore Run73's waypoint-bearing policy: aim at the measured
            # ground-truth goal from the current pose. Keep the newer dynamic
            # predicted-settled endpoint solely in the stop decision below.
            heading_error = point_goal_heading_error_rad(
                current_yaw_rad=yaw,
                goal_dx_m=dx,
                goal_dy_m=dy,
            )
            return {
                "distance_to_goal_m": distance,
                "distance_to_dynamic_stop_target_m": stop_distance,
                "along_track_progress_m": progress,
                "remaining_along_track_to_goal_m": POINT_GOAL_DISTANCE_M - progress,
                "lateral_error_m": lateral,
                "measured_forward_speed_m_s": measured_forward_speed,
                "speed_scaled_stop_advance_m": stop_advance,
                "dynamic_stop_progress_m": stop_progress,
                "stop_model": stop_model,
                "predicted_settled_goal_error_m": predicted_settled_error,
                "target_crossing_detected": progress >= POINT_GOAL_DISTANCE_M,
                "dynamic_stop_line_crossed": progress >= stop_progress,
                "dynamic_stop_target_pose": stop_target,
                "heading_error_rad": heading_error,
                "goal_dx_m": dx,
                "goal_dy_m": dy,
                "pose": dict(pose),
            }

        def on_goal_sample(phase: str, snapshot: Mapping[str, Any]) -> None:
            current = dict(snapshot)
            latest["snapshot"] = current
            if phase == "activation" and state["activation_started_monotonic_s"] is None:
                state["activation_started_monotonic_s"] = float(
                    current.get("sample_monotonic_s", time.monotonic())
                )
            activation_tracker.observe(current, phase="activation" if phase == "activation" else phase)
            stamp_rate = odom_stamp_monitor.update(current)
            if stamp_rate.get("status") == "MEASURED":
                state["yaw_rate_feedback"] = {
                    "signed_net_yaw_rate_rad_s": float(stamp_rate["signed_net_yaw_rate_rad_s"]),
                    "sample_monotonic_s": float(current["sample_monotonic_s"]),
                    "window_s": float(stamp_rate["window_s"]),
                    "coverage_s": float(stamp_rate["coverage_s"]),
                }
            elif stamp_rate.get("status") == "HOST_MONOTONIC_RESET":
                state["yaw_rate_feedback"] = None
            try:
                current_metrics = metrics(current)
                state["last_metrics"] = current_metrics
                trace_row = {
                    "phase": phase,
                    "sample_monotonic_s": current.get("sample_monotonic_s"),
                    "pose_goal_phase": pose_controller.phase if pose_controller is not None else "POSITION_ONLY",
                    **current_metrics,
                    "source_stamp_diagnostic": {
                        "status": stamp_rate.get("status"),
                        "source_stamp_s": stamp_rate.get("source_stamp_s"),
                        "signed_yaw_rate_rad_s": stamp_rate.get("signed_net_yaw_rate_rad_s"),
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
            pose = snapshot.get("pose")
            if not isinstance(pose, Mapping):
                raise RuntimeError("point_goal_has_no_current_pose_for_steering")
            position_bearing_error = float(point_metrics["heading_error_rad"])
            target_yaw_error = point_goal_pose_heading_error_rad(
                current_yaw_rad=float(pose["yaw"]),
                goal_dx_m=float(point_metrics["goal_dx_m"]),
                goal_dy_m=float(point_metrics["goal_dy_m"]),
                target_yaw_rad=path_yaw if has_heading_target else None,
            )
            sample_monotonic_s = float(snapshot["sample_monotonic_s"])
            activation_started = state.get("activation_started_monotonic_s")
            activation_elapsed = (
                max(0.0, sample_monotonic_s - float(activation_started))
                if activation_started is not None else 0.0
            )
            yaw_rate_feedback = state.get("yaw_rate_feedback")
            rate_feedback = None
            if has_heading_target and isinstance(yaw_rate_feedback, Mapping):
                try:
                    rate_feedback = fresh_yaw_rate_feedback(
                        yaw_rate_feedback,
                        sample_monotonic_s=sample_monotonic_s,
                        maximum_age_s=POINT_GOAL_MAX_YAW_RATE_FEEDBACK_AGE_S,
                    )
                except ValueError as exc:
                    state["fault"] = f"pose_goal_yaw_rate_feedback:{exc}"
                    self._publish(0.0, 0.0, 0.0)
                    return
            pose_phase = pose_controller.phase if pose_controller is not None else "POSITION_ONLY"
            path_reference = None
            carrot_reference = None
            path_tracking = None
            approach_yaw_target = None
            heading_warmup_active = (
                has_heading_target
                and pose_phase == APPROACH
                and activation_elapsed < POINT_GOAL_HEADING_CHANGE_WARMUP_S
            )
            angular_taper_factor = 1.0 if has_heading_target else point_goal_pose_angular_taper_factor(
                float(point_metrics["predicted_settled_goal_error_m"]),
                has_heading_target=False,
                taper_start_error_m=POINT_GOAL_ANGULAR_TAPER_START_ERROR_M,
                final_tolerance_m=POINT_GOAL_FINAL_TOLERANCE_M,
            )
            if has_heading_target and pose_phase == TERMINAL_POSE_CONTROL:
                terminal_yaw_target = compensated_terminal_yaw_target_rad(
                    path_yaw,
                    POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD,
                )
                terminal_control = terminal_yaw_rate_command(
                    current_yaw_rad=float(pose["yaw"]),
                    target_yaw_rad=terminal_yaw_target,
                    measured_yaw_rate_rad_s=float(rate_feedback["signed_yaw_rate_rad_s"]),
                    maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    final_yaw_tolerance_rad=POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD,
                )
                forward_command = 0.0
                angular = float(terminal_control["command_yaw_rate_rad_s"])
                state["pose_goal_terminal_control_samples"] += 1
                control_heading_error = float(terminal_control["yaw_error_rad"])
            elif has_heading_target:
                dynamic_stop_pose = point_metrics["dynamic_stop_target_pose"]
                path_model = pose_goal_approach_path
                if not isinstance(path_model, Mapping):
                    state["fault"] = "pose_goal_approach_path_missing_after_preflight"
                    self._publish(0.0, 0.0, 0.0)
                    return
                if (
                    math.hypot(
                        float(path_model["requested_stop_pose"]["x"]) - float(dynamic_stop_pose["x"]),
                        float(path_model["requested_stop_pose"]["y"]) - float(dynamic_stop_pose["y"]),
                    ) > 1e-6
                ):
                    state["fault"] = "pose_goal_dynamic_stop_target_changed_after_geometry_preflight"
                    self._publish(0.0, 0.0, 0.0)
                    return
                approach_yaw_target = float(path_model["curve_endpoint_pose"]["yaw"])
                path_distance = float(path_model["projected_stop_progress_m"])
                progress_fraction = max(
                    0.0,
                    min(1.0, float(point_metrics["along_track_progress_m"]) / path_distance),
                )
                path_reference = pose_goal_approach_path_reference(
                    start_pose, path_model, progress_fraction,
                )
                carrot_reference = pose_goal_approach_path_reference(
                    start_pose, path_model,
                    min(1.0, progress_fraction + POINT_GOAL_POSE_LOOKAHEAD_FRACTION),
                )
                forward_command = (
                    POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S
                    if heading_warmup_active else POINT_GOAL_POSE_CURVE_COMMAND_M_S
                )
                if heading_warmup_active:
                    angular = 0.0
                    control_heading_error = math.atan2(
                        math.sin(path_reference["tangent_yaw_rad"] - float(pose["yaw"])),
                        math.cos(path_reference["tangent_yaw_rad"] - float(pose["yaw"])),
                    )
                else:
                    path_tracking = pose_path_tracking_yaw_rate_command(
                        current_yaw_rad=float(pose["yaw"]),
                        path_tangent_yaw_rad=float(path_reference["tangent_yaw_rad"]),
                        measured_yaw_rate_rad_s=float(rate_feedback["signed_yaw_rate_rad_s"]),
                        path_curvature_per_m=float(path_reference["curvature_per_m"]),
                        forward_command_m_s=forward_command,
                        maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    )
                    angular = float(path_tracking["command_yaw_rate_rad_s"])
                    control_heading_error = float(path_tracking["path_yaw_error_rad"])
            else:
                control_heading_error = float(point_metrics["heading_error_rad"])
                forward_command, angular = point_goal_command_from_bearing_error(
                    control_heading_error,
                    cruise_forward_m_s=POINT_GOAL_COMMAND_M_S,
                    steering_forward_m_s=POINT_GOAL_STEERING_COMMAND_M_S,
                    heading_gain=POINT_GOAL_HEADING_GAIN,
                    maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                )
                angular *= angular_taper_factor

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
                "along_track_progress_m": point_metrics["along_track_progress_m"],
                "lateral_error_m": point_metrics["lateral_error_m"],
                "measured_forward_speed_m_s": point_metrics["measured_forward_speed_m_s"],
                "speed_scaled_stop_advance_m": point_metrics["speed_scaled_stop_advance_m"],
                "dynamic_stop_progress_m": point_metrics["dynamic_stop_progress_m"],
                "dynamic_stop_line_crossed": point_metrics["dynamic_stop_line_crossed"],
                "heading_error_rad": control_heading_error,
                "instantaneous_heading_error_rad": target_yaw_error,
                "position_bearing_error_rad": position_bearing_error,
                "heading_reference": "coordinated_hermite_path" if has_heading_target else "Run73_position_bearing",
                "control_phase": pose_phase,
                "target_yaw_error_rad": target_yaw_error if has_heading_target else None,
                "terminal_yaw_target_rad": (
                    terminal_yaw_target if has_heading_target and pose_phase == TERMINAL_POSE_CONTROL else None
                ),
                "terminal_yaw_setpoint_error_rad": (
                    float(terminal_control["yaw_error_rad"])
                    if has_heading_target and pose_phase == TERMINAL_POSE_CONTROL else None
                ),
                "approach_path_endpoint_yaw_rad": approach_yaw_target,
                "yaw_rate_feedback_rad_s": (
                    float(rate_feedback["signed_yaw_rate_rad_s"])
                    if rate_feedback is not None else None
                ),
                "yaw_rate_feedback_age_s": rate_feedback["age_s"] if rate_feedback is not None else None,
                "path_reference_yaw_rad": path_reference["tangent_yaw_rad"] if path_reference else None,
                "path_reference_curvature_per_m": path_reference["curvature_per_m"] if path_reference else None,
                "path_curvature_feedforward_rad_s": (
                    float(path_tracking["curvature_feedforward_rad_s"]) if path_tracking else None
                ),
                "path_heading_feedback_rad_s": (
                    float(path_tracking["heading_rate_feedback_rad_s"]) if path_tracking else None
                ),
                "path_raw_yaw_rate_command_rad_s": (
                    float(path_tracking["raw_yaw_rate_command_rad_s"]) if path_tracking else None
                ),
                "path_yaw_rate_command_saturated": (
                    bool(path_tracking["command_saturated"]) if path_tracking else None
                ),
                "path_progress_fraction": path_reference["progress_fraction"] if path_reference else None,
                "lookahead_x_m": carrot_reference["x_m"] if carrot_reference else None,
                "lookahead_y_m": carrot_reference["y_m"] if carrot_reference else None,
                "terminal_phase_elapsed_s": (
                    sample_monotonic_s - pose_controller.terminal_started_monotonic_s
                    if pose_controller is not None and pose_controller.terminal_started_monotonic_s is not None else None
                ),
                "activation_elapsed_s": activation_elapsed,
                "heading_turn_warmup_active": heading_warmup_active,
                "predicted_settled_goal_error_m": point_metrics["predicted_settled_goal_error_m"],
                "angular_taper_factor": angular_taper_factor,
                "control_handshake_confirmed": activation_tracker.control_handshake_confirmed,
                "pose_goal_pre_stop_yaw_bias_rad": (
                    POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD if has_heading_target else None
                ),
                "pose_goal_approach_endpoint_offset_m": (
                    pose_goal_approach_path.get("endpoint_position_offset_m")
                    if has_heading_target and isinstance(pose_goal_approach_path, Mapping) else None
                ),
                "pose_goal_path_max_curvature_per_m": (
                    pose_goal_path_feasibility.get("maximum_absolute_curvature_per_m")
                    if has_heading_target and isinstance(pose_goal_path_feasibility, Mapping) else None
                ),
                "pose_goal_path_required_angular_rate_rad_s": (
                    pose_goal_path_feasibility.get("required_angular_rate_at_forward_command_rad_s")
                    if has_heading_target and isinstance(pose_goal_path_feasibility, Mapping) else None
                ),
            })

        def goal_reached(snapshot: Mapping[str, Any]) -> bool:
            if state.get("fault"):
                return True
            point_metrics = metrics(snapshot)
            progress = float(point_metrics["along_track_progress_m"])
            state["target_crossing_detected"] = (
                state["target_crossing_detected"] or bool(point_metrics["target_crossing_detected"])
            )
            stop_reason = point_goal_stop_reason(
                along_track_progress_m=progress,
                lateral_error_m=float(point_metrics["lateral_error_m"]),
                goal_distance_m=POINT_GOAL_DISTANCE_M,
                dynamic_stop_progress_m=float(point_metrics["dynamic_stop_progress_m"]),
                final_tolerance_m=POINT_GOAL_FINAL_TOLERANCE_M,
                predicted_settled_tolerance_m=(
                    float(heading_goal_predicted_stop_tolerance_m)
                    if has_heading_target else POINT_GOAL_FINAL_TOLERANCE_M
                ),
                maximum_lateral_error_m=POINT_GOAL_MAX_LATERAL_ERROR_M,
                maximum_overshoot_m=POINT_GOAL_MAX_OVERSHOOT_M,
            )
            if has_heading_target:
                pose = snapshot.get("pose")
                if not isinstance(pose, Mapping) or pose_controller is None:
                    state["fault"] = "pose_goal_controller_state_missing"
                    return True
                target_yaw_error = math.atan2(
                    math.sin(path_yaw - float(pose["yaw"])),
                    math.cos(path_yaw - float(pose["yaw"])),
                )
                terminal_yaw_target = compensated_terminal_yaw_target_rad(
                    path_yaw,
                    POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD,
                )
                terminal_setpoint_error = math.atan2(
                    math.sin(terminal_yaw_target - float(pose["yaw"])),
                    math.cos(terminal_yaw_target - float(pose["yaw"])),
                )
                feedback = state.get("yaw_rate_feedback")
                measured_yaw_rate = None
                terminal_entry = (
                    pose_controller.phase == APPROACH
                    and stop_reason in {
                        "speed_scaled_stop_line_crossed",
                        "goal_approach_predicted_settle_within_tolerance",
                    }
                )
                if pose_controller.phase == TERMINAL_POSE_CONTROL or terminal_entry:
                    if not active_locomotion_readback_confirmed(
                        snapshot,
                        expected_source=self.source,
                        control_handshake_confirmed=activation_tracker.control_handshake_confirmed,
                    ):
                        pose_controller.failure_reason = "pose_goal_terminal_source_or_mode_lost"
                        pose_controller.phase = POSE_GOAL_FAILED
                    else:
                        try:
                            rate = fresh_yaw_rate_feedback(
                                feedback,
                                sample_monotonic_s=float(snapshot["sample_monotonic_s"]),
                                maximum_age_s=POINT_GOAL_MAX_YAW_RATE_FEEDBACK_AGE_S,
                            )
                            measured_yaw_rate = rate["signed_yaw_rate_rad_s"]
                        except (KeyError, TypeError, ValueError):
                            measured_yaw_rate = None
                decision = pose_controller.update(
                    monotonic_s=float(snapshot["sample_monotonic_s"]),
                    position_stop_reason=stop_reason,
                    distance_to_goal_m=float(point_metrics["distance_to_goal_m"]),
                    lateral_error_m=float(point_metrics["lateral_error_m"]),
                    along_track_progress_m=progress,
                    yaw_error_rad=target_yaw_error,
                    measured_yaw_rate_rad_s=measured_yaw_rate,
                    terminal_setpoint_error_rad=terminal_setpoint_error,
                )
                history = state["pose_goal_phase_history"]
                if not history or history[-1]["phase"] != decision["phase"]:
                    history.append({
                        "phase": decision["phase"],
                        "sample_monotonic_s": float(snapshot["sample_monotonic_s"]),
                        "position_stop_reason": decision.get("position_trigger_reason"),
                        "failure_reason": decision.get("failure_reason"),
                    })
                if decision.get("transitioned"):
                    state["goal_triggered"] = True
                    state["pose_goal_position_triggered"] = True
                    state["stop_reason"] = decision["position_trigger_reason"]
                    if state.get("metrics_at_trigger") is None:
                        state["stop_advance_at_trigger_m"] = point_metrics["speed_scaled_stop_advance_m"]
                        state["measured_forward_speed_at_trigger_m_s"] = point_metrics["measured_forward_speed_m_s"]
                        state["stop_progress_at_trigger_m"] = point_metrics["dynamic_stop_progress_m"]
                        state["stop_lateral_error_at_trigger_m"] = point_metrics["lateral_error_m"]
                        state["metrics_at_trigger"] = dict(point_metrics)
                if decision.get("terminal_pose_ready"):
                    state["terminal_pose_ready"] = True
                    state["stop_reason"] = "coordinated_pose_terminal_ready"
                if decision.get("failure_reason"):
                    state["fault"] = str(decision["failure_reason"])
                    state["stop_reason"] = str(decision["failure_reason"])
                if decision["complete"]:
                    state["goal_triggered"] = state["goal_triggered"] or state["pose_goal_position_triggered"]
                    return True
                return False

            if stop_reason is not None and stop_reason not in {
                "speed_scaled_stop_line_crossed",
                "goal_approach_predicted_settle_within_tolerance",
            }:
                state["fault"] = f"point_goal_{stop_reason}"
                state["stop_reason"] = stop_reason
                return True
            if stop_reason in {
                "speed_scaled_stop_line_crossed",
                "goal_approach_predicted_settle_within_tolerance",
            }:
                state["goal_triggered"] = True
                state["stop_reason"] = stop_reason
                state["stop_advance_at_trigger_m"] = point_metrics["speed_scaled_stop_advance_m"]
                state["measured_forward_speed_at_trigger_m_s"] = point_metrics["measured_forward_speed_m_s"]
                state["stop_progress_at_trigger_m"] = point_metrics["dynamic_stop_progress_m"]
                state["stop_lateral_error_at_trigger_m"] = point_metrics["lateral_error_m"]
                state["metrics_at_trigger"] = dict(point_metrics)
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
            extra_post_stop_observe_s=self.args.extra_stop_observation,
            extra_post_stop_observe_limit_s=(
                MAX_EXTRA_POST_STOP_OBSERVE_S
                if has_heading_target else 5.0
            ),
            after_zero_stop=self.request_postwalk_stand_once,
            during_zero_callback=lambda: self._spin_for(0.001),
            shutdown_event=self.interrupted,
            on_sample=on_goal_sample,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=0.2,
            zero_until_stand_timeout_s=3.5,
            termination_condition=goal_reached,
            termination_condition_name=(
                "coordinated_pose_terminal_yaw_and_position_settled_or_bounded_failure"
                if has_heading_target else "ground_truth_predicted_goal_approach_or_speed_scaled_stop_line"
            ),
        )
        final_metrics = state.get("last_metrics")
        final_distance = (
            float(final_metrics["distance_to_goal_m"])
            if isinstance(final_metrics, Mapping) else None
        )
        final_stop_target_distance = (
            float(final_metrics["distance_to_dynamic_stop_target_m"])
            if isinstance(final_metrics, Mapping) else None
        )
        final_pose = final_metrics.get("pose") if isinstance(final_metrics, Mapping) else None
        final_yaw = (
            float(final_pose["yaw"])
            if isinstance(final_pose, Mapping) and final_pose.get("yaw") is not None else None
        )
        heading_measurement = (
            classify_heading_goal_measurement(
                start_yaw_rad=start_yaw,
                final_yaw_rad=final_yaw,
                target_yaw_rad=path_yaw,
                requested_heading_change_rad=heading_change_rad,
                minimum_measured_change_rad=TURN_MIN_HEADING_CHANGE_RAD,
                final_yaw_tolerance_rad=final_yaw_tolerance_rad,
            ) if final_yaw is not None else None
        )
        final_yaw_change = (
            heading_measurement["measured_heading_change_rad"]
            if heading_measurement else None
        )
        final_yaw_error = (
            heading_measurement["final_yaw_error_rad"] if heading_measurement else None
        )
        measured_heading_change_gate = (
            heading_measurement["measured_heading_change_in_requested_direction"]
            if heading_measurement else abs(heading_change_rad) <= 1e-9
        )
        final_yaw_tolerance_gate = (
            heading_measurement["final_yaw_within_tolerance"]
            if heading_measurement else abs(heading_change_rad) <= 1e-9
        )
        physical_stop = activation.get("physical_stop_gate", {})
        zero_stop = activation.get("zero_stop", {})
        acceptance_gates = {
            "goal_approach_or_stop_line_triggered": state.get("goal_triggered") is True,
            "final_distance_within_50mm": (
                final_distance is not None and final_distance <= POINT_GOAL_FINAL_TOLERANCE_M
            ),
            "lateral_error_within_50mm": (
                isinstance(final_metrics, Mapping)
                and abs(float(final_metrics["lateral_error_m"])) <= POINT_GOAL_MAX_LATERAL_ERROR_M
            ),
            "no_more_than_50mm_along_track_overshoot": (
                isinstance(final_metrics, Mapping)
                and float(final_metrics["along_track_progress_m"]) <= POINT_GOAL_DISTANCE_M + POINT_GOAL_MAX_OVERSHOOT_M
            ),
            "termination_condition_triggered_stop": activation.get("termination_condition_met") is True,
            "measured_physical_motion": activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED",
            "same_sample_source_locomotion_handshake": activation.get("control_handshake_status") == "CONTROL_HANDSHAKE_CONFIRMED",
            "terminal_pose_ready_before_zero": (
                state.get("terminal_pose_ready") is True if has_heading_target else True
            ),
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
            "measured_heading_change_in_requested_direction": measured_heading_change_gate,
            "final_yaw_within_pose_goal_tolerance": final_yaw_tolerance_gate,
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
            "requested_heading_change_rad": heading_change_rad,
            "final_yaw_rad": final_yaw,
            "final_heading_change_rad": final_yaw_change,
            "final_yaw_error_rad": final_yaw_error,
            "final_yaw_tolerance_rad": final_yaw_tolerance_rad,
            "minimum_measured_heading_change_rad": TURN_MIN_HEADING_CHANGE_RAD,
            "heading_goal_measurement": heading_measurement,
            "dynamic_stop_target_pose_at_trigger": (
                state.get("metrics_at_trigger", {}).get("dynamic_stop_target_pose")
                if isinstance(state.get("metrics_at_trigger"), Mapping) else None
            ),
            "distance_m": POINT_GOAL_DISTANCE_M,
            "stop_advance_at_trigger_m": state.get("stop_advance_at_trigger_m"),
            "measured_forward_speed_at_trigger_m_s": state.get("measured_forward_speed_at_trigger_m_s"),
            "dynamic_stop_progress_at_trigger_m": state.get("stop_progress_at_trigger_m"),
            "stop_lateral_error_at_trigger_m": state.get("stop_lateral_error_at_trigger_m"),
            "stop_reason": state.get("stop_reason"),
            "final_tolerance_m": POINT_GOAL_FINAL_TOLERANCE_M,
            "forward_command_m_s": POINT_GOAL_COMMAND_M_S,
            "maximum_active_command_s": POINT_GOAL_MAX_DURATION_S,
            "command_profile": (
                "0.12 m/s forward warmup for the validated Run65 0.5 s turn-start interval, then a bounded cubic-Hermite "
                "start-to-goal pose path tracked with pure-pursuit at 0.14 m/s and +/-0.075 rad/s; when the existing "
                "position stop prediction/line is reached, enter a maximum-3 s forward-zero terminal yaw/rate feedback "
                "phase, then reuse the existing watchdog, zero stream, STAND transition and physical-stop validation"
                if has_heading_target else
                "Run73 waypoint-bearing control for a straight 0.30 m point goal at 0.14 m/s, existing angular taper, "
                "position stop prediction/line, watchdog, zero stream, STAND transition and physical-stop validation"
            ),
            "goal_triggered": state.get("goal_triggered"),
            "target_crossing_detected": state.get("target_crossing_detected"),
            "final_distance_to_goal_m": final_distance,
            "final_along_track_progress_m": (
                float(final_metrics["along_track_progress_m"])
                if isinstance(final_metrics, Mapping) else None
            ),
            "final_lateral_error_m": (
                float(final_metrics["lateral_error_m"])
                if isinstance(final_metrics, Mapping) else None
            ),
            "final_distance_to_dynamic_stop_target_m": final_stop_target_distance,
            "speed_scaled_stop_model": {
                "formula": "stop_advance_m = endpoint_response_intercept_m + max(0, measured_forward_speed_m_s) * endpoint_response_time_s",
                "endpoint_response_intercept_m": POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M,
                "endpoint_response_time_s": POINT_GOAL_ENDPOINT_RESPONSE_TIME_S,
                "maximum_forward_speed_m_s": POINT_GOAL_MAX_PLANAR_SPEED_M_S,
                "maximum_derived_stop_advance_m": (
                    POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M
                    + POINT_GOAL_ENDPOINT_RESPONSE_TIME_S * POINT_GOAL_MAX_PLANAR_SPEED_M_S
                ),
                "calibration_observations": [
                    {
                        "attempt_id": "c1883327-5e43-4e6d-bf41-5dac382c9f49",
                        "zero_start_progress_m": 0.392358095533107,
                        "final_settled_progress_m": 0.37439102569838006,
                        "signed_endpoint_delta_m": -0.01796706983472694,
                        "measured_forward_speed_m_s": 0.15815622063915188,
                    },
                    {
                        "attempt_id": "6c3a6c00-9afb-4eea-92d7-da560dc9e036",
                        "zero_start_progress_m": 0.26712099449260185,
                        "final_settled_progress_m": 0.26294316303228604,
                        "signed_endpoint_delta_m": -0.004177831460315806,
                        "measured_forward_speed_m_s": 0.007654207147844275,
                    },
                ],
                "limitation": "two SIMULATION_ONLY endpoint observations define an experiment-specific linear fit; early-stop behavior remains unverified after the one authorized 10-second trial; final position and physical stop gates remain authoritative",
            },
            "heading_goal_stop_model": {
                "policy": "run92_measured_terminal_entry_position_margin",
                "stop_line_progress_m": POINT_GOAL_DISTANCE_M + float(heading_goal_stop_advance_m),
                "forward_advance_from_goal_m": heading_goal_stop_advance_m,
                "predicted_early_stop_tolerance_m": heading_goal_predicted_stop_tolerance_m,
                "basis_attempt_id": "b56aed69-193d-4f9a-a083-dbd9f6b6b4c3",
                "measured_terminal_entry_position_error_m": 0.04340987581979005,
                "measured_forward_speed_at_terminal_entry_m_s": 0.084330597482439,
                "measured_terminal_position_guard_time_s": 0.309839665,
                "measured_zero_to_settled_lateral_displacement_m": 0.04194785987483458,
                "measured_zero_to_settled_path_length_m": 0.10288167474263706,
                "overshoot_guard_m": POINT_GOAL_MAX_OVERSHOOT_M,
                "margin_to_overshoot_guard_m": POINT_GOAL_MAX_OVERSHOOT_M - float(heading_goal_stop_advance_m),
                "classification": "SIMULATION_ONLY_DERIVED_RUN92_FAILURE_MEASUREMENT",
            } if has_heading_target else None,
            "postwalk_stand_action": (
                "STAND_DEFAULT requested after the bounded zero stream; require measured readback and physical stop gate"
            ),
            "command_profile_parameters": {
                "requested_heading_change_rad": heading_change_rad,
                "maximum_heading_change_rad": POINT_GOAL_MAX_HEADING_CHANGE_RAD,
                "final_yaw_tolerance_rad": final_yaw_tolerance_rad,
                "minimum_measured_heading_change_rad": TURN_MIN_HEADING_CHANGE_RAD,
                "heading_change_warmup_s": POINT_GOAL_HEADING_CHANGE_WARMUP_S,
                "heading_change_warmup_forward_m_s": POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S,
                "cruise_speed_m_s": POINT_GOAL_COMMAND_M_S,
                "steering_forward_speed_m_s": (
                    POINT_GOAL_POSE_CURVE_COMMAND_M_S if has_heading_target else POINT_GOAL_STEERING_COMMAND_M_S
                ),
                "minimum_command_speed_m_s": (
                    POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S
                    if has_heading_target else POINT_GOAL_STEERING_COMMAND_M_S
                ),
                "measured_effective_forward_deadzone_m_s": 0.111,
                "maximum_angular_velocity_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                "heading_gain": POINT_GOAL_HEADING_GAIN,
                "post_stop_observation_s": self.args.extra_stop_observation,
                "heading_control_for_pose_goal": (
                    "warmup-aligned cubic-Hermite approach using curvature feedforward plus measured yaw/yaw-rate feedback; "
                    "cross-track remains a hard safety guard; terminal yaw/rate control uses the existing forward-zero phase"
                    if has_heading_target else "not applicable; position-only goal"
                ),
                "heading_policy": "Run73_position_bearing_for_point_goals; coordinated-pose controller for heading-change goals",
                "heading_goal_stop_line_progress_m": (
                    POINT_GOAL_DISTANCE_M + float(heading_goal_stop_advance_m)
                    if has_heading_target else None
                ),
                "heading_goal_predicted_early_stop_tolerance_m": (
                    float(heading_goal_predicted_stop_tolerance_m)
                    if has_heading_target else None
                ),
                "heading_yaw_rate_estimator": (
                    "0.4 s unwrapped odom yaw regression on monotonic receive timestamps"
                    if has_heading_target else None
                ),
                "heading_yaw_rate_feedback_max_age_s": (
                    POINT_GOAL_MAX_YAW_RATE_FEEDBACK_AGE_S
                    if has_heading_target else None
                ),
                "pose_goal_approach_path": (
                    "straight segment matching the 0.5 s zero-angular warmup, then a sampled cubic Hermite turn tracked by curvature feedforward plus measured path-yaw/yaw-rate feedback; "
                    "cross-track is a hard guard and does not add opposing angular steering; "
                    "curve chord is extended in 1 mm increments only until sampled ideal angular demand fits 95% of the 0.075 rad/s cap, "
                    "with endpoint and lateral errors constrained by the existing 50 mm gates; the 20% path carrot is logged for diagnosis only"
                    if has_heading_target else None
                ),
                "pose_goal_warmup_distance_m": (
                    POINT_GOAL_HEADING_CHANGE_WARMUP_S * POINT_GOAL_HEADING_CHANGE_WARMUP_FORWARD_M_S
                    if has_heading_target else None
                ),
                "pose_goal_curve_forward_command_m_s": (
                    POINT_GOAL_POSE_CURVE_COMMAND_M_S if has_heading_target else None
                ),
                "pose_goal_curve_angular_rate_utilization_limit": 0.95 if has_heading_target else None,
                "pose_goal_terminal_forward_speed_m_s": 0.0 if has_heading_target else None,
                "pose_goal_terminal_phase_max_s": POINT_GOAL_POSE_TERMINAL_CONTROL_MAX_S if has_heading_target else None,
                "pose_goal_terminal_yaw_rate_limit_rad_s": PHYSICAL_STOP_MAX_YAW_RATE_RAD_S if has_heading_target else None,
                "pose_goal_terminal_ready_dwell_s": POSE_GOAL_TERMINAL_READY_DWELL_S if has_heading_target else None,
                "pose_goal_terminal_heading_gain": POSE_GOAL_TERMINAL_HEADING_GAIN if has_heading_target else None,
                "pose_goal_terminal_yaw_rate_damping": POSE_GOAL_TERMINAL_RATE_DAMPING if has_heading_target else None,
                "pose_goal_terminal_min_effective_angular_rad_s": POSE_GOAL_TERMINAL_MIN_ANGULAR_RAD_S if has_heading_target else None,
                "pose_goal_post_stop_yaw_drift_compensation_rad": (
                    POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD if has_heading_target else None
                ),
                "pose_goal_terminal_pre_stop_yaw_bias_rad": (
                    POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD if has_heading_target else None
                ),
                "pose_goal_terminal_bias_drift_basis_attempt_ids": (
                    ["ee98c557-ef2d-4db8-b1c9-15aa811bafce"] if has_heading_target else None
                ),
                "pose_goal_terminal_bias_drift_observation_rad": (
                    POSE_GOAL_RUN95_POST_STOP_YAW_DRIFT_RAD if has_heading_target else None
                ),
                "pose_goal_terminal_direction_margin_rad": (
                    abs(heading_change_rad)
                    + POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD
                    - POSE_GOAL_RUN95_POST_STOP_YAW_DRIFT_RAD
                    - POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD
                    - TURN_MIN_HEADING_CHANGE_RAD
                    if has_heading_target else None
                ),
                "pose_goal_observed_post_stop_yaw_drift_rad": (
                    list(POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD)
                    if has_heading_target else None
                ),
                "pose_goal_pre_stop_yaw_margin_rad": (
                    POSE_GOAL_PRESTOP_YAW_MARGIN_RAD if has_heading_target else None
                ),
                "pose_goal_terminal_setpoint_tolerance_rad": (
                    POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD if has_heading_target else None
                ),
                "pose_goal_approach_path_terminal_heading_policy": (
                    "approach endpoint uses 0.05 rad bias to keep its curve feasible; terminal target uses a separate 0.10 rad bias derived from Run95 drift and the 5-degree minimum requested-turn gate; not calibrated across Runs93/94"
                    if has_heading_target else None
                ),
                "pose_goal_post_stop_drift_basis_attempt_ids": (
                    ["dc70febe-9822-4f77-94da-e3d7a6f193f0", "25204271-4d8c-48d4-9718-0d71ad821174", "ee98c557-ef2d-4db8-b1c9-15aa811bafce"]
                    if has_heading_target else None
                ),
                "pose_goal_post_stop_yaw_drift_observations_rad": (
                    list(POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD)
                    if has_heading_target else None
                ),
                "pose_goal_lookahead_fraction": POINT_GOAL_POSE_LOOKAHEAD_FRACTION if has_heading_target else None,
                "pose_goal_terminal_publisher_rotation_effectiveness": (
                    "SIGNED_ZERO_FORWARD_YAW_RESPONSE_OBSERVED_BUT_VARIABLE_AND_TOO_SLOW_FOR_0.22_RAD_ENTRY_ERROR"
                    if has_heading_target else None
                ),
                "angular_taper_start_predicted_error_m": POINT_GOAL_ANGULAR_TAPER_START_ERROR_M,
                "angular_taper_end_predicted_error_m": POINT_GOAL_FINAL_TOLERANCE_M,
                "maximum_lateral_error_m": POINT_GOAL_MAX_LATERAL_ERROR_M,
                "maximum_overshoot_m": POINT_GOAL_MAX_OVERSHOOT_M,
            },
            "activation_fault": activation.get("fault"),
            "post_stop_fault": activation.get("post_stop_fault"),
            "acceptance_gates": acceptance_gates,
            "controller_fault": state.get("fault"),
            "pose_goal_control": {
                "policy": "coordinated-pose" if has_heading_target else "position-bearing",
                "position_triggered_terminal_phase": state.get("pose_goal_position_triggered"),
                "terminal_pose_ready_before_zero": state.get("terminal_pose_ready"),
                "terminal_control_samples": state.get("pose_goal_terminal_control_samples"),
                "phase_history": state.get("pose_goal_phase_history"),
                "failure_reason": pose_controller.failure_reason if pose_controller is not None else None,
                "classification": "SIMULATION_ONLY_CONTROLLER_DIAGNOSTIC" if has_heading_target else None,
            },
            "pose_goal_geometry_preflight": state.get("pose_goal_geometry_preflight"),
            "pose_goal_approach_path": state.get("pose_goal_approach_path"),
            "ground_truth_trace": goal_trace,
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
                    heading_change_rad=float(getattr(self.args, "heading_change_rad", 0.0)),
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


def validate_first_motion_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the complete child CLI profile without initializing ROS or the simulator."""
    if args.turning_diagnostic and args.point_goal_diagnostic:
        raise ValueError("select only one of turning or point-goal diagnostics")
    profile: dict[str, Any] = {"point_goal": False}
    if args.point_goal_diagnostic:
        heading_change = float(args.heading_change_rad)
        if not math.isfinite(heading_change) or abs(heading_change) > POINT_GOAL_MAX_HEADING_CHANGE_RAD:
            raise ValueError("heading offset must be finite and within +/-0.20 rad")
        has_heading_target = abs(heading_change) > 1e-9
        expected_policy = "coordinated-pose" if has_heading_target else "position-bearing"
        expected_extra_observation = (
            POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S if has_heading_target
            else POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S
        )
        if args.heading_control_policy != expected_policy:
            raise ValueError(
                f"heading-control policy {args.heading_control_policy!r} does not match "
                f"requested heading {heading_change}: expected {expected_policy!r}"
            )
        tolerance = args.final_yaw_tolerance_rad
        if (
            tolerance is None or not math.isfinite(float(tolerance))
            or abs(float(tolerance) - POINT_GOAL_FINAL_YAW_TOLERANCE_RAD) > 1e-9
        ):
            raise ValueError(
                f"final yaw tolerance must be explicitly propagated as "
                f"{POINT_GOAL_FINAL_YAW_TOLERANCE_RAD:.3f} rad"
            )
        if has_heading_target:
            if (
                args.heading_goal_stop_advance_m is None
                or not math.isfinite(float(args.heading_goal_stop_advance_m))
                or abs(float(args.heading_goal_stop_advance_m) - POINT_GOAL_HEADING_STOP_ADVANCE_M) > 1e-9
                or args.heading_goal_predicted_stop_tolerance_m is None
                or not math.isfinite(float(args.heading_goal_predicted_stop_tolerance_m))
                or abs(float(args.heading_goal_predicted_stop_tolerance_m) - POINT_GOAL_HEADING_PREDICTED_STOP_TOLERANCE_M) > 1e-9
            ):
                raise ValueError("coordinated pose-goal position-stop parameters must be explicitly propagated")
        elif (
            args.heading_goal_stop_advance_m is not None
            or args.heading_goal_predicted_stop_tolerance_m is not None
        ):
            raise ValueError("pose-goal position-stop parameters require a coordinated pose-goal controller")
        if (
            args.forward_speed != POINT_GOAL_COMMAND_M_S
            or args.pulse_duration != POINT_GOAL_MAX_DURATION_S
            or args.ramp_duration_s != 0.0
            or args.deceleration_duration_s != 0.0
            or args.startup_speed is not None
            or args.max_observed_planar_speed_m_s != 0.2
            or args.zero_until_stand_timeout_s != 3.5
            or not args.zero_only_stop
            or not POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S
            <= args.extra_stop_observation
            <= expected_extra_observation
        ):
            raise ValueError(
                "point-goal profile must keep the fixed forward/angular envelope, 10 s cap, "
                "zero-only stop, 0.2 m/s speed guard, 3.5 s STAND wait, and "
                f"{expected_extra_observation:g} s observation for this goal type"
            )
        profile = {
            "point_goal": True,
            "requested_heading_change_rad": heading_change,
            "final_yaw_tolerance_rad": float(tolerance),
            "heading_control_policy": args.heading_control_policy,
            "heading_goal_stop_advance_m": args.heading_goal_stop_advance_m,
            "heading_goal_predicted_stop_tolerance_m": args.heading_goal_predicted_stop_tolerance_m,
            "pose_goal_terminal_control_max_s": (
                POINT_GOAL_POSE_TERMINAL_CONTROL_MAX_S if has_heading_target else None
            ),
            "pose_goal_terminal_forward_speed_m_s": 0.0 if has_heading_target else None,
            "pose_goal_terminal_yaw_rate_limit_rad_s": (
                PHYSICAL_STOP_MAX_YAW_RATE_RAD_S if has_heading_target else None
            ),
            "pose_goal_post_stop_yaw_drift_compensation_rad": (
                POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD if has_heading_target else None
            ),
            "pose_goal_terminal_pre_stop_yaw_bias_rad": (
                POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD if has_heading_target else None
            ),
            "pose_goal_terminal_bias_drift_observation_rad": (
                POSE_GOAL_RUN95_POST_STOP_YAW_DRIFT_RAD if has_heading_target else None
            ),
            "pose_goal_terminal_direction_margin_rad": (
                abs(heading_change)
                + POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD
                - POSE_GOAL_RUN95_POST_STOP_YAW_DRIFT_RAD
                - POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD
                - TURN_MIN_HEADING_CHANGE_RAD
                if has_heading_target else None
            ),
            "pose_goal_observed_post_stop_yaw_drift_rad": (
                list(POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD)
                if has_heading_target else None
            ),
            "pose_goal_pre_stop_yaw_margin_rad": (
                POSE_GOAL_PRESTOP_YAW_MARGIN_RAD if has_heading_target else None
            ),
            "pose_goal_terminal_setpoint_tolerance_rad": (
                POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD if has_heading_target else None
            ),
            "pose_goal_approach_path_policy": (
                "0.5 s straight segment matching the zero-angular warmup, then bounded cubic-Hermite turn tracked by curvature feedforward and measured path-yaw/yaw-rate feedback; "
                "lateral error is a hard guard, not an opposing angular correction; "
                "minimal curve-chord extension in 1 mm increments to satisfy the 95% angular-rate headroom and 50 mm gates"
                if has_heading_target else None
            ),
            "pose_goal_warmup_duration_s": (POSE_GOAL_APPROACH_WARMUP_S if has_heading_target else None),
            "pose_goal_warmup_forward_m_s": (POSE_GOAL_WARMUP_FORWARD_M_S if has_heading_target else None),
            "pose_goal_curve_forward_m_s": (POSE_GOAL_CURVE_FORWARD_M_S if has_heading_target else None),
            "pose_goal_curve_angular_rate_utilization_limit": (
                POSE_GOAL_CURVE_ANGULAR_UTILIZATION if has_heading_target else None
            ),
            "pose_goal_path_endpoint_offset_limit_m": (
                POSE_GOAL_PATH_ENDPOINT_OFFSET_LIMIT_M if has_heading_target else None
            ),
            "pose_goal_approach_path_terminal_heading_policy": (
                "approach endpoint uses 0.05 rad bias to keep its curve feasible; terminal target uses a separate 0.10 rad bias derived from Run95 drift and the 5-degree minimum requested-turn gate; not calibrated across Runs93/94"
                if has_heading_target else None
            ),
            "extra_stop_observation_s": float(args.extra_stop_observation),
            "max_extra_stop_observation_s": expected_extra_observation,
            "active_command_limit_s": POINT_GOAL_MAX_DURATION_S,
            "forward_command_m_s": POINT_GOAL_COMMAND_M_S,
            "angular_command_limit_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
            "max_lateral_error_m": POINT_GOAL_MAX_LATERAL_ERROR_M,
            "final_position_tolerance_m": POINT_GOAL_FINAL_TOLERANCE_M,
        }
    elif args.heading_control_policy is not None or args.final_yaw_tolerance_rad is not None:
        raise ValueError("heading policy and final yaw tolerance are valid only for point-goal diagnostics")

    if args.max_observed_planar_speed_m_s is not None and (
        not math.isfinite(args.max_observed_planar_speed_m_s)
        or not 0.0 < args.max_observed_planar_speed_m_s <= 0.2
    ):
        raise ValueError("observed planar speed gate must be positive and no greater than 0.2 m/s")
    if not math.isfinite(args.zero_until_stand_timeout_s) or not 0.0 <= args.zero_until_stand_timeout_s <= 3.5:
        raise ValueError("zero-until-stand timeout must remain within 0..3.5 seconds")
    return profile


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
    first.add_argument("--heading-change-rad", type=float, default=0.0)
    first.add_argument("--final-yaw-tolerance-rad", type=float, default=None)
    first.add_argument(
        "--heading-control-policy", choices=("position-bearing", "coordinated-pose"), default=None,
    )
    first.add_argument("--heading-goal-stop-advance-m", type=float, default=None)
    first.add_argument("--heading-goal-predicted-stop-tolerance-m", type=float, default=None)
    first.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        profile = validate_first_motion_cli_args(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        if not args.maintainer_authorized or not args.reset_gate_verified:
            print("BLOCKED: validation-only invocation must carry the production authorization flags", file=sys.stderr)
            return 2
        print(json.dumps({
            "schema": "robotsim.issue80.m1.cli-validation.v1",
            "status": "VALIDATED_NO_ROS_NO_RUNTIME",
            "attempt_id": args.attempt_id,
            "production_authorization_flags_present": True,
            "network_isolation": "NOT_EVALUATED; validation container is intentionally --network=none",
            "reset_and_standing": "NOT_EVALUATED",
            "ros_initialized": False,
            "mujoco_or_mc_started": False,
            "profile": profile,
        }, indent=2))
        return 0
    try:
        return _focused_diagnostic_command(args)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
