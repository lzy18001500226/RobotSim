#!/usr/bin/env python3
"""Bounded, simulation-only X2 M1 first-motion probe. Requires explicit maintainer authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import replace
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
    MODE_SOURCE_READBACK_SKEW_MAX_S,
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
    validate_runtime_scope, validate_b_post_locomotion_observation,
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
    POSE_GOAL_HEADING_STOP_ADVANCE_M,
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
    POSE_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD,
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
from issue80_m1_mode_logic import (
    ACTIVATION_PROBE_DURATION_S,
    ACTIVATION_PROBE_FORWARD_M_S,
    RAICOM_LOW_CRUISE_FORWARD_M_S,
    MCLocomotionAbortLifecycle,
    activation_probe_abort_is_expected,
    activation_probe_safe_stop_passes,
    classify_locomotion_activation_readbacks,
    classify_set_mc_action_response,
    run_zero_only_locomotion_activation_observation,
)
from issue80_raicom_mjdata_capture import (
    CAPTURE_TRACE_NAME,
    MjDataCaptureError,
    NativeMjDataLivePose,
    NativeMjDataLivePoseReader,
)
from issue80_waypoint_route import (
    DirectiveKind, MotionFeedback, MotionStatus, PhysicalStopEvidence,
    PhysicalStopStatus, PoseEstimate, Profile, RouteState, StopBehavior,
    SafetyObservation, WaypointRouteExecutor, YawPolicy,
    build_table_approach_route_from_mission,
    build_waypoint_docking_route_from_poses, load_waypoint_docking_pose_input,
    distance_to_polygon_boundary,
    validate_waypoint_docking_scene_clearance,
    build_local01_short_scene_route, build_runtime_raw_odom_short_scene_route,
    validate_local01_short_scene_route_input, validate_local01_reset_odom_pose,
    LOCAL01_SHORT_ROUTE_FORWARD_M_S,
    LOCAL01_SHORT_ROUTE_MAX_WINDOWS as SHORT_ROUTE_MAX_WINDOWS,
    LOCAL01_SHORT_ROUTE_MAX_WINDOW_S as SHORT_ROUTE_MAX_WINDOW_S,
    LOCAL01_SHORT_ROUTE_MAX_AGGREGATE_S as SHORT_ROUTE_MAX_AGGREGATE_S,
)
from issue80_local01_scene_route import (
    validate_runtime_raw_odom_table_approach_mission,
    validate_runtime_raw_odom_table_approach_start,
    validate_runtime_raw_odom_aligned_start_table_approach_mission,
    validate_runtime_raw_odom_aligned_start_table_approach_start,
)
from issue80_route_motion_bridge import (
    ROUTE_WINDOW_ZERO_MARGIN_S, WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S,
    WAYPOINT_B_MAX_WINDOWS, WAYPOINT_B_POSITION_TOLERANCE_M,
    WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M, WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
    WAYPOINT_B_CORRIDOR_MAX_FORWARD_M, WAYPOINT_B_MAX_CROSS_TRACK_M,
    CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
    CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
    FIXED_WINDOW_ARC_DEADBAND_RAD_S,
    FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
    WAYPOINT_B_WINDOW_MAX_S,
    CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
    CURVED_PATH_FORWARD_M_S,
    CURVED_PATH_MAX_CROSS_TRACK_M,
    CURVED_PATH_MAX_WINDOW_S,
    CURVED_PATH_MAX_WINDOWS,
    CURVED_PATH_POSITION_TOLERANCE_M,
    CURVED_PATH_STOP_RESERVE_M,
    MAX_POSE_AGE_S,
    RouteMotionBridge, RouteMotionBridgeError, RuntimeReadiness,
    evaluate_route_odom_freshness_window,
    evaluate_route_window_rearm_gate,
    build_short_right_corner_route, build_shallow_curved_path_reference,
    build_curved_waypoint_route, curved_path_metrics, measured_m1_nonzero_window_s,
    waypoint_bearing_angular_command,
    segment_lookahead_point,
    CONTINUOUS_TABLE_LOOKAHEAD_M,
    continuous_table_segment_cross_track_limit_m,
    fixed_window_arc_angular_command,
    fixed_window_heading_acquisition_required,
    route_replan_odom_freshness,
    waypoint_b_arrival_passes, waypoint_b_arrival_trigger_radius_m,
    waypoint_b_arrival_after_verified_stop_passes,
    waypoint_b_window_budget_passes, waypoint_b_forward_guard_reached,
    waypoint_b_cross_track_guard_reached,
    TransitSteeringGate,
    ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
    ZERO_FORWARD_HEADING_ACQUISITION_MAX_S,
    ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD,
    zero_forward_heading_acquisition_plan,
    forward_gait_yaw_command,
    active_locomotion_source_readbacks_valid,
    RouteOdomFreshnessTracker,
    OdomSourceStampFence,
    route_feedback_sample_disposition,
    fresh_stand_mode_readback_valid,
    fresh_input_source_readback_valid,
    validate_curved_path_cli_profile,
)

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
ROUTE_TRANSIT_MAX_FORWARD_M_S = 0.12
ROUTE_STEERING_WARMUP_S = 0.5
# A TABLE_APPROACH segment may consume all 15 approved five-second command
# windows.  Its wall-time cap must also cover the unchanged 1 s zero stream,
# 3.5 s STAND wait, 2 s physical hold, and 7 s extra observation after each
# stopped window: 15 * 19 s = 285 s.  Keep 300 s below the parent's 410 s
# child budget; this does not increase any active-command allowance.
TABLE_APPROACH_SEGMENT_TIMEOUT_S = 300.0
TABLE_APPROACH_CONTINUOUS_WINDOW_S = 10.0
TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS = 8
TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S = 75.0
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
POINT_GOAL_HEADING_STOP_ADVANCE_M = POSE_GOAL_HEADING_STOP_ADVANCE_M
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
        zero_only_authorized = bool(
            getattr(args, "zero_only_activation_verification", False)
            and getattr(args, "maintainer_authorized_zero_only_activation", False)
        )
        activation_probe_authorized = bool(
            getattr(args, "activation_probe_qualification", False)
            and getattr(args, "maintainer_authorized_activation_probe", False)
        )
        if bool(getattr(args, "zero_only_activation_verification", False)) != bool(
            getattr(args, "maintainer_authorized_zero_only_activation", False)
        ):
            raise ValueError("zero-only activation verification requires its separate authorization flag")
        if bool(getattr(args, "activation_probe_qualification", False)) != bool(
            getattr(args, "maintainer_authorized_activation_probe", False)
        ):
            raise ValueError("activation probe requires its separate authorization flag")
        if sum((bool(args.maintainer_authorized), zero_only_authorized, activation_probe_authorized)) > 1:
            raise ValueError("first-motion, zero-only activation, and activation-probe authorizations are separate")
        validate_first_motion_authorization(
            reset_gate_verified=args.reset_gate_verified,
            maintainer_authorized=bool(args.maintainer_authorized or zero_only_authorized or activation_probe_authorized),
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
        self.odom_source_stamp_fence = OdomSourceStampFence(
            allow_bounded_clock_rebase=bool(
                getattr(args, "raicom_native_mjdata_route_feedback", False)
                and getattr(args, "native_x2_timestep", False)
            ),
            allow_forward_sim_clock_leaps=bool(
                getattr(args, "raicom_native_mjdata_route_feedback", False)
                and getattr(args, "native_x2_timestep", False)
            ),
        )
        self.odom_source_stamp_drop_events: list[dict[str, Any]] = []
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
        # This raw stream is not a safety input and the installed runtime
        # publishes an empty payload at a high rate.  Do not let optional
        # diagnostics compete with the five live state subscriptions unless a
        # caller explicitly requested debug capture.
        self.mc_debug_f64_subscription_enabled = bool(
            getattr(args, "capture_mc_rl_debug", False)
        )
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
        self.activation_lifecycle: MCLocomotionAbortLifecycle | None = None

        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # The vendor simulator publishes odometry at 500 Hz while this route
        # controller consumes only the most recent ground-truth pose.  Keep
        # one BEST_EFFORT sample so a slower Python callback cannot build an
        # obsolete pose queue; IMU, joint, and MC-state subscriptions retain
        # their existing depth-10 history.
        self.odom_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom_subscription_configured_qos = self._qos_profile(self.odom_state_qos)
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
            Odometry, "/aima/hal/odom/state", lambda msg: self._store("odom", msg), self.odom_state_qos
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
        if self.mc_debug_f64_subscription_enabled:
            self.create_subscription(
                Float64MultiArray, MC_DEBUG_F64_TOPIC, self._store_mc_debug_f64, self.state_qos
            )

    def _store(self, key: str, msg: Any) -> None:
        received = time.monotonic()
        if key == "odom":
            try:
                stamp = msg.header.stamp
                stamp_ns: int | None = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            except (AttributeError, TypeError, ValueError, OverflowError):
                stamp_ns = None
            fence = self.odom_source_stamp_fence.observe(
                current_stamp_ns=stamp_ns,
                received_steady_s=received,
                received_wall_time_ns=time.time_ns(),
            )
            if fence["action"] in {
                "DROP_INVALID", "DROP_OUT_OF_ORDER", "DROP_FUTURE_LEAP",
            }:
                event = {
                    "event": "odom_source_stamp_packet_dropped",
                    "phase": self.current_phase,
                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_monotonic_sec": received,
                    **fence,
                }
                # Preserve bounded evidence while the fence itself retains the
                # complete count and final dropped-packet diagnostic.
                if len(self.odom_source_stamp_drop_events) < 32:
                    self.odom_source_stamp_drop_events.append(event)
                self._write_trace_row(event)
                return
        self.latest[key] = msg
        self.received_at[key] = received

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
        """Send one bounded STAND recovery request while zero publication continues."""
        # The one-second zero burst services its own ROS callbacks; refresh the
        # snapshot again before the safety gate so its cached odometry is not
        # falsely rejected as stale at the end of that bounded burst.
        snapshot = self.sample_for_first_motion()
        fault = snapshot_fault(
            snapshot, self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
        )
        if fault:
            raise RuntimeError(f"post-stop stand request safety gate failed: {fault}")
        pose = snapshot.get("pose")
        try:
            measured_planar_speed = math.hypot(float(pose["vx"]), float(pose["vy"]))
            speed_cap = float(getattr(self.args, "max_observed_planar_speed_m_s", None) or 0.2)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"post-stop stand request lacks finite planar speed telemetry: {exc}") from exc
        if not math.isfinite(measured_planar_speed) or not math.isfinite(speed_cap) or speed_cap <= 0.0:
            raise RuntimeError("post-stop stand request lacks finite planar speed telemetry")
        if (
            snapshot.get("get_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
            or snapshot.get("common_mode_desc") not in (MODE_STAND, MODE_LOCOMOTION)
        ):
            raise RuntimeError("post-stop stand request observed unrelated MC action")
        if (
            snapshot.get("get_source_name") not in ("", self.source)
            or snapshot.get("common_source_name") not in ("", self.source)
        ):
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
            "measured_planar_speed_m_s": measured_planar_speed,
            "planar_speed_guard_exceeded_during_recovery": measured_planar_speed > speed_cap,
            "planar_speed_cap_m_s": speed_cap,
            "recovery_semantics": (
                "one STAND recovery request may be sent above the motion speed cap if telemetry, "
                "posture, joint limits, mode and source are otherwise valid; the speed violation "
                "remains an experiment failure and cannot authorize rearm"
            ),
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
            "request_sent": True,
            "request": _json_safe(request),
            "request_source": str(request.source),
            "requested_action_desc": str(request.command.action_desc),
            "response": response_payload,
            **diagnostics,
        }
        self.events.append({"event": "set_mc_action_ld", **result})
        return result

    def _activation_speed_cap(self) -> float:
        configured = getattr(self.args, "max_observed_planar_speed_m_s", None)
        return 0.2 if configured is None else float(configured)

    def begin_locomotion_activation_probe_request(
        self,
        *,
        registration: Mapping[str, Any],
        phase: str,
    ) -> tuple[MCLocomotionAbortLifecycle, dict[str, Any], dict[str, Any]]:
        """Prepare one source-grounded activation probe from a fresh STAND state."""
        initial = self.sample_for_first_motion()
        initial_fault = snapshot_fault(
            initial,
            self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=self._activation_speed_cap(),
        )
        self._trace_first_motion_sample(f"{phase}-stand-before-ld", initial)
        stand_confirmed = (
            initial_fault is None
            and stand_mode_readback_confirmed(initial)
            and initial.get("get_mode_status_value") == 100
            and initial.get("common_mode_status_value") == 100
            and initial.get("fsm_state") == MC_FSM_STABLE
            and initial.get("body_status") == 1
        )
        lifecycle = MCLocomotionAbortLifecycle()
        if not lifecycle.verify_initial_stand(confirmed=stand_confirmed, now_s=time.monotonic()):
            result = {
                "status": "BLOCKED",
                "classification": "INITIAL_STAND_NOT_FRESHLY_VERIFIED",
                "fault": initial_fault or "fresh STAND/FSM/body readback did not pass",
                "initial_snapshot": initial,
                "set_mc_action_sent": False,
                "nonzero_publish_calls_returned": 0,
                "final_lifecycle": lifecycle.snapshot(),
            }
            return lifecycle, {"request_sent": False, "accepted_by_service": False}, result

        set_action = self.request_ld_once()
        lifecycle.request_locomotion(
            accepted=set_action.get("accepted_by_service") is True,
            now_s=time.monotonic(),
        )
        result = {
            "classification": "SIMULATION_ONLY_SOURCE_GROUNDED_ACTIVATION_ABORT_PROBE",
            "status": "READY" if set_action.get("accepted_by_service") is True else "BLOCKED",
            "initial_snapshot": initial,
            "input_source_registration": {
                "accepted_by_service": registration.get("accepted_by_service"),
                "task_completion_confirmed": registration.get("task_completion_confirmed"),
                "task_state_value": registration.get("task_state_value"),
                "task_id": registration.get("task_id"),
                "selected_action": registration.get("selected_action"),
            },
            "set_mc_action_ld": set_action,
            "source_contract_evidence": {
                "official_arbitration_docs": "AimDK_X2 1.1.0: new source requires a valid nonzero command to acquire arbitration",
                "official_arbitration_docs_url": "https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/MC_control.html",
                "official_initial_step_docs": "AimDK_X2 1.1.0: forward startup threshold is approximately 0.1 m/s and model dependent",
                "official_initial_step_docs_url": "https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/locomotion.html",
                "installed_trace": "prior installed-runtime attempt published 0.12 m/s, selected the source within 0.024 s, and reported GetMcAction LOCOMOTION at 1.785 s; this probe uses 0.12 m/s for no more than 1.80 s",
                "installed_mc_runtime_identity": (
                    self.mc_runtime_identity.as_dict() if self.mc_runtime_identity else None
                ),
                "installed_mc_app_binary_sha256": (
                    hashlib.sha256(Path(self.mc_runtime_identity.process_executable).read_bytes()).hexdigest()
                    if self.mc_runtime_identity and Path(self.mc_runtime_identity.process_executable).is_file()
                    else None
                ),
                "installed_mc_sim_config_sha256": self.mc_sim_config_sha256,
                "zero_only_preflight_attempt_id": getattr(
                    self.args, "activation_probe_prerequisite_attempt_id", None
                ),
            },
            "probe_command": {
                "forward_velocity_m_s": 0.12,
                "lateral_velocity_m_s": 0.0,
                "angular_velocity_rad_s": 0.0,
                "publish_hz": PUBLISH_HZ,
                "maximum_duration_s": ACTIVATION_PROBE_DURATION_S,
                "measured_planar_speed_limit_m_s": 0.20,
            },
            "route_motion_authorized": False,
            "initial_lifecycle": lifecycle.snapshot(),
        }
        self.events.append({"event": "controlled_activation_probe_prepared", "phase": phase, **result})
        return lifecycle, set_action, result

    def begin_zero_only_locomotion_activation(
        self,
        *,
        registration: Mapping[str, Any],
        phase: str,
    ) -> tuple[MCLocomotionAbortLifecycle, dict[str, Any], dict[str, Any]]:
        """Request LD, then observe only zero commands until fresh mode/source confirmation."""
        initial = self.sample_for_first_motion()
        initial_fault = snapshot_fault(
            initial,
            self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=self._activation_speed_cap(),
        )
        self._trace_first_motion_sample(f"{phase}-stand-before-ld", initial)
        stand_confirmed = (
            initial_fault is None
            and stand_mode_readback_confirmed(initial)
            and initial.get("get_mode_status_value") == 100
            and initial.get("common_mode_status_value") == 100
            and initial.get("fsm_state") == MC_FSM_STABLE
            and initial.get("body_status") == 1
        )
        lifecycle = MCLocomotionAbortLifecycle()
        if not lifecycle.verify_initial_stand(confirmed=stand_confirmed, now_s=time.monotonic()):
            result = {
                "status": "BLOCKED",
                "classification": "INITIAL_STAND_NOT_FRESHLY_VERIFIED",
                "fault": initial_fault or "fresh STAND/FSM/body readback did not pass",
                "initial_snapshot": initial,
                "zero_publish_calls_returned": 0,
                "nonzero_publish_calls_returned": 0,
                "final_lifecycle": lifecycle.snapshot(),
            }
            return lifecycle, {"request_sent": False, "accepted_by_service": False}, result

        set_action = self.request_ld_once()
        accepted = set_action.get("accepted_by_service") is True
        lifecycle.request_locomotion(accepted=accepted, now_s=time.monotonic())
        # Do not make blocking GetMcAction/GetCurrentInputSource service calls
        # between the accepted LD request and the zero publisher. Their fresh
        # asynchronous snapshots are captured by the observer after each zero.
        self.events.append({
            "event": "locomotion_request_pending_zero_only_observation_started",
            "phase": phase,
            "set_mc_action_ld": set_action,
            "monotonic_s": time.monotonic(),
        })
        if not accepted:
            activation = {
                "status": "BLOCKED",
                "classification": "LOCOMOTION_REQUEST_REJECTED",
                "fault": set_action.get("message") or "SetMcAction did not accept LOCOMOTION_DEFAULT",
                "zero_publish_calls_returned": 0,
                "nonzero_publish_calls_returned": 0,
                "final_lifecycle": lifecycle.snapshot(),
            }
            return lifecycle, set_action, activation

        def record_activation_sample(row: Mapping[str, Any]) -> None:
            raw = row.get("raw_snapshot")
            if isinstance(raw, Mapping):
                snapshot = dict(raw)
                snapshot["zero_only_activation_classification"] = row.get("classification")
                snapshot["zero_only_activation_reason"] = row.get("reason")
                snapshot["zero_only_safety_fault"] = row.get("safety_fault")
                self._trace_first_motion_sample(f"{phase}-zero-only-activation", snapshot)

        activation = run_zero_only_locomotion_activation_observation(
            sample=self.sample_for_first_motion,
            publish_zero=lambda: self._publish(0.0, 0.0, 0.0),
            lifecycle=lifecycle,
            safety_fault=lambda snapshot: snapshot_fault(
                snapshot,
                self.joint_limits,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=self._activation_speed_cap(),
            ),
            expected_source=self.source,
            timeout_s=2.0,
            maximum_readback_age_s=self.args.state_timeout,
            maximum_sync_skew_s=MODE_SOURCE_READBACK_SKEW_MAX_S,
            publish_hz=PUBLISH_HZ,
            clock=time.monotonic,
            sleep=self._spin_for,
            on_sample=record_activation_sample,
        )
        activation["set_mc_action_ld"] = set_action
        activation["input_source_registration"] = {
            "accepted_by_service": registration.get("accepted_by_service"),
            "task_completion_confirmed": registration.get("task_completion_confirmed"),
            "task_state_value": registration.get("task_state_value"),
            "task_id": registration.get("task_id"),
            "selected_action": registration.get("selected_action"),
        }
        activation["initial_stand_snapshot"] = initial
        activation["final_lifecycle"] = lifecycle.snapshot()
        self.events.append({"event": "zero_only_locomotion_activation", "phase": phase, **activation})
        return lifecycle, set_action, activation

    def begin_source_grounded_route_activation(
        self,
        *,
        registration: Mapping[str, Any],
        phase: str,
        forward_command_m_s: float = ACTIVATION_PROBE_FORWARD_M_S,
        allow_raicom_low_cruise: bool = False,
        zero_forward_heading_acquisition: bool = False,
    ) -> tuple[MCLocomotionAbortLifecycle, dict[str, Any], dict[str, Any]]:
        """Prepare the reviewed nonzero-acquisition phase for one route window.

        The selected reviewed source-acquisition publish happens in the first
        portion of the existing route window. It therefore consumes that
        window's own time/displacement budget and ends through its unchanged
        zero-stop and physical-stop contract.
        """
        lifecycle, set_action, activation = self.begin_locomotion_activation_probe_request(
            registration=registration,
            phase=phase,
        )
        if activation.get("status") == "READY":
            activation.update({
                "classification": "SIMULATION_ONLY_SOURCE_GROUNDED_ROUTE_ACTIVATION",
                "route_motion_authorized": True,
                "initial_nonzero_is_part_of_route_window": True,
                "maximum_activation_qualification_s": ACTIVATION_PROBE_DURATION_S,
                "forward_velocity_m_s": float(forward_command_m_s),
                "raicom_low_cruise_profile": bool(allow_raicom_low_cruise),
                "zero_forward_heading_acquisition": bool(
                    zero_forward_heading_acquisition
                ),
                "route_source_acquisition_profile": {
                    "forward_command_m_s": float(forward_command_m_s),
                    "first_nonzero_command": (
                        "bounded_zero_forward_heading_acquisition"
                        if zero_forward_heading_acquisition
                        else "forward_transit_command"
                    ),
                    "allowed_profiles_m_s": (
                        [ACTIVATION_PROBE_FORWARD_M_S, RAICOM_LOW_CRUISE_FORWARD_M_S]
                        if allow_raicom_low_cruise else [ACTIVATION_PROBE_FORWARD_M_S]
                    ),
                    "all_other_motion_and_stop_limits_unchanged": True,
                },
            })
        self.events.append({
            "event": "source_grounded_route_activation_prepared",
            "phase": phase,
            **activation,
        })
        return lifecycle, set_action, activation

    def recover_activation_to_stand(
        self,
        *,
        lifecycle: MCLocomotionAbortLifecycle,
        registration: Mapping[str, Any],
        set_action: Mapping[str, Any],
        baseline: Mapping[str, Any] | None,
        reason: str,
        phase: str,
    ) -> dict[str, Any]:
        """Zero-only abort and one STAND request, followed by unchanged stop gates."""
        lifecycle.request_zero_and_stop(now_s=time.monotonic(), reason=reason)
        result = run_first_motion_window(
            sample=self.sample_for_first_motion,
            publish=lambda value: self._publish(value, 0.0, 0.0),
            sleep=self._spin_for,
            registration=registration,
            set_action=set_action,
            joint_limits=self.joint_limits,
            baseline_forward_displacements_m=(
                [float(window["forward_displacement_m"]) for window in baseline.get("windows", [])]
                if isinstance(baseline, Mapping) and isinstance(baseline.get("windows"), list)
                else None
            ),
            baseline_state_summary=(
                baseline.get("physical_state_summary") if isinstance(baseline, Mapping) else None
            ),
            authorized=True,
            forward_command_m_s=self.args.forward_speed,
            pulse_duration_s=self.args.pulse_duration,
            max_pulse_duration_s=MAX_AUTHORIZED_PULSE_S,
            extra_post_stop_observe_s=5.0,
            after_zero_stop=self.request_postwalk_stand_once,
            after_zero_abort=self.request_postwalk_stand_once,
            shutdown_event=self.interrupted,
            on_sample=lambda sample_phase, snapshot: self._trace_first_motion_sample(
                f"{phase}-{sample_phase}", dict(snapshot),
            ),
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=self._activation_speed_cap(),
            zero_until_stand_timeout_s=3.5,
            require_locomotion_confirmation=True,
            pre_motion_abort_reason=reason,
            activation_lifecycle=lifecycle,
        )
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
            "mc_debug_f64_subscription_enabled": self.mc_debug_f64_subscription_enabled,
            "get_mode_readback_fault": self.async_mode_readback_fault,
            "get_source_readback_fault": self.async_source_readback_fault,
            "readback_fault": "; ".join(
                fault for fault in (
                    self.async_mode_readback_fault,
                    self.async_source_readback_fault,
                ) if fault
            ) or None,
            "odom_source_stamp_fence": self.odom_source_stamp_fence.snapshot(),
            "odom_subscription_configured_qos": self.odom_subscription_configured_qos,
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

    def wait_for_fresh_route_replan_snapshot(
        self, *, previous_odom_stamp_ns: int,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        """Do not rearm a stopped route from a stale or duplicate odometry snapshot."""
        deadline = time.monotonic() + float(self.args.state_timeout)
        last: dict[str, object] = {"status": "NOT_OBSERVED"}
        while time.monotonic() < deadline:
            snapshot = self.sample_for_first_motion()
            freshness = route_replan_odom_freshness(
                snapshot,
                previous_stamp_ns=previous_odom_stamp_ns,
                maximum_age_s=float(self.args.state_timeout),
            )
            last = freshness
            if freshness.get("status") == "PASS":
                snapshot["route_replan_freshness"] = freshness
                return snapshot, freshness
            self._spin_for(0.005)
        raise TimeoutError(
            "route replan did not receive a fresh advancing odometry sample within "
            f"{float(self.args.state_timeout):.3f} s: {last}"
        )

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
                    acceptance_pose=goal,
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
                    or pose_goal_path_feasibility["terminal_entry_error_to_acceptance_goal_m"]
                    > POINT_GOAL_FINAL_TOLERANCE_M
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
            after_zero_abort=self.request_postwalk_stand_once,
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
            require_locomotion_confirmation=True,
            activation_lifecycle=self.activation_lifecycle,
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
                    POSE_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD if has_heading_target else None
                ),
                "pose_goal_terminal_direction_margin_rad": (
                    abs(heading_change_rad)
                    + POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD
                    - POSE_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD
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
                    "approach endpoint uses 0.05 rad bias to keep its curve feasible; terminal target uses a separate 0.115 rad bias derived from Run96 drift and the 5-degree minimum requested-turn gate; not calibrated across Runs93/95"
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

    def run_right_turn_qualification(
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
        turn = run_angular_turn_window(
            sample=self.sample_for_first_motion,
            publish_turn=lambda forward, angular: self._publish(forward, 0.0, angular),
            publish_zero=lambda: self._publish(0.0, 0.0, 0.0),
            registration=registration,
            set_action=set_action,
            joint_limits=self.joint_limits,
            baseline_forward_displacements_m=baseline_displacements,
            baseline_state_summary=stationary_baseline.get("physical_state_summary"),
            authorized=True,
            angular_velocity_rad_s=-TURN_COMMAND_RAD_S,
            forward_velocity_m_s=TURN_FORWARD_COMMAND_M_S,
            forward_warmup_s=TURN_FORWARD_WARMUP_S,
            pulse_duration_s=TURN_PULSE_S,
            pre_deadline_zero_margin_s=0.02,
            max_observed_planar_speed_m_s=0.2,
            zero_until_stand_timeout_s=3.5,
            extra_post_stop_observe_s=5.0,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_turn_displacement_m=0.25,
            turn_stop_reserve_m=0.05,
            shutdown_event=self.interrupted,
            on_sample=lambda phase, snapshot: self._trace_first_motion_sample(
                f"route-right-turn-{phase}", snapshot,
            ),
        )
        turn["direction"] = "right"
        turn["classification"] = "SIMULATION_ONLY_DERIVED_FOUNDATIONWALK_RIGHT_ARC"
        turn["turn_displacement_stop_reserve_source"] = {
            "run65_right_turn_max_additional_radial_displacement_after_zero_m": 0.04777158907978671,
            "selected_reserve_m": 0.05,
            "cap_m": 0.25,
            "active_command_trigger_m": 0.20,
        }
        turn["status_scope"] = "one bounded right-moving arc and measured physical stop; not a 90-degree corner"
        turn_start_s = float(turn.get("turn_start_monotonic_s") or 0.0)
        trace_rows = turn.get("samples", [])
        active_rows = [
            row for row in trace_rows
            if row.get("phase") == "turn-activation"
            and float(row.get("sample_monotonic_s", -1.0)) >= turn_start_s
            and isinstance(row.get("pose"), Mapping)
        ]
        origin_row = min(active_rows, key=lambda row: float(row["sample_monotonic_s"])) if active_rows else None
        peak_total_displacement = 0.0
        if origin_row is not None:
            origin_pose = origin_row["pose"]
            for row in trace_rows:
                if row.get("phase") not in {"turn-activation", "post-stop"}:
                    continue
                pose = row.get("pose")
                if not isinstance(pose, Mapping):
                    continue
                try:
                    displacement = math.hypot(
                        float(pose["x"]) - float(origin_pose["x"]),
                        float(pose["y"]) - float(origin_pose["y"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                peak_total_displacement = max(peak_total_displacement, displacement)
        rotation = turn.get("rotation", {})
        stop = turn.get("physical_stop_gate", {})
        zero_stop = turn.get("zero_stop", {})
        hard_deadline = float(turn.get("pulse_start_monotonic_s", 0.0)) + 5.0
        zero_start = float(zero_stop.get("start_monotonic_s", float("inf")))
        yaw_pass = (
            isinstance(rotation, Mapping)
            and rotation.get("status") == "TURN_DIRECTION_PASS"
            and float(rotation.get("signed_yaw_delta_rad", 0.0)) <= -TURN_MIN_HEADING_CHANGE_RAD
        )
        safety_rows = [row for row in trace_rows if row.get("phase") in {"turn-preflight", "turn-forward-warmup", "turn-activation", "post-stop"}]
        safety_clear = not turn.get("fault") and not turn.get("post_stop_fault") and all(
            row.get("safety_fault") is None for row in safety_rows
        )
        gates = {
            "control_handshake": turn.get("control_handshake_confirmed") is True,
            "signed_right_yaw_at_least_0p087_rad": yaw_pass,
            "all_runtime_safety_samples_clear": safety_clear,
            "total_turn_and_settling_displacement_le_0p25m": peak_total_displacement <= 0.25,
            "active_duration_le_5s": float(turn.get("pulse_actual_duration_s", 0.0)) <= 5.0,
            "zero_stream_started_by_hard_deadline": zero_start <= hard_deadline,
            "physical_stop_gate": stop.get("status") == "PASS",
            "zero_window_minimum_met": int(turn.get("zero_stop", {}).get("zero_publish_calls_returned", 0)) >= ZERO_STOP_MIN_MESSAGES,
        }
        turn["right_arc_acceptance"] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "turn_origin": dict(origin_row["pose"]) if origin_row else None,
            "maximum_total_turn_origin_displacement_m": peak_total_displacement,
            "hard_deadline_monotonic_s": hard_deadline,
            "zero_stream_start_monotonic_s": zero_start,
            "legacy_planar_motion_status": turn.get("planar_motion", {}).get("status"),
            "legacy_planar_gate_is_reported_separately": True,
        }
        return {
            "status": "PASS" if turn["right_arc_acceptance"]["status"] == "PASS" else "BLOCKED",
            "classification": "SIMULATION_ONLY_DERIVED_FOUNDATIONWALK_RIGHT_TURN_QUALIFICATION",
            "direction": "RIGHT",
            "forward_velocity_m_s": TURN_FORWARD_COMMAND_M_S,
            "angular_command_rad_s": -TURN_COMMAND_RAD_S,
            "maximum_active_window_s": TURN_FORWARD_WARMUP_S + TURN_PULSE_S,
            "turn": turn,
            "limitations": [
                "The 50 mm reserve is rounded up from Run65's measured 47.772 mm maximum post-zero radial drift; it is an empirical single-run bound, not a statistical guarantee.",
                "Passing qualifies only this short right-moving arc and its measured stop, not a 90-degree route corner.",
                "The installed interface provides simulator odometry and joint state but no foot contact force or MC stop acknowledgement.",
            ],
        }

    def run_zero_forward_table_alignment(
        self,
        *,
        registration: Mapping[str, Any],
        set_action: Mapping[str, Any],
        stationary_baseline: Mapping[str, Any],
        mission: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Qualify the source-pinned START→START_ALIGN yaw turn in one episode.

        The official simulator guide explicitly says that yaw-only starts are
        not guaranteed.  This loop therefore treats every 4.5 s command slice
        as an independently stopped experiment, never assumes success from
        geometry, and abandons the entire table mission on the first failed
        source/mode, physics, displacement, or stop gate.
        """
        alignment = mission.get("initial_alignment")
        if not isinstance(alignment, Mapping):
            return {"status": "BLOCKED", "fault": "table_alignment_missing_contract"}
        try:
            target_yaw = float(alignment["target_yaw_rad"])
            yaw_tolerance = float(alignment["target_yaw_tolerance_rad"])
            max_windows = int(alignment["max_turn_windows"])
            max_total_displacement = float(alignment["max_total_planar_displacement_m"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"status": "BLOCKED", "fault": f"table_alignment_invalid_contract:{type(exc).__name__}"}
        if (
            not all(math.isfinite(value) for value in (target_yaw, yaw_tolerance, max_total_displacement))
            or yaw_tolerance <= 0.0 or max_total_displacement <= 0.0 or max_windows <= 0
        ):
            return {"status": "BLOCKED", "fault": "table_alignment_nonfinite_or_unbounded_contract"}

        baseline_displacements = [
            float(window["forward_displacement_m"])
            for window in stationary_baseline.get("windows", [])
        ]
        baseline_state = stationary_baseline.get("physical_state_summary")
        origin_snapshot = self.sample_for_first_motion()
        origin_pose = origin_snapshot.get("pose")
        if not isinstance(origin_pose, Mapping):
            return {"status": "BLOCKED", "fault": "table_alignment_origin_pose_missing"}
        try:
            runtime_start_gate = validate_runtime_raw_odom_table_approach_start(
                mission,
                {
                    "x_m": float(origin_pose["x"]),
                    "y_m": float(origin_pose["y"]),
                    "z_m": float(origin_pose["z"]),
                    "yaw_rad": float(origin_pose["yaw"]),
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "status": "BLOCKED",
                "fault": f"table_alignment_runtime_start_gate_invalid:{type(exc).__name__}",
            }
        if runtime_start_gate["status"] != "PASS":
            return {
                "status": "BLOCKED",
                "fault": "table_alignment_runtime_start_outside_static_envelope",
                "runtime_start_gate": runtime_start_gate,
            }
        origin_fault = snapshot_fault(
            origin_snapshot, self.joint_limits,
            state_timeout_s=self.args.state_timeout,
            min_root_z=self.args.min_root_z,
            max_tilt_deg=self.args.max_tilt_deg,
            max_observed_planar_speed_m_s=0.2,
        )
        if origin_fault or not stand_mode_readback_confirmed(origin_snapshot) or not isinstance(origin_pose, Mapping):
            return {
                "status": "BLOCKED",
                "fault": "table_alignment_requires_fresh_stable_STAND:" + str(origin_fault or "pose_or_readback"),
                "runtime_start_gate": runtime_start_gate,
            }
        try:
            origin_x, origin_y = float(origin_pose["x"]), float(origin_pose["y"])
        except (KeyError, TypeError, ValueError):
            return {"status": "BLOCKED", "fault": "table_alignment_origin_pose_invalid"}

        turns: list[dict[str, Any]] = []
        lifecycle = self.activation_lifecycle
        current_action = dict(set_action)
        final_snapshot: Mapping[str, Any] = origin_snapshot
        for index in range(max_windows):
            if index > 0:
                lifecycle, next_action, activation_gate = self.begin_source_grounded_route_activation(
                    registration=registration,
                    phase=f"table-start-align-window-{index + 1}-rearm",
                )
                rearm = {
                    "set_mc_action_ld": next_action,
                    "activation_gate": activation_gate,
                    "lifecycle_after_activation": lifecycle.snapshot(),
                }
                if activation_gate.get("status") != "READY":
                    recovery = self.recover_activation_to_stand(
                        lifecycle=lifecycle,
                        registration=registration,
                        set_action=next_action,
                        baseline=stationary_baseline,
                        reason=str(activation_gate.get("fault") or activation_gate.get("terminal_reason") or "table yaw alignment rearm was not ready"),
                        phase=f"table-start-align-window-{index + 1}-recovery",
                    )
                    return {
                        "status": "BLOCKED", "fault": "table_alignment_rearm_not_confirmed",
                        "origin_pose": dict(origin_pose), "turns": turns,
                        "failed_rearm": {**rearm, "stand_recovery": recovery},
                    }
                self.activation_lifecycle = lifecycle
                current_action = next_action

            before = self.sample_for_first_motion()
            before_fault = snapshot_fault(
                before, self.joint_limits,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=0.2,
            )
            pose = before.get("pose")
            if before_fault or not stand_mode_readback_confirmed(before) or not isinstance(pose, Mapping):
                return {
                    "status": "BLOCKED", "fault": "table_alignment_rearm_STAND_not_safe:" + str(before_fault or "pose_or_readback"),
                    "origin_pose": dict(origin_pose), "turns": turns,
                    "runtime_start_gate": runtime_start_gate,
                }
            try:
                yaw_error = math.atan2(math.sin(target_yaw - float(pose["yaw"])), math.cos(target_yaw - float(pose["yaw"])))
                current_displacement = math.hypot(float(pose["x"]) - origin_x, float(pose["y"]) - origin_y)
            except (KeyError, TypeError, ValueError):
                return {"status": "BLOCKED", "fault": "table_alignment_rearm_pose_invalid", "turns": turns}
            if current_displacement > max_total_displacement:
                return {
                    "status": "BLOCKED", "fault": "table_alignment_total_displacement_exceeded_before_next_turn",
                    "origin_pose": dict(origin_pose), "turns": turns,
                    "runtime_start_gate": runtime_start_gate,
                    "current_total_displacement_m": current_displacement,
                }
            if abs(yaw_error) <= yaw_tolerance:
                final_snapshot = before
                break
            angular = math.copysign(TURN_COMMAND_RAD_S, yaw_error)
            turn = run_angular_turn_window(
                sample=self.sample_for_first_motion,
                publish_turn=lambda forward, angular_velocity: self._publish(forward, 0.0, angular_velocity),
                publish_zero=lambda: self._publish(0.0, 0.0, 0.0),
                registration=registration,
                set_action=current_action,
                joint_limits=self.joint_limits,
                baseline_forward_displacements_m=baseline_displacements,
                baseline_state_summary=baseline_state,
                authorized=True,
                angular_velocity_rad_s=angular,
                forward_velocity_m_s=0.0,
                forward_warmup_s=0.0,
                pulse_duration_s=TURN_PULSE_S,
                pre_deadline_zero_margin_s=0.02,
                max_observed_planar_speed_m_s=0.2,
                zero_until_stand_timeout_s=3.5,
                extra_post_stop_observe_s=5.0,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_turn_displacement_m=max_total_displacement,
                turn_stop_reserve_m=0.0,
                shutdown_event=self.interrupted,
                on_sample=lambda phase, snapshot, n=index + 1: self._trace_first_motion_sample(
                    f"table-start-align-window-{n}-{phase}", snapshot,
                ),
            )
            turn["window_index"] = index + 1
            turn["requested_target_yaw_rad"] = target_yaw
            turn["pre_turn_yaw_error_rad"] = yaw_error
            turns.append(turn)
            if turn.get("status") != "PHYSICAL_TURN_AND_STOP_PASS":
                return {
                    "status": "BLOCKED", "fault": "table_alignment_turn_not_qualified:" + str(turn.get("status")),
                    "origin_pose": dict(origin_pose), "turns": turns,
                    "runtime_start_gate": runtime_start_gate,
                }
            final_snapshot = self.sample_for_first_motion()

        final_pose = final_snapshot.get("pose") if isinstance(final_snapshot, Mapping) else None
        if not isinstance(final_pose, Mapping):
            return {"status": "BLOCKED", "fault": "table_alignment_final_pose_missing", "turns": turns}
        try:
            final_yaw_error = math.atan2(math.sin(target_yaw - float(final_pose["yaw"])), math.cos(target_yaw - float(final_pose["yaw"])))
            final_displacement = math.hypot(float(final_pose["x"]) - origin_x, float(final_pose["y"]) - origin_y)
        except (KeyError, TypeError, ValueError):
            return {"status": "BLOCKED", "fault": "table_alignment_final_pose_invalid", "turns": turns}
        passed = (
            abs(final_yaw_error) <= yaw_tolerance
            and final_displacement <= max_total_displacement
            and bool(turns)
            and all(turn.get("status") == "PHYSICAL_TURN_AND_STOP_PASS" for turn in turns)
            and stand_mode_readback_confirmed(final_snapshot)
        )
        return {
            "status": "PASS" if passed else "BLOCKED",
            "classification": "SIMULATION_ONLY_ZERO_FORWARD_TABLE_START_ALIGNMENT",
            "origin_pose": dict(origin_pose),
            "runtime_start_gate": runtime_start_gate,
            "final_pose": dict(final_pose),
            "target_yaw_rad": target_yaw,
            "final_yaw_error_rad": final_yaw_error,
            "yaw_tolerance_rad": yaw_tolerance,
            "final_total_planar_displacement_m": final_displacement,
            "maximum_total_planar_displacement_m": max_total_displacement,
            "turns": turns,
            "turn_windows_used": len(turns),
            "turn_windows_cap": max_windows,
            "physical_stop_after_each_window": True,
            "fault": None if passed else "table_alignment_target_or_stop_gate_not_met",
        }

    def run_short_route_integration(
        self,
        *,
        registration: Mapping[str, Any],
        set_action: Mapping[str, Any],
        stationary_baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute one bounded waypoint, curved-path, or qualified short-route trial."""
        curved_path_diagnostic = bool(
            getattr(self.args, "route_curved_path_diagnostic", False)
        )
        waypoint_b_short_diagnostic = bool(
            getattr(self.args, "route_waypoint_b_short_diagnostic", False)
        )
        waypoint_b_extended_diagnostic = bool(
            getattr(self.args, "route_waypoint_b_extended_diagnostic", False)
        )
        route_config_mode = bool(getattr(self.args, "route_config", None))
        short_scene_route_mode = bool(getattr(self.args, "short_scene_route", False))
        table_approach_mission_mode = bool(getattr(self.args, "table_approach_mission", False))
        configured_docking_route_mode = route_config_mode and not table_approach_mission_mode
        aligned_start_table_approach_mode = bool(
            getattr(self.args, "aligned_start_table_approach", False)
        )
        native_x2_timestep_mode = bool(getattr(self.args, "native_x2_timestep", False))
        native_mjdata_route_feedback_mode = bool(
            getattr(self.args, "raicom_native_mjdata_route_feedback", False)
        )
        zero_forward_heading_acquisition_mode = bool(
            getattr(self.args, "raicom_zero_forward_heading_acquisition", False)
        )
        continuous_ten_second_transit_mode = bool(
            getattr(self.args, "raicom_continuous_ten_second_transit", False)
        )
        continuous_table_approach_mode = bool(
            getattr(self.args, "raicom_continuous_table_approach", False)
        )
        fixed_window_table_approach_mode = bool(
            getattr(self.args, "raicom_fixed_window_table_approach", False)
        )
        fixed_window_open_segment_yaw_acquisition_mode = bool(
            fixed_window_table_approach_mode
        )
        segmented_table_envelope_mode = (
            continuous_table_approach_mode or fixed_window_table_approach_mode
        )
        forward_gait_yaw_inversion_mode = bool(
            getattr(self.args, "raicom_forward_gait_yaw_inversion", False)
        )
        native_nominal_straight_gait_diagnostic_mode = bool(
            getattr(self.args, "raicom_native_nominal_straight_gait_diagnostic", False)
        )
        waypoint_b_only = bool(
            getattr(self.args, "route_waypoint_b_only", False)
            or waypoint_b_short_diagnostic
            or waypoint_b_extended_diagnostic
            or curved_path_diagnostic
        )
        result: dict[str, Any] = {
            "classification": (
                "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING"
                if short_scene_route_mode else
                "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION"
                if aligned_start_table_approach_mode else
                "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION"
                if table_approach_mission_mode else
                "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE"
                if route_config_mode else
                "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC"
                if curved_path_diagnostic else
                "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC"
                if waypoint_b_extended_diagnostic else
                "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC"
                if waypoint_b_short_diagnostic else
                "SIMULATION_ONLY_WAYPOINT_B_AND_PHYSICAL_STOP"
                if waypoint_b_only else "SIMULATION_ONLY_SHORT_ROUTE_INTEGRATION"
            ),
            "status": "NOT_RUN",
            "route_attempt_id": self.args.attempt_id,
            "segments": [],
            "zero_forward_heading_acquisition": {
                "enabled": bool(
                    zero_forward_heading_acquisition_mode
                    or fixed_window_open_segment_yaw_acquisition_mode
                ),
                "classification": (
                    "SIMULATION_ONLY_AIMDK_YAW_SOURCE_ACQUISITION"
                    if (
                        zero_forward_heading_acquisition_mode
                        or fixed_window_open_segment_yaw_acquisition_mode
                    ) else None
                ),
                "scope": (
                    "INITIAL_WAYPOINT_DEPARTURE_ONLY"
                    if zero_forward_heading_acquisition_mode else
                    "EACH_FIXED_WINDOW_IN_OPEN_START_TO_WAYPOINT_SEGMENT_ONLY"
                    if fixed_window_open_segment_yaw_acquisition_mode else None
                ),
                "forward_velocity_m_s": 0.0,
                "angular_command_magnitude_rad_s": (
                    ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S
                    if (
                        zero_forward_heading_acquisition_mode
                        or fixed_window_open_segment_yaw_acquisition_mode
                    ) else None
                ),
                "maximum_duration_s": (
                    ZERO_FORWARD_HEADING_ACQUISITION_MAX_S
                    if (
                        zero_forward_heading_acquisition_mode
                        or fixed_window_open_segment_yaw_acquisition_mode
                    ) else None
                ),
                "minimum_response_rad": (
                    ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD
                    if (
                        zero_forward_heading_acquisition_mode
                        or fixed_window_open_segment_yaw_acquisition_mode
                    ) else None
                ),
                "windows": [],
            },
            "continuous_ten_second_transit": {
                "enabled": continuous_ten_second_transit_mode,
                "classification": (
                    "SIMULATION_ONLY_CONTINUOUS_TEN_SECOND_TABLE_TRANSIT"
                    if continuous_ten_second_transit_mode else None
                ),
                "maximum_window_s": (
                    TABLE_APPROACH_CONTINUOUS_WINDOW_S
                    if continuous_ten_second_transit_mode else None
                ),
                "maximum_windows_total": (
                    TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS
                    if continuous_ten_second_transit_mode else None
                ),
                "aggregate_nonzero_limit_s": 75.0 if continuous_ten_second_transit_mode else None,
                "physical_stop_after_each_window": True,
            },
            "continuous_table_approach": {
                "enabled": continuous_table_approach_mode,
                "classification": (
                    "SIMULATION_ONLY_CONTINUOUS_TABLE_APPROACH_SINGLE_GAIT_STREAM"
                    if continuous_table_approach_mode else None
                ),
                "maximum_aggregate_nonzero_s": (
                    TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                    if continuous_table_approach_mode else None
                ),
                "intermediate_waypoint_stop_behavior": (
                    "ROLLING" if continuous_table_approach_mode else None
                ),
                "terminal_stop_behavior": (
                    "VERIFY_PHYSICAL_STOP" if continuous_table_approach_mode else None
                ),
                "cross_track_limit_by_step_m": (
                    {
                        "WAYPOINT": CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
                        "TABLE_APPROACH": CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
                    }
                    if continuous_table_approach_mode else None
                ),
                "static_tracking_inflation_by_step_requirement_m": (
                    {
                        "START->WAYPOINT": CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
                        "WAYPOINT->TABLE_APPROACH": CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
                    }
                    if continuous_table_approach_mode else None
                ),
            },
            "fixed_window_table_approach": {
                "enabled": fixed_window_table_approach_mode,
                "classification": (
                    "SIMULATION_ONLY_FIXED_FIVE_SECOND_STRAIGHT_OR_RUN65_ARC_TABLE_APPROACH"
                    if fixed_window_table_approach_mode else None
                ),
                "maximum_window_s": 5.0 if fixed_window_table_approach_mode else None,
                "maximum_windows_total": 15 if fixed_window_table_approach_mode else None,
                "aggregate_nonzero_limit_s": 75.0 if fixed_window_table_approach_mode else None,
                "arc_command_magnitude_rad_s": (
                    POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S
                    if fixed_window_table_approach_mode else None
                ),
                "arc_deadband_rad_s": (
                    FIXED_WINDOW_ARC_DEADBAND_RAD_S
                    if fixed_window_table_approach_mode else None
                ),
                "open_segment_yaw_acquisition": (
                    {
                        "scope": "START_TO_WAYPOINT_ONLY",
                        "forward_velocity_m_s": 0.0,
                        "angular_command_magnitude_rad_s": ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
                        "maximum_duration_s": ZERO_FORWARD_HEADING_ACQUISITION_MAX_S,
                        "minimum_measured_response_rad": ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD,
                        "static_yaw_envelope_rad": FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
                    }
                    if fixed_window_table_approach_mode else None
                ),
                "control_lifecycle": (
                    "OPEN_SEGMENT_ZERO_FORWARD_YAW_ACQUISITION_IF_NEEDED->FIXED_5S_STRAIGHT_OR_ARC->COMPOSITE_STOP->REPLAN"
                    if fixed_window_table_approach_mode else None
                ),
            },
            "forward_gait_yaw_mapping": {
                "enabled": forward_gait_yaw_inversion_mode,
                "classification": (
                    "SIMULATION_ONLY_RAICOM_FORWARD_GAIT_YAW_SIGN_INVERTED"
                    if forward_gait_yaw_inversion_mode else None
                ),
                "stationary_source_acquisition_mapping": "documented_direct_sign",
                "forward_gait_mapping": (
                    "negative_of_standard_odom_bearing_feedback"
                    if forward_gait_yaw_inversion_mode else "standard_odom_bearing_feedback"
                ),
            },
            "straight_gait_diagnostic": {
                "enabled": native_nominal_straight_gait_diagnostic_mode,
                "classification": (
                    "SIMULATION_ONLY_NATIVE_NOMINAL_STRAIGHT_GAIT_DIAGNOSTIC"
                    if native_nominal_straight_gait_diagnostic_mode else None
                ),
                "forward_velocity_m_s": 0.12 if native_nominal_straight_gait_diagnostic_mode else None,
                "angular_velocity_rad_s": 0.0 if native_nominal_straight_gait_diagnostic_mode else None,
                "maximum_window_s": (
                    TABLE_APPROACH_CONTINUOUS_WINDOW_S
                    if native_nominal_straight_gait_diagnostic_mode else None
                ),
            },
            "native_mjdata_route_feedback": {
                "enabled": native_mjdata_route_feedback_mode,
                "classification": (
                    "SIMULATION_ONLY_READ_ONLY_NATIVE_MJDATA_GROUND_TRUTH_ROUTE_FEEDBACK"
                    if native_mjdata_route_feedback_mode else None
                ),
                "status": "NOT_INITIALIZED" if native_mjdata_route_feedback_mode else "NOT_ENABLED",
                "source": str(Path(self.args.output_dir) / CAPTURE_TRACE_NAME)
                if native_mjdata_route_feedback_mode else None,
            },
            "route_scope": (
                "LOCAL01_START_HEADING_TWO_TARGET_SHORT_ROUTE"
                if short_scene_route_mode else
                "LOCAL01_ALIGNED_START_WAYPOINT_TABLE_APPROACH"
                if aligned_start_table_approach_mode else
                "LOCAL01_START_ALIGN_WAYPOINT_TABLE_APPROACH"
                if table_approach_mission_mode else
                "CONFIGURED_START_WAYPOINT_TABLE_DOCKING"
                if route_config_mode else
                "SHALLOW_CURVED_FIXED_POINT_ENDPOINT"
                if curved_path_diagnostic else
                "EXTENDED_THREE_WINDOW_WAYPOINT_B_DIAGNOSTIC"
                if waypoint_b_extended_diagnostic else
                "SHORT_DIAGNOSTIC_WAYPOINT_ONLY"
                if waypoint_b_short_diagnostic else
                "WAYPOINT_B_ONLY" if waypoint_b_only else "B_RIGHT_TURN_C"
            ),
            "acceptance_scope": (
                "STATIC_SCENE_VALIDATED_SHORT_ROUTE_AND_RESET_ODOM_GATE_ONLY_NOT_TABLE_DOCKING"
                if short_scene_route_mode else
                "DERIVED_ALIGNED_START_STATIC_SAFE_PREFIX_ONLY_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION"
                if aligned_start_table_approach_mode else
                "STATIC_SAFE_PREFIX_ONLY_NOT_TABLE_DOCKING_OR_MANIPULATION"
                if table_approach_mission_mode else
                "LOCAL01_CONFIGURED_WAYPOINT_AND_TABLE_DOCKING_POSE"
                if route_config_mode else
                "CURVED_PATH_ROUTE_LIFECYCLE_DIAGNOSTIC"
                if curved_path_diagnostic else
                "EXTENDED_PROFILE_ORIGINAL_0_40_M_WAYPOINT_B_DIAGNOSTIC"
                if waypoint_b_extended_diagnostic else
                "DIAGNOSTIC_ONLY_NOT_ORIGINAL_0_40_M_WAYPOINT_B_ACCEPTANCE"
                if waypoint_b_short_diagnostic else
                "WAYPOINT_B_ONLY" if waypoint_b_only else "B_RIGHT_TURN_C"
            ),
            "fault": None,
        }
        evidence_root = Path(self.args.output_dir)
        local01_route_input: Mapping[str, Any] | None = None
        workcell_path = evidence_root / "workcell_geometry_preflight.json"
        native_route_reader: NativeMjDataLivePoseReader | None = None
        try:
            if native_mjdata_route_feedback_mode:
                if not (
                    native_x2_timestep_mode
                    and table_approach_mission_mode
                    and aligned_start_table_approach_mode
                ):
                    raise ValueError(
                        "native MjData route feedback requires the aligned native-timestep TABLE_APPROACH mission"
                    )
                native_route_reader = NativeMjDataLivePoseReader(
                    evidence_root / CAPTURE_TRACE_NAME,
                    maximum_age_s=MAX_POSE_AGE_S,
                )
                result["native_mjdata_route_feedback"].update({
                    "status": "PENDING_FIRST_FRESH_NATIVE_SAMPLE",
                    "native_capture_is_read_only": True,
                    "ros_retained_for": [
                        "MC mode/source readbacks", "IMU", "leg state", "leg command", "safety status",
                    ],
                })
            if native_x2_timestep_mode:
                capture_path = evidence_root / "set_action_capture_result.json"
                if capture_path.is_symlink() or not capture_path.is_file():
                    raise ValueError("native timestep runtime gate lacks the direct-stand capture result")
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
                try:
                    observed_timestep_s = float(
                        capture["native_mjdata"]["support"]["sample"]["timestep_s"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("native timestep runtime gate lacks a finite MjData timestep sample") from exc
                if not math.isclose(observed_timestep_s, 0.001, abs_tol=1e-12):
                    raise ValueError(
                        f"native timestep runtime gate expected 0.001 s, observed {observed_timestep_s:.9f} s"
                    )
                result["native_x2_timestep_runtime_gate"] = {
                    "status": "PASS",
                    "observed_mjdata_timestep_s": observed_timestep_s,
                    "expected_native_x2_timestep_s": 0.001,
                    "source": str(capture_path),
                    "before_nonzero_motion": True,
                }
            if short_scene_route_mode:
                static_path = evidence_root / "local01_scene" / "static_scene_preflight.json"
                bundle_path = evidence_root / "local01_scene" / "runtime_bundle_manifest.json"
                route_config_path = Path(self.args.route_config)
                manifest_path = evidence_root / "route_config_input.json"
                if any(path.is_symlink() or not path.is_file() for path in (
                    static_path, bundle_path, route_config_path, manifest_path,
                )):
                    raise ValueError("Local01 scene route inputs/preflight manifests are missing or unsafe")
                static_scene = json.loads(static_path.read_text(encoding="utf-8"))
                scene_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                route_config_hash = hashlib.sha256(route_config_path.read_bytes()).hexdigest()
                if (
                    static_scene.get("status") != "PASS_STATIC_SCENE_CLEARANCE"
                    or static_scene.get("native_runtime_name_contract", {}).get("status") != "PASS_NATIVE_RUNTIME_NAMES"
                    or static_scene.get("compiled_name_only_equivalence", {}).get("status") != "PASS_COMPILED_PHYSICS_EQUIVALENT"
                    or scene_bundle.get("x2_attach_prefix_change") != {"source": "x2_", "runtime": ""}
                    or scene_bundle.get("generated_runtime_name_contract", {}).get("status") != "PASS_NATIVE_RUNTIME_NAMES"
                    or scene_bundle.get("generated_wrapper_physics_equivalence", {}).get("status") != "PASS_COMPILED_PHYSICS_EQUIVALENT"
                    or static_scene.get("mj_step_calls") != 0
                    or static_scene.get("source_local01_commit") != input_manifest.get("source_local01_commit")
                    or static_scene.get("source_scene_world_poses_sha256") != input_manifest.get("source_scene_world_poses_sha256")
                    or static_scene.get("scene_workspace_sha256")
                        != scene_bundle.get("source_files_sha256", {}).get("scene_workspace.xml")
                    or scene_bundle.get("source_local01_commit") != input_manifest.get("source_local01_commit")
                    or scene_bundle.get("classification") != "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER"
                    or input_manifest.get("derived_route_sha256") != route_config_hash
                ):
                    raise ValueError("Local01 actual-scene preflight, runtime wrapper, or derived route hash did not match")
                route_input = json.loads(route_config_path.read_text(encoding="utf-8"))
                validate_local01_short_scene_route_input(route_input)
                local01_route_input = route_input
                workcell = {
                    "status": "PASS",
                    "classification": "STATIC_RAICOM_ROOM_TABLE_CHAIR_MUG_SCENE; no dynamic collision observer",
                    "robot_collision_footprint_radial_upper_bound_m": float(
                        static_scene["robot_collision_footprint_radial_upper_bound_m"]
                    ),
                    "verified_local_clear_radius_m": float(static_scene["verified_local_clear_radius_m"]),
                    "minimum_swept_aabb_clearance_lower_bound_m": float(
                        static_scene["minimum_swept_aabb_clearance_lower_bound_m"]
                    ),
                    "static_sample_count": int(static_scene["static_sample_count"]),
                    "mj_step_calls": int(static_scene["mj_step_calls"]),
                    "active_world_obstacles": ["RAICOM room walls", "table", "chair", "mug"],
                    "static_preflight_path": str(static_path),
                    "scene_bundle_path": str(bundle_path),
                }
                result["local01_scene_static_preflight"] = static_scene
                result["local01_scene_runtime_bundle"] = scene_bundle
                result["route_config_input"] = {
                    "sha256": route_config_hash,
                    "source_local01_commit": input_manifest.get("source_local01_commit"),
                    "source_scene_world_poses_sha256": input_manifest.get("source_scene_world_poses_sha256"),
                    "route_id": route_input.get("route_id"),
                    "route_kind": route_input.get("route_kind"),
                    "frame_id": route_input.get("frame_id"),
                    "staged_path": str(route_config_path),
                    "parent_manifest_verified": True,
                }
            elif table_approach_mission_mode:
                static_path = evidence_root / "local01_scene" / "static_scene_preflight.json"
                bundle_path = evidence_root / "local01_scene" / "runtime_bundle_manifest.json"
                table_plan_path = evidence_root / "table_approach_static_plan.json"
                route_config_path = Path(self.args.route_config)
                manifest_path = evidence_root / "route_config_input.json"
                if any(path.is_symlink() or not path.is_file() for path in (
                    static_path, bundle_path, table_plan_path, route_config_path, manifest_path,
                )):
                    raise ValueError("table-approach mission inputs/preflight manifests are missing or unsafe")
                static_scene = json.loads(static_path.read_text(encoding="utf-8"))
                scene_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                table_plan = json.loads(table_plan_path.read_text(encoding="utf-8"))
                input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                route_config_hash = hashlib.sha256(route_config_path.read_bytes()).hexdigest()
                mission = json.loads(route_config_path.read_text(encoding="utf-8"))
                if aligned_start_table_approach_mode:
                    validate_runtime_raw_odom_aligned_start_table_approach_mission(mission)
                else:
                    validate_runtime_raw_odom_table_approach_mission(mission)
                expected_bundle_classification = (
                    "SIMULATION_ONLY_ALIGNED_START_NATIVE_X2_TIMESTEP_SCENE_WRAPPER_V1"
                    if native_x2_timestep_mode else
                    "SIMULATION_ONLY_ALIGNED_START_SCENE_WRAPPER_V1"
                    if aligned_start_table_approach_mode
                    else "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER"
                )
                expected_equivalence_status = (
                    "PASS_ALIGNED_START_NATIVE_TIMESTEP_PHYSICAL_INVARIANTS"
                    if native_x2_timestep_mode else
                    "PASS_ALIGNED_START_PHYSICAL_INVARIANTS"
                    if aligned_start_table_approach_mode
                    else "PASS_COMPILED_PHYSICS_EQUIVALENT"
                )
                expected_plan_status = (
                    "STATIC_GEOMETRY_REVIEW_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_DYNAMICALLY_QUALIFIED"
                    if aligned_start_table_approach_mode
                    else "STATIC_GEOMETRY_REVIEW_ONLY_TABLE_APPROACH_NOT_DYNAMICALLY_QUALIFIED"
                )
                if segmented_table_envelope_mode:
                    try:
                        tracking_by_segment = table_plan["clearance_assumptions"][
                            "xy_tracking_inflation_by_segment_m"
                        ]
                        expected_tracking_by_segment = {
                            "START->WAYPOINT": CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
                            "WAYPOINT->TABLE_APPROACH": CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
                        }
                        observed_tracking_by_segment = {
                            key: float(tracking_by_segment[key])
                            for key in expected_tracking_by_segment
                        }
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "continuous table route static plan lacks its per-segment numeric tracking envelope"
                        ) from exc
                    if any(
                        abs(observed_tracking_by_segment[key] - expected) > 1e-9
                        for key, expected in expected_tracking_by_segment.items()
                    ):
                        raise ValueError(
                            "continuous table route requires a static plan generated with its exact "
                            "per-segment tracking envelopes"
                        )
                if (
                    static_scene.get("status") != "PASS_STATIC_SCENE_CLEARANCE"
                    or static_scene.get("native_runtime_name_contract", {}).get("status") != "PASS_NATIVE_RUNTIME_NAMES"
                    or scene_bundle.get("x2_attach_prefix_change") != {"source": "x2_", "runtime": ""}
                    or scene_bundle.get("classification") != expected_bundle_classification
                    or scene_bundle.get("generated_runtime_name_contract", {}).get("status") != "PASS_NATIVE_RUNTIME_NAMES"
                    or scene_bundle.get("generated_wrapper_physics_equivalence", {}).get("status") != expected_equivalence_status
                    or scene_bundle.get("generated_native_robot_equivalence", {}).get("status") != "PASS_NATIVE_ROBOT_UNDER_SCENE_START_TRANSFORM"
                    or bool(scene_bundle.get("aligned_start", {}).get("enabled")) != aligned_start_table_approach_mode
                    or bool(scene_bundle.get("native_x2_timestep", {}).get("enabled")) != native_x2_timestep_mode
                    or (
                        native_x2_timestep_mode
                        and scene_bundle.get("native_x2_timestep", {}).get("native_x2_source_timestep_s") != 0.001
                    )
                    or table_plan.get("status") != expected_plan_status
                    or table_plan.get("safe_table_approach_mission", {}).get("classification") != mission.get("classification")
                    or table_plan.get("safe_table_approach_mission", {}).get("table_docking_reference_unchanged") is not True
                    or input_manifest.get("derived_route_sha256") != route_config_hash
                    or input_manifest.get("source_static_plan_sha256") != hashlib.sha256(table_plan_path.read_bytes()).hexdigest()
                    or bool(input_manifest.get("aligned_start_table_approach")) != aligned_start_table_approach_mode
                ):
                    raise ValueError("table-approach scene wrapper, static plan, or mission hash did not match")
                required_segments = (
                    {("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH")}
                    if aligned_start_table_approach_mode else {
                        ("START", "START_ALIGN"), ("START_ALIGN", "WAYPOINT"),
                        ("WAYPOINT", "TABLE_APPROACH"),
                    }
                )
                passing_segments = {
                    (str(item.get("from")), str(item.get("to")))
                    for item in table_plan.get("segments", []) if isinstance(item, Mapping)
                    and item.get("status") == "PASS_STATIC_ONLY"
                }
                if not required_segments.issubset(passing_segments):
                    raise ValueError("table-approach static plan lacks passing safe-prefix segments")
                local01_route_input = mission
                approach_segments = [
                    item for item in table_plan.get("segments", []) if isinstance(item, Mapping)
                    and (item.get("from"), item.get("to")) == ("WAYPOINT", "TABLE_APPROACH")
                    and item.get("status") == "PASS_STATIC_ONLY"
                ]
                if not approach_segments:
                    raise ValueError("table-approach segment static clearance evidence is missing")
                min_approach_clearance = min(
                    float(item["tracking_inflated_aabb_lower_bound_m"]) for item in approach_segments
                )
                workcell = {
                    "status": "PASS",
                    "classification": (
                        "STATIC_DERIVED_ALIGNED_START_RAICOM_SAFE_PREFIX_ROOM_TABLE_CHAIR_MUG_SCENE; no dynamic collision observer"
                        if aligned_start_table_approach_mode else
                        "STATIC_RAICOM_SAFE_PREFIX_ROOM_TABLE_CHAIR_MUG_SCENE; no dynamic collision observer"
                    ),
                    "robot_collision_footprint_radial_upper_bound_m": float(
                        static_scene["robot_collision_footprint_radial_upper_bound_m"]
                    ),
                    "verified_local_clear_radius_m": float(static_scene["verified_local_clear_radius_m"]),
                    "minimum_swept_aabb_clearance_lower_bound_m": min_approach_clearance,
                    "active_world_obstacles": ["RAICOM room walls", "table", "chair", "mug"],
                    "static_preflight_path": str(table_plan_path),
                    "scene_bundle_path": str(bundle_path),
                    "aligned_start_table_approach": aligned_start_table_approach_mode,
                    "native_x2_timestep": native_x2_timestep_mode,
                }
                result["local01_scene_static_preflight"] = static_scene
                result["local01_scene_runtime_bundle"] = scene_bundle
                result["table_approach_static_plan"] = table_plan
                result["route_config_input"] = {
                    "sha256": route_config_hash,
                    "source_local01_commit": input_manifest.get("source_local01_commit"),
                    "source_scene_world_poses_sha256": input_manifest.get("source_scene_world_poses_sha256"),
                    "mission_id": mission.get("mission_id"),
                    "classification": mission.get("classification"),
                    "frame_id": mission.get("frame_id"),
                    "aligned_start_table_approach": aligned_start_table_approach_mode,
                    "staged_path": str(route_config_path),
                    "parent_manifest_verified": True,
                }
            else:
                if workcell_path.is_symlink() or not workcell_path.is_file():
                    raise ValueError("host workcell geometry preflight is missing")
                workcell = json.loads(workcell_path.read_text(encoding="utf-8"))
                if workcell.get("status") != "PASS":
                    raise ValueError("host workcell geometry preflight did not pass")
            turn_provenance = None
            if not waypoint_b_only and not route_config_mode:
                turn_provenance = validate_route_turn_qualification_evidence(
                    self.args.route_turn_qualification_evidence
                )
            if short_scene_route_mode or table_approach_mission_mode:
                result["scene_geometry_preflight"] = workcell
            else:
                result["workcell_geometry_preflight"] = workcell
            if turn_provenance is not None:
                result["qualified_right_turn_evidence"] = turn_provenance
            if route_config_mode and not short_scene_route_mode and not table_approach_mission_mode:
                route_config_path = Path(self.args.route_config)
                manifest_path = evidence_root / "route_config_input.json"
                if (
                    route_config_path.is_symlink() or not route_config_path.is_file()
                    or manifest_path.is_symlink() or not manifest_path.is_file()
                ):
                    raise ValueError("configured route or its parent-staged input manifest is missing")
                route_config_sha256 = hashlib.sha256(route_config_path.read_bytes()).hexdigest()
                input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if input_manifest.get("source_sha256") != route_config_sha256:
                    raise ValueError("configured route input hash differs from its staged parent manifest")
                result["route_config_input"] = {
                    "sha256": route_config_sha256,
                    "route_id": load_waypoint_docking_pose_input(route_config_path)["route_id"],
                    "staged_path": str(route_config_path),
                    "parent_manifest_verified": True,
                }
        except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
            result["status"] = "BLOCKED"
            result["fault"] = f"route_preflight:{type(exc).__name__}:{exc}"
            return result

        def native_pose_from_snapshot(snapshot: Mapping[str, Any]) -> NativeMjDataLivePose:
            if native_route_reader is None:
                raise ValueError("native MjData route feedback is not configured")
            try:
                sample_time = float(snapshot["sample_monotonic_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("route sample has no monotonic time for native MjData freshness") from exc
            try:
                native = native_route_reader.poll(now_monotonic_s=sample_time)
            except (MjDataCaptureError, OSError, ValueError) as exc:
                raise ValueError(f"native MjData route feedback unavailable: {exc}") from exc
            if native is None:
                raise ValueError("native MjData route feedback has no complete sample yet")
            reader_snapshot = native_route_reader.snapshot()
            if reader_snapshot.get("generation_changed") is True:
                raise ValueError("native MjData generation changed during route execution")
            result["native_mjdata_route_feedback"].update({
                "status": "FRESH_NATIVE_SAMPLE",
                "reader": reader_snapshot,
            })
            return native

        def route_feedback_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
            if native_route_reader is None:
                return dict(snapshot)
            native = native_pose_from_snapshot(snapshot)
            result_snapshot = dict(snapshot)
            native_stamp_ns = int(round(native.sim_time_s * 1_000_000_000.0))
            ros_pose = result_snapshot.get("pose")
            result_snapshot["ros_odom_pose"] = dict(ros_pose) if isinstance(ros_pose, Mapping) else None
            pose = dict(ros_pose) if isinstance(ros_pose, Mapping) else {}
            pose.update({
                "x": native.x_m,
                "y": native.y_m,
                "z": native.z_m,
                "roll": native.roll_rad,
                "pitch": native.pitch_rad,
                "yaw": native.yaw_rad,
                "vx": native.vx_m_s,
                "vy": native.vy_m_s,
                "vz": native.vz_m_s,
                "wx": native.wx_rad_s,
                "wy": native.wy_rad_s,
                "wz": native.wz_rad_s,
                "frame_id": "odom",
                "child_frame_id": "base_link",
                "twist_frame_id": "base_link",
                "stamp_sec": native_stamp_ns // 1_000_000_000,
                "stamp_nanosec": native_stamp_ns % 1_000_000_000,
            })
            ages = dict(result_snapshot.get("state_age_s") or {})
            ages["odom"] = max(0.0, float(result_snapshot["sample_monotonic_s"]) - native.host_monotonic_s)
            result_snapshot["pose"] = pose
            result_snapshot["state_age_s"] = ages
            result_snapshot["native_mjdata_route_pose"] = {
                "generation": native.generation,
                "sim_time_s": native.sim_time_s,
                "host_monotonic_s": native.host_monotonic_s,
                "source": "READ_ONLY_NATIVE_MJDATA",
            }
            return result_snapshot

        def pose_from_snapshot(snapshot: Mapping[str, Any]) -> PoseEstimate:
            pose = snapshot.get("pose")
            ages = snapshot.get("state_age_s")
            if not isinstance(pose, Mapping) or not isinstance(ages, Mapping):
                raise ValueError("route pose or state-age telemetry is unavailable")
            frame_id = str(pose.get("frame_id", ""))
            child_frame_id = str(pose.get("child_frame_id", ""))
            if frame_id != "odom" or child_frame_id != "base_link":
                raise ValueError(f"route requires odom->base_link, received {frame_id}->{child_frame_id}")
            stamp_ns = int(pose["stamp_sec"]) * 1_000_000_000 + int(pose["stamp_nanosec"])
            age = ages.get("odom")
            if age is None or not math.isfinite(float(age)):
                raise ValueError("route odometry receive age is unavailable")
            sample_steady_s = float(snapshot["sample_monotonic_s"])
            receive_steady_s = sample_steady_s - float(age)
            wall_time_ns = None
            wall_utc = snapshot.get("wall_utc")
            if isinstance(wall_utc, str):
                try:
                    wall_dt = datetime.fromisoformat(wall_utc.replace("Z", "+00:00"))
                    if wall_dt.tzinfo is not None:
                        wall_time_ns = int(wall_dt.timestamp() * 1_000_000_000)
                except (OverflowError, OSError, ValueError):
                    wall_time_ns = None
            return PoseEstimate(
                frame_id="odom",
                child_frame_id="base_link",
                x_m=float(pose["x"]),
                y_m=float(pose["y"]),
                yaw_rad=float(pose["yaw"]),
                stamp_ns=stamp_ns,
                received_steady_s=receive_steady_s,
                generation=str(self.args.attempt_id),
                source="MUJOCO_GROUND_TRUTH",
                position_std_m=0.0,
                yaw_std_rad=0.0,
                transform_chain=("odom->base_link",),
                transform_age_s=float(age),
                wall_time_ns=wall_time_ns,
            )

        class LivePoseProvider:
            def __init__(self) -> None:
                self.pose: PoseEstimate | None = None

            def update(self, snapshot: Mapping[str, Any]) -> PoseEstimate:
                self.pose = pose_from_snapshot(snapshot)
                return self.pose

            def latest_pose(self, target_frame: str, now_steady_s: float) -> PoseEstimate | None:
                return self.pose if target_frame == "odom" else None

        class LiveSafetyProvider:
            def __init__(self) -> None:
                self.received_s = 0.0
                self.segment_id = ""
                self.robot_safe = False
                self.reason = "no runtime observation"

            def update(self, snapshot: Mapping[str, Any], segment_id: str) -> None:
                self.received_s = float(snapshot.get("sample_monotonic_s", time.monotonic()))
                self.segment_id = segment_id
                fault = snapshot_fault(
                    snapshot,
                    self.joint_limits,
                    state_timeout_s=self.state_timeout_s,
                    min_root_z=self.min_root_z,
                    max_tilt_deg=self.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                self.robot_safe = fault is None
                self.reason = fault or (
                    "pinned Local01 RAICOM room/table/chair/mug static route corridor passed; no live obstacle sensor"
                    if short_scene_route_mode or table_approach_mission_mode else
                    "pinned open-floor scene has no obstacle geoms inside the local corridor"
                )

            def latest_observation(self, segment_id: str, frame_id: str, now_steady_s: float) -> SafetyObservation | None:
                if frame_id != "odom" or segment_id != self.segment_id:
                    return None
                return SafetyObservation(
                    segment_id=segment_id,
                    frame_id=frame_id,
                    received_steady_s=self.received_s,
                    clear=True,
                    robot_safe=self.robot_safe,
                    minimum_obstacle_clearance_m=float(
                        workcell["minimum_swept_aabb_clearance_lower_bound_m"]
                        if short_scene_route_mode or table_approach_mission_mode
                        else workcell["verified_local_clear_radius_m"]
                    ),
                    reason=self.reason,
                )

        class RouteMotionAdapter:
            def __init__(self) -> None:
                self.latest = None
                self.events: list[dict[str, Any]] = []
                self.zero_requested_at_s: float | None = None

            def dispatch(self, directive) -> None:
                self.latest = directive
                self.events.append({
                    "kind": directive.kind.value,
                    "segment_id": directive.segment_id,
                    "reason": directive.reason,
                    "steady_s": time.monotonic(),
                })

            def request_zero_and_stop(self, reason: str, requested_steady_s: float) -> float:
                actual_request = max(float(requested_steady_s), time.monotonic())
                self.zero_requested_at_s = actual_request
                self.events.append({
                    "kind": "M1_ZERO_STOP_REQUEST",
                    "reason": reason,
                    "steady_s": actual_request,
                })
                return actual_request

        try:
            initial = route_feedback_snapshot(self.sample_for_first_motion())
            initial_fault = snapshot_fault(
                initial,
                self.joint_limits,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=0.2,
            )
            if initial_fault or not stand_mode_readback_confirmed(initial):
                raise ValueError(f"route requires fresh stable STAND preflight: {initial_fault or 'STAND_READBACK_MISSING'}")
            start_pose = pose_from_snapshot(initial)
            if native_route_reader is not None:
                ros_pose = initial.get("ros_odom_pose")
                if not isinstance(ros_pose, Mapping):
                    raise ValueError("native route feedback requires a fresh ROS odom cross-check at startup")
                try:
                    position_delta_m = math.hypot(
                        float(start_pose.x_m) - float(ros_pose["x"]),
                        float(start_pose.y_m) - float(ros_pose["y"]),
                    )
                    yaw_delta_rad = math.atan2(
                        math.sin(float(start_pose.yaw_rad) - float(ros_pose["yaw"])),
                        math.cos(float(start_pose.yaw_rad) - float(ros_pose["yaw"])),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("native route feedback ROS cross-check pose is malformed") from exc
                cross_check = {
                    "status": "PASS" if position_delta_m <= 0.025 and abs(yaw_delta_rad) <= 0.075 else "FAIL",
                    "position_delta_m": position_delta_m,
                    "yaw_delta_rad": yaw_delta_rad,
                    "maximum_position_delta_m": 0.025,
                    "maximum_yaw_delta_rad": 0.075,
                    "ros_pose": dict(ros_pose),
                    "native_pose": {
                        "x_m": start_pose.x_m,
                        "y_m": start_pose.y_m,
                        "z_m": float(initial["pose"]["z"]),
                        "yaw_rad": start_pose.yaw_rad,
                        "generation": initial.get("native_mjdata_route_pose", {}).get("generation"),
                        "sim_time_s": initial.get("native_mjdata_route_pose", {}).get("sim_time_s"),
                    },
                }
                result["native_mjdata_route_feedback"]["startup_cross_check"] = cross_check
                if cross_check["status"] != "PASS":
                    raise ValueError("native route feedback disagrees with fresh ROS odom at startup")
            if short_scene_route_mode:
                if not isinstance(local01_route_input, Mapping):
                    raise ValueError("Local01 short route input was not retained after preflight")
                runtime_contract = local01_route_input.get("runtime_coordinate_contract")
                if not isinstance(runtime_contract, Mapping):
                    raise ValueError("Local01 short route lacks the reviewed raw-odom coordinate contract")
                reset_pose_path = evidence_root / "reset_immediate_posture.json"
                if reset_pose_path.is_symlink() or not reset_pose_path.is_file():
                    raise ValueError("immediate post-Reset odometry evidence is missing")
                reset_record = json.loads(reset_pose_path.read_text(encoding="utf-8"))
                if (
                    reset_record.get("status") != "POST_RESET_UPRIGHT_OBSERVED"
                    or reset_record.get("fresh_all_required_topics") is not True
                    or float(reset_record.get("state_ages_s", {}).get("odom", float("inf"))) > 0.5
                ):
                    raise ValueError("immediate post-Reset odometry is not fresh and upright")
                latest_reset_odom = reset_record.get("latest_odom")
                reset_data = latest_reset_odom.get("data") if isinstance(latest_reset_odom, Mapping) else None
                if not isinstance(reset_data, Mapping):
                    raise ValueError("immediate post-Reset odometry sample has no decoded pose")
                reset_observed = {
                    "x_m": float(reset_data["x"]),
                    "y_m": float(reset_data["y"]),
                    "z_m": float(reset_data["z"]),
                    "yaw_rad": float(reset_data["yaw_rad"]),
                }
                reset_frame_gate = validate_local01_reset_odom_pose(
                    local01_route_input["reset_anchor_expected_odom"], reset_observed,
                    maximum_position_error_m=float(runtime_contract["maximum_start_position_deviation_m"]),
                    maximum_yaw_error_rad=float(runtime_contract["maximum_start_yaw_deviation_rad"]),
                )
                result["runtime_raw_odom_reset_gate"] = {
                    "status": (
                        "PASS" if reset_frame_gate["status"] == "PASS" else "BLOCKED"
                    ),
                    "basis": "actual fresh post-Reset /aima/hal/odom/state numeric pose compared to the static RAICOM scene-world START; the live route is then derived from the fresh post-STAND heading",
                    "expected": local01_route_input["reset_anchor_expected_odom"],
                    "immediate_post_reset": reset_frame_gate,
                    "runtime_reset_odom_observed": True,
                    "post_stand_runtime_forward_route_gate_required": True,
                    "nonzero_motion_allowed": reset_frame_gate["status"] == "PASS",
                }
                if result["runtime_raw_odom_reset_gate"]["status"] != "PASS":
                    result.update({
                        "status": "BLOCKED",
                        "fault": "LOCAL01_RUNTIME_RAW_ODOM_RESET_ANCHOR_MISMATCH",
                        "route_manager_motion_calls": 0,
                        "nonzero_publish_calls_returned": 0,
                    })
                    return result
            if table_approach_mission_mode:
                if not isinstance(local01_route_input, Mapping):
                    raise ValueError("table-approach mission input was not retained after preflight")
                runtime_contract = local01_route_input.get("runtime_coordinate_contract")
                if not isinstance(runtime_contract, Mapping):
                    raise ValueError("table-approach mission lacks the reviewed raw-odom coordinate contract")
                reset_pose_path = evidence_root / "reset_immediate_posture.json"
                if reset_pose_path.is_symlink() or not reset_pose_path.is_file():
                    raise ValueError("table-approach mission lacks immediate post-Reset odometry evidence")
                reset_record = json.loads(reset_pose_path.read_text(encoding="utf-8"))
                latest_reset_odom = reset_record.get("latest_odom")
                reset_data = latest_reset_odom.get("data") if isinstance(latest_reset_odom, Mapping) else None
                if (
                    reset_record.get("status") != "POST_RESET_UPRIGHT_OBSERVED"
                    or reset_record.get("fresh_all_required_topics") is not True
                    or float(reset_record.get("state_ages_s", {}).get("odom", float("inf"))) > 0.5
                    or not isinstance(reset_data, Mapping)
                ):
                    raise ValueError("table-approach immediate post-Reset odometry is not fresh and upright")
                reset_observed = {
                    "x_m": float(reset_data["x"]), "y_m": float(reset_data["y"]),
                    "z_m": float(reset_data["z"]), "yaw_rad": float(reset_data["yaw_rad"]),
                }
                reset_frame_gate = validate_local01_reset_odom_pose(
                    runtime_contract["static_scene_start"], reset_observed,
                    maximum_position_error_m=float(runtime_contract["maximum_start_position_deviation_m"]),
                    maximum_yaw_error_rad=float(runtime_contract["maximum_start_yaw_deviation_rad"]),
                )
                result["runtime_raw_odom_reset_gate"] = {
                    "status": "PASS" if reset_frame_gate["status"] == "PASS" else "BLOCKED",
                    "basis": (
                        "fresh post-Reset odometry against the derived yaw-zero scene START before any locomotion command"
                        if aligned_start_table_approach_mode else
                        "fresh post-Reset odometry against the source scene-world START before zero-forward alignment"
                    ),
                    "expected": runtime_contract["static_scene_start"],
                    "immediate_post_reset": reset_frame_gate,
                    "runtime_reset_odom_observed": True,
                    "nonzero_motion_allowed": reset_frame_gate["status"] == "PASS",
                }
                if reset_frame_gate["status"] != "PASS":
                    result.update({"status": "BLOCKED", "fault": "TABLE_APPROACH_RESET_ANCHOR_MISMATCH",
                                   "route_manager_motion_calls": 0, "nonzero_publish_calls_returned": 0})
                    return result
                if aligned_start_table_approach_mode:
                    post_sd_pose = {
                        "x_m": start_pose.x_m,
                        "y_m": start_pose.y_m,
                        "z_m": float(initial["pose"]["z"]),
                        "yaw_rad": start_pose.yaw_rad,
                    }
                    derived_start_gate = validate_runtime_raw_odom_aligned_start_table_approach_start(
                        local01_route_input, post_sd_pose
                    )
                    result["derived_aligned_start_post_sd_gate"] = derived_start_gate
                    result["initial_yaw_alignment"] = {
                        "status": "NOT_APPLICABLE_DERIVED_INITIAL_YAW_ZERO_BEFORE_PHYSICS",
                        "nonzero_commands_sent_for_alignment": 0,
                        "post_sd_start_gate": derived_start_gate,
                        "original_yaw_only_turn": "NOT_RUN_AND_NOT_QUALIFIED",
                    }
                    if derived_start_gate["status"] != "PASS":
                        result.update({
                            "status": "BLOCKED",
                            "fault": "ALIGNED_START_POST_SD_POSE_OUTSIDE_STATIC_ENVELOPE",
                            "route_manager_motion_calls": 0,
                            "nonzero_publish_calls_returned": 0,
                        })
                        return result
                else:
                    alignment = self.run_zero_forward_table_alignment(
                        registration=registration, set_action=set_action,
                        stationary_baseline=stationary_baseline, mission=local01_route_input,
                    )
                    result["initial_yaw_alignment"] = alignment
                    if alignment.get("status") != "PASS":
                        result.update({"status": "BLOCKED", "fault": alignment.get("fault") or "TABLE_START_ALIGNMENT_NOT_QUALIFIED"})
                        return result
                    # Alignment ends in verified STAND.  Reacquire LOCOMOTION via
                    # the existing source-grounded path before its first transit.
                    lifecycle, next_set_action, activation_gate = self.begin_source_grounded_route_activation(
                        registration=registration, phase="table-approach-first-transit-rearm",
                    )
                    result["post_alignment_source_grounded_activation"] = {
                        "set_mc_action_ld": next_set_action,
                        "activation_gate": activation_gate,
                        "lifecycle_after_activation": lifecycle.snapshot(),
                    }
                    if activation_gate.get("status") != "READY":
                        recovery = self.recover_activation_to_stand(
                            lifecycle=lifecycle, registration=registration, set_action=next_set_action,
                            baseline=stationary_baseline,
                            reason=str(activation_gate.get("fault") or activation_gate.get("terminal_reason") or "post-alignment route activation was not ready"),
                            phase="table-approach-first-transit-recovery",
                        )
                        result["post_alignment_source_grounded_activation"]["stand_recovery"] = recovery
                        result.update({"status": "BLOCKED", "fault": "TABLE_APPROACH_FIRST_TRANSIT_REARM_NOT_CONFIRMED"})
                        return result
                    self.activation_lifecycle = lifecycle
                    set_action = next_set_action
                    initial = route_feedback_snapshot(self.sample_for_first_motion())
                    initial_fault = snapshot_fault(
                        initial, self.joint_limits, state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z, max_tilt_deg=self.args.max_tilt_deg,
                        max_observed_planar_speed_m_s=0.2,
                    )
                    if initial_fault or not stand_mode_readback_confirmed(initial):
                        raise ValueError("post-alignment route start is not fresh stable STAND")
                    start_pose = pose_from_snapshot(initial)
            footprint = float(workcell["robot_collision_footprint_radial_upper_bound_m"])
            clear_radius = float(workcell["verified_local_clear_radius_m"])
            curved_reference = (
                build_shallow_curved_path_reference(start_pose)
                if curved_path_diagnostic else None
            )
            if short_scene_route_mode:
                if local01_route_input is None:
                    raise ValueError("Local01 short route config is unavailable")
                live_start = {
                    "x_m": start_pose.x_m,
                    "y_m": start_pose.y_m,
                    "z_m": float(initial["pose"]["z"]),
                    "yaw_rad": start_pose.yaw_rad,
                }
                route, runtime_forward_plan = build_runtime_raw_odom_short_scene_route(
                    local01_route_input,
                    observed_start_pose=live_start,
                    runtime_robot_footprint_radius_m=footprint,
                    verified_local_clear_radius_m=clear_radius,
                    forward_m_s=float(self.args.forward_speed),
                )
                result["runtime_raw_odom_forward_plan"] = runtime_forward_plan
                result["route_config_preflight"] = {
                    "status": "PASS_STATIC_SCENE_CLEARANCE",
                    "classification": "STATIC_MUJOCO_FORWARD_ONLY_NOT_TABLE_DOCKING_VALIDATION",
                    "static_clearance_lower_bound_m": workcell["minimum_swept_aabb_clearance_lower_bound_m"],
                    "static_samples": workcell["static_sample_count"],
                    "mj_step_calls": workcell["mj_step_calls"],
                    "full_table_docking_reference": local01_route_input["safe_table_docking_reference"],
                }
            elif table_approach_mission_mode:
                if local01_route_input is None:
                    raise ValueError("table-approach mission config is unavailable")
                route = build_table_approach_route_from_mission(
                    local01_route_input,
                    runtime_robot_footprint_radius_m=footprint,
                    verified_local_clear_radius_m=clear_radius,
                    segment_timeout_s=TABLE_APPROACH_SEGMENT_TIMEOUT_S,
                    rolling_intermediate_waypoint=continuous_table_approach_mode,
                )
                if (
                    continuous_ten_second_transit_mode
                    or native_nominal_straight_gait_diagnostic_mode
                ):
                    route = replace(
                        route,
                        steps=tuple(
                            replace(
                                step,
                                max_command_window_s=TABLE_APPROACH_CONTINUOUS_WINDOW_S,
                            )
                            for step in route.steps
                        ),
                    )
                    route.validate()
                if bool(getattr(self.args, "raicom_low_cruise_profile", False)):
                    route = replace(
                        route,
                        steps=tuple(
                            replace(
                                step,
                                target_forward_m_s=RAICOM_LOW_CRUISE_FORWARD_M_S,
                                max_forward_m_s=RAICOM_LOW_CRUISE_FORWARD_M_S,
                            )
                            for step in route.steps
                        ),
                    )
                    route.validate()
                result["route_config_preflight"] = {
                    "status": "PASS_STATIC_SAFE_PREFIX_ONLY",
                    "classification": (
                        "SIMULATION_ONLY_ALIGNED_START_LOW_CRUISE_TABLE_APPROACH_RECOVERY"
                        if bool(getattr(self.args, "raicom_low_cruise_profile", False))
                        else "TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION"
                    ),
                    "static_clearance_lower_bound_m": workcell["minimum_swept_aabb_clearance_lower_bound_m"],
                    "initial_alignment": result.get("initial_yaw_alignment"),
                    "table_docking_reference": local01_route_input["table_docking_reference"],
                }
            elif route_config_mode:
                route = build_waypoint_docking_route_from_poses(
                    load_waypoint_docking_pose_input(self.args.route_config),
                    runtime_robot_footprint_radius_m=footprint,
                    verified_local_clear_radius_m=clear_radius,
                )
                result["route_config_preflight"] = validate_waypoint_docking_scene_clearance(
                    route,
                    runtime_robot_footprint_radius_m=footprint,
                    verified_local_clear_radius_m=clear_radius,
                )
            elif curved_reference is not None:
                route = build_curved_waypoint_route(
                    route_id=f"issue80-curved-route-{self.args.attempt_id}",
                    start_pose=start_pose,
                    reference=curved_reference,
                    robot_footprint_radius_m=footprint,
                    local_clear_radius_m=clear_radius,
                )
            else:
                route = build_short_right_corner_route(
                    route_id=f"issue80-short-route-{self.args.attempt_id}",
                    start_pose=start_pose,
                    robot_footprint_radius_m=footprint,
                    local_clear_radius_m=clear_radius,
                    right_turn_qualified=turn_provenance is not None,
                    waypoint_b_only=waypoint_b_only,
                    waypoint_b_distance_m=(
                        WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M
                        if waypoint_b_short_diagnostic else 0.40 if waypoint_b_only else 0.35
                    ),
                )
            transit_x, transit_y = route.steps[0].x_m, route.steps[0].y_m
            if not waypoint_b_only and not route_config_mode:
                next_x, next_y = route.steps[2].x_m, route.steps[2].y_m
                corner_yaw = math.atan2(next_y - transit_y, next_x - transit_x)
            corridor_half_width = (
                footprint + 0.10 + 0.05
                + float(curved_reference["radius_m"]) * (1.0 - math.cos(float(curved_reference["sweep_rad"])))
                if curved_path_diagnostic else footprint + 0.10 + 0.05
            )
            ideal_turn_lateral_m = (TURN_FORWARD_COMMAND_M_S / TURN_COMMAND_RAD_S) * (
                1.0 - math.cos(TURN_COMMAND_RAD_S * TURN_PULSE_S)
            )
            turn_corridor_half_width = footprint + 0.10 + ideal_turn_lateral_m + 0.05
            pose_provider = LivePoseProvider()
            safety_provider = LiveSafetyProvider()
            safety_provider.joint_limits = self.joint_limits
            safety_provider.state_timeout_s = self.args.state_timeout
            safety_provider.min_root_z = self.args.min_root_z
            safety_provider.max_tilt_deg = self.args.max_tilt_deg
            pose_provider.update(initial)
            safety_provider.update(initial, route.steps[0].waypoint_id)
            route_motion = RouteMotionAdapter()
            executor = WaypointRouteExecutor(route, pose_provider, safety_provider, route_motion)
            route_feedback = MotionFeedback(
                status=MotionStatus.ACTIVE,
                reset_gate_pass=True,
                standing_gate_pass=True,
                source_mode_handshake_pass=False,
                safety_gates_pass=True,
                initial_activation_authorized=True,
            )
            current_directive = executor.start(float(initial["sample_monotonic_s"]), route_feedback)
            if current_directive.kind is not DirectiveKind.FOLLOW_WAYPOINT:
                raise ValueError(f"route manager refused initial transit: {current_directive.reason}")

            scope = self.scope_evidence or {}
            local_identity = scope.get("publisher_local_identity", {})
            publishers = scope.get("velocity_publishers", [])
            subscribers = scope.get("velocity_subscribers", [])
            expected_domain = int(os.environ.get("ROS_DOMAIN_ID", "-1"))
            turn_evidence_id = str(turn_provenance["sha256"]) if turn_provenance else ""

            def same_sample_handshake(snapshot: Mapping[str, Any]) -> bool:
                times = snapshot.get("readback_monotonic_s")
                sample_time = float(snapshot.get("sample_monotonic_s", -1.0))
                if not isinstance(times, Mapping):
                    return False
                try:
                    mode_time = float(times["get_mode"])
                    source_time = float(times["get_source"])
                except (KeyError, TypeError, ValueError):
                    return False
                return (
                    0.0 <= sample_time - mode_time <= float(self.args.state_timeout)
                    and 0.0 <= sample_time - source_time <= float(self.args.state_timeout)
                    and abs(mode_time - source_time) <= MODE_SOURCE_READBACK_SKEW_MAX_S
                    and snapshot.get("get_mode_header_code") == 0
                    and snapshot.get("get_mode_status_value") == 100
                    and snapshot.get("get_mode_desc") == MODE_LOCOMOTION
                    and snapshot.get("common_mode_desc") == MODE_LOCOMOTION
                    and snapshot.get("common_mode_status_value") == 100
                    and snapshot.get("get_source_header_code") == 0
                    and snapshot.get("get_source_name") == self.source
                    and snapshot.get("common_source_name") == self.source
                )

            def readiness_from_snapshot(
                snapshot: Mapping[str, Any], *, measured_turn_guard: bool = False
            ) -> RuntimeReadiness:
                fault = snapshot_fault(
                    snapshot,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                return RuntimeReadiness(
                    observed_steady_s=float(snapshot.get("sample_monotonic_s", time.monotonic())),
                    reset_gate_pass=True,
                    standing_gate_pass=True,
                    safety_gates_pass=fault is None,
                    source_registration_pass=registration.get("success") is True,
                    action_source_setup_pass=set_action.get("accepted_by_service") is True,
                    source_mode_handshake_pass=same_sample_handshake(snapshot),
                    telemetry_fresh=not bool(stale_readback_keys(snapshot, float(self.args.state_timeout))),
                    workspace_clearance_verified=workcell.get("status") == "PASS",
                    corridor_clear=True,
                    swept_path_clear=True,
                    bounded_moving_turn_qualified=turn_provenance is not None,
                    qualified_turn_directions=("RIGHT",) if turn_provenance else (),
                    turn_stop_margin_m=0.05 if turn_provenance else None,
                    turn_stop_margin_direction="RIGHT" if turn_provenance else "",
                    turn_stop_margin_evidence_id=turn_evidence_id,
                    transport_isolated=(scope.get("status") == "PASS" and scope.get("docker_network_mode") == "bridge"),
                    publisher_identity_verified=(
                        len(publishers) == 1
                        and local_identity.get("is_owned_publisher") is True
                        and local_identity.get("node_topic_publisher_count") == 1
                    ),
                    mc_subscriber_identity_verified=(len(subscribers) == 1 and bool(scope.get("verified_mc_runtime_identity"))),
                    qos_compatible=scope.get("status") == "PASS",
                    velocity_publisher_count=len(publishers),
                    mc_velocity_subscriber_count=len(subscribers),
                    ros_domain_id=int(scope.get("ros_domain_id", -1)),
                    expected_ros_domain_id=expected_domain,
                    mode_desc=str(snapshot.get("common_mode_desc") or snapshot.get("get_mode_desc") or ""),
                    selected_source=str(snapshot.get("common_source_name") or snapshot.get("get_source_name") or ""),
                    measured_displacement_guard_verified=measured_turn_guard,
                )

            bridge = RouteMotionBridge(
                route_frame_id="odom",
                incremental_window_s=(
                    TABLE_APPROACH_CONTINUOUS_WINDOW_S
                    if (
                        continuous_ten_second_transit_mode
                        or native_nominal_straight_gait_diagnostic_mode
                    ) else 5.0
                ),
            )
            initial_readiness = readiness_from_snapshot(initial)
            transit_plan = bridge.plan_next(
                current_directive, start_pose, initial_readiness,
                now_steady_s=float(initial["sample_monotonic_s"]),
            )
            if transit_plan.status != "PROPOSAL_ONLY" or transit_plan.forward_m_s <= 0.0:
                raise ValueError(f"route bridge refused transit proposal: {transit_plan.status}: {transit_plan.reason}")
            # Record the actual bridge directive, rather than the legacy generic
            # route default. This must match the value reaching publish_window.
            result["forward_command_m_s"] = float(transit_plan.forward_m_s)
            route_spec = {
                "route_id": route.route_id,
                "frame_id": route.frame_id,
                "start_anchor": {"x_m": route.start_x_m, "y_m": route.start_y_m, "yaw_rad": route.start_yaw_rad},
                "steps": [
                    {"waypoint_id": step.waypoint_id, "profile": step.profile.value, "x_m": step.x_m,
                     "y_m": step.y_m, "position_tolerance_m": step.position_tolerance_m,
                     "yaw_tolerance_rad": step.yaw_tolerance_rad,
                     "segment_timeout_s": step.segment_timeout_s,
                     "maximum_turn_displacement_m": step.maximum_turn_displacement_m}
                    for step in route.steps
                ],
                "corridor_half_width_m": corridor_half_width,
                "corridor_half_width_m_by_step": {
                    step.waypoint_id: min(
                        distance_to_polygon_boundary(
                            step.x_m,
                            step.y_m,
                            step.corridor_polygon,
                        ),
                        distance_to_polygon_boundary(
                            route.start_x_m,
                            route.start_y_m,
                            step.corridor_polygon,
                        ),
                    )
                    for step in route.steps
                },
                "corridor_longitudinal_limits_m": [-0.75, 1.25],
                "maximum_centerline_lateral_error_m": (
                    CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M
                    if segmented_table_envelope_mode else WAYPOINT_B_MAX_CROSS_TRACK_M
                ),
                "robot_footprint_radial_upper_bound_m": footprint,
                "boundary_margin_m": route.corridor_boundary_margin_m,
                "pose_uncertainty_note": "installed simulator ground-truth odometry publishes no covariance; route uses zero covariance only as the deterministic ground-truth contract, not as an estimated localization confidence",
            }
            if table_approach_mission_mode:
                limits = local01_route_input["limits"] if isinstance(local01_route_input, Mapping) else {}
                budget = local01_route_input["bounded_runtime_budget"] if isinstance(local01_route_input, Mapping) else {}
                route_spec.update({
                    "configured_route_input_sha256": hashlib.sha256(
                        Path(self.args.route_config).read_bytes()
                    ).hexdigest(),
                    "route_motion_profile": (
                        "SIMULATION_ONLY_ALIGNED_START_NATIVE_X2_TIMESTEP_LOW_CRUISE_TABLE_APPROACH_RECOVERY"
                        if native_x2_timestep_mode else
                        "SIMULATION_ONLY_ALIGNED_START_LOW_CRUISE_TABLE_APPROACH_RECOVERY"
                        if bool(getattr(self.args, "raicom_low_cruise_profile", False)) else
                        "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION"
                        if aligned_start_table_approach_mode else
                        "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION"
                    ),
                    "maximum_windows_total": (
                        1 if native_nominal_straight_gait_diagnostic_mode
                        else 1 if continuous_table_approach_mode
                        else TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS
                        if continuous_ten_second_transit_mode
                        else int(limits["max_windows"])
                    ),
                    "maximum_window_s": (
                        TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                        if continuous_table_approach_mode else
                        TABLE_APPROACH_CONTINUOUS_WINDOW_S
                        if (
                            continuous_ten_second_transit_mode
                            or native_nominal_straight_gait_diagnostic_mode
                        )
                        else float(limits["max_window_s"])
                    ),
                    "aggregate_nonzero_limit_s": float(limits["max_aggregate_nonzero_s"]),
                    "maximum_forward_m_s": float(route.steps[0].max_forward_m_s),
                    "maximum_angular_rad_s": float(limits["angular_rad_s"]),
                    "maximum_segment_cross_track_m": (
                        {
                            "WAYPOINT": CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
                            "TABLE_APPROACH": CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
                        }
                        if segmented_table_envelope_mode else float(limits["max_cross_track_m"])
                    ),
                    "segment_timeout_s": TABLE_APPROACH_SEGMENT_TIMEOUT_S,
                    "segment_timeout_basis": (
                        "the unchanged 75 s aggregate nonzero allowance plus complete zero/stand/"
                        "physical-stop observation budgets; below the 410 s child-runtime watchdog"
                    ),
                    "post_stand_observation_s": 7.0,
                    "bounded_runtime_budget": budget,
                    "final_state": "COMPLETE_AT_TABLE_APPROACH_WITHOUT_MANIPULATOR_HANDOFF",
                })
            elif short_scene_route_mode:
                route_spec.update({
                    "configured_route_input_sha256": hashlib.sha256(
                        Path(self.args.route_config).read_bytes()
                    ).hexdigest(),
                    "route_motion_profile": (
                        "SIMULATION_ONLY_RAICOM_LOW_CRUISE_RECOVERY"
                        if bool(getattr(self.args, "raicom_low_cruise_profile", False))
                        else "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_NOMINAL"
                    ),
                    "maximum_windows_total": SHORT_ROUTE_MAX_WINDOWS,
                    "maximum_window_s": SHORT_ROUTE_MAX_WINDOW_S,
                    "aggregate_nonzero_limit_s": SHORT_ROUTE_MAX_AGGREGATE_S,
                    "maximum_forward_m_s": float(transit_plan.forward_m_s),
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    "maximum_segment_cross_track_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                    "post_stand_observation_s": 7.0,
                    "final_state": "SHORT_ROUTE_STOPPED_WITHOUT_TABLE_OR_MANIPULATOR_HANDOFF",
                })
            elif route_config_mode:
                route_spec.update({
                    "configured_route_input_sha256": hashlib.sha256(
                        Path(self.args.route_config).read_bytes()
                    ).hexdigest(),
                    "route_motion_profile": "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE",
                    "maximum_windows_total": 3,
                    "maximum_window_s": 5.0,
                    "aggregate_nonzero_limit_s": 15.0,
                    "maximum_forward_m_s": 0.12,
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    "maximum_segment_cross_track_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                    "post_stand_observation_s": 7.0,
                    "final_state": "WAITING_FOR_DOCKING_WITHOUT_MANIPULATOR_HANDOFF",
                })
            elif curved_path_diagnostic:
                curve_forward_extent = float(curved_reference["radius_m"]) * math.sin(
                    float(curved_reference["sweep_rad"])
                )
                curved_corridor_forward_limit = max(
                    1.25, curve_forward_extent + footprint + 0.10 + 0.05,
                )
                route_spec.update({
                    "curved_path_reference": curved_reference,
                    "corridor_longitudinal_limits_m": [-0.75, curved_corridor_forward_limit],
                    "corridor_half_width_m": corridor_half_width,
                    "curved_path_expected_length_m": curved_reference["expected_path_length_m"],
                    "curved_path_target_xy_m": [curved_reference["target_x_m"], curved_reference["target_y_m"]],
                    "curved_path_target_yaw": None,
                    "position_tolerance_m": CURVED_PATH_POSITION_TOLERANCE_M,
                    "maximum_cross_track_error_m": CURVED_PATH_MAX_CROSS_TRACK_M,
                    "post_stop_reserve_m": CURVED_PATH_STOP_RESERVE_M,
                    "arrival_trigger_radius_m": waypoint_b_arrival_trigger_radius_m(
                        acceptance_tolerance_m=CURVED_PATH_POSITION_TOLERANCE_M,
                        stop_reserve_m=CURVED_PATH_STOP_RESERVE_M,
                    ),
                    "maximum_transit_windows": CURVED_PATH_MAX_WINDOWS,
                    "maximum_transit_window_s": CURVED_PATH_MAX_WINDOW_S,
                    "aggregate_nonzero_budget_s": CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
                    "maximum_forward_m_s": CURVED_PATH_FORWARD_M_S,
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                })
            elif waypoint_b_only:
                route_spec["waypoint_b_target_distance_from_start_m"] = math.hypot(
                    route.steps[0].x_m - start_pose.x_m,
                    route.steps[0].y_m - start_pose.y_m,
                )
                route_spec["waypoint_b_position_tolerance_m"] = WAYPOINT_B_POSITION_TOLERANCE_M
                route_spec["waypoint_b_diagnostic_only"] = (
                    waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic
                )
                route_spec["waypoint_b_profile"] = (
                    "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC"
                    if waypoint_b_extended_diagnostic else
                    "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC"
                    if waypoint_b_short_diagnostic else "LEGACY_TWO_WINDOW_WAYPOINT_B"
                )
                route_spec["waypoint_b_stop_reserve_m"] = (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                    if waypoint_b_extended_diagnostic else
                    WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                    if waypoint_b_short_diagnostic else 0.0
                )
                route_spec["waypoint_b_arrival_trigger_radius_m"] = waypoint_b_arrival_trigger_radius_m(
                    acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
                    stop_reserve_m=(
                        WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_extended_diagnostic else
                        WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_short_diagnostic else 0.0
                    ),
                )
                route_spec["maximum_transit_windows"] = (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS
                    if waypoint_b_extended_diagnostic else WAYPOINT_B_MAX_WINDOWS
                )
                route_spec["maximum_transit_window_s"] = WAYPOINT_B_WINDOW_MAX_S
                route_spec["aggregate_transit_budget_s"] = (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S
                    if waypoint_b_extended_diagnostic else WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S
                )
                route_spec["maximum_forward_projection_guard_m"] = (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M
                    if waypoint_b_extended_diagnostic else None
                )
                route_spec["route_corridor_forward_limit_m"] = WAYPOINT_B_CORRIDOR_MAX_FORWARD_M
                route_spec["maximum_centerline_lateral_error_m"] = WAYPOINT_B_MAX_CROSS_TRACK_M
                route_spec["waypoint_b_target_selection"] = (
                    "0.20 m diagnostic target: the latest measured settled position was "
                    "0.095 m from this target with 0.15 m tolerance; this does not pass "
                    "the original 0.40 m waypoint-B acceptance"
                    if waypoint_b_short_diagnostic else
                    "original 0.40 m B target with 0.15 m tolerance and a 45 mm measured post-stop travel reserve"
                    if waypoint_b_extended_diagnostic else
                    "0.40 m acceptance target based on prior measured travel; 0.15 m arrival tolerance retained"
                )
            else:
                route_spec.update({
                    "turn_corridor_half_width_m": turn_corridor_half_width,
                    "expected_ideal_turn_lateral_m": ideal_turn_lateral_m,
                    "turn_corridor_longitudinal_limits_m": [-0.75, 1.25],
                    "waypoint_b_c_center_distance_m": math.hypot(
                        route.steps[2].x_m - route.steps[0].x_m,
                        route.steps[2].y_m - route.steps[0].y_m,
                    ),
                    "waypoint_b_c_region_gap_m": (
                        math.hypot(
                            route.steps[2].x_m - route.steps[0].x_m,
                            route.steps[2].y_m - route.steps[0].y_m,
                        )
                        - route.steps[0].position_tolerance_m
                        - route.steps[2].position_tolerance_m
                    ),
                    "minimum_waypoint_c_distance_after_max_turn_displacement_m": (
                        math.hypot(
                            route.steps[2].x_m - route.steps[0].x_m,
                            route.steps[2].y_m - route.steps[0].y_m,
                        )
                        - float(route.steps[1].maximum_turn_displacement_m)
                    ),
                })
            if curved_path_diagnostic:
                curved_preflight = {
                    "schema": "robotsim.issue80.curved_path_preflight.v1",
                    "status": "PASS",
                    "attempt_id": str(self.args.attempt_id),
                    "captured_utc": datetime.now(timezone.utc).isoformat(),
                    "classification": "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC",
                    "start_pose": {
                        "frame_id": start_pose.frame_id,
                        "child_frame_id": start_pose.child_frame_id,
                        "x_m": start_pose.x_m,
                        "y_m": start_pose.y_m,
                        "yaw_rad": start_pose.yaw_rad,
                        "odom_stamp_ns": start_pose.stamp_ns,
                        "receive_steady_s": start_pose.received_steady_s,
                        "generation": start_pose.generation,
                    },
                    "curve_reference": curved_reference,
                    "target_xy_m": [route.steps[0].x_m, route.steps[0].y_m],
                    "target_yaw_rad": None,
                    "expected_path_length_m": curved_reference["expected_path_length_m"],
                    "endpoint_tolerance_m": CURVED_PATH_POSITION_TOLERANCE_M,
                    "maximum_cross_track_m": CURVED_PATH_MAX_CROSS_TRACK_M,
                    "speed_and_window_bounds": {
                        "forward_m_s": CURVED_PATH_FORWARD_M_S,
                        "angular_magnitude_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                        "nominal_curvature_angular_rad_s": curved_reference["nominal_angular_rate_rad_s"],
                        "per_window_nonzero_max_s": CURVED_PATH_MAX_WINDOW_S,
                        "maximum_windows": CURVED_PATH_MAX_WINDOWS,
                        "aggregate_nonzero_max_s": CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
                        "zero_deadline_reserve_s": ROUTE_WINDOW_ZERO_MARGIN_S,
                    },
                    "verified_workcell": {
                        "source_scene_hashes": {
                            "scene_xml_sha256": workcell.get("scene_sha256"),
                            "x2_xml_sha256": workcell.get("x2_model_sha256"),
                        },
                        "local_clear_radius_m": clear_radius,
                        "robot_footprint_radius_m": footprint,
                    },
                    "research_source_sha256": {
                        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "issue80_route_motion_bridge.py": hashlib.sha256(
                            Path(__file__).with_name("issue80_route_motion_bridge.py").read_bytes()
                        ).hexdigest(),
                        "issue80_waypoint_route.py": hashlib.sha256(
                            Path(__file__).with_name("issue80_waypoint_route.py").read_bytes()
                        ).hexdigest(),
                    },
                }
                preflight_path = evidence_root / "curved_path_preflight.json"
                with preflight_path.open("x", encoding="utf-8") as stream:
                    json.dump(curved_preflight, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                result["curved_path_preflight"] = curved_preflight
            result["route_spec"] = route_spec
            result["route_motion_bridge"] = {
                "transit_window": dict(transit_plan.__dict__),
                "runtime_controller": {
                    "policy": (
                        "fresh odom circular-arc tangent/curvature feedback at every M1 publish"
                        if curved_path_diagnostic else
                        "fresh odom waypoint-bearing feedback at every M1 publish"
                    ),
                    "heading_gain": 0.8,
                    "maximum_forward_m_s": (
                        CURVED_PATH_FORWARD_M_S if curved_path_diagnostic
                        else ROUTE_TRANSIT_MAX_FORWARD_M_S
                    ),
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    "straight_warmup_s": ROUTE_STEERING_WARMUP_S,
                    "path_reference": curved_reference if curved_path_diagnostic else None,
                    "yaw_rate_feedback_source": (
                        "fresh MuJoCo nav_msgs/Odometry child-frame twist.angular.z"
                        if curved_path_diagnostic else None
                    ),
                    "steering_requires_same_sample_locomotion_and_source_readback": True,
                    "after_startup_handshake": (
                        "each GetMcAction/GetCurrentInputSource/CommonState value must agree and each readback "
                        "must remain individually fresh within the existing state timeout; asynchronous service "
                        "timestamp skew is recorded but does not independently abort steering"
                    ),
                    "plan_window_angular_is_not_replayed": True,
                },
                "turn_margin_source": turn_provenance,
                "turn_displacement_guard": None if waypoint_b_only else {
                    "maximum_m": 0.25,
                    "stop_reserve_m": 0.05,
                    "active_trigger_m": 0.20,
                    "runtime_guard": "per-sample helper cutoff from measured turn-origin pose",
                },
            }

            route_fault: str | None = None
            route_stop_reason: str | None = None
            route_events: list[dict[str, Any]] = []
            route_arrival_poses: dict[str, dict[str, float]] = {}
            route_progress_monitor: dict[str, Any] = {
                "cross_track_limit_m": (
                    CURVED_PATH_MAX_CROSS_TRACK_M if curved_path_diagnostic else
                    CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M if segmented_table_envelope_mode else
                    WAYPOINT_B_MAX_CROSS_TRACK_M if route_config_mode else
                    WAYPOINT_B_MAX_CROSS_TRACK_M if waypoint_b_only else None
                ),
                "cross_track_reference": (
                    "CIRCULAR_ARC" if curved_path_diagnostic else
                    "CURRENT_CONFIGURED_ROUTE_SEGMENT" if route_config_mode else "START_CENTERLINE"
                ),
                "maximum_forward_projection_guard_m": (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M
                    if waypoint_b_extended_diagnostic else None
                ),
                "corridor_forward_limit_m": WAYPOINT_B_CORRIDOR_MAX_FORWARD_M,
                "maximum_abs_cross_track_m": 0.0,
                "maximum_abs_yaw_delta_rad": 0.0,
                "duplicate_odom_stamp_sample_count": 0,
                "first_duplicate_odom_stamp_sample": None,
                "last_duplicate_odom_stamp_sample": None,
                "duplicate_odom_hold_max_duration_s": 0.0,
                "duplicate_odom_consecutive_sample_count": 0,
                "duplicate_odom_hold_started_monotonic_s": None,
                "last_unique_odom_stamp_ns": start_pose.stamp_ns,
                "maximum_age_since_last_unique_odom_s": 0.0,
                "odom_stamp_observations": [],
                "last_sample": None,
                "samples": [],
                "guard_triggers": [],
            }
            result["route_progress_monitor"] = route_progress_monitor
            result["pose_clock_diagnostics"] = {
                "policy": "strict_header_order_except_fresh_wall-aligned_MuJoCo_clock_rollback",
                "verified_wall_clock_step_rebases": 0,
            }
            latest_transit_snapshot: Mapping[str, Any] | None = None
            route_segment_origin = {
                "x_m": route.start_x_m,
                "y_m": route.start_y_m,
            }
            odom_freshness_tracker = RouteOdomFreshnessTracker(
                last_unique_stamp_ns=start_pose.stamp_ns,
                last_unique_received_steady_s=start_pose.received_steady_s,
                maximum_age_s=MAX_POSE_AGE_S,
            )
            window_control: dict[str, Any] = {}
            active_window_gate = TransitSteeringGate(warmup_s=ROUTE_STEERING_WARMUP_S)
            configured_docking_control: dict[str, Any] = {
                "step_id": None,
                "phase_controller": None,
                "yaw_rate_monitor": SustainedYawRateWindow() if route_config_mode else None,
                "yaw_rate_feedback": None,
                "last_decision": None,
            }

            def route_sample() -> dict[str, Any]:
                return route_feedback_snapshot(self.sample_for_first_motion())

            def transit_termination(snapshot: Mapping[str, Any]) -> bool:
                nonlocal route_fault, route_stop_reason, latest_transit_snapshot
                latest_transit_snapshot = snapshot
                if window_control.get("feedback_fault"):
                    route_fault = str(window_control["feedback_fault"])
                    return True
                fault = snapshot_fault(
                    snapshot,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                if fault:
                    route_fault = "transit_safety:" + fault
                    return True
                try:
                    pose_provider.update(snapshot)
                except (KeyError, TypeError, ValueError) as exc:
                    route_fault = f"transit_pose:{type(exc).__name__}:{exc}"
                    return True
                active = executor.active_step
                if active is None:
                    return True
                safety_provider.update(snapshot, active.waypoint_id)
                pose = pose_provider.pose
                assert pose is not None
                sample_time = float(snapshot.get("sample_monotonic_s", time.monotonic()))
                pose_receive_age_s = sample_time - pose.received_steady_s
                odom_freshness = odom_freshness_tracker.observe(
                    current_stamp_ns=pose.stamp_ns,
                    sample_steady_s=sample_time,
                    received_steady_s=pose.received_steady_s,
                )
                odom_stamp_action = str(odom_freshness["action"])
                freshness_record = {
                    "sample_monotonic_s": sample_time,
                    "odom_stamp_ns": pose.stamp_ns,
                    **odom_freshness,
                }
                route_progress_monitor["odom_stamp_observations"].append(freshness_record)
                route_progress_monitor["last_unique_odom_stamp_ns"] = odom_freshness.get(
                    "last_unique_stamp_ns"
                )
                unique_age = odom_freshness.get("age_since_last_unique_s")
                if isinstance(unique_age, (int, float)) and math.isfinite(float(unique_age)):
                    route_progress_monitor["maximum_age_since_last_unique_odom_s"] = max(
                        float(route_progress_monitor["maximum_age_since_last_unique_odom_s"]),
                        float(unique_age),
                    )
                relation = str(odom_freshness.get("source_stamp_relation"))
                if relation == "DUPLICATE":
                    route_progress_monitor["duplicate_odom_stamp_sample_count"] += 1
                    route_progress_monitor["duplicate_odom_consecutive_sample_count"] += 1
                    if route_progress_monitor["duplicate_odom_hold_started_monotonic_s"] is None:
                        route_progress_monitor["duplicate_odom_hold_started_monotonic_s"] = sample_time
                    hold_duration = max(0.0, sample_time - float(
                        route_progress_monitor["duplicate_odom_hold_started_monotonic_s"]
                    ))
                    route_progress_monitor["duplicate_odom_hold_max_duration_s"] = max(
                        float(route_progress_monitor["duplicate_odom_hold_max_duration_s"]),
                        hold_duration,
                        float(unique_age) if isinstance(unique_age, (int, float)) else 0.0,
                    )
                    duplicate_record = {
                        **freshness_record,
                        "odom_receive_age_s": pose_receive_age_s,
                        "duplicate_hold_duration_s": hold_duration,
                    }
                    if route_progress_monitor["first_duplicate_odom_stamp_sample"] is None:
                        route_progress_monitor["first_duplicate_odom_stamp_sample"] = duplicate_record
                        route_events.append({
                            "kind": "DUPLICATE_ODOM_SAMPLE_SKIPPED_FOR_ROUTE_TICK",
                            **duplicate_record,
                            "route_executor_core_remains_strictly_increasing": True,
                        })
                    route_progress_monitor["last_duplicate_odom_stamp_sample"] = duplicate_record
                elif relation == "NEW" and odom_stamp_action == "NEW":
                    route_progress_monitor["duplicate_odom_consecutive_sample_count"] = 0
                    route_progress_monitor["duplicate_odom_hold_started_monotonic_s"] = None
                if odom_stamp_action in {"STALE_OR_INVALID", "REGRESSION"}:
                    route_fault = (
                        "transit_pose:ODOM_SOURCE_DUPLICATE_HOLD_EXCEEDED_UNIQUE_SAMPLE_AGE"
                        if relation == "DUPLICATE" and odom_stamp_action == "STALE_OR_INVALID"
                        else f"transit_pose:ODOM_SOURCE_{odom_stamp_action}"
                    )
                    route_events.append({
                        "kind": "ROUTE_ODOMETRY_FRESHNESS_ABORT",
                        "reason": route_fault,
                        **freshness_record,
                        "maximum_unique_sample_age_s": MAX_POSE_AGE_S,
                    })
                    return True
                window_control["odom_stamp_action"] = odom_stamp_action
                window_control["odom_stamp_ns"] = pose.stamp_ns
                window_control["odom_receive_age_s"] = pose_receive_age_s
                window_control["odom_age_since_last_unique_s"] = odom_freshness.get(
                    "age_since_last_unique_s"
                )
                window_control["last_unique_odom_stamp_ns"] = odom_freshness.get(
                    "last_unique_stamp_ns"
                )
                if configured_docking_route_mode and odom_stamp_action == "NEW":
                    yaw_rate_sample = configured_docking_control["yaw_rate_monitor"].update(snapshot)
                    if yaw_rate_sample.get("status") == "MEASURED":
                        configured_docking_control["yaw_rate_feedback"] = {
                            "signed_net_yaw_rate_rad_s": float(yaw_rate_sample["signed_net_yaw_rate_rad_s"]),
                            "sample_monotonic_s": sample_time,
                        }
                    elif yaw_rate_sample.get("status") == "HOST_MONOTONIC_RESET":
                        configured_docking_control["yaw_rate_feedback"] = None
                if configured_docking_route_mode:
                    active_for_axis = executor.active_step
                    if active_for_axis is None:
                        route_fault = "configured_route_active_step_missing"
                        return True
                    axis_dx = active_for_axis.x_m - float(route_segment_origin["x_m"])
                    axis_dy = active_for_axis.y_m - float(route_segment_origin["y_m"])
                    if math.hypot(axis_dx, axis_dy) <= 1e-6:
                        route_fault = f"configured_route_zero_length_segment:{active_for_axis.waypoint_id}"
                        return True
                    segment_yaw = math.atan2(axis_dy, axis_dx)
                    dx = pose.x_m - float(route_segment_origin["x_m"])
                    dy = pose.y_m - float(route_segment_origin["y_m"])
                else:
                    segment_yaw = start_pose.yaw_rad
                    dx = pose.x_m - start_pose.x_m
                    dy = pose.y_m - start_pose.y_m
                forward_projection_m = dx * math.cos(segment_yaw) + dy * math.sin(segment_yaw)
                cross_track_m = -dx * math.sin(segment_yaw) + dy * math.cos(segment_yaw)
                curve_sample = None
                cross_track_guard_m = abs(cross_track_m)
                if curved_path_diagnostic:
                    try:
                        curve_sample = curved_path_metrics(
                            curved_reference, x_m=pose.x_m, y_m=pose.y_m,
                        )
                    except RouteMotionBridgeError as exc:
                        route_fault = f"curved_path_projection:{exc}"
                        return True
                    forward_projection_m = float(curve_sample["path_progress_m"])
                    signed_radial = float(curve_sample["signed_radial_error_m"])
                    cross_track_guard_m = float(curve_sample["cross_track_error_m"])
                    cross_track_m = math.copysign(cross_track_guard_m, signed_radial)
                yaw_delta_rad = math.atan2(
                    math.sin(pose.yaw_rad - start_pose.yaw_rad),
                    math.cos(pose.yaw_rad - start_pose.yaw_rad),
                )
                route_progress_monitor["maximum_abs_cross_track_m"] = max(
                    float(route_progress_monitor["maximum_abs_cross_track_m"]), abs(cross_track_m)
                )
                route_progress_monitor["maximum_abs_yaw_delta_rad"] = max(
                    float(route_progress_monitor["maximum_abs_yaw_delta_rad"]), abs(yaw_delta_rad)
                )
                route_progress_monitor["last_sample"] = {
                    "sample_monotonic_s": sample_time,
                    "odom_stamp_ns": pose.stamp_ns,
                    "odom_stamp_action": odom_stamp_action,
                    "odom_receive_age_s": pose_receive_age_s,
                    "odom_age_since_last_unique_s": odom_freshness.get(
                        "age_since_last_unique_s"
                    ),
                    "last_unique_odom_stamp_ns": odom_freshness.get(
                        "last_unique_stamp_ns"
                    ),
                    "forward_projection_m": forward_projection_m,
                    "cross_track_m": cross_track_m,
                    "cross_track_error_m": cross_track_guard_m,
                    "yaw_delta_rad": yaw_delta_rad,
                    "curve_progress_fraction": (
                        float(curve_sample["path_progress_fraction"]) if curve_sample else None
                    ),
                    "curve_tangent_yaw_rad": (
                        float(curve_sample["path_tangent_yaw_rad"]) if curve_sample else None
                    ),
                }
                route_progress_monitor["samples"].append(
                    dict(route_progress_monitor["last_sample"])
                )
                cross_track_limit = (
                    CURVED_PATH_MAX_CROSS_TRACK_M if curved_path_diagnostic
                    else continuous_table_segment_cross_track_limit_m(active.waypoint_id)
                    if segmented_table_envelope_mode
                    else WAYPOINT_B_MAX_CROSS_TRACK_M if waypoint_b_only or route_config_mode
                    else None
                )
                sample_disposition = (
                    route_feedback_sample_disposition(
                        odom_action=odom_stamp_action,
                        cross_track_error_m=cross_track_guard_m,
                        maximum_cross_track_m=cross_track_limit,
                    )
                    if waypoint_b_only or route_config_mode else
                    "SKIP_DUPLICATE_ROUTE_TICK" if odom_stamp_action == "DUPLICATE"
                    else "ADVANCE_ROUTE_TICK"
                )
                route_progress_monitor["last_sample"]["sample_disposition"] = sample_disposition
                route_progress_monitor["samples"][-1]["sample_disposition"] = sample_disposition
                route_progress_monitor["last_sample"]["cross_track_limit_m"] = cross_track_limit
                route_progress_monitor["samples"][-1]["cross_track_limit_m"] = cross_track_limit
                if sample_disposition == "ABORT_CROSS_TRACK_GUARD":
                    route_fault = (
                        "curved_path_reference_cross_track_limit_exceeded"
                        if curved_path_diagnostic else
                        "configured_route_segment_cross_track_limit_exceeded"
                        if route_config_mode else
                        "waypoint_b_centerline_cross_track_limit_exceeded"
                    )
                    guard_event = {
                        "kind": "CURVED_PATH_CROSS_TRACK_GUARD" if curved_path_diagnostic else "WAYPOINT_B_CROSS_TRACK_GUARD",
                        **route_progress_monitor["last_sample"],
                        "limit_m": cross_track_limit,
                    }
                    route_progress_monitor["guard_triggers"].append(guard_event)
                    route_events.append(guard_event)
                    return True
                if waypoint_b_extended_diagnostic and waypoint_b_forward_guard_reached(
                    forward_projection_m,
                ):
                    route_fault = "extended_waypoint_b_forward_progress_guard_reached_without_arrival"
                    guard_event = {
                        "kind": "EXTENDED_B_FORWARD_PROGRESS_GUARD",
                        **route_progress_monitor["last_sample"],
                        "limit_m": WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
                    }
                    route_progress_monitor["guard_triggers"].append(guard_event)
                    route_events.append(guard_event)
                    return True

                handshake = same_sample_handshake(snapshot)
                route_activation_pending = bool(
                    self.activation_lifecycle is not None
                    and self.activation_lifecycle.activation_probe_command_allowed
                )
                heading_acquisition = window_control.get(
                    "zero_forward_heading_acquisition"
                )
                # ``transit_termination`` is defined outside the per-window
                # loop.  Keep the window-specific choice in its explicit
                # state instead of closing over the loop-local flag.
                heading_acquisition_enabled = bool(
                    window_control.get("zero_forward_heading_acquisition_enabled", False)
                )
                if route_activation_pending:
                    # The documented source contract needs a first bounded
                    # nonzero publish to acquire arbitration.  The explicitly
                    # selected RAICOM profile uses a source-grounded yaw-only
                    # command when fresh bearing error exceeds the documented
                    # angular start threshold; all pose/cross-track guards
                    # above remain active.  Other profiles retain the prior
                    # straight acquisition behavior.
                    window_control.update({
                        "steering_enabled": False,
                        "initial_source_grounded_activation_pending": True,
                        "source_mode_handshake": handshake,
                        "zero_forward_heading_acquisition_active": bool(
                            heading_acquisition_enabled
                            and isinstance(heading_acquisition, Mapping)
                            and heading_acquisition.get("status") == "ACTIVE"
                        ),
                    })
                    if (
                        heading_acquisition_enabled
                        and isinstance(heading_acquisition, Mapping)
                        and heading_acquisition.get("status") == "ACTIVE"
                    ):
                        try:
                            acquisition_elapsed = sample_time - float(
                                heading_acquisition["started_steady_s"]
                            )
                        except (KeyError, TypeError, ValueError):
                            route_fault = "zero_forward_heading_acquisition_state_invalid"
                            return True
                        heading_acquisition["elapsed_s"] = acquisition_elapsed
                        if acquisition_elapsed >= ZERO_FORWARD_HEADING_ACQUISITION_MAX_S:
                            route_fault = "zero_forward_heading_acquisition_source_mode_not_confirmed"
                            heading_acquisition["status"] = "FAILED_SOURCE_MODE_HANDSHAKE"
                            route_events.append({
                                "kind": "ZERO_FORWARD_HEADING_ACQUISITION_ABORT",
                                "reason": route_fault,
                                **dict(heading_acquisition),
                            })
                            return True
                    return False
                active_readbacks_valid = active_locomotion_source_readbacks_valid(
                    snapshot,
                    expected_source=self.source,
                    maximum_age_s=float(self.args.state_timeout),
                )
                window_control["initial_source_grounded_activation_pending"] = False
                if (
                    heading_acquisition_enabled
                    and isinstance(heading_acquisition, Mapping)
                    and heading_acquisition.get("status") == "ACTIVE"
                ):
                    try:
                        started = float(heading_acquisition["started_steady_s"])
                        initial_yaw = float(heading_acquisition["initial_yaw_rad"])
                        commanded = float(
                            heading_acquisition["angular_velocity_rad_s"]
                        )
                        elapsed = sample_time - started
                        yaw_delta = math.atan2(
                            math.sin(pose.yaw_rad - initial_yaw),
                            math.cos(pose.yaw_rad - initial_yaw),
                        )
                        signed_response = yaw_delta * math.copysign(1.0, commanded)
                    except (KeyError, TypeError, ValueError):
                        route_fault = "zero_forward_heading_acquisition_state_invalid"
                        return True
                    heading_acquisition.update({
                        "elapsed_s": elapsed,
                        "measured_yaw_delta_rad": yaw_delta,
                        "commanded_sign_yaw_delta_rad": signed_response,
                        "same_sample_handshake": handshake,
                        "active_readbacks_valid": active_readbacks_valid,
                    })
                    if elapsed < ZERO_FORWARD_HEADING_ACQUISITION_MAX_S:
                        return False
                    if not handshake or not active_readbacks_valid:
                        route_fault = "zero_forward_heading_acquisition_source_mode_not_confirmed"
                        heading_acquisition["status"] = "FAILED_SOURCE_MODE_HANDSHAKE"
                        route_events.append({
                            "kind": "ZERO_FORWARD_HEADING_ACQUISITION_ABORT",
                            "reason": route_fault,
                            **dict(heading_acquisition),
                        })
                        return True
                    if signed_response < ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD:
                        route_fault = (
                            "zero_forward_heading_acquisition_opposite_yaw_response"
                            if signed_response < -ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD
                            else "zero_forward_heading_acquisition_response_too_small"
                        )
                        heading_acquisition["status"] = "FAILED_YAW_RESPONSE"
                        route_events.append({
                            "kind": "ZERO_FORWARD_HEADING_ACQUISITION_ABORT",
                            "reason": route_fault,
                            **dict(heading_acquisition),
                        })
                        return True
                    heading_acquisition["status"] = "QUALIFIED"
                    active_window_gate.seed_verified_acquisition(
                        started_steady_s=started,
                        observed_steady_s=sample_time,
                    )
                    result["zero_forward_heading_acquisition"]["windows"].append(
                        dict(heading_acquisition)
                    )
                gate_state = active_window_gate.observe(
                    sample_time,
                    handshake,
                    active_readbacks_valid=active_readbacks_valid,
                )
                window_control.update(gate_state)
                if gate_state["fault"]:
                    route_fault = str(gate_state["fault"])
                    route_events.append({
                        "kind": "TRANSIT_STEERING_HANDSHAKE_ABORT",
                        "reason": route_fault,
                        "sample_monotonic_s": sample_time,
                        "straight_warmup_elapsed_s": gate_state["elapsed_s"],
                        "source_mode_handshake": handshake,
                    })
                    return True
                if not gate_state["steering_enabled"]:
                    return False
                if sample_disposition == "SKIP_DUPLICATE_ROUTE_TICK":
                    # The command layer may hold the last bounded control for
                    # this fresh-but-unchanged sample; do not advance or abort
                    # the core route state on a duplicate source timestamp.
                    return False
                distance = math.hypot(active.x_m - pose.x_m, active.y_m - pose.y_m)
                docking_decision = None
                if configured_docking_route_mode and active.yaw_policy is YawPolicy.EXPLICIT:
                    if configured_docking_control["step_id"] != active.waypoint_id:
                        segment_length = math.hypot(
                            active.x_m - float(route_segment_origin["x_m"]),
                            active.y_m - float(route_segment_origin["y_m"]),
                        )
                        if segment_length <= 1e-6:
                            route_fault = "configured_docking_segment_has_zero_length"
                            return True
                        configured_docking_control.update({
                            "step_id": active.waypoint_id,
                            "phase_controller": PoseGoalPhaseController(
                                goal_distance_m=segment_length,
                                final_position_tolerance_m=active.position_tolerance_m,
                                maximum_lateral_error_m=WAYPOINT_B_MAX_CROSS_TRACK_M,
                                maximum_overshoot_m=0.05,
                                final_yaw_tolerance_rad=float(active.yaw_tolerance_rad),
                                terminal_yaw_setpoint_tolerance_rad=float(active.yaw_tolerance_rad),
                            ),
                            "last_decision": None,
                        })
                    yaw_error = math.atan2(
                        math.sin(float(active.yaw_rad) - pose.yaw_rad),
                        math.cos(float(active.yaw_rad) - pose.yaw_rad),
                    )
                    try:
                        measured_forward = point_goal_forward_speed_m_s(
                            snapshot["pose"],
                            start_yaw_rad=math.atan2(
                                active.y_m - float(route_segment_origin["y_m"]),
                                active.x_m - float(route_segment_origin["x_m"]),
                            ),
                        )
                        stop_advance = speed_scaled_point_goal_stop_advance_m(
                            measured_forward,
                            endpoint_response_intercept_m=POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M,
                            endpoint_response_time_s=POINT_GOAL_ENDPOINT_RESPONSE_TIME_S,
                            maximum_forward_speed_m_s=0.2,
                        )
                        docking_position_stop_reason = point_goal_stop_reason(
                            along_track_progress_m=forward_projection_m,
                            lateral_error_m=cross_track_m,
                            goal_distance_m=math.hypot(
                                active.x_m - float(route_segment_origin["x_m"]),
                                active.y_m - float(route_segment_origin["y_m"]),
                            ),
                            dynamic_stop_progress_m=math.hypot(
                                active.x_m - float(route_segment_origin["x_m"]),
                                active.y_m - float(route_segment_origin["y_m"]),
                            ) + stop_advance,
                            final_tolerance_m=active.position_tolerance_m,
                            predicted_settled_tolerance_m=max(0.02, active.position_tolerance_m - 0.04),
                            maximum_lateral_error_m=WAYPOINT_B_MAX_CROSS_TRACK_M,
                            maximum_overshoot_m=0.05,
                        )
                        yaw_rate_feedback = configured_docking_control["yaw_rate_feedback"]
                        measured_yaw_rate = None
                        if yaw_rate_feedback is not None:
                            try:
                                measured_yaw_rate = fresh_yaw_rate_feedback(
                                    yaw_rate_feedback,
                                    sample_monotonic_s=sample_time,
                                    maximum_age_s=MAX_POSE_AGE_S,
                                )["signed_yaw_rate_rad_s"]
                            except ValueError:
                                measured_yaw_rate = None
                        docking_decision = configured_docking_control["phase_controller"].update(
                            monotonic_s=sample_time,
                            position_stop_reason=docking_position_stop_reason,
                            distance_to_goal_m=distance,
                            lateral_error_m=cross_track_m,
                            along_track_progress_m=forward_projection_m,
                            yaw_error_rad=yaw_error,
                            measured_yaw_rate_rad_s=measured_yaw_rate,
                            terminal_setpoint_error_rad=yaw_error,
                        )
                        configured_docking_control["last_decision"] = docking_decision
                    except (KeyError, TypeError, ValueError) as exc:
                        route_fault = f"configured_docking_pose_goal:{type(exc).__name__}:{exc}"
                        return True
                    if docking_decision.get("failure_reason"):
                        route_fault = "configured_docking_pose_goal:" + str(docking_decision["failure_reason"])
                        return True
                arrival_trigger_radius_m = waypoint_b_arrival_trigger_radius_m(
                    acceptance_tolerance_m=active.position_tolerance_m,
                    stop_reserve_m=(
                        min(0.04, active.position_tolerance_m * 0.5)
                        if route_config_mode else
                        CURVED_PATH_STOP_RESERVE_M if curved_path_diagnostic else
                        WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_extended_diagnostic else
                        WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_short_diagnostic else 0.0
                    ),
                )
                feedback = MotionFeedback(
                    status=(
                        MotionStatus.ARRIVED
                        if (
                            docking_decision is not None
                            and docking_decision.get("phase") == POSE_GOAL_ZERO_STOP
                        ) or (
                            docking_decision is None
                            and distance <= arrival_trigger_radius_m
                        )
                        else MotionStatus.ACTIVE
                    ),
                    reset_gate_pass=True,
                    standing_gate_pass=True,
                    source_mode_handshake_pass=True,
                    safety_gates_pass=True,
                )
                prior_clock_rebases = executor.verified_wall_clock_step_rebases
                directive = executor.tick(float(snapshot["sample_monotonic_s"]), feedback)
                if (
                    active.stop_behavior is StopBehavior.ROLLING
                    and feedback.status is MotionStatus.ARRIVED
                    and executor.active_step is not None
                    and executor.active_step.waypoint_id != active.waypoint_id
                ):
                    route_segment_origin["x_m"] = active.x_m
                    route_segment_origin["y_m"] = active.y_m
                    route_events.append({
                        "kind": "ROLLING_WAYPOINT_PASSED",
                        "waypoint_id": active.waypoint_id,
                        "next_waypoint_id": executor.active_step.waypoint_id,
                        "sample_monotonic_s": sample_time,
                        "position_error_m": distance,
                        "position_tolerance_m": active.position_tolerance_m,
                        "zero_velocity_published": False,
                    })
                result["pose_clock_diagnostics"] = {
                    "policy": "strict_header_order_except_fresh_wall-aligned_MuJoCo_clock_rollback",
                    "verified_wall_clock_step_rebases": executor.verified_wall_clock_step_rebases,
                    "latest_odom_stamp_ns": pose.stamp_ns,
                    "latest_capture_wall_time_ns": pose.wall_time_ns,
                    "latest_pose_receive_steady_s": pose.received_steady_s,
                }
                if executor.verified_wall_clock_step_rebases > prior_clock_rebases:
                    route_events.append({
                        "kind": "VERIFIED_WALL_CLOCK_STEP_REBASE",
                        "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
                        "odom_stamp_ns": pose.stamp_ns,
                        "capture_wall_time_ns": pose.wall_time_ns,
                        "pose_receive_steady_s": pose.received_steady_s,
                    })
                if not route_events or route_events[-1]["kind"] != directive.kind.value:
                    route_events.append({"kind": directive.kind.value, "reason": directive.reason,
                                         "sample_monotonic_s": snapshot.get("sample_monotonic_s")})
                if directive.kind is DirectiveKind.REQUEST_PHYSICAL_STOP:
                    route_stop_reason = directive.reason
                    route_arrival_poses[active.waypoint_id] = {
                        "x_m": pose.x_m,
                        "y_m": pose.y_m,
                        "yaw_rad": pose.yaw_rad,
                        "sample_monotonic_s": pose.received_steady_s,
                    }
                    return True
                if directive.kind is DirectiveKind.STOP_AND_ABORT:
                    route_fault = directive.reason
                    return True
                return False

            def stop_mapping(gate: Mapping[str, Any]) -> dict[str, Any]:
                final_window = gate.get("final_window", {})
                return {
                    "status": gate.get("status"),
                    "window_start_steady_s": final_window.get("window_start_monotonic_s"),
                    "window_end_steady_s": final_window.get("window_end_monotonic_s"),
                    "sample_count": final_window.get("sample_count"),
                    "maximum_sample_gap_s": final_window.get("max_sample_gap_s"),
                    "maximum_state_age_s": final_window.get("max_state_age_s"),
                    "generation": str(self.args.attempt_id),
                }

            route_total_nonzero_s = 0.0
            # Every route path, including the one-window straight-gait
            # diagnostic, needs the same-run stationary displacement envelope
            # before it can call the shared transit executor.
            baseline_displacements = [
                float(window["forward_displacement_m"])
                for window in (stationary_baseline or {}).get("windows", [])
            ]
            transit_window_limit = (
                SHORT_ROUTE_MAX_WINDOWS if short_scene_route_mode else
                (
                    1 if native_nominal_straight_gait_diagnostic_mode
                    else 1 if continuous_table_approach_mode
                    else TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS
                    if continuous_ten_second_transit_mode
                    else int(local01_route_input["limits"]["max_windows"])
                )
                if table_approach_mission_mode else
                3 if route_config_mode else
                CURVED_PATH_MAX_WINDOWS if curved_path_diagnostic else
                WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS
                if waypoint_b_extended_diagnostic else WAYPOINT_B_MAX_WINDOWS
            )
            transit_aggregate_limit_s = (
                SHORT_ROUTE_MAX_AGGREGATE_S if short_scene_route_mode else
                TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                if continuous_table_approach_mode else
                float(local01_route_input["limits"]["max_aggregate_nonzero_s"])
                if table_approach_mission_mode else
                15.0 if route_config_mode else
                CURVED_PATH_AGGREGATE_NONZERO_MAX_S if curved_path_diagnostic else
                WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S
                if waypoint_b_extended_diagnostic else WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S
            )

            def run_transit_windows(
                initial_directive,
                initial_plan,
                segment_id: str,
            ) -> dict[str, Any]:
                """Run bounded 5 s windows, re-planning only after measured stop."""
                nonlocal route_fault, route_stop_reason, route_total_nonzero_s, latest_transit_snapshot, set_action
                nonlocal active_window_gate
                current_directive = initial_directive
                current_plan = initial_plan
                last_result: dict[str, Any] | None = None
                last_gate: Mapping[str, Any] = {}
                window_index = 0
                segment_nonzero_s = 0.0
                while True:
                    # The legacy low-cruise acquisition remains scoped to the
                    # first departure. The fixed-window profile may use the
                    # measured zero-forward yaw response before every open
                    # START->WAYPOINT window, never near the table.
                    fixed_window_open_segment_acquisition = (
                        fixed_window_heading_acquisition_required(
                            fixed_window_table_approach=fixed_window_table_approach_mode,
                            segment_id=segment_id,
                        )
                    )
                    heading_acquisition_this_window = bool(
                        (
                            zero_forward_heading_acquisition_mode
                            and segment_id == "WAYPOINT"
                            and window_index == 0
                        )
                        or fixed_window_open_segment_acquisition
                    )
                    if route_config_mode and len(result["segments"]) >= (
                        SHORT_ROUTE_MAX_WINDOWS if short_scene_route_mode else
                        (
                            1 if native_nominal_straight_gait_diagnostic_mode
                            else 1 if continuous_table_approach_mode
                            else TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS
                            if continuous_ten_second_transit_mode
                            else int(local01_route_input["limits"]["max_windows"])
                        )
                        if table_approach_mission_mode else 3
                    ):
                        return {
                            "status": "BLOCKED",
                            "fault": "configured_route_total_window_budget_exhausted",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            "segment_nonzero_command_time_s": segment_nonzero_s,
                        }
                    if window_index >= transit_window_limit:
                        return {
                            "status": "BLOCKED",
                            "fault": "waypoint_b_window_count_budget_exhausted",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            "segment_nonzero_command_time_s": segment_nonzero_s,
                        }
                    if window_index > 0:
                        # A prior window may rearm only after its full physical
                        # stop gate passed.  The first nonzero command of this
                        # next bounded window reacquires the same approved source.
                        lifecycle, next_set_action, activation_gate = self.begin_source_grounded_route_activation(
                            registration=registration,
                            phase=f"{segment_id.lower()}-window-{window_index + 1}-rearm",
                            forward_command_m_s=float(current_plan.forward_m_s),
                            allow_raicom_low_cruise=bool(
                                getattr(self.args, "raicom_low_cruise_profile", False)
                            ),
                            zero_forward_heading_acquisition=heading_acquisition_this_window,
                        )
                        result.setdefault("activation_rearms", []).append({
                            "window_index": window_index + 1,
                            "set_mc_action_ld": next_set_action,
                            "activation_gate": activation_gate,
                            "lifecycle_after_activation": lifecycle.snapshot(),
                        })
                        if activation_gate.get("status") != "READY":
                            recovery = self.recover_activation_to_stand(
                                lifecycle=lifecycle,
                                registration=registration,
                                set_action=next_set_action,
                                baseline=stationary_baseline,
                                reason=str(
                                    activation_gate.get("fault")
                                    or activation_gate.get("terminal_reason")
                                    or "route-window LOCOMOTION reactivation was not confirmed"
                                ),
                                phase=f"{segment_id.lower()}-window-{window_index + 1}-rearm-recovery",
                            )
                            result["activation_rearms"][-1]["stand_recovery"] = recovery
                            return {
                                "status": "BLOCKED",
                                "fault": "route_window_locomotion_reactivation_not_confirmed",
                                "activation_gate": activation_gate,
                                "stand_recovery": recovery,
                                "last_result": last_result,
                                "last_gate": last_gate,
                                "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            }
                        set_action = next_set_action
                        self.activation_lifecycle = lifecycle
                    remaining_route_s = (
                        SHORT_ROUTE_MAX_AGGREGATE_S if short_scene_route_mode else
                        TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                        if continuous_table_approach_mode else
                        float(local01_route_input["limits"]["max_aggregate_nonzero_s"])
                        if table_approach_mission_mode else
                        15.0 if route_config_mode else 30.0
                    ) - route_total_nonzero_s
                    remaining_transit_s = transit_aggregate_limit_s - segment_nonzero_s
                    window_cap_s = min(
                        TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                        if continuous_table_approach_mode else
                        float(current_plan.maximum_duration_s),
                        SHORT_ROUTE_MAX_WINDOW_S if short_scene_route_mode else
                        TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                        if continuous_table_approach_mode else
                        CURVED_PATH_MAX_WINDOW_S if curved_path_diagnostic else
                        TABLE_APPROACH_CONTINUOUS_WINDOW_S
                        if (
                            continuous_ten_second_transit_mode
                            or native_nominal_straight_gait_diagnostic_mode
                        ) else WAYPOINT_B_WINDOW_MAX_S,
                        remaining_transit_s,
                        remaining_route_s,
                    )
                    pulse_duration_s = window_cap_s - ROUTE_WINDOW_ZERO_MARGIN_S
                    if pulse_duration_s < 0.10:
                        return {
                            "status": "BLOCKED",
                            "fault": "route_transit_or_action_window_budget_exhausted",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            "segment_nonzero_command_time_s": segment_nonzero_s,
                        }

                    route_fault = None
                    route_stop_reason = None
                    latest_transit_snapshot = None
                    window_control.clear()
                    window_control["zero_forward_heading_acquisition_enabled"] = (
                        heading_acquisition_this_window
                    )
                    active_window_gate = TransitSteeringGate(warmup_s=ROUTE_STEERING_WARMUP_S)
                    steering_samples: list[dict[str, Any]] = []
                    transit_forward_m_s = min(
                        float(current_plan.forward_m_s), ROUTE_TRANSIT_MAX_FORWARD_M_S,
                    )

                    def publish_window(forward_m_s: float) -> None:
                        forward = float(forward_m_s)
                        angular = 0.0
                        control_row: dict[str, Any] = {
                            "sample_monotonic_s": None,
                            "forward_velocity_m_s": forward,
                            "angular_velocity_rad_s": 0.0,
                            "control_phase": "ZERO_STOP" if forward <= 0.0 else "STRAIGHT_HANDSHAKE_WARMUP",
                            "heading_error_rad": None,
                            "saturated": False,
                        }
                        snapshot = latest_transit_snapshot
                        # ``run_first_motion_window`` reuses this callback for
                        # every ZeroStopWindow publication.  A requested zero
                        # must win over every source-acquisition or steering
                        # branch: zero-stop may never be converted into a
                        # yaw-only command.
                        if forward <= 0.0:
                            self._publish(0.0, 0.0, 0.0)
                            return
                        if (
                            heading_acquisition_this_window
                            and isinstance(snapshot, Mapping)
                        ):
                            acquisition = window_control.get(
                                "zero_forward_heading_acquisition"
                            )
                            if acquisition is None:
                                pose = snapshot.get("pose")
                                active = executor.active_step
                                if not isinstance(pose, Mapping) or active is None:
                                    window_control["feedback_fault"] = (
                                        "zero_forward_heading_acquisition_pose_or_goal_missing"
                                    )
                                    self._publish(0.0, 0.0, 0.0)
                                    return
                                try:
                                    target_yaw = math.atan2(
                                        float(active.y_m) - float(pose["y"]),
                                        float(active.x_m) - float(pose["x"]),
                                    )
                                    heading_error = math.atan2(
                                        math.sin(target_yaw - float(pose["yaw"])),
                                        math.cos(target_yaw - float(pose["yaw"])),
                                    )
                                    source_plan = zero_forward_heading_acquisition_plan(
                                        heading_error,
                                        maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                    )
                                except (KeyError, TypeError, ValueError, RouteMotionBridgeError) as exc:
                                    window_control["feedback_fault"] = (
                                        "zero_forward_heading_acquisition_invalid:"
                                        f"{type(exc).__name__}:{exc}"
                                    )
                                    self._publish(0.0, 0.0, 0.0)
                                    return
                                acquisition = {
                                    **source_plan,
                                    "status": (
                                        "ACTIVE" if source_plan["required"]
                                        else "FALLBACK_STRAIGHT_WARMUP"
                                    ),
                                    "started_steady_s": float(
                                        snapshot["sample_monotonic_s"]
                                    ),
                                    "initial_yaw_rad": float(pose["yaw"]),
                                    "target_yaw_rad": target_yaw,
                                    "window_id": int(current_plan.window_id),
                                }
                                window_control["zero_forward_heading_acquisition"] = acquisition
                            if acquisition.get("status") == "ACTIVE":
                                forward = 0.0
                                angular = float(acquisition["angular_velocity_rad_s"])
                                control_row.update({
                                    "forward_velocity_m_s": 0.0,
                                    "angular_velocity_rad_s": angular,
                                    "control_phase": "ZERO_FORWARD_HEADING_SOURCE_ACQUISITION",
                                    "heading_error_rad": float(
                                        acquisition["heading_error_rad"]
                                    ),
                                    "target_yaw_rad": float(acquisition["target_yaw_rad"]),
                                    "initial_yaw_rad": float(acquisition["initial_yaw_rad"]),
                                    "source_acquisition_command_rad_s": angular,
                                    "source_acquisition_started_steady_s": float(
                                        acquisition["started_steady_s"]
                                    ),
                                })
                                steering_samples.append(control_row)
                                self._publish(0.0, 0.0, angular)
                                return
                        if forward > 0.0 and isinstance(snapshot, Mapping):
                            control_row["sample_monotonic_s"] = snapshot.get("sample_monotonic_s")
                            control_row["odom_stamp_action"] = window_control.get("odom_stamp_action")
                            control_row["odom_stamp_ns"] = window_control.get("odom_stamp_ns")
                            control_row["odom_receive_age_s"] = window_control.get("odom_receive_age_s")
                            control_row["odom_age_since_last_unique_s"] = window_control.get(
                                "odom_age_since_last_unique_s"
                            )
                            control_row["last_unique_odom_stamp_ns"] = window_control.get(
                                "last_unique_odom_stamp_ns"
                            )
                            if window_control.get("steering_enabled"):
                                pose = snapshot.get("pose")
                                active = executor.active_step
                                if not isinstance(pose, Mapping) or active is None:
                                    window_control["feedback_fault"] = "transit_feedback_steering_pose_or_goal_missing"
                                    self._publish(0.0, 0.0, 0.0)
                                    return
                                elif window_control.get("odom_stamp_action") == "DUPLICATE":
                                    angular = float(window_control.get("last_route_angular_command_rad_s", 0.0))
                                    control_row.update({
                                        "control_phase": "ZERO_ORDER_HOLD_DUPLICATE_ODOM",
                                        "angular_velocity_rad_s": angular,
                                        "raw_angular_velocity_rad_s": angular,
                                        "duplicate_odom_feedback_reused": True,
                                        "locomotion_handshake_confirmed": True,
                                    })
                                elif (
                                    configured_docking_route_mode
                                    and active.yaw_policy is YawPolicy.EXPLICIT
                                    and configured_docking_control.get("phase_controller") is not None
                                    and configured_docking_control["phase_controller"].phase == TERMINAL_POSE_CONTROL
                                ):
                                    yaw_rate_feedback = configured_docking_control.get("yaw_rate_feedback")
                                    try:
                                        measured = fresh_yaw_rate_feedback(
                                            yaw_rate_feedback,
                                            sample_monotonic_s=float(snapshot["sample_monotonic_s"]),
                                            maximum_age_s=MAX_POSE_AGE_S,
                                        )
                                        terminal = terminal_yaw_rate_command(
                                            current_yaw_rad=float(pose["yaw"]),
                                            target_yaw_rad=float(active.yaw_rad),
                                            measured_yaw_rate_rad_s=float(measured["signed_yaw_rate_rad_s"]),
                                            maximum_angular_rad_s=float(active.max_angular_rad_s),
                                            final_yaw_tolerance_rad=float(active.yaw_tolerance_rad),
                                        )
                                    except (KeyError, TypeError, ValueError) as exc:
                                        window_control["feedback_fault"] = (
                                            f"docking_terminal_yaw_feedback:{type(exc).__name__}:{exc}"
                                        )
                                        self._publish(0.0, 0.0, 0.0)
                                        return
                                    forward = 0.0
                                    angular = float(terminal["command_yaw_rate_rad_s"])
                                    control_row.update({
                                        "forward_velocity_m_s": 0.0,
                                        "control_phase": "DOCKING_TERMINAL_POSE_CONTROL",
                                        "target_yaw_rad": float(active.yaw_rad),
                                        "current_yaw_rad": float(pose["yaw"]),
                                        "yaw_error_rad": float(terminal["yaw_error_rad"]),
                                        "measured_yaw_rate_rad_s": float(measured["signed_yaw_rate_rad_s"]),
                                        "measured_yaw_rate_age_s": float(measured["age_s"]),
                                        "raw_angular_velocity_rad_s": float(terminal["raw_yaw_rate_command_rad_s"]),
                                        "saturated": abs(float(terminal["raw_yaw_rate_command_rad_s"])) > float(active.max_angular_rad_s),
                                        "locomotion_handshake_confirmed": True,
                                    })
                                elif native_nominal_straight_gait_diagnostic_mode:
                                    angular = 0.0
                                    control_row.update({
                                        "control_phase": "NATIVE_NOMINAL_STRAIGHT_GAIT_DIAGNOSTIC",
                                        "target_x_m": float(active.x_m),
                                        "target_y_m": float(active.y_m),
                                        "current_x_m": float(pose["x"]),
                                        "current_y_m": float(pose["y"]),
                                        "current_yaw_rad": float(pose["yaw"]),
                                        "heading_error_rad": None,
                                        "raw_angular_velocity_rad_s": 0.0,
                                        "forward_gait_yaw_mapping": "ZERO_YAW_DIAGNOSTIC",
                                        "locomotion_handshake_confirmed": True,
                                    })
                                elif fixed_window_table_approach_mode:
                                    fixed_arc = window_control.get(
                                        "fixed_window_arc_decision"
                                    )
                                    if not isinstance(fixed_arc, Mapping):
                                        fixed_bearing = waypoint_bearing_angular_command(
                                            current_x_m=float(pose["x"]),
                                            current_y_m=float(pose["y"]),
                                            current_yaw_rad=float(pose["yaw"]),
                                            target_x_m=float(active.x_m),
                                            target_y_m=float(active.y_m),
                                            maximum_angular_rad_s=
                                            POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                        )
                                        fixed_arc = {
                                            **fixed_window_arc_angular_command(
                                                float(fixed_bearing["angular_rad_s"]),
                                                maximum_angular_rad_s=
                                                POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                            ),
                                            "decision_pose": {
                                                "x_m": float(pose["x"]),
                                                "y_m": float(pose["y"]),
                                                "yaw_rad": float(pose["yaw"]),
                                            },
                                            "decision_bearing_error_rad": float(
                                                fixed_bearing["bearing_error_rad"]
                                            ),
                                            "decision_raw_angular_rad_s": float(
                                                fixed_bearing["raw_angular_rad_s"]
                                            ),
                                        }
                                        window_control["fixed_window_arc_decision"] = fixed_arc
                                    angular = float(fixed_arc["angular_rad_s"])
                                    control_row.update({
                                        "control_phase": "FIXED_PREWINDOW_STRAIGHT_OR_RUN65_ARC",
                                        "target_x_m": float(active.x_m),
                                        "target_y_m": float(active.y_m),
                                        "current_x_m": float(pose["x"]),
                                        "current_y_m": float(pose["y"]),
                                        "current_yaw_rad": float(pose["yaw"]),
                                        "planned_angular_velocity_rad_s": float(
                                            fixed_arc["planned_angular_rad_s"]
                                        ),
                                        "fixed_decision_pose": dict(
                                            fixed_arc["decision_pose"]
                                        ),
                                        "fixed_decision_bearing_error_rad": float(
                                            fixed_arc["decision_bearing_error_rad"]
                                        ),
                                        "fixed_arc_mode": str(fixed_arc["mode"]),
                                        "fixed_arc_deadband_rad_s": float(
                                            fixed_arc["deadband_rad_s"]
                                        ),
                                        "raw_angular_velocity_rad_s": angular,
                                        "saturated": bool(
                                            fixed_arc["saturated_to_validated_arc"]
                                        ),
                                        "locomotion_handshake_confirmed": True,
                                    })
                                elif curved_path_diagnostic:
                                    try:
                                        path_feedback = curved_path_metrics(
                                            curved_reference,
                                            x_m=float(pose["x"]),
                                            y_m=float(pose["y"]),
                                        )
                                        measured_yaw_rate = float(pose["wz"])
                                        if not math.isfinite(measured_yaw_rate):
                                            raise ValueError("odometry yaw rate is nonfinite")
                                        steering = pose_path_tracking_yaw_rate_command(
                                            current_yaw_rad=float(pose["yaw"]),
                                            path_tangent_yaw_rad=float(path_feedback["path_tangent_yaw_rad"]),
                                            measured_yaw_rate_rad_s=measured_yaw_rate,
                                            path_curvature_per_m=float(curved_reference["curvature_per_m"]),
                                            forward_command_m_s=forward,
                                            maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                        )
                                    except (KeyError, TypeError, ValueError, RouteMotionBridgeError) as exc:
                                        window_control["feedback_fault"] = f"curved_path_control_feedback:{type(exc).__name__}:{exc}"
                                        self._publish(0.0, 0.0, 0.0)
                                        return
                                    angular = float(steering["command_yaw_rate_rad_s"])
                                    control_row.update({
                                        "control_phase": "CLOSED_LOOP_CURVED_PATH_TRACKING",
                                        "path_progress_m": float(path_feedback["path_progress_m"]),
                                        "path_progress_fraction": float(path_feedback["path_progress_fraction"]),
                                        "path_tangent_yaw_rad": float(path_feedback["path_tangent_yaw_rad"]),
                                        "path_curvature_per_m": float(curved_reference["curvature_per_m"]),
                                        "path_cross_track_error_m": float(path_feedback["cross_track_error_m"]),
                                        "current_x_m": float(pose["x"]),
                                        "current_y_m": float(pose["y"]),
                                        "current_yaw_rad": float(pose["yaw"]),
                                        "measured_yaw_rate_child_z_rad_s": measured_yaw_rate,
                                        "path_yaw_error_rad": float(steering["path_yaw_error_rad"]),
                                        "curvature_feedforward_rad_s": float(steering["curvature_feedforward_rad_s"]),
                                        "yaw_rate_feedback_rad_s": float(steering["heading_rate_feedback_rad_s"]),
                                        "raw_angular_velocity_rad_s": float(steering["raw_yaw_rate_command_rad_s"]),
                                        "saturated": bool(steering["command_saturated"]),
                                        "locomotion_handshake_confirmed": True,
                                    })
                                else:
                                    lookahead_reference = None
                                    if continuous_table_approach_mode:
                                        lookahead_reference = segment_lookahead_point(
                                            segment_start_x_m=float(route_segment_origin["x_m"]),
                                            segment_start_y_m=float(route_segment_origin["y_m"]),
                                            segment_end_x_m=float(active.x_m),
                                            segment_end_y_m=float(active.y_m),
                                            current_x_m=float(pose["x"]),
                                            current_y_m=float(pose["y"]),
                                            lookahead_m=CONTINUOUS_TABLE_LOOKAHEAD_M,
                                        )
                                    steering_target_x_m = float(
                                        lookahead_reference["x_m"]
                                        if lookahead_reference is not None else active.x_m
                                    )
                                    steering_target_y_m = float(
                                        lookahead_reference["y_m"]
                                        if lookahead_reference is not None else active.y_m
                                    )
                                    steering = waypoint_bearing_angular_command(
                                        current_x_m=float(pose["x"]),
                                        current_y_m=float(pose["y"]),
                                        current_yaw_rad=float(pose["yaw"]),
                                        target_x_m=steering_target_x_m,
                                        target_y_m=steering_target_y_m,
                                        maximum_angular_rad_s=POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                    )
                                    standard_angular = float(steering["angular_rad_s"])
                                    angular = forward_gait_yaw_command(
                                        standard_angular,
                                        invert_from_measured_response=
                                        forward_gait_yaw_inversion_mode,
                                        maximum_angular_rad_s=
                                        POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                                    )
                                    control_row.update({
                                        "control_phase": "CLOSED_LOOP_WAYPOINT_BEARING",
                                        "target_x_m": steering_target_x_m,
                                        "target_y_m": steering_target_y_m,
                                        "terminal_target_x_m": float(active.x_m),
                                        "terminal_target_y_m": float(active.y_m),
                                        "segment_lookahead": lookahead_reference,
                                        "current_x_m": float(pose["x"]),
                                        "current_y_m": float(pose["y"]),
                                        "current_yaw_rad": float(pose["yaw"]),
                                        "heading_error_rad": float(steering["bearing_error_rad"]),
                                        "raw_angular_velocity_rad_s": float(steering["raw_angular_rad_s"]),
                                        "standard_odom_bearing_angular_velocity_rad_s": standard_angular,
                                        "forward_gait_yaw_mapping": (
                                            "INVERTED_FROM_MEASURED_NATIVE_RESPONSE"
                                            if forward_gait_yaw_inversion_mode
                                            else "STANDARD_ODOM_BEARING"
                                        ),
                                        "saturated": bool(steering["saturated"]),
                                        "locomotion_handshake_confirmed": True,
                                    })
                            window_control["last_route_angular_command_rad_s"] = angular
                            control_row["angular_velocity_rad_s"] = angular
                            control_row["straight_warmup_elapsed_s"] = window_control.get("elapsed_s")
                            steering_samples.append(control_row)
                        elif forward > 0.0:
                            window_control["last_route_angular_command_rad_s"] = 0.0
                        terminal_rotation = control_row.get("control_phase") == "DOCKING_TERMINAL_POSE_CONTROL"
                        self._publish(
                            forward,
                            0.0,
                            angular if forward > 0.0 or terminal_rotation else 0.0,
                        )

                    last_result = run_first_motion_window(
                        sample=route_sample,
                        publish=publish_window,
                        registration=registration,
                        set_action=set_action,
                        joint_limits=self.joint_limits,
                        baseline_forward_displacements_m=baseline_displacements,
                        baseline_state_summary=(stationary_baseline or {}).get("physical_state_summary"),
                        authorized=True,
                        forward_command_m_s=transit_forward_m_s,
                        pulse_duration_s=pulse_duration_s,
                        max_pulse_duration_s=window_cap_s,
                        continuous_route_authorized=continuous_table_approach_mode,
                        continuous_route_nonzero_limit_s=(
                            TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S
                            if continuous_table_approach_mode else None
                        ),
                        extra_post_stop_observe_s=float(self.args.extra_stop_observation),
                        extra_post_stop_observe_limit_s=(
                            MAX_EXTRA_POST_STOP_OBSERVE_S
                            if route_config_mode
                            or getattr(self.args, "route_waypoint_b_post_locomotion_convergence", False)
                            else 5.0
                        ),
                        after_zero_stop=self.request_postwalk_stand_once,
                        after_zero_abort=self.request_postwalk_stand_once,
                        shutdown_event=self.interrupted,
                        on_sample=lambda phase, snapshot, n=window_index + 1: self._trace_first_motion_sample(
                            f"{segment_id.lower()}-window-{n}-{phase}", snapshot,
                        ),
                        termination_condition=transit_termination,
                        termination_condition_name=f"{segment_id.lower()}_waypoint_and_physical_stop",
                        state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z,
                        max_tilt_deg=self.args.max_tilt_deg,
                        max_observed_planar_speed_m_s=0.2,
                        zero_until_stand_timeout_s=3.5,
                        require_locomotion_confirmation=True,
                        activation_lifecycle=self.activation_lifecycle,
                        source_contract_supported=True,
                        source_grounded_route_activation=True,
                        source_grounded_activation_timeout_s=ACTIVATION_PROBE_DURATION_S,
                        allow_raicom_low_cruise_source_activation=bool(
                            getattr(self.args, "raicom_low_cruise_profile", False)
                        ),
                    )
                    raw_stop_gate = last_result.get("physical_stop_gate")
                    final_stop_window = (
                        raw_stop_gate.get("final_window")
                        if isinstance(raw_stop_gate, Mapping) else None
                    )
                    if (
                        isinstance(final_stop_window, Mapping)
                        and isinstance(final_stop_window.get("window_start_monotonic_s"), (int, float))
                        and isinstance(final_stop_window.get("window_end_monotonic_s"), (int, float))
                    ):
                        route_odom_stop_gate = evaluate_route_odom_freshness_window(
                            last_result.get("samples", []),
                            window_start_steady_s=float(final_stop_window["window_start_monotonic_s"]),
                            window_end_steady_s=float(final_stop_window["window_end_monotonic_s"]),
                            maximum_age_s=MAX_POSE_AGE_S,
                            minimum_window_s=1.90,
                        )
                    else:
                        route_odom_stop_gate = {
                            "status": "FAIL",
                            "reason": "COMPOSITE_STOP_WINDOW_UNAVAILABLE_FOR_ODOMETRY_FRESHNESS",
                            "observations": [],
                        }
                    last_result["route_stop_odom_freshness_gate"] = route_odom_stop_gate
                    route_rearm_gate = evaluate_route_window_rearm_gate(
                        physical_stop_status=str(
                            (raw_stop_gate or {}).get("status", "NOT_EVALUATED")
                            if isinstance(raw_stop_gate, Mapping) else "NOT_EVALUATED"
                        ),
                        odometry_freshness_status=str(route_odom_stop_gate.get("status")),
                    )
                    last_result["route_window_rearm_gate"] = route_rearm_gate
                    angular_values = [
                        float(row["angular_velocity_rad_s"]) for row in steering_samples
                    ]
                    last_result["route_feedback_steering"] = {
                        "policy": (
                            "FRESH_ODOM_ARC_TANGENT_CURVATURE_AND_YAW_RATE_FEEDBACK_EVERY_PUBLISH"
                            if curved_path_diagnostic else
                            "FIXED_FRESH_PREWINDOW_STRAIGHT_OR_RUN65_ARC"
                            if fixed_window_table_approach_mode else
                            "FRESH_ODOM_WAYPOINT_BEARING_EVERY_PUBLISH"
                        ),
                        "continuous_table_segment_lookahead_m": (
                            CONTINUOUS_TABLE_LOOKAHEAD_M
                            if continuous_table_approach_mode else None
                        ),
                        "heading_gain": 0.8,
                        "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                        "straight_warmup_s": ROUTE_STEERING_WARMUP_S,
                        "zero_forward_heading_acquisition": window_control.get(
                            "zero_forward_heading_acquisition"
                        ),
                        "zero_forward_heading_acquisition_profile_enabled": (
                            heading_acquisition_this_window
                        ),
                        "forward_gait_yaw_inversion_profile_enabled": (
                            forward_gait_yaw_inversion_mode
                        ),
                        "fresh_mode_source_handshake_required_before_steering": True,
                        "handshake_state": dict(window_control),
                        "sample_count": len(steering_samples),
                        "zero_order_hold_duplicate_odom_sample_count": sum(
                            1 for row in steering_samples
                            if row.get("duplicate_odom_feedback_reused") is True
                        ),
                        "maximum_odom_receive_age_s": max(
                            (float(row["odom_receive_age_s"]) for row in steering_samples
                             if row.get("odom_receive_age_s") is not None),
                            default=None,
                        ),
                        "angular_command_min_rad_s": min(angular_values) if angular_values else None,
                        "angular_command_max_rad_s": max(angular_values) if angular_values else None,
                        "samples": steering_samples,
                    }
                    last_result["command_angular_velocity_rad_s"] = None
                    last_result["planned_initial_angular_velocity_rad_s"] = float(current_plan.angular_rad_s)
                    last_result["executed_forward_command_m_s"] = transit_forward_m_s
                    last_result["route_window_maximum_duration_s"] = window_cap_s
                    try:
                        measured_window_s = measured_m1_nonzero_window_s(last_result)
                    except RouteMotionBridgeError as exc:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_window_elapsed_unavailable:{exc}",
                            "last_result": last_result,
                            "last_gate": last_result.get("physical_stop_gate") or {},
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    last_result["route_window_measured_nonzero_elapsed_s"] = measured_window_s
                    if measured_window_s > window_cap_s + 1e-6:
                        return {
                            "status": "BLOCKED",
                            "fault": "route_window_nonzero_command_deadline_exceeded",
                            "last_result": last_result,
                            "last_gate": last_result.get("physical_stop_gate") or {},
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    if continuous_table_approach_mode:
                        last_result["route_bridge_accounting"] = {
                            "status": "CONTINUOUS_TERMINAL_WINDOW_NOT_SPLIT_AT_LEGACY_10_S_PLAN_BOUNDARY",
                            "existing_aggregate_nonzero_cap_s": TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S,
                            "measured_nonzero_elapsed_s": measured_window_s,
                            "physical_stop_remains_required_at_terminal": True,
                        }
                    else:
                        try:
                            bridge.complete_window(
                                current_plan.window_id,
                                actual_elapsed_s=measured_window_s,
                            )
                        except RouteMotionBridgeError as exc:
                            return {
                                "status": "BLOCKED",
                                "fault": f"route_window_completion_rejected:{exc}",
                                "last_result": last_result,
                                "last_gate": last_result.get("physical_stop_gate") or {},
                                "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            }
                    route_total_nonzero_s += measured_window_s
                    segment_nonzero_s += measured_window_s
                    window_index += 1
                    last_gate = last_result.get("physical_stop_gate") or {}
                    result["segments"].append({
                        "segment_id": f"{segment_id}-WINDOW-{window_index}",
                        "plan": {
                            **dict(current_plan.__dict__),
                            "commanded_window_duration_s": pulse_duration_s,
                            "bridge_proposed_angular_velocity_rad_s": float(current_plan.angular_rad_s),
                            "runtime_angular_velocity_range_rad_s": [
                                last_result["route_feedback_steering"]["angular_command_min_rad_s"],
                                last_result["route_feedback_steering"]["angular_command_max_rad_s"],
                            ],
                            "runtime_steering_sample_count": last_result["route_feedback_steering"]["sample_count"],
                        },
                        "result": last_result,
                    })

                    if (
                        route_fault
                        or last_result.get("physical_motion_status") != "PHYSICAL_MOTION_CONFIRMED"
                        or last_gate.get("status") != "PASS"
                        or route_rearm_gate.get("rearm_allowed") is not True
                    ):
                        return {
                            "status": "BLOCKED",
                            "fault": (
                                route_fault or last_result.get("fault")
                                or (
                                    "route_stop_rearm_gate_failed:"
                                    + ",".join(str(item) for item in route_rearm_gate.get("blockers", []))
                                    if route_rearm_gate.get("rearm_allowed") is not True
                                    else "route_window_motion_or_stop_gate_failed"
                                )
                            ),
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }

                    if route_stop_reason is not None:
                        return {
                            "status": "ARRIVAL_DURING_WINDOW",
                            "fault": None,
                            "arrived": True,
                            "bridge_stop_rearmed": False,
                            "stop_reason": route_stop_reason,
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }

                    evidence = stop_mapping(last_gate)
                    zero_start_s = float(last_gate.get("zero_start_monotonic_s", 0.0))
                    zero_plan = bridge.zero_window(
                        route_motion.latest,
                        now_steady_s=time.monotonic(),
                        reason=f"{segment_id} action window ended without waypoint arrival; verify stop before replanning",
                        physical_stop_gate_required=True,
                    )
                    bridge_rearmed = bridge.confirm_physical_stop(
                        evidence,
                        zero_request_started_s=zero_start_s,
                        now_steady_s=time.monotonic(),
                        expected_generation=str(self.args.attempt_id),
                    )
                    if not bridge_rearmed:
                        return {
                            "status": "BLOCKED",
                            "fault": "route_window_stop_evidence_failed_to_rearm_bridge",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }

                    if native_nominal_straight_gait_diagnostic_mode:
                        return {
                            "status": "STRAIGHT_GAIT_DIAGNOSTIC_COMPLETE",
                            "fault": None,
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "bridge_stop_rearmed": True,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            "segment_nonzero_command_time_s": segment_nonzero_s,
                        }

                    previous_odom_stamp_ns = int(
                        self.odom_source_stamp_fence.last_accepted_stamp_ns or 0
                    )
                    try:
                        raw_replan_snapshot, replan_freshness = (
                            self.wait_for_fresh_route_replan_snapshot(
                                previous_odom_stamp_ns=previous_odom_stamp_ns
                            )
                        )
                    except TimeoutError as exc:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_replan_fresh_odom_timeout:{exc}",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    route_events.append({
                        "kind": "ROUTE_REPLAN_FRESH_ODOM",
                        "completed_window_index": window_index,
                        **replan_freshness,
                    })
                    snapshot = route_feedback_snapshot(raw_replan_snapshot)
                    try:
                        pose = pose_provider.update(snapshot)
                    except (KeyError, TypeError, ValueError) as exc:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_replan_pose_invalid:{type(exc).__name__}:{exc}",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    active = executor.active_step
                    if active is None:
                        return {
                            "status": "BLOCKED",
                            "fault": "route_active_step_missing_after_verified_stop",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    safety_provider.update(snapshot, active.waypoint_id)
                    safety_fault = snapshot_fault(
                        snapshot,
                        self.joint_limits,
                        state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z,
                        max_tilt_deg=self.args.max_tilt_deg,
                        max_observed_planar_speed_m_s=0.2,
                    )
                    if safety_fault:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_replan_safety:{safety_fault}",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    stop_evidence = PhysicalStopEvidence(
                        status=PhysicalStopStatus.PASS,
                        window_start_steady_s=float(evidence["window_start_steady_s"]),
                        window_end_steady_s=float(evidence["window_end_steady_s"]),
                        sample_count=int(evidence["sample_count"]),
                        maximum_sample_gap_s=float(evidence["maximum_sample_gap_s"]),
                        maximum_state_age_s=float(evidence["maximum_state_age_s"]),
                        generation=str(self.args.attempt_id),
                    )
                    distance = math.hypot(active.x_m - pose.x_m, active.y_m - pose.y_m)
                    arrived_after_stop = distance <= active.position_tolerance_m
                    poststop_yaw_error = None
                    if active.yaw_policy is YawPolicy.EXPLICIT:
                        poststop_yaw_error = math.atan2(
                            math.sin(float(active.yaw_rad) - pose.yaw_rad),
                            math.cos(float(active.yaw_rad) - pose.yaw_rad),
                        )
                        arrived_after_stop = arrived_after_stop and abs(poststop_yaw_error) <= float(
                            active.yaw_tolerance_rad
                        )
                    source_readback_passed = fresh_input_source_readback_valid(
                        snapshot,
                        expected_source=self.source,
                        maximum_age_s=float(self.args.state_timeout),
                    )
                    stand_readback_passed = fresh_stand_mode_readback_valid(
                        snapshot,
                        maximum_age_s=float(self.args.state_timeout),
                    )
                    arrival_after_verified_stop = waypoint_b_arrival_after_verified_stop_passes(
                        distance,
                        distance,
                        tolerance_m=active.position_tolerance_m,
                        composite_stop_gate_passed=last_gate.get("status") == "PASS",
                        bridge_rearmed=bridge_rearmed,
                        fresh_stand_readback_passed=stand_readback_passed,
                        source_readback_passed=source_readback_passed,
                        safety_gates_passed=safety_fault is None,
                    )
                    if arrived_after_stop:
                        # Consume the M1 composite stop exactly once. If this
                        # post-stop pose is an arrival, the route executor
                        # advances or completes from the evidence already
                        # collected instead of issuing another stop request.
                        directive_after_verified_stop = executor.accept_arrival_after_verified_stop(
                            float(snapshot["sample_monotonic_s"]),
                            MotionFeedback(
                                status=MotionStatus.ARRIVED,
                                physical_stop_evidence=stop_evidence,
                                reset_gate_pass=True,
                                standing_gate_pass=True,
                                source_mode_handshake_pass=False,
                                safety_gates_pass=safety_fault is None,
                                stand_readback_pass=stand_readback_passed,
                                input_source_readback_pass=source_readback_passed,
                            ),
                            zero_request_started_s=zero_start_s,
                            bridge_rearmed=bridge_rearmed,
                        )
                        if directive_after_verified_stop.kind is DirectiveKind.STOP_AND_ABORT:
                            return {
                                "status": "BLOCKED",
                                "fault": directive_after_verified_stop.reason,
                                "last_result": last_result,
                                "last_gate": last_gate,
                                "zero_plan": dict(zero_plan.__dict__),
                                "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            }
                        route_stop_reason = f"WAYPOINT_REACHED_AFTER_VERIFIED_STOP:{active.waypoint_id}"
                        route_arrival_poses[active.waypoint_id] = {
                            "x_m": pose.x_m,
                            "y_m": pose.y_m,
                            "yaw_rad": pose.yaw_rad,
                            "sample_monotonic_s": pose.received_steady_s,
                        }
                        route_events.append({
                            "kind": "WAYPOINT_REACHED_AFTER_VERIFIED_STOP",
                            "waypoint_id": active.waypoint_id,
                            "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
                            "position_error_m": distance,
                            "position_tolerance_m": active.position_tolerance_m,
                            "composite_stop_gate": last_gate.get("status"),
                            "bridge_rearmed": bridge_rearmed,
                            "stand_readback_passed": stand_readback_passed,
                            "source_readback_passed": source_readback_passed,
                            "safety_gates_passed": safety_fault is None,
                            "prior_stop_gate_precheck_passed": arrival_after_verified_stop,
                            "route_executor_state": executor.state.value,
                            "route_executor_directive": directive_after_verified_stop.kind.value,
                        })
                        return {
                            "status": "ARRIVAL_AFTER_VERIFIED_STOP",
                            "fault": None,
                            "arrived": True,
                            "completion_evidence": "M1_COMPOSITE_STOP_GATE_CONSUMED_BY_ROUTE_EXECUTOR",
                            "bridge_stop_rearmed": True,
                            "stop_reason": route_stop_reason,
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "next_directive": directive_after_verified_stop,
                            "final_snapshot": snapshot,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    if not stand_readback_passed or not source_readback_passed or safety_fault:
                        return {
                            "status": "BLOCKED",
                            "fault": (
                                "poststop_rearm_requires_fresh_stand_source_and_safety_readbacks:"
                                f"stand={stand_readback_passed},source={source_readback_passed},safety={safety_fault}"
                            ),
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    route_events.append({
                        "kind": "WINDOW_REARM_AFTER_COMPOSITE_STOP",
                        "waypoint_id": active.waypoint_id,
                        "completed_window_index": window_index,
                        "next_window_index": window_index + 1,
                        "physical_stop_gate": last_gate.get("status"),
                        "bridge_rearmed": bridge_rearmed,
                        "stand_readback_passed": stand_readback_passed,
                        "input_source_readback_passed": source_readback_passed,
                        "safety_gates_passed": safety_fault is None,
                        "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
                    })
                    directive_after_stop = executor.tick(
                        float(snapshot["sample_monotonic_s"]),
                        MotionFeedback(
                            status=MotionStatus.ARRIVED if arrived_after_stop else MotionStatus.ACTIVE,
                            physical_stop_evidence=stop_evidence,
                            reset_gate_pass=True,
                            standing_gate_pass=True,
                            source_mode_handshake_pass=False,
                            safety_gates_pass=True,
                            stand_readback_pass=stand_mode_readback_confirmed(snapshot),
                            input_source_readback_pass=source_readback_passed,
                        ),
                    )
                    if directive_after_stop.kind is DirectiveKind.STOP_AND_ABORT:
                        return {
                            "status": "BLOCKED",
                            "fault": directive_after_stop.reason,
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    if arrived_after_stop and directive_after_stop.kind is DirectiveKind.REQUEST_PHYSICAL_STOP:
                        route_stop_reason = directive_after_stop.reason
                        route_arrival_poses[active.waypoint_id] = {
                            "x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad,
                            "sample_monotonic_s": pose.received_steady_s,
                        }
                        snapshot = self.sample_for_first_motion()
                        pose = pose_provider.update(snapshot)
                        safety_provider.update(snapshot, active.waypoint_id)
                        directive_after_stop = executor.tick(
                            float(snapshot["sample_monotonic_s"]),
                            MotionFeedback(
                                status=MotionStatus.ACTIVE,
                                physical_stop_evidence=stop_evidence,
                                reset_gate_pass=True,
                                standing_gate_pass=True,
                                source_mode_handshake_pass=False,
                                safety_gates_pass=True,
                                stand_readback_pass=stand_mode_readback_confirmed(snapshot),
                            ),
                        )
                        if directive_after_stop.kind is DirectiveKind.STOP_AND_ABORT:
                            return {
                                "status": "BLOCKED",
                                "fault": directive_after_stop.reason,
                                "last_result": last_result,
                                "last_gate": last_gate,
                                "zero_plan": dict(zero_plan.__dict__),
                                "route_total_nonzero_command_time_s": route_total_nonzero_s,
                            }
                        return {
                            "status": "ARRIVAL_AFTER_VERIFIED_STOP",
                            "fault": None,
                            "arrived": True,
                            "bridge_stop_rearmed": True,
                            "stop_reason": route_stop_reason,
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "next_directive": directive_after_stop,
                            "final_snapshot": snapshot,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    if directive_after_stop.kind is not DirectiveKind.FOLLOW_WAYPOINT:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_did_not_remain_on_waypoint:{directive_after_stop.kind.value}",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    route_total_limit_s = (
                        SHORT_ROUTE_MAX_AGGREGATE_S if short_scene_route_mode
                        else float(local01_route_input["limits"]["max_aggregate_nonzero_s"])
                        if table_approach_mission_mode
                        else 15.0 if route_config_mode else 30.0
                    )
                    if route_total_nonzero_s >= route_total_limit_s:
                        return {
                            "status": "BLOCKED",
                            "fault": "route_total_nonzero_command_budget_exhausted",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    readiness = readiness_from_snapshot(snapshot)
                    current_plan = bridge.plan_next(
                        directive_after_stop,
                        pose,
                        readiness,
                        now_steady_s=float(snapshot["sample_monotonic_s"]),
                    )
                    if current_plan.status != "PROPOSAL_ONLY" or current_plan.forward_m_s <= 0.0:
                        return {
                            "status": "BLOCKED",
                            "fault": f"route_bridge_refused_next_window:{current_plan.status}:{current_plan.reason}",
                            "last_result": last_result,
                            "last_gate": last_gate,
                            "zero_plan": dict(zero_plan.__dict__),
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        }
                    current_directive = directive_after_stop

            if native_nominal_straight_gait_diagnostic_mode:
                transit_summary = run_transit_windows(
                    current_directive,
                    transit_plan,
                    "WAYPOINT",
                )
                last_result = transit_summary.get("last_result") or {}
                last_gate = transit_summary.get("last_gate") or {}
                diagnostic_pass = bool(
                    transit_summary.get("status") == "STRAIGHT_GAIT_DIAGNOSTIC_COMPLETE"
                    and last_result.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
                    and last_gate.get("status") == "PASS"
                )
                result["straight_gait_diagnostic"].update({
                    "status": "PASS" if diagnostic_pass else "FAIL",
                    "transit_summary": transit_summary,
                    "physical_motion_status": last_result.get("physical_motion_status"),
                    "physical_stop_gate": last_gate,
                    "measured_nonzero_s": transit_summary.get(
                        "route_total_nonzero_command_time_s"
                    ),
                    "acceptance_scope": (
                        "one 10 s native-timestep straight gait and physical stop; "
                        "does not qualify TABLE_APPROACH"
                    ),
                })
                result.update({
                    "status": "BLOCKED",
                    "fault": (
                        "STRAIGHT_GAIT_DIAGNOSTIC_COMPLETE_NOT_TABLE_APPROACH"
                        if diagnostic_pass else
                        transit_summary.get("fault") or "STRAIGHT_GAIT_DIAGNOSTIC_FAILED"
                    ),
                    "route_events": route_events + route_motion.events,
                    "route_total_nonzero_command_time_s": route_total_nonzero_s,
                })
                return result

            if continuous_table_approach_mode:
                transit_summary = run_transit_windows(
                    current_directive,
                    transit_plan,
                    "CONTINUOUS_TABLE_APPROACH",
                )
                last_result = transit_summary.get("last_result") or {}
                last_gate = transit_summary.get("last_gate") or {}
                if (
                    transit_summary.get("status") not in (
                        "ARRIVAL_DURING_WINDOW", "ARRIVAL_AFTER_VERIFIED_STOP",
                    )
                    or last_result.get("physical_motion_status") != "PHYSICAL_MOTION_CONFIRMED"
                    or last_gate.get("status") != "PASS"
                    or route_fault
                ):
                    result.update({
                        "status": "BLOCKED",
                        "fault": (
                            transit_summary.get("fault") or route_fault
                            or last_result.get("fault") or "continuous_table_route_or_stop_gate_failed"
                        ),
                        "route_events": route_events + route_motion.events,
                        "route_total_nonzero_command_time_s": route_total_nonzero_s,
                    })
                    return result
                evidence = stop_mapping(last_gate)
                zero_start = float(last_gate.get("zero_start_monotonic_s", 0.0))
                bridge.zero_window(
                    route_motion.latest,
                    now_steady_s=time.monotonic(),
                    reason="TABLE_APPROACH terminal stop; consume the existing composite physical stop",
                    physical_stop_gate_required=True,
                )
                bridge_rearmed = bridge.confirm_physical_stop(
                    evidence,
                    zero_request_started_s=zero_start,
                    now_steady_s=time.monotonic(),
                    expected_generation=str(self.args.attempt_id),
                )
                final_snapshot = route_sample()
                try:
                    final_pose = pose_provider.update(final_snapshot)
                except (KeyError, TypeError, ValueError) as exc:
                    result.update({
                        "status": "BLOCKED",
                        "fault": f"continuous_table_final_native_pose:{type(exc).__name__}:{exc}",
                        "route_events": route_events + route_motion.events,
                    })
                    return result
                final_active = executor.active_step
                safety_provider.update(
                    final_snapshot,
                    final_active.waypoint_id if final_active is not None else "TABLE_APPROACH",
                )
                final_fault = snapshot_fault(
                    final_snapshot,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                stand_ok = fresh_stand_mode_readback_valid(
                    final_snapshot, maximum_age_s=float(self.args.state_timeout),
                )
                source_ok = fresh_input_source_readback_valid(
                    final_snapshot,
                    expected_source=self.source,
                    maximum_age_s=float(self.args.state_timeout),
                )
                if (
                    final_active is None
                    or not bridge_rearmed
                    or final_fault
                    or not stand_ok
                    or not source_ok
                ):
                    result.update({
                        "status": "BLOCKED",
                        "fault": (
                            "continuous_table_terminal_stop_rearm_or_safety_failed:"
                            f"active={final_active is not None},rearmed={bridge_rearmed},"
                            f"stand={stand_ok},source={source_ok},fault={final_fault}"
                        ),
                        "route_events": route_events + route_motion.events,
                    })
                    return result
                stop_evidence = PhysicalStopEvidence(
                    status=PhysicalStopStatus.PASS,
                    window_start_steady_s=float(evidence["window_start_steady_s"]),
                    window_end_steady_s=float(evidence["window_end_steady_s"]),
                    sample_count=int(evidence["sample_count"]),
                    maximum_sample_gap_s=float(evidence["maximum_sample_gap_s"]),
                    maximum_state_age_s=float(evidence["maximum_state_age_s"]),
                    generation=str(self.args.attempt_id),
                )
                after = executor.accept_arrival_after_verified_stop(
                    float(final_snapshot["sample_monotonic_s"]),
                    MotionFeedback(
                        status=MotionStatus.ARRIVED,
                        physical_stop_evidence=stop_evidence,
                        reset_gate_pass=True,
                        standing_gate_pass=True,
                        source_mode_handshake_pass=False,
                        safety_gates_pass=True,
                        stand_readback_pass=stand_ok,
                        input_source_readback_pass=source_ok,
                    ),
                    zero_request_started_s=zero_start,
                    bridge_rearmed=True,
                )
                final_error_m = math.hypot(
                    final_active.x_m - final_pose.x_m,
                    final_active.y_m - final_pose.y_m,
                )
                final_yaw_error_rad = math.atan2(
                    math.sin(float(final_active.yaw_rad) - final_pose.yaw_rad),
                    math.cos(float(final_active.yaw_rad) - final_pose.yaw_rad),
                )
                rolling_events = [
                    event for event in route_events
                    if event.get("kind") == "ROLLING_WAYPOINT_PASSED"
                ]
                passed = bool(
                    after.kind is DirectiveKind.COMPLETE
                    and executor.state is RouteState.COMPLETE
                    and final_active.waypoint_id == "TABLE_APPROACH"
                    and final_error_m <= final_active.position_tolerance_m
                    and abs(final_yaw_error_rad) <= float(final_active.yaw_tolerance_rad)
                    and bool(rolling_events)
                )
                result.update({
                    "status": "PASS" if passed else "BLOCKED",
                    "classification": (
                        "SIMULATION_ONLY_CONTINUOUS_NATIVE_GROUND_TRUTH_TABLE_APPROACH_PASS"
                        if passed else "SIMULATION_ONLY_CONTINUOUS_NATIVE_GROUND_TRUTH_TABLE_APPROACH_BLOCKED"
                    ),
                    "continuous_table_approach" : {
                        **result["continuous_table_approach"],
                        "status": "PASS" if passed else "FAIL",
                        "rolling_waypoint_events": rolling_events,
                        "terminal_waypoint": final_active.waypoint_id,
                        "final_position_error_m": final_error_m,
                        "final_yaw_error_rad": final_yaw_error_rad,
                        "physical_stop_gate": last_gate.get("status"),
                    },
                    "route_events": route_events + route_motion.events,
                    "route_total_nonzero_command_time_s": route_total_nonzero_s,
                    "route_total_windows": 1,
                    "route_window_limit": 1,
                    "aggregate_nonzero_limit_s": TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S,
                    "final_composite_physical_stop": last_gate.get("status"),
                    "route_manager_state": {
                        "status": executor.state.value,
                        "active_step": executor.active_step.waypoint_id if executor.active_step else None,
                        "phase": executor.phase.value,
                        "steps_completed": executor.step_index,
                    },
                    "manipulator_handoff": "NOT_REQUESTED",
                })
                if not passed:
                    result["fault"] = "continuous_table_approach_terminal_acceptance_failed"
                return result

            if route_config_mode:
                route_step_results: list[dict[str, Any]] = []
                current_directive = current_directive
                current_plan = transit_plan
                for step_index, configured_step in enumerate(route.steps):
                    active = executor.active_step
                    if active is None or active.waypoint_id != configured_step.waypoint_id:
                        result.update({
                            "status": "BLOCKED",
                            "fault": "configured_route_step_order_changed",
                            "route_events": route_events + route_motion.events,
                            "route_step_results": route_step_results,
                        })
                        return result
                    if step_index > 0:
                        lifecycle, next_set_action, activation_gate = self.begin_source_grounded_route_activation(
                            registration=registration,
                            phase=f"configured-route-{configured_step.waypoint_id.lower()}-activation",
                            forward_command_m_s=float(self.args.forward_speed),
                            allow_raicom_low_cruise=bool(
                                getattr(self.args, "raicom_low_cruise_profile", False)
                            ),
                            zero_forward_heading_acquisition=(
                                fixed_window_heading_acquisition_required(
                                    fixed_window_table_approach=fixed_window_table_approach_mode,
                                    segment_id=configured_step.waypoint_id,
                                )
                            ),
                        )
                        result.setdefault("activation_rearms", []).append({
                            "step_id": configured_step.waypoint_id,
                            "set_mc_action_ld": next_set_action,
                            "activation_gate": activation_gate,
                            "lifecycle_after_activation": lifecycle.snapshot(),
                        })
                        if activation_gate.get("status") != "READY":
                            recovery = self.recover_activation_to_stand(
                                lifecycle=lifecycle,
                                registration=registration,
                                set_action=next_set_action,
                                baseline=stationary_baseline,
                                reason=str(activation_gate.get("fault") or "route step activation was not confirmed"),
                                phase=f"configured-route-{configured_step.waypoint_id.lower()}-activation-recovery",
                            )
                            result["activation_rearms"][-1]["stand_recovery"] = recovery
                            result.update({
                                "status": "BLOCKED",
                                "fault": "configured_route_step_locomotion_reactivation_not_confirmed",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        self.activation_lifecycle = lifecycle
                        set_action = next_set_action
                        fresh = self.sample_for_first_motion()
                        try:
                            pose = pose_provider.update(fresh)
                        except (KeyError, TypeError, ValueError) as exc:
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_rearm_pose:{type(exc).__name__}:{exc}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        safety_provider.update(fresh, configured_step.waypoint_id)
                        fault = snapshot_fault(
                            fresh,
                            self.joint_limits,
                            state_timeout_s=self.args.state_timeout,
                            min_root_z=self.args.min_root_z,
                            max_tilt_deg=self.args.max_tilt_deg,
                            max_observed_planar_speed_m_s=0.2,
                        )
                        if fault or not fresh_stand_mode_readback_valid(
                            fresh, maximum_age_s=float(self.args.state_timeout),
                        ):
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_rearm_safety_or_stand_readback:{fault or 'STALE_STAND'}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        route_segment_origin["x_m"] = pose.x_m
                        route_segment_origin["y_m"] = pose.y_m
                        readiness = readiness_from_snapshot(fresh)
                        current_plan = bridge.plan_next(
                            current_directive,
                            pose,
                            readiness,
                            now_steady_s=float(fresh["sample_monotonic_s"]),
                        )
                        if current_plan.status != "PROPOSAL_ONLY" or current_plan.forward_m_s <= 0.0:
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_bridge_refused_step:{current_plan.status}:{current_plan.reason}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result

                    result["active_configured_route_step"] = configured_step.waypoint_id
                    transit_summary = run_transit_windows(
                        current_directive,
                        current_plan,
                        configured_step.waypoint_id,
                    )
                    last_result = transit_summary.get("last_result") or {}
                    last_gate = transit_summary.get("last_gate") or {}
                    if (
                        transit_summary.get("status") not in (
                            "ARRIVAL_DURING_WINDOW", "ARRIVAL_AFTER_VERIFIED_STOP",
                        )
                        or last_result.get("physical_motion_status") != "PHYSICAL_MOTION_CONFIRMED"
                        or last_gate.get("status") != "PASS"
                        or route_fault
                    ):
                        result.update({
                            "status": "BLOCKED",
                            "fault": transit_summary.get("fault") or route_fault or last_result.get("fault") or "configured_route_step_or_stop_gate_failed",
                            "route_events": route_events + route_motion.events,
                            "route_step_results": route_step_results,
                            "route_total_nonzero_command_time_s": route_total_nonzero_s,
                        })
                        return result

                    after = transit_summary.get("next_directive")
                    final_snapshot = transit_summary.get("final_snapshot")
                    stop_rearmed = bool(transit_summary.get("bridge_stop_rearmed"))
                    evidence = stop_mapping(last_gate)
                    zero_start = float(last_gate.get("zero_start_monotonic_s", 0.0))
                    if not stop_rearmed:
                        bridge.zero_window(
                            route_motion.latest,
                            now_steady_s=time.monotonic(),
                            reason=f"{configured_step.waypoint_id} arrived; consume its existing composite stop",
                            physical_stop_gate_required=True,
                        )
                        stop_rearmed = bridge.confirm_physical_stop(
                            evidence,
                            zero_request_started_s=zero_start,
                            now_steady_s=time.monotonic(),
                            expected_generation=str(self.args.attempt_id),
                        )
                        if not stop_rearmed:
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_step_stop_rearm_failed:{configured_step.waypoint_id}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        final_snapshot = self.sample_for_first_motion()
                        try:
                            pose = pose_provider.update(final_snapshot)
                        except (KeyError, TypeError, ValueError) as exc:
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_poststop_pose:{type(exc).__name__}:{exc}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        safety_provider.update(final_snapshot, configured_step.waypoint_id)
                        fault = snapshot_fault(
                            final_snapshot,
                            self.joint_limits,
                            state_timeout_s=self.args.state_timeout,
                            min_root_z=self.args.min_root_z,
                            max_tilt_deg=self.args.max_tilt_deg,
                            max_observed_planar_speed_m_s=0.2,
                        )
                        stand_ok = fresh_stand_mode_readback_valid(
                            final_snapshot, maximum_age_s=float(self.args.state_timeout),
                        )
                        source_ok = fresh_input_source_readback_valid(
                            final_snapshot,
                            expected_source=self.source,
                            maximum_age_s=float(self.args.state_timeout),
                        )
                        if fault or not stand_ok or not source_ok:
                            result.update({
                                "status": "BLOCKED",
                                "fault": f"configured_route_poststop_readback_or_safety:stand={stand_ok},source={source_ok},fault={fault}",
                                "route_events": route_events + route_motion.events,
                                "route_step_results": route_step_results,
                            })
                            return result
                        stop_evidence = PhysicalStopEvidence(
                            status=PhysicalStopStatus.PASS,
                            window_start_steady_s=float(evidence["window_start_steady_s"]),
                            window_end_steady_s=float(evidence["window_end_steady_s"]),
                            sample_count=int(evidence["sample_count"]),
                            maximum_sample_gap_s=float(evidence["maximum_sample_gap_s"]),
                            maximum_state_age_s=float(evidence["maximum_state_age_s"]),
                            generation=str(self.args.attempt_id),
                        )
                        after = executor.accept_arrival_after_verified_stop(
                            float(final_snapshot["sample_monotonic_s"]),
                            MotionFeedback(
                                status=MotionStatus.ARRIVED,
                                physical_stop_evidence=stop_evidence,
                                reset_gate_pass=True,
                                standing_gate_pass=True,
                                source_mode_handshake_pass=False,
                                safety_gates_pass=True,
                                stand_readback_pass=stand_ok,
                                input_source_readback_pass=source_ok,
                            ),
                            zero_request_started_s=zero_start,
                            bridge_rearmed=True,
                        )
                    if after is None or after.kind is DirectiveKind.STOP_AND_ABORT:
                        result.update({
                            "status": "BLOCKED",
                            "fault": getattr(after, "reason", "configured_route_executor_did_not_advance"),
                            "route_events": route_events + route_motion.events,
                            "route_step_results": route_step_results,
                        })
                        return result
                    pose = pose_provider.pose
                    if pose is None:
                        result.update({
                            "status": "BLOCKED",
                            "fault": "configured_route_poststop_pose_missing",
                            "route_events": route_events + route_motion.events,
                            "route_step_results": route_step_results,
                        })
                        return result
                    position_error = math.hypot(configured_step.x_m - pose.x_m, configured_step.y_m - pose.y_m)
                    yaw_error = None
                    if configured_step.yaw_policy is YawPolicy.EXPLICIT:
                        yaw_error = math.atan2(
                            math.sin(float(configured_step.yaw_rad) - pose.yaw_rad),
                            math.cos(float(configured_step.yaw_rad) - pose.yaw_rad),
                        )
                    route_step_results.append({
                        "step_id": configured_step.waypoint_id,
                        "measured_pose": {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
                        "position_error_m": position_error,
                        "position_tolerance_m": configured_step.position_tolerance_m,
                        "yaw_error_rad": yaw_error,
                        "yaw_tolerance_rad": configured_step.yaw_tolerance_rad,
                        "physical_stop_gate": last_gate.get("status"),
                        "zero_receipt_ack": last_gate.get("mc_zero_receipt_ack", False),
                        "pose_goal_terminal_decision": (
                            configured_docking_control.get("last_decision")
                            if configured_step.profile is Profile.DOCKING else None
                        ),
                        "route_executor_state": executor.state.value,
                        "no_duplicate_stop_request": True,
                    })
                    if position_error > configured_step.position_tolerance_m or (
                        yaw_error is not None and abs(yaw_error) > float(configured_step.yaw_tolerance_rad)
                    ):
                        result.update({
                            "status": "BLOCKED",
                            "fault": f"configured_route_settled_pose_outside_tolerance:{configured_step.waypoint_id}",
                            "route_events": route_events + route_motion.events,
                            "route_step_results": route_step_results,
                        })
                        return result
                    route_segment_origin["x_m"] = pose.x_m
                    route_segment_origin["y_m"] = pose.y_m
                    current_directive = after

                final_route_pass = (
                    len(route_step_results) == 2
                    and all(step["physical_stop_gate"] == "PASS" for step in route_step_results)
                    and (
                        (
                            table_approach_mission_mode
                            and executor.state is RouteState.COMPLETE
                            and current_directive.kind is DirectiveKind.COMPLETE
                        )
                        or
                        (
                            short_scene_route_mode
                            and executor.state is RouteState.COMPLETE
                            and current_directive.kind is DirectiveKind.COMPLETE
                        )
                        or (
                            not short_scene_route_mode
                            and executor.state is RouteState.WAITING_FOR_DOCKING
                            and current_directive.kind is DirectiveKind.WAIT
                        )
                    )
                )
                result.update({
                    "status": "PASS" if final_route_pass else "BLOCKED",
                    "classification": (
                        "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION_PASS"
                        if final_route_pass and aligned_start_table_approach_mode else
                        "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION_BLOCKED"
                        if aligned_start_table_approach_mode else
                        "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION_PASS"
                        if final_route_pass and table_approach_mission_mode else
                        "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION_BLOCKED"
                        if table_approach_mission_mode else
                        "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_DIAGNOSTIC_PASS"
                        if final_route_pass and short_scene_route_mode else
                        "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_DIAGNOSTIC_BLOCKED"
                        if short_scene_route_mode else
                        "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_PASS"
                        if final_route_pass else
                        "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_BLOCKED"
                    ),
                    "acceptance_scope": (
                        "DERIVED_ALIGNED_START_GROUND_TRUTH_SAFE_PREFIX_TO_TABLE_APPROACH_WITH_PHYSICAL_STOP; ORIGINAL_START_ALIGNMENT_TABLE_DOCKING_AND_MANIPULATION_NOT_RUN"
                        if aligned_start_table_approach_mode else
                        "GROUND_TRUTH_SAFE_PREFIX_TO_TABLE_APPROACH_WITH_PHYSICAL_STOP; TABLE_DOCKING_AND_MANIPULATION_NOT_RUN"
                        if table_approach_mission_mode else
                        "LOCAL01_SCENE_START_HEADING_SHORT_FORWARD_ROUTE_NOT_TABLE_DOCKING"
                        if short_scene_route_mode else "GROUND_TRUTH_WAYPOINT_AND_DOCKING_POSE_WITH_PHYSICAL_STOP"
                    ),
                    "route_lifecycle_completion": (
                        "DERIVED_ALIGNED_START_TABLE_APPROACH_STOPPED_NO_ORIGINAL_START_QUALIFICATION_OR_MANIPULATOR_HANDOFF"
                        if final_route_pass and aligned_start_table_approach_mode else
                        "TABLE_APPROACH_STOPPED_NO_TABLE_DOCKING_OR_MANIPULATOR_HANDOFF"
                        if final_route_pass and table_approach_mission_mode else
                        "SHORT_ROUTE_STOPPED_NO_TABLE_OR_MANIPULATOR_HANDOFF"
                        if final_route_pass and short_scene_route_mode else
                        "WAITING_FOR_DOCKING_NO_MANIPULATOR_HANDOFF"
                        if final_route_pass else executor.state.value
                    ),
                    "route_manager_state": {
                        "status": executor.state.value,
                        "active_step": executor.active_step.waypoint_id if executor.active_step else None,
                        "phase": executor.phase.value,
                        "steps_completed": executor.step_index,
                    },
                    "route_step_results": route_step_results,
                    "route_events": route_events + route_motion.events,
                    "route_total_nonzero_command_time_s": route_total_nonzero_s,
                    "route_total_windows": len(result["segments"]),
                    "route_window_limit": (
                        SHORT_ROUTE_MAX_WINDOWS if short_scene_route_mode else
                        int(local01_route_input["limits"]["max_windows"])
                        if table_approach_mission_mode else 3
                    ),
                    "aggregate_nonzero_limit_s": (
                        SHORT_ROUTE_MAX_AGGREGATE_S if short_scene_route_mode else
                        float(local01_route_input["limits"]["max_aggregate_nonzero_s"])
                        if table_approach_mission_mode else 15.0
                    ),
                    "cross_track_limit_m": (
                        route_progress_monitor["last_sample"].get("cross_track_limit_m")
                            if segmented_table_envelope_mode and isinstance(
                            route_progress_monitor.get("last_sample"), Mapping
                        ) else WAYPOINT_B_MAX_CROSS_TRACK_M
                    ),
                    "cross_track_limit_by_step_m": (
                        {
                            "WAYPOINT": CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
                            "TABLE_APPROACH": CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
                        }
                        if segmented_table_envelope_mode else None
                    ),
                    "maximum_cross_track_m": route_progress_monitor["maximum_abs_cross_track_m"],
                    "manipulator_handoff": "NOT_REQUESTED",
                    "mc_zero_receipt_ack_available": False,
                })
                if not final_route_pass:
                    result["fault"] = (
                        "table_approach_mission_did_not_stop_at_TABLE_APPROACH"
                        if table_approach_mission_mode else
                        "configured_local01_short_route_did_not_stop_at_SHORT_ROUTE_END"
                        if short_scene_route_mode else "configured_route_did_not_reach_stopped_TABLE_DOCKING"
                    )
                return result

            baseline_displacements = [
                float(window["forward_displacement_m"])
                for window in (stationary_baseline or {}).get("windows", [])
            ]
            transit_summary = run_transit_windows(
                current_directive, transit_plan, "TRANSIT-TO-CORNER"
            )
            transit = transit_summary.get("last_result") or {}
            transit_gate = transit_summary.get("last_gate") or {}
            result["route_events"] = route_events
            if (
                transit_summary.get("status") not in (
                    "ARRIVAL_DURING_WINDOW", "ARRIVAL_AFTER_VERIFIED_STOP",
                )
                or route_fault
                or route_stop_reason is None
                or transit.get("physical_motion_status") != "PHYSICAL_MOTION_CONFIRMED"
                or transit_gate.get("status") != "PASS"
            ):
                result["status"] = "BLOCKED"
                result["fault"] = (
                    transit_summary.get("fault") or route_fault or route_stop_reason
                    or transit.get("fault") or "transit_waypoint_or_physical_stop_not_passed"
                )
                result["route_total_nonzero_command_time_s"] = route_total_nonzero_s
                return result

            transit_evidence = stop_mapping(transit_gate)
            transit_zero_start = float(transit_gate.get("zero_start_monotonic_s", 0.0))
            zero_plan = transit_summary.get("zero_plan")
            transit_stop_rearmed = bool(transit_summary.get("bridge_stop_rearmed"))
            after_transit = transit_summary.get("next_directive")
            route_after_transit = transit_summary.get("final_snapshot")
            if not transit_stop_rearmed:
                zero_plan = bridge.zero_window(
                    route_motion.latest,
                    now_steady_s=time.monotonic(),
                    reason="waypoint B reached; revalidate measured stop before the corner",
                    physical_stop_gate_required=True,
                )
                transit_stop_rearmed = bridge.confirm_physical_stop(
                    transit_evidence,
                    zero_request_started_s=transit_zero_start,
                    now_steady_s=time.monotonic(),
                    expected_generation=str(self.args.attempt_id),
                )
                if not transit_stop_rearmed:
                    result["status"] = "BLOCKED"
                    result["fault"] = "waypoint_b_physical_stop_did_not_rearm_bridge"
                    return result
                route_after_transit = self.sample_for_first_motion()
                pose_provider.update(route_after_transit)
                active = executor.active_step
                if active is None:
                    result["status"] = "BLOCKED"
                    result["fault"] = "waypoint_b_route_step_missing_after_stop"
                    return result
                safety_provider.update(route_after_transit, active.waypoint_id)
                route_after_fault = snapshot_fault(
                    route_after_transit,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                route_after_stand_passed = fresh_stand_mode_readback_valid(
                    route_after_transit, maximum_age_s=float(self.args.state_timeout),
                )
                route_after_source_passed = fresh_input_source_readback_valid(
                    route_after_transit,
                    expected_source=self.source,
                    maximum_age_s=float(self.args.state_timeout),
                )
                stop_evidence_object = PhysicalStopEvidence(
                    status=PhysicalStopStatus.PASS,
                    window_start_steady_s=float(transit_evidence["window_start_steady_s"]),
                    window_end_steady_s=float(transit_evidence["window_end_steady_s"]),
                    sample_count=int(transit_evidence["sample_count"]),
                    maximum_sample_gap_s=float(transit_evidence["maximum_sample_gap_s"]),
                    maximum_state_age_s=float(transit_evidence["maximum_state_age_s"]),
                    generation=str(self.args.attempt_id),
                )
                after_transit = executor.accept_arrival_after_verified_stop(
                    float(route_after_transit["sample_monotonic_s"]),
                    MotionFeedback(
                        status=MotionStatus.ARRIVED,
                        physical_stop_evidence=stop_evidence_object,
                        reset_gate_pass=True,
                        standing_gate_pass=True,
                        source_mode_handshake_pass=False,
                        safety_gates_pass=route_after_fault is None,
                        stand_readback_pass=route_after_stand_passed,
                        input_source_readback_pass=route_after_source_passed,
                    ),
                    zero_request_started_s=transit_zero_start,
                    bridge_rearmed=transit_stop_rearmed,
                )
                if after_transit.kind is DirectiveKind.STOP_AND_ABORT:
                    result["status"] = "BLOCKED"
                    result["fault"] = f"route rejected post-stop arrival: {after_transit.reason}"
                    return result

            if waypoint_b_only:
                after_verified_stop_completion = (
                    executor.state is RouteState.COMPLETE
                    and transit_summary.get("completion_evidence")
                    == "M1_COMPOSITE_STOP_GATE_CONSUMED_BY_ROUTE_EXECUTOR"
                )
                if (
                    (after_transit is None or after_transit.kind is not DirectiveKind.COMPLETE)
                    and not after_verified_stop_completion
                ):
                    result["status"] = "BLOCKED"
                    result["fault"] = (
                        "curved_path_route_did_not_complete_after_physical_stop"
                        if curved_path_diagnostic else
                        "waypoint_b_only_route_did_not_complete_after_physical_stop"
                    )
                    result["route_events"] = route_events + route_motion.events
                    result["route_total_nonzero_command_time_s"] = route_total_nonzero_s
                    return result
                result["route_lifecycle_completion"] = (
                    "ROUTE_EXECUTOR_COMPLETE" if executor.state is RouteState.COMPLETE
                    else "NEXT_WAYPOINT_PENDING_AFTER_VERIFIED_STOP"
                )
                final_b_snapshot = self.sample_for_first_motion()
                try:
                    final_b_pose = pose_provider.update(final_b_snapshot)
                except (KeyError, TypeError, ValueError) as exc:
                    result["status"] = "BLOCKED"
                    result["fault"] = f"waypoint_b_poststop_pose:{type(exc).__name__}:{exc}"
                    return result
                final_b_fault = snapshot_fault(
                    final_b_snapshot,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                b_arrival = route_arrival_poses.get(route.steps[0].waypoint_id)
                b_arrival_error = (
                    math.hypot(
                        route.steps[0].x_m - b_arrival["x_m"],
                        route.steps[0].y_m - b_arrival["y_m"],
                    )
                    if b_arrival else None
                )
                b_poststop_error = math.hypot(
                    route.steps[0].x_m - final_b_pose.x_m,
                    route.steps[0].y_m - final_b_pose.y_m,
                )
                b_tolerance = route.steps[0].position_tolerance_m
                b_window_segments = [
                    segment for segment in result["segments"]
                    if str(segment.get("segment_id", "")).startswith("TRANSIT-TO-CORNER-WINDOW-")
                ]
                b_nonzero_s = sum(
                    float((segment.get("result") or {}).get("route_window_measured_nonzero_elapsed_s", 0.0))
                    for segment in b_window_segments
                )
                b_window_count_pass = (
                    1 <= len(b_window_segments) <= CURVED_PATH_MAX_WINDOWS
                    if curved_path_diagnostic else
                    waypoint_b_window_budget_passes(
                        len(b_window_segments),
                        short_diagnostic=waypoint_b_short_diagnostic,
                        extended_diagnostic=waypoint_b_extended_diagnostic,
                    )
                )
                b_arrival_pass = (
                    route_stop_reason is not None
                    and waypoint_b_arrival_passes(
                        b_arrival_error,
                        b_poststop_error,
                        tolerance_m=b_tolerance,
                    )
                )
                physical_state_gate = transit_gate.get("physical_state_gate", {})
                controller_target_gate = transit_gate.get("controller_target_stability_gate", {})
                b_physical_state_pass = (
                    transit_gate.get("status") == "PASS"
                    and isinstance(physical_state_gate, Mapping)
                    and physical_state_gate.get("status") == "PASS"
                )
                b_controller_target_pass = (
                    transit_gate.get("status") == "PASS"
                    and isinstance(controller_target_gate, Mapping)
                    and controller_target_gate.get("status") == "PASS"
                )
                final_stand_pass = fresh_stand_mode_readback_valid(
                    final_b_snapshot, maximum_age_s=float(self.args.state_timeout),
                )
                final_source_pass = fresh_input_source_readback_valid(
                    final_b_snapshot,
                    expected_source=self.source,
                    maximum_age_s=float(self.args.state_timeout),
                )
                b_stop_pass = (
                    transit_gate.get("status") == "PASS"
                    and transit_stop_rearmed
                    and not final_b_fault
                    and final_stand_pass
                    and final_source_pass
                )
                b_source_mode_pass = bool(b_window_segments) and all(
                    (segment.get("result") or {}).get("control_handshake_confirmed") is True
                    for segment in b_window_segments
                )
                b_runtime_connected = (
                    scope.get("status") == "PASS"
                    and len(publishers) == 1
                    and len(subscribers) == 1
                    and local_identity.get("is_owned_publisher") is True
                    and local_identity.get("node_topic_publisher_count") == 1
                    and bool(scope.get("verified_mc_runtime_identity"))
                    and registration.get("success") is True
                    and set_action.get("accepted_by_service") is True
                )
                b_budget_pass = (
                    b_nonzero_s <= transit_aggregate_limit_s + 1e-6
                    and route_total_nonzero_s <= 30.0 + 1e-6
                    and (
                        not curved_path_diagnostic
                        or route_total_nonzero_s <= CURVED_PATH_AGGREGATE_NONZERO_MAX_S + 1e-6
                    )
                    and all(
                        float((segment.get("result") or {}).get("route_window_measured_nonzero_elapsed_s", math.inf))
                        <= (CURVED_PATH_MAX_WINDOW_S if curved_path_diagnostic else WAYPOINT_B_WINDOW_MAX_S) + 1e-6
                        for segment in b_window_segments
                    )
                )
                b_gates = {
                    "REAL_RUNTIME_CONNECTED": "PASS" if b_runtime_connected else "FAIL",
                    "RESET_STAND": "PASS" if (
                        getattr(self.args, "reset_gate_verified", False)
                        and stand_mode_readback_confirmed(initial)
                        and not initial_fault
                    ) else "FAIL",
                    "SOURCE_MODE_HANDSHAKE": "PASS" if b_source_mode_pass else "FAIL",
                    (
                        "CURVED_PATH_WINDOW_BUDGET"
                        if curved_path_diagnostic else
                        "EXTENDED_THREE_WINDOW_DIAGNOSTIC"
                        if waypoint_b_extended_diagnostic else
                        "SHORT_DIAGNOSTIC_WINDOW_BUDGET"
                        if waypoint_b_short_diagnostic else "MULTI_WINDOW_TRANSIT"
                    ): "PASS" if b_window_count_pass else "FAIL",
                    ("CURVE_ENDPOINT_POSITION" if curved_path_diagnostic else "WAYPOINT_B_ARRIVAL"): "PASS" if b_arrival_pass else "FAIL",
                    "POST_STOP_PHYSICAL": "PASS" if b_physical_state_pass and not final_b_fault else "FAIL",
                    "POST_STOP_CONTROLLER_TARGET": "PASS" if b_controller_target_pass else "FAIL",
                    "FRESH_STAND_AND_SOURCE_READBACKS": "PASS" if final_stand_pass and final_source_pass else "FAIL",
                    "PHYSICAL_STOP_REARM": "PASS" if b_stop_pass else "FAIL",
                    **({
                        "CURVE_CROSS_TRACK": "PASS"
                        if float(route_progress_monitor["maximum_abs_cross_track_m"]) <= CURVED_PATH_MAX_CROSS_TRACK_M
                        else "FAIL"
                    } if curved_path_diagnostic else {}),
                    "TRANSIT_AND_ROUTE_BUDGETS": "PASS" if b_budget_pass else "FAIL",
                    "ROUTE_COMPLETE": "PASS" if executor.state is RouteState.COMPLETE else "FAIL",
                }
                rearm_events = [
                    event for event in route_events
                    if event.get("kind") == "WINDOW_REARM_AFTER_COMPOSITE_STOP"
                ]
                multi_window_rearm_status = (
                    "NOT_REQUIRED_SINGLE_WINDOW_ENDPOINT"
                    if len(b_window_segments) <= 1 else
                    "PASS"
                    if len(rearm_events) >= len(b_window_segments) - 1
                    and all(
                        event.get("physical_stop_gate") == "PASS"
                        and event.get("bridge_rearmed") is True
                        and event.get("stand_readback_passed") is True
                        and event.get("input_source_readback_passed") is True
                        and event.get("safety_gates_passed") is True
                        for event in rearm_events
                    ) else "FAIL"
                )
                if curved_path_diagnostic and len(b_window_segments) > 1:
                    b_gates["MULTI_WINDOW_REARM"] = "PASS" if multi_window_rearm_status == "PASS" else "FAIL"
                route_pass = all(value == "PASS" for value in b_gates.values())

                def progress_metrics(x_m: float, y_m: float, yaw_rad: float) -> dict[str, float]:
                    if curved_path_diagnostic:
                        metrics = curved_path_metrics(curved_reference, x_m=x_m, y_m=y_m)
                        return {
                            **metrics,
                            "yaw_delta_from_start_rad": math.atan2(
                                math.sin(yaw_rad - start_pose.yaw_rad),
                                math.cos(yaw_rad - start_pose.yaw_rad),
                            ),
                        }
                    dx = x_m - start_pose.x_m
                    dy = y_m - start_pose.y_m
                    return {
                        "forward_projection_from_start_m": (
                            dx * math.cos(start_pose.yaw_rad) + dy * math.sin(start_pose.yaw_rad)
                        ),
                        "cross_track_from_start_m": (
                            -dx * math.sin(start_pose.yaw_rad) + dy * math.cos(start_pose.yaw_rad)
                        ),
                        "yaw_delta_from_start_rad": math.atan2(
                            math.sin(yaw_rad - start_pose.yaw_rad),
                            math.cos(yaw_rad - start_pose.yaw_rad),
                        ),
                    }

                arrival_metrics = (
                    progress_metrics(b_arrival["x_m"], b_arrival["y_m"], b_arrival["yaw_rad"])
                    if b_arrival else None
                )
                poststop_metrics = progress_metrics(
                    final_b_pose.x_m, final_b_pose.y_m, final_b_pose.yaw_rad,
                )
                if curved_path_diagnostic:
                    route_detail = {
                        "curved_path": {
                            "reference": curved_reference,
                            "direction": curved_reference["direction"],
                            "signed_sweep_rad": int(curved_reference["turn_sign"]) * float(curved_reference["sweep_rad"]),
                            "radius_m": curved_reference["radius_m"],
                            "expected_path_length_m": curved_reference["expected_path_length_m"],
                            "target_xy_m": [route.steps[0].x_m, route.steps[0].y_m],
                            "target_yaw_rad": None,
                            "arrival_pose": b_arrival,
                            "arrival_position_error_m": b_arrival_error,
                            "arrival_path_metrics": arrival_metrics,
                            "post_stop_pose": {
                                "x_m": final_b_pose.x_m,
                                "y_m": final_b_pose.y_m,
                                "yaw_rad": final_b_pose.yaw_rad,
                                "stamp_ns": final_b_pose.stamp_ns,
                            },
                            "post_stop_position_error_m": b_poststop_error,
                            "post_stop_path_metrics": poststop_metrics,
                            "endpoint_tolerance_m": b_tolerance,
                            "maximum_cross_track_error_m": route_progress_monitor["maximum_abs_cross_track_m"],
                            "cross_track_limit_m": CURVED_PATH_MAX_CROSS_TRACK_M,
                            "maximum_abs_yaw_change_rad": route_progress_monitor["maximum_abs_yaw_delta_rad"],
                            "final_yaw_change_rad": poststop_metrics["yaw_delta_from_start_rad"],
                            "final_yaw_target_status": "NOT_SPECIFIED_REPORT_ONLY",
                        },
                        "window_count_minimum": 1,
                        "window_count_maximum": CURVED_PATH_MAX_WINDOWS,
                        "multi_window_rearm": multi_window_rearm_status,
                        "multi_window_rearm_events": rearm_events,
                        "aggregate_nonzero_limit_s": CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
                        "per_window_nonzero_limit_s": CURVED_PATH_MAX_WINDOW_S,
                    }
                else:
                    route_detail = {
                        "waypoint_b_transit_nonzero_command_time_s": b_nonzero_s,
                        "waypoint_b_transit_nonzero_command_time_limit_s": transit_aggregate_limit_s,
                        "waypoint_b_transit_maximum_windows": transit_window_limit,
                        "waypoint_b_arrival": {
                            "waypoint_id": route.steps[0].waypoint_id,
                            "diagnostic_only": (
                                waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic
                            ),
                            "target_distance_from_start_m": math.hypot(
                                route.steps[0].x_m - route.start_x_m,
                                route.steps[0].y_m - route.start_y_m,
                            ),
                            "target_x_m": route.steps[0].x_m,
                            "target_y_m": route.steps[0].y_m,
                            "arrival_pose": b_arrival,
                            "arrival_error_m": b_arrival_error,
                            "arrival_progress": arrival_metrics,
                            "poststop_pose": {
                                "x_m": final_b_pose.x_m,
                                "y_m": final_b_pose.y_m,
                                "yaw_rad": final_b_pose.yaw_rad,
                                "stamp_ns": final_b_pose.stamp_ns,
                            },
                            "poststop_error_m": b_poststop_error,
                            "poststop_progress": poststop_metrics,
                            "arrival_must_remain_in_tolerance_after_stop": True,
                            "tolerance_m": b_tolerance,
                            "arrival_trigger_radius_m": waypoint_b_arrival_trigger_radius_m(
                                acceptance_tolerance_m=b_tolerance,
                                stop_reserve_m=(
                                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                                    if waypoint_b_extended_diagnostic else
                                    WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                                    if waypoint_b_short_diagnostic else 0.0
                                ),
                            ),
                            "stop_reserve_m": (
                                WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                                if waypoint_b_extended_diagnostic else
                                WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                                if waypoint_b_short_diagnostic else 0.0
                            ),
                        },
                    }
                result.update({
                    "status": "PASS" if route_pass else "BLOCKED",
                    "classification": (
                        "SIMULATION_ONLY_CURVED_PATH_ROUTE_LIFECYCLE_PASS"
                        if route_pass and curved_path_diagnostic else
                        "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC_BLOCKED"
                        if curved_path_diagnostic else
                        "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC_PASS"
                        if route_pass and waypoint_b_extended_diagnostic else
                        "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC_BLOCKED"
                        if waypoint_b_extended_diagnostic else
                        "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_PASS"
                        if route_pass and waypoint_b_short_diagnostic else
                        "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_BLOCKED"
                        if waypoint_b_short_diagnostic else
                        "SIMULATION_ONLY_WAYPOINT_B_AND_PHYSICAL_STOP_PASS"
                        if route_pass else "SIMULATION_ONLY_WAYPOINT_B_BLOCKED"
                    ),
                    "acceptance_scope": (
                        "CLOSED_LOOP_CURVED_PATH_FIXED_POINT_AND_STOP"
                        if curved_path_diagnostic else
                        "EXTENDED_DIAGNOSTIC_ORIGINAL_0_40_M_WAYPOINT_B"
                        if waypoint_b_extended_diagnostic else
                        "SHORT_DIAGNOSTIC_ONLY_NOT_ORIGINAL_0_40_M_WAYPOINT_B_ACCEPTANCE"
                        if waypoint_b_short_diagnostic else "ORIGINAL_0_40_M_WAYPOINT_B"
                    ),
                    "route_gates": b_gates,
                    "route_executor_state": executor.state.value,
                    "route_lifecycle_completion": result.get("route_lifecycle_completion"),
                    "route_events": route_events + route_motion.events,
                    "route_total_nonzero_command_time_s": route_total_nonzero_s,
                    "window_count": len(b_window_segments),
                    "window_count_minimum": route_detail["window_count_minimum"] if curved_path_diagnostic else (
                        1 if waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic else 2
                    ),
                    "window_count_maximum": transit_window_limit,
                    "physical_stop_gate": transit_gate,
                    "limitation": (
                        "This is a bounded SIMULATION_ONLY curved fixed-point route. It does not validate "
                        "general navigation, Nav2, LiDAR, obstacle avoidance, or hardware."
                        if curved_path_diagnostic else
                        "This run ends at waypoint B. No right turn, waypoint C, larger route, "
                        "Nav2, LiDAR, or hardware navigation is validated."
                    ),
                    **route_detail,
                })
                if not route_pass:
                    result["fault"] = (
                        f"waypoint_b_gates_failed:"
                        f"{','.join(name for name,value in b_gates.items() if value != 'PASS')}"
                    )
                return result

            route_after_transit = self.sample_for_first_motion()
            route_pose = pose_provider.update(route_after_transit)
            active = executor.active_step
            if active is None:
                result["status"] = "BLOCKED"
                result["fault"] = "route_corner_step_missing_after_waypoint_b"
                return result
            safety_provider.update(route_after_transit, active.waypoint_id)
            route_after_transit_fault = snapshot_fault(
                route_after_transit,
                self.joint_limits,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=0.2,
            )
            if route_after_transit_fault:
                result["status"] = "BLOCKED"
                result["fault"] = f"route_corner_preflight:{route_after_transit_fault}"
                return result
            stop_feedback = MotionFeedback(
                status=MotionStatus.ACTIVE,
                physical_stop_evidence=PhysicalStopEvidence(
                    status=PhysicalStopStatus.PASS,
                    window_start_steady_s=float(transit_evidence["window_start_steady_s"]),
                    window_end_steady_s=float(transit_evidence["window_end_steady_s"]),
                    sample_count=int(transit_evidence["sample_count"]),
                    maximum_sample_gap_s=float(transit_evidence["maximum_sample_gap_s"]),
                    maximum_state_age_s=float(transit_evidence["maximum_state_age_s"]),
                    generation=str(self.args.attempt_id),
                ),
                reset_gate_pass=True,
                standing_gate_pass=True,
                source_mode_handshake_pass=False,
                safety_gates_pass=True,
                stand_readback_pass=stand_mode_readback_confirmed(route_after_transit),
            )
            to_turn = executor.tick(
                float(route_after_transit["sample_monotonic_s"]),
                MotionFeedback(
                    status=MotionStatus.ARRIVED,
                    physical_stop_evidence=stop_feedback.physical_stop_evidence,
                    reset_gate_pass=True,
                    standing_gate_pass=True,
                    source_mode_handshake_pass=False,
                    safety_gates_pass=True,
                    stand_readback_pass=stand_mode_readback_confirmed(route_after_transit),
                ),
            )
            if not transit_stop_rearmed or to_turn.kind is not DirectiveKind.BOUNDED_MOVING_TURN:
                result["status"] = "BLOCKED"
                result["fault"] = "route_did_not_rearm_on_measured_stop_or_emit_bounded_turn"
                result["bridge_zero_rearm"] = {"plan": dict(zero_plan.__dict__), "passed": transit_stop_rearmed}
                return result
            result["bridge_zero_rearm"] = {"plan": dict(zero_plan.__dict__), "passed": True}

            turn_preflight = route_after_transit
            turn_readiness = readiness_from_snapshot(turn_preflight, measured_turn_guard=True)
            turn_plan = bridge.plan_next(
                to_turn,
                pose_provider.pose,
                turn_readiness,
                now_steady_s=float(turn_preflight["sample_monotonic_s"]),
            )
            if turn_plan.status != "PROPOSAL_ONLY":
                result["status"] = "BLOCKED"
                result["fault"] = f"route_bridge_refused_turn:{turn_plan.status}:{turn_plan.reason}"
                result["route_events"] = route_events
                return result
            route_turn_stop_reason: str | None = None
            route_turn_origin = pose_provider.pose

            def turn_termination(snapshot: Mapping[str, Any]) -> str | None:
                nonlocal route_fault, route_turn_stop_reason
                fault = snapshot_fault(
                    snapshot,
                    self.joint_limits,
                    state_timeout_s=self.args.state_timeout,
                    min_root_z=self.args.min_root_z,
                    max_tilt_deg=self.args.max_tilt_deg,
                    max_observed_planar_speed_m_s=0.2,
                )
                if fault:
                    route_fault = "turn_safety:" + fault
                    return route_fault
                try:
                    pose_provider.update(snapshot)
                except (KeyError, TypeError, ValueError) as exc:
                    route_fault = f"turn_pose:{type(exc).__name__}:{exc}"
                    return route_fault
                pose = pose_provider.pose
                if route_turn_origin is not None and pose is not None:
                    measured_from_route_origin = math.hypot(
                        pose.x_m - route_turn_origin.x_m,
                        pose.y_m - route_turn_origin.y_m,
                    )
                    if measured_from_route_origin >= 0.20:
                        return "measured_route_turn_stop_reserve_trigger"
                if not same_sample_handshake(snapshot):
                    return None
                active = executor.active_step
                if active is None:
                    route_fault = "route_corner_missing_active_step"
                    return route_fault
                safety_provider.update(snapshot, active.waypoint_id)
                assert pose is not None
                yaw_error = math.atan2(
                    math.sin(float(to_turn.goal_yaw_rad) - pose.yaw_rad),
                    math.cos(float(to_turn.goal_yaw_rad) - pose.yaw_rad),
                )
                feedback = MotionFeedback(
                    status=MotionStatus.ARRIVED if abs(yaw_error) <= float(to_turn.yaw_tolerance_rad) else MotionStatus.ACTIVE,
                    reset_gate_pass=True,
                    standing_gate_pass=True,
                    source_mode_handshake_pass=True,
                    safety_gates_pass=True,
                )
                directive = executor.tick(float(snapshot["sample_monotonic_s"]), feedback)
                if directive.kind is DirectiveKind.REQUEST_PHYSICAL_STOP:
                    route_turn_stop_reason = directive.reason
                    route_events.append({"kind": directive.kind.value, "reason": directive.reason,
                                         "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
                                         "pose": {"x_m": pose.x_m, "y_m": pose.y_m,
                                                  "yaw_rad": pose.yaw_rad}})
                    return "ROUTE_CORNER_YAW_REACHED"
                if directive.kind is DirectiveKind.STOP_AND_ABORT:
                    route_fault = directive.reason
                    return route_fault
                return None

            turn = run_angular_turn_window(
                sample=self.sample_for_first_motion,
                publish_turn=lambda forward, angular: self._publish(forward, 0.0, angular),
                publish_zero=lambda: self._publish(0.0, 0.0, 0.0),
                registration=registration,
                set_action=set_action,
                joint_limits=self.joint_limits,
                baseline_forward_displacements_m=baseline_displacements,
                baseline_state_summary=(stationary_baseline or {}).get("physical_state_summary"),
                authorized=True,
                angular_velocity_rad_s=turn_plan.angular_rad_s,
                forward_velocity_m_s=turn_plan.forward_m_s,
                forward_warmup_s=TURN_FORWARD_WARMUP_S,
                pulse_duration_s=min(TURN_PULSE_S, turn_plan.maximum_duration_s - TURN_FORWARD_WARMUP_S),
                pre_deadline_zero_margin_s=0.02,
                max_observed_planar_speed_m_s=0.2,
                zero_until_stand_timeout_s=3.5,
                extra_post_stop_observe_s=5.0,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_turn_displacement_m=0.25,
                turn_stop_reserve_m=0.05,
                termination_condition=turn_termination,
                shutdown_event=self.interrupted,
                on_sample=lambda phase, snapshot: self._trace_first_motion_sample(
                    f"route-right-corner-{phase}", snapshot,
                ),
            )
            turn["route_plan"] = dict(turn_plan.__dict__)
            turn["route_stop_reason"] = route_turn_stop_reason
            result["segments"].append({"segment_id": "RIGHT-CORNER", "plan": dict(turn_plan.__dict__), "result": turn})
            try:
                actual_turn_window_s = measured_m1_nonzero_window_s(turn)
            except RouteMotionBridgeError as exc:
                result["status"] = "BLOCKED"
                result["fault"] = f"turn_window_elapsed_unavailable:{exc}"
                return result
            turn["route_window_measured_nonzero_elapsed_s"] = actual_turn_window_s
            if route_total_nonzero_s + actual_turn_window_s > 30.0:
                result["status"] = "BLOCKED"
                result["fault"] = "route_total_nonzero_command_budget_exceeded_by_turn"
                result["total_actual_nonzero_command_time_s"] = route_total_nonzero_s
                return result
            route_total_nonzero_s += actual_turn_window_s
            bridge.complete_window(turn_plan.window_id, actual_elapsed_s=actual_turn_window_s)
            turn_gate = turn.get("physical_stop_gate") or {}
            turn_rotation = turn.get("rotation") or {}
            turn_guard = turn.get("turn_displacement_guard") or {}
            route_turn_yaw_pass = (
                turn_rotation.get("status") == "TURN_DIRECTION_PASS"
                and float(turn_rotation.get("signed_yaw_delta_rad", 0.0)) <= -TURN_MIN_HEADING_CHANGE_RAD
            )
            route_turn_displacement_pass = (
                turn_guard.get("maximum_total_displacement_including_stop_m") is not None
                and float(turn_guard["maximum_total_displacement_including_stop_m"]) <= 0.25
            )
            if (
                route_fault
                or route_turn_stop_reason is None
                or turn_gate.get("status") != "PASS"
                or not turn.get("control_handshake_confirmed")
                or not route_turn_yaw_pass
                or not route_turn_displacement_pass
                or turn.get("fault")
                or turn.get("post_stop_fault")
            ):
                result["status"] = "BLOCKED"
                result["fault"] = route_fault or route_turn_stop_reason or turn.get("fault") or "route_corner_yaw_displacement_or_physical_stop_not_passed"
                result["route_events"] = route_events
                result["route_turn_gates"] = {
                    "route_yaw_target_reached": route_turn_stop_reason == "ROUTE_CORNER_YAW_REACHED",
                    "rightward_yaw_at_least_0p087_rad": route_turn_yaw_pass,
                    "total_displacement_le_0p25m_including_stop": route_turn_displacement_pass,
                    "physical_stop_gate": turn_gate.get("status"),
                    "control_handshake": turn.get("control_handshake_confirmed"),
                    "turn_helper_fault": turn.get("fault"),
                    "post_stop_fault": turn.get("post_stop_fault"),
                }
                return result

            turn_evidence = stop_mapping(turn_gate)
            turn_zero_start = float(turn_gate.get("zero_start_monotonic_s", 0.0))
            turn_zero_plan = bridge.zero_window(
                route_motion.latest,
                now_steady_s=time.monotonic(),
                reason="route corner target was reached; measured physical-stop gate is being revalidated",
                physical_stop_gate_required=True,
            )
            turn_stop_rearmed = bridge.confirm_physical_stop(
                turn_evidence,
                zero_request_started_s=turn_zero_start,
                now_steady_s=time.monotonic(),
                expected_generation=str(self.args.attempt_id),
            )
            final_snapshot = self.sample_for_first_motion()
            pose_provider.update(final_snapshot)
            safety_provider.update(final_snapshot, executor.active_step.waypoint_id)
            final_evidence = PhysicalStopEvidence(
                status=PhysicalStopStatus.PASS,
                window_start_steady_s=float(turn_evidence["window_start_steady_s"]),
                window_end_steady_s=float(turn_evidence["window_end_steady_s"]),
                sample_count=int(turn_evidence["sample_count"]),
                maximum_sample_gap_s=float(turn_evidence["maximum_sample_gap_s"]),
                maximum_state_age_s=float(turn_evidence["maximum_state_age_s"]),
                generation=str(self.args.attempt_id),
            )
            next_segment = executor.tick(
                float(final_snapshot["sample_monotonic_s"]),
                MotionFeedback(
                    status=MotionStatus.ACTIVE,
                    physical_stop_evidence=final_evidence,
                    reset_gate_pass=True,
                    standing_gate_pass=True,
                    source_mode_handshake_pass=False,
                    safety_gates_pass=True,
                    stand_readback_pass=stand_mode_readback_confirmed(final_snapshot),
                ),
            )
            if not turn_stop_rearmed or executor.step_index != 2:
                result["status"] = "BLOCKED"
                result["fault"] = "route_corner_stop_or_step_transition_failed"
                result["route_events"] = route_events
                return result
            continuation_step = executor.active_step
            if (
                continuation_step is None
                or continuation_step.waypoint_id != "TRANSIT-AFTER-RIGHT-CORNER"
                or next_segment.kind is not DirectiveKind.FOLLOW_WAYPOINT
            ):
                result["status"] = "BLOCKED"
                result["fault"] = "route_manager_did_not_activate_waypoint_c"
                result["route_events"] = route_events
                return result

            continuation_readiness = readiness_from_snapshot(final_snapshot)
            continuation_plan = bridge.plan_next(
                next_segment,
                pose_provider.pose,
                continuation_readiness,
                now_steady_s=float(final_snapshot["sample_monotonic_s"]),
            )
            if (
                continuation_plan.status != "PROPOSAL_ONLY"
                or continuation_plan.forward_m_s <= 0.0
                or abs(continuation_plan.angular_rad_s) > POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S
            ):
                result["status"] = "BLOCKED"
                result["fault"] = (
                    f"route_bridge_refused_waypoint_c:{continuation_plan.status}:"
                    f"{continuation_plan.reason}"
                )
                result["route_events"] = route_events
                return result

            continuation_summary = run_transit_windows(
                next_segment,
                continuation_plan,
                "TRANSIT-AFTER-RIGHT-CORNER",
            )
            continuation = continuation_summary.get("last_result") or {}
            continuation_gate = continuation_summary.get("last_gate") or {}
            if (
                continuation_summary.get("status") not in (
                    "ARRIVAL_DURING_WINDOW", "ARRIVAL_AFTER_VERIFIED_STOP",
                )
                or route_fault
                or route_stop_reason is None
                or continuation.get("physical_motion_status") != "PHYSICAL_MOTION_CONFIRMED"
                or continuation_gate.get("status") != "PASS"
            ):
                result["status"] = "BLOCKED"
                result["fault"] = (
                    continuation_summary.get("fault") or route_fault or route_stop_reason
                    or continuation.get("fault")
                    or "waypoint_c_or_physical_stop_not_passed"
                )
                result["route_events"] = route_events + route_motion.events
                result["waypoint_arrival_poses"] = route_arrival_poses
                result["route_total_nonzero_command_time_s"] = route_total_nonzero_s
                return result

            continuation_evidence = stop_mapping(continuation_gate)
            continuation_zero_start = float(continuation_gate.get("zero_start_monotonic_s", 0.0))
            final_stop_rearmed = bool(continuation_summary.get("bridge_stop_rearmed"))
            continuation_zero_plan = continuation_summary.get("zero_plan")
            if not final_stop_rearmed:
                continuation_zero_plan = bridge.zero_window(
                    route_motion.latest,
                    now_steady_s=time.monotonic(),
                    reason="waypoint C reached; verifying final measured physical stop",
                    physical_stop_gate_required=True,
                )
                final_stop_rearmed = bridge.confirm_physical_stop(
                    continuation_evidence,
                    zero_request_started_s=continuation_zero_start,
                    now_steady_s=time.monotonic(),
                    expected_generation=str(self.args.attempt_id),
                )
                final_snapshot = self.sample_for_first_motion()
                final_pose = pose_provider.update(final_snapshot)
                safety_provider.update(final_snapshot, continuation_step.waypoint_id)
                final_c_evidence = PhysicalStopEvidence(
                    status=PhysicalStopStatus.PASS,
                    window_start_steady_s=float(continuation_evidence["window_start_steady_s"]),
                    window_end_steady_s=float(continuation_evidence["window_end_steady_s"]),
                    sample_count=int(continuation_evidence["sample_count"]),
                    maximum_sample_gap_s=float(continuation_evidence["maximum_sample_gap_s"]),
                    maximum_state_age_s=float(continuation_evidence["maximum_state_age_s"]),
                    generation=str(self.args.attempt_id),
                )
                completion = executor.tick(
                    float(final_snapshot["sample_monotonic_s"]),
                    MotionFeedback(
                        status=MotionStatus.ACTIVE,
                        physical_stop_evidence=final_c_evidence,
                        reset_gate_pass=True,
                        standing_gate_pass=True,
                        source_mode_handshake_pass=False,
                        safety_gates_pass=True,
                        stand_readback_pass=stand_mode_readback_confirmed(final_snapshot),
                    ),
                )
            else:
                final_snapshot = continuation_summary.get("final_snapshot")
                if not isinstance(final_snapshot, Mapping):
                    result["status"] = "BLOCKED"
                    result["fault"] = "waypoint_c_poststop_snapshot_missing"
                    return result
                final_pose = pose_provider.pose
                if final_pose is None:
                    result["status"] = "BLOCKED"
                    result["fault"] = "waypoint_c_poststop_pose_missing"
                    return result
                final_c_evidence = PhysicalStopEvidence(
                    status=PhysicalStopStatus.PASS,
                    window_start_steady_s=float(continuation_evidence["window_start_steady_s"]),
                    window_end_steady_s=float(continuation_evidence["window_end_steady_s"]),
                    sample_count=int(continuation_evidence["sample_count"]),
                    maximum_sample_gap_s=float(continuation_evidence["maximum_sample_gap_s"]),
                    maximum_state_age_s=float(continuation_evidence["maximum_state_age_s"]),
                    generation=str(self.args.attempt_id),
                )
                completion = continuation_summary.get("next_directive")
                if completion is None:
                    result["status"] = "BLOCKED"
                    result["fault"] = "waypoint_c_route_completion_directive_missing"
                    return result
            waypoint_b = route_arrival_poses.get(route.steps[0].waypoint_id)
            waypoint_c = route_arrival_poses.get(route.steps[2].waypoint_id)
            actual_waypoint_separation = (
                math.hypot(
                    waypoint_c["x_m"] - waypoint_b["x_m"],
                    waypoint_c["y_m"] - waypoint_b["y_m"],
                )
                if waypoint_b and waypoint_c else None
            )
            required_region_separation = (
                route.steps[0].position_tolerance_m + route.steps[2].position_tolerance_m
            )
            route_source_mode_pass = all(
                isinstance(segment.get("result"), Mapping)
                and segment["result"].get("control_handshake_confirmed") is True
                for segment in result["segments"]
            )
            total_planned_nonzero_s = (
                sum(float(step.max_command_window_s) for step in route.steps)
            )
            total_actual_nonzero_s = route_total_nonzero_s
            route_complete = (
                final_stop_rearmed
                and route_source_mode_pass
                and executor.state.value == "COMPLETE"
                and completion.kind is DirectiveKind.COMPLETE
                and final_c_evidence.status is PhysicalStopStatus.PASS
                and final_pose.frame_id == "odom"
            )
            if (
                not route_complete
                or actual_waypoint_separation is None
                or actual_waypoint_separation <= required_region_separation
                or total_planned_nonzero_s > 30.0
                or total_actual_nonzero_s > 30.0
            ):
                result["status"] = "BLOCKED"
                result["fault"] = "route_completion_waypoint_separation_or_30s_budget_gate_failed"
                result["route_events"] = route_events + route_motion.events
                result["waypoint_arrival_poses"] = route_arrival_poses
                result["final_stop_rearmed"] = final_stop_rearmed
                result["route_state_after_c"] = executor.state.value
                result["total_planned_nonzero_command_window_s"] = total_planned_nonzero_s
                result["total_actual_nonzero_command_time_s"] = total_actual_nonzero_s
                return result

            waypoint_b_error = math.hypot(
                route.steps[0].x_m - waypoint_b["x_m"],
                route.steps[0].y_m - waypoint_b["y_m"],
            )
            waypoint_c_error_at_arrival = math.hypot(
                route.steps[2].x_m - waypoint_c["x_m"],
                route.steps[2].y_m - waypoint_c["y_m"],
            )

            result.update({
                "status": "PASS",
                "classification": "SIMULATION_ONLY_SHORT_ROUTE_TRANSIT_RIGHT_TURN_TRANSIT_PASS",
                "second_transit": "PASS",
                "route_state_after_route": executor.state.value,
                "next_route_directive": completion.kind.value,
                "next_route_step": None,
                "waypoint_arrival_poses": route_arrival_poses,
                "route_gates": {
                    "RESET_STAND": "PASS",
                    "SOURCE_MODE": "PASS" if route_source_mode_pass else "NOT_CONFIRMED",
                    "WAYPOINT_B_ARRIVAL": "PASS" if (
                        transit.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
                        and (transit.get("physical_stop_gate") or {}).get("status") == "PASS"
                        and waypoint_b_error <= route.steps[0].position_tolerance_m
                    ) else "FAIL",
                    "BOUNDED_RIGHT_TURN": "PASS" if (
                        route_turn_yaw_pass and route_turn_displacement_pass
                        and turn_gate.get("status") == "PASS"
                    ) else "FAIL",
                    "WAYPOINT_C_ARRIVAL": "PASS" if (
                        continuation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
                        and waypoint_c_error_at_arrival <= route.steps[2].position_tolerance_m
                    ) else "FAIL",
                    "FINAL_PHYSICAL_STOP": "PASS" if continuation_gate.get("status") == "PASS" else "FAIL",
                    "ROUTE_COMPLETE": "PASS" if route_complete else "FAIL",
                },
                "waypoint_target_errors": {
                    "TRANSIT-TO-CORNER": {
                        "target_x_m": route.steps[0].x_m,
                        "target_y_m": route.steps[0].y_m,
                        "arrival_error_m": waypoint_b_error,
                        "tolerance_m": route.steps[0].position_tolerance_m,
                    },
                    "TRANSIT-AFTER-RIGHT-CORNER": {
                        "target_x_m": route.steps[2].x_m,
                        "target_y_m": route.steps[2].y_m,
                        "arrival_error_m": waypoint_c_error_at_arrival,
                        "post_stop_error_m": math.hypot(
                            route.steps[2].x_m - final_pose.x_m,
                            route.steps[2].y_m - final_pose.y_m,
                        ),
                        "tolerance_m": route.steps[2].position_tolerance_m,
                    },
                },
                "waypoint_b_c_separation_m": actual_waypoint_separation,
                "minimum_distinct_region_separation_m": required_region_separation,
                "route_events": route_events + route_motion.events,
                "final_odometry": final_snapshot.get("pose"),
                "physical_stop_gate": continuation_gate,
                "final_stop_rearmed": final_stop_rearmed,
                "total_actual_nonzero_command_time_s": total_actual_nonzero_s,
                "total_planned_nonzero_command_window_s": total_planned_nonzero_s,
                "total_nonzero_command_time_limit_s": 30.0,
                "route_turn_gates": {
                    "route_yaw_target_reached": True,
                    "rightward_yaw_at_least_0p087_rad": route_turn_yaw_pass,
                    "signed_yaw_delta_rad": turn_rotation.get("signed_yaw_delta_rad"),
                    "total_displacement_le_0p25m_including_stop": route_turn_displacement_pass,
                    "maximum_total_displacement_including_stop_m": turn_guard.get("maximum_total_displacement_including_stop_m"),
                    "physical_stop_gate": turn_gate.get("status"),
                    "control_handshake": turn.get("control_handshake_confirmed"),
                },
                "route_waypoint_c_gates": {
                    "physical_motion_confirmed": continuation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED",
                    "position_error_m": math.hypot(
                        continuation_step.x_m - final_pose.x_m,
                        continuation_step.y_m - final_pose.y_m,
                    ),
                    "position_tolerance_m": continuation_step.position_tolerance_m,
                    "physical_stop_gate": continuation_gate.get("status"),
                    "command_angular_velocity_rad_s": continuation_angular,
                },
                "limitations": [
                    "This pass is one short measured transit to B, bounded right corner, and transit to C only.",
                    "The route corridor is static open-floor clearance from the pinned scene, not a LiDAR/map-based obstacle guarantee.",
                    "This does not establish a closed L route, rectangle, return, docking, Nav2, or real-hardware behavior.",
                ],
            })
            return result
        except Exception as exc:
            result["status"] = "BLOCKED"
            result["fault"] = f"{type(exc).__name__}: {exc}"
            result["route_events"] = locals().get("route_events", [])
            return result

    def execute(self) -> dict[str, Any]:
        stationary_baseline: dict[str, Any] | None = None
        result: dict[str, Any] = {
            "schema": "robotsim.issue80.m1.first-motion.v2",
            "simulation_only": True,
            "acceptance_scope": (
                "SIMULATION_ONLY direct MuJoCo ground-truth point-goal; not Nav2, localization, LiDAR, or manufacturer-original acceptance"
                if getattr(self.args, "point_goal_diagnostic", False)
                else "SIMULATION_ONLY Local01 start-heading short two-target route; not table docking, full-distance route, Nav2, or manufacturer-original acceptance"
                if getattr(self.args, "short_scene_route", False)
                else "SIMULATION_ONLY 0.20 m short-waypoint diagnostic only; not original 0.40 m waypoint-B acceptance"
                if getattr(self.args, "route_waypoint_b_short_diagnostic", False)
                else "SIMULATION_ONLY measured waypoint-B transit and physical stop only; no turn, C waypoint, or Nav2 acceptance"
                if getattr(self.args, "route_waypoint_b_only", False)
                else "SIMULATION_ONLY waypoint route manager + route bridge to M1 transit and right-turn; not a full route or Nav2 acceptance"
                if getattr(self.args, "route_integration", False)
                else "SIMULATION_ONLY derived-configuration left/right turning and physical-stop validation; "
                "not manufacturer-original acceptance or point-goal navigation"
                if getattr(self.args, "turning_diagnostic", False)
                else "SIMULATION_ONLY derived-configuration right-turn and physical-stop qualification; not a 90-degree corner or manufacturer-original acceptance"
                if getattr(self.args, "right_turn_qualification", False)
                else "SIMULATION_ONLY derived-configuration locomotion and physical-stop validation; not manufacturer-original acceptance, turning, or point-goal navigation"
            ),
            "attempt_id": self.args.attempt_id,
            "maintainer_authorized": bool(self.args.maintainer_authorized),
            "zero_only_activation_authorized": bool(
                getattr(self.args, "maintainer_authorized_zero_only_activation", False)
            ),
            "zero_only_activation_verification": bool(
                getattr(self.args, "zero_only_activation_verification", False)
            ),
            "activation_probe_authorized": bool(
                getattr(self.args, "maintainer_authorized_activation_probe", False)
            ),
            "activation_probe_qualification": bool(
                getattr(self.args, "activation_probe_qualification", False)
            ),
            "activation_probe_prerequisite_attempt_id": getattr(
                self.args, "activation_probe_prerequisite_attempt_id", None
            ),
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
        if getattr(self.args, "right_turn_qualification", False):
            result["forward_command_m_s"] = TURN_FORWARD_COMMAND_M_S
            result["forward_warmup_s"] = TURN_FORWARD_WARMUP_S
            result["pulse_limit_s"] = TURN_PULSE_S
            result["total_turn_command_window_s"] = TURN_FORWARD_WARMUP_S + TURN_PULSE_S
        if getattr(self.args, "route_integration", False):
            result["forward_command_m_s"] = (
                float(self.args.forward_speed)
                if getattr(self.args, "short_scene_route", False)
                else TURN_FORWARD_COMMAND_M_S
            )
            if getattr(self.args, "short_scene_route", False):
                result["route_transit_nonzero_limit_s"] = SHORT_ROUTE_MAX_AGGREGATE_S
                result["route_transit_maximum_windows"] = SHORT_ROUTE_MAX_WINDOWS
                result["route_action_window_limit_s"] = SHORT_ROUTE_MAX_WINDOW_S
                result["route_classification"] = "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING"
            waypoint_b_only = bool(getattr(self.args, "route_waypoint_b_only", False))
            waypoint_b_extended = bool(
                getattr(self.args, "route_waypoint_b_extended_diagnostic", False)
            )
            route_transit_limit_s = (
                WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S
                if waypoint_b_extended else
                WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S
                if waypoint_b_only else 5.0
            )
            result["pulse_limit_s"] = route_transit_limit_s
            result["route_action_window_limit_s"] = 5.0
            if waypoint_b_only:
                result["route_transit_nonzero_limit_s"] = route_transit_limit_s
                result["route_transit_maximum_windows"] = (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS
                    if waypoint_b_extended else WAYPOINT_B_MAX_WINDOWS
                )
                result["route_total_nonzero_limit_s"] = 30.0
        try:
            result["transport_scope"] = self.check_transport_scope()
            self.wait_for_standing()
            result["standing_preflight"] = "FRESH_STAND_DEFAULT_BODY_STAND"
            if getattr(self.args, "short_scene_route", False):
                reset_path = Path(self.args.output_dir) / "reset_immediate_posture.json"
                route_path = Path(self.args.route_config)
                if any(path.is_symlink() or not path.is_file() for path in (reset_path, route_path)):
                    raise ValueError("Local01 route requires staged route config and immediate Reset evidence")
                reset_record = json.loads(reset_path.read_text(encoding="utf-8"))
                route_input = json.loads(route_path.read_text(encoding="utf-8"))
                validate_local01_short_scene_route_input(route_input)
                reset_odom = reset_record.get("latest_odom")
                reset_data = reset_odom.get("data") if isinstance(reset_odom, Mapping) else None
                state_ages = reset_record.get("state_ages_s")
                if (
                    reset_record.get("status") != "POST_RESET_UPRIGHT_OBSERVED"
                    or reset_record.get("fresh_all_required_topics") is not True
                    or not isinstance(state_ages, Mapping)
                    or state_ages.get("odom") is None
                    or float(state_ages["odom"]) > self.args.state_timeout
                    or not isinstance(reset_odom, Mapping)
                    or reset_odom.get("topic_key") != "odom"
                    or not isinstance(reset_data, Mapping)
                ):
                    raise ValueError("Local01 route Reset odometry sample is missing, stale, or not upright/fresh")
                actual_reset_odom = {
                    "x_m": float(reset_data["x"]),
                    "y_m": float(reset_data["y"]),
                    "z_m": float(reset_data["z"]),
                    "yaw_rad": float(reset_data["yaw_rad"]),
                }
                frame_gate = validate_local01_reset_odom_pose(
                    route_input["reset_anchor_expected_odom"], actual_reset_odom,
                )
                result["scene_world_to_odom_reset_gate"] = {
                    **frame_gate,
                    "reset_dispatch_monotonic_ns": reset_record.get("dispatch_monotonic_ns"),
                    "reset_to_odom_sample_monotonic_ns": reset_odom.get("wall_monotonic_ns"),
                    "reset_odom_sample_age_s": state_ages["odom"],
                    "live_reset_odom_sample_source": "parent reset_state_trace.jsonl /aima/hal/odom/state",
                    "navigation_command_published_before_gate": False,
                }
                if frame_gate.get("status") != "PASS":
                    result.update({
                        "status": "BLOCKED",
                        "fault": "LOCAL01_SCENE_WORLD_TO_ODOM_RESET_ANCHOR_MISMATCH",
                        "route_motion": "NOT_STARTED",
                        "nonzero_publish_calls_returned": 0,
                    })
                    return result
            result["source_before_registration"] = self._source_response_before_motion()
            registration = self.register_source()
            result["source_registration"] = registration
            if registration["success"]:
                # The stationary baseline belongs to verified STAND, before the
                # locomotion action is requested. It is also command-free.
                stationary_baseline = self.collect_stationary_odom_baseline()
                if getattr(self.args, "activation_probe_qualification", False):
                    lifecycle, set_action, activation_gate = self.begin_locomotion_activation_probe_request(
                        registration=registration,
                        phase="controlled-activation-probe",
                    )
                elif getattr(self.args, "route_integration", False):
                    lifecycle, set_action, activation_gate = self.begin_source_grounded_route_activation(
                        registration=registration,
                        phase="initial-route-window",
                        forward_command_m_s=float(self.args.forward_speed),
                        allow_raicom_low_cruise=bool(
                            getattr(self.args, "raicom_low_cruise_profile", False)
                        ),
                        zero_forward_heading_acquisition=bool(
                            getattr(self.args, "raicom_zero_forward_heading_acquisition", False)
                            or fixed_window_heading_acquisition_required(
                                fixed_window_table_approach=bool(
                                    getattr(self.args, "raicom_fixed_window_table_approach", False)
                                ),
                                segment_id="WAYPOINT",
                            )
                        ),
                    )
                else:
                    lifecycle, set_action, activation_gate = self.begin_zero_only_locomotion_activation(
                        registration=registration,
                        phase="initial",
                    )
            else:
                lifecycle = None
                activation_gate = {
                    "status": "BLOCKED",
                    "classification": "SOURCE_REGISTRATION_REJECTED",
                    "fault": "input source registration did not pass; SetMcAction was not sent",
                    "zero_publish_calls_returned": 0,
                    "nonzero_publish_calls_returned": 0,
                }
                set_action = {
                    "accepted_by_service": False,
                    "request_sent": False,
                    "header_code": None,
                    "status_value": None,
                    "message": "source registration failed; SetMcAction was not sent",
                }
                self.events.append({"event": "set_mc_action_ld_not_sent", **set_action})
            result["set_mc_action_ld"] = set_action
            result["stationary_odom_baseline"] = stationary_baseline
            result["locomotion_activation_gate"] = activation_gate
            result["zero_only_activation_lifecycle"] = (
                lifecycle.snapshot() if lifecycle is not None else None
            )

            if getattr(self.args, "zero_only_activation_verification", False):
                recovery = None
                if lifecycle is not None:
                    recovery = self.recover_activation_to_stand(
                        lifecycle=lifecycle,
                        registration=registration,
                        set_action=set_action,
                        baseline=stationary_baseline,
                        reason=(
                            "zero-only activation observed LOCOMOTION; cancel to STAND for the bounded verification"
                            if activation_gate.get("status") == "PASS"
                            else str(activation_gate.get("fault") or activation_gate.get("terminal_reason") or "zero-only activation not confirmed")
                        ),
                        phase="zero-only-verification-recovery",
                    )
                stop_gate = recovery.get("physical_stop_gate") if isinstance(recovery, Mapping) else None
                stop_confirmation = recovery.get("stop_transition_confirmation") if isinstance(recovery, Mapping) else None
                activation_pass = activation_gate.get("status") == "PASS"
                recovery_pass = (
                    isinstance(stop_gate, Mapping)
                    and stop_gate.get("status") == "PASS"
                    and isinstance(stop_confirmation, Mapping)
                    and stop_confirmation.get("stand_confirmed") is True
                    and isinstance(recovery.get("after_zero_stop_action"), Mapping)
                    and recovery["after_zero_stop_action"].get("success") is True
                    and recovery.get("post_stop_fault") is None
                    and int(recovery.get("nonzero_publish_calls_returned", 0) or 0) == 0
                    and isinstance(recovery.get("activation_lifecycle"), Mapping)
                    and recovery["activation_lifecycle"].get("state") == "STAND_VERIFIED"
                )
                final_lifecycle = (
                    recovery.get("activation_lifecycle")
                    if isinstance(recovery, Mapping)
                    else lifecycle.snapshot() if lifecycle is not None else None
                )
                late_locomotion = bool(
                    isinstance(final_lifecycle, Mapping)
                    and final_lifecycle.get("late_locomotion_after_abort") is True
                )
                result["zero_only_activation_verification"] = {
                    "classification": "SIMULATION_ONLY_ZERO_COMMAND_ACTIVATION_AND_CANCEL_TEST",
                    "status": "PASS" if activation_pass and recovery_pass else "BLOCKED",
                    "activation": activation_gate,
                    "stand_recovery": recovery,
                    "activation_confirmation_pass": activation_pass,
                    "stand_and_composite_stop_pass": recovery_pass,
                    "late_locomotion_after_abort": late_locomotion,
                    "nonzero_publish_calls_returned": 0,
                    "route_motion_authorized": False,
                    "final_lifecycle": final_lifecycle,
                }
                result["acceptance_gates"] = {
                    "ZERO_ONLY_MODE_TRANSITION": "PASS" if activation_pass else "NOT_CONFIRMED",
                    "MC_ACTIVATION_RECOVERY": "PASS" if recovery_pass else "BLOCKED",
                    "LATE_LOCOMOTION_ABORT": (
                        "RECOVERED_NO_REARM" if late_locomotion and recovery_pass
                        else "UNRESOLVED" if late_locomotion
                        else "NOT_OBSERVED"
                    ),
                    "PHYSICAL_STOP": "PASS" if recovery_pass else "FAIL",
                    "IN_PLACE_ROTATION_SUPPORTED": "NOT_RUN",
                    "ROTATE_TO_YAW": "NOT_RUN",
                    "STRAIGHT_LINE_TRACKING": "NOT_RUN",
                    "GO_TO_POSE_POSITION": "NOT_RUN",
                    "GO_TO_POSE_FINAL_YAW": "NOT_RUN",
                    "GO_TO_POSE_RUNTIME": "NOT_RUN",
                    "HUMAN_VISUAL_REVIEW": "PENDING_USER_REVIEW",
                }
                self.pulse_result = recovery
                result["nonzero_publish_calls_returned"] = 0
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["get_mc_action_final"] = self.read_mode("final_zero_only_activation")
                result["get_current_input_source_final"] = self.read_input_source("final_zero_only_activation")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["status"] = "PASS" if activation_pass and recovery_pass else "BLOCKED"
                result["fault"] = None if result["status"] == "PASS" else (
                    activation_gate.get("fault")
                    or activation_gate.get("terminal_reason")
                    or "zero-only activation or its physical cancellation gate did not pass"
                )
                return result

            if getattr(self.args, "activation_probe_qualification", False):
                probe_result = None
                if (
                    lifecycle is not None
                    and activation_gate.get("status") == "READY"
                    and set_action.get("accepted_by_service") is True
                ):
                    probe_result = run_first_motion_window(
                        sample=self.sample_for_first_motion,
                        publish=self._publish,
                        sleep=self._spin_for,
                        registration=registration,
                        set_action=set_action,
                        joint_limits=self.joint_limits,
                        baseline_forward_displacements_m=[
                            float(window["forward_displacement_m"])
                            for window in stationary_baseline.get("windows", [])
                        ],
                        baseline_state_summary=stationary_baseline.get("physical_state_summary"),
                        authorized=True,
                        forward_command_m_s=ACTIVATION_PROBE_FORWARD_M_S,
                        pulse_duration_s=ACTIVATION_PROBE_DURATION_S,
                        max_pulse_duration_s=ACTIVATION_PROBE_DURATION_S,
                        extra_post_stop_observe_s=5.0,
                        extra_post_stop_observe_limit_s=5.0,
                        after_zero_abort=self.request_postwalk_stand_once,
                        shutdown_event=self.interrupted,
                        on_sample=lambda sample_phase, snapshot: self._trace_first_motion_sample(
                            f"activation-probe-{sample_phase}", dict(snapshot),
                        ),
                        state_timeout_s=self.args.state_timeout,
                        min_root_z=self.args.min_root_z,
                        max_tilt_deg=self.args.max_tilt_deg,
                        max_observed_planar_speed_m_s=self._activation_speed_cap(),
                        zero_until_stand_timeout_s=3.5,
                        require_locomotion_confirmation=False,
                        activation_lifecycle=lifecycle,
                        activation_probe_only=True,
                        source_contract_supported=(
                            (getattr(self.args, "activation_probe_prerequisite", None) or {}).get("status")
                            == "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED"
                        ),
                    )
                else:
                    probe_result = self.recover_activation_to_stand(
                        lifecycle=lifecycle or MCLocomotionAbortLifecycle(),
                        registration=registration,
                        set_action=set_action,
                        baseline=stationary_baseline,
                        reason=str(activation_gate.get("fault") or "activation probe prerequisites did not pass"),
                        phase="activation-probe-preflight-recovery",
                    )
                self.pulse_result = probe_result
                probe = probe_result.get("activation_probe", {}) if isinstance(probe_result, Mapping) else {}
                stand_gate = probe_result.get("physical_stop_gate") if isinstance(probe_result, Mapping) else None
                stand_confirmation = probe_result.get("stop_transition_confirmation") if isinstance(probe_result, Mapping) else None
                final_lifecycle = (
                    probe_result.get("activation_lifecycle") if isinstance(probe_result, Mapping) else None
                )
                mode_source_activation_pass = probe.get("mode_source_confirmed") is True
                intentional_abort_confirmed = activation_probe_abort_is_expected(probe_result)
                physical_stop_pass = activation_probe_safe_stop_passes(
                    probe_result,
                    stand_gate if isinstance(stand_gate, Mapping) else None,
                    stand_confirmation if isinstance(stand_confirmation, Mapping) else None,
                    final_lifecycle if isinstance(final_lifecycle, Mapping) else None,
                )
                late_locomotion = bool(
                    isinstance(final_lifecycle, Mapping)
                    and final_lifecycle.get("late_locomotion_after_abort") is True
                )
                result["activation_probe_qualification"] = {
                    "classification": "SIMULATION_ONLY_SOURCE_GROUNDED_ACTIVATION_ABORT_PROBE",
                    "status": "PASS" if mode_source_activation_pass and physical_stop_pass and not late_locomotion else "BLOCKED",
                    "preflight": activation_gate,
                    "probe": probe_result,
                    "mode_source_activation_pass": mode_source_activation_pass,
                    "intentional_abort_confirmed": intentional_abort_confirmed,
                    "safe_stand_and_physical_stop_pass": physical_stop_pass,
                    "late_locomotion_after_abort": late_locomotion,
                    "route_motion_authorized": False,
                    "final_lifecycle": final_lifecycle,
                }
                result["acceptance_gates"] = {
                    "MC_ACTIVATION_RECOVERY": "PASS" if physical_stop_pass else "BLOCKED",
                    "MC_MODE_SOURCE_ACTIVATION": "PASS" if mode_source_activation_pass else "BLOCKED",
                    "LATE_LOCOMOTION_ABORT": (
                        "RECOVERED_NO_REARM" if late_locomotion and physical_stop_pass
                        else "UNRESOLVED" if late_locomotion
                        else "NOT_OBSERVED"
                    ),
                    "PHYSICAL_STOP": "PASS" if physical_stop_pass else "FAIL",
                    "IN_PLACE_ROTATION_SUPPORTED": "NOT_RUN",
                    "ROTATE_TO_YAW": "NOT_RUN",
                    "STRAIGHT_LINE_TRACKING": "NOT_RUN",
                    "GO_TO_POSE_POSITION": "NOT_RUN",
                    "GO_TO_POSE_FINAL_YAW": "NOT_RUN",
                    "GO_TO_POSE_RUNTIME": "NOT_RUN",
                    "HUMAN_VISUAL_REVIEW": "PENDING_USER_REVIEW",
                }
                result["nonzero_publish_calls_returned"] = int(
                    probe_result.get("nonzero_publish_calls_returned", 0) or 0
                ) if isinstance(probe_result, Mapping) else 0
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["get_mc_action_final"] = self.read_mode("final_activation_probe")
                result["get_current_input_source_final"] = self.read_input_source("final_activation_probe")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["status"] = result["activation_probe_qualification"]["status"]
                result["fault"] = None if result["status"] == "PASS" else (
                    probe_result.get("fault") if isinstance(probe_result, Mapping) else None
                ) or "activation probe mode/source or physical-stop gate did not pass"
                return result

            allowed_activation_statuses = (
                {"READY"} if getattr(self.args, "route_integration", False) else {"PASS"}
            )
            if activation_gate.get("status") not in allowed_activation_statuses or lifecycle is None:
                recovery = None
                if lifecycle is not None:
                    recovery = self.recover_activation_to_stand(
                        lifecycle=lifecycle,
                        registration=registration,
                        set_action=set_action,
                        baseline=stationary_baseline,
                        reason=str(activation_gate.get("fault") or activation_gate.get("terminal_reason") or "LOCOMOTION activation was not confirmed"),
                        phase="activation-abort-recovery",
                    )
                result["activation_abort_recovery"] = recovery
                self.pulse_result = recovery
                result["nonzero_publish_calls_returned"] = 0
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["get_mc_action_final"] = self.read_mode("final_activation_abort_recovery")
                result["get_current_input_source_final"] = self.read_input_source("final_activation_abort_recovery")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["status"] = "BLOCKED"
                result["fault"] = activation_gate.get("fault") or activation_gate.get("terminal_reason") or "LOCOMOTION activation was not confirmed"
                return result

            self.activation_lifecycle = lifecycle
            result["activation_lifecycle_after_confirmation"] = lifecycle.snapshot()

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

            if getattr(self.args, "route_integration", False):
                route_result = self.run_short_route_integration(
                    registration=registration,
                    set_action=set_action,
                    stationary_baseline=stationary_baseline or {},
                )
                self.pulse_result = route_result
                result["route_integration"] = route_result
                result["nonzero_publish_calls_returned"] = int(self.nonzero_publish_calls_returned)
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["mc_velocity_receipt_confirmed"] = False
                result["mc_stop_confirmed"] = False
                result["get_mc_action_final"] = self.read_mode("final_route_post_stop")
                result["get_current_input_source_final"] = self.read_input_source("final_route_post_stop")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["fault"] = route_result.get("fault")
                result["status"] = "PASS" if route_result.get("status") == "PASS" else "BLOCKED"
                return result

            if getattr(self.args, "right_turn_qualification", False):
                right_turn = self.run_right_turn_qualification(
                    registration=registration,
                    set_action=set_action,
                    stationary_baseline=stationary_baseline or {},
                )
                self.pulse_result = right_turn.get("turn")
                result["right_turn_qualification"] = right_turn
                result["activation"] = right_turn.get("turn")
                turn = right_turn.get("turn")
                result["nonzero_publish_calls_returned"] = (
                    int(turn.get("nonzero_publish_calls_returned", 0) or 0)
                    if isinstance(turn, Mapping) else 0
                )
                result["zero_publish_calls_returned"] = int(self.zero_publish_calls_returned)
                result["mc_velocity_receipt_confirmed"] = False
                result["mc_stop_confirmed"] = False
                result["physical_stop_gate"] = turn.get("physical_stop_gate") if isinstance(turn, Mapping) else None
                result["legacy_baseline_stop_gate"] = turn.get("legacy_baseline_stop_gate") if isinstance(turn, Mapping) else None
                result["get_mc_action_final"] = self.read_mode("final_right_turn_post_stop")
                result["get_current_input_source_final"] = self.read_input_source("final_right_turn_post_stop")
                result["final_common_state"] = _json_safe(self.latest.get("common"))
                result["final_odometry"] = self._pose()
                result["final_joint_state"] = _json_safe(self.latest.get("leg_state"))
                result["fault"] = turn.get("fault") if isinstance(turn, Mapping) else "right_turn_result_missing"
                result["status"] = "PASS" if right_turn.get("status") == "PASS" else "BLOCKED"
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
                after_zero_abort=self.request_postwalk_stand_once,
                shutdown_event=self.interrupted,
                on_sample=self._trace_first_motion_sample,
                state_timeout_s=self.args.state_timeout,
                min_root_z=self.args.min_root_z,
                max_tilt_deg=self.args.max_tilt_deg,
                max_observed_planar_speed_m_s=getattr(self.args, "max_observed_planar_speed_m_s", None),
                zero_until_stand_timeout_s=getattr(self.args, "zero_until_stand_timeout_s", 0.0),
                require_locomotion_confirmation=True,
                activation_lifecycle=self.activation_lifecycle,
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
                        after_zero_abort=self.request_postwalk_stand_once,
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
            result["odom_source_stamp_fence"] = {
                **self.odom_source_stamp_fence.snapshot(),
                "recorded_drop_events": self.odom_source_stamp_drop_events,
            }
            result["odom_subscription_configured_qos"] = self.odom_subscription_configured_qos
            result["optional_debug_capture"] = {
                "mc_debug_f64_topic": MC_DEBUG_F64_TOPIC,
                "mc_debug_f64_type": "std_msgs/msg/Float64MultiArray",
                "mc_debug_f64_subscription_enabled": self.mc_debug_f64_subscription_enabled,
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
        maintainer_authorized=bool(
            args.maintainer_authorized
            or getattr(args, "maintainer_authorized_zero_only_activation", False)
            or getattr(args, "maintainer_authorized_activation_probe", False)
        ),
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


def validate_route_turn_qualification_evidence(path: str | None) -> dict[str, Any]:
    """Require the prior bounded right arc and its independent physical stop to pass."""
    if not path:
        raise ValueError("route integration requires the completed right-turn qualification evidence")
    evidence_path = Path(path)
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ValueError("right-turn qualification evidence must be a regular file")
    try:
        result = json.loads(evidence_path.read_text(encoding="utf-8"))
        right = result["right_turn_qualification"]
        turn = right["turn"]
        acceptance = turn["right_arc_acceptance"]
        rotation = turn["rotation"]
        stop = turn["physical_stop_gate"]
        measured_yaw = float(rotation["signed_yaw_delta_rad"])
        maximum_displacement = float(acceptance["maximum_total_turn_origin_displacement_m"])
        active_duration = float(turn["pulse_actual_duration_s"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"right-turn qualification evidence is incomplete: {type(exc).__name__}: {exc}") from exc
    if (
        result.get("simulation_only") is not True
        or right.get("status") != "PASS"
        or acceptance.get("status") != "PASS"
        or rotation.get("status") != "TURN_DIRECTION_PASS"
        or measured_yaw > -TURN_MIN_HEADING_CHANGE_RAD
        or stop.get("status") != "PASS"
        or maximum_displacement > 0.25
        or active_duration > 5.0
    ):
        raise ValueError("prior right-turn evidence did not pass direction, displacement, 5 s, and physical-stop gates")
    return {
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "attempt_id": str(result.get("attempt_id", "")),
        "signed_yaw_delta_rad": measured_yaw,
        "maximum_total_displacement_m": maximum_displacement,
        "active_duration_s": active_duration,
        "physical_stop_gate": stop.get("status"),
    }


def validate_first_motion_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the complete child CLI profile without initializing ROS or the simulator."""
    zero_only_verification = bool(getattr(args, "zero_only_activation_verification", False))
    zero_only_authorized = bool(getattr(args, "maintainer_authorized_zero_only_activation", False))
    activation_probe = bool(getattr(args, "activation_probe_qualification", False))
    activation_probe_authorized = bool(getattr(args, "maintainer_authorized_activation_probe", False))
    if zero_only_verification != zero_only_authorized:
        raise ValueError("zero-only activation verification requires its separate authorization flag")
    if activation_probe != activation_probe_authorized:
        raise ValueError("activation probe requires its separate authorization flag")
    if sum((bool(args.maintainer_authorized), zero_only_authorized, activation_probe_authorized)) > 1:
        raise ValueError("first-motion, zero-only activation, and activation-probe authorizations are separate")
    startup_gate_verified = bool(
        args.reset_gate_verified or getattr(args, "initialization_gate_verified", False)
    )
    if args.reset_gate_verified and getattr(args, "initialization_gate_verified", False):
        raise ValueError("Reset and static initialization gates are mutually exclusive")
    if not startup_gate_verified or not (
        args.maintainer_authorized or zero_only_authorized or activation_probe_authorized
    ):
        raise ValueError("a verified Reset or static initialization gate and matching explicit authorization are required")
    # Existing route and stop logic use this internal boolean as an already
    # qualified startup gate; retain it without mislabelling static startup in
    # the command-line evidence.
    args.reset_gate_verified = startup_gate_verified
    if zero_only_verification and (
        args.maintainer_authorized
        or args.turning_diagnostic
        or args.point_goal_diagnostic
        or getattr(args, "right_turn_qualification", False)
    ):
        raise ValueError("zero-only activation verification cannot include a nonzero-motion authorization")
    if activation_probe:
        if (
            args.turning_diagnostic
            or args.point_goal_diagnostic
            or getattr(args, "right_turn_qualification", False)
            or getattr(args, "route_integration", False)
            or args.forward_speed != ACTIVATION_PROBE_FORWARD_M_S
            or args.pulse_duration != ACTIVATION_PROBE_DURATION_S
            or args.ramp_duration_s != 0.0
            or args.ramp_start_speed is not None
            or args.deceleration_duration_s != 0.0
            or args.startup_speed is not None
            or args.startup_transition_displacement is not None
            or args.startup_transition_timeout_s != 0.0
            or args.max_observed_planar_speed_m_s != 0.2
            or args.zero_until_stand_timeout_s != 3.5
            or not args.zero_only_stop
            or args.extra_stop_observation != 5.0
            or not getattr(args, "activation_probe_prerequisite_attempt_id", None)
        ):
            raise ValueError(
                "activation-probe qualification is fixed to 0.12 m/s for 1.80 s, "
                "zero lateral/yaw, 0.20 m/s measured-speed gate, 3.5 s STAND wait, "
                "and the existing 5 s post-stop window"
            )
        prerequisite_path = Path(str(getattr(args, "activation_probe_prerequisite_path", "")))
        if prerequisite_path.is_symlink() or not prerequisite_path.is_file():
            raise ValueError("activation-probe prerequisite summary is missing or is not a regular file")
        try:
            prerequisite = json.loads(prerequisite_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"activation-probe prerequisite summary is unreadable: {type(exc).__name__}: {exc}") from exc
        if (
            not isinstance(prerequisite, Mapping)
            or prerequisite.get("status") != "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED"
            or prerequisite.get("zero_only_preflight_attempt_id")
                != args.activation_probe_prerequisite_attempt_id
        ):
            raise ValueError("activation-probe prerequisite summary status/attempt identity is invalid")
        try:
            current_config_sha256 = hashlib.sha256(Path(args.mc_sim_config_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"activation-probe MC config is unreadable: {exc}") from exc
        if prerequisite.get("installed_mc_sim_config_sha256") != current_config_sha256:
            raise ValueError("activation-probe prerequisite MC config hash does not match this runtime")
        args.activation_probe_prerequisite = dict(prerequisite)
    selected_motion_modes = sum((
        args.turning_diagnostic,
        args.point_goal_diagnostic,
        getattr(args, "right_turn_qualification", False),
        getattr(args, "route_integration", False),
    ))
    if selected_motion_modes > 1:
        raise ValueError("select only one of turning, right-turn, or point-goal diagnostics")
    if getattr(args, "route_waypoint_b_post_locomotion_convergence", False) and not (
        getattr(args, "route_integration", False)
        and getattr(args, "route_waypoint_b_only", False)
        and getattr(args, "route_waypoint_b_extended_diagnostic", False)
    ):
        raise ValueError("post-locomotion convergence observation requires extended waypoint-B route integration")
    configured_route = bool(getattr(args, "route_config", None))
    short_scene_route = bool(getattr(args, "short_scene_route", False))
    low_cruise_profile = bool(getattr(args, "raicom_low_cruise_profile", False))
    zero_forward_heading_acquisition = bool(
        getattr(args, "raicom_zero_forward_heading_acquisition", False)
    )
    continuous_ten_second_transit = bool(
        getattr(args, "raicom_continuous_ten_second_transit", False)
    )
    continuous_table_approach = bool(
        getattr(args, "raicom_continuous_table_approach", False)
    )
    fixed_window_table_approach = bool(
        getattr(args, "raicom_fixed_window_table_approach", False)
    )
    forward_gait_yaw_inversion = bool(
        getattr(args, "raicom_forward_gait_yaw_inversion", False)
    )
    native_nominal_straight_gait_diagnostic = bool(
        getattr(args, "raicom_native_nominal_straight_gait_diagnostic", False)
    )
    table_approach_mission = bool(getattr(args, "table_approach_mission", False))
    aligned_start_table_approach = bool(
        getattr(args, "aligned_start_table_approach", False)
    )
    native_x2_timestep = bool(getattr(args, "native_x2_timestep", False))
    native_mjdata_route_feedback = bool(
        getattr(args, "raicom_native_mjdata_route_feedback", False)
    )
    configured_route_observation = bool(
        getattr(args, "route_config_post_stand_observation", False)
    )
    if short_scene_route and not configured_route:
        raise ValueError("--short-scene-route requires --route-config")
    low_cruise_supported_route = short_scene_route or (
        table_approach_mission and aligned_start_table_approach
    )
    if low_cruise_profile and not (
        low_cruise_supported_route
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        raise ValueError(
            "RAICOM low-cruise profile requires the short-scene route or aligned-start "
            "TABLE_APPROACH mission and exactly 0.105 m/s"
        )
    if low_cruise_supported_route and not low_cruise_profile and (
        abs(float(args.forward_speed) - LOCAL01_SHORT_ROUTE_FORWARD_M_S) > 1e-9
    ):
        raise ValueError(
            "the short-scene route and aligned-start TABLE_APPROACH mission require 0.12 m/s "
            "unless the explicit RAICOM low-cruise profile is selected"
        )
    if table_approach_mission and not configured_route:
        raise ValueError("--table-approach-mission requires --route-config")
    if table_approach_mission and short_scene_route:
        raise ValueError("table-approach mission cannot be combined with the short-scene route")
    if aligned_start_table_approach and not table_approach_mission:
        raise ValueError("--aligned-start-table-approach requires --table-approach-mission")
    if continuous_table_approach and fixed_window_table_approach:
        raise ValueError(
            "continuous and fixed-window TABLE_APPROACH profiles are mutually exclusive"
        )
    low_cruise_native_profile = (
        low_cruise_profile
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    )
    nominal_straight_native_profile = (
        native_nominal_straight_gait_diagnostic
        and not low_cruise_profile
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    )
    continuous_table_native_profile = (
        continuous_table_approach
        and native_mjdata_route_feedback
        and not low_cruise_profile
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    )
    fixed_window_table_native_profile = (
        fixed_window_table_approach
        and native_mjdata_route_feedback
        and not low_cruise_profile
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    )
    if native_x2_timestep and not (
        table_approach_mission
        and aligned_start_table_approach
        and (
            low_cruise_native_profile
            or nominal_straight_native_profile
            or continuous_table_native_profile
            or fixed_window_table_native_profile
        )
    ):
        raise ValueError(
            "native X2 timestep requires the aligned-start TABLE_APPROACH low-cruise profile "
            "the fixed-window table route, or the isolated nominal straight-gait diagnostic"
        )
    if native_mjdata_route_feedback and not (
        native_x2_timestep
        and table_approach_mission
        and aligned_start_table_approach
        and (
            low_cruise_native_profile
            or nominal_straight_native_profile
            or continuous_table_native_profile
            or fixed_window_table_native_profile
        )
    ):
        raise ValueError(
            "native MjData route feedback requires the aligned native-timestep TABLE_APPROACH profile"
        )
    if zero_forward_heading_acquisition and not (
        table_approach_mission
        and aligned_start_table_approach
        and native_x2_timestep
        and low_cruise_profile
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        raise ValueError(
            "zero-forward heading acquisition requires the native-timestep aligned TABLE_APPROACH low-cruise profile"
        )
    if continuous_ten_second_transit and not (
        zero_forward_heading_acquisition
        and table_approach_mission
        and aligned_start_table_approach
        and native_x2_timestep
        and low_cruise_profile
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        raise ValueError(
            "continuous ten-second transit requires the native-timestep aligned TABLE_APPROACH "
            "low-cruise heading-acquisition profile"
        )
    if continuous_table_approach and not (
        native_mjdata_route_feedback
        and native_x2_timestep
        and table_approach_mission
        and aligned_start_table_approach
        and native_nominal_straight_gait_diagnostic is False
        and not low_cruise_profile
        and not zero_forward_heading_acquisition
        and not continuous_ten_second_transit
        and not forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        raise ValueError(
            "continuous table approach requires native MjData feedback, the aligned native-timestep "
            "TABLE_APPROACH mission, and exactly 0.12 m/s without legacy stop/rearm profiles"
        )
    if fixed_window_table_approach and not (
        native_mjdata_route_feedback
        and native_x2_timestep
        and table_approach_mission
        and aligned_start_table_approach
        and not native_nominal_straight_gait_diagnostic
        and not low_cruise_profile
        and not zero_forward_heading_acquisition
        and not continuous_ten_second_transit
        and not forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        raise ValueError(
            "fixed-window table approach requires native MjData feedback, the aligned native-timestep "
            "TABLE_APPROACH mission, and exactly 0.12 m/s without continuous steering profiles"
        )
    if forward_gait_yaw_inversion and not (
        continuous_ten_second_transit
        and zero_forward_heading_acquisition
        and table_approach_mission
        and aligned_start_table_approach
        and native_x2_timestep
        and low_cruise_profile
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        raise ValueError(
            "forward-gait yaw inversion requires the explicit continuous native-timestep "
            "TABLE_APPROACH heading-acquisition profile"
        )
    if native_nominal_straight_gait_diagnostic and not (
        table_approach_mission
        and aligned_start_table_approach
        and native_x2_timestep
        and not low_cruise_profile
        and not zero_forward_heading_acquisition
        and not continuous_ten_second_transit
        and not forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        raise ValueError(
            "native nominal straight-gait diagnostic requires the aligned native-timestep "
            "TABLE_APPROACH profile at exactly 0.12 m/s without yaw correction"
        )
    if configured_route_observation and not configured_route:
        raise ValueError("configured-route post-STAND observation requires --route-config")
    if configured_route:
        if not getattr(args, "route_integration", False):
            raise ValueError("--route-config requires --route-integration")
        if any((
            args.route_curved_path_diagnostic,
            args.route_waypoint_b_only,
            args.route_waypoint_b_short_diagnostic,
            args.route_waypoint_b_extended_diagnostic,
            args.route_waypoint_b_post_locomotion_convergence,
            args.route_turn_qualification_evidence,
        )):
            raise ValueError("--route-config cannot be combined with built-in B, curve, or turn profiles")
        if not configured_route_observation or abs(float(args.extra_stop_observation) - 7.0) > 1e-9:
            raise ValueError("configured route requires the explicit bounded 7 s post-STAND observation")
        if (
            args.max_observed_planar_speed_m_s != 0.2
            or args.zero_until_stand_timeout_s != 3.5
            or not args.zero_only_stop
            or args.ramp_duration_s != 0.0
            or args.ramp_start_speed is not None
            or args.deceleration_duration_s != 0.0
            or args.deceleration_end_speed_m_s != 0.0
            or args.startup_speed is not None
            or args.startup_transition_displacement is not None
            or args.startup_transition_timeout_s != 0.0
        ):
            raise ValueError("configured route must retain the existing speed, zero-stop, and watchdog profile")
        if table_approach_mission:
            route_input = json.loads(Path(args.route_config).read_text(encoding="utf-8"))
            if aligned_start_table_approach:
                validate_runtime_raw_odom_aligned_start_table_approach_mission(route_input)
            else:
                validate_runtime_raw_odom_table_approach_mission(route_input)
            profile_route = {
                "status": "VALIDATED_NO_RUNTIME",
                "route_kind": route_input["classification"],
                "route_id": route_input["mission_id"],
                "frame_id": route_input["frame_id"],
                "step_ids": route_input["execution_order"],
                "maximum_windows_total": int(route_input["limits"]["max_windows"]),
                "aggregate_nonzero_limit_s": float(route_input["limits"]["max_aggregate_nonzero_s"]),
                "initial_yaw_alignment": dict(route_input["initial_alignment"]),
                "aligned_start_table_approach": aligned_start_table_approach,
                "native_x2_timestep": native_x2_timestep,
                "command_forward_m_s": float(args.forward_speed),
                "raicom_low_cruise_profile": low_cruise_profile,
                "raicom_zero_forward_heading_acquisition": zero_forward_heading_acquisition,
                "raicom_continuous_ten_second_transit": continuous_ten_second_transit,
                "raicom_fixed_window_table_approach": fixed_window_table_approach,
                "raicom_forward_gait_yaw_inversion": forward_gait_yaw_inversion,
                "raicom_native_nominal_straight_gait_diagnostic": native_nominal_straight_gait_diagnostic,
                "zero_forward_heading_profile": (
                    {
                        "classification": "SIMULATION_ONLY_AIMDK_YAW_SOURCE_ACQUISITION",
                        "forward_velocity_m_s": 0.0,
                        "angular_command_magnitude_rad_s": ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
                        "maximum_duration_s": ZERO_FORWARD_HEADING_ACQUISITION_MAX_S,
                        "minimum_measured_response_rad": ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD,
                    }
                    if zero_forward_heading_acquisition else None
                ),
                "continuous_transit_profile": (
                    {
                        "classification": "SIMULATION_ONLY_CONTINUOUS_TEN_SECOND_TABLE_TRANSIT",
                        "maximum_window_s": TABLE_APPROACH_CONTINUOUS_WINDOW_S,
                        "maximum_windows_total": TABLE_APPROACH_CONTINUOUS_MAX_WINDOWS,
                        "aggregate_nonzero_limit_s": 75.0,
                        "physical_stop_after_each_window": True,
                    }
                    if continuous_ten_second_transit else None
                ),
                "fixed_window_table_profile": (
                    {
                        "classification": "SIMULATION_ONLY_FIXED_FIVE_SECOND_STRAIGHT_OR_RUN65_ARC_TABLE_APPROACH",
                        "maximum_window_s": 5.0,
                        "maximum_windows_total": 15,
                        "aggregate_nonzero_limit_s": 75.0,
                        "angular_command_policy": "fresh pre-window bearing quantized to 0 or existing +/-0.075 rad/s Run65 arc",
                        "open_segment_yaw_acquisition": {
                            "scope": "START_TO_WAYPOINT_ONLY",
                            "forward_velocity_m_s": 0.0,
                            "angular_command_magnitude_rad_s": ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
                            "maximum_duration_s": ZERO_FORWARD_HEADING_ACQUISITION_MAX_S,
                            "minimum_measured_response_rad": ZERO_FORWARD_HEADING_ACQUISITION_MIN_RESPONSE_RAD,
                            "static_yaw_envelope_rad": FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
                        },
                        "physical_stop_after_each_window": True,
                    }
                    if fixed_window_table_approach else None
                ),
                "forward_gait_yaw_mapping": (
                    {
                        "classification": "SIMULATION_ONLY_RAICOM_FORWARD_GAIT_YAW_SIGN_INVERTED",
                        "stationary_source_acquisition_mapping": "documented_direct_sign_retained",
                        "forward_gait_mapping": "negative_of_standard_odom_bearing_feedback",
                        "evidence": (
                            "continuous10 native-timestep trace: -0.075 rad/s command accompanied "
                            "by increasing MuJoCo-ground-truth yaw while source and LOCOMOTION remained active"
                        ),
                    }
                    if forward_gait_yaw_inversion else None
                ),
                "straight_gait_diagnostic_profile": (
                    {
                        "classification": "SIMULATION_ONLY_NATIVE_NOMINAL_STRAIGHT_GAIT_DIAGNOSTIC",
                        "forward_velocity_m_s": 0.12,
                        "angular_velocity_rad_s": 0.0,
                        "maximum_window_s": TABLE_APPROACH_CONTINUOUS_WINDOW_S,
                        "maximum_windows_total": 1,
                        "acceptance": "sustained physical forward progress and complete physical stop only; not TABLE_APPROACH",
                    }
                    if native_nominal_straight_gait_diagnostic else None
                ),
                "runtime_reset_odom_observed": False,
            }
        elif short_scene_route:
            route_input = json.loads(Path(args.route_config).read_text(encoding="utf-8"))
            validate_local01_short_scene_route_input(route_input)
            profile_route = {
                "status": "VALIDATED_NO_RUNTIME",
                "route_kind": route_input["route_kind"],
                "route_id": route_input["route_id"],
                "frame_id": route_input["frame_id"],
                "source_frame_id": route_input["source_frame_id"],
                "start_anchor": route_input["start"],
                "step_ids": ["WAYPOINT", "SHORT_ROUTE_END"],
                "maximum_windows_total": 2,
                "aggregate_nonzero_limit_s": 10.0,
                "post_stand_observation_s": 7.0,
                "command_forward_m_s": float(args.forward_speed),
                "raicom_low_cruise_profile": low_cruise_profile,
                "runtime_reset_odom_observed": False,
            }
        else:
            route_input = load_waypoint_docking_pose_input(args.route_config)
            profile_route = {
                "status": "VALIDATED_NO_RUNTIME",
                "route_kind": "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE",
                "route_id": route_input["route_id"],
                "frame_id": route_input["frame_id"],
                "start_anchor": route_input["start"],
                "step_ids": ["WAYPOINT", "TABLE_DOCKING"],
                "maximum_windows_total": 3,
                "aggregate_nonzero_limit_s": 15.0,
                "post_stand_observation_s": 7.0,
            }
    else:
        profile_route = None
    profile: dict[str, Any] = {
        "point_goal": False,
        "zero_only_activation_verification": zero_only_verification,
        "activation_probe_qualification": activation_probe,
    }
    if profile_route is not None:
        profile[
            "configured_table_approach_mission" if table_approach_mission
            else "configured_local01_short_route" if short_scene_route
            else "configured_waypoint_docking_route"
        ] = profile_route
    if activation_probe:
        profile["activation_probe_prerequisite_attempt_id"] = args.activation_probe_prerequisite_attempt_id
    if getattr(args, "route_integration", False):
        profile["route_integration"] = True
        if configured_route:
            return {**profile, "route_integration": True}
        curved_path_diagnostic = bool(getattr(args, "route_curved_path_diagnostic", False))
        profile["route_curved_path_diagnostic"] = curved_path_diagnostic
        waypoint_b_only = bool(getattr(args, "route_waypoint_b_only", False))
        waypoint_b_short_diagnostic = bool(
            getattr(args, "route_waypoint_b_short_diagnostic", False)
        )
        waypoint_b_extended_diagnostic = bool(
            getattr(args, "route_waypoint_b_extended_diagnostic", False)
        )
        waypoint_b_post_locomotion_convergence = bool(
            getattr(args, "route_waypoint_b_post_locomotion_convergence", False)
        )
        profile["route_waypoint_b_only"] = waypoint_b_only
        profile["route_waypoint_b_extended_diagnostic"] = waypoint_b_extended_diagnostic
        profile["route_waypoint_b_post_locomotion_convergence"] = waypoint_b_post_locomotion_convergence
        if (waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic) and not waypoint_b_only:
            raise ValueError("waypoint-B diagnostics require --route-waypoint-b-only")
        if waypoint_b_short_diagnostic and waypoint_b_extended_diagnostic:
            raise ValueError("select only one waypoint-B diagnostic profile")
        if waypoint_b_post_locomotion_convergence and not (
            waypoint_b_only and waypoint_b_extended_diagnostic
        ):
            raise ValueError("post-locomotion convergence observation requires extended waypoint-B-only mode")
        if curved_path_diagnostic and (
            waypoint_b_only or waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic
            or args.route_turn_qualification_evidence
        ):
            raise ValueError("curved path must not be combined with waypoint-B or turn-qualification modes")
        if curved_path_diagnostic:
            profile.update(validate_curved_path_cli_profile(
                route_integration=True,
                curved_path=True,
                waypoint_b_mode=waypoint_b_only or waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic,
                turn_evidence_present=bool(args.route_turn_qualification_evidence),
                extra_stop_observation_s=float(args.extra_stop_observation),
                pulse_duration_s=float(args.pulse_duration),
                generic_forward_speed_m_s=float(args.forward_speed),
                ramp_duration_s=float(args.ramp_duration_s),
                ramp_start_speed_m_s=args.ramp_start_speed,
                deceleration_duration_s=float(args.deceleration_duration_s),
                startup_speed_m_s=args.startup_speed,
                startup_transition_displacement_m=args.startup_transition_displacement,
                startup_transition_timeout_s=float(args.startup_transition_timeout_s),
                max_observed_planar_speed_m_s=args.max_observed_planar_speed_m_s,
                zero_until_stand_timeout_s=float(args.zero_until_stand_timeout_s),
                zero_only_stop=bool(args.zero_only_stop),
            ))
        elif waypoint_b_only:
            convergence_observation = validate_b_post_locomotion_observation(
                args.extra_stop_observation,
                extended_b=waypoint_b_extended_diagnostic,
                convergence_diagnostic=waypoint_b_post_locomotion_convergence,
            )
            expected_b_observation_s = convergence_observation["extra_zero_observation_s"]
            if args.route_turn_qualification_evidence:
                raise ValueError("waypoint-B-only mode must not request turn qualification evidence")
            if (
                args.extra_stop_observation != expected_b_observation_s
                or args.pulse_duration != 1.0
                or args.forward_speed != 0.2
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.deceleration_duration_s != 0.0
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
                or args.max_observed_planar_speed_m_s != 0.2
                or args.zero_until_stand_timeout_s != 3.5
                or not args.zero_only_stop
            ):
                raise ValueError(
                    f"waypoint-B-only profile requires fixed {expected_b_observation_s:g} s post-stop observation, "
                    "default generic pulse fields, 0.2 m/s safety guard, 3.5 s STAND wait, "
                    "and zero-only stop"
                )
            profile.update({
                "route_waypoint_b_short_diagnostic": waypoint_b_short_diagnostic,
                "route_waypoint_b_extended_diagnostic": waypoint_b_extended_diagnostic,
                "waypoint_b_distance_m": (
                    WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M
                    if waypoint_b_short_diagnostic else 0.40
                ),
                "waypoint_b_tolerance_m": WAYPOINT_B_POSITION_TOLERANCE_M,
                "per_window_nonzero_limit_s": WAYPOINT_B_WINDOW_MAX_S,
                "aggregate_transit_nonzero_limit_s": (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S
                    if waypoint_b_extended_diagnostic else WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S
                ),
                "stop_reserve_m": (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                    if waypoint_b_extended_diagnostic else
                    WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                    if waypoint_b_short_diagnostic else 0.0
                ),
                "arrival_trigger_radius_m": waypoint_b_arrival_trigger_radius_m(
                    acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
                    stop_reserve_m=(
                        WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_extended_diagnostic else
                        WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M
                        if waypoint_b_short_diagnostic else 0.0
                    ),
                ),
                "minimum_windows_required": 1 if waypoint_b_short_diagnostic or waypoint_b_extended_diagnostic else 2,
                "maximum_windows": (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS
                    if waypoint_b_extended_diagnostic else WAYPOINT_B_MAX_WINDOWS
                ),
                "maximum_forward_projection_guard_m": (
                    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M
                    if waypoint_b_extended_diagnostic else None
                ),
                "route_corridor_forward_limit_m": WAYPOINT_B_CORRIDOR_MAX_FORWARD_M,
                "maximum_centerline_lateral_error_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                "route_nonzero_limit_s": 30.0,
                "profile_classification": (
                    "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC"
                    if waypoint_b_extended_diagnostic else
                    "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC"
                    if waypoint_b_short_diagnostic else "LEGACY_TWO_WINDOW_WAYPOINT_B"
                ),
                "original_waypoint_b_acceptance": (
                    "ORIGINAL_0_40_M_TARGET_DIAGNOSTIC_PROFILE_NOT_LEGACY_TWO_WINDOW_QUALIFICATION"
                    if waypoint_b_extended_diagnostic else
                    "NOT_SATISFIED_BY_SHORT_DIAGNOSTIC"
                    if waypoint_b_short_diagnostic else "THIS_IS_ORIGINAL_0_40_M_ACCEPTANCE"
                ),
                "post_locomotion_convergence_observation": (
                    convergence_observation if waypoint_b_post_locomotion_convergence else None
                ),
            })
        else:
            profile["route_turn_qualification_evidence"] = validate_route_turn_qualification_evidence(
                args.route_turn_qualification_evidence
            )
    elif (
        getattr(args, "route_curved_path_diagnostic", False)
        or
        getattr(args, "route_waypoint_b_only", False)
        or getattr(args, "route_waypoint_b_short_diagnostic", False)
        or getattr(args, "route_waypoint_b_extended_diagnostic", False)
    ):
        raise ValueError("route-waypoint-b-only requires --route-integration")
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
                POSE_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD if has_heading_target else None
            ),
            "pose_goal_terminal_direction_margin_rad": (
                abs(heading_change)
                + POSE_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD
                - POSE_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD
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
                "approach endpoint uses 0.05 rad bias to keep its curve feasible; terminal target uses a separate 0.115 rad bias derived from Run96 drift and the 5-degree minimum requested-turn gate; not calibrated across Runs93/95"
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
        help="bounded SIMULATION_ONLY activation or motion run following the verified Reset/standing gates",
    )
    first.add_argument("--maintainer-authorized", action="store_true")
    first.add_argument("--maintainer-authorized-zero-only-activation", action="store_true")
    first.add_argument("--maintainer-authorized-activation-probe", action="store_true")
    first.add_argument("--zero-only-activation-verification", action="store_true")
    first.add_argument("--activation-probe-qualification", action="store_true")
    first.add_argument("--activation-probe-prerequisite-attempt-id", default=None)
    first.add_argument("--activation-probe-prerequisite-path", default=None)
    first.add_argument("--reset-gate-verified", action="store_true")
    first.add_argument(
        "--initialization-gate-verified",
        action="store_true",
        help="verified static pre-load startup plus JD/SD standing gate; no Reset is implied",
    )
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
    first.add_argument("--right-turn-qualification", action="store_true")
    first.add_argument("--route-integration", action="store_true")
    first.add_argument("--route-curved-path-diagnostic", action="store_true")
    first.add_argument("--route-waypoint-b-only", action="store_true")
    first.add_argument("--route-waypoint-b-short-diagnostic", action="store_true")
    first.add_argument("--route-waypoint-b-extended-diagnostic", action="store_true")
    first.add_argument("--route-waypoint-b-post-locomotion-convergence", action="store_true")
    first.add_argument("--route-turn-qualification-evidence", default=None)
    first.add_argument(
        "--route-config",
        default=None,
        help="Local01 odom-frame START/WAYPOINT/TABLE_DOCKING x/y/yaw pose JSON",
    )
    first.add_argument(
        "--short-scene-route",
        action="store_true",
        help="interpret --route-config as the two-target Local01 short diagnostic, not table docking",
    )
    first.add_argument(
        "--raicom-low-cruise-profile",
        action="store_true",
        help=(
            "permit only the evidence-backed 0.105 m/s short-route recovery profile; "
            "all route stop, posture, and root-speed gates remain unchanged"
        ),
    )
    first.add_argument(
        "--raicom-zero-forward-heading-acquisition",
        action="store_true",
        help=(
            "SIMULATION_ONLY native-timestep aligned TABLE_APPROACH profile: use a bounded "
            "zero-forward yaw command to acquire the source before forward transit"
        ),
    )
    first.add_argument(
        "--raicom-continuous-ten-second-transit",
        action="store_true",
        help=(
            "SIMULATION_ONLY native-timestep aligned TABLE_APPROACH profile: extend each "
            "continuous transit command to the existing 10 s M1 ceiling before the full stop gate"
        ),
    )
    first.add_argument(
        "--raicom-continuous-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY: keep one 50 Hz gait stream through the rolling WAYPOINT and stop only "
            "at TABLE_APPROACH, bounded by the existing 75 s aggregate route allowance"
        ),
    )
    first.add_argument(
        "--raicom-fixed-window-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY: replan only after each composite-stopped 5 s table-route window, "
            "then publish either straight or the existing Run65 fixed +/-0.075 rad/s arc for that full window"
        ),
    )
    first.add_argument(
        "--raicom-forward-gait-yaw-inversion",
        action="store_true",
        help=(
            "SIMULATION_ONLY native-timestep RAICOM profile: apply the evidence-backed inverse "
            "yaw mapping only while a forward gait is active"
        ),
    )
    first.add_argument(
        "--raicom-native-nominal-straight-gait-diagnostic",
        action="store_true",
        help=(
            "one native-timestep RAICOM 0.12 m/s straight-gait-only diagnostic; no yaw correction "
            "or table-route acceptance"
        ),
    )
    first.add_argument(
        "--table-approach-mission",
        action="store_true",
        help="execute only Local01 START_ALIGN→WAYPOINT→TABLE_APPROACH; never TABLE_DOCKING",
    )
    first.add_argument(
        "--aligned-start-table-approach",
        action="store_true",
        help=(
            "require the versioned SIMULATION_ONLY pre-physics yaw-zero scene; "
            "it does not qualify the original START yaw alignment"
        ),
    )
    first.add_argument(
        "--native-x2-timestep",
        action="store_true",
        help="require the parent-staged aligned scene whose top-level MuJoCo timestep matches native X2 at 0.001 s",
    )
    first.add_argument(
        "--raicom-native-mjdata-route-feedback",
        action="store_true",
        help=(
            "SIMULATION_ONLY: use the parent-staged read-only native MjData trace for route pose feedback, "
            "while retaining ROS MC/IMU/joint safety telemetry"
        ),
    )
    first.add_argument("--route-config-post-stand-observation", action="store_true")
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
        if not args.reset_gate_verified or not (
            args.maintainer_authorized
            or getattr(args, "maintainer_authorized_zero_only_activation", False)
            or getattr(args, "maintainer_authorized_activation_probe", False)
        ):
            print("BLOCKED: validation-only invocation must carry the production authorization flags", file=sys.stderr)
            return 2
        print(json.dumps({
            "schema": "robotsim.issue80.m1.cli-validation.v1",
            "status": "VALIDATED_NO_ROS_NO_RUNTIME",
            "attempt_id": args.attempt_id,
            "production_authorization_flags_present": True,
            "network_isolation": "NOT_EVALUATED; validation container is intentionally --network=none",
            "reset_or_static_initialization_and_standing": "NOT_EVALUATED",
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
