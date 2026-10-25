#!/usr/bin/env python3
"""Proposal-only translation from Issue 80 route intents to bounded M1 windows.

This module deliberately owns no ROS node, publisher, service client, Docker
process, or simulator. A future runtime adapter may pass each plan to the
existing single-publisher M1 runner after revalidating its live gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME,
    COMMAND_PUBLISHER_TOPIC,
    COMMAND_PUBLISHER_MESSAGE_TYPE,
    COMMAND_QOS_DEPTH,
    COMMAND_QOS_DURABILITY,
    COMMAND_QOS_RELIABILITY,
    MAX_AUTHORIZED_PULSE_S,
    PUBLISH_HZ,
    SOURCE_PRIORITY,
    SOURCE_TIMEOUT_MS,
    TURN_COMMAND_RAD_S,
    TURN_FORWARD_COMMAND_M_S,
    TURN_PULSE_S,
    VELOCITY_SOURCE_NAME,
    ZERO_STOP_MIN_MESSAGES,
    ZERO_STOP_MIN_S,
    point_goal_command_from_bearing_error,
)
from issue80_waypoint_route import (
    DirectiveKind, PoseEstimate, Profile, RouteDirective, RouteSpec, RouteStep,
    StopBehavior, YawPolicy,
)


MAX_FORWARD_M_S = 0.14
MIN_EFFECTIVE_FORWARD_M_S = 0.12
MAX_ANGULAR_RAD_S = 0.075
RUN73_BEARING_HEADING_GAIN = 0.8
MAX_INCREMENTAL_WINDOW_S = 1.0
MAX_ROUTE_ACTION_WINDOW_S = 5.0
ROUTE_WINDOW_ZERO_MARGIN_S = 0.02  # A2's timing-recovery margin before a 5 s M1 deadline.
WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M = 0.20
WAYPOINT_B_POSITION_TOLERANCE_M = 0.15
# Run d5df79e0 drifted 36.811 mm from arrival to settled pose; reserve 45 mm.
WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M = 0.045
WAYPOINT_B_MAX_WINDOWS = 2
WAYPOINT_B_WINDOW_MAX_S = 5.0
WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S = 10.0
WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS = 3
WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S = 15.0
# Episode A measured 48.370 mm maximum post-command excursion before the stable hold.
WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M = 0.055
WAYPOINT_B_CORRIDOR_MAX_FORWARD_M = 1.25
# Keep an additional 0.15 m before the existing corridor edge for stopping and drift.
WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M = 1.10
WAYPOINT_B_MAX_CROSS_TRACK_M = 0.05
MAX_POSE_AGE_S = 0.20
MAX_TRANSFORM_AGE_S = 0.50
ALLOWED_MODE_READBACKS = frozenset(("STAND_DEFAULT", "LOCOMOTION_DEFAULT"))
PHYSICAL_STOP_MIN_WINDOW_S = 1.90
PHYSICAL_STOP_MIN_SAMPLES = 20
PHYSICAL_STOP_MAX_SAMPLE_GAP_S = 0.15
PHYSICAL_STOP_MAX_STATE_AGE_S = 0.5


@dataclass(frozen=True)
class RuntimeReadiness:
    """Fresh evidence snapshot supplied by the existing M1 runner.

    Boolean fields are evidence inputs, not claims made by this proposal-only
    planner. In particular, source_mode_handshake_pass is required after the
    first activation window; an empty selected source is permitted only before
    that first command, after registration and the standing preflight pass.
    """

    observed_steady_s: float
    reset_gate_pass: bool
    standing_gate_pass: bool
    safety_gates_pass: bool
    source_registration_pass: bool
    action_source_setup_pass: bool
    source_mode_handshake_pass: bool
    telemetry_fresh: bool
    workspace_clearance_verified: bool
    corridor_clear: bool
    swept_path_clear: bool
    bounded_moving_turn_qualified: bool
    qualified_turn_directions: tuple[str, ...]
    turn_stop_margin_m: float | None
    turn_stop_margin_direction: str
    turn_stop_margin_evidence_id: str
    transport_isolated: bool
    publisher_identity_verified: bool
    mc_subscriber_identity_verified: bool
    qos_compatible: bool
    velocity_publisher_count: int
    mc_velocity_subscriber_count: int
    ros_domain_id: int
    expected_ros_domain_id: int
    mode_desc: str
    selected_source: str
    measured_displacement_guard_verified: bool = False
    source_name: str = VELOCITY_SOURCE_NAME
    action_source_name: str = ACTION_SOURCE_NAME
    command_topic: str = COMMAND_PUBLISHER_TOPIC
    publisher_reliability: str = COMMAND_QOS_RELIABILITY
    publisher_history: str = "keep_last"
    publisher_depth: int = COMMAND_QOS_DEPTH
    publisher_durability: str = COMMAND_QOS_DURABILITY


@dataclass(frozen=True)
class RouteActionWindow:
    """One bounded command proposal; it does not prove publication or motion."""

    window_id: int
    classification: str
    status: str
    route_id: str
    segment_id: str | None
    directive_kind: str
    source_name: str
    source_priority: int
    source_timeout_ms: int
    action_source_name: str
    topic: str
    message_type: str
    publish_hz: float
    created_steady_s: float
    forward_m_s: float
    lateral_m_s: float
    angular_rad_s: float
    maximum_duration_s: float
    zero_publish_minimum_s: float
    zero_publish_minimum_messages: int
    physical_stop_gate_required: bool
    measured_displacement_guard_required: bool
    activation_handshake_required: bool
    pose_stamp_ns: int | None
    pose_generation: str | None
    reason: str


class RouteMotionBridgeError(ValueError):
    """Invalid or out-of-order action-window lifecycle."""


def waypoint_bearing_angular_command(
    *,
    current_x_m: float,
    current_y_m: float,
    current_yaw_rad: float,
    target_x_m: float,
    target_y_m: float,
    maximum_angular_rad_s: float = MAX_ANGULAR_RAD_S,
    heading_gain: float = RUN73_BEARING_HEADING_GAIN,
) -> dict[str, float | bool]:
    """Recompute bounded Run73 line-of-sight steering from one fresh odom pose.

    The function intentionally takes the target in odom coordinates and does
    not retain state between calls. Callers can therefore invoke it for every
    fresh sample without holding a window-level angular command. Positive yaw
    follows the documented odom convention (counter-clockwise).
    """
    values = (
        current_x_m, current_y_m, current_yaw_rad, target_x_m, target_y_m,
        maximum_angular_rad_s, heading_gain,
    )
    try:
        x, y, yaw, target_x, target_y, maximum, gain = map(float, values)
    except (TypeError, ValueError) as exc:
        raise RouteMotionBridgeError("waypoint steering inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (x, y, yaw, target_x, target_y, maximum, gain)):
        raise RouteMotionBridgeError("waypoint steering inputs must be finite")
    if maximum <= 0.0 or gain < 0.0:
        raise RouteMotionBridgeError("waypoint steering limits are invalid")
    dx, dy = target_x - x, target_y - y
    if math.hypot(dx, dy) <= 1e-9:
        return {
            "bearing_error_rad": 0.0,
            "raw_angular_rad_s": 0.0,
            "angular_rad_s": 0.0,
            "saturated": False,
        }
    bearing_error = math.atan2(
        math.sin(math.atan2(dy, dx) - yaw),
        math.cos(math.atan2(dy, dx) - yaw),
    )
    raw = gain * bearing_error
    command = max(-maximum, min(maximum, raw))
    return {
        "bearing_error_rad": bearing_error,
        "raw_angular_rad_s": raw,
        "angular_rad_s": command,
        "saturated": abs(raw) > maximum,
    }


def active_locomotion_source_readbacks_valid(
    snapshot: Mapping[str, object],
    *,
    expected_source: str,
    maximum_age_s: float,
) -> bool:
    """Check current agreeing mode/source values with independent freshness bounds.

    GetMcAction and GetCurrentInputSource are asynchronous service readbacks.
    Their timestamps need not stay within the stricter startup-handshake skew
    once the handshake has been established, but each readback, CommonState,
    and each value must still be fresh and agree with the active source/mode.
    """
    try:
        now = float(snapshot["sample_monotonic_s"])
        maximum_age = float(maximum_age_s)
        readbacks = snapshot["readback_monotonic_s"]
        state_ages = snapshot["state_age_s"]
        mode_age = now - float(readbacks["get_mode"])
        source_age = now - float(readbacks["get_source"])
        common_age = float(state_ages["common"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not all(math.isfinite(value) for value in (now, maximum_age, mode_age, source_age, common_age))
        or maximum_age <= 0.0
        or not (0.0 <= mode_age <= maximum_age)
        or not (0.0 <= source_age <= maximum_age)
        or not (0.0 <= common_age <= maximum_age)
    ):
        return False
    return bool(
        snapshot.get("get_mode_header_code") == 0
        and snapshot.get("get_mode_status_value") == 100
        and snapshot.get("get_mode_desc") == "LOCOMOTION_DEFAULT"
        and snapshot.get("common_mode_status_value") == 100
        and snapshot.get("common_mode_desc") == "LOCOMOTION_DEFAULT"
        and snapshot.get("get_source_header_code") == 0
        and snapshot.get("get_source_name") == expected_source
        and snapshot.get("common_source_name") == expected_source
        and snapshot.get("get_mode_readback_fault") is None
        and snapshot.get("get_source_readback_fault") is None
    )


def fresh_stand_mode_readback_valid(
    snapshot: Mapping[str, object], *, maximum_age_s: float,
) -> bool:
    """Require fresh, positive STAND service/CommonState/FSM/body readbacks."""
    try:
        now = float(snapshot["sample_monotonic_s"])
        maximum_age = float(maximum_age_s)
        readback_time = float(snapshot["readback_monotonic_s"]["get_mode"])
        mode_age = now - readback_time
        common_age = float(snapshot["state_age_s"]["common"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not all(math.isfinite(value) for value in (now, maximum_age, mode_age, common_age))
        or maximum_age <= 0.0
        or not (0.0 <= mode_age <= maximum_age)
        or not (0.0 <= common_age <= maximum_age)
    ):
        return False
    return bool(
        snapshot.get("get_mode_header_code") == 0
        and snapshot.get("get_mode_status_value") == 100
        and snapshot.get("get_mode_desc") == "STAND_DEFAULT"
        and snapshot.get("common_mode_status_value") == 100
        and snapshot.get("common_mode_desc") == "STAND_DEFAULT"
        and snapshot.get("fsm_state") == 4
        and snapshot.get("body_status") == 1
        and snapshot.get("get_mode_readback_fault") is None
        and snapshot.get("readback_fault") is None
    )


def fresh_input_source_readback_valid(
    snapshot: Mapping[str, object],
    *,
    expected_source: str,
    maximum_age_s: float,
) -> bool:
    """Require a fresh selected-source service result agreeing with CommonState."""
    try:
        now = float(snapshot["sample_monotonic_s"])
        maximum_age = float(maximum_age_s)
        readback_time = float(snapshot["readback_monotonic_s"]["get_source"])
        source_age = now - readback_time
        common_age = float(snapshot["state_age_s"]["common"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not all(math.isfinite(value) for value in (now, maximum_age, source_age, common_age))
        or maximum_age <= 0.0
        or not (0.0 <= source_age <= maximum_age)
        or not (0.0 <= common_age <= maximum_age)
    ):
        return False
    return bool(
        snapshot.get("get_source_header_code") == 0
        and snapshot.get("get_source_name") == expected_source
        and snapshot.get("common_source_name") == expected_source
        and snapshot.get("get_source_readback_fault") is None
        and snapshot.get("readback_fault") is None
    )


@dataclass
class TransitSteeringGate:
    """Allow steering only after straight warmup and a same-sample MC handshake."""

    warmup_s: float = 0.5
    window_started_steady_s: float | None = None
    handshake_observed: bool = False
    handshake_observed_steady_s: float | None = None
    steering_enabled: bool = False
    fault: str | None = None

    def observe(
        self,
        steady_s: float,
        same_sample_handshake: bool,
        *,
        active_readbacks_valid: bool | None = None,
    ) -> dict[str, object]:
        try:
            now = float(steady_s)
            warmup = float(self.warmup_s)
        except (TypeError, ValueError) as exc:
            raise RouteMotionBridgeError("transit steering gate time must be numeric") from exc
        if not math.isfinite(now) or not math.isfinite(warmup) or warmup < 0.0:
            raise RouteMotionBridgeError("transit steering gate time must be finite and bounded")
        if self.window_started_steady_s is None:
            self.window_started_steady_s = now
        if now < self.window_started_steady_s:
            self.fault = "transit_steering_monotonic_time_regressed"
        current_readbacks_valid = (
            bool(same_sample_handshake)
            if active_readbacks_valid is None else bool(active_readbacks_valid)
        )
        if self.fault is None:
            if same_sample_handshake:
                if not self.handshake_observed:
                    self.handshake_observed = True
                    self.handshake_observed_steady_s = now
            elif self.handshake_observed:
                if not current_readbacks_valid:
                    self.fault = "transit_locomotion_source_mode_handshake_lost"
            elif now - self.window_started_steady_s >= warmup:
                self.fault = "transit_locomotion_source_mode_handshake_not_confirmed_after_0p5s_straight_warmup"
        elapsed = max(0.0, now - self.window_started_steady_s)
        self.steering_enabled = bool(
            self.fault is None and self.handshake_observed
            and current_readbacks_valid and elapsed >= warmup
        )
        return {
            "elapsed_s": elapsed,
            "same_sample_handshake": bool(same_sample_handshake),
            "active_readbacks_valid": current_readbacks_valid,
            "handshake_observed": self.handshake_observed,
            "handshake_observed_steady_s": self.handshake_observed_steady_s,
            "steering_enabled": self.steering_enabled,
            "fault": self.fault,
        }


def waypoint_b_window_budget_passes(
    window_count: int, *, short_diagnostic: bool, extended_diagnostic: bool = False
) -> bool:
    """Keep legacy two-window acceptance distinct from both diagnostics."""
    if short_diagnostic and extended_diagnostic:
        return False
    minimum = 1 if short_diagnostic or extended_diagnostic else WAYPOINT_B_MAX_WINDOWS
    maximum = (
        WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS
        if extended_diagnostic else WAYPOINT_B_MAX_WINDOWS
    )
    return (
        isinstance(window_count, int)
        and not isinstance(window_count, bool)
        and minimum <= window_count <= maximum
    )


def waypoint_b_arrival_passes(
    arrival_error_m: float | None,
    settled_error_m: float | None,
    *,
    tolerance_m: float,
) -> bool:
    """Require both trigger-time and post-stop position to remain inside tolerance."""
    values = (arrival_error_m, settled_error_m, tolerance_m)
    return (
        all(_finite_number(value) for value in values)
        and float(tolerance_m) > 0.0
        and float(arrival_error_m) <= float(tolerance_m)
        and float(settled_error_m) <= float(tolerance_m)
    )


def waypoint_b_arrival_after_verified_stop_passes(
    arrival_error_m: float | None,
    settled_error_m: float | None,
    *,
    tolerance_m: float,
    composite_stop_gate_passed: bool,
    bridge_rearmed: bool,
    fresh_stand_readback_passed: bool,
    source_readback_passed: bool,
    safety_gates_passed: bool,
) -> bool:
    """Accept arrival discovered after a bounded window already stopped and passed its full gate.

    This path is distinct from in-window goal-triggered stopping. It requires
    both the fresh pose at discovery and the final settled pose within the
    unchanged tolerance, plus the completed composite stop and all rearm
    readbacks. It avoids asking the route FSM for a second stop gate after the
    same already-verified stationary interval.
    """
    return bool(
        composite_stop_gate_passed
        and bridge_rearmed
        and fresh_stand_readback_passed
        and source_readback_passed
        and safety_gates_passed
        and waypoint_b_arrival_passes(
            arrival_error_m,
            settled_error_m,
            tolerance_m=tolerance_m,
        )
    )


def waypoint_b_arrival_trigger_radius_m(
    *, acceptance_tolerance_m: float, stop_reserve_m: float
) -> float:
    """Tighten the trigger inside, without changing, the final acceptance radius."""
    if (
        not _finite_number(acceptance_tolerance_m)
        or not _finite_number(stop_reserve_m)
        or float(acceptance_tolerance_m) <= 0.0
        or float(stop_reserve_m) < 0.0
        or float(stop_reserve_m) >= float(acceptance_tolerance_m)
    ):
        raise RouteMotionBridgeError("waypoint stop reserve must be finite and inside its acceptance tolerance")
    return float(acceptance_tolerance_m) - float(stop_reserve_m)


def waypoint_b_forward_guard_reached(
    forward_projection_m: float,
    *,
    guard_m: float = WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
) -> bool:
    """Fail closed before extended transit reaches the route-corridor edge."""
    if not _finite_number(forward_projection_m) or not _finite_number(guard_m) or float(guard_m) <= 0.0:
        raise RouteMotionBridgeError("extended waypoint-B progress guard inputs must be finite and positive")
    return float(forward_projection_m) >= float(guard_m)


def waypoint_b_cross_track_guard_reached(
    cross_track_m: float, *, limit_m: float = WAYPOINT_B_MAX_CROSS_TRACK_M
) -> bool:
    """Reject B transit after the already-declared centerline lateral bound is crossed."""
    if not _finite_number(cross_track_m) or not _finite_number(limit_m) or float(limit_m) <= 0.0:
        raise RouteMotionBridgeError("waypoint-B cross-track guard inputs must be finite and positive")
    return abs(float(cross_track_m)) > float(limit_m)


def measured_m1_nonzero_window_s(result: Mapping[str, object]) -> float:
    """Read the M1 helper's actual nonzero span without counting its stop tail.

    Linear windows expose the last successful nonzero publish offset; some
    turn helpers expose the pulse duration directly. ``pulse_loop_elapsed_s``
    includes post-deadline observation and is deliberately not a fallback.
    """
    for key in ("last_nonzero_command_offset_s", "pulse_actual_duration_s"):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value) and value > 0.0:
                return value
    raise RouteMotionBridgeError("M1 result lacks a positive measured nonzero command duration")


def build_short_right_corner_route(
    *,
    route_id: str,
    start_pose: PoseEstimate,
    robot_footprint_radius_m: float,
    local_clear_radius_m: float,
    right_turn_qualified: bool,
    waypoint_b_only: bool = False,
    waypoint_b_distance_m: float = 0.35,
) -> RouteSpec:
    """Build the bounded transit/corner/continuation geometry from a measured start.

    waypoint_b_only builds a one-step route for the first waypoint milestone.
    The full three-step route remains available for a later separately authorized
    B -> right corner -> C trial.
    """
    if (
        not route_id
        or start_pose.frame_id != "odom"
        or start_pose.child_frame_id != "base_link"
        or not all(_finite_number(value) for value in (
            start_pose.x_m, start_pose.y_m, start_pose.yaw_rad,
            robot_footprint_radius_m, local_clear_radius_m, waypoint_b_distance_m,
        ))
        or robot_footprint_radius_m <= 0.0
        or local_clear_radius_m <= robot_footprint_radius_m + 0.10
        or waypoint_b_distance_m <= 0.15
        or waypoint_b_distance_m > 0.75
        or (not waypoint_b_only and not right_turn_qualified)
    ):
        raise ValueError("measured start pose or verified open-floor clearance is invalid")

    # The caller supplies either the original waypoint-B acceptance target or
    # a separately labeled short diagnostic target. The full-route C target
    # remains 0.45 m beyond B.
    transit_to_b_m = waypoint_b_distance_m
    waypoint_b_to_c_m = 0.45
    right_turn_delta_rad = -0.12  # A2 measured -0.122523 rad; stay inside its qualified arc.
    boundary_margin_m = 0.10
    maximum_turn_displacement_m = 0.25
    lateral_half_width_m = robot_footprint_radius_m + boundary_margin_m + 0.05
    turn_angle_magnitude = abs(right_turn_delta_rad)
    turn_lateral_envelope_m = (
        TURN_FORWARD_COMMAND_M_S / TURN_COMMAND_RAD_S
        * (1.0 - math.cos(TURN_COMMAND_RAD_S * TURN_PULSE_S))
    )
    turn_lateral_half_width_m = (
        robot_footprint_radius_m + boundary_margin_m + turn_lateral_envelope_m + 0.05
    )
    maximum_local_radius = math.hypot(WAYPOINT_B_CORRIDOR_MAX_FORWARD_M, turn_lateral_half_width_m)
    if maximum_local_radius >= local_clear_radius_m:
        raise ValueError("verified open-floor region does not contain the swept route corridor")

    corner_yaw = math.atan2(
        math.sin(start_pose.yaw_rad - turn_angle_magnitude),
        math.cos(start_pose.yaw_rad - turn_angle_magnitude),
    )
    transit_x = start_pose.x_m + transit_to_b_m * math.cos(start_pose.yaw_rad)
    transit_y = start_pose.y_m + transit_to_b_m * math.sin(start_pose.yaw_rad)
    next_x = transit_x + waypoint_b_to_c_m * math.cos(corner_yaw)
    next_y = transit_y + waypoint_b_to_c_m * math.sin(corner_yaw)

    def corridor(
        center_x_m: float,
        center_y_m: float,
        heading_rad: float,
        lateral_half_width: float,
    ) -> tuple[tuple[float, float], ...]:
        def point(along_m: float, lateral_m: float) -> tuple[float, float]:
            return (
                center_x_m + along_m * math.cos(heading_rad) - lateral_m * math.sin(heading_rad),
                center_y_m + along_m * math.sin(heading_rad) + lateral_m * math.cos(heading_rad),
            )

        return tuple(point(along, lateral) for along, lateral in (
            (-0.75, -lateral_half_width),
            (WAYPOINT_B_CORRIDOR_MAX_FORWARD_M, -lateral_half_width),
            (WAYPOINT_B_CORRIDOR_MAX_FORWARD_M, lateral_half_width),
            (-0.75, lateral_half_width),
        ))

    approach_corridor = corridor(
        start_pose.x_m, start_pose.y_m, start_pose.yaw_rad, lateral_half_width_m
    )
    turn_corridor = corridor(
        transit_x, transit_y, start_pose.yaw_rad, turn_lateral_half_width_m
    )
    continuation_corridor = corridor(
        transit_x, transit_y, corner_yaw, lateral_half_width_m
    )

    def step(
        waypoint_id: str,
        x_m: float,
        y_m: float,
        profile: Profile,
        position_tolerance_m: float,
        yaw_policy: YawPolicy,
        yaw_tolerance_rad: float | None,
        polygon: tuple[tuple[float, float], ...],
        *,
        is_corner: bool = False,
    ) -> RouteStep:
        return RouteStep(
            waypoint_id=waypoint_id,
            x_m=x_m,
            y_m=y_m,
            profile=profile,
            position_tolerance_m=position_tolerance_m,
            yaw_policy=yaw_policy,
            yaw_rad=None,
            yaw_tolerance_rad=yaw_tolerance_rad,
            max_position_std_m=0.01,
            max_yaw_std_rad=0.05,
            stop_behavior=StopBehavior.VERIFY_PHYSICAL_STOP,
            segment_timeout_s=20.0 if is_corner else 60.0,
            recovery_limit=0,
            corridor_polygon=polygon,
            turn_corridor_polygon=turn_corridor if is_corner else None,
            minimum_obstacle_clearance_m=local_clear_radius_m,
            target_forward_m_s=0.12,
            max_forward_m_s=0.12,
            max_angular_rad_s=MAX_ANGULAR_RAD_S,
            max_command_window_s=5.0 if is_corner else 10.0,
            turn_strategy="BOUNDED_MOVING_TURN" if is_corner else None,
            maximum_turn_displacement_m=maximum_turn_displacement_m if is_corner else None,
        )

    steps = (
        step("TRANSIT-TO-CORNER", transit_x, transit_y, Profile.TRANSIT, 0.15,
             YawPolicy.IGNORE, None, approach_corridor),
    ) if waypoint_b_only else (
        step("TRANSIT-TO-CORNER", transit_x, transit_y, Profile.TRANSIT, 0.15,
             YawPolicy.IGNORE, None, approach_corridor),
        step("RIGHT-CORNER", transit_x, transit_y, Profile.CORNER, 0.20,
             YawPolicy.NEXT_SEGMENT, 0.03, approach_corridor, is_corner=True),
        step("TRANSIT-AFTER-RIGHT-CORNER", next_x, next_y, Profile.TRANSIT, 0.15,
             YawPolicy.IGNORE, None, continuation_corridor),
    )
    route = RouteSpec(
        route_id=route_id,
        frame_id="odom",
        workspace_verified=True,
        bounded_moving_turn_qualified=right_turn_qualified if not waypoint_b_only else False,
        zero_forward_turn_qualified=False,
        start_x_m=start_pose.x_m,
        start_y_m=start_pose.y_m,
        start_yaw_rad=start_pose.yaw_rad,
        start_position_tolerance_m=0.02,
        start_yaw_tolerance_rad=0.02,
        robot_footprint_radius_m=robot_footprint_radius_m,
        corridor_boundary_margin_m=boundary_margin_m,
        max_pose_age_s=0.20,
        max_transform_age_s=0.50,
        max_safety_age_s=0.50,
        max_position_std_m=0.01,
        max_yaw_std_rad=0.05,
        steps=steps,
    )
    route.validate()
    return route


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class RouteMotionBridge:
    """Translate one measured-pose route intent at a time into M1 proposals.

    plan_next is pure with respect to the robot. The caller must execute a
    nonzero proposal only through the already-reviewed M1 runner, preserve its
    source/mode handshake, and supply fresh state before requesting another
    window. A plan never means that a ROS message was published or received.
    """

    def __init__(
        self,
        *,
        route_frame_id: str,
        incremental_window_s: float = MAX_INCREMENTAL_WINDOW_S,
    ) -> None:
        if route_frame_id not in ("odom", "map"):
            raise ValueError("route frame must be odom or map")
        if not math.isfinite(incremental_window_s) or not 0.0 < incremental_window_s <= MAX_ROUTE_ACTION_WINDOW_S:
            raise ValueError("incremental window must be in (0, 5.0] seconds")
        self.route_frame_id = route_frame_id
        self.incremental_window_s = incremental_window_s
        self._next_window_id = 1
        self._active_plan: RouteActionWindow | None = None
        self._active_key: tuple[str, str, str] | None = None
        self._elapsed_by_key: dict[tuple[str, str, str], float] = {}
        self._turn_origin_by_key: dict[tuple[str, str, str], tuple[float, float]] = {}
        self._first_activation_window_issued = False
        self._awaiting_physical_stop = False
        self._last_pose_stamp_ns: int | None = None
        self._pose_generation: str | None = None

    def plan_next(
        self,
        directive: RouteDirective,
        pose: PoseEstimate,
        readiness: RuntimeReadiness,
        *,
        now_steady_s: float,
    ) -> RouteActionWindow:
        """Create one plan using current pose/readiness; non-motion intents are zero-only."""
        if self._active_plan is not None:
            raise RouteMotionBridgeError("previous action window has not been completed")
        if directive.kind in {
            DirectiveKind.REQUEST_PHYSICAL_STOP,
            DirectiveKind.STOP_AND_ABORT,
            DirectiveKind.DOCKING_HANDOFF,
            DirectiveKind.WAIT,
            DirectiveKind.COMPLETE,
        }:
            return self.zero_window(
                directive,
                now_steady_s=now_steady_s,
                reason=directive.reason or directive.kind.value,
                physical_stop_gate_required=True,
            )

        if directive.kind not in {DirectiveKind.FOLLOW_WAYPOINT, DirectiveKind.BOUNDED_MOVING_TURN}:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="unsupported route directive", physical_stop_gate_required=True,
            )

        blocker = self._readiness_blocker(readiness, now_steady_s)
        if blocker:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason=blocker, physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )
        pose_error = self._pose_error(pose, now_steady_s)
        if pose_error:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason=pose_error, physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )
        if (
            readiness.workspace_clearance_verified is not True
            or readiness.corridor_clear is not True
            or readiness.swept_path_clear is not True
        ):
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="workspace, corridor, or proposed swept-path clearance is unverified",
                physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )
        if self._first_activation_window_issued:
            if (
                not readiness.source_mode_handshake_pass
                or readiness.selected_source != VELOCITY_SOURCE_NAME
                or readiness.mode_desc != "LOCOMOTION_DEFAULT"
            ):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="fresh same-sample active source/LOCOMOTION handshake is required after activation",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )

        if directive.kind is DirectiveKind.BOUNDED_MOVING_TURN:
            if (
                not _finite_number(directive.goal_yaw_rad)
                or not _finite_number(directive.yaw_tolerance_rad)
                or directive.yaw_tolerance_rad <= 0.0
            ):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="moving-turn directive lacks a target yaw or tolerance",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            heading_error = _wrap_angle(directive.goal_yaw_rad - pose.yaw_rad)
            if abs(heading_error) <= directive.yaw_tolerance_rad:
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="measured turn target reached; physical stop still required",
                    physical_stop_gate_required=True, status="TURN_TARGET_REACHED_ZERO_ONLY",
                )
            turn_direction = "RIGHT" if heading_error < 0.0 else "LEFT"
            if (
                readiness.bounded_moving_turn_qualified is not True
                or turn_direction not in readiness.qualified_turn_directions
            ):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason=f"{turn_direction.lower()} moving-turn capability is not qualified",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            if (
                readiness.turn_stop_margin_direction != turn_direction
                or not _finite_number(readiness.turn_stop_margin_m)
                or readiness.turn_stop_margin_m <= 0.0
                or not isinstance(readiness.turn_stop_margin_evidence_id, str)
                or not readiness.turn_stop_margin_evidence_id
            ):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason=f"{turn_direction.lower()} turn lacks its own evidence-backed stopping-distance reserve",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            if readiness.measured_displacement_guard_verified is not True:
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="moving turn requires a verified per-sample measured-displacement stop guard",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
        else:
            if not _finite_number(directive.goal_x_m) or not _finite_number(directive.goal_y_m):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="waypoint directive lacks a target position",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            dx = directive.goal_x_m - pose.x_m
            dy = directive.goal_y_m - pose.y_m
            distance = math.hypot(dx, dy)
            tolerance = directive.position_tolerance_m
            if not _finite_number(tolerance) or tolerance <= 0.0:
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="waypoint directive lacks a positive position tolerance",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            if distance <= tolerance and directive.goal_yaw_rad is None:
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="measured waypoint position reached",
                    physical_stop_gate_required=False,
                    status="WAYPOINT_REACHED_ZERO_ONLY",
                )
            heading_target = (
                directive.goal_yaw_rad
                if distance <= tolerance and directive.goal_yaw_rad is not None
                else math.atan2(dy, dx)
            )
            heading_error = _wrap_angle(heading_target - pose.yaw_rad)

        speed_error = self._speed_error(directive)
        if speed_error:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason=speed_error, physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )
        if (
            not _finite_number(directive.max_command_window_s)
            or not 0.0 < directive.max_command_window_s <= MAX_AUTHORIZED_PULSE_S
        ):
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="directive command window is outside the reviewed M1 action limit",
                physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )
        if (
            not _finite_number(directive.max_angular_rad_s)
            or directive.max_angular_rad_s <= 0.0
            or directive.max_angular_rad_s > MAX_ANGULAR_RAD_S
        ):
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="directive angular cap is outside the reviewed M1 limit",
                physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
            )

        key = (directive.route_id, directive.segment_id or "", directive.kind.value)
        if self._active_key != key:
            self._active_key = key
        if directive.kind is DirectiveKind.BOUNDED_MOVING_TURN:
            if (
                not _finite_number(directive.maximum_turn_displacement_m)
                or directive.maximum_turn_displacement_m <= 0.0
            ):
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="moving turn lacks a positive measured-displacement budget",
                    physical_stop_gate_required=True, status="BLOCKED_ZERO_ONLY",
                )
            origin = self._turn_origin_by_key.setdefault(key, (pose.x_m, pose.y_m))
            displacement = math.hypot(pose.x_m - origin[0], pose.y_m - origin[1])
            remaining_displacement = (
                directive.maximum_turn_displacement_m
                - displacement
                - float(readiness.turn_stop_margin_m)
            )
            if remaining_displacement <= 1e-6:
                return self.zero_window(
                    directive, now_steady_s=now_steady_s,
                    reason="measured turn displacement budget reached; physical stop remains unverified",
                    physical_stop_gate_required=True, status="TURN_DISPLACEMENT_BUDGET_REACHED_ZERO_ONLY",
                )
        else:
            remaining_displacement = math.inf

        elapsed = self._elapsed_by_key.get(key, 0.0)
        remaining_window = directive.max_command_window_s - elapsed
        if remaining_window <= 1e-9:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="directive command-window budget exhausted; route must be re-evaluated",
                physical_stop_gate_required=True, status="WINDOW_BUDGET_EXHAUSTED_ZERO_ONLY",
            )
        maximum_duration = min(self.incremental_window_s, remaining_window)
        # Moving turns rely on the consuming M1 runner's continuously sampled
        # displacement cutoff; the proposal must not turn a measured guard into
        # a timer estimate derived from commanded speed.
        if maximum_duration <= 1e-6:
            return self.zero_window(
                directive, now_steady_s=now_steady_s,
                reason="no bounded command slice remains inside the measured turn displacement budget",
                physical_stop_gate_required=True, status="TURN_DISPLACEMENT_MARGIN_ZERO_ONLY",
            )

        _, angular = point_goal_command_from_bearing_error(
            heading_error,
            cruise_forward_m_s=directive.target_forward_m_s,
            steering_forward_m_s=directive.target_forward_m_s,
            heading_gain=0.8,
            maximum_angular_rad_s=directive.max_angular_rad_s,
            steering_threshold_rad=0.0,
        )
        window = RouteActionWindow(
            window_id=self._next_window_id,
            classification="SIMULATION_ONLY_ROUTE_ACTION_PROPOSAL",
            status="PROPOSAL_ONLY",
            route_id=directive.route_id,
            segment_id=directive.segment_id,
            directive_kind=directive.kind.value,
            source_name=VELOCITY_SOURCE_NAME,
            source_priority=SOURCE_PRIORITY,
            source_timeout_ms=SOURCE_TIMEOUT_MS,
            action_source_name=ACTION_SOURCE_NAME,
            topic=COMMAND_PUBLISHER_TOPIC,
            message_type=COMMAND_PUBLISHER_MESSAGE_TYPE,
            publish_hz=PUBLISH_HZ,
            created_steady_s=now_steady_s,
            forward_m_s=directive.target_forward_m_s,
            lateral_m_s=0.0,
            angular_rad_s=angular,
            maximum_duration_s=maximum_duration,
            zero_publish_minimum_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0),
            zero_publish_minimum_messages=ZERO_STOP_MIN_MESSAGES,
            physical_stop_gate_required=False,
            measured_displacement_guard_required=directive.kind is DirectiveKind.BOUNDED_MOVING_TURN,
            activation_handshake_required=not self._first_activation_window_issued,
            pose_stamp_ns=pose.stamp_ns,
            pose_generation=pose.generation,
            reason=(
                "bounded proposal from fresh measured pose; caller must use the existing M1 publisher and recheck state before the next window"
            ),
        )
        self._next_window_id += 1
        self._active_plan = window
        self._first_activation_window_issued = True
        self._last_pose_stamp_ns = pose.stamp_ns
        self._pose_generation = pose.generation
        return window

    def complete_window(self, window_id: int, *, actual_elapsed_s: float) -> None:
        """Close a proposal window after caller-side execution/observation returns.

        The elapsed value is accounting only. This method does not verify that
        a publisher sent, MC received, or the robot moved.
        """
        plan = self._active_plan
        if plan is None or plan.window_id != window_id:
            raise RouteMotionBridgeError("window completion does not match the active plan")
        if (
            not math.isfinite(actual_elapsed_s)
            or actual_elapsed_s <= 0.0
            or actual_elapsed_s > plan.maximum_duration_s + 1e-9
        ):
            raise RouteMotionBridgeError("actual window duration is invalid or exceeded its cap")
        key = (plan.route_id, plan.segment_id or "", plan.directive_kind)
        self._elapsed_by_key[key] = self._elapsed_by_key.get(key, 0.0) + actual_elapsed_s
        self._active_plan = None

    def confirm_physical_stop(
        self,
        evidence: Mapping[str, float | int | str],
        *,
        zero_request_started_s: float,
        now_steady_s: float,
        expected_generation: str,
    ) -> bool:
        """Rearm one new activation only after the existing physical-stop gate passes."""
        if not self._awaiting_physical_stop:
            return False
        if not validate_physical_stop_evidence(
            evidence,
            zero_request_started_s=zero_request_started_s,
            now_steady_s=now_steady_s,
            expected_generation=expected_generation,
        ):
            return False
        self._awaiting_physical_stop = False
        self._first_activation_window_issued = False
        self._active_key = None
        return True

    def zero_window(
        self,
        directive: RouteDirective,
        *,
        now_steady_s: float,
        reason: str,
        physical_stop_gate_required: bool = True,
        status: str = "ZERO_ONLY_PROPOSAL",
    ) -> RouteActionWindow:
        """Produce an emergency/cancel zero plan without claiming stop delivery."""
        if not _finite_number(now_steady_s):
            raise RouteMotionBridgeError("zero-plan timestamp must be finite")
        # A zero plan invalidates any uncompleted nonzero proposal. The actual
        # runtime must serialize it through the same sole, verified publisher.
        self._active_plan = None
        if physical_stop_gate_required:
            self._awaiting_physical_stop = True
        plan = RouteActionWindow(
            window_id=self._next_window_id,
            classification="SIMULATION_ONLY_ROUTE_ACTION_PROPOSAL",
            status=status,
            route_id=directive.route_id,
            segment_id=directive.segment_id,
            directive_kind=directive.kind.value,
            source_name=VELOCITY_SOURCE_NAME,
            source_priority=SOURCE_PRIORITY,
            source_timeout_ms=SOURCE_TIMEOUT_MS,
            action_source_name=ACTION_SOURCE_NAME,
            topic=COMMAND_PUBLISHER_TOPIC,
            message_type=COMMAND_PUBLISHER_MESSAGE_TYPE,
            publish_hz=PUBLISH_HZ,
            created_steady_s=now_steady_s,
            forward_m_s=0.0,
            lateral_m_s=0.0,
            angular_rad_s=0.0,
            maximum_duration_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0) + 1.0 / PUBLISH_HZ,
            zero_publish_minimum_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0),
            zero_publish_minimum_messages=ZERO_STOP_MIN_MESSAGES,
            physical_stop_gate_required=physical_stop_gate_required,
            measured_displacement_guard_required=False,
            activation_handshake_required=False,
            pose_stamp_ns=None,
            pose_generation=None,
            reason=reason + "; zero publication proposal is not a physical-stop pass",
        )
        self._next_window_id += 1
        return plan

    def _readiness_blocker(self, readiness: RuntimeReadiness, now: float) -> str | None:
        if self._awaiting_physical_stop:
            return "a new route action is blocked until measured physical-stop evidence passes"
        if not _finite_number(now) or not _finite_number(readiness.observed_steady_s):
            return "runtime readiness timestamp is invalid"
        age = now - readiness.observed_steady_s
        if age < 0.0 or age > MAX_POSE_AGE_S:
            return "runtime readiness snapshot is stale or from the future"
        required = (
            (readiness.reset_gate_pass, "Reset gate did not pass"),
            (readiness.standing_gate_pass, "standing gate did not pass"),
            (readiness.safety_gates_pass, "M1 physical safety gates did not pass"),
            (readiness.source_registration_pass, "locomotion source registration is unverified"),
            (readiness.action_source_setup_pass, "M1 action-source setup is unverified"),
            (readiness.telemetry_fresh, "critical telemetry is stale"),
            (readiness.transport_isolated, "ROS transport is not isolated"),
            (readiness.publisher_identity_verified, "authorized velocity publisher identity is unverified"),
            (readiness.mc_subscriber_identity_verified, "MC velocity subscriber identity is unverified"),
            (readiness.qos_compatible, "velocity endpoint QoS is not verified compatible"),
        )
        for passed, message in required:
            if passed is not True:
                return message
        if type(readiness.velocity_publisher_count) is not int or readiness.velocity_publisher_count != 1:
            return "exactly one authorized velocity publisher is required"
        if type(readiness.mc_velocity_subscriber_count) is not int or readiness.mc_velocity_subscriber_count != 1:
            return "exactly one verified MC velocity subscriber is required"
        if type(readiness.ros_domain_id) is not int or type(readiness.expected_ros_domain_id) is not int:
            return "ROS domain identity is malformed"
        if readiness.ros_domain_id != readiness.expected_ros_domain_id:
            return "ROS domain does not match the isolated M1 runtime"
        if readiness.source_name != VELOCITY_SOURCE_NAME or readiness.action_source_name != ACTION_SOURCE_NAME:
            return "M1 source/action identity differs from the reviewed runtime"
        if readiness.command_topic != COMMAND_PUBLISHER_TOPIC:
            return "velocity topic differs from the reviewed M1 endpoint"
        if (
            readiness.publisher_reliability != COMMAND_QOS_RELIABILITY
            or readiness.publisher_history != "keep_last"
            or readiness.publisher_depth != COMMAND_QOS_DEPTH
            or readiness.publisher_durability != COMMAND_QOS_DURABILITY
        ):
            return "local M1 publisher QoS differs from the verified profile"
        if not isinstance(readiness.mode_desc, str) or readiness.mode_desc not in ALLOWED_MODE_READBACKS:
            return "mode readback is missing, unknown, or outside the active locomotion contract"
        if readiness.selected_source not in ("", VELOCITY_SOURCE_NAME):
            return "a conflicting input source is selected"
        if readiness.source_mode_handshake_pass is True and (
            readiness.mode_desc != "LOCOMOTION_DEFAULT"
            or readiness.selected_source != VELOCITY_SOURCE_NAME
        ):
            return "source/mode handshake flag disagrees with its mode or selected-source values"
        if self._first_activation_window_issued and readiness.source_mode_handshake_pass is not True:
            return "source/mode activation handshake is not confirmed"
        if self._first_activation_window_issued and readiness.mode_desc != "LOCOMOTION_DEFAULT":
            return "active windows require LOCOMOTION_DEFAULT readback"
        if self._first_activation_window_issued and readiness.selected_source != VELOCITY_SOURCE_NAME:
            return "active windows require the expected selected source"
        return None

    def _pose_error(self, pose: PoseEstimate, now: float) -> str | None:
        values = (pose.x_m, pose.y_m, pose.yaw_rad, pose.received_steady_s, pose.transform_age_s)
        if not all(_finite_number(v) for v in values):
            return "measured pose contains nonfinite values"
        if pose.frame_id != self.route_frame_id or pose.child_frame_id != "base_link":
            return "measured pose frame does not match the route frame/base_link contract"
        expected_chain = ("odom->base_link",) if self.route_frame_id == "odom" else ("map->odom", "odom->base_link")
        if tuple(pose.transform_chain) != expected_chain:
            return "measured pose transform chain is incomplete or unexpected"
        age = now - pose.received_steady_s
        if age < 0.0 or age > MAX_POSE_AGE_S:
            return "measured pose is stale or from the future"
        if pose.transform_age_s < 0.0 or pose.transform_age_s > MAX_TRANSFORM_AGE_S:
            return "measured pose transform is stale"
        if not isinstance(pose.stamp_ns, int) or pose.stamp_ns <= 0 or not pose.generation or not pose.source:
            return "measured pose lacks a valid stamp, generation, or source"
        if self._pose_generation is not None and pose.generation != self._pose_generation:
            return "pose generation changed; route must stop and reinitialize"
        if self._last_pose_stamp_ns is not None and pose.stamp_ns <= self._last_pose_stamp_ns:
            return "pose stamp is not strictly increasing"
        if not all(_finite_number(v) and v >= 0.0 for v in (pose.position_std_m, pose.yaw_std_rad)):
            return "measured pose uncertainty is invalid"
        return None

    @staticmethod
    def _speed_error(directive: RouteDirective) -> str | None:
        values = (directive.target_forward_m_s, directive.max_forward_m_s, directive.max_angular_rad_s)
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            return "directive velocity bounds are nonfinite"
        target, maximum, angular = (float(value) for value in values)
        if not MIN_EFFECTIVE_FORWARD_M_S <= target <= maximum:
            return "forward target is below the measured effective command floor or above its cap"
        if maximum > MAX_FORWARD_M_S:
            return "forward cap exceeds the reviewed 0.14 m/s route profile"
        if angular <= 0.0 or angular > MAX_ANGULAR_RAD_S:
            return "angular cap exceeds the reviewed 0.075 rad/s M1 limit"
        return None


def validate_physical_stop_evidence(
    evidence: Mapping[str, float | int | str],
    *,
    zero_request_started_s: float,
    now_steady_s: float,
    expected_generation: str,
) -> bool:
    """Recheck the route executor's existing measured-stop evidence contract."""
    try:
        start = float(evidence["window_start_steady_s"])
        end = float(evidence["window_end_steady_s"])
        samples = int(evidence["sample_count"])
        max_gap = float(evidence["maximum_sample_gap_s"])
        max_age = float(evidence["maximum_state_age_s"])
        generation = str(evidence["generation"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (start, end, max_gap, max_age, zero_request_started_s, now_steady_s)
    return (
        evidence.get("status") == "PASS"
        and all(math.isfinite(value) for value in values)
        and generation == expected_generation
        and start >= zero_request_started_s
        and end >= start
        and end <= now_steady_s + 1e-9
        and now_steady_s - end <= 0.5
        and end - start >= PHYSICAL_STOP_MIN_WINDOW_S
        and samples >= PHYSICAL_STOP_MIN_SAMPLES
        and max_gap <= PHYSICAL_STOP_MAX_SAMPLE_GAP_S
        and max_age <= PHYSICAL_STOP_MAX_STATE_AGE_S
    )
