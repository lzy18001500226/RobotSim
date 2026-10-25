#!/usr/bin/env python3
"""Offline, fail-closed waypoint route manager prototype for Issue 80.

This module has no ROS, MuJoCo, MC, or hardware dependency.  It plans and
supervises route intents from injected pose/safety providers; it never
publishes velocity commands.  A future motion adapter must implement the
protocols below and independently preserve the existing source/mode, bounded
command, zero-stop, and measured physical-stop gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "robotsim.issue80.waypoint-route.v1"
WAYPOINT_DOCKING_POSE_SCHEMA_VERSION = "robotsim.issue80.waypoint-docking-poses.v1"
MAX_FORWARD_M_S = 0.14
MAX_ANGULAR_RAD_S = 0.075
MIN_EFFECTIVE_FORWARD_M_S = 0.12
CONFIGURED_ROUTE_MAX_CROSS_TRACK_M = 0.05
MAX_ACTION_WINDOW_S = 10.0
REALTIME_CLOCK_ALIGNMENT_TOLERANCE_NS = 50_000_000
REALTIME_CLOCK_BACKSTEP_MIN_NS = 50_000_000
REALTIME_CLOCK_BACKSTEP_MAX_NS = 2_000_000_000


class RouteError(ValueError):
    """Route configuration or provider data is invalid."""


class Profile(str, Enum):
    TRANSIT = "TRANSIT"
    CORNER = "CORNER"
    DOCKING = "DOCKING"


@dataclass(frozen=True)
class ProfilePolicy:
    """Initial task-level defaults; route/workspace review may tighten them."""

    position_tolerance_m: float
    yaw_policy: str
    yaw_tolerance_rad: float | None
    max_position_std_m: float
    max_yaw_std_rad: float
    target_forward_m_s: float
    max_forward_m_s: float
    max_angular_rad_s: float
    max_command_window_s: float
    verify_physical_stop: bool
    max_corrections: int


PROPOSED_PROFILE_POLICIES: Mapping[Profile, ProfilePolicy] = {
    Profile.TRANSIT: ProfilePolicy(
        position_tolerance_m=0.20, yaw_policy="IGNORE", yaw_tolerance_rad=None,
        max_position_std_m=0.10, max_yaw_std_rad=0.20,
        target_forward_m_s=0.14, max_forward_m_s=0.14,
        max_angular_rad_s=0.075, max_command_window_s=10.0,
        verify_physical_stop=False, max_corrections=1,
    ),
    Profile.CORNER: ProfilePolicy(
        position_tolerance_m=0.25, yaw_policy="NEXT_SEGMENT", yaw_tolerance_rad=0.26,
        max_position_std_m=0.10, max_yaw_std_rad=0.20,
        target_forward_m_s=0.12, max_forward_m_s=0.12,
        max_angular_rad_s=0.075, max_command_window_s=5.0,
        verify_physical_stop=True, max_corrections=0,
    ),
    Profile.DOCKING: ProfilePolicy(
        position_tolerance_m=0.08, yaw_policy="EXPLICIT", yaw_tolerance_rad=0.10,
        max_position_std_m=0.025, max_yaw_std_rad=0.05,
        target_forward_m_s=0.12, max_forward_m_s=0.12,
        max_angular_rad_s=0.075, max_command_window_s=10.0,
        verify_physical_stop=True, max_corrections=2,
    ),
}


class YawPolicy(str, Enum):
    IGNORE = "IGNORE"
    EXPLICIT = "EXPLICIT"
    NEXT_SEGMENT = "NEXT_SEGMENT"


class StopBehavior(str, Enum):
    ROLLING = "ROLLING"
    VERIFY_PHYSICAL_STOP = "VERIFY_PHYSICAL_STOP"


class RouteState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    WAITING_FOR_DOCKING = "WAITING_FOR_DOCKING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


class RoutePhase(str, Enum):
    FOLLOW = "FOLLOW"
    TURN = "TURN"
    STOP = "STOP"
    RECOVERY_STOP = "RECOVERY_STOP"
    HANDOFF = "HANDOFF"


class DirectiveKind(str, Enum):
    FOLLOW_WAYPOINT = "FOLLOW_WAYPOINT"
    BOUNDED_MOVING_TURN = "BOUNDED_MOVING_TURN"
    REQUEST_PHYSICAL_STOP = "REQUEST_PHYSICAL_STOP"
    DOCKING_HANDOFF = "DOCKING_HANDOFF"
    WAIT = "WAIT"
    COMPLETE = "COMPLETE"
    STOP_AND_ABORT = "STOP_AND_ABORT"


class MotionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ARRIVED = "ARRIVED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    SAFETY_FAILURE = "SAFETY_FAILURE"


class PhysicalStopStatus(str, Enum):
    NOT_CHECKED = "NOT_CHECKED"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class PhysicalStopEvidence:
    """Fresh summary from the existing measured physical-stop gate."""

    status: PhysicalStopStatus
    window_start_steady_s: float
    window_end_steady_s: float
    sample_count: int
    maximum_sample_gap_s: float
    maximum_state_age_s: float
    generation: str


@dataclass(frozen=True)
class PoseEstimate:
    """Measured planar base pose expressed in the requested route frame."""

    frame_id: str
    child_frame_id: str
    x_m: float
    y_m: float
    yaw_rad: float
    stamp_ns: int
    received_steady_s: float
    generation: str
    source: str
    position_std_m: float
    yaw_std_rad: float
    transform_chain: tuple[str, ...]
    transform_age_s: float
    wall_time_ns: int | None = None


@dataclass(frozen=True)
class SafetyObservation:
    """Fresh corridor/obstacle assessment from a separately owned provider.

    minimum_obstacle_clearance_m must already account for the robot footprint;
    robot_safe carries the existing posture/joint/safety-gate result.
    """

    segment_id: str
    frame_id: str
    received_steady_s: float
    clear: bool
    robot_safe: bool
    minimum_obstacle_clearance_m: float
    reason: str = ""


@dataclass(frozen=True)
class DockingReadiness:
    """Independent gates required before manipulation may take control."""

    object_or_target_pose_frame_id: str | None
    object_or_target_pose_age_s: float | None
    object_or_target_pose_confidence: float
    arm_ik_reachable: bool
    manipulator_ready: bool

    @property
    def ready(self) -> bool:
        return (
            self.object_or_target_pose_frame_id == "base_link"
            and self.object_or_target_pose_age_s is not None
            and math.isfinite(self.object_or_target_pose_age_s)
            and 0.0 <= self.object_or_target_pose_age_s <= 0.2
            and math.isfinite(self.object_or_target_pose_confidence)
            and self.object_or_target_pose_confidence >= 0.8
            and self.arm_ik_reachable
            and self.manipulator_ready
        )


@dataclass(frozen=True)
class MotionFeedback:
    status: MotionStatus = MotionStatus.ACTIVE
    physical_stop_evidence: PhysicalStopEvidence | None = None
    reset_gate_pass: bool = False
    standing_gate_pass: bool = False
    source_mode_handshake_pass: bool = False
    safety_gates_pass: bool = False
    initial_activation_authorized: bool = False
    stand_readback_pass: bool = False
    input_source_readback_pass: bool = False
    reason: str = ""


@dataclass(frozen=True)
class RouteDirective:
    kind: DirectiveKind
    route_id: str
    segment_id: str | None
    waypoint_id: str | None
    profile: Profile | None
    goal_x_m: float | None = None
    goal_y_m: float | None = None
    goal_yaw_rad: float | None = None
    position_tolerance_m: float | None = None
    yaw_tolerance_rad: float | None = None
    target_forward_m_s: float = 0.0
    max_forward_m_s: float = 0.0
    max_angular_rad_s: float = 0.0
    max_command_window_s: float = 0.0
    maximum_turn_displacement_m: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class RouteStep:
    waypoint_id: str
    x_m: float
    y_m: float
    profile: Profile
    position_tolerance_m: float
    yaw_policy: YawPolicy
    yaw_rad: float | None
    yaw_tolerance_rad: float | None
    max_position_std_m: float
    max_yaw_std_rad: float
    stop_behavior: StopBehavior
    segment_timeout_s: float
    recovery_limit: int
    corridor_polygon: tuple[tuple[float, float], ...]
    turn_corridor_polygon: tuple[tuple[float, float], ...] | None
    minimum_obstacle_clearance_m: float
    target_forward_m_s: float
    max_forward_m_s: float
    max_angular_rad_s: float
    max_command_window_s: float
    turn_strategy: str | None
    maximum_turn_displacement_m: float | None


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    frame_id: str
    workspace_verified: bool
    bounded_moving_turn_qualified: bool
    zero_forward_turn_qualified: bool
    start_x_m: float
    start_y_m: float
    start_yaw_rad: float
    start_position_tolerance_m: float
    start_yaw_tolerance_rad: float
    robot_footprint_radius_m: float
    corridor_boundary_margin_m: float
    max_pose_age_s: float
    max_transform_age_s: float
    max_safety_age_s: float
    max_position_std_m: float
    max_yaw_std_rad: float
    steps: tuple[RouteStep, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RouteSpec":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise RouteError("unsupported route schema_version")
        frame_id = str(raw.get("frame_id", ""))
        if frame_id not in {"map", "odom"}:
            raise RouteError("route frame_id must be map or odom")
        start = raw.get("start_anchor")
        if not isinstance(start, Mapping):
            raise RouteError("start_anchor must be an object")
        localization = raw.get("localization")
        if not isinstance(localization, Mapping):
            raise RouteError("localization policy must be an object")
        safety = raw.get("safety")
        if not isinstance(safety, Mapping):
            raise RouteError("safety policy must be an object")
        turn_capabilities = raw.get("turn_capabilities")
        if not isinstance(turn_capabilities, Mapping):
            raise RouteError("turn_capabilities must explicitly classify each turning strategy")
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RouteError("steps must be a non-empty list")

        steps = tuple(_parse_step(item) for item in raw_steps)
        spec = cls(
            route_id=str(raw.get("route_id", "")),
            frame_id=frame_id,
            workspace_verified=raw.get("workspace_clearance_verified") is True,
            bounded_moving_turn_qualified=turn_capabilities.get("bounded_moving_turn_qualified") is True,
            zero_forward_turn_qualified=turn_capabilities.get("zero_forward_turn_qualified") is True,
            start_x_m=_number(start, "x_m"),
            start_y_m=_number(start, "y_m"),
            start_yaw_rad=_number(start, "yaw_rad"),
            start_position_tolerance_m=_positive(start, "position_tolerance_m"),
            start_yaw_tolerance_rad=_positive(start, "yaw_tolerance_rad"),
            robot_footprint_radius_m=_positive(safety, "robot_footprint_radius_m"),
            corridor_boundary_margin_m=_nonnegative(safety, "boundary_margin_m"),
            max_pose_age_s=_positive(localization, "max_pose_age_s"),
            max_transform_age_s=_positive(localization, "max_transform_age_s"),
            max_safety_age_s=_positive(safety, "max_observation_age_s"),
            max_position_std_m=_positive(localization, "max_position_std_m"),
            max_yaw_std_rad=_positive(localization, "max_yaw_std_rad"),
            steps=steps,
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "RouteSpec":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, Mapping):
            raise RouteError("route root must be a JSON object")
        return cls.from_mapping(raw)

    def validate(self) -> None:
        if not self.route_id or not self.steps:
            raise RouteError("route_id and at least one step are required")
        numeric = (
            self.start_x_m, self.start_y_m, self.start_yaw_rad,
            self.start_position_tolerance_m, self.start_yaw_tolerance_rad,
            self.robot_footprint_radius_m, self.corridor_boundary_margin_m,
            self.max_pose_age_s, self.max_transform_age_s, self.max_safety_age_s,
            self.max_position_std_m, self.max_yaw_std_rad,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise RouteError("route-level numeric fields must be finite")
        if self.max_pose_age_s > 0.5 or self.max_safety_age_s > 0.5:
            raise RouteError("pose and corridor observation freshness limits may not exceed 0.5 s")
        if self.max_transform_age_s > 1.0:
            raise RouteError("transform freshness limit may not exceed 1.0 s")
        ids = [step.waypoint_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise RouteError("waypoint ids must be unique")
        if self.steps[-1].stop_behavior is not StopBehavior.VERIFY_PHYSICAL_STOP:
            raise RouteError("the final route step must require a measured physical stop")
        for index, step in enumerate(self.steps):
            _validate_step(step)
            start_x, start_y = (
                (self.start_x_m, self.start_y_m)
                if index == 0 else (self.steps[index - 1].x_m, self.steps[index - 1].y_m)
            )
            if not point_in_polygon(start_x, start_y, step.corridor_polygon):
                raise RouteError(f"{step.waypoint_id}: incoming segment starts outside its corridor")
            if not point_in_polygon(step.x_m, step.y_m, step.corridor_polygon):
                raise RouteError(f"{step.waypoint_id}: target pose lies outside its route corridor")
            if step.profile is Profile.CORNER:
                assert step.turn_corridor_polygon is not None
                if not point_in_polygon(step.x_m, step.y_m, step.turn_corridor_polygon):
                    raise RouteError(f"{step.waypoint_id}: corner target lies outside its turn corridor")
            if step.yaw_policy is YawPolicy.NEXT_SEGMENT and index + 1 >= len(self.steps):
                raise RouteError(f"{step.waypoint_id}: NEXT_SEGMENT yaw requires a following waypoint")
            if step.profile is Profile.CORNER and index + 1 >= len(self.steps):
                raise RouteError(f"{step.waypoint_id}: a corner requires a following segment")


def validate_waypoint_docking_route(route: RouteSpec) -> RouteSpec:
    """Validate the bounded Local01 START -> WAYPOINT -> TABLE_DOCKING route.

    This is a narrow profile for the existing route manager and M1 bridge, not
    a second route format. It deliberately accepts only odom-frame two-step
    routes, requires a physical stop at both steps, and freezes the live
    motion envelope to the previously exercised 0.12 m/s profile.
    """
    route.validate()
    if route.frame_id != "odom":
        raise RouteError("waypoint-docking route must use the ground-truth odom frame")
    if not route.workspace_verified:
        raise RouteError("waypoint-docking route requires verified workspace clearance")
    if len(route.steps) != 2:
        raise RouteError("waypoint-docking route must contain exactly WAYPOINT and TABLE_DOCKING")
    if route.start_position_tolerance_m > 0.02 or route.start_yaw_tolerance_rad > 0.02:
        raise RouteError("waypoint-docking start anchor tolerance may not exceed 0.02 m/rad")
    waypoint, docking = route.steps
    if waypoint.waypoint_id != "WAYPOINT" or waypoint.profile is not Profile.TRANSIT:
        raise RouteError("first route step must be id WAYPOINT with TRANSIT profile")
    if waypoint.yaw_policy is not YawPolicy.EXPLICIT:
        raise RouteError("WAYPOINT requires the Local01 explicit target yaw")
    if float(waypoint.yaw_tolerance_rad or 0.0) > 0.10:
        raise RouteError("WAYPOINT yaw tolerance may not exceed 0.10 rad")
    if abs(waypoint.position_tolerance_m - 0.15) > 1e-9:
        raise RouteError("WAYPOINT tolerance must retain the existing 0.15 m diagnostic limit")
    if docking.waypoint_id != "TABLE_DOCKING" or docking.profile is not Profile.DOCKING:
        raise RouteError("final route step must be id TABLE_DOCKING with DOCKING profile")
    if docking.yaw_policy is not YawPolicy.EXPLICIT:
        raise RouteError("TABLE_DOCKING requires an explicit target yaw")
    if math.hypot(waypoint.x_m - route.start_x_m, waypoint.y_m - route.start_y_m) <= 0.05:
        raise RouteError("START to WAYPOINT segment must exceed the 0.05 m stop/noise envelope")
    if math.hypot(docking.x_m - waypoint.x_m, docking.y_m - waypoint.y_m) <= 0.05:
        raise RouteError("WAYPOINT to TABLE_DOCKING segment must exceed the 0.05 m stop/noise envelope")
    for step in route.steps:
        if step.stop_behavior is not StopBehavior.VERIFY_PHYSICAL_STOP:
            raise RouteError(f"{step.waypoint_id}: every route step must require a measured physical stop")
        if step.recovery_limit != 0:
            raise RouteError(f"{step.waypoint_id}: automatic route retries are disabled")
        if (
            step.target_forward_m_s != MIN_EFFECTIVE_FORWARD_M_S
            or step.max_forward_m_s > MIN_EFFECTIVE_FORWARD_M_S
            or step.max_angular_rad_s > MAX_ANGULAR_RAD_S
            or step.max_command_window_s > 5.0
        ):
            raise RouteError(
                f"{step.waypoint_id}: route commands must stay at 0.12 m/s, "
                "0.075 rad/s, and 5 s per window"
            )
    if docking.position_tolerance_m > 0.08 or float(docking.yaw_tolerance_rad or 0.0) > 0.10:
        raise RouteError("TABLE_DOCKING tolerance may not exceed 0.08 m / 0.10 rad")
    return route


def build_waypoint_docking_route_from_poses(
    pose_input: Mapping[str, Any],
    *,
    runtime_robot_footprint_radius_m: float,
    verified_local_clear_radius_m: float,
) -> RouteSpec:
    """Build the existing RouteSpec from three ground-truth poses and inspected scene bounds."""
    try:
        input_id = str(pose_input["route_id"])
        start = pose_input["start"]
        waypoint = pose_input["waypoint"]
        docking = pose_input["table_docking"]
        if pose_input.get("schema_version") != WAYPOINT_DOCKING_POSE_SCHEMA_VERSION:
            raise RouteError("unsupported waypoint-docking pose schema_version")
        if pose_input.get("frame_id") != "odom":
            raise RouteError("waypoint-docking poses must be expressed in ground-truth odom")
        start_x, start_y, start_yaw = (float(start[k]) for k in ("x_m", "y_m", "yaw_rad"))
        waypoint_x, waypoint_y, waypoint_yaw = (float(waypoint[k]) for k in ("x_m", "y_m", "yaw_rad"))
        docking_x, docking_y, docking_yaw = (float(docking[k]) for k in ("x_m", "y_m", "yaw_rad"))
        footprint = float(runtime_robot_footprint_radius_m)
        local_clearance = float(verified_local_clear_radius_m)
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteError("waypoint-docking poses or scene measurements are malformed") from exc
    if not all(math.isfinite(v) for v in (
        start_x, start_y, start_yaw, waypoint_x, waypoint_y, waypoint_yaw,
        docking_x, docking_y, docking_yaw, footprint, local_clearance,
    )):
        raise RouteError("waypoint-docking poses and scene measurements must be finite")
    if footprint <= 0.0 or local_clearance <= 0.0:
        raise RouteError("runtime footprint and local clear radius must be positive")

    boundary_margin = 0.10
    cross_track_allowance = CONFIGURED_ROUTE_MAX_CROSS_TRACK_M
    corridor_half_width = footprint + boundary_margin + cross_track_allowance
    corridor_extension = footprint + boundary_margin + cross_track_allowance

    def corridor(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> tuple[tuple[float, float], ...]:
        sx, sy = start_xy
        ex, ey = end_xy
        length = math.hypot(ex - sx, ey - sy)
        if length <= 0.05:
            raise RouteError("configured route segment is too short for the existing stop/noise envelope")
        heading = math.atan2(ey - sy, ex - sx)
        ux, uy = math.cos(heading), math.sin(heading)
        nx, ny = -uy, ux

        def point(along: float, lateral: float) -> tuple[float, float]:
            return (
                sx + along * ux + lateral * nx,
                sy + along * uy + lateral * ny,
            )

        return tuple(point(along, lateral) for along, lateral in (
            (-corridor_extension, -corridor_half_width),
            (length + corridor_extension, -corridor_half_width),
            (length + corridor_extension, corridor_half_width),
            (-corridor_extension, corridor_half_width),
        ))

    waypoint_corridor = corridor((start_x, start_y), (waypoint_x, waypoint_y))
    docking_corridor = corridor((waypoint_x, waypoint_y), (docking_x, docking_y))
    common = {
        "max_position_std_m": 0.01,
        "max_yaw_std_rad": 0.05,
        "stop_behavior": StopBehavior.VERIFY_PHYSICAL_STOP,
        "segment_timeout_s": 60.0,
        "recovery_limit": 0,
        "minimum_obstacle_clearance_m": 0.0,
        "target_forward_m_s": MIN_EFFECTIVE_FORWARD_M_S,
        "max_forward_m_s": MIN_EFFECTIVE_FORWARD_M_S,
        "max_angular_rad_s": MAX_ANGULAR_RAD_S,
        "max_command_window_s": 5.0,
        "turn_strategy": None,
        "maximum_turn_displacement_m": None,
    }
    route = RouteSpec(
        route_id=input_id,
        frame_id="odom",
        workspace_verified=True,
        bounded_moving_turn_qualified=False,
        zero_forward_turn_qualified=False,
        start_x_m=start_x,
        start_y_m=start_y,
        start_yaw_rad=start_yaw,
        start_position_tolerance_m=0.02,
        start_yaw_tolerance_rad=0.02,
        robot_footprint_radius_m=footprint,
        corridor_boundary_margin_m=boundary_margin,
        max_pose_age_s=0.20,
        max_transform_age_s=0.50,
        max_safety_age_s=0.20,
        max_position_std_m=0.01,
        max_yaw_std_rad=0.05,
        steps=(
            RouteStep(
                waypoint_id="WAYPOINT", x_m=waypoint_x, y_m=waypoint_y,
                profile=Profile.TRANSIT, position_tolerance_m=0.15,
                yaw_policy=YawPolicy.EXPLICIT, yaw_rad=waypoint_yaw,
                yaw_tolerance_rad=0.10, corridor_polygon=waypoint_corridor,
                turn_corridor_polygon=None, **common,
            ),
            RouteStep(
                waypoint_id="TABLE_DOCKING", x_m=docking_x, y_m=docking_y,
                profile=Profile.DOCKING, position_tolerance_m=0.08,
                yaw_policy=YawPolicy.EXPLICIT, yaw_rad=docking_yaw,
                yaw_tolerance_rad=0.10, corridor_polygon=docking_corridor,
                turn_corridor_polygon=None, **common,
            ),
        ),
    )
    return validate_waypoint_docking_route(route)


def load_waypoint_docking_pose_input(path: str | Path) -> dict[str, Any]:
    """Load the minimal Local01 pose input without inventing route coordinates."""
    route_path = Path(path)
    if route_path.is_symlink() or not route_path.is_file():
        raise RouteError("route config must be a regular, non-symlink file")
    try:
        raw = json.loads(route_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RouteError(f"route pose input is unreadable: {type(exc).__name__}") from exc
    required = {"schema_version", "route_id", "frame_id", "start", "waypoint", "table_docking"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise RouteError("route pose input must contain exactly schema_version, route_id, frame_id, start, waypoint, table_docking")
    if raw.get("schema_version") != WAYPOINT_DOCKING_POSE_SCHEMA_VERSION:
        raise RouteError("unsupported waypoint-docking pose schema_version")
    if not isinstance(raw.get("route_id"), str) or not str(raw["route_id"]).strip():
        raise RouteError("route_id must be a non-empty string")
    if raw.get("frame_id") != "odom":
        raise RouteError("waypoint-docking poses must be expressed in ground-truth odom")
    result: dict[str, Any] = {
        "schema_version": WAYPOINT_DOCKING_POSE_SCHEMA_VERSION,
        "route_id": str(raw["route_id"]),
        "frame_id": "odom",
    }
    for key in ("start", "waypoint", "table_docking"):
        pose = raw.get(key)
        if not isinstance(pose, Mapping) or set(pose) != {"x_m", "y_m", "yaw_rad"}:
            raise RouteError(f"{key} must contain exactly x_m, y_m, yaw_rad")
        values: dict[str, float] = {}
        for name in ("x_m", "y_m", "yaw_rad"):
            raw_value = pose[name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise RouteError(f"{key}.{name} must be a JSON number")
            values[name] = _number(pose, name)
        result[key] = values
    if math.hypot(
        result["waypoint"]["x_m"] - result["start"]["x_m"],
        result["waypoint"]["y_m"] - result["start"]["y_m"],
    ) <= 0.05:
        raise RouteError("START to WAYPOINT segment must exceed the 0.05 m stop/noise envelope")
    if math.hypot(
        result["table_docking"]["x_m"] - result["waypoint"]["x_m"],
        result["table_docking"]["y_m"] - result["waypoint"]["y_m"],
    ) <= 0.05:
        raise RouteError("WAYPOINT to TABLE_DOCKING segment must exceed the 0.05 m stop/noise envelope")
    return result


def load_waypoint_docking_route(
    path: str | Path,
    *,
    runtime_robot_footprint_radius_m: float,
    verified_local_clear_radius_m: float,
) -> RouteSpec:
    """Build the existing RouteSpec from the minimal Local01 pose input."""
    pose_input = load_waypoint_docking_pose_input(path)
    route = build_waypoint_docking_route_from_poses(
        pose_input,
        runtime_robot_footprint_radius_m=runtime_robot_footprint_radius_m,
        verified_local_clear_radius_m=verified_local_clear_radius_m,
    )
    validate_waypoint_docking_scene_clearance(
        route,
        runtime_robot_footprint_radius_m=runtime_robot_footprint_radius_m,
        verified_local_clear_radius_m=verified_local_clear_radius_m,
    )
    return route


def validate_waypoint_docking_scene_clearance(
    route: RouteSpec,
    *,
    runtime_robot_footprint_radius_m: float,
    verified_local_clear_radius_m: float,
) -> dict[str, Any]:
    """Fail closed when a configured route exceeds the inspected simple scene."""
    validate_waypoint_docking_route(route)
    values = (runtime_robot_footprint_radius_m, verified_local_clear_radius_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise RouteError("scene clearance measurements must be finite")
    runtime_footprint = float(runtime_robot_footprint_radius_m)
    local_clearance = float(verified_local_clear_radius_m)
    if runtime_footprint <= 0.0 or local_clearance <= 0.0:
        raise RouteError("scene footprint and local clearance must be positive")
    if route.robot_footprint_radius_m + 1e-9 < runtime_footprint:
        raise RouteError("route footprint is smaller than the inspected runtime robot footprint")
    if route.robot_footprint_radius_m + route.corridor_boundary_margin_m >= local_clearance:
        raise RouteError("route footprint and margin do not fit inside the verified clear radius")
    maximum_center_radius = (
        local_clearance - route.robot_footprint_radius_m - route.corridor_boundary_margin_m
    )
    maximum_vertex_radius = 0.0
    for step in route.steps:
        if step.minimum_obstacle_clearance_m > local_clearance:
            raise RouteError(f"{step.waypoint_id}: requested clearance exceeds the inspected scene")
        for x_m, y_m in step.corridor_polygon:
            radius = math.hypot(x_m - route.start_x_m, y_m - route.start_y_m)
            maximum_vertex_radius = max(maximum_vertex_radius, radius)
            if radius > maximum_center_radius + 1e-9:
                raise RouteError(
                    f"{step.waypoint_id}: corridor exceeds clear radius after footprint and boundary margin"
                )
    return {
        "status": "PASS",
        "runtime_robot_footprint_radius_m": runtime_footprint,
        "declared_route_footprint_radius_m": route.robot_footprint_radius_m,
        "verified_local_clear_radius_m": local_clearance,
        "maximum_corridor_center_radius_m": maximum_center_radius,
        "maximum_corridor_vertex_radius_from_start_m": maximum_vertex_radius,
    }


def _number(raw: Mapping[str, Any], key: str) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise RouteError(f"{key} must be finite")
    return value


def _startup_gates_pass(feedback: MotionFeedback) -> bool:
    return (
        feedback.reset_gate_pass
        and feedback.standing_gate_pass
        and (feedback.source_mode_handshake_pass or feedback.initial_activation_authorized)
        and feedback.safety_gates_pass
    )


def _positive(raw: Mapping[str, Any], key: str) -> float:
    value = _number(raw, key)
    if value <= 0.0:
        raise RouteError(f"{key} must be positive")
    return value


def _nonnegative(raw: Mapping[str, Any], key: str) -> float:
    value = _number(raw, key)
    if value < 0.0:
        raise RouteError(f"{key} must be nonnegative")
    return value


def _polygon(raw: Any, name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise RouteError(f"{name} must have at least three 2D vertices")
    vertices: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise RouteError(f"{name} vertices must be [x_m, y_m]")
        vertices.append((float(point[0]), float(point[1])))
    if not all(math.isfinite(value) for point in vertices for value in point):
        raise RouteError(f"{name} vertices must be finite")
    if abs(_polygon_signed_area(vertices)) < 1e-8:
        raise RouteError(f"{name} polygon area must be nonzero")
    return tuple(vertices)


def _parse_step(raw: Any) -> RouteStep:
    if not isinstance(raw, Mapping):
        raise RouteError("each route step must be an object")
    pose = raw.get("pose")
    accept = raw.get("acceptance")
    corridor = raw.get("corridor")
    bounds = raw.get("command_bounds")
    if not all(isinstance(value, Mapping) for value in (pose, accept, corridor, bounds)):
        raise RouteError("step pose, acceptance, corridor, and command_bounds must be objects")
    profile = Profile(str(raw.get("profile", "")))
    yaw_policy = YawPolicy(str(accept.get("yaw_policy", "")))
    stop_behavior = StopBehavior(str(raw.get("stop_behavior", "")))
    yaw_value = pose.get("yaw_rad")
    yaw_rad = None if yaw_value is None else float(yaw_value)
    yaw_tol_value = accept.get("yaw_tolerance_rad")
    yaw_tol = None if yaw_tol_value is None else float(yaw_tol_value)
    turn = raw.get("turn")
    turn = turn if isinstance(turn, Mapping) else {}
    turn_polygon = corridor.get("turn_polygon")
    return RouteStep(
        waypoint_id=str(raw.get("id", "")),
        x_m=_number(pose, "x_m"),
        y_m=_number(pose, "y_m"),
        profile=profile,
        position_tolerance_m=_positive(accept, "position_tolerance_m"),
        yaw_policy=yaw_policy,
        yaw_rad=yaw_rad,
        yaw_tolerance_rad=yaw_tol,
        max_position_std_m=_positive(accept, "max_position_std_m"),
        max_yaw_std_rad=_positive(accept, "max_yaw_std_rad"),
        stop_behavior=stop_behavior,
        segment_timeout_s=_positive(raw, "segment_timeout_s"),
        recovery_limit=int(raw.get("recovery_limit", 0)),
        corridor_polygon=_polygon(corridor.get("polygon"), "corridor.polygon"),
        turn_corridor_polygon=(
            None if turn_polygon is None else _polygon(turn_polygon, "corridor.turn_polygon")
        ),
        minimum_obstacle_clearance_m=_nonnegative(corridor, "minimum_obstacle_clearance_m"),
        target_forward_m_s=_positive(bounds, "target_forward_m_s"),
        max_forward_m_s=_positive(bounds, "max_forward_m_s"),
        max_angular_rad_s=_positive(bounds, "max_angular_rad_s"),
        max_command_window_s=_positive(bounds, "max_command_window_s"),
        turn_strategy=(None if not turn else str(turn.get("strategy", ""))),
        maximum_turn_displacement_m=(
            None if not turn else _positive(turn, "maximum_displacement_m")
        ),
    )


def _validate_step(step: RouteStep) -> None:
    values = (
        step.x_m, step.y_m, step.position_tolerance_m, step.segment_timeout_s,
        step.minimum_obstacle_clearance_m, step.max_forward_m_s,
        step.max_angular_rad_s, step.max_command_window_s,
        step.max_position_std_m, step.max_yaw_std_rad, step.target_forward_m_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise RouteError(f"{step.waypoint_id}: step numeric fields must be finite")
    if step.yaw_rad is not None and not math.isfinite(step.yaw_rad):
        raise RouteError(f"{step.waypoint_id}: yaw_rad must be finite")
    if step.yaw_tolerance_rad is not None and (
        not math.isfinite(step.yaw_tolerance_rad) or step.yaw_tolerance_rad <= 0.0
    ):
        raise RouteError(f"{step.waypoint_id}: yaw tolerance must be finite and positive")
    if step.recovery_limit < 0:
        raise RouteError(f"{step.waypoint_id}: recovery_limit must be nonnegative")
    if step.recovery_limit > 3:
        raise RouteError(f"{step.waypoint_id}: recovery_limit exceeds the route prototype bound of 3")
    if step.max_forward_m_s > MAX_FORWARD_M_S + 1e-9:
        raise RouteError(f"{step.waypoint_id}: forward command exceeds the existing 0.14 m/s envelope")
    if step.target_forward_m_s < MIN_EFFECTIVE_FORWARD_M_S - 1e-9:
        raise RouteError(f"{step.waypoint_id}: target forward speed is below the measured effective command deadzone")
    if step.target_forward_m_s > step.max_forward_m_s + 1e-9:
        raise RouteError(f"{step.waypoint_id}: target forward speed exceeds its declared bound")
    if step.max_angular_rad_s > MAX_ANGULAR_RAD_S + 1e-9:
        raise RouteError(f"{step.waypoint_id}: yaw command exceeds the existing 0.075 rad/s envelope")
    if step.max_command_window_s > MAX_ACTION_WINDOW_S + 1e-9:
        raise RouteError(f"{step.waypoint_id}: command window exceeds the existing 10 s bound")
    if step.yaw_policy is YawPolicy.EXPLICIT:
        if step.yaw_rad is None or step.yaw_tolerance_rad is None or step.yaw_tolerance_rad <= 0:
            raise RouteError(f"{step.waypoint_id}: EXPLICIT yaw needs a target and positive tolerance")
    elif step.yaw_policy is YawPolicy.NEXT_SEGMENT:
        if step.yaw_tolerance_rad is None or step.yaw_tolerance_rad <= 0:
            raise RouteError(f"{step.waypoint_id}: NEXT_SEGMENT yaw needs a positive tolerance")
    elif step.yaw_rad is not None or step.yaw_tolerance_rad is not None:
        raise RouteError(f"{step.waypoint_id}: IGNORE yaw cannot carry a yaw target or tolerance")
    if step.profile is Profile.TRANSIT and not 0.15 <= step.position_tolerance_m <= 0.25:
        raise RouteError(f"{step.waypoint_id}: TRANSIT tolerance must remain in the proposed 0.15..0.25 m range")
    if step.profile is Profile.CORNER and step.position_tolerance_m > 0.25:
        raise RouteError(f"{step.waypoint_id}: CORNER region may not exceed the proposed 0.25 m")
    if step.profile is Profile.CORNER and float(step.yaw_tolerance_rad or 0.0) > 0.26:
        raise RouteError(f"{step.waypoint_id}: CORNER yaw tolerance may not exceed the proposed 0.26 rad")
    if step.profile is Profile.CORNER:
        if step.yaw_policy is not YawPolicy.NEXT_SEGMENT:
            raise RouteError(f"{step.waypoint_id}: CORNER must face its next route segment")
        if step.turn_strategy != "BOUNDED_MOVING_TURN" or step.turn_corridor_polygon is None:
            raise RouteError(f"{step.waypoint_id}: CORNER needs a moving-turn strategy and turn corridor")
        if step.maximum_turn_displacement_m is None or step.maximum_turn_displacement_m <= 0:
            raise RouteError(f"{step.waypoint_id}: CORNER needs a finite displacement limit")
        if step.max_forward_m_s < MIN_EFFECTIVE_FORWARD_M_S:
            raise RouteError(f"{step.waypoint_id}: moving turn must stay above the measured 0.111 m/s deadzone")
        if step.stop_behavior is not StopBehavior.VERIFY_PHYSICAL_STOP:
            raise RouteError(f"{step.waypoint_id}: CORNER must verify physical stop after turning")
    if step.profile is Profile.DOCKING:
        if step.position_tolerance_m > 0.10 or float(step.yaw_tolerance_rad or 0.0) > 0.10:
            raise RouteError(f"{step.waypoint_id}: DOCKING tolerances may not exceed the proposed 0.10 m/rad")
        if step.yaw_policy is not YawPolicy.EXPLICIT:
            raise RouteError(f"{step.waypoint_id}: DOCKING requires an explicit yaw target")
        if step.stop_behavior is not StopBehavior.VERIFY_PHYSICAL_STOP:
            raise RouteError(f"{step.waypoint_id}: DOCKING must verify physical stop before handoff")


class PoseProvider(Protocol):
    """Return a pose transformed into the route frame at the sample timestamp."""

    def latest_pose(self, target_frame: str, now_steady_s: float) -> PoseEstimate | None: ...


class CorridorSafetyProvider(Protocol):
    """Return current static-map/LiDAR clearance for a route segment."""

    def latest_observation(
        self, segment_id: str, frame_id: str, now_steady_s: float
    ) -> SafetyObservation | None: ...


class MotionAdapter(Protocol):
    """Adapter boundary. A production implementation must own real safety gates."""

    def dispatch(self, directive: RouteDirective) -> None: ...
    def request_zero_and_stop(self, reason: str, requested_steady_s: float) -> float | None:
        """Start zero publication and return its actual monotonic start time.

        This is command-delivery timing only, never proof of physical stopping.
        The existing measured stop gate must provide the separate evidence.
        """
        ...


class WaypointRouteExecutor:
    """Supervise route phases from measured pose and fail closed on bad inputs."""

    def __init__(
        self,
        route: RouteSpec,
        pose_provider: PoseProvider,
        safety_provider: CorridorSafetyProvider,
        motion_adapter: MotionAdapter,
    ) -> None:
        self.route = route
        self.pose_provider = pose_provider
        self.safety_provider = safety_provider
        self.motion_adapter = motion_adapter
        self.state = RouteState.NOT_STARTED
        self.phase = RoutePhase.FOLLOW
        self.step_index = 0
        self._phase_started_s: float | None = None
        self._step_started_s: float | None = None
        self._last_now_s: float | None = None
        self._last_stamp_ns: int | None = None
        self._last_wall_time_ns: int | None = None
        self._last_received_steady_s: float | None = None
        self.verified_wall_clock_step_rebases = 0
        self._generation: str | None = None
        self._turn_origin: tuple[float, float] | None = None
        self._recovery_count = 0
        self._recovery_resume_phase = RoutePhase.FOLLOW
        self._last_directive: RouteDirective | None = None
        self._handoff_sent = False
        self._stop_requested_at_s: float | None = None
        self._stop_verified_for_current_step = False
        self._last_error: str | None = None
        self._return_error_m: float | None = None
        self._return_yaw_error_rad: float | None = None

    @property
    def active_step(self) -> RouteStep | None:
        return self.route.steps[self.step_index] if self.step_index < len(self.route.steps) else None

    def start(self, now_steady_s: float, feedback: MotionFeedback = MotionFeedback()) -> RouteDirective:
        if self.state is not RouteState.NOT_STARTED:
            return self._abort("ROUTE_ALREADY_STARTED")
        if not math.isfinite(now_steady_s):
            return self._abort("INVALID_MONOTONIC_TIME")
        if not self.route.workspace_verified:
            return self._abort("WORKSPACE_CLEARANCE_NOT_VERIFIED")
        if any(step.profile is Profile.CORNER for step in self.route.steps) and not self.route.bounded_moving_turn_qualified:
            return self._abort("BOUNDED_MOVING_TURN_CAPABILITY_NOT_QUALIFIED")
        if not _startup_gates_pass(feedback):
            return self._abort("RESET_STANDING_SOURCE_MODE_OR_SAFETY_GATE_NOT_READY")
        pose, error = self._validated_pose(now_steady_s, initializing=True)
        if error:
            return self._abort(error)
        assert pose is not None
        start_error = math.hypot(pose.x_m - self.route.start_x_m, pose.y_m - self.route.start_y_m)
        start_yaw_error = abs(wrap_angle(pose.yaw_rad - self.route.start_yaw_rad))
        if start_error > self.route.start_position_tolerance_m:
            return self._abort("START_ANCHOR_POSITION_OUTSIDE_TOLERANCE")
        if start_yaw_error > self.route.start_yaw_tolerance_rad:
            return self._abort("START_ANCHOR_YAW_OUTSIDE_TOLERANCE")
        step = self.active_step
        assert step is not None
        safety_error = self._safety_error(step, pose, now_steady_s)
        if safety_error:
            return self._abort(safety_error)
        self.state = RouteState.RUNNING
        self._last_now_s = now_steady_s
        self._phase_started_s = now_steady_s
        self._step_started_s = now_steady_s
        self._generation = pose.generation
        self._last_stamp_ns = pose.stamp_ns
        return self._dispatch_current(pose)

    def tick(
        self,
        now_steady_s: float,
        feedback: MotionFeedback = MotionFeedback(),
        *,
        docking_readiness: DockingReadiness | None = None,
        manipulation_complete: bool | None = None,
    ) -> RouteDirective:
        if self.state in {RouteState.ABORTED, RouteState.COMPLETE}:
            return self._last_directive or self._terminal_directive()
        if self.state is RouteState.NOT_STARTED:
            return self.start(now_steady_s, feedback)
        if not math.isfinite(now_steady_s) or (self._last_now_s is not None and now_steady_s < self._last_now_s):
            return self._abort("MONOTONIC_CLOCK_REGRESSION")
        self._last_now_s = now_steady_s
        stop_evidence = feedback.physical_stop_evidence
        stopped_in_stand = bool(
            feedback.stand_readback_pass is True
            and stop_evidence is not None
            and stop_evidence.status is PhysicalStopStatus.PASS
            and stop_evidence.generation == self._generation
            and stop_evidence.sample_count >= 20
            and math.isfinite(stop_evidence.window_start_steady_s)
            and math.isfinite(stop_evidence.window_end_steady_s)
            and math.isfinite(stop_evidence.maximum_sample_gap_s)
            and math.isfinite(stop_evidence.maximum_state_age_s)
            and stop_evidence.window_end_steady_s - stop_evidence.window_start_steady_s >= 1.9
            and stop_evidence.maximum_sample_gap_s <= 0.15
            and stop_evidence.maximum_state_age_s <= 0.5
            and 0.0 <= now_steady_s - stop_evidence.window_end_steady_s <= 0.5
        )
        if not feedback.safety_gates_pass or (
            not feedback.source_mode_handshake_pass and not stopped_in_stand
        ):
            return self._abort("SAFETY_OR_SOURCE_MODE_HANDSHAKE_LOST")
        pose, error = self._validated_pose(now_steady_s)
        if error:
            return self._abort(error)
        assert pose is not None
        step = self.active_step
        if step is None:
            return self._complete(pose)
        if self._step_started_s is not None and now_steady_s - self._step_started_s > step.segment_timeout_s:
            return self._abort(f"SEGMENT_TIMEOUT:{step.waypoint_id}")
        safety_error = self._safety_error(step, pose, now_steady_s)
        if safety_error:
            return self._abort(safety_error)
        if feedback.status is MotionStatus.SAFETY_FAILURE:
            return self._abort("MOTION_ADAPTER_SAFETY_FAILURE:" + feedback.reason)
        if self.phase is RoutePhase.RECOVERY_STOP:
            evidence = feedback.physical_stop_evidence
            if evidence is not None and evidence.status is PhysicalStopStatus.FAIL:
                return self._abort("PHYSICAL_STOP_GATE_FAILED_DURING_RECOVERY")
            if not self._physical_stop_passed(feedback, now_steady_s):
                return self._request_stop("RETRYABLE_FAILURE_REQUIRES_VERIFIED_STOP_BEFORE_RETRY")
            self.phase = self._recovery_resume_phase
            self._phase_started_s = now_steady_s
            self._step_started_s = now_steady_s
            self._stop_requested_at_s = None
            self._stop_verified_for_current_step = False
            self._last_directive = None
            return self._dispatch_current(pose)
        if feedback.status is MotionStatus.RETRYABLE_FAILURE:
            return self._retry_or_abort(step, feedback.reason, now_steady_s, pose)

        if self.phase is RoutePhase.FOLLOW:
            if feedback.status is MotionStatus.ARRIVED and self._position_reached(step, pose):
                if step.profile is Profile.CORNER:
                    self.phase = RoutePhase.TURN
                    self._turn_origin = (pose.x_m, pose.y_m)
                    self._phase_started_s = now_steady_s
                    return self._dispatch_current(pose)
                if step.stop_behavior is StopBehavior.VERIFY_PHYSICAL_STOP:
                    self.phase = RoutePhase.STOP
                    self._phase_started_s = now_steady_s
                    return self._request_stop(f"WAYPOINT_REACHED:{step.waypoint_id}")
                return self._advance(pose, now_steady_s)
            return self._dispatch_current(pose)

        if self.phase is RoutePhase.TURN:
            assert self._turn_origin is not None
            displacement = math.hypot(pose.x_m - self._turn_origin[0], pose.y_m - self._turn_origin[1])
            if step.maximum_turn_displacement_m is not None and displacement > step.maximum_turn_displacement_m:
                return self._abort(f"CORNER_DISPLACEMENT_BUDGET_EXCEEDED:{step.waypoint_id}")
            target_yaw = self._target_yaw(step)
            yaw_error = abs(wrap_angle(target_yaw - pose.yaw_rad))
            if feedback.status is MotionStatus.ARRIVED and yaw_error <= float(step.yaw_tolerance_rad):
                self.phase = RoutePhase.STOP
                self._phase_started_s = now_steady_s
                return self._request_stop(f"CORNER_YAW_REACHED:{step.waypoint_id}")
            return self._dispatch_current(pose)

        if self.phase is RoutePhase.STOP:
            evidence = feedback.physical_stop_evidence
            if evidence is not None and evidence.status is PhysicalStopStatus.FAIL:
                return self._abort("PHYSICAL_STOP_GATE_FAILED:" + step.waypoint_id)
            if not self._physical_stop_passed(feedback, now_steady_s):
                self.state = RouteState.RUNNING
                return self._last_directive or self._request_stop(f"VERIFY_PHYSICAL_STOP:{step.waypoint_id}")
            self._stop_verified_for_current_step = True
            if step.profile is Profile.DOCKING:
                self.phase = RoutePhase.HANDOFF
                self.state = RouteState.WAITING_FOR_DOCKING
                self._phase_started_s = now_steady_s
                return self._docking_wait(step)
            return self._advance(pose, now_steady_s)

        if self.phase is RoutePhase.HANDOFF:
            if not self._stop_verified_for_current_step or not feedback.safety_gates_pass:
                return self._abort("PHYSICAL_STOP_LOST_BEFORE_DOCKING_HANDOFF")
            if self._handoff_sent:
                if manipulation_complete is True:
                    return self._advance(pose, now_steady_s)
                if manipulation_complete is False:
                    return self._abort("MANIPULATION_HANDOFF_FAILED")
                self.state = RouteState.WAITING_FOR_DOCKING
                return self._last_directive or self._docking_wait(step)
            if docking_readiness is None or not docking_readiness.ready:
                self.state = RouteState.WAITING_FOR_DOCKING
                return self._docking_wait(step)
            directive = self._make_directive(
                DirectiveKind.DOCKING_HANDOFF,
                reason="pose, physical stop, object-relative pose, arm IK, and manipulator readiness passed",
            )
            self.motion_adapter.dispatch(directive)
            self._last_directive = directive
            self._handoff_sent = True
            self.state = RouteState.WAITING_FOR_DOCKING
            return directive

        return self._abort("UNKNOWN_ROUTE_PHASE")

    def accept_arrival_after_verified_stop(
        self,
        now_steady_s: float,
        feedback: MotionFeedback,
        *,
        zero_request_started_s: float,
        bridge_rearmed: bool,
    ) -> RouteDirective:
        """Advance a waypoint whose arrival is confirmed after its stop gate.

        This path is for adapters that only discover arrival after an M1 window
        has already completed the composite physical-stop gate. It consumes
        that existing evidence; it does not publish another zero request or
        ask the caller to repeat the physical-stop window.
        """
        if (
            self.state is not RouteState.RUNNING
            or self.phase not in {RoutePhase.FOLLOW, RoutePhase.STOP}
            or self.active_step is None
            or self.active_step.stop_behavior is not StopBehavior.VERIFY_PHYSICAL_STOP
        ):
            return self._abort("POSTSTOP_ARRIVAL_NOT_VALID_IN_CURRENT_ROUTE_PHASE")
        if not math.isfinite(now_steady_s) or not math.isfinite(zero_request_started_s):
            return self._abort("POSTSTOP_ARRIVAL_TIME_INVALID")
        if (
            self.phase is RoutePhase.STOP
            and (
                self._stop_requested_at_s is None
                or zero_request_started_s < self._stop_requested_at_s - 1e-6
            )
        ):
            return self._abort("POSTSTOP_ARRIVAL_ZERO_STREAM_PRECEDES_ROUTE_STOP_REQUEST")
        if self.phase is RoutePhase.FOLLOW and self._stop_requested_at_s is not None:
            return self._abort("POSTSTOP_ARRIVAL_HAS_UNEXPECTED_PRIOR_STOP_STATE")
        if feedback.status is not MotionStatus.ARRIVED:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_ARRIVED_STATUS")
        if not feedback.reset_gate_pass or not feedback.standing_gate_pass:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_RESET_AND_STANDING_GATES")
        if not feedback.safety_gates_pass:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_PASSING_SAFETY_GATES")
        if bridge_rearmed is not True:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_STOP_GATED_BRIDGE_REARM")
        if not feedback.stand_readback_pass or not feedback.input_source_readback_pass:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_FRESH_STAND_AND_SOURCE_READBACKS")

        evidence = feedback.physical_stop_evidence
        if evidence is None or evidence.status is not PhysicalStopStatus.PASS:
            return self._abort("POSTSTOP_ARRIVAL_REQUIRES_COMPOSITE_PHYSICAL_STOP_PASS")
        if (
            evidence.generation != self._generation
            or evidence.window_start_steady_s < zero_request_started_s
            or evidence.window_end_steady_s > now_steady_s + 1e-9
            or now_steady_s - evidence.window_end_steady_s > 0.5
            or evidence.window_end_steady_s - evidence.window_start_steady_s < 1.9
            or evidence.sample_count < 20
            or evidence.maximum_sample_gap_s > 0.15
            or evidence.maximum_state_age_s > 0.5
        ):
            return self._abort("POSTSTOP_ARRIVAL_STOP_EVIDENCE_INVALID_OR_STALE")

        pose, pose_error = self._validated_pose(now_steady_s, record=False)
        if pose_error or pose is None:
            return self._abort("POSTSTOP_ARRIVAL_POSE_INVALID:" + str(pose_error))
        step = self.active_step
        if not self._position_reached(step, pose):
            return self._abort("POSTSTOP_ARRIVAL_POSE_OUTSIDE_WAYPOINT_TOLERANCE")
        safety_error = self._safety_error(step, pose, now_steady_s)
        if safety_error:
            return self._abort("POSTSTOP_ARRIVAL_SAFETY_INVALID:" + safety_error)

        # Seed the STOP phase with the already observed zero-request time and
        # consume the same complete evidence through the normal route advance.
        # _request_stop is intentionally not called here: doing so would create
        # a duplicate zero/stop lifecycle after the M1 gate already passed.
        self._last_now_s = now_steady_s
        self._stop_requested_at_s = zero_request_started_s
        self.phase = RoutePhase.STOP
        return self.tick(now_steady_s, feedback)

    def _validated_pose(
        self, now_steady_s: float, *, initializing: bool = False, record: bool = True
    ) -> tuple[PoseEstimate | None, str | None]:
        pose = self.pose_provider.latest_pose(self.route.frame_id, now_steady_s)
        if pose is None:
            return None, "POSE_MISSING"
        values = (pose.x_m, pose.y_m, pose.yaw_rad, pose.received_steady_s,
                  pose.position_std_m, pose.yaw_std_rad, pose.transform_age_s)
        if not pose.frame_id or pose.child_frame_id != "base_link" or not all(math.isfinite(v) for v in values):
            return None, "POSE_INVALID"
        if pose.frame_id != self.route.frame_id:
            return None, "POSE_FRAME_MISMATCH"
        if pose.stamp_ns <= 0 or not pose.generation or not pose.source:
            return None, "POSE_STAMP_SOURCE_OR_GENERATION_INVALID"
        if pose.wall_time_ns is not None and pose.wall_time_ns <= 0:
            return None, "POSE_WALL_TIME_INVALID"
        receive_age = now_steady_s - pose.received_steady_s
        if receive_age < 0.0 or receive_age > self.route.max_pose_age_s:
            return None, "POSE_STALE_OR_FUTURE"
        if pose.transform_age_s < 0.0 or pose.transform_age_s > self.route.max_transform_age_s:
            return None, "POSE_TRANSFORM_STALE"
        required_chain = ("odom->base_link",) if self.route.frame_id == "odom" else ("map->odom", "odom->base_link")
        if tuple(pose.transform_chain) != required_chain:
            return None, "POSE_TRANSFORM_CHAIN_MISMATCH"
        step = self.active_step
        position_std_limit = min(self.route.max_position_std_m, step.max_position_std_m) if step else self.route.max_position_std_m
        yaw_std_limit = min(self.route.max_yaw_std_rad, step.max_yaw_std_rad) if step else self.route.max_yaw_std_rad
        if pose.position_std_m > position_std_limit or pose.yaw_std_rad > yaw_std_limit:
            return None, "POSE_CONFIDENCE_BELOW_ROUTE_PROFILE"
        if not initializing:
            if self._generation is not None and pose.generation != self._generation:
                return None, "POSE_GENERATION_CHANGED_RESET_ROUTE"
            if self._last_stamp_ns is not None and pose.stamp_ns <= self._last_stamp_ns:
                if not self._verified_wall_clock_step(pose):
                    return None, "POSE_STAMP_NOT_STRICTLY_INCREASING"
                if record:
                    self.verified_wall_clock_step_rebases += 1
        if record:
            self._last_stamp_ns = pose.stamp_ns
            self._last_wall_time_ns = pose.wall_time_ns
            self._last_received_steady_s = pose.received_steady_s
            self._generation = pose.generation
        return pose, None

    def _verified_wall_clock_step(self, pose: PoseEstimate) -> bool:
        """Accept a stamp epoch rebase only when fresh MuJoCo data tracks wall time."""
        if (
            pose.source != "MUJOCO_GROUND_TRUTH"
            or self._last_stamp_ns is None
            or self._last_wall_time_ns is None
            or self._last_received_steady_s is None
            or pose.wall_time_ns is None
            or pose.received_steady_s <= self._last_received_steady_s
            or pose.received_steady_s - self._last_received_steady_s > self.route.max_pose_age_s
        ):
            return False

        stamp_backstep_ns = self._last_stamp_ns - pose.stamp_ns
        wall_backstep_ns = self._last_wall_time_ns - pose.wall_time_ns
        return (
            REALTIME_CLOCK_BACKSTEP_MIN_NS <= stamp_backstep_ns <= REALTIME_CLOCK_BACKSTEP_MAX_NS
            and REALTIME_CLOCK_BACKSTEP_MIN_NS <= wall_backstep_ns <= REALTIME_CLOCK_BACKSTEP_MAX_NS
            and abs(stamp_backstep_ns - wall_backstep_ns) <= REALTIME_CLOCK_ALIGNMENT_TOLERANCE_NS
            and abs(self._last_stamp_ns - self._last_wall_time_ns) <= REALTIME_CLOCK_ALIGNMENT_TOLERANCE_NS
            and abs(pose.stamp_ns - pose.wall_time_ns) <= REALTIME_CLOCK_ALIGNMENT_TOLERANCE_NS
        )

    def _safety_error(self, step: RouteStep, pose: PoseEstimate, now_steady_s: float) -> str | None:
        corridor = self._active_corridor(step)
        if not point_in_polygon(pose.x_m, pose.y_m, corridor):
            return f"SAFE_CORRIDOR_VIOLATION:{step.waypoint_id}"
        clearance = distance_to_polygon_boundary(pose.x_m, pose.y_m, corridor)
        required_margin = self.route.robot_footprint_radius_m + self.route.corridor_boundary_margin_m
        if clearance < required_margin:
            return f"ROBOT_FOOTPRINT_CLEARANCE_VIOLATION:{step.waypoint_id}"
        observation = self.safety_provider.latest_observation(step.waypoint_id, self.route.frame_id, now_steady_s)
        if observation is None:
            return "CORRIDOR_OBSERVATION_MISSING"
        if not math.isfinite(observation.received_steady_s) or not math.isfinite(observation.minimum_obstacle_clearance_m):
            return "CORRIDOR_OBSERVATION_INVALID"
        if observation.segment_id != step.waypoint_id or observation.frame_id != self.route.frame_id:
            return "CORRIDOR_OBSERVATION_FRAME_OR_SEGMENT_MISMATCH"
        safety_age = now_steady_s - observation.received_steady_s
        if safety_age < 0.0 or safety_age > self.route.max_safety_age_s:
            return "CORRIDOR_OBSERVATION_STALE"
        if not observation.robot_safe:
            return "ROBOT_PHYSICAL_SAFETY_GATE_FAILED:" + (observation.reason or step.waypoint_id)
        if not observation.clear:
            return "ROUTE_OBSTRUCTED:" + (observation.reason or step.waypoint_id)
        if observation.minimum_obstacle_clearance_m < step.minimum_obstacle_clearance_m:
            return "OBSTACLE_CLEARANCE_BELOW_ROUTE_MINIMUM"
        return None

    def _active_corridor(self, step: RouteStep) -> tuple[tuple[float, float], ...]:
        if self.phase is RoutePhase.TURN:
            assert step.turn_corridor_polygon is not None
            return step.turn_corridor_polygon
        return step.corridor_polygon

    def _position_reached(self, step: RouteStep, pose: PoseEstimate) -> bool:
        if math.hypot(step.x_m - pose.x_m, step.y_m - pose.y_m) > step.position_tolerance_m:
            return False
        if step.profile is Profile.CORNER:
            return True
        if step.yaw_policy is YawPolicy.IGNORE:
            return True
        return abs(wrap_angle(self._target_yaw(step) - pose.yaw_rad)) <= float(step.yaw_tolerance_rad)

    def _target_yaw(self, step: RouteStep) -> float:
        if step.yaw_policy is YawPolicy.EXPLICIT:
            assert step.yaw_rad is not None
            return step.yaw_rad
        if step.yaw_policy is YawPolicy.NEXT_SEGMENT:
            next_step = self.route.steps[self.step_index + 1]
            return math.atan2(next_step.y_m - step.y_m, next_step.x_m - step.x_m)
        return step.yaw_rad if step.yaw_rad is not None else 0.0

    def _directive(self) -> RouteDirective:
        step = self.active_step
        if step is None:
            return self._terminal_directive()
        if self.phase is RoutePhase.TURN:
            kind = DirectiveKind.BOUNDED_MOVING_TURN
            yaw = self._target_yaw(step)
            max_window = min(step.max_command_window_s, 5.0)
        elif self.phase is RoutePhase.STOP:
            return RouteDirective(
                DirectiveKind.REQUEST_PHYSICAL_STOP, self.route.route_id,
                step.waypoint_id, step.waypoint_id, step.profile,
                reason="zero is not a stop pass; wait for measured physical-stop gate",
            )
        elif self.phase is RoutePhase.HANDOFF:
            return self._docking_wait(step)
        else:
            kind = DirectiveKind.FOLLOW_WAYPOINT
            yaw = None if step.profile is Profile.CORNER or step.yaw_policy is YawPolicy.IGNORE else self._target_yaw(step)
            max_window = step.max_command_window_s
        return RouteDirective(
            kind, self.route.route_id, step.waypoint_id, step.waypoint_id, step.profile,
            goal_x_m=step.x_m, goal_y_m=step.y_m, goal_yaw_rad=yaw,
            position_tolerance_m=step.position_tolerance_m,
            yaw_tolerance_rad=step.yaw_tolerance_rad,
            target_forward_m_s=step.target_forward_m_s,
            max_forward_m_s=step.max_forward_m_s,
            max_angular_rad_s=step.max_angular_rad_s,
            max_command_window_s=max_window,
            maximum_turn_displacement_m=step.maximum_turn_displacement_m if self.phase is RoutePhase.TURN else None,
            reason=("measured-pose corner turn; re-evaluate after each bounded command slice"
                    if self.phase is RoutePhase.TURN else "measured-pose waypoint intent; no direct velocity publication"),
        )

    def _dispatch_current(self, pose: PoseEstimate) -> RouteDirective:
        directive = self._directive()
        if directive != self._last_directive:
            self.motion_adapter.dispatch(directive)
            self._last_directive = directive
        return directive

    def _request_stop(self, reason: str) -> RouteDirective:
        step = self.active_step
        if self._stop_requested_at_s is None:
            requested_at = self._last_now_s if self._last_now_s is not None else 0.0
            zero_start = self.motion_adapter.request_zero_and_stop(reason, requested_at)
            if (
                zero_start is None or not math.isfinite(zero_start)
                or zero_start < requested_at
            ):
                return self._abort("ZERO_STOP_STREAM_DID_NOT_START_WITH_VALID_MONOTONIC_TIME")
            self._stop_requested_at_s = zero_start
        directive = RouteDirective(
            DirectiveKind.REQUEST_PHYSICAL_STOP, self.route.route_id,
            step.waypoint_id if step else None, step.waypoint_id if step else None,
            step.profile if step else None,
            reason=reason + "; zero publication alone is insufficient",
        )
        if directive != self._last_directive:
            self.motion_adapter.dispatch(directive)
            self._last_directive = directive
        self.state = RouteState.RUNNING
        return directive

    def _physical_stop_passed(self, feedback: MotionFeedback, now_steady_s: float) -> bool:
        evidence = feedback.physical_stop_evidence
        if evidence is None or evidence.status is not PhysicalStopStatus.PASS:
            return False
        values = (
            evidence.window_start_steady_s,
            evidence.window_end_steady_s,
            evidence.maximum_sample_gap_s,
            evidence.maximum_state_age_s,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        if self._stop_requested_at_s is None or evidence.generation != self._generation:
            return False
        return (
            evidence.window_start_steady_s >= self._stop_requested_at_s
            and evidence.window_end_steady_s <= now_steady_s + 1e-9
            and now_steady_s - evidence.window_end_steady_s <= 0.5
            and evidence.window_end_steady_s - evidence.window_start_steady_s >= 1.9
            and evidence.sample_count >= 20
            and evidence.maximum_sample_gap_s <= 0.15
            and evidence.maximum_state_age_s <= 0.5
        )

    def _retry_or_abort(
        self, step: RouteStep, reason: str, now_steady_s: float, pose: PoseEstimate
    ) -> RouteDirective:
        if self._recovery_count >= step.recovery_limit:
            return self._abort(f"RECOVERY_LIMIT_EXCEEDED:{step.waypoint_id}:{reason}")
        self._recovery_count += 1
        self._recovery_resume_phase = self.phase
        self.phase = RoutePhase.RECOVERY_STOP
        self._phase_started_s = now_steady_s
        return self._request_stop("RETRYABLE_FAILURE_REQUIRES_VERIFIED_STOP_BEFORE_RETRY")

    def _advance(self, pose: PoseEstimate, now_steady_s: float) -> RouteDirective:
        self.step_index += 1
        self._recovery_count = 0
        self._phase_started_s = now_steady_s
        self._step_started_s = now_steady_s
        self._turn_origin = None
        self._handoff_sent = False
        self._stop_requested_at_s = None
        self._stop_verified_for_current_step = False
        self.phase = RoutePhase.FOLLOW
        self._last_directive = None
        if self.active_step is None:
            return self._complete(pose)
        return self._dispatch_current(pose)

    def _docking_wait(self, step: RouteStep) -> RouteDirective:
        self.state = RouteState.WAITING_FOR_DOCKING
        directive = RouteDirective(
            DirectiveKind.WAIT, self.route.route_id, step.waypoint_id, step.waypoint_id,
            step.profile, goal_x_m=step.x_m, goal_y_m=step.y_m, goal_yaw_rad=step.yaw_rad,
            position_tolerance_m=step.position_tolerance_m,
            yaw_tolerance_rad=step.yaw_tolerance_rad,
            reason="handoff blocked until physical stop, fresh object-relative pose, arm IK, and manipulation readiness pass",
        )
        self._last_directive = directive
        return directive

    def _make_directive(self, kind: DirectiveKind, *, reason: str = "") -> RouteDirective:
        step = self.active_step
        return RouteDirective(
            kind, self.route.route_id,
            step.waypoint_id if step else None,
            step.waypoint_id if step else None,
            step.profile if step else None,
            goal_x_m=step.x_m if step else None,
            goal_y_m=step.y_m if step else None,
            goal_yaw_rad=step.yaw_rad if step else None,
            position_tolerance_m=step.position_tolerance_m if step else None,
            yaw_tolerance_rad=step.yaw_tolerance_rad if step else None,
            target_forward_m_s=step.target_forward_m_s if step else 0.0,
            reason=reason,
        )

    def _abort(self, reason: str) -> RouteDirective:
        self.state = RouteState.ABORTED
        self._last_error = reason
        self.motion_adapter.request_zero_and_stop(
            reason, self._last_now_s if self._last_now_s is not None else 0.0
        )
        directive = RouteDirective(
            DirectiveKind.STOP_AND_ABORT, self.route.route_id,
            self.active_step.waypoint_id if self.active_step else None,
            self.active_step.waypoint_id if self.active_step else None,
            self.active_step.profile if self.active_step else None,
            reason=reason,
        )
        self.motion_adapter.dispatch(directive)
        self._last_directive = directive
        return directive

    def _complete(self, pose: PoseEstimate) -> RouteDirective:
        self.state = RouteState.COMPLETE
        self._return_error_m = math.hypot(pose.x_m - self.route.start_x_m, pose.y_m - self.route.start_y_m)
        self._return_yaw_error_rad = abs(wrap_angle(pose.yaw_rad - self.route.start_yaw_rad))
        directive = self._terminal_directive(
            reason=(f"return position error {self._return_error_m:.3f} m; "
                    f"yaw error {self._return_yaw_error_rad:.3f} rad; final physical-stop gate passed")
        )
        self._last_directive = directive
        return directive

    def _terminal_directive(self, reason: str = "") -> RouteDirective:
        kind = DirectiveKind.COMPLETE if self.state is RouteState.COMPLETE else DirectiveKind.STOP_AND_ABORT
        return RouteDirective(kind, self.route.route_id, None, None, None, reason=reason or self._last_error or "route not active")

    @property
    def result(self) -> dict[str, Any]:
        return {
            "route_id": self.route.route_id,
            "status": self.state.value,
            "active_step": self.active_step.waypoint_id if self.active_step else None,
            "phase": self.phase.value,
            "failure_reason": self._last_error,
            "return_position_error_m": self._return_error_m,
            "return_yaw_error_rad": self._return_yaw_error_rad,
            "steps_completed": self.step_index if self.state is RouteState.COMPLETE else self.step_index,
            "physical_simulation_performed": False,
        }


def wrap_angle(angle_rad: float) -> float:
    if not math.isfinite(angle_rad):
        raise ValueError("angle must be finite")
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _polygon_signed_area(vertices: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(vertices, (*vertices[1:], vertices[0]))
    )


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0])):
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def distance_to_polygon_boundary(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(polygon, (*polygon[1:], polygon[0])):
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        if length2 <= 1e-15:
            distance = math.hypot(x - x1, y - y1)
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length2))
            distance = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        best = min(best, distance)
    return best
