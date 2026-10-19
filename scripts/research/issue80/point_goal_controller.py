"""Offline planar point-goal controller prototype.

No ROS, MuJoCo, or hardware dependencies. A future adapter must provide
measured poses and already-transformed goals in one declared frame. This is
not a live navigation node and does not plan around obstacles.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class NavState(str, Enum):
    WAIT_GOAL = "wait_goal"
    WAIT_POSE = "wait_pose"
    ALIGN = "align"
    DRIVE = "drive"
    FINAL_ALIGN = "final_align"
    SETTLING = "settling"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STALE_STATE = "stale_state"
    FRAME_MISMATCH = "frame_mismatch"
    INVALID_INPUT = "invalid_input"
    CLOCK_DISCONTINUITY = "clock_discontinuity"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str
    received_steady_s: float


@dataclass(frozen=True)
class Goal2D:
    x: float
    y: float
    yaw: float
    frame_id: str


@dataclass(frozen=True)
class ControllerOutput:
    state: NavState
    linear_x: float = 0.0
    angular_z: float = 0.0
    distance_m: float | None = None
    yaw_error_rad: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ControllerConfig:
    xy_tolerance_m: float = 0.12
    yaw_tolerance_rad: float = 0.12
    align_tolerance_rad: float = 0.20
    k_linear: float = 0.6
    k_angular: float = 0.9
    max_forward_mps: float = 0.20
    max_angular_rps: float = 0.30
    pose_timeout_s: float = 0.20
    goal_timeout_s: float = 30.0
    arrival_dwell_s: float = 0.50

    def __post_init__(self) -> None:
        numeric = (
            self.xy_tolerance_m, self.yaw_tolerance_rad,
            self.align_tolerance_rad, self.k_linear, self.k_angular,
            self.max_forward_mps, self.max_angular_rps,
            self.pose_timeout_s, self.goal_timeout_s, self.arrival_dwell_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("controller configuration must be finite")
        if self.xy_tolerance_m <= 0 or self.max_forward_mps <= 0:
            raise ValueError("distance tolerance and forward bound must be positive")
        if self.max_angular_rps <= 0 or self.pose_timeout_s <= 0:
            raise ValueError("angular bound and pose timeout must be positive")
        if self.goal_timeout_s <= 0 or self.arrival_dwell_s < 0:
            raise ValueError("goal timeout must be positive and dwell nonnegative")
        if min(self.yaw_tolerance_rad, self.align_tolerance_rad,
               self.k_linear, self.k_angular) < 0:
            raise ValueError("tolerances and gains must be nonnegative")


def wrap_angle(angle_rad: float) -> float:
    """Return an angle in [-pi, pi)."""
    if not math.isfinite(angle_rad):
        raise ValueError("angle must be finite")
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _planar_yaw_from_quaternion(
    q_xyzw: Iterable[float], *, max_tilt_rad: float
) -> float:
    q = tuple(float(component) for component in q_xyzw)
    if len(q) != 4 or not all(math.isfinite(component) for component in q):
        raise ValueError("quaternion must contain four finite components")
    norm = math.sqrt(sum(component * component for component in q))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = (component / norm for component in q)

    roll = math.atan2(
        2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
    )
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    if abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad:
        raise ValueError("planar goal quaternion contains excessive roll/pitch")

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return wrap_angle(math.atan2(sin_yaw, cos_yaw))


def goal_from_pose_fields(
    *,
    frame_id: str,
    expected_frame: str,
    position_xyz: Iterable[float],
    orientation_xyzw: Iterable[float],
    max_planar_tilt_rad: float = 0.10,
) -> Goal2D:
    """Validate planar PoseStamped fields already transformed into expected_frame.

    A ROS adapter must transform the goal with tf2 at its header timestamp
    before calling this helper. It deliberately refuses frame guessing.
    Timestamp freshness and clock-domain checks belong in that adapter.
    """
    position = tuple(float(value) for value in position_xyz)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("goal position must contain three finite values")
    if not frame_id or not expected_frame or frame_id != expected_frame:
        raise ValueError("goal must already be transformed into the expected frame")
    if not math.isfinite(max_planar_tilt_rad) or max_planar_tilt_rad < 0:
        raise ValueError("max_planar_tilt_rad must be finite and nonnegative")
    yaw = _planar_yaw_from_quaternion(
        orientation_xyzw, max_tilt_rad=max_planar_tilt_rad
    )
    return Goal2D(position[0], position[1], yaw, frame_id)


class PointGoalController:
    """One-goal ALIGN -> DRIVE -> FINAL_ALIGN -> SETTLING state machine.

    Pose reception age uses a local monotonic clock. Do not integrate commands
    to estimate pose. Terminal safety states latch until reset or a new goal.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._goal: Goal2D | None = None
        self._started_at: float | None = None
        self._arrival_since: float | None = None
        self._last_now: float | None = None
        self._terminal: ControllerOutput | None = None

    def reset(self) -> None:
        self._goal = None
        self._started_at = None
        self._arrival_since = None
        self._last_now = None
        self._terminal = None

    @staticmethod
    def _zero(
        state: NavState,
        reason: str,
        *,
        distance: float | None = None,
        yaw_error: float | None = None,
    ) -> ControllerOutput:
        return ControllerOutput(state, 0.0, 0.0, distance, yaw_error, reason)

    def step(
        self,
        pose: Pose2D | None,
        goal: Goal2D | None,
        *,
        now_steady_s: float,
        cancelled: bool = False,
    ) -> ControllerOutput:
        if not math.isfinite(now_steady_s):
            return self._zero(NavState.INVALID_INPUT, "non-finite steady time")
        if self._last_now is not None and now_steady_s < self._last_now:
            self._terminal = self._zero(
                NavState.CLOCK_DISCONTINUITY, "monotonic clock moved backwards"
            )
            return self._terminal
        self._last_now = now_steady_s

        if goal != self._goal:
            self._goal = goal
            self._started_at = now_steady_s if goal is not None else None
            self._arrival_since = None
            self._terminal = None

        if cancelled:
            self._terminal = self._zero(NavState.CANCELLED, "goal cancelled")
        if self._terminal is not None:
            return self._terminal
        if goal is None:
            return self._zero(NavState.WAIT_GOAL, "no active goal")
        if not goal.frame_id or not all(
            math.isfinite(value) for value in (goal.x, goal.y, goal.yaw)
        ):
            self._terminal = self._zero(NavState.INVALID_INPUT, "invalid goal")
            return self._terminal

        assert self._started_at is not None
        if now_steady_s - self._started_at > self.config.goal_timeout_s:
            self._terminal = self._zero(NavState.TIMED_OUT, "goal deadline exceeded")
            return self._terminal
        if pose is None:
            return self._zero(NavState.WAIT_POSE, "no measured pose")
        if not all(
            math.isfinite(value)
            for value in (pose.x, pose.y, pose.yaw, pose.received_steady_s)
        ) or not pose.frame_id:
            self._terminal = self._zero(
                NavState.INVALID_INPUT, "invalid measured pose"
            )
            return self._terminal
        if pose.frame_id != goal.frame_id:
            self._terminal = self._zero(
                NavState.FRAME_MISMATCH,
                f"pose frame {pose.frame_id!r} != goal frame {goal.frame_id!r}",
            )
            return self._terminal

        pose_age = now_steady_s - pose.received_steady_s
        if pose_age < 0.0 or pose_age > self.config.pose_timeout_s:
            self._terminal = self._zero(
                NavState.STALE_STATE,
                f"measured pose age {pose_age:.3f}s is invalid/stale",
            )
            return self._terminal

        distance = math.hypot(goal.x - pose.x, goal.y - pose.y)
        if distance > self.config.xy_tolerance_m:
            bearing = math.atan2(goal.y - pose.y, goal.x - pose.x)
            bearing_error = wrap_angle(bearing - pose.yaw)
            self._arrival_since = None
            if abs(bearing_error) > self.config.align_tolerance_rad:
                return ControllerOutput(
                    NavState.ALIGN,
                    0.0,
                    self._clamp(
                        self.config.k_angular * bearing_error,
                        self.config.max_angular_rps,
                    ),
                    distance,
                    bearing_error,
                )
            return ControllerOutput(
                NavState.DRIVE,
                min(
                    self.config.k_linear * distance,
                    self.config.max_forward_mps,
                ),
                self._clamp(
                    self.config.k_angular * bearing_error,
                    self.config.max_angular_rps,
                ),
                distance,
                bearing_error,
            )

        final_error = wrap_angle(goal.yaw - pose.yaw)
        if abs(final_error) > self.config.yaw_tolerance_rad:
            self._arrival_since = None
            return ControllerOutput(
                NavState.FINAL_ALIGN,
                0.0,
                self._clamp(
                    self.config.k_angular * final_error,
                    self.config.max_angular_rps,
                ),
                distance,
                final_error,
            )

        if self._arrival_since is None:
            self._arrival_since = now_steady_s
        if now_steady_s - self._arrival_since >= self.config.arrival_dwell_s:
            self._terminal = self._zero(
                NavState.ARRIVED,
                "pose stayed inside both tolerances",
                distance=distance,
                yaw_error=final_error,
            )
            return self._terminal
        return self._zero(
            NavState.SETTLING,
            "waiting for arrival dwell",
            distance=distance,
            yaw_error=final_error,
        )

    @staticmethod
    def _clamp(value: float, magnitude: float) -> float:
        return max(-magnitude, min(magnitude, value))