"""Pure geometry and phase logic for the bounded Issue 80 pose-goal controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


APPROACH = "APPROACH"
TERMINAL_POSE_CONTROL = "TERMINAL_POSE_CONTROL"
ZERO_STOP = "ZERO_STOP"
FAILED = "FAILED"
POSE_POSITION_STOP_REASONS = {
    "speed_scaled_stop_line_crossed",
    "goal_approach_predicted_settle_within_tolerance",
}
POSE_GOAL_LOOKAHEAD_FRACTION = 0.20
POSE_GOAL_TERMINAL_CONTROL_MAX_S = 3.0
POSE_GOAL_TERMINAL_READY_DWELL_S = 0.4
POSE_GOAL_TERMINAL_MIN_ANGULAR_RAD_S = 0.03
POSE_GOAL_TERMINAL_HEADING_GAIN = 0.8
POSE_GOAL_TERMINAL_RATE_DAMPING = 1.0
# Run92/93 zero-to-settled yaw drift was +0.0827/+0.1695 rad; retain a
# simulation-only pre-stop bias inside the unchanged 0.17 rad final tolerance.
POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD = 0.16
POSE_GOAL_TERMINAL_TARGET_TOLERANCE_RAD = 0.01
POSE_SAFETY_STOP_REASONS = {
    "lateral_error_limit",
    "hard_overshoot_guard",
}


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_path_reference(
    start_pose: Mapping[str, Any],
    goal_pose: Mapping[str, Any],
    progress_fraction: float,
) -> dict[str, float]:
    """Return a cubic Hermite path point, tangent, and curvature for one pose goal.

    Endpoint tangent directions are the measured starting yaw and requested final
    yaw. Tangent magnitudes equal the chord length, giving a geometry-scaled path
    with no per-run position or yaw search parameter.
    """
    try:
        x0, y0, yaw0 = (float(start_pose[key]) for key in ("x", "y", "yaw"))
        x1, y1, yaw1 = (float(goal_pose[key]) for key in ("x", "y", "yaw"))
        fraction = float(progress_fraction)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pose path requires numeric start and goal poses") from exc
    if not _finite(x0, y0, yaw0, x1, y1, yaw1, fraction):
        raise ValueError("pose path inputs must be finite")
    if start_pose.get("frame_id") != goal_pose.get("frame_id"):
        raise ValueError("pose path start and goal frames must match")
    if start_pose.get("frame_id") not in (None, "odom"):
        raise ValueError("pose path must be expressed in odom")

    chord = math.hypot(x1 - x0, y1 - y0)
    if chord <= 1e-9:
        raise ValueError("pose path endpoints must be separated")
    t = min(1.0, max(0.0, fraction))
    t2, t3 = t * t, t * t * t
    m0x, m0y = chord * math.cos(yaw0), chord * math.sin(yaw0)
    m1x, m1y = chord * math.cos(yaw1), chord * math.sin(yaw1)

    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
    dh00, dh10, dh01, dh11 = 6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t
    ddh00, ddh10, ddh01, ddh11 = 12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2

    x = h00 * x0 + h10 * m0x + h01 * x1 + h11 * m1x
    y = h00 * y0 + h10 * m0y + h01 * y1 + h11 * m1y
    dx = dh00 * x0 + dh10 * m0x + dh01 * x1 + dh11 * m1x
    dy = dh00 * y0 + dh10 * m0y + dh01 * y1 + dh11 * m1y
    ddx = ddh00 * x0 + ddh10 * m0x + ddh01 * x1 + ddh11 * m1x
    ddy = ddh00 * y0 + ddh10 * m0y + ddh01 * y1 + ddh11 * m1y
    speed_sq = dx * dx + dy * dy
    if speed_sq <= 1e-12:
        raise ValueError("pose path tangent is singular")
    tangent_yaw = math.atan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / (speed_sq ** 1.5)
    return {
        "x_m": x,
        "y_m": y,
        "tangent_yaw_rad": tangent_yaw,
        "curvature_per_m": curvature,
        "progress_fraction": t,
        "chord_length_m": chord,
    }


def pure_pursuit_yaw_rate(
    *,
    current_x_m: float,
    current_y_m: float,
    current_yaw_rad: float,
    carrot_x_m: float,
    carrot_y_m: float,
    forward_command_m_s: float,
    maximum_angular_rad_s: float,
) -> dict[str, float]:
    """Return the bounded unicycle pure-pursuit yaw-rate command."""
    values = (
        current_x_m, current_y_m, current_yaw_rad, carrot_x_m, carrot_y_m,
        forward_command_m_s, maximum_angular_rad_s,
    )
    try:
        x, y, yaw, cx, cy, speed, limit = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("pure-pursuit inputs must be numeric") from exc
    if not _finite(x, y, yaw, cx, cy, speed, limit) or speed < 0.0 or limit <= 0.0:
        raise ValueError("pure-pursuit inputs must be finite with nonnegative speed and positive limit")
    dx, dy = cx - x, cy - y
    lookahead = math.hypot(dx, dy)
    if lookahead <= 1e-6:
        raise ValueError("pure-pursuit carrot must be separated from the current pose")
    bearing = math.atan2(dy, dx)
    alpha = _wrap(bearing - yaw)
    raw_rate = 2.0 * speed * math.sin(alpha) / lookahead
    command = max(-limit, min(limit, raw_rate))
    return {
        "bearing_error_rad": alpha,
        "lookahead_distance_m": lookahead,
        "raw_yaw_rate_rad_s": raw_rate,
        "command_yaw_rate_rad_s": command,
    }


def terminal_yaw_rate_command(
    *,
    current_yaw_rad: float,
    target_yaw_rad: float,
    measured_yaw_rate_rad_s: float,
    maximum_angular_rad_s: float,
    minimum_effective_angular_rad_s: float = POSE_GOAL_TERMINAL_MIN_ANGULAR_RAD_S,
    heading_gain: float = POSE_GOAL_TERMINAL_HEADING_GAIN,
    yaw_rate_damping: float = POSE_GOAL_TERMINAL_RATE_DAMPING,
    final_yaw_tolerance_rad: float = 0.17,
    stable_yaw_rate_limit_rad_s: float = math.radians(1.0),
) -> dict[str, float | bool]:
    """Bounded yaw-error plus measured-rate damping for a near-zero-forward phase."""
    values = (
        current_yaw_rad, target_yaw_rad, measured_yaw_rate_rad_s,
        maximum_angular_rad_s, minimum_effective_angular_rad_s, heading_gain,
        yaw_rate_damping, final_yaw_tolerance_rad, stable_yaw_rate_limit_rad_s,
    )
    try:
        yaw, target, rate, limit, minimum, kp, kd, tolerance, rate_limit = tuple(
            float(value) for value in values
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal yaw inputs must be numeric") from exc
    if not _finite(yaw, target, rate, limit, minimum, kp, kd, tolerance, rate_limit):
        raise ValueError("terminal yaw inputs must be finite")
    if limit <= 0.0 or minimum < 0.0 or minimum > limit or kp <= 0.0 or kd < 0.0 or tolerance <= 0.0 or rate_limit <= 0.0:
        raise ValueError("terminal yaw limits or gains are invalid")
    error = _wrap(target - yaw)
    settled = abs(error) <= tolerance and abs(rate) <= rate_limit
    raw = 0.0 if settled else kp * error - kd * rate
    if not settled and abs(raw) < minimum:
        direction = -math.copysign(1.0, rate) if abs(rate) > rate_limit else math.copysign(1.0, error)
        raw = direction * minimum
    command = max(-limit, min(limit, raw))
    return {
        "yaw_error_rad": error,
        "measured_yaw_rate_rad_s": rate,
        "raw_yaw_rate_command_rad_s": raw,
        "command_yaw_rate_rad_s": command,
        "within_yaw_and_rate_gates": settled,
    }


def fresh_yaw_rate_feedback(
    feedback: Mapping[str, Any] | None,
    *,
    sample_monotonic_s: float,
    maximum_age_s: float,
) -> dict[str, float]:
    """Require a measured yaw-rate fit that is fresh in the controller clock."""
    now = _number(sample_monotonic_s)
    max_age = _number(maximum_age_s)
    if now is None or max_age is None or max_age <= 0.0:
        raise ValueError("yaw-rate freshness limits must be finite and positive")
    if not isinstance(feedback, Mapping):
        raise ValueError("measured yaw-rate feedback is unavailable")
    rate = _number(feedback.get("signed_net_yaw_rate_rad_s"))
    measured_at = _number(feedback.get("sample_monotonic_s"))
    if rate is None or measured_at is None:
        raise ValueError("measured yaw-rate feedback is malformed")
    age = now - measured_at
    if not 0.0 <= age <= max_age:
        raise ValueError("measured yaw-rate feedback is stale or from the future")
    return {"signed_yaw_rate_rad_s": rate, "age_s": age}


def compensated_terminal_yaw_target_rad(
    requested_target_yaw_rad: float,
    measured_post_stop_yaw_drift_rad: float,
) -> float:
    """Bias the pre-stop setpoint against the measured signed post-stop drift."""
    try:
        target, drift = float(requested_target_yaw_rad), float(measured_post_stop_yaw_drift_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal yaw target and drift must be numeric") from exc
    if not _finite(target, drift):
        raise ValueError("terminal yaw target and drift must be finite")
    return _wrap(target - drift)


def active_locomotion_readback_confirmed(
    snapshot: Mapping[str, Any],
    *,
    expected_source: str,
    control_handshake_confirmed: bool,
) -> bool:
    """Require same-sample source and mode readbacks before terminal steering."""
    return bool(
        control_handshake_confirmed
        and snapshot.get("get_mode_header_code") == 0
        and snapshot.get("get_mode_desc") == "LOCOMOTION_DEFAULT"
        and snapshot.get("common_mode_desc") == "LOCOMOTION_DEFAULT"
        and snapshot.get("get_source_header_code") == 0
        and snapshot.get("get_source_name") == expected_source
        and snapshot.get("common_source_name") == expected_source
    )


@dataclass
class PoseGoalPhaseController:
    """Bounded APPROACH -> terminal yaw control -> ZERO_STOP state machine."""

    goal_distance_m: float = 0.30
    final_position_tolerance_m: float = 0.05
    maximum_lateral_error_m: float = 0.05
    maximum_overshoot_m: float = 0.05
    final_yaw_tolerance_rad: float = 0.17
    stable_yaw_rate_limit_rad_s: float = math.radians(1.0)
    terminal_yaw_setpoint_tolerance_rad: float | None = None
    terminal_ready_dwell_s: float = POSE_GOAL_TERMINAL_READY_DWELL_S
    terminal_timeout_s: float = POSE_GOAL_TERMINAL_CONTROL_MAX_S
    phase: str = APPROACH
    terminal_started_monotonic_s: float | None = None
    ready_since_monotonic_s: float | None = None
    position_trigger_reason: str | None = None
    terminal_pose_ready: bool = False
    failure_reason: str | None = None

    def update(
        self,
        *,
        monotonic_s: float,
        position_stop_reason: str | None,
        distance_to_goal_m: float,
        lateral_error_m: float,
        along_track_progress_m: float,
        yaw_error_rad: float,
        measured_yaw_rate_rad_s: float | None,
        terminal_setpoint_error_rad: float | None = None,
    ) -> dict[str, Any]:
        values = (
            monotonic_s, distance_to_goal_m, lateral_error_m, along_track_progress_m, yaw_error_rad,
        )
        if not _finite(*values):
            self.failure_reason = "pose_goal_nonfinite_state"
            self.phase = FAILED
            return self._decision(complete=True)
        if terminal_setpoint_error_rad is not None and not math.isfinite(float(terminal_setpoint_error_rad)):
            self.failure_reason = "pose_goal_nonfinite_terminal_setpoint_error"
            self.phase = FAILED
            return self._decision(complete=True)

        if self.phase == APPROACH:
            if position_stop_reason in POSE_SAFETY_STOP_REASONS:
                self.failure_reason = f"pose_goal_{position_stop_reason}"
                self.phase = FAILED
                return self._decision(complete=True)
            if position_stop_reason is not None and position_stop_reason not in POSE_POSITION_STOP_REASONS:
                self.failure_reason = f"pose_goal_unrecognized_position_stop_reason:{position_stop_reason}"
                self.phase = FAILED
                return self._decision(complete=True)
            if position_stop_reason not in POSE_POSITION_STOP_REASONS:
                return self._decision(complete=False)
            if (
                distance_to_goal_m > self.final_position_tolerance_m
                or abs(lateral_error_m) > self.maximum_lateral_error_m
            ):
                self.failure_reason = "pose_goal_terminal_entry_outside_position_gate"
                self.phase = FAILED
                return self._decision(complete=True)
            self.phase = TERMINAL_POSE_CONTROL
            self.position_trigger_reason = position_stop_reason
            self.terminal_started_monotonic_s = float(monotonic_s)
            return self._decision(complete=False, transitioned=True)

        if self.phase == TERMINAL_POSE_CONTROL:
            if position_stop_reason in POSE_SAFETY_STOP_REASONS:
                self.failure_reason = f"pose_goal_{position_stop_reason}"
            elif position_stop_reason is not None and position_stop_reason not in POSE_POSITION_STOP_REASONS:
                self.failure_reason = f"pose_goal_unrecognized_position_stop_reason:{position_stop_reason}"
            elif distance_to_goal_m > self.final_position_tolerance_m:
                self.failure_reason = "pose_goal_terminal_position_tolerance_exceeded"
            elif abs(lateral_error_m) > self.maximum_lateral_error_m:
                self.failure_reason = "pose_goal_terminal_lateral_guard"
            elif along_track_progress_m > self.goal_distance_m + self.maximum_overshoot_m:
                self.failure_reason = "pose_goal_terminal_overshoot_guard"
            elif measured_yaw_rate_rad_s is None or not math.isfinite(float(measured_yaw_rate_rad_s)):
                self.failure_reason = "pose_goal_terminal_yaw_rate_unavailable"
            if self.failure_reason:
                self.phase = FAILED
                return self._decision(complete=True)

            terminal_setpoint_tolerance = (
                self.final_yaw_tolerance_rad
                if self.terminal_yaw_setpoint_tolerance_rad is None
                else float(self.terminal_yaw_setpoint_tolerance_rad)
            )
            if not math.isfinite(terminal_setpoint_tolerance) or terminal_setpoint_tolerance <= 0.0:
                self.failure_reason = "pose_goal_invalid_terminal_setpoint_tolerance"
                self.phase = FAILED
                return self._decision(complete=True)
            yaw_ready = (
                abs(yaw_error_rad) <= self.final_yaw_tolerance_rad
                and abs(
                    yaw_error_rad if terminal_setpoint_error_rad is None
                    else float(terminal_setpoint_error_rad)
                ) <= terminal_setpoint_tolerance
                and abs(float(measured_yaw_rate_rad_s)) <= self.stable_yaw_rate_limit_rad_s
            )
            if yaw_ready:
                if self.ready_since_monotonic_s is None:
                    self.ready_since_monotonic_s = float(monotonic_s)
                if monotonic_s - self.ready_since_monotonic_s >= self.terminal_ready_dwell_s:
                    self.phase = ZERO_STOP
                    self.terminal_pose_ready = True
                    return self._decision(complete=True)
            else:
                self.ready_since_monotonic_s = None

            if (
                self.terminal_started_monotonic_s is not None
                and monotonic_s - self.terminal_started_monotonic_s >= self.terminal_timeout_s
            ):
                self.failure_reason = "pose_goal_terminal_yaw_timeout"
                self.phase = FAILED
                return self._decision(complete=True)

        if self.phase in (ZERO_STOP, FAILED):
            return self._decision(complete=True)
        return self._decision(complete=False)

    def _decision(self, *, complete: bool, transitioned: bool = False) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "complete": complete,
            "transitioned": transitioned,
            "terminal_pose_ready": self.terminal_pose_ready,
            "position_trigger_reason": self.position_trigger_reason,
            "failure_reason": self.failure_reason,
            "terminal_started_monotonic_s": self.terminal_started_monotonic_s,
            "terminal_ready_since_monotonic_s": self.ready_since_monotonic_s,
        }
