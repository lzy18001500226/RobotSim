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
POSE_GOAL_APPROACH_WARMUP_S = 0.5
POSE_GOAL_WARMUP_FORWARD_M_S = 0.12
POSE_GOAL_CURVE_FORWARD_M_S = 0.13
POSE_GOAL_CURVE_ANGULAR_UTILIZATION = 0.95
POSE_GOAL_PATH_ENDPOINT_OFFSET_LIMIT_M = 0.05
# Runs93/94, which used the coordinated pose controller, measured +0.1695 and
# +0.1887 rad from the first zero command to the settled pose. A 0.05 rad
# pre-stop bias leaves at least 0.031 rad inside the unchanged 0.17 rad final
# yaw tolerance for those two observations. This is an empirical design point,
# not a calibrated or guaranteed settling model; the actual final gate remains
# authoritative.
POSE_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD = 0.05
POSE_GOAL_RELEVANT_POST_STOP_DRIFT_OBSERVATIONS_RAD = (0.16952762228006262, 0.1886899134656192)
POSE_GOAL_PRESTOP_YAW_MARGIN_RAD = 0.03
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


def pose_path_target_with_yaw_compensation(
    goal_pose: Mapping[str, Any],
    measured_post_stop_yaw_drift_rad: float,
) -> dict[str, Any]:
    """Keep the goal position but bias the approach path's endpoint tangent."""
    if not isinstance(goal_pose, Mapping):
        raise ValueError("pose path goal must be a mapping")
    result = dict(goal_pose)
    if "yaw" not in result:
        raise ValueError("pose path goal is missing yaw")
    result["yaw"] = compensated_terminal_yaw_target_rad(
        result["yaw"], measured_post_stop_yaw_drift_rad,
    )
    return result


def pose_goal_approach_target(
    start_pose: Mapping[str, Any],
    stop_pose: Mapping[str, Any],
    requested_target_yaw_rad: float,
    pre_stop_yaw_bias_rad: float,
    *,
    maximum_endpoint_offset_m: float = 0.05,
) -> dict[str, Any]:
    """Build a bounded approach endpoint with balanced start/end tangent turns.

    The requested stop position stays the acceptance target. The path endpoint
    may move within its position tolerance: its chord direction is the angular
    midpoint between the measured start yaw and drift-biased terminal yaw. This
    avoids forcing the whole tangent change into one end of a short Hermite
    chord. The offset is reported and rejected when it exceeds the caller's
    positional budget.
    """
    try:
        x0, y0, start_yaw = (float(start_pose[key]) for key in ("x", "y", "yaw"))
        x1, y1 = (float(stop_pose[key]) for key in ("x", "y"))
        target_yaw = float(requested_target_yaw_rad)
        bias = float(pre_stop_yaw_bias_rad)
        max_offset = float(maximum_endpoint_offset_m)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("approach target requires numeric start, stop, and yaw values") from exc
    if not _finite(x0, y0, start_yaw, x1, y1, target_yaw, bias, max_offset):
        raise ValueError("approach target inputs must be finite")
    start_frame = start_pose.get("frame_id")
    stop_frame = stop_pose.get("frame_id")
    if start_frame != stop_frame or start_frame not in (None, "odom"):
        raise ValueError("approach target start and stop frames must match in odom")
    if max_offset < 0.0:
        raise ValueError("maximum endpoint offset must be nonnegative")
    chord_length = math.hypot(x1 - x0, y1 - y0)
    if chord_length <= 1e-9:
        raise ValueError("approach stop position must be separated from the start")

    terminal_yaw = compensated_terminal_yaw_target_rad(target_yaw, bias)
    yaw_delta = _wrap(terminal_yaw - start_yaw)
    chord_yaw = _wrap(start_yaw + 0.5 * yaw_delta)
    endpoint_x = x0 + chord_length * math.cos(chord_yaw)
    endpoint_y = y0 + chord_length * math.sin(chord_yaw)
    offset = math.hypot(endpoint_x - x1, endpoint_y - y1)
    if offset > max_offset + 1e-12:
        raise ValueError(
            f"balanced approach endpoint offset {offset:.6f} m exceeds {max_offset:.6f} m"
        )
    return {
        "frame_id": stop_frame,
        "x": endpoint_x,
        "y": endpoint_y,
        "yaw": terminal_yaw,
        "requested_stop_x_m": x1,
        "requested_stop_y_m": y1,
        "requested_stop_yaw_rad": target_yaw,
        "approach_chord_yaw_rad": chord_yaw,
        "approach_chord_length_m": chord_length,
        "approach_endpoint_offset_m": offset,
        "pre_stop_yaw_bias_rad": bias,
    }


def sampled_pose_path_feasibility(
    start_pose: Mapping[str, Any],
    endpoint_pose: Mapping[str, Any],
    *,
    reference_line_pose: Mapping[str, Any] | None = None,
    forward_command_m_s: float,
    maximum_angular_rad_s: float,
    lookahead_fraction: float = POSE_GOAL_LOOKAHEAD_FRACTION,
    sample_count: int = 201,
    zero_angular_warmup_s: float = 0.0,
    warmup_forward_m_s: float | None = None,
    terminal_yaw_control_budget_s: float = 0.0,
) -> dict[str, Any]:
    """Measure path geometry and ideal pure-pursuit angular demand offline.

    This is a kinematic feasibility check, not a prediction of MuJoCo/MC
    tracking. Runtime callers fail closed if the sampled command exceeds the
    existing angular cap; the saved result retains both geometry and limits.
    """
    try:
        speed = float(forward_command_m_s)
        angular_limit = float(maximum_angular_rad_s)
        lookahead = float(lookahead_fraction)
        count = int(sample_count)
        warmup_s = float(zero_angular_warmup_s)
        warmup_speed = speed if warmup_forward_m_s is None else float(warmup_forward_m_s)
        terminal_budget = float(terminal_yaw_control_budget_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("path feasibility limits must be numeric") from exc
    if not _finite(speed, angular_limit, lookahead, warmup_s, warmup_speed, terminal_budget) or speed <= 0.0 or angular_limit <= 0.0:
        raise ValueError("path feasibility limits must be finite and positive")
    if not 0.0 < lookahead <= 1.0 or count < 2 or warmup_s < 0.0 or warmup_speed <= 0.0 or terminal_budget < 0.0:
        raise ValueError("path feasibility sampling parameters are invalid")

    references = [pose_path_reference(start_pose, endpoint_pose, i / (count - 1)) for i in range(count)]
    segment_lengths = [
        math.hypot(
            references[i + 1]["x_m"] - references[i]["x_m"],
            references[i + 1]["y_m"] - references[i]["y_m"],
        )
        for i in range(count - 1)
    ]
    path_length = sum(segment_lengths)
    cumulative_lengths = [0.0]
    for segment_length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + segment_length)
    max_curvature = max(abs(row["curvature_per_m"]) for row in references)
    required_rate = max_curvature * speed
    rate_limited_speed = angular_limit / max_curvature if max_curvature > 1e-12 else math.inf

    pp_commands: list[float] = []
    pp_raw_commands: list[float] = []
    for index, reference in enumerate(references[:-1]):
        carrot_fraction = min(1.0, index / (count - 1) + lookahead)
        carrot = pose_path_reference(start_pose, endpoint_pose, carrot_fraction)
        command = pure_pursuit_yaw_rate(
            current_x_m=reference["x_m"],
            current_y_m=reference["y_m"],
            current_yaw_rad=reference["tangent_yaw_rad"],
            carrot_x_m=carrot["x_m"],
            carrot_y_m=carrot["y_m"],
            forward_command_m_s=speed,
            maximum_angular_rad_s=angular_limit,
        )
        pp_commands.append(float(command["command_yaw_rate_rad_s"]))
        pp_raw_commands.append(float(command["raw_yaw_rate_rad_s"]))
    max_pp_rate = max((abs(value) for value in pp_commands), default=0.0)
    max_pp_raw_rate = max((abs(value) for value in pp_raw_commands), default=0.0)
    chord_dx = float(endpoint_pose["x"]) - float(start_pose["x"])
    chord_dy = float(endpoint_pose["y"]) - float(start_pose["y"])
    chord_norm = math.hypot(chord_dx, chord_dy)
    lateral_samples = [
        abs(chord_dx * (row["y_m"] - float(start_pose["y"]))
            - chord_dy * (row["x_m"] - float(start_pose["x"]))) / chord_norm
        for row in references
    ]
    reference = reference_line_pose or endpoint_pose
    ref_dx = float(reference["x"]) - float(start_pose["x"])
    ref_dy = float(reference["y"]) - float(start_pose["y"])
    ref_norm = math.hypot(ref_dx, ref_dy)
    if ref_norm <= 1e-9:
        raise ValueError("reference line stop pose must be separated from the start")
    reference_lateral_samples = [
        abs(ref_dx * (row["y_m"] - float(start_pose["y"]))
            - ref_dy * (row["x_m"] - float(start_pose["x"]))) / ref_norm
        for row in references
    ]
    correction = abs(_wrap(float(endpoint_pose["yaw"]) - float(start_pose["yaw"])))
    warmup_distance = min(path_length, warmup_s * warmup_speed)
    warmup_index = min(
        count - 1,
        next((index for index, distance in enumerate(cumulative_lengths) if distance >= warmup_distance), count - 1),
    )
    warmup_reference = references[warmup_index]
    remaining_path_length = max(0.0, path_length - cumulative_lengths[warmup_index])
    # No angular command is sent during this warmup; conservatively assume the
    # body yaw itself does not advance with the geometric tangent.
    remaining_yaw_correction = abs(
        _wrap(float(endpoint_pose["yaw"]) - float(start_pose["yaw"]))
    )
    remaining_yaw_time = remaining_yaw_correction / angular_limit
    remaining_path_time = remaining_path_length / speed
    terminal_required_time = max(0.0, remaining_yaw_time - remaining_path_time)
    ideal_integral = sum(
        0.5 * (pp_commands[i] + pp_commands[i + 1]) * segment_lengths[i] / speed
        for i in range(min(len(pp_commands) - 1, len(segment_lengths)))
    )
    return {
        "classification": "KINEMATIC_PATH_FEASIBILITY_NOT_DYNAMICS_VALIDATION",
        "sample_count": count,
        "path_length_m": path_length,
        "chord_length_m": chord_norm,
        "maximum_absolute_curvature_per_m": max_curvature,
        "required_angular_rate_at_forward_command_rad_s": required_rate,
        "curvature_rate_feasible_at_command": required_rate <= angular_limit + 1e-12,
        "maximum_forward_speed_under_angular_cap_m_s": rate_limited_speed,
        "commanded_forward_speed_m_s": speed,
        "maximum_sampled_pure_pursuit_rate_rad_s": max_pp_rate,
        "maximum_sampled_pure_pursuit_raw_rate_rad_s": max_pp_raw_rate,
        "pure_pursuit_saturation_fraction": (
            sum(abs(value) >= angular_limit - 1e-12 for value in pp_raw_commands) / len(pp_raw_commands)
            if pp_raw_commands else 0.0
        ),
        "pure_pursuit_rate_feasible_at_command": max_pp_raw_rate <= angular_limit + 1e-12,
        "ideal_command_integral_rad": ideal_integral,
        "required_yaw_change_rad": correction,
        "minimum_yaw_time_at_angular_cap_s": correction / angular_limit,
        "nominal_path_time_at_forward_command_s": path_length / speed,
        "zero_angular_warmup_s": warmup_s,
        "warmup_forward_command_m_s": warmup_speed,
        "warmup_commanded_distance_m": warmup_distance,
        "warmup_path_fraction": warmup_index / (count - 1),
        "geometric_tangent_change_during_zero_angular_warmup_rad": abs(
            _wrap(warmup_reference["tangent_yaw_rad"] - float(start_pose["yaw"]))
        ),
        "remaining_yaw_correction_after_warmup_rad": remaining_yaw_correction,
        "minimum_remaining_yaw_time_at_angular_cap_s": remaining_yaw_time,
        "remaining_path_time_at_forward_command_s": remaining_path_time,
        "remaining_yaw_feasible_after_warmup": remaining_yaw_time <= remaining_path_time + 1e-12,
        "remaining_yaw_time_to_terminal_phase_at_cap_s": terminal_required_time,
        "terminal_yaw_control_budget_s": terminal_budget,
        "remaining_yaw_feasible_with_terminal_budget": terminal_required_time <= terminal_budget + 1e-12,
        "nominal_total_approach_time_with_warmup_s": warmup_s + remaining_path_time,
        "maximum_deviation_from_stop_chord_m": max(lateral_samples),
        "maximum_deviation_from_requested_stop_line_m": max(reference_lateral_samples),
        "solver_or_mc_tracking_predicted": False,
    }


def pose_goal_approach_path_reference(
    start_pose: Mapping[str, Any],
    path: Mapping[str, Any],
    progress_fraction: float,
) -> dict[str, float]:
    """Reference a straight warmup segment followed by the curved pose path."""
    try:
        fraction = float(progress_fraction)
        warmup_distance = float(path["warmup_distance_m"])
        total_length = float(path["total_path_length_m"])
        curve_length = float(path["curve_path_length_m"])
        curve_start = path["curve_start_pose"]
        curve_endpoint = path["curve_endpoint_pose"]
        x0, y0, yaw0 = (float(start_pose[key]) for key in ("x", "y", "yaw"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("warmup-aware path reference is malformed") from exc
    if not _finite(fraction, warmup_distance, total_length, curve_length, x0, y0, yaw0):
        raise ValueError("warmup-aware path reference must be finite")
    if total_length <= 0.0 or curve_length <= 0.0 or warmup_distance < 0.0:
        raise ValueError("warmup-aware path lengths must be positive")
    t = min(1.0, max(0.0, fraction))
    distance = min(total_length, t * total_length)
    if distance <= warmup_distance:
        x = x0 + distance * math.cos(yaw0)
        y = y0 + distance * math.sin(yaw0)
        return {
            "x_m": x,
            "y_m": y,
            "tangent_yaw_rad": yaw0,
            "curvature_per_m": 0.0,
            "progress_fraction": t,
            "path_distance_m": distance,
            "chord_length_m": float(path["projected_stop_progress_m"]),
        }
    curve_fraction = min(1.0, max(0.0, (distance - warmup_distance) / curve_length))
    reference = pose_path_reference(curve_start, curve_endpoint, curve_fraction)
    reference["path_distance_m"] = distance
    reference["chord_length_m"] = float(path["projected_stop_progress_m"])
    return reference


def build_warmup_aware_pose_goal_path(
    start_pose: Mapping[str, Any],
    stop_pose: Mapping[str, Any],
    requested_target_yaw_rad: float,
    pre_stop_yaw_bias_rad: float,
    *,
    warmup_duration_s: float,
    warmup_forward_m_s: float,
    curve_forward_m_s: float,
    maximum_angular_rad_s: float,
    maximum_endpoint_offset_m: float = 0.05,
    angular_utilization_limit: float = 0.95,
    lookahead_fraction: float = POSE_GOAL_LOOKAHEAD_FRACTION,
    sample_count: int = 201,
    maximum_curve_extension_m: float = 0.10,
) -> dict[str, Any]:
    """Build the shortest warmup-aligned curve satisfying angular and position bounds.

    A bounded 1 mm scalar chord-length search extends only the curved segment
    as needed. It does not search robot poses or controller gains. The initial
    straight segment exactly matches the existing zero-angular forward warmup.
    """
    try:
        x0, y0, start_yaw = (float(start_pose[key]) for key in ("x", "y", "yaw"))
        x_stop, y_stop = (float(stop_pose[key]) for key in ("x", "y"))
        target_yaw = float(requested_target_yaw_rad)
        bias = float(pre_stop_yaw_bias_rad)
        warmup_s = float(warmup_duration_s)
        warmup_speed = float(warmup_forward_m_s)
        curve_speed = float(curve_forward_m_s)
        angular_limit = float(maximum_angular_rad_s)
        endpoint_limit = float(maximum_endpoint_offset_m)
        utilization_limit = float(angular_utilization_limit)
        lookahead = float(lookahead_fraction)
        max_extension = float(maximum_curve_extension_m)
        count = int(sample_count)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("warmup-aware path inputs must be numeric") from exc
    if not _finite(
        x0, y0, start_yaw, x_stop, y_stop, target_yaw, bias, warmup_s,
        warmup_speed, curve_speed, angular_limit, endpoint_limit,
        utilization_limit, lookahead, max_extension,
    ):
        raise ValueError("warmup-aware path inputs must be finite")
    if start_pose.get("frame_id") != stop_pose.get("frame_id") or start_pose.get("frame_id") not in (None, "odom"):
        raise ValueError("warmup-aware path start and stop frames must match in odom")
    if (
        warmup_s < 0.0 or warmup_speed <= 0.0 or curve_speed <= 0.0
        or angular_limit <= 0.0 or endpoint_limit <= 0.0
        or not 0.0 < utilization_limit <= 1.0
        or not 0.0 < lookahead <= 1.0 or count < 11 or max_extension < 0.0
    ):
        raise ValueError("warmup-aware path bounds are invalid")

    warmup_distance = warmup_s * warmup_speed
    stop_dx, stop_dy = x_stop - x0, y_stop - y0
    stop_distance = math.hypot(stop_dx, stop_dy)
    if stop_distance <= warmup_distance + 1e-6:
        raise ValueError("stop target is too close for the configured warmup segment")
    stop_unit = (stop_dx / stop_distance, stop_dy / stop_distance)
    curve_start = {
        "frame_id": start_pose.get("frame_id"),
        "x": x0 + warmup_distance * math.cos(start_yaw),
        "y": y0 + warmup_distance * math.sin(start_yaw),
        "yaw": start_yaw,
    }
    terminal_yaw = compensated_terminal_yaw_target_rad(target_yaw, bias)
    yaw_delta = _wrap(terminal_yaw - start_yaw)
    curve_chord_yaw = _wrap(start_yaw + 0.5 * yaw_delta)
    base_curve_chord = max(stop_distance - warmup_distance, 1e-6)
    steps = int(math.floor(max_extension / 0.001))
    last_endpoint_offset = None
    for step in range(steps + 1):
        curve_chord = base_curve_chord + step * 0.001
        curve_endpoint = {
            "frame_id": stop_pose.get("frame_id"),
            "x": curve_start["x"] + curve_chord * math.cos(curve_chord_yaw),
            "y": curve_start["y"] + curve_chord * math.sin(curve_chord_yaw),
            "yaw": terminal_yaw,
        }
        endpoint_offset = math.hypot(curve_endpoint["x"] - x_stop, curve_endpoint["y"] - y_stop)
        last_endpoint_offset = endpoint_offset
        if endpoint_offset > endpoint_limit + 1e-12:
            if step > 0:
                break
            continue

        curve_geometry = sampled_pose_path_feasibility(
            curve_start,
            curve_endpoint,
            forward_command_m_s=curve_speed,
            maximum_angular_rad_s=angular_limit,
            lookahead_fraction=lookahead,
            sample_count=count,
        )
        if curve_geometry["maximum_sampled_pure_pursuit_raw_rate_rad_s"] > angular_limit * utilization_limit + 1e-12:
            continue

        curve_length = float(curve_geometry["path_length_m"])
        total_length = warmup_distance + curve_length
        projected_stop_progress = (
            (curve_endpoint["x"] - x0) * stop_unit[0]
            + (curve_endpoint["y"] - y0) * stop_unit[1]
        )
        if projected_stop_progress <= stop_distance:
            continue
        path_model = {
            "schema": "robotsim.issue80.warmup_aware_pose_path.v1",
            "frame_id": start_pose.get("frame_id"),
            "warmup_distance_m": warmup_distance,
            "warmup_duration_s": warmup_s,
            "warmup_forward_m_s": warmup_speed,
            "curve_forward_m_s": curve_speed,
            "curve_start_pose": curve_start,
            "curve_endpoint_pose": curve_endpoint,
            "curve_chord_length_m": curve_chord,
            "curve_path_length_m": curve_length,
            "total_path_length_m": total_length,
            "projected_stop_progress_m": projected_stop_progress,
            "requested_stop_pose": {"frame_id": stop_pose.get("frame_id"), "x": x_stop, "y": y_stop},
            "requested_stop_yaw_rad": target_yaw,
            "pre_stop_yaw_bias_rad": bias,
            "endpoint_position_offset_m": endpoint_offset,
            "curve_geometry": curve_geometry,
        }
        warmup_fraction = min(1.0, warmup_distance / total_length)
        raw_rates = [0.0]
        transition_current = pose_goal_approach_path_reference(start_pose, path_model, warmup_fraction)
        transition_carrot = pose_goal_approach_path_reference(
            start_pose, path_model, min(1.0, warmup_fraction + lookahead),
        )
        transition = pure_pursuit_yaw_rate(
            current_x_m=transition_current["x_m"],
            current_y_m=transition_current["y_m"],
            current_yaw_rad=start_yaw,
            carrot_x_m=transition_carrot["x_m"],
            carrot_y_m=transition_carrot["y_m"],
            forward_command_m_s=curve_speed,
            maximum_angular_rad_s=angular_limit,
        )
        raw_rates.append(abs(float(transition["raw_yaw_rate_rad_s"])))
        max_fraction = max(0.0, 1.0 - lookahead)
        reference_fractions = [
            warmup_fraction + (max_fraction - warmup_fraction) * index / (count - 1)
            for index in range(count)
        ]
        lateral_deviations = []
        for index in range(count):
            fraction = index / (count - 1)
            reference = pose_goal_approach_path_reference(start_pose, path_model, fraction)
            lateral_deviations.append(
                abs(stop_dx * (reference["y_m"] - y0) - stop_dy * (reference["x_m"] - x0)) / stop_distance
            )
        for fraction in reference_fractions:
            current = pose_goal_approach_path_reference(start_pose, path_model, fraction)
            carrot = pose_goal_approach_path_reference(
                start_pose, path_model, min(1.0, fraction + lookahead),
            )
            command = pure_pursuit_yaw_rate(
                current_x_m=current["x_m"],
                current_y_m=current["y_m"],
                current_yaw_rad=current["tangent_yaw_rad"],
                carrot_x_m=carrot["x_m"],
                carrot_y_m=carrot["y_m"],
                forward_command_m_s=curve_speed,
                maximum_angular_rad_s=angular_limit,
            )
            raw_rates.append(abs(float(command["raw_yaw_rate_rad_s"])))
        maximum_raw_rate = max(raw_rates)
        if maximum_raw_rate > angular_limit * utilization_limit + 1e-12:
            continue
        if max(lateral_deviations, default=0.0) > 0.05 + 1e-12:
            continue
        stop_fraction = min(1.0, stop_distance / projected_stop_progress)
        stop_reference = pose_goal_approach_path_reference(start_pose, path_model, stop_fraction)
        terminal_entry_position_error = math.hypot(
            stop_reference["x_m"] - x_stop,
            stop_reference["y_m"] - y_stop,
        )
        terminal_entry_lateral = abs(
            stop_dx * (stop_reference["y_m"] - y0)
            - stop_dy * (stop_reference["x_m"] - x0)
        ) / stop_distance
        terminal_residual = abs(_wrap(terminal_yaw - stop_reference["tangent_yaw_rad"]))
        curve_elapsed_to_stop = max(0.0, stop_fraction * total_length - warmup_distance) / curve_speed
        path_model["feasibility"] = {
            "classification": "KINEMATIC_WARMUP_ALIGNED_PATH_NOT_DYNAMICS_VALIDATION",
            "sample_count": count,
            "search_step_m": 0.001,
            "curve_extension_from_minimum_m": curve_chord - base_curve_chord,
            "endpoint_position_error_m": endpoint_offset,
            "maximum_lateral_error_to_requested_stop_line_m": max(lateral_deviations, default=0.0),
            "maximum_curvature_per_m": curve_geometry["maximum_absolute_curvature_per_m"],
            "required_angular_rate_at_curve_forward_command_rad_s": curve_geometry["required_angular_rate_at_forward_command_rad_s"],
            "maximum_ideal_or_warmup_transition_raw_angular_rate_rad_s": maximum_raw_rate,
            "angular_command_limit_rad_s": angular_limit,
            "angular_rate_utilization_limit": utilization_limit,
            "maximum_command_margin_rad_s": angular_limit - maximum_raw_rate,
            "warmup_transition_raw_rate_rad_s": abs(float(transition["raw_yaw_rate_rad_s"])),
            "warmup_transition_saturation_required": abs(float(transition["raw_yaw_rate_rad_s"])) > angular_limit,
            "minimum_yaw_correction_time_at_command_cap_s": abs(yaw_delta) / angular_limit,
            "curve_time_after_warmup_s": curve_length / curve_speed,
            "nominal_total_approach_time_s": warmup_s + curve_length / curve_speed,
            "nominal_terminal_entry_time_s": warmup_s + curve_elapsed_to_stop,
            "terminal_entry_progress_fraction": stop_fraction,
            "terminal_entry_position_error_m": terminal_entry_position_error,
            "terminal_entry_lateral_error_m": terminal_entry_lateral,
            "terminal_entry_reference_yaw_rad": stop_reference["tangent_yaw_rad"],
            "terminal_entry_residual_to_pre_stop_target_rad": terminal_residual,
            "terminal_entry_residual_time_at_command_cap_s": terminal_residual / angular_limit,
            "terminal_ready_dwell_s": POSE_GOAL_TERMINAL_READY_DWELL_S,
            "terminal_control_max_s": POSE_GOAL_TERMINAL_CONTROL_MAX_S,
            "terminal_residual_and_dwell_fit": (
                terminal_residual / angular_limit + POSE_GOAL_TERMINAL_READY_DWELL_S
                <= POSE_GOAL_TERMINAL_CONTROL_MAX_S
            ),
            "post_stop_yaw_drift_is_calibrated": False,
            "mujoco_or_mc_tracking_predicted": False,
        }
        return path_model
    raise ValueError(
        "no warmup-aligned path satisfies angular command headroom and the existing 50 mm positional corridor "
        f"within {max_extension:.3f} m extension (last endpoint offset {last_endpoint_offset})"
    )


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
