#!/usr/bin/env python3
"""Build timestamp-aligned offline evidence for Issue 80 heading-goal runs."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from issue80_coordinated_pose_goal import (
    POSE_GOAL_LOOKAHEAD_FRACTION,
    fresh_yaw_rate_feedback,
    pose_path_reference,
    pure_pursuit_yaw_rate,
    terminal_yaw_rate_command,
)


DEFAULT_REVIEW_ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-80")
DEFAULT_RUNS = {
    "run86": "stage-b-heading-goal-run86-20261022T190133Z",
    "run88": "stage-b-heading-goal-run88-20261023-20261023T005951Z",
    "run89": "stage-b-heading-goal-run89-20261023-20261023T012443Z",
    "run90": "stage-b-heading-goal-run90-20261023T0150Z-f68457f3",
    "run91": "stage-b-heading-goal-run91-20261023T0204Z-b00f4a02",
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def wrapped_delta(end: Any, start: Any) -> float | None:
    end_value = _number(end)
    start_value = _number(start)
    if end_value is None or start_value is None:
        return None
    delta = end_value - start_value
    return math.atan2(math.sin(delta), math.cos(delta))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _linear_slope(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    t0 = samples[0][0]
    times = [t - t0 for t, _ in samples]
    values = [v for _, v in samples]
    mean_t = statistics.fmean(times)
    mean_v = statistics.fmean(values)
    denom = sum((t - mean_t) ** 2 for t in times)
    if denom <= 0.0:
        return None
    return sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denom


def _unwrapped_yaw_samples(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    previous_raw: float | None = None
    unwrapped = 0.0
    for row in rows:
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        pose = snapshot.get("pose")
        if not isinstance(pose, Mapping):
            continue
        timestamp = _number(snapshot.get("sample_monotonic_s"))
        yaw = _number(pose.get("yaw"))
        if timestamp is None or yaw is None:
            continue
        if previous_raw is None:
            unwrapped = yaw
        else:
            delta = wrapped_delta(yaw, previous_raw)
            if delta is None:
                continue
            unwrapped += delta
        previous_raw = yaw
        samples.append((timestamp, unwrapped))
    return samples


def _window_yaw_rate(samples: list[tuple[float, float]], start: float, end: float) -> float | None:
    selected = [(t, yaw) for t, yaw in samples if start <= t <= end]
    return _linear_slope(selected)


def _asof_state(state_rows: list[dict[str, Any]], times: list[float], timestamp: float) -> dict[str, Any] | None:
    index = bisect.bisect_right(times, timestamp) - 1
    if index < 0:
        return None
    return state_rows[index]


def _yaw_rate_at(state_rows: list[dict[str, Any]], timestamp: float, window_s: float = 0.4) -> tuple[float | None, float]:
    samples = []
    previous_raw: float | None = None
    unwrapped = 0.0
    for row in state_rows:
        snapshot = row.get("snapshot", {})
        t = _number(snapshot.get("sample_monotonic_s"))
        pose = snapshot.get("pose", {})
        yaw = _number(pose.get("yaw")) if isinstance(pose, Mapping) else None
        if t is None or yaw is None or t > timestamp or t < timestamp - window_s:
            continue
        if previous_raw is None:
            unwrapped = yaw
        else:
            delta = wrapped_delta(yaw, previous_raw)
            if delta is None:
                continue
            unwrapped += delta
        previous_raw = yaw
        samples.append((t, unwrapped))
    coverage = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    if coverage < window_s - 0.02:
        return None, coverage
    return _linear_slope(samples), coverage


def _event_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    commands = [row for row in events if row.get("event") == "velocity_command"]
    returns = [row for row in events if row.get("event") == "velocity_publish_result"]
    command_times = [_number(row.get("wall_monotonic_sec")) for row in commands]
    command_times = [t for t in command_times if t is not None]
    gaps = [b - a for a, b in zip(command_times, command_times[1:])]
    durations = []
    for command, returned in zip(commands, returns):
        start = _number(command.get("wall_monotonic_sec"))
        end = _number(returned.get("wall_monotonic_sec"))
        if start is not None and end is not None and end >= start:
            durations.append(end - start)
    nonzero = [
        row for row in commands
        if any(abs(_number(row.get(key)) or 0.0) > 1e-9 for key in (
            "forward_velocity_m_s", "lateral_velocity_m_s", "angular_velocity_rad_s",
        ))
    ]
    zeros = [row for row in commands if row not in nonzero]

    def timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
        times = [_number(row.get("wall_monotonic_sec")) for row in rows]
        times = [t for t in times if t is not None]
        periods = [b - a for a, b in zip(times, times[1:])]
        return {
            "count": len(rows),
            "period_median_s": statistics.median(periods) if periods else None,
            "period_p95_s": sorted(periods)[min(len(periods) - 1, int(0.95 * len(periods)))] if periods else None,
            "period_max_s": max(periods) if periods else None,
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in commands:
        command_group = "nonzero" if row in nonzero else "zero"
        key = f"{row.get('phase')}|{row.get('command_kind')}|{command_group}"
        groups.setdefault(key, []).append(row)
    return {
        "command_event_count": len(commands),
        "publish_return_count": len(returns),
        "nonzero_command_event_count": len(nonzero),
        "zero_command_event_count": len(zeros),
        "publish_return_ratio": len(returns) / len(commands) if commands else None,
        "command_period_median_s": statistics.median(gaps) if gaps else None,
        "command_period_p95_s": sorted(gaps)[min(len(gaps) - 1, int(0.95 * len(gaps)))] if gaps else None,
        "command_period_max_s": max(gaps) if gaps else None,
        "publisher_call_duration_median_s": statistics.median(durations) if durations else None,
        "publisher_call_duration_max_s": max(durations) if durations else None,
        "period_by_phase_kind_and_zero_state": {key: timing(rows) for key, rows in sorted(groups.items())},
        "mc_receipt_ack_available": False,
        "interpretation": (
            "velocity_command and velocity_publish_result establish local publication calls only; "
            "active input-source readback and changing MC joint targets are independent response evidence, "
            "not per-message command receipt acknowledgement"
        ),
    }


def _mc_output_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for row in rows:
        if row.get("event") != "state_sample":
            continue
        snapshot = row.get("snapshot")
        if isinstance(snapshot, Mapping) and isinstance(snapshot.get("mc_leg_command"), Mapping):
            samples.append((row.get("phase"), snapshot))
    out: dict[str, Any] = {}
    for phase_name in ("point-goal-activation", "point-goal-post-stop"):
        phase_samples = [s for phase, s in samples if phase == phase_name]
        per_joint: dict[str, list[float]] = {}
        changing_samples = 0
        previous: dict[str, float] | None = None
        for snapshot in phase_samples:
            command = snapshot.get("mc_leg_command", {})
            positions = {
                str(j.get("name")): _number(j.get("position"))
                for j in command.get("joints", []) if isinstance(j, Mapping)
            }
            positions = {name: value for name, value in positions.items() if value is not None}
            if previous is not None and any(
                abs(value - previous.get(name, value)) > 1e-5 for name, value in positions.items()
            ):
                changing_samples += 1
            for name, value in positions.items():
                per_joint.setdefault(name, []).append(value)
            previous = positions
        out[phase_name] = {
            "sample_count_with_leg_targets": len(phase_samples),
            "target_vector_change_samples_gt_1e-5": changing_samples,
            "joint_target_ranges_rad": {
                name: max(values) - min(values) for name, values in sorted(per_joint.items()) if values
            },
            "leg_command_receive_monotonic_available": all(
                isinstance(snapshot.get("mc_leg_command", {}).get("received_monotonic_s"), (int, float))
                for snapshot in phase_samples
            ) if phase_samples else False,
        }
    return out


def _rl_debug_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_CAPTURED"}
    rows = _load_jsonl(path)
    obs_lengths = []
    action_lengths = []
    seqs = []
    stamps = []
    action_changed = 0
    previous_action: tuple[float, ...] | None = None
    for row in rows:
        payload = row.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        obs = payload.get("obs_vec")
        action = payload.get("action_vec")
        if isinstance(obs, list):
            obs_lengths.append(len(obs))
        if isinstance(action, list):
            action_tuple = tuple(float(x) for x in action if _number(x) is not None)
            action_lengths.append(len(action_tuple))
            if previous_action is not None and action_tuple != previous_action:
                action_changed += 1
            previous_action = action_tuple
        header = payload.get("header", {})
        if isinstance(header, Mapping):
            if isinstance(header.get("sequence"), int):
                seqs.append(header["sequence"])
            stamp = header.get("stamp", {})
            if isinstance(stamp, Mapping):
                sec, ns = _number(stamp.get("sec")), _number(stamp.get("nanosec"))
                if sec is not None and ns is not None:
                    stamps.append(sec + ns * 1e-9)
    regressions = sum(b < a for a, b in zip(stamps, stamps[1:]))
    duplicates = sum(b == a for a, b in zip(stamps, stamps[1:]))
    return {
        "status": "CAPTURED" if rows else "EMPTY",
        "message_count": len(rows),
        "observation_vector_lengths": sorted(set(obs_lengths)),
        "action_vector_lengths": sorted(set(action_lengths)),
        "action_vector_change_count": action_changed,
        "sequence_monotonic": all(b > a for a, b in zip(seqs, seqs[1:])),
        "source_stamp_regression_count": regressions,
        "source_stamp_duplicate_count": duplicates,
        "semantics": "raw version-matched float arrays; no index meanings are inferred",
    }


def _mc_timing_stats(
    run_dir: Path,
    state_rows: list[dict[str, Any]],
    *,
    active_start: float | None,
    zero_start: float | None,
    stand_confirmed: float | None,
    physical_window_start: float | None,
    physical_window_end: float | None,
) -> dict[str, Any]:
    log_path = run_dir / "mc-console.log"
    if not log_path.is_file():
        log_path = run_dir / "mc_logs/mc.log"
    if not log_path.is_file():
        return {"status": "NOT_CAPTURED"}
    offsets = []
    for row in state_rows:
        snapshot = row.get("snapshot", {})
        wall = snapshot.get("wall_utc")
        mono = _number(snapshot.get("sample_monotonic_s"))
        if isinstance(wall, str) and mono is not None:
            try:
                wall_time = datetime.fromisoformat(wall.replace("Z", "+00:00"))
                if wall_time.tzinfo is None:
                    wall_time = wall_time.replace(tzinfo=timezone.utc)
                offsets.append(mono - wall_time.timestamp())
            except ValueError:
                pass
    if not offsets:
        return {"status": "BLOCKED_NO_WALL_MONOTONIC_ALIGNMENT"}
    clock_offset = statistics.median(offsets)
    pattern = re.compile(
        r"^\[(?P<wall>[^]]+)\].*?\[(?:Warn|Warning)\].*?'(?P<worker>[^']+)' "
        r"worker missed wakeup time! diff (?P<diff>\d+) us, period: (?P<period>\d+) us, "
        r"miss (?P<miss>[0-9.]+)%"
    )
    events = []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ansi.sub("", raw_line)
        match = pattern.search(line)
        if not match:
            continue
        try:
            wall_time = datetime.strptime(match.group("wall"), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        host_mono = wall_time.timestamp() + clock_offset
        if active_start is not None and zero_start is not None and active_start <= host_mono <= zero_start:
            phase = "active_motion"
        elif zero_start is not None and stand_confirmed is not None and zero_start < host_mono <= stand_confirmed:
            phase = "zero_to_stand_transition"
        elif stand_confirmed is not None and physical_window_start is not None and stand_confirmed < host_mono < physical_window_start:
            phase = "stand_settling_before_physical_window"
        elif physical_window_start is not None and physical_window_end is not None and physical_window_start <= host_mono <= physical_window_end:
            phase = "final_physical_window"
        else:
            phase = "outside_motion_and_stop_windows"
        events.append({
            "wall_utc": wall_time.isoformat(),
            "host_monotonic_s": host_mono,
            "worker": match.group("worker"),
            "diff_us": int(match.group("diff")),
            "period_us": int(match.group("period")),
            "miss_percent": float(match.group("miss")),
            "phase": phase,
        })
    by_phase: dict[str, dict[str, Any]] = {}
    for phase in sorted({event["phase"] for event in events}):
        selected = [event for event in events if event["phase"] == phase]
        misses = [event["miss_percent"] for event in selected]
        by_phase[phase] = {
            "count": len(selected),
            "first_wall_utc": selected[0]["wall_utc"],
            "last_wall_utc": selected[-1]["wall_utc"],
            "median_miss_percent": statistics.median(misses),
            "p95_miss_percent": sorted(misses)[min(len(misses) - 1, int(0.95 * len(misses)))],
            "max_miss_percent": max(misses),
            "largest_three_events": sorted(selected, key=lambda event: event["miss_percent"], reverse=True)[:3],
        }
    by_worker: dict[str, dict[str, Any]] = {}
    for worker in sorted({event["worker"] for event in events}):
        selected = [event for event in events if event["worker"] == worker]
        misses = [event["miss_percent"] for event in selected]
        by_worker[worker] = {
            "count": len(selected),
            "median_miss_percent": statistics.median(misses),
            "p95_miss_percent": sorted(misses)[min(len(misses) - 1, int(0.95 * len(misses)))],
            "max_miss_percent": max(misses),
        }
    return {
        "status": "CAPTURED",
        "log_path": str(log_path),
        "host_clock_offset_s_median": clock_offset,
        "missed_wakeup_event_count": len(events),
        "missed_wakeups_by_phase": by_phase,
        "by_worker": by_worker,
        "interpretation": "missed-wakeup warnings are timing evidence; counts alone do not prove they caused the yaw error",
    }


def _counterfactual_pose_replay(
    pg: Mapping[str, Any],
    state_rows: list[dict[str, Any]],
    state_times: list[float],
    command_trace: list[dict[str, Any]],
    *,
    zero_start: float | None,
    stand_request_time: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    start_pose = pg.get("start_pose", {})
    target_pose = pg.get("target_pose", {})
    target_yaw = _number(target_pose.get("yaw"))
    stop_pose = dict(pg.get("dynamic_stop_target_pose_at_trigger") or target_pose)
    if target_yaw is not None:
        stop_pose["yaw"] = target_yaw
    if not isinstance(start_pose, Mapping) or not isinstance(stop_pose, Mapping) or target_yaw is None:
        return {"status": "NOT_RUN_MISSING_POSE", "classification": "OFFLINE_COMMAND_REPLAY_NOT_DYNAMICS"}, [], []

    profile = pg.get("command_profile_parameters", {})
    if not isinstance(profile, Mapping):
        profile = {}
    heading_policy = str(profile.get("heading_policy", ""))
    prediction_horizon = _number(profile.get("heading_yaw_rate_prediction_horizon_s"))
    if prediction_horizon is not None:
        legacy_policy_kind = "PREDICTED_TARGET_YAW"
    elif "target-yaw" in heading_policy.lower():
        legacy_policy_kind = "DIRECT_TARGET_YAW"
    else:
        legacy_policy_kind = "RUN73_POSITION_BEARING"

    active_rows = []
    controller_rows = []
    legacy_replay_commands = []
    coordinated_replay_commands = []
    stale_or_missing = 0
    legacy_feedback_missing = 0
    legacy_command_mismatches = []
    for command in command_trace:
        timestamp = _number(command.get("sample_monotonic_s"))
        if timestamp is None:
            continue
        idx = bisect.bisect_right(state_times, timestamp) - 1
        if idx < 0:
            stale_or_missing += 1
            continue
        state_row = state_rows[idx]
        snapshot = state_row.get("snapshot", {})
        pose = snapshot.get("pose", {}) if isinstance(snapshot, Mapping) else {}
        if not isinstance(pose, Mapping):
            stale_or_missing += 1
            continue
        progress = _number(command.get("along_track_progress_m")) or 0.0
        stop_progress = _number(pg.get("dynamic_stop_progress_at_trigger_m")) or 0.335
        fraction = max(0.0, min(1.0, progress / max(stop_progress, 1e-6)))
        carrot_fraction = min(1.0, fraction + POSE_GOAL_LOOKAHEAD_FRACTION)
        reference = pose_path_reference(start_pose, stop_pose, fraction)
        carrot = pose_path_reference(start_pose, stop_pose, carrot_fraction)
        warmup = float(command.get("activation_elapsed_s") or 0.0) < 0.5
        forward = 0.12 if warmup else 0.14
        legacy_rate = _number(command.get("yaw_rate_feedback_rad_s"))
        legacy_horizon = _number(command.get("yaw_prediction_horizon_s")) or prediction_horizon
        if warmup:
            legacy_error = _number(command.get("instantaneous_heading_error_rad"))
            legacy_angular = 0.0
        else:
            if legacy_policy_kind == "RUN73_POSITION_BEARING":
                dx = float(target_pose["x"]) - float(pose["x"])
                dy = float(target_pose["y"]) - float(pose["y"])
                bearing = math.atan2(dy, dx)
                legacy_error = math.atan2(
                    math.sin(bearing - float(pose["yaw"])),
                    math.cos(bearing - float(pose["yaw"])),
                )
            elif legacy_policy_kind == "DIRECT_TARGET_YAW":
                legacy_error = math.atan2(
                    math.sin(target_yaw - float(pose["yaw"])),
                    math.cos(target_yaw - float(pose["yaw"])),
                )
            else:
                if legacy_rate is None and legacy_horizon is not None:
                    legacy_rate, rate_coverage = _yaw_rate_at(state_rows, timestamp)
                    if rate_coverage < 0.38:
                        legacy_rate = None
                if legacy_rate is None or legacy_horizon is None:
                    legacy_error = None
                    legacy_angular = None
                    legacy_feedback_missing += 1
                else:
                    legacy_error = math.atan2(
                        math.sin(target_yaw - (float(pose["yaw"]) + legacy_rate * legacy_horizon)),
                        math.cos(target_yaw - (float(pose["yaw"]) + legacy_rate * legacy_horizon)),
                    )
            if legacy_policy_kind != "PREDICTED_TARGET_YAW" or legacy_rate is not None:
                gain = _number(profile.get("heading_gain")) or 0.8
                angular_limit = _number(profile.get("maximum_angular_velocity_rad_s")) or 0.075
                legacy_angular = max(-angular_limit, min(angular_limit, gain * float(legacy_error)))
                if legacy_policy_kind == "RUN73_POSITION_BEARING":
                    taper = _number(command.get("angular_taper_factor"))
                    if taper is None:
                        taper = 1.0
                    legacy_angular *= taper
        if warmup:
            angular = 0.0
        else:
            try:
                angular = pure_pursuit_yaw_rate(
                    current_x_m=float(pose["x"]),
                    current_y_m=float(pose["y"]),
                    current_yaw_rad=float(pose["yaw"]),
                    carrot_x_m=carrot["x_m"],
                    carrot_y_m=carrot["y_m"],
                    forward_command_m_s=forward,
                    maximum_angular_rad_s=0.075,
                )["command_yaw_rate_rad_s"]
            except (KeyError, TypeError, ValueError):
                stale_or_missing += 1
                continue
        actual = _number(command.get("angular_velocity_rad_s")) or 0.0
        if legacy_angular is not None:
            legacy_command_mismatches.append(abs(legacy_angular - actual))
        controller_rows.append({
            "run": pg.get("attempt_id"),
            "sample_monotonic_s": timestamp,
            "phase": "APPROACH_SAVED_POSE_POLICY_REPLAY",
            "legacy_policy_replayed_angular_command_rad_s": legacy_angular,
            "legacy_actual_angular_command_rad_s": actual,
            "legacy_policy_kind": legacy_policy_kind,
            "legacy_controller_error_rad": legacy_error,
            "legacy_measured_yaw_rate_rad_s": legacy_rate,
            "legacy_prediction_horizon_s": legacy_horizon,
            "coordinated_curve_angular_command_rad_s": angular,
            "coordinated_curve_forward_command_m_s": forward,
            "target_yaw_error_rad": math.atan2(
                math.sin(target_yaw - float(pose["yaw"])),
                math.cos(target_yaw - float(pose["yaw"])),
            ),
            "actual_position_bearing_error_rad": command.get("position_bearing_error_rad"),
            "coordinated_path_tangent_yaw_rad": reference["tangent_yaw_rad"],
            "coordinated_path_curvature_per_m": reference["curvature_per_m"],
            "path_fraction": fraction,
            "lookahead_fraction": carrot_fraction,
        })
        if legacy_angular is not None:
            active_rows.append((timestamp, float(legacy_angular), float(angular), actual))
            legacy_replay_commands.append(float(legacy_angular))
        else:
            active_rows.append((timestamp, float("nan"), float(angular), actual))
        coordinated_replay_commands.append(float(angular))

    def integrate(rows: list[tuple[float, ...]], index: int) -> float | None:
        valid = [row for row in rows if math.isfinite(row[index])]
        if len(valid) < 2:
            return None
        return sum(
            (valid[i][index] + valid[i + 1][index]) * 0.5 * (valid[i + 1][0] - valid[i][0])
            for i in range(len(valid) - 1)
        )

    terminal_rows = []
    yaw_rate_missing = 0
    zero_to_stand_state_count = 0
    stop_states = []
    if zero_start is not None and stand_request_time is not None:
        for row in state_rows:
            snapshot = row.get("snapshot", {})
            timestamp = _number(snapshot.get("sample_monotonic_s")) if isinstance(snapshot, Mapping) else None
            if timestamp is not None and zero_start <= timestamp <= stand_request_time:
                zero_to_stand_state_count += 1
                stop_states.append(row)
    for row in stop_states:
        snapshot = row.get("snapshot", {})
        timestamp = _number(snapshot.get("sample_monotonic_s")) if isinstance(snapshot, Mapping) else None
        if timestamp is None or zero_start is None or stand_request_time is None:
            continue
        if not zero_start <= timestamp <= stand_request_time:
            continue
        pose = snapshot.get("pose", {})
        if not isinstance(pose, Mapping):
            continue
        yaw_rate, coverage = _yaw_rate_at(state_rows, timestamp)
        try:
            feedback = fresh_yaw_rate_feedback(
                {"signed_net_yaw_rate_rad_s": yaw_rate, "sample_monotonic_s": timestamp},
                sample_monotonic_s=timestamp,
                maximum_age_s=0.15,
            ) if yaw_rate is not None and coverage >= 0.38 else None
        except ValueError:
            feedback = None
        if feedback is None:
            yaw_rate_missing += 1
            continue
        terminal = terminal_yaw_rate_command(
            current_yaw_rad=float(pose["yaw"]),
            target_yaw_rad=target_yaw,
            measured_yaw_rate_rad_s=feedback["signed_yaw_rate_rad_s"],
            maximum_angular_rad_s=0.075,
        )
        terminal_rows.append({
            "run": pg.get("attempt_id"),
            "sample_monotonic_s": timestamp,
            "phase": "TERMINAL_YAW_COUNTERFACTUAL_FIXED_POSE",
            "actual_forward_velocity_command_m_s": 0.0,
            "actual_angular_velocity_command_rad_s": 0.0,
            "counterfactual_angular_velocity_command_rad_s": terminal["command_yaw_rate_rad_s"],
            "yaw_error_rad": terminal["yaw_error_rad"],
            "measured_world_yaw_rate_fit_rad_s": feedback["signed_yaw_rate_rad_s"],
            "measured_body_wz_rad_s": pose.get("wz"),
            "mode_desc": snapshot.get("get_mode_desc"),
            "fsm_state_raw": snapshot.get("fsm_state"),
            "rate_window_coverage_s": coverage,
        })
    terminal_integral = 0.0
    for left, right in zip(terminal_rows, terminal_rows[1:]):
        dt = float(right["sample_monotonic_s"]) - float(left["sample_monotonic_s"])
        terminal_integral += (
            float(left["counterfactual_angular_velocity_command_rad_s"])
            + float(right["counterfactual_angular_velocity_command_rad_s"])
        ) * 0.5 * dt
    summary = {
        "status": "REPLAYED" if controller_rows else "NOT_RUN_NO_ACTIVATION_COMMANDS",
        "classification": "COUNTERFACTUAL_SAVED_POSE_COMMAND_REPLAY_NOT_DYNAMICS_VALIDATION",
        "legacy_policy_kind": legacy_policy_kind,
        "approach_sample_count": len(controller_rows),
        "approach_samples_missing_state_or_path": stale_or_missing,
        "legacy_missing_rate_or_horizon_samples": legacy_feedback_missing,
        "legacy_replay_max_abs_command_error_rad_s": max(legacy_command_mismatches) if legacy_command_mismatches else None,
        "legacy_policy_replay_command_integral_rad": integrate(active_rows, 1) if active_rows else None,
        "legacy_actual_command_integral_rad": integrate(active_rows, 3) if active_rows else None,
        "coordinated_curve_command_integral_rad": integrate(active_rows, 2) if active_rows else None,
        "legacy_replay_positive_negative_command_count": [
            sum(v > 1e-9 for v in legacy_replay_commands), sum(v < -1e-9 for v in legacy_replay_commands),
        ],
        "coordinated_positive_negative_command_count": [
            sum(v > 1e-9 for v in coordinated_replay_commands), sum(v < -1e-9 for v in coordinated_replay_commands),
        ],
        "zero_to_stand_saved_state_sample_count": zero_to_stand_state_count,
        "terminal_replay_status": (
            "REPLAYED_SAVED_STATES" if terminal_rows
            else "NOT_REPLAYED_NO_STATE_SAMPLES_IN_ZERO_TO_STAND_INTERVAL"
            if zero_start is not None and stand_request_time is not None and zero_to_stand_state_count == 0
            else "NOT_REPLAYED_MISSING_ZERO_OR_STAND_TIMESTAMPS"
            if zero_start is None or stand_request_time is None
            else "NOT_REPLAYED_YAW_RATE_FIT_UNAVAILABLE"
        ),
        "terminal_zero_to_stand_saved_samples": len(terminal_rows),
        "terminal_yaw_rate_fit_unavailable_samples": yaw_rate_missing,
        "terminal_counterfactual_command_integral_rad": terminal_integral if terminal_rows else None,
        "limitations": [
            "historical poses are held fixed while the replayed commands change",
            "the replay cannot predict path, gait response, terminal yaw, or stopping outcome",
            "the legacy approach command is recomputed from each saved pose, measured-rate estimate, and run-specific prediction horizon, then checked against the command actually recorded",
            "the new coordinated approach command is evaluated on the same saved poses; no state feedback is generated by either replay",
            "terminal replay requires saved state samples between first zero and STAND request; absent samples are reported as missing evidence, not zero controller response",
        ],
    }
    return summary, controller_rows, terminal_rows


def _run_summary(review_root: Path, label: str, folder: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = review_root / folder
    result = _load_json(run_dir / "m1_first_motion_result.json")
    parent_path = run_dir / "auto-reset-run-result.json"
    parent = _load_json(parent_path) if parent_path.is_file() else {}
    pg = dict(result.get("point_goal_diagnostic", {}))
    pg["analysis_run_label"] = label
    pg["attempt_id"] = result.get("attempt_id")
    activation = result.get("activation", {})
    gate = result.get("physical_stop_gate", {})
    post_odom = activation.get("post_stop_odometry", {})
    trace = _load_jsonl(run_dir / "m1_first_motion_trace.jsonl")
    state_rows = [
        row for row in trace
        if row.get("event") == "state_sample"
        and isinstance(row.get("snapshot"), Mapping)
        and _number(row["snapshot"].get("sample_monotonic_s")) is not None
    ]
    state_rows.sort(key=lambda r: float(r["snapshot"]["sample_monotonic_s"]))
    state_times = [float(r["snapshot"]["sample_monotonic_s"]) for r in state_rows]
    command_rows = [row for row in trace if row.get("event") == "velocity_command"]
    stop = activation.get("zero_stop", {})
    zero_start = _number(stop.get("start_monotonic_s"))
    stand_transition = activation.get("stop_transition_confirmation", {})
    stand_confirmed = _number(stand_transition.get("confirmed_monotonic_s"))
    stand_request = stand_transition.get("request_result", {}) if isinstance(stand_transition, Mapping) else {}
    stand_request_time = _number(stand_transition.get("started_monotonic_s")) if isinstance(stand_transition, Mapping) else None
    stand_request_response = stand_request.get("response", {}) if isinstance(stand_request, Mapping) else {}
    stand_post_request_readback = stand_request.get("mode_readback_after_request", {}) if isinstance(stand_request, Mapping) else {}
    gate_window = gate.get("final_window", {}) if isinstance(gate, Mapping) else {}
    gate_start = _number(gate_window.get("window_start_monotonic_s"))
    gate_end = _number(gate_window.get("window_end_monotonic_s"))

    active_states = [r for r in state_rows if r.get("phase") == "point-goal-activation"]
    post_states = [r for r in state_rows if r.get("phase") == "point-goal-post-stop"]
    start_pose = pg.get("start_pose", {})
    zero_state = _asof_state(state_rows, state_times, zero_start) if zero_start is not None else None
    stand_state = _asof_state(state_rows, state_times, stand_confirmed) if stand_confirmed is not None else None
    gate_start_state = _asof_state(state_rows, state_times, gate_start) if gate_start is not None else None
    gate_end_state = _asof_state(state_rows, state_times, gate_end) if gate_end is not None else None

    def pose_yaw(row: dict[str, Any] | None) -> float | None:
        if row is None:
            return None
        return _number(row.get("snapshot", {}).get("pose", {}).get("yaw"))

    first_active_command = next((r for r in command_rows if r.get("phase") == "point-goal-activation"), None)
    first_command_time = _number(first_active_command.get("wall_monotonic_sec")) if first_active_command else None
    last_active_state = active_states[-1] if active_states else None
    final_state = post_states[-1] if post_states else (state_rows[-1] if state_rows else None)
    final_pose = pg.get("activation", {}).get("post_stop_odometry", {}).get("last_pose", {})
    yaw_start = _number(start_pose.get("yaw"))
    target_yaw = _number(pg.get("target_pose", {}).get("yaw"))
    yaw_zero = pose_yaw(zero_state)
    yaw_stand = pose_yaw(stand_state)
    yaw_final = _number(final_pose.get("yaw"))
    zero_pose = zero_state.get("snapshot", {}).get("pose", {}) if isinstance(zero_state, Mapping) else {}
    zero_distance = None
    if isinstance(zero_pose, Mapping) and target_yaw is not None:
        target_pose = pg.get("target_pose", {})
        try:
            zero_distance = math.hypot(
                float(target_pose["x"]) - float(zero_pose["x"]),
                float(target_pose["y"]) - float(zero_pose["y"]),
            )
        except (KeyError, TypeError, ValueError):
            zero_distance = None

    odom_samples: list[tuple[float, float]] = []
    source_stamp_samples: list[tuple[float, float, str | None]] = []
    source_stamp_rows = 0
    get_sources: list[str] = []
    common_sources: list[str] = []
    get_modes: list[str] = []
    fsm_states: list[int] = []
    state_ages: list[float] = []
    phase_gaps: dict[str, list[float]] = {}
    mode_fsm_timeline: list[dict[str, Any]] = []
    previous_mode_tuple: tuple[Any, ...] | None = None
    for row in state_rows:
        snapshot = row["snapshot"]
        pose = snapshot.get("pose", {})
        t = _number(snapshot.get("sample_monotonic_s"))
        yaw = _number(pose.get("yaw")) if isinstance(pose, Mapping) else None
        if t is not None and yaw is not None:
            odom_samples.append((t, yaw))
        if isinstance(pose, Mapping) and pose.get("stamp_sec") is not None and pose.get("stamp_nanosec") is not None:
            stamp = float(pose["stamp_sec"]) + float(pose["stamp_nanosec"]) * 1e-9
            if t is not None:
                source_stamp_samples.append((t, stamp, str(row.get("phase")) if row.get("phase") is not None else None))
            source_stamp_rows += 1
        if snapshot.get("get_source_name"):
            get_sources.append(str(snapshot["get_source_name"]))
        if snapshot.get("common_source_name"):
            common_sources.append(str(snapshot["common_source_name"]))
        if snapshot.get("get_mode_desc"):
            get_modes.append(str(snapshot["get_mode_desc"]))
        fsm = snapshot.get("fsm_state")
        if isinstance(fsm, int):
            fsm_states.append(fsm)
        mode_tuple = (
            snapshot.get("get_mode_desc"), snapshot.get("common_mode_desc"),
            snapshot.get("fsm_state"), snapshot.get("body_status"),
            snapshot.get("get_source_name"), snapshot.get("common_source_name"),
        )
        if mode_tuple != previous_mode_tuple:
            mode_fsm_timeline.append({
                "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
                "phase": row.get("phase"),
                "get_mode_desc": snapshot.get("get_mode_desc"),
                "common_mode_desc": snapshot.get("common_mode_desc"),
                "fsm_state_raw": snapshot.get("fsm_state"),
                "body_status_raw": snapshot.get("body_status"),
                "get_source_name": snapshot.get("get_source_name"),
                "common_source_name": snapshot.get("common_source_name"),
            })
            previous_mode_tuple = mode_tuple
        age = snapshot.get("state_age_s", {}).get("odom")
        if _number(age) is not None:
            state_ages.append(float(age))
        phase = str(row.get("phase"))
        if t is not None:
            phase_gaps.setdefault(phase, []).append(t)
    source_deltas = [b[1] - a[1] for a, b in zip(source_stamp_samples, source_stamp_samples[1:])]
    odom_regressions = sum(delta < -1e-9 for delta in source_deltas)
    odom_duplicates = sum(abs(delta) <= 1e-9 for delta in source_deltas)
    odom_regression_first = next((i for i, delta in enumerate(source_deltas) if delta < -1e-9), None)
    odom_duplicate_first = next((i for i, delta in enumerate(source_deltas) if abs(delta) <= 1e-9), None)
    stamp_anomalies = []
    for index, delta in enumerate(source_deltas):
        if delta < -1e-9 or abs(delta) <= 1e-9:
            previous, current = source_stamp_samples[index], source_stamp_samples[index + 1]
            stamp_anomalies.append({
                "kind": "regression" if delta < -1e-9 else "duplicate",
                "previous_source_stamp_s": previous[1],
                "current_source_stamp_s": current[1],
                "delta_s": delta,
                "current_host_monotonic_s": current[0],
                "phase": current[2],
            })

    def segment_slope(start: float | None, end: float | None) -> float | None:
        if start is None or end is None or end <= start:
            return None
        raw = [(t, yaw) for t, yaw in odom_samples if start <= t <= end]
        if len(raw) < 2:
            return None
        unwrapped = [raw[0]]
        yaw_value = raw[0][1]
        for t, yaw in raw[1:]:
            delta = wrapped_delta(yaw, unwrapped[-1][1])
            if delta is None:
                continue
            yaw_value += delta
            unwrapped.append((t, yaw_value))
        return _linear_slope(unwrapped)

    state_for_timeline = {
        "first_active_command": _asof_state(state_rows, state_times, first_command_time) if first_command_time is not None else None,
        "last_active_sample": last_active_state,
        "at_first_zero": zero_state,
        "at_stand_confirmed": stand_state,
        "at_physical_window_start": gate_start_state,
        "at_physical_window_end": gate_end_state,
        "final_state_sample": final_state,
    }
    timeline = []
    for label_key, row in state_for_timeline.items():
        if row is None:
            continue
        snapshot = row["snapshot"]
        pose = snapshot.get("pose", {})
        timeline.append({
            "milestone": label_key,
            "phase": row.get("phase"),
            "host_monotonic_s": snapshot.get("sample_monotonic_s"),
            "wall_utc": snapshot.get("wall_utc"),
            "odom_stamp_s": (
                float(pose["stamp_sec"]) + float(pose["stamp_nanosec"]) * 1e-9
                if isinstance(pose, Mapping) and pose.get("stamp_sec") is not None else None
            ),
            "x_m": pose.get("x"), "y_m": pose.get("y"), "yaw_world_rad": pose.get("yaw"),
            "vx_body_m_s": pose.get("vx"), "vy_body_m_s": pose.get("vy"), "wz_body_rad_s": pose.get("wz"),
            "get_mode": snapshot.get("get_mode_desc"),
            "common_mode": snapshot.get("common_mode_desc"),
            "input_source_get": snapshot.get("get_source_name"),
            "input_source_common": snapshot.get("common_source_name"),
            "fsm_state_raw": snapshot.get("fsm_state"),
            "body_status_raw": snapshot.get("body_status"),
        })

    zero_end = _number(stop.get("end_monotonic_s"))
    post_first = activation.get("post_stop_first_observation", {})
    post_first_time = _number(post_first.get("sample_monotonic_s"))
    last_command_time = _number(command_rows[-1].get("wall_monotonic_sec")) if command_rows else None
    source_selected_active = sum(
        1 for row in active_states
        if row["snapshot"].get("get_source_name") == result.get("source")
        and row["snapshot"].get("common_source_name") == result.get("source")
    )
    mode_active = sum(
        1 for row in active_states
        if row["snapshot"].get("get_mode_desc") == "LOCOMOTION_DEFAULT"
        and row["snapshot"].get("common_mode_desc") == "LOCOMOTION_DEFAULT"
    )
    sample_gaps = {
        phase: max((b - a for a, b in zip(times, times[1:])), default=None)
        for phase, times in phase_gaps.items()
    }
    summary = {
        "run": label,
        "evidence_dir": str(run_dir),
        "attempt_id": result.get("attempt_id"),
        "run_status": pg.get("status"),
        "requested_heading_change_rad": pg.get("requested_heading_change_rad"),
        "target_yaw_rad": pg.get("target_pose", {}).get("yaw"),
        "final_yaw_rad": pg.get("final_yaw_rad"),
        "measured_yaw_change_rad": pg.get("final_heading_change_rad"),
        "final_yaw_error_rad": pg.get("final_yaw_error_rad"),
        "final_position_error_m": pg.get("final_distance_to_goal_m"),
        "final_lateral_error_m": pg.get("final_lateral_error_m"),
        "stop_reason": pg.get("stop_reason"),
        "physical_stop_status": gate.get("status"),
        "physical_stop_violations": gate.get("final_window_violations", []),
        "reset_status": parent.get("reset_classification"),
        "control_handshake_status": activation.get("control_handshake_status"),
        "input_source_readback_active_fraction": source_selected_active / len(active_states) if active_states else None,
        "locomotion_readback_active_fraction": mode_active / len(active_states) if active_states else None,
        "velocity_publish": _event_stats(trace),
        "mc_joint_output": _mc_output_stats(trace),
        "mc_rl_debug": _rl_debug_stats(run_dir / "mc_rl_debug_trace.jsonl"),
        "mc_debug_f64": {
            "message_count": sum(1 for row in trace if row.get("event") == "mc_debug_f64"),
            "nonempty_data_count": sum(
                1 for row in trace if row.get("event") == "mc_debug_f64"
                and row.get("payload", {}).get("data")
            ),
            "interpretation": "raw installed Float64MultiArray topic; empty payloads carry no command semantics",
        },
        "odometry_timestamps": {
            "sample_count": source_stamp_rows,
            "stamp_regression_count": odom_regressions,
            "stamp_duplicate_count": odom_duplicates,
            "first_regression_left_index": odom_regression_first,
            "first_duplicate_left_index": odom_duplicate_first,
            "anomalies": stamp_anomalies,
            "control_receive_clock": "host_monotonic; source-stamp anomalies are reported, not used for yaw slope",
            "maximum_odom_age_s": max(state_ages) if state_ages else None,
            "maximum_receive_gap_by_phase_s": sample_gaps,
        },
        "timeline": timeline,
        "event_times": {
            "first_command_monotonic_s": first_command_time,
            "last_command_monotonic_s": last_command_time,
            "first_zero_stop_monotonic_s": zero_start,
            "zero_stream_end_monotonic_s": zero_end,
            "stand_request_started_monotonic_s": stand_request_time,
            "stand_confirmed_monotonic_s": stand_confirmed,
            "post_stop_first_observation_monotonic_s": post_first_time,
            "physical_window_start_monotonic_s": gate_start,
            "physical_window_end_monotonic_s": gate_end,
        },
        "mode_fsm_source_transitions_raw": mode_fsm_timeline,
        "action_requests": {
            "set_locomotion_default": activation.get("set_mc_action"),
            "set_stand_default": {
                "started_monotonic_s": stand_request_time,
                "wall_utc_submitted": stand_request.get("wall_utc_submitted") if isinstance(stand_request, Mapping) else None,
                "wall_utc_returned": stand_request.get("wall_utc_returned") if isinstance(stand_request, Mapping) else None,
                "request": stand_request.get("request") if isinstance(stand_request, Mapping) else None,
                "response_header_code": stand_request.get("response_header_code") if isinstance(stand_request, Mapping) else None,
                "response_status_value": stand_request.get("response_status_value") if isinstance(stand_request, Mapping) else None,
                "response_message": stand_request.get("response_message") if isinstance(stand_request, Mapping) else None,
                "mode_readback_immediately_after_request": stand_post_request_readback,
                "confirmed_stand_monotonic_s": stand_confirmed,
            },
        },
        "yaw_segments_rad": {
            "start_to_zero": wrapped_delta(yaw_zero, yaw_start),
            "zero_to_stand_confirmed": wrapped_delta(yaw_stand, yaw_zero),
            "stand_confirmed_to_final": wrapped_delta(yaw_final, yaw_stand),
            "last_in_pulse_to_final": wrapped_delta(yaw_final, _number(post_odom.get("last_in_pulse_pose", {}).get("yaw"))),
        },
        "yaw_rate_fit_rad_s": {
            "activation_start_to_zero": segment_slope(first_command_time, zero_start),
            "last_0p4s_before_zero": segment_slope(
                max(first_command_time or 0.0, (zero_start or 0.0) - 0.4), zero_start
            ),
            "zero_to_stand_confirmed": segment_slope(zero_start, stand_confirmed),
            "stand_confirmed_to_final_window_start": segment_slope(stand_confirmed, gate_start),
            "final_physical_window": gate_window.get("yaw_rate_fit_rad_s"),
        },
        "pose_at_first_zero": {
            "world_yaw_rad": yaw_zero,
            "target_yaw_error_rad": wrapped_delta(target_yaw, yaw_zero),
            "position_error_m": zero_distance,
            "body_wz_rad_s": zero_pose.get("wz") if isinstance(zero_pose, Mapping) else None,
            "twist_frame_id": zero_pose.get("twist_frame_id") if isinstance(zero_pose, Mapping) else None,
        },
        "mc_timing": _mc_timing_stats(
            run_dir,
            state_rows,
            active_start=first_command_time,
            zero_start=zero_start,
            stand_confirmed=stand_confirmed,
            physical_window_start=gate_start,
            physical_window_end=gate_end,
        ),
        "publisher_ack_limit": "No per-message MC receive ACK exists in this runtime path; CALL_RETURNED, selected source, mode readbacks, and changing MC leg targets are separate evidence types.",
    }

    command_aligned: list[dict[str, Any]] = []
    publish_returns = [row for row in trace if row.get("event") == "velocity_publish_result"]
    actual_nonzero = [
        (index, command) for index, command in enumerate(command_rows)
        if any(abs(_number(command.get(key)) or 0.0) > 1e-9 for key in (
            "forward_velocity_m_s", "lateral_velocity_m_s", "angular_velocity_rad_s",
        ))
    ]
    controller_trace = pg.get("command_trace", [])
    nonzero_controller_trace = [
        row for row in controller_trace if any(abs(_number(row.get(key)) or 0.0) > 1e-9 for key in (
            "forward_velocity_m_s", "lateral_velocity_m_s", "angular_velocity_rad_s",
        ))
    ]
    for nonzero_index, (original_index, command) in enumerate(actual_nonzero):
        timestamp = _number(command.get("wall_monotonic_sec"))
        if timestamp is None:
            continue
        asof = _asof_state(state_rows, state_times, timestamp)
        snapshot = asof.get("snapshot", {}) if isinstance(asof, Mapping) else {}
        pose = snapshot.get("pose", {}) if isinstance(snapshot, Mapping) else {}
        controller = nonzero_controller_trace[nonzero_index] if nonzero_index < len(nonzero_controller_trace) else {}
        returned = publish_returns[original_index] if original_index < len(publish_returns) else {}
        yaw_rate_fit, yaw_rate_coverage = _yaw_rate_at(state_rows, timestamp)
        odom_pose = snapshot.get("pose", {}) if isinstance(snapshot, Mapping) else {}
        point_sample = snapshot.get("point_goal", {}) if isinstance(snapshot, Mapping) else {}
        command_aligned.append({
            "run": label,
            "command_index": original_index,
            "command_host_monotonic_s": timestamp,
            "command_wall_utc": command.get("wall_utc"),
            "command_phase": command.get("phase"),
            "command_kind": command.get("command_kind"),
            "forward_velocity_command_m_s": command.get("forward_velocity_m_s"),
            "lateral_velocity_command_m_s": command.get("lateral_velocity_m_s"),
            "angular_velocity_command_rad_s": command.get("angular_velocity_rad_s"),
            "publish_call_outcome": command.get("publish_outcome"),
            "publish_return_outcome": returned.get("publish_outcome"),
            "publish_return_monotonic_s": returned.get("wall_monotonic_sec"),
            "publish_call_duration_s": (
                _number(returned.get("wall_monotonic_sec")) - timestamp
                if _number(returned.get("wall_monotonic_sec")) is not None else None
            ),
            "asof_state_host_monotonic_s": snapshot.get("sample_monotonic_s") if isinstance(snapshot, Mapping) else None,
            "asof_state_age_s": timestamp - _number(snapshot.get("sample_monotonic_s")) if isinstance(snapshot, Mapping) and _number(snapshot.get("sample_monotonic_s")) is not None else None,
            "measured_world_yaw_rad": pose.get("yaw") if isinstance(pose, Mapping) else None,
            "measured_world_yaw_rate_fit_0p4s_rad_s": yaw_rate_fit,
            "yaw_rate_fit_coverage_s": yaw_rate_coverage,
            "measured_body_wz_rad_s": odom_pose.get("wz") if isinstance(odom_pose, Mapping) else None,
            "twist_frame_id": odom_pose.get("twist_frame_id") if isinstance(odom_pose, Mapping) else None,
            "controller_phase": controller.get("control_phase"),
            "controller_heading_error_rad": controller.get("heading_error_rad"),
            "controller_instantaneous_yaw_error_rad": controller.get("instantaneous_heading_error_rad"),
            "controller_position_bearing_error_rad": controller.get("position_bearing_error_rad"),
            "controller_yaw_rate_feedback_rad_s": controller.get("yaw_rate_feedback_rad_s"),
            "controller_yaw_rate_feedback_age_s": controller.get("yaw_rate_feedback_age_s"),
            "controller_predicted_yaw_error_rad": controller.get("predicted_yaw_error_rad"),
            "controller_yaw_prediction_horizon_s": controller.get("yaw_prediction_horizon_s"),
            "controller_target_yaw_rad": controller.get("target_yaw_rad"),
            "controller_target_path_yaw_rad": controller.get("target_path_yaw_rad"),
            "controller_path_progress_fraction": controller.get("path_progress_fraction"),
            "controller_terminal_phase_elapsed_s": controller.get("terminal_phase_elapsed_s"),
            "controller_pose_goal_phase": controller.get("pose_goal_phase"),
            "snapshot_point_goal_phase": point_sample.get("control_phase") if isinstance(point_sample, Mapping) else None,
            "snapshot_point_goal_heading_error_rad": point_sample.get("heading_error_rad") if isinstance(point_sample, Mapping) else None,
            "get_mode_desc": snapshot.get("get_mode_desc") if isinstance(snapshot, Mapping) else None,
            "common_mode_desc": snapshot.get("common_mode_desc") if isinstance(snapshot, Mapping) else None,
            "fsm_state_raw": snapshot.get("fsm_state") if isinstance(snapshot, Mapping) else None,
            "input_source_get": snapshot.get("get_source_name") if isinstance(snapshot, Mapping) else None,
            "input_source_common": snapshot.get("common_source_name") if isinstance(snapshot, Mapping) else None,
            "leg_target_header_stamp_s": (
                float(snapshot["mc_leg_command"]["header_stamp"]["sec"])
                + float(snapshot["mc_leg_command"]["header_stamp"]["nanosec"]) * 1e-9
                if isinstance(snapshot, Mapping) and isinstance(snapshot.get("mc_leg_command"), Mapping)
                and snapshot["mc_leg_command"].get("header_stamp") else None
            ),
        })
    replay_summary, replay_approach_rows, replay_terminal_rows = _counterfactual_pose_replay(
        pg,
        state_rows,
        state_times,
        pg.get("command_trace", []),
        zero_start=zero_start,
        stand_request_time=stand_request_time,
    )
    summary["counterfactual_pose_controller_replay"] = replay_summary
    return summary, command_aligned, replay_approach_rows + replay_terminal_rows


def _validate_reserved_output(path: Path) -> None:
    helper = Path(__file__).resolve().parents[1] / "agent"
    sys.path.insert(0, str(helper))
    from path_governance import validate_output_path

    location = validate_output_path(path, require_exists=True, expected_kind="directory")
    marker = location.native_path / ".robotsim-output-reservation.json"
    reservation = _load_json(marker)
    if reservation != {"schema_version": "robotsim.output-reservation.v1", "path": location.wsl_path}:
        raise ValueError("output directory reservation marker does not match its canonical path")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    review_root = args.review_root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    _validate_reserved_output(output_dir)

    summaries = []
    aligned = []
    replayed = []
    for label, folder in DEFAULT_RUNS.items():
        summary, rows, replay_rows = _run_summary(review_root, label, folder)
        summaries.append(summary)
        aligned.extend(rows)
        replayed.extend(replay_rows)

    result = {
        "schema": "robotsim.issue80.pose-goal-offline-analysis.v1",
        "classification": "OFFLINE_TRACE_ANALYSIS_NOT_DYNAMICS_VALIDATION",
        "analysis_scope": ["run86", "run88", "run89", "run90", "run91"],
        "runs": summaries,
        "limitations": [
            "Velocity publish callback return is not an MC per-message receipt acknowledgement.",
            "Common-state input-source selection is arbitration evidence, not proof each velocity message was consumed.",
            "McRLDebug arrays are compared only as raw vectors; no field-index semantics are inferred.",
            "Body-frame twist wz and the derivative of world-frame odometry yaw are reported separately.",
            "Offline replay and timestamp-aligned traces do not predict full MuJoCo + MC dynamics.",
        ],
    }
    (output_dir / "POSE_GOAL_TRACE_ANALYSIS.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "COMMAND_RESPONSE_TIMELINE.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(aligned[0]) if aligned else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aligned)
    if replayed:
        replay_fields = sorted({key for row in replayed for key in row})
        with (output_dir / "COORDINATED_CONTROLLER_REPLAY.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=replay_fields)
            writer.writeheader()
            writer.writerows(replayed)
    print(json.dumps({
        "status": "ANALYSIS_COMPLETE",
        "classification": result["classification"],
        "run_count": len(summaries),
        "command_aligned_rows": len(aligned),
        "counterfactual_replay_rows": len(replayed),
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
