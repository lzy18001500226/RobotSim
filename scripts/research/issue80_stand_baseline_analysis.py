#!/usr/bin/env python3
"""Analyze a zero-locomotion X2 STAND_DEFAULT trace in rolling 2 s windows."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
from typing import Any, Mapping

from issue80_m1_first_motion import (
    MC_FSM_STABLE,
    MODE_STAND,
    PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD,
    PHYSICAL_STOP_MAX_HEIGHT_RANGE_M,
    PHYSICAL_STOP_MAX_JOINT_RANGE_RAD,
    PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S,
    PHYSICAL_STOP_MAX_PLANAR_DRIFT_M,
    PHYSICAL_STOP_MAX_PLANAR_PATH_M,
    PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S,
    PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD,
    PHYSICAL_STOP_MAX_TILT_RAD,
    PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S,
    PHYSICAL_STOP_MAX_YAW_DRIFT_RAD,
    PHYSICAL_STOP_MAX_YAW_RATE_RAD_S,
    PHYSICAL_STOP_MAX_SAMPLE_GAP_S,
    PHYSICAL_STOP_MAX_STATE_AGE_S,
    PHYSICAL_STOP_MIN_SAMPLES,
    PHYSICAL_STOP_MIN_WINDOW_S,
    PHYSICAL_STOP_WINDOW_S,
)
from issue80_m1_set_action_capture import MAX_TILT_DEG, ROOT_Z_MIN_M


REQUIRED_TOPICS = ("odom", "imu", "common", "leg_state", "leg_command")
ROLLING_STEP_S = 0.5


class StandingAnalysisError(ValueError):
    pass


def _enum(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "value" in value:
            return value["value"]
        if "current_state" in value:
            return value["current_state"]
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _latest_at(rows: list[dict[str, Any]], times: list[int], stamp_ns: int) -> dict[str, Any] | None:
    index = bisect_right(times, stamp_ns) - 1
    return rows[index] if index >= 0 else None


def normalize_trace_samples(
    trace_rows: list[dict[str, Any]], *, start_ns: int, end_ns: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Align latest state to leg-target samples without interpolating telemetry."""
    streams: dict[str, list[dict[str, Any]]] = {key: [] for key in REQUIRED_TOPICS}
    for row in trace_rows:
        if row.get("event") == "state" and row.get("topic_key") in streams:
            streams[str(row["topic_key"])].append(row)
    stamps: dict[str, list[int]] = {}
    for key, rows in streams.items():
        ordered = sorted(rows, key=lambda row: int(row.get("wall_monotonic_ns", 0)))
        streams[key] = ordered
        stamps[key] = [int(row.get("wall_monotonic_ns", 0)) for row in ordered]

    aligned: list[dict[str, Any]] = []
    dropped = 0
    for command_row, sample_ns in zip(streams["leg_command"], stamps["leg_command"]):
        if not start_ns <= sample_ns <= end_ns:
            continue
        latest = {
            key: _latest_at(streams[key], stamps[key], sample_ns)
            for key in REQUIRED_TOPICS if key != "leg_command"
        }
        if any(row is None for row in latest.values()):
            dropped += 1
            continue
        odom = latest["odom"].get("data") or {}
        common = latest["common"].get("data") or {}
        imu = latest["imu"].get("data") or {}
        state = latest["leg_state"].get("data") or {}
        command = command_row.get("data") or {}
        state_joints = state.get("joints")
        command_joints = command.get("joints")
        if not isinstance(state_joints, list) or not isinstance(command_joints, list):
            dropped += 1
            continue
        linear = odom.get("linear_velocity") or {}
        angular = odom.get("angular_velocity") or {}
        action = common.get("action_info") or {}
        input_source = common.get("input_source") or {}
        aligned.append({
            "sample_monotonic_s": sample_ns / 1e9,
            "sample_monotonic_ns": sample_ns,
            "state_age_s": {
                key: (sample_ns - int(row["wall_monotonic_ns"])) / 1e9
                for key, row in latest.items() if row is not None
            },
            "pose": {
                "x": odom.get("x"), "y": odom.get("y"), "z": odom.get("z"),
                "roll": odom.get("roll_rad"), "pitch": odom.get("pitch_rad"),
                "yaw": odom.get("yaw_rad"),
                "vx": linear.get("x"), "vy": linear.get("y"), "vz": linear.get("z"),
            },
            "imu": imu,
            "leg_states": state_joints,
            "mc_leg_command": {"joints": command_joints},
            "common_mode_desc": action.get("action_desc"),
            "common_mode_status_value": _enum(action.get("status")),
            "fsm_state": _enum(common.get("fsm_state")),
            "body_status": _enum(common.get("body_status")),
            "input_source": input_source,
        })
    return aligned, {"dropped_leg_command_rows": dropped, "aligned_sample_count": len(aligned)}


def _unwrap(values: list[float]) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        delta = (value - result[-1] + math.pi) % (2.0 * math.pi) - math.pi
        result.append(result[-1] + delta)
    return result


def _slope(times: list[float], values: list[float]) -> float | None:
    if len(times) < 2 or len(times) != len(values):
        return None
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    denominator = sum((time - mean_t) ** 2 for time in times)
    if denominator <= 0.0:
        return None
    return sum((time - mean_t) * (value - mean_v) for time, value in zip(times, values)) / denominator


def _quaternion_rpy(quaternion: Mapping[str, Any]) -> tuple[float, float, float] | None:
    try:
        x, y, z, w = (float(quaternion[key]) for key in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        return None
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def summarize_window(
    samples: list[dict[str, Any]], *, expected_body_status: Any = None,
) -> dict[str, Any]:
    if len(samples) < 2:
        raise StandingAnalysisError("window needs at least two synchronized samples")
    rows = sorted(samples, key=lambda row: float(row["sample_monotonic_s"]))
    times = [float(row["sample_monotonic_s"]) for row in rows]
    poses = [row["pose"] for row in rows]
    fields = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")
    if any(not isinstance(pose, Mapping) or any(not _finite(pose.get(key)) for key in fields) for pose in poses):
        raise StandingAnalysisError("window has missing or nonfinite odometry")
    yaw = _unwrap([float(pose["yaw"]) for pose in poses])
    x0, y0 = float(poses[0]["x"]), float(poses[0]["y"])
    excursion = [math.hypot(float(pose["x"]) - x0, float(pose["y"]) - y0) for pose in poses]
    path = sum(
        math.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"]))
        for a, b in zip(poses, poses[1:])
    )
    state_values: dict[str, dict[str, list[float]]] = {}
    targets: dict[str, list[float]] = {}
    errors: set[int] = set()
    for row in rows:
        actual = {str(joint["name"]): joint for joint in row["leg_states"]}
        commanded = {str(joint["name"]): joint for joint in row["mc_leg_command"]["joints"]}
        if not actual or set(actual) != set(commanded):
            raise StandingAnalysisError("measured joints and MC target names differ")
        for name, joint in actual.items():
            bucket = state_values.setdefault(name, {"position": [], "velocity": []})
            bucket["position"].append(float(joint["position"]))
            bucket["velocity"].append(float(joint["velocity"]))
            targets.setdefault(name, []).append(float(commanded[name]["position"]))
            errors.add(int(joint.get("error_code", 0)))
    per_joint_motion = {}
    for name, values in state_values.items():
        target_values = targets[name]
        position_values = values["position"]
        velocity_values = values["velocity"]
        per_joint_motion[name] = {
            "target_rad": {
                "start": target_values[0], "end": target_values[-1],
                "min": min(target_values), "max": max(target_values),
                "range": max(target_values) - min(target_values),
            },
            "actual_position_rad": {
                "start": position_values[0], "end": position_values[-1],
                "min": min(position_values), "max": max(position_values),
                "range": max(position_values) - min(position_values),
            },
            "actual_velocity_rad_s": {
                "start": velocity_values[0], "end": velocity_values[-1],
                "min": min(velocity_values), "max": max(velocity_values),
                "max_abs": max(abs(value) for value in velocity_values),
            },
        }
    target_ranges = {name: max(values) - min(values) for name, values in targets.items()}
    position_ranges = {
        name: max(values["position"]) - min(values["position"])
        for name, values in state_values.items()
    }
    joint_velocity_max = max(abs(value) for values in state_values.values() for value in values["velocity"])
    ages = [float(age) for row in rows for age in row["state_age_s"].values()]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    modes = sorted({str(row.get("common_mode_desc")) for row in rows})
    mode_statuses = sorted({row.get("common_mode_status_value") for row in rows}, key=str)
    fsm = sorted({row.get("fsm_state") for row in rows}, key=str)
    body = sorted({row.get("body_status") for row in rows}, key=str)
    sources = sorted({str((row.get("input_source") or {}).get("name", "")) for row in rows})
    duration = times[-1] - times[0]
    yaw_rate = _slope(times, yaw)
    metrics = {
        "max_planar_speed_m_s": max(math.hypot(float(p["vx"]), float(p["vy"])) for p in poses),
        "max_vertical_speed_m_s": max(abs(float(p["vz"])) for p in poses),
        "planar_endpoint_drift_m": math.hypot(float(poses[-1]["x"]) - x0, float(poses[-1]["y"]) - y0),
        "max_planar_excursion_m": max(excursion),
        "planar_path_length_m": path,
        "max_abs_yaw_excursion_rad": max(abs(value - yaw[0]) for value in yaw),
        "yaw_rate_fit_rad_s": abs(yaw_rate) if yaw_rate is not None else math.inf,
        "z_range_m": max(float(p["z"]) for p in poses) - min(float(p["z"]) for p in poses),
        "max_abs_tilt_rad": max(math.hypot(float(p["roll"]), float(p["pitch"])) for p in poses),
        "roll_range_rad": max(float(p["roll"]) for p in poses) - min(float(p["roll"]) for p in poses),
        "pitch_range_rad": max(float(p["pitch"]) for p in poses) - min(float(p["pitch"]) for p in poses),
        "max_abs_joint_velocity_rad_s": joint_velocity_max,
        "max_joint_position_range_rad": max(position_ranges.values()),
    }
    limits = {
        "max_planar_speed_m_s": PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S,
        "max_vertical_speed_m_s": PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S,
        "planar_endpoint_drift_m": PHYSICAL_STOP_MAX_PLANAR_DRIFT_M,
        "max_planar_excursion_m": PHYSICAL_STOP_MAX_PLANAR_DRIFT_M,
        "planar_path_length_m": PHYSICAL_STOP_MAX_PLANAR_PATH_M,
        "max_abs_yaw_excursion_rad": PHYSICAL_STOP_MAX_YAW_DRIFT_RAD,
        "yaw_rate_fit_rad_s": PHYSICAL_STOP_MAX_YAW_RATE_RAD_S,
        "z_range_m": PHYSICAL_STOP_MAX_HEIGHT_RANGE_M,
        "max_abs_tilt_rad": PHYSICAL_STOP_MAX_TILT_RAD,
        "roll_range_rad": PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD,
        "pitch_range_rad": PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD,
        "max_abs_joint_velocity_rad_s": PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S,
        "max_joint_position_range_rad": PHYSICAL_STOP_MAX_JOINT_RANGE_RAD,
    }
    violations = []
    if duration < PHYSICAL_STOP_MIN_WINDOW_S:
        violations.append("window_too_short")
    if len(rows) < PHYSICAL_STOP_MIN_SAMPLES:
        violations.append("too_few_samples")
    if not gaps or max(gaps) > PHYSICAL_STOP_MAX_SAMPLE_GAP_S:
        violations.append("sample_gap_exceeded")
    if not ages or max(ages) > PHYSICAL_STOP_MAX_STATE_AGE_S:
        violations.append("telemetry_stale")
    if modes != [MODE_STAND] or mode_statuses != [100]:
        violations.append("common_state_not_stand_throughout")
    if fsm != [MC_FSM_STABLE]:
        violations.append("raw_fsm_not_stable_throughout")
    if expected_body_status is not None and body != [expected_body_status]:
        violations.append("raw_body_status_changed")
    elif len(body) != 1:
        violations.append("raw_body_status_changed")
    if errors != {0}:
        violations.append("joint_error_code_present")
    if min(float(pose["z"]) for pose in poses) < ROOT_Z_MIN_M:
        violations.append("root_height_below_existing_upright_gate")
    if max(max(abs(float(pose["roll"])), abs(float(pose["pitch"]))) for pose in poses) >= math.radians(MAX_TILT_DEG):
        violations.append("tilt_exceeded_existing_upright_gate")
    violations.extend(f"{key}_exceeded" for key, limit in limits.items() if metrics[key] > limit)
    target_ranges = {name: max(values) - min(values) for name, values in targets.items()}
    target_gate = {
        "status": "PASS" if max(target_ranges.values()) <= PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD else "FAIL",
        "threshold_rad": PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD,
        "max_target_range_rad": max(target_ranges.values()),
        "violating_joints": sorted(name for name, value in target_ranges.items() if value > PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD),
        "per_joint_target_range_rad": target_ranges,
        "per_joint_actual_position_range_rad": position_ranges,
    }
    imu_rpy = [
        value for value in (_quaternion_rpy((row.get("imu") or {}).get("orientation") or {}) for row in rows)
        if value is not None
    ]
    return {
        "start_monotonic_s": times[0], "end_monotonic_s": times[-1], "duration_s": duration,
        "sample_count": len(rows), "max_sample_gap_s": max(gaps) if gaps else None,
        "max_state_age_s": max(ages) if ages else None,
        "common_mode_values": modes, "common_mode_status_values": mode_statuses,
        "raw_fsm_values": fsm, "raw_body_status_values": body, "input_source_values": sources,
        "joint_error_codes": sorted(errors), "metrics": metrics, "thresholds": limits,
        "physical_state_gate": {"status": "PASS" if not violations else "FAIL", "violations": violations},
        "controller_target_stability_gate": target_gate,
        "per_joint_leg_motion": per_joint_motion,
        "yaw_drift_rad": yaw[-1] - yaw[0],
        "imu_orientation_rpy_range_rad": (
            {axis: max(value[index] for value in imu_rpy) - min(value[index] for value in imu_rpy)
             for index, axis in enumerate(("roll", "pitch", "yaw"))}
            if imu_rpy else None
        ),
        "imu_angular_velocity_max_abs_rad_s": {
            axis: max(abs(float((row.get("imu") or {}).get("angular_velocity", {}).get(axis, 0.0))) for row in rows)
            for axis in ("x", "y", "z")
        },
    }


def rolling_windows(
    samples: list[dict[str, Any]], *, window_s: float = PHYSICAL_STOP_WINDOW_S,
    step_s: float = ROLLING_STEP_S, expected_body_status: Any = None,
) -> list[dict[str, Any]]:
    if not samples or not _finite(window_s) or window_s <= 0 or not _finite(step_s) or step_s <= 0:
        raise StandingAnalysisError("rolling-window inputs are invalid")
    ordered = sorted(samples, key=lambda row: float(row["sample_monotonic_s"]))
    first, last = float(ordered[0]["sample_monotonic_s"]), float(ordered[-1]["sample_monotonic_s"])
    output = []
    start = first
    while start + window_s <= last + 1e-9:
        selected = [row for row in ordered if start <= float(row["sample_monotonic_s"]) <= start + window_s]
        if len(selected) >= 2:
            output.append({"relative_start_s": start - first, "metrics": summarize_window(selected, expected_body_status=expected_body_status)})
        start += step_s
    return output


def analyze_standing_trace(
    trace_rows: list[dict[str, Any]], takeover: Mapping[str, Any], observer: Mapping[str, Any],
    *, minimum_stand_observation_s: float = 8.0,
) -> dict[str, Any]:
    stand_ns = takeover.get("stand_default_mode_confirmed_monotonic_ns")
    observation = observer.get("standing_baseline_observation") or {}
    end_ns = observation.get("observed_end_monotonic_ns")
    observed_s = observation.get("observed_duration_s", 0.0)
    result: dict[str, Any] = {
        "schema": "robotsim.issue80.stand-baseline-analysis.v1",
        "classification": "STAND_BASELINE_INCOMPLETE",
        "standing_observation": observation,
        "stand_mode_readback": takeover.get("stand_default_set_mc_action"),
        "stand_default_mode_confirmed_monotonic_ns": stand_ns,
        "final_stand_mode_readback": observer.get("final_stand_mode_readback"),
        "source_end": observer.get("final_common_input_source"),
        "zero_velocity_publisher_or_command_created": False,
        "minimum_stand_observation_s": minimum_stand_observation_s,
        "minimum_duration_pass": _finite(observed_s) and float(observed_s) >= minimum_stand_observation_s,
    }
    if not isinstance(stand_ns, int) or not isinstance(end_ns, int) or not _finite(observed_s) or float(observed_s) < minimum_stand_observation_s:
        result["blocker"] = "positive STAND readback or complete >=8 s post-readback observation missing"
        return result
    aligned, alignment = normalize_trace_samples(trace_rows, start_ns=stand_ns, end_ns=end_ns)
    trace_fsm_values = []
    for row in aligned:
        value = row.get("fsm_state")
        if value not in trace_fsm_values:
            trace_fsm_values.append(value)
    trace_fsm_values.sort(key=str)
    summary_fsm_values = observation.get("raw_fsm_state_values")
    result["fsm_trace_reconciliation"] = {
        "trace_raw_current_state_values": trace_fsm_values,
        "observer_summary_values": summary_fsm_values,
        "observer_summary_mismatch": summary_fsm_values != trace_fsm_values,
        "authoritative_field": "CommonState.fsm_state.current_state in the preserved raw trace",
        "interpretation": "raw FSM integers are preserved; this report does not assign undocumented meanings",
    }
    result["source_start"] = aligned[0].get("input_source") if aligned else None
    aligned_duration_s = (
        float(aligned[-1]["sample_monotonic_s"]) - float(aligned[0]["sample_monotonic_s"])
        if len(aligned) >= 2 else 0.0
    )
    result["aligned_target_trace_duration_s"] = aligned_duration_s
    edge_start_gap_s = float(aligned[0]["sample_monotonic_s"]) - stand_ns / 1e9 if aligned else math.inf
    edge_end_gap_s = end_ns / 1e9 - float(aligned[-1]["sample_monotonic_s"]) if aligned else math.inf
    result["aligned_trace_edge_gaps_s"] = {"start": edge_start_gap_s, "end": edge_end_gap_s}
    edge_coverage_limit_s = PHYSICAL_STOP_MAX_STATE_AGE_S
    edge_coverage_pass = (
        math.isfinite(edge_start_gap_s) and math.isfinite(edge_end_gap_s)
        and 0.0 <= edge_start_gap_s <= edge_coverage_limit_s
        and 0.0 <= edge_end_gap_s <= edge_coverage_limit_s
    )
    result["aligned_trace_edge_coverage"] = {
        "status": "PASS" if edge_coverage_pass else "FAIL",
        "maximum_gap_s": edge_coverage_limit_s,
        "basis": "existing maximum telemetry age threshold",
    }
    if not edge_coverage_pass:
        result["classification"] = "STAND_BASELINE_INCOMPLETE"
        result["blocker"] = "aligned target telemetry is missing or outside the existing freshness bound at an observation edge"
        result["alignment"] = alignment
        return result
    body_status = _enum((takeover.get("joint_to_stand_gate") or {}).get("raw_body_status"))
    windows = rolling_windows(aligned, expected_body_status=body_status)
    if not windows:
        result["blocker"] = "no complete 2 s windows after confirmed STAND"
        return result
    physical_failures = sum(row["metrics"]["physical_state_gate"]["status"] != "PASS" for row in windows)
    target_failures = sum(row["metrics"]["controller_target_stability_gate"]["status"] != "PASS" for row in windows)
    final_window = windows[-1]["metrics"]
    final_physical_pass = final_window["physical_state_gate"]["status"] == "PASS"
    final_target_pass = final_window["controller_target_stability_gate"]["status"] == "PASS"
    if not final_physical_pass:
        classification = "STAND_BASELINE_FINAL_PHYSICAL_WINDOW_UNSTABLE_OR_UNVERIFIED"
    elif not final_target_pass:
        classification = "TARGET_VARIATION_REPRODUCED_DURING_ZERO_COMMAND_STANDING"
    elif physical_failures or target_failures:
        classification = "FINAL_2S_GATES_PASS_AFTER_EARLY_WINDOW_TRANSIENT"
    else:
        classification = "TARGET_STABLE_DURING_ZERO_COMMAND_STANDING_POST_LOCOMOTION_TRANSITION_SUSPECTED"
    result.update({
        "classification": classification,
        "alignment": alignment,
        "window_s": PHYSICAL_STOP_WINDOW_S,
        "rolling_step_s": ROLLING_STEP_S,
        "rolling_windows": windows,
        "final_2s_window": final_window,
        "source_at_start_of_stand_window": (aligned[0].get("input_source") or {}),
        "source_at_end_of_stand_window": (aligned[-1].get("input_source") or {}),
        "physical_failure_window_count": physical_failures,
        "target_failure_window_count": target_failures,
        "final_physical_window_pass": final_physical_pass,
        "final_target_stability_window_pass": final_target_pass,
        "positive_get_mc_action_readbacks": [
            row for row in (takeover.get("get_mc_action_readbacks") or [])
            if row.get("action_desc") == MODE_STAND and row.get("action_status_value") == 100
        ],
        "controller_target_interpretation": (
            "The unchanged 0.020 rad criterion was exceeded during a zero-locomotion STAND baseline; this is measured baseline behavior, not proof of acceptable hardware behavior."
            if target_failures else
            "The target criterion passed this zero-locomotion baseline; a post-locomotion transition remains suspected but is not proven by this experiment."
        ),
        "limitations": [
            "No foot contact-force or ground-reaction-force topic is exported by the installed simulator interface.",
            "CommonState mode/source telemetry is continuous; GetMcAction is explicitly sampled at the standing handoff and observation end, not every state sample.",
            "This baseline applies to the captured Reset/PD/JD/SD sequence and SIMULATION_ONLY diagnostic controller configuration only.",
            "A target-stable baseline alone does not authorize route rearm; the existing composite stop gate remains unchanged.",
        ],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--takeover-result", required=True, type=Path)
    parser.add_argument("--observer-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = []
        for line in args.trace.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        result = analyze_standing_trace(
            rows,
            json.loads(args.takeover_result.read_text(encoding="utf-8")),
            json.loads(args.observer_summary.read_text(encoding="utf-8")),
        )
        from agent.path_governance import write_new_text
        write_new_text(args.output, json.dumps(result, indent=2) + "\n")
    except (OSError, KeyError, TypeError, ValueError, StandingAnalysisError) as exc:
        parser.error(f"standing baseline analysis failed: {type(exc).__name__}: {exc}")
    print(json.dumps({k: result.get(k) for k in ("classification", "minimum_stand_observation_s", "physical_failure_window_count", "target_failure_window_count")}, indent=2))
    return 0 if result.get("classification") != "STAND_BASELINE_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
