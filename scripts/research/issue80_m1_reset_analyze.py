#!/usr/bin/env python3
"""Classify a one-shot MuJoCo Reset from the matching ROS state trace."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


ROOT_Z_MIN = 0.52
MAX_TILT_DEG = 15.0
STABLE_WINDOW_SEC = 2.5
MAX_STATE_GAP_SEC = 0.5


def stamp_ns(value: dict[str, Any]) -> int:
    return int(value.get("sec", 0)) * 1_000_000_000 + int(value.get("nanosec", 0))


def clock_seconds(value: dict[str, Any]) -> float:
    return float(value.get("sec", 0)) + float(value.get("nanosec", 0)) / 1_000_000_000


def qpos_map(message: dict[str, Any] | None) -> dict[str, float]:
    if not message:
        return {}
    joints = message.get("joints", [])
    return {
        str(joint.get("name", "")): float(joint.get("position", 0.0))
        for joint in joints if joint.get("name") is not None
    }


def joint_difference(first: dict[str, float], second: dict[str, float]) -> dict[str, Any]:
    common = sorted(set(first) & set(second))
    differences = {name: second[name] - first[name] for name in common}
    return {
        "common_joint_count": len(common),
        "l2_rad": math.sqrt(sum(value * value for value in differences.values())) if differences else None,
        "max_abs_rad": max((abs(value) for value in differences.values()), default=None),
        "per_joint_rad": differences,
    }


def classify(trace_path: Path, dispatch_path: Path) -> dict[str, Any]:
    dispatch_doc = json.loads(dispatch_path.read_text(encoding="utf-8"))
    event = dispatch_doc.get("event", {})
    if event.get("status") != "KEY_DISPATCHED":
        return {
            "schema": "robotsim.issue80.reset-analysis.v1",
            "classification": "RESET_NOT_DELIVERED",
            "input_dispatch": event,
            "reason": "X11 helper did not confirm a focused one-shot BackSpace dispatch.",
        }
    event_mono = int(event["dispatch_monotonic_ns"])
    event_wall = event["dispatch_wall_utc"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row.get("wall_monotonic_ns", 0)))
    latest: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    clocks: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event") == "clock":
            clocks.append(row)
        elif row.get("event") == "state":
            key = row.get("topic_key")
            latest[key] = row
            if key == "odom":
                states.append({**row, "latest_common": latest.get("common"), "latest_leg_state": latest.get("leg_state")})
    pre_ns = event_mono - 750_000_000
    pre = [row for row in states if pre_ns <= int(row["wall_monotonic_ns"]) < event_mono]
    post = [row for row in states if int(row["wall_monotonic_ns"]) >= event_mono]
    pre_z = median([float(row["data"]["z"]) for row in pre]) if pre else None
    pre_joint_row = next((row.get("latest_leg_state") for row in reversed(pre) if row.get("latest_leg_state")), None)
    pre_q = qpos_map(pre_joint_row.get("data") if pre_joint_row else None)
    initial_clock = [row for row in clocks if int(row["wall_monotonic_ns"]) < event_mono]
    final_clock = [row for row in clocks if int(row["wall_monotonic_ns"]) >= event_mono]
    clock_values = [clock_seconds(row["clock"]) for row in clocks]
    clock_is_sim_time_candidate = bool(clock_values) and max(clock_values) < 100_000_000
    clock_reset = False
    if clock_is_sim_time_candidate and initial_clock and final_clock:
        before_clock = clock_seconds(initial_clock[-1]["clock"])
        clock_reset = any(clock_seconds(row["clock"]) + 0.01 < before_clock for row in final_clock[:10])
    post_z = [float(row["data"]["z"]) for row in post]
    max_post_z = max(post_z) if post_z else None
    first_upright = next(
        (row for row in post
         if float(row["data"]["z"]) >= ROOT_Z_MIN
         and max(abs(float(row["data"]["roll_deg"])), abs(float(row["data"]["pitch_deg"]))) < MAX_TILT_DEG),
        None,
    )
    first_upright_ns = int(first_upright["wall_monotonic_ns"]) if first_upright else None
    post_q = qpos_map(first_upright.get("latest_leg_state", {}).get("data") if first_upright else None)
    joint_delta = joint_difference(pre_q, post_q)
    first_ready = None
    for row in post:
        pose_ok = (
            float(row["data"]["z"]) >= ROOT_Z_MIN
            and max(abs(float(row["data"]["roll_deg"])), abs(float(row["data"]["pitch_deg"]))) < MAX_TILT_DEG
        )
        common_row = row.get("latest_common")
        if not pose_ok or not common_row:
            continue
        age = (int(row["wall_monotonic_ns"]) - int(common_row["wall_monotonic_ns"])) / 1e9
        common_data = common_row.get("data", {})
        if (
            age <= MAX_STATE_GAP_SEC
            and common_data.get("action_info", {}).get("action_desc") == "STAND_DEFAULT"
            and int(common_data.get("body_status", {}).get("value", -1)) == 1
        ):
            first_ready = row
            break
    first_ready_ns = int(first_ready["wall_monotonic_ns"]) if first_ready else None
    stable_rows: list[dict[str, Any]] = []
    stable_window_complete = False
    stable = False
    fall_reason = None
    max_tilt = None
    max_gap = None
    common_ages: list[float] = []
    if first_ready_ns is not None:
        # Start the stability hold only after pose, MC mode, and body status
        # are simultaneously valid. Immediately after Reset, odometry can
        # become upright one telemetry sample before MC reports body_status=1.
        stable_end = first_ready_ns + int(STABLE_WINDOW_SEC * 1_000_000_000)
        stable_rows = [row for row in post if first_ready_ns <= int(row["wall_monotonic_ns"]) <= stable_end]
        stable_window_complete = bool(post and int(post[-1]["wall_monotonic_ns"]) >= stable_end)
        z_values = [float(row["data"]["z"]) for row in stable_rows]
        tilts = [max(abs(float(row["data"]["roll_deg"])), abs(float(row["data"]["pitch_deg"]))) for row in stable_rows]
        max_tilt = max(tilts) if tilts else None
        times = [int(row["wall_monotonic_ns"]) for row in stable_rows]
        max_gap = max((b - a) / 1e9 for a, b in zip(times, times[1:])) if len(times) > 1 else None
        bad_pose = any(z < ROOT_Z_MIN for z in z_values) or any(t >= MAX_TILT_DEG for t in tilts)
        bad_mode = False
        for row in stable_rows:
            common_row = row.get("latest_common")
            if not common_row:
                bad_mode = True
                continue
            age = (int(row["wall_monotonic_ns"]) - int(common_row["wall_monotonic_ns"])) / 1e9
            common_ages.append(age)
            common_data = common_row.get("data", {})
            if (
                age > MAX_STATE_GAP_SEC
                or common_data.get("action_info", {}).get("action_desc") != "STAND_DEFAULT"
                or int(common_data.get("body_status", {}).get("value", -1)) != 1
            ):
                bad_mode = True
        stable = (
            stable_window_complete
            and len(stable_rows) >= 25
            and not bad_pose
            and not bad_mode
            and (max_gap is None or max_gap <= MAX_STATE_GAP_SEC)
        )
        if bad_pose:
            fall_reason = "root height or tilt crossed the established safety gate in the 2.5 s hold window"
        elif bad_mode:
            fall_reason = "MC common state did not remain in STAND_DEFAULT through the hold window"
        elif max_gap is not None and max_gap > MAX_STATE_GAP_SEC:
            fall_reason = "telemetry gap exceeded 0.5 s"

    if pre_z is not None and pre_z < ROOT_Z_MIN and first_ready_ns is not None:
        classification = "AUTO_RESET_PASS" if stable else (
            "RESET_APPLIED_THEN_FELL" if fall_reason else "RESET_APPLIED_INSUFFICIENT_STABILITY_WINDOW"
        )
    elif pre_z is not None and pre_z < ROOT_Z_MIN and first_upright_ns is not None:
        classification = "RESET_APPLIED_INSUFFICIENT_STABILITY_WINDOW"
        fall_reason = fall_reason or "upright pose observed but fresh STAND_DEFAULT/body_status=1 was not confirmed"
    elif clock_reset and pre_z is not None and pre_z < ROOT_Z_MIN:
        classification = "RESET_APPLIED_BUT_INVALID_INIT"
    else:
        classification = "RESET_NOT_DELIVERED"
    return {
        "schema": "robotsim.issue80.reset-analysis.v1",
        "classification": classification,
        "input_dispatch": event,
        "window": dispatch_doc.get("window"),
        "before_image": dispatch_doc.get("before_image"),
        "after_image": dispatch_doc.get("after_image"),
        "clock_topic_has_messages": bool(clocks),
        "clock_values_look_like_sim_time": clock_is_sim_time_candidate,
        "clock_backwards_reset_detected": clock_reset,
        "clock_first_value_s": clock_values[0] if clock_values else None,
        "clock_last_value_s": clock_values[-1] if clock_values else None,
        "pre_reset_odom_sample_count": len(pre),
        "post_reset_odom_sample_count": len(post),
        "pre_reset_z_median_m": pre_z,
        "post_reset_z_max_m": max_post_z,
        "first_upright_wall_monotonic_ns": first_upright_ns,
        "first_full_stand_ready_wall_monotonic_ns": first_ready_ns,
        "joint_position_change_to_first_upright": joint_delta,
        "stable_window_s": STABLE_WINDOW_SEC,
        "stable_window_start": "first simultaneous pose + fresh STAND_DEFAULT + body_status=1 sample",
        "stable_window_complete": stable_window_complete,
        "stable_window_odom_samples": len(stable_rows),
        "stable_window_max_tilt_deg": max_tilt,
        "stable_window_max_odom_gap_s": max_gap,
        "stable_window_max_common_state_age_s": max(common_ages) if common_ages else None,
        "stable_window_pass": stable,
        "failure_reason": fall_reason,
        "disclaimer": "XTest API return and key acknowledgement are not reset proof; classification uses the correlated live state trace.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify(args.trace, args.dispatch)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["classification"] == "AUTO_RESET_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
