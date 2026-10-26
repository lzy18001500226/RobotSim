#!/usr/bin/env python3
"""Offline forensic analysis for the authorized RAICOM JD-to-SD episodes.

It does not launch Docker, MuJoCo, MC, ROS, or a GUI. It retains a partial
final capture record as a limitation and evaluates only recorded state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from issue80_raicom_mjdata_capture import (
    MjDataCaptureError,
    _is_floor_contact,
    _trace_rows,
    summarize_mjdata_capture,
    write_new_json,
)

MIN_ROOT_Z_M = 0.52
MAX_ABS_TILT_RAD = math.radians(15)
MAX_PLANAR_SPEED_M_S = 0.20
STABLE_HOLD_S = 2.5
MIN_UNIQUE_HOLD_SAMPLES = 25


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


def mode_agrees(common: Mapping[str, Any], readback: Mapping[str, Any]) -> bool:
    return (
        readback.get("header", {}).get("code") == 0
        and readback.get("info", {}).get("action_desc") == "STAND_DEFAULT"
        and readback.get("info", {}).get("status", {}).get("value") == 100
        and common.get("action_info", {}).get("action_desc") == "STAND_DEFAULT"
        and common.get("action_info", {}).get("status", {}).get("value") == 100
        and common.get("body_status", {}).get("value") == 1
        and common.get("fsm_state", {}).get("current_state") == 4
    )


def safe_posture(pose: Mapping[str, Any]) -> bool:
    try:
        z = number(pose["z_m"])
        roll = number(pose["roll_rad"])
        pitch = number(pose["pitch_rad"])
        velocity = pose["linear_velocity_m_s"]
        vx, vy = number(velocity[0]), number(velocity[1])
    except (KeyError, IndexError, TypeError):
        return False
    return (
        None not in (z, roll, pitch, vx, vy)
        and z >= MIN_ROOT_Z_M
        and max(abs(roll), abs(pitch)) < MAX_ABS_TILT_RAD
        and math.hypot(vx, vy) <= MAX_PLANAR_SPEED_M_S
    )


def active_command(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                key in {"stiffness", "damping", "effort"}
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                and abs(float(item)) > 1e-8
            )
            or active_command(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(active_command(item) for item in value)
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise MjDataCaptureError(f"missing or unsafe JSONL evidence: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MjDataCaptureError(f"invalid observer trace row {line_number}") from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def joint_issues(
    sample: Mapping[str, Any],
    limits: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    by_name = {
        str(joint.get("name")): joint
        for joint in sample.get("leg_state", {}).get("joints", [])
        if isinstance(joint, Mapping)
    }
    if set(by_name) != set(limits):
        return ["LEG_SOURCE_LIMIT_MAPPING_INCOMPLETE"]
    issues: list[str] = []
    for name, limit in limits.items():
        joint = by_name[name]
        position, velocity = number(joint.get("position")), number(joint.get("velocity"))
        if position is None or velocity is None:
            issues.append(f"NONFINITE_JOINT:{name}")
            continue
        if not float(limit["lower_position"]) <= position <= float(limit["upper_position"]):
            issues.append(f"SOURCE_JOINT_POSITION_LIMIT:{name}")
        if abs(velocity) > float(limit["max_velocity"]):
            issues.append(f"SOURCE_JOINT_VELOCITY_LIMIT:{name}")
        if joint.get("error_code"):
            issues.append(f"JOINT_ERROR_CODE:{name}")
    return issues


def replay_hold(
    observer_rows: list[dict[str, Any]],
    limits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    latest_get: Mapping[str, Any] | None = None
    latest_get_ns: int | None = None
    sd_ready_ns: int | None = None
    jd_requests = sd_requests = 0
    pre_sd_joint_observations: list[str] = []
    qualified_window_joint_violations: list[str] = []
    post_rows: list[tuple[dict[str, Any], bool]] = []
    source_stamps: set[int] = set()
    duplicate_stamps = 0

    for row in observer_rows:
        event = row.get("event")
        if event == "GET_MC_ACTION":
            response = row.get("response")
            if isinstance(response, Mapping):
                latest_get, latest_get_ns = response, row.get("receipt_monotonic_ns")
            continue
        if event == "JD_REQUEST":
            jd_requests += 1
            continue
        if event == "SD_REQUEST":
            sd_requests += 1
            continue
        if (
            event == "NATIVE_TRANSITION"
            and row.get("phase") == "SD"
            and row.get("state") == "SD_READY"
            and isinstance(row.get("receipt_monotonic_ns"), int)
        ):
            sd_ready_ns = row["receipt_monotonic_ns"]
            continue
        if event != "post_reset_odom_sample":
            continue

        receipt = row.get("receipt_monotonic_ns")
        stamp = row.get("source_stamp_ns")
        if isinstance(stamp, int):
            if stamp in source_stamps:
                duplicate_stamps += 1
            source_stamps.add(stamp)
        else:
            pre_sd_joint_observations.append("MISSING_ODOM_SOURCE_STAMP")
        local_issues = joint_issues(row, limits)
        if sd_ready_ns is not None and isinstance(receipt, int) and receipt >= sd_ready_ns:
            qualified_window_joint_violations.extend(local_issues)
        else:
            pre_sd_joint_observations.extend(local_issues)
        get_fresh = (
            isinstance(receipt, int)
            and isinstance(latest_get_ns, int)
            and 0 <= (receipt - latest_get_ns) / 1e9 <= 0.5
        )
        qualifies = (
            isinstance(receipt, int)
            and sd_ready_ns is not None
            and receipt >= sd_ready_ns
            and isinstance(latest_get, Mapping)
            and isinstance(row.get("raw_common_state"), Mapping)
            and mode_agrees(row["raw_common_state"], latest_get)
            and bool(row.get("fresh_all_required_topics"))
            and get_fresh
            and isinstance(row.get("pose"), Mapping)
            and safe_posture(row["pose"])
            and active_command(row.get("leg_command", {}))
            and not local_issues
        )
        post_rows.append((row, qualifies))

    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row, qualifies in post_rows:
        if qualifies:
            current.append(row)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    longest = max(windows, key=len, default=[])
    duration = (
        (longest[-1]["receipt_monotonic_ns"] - longest[0]["receipt_monotonic_ns"]) / 1e9
        if longest
        else 0.0
    )
    return {
        "jd_request_count": jd_requests,
        "sd_request_count": sd_requests,
        "sd_ready_monotonic_ns": sd_ready_ns,
        "post_reset_odom_sample_count": len(post_rows),
        "unique_odom_source_stamp_count": len(source_stamps),
        "duplicate_odom_source_stamp_count": duplicate_stamps,
        "pre_sd_joint_limit_observations": sorted(set(pre_sd_joint_observations)),
        "qualified_window_joint_limit_violations": sorted(
            set(qualified_window_joint_violations)
        ),
        "longest_continuous_qualified_window": {
            "samples": len(longest),
            "start_monotonic_ns": longest[0]["receipt_monotonic_ns"] if longest else None,
            "end_monotonic_ns": longest[-1]["receipt_monotonic_ns"] if longest else None,
            "duration_s": duration,
        },
        "pass": (
            jd_requests == 1
            and sd_requests == 1
            and len(longest) >= MIN_UNIQUE_HOLD_SAMPLES
            and duration >= STABLE_HOLD_S
            and duplicate_stamps == 0
            and not qualified_window_joint_violations
        ),
    }


def native_metrics(
    native_rows: list[dict[str, Any]],
    begin_ns: int | None,
    end_ns: int | None,
) -> dict[str, Any]:
    samples = [
        row
        for row in native_rows
        if row.get("event") in {"post_forward_initial", "post_forward_reset_generation", "post_step"}
        and row.get("generation") == 1
        and isinstance(row.get("host_monotonic_ns"), int)
        and (begin_ns is None or row["host_monotonic_ns"] >= begin_ns)
        and (end_ns is None or row["host_monotonic_ns"] <= end_ns)
    ]
    if not samples:
        return {"sample_count": 0}

    speeds: list[float] = []
    left_forces: list[float] = []
    right_forces: list[float] = []
    tangent_forces: list[float] = []
    tangent_speeds: list[float] = []
    min_distance: float | None = None
    bilateral = True
    actuator_peak = qfrc_peak = 0.0
    for sample in samples:
        state = sample["pelvis"]["state"]
        velocity = state.get("world_spatial_velocity_rot_lin", [])
        if isinstance(velocity, list) and len(velocity) == 6:
            vx, vy = number(velocity[3]), number(velocity[4])
            if vx is not None and vy is not None:
                speeds.append(math.hypot(vx, vy))
        left = right = 0.0
        for contact in sample.get("contacts", []):
            if not isinstance(contact, Mapping) or not _is_floor_contact(contact):
                continue
            normal = number(contact.get("normal_force_n"))
            tangent = number(contact.get("tangential_force_norm_n"))
            slide = number(contact.get("relative_tangential_speed_m_s"))
            distance = number(contact.get("distance_m"))
            if normal is not None:
                if contact.get("involves_selected_left_foot"):
                    left += normal
                if contact.get("involves_selected_right_foot"):
                    right += normal
            if tangent is not None:
                tangent_forces.append(abs(tangent))
            if slide is not None:
                tangent_speeds.append(abs(slide))
            if distance is not None:
                min_distance = distance if min_distance is None else min(min_distance, distance)
        left_forces.append(left)
        right_forces.append(right)
        bilateral = bilateral and left > 0.0 and right > 0.0
        for value in sample.get("actuator_force", []):
            parsed = number(value)
            if parsed is not None:
                actuator_peak = max(actuator_peak, abs(parsed))
        for value in sample.get("qfrc_actuator", []):
            parsed = number(value)
            if parsed is not None:
                qfrc_peak = max(qfrc_peak, abs(parsed))

    first = samples[0]["pelvis"]["state"]
    last = samples[-1]["pelvis"]["state"]
    first_pos, last_pos = first["world_position_m"], last["world_position_m"]
    dx, dy, dz = (
        float(last_pos[0]) - float(first_pos[0]),
        float(last_pos[1]) - float(first_pos[1]),
        float(last_pos[2]) - float(first_pos[2]),
    )
    return {
        "sample_count": len(samples),
        "first_sim_time_s": samples[0].get("sim_time_s"),
        "last_sim_time_s": samples[-1].get("sim_time_s"),
        "first_world_pelvis": first,
        "last_world_pelvis": last,
        "world_pelvis_displacement": {
            "dx_m": dx,
            "dy_m": dy,
            "planar_m": math.hypot(dx, dy),
            "dz_m": dz,
        },
        "max_world_planar_speed_m_s": max(speeds) if speeds else None,
        "max_left_foot_floor_normal_force_n": max(left_forces) if left_forces else None,
        "max_right_foot_floor_normal_force_n": max(right_forces) if right_forces else None,
        "min_left_foot_floor_normal_force_n": min(left_forces) if left_forces else None,
        "min_right_foot_floor_normal_force_n": min(right_forces) if right_forces else None,
        "max_tangential_force_n": max(tangent_forces) if tangent_forces else None,
        "max_relative_tangential_speed_m_s": max(tangent_speeds) if tangent_speeds else None,
        "min_contact_distance_m": min_distance,
        "bilateral_foot_floor_contact_every_native_sample": bilateral,
        "max_abs_actuator_force": actuator_peak,
        "max_abs_generalized_actuator_force": qfrc_peak,
    }


def native_landmarks(
    native_rows: list[dict[str, Any]],
    observer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = [
        row
        for row in native_rows
        if row.get("generation") == 1
        and row.get("event") in {"post_forward_initial", "post_forward_reset_generation", "post_step"}
        and isinstance(row.get("host_monotonic_ns"), int)
    ]
    if not samples:
        return {"status": "NO_NATIVE_GENERATION_ONE_SAMPLES"}
    requested: dict[str, int] = {}
    for row in observer_rows:
        event = row.get("event")
        if event in {"JD_REQUEST", "SD_REQUEST", "JD_RESPONSE", "SD_RESPONSE"}:
            if isinstance(row.get("receipt_monotonic_ns"), int):
                requested[event] = row["receipt_monotonic_ns"]
        if (
            event == "NATIVE_TRANSITION"
            and row.get("state") in {"JD_READY", "SD_READY"}
            and isinstance(row.get("receipt_monotonic_ns"), int)
        ):
            requested[row["state"]] = row["receipt_monotonic_ns"]

    def contact_forces(sample: Mapping[str, Any]) -> tuple[float, float]:
        left = right = 0.0
        for contact in sample.get("contacts", []):
            if not isinstance(contact, Mapping) or not _is_floor_contact(contact):
                continue
            normal = number(contact.get("normal_force_n"))
            if normal is None:
                continue
            if contact.get("involves_selected_left_foot"):
                left += normal
            if contact.get("involves_selected_right_foot"):
                right += normal
        return left, right

    def compact(sample: Mapping[str, Any], target: int | None = None) -> dict[str, Any]:
        state = sample["pelvis"]["state"]
        vector = state["world_spatial_velocity_rot_lin"]
        left, right = contact_forces(sample)
        actuator_forces = [
            abs(float(value)) for value in sample.get("actuator_force", [])
            if number(value) is not None
        ]
        return {
            "target_host_monotonic_ns": target,
            "sample_host_delta_ms": (
                (sample["host_monotonic_ns"] - target) / 1e6
                if target is not None
                else None
            ),
            "sim_time_s": sample.get("sim_time_s"),
            "world_pelvis_position_m": state.get("world_position_m"),
            "world_pelvis_linear_velocity_m_s": vector[3:6],
            "world_planar_speed_m_s": math.hypot(float(vector[3]), float(vector[4])),
            "all_contact_count": sample.get("contact_count"),
            "captured_foot_floor_contact_count": sample.get(
                "captured_foot_floor_contact_count"
            ),
            "left_foot_floor_normal_force_n": left,
            "right_foot_floor_normal_force_n": right,
            "max_abs_actuator_force": max(actuator_forces, default=0.0),
        }

    result = {"reset_generation_initial": compact(samples[0])}
    first_contact = next(
        (
            sample
            for sample in samples
            if int(sample.get("captured_foot_floor_contact_count", 0)) > 0
        ),
        None,
    )
    result["first_bilateral_foot_floor_contact"] = (
        compact(first_contact) if first_contact is not None else None
    )
    for label, timestamp in requested.items():
        result[label] = compact(
            min(samples, key=lambda sample: abs(sample["host_monotonic_ns"] - timestamp)),
            timestamp,
        )
    result["last_complete_native_sample"] = compact(samples[-1])
    return result


def build_report(standing_evidence: Path, run65_comparison: Path) -> dict[str, Any]:
    source = standing_evidence.resolve()
    observer_rows = load_jsonl(source / "reset_only_state_trace.jsonl")
    limits = json.loads((source / "standing_joint_limits.json").read_text())["limits"]
    replay = replay_hold(observer_rows, limits)
    native_rows, truncated = _trace_rows(source / "native_mjdata_trace.jsonl")
    window = replay["longest_continuous_qualified_window"]
    native = native_metrics(
        native_rows, window["start_monotonic_ns"], window["end_monotonic_ns"]
    )
    full_native = native_metrics(native_rows, None, None)
    timeline = native_landmarks(native_rows, observer_rows)
    native_summary = summarize_mjdata_capture(source / "native_mjdata_trace.jsonl")
    historical = json.loads(run65_comparison.read_text())
    first_contact = timeline.get("first_bilateral_foot_floor_contact") or {}
    reset_initial = timeline.get("reset_generation_initial") or {}
    physical_pass = (
        replay["pass"]
        and native.get("sample_count", 0) > 0
        and native.get("max_world_planar_speed_m_s", float("inf")) <= MAX_PLANAR_SPEED_M_S
        and native.get("bilateral_foot_floor_contact_every_native_sample")
    )
    return {
        "schema": "robotsim.issue80.raicom-standing-drift-forensics.v1",
        "source_evidence": str(source),
        "native_trace_truncated_final_record": truncated,
        "replayed_observer_gate": replay,
        "native_mjdata_during_qualified_hold": native,
        "native_mjdata_full_reset_generation": full_native,
        "synchronized_transition_timeline": timeline,
        "native_trace_summary": native_summary,
        "historical_run65_vs_prior_raicom_comparison": historical,
        "physical_standing_evidence": (
            "PASS_RECONSTRUCTED_FROM_RAW_NATIVE_AND_ROS_TRACE"
            if physical_pass
            else "NOT_PROVEN"
        ),
        "original_runner_closeout": "BLOCKED_NO_STAND_TAKEOVER_SUMMARY",
        "prior_0p209_m_s_root_cause": (
            "UNRESOLVED: the authorized native MjData runs did not reproduce a "
            "world-frame speed above the unchanged 0.20 m/s safety gate."
        ),
        "observed_startup_support_mechanism": {
            "finding": (
                "The Reset state begins with zero captured foot-floor contacts; "
                "the root settles under gravity before bilateral support is present "
                "and before JD is requested."
            ),
            "reset_initial_world_pelvis": reset_initial.get("world_pelvis_position_m"),
            "first_foot_floor_contact_sim_time_s": first_contact.get("sim_time_s"),
            "first_foot_floor_contact_world_pelvis": first_contact.get(
                "world_pelvis_position_m"
            ),
            "first_foot_floor_contact_world_linear_velocity_m_s": first_contact.get(
                "world_pelvis_linear_velocity_m_s"
            ),
            "claim_boundary": (
                "This directly explains the ordinary reset-settling transient in "
                "the instrumented runs. It does not uniquely establish why the "
                "previous uninstrumented RAICOM trial reached 0.209026 m/s."
            ),
        },
        "timestep_hypothesis": "NOT_SUPPORTED; no 0.001 s A/B was run.",
        "limitations": [
            "The final capture JSONL record was incomplete during teardown and was not used.",
            "The original runner did not write stand_takeover_summary.json.",
            "This task sent no locomotion or velocity command.",
            "MuJoCo timer duration values are retained as raw instrumentation values.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    replay = report["replayed_observer_gate"]
    window = replay["longest_continuous_qualified_window"]
    native = report["native_mjdata_during_qualified_hold"]
    historical = report["historical_run65_vs_prior_raicom_comparison"]
    return """# RAICOM JD-to-SD standing drift forensic

Physical evidence: {physical}

Runner closeout: {closeout}

## Replayed continuous standing window

- JD requests: {jd}
- SD requests: {sd}
- SD ready: {ready}
- Unique qualified samples: {samples}
- Continuous hold: {duration:.9f} s
- Joint-limit violations during hold: {violations}

## Native MjData over that same window

- Native samples: {native_samples}
- Peak world-frame pelvis planar speed: {speed} m/s
- World XY displacement: {displacement} m
- Left / right foot normal-force maxima: {left} / {right} N
- Peak tangential relative speed: {slide} m/s
- Bilateral foot-floor support each sampled instant: {bilateral}

## Synchronized transition facts

- First physical foot-floor contact at simulation time:
  {first_contact} s
- JD request physical world planar speed:
  {jd_speed} m/s
- SD request physical world planar speed:
  {sd_speed} m/s
- SD-ready physical world planar speed:
  {sd_ready_speed} m/s

## Historical comparison

Run65 published-speed maximum was {run65_speed} m/s. The prior RAICOM
published-speed maximum was {prior_speed} m/s. They came from different
scenes/runners and do not prove a unique causal difference.

## Limits

{limits}
""".format(
        physical=report["physical_standing_evidence"],
        closeout=report["original_runner_closeout"],
        jd=replay["jd_request_count"],
        sd=replay["sd_request_count"],
        ready=replay["sd_ready_monotonic_ns"],
        samples=window["samples"],
        duration=window["duration_s"],
        violations=len(replay["qualified_window_joint_limit_violations"]),
        native_samples=native.get("sample_count"),
        speed=native.get("max_world_planar_speed_m_s"),
        displacement=native.get("world_pelvis_displacement", {}).get("planar_m"),
        left=native.get("max_left_foot_floor_normal_force_n"),
        right=native.get("max_right_foot_floor_normal_force_n"),
        slide=native.get("max_relative_tangential_speed_m_s"),
        bilateral=native.get("bilateral_foot_floor_contact_every_native_sample"),
        first_contact=report["synchronized_transition_timeline"]
        .get("first_bilateral_foot_floor_contact", {})
        .get("sim_time_s"),
        jd_speed=report["synchronized_transition_timeline"]
        .get("JD_REQUEST", {})
        .get("world_planar_speed_m_s"),
        sd_speed=report["synchronized_transition_timeline"]
        .get("SD_REQUEST", {})
        .get("world_planar_speed_m_s"),
        sd_ready_speed=report["synchronized_transition_timeline"]
        .get("SD_READY", {})
        .get("world_planar_speed_m_s"),
        run65_speed=historical["run65"]["metrics"]["max_planar_speed_m_s"],
        prior_speed=historical["raicom"]["metrics"]["max_planar_speed_m_s"],
        limits="\n".join("- " + item for item in report["limitations"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standing-evidence", type=Path, required=True)
    parser.add_argument("--run65-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        if args.output_dir.is_symlink() or any(
            child.name != ".robotsim-output-reservation.json"
            for child in args.output_dir.iterdir()
        ):
            parser.error("output directory must be new or a governance reservation only")
    else:
        args.output_dir.mkdir(parents=True)
    report = build_report(args.standing_evidence, args.run65_comparison)
    write_new_json(args.output_dir / "standing_drift_forensics.json", report)
    (args.output_dir / "STANDING_DRIFT_FORENSIC.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "physical": report["physical_standing_evidence"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
