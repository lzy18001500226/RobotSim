"""Unchanged physical-stop contract reused from Issue #80.

Source: RobotSim commit 70a84cd094a8d686864a26a260d2d17d1d65ea8e,
scripts/research/issue80_m1_first_motion.py. Runtime FSM decoder is binary-hash
bound and intentionally differs from stale installed message constants.
No ROS, simulator stepping, motion publishing or vendor modifications.
"""
from __future__ import annotations
import math,hashlib
from pathlib import Path
from enum import IntEnum
from typing import Mapping,Any

PHYSICAL_STOP_WINDOW_S = 2.0

PHYSICAL_STOP_ROLLING_STEP_S = 0.5

PHYSICAL_STOP_MIN_WINDOW_S = 1.90

PHYSICAL_STOP_MIN_SAMPLES = 20

PHYSICAL_STOP_MAX_SAMPLE_GAP_S = 0.15

PHYSICAL_STOP_MAX_STATE_AGE_S = 0.5

PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S = 0.005

PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S = 0.005

PHYSICAL_STOP_MAX_PLANAR_DRIFT_M = 0.005

PHYSICAL_STOP_MAX_PLANAR_PATH_M = 0.010

PHYSICAL_STOP_MAX_YAW_DRIFT_RAD = math.radians(2.0)

PHYSICAL_STOP_MAX_YAW_RATE_RAD_S = math.radians(1.0)

PHYSICAL_STOP_MAX_HEIGHT_RANGE_M = 0.005

PHYSICAL_STOP_MAX_TILT_RAD = math.radians(15.0)

PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD = math.radians(2.0)

PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S = 0.05

PHYSICAL_STOP_MAX_JOINT_RANGE_RAD = 0.02

PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD = 0.02

PHYSICAL_STOP_TARGET_STABILITY_WINDOW_S = 1.0

PHYSICAL_STOP_TARGET_STABILITY_RANGE_RAD = 0.01

MODE_STAND = "STAND_DEFAULT"

class McFsmStateCode(IntEnum):
    """Wire codes verified in the pinned MC state library, not stale SDK constants."""

    UNKNOWN = 0
    PREPARE = 1
    STARTING = 2
    FINISH = 3
    STABLE = 4
    MOVING = 5
    SAFE = 6
    SPECIAL = 7
    TEST = 8

MC_FSM_RUNTIME_LIBRARY_SHA256 = {
    "libmc_manager_state.so":
        "554e3e58752f3c640fa0074d7121663ebeda33037acd73a8d9234e94174c33aa",
    "libmc_module_main_publisher.so":
        "c159d55d6cf1c3217dc9fb4f655c25125b33204ee36432d98e5cc180653e72b9",
}

MC_FSM_ACTIVE_STATES = frozenset((
    int(McFsmStateCode.STARTING),
    int(McFsmStateCode.STABLE),
    int(McFsmStateCode.MOVING),
))

MC_FSM_STABLE = int(McFsmStateCode.STABLE)

MC_FSM_SAFE = int(McFsmStateCode.SAFE)

def verify_mc_fsm_runtime_contract(library_dir: str | Path) -> dict[str, str]:
    """Fail closed before motion if the binary-backed decoder no longer applies."""
    actual = {
        name: hashlib.sha256((Path(library_dir) / name).read_bytes()).hexdigest()
        for name in MC_FSM_RUNTIME_LIBRARY_SHA256
    }
    if actual != MC_FSM_RUNTIME_LIBRARY_SHA256:
        raise ValueError("MC FSM binary contract mismatch; motion prohibited")
    return actual

def _unwrap_yaw(values: list[float]) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        delta = (value - result[-1] + math.pi) % (2.0 * math.pi) - math.pi
        result.append(result[-1] + delta)
    return result

def _least_squares_slope(times: list[float], values: list[float]) -> float | None:
    if len(times) != len(values) or len(times) < 2:
        return None
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    denominator = sum((value - mean_t) ** 2 for value in times)
    if denominator <= 0.0:
        return None
    return sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denominator

def _physical_window_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one time-aligned window without treating mode readback as physics."""
    if len(rows) < 2:
        raise ValueError("physical-state window requires at least two samples")
    ordered = sorted(rows, key=lambda row: float(row["sample_monotonic_s"]))
    times = [float(row["sample_monotonic_s"]) for row in ordered]
    if not all(math.isfinite(value) for value in times):
        raise ValueError("physical-state window has invalid monotonic timestamps")
    poses = [row.get("pose") for row in ordered]
    required_pose_fields = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")
    if any(not isinstance(pose, Mapping) for pose in poses):
        raise ValueError("physical-state window has no complete odometry pose")
    numeric_poses: list[dict[str, float]] = []
    for pose in poses:
        parsed = {name: float(pose[name]) for name in required_pose_fields}
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError("physical-state window contains nonfinite odometry")
        numeric_poses.append(parsed)
    x0, y0 = numeric_poses[0]["x"], numeric_poses[0]["y"]
    planar_excursions = [math.hypot(pose["x"] - x0, pose["y"] - y0) for pose in numeric_poses]
    planar_path = sum(
        math.hypot(later["x"] - earlier["x"], later["y"] - earlier["y"])
        for earlier, later in zip(numeric_poses, numeric_poses[1:])
    )
    yaws = _unwrap_yaw([pose["yaw"] for pose in numeric_poses])
    yaw_rate = _least_squares_slope(times, yaws)

    joint_samples: dict[str, list[tuple[float, float, int]]] = {}
    command_samples: dict[str, list[float]] = {}
    for row in ordered:
        joints = row.get("leg_states")
        if not isinstance(joints, list) or not joints:
            raise ValueError("physical-state window is missing measured leg joints")
        for joint in joints:
            name = str(joint["name"])
            position, velocity = float(joint["position"]), float(joint["velocity"])
            error_code = int(joint.get("error_code", 0))
            if not math.isfinite(position) or not math.isfinite(velocity):
                raise ValueError("physical-state window contains nonfinite joint state")
            joint_samples.setdefault(name, []).append((position, velocity, error_code))
        command = row.get("mc_leg_command")
        command_joints = command.get("joints") if isinstance(command, Mapping) else None
        if not isinstance(command_joints, list) or not command_joints:
            raise ValueError("physical-state window is missing MC leg-command targets")
        for joint in command_joints:
            name = str(joint["name"])
            position = float(joint["position"])
            if not math.isfinite(position):
                raise ValueError("physical-state window contains nonfinite MC leg target")
            command_samples.setdefault(name, []).append(position)
    expected_names = set(joint_samples)
    if not expected_names or any(len(values) != len(ordered) for values in joint_samples.values()):
        raise ValueError("measured leg-joint coverage changes inside stop window")
    if set(command_samples) != expected_names or any(
        len(values) != len(ordered) for values in command_samples.values()
    ):
        raise ValueError("MC leg-command coverage does not match measured leg-joint coverage")

    ages: list[float] = []
    age_coverage = True
    for row in ordered:
        state_ages = row.get("state_age_s")
        if not isinstance(state_ages, Mapping):
            age_coverage = False
            continue
        for key in ("odom", "imu", "common", "leg_state", "leg_command", "get_mode"):
            value = state_ages.get(key)
            if value is None:
                age_coverage = False
                continue
            age = float(value)
            if not math.isfinite(age) or age < 0.0:
                age_coverage = False
            else:
                ages.append(age)

    frame_pairs = {
        (str(pose.get("frame_id", "")), str(pose.get("child_frame_id", "")))
        for pose in poses if isinstance(pose, Mapping)
    }
    frame_pair_available = bool(frame_pairs) and all(all(pair) for pair in frame_pairs)
    errors = [
        str(row.get("safety_fault")) for row in ordered
        if row.get("safety_fault") or row.get("safety_check") == "FAIL"
    ]
    return {
        "window_start_monotonic_s": times[0],
        "window_end_monotonic_s": times[-1],
        "duration_s": times[-1] - times[0],
        "sample_count": len(ordered),
        "max_sample_gap_s": max(later - earlier for earlier, later in zip(times, times[1:])),
        "max_state_age_s": max(ages) if ages else None,
        "state_age_coverage_complete": age_coverage,
        "frame_pairs": [list(pair) for pair in sorted(frame_pairs)],
        "frame_pair_available": frame_pair_available,
        "frame_pair_consistent": len(frame_pairs) == 1 if frame_pair_available else None,
        "max_planar_speed_m_s": max(math.hypot(pose["vx"], pose["vy"]) for pose in numeric_poses),
        "max_abs_vertical_speed_m_s": max(abs(pose["vz"]) for pose in numeric_poses),
        "planar_endpoint_drift_m": math.hypot(
            numeric_poses[-1]["x"] - x0, numeric_poses[-1]["y"] - y0
        ),
        "max_planar_excursion_m": max(planar_excursions),
        "planar_path_length_m": planar_path,
        "yaw_drift_rad": yaws[-1] - yaws[0],
        "max_abs_yaw_excursion_rad": max(abs(value - yaws[0]) for value in yaws),
        "yaw_rate_fit_rad_s": yaw_rate,
        "z_range_m": max(pose["z"] for pose in numeric_poses) - min(pose["z"] for pose in numeric_poses),
        "max_abs_tilt_rad": max(math.hypot(pose["roll"], pose["pitch"]) for pose in numeric_poses),
        "roll_range_rad": max(pose["roll"] for pose in numeric_poses) - min(pose["roll"] for pose in numeric_poses),
        "pitch_range_rad": max(pose["pitch"] for pose in numeric_poses) - min(pose["pitch"] for pose in numeric_poses),
        "max_abs_joint_velocity_rad_s": max(
            abs(velocity) for values in joint_samples.values() for _, velocity, _ in values
        ),
        "max_joint_position_range_rad": max(
            max(position for position, _, _ in values) - min(position for position, _, _ in values)
            for values in joint_samples.values()
        ),
        "max_joint_command_range_rad": max(
            max(values) - min(values) for values in command_samples.values()
        ),
        "joint_error_codes": sorted({
            error for values in joint_samples.values() for _, _, error in values
        }),
        "safety_faults": errors,
        "mode_desc_values": sorted({str(row.get("get_mode_desc")) for row in ordered}),
        "common_mode_desc_values": sorted({str(row.get("common_mode_desc")) for row in ordered}),
        "fsm_state_values": sorted({row.get("fsm_state") for row in ordered}, key=str),
        "body_status_values": sorted({row.get("body_status") for row in ordered}, key=str),
        "get_mode_header_codes": sorted({row.get("get_mode_header_code") for row in ordered}, key=str),
        "get_mode_status_values": sorted({row.get("get_mode_status_value") for row in ordered}, key=str),
        "common_mode_status_values": sorted({row.get("common_mode_status_value") for row in ordered}, key=str),
        "readback_faults": sorted({str(row.get("readback_fault")) for row in ordered if row.get("readback_fault")}),
    }

def _physical_stop_violations(
    metrics: Mapping[str, Any], *, baseline_body_status: Any = None,
) -> list[str]:
    violations: list[str] = []
    if metrics.get("duration_s", 0.0) < PHYSICAL_STOP_MIN_WINDOW_S:
        violations.append("stable_window_too_short")
    if metrics.get("sample_count", 0) < PHYSICAL_STOP_MIN_SAMPLES:
        violations.append("too_few_state_samples")
    if metrics.get("max_sample_gap_s", math.inf) > PHYSICAL_STOP_MAX_SAMPLE_GAP_S:
        violations.append("state_sample_gap_exceeded")
    if not metrics.get("state_age_coverage_complete") or metrics.get("max_state_age_s") is None:
        violations.append("state_freshness_unverified")
    elif metrics["max_state_age_s"] > PHYSICAL_STOP_MAX_STATE_AGE_S:
        violations.append("state_stale")
    if metrics.get("safety_faults"):
        violations.append("existing_safety_gate_fault")
    if metrics.get("readback_faults"):
        violations.append("readback_fault")
    if metrics.get("mode_desc_values") != [MODE_STAND]:
        violations.append("get_mc_action_not_stand_throughout_window")
    if metrics.get("common_mode_desc_values") != [MODE_STAND]:
        violations.append("common_state_not_stand_throughout_window")
    if metrics.get("fsm_state_values") != [MC_FSM_STABLE]:
        violations.append("raw_fsm_not_stable_throughout_window")
    if baseline_body_status is not None and metrics.get("body_status_values") != [baseline_body_status]:
        violations.append("raw_body_status_changed_from_standing_baseline")
    if metrics.get("get_mode_header_codes") != [0]:
        violations.append("get_mc_action_readback_failed")
    if metrics.get("get_mode_status_values") != [100] or metrics.get("common_mode_status_values") != [100]:
        violations.append("mode_status_not_idle_throughout_window")
    if metrics.get("joint_error_codes") != [0]:
        violations.append("joint_error_code_present")
    limits = (
        ("max_planar_speed_m_s", PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S, "planar_speed_exceeded"),
        ("max_abs_vertical_speed_m_s", PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S, "vertical_speed_exceeded"),
        ("planar_endpoint_drift_m", PHYSICAL_STOP_MAX_PLANAR_DRIFT_M, "planar_endpoint_drift_exceeded"),
        ("max_planar_excursion_m", PHYSICAL_STOP_MAX_PLANAR_DRIFT_M, "planar_excursion_exceeded"),
        ("planar_path_length_m", PHYSICAL_STOP_MAX_PLANAR_PATH_M, "planar_path_exceeded"),
        ("max_abs_yaw_excursion_rad", PHYSICAL_STOP_MAX_YAW_DRIFT_RAD, "heading_excursion_exceeded"),
        ("yaw_rate_fit_rad_s", PHYSICAL_STOP_MAX_YAW_RATE_RAD_S, "heading_rate_exceeded"),
        ("z_range_m", PHYSICAL_STOP_MAX_HEIGHT_RANGE_M, "root_height_variation_exceeded"),
        ("max_abs_tilt_rad", PHYSICAL_STOP_MAX_TILT_RAD, "existing_tilt_gate_exceeded"),
        ("roll_range_rad", PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD, "roll_variation_exceeded"),
        ("pitch_range_rad", PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD, "pitch_variation_exceeded"),
        ("max_abs_joint_velocity_rad_s", PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S, "joint_velocity_not_settled"),
        ("max_joint_position_range_rad", PHYSICAL_STOP_MAX_JOINT_RANGE_RAD, "joint_position_not_settled"),
        ("max_joint_command_range_rad", PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD, "joint_target_not_settled"),
    )
    for key, limit, fault in limits:
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)) or abs(float(value)) > limit:
            violations.append(fault)
    if metrics.get("frame_pair_consistent") is False:
        violations.append("odometry_frame_pair_changed_or_missing")
    return violations

def evaluate_stop_window(rows, *, baseline_body_status):
    metrics=_physical_window_metrics(rows)
    violations=_physical_stop_violations(metrics,baseline_body_status=baseline_body_status)
    return {"pass":not violations,"metrics":metrics,"violations":violations,"contract_revision":"70a84cd094a8d686864a26a260d2d17d1d65ea8e"}
