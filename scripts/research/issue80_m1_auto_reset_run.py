#!/usr/bin/env python3
"""Bounded simulation-only Reset verification followed by the guarded Issue 80 M1 probe."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from issue80_m1_first_motion import (
    MAX_EXTRA_POST_STOP_OBSERVE_S,
    MODE_STAND,
    TURN_COMMAND_RAD_S, TURN_FORWARD_COMMAND_M_S, TURN_FORWARD_WARMUP_S,
    TURN_MIN_HEADING_CHANGE_RAD, TURN_PULSE_S,
    ZERO_STOP_MIN_MESSAGES,
    validate_b_post_locomotion_observation,
)
from issue80_m1_mode_logic import RAICOM_LOW_CRUISE_FORWARD_M_S
from issue80_m1_set_action_capture import native_foot_support, native_root_within_existing_safety_gate
from issue80_stand_baseline_analysis import analyze_standing_trace
from issue80_coordinated_pose_goal import (
    POSE_GOAL_APPROACH_WARMUP_S,
    POSE_GOAL_CURVE_FORWARD_M_S,
    POSE_GOAL_HEADING_STOP_ADVANCE_M,
    POSE_GOAL_WARMUP_FORWARD_M_S,
)
from issue80_route_motion_bridge import (
    CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
    CURVED_PATH_FORWARD_M_S,
    CURVED_PATH_MAX_CROSS_TRACK_M,
    CURVED_PATH_MAX_WINDOW_S,
    CURVED_PATH_MAX_WINDOWS,
    CURVED_PATH_POSITION_TOLERANCE_M,
    CURVED_PATH_RADIUS_M,
    CURVED_PATH_STOP_RESERVE_M,
    CURVED_PATH_SWEEP_RAD,
    WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S,
    WAYPOINT_B_MAX_WINDOWS,
    WAYPOINT_B_POSITION_TOLERANCE_M,
    WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M,
    WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
    WAYPOINT_B_CORRIDOR_MAX_FORWARD_M,
    WAYPOINT_B_MAX_CROSS_TRACK_M,
    WAYPOINT_B_WINDOW_MAX_S,
    FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
    waypoint_b_arrival_trigger_radius_m,
)
from issue80_waypoint_route import (
    build_local01_short_scene_route,
    build_waypoint_docking_route_from_poses,
    load_waypoint_docking_pose_input,
    validate_waypoint_docking_scene_clearance,
)
from issue80_local01_scene_route import (
    LOCAL01_COMMIT,
    build_runtime_raw_odom_short_route_config,
    build_runtime_raw_odom_table_approach_mission,
    build_runtime_raw_odom_aligned_start_table_approach_mission,
    load_local01_scene_inputs,
    prepare_local01_scene_runtime_bundle,
    validate_short_route_static_scene,
    validate_runtime_raw_odom_table_approach_mission,
    validate_runtime_raw_odom_aligned_start_table_approach_mission,
)
from issue80_raicom_mjdata_capture import (
    CAPTURE_SOURCE,
    CAPTURE_SUMMARY_NAME,
    CAPTURE_TRACE_NAME,
    DEFAULT_INTERVAL_STEPS,
    MjDataCaptureError,
    prepare_mjdata_capture_overlay,
    run_mjdata_capture_smoke,
    summarize_mjdata_capture,
    write_new_json as write_new_capture_json,
)


REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = REPO / "simulation/mujoco/raicom2026"
X11_HELPER = Path(__file__).with_name("issue80_mujoco_x11_reset.py")
X11_RECORDER = Path(__file__).with_name("issue80_mujoco_x11_record.py")
MJDATA_CAPTURE = Path(__file__).with_name("issue80_raicom_mjdata_capture.py")
MJDATA_CAPTURE_C = Path(__file__).with_name("issue80_mjdata_capture.c")
WINDOWS_FOCUS_HELPER = Path(__file__).with_name("issue80_mujoco_windows_focus.ps1")
ANALYZER = Path(__file__).with_name("issue80_m1_reset_analyze.py")
M1_RUNNER = Path(__file__).with_name("issue80_m1_locomotion_probe.py")
LIDAR_LOCALIZER = Path(__file__).with_name("issue80_lidar_localization.py")
LIDAR_SCAN_MATH = Path(__file__).with_name("issue80_lidar_scan_math.py")
VENDOR_DIRECT_TRUTH_ROUTE = Path(__file__).with_name("issue80_vendor_direct_truth_route.py")
POSE_GOAL_CONTROLLER = Path(__file__).with_name("issue80_coordinated_pose_goal.py")
M1_MODE_LOGIC = Path(__file__).with_name("issue80_m1_mode_logic.py")
M1_FIRST_MOTION_LOGIC = Path(__file__).with_name("issue80_m1_first_motion.py")
WAYPOINT_ROUTE = Path(__file__).with_name("issue80_waypoint_route.py")
LOCAL01_SCENE_ROUTE = Path(__file__).with_name("issue80_local01_scene_route.py")
ROUTE_MOTION_BRIDGE = Path(__file__).with_name("issue80_route_motion_bridge.py")
SET_ACTION_CAPTURE = Path(__file__).with_name("issue80_m1_set_action_capture.py")
MC_RL_DEBUG_CAPTURE = Path(__file__).with_name("issue80_mc_rl_debug_capture.py")
SIM_LAUNCH = Path(__file__).with_name("issue80_launch_sim.sh")
MC_LAUNCH = Path(__file__).with_name("issue80_launch_mc.sh")
IMAGE = "robotsim:humble-raicom-runtime-20261019"
IMAGE_DIGEST = "sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31"
ROUTE_SCENE_RELATIVE = Path("sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml")
ROUTE_X2_RELATIVE = Path("sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml")
ROUTE_SCENE_SHA256 = "7fceb3e1357be29b72344db2f571d7e5655a7d1b1db4ea2571556aee28bb1b63"
ROUTE_X2_SHA256 = "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3"
ROUTE_QPOS0_FOOTPRINT_BOUND_M = 0.41565748847422734
ROUTE_LOCAL_CLEAR_RADIUS_M = 2.0
RESET_OBSERVER_DURATION_S = 8.0
RESET_OBSERVER_COMPLETION_MARGIN_S = 3.0
# Covers the existing bounded Reset/upright, JD, and SD handoff before capture.
RESET_STAND_HANDOFF_MAX_S = 16.0
RESET_STARTUP_OBSERVATION_TIMEOUT_S = 4.0
POINT_GOAL_FINAL_TOLERANCE_M = 0.05
POINT_GOAL_FORWARD_COMMAND_M_S = 0.14
POINT_GOAL_DISTANCE_M = 0.30
POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S = 5.0
POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S = 7.0
POINT_GOAL_MAX_HEADING_CHANGE_RAD = 0.20
POINT_GOAL_FINAL_YAW_TOLERANCE_RAD = 0.17
POINT_GOAL_HEADING_STOP_ADVANCE_M = POSE_GOAL_HEADING_STOP_ADVANCE_M
POINT_GOAL_HEADING_PREDICTED_STOP_TOLERANCE_M = 0.005
POINT_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD = 0.05
POINT_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD = 0.115
POINT_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD = 0.11341906640050358
POINT_GOAL_TERMINAL_SETPOINT_TOLERANCE_RAD = 0.01
POINT_GOAL_PRESTOP_YAW_MARGIN_RAD = 0.03
POINT_GOAL_POST_STOP_DRIFT_OBSERVATIONS_RAD = (
    0.09861381357287352,
    0.11341906640050358,
    0.16952762228006262,
    0.1886899134656192,
)
POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M = 0.00347654
POINT_GOAL_ENDPOINT_RESPONSE_TIME_S = 0.09162162
POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S = 0.075
TABLE_APPROACH_CHILD_TIMEOUT_S = 410.0
TABLE_APPROACH_MIN_PARENT_RUNTIME_S = 500.0
PREARMED_STANDING_MIN_PARENT_RUNTIME_S = 90.0
STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD = 0.020
# The already-established root-motion guard is 0.20 m/s.  The pre-armed
# handoff uses it only to reject an already unsafe start; it is not a standing
# or physical-stop acceptance threshold.
PREARMED_NOMINAL_MAX_ROOT_SPEED_M_S = 0.20
POINT_GOAL_ANGULAR_TAPER_START_ERROR_M = 0.25
POINT_GOAL_MAX_LATERAL_ERROR_M = 0.05
POINT_GOAL_MAX_OVERSHOOT_M = 0.05
TOPIC_PROBES = (
    "/aima/hal/odom/state",
    "/aima/hal/imu/torso/state",
    "/aima/hal/joint/leg/state",
)
FOUNDATIONWALK_CONFIG_RELATIVE = Path(
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl/foundationwalk_config.yaml"
)
FOUNDATIONWALK_CONFIG_CONTAINER_PATH = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/"
    "lx2501_3_t2d5/rl/foundationwalk_config.yaml"
)
MC_SIM_CONFIG_DUMP_CONTAINER_PATH = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/"
    "lx2501_3_t2d5/mc_sim.yaml.dump"
)
MC_SIM_GENERATED_DIRECTORY = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim"
)
MC_SIM_CONFIG_RELATIVE = Path(
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"
)
MC_SIM_CONFIG_DUMP_RELATIVE = Path(
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump"
)
MC_SIM_GENERATED_SEED_FILENAMES = (
    "action_setting.yaml",
    "action_ruler.yaml",
    "action_state.yaml",
)
RUN38_FOUNDATIONWALK_CONFIG_SHA256 = "b3081e9bca2035d4a986657bdc03b79ca0ed5d4e852c7972d86816f569acc2e9"
RUN38_FOUNDATIONWALK_ONNX_SHA256 = "b8cca5e9048c3c056e94c56ad06f171f9d5a76149aebd893ad524b365507cc9b"
RUN38_FOUNDATIONWALK_RUNNER_SHA256 = "0fa34f81a7ec87179309cd78859f5469c5e84ec595588bf7cc7ff935c865556b"
RUN38_ACTION_SETTING_SHA256 = "60e4b37a2d185515efb63192a67f9cb425558da3f4caabd432aa62bb23f05479"
RUN38_ROBOT_MODEL_SHA256 = "614b0644976204e4f458f2c3a9d6f92f9a463730ddcaec5bc1cbdd9748727670"
RUN38_MC_SIM_CONFIG_SHA256 = "999d75f0caed5f20479c670fb95b624910c66722b47d37b60217717a3fe34b60"
DEFAULT_MC_RL_DEBUG_OVERLAY_SOURCE = REPO / (
    "reviews/issue-80/autonomous-walking-38-step-boundary-0112-20261021T194921Z"
)
RUN38_DIAGNOSTIC_KNEE_ACTION_INDICES = (9, 10)
RUN38_BASE_KNEE_ACTION_SCALE = 0.25
RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE = 0.19
RUN41_BASE_MIN_RUN_VELX = 0.5
RUN41_DIAGNOSTIC_MIN_RUN_VELX = 0.1


def standing_baseline_observation_passes(
    observation: Mapping[str, Any] | None,
    *,
    minimum_duration_s: float = RESET_OBSERVER_DURATION_S,
) -> bool:
    """Require the full duration after a positive STAND GetMcAction readback."""
    if not isinstance(observation, Mapping):
        return False
    initial_readback = observation.get("initial_stand_mode_readback")
    final_readback = observation.get("final_stand_mode_readback")

    def confirmed(readback: Any) -> bool:
        return (
            isinstance(readback, Mapping)
            and readback.get("transport_state") == "RESPONSE_RECEIVED"
            and readback.get("action_desc") == "STAND_DEFAULT"
            and readback.get("action_status_value") == 100
        )

    duration = observation.get("observed_duration_s")
    expected_body_status = observation.get("expected_body_status")
    common_sample_count = observation.get("common_state_sample_count")
    raw_fsm_values = observation.get("raw_fsm_state_values")
    return (
        observation.get("anchor") in {
            "STAND_DEFAULT_GETMC_READBACK",
            "STAND_DEFAULT_GETMC_AND_BODY_STAND",
        }
        and observation.get("observer_status") == "STAND_DEFAULT_BASELINE_COMPLETE"
        and observation.get("complete") is True
        and observation.get("first_unsafe_monotonic_ns") is None
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) >= float(minimum_duration_s)
        and confirmed(initial_readback)
        and confirmed(final_readback)
        and observation.get("common_mode_desc_values") == [MODE_STAND]
        and observation.get("common_mode_status_values") == [100]
        and isinstance(raw_fsm_values, list)
        and bool(raw_fsm_values)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_fsm_values)
        and expected_body_status is not None
        and observation.get("raw_body_status_values") == [expected_body_status]
        and isinstance(common_sample_count, int)
        and not isinstance(common_sample_count, bool)
        and common_sample_count >= 20
    )


def standing_warmup_composite_result(
    observation: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require an 8 s capture and a passing final 2 s physical+target window."""
    final_window = analysis.get("final_2s_window") if isinstance(analysis, Mapping) else None
    physical = final_window.get("physical_state_gate") if isinstance(final_window, Mapping) else None
    targets = final_window.get("controller_target_stability_gate") if isinstance(final_window, Mapping) else None
    capture_pass = standing_baseline_observation_passes(observation)
    physical_pass = isinstance(physical, Mapping) and physical.get("status") == "PASS"
    target_pass = isinstance(targets, Mapping) and targets.get("status") == "PASS"
    minimum_duration_pass = (
        isinstance(analysis, Mapping) and analysis.get("minimum_duration_pass") is True
    )
    return {
        "status": "PASS" if capture_pass and minimum_duration_pass and physical_pass and target_pass else "FAIL",
        "classification": "SIMULATION_ONLY_STAND_WARMUP_COMPOSITE",
        "capture_completeness": "PASS" if capture_pass else "FAIL",
        "minimum_eight_second_observation": "PASS" if minimum_duration_pass else "FAIL",
        "final_physical_state_gate": "PASS" if physical_pass else "FAIL",
        "final_controller_target_stability_gate": "PASS" if target_pass else "FAIL",
        "final_window_start_monotonic_s": final_window.get("start_monotonic_s") if isinstance(final_window, Mapping) else None,
        "final_window_end_monotonic_s": final_window.get("end_monotonic_s") if isinstance(final_window, Mapping) else None,
        "final_max_joint_target_range_rad": (
            targets.get("max_target_range_rad") if isinstance(targets, Mapping) else None
        ),
        "original_stop_thresholds_preserved": True,
    }


def validate_stand_convergence_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Validate pre-motion standing and the separate extended-B observation profile."""
    require_warmup = bool(getattr(args, "require_stand_warmup_composite", False))
    extended_b = bool(getattr(args, "run65_waypoint_b_extended_diagnostic", False))
    configured_route = bool(
        getattr(args, "route_config", None)
        or getattr(args, "local01_scene_world_poses", None)
    )
    post_observation = getattr(args, "post_locomotion_convergence_observation_s", None)
    prearmed_standing_only = bool(
        getattr(args, "mc_prearm_before_simulator", False)
        and getattr(args, "static_nominal_start_initialization", False)
        and getattr(args, "maintainer_authorized_standing_takeover", False)
        and not getattr(args, "maintainer_authorized_first_motion", False)
    )
    direct_standing_only = bool(
        getattr(args, "direct_stand_from_passive", False)
        and getattr(args, "maintainer_authorized_standing_takeover", False)
        and not getattr(args, "maintainer_authorized_first_motion", False)
    )
    if require_warmup and (
        not (
            (getattr(args, "maintainer_authorized_first_motion", False)
             and (extended_b or configured_route))
            or ((prearmed_standing_only or direct_standing_only) and configured_route)
        )
    ):
        raise ValueError(
            "composite stand warmup gate requires authorized extended-B or configured-route motion, "
            "or prearmed standing-only qualification"
        )
    if post_observation is None:
        if require_warmup and not configured_route:
            raise ValueError("composite stand warmup diagnostic requires the explicit post-locomotion observation horizon")
        return {
            "status": (
                "SIMULATION_ONLY_STAND_WARMUP_COMPOSITE"
                if require_warmup else "STANDARD_PROFILE"
            ),
            "extra_stop_observation_s": getattr(args, "extra_stop_observation", 0.0),
            "existing_composite_stop_gate_unchanged": require_warmup,
        }
    if not extended_b or not require_warmup:
        raise ValueError("post-locomotion convergence observation requires extended-B and the composite warmup gate")
    if (
        not math.isfinite(float(post_observation))
        or abs(float(post_observation) - MAX_EXTRA_POST_STOP_OBSERVE_S) > 1e-9
        or float(args.extra_stop_observation) != 0.0
    ):
        raise ValueError(
            f"post-locomotion convergence observation is fixed at {MAX_EXTRA_POST_STOP_OBSERVE_S:g} s; "
            "do not combine it with generic --extra-stop-observation"
        )
    args.extra_stop_observation = float(post_observation)
    return {
        "status": "SIMULATION_ONLY_POST_LOCOMOTION_STAND_CONVERGENCE_OBSERVATION",
        "extra_stop_observation_s": float(post_observation),
        "maximum_extra_observation_s": MAX_EXTRA_POST_STOP_OBSERVE_S,
        "existing_composite_stop_gate_unchanged": True,
    }


def validate_configured_route_motion_profile(args: argparse.Namespace) -> None:
    """Keep configured routes on the existing bounded transit and stop profile."""
    if not (getattr(args, "route_config", None) or getattr(args, "local01_scene_world_poses", None)):
        return
    low_cruise = bool(getattr(args, "raicom_low_cruise_profile", False))
    vendor_example = bool(
        getattr(args, "raicom_vendor_example_forward_baseline", False)
        or getattr(args, "raicom_vendor_table_approach", False)
    )
    if vendor_example and low_cruise:
        raise ValueError("vendor-example baseline cannot combine with the RAICOM low-cruise profile")
    expected_forward = (
        0.2 if vendor_example else RAICOM_LOW_CRUISE_FORWARD_M_S if low_cruise else 0.12
    )
    expected_pulse = 5.0 if vendor_example else 1.0
    if (
        abs(float(args.forward_speed) - expected_forward) > 1e-9
        or abs(float(args.pulse_duration) - expected_pulse) > 1e-9
        or abs(float(args.extra_stop_observation) - 7.0) > 1e-9
        or float(args.ramp_duration_s) != 0.0
        or args.ramp_start_speed is not None
        or args.startup_speed is not None
        or args.startup_transition_displacement is not None
        or float(args.startup_transition_timeout_s) != 0.0
        or float(getattr(args, "deceleration_duration_s", 0.0)) != 0.0
        or float(getattr(args, "deceleration_end_speed_m_s", 0.0)) != 0.0
        or bool(getattr(args, "capture_mc_rl_debug", False))
        or getattr(args, "route_turn_evidence", None) is not None
    ):
        raise ValueError(
            f"configured route requires {expected_forward:g} m/s for {expected_pulse:g} s, "
            "7 s existing post-STAND observation, and no ramp/debug/turn overlays"
        )


def assess_static_nominal_start_sample(
    sample: Mapping[str, Any], expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess vendor-applied nominal joints plus static root geometry.

    This is an observation-only pre-JD gate.  It checks the generated scene's
    source-MC nominal target against native capture; it neither steps MuJoCo
    nor writes a state vector.
    """
    try:
        qpos = [float(value) for value in sample["qpos"]]
        target = [float(value) for value in expected["qpos"]]
    except (KeyError, TypeError, ValueError):
        return {"eligible": False, "failure_reason": "missing_native_or_expected_qpos"}
    if len(qpos) != len(target) or len(qpos) < 19:
        return {"eligible": False, "failure_reason": "native_qpos_dimension_mismatch"}
    residuals = [abs(qpos[index] - target[index]) for index in range(7, 19)]
    max_leg_residual = max(residuals, default=math.inf)
    support = native_foot_support(sample)
    root_safe = native_root_within_existing_safety_gate(support)
    eligible = (
        math.isfinite(max_leg_residual)
        and max_leg_residual <= STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD
        and support.get("safe_to_begin_ros_upright_gate") is True
        and root_safe
    )
    return {
        "eligible": eligible,
        "maximum_leg_nominal_residual_rad": max_leg_residual,
        "maximum_leg_nominal_residual_limit_rad": STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD,
        "native_root_within_existing_safety_gate": root_safe,
        "support": support,
        "failure_reason": None if eligible else "static_nominal_start_native_gate_not_met",
    }


def assess_prearmed_nominal_handoff_sample(
    sample: Mapping[str, Any], expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate the one early JD request while MC is already service-ready.

    This differs deliberately from ``assess_static_nominal_start_sample``.
    The latter proves *settled* bilateral support before an action request;
    that is impossible here because the vendor simulator begins stepping before
    it has received a controller command.  The pre-armed gate accepts only the
    first source-nominal, bilateral-contact sample while root geometry and both
    linear-speed components remain within the existing 0.20 m/s guard.  It is
    an eligibility check for one JD request, never proof of standing.
    """
    try:
        qpos = [float(value) for value in sample["qpos"]]
        target = [float(value) for value in expected["qpos"]]
    except (KeyError, TypeError, ValueError):
        return {"eligible": False, "failure_reason": "missing_native_or_expected_qpos"}
    if len(qpos) != len(target) or len(qpos) < 19:
        return {"eligible": False, "failure_reason": "native_qpos_dimension_mismatch"}

    residuals = [abs(qpos[index] - target[index]) for index in range(7, 19)]
    max_leg_residual = max(residuals, default=math.inf)
    support = native_foot_support(sample)
    root_safe = native_root_within_existing_safety_gate(support)
    vertical_speed = support.get("pelvis_vertical_speed_world_m_s")
    contact_count = sample.get("captured_foot_floor_contact_count")
    try:
        vertical_safe = (
            math.isfinite(float(vertical_speed))
            and abs(float(vertical_speed)) <= PREARMED_NOMINAL_MAX_ROOT_SPEED_M_S
        )
    except (TypeError, ValueError):
        vertical_safe = False
    try:
        bilateral_contact_geometry = int(contact_count) >= 2
    except (TypeError, ValueError):
        bilateral_contact_geometry = False
    eligible = (
        math.isfinite(max_leg_residual)
        and max_leg_residual <= STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD
        and root_safe
        and vertical_safe
        and bilateral_contact_geometry
    )
    return {
        "eligible": eligible,
        "maximum_leg_nominal_residual_rad": max_leg_residual,
        "maximum_leg_nominal_residual_limit_rad": STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD,
        "native_root_within_existing_safety_gate": root_safe,
        "vertical_speed_within_existing_root_speed_guard": vertical_safe,
        "maximum_abs_root_speed_guard_m_s": PREARMED_NOMINAL_MAX_ROOT_SPEED_M_S,
        "bilateral_foot_contact_geometry_observed": bilateral_contact_geometry,
        "captured_foot_floor_contact_count": contact_count,
        "support": support,
        "classification": "PREARMED_NOMINAL_HANDOFF_ELIGIBILITY_NOT_STANDING",
        "failure_reason": None if eligible else "prearmed_nominal_handoff_native_gate_not_met",
    }


def stage_mc_generated_config_writeback(evidence: Path) -> tuple[Path, dict[str, Any]]:
    """Stage MC's known generated config outputs outside the vendor runtime.

    The mounted runtime stays read-only.  MC receives one evidence-local dump
    bind mount and a tmpfs-backed ``sim/`` directory populated from verified
    vendor inputs.  This mirrors the successful read-only MC pre-arm probe.
    """
    config = RUNTIME / MC_SIM_CONFIG_RELATIVE
    original_dump = RUNTIME / MC_SIM_CONFIG_DUMP_RELATIVE
    for path in (config, original_dump):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required MC config input is missing or unsafe: {path.name}")
    staged_directory = evidence / "mc_runtime_writeback"
    staged_directory.mkdir(mode=0o755)
    staged_dump = staged_directory / "mc_sim.yaml.dump"
    with staged_dump.open("xb") as stream:
        stream.write(original_dump.read_bytes())
    seed_directory = staged_directory / "sim_seed"
    seed_directory.mkdir(mode=0o755)
    seed_hashes: dict[str, str] = {}
    for name in MC_SIM_GENERATED_SEED_FILENAMES:
        source = config.parent / "sim" / name
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"required MC generated-config seed is missing or unsafe: {name}")
        payload = source.read_bytes()
        with (seed_directory / name).open("xb") as stream:
            stream.write(payload)
        seed_hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "robotsim.issue80.mc-generated-config-writeback.v1",
        "classification": "NARROW_EPHEMERAL_MC_GENERATED_CONFIG_ONLY",
        "runtime_readonly": True,
        "vendor_runtime_tree_modified": False,
        "vendor_source_config": str(config),
        "vendor_source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "vendor_source_dump": str(original_dump),
        "vendor_source_dump_sha256": hashlib.sha256(original_dump.read_bytes()).hexdigest(),
        "staged_dump": str(staged_dump),
        "staged_dump_initial_sha256": hashlib.sha256(staged_dump.read_bytes()).hexdigest(),
        "container_dump_target": MC_SIM_CONFIG_DUMP_CONTAINER_PATH,
        "generated_sim_tmpfs_target": MC_SIM_GENERATED_DIRECTORY,
        "seed_hashes": seed_hashes,
        "permitted_write_scope": [
            MC_SIM_CONFIG_DUMP_CONTAINER_PATH,
            MC_SIM_GENERATED_DIRECTORY,
        ],
    }
    with (evidence / "mc_runtime_writeback_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return staged_dump, manifest


def validate_static_nominal_start_table_approach_clearance(
    static_plan: Mapping[str, Any], *, root_z_offset_m: float,
) -> dict[str, Any]:
    """Transfer the reviewed table-approach clearance bound to a Z-shifted start.

    Translating every robot point by ``d`` can reduce a point-to-environment
    Euclidean clearance lower bound by no more than ``abs(d)``.  This is a
    conservative static certificate for START -> WAYPOINT -> TABLE_APPROACH,
    never a dynamic navigation result or TABLE_DOCKING clearance claim.
    """
    if not math.isfinite(root_z_offset_m):
        raise ValueError("static nominal-start root Z offset is non-finite")
    segments = static_plan.get("segments")
    if not isinstance(segments, list):
        raise ValueError("table static plan has no segment list")
    required = {("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH")}
    records: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        key = (segment.get("from"), segment.get("to"))
        if key not in required or segment.get("posture") != "runtime_nominal":
            continue
        try:
            original = float(segment["tracking_inflated_aabb_lower_bound_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("required runtime-nominal route clearance is malformed") from exc
        if (
            not math.isfinite(original)
            or segment.get("status") != "PASS_STATIC_ONLY"
            or segment.get("active_environment_contacts")
            or segment.get("first_tracking_inflated_below_20mm") is not None
        ):
            raise ValueError("reviewed runtime-nominal route segment is not clearance-qualified")
        records.append({
            "from": key[0],
            "to": key[1],
            "original_tracking_inflated_lower_bound_m": original,
            "root_z_translation_magnitude": abs(root_z_offset_m),
            "conservative_transferred_lower_bound_m": max(0.0, original - abs(root_z_offset_m)),
        })
    if { (record["from"], record["to"]) for record in records } != required:
        raise ValueError("table static plan lacks both runtime-nominal approach segments")
    minimum = min(record["conservative_transferred_lower_bound_m"] for record in records)
    return {
        "schema": "robotsim.issue80.static-nominal-start-table-approach-clearance.v1",
        "status": "PASS_STATIC_NOMINAL_START_TABLE_APPROACH_CLEARANCE" if minimum >= 0.020 else "FAIL_STATIC_NOMINAL_START_TABLE_APPROACH_CLEARANCE",
        "segments": records,
        "minimum_conservative_transferred_lower_bound_m": minimum,
        "minimum_required_clearance_m": 0.020,
        "classification": "STATIC_GEOMETRY_TRANSFER_ONLY_NOT_DYNAMIC_NAVIGATION_OR_TABLE_DOCKING",
    }


def validate_fixed_window_open_segment_yaw_envelope(
    static_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the static plan that bounds the fixed-profile yaw-only phase.

    The per-window source acquisition is intentionally available only in the
    open START->WAYPOINT segment.  This prevents a plan generated with the
    legacy +/-0.075 rad envelope from silently qualifying a measured
    +/-0.10 rad open-segment yaw response.
    """
    try:
        assumptions = static_plan["clearance_assumptions"]
        observed = assumptions["yaw_envelope_by_segment_rad"]
        expected = {
            "START->WAYPOINT": FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
            "WAYPOINT->TABLE_APPROACH": 0.075,
        }
        actual = {key: float(observed[key]) for key in expected}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "fixed-window table route static plan lacks its per-segment yaw envelope"
        ) from exc
    if any(abs(actual[key] - value) > 1e-9 for key, value in expected.items()):
        raise ValueError(
            "fixed-window table route requires the measured open-segment yaw envelope "
            "and the reviewed table-facing yaw envelope"
        )
    return {
        "status": "PASS_FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE",
        "yaw_envelope_by_segment_rad": actual,
        "classification": "STATIC_GEOMETRY_ENVELOPE_ONLY_NOT_DYNAMIC_YAW_QUALIFICATION",
    }


class RunDeadline(TimeoutError):
    pass


def validate_reserved_evidence_directory(path: str | os.PathLike[str]) -> Path:
    """Require a canonical, path-governance-reserved run directory before runtime."""
    helper_path = str(REPO / "scripts/agent")
    if helper_path not in sys.path:
        sys.path.insert(0, helper_path)
    from path_governance import validate_output_path

    location = validate_output_path(
        path, require_exists=True, expected_kind="directory"
    )
    marker = location.native_path / ".robotsim-output-reservation.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("evidence directory has no regular output reservation marker")
    try:
        reservation = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence directory reservation marker is unreadable") from exc
    if reservation != {
        "schema_version": "robotsim.output-reservation.v1",
        "path": location.wsl_path,
    }:
        raise ValueError("evidence directory reservation marker does not match its path")
    return location.native_path


def build_runtime_bind_mount(runtime_root: Path, *, readonly: bool = False) -> str:
    """Build the existing runtime bind mount; capture-only callers may lock it read-only."""
    mount = f"type=bind,source={runtime_root},target=/workspace/raicom2026"
    return mount + (",readonly" if readonly else "")


def inspect_route_workcell_geometry(runtime_root: Path) -> dict[str, Any]:
    """Fail closed unless the pinned route scene is the verified open-floor workcell."""
    scene_path = runtime_root / ROUTE_SCENE_RELATIVE
    x2_path = runtime_root / ROUTE_X2_RELATIVE
    for name, path, expected in (
        ("scene", scene_path, ROUTE_SCENE_SHA256),
        ("x2_model", x2_path, ROUTE_X2_SHA256),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"route workcell {name} is missing or not a regular file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"route workcell {name} identity changed: expected {expected}, got {digest}")
    root = ET.parse(scene_path).getroot()
    includes = [item.get("file") for item in root.findall("include")]
    world = root.find("worldbody")
    if world is None or includes != ["x2.xml"]:
        raise ValueError("route scene no longer consists of the pinned X2 include and world body")
    direct_geoms = [item for item in world if item.tag == "geom"]
    direct_bodies = [item for item in world if item.tag == "body"]
    if (
        len(direct_geoms) != 1
        or direct_bodies
        or direct_geoms[0].get("name") != "floor"
        or direct_geoms[0].get("type") != "plane"
    ):
        raise ValueError("route scene has an unreviewed world collision geom or finite obstacle")
    floor = direct_geoms[0]
    return {
        "schema": "robotsim.issue80.route-workcell-geometry-preflight.v1",
        "status": "PASS",
        "classification": "STATIC_OPEN_FLOOR_SCENE; runtime route movement remains separately gated",
        "scene_path": str(scene_path),
        "scene_sha256": ROUTE_SCENE_SHA256,
        "x2_model_path": str(x2_path),
        "x2_model_sha256": ROUTE_X2_SHA256,
        "active_world_collision_geoms": [{
            "name": floor.get("name"),
            "type": floor.get("type"),
            "size": floor.get("size"),
            "contype": floor.get("contype", "1"),
            "conaffinity": floor.get("conaffinity", "1"),
        }],
        "active_world_obstacles": [],
        "planar_floor": "infinite MuJoCo plane; no finite table edge or world boundary",
        "verified_local_clear_radius_m": ROUTE_LOCAL_CLEAR_RADIUS_M,
        "robot_collision_footprint_radial_upper_bound_m": ROUTE_QPOS0_FOOTPRINT_BOUND_M,
        "footprint_method": "compiled geom world centers plus geom_rbound at model qpos0 using host MuJoCo 3.3.6; conservative static reference, not dynamic swept-footprint telemetry",
        "route_corridor_margin_m": 0.10,
        "limitations": [
            "The simulator ROS interface does not publish collision contacts or foot forces; runtime posture, joint, pose, and stop gates remain mandatory.",
            "The 2 m local clearance is a bounded local work area in an otherwise infinite plane, not a mapped or obstacle-aware workspace claim.",
        ],
    }


def validate_right_turn_qualification_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Require a governed, simulation-only right-turn and physical-stop PASS."""
    evidence_path = Path(path)
    if evidence_path.name != "m1_first_motion_result.json" or evidence_path.is_symlink():
        raise ValueError("route prerequisite must be a regular m1_first_motion_result.json")
    validate_reserved_evidence_directory(evidence_path.parent)
    if not str(evidence_path.resolve()).startswith(str(REPO / "reviews/issue-80") + os.sep):
        raise ValueError("right-turn prerequisite must be inside the governed Issue 80 evidence root")
    result = json.loads(evidence_path.read_text(encoding="utf-8"))
    right = result.get("right_turn_qualification")
    if not isinstance(right, Mapping) or right.get("status") != "PASS":
        raise ValueError("right-turn prerequisite did not pass its wrapper acceptance")
    turn = right.get("turn")
    if not isinstance(turn, Mapping):
        raise ValueError("right-turn prerequisite lacks the measured turn result")
    acceptance = turn.get("right_arc_acceptance")
    rotation = turn.get("rotation")
    physical_stop = turn.get("physical_stop_gate")
    if (
        not isinstance(acceptance, Mapping)
        or acceptance.get("status") != "PASS"
        or not isinstance(rotation, Mapping)
        or rotation.get("status") != "TURN_DIRECTION_PASS"
        or float(rotation.get("signed_yaw_delta_rad", 0.0)) > -TURN_MIN_HEADING_CHANGE_RAD
        or not isinstance(physical_stop, Mapping)
        or physical_stop.get("status") != "PASS"
        or float(acceptance.get("maximum_total_turn_origin_displacement_m", float("inf"))) > 0.25
    ):
        raise ValueError("right-turn prerequisite lacks measured yaw, displacement, or physical-stop evidence")
    return {
        "source_path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "attempt_id": str(result.get("attempt_id", "")),
        "classification": right.get("classification"),
        "signed_yaw_delta_rad": rotation.get("signed_yaw_delta_rad"),
        "maximum_total_displacement_m": acceptance.get("maximum_total_turn_origin_displacement_m"),
        "physical_stop_gate": physical_stop.get("status"),
        "status": "PASS",
    }


def validate_turning_reassessment_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Require a reserved, hashed Run65 reassessment with both turn and stop gates passing."""
    helper_path = str(REPO / "scripts/agent")
    if helper_path not in sys.path:
        sys.path.insert(0, helper_path)
    from path_governance import validate_output_path

    file_location = validate_output_path(path, require_exists=True, expected_kind="file")
    if file_location.native_path.is_symlink() or file_location.native_path.name != "result.json":
        raise ValueError("point-goal precondition must be a regular result.json in governed evidence")
    if not file_location.wsl_path.startswith(str(REPO / "reviews/issue-80") + "/"):
        raise ValueError("Run65 prerequisite must come from the governed Issue 80 evidence root")
    run_location = validate_output_path(
        file_location.native_path.parent, require_exists=True, expected_kind="directory"
    )
    if run_location.native_path.is_symlink():
        raise ValueError("Run65 reassessment directory must not be a symlink")
    marker = run_location.native_path / ".robotsim-output-reservation.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("turning reassessment directory lacks its governed reservation")
    reservation = json.loads(marker.read_text(encoding="utf-8"))
    if reservation != {
        "schema_version": "robotsim.output-reservation.v1",
        "path": run_location.wsl_path,
    }:
        raise ValueError("turning reassessment reservation path does not match")
    artifact_manifest = run_location.native_path / "artifact_paths.json"
    if artifact_manifest.is_symlink() or not artifact_manifest.is_file():
        raise ValueError("Run65 reassessment has no regular artifact_paths.json manifest")
    manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Run65 reassessment artifact manifest is malformed")
    if manifest.get("schema_version") != "robotsim.local-artifacts.v1":
        raise ValueError("Run65 reassessment artifact manifest has an unsupported schema")
    report_path = run_location.native_path / "REPORT.md"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("Run65 reassessment report is missing or not a regular file")
    artifacts = manifest.get("artifacts")
    required_paths = {
        file_location.wsl_path,
        f"{run_location.wsl_path}/REPORT.md",
        f"{run_location.wsl_path}/artifact_paths.json",
    }
    if not isinstance(artifacts, list) or not required_paths.issubset({
        item.get("WSL_PATH") for item in artifacts if isinstance(item, Mapping)
    }):
        raise ValueError("Run65 reassessment artifact manifest omits its report or result")
    evidence = json.loads(file_location.native_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping):
        raise ValueError("Run65 reassessment result must be a JSON object")
    return validate_turning_reassessment_evidence(
        evidence,
        evidence_path=file_location.wsl_path,
        sha256=hashlib.sha256(file_location.native_path.read_bytes()).hexdigest(),
    )


def validate_turning_reassessment_evidence(
    evidence: Mapping[str, Any], *, evidence_path: str, sha256: str,
) -> dict[str, Any]:
    """Check the independently reviewed Run65 turn/stop result before point-goal motion."""
    if (
        evidence.get("schema") != "robotsim.issue80.run65-forward-arc-reassessment.v1"
        or evidence.get("classification") != "SIMULATION_ONLY_DERIVED_FOUNDATIONWALK_TURN_PAIR_PASS_REASSESSED"
    ):
        raise ValueError("point-goal requires the verified Run65 forward-arc reassessment PASS")
    acceptance = evidence.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Run65 reassessment acceptance record is missing")
    turns = acceptance.get("turns")
    if not isinstance(turns, list) or len(turns) != 2 or any(not isinstance(turn, Mapping) for turn in turns):
        raise ValueError("point-goal requires two well-formed Run65 turn records")
    if {turn.get("direction") for turn in turns} != {"left", "right"}:
        raise ValueError("point-goal requires reassessed left and right turn evidence")
    for turn in turns:
        gates = turn.get("gates", {})
        if not isinstance(gates, Mapping):
            raise ValueError(f"turning reassessment gates are malformed for {turn.get('direction')}")
        if (
            turn.get("reassessed_status") != "PHYSICAL_TURN_AND_STOP_PASS"
            or any(gates.get(key) is not True for key in (
                "same_sample_source_and_locomotion_handshake",
                "measured_heading_in_commanded_direction",
                "commanded_forward_arc_envelope",
                "physical_stop_gate",
                "no_controller_or_post_stop_fault",
            ))
        ):
            raise ValueError(f"turning reassessment has a failed gate for {turn.get('direction')}")
    original = evidence.get("original_run")
    if not isinstance(original, Mapping):
        raise ValueError("Run65 reassessment original-run provenance is missing")
    if (
        original.get("reset_status") != "AUTO_RESET_PASS"
        or original.get("container_cleanup_verified") is not True
        or original.get("runtime_image_digest") != IMAGE_DIGEST
    ):
        raise ValueError("Run65 reassessment provenance or cleanup did not pass")
    source_faithfulness = acceptance.get("source_faithful_original_config")
    if source_faithfulness != "NOT_TESTED":
        raise ValueError("point-goal must retain the explicit derived-config/source-faithful boundary")
    return {
        "status": "PASS",
        "schema": evidence["schema"],
        "classification": evidence["classification"],
        "path": evidence_path,
        "sha256": sha256,
        "source_attempt_id": original.get("attempt_id"),
        "source_run_evidence_dir": original.get("evidence_dir"),
        "reassessed_directions": [turn.get("direction") for turn in turns],
        "source_faithful_original_config": source_faithfulness,
        "navigation_timestamp_limitation": acceptance.get("odometry_stamp_limitation"),
    }


def validate_run65_point_goal_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Enforce the single reviewed point-goal profile before any runtime starts."""
    if not getattr(args, "maintainer_authorized_first_motion", False):
        raise ValueError("point-goal diagnostic requires first-motion authorization")
    heading_change = float(getattr(args, "heading_change_rad", 0.0))
    if not math.isfinite(heading_change) or abs(heading_change) > POINT_GOAL_MAX_HEADING_CHANGE_RAD:
        raise ValueError(
            f"goal heading change must be within +/-{POINT_GOAL_MAX_HEADING_CHANGE_RAD:.3f} rad"
        )
    max_extra_observation = (
        POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S
        if abs(heading_change) > 1e-9
        else POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S
    )
    if (
        not getattr(args, "capture_mc_rl_debug", False)
        or not POINT_GOAL_MIN_EXTRA_STOP_OBSERVATION_S
        <= float(args.extra_stop_observation)
        <= max_extra_observation
        or abs(float(args.forward_speed) - POINT_GOAL_FORWARD_COMMAND_M_S) > 1e-9
        or abs(float(args.pulse_duration) - 10.0) > 1e-9
        or float(args.ramp_duration_s) != 0.0
        or args.ramp_start_speed is not None
        or args.startup_speed is not None
        or args.startup_transition_displacement is not None
        or float(args.startup_transition_timeout_s) != 0.0
        or not getattr(args, "turning_reassessment_json", None)
    ):
        raise ValueError(
            "point-goal uses the 0.14 m/s forward command, 10-second hard cap, "
            "5-7 second post-stop observation, optional bounded heading change, McRLDebug capture, "
            "and verified Run65 reassessment"
        )
    return validate_turning_reassessment_file(args.turning_reassessment_json)


def run65_point_goal_profile_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the child point-goal policy and the observation actually requested."""
    heading_change = float(getattr(args, "heading_change_rad", 0.0))
    has_heading_target = abs(heading_change) > 1e-9
    return {
        "final_yaw_tolerance_rad": POINT_GOAL_FINAL_YAW_TOLERANCE_RAD,
        "heading_policy": (
            "Run73 position-bearing for point goals; coordinated-pose path and terminal yaw control for heading-change goals"
        ),
        "heading_control_for_pose_goal": (
            "straight zero-angular warmup segment, then curvature-feedforward plus measured-yaw/rate cubic-Hermite pre-turn; lateral error remains a hard guard; bounded forward-zero terminal yaw/rate control" if has_heading_target
            else "not applicable; position-only point goal"
        ),
        "angular_taper_policy": (
            "disabled for coordinated pose goals; path curvature supplies approach heading" if has_heading_target
            else "linear from predicted settled error 0.25 m to 0.05 m position tolerance"
        ),
        "angular_taper_start_predicted_error_m": (
            None if has_heading_target else POINT_GOAL_ANGULAR_TAPER_START_ERROR_M
        ),
        "angular_taper_end_predicted_error_m": (
            None if has_heading_target else POINT_GOAL_FINAL_TOLERANCE_M
        ),
        "post_stop_observation_s": float(args.extra_stop_observation),
        "stop_model": (
            "run92_measured_terminal_entry_position_margin" if has_heading_target
            else "speed_scaled_endpoint_response"
        ),
        "heading_goal_stop_advance_m": (
            POINT_GOAL_HEADING_STOP_ADVANCE_M if has_heading_target else None
        ),
        "heading_goal_predicted_stop_tolerance_m": (
            POINT_GOAL_HEADING_PREDICTED_STOP_TOLERANCE_M if has_heading_target else None
        ),
        "pose_goal_terminal_control": (
            "yaw-only terminal: 0.0 m/s forward, <=0.075 rad/s angular, separate 0.115 rad Run96-conditioned terminal bias, "
            "<=3 s, 0.4 s dwell; final-goal tolerance remains 0.17 rad"
            if has_heading_target else None
        ),
        "pose_goal_post_stop_yaw_drift_compensation_rad": (
            POINT_GOAL_POST_STOP_YAW_DRIFT_COMPENSATION_RAD if has_heading_target else None
        ),
        "pose_goal_terminal_pre_stop_yaw_bias_rad": (
            POINT_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD if has_heading_target else None
        ),
        "pose_goal_terminal_bias_drift_observation_rad": (
            POINT_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD if has_heading_target else None
        ),
        "pose_goal_terminal_direction_margin_rad": (
            abs(heading_change)
            + POINT_GOAL_TERMINAL_PRESTOP_YAW_BIAS_RAD
            - POINT_GOAL_RUN96_POST_STOP_YAW_DRIFT_RAD
            - POINT_GOAL_TERMINAL_SETPOINT_TOLERANCE_RAD
            - TURN_MIN_HEADING_CHANGE_RAD
            if has_heading_target else None
        ),
        "pose_goal_warmup_duration_s": (POSE_GOAL_APPROACH_WARMUP_S if has_heading_target else None),
        "pose_goal_warmup_forward_m_s": (POSE_GOAL_WARMUP_FORWARD_M_S if has_heading_target else None),
        "pose_goal_curve_forward_command_m_s": (POSE_GOAL_CURVE_FORWARD_M_S if has_heading_target else None),
        "pose_goal_curve_angular_rate_utilization_limit": (0.95 if has_heading_target else None),
        "pose_goal_observed_post_stop_yaw_drift_rad": (
            list(POINT_GOAL_POST_STOP_DRIFT_OBSERVATIONS_RAD) if has_heading_target else None
        ),
        "pose_goal_pre_stop_yaw_margin_rad": (
            POINT_GOAL_PRESTOP_YAW_MARGIN_RAD if has_heading_target else None
        ),
        "pose_goal_terminal_setpoint_tolerance_rad": (
            POINT_GOAL_TERMINAL_SETPOINT_TOLERANCE_RAD if has_heading_target else None
        ),
        "pose_goal_approach_path_terminal_heading_policy": (
            "0.05 rad approach-path bias preserves curve feasibility; 0.115 rad terminal bias is conditioned on Run96 drift and the 5-degree direction gate; not calibrated across Runs93/95"
            if has_heading_target else None
        ),
    }


def run65_child_heading_policy(args: argparse.Namespace) -> str:
    """Select the exact child controller branch for the requested goal type."""
    return (
        "coordinated-pose"
        if abs(float(getattr(args, "heading_change_rad", 0.0))) > 1e-9
        else "position-bearing"
    )


PRIOR_CURVED_ABORT_TRACE = (
    REPO / "reviews/issue-80/issue80-curved-route-run-20261025T033414Z/m1_first_motion_trace.jsonl"
)
PRIOR_CURVED_ABORT_TRACE_SHA256 = "1767cab680b0f8124a4d67f4f479ad6d6553c78d3874647a707a002a8b581d11"
ACTIVATION_PROBE_FORWARD_M_S = 0.12
ACTIVATION_PROBE_DURATION_S = 1.8


def assess_activation_probe_prerequisite(
    reset_run: Mapping[str, Any],
    zero_preflight: Mapping[str, Any],
    prior_replay: Mapping[str, Any],
    *,
    current_runtime_image_digest: str = IMAGE_DIGEST,
    current_mc_config_sha256: str = RUN38_MC_SIM_CONFIG_SHA256,
) -> dict[str, Any]:
    """Require safe zero-preflight recovery plus independent installed-run start evidence."""
    if reset_run.get("reset_classification") != "AUTO_RESET_PASS":
        raise ValueError("activation probe requires a preserved AUTO_RESET_PASS zero-only preflight")
    if reset_run.get("runtime_image_digest") != current_runtime_image_digest:
        raise ValueError("zero-only preflight runtime image differs from the selected simulation runtime")
    child_attempt = zero_preflight.get("attempt_id")
    if child_attempt != reset_run.get("attempt_id"):
        raise ValueError("zero-only preflight parent/child attempt identities do not match")
    if zero_preflight.get("simulation_only") is not True:
        raise ValueError("activation probe prerequisite is not classified SIMULATION_ONLY")
    if zero_preflight.get("installed_mc_sim_config_sha256") != current_mc_config_sha256:
        raise ValueError("zero-only preflight MC config hash differs from the current installed config")
    zero_result = zero_preflight.get("zero_only_activation_verification")
    if not isinstance(zero_result, Mapping):
        raise ValueError("zero-only activation result is missing")
    if zero_result.get("activation_confirmation_pass") is not False:
        raise ValueError("activation probe is only for a zero-only preflight that did not confirm LOCOMOTION")
    if zero_result.get("stand_and_composite_stop_pass") is not True:
        raise ValueError("activation probe requires completed STAND recovery and physical-stop PASS")
    if zero_result.get("nonzero_publish_calls_returned") != 0:
        raise ValueError("zero-only prerequisite unexpectedly published a nonzero command")
    if zero_result.get("late_locomotion_after_abort") is True:
        raise ValueError("zero-only preflight observed unresolved late LOCOMOTION after abort")

    activation = zero_result.get("activation")
    if not isinstance(activation, Mapping):
        raise ValueError("zero-only activation trace is missing")
    registration = activation.get("input_source_registration")
    set_action = activation.get("set_mc_action_ld")
    observations = activation.get("observations")
    if not isinstance(registration, Mapping) or registration.get("accepted_by_service") is not True:
        raise ValueError("zero-only preflight did not receive an accepted ADD operation")
    if not isinstance(set_action, Mapping) or set_action.get("accepted_by_service") is not True:
        raise ValueError("zero-only preflight did not receive an accepted SetMcAction LD response")
    if not isinstance(observations, list) or not observations:
        raise ValueError("zero-only preflight has no fresh activation observations")
    if not all(
        row.get("classification") == "PENDING_STAND_REMAINS_ACTIVE"
        and row.get("get_mode_desc") == MODE_STAND
        and row.get("common_mode_desc") == MODE_STAND
        and row.get("get_source_name", "") == ""
        and row.get("common_source_name", "") == ""
        and not row.get("safety_fault")
        for row in observations
    ):
        raise ValueError("zero-only preflight did not consistently remain in safe unselected STAND")

    if prior_replay.get("source_trace_sha256") != PRIOR_CURVED_ABORT_TRACE_SHA256:
        raise ValueError("offline replay fixture does not bind the preserved installed-runtime trace")
    commands = prior_replay.get("command_timeline")
    source_evidence = prior_replay.get("installed_runtime_nonzero_activation_evidence")
    if not isinstance(commands, Mapping) or not isinstance(source_evidence, Mapping):
        raise ValueError("offline replay lacks the prior source/mode activation evidence")
    if (
        commands.get("nonzero_publish_calls_returned") != 14
        or commands.get("first_nonzero_forward_velocity_m_s") != ACTIVATION_PROBE_FORWARD_M_S
        or commands.get("first_nonzero_lateral_velocity_m_s") != 0.0
        or commands.get("first_nonzero_angular_velocity_rad_s") != 0.0
        or commands.get("last_nonzero_forward_velocity_m_s") != ACTIVATION_PROBE_FORWARD_M_S
        or commands.get("last_nonzero_lateral_velocity_m_s") != 0.0
        or commands.get("last_nonzero_angular_velocity_rad_s") != 0.0
        or source_evidence.get("selected_source_get_service") != "node.locomotion_velocity"
        or source_evidence.get("selected_source_common_state") != "node.locomotion_velocity"
        or float(source_evidence.get("common_state_locomotion_offset_s", 99.0)) >= 0.5
        or float(source_evidence.get("get_mc_action_locomotion_offset_s", 99.0)) > ACTIVATION_PROBE_DURATION_S
    ):
        raise ValueError("preserved runtime trace does not support the fixed bounded activation-probe profile")

    return {
        "status": "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED",
        "runtime_image_digest": current_runtime_image_digest,
        "installed_mc_sim_config_sha256": current_mc_config_sha256,
        "zero_only_preflight_attempt_id": child_attempt,
        "zero_only_preflight_status": zero_result.get("status"),
        "zero_only_recovery_physical_stop": "PASS",
        "zero_only_source_registration_task_completion_confirmed": registration.get("task_completion_confirmed"),
        "prior_installed_runtime_attempt_id": prior_replay.get("source_attempt_id"),
        "prior_installed_runtime_trace_sha256": PRIOR_CURVED_ABORT_TRACE_SHA256,
        "prior_nonzero_publish_count": commands.get("nonzero_publish_calls_returned"),
        "prior_forward_velocity_m_s": ACTIVATION_PROBE_FORWARD_M_S,
        "prior_source_selected_offset_s": source_evidence.get("selected_source_observation_offset_s"),
        "prior_common_locomotion_offset_s": source_evidence.get("common_state_locomotion_offset_s"),
        "prior_get_mc_action_locomotion_offset_s": source_evidence.get("get_mc_action_locomotion_offset_s"),
        "activation_probe_forward_velocity_m_s": ACTIVATION_PROBE_FORWARD_M_S,
        "activation_probe_max_duration_s": ACTIVATION_PROBE_DURATION_S,
        "measured_speed_limit_m_s": 0.20,
        "version_match_limitation": (
            "online arbitration/start-threshold docs are AimDK_X2 1.1.0 and are not declared binary-version matched; "
            "the preserved 2026-10-25 trace independently confirms this installed runtime selected the source and "
            "reported CommonState LOCOMOTION after a 0.12 m/s input"
        ),
    }


def validate_activation_probe_prerequisite(evidence_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the reserved zero-only run and require the exact source-grounded probe gate."""
    directory = validate_reserved_evidence_directory(evidence_dir)
    reset_path = directory / "auto-reset-run-result.json"
    child_path = directory / "m1_first_motion_result.json"
    replay_path = Path(__file__).with_name("fixtures") / "issue80_curved_abort_transition_replay.json"
    prior_parent_path = PRIOR_CURVED_ABORT_TRACE.parent / "auto-reset-run-result.json"
    prior_child_path = PRIOR_CURVED_ABORT_TRACE.parent / "m1_first_motion_result.json"
    if not all(
        path.is_file() and not path.is_symlink()
        for path in (
            reset_path, child_path, replay_path, PRIOR_CURVED_ABORT_TRACE,
            prior_parent_path, prior_child_path,
        )
    ):
        raise ValueError("zero-only run results or installed-runtime replay fixture are missing/invalid")
    if hashlib.sha256(PRIOR_CURVED_ABORT_TRACE.read_bytes()).hexdigest() != PRIOR_CURVED_ABORT_TRACE_SHA256:
        raise ValueError("the preserved installed-runtime source-acquisition trace changed")
    reset_run = json.loads(reset_path.read_text(encoding="utf-8"))
    zero_preflight = json.loads(child_path.read_text(encoding="utf-8"))
    prior_replay = json.loads(replay_path.read_text(encoding="utf-8"))
    prior_parent = json.loads(prior_parent_path.read_text(encoding="utf-8"))
    prior_child = json.loads(prior_child_path.read_text(encoding="utf-8"))
    if (
        hashlib.sha256(prior_parent_path.read_bytes()).hexdigest()
        != prior_replay.get("source_parent_result_sha256")
        or hashlib.sha256(prior_child_path.read_bytes()).hexdigest()
        != prior_replay.get("source_result_sha256")
        or prior_parent.get("runtime_image_digest") != IMAGE_DIGEST
        or prior_parent.get("reset_classification") != "AUTO_RESET_PASS"
        or prior_child.get("attempt_id") != prior_replay.get("source_attempt_id")
        or prior_child.get("installed_mc_sim_config_sha256") != RUN38_MC_SIM_CONFIG_SHA256
    ):
        raise ValueError("prior installed-runtime activation trace lacks matching image/config/result provenance")
    result = assess_activation_probe_prerequisite(reset_run, zero_preflight, prior_replay)
    result["prior_runtime_image_digest"] = prior_parent.get("runtime_image_digest")
    result["prior_mc_sim_config_sha256"] = prior_child.get("installed_mc_sim_config_sha256")
    result["prior_m1_result_sha256"] = prior_replay.get("source_result_sha256")
    result["prior_parent_result_sha256"] = prior_replay.get("source_parent_result_sha256")
    return result


def validate_waypoint_docking_route_input(route_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a Local01 route config against the pinned simple scene, without runtime startup."""
    source = Path(route_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("route config must be a regular, non-symlink input file")
    pose_input = load_waypoint_docking_pose_input(source)
    workcell = inspect_route_workcell_geometry(RUNTIME)
    if workcell.get("status") != "PASS":
        raise ValueError("pinned simple-scene workcell preflight did not pass")
    route = build_waypoint_docking_route_from_poses(
        pose_input,
        runtime_robot_footprint_radius_m=float(
            workcell["robot_collision_footprint_radial_upper_bound_m"]
        ),
        verified_local_clear_radius_m=float(workcell["verified_local_clear_radius_m"]),
    )
    clearance = validate_waypoint_docking_scene_clearance(
        route,
        runtime_robot_footprint_radius_m=float(workcell["robot_collision_footprint_radial_upper_bound_m"]),
        verified_local_clear_radius_m=float(workcell["verified_local_clear_radius_m"]),
    )
    return {
        "route": route,
        "pose_input": pose_input,
        "workcell": workcell,
        "clearance": clearance,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def validate_local01_scene_route_input(poses_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Pin Local01's scene config and preflight the live raw-odom short route."""
    source = load_local01_scene_inputs(poses_path, require_clean_commit=True)
    route_input = build_runtime_raw_odom_short_route_config(source)
    static = validate_short_route_static_scene(
        Path(source["repository_root"]) / "simulation/mujoco/scenes/raicom2026",
        route_input,
    )
    if static.get("status") != "PASS_STATIC_SCENE_CLEARANCE":
        raise ValueError(
            f"Local01 short route failed the actual RAICOM scene static preflight: {static.get('status')}"
        )
    if abs(
        float(static["robot_collision_footprint_radial_upper_bound_m"])
        - ROUTE_QPOS0_FOOTPRINT_BOUND_M
    ) > 1e-9:
        raise ValueError("Local01 scene compiled robot footprint differs from the pinned X2 runtime bound")
    route = build_local01_short_scene_route(
        route_input,
        runtime_robot_footprint_radius_m=float(static["robot_collision_footprint_radial_upper_bound_m"]),
        verified_local_clear_radius_m=float(static["verified_local_clear_radius_m"]),
    )
    return {
        "source": source,
        "route_input": route_input,
        "route": route,
        "static_scene_preflight": static,
        "source_sha256": source["source_hashes"]["scene_world_poses"],
        "source_commit": LOCAL01_COMMIT,
    }


def validate_local01_table_approach_mission_input(
    poses_path: str | os.PathLike[str], mission_path: str | os.PathLike[str],
    static_plan_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the non-docking mission against the pinned source and static plan.

    This remains an L0/L1 preflight.  The saved plan contains only mj_forward
    geometry evidence and cannot make the yaw alignment or gait dynamic PASS.
    """
    source = load_local01_scene_inputs(poses_path, require_clean_commit=True)
    mission_file = Path(mission_path)
    plan_file = Path(static_plan_path)
    if any(path.is_symlink() or not path.is_file() for path in (mission_file, plan_file)):
        raise ValueError("table-approach mission and static plan must be regular files")
    try:
        mission = json.loads(mission_file.read_text(encoding="utf-8"))
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"table-approach mission or static plan is unreadable: {type(exc).__name__}") from exc
    if not isinstance(mission, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("table-approach mission and static plan must be JSON objects")
    validate_runtime_raw_odom_table_approach_mission(mission)
    expected = build_runtime_raw_odom_table_approach_mission(source)
    if mission != expected:
        raise ValueError("table-approach mission differs from the pinned source-derived contract")
    if (
        plan.get("status") != "STATIC_GEOMETRY_REVIEW_ONLY_TABLE_APPROACH_NOT_DYNAMICALLY_QUALIFIED"
        or plan.get("source_commit") != source["source_commit"]
        or plan.get("source_hashes") != source["source_hashes"]
        or plan.get("safe_table_approach_mission", {}).get("classification")
            != mission.get("classification")
        or plan.get("safe_table_approach_mission", {}).get("table_docking_reference_unchanged") is not True
    ):
        raise ValueError("table-approach static plan does not match the pinned source/mission")
    required_segments = {
        ("START", "START_ALIGN"), ("START_ALIGN", "WAYPOINT"),
        ("WAYPOINT", "TABLE_APPROACH"),
    }
    observed = {
        (str(item.get("from")), str(item.get("to")))
        for item in plan.get("segments", []) if isinstance(item, Mapping)
        and item.get("status") == "PASS_STATIC_ONLY"
    }
    if not required_segments.issubset(observed):
        raise ValueError("table-approach static plan lacks a passing full safe-prefix segment")
    return {
        "source": source,
        "mission": mission,
        "plan": plan,
        "mission_sha256": hashlib.sha256(mission_file.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
    }


def validate_local01_aligned_start_table_approach_mission_input(
    poses_path: str | os.PathLike[str], mission_path: str | os.PathLike[str],
    static_plan_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the separately versioned pre-physics aligned-start integration.

    The original source mission is retained by the generated input as a
    reference, but cannot be silently substituted by this derived scene.
    """
    source = load_local01_scene_inputs(poses_path, require_clean_commit=True)
    mission_file = Path(mission_path)
    plan_file = Path(static_plan_path)
    if any(path.is_symlink() or not path.is_file() for path in (mission_file, plan_file)):
        raise ValueError("aligned-start mission and static plan must be regular files")
    try:
        mission = json.loads(mission_file.read_text(encoding="utf-8"))
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"aligned-start mission or static plan is unreadable: {type(exc).__name__}"
        ) from exc
    if not isinstance(mission, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("aligned-start mission and static plan must be JSON objects")
    validate_runtime_raw_odom_aligned_start_table_approach_mission(mission)
    expected = build_runtime_raw_odom_aligned_start_table_approach_mission(source)
    if mission != expected:
        raise ValueError("aligned-start mission differs from the pinned source-derived contract")
    if (
        plan.get("status")
            != "STATIC_GEOMETRY_REVIEW_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_DYNAMICALLY_QUALIFIED"
        or plan.get("source_commit") != source["source_commit"]
        or plan.get("source_hashes") != source["source_hashes"]
        or plan.get("safe_table_approach_mission", {}).get("classification")
            != mission.get("classification")
        or plan.get("safe_table_approach_mission", {}).get("table_docking_reference_unchanged") is not True
        or plan.get("aligned_start_scene_derivation")
            != mission.get("derived_scene_initialization")
    ):
        raise ValueError("aligned-start static plan does not match the pinned derived mission")
    required_segments = {("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH")}
    observed = {
        (str(item.get("from")), str(item.get("to")))
        for item in plan.get("segments", []) if isinstance(item, Mapping)
        and item.get("status") == "PASS_STATIC_ONLY"
    }
    if not required_segments.issubset(observed):
        raise ValueError("aligned-start static plan lacks passing full derived-corridor segments")
    return {
        "source": source,
        "mission": mission,
        "plan": plan,
        "mission_sha256": hashlib.sha256(mission_file.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
        "aligned_start": True,
    }


def build_m1_child_command(args: argparse.Namespace, container: str) -> list[str]:
    """Build the exact production Docker exec command for the M1 child runner."""
    zero_only_activation = bool(
        getattr(args, "maintainer_authorized_zero_only_activation", False)
    )
    activation_probe = bool(getattr(args, "maintainer_authorized_activation_probe", False))
    child_authorization = (
        "--maintainer-authorized-zero-only-activation --zero-only-activation-verification"
        if zero_only_activation else
        "--maintainer-authorized-activation-probe --activation-probe-qualification"
        if activation_probe else "--maintainer-authorized"
    )
    configured_waypoint_route = bool(
        getattr(args, "route_config", None)
        or getattr(args, "local01_scene_world_poses", None)
        or getattr(args, "table_approach_mission", None)
    )
    local01_short_route = bool(
        getattr(args, "local01_scene_world_poses", None)
        and not getattr(args, "table_approach_mission", None)
    )
    child_extra_stop = (
        5.0 if activation_probe else 7.0 if configured_waypoint_route
        else args.extra_stop_observation
    )
    child_pulse = ACTIVATION_PROBE_DURATION_S if activation_probe else args.pulse_duration
    child_forward = 0.12 if activation_probe else args.forward_speed
    startup_gate_flag = (
        "--initialization-gate-verified"
        if getattr(args, "static_nominal_start_initialization", False)
        else "--reset-gate-verified"
    )
    command = [
        "docker", "exec", "-e", "ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", container, "bash", "-lc",
        "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && python3 /workspace/evidence/issue80_m1_locomotion_probe.py first-motion " + child_authorization + " " + startup_gate_flag
        + " --extra-stop-observation " + str(child_extra_stop)
        + " --pulse-duration " + str(child_pulse)
        + " --forward-speed " + str(child_forward)
        + " --attempt-id " + str(args.attempt_id)
        + " --robot-model-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/robot_model.yaml"
        + " --mc-sim-config-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"
        + " --state-timeout 0.5 --ready-timeout 4 --max-tilt-deg 15 --min-root-z 0.52"
        + " --output-dir /workspace/evidence"
    ]
    direct_distance_m = getattr(args, "raicom_vendor_direct_distance_m", None)
    direct_table_target = bool(getattr(args, "raicom_vendor_direct_table_target", False))
    if direct_distance_m is not None or direct_table_target:
        # This is intentionally a separate, official-example-shaped baseline.
        # It is launched only after the parent has completed the existing
        # Reset/JD/SD standing gate, and it writes the same child result file
        # consumed by the parent.  Do not route it through the experimental
        # activation/route wrapper that previously returned before publishing.
        command[-1] = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "python3 /workspace/evidence/issue80_vendor_direct_truth_route.py"
            " --distance-m " + str(2.5 if direct_table_target else float(direct_distance_m))
            + " --timeout-s " + ("25" if direct_table_target else "20")
            + " --output /workspace/evidence/m1_first_motion_result.json"
            + " --trace /workspace/evidence/m1_vendor_direct_truth_trace.jsonl"
        )
        if direct_table_target:
            command[-1] += (
                " --target-route-config /workspace/evidence/route_config.json"
                " --target-name TABLE_APPROACH --target-tolerance-m 0.15"
            )
            if bool(getattr(args, "raicom_vendor_direct_heading_hold", False)):
                command[-1] += " --heading-hold"
        return command
    ramp_start_speed = getattr(args, "ramp_start_speed", None)
    ramp_duration_s = float(getattr(args, "ramp_duration_s", 0.0))
    if ramp_start_speed is not None or ramp_duration_s != 0.0:
        command[-1] += (
            " --ramp-start-speed " + str(ramp_start_speed or args.forward_speed)
            + " --ramp-duration-s " + str(ramp_duration_s)
        )
    startup_speed = getattr(args, "startup_speed", None)
    if startup_speed is not None:
        command[-1] += (
            " --startup-speed " + str(startup_speed)
            + " --startup-transition-displacement "
            + str(getattr(args, "startup_transition_displacement", None))
            + " --startup-transition-timeout-s "
            + str(getattr(args, "startup_transition_timeout_s", 0.0))
        )
    diagnostic_names = (
        "run41_min_run_diagnostic", "run42_stop_confirmation_diagnostic",
        "run43_zero_only_stop_diagnostic", "run44_threshold_margin_diagnostic",
        "run45_ten_second_gait_diagnostic", "run46_startup_cruise_threshold_diagnostic",
        "run48_low_cruise_baseline_diagnostic", "run49_smoothstop_diagnostic",
        "run50_threshold_respecting_smoothstop_diagnostic", "run53_turning_diagnostic",
        "run65_right_turn_qualification", "run65_route_integration",
        "run65_waypoint_b_only", "run65_waypoint_b_short_diagnostic",
        "run65_waypoint_b_extended_diagnostic", "run65_curved_path_diagnostic",
        "route_config", "local01_scene_world_poses",
    )
    if zero_only_activation or activation_probe or any(bool(getattr(args, name, False)) for name in diagnostic_names) or run65_point_goal_enabled(args):
        command[-1] += (
            " --max-observed-planar-speed-m-s "
            + (
                "0.8"
                if (
                    getattr(args, "raicom_vendor_example_forward_baseline", False)
                    or getattr(args, "raicom_vendor_table_approach", False)
                ) else "0.2"
            )
        )
    stand_timeout_diagnostics = (
        "run42_stop_confirmation_diagnostic", "run43_zero_only_stop_diagnostic",
        "run44_threshold_margin_diagnostic", "run45_ten_second_gait_diagnostic",
        "run46_startup_cruise_threshold_diagnostic", "run48_low_cruise_baseline_diagnostic",
        "run49_smoothstop_diagnostic", "run50_threshold_respecting_smoothstop_diagnostic",
        "run53_turning_diagnostic", "run65_right_turn_qualification",
        "run65_route_integration", "run65_waypoint_b_only",
        "run65_waypoint_b_short_diagnostic", "run65_waypoint_b_extended_diagnostic",
        "run65_curved_path_diagnostic", "route_config", "local01_scene_world_poses",
    )
    if zero_only_activation or activation_probe or any(bool(getattr(args, name, False)) for name in stand_timeout_diagnostics) or run65_point_goal_enabled(args):
        command[-1] += " --zero-until-stand-timeout-s 3.5"
    zero_only_diagnostics = (
        "run43_zero_only_stop_diagnostic", "run44_threshold_margin_diagnostic",
        "run45_ten_second_gait_diagnostic", "run46_startup_cruise_threshold_diagnostic",
        "run48_low_cruise_baseline_diagnostic", "run49_smoothstop_diagnostic",
        "run50_threshold_respecting_smoothstop_diagnostic", "run53_turning_diagnostic",
        "run65_right_turn_qualification", "run65_route_integration",
        "run65_waypoint_b_only", "run65_waypoint_b_short_diagnostic",
        "run65_waypoint_b_extended_diagnostic", "run65_curved_path_diagnostic",
        "route_config", "local01_scene_world_poses",
    )
    if zero_only_activation or activation_probe or any(bool(getattr(args, name, False)) for name in zero_only_diagnostics) or run65_point_goal_enabled(args):
        command[-1] += " --zero-only-stop"
    if activation_probe:
        prerequisite = getattr(args, "activation_probe_prerequisite", None)
        if not isinstance(prerequisite, Mapping) or prerequisite.get("status") != "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED":
            raise ValueError("controlled activation probe has no validated zero-only/source-contract prerequisite")
        command[-1] += " --activation-probe-prerequisite-attempt-id " + str(
            prerequisite["zero_only_preflight_attempt_id"]
        )
        command[-1] += " --activation-probe-prerequisite-path /workspace/evidence/activation_probe_prerequisite.json"
    if getattr(args, "run49_smoothstop_diagnostic", False):
        command[-1] += " --deceleration-duration-s 1.0"
    if getattr(args, "run50_threshold_respecting_smoothstop_diagnostic", False):
        command[-1] += " --deceleration-duration-s 1.0 --deceleration-end-speed-m-s 0.112"
    if getattr(args, "run53_turning_diagnostic", False):
        command[-1] += " --turning-diagnostic"
    if getattr(args, "run65_right_turn_qualification", False):
        command[-1] += " --right-turn-qualification"
    if getattr(args, "run65_route_integration", False):
        command[-1] += (
            " --route-integration --route-turn-qualification-evidence "
            "/workspace/evidence/route_turn_qualification_evidence.json"
        )
    if getattr(args, "run65_curved_path_diagnostic", False):
        command[-1] += " --route-integration --route-curved-path-diagnostic"
    elif getattr(args, "run65_waypoint_b_extended_diagnostic", False):
        command[-1] += (
            " --route-integration --route-waypoint-b-only "
            "--route-waypoint-b-extended-diagnostic"
        )
        if getattr(args, "post_locomotion_convergence_observation_s", None) is not None:
            command[-1] += " --route-waypoint-b-post-locomotion-convergence"
    elif getattr(args, "run65_waypoint_b_short_diagnostic", False):
        command[-1] += (
            " --route-integration --route-waypoint-b-only "
            "--route-waypoint-b-short-diagnostic"
        )
    elif getattr(args, "run65_waypoint_b_only", False):
        command[-1] += " --route-integration --route-waypoint-b-only"
    if configured_waypoint_route:
        command[-1] += (
            " --route-integration --route-config /workspace/evidence/route_config.json"
            " --route-config-post-stand-observation"
        )
        if local01_short_route:
            command[-1] += " --short-scene-route"
        if getattr(args, "table_approach_mission", None):
            command[-1] += " --table-approach-mission"
            if getattr(args, "aligned_start_table_approach", False):
                command[-1] += " --aligned-start-table-approach"
            if getattr(args, "native_x2_timestep", False):
                command[-1] += " --native-x2-timestep"
            if getattr(args, "raicom_native_mjdata_route_feedback", False):
                command[-1] += " --raicom-native-mjdata-route-feedback"
        if getattr(args, "raicom_low_cruise_profile", False):
            command[-1] += " --raicom-low-cruise-profile"
        if getattr(args, "raicom_zero_forward_heading_acquisition", False):
            command[-1] += " --raicom-zero-forward-heading-acquisition"
        if getattr(args, "raicom_continuous_ten_second_transit", False):
            command[-1] += " --raicom-continuous-ten-second-transit"
        if getattr(args, "raicom_continuous_table_approach", False):
            command[-1] += " --raicom-continuous-table-approach"
        if getattr(args, "raicom_fixed_window_table_approach", False):
            command[-1] += " --raicom-fixed-window-table-approach"
        if getattr(args, "raicom_straight_ten_second_table_approach", False):
            command[-1] += " --raicom-straight-ten-second-table-approach"
        if getattr(args, "raicom_forward_gait_yaw_inversion", False):
            command[-1] += " --raicom-forward-gait-yaw-inversion"
        if getattr(args, "raicom_native_nominal_straight_gait_diagnostic", False):
            command[-1] += " --raicom-native-nominal-straight-gait-diagnostic"
        if getattr(args, "raicom_vendor_example_forward_baseline", False):
            command[-1] += " --raicom-vendor-example-forward-baseline"
        if getattr(args, "raicom_vendor_table_approach", False):
            command[-1] += " --raicom-vendor-table-approach"
    if run65_point_goal_enabled(args):
        command[-1] += (
            " --point-goal-diagnostic --heading-change-rad "
            + str(float(getattr(args, "heading_change_rad", 0.0)))
            + " --final-yaw-tolerance-rad "
            + str(POINT_GOAL_FINAL_YAW_TOLERANCE_RAD)
            + " --heading-control-policy "
            + run65_child_heading_policy(args)
        )
        if run65_child_heading_policy(args) == "coordinated-pose":
            command[-1] += (
                " --heading-goal-stop-advance-m "
                + str(POINT_GOAL_HEADING_STOP_ADVANCE_M)
                + " --heading-goal-predicted-stop-tolerance-m "
                + str(POINT_GOAL_HEADING_PREDICTED_STOP_TOLERANCE_M)
            )
    return command


def run65_point_goal_enabled(args: argparse.Namespace) -> bool:
    """Return the single argparse destination used by every Run65 execution stage."""
    return bool(getattr(args, "run65_ground_truth_point_goal", False))


def add_run65_point_goal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run65-ground-truth-point-goal",
        dest="run65_ground_truth_point_goal",
        action="store_true",
        help=(
            "run one SIMULATION_ONLY 0.30 m odom-frame point/yaw goal with along-track/lateral "
            "ground-truth feedback, optional heading-change offset, and a speed-scaled Run69 endpoint correction, verified Run65 "
            "turn/stop prerequisite, 0.14 m/s forward / 0.075 rad/s turn envelope, and 10-second "
            "hard command cap"
        ),
    )
    parser.add_argument(
        "--turning-reassessment-json",
        default=None,
        help="governed Run65 forward-arc reassessment result.json required before point-goal runtime",
    )
    parser.add_argument(
        "--heading-change-rad",
        type=float,
        default=0.0,
        help=(
            "bounded target path/yaw offset from the measured start heading, within +/-0.20 rad; "
            "zero runs a straight goal"
        ),
    )


def classify_point_goal_result(point_goal: Mapping[str, Any] | None) -> str:
    """Classify only a completed ground-truth goal plus the child's guarded stop gates as pass."""
    if not isinstance(point_goal, Mapping):
        return "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED"
    activation = point_goal.get("activation")
    fault = str(point_goal.get("controller_fault") or "")
    if isinstance(activation, Mapping):
        fault = " ".join((fault, str(activation.get("fault") or ""))).strip()
    if any(token in fault.lower() for token in M1_SAFETY_FAULT_MARKERS):
        return "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_FAIL"
    activation = activation if isinstance(activation, Mapping) else {}
    final_distance = point_goal.get("final_distance_to_goal_m")
    acceptance_gates = point_goal.get("acceptance_gates")
    physical_stop = activation.get("physical_stop_gate")
    zero_stop = activation.get("zero_stop")
    if (
        point_goal.get("status") == "PASS"
        and isinstance(acceptance_gates, Mapping)
        and acceptance_gates
        and all(value is True for value in acceptance_gates.values())
        and point_goal.get("goal_triggered") is True
        and isinstance(final_distance, (int, float))
        and math.isfinite(float(final_distance))
        and float(final_distance) <= POINT_GOAL_FINAL_TOLERANCE_M
        and activation.get("termination_condition_met") is True
        and activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
        and activation.get("control_handshake_status") == "CONTROL_HANDSHAKE_CONFIRMED"
        and isinstance(physical_stop, Mapping)
        and physical_stop.get("status") == "PASS"
        and isinstance(zero_stop, Mapping)
        and int(zero_stop.get("zero_publish_calls_returned", 0) or 0) >= ZERO_STOP_MIN_MESSAGES
    ):
        return "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_PASS"
    return "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED"


M1_CHILD_RESULT_STATUSES = frozenset({"PASS", "BLOCKED"})
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
M1_SAFETY_FAULT_MARKERS = (
    "fall_",
    "mc_safety",
    "non_standing_body_status",
    "stale_or_missing_state",
    "source_joint_position_limit",
    "source_joint_velocity_limit",
    "measured_root_planar_speed_limit",
)


def decode_m1_child_result(serialized: str) -> dict[str, Any]:
    """Decode one child result object, accepting one legacy string wrapper.

    The child currently writes a JSON string containing its Python mapping
    representation because its local _json_safe helper does not normalize
    Mapping instances. Prefer one extra JSON decode; retain a literal-only
    compatibility path for that observed representation. Never eval code.
    """
    try:
        value: Any = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"child result is not valid JSON: {exc}") from exc

    if isinstance(value, str):
        wrapped_value = value
        try:
            value = json.loads(wrapped_value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(wrapped_value)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise ValueError("child result string wrapper is not a mapping literal") from exc

    if not isinstance(value, Mapping):
        raise ValueError(f"child result must decode to a mapping, got {type(value).__name__}")
    result = dict(value)
    status = result.get("status")
    if not isinstance(status, str) or status not in M1_CHILD_RESULT_STATUSES:
        raise ValueError(f"child result has invalid status: {status!r}")
    return result


def classify_first_motion_outcome(activation: Mapping[str, Any] | None) -> str:
    """Require measured motion and one explicitly recorded physical stop gate."""
    if not isinstance(activation, Mapping):
        return "VELOCITY_COMMAND_NOT_EXECUTED"
    if int(activation.get("nonzero_publish_calls_returned", 0) or 0) <= 0:
        return "VELOCITY_COMMAND_NOT_EXECUTED"
    activation_fault = str(activation.get("fault") or "").lower()
    if any(marker in activation_fault for marker in M1_SAFETY_FAULT_MARKERS):
        return "FIRST_PHYSICAL_WALKING_FAIL"
    if (
        activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED"
        and isinstance(activation.get("physical_stop_gate"), Mapping)
        and activation["physical_stop_gate"].get("status") == "PASS"
    ):
        return "FIRST_PHYSICAL_WALKING_PASS"
    if activation.get("status") == "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED":
        return "FIRST_PHYSICAL_WALKING_PASS"
    if activation.get("physical_motion_status") == "PHYSICAL_MOTION_CONFIRMED":
        return "FIRST_PHYSICAL_WALKING_MOTION_CONFIRMED_STOP_UNVERIFIED"
    return "FIRST_PHYSICAL_WALKING_FAIL"


def classify_activation_probe_outcome(qualification: Mapping[str, Any] | None) -> str:
    """Report activation and abort-stop evidence independently of the intentional abort marker."""
    if not isinstance(qualification, Mapping):
        return "SIMULATION_ONLY_ACTIVATION_PROBE_EVIDENCE_MISSING"
    if qualification.get("late_locomotion_after_abort") is True:
        return "SIMULATION_ONLY_ACTIVATION_ABORT_LATE_LOCOMOTION"
    if (
        qualification.get("mode_source_activation_pass") is True
        and qualification.get("safe_stand_and_physical_stop_pass") is True
        and qualification.get("status") == "PASS"
    ):
        return "SIMULATION_ONLY_ACTIVATION_ABORT_AND_PHYSICAL_STOP_PASS"
    if qualification.get("mode_source_activation_pass") is True:
        return "SIMULATION_ONLY_ACTIVATION_ABORT_PHYSICAL_STOP_FAIL"
    return "SIMULATION_ONLY_ACTIVATION_NOT_CONFIRMED"


def assess_container_absence(
    *,
    inspect_returncode: int,
    list_returncode: int,
    list_stdout: str,
    container_name: str,
    container_id: str | None,
) -> dict[str, Any]:
    """Require inspect failure plus a successful exact name/ID inventory miss."""
    entries: list[tuple[str, str]] = []
    inventory_valid = bool(container_name)
    for line in list_stdout.splitlines():
        if not line:
            continue
        fields = line.split("\t", 1)
        if (
            len(fields) != 2
            or CONTAINER_ID_RE.fullmatch(fields[0]) is None
            or not fields[1]
        ):
            inventory_valid = False
            break
        entries.append((fields[0], fields[1]))
    if container_id is not None and CONTAINER_ID_RE.fullmatch(container_id) is None:
        inventory_valid = False

    name_present = any(name == container_name for _, name in entries)
    id_present = bool(container_id) and any(cid == container_id for cid, _ in entries)
    verified_absent = (
        inspect_returncode != 0
        and list_returncode == 0
        and inventory_valid
        and not name_present
        and not id_present
    )
    return {
        "container_absent": verified_absent,
        "inspect_returncode": inspect_returncode,
        "container_list_returncode": list_returncode,
        "container_list_valid": inventory_valid,
        "container_name_present": name_present,
        "container_id_present": id_present,
        "container_list_entry_count": len(entries),
    }


def first_post_action_fall(
    serialized_trace: str,
    *,
    action_monotonic_ns: int,
    post_reset_upright_anchor_monotonic_ns: int | None = None,
) -> dict[str, Any] | None:
    """Return the first unsafe physical odometry sample after the Reset anchor.

    The simulator can deliver queued pre-Reset odometry after the SD service
    request. Those samples cannot establish a post-SD fall. When the parent
    has independently observed a fresh, upright post-Reset odometry anchor,
    fence this detector at that anchor while retaining the unchanged root
    height and tilt thresholds for every later sample.
    """
    observation_start_ns = int(action_monotonic_ns)
    if post_reset_upright_anchor_monotonic_ns is not None:
        observation_start_ns = max(
            observation_start_ns, int(post_reset_upright_anchor_monotonic_ns)
        )
    rows: list[dict[str, Any]] = []
    for line in serialized_trace.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("event") == "state"
            and row.get("topic_key") == "odom"
            and int(row.get("wall_monotonic_ns", 0)) >= observation_start_ns
        ):
            rows.append(row)
    rows.sort(key=lambda row: int(row.get("wall_monotonic_ns", 0)))
    for row in rows:
        data = row.get("data", {})
        try:
            tilt = max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"])))
            unsafe = float(data["z"]) < 0.52 or tilt >= 15.0
        except (KeyError, TypeError, ValueError):
            unsafe = True
        if unsafe:
            return {
                "wall_utc": row.get("wall_utc"),
                "wall_monotonic_ns": int(row.get("wall_monotonic_ns", 0)),
                "observation_start_monotonic_ns": observation_start_ns,
                "odom": data,
                "reason": "post-action odometry crossed the unchanged root-height or tilt gate",
            }
    return None


def build_run38_foundationwalk_knee_scale_overlay(
    source_bytes: bytes,
    *,
    expected_source_sha256: str = RUN38_FOUNDATIONWALK_CONFIG_SHA256,
) -> tuple[bytes, dict[str, Any]]:
    """Derive the one Run38-only knee-scale counterfactual from its pinned source.

    Only action_scale entries 9 and 10 are changed, symmetrically, from 0.25
    to 0.19. The ONNX policy, all other configuration values, the robot model,
    MC binary, and the source joint limits remain outside this overlay.
    """
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "FoundationWalk source config SHA-256 differs from the Run38 baseline: "
            f"{source_sha256}"
        )
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("FoundationWalk config is not UTF-8") from exc
    try:
        import yaml

        baseline = yaml.safe_load(source_text)
    except Exception as exc:
        raise ValueError(f"FoundationWalk config YAML could not be parsed: {exc}") from exc
    if not isinstance(baseline, Mapping):
        raise ValueError("FoundationWalk config root must be a mapping")
    base_config = baseline.get("BaseConfig")
    if not isinstance(base_config, Mapping):
        raise ValueError("FoundationWalk config lacks BaseConfig")
    action_seq = base_config.get("action_seq")
    action_scale = base_config.get("action_scale")
    if not isinstance(action_seq, list) or len(action_seq) != 29:
        raise ValueError("Run38 FoundationWalk action sequence must contain 29 joints")
    if action_seq[9:11] != ["left_knee_joint", "right_knee_joint"]:
        raise ValueError("Run38 action indices 9 and 10 are not the left/right knees")
    if not isinstance(action_scale, list) or len(action_scale) != 29:
        raise ValueError("Run38 FoundationWalk action_scale must contain 29 values")
    if any(float(action_scale[index]) != RUN38_BASE_KNEE_ACTION_SCALE
           for index in RUN38_DIAGNOSTIC_KNEE_ACTION_INDICES):
        raise ValueError("Run38 source knee action scales are not both 0.25")

    matches = list(re.finditer(
        r"(?m)^(?P<prefix>\s*action_scale:\s*)\[(?P<values>[^\]]*)\](?P<suffix>\s*)$",
        source_text,
    ))
    if len(matches) != 1:
        raise ValueError("expected exactly one inline action_scale list in the pinned config")
    match = matches[0]
    tokens = [token.strip() for token in match.group("values").split(",")]
    if len(tokens) != 29 or any(not token for token in tokens):
        raise ValueError("inline action_scale list must contain exactly 29 values")
    for index in RUN38_DIAGNOSTIC_KNEE_ACTION_INDICES:
        if float(tokens[index]) != RUN38_BASE_KNEE_ACTION_SCALE:
            raise ValueError("text and parsed knee action scales disagree")
        tokens[index] = f"{RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE:.2f}"
    replacement = (
        match.group("prefix") + "[" + ", ".join(tokens) + "]" + match.group("suffix")
    )
    overlay_text = source_text[:match.start()] + replacement + source_text[match.end():]
    try:
        derived = yaml.safe_load(overlay_text)
    except Exception as exc:
        raise ValueError(f"derived FoundationWalk config YAML is invalid: {exc}") from exc
    expected = copy.deepcopy(baseline)
    expected["BaseConfig"]["action_scale"][9:11] = [
        RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE,
        RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE,
    ]
    if derived != expected:
        raise ValueError("derived config changed values outside the two knee action scales")
    overlay_bytes = overlay_text.encode("utf-8")
    return overlay_bytes, {
        "schema": "robotsim.issue80.foundationwalk-knee-scale-diagnostic.v1",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_CONFIG",
        "source_sha256": source_sha256,
        "overlay_sha256": hashlib.sha256(overlay_bytes).hexdigest(),
        "changed_values": [
            {"action_index": index, "joint": action_seq[index],
             "source": RUN38_BASE_KNEE_ACTION_SCALE,
             "diagnostic": RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE}
            for index in RUN38_DIAGNOSTIC_KNEE_ACTION_INDICES
        ],
        "all_other_yaml_values_unchanged": True,
        "trained_policy_weights_changed": False,
        "policy_output_config_changed": True,
        "vendor_model_or_joint_limits_changed": False,
    }


def build_run41_foundationwalk_min_run_overlay(
    source_bytes: bytes,
    *,
    expected_source_sha256: str = RUN38_FOUNDATIONWALK_CONFIG_SHA256,
) -> tuple[bytes, dict[str, Any]]:
    """Keep Run40's knee scale and lower only the observed min_run_velx floor."""
    knee_overlay, manifest = build_run38_foundationwalk_knee_scale_overlay(
        source_bytes,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        import yaml

        baseline = yaml.safe_load(source_bytes.decode("utf-8"))
        intermediate = yaml.safe_load(knee_overlay.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Run41 FoundationWalk YAML could not be parsed: {exc}") from exc
    foundationwalk = baseline.get("FoundationWalkConfig", {}).get("foundationwalk", {})
    if float(foundationwalk.get("min_run_velx", float("nan"))) != RUN41_BASE_MIN_RUN_VELX:
        raise ValueError("Run41 source min_run_velx is not the measured 0.5 baseline")
    pattern = re.compile(
        r"(?m)^(?P<prefix>\s*min_run_velx:\s*)(?P<value>[^\s#]+)(?P<suffix>\s*(?:#.*)?)$"
    )
    text = knee_overlay.decode("utf-8")
    matches = list(pattern.finditer(text))
    if len(matches) != 1 or float(matches[0].group("value")) != RUN41_BASE_MIN_RUN_VELX:
        raise ValueError("Run41 expected one min_run_velx value of 0.5 in the derived knee config")
    match = matches[0]
    replacement = (
        match.group("prefix")
        + f"{RUN41_DIAGNOSTIC_MIN_RUN_VELX:.1f}"
        + match.group("suffix")
    )
    text = text[:match.start()] + replacement + text[match.end():]
    overlay_bytes = text.encode("utf-8")
    try:
        derived = yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"Run41 derived FoundationWalk YAML is invalid: {exc}") from exc
    expected = copy.deepcopy(baseline)
    expected["BaseConfig"]["action_scale"][9:11] = [
        RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE,
        RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE,
    ]
    expected["FoundationWalkConfig"]["foundationwalk"]["min_run_velx"] = (
        RUN41_DIAGNOSTIC_MIN_RUN_VELX
    )
    if derived != expected or intermediate["FoundationWalkConfig"]["foundationwalk"]["min_run_velx"] != RUN41_BASE_MIN_RUN_VELX:
        raise ValueError("Run41 overlay changed values outside knee action scales and min_run_velx")
    manifest.update({
        "schema": "robotsim.issue80.foundationwalk-knee-minrun-diagnostic.v1",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_CONFIG",
        "changed_values": manifest["changed_values"] + [{
            "config_path": "FoundationWalkConfig.foundationwalk.min_run_velx",
            "source": RUN41_BASE_MIN_RUN_VELX,
            "diagnostic": RUN41_DIAGNOSTIC_MIN_RUN_VELX,
            "semantics": "empirical floor hypothesis; vendor field meaning remains unverified",
        }],
        "overlay_sha256": hashlib.sha256(overlay_bytes).hexdigest(),
        "all_other_yaml_values_unchanged": True,
        "command_speed_m_s": 0.112,
        "trained_policy_weights_changed": False,
        "policy_output_config_changed": True,
        "additional_safety_guard": {
            "measured_root_planar_speed_m_s": 0.2,
            "basis": "same as the existing maximum authorized forward command speed",
        },
    })
    return overlay_bytes, manifest


def classify_run38_diagnostic_outcome(baseline_classification: str) -> str:
    """Keep a derived config result distinct from original-config acceptance."""
    if baseline_classification in (
        "SIMULATION_ONLY_DIAGNOSTIC_TURNING_PASS",
        "SIMULATION_ONLY_DIAGNOSTIC_TURNING_FAIL",
        "SIMULATION_ONLY_DIAGNOSTIC_TURNING_BLOCKED",
        "SIMULATION_ONLY_RIGHT_TURN_AND_STOP_PASS",
        "SIMULATION_ONLY_RIGHT_TURN_AND_STOP_BLOCKED",
        "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_PASS",
        "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_BLOCKED",
        "SIMULATION_ONLY_SHORT_ROUTE_TRANSIT_AND_RIGHT_TURN_PASS",
        "SIMULATION_ONLY_SHORT_ROUTE_INTEGRATION_BLOCKED",
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_PASS",
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_FAIL",
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
        "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_PASS",
        "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_BLOCKED",
        "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_DISTANCE_BASELINE_PASS",
        "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_DISTANCE_BASELINE_BLOCKED",
        "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET_PASS",
        "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET_BLOCKED",
        "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_DIAGNOSTIC_PASS",
        "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_DIAGNOSTIC_BLOCKED",
    ):
        return baseline_classification
    if baseline_classification == "FIRST_PHYSICAL_WALKING_PASS":
        return "SIMULATION_ONLY_DIAGNOSTIC_MOTION_AND_STOP_OBSERVED"
    if baseline_classification == "FIRST_PHYSICAL_WALKING_MOTION_CONFIRMED_STOP_UNVERIFIED":
        return "SIMULATION_ONLY_DIAGNOSTIC_MOTION_STOP_UNVERIFIED"
    if baseline_classification == "FIRST_PHYSICAL_WALKING_FAIL":
        return "SIMULATION_ONLY_DIAGNOSTIC_SAFETY_FAILURE"
    return "SIMULATION_ONLY_DIAGNOSTIC_NO_WALKING_PASS"


def copy_verified_mc_rl_debug_overlay_bundle(
    source_dir: str | Path,
    evidence_dir: str | Path,
    *,
    approved_root: str | Path,
) -> dict[str, Any]:
    """Copy the previously built schema-matched diagnostic overlay without mutation.

    The copied package is generated only from the version-matched Run38 evidence;
    live runtime introspection is still repeated by ``verify_mc_rl_debug_overlay``.
    """
    approved = Path(approved_root).resolve(strict=True)
    source = Path(source_dir).resolve(strict=True)
    evidence = Path(evidence_dir).resolve(strict=True)
    if not source.is_relative_to(approved):
        raise ValueError("McRLDebug overlay source must remain beneath the approved Issue 80 evidence root")
    if source == evidence or source.is_relative_to(evidence):
        raise ValueError("McRLDebug overlay source and destination must be separate runs")
    source_manifest_path = source / "artifact_paths.json"
    if source_manifest_path.exists() and (
        source_manifest_path.is_symlink() or not source_manifest_path.is_file()
    ):
        raise ValueError("McRLDebug overlay source artifact manifest is not a regular file")
    source_manifest_sha256 = None
    if source_manifest_path.is_file():
        try:
            json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("McRLDebug overlay source artifact manifest is invalid") from exc
        source_manifest_sha256 = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()

    source_files = (
        Path("mc_rl_debug_schema.json"),
        Path("MC_RL_DEBUG_SCHEMA_PROVENANCE.json"),
        Path("build_mc_debug_overlay.sh"),
    )
    overlay_subtrees = (
        Path("mc_debug_overlay/src"),
        Path("mc_debug_overlay/install"),
    )
    copied_hashes: dict[str, str] = {}

    def copy_regular_file(relative: Path) -> None:
        origin = source / relative
        target = evidence / relative
        if origin.is_symlink() or not origin.is_file():
            raise ValueError(f"McRLDebug overlay input is missing or not a regular file: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = origin.read_bytes()
        with target.open("xb") as stream:
            stream.write(payload)
        copied_hashes[relative.as_posix()] = hashlib.sha256(payload).hexdigest()

    for relative in source_files:
        copy_regular_file(relative)
    for relative_root in overlay_subtrees:
        origin_root = source / relative_root
        target_root = evidence / relative_root
        if origin_root.is_symlink() or not origin_root.is_dir() or target_root.exists():
            raise ValueError(f"McRLDebug overlay subtree is missing or already exists: {origin_root}")
        for origin in origin_root.rglob("*"):
            if origin.is_symlink():
                raise ValueError(f"McRLDebug overlay subtree contains a symlink: {origin}")
        shutil.copytree(origin_root, target_root)
        for copied in target_root.rglob("*"):
            if copied.is_file():
                relative = copied.relative_to(evidence).as_posix()
                copied_hashes[relative] = hashlib.sha256(copied.read_bytes()).hexdigest()

    if "mc_debug_overlay/install/setup.bash" not in copied_hashes:
        raise ValueError("McRLDebug overlay source does not contain a built install/setup.bash")
    tree_descriptor = "\n".join(
        f"{name} {digest}" for name, digest in sorted(copied_hashes.items())
    ).encode("utf-8")
    return {
        "schema": "robotsim.issue80.mcrldebug-overlay-copy.v1",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY",
        "source_evidence_dir": str(source),
        "source_artifact_manifest_present": source_manifest_sha256 is not None,
        "source_artifact_manifest_sha256": source_manifest_sha256,
        "copied_file_sha256": copied_hashes,
        "copied_tree_sha256": hashlib.sha256(tree_descriptor).hexdigest(),
        "runtime_mc_and_sim_library_hashes_reverified_before_capture": True,
    }


class BoundedRun:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.evidence = Path(args.evidence_dir).resolve(strict=True)
        self.route_config_sha256: str | None = None
        self.local01_scene_mode = bool(getattr(args, "local01_scene_world_poses", None))
        self.local01_scene_bundle: dict[str, Any] | None = None
        self.local01_scene_source: dict[str, Any] | None = None
        self.local01_scene_readonly_mounts: list[tuple[Path, str]] = []
        self.aligned_start_table_approach = bool(
            getattr(args, "aligned_start_table_approach", False)
        )
        self.native_x2_timestep = bool(getattr(args, "native_x2_timestep", False))
        self.synchronized_paused_initialization = bool(
            getattr(args, "synchronized_paused_initialization", False)
        )
        self.static_nominal_start_initialization = bool(
            getattr(args, "static_nominal_start_initialization", False)
        )
        self.mc_prearm_before_simulator = bool(
            getattr(args, "mc_prearm_before_simulator", False)
        )
        self.direct_stand_from_passive = bool(
            getattr(args, "direct_stand_from_passive", False)
        )
        lidar_source = getattr(args, "raicom_lidar_overlay_source", None)
        self.lidar_overlay_source = Path(lidar_source).resolve(strict=True) if lidar_source else None
        self.lidar_overlay_dir: Path | None = None
        if self.lidar_overlay_source is not None:
            source_overlay = self.lidar_overlay_source / "lidar_overlay"
            target_overlay = self.evidence / "lidar_overlay"
            if source_overlay.is_symlink() or not source_overlay.is_dir() or target_overlay.exists():
                raise ValueError("LiDAR overlay source/target directory is unsafe or already exists")
            for path in source_overlay.rglob("*"):
                if path.is_symlink() or (path.is_file() and not path.is_relative_to(source_overlay)):
                    raise ValueError("LiDAR overlay source contains an unsafe path")
            shutil.copytree(source_overlay, target_overlay)
            self.lidar_overlay_dir = target_overlay
        # Exception-only nested mounts for a known MC-generated runtime dump.
        # The base vendor runtime stays read-only, and every source must live
        # in the evidence directory.
        self.runtime_writeback_mounts: list[tuple[Path, str]] = []
        self.mc_prearm_writeback_manifest: dict[str, Any] | None = None
        self.mc_prearm_status: dict[str, Any] | None = None
        if getattr(args, "table_approach_mission", None):
            preflight = getattr(args, "configured_table_approach_preflight", None)
            if not isinstance(preflight, Mapping) or not self.local01_scene_mode:
                raise ValueError("table-approach mission requires its pinned Local01 scene preflight")
            source = preflight.get("source")
            mission = preflight.get("mission")
            if not isinstance(source, Mapping) or not isinstance(mission, Mapping):
                raise ValueError("table-approach preflight did not retain source and mission mappings")
            self.local01_scene_source = dict(source)
            route_bytes = (
                json.dumps(mission, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8")
            staged = self.evidence / "route_config.json"
            with staged.open("xb") as stream:
                stream.write(route_bytes)
            self.route_config_sha256 = hashlib.sha256(route_bytes).hexdigest()
            plan_source = Path(str(getattr(args, "table_approach_static_plan", "")))
            staged_plan = self.evidence / "table_approach_static_plan.json"
            shutil.copyfile(plan_source, staged_plan)
            self.local01_scene_bundle = prepare_local01_scene_runtime_bundle(
                self.local01_scene_source,
                self.evidence,
                runtime_root=RUNTIME,
                aligned_start=self.aligned_start_table_approach,
                native_x2_timestep=self.native_x2_timestep,
                synchronized_nominal_initialization=self.synchronized_paused_initialization,
                static_nominal_start_initialization=self.static_nominal_start_initialization,
            )
            # Retain the existing name/physics-equivalence evidence used by
            # the Local01 runtime wrapper.  The separate full table-plan file
            # remains the authority for the longer safe-prefix geometry.
            if self.aligned_start_table_approach:
                aligned_plan_segments = [
                    item for item in preflight["plan"].get("segments", [])
                    if isinstance(item, Mapping)
                    and item.get("status") == "PASS_STATIC_ONLY"
                    and (item.get("from"), item.get("to")) in {
                        ("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH"),
                    }
                ]
                if {
                    (item.get("from"), item.get("to"))
                    for item in aligned_plan_segments
                } != {("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH")}:
                    raise ValueError("aligned-start table plan lacks both source-grounded safe-prefix segments")
                short_static = {
                    "schema": "robotsim.issue80.aligned-start-runtime-static-preflight.v1",
                    "status": "PASS_STATIC_SCENE_CLEARANCE",
                    "classification": "STATIC_COMPILED_ALIGNED_START_WRAPPER_NO_DYNAMICS",
                    "mj_step_calls": 0,
                    "source_local01_commit": source["source_commit"],
                    "source_scene_world_poses_sha256": source["source_hashes"]["scene_world_poses"],
                    "native_runtime_name_contract": self.local01_scene_bundle[
                        "generated_runtime_name_contract"
                    ],
                    "compiled_name_only_equivalence": self.local01_scene_bundle[
                        "generated_wrapper_physics_equivalence"
                    ],
                    "native_robot_equivalence": self.local01_scene_bundle[
                        "generated_native_robot_equivalence"
                    ],
                    "scene_wrapper_change_classification": self.local01_scene_bundle[
                        "scene_wrapper_change_classification"
                    ],
                    "native_x2_timestep": self.local01_scene_bundle["native_x2_timestep"],
                    # The aligned wrapper is compiled-equivalent in all robot
                    # collision geometry except the pre-physics root yaw. The
                    # established bound therefore remains applicable, while
                    # clearance is taken from the matching aligned table plan.
                    "robot_collision_footprint_radial_upper_bound_m": ROUTE_QPOS0_FOOTPRINT_BOUND_M,
                    "verified_local_clear_radius_m": ROUTE_LOCAL_CLEAR_RADIUS_M,
                    "minimum_swept_aabb_clearance_lower_bound_m": min(
                        float(item["tracking_inflated_aabb_lower_bound_m"])
                        for item in aligned_plan_segments
                    ),
                    "static_sample_count": sum(
                        int(item["samples"]) for item in aligned_plan_segments
                    ),
                    "footprint_and_clearance_provenance": (
                        "unchanged X2 collision geometry verified by the aligned wrapper equivalence; "
                        "safe-prefix clearance and samples are derived from the matching aligned table plan"
                    ),
                    "initial_full_body_static_clearance": "AUTHORITY_IS_MATCHING_TABLE_APPROACH_STATIC_PLAN",
                    "limitations": [
                        "This wrapper preflight validates native names and compiled invariants only.",
                        "The matching aligned-start table plan owns full-body initial and corridor clearance.",
                    ],
                }
            else:
                short_static = validate_short_route_static_scene(
                    Path(str(source["repository_root"])) / "simulation/mujoco/scenes/raicom2026",
                    build_runtime_raw_odom_short_route_config(source),
                )
            if short_static.get("status") != "PASS_STATIC_SCENE_CLEARANCE":
                raise ValueError("Local01 wrapper short static preflight no longer passes")
            with (self.evidence / "local01_scene/static_scene_preflight.json").open("x", encoding="utf-8") as stream:
                json.dump(short_static, stream, indent=2, allow_nan=False)
                stream.write("\n")
            route_manifest = {
                "schema": "robotsim.issue80.table-approach-mission-input.v1",
                "classification": mission["classification"],
                "aligned_start_table_approach": self.aligned_start_table_approach,
                "native_x2_timestep": self.native_x2_timestep,
                "source_local01_commit": source["source_commit"],
                "source_scene_world_poses_sha256": source["source_hashes"]["scene_world_poses"],
                "derived_route_sha256": self.route_config_sha256,
                "source_static_plan_sha256": preflight["plan_sha256"],
                "staged_route_path": str(staged),
                "staged_static_plan_path": str(staged_plan),
                "scene_bundle_classification": self.local01_scene_bundle["classification"],
                "runtime_reset_odom_observed": False,
                "physical_table_docking": "NOT_TARGETED_BY_THIS_TABLE_APPROACH_MISSION",
            }
            with (self.evidence / "route_config_input.json").open("x", encoding="utf-8") as stream:
                json.dump(route_manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            runtime_scene_filename = str(self.local01_scene_bundle["runtime_scene_filename"])
            runtime_config_filename = str(self.local01_scene_bundle["runtime_simulator_config_filename"])
            self.local01_scene_readonly_mounts = [
                (self.evidence / "route_config.json", "/workspace/evidence/route_config.json"),
                (self.evidence / "route_config_input.json", "/workspace/evidence/route_config_input.json"),
                (staged_plan, "/workspace/evidence/table_approach_static_plan.json"),
                (self.evidence / "local01_scene/scene_world_poses.json", "/workspace/evidence/local01_scene/scene_world_poses.json"),
                (self.evidence / "local01_scene/route_waypoints.json", "/workspace/evidence/local01_scene/route_waypoints.json"),
                (self.evidence / "local01_scene/scene_workspace.xml", "/workspace/evidence/local01_scene/scene_workspace.xml"),
                (self.evidence / "local01_scene/mug_portable.xml", "/workspace/evidence/local01_scene/mug_portable.xml"),
                (self.evidence / "local01_scene" / runtime_scene_filename, "/workspace/evidence/local01_scene/" + runtime_scene_filename),
                (self.evidence / "local01_scene/mug_portable_runtime.xml", "/workspace/evidence/local01_scene/mug_portable_runtime.xml"),
                (self.evidence / "local01_scene" / runtime_config_filename, "/workspace/evidence/local01_scene/" + runtime_config_filename),
                (self.evidence / "local01_scene/runtime_bundle_manifest.json", "/workspace/evidence/local01_scene/runtime_bundle_manifest.json"),
                (self.evidence / "local01_scene/static_scene_preflight.json", "/workspace/evidence/local01_scene/static_scene_preflight.json"),
            ]
        elif getattr(args, "route_config", None):
            source = Path(args.route_config)
            if source.is_symlink() or not source.is_file():
                raise ValueError("route config must be a regular, non-symlink input file")
            route_pose_input = load_waypoint_docking_pose_input(source)
            route_bytes = source.read_bytes()
            configured_sha = getattr(self.args, "configured_waypoint_docking_config_sha256", None)
            if configured_sha is not None and configured_sha != hashlib.sha256(route_bytes).hexdigest():
                raise ValueError("route config changed after its pre-runtime validation")
            staged = self.evidence / "route_config.json"
            with staged.open("xb") as stream:
                stream.write(route_bytes)
            self.route_config_sha256 = hashlib.sha256(route_bytes).hexdigest()
            route_input = {
                "schema": "robotsim.issue80.route-config-input.v1",
                "classification": "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE",
                "source_sha256": self.route_config_sha256,
                "staged_path": str(staged),
                "frame_id": "odom",
                "route_id": route_pose_input["route_id"],
                "step_ids": ["WAYPOINT", "TABLE_DOCKING"],
                "not_a_motion_or_safety_gate": True,
            }
            with (self.evidence / "route_config_input.json").open("x", encoding="utf-8") as stream:
                json.dump(route_input, stream, indent=2)
                stream.write("\n")
        elif self.local01_scene_mode:
            local01 = getattr(self.args, "configured_local01_scene_preflight", None)
            if not isinstance(local01, Mapping):
                raise ValueError("Local01 scene route lacks its pinned static preflight result")
            source_path = Path(self.args.local01_scene_world_poses)
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError("Local01 scene pose source is missing or unsafe")
            source = load_local01_scene_inputs(source_path, require_clean_commit=True)
            self.local01_scene_source = source
            if source["source_hashes"]["scene_world_poses"] != local01.get("source_sha256"):
                raise ValueError("Local01 scene pose source changed after static preflight")
            route_input = local01.get("route_input")
            if not isinstance(route_input, Mapping):
                raise ValueError("Local01 preflight did not produce a short route input")
            validate_route_input = dict(route_input)
            route_bytes = (
                json.dumps(validate_route_input, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            staged = self.evidence / "route_config.json"
            with staged.open("xb") as stream:
                stream.write(route_bytes)
            self.route_config_sha256 = hashlib.sha256(route_bytes).hexdigest()
            self.local01_scene_bundle = prepare_local01_scene_runtime_bundle(
                source, self.evidence, runtime_root=RUNTIME,
            )
            with (self.evidence / "local01_scene_static_preflight.json").open("x", encoding="utf-8") as stream:
                json.dump(local01["static_scene_preflight"], stream, indent=2, allow_nan=False)
                stream.write("\n")
            route_manifest = {
                "schema": "robotsim.issue80.local01-short-route-input.v1",
                "classification": "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING",
                "source_local01_commit": source["source_commit"],
                "source_scene_world_poses_sha256": source["source_hashes"]["scene_world_poses"],
                "derived_route_sha256": self.route_config_sha256,
                "staged_route_path": str(staged),
                "scene_bundle_classification": self.local01_scene_bundle["classification"],
                "runtime_reset_odom_observed": False,
                "first_nonzero_command_gate": "fresh reset odometry must match transformed START within 0.02 m/rad",
                "physical_table_docking": "NOT_TARGETED_BY_THIS_SHORT_ROUTE",
            }
            with (self.evidence / "route_config_input.json").open("x", encoding="utf-8") as stream:
                json.dump(route_manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            self.local01_scene_readonly_mounts = [
                (self.evidence / "route_config.json", "/workspace/evidence/route_config.json"),
                (self.evidence / "route_config_input.json", "/workspace/evidence/route_config_input.json"),
                (self.evidence / "local01_scene/scene_world_poses.json", "/workspace/evidence/local01_scene/scene_world_poses.json"),
                (self.evidence / "local01_scene/route_waypoints.json", "/workspace/evidence/local01_scene/route_waypoints.json"),
                (self.evidence / "local01_scene/scene_workspace.xml", "/workspace/evidence/local01_scene/scene_workspace.xml"),
                (self.evidence / "local01_scene/mug_portable.xml", "/workspace/evidence/local01_scene/mug_portable.xml"),
                (self.evidence / "local01_scene/scene_workspace_runtime.xml", "/workspace/evidence/local01_scene/scene_workspace_runtime.xml"),
                (self.evidence / "local01_scene/mug_portable_runtime.xml", "/workspace/evidence/local01_scene/mug_portable_runtime.xml"),
                (self.evidence / "local01_scene/simulator_default_local01_scene.yaml", "/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml"),
                (self.evidence / "local01_scene/runtime_bundle_manifest.json", "/workspace/evidence/local01_scene/runtime_bundle_manifest.json"),
                (self.evidence / "local01_scene/static_scene_preflight.json", "/workspace/evidence/local01_scene/static_scene_preflight.json"),
            ]
        self.container = args.container
        self.source = "node.locomotion_velocity"
        self.started = time.monotonic()
        self.deadline = self.started + args.max_runtime_s
        self.step = 0
        self.steps: list[dict[str, Any]] = []
        self.container_created = False
        self.container_id: str | None = None
        self.network_mode: str | None = None
        self.cleanup_evidence: dict[str, Any] = {
            "attempted": False,
            "normal_path": "BoundedRun.execute() finally calls cleanup().",
            "process_crash_or_sigkill_cleanup_guaranteed": False,
            "operations": [],
        }
        self.reset_classification = "NOT_ATTEMPTED"
        self.m1_result: dict[str, Any] | None = None
        self.mc_rl_debug_capture: dict[str, Any] | None = None
        self.motion_video_process: subprocess.Popen[str] | None = None
        self.motion_video_capture: dict[str, Any] = {"status": "NOT_REQUESTED"}
        self.native_mjdata_capture = bool(getattr(args, "native_mjdata_capture", False))
        self.native_mjdata_capture_interval_steps = int(
            getattr(args, "native_mjdata_capture_interval_steps", DEFAULT_INTERVAL_STEPS)
        )
        self.native_mjdata_overlay: dict[str, Any] | None = None
        self.native_mjdata_smoke: dict[str, Any] | None = None
        self.foundationwalk_config_overlay: Path | None = None
        self.foundationwalk_config_overlay_manifest: dict[str, Any] | None = None
        self.mc_rl_debug_overlay_manifest: dict[str, Any] | None = None
        self.mc_rl_debug_schema_verification: dict[str, Any] | None = None
        self.reset_analysis: dict[str, Any] | None = None
        self.before_reset_image: dict[str, Any] | None = None
        self.reset_dispatch_payload: dict[str, Any] | None = None
        self.immediate_reset_pose: dict[str, Any] | None = None
        self.sd_result: dict[str, Any] | None = None
        self.outcome_classification = "STANDING_CONTROL_NOT_READY"
        self.fault: str | None = None

    def remaining(self) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise RunDeadline("180-second wall-time budget expired")
        return value

    def save_command(self, name: str, command: list[str], result: subprocess.CompletedProcess[str] | None,
                     started: float, ended: float, error: str | None = None) -> None:
        self.step += 1
        record = {
            "step": self.step,
            "name": name,
            "command": command,
            "started_wall_utc": datetime.fromtimestamp(time.time() - (time.monotonic() - started), timezone.utc).isoformat(),
            "started_monotonic_s": started,
            "ended_monotonic_s": ended,
            "wall_utc_basis": "reconstructed_at_record_time; use monotonic fields for elapsed ordering",
            "duration_s": ended - started,
            "returncode": result.returncode if result else None,
            "error": error,
        }
        stdout_path = self.evidence / f"step-{self.step:02d}-{name}.stdout.txt"
        stderr_path = self.evidence / f"step-{self.step:02d}-{name}.stderr.txt"
        with stdout_path.open("x", encoding="utf-8") as f:
            f.write(result.stdout if result else "")
        with stderr_path.open("x", encoding="utf-8") as f:
            f.write(result.stderr if result else error or "")
        record["stdout_path"] = str(stdout_path)
        record["stderr_path"] = str(stderr_path)
        self.steps.append(record)
        with (self.evidence / "execution.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def run(self, name: str, command: list[str], *, timeout_s: float = 8.0,
            check: bool = True) -> subprocess.CompletedProcess[str]:
        available = self.remaining()
        timeout = min(timeout_s, available)
        started = time.monotonic()
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            ended = time.monotonic()
            partial = subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
            self.save_command(name, command, partial, started, ended, "subprocess timeout")
            raise RunDeadline(f"{name} timed out") from exc
        ended = time.monotonic()
        self.save_command(name, command, result, started, ended)
        if check and result.returncode != 0:
            raise RuntimeError(f"{name} returned {result.returncode}: {result.stderr[-500:]}")
        return result

    def docker_exec(self, name: str, shell: str, *, timeout_s: float = 8.0,
                    detached: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = ["docker", "exec"]
        if detached:
            cmd.append("-d")
        cmd += [self.container, "bash", "-lc", shell]
        return self.run(name, cmd, timeout_s=timeout_s, check=check)

    def prepare_foundationwalk_diagnostic_overlay(self) -> None:
        """Create an evidence-local, immutable Run38 knee-scale diagnostic copy."""
        run38 = bool(getattr(self.args, "run38_knee_scale_diagnostic", False))
        run41 = bool(getattr(self.args, "run41_min_run_diagnostic", False))
        run42 = bool(getattr(self.args, "run42_stop_confirmation_diagnostic", False))
        run43 = bool(getattr(self.args, "run43_zero_only_stop_diagnostic", False))
        run44 = bool(getattr(self.args, "run44_threshold_margin_diagnostic", False))
        run45 = bool(getattr(self.args, "run45_ten_second_gait_diagnostic", False))
        run46 = bool(getattr(self.args, "run46_startup_cruise_threshold_diagnostic", False))
        run48 = bool(getattr(self.args, "run48_low_cruise_baseline_diagnostic", False))
        run49 = bool(getattr(self.args, "run49_smoothstop_diagnostic", False))
        run50 = bool(getattr(self.args, "run50_threshold_respecting_smoothstop_diagnostic", False))
        run53 = bool(getattr(self.args, "run53_turning_diagnostic", False))
        run65_point_goal = run65_point_goal_enabled(self.args)
        run65_right_turn = bool(getattr(self.args, "run65_right_turn_qualification", False))
        run65_route = bool(getattr(self.args, "run65_route_integration", False))
        run65_waypoint_b_only = bool(getattr(self.args, "run65_waypoint_b_only", False))
        run65_waypoint_b_short = bool(
            getattr(self.args, "run65_waypoint_b_short_diagnostic", False)
        )
        run65_waypoint_b_extended = bool(
            getattr(self.args, "run65_waypoint_b_extended_diagnostic", False)
        )
        run65_curved_path = bool(getattr(self.args, "run65_curved_path_diagnostic", False))
        run65_waypoint_docking = bool(
            getattr(self.args, "route_config", None)
            or getattr(self.args, "local01_scene_world_poses", None)
            or getattr(self.args, "table_approach_mission", None)
        )
        run65_local01_short_route = bool(
            getattr(self.args, "local01_scene_world_poses", None)
            and not getattr(self.args, "table_approach_mission", None)
        )
        run65_waypoint_b_mode = (
            run65_waypoint_b_only or run65_waypoint_b_short or run65_waypoint_b_extended
        )
        run65_overlay = run65_point_goal or run65_right_turn or run65_route or run65_waypoint_b_mode or run65_curved_path or run65_waypoint_docking
        selected_variants = sum((run38, run41, run42, run43, run44, run45, run46, run48, run49, run50, run53, run65_point_goal, run65_right_turn, run65_route, run65_waypoint_b_mode, run65_curved_path, run65_waypoint_docking))
        if selected_variants == 0:
            return
        if selected_variants != 1:
            raise ValueError("select exactly one FoundationWalk diagnostic config variant")
        runtime_inputs = {
            "foundationwalk_config": (
                FOUNDATIONWALK_CONFIG_RELATIVE,
                RUN38_FOUNDATIONWALK_CONFIG_SHA256,
            ),
            "foundationwalk_onnx": (
                Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl_models/foundationwalk_v1.onnx"),
                RUN38_FOUNDATIONWALK_ONNX_SHA256,
            ),
            "foundationwalk_runner": (
                Path("mc/lib/libmc_runner_rl_foundationwalk.so.0.0.0"),
                RUN38_FOUNDATIONWALK_RUNNER_SHA256,
            ),
            "action_setting": (
                Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/action_setting.yaml"),
                RUN38_ACTION_SETTING_SHA256,
            ),
            "robot_model": (
                Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/robot_model.yaml"),
                RUN38_ROBOT_MODEL_SHA256,
            ),
            "mc_sim_config": (
                Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"),
                RUN38_MC_SIM_CONFIG_SHA256,
            ),
        }
        runtime_hashes: dict[str, str] = {}
        for name, (relative_path, expected_sha256) in runtime_inputs.items():
            actual_sha256 = hashlib.sha256((RUNTIME / relative_path).read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Run38 {name} provenance mismatch: expected {expected_sha256}, got {actual_sha256}"
                )
            runtime_hashes[name] = actual_sha256
        try:
            import yaml

            action_setting = yaml.safe_load(
                (RUNTIME / runtime_inputs["action_setting"][0]).read_text(encoding="utf-8")
            )
            locomotion_default = action_setting["transition_actions"]["LOCOMOTION_DEFAULT"]
            if (
                len(locomotion_default) != 1
                or locomotion_default[0]["runners"] != [{
                    "name": "rl_foundationwalk",
                    "cfg_path": "./rl/foundationwalk_config.yaml",
                }]
            ):
                raise ValueError("LOCOMOTION_DEFAULT no longer resolves to the Run38 FoundationWalk config")
        except Exception as exc:
            raise RuntimeError(f"Run38 action-setting mapping verification failed: {exc}") from exc
        source = RUNTIME / FOUNDATIONWALK_CONFIG_RELATIVE
        source_bytes = source.read_bytes()
        if run41 or run42 or run43 or run44 or run45 or run46 or run48 or run49 or run50 or run53 or run65_overlay:
            overlay_bytes, manifest = build_run41_foundationwalk_min_run_overlay(source_bytes)
            overlay_name = "foundationwalk_knee019_minrun010.yaml"
        else:
            overlay_bytes, manifest = build_run38_foundationwalk_knee_scale_overlay(source_bytes)
            overlay_name = "foundationwalk_knee_scale_019.yaml"
        overlay = self.evidence / overlay_name
        with overlay.open("xb") as stream:
            stream.write(overlay_bytes)
        manifest.update({
            "source_path": str(source),
            "overlay_path": str(overlay),
            "verified_runtime_input_sha256": runtime_hashes,
            "runtime_image_digest": IMAGE_DIGEST,
            "runtime_mc_binary_modified": False,
            "runtime_onnx_modified": False,
            "source_limits_modified": False,
            "safety_gates_modified": False,
        })
        manifest["command_speed_m_s"] = (
            TURN_FORWARD_COMMAND_M_S if run53
            else POINT_GOAL_FORWARD_COMMAND_M_S if run65_point_goal
            else CURVED_PATH_FORWARD_M_S if run65_curved_path
            else float(self.args.forward_speed) if getattr(self.args, "raicom_low_cruise_profile", False)
            else TURN_FORWARD_COMMAND_M_S if (run65_right_turn or run65_route or run65_waypoint_b_mode or run65_waypoint_docking)
            else float(self.args.forward_speed)
        )
        manifest["command_profile"] = ({
            "shape": "left-right-forward-warmup-arc",
            "forward_velocity_m_s": TURN_FORWARD_COMMAND_M_S,
            "forward_warmup_s": TURN_FORWARD_WARMUP_S,
            "lateral_velocity_m_s": 0.0,
            "angular_velocity_rad_s": [TURN_COMMAND_RAD_S, -TURN_COMMAND_RAD_S],
            "pulse_duration_s_each": TURN_PULSE_S,
            "total_active_command_s": 2.0 * (TURN_FORWARD_WARMUP_S + TURN_PULSE_S),
        } if run53 else {
            "shape": "ground-truth-point-goal",
            "target_distance_m": POINT_GOAL_DISTANCE_M,
            "heading_change_rad": float(getattr(self.args, "heading_change_rad", 0.0)),
            "maximum_heading_change_rad": POINT_GOAL_MAX_HEADING_CHANGE_RAD,
            "minimum_measured_heading_change_rad": TURN_MIN_HEADING_CHANGE_RAD,
            "heading_change_warmup_s": TURN_FORWARD_WARMUP_S,
            "heading_change_warmup_forward_m_s": TURN_FORWARD_COMMAND_M_S,
            "stop_model": "speed_scaled_endpoint_response",
            "endpoint_response_intercept_m": POINT_GOAL_ENDPOINT_RESPONSE_INTERCEPT_M,
            "endpoint_response_time_s": POINT_GOAL_ENDPOINT_RESPONSE_TIME_S,
            "max_angular_command_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
            **run65_point_goal_profile_metadata(self.args),
            "max_lateral_error_m": POINT_GOAL_MAX_LATERAL_ERROR_M,
            "max_overshoot_m": POINT_GOAL_MAX_OVERSHOOT_M,
            "final_tolerance_m": POINT_GOAL_FINAL_TOLERANCE_M,
            "forward_velocity_m_s": POINT_GOAL_FORWARD_COMMAND_M_S,
            "angular_control": "ground_truth_heading_feedback_within_validated_run65_envelope",
            "active_command_limit_s": 10.0,
            "maximum_planar_speed_m_s": 0.2,
            "stop_trigger": "along_track_progress_crosses_speed_scaled_stop_line_or_safe_guard",
            "stop_behavior": "zero-only; wait for fresh automatic STAND readback and physical-stop gate",
            "zero_stop_min_duration_s": 1.0,
            "endpoint_response_basis_attempt_id": "c1883327-5e43-4e6d-bf41-5dac382c9f49",
            "prior_endpoint_error_m": 0.07699227258252467,
        } if run65_point_goal else {
            "shape": "motion-triggered-startup-then-cruise"
            if getattr(self.args, "startup_speed", None) is not None
            else "constant-then-smoothstep-deceleration"
            if getattr(self.args, "run49_smoothstop_diagnostic", False)
            or getattr(self.args, "run50_threshold_respecting_smoothstop_diagnostic", False)
            else "constant",
            "startup_speed_m_s": getattr(self.args, "startup_speed", None),
            "cruise_speed_m_s": float(self.args.forward_speed),
            "transition_displacement_m": getattr(self.args, "startup_transition_displacement", None),
            "startup_timeout_s": getattr(self.args, "startup_transition_timeout_s", 0.0),
            "deceleration_duration_s": 1.0
            if getattr(self.args, "run49_smoothstop_diagnostic", False)
            or getattr(self.args, "run50_threshold_respecting_smoothstop_diagnostic", False) else 0.0,
            "deceleration_end_speed_m_s": 0.112
            if getattr(self.args, "run50_threshold_respecting_smoothstop_diagnostic", False)
            else 0.0 if getattr(self.args, "run49_smoothstop_diagnostic", False) else None,
        })
        if getattr(self.args, "raicom_low_cruise_profile", False):
            manifest["low_cruise_recovery_profile"] = {
                "classification": "SIMULATION_ONLY_RAICOM_LOW_CRUISE_RECOVERY",
                "forward_speed_m_s": float(self.args.forward_speed),
                "basis": (
                    "0.12 m/s produced a measured 0.228394 m/s root-speed safety abort in "
                    "raicom-direct-sd-short-route-motion-20261026T200622Z; the preserved "
                    "0.105 m/s FoundationWalk baseline peaked at 0.143708 m/s"
                ),
                "root_speed_guard_m_s": 0.2,
                "no_vendor_or_safety_parameter_changed": True,
            }
        if run65_right_turn:
            manifest["command_profile"] = {
                "shape": "one-bounded-right-moving-arc",
                "forward_velocity_m_s": TURN_FORWARD_COMMAND_M_S,
                "forward_warmup_s": TURN_FORWARD_WARMUP_S,
                "angular_velocity_rad_s": -TURN_COMMAND_RAD_S,
                "maximum_active_window_s": TURN_FORWARD_WARMUP_S + TURN_PULSE_S,
                "pre_deadline_zero_margin_s": 0.02,
                "maximum_turn_displacement_m": 0.25,
                "empirical_zero-stop_reserve_m": 0.05,
                "displacement-trigger_m": 0.20,
                "stop_behavior": "zero-only; wait for fresh automatic STAND readback and physical-stop gate",
            }
        if run65_route:
            manifest["command_profile"] = {
                "shape": "waypoint-transit-plus-bounded-right-corner",
                "transit_forward_m_s": TURN_FORWARD_COMMAND_M_S,
                "right_turn_forward_m_s": TURN_FORWARD_COMMAND_M_S,
                "right_turn_angular_rad_s": -TURN_COMMAND_RAD_S,
                "pre_deadline_zero_margin_s": 0.02,
                "maximum_turn_displacement_m": 0.25,
                "empirical_zero-stop_reserve_m": 0.05,
                "displacement-trigger_m": 0.20,
                "each_M1_action_window_cap_s": 5.0,
                "second_transit": "OPTIONAL_NOT_EXECUTED_IN_THIS_SHORT_ROUTE_TRIAL",
            }
        if run65_waypoint_b_extended:
            manifest["command_profile"] = {
                "shape": "extended-three-window-transit-to-original-waypoint-B",
                "classification": "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC",
                "waypoint_b_distance_from_start_m": 0.40,
                "position_tolerance_m": WAYPOINT_B_POSITION_TOLERANCE_M,
                "stop_trigger_reserve_m": WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
                "arrival_trigger_radius_m": waypoint_b_arrival_trigger_radius_m(
                    acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
                    stop_reserve_m=WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
                ),
                "transit_forward_m_s": TURN_FORWARD_COMMAND_M_S,
                "per_window_nonzero_limit_s": WAYPOINT_B_WINDOW_MAX_S,
                "maximum_windows": WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS,
                "aggregate_transit_nonzero_limit_s": WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S,
                "route_nonzero_limit_s": 30.0,
                "pre_deadline_zero_margin_s": 0.02,
                "maximum_centerline_lateral_error_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                "maximum_forward_projection_guard_m": WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
                "route_corridor_forward_limit_m": WAYPOINT_B_CORRIDOR_MAX_FORWARD_M,
                "inter_window_requirement": "fresh pose and full composite physical-stop PASS before another window",
                "post_waypoint_actions": "end after original B target; no turn or C waypoint",
            }
        if run65_curved_path:
            manifest["command_profile"] = {
                "shape": "closed_loop_shallow_right_circular_arc",
                "classification": "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC",
                "path_radius_m": CURVED_PATH_RADIUS_M,
                "path_sweep_rad": CURVED_PATH_SWEEP_RAD,
                "expected_path_length_m": CURVED_PATH_RADIUS_M * CURVED_PATH_SWEEP_RAD,
                "nominal_forward_m_s": CURVED_PATH_FORWARD_M_S,
                "nominal_angular_rad_s": -CURVED_PATH_FORWARD_M_S / CURVED_PATH_RADIUS_M,
                "maximum_forward_m_s": CURVED_PATH_FORWARD_M_S,
                "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                "maximum_cross_track_m": CURVED_PATH_MAX_CROSS_TRACK_M,
                "endpoint_tolerance_m": CURVED_PATH_POSITION_TOLERANCE_M,
                "stop_reserve_m": CURVED_PATH_STOP_RESERVE_M,
                "window_cap_s": CURVED_PATH_MAX_WINDOW_S,
                "maximum_windows": CURVED_PATH_MAX_WINDOWS,
                "aggregate_nonzero_limit_s": CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
                "inter_window_requirement": "fresh odom, source, STAND and full physical/controller-target stop PASS before rearm",
                "final_yaw_target": None,
            }
        elif run65_waypoint_docking:
            route = (
                getattr(self.args, "configured_local01_short_route", None)
                if run65_local01_short_route
                else getattr(self.args, "configured_waypoint_docking_route", None)
            )
            if getattr(self.args, "table_approach_mission", None):
                mission = self.args.configured_table_approach_preflight["mission"]
                manifest["command_profile"] = {
                    "shape": "source-grounded-zero-forward-start-alignment-then-two-step-table-approach",
                    "classification": mission["classification"],
                    "mission_id": mission["mission_id"],
                    "frame_id": "odom",
                    "source_frame_id": "raicom_scene_world",
                    "source_local01_commit": self.args.local01_scene_source["source_commit"],
                    "step_ids": mission["execution_order"],
                    "forward_m_s": mission["limits"]["forward_m_s"],
                    "maximum_angular_rad_s": mission["limits"]["angular_rad_s"],
                    "maximum_windows_total": mission["limits"]["max_windows"],
                    "maximum_window_s": mission["limits"]["max_window_s"],
                    "aggregate_nonzero_limit_s": mission["limits"]["max_aggregate_nonzero_s"],
                    "child_runtime_timeout_s": TABLE_APPROACH_CHILD_TIMEOUT_S,
                    "minimum_parent_runtime_s": TABLE_APPROACH_MIN_PARENT_RUNTIME_S,
                    "zero_forward_alignment": mission["initial_alignment"],
                    "zero_stop_gate": "unchanged composite physical and joint-target gate after every bounded window",
                    "post_stand_observation_s": 7.0,
                    "terminal_state": "TABLE_APPROACH_STOPPED_NO_TABLE_DOCKING_OR_MANIPULATOR_HANDOFF",
                    "table_docking_reference": mission["table_docking_reference"],
                }
            elif run65_local01_short_route:
                manifest["command_profile"] = {
                    "shape": "local01-start-heading-two-target-short-forward-route",
                    "classification": "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING",
                    "route_id": route.route_id if route is not None else None,
                    "frame_id": "odom",
                    "source_frame_id": "raicom_scene_world",
                    "source_local01_commit": self.args.local01_scene_source["source_commit"],
                    "step_ids": [step.waypoint_id for step in route.steps] if route is not None else [],
                    "forward_m_s": 0.12,
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    "maximum_windows_total": 2,
                    "maximum_window_s": 5.0,
                    "aggregate_nonzero_limit_s": 10.0,
                    "validated_envelope_ceiling_windows": 3,
                    "validated_envelope_ceiling_aggregate_nonzero_s": 15.0,
                    "maximum_cross_track_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                    "zero_stop_gate": "unchanged composite physical and joint-target gate after fresh STAND",
                    "post_stand_observation_s": 7.0,
                    "terminal_state": "SHORT_ROUTE_STOPPED_WITHOUT_TABLE_OR_MANIPULATOR_HANDOFF",
                    "safe_table_docking_reference": self.args.configured_local01_short_route_input["safe_table_docking_reference"],
                }
            else:
                manifest["command_profile"] = {
                    "shape": "ground-truth-two-step-waypoint-and-table-docking-route",
                    "classification": "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE",
                    "route_id": route.route_id if route is not None else None,
                    "frame_id": "odom",
                    "step_ids": [step.waypoint_id for step in route.steps] if route is not None else [],
                    "forward_m_s": 0.12,
                    "maximum_angular_rad_s": POINT_GOAL_MAX_ANGULAR_COMMAND_RAD_S,
                    "maximum_windows_total": 3,
                    "maximum_window_s": 5.0,
                    "aggregate_nonzero_limit_s": 15.0,
                    "maximum_cross_track_m": WAYPOINT_B_MAX_CROSS_TRACK_M,
                    "zero_stop_gate": "unchanged composite physical and joint-target gate after fresh STAND",
                    "post_stand_observation_s": 7.0,
                    "terminal_state": "WAITING_FOR_DOCKING_WITHOUT_MANIPULATOR_HANDOFF",
                }
        elif run65_waypoint_b_short:
            manifest["command_profile"] = {
                "shape": "short-waypoint-diagnostic-only",
                "classification": "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC",
                "waypoint_distance_from_start_m": WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M,
                "position_tolerance_m": WAYPOINT_B_POSITION_TOLERANCE_M,
                "stop_trigger_reserve_m": WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
                "arrival_trigger_radius_m": waypoint_b_arrival_trigger_radius_m(
                    acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
                    stop_reserve_m=WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
                ),
                "original_waypoint_b_distance_m": 0.40,
                "original_waypoint_b_acceptance_unchanged": True,
                "transit_forward_m_s": TURN_FORWARD_COMMAND_M_S,
                "per_window_nonzero_limit_s": WAYPOINT_B_WINDOW_MAX_S,
                "minimum_windows_required": 1,
                "maximum_windows": WAYPOINT_B_MAX_WINDOWS,
                "aggregate_transit_nonzero_limit_s": WAYPOINT_B_AGGREGATE_TRANSIT_MAX_S,
                "route_nonzero_limit_s": 30.0,
                "pre_deadline_zero_margin_s": 0.02,
                "inter_window_requirement": "fresh pose and physical-stop PASS before another window",
                "post_waypoint_actions": "end after diagnostic waypoint; no turn or C waypoint",
            }
        elif run65_waypoint_b_only:
            manifest["command_profile"] = {
                "shape": "measured-multi-window-waypoint-B-only",
                "waypoint_b_distance_from_start_m": 0.40,
                "waypoint_b_position_tolerance_m": 0.15,
                "transit_forward_m_s": TURN_FORWARD_COMMAND_M_S,
                "per_window_nonzero_limit_s": 5.0,
                "minimum_windows_required": 2,
                "maximum_windows": WAYPOINT_B_MAX_WINDOWS,
                "aggregate_waypoint_b_nonzero_limit_s": 10.0,
                "route_nonzero_limit_s": 30.0,
                "pre_deadline_zero_margin_s": 0.02,
                "inter_window_requirement": "fresh measured pose, safe state and physical-stop PASS before replanning",
                "post_waypoint_actions": "zero-only physical stop verification; right turn and waypoint C NOT RUN",
            }
        manifest_path = self.evidence / "foundationwalk_config_overlay_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        self.foundationwalk_config_overlay = overlay
        self.foundationwalk_config_overlay_manifest = manifest

    def prepare_mc_rl_debug_overlay(self) -> None:
        """Stage and verify the immutable diagnostic message overlay before runtime."""
        if not getattr(self.args, "capture_mc_rl_debug", False):
            return
        source = Path(getattr(
            self.args,
            "mc_rl_debug_overlay_source",
            str(DEFAULT_MC_RL_DEBUG_OVERLAY_SOURCE),
        ))
        manifest = copy_verified_mc_rl_debug_overlay_bundle(
            source,
            self.evidence,
            approved_root=REPO / "reviews/issue-80",
        )
        manifest_path = self.evidence / "mc_rl_debug_overlay_copy_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        self.mc_rl_debug_overlay_manifest = manifest
        self.mc_rl_debug_schema_verification = self.verify_mc_rl_debug_overlay()

    def prepare_native_mjdata_capture(self) -> None:
        """Stage the existing read-only native-MjData overlay for one run."""
        if not self.native_mjdata_capture:
            return
        staged_launch = self.evidence / "launch_sim.sh"
        self.native_mjdata_overlay = prepare_mjdata_capture_overlay(
            self.evidence,
            staged_launch_script=staged_launch,
            interval_steps=self.native_mjdata_capture_interval_steps,
        )
        self.native_mjdata_smoke = run_mjdata_capture_smoke(
            self.evidence,
            overlay=self.native_mjdata_overlay,
            runtime_root=RUNTIME,
        )
        write_new_capture_json(
            self.evidence / "native_mjdata_capture_smoke_result.json",
            self.native_mjdata_smoke,
        )
        write_new_capture_json(
            self.evidence / "native_mjdata_capture_overlay.json",
            self.native_mjdata_overlay,
        )

    def finalize_native_mjdata_capture(self) -> dict[str, Any] | None:
        """Summarize captured native state only after the simulator has stopped."""
        if not self.native_mjdata_capture:
            return None
        trace = self.evidence / CAPTURE_TRACE_NAME
        try:
            summary = summarize_mjdata_capture(
                trace,
                standing_trace=self.evidence / "reset_state_trace.jsonl",
            )
        except (MjDataCaptureError, OSError, ValueError) as exc:
            summary = {
                "schema": "robotsim.issue80.native-mjdata-capture-summary.v1",
                "status": "BLOCKED_NATIVE_MJDATA_CAPTURE",
                "failure_reasons": [f"{type(exc).__name__}: {exc}"],
                "trace": str(trace),
            }
        summary_path = self.evidence / CAPTURE_SUMMARY_NAME
        if not summary_path.exists():
            write_new_capture_json(summary_path, summary)
        return summary

    def stage_research_scripts(self) -> None:
        """Stage exact research code snapshots without overwriting evidence."""
        sources = (
            (Path(__file__), Path(__file__).name),
            (X11_HELPER, X11_HELPER.name),
            (X11_RECORDER, X11_RECORDER.name),
            (MJDATA_CAPTURE, MJDATA_CAPTURE.name),
            (MJDATA_CAPTURE_C, MJDATA_CAPTURE_C.name),
            (WINDOWS_FOCUS_HELPER, WINDOWS_FOCUS_HELPER.name),
            (ANALYZER, ANALYZER.name),
            (M1_RUNNER, M1_RUNNER.name),
            (VENDOR_DIRECT_TRUTH_ROUTE, VENDOR_DIRECT_TRUTH_ROUTE.name),
            (LIDAR_LOCALIZER, LIDAR_LOCALIZER.name),
            (LIDAR_SCAN_MATH, LIDAR_SCAN_MATH.name),
            (POSE_GOAL_CONTROLLER, POSE_GOAL_CONTROLLER.name),
            (M1_MODE_LOGIC, M1_MODE_LOGIC.name),
            (M1_FIRST_MOTION_LOGIC, M1_FIRST_MOTION_LOGIC.name),
            (WAYPOINT_ROUTE, WAYPOINT_ROUTE.name),
            (LOCAL01_SCENE_ROUTE, LOCAL01_SCENE_ROUTE.name),
            (ROUTE_MOTION_BRIDGE, ROUTE_MOTION_BRIDGE.name),
            (SET_ACTION_CAPTURE, SET_ACTION_CAPTURE.name),
            (MC_RL_DEBUG_CAPTURE, MC_RL_DEBUG_CAPTURE.name),
            (SIM_LAUNCH, "launch_sim.sh"),
            (MC_LAUNCH, "launch_mc.sh"),
        )
        records: list[dict[str, str]] = []
        for source, staged_name in sources:
            payload = source.read_bytes()
            target = self.evidence / staged_name
            digest = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"staged runtime input differs from current source: {target.name}")
                disposition = "PRESTAGED_IDENTICAL"
            else:
                with target.open("xb") as stream:
                    stream.write(payload)
                disposition = "COPIED"
            records.append({
                "name": source.name,
                "staged_name": staged_name,
                "sha256": digest,
                "disposition": disposition,
            })
        manifest = self.evidence / "staged_research_scripts.json"
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "robotsim.issue80.staged-research-scripts.v1", "scripts": records}, stream, indent=2)
            stream.write("\n")

    def stage_route_turn_qualification_evidence(self) -> dict[str, Any]:
        """Copy the separately governed passing right-turn result into this route run."""
        source = Path(self.args.route_turn_evidence)
        verified = validate_right_turn_qualification_file(source)
        target = self.evidence / "route_turn_qualification_evidence.json"
        payload = source.read_bytes()
        with target.open("xb") as stream:
            stream.write(payload)
        staged = {
            **verified,
            "staged_path": str(target),
            "staged_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes_identical": hashlib.sha256(payload).hexdigest() == verified["sha256"],
        }
        if not staged["bytes_identical"]:
            raise RuntimeError("staged right-turn qualification evidence changed during copy")
        with (self.evidence / "route_turn_qualification_evidence_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(staged, stream, indent=2)
            stream.write("\n")
        return staged

    def prepare_mc_prearm_writeback(self) -> None:
        """Prepare the only writable MC paths for the pre-simulator phase."""
        if not self.mc_prearm_before_simulator:
            return
        if not self.static_nominal_start_initialization:
            raise RuntimeError("MC pre-arm is restricted to static nominal-start initialization")
        if not bool(getattr(self.args, "runtime_readonly", False)):
            raise RuntimeError("MC pre-arm requires the vendor runtime to be mounted read-only")
        # The tmpfs is mounted only inside the isolated container and is seeded
        # below from evidence-local copies of the three known source files.
        self.args.mc_sim_generated_tmpfs = True
        staged_dump, manifest = stage_mc_generated_config_writeback(self.evidence)
        self.runtime_writeback_mounts = [
            (staged_dump, MC_SIM_CONFIG_DUMP_CONTAINER_PATH),
        ]
        self.mc_prearm_writeback_manifest = manifest

    def seed_mc_prearm_generated_config(self) -> dict[str, Any]:
        """Copy verified generated-config seeds into the container tmpfs once."""
        if not self.mc_prearm_before_simulator:
            return {"status": "NOT_REQUESTED"}
        manifest = self.mc_prearm_writeback_manifest
        if not isinstance(manifest, Mapping):
            raise RuntimeError("MC pre-arm writeback manifest is missing")
        seed_directory = self.evidence / "mc_runtime_writeback" / "sim_seed"
        container_seed_directory = "/workspace/evidence/mc_runtime_writeback/sim_seed"
        if (
            seed_directory.is_symlink()
            or not seed_directory.is_dir()
            or any(not (seed_directory / name).is_file() for name in MC_SIM_GENERATED_SEED_FILENAMES)
        ):
            raise RuntimeError("MC pre-arm evidence-local generated-config seed directory is missing or unsafe")
        expected = manifest.get("seed_hashes")
        if not isinstance(expected, Mapping) or set(expected) != set(MC_SIM_GENERATED_SEED_FILENAMES):
            raise RuntimeError("MC pre-arm seed manifest is malformed")
        copy_command = " && ".join(
            "cp "
            + shlex.quote(container_seed_directory + "/" + name)
            + " "
            + shlex.quote(MC_SIM_GENERATED_DIRECTORY + "/" + name)
            for name in MC_SIM_GENERATED_SEED_FILENAMES
        )
        self.docker_exec(
            "seed_mc_prearm_generated_config",
            copy_command,
            timeout_s=3.0,
        )
        observed: dict[str, str] = {}
        for name in MC_SIM_GENERATED_SEED_FILENAMES:
            result = self.docker_exec(
                f"verify_mc_prearm_seed_{name}",
                "sha256sum " + shlex.quote(MC_SIM_GENERATED_DIRECTORY + "/" + name),
                timeout_s=2.0,
            )
            parts = result.stdout.split()
            observed[name] = parts[0] if parts else ""
        if observed != dict(expected):
            raise RuntimeError("MC pre-arm generated config seeds differ from verified source inputs")
        record = {
            "schema": "robotsim.issue80.mc-prearm-seed.v1",
            "status": "PASS_SEEDED_EPHEMERAL_GENERATED_CONFIG",
            "source_manifest": str(self.evidence / "mc_runtime_writeback_manifest.json"),
            "generated_tmpfs_target": MC_SIM_GENERATED_DIRECTORY,
            "observed_seed_hashes": observed,
            "vendor_runtime_tree_modified": False,
        }
        with (self.evidence / "mc_prearm_seed.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        return record

    def start_mc_prearmed_before_simulator(self) -> dict[str, Any]:
        """Start the unmodified MC core before the simulator can take a step.

        The core is allowed to wait for the simulator's leg/IMU topics.  A
        running process, completed AimRT startup, and no generated-config
        write failure are the only claims here; controller readiness still
        requires live state after the simulator starts.
        """
        if not self.mc_prearm_before_simulator:
            return {"status": "NOT_REQUESTED"}
        self.docker_exec(
            "start_mc_prearmed_before_simulator",
            "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )
        shell = (
            "timeout 12s bash -lc " + shlex.quote(
                "until grep -q 'AimRT startup completed' /workspace/evidence/mc-console.log; do "
                "if grep -qE 'Failed to write file|Aborted|SIGABRT|Segmentation fault' "
                "/workspace/evidence/mc-console.log; then exit 3; fi; sleep 0.05; done; "
                "pid=$(pgrep -f '^./aima-mc-app-main --cfg_file_path=' | head -n1); "
                "test -n \"$pid\"; test -x /proc/$pid/exe; "
                "grep -q 'MotionControlModule Started Succeeded' /workspace/evidence/mc-console.log"
            )
        )
        result = self.docker_exec(
            "wait_mc_prearmed_core",
            shell,
            timeout_s=13.0,
            check=False,
        )
        log = self.docker_exec(
            "capture_mc_prearm_log_tail",
            "tail -n 160 /workspace/evidence/mc-console.log",
            timeout_s=3.0,
            check=False,
        )
        record = {
            "schema": "robotsim.issue80.mc-prearm.v1",
            "status": (
                "PASS_MC_CORE_PREARMED_AWAITING_SIMULATOR_STATE"
                if result.returncode == 0 else "FAIL_MC_CORE_PREARM"
            ),
            "simulator_started_before_record": False,
            "set_mc_action_requests": 0,
            "velocity_publishers_created": 0,
            "wait_returncode": result.returncode,
            "wait_stdout": result.stdout,
            "wait_stderr": result.stderr,
            "log_tail": log.stdout,
            "controller_state_claim": "NOT_READY_UNTIL_FRESH_LEG_AND_IMU_STATE",
            "vendor_runtime_tree_modified": False,
        }
        with (self.evidence / "mc_prearm_status.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        self.mc_prearm_status = record
        if result.returncode != 0:
            raise RuntimeError("MC core did not become ready before the simulator was started")
        return record

    def start_container(self) -> None:
        cmd = [
            "docker", "run", "-d", "--pull=never", "--name", self.container,
            "--network=bridge", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770,size=64m",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", build_runtime_bind_mount(
                RUNTIME,
                readonly=bool(getattr(self.args, "runtime_readonly", False)),
            ),
            "--mount", f"type=bind,source={self.evidence},target=/workspace/evidence",
            "--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly",
        ]
        if bool(getattr(self.args, "mc_sim_generated_tmpfs", False)):
            cmd += ["--tmpfs", MC_SIM_GENERATED_DIRECTORY
                    + ":rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0755,size=8m"]
        if self.local01_scene_bundle is not None:
            if not isinstance(self.local01_scene_source, Mapping):
                raise RuntimeError("Local01 scene bundle lacks pinned source provenance")
            reference_root = Path(str(self.local01_scene_source["reference_root"]))
            if reference_root.is_symlink() or not reference_root.is_dir():
                raise RuntimeError("pinned RAICOM scene asset root is missing or unsafe to mount")
            cmd += [
                "--mount",
                "type=bind,source=" + str(reference_root)
                + ",target=/workspace/raicom_scene_source,readonly",
            ]
            for source_path, container_path in self.local01_scene_readonly_mounts:
                if source_path.is_symlink() or not source_path.is_file() or source_path.parent != self.evidence and self.evidence not in source_path.parents:
                    raise RuntimeError(f"Local01 runtime input is missing, unsafe, or outside evidence: {source_path.name}")
                cmd += [
                    "--mount",
                    "type=bind,source=" + str(source_path)
                    + ",target=" + container_path + ",readonly",
                ]
        if self.lidar_overlay_source is not None:
            plugins = self.lidar_overlay_source / "mujoco_plugin"
            required = (
                self.lidar_overlay_dir / "simulator_lidar_overlay.yaml",
                self.lidar_overlay_dir / "scene_lidar_overlay.xml",
                self.lidar_overlay_dir / "robot_lidar_overlay.yaml",
                self.lidar_overlay_dir / "LIDAR_OVERLAY_PROVENANCE.json",
                plugins / "libsensor_ray.so",
                plugins / "libissue80_ray.so",
            )
            if any(path.is_symlink() or not path.is_file() for path in required):
                raise RuntimeError("requested LiDAR overlay source is incomplete or unsafe")
            cmd += [
                "--mount", "type=bind,source=" + str(plugins) + ",target=/workspace/raicom2026/sim_mujoco/bin/mujoco_plugin,readonly",
            ]
        if self.foundationwalk_config_overlay is not None:
            if (
                self.foundationwalk_config_overlay.is_symlink()
                or not self.foundationwalk_config_overlay.is_file()
                or self.foundationwalk_config_overlay.parent != self.evidence
            ):
                raise RuntimeError("diagnostic FoundationWalk overlay must be a regular file in this evidence directory")
            cmd += [
                "--mount",
                "type=bind,source=" + str(self.foundationwalk_config_overlay)
                + ",target=" + FOUNDATIONWALK_CONFIG_CONTAINER_PATH + ",readonly",
            ]
        for source_path, container_path in self.runtime_writeback_mounts:
            try:
                source_path.resolve(strict=True).relative_to(self.evidence.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise RuntimeError("runtime writeback source must be a regular evidence-local file") from exc
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or container_path != MC_SIM_CONFIG_DUMP_CONTAINER_PATH
            ):
                raise RuntimeError("only the staged mc_sim.yaml.dump writeback target is permitted")
            cmd += [
                "--mount",
                "type=bind,source=" + str(source_path) + ",target=" + container_path,
            ]
        if self.mc_rl_debug_overlay_manifest is not None:
            mc_debug_install = self.evidence / "mc_debug_overlay/install"
            if mc_debug_install.is_symlink() or not (mc_debug_install / "setup.bash").is_file():
                raise RuntimeError("verified McRLDebug install prefix is missing or unsafe to mount")
            cmd += [
                "--mount",
                "type=bind,source=" + str(mc_debug_install)
                + ",target=/workspace/evidence/mc_debug_overlay/install,readonly",
            ]
        cmd += [
            "-e", "DISPLAY=:0", "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20",
            "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        self.run("docker_run", cmd, timeout_s=10.0)
        self.container_created = True
        inspect = self.run("container_inspect_start", ["docker", "inspect", self.container], timeout_s=5.0)
        payload = json.loads(inspect.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError("container inspect did not return one isolated runtime")
        container = payload[0]
        inspected_id = container.get("Id")
        if isinstance(inspected_id, str) and CONTAINER_ID_RE.fullmatch(inspected_id):
            self.container_id = inspected_id
        self.network_mode = str(container.get("HostConfig", {}).get("NetworkMode", ""))
        if self.network_mode != "bridge":
            raise RuntimeError(f"first-motion test requires bridge-isolated container; got {self.network_mode!r}")
        environment = set(container.get("Config", {}).get("Env", []))
        required_environment = {
            "ROS_DOMAIN_ID=20",
            "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        }
        if not required_environment.issubset(environment):
            raise RuntimeError("container ROS domain/RMW do not match the isolated M1 contract")
        if self.foundationwalk_config_overlay is not None:
            mounts = container.get("Mounts", [])
            matching_mounts = [
                mount for mount in mounts
                if mount.get("Destination") == FOUNDATIONWALK_CONFIG_CONTAINER_PATH
            ]
            if len(matching_mounts) != 1:
                raise RuntimeError("diagnostic FoundationWalk config mount was not unique")
            mount = matching_mounts[0]
            if (
                Path(str(mount.get("Source", ""))).resolve() != self.foundationwalk_config_overlay.resolve()
                or mount.get("RW") is not False
                or mount.get("Type") != "bind"
            ):
                raise RuntimeError("diagnostic FoundationWalk config mount source/type/read-only state is invalid")
        for source_path, container_path in self.runtime_writeback_mounts:
            matches = [
                mount for mount in container.get("Mounts", [])
                if mount.get("Destination") == container_path
            ]
            if (
                len(matches) != 1
                or Path(str(matches[0].get("Source", ""))).resolve() != source_path.resolve()
                or matches[0].get("RW") is not True
                or matches[0].get("Type") != "bind"
            ):
                raise RuntimeError("MC runtime writeback mount is not the verified narrow writable target")
        if self.mc_rl_debug_overlay_manifest is not None:
            target = "/workspace/evidence/mc_debug_overlay/install"
            matching_mounts = [
                mount for mount in container.get("Mounts", [])
                if mount.get("Destination") == target
            ]
            expected_source = (self.evidence / "mc_debug_overlay/install").resolve()
            if len(matching_mounts) != 1:
                raise RuntimeError("McRLDebug install overlay mount was not unique")
            mount = matching_mounts[0]
            if (
                Path(str(mount.get("Source", ""))).resolve() != expected_source
                or mount.get("RW") is not False
                or mount.get("Type") != "bind"
            ):
                raise RuntimeError("McRLDebug install overlay mount source/type/read-only state is invalid")
        if self.local01_scene_bundle is not None:
            scene_mounts = [
                mount for mount in container.get("Mounts", [])
                if mount.get("Destination") == "/workspace/raicom_scene_source"
            ]
            expected_reference = Path(str(self.local01_scene_source["reference_root"])).resolve()
            if (
                len(scene_mounts) != 1
                or Path(str(scene_mounts[0].get("Source", ""))).resolve() != expected_reference
                or scene_mounts[0].get("RW") is not False
                or scene_mounts[0].get("Type") != "bind"
            ):
                raise RuntimeError("RAICOM scene source mount is missing, writable, or does not match the pinned root")
            for source_path, container_path in self.local01_scene_readonly_mounts:
                matches = [
                    mount for mount in container.get("Mounts", [])
                    if mount.get("Destination") == container_path
                ]
                if (
                    len(matches) != 1
                    or Path(str(matches[0].get("Source", ""))).resolve() != source_path.resolve()
                    or matches[0].get("RW") is not False
                    or matches[0].get("Type") != "bind"
                ):
                    raise RuntimeError(f"Local01 runtime input mount is not uniquely read-only: {container_path}")
        with (self.evidence / "container-inspect-start.json").open("x", encoding="utf-8") as f:
            f.write(inspect.stdout)

    def start_sim_first(self, *, start_mc: bool = True) -> None:
        launch_command = "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"
        if self.local01_scene_bundle is not None:
            config_path = str(self.local01_scene_bundle.get(
                "runtime_simulator_config_path",
                "/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml",
            ))
            launch_command = (
                "ROBOTSIM_SIMULATOR_CFG_OVERRIDE="
                + shlex.quote(config_path) + " "
                + launch_command
            )
        if self.lidar_overlay_source is not None:
            config_path = "/workspace/evidence/lidar_overlay/simulator_lidar_overlay.yaml"
            launch_command = (
                "ROBOTSIM_SIMULATOR_CFG_OVERRIDE=" + config_path
                + " bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"
            )
        self.docker_exec(
            "launch_simulator",
            launch_command,
            timeout_s=4.0,
            detached=True,
        )
        # Do not use repeated short-lived `ros2 topic echo --once` processes as
        # startup readiness probes. On the pinned runtime they can time out
        # during ROS graph discovery (`!rclpy.ok()`) even after the simulator
        # has initialized and started its state-publisher threads. Wait for
        # the simulator's own module-ready log marker here; the persistent
        # read-only state observer below verifies actual live odom/IMU/leg
        # traffic after MC starts.
        self.docker_exec(
            "wait_simulator_module_ready",
            "timeout 25s bash -lc 'until grep -q \"MujocoSimModule Started Succeeded\" /workspace/evidence/mujoco-console.log; do sleep 0.2; done'",
            timeout_s=26.0,
        )
        if self.local01_scene_bundle is not None and self.lidar_overlay_source is None:
            scene_path = str(self.local01_scene_bundle.get(
                "runtime_scene_path",
                "/workspace/evidence/local01_scene/scene_workspace_runtime.xml",
            ))
            scene_filename = str(self.local01_scene_bundle.get(
                "runtime_scene_filename", "scene_workspace_runtime.xml"
            ))
            expected_scene = str(self.local01_scene_bundle["runtime_files_sha256"][scene_filename])
            checked = self.docker_exec(
                "verify_local01_runtime_scene_wrapper_sha256",
                "sha256sum " + shlex.quote(scene_path),
                timeout_s=4.0,
            )
            actual_scene = checked.stdout.split()[0] if checked.stdout.split() else ""
            if actual_scene != expected_scene:
                raise RuntimeError("container scene wrapper differs from the statically validated Local01 scene")
            checked = self.docker_exec(
                "verify_local01_simulator_config_sha256",
                "sha256sum " + shlex.quote(config_path),
                timeout_s=4.0,
            )
            actual_config = checked.stdout.split()[0] if checked.stdout.split() else ""
            config_filename = str(self.local01_scene_bundle.get(
                "runtime_simulator_config_filename", "simulator_default_local01_scene.yaml"
            ))
            expected_config = str(self.local01_scene_bundle["runtime_files_sha256"][config_filename])
            if actual_config != expected_config:
                raise RuntimeError("container simulator config differs from the validated Local01 scene overlay")
        if self.lidar_overlay_source is not None:
            assert self.lidar_overlay_dir is not None
            for filename in ("scene_lidar_overlay.xml", "simulator_lidar_overlay.yaml"):
                source_path = self.lidar_overlay_dir / filename
                expected = hashlib.sha256(source_path.read_bytes()).hexdigest()
                checked = self.docker_exec(
                    "verify_lidar_overlay_" + filename.replace(".", "_"),
                    "sha256sum /workspace/evidence/lidar_overlay/" + filename,
                    timeout_s=4.0,
                )
                actual = checked.stdout.split()[0] if checked.stdout.split() else ""
                if actual != expected:
                    raise RuntimeError("container LiDAR overlay differs from the evidence-local verified input")
        if self.foundationwalk_config_overlay is not None:
            expected = str(self.foundationwalk_config_overlay_manifest["overlay_sha256"])
            checked = self.docker_exec(
                "verify_mounted_foundationwalk_config_overlay",
                "sha256sum " + shlex.quote(FOUNDATIONWALK_CONFIG_CONTAINER_PATH),
                timeout_s=4.0,
            )
            actual = checked.stdout.split()[0] if checked.stdout.split() else ""
            if actual != expected:
                raise RuntimeError(
                    "container sees an unexpected FoundationWalk config overlay SHA-256: "
                    f"{actual}"
                )
            verification = {
                "schema": "robotsim.issue80.foundationwalk-overlay-runtime-verification.v1",
                "container_path": FOUNDATIONWALK_CONFIG_CONTAINER_PATH,
                "expected_sha256": expected,
                "observed_sha256": actual,
                "matched_before_mc_launch": True,
            }
            with (self.evidence / "foundationwalk_config_overlay_runtime_verification.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump(verification, stream, indent=2)
                stream.write("\n")
        if start_mc:
            self.start_mc_after_sim_topics()

    def start_mc_after_sim_topics(self) -> None:
        """Launch the unchanged MC only after the simulator ordering gate permits it."""
        self.docker_exec(
            "start_mc_after_sim_topics",
            "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )

    def wait_for_immediate_reset_pose(self) -> dict[str, Any]:
        event = (self.reset_dispatch_payload or {}).get("event", {})
        if event.get("status") not in {"KEY_DISPATCHED", "RESET_BUTTON_CLICKED"} or not event.get("dispatch_monotonic_ns"):
            raise RuntimeError("verified Reset dispatch timestamp is missing")
        dispatch_ns = int(event["dispatch_monotonic_ns"])
        trace = self.evidence / "reset_state_trace.jsonl"
        required = {"odom", "imu", "common", "leg_state", "leg_command"}
        # Run70 proved the key event can be processed after a 2.216 s GUI
        # delay. Keep the physical upright/freshness gate unchanged; this is
        # only a bounded window for reset delivery and state publication.
        deadline = min(self.deadline, time.monotonic() + RESET_STARTUP_OBSERVATION_TIMEOUT_S)
        last = {"status": "POST_RESET_TRACE_NOT_AVAILABLE", "dispatch_monotonic_ns": dispatch_ns}
        latest: dict[str, dict[str, Any]] = {}
        upright_tail: list[dict[str, Any]] = []
        if not trace.exists():
            while time.monotonic() < deadline and not trace.exists():
                time.sleep(0.02)
        if trace.exists():
            with trace.open("r", encoding="utf-8") as trace_file:
                while time.monotonic() < deadline:
                    for line in trace_file:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("event") != "state" or int(row.get("wall_monotonic_ns", 0)) <= dispatch_ns:
                            continue
                        key = row.get("topic_key")
                        if key in required:
                            latest[key] = row
                        if key == "odom":
                            data = row.get("data", {})
                            try:
                                ok = float(data["z"]) >= 0.52 and max(abs(float(data["roll_deg"])), abs(float(data["pitch_deg"]))) < 15.0
                            except (KeyError, TypeError, ValueError):
                                ok = False
                            if ok:
                                upright_tail.append(row)
                                upright_tail = upright_tail[-3:]
                            else:
                                upright_tail.clear()
                    now_ns = time.monotonic_ns()
                    ages = {key: (now_ns - int(row["wall_monotonic_ns"])) / 1e9 for key, row in latest.items()}
                    fresh = required.issubset(latest) and all(0.0 <= ages[key] <= 0.5 for key in required)
                    last = {
                        "status": "POST_RESET_UPRIGHT_NOT_YET_CONFIRMED",
                        "dispatch_monotonic_ns": dispatch_ns,
                        "startup_observation_timeout_s": RESET_STARTUP_OBSERVATION_TIMEOUT_S,
                        "state_ages_s": ages,
                        "fresh_all_required_topics": fresh,
                        "consecutive_upright_odom_samples": len(upright_tail),
                        "latest_odom": latest.get("odom"),
                        "latest_raw_common_state": latest.get("common"),
                    }
                    if fresh and len(upright_tail) >= 3:
                        last.update({
                            "status": "POST_RESET_UPRIGHT_OBSERVED",
                            "wall_utc": datetime.now(timezone.utc).isoformat(),
                            "dispatch_to_upright_ms": (int(upright_tail[0]["wall_monotonic_ns"]) - dispatch_ns) / 1e6,
                            "first_post_reset_upright_odom_monotonic_ns": int(
                                upright_tail[0]["wall_monotonic_ns"]
                            ),
                            "pose": upright_tail[-1].get("data"),
                            "raw_common_state": latest.get("common", {}).get("data"),
                            "imu": latest.get("imu", {}).get("data"),
                            "leg_state": latest.get("leg_state", {}).get("data"),
                            "leg_command": latest.get("leg_command", {}).get("data"),
                            "physical_gate": {"minimum_root_z_m": 0.52, "maximum_abs_roll_pitch_deg_exclusive": 15.0},
                            "consecutive_upright_odom_required": 3,
                        })
                        self.immediate_reset_pose = last
                        with (self.evidence / "reset_immediate_posture.json").open("x", encoding="utf-8") as f:
                            json.dump(last, f, indent=2)
                            f.write("\n")
                        return last
                    time.sleep(0.02)
        last["status"] = "POST_RESET_UPRIGHT_TIMEOUT"
        last["timeout_s"] = RESET_STARTUP_OBSERVATION_TIMEOUT_S
        self.immediate_reset_pose = last
        self.outcome_classification = "STANDING_CONTROL_NOT_READY"
        with (self.evidence / "reset_immediate_posture.json").open("x", encoding="utf-8") as f:
            json.dump(last, f, indent=2)
            f.write("\n")
        raise RuntimeError(
            "Reset did not produce a fresh upright pose within the bounded "
            f"{RESET_STARTUP_OBSERVATION_TIMEOUT_S:.1f} s startup observation; SD was not sent"
        )

    def request_sd_and_verify(self) -> dict[str, Any]:
        # The ROS node, all state subscriptions, and both MC service clients
        # were created before Reset by start_observer(). Read the transaction
        # produced by that persistent process; do not spawn Python/DDS here.
        result_path = self.evidence / "set_action_capture_result.json"
        summary_path = self.evidence / "reset_state_summary.json"
        deadline = min(self.deadline, time.monotonic() + RESET_STAND_HANDOFF_MAX_S + 2.0)
        while time.monotonic() < deadline and not result_path.exists():
            if summary_path.exists():
                break
            time.sleep(0.01)
        if result_path.exists():
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raw = {
                    "status": "BLOCKED_RESULT_READ",
                    "blocking_stage": "RESULT_READ",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            raw = {
                "status": "BLOCKED_RESULT_TIMEOUT",
                "blocking_stage": "RESULT_TIMEOUT",
                "error": "the pre-armed ROS process did not publish its result before the bounded capture window ended",
            }
        raw_sd = raw.get("stand_default_set_mc_action") or {}
        record: dict[str, Any] = {
            "wall_utc_observed": datetime.now(timezone.utc).isoformat(),
            "requested_action_desc": "STAND_DEFAULT",
            "request_source": "node.set_mc_action",
            "takeover_status": raw.get("status"),
            "blocking_stage": raw.get("blocking_stage"),
            "error": raw.get("error"),
            "pre_reset_get_mc_action": raw.get("pre_reset_get_mc_action"),
            "post_reset_upright_gate": raw.get("post_reset_upright_gate"),
            "joint_default_set_mc_action": raw.get("joint_default_set_mc_action"),
            "joint_to_stand_gate": raw.get("joint_to_stand_gate"),
            "raw_set_mc_action": raw_sd,
            "get_mc_action_readback": raw.get("get_mc_action_readbacks", []),
            "timing_ms": raw.get("timing_ms"),
            "limitations": raw.get("limitations", []),
            "response_status_value": raw_sd.get("response_status_value"),
            "response_header_code": raw_sd.get("response_header_code"),
            "response_message": raw_sd.get("response_message"),
        }
        self.sd_result = record
        if raw.get("status") != "TAKEOVER_SEQUENCE_COMPLETE":
            jd = raw.get("joint_default_set_mc_action") or {}
            sd = raw.get("stand_default_set_mc_action") or {}
            stage = raw.get("blocking_stage")
            detail = sd if sd else jd
            if (
                stage == "STAND_DEFAULT"
                and detail.get("transport_state") == "RESPONSE_RECEIVED"
                and (detail.get("service_success") is not True or detail.get("action_result_success") is not True)
            ):
                self.outcome_classification = "SD_REJECTED_WITH_EXACT_CODE"
            else:
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            self._write_sd_record(record)
            raise RuntimeError(
                "source-supported PD -> JD -> SD handoff did not complete: "
                f"stage={raw.get('blocking_stage')!r} status={detail.get('response_status_value')} "
                f"header.code={detail.get('response_header_code')} message={detail.get('response_message')!r} "
                f"reason={raw.get('error')!r}"
            )
        if raw_sd.get("service_success") is not True or raw_sd.get("action_result_success") is not True:
            self.outcome_classification = "SD_REJECTED_WITH_EXACT_CODE"
            self._write_sd_record(record)
            raise RuntimeError(
                "SetMcAction SD was rejected: "
                f"status={record['response_status_value']} header.code={record['response_header_code']} "
                f"message={record['response_message']!r}"
            )
        stand_readbacks = [
            item for item in record["get_mc_action_readback"]
            if item.get("action_desc") == "STAND_DEFAULT" and item.get("action_status_value") == 100
        ]
        if not stand_readbacks:
            self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            self._write_sd_record(record)
            raise RuntimeError("the persistent GetMcAction client did not confirm STAND_DEFAULT/status=100")
        self._write_sd_record(record)
        return record

    def _write_sd_record(self, record: dict[str, Any]) -> None:
        path = self.evidence / "set_mc_action_sd.json"
        if path.exists():
            return
        with path.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")

    def start_verified_motion_video_capture(self) -> None:
        """Passively record the already Reset-verified MuJoCo window.

        The established recorder watches the child trace for the first actual
        nonzero command; it sends no X11 input and has no path to influence the
        simulator or MC.  A missing recording never changes a safety decision.
        """
        if not getattr(self.args, "record_mujoco_video", False):
            return
        if self.motion_video_process is not None:
            raise RuntimeError("motion video recorder was started more than once")
        recorder = self.evidence / X11_RECORDER.name
        if recorder.is_symlink() or not recorder.is_file():
            raise RuntimeError("staged passive MuJoCo video recorder is missing or unsafe")
        stdout_path = self.evidence / "x11_walk_capture.stdout.txt"
        stderr_path = self.evidence / "x11_walk_capture.stderr.txt"
        command = [
            sys.executable, str(recorder),
            "--evidence-dir", str(self.evidence),
            "--fps", "4",
            "--duration-s", "12",
            "--wait-timeout-s", str(min(180.0, max(30.0, self.remaining()))),
            "--output-name", "mujoco_aligned_start_route.gif",
        ]
        environment = dict(os.environ)
        environment.setdefault("DISPLAY", ":0")
        stdout = stdout_path.open("x", encoding="utf-8")
        stderr = stderr_path.open("x", encoding="utf-8")
        try:
            self.motion_video_process = subprocess.Popen(
                command,
                text=True,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
            )
        finally:
            stdout.close()
            stderr.close()
        self.motion_video_capture = {
            "status": "STARTED_PASSIVE_RECORDING",
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "input_events_sent": False,
            "starts_after_verified_reset": True,
        }

    def finalize_motion_video_capture(self) -> dict[str, Any]:
        """Collect or terminate only the bounded passive recorder at teardown."""
        if self.motion_video_process is None:
            return self.motion_video_capture
        process = self.motion_video_process
        try:
            returncode = process.wait(timeout=15.0)
            termination = "RECORDER_EXITED"
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                returncode = process.wait(timeout=5.0)
                termination = "RECORDER_TERMINATED_AFTER_BOUNDED_ROUTE"
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5.0)
                termination = "RECORDER_KILLED_AFTER_TERMINATE_TIMEOUT"
        manifest_path = self.evidence / "x11_walk_capture.json"
        manifest: Any = None
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                manifest = {"status": "INVALID_MANIFEST", "error": f"{type(exc).__name__}: {exc}"}
        self.motion_video_capture.update({
            "termination": termination,
            "returncode": returncode,
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "manifest": manifest,
        })
        return self.motion_video_capture

    def capture_service_topic_inventory(self) -> None:
        self.docker_exec(
            "service_inventory",
            "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && ros2 service list -t",
            timeout_s=5.0,
        )
        self.docker_exec(
            "topic_inventory",
            "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && ros2 topic list -t",
            timeout_s=5.0,
        )

    def start_observer(self, *, prearmed_before_simulator: bool = False) -> None:
        """Start the one persistent JD/SD client before its event barrier.

        In MC-prearm mode the client is constructed while only the MC service
        graph exists.  It must not claim sensor readiness until the simulator
        begins publishing after the static nominal-start barrier.
        """
        if prearmed_before_simulator and not self.mc_prearm_before_simulator:
            raise RuntimeError("pre-simulator observer requires MC pre-arm")
        native_capture_args = (
            " --native-mjdata-trace /workspace/evidence/" + CAPTURE_TRACE_NAME
            + ("" if self.static_nominal_start_initialization else " --require-native-reset-generation")
            if self.native_mjdata_capture else ""
        )
        if self.native_mjdata_capture and self.lidar_overlay_source is not None:
            native_capture_args += " --allow-initial-native-generation"
        initialization_args = (
            " --synchronized-keyframe-event-file /workspace/evidence/keyframe_load_event.json"
            " --synchronized-keyframe-spec /workspace/evidence/local01_scene/synchronized_nominal_keyframe.json"
            " --physics-release-event-file /workspace/evidence/physics_release_event.json"
            " --synchronized-ready-file /workspace/evidence/synchronized_jd_ready.json"
            if self.synchronized_paused_initialization
            else " --static-nominal-start-event-file /workspace/evidence/static_nominal_start_event.json"
            if self.static_nominal_start_initialization
            else " --reset-event-file /workspace/evidence/reset_dispatch_event.json"
        )
        prearm_args = " --prearmed-before-simulator" if prearmed_before_simulator else ""
        direct_stand_args = " --direct-stand-from-passive" if self.direct_stand_from_passive else ""
        self.docker_exec(
            "start_prearmed_standing_takeover_observer",
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "timeout 45s python3 /workspace/evidence/issue80_m1_set_action_capture.py "
            "--action STAND_DEFAULT --source node.set_mc_action "
            "--output-dir /workspace/evidence "
            f"--ready-timeout-s 3 --reset-wait-timeout-s 30 --upright-timeout-s {RESET_STARTUP_OBSERVATION_TIMEOUT_S:g} "
            "--joint-stand-timeout-s 4 --set-timeout-s 2 --get-timeout-s 0.75 "
            f"--duration-s {RESET_OBSERVER_DURATION_S:.1f} "
            + initialization_args + prearm_args + direct_stand_args + native_capture_args + " "
            "> /workspace/evidence/takeover-capture-console.log 2>&1",
            timeout_s=4.0,
            detached=True,
        )
        ready_path = self.evidence / "reset_watch_ready.json"
        # Child readiness has its own unchanged 3 s budget after Python/ROS
        # construction. Parent also allows bounded process bootstrap time.
        deadline = min(self.deadline, time.monotonic() + 5.0)
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.05)
        if not ready_path.exists():
            raise RuntimeError("Reset observer readiness result missing after 5 s including process bootstrap")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("status") != "READY":
            raise RuntimeError(f"pre-Reset ROS2 observer/client readiness failed: {ready}")
        if not prearmed_before_simulator and len(ready.get("fresh_topics", [])) != 5:
            raise RuntimeError(f"Reset precondition failed: live state streams are incomplete: {ready}")
        if prearmed_before_simulator and ready.get("fresh_topics"):
            raise RuntimeError("MC-prearmed observer unexpectedly claimed live simulator state before launch")
        if ready.get("set_mc_action_service_ready") is not True or ready.get("get_mc_action_service_ready") is not True:
            raise RuntimeError(f"Reset precondition failed: persistent MC action services are not ready: {ready}")
        pre_mode = ready.get("pre_reset_get_mc_action", {})
        if prearmed_before_simulator:
            if pre_mode.get("transport_state") != "RESPONSE_RECEIVED" or pre_mode.get("response_header_code") != 0:
                raise RuntimeError(
                    "MC-prearmed observer did not receive a valid service response before simulator launch: "
                    f"{pre_mode}"
                )
            # The installed MC returns action_desc='' / UNKNOWN(0) before its
            # first leg and IMU state.  That is recorded as uninitialized;
            # the child independently requires PASSIVE_DEFAULT only after
            # fresh state arrives, immediately before JD.
            return
        if pre_mode.get("action_desc") != "PASSIVE_DEFAULT" or pre_mode.get("action_status_value") != 100:
            raise RuntimeError(f"Reset precondition failed: expected PASSIVE_DEFAULT, got {pre_mode}")
        # Pre-SD mode/FSM values are retained raw and are not a standing gate.
        with (self.evidence / "pre-reset-telemetry-snapshot.json").open("x", encoding="utf-8") as f:
            json.dump(ready, f, indent=2)
            f.write("\n")

    def capture_reset_image(self, pid: int, title: str, window_id: int, phase: str) -> dict[str, Any]:
        """Capture images outside the live reset-observation interval when possible."""
        output = self.evidence / f"mujoco_{phase}_reset.png"
        command = [
            sys.executable, str(X11_HELPER), "capture",
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id), "--output", str(output),
        ]
        try:
            result = self.run(f"x11_capture_{phase}_reset", command, timeout_s=8.0, check=False)
        except RunDeadline as exc:
            # Screenshots are review evidence, not a Reset or posture-safety gate.
            # The Reset path independently rechecks the exact process/title/window
            # and pointer target before it can dispatch any GUI input.
            return {
                "status": "NOT_CAPTURED",
                "reason": f"capture_helper_timeout: {exc}",
                "timeout_s": 8.0,
                "input_sent": False,
            }
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "status": "NOT_CAPTURED",
                "reason": f"capture helper returned no JSON (exit={result.returncode})",
            }
        image = payload.get("image", {})
        if result.returncode != 0 or image.get("status") != "CAPTURED":
            return {
                "status": "NOT_CAPTURED",
                "reason": payload.get("reason") or payload.get("error") or result.stderr[-300:],
                "helper_result": payload,
            }
        return image

    def inspect_window(self) -> tuple[int, str, int]:
        pid_result = self.docker_exec(
            "sim_process_pid",
            "pgrep -x soc_sim",
            timeout_s=3.0,
        )
        container_pid = int(pid_result.stdout.strip().splitlines()[0])
        top = self.run("docker_top_processes", ["docker", "top", self.container, "-eo", "pid,comm,args"], timeout_s=4.0)
        host_pid = None
        for line in top.stdout.splitlines()[1:]:
            fields = line.strip().split(None, 2)
            if len(fields) >= 2 and fields[1] == "soc_sim":
                host_pid = int(fields[0])
                break
        pids = [container_pid] + ([host_pid] if host_pid else [])
        discoveries = []
        for pid in pids:
            result = self.run(
                f"x11_inspect_pid_{pid}",
                [sys.executable, str(X11_HELPER), "inspect", "--expected-pid", str(pid)],
                timeout_s=4.0,
                check=False,
            )
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
            discoveries.append(payload)
            if payload.get("windows"):
                windows = payload["windows"]
                if len(windows) != 1:
                    raise RuntimeError(f"ambiguous MuJoCo windows for PID {pid}: {windows}")
                window = windows[0]
                return int(pid), str(window["title"]), int(window["window_id"])
        with (self.evidence / "window-discovery.json").open("x", encoding="utf-8") as f:
            json.dump({"container_pid": container_pid, "host_pid": host_pid, "inspections": discoveries}, f, indent=2)
            f.write("\n")
        raise RuntimeError("no visible X11 window with exact aima-sim-app PID was found")

    def dispatch_reset(
        self,
        pid: int,
        title: str,
        window_id: int,
        *,
        prior_native_button_attempt: Mapping[str, Any] | None = None,
    ) -> None:
        """Legacy verified BackSpace path retained for comparison; not used by execute()."""
        event_path = self.evidence / "reset_dispatch_event.json"
        first_attempt_path = self.evidence / "reset_dispatch_event_x11_attempt1.json"

        def send_one_verified_x11_key(step_name: str, output_path: Path) -> dict[str, Any]:
            command = [
                sys.executable, str(X11_HELPER), "send-reset",
                "--expected-pid", str(pid), "--expected-title", title,
                "--window-id", str(window_id),
                "--dispatch-event-json", str(output_path),
            ]
            try:
                completed = self.run(step_name, command, timeout_s=5.0, check=False)
            except RunDeadline as exc:
                if not output_path.exists():
                    raise
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                payload["helper_completion"] = f"TIMED_OUT_AFTER_EVENT_RECORD: {exc}"
                return payload
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                if output_path.exists():
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    payload["helper_completion"] = "NO_COMPLETE_RESULT_JSON"
                    return payload
                raise RuntimeError(f"X11 helper did not return JSON: {completed.stdout[-500:]}") from exc

        first = send_one_verified_x11_key("x11_backspace_dispatch_attempt1", first_attempt_path)
        first_event = first.get("event", {})
        final_payload = first
        if (
            first_event.get("status") == "RESET_NOT_DELIVERED"
            and int(first_event.get("xtest_key_events_sent", -1)) == 0
        ):
            identity = first_event.get("identity_before_dispatch") or first_event.get("identity_before")
            identity_verified = (
                isinstance(identity, Mapping)
                and identity.get("window_id") == window_id
                and identity.get("pid") == pid
                and identity.get("title") == title
                and identity.get("viewable") is True
            )
            if identity_verified:
                windows_focus = self.activate_verified_windows_foreground(
                    title=title,
                    pid=pid,
                    window_id=window_id,
                    width=int(identity.get("width", 0)),
                    height=int(identity.get("height", 0)),
                    mode="Activate",
                )
                if windows_focus.get("status") == "FOREGROUND_VERIFIED":
                    x11_identity = self.inspect_x11_target_identity(pid, title, window_id)
                    if x11_identity.get("status") == "IDENTITY_VERIFIED":
                        windows_key = self.activate_verified_windows_foreground(
                            title=title,
                            pid=pid,
                            window_id=window_id,
                            width=int(identity.get("width", 0)),
                            height=int(identity.get("height", 0)),
                            mode="SendReset",
                        )
                        if (
                            windows_key.get("status") == "WIN32_KEY_EVENTS_INSERTED"
                            and int(windows_key.get("key_events_sent", 0)) == 2
                        ):
                            target_identity = x11_identity["window"]
                            dispatch_wall = datetime.now(timezone.utc).isoformat()
                            final_payload = {
                                "schema": "robotsim.issue80.reset-dispatch-event.v1",
                                "window": target_identity,
                                "event": {
                                    "status": "KEY_DISPATCHED",
                                    "key": "BackSpace",
                                    "virtual_key": 0x08,
                                    "dispatch_transport": "WIN32_SENDINPUT_TO_VERIFIED_WSLG_RAIL_FOREGROUND",
                                    "target_window_id": window_id,
                                    "expected_pid": pid,
                                    "expected_title": title,
                                    "identity_before_dispatch": target_identity,
                                    "xtest_key_events_sent": 0,
                                    "windows_key_events_inserted": 2,
                                    "dispatch_wall_utc": dispatch_wall,
                                    "dispatch_monotonic_ns": time.monotonic_ns(),
                                    "wall_utc": datetime.now(timezone.utc).isoformat(),
                                },
                                "classification": "WIN32_KEY_DISPATCH_ONLY_STATE_CHANGE_MUST_BE_VERIFIED_BY_ROS_TRACE",
                                "initial_x11_focus_attempt": first,
                                "windows_foreground_activation": windows_focus,
                                "x11_identity_recheck_after_windows_focus": x11_identity,
                                "windows_key_dispatch": windows_key,
                            }
                        else:
                            first["windows_foreground_activation"] = windows_focus
                            first["x11_identity_recheck_after_windows_focus"] = x11_identity
                            first["windows_key_dispatch"] = windows_key
                            if int(windows_key.get("key_events_sent", 0)) > 0:
                                first["classification"] = "PARTIAL_WIN32_KEY_SEQUENCE_STOP_NO_RETRY"
                            final_payload = first
                    else:
                        first["windows_foreground_activation"] = windows_focus
                        first["x11_identity_recheck_after_windows_focus"] = x11_identity
                        final_payload = first
                else:
                    first["windows_foreground_activation"] = windows_focus
                    final_payload = first
        if prior_native_button_attempt is not None:
            final_payload["native_button_attempt_before_keyboard_fallback"] = dict(
                prior_native_button_attempt
            )
        final_payload["reset_delivery_attempt_count"] = (
            1 if final_payload.get("event", {}).get("status") == "KEY_DISPATCHED" else 0
        )
        final_payload["reset_input_path_attempt_count"] = (
            (1 if prior_native_button_attempt is not None else 0)
            + (2 if final_payload is not first else 1)
        )
        if event_path.exists():
            raise RuntimeError("canonical reset event already exists; refusing to overwrite it")
        with event_path.open("x", encoding="utf-8") as stream:
            json.dump(final_payload, stream, indent=2)
            stream.write("\n")
        self.reset_dispatch_payload = final_payload
        if final_payload.get("event", {}).get("status") != "KEY_DISPATCHED":
            self.reset_classification = "RESET_NOT_DELIVERED"
            raise RuntimeError(f"Reset input was not safely dispatched: {final_payload}")

    def dispatch_native_reset_button(self, pid: int, title: str, window_id: int) -> None:
        """Use the verified MuJoCo Simulation/Reset UI control, not keyboard focus."""
        event_path = self.evidence / "reset_dispatch_event.json"
        attempt_path = self.evidence / "reset_dispatch_event_native_button.json"
        command = [
            sys.executable, str(X11_HELPER), "click-native-reset",
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id),
            "--dispatch-event-json", str(attempt_path),
        ]
        try:
            completed = self.run(
                "x11_click_native_simulation_reset",
                command,
                timeout_s=5.0,
                check=False,
            )
        except RunDeadline as exc:
            if not attempt_path.exists():
                raise
            payload = json.loads(attempt_path.read_text(encoding="utf-8"))
            payload["helper_completion"] = f"TIMED_OUT_AFTER_NATIVE_BUTTON_EVENT_RECORD: {exc}"
        else:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                if not attempt_path.exists():
                    raise RuntimeError(
                        f"native MuJoCo Reset helper did not return JSON: {completed.stdout[-500:]}"
                    ) from exc
                payload = json.loads(attempt_path.read_text(encoding="utf-8"))
                payload["helper_completion"] = "NO_COMPLETE_NATIVE_BUTTON_RESULT_JSON"
        event = payload.get("event", {})
        event["dispatch_transport"] = "XTEST_MOUSE_CLICK_TO_VERIFIED_MUJOCO_NATIVE_RESET_BUTTON"
        # A rejected pixel signature has sent no click.  Fall back only then
        # to the reviewed, process-verified BackSpace path.  A real or partial
        # mouse click is never followed by a second state-changing input.
        if (
            event.get("status") == "RESET_NOT_DELIVERED"
            and int(event.get("mouse_button_events_sent", -1)) == 0
        ):
            self.dispatch_reset(
                pid,
                title,
                window_id,
                prior_native_button_attempt=payload,
            )
            return
        payload["reset_delivery_attempt_count"] = int(event.get("status") == "RESET_BUTTON_CLICKED")
        payload["reset_input_path_attempt_count"] = 1
        if event_path.exists():
            raise RuntimeError("canonical reset event already exists; refusing to overwrite it")
        with event_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        self.reset_dispatch_payload = payload
        if event.get("status") != "RESET_BUTTON_CLICKED":
            self.reset_classification = "RESET_NOT_DELIVERED"
            raise RuntimeError(f"native MuJoCo Reset button click was not verified: {payload}")

    def _dispatch_native_simulation_control(
        self,
        *,
        command_name: str,
        step_name: str,
        event_path: Path,
        expected_status: str,
        pid: int,
        title: str,
        window_id: int,
    ) -> dict[str, Any]:
        """Send one process-verified native MuJoCo UI control and persist its event."""
        if event_path.exists():
            raise RuntimeError(f"native control event already exists: {event_path.name}")
        command = [
            sys.executable, str(X11_HELPER), command_name,
            "--expected-pid", str(pid), "--expected-title", title,
            "--window-id", str(window_id),
            "--dispatch-event-json", str(event_path),
        ]
        completed = self.run(step_name, command, timeout_s=5.0, check=False)
        if event_path.exists():
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"native MuJoCo {command_name} helper did not persist or print JSON"
                ) from exc
        event = payload.get("event", {})
        if event.get("status") != expected_status:
            raise RuntimeError(
                f"native MuJoCo {command_name} was not safely dispatched: {payload}"
            )
        return payload

    def dispatch_native_pause_toggle(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        return self._dispatch_native_simulation_control(
            command_name="send-pause",
            step_name="x11_pause_native_mujoco_physics",
            event_path=self.evidence / "pause_dispatch_event.json",
            expected_status="KEY_DISPATCHED",
            pid=pid,
            title=title,
            window_id=window_id,
        )

    def dispatch_native_keyframe_zero(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        return self._dispatch_native_simulation_control(
            command_name="select-key-zero",
            step_name="x11_select_native_keyframe_zero",
            event_path=self.evidence / "keyframe_selection_event.json",
            expected_status="KEYFRAME_INDEX_SELECTION_INPUT_DISPATCHED",
            pid=pid,
            title=title,
            window_id=window_id,
        )

    def dispatch_native_load_key(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        return self._dispatch_native_simulation_control(
            command_name="click-native-load-key",
            step_name="x11_load_native_synchronized_keyframe",
            event_path=self.evidence / "keyframe_load_event.json",
            expected_status="LOAD_KEY_BUTTON_CLICKED",
            pid=pid,
            title=title,
            window_id=window_id,
        )

    def wait_for_synchronized_jd_ready(self) -> dict[str, Any]:
        path = self.evidence / "synchronized_jd_ready.json"
        deadline = min(self.deadline, time.monotonic() + RESET_STAND_HANDOFF_MAX_S)
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("synchronized JD readiness artifact is unreadable") from exc
                if payload.get("status") != "JD_CONTROL_READY_WHILE_PHYSICS_PAUSED":
                    raise RuntimeError(f"synchronized JD control barrier failed: {payload}")
                return payload
            result = self.evidence / "set_action_capture_result.json"
            if result.exists():
                payload = json.loads(result.read_text(encoding="utf-8"))
                raise RuntimeError(f"standing observer terminated before physics release: {payload}")
            time.sleep(0.01)
        raise RuntimeError("synchronized JD control readiness did not arrive before the bounded deadline")

    def dispatch_native_physics_release(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        return self._dispatch_native_simulation_control(
            command_name="send-pause",
            step_name="x11_resume_native_mujoco_physics",
            event_path=self.evidence / "physics_release_event.json",
            expected_status="KEY_DISPATCHED",
            pid=pid,
            title=title,
            window_id=window_id,
        )

    def verify_native_pause_quiescence(self, pause_event: Mapping[str, Any]) -> dict[str, Any]:
        """Require native ``mj_step`` calls to quiesce after the documented Pause key."""
        event = pause_event.get("event", {})
        dispatch_ns = event.get("dispatch_monotonic_ns")
        if not isinstance(dispatch_ns, int) or dispatch_ns <= 0:
            raise RuntimeError("Pause event lacks a valid monotonic dispatch timestamp")
        trace = self.evidence / CAPTURE_TRACE_NAME
        deadline = min(self.deadline, time.monotonic() + 2.0)
        last_step_ns: int | None = None
        pre_dispatch_steps = 0
        while time.monotonic() < deadline:
            rows: list[dict[str, Any]] = []
            if trace.is_file() and not trace.is_symlink():
                for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and isinstance(row.get("mj_step_call_index"), int):
                        rows.append(row)
            pre_dispatch_steps = sum(
                1 for row in rows if int(row.get("host_monotonic_ns", 0)) < dispatch_ns
            )
            post = [row for row in rows if int(row.get("host_monotonic_ns", 0)) >= dispatch_ns]
            if post:
                last_step_ns = max(int(row["host_monotonic_ns"]) for row in post)
            reference_ns = last_step_ns if last_step_ns is not None else dispatch_ns
            quiet_s = (time.monotonic_ns() - reference_ns) / 1e9
            if pre_dispatch_steps and quiet_s >= 0.15:
                record = {
                    "schema": "robotsim.issue80.native-pause-quiescence.v1",
                    "status": "PAUSE_CONFIRMED_BY_MJ_STEP_QUIESCENCE",
                    "pause_dispatch_monotonic_ns": dispatch_ns,
                    "last_post_dispatch_mj_step_monotonic_ns": last_step_ns,
                    "quiescent_duration_s": quiet_s,
                    "pre_dispatch_capture_step_samples": pre_dispatch_steps,
                    "physics_stepping_released": False,
                }
                with (self.evidence / "pause_quiescence.json").open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2)
                    stream.write("\n")
                return record
            time.sleep(0.02)
        record = {
            "schema": "robotsim.issue80.native-pause-quiescence.v1",
            "status": "PAUSE_NOT_CONFIRMED",
            "pause_dispatch_monotonic_ns": dispatch_ns,
            "last_post_dispatch_mj_step_monotonic_ns": last_step_ns,
            "pre_dispatch_capture_step_samples": pre_dispatch_steps,
            "physics_stepping_released": False,
        }
        with (self.evidence / "pause_quiescence.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        raise RuntimeError("native MuJoCo Pause could not be confirmed from MjData step quiescence")

    def verify_native_static_nominal_start(self) -> dict[str, Any]:
        """Require a stable vendor-applied nominal configuration before JD.

        The static wrapper is already present when the vendor loads the model.
        This method only reads the evidence-local capture stream and records a
        single barrier; it has no GUI, qpos, actuator, or MC-action path.
        """
        initialization = (
            self.local01_scene_bundle or {}
        ).get("static_nominal_start_initialization", {})
        static = initialization.get("initialization") if isinstance(initialization, Mapping) else None
        if not isinstance(static, Mapping):
            raise RuntimeError("static nominal-start scene bundle is missing its derived target")
        expected = static.get("source_nominal_target")
        if not isinstance(expected, Mapping):
            raise RuntimeError("static nominal-start scene bundle has no source nominal target")
        trace = self.evidence / CAPTURE_TRACE_NAME
        deadline = min(self.deadline, time.monotonic() + RESET_STAND_HANDOFF_MAX_S)
        last: dict[str, Any] = {"status": "WAITING_FOR_NATIVE_STATIC_NOMINAL_START"}
        consecutive = 0
        required_samples = 1 if self.mc_prearm_before_simulator else 2
        saw_nominal = False
        unsafe_after_nominal: dict[str, Any] | None = None
        seen_native_sample_ns: set[int] = set()
        while time.monotonic() < deadline:
            rows: list[dict[str, Any]] = []
            if trace.is_file() and not trace.is_symlink():
                for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") in {"post_step", "post_step_after_reset_generation"}:
                        rows.append(row)
            for row in rows:
                host_ns = row.get("host_monotonic_ns")
                if not isinstance(host_ns, int) or host_ns in seen_native_sample_ns:
                    continue
                seen_native_sample_ns.add(host_ns)
                assessment = (
                    assess_prearmed_nominal_handoff_sample(row, expected)
                    if self.mc_prearm_before_simulator
                    else assess_static_nominal_start_sample(row, expected)
                )
                max_residual = assessment.get("maximum_leg_nominal_residual_rad")
                if isinstance(max_residual, (int, float)) and math.isfinite(float(max_residual)):
                    saw_nominal = saw_nominal or (
                        float(max_residual) <= STATIC_NOMINAL_START_MAX_LEG_RESIDUAL_RAD
                    )
                if saw_nominal and assessment.get("native_root_within_existing_safety_gate") is not True:
                    unsafe_after_nominal = {"sample": row, "assessment": assessment}
                    break
                if assessment.get("eligible") is True:
                    consecutive += 1
                    last = {
                        "status": (
                            "PREARMED_NOMINAL_HANDOFF_NATIVE_READY"
                            if self.mc_prearm_before_simulator
                            else "STATIC_NOMINAL_START_NATIVE_READY"
                        ),
                        "required_consecutive_samples": required_samples,
                        "consecutive_samples": consecutive,
                        "sample": row,
                        "assessment": assessment,
                    }
                    if consecutive >= required_samples:
                        record = {
                            "schema": "robotsim.issue80.static-nominal-start-native-preflight.v1",
                            "status": (
                                "PASS_PREARMED_NOMINAL_HANDOFF_NATIVE_READY"
                                if self.mc_prearm_before_simulator
                                else "PASS_STATIC_NOMINAL_START_NATIVE_READY"
                            ),
                            "source": "read_only_native_MjData_capture",
                            "initialization": static,
                            "native_gate": last,
                            "unsafe_after_nominal": None,
                            "mc_prearm_before_simulator": self.mc_prearm_before_simulator,
                            "classification": (
                                "PREARMED_EARLY_JD_ELIGIBILITY_NOT_STANDING"
                                if self.mc_prearm_before_simulator
                                else "SETTLED_STATIC_NOMINAL_SUPPORT_BEFORE_JD"
                            ),
                            "mj_step_calls_by_parent": 0,
                            "qpos_writes_by_parent": 0,
                        }
                        with (self.evidence / "static_nominal_start_native_preflight.json").open("x", encoding="utf-8") as stream:
                            json.dump(record, stream, indent=2)
                            stream.write("\n")
                        return record
                else:
                    consecutive = 0
                    last = {
                        "status": "WAITING_FOR_NATIVE_STATIC_NOMINAL_START",
                        "assessment": assessment,
                        "sample": row,
                    }
            if unsafe_after_nominal is not None:
                break
            time.sleep(0.02)
        record = {
            "schema": "robotsim.issue80.static-nominal-start-native-preflight.v1",
            "status": (
                "FAIL_PREARMED_NOMINAL_HANDOFF_NATIVE_READY"
                if self.mc_prearm_before_simulator
                else "FAIL_STATIC_NOMINAL_START_NATIVE_READY"
            ),
            "source": "read_only_native_MjData_capture",
            "initialization": static,
            "last": last,
            "unsafe_after_nominal": unsafe_after_nominal,
            "mc_prearm_before_simulator": self.mc_prearm_before_simulator,
            "direct_stand_from_passive": self.direct_stand_from_passive,
            "mj_step_calls_by_parent": 0,
            "qpos_writes_by_parent": 0,
        }
        with (self.evidence / "static_nominal_start_native_preflight.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        if self.mc_prearm_before_simulator:
            raise RuntimeError(
                "MC-prearmed nominal handoff did not observe one source-nominal, "
                "bilateral-contact sample inside the existing root-speed guard"
            )
        raise RuntimeError("static nominal-start native preflight did not establish two safe nominal-support samples")

    def dispatch_static_nominal_start_barrier(self, native_preflight: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one evidence-only barrier for the pre-armed ROS observer."""
        path = self.evidence / "static_nominal_start_event.json"
        if path.exists():
            raise RuntimeError("static nominal-start barrier already exists")
        event = {
            "status": "STATIC_NOMINAL_START_READY",
            "dispatch_wall_utc": datetime.now(timezone.utc).isoformat(),
            "dispatch_monotonic_ns": time.monotonic_ns(),
            "input_sent": False,
            "qpos_written": False,
            "native_preflight_status": native_preflight.get("status"),
            "classification": "STATIC_SCENE_INITIALIZATION_BARRIER_NOT_MUJOCO_RESET",
        }
        payload = {
            "schema": "robotsim.issue80.static-nominal-start-event.v1",
            "event": event,
            "native_preflight": native_preflight,
        }
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        self.reset_dispatch_payload = payload
        return payload

    def inspect_x11_target_identity(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        """Recheck the exact X11 process/window identity without changing focus or sending input."""
        result = self.run(
            "x11_identity_recheck_after_windows_focus",
            [sys.executable, str(X11_HELPER), "inspect", "--expected-pid", str(pid)],
            timeout_s=3.0,
            check=False,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {"status": "INVALID_JSON", "stdout": result.stdout[-1000:]}
        matches = [
            window for window in payload.get("windows", [])
            if isinstance(window, Mapping)
            and window.get("pid") == pid
            and window.get("window_id") == window_id
            and window.get("title") == title
            and window.get("viewable") is True
        ]
        verified = result.returncode == 0 and len(matches) == 1
        evidence = {
            "schema": "robotsim.issue80.x11-target-recheck.v1",
            "status": "IDENTITY_VERIFIED" if verified else "IDENTITY_NOT_VERIFIED",
            "expected": {"pid": pid, "window_id": window_id, "title": title},
            "window": matches[0] if verified else None,
            "matching_windows": matches,
            "inspect": payload,
            "returncode": result.returncode,
            "input_sent": False,
            "focus_changed": False,
        }
        path = self.evidence / "reset_x11_identity_recheck.json"
        if path.exists():
            raise RuntimeError("X11 target identity recheck evidence already exists")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
        return evidence

    def activate_verified_windows_foreground(
        self, *, title: str, pid: int, window_id: int, width: int, height: int,
        mode: str = "Activate",
    ) -> dict[str, Any]:
        """Inspect, foreground, or send one reset key through a verified WSLg RAIL window."""
        if mode not in {"Activate", "SendReset"}:
            return {"status": "UNSUPPORTED_WINDOWS_HELPER_MODE", "key_events_sent": 0, "mode": mode}
        helper_copy = self.evidence / WINDOWS_FOCUS_HELPER.name
        if helper_copy.is_symlink() or not helper_copy.is_file():
            return {"status": "WINDOWS_HELPER_NOT_STAGED", "key_events_sent": 0}
        path_result = self.run(
            "windows_foreground_helper_path",
            ["wslpath", "-w", str(helper_copy)],
            timeout_s=3.0,
            check=False,
        )
        windows_script = path_result.stdout.strip()
        if path_result.returncode != 0 or not windows_script:
            return {
                "status": "WINDOWS_HELPER_PATH_UNAVAILABLE",
                "returncode": path_result.returncode,
                "stdout": path_result.stdout[-1000:],
                "stderr": path_result.stderr[-1000:],
                "key_events_sent": 0,
            }
        command = [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
            "-ExecutionPolicy", "Bypass", "-File", windows_script,
            "-ExpectedTitle", title,
            "-ExpectedX11Pid", str(pid),
            "-ExpectedX11WindowId", str(window_id),
            "-ExpectedWidth", str(width),
            "-ExpectedHeight", str(height),
            "-Mode", mode,
        ]
        step_name = "windows_foreground_activation" if mode == "Activate" else "windows_verified_reset_key_dispatch"
        result = self.run(step_name, command, timeout_s=5.0, check=False)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = {
                "status": "WINDOWS_FOCUS_RESULT_INVALID",
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:],
            }
        payload["powershell_returncode"] = result.returncode
        payload["x11_identity_context"] = {
            "pid": pid, "window_id": window_id, "title": title,
        }
        payload["key_events_sent"] = int(payload.get("key_events_sent", 0) or 0)
        return payload

    def finalize_reset_dispatch(self, pid: int, title: str, window_id: int) -> None:
        """Attach image evidence only after the continuous telemetry window ends."""
        event_path = self.evidence / "reset_dispatch_event.json"
        payload = self.reset_dispatch_payload
        if payload is None and event_path.exists():
            payload = json.loads(event_path.read_text(encoding="utf-8"))
        if payload is None:
            raise RuntimeError("cannot finalize Reset evidence without a persisted key-event record")
        payload["before_image"] = self.before_reset_image or {"status": "NOT_CAPTURED"}
        payload["after_image"] = self.capture_reset_image(pid, title, window_id, "after")
        with (self.evidence / "reset_dispatch.json").open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        self.reset_dispatch_payload = payload

    def wait_and_classify_reset(self, pid: int, title: str, window_id: int) -> dict[str, Any]:
        trace = self.evidence / "reset_state_trace.jsonl"
        deadline = min(
            self.deadline,
            time.monotonic() + RESET_OBSERVER_DURATION_S + RESET_OBSERVER_COMPLETION_MARGIN_S,
        )
        while time.monotonic() < deadline and not trace.exists():
            time.sleep(0.05)
        if not trace.exists():
            raise RuntimeError("reset state trace was not produced")
        # The observer is capped at 8 s; wait for its atomic summary with the same bound.
        summary = self.evidence / "reset_state_summary.json"
        action_ns = int(
            ((self.sd_result or {}).get("raw_set_mc_action") or {}).get("wall_monotonic_ns_submitted", 0)
        )
        while time.monotonic() < deadline and not summary.exists():
            if trace.exists() and action_ns:
                anchor_ns = (self.immediate_reset_pose or {}).get(
                    "first_post_reset_upright_odom_monotonic_ns"
                )
                fall = first_post_action_fall(
                    trace.read_text(encoding="utf-8"),
                    action_monotonic_ns=action_ns,
                    post_reset_upright_anchor_monotonic_ns=(
                        int(anchor_ns) if isinstance(anchor_ns, int) else None
                    ),
                )
                if fall:
                    self.reset_classification = "RESET_APPLIED_THEN_FELL"
                    self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                    self.reset_analysis = {
                        "classification": self.reset_classification,
                        "post_action_fall": fall,
                        "safety_window_completed": False,
                        "stop_reason": "stopped on first post-SD unsafe odometry sample",
                    }
                    raise RuntimeError(
                        "post-SD odometry crossed the existing standing safety gate; stopped observer and did not proceed to LD"
                    )
            time.sleep(0.02)
        if not summary.exists():
            raise RuntimeError("bounded Reset observer did not finish before its duration plus completion margin")

        # The state trace, not screenshot completion, owns the stability gate.
        self.finalize_reset_dispatch(pid, title, window_id)
        out = self.evidence / "reset_analysis.json"
        result = self.run(
            "classify_reset_state",
            [sys.executable, str(ANALYZER), "--trace", str(trace), "--dispatch", str(self.evidence / "reset_dispatch.json"), "--output", str(out)],
            timeout_s=4.0,
            check=False,
        )
        if out.exists():
            self.reset_analysis = json.loads(out.read_text(encoding="utf-8"))
        else:
            try:self.reset_analysis=json.loads(result.stdout)
            except json.JSONDecodeError:self.reset_analysis={"classification":"ANALYSIS_FAILED","stdout":result.stdout,"stderr":result.stderr}
        observer_summary = json.loads(summary.read_text(encoding="utf-8"))
        self.reset_analysis["standing_baseline_observation"] = observer_summary.get(
            "standing_baseline_observation"
        )
        self.reset_classification = str(self.reset_analysis.get("classification","ANALYSIS_FAILED"))
        return self.reset_analysis

    def wait_and_classify_synchronized_initialization(
        self, pid: int, title: str, window_id: int,
    ) -> dict[str, Any]:
        """Classify the keyframe/JD/SD observation without mislabelling it as Reset."""
        summary_path = self.evidence / "reset_state_summary.json"
        takeover_path = self.evidence / "set_action_capture_result.json"
        deadline = min(
            self.deadline,
            time.monotonic() + RESET_OBSERVER_DURATION_S + RESET_OBSERVER_COMPLETION_MARGIN_S,
        )
        while time.monotonic() < deadline and not summary_path.exists():
            time.sleep(0.02)
        if not summary_path.exists() or not takeover_path.exists():
            raise RuntimeError("synchronized initialization observer did not produce its terminal summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        takeover = json.loads(takeover_path.read_text(encoding="utf-8"))
        after = self.capture_reset_image(pid, title, window_id, "synchronized_after_release")
        baseline = summary.get("standing_baseline_observation")
        passed = (
            takeover.get("status") == "TAKEOVER_SEQUENCE_COMPLETE"
            and summary.get("observer_status") == "STAND_DEFAULT_BASELINE_COMPLETE"
            and isinstance(baseline, Mapping)
            and baseline.get("complete") is True
        )
        result = {
            "schema": "robotsim.issue80.synchronized-initialization-observation.v1",
            "classification": (
                "SYNCHRONIZED_INITIALIZATION_STAND_BASELINE_PASS"
                if passed else "SYNCHRONIZED_INITIALIZATION_STAND_BASELINE_FAIL"
            ),
            "takeover_status": takeover.get("status"),
            "observer_status": summary.get("observer_status"),
            "standing_baseline_observation": baseline,
            "native_mjdata": takeover.get("native_mjdata"),
            "after_release_image": after,
            "reset_or_qpos_teleportation": False,
            "initialization_method": "native_MuJoCo_Pause_Load_key_then_MC_JD_before_Play",
        }
        with (self.evidence / "synchronized_initialization_analysis.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        self.reset_analysis = result
        self.reset_classification = str(result["classification"])
        return result

    def wait_and_classify_static_nominal_start(self) -> dict[str, Any]:
        """Classify the no-Reset static-start JD/SD observation."""
        summary_path = self.evidence / "reset_state_summary.json"
        takeover_path = self.evidence / "set_action_capture_result.json"
        deadline = min(
            self.deadline,
            time.monotonic() + RESET_OBSERVER_DURATION_S + RESET_OBSERVER_COMPLETION_MARGIN_S,
        )
        while time.monotonic() < deadline and not summary_path.exists():
            time.sleep(0.02)
        if not summary_path.exists() or not takeover_path.exists():
            raise RuntimeError("static nominal-start observer did not produce its terminal summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        takeover = json.loads(takeover_path.read_text(encoding="utf-8"))
        baseline = summary.get("standing_baseline_observation")
        native_preflight_path = self.evidence / "static_nominal_start_native_preflight.json"
        native_preflight = (
            json.loads(native_preflight_path.read_text(encoding="utf-8"))
            if native_preflight_path.is_file() else None
        )
        native_status = native_preflight.get("status") if isinstance(native_preflight, Mapping) else None
        native_ready = native_status in {
            "PASS_STATIC_NOMINAL_START_NATIVE_READY",
            "PASS_PREARMED_NOMINAL_HANDOFF_NATIVE_READY",
        }
        passed = (
            isinstance(native_preflight, Mapping)
            and native_ready
            and takeover.get("status") == "TAKEOVER_SEQUENCE_COMPLETE"
            and summary.get("observer_status") == "STAND_DEFAULT_BASELINE_COMPLETE"
            and isinstance(baseline, Mapping)
            and baseline.get("complete") is True
        )
        result = {
            "schema": "robotsim.issue80.static-nominal-start-observation.v1",
            "classification": (
                "STATIC_NOMINAL_START_STAND_BASELINE_PASS"
                if passed else "STATIC_NOMINAL_START_STAND_BASELINE_FAIL"
            ),
            "takeover_status": takeover.get("status"),
            "observer_status": summary.get("observer_status"),
            "standing_baseline_observation": baseline,
            "native_static_preflight": native_preflight,
            "native_mjdata": takeover.get("native_mjdata"),
            "reset_or_runtime_qpos_teleportation": False,
            "initialization_method": "static_x2_start_frame_z_before_vendor_model_load",
            "mc_prearm_before_simulator": self.mc_prearm_before_simulator,
        }
        with (self.evidence / "static_nominal_start_analysis.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        self.reset_analysis = result
        self.reset_classification = str(result["classification"])
        return result

    def evaluate_standing_warmup(self) -> dict[str, Any]:
        """Analyze the completed post-STAND physics window before any motion command."""
        trace_path = self.evidence / "reset_state_trace.jsonl"
        takeover_path = self.evidence / "set_action_capture_result.json"
        observer_path = self.evidence / "reset_state_summary.json"
        if not all(path.is_file() for path in (trace_path, takeover_path, observer_path)):
            raise RuntimeError("required fresh standing warmup trace/results are missing")
        trace_rows = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                trace_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        takeover = json.loads(takeover_path.read_text(encoding="utf-8"))
        observer = json.loads(observer_path.read_text(encoding="utf-8"))
        analysis = analyze_standing_trace(trace_rows, takeover, observer)
        gate = standing_warmup_composite_result(
            observer.get("standing_baseline_observation"),
            analysis,
        )
        gate["analysis_classification"] = analysis.get("classification")
        gate["trace_fsm_reconciliation"] = analysis.get("fsm_trace_reconciliation")
        gate["zero_velocity_publisher_or_command_created"] = False
        output = self.evidence / "standing_baseline_warmup_analysis.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump({"schema": "robotsim.issue80.stand-warmup-composite.v1", "gate": gate, "analysis": analysis}, stream, indent=2)
            stream.write("\n")
        return gate

    def run_m1(self) -> None:
        zero_only_activation = bool(
            getattr(self.args, "maintainer_authorized_zero_only_activation", False)
        )
        activation_probe = bool(getattr(self.args, "maintainer_authorized_activation_probe", False))
        if not self.args.maintainer_authorized_first_motion and not zero_only_activation and not activation_probe:
            self.m1_result = {
                "status": "NOT_AUTHORIZED",
                "reason": "a separate explicit first-motion, zero-only activation, or activation-probe authorization is required",
            }
            return
        if activation_probe and (
            not isinstance(getattr(self.args, "activation_probe_prerequisite", None), Mapping)
            or self.args.activation_probe_prerequisite.get("status") != "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED"
        ):
            self.m1_result = {
                "status": "BLOCKED",
                "reason": "activation-probe prerequisite evidence was not validated before runtime",
            }
            return
        if activation_probe:
            prerequisite_path = self.evidence / "activation_probe_prerequisite.json"
            with prerequisite_path.open("x", encoding="utf-8") as stream:
                json.dump(self.args.activation_probe_prerequisite, stream, indent=2)
                stream.write("\n")
        if self.reset_classification != "AUTO_RESET_PASS":
            raise RuntimeError("first motion requires the unchanged post-Reset standing gate to pass")
        required_remaining_s = (
            TABLE_APPROACH_CHILD_TIMEOUT_S
            if getattr(self.args, "table_approach_mission", None) else 35.0
        )
        if self.remaining() < required_remaining_s:
            raise RunDeadline(
                f"fewer than {required_remaining_s:g} seconds remain after AUTO_RESET_PASS; M1 was not started"
            )
        if self.network_mode != "bridge":
            raise RuntimeError("verified bridge isolation is required before M1")
        if getattr(self.args, "capture_mc_rl_debug", False):
            self.start_mc_rl_debug_capture()
        if self.lidar_overlay_source is not None:
            self.docker_exec(
                "start_lidar_localizer",
                "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
                "timeout 12s python3 /workspace/evidence/issue80_lidar_localization.py --duration-s 9.5 "
                "> /workspace/evidence/lidar_localizer_runtime.json 2> /workspace/evidence/lidar_localizer_runtime.stderr",
                timeout_s=4.0,
                detached=True,
            )
        command = build_m1_child_command(self.args, self.container)
        child_timeout_s = TABLE_APPROACH_CHILD_TIMEOUT_S if getattr(self.args, "table_approach_mission", None) else 30.0 if zero_only_activation or activation_probe else (
            75.0
            if getattr(self.args, "post_locomotion_convergence_observation_s", None) is not None
            else 45.0
        )
        result = self.run(
            "m1_first_motion_diagnostic", command,
            timeout_s=min(child_timeout_s, self.remaining()), check=False,
        )
        result_file = self.evidence / "m1_first_motion_result.json"
        if result_file.exists():
            try:
                self.m1_result = decode_m1_child_result(result_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.m1_result = {
                    "status": "CHILD_RESULT_INVALID",
                    "fault": f"child result could not be decoded: {type(exc).__name__}: {exc}",
                    "child_returncode": result.returncode,
                    "child_stdout_tail": result.stdout[-4000:],
                    "child_stderr_tail": result.stderr[-4000:],
                }
        else:
            self.m1_result = {"status":"NO_RESULT_FILE","returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}

    def start_mc_rl_debug_capture(self) -> None:
        """Start one passive, schema-verified MC debug subscriber before M1."""
        schema_verification = self.mc_rl_debug_schema_verification
        if schema_verification is None:
            schema_verification = self.verify_mc_rl_debug_overlay()
        if schema_verification.get("status") != "PASS":
            raise RuntimeError("McRLDebug overlay did not pass source verification before capture")
        overlay_setup = self.evidence / "mc_debug_overlay/install/setup.bash"
        if not overlay_setup.is_file():
            raise RuntimeError("McRLDebug capture requested but the evidence-local interface overlay is not built")
        start = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/evidence/mc_debug_overlay/install/setup.bash && "
            "python3 /workspace/evidence/issue80_mc_rl_debug_capture.py "
            "--output /workspace/evidence/mc_rl_debug_trace.jsonl "
            "--ready-marker /workspace/evidence/mc_rl_debug_ready.json "
            "--summary /workspace/evidence/mc_rl_debug_summary.json "
            "--ready-timeout-s 5 --max-duration-s 80 "
            "> /workspace/evidence/mc_rl_debug_capture_console.log 2>&1"
        )
        self.docker_exec("start_mc_rl_debug_capture", start, timeout_s=5.0, detached=True)
        wait_ready = (
            "timeout 7s bash -lc 'until test -f /workspace/evidence/mc_rl_debug_ready.json; "
            "do sleep 0.05; done'"
        )
        wait_result = self.docker_exec(
            "wait_mc_rl_debug_capture_ready", wait_ready, timeout_s=8.0, check=False
        )
        ready_path = self.evidence / "mc_rl_debug_ready.json"
        if not ready_path.is_file():
            console_path = self.evidence / "mc_rl_debug_capture_console.log"
            console_tail = console_path.read_text(encoding="utf-8", errors="replace")[-3000:] if console_path.is_file() else ""
            raise RuntimeError(
                "McRLDebug observer did not write its readiness record; "
                f"wait rc={wait_result.returncode}; observer log tail={console_tail!r}"
            )
        self.mc_rl_debug_capture = json.loads(ready_path.read_text(encoding="utf-8"))
        if self.mc_rl_debug_capture.get("status") != "READY":
            raise RuntimeError(
                "version-matched McRLDebug publisher was not uniquely discovered before M1: "
                f"{self.mc_rl_debug_capture.get('status')}: {self.mc_rl_debug_capture.get('fault')}"
            )
        self.mc_rl_debug_capture["schema_verification"] = schema_verification

    def verify_mc_rl_debug_overlay(self) -> dict[str, Any]:
        """Fail closed unless the temporary ROS interface overlay matches both installed runtimes."""
        schema_path = self.evidence / "mc_rl_debug_schema.json"
        provenance_path = self.evidence / "MC_RL_DEBUG_SCHEMA_PROVENANCE.json"
        message_path = self.evidence / "mc_debug_overlay/src/aimdk_msgs/msg/McRLDebug.msg"
        header_path = self.evidence / "mc_debug_overlay/src/aimdk_msgs/msg/MessageHeader.msg"
        overlay_setup = self.evidence / "mc_debug_overlay/install/setup.bash"
        if not all(path.is_file() for path in (schema_path, provenance_path, message_path, header_path, overlay_setup)):
            raise RuntimeError("McRLDebug schema evidence or generated overlay is incomplete")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_fields = [
            {"name": "header", "type": "aimdk_msgs/msg/MessageHeader"},
            {"name": "obs_vec", "type": "float32[]"},
            {"name": "action_vec", "type": "float32[]"},
        ]
        provenance_fields = provenance.get("fields_in_order", [])
        provenance_field_projection = [
            {"name": item.get("name"), "type": item.get("type")}
            for item in provenance_fields if isinstance(item, dict)
        ]
        if (
            provenance.get("classification") != "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY"
            or provenance.get("type") != "aimdk_msgs/msg/McRLDebug"
            or provenance_field_projection != expected_fields
            or provenance.get("field_metadata_matches_in_mc_and_sim") is not True
        ):
            raise RuntimeError("McRLDebug provenance does not match the reviewed diagnostic schema")
        expected_message = "MessageHeader header\nfloat32[] obs_vec\nfloat32[] action_vec\n"
        if message_path.read_text(encoding="utf-8") != expected_message:
            raise RuntimeError("McRLDebug overlay message definition differs from the introspected fields")
        header_source = RUNTIME / "aimdk_msgs_ws/src/aimdk_msgs/interface/robot/common/msg/MessageHeader.msg"
        if not header_source.is_file():
            raise RuntimeError("version-matched MessageHeader source is unavailable")
        header_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
        if (
            header_hash != provenance.get("message_header_source_sha256")
            or hashlib.sha256(header_source.read_bytes()).hexdigest() != header_hash
        ):
            raise RuntimeError("McRLDebug overlay MessageHeader is not the exact installed source file")
        expected_libraries = {
            "mc": RUNTIME / "mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so",
            "sim": RUNTIME / "sim_mujoco/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so",
        }
        results = schema.get("results")
        if not isinstance(results, list) or len(results) != 2:
            raise RuntimeError("introspection evidence must include both MC and simulator libraries")
        observed_by_sha = {item.get("sha256"): item for item in results if isinstance(item, dict)}
        library_hashes = {}
        for label, path in expected_libraries.items():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            recorded = provenance.get(f"{label}_introspection_library_sha256")
            item = observed_by_sha.get(actual)
            if actual != recorded or item is None:
                raise RuntimeError(f"{label} McRLDebug introspection library hash does not match evidence")
            fields = item.get("fields", [])
            if [field.get("name") for field in fields] != ["header", "obs_vec", "action_vec"]:
                raise RuntimeError(f"{label} McRLDebug field order differs from the overlay")
            if (
                fields[0].get("nested", {}).get("message") != "MessageHeader"
                or any(field.get("type_id") != 1 or field.get("is_array") is not True for field in fields[1:])
                or any(field.get("array_size") != 0 or field.get("is_upper_bound") is not False for field in fields[1:])
            ):
                raise RuntimeError(f"{label} McRLDebug vector types are not unbounded float32 sequences")
            library_hashes[label] = actual
        return {
            "status": "PASS",
            "classification": "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY",
            "type": "aimdk_msgs/msg/McRLDebug",
            "fields": expected_fields,
            "message_header_sha256": header_hash,
            "introspection_library_sha256": library_hashes,
            "overlay_setup": str(overlay_setup),
        }

    def cleanup(self) -> None:
        if not self.container_created:
            self.cleanup_evidence["attempted"] = False
            self.cleanup_evidence["reason"] = "container was not created"
            return
        self.cleanup_evidence["attempted"] = True
        self.cleanup_evidence["normal_path"] = (
            "Python finally copies audit files, then issues docker stop --time 2 and docker rm."
        )
        self.cleanup_evidence["process_crash_or_sigkill_cleanup_guaranteed"] = False
        try:
            copied = self.run(
                "copy_tmpfs_audit",
                ["docker", "exec", self.container, "bash", "-lc",
                 "cp -n /agibot/data/var/MotionControlModule/runtime.yaml /workspace/evidence/runtime_audit_mc.yaml 2>/dev/null || true; cp -n /agibot/data/var/mc/audit.log /workspace/evidence/runtime_audit_mc-audit.log 2>/dev/null || true"],
                timeout_s=3.0,
                check=False,
            )
            self.cleanup_evidence["operations"].append({
                "operation": "copy_tmpfs_audit",
                "returncode": copied.returncode,
            })
        except Exception:
            self.cleanup_evidence["operations"].append({"operation": "copy_tmpfs_audit", "error": "exception"})
        try:
            inspect = subprocess.run(
                ["docker", "inspect", "--type=container", self.container_id or self.container],
                text=True, capture_output=True, timeout=3.0,
            )
            if inspect.returncode == 0:
                with (self.evidence / "container-inspect-final.json").open("x", encoding="utf-8") as f:
                    f.write(inspect.stdout)
            self.cleanup_evidence["operations"].append({
                "operation": "inspect_before_stop",
                "returncode": inspect.returncode,
            })
        except Exception:
            self.cleanup_evidence["operations"].append({"operation": "inspect_before_stop", "error": "exception"})
        for operation, cmd in (
            ("docker_stop", ["docker", "stop", "--time", "2", self.container]),
            ("docker_rm", ["docker", "rm", self.container]),
        ):
            try:
                completed = subprocess.run(cmd, text=True, capture_output=True, timeout=4.0)
                self.cleanup_evidence["operations"].append({
                    "operation": operation,
                    "returncode": completed.returncode,
                })
            except Exception:
                self.cleanup_evidence["operations"].append({"operation": operation, "error": "exception"})
        verify: subprocess.CompletedProcess[str] | None = None
        listed: subprocess.CompletedProcess[str] | None = None
        verify_error = None
        list_error = None
        try:
            verify = subprocess.run(
                ["docker", "inspect", "--type=container", self.container_id or self.container],
                text=True, capture_output=True, timeout=3.0,
            )
        except Exception as exc:
            verify_error = f"{type(exc).__name__}: {exc}"
        try:
            listed = subprocess.run(
                ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
                text=True, capture_output=True, timeout=3.0,
            )
        except Exception as exc:
            list_error = f"{type(exc).__name__}: {exc}"
        absence = assess_container_absence(
            inspect_returncode=verify.returncode if verify else -1,
            list_returncode=listed.returncode if listed else -1,
            list_stdout=listed.stdout if listed else "",
            container_name=self.container,
            container_id=self.container_id,
        )
        self.cleanup_evidence["container_absent_after_cleanup"] = absence["container_absent"]
        self.cleanup_evidence["operations"].append({
            "operation": "verify_container_absent",
            **absence,
            "inspect_error": verify_error,
            "container_list_error": list_error,
        })

    def apply_container_teardown_gate(self, result: dict[str, Any]) -> None:
        result["container_teardown"] = self.cleanup_evidence
        if (
            self.cleanup_evidence.get("attempted")
            and not self.cleanup_evidence.get("container_absent_after_cleanup", False)
        ):
            result["status"] = "BLOCKED"
            result["fault"] = result.get("fault") or "container teardown was not verified after runtime"

    def classify_mc_shutdown(self) -> dict[str, Any]:
        candidates = (
            self.evidence / "mc-console.log",
            self.evidence / "mc_logs" / "mc.log",
        )
        inspected = []
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            inspected.append(str(path))
            if re.search(r"SIGSEGV|segmentation fault", text, flags=re.IGNORECASE):
                return {
                    "classification": "MC_SHUTDOWN_SIGSEGV",
                    "evidence_files": inspected,
                    "note": "shutdown crash is separate from the standing/walking experiment result",
                }
        return {
            "classification": "MC_SHUTDOWN_SIGSEGV_NOT_OBSERVED_IN_CAPTURED_LOGS",
            "evidence_files": inspected,
            "note": "absence from captured logs does not prove a clean MC shutdown",
        }

    def execute(self) -> dict[str, Any]:
        result = {
            "schema": "robotsim.issue80.m1.auto-reset-run.v4",
            "status": "NOT_AUTHORIZED",
            "attempt_id": self.args.attempt_id,
            "source": "node.locomotion_velocity",
            "container": self.container,
            "runtime_image_digest": IMAGE_DIGEST,
            "runtime_limit_s": self.args.max_runtime_s,
            "reset_classification": "NOT_ATTEMPTED",
            "synchronized_paused_initialization": self.synchronized_paused_initialization,
            "static_nominal_start_initialization": self.static_nominal_start_initialization,
            "mc_prearm_before_simulator": self.mc_prearm_before_simulator,
            "m1_status": "NOT_RUN",
            "mc_rl_debug_capture": None,
            "mc_rl_debug_overlay": None,
            "turning_reassessment_prerequisite": getattr(
                self.args, "turning_reassessment_provenance", None
            ),
            "outcome_classification": "STANDING_CONTROL_NOT_READY",
            "standing_warmup_composite": None,
            "post_locomotion_convergence_observation_s": getattr(
                self.args, "post_locomotion_convergence_observation_s", None
            ),
            "post_locomotion_convergence_profile": getattr(
                self.args, "post_locomotion_convergence_profile", None
            ),
            "fault": None,
        }
        standing_only = bool(getattr(self.args, "maintainer_authorized_standing_takeover", False))
        first_motion_authorized = bool(getattr(self.args, "maintainer_authorized_first_motion", False))
        zero_only_activation_authorized = bool(
            getattr(self.args, "maintainer_authorized_zero_only_activation", False)
        )
        activation_probe_authorized = bool(
            getattr(self.args, "maintainer_authorized_activation_probe", False)
        )
        result["zero_only_activation_authorized"] = zero_only_activation_authorized
        result["activation_probe_authorized"] = activation_probe_authorized
        if not standing_only and not first_motion_authorized and not zero_only_activation_authorized and not activation_probe_authorized:
            result["fault"] = "explicit standing, zero-only activation, activation-probe, or first-motion authorization is required before runtime"
            return result
        if activation_probe_authorized:
            try:
                prerequisite = getattr(self.args, "activation_probe_prerequisite", None)
                if not isinstance(prerequisite, Mapping):
                    prerequisite = validate_activation_probe_prerequisite(
                        getattr(self.args, "activation_probe_prerequisite_evidence", "")
                    )
                if prerequisite.get("status") != "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED":
                    raise ValueError("activation-probe prerequisite is not source-supported")
                self.args.activation_probe_prerequisite = dict(prerequisite)
                result["activation_probe_prerequisite"] = dict(prerequisite)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                result["status"] = "BLOCKED"
                result["m1_status"] = "NOT_RUN_ACTIVATION_PROBE_PREREQUISITE_FAILED"
                result["fault"] = f"activation-probe prerequisite failed before runtime: {type(exc).__name__}: {exc}"
                return result
        try:
            if (
                getattr(self.args, "run65_right_turn_qualification", False)
                or getattr(self.args, "run65_route_integration", False)
                or getattr(self.args, "run65_waypoint_b_only", False)
                or getattr(self.args, "run65_waypoint_b_short_diagnostic", False)
                or getattr(self.args, "run65_waypoint_b_extended_diagnostic", False)
                or getattr(self.args, "run65_curved_path_diagnostic", False)
                or bool(getattr(self.args, "route_config", None))
                or bool(getattr(self.args, "local01_scene_world_poses", None))
            ):
                if getattr(self.args, "table_approach_mission", None):
                    preflight = self.args.configured_table_approach_preflight
                    staged_static_path = self.evidence / "local01_scene" / "static_scene_preflight.json"
                    if staged_static_path.is_symlink() or not staged_static_path.is_file():
                        raise RuntimeError("staged Local01 wrapper static preflight is missing or unsafe")
                    local01_static = json.loads(staged_static_path.read_text(encoding="utf-8"))
                    if local01_static.get("status") != "PASS_STATIC_SCENE_CLEARANCE":
                        raise RuntimeError("Local01 wrapper static clearance preflight did not pass")
                    result["local01_scene_static_preflight"] = local01_static
                    result["table_approach_mission_input"] = {
                        "source_local01_commit": preflight["source"]["source_commit"],
                        "source_scene_world_poses_sha256": preflight["source"]["source_hashes"]["scene_world_poses"],
                        "mission_sha256": preflight["mission_sha256"],
                        "static_plan_sha256": preflight["plan_sha256"],
                        "classification": preflight["mission"]["classification"],
                        "execution_status": "INPUT_PREFLIGHT_ONLY",
                    }
                    if self.static_nominal_start_initialization:
                        static_bundle = (
                            self.local01_scene_bundle or {}
                        ).get("static_nominal_start_initialization", {})
                        static_init = (
                            static_bundle.get("initialization")
                            if isinstance(static_bundle, Mapping) else None
                        )
                        if not isinstance(static_init, Mapping):
                            raise RuntimeError("static nominal-start bundle has no derived frame-Z offset")
                        transferred = validate_static_nominal_start_table_approach_clearance(
                            preflight["plan"],
                            root_z_offset_m=float(static_init["derived_frame_z_offset_m"]),
                        )
                        result["static_nominal_start_table_approach_clearance"] = transferred
                        if transferred.get("status") != "PASS_STATIC_NOMINAL_START_TABLE_APPROACH_CLEARANCE":
                            raise RuntimeError("static nominal-start table-approach clearance transfer did not pass")
                elif getattr(self.args, "local01_scene_world_poses", None):
                    local01_static = self.args.local01_scene_static_preflight
                    if local01_static.get("status") != "PASS_STATIC_SCENE_CLEARANCE":
                        raise RuntimeError("Local01 actual-scene static clearance preflight did not pass")
                    with (self.evidence / "local01_scene" / "static_scene_preflight.json").open("x", encoding="utf-8") as stream:
                        json.dump(local01_static, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                    result["local01_scene_static_preflight"] = local01_static
                    result["local01_scene_route_input"] = {
                        "source_local01_commit": self.args.local01_scene_source["source_commit"],
                        "source_scene_world_poses_sha256": self.args.local01_scene_config_sha256,
                        "derived_route_sha256": self.route_config_sha256,
                        "route_kind": "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING",
                        "route_id": self.args.configured_local01_short_route_input["route_id"],
                        "expected_reset_odom_pose": self.args.configured_local01_short_route_input["reset_anchor_expected_odom"],
                        "runtime_reset_odom_observed": False,
                        "full_table_docking_reference": self.args.configured_local01_short_route_input["safe_table_docking_reference"],
                    }
                else:
                    workcell = inspect_route_workcell_geometry(RUNTIME)
                    geometry_path = self.evidence / "workcell_geometry_preflight.json"
                    with geometry_path.open("x", encoding="utf-8") as stream:
                        json.dump(workcell, stream, indent=2)
                        stream.write("\n")
                    result["workcell_geometry_preflight"] = workcell
                if getattr(self.args, "route_config", None):
                    result["route_config_input"] = {
                        "sha256": self.route_config_sha256,
                        "staged_path": str(self.evidence / "route_config.json"),
                        "route_id": self.args.configured_waypoint_docking_route.route_id,
                        "scene_clearance": validate_waypoint_docking_scene_clearance(
                            self.args.configured_waypoint_docking_route,
                            runtime_robot_footprint_radius_m=float(
                                workcell["robot_collision_footprint_radial_upper_bound_m"]
                            ),
                            verified_local_clear_radius_m=float(workcell["verified_local_clear_radius_m"]),
                        ),
                        "execution_status": "INPUT_PREFLIGHT_ONLY",
                    }
            if getattr(self.args, "run65_route_integration", False):
                result["route_turn_qualification_evidence"] = self.stage_route_turn_qualification_evidence()
            self.prepare_foundationwalk_diagnostic_overlay()
            self.prepare_mc_rl_debug_overlay()
            self.stage_research_scripts()
            self.prepare_native_mjdata_capture()
            self.prepare_mc_prearm_writeback()
            self.start_container()
            if self.mc_prearm_before_simulator:
                result["mc_prearm_seed"] = self.seed_mc_prearm_generated_config()
                result["mc_prearm"] = self.start_mc_prearmed_before_simulator()
                # The persistent action client is ready before any native
                # simulator step, but it deliberately has no state samples yet.
                self.start_observer(prearmed_before_simulator=True)
                self.start_sim_first(start_mc=False)
            elif self.synchronized_paused_initialization:
                self.start_sim_first(start_mc=False)
            else:
                self.start_sim_first()
            pid: int | None = None
            title: str | None = None
            window_id: int | None = None
            if not self.static_nominal_start_initialization:
                pid, title, window_id = self.inspect_window()
            if self.local01_scene_mode:
                runtime_scene_filename = str(
                    self.local01_scene_bundle["runtime_scene_filename"]
                ) if self.local01_scene_bundle is not None else "scene_workspace_runtime.xml"
                scene_sha = hashlib.sha256(
                    (self.evidence / "local01_scene" / runtime_scene_filename).read_bytes()
                ).hexdigest()
                if self.static_nominal_start_initialization:
                    result["scene_runtime_verification"] = {
                        "status": "PASS_STATIC_WRAPPER_HASH_VERIFIED_NO_GUI_DEPENDENCY",
                        "expected_scene_model_title_fragment": "RAICOM X2 room workspace",
                        "observed_window_title": None,
                        "scene_runtime_bundle_sha256": scene_sha,
                        "aligned_start_table_approach": self.aligned_start_table_approach,
                    }
                    static_start = self.local01_scene_bundle.get(
                        "static_nominal_start_initialization"
                    ) if self.local01_scene_bundle is not None else None
                    if not isinstance(static_start, Mapping) or static_start.get("enabled") is not True:
                        raise RuntimeError("static nominal-start scene bundle is missing")
                    result["static_nominal_start_preflight"] = static_start
                else:
                    assert title is not None
                    title_matches = (
                        "RAICOM X2 room workspace" in title
                        or (self.lidar_overlay_source is not None and title == "MuJoCo")
                    )
                    result["scene_window_verification"] = {
                        "status": "PASS" if title_matches else "FAIL",
                        "expected_scene_model_title_fragment": (
                            "RAICOM X2 room workspace or generic MuJoCo after verified LiDAR overlay"
                            if self.lidar_overlay_source is not None else "RAICOM X2 room workspace"
                        ),
                        "observed_window_title": title,
                        "scene_runtime_bundle_sha256": scene_sha,
                        "aligned_start_table_approach": self.aligned_start_table_approach,
                    }
                    if not title_matches:
                        raise RuntimeError("MuJoCo window does not identify the configured Local01 RAICOM scene")
                if self.synchronized_paused_initialization:
                    synchronization = self.local01_scene_bundle.get(
                        "synchronized_nominal_initialization"
                    ) if self.local01_scene_bundle is not None else None
                    if not isinstance(synchronization, Mapping) or synchronization.get("enabled") is not True:
                        raise RuntimeError("synchronized nominal keyframe bundle is missing")
                    result["synchronized_initialization_preflight"] = synchronization
            if not self.static_nominal_start_initialization:
                assert pid is not None and title is not None and window_id is not None
                self.before_reset_image = self.capture_reset_image(pid, title, window_id, "before")
            if self.synchronized_paused_initialization:
                assert pid is not None and title is not None and window_id is not None
                pause = self.dispatch_native_pause_toggle(pid, title, window_id)
                result["synchronized_pause_dispatch"] = pause
                result["synchronized_pause_quiescence"] = self.verify_native_pause_quiescence(pause)
                self.start_mc_after_sim_topics()
            if not self.mc_prearm_before_simulator:
                self.capture_service_topic_inventory()
                self.start_observer()
            if self.synchronized_paused_initialization:
                assert pid is not None and title is not None and window_id is not None
                result["synchronized_keyframe_selection"] = self.dispatch_native_keyframe_zero(
                    pid, title, window_id
                )
                keyframe_load = self.dispatch_native_load_key(pid, title, window_id)
                self.reset_dispatch_payload = keyframe_load
                result["synchronized_keyframe_load"] = keyframe_load
                result["synchronized_jd_ready"] = self.wait_for_synchronized_jd_ready()
                result["synchronized_physics_release"] = self.dispatch_native_physics_release(
                    pid, title, window_id
                )
                # The pre-armed node performs JD before this release, then
                # validates resumed native stepping before SD.
                result["set_mc_action_sd"] = self.request_sd_and_verify()
                reset = self.wait_and_classify_synchronized_initialization(pid, title, window_id)
            elif self.static_nominal_start_initialization:
                result["static_nominal_start_native_preflight"] = self.verify_native_static_nominal_start()
                result["static_nominal_start_barrier"] = self.dispatch_static_nominal_start_barrier(
                    result["static_nominal_start_native_preflight"]
                )
                result["set_mc_action_sd"] = self.request_sd_and_verify()
                if self.mc_prearm_before_simulator:
                    # Inventory is evidence only.  It occurs after the early
                    # handoff so it cannot delay the first JD request.
                    self.capture_service_topic_inventory()
                reset = self.wait_and_classify_static_nominal_start()
            else:
                assert pid is not None and title is not None and window_id is not None
                self.dispatch_native_reset_button(pid, title, window_id)
                result["reset_immediate_posture"] = self.wait_for_immediate_reset_pose()
                # The pre-armed node sends the source-supported PD -> JD -> SD
                # handoff; SD is gated on fresh upright telemetry and MC STAND state.
                result["set_mc_action_sd"] = self.request_sd_and_verify()
                reset = self.wait_and_classify_reset(pid, title, window_id)
            result["reset_classification"] = reset.get("classification")
            result["reset_analysis"] = reset
            required_startup_status = (
                "SYNCHRONIZED_INITIALIZATION_STAND_BASELINE_PASS"
                if self.synchronized_paused_initialization else "AUTO_RESET_PASS"
            )
            if self.static_nominal_start_initialization:
                required_startup_status = "STATIC_NOMINAL_START_STAND_BASELINE_PASS"
            if reset.get("classification") != required_startup_status:
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                raise RuntimeError(f"post-SD standing safety gate failed: {reset.get('classification')}")

            if getattr(self.args, "require_stand_warmup_composite", False):
                warmup = self.evaluate_standing_warmup()
                result["standing_warmup_composite"] = warmup
                if warmup.get("status") != "PASS":
                    self.outcome_classification = "STAND_WARMUP_COMPOSITE_FAILED"
                    result["m1_status"] = "NOT_RUN_STAND_WARMUP_COMPOSITE_FAILED"
                    result["outcome_classification"] = self.outcome_classification
                    result["status"] = "BLOCKED"
                    result["fault"] = "original final two-second stand warmup composite did not pass"
                    result["next_motion_stage"] = "NOT_RUN_STAND_WARMUP_COMPOSITE_FAILED"
                    return result

            # Video has no bearing on Reset/JD/SD readiness.  Start its passive
            # trace-waiter only after the full standing gate has passed so it
            # cannot contend with the time-critical takeover interval.
            if first_motion_authorized:
                self.start_verified_motion_video_capture()

            if standing_only and not first_motion_authorized:
                # This authorization is limited to the standing takeover gate.
                # It must not fall through to source registration, LD, or velocity.
                baseline = reset.get("standing_baseline_observation")
                result["standing_baseline_observation"] = baseline
                if getattr(self.args, "require_complete_stand_baseline", False) and not standing_baseline_observation_passes(baseline):
                    self.outcome_classification = "STAND_BASELINE_OBSERVATION_INCOMPLETE"
                    result["m1_status"] = "NOT_RUN_STANDING_BASELINE_INCOMPLETE"
                    result["outcome_classification"] = self.outcome_classification
                    result["status"] = "BLOCKED"
                    result["fault"] = "required complete post-readback 8 s STAND_DEFAULT observation did not pass"
                    result["next_motion_stage"] = "NOT_RUN_REQUIRES_COMPLETE_STANDING_BASELINE"
                    return result
                self.outcome_classification = "STANDING_TAKEOVER_PASS"
                result["m1_status"] = "NOT_RUN_STANDING_ONLY"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "PASS"
                result["next_motion_stage"] = "NOT_RUN_REQUIRES_SEPARATE_AUTHORIZATION"
                return result

            self.run_m1()
            result["m1_result"] = self.m1_result
            result["mc_rl_debug_capture"] = self.mc_rl_debug_capture
            result["m1_status"] = (self.m1_result or {}).get("status", "UNKNOWN")
            child_fault = (self.m1_result or {}).get("fault")
            if isinstance(child_fault, str) and child_fault:
                result["fault"] = child_fault
            m1 = self.m1_result or {}
            if getattr(self.args, "maintainer_authorized_activation_probe", False):
                qualification = m1.get("activation_probe_qualification")
                outcome = classify_activation_probe_outcome(
                    qualification if isinstance(qualification, Mapping) else None
                )
                probe = (
                    qualification.get("probe")
                    if isinstance(qualification, Mapping)
                    and isinstance(qualification.get("probe"), Mapping)
                    else {}
                )
                physical_stop = probe.get("physical_stop_gate")
                result["activation_probe_qualification"] = {
                    "classification": outcome,
                    "status": qualification.get("status") if isinstance(qualification, Mapping) else None,
                    "mode_source_activation_pass": (
                        qualification.get("mode_source_activation_pass")
                        if isinstance(qualification, Mapping) else False
                    ),
                    "safe_stand_and_physical_stop_pass": (
                        qualification.get("safe_stand_and_physical_stop_pass")
                        if isinstance(qualification, Mapping) else False
                    ),
                    "late_locomotion_after_abort": (
                        qualification.get("late_locomotion_after_abort")
                        if isinstance(qualification, Mapping) else None
                    ),
                    "route_motion_authorized": False,
                    "physical_stop_gate": (
                        {
                            "status": physical_stop.get("status"),
                            "reason": physical_stop.get("reason"),
                            "final_window": physical_stop.get("final_window"),
                            "final_window_violations": physical_stop.get("final_window_violations"),
                            "braking_travel_from_last_in_pulse_pose": physical_stop.get(
                                "braking_travel_from_last_in_pulse_pose"
                            ),
                        }
                        if isinstance(physical_stop, Mapping) else None
                    ),
                }
                self.outcome_classification = outcome
                result["outcome_classification"] = outcome
                if outcome == "SIMULATION_ONLY_ACTIVATION_ABORT_PHYSICAL_STOP_FAIL":
                    reason = (
                        physical_stop.get("reason")
                        if isinstance(physical_stop, Mapping) else "physical stop evidence unavailable"
                    )
                    result["fault"] = (
                        "LOCOMOTION mode/source activation was confirmed, but the independent "
                        f"STAND/physical-stop gate failed: {reason}"
                    )
                elif outcome == "SIMULATION_ONLY_ACTIVATION_NOT_CONFIRMED":
                    result["fault"] = "fresh synchronized LOCOMOTION/source activation was not confirmed"
                elif outcome == "SIMULATION_ONLY_ACTIVATION_ABORT_LATE_LOCOMOTION":
                    result["fault"] = "LOCOMOTION was observed after abort; rearm remains forbidden"
                result["status"] = (
                    "PASS"
                    if outcome == "SIMULATION_ONLY_ACTIVATION_ABORT_AND_PHYSICAL_STOP_PASS"
                    else "BLOCKED"
                )
                result["next_motion_stage"] = (
                    "NOT_RUN_REQUIRES_SEPARATE_ROTATION_QUALIFICATION"
                    if result["status"] == "PASS"
                    else "NOT_RUN_ACTIVATION_ABORT_OR_PHYSICAL_STOP_GATE_FAILED"
                )
                return result
            if getattr(self.args, "run53_turning_diagnostic", False):
                turning = m1.get("turning_diagnostic")
                result["turning_diagnostic"] = turning
                turn_text = json.dumps(turning, sort_keys=True).lower() if isinstance(turning, Mapping) else ""
                safety_fault = any(
                    token in turn_text
                    for token in M1_SAFETY_FAULT_MARKERS
                )
                self.outcome_classification = (
                    "SIMULATION_ONLY_DIAGNOSTIC_TURNING_PASS"
                    if isinstance(turning, Mapping) and turning.get("status") == "PASS"
                    else "SIMULATION_ONLY_DIAGNOSTIC_TURNING_FAIL"
                    if safety_fault
                    else "SIMULATION_ONLY_DIAGNOSTIC_TURNING_BLOCKED"
                )
                result["outcome_classification"] = self.outcome_classification
                result["status"] = (
                    "PASS" if self.outcome_classification == "SIMULATION_ONLY_DIAGNOSTIC_TURNING_PASS"
                    else "BLOCKED"
                )
                return result
            if getattr(self.args, "run65_right_turn_qualification", False):
                right_turn = m1.get("right_turn_qualification")
                result["right_turn_qualification"] = right_turn
                status = right_turn.get("status") if isinstance(right_turn, Mapping) else None
                self.outcome_classification = (
                    "SIMULATION_ONLY_RIGHT_TURN_AND_STOP_PASS"
                    if status == "PASS"
                    else "SIMULATION_ONLY_RIGHT_TURN_AND_STOP_BLOCKED"
                )
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "PASS" if status == "PASS" else "BLOCKED"
                return result
            if (
                getattr(self.args, "run65_route_integration", False)
                or getattr(self.args, "run65_curved_path_diagnostic", False)
                or getattr(self.args, "run65_waypoint_b_only", False)
                or getattr(self.args, "run65_waypoint_b_short_diagnostic", False)
                or getattr(self.args, "run65_waypoint_b_extended_diagnostic", False)
                or bool(getattr(self.args, "route_config", None))
                or bool(getattr(self.args, "local01_scene_world_poses", None))
            ):
                route = m1.get("route_integration")
                direct_vendor = m1.get("classification") in {
                    "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_DISTANCE_BASELINE",
                    "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET",
                }
                if direct_vendor:
                    direct_pass = m1.get("status") == "PASS"
                    result["vendor_direct_truth_route"] = m1
                    direct_label = (
                        "TABLE_TARGET" if m1.get("classification")
                        == "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET" else "DISTANCE_BASELINE"
                    )
                    self.outcome_classification = (
                        f"SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_{direct_label}_PASS"
                        if direct_pass else f"SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_{direct_label}_BLOCKED"
                    )
                    result["outcome_classification"] = self.outcome_classification
                    result["status"] = "PASS" if direct_pass else "BLOCKED"
                    return result
                result["route_integration"] = route
                route_pass = isinstance(route, Mapping) and route.get("status") == "PASS"
                self.outcome_classification = (
                    str(route.get("classification"))
                    if route_pass and isinstance(route, Mapping)
                    else "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION_BLOCKED"
                    if getattr(self.args, "aligned_start_table_approach", False)
                    else "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION_BLOCKED"
                    if getattr(self.args, "table_approach_mission", None)
                    else "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC_BLOCKED"
                    if getattr(self.args, "run65_curved_path_diagnostic", False)
                    else "SIMULATION_ONLY_EXTENDED_B_DIAGNOSTIC_BLOCKED"
                    if getattr(self.args, "run65_waypoint_b_extended_diagnostic", False)
                    else "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_BLOCKED"
                    if getattr(self.args, "run65_waypoint_b_short_diagnostic", False)
                    else "SIMULATION_ONLY_LOCAL01_SHORT_ROUTE_DIAGNOSTIC_BLOCKED"
                    if getattr(self.args, "local01_scene_world_poses", None)
                    else "SIMULATION_ONLY_WAYPOINT_DOCKING_ROUTE_BLOCKED"
                    if getattr(self.args, "route_config", None)
                    else "SIMULATION_ONLY_WAYPOINT_B_AND_PHYSICAL_STOP_BLOCKED"
                    if getattr(self.args, "run65_waypoint_b_only", False)
                    else "SIMULATION_ONLY_SHORT_ROUTE_INTEGRATION_BLOCKED"
                )
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "PASS" if route_pass else "BLOCKED"
                return result
            if run65_point_goal_enabled(self.args):
                point_goal = m1.get("point_goal_diagnostic")
                result["point_goal_diagnostic"] = point_goal
                self.outcome_classification = classify_point_goal_result(
                    point_goal if isinstance(point_goal, Mapping) else None
                )
                result["outcome_classification"] = self.outcome_classification
                result["status"] = (
                    "PASS"
                    if self.outcome_classification == "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_PASS"
                    else "BLOCKED"
                )
                return result
            activation = m1.get("activation")
            activation_fault = (
                str(activation.get("fault") or "")
                if isinstance(activation, Mapping) else ""
            )
            if activation_fault and not result.get("fault"):
                result["fault"] = activation_fault
            fault_text = " ".join(
                value for value in (str(m1.get("fault") or ""), activation_fault)
                if value
            ).lower()
            registration = m1.get("source_registration")
            ld_action = m1.get("set_mc_action_ld")
            nonzero_calls = int(activation.get("nonzero_publish_calls_returned", 0) or 0) if isinstance(activation, Mapping) else 0
            safety_fault = any(
                token in fault_text
                for token in (*M1_SAFETY_FAULT_MARKERS, "standing")
            )
            if safety_fault and nonzero_calls > 0:
                self.outcome_classification = "FIRST_PHYSICAL_WALKING_FAIL"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "BLOCKED"
                return result
            if safety_fault:
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                result["outcome_classification"] = self.outcome_classification
                result["status"] = "BLOCKED"
                return result
            if isinstance(registration, Mapping) and registration.get("success") is not True:
                self.outcome_classification = "LOCOMOTION_SOURCE_REJECTED"
            elif isinstance(ld_action, Mapping) and (
                ld_action.get("accepted_by_service") is not True
                or ld_action.get("response_header_code") not in (0, None)
            ):
                self.outcome_classification = "LD_REJECTED_WITH_EXACT_CODE"
            elif isinstance(activation, Mapping):
                self.outcome_classification = classify_first_motion_outcome(activation)
            elif result["m1_status"] == "BLOCKED":
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
            result["outcome_classification"] = self.outcome_classification
            result["status"] = (
                "PASS" if self.outcome_classification == "FIRST_PHYSICAL_WALKING_PASS" else "BLOCKED"
            )
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            result["fault"] = self.fault
            result["reset_classification"] = self.reset_classification
            result["set_mc_action_sd"] = self.sd_result
            result["m1_result"] = self.m1_result
            result["outcome_classification"] = self.outcome_classification
            result["status"] = "BLOCKED"
        finally:
            elapsed = time.monotonic() - self.started
            result["motion_video_capture"] = self.finalize_motion_video_capture()
            result["mc_rl_debug_capture"] = self.mc_rl_debug_capture
            result["mc_rl_debug_overlay"] = self.mc_rl_debug_overlay_manifest
            result["wall_elapsed_s"] = elapsed
            result["wall_budget_exhausted"] = elapsed > self.args.max_runtime_s
            if self.reset_analysis:
                result["reset_analysis"] = self.reset_analysis
            self.cleanup()
            result["native_mjdata_capture"] = self.finalize_native_mjdata_capture()
            result["native_mjdata_capture_overlay"] = self.native_mjdata_overlay
            result["native_mjdata_capture_smoke"] = self.native_mjdata_smoke
            result["shutdown_classification"] = self.classify_mc_shutdown()
            self.apply_container_teardown_gate(result)
            if self.foundationwalk_config_overlay_manifest is not None:
                baseline_classification = str(
                    result.get("outcome_classification", self.outcome_classification)
                )
                result["foundationwalk_config_diagnostic"] = {
                    "classification": "SIMULATION_ONLY_DIAGNOSTIC_CONFIG",
                    "manifest": self.foundationwalk_config_overlay_manifest,
                    "underlying_m1_classification": baseline_classification,
                    "source_faithful_original_config_acceptance": "NOT_TESTED",
                }
                result["outcome_classification"] = classify_run38_diagnostic_outcome(
                    baseline_classification
                )
                result["status_scope"] = "diagnostic_config_only; never original_config_acceptance"
            output = self.evidence / "auto-reset-run-result.json"
            with output.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
        return result



def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",required=True)
    parser.add_argument("--container",required=True)
    parser.add_argument("--attempt-id",required=True)
    parser.add_argument("--max-runtime-s",type=float,default=180.0)
    parser.add_argument(
        "--record-mujoco-video",
        action="store_true",
        help="passively record the already Reset-verified MuJoCo window after its first nonzero route command",
    )
    parser.add_argument(
        "--native-mjdata-capture",
        action="store_true",
        help="enable the existing read-only native MuJoCo state/contact capture for this bounded run",
    )
    parser.add_argument(
        "--native-mjdata-capture-interval-steps",
        type=int,
        default=DEFAULT_INTERVAL_STEPS,
        help="native capture sample interval; must remain within the established overlay bounds",
    )
    parser.add_argument("--extra-stop-observation", type=float, default=0.0)
    parser.add_argument("--pulse-duration", type=float, default=1.0)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument("--ramp-start-speed", type=float, default=None)
    parser.add_argument("--ramp-duration-s", type=float, default=0.0)
    parser.add_argument("--startup-speed", type=float, default=None)
    parser.add_argument("--startup-transition-displacement", type=float, default=None)
    parser.add_argument("--startup-transition-timeout-s", type=float, default=0.0)
    parser.add_argument(
        "--maintainer-authorized-first-motion",
        action="store_true",
        help="explicit human authorization for one bounded simulation-only first-motion run",
    )
    parser.add_argument(
        "--maintainer-authorized-zero-only-activation",
        action="store_true",
        help="explicit authorization for one Reset-gated LD/zero-only/stand-recovery test; no nonzero command may be published",
    )
    parser.add_argument(
        "--maintainer-authorized-activation-probe",
        action="store_true",
        help="explicit authorization for one evidence-gated 0.12 m/s, 1.80 s activation/abort qualification; no route motion",
    )
    parser.add_argument(
        "--activation-probe-prerequisite-evidence",
        default=None,
        help="unique governed zero-only preflight run directory required before the one activation probe",
    )
    parser.add_argument(
        "--maintainer-authorized-standing-takeover",
        action="store_true",
        help="explicit human authorization for one simulation-only standing-takeover test; no LD or velocity command",
    )
    parser.add_argument(
        "--require-complete-stand-baseline",
        action="store_true",
        help="fail closed unless 8 s are observed after positive STAND_DEFAULT GetMcAction readback",
    )
    parser.add_argument(
        "--require-stand-warmup-composite",
        action="store_true",
        help="fail closed before M1 unless the post-STAND 8 s warmup and final 2 s physical/target gates pass",
    )
    parser.add_argument(
        "--post-locomotion-convergence-observation-s",
        type=float,
        default=None,
        help="explicit SIMULATION_ONLY extended-B zero-only observation; this task fixes it at the existing 7 s cap",
    )
    parser.add_argument(
        "--capture-mc-rl-debug",
        action="store_true",
        help="start the optional version-matched read-only McRLDebug observer before M1 motion",
    )
    parser.add_argument(
        "--mc-rl-debug-overlay-source",
        default=str(DEFAULT_MC_RL_DEBUG_OVERLAY_SOURCE),
        help="governed prior Issue 80 evidence directory containing the verified local McRLDebug overlay",
    )
    parser.add_argument(
        "--run38-knee-scale-diagnostic",
        action="store_true",
        help=(
            "one SIMULATION_ONLY counterfactual: derived FoundationWalk config with only the two "
            "knee action_scale entries changed from 0.25 to 0.19"
        ),
    )
    parser.add_argument(
        "--run41-min-run-diagnostic",
        action="store_true",
        help=(
            "one SIMULATION_ONLY counterfactual: keep both knee action_scale values at 0.19, "
            "set min_run_velx to 0.1, and abort if measured root speed exceeds 0.2 m/s"
        ),
    )
    parser.add_argument(
        "--run42-stop-confirmation-diagnostic",
        action="store_true",
        help=(
            "reuse Run41 config, keep zero commands until applied STAND_DEFAULT readback or timeout, "
            "and enforce the 0.2 m/s measured root-speed guard"
        ),
    )
    parser.add_argument(
        "--run43-zero-only-stop-diagnostic",
        action="store_true",
        help=(
            "reuse Run41 config and bounded speed guard; follow upstream zero-only stop behavior "
            "without an explicit SetMcAction STAND request"
        ),
    )
    parser.add_argument(
        "--run44-threshold-margin-diagnostic",
        action="store_true",
        help=(
            "reuse the Run41 derived config and Run43 zero-only stop, test only 0.14 m/s "
            "with the same 0.2 m/s measured root-speed guard"
        ),
    )
    parser.add_argument(
        "--run45-ten-second-gait-diagnostic",
        action="store_true",
        help=(
            "reuse Run44 config and zero-only stop for one continuous 10-second command window "
            "with the same 0.2 m/s measured root-speed guard"
        ),
    )
    parser.add_argument(
        "--run46-startup-cruise-threshold-diagnostic",
        action="store_true",
        help=(
            "reuse the Run41 SIMULATION_ONLY config and Run43 zero-only stop; test one "
            "0.12 m/s startup that changes to 0.105 m/s after 10 mm, with a 3-second "
            "pulse and the unchanged 0.2 m/s measured root-speed guard"
        ),
    )
    parser.add_argument(
        "--run48-low-cruise-baseline-diagnostic",
        action="store_true",
        help=(
            "reuse the Run41 SIMULATION_ONLY config and Run43 zero-only stop for one "
            "constant 0.105 m/s, 3-second diagnostic, isolating the low cruise level "
            "without a motion-triggered startup profile"
        ),
    )
    parser.add_argument(
        "--run49-smoothstop-diagnostic",
        action="store_true",
        help=(
            "reuse the Run41 SIMULATION_ONLY config for one 10-second 0.14 m/s "
            "forward run, smoothly decelerating to zero over the final second; keep "
            "the 0.2 m/s root-speed guard and existing zero-only stop gates"
        ),
    )
    parser.add_argument(
        "--run50-threshold-respecting-smoothstop-diagnostic",
        action="store_true",
        help=(
            "use the Run41 SIMULATION_ONLY config for one 10-second 0.14 m/s run, "
            "smoothly reduce only to 0.112 m/s in the final second (above the measured "
            "command cutoff), then use the unchanged zero-only stop protocol"
        ),
    )
    parser.add_argument(
        "--run53-turning-diagnostic",
        action="store_true",
        help=(
            "reuse the verified SIMULATION_ONLY FoundationWalk overlay for one bounded pair: "
            "0.5 s forward warmup plus 4.5 s at 0.12 m/s forward and +/-0.075 rad/s, "
            "capped at 10 s total active commands with unchanged physical safety gates"
        ),
    )
    parser.add_argument(
        "--run65-right-turn-qualification",
        action="store_true",
        help=(
            "reuse the existing Run65 SIMULATION_ONLY FoundationWalk config for one "
            "0.12 m/s, -0.075 rad/s right arc, capped at 5 seconds and stopped at "
            "0.20 m measured turn-origin displacement to reserve 0.05 m for settling"
        ),
    )
    parser.add_argument(
        "--run65-route-integration",
        action="store_true",
        help="execute one short measured waypoint transit and evidence-qualified bounded right corner through M1",
    )
    parser.add_argument(
        "--run65-waypoint-b-only",
        action="store_true",
        help="execute one measured multi-window transit to waypoint B, verify physical stop, then end the route",
    )
    parser.add_argument(
        "--run65-waypoint-b-short-diagnostic",
        action="store_true",
        help=(
            "execute a distinct 0.20 m, 0.15 m-tolerance waypoint diagnostic; "
            "this does not satisfy the original 0.40 m waypoint-B acceptance"
        ),
    )
    parser.add_argument(
        "--run65-waypoint-b-extended-diagnostic",
        action="store_true",
        help=(
            "execute original 0.40 m waypoint B under a SIMULATION_ONLY profile of at most "
            "three independently stopped 5-second windows and 15 seconds aggregate"
        ),
    )
    parser.add_argument(
        "--run65-curved-path-diagnostic",
        action="store_true",
        help=(
            "execute one source-odom shallow right-arc fixed-point route with at most "
            "three independently stopped 5-second windows and 15 seconds aggregate"
        ),
    )
    parser.add_argument(
        "--route-turn-evidence",
        default=None,
        help="governed m1_first_motion_result.json from the passing right-turn qualification run",
    )
    parser.add_argument(
        "--route-config",
        default=None,
        help="generic odom-frame START/WAYPOINT/TABLE_DOCKING JSON for the pinned open-floor workcell; not RAICOM table-clearance validation",
    )
    parser.add_argument(
        "--table-approach-mission",
        default=None,
        help="pinned non-docking Local01 START_ALIGN→WAYPOINT→TABLE_APPROACH mission JSON",
    )
    parser.add_argument(
        "--table-approach-static-plan",
        default=None,
        help="matching read-only RAICOM static table-plan JSON with safe-prefix evidence",
    )
    parser.add_argument(
        "--aligned-start-table-approach",
        action="store_true",
        help=(
            "require the separately versioned SIMULATION_ONLY pre-physics yaw-zero "
            "RAICOM scene wrapper; never substitutes for the original START turn"
        ),
    )
    parser.add_argument(
        "--native-x2-timestep",
        action="store_true",
        help=(
            "simulation-only aligned TABLE_APPROACH wrapper: set the parent scene to the pinned "
            "native X2 0.001 s MuJoCo timestep before physics; no robot or controller field changes"
        ),
    )
    parser.add_argument(
        "--raicom-native-mjdata-route-feedback",
        action="store_true",
        help=(
            "SIMULATION_ONLY route feedback from the staged read-only native MjData trace; "
            "requires the aligned native-timestep TABLE_APPROACH mission and native capture"
        ),
    )
    parser.add_argument(
        "--raicom-continuous-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY: one continuous 50 Hz native-ground-truth TABLE_APPROACH gait stream, "
            "with a rolling intermediate waypoint and the existing 75 s aggregate motion cap"
        ),
    )
    parser.add_argument(
        "--raicom-fixed-window-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY: use fresh per-window 5 s straight or existing Run65 fixed-arc commands, "
            "with the unchanged full stop/rearm gate between windows"
        ),
    )
    parser.add_argument(
        "--raicom-straight-ten-second-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY: use the physically measured 0.12 m/s zero-yaw 10 s gait "
            "window through the static collinear TABLE_APPROACH corridor, with a complete "
            "physical stop/rearm gate between windows"
        ),
    )
    parser.add_argument(
        "--synchronized-paused-initialization",
        action="store_true",
        help=(
            "simulation-only native MuJoCo Pause -> source-MC nominal keyframe -> JD handoff; "
            "requires the aligned-start TABLE_APPROACH mission and native MjData capture"
        ),
    )
    parser.add_argument(
        "--static-nominal-start-initialization",
        action="store_true",
        help=(
            "simulation-only source-MC nominal configuration with a collision-derived "
            "pre-load X2 start-frame Z offset; no GUI Reset, keyframe load, or runtime qpos write"
        ),
    )
    parser.add_argument(
        "--mc-prearm-before-simulator",
        action="store_true",
        help=(
            "simulation-only coordinated startup: start the unmodified MC core and its persistent "
            "JD/SD client before the static nominal-start simulator takes its first physical step; "
            "uses only evidence-local generated-config writeback and a read-only vendor runtime"
        ),
    )
    parser.add_argument(
        "--direct-stand-from-passive",
        action="store_true",
        help=(
            "simulation-only native-Reset startup diagnostic: use the pinned runtime's advertised "
            "PASSIVE_DEFAULT -> STAND_DEFAULT edge directly after the existing fresh bilateral-support gate; "
            "never combine with static/keyframe initialization"
        ),
    )
    parser.add_argument(
        "--raicom-low-cruise-profile",
        action="store_true",
        help=(
            "one evidence-based RAICOM recovery profile: hold the existing first-route command at "
            "0.105 m/s, the prior physical PASS below the 0.20 m/s root-speed guard; "
            "does not change MC, model, limits, or any safety threshold"
        ),
    )
    parser.add_argument(
        "--raicom-zero-forward-heading-acquisition",
        action="store_true",
        help=(
            "SIMULATION_ONLY aligned TABLE_APPROACH profile: acquire the registered source with "
            "a bounded zero-forward yaw command before the first forward gait command; requires "
            "the native-timestep low-cruise route and preserves every existing safety gate"
        ),
    )
    parser.add_argument(
        "--raicom-continuous-ten-second-transit",
        action="store_true",
        help=(
            "SIMULATION_ONLY aligned TABLE_APPROACH profile: use the existing 10 s M1 continuous "
            "command ceiling between measured physical stops, retaining the 75 s aggregate limit "
            "and all root-speed, posture, collision, freshness, and final-stop gates"
        ),
    )
    parser.add_argument(
        "--raicom-forward-gait-yaw-inversion",
        action="store_true",
        help=(
            "SIMULATION_ONLY native-timestep RAICOM profile: invert only the closed-loop "
            "forward-gait yaw command after observed opposite signed yaw response; stationary "
            "zero-forward acquisition retains the documented direct sign"
        ),
    )
    parser.add_argument(
        "--raicom-native-nominal-straight-gait-diagnostic",
        action="store_true",
        help=(
            "one SIMULATION_ONLY native-timestep RAICOM 10 s straight-gait diagnostic at the "
            "existing 0.12 m/s route ceiling; no yaw correction or waypoint completion claim"
        ),
    )
    parser.add_argument(
        "--raicom-vendor-example-forward-baseline",
        action="store_true",
        help=(
            "one SIMULATION_ONLY 0.20 m/s, 5 s forward baseline matching the pinned vendor "
            "locomotion example's command envelope; never a TABLE_APPROACH acceptance claim"
        ),
    )
    parser.add_argument(
        "--raicom-vendor-table-approach",
        action="store_true",
        help=(
            "SIMULATION_ONLY fixed TABLE_APPROACH route using the pinned vendor 0.20 m/s "
            "direct-velocity gait and MuJoCo ground-truth terminal stop"
        ),
    )
    parser.add_argument(
        "--raicom-vendor-direct-distance-m",
        type=float,
        default=None,
        help=(
            "bounded Stage-1 direct vendor-example truth-distance baseline; use only with "
            "--raicom-vendor-table-approach and require >1.0 m actual odom displacement"
        ),
    )
    parser.add_argument(
        "--raicom-vendor-direct-table-target",
        action="store_true",
        help=(
            "Stage-2 direct official-velocity run: stop at the staged raw-odom TABLE_APPROACH "
            "coordinate within the existing 0.15 m preliminary tolerance"
        ),
    )
    parser.add_argument(
        "--raicom-vendor-direct-heading-hold",
        action="store_true",
        help=(
            "Stage-3 direct truth target: use the measured +/-0.075 rad/s forward-arc response "
            "with a 0.04 rad yaw deadband while retaining the existing TABLE_APPROACH stop target"
        ),
    )
    parser.add_argument(
        "--raicom-lidar-overlay-source",
        default=None,
        help="governed evidence directory containing a verified SIMULATION_ONLY LiDAR overlay and plugin directory",
    )
    parser.add_argument(
        "--local01-scene-world-poses",
        default=None,
        help=(
            "pinned Local01 scene_world_poses.json; derives a 0.40 m two-target route along "
            "the Reset heading and validates/loads the RAICOM room/table/chair/mug scene; not table docking"
        ),
    )
    add_run65_point_goal_arguments(parser)
    args=parser.parse_args()
    if sum((
        bool(args.maintainer_authorized_zero_only_activation),
        bool(args.maintainer_authorized_activation_probe),
        bool(args.maintainer_authorized_first_motion),
    )) > 1:
        parser.error("first-motion, zero-only activation, and activation-probe authorization are separate")
    if args.maintainer_authorized_activation_probe:
        if args.maintainer_authorized_standing_takeover or args.require_complete_stand_baseline:
            parser.error("activation-probe qualification cannot be combined with standing-only authorization")
        if any((
            args.run38_knee_scale_diagnostic,
            args.run41_min_run_diagnostic,
            args.run42_stop_confirmation_diagnostic,
            args.run43_zero_only_stop_diagnostic,
            args.run44_threshold_margin_diagnostic,
            args.run45_ten_second_gait_diagnostic,
            args.run46_startup_cruise_threshold_diagnostic,
            args.run48_low_cruise_baseline_diagnostic,
            args.run49_smoothstop_diagnostic,
            args.run50_threshold_respecting_smoothstop_diagnostic,
            args.run53_turning_diagnostic,
            args.run65_right_turn_qualification,
            args.run65_route_integration,
            args.run65_waypoint_b_only,
            args.run65_waypoint_b_short_diagnostic,
            args.run65_waypoint_b_extended_diagnostic,
            args.run65_curved_path_diagnostic,
            args.run65_ground_truth_point_goal,
            args.capture_mc_rl_debug,
        )):
            parser.error("activation-probe qualification is isolated and cannot include a route/controller diagnostic profile")
        if not args.activation_probe_prerequisite_evidence:
            parser.error("activation-probe qualification requires the reviewed zero-only prerequisite evidence directory")
        try:
            args.activation_probe_prerequisite = validate_activation_probe_prerequisite(
                args.activation_probe_prerequisite_evidence
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            parser.error(f"activation-probe prerequisite validation failed before runtime: {exc}")
    elif args.activation_probe_prerequisite_evidence:
        parser.error("activation-probe prerequisite path is only valid with its explicit authorization")
    if args.require_complete_stand_baseline and (
        not args.maintainer_authorized_standing_takeover
        or args.maintainer_authorized_first_motion
    ):
        parser.error("complete STAND baseline gate is only valid for standing-takeover-only runs")
    try:
        convergence_profile = validate_stand_convergence_arguments(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if convergence_profile.get("status") == "SIMULATION_ONLY_POST_LOCOMOTION_STAND_CONVERGENCE_OBSERVATION":
        args.post_locomotion_convergence_profile = convergence_profile
    extra_observation_limit = (
        POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S
        if args.run65_ground_truth_point_goal
        else MAX_EXTRA_POST_STOP_OBSERVE_S
        if args.route_config or args.local01_scene_world_poses
        else MAX_EXTRA_POST_STOP_OBSERVE_S
        if args.post_locomotion_convergence_observation_s is not None
        else 5.0
    )
    if not 0.0 <= args.extra_stop_observation <= extra_observation_limit:
        parser.error(
            f"extra zero-only observation must remain within 0..{extra_observation_limit:g} seconds"
        )
    if not 0.0 < args.pulse_duration <= 10.0:
        parser.error("pulse duration must be positive and no greater than 10 seconds")
    if not 1 <= args.native_mjdata_capture_interval_steps <= 1000:
        parser.error("native MjData capture interval must be in [1, 1000]")
    if not 0.0 < args.forward_speed <= 0.2:
        parser.error("forward speed must be positive and no greater than 0.2 m/s")
    if not 0.0 <= args.ramp_duration_s <= args.pulse_duration:
        parser.error("velocity ramp duration must fit within the pulse duration")
    if args.ramp_duration_s > 0.0 and (
        args.ramp_start_speed is None
        or not 0.0 < args.ramp_start_speed <= args.forward_speed
    ):
        parser.error("ramp start must be positive and no faster than target speed")
    diagnostic_variants = (
        args.run38_knee_scale_diagnostic,
        args.run41_min_run_diagnostic,
        args.run42_stop_confirmation_diagnostic,
        args.run43_zero_only_stop_diagnostic,
        args.run44_threshold_margin_diagnostic,
        args.run45_ten_second_gait_diagnostic,
        args.run46_startup_cruise_threshold_diagnostic,
        args.run48_low_cruise_baseline_diagnostic,
        args.run49_smoothstop_diagnostic,
        args.run50_threshold_respecting_smoothstop_diagnostic,
        args.run53_turning_diagnostic,
        args.run65_right_turn_qualification,
        args.run65_route_integration,
        args.run65_waypoint_b_only,
        args.run65_waypoint_b_short_diagnostic,
        args.run65_waypoint_b_extended_diagnostic,
        args.run65_curved_path_diagnostic,
        args.run65_ground_truth_point_goal,
        bool(args.route_config),
        bool(args.local01_scene_world_poses and not args.table_approach_mission),
        bool(args.table_approach_mission),
    )
    if sum(diagnostic_variants) > 1:
        parser.error("select only one FoundationWalk diagnostic config variant")
    if args.table_approach_mission:
        if args.route_config:
            parser.error("table-approach mission owns its route input; do not also pass --route-config")
        if not args.local01_scene_world_poses or not args.table_approach_static_plan:
            parser.error("table-approach mission requires Local01 scene poses and its matching static plan")
        standing_prearm_only = bool(
            args.mc_prearm_before_simulator
            and args.maintainer_authorized_standing_takeover
            and not args.maintainer_authorized_first_motion
        )
        if not args.maintainer_authorized_first_motion and not standing_prearm_only:
            parser.error(
                "table-approach mission requires explicit first-motion authorization "
                "or the isolated MC-prearmed standing qualification"
            )
        if not args.require_stand_warmup_composite:
            parser.error("table-approach mission requires the complete STAND warmup composite gate")
        if args.route_turn_evidence is not None or args.capture_mc_rl_debug:
            parser.error("table-approach mission uses its own source-grounded yaw-alignment contract")
        required_runtime_s = (
            PREARMED_STANDING_MIN_PARENT_RUNTIME_S
            if standing_prearm_only else TABLE_APPROACH_MIN_PARENT_RUNTIME_S
        )
        if args.max_runtime_s < required_runtime_s:
            parser.error(
                f"table-approach mission requires at least {required_runtime_s:g} s "
                + (
                    "for the bounded prearmed standing qualification and teardown"
                    if standing_prearm_only
                    else "for bounded turn, transit, stop, and teardown windows"
                )
            )
        try:
            table_preflight = (
                validate_local01_aligned_start_table_approach_mission_input(
                    args.local01_scene_world_poses,
                    args.table_approach_mission,
                    args.table_approach_static_plan,
                )
                if args.aligned_start_table_approach
                else validate_local01_table_approach_mission_input(
                    args.local01_scene_world_poses,
                    args.table_approach_mission,
                    args.table_approach_static_plan,
                )
            )
            args.configured_table_approach_preflight = table_preflight
            args.local01_scene_source = table_preflight["source"]
            if args.raicom_fixed_window_table_approach:
                args.fixed_window_open_segment_yaw_preflight = (
                    validate_fixed_window_open_segment_yaw_envelope(
                        table_preflight["plan"]
                    )
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            parser.error(f"table-approach mission rejected before runtime: {type(exc).__name__}: {exc}")
        args.extra_stop_observation = 7.0
    elif args.aligned_start_table_approach:
        parser.error("--aligned-start-table-approach requires --table-approach-mission")
    if sum((
        bool(args.raicom_continuous_table_approach),
        bool(args.raicom_fixed_window_table_approach),
        bool(args.raicom_straight_ten_second_table_approach),
    )) > 1:
        parser.error("continuous, fixed-window, and straight-ten-second TABLE_APPROACH profiles are mutually exclusive")
    if args.native_x2_timestep:
        low_cruise_native_profile = (
            args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
        )
        nominal_straight_native_profile = (
            args.raicom_native_nominal_straight_gait_diagnostic
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.12) <= 1e-9
        )
        vendor_example_native_profile = (
            args.raicom_vendor_example_forward_baseline
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.2) <= 1e-9
            and abs(float(args.pulse_duration) - 5.0) <= 1e-9
        )
        vendor_table_native_profile = (
            args.raicom_vendor_table_approach
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.2) <= 1e-9
            and abs(float(args.pulse_duration) - 5.0) <= 1e-9
        )
        continuous_table_native_profile = (
            args.raicom_continuous_table_approach
            and args.raicom_native_mjdata_route_feedback
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.12) <= 1e-9
        )
        fixed_window_table_native_profile = (
            args.raicom_fixed_window_table_approach
            and args.raicom_native_mjdata_route_feedback
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.12) <= 1e-9
        )
        straight_ten_second_table_native_profile = (
            args.raicom_straight_ten_second_table_approach
            and args.raicom_native_mjdata_route_feedback
            and not args.raicom_low_cruise_profile
            and abs(float(args.forward_speed) - 0.12) <= 1e-9
        )
        if not (
            args.table_approach_mission
            and args.aligned_start_table_approach
            and args.direct_stand_from_passive
            and args.maintainer_authorized_first_motion
            and args.require_stand_warmup_composite
            and args.native_mjdata_capture
            and (
                low_cruise_native_profile
                or nominal_straight_native_profile
                or vendor_example_native_profile
                or vendor_table_native_profile
                or continuous_table_native_profile
                or fixed_window_table_native_profile
                or straight_ten_second_table_native_profile
            )
        ):
            parser.error(
                "native X2 timestep synchronization requires direct-stand, the aligned-start "
                "TABLE_APPROACH mission, native capture, and either the explicit 0.105 m/s "
                "low-cruise profile, the isolated 0.12 m/s straight-gait diagnostic, the isolated "
                "0.20 m/s vendor-example compatibility baseline, or the explicit "
                "continuous 0.12 m/s native-ground-truth route, the fixed-window 0.12 m/s route, "
                "or the straight-ten-second 0.12 m/s route"
            )
    if args.raicom_native_mjdata_route_feedback and not (
        args.native_x2_timestep
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_mjdata_capture
    ):
        parser.error(
            "native MjData route feedback requires native capture and the aligned native-timestep TABLE_APPROACH mission"
        )
    if args.raicom_continuous_table_approach and not (
        args.raicom_native_mjdata_route_feedback
        and args.native_x2_timestep
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_mjdata_capture
        and not args.raicom_native_nominal_straight_gait_diagnostic
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        parser.error(
            "continuous table approach requires native MjData feedback and the aligned 0.12 m/s native TABLE_APPROACH profile"
        )
    if args.raicom_fixed_window_table_approach and not (
        args.raicom_native_mjdata_route_feedback
        and args.native_x2_timestep
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_mjdata_capture
        and not args.raicom_native_nominal_straight_gait_diagnostic
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        parser.error(
            "fixed-window table approach requires native MjData feedback and the aligned 0.12 m/s native TABLE_APPROACH profile"
        )
    if args.raicom_straight_ten_second_table_approach and not (
        args.raicom_native_mjdata_route_feedback
        and args.native_x2_timestep
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_mjdata_capture
        and not args.raicom_native_nominal_straight_gait_diagnostic
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        parser.error(
            "straight-ten-second table approach requires native MjData feedback and the aligned 0.12 m/s native TABLE_APPROACH profile"
        )
    if args.synchronized_paused_initialization:
        if not (
            args.table_approach_mission
            and args.aligned_start_table_approach
            and args.maintainer_authorized_first_motion
            and args.require_stand_warmup_composite
            and args.native_mjdata_capture
        ):
            parser.error(
                "synchronized paused initialization requires the authorized aligned-start TABLE_APPROACH mission, "
                "standing warmup, and native MjData capture"
            )
    if args.static_nominal_start_initialization:
        if args.synchronized_paused_initialization:
            parser.error("static nominal-start initialization cannot be combined with keyframe initialization")
        if not (
            args.table_approach_mission
            and args.aligned_start_table_approach
            and (args.maintainer_authorized_first_motion or args.maintainer_authorized_standing_takeover)
            and args.require_stand_warmup_composite
            and args.native_mjdata_capture
        ):
            parser.error(
                "static nominal-start initialization requires the authorized aligned-start TABLE_APPROACH mission, "
                "standing warmup, and native MjData capture"
            )
        if args.maintainer_authorized_standing_takeover and not args.maintainer_authorized_first_motion and not args.mc_prearm_before_simulator:
            parser.error("static nominal standing-only qualification requires MC pre-arm")
    if args.mc_prearm_before_simulator:
        if not args.static_nominal_start_initialization:
            parser.error("MC pre-arm requires static nominal-start initialization")
        if args.synchronized_paused_initialization:
            parser.error("MC pre-arm cannot be combined with paused-keyframe initialization")
        # The runner sets these internal container properties only for this
        # mode.  No caller can accidentally use a writable vendor runtime.
        args.runtime_readonly = True
        args.mc_sim_generated_tmpfs = True
    if args.direct_stand_from_passive:
        if args.static_nominal_start_initialization or args.synchronized_paused_initialization:
            parser.error("direct stand from passive is only valid for the native Reset startup path")
        if not args.native_mjdata_capture:
            parser.error("direct stand from passive requires native MjData capture")
    if args.raicom_low_cruise_profile:
        if not (
            args.direct_stand_from_passive
            and args.local01_scene_world_poses
            and args.maintainer_authorized_first_motion
            and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
            and (
                not args.table_approach_mission
                or args.aligned_start_table_approach
            )
        ):
            parser.error(
                "RAICOM low-cruise profile requires direct-stand, first-motion authorization, "
                "exactly 0.105 m/s, and either the Local01 short route or its aligned-start "
                "TABLE_APPROACH mission"
            )
    if args.raicom_zero_forward_heading_acquisition and not (
        args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.raicom_low_cruise_profile
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        parser.error(
            "zero-forward heading acquisition is limited to the native-timestep aligned "
            "TABLE_APPROACH low-cruise first-motion profile"
        )
    if args.raicom_continuous_ten_second_transit and not (
        args.raicom_zero_forward_heading_acquisition
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.raicom_low_cruise_profile
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        parser.error(
            "continuous ten-second transit requires the native-timestep aligned TABLE_APPROACH "
            "low-cruise profile with zero-forward heading source acquisition"
        )
    if args.raicom_forward_gait_yaw_inversion and not (
        args.raicom_continuous_ten_second_transit
        and args.raicom_zero_forward_heading_acquisition
        and args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.raicom_low_cruise_profile
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and abs(float(args.forward_speed) - RAICOM_LOW_CRUISE_FORWARD_M_S) <= 1e-9
    ):
        parser.error(
            "forward-gait yaw inversion requires the explicit continuous native-timestep "
            "TABLE_APPROACH heading-acquisition profile"
        )
    if args.raicom_native_nominal_straight_gait_diagnostic and not (
        args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.12) <= 1e-9
    ):
        parser.error(
            "native nominal straight-gait diagnostic requires the aligned native-timestep "
            "TABLE_APPROACH profile at exactly 0.12 m/s without yaw correction"
        )
    if args.raicom_vendor_example_forward_baseline and not (
        args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and args.native_mjdata_capture
        and not args.raicom_native_nominal_straight_gait_diagnostic
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.2) <= 1e-9
        and abs(float(args.pulse_duration) - 5.0) <= 1e-9
    ):
        parser.error(
            "vendor-example forward baseline requires the aligned native-timestep TABLE_APPROACH "
            "startup and exactly 0.20 m/s for 5 s, without route steering"
        )
    if args.raicom_vendor_table_approach and not (
        args.table_approach_mission
        and args.aligned_start_table_approach
        and args.native_x2_timestep
        and args.direct_stand_from_passive
        and args.maintainer_authorized_first_motion
        and args.require_stand_warmup_composite
        and args.native_mjdata_capture
        and not args.raicom_vendor_example_forward_baseline
        and not args.raicom_native_nominal_straight_gait_diagnostic
        and not args.raicom_low_cruise_profile
        and not args.raicom_zero_forward_heading_acquisition
        and not args.raicom_continuous_ten_second_transit
        and not args.raicom_continuous_table_approach
        and not args.raicom_fixed_window_table_approach
        and not args.raicom_straight_ten_second_table_approach
        and not args.raicom_forward_gait_yaw_inversion
        and abs(float(args.forward_speed) - 0.2) <= 1e-9
        and abs(float(args.pulse_duration) - 5.0) <= 1e-9
    ):
        parser.error(
            "vendor table approach requires the aligned native-timestep TABLE_APPROACH "
            "startup and exactly 0.20 m/s for the vendor 5 s input unit"
        )
    if args.raicom_vendor_direct_distance_m is not None:
        if not args.raicom_vendor_table_approach:
            parser.error("--raicom-vendor-direct-distance-m requires --raicom-vendor-table-approach")
        if not 1.0 < float(args.raicom_vendor_direct_distance_m) <= 2.5:
            parser.error("--raicom-vendor-direct-distance-m must be in (1.0, 2.5]")
    if args.raicom_vendor_direct_table_target:
        if not args.raicom_vendor_table_approach:
            parser.error("--raicom-vendor-direct-table-target requires --raicom-vendor-table-approach")
        if args.raicom_vendor_direct_distance_m is not None:
            parser.error("select either direct distance baseline or direct table target, not both")
    if args.raicom_vendor_direct_heading_hold and not args.raicom_vendor_direct_table_target:
        parser.error("--raicom-vendor-direct-heading-hold requires --raicom-vendor-direct-table-target")
    if args.raicom_lidar_overlay_source:
        try:
            lidar_source = Path(args.raicom_lidar_overlay_source).resolve(strict=True)
            lidar_source.relative_to(REPO / "reviews/issue-80")
        except (OSError, ValueError) as exc:
            parser.error(f"LiDAR overlay source must be a governed Issue 80 evidence directory: {exc}")
        if not (
            args.table_approach_mission
            and args.aligned_start_table_approach
            and args.native_x2_timestep
            and args.direct_stand_from_passive
            and args.maintainer_authorized_first_motion
            and args.require_stand_warmup_composite
        ):
            parser.error("LiDAR overlay runtime requires the existing aligned direct-stand table route gates")
    if args.route_config:
        if not args.maintainer_authorized_first_motion:
            parser.error("configured waypoint-docking route requires explicit first-motion authorization")
        if not args.require_stand_warmup_composite:
            parser.error("configured waypoint-docking route requires the complete STAND warmup composite gate")
        if args.route_turn_evidence is not None or args.capture_mc_rl_debug:
            parser.error("configured waypoint-docking route does not accept turn evidence or extra runtime observers")
        try:
            route_input_preflight = validate_waypoint_docking_route_input(args.route_config)
            args.configured_waypoint_docking_route = route_input_preflight["route"]
            args.configured_waypoint_docking_workcell = route_input_preflight["workcell"]
            args.configured_waypoint_docking_clearance = route_input_preflight["clearance"]
            args.configured_waypoint_docking_config_sha256 = route_input_preflight["source_sha256"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            parser.error(f"configured waypoint-docking route rejected before runtime: {exc}")
        # This route uses the already bounded post-STAND observation allowance;
        # it does not relax the composite physical/controller-target gate.
        args.extra_stop_observation = 7.0
    if args.route_config and args.local01_scene_world_poses:
        parser.error("select either the generic odom route JSON or the pinned Local01 scene-world route")
    if args.local01_scene_world_poses and not args.table_approach_mission:
        direct_standing_only = bool(
            args.direct_stand_from_passive
            and args.maintainer_authorized_standing_takeover
            and not args.maintainer_authorized_first_motion
        )
        if not args.maintainer_authorized_first_motion and not direct_standing_only:
            parser.error(
                "Local01 scene route requires separate explicit first-motion authorization "
                "or the direct-stand startup qualification"
            )
        if not args.require_stand_warmup_composite:
            parser.error("Local01 scene route requires the complete STAND warmup composite gate")
        if args.route_turn_evidence is not None or args.capture_mc_rl_debug:
            parser.error("Local01 short-route diagnostic cannot be combined with turn evidence or extra observers")
        try:
            local01_preflight = validate_local01_scene_route_input(args.local01_scene_world_poses)
            args.configured_local01_scene_preflight = local01_preflight
            args.configured_local01_short_route = local01_preflight["route"]
            args.configured_local01_short_route_input = local01_preflight["route_input"]
            args.local01_scene_source = local01_preflight["source"]
            args.local01_scene_static_preflight = local01_preflight["static_scene_preflight"]
            args.local01_scene_config_sha256 = local01_preflight["source_sha256"]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            parser.error(f"Local01 scene-world route rejected before runtime: {type(exc).__name__}: {exc}")
        args.extra_stop_observation = 7.0
    if args.maintainer_authorized_activation_probe and (
        any(diagnostic_variants)
        or args.capture_mc_rl_debug
        or args.ramp_duration_s != 0.0
        or args.ramp_start_speed is not None
        or float(getattr(args, "deceleration_duration_s", 0.0)) != 0.0
        or float(getattr(args, "deceleration_end_speed_m_s", 0.0)) != 0.0
        or args.startup_speed is not None
        or args.startup_transition_displacement is not None
        or args.startup_transition_timeout_s != 0.0
    ):
        parser.error("activation-probe authorization is isolated and cannot be combined with route/controller tuning profiles")
    if any(diagnostic_variants):
        prearmed_standing_variant = bool(
            args.mc_prearm_before_simulator
            and args.static_nominal_start_initialization
            and args.maintainer_authorized_standing_takeover
            and not args.maintainer_authorized_first_motion
        )
        direct_standing_variant = bool(
            args.direct_stand_from_passive
            and args.maintainer_authorized_standing_takeover
            and not args.maintainer_authorized_first_motion
        )
        if not args.maintainer_authorized_first_motion and not (
            args.maintainer_authorized_zero_only_activation and args.run65_curved_path_diagnostic
        ) and not prearmed_standing_variant and not direct_standing_variant:
            parser.error(
                "FoundationWalk diagnostic requires first-motion authorization, except the zero-only activation preflight "
                "or a bounded standing qualification"
            )
        if args.run65_ground_truth_point_goal:
            try:
                args.turning_reassessment_provenance = validate_run65_point_goal_profile(args)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
                parser.error(f"point-goal preflight failed before runtime: {exc}")
        elif args.run53_turning_diagnostic:
            if (
                not args.capture_mc_rl_debug
                or abs(args.extra_stop_observation - 5.0) > 1e-9
                or args.pulse_duration != 1.0
                or abs(args.forward_speed - 0.2) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
            ):
                parser.error(
                    "Run53 uses a fixed 0.5-second forward warmup followed by 4.5-second "
                    "±0.075 rad/s forward-arc turns, 5-second observation, and McRLDebug capture"
                )
        elif args.run65_right_turn_qualification:
            if (
                abs(args.extra_stop_observation - 5.0) > 1e-9
                or args.pulse_duration != 1.0
                or abs(args.forward_speed - 0.2) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
            ):
                parser.error(
                    "right-turn qualification uses the fixed 0.5 s warmup plus 4.5 s "
                    "0.12 m/s and -0.075 rad/s arc, 5 s post-stop observation, "
                    "and the measured 0.20 m trigger with 0.05 m stop reserve"
                )
        elif args.run65_curved_path_diagnostic:
            if (
                abs(args.extra_stop_observation - 5.0) > 1e-9
                or args.pulse_duration != 1.0
                or abs(args.forward_speed - 0.2) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
                or args.route_turn_evidence is not None
            ):
                parser.error(
                    "curved-path profile requires the fixed 5-second zero-stop observation, "
                    "default generic pulse fields, and no right-turn qualification input"
                )
        elif args.run65_route_integration:
            try:
                validate_right_turn_qualification_file(args.route_turn_evidence)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                parser.error(f"route right-turn prerequisite failed before runtime: {exc}")
            if (
                abs(args.extra_stop_observation - 5.0) > 1e-9
                or args.pulse_duration != 1.0
                or abs(args.forward_speed - 0.2) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
            ):
                parser.error(
                    "route integration requires the fixed 5-second zero-stop observation and default generic pulse fields"
                )
        elif (
            args.run65_waypoint_b_only
            or args.run65_waypoint_b_short_diagnostic
            or args.run65_waypoint_b_extended_diagnostic
        ):
            expected_b_observation_s = (
                MAX_EXTRA_POST_STOP_OBSERVE_S
                if args.post_locomotion_convergence_observation_s is not None else 5.0
            )
            validate_b_post_locomotion_observation(
                args.extra_stop_observation,
                extended_b=bool(args.run65_waypoint_b_extended_diagnostic),
                convergence_diagnostic=args.post_locomotion_convergence_observation_s is not None,
            )
            if (
                abs(args.extra_stop_observation - expected_b_observation_s) > 1e-9
                or args.pulse_duration != 1.0
                or abs(args.forward_speed - 0.2) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or args.startup_speed is not None
                or args.startup_transition_displacement is not None
                or args.startup_transition_timeout_s != 0.0
            ):
                parser.error(
                    f"waypoint-B-only route requires its fixed {expected_b_observation_s:g}-second zero-stop observation "
                    "and default generic pulse fields"
                )
        elif args.route_config or args.local01_scene_world_poses:
            try:
                validate_configured_route_motion_profile(args)
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
        else:
            if args.run44_threshold_margin_diagnostic or args.run45_ten_second_gait_diagnostic or args.run49_smoothstop_diagnostic or args.run50_threshold_respecting_smoothstop_diagnostic:
                target_speed = 0.14
            elif args.run46_startup_cruise_threshold_diagnostic or args.run48_low_cruise_baseline_diagnostic:
                target_speed = 0.105
            else:
                target_speed = 0.112
            target_pulse = 10.0 if args.run45_ten_second_gait_diagnostic or args.run49_smoothstop_diagnostic or args.run50_threshold_respecting_smoothstop_diagnostic else 3.0
            if args.run46_startup_cruise_threshold_diagnostic:
                startup_profile_valid = (
                    args.startup_speed is not None
                    and abs(args.startup_speed - 0.12) <= 1e-9
                    and args.startup_transition_displacement is not None
                    and abs(args.startup_transition_displacement - 0.01) <= 1e-9
                    and abs(args.startup_transition_timeout_s - 0.75) <= 1e-9
                )
            else:
                startup_profile_valid = (
                    args.startup_speed is None
                    and args.startup_transition_displacement is None
                    and args.startup_transition_timeout_s == 0.0
                )
            if (
                abs(args.forward_speed - target_speed) > 1e-9
                or abs(args.pulse_duration - target_pulse) > 1e-9
                or abs(args.extra_stop_observation - 5.0) > 1e-9
                or args.ramp_duration_s != 0.0
                or args.ramp_start_speed is not None
                or not startup_profile_valid
                or not args.capture_mc_rl_debug
            ):
                if args.run46_startup_cruise_threshold_diagnostic:
                    parser.error(
                        "Run46 is fixed to 0.12 m/s startup, 0.105 m/s cruise after 10 mm, "
                        "0.75 s transition timeout, 3 s pulse, 5 s post-stop observation, "
                        "no additional ramp, and McRLDebug capture"
                    )
                if args.run48_low_cruise_baseline_diagnostic:
                    parser.error(
                        "Run48 is fixed to constant 0.105 m/s, 3 s pulse, 5 s post-stop "
                        "observation, no ramp or startup profile, and McRLDebug capture"
                    )
                if args.run49_smoothstop_diagnostic:
                    parser.error(
                        "Run49 is fixed to a 10-second 0.14 m/s command with a 1-second "
                        "smooth deceleration, 5-second post-stop observation, no startup/ramp, "
                        "and McRLDebug capture"
                    )
                if args.run50_threshold_respecting_smoothstop_diagnostic:
                    parser.error(
                        "Run50 is fixed to a 10-second 0.14 m/s command with a 1-second "
                        "smooth reduction to 0.112 m/s, 5-second post-stop observation, "
                        "no startup/ramp, and McRLDebug capture"
                    )
                parser.error(
                    f"FoundationWalk diagnostic is fixed to {target_speed} m/s, {target_pulse} s, "
                    "5 s post-stop observation, no velocity ramp, and McRLDebug capture"
                )
    try:
        validate_reserved_evidence_directory(args.evidence_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"evidence directory failed path-governance reservation check: {exc}")
    run=BoundedRun(args)
    result=run.execute()
    print(json.dumps(result,indent=2))
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
