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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from issue80_m1_first_motion import (
    TURN_COMMAND_RAD_S, TURN_FORWARD_COMMAND_M_S, TURN_FORWARD_WARMUP_S,
    TURN_MIN_HEADING_CHANGE_RAD, TURN_PULSE_S,
    ZERO_STOP_MIN_MESSAGES,
)
from issue80_coordinated_pose_goal import (
    POSE_GOAL_APPROACH_WARMUP_S,
    POSE_GOAL_CURVE_FORWARD_M_S,
    POSE_GOAL_HEADING_STOP_ADVANCE_M,
    POSE_GOAL_WARMUP_FORWARD_M_S,
)


REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = REPO / "simulation/mujoco/raicom2026"
X11_HELPER = Path(__file__).with_name("issue80_mujoco_x11_reset.py")
WINDOWS_FOCUS_HELPER = Path(__file__).with_name("issue80_mujoco_windows_focus.ps1")
ANALYZER = Path(__file__).with_name("issue80_m1_reset_analyze.py")
M1_RUNNER = Path(__file__).with_name("issue80_m1_locomotion_probe.py")
POSE_GOAL_CONTROLLER = Path(__file__).with_name("issue80_coordinated_pose_goal.py")
M1_MODE_LOGIC = Path(__file__).with_name("issue80_m1_mode_logic.py")
M1_FIRST_MOTION_LOGIC = Path(__file__).with_name("issue80_m1_first_motion.py")
SET_ACTION_CAPTURE = Path(__file__).with_name("issue80_m1_set_action_capture.py")
MC_RL_DEBUG_CAPTURE = Path(__file__).with_name("issue80_mc_rl_debug_capture.py")
SIM_LAUNCH = Path(__file__).with_name("issue80_launch_sim.sh")
MC_LAUNCH = Path(__file__).with_name("issue80_launch_mc.sh")
IMAGE = "robotsim:humble-raicom-runtime-20261019"
IMAGE_DIGEST = "sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31"
RESET_OBSERVER_DURATION_S = 8.0
RESET_OBSERVER_COMPLETION_MARGIN_S = 3.0
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


def build_m1_child_command(args: argparse.Namespace, container: str) -> list[str]:
    """Build the exact production Docker exec command for the M1 child runner."""
    command = [
        "docker", "exec", "-e", "ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", container, "bash", "-lc",
        "source /opt/ros/humble/setup.bash && source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && python3 /workspace/evidence/issue80_m1_locomotion_probe.py first-motion --maintainer-authorized --reset-gate-verified"
        + " --extra-stop-observation " + str(args.extra_stop_observation)
        + " --pulse-duration " + str(args.pulse_duration)
        + " --forward-speed " + str(args.forward_speed)
        + " --ramp-start-speed " + str(getattr(args, "ramp_start_speed", None) or args.forward_speed)
        + " --ramp-duration-s " + str(getattr(args, "ramp_duration_s", 0.0))
        + " --attempt-id " + str(args.attempt_id)
        + " --robot-model-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/robot_model.yaml"
        + " --mc-sim-config-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml"
        + " --state-timeout 0.5 --ready-timeout 4 --max-tilt-deg 15 --min-root-z 0.52"
        + " --output-dir /workspace/evidence"
    ]
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
    )
    if any(bool(getattr(args, name, False)) for name in diagnostic_names) or run65_point_goal_enabled(args):
        command[-1] += " --max-observed-planar-speed-m-s 0.2"
    stand_timeout_diagnostics = (
        "run42_stop_confirmation_diagnostic", "run43_zero_only_stop_diagnostic",
        "run44_threshold_margin_diagnostic", "run45_ten_second_gait_diagnostic",
        "run46_startup_cruise_threshold_diagnostic", "run48_low_cruise_baseline_diagnostic",
        "run49_smoothstop_diagnostic", "run50_threshold_respecting_smoothstop_diagnostic",
        "run53_turning_diagnostic",
    )
    if any(bool(getattr(args, name, False)) for name in stand_timeout_diagnostics) or run65_point_goal_enabled(args):
        command[-1] += " --zero-until-stand-timeout-s 3.5"
    zero_only_diagnostics = (
        "run43_zero_only_stop_diagnostic", "run44_threshold_margin_diagnostic",
        "run45_ten_second_gait_diagnostic", "run46_startup_cruise_threshold_diagnostic",
        "run48_low_cruise_baseline_diagnostic", "run49_smoothstop_diagnostic",
        "run50_threshold_respecting_smoothstop_diagnostic", "run53_turning_diagnostic",
    )
    if any(bool(getattr(args, name, False)) for name in zero_only_diagnostics) or run65_point_goal_enabled(args):
        command[-1] += " --zero-only-stop"
    if getattr(args, "run49_smoothstop_diagnostic", False):
        command[-1] += " --deceleration-duration-s 1.0"
    if getattr(args, "run50_threshold_respecting_smoothstop_diagnostic", False):
        command[-1] += " --deceleration-duration-s 1.0 --deceleration-end-speed-m-s 0.112"
    if getattr(args, "run53_turning_diagnostic", False):
        command[-1] += " --turning-diagnostic"
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


def first_post_action_fall(serialized_trace: str, *, action_monotonic_ns: int) -> dict[str, Any] | None:
    """Return the first post-SD odometry sample outside the existing height/tilt gate."""
    rows: list[dict[str, Any]] = []
    for line in serialized_trace.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("event") == "state"
            and row.get("topic_key") == "odom"
            and int(row.get("wall_monotonic_ns", 0)) >= action_monotonic_ns
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
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_PASS",
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_FAIL",
        "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
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
        run65 = run65_point_goal_enabled(self.args)
        selected_variants = sum((run38, run41, run42, run43, run44, run45, run46, run48, run49, run50, run53, run65))
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
        if run41 or run42 or run43 or run44 or run45 or run46 or run48 or run49 or run50 or run53 or run65:
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
            else POINT_GOAL_FORWARD_COMMAND_M_S if run65
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
        } if run65 else {
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

    def stage_research_scripts(self) -> None:
        """Stage exact research code snapshots without overwriting evidence."""
        sources = (
            (Path(__file__), Path(__file__).name),
            (X11_HELPER, X11_HELPER.name),
            (WINDOWS_FOCUS_HELPER, WINDOWS_FOCUS_HELPER.name),
            (ANALYZER, ANALYZER.name),
            (M1_RUNNER, M1_RUNNER.name),
            (POSE_GOAL_CONTROLLER, POSE_GOAL_CONTROLLER.name),
            (M1_MODE_LOGIC, M1_MODE_LOGIC.name),
            (M1_FIRST_MOTION_LOGIC, M1_FIRST_MOTION_LOGIC.name),
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

    def start_container(self) -> None:
        cmd = [
            "docker", "run", "-d", "--pull=never", "--name", self.container,
            "--network=bridge", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770,size=64m",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026",
            "--mount", f"type=bind,source={self.evidence},target=/workspace/evidence",
            "--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly",
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
        with (self.evidence / "container-inspect-start.json").open("x", encoding="utf-8") as f:
            f.write(inspect.stdout)

    def start_sim_first(self) -> None:
        self.docker_exec(
            "launch_simulator",
            "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1",
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
        deadline = min(self.deadline, time.monotonic() + RESET_OBSERVER_DURATION_S + 2.0)
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

    def start_observer(self) -> None:
        self.docker_exec(
            "start_prearmed_standing_takeover_observer",
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
            "timeout 45s python3 /workspace/evidence/issue80_m1_set_action_capture.py "
            "--action STAND_DEFAULT --source node.set_mc_action "
            "--output-dir /workspace/evidence "
            "--reset-event-file /workspace/evidence/reset_dispatch_event.json "
            f"--ready-timeout-s 3 --reset-wait-timeout-s 30 --upright-timeout-s {RESET_STARTUP_OBSERVATION_TIMEOUT_S:g} "
            "--joint-stand-timeout-s 4 --set-timeout-s 2 --get-timeout-s 0.75 "
            f"--duration-s {RESET_OBSERVER_DURATION_S:.1f} "
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
        if len(ready.get("fresh_topics", [])) != 5:
            raise RuntimeError(f"Reset precondition failed: live state streams are incomplete: {ready}")
        if ready.get("set_mc_action_service_ready") is not True or ready.get("get_mc_action_service_ready") is not True:
            raise RuntimeError(f"Reset precondition failed: persistent MC action services are not ready: {ready}")
        pre_mode = ready.get("pre_reset_get_mc_action", {})
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

    def dispatch_reset(self, pid: int, title: str, window_id: int) -> None:
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
        final_payload["reset_delivery_attempt_count"] = (
            1 if final_payload.get("event", {}).get("status") == "KEY_DISPATCHED" else 0
        )
        final_payload["reset_input_path_attempt_count"] = 2 if final_payload is not first else 1
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
                fall = first_post_action_fall(
                    trace.read_text(encoding="utf-8"),
                    action_monotonic_ns=action_ns,
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
        self.reset_classification = str(self.reset_analysis.get("classification","ANALYSIS_FAILED"))
        return self.reset_analysis

    def run_m1(self) -> None:
        if not self.args.maintainer_authorized_first_motion:
            self.m1_result = {
                "status": "NOT_AUTHORIZED",
                "reason": "first nonzero simulation-only command requires explicit maintainer authorization",
            }
            return
        if self.reset_classification != "AUTO_RESET_PASS":
            raise RuntimeError("first motion requires the unchanged post-Reset standing gate to pass")
        if self.remaining() < 35.0:
            raise RunDeadline("fewer than 35 seconds remain after AUTO_RESET_PASS; M1 was not started")
        if self.network_mode != "bridge":
            raise RuntimeError("verified bridge isolation is required before M1")
        if getattr(self.args, "capture_mc_rl_debug", False):
            self.start_mc_rl_debug_capture()
        command = build_m1_child_command(self.args, self.container)
        result = self.run("m1_first_motion_diagnostic", command, timeout_s=min(45.0, self.remaining()), check=False)
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
            "m1_status": "NOT_RUN",
            "mc_rl_debug_capture": None,
            "mc_rl_debug_overlay": None,
            "turning_reassessment_prerequisite": getattr(
                self.args, "turning_reassessment_provenance", None
            ),
            "outcome_classification": "STANDING_CONTROL_NOT_READY",
            "fault": None,
        }
        standing_only = bool(getattr(self.args, "maintainer_authorized_standing_takeover", False))
        first_motion_authorized = bool(getattr(self.args, "maintainer_authorized_first_motion", False))
        if not standing_only and not first_motion_authorized:
            result["fault"] = "explicit maintainer standing-takeover or first-motion authorization is required before runtime"
            return result
        try:
            self.prepare_foundationwalk_diagnostic_overlay()
            self.prepare_mc_rl_debug_overlay()
            self.stage_research_scripts()
            self.start_container()
            self.start_sim_first()
            self.capture_service_topic_inventory()
            pid, title, window_id = self.inspect_window()
            self.before_reset_image = self.capture_reset_image(pid, title, window_id, "before")
            self.start_observer()
            self.dispatch_native_reset_button(pid, title, window_id)
            result["reset_immediate_posture"] = self.wait_for_immediate_reset_pose()
            # The pre-armed node sends the source-supported PD -> JD -> SD
            # handoff; SD is gated on fresh upright telemetry and MC STAND state.
            result["set_mc_action_sd"] = self.request_sd_and_verify()
            reset = self.wait_and_classify_reset(pid, title, window_id)
            result["reset_classification"] = reset.get("classification")
            result["reset_analysis"] = reset
            if reset.get("classification") != "AUTO_RESET_PASS":
                self.outcome_classification = "STANDING_CONTROL_NOT_READY"
                raise RuntimeError(f"post-SD standing safety gate failed: {reset.get('classification')}")

            if standing_only and not first_motion_authorized:
                # This authorization is limited to the standing takeover gate.
                # It must not fall through to source registration, LD, or velocity.
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
            result["mc_rl_debug_capture"] = self.mc_rl_debug_capture
            result["mc_rl_debug_overlay"] = self.mc_rl_debug_overlay_manifest
            result["wall_elapsed_s"] = elapsed
            result["wall_budget_exhausted"] = elapsed > self.args.max_runtime_s
            if self.reset_analysis:
                result["reset_analysis"] = self.reset_analysis
            self.cleanup()
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
        "--maintainer-authorized-standing-takeover",
        action="store_true",
        help="explicit human authorization for one simulation-only standing-takeover test; no LD or velocity command",
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
    add_run65_point_goal_arguments(parser)
    args=parser.parse_args()
    extra_observation_limit = (
        POINT_GOAL_MAX_EXTRA_STOP_OBSERVATION_S
        if args.run65_ground_truth_point_goal
        else 5.0
    )
    if not 0.0 <= args.extra_stop_observation <= extra_observation_limit:
        parser.error(
            f"extra zero-only observation must remain within 0..{extra_observation_limit:g} seconds"
        )
    if not 0.0 < args.pulse_duration <= 10.0:
        parser.error("pulse duration must be positive and no greater than 10 seconds")
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
        args.run65_ground_truth_point_goal,
    )
    if sum(diagnostic_variants) > 1:
        parser.error("select only one FoundationWalk diagnostic config variant")
    if any(diagnostic_variants):
        if not args.maintainer_authorized_first_motion:
            parser.error("FoundationWalk diagnostic requires the authorized first-motion gate")
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
