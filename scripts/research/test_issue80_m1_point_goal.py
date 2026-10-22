#!/usr/bin/env python3
"""Offline regression tests for the Issue 80 guarded ground-truth point-goal path."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import issue80_m1_auto_reset_run as runner
from issue80_m1_first_motion import (
    build_forward_point_goal, point_goal_forward_speed_m_s, point_goal_heading_error_rad,
    point_goal_stop_reason, speed_scaled_point_goal_stop_advance_m,
    stale_readback_keys,
)


def valid_reassessment() -> dict:
    passing_gates = {
        "same_sample_source_and_locomotion_handshake": True,
        "measured_heading_in_commanded_direction": True,
        "commanded_forward_arc_envelope": True,
        "physical_stop_gate": True,
        "no_controller_or_post_stop_fault": True,
    }
    return {
        "schema": "robotsim.issue80.run65-forward-arc-reassessment.v1",
        "classification": "SIMULATION_ONLY_DERIVED_FOUNDATIONWALK_TURN_PAIR_PASS_REASSESSED",
        "original_run": {
            "reset_status": "AUTO_RESET_PASS",
            "container_cleanup_verified": True,
            "runtime_image_digest": runner.IMAGE_DIGEST,
            "attempt_id": "run65-test",
            "evidence_dir": "/reviews/run65",
        },
        "acceptance": {
            "source_faithful_original_config": "NOT_TESTED",
            "odometry_stamp_limitation": "source header may regress; receive monotonic time used",
            "turns": [
                {"direction": "left", "reassessed_status": "PHYSICAL_TURN_AND_STOP_PASS", "gates": dict(passing_gates)},
                {"direction": "right", "reassessed_status": "PHYSICAL_TURN_AND_STOP_PASS", "gates": dict(passing_gates)},
            ],
        },
    }


def valid_profile(**overrides) -> argparse.Namespace:
    values = {
        "maintainer_authorized_first_motion": True,
        "capture_mc_rl_debug": True,
        "extra_stop_observation": 5.0,
        "forward_speed": 0.14,
        "pulse_duration": 10.0,
        "ramp_duration_s": 0.0,
        "ramp_start_speed": None,
        "startup_speed": None,
        "startup_transition_displacement": None,
        "startup_transition_timeout_s": 0.0,
        "turning_reassessment_json": "/reviews/run65/result.json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PointGoalPrerequisiteTests(unittest.TestCase):
    def test_cli_point_goal_flag_uses_the_unified_execution_field(self) -> None:
        parser = argparse.ArgumentParser()
        runner.add_run65_point_goal_arguments(parser)
        disabled = parser.parse_args([])
        enabled = parser.parse_args(["--run65-ground-truth-point-goal"])
        self.assertTrue(hasattr(enabled, "run65_ground_truth_point_goal"))
        self.assertFalse(hasattr(enabled, "run65_point_goal_diagnostic"))
        self.assertFalse(runner.run65_point_goal_enabled(disabled))
        self.assertTrue(runner.run65_point_goal_enabled(enabled))
        self.assertFalse(runner.run65_point_goal_enabled(
            argparse.Namespace(run65_point_goal_diagnostic=True)
        ))

    def test_reassessment_requires_both_turns_and_all_independent_gates(self) -> None:
        evidence = valid_reassessment()
        verified = runner.validate_turning_reassessment_evidence(
            evidence, evidence_path="/reviews/run65/result.json", sha256="a" * 64,
        )
        self.assertEqual(verified["status"], "PASS")
        self.assertEqual(verified["reassessed_directions"], ["left", "right"])
        self.assertEqual(verified["source_faithful_original_config"], "NOT_TESTED")

        evidence["acceptance"]["turns"][1]["gates"]["physical_stop_gate"] = False
        with self.assertRaisesRegex(ValueError, "failed gate for right"):
            runner.validate_turning_reassessment_evidence(
                evidence, evidence_path="/reviews/run65/result.json", sha256="a" * 64,
            )

    def test_reassessment_rejects_source_faithful_relabel(self) -> None:
        evidence = valid_reassessment()
        evidence["acceptance"]["source_faithful_original_config"] = "PASS"
        with self.assertRaisesRegex(ValueError, "source-faithful boundary"):
            runner.validate_turning_reassessment_evidence(
                evidence, evidence_path="/reviews/run65/result.json", sha256="a" * 64,
            )

    def test_file_preflight_requires_governed_reservation_and_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="robotsim-run65-preflight-") as temp:
            run_dir = Path(temp)
            result_path = run_dir / "result.json"
            report_path = run_dir / "REPORT.md"
            manifest_path = run_dir / "artifact_paths.json"
            canonical_dir = "/home/lzy18001500226/projects/RobotSim/reviews/issue-80/test"
            canonical_result = f"{canonical_dir}/result.json"
            reservation_path = run_dir / ".robotsim-output-reservation.json"
            reservation_path.write_text(json.dumps({
                "schema_version": "robotsim.output-reservation.v1",
                "path": canonical_dir,
            }), encoding="utf-8")
            report_path.write_text("reviewed Run65 reassessment\n", encoding="utf-8")
            result_path.write_text(json.dumps(valid_reassessment()), encoding="utf-8")
            manifest = {
                "schema_version": "robotsim.local-artifacts.v1",
                "artifacts": [
                    {"WSL_PATH": f"{canonical_dir}/REPORT.md"},
                    {"WSL_PATH": canonical_result},
                    {"WSL_PATH": f"{canonical_dir}/artifact_paths.json"},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            path_governance = ModuleType("path_governance")

            def fake_validate(path, *, require_exists, expected_kind):
                native = Path(path)
                canonical = canonical_result if expected_kind == "file" else canonical_dir
                return SimpleNamespace(native_path=native, wsl_path=canonical)

            path_governance.validate_output_path = fake_validate
            with patch.dict(sys.modules, {"path_governance": path_governance}):
                verified = runner.validate_turning_reassessment_file(result_path)
                self.assertEqual(verified["status"], "PASS")
                manifest["artifacts"] = [
                    item for item in manifest["artifacts"] if item["WSL_PATH"] != canonical_result
                ]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest omits"):
                    runner.validate_turning_reassessment_file(result_path)

    def test_profile_accepts_only_fixed_bounded_point_goal_contract(self) -> None:
        expected = {"status": "PASS", "path": "/reviews/run65/result.json"}
        with patch.object(runner, "validate_turning_reassessment_file", return_value=expected) as verify:
            self.assertEqual(runner.validate_run65_point_goal_profile(valid_profile()), expected)
            verify.assert_called_once_with("/reviews/run65/result.json")

        for override in (
            {"maintainer_authorized_first_motion": False},
            {"forward_speed": 0.13},
            {"forward_speed": 0.12},
            {"pulse_duration": 10.01},
            {"extra_stop_observation": 4.9},
            {"ramp_duration_s": 0.1},
            {"capture_mc_rl_debug": False},
            {"turning_reassessment_json": None},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                runner.validate_run65_point_goal_profile(valid_profile(**override))


class PointGoalAcceptanceTests(unittest.TestCase):
    def test_goal_uses_measured_heading_without_a_fixed_stop_waypoint(self) -> None:
        start = {
            "x": 1.2,
            "y": -0.4,
            "yaw": 1.5707963267948966,
            "frame_id": "odom",
            "child_frame_id": "base_link",
        }
        goal = build_forward_point_goal(start, distance_m=0.30)
        self.assertAlmostEqual(goal["x"], 1.2, places=12)
        self.assertAlmostEqual(goal["y"], -0.1, places=12)
        with self.assertRaisesRegex(ValueError, "odom -> base_link"):
            build_forward_point_goal({**start, "frame_id": "map"}, distance_m=0.30)

    def test_stop_advance_scales_with_measured_speed_and_respects_existing_envelope(self) -> None:
        response_intercept = 0.00347654
        response_time = 0.09162162
        measured_run69_speed = 0.15815622063915188
        advance = speed_scaled_point_goal_stop_advance_m(
            measured_run69_speed,
            endpoint_response_intercept_m=response_intercept,
            endpoint_response_time_s=response_time,
            maximum_forward_speed_m_s=0.2,
        )
        self.assertAlmostEqual(advance, 0.017967, places=5)
        self.assertAlmostEqual(
            speed_scaled_point_goal_stop_advance_m(
                0.007654207147844275,
                endpoint_response_intercept_m=response_intercept,
                endpoint_response_time_s=response_time,
            ),
            0.004178,
            places=5,
        )
        self.assertLessEqual(
            speed_scaled_point_goal_stop_advance_m(
                0.2,
                endpoint_response_intercept_m=response_intercept,
                endpoint_response_time_s=response_time,
            ),
            0.05,
        )
        self.assertEqual(
            speed_scaled_point_goal_stop_advance_m(
                -0.1,
                endpoint_response_intercept_m=response_intercept,
                endpoint_response_time_s=response_time,
            ),
            response_intercept,
        )
        with self.assertRaisesRegex(ValueError, "existing point-goal envelope"):
            speed_scaled_point_goal_stop_advance_m(
                0.2001,
                endpoint_response_intercept_m=response_intercept,
                endpoint_response_time_s=response_time,
            )

    def test_measured_speed_projection_respects_declared_twist_frame(self) -> None:
        body_frame = {
            "vx": 0.16, "vy": 0.0, "yaw": 0.0, "twist_frame_id": "base_link",
        }
        self.assertAlmostEqual(
            point_goal_forward_speed_m_s(body_frame, start_yaw_rad=0.0), 0.16,
        )
        rotated_body = {
            "vx": 0.16, "vy": 0.0, "yaw": math.pi / 2, "twist_frame_id": "base_link",
        }
        self.assertAlmostEqual(
            point_goal_forward_speed_m_s(rotated_body, start_yaw_rad=0.0), 0.0, places=12,
        )
        world_frame = {
            "vx": 0.0, "vy": 0.16, "yaw": math.pi / 2, "twist_frame_id": "odom",
        }
        self.assertAlmostEqual(
            point_goal_forward_speed_m_s(world_frame, start_yaw_rad=math.pi / 2), 0.16,
        )

    def test_point_goal_heading_holds_start_orientation_and_uses_bounded_lateral_correction(self) -> None:
        self.assertAlmostEqual(
            point_goal_heading_error_rad(
                current_yaw_rad=0.60,
                start_yaw_rad=0.50,
                lateral_error_m=0.0,
            ),
            -0.10,
        )
        self.assertAlmostEqual(
            point_goal_heading_error_rad(
                current_yaw_rad=0.50,
                start_yaw_rad=0.50,
                lateral_error_m=0.02,
            ),
            -0.01,
        )
        self.assertAlmostEqual(
            point_goal_heading_error_rad(
                current_yaw_rad=0.50,
                start_yaw_rad=0.50,
                lateral_error_m=-1.0,
            ),
            0.04,
        )

    def test_goal_controller_stops_on_dynamic_crossing_and_bounded_guards(self) -> None:
        common = {
            "goal_distance_m": 0.30,
            "dynamic_stop_progress_m": 0.318,
            "maximum_lateral_error_m": 0.05,
            "maximum_overshoot_m": 0.05,
        }
        self.assertEqual(point_goal_stop_reason(
            along_track_progress_m=0.317,
            lateral_error_m=0.01,
            final_tolerance_m=0.05,
            **common,
        ), "goal_approach_predicted_settle_within_tolerance")
        self.assertEqual(point_goal_stop_reason(
            along_track_progress_m=0.319,
            lateral_error_m=0.01,
            final_tolerance_m=0.05,
            **common,
        ), "speed_scaled_stop_line_crossed")
        self.assertEqual(point_goal_stop_reason(
            along_track_progress_m=0.263,
            lateral_error_m=0.0069,
            dynamic_stop_progress_m=0.30418,
            final_tolerance_m=0.05,
            goal_distance_m=0.30,
            maximum_lateral_error_m=0.05,
            maximum_overshoot_m=0.05,
        ), "goal_approach_predicted_settle_within_tolerance")
        self.assertEqual(point_goal_stop_reason(
            along_track_progress_m=0.20,
            lateral_error_m=0.051,
            final_tolerance_m=0.05,
            **common,
        ), "lateral_error_limit")
        self.assertEqual(point_goal_stop_reason(
            along_track_progress_m=0.351,
            lateral_error_m=0.01,
            final_tolerance_m=0.05,
            **common,
        ), "hard_overshoot_guard")
        self.assertIsNone(point_goal_stop_reason(
            along_track_progress_m=0.24,
            lateral_error_m=0.01,
            final_tolerance_m=0.05,
            **common,
        ))

    def test_readback_refresh_does_not_change_the_existing_freshness_threshold(self) -> None:
        self.assertEqual(
            stale_readback_keys(
                {"state_age_s": {"get_mode": 0.499, "get_source": 0.5}}, 0.5,
            ),
            (),
        )
        self.assertEqual(
            stale_readback_keys(
                {"state_age_s": {"get_mode": 1.178, "get_source": 0.1}}, 0.5,
            ),
            ("get_mode",),
        )
        self.assertEqual(stale_readback_keys({}, 0.5), ("get_mode", "get_source"))

    def passing_result(self) -> dict:
        return {
            "status": "PASS",
            "goal_triggered": True,
            "target_crossing_detected": True,
            "final_distance_to_goal_m": 0.04,
            "acceptance_gates": {
                "goal_approach_or_stop_line_triggered": True,
                "final_distance_within_50mm": True,
                "lateral_error_within_50mm": True,
                "no_more_than_50mm_along_track_overshoot": True,
                "termination_condition_triggered_stop": True,
                "measured_physical_motion": True,
                "same_sample_source_locomotion_handshake": True,
                "independent_physical_stop_gate": True,
                "zero_stop_stream_minimum": True,
            },
            "activation": {
                "termination_condition_met": True,
                "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
                "control_handshake_status": "CONTROL_HANDSHAKE_CONFIRMED",
                "physical_stop_gate": {"status": "PASS"},
                "zero_stop": {"zero_publish_calls_returned": 50},
            },
        }

    def test_goal_pass_requires_goal_motion_handshake_and_physical_stop(self) -> None:
        self.assertEqual(
            runner.classify_point_goal_result(self.passing_result()),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_PASS",
        )
        failed_gate = self.passing_result()
        failed_gate["acceptance_gates"]["independent_physical_stop_gate"] = False
        self.assertEqual(
            runner.classify_point_goal_result(failed_gate),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
        )
        failed_gate = self.passing_result()
        failed_gate["activation"]["physical_motion_status"] = "NO_PHYSICAL_MOTION"
        self.assertEqual(
            runner.classify_point_goal_result(failed_gate),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
        )
        failed_gate = self.passing_result()
        failed_gate["activation"]["control_handshake_status"] = "NOT_CONFIRMED"
        self.assertEqual(
            runner.classify_point_goal_result(failed_gate),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
        )
        failed_gate = self.passing_result()
        failed_gate["activation"]["zero_stop"]["zero_publish_calls_returned"] = 49
        self.assertEqual(
            runner.classify_point_goal_result(failed_gate),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_BLOCKED",
        )

    def test_safety_fault_is_not_reported_as_goal_pass(self) -> None:
        result = self.passing_result()
        result["controller_fault"] = "fall_root_height"
        self.assertEqual(
            runner.classify_point_goal_result(result),
            "SIMULATION_ONLY_GROUND_TRUTH_POINT_GOAL_FAIL",
        )


if __name__ == "__main__":
    unittest.main()
