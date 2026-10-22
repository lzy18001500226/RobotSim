#!/usr/bin/env python3
"""Offline regression tests for the Issue 80 guarded ground-truth point-goal path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import issue80_m1_auto_reset_run as runner


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
        "forward_speed": 0.12,
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
            {"pulse_duration": 10.01},
            {"extra_stop_observation": 4.9},
            {"ramp_duration_s": 0.1},
            {"capture_mc_rl_debug": False},
            {"turning_reassessment_json": None},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                runner.validate_run65_point_goal_profile(valid_profile(**override))


class PointGoalAcceptanceTests(unittest.TestCase):
    def passing_result(self) -> dict:
        return {
            "status": "PASS",
            "goal_triggered": True,
            "final_distance_to_goal_m": 0.04,
            "acceptance_gates": {
                "goal_triggered_inside_stop_radius": True,
                "final_distance_within_50mm": True,
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
