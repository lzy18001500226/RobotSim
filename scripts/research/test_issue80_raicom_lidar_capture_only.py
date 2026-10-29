from __future__ import annotations

import ast
import unittest
from pathlib import Path

from issue80_raicom_lidar_capture_only import (
    ALLOWED_RUNTIME_METHODS,
    MAX_CAPTURE_S,
    audit_capture_only_source,
    snapshot_runtime_immutable_inputs,
)


class CaptureOnlyRunnerTests(unittest.TestCase):
    def test_source_audit_forbids_control_and_reset(self) -> None:
        path = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py")
        result = audit_capture_only_source(path)
        self.assertEqual(result["status"], "PASS_CAPTURE_ONLY_STATIC_CAPABILITY_AUDIT")
        self.assertEqual(result["set_mc_action_requests"], 0)
        self.assertEqual(result["native_reset_dispatches"], 0)
        self.assertTrue(result["mc_process_started"])
        self.assertTrue(set(result["runtime_methods"]).issubset(ALLOWED_RUNTIME_METHODS))

    def test_runner_source_does_not_invoke_mj_step_or_action_symbols(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("dispatch_native_reset_button", calls)
        self.assertIn("start_mc_after_sim_topics", calls)
        self.assertIn("docker_exec", calls)

    def test_runner_requests_bounded_full_point_capture(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py").read_text(encoding="utf-8")
        self.assertIn("--expected-frame-id map --save-all-finite-points", source)

    def test_runner_reuses_the_verified_readonly_mjdata_overlay(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py").read_text(encoding="utf-8")
        self.assertIn("native_mjdata_capture=True", source)
        self.assertIn("runtime.prepare_native_mjdata_capture()", source)
        self.assertIn("runtime.finalize_native_mjdata_capture()", source)

    def test_known_good_startup_profile_stages_only_inputs_before_capture(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py").read_text(encoding="utf-8")
        self.assertIn("--known-good-startup-profile", source)
        self.assertIn("validate_local01_table_approach_mission_input", source)
        self.assertIn("runtime.prepare_foundationwalk_diagnostic_overlay()", source)
        self.assertIn("runtime.start_sim_first(start_mc=False)", source)

    def test_runner_uses_the_existing_narrow_mc_writeback_contract(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_capture_only.py").read_text(encoding="utf-8")
        self.assertIn("_stage_mc_config_dump_writeback", source)
        self.assertIn("MC_SIM_CONFIG_DUMP_CONTAINER_PATH", source)
        self.assertIn("MC_SIM_GENERATED_DIRECTORY", source)
        self.assertIn("mc_sim_generated_tmpfs=True", source)

    def test_capture_duration_is_bounded(self) -> None:
        self.assertLessEqual(MAX_CAPTURE_S, 12.0)

    def test_runtime_write_comparison_hashes_inputs_before_any_launch(self) -> None:
        snapshot = snapshot_runtime_immutable_inputs()
        self.assertGreater(snapshot["immutable_file_count"], 0)
        self.assertIn("sim_mujoco/bin/aima-sim-app", snapshot["immutable_file_sha256"])
        self.assertIn("mc/bin/aima-mc-app-main", snapshot["immutable_file_sha256"])
        self.assertFalse(any(path.endswith("/sim/classic") for path in snapshot["immutable_file_sha256"]))


if __name__ == "__main__":
    unittest.main()
