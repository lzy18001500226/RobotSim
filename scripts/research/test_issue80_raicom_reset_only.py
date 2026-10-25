#!/usr/bin/env python3
"""Offline safety tests for the one-shot RAICOM Reset-only diagnostic."""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from typing import Any, Mapping

from issue80_raicom_reset_only import (
    ACTIVE_CONTAINER_WATCHDOG_S,
    LOCAL01_POSES_DEFAULT,
    MAX_RUNTIME_S,
    RESET_OBSERVER,
    ResetOnlyError,
    ResetOnlyRun,
    audit_reset_only_source,
    build_reset_only_runtime_args,
    classify_odom_frame,
    parse_attempt_id,
    required_state_ages_pass,
    _static_preflight,
    validate_policy,
)
from issue80_m1_auto_reset_run import build_runtime_bind_mount


RUNNER = Path(__file__).with_name("issue80_raicom_reset_only.py")


class FakeRuntime:
    def __init__(self):
        self.calls = 0

    def dispatch_native_reset_button(self, pid: int, title: str, window_id: int) -> None:
        self.calls += 1


class ResetOnlySafetyTests(unittest.TestCase):
    @staticmethod
    def _reset_effect_classifier():
        tree = ast.parse(RESET_OBSERVER.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "classify_reset_effect"
        )
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {
            "math": math,
            "Any": Any,
            "Mapping": Mapping,
            "RESET_EFFECT_MIN_POSITION_DELTA_M": 0.001,
            "RESET_EFFECT_MIN_YAW_DELTA_RAD": 0.005,
        }
        exec(compile(ast.fix_missing_locations(module), str(RESET_OBSERVER), "exec"), namespace)
        return namespace["classify_reset_effect"]

    def test_static_capability_audit_has_one_reset_and_no_motion_path(self):
        result = audit_reset_only_source(RUNNER, RESET_OBSERVER)
        self.assertEqual(result["status"], "PASS_RESET_ONLY_STATIC_CAPABILITY_AUDIT")
        self.assertEqual(result["reset_dispatch_call_sites"], 1)
        self.assertEqual(result["observer_publishers"], 0)
        self.assertEqual(result["observer_service_clients"], 0)
        self.assertFalse(result["route_executor_imported"])
        self.assertEqual(result["runtime_wall_cap_s"], 30.0)
        self.assertEqual(result["native_reset_clicks_max"], 1)
        observer_source = RESET_OBSERVER.read_text(encoding="utf-8")
        self.assertIn("self.subscription_handles = [", observer_source)
        self.assertIn("reset_only_observer_startup_error.json", observer_source)
        self.assertNotIn("self.subscriptions = [", observer_source)
        self.assertIn("enable_rosout=False", observer_source)
        self.assertIn("start_parameter_services=False", observer_source)
        self.assertIn("self.destroy_publisher(parameter_events)", observer_source)
        self.assertIn("--construction-smoke", observer_source)
        self.assertIn("--inject-construction-failure-for-test", observer_source)
        self.assertIn("subscriptions = list(self.subscriptions)", observer_source)
        self.assertIn("publishers = list(self.publishers)", observer_source)
        self.assertIn("clients = list(self.clients)", observer_source)
        runner_source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("Odom initialized:", runner_source)
        self.assertIn("name=pelvis", runner_source)

    def test_reset_dispatch_guard_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(attempt_id=str(uuid.uuid4()))
            runner = ResetOnlyRun(args, Path(directory), "robotsim-issue80-resetonly-test")
            fake = FakeRuntime()
            runner.runtime = fake
            runner.dispatch_reset_once(19, "MuJoCo test", 1234)
            self.assertEqual(fake.calls, 1)
            self.assertEqual(runner.reset_dispatch_count, 1)
            with self.assertRaisesRegex(ResetOnlyError, "second native Reset"):
                runner.dispatch_reset_once(19, "MuJoCo test", 1234)
            self.assertEqual(fake.calls, 1)

    def test_runtime_arguments_carry_the_verified_unique_container_name(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_reset_only_runtime_args(Path(directory), "robotsim-issue80-resetonly-test")
            self.assertEqual(args.container, "robotsim-issue80-resetonly-test")
            self.assertEqual(args.max_runtime_s, 30.0)
            self.assertIsNone(args.route_config)
            self.assertIsNone(args.local01_scene_world_poses)
            self.assertTrue(args.runtime_readonly)

    def test_observer_startup_error_aborts_readiness_before_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(attempt_id=str(uuid.uuid4()))
            runner = ResetOnlyRun(args, root, "robotsim-issue80-resetonly-test")
            runner.runtime = argparse.Namespace(deadline=time.monotonic() + 5.0)
            error = root / "observer_error.json"
            error.write_text(json.dumps({"status": "BLOCKED_OBSERVER_STARTUP", "error": "test"}), encoding="utf-8")
            with self.assertRaisesRegex(ResetOnlyError, "observer startup failed"):
                runner._wait_for_file(
                    root / "ready.json", timeout_s=4.0,
                    label="observer startup", fail_if=error,
                )
            self.assertEqual(runner.reset_dispatch_count, 0)

    def test_policy_requires_one_reset_and_disallows_motion_or_routes(self):
        validate_policy(max_runtime_s=30, reset_dispatches=1)
        for kwargs in (
            {"max_runtime_s": 30.001, "reset_dispatches": 1},
            {"max_runtime_s": 30, "reset_dispatches": 0},
            {"max_runtime_s": 30, "reset_dispatches": 2},
            {"max_runtime_s": 30, "reset_dispatches": 1, "route_requested": True},
            {"max_runtime_s": 30, "reset_dispatches": 1, "motion_requested": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_policy(**kwargs)

    def test_telemetry_readiness_requires_exact_fresh_topic_set(self):
        good = {key: 0.1 for key in ("odom", "imu", "common", "leg_state", "leg_command")}
        self.assertTrue(required_state_ages_pass(good))
        self.assertFalse(required_state_ages_pass({**good, "imu": 0.501}))
        self.assertFalse(required_state_ages_pass({key: value for key, value in good.items() if key != "imu"}))
        self.assertFalse(required_state_ages_pass({**good, "extra": 0.0}))

    def test_reset_effect_requires_a_generation_or_pose_discontinuity(self):
        classify = self._reset_effect_classifier()
        pre = {
            "latest_source_stamp_ns": {"odom": 1000},
            "odom_pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0},
        }
        unchanged = [{
            "source_stamp_ns": 1100,
            "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0},
        }]
        self.assertEqual(classify(pre, unchanged)["status"], "RESET_EFFECT_UNCONFIRMED")
        identical_stamp = [{
            "source_stamp_ns": 1000,
            "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0},
        }]
        identical = classify(pre, identical_stamp)
        self.assertEqual(identical["status"], "RESET_EFFECT_UNCONFIRMED")
        self.assertTrue(identical["source_stamp_unchanged_duplicate"])
        self.assertFalse(identical["source_stamp_rebased"])
        rebased = [{
            "source_stamp_ns": 10,
            "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0},
        }]
        rebased_result = classify(pre, rebased)
        self.assertEqual(rebased_result["status"], "RESET_EFFECT_OBSERVED")
        self.assertTrue(rebased_result["source_stamp_rebased"])
        displaced = [{
            "source_stamp_ns": 1100,
            "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.682, "yaw_rad": 0.0},
        }]
        self.assertEqual(classify(pre, displaced)["status"], "RESET_EFFECT_OBSERVED")

    def test_pre_reset_failure_classification_is_not_post_reset_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ResetOnlyRun(
                argparse.Namespace(attempt_id=str(uuid.uuid4())),
                Path(directory),
                "robotsim-issue80-resetonly-test",
            )
            self.assertEqual(runner._failure_classification(), "PRE_RESET_SETUP_BLOCKED")
            runner.reset_dispatch_count = 1
            self.assertEqual(runner._failure_classification(), "RESET_NOT_DISPATCHED")
            (Path(directory) / "reset_dispatch_event.json").write_text(
                json.dumps({"event": {"status": "RESET_BUTTON_CLICKED"}}),
                encoding="utf-8",
            )
            self.assertEqual(runner._failure_classification(), "POST_RESET_CAPTURE_BLOCKED")

    def test_canonical_uuid_is_required(self):
        value = str(uuid.uuid4())
        self.assertEqual(parse_attempt_id(value), value)
        for invalid in ("not-a-uuid", value.upper(), value.replace("-", "")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_attempt_id(invalid)

    def test_cli_rejects_first_motion_route_and_velocity_flags_before_runtime(self):
        base = [sys.executable, str(RUNNER), "--attempt-id", str(uuid.uuid4())]
        for extra in (
            ["--maintainer-authorized-first-motion"],
            ["--route-config", "/tmp/route.json"],
            ["--forward-speed", "0.2"],
            ["--maintainer-authorized-standing-takeover"],
        ):
            completed = subprocess.run(base + extra, text=True, capture_output=True, timeout=10)
            with self.subTest(extra=extra):
                self.assertEqual(completed.returncode, 2)
                self.assertIn("unrecognized arguments", completed.stderr)

    def test_frame_assessment_compares_candidates_without_authorizing_motion(self):
        candidates = {
            "scene_world": {"x_m": -1.5, "y_m": -1.5, "z_m": 0.68, "yaw_rad": 1.5707963267948966},
            "reset_local_odom": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0},
        }
        observed = {"x_m": 0.001, "y_m": -0.001, "z_m": 0.679, "yaw_rad": 0.005}
        result = classify_odom_frame(observed, candidates)
        self.assertEqual(result["classification"], "reset_local_odom")
        self.assertFalse(result["used_as_motion_gate"])
        world = classify_odom_frame(candidates["scene_world"], candidates)
        self.assertEqual(world["classification"], "scene_world")
        self.assertEqual(classify_odom_frame(None, candidates)["classification"], "UNRESOLVED_NO_POST_RESET_ODOM")

    def test_hard_runtime_bounds_are_capture_only(self):
        self.assertEqual(MAX_RUNTIME_S, 30.0)
        self.assertEqual(ACTIVE_CONTAINER_WATCHDOG_S, 26.0)
        self.assertLess(ACTIVE_CONTAINER_WATCHDOG_S, MAX_RUNTIME_S)

    def test_reset_only_mounts_vendor_runtime_read_only(self):
        mount = build_runtime_bind_mount(Path("/runtime"), readonly=True)
        self.assertEqual(mount, "type=bind,source=/runtime,target=/workspace/raicom2026,readonly")
        self.assertEqual(
            build_runtime_bind_mount(Path("/runtime"), readonly=False),
            "type=bind,source=/runtime,target=/workspace/raicom2026",
        )

    def test_static_preflight_stages_scene_and_existing_overlay_without_route_runtime_input(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(local01_scene_world_poses=LOCAL01_POSES_DEFAULT)
            prepared = _static_preflight(args, Path(directory))
            self.assertEqual(prepared["source"]["source_commit"], "95d76d82ea7def0dd719da4ebe07f9654044415e")
            self.assertEqual(prepared["static_scene"]["status"], "PASS_STATIC_SCENE_CLEARANCE")
            self.assertEqual(prepared["static_scene"]["native_timestep_s"], 0.001)
            self.assertEqual(prepared["static_scene"]["combined_scene_timestep_s"], 0.002)
            self.assertEqual(prepared["bundle"]["x2_attach_prefix_change"], {"source": "x2_", "runtime": ""})
            self.assertEqual(prepared["overlay_manifest"]["classification"], "SIMULATION_ONLY_EXPERIMENTAL_FOUNDATIONWALK_OVERLAY")
            self.assertTrue(prepared["overlay_manifest"]["safety_gates_modified"] is False)
            self.assertTrue(all(source_path.is_file() for source_path, _ in prepared["scene_mounts"]))
            staged = Path(directory)
            self.assertFalse((staged / "route_config.json").exists())
            self.assertFalse((staged / "route_config_input.json").exists())


if __name__ == "__main__":
    unittest.main()
