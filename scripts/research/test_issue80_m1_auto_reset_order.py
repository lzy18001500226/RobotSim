"""Offline ordering checks for Reset-before-SD first-motion preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import issue80_m1_auto_reset_run as reset_runner
from issue80_m1_auto_reset_run import (
    BoundedRun,
    RunDeadline,
    first_post_action_fall,
    standing_baseline_observation_passes,
)
from issue80_mujoco_x11_reset import (
    NATIVE_RESET_BUTTON_PIXEL_POINTS,
    native_reset_button_signature_matches,
)


class ResetBeforeSdOrderingTests(unittest.TestCase):
    def test_parent_forwards_heading_change_into_child_point_goal_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                attempt_id="heading-goal-test",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=10.0,
                forward_speed=0.14,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
                turning_reassessment_json="/reviews/run65/result.json",
                run65_ground_truth_point_goal=True,
                run53_turning_diagnostic=False,
                heading_change_rad=0.20,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            runner.deadline = time.monotonic() + 120.0
            commands = []

            def fake_run(_name, command, **_kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            runner.run = fake_run
            runner.run_m1()

        child = commands[0][-1]
        self.assertIn("--point-goal-diagnostic", child)
        self.assertIn("--heading-change-rad 0.2", child)

    def test_native_reset_layout_uses_button_pixel_signature_not_full_window_size(self) -> None:
        # MuJoCo 3.3.6 Run76 screenshot at 1866x840 shows the same fixed
        # Simulation/Reset menu control at the calibrated client coordinates.
        self.assertEqual(NATIVE_RESET_BUTTON_PIXEL_POINTS, {
            "top": (68, 444), "left": (13, 456), "right": (124, 456),
            "bottom": (68, 468), "center": (68, 456),
        })
        pixels = {
            "top": (153, 102, 102),
            "left": (153, 102, 102),
            "right": (153, 102, 102),
            "bottom": (153, 102, 102),
            "center": (26, 26, 26),
        }
        self.assertTrue(native_reset_button_signature_matches(pixels))
        self.assertFalse(native_reset_button_signature_matches({"center": (26, 26, 26)}))

        wrong_control = dict(pixels)
        wrong_control["right"] = (26, 26, 26)
        self.assertFalse(native_reset_button_signature_matches(wrong_control))

    def test_timed_out_review_screenshot_does_not_block_reset_safety_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                attempt_id="screenshot-timeout-test",
                max_runtime_s=30.0,
            ))

            def timeout(_name: str, _command: list[str], **_kwargs):
                raise RunDeadline("x11_capture_before_reset timed out")

            runner.run = timeout
            result = runner.capture_reset_image(19, "MuJoCo : test", 99, "before")

        self.assertEqual(result["status"], "NOT_CAPTURED")
        self.assertIn("x11_capture_before_reset timed out", result["reason"])
        self.assertFalse(result["input_sent"])

    def test_runner_dispatches_native_reset_button_without_keyboard_or_windows_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                attempt_id="native-reset-test",
                max_runtime_s=30.0,
            ))
            event = {
                "schema": "robotsim.issue80.reset-dispatch-event.v1",
                "window": {
                    "window_id": 99, "pid": 7, "title": "MuJoCo : test", "viewable": True,
                },
                "event": {
                    "status": "RESET_BUTTON_CLICKED",
                    "dispatch_monotonic_ns": 123,
                    "dispatch_wall_utc": "test",
                    "key_events_sent": 0,
                    "mouse_button_events_sent": 2,
                },
            }
            commands: list[list[str]] = []

            def fake_run(_name: str, command: list[str], **_kwargs):
                commands.append(command)
                return SimpleNamespace(stdout=json.dumps(event))

            runner.run = fake_run
            runner.activate_verified_windows_foreground = lambda **_kwargs: self.fail(
                "native Reset dispatch must not fall back to Windows foreground activation"
            )
            runner.dispatch_native_reset_button(7, "MuJoCo : test", 99)
            persisted = json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())

        self.assertEqual(len(commands), 1)
        self.assertIn("click-native-reset", commands[0])
        self.assertNotIn("send-reset", commands[0])
        self.assertEqual(persisted["event"]["status"], "RESET_BUTTON_CLICKED")
        self.assertEqual(persisted["reset_delivery_attempt_count"], 1)
        self.assertEqual(persisted["reset_input_path_attempt_count"], 1)

    def test_parent_and_prearmed_observer_share_bounded_late_reset_window(self) -> None:
        self.assertEqual(reset_runner.RESET_STARTUP_OBSERVATION_TIMEOUT_S, 4.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                attempt_id="observer-window-test",
                max_runtime_s=30.0,
            ))
            (Path(temp_dir) / "reset_watch_ready.json").write_text(json.dumps({
                "status": "READY",
                "fresh_topics": ["odom", "imu", "common", "leg_state", "leg_command"],
                "set_mc_action_service_ready": True,
                "get_mc_action_service_ready": True,
                "pre_reset_get_mc_action": {"action_desc": "PASSIVE_DEFAULT", "action_status_value": 100},
            }), encoding="utf-8")
            captured: list[str] = []
            runner.docker_exec = lambda _label, command, **_kwargs: captured.append(command)
            runner.start_observer()

        self.assertEqual(len(captured), 1)
        self.assertIn("--upright-timeout-s 4", captured[0])

    def make_runner(
        self,
        events: list[str],
        temp_dir: str,
        *,
        posture_ok: bool = True,
        standing_only: bool = False,
        m1_result: dict | None = None,
    ) -> BoundedRun:
        class RecordingRun(BoundedRun):
            def stage_research_scripts(self) -> None:
                events.append("stage")

            def start_container(self) -> None:
                events.append("container")

            def start_sim_first(self) -> None:
                events.append("sim_and_mc")

            def capture_service_topic_inventory(self) -> None:
                events.append("inventory")

            def inspect_window(self) -> tuple[int, str, int]:
                events.append("inspect_window")
                return 7, "MuJoCo : test", 99

            def capture_reset_image(self, pid: int, title: str, window_id: int, phase: str) -> dict[str, str]:
                events.append(f"capture_{phase}")
                return {"status": "CAPTURED"}

            def start_observer(self) -> None:
                events.append("observer_ready")

            def dispatch_reset(self, pid: int, title: str, window_id: int) -> None:
                events.append("reset_once")
                self.reset_dispatch_payload = {"event": {"status": "KEY_DISPATCHED", "dispatch_monotonic_ns": 1}}

            def dispatch_native_reset_button(self, pid: int, title: str, window_id: int) -> None:
                events.append("reset_once")
                self.reset_dispatch_payload = {"event": {"status": "RESET_BUTTON_CLICKED", "dispatch_monotonic_ns": 1}}

            def wait_for_immediate_reset_pose(self) -> dict[str, str]:
                events.append("fresh_post_reset_pose")
                if not posture_ok:
                    raise RuntimeError("no suitable posture")
                return {"status": "POST_RESET_UPRIGHT_OBSERVED"}

            def request_sd_and_verify(self) -> dict[str, str]:
                events.append("sd_once")
                self.sd_result = {"response_status_value": 0, "response_header_code": 0}
                return {"response_status_value": 0, "response_header_code": 0}

            def wait_and_classify_reset(self, pid: int, title: str, window_id: int) -> dict[str, str]:
                events.append("post_sd_standing_window")
                self.reset_analysis = {"classification": "AUTO_RESET_PASS"}
                self.reset_classification = "AUTO_RESET_PASS"
                return self.reset_analysis

            def run_m1(self) -> None:
                events.append("first_motion_after_standing_gate")
                self.m1_result = m1_result if m1_result is not None else {
                    "status": "PASS",
                    "source_registration": {"success": True},
                    "set_mc_action_ld": {"accepted_by_service": True, "response_header_code": 0},
                    "activation": {
                        "nonzero_publish_calls_returned": 1,
                        "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
                        "status": "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED",
                        "post_stop_forward_tail": {"status": "WITHIN_BASELINE_ENVELOPE"},
                    },
                }

            def cleanup(self) -> None:
                events.append("cleanup")

        return RecordingRun(argparse.Namespace(
            evidence_dir=temp_dir,
            container="test-container",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_standing_takeover=standing_only,
            attempt_id="test-attempt",
            max_runtime_s=30.0,
        ))

    def test_immediate_post_reset_gate_uses_fresh_pose_and_keeps_raw_fsm_body_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                maintainer_authorized_first_motion=True,
                attempt_id="test-attempt",
                max_runtime_s=30.0,
            ))
            base_ns = time.monotonic_ns()
            dispatch_ns = base_ns - 1_000_000_000
            runner.reset_dispatch_payload = {
                "event": {"status": "KEY_DISPATCHED", "dispatch_monotonic_ns": dispatch_ns}
            }
            rows = []
            for i in range(3):
                rows.append({
                    "event": "state", "topic_key": "odom",
                    "wall_monotonic_ns": base_ns - (30 - i * 10) * 1_000_000,
                    "wall_utc": f"t{i}",
                    "data": {"z": 0.60, "roll_deg": 1.0, "pitch_deg": 2.0},
                })
            for key in ("imu", "common", "leg_state", "leg_command"):
                rows.append({
                    "event": "state", "topic_key": key,
                    "wall_monotonic_ns": base_ns - 5_000_000,
                    "wall_utc": "t",
                    "data": {"fsm_state": 6, "body_status": 2} if key == "common" else {},
                })
            (Path(temp_dir) / "reset_state_trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            snapshot = runner.wait_for_immediate_reset_pose()

        self.assertEqual(snapshot.get("status"), "POST_RESET_UPRIGHT_OBSERVED", snapshot)
        self.assertEqual(snapshot["raw_common_state"], {"fsm_state": 6, "body_status": 2})
        self.assertGreaterEqual(snapshot["dispatch_to_upright_ms"], 0.0)

    def test_late_reset_event_is_observed_without_relaxing_the_pose_gate(self) -> None:
        self.assertGreaterEqual(reset_runner.RESET_STARTUP_OBSERVATION_TIMEOUT_S, 2.216)
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="test-container",
                maintainer_authorized_first_motion=True,
                attempt_id="late-reset-test",
                max_runtime_s=30.0,
            ))
            base_ns = 10_000_000_000
            dispatch_ns = base_ns
            runner.reset_dispatch_payload = {
                "event": {"status": "KEY_DISPATCHED", "dispatch_monotonic_ns": dispatch_ns}
            }
            trace = Path(temp_dir) / "reset_state_trace.jsonl"
            trace.write_text(json.dumps({
                "event": "state", "topic_key": "odom",
                "wall_monotonic_ns": dispatch_ns - 1,
                "wall_utc": "before-dispatch",
                "data": {"z": 0.076, "roll_deg": 0.0, "pitch_deg": -88.0},
            }) + "\n", encoding="utf-8")
            clock = {"seconds": 0.0, "written": False}

            def fake_sleep(seconds: float) -> None:
                clock["seconds"] += seconds
                if clock["seconds"] >= 2.216 and not clock["written"]:
                    rows = []
                    for i in range(3):
                        rows.append({
                            "event": "state", "topic_key": "odom",
                            "wall_monotonic_ns": base_ns + int((2.22 + i * 0.02) * 1e9),
                            "wall_utc": f"late-upright-{i}",
                            "data": {"z": 0.676, "roll_deg": 0.0, "pitch_deg": 0.0},
                        })
                    for key in ("imu", "common", "leg_state", "leg_command"):
                        rows.append({
                            "event": "state", "topic_key": key,
                            "wall_monotonic_ns": base_ns + int(2.26 * 1e9),
                            "wall_utc": f"late-{key}",
                            "data": {"fsm_state": 6, "body_status": 1} if key == "common" else {},
                        })
                    with trace.open("a", encoding="utf-8") as stream:
                        stream.writelines(json.dumps(row) + "\n" for row in rows)
                    clock["written"] = True

            with (
                patch.object(reset_runner.time, "monotonic", side_effect=lambda: clock["seconds"]),
                patch.object(reset_runner.time, "monotonic_ns", side_effect=lambda: base_ns + int(clock["seconds"] * 1e9)),
                patch.object(reset_runner.time, "sleep", side_effect=fake_sleep),
            ):
                result = runner.wait_for_immediate_reset_pose()

        self.assertEqual(result["status"], "POST_RESET_UPRIGHT_OBSERVED")
        self.assertGreaterEqual(result["dispatch_to_upright_ms"], 2220.0)
        self.assertEqual(result["physical_gate"], {
            "minimum_root_z_m": 0.52,
            "maximum_abs_roll_pitch_deg_exclusive": 15.0,
        })
        self.assertEqual(result["consecutive_upright_odom_required"], 3)

    def test_post_sd_fall_detector_ignores_pre_action_pose_and_stops_on_first_fall(self) -> None:
        rows = [
            {"event": "state", "topic_key": "odom", "wall_monotonic_ns": 90, "wall_utc": "pre",
             "data": {"z": 0.35, "roll_deg": 0.0, "pitch_deg": -88.0}},
            {"event": "state", "topic_key": "odom", "wall_monotonic_ns": 110, "wall_utc": "upright",
             "data": {"z": 0.68, "roll_deg": 0.1, "pitch_deg": 0.2}},
            {"event": "state", "topic_key": "odom", "wall_monotonic_ns": 120, "wall_utc": "fell",
             "data": {"z": 0.35, "roll_deg": -170.0, "pitch_deg": -88.0}},
        ]
        trace = "".join(json.dumps(row) + "\n" for row in rows)
        result = first_post_action_fall(trace, action_monotonic_ns=100)
        self.assertEqual(result["wall_utc"], "fell")
        self.assertEqual(result["wall_monotonic_ns"], 120)

    def test_reset_and_immediate_posture_precede_sd_and_standing_hold(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(events, temp_dir)
            result = runner.execute()

        self.assertEqual(result["outcome_classification"], "FIRST_PHYSICAL_WALKING_PASS")
        self.assertLess(events.index("sim_and_mc"), events.index("reset_once"))
        self.assertLess(events.index("observer_ready"), events.index("reset_once"))
        self.assertLess(events.index("reset_once"), events.index("fresh_post_reset_pose"))
        self.assertLess(events.index("fresh_post_reset_pose"), events.index("sd_once"))
        self.assertLess(events.index("sd_once"), events.index("post_sd_standing_window"))
        self.assertLess(events.index("post_sd_standing_window"), events.index("first_motion_after_standing_gate"))

    def test_parent_reports_activation_joint_limit_fault_as_walking_failure(self) -> None:
        events: list[str] = []
        m1_result = {
            "status": "BLOCKED",
            "source_registration": {"success": True},
            "set_mc_action_ld": {"accepted_by_service": True, "response_header_code": 0},
            "activation": {
                "nonzero_publish_calls_returned": 54,
                "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
                "status": "ABORTED",
                "fault": "source_joint_velocity_limit:left_knee_joint",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(events, temp_dir, m1_result=m1_result)
            result = runner.execute()

        self.assertEqual(result["outcome_classification"], "FIRST_PHYSICAL_WALKING_FAIL")
        self.assertEqual(result["fault"], "source_joint_velocity_limit:left_knee_joint")

    def test_unsuitable_post_reset_pose_stops_before_sd(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(events, temp_dir, posture_ok=False)
            result = runner.execute()

        self.assertEqual(result["outcome_classification"], "STANDING_CONTROL_NOT_READY")
        self.assertNotIn("sd_once", events)
        self.assertNotIn("first_motion_after_standing_gate", events)
        self.assertEqual(result["m1_status"], "NOT_RUN")

    def test_standing_takeover_only_does_not_reach_m1(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(events, temp_dir, standing_only=True)
            runner.args.maintainer_authorized_first_motion = False
            result = runner.execute()

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["outcome_classification"], "STANDING_TAKEOVER_PASS")
        self.assertEqual(result["m1_status"], "NOT_RUN_STANDING_ONLY")
        self.assertEqual(result["next_motion_stage"], "NOT_RUN_REQUIRES_SEPARATE_AUTHORIZATION")
        self.assertNotIn("first_motion_after_standing_gate", events)

    def test_complete_stand_baseline_requires_confirmed_readbacks_and_full_duration(self) -> None:
        readback = {
            "transport_state": "RESPONSE_RECEIVED",
            "action_desc": "STAND_DEFAULT",
            "action_status_value": 100,
        }
        observation = {
            "anchor": "STAND_DEFAULT_GETMC_READBACK",
            "observer_status": "STAND_DEFAULT_BASELINE_COMPLETE",
            "complete": True,
            "observed_duration_s": 8.0,
            "first_unsafe_monotonic_ns": None,
            "expected_body_status": 1,
            "common_mode_desc_values": ["STAND_DEFAULT"],
            "common_mode_status_values": [100],
            "raw_fsm_state_values": [4],
            "raw_body_status_values": [1],
            "common_state_sample_count": 160,
            "initial_stand_mode_readback": readback,
            "final_stand_mode_readback": readback,
        }
        self.assertTrue(standing_baseline_observation_passes(observation))
        observation["raw_fsm_state_values"] = [1, 4]
        self.assertTrue(standing_baseline_observation_passes(observation))
        observation["raw_fsm_state_values"] = [None, 4]
        self.assertFalse(standing_baseline_observation_passes(observation))
        observation["raw_fsm_state_values"] = [4]
        observation["observed_duration_s"] = 7.999
        self.assertFalse(standing_baseline_observation_passes(observation))
        observation["observed_duration_s"] = 8.0
        observation["anchor"] = "RESET_FALLBACK_NO_CONFIRMED_STAND"
        self.assertFalse(standing_baseline_observation_passes(observation))

    def test_required_standing_baseline_fails_closed_without_eight_confirmed_seconds(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(events, temp_dir, standing_only=True)
            runner.args.maintainer_authorized_first_motion = False
            runner.args.require_complete_stand_baseline = True
            def incomplete_reset(_pid, _title, _window_id):
                runner.reset_classification = "AUTO_RESET_PASS"
                runner.reset_analysis = {
                    "classification": "AUTO_RESET_PASS",
                    "standing_baseline_observation": {
                        "anchor": "STAND_DEFAULT_GETMC_READBACK",
                        "complete": False,
                        "observed_duration_s": 7.9,
                    },
                }
                return runner.reset_analysis
            runner.wait_and_classify_reset = incomplete_reset
            result = runner.execute()

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["outcome_classification"], "STAND_BASELINE_OBSERVATION_INCOMPLETE")
        self.assertEqual(result["m1_status"], "NOT_RUN_STANDING_BASELINE_INCOMPLETE")
        self.assertNotIn("first_motion_after_standing_gate", events)


if __name__ == "__main__":
    unittest.main()
