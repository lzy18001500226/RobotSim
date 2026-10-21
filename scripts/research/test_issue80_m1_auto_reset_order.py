"""Offline ordering checks for Reset-before-SD first-motion preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path

from issue80_m1_auto_reset_run import BoundedRun, first_post_action_fall


class ResetBeforeSdOrderingTests(unittest.TestCase):
    def make_runner(
        self,
        events: list[str],
        temp_dir: str,
        *,
        posture_ok: bool = True,
        standing_only: bool = False,
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
                self.m1_result = {
                    "status": "PASS",
                    "source_registration": {"success": True},
                    "set_mc_action_ld": {"accepted_by_service": True, "response_header_code": 0},
                    "activation": {
                        "nonzero_publish_calls_returned": 1,
                        "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
                        "post_stop_odometry": {"kinematic_status": "WITHIN_BASELINE_ENVELOPE"},
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


if __name__ == "__main__":
    unittest.main()
