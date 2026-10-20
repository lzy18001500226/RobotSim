"""Offline ordering test for bounded Reset telemetry and screenshot capture."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from issue80_m1_auto_reset_run import BoundedRun


class ResetObservationOrderingTests(unittest.TestCase):
    def test_window_discovery_and_before_image_precede_observer_and_key(self) -> None:
        events: list[str] = []

        class RecordingRun(BoundedRun):
            def stage_research_scripts(self) -> None:
                events.append("stage")

            def start_container(self) -> None:
                events.append("container")

            def start_sim_first(self) -> None:
                events.append("sim_and_mc")

            def capture_service_topic_inventory(self) -> None:
                events.append("inventory")

            def request_sd_and_verify(self) -> str:
                events.append("sd_confirmed")
                return "Mode: STAND_DEFAULT"

            def inspect_window(self) -> tuple[int, str, int]:
                events.append("inspect_window")
                return 7, "MuJoCo : test", 99

            def capture_reset_image(self, pid: int, title: str, window_id: int, phase: str) -> dict[str, str]:
                events.append(f"capture_{phase}")
                return {"status": "CAPTURED"}

            def start_observer(self) -> None:
                events.append("observer_primed")

            def dispatch_reset(self, pid: int, title: str, window_id: int) -> None:
                events.append("reset_key")

            def wait_and_classify_reset(self, pid: int, title: str, window_id: int) -> dict[str, str]:
                events.append("complete_reset_state_window")
                self.capture_reset_image(pid, title, window_id, "after")
                self.reset_analysis = {"classification": "AUTO_RESET_PASS"}
                return self.reset_analysis

            def run_m1(self) -> None:
                events.append("m1_after_reset_gate")
                self.m1_result = {"status": "PASS"}

            def cleanup(self) -> None:
                events.append("cleanup")

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = RecordingRun(
                argparse.Namespace(
                    evidence_dir=temp_dir,
                    container="test-container",
                    source="issue80_m1_test_12345678",
                    set_action_source="node.set_mc_action",
                    attempt_id="test-attempt",
                    max_runtime_s=10.0,
                )
            )
            result = runner.execute()

        self.assertEqual(result["status"], "PASS")
        self.assertLess(events.index("inspect_window"), events.index("capture_before"))
        self.assertLess(events.index("capture_before"), events.index("observer_primed"))
        self.assertLess(events.index("observer_primed"), events.index("reset_key"))
        self.assertLess(events.index("reset_key"), events.index("complete_reset_state_window"))
        self.assertLess(events.index("complete_reset_state_window"), events.index("capture_after"))
        self.assertLess(events.index("capture_after"), events.index("m1_after_reset_gate"))


if __name__ == "__main__":
    unittest.main()
