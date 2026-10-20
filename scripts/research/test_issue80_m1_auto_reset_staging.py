"""Offline tests for exact, no-overwrite research-code staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from issue80_m1_auto_reset_run import (
    BoundedRun, MC_LAUNCH, M1_FIRST_MOTION_LOGIC, M1_MODE_LOGIC, SIM_LAUNCH,
)


class ResearchScriptStagingTests(unittest.TestCase):
    def make_runner(self, evidence: Path) -> BoundedRun:
        return BoundedRun(
            argparse.Namespace(
                evidence_dir=str(evidence),
                container="test-container",
                maintainer_authorized_first_motion=True,
                attempt_id="test-attempt",
                max_runtime_s=10.0,
            )
        )

    def test_stages_current_mode_logic_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / M1_MODE_LOGIC.name
            self.assertEqual(staged.read_bytes(), M1_MODE_LOGIC.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == M1_MODE_LOGIC.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())
            self.assertEqual(record["disposition"], "COPIED")

    def test_stages_first_motion_safety_logic_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / M1_FIRST_MOTION_LOGIC.name
            self.assertEqual(staged.read_bytes(), M1_FIRST_MOTION_LOGIC.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == M1_FIRST_MOTION_LOGIC.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())

    def test_stages_verified_runtime_launchers_under_names_used_by_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            by_name = {row["staged_name"]: row for row in manifest["scripts"]}
            for staged_name, source in (("launch_sim.sh", SIM_LAUNCH), ("launch_mc.sh", MC_LAUNCH)):
                staged = evidence / staged_name
                self.assertEqual(staged.read_bytes(), source.read_bytes())
                self.assertEqual(by_name[staged_name]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_refuses_to_overwrite_a_stale_staged_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            stale = evidence / M1_MODE_LOGIC.name
            stale.write_bytes(b"stale module")
            runner = self.make_runner(evidence)
            with self.assertRaisesRegex(RuntimeError, "differs from current source"):
                runner.stage_research_scripts()
            self.assertEqual(stale.read_bytes(), b"stale module")
            self.assertFalse((evidence / "staged_research_scripts.json").exists())

    def test_runner_does_not_start_first_motion_without_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.maintainer_authorized_first_motion = False
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(captured, [])
            self.assertEqual(runner.m1_result["status"], "NOT_AUTHORIZED")

    def test_parent_does_not_start_container_without_first_motion_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.maintainer_authorized_first_motion = False
            runner.start_container = lambda: self.fail("unauthorized run must not start Docker")
            result = runner.execute()
            self.assertEqual(result["status"], "NOT_AUTHORIZED")
            self.assertFalse(runner.container_created)

    def test_runtime_container_must_be_bridge_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            inspect_payload = [{
                "HostConfig": {"NetworkMode": "host"},
                "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
            }]

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                if name == "container_inspect_start":
                    return subprocess.CompletedProcess(command, 0, json.dumps(inspect_payload), "")
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            with self.assertRaisesRegex(RuntimeError, "bridge-isolated"):
                runner.start_container()
            self.assertEqual(runner.network_mode, "host")

    def test_container_inspect_rejects_host_network_for_first_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            calls: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                payload = [{
                    "HostConfig": {"NetworkMode": "host"},
                    "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
                }]
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

            runner.run = fake_run
            with self.assertRaisesRegex(RuntimeError, "bridge-isolated"):
                runner.start_container()
            self.assertTrue(runner.container_created)
            self.assertIn("--network=bridge", calls[0])

    def test_authorized_runner_uses_one_pinned_source_and_hard_limited_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.maintainer_authorized_first_motion = True
            runner.network_mode = "bridge"
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()
            shell_command = captured[0][-1]
            self.assertIn("first-motion --maintainer-authorized", shell_command)
            self.assertIn("--reset-gate-verified", shell_command)
            self.assertIn("--attempt-id test-attempt", shell_command)
            self.assertIn("ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", captured[0])
            self.assertIn("--robot-model-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/robot_model.yaml", shell_command)
            self.assertNotIn("--source ", shell_command)
            self.assertNotIn("--activation-pulse", shell_command)
            self.assertNotIn("--zero-source-arbitration", shell_command)


if __name__ == "__main__":
    unittest.main()
