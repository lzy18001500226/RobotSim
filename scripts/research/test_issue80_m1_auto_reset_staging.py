"""Offline tests for exact, no-overwrite research-code staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from issue80_m1_auto_reset_run import BoundedRun, M1_MODE_LOGIC


class ResearchScriptStagingTests(unittest.TestCase):
    def make_runner(self, evidence: Path) -> BoundedRun:
        return BoundedRun(
            argparse.Namespace(
                evidence_dir=str(evidence),
                container="test-container",
                source="test-source",
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

    def test_runner_keeps_sdk_action_identity_distinct_from_registered_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.set_action_source = "node.set_mc_action"
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(len(captured), 1)
            shell_command = captured[0][-1]
            self.assertIn("--source test-source", shell_command)
            self.assertIn("--set-action-source node.set_mc_action", shell_command)


if __name__ == "__main__":
    unittest.main()
