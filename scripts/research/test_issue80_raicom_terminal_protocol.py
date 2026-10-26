"""Deterministic process tests for the RAICOM observer closeout protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from issue80_raicom_observer_wrapper import run_child
from issue80_raicom_reset_only import ResetOnlyError, ResetOnlyRun
from issue80_raicom_terminal_protocol import (
    BASE_SUMMARY_FILENAME,
    PROCESS_EXIT_FILENAME,
    TERMINAL_MANIFEST_FILENAME,
    TerminalProtocolError,
    atomic_write_new_json,
    publish_process_exit,
    validate_process_exit,
    validate_terminal_manifest,
)

STANDING_SUMMARY = "stand_takeover_summary.json"


def fake_observer_code(mode: str, delay_s: float = 0.2) -> str:
    return f"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from issue80_raicom_terminal_protocol import atomic_write_new_json, publish_terminal_manifest

out = Path(sys.argv[1])
mode = {mode!r}
delay_s = {delay_s!r}
base = out / {BASE_SUMMARY_FILENAME!r}
stand = out / {STANDING_SUMMARY!r}
trace = out / "reset_only_state_trace.jsonl"
atomic_write_new_json(base, {{"observer_status": "CAPTURE_COMPLETE", "status": "PASS"}})
trace.write_text('{{"event":"state_message"}}\\n', encoding="utf-8")
if mode == "base_then_delay":
    time.sleep(delay_s)
    atomic_write_new_json(stand, {{"status": "PASS"}})
    publish_terminal_manifest(
        out,
        observer_status="CAPTURE_COMPLETE",
        failure=None,
        summary_filenames=[base.name, stand.name],
        trace_filename=trace.name,
        finalization_monotonic_ns=time.monotonic_ns(),
    )
elif mode == "success":
    atomic_write_new_json(stand, {{"status": "PASS"}})
    publish_terminal_manifest(
        out,
        observer_status="CAPTURE_COMPLETE",
        failure=None,
        summary_filenames=[base.name, stand.name],
        trace_filename=trace.name,
        finalization_monotonic_ns=time.monotonic_ns(),
    )
elif mode == "truncated_trace":
    atomic_write_new_json(stand, {{"status": "PASS"}})
    trace.write_bytes(b'{{"event":"state_message"}}')
    summaries = {{
        base.name: {{"sha256": hashlib.sha256(base.read_bytes()).hexdigest()}},
        stand.name: {{"sha256": hashlib.sha256(stand.read_bytes()).hexdigest()}},
    }}
    atomic_write_new_json(out / "observer_terminal.json", {{
        "schema": "robotsim.issue80.observer-terminal.v1",
        "terminal_state": "FINALIZED",
        "all_summaries_durable_before_terminal": True,
        "trace_closed_before_terminal": True,
        "summaries": summaries,
        "trace": {{
            "filename": trace.name,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "records": 1,
        }},
    }})
elif mode == "incomplete_standing":
    stand.write_text('{{"status":', encoding="utf-8")
    summaries = {{
        base.name: {{"sha256": hashlib.sha256(base.read_bytes()).hexdigest()}},
        stand.name: {{"sha256": hashlib.sha256(stand.read_bytes()).hexdigest()}},
    }}
    atomic_write_new_json(out / "observer_terminal.json", {{
        "schema": "robotsim.issue80.observer-terminal.v1",
        "terminal_state": "FINALIZED",
        "all_summaries_durable_before_terminal": True,
        "trace_closed_before_terminal": True,
        "summaries": summaries,
        "trace": {{
            "filename": trace.name,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "records": 1,
        }},
    }})
elif mode == "exception":
    try:
        raise RuntimeError("controlled observer failure")
    except RuntimeError:
        sys.exit(7)
elif mode == "timeout":
    time.sleep(delay_s)
else:
    raise RuntimeError("unknown fake mode")
"""


class TerminalProtocolTests(unittest.TestCase):
    def child_command(self, directory: Path, mode: str, delay_s: float = 0.2) -> list[str]:
        return [
            sys.executable,
            "-c",
            fake_observer_code(mode, delay_s),
            str(directory),
        ]

    def test_legacy_base_summary_race_is_reproducible(self):
        """This is the old parent ordering: base summary permits teardown."""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            process = subprocess.Popen(
                self.child_command(directory, "base_then_delay", 1.0),
                start_new_session=True,
            )
            deadline = time.monotonic() + 2.0
            while not (directory / BASE_SUMMARY_FILENAME).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((directory / BASE_SUMMARY_FILENAME).is_file())
            os.killpg(process.pid, 15)
            process.wait(timeout=2.0)
            self.assertFalse((directory / STANDING_SUMMARY).exists())
            self.assertFalse((directory / TERMINAL_MANIFEST_FILENAME).exists())

    def test_starting_revision_has_the_same_base_before_standing_order(self):
        observer = subprocess.check_output(
            [
                "git",
                "show",
                "3663b6dcd2e1dad4cb1b0cb0f3afc0bbb6ecdf35:"
                "scripts/research/issue80_raicom_stand_observer.py",
            ],
            text=True,
        )
        parent = subprocess.check_output(
            [
                "git",
                "show",
                "3663b6dcd2e1dad4cb1b0cb0f3afc0bbb6ecdf35:"
                "scripts/research/issue80_raicom_reset_only.py",
            ],
            text=True,
        )
        self.assertLess(
            observer.index("super()._finish(status,reason)"),
            observer.index("_write_new_json(self.output_dir/'stand_takeover_summary.json'"),
        )
        self.assertIn(
            'self.evidence / "reset_only_observer_summary.json"',
            parent,
        )

    def test_correct_wrapper_waits_for_all_summaries_and_child_exit(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            status = run_child(
                self.child_command(directory, "base_then_delay", 0.1),
                output_dir=directory,
                timeout_s=2.0,
                required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
            )
            self.assertEqual(status, 0)
            self.assertTrue((directory / BASE_SUMMARY_FILENAME).is_file())
            self.assertTrue((directory / STANDING_SUMMARY).is_file())
            terminal = validate_terminal_manifest(
                directory,
                required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
            )
            exit_record = validate_process_exit(
                directory,
                required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
            )
            self.assertEqual(terminal["terminal_state"], "FINALIZED")
            self.assertEqual(exit_record["returncode"], 0)

    def test_timeout_and_exception_never_validate_as_terminal_success(self):
        for mode, timeout_s in (("timeout", 0.05), ("exception", 1.0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                status = run_child(
                    self.child_command(directory, mode, 0.5),
                    output_dir=directory,
                    timeout_s=timeout_s,
                    required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
                )
                self.assertEqual(status, 2)
                exit_record = json.loads((directory / PROCESS_EXIT_FILENAME).read_text())
                self.assertEqual(exit_record["terminal_validation"]["status"], "BLOCKED")
                with self.assertRaises(TerminalProtocolError):
                    validate_process_exit(
                        directory,
                        required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
                    )

    def test_truncated_trace_and_incomplete_standing_summary_fail_closed(self):
        for mode in ("truncated_trace", "incomplete_standing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                status = run_child(
                    self.child_command(directory, mode),
                    output_dir=directory,
                    timeout_s=1.0,
                    required_summary_filenames=[BASE_SUMMARY_FILENAME, STANDING_SUMMARY],
                )
                self.assertEqual(status, 2)
                exit_record = json.loads((directory / PROCESS_EXIT_FILENAME).read_text())
                self.assertEqual(exit_record["returncode"], 0)
                self.assertEqual(exit_record["terminal_validation"]["status"], "BLOCKED")

    def test_duplicate_terminal_publication_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "artifact.json"
            atomic_write_new_json(path, {"status": "first"})
            with self.assertRaises(TerminalProtocolError):
                atomic_write_new_json(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text())["status"], "first")

    def test_parent_refuses_cleanup_without_process_completion_proof(self):
        with tempfile.TemporaryDirectory() as raw:
            run = object.__new__(ResetOnlyRun)
            run.evidence = Path(raw)
            run.runtime = type("Runtime", (), {"deadline": time.monotonic() + 1.0})()
            run.container_watchdog_fired = threading.Event()
            run.container_watchdog_fired.set()
            run.required_observer_summaries = (BASE_SUMMARY_FILENAME, STANDING_SUMMARY)
            with self.assertRaises(ResetOnlyError) as raised:
                run._wait_for_observer_completion()
            self.assertIn("watchdog", str(raised.exception))

        with tempfile.TemporaryDirectory() as raw:
            run = object.__new__(ResetOnlyRun)
            run.evidence = Path(raw)
            run.runtime = type("Runtime", (), {"deadline": time.monotonic() - 1.0})()
            run.container_watchdog_fired = threading.Event()
            run.required_observer_summaries = (BASE_SUMMARY_FILENAME, STANDING_SUMMARY)
            with self.assertRaises(ResetOnlyError) as raised:
                run._wait_for_observer_completion()
            self.assertIn("COMPLETION_NOT_PROVEN", str(raised.exception))

    def test_standing_wiring_requires_standing_specific_summary(self):
        from issue80_raicom_stand_takeover import StandingRun
        import argparse
        self.assertIn(STANDING_SUMMARY, StandingRun.required_observer_summaries)
        with tempfile.TemporaryDirectory() as raw:
            run = StandingRun(
                argparse.Namespace(attempt_id="offline", protocol="jd-sd"),
                Path(raw),
                "offline",
            )
            tokens = run.observer_command_tokens()
        self.assertIn("issue80_raicom_observer_wrapper.py", " ".join(tokens))
        self.assertIn("--required-summary", tokens)
        self.assertIn(STANDING_SUMMARY, tokens)
        self.assertNotIn("timeout", tokens)
        source = Path(
            "issue80_raicom_stand_observer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _terminal_extra_summaries", source)
        self.assertNotIn("super()._finish(status,reason)", source)


if __name__ == "__main__":
    unittest.main()
