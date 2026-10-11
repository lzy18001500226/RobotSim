from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.agent import task_notifications


ROOT = Path(__file__).resolve().parents[1]


class TaskLifecycleCliTests(unittest.TestCase):
    def test_real_shaped_events_are_recorded_without_prompt_or_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "notifications.sqlite3"
            env = dict(os.environ)
            env["ROBOTSIM_TASK_NOTIFICATION_DB"] = str(database)
            events = (
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "synthetic-session",
                    "turn_id": "synthetic-turn",
                    "cwd": "/tmp/synthetic-project",
                    "prompt": "private synthetic task prompt",
                },
                {
                    "hook_event_name": "Stop",
                    "session_id": "synthetic-session",
                    "turn_id": "synthetic-turn",
                    "stop_hook_active": False,
                    "last_assistant_message": "private synthetic completion",
                },
            )
            for event in events:
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "scripts/agent/task_lifecycle.py")],
                    input=json.dumps(event),
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")
            snapshot = task_notifications.snapshot(env)
            self.assertEqual(len(snapshot["turns"]), 1)
            turn = snapshot["turns"][0]
            self.assertEqual(turn["session_id"], "synthetic-session")
            self.assertEqual(turn["turn_id"], "synthetic-turn")
            self.assertEqual(turn["prompt_present"], 1)
            self.assertEqual(turn["completion_present"], 1)
            contents = database.read_bytes()
            self.assertNotIn(b"private synthetic task prompt", contents)
            self.assertNotIn(b"private synthetic completion", contents)
            self.assertNotIn(b"/tmp/synthetic-project", contents)

    def test_invalid_and_oversized_payloads_fail_open_without_logging_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["ROBOTSIM_TASK_NOTIFICATION_DB"] = str(Path(directory) / "notifications.sqlite3")
            for payload in ("not-json", "x" * (1_048_576 + 1)):
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "scripts/agent/task_lifecycle.py")],
                    input=payload,
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")
            self.assertFalse(Path(env["ROBOTSIM_TASK_NOTIFICATION_DB"]).exists())


if __name__ == "__main__":
    unittest.main()
