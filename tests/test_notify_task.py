from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agent import notify_task


class NotifyTaskTests(unittest.TestCase):
    def test_status_subjects_and_body_are_concise_single_line_headers(self) -> None:
        expected = {
            "ready_for_review": "Ready for review",
            "completed": "Completed",
            "blocked": "Blocked",
        }
        for status, label in expected.items():
            with self.subTest(status=status):
                subject, body = notify_task.format_notification(status, "task-7", "Docs\nupdated")
                self.assertEqual(subject, f"[RobotSim] {label}: Docs updated")
                self.assertIn("Task: task-7", body)
                self.assertIn("Summary: Docs updated", body)
                self.assertNotIn("\n", subject)

    def test_dry_run_formats_without_smtp_settings(self) -> None:
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout):
            code = notify_task.main(
                ["ready_for_review", "--task-id", "task-7", "--summary", "Review docs", "--dry-run"]
            )
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no SMTP connection was made", stdout.getvalue())
        self.assertIn("Subject: [RobotSim] Ready for review: Review docs", stdout.getvalue())

    def test_duplicate_status_for_task_is_sent_only_once(self) -> None:
        settings = {
            "ROBOTSIM_SMTP_HOST": "smtp.example.invalid",
            "ROBOTSIM_SMTP_FROM": "robot@example.invalid",
            "ROBOTSIM_NOTIFY_TO": "reviewer@example.invalid",
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(notify_task, "_send") as send:
            settings["ROBOTSIM_NOTIFY_STATE_DIR"] = directory
            first = notify_task.notify("completed", "task-7", "Checks passed", environ=settings)
            second = notify_task.notify("completed", "task-7", "Checks passed", environ=settings)
            self.assertEqual(first.state, "sent")
            self.assertEqual(second.state, "duplicate")
            send.assert_called_once()
            state_files = list(Path(directory).iterdir())
            self.assertEqual(len(state_files), 1)
            self.assertEqual(state_files[0].suffix, ".sent")

    def test_missing_auth_pair_skips_and_does_not_fail_task(self) -> None:
        settings = {
            "ROBOTSIM_SMTP_HOST": "smtp.example.invalid",
            "ROBOTSIM_SMTP_FROM": "robot@example.invalid",
            "ROBOTSIM_NOTIFY_TO": "reviewer@example.invalid",
            "ROBOTSIM_SMTP_USERNAME": "configured-user",
        }
        stderr = io.StringIO()
        with patch.dict(os.environ, settings, clear=True), patch.object(notify_task, "_send") as send:
            with contextlib.redirect_stderr(stderr):
                code = notify_task.main(
                    ["blocked", "--task-id", "task-8", "--summary", "Waiting for access"]
                )
        self.assertEqual(code, 0)
        self.assertIn("ROBOTSIM_SMTP_USERNAME and ROBOTSIM_SMTP_PASSWORD", stderr.getvalue())
        self.assertNotIn("configured-user", stderr.getvalue())
        send.assert_not_called()

    def test_smtp_uses_certificate_checked_starttls_and_environment_auth(self) -> None:
        calls: list[object] = []

        class FakeSMTP:
            def __init__(self, host: str, port: int, timeout: float) -> None:
                calls.append((host, port, timeout))

            def __enter__(self) -> "FakeSMTP":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def ehlo(self) -> None:
                calls.append("ehlo")

            def starttls(self, *, context: object) -> None:
                calls.append(("starttls", context.check_hostname, context.verify_mode))

            def login(self, username: str, password: str) -> None:
                calls.append(("login", username, password))

            def send_message(self, message: object) -> dict[str, str]:
                calls.append(("message", message["Subject"]))
                return {}

        settings = {
            "host": "smtp.example.invalid",
            "port": 587,
            "timeout": 10,
            "sender": "robot@example.invalid",
            "recipient": "reviewer@example.invalid",
            "username": "test-user",
            "password": "test-password",
        }
        with patch.object(notify_task.smtplib, "SMTP", FakeSMTP):
            notify_task._send("subject", "body", settings)
        self.assertEqual(calls[0], ("smtp.example.invalid", 587, 10.0))
        self.assertIn(("starttls", True, notify_task.ssl.CERT_REQUIRED), calls)
        self.assertIn(("login", "test-user", "test-password"), calls)
        self.assertIn(("message", "subject"), calls)

    def test_delivery_failure_releases_duplicate_claim_and_returns_success_code(self) -> None:
        settings = {
            "ROBOTSIM_SMTP_HOST": "smtp.example.invalid",
            "ROBOTSIM_SMTP_FROM": "robot@example.invalid",
            "ROBOTSIM_NOTIFY_TO": "reviewer@example.invalid",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, settings, clear=True):
            os.environ["ROBOTSIM_NOTIFY_STATE_DIR"] = directory
            stderr = io.StringIO()
            with patch.object(notify_task, "_send", side_effect=OSError("relay unavailable")):
                with contextlib.redirect_stderr(stderr):
                    code = notify_task.main(
                        ["blocked", "--task-id", "task-9", "--summary", "External service unavailable"]
                    )
            self.assertEqual(code, 0)
            self.assertIn("SMTP delivery failed (OSError)", stderr.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
