from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts.agent import notify_task


class FakeResponse:
    status = 202

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class NotifyTaskTests(unittest.TestCase):
    def _settings(self, directory: str) -> dict[str, str]:
        return {
            "AGENTMAIL_API_KEY": "fake-test-key",
            "AGENTMAIL_INBOX_ID": "lzy18001500226@agentmail.to",
            "ROBOTSIM_NOTIFY_TO": "maintainer@example.invalid",
            "ROBOTSIM_NOTIFY_STATE_DIR": directory,
        }

    def test_notification_is_short_and_includes_available_context(self) -> None:
        subject, body = notify_task.format_notification(
            "ready_for_review",
            "issue-28-pr-12",
            "Issue #28 notifications implemented",
            task="Issue #28: AgentMail notifications",
            worker="Codex Cloud",
            branch="issue/28-agentmail-notifications",
            issue="28",
        )
        self.assertEqual(subject, "[RobotSim] Ready for review: Issue #28 notifications implemented")
        self.assertIn("Task: Issue #28: AgentMail notifications", body)
        self.assertIn("Worker: Codex Cloud", body)
        self.assertIn("Branch: issue/28-agentmail-notifications", body)
        self.assertIn("Issue: #28", body)
        self.assertIn("Summary: Issue #28 notifications implemented", body)
        self.assertNotIn("\n", subject)

    def test_successful_agentmail_send_uses_current_rest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", return_value=FakeResponse()) as urlopen:
                result = notify_task.notify(
                    "completed", "task-1", "Tests passed", worker="Codex Cloud", environ=settings
                )

        self.assertEqual(result.state, "sent")
        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.agentmail.to/v0/inboxes/lzy18001500226@agentmail.to/messages/send",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertTrue(request.get_header("Idempotency-key"))
        self.assertEqual(json.loads(request.data)["to"], ["maintainer@example.invalid"])
        self.assertEqual(json.loads(request.data)["subject"], result.subject)

    def test_auth_provider_and_http_failures_are_non_blocking(self) -> None:
        for status in (401, 403, 429, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                settings = self._settings(directory)
                error = urllib.error.HTTPError(
                    "https://api.agentmail.to", status, "mock response", {}, io.BytesIO(b"mock")
                )
                with patch.object(notify_task, "_open_request", side_effect=error):
                    result = notify_task.notify("blocked", f"task-{status}", "Waiting", environ=settings)
                self.assertEqual(result.state, "failed")
                self.assertIn(f"HTTP {status}", result.message)
                self.assertNotIn(settings["AGENTMAIL_API_KEY"], result.message)

    def test_provider_transport_failure_is_non_blocking_and_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(
                notify_task, "_open_request", side_effect=urllib.error.URLError("mock network failure")
            ) as urlopen:
                result = notify_task.notify("completed", "task-network", "Finished", environ=settings)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(result.state, "failed")
        self.assertEqual(result.message, "AgentMail delivery failed (URLError)")
        urlopen.assert_called_once()

    def test_missing_configuration_skips_without_a_request(self) -> None:
        with patch.object(notify_task, "_open_request") as urlopen:
            result = notify_task.notify("completed", "task-2", "Finished", environ={})
        self.assertEqual(result.state, "skipped")
        self.assertIn("AGENTMAIL_API_KEY", result.message)
        self.assertIn("AGENTMAIL_INBOX_ID", result.message)
        self.assertIn("ROBOTSIM_NOTIFY_TO", result.message)
        urlopen.assert_not_called()

    def test_repeated_task_event_is_deduplicated_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", return_value=FakeResponse()) as urlopen:
                first = notify_task.notify("completed", "task-3", "Finished", environ=settings)
                second = notify_task.notify("completed", "task-3", "Finished", environ=settings)
            state_files = list(Path(directory).iterdir())

        self.assertEqual(first.state, "sent")
        self.assertEqual(second.state, "duplicate")
        urlopen.assert_called_once()
        self.assertEqual(len(state_files), 1)
        self.assertEqual(state_files[0].suffix, ".sent")

    def test_redirect_handler_refuses_to_forward_credentials(self) -> None:
        result = notify_task._NoRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://example.invalid/redirect"
        )
        self.assertIsNone(result)

    def test_retry_after_ambiguous_failure_reuses_agentmail_idempotency_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            error = urllib.error.URLError("mock connection closed after send")
            with patch.object(
                notify_task, "_open_request", side_effect=[error, FakeResponse()]
            ) as urlopen:
                failed = notify_task.notify("completed", "task-retry", "Finished", environ=settings)
                retried = notify_task.notify("completed", "task-retry", "Finished", environ=settings)

        self.assertEqual(failed.state, "failed")
        self.assertEqual(retried.state, "sent")
        first_request = urlopen.call_args_list[0].args[0]
        second_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(
            first_request.get_header("Idempotency-key"), second_request.get_header("Idempotency-key")
        )

    def test_stop_hook_payload_sends_one_turn_with_issue_and_branch_context(self) -> None:
        payload = {
            "session_id": "mock-session-id",
            "turn_id": "mock-turn-id",
            "transcript_path": None,
            "cwd": "/mock/checkout",
            "hook_event_name": "Stop",
            "model": "mock-model",
            "permission_mode": "default",
            "stop_hook_active": False,
            "last_assistant_message": "# Issue #28 complete\nPR opened and validation passed.",
        }
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_branch_at", return_value="issue/28-agentmail-notifications"),
                patch.object(notify_task, "_send") as send,
                patch.dict("os.environ", self._settings(directory), clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(stdout),
            ):
                code = notify_task.main(["stop"])

        self.assertEqual(code, 0)
        self.assertIn("notify_task: sent", stdout.getvalue())
        send.assert_called_once()
        body = send.call_args.args[1]
        self.assertIn("Task: Local Codex turn", body)
        self.assertIn("Worker: Local Codex", body)
        self.assertIn("Branch: issue/28-agentmail-notifications", body)
        self.assertIn("Issue: #28", body)
        self.assertIn("Summary: Issue #28 complete", body)

    def test_stop_hook_recursion_guard_skips_without_sending(self) -> None:
        payload = {
            "session_id": "mock-session-id",
            "turn_id": "mock-turn-id",
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }
        with patch.object(notify_task, "_send") as send:
            result = notify_task.process_stop_payload(payload, environ={})
        self.assertEqual(result.state, "skipped")
        self.assertIn("recursion guard", result.message)
        send.assert_not_called()

    def test_repeated_stop_payload_for_same_turn_sends_once(self) -> None:
        payload = {
            "session_id": "same-session",
            "turn_id": "same-turn",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Completed the task.",
            "cwd": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(notify_task, "_branch_at", return_value=""),
                patch.object(notify_task, "_send") as send,
            ):
                first = notify_task.process_stop_payload(payload, environ=self._settings(directory))
                second = notify_task.process_stop_payload(payload, environ=self._settings(directory))
        self.assertEqual(first.state, "sent")
        self.assertEqual(second.state, "duplicate")
        send.assert_called_once()

    def test_dry_run_does_not_read_configuration_or_call_agentmail(self) -> None:
        stdout = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(notify_task, "_open_request") as urlopen,
            contextlib.redirect_stdout(stdout),
        ):
            code = notify_task.main(
                ["ready_for_review", "--task-id", "issue-28-pr-12", "--summary", "PR opened", "--dry-run"]
            )
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no AgentMail request was made", stdout.getvalue())
        self.assertIn("[RobotSim] Ready for review: PR opened", stdout.getvalue())
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
