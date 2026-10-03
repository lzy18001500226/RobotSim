from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.parse
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts.agent import notify_task


class FakeResponse:
    def __init__(self, payload: object = None, status: int = 202) -> None:
        self.status = status
        self.headers = {}
        self._body = json.dumps(payload if payload is not None else {}).encode("utf-8")

    def read(self) -> bytes:
        return self._body

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
            "GITHUB_TOKEN": "fake-github-token",
            "ROBOTSIM_LOCAL_STOP_HOOK": "1",
            "ROBOTSIM_NOTIFY_STATE_DIR": directory,
        }

    def _research_event(self, **changes: object) -> dict[str, object]:
        event: dict[str, object] = {
            "schema_version": notify_task.RESEARCH_SCHEMA_VERSION,
            "event_id": "auto",
            "repository": notify_task.ROBOTSIM_REPOSITORY,
            "task_id": "issue-38-agentmail-research",
            "worker": "Codex Cloud",
            "final_status": "completed",
            "summary": "Compared the research candidates and recorded the reuse findings.",
            "blockers": [],
            "recommended_next_action": "Review the findings and create a focused follow-up issue.",
            "evidence": ["https://github.com/lzy18001500226/RobotSim/issues/38"],
            "timestamp": "2026-10-04T12:00:00Z",
        }
        event.update(changes)
        return event

    def _mock_research_transport(
        self,
        *,
        comments: list[dict[str, object]] | None = None,
        fail_mail_attempts: int = 0,
    ) -> tuple[object, list[object], dict[str, int]]:
        stored_comments = [] if comments is None else comments
        requests: list[object] = []
        counts = {"github_get": 0, "github_post": 0, "mail_post": 0}

        def open_request(request: object, timeout: int) -> FakeResponse:
            requests.append(request)
            parsed = urllib.parse.urlsplit(request.full_url)  # type: ignore[attr-defined]
            if parsed.netloc == "api.github.com":
                if request.get_method() == "GET":  # type: ignore[attr-defined]
                    counts["github_get"] += 1
                    return FakeResponse(stored_comments, 200)
                counts["github_post"] += 1
                payload = json.loads(request.data)  # type: ignore[attr-defined]
                stored_comments.append({"body": payload["body"]})
                return FakeResponse({"id": 100 + counts["github_post"]}, 201)
            if parsed.netloc == "api.agentmail.to":
                counts["mail_post"] += 1
                if counts["mail_post"] <= fail_mail_attempts:
                    raise urllib.error.URLError("AGENTMAIL_API_KEY=fake-test-key Bearer fake-test-key")
                return FakeResponse({}, 202)
            raise AssertionError(f"unexpected mocked host: {parsed.netloc}")

        return open_request, requests, counts

    def _research_stop_payload(self, event: dict[str, object], tail: str = "") -> dict[str, object]:
        envelope = json.dumps(event, ensure_ascii=False)
        message = (
            "Research task finished.\n\n"
            "<!-- robotsim-research-completion:v1\n"
            f"{envelope}\n"
            "-->"
        )
        if tail:
            message += "\n" + tail
        return {
            "session_id": "research-session",
            "turn_id": "research-turn",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": message,
            "cwd": "",
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

    def test_successful_stop_hook_send_has_empty_stdout(self) -> None:
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
        self.assertEqual(stdout.getvalue(), "")
        send.assert_called_once()
        body = send.call_args.args[1]
        self.assertIn("Task: Local Codex turn", body)
        self.assertIn("Worker: Local Codex", body)
        self.assertIn("Branch: issue/28-agentmail-notifications", body)
        self.assertIn("Issue: #28", body)
        self.assertIn("Summary: Issue #28 complete", body)

    def test_recursive_stop_event_has_empty_stdout_and_does_not_send(self) -> None:
        payload = {
            "session_id": "mock-session-id",
            "turn_id": "mock-turn-id",
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }
        stdout = io.StringIO()
        with (
            patch.object(notify_task, "_send") as send,
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            contextlib.redirect_stdout(stdout),
        ):
            code = notify_task.main(["stop"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        send.assert_not_called()

    def test_duplicate_stop_event_has_empty_stdout_and_sends_once(self) -> None:
        payload = {
            "session_id": "same-session",
            "turn_id": "same-turn",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Completed the task.",
            "cwd": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_branch_at", return_value=""),
                patch.object(notify_task, "_send") as send,
                patch.dict("os.environ", self._settings(directory), clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                with patch("sys.stdin", io.StringIO(json.dumps(payload))):
                    first_code = notify_task.main(["stop"])
                with patch("sys.stdin", io.StringIO(json.dumps(payload))):
                    second_code = notify_task.main(["stop"])
        self.assertEqual(first_code, 0)
        self.assertEqual(second_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        send.assert_called_once()

    def test_different_turn_ids_in_one_session_notify_independently(self) -> None:
        payload = {
            "session_id": "shared-session",
            "turn_id": "turn-one",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "First turn finished.",
            "cwd": "",
        }
        second_payload = {**payload, "turn_id": "turn-two", "last_assistant_message": "Second turn finished."}
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_send") as send,
                patch.dict("os.environ", self._settings(directory), clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                for event in (payload, second_payload):
                    with patch("sys.stdin", io.StringIO(json.dumps(event))):
                        self.assertEqual(notify_task.main(["stop"]), 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(send.call_count, 2)
        self.assertNotEqual(
            send.call_args_list[0].args[3], send.call_args_list[1].args[3]
        )

    def test_missing_stop_hook_configuration_is_silent_and_non_blocking(self) -> None:
        payload = {
            "session_id": "session-without-config",
            "turn_id": "turn-without-config",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Finished.",
            "cwd": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(notify_task, "_send") as send,
                patch.dict(
                    "os.environ",
                    {"ROBOTSIM_LOCAL_STOP_HOOK": "1", "ROBOTSIM_NOTIFY_STATE_DIR": directory},
                    clear=True,
                ),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["stop"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        send.assert_not_called()

    def test_stop_hook_is_disabled_without_local_opt_in_even_when_cloud_is_configured(self) -> None:
        payload = {
            "session_id": "cloud-session",
            "turn_id": "cloud-turn",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Cloud task finished.",
            "cwd": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            cloud_settings = self._settings(directory)
            cloud_settings.pop("ROBOTSIM_LOCAL_STOP_HOOK")
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_send") as send,
                patch.dict("os.environ", cloud_settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(stdout),
            ):
                code = notify_task.main(["stop"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        send.assert_not_called()

    def test_stop_dry_run_keeps_manual_output_without_local_opt_in(self) -> None:
        payload = {
            "session_id": "dry-session",
            "turn_id": "dry-turn",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "Dry-run turn.",
            "cwd": "",
        }
        stdout = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            contextlib.redirect_stdout(stdout),
        ):
            code = notify_task.main(["stop", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no AgentMail request was made", stdout.getvalue())
        self.assertIn("[RobotSim] Completed: Dry-run turn.", stdout.getvalue())

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

    def test_research_event_schema_and_deterministic_event_id(self) -> None:
        first = notify_task.normalize_research_event(self._research_event())
        retry = notify_task.normalize_research_event(self._research_event())
        changed_payload = notify_task.normalize_research_event(
            self._research_event(summary="A changed payload with the same event identity.")
        )
        self.assertEqual(first["event_id"], retry["event_id"])
        self.assertEqual(first["event_id"], changed_payload["event_id"])
        self.assertTrue(notify_task.RESEARCH_EVENT_ID.fullmatch(str(first["event_id"])))
        self.assertEqual(set(first), notify_task.RESEARCH_EVENT_FIELDS)
        self.assertEqual(first["schema_version"], "robotsim.research-completion.v1")

    def test_research_event_schema_rejects_missing_fields_bad_status_and_transcript(self) -> None:
        missing = self._research_event()
        missing.pop("evidence")
        unsupported = self._research_event(transcript="raw transcript")
        bad_status = self._research_event(final_status="in_progress")
        for event in (missing, unsupported, bad_status):
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    notify_task.normalize_research_event(event)
        with self.assertRaisesRegex(ValueError, "timestamp must be an ISO 8601"):
            notify_task.normalize_research_event(
                self._research_event(timestamp="0001-01-01T00:00:00+23:59")
            )

    def test_research_event_rejects_malformed_local_evidence_without_api_calls(self) -> None:
        transport, _, _ = self._mock_research_transport()
        bad_evidence = (
            ["/tmp/research-notes.md"],
            ["https://localhost./issue/38"],
            ["https://internal/issue/38"],
            ["https://example.invalid/?%74oken=synthetic-secret"],
            ["https://example.invalid/?authorization=synthetic-secret"],
            ["https://user:synthetic-secret@example.invalid/issue/38"],
            ["https://example.invalid/\nissue/38"],
        )
        with patch.object(notify_task, "_open_request", side_effect=transport) as urlopen:
            for evidence in bad_evidence:
                with self.subTest(evidence=evidence):
                    result = notify_task.process_research_completion(
                        self._research_event(evidence=evidence),
                        environ={"GITHUB_TOKEN": "fake-github-token"},
                    )
                    self.assertEqual(result.state, "failed")
        urlopen.assert_not_called()

    def test_local_research_stop_emits_one_event_and_no_generic_completion(self) -> None:
        transport, requests, counts = self._mock_research_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            payload = self._research_stop_payload(self._research_event(), "Transcript tail must not be forwarded.")
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(stdout),
            ):
                first = notify_task.main(["stop"])
                with patch("sys.stdin", io.StringIO(json.dumps(payload))):
                    second = notify_task.main(["stop"])
        self.assertEqual((first, second), (0, 0))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)
        self.assertFalse(any("Local Codex turn" in str(request.data) for request in requests))

    def test_cloud_research_closeout_emits_one_event_without_local_opt_in(self) -> None:
        transport, requests, counts = self._mock_research_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            settings.pop("ROBOTSIM_LOCAL_STOP_HOOK")
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(self._research_event()))),
                contextlib.redirect_stdout(stdout),
            ):
                code = notify_task.main(["research-completion"])
        self.assertEqual(code, 0)
        self.assertIn("GitHub event persisted", stdout.getvalue())
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)
        github_requests = [request for request in requests if "api.github.com" in request.full_url]
        self.assertEqual(
            github_requests[0].full_url.split("?", 1)[0],
            "https://api.github.com/repos/lzy18001500226/RobotSim/issues/38/comments",
        )
        self.assertEqual(github_requests[0].get_header("Authorization"), "Bearer fake-github-token")
        github_comment = json.loads(github_requests[1].data)["body"]
        event_payload = notify_task.RESEARCH_COMMENT_JSON.search(github_comment)
        self.assertIsNotNone(event_payload)
        assert event_payload is not None
        expected_event = notify_task.normalize_research_event(self._research_event())
        self.assertEqual(json.loads(event_payload.group(1)), expected_event)
        agentmail_request = next(request for request in requests if "api.agentmail.to" in request.full_url)
        mail_body = json.loads(agentmail_request.data)["text"]
        self.assertEqual(json.loads(mail_body), expected_event)

    def test_github_persistence_failure_is_sanitized_and_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            error = urllib.error.HTTPError(
                "https://api.github.com", 403, "Bearer fake-github-token", {}, io.BytesIO(b"secret")
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=error) as urlopen,
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(self._research_event()))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["research-completion"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("HTTP 403", stderr.getvalue())
        self.assertNotIn("fake-github-token", stderr.getvalue())
        self.assertNotIn("fake-test-key", stderr.getvalue())
        urlopen.assert_called_once()

    def test_existing_same_research_event_comment_is_not_reposted(self) -> None:
        event = notify_task.normalize_research_event(self._research_event())
        comments = [{"body": notify_task._research_comment_body(event)}]
        transport, _, counts = self._mock_research_transport(comments=comments)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                result = notify_task.process_research_completion(event, environ=settings)
        self.assertEqual(result.state, "sent")
        self.assertIn("already exists on Issue #38", result.message)
        self.assertEqual(counts["github_post"], 0)
        self.assertEqual(counts["mail_post"], 1)

    def test_issue_comment_json_cannot_terminate_its_hidden_marker(self) -> None:
        event = notify_task.normalize_research_event(
            self._research_event(summary="Text with --> and --!> comment terminators.")
        )
        body = notify_task._research_comment_body(event)
        stored_json = notify_task.RESEARCH_COMMENT_JSON.search(body)
        self.assertIsNotNone(stored_json)
        assert stored_json is not None
        self.assertNotIn("-->", stored_json.group(1))
        self.assertNotIn("--!>", stored_json.group(1))
        self.assertEqual(json.loads(stored_json.group(1)), event)
        self.assertEqual(notify_task._existing_event_state([{"body": body}], event), "same")

    def test_agentmail_deduplicates_a_repeated_research_event(self) -> None:
        transport, _, counts = self._mock_research_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                first = notify_task.process_research_completion(self._research_event(), environ=settings)
                second = notify_task.process_research_completion(self._research_event(), environ=settings)
        self.assertEqual(first.state, "sent")
        self.assertEqual(second.state, "duplicate")
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)

    def test_retry_after_agentmail_failure_reuses_same_event_and_idempotency_key(self) -> None:
        transport, requests, counts = self._mock_research_transport(fail_mail_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                failed = notify_task.process_research_completion(self._research_event(), environ=settings)
                retried = notify_task.process_research_completion(self._research_event(), environ=settings)
        self.assertEqual(failed.state, "failed")
        self.assertIn("GitHub event persisted", failed.message)
        self.assertEqual(retried.state, "sent")
        self.assertEqual(counts["github_post"], 1)
        mail_requests = [request for request in requests if "api.agentmail.to" in request.full_url]
        self.assertEqual(len(mail_requests), 2)
        self.assertEqual(
            mail_requests[0].get_header("Idempotency-key"),
            mail_requests[1].get_header("Idempotency-key"),
        )

    def test_same_event_id_with_changed_payload_is_rejected(self) -> None:
        transport, _, counts = self._mock_research_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                first = notify_task.process_research_completion(self._research_event(), environ=settings)
                changed = notify_task.process_research_completion(
                    self._research_event(summary="The payload changed under the same event identity."),
                    environ=settings,
                )
        self.assertEqual(first.state, "sent")
        self.assertEqual(changed.state, "failed")
        self.assertIn("different or invalid payload", changed.message)
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)

    def test_research_event_does_not_forward_transcript_and_redacts_credentials(self) -> None:
        transport, requests, _ = self._mock_research_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            event = self._research_event(
                summary="Bearer fake-secret-token; notes in /home/codex/private research dump.txt; after path"
            )
            payload = self._research_stop_payload(event, "RAW TRANSCRIPT: do not forward this text")
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(notify_task.main(["stop"]), 0)
        request_bodies = [str(request.data) for request in requests]
        joined = "\n".join(request_bodies)
        self.assertNotIn("RAW TRANSCRIPT", joined)
        self.assertNotIn("fake-secret-token", joined)
        self.assertNotIn("/home/codex/private", joined)
        self.assertNotIn("research dump.txt", joined)
        self.assertNotIn("after path", joined)
        self.assertIn("[redacted]", joined)
        self.assertIn("[local path omitted]", joined)

    def test_research_event_redacts_generic_token_assignments(self) -> None:
        event = notify_task.normalize_research_event(
            self._research_event(
                summary=(
                    "token=synthetic-secret credential:synthetic-credential "
                    "GITHUB_TOKEN=fake-gh-token https://example.invalid/?signature=synthetic-signature"
                )
            )
        )
        self.assertNotIn("synthetic-secret", str(event))
        self.assertNotIn("synthetic-credential", str(event))
        self.assertNotIn("fake-gh-token", str(event))
        self.assertNotIn("synthetic-signature", str(event))

    def test_research_event_redacts_complete_private_key_blocks(self) -> None:
        event = notify_task.normalize_research_event(
            self._research_event(
                summary=(
                    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
                    "SYNTHETIC_PRIVATE_KEY_MATERIAL\n"
                    "-----END OPENSSH PRIVATE KEY----- trailing text"
                )
            )
        )
        self.assertNotIn("SYNTHETIC_PRIVATE_KEY_MATERIAL", str(event))
        self.assertEqual(event["summary"], "[redacted credential] trailing text")

    def test_research_delivery_failure_is_sanitized_and_non_blocking(self) -> None:
        transport, _, counts = self._mock_research_transport(fail_mail_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(self._research_event()))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["research-completion"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("AgentMail delivery failed", stderr.getvalue())
        self.assertNotIn("fake-test-key", stderr.getvalue())
        self.assertNotIn("fake-github-token", stderr.getvalue())
        self.assertEqual(counts["github_post"], 1)

    def test_missing_github_persistence_capability_does_not_send_agentmail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            settings.pop("GITHUB_TOKEN")
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_open_request") as urlopen,
                patch.dict("os.environ", settings, clear=True),
                contextlib.redirect_stdout(stdout),
            ):
                result = notify_task.process_research_completion(self._research_event())
        self.assertEqual(result.state, "failed")
        self.assertIn("GitHub Issue persistence unavailable", result.message)
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
