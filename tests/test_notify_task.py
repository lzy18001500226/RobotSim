from __future__ import annotations

import contextlib
import concurrent.futures
import io
import json
import threading
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

    def _closeout_event(self, **changes: object) -> dict[str, object]:
        event: dict[str, object] = {
            "schema_version": notify_task.TASK_CLOSEOUT_SCHEMA_VERSION,
            "event_id": "auto",
            "repository": notify_task.ROBOTSIM_REPOSITORY,
            "task_id": "issue-44-unified-task-closeout",
            "attempt_id": "00000000-0000-4000-8000-000000000001",
            "worker": "Codex Cloud",
            "task_kind": "implementation",
            "status": "completed",
            "summary": "Unified task closeout is ready for review.",
            "branch": "issue/38-research-completion-event",
            "head_sha": "a" * 40,
            "pr_number": 42,
            "validation": ["python3 -m unittest discover -s tests: passed"],
            "evidence": ["https://github.com/lzy18001500226/RobotSim/pull/42"],
            "blockers": [],
            "recommended_next_action": "Review PR #42.",
            "completed_at": "2026-10-04T12:00:00Z",
        }
        event.update(changes)
        return event

    def _mock_closeout_transport(
        self,
        *,
        comments: list[dict[str, object]] | None = None,
        comment_pages: dict[int, list[dict[str, object]]] | None = None,
        fail_mail_attempts: int = 0,
        pr_repository: str = notify_task.ROBOTSIM_REPOSITORY,
        pr_body: str = "Closes #44",
        pr_branch: str = "issue/38-research-completion-event",
        fail_comment_response_once: bool = False,
        race_read_barrier: threading.Barrier | None = None,
        race_post_barrier: threading.Barrier | None = None,
    ) -> tuple[object, list[object], dict[str, int]]:
        stored_comments = [] if comments is None else comments
        requests: list[object] = []
        counts = {"github_get": 0, "github_post": 0, "github_delete": 0, "mail_post": 0}
        store_guard = threading.Lock()
        first_page_reads = 0
        writes = 0

        def open_request(request: object, timeout: int) -> FakeResponse:
            nonlocal first_page_reads, writes
            requests.append(request)
            parsed = urllib.parse.urlsplit(request.full_url)  # type: ignore[attr-defined]
            if parsed.netloc == "api.github.com":
                method = request.get_method()  # type: ignore[attr-defined]
                if method == "GET":
                    counts["github_get"] += 1
                    if parsed.path == "/user":
                        return FakeResponse({"login": "robotsim-test"}, 200)
                    if "/pulls/" in parsed.path:
                        number = int(parsed.path.rsplit("/", 1)[-1])
                        return FakeResponse({
                            "number": number,
                            "base": {"repo": {"full_name": pr_repository}},
                            "head": {"sha": "a" * 40, "ref": pr_branch},
                            "body": pr_body,
                        }, 200)
                    query = urllib.parse.parse_qs(parsed.query)
                    page = int(query.get("page", ["1"])[0])
                    if comment_pages is not None:
                        return FakeResponse(comment_pages.get(page, []), 200)
                    with store_guard:
                        snapshot = list(stored_comments) if page == 1 else []
                        if page == 1:
                            first_page_reads += 1
                            wait_for_race = race_read_barrier is not None and first_page_reads <= 2
                        else:
                            wait_for_race = False
                    if wait_for_race:
                        race_read_barrier.wait(timeout=5)
                    return FakeResponse(snapshot, 200)
                if method == "POST":
                    payload = json.loads(request.data)  # type: ignore[attr-defined]
                    with store_guard:
                        counts["github_post"] += 1
                        comment_id = 100 + counts["github_post"]
                        stored_comments.append({
                            "id": comment_id,
                            "body": payload["body"],
                            "user": {"login": "robotsim-test"},
                        })
                        writes += 1
                        wait_for_race = race_post_barrier is not None and writes <= 2
                    if wait_for_race:
                        race_post_barrier.wait(timeout=5)
                    if fail_comment_response_once and counts["github_post"] == 1:
                        raise urllib.error.URLError("mock response lost after GitHub persisted the comment")
                    return FakeResponse({"id": comment_id}, 201)
                if method == "DELETE":
                    counts["github_delete"] += 1
                    comment_id = int(parsed.path.rsplit("/", 1)[-1])
                    with store_guard:
                        stored_comments[:] = [c for c in stored_comments if c.get("id") != comment_id]
                    return FakeResponse({}, 204)
                raise AssertionError(f"unexpected GitHub method: {method}")
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

    def _closeout_stop_payload(self, event: dict[str, object], tail: str = "") -> dict[str, object]:
        envelope = json.dumps(event, ensure_ascii=False)
        message = (
            "Task finished.\n\n"
            "<!-- robotsim.task-closeout.v1\n"
            f"{envelope}\n"
            "-->"
        )
        if tail:
            message += "\n" + tail
        return {
            "session_id": "closeout-session",
            "turn_id": "closeout-turn",
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

    def test_closeout_schema_and_attempt_identity(self) -> None:
        first = notify_task.normalize_task_closeout(self._closeout_event())
        retry = notify_task.normalize_task_closeout(self._closeout_event())
        next_attempt = notify_task.normalize_task_closeout(
                self._closeout_event(attempt_id="00000000-0000-4000-8000-000000000002")
        )
        other_worker = notify_task.normalize_task_closeout(
            self._closeout_event(worker="Local01")
        )
        self.assertEqual(first["event_id"], retry["event_id"])
        self.assertNotEqual(first["event_id"], next_attempt["event_id"])
        self.assertNotEqual(first["event_id"], other_worker["event_id"])
        self.assertEqual(set(first), notify_task.TASK_CLOSEOUT_FIELDS)
        self.assertEqual(first["schema_version"], "robotsim.task-closeout.v1")
        for worker in ("Bearer synthetic-secret", "worker /secrets/credential", "https://user:pass@example.invalid"):
            with self.subTest(worker=worker), self.assertRaises(ValueError):
                notify_task.normalize_task_closeout(self._closeout_event(worker=worker))

    def test_closeout_supports_pr_issue_and_all_task_kinds(self) -> None:
        cases = (
            ("implementation", 42, "issues/42/comments"),
            ("experiment", None, "issues/44/comments"),
            ("research", None, "issues/44/comments"),
            ("review", 42, "issues/42/comments"),
        )
        for index, (kind, pr_number, expected_path) in enumerate(cases):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                settings = self._settings(directory)
                event = self._closeout_event(
                    task_kind=kind,
                    pr_number=pr_number,
                    task_id=f"issue-44-closeout-{kind}",
                    attempt_id=f"00000000-0000-4000-8000-00000000000{index}",
                )
                transport, requests, counts = self._mock_closeout_transport()
                with patch.object(notify_task, "_open_request", side_effect=transport):
                    result = notify_task.process_task_closeout(event, environ=settings)
                self.assertIn("GitHub event persisted", result.message)
                self.assertEqual(counts["github_post"], 1)
                post = next(request for request in requests if request.get_method() == "POST")
                self.assertIn(expected_path, post.full_url)
                if pr_number is not None:
                    self.assertTrue(any("/pulls/42" in r.full_url for r in requests))

    def test_pr_destination_must_link_the_originating_issue(self) -> None:
        transport, _, counts = self._mock_closeout_transport(pr_body="Closes #45")
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                result = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
        self.assertEqual(result.state, "failed")
        self.assertIn("must link the originating Issue", result.message)
        self.assertEqual(counts["github_post"], 0)
        self.assertEqual(counts["mail_post"], 0)

    def test_pr_closing_reference_validation_accepts_only_real_canonical_references(self) -> None:
        task_id = "issue-44-unified-task-closeout"
        accepted_bodies = (
            "Closes #44",
            "Resolves lzy18001500226/RobotSim#44",
        )
        rejected_bodies = (
            "Closes someoneelse/OtherRepo#44",
            "```text\nCloses #44\n```",
            "Example: `Closes #44`",
            "    Closes #44",
            "Closes #45",
        )
        for body in accepted_bodies:
            with self.subTest(body=body):
                self.assertTrue(notify_task._pr_closes_task_issue({"body": body}, task_id))
        for body in rejected_bodies:
            with self.subTest(body=body):
                self.assertFalse(notify_task._pr_closes_task_issue({"body": body}, task_id))

    def test_pr_wrong_repository_or_fenced_example_cannot_persist_closeout(self) -> None:
        for body in ("Closes someoneelse/OtherRepo#44", "```md\nCloses #44\n```"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                transport, _, counts = self._mock_closeout_transport(pr_body=body)
                with patch.object(notify_task, "_open_request", side_effect=transport):
                    result = notify_task.process_task_closeout(
                        self._closeout_event(), environ=self._settings(directory)
                    )
            self.assertEqual(result.state, "failed")
            self.assertEqual(counts["github_post"], 0)
            self.assertEqual(counts["mail_post"], 0)

    def test_pr_closeout_rejects_stale_head_sha(self) -> None:
        transport, _, counts = self._mock_closeout_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            event = self._closeout_event(head_sha="b" * 40)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                result = notify_task.process_task_closeout(event, environ=settings)
        self.assertEqual(result.state, "failed")
        self.assertIn("branch/head_sha do not match", result.message)
        self.assertEqual(counts["github_post"], 0)

    def test_pr_closeout_rejects_branch_mismatch(self) -> None:
        transport, _, counts = self._mock_closeout_transport(pr_branch="different-branch")
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                result = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
        self.assertEqual(result.state, "failed")
        self.assertIn("branch/head_sha do not match", result.message)
        self.assertEqual(counts["github_post"], 0)

    def test_blocked_attempt_then_completed_rerun_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            transport, _, counts = self._mock_closeout_transport()
            blocked = self._closeout_event(
                pr_number=None,
                attempt_id="00000000-0000-4000-8000-000000000002",
                status="blocked",
                completed_at="2026-10-04T12:00:00Z",
                blockers=["Waiting for a simulator artifact."],
            )
            completed = self._closeout_event(
                pr_number=None,
                attempt_id="00000000-0000-4000-8000-000000000003",
                status="completed",
                completed_at="2026-10-04T13:00:00Z",
                blockers=[],
            )
            self.assertNotEqual(
                notify_task.normalize_task_closeout(blocked)["event_id"],
                notify_task.normalize_task_closeout(completed)["event_id"],
            )
            with patch.object(notify_task, "_open_request", side_effect=transport):
                self.assertEqual(notify_task.process_task_closeout(blocked, environ=settings).state, "sent")
                self.assertEqual(notify_task.process_task_closeout(completed, environ=settings).state, "sent")
        self.assertEqual(counts["github_post"], 2)

    def test_same_attempt_retry_is_deduplicated_and_payload_changes_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            transport, _, counts = self._mock_closeout_transport()
            with patch.object(notify_task, "_open_request", side_effect=transport):
                first = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
                retry = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
                changed = notify_task.process_task_closeout(
                    self._closeout_event(summary="Changed content under the same attempt."),
                    environ=settings,
                )
        self.assertEqual(first.state, "sent")
        self.assertEqual(retry.state, "duplicate")
        self.assertEqual(changed.state, "failed")
        self.assertIn("different closeout payload", changed.message)
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)

    def test_different_workers_on_same_issue_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            transport, _, counts = self._mock_closeout_transport()
            event = self._closeout_event(
                pr_number=None, attempt_id="00000000-0000-4000-8000-000000000004"
            )
            with patch.object(notify_task, "_open_request", side_effect=transport):
                first = notify_task.process_task_closeout(event, environ=settings)
                second = notify_task.process_task_closeout(
                    dict(event, worker="Local01"), environ=settings
                )
        self.assertEqual(first.state, "sent")
        self.assertEqual(second.state, "sent")
        self.assertEqual(counts["github_post"], 2)

    def test_legacy_research_event_is_migrated_to_unified_schema(self) -> None:
        event = notify_task.normalize_research_event(self._research_event())
        self.assertEqual(event["schema_version"], notify_task.TASK_CLOSEOUT_SCHEMA_VERSION)
        self.assertEqual(event["task_kind"], "research")
        self.assertEqual(event["status"], "completed")
        self.assertTrue(str(event["attempt_id"]).startswith("legacy-"))

    def test_legacy_research_stop_envelope_uses_unified_persistence(self) -> None:
        transport, requests, counts = self._mock_closeout_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            payload = self._research_stop_payload(self._research_event(), "Transcript is not durable.")
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(notify_task.main(["stop"]), 0)
        self.assertEqual(counts["github_post"], 1)
        github_body = json.loads(next(r for r in requests if r.get_method() == "POST").data)["body"]
        payload_match = notify_task.TASK_CLOSEOUT_COMMENT_JSON.search(github_body)
        self.assertIsNotNone(payload_match)
        assert payload_match is not None
        migrated = json.loads(payload_match.group(1))
        self.assertEqual(migrated["task_kind"], "research")
        self.assertEqual(migrated["schema_version"], notify_task.TASK_CLOSEOUT_SCHEMA_VERSION)
        self.assertNotIn("Transcript is not durable", github_body)

    def test_comment_json_is_safe_inside_hidden_marker(self) -> None:
        event = notify_task.normalize_task_closeout(
            self._closeout_event(summary="Text with --> and --!> comment delimiters.")
        )
        body = notify_task._task_closeout_comment_body(event)
        payload_match = notify_task.TASK_CLOSEOUT_COMMENT_JSON.search(body)
        self.assertIsNotNone(payload_match)
        assert payload_match is not None
        self.assertNotIn("-->", payload_match.group(1))
        self.assertNotIn("--!>", payload_match.group(1))
        self.assertEqual(json.loads(payload_match.group(1)), event)
        self.assertEqual(
            notify_task._existing_event_state(
                [{"id": 1, "body": body, "user": {"login": "robotsim-test"}}],
                event,
                "robotsim-test",
            ),
            "same",
        )

    def test_local_closeout_stop_emits_one_event_without_generic_notification(self) -> None:
        transport, requests, counts = self._mock_closeout_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            payload = self._closeout_stop_payload(
                self._closeout_event(), "Transcript tail must not be forwarded."
            )
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
        request_data = "\n".join(str(request.data) for request in requests if request.data)
        self.assertNotIn("Transcript tail", request_data)
        self.assertNotIn("Local Codex turn", request_data)

    def test_cloud_closeout_command_uses_same_path_without_local_hook_opt_in(self) -> None:
        transport, requests, counts = self._mock_closeout_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            settings.pop("ROBOTSIM_LOCAL_STOP_HOOK")
            stdout = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(self._closeout_event()))),
                contextlib.redirect_stdout(stdout),
            ):
                code = notify_task.main(["task-closeout"])
        self.assertEqual(code, 0)
        self.assertIn("GitHub event persisted", stdout.getvalue())
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)
        post = next(request for request in requests if request.get_method() == "POST")
        body = json.loads(post.data)["body"]
        stored = notify_task.TASK_CLOSEOUT_COMMENT_JSON.search(body)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(json.loads(stored.group(1)), notify_task.normalize_task_closeout(self._closeout_event()))

    def test_stop_rejects_multiple_or_malformed_closeout_envelopes(self) -> None:
        transport, _, _ = self._mock_closeout_transport()
        valid = self._closeout_stop_payload(self._closeout_event())
        valid_text = valid["last_assistant_message"]
        malformed = dict(valid, last_assistant_message="<!-- robotsim.task-closeout.v1\n{oops}\n-->")
        multiple = dict(valid, last_assistant_message=f"{valid_text}\n{valid_text}")
        for payload in (malformed, multiple):
            with tempfile.TemporaryDirectory() as directory:
                settings = self._settings(directory)
                stderr = io.StringIO()
                with (
                    patch.object(notify_task, "_open_request", side_effect=transport) as urlopen,
                    patch.dict("os.environ", settings, clear=True),
                    patch("sys.stdin", io.StringIO(json.dumps(payload))),
                    contextlib.redirect_stderr(stderr),
                ):
                    self.assertEqual(notify_task.main(["stop"]), 0)
                self.assertIn("invalid task closeout", stderr.getvalue())
                urlopen.assert_not_called()

    def test_github_persistence_failure_never_sends_agentmail(self) -> None:
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
                patch("sys.stdin", io.StringIO(json.dumps(self._closeout_event()))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["task-closeout"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("HTTP 403", stderr.getvalue())
        self.assertNotIn("fake-github-token", stderr.getvalue())
        self.assertNotIn("fake-test-key", stderr.getvalue())
        urlopen.assert_called_once()

    def test_local_stop_stays_nonblocking_when_github_persistence_fails(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.github.com", 403, "Bearer fake-github-token", {}, io.BytesIO(b"secret")
        )
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            payload = self._closeout_stop_payload(self._closeout_event())
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=error),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["stop"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("HTTP 403", stderr.getvalue())
        self.assertNotIn("fake-github-token", stderr.getvalue())

    def test_cloud_agentmail_failure_does_not_fail_durable_closeout_command(self) -> None:
        transport, _, _ = self._mock_closeout_transport(fail_mail_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(notify_task, "_open_request", side_effect=transport),
                patch.dict("os.environ", settings, clear=True),
                patch("sys.stdin", io.StringIO(json.dumps(self._closeout_event()))),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = notify_task.main(["task-closeout"])
        self.assertEqual(code, 0)
        self.assertIn("GitHub event persisted", stderr.getvalue())
        self.assertIn("AgentMail delivery failed", stderr.getvalue())

    def test_retry_after_ambiguous_github_comment_response_does_not_post_twice(self) -> None:
        transport, _, counts = self._mock_closeout_transport(fail_comment_response_once=True)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                first = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
                retry = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
        self.assertEqual(first.state, "failed")
        self.assertIn("GitHub closeout persistence failed", first.message)
        self.assertEqual(retry.state, "sent")
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 1)

    def test_agentmail_failure_after_github_success_keeps_closeout_for_retry(self) -> None:
        transport, _, counts = self._mock_closeout_transport(fail_mail_attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                failed = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
                retried = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
        self.assertEqual(failed.state, "failed")
        self.assertIn("GitHub event persisted", failed.message)
        self.assertIn("AgentMail delivery failed", failed.message)
        self.assertEqual(retried.state, "sent")
        self.assertEqual(counts["github_post"], 1)
        self.assertEqual(counts["mail_post"], 2)

    def test_missing_github_persistence_capability_skips_agentmail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            settings.pop("GITHUB_TOKEN")
            with patch.object(notify_task, "_open_request") as urlopen:
                result = notify_task.process_task_closeout(self._closeout_event(), environ=settings)
        self.assertEqual(result.state, "failed")
        self.assertIn("GitHub closeout persistence unavailable", result.message)
        urlopen.assert_not_called()

    def test_closeout_redacts_credentials_and_arbitrary_absolute_paths(self) -> None:
        event = notify_task.normalize_task_closeout(
            self._closeout_event(
                summary=(
                    "token=synthetic-secret GITHUB_TOKEN=fake-gh-token "
                    "client_secret=synthetic-client-secret "
                    "https://user:synthetic-url-password@example.invalid/path "
                    "checked /secrets/project-key.txt after review"
                )
            )
        )
        rendered = json.dumps(event)
        for private_text in (
            "synthetic-secret", "fake-gh-token", "synthetic-client-secret",
            "synthetic-url-password", "/secrets/project-key.txt", "after review",
        ):
            self.assertNotIn(private_text, rendered)
        self.assertIn("[redacted credential]", event["summary"])
        self.assertIn("[local path omitted]", event["summary"])
        for path in (r"C:\Users\Local01\results.json", r"\\server\share\results.csv"):
            sanitized = notify_task.normalize_task_closeout(
                self._closeout_event(summary=f"Diagnostic saved to {path}")
            )
            self.assertNotIn(path, sanitized["summary"])

    def test_closeout_rejects_local_or_credential_bearing_evidence(self) -> None:
        invalid_evidence = (
            ["/tmp/results.json"],
            ["https://localhost./issue/44"],
            ["https://internal/issue/44"],
            ["https://example.invalid/?%74oken=synthetic-secret"],
            ["https://example.invalid/#token=synthetic-secret"],
            ["https://user:synthetic-secret@example.invalid/issue/44"],
            ["https://example.invalid/\nissue/44"],
        )
        for evidence in invalid_evidence:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                notify_task.normalize_task_closeout(self._closeout_event(evidence=evidence))

    def test_comment_marker_authentication_ignores_forged_marker(self) -> None:
        event = notify_task.normalize_task_closeout(self._closeout_event())
        forged = (
            f"<!-- robotsim-task-closeout:v1:{event['event_id']}:{'0' * 64} -->\n"
            f"<!-- robotsim-task-closeout-json:v1\n{notify_task._event_json(event)}\n-->"
        )
        self.assertEqual(
            notify_task._existing_event_state(
                [{"id": 1, "body": forged, "user": {"login": "attacker"}}],
                event,
                "robotsim-test",
            ),
            "absent",
        )

    def test_comment_pagination_finds_existing_closeout_without_posting(self) -> None:
        event = notify_task.normalize_task_closeout(self._closeout_event())
        existing = {
            "id": 555,
            "body": notify_task._task_closeout_comment_body(event),
            "user": {"login": "robotsim-test"},
        }
        pages = {1: [{"id": i, "body": "unrelated"} for i in range(100)], 2: [existing]}
        transport, _, counts = self._mock_closeout_transport(comment_pages=pages)
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            with patch.object(notify_task, "_open_request", side_effect=transport):
                result = notify_task.process_task_closeout(event, environ=settings)
        self.assertEqual(result.state, "sent")
        self.assertEqual(counts["github_get"], 4)  # Actor identity, PR check, and both comment pages.
        self.assertEqual(counts["github_post"], 0)

    def test_concurrent_same_attempt_writers_create_one_comment(self) -> None:
        transport, _, counts = self._mock_closeout_transport()
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(directory)
            event = self._closeout_event()
            with patch.object(notify_task, "_open_request", side_effect=transport):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(
                        lambda _: notify_task.process_task_closeout(event, environ=settings), range(2)
                    ))
        self.assertEqual(counts["github_post"], 1)
        self.assertTrue(all(outcome.state in {"sent", "duplicate"} for outcome in outcomes))
        self.assertEqual(counts["mail_post"], 1)

    def test_racing_writers_from_separate_state_dirs_consolidate_comments(self) -> None:
        stored_comments: list[dict[str, object]] = []
        read_barrier = threading.Barrier(2)
        post_barrier = threading.Barrier(2)
        transport, _, counts = self._mock_closeout_transport(
            comments=stored_comments,
            race_read_barrier=read_barrier,
            race_post_barrier=post_barrier,
        )
        with tempfile.TemporaryDirectory() as root:
            first_settings = self._settings(str(Path(root) / "local"))
            second_settings = self._settings(str(Path(root) / "cloud"))
            event = self._closeout_event()
            with patch.object(notify_task, "_open_request", side_effect=transport):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(
                        lambda settings: notify_task.process_task_closeout(event, environ=settings),
                        (first_settings, second_settings),
                    ))
        self.assertTrue(all(outcome.state in {"sent", "duplicate"} for outcome in outcomes))
        self.assertEqual(counts["github_post"], 2)
        self.assertGreaterEqual(counts["github_delete"], 1)
        self.assertEqual(len(stored_comments), 1)

    def test_racing_different_payloads_from_separate_state_dirs_keep_one_and_conflict(self) -> None:
        stored_comments: list[dict[str, object]] = []
        read_barrier = threading.Barrier(2)
        post_barrier = threading.Barrier(2)
        transport, _, counts = self._mock_closeout_transport(
            comments=stored_comments,
            race_read_barrier=read_barrier,
            race_post_barrier=post_barrier,
        )
        with tempfile.TemporaryDirectory() as root:
            local_settings = self._settings(str(Path(root) / "local"))
            cloud_settings = self._settings(str(Path(root) / "cloud"))
            events = (
                self._closeout_event(summary="Local writer payload."),
                self._closeout_event(summary="Cloud writer payload."),
            )
            self.assertEqual(
                notify_task.normalize_task_closeout(events[0])["event_id"],
                notify_task.normalize_task_closeout(events[1])["event_id"],
            )
            with patch.object(notify_task, "_open_request", side_effect=transport):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(
                        lambda pair: notify_task.process_task_closeout(pair[0], environ=pair[1]),
                        zip(events, (local_settings, cloud_settings)),
                    ))
        self.assertEqual(counts["github_post"], 2)
        self.assertEqual(len(stored_comments), 1)
        self.assertEqual(sum(outcome.state == "failed" for outcome in outcomes), 1)
        loser = next(outcome for outcome in outcomes if outcome.state == "failed")
        self.assertIn("concurrently used with a different payload", loser.message)
        winner_index = next(index for index, outcome in enumerate(outcomes) if outcome.state != "failed")
        canonical_payload = notify_task.TASK_CLOSEOUT_COMMENT_JSON.search(
            str(stored_comments[0]["body"])
        )
        self.assertIsNotNone(canonical_payload)
        assert canonical_payload is not None
        self.assertEqual(
            json.loads(canonical_payload.group(1)),
            notify_task.normalize_task_closeout(events[winner_index]),
        )
        self.assertEqual(counts["github_delete"], 1)


if __name__ == "__main__":
    unittest.main()
