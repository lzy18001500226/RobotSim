from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from scripts.agent import task_notifications


class TaskNotificationStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "state" / "notifications.sqlite3"
        self.environ = {
            "ROBOTSIM_TASK_NOTIFICATION_DB": str(self.database),
            "AGENTMAIL_API_KEY": "synthetic-agentmail-key",
            "AGENTMAIL_INBOX_ID": "synthetic-inbox@example.invalid",
            "ROBOTSIM_NOTIFY_TO": "synthetic-recipient@example.invalid",
        }

    def event(self, **changes: object) -> dict[str, object]:
        result: dict[str, object] = {
            "event_id": "sha256:" + "a" * 64,
            "task_id": "issue-123-notification-state-test",
            "attempt_id": "00000000-0000-4000-8000-000000000001",
            "worker": "Codex",
            "task_kind": "implementation",
            "status": "completed",
            "summary": "Synthetic closeout",
        }
        result.update(changes)
        return result

    def test_lifecycle_records_metadata_without_message_text(self) -> None:
        prompt = "private synthetic prompt body"
        completion = "private synthetic assistant body"
        self.assertTrue(task_notifications.record_turn_event({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session-a",
            "turn_id": "turn-a",
            "cwd": "/home/user/RobotSim",
            "prompt": prompt,
        }, environ=self.environ))
        self.assertTrue(task_notifications.record_turn_event({
            "hook_event_name": "Interrupt",
            "session_id": "session-a",
            "turn_id": "turn-a",
        }, environ=self.environ))
        self.assertTrue(task_notifications.record_turn_event({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "session-a",
            "turn_id": "turn-b",
            "prompt": "resumed synthetic prompt",
        }, environ=self.environ))
        self.assertTrue(task_notifications.record_turn_event({
            "hook_event_name": "Stop",
            "session_id": "session-a",
            "turn_id": "turn-b",
            "stop_hook_active": False,
            "last_assistant_message": completion,
        }, environ=self.environ))

        snapshot = task_notifications.snapshot(self.environ)
        first, resumed = snapshot["turns"]
        self.assertEqual(first["task_state"], "PAUSED")
        self.assertEqual(resumed["task_state"], "UNCERTAIN")
        self.assertEqual(resumed["turn_state"], "ENDED")
        self.assertEqual(resumed["resume_candidate"], 1)
        self.assertEqual(resumed["completion_present"], 1)
        self.assertEqual(resumed["cwd_present"], 0)
        data = self.database.read_bytes()
        self.assertNotIn(prompt.encode(), data)
        self.assertNotIn(completion.encode(), data)
        self.assertNotIn(b"/home/user/RobotSim", data)

    def test_structured_outcome_is_idempotent_and_channels_are_independent(self) -> None:
        event = self.event()
        self.assertTrue(task_notifications.record_terminal_outcome(
            event,
            publication_state="pending",
            notifications_enabled=True,
            session_id="session-a",
            turn_id="turn-a",
            environ=self.environ,
        ))
        self.assertTrue(task_notifications.record_terminal_outcome(
            event,
            publication_state="pending",
            notifications_enabled=True,
            session_id="session-a",
            turn_id="turn-a",
            environ=self.environ,
        ))
        snapshot = task_notifications.snapshot(self.environ)
        self.assertEqual(len(snapshot["outcomes"]), 1)
        self.assertEqual(task_notifications.delivery_state(event["event_id"], "agentmail", self.environ), "pending")
        self.assertEqual(task_notifications.delivery_state(event["event_id"], "telephone", self.environ), "unavailable")
        self.assertEqual(snapshot["publications"][0]["state"], "pending")

        with self.assertRaises(task_notifications.EventConflict):
            task_notifications.record_terminal_outcome(
                self.event(summary="changed payload"),
                notifications_enabled=True,
                environ=self.environ,
            )

    def test_publication_and_email_claims_are_single_use_with_bounded_backoff(self) -> None:
        event = self.event()
        task_notifications.record_terminal_outcome(
            event,
            publication_state="pending",
            notifications_enabled=True,
            environ=self.environ,
        )

        def claim_publication(_: int) -> object:
            return task_notifications.claim_publication(self.environ, now=100)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(claim_publication, range(2)))
        self.assertEqual(sum(claim is not None for claim in claims), 1)
        task_notifications.finish_publication(event["event_id"], "failed", "HTTP 503", environ=self.environ, now=100)
        self.assertIsNone(task_notifications.claim_publication(self.environ, now=159))
        retry = task_notifications.claim_publication(self.environ, now=160)
        self.assertEqual(retry["attempt"], 2)

        email = task_notifications.claim_agentmail(self.environ, now=200)
        self.assertEqual(email["attempt"], 1)
        self.assertIsNone(task_notifications.claim_agentmail(self.environ, now=200))
        task_notifications.finish_agentmail(event["event_id"], "failed", "HTTP 503", environ=self.environ, now=200)
        self.assertIsNone(task_notifications.claim_agentmail(self.environ, now=259))
        self.assertEqual(task_notifications.claim_agentmail(self.environ, now=260)["attempt"], 2)

    def test_expired_delivery_lease_is_recovered_and_retry_limit_fails_closed(self) -> None:
        event = self.event()
        task_notifications.record_terminal_outcome(
            event,
            publication_state="pending",
            notifications_enabled=True,
            environ=self.environ,
        )
        first = task_notifications.claim_agentmail(self.environ, now=100)
        self.assertEqual(first["attempt"], 1)
        recovered = task_notifications.claim_agentmail(
            self.environ,
            now=100 + task_notifications.EMAIL_LEASE_SECONDS + 1,
        )
        self.assertEqual(recovered["attempt"], 2)
        task_notifications.finish_agentmail(event["event_id"], "failed", environ=self.environ, now=200)
        self.assertEqual(task_notifications.delivery_state(event["event_id"], "agentmail", self.environ), "pending")

    def test_private_database_permissions_are_restricted(self) -> None:
        task_notifications.record_turn_event({
            "hook_event_name": "Stop",
            "session_id": "session-a",
            "turn_id": "turn-a",
            "stop_hook_active": False,
        }, environ=self.environ)
        self.assertEqual(self.database.stat().st_mode & 0o077, 0)
        self.assertEqual(self.database.parent.stat().st_mode & 0o077, 0)


if __name__ == "__main__":
    unittest.main()
