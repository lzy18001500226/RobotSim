from __future__ import annotations

import concurrent.futures
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agent import notify_task, agentmail_stop_hook, path_governance


class MilestoneNotificationTests(unittest.TestCase):
    def settings(self, directory):
        return {
            "AGENTMAIL_API_KEY": "synthetic-key",
            "AGENTMAIL_INBOX_ID": "sender@example.invalid",
            "ROBOTSIM_NOTIFY_TO": "recipient@example.invalid",
            "ROBOTSIM_NOTIFY_STATE_DIR": directory,
        }

    def event(self, **overrides):
        args = dict(
            task_id="issue-46-bottle-transfer",
            milestone_id="h220-rgbd-grasp-lift",
            summary="22 cm grasp/lift stage passed; transfer remains pending",
            evidence=["https://github.com/lzy18001500226/RobotSim/issues/46"],
        )
        args.update(overrides)
        return args

    def test_same_stage_from_two_workers_sends_once_without_git_closeout(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(notify_task, "_send") as send, patch.object(notify_task, "check_closeout") as gate:
            env = self.settings(directory)
            first = notify_task.notify_milestone(**self.event(), worker="Local02", environ=env)
            second = notify_task.notify_milestone(**self.event(), worker="Orchestrator", environ=env)
            self.assertEqual((first.state, second.state), ("sent", "duplicate"))
            send.assert_called_once()
            gate.assert_not_called()
            self.assertIn("recipient delivery is not yet confirmed", first.message)
            self.assertIn("does not close the overall task", first.body)

    def test_next_stage_sends_an_independent_notification(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(notify_task, "_send") as send:
            env = self.settings(directory)
            notify_task.notify_milestone(**self.event(), environ=env)
            second = notify_task.notify_milestone(**self.event(milestone_id="h220-second-table-release"), environ=env)
            self.assertEqual(second.state, "sent")
            self.assertEqual(send.call_count, 2)

    def test_concurrent_orchestrators_cannot_double_send(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(notify_task, "_send") as send:
            env = self.settings(directory)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(lambda _: notify_task.notify_milestone(**self.event(), environ=env), range(2)))
            self.assertEqual(sorted(x.state for x in outcomes), ["duplicate", "sent"])
            send.assert_called_once()

    def test_ordinary_stop_is_silent_and_cannot_emit_a_milestone(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(notify_task, "_send") as send:
            env = self.settings(directory)
            env["ROBOTSIM_TASK_NOTIFICATION_DB"] = str(Path(directory) / "events.sqlite3")
            env["ROBOTSIM_LOCAL_STOP_HOOK"] = "1"
            for message in ["hello", "22 cm grasp succeeded", "milestone: passed"]:
                result = notify_task.process_stop_payload({
                    "hook_event_name": "Stop", "last_assistant_message": message,
                }, environ=env)
                self.assertEqual(result.state, "skipped")
            send.assert_not_called()

    def test_dry_run_requires_no_credentials_and_never_sends(self):
        with patch.object(agentmail_stop_hook, "_read_bundle") as bridge, patch.object(notify_task, "_send") as send:
            result = notify_task.notify_milestone(**self.event(), dry_run=True, environ={})
            self.assertEqual(result.state, "dry_run")
            bridge.assert_not_called()
            send.assert_not_called()

    def test_missing_invalid_and_unsafe_evidence_fail_before_delivery(self):
        for changes in [
            {"evidence": []},
            {"evidence": ["/does-not-exist/result.json"]},
            {"evidence": ["https://127.0.0.1/proof"]},
            {"milestone_id": "turn 1"},
            {"task_id": "random-chat"},
        ]:
            with self.subTest(changes=changes), patch.object(notify_task, "_send") as send:
                result = notify_task.notify_milestone(**self.event(**changes), environ={})
                self.assertEqual(result.exit_code, 2)
                send.assert_not_called()

    def test_local_proof_is_referenced_without_uploading_its_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "result.json"
            proof.write_text('{"PASS": true, "private_trace": "never upload this"}')
            with patch.object(path_governance, "describe_output_path", return_value={"WSL_PATH": "/governed/result.json"}), patch.object(notify_task, "_send") as send:
                result = notify_task.notify_milestone(**self.event(evidence=[str(proof)]), environ=self.settings(directory))
            self.assertEqual(result.state, "sent")
            self.assertIn("/governed/result.json", result.body)
            self.assertNotIn("never upload this", result.body)
            send.assert_called_once()

    def test_dpapi_fallback_loads_existing_credentials_for_explicit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            bundle = {k: settings[k] for k in notify_task.REQUIRED_SETTINGS}
            with patch.object(agentmail_stop_hook, "_read_bundle", return_value=bundle) as bridge, patch.object(notify_task, "_send"):
                result = notify_task.notify_milestone(**self.event(), environ={"ROBOTSIM_NOTIFY_STATE_DIR": directory})
            self.assertEqual(result.state, "sent")
            bridge.assert_called_once()

    def test_cli_missing_evidence_has_visible_failure_status(self):
        with contextlib.redirect_stderr(io.StringIO()):
            rc = notify_task.main([
                "milestone", "--task-id", "issue-46-bottle-transfer",
                "--milestone-id", "h220-grasp", "--summary", "synthetic", "--dry-run",
            ])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
