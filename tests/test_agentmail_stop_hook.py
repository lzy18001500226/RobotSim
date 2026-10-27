from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.agent import agentmail_stop_hook


SYNTHETIC_BUNDLE = {
    "AGENTMAIL_API_KEY": "synthetic-agentmail-key-not-real",
    "AGENTMAIL_INBOX_ID": "synthetic-inbox@example.invalid",
    "ROBOTSIM_NOTIFY_TO": "synthetic-recipient@example.invalid",
}


def stop_payload(*, closeout: bool = True) -> str:
    message = "ordinary synthetic stop"
    if closeout:
        event = {
            "schema_version": "robotsim.task-closeout.v1",
            "event_id": "auto",
            "repository": "lzy18001500226/RobotSim",
            "task_id": "issue-1-synthetic-notifier-test",
            "attempt_id": "00000000-0000-4000-8000-000000000001",
            "worker": "Codex",
            "task_kind": "implementation",
            "status": "completed",
            "summary": "Synthetic closeout for notifier tests.",
            "branch": "codex/notifier-test",
            "head_sha": "a" * 40,
            "pr_number": None,
            "validation": [],
            "evidence": [],
            "blockers": [],
            "recommended_next_action": "Review the synthetic test.",
            "completed_at": "2026-10-11T00:00:00Z",
        }
        message = (
            "<!-- robotsim.task-closeout.v1\n"
            + json.dumps(event)
            + "\n-->"
        )
    return json.dumps({
        "hook_event_name": "Stop",
        "session_id": "synthetic-session",
        "turn_id": "synthetic-turn",
        "stop_hook_active": False,
        "last_assistant_message": message,
    })


class AgentMailStopHookTests(unittest.TestCase):
    def test_bounded_process_budget_fits_stop_hook_timeout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / ".codex/hooks.json").read_text(encoding="utf-8"))
        stop_timeout = config["hooks"]["Stop"][0]["hooks"][0]["timeout"]
        process_budget = (
            agentmail_stop_hook.PATH_CONVERSION_TIMEOUT_SECONDS
            + agentmail_stop_hook.CREDENTIAL_BRIDGE_TIMEOUT_SECONDS
            + agentmail_stop_hook.NOTIFIER_TIMEOUT_SECONDS
        )
        self.assertLess(process_budget, stop_timeout)

    def test_disabled_stop_does_not_load_credentials_or_enable_notifications(self) -> None:
        with patch.object(agentmail_stop_hook, "_read_bundle") as load, patch.object(
            agentmail_stop_hook.subprocess, "run"
        ) as run, patch("sys.stdin", io.StringIO(stop_payload())):
            result = agentmail_stop_hook.main([], environ={"ROBOTSIM_LOCAL_STOP_HOOK": ""})
        self.assertEqual(result, 0)
        load.assert_not_called()
        self.assertNotIn("AGENTMAIL_API_KEY", run.call_args.kwargs["env"])

    def test_enabled_stop_loads_local_values_into_child_environment_only(self) -> None:
        with patch.object(agentmail_stop_hook, "_read_bundle", return_value=SYNTHETIC_BUNDLE) as load, patch.object(
            agentmail_stop_hook.subprocess, "run"
        ) as run, patch("sys.stdin", io.StringIO(stop_payload())):
            result = agentmail_stop_hook.main([], environ={"ROBOTSIM_LOCAL_STOP_HOOK": "1"})
        self.assertEqual(result, 0)
        load.assert_called_once_with("Emit")
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env["AGENTMAIL_API_KEY"], SYNTHETIC_BUNDLE["AGENTMAIL_API_KEY"])
        self.assertEqual(child_env["AGENTMAIL_INBOX_ID"], SYNTHETIC_BUNDLE["AGENTMAIL_INBOX_ID"])
        self.assertEqual(child_env["ROBOTSIM_NOTIFY_TO"], SYNTHETIC_BUNDLE["ROBOTSIM_NOTIFY_TO"])
        self.assertNotIn(SYNTHETIC_BUNDLE["AGENTMAIL_API_KEY"], run.call_args.args[0])

    def test_bridge_failure_is_sanitized_and_stop_remains_nonblocking(self) -> None:
        output = io.StringIO()
        with patch.object(
            agentmail_stop_hook,
            "_read_bundle",
            side_effect=agentmail_stop_hook.CredentialBridgeError("synthetic-secret-must-not-appear"),
        ), patch.object(agentmail_stop_hook.subprocess, "run") as run, patch(
            "sys.stdin", io.StringIO(stop_payload())
        ), contextlib.redirect_stderr(output):
            result = agentmail_stop_hook.main([], environ={"ROBOTSIM_LOCAL_STOP_HOOK": "1"})
        self.assertEqual(result, 0)
        self.assertNotIn("synthetic-secret", output.getvalue())
        self.assertIn("email delivery skipped", output.getvalue())
        self.assertNotIn("AGENTMAIL_API_KEY", run.call_args.kwargs["env"])

    def test_ordinary_stop_does_not_load_credentials_even_when_opted_in(self) -> None:
        with patch.object(agentmail_stop_hook, "_read_bundle") as load, patch.object(
            agentmail_stop_hook.subprocess, "run"
        ) as run, patch("sys.stdin", io.StringIO(stop_payload(closeout=False))):
            result = agentmail_stop_hook.main([], environ={"ROBOTSIM_LOCAL_STOP_HOOK": "1"})
        self.assertEqual(result, 0)
        load.assert_not_called()
        self.assertNotIn("AGENTMAIL_API_KEY", run.call_args.kwargs["env"])

    def test_malformed_and_recursive_closeouts_do_not_load_credentials(self) -> None:
        recursive = json.loads(stop_payload())
        recursive["stop_hook_active"] = True
        malformed = json.loads(stop_payload())
        malformed["last_assistant_message"] = "<!-- robotsim.task-closeout.v1\n{oops}\n-->"
        for payload in (recursive, malformed):
            with self.subTest(payload=payload["stop_hook_active"]), patch.object(
                agentmail_stop_hook, "_read_bundle"
            ) as load, patch.object(agentmail_stop_hook.subprocess, "run"), patch(
                "sys.stdin", io.StringIO(json.dumps(payload))
            ):
                self.assertEqual(
                    agentmail_stop_hook.main([], environ={"ROBOTSIM_LOCAL_STOP_HOOK": "1"}),
                    0,
                )
                load.assert_not_called()

    def test_presence_check_never_prints_secret_values(self) -> None:
        output = io.StringIO()
        with patch.object(agentmail_stop_hook, "_read_bundle", return_value=SYNTHETIC_BUNDLE), contextlib.redirect_stdout(output):
            result = agentmail_stop_hook.main(["--check-config"], environ={})
        self.assertEqual(result, 0)
        text = output.getvalue()
        self.assertIn("AGENTMAIL_API_KEY=present", text)
        self.assertIn("AGENTMAIL_INBOX_ID=present", text)
        self.assertIn("ROBOTSIM_NOTIFY_TO=present", text)
        self.assertIn("ROBOTSIM_LOCAL_STOP_HOOK=disabled", text)
        self.assertNotIn(SYNTHETIC_BUNDLE["AGENTMAIL_API_KEY"], text)

    def test_presence_check_reports_missing_configuration_without_detail(self) -> None:
        output = io.StringIO()
        with patch.object(
            agentmail_stop_hook,
            "_read_bundle",
            side_effect=agentmail_stop_hook.CredentialBridgeError("synthetic-secret-must-not-appear"),
        ), contextlib.redirect_stdout(output):
            result = agentmail_stop_hook.main(["--check-config"], environ={})
        self.assertEqual(result, 0)
        self.assertIn("CONFIG_READY=no", output.getvalue())
        self.assertNotIn("synthetic-secret", output.getvalue())

    def test_synthetic_bridge_self_test_uses_only_presence_output(self) -> None:
        output = io.StringIO()
        with patch.object(agentmail_stop_hook, "_read_bundle", return_value=SYNTHETIC_BUNDLE), contextlib.redirect_stdout(output):
            result = agentmail_stop_hook.main(["--self-test-bridge"], environ={})
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().strip(), "DPAPI_WSL_BRIDGE=PASS")
        self.assertNotIn(SYNTHETIC_BUNDLE["AGENTMAIL_API_KEY"], output.getvalue())


if __name__ == "__main__":
    unittest.main()
