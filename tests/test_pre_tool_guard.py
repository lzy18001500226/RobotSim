from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.agent import pre_tool_guard


def event(tool_name: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}


def run_guard(payload: dict[str, object]) -> dict[str, object] | None:
    output = io.StringIO()
    with patch("sys.stdin", io.StringIO(json.dumps(payload))), redirect_stdout(output):
        code = pre_tool_guard.main()
    if code != 0:
        raise AssertionError(f"guard returned {code}")
    return json.loads(output.getvalue()) if output.getvalue() else None


class PreToolGuardTests(unittest.TestCase):
    def test_blocks_force_push_hard_reset_and_forced_clean(self) -> None:
        for command in (
            "git push origin main --force-with-lease",
            "git reset --hard HEAD~1",
            "git clean -fdx",
        ):
            with self.subTest(command=command):
                result = run_guard(event("Bash", {"command": command}))
                self.assertEqual(result["decision"], "block")

    def test_blocks_obvious_secret_file_staging_and_secret_redirection(self) -> None:
        for command in (
            "git add .env.local",
            "printf '%s' \"$ROBOTSIM_SMTP_PASSWORD\" > credentials.txt",
        ):
            with self.subTest(command=command):
                result = run_guard(event("exec_command", {"cmd": command}))
                self.assertEqual(result["decision"], "block")

    def test_write_hook_blocks_secret_files_but_allows_example_files(self) -> None:
        blocked = run_guard(event("Write", {"file_path": "/repo/.env"}))
        allowed = run_guard(event("Write", {"file_path": "/repo/.env.example"}))
        self.assertEqual(blocked["decision"], "block")
        self.assertIsNone(allowed)

    def test_write_hook_blocks_credential_shaped_literals_in_content(self) -> None:
        fake_token = "gh" + "p_" + "A" * 32
        result = run_guard(event("Write", {"file_path": "/repo/config.txt", "content": fake_token}))
        self.assertEqual(result["decision"], "block")

    def test_safe_repository_commands_are_not_blocked(self) -> None:
        for command in ("git status --short", "git push origin task-branch", "git clean -n"):
            with self.subTest(command=command):
                self.assertIsNone(run_guard(event("Bash", {"command": command})))

    def test_hook_json_is_project_local_and_limited_to_pre_tool_use(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / ".codex" / "hooks.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["hooks"]), {"PreToolUse"})
        matcher = config["hooks"]["PreToolUse"][0]
        self.assertRegex("Bash", matcher["matcher"])
        self.assertEqual(matcher["hooks"][0]["type"], "command")
        self.assertGreater(matcher["hooks"][0]["timeout"], 0)
        self.assertIn("scripts/agent/pre_tool_guard.py", matcher["hooks"][0]["command"])


if __name__ == "__main__":
    unittest.main()
