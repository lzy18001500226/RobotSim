from __future__ import annotations

import io
import json
import re
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
                result = run_guard(event("Bash", {"command": command}))
                self.assertEqual(result["decision"], "block")

    def test_apply_patch_hook_blocks_secret_files_but_allows_example_files(self) -> None:
        blocked_patch = "*** Begin Patch\n*** Add File: .env\n+TOKEN=synthetic\n*** End Patch"
        allowed_patch = "*** Begin Patch\n*** Add File: .env.example\n+TOKEN=placeholder\n*** End Patch"
        blocked = run_guard(event("apply_patch", {"command": blocked_patch}))
        allowed = run_guard(event("apply_patch", {"command": allowed_patch}))
        self.assertEqual(blocked["decision"], "block")
        self.assertIsNone(allowed)

    def test_apply_patch_hook_blocks_credential_shaped_literals_in_command(self) -> None:
        fake_token = "gh" + "p_" + "A" * 32
        patch = f"*** Begin Patch\n*** Add File: config.txt\n+{fake_token}\n*** End Patch"
        result = run_guard(event("apply_patch", {"command": patch}))
        self.assertEqual(result["decision"], "block")

    def test_safe_repository_commands_are_not_blocked(self) -> None:
        for command in ("git status --short", "git push origin task-branch", "git clean -n"):
            with self.subTest(command=command):
                self.assertIsNone(run_guard(event("Bash", {"command": command})))

    def test_hook_config_wires_guard_for_canonical_codex_tools(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / ".codex" / "hooks.json").read_text(encoding="utf-8"))
        self.assertIn("PreToolUse", config.get("hooks", {}))
        tool_names = {"Bash", "apply_patch"}
        guard_path = "scripts/agent/pre_tool_guard.py"
        covered = set()

        for group in config["hooks"]["PreToolUse"]:
            matcher_text = group.get("matcher")
            if not isinstance(matcher_text, str):
                continue
            matcher = re.compile(matcher_text)
            for handler in group["hooks"]:
                if handler.get("type") != "command":
                    continue
                command = handler.get("command", "")
                if guard_path not in command:
                    continue
                self.assertGreater(handler.get("timeout", 0), 0)
                self.assertTrue((root / guard_path).is_file())
                covered.update(name for name in tool_names if matcher.search(name))

        self.assertEqual(covered, tool_names)


if __name__ == "__main__":
    unittest.main()
