from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.agent import filesystem_guard as guard, pre_tool_guard


class FilesystemGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.patcher = patch.object(guard, "REPOSITORY", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_blocks_system_drive_and_outside_deletion(self) -> None:
        for command in (
            "rm -rf /", "rm -rf /home", "rm /etc/passwd", "rm -rf ~/projects",
            "Remove-Item -LiteralPath 'C:\\Windows' -Recurse -Force",
            'cmd.exe /c "rd /s /q C:\\"', "rm -rf /mnt/c/Users/HP",
            "wsl.exe --exec rm -rf /home", "sudo rm -rf /usr",
        ):
            with self.subTest(command=command):
                self.assertIsNotNone(guard.command_reason(command, str(self.root)))

    def test_rejects_opaque_targets_and_protected_evidence(self) -> None:
        for target in (".", "..", "reviews/old", "reference/vendor", ".git", "third_party/vendor", "$target", "tmp/*", "'~/cache'"):
            with self.subTest(target=target):
                self.assertIsNotNone(guard.command_reason(f"rm -rf {target}", str(self.root)))

    def test_allows_literal_scoped_paths_but_does_not_execute(self) -> None:
        target = self.root / "generated temp"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("preserved")
        self.assertIsNone(guard.command_reason(f"rm -rf '{target}'"))
        self.assertIsNone(guard.command_reason("rm -rf generated", str(self.root)))
        self.assertEqual(sentinel.read_text(), "preserved")

    def test_blocks_symlink_escape(self) -> None:
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        self.assertIsNotNone(guard.command_reason("rm -rf escape/outside", str(self.root)))

    def test_relative_target_requires_known_cwd_and_cd_invalidates_it(self) -> None:
        for command, cwd in (("rm -rf generated", None), ("cd /; rm -rf generated", str(self.root)), ("rm -rf generated", "C:\\Users\\HP")):
            self.assertIsNotNone(guard.command_reason(command, cwd))

    def test_shell_wrappers_newlines_and_pipelines(self) -> None:
        for command in ("bash -c 'rm -rf /etc'", "bash -lc 'rm -rf /etc'", 'powershell.exe -Command "Remove-Item C:\\ -Recurse"', "echo harmless\nrm -rf /etc", "echo hi | rm -rf /etc", "env FLAG=1 bash -c 'rm -rf /etc'"):
            with self.subTest(command=command):
                self.assertIsNotNone(guard.command_reason(command))

    def test_directory_changing_wrappers_invalidate_relative_scope(self) -> None:
        wrappers = (
            "wsl.exe -d Ubuntu-22.04 --cd /tmp --exec",
            "wsl.exe --cd=/tmp --exec",
            "env -C /tmp", "env -C/tmp", "env --chdir=/tmp",
            "sudo -D /tmp", "sudo -D/tmp", "sudo --chdir=/tmp",
            "env -iC /tmp", "env --chd=/tmp", "sudo -HD /tmp",
            "sudo --chd=/tmp", "sudo -i", "sudo --login",
            "command env -C /tmp sudo",
        )
        for wrapper in wrappers:
            with self.subTest(wrapper=wrapper):
                self.assertIsNotNone(guard.command_reason(f"{wrapper} rm -rf generated", str(self.root)))
                self.assertIsNone(guard.command_reason(f"{wrapper} rm -rf '{self.root}/generated'", str(self.root)))
        for wrapper in ("sudo", "env FLAG=1", "command", "wsl.exe --exec"):
            self.assertIsNone(guard.command_reason(f"{wrapper} rm -rf generated", str(self.root)))

    def test_escaped_and_concatenated_protected_operands_fail_closed(self) -> None:
        for target in (r"revie\ws", 're"views"', './re""views', '"re"views', "'re'views", r"revi\ews/old", "'.g'it", "ref'erence'/vendor"):
            with self.subTest(target=target):
                self.assertIsNotNone(guard.command_reason(f"rm -rf {target}", str(self.root)))
                self.assertIsNotNone(guard.command_reason(f"bash -c 'rm -rf {target}'", str(self.root)))
        self.assertIsNone(guard.command_reason('rm -rf gen"erated"', str(self.root)))
        self.assertIsNone(guard.command_reason('bash -c "rm -rf \'generated temp\'"', str(self.root)))

    def test_escaped_supported_executable_names_remain_inspected(self) -> None:
        for executable in (r"r\m", r"/usr/bin/r\m", 'r"m"'):
            for target in ("reviews", "/tmp"):
                with self.subTest(executable=executable, target=target):
                    self.assertIsNotNone(guard.command_reason(f"{executable} -rf {target}", str(self.root)))
        self.assertIsNone(guard.command_reason(r"r\m -rf generated", str(self.root)))

    def test_hash_in_filename_cannot_hide_subsequent_commands(self) -> None:
        self.assertIsNotNone(guard.command_reason("rm -rf generated#; rm -rf reviews", str(self.root)))
        self.assertIsNone(guard.command_reason("rm -rf generated#", str(self.root)))
        self.assertIsNone(guard.command_reason("echo '#; rm -rf reviews'", str(self.root)))

    def test_blocks_disk_commands(self) -> None:
        for command in ("mkfs.ext4 /dev/sda", "diskpart.exe /s commands.txt", "format.com C:", "dd if=image of=/dev/sda", "sudo wipefs -a /dev/sda"):
            with self.subTest(command=command):
                self.assertIsNotNone(guard.command_reason(command))

    def test_quoted_code_or_documentation_is_not_executed_by_guard(self) -> None:
        for command in ("python3 -c 'print(\"rm -rf /\")'", "echo 'mkfs.ext4 /dev/sda'", "git status --short", "dd if=input of=output"):
            with self.subTest(command=command):
                self.assertIsNone(guard.command_reason(command))

    def test_raw_patch_and_malformed_event_return_deny_schema(self) -> None:
        payloads = ("not json", json.dumps({"hook_event_name": "PreToolUse", "tool_name": "apply_patch", "tool_input": "*** Begin Patch\n*** Add File: .env\n+x\n*** End Patch"}))
        for payload in payloads:
            output = io.StringIO()
            with patch("sys.stdin", io.StringIO(payload)), redirect_stdout(output):
                self.assertEqual(pre_tool_guard.main(), 0)
            result = json.loads(output.getvalue())["hookSpecificOutput"]
            self.assertEqual(result["hookEventName"], "PreToolUse")
            self.assertEqual(result["permissionDecision"], "deny")

    def test_hook_dispatches_deletion_and_allows_readonly(self) -> None:
        self.assertIsNotNone(pre_tool_guard._reason({"tool_name": "exec_command", "tool_input": {"cmd": "rm -rf /home", "workdir": str(self.root)}}))
        self.assertIsNone(pre_tool_guard._reason({"tool_name": "exec_command", "tool_input": {"cmd": "git status --short"}}))


if __name__ == "__main__":
    unittest.main()
