from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.agent.closeout_gate import check_closeout


class CloseoutGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.repo = root / "repo"
        self.remote = root / "origin.git"
        self.remote.mkdir()
        self._run(["git", "init", "--bare", str(self.remote)])
        self._run(["git", "init", "-b", "codex/closeout-fixture", str(self.repo)])
        self._git("config", "user.name", "Closeout Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        (self.repo / "README.md").write_text("fixture\n", encoding="utf-8")
        self._git("add", "README.md")
        self._git("commit", "-m", "fixture base")
        self._git("remote", "add", "origin", str(self.remote))
        self._git("push", "-u", "origin", "HEAD")

    def _run(self, command: list[str]) -> str:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            self.fail(f"fixture command failed: {command[0]} ({result.returncode})")
        return result.stdout.strip()

    def _git(self, *args: str) -> str:
        return self._run(["git", "-C", str(self.repo), *args])

    def _check(self, *, task_id: str = "issue-900-closeout-fixture", pr_number: int | None = 17) -> dict[str, object]:
        return check_closeout(
            self.repo,
            task_id=task_id,
            branch="codex/closeout-fixture",
            head_sha=self._git("rev-parse", "HEAD"),
            pr_number=pr_number,
        )

    def test_a_uncommitted_intended_change_blocks_closeout(self) -> None:
        (self.repo / "src-change.txt").write_text("intended task work\n", encoding="utf-8")
        report = self._check()
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "CLOSEOUT BLOCKED")
        self.assertIn("working tree has uncommitted task changes", report["blockers"])
        self.assertEqual(report["dirty_paths"], ["src-change.txt"])

    def test_b_committed_but_unpushed_sha_blocks_closeout(self) -> None:
        (self.repo / "src-change.txt").write_text("committed but local only\n", encoding="utf-8")
        self._git("add", "src-change.txt")
        self._git("commit", "-m", "unpublished task change")
        self._git("push", "origin", "--delete", "codex/closeout-fixture")
        report = self._check()
        self.assertFalse(report["ok"])
        self.assertIn("task branch is not present on the push remote", report["blockers"])
        self.assertEqual(report["remote_head"], "")

    def test_c_remote_sha_different_from_local_head_blocks_closeout(self) -> None:
        old_remote_head = self._git("rev-parse", "HEAD")
        (self.repo / "src-change.txt").write_text("new local head\n", encoding="utf-8")
        self._git("add", "src-change.txt")
        self._git("commit", "-m", "new local head")
        report = self._check()
        self.assertFalse(report["ok"])
        self.assertEqual(report["remote_head"], old_remote_head)
        self.assertIn("remote branch SHA differs from local HEAD", report["blockers"])

    def test_d_clean_exactly_pushed_head_and_reference_pass(self) -> None:
        report = self._check()
        self.assertTrue(report["ok"], report["blockers"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["local_head"], report["remote_head"])
        self.assertEqual(
            report["reference_url"],
            "https://github.com/lzy18001500226/RobotSim/pull/17",
        )
        self.assertEqual(report["dirty_paths"], [])
        issue_report = self._check(pr_number=None)
        self.assertTrue(issue_report["ok"], issue_report["blockers"])
        self.assertEqual(
            issue_report["reference_url"],
            "https://github.com/lzy18001500226/RobotSim/issues/900",
        )
        self.assertNotIn(str(self.remote), str(report))

    def test_remote_check_uses_configured_push_url_not_fetch_url(self) -> None:
        fetch_remote = Path(self.temp.name) / "fetch-only.git"
        self._run(["git", "init", "--bare", str(fetch_remote)])
        self._git("remote", "set-url", "origin", str(fetch_remote))
        self._git("remote", "set-url", "--push", "origin", str(self.remote))
        report = self._check()
        self.assertTrue(report["ok"], report["blockers"])
        self.assertEqual(report["local_head"], report["remote_head"])

    def test_inline_push_url_credentials_are_rejected_without_disclosure(self) -> None:
        credential_url = "https://user:fixture-secret@example.invalid/repo.git?token=fixture-query-secret"
        self._git("remote", "set-url", "--push", "origin", credential_url)
        report = self._check()
        self.assertFalse(report["ok"])
        self.assertIn(
            "origin push URL contains inline credentials; use SSH or an OS credential helper",
            report["blockers"],
        )
        self.assertNotIn("fixture-secret", str(report))
        self.assertNotIn("fixture-query-secret", str(report))
        self.assertNotIn(credential_url, str(report))

    def test_missing_issue_reference_blocks_closeout(self) -> None:
        report = self._check(task_id="task-without-issue", pr_number=None)
        self.assertFalse(report["ok"])
        self.assertIn("a corresponding GitHub Issue or PR reference is required", report["blockers"])

    def test_documented_unrelated_subtree_is_the_only_dirty_exception(self) -> None:
        unrelated = self.repo / "preexisting" / "sentinel.txt"
        unrelated.parent.mkdir()
        unrelated.write_text("preserve me\n", encoding="utf-8")
        report = check_closeout(
            self.repo,
            task_id="issue-900-closeout-fixture",
            branch="codex/closeout-fixture",
            head_sha=self._git("rev-parse", "HEAD"),
            pr_number=17,
            documented_unrelated=(
                "Unrelated repository path: preexisting/ | reason: pre-existing user fixture, untouched",
            ),
        )
        self.assertTrue(report["ok"], report["blockers"])
        self.assertEqual(report["dirty_paths"], [])
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "preserve me\n")


if __name__ == "__main__":
    unittest.main()
