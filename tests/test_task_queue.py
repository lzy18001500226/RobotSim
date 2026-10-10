"""Deterministic tests for the single-host Issue queue pilot."""

from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.agent import task_queue as queue


def make_issue(
    number: int = 49,
    *,
    body: str = "",
    labels: tuple[str, ...] = ("agent:ready",),
    state: str = "OPEN",
) -> queue.Issue:
    url = f"https://github.com/lzy18001500226/RobotSim/issues/{number}"
    return queue.Issue(number, f"Task {number}", body, state, frozenset(labels), url)


def make_pull() -> queue.PullRequest:
    return queue.PullRequest(
        50, "https://github.com/lzy18001500226/RobotSim/pull/50",
        "issue/49-task", "a" * 40, "OPEN", "success", "approved",
    )


class TaskQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_codex_commands_preserve_only_the_discovered_model_route(self) -> None:
        model_config = queue.worker_security.CodexModelConfig(
            "gpt-5.5", "OpenAI", "https://models.example.invalid/v1", "responses",
        )
        command = queue.codex_command(
            self.root, workspace_write_roots=(), model_config=model_config,
        )
        resume = queue.codex_resume_command(
            "session-49", self.root / "result.json", workspace=self.root,
            workspace_write_roots=(), model_config=model_config,
        )
        self.assertNotIn("--cd", resume)
        self.assertNotIn("--add-dir", resume)
        self.assertLess(resume.index("--strict-config"), resume.index("session-49"))
        self.assertEqual(resume[-2:], ["session-49", "-"])

        for candidate in (command, resume):
            self.assertEqual(candidate[candidate.index("--model") + 1], "gpt-5.5")
            overrides = [
                candidate[index + 1]
                for index, value in enumerate(candidate[:-1])
                if value == "--config"
            ]
            self.assertIn('model_provider="OpenAI"', overrides)
            self.assertIn('model_providers.OpenAI.base_url="https://models.example.invalid/v1"', overrides)
            self.assertIn('model_providers.OpenAI.wire_api="responses"', overrides)
            self.assertIn("model_providers.OpenAI.requires_openai_auth=true", overrides)
            self.assertIn("permissions.robotsim_worker.network.enabled=false", overrides)

        baseline = queue.codex_command(self.root, workspace_write_roots=())
        self.assertNotIn("--model", baseline)

    def store(self, name: str = "state.sqlite3") -> queue.RunStore:
        store = queue.RunStore(self.root / name)
        self.addCleanup(store.close)
        return store

    def claim(self, store: queue.RunStore, task: queue.Issue) -> dict[str, object]:
        row = store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(row)
        assert row is not None
        return row

    def create_task_workspace(self, task: queue.Issue) -> tuple[Path, str]:
        repository = self.root / "queue-repository"
        if not repository.exists():
            repository.mkdir()
            commands = (
                ["git", "init", "--initial-branch=main", str(repository)],
                ["git", "-C", str(repository), "config", "user.name", "RobotSim Test"],
                ["git", "-C", str(repository), "config", "user.email", "robotsim-test@example.invalid"],
            )
            for command in commands:
                subprocess.run(command, check=True, capture_output=True, text=True)
            (repository / "README.md").write_text("queue fixture\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-m", "queue fixture"],
                check=True, capture_output=True, text=True,
            )
        self.queue_repository = repository
        plan = queue.workspace_plan(task, self.root / "worktrees")
        workspace = Path(plan.path)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "-b", plan.branch, str(workspace), "HEAD"],
            check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return workspace, result.stdout.strip()

    def test_ready_issue_is_dispatchable_with_deterministic_workspace(self) -> None:
        task = make_issue()
        decision = queue.dispatch_decision(task, {})
        first = queue.workspace_plan(task, self.root / "worktrees")
        second = queue.workspace_plan(task, self.root / "worktrees")
        renamed = queue.workspace_plan(
            queue.Issue(task.number, "Retitled after triage", task.body, task.state, task.labels),
            self.root / "worktrees",
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(first, second)
        self.assertEqual(first, renamed)
        self.assertEqual(first.branch, "issue/49-task")
        self.assertTrue(first.path.endswith("/issue-49"))

    def test_review_checkout_matches_checks_branch_head_and_clean_state(self) -> None:
        task = make_issue()
        workspace, head = self.create_task_workspace(task)
        self.assertTrue(queue.review_checkout_matches(workspace, "issue/49-task", head))
        self.assertFalse(queue.review_checkout_matches(workspace, "codex/issue-61", head))
        self.assertFalse(queue.review_checkout_matches(workspace, "issue/49-task", "b" * 40))
        (workspace / "README.md").write_text("dirty task fixture\n", encoding="utf-8")
        self.assertFalse(queue.review_checkout_matches(workspace, "issue/49-task", head))

    def test_closeout_rejects_cross_issue_and_infra_branch_identities(self) -> None:
        store = self.store()
        issue49 = make_issue(49)
        row49 = self.claim(store, issue49)
        with self.assertRaisesRegex(ValueError, "different Issue"):
            queue.closeout_event(make_issue(62), row49, status="blocked", summary="Mismatch")
        wrong_branch = dict(row49, branch="codex/issue-61-orchestration-pilot")
        with self.assertRaisesRegex(ValueError, "branch does not match"):
            queue.closeout_event(issue49, wrong_branch, status="blocked", summary="Mismatch")

    def test_simultaneous_issue_runs_keep_distinct_closeout_identity(self) -> None:
        store = self.store()
        issue49 = make_issue(49)
        issue62 = make_issue(62)
        run49 = self.claim(store, issue49)
        run62 = store.claim(issue62, queue.workspace_plan(issue62, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(run62)
        assert run62 is not None
        event49 = queue.closeout_event(
            issue49, run49, status="blocked", summary="Issue 49 stopped."
        )
        event62 = queue.closeout_event(
            issue62, run62, status="blocked", summary="Issue 62 stopped."
        )
        self.assertEqual(event49["task_id"], "issue-49-symphony-task")
        self.assertEqual(event49["branch"], "issue/49-task")
        self.assertEqual(event62["task_id"], "issue-62-symphony-task")
        self.assertEqual(event62["branch"], "issue/62-task")
        self.assertNotEqual(event49["attempt_id"], event62["attempt_id"])

    def test_closeout_rejects_unrelated_infrastructure_worktree(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        _valid_workspace, _valid_head = self.create_task_workspace(task)
        outside = self.root / "robotsim-infra" / "issue-49"
        outside.mkdir(parents=True)
        subprocess.run(
            ["git", "init", "--initial-branch", "issue/49-task", str(outside)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["git", "-C", str(outside), "config", "user.name", "RobotSim Test"], check=True)
        subprocess.run(["git", "-C", str(outside), "config", "user.email", "robotsim-test@example.invalid"], check=True)
        (outside / "README.md").write_text("unrelated infrastructure tree\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(outside), "add", "README.md"], check=True)
        subprocess.run(["git", "-C", str(outside), "commit", "-m", "unrelated"], check=True, capture_output=True, text=True)
        calls: list[dict[str, object]] = []

        def notifier(_root: Path, event: object, _workspace: Path) -> int:
            calls.append(dict(event))
            return 0

        pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, store,
            object(), object(), notifier=notifier,
        )
        self.assertEqual(
            pilot._emit_closeout(
                task, dict(row, workspace=outside.as_posix()),
                status="blocked", summary="Must not bind to an infra worktree.",
            ),
            1,
        )
        self.assertEqual(calls, [])
        persisted = store.get(task.number)
        self.assertEqual(persisted["closeout_event_json"], "")
        self.assertEqual(persisted["notification_state"], "identity_rejected")

    def test_closeout_requires_exact_remote_head_and_pr_association(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        workspace, head = self.create_task_workspace(task)
        row = store.update(task.number, head_sha=head, pr_number=50)
        pull = queue.PullRequest(
            50, "https://github.com/lzy18001500226/RobotSim/pull/50",
            "issue/49-task", head, "OPEN", "success", "approved",
        )
        event = queue.closeout_event(
            task, row, status="completed", summary="Ready for review.",
            pr_url=pull.url, head_sha=head, remote_head_sha=head, pull=pull,
        )
        self.assertEqual(event["head_sha"], head)
        self.assertIn(pull.url, event["evidence"])
        self.assertTrue(any(str(item).startswith("PR association verified") for item in event["validation"]))
        with self.assertRaisesRegex(ValueError, "verified PR association"):
            queue.closeout_event(
                task, dict(row, pr_number=None), status="completed", summary="No PR.",
                head_sha=head, remote_head_sha=head,
            )
        with self.assertRaisesRegex(ValueError, "remote HEAD"):
            queue.closeout_event(
                task, row, status="completed", summary="Mismatch.",
                head_sha=head, remote_head_sha="b" * 40, pull=pull, pr_url=pull.url,
            )
        wrong_pull = queue.PullRequest(
            51, "https://github.com/lzy18001500226/RobotSim/pull/51",
            "codex/issue-61-orchestration-pilot", head, "OPEN", "success", "approved",
        )
        with self.assertRaisesRegex(ValueError, "PR number, URL, branch, or head"):
            queue.closeout_event(
                task, row, status="completed", summary="Mismatch.",
                head_sha=head, remote_head_sha=head, pull=wrong_pull, pr_url=wrong_pull.url,
            )
        wrong_repository_pull = queue.PullRequest(
            50, pull.url, "issue/49-task", head, "OPEN", "success", "approved",
            head_repository="another-owner/RobotSim",
        )
        with self.assertRaisesRegex(ValueError, "PR number, URL, branch, or head"):
            queue.closeout_event(
                task, row, status="completed", summary="Wrong head repository.",
                head_sha=head, remote_head_sha=head, pull=wrong_repository_pull,
                pr_url=wrong_repository_pull.url,
            )
        wrong_base_pull = queue.PullRequest(
            50, pull.url, "issue/49-task", head, "OPEN", "success", "approved",
            base_branch="develop",
        )
        with self.assertRaisesRegex(ValueError, "PR number, URL, branch, or head"):
            queue.closeout_event(
                task, row, status="completed", summary="Wrong base branch.",
                head_sha=head, remote_head_sha=head, pull=wrong_base_pull,
                pr_url=wrong_base_pull.url,
            )
        fork_pull = queue.PullRequest(
            50, pull.url, "issue/49-task", head, "OPEN", "success", "approved",
            cross_repository=True,
        )
        with self.assertRaisesRegex(ValueError, "PR number, URL, branch, or head"):
            queue.closeout_event(
                task, row, status="completed", summary="Fork PR.",
                head_sha=head, remote_head_sha=head, pull=fork_pull, pr_url=fork_pull.url,
            )

    def test_reconcile_rejects_stale_pr_head_without_advancing_task_state(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        before = store.update(task.number, status="awaiting_pr")
        pull = make_pull()

        class Github:
            def issue(self, _number: int) -> queue.Issue:
                return task

            def pull_for_branch(self, _branch: str) -> queue.PullRequest:
                return pull

            def branch_sha(self, _root: Path, _branch: str) -> str:
                return "b" * 40

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store,
            Github(), object(), notifier=lambda _root, _event, _workspace: 0,
        )
        result = pilot.reconcile()
        after = store.get(task.number)
        self.assertEqual(result[0]["status"], "awaiting_pr")
        self.assertEqual(after["status"], before["status"])
        self.assertEqual(after["pr_number"], before["pr_number"])
        self.assertEqual(store.events(task.number)[-1]["event_type"], "pull_request_identity_rejected")

    def test_worktree_is_isolated_and_reusable_at_the_same_issue_path(self) -> None:
        repository = self.root / "repo"
        repository.mkdir()
        commands = (
            ["git", "init", "--initial-branch=main", str(repository)],
            ["git", "-C", str(repository), "config", "user.name", "RobotSim Test"],
            ["git", "-C", str(repository), "config", "user.email", "robotsim-test@example.invalid"],
        )
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)
        (repository / "README.md").write_text("baseline\n", encoding="utf-8")
        (repository / "AGENTS.md").write_text("trusted baseline guidance\n", encoding="utf-8")
        workflow = repository / "docs/workflows/goal_driven_development.md"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("trusted goal workflow\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repository), "commit", "-m", "baseline"], check=True, capture_output=True)
        remote = self.root / "origin.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repository), "remote", "add", "origin", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "push", "--set-upstream", "origin", "main"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "fetch", "--no-tags", "origin",
             "+refs/heads/main:refs/remotes/origin/main"],
            check=True, capture_output=True, text=True,
        )

        plan = queue.workspace_plan(make_issue(), self.root / "worktrees")
        workspace = queue.create_worktree(repository, plan)
        self.assertEqual(workspace, Path(plan.path))
        self.assertEqual(subprocess.check_output(["git", "-C", str(workspace), "branch", "--show-current"], text=True).strip(), plan.branch)
        writable_dirs = queue.codex_worktree_write_dirs(workspace)
        common_dir = Path(subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip())
        git_dir = Path(subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "--path-format=absolute", "--git-dir"],
            text=True,
        ).strip())
        self.assertEqual(writable_dirs, ())
        self.assertEqual(git_dir, common_dir)
        self.assertEqual(git_dir, workspace / ".git")
        self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
        self.assertFalse((git_dir / "objects/info/alternates").exists())
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(workspace), "config", "--local", "--get", "user.name"], text=True,
        ).strip(), "RobotSim Test")
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(workspace), "config", "--local", "--get", "user.email"], text=True,
        ).strip(), "robotsim-test@example.invalid")
        self.assertEqual(subprocess.check_output(["git", "-C", str(workspace), "remote"], text=True).strip(), "")
        self.assertNotEqual(subprocess.run(
            ["git", "-C", str(workspace), "config", "--local", "--get", "branch.issue/49-task.remote"],
            check=False, capture_output=True, text=True,
        ).returncode, 0)
        command = queue.codex_command(workspace)
        self.assertNotIn("--sandbox", command)
        self.assertNotIn("--danger-full-access", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--strict-config", command)
        self.assertIn("default_permissions=robotsim_worker", command)
        filesystem_override = next(
            value for value in command
            if value.startswith("permissions.robotsim_worker.filesystem=")
        )
        self.assertIn('"/root" = "read"', filesystem_override)
        self.assertIn('"/root/auth.json" = "deny"', filesystem_override)
        self.assertIn('"/root/.env" = "deny"', filesystem_override)
        self.assertIn('"/home/.git" = "write"', filesystem_override)
        self.assertNotIn('"/root" = "write"', filesystem_override)
        self.assertIn("permissions.robotsim_worker.network.enabled=false", command)
        self.assertIn("shell_environment_policy.inherit=none", command)
        add_dirs = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--add-dir"]
        self.assertNotIn("/root", add_dirs)
        self.assertNotIn("/var/tmp", add_dirs)
        self.assertIn("history.persistence=none", command)
        read_only = queue.codex_command(workspace, read_only=True)
        self.assertNotIn("--sandbox", read_only)
        review_filesystem = next(
            value for value in read_only
            if value.startswith("permissions.robotsim_worker.filesystem=")
        )
        self.assertIn('"/home" = "read"', review_filesystem)
        self.assertNotIn("--add-dir", read_only)
        resume = queue.codex_resume_command("session-49", self.root / "result.json", workspace=workspace)
        self.assertNotIn("--sandbox", resume)
        self.assertIn("--strict-config", resume)
        self.assertNotIn("--cd", resume)
        self.assertNotIn("--add-dir", resume)
        self.assertLess(resume.index("--strict-config"), resume.index("session-49"))
        baseline_head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "main"], text=True,
        ).strip()
        self.assertNotEqual(
            subprocess.run(
                ["git", "-C", str(repository), "show-ref", "--verify", "--quiet", f"refs/heads/{plan.branch}"],
                check=False,
            ).returncode,
            0,
        )
        (workspace / "worktree-edit-probe.txt").write_text("task worktree edit\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(workspace), "add", "worktree-edit-probe.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(workspace), "commit", "-m", "task worktree edit"],
            check=True, capture_output=True, text=True,
        )
        task_head = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
        trusted_base = baseline_head
        task_git = workspace / ".git"
        alternate = task_git / "objects/info/alternates"
        alternate.parent.mkdir(parents=True, exist_ok=True)
        alternate.write_text("/untrusted/object-store\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "alternate, grafted, or shallow"):
            queue._publish_validated_task_branch(
                repository, workspace, plan.issue_number, base_sha=trusted_base,
                require_canonical_remote=False,
            )
        alternate.unlink()
        subprocess.run(
            ["git", "-C", str(workspace), "update-ref", "refs/remotes/origin/main", task_head],
            check=True, capture_output=True, text=True,
        )
        published_head = queue._publish_validated_task_branch(
            repository, workspace, plan.issue_number, base_sha=trusted_base,
            require_canonical_remote=False,
        )
        pushed_head = subprocess.check_output(
            ["git", "--git-dir", str(remote), "rev-parse", f"refs/heads/{plan.branch}"], text=True,
        ).strip()
        task_head = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
        ).strip()
        self.assertEqual(published_head, task_head)
        self.assertEqual(pushed_head, task_head)
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "main"], text=True,
        ).strip(), baseline_head)
        self.assertEqual(queue.create_worktree(repository, plan), workspace)
        self.assertFalse(subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip())
        (workspace / "AGENTS.md").write_text("untrusted task-branch guidance\n", encoding="utf-8")
        guidance = queue.read_default_guidance(workspace)
        self.assertEqual(guidance["AGENTS.md"], "trusted baseline guidance\n")
        self.assertEqual(guidance["docs/workflows/goal_driven_development.md"], "trusted goal workflow\n")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            queue.create_worktree(repository, plan)
        self.assertEqual(queue.create_worktree(repository, plan, allow_dirty=True), workspace)
        self.assertEqual((workspace / "AGENTS.md").read_text(encoding="utf-8"), "untrusted task-branch guidance\n")

    def test_clean_linked_issue_checkout_migrates_and_preserves_ignored_files(self) -> None:
        repository = self.root / "legacy-repo"
        repository.mkdir()
        for command in (
            ["git", "init", "--initial-branch=main", str(repository)],
            ["git", "-C", str(repository), "config", "user.name", "RobotSim Test"],
            ["git", "-C", str(repository), "config", "user.email", "robotsim-test@example.invalid"],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True)
        (repository / ".gitignore").write_text("docs/research/\n", encoding="utf-8")
        (repository / "README.md").write_text("baseline\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "baseline"],
            check=True, capture_output=True, text=True,
        )
        remote = self.root / "legacy-origin.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repository), "remote", "add", "origin", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "push", "--set-upstream", "origin", "main"],
            check=True, capture_output=True, text=True,
        )
        task = make_issue(71)
        plan = queue.workspace_plan(task, self.root / "legacy-worktrees")
        legacy = Path(plan.path)
        legacy.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "-b", plan.branch, str(legacy), "origin/main"],
            check=True, capture_output=True, text=True,
        )
        expected_head = subprocess.check_output(["git", "-C", str(legacy), "rev-parse", "HEAD"], text=True).strip()
        ignored_file = legacy / "docs/research/evidence.txt"
        ignored_file.parent.mkdir(parents=True)
        ignored_file.write_text("retain this local evidence\n", encoding="utf-8")

        migrated = queue.create_worktree(repository, plan)

        self.assertEqual(migrated, legacy)
        self.assertEqual(subprocess.check_output(["git", "-C", str(migrated), "branch", "--show-current"], text=True).strip(), plan.branch)
        self.assertEqual(subprocess.check_output(["git", "-C", str(migrated), "rev-parse", "HEAD"], text=True).strip(), expected_head)
        self.assertEqual((migrated / "docs/research/evidence.txt").read_text(encoding="utf-8"), "retain this local evidence\n")
        self.assertEqual(queue.codex_worktree_write_dirs(migrated), ())
        self.assertEqual(Path(subprocess.check_output(
            ["git", "-C", str(migrated), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True,
        ).strip()), migrated / ".git")
        self.assertFalse((migrated / ".git/objects/info/alternates").exists())
        registered_worktrees = subprocess.check_output(
            ["git", "-C", str(repository), "worktree", "list", "--porcelain"], text=True,
        ).splitlines()
        self.assertNotIn(f"worktree {migrated}", registered_worktrees)
        self.assertEqual(subprocess.run(
            ["git", "-C", str(repository), "show-ref", "--verify", "--quiet", f"refs/heads/{plan.branch}"],
            check=False,
        ).returncode, 0)
        stage, marker = queue._migration_paths(migrated)
        self.assertFalse(stage.exists())
        self.assertFalse(marker.exists())

    def test_dirty_linked_issue_checkout_is_preserved_without_migration(self) -> None:
        repository = self.root / "dirty-legacy-repo"
        repository.mkdir()
        for command in (
            ["git", "init", "--initial-branch=main", str(repository)],
            ["git", "-C", str(repository), "config", "user.name", "RobotSim Test"],
            ["git", "-C", str(repository), "config", "user.email", "robotsim-test@example.invalid"],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True)
        (repository / "README.md").write_text("baseline\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "baseline"],
            check=True, capture_output=True, text=True,
        )
        remote = self.root / "dirty-legacy-origin.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repository), "remote", "add", "origin", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "push", "--set-upstream", "origin", "main"],
            check=True, capture_output=True, text=True,
        )
        conflict_plan = queue.workspace_plan(make_issue(73), self.root / "conflict-worktrees")
        conflict_path = self.root / "already-checked-out-73"
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "-b", conflict_plan.branch,
             str(conflict_path), "origin/main"],
            check=True, capture_output=True, text=True,
        )
        with self.assertRaisesRegex(RuntimeError, "already checked out"):
            queue.create_worktree(repository, conflict_plan)
        self.assertFalse(Path(conflict_plan.path).exists())

        plan = queue.workspace_plan(make_issue(72), self.root / "dirty-legacy-worktrees")
        legacy = Path(plan.path)
        legacy.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "-b", plan.branch, str(legacy), "origin/main"],
            check=True, capture_output=True, text=True,
        )
        edit = legacy / "uncommitted.txt"
        edit.write_text("do not discard\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            queue.create_worktree(repository, plan)
        self.assertTrue((legacy / ".git").is_file())
        self.assertEqual(edit.read_text(encoding="utf-8"), "do not discard\n")
        self.assertIn(f"worktree {legacy}", subprocess.check_output(
            ["git", "-C", str(repository), "worktree", "list", "--porcelain"], text=True,
        ).splitlines())

    def test_inline_remote_credentials_are_rejected_without_disclosure(self) -> None:
        repository = self.root / "credential-url-repo"
        repository.mkdir()
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(repository)],
            check=True, capture_output=True, text=True,
        )
        (repository / "README.md").write_text("baseline\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "-c", "user.name=RobotSim Test",
             "-c", "user.email=robotsim-test@example.invalid", "commit", "-m", "baseline"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "remote", "add", "origin", "https://github.com/example/RobotSim.git"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "remote", "set-url", "--push", "origin",
             "https://user:synthetic-secret@github.com/example/RobotSim.git"],
            check=True,
        )
        plan = queue.workspace_plan(make_issue(74), self.root / "credential-url-worktrees")
        with self.assertRaisesRegex(RuntimeError, "inline credentials") as raised:
            queue.create_worktree(repository, plan)
        self.assertNotIn("synthetic-secret", str(raised.exception))
        self.assertFalse(Path(plan.path).exists())

    def test_open_dependency_prevents_dispatch(self) -> None:
        task = make_issue(body="Depends on: #12, #13")
        dependencies = {12: make_issue(12), 13: make_issue(13, state="CLOSED")}
        decision = queue.dispatch_decision(task, dependencies)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.blocked_dependencies, (12,))

    def test_running_issue_label_prevents_a_second_primary(self) -> None:
        task = make_issue(labels=("agent:ready", "agent:running"))
        decision = queue.dispatch_decision(task, {})
        self.assertFalse(decision.eligible)
        self.assertIn("already has", decision.reason)

    def test_local_writer_slot_cap_is_enforced(self) -> None:
        store = self.store()
        self.claim(store, make_issue(1))
        self.claim(store, make_issue(2))
        third = store.claim(make_issue(3), queue.workspace_plan(make_issue(3), self.root / "worktrees"), writer_slots=2)
        self.assertEqual(store.active_count(), 2)
        self.assertIsNone(third)
        self.assertFalse(queue.dispatch_decision(make_issue(4), {}, writer_slots=3).eligible)

    def test_concurrent_store_clients_cannot_exceed_the_writer_cap(self) -> None:
        path = self.root / "shared.sqlite3"
        first, second = queue.RunStore(path), queue.RunStore(path)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        barrier = threading.Barrier(3)

        def claim(store: queue.RunStore, number: int) -> dict[str, object] | None:
            task = make_issue(number)
            barrier.wait()
            return store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(claim, first, 21), pool.submit(claim, second, 22)]
            barrier.wait()
            results = [future.result() for future in tasks]
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(first.active_count(), 1)

    def test_operational_paths_are_kept_outside_the_source_checkout(self) -> None:
        repository = self.root / "checkout"
        repository.mkdir()
        self.assertEqual(queue.outside_repository(self.root / "cache", repository), self.root / "cache")
        with self.assertRaises(ValueError):
            queue.outside_repository(repository / "state.sqlite3", repository)

    def test_runner_lock_rejects_a_second_watcher(self) -> None:
        lock_path = self.root / "state" / "runner.lock"
        with queue.RunnerLock(lock_path):
            with self.assertRaisesRegex(RuntimeError, "another task queue watcher"):
                with queue.RunnerLock(lock_path):
                    pass

    def test_repeated_root_cause_retries_once_then_blocks(self) -> None:
        store = self.store()
        task = make_issue()
        first = self.claim(store, task)
        store.set_codex_session(task.number, "session-49")
        self.assertEqual(store.fail(task.number, "same-root-cause") ["status"], "retry")
        second = self.claim(store, task)
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["worker_id"], first["worker_id"])
        self.assertNotEqual(second["attempt_id"], first["attempt_id"])
        self.assertEqual(second["codex_session_id"], "session-49")
        blocked = store.fail(task.number, "same-root-cause")
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["same_failure_count"], 2)
        task_ready = make_issue(labels=("agent:ready",))
        self.assertIsNone(store.claim(task_ready, queue.workspace_plan(task_ready, self.root / "worktrees"), writer_slots=2))
        approved = make_issue(labels=("agent:ready", "human:retry-approved"))
        released = store.claim(approved, queue.workspace_plan(approved, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(released)
        self.assertEqual(released["attempts"], 3)
        self.assertEqual(released["retry_authorization_consumed"], 1)
        self.assertEqual(released["run_id"], first["run_id"])

    def test_retry_approval_is_single_use_across_failure_and_store_restart(self) -> None:
        path = self.root / "single-use.sqlite3"
        store = queue.RunStore(path)
        task = make_issue()
        first = self.claim(store, task)
        store.fail(task.number, "same failure")
        self.claim(store, task)
        blocked = store.fail(task.number, "same failure")
        self.assertEqual(blocked["status"], "blocked")

        approved = make_issue(labels=("agent:retry", "human:retry-approved"))
        authorized = store.claim(
            approved, queue.workspace_plan(approved, self.root / "worktrees"), writer_slots=2,
        )
        self.assertEqual(authorized["attempts"], 3)
        self.assertEqual(authorized["retry_authorization_consumed"], 1)
        self.assertEqual(authorized["run_id"], first["run_id"])
        failed = store.fail(task.number, "authorized attempt failed")
        self.assertEqual(failed["status"], "blocked")
        store.close()

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)
        self.assertIsNone(reopened.claim(
            approved, queue.workspace_plan(approved, self.root / "worktrees"), writer_slots=2,
        ))
        self.assertTrue(reopened.observe_retry_approval(task.number, approval_present=False))
        reapproved = reopened.claim(
            approved, queue.workspace_plan(approved, self.root / "worktrees"), writer_slots=2,
        )
        self.assertEqual(reapproved["attempts"], 4)
        self.assertEqual(reapproved["retry_authorization_consumed"], 1)
        reopened.update(task.number, status="retry")
        self.assertIsNone(reopened.claim(
            make_issue(labels=("agent:retry",)),
            queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2,
        ))
        exhausted = reopened.get(task.number)
        self.assertEqual(exhausted["status"], "blocked")
        self.assertEqual(exhausted["attempts"], 4)

    def test_watcher_durably_rearms_only_after_observing_absent_approval(self) -> None:
        path = self.root / "watch-rearm.sqlite3"
        store = queue.RunStore(path)
        task = make_issue()
        self.claim(store, task)
        store.fail(task.number, "same failure")
        self.claim(store, task)
        store.fail(task.number, "same failure")
        approved = make_issue(labels=("agent:ready", "human:retry-approved"))
        self.assertIsNotNone(store.claim(
            approved, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2,
        ))
        store.fail(task.number, "authorized attempt failed")
        store.close()

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)

        class Github:
            def issue(_self, number: int) -> queue.Issue:
                return make_issue(number, labels=("agent:blocked",))

            def ready_issues(_self) -> list[queue.Issue]:
                return []

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, reopened, Github(), object(),
            notifier=lambda _root, _event, _workspace: 0,
        )
        cycles = pilot.watch(
            poll_interval=1, stop_event=threading.Event(), issue_numbers=(task.number,),
            stop_at_review_checkpoint=True,
        )
        self.assertEqual(cycles[-1]["retry_authorizations_rearmed"], [task.number])
        self.assertEqual(reopened.get(task.number)["retry_authorization_consumed"], 0)
        self.assertEqual(
            json.loads(reopened.events(task.number)[-1]["details_json"])["reason"],
            "approval_label_absence_observed",
        )

    def test_saved_session_base_mismatch_is_rejected_before_claim_or_label_mutation(self) -> None:
        source_path = self.root / "legacy-inconsistent.sqlite3"
        source = queue.RunStore(source_path)
        task_number = 62062
        task = make_issue(
            task_number,
            labels=("agent:retry", "human:retry-approved"),
        )
        first = source.claim(
            task, queue.workspace_plan(task, self.root / "source-worktrees"), writer_slots=2,
        )
        source.set_codex_session(task_number, "saved-session-fixture")
        source.fail(task_number, "legacy setup failure")
        source.claim(task, queue.workspace_plan(task, self.root / "source-worktrees"), writer_slots=2)
        source.set_codex_session(task_number, "saved-session-fixture")
        blocked = source.fail(task_number, "legacy setup failure")
        self.assertEqual(blocked["status"], "blocked")
        original_attempt_id = str(blocked["attempt_id"])
        source.close()

        copied_path = self.root / "copied-inconsistent.sqlite3"
        with sqlite3.connect(source_path) as source_db, sqlite3.connect(copied_path) as copied_db:
            source_db.backup(copied_db)

        store = queue.RunStore(copied_path)
        self.addCleanup(store.close)
        mutations: list[tuple[int, str | None, tuple[str, ...], bool]] = []

        class Github:
            def set_status(_self, number: int, *, add: str | None, remove: object) -> None:
                details = json.loads(store.events(number)[-1]["details_json"])
                mutations.append((number, add, tuple(remove), details.get("reason_code") == "saved_session_base_sha_missing"))

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store, Github(), object(),
            notifier=lambda _root, _event, _workspace: 0,
        )
        with patch.object(queue, "create_worktree", side_effect=AssertionError("must reject before worktree setup")):
            self.assertEqual(pilot.dispatch_batch([task]), [])

        persisted = store.get(task_number)
        self.assertEqual(persisted["status"], "blocked")
        self.assertEqual(persisted["attempts"], 2)
        self.assertEqual(persisted["attempt_id"], original_attempt_id)
        self.assertEqual(persisted["retry_authorization_consumed"], 0)
        rejected = [event for event in store.events(task_number) if event["event_type"] == "dispatch_preflight_rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(json.loads(rejected[0]["details_json"]), {
            "reason_code": "saved_session_base_sha_missing", "stage": "resume_binding",
        })
        self.assertEqual(mutations, [(task_number, queue.BLOCKED_LABEL,
                                      (queue.RUNNING_LABEL, queue.READY_LABEL, queue.RETRY_LABEL,
                                       queue.HUMAN_RETRY_APPROVED), True)])
        self.assertEqual(first["run_id"], persisted["run_id"])

    def test_codex_session_cannot_be_reused_across_task_runs(self) -> None:
        store = self.store()
        first_issue = make_issue(49)
        second_issue = make_issue(50)
        first = self.claim(store, first_issue)
        second = self.claim(store, second_issue)

        store.set_codex_session(first_issue.number, "shared-session-fixture")
        with self.assertRaisesRegex(RuntimeError, "already bound to another task run"):
            store.set_codex_session(second_issue.number, "shared-session-fixture")
        with self.assertRaisesRegex(RuntimeError, "already bound to this task run"):
            store.set_codex_session(first_issue.number, "replacement-session-fixture")

        self.assertEqual(store.get(second_issue.number)["codex_session_id"], "")
        self.assertEqual(store.get(first_issue.number)["run_id"], first["run_id"])
        self.assertEqual(store.get(second_issue.number)["run_id"], second["run_id"])
        store.set_codex_session(second_issue.number, "independent-session-fixture")
        with self.assertRaisesRegex(RuntimeError, "already bound to this task run"):
            store.set_codex_session(second_issue.number, "replacement-session-fixture")
        self.assertEqual(
            store.set_codex_session(first_issue.number, "shared-session-fixture")["codex_session_id"],
            "shared-session-fixture",
        )

    def test_task_packet_binding_rejects_cross_run_and_workspace_identity(self) -> None:
        store = self.store()
        task = make_issue(49)
        row = self.claim(store, task)
        workspace = queue.workspace_plan(task, self.root / "worktrees")
        packet = queue.TaskPacket(
            issue_number=task.number,
            run_id=str(row["run_id"]),
            attempt_id=str(row["attempt_id"]),
            worker_id=str(row["worker_id"]),
            workspace=workspace,
            prompt="fixture prompt",
            result_path=str(self.root / "result.json"),
            codex_session_id="",
            workspace_write_roots=queue.ISSUE_WORKSPACE_WRITE_ROOTS.get(task.number),
        )
        self.assertIsNone(store.packet_binding_error(packet))

        cases = (
            (replace(packet, run_id="wrong-run"), "packet_run_id_mismatch"),
            (replace(packet, attempt_id="wrong-attempt"), "packet_attempt_id_mismatch"),
            (replace(packet, worker_id="Local01"), "packet_worker_id_mismatch"),
            (replace(packet, workspace=replace(workspace, issue_number=50)),
             "packet_workspace_issue_mismatch"),
            (replace(packet, workspace=replace(workspace, branch="issue/50-task")),
             "packet_branch_mismatch"),
            (replace(packet, workspace=replace(workspace, path=str(self.root / "other"))),
             "packet_workspace_path_mismatch"),
            (replace(packet, codex_session_id="wrong-session"), "packet_session_id_mismatch"),
            (replace(packet, workspace_write_roots=("/tmp/unrelated",)),
             "packet_write_roots_mismatch"),
        )
        for candidate, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(store.packet_binding_error(candidate), reason)

    def test_packet_binding_rejection_is_durable_and_blocks_the_run(self) -> None:
        path = self.root / "packet-rejection.sqlite3"
        store = queue.RunStore(path)
        task = make_issue(49)
        self.claim(store, task)
        blocked = store.reject_packet_binding(task.number, "packet_worker_id_mismatch")
        self.assertEqual(blocked["status"], "blocked")
        store.close()

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get(task.number)["status"], "blocked")
        event = reopened.events(task.number)[-1]
        self.assertEqual(event["event_type"], "task_packet_binding_rejected")
        self.assertEqual(json.loads(event["details_json"]), {
            "reason_code": "packet_worker_id_mismatch", "stage": "executor_handoff",
        })

    def test_setup_failure_persists_safe_reason_without_exception_text(self) -> None:
        store = self.store()
        task = make_issue()

        class Github:
            def set_status(self, _number: int, *, add: str | None, remove: object) -> None:
                return None

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store, Github(), object(),
            notifier=lambda _root, _event, _workspace: 0,
        )
        with patch.object(
            queue, "create_worktree",
            side_effect=RuntimeError("existing issue workspace has local changes at /private/user/path"),
        ):
            self.assertEqual(pilot.dispatch_batch([task]), [])
        failure = next(
            event for event in reversed(store.events(task.number))
            if event["event_type"] == "attempt_failed"
        )
        details = json.loads(failure["details_json"])
        self.assertEqual(details["stage"], "worktree_create")
        self.assertEqual(details["reason_code"], "worktree_has_local_changes")
        self.assertEqual(details["exception_type"], "RuntimeError")
        self.assertNotIn("/private/user/path", failure["details_json"])
        self.assertNotIn("workspace-or-issue-update-failed", failure["details_json"])

    def test_ci_feedback_retry_resumes_the_same_codex_owner(self) -> None:
        store = self.store()
        task = make_issue()
        first = self.claim(store, task)
        store.set_codex_session(task.number, "resume-session")
        queue.record_execution_success(store, task.number, "a" * 40)
        failed = queue.route_review_feedback(store, task.number, "Fix the CI regression.")
        self.assertEqual(failed["status"], "retry")

        resumed = self.claim(store, task)
        self.assertEqual(resumed["run_id"], first["run_id"])
        self.assertEqual(resumed["worker_id"], first["worker_id"])
        self.assertEqual(resumed["codex_session_id"], "resume-session")
        command = queue.codex_resume_command(
            resumed["codex_session_id"], self.root / "result.json"
        )
        self.assertEqual(command[1:3], ["exec", "resume"])
        self.assertLess(command.index("--strict-config"), command.index("resume-session"))
        self.assertNotIn("--add-dir", command)
        self.assertEqual(command[-2:], ["resume-session", "-"])
        self.assertNotIn("--sandbox", command)
        self.assertIn("default_permissions=robotsim_worker", command)
        self.assertIn("permissions.robotsim_worker.network.enabled=false", command)

    def test_ci_failure_context_reuses_owner_then_blocks_same_root_cause(self) -> None:
        store = self.store()
        task = make_issue()
        first = self.claim(store, task)
        workspace, head = self.create_task_workspace(task)
        store.set_codex_session(task.number, "ci-session")
        store.update(task.number, status="awaiting_pr")
        failed_pull = queue.PullRequest(
            50, "https://github.com/lzy18001500226/RobotSim/pull/50",
            "issue/49-task", head, "OPEN", "failure", "review_required",
            check_details=({"name": "Agent infrastructure checks", "state": "failure"},),
        )

        class Github:
            def issue(self, number: int) -> queue.Issue:
                return task

            def pull_for_branch(self, branch: str) -> queue.PullRequest:
                return failed_pull

            def branch_sha(self, _root: Path, branch: str) -> str | None:
                return head if branch == failed_pull.branch else None

            def set_status(self, number: int, *, add: str | None, remove: object) -> None:
                return None

        pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, store,
            Github(), object(), notifier=lambda _root, _event, _workspace: 0,
        )
        self.assertEqual(pilot.reconcile()[0]["status"], "retry")
        retry = store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2)
        self.assertEqual(retry["run_id"], first["run_id"])
        self.assertEqual(retry["worker_id"], first["worker_id"])
        self.assertNotEqual(retry["attempt_id"], first["attempt_id"])
        self.assertEqual(retry["codex_session_id"], "ci-session")
        failure_context = json.loads(retry["review_feedback"])
        self.assertEqual(failure_context["failed_required_checks"], ["Agent infrastructure checks"])

        store.update(task.number, status="awaiting_pr")
        self.assertEqual(pilot.reconcile()[0]["status"], "blocked")
        self.assertIsNone(store.claim(
            make_issue(labels=("agent:ready", "agent:retry")),
            queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2,
        ))

    def test_distinct_failures_still_stop_at_the_attempt_limit(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        for attempt in range(queue.MAX_ATTEMPTS):
            failed = store.fail(task.number, f"distinct-root-cause-{attempt}")
            expected = "blocked" if attempt + 1 == queue.MAX_ATTEMPTS else "retry"
            self.assertEqual(failed["status"], expected)
            if expected == "retry":
                self.claim(store, task)
        self.assertEqual(store.get(task.number)["attempts"], queue.MAX_ATTEMPTS)

    def test_state_recovers_after_store_reopen_and_prevents_duplicate_claim(self) -> None:
        path = self.root / "resume.sqlite3"
        task = make_issue()
        first = queue.RunStore(path)
        row = first.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(row)
        first.close()

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.active_count(), 1)
        self.assertIsNone(reopened.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2))

    def test_attempt_history_survives_retry_and_store_restart(self) -> None:
        path = self.root / "history.sqlite3"
        store = queue.RunStore(path)
        task = make_issue()
        first = self.claim(store, task)
        store.set_codex_session(task.number, "durable-session")
        store.fail(task.number, "same-root-cause")
        second = self.claim(store, task)
        events = store.events(task.number)
        store.close()

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)
        persisted = reopened.events(task.number)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual([event["event_type"] for event in persisted], [
            "attempt_claimed", "codex_session_started", "attempt_failed", "attempt_claimed",
        ])
        self.assertEqual(json.loads(persisted[1]["details_json"])["session_id"], "durable-session")
        self.assertEqual(len(events), len(persisted))

    def test_existing_queue_database_gets_closeout_payload_column(self) -> None:
        path = self.root / "old-state.sqlite3"
        with sqlite3.connect(path) as database:
            database.execute(
                "CREATE TABLE task_runs (issue_number INTEGER PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL)"
            )
        store = queue.RunStore(path)
        self.addCleanup(store.close)
        columns = {row[1] for row in store._db.execute("PRAGMA table_info(task_runs)")}
        self.assertIn("closeout_event_json", columns)
        self.assertIn("notification_state", columns)
        self.assertIn("notification_attempts", columns)
        self.assertIn("retry_authorization_consumed", columns)

    def test_legacy_blocked_run_quarantines_unknown_retry_authorization(self) -> None:
        path = self.root / "legacy-blocked.sqlite3"
        legacy = queue.RunStore(path)
        task = make_issue()
        self.claim(legacy, task)
        legacy.update(task.number, status="blocked")
        legacy.close()
        with sqlite3.connect(path) as database:
            database.execute("ALTER TABLE task_runs DROP COLUMN retry_authorization_consumed")

        migrated = queue.RunStore(path)
        self.addCleanup(migrated.close)
        row = migrated.get(task.number)
        self.assertEqual(row["retry_authorization_consumed"], 1)
        self.assertIn(
            "retry_authorization_quarantined",
            [event["event_type"] for event in migrated.events(task.number)],
        )
        self.assertFalse(migrated.observe_retry_approval(task.number, approval_present=True))
        self.assertTrue(migrated.observe_retry_approval(task.number, approval_present=False))

    def test_dead_primary_recovers_to_retry_after_restart(self) -> None:
        path = self.root / "interrupted.sqlite3"
        task = make_issue()
        first = queue.RunStore(path)
        claimed = first.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(claimed)
        first.update(task.number, executor_pid=2_000_000_000, executor_start_token="no-such-process")
        first.close()

        class Github:
            def branch_pushed(self, repository_root: Path, branch: str) -> bool:
                return False

            def set_status(self, number: int, *, add: str | None, remove: object) -> None:
                self.last_status = add

        reopened = queue.RunStore(path)
        self.addCleanup(reopened.close)
        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, reopened, Github(),
            object(), notifier=lambda _root, _event: 0,
        )
        recovered = pilot.recover()
        self.assertEqual(recovered[0]["status"], "retry")
        self.assertEqual(reopened.active_count(), 0)
        self.assertIsNotNone(reopened.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2))

    def test_recovery_keeps_writer_slot_when_remote_state_is_unavailable(self) -> None:
        store = self.store()
        task = make_issue()
        claimed = store.claim(
            task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=1,
        )
        self.assertIsNotNone(claimed)
        store.update(task.number, executor_pid=2_000_000_000, executor_start_token="no-such-process")

        class Github:
            def branch_pushed(self, _root: Path, _branch: str) -> bool | None:
                return None

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store, Github(), object(),
            writer_slots=1,
        )
        recovered = pilot.recover()
        self.assertEqual(recovered[0]["status"], "running")
        self.assertEqual(store.get(task.number)["status"], "running")
        self.assertEqual(store.active_count(), 1)
        next_issue = make_issue(50)
        self.assertIsNone(store.claim(
            next_issue, queue.workspace_plan(next_issue, self.root / "worktrees"),
            writer_slots=1,
        ))

    def test_duplicate_dispatch_for_same_issue_is_rejected(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        self.assertIsNone(store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2))
        self.assertEqual(store.active_count(), 1)

    def test_retry_keeps_the_persisted_worktree_path(self) -> None:
        store = self.store()
        task = make_issue()
        original = queue.workspace_plan(task, self.root / "original-worktrees")
        first = store.claim(task, original, writer_slots=2)
        self.assertIsNotNone(first)
        store.fail(task.number, "controlled-retry")
        changed_default = queue.workspace_plan(task, self.root / "new-worktrees")
        resumed = store.claim(task, changed_default, writer_slots=2)
        self.assertEqual(resumed["workspace"], original.path)
        self.assertEqual(resumed["branch"], original.branch)

    def test_concurrent_dispatchers_claim_one_primary_for_the_same_issue(self) -> None:
        path = self.root / "duplicate.sqlite3"
        first, second = queue.RunStore(path), queue.RunStore(path)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        task = make_issue()
        plan = queue.workspace_plan(task, self.root / "worktrees")
        barrier = threading.Barrier(3)

        def claim(store: queue.RunStore) -> dict[str, object] | None:
            barrier.wait()
            return store.claim(task, plan, writer_slots=2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim, first), pool.submit(claim, second)]
            barrier.wait()
            results = [future.result() for future in futures]
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(first.active_count(), 1)

    def test_pending_pr_state_prevents_reclaim_even_if_ready_label_is_stale(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        store.update(task.number, status="awaiting_pr")
        self.assertIsNone(store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2))

    def test_watch_stops_at_review_checkpoint_without_dispatching_again(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        store.update(task.number, status="awaiting_review")

        class Github:
            def ready_issues(self) -> list[queue.Issue]:
                raise AssertionError("watch should stop before another queue cycle")

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store,
            Github(), object(), notifier=lambda _root, _event: 0,
        )
        pilot.recover = lambda: []
        pilot.reconcile = lambda _numbers=None: []
        cycles = pilot.watch(
            poll_interval=1,
            stop_event=threading.Event(),
            issue_numbers=(task.number,),
            stop_at_review_checkpoint=True,
        )
        self.assertEqual(len(cycles), 1)
        self.assertEqual(store.get(task.number)["status"], "awaiting_review")

    def test_human_visual_gate_stops_then_gate_approval_restores_eligibility(self) -> None:
        store = self.store()
        task = make_issue(labels=("agent:review",))
        row = self.claim(store, task)
        store.update(task.number, status="awaiting_pr")
        stopped = queue.reconcile_pull_request(
            store, task.number, make_pull(), human_gates=("human:visual",),
            independent_review_passed=True,
        )
        self.assertEqual(stopped["status"], "human:visual")
        self.assertFalse(queue.can_merge(stopped))

        released = queue.reconcile_pull_request(
            store, task.number, make_pull(), approved_gates=("human:visual",),
            independent_review_passed=True,
        )
        self.assertEqual(released["status"], "merge_eligible")
        self.assertTrue(queue.can_merge(released))

    def test_human_repro_gate_stops_until_explicit_approval(self) -> None:
        store = self.store()
        task = make_issue(labels=("agent:review", "human:repro"))
        self.claim(store, task)
        store.update(task.number, status="awaiting_pr")
        gates = queue.derive_human_gates(task, ())
        self.assertEqual(gates, ("human:repro",))
        stopped = queue.reconcile_pull_request(
            store, task.number, make_pull(), human_gates=gates,
            independent_review_passed=True,
        )
        self.assertEqual(stopped["status"], "human_gate")
        packet = queue.human_review_packet(49, make_pull(), gates=gates)
        self.assertEqual(packet["next_action"], "Human reproduction check")
        released = queue.reconcile_pull_request(
            store, task.number, make_pull(), approved_gates=("human:repro",),
            independent_review_passed=True,
        )
        self.assertEqual(released["status"], "merge_eligible")

    def test_done_is_derived_from_closed_issue_not_just_merged_pr(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        store.update(task.number, status="awaiting_pr")
        merged = queue.PullRequest(
            50, "https://github.com/lzy18001500226/RobotSim/pull/50",
            "issue/49-task", "b" * 40, "CLOSED", "success", "approved", merged=True,
        )
        waiting = queue.reconcile_pull_request(store, task.number, merged)
        self.assertEqual(waiting["status"], "awaiting_issue_close")
        done = queue.reconcile_pull_request(store, task.number, merged, issue_closed=True)
        self.assertEqual(done["status"], "done")
        self.assertFalse(queue.can_merge(done))
        self.assertIsNone(store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2))

    def test_sensitive_change_paths_require_matching_human_gates(self) -> None:
        task = make_issue()
        gates = queue.derive_human_gates(
            task,
            ("docs/adr/0007.md", "scripts/agent/task_queue.py", "robots/unitree_g1/driver.cpp", "apps/unity/scene.unity"),
        )
        self.assertEqual(
            gates,
            ("human:architecture", "human:hardware", "human:security", "human:visual"),
        )
        approved = queue.Issue(
            task.number, task.title, task.body, task.state,
            task.labels | frozenset({"human:security-approved"}),
        )
        remaining = queue.derive_human_gates(approved, ("scripts/agent/task_queue.py", "apps/unity/scene.unity"))
        self.assertEqual(remaining, ("human:visual",))

    def test_review_feedback_routes_to_retry_and_same_feedback_blocks(self) -> None:
        store = self.store()
        task = make_issue(labels=("agent:review",))
        self.claim(store, task)
        first = queue.route_review_feedback(store, task.number, "Correct the timestamp reset handling.")
        self.assertEqual(first["status"], "retry")
        self.claim(store, task)
        second = queue.route_review_feedback(store, task.number, "Correct the timestamp reset handling.")
        self.assertEqual(second["status"], "blocked")
        prompt = queue.render_task(
            task, self.root, review_feedback=str(second["review_feedback"]),
            guidance={
                "revision": "a" * 40,
                "AGENTS.md": "Follow repository instructions.",
                "docs/workflows/goal_driven_development.md": "Use the Goal workflow.",
            },
        )
        self.assertIn("Correct the timestamp reset handling.", prompt)
        review_command = queue.codex_command(self.root, read_only=True)
        self.assertIn("default_permissions=robotsim_worker", review_command)
        review_filesystem = next(
            value for value in review_command
            if value.startswith("permissions.robotsim_worker.filesystem=")
        )
        self.assertIn('"/home" = "read"', review_filesystem)

    def test_next_task_waits_for_issue_dependency_to_close(self) -> None:
        task = make_issue(60, body="Depends on: #49")
        open_parent = make_issue(49)
        closed_parent = make_issue(49, state="CLOSED")
        self.assertFalse(queue.next_task_eligible(task, {49: open_parent}))
        self.assertTrue(queue.next_task_eligible(task, {49: closed_parent}))
        self.assertFalse(queue.next_task_eligible(make_issue(60, state="CLOSED"), {49: closed_parent}))

    def test_persisted_run_keeps_issue_attempt_and_closeout_identity(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        event = queue.closeout_event(task, row, status="blocked", summary="A repeated root cause blocked the task.")
        self.assertEqual(event["task_id"], "issue-49-symphony-task")
        self.assertEqual(event["attempt_id"], row["attempt_id"])
        self.assertEqual(event["pr_number"], None)

    def test_task_base_sha_is_immutable_in_run_state(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)

        pinned = store.set_base_sha(task.number, "a" * 40)

        self.assertEqual(pinned["base_sha"], "a" * 40)
        self.assertEqual(store.set_base_sha(task.number, "a" * 40)["base_sha"], "a" * 40)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            store.set_base_sha(task.number, "b" * 40)

    def test_disposable_issue_scope_checks_every_commit_not_only_final_diff(self) -> None:
        repository = self.root / "scope-repository"
        repository.mkdir()
        for command in (
            ["git", "init", "--initial-branch=main", str(repository)],
            ["git", "-C", str(repository), "config", "user.name", "RobotSim Test"],
            ["git", "-C", str(repository), "config", "user.email", "robotsim-test@example.invalid"],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True)
        (repository / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "base"],
            check=True, capture_output=True, text=True,
        )
        base_sha = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
        ).strip()
        subprocess.run(
            ["git", "-C", str(repository), "checkout", "-b", "issue/62-task"],
            check=True, capture_output=True, text=True,
        )

        note = repository / "docs/research/issue49-queue-e2e-note.md"
        probe = repository / "tests/test_issue49_recovery_probe.py"
        note.parent.mkdir(parents=True)
        probe.parent.mkdir(parents=True)
        note.write_text("fixture note\n", encoding="utf-8")
        probe.write_text("pass\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "allowed fixture files"],
            check=True, capture_output=True, text=True,
        )
        queue._validate_task_commit_scope(repository, base_sha, 62)

        source = repository / "robots/unitree_g1/controller.cpp"
        source.parent.mkdir(parents=True)
        source.write_text("temporary production change\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "robots"], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "out of scope"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "rm", "robots/unitree_g1/controller.cpp"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", str(repository), "commit", "-m", "revert out of scope"],
            check=True, capture_output=True, text=True,
        )
        with self.assertRaisesRegex(RuntimeError, "exact path allowlist"):
            queue._validate_task_commit_scope(repository, base_sha, 62)

    def test_failed_closeout_retry_reuses_exact_payload_and_is_reported(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        self.create_task_workspace(task)
        calls: list[dict[str, object]] = []
        results = iter((1, 0))

        def notifier(_root: Path, event: object, _workspace: Path) -> int:
            self.assertIsInstance(event, dict)
            calls.append(dict(event))
            return next(results)

        class Github:
            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "changed-worktree-root", self.root, store,
            Github(), object(), notifier=notifier,
        )
        first = pilot._emit_closeout(
            task, row, status="blocked", summary="Blocked on the first call.",
        )
        self.assertEqual(first, 1)
        self.assertIn(task.number, pilot.notification_failures)
        retry_row = store.get(task.number)
        self.assertEqual(retry_row["notified"], 0)
        second = pilot._emit_closeout(
            task, retry_row, status="blocked", summary="Different retry text must not replace payload.",
        )
        self.assertEqual(second, 0)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(store.get(task.number)["notified"], 1)
        self.assertEqual(store.get(task.number)["notification_state"], "sent")
        self.assertEqual(store.get(task.number)["notification_attempts"], 2)
        self.assertNotIn(task.number, pilot.notification_failures)

    def test_pending_closeout_notification_resumes_after_store_restart(self) -> None:
        path = self.root / "pending-closeout.sqlite3"
        first_store = queue.RunStore(path)
        task = make_issue()
        row = self.claim(first_store, task)
        self.create_task_workspace(task)
        first_payloads: list[dict[str, object]] = []

        def fail_once(_root: Path, event: object, _workspace: Path) -> int:
            first_payloads.append(dict(event))
            return 1

        class Github:
            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        first_pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, first_store,
            Github(), object(), notifier=fail_once,
        )
        self.assertEqual(first_pilot._emit_closeout(
            task, row, status="blocked", summary="Blocked until retry.",
        ), 1)
        self.assertEqual(first_store.get(task.number)["notification_state"], "failed")
        first_store.close()

        second_store = queue.RunStore(path)
        self.addCleanup(second_store.close)
        second_payloads: list[dict[str, object]] = []

        def succeed(_root: Path, event: object, _workspace: Path) -> int:
            second_payloads.append(dict(event))
            return 0

        second_pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, second_store,
            Github(), object(), notifier=succeed,
        )
        self.assertEqual(second_pilot._emit_closeout(
            task, second_store.get(task.number), status="blocked",
            summary="The retry must preserve the original event.",
        ), 0)
        self.assertEqual(first_payloads[0], second_payloads[0])
        self.assertEqual(second_store.get(task.number)["notification_state"], "sent")
        self.assertEqual(second_store.get(task.number)["notification_attempts"], 2)

    def test_closeout_notifier_runs_from_the_task_worktree(self) -> None:
        repository = self.root / "queue-repository"
        workspace = self.root / "worktrees" / "issue-49"
        mock_result = SimpleNamespace(
            returncode=0, stdout="notify_task: skipped: missing settings: AGENTMAIL_API_KEY\n",
            stderr="",
        )
        with patch.object(queue.subprocess, "run", return_value=mock_result) as run:
            result = queue.invoke_closeout(repository, {"task_id": "issue-49-symphony-task"}, workspace)
        self.assertEqual(result, queue.CloseoutResult(0, "skipped"))
        self.assertEqual(run.call_args.kwargs["cwd"], workspace)
        self.assertEqual(run.call_args.args[0][-1], "task-closeout")

    def test_closeout_notification_retry_exhaustion_is_durable(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        self.create_task_workspace(task)
        calls = 0

        def notifier(_root: Path, _event: object, _workspace: Path) -> int:
            nonlocal calls
            calls += 1
            return 1

        class Github:
            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, store,
            Github(), object(), notifier=notifier,
        )
        for _ in range(queue.MAX_NOTIFICATION_ATTEMPTS):
            row = store.get(task.number)
            pilot._emit_closeout(task, row, status="blocked", summary="Blocked.")
        row = store.get(task.number)
        self.assertEqual(calls, queue.MAX_NOTIFICATION_ATTEMPTS)
        self.assertEqual(row["notification_state"], "exhausted")
        self.assertEqual(row["notification_attempts"], queue.MAX_NOTIFICATION_ATTEMPTS)

    def test_status_report_omits_prompts_and_reports_operational_identity(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        store.update(task.number, review_feedback="private review transcript")

        class Github:
            def ready_issues(self) -> list[queue.Issue]:
                return [task]

            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        report = queue.build_status_report(
            store, Github(), repository_root=self.root, state_path=self.root / "state.sqlite3",
            worktree_root=self.root / "worktrees", writer_slots=2,
        )
        encoded = json.dumps(report)
        self.assertEqual(report["running_workers"][0]["issue"], task.number)
        self.assertEqual(report["running_workers"][0]["attempt_id"], row["attempt_id"])
        self.assertIn("reproduction_command", report)
        self.assertNotIn("private review transcript", encoded)
        self.assertNotIn("review_feedback", encoded)

    def test_status_report_does_not_migrate_or_write_legacy_state(self) -> None:
        path = self.root / "legacy-state.sqlite3"
        with sqlite3.connect(path) as database:
            database.execute(
                """CREATE TABLE task_runs (
                    issue_number INTEGER PRIMARY KEY, status TEXT NOT NULL,
                    attempts INTEGER NOT NULL, last_failure_hash TEXT NOT NULL,
                    same_failure_count INTEGER NOT NULL, run_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL, worker_id TEXT NOT NULL,
                    branch TEXT NOT NULL, workspace TEXT NOT NULL,
                    pr_number INTEGER, head_sha TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )
            database.execute(
                """CREATE TABLE task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_number INTEGER NOT NULL, run_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL, worker_id TEXT NOT NULL,
                    event_type TEXT NOT NULL, status TEXT NOT NULL,
                    head_sha TEXT NOT NULL DEFAULT '', pr_number INTEGER,
                    details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                )"""
            )
            database.execute(
                "INSERT INTO task_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (62, "blocked", 2, "fingerprint", 2, "run", "attempt", "worker",
                 "issue/62-task", (self.root / "issue-62").as_posix(), None, "", "old-time"),
            )
        original_bytes = path.read_bytes()

        class Github:
            def ready_issues(self) -> list[queue.Issue]:
                return []

            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        with patch.object(queue, "GitHubCLI", return_value=Github()):
            with redirect_stdout(io.StringIO()) as output:
                result = queue.main([
                    "--status-report", "--state", str(path),
                    "--worktree-root", str(self.root / "worktrees"),
                ])

        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["blocked_tasks"][0]["issue"], 62)
        self.assertEqual(report["blocked_tasks"][0]["notification_state"], "not_started")
        self.assertEqual(path.read_bytes(), original_bytes)
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
            columns = {row[1] for row in database.execute("PRAGMA table_info(task_runs)")}
            self.assertNotIn("notification_state", columns)
            self.assertNotIn("notification_attempts", columns)

        missing_path = self.root / "missing-state" / "state.sqlite3"
        with patch.object(queue, "GitHubCLI", return_value=Github()):
            with redirect_stdout(io.StringIO()) as output:
                result = queue.main([
                    "--status-report", "--state", str(missing_path),
                    "--worktree-root", str(self.root / "worktrees"),
                ])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["tasks"], [])
        self.assertFalse(missing_path.parent.exists())

    def test_status_report_surfaces_issue_close_gate_and_pr_checkpoint(self) -> None:
        store = self.store()
        task = make_issue()
        self.claim(store, task)
        store.record_event(task.number, "pull_request_observed", {
            "pull_number": 88, "head_sha": "a" * 40, "checks": "success",
        })
        store.update(task.number, status="awaiting_issue_close")

        class Github:
            def ready_issues(self) -> list[queue.Issue]:
                return []

            def branch_sha(self, _root: Path, _branch: str) -> None:
                return None

        report = queue.build_status_report(
            store, Github(), repository_root=self.root, state_path=self.root / "state.sqlite3",
            worktree_root=self.root / "worktrees", writer_slots=2,
        )
        self.assertEqual(report["waiting_human_approval"][0]["issue"], task.number)
        self.assertEqual(
            report["tasks"][0]["last_successful_checkpoint"]["pr_number"], 88,
        )

    def test_human_review_handoff_explicitly_notifies_once(self) -> None:
        store = self.store()
        task = make_issue(labels=("agent:review",))
        self.claim(store, task)
        workspace, head = self.create_task_workspace(task)
        pull = queue.PullRequest(
            50, "https://github.com/lzy18001500226/RobotSim/pull/50",
            "issue/49-task", head, "OPEN", "success", "approved",
        )
        store.update(
            task.number, status="awaiting_pr", head_sha=head, pr_number=pull.number,
            reviewed_head_sha=pull.head_sha, review_report="clear",
        )

        class Github:
            def __init__(self) -> None:
                self.comments: list[str] = []
                self.status_updates: list[tuple[int, str | None]] = []

            def issue(self, number: int) -> queue.Issue:
                return task

            def pull_for_branch(self, branch: str) -> queue.PullRequest:
                return pull

            def branch_sha(self, _root: Path, branch: str) -> str | None:
                return head if branch == pull.branch else None

            def changed_paths(self, number: int) -> tuple[str, ...]:
                return ("apps/unity/demo.unity",)

            def comment(self, number: int, body: str) -> None:
                self.comments.append(body)

            def set_status(self, number: int, *, add: str | None, remove: object) -> None:
                self.status_updates.append((number, add))

        github = Github()
        notifications: list[dict[str, object]] = []

        def notifier(_root: Path, event: object, task_workspace: Path) -> int:
            self.assertIsInstance(event, dict)
            self.assertEqual(task_workspace, workspace)
            notifications.append(dict(event))
            return 0

        pilot = queue.TaskQueuePilot(
            self.queue_repository, self.root / "worktrees", self.root, store, github,
            object(), notifier=notifier,
        )
        first = pilot.reconcile()
        second = pilot.reconcile()
        self.assertEqual(first[0]["status"], "human:visual")
        self.assertEqual(second[0]["status"], "human:visual")
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
