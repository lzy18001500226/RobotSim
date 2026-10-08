"""Deterministic tests for the single-host Issue queue pilot."""

from __future__ import annotations

import sqlite3
import json
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

    def store(self, name: str = "state.sqlite3") -> queue.RunStore:
        store = queue.RunStore(self.root / name)
        self.addCleanup(store.close)
        return store

    def claim(self, store: queue.RunStore, task: queue.Issue) -> dict[str, object]:
        row = store.claim(task, queue.workspace_plan(task, self.root / "worktrees"), writer_slots=2)
        self.assertIsNotNone(row)
        assert row is not None
        return row

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
        subprocess.run(
            ["git", "-C", str(repository), "update-ref", "refs/remotes/origin/main", "HEAD"],
            check=True,
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
        self.assertEqual(writable_dirs[0], Path(subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "--path-format=absolute", "--git-dir"],
            text=True,
        ).strip()))
        self.assertIn(common_dir / "objects", writable_dirs)
        self.assertIn(common_dir / "refs/heads/issue", writable_dirs)
        self.assertIn(common_dir / "logs/refs/heads/issue", writable_dirs)
        self.assertNotIn(common_dir, writable_dirs)
        command = queue.codex_command(workspace)
        self.assertEqual(command.count("--add-dir"), len(writable_dirs))
        resume = queue.codex_resume_command("session-49", self.root / "result.json", workspace=workspace)
        roots_option = resume[resume.index("--config") + 1]
        self.assertTrue(roots_option.startswith("sandbox_workspace_write.writable_roots=["))
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
        self.assertEqual(released["attempts"], 1)
        self.assertEqual(released["run_id"], first["run_id"])

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
        self.assertEqual(command[1:4], ["exec", "resume", "resume-session"])
        self.assertNotIn("--sandbox", command)

    def test_ci_failure_context_reuses_owner_then_blocks_same_root_cause(self) -> None:
        store = self.store()
        task = make_issue()
        first = self.claim(store, task)
        store.set_codex_session(task.number, "ci-session")
        store.update(task.number, status="awaiting_pr")
        failed_pull = queue.PullRequest(
            50, "https://github.com/lzy18001500226/RobotSim/pull/50",
            "issue/49-task", "c" * 40, "OPEN", "failure", "review_required",
            check_details=({"name": "Agent infrastructure checks", "state": "failure"},),
        )

        class Github:
            def issue(self, number: int) -> queue.Issue:
                return task

            def pull_for_branch(self, branch: str) -> queue.PullRequest:
                return failed_pull

            def set_status(self, number: int, *, add: str | None, remove: object) -> None:
                return None

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store,
            Github(), object(), notifier=lambda _root, _event: 0,
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
        self.assertIn("--sandbox", queue.codex_command(self.root, read_only=True))
        self.assertIn("read-only", queue.codex_command(self.root, read_only=True))

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

    def test_failed_closeout_retry_reuses_exact_payload_and_is_reported(self) -> None:
        store = self.store()
        task = make_issue()
        row = self.claim(store, task)
        calls: list[dict[str, object]] = []
        results = iter((1, 0))

        def notifier(_root: Path, event: object) -> int:
            self.assertIsInstance(event, dict)
            calls.append(dict(event))
            return next(results)

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store,
            object(), object(), notifier=notifier,
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
        self.assertNotIn(task.number, pilot.notification_failures)

    def test_human_review_handoff_explicitly_notifies_once(self) -> None:
        store = self.store()
        task = make_issue(labels=("agent:review",))
        self.claim(store, task)
        pull = make_pull()
        store.update(
            task.number, status="awaiting_pr",
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

            def changed_paths(self, number: int) -> tuple[str, ...]:
                return ("apps/unity/demo.unity",)

            def comment(self, number: int, body: str) -> None:
                self.comments.append(body)

            def set_status(self, number: int, *, add: str | None, remove: object) -> None:
                self.status_updates.append((number, add))

        github = Github()
        notifications: list[dict[str, object]] = []

        def notifier(_root: Path, event: object) -> int:
            self.assertIsInstance(event, dict)
            notifications.append(dict(event))
            return 0

        pilot = queue.TaskQueuePilot(
            self.root, self.root / "worktrees", self.root, store, github,
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
