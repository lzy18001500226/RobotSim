from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.agent import eval_harness, infra_healthcheck as health


class InfraHealthcheckTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="robotsim-health-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.database = self.root / "not-created" / "outbox.sqlite3"
        self.env = {"ROBOTSIM_TASK_NOTIFICATION_DB": str(self.database)}
        self.hooks = {event: [{"hooks": [{
            "type": "command",
            "command": f'python3 "$(git rev-parse --show-toplevel)/scripts/agent/{handler}"',
            "commandWindows": "synthetic Windows command; not executed",
        }]}] for event, handler in health.HANDLERS.items()}
        files = {f"scripts/agent/{handler}": "# synthetic handler\n" for handler in health.HANDLERS.values()}
        files[".codex/hooks.json"] = json.dumps({"hooks": self.hooks}) + "\n"
        eval_harness._init_repo(self.repo, files)

    def write_hooks(self, content: object) -> None:
        (self.repo / ".codex/hooks.json").write_text(json.dumps(content), encoding="utf-8")

    def snapshot(self) -> dict:
        return {str(path.relative_to(self.root)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def create_outbox(self) -> None:
        self.database.parent.mkdir()
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript("""
                CREATE TABLE outcomes(publication_state TEXT, event_json TEXT);
                CREATE TABLE deliveries(channel TEXT, state TEXT, last_error TEXT);
                CREATE TABLE publication_jobs(state TEXT, last_error TEXT);
                INSERT INTO outcomes VALUES('verified', 'PRIVATE_EVENT_BODY');
                INSERT INTO outcomes VALUES('PRIVATE_STATE', 'PRIVATE_EVENT_BODY');
                INSERT INTO deliveries VALUES('agentmail', 'sent', 'PRIVATE_ERROR');
                INSERT INTO deliveries VALUES('PRIVATE_CHANNEL', 'PRIVATE_STATE', 'PRIVATE_ERROR');
                INSERT INTO publication_jobs VALUES('sending', 'PRIVATE_ERROR');
                INSERT INTO publication_jobs VALUES('disabled', 'PRIVATE_ERROR');
                INSERT INTO publication_jobs VALUES('PRIVATE_STATE', 'PRIVATE_ERROR');
            """)
            connection.commit()

    def test_reads_checkout_and_hooks_without_creating_database_or_changing_files(self) -> None:
        before = self.snapshot()
        report = health.inspect(self.repo, self.env)
        self.assertEqual(report["checkout"]["status"], "PASS")
        self.assertEqual(report["checkout"]["branch"], "main")
        self.assertEqual(report["checkout"]["worktree_status"], "")
        self.assertEqual(report["hooks"]["status"], "PASS")
        self.assertEqual(report["hooks"]["client_trust_enabled_and_invocation"], "NOT CHECKED")
        self.assertEqual(report["outbox"]["status"], "ABSENT")
        self.assertEqual(report["notifications"]["delivery_probe"], "NOT RUN")
        self.assertFalse(self.database.parent.exists())
        self.assertEqual(before, self.snapshot())

    def test_redacts_environment_and_reports_only_current_process_switch(self) -> None:
        env = {**self.env, "ROBOTSIM_LOCAL_STOP_HOOK": "1",
               **{name: "PRIVATE_ENV_VALUE" for name in health.MAIL_SETTINGS}}
        report = health.inspect(self.repo, env)
        self.assertTrue(report["notifications"]["local_stop_opt_in"])
        self.assertTrue(all(report["notifications"]["environment_setting_present"].values()))
        self.assertNotIn("PRIVATE_ENV_VALUE", json.dumps(report))
        self.assertFalse(health.inspect(self.repo, self.env)["notifications"]["local_stop_opt_in"])

    def test_aggregates_outbox_without_leaking_payloads_or_writing_sidecars(self) -> None:
        self.create_outbox()
        before = self.snapshot()
        report = health.inspect(self.repo, self.env)
        self.assertEqual(report["outbox"]["status"], "READ")
        self.assertEqual(report["outbox"]["publication_counts"], {"verified": 1, "UNKNOWN": 1})
        self.assertEqual(report["outbox"]["delivery_counts"], {"agentmail": {"sent": 1}, "UNKNOWN": {"UNKNOWN": 1}})
        self.assertEqual(report["outbox"]["publication_job_counts"], {"sending": 1, "disabled": 1, "UNKNOWN": 1})
        self.assertNotIn("PRIVATE_", json.dumps(report))
        self.assertEqual(before, self.snapshot())

    def test_corrupt_or_incompatible_outbox_fails_without_migration(self) -> None:
        self.database.parent.mkdir()
        for content in (b"not a database", b""):
            with self.subTest(content=content):
                self.database.write_bytes(content)
                before = self.snapshot()
                self.assertEqual(health.inspect(self.repo, self.env)["outbox"]["status"], "FAIL")
                self.assertEqual(before, self.snapshot())

    def test_active_wal_or_journal_defers_without_reading_or_writing(self) -> None:
        self.create_outbox()
        for suffix in ("-wal", "-journal"):
            with self.subTest(suffix=suffix):
                sidecar = Path(str(self.database) + suffix)
                sidecar.write_bytes(b"synthetic active sidecar")
                before = self.snapshot()
                with patch.object(health.sqlite3, "connect", side_effect=AssertionError("must not open")):
                    self.assertEqual(health._outbox(self.database)["status"], "DEFERRED")
                self.assertEqual(before, self.snapshot())
                sidecar.unlink()

    def test_changing_database_discards_potentially_stale_counts(self) -> None:
        self.create_outbox()
        with patch.object(health, "_stamp", side_effect=[((1, 2, 3), None, None), ((1, 3, 4), None, None)]):
            report = health._outbox(self.database)
        self.assertEqual(report["status"], "DEFERRED")
        self.assertNotIn("publication_counts", report)

    def test_missing_or_malformed_hooks_fail_without_executing_commands(self) -> None:
        for content in ({}, {"hooks": []}, {"hooks": {"Stop": "invalid"}},
                        {"hooks": {"Stop": ["invalid"]}}, {"hooks": {"Stop": [{"hooks": ["invalid"]}]}}):
            with self.subTest(content=content):
                self.write_hooks(content)
                self.assertEqual(health._hooks(self.repo)["status"], "FAIL")
        (self.repo / ".codex/hooks.json").unlink()
        self.assertEqual(health._hooks(self.repo)["status"], "FAIL")

    def test_default_hooks_need_no_prompt_or_interrupt_recorders(self) -> None:
        self.assertEqual(set(self.hooks), {"PreToolUse", "Stop"})
        self.assertEqual(health._hooks(self.repo)["status"], "PASS")
        del self.hooks["PreToolUse"]
        self.write_hooks({"hooks": self.hooks})
        self.assertEqual(health._hooks(self.repo)["status"], "FAIL")

    def test_missing_handler_or_drifted_command_is_not_reported_healthy(self) -> None:
        self.hooks["Stop"][0]["hooks"][0]["command"] = "synthetic unsafe command; not executed"
        self.write_hooks({"hooks": self.hooks})
        report = health._hooks(self.repo)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["events"]["Stop"]["linux_command_matches"])
        (self.repo / "scripts/agent/pre_tool_guard.py").unlink()
        self.assertFalse(health._hooks(self.repo)["events"]["PreToolUse"]["handler_exists"])

    def test_cli_has_json_and_nonzero_exit_for_failed_configuration(self) -> None:
        self.write_hooks({})
        output = io.StringIO()
        with patch.dict(health.os.environ, self.env, clear=True), redirect_stdout(output):
            self.assertEqual(health.main(["--repo-root", str(self.repo)]), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["schema"], "robotsim.infra-health.v1")
        self.assertTrue(report["read_only"])
        self.assertFalse(self.database.exists())

    def test_non_repository_fails_without_creating_any_files(self) -> None:
        before = self.snapshot()
        self.assertEqual(health.inspect(self.root, self.env)["checkout"]["status"], "FAIL")
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
