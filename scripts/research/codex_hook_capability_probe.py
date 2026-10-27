#!/usr/bin/env python3
"""Observe Codex prompt/stop hooks without sending notifications or storing prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DIR = Path("/tmp/robotsim-codex-hook-probe")
ACTION_PREFIX = re.compile(
    r"^\s*(?:(?:please|kindly|can you|could you)\s+|(?:\u8bf7|\u9ebb\u70e6|\u5e2e\u6211)\s*)?"
    r"(?:read|compare|check|inspect|review|run|test|verify|create|write|edit|modify|implement|analyze|investigate|"
    r"\u68c0\u67e5|\u5ba1\u67e5|\u8fd0\u884c|\u6d4b\u8bd5|\u9a8c\u8bc1|\u521b\u5efa|\u7f16\u5199|"
    r"\u4fee\u6539|\u5b9e\u73b0|\u5206\u6790|\u6392\u67e5|\u8c03\u67e5|\u6574\u7406|\u751f\u6210)",
    re.IGNORECASE,
)
QUESTION_MARKERS = (
    "?",
    "\uff1f",
    "\u6709\u4ec0\u4e48",
    "\u662f\u4ec0\u4e48",
    "\u4e3a\u4ec0\u4e48",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u533a\u522b",
    "\u89e3\u91ca",
    "\u4ecb\u7ecd",
)


def classify_prompt(prompt: str) -> tuple[str, str]:
    if ACTION_PREFIX.search(prompt):
        return "TASK", "explicit_action_prefix"
    if any(marker in prompt for marker in QUESTION_MARKERS):
        return "CHAT", "question_or_explanation_request"
    return "UNCERTAIN", "no_clear_action_or_question_signal"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ProbeStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.database = directory / "events.sqlite3"
        self.log = directory / "events.jsonl"
        fd = os.open(self.database, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with sqlite3.connect(self.database, timeout=5) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "seq INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, "
                "session_id TEXT, turn_id TEXT, record_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS event_lookup "
                "ON events(event_type, session_id, turn_id)"
            )

    def process(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"status": "invalid_payload"}

        event_type = payload.get("hook_event_name")
        if event_type not in {"UserPromptSubmit", "Stop", "SubagentStop"}:
            return {"status": "ignored_event"}

        session_id = payload.get("session_id") if isinstance(payload.get("session_id"), str) else None
        turn_id = payload.get("turn_id") if isinstance(payload.get("turn_id"), str) else None
        prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else None
        is_subagent = event_type == "SubagentStop" or isinstance(payload.get("agent_id"), str)
        scope = "subagent" if is_subagent else "main_thread_event"

        record: dict[str, Any] = {
            "event_type": event_type,
            "hook_event_name": event_type,
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp_utc": _timestamp(),
            "cwd": payload.get("cwd") if isinstance(payload.get("cwd"), str) else None,
            "scope": scope,
            "prompt_origin": "not_exposed_by_hook_schema",
            "prompt_text_stored": False,
            "local_stop_opt_in_enabled": os.environ.get("ROBOTSIM_LOCAL_STOP_HOOK", "").strip() == "1",
        }
        if event_type == "UserPromptSubmit":
            record["prompt_present"] = prompt is not None
            record["prompt_length_chars"] = len(prompt) if prompt is not None else None
            record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt is not None else None
            record["classification"], record["classification_reason"] = classify_prompt(prompt or "")
            record["main_thread"] = not is_subagent

        if event_type == "Stop":
            completion_message = payload.get("last_assistant_message")
            has_completion_message = isinstance(completion_message, str)
            record["stop_hook_active"] = payload.get("stop_hook_active") is True
            record["main_thread"] = not is_subagent
            record["completion_message_present"] = has_completion_message
            record["completion_message_length_chars"] = (
                len(completion_message) if has_completion_message else None
            )
            record["completion_status"] = "unverified"
            record["notification_eligibility"] = "unverified"

        with sqlite3.connect(self.database, timeout=5) as conn:
            conn.execute("BEGIN IMMEDIATE")
            matched_prompt: dict[str, Any] | None = None
            duplicate_stop = False
            if event_type == "Stop" and session_id and turn_id:
                row = conn.execute(
                    "SELECT record_json FROM events WHERE event_type='UserPromptSubmit' "
                    "AND session_id=? AND turn_id=? ORDER BY seq DESC LIMIT 1",
                    (session_id, turn_id),
                ).fetchone()
                if row:
                    matched_prompt = json.loads(row[0])
                duplicate_stop = conn.execute(
                    "SELECT 1 FROM events WHERE event_type='Stop' AND session_id=? AND turn_id=? LIMIT 1",
                    (session_id, turn_id),
                ).fetchone() is not None

            record["matched_prompt_submit"] = matched_prompt is not None
            record["duplicate_stop"] = duplicate_stop if event_type == "Stop" else False
            if event_type == "Stop" and matched_prompt is not None:
                record["classification"] = matched_prompt.get("classification", "UNCERTAIN")
            record["simulated_task_turn_candidate"] = bool(
                event_type == "Stop"
                and not is_subagent
                and matched_prompt is not None
                and matched_prompt.get("classification") == "TASK"
                and not duplicate_stop
                and payload.get("stop_hook_active") is not True
            )

            encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            conn.execute(
                "INSERT INTO events(event_type, session_id, turn_id, record_json) VALUES (?, ?, ?, ?)",
                (event_type, session_id, turn_id, encoded),
            )
            conn.commit()

        line = (encoded + "\n").encode("utf-8")
        fd = os.open(self.log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, line)
        finally:
            os.close(fd)
        return record


def _sample(event_type: str, session: str, turn: str, **fields: Any) -> dict[str, Any]:
    return {
        "hook_event_name": event_type,
        "session_id": session,
        "turn_id": turn,
        "cwd": "/tmp/codex-hook-probe-fixture",
        **fields,
    }


def self_test() -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="codex-hook-probe-") as temp:
        store = ProbeStore(Path(temp))
        chat = "MuJoCo \u548c Gazebo \u6709\u4ec0\u4e48\u4e3b\u8981\u533a\u522b\uff1f"
        task = "\u68c0\u67e5\u5f53\u524d\u4ed3\u5e93 README \u4e0e\u5b9e\u9645\u9876\u5c42\u76ee\u5f55\u662f\u5426\u4e00\u81f4\uff0c\u7ed9\u51fa\u53ea\u8bfb\u5ba1\u67e5\u7ed3\u679c\u3002"
        read_task = "Read the current README and compare it with the actual top-level repository directories. Report the findings without editing files."
        continuation = "Run one more pass over the task."

        chat_prompt = store.process(_sample("UserPromptSubmit", "session-a", "turn-chat", prompt=chat))
        chat_stop = store.process(_sample("Stop", "session-a", "turn-chat", stop_hook_active=False))
        task_prompt = store.process(_sample("UserPromptSubmit", "session-a", "turn-task", prompt=task))
        task_stop = store.process(
            _sample("Stop", "session-a", "turn-task", stop_hook_active=False, last_assistant_message="synthetic")
        )
        read_prompt = store.process(_sample("UserPromptSubmit", "session-read", "turn-read", prompt=read_task))
        read_stop = store.process(
            _sample("Stop", "session-read", "turn-read", stop_hook_active=False, last_assistant_message="BLOCKED")
        )
        repeated = store.process(_sample("Stop", "session-a", "turn-task", stop_hook_active=False))
        other_session = store.process(_sample("Stop", "session-b", "turn-task", stop_hook_active=False))
        subagent = store.process(
            _sample("SubagentStop", "session-a", "turn-task", agent_id="agent-fixture", agent_type="reviewer")
        )
        auto_prompt = store.process(
            _sample("UserPromptSubmit", "session-a", "turn-auto", prompt=continuation)
        )
        for session in ("session-c", "session-d"):
            store.process(_sample("UserPromptSubmit", session, "turn-parallel", prompt=task))
        with ThreadPoolExecutor(max_workers=2) as executor:
            parallel_stops = list(
                executor.map(
                    lambda session: store.process(_sample("Stop", session, "turn-parallel", stop_hook_active=False)),
                    ("session-c", "session-d"),
                )
            )

        cli_directory = Path(temp) / "cli-contract"
        cli_env = dict(os.environ)
        cli_env["ROBOTSIM_HOOK_PROBE_DIR"] = str(cli_directory)
        cli_prompt = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve())],
            input=json.dumps(_sample("UserPromptSubmit", "session-cli", "turn-cli", prompt=chat)).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=cli_env,
            timeout=3,
            check=False,
        )
        cli_stop = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve())],
            input=json.dumps(_sample("Stop", "session-cli", "turn-cli", stop_hook_active=False)).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=cli_env,
            timeout=3,
            check=False,
        )
        cli_stop_response = json.loads(cli_stop.stdout.decode("utf-8"))

        lines = store.log.read_text(encoding="utf-8").splitlines()
        persisted = "\n".join(lines)
        assert chat_prompt["classification"] == "CHAT"
        assert chat_stop["matched_prompt_submit"] is True
        assert chat_stop["simulated_task_turn_candidate"] is False
        assert task_prompt["classification"] == "TASK"
        assert task_stop["matched_prompt_submit"] is True
        assert task_stop["hook_event_name"] == "Stop"
        assert task_stop["simulated_task_turn_candidate"] is True
        assert task_stop["completion_message_present"] is True
        assert task_stop["completion_message_length_chars"] == len("synthetic")
        assert read_prompt["classification"] == "TASK"
        assert read_stop["completion_status"] == "unverified"
        assert read_stop["notification_eligibility"] == "unverified"
        assert chat_stop["completion_message_present"] is False
        assert repeated["duplicate_stop"] is True
        assert repeated["simulated_task_turn_candidate"] is False
        assert other_session["matched_prompt_submit"] is False
        assert subagent["scope"] == "subagent"
        assert subagent["simulated_task_turn_candidate"] is False
        assert auto_prompt["prompt_origin"] == "not_exposed_by_hook_schema"
        assert all(item["matched_prompt_submit"] for item in parallel_stops)
        assert all(item["simulated_task_turn_candidate"] for item in parallel_stops)
        assert cli_prompt.returncode == 0 and cli_prompt.stdout == b""
        assert cli_stop.returncode == 0 and cli_stop_response == {"continue": True}
        assert chat not in persisted and task not in persisted and read_task not in persisted and continuation not in persisted
        assert "synthetic" not in persisted and "BLOCKED" not in persisted
        assert store.directory.stat().st_mode & 0o077 == 0
        assert store.database.stat().st_mode & 0o077 == 0
        assert store.log.stat().st_mode & 0o077 == 0

    return {
        "prompt_classification_and_turn_correlation": "PASS",
        "duplicate_stop_suppression": "PASS",
        "session_isolation": "PASS",
        "concurrent_session_isolation": "PASS",
        "subagent_exclusion": "PASS",
        "command_hook_output_contract": "PASS",
        "completion_status_not_inferred": "PASS",
        "prompt_redaction": "PASS",
        "automatic_continuation_provenance": "NOT_VERIFIED_SCHEMA_HAS_NO_SOURCE_FIELD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return 0

    raw = sys.stdin.buffer.read(1_048_577)
    payload: object = None
    try:
        if len(raw) <= 1_048_576:
            payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    event_type = payload.get("hook_event_name") if isinstance(payload, dict) else None
    try:
        directory = Path(os.environ.get("ROBOTSIM_HOOK_PROBE_DIR", str(DEFAULT_DIR)))
        if payload is not None:
            ProbeStore(directory).process(payload)
    except Exception:
        print("Codex hook probe failed; no notification was sent.", file=sys.stderr)

    if event_type in {"Stop", "SubagentStop"}:
        sys.stdout.write('{"continue":true}\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
