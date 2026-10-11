#!/usr/bin/env python3
"""Durable, redacted local task lifecycle and notification outbox state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


TASK_STATES = {"RUNNING", "PAUSED", "RESUMED", "COMPLETED", "FAILED", "BLOCKED", "CANCELLED", "UNCERTAIN"}
TURN_STATES = {"RUNNING", "PAUSED", "ENDED"}
PUBLICATION_STATES = {"pending", "verified", "blocked", "not_attempted", "failed"}
DELIVERY_STATES = {"disabled", "pending", "sending", "sent", "failed", "uncertain", "unavailable"}
EMAIL_MAX_ATTEMPTS = 4
EMAIL_RETRY_SECONDS = (60, 300, 1800)
EMAIL_LEASE_SECONDS = 3605
PUBLICATION_MAX_ATTEMPTS = 4
PUBLICATION_RETRY_SECONDS = (60, 300, 1800)
PUBLICATION_LEASE_SECONDS = 3605
_SECRET_TEXT = re.compile(
    r"(?i)\b(AGENTMAIL_API_KEY|GH_TOKEN|GITHUB_TOKEN|api[_ -]?key|access[_ -]?token|password|secret)\s*[:=]\s*[^\s,;]+"
)
_BEARER_TEXT = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_TOKEN_TEXT = re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|(?:gh[pousr])_[A-Za-z0-9]{20,})")
_LOCAL_PATH_TEXT = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|/home/|/tmp/|/mnt/|/workspace/)[^\s,;]+"
)


class EventConflict(ValueError):
    """A stable task event ID was reused with different task data."""


def _safe_detail(value: object, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    text = _SECRET_TEXT.sub(r"\1=[redacted]", text)
    text = _BEARER_TEXT.sub("Bearer [redacted]", text)
    text = _TOKEN_TEXT.sub("[redacted credential]", text)
    text = _LOCAL_PATH_TEXT.sub("[local path omitted]", text)
    return text[:limit]


def _database_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get("ROBOTSIM_TASK_NOTIFICATION_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    state_home = env.get("XDG_STATE_HOME", "").strip()
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return root / "robotsim" / "task-notifications.sqlite3"


def _connect(environ: Mapping[str, str] | None = None) -> sqlite3.Connection:
    path = _database_path(environ)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        "CREATE TABLE IF NOT EXISTS turns ("
        "session_id TEXT NOT NULL, turn_id TEXT NOT NULL, turn_state TEXT NOT NULL, "
        "task_state TEXT NOT NULL, prompt_present INTEGER NOT NULL DEFAULT 0, "
        "prompt_length INTEGER, cwd_present INTEGER NOT NULL DEFAULT 0, "
        "completion_present INTEGER NOT NULL DEFAULT 0, completion_length INTEGER, "
        "terminal_evidence_present INTEGER NOT NULL DEFAULT 0, resume_candidate INTEGER NOT NULL DEFAULT 0, "
        "stop_hook_active INTEGER, first_seen TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "PRIMARY KEY(session_id, turn_id));"
        "CREATE TABLE IF NOT EXISTS turn_events ("
        "event_key TEXT PRIMARY KEY, event_type TEXT NOT NULL, session_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, occurred_at TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS outcomes ("
        "event_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, task_id TEXT NOT NULL, "
        "attempt_id TEXT NOT NULL, worker TEXT NOT NULL, task_kind TEXT NOT NULL, "
        "execution_outcome TEXT NOT NULL, engineering_acceptance TEXT NOT NULL, "
        "publication_state TEXT NOT NULL, publication_detail TEXT NOT NULL, "
        "session_id TEXT, turn_id TEXT, event_json TEXT NOT NULL, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS deliveries ("
        "event_id TEXT NOT NULL REFERENCES outcomes(event_id), channel TEXT NOT NULL, "
        "state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL, "
        "lease_until REAL, last_error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL, "
        "PRIMARY KEY(event_id, channel));"
        "CREATE TABLE IF NOT EXISTS publication_jobs ("
        "event_id TEXT PRIMARY KEY REFERENCES outcomes(event_id), state TEXT NOT NULL, "
        "attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL, lease_until REAL, "
        "last_error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL);"
    )
    turn_columns = {row["name"] for row in connection.execute("PRAGMA table_info(turns)")}
    for name, definition in (
        ("cwd_present", "INTEGER NOT NULL DEFAULT 0"),
        ("completion_present", "INTEGER NOT NULL DEFAULT 0"),
        ("completion_length", "INTEGER"),
        ("terminal_evidence_present", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if name not in turn_columns:
            connection.execute(f"ALTER TABLE turns ADD COLUMN {name} {definition}")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _bounded_id(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 256:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    return value


def record_turn_event(payload: object, *, environ: Mapping[str, str] | None = None) -> bool:
    """Persist only event metadata; prompt and assistant text are never stored."""
    if not isinstance(payload, dict):
        return False
    event_type = payload.get("hook_event_name")
    if event_type not in {"UserPromptSubmit", "Interrupt", "Stop"}:
        return False
    session_id = _bounded_id(payload.get("session_id"))
    turn_id = _bounded_id(payload.get("turn_id"))
    if not session_id or not turn_id:
        return False

    now = _now()
    event_key = hashlib.sha256(f"{event_type}\0{session_id}\0{turn_id}".encode()).hexdigest()
    prompt = payload.get("prompt") if event_type == "UserPromptSubmit" else None
    prompt_present = isinstance(prompt, str)
    prompt_length = len(prompt) if prompt_present else None
    cwd_present = isinstance(payload.get("cwd"), str) and bool(payload.get("cwd"))
    completion = payload.get("last_assistant_message") if event_type == "Stop" else None
    completion_present = isinstance(completion, str)
    completion_length = len(completion) if completion_present else None
    terminal_evidence_present = bool(
        completion_present
        and ("robotsim.task-closeout.v1" in completion or "robotsim-research-completion:v1" in completion)
    )
    stop_active = payload.get("stop_hook_active") is True if event_type == "Stop" else None
    turn_state = {"UserPromptSubmit": "RUNNING", "Interrupt": "PAUSED", "Stop": "ENDED"}[event_type]
    if event_type == "Stop" and stop_active:
        turn_state = "RUNNING"

    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous_pause = False
        if event_type == "UserPromptSubmit":
            previous_pause = conn.execute(
                "SELECT 1 FROM turns WHERE session_id=? AND turn_state='PAUSED' LIMIT 1",
                (session_id,),
            ).fetchone() is not None
        conn.execute(
            "INSERT OR IGNORE INTO turn_events(event_key,event_type,session_id,turn_id,occurred_at) "
            "VALUES(?,?,?,?,?)",
            (event_key, event_type, session_id, turn_id, now),
        )
        existing = conn.execute(
            "SELECT first_seen FROM turns WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        first_seen = existing["first_seen"] if existing else now
        task_state = (
            "RESUMED" if event_type == "UserPromptSubmit" and previous_pause
            else "RUNNING" if event_type == "UserPromptSubmit" or stop_active
            else "PAUSED" if event_type == "Interrupt"
            else "UNCERTAIN"
        )
        conn.execute(
            "INSERT INTO turns(session_id,turn_id,turn_state,task_state,prompt_present,prompt_length,"
            "cwd_present,completion_present,completion_length,terminal_evidence_present,resume_candidate,"
            "stop_hook_active,first_seen,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(session_id,turn_id) DO UPDATE SET turn_state=excluded.turn_state, "
            "task_state=excluded.task_state, "
            "prompt_present=MAX(turns.prompt_present,excluded.prompt_present), "
            "prompt_length=COALESCE(excluded.prompt_length,turns.prompt_length), "
            "cwd_present=MAX(turns.cwd_present,excluded.cwd_present), "
            "completion_present=MAX(turns.completion_present,excluded.completion_present), "
            "completion_length=COALESCE(excluded.completion_length,turns.completion_length), "
            "terminal_evidence_present=MAX(turns.terminal_evidence_present,excluded.terminal_evidence_present), "
            "resume_candidate=MAX(turns.resume_candidate,excluded.resume_candidate), "
            "stop_hook_active=COALESCE(excluded.stop_hook_active,turns.stop_hook_active), "
            "updated_at=excluded.updated_at",
            (
                session_id,
                turn_id,
                turn_state,
                task_state,
                int(prompt_present),
                prompt_length,
                int(cwd_present),
                int(completion_present),
                completion_length,
                int(terminal_evidence_present),
                int(previous_pause),
                None if stop_active is None else int(stop_active),
                first_seen,
                now,
            ),
        )
    return True


def record_terminal_outcome(
    event: Mapping[str, object],
    *,
    publication_state: str = "pending",
    publication_detail: str = "",
    notifications_enabled: bool = False,
    session_id: str | None = None,
    turn_id: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Persist one normalized closeout separately from publication and channels."""
    if publication_state not in PUBLICATION_STATES:
        raise ValueError("unsupported publication state")
    event_data = dict(event)
    event_json = json.dumps(event_data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    event_id = str(event_data["event_id"])
    payload_digest = hashlib.sha256(event_json.encode("utf-8")).hexdigest()
    now = _now()
    env = os.environ if environ is None else environ
    email_configured = all(env.get(key, "").strip() for key in (
        "AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "ROBOTSIM_NOTIFY_TO"
    ))
    mail_state = "pending" if notifications_enabled and email_configured else (
        "disabled" if not notifications_enabled else "unavailable"
    )
    safe_detail = _safe_detail(publication_detail)

    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT payload_sha256 FROM outcomes WHERE event_id=?", (event_id,)
        ).fetchone()
        if existing and existing["payload_sha256"] != payload_digest:
            raise EventConflict("task outcome identity was reused with different data")
        conn.execute(
            "INSERT OR IGNORE INTO outcomes(event_id,payload_sha256,task_id,attempt_id,worker,task_kind,"
            "execution_outcome,engineering_acceptance,publication_state,publication_detail,session_id,turn_id,"
            "event_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                payload_digest,
                str(event_data["task_id"]),
                str(event_data["attempt_id"]),
                str(event_data["worker"]),
                str(event_data["task_kind"]),
                str(event_data["status"]).upper(),
                "UNVERIFIED",
                publication_state,
                safe_detail,
                _bounded_id(session_id),
                _bounded_id(turn_id),
                event_json,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO deliveries(event_id,channel,state,updated_at) VALUES(?,?,?,?)",
            (event_id, "agentmail", mail_state, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO deliveries(event_id,channel,state,updated_at) VALUES(?,?,?,?)",
            (event_id, "telephone", "unavailable", now),
        )
        publication_job_state = "pending" if publication_state == "pending" else (
            "disabled" if publication_state == "not_attempted" else publication_state
        )
        conn.execute(
            "INSERT OR IGNORE INTO publication_jobs(event_id,state,updated_at) VALUES(?,?,?)",
            (event_id, publication_job_state, now),
        )
        if session_id and turn_id:
            conn.execute(
                "UPDATE turns SET task_state=?,updated_at=? WHERE session_id=? AND turn_id=?",
                (str(event_data["status"]).upper(), now, session_id, turn_id),
            )
    return True


def set_publication_state(
    event_id: str,
    state: str,
    detail: str = "",
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    if state not in PUBLICATION_STATES:
        raise ValueError("unsupported publication state")
    safe_detail = _safe_detail(detail)
    with _connect(environ) as conn:
        conn.execute(
            "UPDATE outcomes SET publication_state=?,publication_detail=?,updated_at=? WHERE event_id=?",
            (state, safe_detail, _now(), event_id),
        )
        job_state = "pending" if state == "pending" else state
        conn.execute(
            "UPDATE publication_jobs SET state=?,last_error=?,next_attempt=NULL,lease_until=NULL,updated_at=? "
            "WHERE event_id=? AND state!='sending'",
            (job_state, safe_detail, _now(), event_id),
        )


def claim_publication(
    environ: Mapping[str, str] | None = None,
    *,
    now: float | None = None,
) -> dict[str, object] | None:
    """Atomically claim one due GitHub closeout publication for a bounded worker attempt."""
    current = time.time() if now is None else now
    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE publication_jobs SET state='pending',next_attempt=?,lease_until=NULL,updated_at=? "
            "WHERE state='sending' AND lease_until<? AND attempts<?",
            (current, _now(), current, PUBLICATION_MAX_ATTEMPTS),
        )
        conn.execute(
            "UPDATE publication_jobs SET state='failed',last_error='publication worker lease expired after retry limit',"
            "lease_until=NULL,updated_at=? WHERE state='sending' AND lease_until<? AND attempts>=?",
            (_now(), current, PUBLICATION_MAX_ATTEMPTS),
        )
        row = conn.execute(
            "SELECT o.*,j.attempts FROM outcomes o JOIN publication_jobs j USING(event_id) "
            "WHERE j.state='pending' AND j.attempts<? AND COALESCE(j.next_attempt,0)<=? "
            "ORDER BY o.created_at LIMIT 1",
            (PUBLICATION_MAX_ATTEMPTS, current),
        ).fetchone()
        if row is None:
            return None
        attempt = int(row["attempts"]) + 1
        conn.execute(
            "UPDATE publication_jobs SET state='sending',attempts=?,lease_until=?,updated_at=? "
            "WHERE event_id=? AND state='pending'",
            (attempt, current + PUBLICATION_LEASE_SECONDS, _now(), row["event_id"]),
        )
        return {
            "event_id": row["event_id"],
            "event": json.loads(row["event_json"]),
            "attempt": attempt,
        }


def finish_publication(
    event_id: str,
    result: str,
    detail: str = "",
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
) -> None:
    if result not in {"verified", "blocked", "failed", "not_attempted"}:
        raise ValueError("unsupported publication result")
    current = time.time() if now is None else now
    error = _safe_detail(detail)
    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT attempts FROM publication_jobs WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            return
        attempts = int(row["attempts"])
        if result == "failed" and attempts < PUBLICATION_MAX_ATTEMPTS:
            delay = PUBLICATION_RETRY_SECONDS[min(attempts - 1, len(PUBLICATION_RETRY_SECONDS) - 1)]
            job_state, next_attempt = "pending", current + delay
            outcome_state = "pending"
        else:
            job_state = result
            next_attempt = None
            outcome_state = result
        conn.execute(
            "UPDATE publication_jobs SET state=?,next_attempt=?,lease_until=NULL,last_error=?,updated_at=? "
            "WHERE event_id=?",
            (job_state, next_attempt, error, _now(), event_id),
        )
        conn.execute(
            "UPDATE outcomes SET publication_state=?,publication_detail=?,updated_at=? WHERE event_id=?",
            (outcome_state, error, _now(), event_id),
        )


def claim_agentmail(
    environ: Mapping[str, str] | None = None,
    *,
    now: float | None = None,
    event_id: str | None = None,
) -> dict[str, object] | None:
    """Atomically claim one due email event for a bounded worker attempt."""
    current = time.time() if now is None else now
    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE deliveries SET state='pending',next_attempt=?,lease_until=NULL,updated_at=? "
            "WHERE channel='agentmail' AND state='sending' AND lease_until<? AND attempts<?",
            (current, _now(), current, EMAIL_MAX_ATTEMPTS),
        )
        conn.execute(
            "UPDATE deliveries SET state='failed',last_error='delivery worker lease expired after retry limit',"
            "lease_until=NULL,updated_at=? WHERE channel='agentmail' AND state='sending' "
            "AND lease_until<? AND attempts>=?",
            (_now(), current, EMAIL_MAX_ATTEMPTS),
        )
        sql = (
            "SELECT o.*,d.attempts,d.state AS delivery_state FROM outcomes o "
            "JOIN deliveries d USING(event_id) WHERE d.channel='agentmail' AND d.state='pending' "
            "AND d.attempts<? AND COALESCE(d.next_attempt,0)<=?"
        )
        params: tuple[object, ...] = (EMAIL_MAX_ATTEMPTS, current)
        if event_id:
            sql += " AND o.event_id=?"
            params += (event_id,)
        sql += " ORDER BY o.created_at LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"]) + 1
        conn.execute(
            "UPDATE deliveries SET state='sending',attempts=?,lease_until=?,updated_at=? "
            "WHERE event_id=? AND channel='agentmail' AND state='pending'",
            (attempts, current + EMAIL_LEASE_SECONDS, _now(), row["event_id"]),
        )
        return {
            "event_id": row["event_id"],
            "event": json.loads(row["event_json"]),
            "publication_state": row["publication_state"],
            "publication_detail": row["publication_detail"],
            "attempt": attempts,
        }


def finish_agentmail(
    event_id: str,
    result: str,
    detail: str = "",
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
) -> None:
    if result not in {"sent", "failed", "uncertain", "unavailable"}:
        raise ValueError("unsupported AgentMail result")
    current = time.time() if now is None else now
    error = _safe_detail(detail, 200)
    with _connect(environ) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT attempts FROM deliveries WHERE event_id=? AND channel='agentmail'",
            (event_id,),
        ).fetchone()
        if row is None:
            return
        attempts = int(row["attempts"])
        if result == "failed" and attempts < EMAIL_MAX_ATTEMPTS:
            delay = EMAIL_RETRY_SECONDS[min(attempts - 1, len(EMAIL_RETRY_SECONDS) - 1)]
            state, next_attempt = "pending", current + delay
        else:
            state, next_attempt = result, None
        conn.execute(
            "UPDATE deliveries SET state=?,next_attempt=?,lease_until=NULL,last_error=?,updated_at=? "
            "WHERE event_id=? AND channel='agentmail'",
            (state, next_attempt, error, _now(), event_id),
        )


def delivery_state(
    event_id: str,
    channel: str,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    with _connect(environ) as conn:
        row = conn.execute(
            "SELECT state FROM deliveries WHERE event_id=? AND channel=?",
            (event_id, channel),
        ).fetchone()
    return str(row["state"]) if row else None


def next_agentmail_delay(
    environ: Mapping[str, str] | None = None,
    *,
    now: float | None = None,
) -> float | None:
    current = time.time() if now is None else now
    with _connect(environ) as conn:
        row = conn.execute(
            "SELECT MIN(COALESCE(next_attempt,0)) AS due FROM deliveries "
            "WHERE channel='agentmail' AND state='pending' AND attempts<?",
            (EMAIL_MAX_ATTEMPTS,),
        ).fetchone()
    if row is None or row["due"] is None:
        return None
    return max(0.0, float(row["due"]) - current)


def next_publication_delay(
    environ: Mapping[str, str] | None = None,
    *,
    now: float | None = None,
) -> float | None:
    current = time.time() if now is None else now
    with _connect(environ) as conn:
        row = conn.execute(
            "SELECT MIN(COALESCE(next_attempt,0)) AS due FROM publication_jobs "
            "WHERE state='pending' AND attempts<?",
            (PUBLICATION_MAX_ATTEMPTS,),
        ).fetchone()
    if row is None or row["due"] is None:
        return None
    return max(0.0, float(row["due"]) - current)


def snapshot(environ: Mapping[str, str] | None = None) -> dict[str, list[dict[str, object]]]:
    """Read redacted lifecycle and delivery state for tests/support diagnostics."""
    with _connect(environ) as conn:
        turns = [dict(row) for row in conn.execute("SELECT * FROM turns ORDER BY first_seen,session_id,turn_id")]
        outcomes = [dict(row) for row in conn.execute("SELECT * FROM outcomes ORDER BY created_at,event_id")]
        deliveries = [dict(row) for row in conn.execute("SELECT * FROM deliveries ORDER BY event_id,channel")]
        publications = [dict(row) for row in conn.execute("SELECT * FROM publication_jobs ORDER BY event_id")]
    for row in outcomes:
        row.pop("event_json", None)
    return {
        "turns": turns,
        "outcomes": outcomes,
        "deliveries": deliveries,
        "publications": publications,
    }
