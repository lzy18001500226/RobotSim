#!/usr/bin/env python3
"""Send one best-effort RobotSim Codex notification through AgentMail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


STATUS_LABELS = {
    "ready_for_review": "Ready for review",
    "completed": "Completed",
    "blocked": "Blocked",
}
REQUIRED_SETTINGS = ("AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "ROBOTSIM_NOTIFY_TO")
AGENTMAIL_API_BASE = "https://api.agentmail.to"
REQUEST_TIMEOUT_SECONDS = 8
STALE_CLAIM_SECONDS = 3600

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(AGENTMAIL_API_KEY|(?:api[_ -]?key|access[_ -]?token|password|secret))\s*[:=]\s*([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_ISSUE_REFERENCE = re.compile(r"(?i)(?:issue[/#-]\s*|#)(\d+)\b")


@dataclass(frozen=True)
class Outcome:
    state: str
    message: str
    subject: str = ""
    body: str = ""


def _one_line(value: str, limit: int) -> str:
    value = value.replace("\x00", " ")
    value = _SECRET_ASSIGNMENT.sub(r"\1=[redacted]", value)
    value = _BEARER_VALUE.sub("Bearer [redacted]", value)
    return " ".join(value.split())[:limit]


def format_notification(
    status: str,
    task_id: str,
    summary: str,
    *,
    task: str | None = None,
    worker: str | None = None,
    branch: str | None = None,
    issue: str | None = None,
) -> tuple[str, str]:
    label = STATUS_LABELS[status]
    safe_summary = _one_line(summary, 120) or "Codex turn finished"
    subject = f"[RobotSim] {label}: {safe_summary[:96]}"
    fields = [
        f"Task: {_one_line(task or task_id, 100)}",
        f"Worker: {_one_line(worker or 'Codex', 40)}",
    ]
    if branch:
        fields.append(f"Branch: {_one_line(branch, 100)}")
    if issue:
        safe_issue = _one_line(issue, 40).lstrip("#")
        fields.append(f"Issue: #{safe_issue}")
    fields.append(f"Summary: {safe_summary}")
    body = "RobotSim Codex notification\n" + "\n".join(fields) + "\n"
    return subject, body


def _load_settings(environ: Mapping[str, str]) -> tuple[dict[str, str] | None, str]:
    missing = [name for name in REQUIRED_SETTINGS if not environ.get(name, "").strip()]
    if missing:
        return None, "missing settings: " + ", ".join(missing)
    return {
        "api_key": environ["AGENTMAIL_API_KEY"].strip(),
        "inbox_id": environ["AGENTMAIL_INBOX_ID"].strip(),
        "recipient": environ["ROBOTSIM_NOTIFY_TO"].strip(),
    }, ""


def _state_directory(environ: Mapping[str, str]) -> Path:
    configured = environ.get("ROBOTSIM_NOTIFY_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "robotsim" / "task-notifications"


def _delivery_digest(status: str, task_id: str) -> str:
    return hashlib.sha256(f"{status}\0{task_id}".encode("utf-8")).hexdigest()


def _claim_delivery(state_dir: Path, digest: str) -> tuple[Path | None, str]:
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    sent = state_dir / f"{digest}.sent"
    pending = state_dir / f"{digest}.pending"
    if sent.exists():
        return None, "duplicate"

    for attempt in range(2):
        try:
            descriptor = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                stale = time.time() - pending.stat().st_mtime > STALE_CLAIM_SECONDS
                if stale and attempt == 0:
                    pending.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            return None, "duplicate or already in progress"
        with os.fdopen(descriptor, "w", encoding="ascii") as claim:
            claim.write(str(int(time.time())))
        return pending, ""
    return None, "duplicate or already in progress"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward the inbox Bearer key to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _open_request(request: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(_NoRedirect())
    return opener.open(request, timeout=timeout)


def _send(
    subject: str,
    body: str,
    settings: Mapping[str, str],
    idempotency_key: str,
) -> None:
    inbox_path = urllib.parse.quote(settings["inbox_id"], safe="@.")
    url = f"{AGENTMAIL_API_BASE}/v0/inboxes/{inbox_path}/messages/send"
    payload = {
        "to": [settings["recipient"]],
        "subject": subject,
        "text": body,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        },
        method="POST",
    )
    with _open_request(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        status = getattr(response, "status", 200)
        if not 200 <= status < 300:
            raise urllib.error.HTTPError(url, status, "AgentMail send failed", response.headers, None)


def notify(
    status: str,
    task_id: str,
    summary: str,
    *,
    task: str | None = None,
    worker: str | None = None,
    branch: str | None = None,
    issue: str | None = None,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    subject, body = format_notification(
        status, task_id, summary, task=task, worker=worker, branch=branch, issue=issue
    )
    if not task_id.strip() or not summary.strip():
        return Outcome("skipped", "task ID and summary must be non-empty", subject, body)
    if dry_run:
        return Outcome("dry_run", "no AgentMail request was made", subject, body)

    env = os.environ if environ is None else environ
    settings, problem = _load_settings(env)
    if settings is None:
        return Outcome("skipped", problem, subject, body)

    digest = _delivery_digest(status, task_id)
    try:
        pending, problem = _claim_delivery(_state_directory(env), digest)
    except OSError as exc:
        return Outcome("failed", f"could not prepare duplicate state ({type(exc).__name__})", subject, body)
    if pending is None:
        return Outcome("duplicate", problem, subject, body)

    try:
        # The same key protects retries after an ambiguous network failure. AgentMail
        # retains successful send keys for 24 hours; the local marker is durable.
        _send(subject, body, settings, digest)
        os.replace(pending, pending.with_suffix(".sent"))
    except urllib.error.HTTPError as exc:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        return Outcome("failed", f"AgentMail HTTP request failed (HTTP {exc.code})", subject, body)
    except Exception as exc:  # Notification is best-effort; never fail the engineering task.
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        return Outcome("failed", f"AgentMail delivery failed ({type(exc).__name__})", subject, body)
    return Outcome("sent", "notification delivered", subject, body)


def _branch_at(cwd: object) -> str:
    if not isinstance(cwd, str) or not cwd:
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _issue_from_context(branch: str, summary: str) -> str:
    match = _ISSUE_REFERENCE.search(f"{branch} {summary}")
    return match.group(1) if match else ""


def _summary_from_stop(message: object) -> str:
    if not isinstance(message, str) or not message.strip():
        return "Codex turn completed"
    for line in message.splitlines():
        candidate = _one_line(line.strip(" #*-`\t"), 120)
        if candidate and candidate.casefold().rstrip(":") not in {"summary", "result", "completed", "final"}:
            return candidate
    return "Codex turn completed"


def _stop_event_id(payload: Mapping[str, object]) -> str:
    session_id = payload.get("session_id")
    turn_id = payload.get("turn_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return ""
    if isinstance(turn_id, str) and turn_id.strip():
        event_data = f"{session_id}\0{turn_id}"
    else:
        # Older payloads may omit Codex's current turn_id extension. Transcript
        # size plus the last answer distinguishes consecutive fallback events.
        transcript = payload.get("transcript_path")
        transcript_size = ""
        if isinstance(transcript, str) and transcript:
            try:
                transcript_size = str(Path(transcript).stat().st_size)
            except OSError:
                pass
        event_data = f"{session_id}\0{transcript_size}\0{payload.get('last_assistant_message', '')}"
    return "local-stop-" + hashlib.sha256(event_data.encode("utf-8")).hexdigest()


def process_stop_payload(
    payload: object,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    if not isinstance(payload, dict) or payload.get("hook_event_name") != "Stop":
        return Outcome("skipped", "not a Codex Stop event")
    if payload.get("stop_hook_active") is True:
        return Outcome("skipped", "Stop hook recursion guard is active")
    env = os.environ if environ is None else environ
    if not dry_run and env.get("ROBOTSIM_LOCAL_STOP_HOOK", "").strip() != "1":
        return Outcome("skipped", "local Stop notifications are not enabled")
    task_id = _stop_event_id(payload)
    if not task_id:
        return Outcome("skipped", "Stop payload has no session identifier")

    summary = _summary_from_stop(payload.get("last_assistant_message"))
    branch = _branch_at(payload.get("cwd"))
    issue = _issue_from_context(branch, summary)
    return notify(
        "completed",
        task_id,
        summary,
        task="Local Codex turn",
        worker="Local Codex",
        branch=branch,
        issue=issue,
        dry_run=dry_run,
        environ=environ,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "event",
        choices=(*STATUS_LABELS, "stop"),
        help="terminal Cloud outcome, or 'stop' for a local Codex Stop payload on stdin",
    )
    parser.add_argument("--task-id", help="stable task/turn identifier used for duplicate protection")
    parser.add_argument("--task", help="short task label to include in the message")
    parser.add_argument("--summary", help="short, non-sensitive final summary")
    parser.add_argument("--worker", default="Codex", help="worker label, for example 'Codex Cloud'")
    parser.add_argument("--branch", help="branch name; otherwise detected from the current checkout")
    parser.add_argument("--issue", help="issue number, when available")
    parser.add_argument("--dry-run", action="store_true", help="format the message without sending mail")
    return parser


def _print_outcome(outcome: Outcome) -> None:
    if outcome.state == "dry_run":
        print(f"Dry run: {outcome.message}\nSubject: {outcome.subject}\n\n{outcome.body}", end="")
    elif outcome.state in {"sent", "duplicate", "skipped"}:
        print(f"notify_task: {outcome.state}: {outcome.message}")
    else:
        print(f"notify_task: {outcome.state}: {outcome.message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.event == "stop":
        try:
            payload = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError):
            outcome = Outcome("skipped", "invalid Stop hook JSON payload")
        else:
            outcome = process_stop_payload(payload, dry_run=args.dry_run)
    elif not args.task_id or not args.summary:
        _parser().error("Cloud outcome events require --task-id and --summary")
    else:
        branch = args.branch or _branch_at(os.getcwd())
        issue = args.issue or _issue_from_context(branch, args.summary)
        outcome = notify(
            args.event,
            args.task_id,
            args.summary,
            task=args.task,
            worker=args.worker,
            branch=branch,
            issue=issue,
            dry_run=args.dry_run,
        )
    if args.event == "stop" and not args.dry_run:
        # Codex Stop hooks accept empty stdout or a Stop result object. Keep
        # routine outcomes silent; only diagnostics use stderr.
        if outcome.state == "failed":
            print(f"notify_task: {outcome.state}: {outcome.message}", file=sys.stderr)
    else:
        _print_outcome(outcome)
    # Mail delivery is best-effort and must never keep a completed turn open.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
