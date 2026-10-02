#!/usr/bin/env python3
"""Send one best-effort RobotSim task status email through configured SMTP."""

from __future__ import annotations

import argparse
import hashlib
import os
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Mapping


STATUS_LABELS = {
    "ready_for_review": "Ready for review",
    "completed": "Completed",
    "blocked": "Blocked",
}
REQUIRED_SETTINGS = (
    "ROBOTSIM_SMTP_HOST",
    "ROBOTSIM_SMTP_FROM",
    "ROBOTSIM_NOTIFY_TO",
)
MAX_TIMEOUT = 60
STALE_CLAIM_SECONDS = 3600


@dataclass(frozen=True)
class Outcome:
    state: str
    message: str
    subject: str = ""
    body: str = ""


def _one_line(value: str, limit: int) -> str:
    return " ".join(value.replace("\x00", "").split())[:limit]


def format_notification(status: str, task_id: str, summary: str) -> tuple[str, str]:
    label = STATUS_LABELS[status]
    safe_task_id = _one_line(task_id, 120)
    safe_summary = _one_line(summary, 180)
    subject = f"[RobotSim] {label}: {safe_summary}"
    body = f"RobotSim task status: {label}\nTask: {safe_task_id}\nSummary: {safe_summary}\n"
    return subject, body


def _load_settings(environ: Mapping[str, str]) -> tuple[dict[str, object] | None, str]:
    missing = [name for name in REQUIRED_SETTINGS if not environ.get(name, "").strip()]
    username = environ.get("ROBOTSIM_SMTP_USERNAME", "").strip()
    password = environ.get("ROBOTSIM_SMTP_PASSWORD", "")
    if bool(username) != bool(password):
        missing.append("ROBOTSIM_SMTP_USERNAME and ROBOTSIM_SMTP_PASSWORD (set both or neither)")
    if missing:
        return None, "missing settings: " + ", ".join(missing)

    try:
        port = int(environ.get("ROBOTSIM_SMTP_PORT", "587"))
        timeout = float(environ.get("ROBOTSIM_SMTP_TIMEOUT", "10"))
    except ValueError:
        return None, "ROBOTSIM_SMTP_PORT and ROBOTSIM_SMTP_TIMEOUT must be numeric"
    if not 1 <= port <= 65535:
        return None, "ROBOTSIM_SMTP_PORT must be between 1 and 65535"
    if not 0 < timeout <= MAX_TIMEOUT:
        return None, f"ROBOTSIM_SMTP_TIMEOUT must be greater than 0 and at most {MAX_TIMEOUT} seconds"

    return {
        "host": environ["ROBOTSIM_SMTP_HOST"].strip(),
        "port": port,
        "timeout": timeout,
        "sender": environ["ROBOTSIM_SMTP_FROM"].strip(),
        "recipient": environ["ROBOTSIM_NOTIFY_TO"].strip(),
        "username": username,
        "password": password,
    }, ""


def _state_directory(environ: Mapping[str, str]) -> Path:
    configured = environ.get("ROBOTSIM_NOTIFY_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "robotsim" / "task-notifications"


def _claim_delivery(state_dir: Path, status: str, task_id: str) -> tuple[Path | None, str]:
    digest = hashlib.sha256(f"{status}\0{task_id}".encode("utf-8")).hexdigest()
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


def _send(subject: str, body: str, settings: Mapping[str, object]) -> None:
    message = EmailMessage()
    message["From"] = str(settings["sender"])
    message["To"] = str(settings["recipient"])
    message["Subject"] = subject
    message.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(
        str(settings["host"]), int(settings["port"]), timeout=float(settings["timeout"])
    ) as client:
        client.ehlo()
        client.starttls(context=context)
        client.ehlo()
        if settings["username"]:
            client.login(str(settings["username"]), str(settings["password"]))
        refused = client.send_message(message)
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)


def notify(
    status: str,
    task_id: str,
    summary: str,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    subject, body = format_notification(status, task_id, summary)
    if not task_id.strip() or not summary.strip():
        return Outcome("skipped", "task ID and summary must be non-empty", subject, body)
    if dry_run:
        return Outcome("dry_run", "no SMTP connection was made", subject, body)

    settings, problem = _load_settings(os.environ if environ is None else environ)
    if settings is None:
        return Outcome("skipped", problem, subject, body)

    env = os.environ if environ is None else environ
    try:
        pending, problem = _claim_delivery(_state_directory(env), status, task_id)
    except OSError as exc:
        return Outcome("failed", f"could not prepare duplicate state ({type(exc).__name__})", subject, body)
    if pending is None:
        return Outcome("duplicate", problem, subject, body)

    try:
        _send(subject, body, settings)
        sent = pending.with_suffix(".sent")
        os.replace(pending, sent)
    except Exception as exc:  # Notification is best-effort; never fail the engineering task.
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
        return Outcome("failed", f"SMTP delivery failed ({type(exc).__name__})", subject, body)
    return Outcome("sent", "notification delivered", subject, body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", choices=tuple(STATUS_LABELS))
    parser.add_argument("--task-id", required=True, help="stable task identifier used for duplicate protection")
    parser.add_argument("--summary", required=True, help="short, non-sensitive task summary")
    parser.add_argument("--dry-run", action="store_true", help="format the message without reading SMTP settings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outcome = notify(args.status, args.task_id, args.summary, dry_run=args.dry_run)
    if outcome.state == "dry_run":
        print(f"Dry run: {outcome.message}\nSubject: {outcome.subject}\n\n{outcome.body}", end="")
    elif outcome.state in {"sent", "duplicate"}:
        print(f"notify_task: {outcome.state}: {outcome.message}")
    else:
        print(f"notify_task: {outcome.state}: {outcome.message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
