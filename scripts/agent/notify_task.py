#!/usr/bin/env python3
"""Send one best-effort RobotSim Codex notification through AgentMail."""

from __future__ import annotations

import argparse
import hashlib
import html
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


STATUS_LABELS = {
    "ready_for_review": "Ready for review",
    "completed": "Completed",
    "blocked": "Blocked",
}
REQUIRED_SETTINGS = ("AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "ROBOTSIM_NOTIFY_TO")
AGENTMAIL_API_BASE = "https://api.agentmail.to"
GITHUB_API_BASE = "https://api.github.com"
ROBOTSIM_REPOSITORY = "lzy18001500226/RobotSim"
REQUEST_TIMEOUT_SECONDS = 8
STALE_CLAIM_SECONDS = 3600
RESEARCH_SCHEMA_VERSION = "robotsim.research-completion.v1"
RESEARCH_STATUSES = {"completed", "blocked", "deferred", "cancelled", "failed"}
RESEARCH_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "repository",
    "task_id",
    "worker",
    "final_status",
    "summary",
    "blockers",
    "recommended_next_action",
    "evidence",
    "timestamp",
}
RESEARCH_TASK_ID = re.compile(r"^issue-([1-9][0-9]{0,8})-[a-z0-9][a-z0-9._-]{0,100}$")
RESEARCH_EVENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
RESEARCH_ENVELOPE_MARKER = "<!-- robotsim-research-completion:v1"
RESEARCH_ENVELOPE_RE = re.compile(
    r"<!--\s*robotsim-research-completion:v1\s*\n(.*?)\n-->", re.S
)
RESEARCH_COMMENT_MARKER = re.compile(
    r"<!-- robotsim-research-event:v1:(sha256:[0-9a-f]{64}) -->"
)
RESEARCH_COMMENT_JSON = re.compile(
    r"<!-- robotsim-research-json:v1\s*\n(.*?)\n-->", re.S
)

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(AGENTMAIL_API_KEY|(?:api[_ -]?key|access[_ -]?token|password|secret))\s*[:=]\s*([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_ISSUE_REFERENCE = re.compile(r"(?i)(?:issue[/#-]\s*|#)(\d+)\b")
_RESEARCH_TOKEN_LITERAL = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?(?:"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|$)|"
    r"github_pat_[A-Za-z0-9_]{30,}|(?:gh[pousr])_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[0-9A-Z]{16}|"
    r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|ya29\.[A-Za-z0-9_-]{20,}|"
    r"AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    re.S,
)
_LOCAL_PATH = re.compile(
    r"(?i)(?:/home/|/root/|/tmp/|/var/|/workspace/|/workspaces/|/users/|/mnt/|/media/|"
    r"/opt/|/srv/|/private/|/volumes/|/repo/|/app/|/etc/|/run/|/proc/|/sys/|/dev/|"
    r"[a-z]:[\\/]|~[\\/]|\.\.?/)[^\s,;\"'<>)]*"
)
_RESEARCH_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[A-Z][A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)|"
    r"api[_ -]?key|access[_ -]?token|token|key|credential|authorization|password|secret|signature|sig|auth)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_RESEARCH_URL_SECRET_PARAMETER = re.compile(
    r"(?i)([?&](?:token|access_token|api_key|key|secret|password|signature|sig|auth)=[^&#\s]+)"
)
_URL_SECRET_PARAMETER = {
    "token",
    "access_token",
    "api_key",
    "key",
    "secret",
    "password",
    "signature",
    "sig",
    "auth",
    "authorization",
    "credential",
}


@dataclass(frozen=True)
class Outcome:
    state: str
    message: str
    subject: str = ""
    body: str = ""


@dataclass(frozen=True)
class PersistenceResult:
    state: str
    message: str


def _research_event_id(repository: str, task_id: str, worker: str) -> str:
    identity = "\0".join((RESEARCH_SCHEMA_VERSION, repository, task_id, worker))
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _research_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must contain valid UTF-8 text") from exc
    if len(value) > limit:
        raise ValueError(f"{field} exceeds its maximum length")
    safe = _one_line(value, limit).strip()
    safe = _RESEARCH_SECRET_ASSIGNMENT.sub("[redacted credential]", safe)
    safe = _RESEARCH_URL_SECRET_PARAMETER.sub("[redacted URL credential]", safe)
    safe = _RESEARCH_TOKEN_LITERAL.sub("[redacted credential]", safe)
    local_path = _LOCAL_PATH.search(safe)
    if local_path is not None:
        safe = safe[: local_path.start()] + "[local path omitted]"
    return safe


def _durable_evidence(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError("evidence must be a list of at most 20 durable HTTPS references")
    evidence: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > 512:
            raise ValueError("each evidence reference must be a short HTTPS URL")
        try:
            item.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("evidence URL must contain valid UTF-8 text") from exc
        if item != item.strip() or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in item):
            raise ValueError("evidence URL must not contain whitespace or control characters")
        try:
            parts = urllib.parse.urlsplit(item)
            host = parts.hostname
            port = parts.port
        except ValueError as exc:
            raise ValueError("evidence contains an invalid URL") from exc
        if (
            parts.scheme != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or not parts.path.startswith("/")
            or host.rstrip(".") == "localhost"
            or host.rstrip(".").endswith(".local")
        ):
            raise ValueError("evidence must use a durable HTTPS URL, not a local path")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("evidence must not point to a local or private address")
        dns_host = host.rstrip(".")
        if address is None and ("." not in dns_host or dns_host.startswith(".") or ".." in dns_host):
            raise ValueError("evidence must use a durable public HTTPS host")
        if port is not None and port not in {443}:
            raise ValueError("evidence URLs must use the standard HTTPS port")
        query_keys = (
            key.lower().replace("-", "_")
            for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        )
        if any(
            key in _URL_SECRET_PARAMETER
            or re.search(r"(?:^|_)(?:token|secret|password|signature|sig|auth|credential|api_key|access_key)(?:$|_)", key)
            for key in query_keys
        ):
            raise ValueError("evidence URL contains a credential-like query parameter")
        if _RESEARCH_TOKEN_LITERAL.search(item) or _BEARER_VALUE.search(item):
            raise ValueError("evidence URL contains credential-shaped text")
        evidence.append(item)
    return evidence


def normalize_research_event(payload: object) -> dict[str, object]:
    """Validate an envelope and return its canonical, versioned event."""
    if not isinstance(payload, dict):
        raise ValueError("research completion must be a JSON object")
    if set(payload) != RESEARCH_EVENT_FIELDS:
        raise ValueError("research completion has missing or unsupported fields")
    if payload.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported research completion schema_version")
    repository = payload.get("repository")
    if repository != ROBOTSIM_REPOSITORY:
        raise ValueError("repository must be the canonical RobotSim repository")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or RESEARCH_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("task_id must identify its issue as issue-<number>-<stable-slug>")
    if _RESEARCH_TOKEN_LITERAL.search(task_id):
        raise ValueError("task_id must not contain credential-shaped text")
    issue_match = RESEARCH_TASK_ID.fullmatch(task_id)
    assert issue_match is not None
    worker = _research_text(payload.get("worker"), "worker", 80)
    if not worker:
        raise ValueError("worker must not be empty")
    final_status = payload.get("final_status")
    if not isinstance(final_status, str) or final_status not in RESEARCH_STATUSES:
        raise ValueError("final_status is not an allowed research completion status")
    summary = _research_text(payload.get("summary"), "summary", 2000)
    next_action = _research_text(payload.get("recommended_next_action"), "recommended_next_action", 1000)
    if not summary or not next_action:
        raise ValueError("summary and recommended_next_action must not be empty")
    blockers_value = payload.get("blockers")
    if not isinstance(blockers_value, list) or len(blockers_value) > 10:
        raise ValueError("blockers must be a list of at most 10 items")
    blockers = [_research_text(item, "blocker", 500) for item in blockers_value]
    if any(not item for item in blockers):
        raise ValueError("blockers must not contain empty items")
    evidence = _durable_evidence(payload.get("evidence"))
    timestamp_value = payload.get("timestamp")
    if not isinstance(timestamp_value, str) or len(timestamp_value) > 64:
        raise ValueError("timestamp must be an ISO 8601 timestamp with a timezone")
    try:
        timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("missing timezone")
        timestamp = timestamp.astimezone(timezone.utc).replace(microsecond=0)
    except (ValueError, OverflowError) as exc:
        raise ValueError("timestamp must be an ISO 8601 timestamp with a timezone") from exc
    normalized_timestamp = timestamp.isoformat().replace("+00:00", "Z")
    event_id = _research_event_id(repository, task_id, worker)
    provided_event_id = payload.get("event_id")
    if not isinstance(provided_event_id, str) or provided_event_id not in {"auto", event_id}:
        raise ValueError("event_id must be 'auto' or match the deterministic event identity")
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "event_id": event_id,
        "repository": repository,
        "task_id": task_id,
        "worker": worker,
        "final_status": final_status,
        "summary": summary,
        "blockers": blockers,
        "recommended_next_action": next_action,
        "evidence": evidence,
        "timestamp": normalized_timestamp,
    }


def _event_json(event: Mapping[str, object], *, pretty: bool = False) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _research_comment_body(event: Mapping[str, object]) -> str:
    # Escape HTML comment terminators while retaining valid, semantically
    # identical JSON for the duplicate detector.
    canonical = _event_json(event).replace(">", r"\u003e")
    display = html.escape(_event_json(event, pretty=True), quote=False)
    event_id = str(event["event_id"])
    return (
        f"<!-- robotsim-research-event:v1:{event_id} -->\n"
        f"<!-- robotsim-research-json:v1\n{canonical}\n-->\n"
        "<h3>RobotSim research completion</h3>\n"
        f"<pre>\n{display}\n</pre>\n"
    )


def _github_token(environ: Mapping[str, str]) -> str:
    return (environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN") or "").strip()


def _github_json(method: str, path: str, token: str, payload: object | None = None) -> object:
    try:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{GITHUB_API_BASE}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
            method=method,
        )
        with _open_request(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None
    except Exception as exc:
        raise RuntimeError(type(exc).__name__) from None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("invalid GitHub API response") from None


def _list_issue_comments(issue_number: str, token: str) -> list[dict[str, object]]:
    comments: list[dict[str, object]] = []
    page = 1
    while page <= 1000:
        path = (
            f"/repos/{ROBOTSIM_REPOSITORY}/issues/{issue_number}/comments"
            f"?per_page=100&page={page}"
        )
        result = _github_json("GET", path, token)
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise RuntimeError("invalid GitHub comments response")
        comments.extend(result)
        if len(result) < 100:
            return comments
        page += 1
    raise RuntimeError("GitHub comments pagination limit exceeded")


def _existing_event_state(
    comments: list[dict[str, object]], event: Mapping[str, object]
) -> str:
    expected_id = str(event["event_id"])
    found = False
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        for marker in RESEARCH_COMMENT_MARKER.finditer(body):
            if marker.group(1) != expected_id:
                continue
            payload = RESEARCH_COMMENT_JSON.search(body)
            if payload is None:
                return "conflict"
            try:
                stored = json.loads(payload.group(1))
            except json.JSONDecodeError:
                return "conflict"
            if stored != event:
                return "conflict"
            found = True
    return "same" if found else "absent"


def persist_research_event(
    event: Mapping[str, object], *, environ: Mapping[str, str] | None = None
) -> PersistenceResult:
    env = os.environ if environ is None else environ
    token = _github_token(env)
    if not token:
        return PersistenceResult(
            "failed",
            "GitHub Issue persistence unavailable; provide GH_TOKEN or GITHUB_TOKEN with comment-write access",
        )
    task_id = str(event["task_id"])
    issue_match = RESEARCH_TASK_ID.fullmatch(task_id)
    if issue_match is None:
        return PersistenceResult("failed", "research task_id does not identify an originating Issue")
    issue_number = issue_match.group(1)
    try:
        comments = _list_issue_comments(issue_number, token)
        existing = _existing_event_state(comments, event)
        if existing == "same":
            return PersistenceResult("duplicate", f"event already exists on Issue #{issue_number}")
        if existing == "conflict":
            return PersistenceResult("failed", "event_id already exists with a different or invalid payload")
        result = _github_json(
            "POST",
            f"/repos/{ROBOTSIM_REPOSITORY}/issues/{issue_number}/comments",
            token,
            {"body": _research_comment_body(event)},
        )
        if not isinstance(result, dict) or not isinstance(result.get("id"), int):
            return PersistenceResult("failed", "GitHub did not confirm the Issue comment")
        return PersistenceResult("persisted", f"event persisted to Issue #{issue_number}")
    except RuntimeError as exc:
        return PersistenceResult("failed", f"GitHub Issue persistence failed ({exc})")


def _research_mail(event: Mapping[str, object], *, environ: Mapping[str, str] | None) -> Outcome:
    event_id = str(event["event_id"])
    issue_match = RESEARCH_TASK_ID.fullmatch(str(event["task_id"]))
    if issue_match is None:
        return Outcome("failed", "research task_id does not identify an originating Issue")
    issue_number = issue_match.group(1)
    subject = f"[RobotSim] Research {event['final_status']}: Issue #{issue_number}"
    body = _event_json(event, pretty=True) + "\n"
    digest = _delivery_digest("research-completion", event_id)
    return _deliver(subject, body, digest, dry_run=False, environ=environ)


def extract_research_envelope(message: object) -> tuple[bool, object | None, str]:
    """Return whether a Stop message contains a research envelope and its parsed body."""
    if not isinstance(message, str) or RESEARCH_ENVELOPE_MARKER not in message:
        return False, None, ""
    matches = list(RESEARCH_ENVELOPE_RE.finditer(message))
    if len(matches) != 1 or message.count(RESEARCH_ENVELOPE_MARKER) != 1:
        return True, None, "research completion must contain exactly one marked JSON envelope"
    try:
        return True, json.loads(matches[0].group(1)), ""
    except json.JSONDecodeError:
        return True, None, "research completion envelope is not valid JSON"


def process_research_completion(
    payload: object,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    try:
        event = normalize_research_event(payload)
    except (ValueError, TypeError) as exc:
        return Outcome("failed", f"invalid research completion: {exc}")
    subject = f"[RobotSim] Research {event['final_status']}: {event['task_id']}"
    body = _event_json(event, pretty=True) + "\n"
    if dry_run:
        return Outcome("dry_run", "no GitHub comment or AgentMail request was made", subject, body)
    persisted = persist_research_event(event, environ=environ)
    if persisted.state == "failed":
        return Outcome("failed", persisted.message, subject, body)
    mailed = _research_mail(event, environ=environ)
    prefix = f"GitHub {persisted.message};"
    if mailed.state == "failed":
        return Outcome("failed", f"{prefix} AgentMail delivery failed ({mailed.message})", subject, body)
    if mailed.state == "skipped":
        return Outcome("skipped", f"{prefix} AgentMail skipped ({mailed.message})", subject, body)
    if mailed.state == "duplicate":
        return Outcome("duplicate", f"{prefix} AgentMail already sent", subject, body)
    return Outcome("sent", f"{prefix} AgentMail notified", subject, body)


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


def _deliver(
    subject: str,
    body: str,
    digest: str,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    if dry_run:
        return Outcome("dry_run", "no AgentMail request was made", subject, body)

    env = os.environ if environ is None else environ
    settings, problem = _load_settings(env)
    if settings is None:
        return Outcome("skipped", problem, subject, body)

    try:
        pending, problem = _claim_delivery(_state_directory(env), digest)
    except OSError as exc:
        return Outcome("failed", f"could not prepare duplicate state ({type(exc).__name__})", subject, body)
    if pending is None:
        return Outcome("duplicate", problem, subject, body)

    try:
        # AgentMail's idempotency key also protects retries after an ambiguous response.
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
    return _deliver(
        subject,
        body,
        _delivery_digest(status, task_id),
        dry_run=dry_run,
        environ=environ,
    )


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

    has_research_event, research_payload, envelope_problem = extract_research_envelope(
        payload.get("last_assistant_message")
    )
    if has_research_event:
        if envelope_problem:
            return Outcome("failed", f"invalid research completion: {envelope_problem}")
        return process_research_completion(research_payload, dry_run=dry_run, environ=environ)

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
        choices=(*STATUS_LABELS, "research-completion", "stop"),
        help="terminal outcome, research-completion JSON, or 'stop' for a local Codex Stop payload",
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
    elif args.event == "research-completion":
        try:
            payload = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError):
            outcome = Outcome("failed", "invalid research completion: input is not valid JSON")
        else:
            outcome = process_research_completion(payload, dry_run=args.dry_run)
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
