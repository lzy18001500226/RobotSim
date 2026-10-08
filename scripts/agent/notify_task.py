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
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

try:
    from .closeout_gate import check_closeout
except ImportError:  # pragma: no cover - direct script execution
    from closeout_gate import check_closeout

try:
    import fcntl
except ImportError:  # pragma: no cover - native Windows uses the process-local lock below.
    fcntl = None  # type: ignore[assignment]


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
TASK_CLOSEOUT_SCHEMA_VERSION = "robotsim.task-closeout.v1"
TASK_CLOSEOUT_KINDS = {"implementation", "experiment", "research", "review", "audit"}
TASK_CLOSEOUT_STATUSES = {"completed", "blocked", "deferred", "failed", "cancelled"}
TASK_CLOSEOUT_FIELDS = {
    "schema_version",
    "event_id",
    "repository",
    "task_id",
    "attempt_id",
    "worker",
    "task_kind",
    "status",
    "summary",
    "branch",
    "head_sha",
    "pr_number",
    "validation",
    "evidence",
    "blockers",
    "recommended_next_action",
    "completed_at",
}
TASK_CLOSEOUT_TASK_ID = re.compile(r"^issue-([1-9][0-9]{0,8})-[a-z0-9][a-z0-9._-]{0,100}$")
TASK_CLOSEOUT_ATTEMPT_ID = re.compile(
    r"^(?:[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}|legacy-[0-9a-f]{32})$"
)
TASK_CLOSEOUT_EVENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
TASK_CLOSEOUT_SHA = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
TASK_CLOSEOUT_ENVELOPE_MARKER = "<!-- robotsim.task-closeout.v1"
TASK_CLOSEOUT_ENVELOPE_RE = re.compile(
    r"<!--\s*robotsim\.task-closeout\.v1\s*\n(.*?)\n-->", re.S
)
TASK_CLOSEOUT_COMMENT_MARKER = re.compile(
    r"<!-- robotsim-task-closeout:v1:(sha256:[0-9a-f]{64}):([0-9a-f]{64}) -->"
)
TASK_CLOSEOUT_COMMENT_JSON = re.compile(
    r"<!-- robotsim-task-closeout-json:v1\s*\n(.*?)\n-->", re.S
)

# Backward-compatible input aliases. New persistence always uses task-closeout.v1.
RESEARCH_SCHEMA_VERSION = "robotsim.research-completion.v1"
RESEARCH_STATUSES = TASK_CLOSEOUT_STATUSES
RESEARCH_EVENT_FIELDS = {
    "schema_version", "event_id", "repository", "task_id", "worker",
    "final_status", "summary", "blockers", "recommended_next_action",
    "evidence", "timestamp",
}
RESEARCH_TASK_ID = TASK_CLOSEOUT_TASK_ID
RESEARCH_EVENT_ID = TASK_CLOSEOUT_EVENT_ID
RESEARCH_ENVELOPE_MARKER = "<!-- robotsim-research-completion:v1"
RESEARCH_ENVELOPE_RE = re.compile(
    r"<!--\s*robotsim-research-completion:v1\s*\n(.*?)\n-->", re.S
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
    r"[a-z]:[\\/]|\\\\[^\\/\s]+[\\/]|~[\\/]|\.\.?/)[^\s,;\"'<>)]*"
)
_GENERIC_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9/])/(?!/)[^\s,;\"'<>)]*")
_RESEARCH_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[A-Z][A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)|"
    r"(?:[A-Z0-9]+[_-])*(?:token|secret|password|signature|sig|auth|credential|"
    r"api[_ -]?key|access[_ -]?key|access[_ -]?token|key|authorization))"
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
_URL_USERINFO = re.compile(r"(?i)\bhttps?://[^/@\s:]+:[^/@\s]+@")


@dataclass(frozen=True)
class Outcome:
    state: str
    message: str
    subject: str = ""
    body: str = ""
    exit_code: int = 0


@dataclass(frozen=True)
class PersistenceResult:
    state: str
    message: str


_EVENT_LOCKS: dict[str, threading.Lock] = {}
_EVENT_LOCKS_GUARD = threading.Lock()


def _research_event_id(repository: str, task_id: str, worker: str) -> str:
    """Return the v1 ID only for validating legacy research envelopes."""
    identity = "\0".join((RESEARCH_SCHEMA_VERSION, repository, task_id, worker))
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _closeout_event_id(repository: str, task_id: str, attempt_id: str, worker: str) -> str:
    identity = "\0".join(
        (TASK_CLOSEOUT_SCHEMA_VERSION, repository, task_id, attempt_id, worker)
    )
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _closeout_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must contain valid UTF-8 text") from exc
    if len(value) > limit:
        raise ValueError(f"{field} exceeds its maximum length")
    safe = _one_line(value, limit).strip()
    safe = _URL_USERINFO.sub("https://[redacted]@", safe)
    safe = _RESEARCH_SECRET_ASSIGNMENT.sub("[redacted credential]", safe)
    safe = _RESEARCH_URL_SECRET_PARAMETER.sub("[redacted URL credential]", safe)
    safe = _RESEARCH_TOKEN_LITERAL.sub("[redacted credential]", safe)
    local_path = min(
        (match for pattern in (_LOCAL_PATH, _GENERIC_ABSOLUTE_PATH) if (match := pattern.search(safe))),
        key=lambda match: match.start(),
        default=None,
    )
    if local_path is not None:
        safe = safe[: local_path.start()] + "[local path omitted]"
    return safe


def _identity_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or value != value.strip():
        raise ValueError(f"{field} must be non-empty, bounded text without surrounding whitespace")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must contain valid UTF-8 text") from exc
    if any(
        (ord(character) <= 0x20 and character != " ") or ord(character) == 0x7F
        for character in value
    ) or "  " in value:
        raise ValueError(f"{field} must not contain control characters or non-canonical spacing")
    if (
        _RESEARCH_TOKEN_LITERAL.search(value)
        or _RESEARCH_SECRET_ASSIGNMENT.search(value)
        or _BEARER_VALUE.search(value)
        or _URL_USERINFO.search(value)
        or _RESEARCH_URL_SECRET_PARAMETER.search(value)
        or _LOCAL_PATH.search(value)
        or _GENERIC_ABSOLUTE_PATH.search(value)
    ):
        raise ValueError(f"{field} must not contain credentials or local paths")
    return value


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
        fragment_keys = (
            key.lower().replace("-", "_")
            for key, _ in urllib.parse.parse_qsl(parts.fragment, keep_blank_values=True)
        )
        if any(
            key in _URL_SECRET_PARAMETER
            or re.search(r"(?:^|_)(?:token|secret|password|signature|sig|auth|credential|api_key|access_key)(?:$|_)", key)
            for key in fragment_keys
        ):
            raise ValueError("evidence URL contains a credential-like fragment")
        decoded_url = urllib.parse.unquote(item)
        if (
            _RESEARCH_TOKEN_LITERAL.search(decoded_url)
            or _BEARER_VALUE.search(decoded_url)
            or _RESEARCH_SECRET_ASSIGNMENT.search(parts.fragment)
        ):
            raise ValueError("evidence URL contains credential-shaped text")
        evidence.append(item)
    return evidence


def _normalize_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError(f"{field} must be an ISO 8601 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("missing timezone")
        parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp with a timezone") from exc
    return parsed.isoformat().replace("+00:00", "Z")


def normalize_task_closeout(payload: object) -> dict[str, object]:
    """Validate and canonicalize a task-closeout.v1 envelope."""
    if not isinstance(payload, dict):
        raise ValueError("task closeout must be a JSON object")
    if set(payload) != TASK_CLOSEOUT_FIELDS:
        raise ValueError("task closeout has missing or unsupported fields")
    if payload.get("schema_version") != TASK_CLOSEOUT_SCHEMA_VERSION:
        raise ValueError("unsupported task-closeout schema_version")
    repository = payload.get("repository")
    if repository != ROBOTSIM_REPOSITORY:
        raise ValueError("repository must be the canonical RobotSim repository")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or TASK_CLOSEOUT_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("task_id must identify its issue as issue-<number>-<stable-slug>")
    if _RESEARCH_TOKEN_LITERAL.search(task_id):
        raise ValueError("task_id must not contain credential-shaped text")
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or TASK_CLOSEOUT_ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id must be a stable per-run identifier")
    if _RESEARCH_TOKEN_LITERAL.search(attempt_id):
        raise ValueError("attempt_id must not contain credential-shaped text")
    worker = _identity_text(payload.get("worker"), "worker", 80)
    task_kind = payload.get("task_kind")
    if not isinstance(task_kind, str) or task_kind not in TASK_CLOSEOUT_KINDS:
        raise ValueError("task_kind is not an allowed task kind")
    status = payload.get("status")
    if not isinstance(status, str) or status not in TASK_CLOSEOUT_STATUSES:
        raise ValueError("status is not an allowed task-closeout status")
    summary = _closeout_text(payload.get("summary"), "summary", 2000)
    next_action = _closeout_text(payload.get("recommended_next_action"), "recommended_next_action", 1000)
    if not summary or not next_action:
        raise ValueError("summary and recommended_next_action must not be empty")
    branch_value = payload.get("branch")
    branch = None if branch_value is None else _closeout_text(branch_value, "branch", 200)
    if branch == "":
        branch = None
    head_sha_value = payload.get("head_sha")
    if head_sha_value is not None and (
        not isinstance(head_sha_value, str) or TASK_CLOSEOUT_SHA.fullmatch(head_sha_value) is None
    ):
        raise ValueError("head_sha must be a 40- or 64-character hexadecimal commit ID")
    head_sha = head_sha_value.lower() if isinstance(head_sha_value, str) else None
    pr_number_value = payload.get("pr_number")
    if pr_number_value is not None and (
        isinstance(pr_number_value, bool)
        or not isinstance(pr_number_value, int)
        or not 1 <= pr_number_value <= 999_999_999
    ):
        raise ValueError("pr_number must be a positive GitHub pull request number or null")
    if pr_number_value is not None and (branch is None or head_sha is None):
        raise ValueError("PR closeouts must include the branch and head_sha")
    validation_value = payload.get("validation")
    if not isinstance(validation_value, list) or len(validation_value) > 20:
        raise ValueError("validation must be a list of at most 20 concise results")
    validation = [_closeout_text(item, "validation result", 500) for item in validation_value]
    if any(not item for item in validation):
        raise ValueError("validation results must not be empty")
    blockers_value = payload.get("blockers")
    if not isinstance(blockers_value, list) or len(blockers_value) > 10:
        raise ValueError("blockers must be a list of at most 10 items")
    blockers = [_closeout_text(item, "blocker", 500) for item in blockers_value]
    if any(not item for item in blockers):
        raise ValueError("blockers must not contain empty items")
    evidence = _durable_evidence(payload.get("evidence"))
    completed_at = _normalize_timestamp(payload.get("completed_at"), "completed_at")
    event_id = _closeout_event_id(repository, task_id, attempt_id, worker)
    provided_event_id = payload.get("event_id")
    if not isinstance(provided_event_id, str) or provided_event_id not in {"auto", event_id}:
        raise ValueError("event_id must be 'auto' or match this attempt's deterministic identity")
    return {
        "schema_version": TASK_CLOSEOUT_SCHEMA_VERSION,
        "event_id": event_id,
        "repository": repository,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "worker": worker,
        "task_kind": task_kind,
        "status": status,
        "summary": summary,
        "branch": branch,
        "head_sha": head_sha,
        "pr_number": pr_number_value,
        "validation": validation,
        "evidence": evidence,
        "blockers": blockers,
        "recommended_next_action": next_action,
        "completed_at": completed_at,
    }


def normalize_research_event(payload: object) -> dict[str, object]:
    """Convert a v1 research envelope into the unified v1 closeout schema."""
    if not isinstance(payload, dict) or set(payload) != RESEARCH_EVENT_FIELDS:
        raise ValueError("research completion has missing or unsupported fields")
    if payload.get("schema_version") != RESEARCH_SCHEMA_VERSION:
        raise ValueError("unsupported research-completion schema_version")
    repository = payload.get("repository")
    task_id = payload.get("task_id")
    if repository != ROBOTSIM_REPOSITORY or not isinstance(task_id, str):
        raise ValueError("research completion must identify the canonical repository and Issue")
    if TASK_CLOSEOUT_TASK_ID.fullmatch(task_id) is None:
        raise ValueError("task_id must identify its Issue")
    worker = _closeout_text(payload.get("worker"), "worker", 80)
    expected_legacy_id = _research_event_id(repository, task_id, worker)
    if payload.get("event_id") not in {"auto", expected_legacy_id}:
        raise ValueError("legacy event_id does not match the research event")
    completed_at = _normalize_timestamp(payload.get("timestamp"), "timestamp")
    legacy_attempt = "legacy-" + hashlib.sha256(completed_at.encode("ascii")).hexdigest()[:32]
    unified = {
        "schema_version": TASK_CLOSEOUT_SCHEMA_VERSION,
        "event_id": "auto",
        "repository": repository,
        "task_id": task_id,
        "attempt_id": legacy_attempt,
        "worker": worker,
        "task_kind": "research",
        "status": payload.get("final_status"),
        "summary": payload.get("summary"),
        "branch": None,
        "head_sha": None,
        "pr_number": None,
        "validation": [],
        "evidence": payload.get("evidence"),
        "blockers": payload.get("blockers"),
        "recommended_next_action": payload.get("recommended_next_action"),
        "completed_at": completed_at,
    }
    return normalize_task_closeout(unified)


def _event_json(event: Mapping[str, object], *, pretty: bool = False) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _closeout_payload_digest(event: Mapping[str, object]) -> str:
    canonical = _event_json(event).replace(">", r"\u003e")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _task_closeout_comment_body(event: Mapping[str, object]) -> str:
    # The digest binds this discoverable marker to the canonical event payload.
    canonical = _event_json(event).replace(">", r"\u003e")
    display = html.escape(_event_json(event, pretty=True), quote=False)
    event_id = str(event["event_id"])
    payload_digest = _closeout_payload_digest(event)
    destination = "PR" if event["pr_number"] is not None else "Issue"
    number = event["pr_number"]
    if number is None:
        issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(str(event["task_id"]))
        assert issue_match is not None
        number = int(issue_match.group(1))
    return (
        f"<!-- robotsim-task-closeout:v1:{event_id}:{payload_digest} -->\n"
        f"<!-- robotsim-task-closeout-json:v1\n{canonical}\n-->\n"
        f"<h3>RobotSim task closeout · {destination} #{number}</h3>\n"
        f"<pre>\n{display}\n</pre>\n"
    )


def _github_token(environ: Mapping[str, str]) -> str:
    token = (environ.get("GH_TOKEN") or environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        return token
    gh = shutil.which("gh")
    if not gh:
        return ""
    try:
        result = subprocess.run(
            [gh, "auth", "token"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    token = result.stdout.strip()
    return token if result.returncode == 0 and token and "\n" not in token else ""


def _github_actor(token: str) -> str:
    result = _github_json("GET", "/user", token)
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("login"), str)
        or not result["login"]
        or len(result["login"]) > 80
        or result["login"] != result["login"].strip()
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in result["login"])
    ):
        raise RuntimeError("could not identify authenticated GitHub account")
    return result["login"]


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
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("invalid GitHub API response") from None


def _list_repository_issue_comments(token: str) -> list[dict[str, object]]:
    """List comments across every Issue and PR in the canonical repository.

    Event IDs are repository-global, so target-local scans cannot arbitrate
    two writers that chose different valid destinations.
    """
    comments: list[dict[str, object]] = []
    page = 1
    while page <= 1000:
        path = (
            f"/repos/{ROBOTSIM_REPOSITORY}/issues/comments"
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


def _trusted_closeout_records(
    comments: list[dict[str, object]], event_id: str, actor: str
) -> list[tuple[dict[str, object], dict[str, object]]]:
    records: list[tuple[dict[str, object], dict[str, object]]] = []
    for comment in comments:
        user = comment.get("user")
        if (
            not isinstance(user, dict)
            or not isinstance(user.get("login"), str)
            or user["login"].casefold() != actor.casefold()
        ):
            continue
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        markers = list(TASK_CLOSEOUT_COMMENT_MARKER.finditer(body))
        if not any(marker.group(1) == event_id for marker in markers):
            continue
        payloads = list(TASK_CLOSEOUT_COMMENT_JSON.finditer(body))
        if len(markers) != 1 or len(payloads) != 1:
            continue
        marker = markers[0]
        try:
            stored = json.loads(payloads[0].group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(stored, dict) or stored.get("event_id") != event_id:
            continue
        try:
            normalized = normalize_task_closeout(stored)
        except (TypeError, ValueError):
            continue
        if normalized != stored or marker.group(2) != _closeout_payload_digest(stored):
            continue
        records.append((comment, stored))
    return records


def _trusted_closeout_comments(
    comments: list[dict[str, object]], event: Mapping[str, object], actor: str
) -> tuple[str, list[dict[str, object]]]:
    records = _trusted_closeout_records(comments, str(event["event_id"]), actor)
    matching = [comment for comment, stored in records if stored == event]
    if any(stored != event for _, stored in records):
        return "conflict", matching
    return ("same" if matching else "absent"), matching


def _existing_event_state(
    comments: list[dict[str, object]], event: Mapping[str, object], actor: str = ""
) -> str:
    """Compatibility test helper, backed by authenticated account markers."""
    return _trusted_closeout_comments(comments, event, actor)[0]


@contextmanager
def _task_closeout_lock(event_id: str, environ: Mapping[str, str]):
    state_dir = _state_directory(environ)
    key = hashlib.sha256(f"{state_dir.resolve()}\0{event_id}".encode("utf-8")).hexdigest()
    with _EVENT_LOCKS_GUARD:
        local_lock = _EVENT_LOCKS.setdefault(key, threading.Lock())
    local_lock.acquire()
    lock_file = None
    try:
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_file = (state_dir / f"github-closeout-{key}.lock").open("a+b")
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lock_file is not None:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        local_lock.release()


def _reconcile_closeout_comments(
    comments: list[dict[str, object]],
    event_id: str,
    actor: str,
    token: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, bool]:
    """Converge repository-wide records to the earliest authenticated comment.

    GitHub has no conditional-create operation for Issue comments. Concurrent
    writers therefore reconcile after writing: the lowest comment ID wins,
    and every other valid record for that event is removed. Re-reading after
    deletion makes retries and overlapping reconcilers converge on the same
    canonical record.
    """
    snapshot = comments
    for _ in range(5):
        records = _trusted_closeout_records(snapshot, event_id, actor)
        if not records:
            return None, None, True
        if any(
            not isinstance(comment.get("id"), int) or isinstance(comment.get("id"), bool)
            for comment, _ in records
        ):
            return None, None, False
        ordered = sorted(records, key=lambda pair: int(pair[0]["id"]))
        canonical_comment, canonical_event = ordered[0]
        if len(ordered) == 1:
            return canonical_comment, canonical_event, True
        for duplicate_comment, _ in ordered[1:]:
            comment_id = int(duplicate_comment["id"])
            try:
                _github_json(
                    "DELETE",
                    f"/repos/{ROBOTSIM_REPOSITORY}/issues/comments/{comment_id}",
                    token,
                )
            except RuntimeError as exc:
                if str(exc) != "HTTP 404":
                    return None, None, False
        snapshot = _list_repository_issue_comments(token)
        remaining = _trusted_closeout_records(snapshot, event_id, actor)
        if len(remaining) == 1 and remaining[0][0].get("id") == canonical_comment.get("id"):
            return remaining[0][0], remaining[0][1], True
    return None, None, False


def _without_inline_code_spans(text: str) -> str:
    """Remove CommonMark backtick spans, including spans crossing line breaks."""
    runs: list[tuple[int, int, int, bool]] = []
    index = 0
    while index < len(text):
        if text[index] != "`":
            index += 1
            continue
        run_end = index + 1
        while run_end < len(text) and text[run_end] == "`":
            run_end += 1
        # A backslash-escaped delimiter is ordinary text.
        preceding_slashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            preceding_slashes += 1
            cursor -= 1
        runs.append((index, run_end, run_end - index, preceding_slashes % 2 == 1))
        index = run_end

    # Match each run to the next equal-length run in one reverse pass. This
    # avoids repeatedly rescanning long adversarial PR descriptions.
    next_equal_run: dict[int, int] = {}
    next_for_run: dict[int, int] = {}
    for run_index in range(len(runs) - 1, -1, -1):
        next_for_run[run_index] = next_equal_run.get(runs[run_index][2], -1)
        next_equal_run[runs[run_index][2]] = run_index

    output: list[str] = []
    text_cursor = 0
    run_index = 0
    while run_index < len(runs):
        start, end, _, escaped = runs[run_index]
        closing_index = next_for_run[run_index]
        if escaped or closing_index < 0:
            run_index += 1
            continue
        closing_end = runs[closing_index][1]
        output.extend((text[text_cursor:start], " "))
        text_cursor = closing_end
        run_index = closing_index + 1
    output.append(text[text_cursor:])
    return "".join(output)


def _closeout_target(event: Mapping[str, object]) -> tuple[int, str]:
    pr_number = event.get("pr_number")
    if isinstance(pr_number, int) and not isinstance(pr_number, bool):
        return pr_number, "PR"
    task_id = str(event["task_id"])
    issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(task_id)
    if issue_match is None:
        raise ValueError("task_id does not identify an originating Issue")
    return int(issue_match.group(1)), "Issue"


def _pr_closes_task_issue(pull: Mapping[str, object], task_id: str) -> bool:
    issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(task_id)
    body = pull.get("body")
    if issue_match is None or not isinstance(body, str):
        return False
    issue_number = issue_match.group(1)
    # GitHub closing keywords in examples are inert. Remove fenced and
    # indented code blocks, inline code spans, and HTML code/pre examples
    # before considering issue references.
    without_html_code = re.sub(r"(?is)<(pre|code)\b[^>]*>.*?</\1\s*>", " ", body)
    visible_segments: list[str] = []
    paragraph: list[str] = []

    def finish_paragraph() -> None:
        if paragraph:
            visible_segments.append(_without_inline_code_spans("\n".join(paragraph)))
            paragraph.clear()

    fence_character = ""
    fence_length = 0
    for line in without_html_code.splitlines():
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_character:
            if fence:
                run = fence.group(1)
                if run[0] == fence_character and len(run) >= fence_length and not line[fence.end():].strip():
                    fence_character = ""
                    fence_length = 0
            continue
        if fence:
            finish_paragraph()
            run = fence.group(1)
            fence_character = run[0]
            fence_length = len(run)
            continue
        if not line.strip() or line.startswith("\t") or line.startswith("    "):
            finish_paragraph()
            continue
        paragraph.append(line)
    finish_paragraph()
    visible_body = "\x00".join(visible_segments)
    closing_reference = re.compile(
        r"(?i)\b(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+"
        r"(?:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)#(?P<qualified_number>[1-9][0-9]*)|"
        r"#(?P<local_number>[1-9][0-9]*))\b"
    )
    for reference in closing_reference.finditer(visible_body):
        local_number = reference.group("local_number")
        if local_number == issue_number:
            return True
        if (
            reference.group("qualified_number") == issue_number
            and reference.group("owner") is not None
            and reference.group("owner").casefold() == "lzy18001500226"
            and reference.group("repo").casefold() == "robotsim"
        ):
            return True
    return False


def persist_task_closeout(
    event: Mapping[str, object], *, environ: Mapping[str, str] | None = None
) -> PersistenceResult:
    env = os.environ if environ is None else environ
    try:
        event = normalize_task_closeout(dict(event))
    except (ValueError, TypeError) as exc:
        return PersistenceResult("failed", f"invalid task closeout: {exc}")
    token = _github_token(env)
    if not token:
        return PersistenceResult(
            "failed",
            "GitHub closeout persistence unavailable; configure GH_TOKEN/GITHUB_TOKEN or authenticate the GitHub CLI with Issue and PR comment read/write access",
        )
    try:
        target_number, target_kind = _closeout_target(event)
    except ValueError as exc:
        return PersistenceResult("failed", str(exc))
    comments_path = f"/repos/{ROBOTSIM_REPOSITORY}/issues/{target_number}/comments"
    try:
        actor = _github_actor(token)
        if target_kind == "PR":
            pull = _github_json(
                "GET", f"/repos/{ROBOTSIM_REPOSITORY}/pulls/{target_number}", token
            )
            if (
                not isinstance(pull, dict)
                or pull.get("number") != target_number
                or not isinstance(pull.get("base"), dict)
                or not isinstance(pull["base"].get("repo"), dict)
                or pull["base"]["repo"].get("full_name") != ROBOTSIM_REPOSITORY
            ):
                return PersistenceResult("failed", "pull request is not in the canonical RobotSim repository")
            head = pull.get("head")
            if (
                not isinstance(head, dict)
                or not isinstance(head.get("sha"), str)
                or head["sha"].lower() != event["head_sha"]
                or head.get("ref") != event["branch"]
            ):
                return PersistenceResult("failed", "closeout branch/head_sha do not match the current pull request head")
            if not _pr_closes_task_issue(pull, str(event["task_id"])):
                return PersistenceResult(
                    "failed", "pull request description must link the originating Issue with a closing keyword"
                )
        with _task_closeout_lock(str(event["event_id"]), env):
            comments = _list_repository_issue_comments(token)
            canonical_comment, canonical_event, reconciled = _reconcile_closeout_comments(
                comments, str(event["event_id"]), actor, token
            )
            if not reconciled:
                return PersistenceResult("failed", "closeout comments could not be consolidated")
            if canonical_event is not None:
                if canonical_event != event:
                    return PersistenceResult("failed", "attempt ID already exists with a different closeout payload")
                return PersistenceResult("duplicate", f"event already exists on {target_kind} #{target_number}")
            result = _github_json(
                "POST", comments_path, token,
                {"body": _task_closeout_comment_body(event)},
            )
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("id"), int)
                or isinstance(result.get("id"), bool)
            ):
                return PersistenceResult("failed", "GitHub did not confirm the closeout comment")
            # Re-read after creation. This also lets racing writers consolidate
            # their authenticated records, including conflicting payloads.
            comments = _list_repository_issue_comments(token)
            canonical_comment, canonical_event, reconciled = _reconcile_closeout_comments(
                comments, str(event["event_id"]), actor, token
            )
            if not reconciled or canonical_event is None or canonical_comment is None:
                return PersistenceResult("failed", "concurrent closeout comments could not be consolidated")
            if canonical_event != event:
                return PersistenceResult("failed", "attempt ID was concurrently used with a different payload")
            if canonical_comment.get("id") == result["id"]:
                return PersistenceResult("persisted", f"event persisted to {target_kind} #{target_number}")
            return PersistenceResult("duplicate", f"event already exists on {target_kind} #{target_number}")
    except RuntimeError as exc:
        return PersistenceResult("failed", f"GitHub closeout persistence failed ({exc})")
    except OSError as exc:
        return PersistenceResult("failed", f"could not prepare closeout lock ({type(exc).__name__})")


def persist_research_event(
    event: Mapping[str, object], *, environ: Mapping[str, str] | None = None
) -> PersistenceResult:
    """Compatibility alias; all research events use unified closeout storage."""
    return persist_task_closeout(event, environ=environ)


def _task_closeout_mail(event: Mapping[str, object], *, environ: Mapping[str, str] | None) -> Outcome:
    event_id = str(event["event_id"])
    issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(str(event["task_id"]))
    if issue_match is None:
        return Outcome("failed", "task_id does not identify an originating Issue")
    issue_number = issue_match.group(1)
    subject = f"[RobotSim] {event['status']}: Issue #{issue_number} {event['task_kind']} closeout"
    body = _event_json(event, pretty=True) + "\n"
    digest = _delivery_digest("task-closeout", event_id)
    return _deliver(subject, body, digest, dry_run=False, environ=environ)


def _research_mail(event: Mapping[str, object], *, environ: Mapping[str, str] | None) -> Outcome:
    """Backward-compatible wrapper for the former research-only notifier."""
    return _task_closeout_mail(event, environ=environ)


def extract_task_closeout_envelope(message: object) -> tuple[bool, object | None, str]:
    """Find exactly one unified or legacy marked JSON closeout in a Stop message."""
    if not isinstance(message, str):
        return False, None, ""
    has_closeout = TASK_CLOSEOUT_ENVELOPE_MARKER in message
    has_legacy = RESEARCH_ENVELOPE_MARKER in message
    if not has_closeout and not has_legacy:
        return False, None, ""
    closeout_matches = list(TASK_CLOSEOUT_ENVELOPE_RE.finditer(message))
    legacy_matches = list(RESEARCH_ENVELOPE_RE.finditer(message))
    if (
        int(has_closeout) + int(has_legacy) != 1
        or len(closeout_matches) + len(legacy_matches) != 1
        or message.count(TASK_CLOSEOUT_ENVELOPE_MARKER) > 1
        or message.count(RESEARCH_ENVELOPE_MARKER) > 1
    ):
        return True, None, "task closeout must contain exactly one marked JSON envelope"
    candidate = closeout_matches[0] if closeout_matches else legacy_matches[0]
    try:
        return True, json.loads(candidate.group(1)), ""
    except json.JSONDecodeError:
        return True, None, "task closeout envelope is not valid JSON"


def extract_research_envelope(message: object) -> tuple[bool, object | None, str]:
    """Backward-compatible name for Stop-envelope extraction."""
    return extract_task_closeout_envelope(message)


def process_task_closeout(
    payload: object,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> Outcome:
    try:
        if isinstance(payload, dict) and payload.get("schema_version") == RESEARCH_SCHEMA_VERSION:
            event = normalize_research_event(payload)
        else:
            event = normalize_task_closeout(payload)
    except (ValueError, TypeError) as exc:
        return Outcome("blocked", f"CLOSEOUT BLOCKED: invalid task closeout: {exc}", exit_code=2)

    documented_unrelated = [
        item for item in event["validation"]
        if isinstance(item, str) and item.startswith("Unrelated repository path: ")
    ]
    gate = check_closeout(
        Path.cwd() if cwd is None else cwd,
        task_id=str(event["task_id"]),
        branch=str(event["branch"] or ""),
        head_sha=str(event["head_sha"] or ""),
        pr_number=event["pr_number"] if isinstance(event["pr_number"], int) else None,
        documented_unrelated=documented_unrelated,
    )
    issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(str(event["task_id"]))
    assert issue_match is not None
    issue_url = f"https://github.com/{ROBOTSIM_REPOSITORY}/issues/{issue_match.group(1)}"
    reference_url = str(gate.get("reference_url") or issue_url)
    local_head = str(gate.get("local_head") or "unavailable")
    remote_head = str(gate.get("remote_head") or "unpublished-or-unavailable")
    actual_branch = str(gate.get("branch") or event["branch"] or "unavailable")
    gate_blockers = [str(item) for item in gate.get("blockers", [])]
    if gate.get("ok") is not True and not gate_blockers:
        gate_blockers.append("publication gate did not confirm closeout")
    result_labels = {
        "completed": "PASS",
        "failed": "FAIL",
        "blocked": "BLOCKED",
        "deferred": "BLOCKED",
        "cancelled": "BLOCKED",
    }
    if len(event["validation"]) >= 20:
        gate_blockers.append("closeout validation list has no room for publication evidence")
    if reference_url not in event["evidence"] and len(event["evidence"]) >= 20:
        gate_blockers.append("closeout evidence list has no room for the canonical GitHub URL")
    if not gate_blockers:
        if reference_url not in event["evidence"]:
            event["evidence"].append(reference_url)
        event["validation"].append(
            f"Closeout verdict: {result_labels[str(event['status'])]}; publication verified: branch={actual_branch}; local_head={local_head}; remote_head={remote_head}; GitHub={reference_url}"
        )
    else:
        original_status = str(event["status"])
        event["status"] = "blocked"
        event["pr_number"] = None
        event["branch"] = actual_branch if actual_branch != "unavailable" else event["branch"]
        if local_head != "unavailable":
            event["head_sha"] = local_head
        event["summary"] = f"CLOSEOUT BLOCKED after task result {original_status}: {event['summary']}"
        gate_detail = (
            f"Closeout verdict: BLOCKED; original task result={result_labels[original_status]}; "
            f"branch={actual_branch}; local_head={local_head}; remote_head={remote_head}; "
            f"GitHub={issue_url}; blockers={'; '.join(gate_blockers)}"
        )
        if len(event["validation"]) < 20:
            event["validation"].append(gate_detail)
        else:
            event["validation"][-1] = gate_detail
        event["blockers"] = (list(event["blockers"]) + [
            "CLOSEOUT BLOCKED: " + "; ".join(gate_blockers)
        ])[:10]
        evidence = list(event["evidence"])
        if reference_url not in evidence:
            evidence.append(reference_url)
        if issue_url not in evidence:
            evidence.append(issue_url)
        event["evidence"] = evidence[-20:]
        reference_url = issue_url

    try:
        event = normalize_task_closeout(event)
    except (ValueError, TypeError) as exc:
        return Outcome("blocked", f"CLOSEOUT BLOCKED: could not record publication result ({exc})", exit_code=2)

    issue_match = TASK_CLOSEOUT_TASK_ID.fullmatch(str(event["task_id"]))
    assert issue_match is not None
    destination = f"PR #{event['pr_number']}" if event["pr_number"] is not None else f"Issue #{issue_match.group(1)}"
    subject = f"[RobotSim] {event['status']}: {destination} {event['task_kind']} closeout"
    body = _event_json(event, pretty=True) + "\n"
    if gate_blockers:
        if dry_run:
            return Outcome(
                "blocked",
                f"CLOSEOUT BLOCKED: {'; '.join(gate_blockers)}",
                subject,
                body,
                exit_code=2,
            )
        persisted = persist_task_closeout(event, environ=environ)
        if persisted.state == "failed":
            return Outcome(
                "blocked",
                f"CLOSEOUT BLOCKED: {'; '.join(gate_blockers)}; GitHub update failed ({persisted.message})",
                subject,
                body,
                exit_code=2,
            )
        return Outcome(
            "blocked",
            f"CLOSEOUT BLOCKED: {'; '.join(gate_blockers)}; blocked record persisted to Issue #{issue_match.group(1)}",
            subject,
            body,
            exit_code=2,
        )
    if dry_run:
        return Outcome("dry_run", "no GitHub comment or AgentMail request was made", subject, body)
    persisted = persist_task_closeout(event, environ=environ)
    if persisted.state == "failed":
        return Outcome("blocked", f"CLOSEOUT BLOCKED: GitHub update failed ({persisted.message})", subject, body, exit_code=2)
    mailed = _task_closeout_mail(event, environ=environ)
    prefix = f"GitHub {persisted.message};"
    if mailed.state == "failed":
        return Outcome("failed", f"{prefix} AgentMail delivery failed ({mailed.message})", subject, body)
    if mailed.state == "skipped":
        return Outcome("skipped", f"{prefix} AgentMail skipped ({mailed.message})", subject, body)
    if mailed.state == "duplicate":
        return Outcome("duplicate", f"{prefix} AgentMail already sent", subject, body)
    return Outcome("sent", f"{prefix} AgentMail notified", subject, body)


def process_research_completion(
    payload: object,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Outcome:
    """Backward-compatible entry point using the unified closeout path."""
    return process_task_closeout(payload, dry_run=dry_run, environ=environ)
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


def config_preflight(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Report only which AgentMail setting names are missing."""
    env = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_SETTINGS if not env.get(name, "").strip()]
    return {"ready": not missing, "missing_settings": missing}


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
        # A concurrent sender may have moved its claim to .sent after our
        # initial sent.exists() check but before this exclusive create.
        if sent.exists():
            pending.unlink(missing_ok=True)
            return None, "duplicate"
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

    has_closeout_event, closeout_payload, envelope_problem = extract_task_closeout_envelope(
        payload.get("last_assistant_message")
    )
    if has_closeout_event:
        if envelope_problem:
            return Outcome("blocked", f"CLOSEOUT BLOCKED: invalid task closeout: {envelope_problem}", exit_code=2)
        return process_task_closeout(
            closeout_payload,
            dry_run=dry_run,
            environ=environ,
            cwd=payload.get("cwd") if isinstance(payload.get("cwd"), str) else None,
        )

    return Outcome(
        "skipped",
        "no structured task-closeout envelope; no terminal completion was recorded",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "event",
        choices=("ready_for_review", "task-closeout", "research-completion", "config-preflight", "stop"),
        help="attention notification, structured task closeout JSON, or 'stop' for a local Codex Stop payload",
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
    if args.event == "config-preflight":
        result = config_preflight()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ready"] else 1
    if args.event == "stop":
        try:
            payload = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError):
            outcome = Outcome("skipped", "invalid Stop hook JSON payload")
        else:
            outcome = process_stop_payload(payload, dry_run=args.dry_run)
    elif args.event in {"task-closeout", "research-completion"}:
        try:
            payload = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeDecodeError):
            outcome = Outcome("blocked", "CLOSEOUT BLOCKED: input is not valid JSON", exit_code=2)
        else:
            outcome = process_task_closeout(payload, dry_run=args.dry_run, cwd=Path.cwd())
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
        if outcome.state in {"failed", "blocked"}:
            print(f"notify_task: {outcome.state}: {outcome.message}", file=sys.stderr)
    else:
        _print_outcome(outcome)
    # Local Stop is always non-blocking. Explicit Cloud closeout commands fail
    # when canonical GitHub persistence failed, but AgentMail remains best-effort.
    if args.event == "stop" and not args.dry_run:
        return 0
    if args.event in {"task-closeout", "research-completion"}:
        return outcome.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
