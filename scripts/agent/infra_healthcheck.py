#!/usr/bin/env python3
"""Read-only checkout, hook configuration and redacted notification diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Mapping, Sequence

try:
    from . import task_notifications
except ImportError:
    import task_notifications


HANDLERS = {
    "PreToolUse": "pre_tool_guard.py",
    "UserPromptSubmit": "task_lifecycle.py",
    "Interrupt": "task_lifecycle.py",
    "Stop": "agentmail_stop_hook.py",
}
MAIL_SETTINGS = ("AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "ROBOTSIM_NOTIFY_TO")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *args],
        text=True, capture_output=True, timeout=10, check=True,
    )
    return result.stdout.strip()


def _hooks(root: Path) -> dict:
    path = root / ".codex" / "hooks.json"
    try:
        hooks = json.loads(path.read_text(encoding="utf-8"))["hooks"]
        if not isinstance(hooks, dict):
            raise ValueError("invalid hooks object")
        events = {}
        for event, handler in HANDLERS.items():
            entries = hooks.get(event, [])
            if not isinstance(entries, list):
                raise ValueError("invalid event entries")
            commands = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
                    raise ValueError("invalid hook entry")
                if not all(isinstance(command, dict) for command in entry["hooks"]):
                    raise ValueError("invalid command entry")
                commands.extend(entry["hooks"])
            expected = f'python3 "$(git rev-parse --show-toplevel)/scripts/agent/{handler}"'
            events[event] = {
                "configured": bool(commands),
                "handler": f"scripts/agent/{handler}",
                "handler_exists": (root / "scripts" / "agent" / handler).is_file(),
                "linux_command_matches": bool(commands) and all(
                    command.get("type") == "command" and command.get("command") == expected
                    for command in commands
                ),
                "windows_command_configured": bool(commands) and all(
                    bool(command.get("commandWindows")) for command in commands
                ),
            }
        return {
            "status": "PASS" if all(item["configured"] and item["handler_exists"]
                                      and item["linux_command_matches"] for item in events.values()) else "FAIL",
            "events": events,
            "client_trust_enabled_and_invocation": "NOT CHECKED",
            "detail": "configuration/file checks only; inspect /hooks and a live trace in the target client",
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "FAIL", "detail": "hook configuration missing or invalid; contents omitted"}


def _stamp(paths: list[Path]) -> tuple:
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            result.append(None)
    return tuple(result)


def _outbox(path: Path) -> dict:
    """Never call the ledger's creating/migrating _connect or snapshot helpers."""
    paths = [path, Path(str(path) + "-wal"), Path(str(path) + "-journal")]
    try:
        before = _stamp(paths)
        if before[0] is None:
            return {"status": "ABSENT", "detail": "no database created"}
        if any(item and item[1] for item in before[1:]):
            return {"status": "DEFERRED", "detail": "active WAL/journal; skipped to avoid stale counts or SQLite sidecar writes"}
        # immutable mode cannot create journals or shared-memory sidecars. Refuse
        # active journals and discard counts if the database changes during read.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", timeout=1)) as connection:
            publication = {}
            for state, count in connection.execute("SELECT publication_state, COUNT(*) FROM outcomes GROUP BY publication_state"):
                key = state if state in task_notifications.PUBLICATION_STATES else "UNKNOWN"
                publication[key] = publication.get(key, 0) + count
            delivery = {}
            for channel, state, count in connection.execute("SELECT channel, state, COUNT(*) FROM deliveries GROUP BY channel, state"):
                key = channel if channel in {"agentmail", "telephone"} else "UNKNOWN"
                state = state if state in task_notifications.DELIVERY_STATES else "UNKNOWN"
                bucket = delivery.setdefault(key, {})
                bucket[state] = bucket.get(state, 0) + count
            jobs = {}
            for state, count in connection.execute("SELECT state, COUNT(*) FROM publication_jobs GROUP BY state"):
                key = state if state in task_notifications.PUBLICATION_STATES | {"sending", "disabled"} else "UNKNOWN"
                jobs[key] = jobs.get(key, 0) + count
        if _stamp(paths) != before:
            return {"status": "DEFERRED", "detail": "database changed during inspection; counts discarded"}
        return {"status": "READ", "publication_counts": publication, "delivery_counts": delivery,
                "publication_job_counts": jobs, "detail": "local state only; no external delivery verification"}
    except (OSError, sqlite3.Error, ValueError):
        return {"status": "FAIL", "detail": "database unreadable or incompatible; no migration attempted"}


def inspect(root: Path, environ: Mapping[str, str] | None = None) -> dict:
    env = os.environ if environ is None else environ
    root = root.resolve()
    checkout = {"path": str(root)}
    try:
        actual_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
        checkout.update(status="PASS" if actual_root == root else "FAIL", actual_root=str(actual_root),
                        branch=_git(root, "branch", "--show-current"), head=_git(root, "rev-parse", "HEAD"),
                        worktree_status=_git(root, "status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        checkout.update(status="FAIL", detail="Git checkout inspection failed; command output omitted")
    database = task_notifications._database_path(env).resolve()
    return {
        "schema": "robotsim.infra-health.v1",
        "read_only": True,
        "checkout": checkout,
        "hooks": _hooks(root),
        "notifications": {
            "scope": "this process environment only; not other Codex threads",
            "local_stop_opt_in": env.get("ROBOTSIM_LOCAL_STOP_HOOK") == "1",
            "environment_setting_present": {name: bool(env.get(name, "").strip()) for name in MAIL_SETTINGS},
            "windows_credential_bridge": "NOT CHECKED",
            "delivery_probe": "NOT RUN",
            "detail": "no secrets loaded, worker started, GitHub write or email sent",
        },
        "outbox": {"path": str(database), **_outbox(database)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    report = inspect(args.repo_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(any(report[key]["status"] == "FAIL" for key in ("checkout", "hooks", "outbox")))


if __name__ == "__main__":
    raise SystemExit(main())
