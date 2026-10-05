#!/usr/bin/env python3
"""Check that a RobotSim task closeout is committed and published exactly."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Iterable


TASK_ID = re.compile(r"^issue-([1-9][0-9]{0,8})-[a-z0-9][a-z0-9._-]{0,100}$")
SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
UNRELATED_PREFIX = "Unrelated repository path: "


def _git(
    root: Path,
    *args: str,
    timeout: int = 10,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(type(exc).__name__) from None


def _text(result: subprocess.CompletedProcess[bytes]) -> str:
    return result.stdout.decode("utf-8", errors="replace").strip()


def _parse_allowlist(entries: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    allowed: dict[str, str] = {}
    errors: list[str] = []
    for entry in entries:
        if not entry.startswith(UNRELATED_PREFIX) or " | reason: " not in entry:
            errors.append("unrelated-path exceptions must use the documented path/reason format")
            continue
        path_text, reason = entry[len(UNRELATED_PREFIX):].split(" | reason: ", 1)
        directory_scope = path_text.endswith("/")
        candidate = path_text[:-1] if directory_scope else path_text
        path = PurePosixPath(candidate)
        if (
            not candidate
            or path.is_absolute()
            or "\\" in candidate
            or any(part in {"", ".", ".."} for part in candidate.split("/"))
            or not reason.strip()
            or len(reason) > 240
        ):
            errors.append("unrelated-path exception has an invalid relative path or missing reason")
            continue
        allowed[candidate + ("/" if directory_scope else "")] = reason.strip()
    return allowed, errors


def _status_paths(raw: bytes) -> list[str]:
    fields = raw.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        item = fields[index]
        index += 1
        if not item:
            continue
        if len(item) < 4 or item[2:3] != b" ":
            paths.append("<unparseable git status entry>")
            continue
        status = item[:2]
        paths.append(item[3:].decode("utf-8", errors="surrogateescape"))
        if b"R" in status or b"C" in status:
            if index < len(fields) and fields[index]:
                paths.append(fields[index].decode("utf-8", errors="surrogateescape"))
                index += 1
    return paths


def _is_allowed(path: str, allowed: dict[str, str]) -> bool:
    normalized = path.replace(os.sep, "/")
    for candidate in allowed:
        if candidate.endswith("/"):
            if normalized.startswith(candidate):
                return True
        elif normalized == candidate:
            return True
    return False


def _remote_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _contains_inline_credentials(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return True
    if parts.password is not None or (parts.scheme in {"http", "https"} and parts.username is not None):
        return True
    credential_keys = {"token", "access_token", "key", "password", "secret", "auth", "authorization"}
    if any(key.casefold() in credential_keys for key, _ in urllib.parse.parse_qsl(parts.query)):
        return True
    return False


def check_closeout(
    repo: str | os.PathLike[str],
    *,
    task_id: str,
    branch: str,
    head_sha: str,
    pr_number: int | None = None,
    documented_unrelated: Iterable[str] = (),
) -> dict[str, object]:
    """Return a non-secret diagnostic report; never returns remote URLs or stderr."""
    root = Path(repo).resolve()
    blockers: list[str] = []
    match = TASK_ID.fullmatch(task_id)
    if match is None:
        blockers.append("task_id must identify a RobotSim Issue")
    if pr_number is not None and (isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number < 1):
        blockers.append("PR reference must be a positive number")
    issue_number = match.group(1) if match else ""
    reference_url = (
        f"https://github.com/lzy18001500226/RobotSim/pull/{pr_number}"
        if pr_number is not None and isinstance(pr_number, int) and pr_number > 0
        else f"https://github.com/lzy18001500226/RobotSim/issues/{issue_number}"
        if issue_number
        else ""
    )
    allowed, allowlist_errors = _parse_allowlist(documented_unrelated)
    blockers.extend(allowlist_errors)

    try:
        top = _git(root, "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            raise RuntimeError("not a Git checkout")
        root = Path(_text(top)).resolve()
        branch_result = _git(root, "branch", "--show-current")
        current_branch = _text(branch_result) if branch_result.returncode == 0 else ""
        head_result = _git(root, "rev-parse", "HEAD")
        local_head = _text(head_result).lower() if head_result.returncode == 0 else ""
        if not current_branch:
            blockers.append("checkout is detached or current branch is unavailable")
        elif current_branch in {"main", "master"}:
            blockers.append("task closeout must use a dedicated task branch")
        elif branch != current_branch:
            blockers.append("reported branch does not match the current checkout")
        if not SHA.fullmatch(local_head):
            blockers.append("local HEAD could not be resolved")
        elif head_sha.lower() != local_head:
            blockers.append("reported head_sha does not match local HEAD")

        status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status.returncode != 0:
            blockers.append("git status could not be read")
            dirty_paths: list[str] = []
        else:
            dirty_paths = [path for path in _status_paths(status.stdout) if not _is_allowed(path, allowed)]
            if dirty_paths:
                blockers.append("working tree has uncommitted task changes")

        remote_head = ""
        push_url = _git(root, "remote", "get-url", "--all", "--push", "origin")
        push_urls = _text(push_url).splitlines() if push_url.returncode == 0 else []
        if len(push_urls) != 1:
            blockers.append("origin push remote is unavailable")
        elif _contains_inline_credentials(push_urls[0]):
            blockers.append("origin push URL contains inline credentials; use SSH or an OS credential helper")
        elif current_branch and current_branch not in {"main", "master"}:
            remote = _git(
                root,
                "ls-remote",
                "--heads",
                push_urls[0],
                f"refs/heads/{current_branch}",
                timeout=20,
                env=_remote_environment(),
            )
            if remote.returncode != 0:
                blockers.append("remote branch could not be verified")
            else:
                lines = [line.split() for line in _text(remote).splitlines() if line.split()]
                matching = [parts[0].lower() for parts in lines if len(parts) == 2 and parts[1] == f"refs/heads/{current_branch}"]
                remote_head = matching[0] if len(matching) == 1 else ""
                if not remote_head:
                    blockers.append("task branch is not present on the push remote")
                elif remote_head != local_head:
                    blockers.append("remote branch SHA differs from local HEAD")
    except RuntimeError as exc:
        current_branch = ""
        local_head = ""
        remote_head = ""
        dirty_paths = []
        blockers.append(f"Git publication check failed ({exc})")

    if not reference_url:
        blockers.append("a corresponding GitHub Issue or PR reference is required")
    return {
        "ok": not blockers,
        "status": "PASS" if not blockers else "CLOSEOUT BLOCKED",
        "branch": current_branch,
        "local_head": local_head,
        "remote_head": remote_head,
        "reference_url": reference_url,
        "dirty_paths": dirty_paths,
        "allowed_unrelated_paths": sorted(allowed),
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="RobotSim Git checkout")
    parser.add_argument("--task-id", required=True, help="issue-<number>-<stable-slug>")
    parser.add_argument("--branch", required=True, help="expected task branch")
    parser.add_argument("--head-sha", required=True, help="expected local HEAD SHA")
    parser.add_argument("--pr-number", type=int, help="corresponding PR; otherwise the task Issue is used")
    parser.add_argument(
        "--allow-unrelated",
        action="append",
        default=[],
        metavar="PATH=REASON",
        help="exact repo-relative unrelated path exception; append / to allow a subtree",
    )
    args = parser.parse_args(argv)
    entries = [f"{UNRELATED_PREFIX}{value.split('=', 1)[0]} | reason: {value.split('=', 1)[1]}" if "=" in value else value for value in args.allow_unrelated]
    report = check_closeout(
        args.repo,
        task_id=args.task_id,
        branch=args.branch,
        head_sha=args.head_sha,
        pr_number=args.pr_number,
        documented_unrelated=entries,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
