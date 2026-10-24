#!/usr/bin/env python3
"""Verify X2 runtime processes from procfs without relying on truncated comm names."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def resolve_argv0(argv0: str, cwd: str) -> str:
    if not argv0:
        return ""
    return os.path.realpath(argv0 if os.path.isabs(argv0) else os.path.join(cwd, argv0))


def _read_proc(pid: int, expected_names: set[str] | None = None) -> dict[str, Any] | None:
    proc = Path(f"/proc/{pid}")
    try:
        exe_link = os.readlink(proc / "exe")
        exe_path = os.path.realpath(exe_link.removesuffix(" (deleted)"))
        exe_basename = Path(exe_path).name
        if expected_names is not None and exe_basename not in expected_names:
            return None
        argv = [arg.decode("utf-8", "surrogateescape") for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
        cwd = os.path.realpath(proc / "cwd")
        comm = (proc / "comm").read_text(encoding="utf-8", errors="replace").strip()
        stat = (proc / "stat").read_text(encoding="utf-8", errors="replace")
        # The comm field may contain spaces or parentheses. Parse after its final ')'.
        fields_after_comm = stat[stat.rfind(")") + 2 :].split()
        start_ticks = int(fields_after_comm[19])  # proc stat field 22, after fields 1-2 are removed.
        digest = hashlib.sha256()
        try:
            with (proc / "exe").open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            exe_sha256: str | None = digest.hexdigest()
        except PermissionError:
            # Exact path and cmdline remain useful for conflict detection; the
            # runtime gate later requires a readable binary hash.
            exe_sha256 = None
        argv0 = argv[0] if argv else ""
        argv0_path = resolve_argv0(argv0, cwd)
        return {
            "pid": pid,
            "comm": comm,
            "exe_link": exe_link,
            "exe_path": exe_path,
            "exe_basename": exe_basename,
            "exe_sha256": exe_sha256,
            "cwd": cwd,
            "argv": argv,
            "argv0_path": argv0_path,
            "start_ticks": start_ticks,
            "cmdline_identity_verified": bool(argv0 and argv0_path == exe_path),
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
        return None


def scan_processes(names: Iterable[str]) -> list[dict[str, Any]]:
    expected_names = set(names)
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        row = _read_proc(int(entry.name), expected_names)
        if row and row["exe_basename"] in expected_names:
            row["application_identity_verified"] = row["cmdline_identity_verified"]
            rows.append(row)
    return sorted(rows, key=lambda row: row["pid"])


def verify_expected(row: dict[str, Any] | None, executable: str, sha256: str,
                    start_ticks: int | None = None) -> tuple[bool, str]:
    if row is None:
        return False, "pid_not_running_or_proc_unreadable"
    expected_path = os.path.realpath(executable)
    if row["exe_path"] != expected_path:
        return False, "executable_path_mismatch"
    if row["exe_sha256"] != sha256:
        return False, "executable_sha256_mismatch"
    if not row["cmdline_identity_verified"] or row["argv0_path"] != expected_path:
        return False, "cmdline_argv0_does_not_resolve_to_executable"
    if start_ticks is not None and row["start_ticks"] != start_ticks:
        return False, "pid_reused_or_process_restarted"
    return True, "verified"


def wait_for_expected(executable: str, sha256: str, timeout_s: float, poll_s: float) -> dict[str, Any]:
    name = Path(executable).name
    deadline = time.monotonic() + timeout_s
    last_candidates: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last_candidates = scan_processes([name])
        if last_candidates:
            verified: list[dict[str, Any]] = []
            invalid: list[dict[str, Any]] = []
            for row in last_candidates:
                valid, reason = verify_expected(row, executable, sha256)
                if valid:
                    verified.append(row)
                else:
                    invalid.append({"pid": row["pid"], "reason": reason, "process": row})
            if invalid:
                return {"status": "IDENTITY_MISMATCH", "processes": verified, "invalid_candidates": invalid}
            if len(verified) != 1:
                return {"status": "AMBIGUOUS_PROCESS_COUNT", "processes": verified,
                        "candidate_count": len(verified)}
            return {"status": "VERIFIED", "process": verified[0]}
        time.sleep(poll_s)
    return {"status": "PROCESS_NOT_FOUND", "expected_executable": os.path.realpath(executable),
            "expected_sha256": sha256, "candidates": last_candidates, "timeout_s": timeout_s}


def verify_pid(pid: int, executable: str, sha256: str, start_ticks: int) -> dict[str, Any]:
    row = _read_proc(pid)
    valid, reason = verify_expected(row, executable, sha256, start_ticks)
    return {"verified": valid, "reason": reason, "process": row}


def signal_pid(pid: int, executable: str, sha256: str, start_ticks: int) -> dict[str, Any]:
    state = verify_pid(pid, executable, sha256, start_ticks)
    if not state["verified"]:
        if state["reason"] == "pid_not_running_or_proc_unreadable":
            return {**state, "status": "ALREADY_EXITED"}
        return {**state, "status": "IDENTITY_REFUSED_SIGNAL"}
    os.kill(pid, signal.SIGTERM)
    return {"status": "SIGTERM_SENT", "verified_process": state["process"]}


def wait_for_exit(pid: int, executable: str, sha256: str, start_ticks: int, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = verify_pid(pid, executable, sha256, start_ticks)
    while time.monotonic() < deadline and last["verified"]:
        time.sleep(0.05)
        last = verify_pid(pid, executable, sha256, start_ticks)
    if last["verified"]:
        return {"status": "STILL_RUNNING_AFTER_TIMEOUT", "process": last["process"], "timeout_s": timeout_s}
    if last["reason"] == "pid_reused_or_process_restarted":
        return {"status": "ORIGINAL_EXITED_PID_REUSED", "replacement_process": last["process"]}
    return {"status": "EXITED", "last_check": last, "timeout_s": timeout_s}


def emit(payload: dict[str, Any], status: int) -> int:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--name", action="append", required=True)
    wait = sub.add_parser("wait")
    wait.add_argument("--expected-executable", required=True)
    wait.add_argument("--expected-sha256", required=True)
    wait.add_argument("--timeout-s", type=float, default=15.0)
    wait.add_argument("--poll-s", type=float, default=0.2)
    verify = sub.add_parser("verify")
    verify.add_argument("--pid", type=int, required=True)
    verify.add_argument("--expected-executable", required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--start-ticks", type=int, required=True)
    terminate = sub.add_parser("terminate")
    terminate.add_argument("--pid", type=int, required=True)
    terminate.add_argument("--expected-executable", required=True)
    terminate.add_argument("--expected-sha256", required=True)
    terminate.add_argument("--start-ticks", type=int, required=True)
    exit_wait = sub.add_parser("wait-exit")
    exit_wait.add_argument("--pid", type=int, required=True)
    exit_wait.add_argument("--expected-executable", required=True)
    exit_wait.add_argument("--expected-sha256", required=True)
    exit_wait.add_argument("--start-ticks", type=int, required=True)
    exit_wait.add_argument("--timeout-s", type=float, default=1.0)
    args = parser.parse_args()

    if args.command == "scan":
        rows = scan_processes(args.name)
        unverified = [row for row in rows if not row["application_identity_verified"]]
        return emit({"processes": rows, "unverified_candidates": unverified}, 2 if unverified else 0)
    if args.command == "wait":
        if args.timeout_s <= 0 or args.poll_s <= 0:
            return emit({"status": "INVALID_TIMEOUT"}, 2)
        result = wait_for_expected(args.expected_executable, args.expected_sha256,
                                   args.timeout_s, args.poll_s)
        return emit(result, 0 if result.get("status") == "VERIFIED" else 2)
    if args.command == "verify":
        result = verify_pid(args.pid, args.expected_executable, args.expected_sha256, args.start_ticks)
        return emit(result, 0 if result["verified"] else 2)
    if args.command == "terminate":
        try:
            result = signal_pid(args.pid, args.expected_executable, args.expected_sha256, args.start_ticks)
        except (PermissionError, ProcessLookupError, OSError) as exc:
            return emit({"status": "SIGNAL_ERROR", "error": f"{type(exc).__name__}: {exc}"}, 2)
        return emit(result, 0 if result["status"] in {"SIGTERM_SENT", "ALREADY_EXITED"} else 2)
    result = wait_for_exit(args.pid, args.expected_executable, args.expected_sha256,
                           args.start_ticks, args.timeout_s)
    return emit(result, 0 if result["status"] in {"EXITED", "ORIGINAL_EXITED_PID_REUSED"} else 2)


if __name__ == "__main__":
    raise SystemExit(main())
