#!/usr/bin/env python3
"""Run one observer child and atomically record completion evidence.

The wrapper is used only inside the isolated runtime container. It is equally
usable with tiny fake observer processes in offline regression tests.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Sequence

from issue80_raicom_terminal_protocol import (
    PROCESS_EXIT_FILENAME,
    TERMINAL_MANIFEST_FILENAME,
    TerminalProtocolError,
    publish_process_exit,
    validate_terminal_manifest,
)


def run_child(
    command: Sequence[str],
    *,
    output_dir: Path,
    timeout_s: float,
    required_summary_filenames: Sequence[str],
) -> int:
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    child = subprocess.Popen(list(command), start_new_session=True)
    timed_out = False
    error: str | None = None
    try:
        returncode = child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1.0)
        returncode = child.returncode if child.returncode is not None else 124

    terminal_validation = None
    if returncode == 0 and not timed_out:
        try:
            manifest = validate_terminal_manifest(
                output_dir,
                required_summary_filenames=required_summary_filenames,
            )
            terminal_validation = {
                "status": "PASS",
                "terminal_file": TERMINAL_MANIFEST_FILENAME,
                "observer_status": manifest.get("observer_status"),
                "summary_filenames": list(required_summary_filenames),
            }
        except TerminalProtocolError as exc:
            error = f"{type(exc).__name__}: {exc}"
            terminal_validation = {"status": "BLOCKED", "error": error}
    else:
        error = "OBSERVER_TIMEOUT" if timed_out else f"OBSERVER_EXIT_{returncode}"
        terminal_validation = {"status": "BLOCKED", "error": error}

    publish_process_exit(
        output_dir,
        returncode=returncode,
        timed_out=timed_out,
        terminal_validation=terminal_validation,
        error=error,
    )
    return 0 if returncode == 0 and not timed_out and error is None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--required-summary", action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("observer command is required after --")
    try:
        return run_child(
            command,
            output_dir=args.output_dir,
            timeout_s=args.timeout_s,
            required_summary_filenames=args.required_summary,
        )
    except Exception as exc:
        try:
            publish_process_exit(
                args.output_dir,
                returncode=2,
                timed_out=False,
                terminal_validation={"status": "BLOCKED", "error": repr(exc)},
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
