#!/usr/bin/env python3
"""Persist redacted Codex turn lifecycle metadata without retaining message text."""

from __future__ import annotations

import json
import sys

try:
    from .task_notifications import record_turn_event
except ImportError:  # pragma: no cover - direct script execution
    from task_notifications import record_turn_event


def main() -> int:
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        return 0
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    try:
        record_turn_event(payload)
    except Exception:
        print("RobotSim lifecycle metadata could not be recorded.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
