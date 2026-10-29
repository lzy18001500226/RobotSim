#!/usr/bin/env python3
"""Extract a consumer-safe TABLE_APPROACH balance snapshot from one passed run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class HandoffError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_complete_post_step(trace: Path) -> dict[str, Any]:
    if trace.is_symlink() or not trace.is_file():
        raise HandoffError("native MjData trace is missing or unsafe")
    latest = None
    lines = trace.read_bytes().splitlines(keepends=True)
    for index, raw in enumerate(lines, start=1):
        if not raw.endswith(b"\n"):
            if index == len(lines):
                break
            raise HandoffError("native trace has an incomplete non-final record")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HandoffError(f"native trace JSON is invalid at line {index}") from exc
        if row.get("event") == "post_step" and all(key in row for key in ("qpos", "qvel", "ctrl", "pelvis", "feet", "contacts")):
            latest = row
    if not isinstance(latest, dict):
        raise HandoffError("native trace has no complete post-step balance record")
    return latest


def extract(run_dir: Path) -> dict[str, Any]:
    root = run_dir.resolve(strict=True)
    result_path = root / "m1_first_motion_result.json"
    auto_path = root / "auto-reset-run-result.json"
    trace = root / "native_mjdata_trace.jsonl"
    route = root / "route_config.json"
    if any(path.is_symlink() or not path.is_file() for path in (result_path, auto_path, trace, route)):
        raise HandoffError("required TABLE_APPROACH evidence input is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS"
        or result.get("classification") != "SIMULATION_ONLY_VENDOR_DIRECT_TRUTH_TABLE_TARGET"
        or result.get("terminal_reason") != "target_reached"
        or result.get("physical_stop", {}).get("status") != "PASS"
    ):
        raise HandoffError("run is not a passed TABLE_APPROACH physical-stop result")
    row = last_complete_post_step(trace)
    pelvis = row.get("pelvis", {}).get("state")
    feet = row.get("feet")
    contacts = row.get("contacts")
    if not isinstance(pelvis, dict) or pelvis.get("name") != "pelvis" or not isinstance(feet, dict) or not isinstance(contacts, list):
        raise HandoffError("balance record lacks named pelvis, feet, or contacts")
    return {
        "schema": "robotsim.issue80.local04-balance-at-approach.v1",
        "classification": "SIMULATION_ONLY_TABLE_APPROACH_POST_STOP_NATIVE_MJDATA",
        "status": "PASS_BALANCE_AT_APPROACH_ENDPOINT",
        "run_directory": str(root),
        "target_pose_odom": result.get("target_pose_odom"),
        "final_pose_odom": result.get("final_pose_odom"),
        "final_target_error_m": result.get("final_target_error_m"),
        "physical_stop": result.get("physical_stop"),
        "native_post_stop_sample": {
            "generation": row.get("generation"),
            "sim_time_s": row.get("sim_time_s"),
            "host_monotonic_ns": row.get("host_monotonic_ns"),
            "qpos": row.get("qpos"),
            "qvel": row.get("qvel"),
            "ctrl": row.get("ctrl"),
            "pelvis": pelvis,
            "feet": feet,
            "contacts": contacts,
            "contact_count": row.get("contact_count"),
        },
        "source_sha256": {
            "m1_first_motion_result.json": sha256(result_path),
            "auto-reset-run-result.json": sha256(auto_path),
            "native_mjdata_trace.jsonl": sha256(trace),
            "route_config.json": sha256(route),
        },
        "limitations": [
            "TABLE_DOCKING and grasp are NOT RUN.",
            "Native trace final partial record, if any, is skipped; this is the latest complete post_step record.",
            "This is a MuJoCo SIMULATION_ONLY state, not hardware balance evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be new")
    try:
        payload = extract(args.run_dir)
    except (OSError, HandoffError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
