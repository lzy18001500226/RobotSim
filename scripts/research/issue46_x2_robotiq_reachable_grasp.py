#!/usr/bin/env python3
"""Evaluate the isolated-lift-calibrated X2 + Robotiq grasp frame without dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue46_x2_robotiq_m0 as m0
import issue46_x2_robotiq_reachability as reach


DEFAULT_OUT = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "robotiq-reachable-contact-20261009/grasp-frames"
)
# This station is the previously measured clear outboard placement. The TCP
# offset is taken from the successful isolated Robotiq lift, relative to the
# canonical bottle body center; it is not a newly optimized grasp point.
STATION = [-0.08, 0.18, 0.68]
CANDIDATES = (
    {
        "name": "isolated_lift_calibrated_plus_x_approach",
        "base": STATION,
        "approach_yaw_deg": 0.0,
        "pitch_about_jaw_axis_deg": 0.0,
        "grasp_offset_world_m": [-0.007, 0.0, 0.015],
        "rationale": "Reuse the pad-midpoint offset and neutral gripper orientation from the successful isolated 50 mm lift; keep the prior outboard station that cleared the table legs.",
    },
    {
        "name": "isolated_lift_calibrated_plus_y_approach",
        "base": STATION,
        "approach_yaw_deg": 270.0,
        "pitch_about_jaw_axis_deg": 0.0,
        "grasp_offset_world_m": [-0.007, 0.0, 0.015],
        "rationale": "Use the same bottle-relative contact point on the cylindrical body, but approach from the established outboard +Y side; the cylinder is rotationally symmetric and this materially changes the arm corridor.",
    },
    {
        "name": "isolated_lift_calibrated_forward_station",
        "base": [0.0, 0.16, 0.68],
        "approach_yaw_deg": 0.0,
        "pitch_about_jaw_axis_deg": 0.0,
        "grasp_offset_world_m": [-0.007, 0.0, 0.015],
        "rationale": "Move the fixed base 80 mm toward the target and 20 mm inward from the established station to reduce the measured arm-extension deficit while retaining the +X corridor and measured contact frame.",
    },
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=m0.X2_DEFAULT)
    parser.add_argument("--menagerie-root", type=Path, default=m0.MENAGERIE_DEFAULT)
    parser.add_argument("--canonical-helper", type=Path, default=m0.CANONICAL_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-name", action="append",
                        help="Run only the named bounded candidate; may be repeated")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = [c for c in CANDIDATES if not args.candidate_name or c["name"] in args.candidate_name]
    if args.candidate_name and len(candidates) != len(set(args.candidate_name)):
        raise ValueError("Unknown or duplicate --candidate-name")
    script = Path(__file__).resolve()
    repo = script.parents[2]
    result: dict[str, Any] = {
        "experiment": "Bounded geometry-derived source-limited X2 + Robotiq grasp frames; static only",
        "status": "BLOCKED_STATIC_REACHABILITY",
        "runner": {
            "path": str(script), "sha256": sha256(script),
            "solver_path": str(Path(reach.__file__).resolve()),
            "solver_sha256": sha256(Path(reach.__file__).resolve()),
            "robot_sim_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "command": sys.argv,
        },
        "candidate_budget": {"maximum": len(CANDIDATES), "selected_names": [c["name"] for c in candidates],
                             "candidates": CANDIDATES},
        "physics_run": False,
        "contact_exclusion": {
            "name": "issue46_simonly_head_pitch_torso",
            "pair": ["head_pitch_link", "torso_link"],
            "additional_exclusions": 0,
        },
        "candidates": [],
    }
    try:
        if sha256(args.canonical_helper) != m0.CANONICAL_SHA:
            raise RuntimeError("Canonical scene helper hash differs from the accepted immutable helper")
        ident = m0.identity(args, args.canonical_helper)
        canonical = m0.load_module(args.canonical_helper)
        result["identity"] = ident
        for config in candidates:
            index = CANDIDATES.index(config) + 1
            candidate = reach.evaluate_candidate(
                index=index,
                config=config,
                args=argparse.Namespace(
                    output_dir=out,
                    x2_root=args.x2_root,
                    menagerie_root=args.menagerie_root,
                ),
                ident=ident,
                canonical=canonical,
                source_pose=None,
            )
            raw_path = out / config["name"] / "raw" / "candidate_trace.json"
            write_json(raw_path, candidate)
            summary = {k: candidate.get(k) for k in (
                "candidate_index", "configuration", "status", "model",
                "grasp_frame_definition", "gates", "screenshots",
            )}
            summary["raw_candidate_trace"] = str(raw_path)
            result["candidates"].append(summary)
            write_json(out / "result.json", result)
            if candidate.get("status") == "STATIC_PASS":
                result["status"] = "STATIC_PASS"
                result["selected_candidate"] = config["name"]
                break
        if result["status"] != "STATIC_PASS":
            result["status"] = "BLOCKED_STATIC_REACHABILITY"
    except Exception as exc:
        result["status"] = "FAIL"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result_json": str(out / "result.json"),
                      "candidate_summaries": [{"configuration": c.get("configuration"), "status": c.get("status"),
                                               "gates": c.get("gates")} for c in result["candidates"]],
                      "failure": result.get("failure")}, indent=2))
    return 0 if result["status"] == "STATIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
