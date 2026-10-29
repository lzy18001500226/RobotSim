#!/usr/bin/env python3
"""Open one static TABLE_APPROACH STL IK candidate in the native MuJoCo GUI.

The viewer calls ``mj_forward`` once and then only synchronizes its display.
It deliberately does not call ``mj_step`` or issue actuator controls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_candidate(audit: dict[str, object], candidate_index: int) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, object]]:
    candidates = audit.get("valid_dual_pad_candidates", [])
    if not 0 <= candidate_index < len(candidates):
        raise RuntimeError("candidate index is outside the valid dual-pad candidate set")
    overlay = Path(audit["overlay"]["path"])
    review_path = Path(audit["review_state"]["path"])
    if not overlay.is_file() or sha256(overlay) != audit["overlay"]["sha256"]:
        raise RuntimeError("static overlay is missing or has changed")
    if not review_path.is_file() or sha256(review_path) != audit["review_state"]["sha256"]:
        raise RuntimeError("review state is missing or has changed")
    candidate = candidates[candidate_index]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    model = mujoco.MjModel.from_xml_path(str(overlay))
    data = mujoco.MjData(model)
    q = candidate["joint_q_rad"]
    arm = np.asarray([q[name] for name in static.ARM_JOINTS], dtype=float)
    waist = {name: float(q[name]) for name in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")}
    static.apply_prior_state(model, data, review["state"], arm, waist)
    force.apply_positions(model, data, force.fit_positions(float(candidate["command_percent"])))
    mujoco.mj_forward(model, data)
    return model, data, candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.audit.is_file():
        raise SystemExit("static audit result is missing")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("classification") != "SIMULATION_ONLY_TABLE_APPROACH_STATIC_IK" or audit.get("gates", {}).get("STATIC_IK") != "PASS":
        raise SystemExit("audit is not a passing TABLE_APPROACH static-IK result")
    model, data, candidate = apply_candidate(audit, args.candidate_index)
    summary = {
        "viewer": "MuJoCo static native GUI; 0 mj_step",
        "candidate_index": args.candidate_index,
        "height_offset_mm": candidate["height_offset_mm"],
        "command_percent": candidate["command_percent"],
        "contacts": candidate["contacts"],
        "robot_table_contacts": candidate["robot_table_contacts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
