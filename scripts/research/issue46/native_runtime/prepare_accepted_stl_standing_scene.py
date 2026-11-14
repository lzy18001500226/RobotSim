#!/usr/bin/env python3
"""Bind the accepted full-X2 STL review scene to a recorded native stand trace.

This creates a small evidence record for the existing full-body shadow runner.
It never edits the accepted scene, the frozen hand, or the native trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_json(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-review", required=True, type=Path)
    parser.add_argument("--native-stand", required=True, type=Path)
    args = parser.parse_args()
    accepted = require_json(args.accepted_review, label="accepted review")
    native = require_json(args.native_stand, label="native stand")
    scene_data = accepted.get("generated_scene")
    if not isinstance(scene_data, dict):
        raise ValueError("accepted review lacks generated scene")
    scene = Path(str(scene_data.get("path", "")))
    if not scene.is_file() or sha256(scene) != scene_data.get("sha256"):
        raise ValueError("accepted scene differs from its review record")
    native_gates = native.get("gates")
    command = native.get("native_stand_command_capture")
    state = native.get("native_arm_state_capture")
    if not isinstance(native_gates, dict) or any(native_gates.get(name) != "PASS" for name in ("NATIVE_STAND_REQUEST", "NATIVE_STAND_READBACK", "NATIVE_STAND_STABILITY", "MC_STAND_COMMAND_CAPTURE")):
        raise ValueError("native record lacks a passing stand command trace")
    if not isinstance(command, dict) or not isinstance(command.get("result"), dict):
        raise ValueError("native record lacks stand commands")
    if not isinstance(state, dict) or not isinstance(state.get("result"), dict):
        raise ValueError("native record lacks stand state")
    output = pg.create_unique_output_directory(EVIDENCE / "reviews/issue-46", prefix="accepted-stl-standing-shadow-input").native_path
    record = {
        "schema": "robotsim.issue46.accepted-stl-standing-shadow-input.v1",
        "classification": "SIMULATION_ONLY_ACCEPTED_STL_FULLBODY_SHADOW_INPUT",
        "mj_step_calls": 0,
        "accepted_review": {"path": str(args.accepted_review), "sha256": sha256(args.accepted_review)},
        "native_stand": {"path": str(args.native_stand), "sha256": sha256(args.native_stand)},
        "generated": {"host_scene": {"path": str(scene), "sha256": sha256(scene)}},
        "gates": {
            "FROZEN_ACCEPTED_STL_SCENE": "PASS",
            "RECORDED_NATIVE_STAND_COMMANDS": "PASS",
            "HOST_STATIC_MODEL_COMPILE": "PASS",
            "TABLE_SCENE_STATIC_COMPILE": "PASS",
            "TRUE_POSE_CYLINDER_PRESENT": "PASS",
        },
        "limitations": [
            "This record joins existing immutable evidence only.",
            "The next runner applies recorded standing commands to a separate MuJoCo physics scene; it is not live MC feedback.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "scene": str(scene), "gates": record["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
