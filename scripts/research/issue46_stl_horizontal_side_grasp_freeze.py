#!/usr/bin/env python3
"""Freeze the human-accepted horizontal STL side-grasp visual baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-state", required=True, type=Path)
    args = parser.parse_args()
    if not args.review_state.is_file():
        raise SystemExit("review state is missing")
    review = json.loads(args.review_state.read_text(encoding="utf-8"))
    if review.get("mode") != "STATIC_MJ_FORWARD_ONLY" or review.get("mj_step_calls") != 0:
        raise SystemExit("review state is not a zero-step static review")
    if review.get("model", {}).get("id") != MODEL_ID:
        raise SystemExit("review state identifies a different gripper")
    scene = Path(review["scene"]["path"])
    if not scene.is_file() or sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("review scene is missing or hash-mismatched")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != review["model"]["sha256"]:
        raise SystemExit("registered STL model hash differs from accepted review")
    gates = review.get("gates", {})
    required = ("UPRIGHT_CYLINDER", "UPPER_BOTTLE_TARGET", "FOREARM_LEVEL", "DISTAL_REFERENCE_LEVEL", "DISTAL_PAD_ONLY_CONTACT")
    if any(gates.get(name) != "PASS" for name in required):
        raise SystemExit("accepted review lacks a required static geometry gate")

    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-side-grasp-freeze")
    output = location.native_path
    record = {
        "schema": "robotsim.issue46.stl.horizontal-side-grasp-freeze.v1",
        "baseline_id": "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1",
        "classification": "HUMAN_ACCEPTED_STATIC_IK_REFERENCE",
        "human_acceptance": "User accepted the upright-cylinder, horizontal-forearm, side-pinch visual semantics in this chat.",
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "accepted_review": {
            "path": str(args.review_state),
            "sha256": sha256(args.review_state),
            "scene": review["scene"],
            "measurements": review["measurements"],
            "gates": {name: gates[name] for name in required},
        },
        "frozen_inputs": [
            "X2-OmniPicker-STL gripper geometry and registered kinematic trajectory",
            "official-source visual meshes and the reviewed source scene",
            "upright 70 mm diameter, 300 mm tall, 0.20 kg cylinder geometry and table geometry",
            "horizontal side-grasp semantic: level elbow-to-wrist line and two level distal-link references",
        ],
        "allowed_next_variables": [
            "right-arm IK approach distance along the horizontal side-pinch normal",
            "vertical grasp height on the upright bottle body",
            "horizontal approach azimuth around the bottle",
            "single frozen-gripper root command",
        ],
        "prohibited_next_changes": [
            "STL/URDF/vendor-asset edits",
            "gripper joint topology, coupling, collision-proxy or collision-exclusion changes",
            "object attachment, weld, runtime object qpos write, or hidden support",
        ],
        "qualification_boundary": {
            "static_review": "PASS",
            "physical_grasp": "NOT_RUN",
            "dynamics": "NOT_RUN",
            "manufacturer_equivalence": "NOT_CLAIMED",
        },
    }
    pg.write_new_text(output / "freeze_record.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"freeze_record": str(output / "freeze_record.json"), "baseline_id": record["baseline_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
