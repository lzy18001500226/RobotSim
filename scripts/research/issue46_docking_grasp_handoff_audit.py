#!/usr/bin/env python3
"""Audit a navigation result before it may hand off to OmniPicker grasp IK."""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-input", required=True, type=Path)
    parser.add_argument("--motion-result", required=True, type=Path)
    args = parser.parse_args()
    if not args.route_input.is_file() or not args.motion_result.is_file():
        raise SystemExit("route input and motion result are required")
    route = json.loads(args.route_input.read_text(encoding="utf-8"))
    motion = json.loads(args.motion_result.read_text(encoding="utf-8"))
    if route.get("classification") != "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION":
        raise SystemExit("route input is not an auditable TABLE_APPROACH source")
    docking = route.get("table_docking_reference", {})
    pose = docking.get("pose", {})
    if route.get("frame_id") != "odom" or set(pose) != {"x_m", "y_m", "yaw_rad", "z_m"}:
        raise SystemExit("route lacks an odom TABLE_DOCKING reference pose")
    stop = motion.get("physical_stop", {})
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-docking-grasp-handoff-audit")
    output = location.native_path
    handoff = {
        "schema": "robotsim.issue46.docking-grasp-handoff.v1",
        "classification": "NAVIGATION_TO_GRASP_HANDOFF_CONTRACT",
        "mj_step_calls": 0,
        "route_input": {"path": str(args.route_input), "sha256": sha256(args.route_input)},
        "motion_result": {"path": str(args.motion_result), "sha256": sha256(args.motion_result)},
        "docking_pose": {"frame_id": "odom", **pose},
        "table_approach": route["poses"]["TABLE_APPROACH"],
        "final_docking_delta_m": docking.get("distance_from_table_approach_m"),
        "required_handoff_preconditions": {
            "navigation_pose_goal": "TABLE_DOCKING, not TABLE_APPROACH",
            "base_frame": "odom",
            "physical_stop": {"minimum_duration_s": 1.9, "maximum_planar_speed_m_s": 0.05, "maximum_planar_drift_m": 0.05},
            "official_stand_state": "must be live-readback verified before whole-body IK commands",
            "grasp_input": "camera optical pose and object pose must be timestamp-aligned to docking state",
        },
        "observed": {
            "table_approach_physical_stop": stop,
            "table_docking_dynamic_status": docking.get("dynamic_docking_status"),
            "motion_result_status": motion.get("status"),
        },
        "gates": {
            "TABLE_APPROACH_PHYSICAL_STOP": "PASS" if stop.get("status") == "PASS" else "FAIL",
            "TABLE_DOCKING_DYNAMIC_COMPLETION": "NOT_RUN",
            "OFFICIAL_STAND_READBACK": "NOT_RUN",
            "WHOLE_BODY_IK": "NOT_RUN",
            "GRASP_HANDOFF": "BLOCKED_BY_DOCKING_AND_STAND",
        },
        "limits": [
            "This contract never publishes locomotion or upper-body commands.",
            "The 0.1 m final docking motion remains owned by the navigation workstream.",
            "A grasp module may consume this contract only after TABLE_DOCKING physical-stop evidence is present.",
        ],
    }
    pg.write_new_text(output / "handoff.json", json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": handoff["gates"], "docking_pose": handoff["docking_pose"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
