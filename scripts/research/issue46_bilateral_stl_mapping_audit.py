#!/usr/bin/env python3
"""Produce the source audit required before a left STL table adaptation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
OFFICIAL_URDF = EVIDENCE_REPO / "reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
RIGHT_TRAJECTORY = EVIDENCE_REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic/trajectory.csv"
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


RIGHT_TO_LEFT = {
    "right_claw_joint": "left_claw_joint",
    "R_hand_wide1_joint": "L_hand_wide1_joint",
    "R_hand_narrow3_joint": "L_hand_narrow3_joint",
    "R_hand_narrow_loop_joint": "L_hand_narrow_loop_joint",
    "R_hand_wide3_joint": "L_hand_wide3_joint",
    "R_hand_wide_loop_joint": "L_hand_wide_loop_joint",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_record(joint: ET.Element) -> dict[str, object]:
    def attrs(tag: str) -> dict[str, str]:
        node = joint.find(tag)
        return {} if node is None else dict(node.attrib)

    return {
        "type": joint.get("type"),
        "parent": attrs("parent"),
        "child": attrs("child"),
        "origin": attrs("origin"),
        "axis": attrs("axis"),
        "limit": attrs("limit"),
        "mimic": attrs("mimic"),
    }


def normalized(record: dict[str, object]) -> dict[str, object]:
    value = json.loads(json.dumps(record))
    for key in ("parent", "child"):
        link = value[key].get("link")
        if isinstance(link, str):
            value[key]["link"] = link.replace("R_", "SIDE_").replace("L_", "SIDE_")
    mimic = value["mimic"]
    joint = mimic.get("joint")
    if isinstance(joint, str):
        mimic["joint"] = joint.replace("right_", "side_").replace("left_", "side_").replace("R_", "SIDE_").replace("L_", "SIDE_")
    return value


def main() -> None:
    if not OFFICIAL_URDF.is_file() or not RIGHT_TRAJECTORY.is_file():
        raise SystemExit("official URDF or frozen right STL trajectory is missing")
    root = ET.parse(OFFICIAL_URDF).getroot()
    joints = {node.get("name", ""): node for node in root.findall("joint")}
    comparisons = []
    for right, left in RIGHT_TO_LEFT.items():
        if right not in joints or left not in joints:
            raise RuntimeError(f"official URDF lacks {right} or {left}")
        right_record = joint_record(joints[right])
        left_record = joint_record(joints[left])
        comparisons.append(
            {
                "right_joint": right,
                "left_joint": left,
                "right": right_record,
                "left": left_record,
                "normalized_frames_equal": normalized(right_record) == normalized(left_record),
            }
        )
    all_equal = all(item["normalized_frames_equal"] for item in comparisons)
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="bilateral-stl-mapping-audit")
    output = location.native_path
    report = {
        "schema": "robotsim.issue46.bilateral-stl-mapping-audit.v1",
        "classification": "SOURCE_AUDIT_AND_MECHANICAL_ADAPTATION_PROPOSAL",
        "mj_step_calls": 0,
        "official_source": {
            "urdf": str(OFFICIAL_URDF),
            "sha256": sha256(OFFICIAL_URDF),
            "revision": "575cc6b988f976c23550e0db85aa1e5475d3652d",
        },
        "right_stl_trajectory": {"path": str(RIGHT_TRAJECTORY), "sha256": sha256(RIGHT_TRAJECTORY)},
        "joint_frame_comparisons": comparisons,
        "proposed_delta": {
            "kind": "source-supported left table-driven kinematic adaptation",
            "mapping": RIGHT_TO_LEFT,
            "numeric_rule": "copy each right-side reviewed trajectory value to its official left-side joint with no sign inversion",
            "unchanged": ["official left mesh files", "official joint local origins", "official axes", "official limits", "official wide-root mimic multiplier", "collision geometry", "collision exclusions"],
        },
        "evidence": {
            "all_normalized_joint_frames_equal": all_equal,
            "source_has_no_separate_left_passive_trajectory_table": True,
            "inference_boundary": "The URDF proves matched local coordinate frames and mimic sign. It does not independently prove unobserved left passive transmission behavior beyond the source-symmetric table adaptation.",
        },
        "validation_plan": [
            "Compile a separate bilateral STL review scene without modifying source assets.",
            "For commands 0..100, apply the same table row to left and right mapped joints and verify each joint remains in official limits.",
            "Human-review visual left/right symmetry and slider behavior in MuJoCo.",
            "Do not claim bilateral physical grasp or manufacturer transmission equivalence from this kinematic table mapping.",
        ],
        "gates": {
            "OFFICIAL_JOINT_FRAME_SYMMETRY": "PASS" if all_equal else "FAIL",
            "LEFT_PASSIVE_TRAJECTORY_DIRECT_SOURCE_TABLE": "NOT_AVAILABLE",
            "MAINTAINER_CASE_SPECIFIC_APPROVAL": "REQUIRED",
            "HUMAN_VISUAL_REVIEW": "REQUIRED",
        },
    }
    pg.write_new_text(output / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "all_normalized_joint_frames_equal": all_equal, "gates": report["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
