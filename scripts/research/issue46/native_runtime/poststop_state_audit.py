#!/usr/bin/env python3
"""Map one observed RAICOM TABLE_APPROACH post-stop state into the STL overlay.

This is a static state-transfer audit.  It maps named source joints and
actuators, applies reviewed kinematic hand commands, and calls ``mj_forward``.
It never steps physics or claims that the altered hand overlay is dynamically
balanced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def dof_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def apply_named_state(model: mujoco.MjModel, data: mujoco.MjData, sample: dict[str, object], metadata: dict[str, object]) -> dict[str, object]:
    source_qpos = np.asarray(sample["qpos"], dtype=float)
    source_qvel = np.asarray(sample["qvel"], dtype=float)
    source_ctrl = np.asarray(sample["ctrl"], dtype=float)
    joints = metadata["joint_names"]
    actuators = metadata["actuator_names"]
    mapped_joints, skipped_joints = [], []
    for item in joints:
        name = item.get("name")
        if not name:
            skipped_joints.append({"source_id": item["id"], "reason": "unnamed source joint"})
            continue
        destination = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if destination < 0:
            skipped_joints.append({"name": name, "reason": "missing in overlay"})
            continue
        source_qadr = int(item["qpos_adr"])
        source_dadr = int(item["dof_adr"])
        width_q = qpos_width(int(item["type"]))
        width_d = dof_width(int(item["type"]))
        destination_qadr = int(model.jnt_qposadr[destination])
        destination_dadr = int(model.jnt_dofadr[destination])
        data.qpos[destination_qadr : destination_qadr + width_q] = source_qpos[source_qadr : source_qadr + width_q]
        data.qvel[destination_dadr : destination_dadr + width_d] = source_qvel[source_dadr : source_dadr + width_d]
        mapped_joints.append(name)
    mapped_actuators, skipped_actuators = [], []
    for source_id, item in enumerate(actuators):
        name = item.get("name")
        destination = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if destination < 0:
            skipped_actuators.append(name)
            continue
        data.ctrl[destination] = source_ctrl[source_id]
        mapped_actuators.append(name)
    return {"mapped_joint_names": mapped_joints, "skipped_joints": skipped_joints, "mapped_actuator_names": mapped_actuators, "skipped_actuators": skipped_actuators}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-audit", required=True, type=Path)
    parser.add_argument("--balance-handoff", required=True, type=Path)
    parser.add_argument("--left-command-percent", type=float, default=0.0)
    parser.add_argument("--right-command-percent", type=float, default=0.0)
    args = parser.parse_args()
    if not args.mount_audit.is_file() or not args.balance_handoff.is_file() or not all(0.0 <= item <= 100.0 for item in (args.left_command_percent, args.right_command_percent)):
        raise SystemExit("audit/handoff or hand command is invalid")
    audit = json.loads(args.mount_audit.read_text(encoding="utf-8"))
    handoff = json.loads(args.balance_handoff.read_text(encoding="utf-8"))
    if audit.get("gates", {}).get("BILATERAL_STL_STATIC_MJ_FORWARD") != "PASS" or handoff.get("status") != "PASS_BALANCE_AT_APPROACH_ENDPOINT":
        raise SystemExit("mount audit or balance handoff did not pass")
    overlay = Path(audit["generated_overlay"]["path"])
    if not overlay.is_file() or sha256(overlay) != audit["generated_overlay"]["sha256"]:
        raise SystemExit("mount overlay is missing or changed")
    summary_path = Path(handoff["run_directory"]) / "native_mjdata_capture_summary.json"
    if not summary_path.is_file():
        raise SystemExit("source native MjData summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = handoff["native_post_stop_sample"]
    metadata = summary["model_metadata"]
    model = mujoco.MjModel.from_xml_path(str(overlay))
    data = mujoco.MjData(model)
    mapping = apply_named_state(model, data, sample, metadata)
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    left = apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.left_command_percent, side="left", forward=True)
    right = apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.right_command_percent, side="right", forward=True)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    missing_named_joints = [item for item in mapping["skipped_joints"] if item.get("reason") != "unnamed source joint"]
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-poststop-state-audit").native_path
    record = {
        "schema": "robotsim.issue46.raicom-stl-poststop-state-audit.v1",
        "classification": "SIMULATION_ONLY_STATIC_POSTSTOP_STATE_TRANSFER",
        "mj_step_calls": 0,
        "balance_handoff": {"path": str(args.balance_handoff), "sha256": sha256(args.balance_handoff), "status": handoff["status"]},
        "source_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        "mount_overlay": {"path": str(overlay), "sha256": sha256(overlay)},
        "mapping": mapping,
        "commands": {"left_percent": args.left_command_percent, "right_percent": args.right_command_percent, "left_joint_count": len(left.positions_rad), "right_joint_count": len(right.positions_rad)},
        "post_forward": {"pelvis_world_position_m": data.xpos[pelvis].tolist(), "pelvis_world_quaternion_wxyz": data.xquat[pelvis].tolist(), "finite_qpos": bool(np.isfinite(data.qpos).all()), "finite_qvel": bool(np.isfinite(data.qvel).all())},
        "gates": {"SOURCE_POSTSTOP_STATE": "PASS", "NAMED_ROBOT_STATE_MAPPING": "PASS" if not missing_named_joints else "BLOCKED", "UNNAMED_SOURCE_OBJECT_STATE": "SKIPPED_NOT_PRESENT_IN_ROBOT_ONLY_OVERLAY" if mapping["skipped_joints"] else "NOT_PRESENT", "BILATERAL_STL_POSE": "PASS", "STATIC_MJ_FORWARD": "PASS", "DYNAMICS_WITH_STL_HANDS": "NOT_RUN", "GRASP": "NOT_RUN"},
        "limitations": ["The source post-stop balance state belongs to the no-STL RAICOM runtime; this audit only verifies its static coordinate transfer into a separate hand overlay.", "No physics step, balance replay, controller change, or contact experiment occurred."],
    }
    pg.write_new_text(output / "result.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": record["gates"], "mapped_joints": len(mapping["mapped_joint_names"]), "skipped_joints": mapping["skipped_joints"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
