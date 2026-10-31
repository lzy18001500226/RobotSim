#!/usr/bin/env python3
"""Validate an external IK reference against the actual native standing table scene.

The external IK implementation is an offline reference only. This tool writes
no qpos to a running simulator and calls no ``mj_step``. It loads the recorded
native standing state, solves one right-arm pregrasp target, writes that state
to an evidence-local static scene, and reports exact MuJoCo contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hashed(path: Path, expected: str, *, label: str) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute regular file")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return path.resolve()


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise ValueError(f"native table scene lacks joint {name}")
    data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[model.jnt_dofadr[joint]] = 0.0


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    result = []
    for contact in data.contact:
        bodies = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or f"body#{int(model.geom_bodyid[geom])}"
            for geom in (contact.geom1, contact.geom2)
        )
        result.append({"body_names": bodies, "distance_m": float(contact.dist)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-result", required=True, type=Path)
    parser.add_argument("--placement-result-sha256", required=True)
    parser.add_argument("--standing-state-result", required=True, type=Path)
    parser.add_argument("--standing-state-result-sha256", required=True)
    parser.add_argument("--ik-sdk-root", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--target-height-above-cylinder-m", type=float, default=0.12)
    parser.add_argument("--target-offset-xyz-m", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()
    placement_path = require_hashed(args.placement_result, args.placement_result_sha256, label="placement result")
    standing_path = require_hashed(args.standing_state_result, args.standing_state_result_sha256, label="standing state result")
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    standing = json.loads(standing_path.read_text(encoding="utf-8"))
    candidates = placement.get("scan", {}).get("best_candidates", [])
    if not isinstance(candidates, list) or not 0 <= args.candidate_index < len(candidates):
        raise ValueError("candidate index is outside collision-free placement result")
    candidate = candidates[args.candidate_index]
    scan = placement.get("generated_scan_scene", {})
    scan_scene = require_hashed(Path(str(scan.get("path", ""))), str(scan.get("sha256", "")), label="placement scan scene")
    capture_record = standing.get("native_arm_state_capture", {})
    captured = capture_record.get("result", {}) if isinstance(capture_record, dict) and isinstance(capture_record.get("result"), dict) else capture_record
    arm = captured.get("arm_positions_rad", {})
    leg = captured.get("leg_positions_rad", {})
    waist = captured.get("waist_positions_rad", {})
    if not all(isinstance(group, dict) for group in (arm, leg, waist)):
        raise ValueError("standing-state result lacks complete joint groups")
    if not args.ik_sdk_root.is_absolute() or not (args.ik_sdk_root / "src/x2_ik_sdk/src").is_dir():
        raise ValueError("IK SDK root is missing its explicit source package")
    sys.path.insert(0, str(args.ik_sdk_root / "src/x2_ik_sdk/src"))
    sdk = importlib.import_module("x2_ik_sdk")
    config = importlib.import_module("x2_ik_sdk.config")
    solver = sdk.X2ArmIKSolver(sdk.X2IKConfig.default_omnipicker())
    arm_order = list(config.ARM_POS_ORDER)
    if any(name not in arm for name in arm_order):
        raise ValueError("standing state does not contain the SDK arm order")
    model = mujoco.MjModel.from_xml_path(str(scan_scene))
    data = mujoco.MjData(model)
    mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "raicom_table_room_frame")
    cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cylinder_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if min(mocap_body, cylinder_joint, pelvis_body, cylinder_body) < 0:
        raise ValueError("scan scene lacks mocap/table/cylinder/pelvis identifiers")
    mocap = int(model.body_mocapid[mocap_body])
    data.mocap_pos[mocap] = np.asarray(candidate["room_frame_xyz_m"], dtype=float)
    qadr = int(model.jnt_qposadr[cylinder_joint])
    data.qpos[qadr:qadr + 3] = np.asarray(candidate["cylinder_world_xyz_m"], dtype=float)
    data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    for group in (leg, waist, arm):
        for name, value in group.items():
            set_joint(model, data, str(name), float(value))
    mujoco.mj_forward(model, data)
    pelvis = data.xpos[pelvis_body].copy()
    cylinder = data.xpos[cylinder_body].copy()
    target_offset = np.asarray(args.target_offset_xyz_m, dtype=float)
    if target_offset.shape != (3,) or not np.isfinite(target_offset).all():
        raise ValueError("target offset must be a finite three-vector")
    target = cylinder - pelvis + target_offset + np.array((0.0, 0.0, args.target_height_above_cylinder_m))
    seed = [float(arm[name]) for name in arm_order]
    ik = solver.solve_position(sdk.ArmSide.RIGHT, target.tolist(), seed)
    for name, value in zip(arm_order[7:], ik.active_arm):
        set_joint(model, data, name, float(value))
    mujoco.mj_forward(model, data)
    contacts = contact_records(model, data)
    table_contacts = [row for row in contacts if "raicom_table_table" in row["body_names"] and any(not name.startswith("raicom_table_") and name != "test_cylinder" for name in row["body_names"])]
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-standing-table-ik-validation").native_path
    result = {
        "schema": "robotsim.issue46.raicom-standing-table-ik-validation.v1",
        "classification": "SIMULATION_ONLY_STATIC_EXTERNAL_IK_REFERENCE",
        "mj_step_calls": 0,
        "sources": {
            "placement_result": {"path": str(placement_path), "sha256": sha256(placement_path)},
            "standing_state_result": {"path": str(standing_path), "sha256": sha256(standing_path)},
            "scan_scene": {"path": str(scan_scene), "sha256": sha256(scan_scene)},
            "ik_sdk_root": str(args.ik_sdk_root),
        },
        "candidate": candidate,
        "target_pelvis_xyz_m": target.tolist(), "target_offset_xyz_m": target_offset.tolist(),
        "ik": {"success": bool(ik.success), "error_m": float(ik.error_norm), "iterations": int(ik.iterations), "right_arm_q_rad": ik.active_arm, "final_pelvis_xyz_m": ik.final_xyz},
        "contacts": contacts,
        "table_robot_contacts": table_contacts,
        "gates": {"STANDING_STATE_INPUT": "PASS", "IK_CONVERGED": "PASS" if ik.success else "FAIL", "POST_IK_TABLE_CLEARANCE": "PASS" if not table_contacts else "FAIL", "NATIVE_STANDING_ARM_MOTION": "NOT_RUN", "GRASP": "NOT_RUN"},
        "limitations": ["The IK SDK is an external research reference and this is a static verification only.", "No arm command or physics step ran.", "Orientation, finger contact and dynamic stability require later validation."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "ik": result["ik"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
