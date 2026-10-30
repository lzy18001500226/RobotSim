#!/usr/bin/env python3
"""Find an IK lift path that retains pad contact while the bottle is table-fixed.

Unlike a pure vertical end-effector lift, every waypoint is checked against the
stationary table-supported cylinder.  A small horizontal approach bias is
scanned to retain bilateral distal-pad contact without loop contact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


MODEL_ID = "X2-OmniPicker-STL"


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


def robot_table_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "raicom_table")
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if table not in (first, second):
            continue
        other = second if first == table else first
        if other != cylinder:
            result.append({"distance_m": float(contact.dist), "body_ids": [first, second]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--azimuth-scan", required=True, type=Path)
    parser.add_argument("--height-offset-mm", type=float, default=100.0)
    parser.add_argument("--azimuth-deg", type=float, default=20.0)
    parser.add_argument("--tool-pitch-deg", type=float, default=8.0)
    parser.add_argument("--command-percent", type=int, default=32)
    parser.add_argument("--lift-waypoints-mm", nargs="+", type=float, default=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0))
    parser.add_argument("--approach-bias-mm", nargs="+", type=float, default=(-10.0, -5.0, 0.0, 5.0, 10.0))
    parser.add_argument("--object-follow-ratio", type=float, default=0.0)
    args = parser.parse_args()
    if not args.freeze_record.is_file() or not args.azimuth_scan.is_file():
        raise SystemExit("freeze record or azimuth scan is missing")
    if not 0.0 <= args.object_follow_ratio <= 1.0:
        raise SystemExit("object-follow ratio must lie in [0, 1]")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    scan = json.loads(args.azimuth_scan.read_text(encoding="utf-8"))
    if frozen.get("baseline_id") != "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1" or scan.get("classification") != "STATIC_IK_AZIMUTH_CONTACT_SCREENING":
        raise SystemExit("inputs do not identify the accepted horizontal baseline and azimuth scan")
    review_path = Path(frozen["accepted_review"]["path"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    scene = Path(review["scene"]["path"])
    if sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted source scene changed")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"] or loaded.model_sha256 != scan["model"]["sha256"]:
        raise SystemExit("frozen model differs from evidence")
    candidates = [
        row for row in scan["valid_dual_distal_pad_candidates"]
        if row["height_offset_mm"] == args.height_offset_mm
        and row["azimuth_deg"] == args.azimuth_deg
        and row["tool_pitch_deg"] == args.tool_pitch_deg
        and row["command_percent"] == args.command_percent
    ]
    if len(candidates) != 1:
        raise SystemExit("selected contact candidate is not unique")
    candidate = candidates[0]
    ik = load_module("horizontal_ik", REPO / "scripts/research/issue46/grasp/horizontal_side_grasp_ik_scan.py")
    azimuth = load_module("azimuth", REPO / "scripts/research/issue46/grasp/horizontal_side_grasp_azimuth_scan.py")
    static = load_module("stl_static", REPO / "scripts/research/issue46/grasp/cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46/grasp/force_control_static_preflight.py")
    orient = load_module("stl_orientation", REPO / "scripts/research/issue46/grasp/centered_orientation_search.py")
    level = load_module("stl_level", REPO / "scripts/research/issue46/grasp/level_side_grasp_search.py")
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = review["state"]
    names = ik.WAIST_JOINTS + static.ARM_JOINTS
    limits = ik.joint_limits(model, names)
    q0 = np.asarray([candidate["waist_q_rad"][name] for name in ik.WAIST_JOINTS] + [candidate["arm_q_rad"][name] for name in static.ARM_JOINTS], dtype=float)
    ids = {key: ik.body_id(model, body) for key, body in {"elbow": "right_elbow_link", "wrist": "right_wrist_roll_link", "narrow": "R_hand_narrow3_Link", "wide": "R_hand_wide3_Link", "base": "R_omnipicker_base_link", "cylinder": "test_cylinder"}.items()}
    cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    if cylinder_joint < 0:
        raise RuntimeError("scene lacks the free cylinder joint")
    cylinder_qadr = int(model.jnt_qposadr[cylinder_joint])
    reference = mujoco.MjData(model)
    ik.apply_pose(static, force, model, reference, state, names, q0, args.command_percent)
    mujoco.mj_forward(model, reference)
    base_center = orient.jaw_center(model, reference).copy()
    base_rotation = reference.xmat[ids["base"]].reshape(3, 3).copy()
    approach_axis = reference.xpos[ids["base"]] - reference.xipos[ids["cylinder"]]
    approach_axis[2] = 0.0
    approach_axis /= np.linalg.norm(approach_axis)
    branches = []
    for bias_mm in args.approach_bias_mm:
        q_seed = q0.copy()
        waypoints = []
        for lift_mm in args.lift_waypoints_mm:
            target = base_center + np.array((0.0, 0.0, lift_mm / 1000.0)) + approach_axis * (bias_mm / 1000.0)
            q_seed, metrics = azimuth.solve(
                ik=ik, static=static, force=force, orient=orient, model=model, state=state, names=names, limits=limits,
                seed=q_seed, command=args.command_percent, target_center=target, target_rotation=base_rotation, ids=ids,
            )
            data = mujoco.MjData(model)
            ik.apply_pose(static, force, model, data, state, names, q_seed, args.command_percent)
            # Planning-only predicted object motion.  The actual physics run
            # never writes this qpos after its initialization.
            data.qpos[cylinder_qadr + 2] += args.object_follow_ratio * lift_mm / 1000.0
            mujoco.mj_forward(model, data)
            contacts = level.actual_contacts(model, data, ids["cylinder"])
            table = robot_table_contacts(model, data)
            passed = (
                metrics["jaw_center_error_m"] <= 1e-4
                and abs(metrics["forearm_height_delta_m"]) <= 1e-4
                and abs(metrics["distal_reference_height_delta_m"]) <= 1e-4
                and metrics["base_orientation_error_rad"] <= 1e-3
                and all(contacts["counts"][name] > 0 for name in ik.DISTAL_BODIES)
                and not contacts["nonpad_bodies"]
                and not table
            )
            waypoints.append({"lift_mm": lift_mm, "q": {name: float(value) for name, value in zip(names, q_seed)}, "metrics": metrics, "contacts": contacts, "robot_table_contacts": table, "pass": passed})
        branches.append({"approach_bias_mm": bias_mm, "waypoints": waypoints, "pass": all(item["pass"] for item in waypoints)})
    valid = [branch for branch in branches if branch["pass"]]
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-contact-maintaining-lift-path")
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl.contact-maintaining-lift-path.v1",
        "classification": "STATIC_CONTACT_MAINTAINING_IK_PATH",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "freeze_record": {"path": str(args.freeze_record), "sha256": sha256(args.freeze_record)},
        "azimuth_scan": {"path": str(args.azimuth_scan), "sha256": sha256(args.azimuth_scan)},
        "selected_contact": {"height_offset_mm": args.height_offset_mm, "azimuth_deg": args.azimuth_deg, "tool_pitch_deg": args.tool_pitch_deg, "command_percent": args.command_percent},
        "approach_axis_world": approach_axis.tolist(),
        "object_follow_ratio": args.object_follow_ratio,
        "branches": branches,
        "valid_branches": valid,
        "gates": {"CONTACT_MAINTAINING_PATH": "PASS" if valid else "BLOCKED", "DYNAMICS": "NOT_RUN"},
        "limitations": ["The bottle stays fixed during this static screen; a passing path is only a necessary condition for physical load transfer."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "valid_branch_count": len(valid), "valid_bias_mm": [branch["approach_bias_mm"] for branch in valid], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
