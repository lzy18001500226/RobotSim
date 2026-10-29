#!/usr/bin/env python3
"""Static STL grasp-reach audit at Local03's verified TABLE_APPROACH endpoint.

This tool consumes navigation evidence but never runs the walker or physics.
It only moves the fixed review-scene pelvis in a generated XML overlay, solves
right arm/waist IK within source limits, and calls ``mj_forward``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
WAIST_JOINTS = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
DISTAL_BODIES = ("R_hand_narrow3_Link", "R_hand_wide3_Link")


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


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"scene lacks body: {name}")


def handoff_pose(handoff: dict[str, object], route: dict[str, object]) -> np.ndarray:
    """Accept only Local03's explicitly aligned raw-odom coordinate contract."""
    contract_record = route.get("runtime_coordinate_contract", {})
    contract = str(contract_record.get("numeric_coordinate_convention", "")) if isinstance(contract_record, dict) else str(contract_record)
    if contract != "DERIVED_ALIGNED_SCENE_WORLD_RAW_ODOM_AFTER_LIVE_RESET_GATE":
        raise RuntimeError("route does not establish direct odom-to-scene coordinates")
    final = handoff.get("final odom")
    if not isinstance(final, list) or len(final) != 4:
        raise RuntimeError("handoff does not contain the four-value final odom pose")
    pose = np.asarray(final, dtype=float)
    if not np.isfinite(pose).all():
        raise RuntimeError("handoff final pose is non-finite")
    return pose


def set_pelvis_pose(root: ET.Element, pose: np.ndarray) -> dict[str, object]:
    pelvis = find_body(root, "pelvis")
    before = {name: pelvis.get(name) for name in ("pos", "euler", "quat")}
    pelvis.set("pos", " ".join(f"{value:.12g}" for value in pose[:3]))
    pelvis.set("euler", f"0 0 {pose[3]:.12g}")
    pelvis.attrib.pop("quat", None)
    return {"body": "pelvis", "before": before, "after": {"pos": pelvis.get("pos"), "euler": pelvis.get("euler")}}


def joint_limits(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    limits = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not bool(model.jnt_limited[joint]):
            raise RuntimeError(f"finite source joint range is required: {name}")
        limits.append(model.jnt_range[joint])
    return np.asarray(limits, dtype=float)


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"scene lacks body: {name}")
    return value


def body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body#{body}"


def robot_table_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    table = body_id(model, "raicom_table")
    cylinder = body_id(model, "test_cylinder")
    records = []
    for contact in data.contact:
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if table not in (first, second):
            continue
        other = second if first == table else first
        if other == cylinder:
            continue
        records.append({"body_names": (body_name(model, first), body_name(model, second)), "distance_m": float(contact.dist)})
    return records


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body: int, dofs: np.ndarray) -> np.ndarray:
    position = np.zeros((3, model.nv))
    rotation = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, position, rotation, body)
    return position[:, dofs]


def solve_level_pose(static, force, orient, model, state, names, limits, seed, command, target, elbow, wrist, narrow, wide) -> tuple[np.ndarray, dict[str, float]]:
    dofs = np.asarray([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names], dtype=int)
    q = seed.copy()
    for iteration in range(1, 251):
        data = mujoco.MjData(model)
        static.apply_prior_state(model, data, state, q[3:], {name: float(value) for name, value in zip(names[:3], q[:3])})
        force.apply_positions(model, data, force.fit_positions(float(command)))
        mujoco.mj_forward(model, data)
        centre_error = target - orient.jaw_center(model, data)
        forearm_delta = float(data.xpos[wrist, 2] - data.xpos[elbow, 2])
        distal_delta = float(data.xpos[narrow, 2] - data.xpos[wide, 2])
        if np.linalg.norm(centre_error) <= 1e-4 and abs(forearm_delta) <= 1e-4 and abs(distal_delta) <= 1e-4:
            break
        centre_jac = orient.jaw_jacobian(model, data)[:, dofs]
        elbow_jac = point_jacobian(model, data, elbow, dofs)
        wrist_jac = point_jacobian(model, data, wrist, dofs)
        narrow_jac = point_jacobian(model, data, narrow, dofs)
        wide_jac = point_jacobian(model, data, wide, dofs)
        error = np.concatenate((centre_error, (-forearm_delta,), (-distal_delta,)))
        jacobian = np.vstack((centre_jac, wrist_jac[2] - elbow_jac[2], narrow_jac[2] - wide_jac[2]))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-7 * np.eye(5), error)
        delta -= 1e-4 * (q - seed)
        q += np.clip(delta, -0.025, 0.025)
        q = np.clip(q, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)
    data = mujoco.MjData(model)
    static.apply_prior_state(model, data, state, q[3:], {name: float(value) for name, value in zip(names[:3], q[:3])})
    force.apply_positions(model, data, force.fit_positions(float(command)))
    mujoco.mj_forward(model, data)
    return q, {
        "iterations": iteration,
        "jaw_center_error_m": float(np.linalg.norm(orient.jaw_center(model, data) - target)),
        "forearm_height_delta_m": float(data.xpos[wrist, 2] - data.xpos[elbow, 2]),
        "distal_reference_height_delta_m": float(data.xpos[narrow, 2] - data.xpos[wide, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--route-config", required=True, type=Path)
    parser.add_argument("--height-offset-mm", nargs="+", type=float, default=(80.0, 100.0, 120.0))
    parser.add_argument("--commands", nargs="+", type=int, default=(30, 32, 34, 36, 38, 40))
    args = parser.parse_args()
    if not all(path.is_file() for path in (args.freeze_record, args.handoff, args.route_config)):
        raise SystemExit("freeze record, navigation handoff, or route config is missing")
    if any(not math.isfinite(value) for value in args.height_offset_mm) or any(command < 0 or command > 100 for command in args.commands):
        raise SystemExit("scan parameters are invalid")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    handoff = json.loads(args.handoff.read_text(encoding="utf-8")) if args.handoff.suffix == ".json" else {"final odom": None}
    if handoff["final odom"] is None:
        # The published handoff is Markdown; its referenced dynamic result is
        # authoritative for the machine-readable final pose.
        dynamic_path = args.handoff.parent.parent / "raicom-lidar-map-table-repeat-tight-20261029T072316Z/m1_first_motion_result.json"
        dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
        final = dynamic.get("final_pose_odom")
        if isinstance(final, dict):
            handoff["final odom"] = [final[key] for key in ("x_m", "y_m", "z_m", "yaw_rad")]
        elif isinstance(final, list):
            handoff["final odom"] = final
        else:
            raise SystemExit("navigation dynamic result lacks final odom pose")
    route = json.loads(args.route_config.read_text(encoding="utf-8"))
    base_pose = handoff_pose(handoff, route)
    review_path = Path(frozen["accepted_review"]["path"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    source_scene = Path(review["scene"]["path"])
    if sha256(source_scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted source scene hash has changed")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"]:
        raise SystemExit("frozen STL model does not match review record")
    capture = load_module("rgbd_capture", REPO / "scripts/research/issue46_rgbd_truth_capture.py")
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    orient = load_module("stl_orientation", REPO / "scripts/research/issue46_stl_centered_orientation_search.py")
    level = load_module("stl_level", REPO / "scripts/research/issue46_stl_level_side_grasp_search.py")
    tree = ET.parse(source_scene)
    root = tree.getroot()
    rebindings = capture.rebind_missing_workspace_assets(root)
    base_overlay = set_pelvis_pose(root, base_pose)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-table-approach-static-ik").native_path
    overlay = output / "table_approach_stl_static_overlay.xml"
    ET.indent(tree, space="  ")
    pg.write_new_bytes(overlay, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(overlay))
    names = WAIST_JOINTS + static.ARM_JOINTS
    limits = joint_limits(model, names)
    initial_command = int(review["initial_command_percent"])
    seed = np.asarray([review["motor_override_rad"][name] for name in WAIST_JOINTS] + [review["samples"][str(initial_command)]["arm_q_rad"][name] for name in static.ARM_JOINTS], dtype=float)
    state = review["state"]
    reference = mujoco.MjData(model)
    static.apply_prior_state(model, reference, state, seed[3:], {name: float(value) for name, value in zip(names[:3], seed[:3])})
    force.apply_positions(model, reference, force.fit_positions(float(initial_command)))
    mujoco.mj_forward(model, reference)
    cylinder = body_id(model, "test_cylinder")
    elbow, wrist, narrow, wide = (body_id(model, name) for name in ("right_elbow_link", "right_wrist_roll_link", *DISTAL_BODIES))
    cylinder_center = reference.xipos[cylinder].copy()
    rows = []
    for height_mm in args.height_offset_mm:
        target = cylinder_center + np.array([0.0, 0.0, height_mm / 1000.0])
        q, metrics = solve_level_pose(static, force, orient, model, state, names, limits, seed, initial_command, target, elbow, wrist, narrow, wide)
        for command in args.commands:
            data = mujoco.MjData(model)
            static.apply_prior_state(model, data, state, q[3:], {name: float(value) for name, value in zip(names[:3], q[:3])})
            force.apply_positions(model, data, force.fit_positions(float(command)))
            mujoco.mj_forward(model, data)
            contacts = level.actual_contacts(model, data, cylinder)
            rows.append({"height_offset_mm": height_mm, "command_percent": command, "joint_q_rad": {name: float(value) for name, value in zip(names, q)}, "metrics": metrics, "contacts": contacts, "robot_table_contacts": robot_table_contacts(model, data)})
    valid = [row for row in rows if row["metrics"]["jaw_center_error_m"] <= 1e-4 and abs(row["metrics"]["forearm_height_delta_m"]) <= 1e-4 and abs(row["metrics"]["distal_reference_height_delta_m"]) <= 1e-4 and all(row["contacts"]["counts"][name] > 0 for name in DISTAL_BODIES) and not row["contacts"]["nonpad_bodies"] and not row["robot_table_contacts"]]
    result = {
        "schema": "robotsim.issue46.stl.table-approach-static-ik.v1",
        "classification": "SIMULATION_ONLY_TABLE_APPROACH_STATIC_IK",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256},
        "navigation": {"handoff_path": str(args.handoff), "handoff_sha256": sha256(args.handoff), "route_path": str(args.route_config), "route_sha256": sha256(args.route_config), "base_pose_scene_world": base_pose.tolist(), "contract": route["runtime_coordinate_contract"]},
        "source_scene": {"path": str(source_scene), "sha256": sha256(source_scene), "asset_rebindings": rebindings},
        "review_state": {"path": str(review_path), "sha256": sha256(review_path), "initial_command_percent": initial_command},
        "overlay": {"path": str(overlay), "sha256": sha256(overlay), "pelvis_pose_change": base_overlay},
        "rows": rows,
        "valid_dual_pad_candidates": valid,
        "gates": {"TABLE_APPROACH_NAVIGATION_EVIDENCE": "PASS", "TABLE_DOCKING": "NOT_COMPLETE", "ROBOT_TABLE_CLEARANCE": "PASS" if valid else "BLOCKED", "STATIC_IK": "PASS" if valid else "BLOCKED", "BALANCE_AT_APPROACH_ENDPOINT": "NOT_RUN", "DYNAMICS": "NOT_RUN"},
        "limitations": ["The navigation evidence ends at TABLE_APPROACH, not TABLE_DOCKING.", "The pelvis change is a static scene overlay only; it does not prove walking-to-standing or balance after navigation.", "No model geometry, gripper parameter, collision setting, or MuJoCo dynamic state was changed."],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "candidate_count": len(valid), "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
