#!/usr/bin/env python3
"""Find a source-STL vertical-cylinder pose with level bilateral pad contact."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402

MODEL_ID = "X2-OmniPicker-STL"
ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


def load_pre():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("pre", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load static preflight")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation_y(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    return axis * (0.5 * angle / max(math.sin(angle), 1e-12))


def solve_pose(pre, model, trajectory, state, command: int, start: np.ndarray, target_pos: np.ndarray, target_mat: np.ndarray):
    q = start.copy()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in ARM]
    bounds = [tuple(float(value) for value in model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in ARM]
    for iteration in range(1, 601):
        data = mujoco.MjData(model)
        pre.apply_prior_state(model, data, state, q)
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command))
        position = target_pos - data.xpos[body]
        orientation = rotation_error(target_mat, data.xmat[body].reshape(3, 3))
        if np.linalg.norm(position) < 1e-4 and np.linalg.norm(orientation) < 1e-3:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body)
        J = np.vstack((jacp[:, dofs], 0.08 * jacr[:, dofs]))
        error = np.concatenate((position, 0.08 * orientation))
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), error)
        q += np.clip(dq, -0.05, 0.05)
        for index, (lo, hi) in enumerate(bounds):
            q[index] = float(np.clip(q[index], lo + 1e-5, hi - 1e-5))
    result = pre.evaluate_pose(model, trajectory, state, command, arm_override=q)
    result["arm_q_rad"] = {name: float(value) for name, value in zip(ARM, q)}
    result["iterations"] = iteration
    result["position_error_m"] = float(np.linalg.norm(target_pos - data.xpos[body]))
    result["orientation_error_rad"] = float(np.linalg.norm(rotation_error(target_mat, data.xmat[body].reshape(3, 3))))
    return result


def pad_height_delta(arm: np.ndarray, command_percent: int, model, trajectory, pre, state) -> float:
    data = mujoco.MjData(model)
    pre.apply_prior_state(model, data, state, arm)
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command_percent))
    points = {}
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    for contact in data.contact:
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, value) or "" for value in bodies}
        for pad in names & PADS:
            if cylinder in bodies:
                points[pad] = np.asarray(contact.pos, dtype=float)
    if set(points) != PADS:
        return float("inf")
    return abs(float(points["R_hand_narrow3_Link"][2] - points["R_hand_wide3_Link"][2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertical-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    pre = load_pre()
    baseline = json.loads(args.vertical_static_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    scene = Path(baseline["generated_scene"]["path"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    start = np.asarray([baseline["aligned_contact_static"]["arm_q_rad"][name] for name in ARM])
    command = int(baseline["aligned_contact_static"]["highest_safe_bilateral_percent"])
    reference = mujoco.MjData(model)
    pre.apply_prior_state(model, reference, state, start)
    apply_stl_kinematic_pose(mujoco, model, reference, trajectory, float(command))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    target_pos = reference.xpos[body].copy()
    target_mat = reference.xmat[body].reshape(3, 3).copy()
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-vertical-pose-static") if args.output_dir is None else pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
    output = location.native_path
    if any(item.name != ".robotsim-output-reservation.json" for item in output.iterdir()):
        raise SystemExit("output is not fresh")
    candidates = []
    for degrees in (-40, -35, -30, -25, -20, 20, 25, 30, 35, 40):
        solved = solve_pose(pre, model, trajectory, state, command, start, target_pos, rotation_y(math.radians(degrees)) @ target_mat)
        arm = np.asarray([solved["arm_q_rad"][name] for name in ARM])
        scans = [pre.evaluate_pose(model, trajectory, state, percent, arm_override=arm) for percent in range(100, -1, -1)]
        eligible = [item for item in scans if item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"]]
        selected = max(eligible, key=lambda item: item["command_percent"]) if eligible else None
        if selected is not None:
            selected["height_delta_m"] = pad_height_delta(arm, int(selected["command_percent"]), model, trajectory, pre, state)
            selected["arm_q_rad"] = solved["arm_q_rad"]
        candidates.append({"world_y_rotation_deg": degrees, "pose": solved, "scan": scans, "selected": selected})
    viable = [item for item in candidates if item["selected"] is not None]
    best = min(viable, key=lambda item: item["selected"]["height_delta_m"]) if viable else None
    result = {
        "schema": "robotsim.issue46.full-x2-stl-vertical-cylinder-pose-search.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256},
        "baseline_static": str(args.vertical_static_result),
        "target_base_world_m": target_pos.tolist(),
        "candidates": candidates,
        "best": best,
        "gates": {"LEVEL_BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS" if best else "BLOCKED", "DYNAMICS": "NOT_RUN"},
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "best": best, "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
