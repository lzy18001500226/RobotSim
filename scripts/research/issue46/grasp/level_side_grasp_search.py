#!/usr/bin/env python3
"""Solve and verify a level side-pinch from actual source-CoACD contact anchors.

The target is a vertical cylinder with opposite distal contact regions at the
same world height.  This is a static MuJoCo ``mj_forward`` search only.
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
PADS = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
COMMAND = 35.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_pose(static, force_static, model, data, state, arm: np.ndarray) -> None:
    static.apply_prior_state(model, data, state, arm)
    force_static.apply_positions(model, data, force_static.fit_positions(COMMAND))
    mujoco.mj_forward(model, data)


def anchor_world(model: mujoco.MjModel, data: mujoco.MjData, anchors: dict[str, np.ndarray], name: str) -> np.ndarray:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ anchors[name]


def anchor_jacobian(model: mujoco.MjModel, data: mujoco.MjData, anchors: dict[str, np.ndarray], name: str) -> np.ndarray:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    point = anchor_world(model, data, anchors, name)
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, linear, angular, point, body)
    return linear


def actual_contacts(model: mujoco.MjModel, data: mujoco.MjData, cylinder: int) -> dict[str, object]:
    locations = {name: [] for name in PADS}
    nonpad = []
    for contact in data.contact:
        body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if cylinder not in (body1, body2):
            continue
        other = body2 if body1 == cylinder else body1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if name in locations:
            locations[name].append(np.asarray(contact.pos, dtype=float))
        elif name.startswith("R_hand_"):
            nonpad.append(name)
    means = {name: (np.mean(points, axis=0).tolist() if points else None) for name, points in locations.items()}
    level_delta = None if any(means[name] is None for name in PADS) else float(means[PADS[0]][2] - means[PADS[1]][2])
    return {"counts": {name: len(points) for name, points in locations.items()}, "mean_world_m": means, "height_delta_m": level_delta, "nonpad_bodies": sorted(set(nonpad))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--force-fit-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    args = parser.parse_args()
    fit = json.loads(args.force_fit_result.read_text(encoding="utf-8"))
    if sha256(args.source_scene) != fit["source_scene"]["sha256"]:
        raise SystemExit("source scene differs from centered fit evidence")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != fit["model"]["sha256"]:
        raise SystemExit("frozen model differs from centered fit evidence")
    static = load_module("static", REPO / "scripts/research/issue46/grasp/cylinder_static_preflight.py")
    force_static = load_module("force_static", REPO / "scripts/research/issue46/grasp/force_control_static_preflight.py")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    state = static.state_by_phase(config, "HORIZONTAL_APPROACH")
    start = np.asarray([fit["force_fit_centering"]["arm_q_rad"][name] for name in static.ARM_JOINTS], dtype=float)
    model = mujoco.MjModel.from_xml_path(str(args.source_scene))
    data = mujoco.MjData(model)
    apply_pose(static, force_static, model, data, state, start)
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    initial = actual_contacts(model, data, cylinder)
    if any(initial["mean_world_m"][name] is None for name in PADS):
        raise RuntimeError("initial centered pose lacks bilateral distal contacts")
    anchors = {}
    for name in PADS:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        anchors[name] = data.xmat[body].reshape(3, 3).T @ (np.asarray(initial["mean_world_m"][name]) - data.xpos[body])
    target_average = np.mean([np.asarray(initial["mean_world_m"][name]) for name in PADS], axis=0)
    dofs, bounds = [], []
    for name in static.ARM_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofs.append(int(model.jnt_dofadr[joint]))
        bounds.append(tuple(float(value) for value in model.jnt_range[joint]))
    q = start.copy()
    for iteration in range(1, 501):
        current = mujoco.MjData(model)
        apply_pose(static, force_static, model, current, state, q)
        points = {name: anchor_world(model, current, anchors, name) for name in PADS}
        error = np.concatenate((target_average - np.mean(list(points.values()), axis=0), [-float(points[PADS[0]][2] - points[PADS[1]][2])]))
        if float(np.linalg.norm(error[:3])) <= 1e-5 and abs(float(error[3])) <= 1e-5:
            break
        jacs = {name: anchor_jacobian(model, current, anchors, name)[:, dofs] for name in PADS}
        jacobian = np.vstack(((jacs[PADS[0]] + jacs[PADS[1]]) * 0.5, (jacs[PADS[0]][2] - jacs[PADS[1]][2])[None, :]))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(4), error) - 0.002 * (q - start)
        q += np.clip(delta, -0.04, 0.04)
        for index, (low, high) in enumerate(bounds):
            q[index] = float(np.clip(q[index], low + 1e-5, high - 1e-5))
    final = mujoco.MjData(model)
    apply_pose(static, force_static, model, final, state, q)
    final_contacts = actual_contacts(model, final, cylinder)
    anchor_points = {name: anchor_world(model, final, anchors, name) for name in PADS}
    anchor_delta = float(anchor_points[PADS[0]][2] - anchor_points[PADS[1]][2])
    position_error = float(np.linalg.norm(np.mean(list(anchor_points.values()), axis=0) - target_average))
    level = bool(final_contacts["height_delta_m"] is not None and abs(float(final_contacts["height_delta_m"])) <= 0.002 and not final_contacts["nonpad_bodies"])
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-level-side-grasp-search")
    output = location.native_path
    payload = {"schema": "robotsim.issue46.stl.level-side-grasp-search.v1", "classification": "STATIC_LEVEL_SIDE_GRASP_SEARCH", "mj_step_calls": 0, "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]}, "source_scene": {"path": str(args.source_scene), "sha256": sha256(args.source_scene)}, "command_percent": COMMAND, "initial_contacts": initial, "anchor_local_m": {name: value.tolist() for name, value in anchors.items()}, "target_anchor_average_world_m": target_average.tolist(), "solution": {"arm_q_rad": dict(zip(static.ARM_JOINTS, q.tolist())), "iterations": iteration, "anchor_average_position_error_m": position_error, "anchor_height_delta_m": anchor_delta, "actual_contacts": final_contacts}, "gates": {"BILATERAL_DISTAL_CONTACT": "PASS" if all(final_contacts["counts"][name] > 0 for name in PADS) else "FAIL", "ACTUAL_CONTACT_HEIGHT_DELTA_LE_2MM": "PASS" if final_contacts["height_delta_m"] is not None and abs(float(final_contacts["height_delta_m"])) <= 0.002 else "FAIL", "NO_LOOP_OR_NONPAD_CONTACT": "PASS" if not final_contacts["nonpad_bodies"] else "FAIL", "LEVEL_SIDE_GRASP_CANDIDATE": "PASS" if level else "FAIL", "DYNAMICS": "NOT_RUN"}, "limitations": ["Contact means are sampled from current source-CoACD contacts; a level static candidate is not a physical lift result.", "No collision geometry or model parameter was modified."]}
    pg.write_new_text(output / "result.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "gates": payload["gates"], "actual_height_delta_mm": None if final_contacts["height_delta_m"] is None else final_contacts["height_delta_m"] * 1000.0, "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
