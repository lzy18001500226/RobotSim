#!/usr/bin/env python3
"""Search small wrist-frame rotations for an upward-capable centered STL contact.

The frozen gripper, cylinder, assets, source collision proxies, and existing
one-root force/equality overlay are retained.  Each candidate is static
``mj_forward`` only; it searches the robot arm pose, not a new gripper model.
"""
from __future__ import annotations

import argparse
import hashlib
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


MODEL_ID = "X2-OmniPicker-STL"
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
INNER_FACE_LOCAL = {
    "R_hand_narrow3_Link": np.array([0.0380, -0.0207, 0.0]),
    "R_hand_wide3_Link": np.array([0.0375, 0.0206, 0.0]),
}


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


def rotation(axis: int, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    if axis == 1:
        return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def jaw_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    points = []
    for name, local in INNER_FACE_LOCAL.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        points.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ local)
    return np.mean(np.asarray(points), axis=0)


def jaw_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    value = np.zeros((3, model.nv), dtype=float)
    for name, local in INNER_FACE_LOCAL.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, linear, angular, point, body)
        value += 0.5 * linear
    return value


def apply_pose(static_geometry, force_static, model, data, state, arm: np.ndarray, command_percent: float) -> None:
    static_geometry.apply_prior_state(model, data, state, arm)
    force_static.apply_positions(model, data, force_static.fit_positions(command_percent))


def solve_candidate(static_geometry, force_static, model, state, start: np.ndarray, target_center: np.ndarray, target_base_mat: np.ndarray, command_percent: float) -> dict[str, object]:
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    dofs, bounds = [], []
    for name in static_geometry.ARM_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofs.append(int(model.jnt_dofadr[joint]))
        bounds.append(tuple(float(v) for v in model.jnt_range[joint]))
    q = np.asarray(start, dtype=float).copy()
    for iteration in range(1, 401):
        data = mujoco.MjData(model)
        apply_pose(static_geometry, force_static, model, data, state, q, command_percent)
        mujoco.mj_forward(model, data)
        pos_error = target_center - jaw_center(model, data)
        rot_error = static_geometry.rotation_error_world(target_base_mat, data.xmat[base].reshape(3, 3))
        if float(np.linalg.norm(pos_error)) <= 1e-5 and float(np.linalg.norm(rot_error)) <= 1e-4:
            break
        base_pos, base_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, base_pos, base_rot, base)
        jacobian = np.vstack((jaw_jacobian(model, data)[:, dofs], 0.10 * base_rot[:, dofs]))
        error = np.concatenate((pos_error, 0.10 * rot_error))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-5 * np.eye(6), error)
        delta -= 0.002 * (q - start)
        q += np.clip(delta, -0.04, 0.04)
        for index, (low, high) in enumerate(bounds):
            q[index] = float(np.clip(q[index], low + 1e-5, high - 1e-5))
    final = mujoco.MjData(model)
    apply_pose(static_geometry, force_static, model, final, state, q, command_percent)
    mujoco.mj_forward(model, final)
    return {
        "arm_q_rad": {name: float(value) for name, value in zip(static_geometry.ARM_JOINTS, q)},
        "center_error_m": (jaw_center(model, final) - target_center).tolist(),
        "orientation_error_rad": static_geometry.rotation_error_world(target_base_mat, final.xmat[base].reshape(3, 3)).tolist(),
        "iterations": iteration,
    }


def force_on_cylinder(model: mujoco.MjModel, data: mujoco.MjData, index: int, cylinder: int) -> tuple[np.ndarray, float]:
    contact = data.contact[index]
    wrench = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(model, data, index, wrench)
    force_geom2 = np.asarray(contact.frame, dtype=float).reshape(3, 3).T @ wrench[:3]
    body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
    if cylinder == body2:
        return force_geom2, float(wrench[0])
    if cylinder == body1:
        return -force_geom2, float(wrench[0])
    raise RuntimeError("cylinder not part of contact")


def evaluate(model: mujoco.MjModel, data: mujoco.MjData, cylinder: int) -> dict[str, object]:
    pad_force = np.zeros(3, dtype=float)
    table_force = np.zeros(3, dtype=float)
    normal_by_pad = {name: 0.0 for name in PADS}
    other_bodies: set[str] = set()
    for index, contact in enumerate(data.contact):
        body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if cylinder not in (body1, body2):
            continue
        other = body2 if body1 == cylinder else body1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        force, normal = force_on_cylinder(model, data, index, cylinder)
        if name in PADS:
            pad_force += force
            normal_by_pad[name] += normal
        elif name == "raicom_table":
            table_force += force
        else:
            other_bodies.add(name)
    return {
        "pad_force_world_N": pad_force.tolist(),
        "pad_vertical_force_N": float(pad_force[2]),
        "table_vertical_force_N": float(table_force[2]),
        "normal_by_pad_N": normal_by_pad,
        "other_cylinder_contact_bodies": sorted(other_bodies),
        "bilateral_pad_contact": all(value > 1e-8 for value in normal_by_pad.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--source-static-result", required=True, type=Path)
    parser.add_argument("--force-fit-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--angles-deg", nargs="+", type=float, default=(-15.0, 0.0, 15.0))
    parser.add_argument("--yaw-angles-deg", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--vertical-offsets-mm", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--world-y-offsets-mm", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--local-z-offsets-mm", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--commands", nargs="+", type=int, default=(35,))
    args = parser.parse_args()
    source_static = json.loads(args.source_static_result.read_text(encoding="utf-8"))
    fit = json.loads(args.force_fit_result.read_text(encoding="utf-8"))
    if Path(source_static["generated_scene"]["path"]) != args.source_scene or sha256(args.source_scene) != source_static["generated_scene"]["sha256"]:
        raise SystemExit("source static result does not identify source scene")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if fit["model"]["sha256"] != loaded.model_sha256:
        raise SystemExit("frozen STL model hash differs from force-fit evidence")
    static_geometry = load_module("static_geometry", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force_static = load_module("force_static", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    state = static_geometry.state_by_phase(config, "HORIZONTAL_APPROACH")
    start = np.asarray([fit["force_fit_centering"]["arm_q_rad"][name] for name in static_geometry.ARM_JOINTS], dtype=float)
    source_model = mujoco.MjModel.from_xml_path(str(args.source_scene))
    source_data = mujoco.MjData(source_model)
    apply_pose(static_geometry, force_static, source_model, source_data, state, start, 35.0)
    mujoco.mj_forward(source_model, source_data)
    cylinder = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    base = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    target_center = source_data.xipos[cylinder].copy()
    reference_mat = source_data.xmat[base].reshape(3, 3).copy()
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-centered-orientation-search")
    output = location.native_path
    overlay_scene = output / "orientation_force_overlay.xml"
    overlay = force_static.build_force_control_scene(args.source_scene, overlay_scene, torque_cap_nm=0.10)
    model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    root_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_claw_stl_force")
    cylinder_overlay = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    rows = []
    if any(command < 0 or command > 100 for command in args.commands):
        raise SystemExit("commands must be in 0..100")
    for command in args.commands:
        for offset_mm in args.vertical_offsets_mm:
            for y_offset_mm in args.world_y_offsets_mm:
                for local_z_offset_mm in args.local_z_offsets_mm:
                    for yaw in args.yaw_angles_deg:
                        for roll in args.angles_deg:
                            for pitch in args.angles_deg:
                                target_mat = reference_mat @ rotation(2, yaw) @ rotation(1, pitch) @ rotation(0, roll)
                                requested_center = target_center + np.array((0.0, y_offset_mm / 1000.0, offset_mm / 1000.0)) + reference_mat[:, 2] * (local_z_offset_mm / 1000.0)
                                solved = solve_candidate(static_geometry, force_static, source_model, state, start, requested_center, target_mat, float(command))
                                arm = np.asarray([solved["arm_q_rad"][name] for name in static_geometry.ARM_JOINTS], dtype=float)
                                data = mujoco.MjData(model)
                                apply_pose(static_geometry, force_static, model, data, state, arm, float(command))
                                data.ctrl[root_actuator] = 0.10
                                mujoco.mj_forward(model, data)
                                contact = evaluate(model, data, cylinder_overlay)
                                rows.append({"command_percent": command, "vertical_offset_mm": offset_mm, "world_y_offset_mm": y_offset_mm, "local_z_offset_mm": local_z_offset_mm, "local_yaw_deg": yaw, "local_roll_deg": roll, "local_pitch_deg": pitch, "ik": solved, "contact": contact})
    valid = [row for row in rows if row["contact"]["bilateral_pad_contact"] and not row["contact"]["other_cylinder_contact_bodies"]]
    ranked = sorted(valid, key=lambda row: (row["contact"]["pad_vertical_force_N"], -row["contact"]["table_vertical_force_N"]), reverse=True)
    payload = {
        "schema": "robotsim.issue46.stl.centered-orientation-search.v1",
        "classification": "STATIC_WRIST_ORIENTATION_CONTACT_SEARCH",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "source_scene": {"path": str(args.source_scene), "sha256": sha256(args.source_scene)},
        "force_fit_result": {"path": str(args.force_fit_result), "sha256": sha256(args.force_fit_result)},
        "overlay": overlay,
        "commands_percent": list(args.commands),
        "root_torque_command_Nm": 0.10,
        "angles_deg": list(args.angles_deg),
        "yaw_angles_deg": list(args.yaw_angles_deg),
        "vertical_offsets_mm": list(args.vertical_offsets_mm),
        "world_y_offsets_mm": list(args.world_y_offsets_mm),
        "local_z_offsets_mm": list(args.local_z_offsets_mm),
        "rows": rows,
        "valid_candidate_count": len(valid),
        "ranked_valid_candidates": ranked,
        "limitations": ["Static mj_forward search only; no contact dynamics or lift executed.", "A selected pose requires separate human visual review before a dynamics trial."],
    }
    pg.write_new_text(output / "result.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "valid_candidate_count": len(valid), "top_candidates": [{"command": row["command_percent"], "vertical_offset_mm": row["vertical_offset_mm"], "world_y_offset_mm": row["world_y_offset_mm"], "local_z_offset_mm": row["local_z_offset_mm"], "yaw": row["local_yaw_deg"], "roll": row["local_roll_deg"], "pitch": row["local_pitch_deg"], "pad_z_N": row["contact"]["pad_vertical_force_N"], "table_z_N": row["contact"]["table_vertical_force_N"]} for row in ranked[:5]], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
