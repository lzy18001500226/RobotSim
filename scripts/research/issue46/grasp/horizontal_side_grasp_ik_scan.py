#!/usr/bin/env python3
"""Static IK/contact scan around the frozen horizontal STL side-grasp pose.

The scan changes only the full-X2 right-arm/waist configuration and the
already-approved single root command.  It never changes the frozen gripper,
object, table, collision setup, or source meshes, and it never calls mj_step.
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


REPO = Path(__file__).resolve().parents[4]
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


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"missing body: {name}")
    return value


def joint_limits(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not bool(model.jnt_limited[joint]):
            raise RuntimeError(f"scan joint must have finite source limits: {name}")
        values.append(model.jnt_range[joint])
    return np.asarray(values, dtype=float)


def apply_pose(static, force, model, data, state, names: tuple[str, ...], q: np.ndarray, command: int) -> None:
    motor_override = {name: float(value) for name, value in zip(names[:3], q[:3])}
    static.apply_prior_state(model, data, state, q[3:], motor_override)
    force.apply_positions(model, data, force.fit_positions(float(command)))


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body: int, dofs: np.ndarray) -> np.ndarray:
    position = np.zeros((3, model.nv))
    rotation = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, position, rotation, body)
    return position[:, dofs]


def solve_level_horizontal_pose(
    *,
    static,
    force,
    orient,
    model: mujoco.MjModel,
    state: dict[str, object],
    names: tuple[str, ...],
    limits: np.ndarray,
    seed: np.ndarray,
    command: int,
    target_center: np.ndarray,
    elbow: int,
    wrist: int,
    narrow: int,
    wide: int,
) -> tuple[np.ndarray, dict[str, float]]:
    dofs = np.asarray(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
        dtype=int,
    )
    q = seed.copy()
    for iteration in range(1, 251):
        data = mujoco.MjData(model)
        apply_pose(static, force, model, data, state, names, q, command)
        mujoco.mj_forward(model, data)
        center_error = target_center - orient.jaw_center(model, data)
        forearm_delta = float(data.xpos[wrist, 2] - data.xpos[elbow, 2])
        distal_delta = float(data.xpos[narrow, 2] - data.xpos[wide, 2])
        if np.linalg.norm(center_error) <= 1e-4 and abs(forearm_delta) <= 1e-4 and abs(distal_delta) <= 1e-4:
            break
        center_jac = orient.jaw_jacobian(model, data)[:, dofs]
        elbow_jac = point_jacobian(model, data, elbow, dofs)
        wrist_jac = point_jacobian(model, data, wrist, dofs)
        narrow_jac = point_jacobian(model, data, narrow, dofs)
        wide_jac = point_jacobian(model, data, wide, dofs)
        error = np.concatenate((center_error, (-forearm_delta,), (-distal_delta,)))
        jacobian = np.vstack((center_jac, wrist_jac[2] - elbow_jac[2], narrow_jac[2] - wide_jac[2]))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-7 * np.eye(5), error)
        delta -= 1e-4 * (q - seed)
        q += np.clip(delta, -0.025, 0.025)
        q = np.clip(q, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)

    final = mujoco.MjData(model)
    apply_pose(static, force, model, final, state, names, q, command)
    mujoco.mj_forward(model, final)
    return q, {
        "iterations": iteration,
        "jaw_center_error_m": float(np.linalg.norm(orient.jaw_center(model, final) - target_center)),
        "forearm_height_delta_m": float(final.xpos[wrist, 2] - final.xpos[elbow, 2]),
        "distal_reference_height_delta_m": float(final.xpos[narrow, 2] - final.xpos[wide, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--height-offset-mm", nargs="+", type=float, default=(80.0, 100.0, 120.0))
    parser.add_argument("--approach-offset-mm", nargs="+", type=float, default=(-10.0, 0.0, 10.0))
    parser.add_argument("--pinch-offset-mm", nargs="+", type=float, default=(-5.0, 0.0, 5.0))
    parser.add_argument("--commands", nargs="+", type=int, default=(32, 34, 36, 38, 40))
    args = parser.parse_args()
    if not args.freeze_record.is_file():
        raise SystemExit("freeze record is missing")
    if any(not math.isfinite(value) for value in (*args.height_offset_mm, *args.approach_offset_mm, *args.pinch_offset_mm)):
        raise SystemExit("scan offsets must be finite")
    if any(command < 0 or command > 100 for command in args.commands):
        raise SystemExit("commands must lie in 0..100")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    if frozen.get("baseline_id") != "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1":
        raise SystemExit("freeze record is not the accepted horizontal STL baseline")
    review_path = Path(frozen["accepted_review"]["path"])
    if not review_path.is_file() or sha256(review_path) != frozen["accepted_review"]["sha256"]:
        raise SystemExit("accepted review state is missing or changed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    scene = Path(review["scene"]["path"])
    if not scene.is_file() or sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted source scene is missing or changed")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"]:
        raise SystemExit("frozen STL model hash differs from freeze record")

    static = load_module("stl_static", REPO / "scripts/research/issue46/grasp/cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46/grasp/force_control_static_preflight.py")
    orient = load_module("stl_orientation", REPO / "scripts/research/issue46/grasp/centered_orientation_search.py")
    level = load_module("stl_level", REPO / "scripts/research/issue46/grasp/level_side_grasp_search.py")
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = review["state"]
    names = WAIST_JOINTS + static.ARM_JOINTS
    limits = joint_limits(model, names)
    base_arm = review["samples"][str(review["initial_command_percent"])]["arm_q_rad"]
    base_waist = review["motor_override_rad"]
    seed = np.asarray([base_waist[name] for name in WAIST_JOINTS] + [base_arm[name] for name in static.ARM_JOINTS], dtype=float)
    elbow = body_id(model, "right_elbow_link")
    wrist = body_id(model, "right_wrist_roll_link")
    narrow = body_id(model, DISTAL_BODIES[0])
    wide = body_id(model, DISTAL_BODIES[1])
    cylinder = body_id(model, "test_cylinder")

    reference = mujoco.MjData(model)
    apply_pose(static, force, model, reference, state, names, seed, int(review["initial_command_percent"]))
    mujoco.mj_forward(model, reference)
    cylinder_center = reference.xipos[cylinder].copy()
    pinch_axis = reference.xpos[wide] - reference.xpos[narrow]
    pinch_axis[2] = 0.0
    pinch_axis /= np.linalg.norm(pinch_axis)
    approach_axis = reference.xpos[body_id(model, "R_omnipicker_base_link")] - cylinder_center
    approach_axis[2] = 0.0
    approach_axis /= np.linalg.norm(approach_axis)

    rows = []
    for height_mm in args.height_offset_mm:
        for approach_mm in args.approach_offset_mm:
            for pinch_mm in args.pinch_offset_mm:
                target = (
                    cylinder_center
                    + np.array((0.0, 0.0, height_mm / 1000.0))
                    + approach_axis * (approach_mm / 1000.0)
                    + pinch_axis * (pinch_mm / 1000.0)
                )
                q, pose_metrics = solve_level_horizontal_pose(
                    static=static,
                    force=force,
                    orient=orient,
                    model=model,
                    state=state,
                    names=names,
                    limits=limits,
                    seed=seed,
                    command=int(review["initial_command_percent"]),
                    target_center=target,
                    elbow=elbow,
                    wrist=wrist,
                    narrow=narrow,
                    wide=wide,
                )
                for command in args.commands:
                    data = mujoco.MjData(model)
                    apply_pose(static, force, model, data, state, names, q, command)
                    mujoco.mj_forward(model, data)
                    contacts = level.actual_contacts(model, data, cylinder)
                    rows.append(
                        {
                            "height_offset_mm": height_mm,
                            "approach_offset_mm": approach_mm,
                            "pinch_offset_mm": pinch_mm,
                            "command_percent": command,
                            "arm_q_rad": {name: float(value) for name, value in zip(static.ARM_JOINTS, q[3:])},
                            "waist_q_rad": {name: float(value) for name, value in zip(WAIST_JOINTS, q[:3])},
                            "pose_metrics": pose_metrics,
                            "actual_contacts": contacts,
                        }
                    )

    valid = [
        row
        for row in rows
        if row["pose_metrics"]["jaw_center_error_m"] <= 1e-4
        and abs(row["pose_metrics"]["forearm_height_delta_m"]) <= 1e-4
        and abs(row["pose_metrics"]["distal_reference_height_delta_m"]) <= 1e-4
        and all(row["actual_contacts"]["counts"][name] > 0 for name in DISTAL_BODIES)
        and not row["actual_contacts"]["nonpad_bodies"]
    ]
    def rank_key(row: dict[str, object]) -> tuple[float, float, float]:
        height_delta = row["actual_contacts"]["height_delta_m"]
        return (
            float("inf") if height_delta is None else abs(float(height_delta)),
            abs(float(row["approach_offset_mm"])),
            abs(float(row["pinch_offset_mm"])),
        )

    ranked = sorted(valid, key=rank_key)
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-side-grasp-ik-scan")
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl.horizontal-side-grasp-ik-scan.v1",
        "classification": "STATIC_IK_CONTACT_SCREENING",
        "mj_step_calls": 0,
        "freeze_record": {"path": str(args.freeze_record), "sha256": sha256(args.freeze_record)},
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "scene": review["scene"],
        "coordinate_definition": {
            "cylinder_axis": "world +Z; cylinder remains upright",
            "pinch_axis_world": pinch_axis.tolist(),
            "approach_axis_world": approach_axis.tolist(),
            "height_offset_reference": "cylinder center",
        },
        "scan": {
            "height_offset_mm": args.height_offset_mm,
            "approach_offset_mm": args.approach_offset_mm,
            "pinch_offset_mm": args.pinch_offset_mm,
            "commands": args.commands,
        },
        "rows": rows,
        "valid_dual_distal_pad_candidates": ranked,
        "gates": {
            "STATIC_IK_SCREEN": "PASS" if ranked else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
            "PHYSICAL_GRASP": "NOT_RUN",
        },
        "limitations": [
            "Contact is evaluated against the selected CoACD collision proxies; original STL visual geometry is unchanged.",
            "This scan does not prove force closure, load transfer, stability, or a lift.",
            "Only a separately authorized mj_step trial can establish physical grasping.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "candidate_count": len(ranked),
                "top": [
                    {
                        "height_offset_mm": item["height_offset_mm"],
                        "approach_offset_mm": item["approach_offset_mm"],
                        "pinch_offset_mm": item["pinch_offset_mm"],
                        "command_percent": item["command_percent"],
                        "contact_height_delta_mm": item["actual_contacts"]["height_delta_m"] * 1000.0,
                    }
                    for item in ranked[:10]
                ],
                "mj_step_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
