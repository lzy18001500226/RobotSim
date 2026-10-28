#!/usr/bin/env python3
"""Screen horizontal side-pinches around an upright bottle at several azimuths.

The baseline's visible horizontal-forearm semantics stay hard constraints.  A
world-Z yaw is applied to the reviewed wrist/base frame, so this explores where
around the upright bottle the same side-pinch may first contact.  No mj_step is
called and no model asset or collision setting is changed.
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


def rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def rotation_axis(axis: np.ndarray, degrees: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    angle = math.radians(degrees)
    cross = np.array(((0.0, -unit[2], unit[1]), (unit[2], 0.0, -unit[0]), (-unit[1], unit[0], 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def solve(
    *,
    ik,
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
    target_rotation: np.ndarray,
    ids: dict[str, int],
) -> tuple[np.ndarray, dict[str, float]]:
    dofs = np.asarray(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in names],
        dtype=int,
    )
    q = seed.copy()
    for iteration in range(1, 301):
        data = mujoco.MjData(model)
        ik.apply_pose(static, force, model, data, state, names, q, command)
        mujoco.mj_forward(model, data)
        center_error = target_center - orient.jaw_center(model, data)
        forearm_delta = float(data.xpos[ids["wrist"], 2] - data.xpos[ids["elbow"], 2])
        distal_delta = float(data.xpos[ids["narrow"], 2] - data.xpos[ids["wide"], 2])
        base_rotation = data.xmat[ids["base"]].reshape(3, 3)
        rotation_error = static.rotation_error_world(target_rotation, base_rotation)
        if (
            np.linalg.norm(center_error) <= 1e-4
            and abs(forearm_delta) <= 1e-4
            and abs(distal_delta) <= 1e-4
            and np.linalg.norm(rotation_error) <= 1e-3
        ):
            break
        center_jac = orient.jaw_jacobian(model, data)[:, dofs]
        elbow_jac = ik.point_jacobian(model, data, ids["elbow"], dofs)
        wrist_jac = ik.point_jacobian(model, data, ids["wrist"], dofs)
        narrow_jac = ik.point_jacobian(model, data, ids["narrow"], dofs)
        wide_jac = ik.point_jacobian(model, data, ids["wide"], dofs)
        base_position_jac = np.zeros((3, model.nv))
        base_rotation_jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, base_position_jac, base_rotation_jac, ids["base"])
        error = np.concatenate((center_error, (-forearm_delta,), (-distal_delta,), 0.10 * rotation_error))
        jacobian = np.vstack(
            (
                center_jac,
                wrist_jac[2] - elbow_jac[2],
                narrow_jac[2] - wide_jac[2],
                0.10 * base_rotation_jac[:, dofs],
            )
        )
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-7 * np.eye(8), error)
        delta -= 1e-4 * (q - seed)
        q += np.clip(delta, -0.025, 0.025)
        q = np.clip(q, limits[:, 0] + 1e-5, limits[:, 1] - 1e-5)

    final = mujoco.MjData(model)
    ik.apply_pose(static, force, model, final, state, names, q, command)
    mujoco.mj_forward(model, final)
    return q, {
        "iterations": iteration,
        "jaw_center_error_m": float(np.linalg.norm(orient.jaw_center(model, final) - target_center)),
        "forearm_height_delta_m": float(final.xpos[ids["wrist"], 2] - final.xpos[ids["elbow"], 2]),
        "distal_reference_height_delta_m": float(final.xpos[ids["narrow"], 2] - final.xpos[ids["wide"], 2]),
        "base_orientation_error_rad": float(
            np.linalg.norm(static.rotation_error_world(target_rotation, final.xmat[ids["base"]].reshape(3, 3)))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--azimuth-deg", nargs="+", type=float, default=(-20.0, -10.0, 0.0, 10.0, 20.0))
    parser.add_argument("--tool-pitch-deg", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--height-offset-mm", nargs="+", type=float, default=(80.0, 100.0, 120.0))
    parser.add_argument("--commands", nargs="+", type=int, default=(32, 34, 36, 38, 40))
    args = parser.parse_args()
    if not args.freeze_record.is_file() or any(command < 0 or command > 100 for command in args.commands):
        raise SystemExit("freeze record is missing or command is outside 0..100")
    frozen = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    if frozen.get("baseline_id") != "X2-OmniPicker-STL-Horizontal-Side-Grasp-Visual-v1":
        raise SystemExit("freeze record is not the accepted horizontal STL baseline")
    review_path = Path(frozen["accepted_review"]["path"])
    if not review_path.is_file() or sha256(review_path) != frozen["accepted_review"]["sha256"]:
        raise SystemExit("accepted review state is missing or changed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    scene = Path(review["scene"]["path"])
    if not scene.is_file() or sha256(scene) != review["scene"]["sha256"]:
        raise SystemExit("accepted scene is missing or changed")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != frozen["model"]["sha256"]:
        raise SystemExit("frozen STL model hash differs from freeze record")
    ik = load_module("horizontal_ik", REPO / "scripts/research/issue46_stl_horizontal_side_grasp_ik_scan.py")
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    orient = load_module("stl_orientation", REPO / "scripts/research/issue46_stl_centered_orientation_search.py")
    level = load_module("stl_level", REPO / "scripts/research/issue46_stl_level_side_grasp_search.py")
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = review["state"]
    names = ik.WAIST_JOINTS + static.ARM_JOINTS
    limits = ik.joint_limits(model, names)
    arm = review["samples"][str(review["initial_command_percent"])]["arm_q_rad"]
    waist = review["motor_override_rad"]
    seed = np.asarray([waist[name] for name in ik.WAIST_JOINTS] + [arm[name] for name in static.ARM_JOINTS], dtype=float)
    ids = {name: ik.body_id(model, body) for name, body in {"elbow": "right_elbow_link", "wrist": "right_wrist_roll_link", "narrow": "R_hand_narrow3_Link", "wide": "R_hand_wide3_Link", "base": "R_omnipicker_base_link", "cylinder": "test_cylinder"}.items()}
    reference = mujoco.MjData(model)
    ik.apply_pose(static, force, model, reference, state, names, seed, int(review["initial_command_percent"]))
    mujoco.mj_forward(model, reference)
    cylinder_center = reference.xipos[ids["cylinder"]].copy()
    reference_rotation = reference.xmat[ids["base"]].reshape(3, 3).copy()
    rows = []
    for height_mm in args.height_offset_mm:
        for azimuth_deg in args.azimuth_deg:
            target_center = cylinder_center + np.array((0.0, 0.0, height_mm / 1000.0))
            yaw_rotation = rotation_z(azimuth_deg) @ reference_rotation
            for tool_pitch_deg in args.tool_pitch_deg:
                # Pitch about the horizontal jaw-separation axis.  This keeps
                # that axis level while slightly changing the approach normal.
                target_rotation = rotation_axis(yaw_rotation[:, 1], tool_pitch_deg) @ yaw_rotation
                q, metrics = solve(
                    ik=ik,
                    static=static,
                    force=force,
                    orient=orient,
                    model=model,
                    state=state,
                    names=names,
                    limits=limits,
                    seed=seed,
                    command=int(review["initial_command_percent"]),
                    target_center=target_center,
                    target_rotation=target_rotation,
                    ids=ids,
                )
                for command in args.commands:
                    data = mujoco.MjData(model)
                    ik.apply_pose(static, force, model, data, state, names, q, command)
                    mujoco.mj_forward(model, data)
                    contacts = level.actual_contacts(model, data, ids["cylinder"])
                    rows.append(
                        {
                            "height_offset_mm": height_mm,
                            "azimuth_deg": azimuth_deg,
                            "tool_pitch_deg": tool_pitch_deg,
                            "command_percent": command,
                            "arm_q_rad": {name: float(value) for name, value in zip(static.ARM_JOINTS, q[3:])},
                            "waist_q_rad": {name: float(value) for name, value in zip(ik.WAIST_JOINTS, q[:3])},
                            "pose_metrics": metrics,
                            "actual_contacts": contacts,
                        }
                    )
    valid = [
        row
        for row in rows
        if row["pose_metrics"]["jaw_center_error_m"] <= 1e-4
        and abs(row["pose_metrics"]["forearm_height_delta_m"]) <= 1e-4
        and abs(row["pose_metrics"]["distal_reference_height_delta_m"]) <= 1e-4
        and row["pose_metrics"]["base_orientation_error_rad"] <= 1e-3
        and all(row["actual_contacts"]["counts"][name] > 0 for name in ik.DISTAL_BODIES)
        and not row["actual_contacts"]["nonpad_bodies"]
    ]
    ranked = sorted(
        valid,
        key=lambda row: (
            float("inf") if row["actual_contacts"]["height_delta_m"] is None else abs(float(row["actual_contacts"]["height_delta_m"])),
            abs(float(row["azimuth_deg"])),
            abs(float(row["tool_pitch_deg"])),
        ),
    )
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-side-grasp-azimuth-scan")
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl.horizontal-side-grasp-azimuth-scan.v1",
        "classification": "STATIC_IK_AZIMUTH_CONTACT_SCREENING",
        "mj_step_calls": 0,
        "freeze_record": {"path": str(args.freeze_record), "sha256": sha256(args.freeze_record)},
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "scene": review["scene"],
        "scan": {"azimuth_deg": args.azimuth_deg, "tool_pitch_deg": args.tool_pitch_deg, "height_offset_mm": args.height_offset_mm, "commands": args.commands},
        "rows": rows,
        "valid_dual_distal_pad_candidates": ranked,
        "gates": {"STATIC_AZIMUTH_SCREEN": "PASS" if ranked else "BLOCKED", "DYNAMICS": "NOT_RUN", "PHYSICAL_GRASP": "NOT_RUN"},
        "limitations": [
            "Yaw is a planning/IK variable, not a gripper-mechanism modification.",
            "Every measurement is a static collision-proxy contact evaluation; no force closure or lift is claimed.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "candidate_count": len(ranked), "top": [{"height_offset_mm": row["height_offset_mm"], "azimuth_deg": row["azimuth_deg"], "tool_pitch_deg": row["tool_pitch_deg"], "command_percent": row["command_percent"], "contact_height_delta_mm": row["actual_contacts"]["height_delta_m"] * 1000.0} for row in ranked[:10]], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
