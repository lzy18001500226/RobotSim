#!/usr/bin/env python3
"""Map existing root-force overlay contact wrench for screened horizontal IK poses.

This is deliberately an ``mj_forward``-only ranking step.  It does not run a
physics rollout and does not alter the accepted STL baseline or its assets.
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


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


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


def body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body#{body}"


def force_on_body(model: mujoco.MjModel, data: mujoco.MjData, contact_index: int, body: int) -> tuple[np.ndarray, float]:
    contact = data.contact[contact_index]
    wrench = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(model, data, contact_index, wrench)
    world_force_on_geom2 = np.asarray(contact.frame, dtype=float).reshape(3, 3).T @ wrench[:3]
    body1 = int(model.geom_bodyid[contact.geom1])
    body2 = int(model.geom_bodyid[contact.geom2])
    if body == body2:
        return world_force_on_geom2, float(wrench[0])
    if body == body1:
        return -world_force_on_geom2, float(wrench[0])
    raise RuntimeError("contact does not contain requested body")


def measure(model: mujoco.MjModel, data: mujoco.MjData, cylinder: int) -> dict[str, object]:
    pad_force = np.zeros(3, dtype=float)
    table_force = np.zeros(3, dtype=float)
    normal_by_pad = {name: 0.0 for name in PADS}
    nonpad: set[str] = set()
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if cylinder not in (body1, body2):
            continue
        other = body2 if body1 == cylinder else body1
        other_name = body_name(model, other)
        force, normal = force_on_body(model, data, index, cylinder)
        contacts.append(
            {
                "other_body": other_name,
                "distance_m": float(contact.dist),
                "force_on_cylinder_world_N": force.tolist(),
                "normal_force_N": normal,
            }
        )
        if other_name in PADS:
            pad_force += force
            normal_by_pad[other_name] += normal
        elif other_name == "raicom_table":
            table_force += force
        elif other_name.startswith("R_hand_"):
            nonpad.add(other_name)
    return {
        "pad_force_on_cylinder_world_N": pad_force.tolist(),
        "pad_vertical_force_N": float(pad_force[2]),
        "pad_normal_force_by_body_N": normal_by_pad,
        "table_vertical_force_N": float(table_force[2]),
        "nonpad_cylinder_bodies": sorted(nonpad),
        "contacts": contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik-scan", required=True, type=Path)
    parser.add_argument("--root-torque-nm", type=float, default=0.05)
    args = parser.parse_args()
    if not args.ik_scan.is_file() or not (0.0 < args.root_torque_nm <= 0.50):
        raise SystemExit("IK scan must exist and torque must lie in (0, 0.50]")
    scan = json.loads(args.ik_scan.read_text(encoding="utf-8"))
    if scan.get("classification") not in {"STATIC_IK_CONTACT_SCREENING", "STATIC_IK_AZIMUTH_CONTACT_SCREENING"}:
        raise SystemExit("input is not a supported horizontal static IK scan")
    scene = Path(scan["scene"]["path"])
    if not scene.is_file() or sha256(scene) != scan["scene"]["sha256"]:
        raise SystemExit("IK scan scene is missing or hash-mismatched")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != scan["model"]["sha256"]:
        raise SystemExit("frozen STL model differs from IK scan")
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    review = json.loads(Path(scan["freeze_record"]["path"]).read_text(encoding="utf-8"))
    review_state = json.loads(Path(review["accepted_review"]["path"]).read_text(encoding="utf-8"))
    state = review_state["state"]
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-side-grasp-wrench-map")
    output = location.native_path
    overlay_scene = output / "existing_root_force_overlay.xml"
    overlay = force.build_force_control_scene(scene, overlay_scene, torque_cap_nm=0.50)
    model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_claw_stl_force")
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if actuator < 0 or cylinder < 0:
        raise RuntimeError("force overlay lacks root actuator or cylinder")
    rows = []
    for candidate in scan["valid_dual_distal_pad_candidates"]:
        data = mujoco.MjData(model)
        arm = np.asarray([candidate["arm_q_rad"][name] for name in static.ARM_JOINTS], dtype=float)
        static.apply_prior_state(model, data, state, arm, candidate["waist_q_rad"])
        force.apply_positions(model, data, force.fit_positions(float(candidate["command_percent"])))
        data.ctrl[actuator] = args.root_torque_nm
        mujoco.mj_forward(model, data)
        retained = ("height_offset_mm", "approach_offset_mm", "pinch_offset_mm", "azimuth_deg", "tool_pitch_deg", "command_percent", "arm_q_rad", "waist_q_rad", "pose_metrics", "actual_contacts")
        row = {key: candidate[key] for key in retained if key in candidate}
        row["wrench"] = measure(model, data, cylinder)
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_claw_joint")
        root_q = float(data.qpos[model.jnt_qposadr[root_joint]])
        row["max_follower_equality_residual_rad"] = max(
            abs(
                float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]])
                - force.poly(root_q, coefficients)
            )
            for name, coefficients in force.FIT_POLYNOMIALS.items()
        )
        rows.append(row)
    valid = [
        row
        for row in rows
        if not row["wrench"]["nonpad_cylinder_bodies"]
        and all(value > 1e-8 for value in row["wrench"]["pad_normal_force_by_body_N"].values())
        and row["max_follower_equality_residual_rad"] <= np.deg2rad(1.0)
    ]
    ranked = sorted(
        valid,
        key=lambda row: (
            -float(row["wrench"]["pad_vertical_force_N"]),
            -min(float(value) for value in row["wrench"]["pad_normal_force_by_body_N"].values()),
        ),
    )
    result = {
        "schema": "robotsim.issue46.stl.horizontal-side-grasp-wrench-map.v1",
        "classification": "STATIC_EXISTING_ROOT_FORCE_OVERLAY_RANKING",
        "mj_step_calls": 0,
        "ik_scan": {"path": str(args.ik_scan), "sha256": sha256(args.ik_scan)},
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "scene": scan["scene"],
        "overlay": {**overlay, "path": str(overlay_scene), "sha256": sha256(overlay_scene)},
        "root_torque_command_Nm": args.root_torque_nm,
        "rows": rows,
        "ranked_candidates": ranked,
        "gates": {
            "STATIC_WRENCH_SCREEN": "PASS" if ranked else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
            "PHYSICAL_GRASP": "NOT_RUN",
        },
        "limitations": [
            "This uses the existing experimental root-force/equality overlay only to rank candidates.",
            "No physics step, lift, load transfer, or force-closure PASS is claimed.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "ranked_candidate_count": len(ranked),
                "top": [
                    {
                        "pose": {key: item[key] for key in ("height_offset_mm", "approach_offset_mm", "pinch_offset_mm", "azimuth_deg", "tool_pitch_deg") if key in item},
                        "command_percent": item["command_percent"],
                        "pad_vertical_force_N": item["wrench"]["pad_vertical_force_N"],
                        "table_vertical_force_N": item["wrench"]["table_vertical_force_N"],
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
