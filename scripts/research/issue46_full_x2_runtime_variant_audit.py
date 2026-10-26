#!/usr/bin/env python3
"""Read-only comparison of the accepted v1.4 full-X2 scene and RAICOM X2 XML.

This probe exists to prevent silently substituting the RAICOM runtime body for
the accepted full-X2 + frozen OmniPicker scene merely because both happen to
load in MuJoCo.  It only calls ``mj_forward`` and writes a governed evidence
record; it never modifies vendor resources or executes dynamics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
from agent import path_governance as pg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_relative_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, object] | None:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body < 0 or pelvis < 0:
        return None
    pelvis_rotation = data.xmat[pelvis].reshape(3, 3)
    body_rotation = data.xmat[body].reshape(3, 3)
    return {
        "translation_m": (pelvis_rotation.T @ (data.xpos[body] - data.xpos[pelvis])).tolist(),
        "rotation": (pelvis_rotation.T @ body_rotation).tolist(),
    }


def rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def active_head_geoms(model: mujoco.MjModel) -> list[dict[str, object]]:
    output = []
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        if body not in {"head_yaw_link", "head_pitch_link"}:
            continue
        output.append({
            "geom_id": int(geom),
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
            "body": body,
            "type": int(model.geom_type[geom]),
            "contype": int(model.geom_contype[geom]),
            "conaffinity": int(model.geom_conaffinity[geom]),
            "size_m": model.geom_size[geom].tolist(),
            "pos_m": model.geom_pos[geom].tolist(),
            "quat_wxyz": model.geom_quat[geom].tolist(),
        })
    return output


def head_torso_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    output = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1]))
        second_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2]))
        if {first_body, second_body} != {"head_pitch_link", "torso_link"}:
            continue
        output.append({
            "distance_m": float(contact.dist),
            "body_names": [first_body, second_body],
            "geom_ids": [int(contact.geom1), int(contact.geom2)],
        })
    return output


def joint_record(model: mujoco.MjModel, name: str) -> dict[str, object] | None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        return None
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
    return {
        "range_rad": model.jnt_range[joint].tolist(),
        "axis": model.jnt_axis[joint].tolist(),
        "pos_m": model.jnt_pos[joint].tolist(),
        "actuator_ctrlrange": None if actuator < 0 else model.actuator_ctrlrange[actuator].tolist(),
    }


def snapshot(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    ]
    names = [name for name in bodies if name]
    return model, data, {
        "path": str(path),
        "sha256": sha256(path),
        "counts": {
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            "nbody": int(model.nbody), "njnt": int(model.njnt),
            "ngeom": int(model.ngeom), "neq": int(model.neq), "initial_ncon": int(data.ncon),
        },
        "canonical_omnipicker_bodies": sorted(
            name for name in names if "omnipicker" in name.lower() or "hand_narrow" in name.lower() or "hand_wide" in name.lower()
        ),
        "head_collision_geoms": active_head_geoms(model),
        "head_torso_contacts": head_torso_contacts(model, data),
        "joint_records": {
            name: joint_record(model, name)
            for name in [
                "head_yaw_joint", "head_pitch_joint", "right_wrist_roll_joint",
                "right_shoulder_pitch_joint", "waist_pitch_joint",
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--accepted-scene", required=True, type=Path)
    parser.add_argument("--raicom-reference-scene", required=True, type=Path)
    parser.add_argument("--raicom-runtime-scene", required=True, type=Path)
    args = parser.parse_args()
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite evidence")
    paths = [args.accepted_scene.resolve(), args.raicom_reference_scene.resolve(), args.raicom_runtime_scene.resolve()]
    if any(not path.is_file() for path in paths):
        raise SystemExit("all source XML paths must exist")

    accepted_model, accepted_data, accepted = snapshot(paths[0])
    reference_model, reference_data, reference = snapshot(paths[1])
    runtime_model, runtime_data, runtime = snapshot(paths[2])
    relevant_bodies = ["torso_link", "head_yaw_link", "head_pitch_link", "right_wrist_roll_link", "left_wrist_roll_link"]
    frame_comparison = []
    for name in relevant_bodies:
        first = body_relative_pose(accepted_model, accepted_data, name)
        second = body_relative_pose(reference_model, reference_data, name)
        if first is None or second is None:
            continue
        first_rotation = np.asarray(first["rotation"], dtype=float)
        second_rotation = np.asarray(second["rotation"], dtype=float)
        frame_comparison.append({
            "body": name,
            "accepted_relative_to_pelvis": first,
            "raicom_relative_to_pelvis": second,
            "translation_delta_mm": float(1000.0 * np.linalg.norm(np.asarray(first["translation_m"]) - np.asarray(second["translation_m"]))),
            "rotation_delta_deg": rotation_delta_deg(first_rotation, second_rotation),
        })

    result = {
        "schema": "robotsim.issue46.full_x2_raicom_runtime_variant_audit.v1",
        "mujoco_version": mujoco.__version__,
        "mj_step_calls": 0,
        "accepted_option_d": accepted,
        "raicom_reference": reference,
        "raicom_runtime_copy": runtime,
        "accepted_vs_raicom_reference_relative_frames": frame_comparison,
        "decision": {
            "status": "NOT_SUBSTITUTABLE_WITHOUT_CASE_SPECIFIC_MODEL_APPROVAL",
            "reasons": [
                "The RAICOM XML has no canonical OmniPicker bodies or equality constraints.",
                "It has a different full-body count, joint/actuator parameter set, and measured torso/head/wrist frames.",
                "Its head collision policy and proxy dimensions differ from the accepted Option-D source.",
                "Using it as a replacement would change the accepted full-X2 model boundary rather than repair a controller or trajectory.",
            ],
            "potential_source_evidence_only": "The runtime XML is useful evidence that vendor runtime variants can use a different head collision policy; it does not authorize transplantation into the frozen v1.4 scene.",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": result["decision"]["status"], "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
