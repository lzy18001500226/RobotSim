#!/usr/bin/env python3
"""Bounded right-side Robotiq mount comparison for Issue #46.

Candidate A is the preserved official X2 tool mount. Candidate B applies a
single 180-degree local-Y half-turn to the right Robotiq adapter. B is diagnostic and
does not modify the pinned X2 or Menagerie sources.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def obj_name(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def body_geoms(model, body_name: str) -> list[int]:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise RuntimeError(f"Missing body {body_name}")
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == body_id
            and (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))]


def pair_distances(model, data, wrist_body: str, gripper_prefix: str) -> list[dict[str, Any]]:
    rows = []
    for wrist_geom in body_geoms(model, wrist_body):
        for gripper_body_id in range(model.nbody):
            body = obj_name(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_id)
            if not body.startswith(gripper_prefix):
                continue
            for gripper_geom in range(model.ngeom):
                if int(model.geom_bodyid[gripper_geom]) != gripper_body_id:
                    continue
                if not (int(model.geom_contype[gripper_geom]) or int(model.geom_conaffinity[gripper_geom])):
                    continue
                segment = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(
                    model, data, wrist_geom, gripper_geom, 0.5, segment))
                rows.append({
                    "wrist_geom": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, wrist_geom),
                    "gripper_body": body,
                    "gripper_geom": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gripper_geom),
                    "distance_m": distance,
                    "witness_segment_m": segment.tolist(),
                })
    return sorted(rows, key=lambda row: row["distance_m"])


def compiled_hand_axes(model, data, side: str) -> dict[str, Any]:
    prefix = f"rq_{side}_"
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   prefix + "base_mount"))
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    origin = np.asarray(data.xpos[body_id])
    left_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      prefix + f"left_pad{index}")) for index in (1, 2)]
    right_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       prefix + f"right_pad{index}")) for index in (1, 2)]
    left_center = np.asarray(data.geom_xpos[left_ids]).mean(axis=0)
    right_center = np.asarray(data.geom_xpos[right_ids]).mean(axis=0)
    insertion = (left_center + right_center) / 2.0 - origin
    insertion /= np.linalg.norm(insertion)
    opening = left_center - right_center
    opening /= np.linalg.norm(opening)
    return {
        "insertion_axis_local": (rotation.T @ insertion).tolist(),
        "jaw_opening_axis_local": (rotation.T @ opening).tolist(),
    }


def summarize_contacts(model, data) -> list[dict[str, Any]]:
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        force = np.zeros(6)
        if int(contact.efc_address) >= 0:
            mujoco.mj_contactForce(model, data, index, force)
        result.append({
            "body1": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
            "geom1": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body2": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
            "geom2": obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "distance_m": float(contact.dist),
            "normal_world": np.asarray(contact.frame[:3], dtype=float).tolist(),
            "force_contact_frame_n": force.tolist(),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
        })
    return result


def write_halfturn_candidate(source_xml: Path, candidate_xml: Path) -> dict[str, Any]:
    root = ET.parse(source_xml).getroot()
    source_root = copy.deepcopy(root)
    body = next((node for node in root.iter("body")
                 if node.get("name") == "rq_right_base_mount"), None)
    if body is None:
        raise RuntimeError("Expected right adapter body rq_right_base_mount")
    old_wxyz = np.asarray([float(value) for value in body.get("quat", "1 0 0 0").split()])
    old = Rotation.from_quat([old_wxyz[1], old_wxyz[2], old_wxyz[3], old_wxyz[0]])
    # In the compiled adapter local Z is insertion and local -X is jaw opening.
    # A local-Y half-turn reverses both axes; it is not axial roll.
    delta = Rotation.from_euler("y", np.pi)
    new = old * delta
    xyzw = new.as_quat()
    new_wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    body.set("quat", " ".join(f"{value:.15g}" for value in new_wxyz))
    ET.indent(root, space="  ")
    candidate_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(candidate_xml, encoding="utf-8", xml_declaration=True)
    source_body = next(node for node in source_root.iter("body")
                       if node.get("name") == "rq_right_base_mount")
    candidate_root = ET.parse(candidate_xml).getroot()
    candidate_body = next(node for node in candidate_root.iter("body")
                          if node.get("name") == "rq_right_base_mount")
    candidate_body.set("quat", source_body.get("quat", "1 0 0 0"))
    for tree in (source_root, candidate_root):
        for node in tree.iter():
            node.text = None
            node.tail = None
    if ET.tostring(source_root) != ET.tostring(candidate_root):
        raise RuntimeError("Candidate B changed XML beyond the right adapter quaternion")
    return {
        "operation": "right adapter local-Y half-turn 180 degrees",
        "local_z_is_insertion_axis": True,
        "local_minus_x_is_jaw_opening_axis": True,
        "halfturn_reverses_insertion_and_opening": True,
        "translation_unchanged": True,
        "old_quat_wxyz": old_wxyz.tolist(),
        "new_quat_wxyz": new_wxyz.tolist(),
        "source_mount_translation_m": [float(value) for value in body.get("pos", "0 0 0").split()],
        "source_geometry_modified": False,
        "joint_limits_modified": False,
        "collision_filters_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-xml", required=True, type=Path,
                        help="Preserved SIMULATION_ONLY baseline with 0.001 rad activation margin")
    parser.add_argument("--platform-module", required=True, type=Path)
    parser.add_argument("--cycle-runner", required=True, type=Path)
    parser.add_argument("--controller-helper", required=True, type=Path)
    parser.add_argument("--tool-urdf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--diagnostic-margin-rad", type=float, default=0.001)
    parser.add_argument("--run-dynamics", action="store_true")
    args = parser.parse_args()
    if args.diagnostic_margin_rad != 0.001:
        raise ValueError("This recovery runner reuses only the previously authorized 0.001 rad margin")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    prior = load_module(args.controller_helper.resolve(), "issue46_x2_robotiq_m0")
    platform = load_module(args.platform_module.resolve(), "issue46_x2_dual_robotiq_platform")
    runner = load_module(args.cycle_runner.resolve(), "issue46_x2_robotiq_coupler_recovery")
    baseline = args.baseline_xml.resolve()
    candidate = out / "candidate_b_right_local_y_halfturn_180_simulation_only.xml"
    transformation = write_halfturn_candidate(baseline, candidate)

    baseline_model = mujoco.MjModel.from_xml_path(str(baseline))
    baseline_data, _ = runner.prepare_dual_data(baseline_model, platform, prior)
    candidate_model = mujoco.MjModel.from_xml_path(str(candidate))
    unchanged_model_properties = {
        "joint_ranges": bool(np.array_equal(baseline_model.jnt_range, candidate_model.jnt_range)),
        "actuator_force_ranges": bool(np.array_equal(
            baseline_model.actuator_forcerange, candidate_model.actuator_forcerange)),
        "geom_contype": bool(np.array_equal(baseline_model.geom_contype, candidate_model.geom_contype)),
        "geom_conaffinity": bool(np.array_equal(
            baseline_model.geom_conaffinity, candidate_model.geom_conaffinity)),
        "body_masses": bool(np.array_equal(baseline_model.body_mass, candidate_model.body_mass)),
        "body_inertias": bool(np.array_equal(baseline_model.body_inertia, candidate_model.body_inertia)),
    }
    if not all(unchanged_model_properties.values()):
        raise RuntimeError("Candidate B changed a source range, actuator bound, collision filter, or inertia")
    candidate_data, _ = runner.prepare_dual_data(candidate_model, platform, prior)
    tool_urdf = args.tool_urdf.resolve()
    wrist_mesh_hashes = {
        side: {
            role: {
                "file": filename,
                "sha256": sha256(tool_urdf.parent / "meshes" / filename),
            }
            for role, filename in files.items()
        }
        for side, files in platform.WRIST_MESH.items()
    }
    baseline_geom = pair_distances(baseline_model, baseline_data,
                                   "right_wrist_roll_link", "rq_right_")
    candidate_geom = pair_distances(candidate_model, candidate_data,
                                    "right_wrist_roll_link", "rq_right_")

    result: dict[str, Any] = {
        "classification": "SIMULATION_ONLY adapter candidate; not source-faithful hardware validation",
        "candidate_a": {
            "name": "pinned X2 OmniPicker wrist mount with Robotiq adapter basis",
            "model_path": str(baseline),
            "model_sha256": sha256(baseline),
            "closest_initial_right_wrist_to_gripper_pairs": baseline_geom[:12],
            "initial_contacts": summarize_contacts(baseline_model, baseline_data),
        },
        "candidate_b": {
            "name": "same mount position plus 180 degree local-Y adapter half-turn",
            "transformation": transformation,
            "model_path": str(candidate),
            "model_sha256": sha256(candidate),
            "closest_initial_right_wrist_to_gripper_pairs": candidate_geom[:12],
            "initial_contacts": summarize_contacts(candidate_model, candidate_data),
        },
        "candidate_invariants": unchanged_model_properties,
        "official_x2_mounts": platform.official_mounts(tool_urdf),
        "source_wrist_mesh_hashes": wrist_mesh_hashes,
        "compiled_tool_axes_at_open": {
            side: compiled_hand_axes(baseline_model, baseline_data, side)
            for side in ("left", "right")
        },
        "source_identity": {
            "x2_commit": prior.X2_PIN,
            "menagerie_commit": prior.MENAGERIE_PIN,
            "tool_urdf_sha256": sha256(args.tool_urdf.resolve()),
            "platform_module_sha256": sha256(args.platform_module.resolve()),
            "cycle_runner_sha256": sha256(args.cycle_runner.resolve()),
            "controller_helper_sha256": sha256(args.controller_helper.resolve()),
            "python": sys.version,
            "mujoco": mujoco.__version__,
        },
        "dynamic_cycles": {},
    }

    if args.run_dynamics:
        source_limits = platform.source_joint_limits(args.tool_urdf.resolve())
        cycle_out = out / "candidate_b"
        cycle_out.mkdir(parents=True, exist_ok=True)
        right = runner.run_cycle(candidate, "right", args.diagnostic_margin_rad,
                                 cycle_out, platform, prior, source_limits, render=True)
        result["dynamic_cycles"]["right"] = right
        if right["status"] == "PASS":
            for mode in ("left", "both"):
                result["dynamic_cycles"][mode] = runner.run_cycle(
                    candidate, mode, args.diagnostic_margin_rad, cycle_out,
                    platform, prior, source_limits, render=True)

    result["decision"] = (
        "DYNAMIC_CAPTURE_COMPLETE_FULL_CONTACT_AUDIT_REQUIRED"
        if result["dynamic_cycles"].get("right", {}).get("status") == "PASS"
        else "STOP_RIGHT_MOUNT_CANDIDATE_FAILED"
        if args.run_dynamics else "STATIC_COMPARISON_ONLY")
    result_path = out / "mount_recovery_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(result_path),
        "decision": result["decision"],
        "candidate_a_min_initial_distance_m": baseline_geom[0]["distance_m"] if baseline_geom else None,
        "candidate_b_min_initial_distance_m": candidate_geom[0]["distance_m"] if candidate_geom else None,
        "dynamic_cycles": {key: value["status"] for key, value in result["dynamic_cycles"].items()},
    }, sort_keys=True))
    return 0 if result["decision"] in (
        "DYNAMIC_CAPTURE_COMPLETE_FULL_CONTACT_AUDIT_REQUIRED", "STATIC_COMPARISON_ONLY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
