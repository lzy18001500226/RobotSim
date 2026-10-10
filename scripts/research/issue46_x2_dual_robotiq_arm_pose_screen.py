#!/usr/bin/env python3
"""Bounded source-limit arm-pose screen for the Issue #46 dual-Robotiq assembly.

This reconstructs the retained actuator-driven left jaw trace as right-side
joint configurations and evaluates compiled collision geometry with MuJoCo FK.
It does not step physics and cannot qualify an adapter mechanically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
CANONICAL_BOTTLE_BODY_CENTER = np.array([0.300, 0.000, 0.9175], dtype=float)
POSES = {
    "current_neutral": np.array([0.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0]),
    "forward_outboard_flex": np.array([-0.55, -0.35, 0.0, -0.80, 0.0, 0.0, 0.0]),
    "elevated_outboard_flex": np.array([-0.30, -0.75, 0.0, -1.15, 0.0, 0.20, 0.0]),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def name(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def ident(model, kind, label: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, label))
    if value < 0:
        raise RuntimeError(f"Missing {kind.name}: {label}")
    return value


def collidable_geoms(model, predicate) -> list[int]:
    return [g for g in range(model.ngeom)
            if predicate(name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])))
            and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]


def pair_allowed(model, geom_a: int, geom_b: int) -> bool:
    contype_a, conaffinity_a = int(model.geom_contype[geom_a]), int(model.geom_conaffinity[geom_a])
    contype_b, conaffinity_b = int(model.geom_contype[geom_b]), int(model.geom_conaffinity[geom_b])
    if not ((contype_a & conaffinity_b) or (contype_b & conaffinity_a)):
        return False
    body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
    if int(model.body_weldid[body_a]) == int(model.body_weldid[body_b]):
        return False
    if int(model.body_parentid[body_a]) == body_b or int(model.body_parentid[body_b]) == body_a:
        return False
    return True


def group_for_body(body: str) -> str:
    if body == "right_wrist_roll_link":
        return "right_wrist"
    if body.startswith("right_") and body.endswith("_link"):
        return "right_arm"
    if body.startswith("left_") and body.endswith("_link"):
        return "left_side"
    if body.startswith("rq_left_"):
        return "left_gripper"
    return "other_robot"


def collision_summary(model, data, right_geom_ids: list[int], other_geom_ids: list[int]) -> dict[str, Any]:
    minima = {key: {"distance_m": float("inf")} for key in
              ("right_wrist", "right_arm", "left_side", "left_gripper", "other_robot", "all")}
    for grip in right_geom_ids:
        grip_body = int(model.geom_bodyid[grip])
        for other in other_geom_ids:
            other_body = int(model.geom_bodyid[other])
            if not pair_allowed(model, grip, other):
                continue
            segment = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(model, data, grip, other, 2.0, segment))
            body_name = name(model, mujoco.mjtObj.mjOBJ_BODY, other_body)
            group = group_for_body(body_name)
            row = {
                "distance_m": distance,
                "gripper_body": name(model, mujoco.mjtObj.mjOBJ_BODY, grip_body),
                "gripper_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, grip),
                "other_body": body_name,
                "other_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                "witness_segment_world_m": segment.tolist(),
            }
            if distance < minima[group]["distance_m"]:
                minima[group] = row
            if distance < minima["all"]["distance_m"]:
                minima["all"] = row
    return minima


def contact_rows(model, data, right_geom_ids: set[int]) -> list[dict[str, Any]]:
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in right_geom_ids and g2 not in right_geom_ids:
            continue
        other = g2 if g1 in right_geom_ids else g1
        other_body = int(model.geom_bodyid[other])
        grip_body = int(model.geom_bodyid[g1 if g1 in right_geom_ids else g2])
        other_body_name = name(model, mujoco.mjtObj.mjOBJ_BODY, other_body)
        if other_body_name.startswith("rq_right_"):
            continue
        contacts.append({
            "geom_pair": sorted([name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                                 name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)]),
            "body_pair": sorted([name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
                                 name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2]))]),
            "distance_m": float(c.dist),
            "right_gripper_body": name(model, mujoco.mjtObj.mjOBJ_BODY, grip_body),
            "other_body": other_body_name,
        })
    return contacts


def source_limit_check(model, data, helper) -> dict[str, Any]:
    rows = []
    for joint in ARM:
        jid = ident(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        q = float(data.qpos[helper.qpos_id(model, joint)])
        lo, hi = map(float, model.jnt_range[jid])
        rows.append({"joint": joint, "qpos_rad": q, "source_range_rad": [lo, hi],
                     "lower_margin_rad": q - lo, "upper_margin_rad": hi - q,
                     "inside_source_range": lo - 1e-9 <= q <= hi + 1e-9})
    return {"all_inside": all(r["inside_source_range"] for r in rows), "joints": rows}


def pad_metrics(model, data, helper) -> dict[str, Any]:
    pad_names = ["rq_right_right_pad1", "rq_right_right_pad2",
                 "rq_right_left_pad1", "rq_right_left_pad2"]
    pad_ids = [ident(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in pad_names]
    midpoint = np.mean(data.geom_xpos[pad_ids], axis=0)
    target = CANONICAL_BOTTLE_BODY_CENTER.copy()
    target[2] = 0.9175 - 0.040  # center of the canonical cylindrical body segment
    base = ident(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_base_mount")
    wrist = ident(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    return {
        "compiled_pad_midpoint_world_m": midpoint.tolist(),
        "canonical_bottle_cylinder_center_reference_m": target.tolist(),
        "pad_midpoint_to_canonical_cylinder_center_distance_m": float(np.linalg.norm(midpoint - target)),
        "right_mount_world_m": data.xpos[base].tolist(),
        "right_wrist_world_m": data.xpos[wrist].tolist(),
        "warning": "FK point distance only; not an IK, approach, or collision-clearance pass",
    }


def render(model, data, path: Path, lookat: np.ndarray, distance: float, azimuth: float, elevation: float):
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def run_pose(label: str, arm_q: np.ndarray, args, helper, platform, runner,
             trace: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(args.baseline_xml.resolve()))
    data, _ = runner.prepare_dual_data(model, platform, helper)
    qids = [helper.qpos_id(model, j) for j in ARM]
    data.qpos[qids] = arm_q
    mujoco.mj_forward(model, data)
    limits = source_limit_check(model, data, helper)
    if not limits["all_inside"]:
        raise RuntimeError(f"Candidate {label} is outside compiled source joint ranges")
    right_geoms = collidable_geoms(model, lambda b: b.startswith("rq_right_"))
    right_geom_set = set(right_geoms)
    other_geoms = collidable_geoms(model, lambda b: not b.startswith("rq_right_"))
    arm_names = ["right_shoulder_pitch_link", "right_shoulder_roll_link",
                 "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
                 "right_wrist_pitch_link", "right_wrist_roll_link"]
    existing_contacts = contact_rows(model, data, right_geom_set)
    pose_result = {
        "pose_id": label,
        "arm_joint_names": list(ARM),
        "arm_qpos_rad": arm_q.tolist(),
        "source_limit_check_at_open": limits,
        "initial_right_gripper_contacts": existing_contacts,
        "initial_self_contact_count_all_model": int(data.ncon),
        "initial_self_contact_pairs_all_model": [
            {"geom_pair": sorted([name(model, mujoco.mjtObj.mjOBJ_GEOM, int(data.contact[i].geom1)),
                                  name(model, mujoco.mjtObj.mjOBJ_GEOM, int(data.contact[i].geom2))]),
             "distance_m": float(data.contact[i].dist)} for i in range(data.ncon)
        ],
        "cycle_geometry_samples": 0,
        "cycle_minima": {},
        "cycle_contact_pair_counts": {},
        "cycle_pass": False,
    }
    pose_result["workspace_fk_reference"] = pad_metrics(model, data, helper)
    if not pose_result["initial_right_gripper_contacts"]:
        pose_result["initial_right_gripper_open_clear"] = True
    else:
        pose_result["initial_right_gripper_open_clear"] = False

    trace_path = out / "raw" / f"{label}_jaw_sweep.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pair_counts: Counter[str] = Counter()
    minima = {g: {"distance_m": float("inf")} for g in
              ("right_wrist", "right_arm", "left_side", "left_gripper", "other_robot", "all")}
    first_penetration = None
    with trace_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(trace):
            for joint, value in row["qpos_rad"].items():
                if joint.startswith("rq_left_"):
                    right_joint = "rq_right_" + joint[len("rq_left_"):]
                    data.qpos[helper.qpos_id(model, right_joint)] = float(value)
            data.qpos[qids] = arm_q
            mujoco.mj_forward(model, data)
            distances = collision_summary(model, data, right_geoms, other_geoms)
            contacts = contact_rows(model, data, right_geom_set)
            for contact in contacts:
                pair_counts[" <-> ".join(contact["body_pair"])] += 1
                if contact["distance_m"] < -1e-8 and first_penetration is None:
                    first_penetration = {"step": int(row["step"]), "phase": row["phase"], **contact}
            for group, item in distances.items():
                if item["distance_m"] < minima[group]["distance_m"]:
                    minima[group] = {**item, "step": int(row["step"]), "phase": row["phase"]}
            outrow = {
                "trace_index": index,
                "step": int(row["step"]),
                "time_s": float(row["time_s"]),
                "phase": row["phase"],
                "right_arm_qpos_rad": arm_q.tolist(),
                "right_pad_gap_m": float(platform.mounted_pad_gap(model, data, "right")),
                "collision_distance_minima": distances,
                "right_gripper_contacts": contacts,
            }
            stream.write(json.dumps(outrow, separators=(",", ":")) + "\n")
            pose_result["cycle_geometry_samples"] += 1
    pose_result["cycle_minima"] = minima
    pose_result["cycle_contact_pair_counts"] = dict(sorted(pair_counts.items()))
    pose_result["first_nonmount_penetration"] = first_penetration
    pose_result["cycle_pass"] = all(v["distance_m"] >= 0.0 for v in minima.values()) and not pair_counts
    pose_result["trace_path"] = str(trace_path)
    pose_result["trace_sha256"] = sha256(trace_path)

    # Render the same open configuration used to initialize the sweep.
    for joint, value in trace[0]["qpos_rad"].items():
        if joint.startswith("rq_left_"):
            data.qpos[helper.qpos_id(model, "rq_right_" + joint[len("rq_left_"):])] = float(value)
    data.qpos[qids] = arm_q
    mujoco.mj_forward(model, data)
    pelvis = ident(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    wrist = ident(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    render(model, data, out / "images" / f"{label}_overview.png",
           data.xpos[pelvis].copy() + np.array([0.0, 0.18, 0.15]), 2.2, 135.0, -8.0)
    render(model, data, out / "images" / f"{label}_right_wrist_closeup.png",
           data.xpos[wrist].copy(), 0.45, 135.0, -5.0)
    pose_result["images"] = {
        "overview": str(out / "images" / f"{label}_overview.png"),
        "wrist_closeup": str(out / "images" / f"{label}_right_wrist_closeup.png"),
    }
    pose_result["image_sha256"] = {k: sha256(Path(v)) for k, v in pose_result["images"].items()}
    return pose_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-xml", required=True, type=Path)
    parser.add_argument("--reference-left-trace", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--platform", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    helper = load_module(args.helper.resolve(), "issue46_x2_robotiq_m0")
    platform = load_module(args.platform.resolve(), "issue46_x2_dual_robotiq_platform")
    runner = load_module(args.runner.resolve(), "issue46_x2_robotiq_coupler_recovery")
    trace = [json.loads(line) for line in args.reference_left_trace.open(encoding="utf-8")]
    if len(trace) != 1700:
        raise RuntimeError(f"Expected the exact retained 1700-row left cycle trace; got {len(trace)}")
    model = mujoco.MjModel.from_xml_path(str(args.baseline_xml.resolve()))
    if abs(float(model.opt.timestep) - 0.001) > 1e-12:
        raise RuntimeError(f"Unexpected timestep {model.opt.timestep}")
    pose_results = []
    for label, q in POSES.items():
        pose_results.append(run_pose(label, q, args, helper, platform, runner, trace, out))

    wrist_minima = [p["cycle_minima"]["right_wrist"] for p in pose_results]
    invariant = max(float(x["distance_m"]) for x in wrist_minima) - min(
        float(x["distance_m"]) for x in wrist_minima)
    result = {
        "issue": 46,
        "task": "right-arm work-pose feasibility before adapter redesign",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_ASSEMBLY",
        "method": "three named source-range arm poses; full retained jaw-cycle FK reconstruction; compiled collision geometry and active contact scan; no physics stepping",
        "runtime": {"mujoco_python": mujoco.__version__, "mujoco_native": mujoco.mj_versionString(),
                    "model_timestep_s": float(model.opt.timestep), "environment_muJoCo_GL": os.environ.get("MUJOCO_GL")},
        "inputs": {"baseline_xml": str(args.baseline_xml.resolve()),
                   "baseline_xml_sha256": sha256(args.baseline_xml.resolve()),
                   "retained_trace": str(args.reference_left_trace.resolve()),
                   "retained_trace_sha256": sha256(args.reference_left_trace.resolve()),
                   "helper": str(args.helper.resolve()), "helper_sha256": sha256(args.helper.resolve()),
                   "platform": str(args.platform.resolve()), "platform_sha256": sha256(args.platform.resolve()),
                   "runner": str(args.runner.resolve()), "runner_sha256": sha256(args.runner.resolve())},
        "arm_joint_order": list(ARM),
        "candidate_count": len(pose_results),
        "poses": pose_results,
        "right_wrist_to_gripper_distance_spread_across_arm_poses_m": invariant,
        "wrist_mount_relative_collision_pose_invariant_to_arm_pose": invariant < 1e-7,
        "gate": {
            "source_limit_pass": all(p["source_limit_check_at_open"]["all_inside"] for p in pose_results),
            "full_open_close_reopen_clearance_pass": all(p["cycle_pass"] for p in pose_results),
            "right_wrist_to_gripper_rigid_interface_clearance_pass": all(
                p["cycle_minima"]["right_wrist"]["distance_m"] >= 0.0 for p in pose_results),
            "mounted_dynamic_cycles_eligible": all(p["cycle_pass"] for p in pose_results),
            "collision_aware_bottle_ik_status": "NOT RUN: full-cycle static clearance prerequisite failed if any pose is blocked",
            "physical_bottle_test_status": "NOT RUN",
        },
        "conclusion": "",
    }
    if all(p["cycle_pass"] for p in pose_results):
        result["conclusion"] = "A tested source-valid right-arm work pose clears the reconstructed full jaw sweep; proceed to dynamic validation."
    else:
        result["conclusion"] = "No tested source-valid arm pose clears the complete jaw sweep. If wrist clearance is invariant, arm repositioning cannot resolve the limiting rigid wrist/gripper interface; adapter geometry remains the next dependency. This bounded result is not a global infeasibility claim."
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "candidate_count": len(pose_results),
                      "wrist_distance_spread_m": invariant, "all_sweeps_clear": result["gate"]["full_open_close_reopen_clearance_pass"],
                      "conclusion": result["conclusion"]}, indent=2))
    return 0 if result["gate"]["full_open_close_reopen_clearance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
