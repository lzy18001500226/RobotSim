#!/usr/bin/env python3
"""Screen rigid Robotiq wrist translations against a retained jaw-motion envelope."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def objname(model, kind, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{int(ident)}"


def objid(model, kind, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise RuntimeError(f"Missing {kind.name} {name}")
    return value


def trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if len(rows) != 1700:
        raise RuntimeError(f"Expected the retained 1700-step left control trace, got {len(rows)}")
    return rows


def initialize(model, runner, platform, helper):
    data, _ = runner.prepare_dual_data(model, platform, helper)
    return data


def apply_reference_right_pose(model, data, helper, row):
    # The two namespaced Menagerie grippers share the same source kinematics.
    # This is a static FK envelope reconstruction, not a physics rollout.
    for name, value in row["qpos_rad"].items():
        if name.startswith("rq_left_"):
            right_name = "rq_right_" + name[len("rq_left_"):]
            data.qpos[helper.qpos_id(model, right_name)] = float(value)


def body_geom_ids(model, predicate) -> list[int]:
    result = []
    for geom in range(model.ngeom):
        body = objname(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        if predicate(body) and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            result.append(geom)
    return result


def min_clearance(model, data, gripper_geoms, robot_geoms) -> dict[str, Any]:
    best = {"distance_m": float("inf")}
    for robot_geom in robot_geoms:
        for grip_geom in gripper_geoms:
            robot_body = int(model.geom_bodyid[robot_geom])
            grip_body = int(model.geom_bodyid[grip_geom])
            # MuJoCo filters collision inside one welded body and immediate
            # parent-child interfaces. They are mounting interfaces, not free
            # collision gaps; the stepped contact trace remains the final gate.
            if model.body_weldid[robot_body] == model.body_weldid[grip_body]:
                continue
            if int(model.body_parentid[robot_body]) == grip_body or int(model.body_parentid[grip_body]) == robot_body:
                continue
            segment = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(model, data, robot_geom, grip_geom, 0.5, segment))
            if distance < best["distance_m"]:
                best = {
                    "distance_m": distance,
                    "robot_body": objname(model, mujoco.mjtObj.mjOBJ_BODY, robot_body),
                    "robot_geom": objname(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom),
                    "gripper_body": objname(model, mujoco.mjtObj.mjOBJ_BODY, grip_body),
                    "gripper_geom": objname(model, mujoco.mjtObj.mjOBJ_GEOM, grip_geom),
                    "witness_segment_m": segment.tolist(),
                }
    return best


def envelope_at(model, data, helper, rows, base_pos, axis_parent, offset_m,
                sample_stride=10) -> dict[str, Any]:
    root = objid(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_base_mount")
    model.body_pos[root] = base_pos + axis_parent * offset_m
    gripper_geoms = body_geom_ids(model, lambda name: name.startswith("rq_right_")
                                  and name not in {"rq_right_base_mount", "rq_right_base"})
    robot_geoms = body_geom_ids(model, lambda name: not name.startswith("rq_right_") and
                                (name.endswith("_link") or name in
                                 {"pelvis", "torso_link", "head_pitch_link"}))
    selected = list(range(0, len(rows), sample_stride))
    if selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)
    worst = {"distance_m": float("inf")}
    samples = []
    for idx in selected:
        row = rows[idx]
        apply_reference_right_pose(model, data, helper, row)
        mujoco.mj_forward(model, data)
        nearest = min_clearance(model, data, gripper_geoms, robot_geoms)
        nearest.update({"step": int(row["step"]), "phase": row["phase"], "time_s": row["time_s"]})
        samples.append(nearest)
        if nearest["distance_m"] < worst["distance_m"]:
            worst = nearest
    return {"offset_m": offset_m, "sample_count": len(selected), "minimum": worst,
            "samples": samples}


def worst_envelope_normal(model, data, helper, rows, base_pos) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = envelope_at(model, data, helper, rows, base_pos, np.zeros(3), 0.0, 1)
    minimum = baseline["minimum"]
    row = next(item for item in rows if int(item["step"]) == minimum["step"])
    apply_reference_right_pose(model, data, helper, row)
    mujoco.mj_forward(model, data)
    robot_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                       minimum["robot_geom"]))
    if robot_geom < 0 and minimum["robot_geom"].startswith("mjobj_geom_"):
        robot_geom = int(minimum["robot_geom"][len("mjobj_geom_"):])
    if robot_geom < 0:
        raise RuntimeError(f"Cannot resolve limiting geometry {minimum['robot_geom']}")
    grip_geom = objid(model, mujoco.mjtObj.mjOBJ_GEOM, minimum["gripper_geom"])
    root = objid(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_base_mount")
    wrist = objid(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    wrist_rotation = data.xmat[wrist].reshape(3, 3)
    segment = np.asarray(minimum["witness_segment_m"], dtype=float)
    world_delta = segment[3:] - segment[:3]
    direction_parent = wrist_rotation.T @ world_delta
    norm = float(np.linalg.norm(direction_parent))
    if norm < 1e-9:
        raise RuntimeError("Worst-envelope witness segment has no usable direction")
    direction_parent /= norm
    saved_pos = model.body_pos[root].copy()
    probe = []
    for sign in (-1.0, 1.0):
        model.body_pos[root] = saved_pos + sign * 0.001 * direction_parent
        mujoco.mj_forward(model, data)
        segment_probe = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, robot_geom, grip_geom,
                                                0.5, segment_probe))
        probe.append(distance)
    model.body_pos[root] = saved_pos
    mujoco.mj_forward(model, data)
    if probe[1] < probe[0]:
        direction_parent *= -1.0
        probe.reverse()
    evidence = {
        "method": "full-cycle minimum pair witness segment, finite-difference sign checked",
        "reference_step": minimum["step"],
        "reference_phase": minimum["phase"],
        "reference_pair": [minimum["robot_geom"], minimum["gripper_geom"]],
        "reference_distance_m": minimum["distance_m"],
        "witness_segment_m": segment.tolist(),
        "direction_parent_unit": direction_parent.tolist(),
        "clearance_at_minus_1mm_m": probe[0],
        "clearance_at_plus_1mm_m": probe[1],
    }
    return direction_parent, evidence


def solve_collision_clear_translation(model, data, helper, rows, base_pos,
                                       max_offset_m: float = 0.05,
                                       required_clearance_m: float = 0.003) -> dict[str, Any]:
    """Find the minimum added mount translation satisfying critical full-cycle frames."""
    root = objid(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_base_mount")
    grip_geoms = body_geom_ids(model, lambda name: name.startswith("rq_right_")
                               and name not in {"rq_right_base_mount", "rq_right_base"})
    robot_geoms = body_geom_ids(model, lambda name: not name.startswith("rq_right_") and
                                (name.endswith("_link") or name in
                                 {"pelvis", "torso_link", "head_pitch_link"}))
    row_by_step = {int(row["step"]): row for row in rows}
    critical_steps = [1, 11, 277, 292, 316, 318, 320, 583, 584, 781, 961, 1461, 1699]
    critical_steps = [step for step in critical_steps if step in row_by_step]

    def frame_minimum(offset_m: np.ndarray, step: int) -> dict[str, Any]:
        model.body_pos[root] = base_pos + offset_m
        apply_reference_right_pose(model, data, helper, row_by_step[step])
        mujoco.mj_forward(model, data)
        result = min_clearance(model, data, grip_geoms, robot_geoms)
        result["step"] = step
        result["phase"] = row_by_step[step]["phase"]
        return result

    def constraints(offset_mm: np.ndarray) -> np.ndarray:
        offset_m = np.asarray(offset_mm, dtype=float) / 1000.0
        return np.asarray([
            (frame_minimum(offset_m, step)["distance_m"] - required_clearance_m) * 1000.0
            for step in critical_steps
        ])

    result = minimize(
        lambda value: 0.5 * float(value @ value),
        np.zeros(3), method="SLSQP",
        bounds=[(-max_offset_m * 1000.0, max_offset_m * 1000.0)] * 3,
        constraints=[
            {"type": "ineq", "fun": constraints},
            {"type": "ineq", "fun": lambda value:
             (max_offset_m * 1000.0) ** 2 - float(value @ value)},
        ],
        options={"maxiter": 80, "ftol": 1e-7, "disp": False},
    )
    offset_m = np.asarray(result.x, dtype=float) / 1000.0
    critical = [frame_minimum(offset_m, step) for step in critical_steps]
    return {
        "solver": "SLSQP minimum-norm translation with signed MuJoCo geom-distance constraints",
        "critical_steps": critical_steps,
        "required_clearance_m": required_clearance_m,
        "translation_radius_cap_m": max_offset_m,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "added_translation_parent_frame_m": offset_m.tolist(),
        "added_translation_norm_m": float(np.linalg.norm(offset_m)),
        "critical_frame_minima": critical,
        "all_critical_clear": bool(all(
            item["distance_m"] >= required_clearance_m for item in critical
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-xml", required=True, type=Path)
    parser.add_argument("--reference-left-trace", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--platform", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-translation-mm", type=float, default=50.0)
    parser.add_argument("--direction-source", choices=("insertion", "saved-contact-normal",
                                                        "worst-envelope-normal",
                                                        "collision-clear-solve"),
                        default="insertion")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    helper = load_module(args.helper.resolve(), "issue46_x2_robotiq_m0")
    platform = load_module(args.platform.resolve(), "issue46_x2_dual_robotiq_platform")
    runner = load_module(args.runner.resolve(), "issue46_x2_robotiq_coupler_recovery")
    rows = trace_rows(args.reference_left_trace.resolve())
    model = mujoco.MjModel.from_xml_path(str(args.baseline_xml.resolve()))
    data = initialize(model, runner, platform, helper)
    root_id = objid(model, mujoco.mjtObj.mjOBJ_BODY, "rq_right_base_mount")
    base_pos = model.body_pos[root_id].copy()
    quat = model.body_quat[root_id].copy()
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    if args.direction_source == "insertion":
        axis_parent = rotation.apply([0, 0, 1])
        direction_evidence = {"source": "compiled source mount local +Z insertion axis"}
    elif args.direction_source == "saved-contact-normal":
        # Source-mount first-contact normal from the deterministic replay at
        # step 584; normal is from wrist (geom1) toward follower (geom2).
        normal_world = np.asarray([-0.029113241, 0.807202119, -0.589556748])
        wrist_id = objid(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
        parent_world_rotation = data.xmat[wrist_id].reshape(3, 3)
        axis_parent = parent_world_rotation.T @ normal_world
        direction_evidence = {"source": "preserved contact-normal replay, candidate A step 584",
                              "normal_world": normal_world.tolist(),
                              "normal_direction": "geom1 right_wrist_roll_link to geom2 right follower",
                              "parent_frame_direction": axis_parent.tolist()}
    elif args.direction_source == "worst-envelope-normal":
        axis_parent, direction_evidence = worst_envelope_normal(
            model, data, helper, rows, base_pos)
    else:
        solution = solve_collision_clear_translation(
            model, data, helper, rows, base_pos,
            max_offset_m=args.max_translation_mm / 1000.0)
        if not solution["all_critical_clear"]:
            result = {"status": "CRITICAL_FRAME_CLEARANCE_REJECTED",
                      "source_classification": "no adapter built; geometry-only diagnostic",
                      "baseline_xml_sha256": sha256(args.baseline_xml.resolve()),
                      "reference_trace_sha256": sha256(args.reference_left_trace.resolve()),
                      "runtime": {"mujoco_python": mujoco.__version__,
                                  "mujoco_native": mujoco.mj_versionString(),
                                  "python": sys.version.split()[0]},
                      "collision_clear_translation": solution}
            (out / "collision_clear_translation.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": result["status"],
                              "solver_success": solution["success"],
                              "offset_m": solution["added_translation_parent_frame_m"],
                              "worst_critical": min(solution["critical_frame_minima"],
                                                    key=lambda item: item["distance_m"])},
                             sort_keys=True))
            return 2
        offset_parent = np.asarray(solution["added_translation_parent_frame_m"], dtype=float)
        original_full = envelope_at(model, data, helper, rows, base_pos,
                                    offset_parent, 1.0, 1)
        if original_full["minimum"]["distance_m"] < 0.003:
            result = {"status": "FULL_SWEEP_CLEARANCE_REJECTED",
                      "source_classification": "no adapter built; geometry-only diagnostic",
                      "baseline_xml_sha256": sha256(args.baseline_xml.resolve()),
                      "reference_trace_sha256": sha256(args.reference_left_trace.resolve()),
                      "runtime": {"mujoco_python": mujoco.__version__,
                                  "mujoco_native": mujoco.mj_versionString(),
                                  "python": sys.version.split()[0]},
                      "collision_clear_translation": solution,
                      "full_1700_step_original_gripper_envelope": original_full}
            (out / "collision_clear_translation.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"status": result["status"],
                              "offset_m": offset_parent.tolist(),
                              "full_sweep_minimum": original_full["minimum"]}, sort_keys=True))
            return 2

        result = {"status": "TRANSLATION_SCREEN_CLEAR_BRACKET_UNVALIDATED",
                  "source_classification": "geometry-only translation screen; no adapter built",
                  "baseline_xml_sha256": sha256(args.baseline_xml.resolve()),
                  "reference_trace_sha256": sha256(args.reference_left_trace.resolve()),
                  "runtime": {"mujoco_python": mujoco.__version__,
                              "mujoco_native": mujoco.mj_versionString(),
                              "python": sys.version.split()[0]},
                  "collision_clear_translation": solution,
                  "full_1700_step_original_gripper_envelope": original_full,
                  "bracket_inclusive_envelope": "NOT RUN: no bracket geometry was defined or built"}
        (out / "collision_clear_translation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"],
                          "offset_m": offset_parent.tolist(),
                          "full_sweep_minimum": original_full["minimum"],
                          "bracket": "not built"}, sort_keys=True))
        return 2
    axis_parent = axis_parent / np.linalg.norm(axis_parent)

    # Record a bounded scalar translation profile; never infer a bracket from it.
    max_offset = 0.10
    bracket = {"low_m": 0.0, "high_m": max_offset, "required_clearance_m": 0.003,
               "geometry_sample_stride_steps": 10}
    low = envelope_at(model, data, helper, rows, base_pos, axis_parent, 0.0, 10)
    offset_profile = [low]
    for offset in (0.025, 0.050, 0.075, max_offset):
        offset_profile.append(envelope_at(model, data, helper, rows, base_pos,
                                          axis_parent, offset, 10))
    bracket["low_minimum"] = low["minimum"]
    bracket["offset_profile"] = [
        {"offset_m": item["offset_m"], "minimum": item["minimum"]}
        for item in offset_profile
    ]
    result = {"status": "AXIAL_TRANSLATION_SCREEN_REJECTED",
              "source_classification": "no bracket built; geometry-only diagnostic",
              "baseline_xml_sha256": sha256(args.baseline_xml.resolve()),
              "reference_trace_sha256": sha256(args.reference_left_trace.resolve()),
              "runtime": {"mujoco_python": mujoco.__version__,
                          "mujoco_native": mujoco.mj_versionString(),
                          "python": sys.version.split()[0]},
              "offset_profile": bracket["offset_profile"],
              "direction_evidence": direction_evidence,
              "required_clearance_m": bracket["required_clearance_m"]}
    (out / "axis_direction_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "profile": bracket["offset_profile"],
                      "direction_evidence": direction_evidence}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
