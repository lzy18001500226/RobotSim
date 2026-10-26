#!/usr/bin/env python3
"""Exact source-STL head/torso surface audit over the planned cylinder path.

The tool intentionally uses only ``mj_forward`` for the accepted derived
scene.  It checks source triangle surfaces for each planned static state before
an explicitly authorized pair-specific proxy exception is ever constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
from agent import path_governance as pg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preflight_module(path: Path):
    spec = importlib.util.spec_from_file_location("full_x2_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounds(triangles: np.ndarray) -> np.ndarray:
    return np.concatenate([triangles.min(axis=1), triangles.max(axis=1)], axis=1)


def sat_intersects(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """17-axis triangle SAT for matching batches of world-space triangles."""
    first_edges = np.stack([first[:, 1] - first[:, 0], first[:, 2] - first[:, 1], first[:, 0] - first[:, 2]], axis=1)
    second_edges = np.stack([second[:, 1] - second[:, 0], second[:, 2] - second[:, 1], second[:, 0] - second[:, 2]], axis=1)
    first_normal = np.cross(first_edges[:, 0], first_edges[:, 1])
    second_normal = np.cross(second_edges[:, 0], second_edges[:, 1])
    axes = [first_normal, second_normal]
    axes.extend(np.cross(first_edges[:, i], second_edges[:, j]) for i in range(3) for j in range(3))
    axes.extend(np.cross(first_normal, first_edges[:, i]) for i in range(3))
    axes.extend(np.cross(second_normal, second_edges[:, i]) for i in range(3))
    intersects = np.ones(len(first), dtype=bool)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        usable = norm > 1e-12
        if not np.any(usable):
            continue
        left = np.einsum("nvc,nc->nv", first, axis)
        right = np.einsum("nvc,nc->nv", second, axis)
        separated = (left.max(axis=1) < right.min(axis=1) - tolerance) | (right.max(axis=1) < left.min(axis=1) - tolerance)
        intersects &= ~(usable & separated)
        if not np.any(intersects):
            break
    return intersects


def segment_distance(first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized exact segment distance and closest points."""
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    relative = first_start - second_start
    aa = np.einsum("ij,ij->i", first_vector, first_vector)
    bb = np.einsum("ij,ij->i", first_vector, second_vector)
    cc = np.einsum("ij,ij->i", second_vector, second_vector)
    dd = np.einsum("ij,ij->i", first_vector, relative)
    ee = np.einsum("ij,ij->i", second_vector, relative)
    determinant = aa * cc - bb * bb
    first_numerator = bb * ee - cc * dd
    first_denominator = determinant.copy()
    second_numerator = aa * ee - bb * dd
    second_denominator = determinant.copy()
    parallel = determinant < 1e-18
    first_numerator[parallel] = 0.0
    first_denominator[parallel] = 1.0
    second_numerator[parallel] = ee[parallel]
    second_denominator[parallel] = cc[parallel]
    lower = (~parallel) & (first_numerator < 0.0)
    first_numerator[lower] = 0.0
    second_numerator[lower] = ee[lower]
    second_denominator[lower] = cc[lower]
    upper = (~parallel) & (first_numerator > first_denominator)
    first_numerator[upper] = first_denominator[upper]
    second_numerator[upper] = ee[upper] + bb[upper]
    second_denominator[upper] = cc[upper]
    lower_second = second_numerator < 0.0
    second_numerator[lower_second] = 0.0
    first_numerator[lower_second & (-dd < 0.0)] = 0.0
    first_numerator[lower_second & (-dd > aa)] = first_denominator[lower_second & (-dd > aa)]
    middle = lower_second & ~((-dd < 0.0) | (-dd > aa))
    first_numerator[middle] = -dd[middle]
    first_denominator[middle] = aa[middle]
    upper_second = second_numerator > second_denominator
    second_numerator[upper_second] = second_denominator[upper_second]
    first_numerator[upper_second & (-dd + bb < 0.0)] = 0.0
    first_numerator[upper_second & (-dd + bb > aa)] = first_denominator[upper_second & (-dd + bb > aa)]
    middle = upper_second & ~((-dd + bb < 0.0) | (-dd + bb > aa))
    first_numerator[middle] = -dd[middle] + bb[middle]
    first_denominator[middle] = aa[middle]
    first_scale = np.divide(first_numerator, first_denominator, out=np.zeros_like(first_numerator), where=np.abs(first_denominator) > 1e-18)
    second_scale = np.divide(second_numerator, second_denominator, out=np.zeros_like(second_numerator), where=np.abs(second_denominator) > 1e-18)
    first_point = first_start + first_scale[:, None] * first_vector
    second_point = second_start + second_scale[:, None] * second_vector
    return np.linalg.norm(first_point - second_point, axis=1), first_point, second_point


def triangle_distance(first: np.ndarray, second: np.ndarray, triangles_module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact finite-triangle distance for matching batches after SAT rejection."""
    count = len(first)
    best = np.full(count, np.inf)
    first_witness = np.zeros((count, 3))
    second_witness = np.zeros((count, 3))
    for index in range(3):
        closest = triangles_module.closest_point(second, first[:, index, :])
        distance = np.linalg.norm(closest - first[:, index, :], axis=1)
        replace = distance < best
        best[replace] = distance[replace]
        first_witness[replace] = first[replace, index, :]
        second_witness[replace] = closest[replace]
        closest = triangles_module.closest_point(first, second[:, index, :])
        distance = np.linalg.norm(closest - second[:, index, :], axis=1)
        replace = distance < best
        best[replace] = distance[replace]
        first_witness[replace] = closest[replace]
        second_witness[replace] = second[replace, index, :]
    for first_edge in range(3):
        for second_edge in range(3):
            distance, first_point, second_point = segment_distance(
                first[:, first_edge, :], first[:, (first_edge + 1) % 3, :],
                second[:, second_edge, :], second[:, (second_edge + 1) % 3, :],
            )
            replace = distance < best
            best[replace] = distance[replace]
            first_witness[replace] = first_point[replace]
            second_witness[replace] = second_point[replace]
    return best, first_witness, second_witness


def exact_surface_clearance(first: np.ndarray, second: np.ndarray, rtree_index, triangles_module, upper_bound_m: float = 0.006) -> dict[str, object]:
    """Use a triangle AABB R-tree, SAT, and exact finite-triangle distances."""
    second_bounds = bounds(second)
    properties = rtree_index.Property()
    properties.dimension = 3
    tree = rtree_index.Index(((index, tuple(value), None) for index, value in enumerate(second_bounds)), properties=properties)
    first_bounds = bounds(first)

    def pairs(expand: float):
        for first_index, value in enumerate(first_bounds):
            query = (value[0] - expand, value[1] - expand, value[2] - expand, value[3] + expand, value[4] + expand, value[5] + expand)
            for second_index in tree.intersection(query):
                yield first_index, int(second_index)

    overlap_pairs = []
    batch = []
    for pair in pairs(0.0):
        batch.append(pair)
        if len(batch) >= 20000:
            indices = np.asarray(batch, dtype=np.int32)
            hits = sat_intersects(first[indices[:, 0]], second[indices[:, 1]])
            overlap_pairs.extend(indices[hits].tolist())
            batch = []
    if batch:
        indices = np.asarray(batch, dtype=np.int32)
        hits = sat_intersects(first[indices[:, 0]], second[indices[:, 1]])
        overlap_pairs.extend(indices[hits].tolist())
    if overlap_pairs:
        pair = overlap_pairs[0]
        return {
            "surface_intersection": True,
            "minimum_surface_distance_m": 0.0,
            "intersection_pair_count": len(overlap_pairs),
            "witness": {
                "head_triangle_index": int(pair[0]), "torso_triangle_index": int(pair[1]),
                "head_point_world_m": first[pair[0], 0].tolist(), "torso_point_world_m": second[pair[1], 0].tolist(),
            },
            "candidate_pairs_evaluated": 0,
        }

    best = upper_bound_m
    witness = None
    candidate_pairs = 0
    evaluated = 0
    batch = []

    def process(items: list[tuple[int, int]]) -> None:
        nonlocal best, witness, evaluated
        indices = np.asarray(items, dtype=np.int32)
        distance, first_points, second_points = triangle_distance(first[indices[:, 0]], second[indices[:, 1]], triangles_module)
        evaluated += len(indices)
        closest = int(np.argmin(distance))
        if float(distance[closest]) < best:
            best = float(distance[closest])
            witness = {
                "head_triangle_index": int(indices[closest, 0]),
                "torso_triangle_index": int(indices[closest, 1]),
                "head_point_world_m": first_points[closest].tolist(),
                "torso_point_world_m": second_points[closest].tolist(),
            }

    for pair in pairs(upper_bound_m):
        candidate_pairs += 1
        batch.append(pair)
        if len(batch) >= 10000:
            process(batch)
            batch = []
    if batch:
        process(batch)
    return {
        "surface_intersection": False,
        "minimum_surface_distance_m": best,
        "minimum_is_exact_within_upper_bound": witness is not None,
        "intersection_pair_count": 0,
        "witness": witness,
        "candidate_pairs_evaluated": evaluated,
        "candidate_pairs_with_aabb_gap_below_upper_bound": candidate_pairs,
    }


def proxy_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1]))
        second_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2]))
        if {first_body, second_body} != {"head_pitch_link", "torso_link"}:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        records.append({
            "index": index, "distance_m": float(contact.dist), "body_names": [first_body, second_body],
            "geom_ids": [int(contact.geom1), int(contact.geom2)],
            "force_contact_frame_N_Nm": wrench.tolist(),
        })
    return records


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint = model.joint(name).id
    return float(data.qpos[model.jnt_qposadr[joint]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--preflight-result", required=True, type=Path)
    parser.add_argument("--full-urdf", required=True, type=Path)
    parser.add_argument("--mesh-query-root", required=True, type=Path)
    args = parser.parse_args()
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite evidence")
    for path in [args.scene, args.preflight_result, args.full_urdf, args.mesh_query_root]:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    sys.path.insert(0, str(args.mesh_query_root.resolve()))
    from rtree import index as rtree_index
    import trimesh

    preflight = load_preflight_module(WORKTREE / "scripts/research/issue46_full_x2_horizontal_cylinder_preflight.py")
    result = json.loads(args.preflight_result.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    xml = ET.parse(args.full_urdf).getroot()
    source = {}
    for name in ["head_pitch_link", "torso_link"]:
        link = next(item for item in xml.findall("link") if item.get("name") == name)
        mesh = link.find("./collision/geometry/mesh")
        origin = link.find("./collision/origin")
        if mesh is None or origin is None or origin.get("xyz", "0 0 0") != "0 0 0" or origin.get("rpy", "0 0 0") != "0 0 0":
            raise RuntimeError(f"unexpected source collision declaration for {name}")
        path = (args.full_urdf.parent / mesh.get("filename", "")).resolve()
        source[name] = {"path": path, "sha256": sha256(path), "mesh": trimesh.load(path, force="mesh", process=False)}

    right_open = result["gripper"]["right_open_hand_q_rad"]
    right_closed = result["gripper"]["right_contact_hand_q_rad"]
    left_open = result["gripper"]["left_idle_hand_q_rad"]
    waypoints = result["waypoints"]
    initial = result["root_hold_preflight"]
    states = [
        ("INITIAL_SETTLE", np.asarray(initial["equilibrium_arm_q_rad"], dtype=float), initial["equilibrium_right_hand_q_rad"], None),
        ("HORIZONTAL_PREGRASP", np.asarray(waypoints["HORIZONTAL_PREGRASP"]["arm_q_rad"], dtype=float), right_open, None),
        ("HORIZONTAL_APPROACH", np.asarray(waypoints["HORIZONTAL_APPROACH"]["arm_q_rad"], dtype=float), right_open, None),
        ("CLOSE_75_PERCENT", np.asarray(waypoints["HORIZONTAL_CONTACT"]["arm_q_rad"], dtype=float), right_closed, None),
        ("VERTICAL_LIFT_50MM", np.asarray(waypoints["VERTICAL_LIFT_50MM"]["arm_q_rad"], dtype=float), right_closed, np.asarray(result["cylinder"]["center_world_m"], dtype=float) + np.array([0.0, 0.0, 0.05])),
    ]
    phase_results = []
    for name, arm_q, right_hand, cylinder in states:
        started = time.monotonic()
        preflight.set_static_state(model, data, arm_q, right_hand, left_open, cylinder)
        head = model.body("head_pitch_link").id
        torso = model.body("torso_link").id
        head_rotation = data.xmat[head].reshape(3, 3)
        torso_rotation = data.xmat[torso].reshape(3, 3)
        head_triangles = source["head_pitch_link"]["mesh"].triangles @ head_rotation.T + data.xpos[head]
        torso_triangles = source["torso_link"]["mesh"].triangles @ torso_rotation.T + data.xpos[torso]
        exact = exact_surface_clearance(head_triangles, torso_triangles, rtree_index, trimesh.triangles)
        phase_results.append({
            "phase": name,
            "head_torso_relative_transform": {
                "translation_m": (torso_rotation.T @ (data.xpos[head] - data.xpos[torso])).tolist(),
                "rotation": (torso_rotation.T @ head_rotation).tolist(),
            },
            "head_and_waist_qpos_rad": {
                joint: joint_qpos(model, data, joint)
                for joint in ["head_yaw_joint", "head_pitch_joint", "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
            },
            "proxy_contacts": proxy_contacts(model, data),
            "source_surface": exact,
            "elapsed_s": time.monotonic() - started,
        })
    no_intersections = all(not phase["source_surface"]["surface_intersection"] for phase in phase_results)
    min_clearance = min(phase["source_surface"]["minimum_surface_distance_m"] for phase in phase_results)
    output_result = {
        "schema": "robotsim.issue46.full_x2_head_torso_source_trajectory_audit.v1",
        "mj_step_calls": 0,
        "mujoco_version": mujoco.__version__,
        "scene": str(args.scene.resolve()),
        "scene_sha256": sha256(args.scene.resolve()),
        "preflight_result": str(args.preflight_result.resolve()),
        "preflight_result_sha256": sha256(args.preflight_result.resolve()),
        "full_urdf": str(args.full_urdf.resolve()),
        "full_urdf_sha256": sha256(args.full_urdf.resolve()),
        "source_meshes": {
            name: {"path": str(record["path"]), "sha256": record["sha256"], "faces": int(len(record["mesh"].faces)), "watertight": bool(record["mesh"].is_watertight)}
            for name, record in source.items()
        },
        "phases": phase_results,
        "gate": {
            "source_surface_no_intersection_all_phases": no_intersections,
            "minimum_authentic_surface_clearance_mm": 1000.0 * min_clearance,
            "decision": "PASS" if no_intersections and min_clearance > 0.0 else "FAIL",
            "limitation": "Source meshes are non-watertight, so this is an exact triangle-surface intersection and clearance result, not a signed enclosed-volume claim.",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(output_result, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"gate": output_result["gate"], "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
