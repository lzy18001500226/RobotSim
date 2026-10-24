#!/usr/bin/env python3
"""Qualify one bounded X2+Robotiq arm-only approach variant without stepping physics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import sys
import uuid
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
REVIEWS = Path("/home/lzy18001500226/projects/RobotSim/reviews")
EXPECTED_MODEL_SHA = "a5dcd589fedaea5645ccf29b708b560f2e0dc923c8955a7d5901a996df57be94"
EXPECTED_X2_SHA = "20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292"
EXPECTED_ROBOTIQ_SHA = "d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de"
EXPECTED_SPRING_STL_SHA = "4fb74a8a0d76c0e471cf19fd48bc676fd5b19123798e7a39eb6aa56869354283"
EXPECTED_APPROACH_SHA = "3f938d033a72ec9aba5dd820595abc27d5aeebfbb0c5396e77b71bbbf79d24a5"
ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
CLEARANCE_M = 0.002
SWEEP_MAX_DEPTH = 11
SWEEP_NODE_BUDGET = 5000
CYLINDER_SIDES = 96

sys.path.insert(0, str(ROOT / "scripts" / "agent"))
import path_governance as pg


class GeometryUnresolved(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(model, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(object_id)) or f"unnamed_{int(object_id)}"


def body_name(model, geom_id: int) -> str:
    return name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))


def pair_key(first: int, second: int) -> tuple[int, int]:
    return tuple(sorted((int(first), int(second))))


def load_exclusions(model_xml: Path) -> set[tuple[str, str]]:
    root = ET.parse(model_xml).getroot()
    return {
        tuple(sorted((entry.attrib.get("body1", ""), entry.attrib.get("body2", ""))))
        for entry in root.findall("./contact/exclude")
    }


def collision_enabled(model, geom1: int, geom2: int, exclusions: set[tuple[str, str]]) -> bool:
    if geom1 == geom2:
        return False
    body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
    if body1 == body2 or int(model.body_weldid[body1]) == int(model.body_weldid[body2]):
        return False
    if int(model.body_parentid[body1]) == body2 or int(model.body_parentid[body2]) == body1:
        return False
    names = tuple(sorted((name(model, mujoco.mjtObj.mjOBJ_BODY, body1), name(model, mujoco.mjtObj.mjOBJ_BODY, body2))))
    if names in exclusions:
        return False
    return bool(
        (int(model.geom_contype[geom1]) & int(model.geom_conaffinity[geom2]))
        or (int(model.geom_contype[geom2]) & int(model.geom_conaffinity[geom1]))
    )


def icosahedron_unit() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            points.extend(((0.0, a, b), (a, b, 0.0), (b, 0.0, a)))
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    points /= np.linalg.norm(points[0])
    hull = ConvexHull(points)
    inradius = float(np.min(-hull.equations[:, 3]))
    return points / inradius


def primitive_collision_points(model, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.asarray([(x, y, z) for x in (-size[0], size[0])
                           for y in (-size[1], size[1]) for z in (-size[2], size[2])])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        theta = np.arange(CYLINDER_SIDES) * 2.0 * math.pi / CYLINDER_SIDES
        radius = float(size[0]) / math.cos(math.pi / CYLINDER_SIDES)
        ring = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        return np.vstack((np.column_stack((ring, np.full(CYLINDER_SIDES, -size[1]))),
                          np.column_stack((ring, np.full(CYLINDER_SIDES, size[1])))))
    sphere = icosahedron_unit()
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return sphere * float(size[0])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return sphere * size[:3]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half_segment = float(size[1])
        return np.vstack((sphere * float(size[0]) + [0.0, 0.0, -half_segment],
                          sphere * float(size[0]) + [0.0, 0.0, half_segment]))
    raise GeometryUnresolved(f"unsupported active collision geom type {geom_type} for {name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)}")


def convex_shape(model, geom_id: int, mesh_cache: dict[int, dict]) -> dict | None:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id >= 0:
        if mesh_id not in mesh_cache:
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            points = np.asarray(model.mesh_vert[start:start + count], dtype=float)
            hull = ConvexHull(points)
            points = points[hull.vertices]
            hull = ConvexHull(points)
            edges = set()
            for face in hull.simplices:
                for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    edges.add(tuple(sorted((int(i), int(j)))) )
            edge_vectors = points[[edge[1] for edge in sorted(edges)]] - points[[edge[0] for edge in sorted(edges)]]
            mesh_cache[mesh_id] = {
                "points": points,
                "normals": hull.equations[:, :3].copy(),
                "edges": edge_vectors,
            }
        return mesh_cache[mesh_id]
    points = primitive_collision_points(model, geom_id)
    hull = ConvexHull(points)
    points = points[hull.vertices]
    hull = ConvexHull(points)
    edges = set()
    for face in hull.simplices:
        for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add(tuple(sorted((int(i), int(j)))) )
    edge_vectors = points[[edge[1] for edge in sorted(edges)]] - points[[edge[0] for edge in sorted(edges)]]
    return {"points": points, "normals": hull.equations[:, :3].copy(), "edges": edge_vectors}


def world_shape(model, data, geom_id: int, shape: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=float)
    points = shape["points"] @ rotation.T + position
    normals = shape["normals"] @ rotation.T
    return points, normals, position, rotation


def all_world_shapes(model, data, shapes: dict[int, dict]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return {geom_id: world_shape(model, data, geom_id, shape) for geom_id, shape in shapes.items()}


def aabb_gap(first: np.ndarray, second: np.ndarray) -> float:
    first_min, first_max = first.min(axis=0), first.max(axis=0)
    second_min, second_max = second.min(axis=0), second.max(axis=0)
    return aabb_gap_bounds(first_min, first_max, second_min, second_max)


def aabb_gap_bounds(first_min: np.ndarray, first_max: np.ndarray,
                    second_min: np.ndarray, second_max: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(first_min - second_max, second_min - first_max))
    return float(np.linalg.norm(delta))


def sat_lower_bound(first_points: np.ndarray, first_normals: np.ndarray,
                    second_points: np.ndarray, second_normals: np.ndarray) -> tuple[float, list[float] | None]:
    axes = np.vstack((first_normals, second_normals, np.eye(3),
                      (np.mean(second_points, axis=0) - np.mean(first_points, axis=0))[None, :]))
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    best, best_axis = -math.inf, None
    # Every tested projection gap is a valid lower bound; omitted SAT axes fail closed as zero.
    for start in range(0, len(axes), 32):
        current = axes[start:start + 32]
        first = first_points @ current.T
        second = second_points @ current.T
        gaps = np.maximum(second.min(axis=0) - first.max(axis=0),
                          first.min(axis=0) - second.max(axis=0))
        idx = int(np.argmax(gaps))
        if float(gaps[idx]) > best:
            best, best_axis = float(gaps[idx]), current[idx].tolist()
    return max(0.0, best), best_axis


def geom_pair_lower_bound(model, geom1: int, geom2: int,
                          world_shapes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                          contact_map: dict[tuple[int, int], list]) -> dict:
    p1, n1, _, _ = world_shapes[geom1]
    p2, n2, _, _ = world_shapes[geom2]
    bound = aabb_gap(p1, p2)
    axis = None
    method = "world_aabb_lower_bound"
    if bound < CLEARANCE_M:
        sat, axis = sat_lower_bound(p1, n1, p2, n2)
        if sat > bound:
            bound, method = sat, "convex_sat_projection_lower_bound"
    contacts = contact_map.get(pair_key(geom1, geom2), [])
    if contacts:
        active_distance = min(float(contact.dist) for contact in contacts)
        if active_distance <= 0.0:
            return {"clearance_m": active_distance, "method": "active_mujoco_contact", "axis": axis}
    return {"clearance_m": bound, "method": method, "axis": axis}


def contact_map_for_data(data) -> dict[tuple[int, int], list]:
    result = defaultdict(list)
    for contact in data.contact[:data.ncon]:
        result[pair_key(contact.geom1, contact.geom2)].append(contact)
    return result


def binary_stl_triangles(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if len(blob) < 84:
        raise ValueError("source STL is too short")
    count = struct.unpack_from("<I", blob, 80)[0]
    if len(blob) != 84 + 50 * count:
        raise ValueError("source STL is not the pinned binary STL form")
    records = np.ndarray((count, 12), dtype="<f4", buffer=blob, offset=84, strides=(50, 4))
    return records[:, 3:12].reshape(count, 3, 3).astype(float)


def cylinder_sdf(points: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    radial = np.linalg.norm(points[:, :2], axis=1) - radius
    axial = np.abs(points[:, 2]) - half_height
    q = np.column_stack((radial, axial))
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)


def source_mesh_frame(model, geom_id: int, stl_path: Path, expected_sha: str) -> tuple[np.ndarray, dict]:
    actual_sha = sha256(stl_path)
    if actual_sha != expected_sha:
        raise ValueError("pinned source spring_link.stl hash mismatch")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0 or name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) != "rq_spring_link":
        raise ValueError("spring-link collision geom does not use the expected source mesh")
    triangles = binary_stl_triangles(stl_path)
    quaternion_xyzw = np.asarray(model.mesh_quat[mesh_id], dtype=float)[[1, 2, 3, 0]]
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    source_local = (triangles * np.asarray(model.mesh_scale[mesh_id]) - np.asarray(model.mesh_pos[mesh_id])) @ rotation
    compiled_start = int(model.mesh_vertadr[mesh_id])
    compiled_count = int(model.mesh_vertnum[mesh_id])
    compiled_hull = np.asarray(model.mesh_vert[compiled_start:compiled_start + compiled_count], dtype=float)
    alignment_max = float(cKDTree(source_local.reshape(-1, 3)).query(compiled_hull, k=1)[0].max())
    if alignment_max > 1e-6:
        raise ValueError("source STL to compiled spring-link mesh-frame registration check failed")
    return source_local, {
        "path": str(stl_path), "sha256": actual_sha, "triangle_count": int(len(triangles)),
        "compiled_mesh_name": "rq_spring_link", "compiled_vertex_alignment_max_error_m": alignment_max,
        "transform": "(source_vertices_m * mesh_scale - mesh_pos) rotated by source mesh quaternion into compiled geom-local coordinates",
    }


def source_triangle_cylinder_minimum(triangles: np.ndarray, geom_position: np.ndarray,
                                     geom_rotation: np.ndarray, bottle_position: np.ndarray,
                                     bottle_rotation: np.ndarray, radius: float, half_height: float,
                                     tolerance_m: float = 1e-5, max_nodes: int = 50000,
                                     required_clearance_m: float = CLEARANCE_M) -> dict:
    world = triangles.reshape(-1, 3) @ geom_rotation.T + geom_position
    bottle_local = (world - bottle_position) @ bottle_rotation
    triangles_local = bottle_local.reshape(-1, 3, 3)
    best = math.inf
    best_index = -1
    best_point = None
    heap: list[tuple[float, int, int, int, np.ndarray]] = []
    sequence = 0

    def add_patch(triangle: np.ndarray, tri_index: int, depth: int) -> None:
        nonlocal best, best_index, best_point, sequence
        center = np.mean(triangle, axis=0)
        candidates = np.vstack((triangle, center[None, :]))
        values = cylinder_sdf(candidates, radius, half_height)
        local_index = int(np.argmin(values))
        if float(values[local_index]) < best:
            best = float(values[local_index])
            best_index = tri_index
            best_point = candidates[local_index].tolist()
        radius_bound = float(np.max(np.linalg.norm(triangle - center, axis=1)))
        lower = float(cylinder_sdf(center[None, :], radius, half_height)[0] - radius_bound)
        if lower < best - tolerance_m:
            heapq.heappush(heap, (lower, sequence, depth, tri_index, triangle))
            sequence += 1

    for idx, triangle in enumerate(triangles_local):
        add_patch(triangle, idx, 0)
    nodes = 0
    depth_limited = False
    depth_limited_lower = math.inf
    threshold_proven = False
    while heap and nodes < max_nodes:
        lower, _, depth, tri_index, triangle = heapq.heappop(heap)
        if best <= required_clearance_m:
            break
        if lower >= required_clearance_m:
            threshold_proven = True
            depth_limited_lower = min(depth_limited_lower, lower)
            break
        if lower >= best - tolerance_m:
            break
        if depth >= 16:
            depth_limited = True
            depth_limited_lower = min(depth_limited_lower, lower)
            continue
        a, b, c = triangle
        ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
        for child in (np.asarray((a, ab, ca)), np.asarray((ab, b, bc)),
                      np.asarray((ca, bc, c)), np.asarray((ab, bc, ca))):
            add_patch(child, tri_index, depth + 1)
        nodes += 1
    unresolved = (bool(heap) and nodes >= max_nodes) or depth_limited
    lower_bound = min(best, heap[0][0]) if heap else best
    lower_bound = min(lower_bound, depth_limited_lower)
    threshold_proven = threshold_proven or lower_bound >= required_clearance_m or best <= required_clearance_m
    unresolved = unresolved and not threshold_proven
    return {
        "lower_bound_m": float(lower_bound), "upper_bound_m": float(best),
        "error_bound_m": float(max(0.0, best - lower_bound)), "best_triangle_index": int(best_index),
        "best_point_bottle_local_m": best_point, "branch_nodes": nodes,
        "tolerance_m": tolerance_m, "required_clearance_m": required_clearance_m,
        "threshold_proven": threshold_proven, "unresolved": unresolved,
    }


def solve_pregrasp(model, data, result: dict, retreat_m: float) -> dict:
    grasp = result["ik"]["grasp_first"]
    adapter = result["adapter"]
    arm_joints = list(ARM)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in arm_joints]
    if any(joint < 0 for joint in joint_ids):
        raise ValueError("model lacks one of the seven right-arm joints")
    qpos_ids = np.asarray([int(model.jnt_qposadr[joint]) for joint in joint_ids], dtype=int)
    lower = np.asarray([model.jnt_range[joint, 0] for joint in joint_ids], dtype=float)
    upper = np.asarray([model.jnt_range[joint, 1] for joint in joint_ids], dtype=float)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rq_m0_tcp")
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rq_base_mount")
    if min(site_id, bottle_geom, gripper_body) < 0:
        raise ValueError("compiled model is missing required TCP/bottle/gripper frames")
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    bottle_center = np.asarray(data.geom_xpos[bottle_geom], dtype=float).copy()
    tool_offset = np.asarray(adapter["isolated_lift_grasp_offset_tool_local_m"], dtype=float)
    insertion_local = np.asarray(adapter["insertion_axis_local"], dtype=float)
    vertical_local = np.cross(insertion_local, np.asarray(adapter["pad_opening_axis_local"], dtype=float))
    vertical_local /= np.linalg.norm(vertical_local)
    seed = np.asarray(grasp["q_rad"], dtype=float)
    evaluations: list[dict] = []

    def metrics(q: np.ndarray):
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        position = np.asarray(data.site_xpos[site_id], dtype=float).copy()
        rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()
        insertion = rotation @ insertion_local
        target = bottle_center + rotation @ tool_offset + np.array([0.0, 0.0, float(result["scene_configuration"]["grasp_height_offset_world_m"])]) - insertion * retreat_m
        vertical = rotation @ vertical_local
        desired_xy = bottle_center - np.asarray(data.xpos[gripper_body], dtype=float)
        desired_xy[2] = 0.0
        desired_xy = desired_xy[:2]
        desired_xy /= max(float(np.linalg.norm(desired_xy)), 1e-12)
        insertion_xy = insertion[:2].copy()
        insertion_xy /= max(float(np.linalg.norm(insertion_xy)), 1e-12)
        yaw_error = math.atan2(desired_xy[0] * insertion_xy[1] - desired_xy[1] * insertion_xy[0],
                               float(np.dot(desired_xy, insertion_xy)))
        vertical_error = float(math.acos(np.clip(np.dot(vertical, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
        return position, target, rotation, insertion, vertical, yaw_error, vertical_error

    def residual(q: np.ndarray) -> np.ndarray:
        position, target, _, _, vertical, yaw_error, _ = metrics(q)
        position_error = position - target
        vertical_vector_error = vertical - np.array([0.0, 0.0, 1.0])
        evaluations.append({
            "evaluation": len(evaluations), "q_rad": q.tolist(),
            "position_error_m": position_error.tolist(), "yaw_error_rad": yaw_error,
        })
        return np.concatenate((position_error / 0.005, vertical_vector_error / 0.04,
                               np.array([yaw_error / 0.30])))

    solution = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                             bounds=(lower, upper), xtol=1e-11, ftol=1e-11, gtol=1e-11,
                             max_nfev=300, x_scale="jac")
    position, target, rotation, insertion, vertical, yaw_error, vertical_error = metrics(solution.x)
    position_error = float(np.linalg.norm(position - target))
    orientation_error = float(math.hypot(vertical_error, yaw_error))
    limits_pass = bool(np.all(solution.x >= lower) and np.all(solution.x <= upper))
    return {
        "retreat_distance_m": retreat_m,
        "target_position_world_m": target.tolist(),
        "target_rotation_world": rotation.tolist(),
        "insertion_axis_world": insertion.tolist(),
        "joint_names": arm_joints,
        "joint_positions_rad": solution.x.tolist(),
        "position_error_m": position_error,
        "vertical_error_rad": vertical_error,
        "approach_yaw_error_rad": yaw_error,
        "orientation_error_rad": orientation_error,
        "minimum_joint_limit_margin_rad": float(np.min(np.minimum(solution.x - lower, upper - solution.x))),
        "source_joint_limits_pass": limits_pass,
        "ik_pass": bool(solution.success and position_error <= 0.003 and vertical_error <= 0.03 and abs(yaw_error) <= math.radians(25.0)),
        "solver_message": str(solution.message), "nfev": int(solution.nfev),
        "mj_forward_calls": len(evaluations) + 2,
    }


def model_path_pairs(model, geoms: list[int], exclusions: set[tuple[str, str]]) -> dict[str, list[tuple[int, int]]]:
    world_body = 0
    bottle_bodies = {i for i in range(model.nbody) if name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith("m0_bottle")}
    table_bodies = {i for i in range(model.nbody) if name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith("m0_table")}
    robot = [g for g in geoms if int(model.geom_bodyid[g]) != world_body
             and int(model.geom_bodyid[g]) not in bottle_bodies | table_bodies
             and name(model, mujoco.mjtObj.mjOBJ_GEOM, g) != "m0_floor"]
    bottle = [g for g in geoms if int(model.geom_bodyid[g]) in bottle_bodies]
    table = [g for g in geoms if int(model.geom_bodyid[g]) in table_bodies]
    object_pairs = [(g, h) for g in robot for h in bottle if collision_enabled(model, g, h, exclusions)]
    pad_bodies = {i for i in range(model.nbody) if name(model, mujoco.mjtObj.mjOBJ_BODY, i) in {"rq_left_pad", "rq_right_pad"}}
    nonpad_object_pairs = [(g, h) for g in robot if int(model.geom_bodyid[g]) not in pad_bodies
                           for h in bottle if collision_enabled(model, g, h, exclusions)]
    table_pairs = [(g, h) for g in robot for h in table if collision_enabled(model, g, h, exclusions)]
    all_self_pairs = [(robot[i], robot[j]) for i in range(len(robot)) for j in range(i + 1, len(robot))
                      if collision_enabled(model, robot[i], robot[j], exclusions)]
    internal_mechanism_pairs = [
        pair for pair in all_self_pairs
        if (("spring_link" in body_name(model, pair[0]) and "pad" in body_name(model, pair[1]))
            or ("pad" in body_name(model, pair[0]) and "spring_link" in body_name(model, pair[1])))
    ]
    internal_ids = {pair_key(*pair) for pair in internal_mechanism_pairs}
    self_pairs = [pair for pair in all_self_pairs if pair_key(*pair) not in internal_ids]
    return {"robot_bottle": object_pairs, "robot_bottle_nonpad": nonpad_object_pairs,
            "robot_table": table_pairs, "robot_self": self_pairs,
            "internal_mechanism": internal_mechanism_pairs,
            "robot_geoms": robot, "bottle_geoms": bottle, "table_geoms": table}


def group_clearance(model, pairs: list[tuple[int, int]], world_shapes, contact_map) -> dict:
    minimum = math.inf
    witness = None
    unresolved = False
    for first, second in pairs:
        result = geom_pair_lower_bound(model, first, second, world_shapes, contact_map)
        gap = float(result["clearance_m"])
        if gap < minimum:
            minimum = gap
            witness = {
                "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                "body1": body_name(model, first),
                "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                "body2": body_name(model, second),
                "measurement": result["method"], "lower_bound_m": gap,
                "separating_axis_world": result.get("axis"),
            }
        unresolved |= gap <= 0.0
    if not math.isfinite(minimum):
        return {"minimum_clearance_lower_bound_m": None, "witness": None, "pairs_checked": len(pairs), "unresolved": False}
    return {"minimum_clearance_lower_bound_m": minimum, "witness": witness,
            "pairs_checked": len(pairs), "unresolved": unresolved}


def source_stl_cylinder_check(model, data, spring_geom: int, bottle_geom: int,
                              source_triangles_local: np.ndarray, tolerance_m: float = 1e-5) -> dict:
    spring_rotation = np.asarray(data.geom_xmat[spring_geom], dtype=float).reshape(3, 3)
    spring_position = np.asarray(data.geom_xpos[spring_geom], dtype=float)
    bottle_rotation = np.asarray(data.geom_xmat[bottle_geom], dtype=float).reshape(3, 3)
    bottle_position = np.asarray(data.geom_xpos[bottle_geom], dtype=float)
    world = source_triangles_local.reshape(-1, 3) @ spring_rotation.T + spring_position
    bottle_local = (world - bottle_position) @ bottle_rotation
    triangles = bottle_local.reshape(-1, 3, 3)
    radius, half_height = (float(value) for value in model.geom_size[bottle_geom, :2])
    measured = source_triangle_cylinder_minimum(triangles, np.zeros(3), np.eye(3), np.zeros(3), np.eye(3),
                                                radius, half_height, tolerance_m=tolerance_m)
    measured["source_stl_geom1"] = name(model, mujoco.mjtObj.mjOBJ_GEOM, spring_geom)
    measured["source_stl_body1"] = body_name(model, spring_geom)
    measured["bottle_geom2"] = name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom)
    return measured


def geometry_state(model, data, shapes, pair_sets) -> dict:
    mujoco.mj_forward(model, data)
    world_shapes = all_world_shapes(model, data, shapes)
    contact_map = contact_map_for_data(data)
    return {group: group_clearance(model, pairs, world_shapes, contact_map)
            for group, pairs in pair_sets.items()
            if group in {"robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self", "internal_mechanism"}}


def compute_reach_bounds(model, data, shapes, robot_geoms, joint_ids) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = {}
    for geom_id in robot_geoms:
        body_id = int(model.geom_bodyid[geom_id])
        chain = []
        body = body_id
        while body != 0:
            chain.append(body)
            body = int(model.body_parentid[body])
        chain.reverse()
        result[geom_id] = {}
        radius = float(np.max(np.linalg.norm(shapes[geom_id]["points"], axis=1)))
        for joint_id in joint_ids:
            joint_body = int(model.jnt_bodyid[joint_id])
            if joint_body not in chain:
                continue
            index = chain.index(joint_body)
            reach = float(np.linalg.norm(data.xpos[joint_body] - data.xanchor[joint_id]))
            parent = joint_body
            for child in chain[index + 1:]:
                child_joints = [j for j in range(int(model.body_jntadr[child]),
                                                   int(model.body_jntadr[child] + model.body_jntnum[child]))
                                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
                parent_pos, child_pos = data.xpos[parent], data.xpos[child]
                if child_joints:
                    anchor = data.xanchor[child_joints[0]]
                    reach += float(np.linalg.norm(anchor - parent_pos) + np.linalg.norm(child_pos - anchor))
                    for other in child_joints[1:]:
                        reach += float(np.linalg.norm(data.xanchor[other] - anchor))
                else:
                    reach += float(np.linalg.norm(child_pos - parent_pos))
                parent = child
            reach += float(np.linalg.norm(data.geom_xpos[geom_id] - data.xpos[body_id])) + radius
            result[geom_id][joint_id] = reach
    return result


def swept_segment(model, data, shapes, pair_sets, qpos_ids, q0, q1, joint_ids,
                  depth: int, nodes: dict[str, int], reach_bounds,
                  max_depth: int, max_nodes: int) -> dict:
    nodes["count"] += 1
    if nodes["count"] > max_nodes:
        return {"pass": False, "status": "UNPROVEN_NODE_BUDGET", "depth": depth, "node_count": nodes["count"]}
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    midpoint = (q0 + q1) / 2.0
    data.qpos[qpos_ids] = midpoint
    mujoco.mj_forward(model, data)
    world_shapes = all_world_shapes(model, data, shapes)
    contact_map = contact_map_for_data(data)
    delta_by_joint = {int(joint): abs(float(b - a)) / 2.0 for joint, a, b in zip(joint_ids, q0, q1)}
    moving = {}
    for geom_id, reach in reach_bounds.items():
        moving[geom_id] = sum(reach.get(joint, 0.0) * delta for joint, delta in delta_by_joint.items())
    group_minimum = {}
    first_failure = None
    for group, pairs in pair_sets.items():
        if group not in {"robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self"}:
            continue
        worst = math.inf
        worst_pair = None
        for first, second in pairs:
            pair_state = geom_pair_lower_bound(model, first, second, world_shapes, contact_map)
            robust = float(pair_state["clearance_m"]) - moving.get(first, 0.0) - moving.get(second, 0.0)
            if robust < worst:
                worst = robust
                worst_pair = {
                    "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                    "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                    "body1": body_name(model, first), "body2": body_name(model, second),
                    "midpoint_clearance_lower_bound_m": float(pair_state["clearance_m"]),
                    "motion_bound_m": moving.get(first, 0.0) + moving.get(second, 0.0),
                    "swept_lower_bound_m": robust,
                    "measurement": pair_state["method"],
                }
        group_minimum[group] = {"swept_lower_bound_m": worst, "worst_pair": worst_pair}
        if worst < CLEARANCE_M and first_failure is None:
            first_failure = {"group": group, **group_minimum[group]}
    if first_failure is None:
        return {"pass": True, "status": "PASS", "depth": depth, "node_count": nodes["count"],
                "group_minimum": group_minimum}
    if depth >= max_depth:
        return {"pass": False, "status": "UNPROVEN_DEPTH_LIMIT", "depth": depth,
                "node_count": nodes["count"], "first_failure": first_failure, "group_minimum": group_minimum}
    left = swept_segment(model, data, shapes, pair_sets, qpos_ids, q0, midpoint, joint_ids,
                         depth + 1, nodes, reach_bounds, max_depth, max_nodes)
    if not left["pass"]:
        return left
    right = swept_segment(model, data, shapes, pair_sets, qpos_ids, midpoint, q1, joint_ids,
                          depth + 1, nodes, reach_bounds, max_depth, max_nodes)
    if not right["pass"]:
        return right
    return {"pass": True, "status": "PASS", "depth": depth, "node_count": nodes["count"],
            "group_minimum": {group: {"swept_lower_bound_m": min(left["group_minimum"][group]["swept_lower_bound_m"],
                                                                      right["group_minimum"][group]["swept_lower_bound_m"]),
                                      "worst_pair": left["group_minimum"][group]["worst_pair"]
                                      if left["group_minimum"][group]["swept_lower_bound_m"] <= right["group_minimum"][group]["swept_lower_bound_m"]
                                      else right["group_minimum"][group]["worst_pair"]}
                              for group in left["group_minimum"]}}


def path_sweep(model, data, shapes, pair_sets, qpos_ids, path, joint_ids,
               max_depth: int = SWEEP_MAX_DEPTH, max_nodes: int = SWEEP_NODE_BUDGET) -> dict:
    segments = []
    nodes = {"count": 0}
    groups = [group for group in pair_sets if group in {"robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self"}]
    overall = {group: math.inf for group in groups}
    worst = {group: None for group in groups}
    data.qpos[qpos_ids] = np.asarray(path[0], dtype=float)
    mujoco.mj_forward(model, data)
    reach_bounds = compute_reach_bounds(model, data, shapes, pair_sets["robot_geoms"], joint_ids)
    for index, (q0, q1) in enumerate(zip(path, path[1:])):
        row = swept_segment(model, data, shapes, pair_sets, qpos_ids, q0, q1, joint_ids,
                            0, nodes, reach_bounds, max_depth, max_nodes)
        row["segment_index"] = index
        segments.append(row)
        if not row["pass"]:
            return {"pass": False, "status": row["status"], "segments_checked": index + 1,
                    "total_nodes": nodes["count"], "mj_forward_calls": nodes["count"] + 1,
                    "first_failed_segment": row, "segments": segments}
        for group, entry in row["group_minimum"].items():
            if entry["swept_lower_bound_m"] < overall[group]:
                overall[group] = float(entry["swept_lower_bound_m"])
                worst[group] = entry["worst_pair"]
    return {"pass": True, "status": "PASS", "segments_checked": len(segments), "total_nodes": nodes["count"],
            "mj_forward_calls": nodes["count"] + 1,
            "minimum_swept_clearance_lower_bound_m": overall, "worst_pairs": worst, "segments": segments}


def episode_contact_order(trace_path: Path) -> dict:
    per_pad: dict[str, list[dict]] = {"rq_left_pad": [], "rq_right_pad": []}
    non_pad: list[dict] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "m0_bottle" not in (row.get("body1"), row.get("body2")):
            continue
        other = row.get("body2") if row.get("body1") == "m0_bottle" else row.get("body1")
        if other in per_pad and float(row.get("normal_force_n") or 0.0) > 0.0:
            per_pad[other].append({"step": int(row["step"]), "time_s": float(row["time_s"]),
                                   "geom1": row.get("geom1"), "geom2": row.get("geom2"),
                                   "normal_force_n": float(row["normal_force_n"])})
        elif other != "m0_table":
            non_pad.append({"step": int(row["step"]), "time_s": float(row["time_s"]),
                            "body": other, "geom1": row.get("geom1"), "geom2": row.get("geom2"),
                            "distance_m": float(row.get("distance_m", 0.0))})
    return {
        "first_force_bearing_pad_contacts": {pad: (min(rows, key=lambda row: row["step"]) if rows else None)
                                             for pad, rows in per_pad.items()},
        "force_bearing_pad_contact_rows": {pad: len(rows) for pad, rows in per_pad.items()},
        "non_pad_bottle_contact_rows": len(non_pad), "first_non_pad_bottle_contact": min(non_pad, key=lambda row: row["step"]) if non_pad else None,
        "contact_trace_path": str(trace_path),
    }


def summarize_internal_clearance(records: list[dict]) -> dict:
    samples = [row for record in records for row in record.get("internal_mechanism_clearances", [])]
    if not samples:
        return {"measured": False, "reason": "NO_COLLISION_ELIGIBLE_SPRING_PAD_PAIR"}
    by_pair: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        by_pair[(row["body1"], row["geom1"], row["body2"], row["geom2"])].append(row)
    per_pair = []
    for pair, rows in by_pair.items():
        distances = [float(row["clearance_lower_bound_m"]) for row in rows]
        per_pair.append({"body1": pair[0], "geom1": pair[1], "body2": pair[2], "geom2": pair[3],
                         "minimum_clearance_lower_bound_m": min(distances),
                         "maximum_clearance_lower_bound_m": max(distances),
                         "range_m": max(distances) - min(distances),
                         "measurement": rows[0]["measurement"],
                         "mujoco_geom_distance_sample_m": rows[0].get("mujoco_geom_distance_m"),
                         "active_contact_seen": any(row["active_contact"] for row in rows),
                         "samples": len(rows)})
    worst = min(per_pair, key=lambda row: row["minimum_clearance_lower_bound_m"])
    return {
        "measured": True,
        "sample_pair_count": len(samples),
        "pair_count": len(per_pair),
        "pair_results": per_pair,
        "worst_pair": worst,
        "worst_pair_invariant_within_1um": worst["range_m"] <= 1e-6,
        "subthreshold_pairs_invariant_within_1um": all(
            row["range_m"] <= 1e-6 for row in per_pair
            if row["minimum_clearance_lower_bound_m"] < CLEARANCE_M
        ),
        "minimum_clearance_target_m": CLEARANCE_M,
    }


def qualify(model_path: Path, history_path: Path, samples_path: Path, contact_trace_path: Path,
            retreat_m: float, source_stl_path: Path, previous_static_path: Path | None = None) -> dict:
    model_hash = sha256(model_path)
    sample_hash = sha256(samples_path)
    result_hash = sha256(history_path)
    if model_hash != EXPECTED_MODEL_SHA:
        raise ValueError("historical full-X2/Robotiq model hash mismatch")
    if sample_hash != EXPECTED_APPROACH_SHA:
        raise ValueError("historical 101-point approach sample hash mismatch")
    prior = json.loads(history_path.read_text(encoding="utf-8"))
    identity = prior["identity"]
    if prior.get("status") != "PASS" or identity.get("robotiq_xml_sha256") != EXPECTED_ROBOTIQ_SHA:
        raise ValueError("historical episode identity/status mismatch")
    if identity.get("x2_mjcf_sha256") != EXPECTED_X2_SHA:
        raise ValueError("historical X2 source identity mismatch")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    exclusions = load_exclusions(model_path)
    shape_cache: dict[int, dict] = {}
    shapes = {}
    for geom_id in range(model.ngeom):
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
            shape = convex_shape(model, geom_id, shape_cache)
            if shape is not None:
                shapes[geom_id] = shape
    pair_sets = model_path_pairs(model, list(shapes), exclusions)
    qpos_ids = np.asarray([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]) for joint in ARM], dtype=int)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in ARM]
    if any(joint < 0 for joint in joint_ids):
        raise ValueError("full-X2 model missing a right arm joint")
    rows = json.loads(samples_path.read_text(encoding="utf-8"))
    if len(rows) != 101 or [round(float(row["fraction"]), 2) for row in rows] != [i / 100 for i in range(101)]:
        raise ValueError("saved reference path must contain exactly 101 ordered samples")
    original_path = [np.asarray(row["right_arm_qpos_rad"], dtype=float) for row in rows]
    if any(q.shape != (7,) for q in original_path):
        raise ValueError("historical right arm sample has incorrect joint count")
    internal_mechanism_pairs = pair_sets["internal_mechanism"]

    grasp = prior["ik"]["grasp_first"]
    q_grasp = np.asarray(grasp["q_rad"], dtype=float)
    q_pre = solve_pregrasp(model, data, prior, retreat_m)
    if not q_pre["ik_pass"]:
        return {
            "status": "BLOCKED_PREGRASP_IK", "input_model_sha256": model_hash,
            "input_history_sha256": result_hash, "input_path_sha256": sample_hash,
            "retreat_m": retreat_m, "pregrasp_ik": q_pre,
        }
    q_pre_rad = np.asarray(q_pre["joint_positions_rad"], dtype=float)
    corrected_path = [q_pre_rad + fraction * (q_grasp - q_pre_rad)
                      for fraction in np.linspace(0.0, 1.0, 101)]

    def evaluate_discrete(path):
        records = []
        violations = []
        for index, q in enumerate(path):
            data.qpos[:] = model.qpos0
            data.qvel[:] = 0.0
            data.qpos[qpos_ids] = q
            mujoco.mj_forward(model, data)
            for joint_id in range(model.njnt):
                if not model.jnt_limited[joint_id] or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                position = float(data.qpos[int(model.jnt_qposadr[joint_id])])
                lower, upper = map(float, model.jnt_range[joint_id])
                if not lower - 1e-9 <= position <= upper + 1e-9:
                    violations.append({"sample": index, "joint": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                                       "q_rad": position, "range_rad": [lower, upper]})
            clearance = geometry_state(model, data, shapes, pair_sets)
            contact_map = contact_map_for_data(data)
            contact_counts = {group: sum(len(contact_map.get(pair_key(a, b), [])) for a, b in pairs)
                              for group, pairs in pair_sets.items()
                              if group in {"robot_bottle", "robot_table", "robot_self", "internal_mechanism"}}
            internal_clearances = []
            world_shapes = all_world_shapes(model, data, shapes)
            for first, second in internal_mechanism_pairs:
                pair_result = geom_pair_lower_bound(model, first, second, world_shapes, contact_map)
                internal_clearances.append({
                    "geom1": name(model, mujoco.mjtObj.mjOBJ_GEOM, first),
                    "body1": body_name(model, first),
                    "geom2": name(model, mujoco.mjtObj.mjOBJ_GEOM, second),
                    "body2": body_name(model, second),
                    "clearance_lower_bound_m": float(pair_result["clearance_m"]),
                    "measurement": pair_result["method"],
                    "mujoco_geom_distance_m": float(mujoco.mj_geomDistance(model, data, first, second, 0.1, None)),
                    "active_contact": bool(contact_map.get(pair_key(first, second))),
                })
            records.append({"sample": index, "fraction": index / 100.0, "right_arm_qpos_rad": q.tolist(),
                            "clearance": clearance, "active_contact_counts": contact_counts,
                            "internal_mechanism_clearances": internal_clearances})
        return records, violations

    baseline_records, baseline_limit_violations = evaluate_discrete(original_path)
    corrected_records, corrected_limit_violations = evaluate_discrete(corrected_path)
    spring_mesh_ids = [geom for geom in pair_sets["robot_bottle"]
                       if int(model.geom_dataid[geom[0]]) >= 0
                       and name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom[0]])) == "rq_spring_link"
                       and name(model, mujoco.mjtObj.mjOBJ_GEOM, geom[1]) == "bottle_body"]
    if not spring_mesh_ids:
        raise ValueError("no enabled spring-link/bottle_body collision pair exists")
    right_spring_geom = next((a for a, b in spring_mesh_ids if body_name(model, a) == "rq_right_spring_link"), None)
    if right_spring_geom is None:
        raise ValueError("right spring-link collision geom not found")
    source_triangles_local, source_mesh_identity = source_mesh_frame(model, right_spring_geom, source_stl_path, EXPECTED_SPRING_STL_SHA)
    bottle_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bottle_body")
    spring_mesh_geoms = sorted({a for a, _ in spring_mesh_ids})
    source_mesh_checks = {}
    for label, path, records in (("baseline", original_path, baseline_records), ("corrected", corrected_path, corrected_records)):
        closest = {"minimum_hull_aabb_lower_bound_m": math.inf, "sample": None, "spring_geom": None}
        for record, q in zip(records, path):
            data.qpos[:] = model.qpos0
            data.qpos[qpos_ids] = q
            mujoco.mj_forward(model, data)
            bottle_points = world_shape(model, data, bottle_geom, shapes[bottle_geom])[0]
            for spring_geom in spring_mesh_geoms:
                spring_points = world_shape(model, data, spring_geom, shapes[spring_geom])[0]
                broadphase = aabb_gap_bounds(spring_points.min(0), spring_points.max(0),
                                             bottle_points.min(0), bottle_points.max(0))
                if broadphase < closest["minimum_hull_aabb_lower_bound_m"]:
                    closest = {"minimum_hull_aabb_lower_bound_m": broadphase,
                               "sample": int(record["sample"]), "fraction": float(record["fraction"]),
                               "spring_geom": int(spring_geom), "spring_geom_name": name(model, mujoco.mjtObj.mjOBJ_GEOM, spring_geom),
                               "spring_body_name": body_name(model, spring_geom)}
        exact_check = None
        if closest["spring_geom"] is not None:
            data.qpos[:] = model.qpos0
            data.qpos[qpos_ids] = path[closest["sample"]]
            mujoco.mj_forward(model, data)
            exact_check = source_stl_cylinder_check(model, data, closest["spring_geom"], bottle_geom, source_triangles_local)
        source_mesh_checks[label] = {
            "minimum_hull_aabb_lower_bound_m": closest["minimum_hull_aabb_lower_bound_m"],
            "closest_sample": closest,
            "source_triangle_distance_at_closest_hull_sample": exact_check,
            "source_check_scope": "exact source STL SDF is run at the tightest collision-hull AABB sample; swept convex-hull qualification bounds the full interval because the STL surface is contained by its collision hull",
        }

    legacy_minimum_reconciliation = None
    previous_static_hash = None
    if previous_static_path is not None:
        previous_static_hash = sha256(previous_static_path)
        previous_static = json.loads(previous_static_path.read_text(encoding="utf-8"))
        if (previous_static.get("input_model", {}).get("sha256") != model_hash
                or previous_static.get("input_saved_approach", {}).get("sha256") != sample_hash):
            raise ValueError("prior static minimum must use the exact same model and 101 saved arm samples")
        old_witness = previous_static["geometry"]["robot_bottle_first_minimum"]
        witness_sample = int(old_witness["sample"])
        if not 0 <= witness_sample < len(original_path):
            raise ValueError("prior static minimum sample is outside the saved 101-sample path")
        prior_pair_names = str(old_witness["geom_pair"]).split("<->")
        if len(prior_pair_names) != 2 or "bottle_body" not in prior_pair_names[1].strip():
            raise ValueError("prior static witness did not identify the expected spring-link/bottle body pair")
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[qpos_ids] = original_path[witness_sample]
        mujoco.mj_forward(model, data)
        contact_map = contact_map_for_data(data)
        world_shapes = all_world_shapes(model, data, shapes)
        proxy_clearance = geom_pair_lower_bound(model, right_spring_geom, bottle_geom, world_shapes, contact_map)
        legacy_source_check = source_stl_cylinder_check(
            model, data, right_spring_geom, bottle_geom, source_triangles_local
        )
        legacy_minimum_reconciliation = {
            "prior_static_result_path": str(previous_static_path),
            "prior_static_result_sha256": previous_static_hash,
            "prior_static_run_id": previous_static.get("run_id"),
            "saved_arm_path_sha256": sample_hash,
            "sample_index": witness_sample,
            "fraction": float(rows[witness_sample]["fraction"]),
            "body_pair": [body_name(model, right_spring_geom), body_name(model, bottle_geom)],
            "geom_pair": [name(model, mujoco.mjtObj.mjOBJ_GEOM, right_spring_geom),
                          name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom)],
            "prior_conservative_world_aabb_lower_bound_mm": float(
                previous_static["geometry"]["robot_bottle_min_clearance_mm"]
            ),
            "independent_current_compiled_proxy_measurement": {
                "clearance_lower_bound_m": float(proxy_clearance["clearance_m"]),
                "clearance_lower_bound_mm": float(proxy_clearance["clearance_m"] * 1000.0),
                "method": proxy_clearance["method"],
                "separating_axis_world": proxy_clearance.get("axis"),
                "mujoco_geom_distance_m": float(mujoco.mj_geomDistance(
                    model, data, right_spring_geom, bottle_geom, 0.1, None
                )),
                "active_contact": bool(contact_map.get(pair_key(right_spring_geom, bottle_geom))),
            },
            "source_stl_surface_distance_bound": legacy_source_check,
            "interpretation": "The prior 0.005701 mm value is a conservative world-AABB lower bound, not a measured source-surface gap. The independent source STL distance bound determines whether the visual source surface is actually within 2 mm; the compiled collision-proxy clearance remains separately recorded.",
        }

    def sweep_by_group(path):
        return {
            group: path_sweep(model, data, shapes,
                              {"robot_geoms": pair_sets["robot_geoms"], group: pair_sets[group]},
                              qpos_ids, path, joint_ids)
            for group in ("robot_bottle", "robot_bottle_nonpad", "robot_table", "robot_self")
        }

    baseline_sweep = sweep_by_group(original_path)
    corrected_sweep = sweep_by_group(corrected_path)
    original_contact_order = episode_contact_order(contact_trace_path)
    grip_history = prior["stages"].get("bilateral_grasp_hold", {})

    def trajectory_rows(path):
        return [
            {"sample_index": index, "fraction": float(index / (len(path) - 1)),
             "right_arm_qpos_rad": np.asarray(q, dtype=float).tolist()}
            for index, q in enumerate(path)
        ]

    result = {
        "schema": "robotsim.issue46.robotiq_static_approach_qualification.v1",
        "classification": "SIMULATION_ONLY_DIAGNOSTIC_STATIC_AND_SWEPT_GEOMETRY",
        "physics_steps": 0,
        "mj_forward_calls": int(q_pre["mj_forward_calls"] + 404
                                 + sum(item["mj_forward_calls"] for item in baseline_sweep.values())
                                 + sum(item["mj_forward_calls"] for item in corrected_sweep.values())
                                 + 2 + (1 if previous_static_path is not None else 0)),
        "run_id": str(uuid.uuid4()),
        "input_identity": {
            "diagnostic_model_path": str(model_path), "diagnostic_model_sha256": model_hash,
            "historical_result_path": str(history_path), "historical_result_sha256": result_hash,
            "historical_approach_samples_path": str(samples_path), "historical_approach_samples_sha256": sample_hash,
            "previous_static_result_path": str(previous_static_path) if previous_static_path else None,
            "previous_static_result_sha256": previous_static_hash,
            "x2_commit": identity["x2_checkout"]["head"], "x2_mjcf_sha256": identity["x2_mjcf_sha256"],
            "robotiq_commit": identity["menagerie_checkout"]["head"], "robotiq_xml_sha256": identity["robotiq_xml_sha256"],
            "bottle_mass_kg": float(model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")]),
            "bottle_root_qpos0": model.qpos0[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")]):
                                                   int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "m0_bottle_free")]) + 7].tolist(),
            "source_spring_mesh": source_mesh_identity,
        },
        "historical_exception_audit": {
            "head_torso_exclusion": {"present": tuple(sorted(("head_pitch_link", "torso_link"))) in exclusions,
                                      "classification": "DIAGNOSTIC_ONLY; original pair exclusion remains in input model"},
            "wrist_mount": {"translation_m": identity["official_x2_tool_mount"]["xyz"],
                            "rpy": identity["official_x2_tool_mount"]["rpy"],
                            "classification": "SIMULATION_DERIVED_ADAPTER; not manufacturer Robotiq mount data"},
            "coupler_margin_rad": prior["adapter"]["coupler_limit_activation_margin_simulation_derived_rad"],
            "coupler_classification": "SIMULATION_DERIVED; original source-faithful coupled stop was not dynamically qualified",
            "collision_exclusions_in_input": [list(pair) for pair in sorted(exclusions)],
            "new_exclusions_or_parameter_changes": False,
        },
        "baseline_20mm": {
            "right_arm_joint_names": list(ARM),
            "path_interpolation": "linear in the seven source right-arm joint coordinates; static witness only, not a timed trajectory",
            "joint_trajectory": trajectory_rows(original_path),
            "pregrasp_qpos_rad": original_path[0].tolist(), "approach_qpos_rad": original_path[-1].tolist(),
            "minimum_discrete": {group: min((row["clearance"][group]["minimum_clearance_lower_bound_m"]
                                              for row in baseline_records if row["clearance"][group]["minimum_clearance_lower_bound_m"] is not None),
                                             default=None)
                                 for group in ("robot_bottle", "robot_table", "robot_self")},
            "limit_violations": baseline_limit_violations,
            "swept": baseline_sweep,
            "source_spring_to_bottle_checks": source_mesh_checks["baseline"],
            "internal_mechanism_clearance": summarize_internal_clearance(baseline_records),
        },
        "candidate_25mm_retreat": {
            "ik": q_pre,
            "right_arm_joint_names": list(ARM),
            "path_interpolation": "linear in the seven source right-arm joint coordinates; static witness only, not a timed trajectory",
            "joint_trajectory": trajectory_rows(corrected_path),
            "final_grasp_qpos_preserved": bool(np.allclose(corrected_path[-1], q_grasp, atol=1e-12)),
            "path_sample_count": len(corrected_path),
            "minimum_discrete": {group: min((row["clearance"][group]["minimum_clearance_lower_bound_m"]
                                              for row in corrected_records if row["clearance"][group]["minimum_clearance_lower_bound_m"] is not None),
                                             default=None)
                                 for group in ("robot_bottle", "robot_table", "robot_self")},
            "limit_violations": corrected_limit_violations,
            "swept": corrected_sweep,
            "source_spring_to_bottle_checks": source_mesh_checks["corrected"],
            "internal_mechanism_clearance": summarize_internal_clearance(corrected_records),
        },
        "historical_pad_first_evidence": {
            **original_contact_order,
            "hold_contact_fraction": grip_history.get("bilateral_pad_contact_fraction"),
            "hold_slip_m": grip_history.get("bottle_hold_slip_m"),
            "note": "Historical physical trace only; changed retreat candidate has no new dynamic contact run.",
        },
        "legacy_minimum_reconciliation": legacy_minimum_reconciliation,
        "limitations": [
            "Static states use mj_forward only; no mj_step was called.",
            f"Swept clearance uses a conservative revolute-axis arc-length bound, recursive midpoint subdivision, depth {SWEEP_MAX_DEPTH}, and a {SWEEP_NODE_BUDGET}-node budget; unresolved intervals fail closed.",
            "The source STL check bounds source triangle distance to the canonical bottle-body cylinder independently of the compiled convex hull; the active compiled convex-hull collision gate remains separately required.",
            "The historical model retains its simulation-only head/torso exclusion, derived Robotiq mount, and 0.001 rad coupler margin. This qualification does not make these hardware-equivalent.",
            "The new candidate preserves the historical final grasp joint configuration. Historical pad contact is not evidence of physical success for the changed retreat path.",
        ],
    }
    gates = {
        "candidate_pregrasp_ik": bool(q_pre["ik_pass"] and q_pre["source_joint_limits_pass"]),
        "candidate_all_source_position_limits": not corrected_limit_violations,
        "candidate_robot_bottle_clearance_ge_2mm": corrected_sweep["robot_bottle"].get("pass", False),
        "candidate_nonpad_robot_bottle_clearance_ge_2mm": corrected_sweep["robot_bottle_nonpad"].get("pass", False),
        "candidate_robot_table_clearance_ge_2mm": corrected_sweep["robot_table"].get("pass", False),
        "candidate_robot_self_clearance_ge_2mm": corrected_sweep["robot_self"].get("pass", False),
        "candidate_internal_mechanism_clearance_ge_2mm": (
            summarize_internal_clearance(corrected_records).get("worst_pair", {}).get("minimum_clearance_lower_bound_m", -math.inf)
            >= CLEARANCE_M
        ),
        "candidate_internal_mechanism_clearance_invariant_under_arm_motion": summarize_internal_clearance(corrected_records).get("subthreshold_pairs_invariant_within_1um", False),
        "candidate_source_mesh_clearance_ge_2mm": (
            source_mesh_checks["corrected"]["minimum_hull_aabb_lower_bound_m"] >= CLEARANCE_M
            or source_mesh_checks["corrected"]["source_triangle_distance_at_closest_hull_sample"] is not None
            and source_mesh_checks["corrected"]["source_triangle_distance_at_closest_hull_sample"]["lower_bound_m"] >= CLEARANCE_M
        ),
        "candidate_historical_final_grasp_preserved": bool(np.allclose(corrected_path[-1], q_grasp, atol=1e-12)),
    }
    result["gates"] = gates
    result["status"] = "PASS_STATIC_SWEPT_QUALIFIED" if all(gates.values()) else "BLOCKED_STATIC_APPROACH_QUALIFICATION"
    result["first_failed_gates"] = [gate for gate, passed in gates.items() if not passed]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-model", required=True)
    parser.add_argument("--historical-result", required=True)
    parser.add_argument("--historical-approach-samples", required=True)
    parser.add_argument("--contact-trace", required=True)
    parser.add_argument("--spring-link-stl", required=True)
    parser.add_argument("--previous-static-result")
    parser.add_argument("--retreat-m", type=float, default=0.025)
    parser.add_argument("--output-parent", default=str(REVIEWS / "issue-46"))
    args = parser.parse_args()
    inputs = {
        "diagnostic model": Path(args.diagnostic_model),
        "historical result": Path(args.historical_result),
        "historical approach samples": Path(args.historical_approach_samples),
        "historical contact trace": Path(args.contact_trace),
        "source spring-link STL": Path(args.spring_link_stl),
    }
    if args.previous_static_result:
        inputs["previous static result"] = Path(args.previous_static_result)
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    result = qualify(inputs["diagnostic model"], inputs["historical result"],
                     inputs["historical approach samples"], inputs["historical contact trace"],
                     args.retreat_m, inputs["source spring-link STL"], inputs.get("previous static result"))
    output_parent = pg.validate_output_path(args.output_parent, authorized_roots=(str(REVIEWS),),
                                            require_exists=True, expected_kind="directory", environment="wsl")
    output = pg.create_unique_output_directory(output_parent.wsl_path, prefix="robotiq-arm-approach-qualified",
                                               authorized_roots=(str(REVIEWS),), environment="wsl")
    result["output"] = output.as_dict()
    result["run_id"] = str(uuid.uuid4())
    result["runner_path"] = str(Path(__file__).resolve())
    result["runner_sha256"] = sha256(Path(__file__).resolve())
    report = [
        "# X2 + Robotiq 2F-85 static approach qualification",
        "",
        f"Run ID: {result['run_id']}",
        f"Decision: {result['status']}",
        f"Physics steps: {result['physics_steps']}",
        f"Candidate retreat: {args.retreat_m * 1000:.1f} mm",
        f"Failed gates: {result['first_failed_gates']}",
        f"Baseline 20 mm min discrete clearances: {result['baseline_20mm']['minimum_discrete']}",
        f"Candidate swept group results: { {group: {'pass': item.get('pass'), 'minimum_swept_clearance_lower_bound_m': item.get('minimum_swept_clearance_lower_bound_m')} for group, item in result['candidate_25mm_retreat']['swept'].items()} }",
        f"Minimum internal spring/pad distance: {result['candidate_25mm_retreat']['internal_mechanism_clearance']}",
        f"Legacy sample-20 0.005701 mm reconciliation: {result.get('legacy_minimum_reconciliation')}",
        "",
        "The gripper, bottle, table, collision masks, couplings, and actuator parameters are unchanged. The candidate keeps the historical final grasp joint vector and changes only the approach retreat target.",
        "The diagnostic model retains its recorded simulation-only exceptions. No physical outcome is claimed for this statically evaluated candidate.",
    ]
    reproduce = [
        f"cd {ROOT}",
        f"PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/qualify_robotiq_static_approach.py --diagnostic-model {inputs['diagnostic model']} --historical-result {inputs['historical result']} --historical-approach-samples {inputs['historical approach samples']} --contact-trace {inputs['historical contact trace']} --spring-link-stl {inputs['source spring-link STL']} --previous-static-result {inputs.get('previous static result', '')} --retreat-m {args.retreat_m}",
    ]
    pg.write_new_text(output.native_path / "result.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPORT.md", "\n".join(report) + "\n", authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.write_new_text(output.native_path / "REPRODUCE.md", "\n".join(reproduce) + "\n", authorized_roots=(str(REVIEWS),), environment="wsl")
    manifest = pg.write_artifact_manifest(output.native_path, authorized_roots=(str(REVIEWS),), environment="wsl")
    pg.verify_artifacts(manifest["artifacts"], authorized_roots=(str(REVIEWS),), environment="wsl")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "output": output.as_dict(),
                      "gates": result["gates"], "manifest_count": len(manifest["artifacts"])}, indent=2))
    return 0 if result["status"] == "PASS_STATIC_SWEPT_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
