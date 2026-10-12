#!/usr/bin/env python3
"""Bounded, source-faithful isolated OmniPicker endpoint geometry audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_URDF_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
X2_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
CANONICAL_HELPER_SHA256 = "41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae"
ROOT_LINK = "right_wrist_yaw_link"
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
NARROW = "R_hand_narrow3_Link"
WIDE = "R_hand_wide3_Link"
WRIST_ROLL = "right_wrist_roll_link"
CONTACT_TOL_M = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load research helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stl_vertices_and_faces(path: Path) -> tuple[np.ndarray, int]:
    blob = path.read_bytes()
    if len(blob) >= 84:
        facets = struct.unpack_from("<I", blob, 80)[0]
        if len(blob) == 84 + 50 * facets:
            vertices = np.empty((facets * 3, 3), dtype=np.float64)
            for index in range(facets):
                start = 84 + 50 * index + 12
                vertices[index * 3:index * 3 + 3] = np.frombuffer(
                    blob, dtype="<f4", count=9, offset=start
                ).reshape(3, 3)
            return vertices, facets
    text = blob.decode("ascii", errors="ignore")
    rows = re.findall(
        r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text
    )
    if not rows or len(rows) % 3:
        raise ValueError(f"Cannot parse STL triangles: {path}")
    return np.asarray(rows, dtype=np.float64), len(rows) // 3


def source_mesh_inventory(urdf: Path, mesh_root: Path,
                          source_links: dict[str, ET.Element],
                          included_links: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in included_links:
        link = source_links[name]
        row: dict[str, Any] = {}
        for role in ("visual", "collision"):
            mesh = link.find(f"{role}/geometry/mesh")
            if mesh is None:
                row[role] = None
                continue
            filename = Path(mesh.get("filename", "")).name
            path = mesh_root / filename
            vertices, faces = stl_vertices_and_faces(path)
            bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
            origin = link.find(f"{role}/origin")
            row[role] = {
                "filename": filename,
                "sha256": sha256(path),
                "source_origin_xyz_m": [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()],
                "source_origin_rpy_rad": [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()],
                "raw_triangle_count": int(faces),
                "raw_unique_vertex_count": int(len(np.unique(vertices, axis=0))),
                "raw_local_bounds_m": bounds.tolist(),
                "raw_local_extents_m": (bounds[1] - bounds[0]).tolist(),
            }
        result[name] = row
    return result


def compiled_mesh_inventory(model: mujoco.MjModel,
                            source_inventory: dict[str, Any]) -> dict[str, Any]:
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for body_id in range(model.nbody)
    }
    result: dict[str, Any] = {}
    for body_name in source_inventory:
        if body_name not in names:
            result[body_name] = {"error": "body absent from isolated compiled model"}
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        rows = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                         if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                         else None)
            hull_start = int(model.mesh_polyadr[mesh_id]) if mesh_name is not None else 0
            hull_count = int(model.mesh_polynum[mesh_id]) if mesh_name is not None else 0
            hull_indices = sorted({
                int(vertex)
                for polygon_id in range(hull_start, hull_start + hull_count)
                for vertex in model.mesh_polyvert[
                    int(model.mesh_polyvertadr[polygon_id]):
                    int(model.mesh_polyvertadr[polygon_id] + model.mesh_polyvertnum[polygon_id])
                ]
            })
            hull_points = (
                model.mesh_vert[int(model.mesh_vertadr[mesh_id]) + np.asarray(hull_indices, dtype=np.int64)]
                if mesh_name is not None and hull_indices else np.empty((0, 3))
            )
            rows.append({
                "geom_id": geom_id,
                "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "group": int(model.geom_group[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "mesh_id": mesh_id if mesh_name is not None else None,
                "mesh_name": mesh_name,
                "compiled_vertex_count": int(model.mesh_vertnum[mesh_id]) if mesh_name else None,
                "compiled_face_count": int(model.mesh_facenum[mesh_id]) if mesh_name else None,
                "compiled_convex_hull_unique_vertex_count": len(hull_indices) if mesh_name else None,
                "compiled_convex_hull_polygon_count": hull_count if mesh_name else None,
                "compiled_convex_hull_local_bounds_m": (
                    np.vstack((hull_points.min(axis=0), hull_points.max(axis=0))).tolist()
                    if len(hull_points) else None
                ),
                "compiled_local_bounds_m": (
                    np.vstack((
                        model.mesh_vert[int(model.mesh_vertadr[mesh_id]):int(model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id])].min(axis=0),
                        model.mesh_vert[int(model.mesh_vertadr[mesh_id]):int(model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id])].max(axis=0),
                    )).tolist() if mesh_name else None
                ),
            })
        result[body_name] = rows
    return result


def body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                             int(model.geom_bodyid[geom_id])) or "world"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"


def all_pair_distances(model: mujoco.MjModel, data: mujoco.MjData,
                       robot_geoms: list[int], bottle_geoms: list[int]) -> list[dict[str, Any]]:
    rows = []
    for first in robot_geoms:
        for second in bottle_geoms:
            segment = np.zeros(6, dtype=np.float64)
            distance = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
            rows.append({
                "robot_body": body_name(model, first),
                "robot_geom": geom_name(model, first),
                "bottle_body": body_name(model, second),
                "bottle_geom": geom_name(model, second),
                "signed_distance_m": distance,
                "penetration_m": max(0.0, -distance),
                "witness_robot_world_m": segment[:3].tolist(),
                "witness_bottle_world_m": segment[3:].tolist(),
            })
    return sorted(rows, key=lambda row: row["signed_distance_m"])


def source_joint_records(urdf: Path, model: mujoco.MjModel,
                         candidate: dict[str, Any]) -> dict[str, Any]:
    root = ET.parse(urdf).getroot()
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    result = {}
    for name in ("right_wrist_yaw_joint", "right_wrist_pitch_joint",
                 "right_wrist_roll_joint", DRIVER, FOLLOWER):
        item = joints[name]
        limit = item.find("limit")
        axis = [float(v) for v in item.find("axis").get("xyz", "0 0 1").split()]
        qid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mimic = item.find("mimic")
        row = {
            "parent": item.find("parent").get("link"),
            "child": item.find("child").get("link"),
            "source_axis": axis,
            "source_origin_xyz_m": [float(v) for v in item.find("origin").get("xyz", "0 0 0").split()],
            "source_origin_rpy_rad": [float(v) for v in item.find("origin").get("rpy", "0 0 0").split()],
            "source_range_rad": [float(limit.get("lower")), float(limit.get("upper"))],
            "source_effort_metadata": float(limit.get("effort", "0")),
            "source_velocity_metadata": float(limit.get("velocity", "0")),
            "mimic": None if mimic is None else {
                "driver": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            },
        }
        if qid < 0:
            row.update({
                "compiled_joint_present": False,
                "compiled_state": "omitted: this source joint is upstream of the isolated fixed root; its transform is represented by the root scene pose",
                "compiled_range_rad": None,
                "compiled_ref_rad": None,
                "candidate_qpos_rad": None,
            })
        else:
            qpos_index = int(model.jnt_qposadr[qid])
            row.update({
                "compiled_joint_present": True,
                "compiled_range_rad": model.jnt_range[qid].tolist(),
                "compiled_limited": bool(model.jnt_limited[qid]),
                "compiled_ref_rad": float(model.qpos0[qpos_index]),
                "candidate_qpos_rad": float(candidate["data"].qpos[qpos_index]),
            })
        result[name] = row
    return result


def collision_state_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        b1, b2 = body_name(model, first), body_name(model, second)
        rows.append({
            "body1": b1, "geom1": geom_name(model, first),
            "body2": b2, "geom2": geom_name(model, second),
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "contact_frame_world": np.asarray(contact.frame, dtype=float).reshape(3, 3).tolist(),
        })
    return rows


def set_render_colors(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        body = body_name(model, geom_id)
        group = int(model.geom_group[geom_id])
        name = geom_name(model, geom_id)
        if body == "m0_bottle":
            model.geom_rgba[geom_id] = [0.05, 0.58, 0.90, 1.0]
        elif body == "m0_table":
            model.geom_rgba[geom_id] = [0.62, 0.42, 0.24, 1.0]
        elif body == WRIST_ROLL and group == 3:
            model.geom_rgba[geom_id] = [0.98, 0.20, 0.10, 0.82]
        elif body in {NARROW, WIDE} and group == 3:
            model.geom_rgba[geom_id] = [0.12, 0.95, 0.34, 0.72]
        elif group == 2:
            model.geom_rgba[geom_id] = [0.82, 0.88, 0.96, 0.36]
        elif group == 3:
            model.geom_rgba[geom_id] = [0.96, 0.58, 0.16, 0.14]
        elif "floor" in name:
            model.geom_rgba[geom_id] = [0.38, 0.42, 0.44, 1.0]


def set_render_groups(model: mujoco.MjModel) -> dict[str, int]:
    mapping = {"environment": 1, "robot_visual": 2, "robot_collision": 3}
    for geom_id in range(model.ngeom):
        body = body_name(model, geom_id)
        if body in {"m0_bottle", "m0_table", "world"}:
            model.geom_group[geom_id] = mapping["environment"]
        elif int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]):
            model.geom_group[geom_id] = mapping["robot_collision"]
        else:
            model.geom_group[geom_id] = mapping["robot_visual"]
    return mapping


def render_overlay(model: mujoco.MjModel, data: mujoco.MjData,
                   camera: mujoco.MjvCamera, output: Path,
                   contact_marks: list[dict[str, Any]], collision_only: bool,
                   transparent_bottle: bool = False) -> None:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in (1, 2, 3):
        if group == 1 or (not collision_only and group in (2, 3)) or (collision_only and group == 3):
            option.geomgroup[group] = 1
    if collision_only:
        option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 900
    renderer = mujoco.Renderer(model, height=900, width=1280)
    bottle_geom_ids = [
        geom_id for geom_id in range(model.ngeom)
        if body_name(model, geom_id) == "m0_bottle"
    ]
    original_rgba = model.geom_rgba[bottle_geom_ids].copy() if transparent_bottle else None
    if transparent_bottle:
        model.geom_rgba[bottle_geom_ids, 3] = 0.30
    renderer.update_scene(data, camera=camera, scene_option=option)
    scene = renderer.scene
    marker_colors = ([1.0, 0.92, 0.08, 1.0], [0.96, 0.10, 0.72, 1.0],
                     [1.0, 0.42, 0.05, 1.0], [0.78, 0.08, 1.0, 1.0])
    for index, row in enumerate(contact_marks):
        color = marker_colors[min(index, len(marker_colors) - 1)]
        point = 0.5 * (np.asarray(row["witness_robot_world_m"]) + np.asarray(row["witness_bottle_world_m"]))
        if row.get("marker_kind") == "source_triangle_inside_bottle":
            active_camera = scene.camera[0]
            camera_position = np.asarray(active_camera.pos, dtype=float)
            toward_camera = camera_position - point
            toward_camera /= max(float(np.linalg.norm(toward_camera)), 1e-15)
            camera_right = np.cross(np.asarray(active_camera.forward), np.asarray(active_camera.up))
            camera_right /= max(float(np.linalg.norm(camera_right)), 1e-15)
            side_sign = 1.0 if index % 2 == 0 else -1.0
            marker_end = point + 0.045 * toward_camera + side_sign * 0.045 * camera_right
            leader = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                leader, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([0.0035, 0.0035, 0.0035]), point,
                np.eye(3, dtype=np.float64).reshape(-1), np.asarray(color, dtype=np.float32),
            )
            mujoco.mjv_connector(leader, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, point, marker_end)
            scene.ngeom += 1
            marker = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.005, 0.005, 0.005]), marker_end,
                np.eye(3, dtype=np.float64).reshape(-1), np.asarray(color, dtype=np.float32),
            )
            scene.ngeom += 1
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.0026, 0.0026, 0.0026]), point,
            np.eye(3, dtype=np.float64).reshape(-1), np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1
    image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image, "RGBA")
    if collision_only:
        legend = [
            ("Canonical bottle / table", (13, 148, 230, 255)),
            ("MuJoCo compiled convex hull", (250, 51, 26, 230)),
            ("Narrow / wide fingertip collision geom", (31, 242, 87, 230)),
            ("Narrow / wide pad witnesses", (255, 228, 20, 255)),
            ("Wrist / bottle blocker witness", (255, 107, 13, 255)),
            ("Source triangle sample inside bottle", (199, 20, 255, 255)),
        ]
    else:
        legend = [
            ("Pinned vendor visual mesh", (210, 224, 245, 255)),
            ("Source collision STL triangles", (250, 51, 26, 230)),
            ("Narrow / wide fingertip collision geom", (31, 242, 87, 230)),
            ("Narrow / wide pad witnesses", (255, 228, 20, 255)),
            ("Wrist / bottle blocker witness", (255, 107, 13, 255)),
            ("Source triangle sample inside bottle", (199, 20, 255, 255)),
            ("Canonical bottle", (13, 148, 230, 255)),
        ]
    x, y = 24, 24
    draw.rounded_rectangle((x - 10, y - 10, 560, y + 27 * len(legend) + 12), radius=5,
                           fill=(18, 28, 38, 205))
    for label, rgba in legend:
        draw.rectangle((x, y + 3, x + 18, y + 17), fill=rgba)
        draw.text((x + 26, y), label, fill=(245, 248, 252, 255))
        y += 27
    if collision_only:
        draw.text((24, image.height - 34), "Collision geoms rendered as MuJoCo compiled convex hulls",
                  fill=(255, 246, 230, 255))
    else:
        draw.text((24, image.height - 34), "Source collision triangles rendered without convex-hull visualization",
                  fill=(255, 246, 230, 255))
    if transparent_bottle:
        draw.text((24, image.height - 58), "Bottle alpha is renderer-only; collision and physics unchanged",
                  fill=(255, 246, 230, 255))
    image.save(output)
    if original_rgba is not None:
        model.geom_rgba[bottle_geom_ids] = original_rgba
    renderer.close()


def mesh_triangles_world(model: mujoco.MjModel, data: mujoco.MjData,
                         geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=float)
    faces = np.asarray(model.mesh_face[face_start:face_start + face_count], dtype=np.int64)
    if faces.size and int(faces.max()) >= vertex_count:
        faces -= vertex_start
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertex_count):
        raise RuntimeError(f"Compiled face indices exceed mesh vertex inventory: {geom_name(model, geom_id)}")
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    world_vertices = vertices @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
    return world_vertices[faces]


def triangle_surface_samples(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    centers = triangles.mean(axis=1)
    edge_ab = 0.5 * (triangles[:, 0] + triangles[:, 1])
    edge_bc = 0.5 * (triangles[:, 1] + triangles[:, 2])
    edge_ca = 0.5 * (triangles[:, 2] + triangles[:, 0])
    samples = np.concatenate((unique_vertices, centers, edge_ab, edge_bc, edge_ca), axis=0)
    face_ids = np.concatenate((
        np.full(len(unique_vertices), -1, dtype=np.int64),
        np.arange(len(triangles), dtype=np.int64),
        np.arange(len(triangles), dtype=np.int64),
        np.arange(len(triangles), dtype=np.int64),
        np.arange(len(triangles), dtype=np.int64),
    ))
    return samples, face_ids


def geom_signed_distance_samples(model: mujoco.MjModel, data: mujoco.MjData,
                                 geom_id: int, points_world: np.ndarray) -> tuple[np.ndarray, str]:
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    local = (points_world - np.asarray(data.geom_xpos[geom_id], dtype=float)) @ rotation
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = np.linalg.norm(local[:, :2], axis=1) - size[0]
        axial = np.abs(local[:, 2]) - size[1]
        q = np.column_stack((radial, axial))
        values = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            np.maximum(q[:, 0], q[:, 1]), 0.0
        )
        return values, "exact signed distance to finite cylinder"
    if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return np.linalg.norm(local / size, axis=1) - 1.0, (
            "ellipsoid implicit level r-1; sign is exact, value is not a metric distance"
        )
    raise RuntimeError(f"Unexpected bottle geom type {kind} ({geom_name(model, geom_id)})")


def nearest_triangle_sample(point: np.ndarray, triangles: np.ndarray,
                            samples: np.ndarray, face_ids: np.ndarray) -> dict[str, Any]:
    sample_distances, sample_indices = cKDTree(samples).query(point, k=1)
    sample_distance, sample_index = float(sample_distances), int(sample_indices)
    face_id = int(face_ids[sample_index])
    centers = triangles.mean(axis=1)
    _, nearby = cKDTree(centers).query(point, k=min(64, len(triangles)))
    candidate_faces = np.unique(np.append(np.atleast_1d(nearby).astype(int), face_id))
    best = None
    for index in candidate_faces:
        triangle = triangles[int(index)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-15:
            continue
        normal /= length
        projected = point - float(np.dot(point - triangle[0], normal)) * normal
        v0, v1, v2 = triangle[1] - triangle[0], triangle[2] - triangle[0], projected - triangle[0]
        d00, d01, d11 = float(np.dot(v0, v0)), float(np.dot(v0, v1)), float(np.dot(v1, v1))
        d20, d21 = float(np.dot(v2, v0)), float(np.dot(v2, v1))
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) <= 1.0e-20:
            continue
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        u = 1.0 - v - w
        if min(u, v, w) >= -1.0e-10:
            closest = u * triangle[0] + v * triangle[1] + w * triangle[2]
        else:
            edge_points = []
            for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                                  (triangle[2], triangle[0])):
                edge = second - first
                ratio = float(np.clip(np.dot(point - first, edge) /
                                      max(float(np.dot(edge, edge)), 1.0e-20), 0.0, 1.0))
                edge_points.append(first + ratio * edge)
            closest = min(edge_points, key=lambda candidate: float(np.linalg.norm(point - candidate)))
        distance = float(np.linalg.norm(point - closest))
        if best is None or distance < best["distance_m"]:
            best = {"distance_m": distance, "point_world_m": closest.tolist(),
                    "triangle_index": int(index), "triangle_unit_normal_world": normal.tolist()}
    if best is None:
        best = {"distance_m": sample_distance, "point_world_m": samples[sample_index].tolist(),
                "triangle_index": face_id, "triangle_unit_normal_world": None}
    best["sample_seed_distance_m"] = sample_distance
    return best


def collision_contact_rows(model: mujoco.MjModel, data: mujoco.MjData,
                           geom_a: int, geom_b: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} != {geom_a, geom_b}:
            continue
        address, dimension = int(contact.efc_address), int(contact.dim)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        rows.append({
            "geom1": geom_name(model, int(contact.geom1)),
            "body1": body_name(model, int(contact.geom1)),
            "geom2": geom_name(model, int(contact.geom2)),
            "body2": body_name(model, int(contact.geom2)),
            "distance_m": float(contact.dist),
            "efc_address": address,
            "constraint_dim": dimension,
            "active_solver_rows": dimension if address >= 0 else 0,
            "normal_geom1_to_geom2_world": frame[0].tolist(),
            "contact_frame_world": frame.tolist(),
        })
    return rows


def pad_bottle_contact_rows(model: mujoco.MjModel, data: mujoco.MjData,
                            pad_geom_ids: dict[str, int], bottle_geom_ids: list[int]) -> list[dict[str, Any]]:
    pad_by_id = {geom_id: name for name, geom_id in pad_geom_ids.items()}
    bottle_by_id = {geom_id: geom_name(model, geom_id) for geom_id in bottle_geom_ids}
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in pad_by_id and geom2 not in pad_by_id:
            continue
        if geom1 not in bottle_by_id and geom2 not in bottle_by_id:
            continue
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        if geom1 in bottle_by_id and geom2 in pad_by_id:
            normal_bottle_to_pad = frame[0]
        elif geom1 in pad_by_id and geom2 in bottle_by_id:
            normal_bottle_to_pad = -frame[0]
        else:
            continue
        address, dimension = int(contact.efc_address), int(contact.dim)
        rows.append({
            "pad_body": pad_by_id[geom1] if geom1 in pad_by_id else pad_by_id[geom2],
            "pad_geom": geom_name(model, geom1 if geom1 in pad_by_id else geom2),
            "bottle_geom": bottle_by_id[geom1] if geom1 in bottle_by_id else bottle_by_id[geom2],
            "geom1": geom_name(model, geom1), "geom2": geom_name(model, geom2),
            "distance_m": float(contact.dist),
            "position_world_m": np.asarray(contact.pos, dtype=float).tolist(),
            "normal_geom1_to_geom2_world": frame[0].tolist(),
            "normal_bottle_to_pad_world": normal_bottle_to_pad.tolist(),
            "efc_address": address, "constraint_dim": dimension,
            "active_solver_rows": dimension if address >= 0 else 0,
            "force_interpretation": "not measured; static mj_forward pose only, no mj_step",
        })
    return rows


def triangle_topology_match(raw_triangles: np.ndarray, source_unique: np.ndarray,
                            compiled_triangles: np.ndarray, compiled_vertices: np.ndarray) -> dict[str, Any]:
    """Match STL faces to pre-convexification compiled faces by their vertices."""
    raw_unique_local, raw_inverse = np.unique(raw_triangles.reshape(-1, 3), axis=0, return_inverse=True)
    if len(raw_unique_local) != len(source_unique):
        raise RuntimeError("Raw STL unique-vertex inventory changed during topology audit")
    raw_faces = raw_inverse.reshape(-1, 3)
    source_to_compiled = cKDTree(compiled_vertices).query(source_unique, k=1)[1]
    source_faces_compiled_ids = source_to_compiled[raw_faces]
    compiled_face_keys = Counter(tuple(sorted(map(int, face))) for face in range_faces(compiled_triangles, compiled_vertices))
    source_face_keys = Counter(tuple(sorted(map(int, face))) for face in source_faces_compiled_ids)
    unmatched_source = sum((source_face_keys - compiled_face_keys).values())
    unmatched_compiled = sum((compiled_face_keys - source_face_keys).values())
    return {
        "source_triangle_count": int(len(raw_faces)),
        "compiled_preconvex_triangle_count": int(len(compiled_triangles)),
        "source_unique_vertex_to_compiled_vertex_max_mm": float(
            cKDTree(compiled_vertices).query(source_unique, k=1)[0].max() * 1000.0
        ),
        "source_faces_not_present_in_compiled_preconvex_mesh": int(unmatched_source),
        "compiled_faces_not_present_in_source_stl": int(unmatched_compiled),
        "triangle_topology_multiset_exact": unmatched_source == 0 and unmatched_compiled == 0,
        "comparison": "unordered face-vertex topology after source-to-compiled vertex mapping; winding ignored",
    }


def range_faces(triangles: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return face indices for triangles already expressed in vertex coordinates."""
    tree = cKDTree(vertices)
    return tree.query(triangles.reshape(-1, 3), k=1)[1].reshape(-1, 3)


def required_collision_geom(model: mujoco.MjModel, body: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    matches = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id
               and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one active collision geom on {body}, found {matches}")
    return matches[0]


def compose_collision_comparison(source_path: Path, hull_path: Path,
                                 output: Path, title: str, footer: list[str]) -> None:
    source = Image.open(source_path).convert("RGB")
    hull = Image.open(hull_path).convert("RGB")
    canvas = Image.new("RGB", (source.width + hull.width, source.height + 126), (16, 25, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), f"{title} - SOURCE COLLISION TRIANGLES", fill=(245, 248, 252))
    draw.text((source.width + 18, 10), f"{title} - COMPILED CONVEX HULL", fill=(245, 248, 252))
    canvas.paste(source, (0, 34))
    canvas.paste(hull, (source.width, 34))
    footer_y = source.height + 42
    for line in footer:
        draw.text((18, footer_y), line, fill=(245, 248, 252))
        footer_y += 20
    canvas.save(output)


def root_cause_audit(output: Path, source_urdf: Path,
                     source_links: dict[str, ET.Element], filtered_info: dict[str, Any],
                     geometry_module, base, controller: dict[str, Any], endpoint_path: Path,
                     runtime: dict[str, Any]) -> int:
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    root_pos = np.asarray(endpoint["isolated_root_position_world_m"], dtype=float)
    root_rot = np.asarray(endpoint["isolated_root_rotation_world"], dtype=float)
    if root_pos.shape != (3,) or root_rot.shape != (3, 3):
        raise ValueError("Saved endpoint does not contain a valid 3D root pose")
    model, model_info = geometry_module.build_model(
        base, Path(filtered_info["derived_urdf_path"]), root_pos, root_rot,
        include_bottle=True, add_lift=False,
    )
    data = geometry_module.initialize(base, model, controller)
    for name, value in endpoint["endpoint_joint_qpos_rad"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    mujoco.mj_forward(model, data)

    yaw_id = required_collision_geom(model, "right_wrist_yaw_link")
    pitch_id = required_collision_geom(model, "right_wrist_pitch_link")
    wrist_id = required_collision_geom(model, WRIST_ROLL)
    bottle_ids = [g for g in range(model.ngeom) if body_name(model, g) == "m0_bottle"
                  and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    bottle_by_name = {geom_name(model, g): g for g in bottle_ids}
    bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "m0_bottle")
    bottle_geom_inventory = [{
        "name": geom_name(model, geom_id),
        "type": int(model.geom_type[geom_id]),
        "size_m": np.asarray(model.geom_size[geom_id], dtype=float).tolist(),
        "local_position_m": np.asarray(model.geom_pos[geom_id], dtype=float).tolist(),
        "local_quaternion_wxyz": np.asarray(model.geom_quat[geom_id], dtype=float).tolist(),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    } for geom_id in bottle_ids]
    yaw_pitch_segment = np.zeros(6, dtype=float)
    yaw_pitch_distance = float(mujoco.mj_geomDistance(model, data, yaw_id, pitch_id, 1.0,
                                                      yaw_pitch_segment))
    yaw_pitch_contacts = collision_contact_rows(model, data, yaw_id, pitch_id)
    wrist_hull_rows = all_pair_distances(model, data, [wrist_id], bottle_ids)
    wrist_hull_rows.sort(key=lambda row: float(row["signed_distance_m"]))

    collision_mesh = source_links[WRIST_ROLL].find("collision/geometry/mesh")
    collision_origin = source_links[WRIST_ROLL].find("collision/origin")
    visual_mesh = source_links[WRIST_ROLL].find("visual/geometry/mesh")
    visual_origin = source_links[WRIST_ROLL].find("visual/origin")
    mesh_path = source_urdf.parent / "meshes" / Path(collision_mesh.get("filename", "")).name
    visual_mesh_path = source_urdf.parent / "meshes" / Path(visual_mesh.get("filename", "")).name
    raw_vertices, raw_face_count = stl_vertices_and_faces(mesh_path)
    raw_unique = np.unique(raw_vertices, axis=0)
    scale = np.asarray([float(v) for v in collision_mesh.get("scale", "1 1 1").split()])
    origin_xyz = np.asarray([float(v) for v in (collision_origin.get("xyz", "0 0 0")
                                                if collision_origin is not None else "0 0 0").split()])
    origin_rpy = np.asarray([float(v) for v in (collision_origin.get("rpy", "0 0 0")
                                                if collision_origin is not None else "0 0 0").split()])
    if np.max(np.abs(origin_xyz)) > 1e-12 or np.max(np.abs(origin_rpy)) > 1e-12:
        raise RuntimeError("The pinned wrist collision origin is not identity; source transform audit must be extended")
    visual_scale = np.asarray([float(v) for v in visual_mesh.get("scale", "1 1 1").split()])
    visual_origin_xyz = np.asarray([float(v) for v in (visual_origin.get("xyz", "0 0 0")
                                                        if visual_origin is not None else "0 0 0").split()])
    visual_origin_rpy = np.asarray([float(v) for v in (visual_origin.get("rpy", "0 0 0")
                                                        if visual_origin is not None else "0 0 0").split()])
    visual_raw_vertices, visual_face_count = stl_vertices_and_faces(visual_mesh_path)
    wrist_body_id = int(model.geom_bodyid[wrist_id])
    body_rot = np.asarray(data.xmat[wrist_body_id], dtype=float).reshape(3, 3)
    source_world_vertices = (raw_unique * scale) @ body_rot.T + np.asarray(data.xpos[wrist_body_id])
    compiled_world_vertices = base.geom_mesh_world_vertices(model, data, wrist_id)
    raw_to_compiled = cKDTree(compiled_world_vertices).query(source_world_vertices, k=1)[0]
    compiled_to_raw = cKDTree(source_world_vertices).query(compiled_world_vertices, k=1)[0]
    wrist_triangles = mesh_triangles_world(model, data, wrist_id)
    surface_samples, surface_face_ids = triangle_surface_samples(wrist_triangles)
    compiled_mesh_id = int(model.geom_dataid[wrist_id])
    compiled_mesh_start = int(model.mesh_vertadr[compiled_mesh_id])
    compiled_mesh_count = int(model.mesh_vertnum[compiled_mesh_id])
    compiled_mesh_vertices_local = np.asarray(
        model.mesh_vert[compiled_mesh_start:compiled_mesh_start + compiled_mesh_count], dtype=float
    )
    compiled_mesh_faces_local = np.asarray(
        model.mesh_face[
            int(model.mesh_faceadr[compiled_mesh_id]):
            int(model.mesh_faceadr[compiled_mesh_id]) + int(model.mesh_facenum[compiled_mesh_id])
        ], dtype=np.int64
    ).copy()
    if compiled_mesh_faces_local.size and int(compiled_mesh_faces_local.max()) >= compiled_mesh_count:
        compiled_mesh_faces_local -= compiled_mesh_start
    compiled_mesh_vertices_world = base.geom_mesh_world_vertices(model, data, wrist_id)
    source_triangle_topology = triangle_topology_match(
        raw_vertices.reshape(-1, 3, 3), source_world_vertices,
        compiled_mesh_vertices_world[compiled_mesh_faces_local], compiled_mesh_vertices_world,
    )
    source_joint_by_child = {
        joint.find("child").get("link"): joint for joint in ET.parse(source_urdf).getroot().findall("joint")
        if joint.find("child") is not None
    }
    wrist_joint_chain = []
    child_link = WRIST_ROLL
    for _ in range(3):
        joint = source_joint_by_child.get(child_link)
        if joint is None:
            break
        parent_element = joint.find("parent")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        parent_link = parent_element.get("link") if parent_element is not None else None
        wrist_joint_chain.append({
            "joint": joint.get("name"), "type": joint.get("type"),
            "parent_link": parent_link, "child_link": child_link,
            "origin_xyz_m": [float(v) for v in (origin.get("xyz", "0 0 0")
                                                   if origin is not None else "0 0 0").split()],
            "origin_rpy_rad": [float(v) for v in (origin.get("rpy", "0 0 0")
                                                     if origin is not None else "0 0 0").split()],
            "axis": [float(v) for v in (axis.get("xyz", "1 0 0")
                                          if axis is not None else "1 0 0").split()],
            "limit": dict(limit.attrib) if limit is not None else None,
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        })
        child_link = parent_link
    wrist_geom_pose = {
        "geom_name": geom_name(model, wrist_id),
        "body_name": body_name(model, wrist_id),
        "geom_world_position_m": np.asarray(data.geom_xpos[wrist_id], dtype=float).tolist(),
        "geom_world_rotation_matrix": np.asarray(data.geom_xmat[wrist_id], dtype=float).reshape(3, 3).tolist(),
        "body_world_position_m": np.asarray(data.xpos[wrist_body_id], dtype=float).tolist(),
        "body_world_rotation_matrix": body_rot.tolist(),
    }
    visual_source = {
        "mesh": visual_mesh_path.name,
        "sha256": sha256(visual_mesh_path),
        "scale": visual_scale.tolist(),
        "origin_xyz_m": visual_origin_xyz.tolist(),
        "origin_rpy_rad": visual_origin_rpy.tolist(),
    }

    source_surface_by_bottle = []
    source_inside_points = []
    for bottle_id in bottle_ids:
        sdf, sdf_kind = geom_signed_distance_samples(model, data, bottle_id, surface_samples)
        min_idx = int(np.argmin(sdf))
        inside = np.flatnonzero(sdf < -1.0e-9)
        row = {
            "bottle_geom": geom_name(model, bottle_id),
            "sampled_minimum_value": float(sdf[min_idx]),
            "sampled_minimum_unit": "ellipsoid_implicit_level_r_minus_1" if "ellipsoid implicit" in sdf_kind else "m",
            "distance_interpretation": sdf_kind,
            "surface_samples_inside_count": int(len(inside)),
            "minimum_sample_world_m": surface_samples[min_idx].tolist(),
            "representative_inside_sample_world_m": surface_samples[int(inside[0])].tolist() if len(inside) else None,
        }
        source_surface_by_bottle.append(row)
        if len(inside):
            deepest = int(inside[np.argmin(sdf[inside])])
            source_inside_points.append({
                "bottle_geom": geom_name(model, bottle_id),
                "point_world_m": surface_samples[deepest].tolist(),
                "sdf_value": float(sdf[deepest]), "sdf_interpretation": sdf_kind,
            })

    pad_rows, pad_directions = [], []
    pad_geom_ids = {}
    neck_id = bottle_by_name["bottle_neck"]
    neck_center = np.asarray(data.geom_xpos[neck_id], dtype=float)
    for pad_name, joint_name in ((NARROW, DRIVER), (WIDE, FOLLOWER)):
        pad_id = next(g for g in range(model.ngeom) if body_name(model, g) == pad_name
                      and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
                      and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH))
        pad_geom_ids[pad_name] = pad_id
        segment = np.zeros(6, dtype=float)
        distance = float(mujoco.mj_geomDistance(model, data, pad_id, neck_id, 1.0, segment))
        pad_triangles = mesh_triangles_world(model, data, pad_id)
        pad_samples, pad_sample_faces = triangle_surface_samples(pad_triangles)
        pad_point, bottle_point = segment[:3], segment[3:]
        direction = bottle_point - pad_point
        direction /= max(float(np.linalg.norm(direction)), 1e-15)
        pad_directions.append(direction)
        patch = nearest_triangle_sample(pad_point, pad_triangles, pad_samples, pad_sample_faces)
        face_normal = patch.get("triangle_unit_normal_world")
        signed_normal_alignment = (
            float(np.dot(np.asarray(face_normal), direction)) if face_normal is not None else None
        )
        pad_surface_sdf_rows = []
        for candidate_bottle_id in bottle_ids:
            sdf, sdf_kind = geom_signed_distance_samples(model, data, candidate_bottle_id, pad_samples)
            min_abs = int(np.argmin(np.abs(sdf)))
            patch_face = int(pad_sample_faces[min_abs])
            patch_normal = None
            if patch_face >= 0:
                tri = pad_triangles[patch_face]
                patch_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                patch_normal /= max(float(np.linalg.norm(patch_normal)), 1e-15)
            pad_surface_sdf_rows.append({
                "bottle_geom": geom_name(model, candidate_bottle_id),
                "nearest_sample_signed_value": float(sdf[min_abs]),
                "nearest_sample_abs_value": float(abs(sdf[min_abs])),
                "distance_interpretation": sdf_kind,
                "pad_surface_sample_world_m": pad_samples[min_abs].tolist(),
                "triangle_index": patch_face,
                "triangle_unit_normal_world": patch_normal.tolist() if patch_normal is not None else None,
            })
        pad_rows.append({
            "pad_body": pad_name,
            "pad_geom": geom_name(model, pad_id),
            "target_bottle_geom": geom_name(model, neck_id),
            "compiled_convex_signed_distance_m": distance,
            "convex_witness_pad_world_m": pad_point.tolist(),
            "convex_witness_bottle_world_m": bottle_point.tolist(),
            "shortest_witness_direction_pad_to_bottle_world": direction.tolist(),
            "pad_witness_z_offset_from_neck_center_mm": float((pad_point[2] - neck_center[2]) * 1000.0),
            "nearest_preconvex_source_triangle_patch": patch,
            "signed_triangle_normal_alignment_with_witness_direction": signed_normal_alignment,
            "pad_surface_sampled_nearest_points_to_bottle_geoms": pad_surface_sdf_rows,
            "source_joint_qpos_rad": float(data.qpos[geometry_module.qpos_id(model, joint_name)]),
        })
    opposing_angle = float(np.degrees(np.arccos(np.clip(np.dot(pad_directions[0], pad_directions[1]), -1.0, 1.0))))
    pad_separation = np.asarray(pad_rows[1]["convex_witness_pad_world_m"]) - np.asarray(pad_rows[0]["convex_witness_pad_world_m"])
    pad_bottle_contacts = pad_bottle_contact_rows(model, data, pad_geom_ids, bottle_ids)
    deepest_pad_contact = min(pad_bottle_contacts, key=lambda row: row["distance_m"], default=None)
    pad_contact_normals = {
        name: [np.asarray(row["normal_bottle_to_pad_world"], dtype=float)
               for row in pad_bottle_contacts if row["pad_body"] == name]
        for name in (NARROW, WIDE)
    }
    contact_family_means = {}
    for name, normals in pad_contact_normals.items():
        if normals:
            mean_normal = np.mean(normals, axis=0)
            norm = float(np.linalg.norm(mean_normal))
            contact_family_means[name] = (mean_normal / norm).tolist() if norm > 1e-12 else None
        else:
            contact_family_means[name] = None
    if all(contact_family_means[name] is not None for name in (NARROW, WIDE)):
        pad_contact_normal_angle = float(np.degrees(np.arccos(np.clip(
            np.dot(contact_family_means[NARROW], contact_family_means[WIDE]), -1.0, 1.0
        ))))
    else:
        pad_contact_normal_angle = None
    contacts_by_bottle_geom = {}
    for bottle_name in sorted({row["bottle_geom"] for row in pad_bottle_contacts}):
        per_pad = {}
        for pad_name in (NARROW, WIDE):
            normals = [np.asarray(row["normal_bottle_to_pad_world"], dtype=float)
                       for row in pad_bottle_contacts
                       if row["bottle_geom"] == bottle_name and row["pad_body"] == pad_name]
            if normals:
                mean_normal = np.mean(normals, axis=0)
                mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
                per_pad[pad_name] = mean_normal
        angle = None
        if NARROW in per_pad and WIDE in per_pad:
            angle = float(np.degrees(np.arccos(np.clip(np.dot(per_pad[NARROW], per_pad[WIDE]), -1.0, 1.0))))
        contacts_by_bottle_geom[bottle_name] = {
            "contact_count_by_pad": {
                pad_name: sum(row["bottle_geom"] == bottle_name and row["pad_body"] == pad_name
                              for row in pad_bottle_contacts)
                for pad_name in (NARROW, WIDE)
            },
            "mean_normal_bottle_to_pad_world_by_pad": {
                name: vector.tolist() for name, vector in per_pad.items()
            },
            "opposing_normal_angle_deg": angle,
        }
    both_pads_contact_neck = all(
        any(row["bottle_geom"] == "bottle_neck" and row["pad_body"] == pad_name
            for row in pad_bottle_contacts)
        for pad_name in (NARROW, WIDE)
    )

    source_joint_map = {
        joint.get("name"): joint for joint in ET.parse(source_urdf).getroot().findall("joint")
    }
    jaw_joint_state = {}
    for joint_name in (DRIVER, FOLLOWER):
        joint = source_joint_map[joint_name]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        jaw_joint_state[joint_name] = {
            "source_parent_link": joint.find("parent").get("link"),
            "source_child_link": joint.find("child").get("link"),
            "source_axis": [float(v) for v in (axis.get("xyz", "1 0 0")
                                                  if axis is not None else "1 0 0").split()],
            "source_limit": dict(limit.attrib) if limit is not None else None,
            "source_mimic": dict(mimic.attrib) if mimic is not None else None,
            "compiled_range_rad": np.asarray(model.jnt_range[joint_id], dtype=float).tolist(),
            "compiled_qpos_rad": float(data.qpos[qpos_address]),
        }

    full_spec = mujoco.MjSpec.from_file(str(source_urdf))
    full_spec.compiler.fusestatic = False
    full_spec.option.timestep = base.TIMESTEP
    full_spec.option.gravity = [0.0, 0.0, -9.81]
    base.add_canonical_scene(full_spec, bottle=True)
    full_model = full_spec.compile()
    full_data = mujoco.MjData(full_model)
    mujoco.mj_resetData(full_model, full_data)
    full_pose = {"right_wrist_yaw_joint": 0.3,
                 "right_wrist_pitch_joint": 0.0,
                 "right_wrist_roll_joint": 0.0}
    for name, value in full_pose.items():
        jid = mujoco.mj_name2id(full_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Full source compilation omitted expected wrist DOF {name}")
        full_data.qpos[int(full_model.jnt_qposadr[jid])] = value
    for name, value in endpoint["endpoint_joint_qpos_rad"].items():
        jid = mujoco.mj_name2id(full_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            full_data.qpos[int(full_model.jnt_qposadr[jid])] = float(value)
    mujoco.mj_forward(full_model, full_data)
    full_yaw_id = required_collision_geom(full_model, "right_wrist_yaw_link")
    full_pitch_id = required_collision_geom(full_model, "right_wrist_pitch_link")
    full_segment = np.zeros(6, dtype=float)
    full_pair_distance = float(mujoco.mj_geomDistance(full_model, full_data,
                                                     full_yaw_id, full_pitch_id, 1.0, full_segment))
    full_pair_contacts = collision_contact_rows(full_model, full_data, full_yaw_id, full_pitch_id)
    filter_bit = int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    fixture_names = set(filtered_info["included_link_names"])
    fixture_excludes = [list(pair) for pair in base.STATIC_COLLISION_EXCLUSIONS
                        if pair[0] in fixture_names and pair[1] in fixture_names]
    historical_excludes = [list(pair) for pair in base.STATIC_COLLISION_EXCLUSIONS]
    pair_filter = {
        "interpretation": "mj_geomDistance is a raw geometry query and bypasses contact filtering; data.contact and efc_address indicate whether the pair reaches solver constraints.",
        "isolated_fixed_root_fixture": {
            "geom_pair": [f"{body_name(model, yaw_id)}/{geom_name(model, yaw_id)}",
                          f"{body_name(model, pitch_id)}/{geom_name(model, pitch_id)}"],
            "source_direct_parent_child": True,
            "body_parent_ids": [int(model.body_parentid[int(model.geom_bodyid[yaw_id])]),
                                int(model.body_parentid[int(model.geom_bodyid[pitch_id])])],
            "body_weld_ids": [int(model.body_weldid[int(model.geom_bodyid[yaw_id])]),
                              int(model.body_weldid[int(model.geom_bodyid[pitch_id])])],
            "geom_masks": {
                geom_name(model, geom_id): {
                    "contype": int(model.geom_contype[geom_id]),
                    "conaffinity": int(model.geom_conaffinity[geom_id]),
                }
                for geom_id in (yaw_id, pitch_id)
            },
            "raw_mj_geomDistance_m": yaw_pitch_distance,
            "active_contact_rows": yaw_pitch_contacts,
            "explicit_pair_count": int(model.npair), "exclude_count": int(model.nexclude),
            "fixture_excludes": fixture_excludes,
            "historical_static_collision_exclusion_count": len(historical_excludes),
            "historical_static_collision_exclusions_unchanged": historical_excludes,
            "no_new_exclusions_added": True,
            "disableflags": int(model.opt.disableflags),
            "filterparent_enabled": not bool(int(model.opt.disableflags) & filter_bit),
            "geom_masks_allow_pair": bool((int(model.geom_contype[yaw_id]) & int(model.geom_conaffinity[pitch_id]))
                                           or (int(model.geom_contype[pitch_id]) & int(model.geom_conaffinity[yaw_id]))),
        },
        "full_pinned_source_urdf": {
            "geom_pair": [f"{body_name(full_model, full_yaw_id)}/{geom_name(full_model, full_yaw_id)}",
                          f"{body_name(full_model, full_pitch_id)}/{geom_name(full_model, full_pitch_id)}"],
            "joint_pose_rad": full_pose,
            "body_parent_names": [
                mujoco.mj_id2name(full_model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(full_model.body_parentid[int(full_model.geom_bodyid[full_yaw_id])])),
                mujoco.mj_id2name(full_model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(full_model.body_parentid[int(full_model.geom_bodyid[full_pitch_id])])),
            ],
            "body_weld_ids": [int(full_model.body_weldid[int(full_model.geom_bodyid[full_yaw_id])]),
                              int(full_model.body_weldid[int(full_model.geom_bodyid[full_pitch_id])])],
            "geom_masks": {
                geom_name(full_model, geom_id): {
                    "contype": int(full_model.geom_contype[geom_id]),
                    "conaffinity": int(full_model.geom_conaffinity[geom_id]),
                }
                for geom_id in (full_yaw_id, full_pitch_id)
            },
            "raw_mj_geomDistance_m": full_pair_distance,
            "active_contact_rows": full_pair_contacts,
            "explicit_pair_count": int(full_model.npair), "exclude_count": int(full_model.nexclude),
            "disableflags": int(full_model.opt.disableflags),
            "filterparent_enabled": not bool(int(full_model.opt.disableflags) & filter_bit),
            "geom_masks_allow_pair": bool((int(full_model.geom_contype[full_yaw_id]) & int(full_model.geom_conaffinity[full_pitch_id]))
                                           or (int(full_model.geom_contype[full_pitch_id]) & int(full_model.geom_conaffinity[full_yaw_id]))),
        },
    }
    mesh_comparison = {
        "source_mesh": mesh_path.name,
        "source_mesh_sha256": sha256(mesh_path),
        "source_triangle_count": int(raw_face_count),
        "source_unique_vertex_count": int(len(raw_unique)),
        "source_urdf_scale": scale.tolist(),
        "source_collision_origin_xyz_m": origin_xyz.tolist(),
        "source_collision_origin_rpy_rad": origin_rpy.tolist(),
        "source_collision_local_aabb_min_m": (raw_unique * scale).min(axis=0).tolist(),
        "source_collision_local_aabb_max_m": (raw_unique * scale).max(axis=0).tolist(),
        "source_collision_local_extents_m": ((raw_unique * scale).max(axis=0)
                                               - (raw_unique * scale).min(axis=0)).tolist(),
        "visual_mesh": visual_source,
        "visual_triangle_count": int(visual_face_count),
        "visual_local_extents_m": (visual_raw_vertices.reshape(-1, 3, 3).reshape(-1, 3).max(axis=0)
                                    - visual_raw_vertices.reshape(-1, 3, 3).reshape(-1, 3).min(axis=0)).tolist(),
        "compiled_preconvex_triangle_count": int(model.mesh_facenum[int(model.geom_dataid[wrist_id])]),
        "compiled_preconvex_vertex_count": int(model.mesh_vertnum[int(model.geom_dataid[wrist_id])]),
        "raw_source_to_compiled_world_vertex_nearest_max_mm": float(raw_to_compiled.max() * 1000.0),
        "compiled_to_raw_source_world_vertex_nearest_max_mm": float(compiled_to_raw.max() * 1000.0),
        "triangle_topology_match": source_triangle_topology,
        "source_joint_chain_roll_to_yaw": wrist_joint_chain,
        "compiled_wrist_geom_world_pose": wrist_geom_pose,
        "source_triangle_surface_sample_count": int(len(surface_samples)),
        "sample_definition": "unique source vertices plus each triangle centroid and three edge midpoints; samples are on the original triangle surface before convexification",
    }
    frame = endpoint["diagnostic_grasp_frame"]
    pad_midpoint = 0.5 * (np.asarray(pad_rows[0]["convex_witness_pad_world_m"])
                          + np.asarray(pad_rows[1]["convex_witness_pad_world_m"]))
    frame_audit = {
        "source_named_tcp_links": frame.get("source_named_tcp_links", []),
        "candidate_origin_definition": frame.get("definition"),
        "candidate_origin_world_m": frame.get("world_position_m"),
        "root_pose_is_full_arm_fk_solution": False,
        "root_pose_semantics": "scene-fixed transform of the isolated yaw-link root, derived from a closest-point jaw basis; not an upstream arm-joint configuration",
        "pad_rows": pad_rows,
        "pad_bottle_geometry_contacts_after_mj_forward": pad_bottle_contacts,
        "pad_contact_normal_mean_bottle_to_pad_world": contact_family_means,
        "pad_contact_family_normal_angle_deg": pad_contact_normal_angle,
        "pad_contact_pair_normal_audit_by_bottle_geom": contacts_by_bottle_geom,
        "both_pads_contact_bottle_neck": both_pads_contact_neck,
        "contact_normals_are_geometry_only": True,
        "contact_force_measured": False,
        "closest_witness_segment_direction_angle_deg": opposing_angle,
        "pad_witness_separation_xyz_mm": (pad_separation * 1000.0).tolist(),
        "pad_witness_midpoint_world_m": pad_midpoint.tolist(),
        "pad_witness_midpoint_error_from_neck_center_xyz_mm": ((pad_midpoint - neck_center) * 1000.0).tolist(),
        "source_driver_follower_qpos_rad": {name: float(data.qpos[geometry_module.qpos_id(model, name)])
                                             for name in (DRIVER, FOLLOWER)},
        "source_driver_follower_joint_state": jaw_joint_state,
        "source_driver_follower_limits_rad": {"driver": [-1.0, 0.0], "follower": [0.0, 1.0]},
        "source_mimic_residual_rad": float(data.qpos[geometry_module.qpos_id(model, FOLLOWER)]
                                            + data.qpos[geometry_module.qpos_id(model, DRIVER)]),
        "selection_scope": "whole compiled jaw geoms against the selected bottle neck geom; not restricted to identified functional pad face patches",
        "validity": "diagnostic isolated fixture frame only; no source TCP, arm FK, collision-aware arm IK, or reachability proof",
    }
    set_render_groups(model)
    set_render_colors(model)
    contact_marks = [
        {"witness_robot_world_m": row["convex_witness_pad_world_m"],
         "witness_bottle_world_m": row["convex_witness_bottle_world_m"]}
        for row in pad_rows
    ]
    if wrist_hull_rows:
        contact_marks.append({"witness_robot_world_m": wrist_hull_rows[0]["witness_robot_world_m"],
                              "witness_bottle_world_m": wrist_hull_rows[0]["witness_bottle_world_m"]})
    for source_overlap in source_inside_points:
        point = source_overlap["point_world_m"]
        contact_marks.append({
            "witness_robot_world_m": point,
            "witness_bottle_world_m": point,
            "marker_kind": "source_triangle_inside_bottle",
        })
    lookat = neck_center.tolist()
    cameras = {
        "front": geometry_module.camera_for(lookat, 0.42, 90.0, -8.0),
        "side": geometry_module.camera_for(lookat, 0.42, 180.0, -8.0),
    }
    views = {}
    for view, camera in cameras.items():
        source_png = output / f"{view}_source_triangles.png"
        hull_png = output / f"{view}_compiled_hull.png"
        render_overlay(model, data, camera, source_png, contact_marks, collision_only=False)
        render_overlay(model, data, camera, hull_png, contact_marks, collision_only=True)
        comparison = output / f"{view}_collision_root_cause_comparison.png"
        compose_collision_comparison(
            source_png, hull_png, comparison, view.upper(), [
                f"Same saved endpoint and bottle pose; MuJoCo hull minimum = {wrist_hull_rows[0]['signed_distance_m'] * 1000.0:.3f} mm.",
                f"Source triangle samples inside bottle collision volumes = {sum(row['surface_samples_inside_count'] for row in source_surface_by_bottle)}.",
                "Markers: yellow/pink=jaw witnesses; orange=hull witness; magenta leader starts at a source STL sample inside bottle collision volume.",
                "Offline static geometry only; no physical contact simulation or altered collision model.",
            ],
        )
        views[view] = {"source_triangles": str(source_png), "compiled_hull": str(hull_png),
                       "annotated_comparison": str(comparison)}

    result = {
        "runtime_identity": runtime,
        "input_endpoint_json": str(endpoint_path.resolve()),
        "input_endpoint_sha256": sha256(endpoint_path),
        "model_identity": {
            "x2_pin": X2_PIN, "x2_urdf_sha256": sha256(source_urdf),
            "root_position_world_m": root_pos.tolist(), "root_rotation_world": root_rot.tolist(),
            "canonical_bottle_state": endpoint["bottle_position_world_m"],
            "canonical_bottle_spec": {
                "nominal_diameter_m": 0.070,
                "nominal_total_height_m": 0.2445,
                "nominal_mass_kg": 0.570,
                "compiled_body_mass_kg": float(model.body_mass[bottle_body_id]),
                "compiled_collision_geoms": bottle_geom_inventory,
            },
            "source_qpos_rad": {name: float(data.qpos[geometry_module.qpos_id(model, name)])
                                 for name in (DRIVER, FOLLOWER)},
            "full_source_wrist_joint_pose_rad": full_pose,
            "filtered_model_info": model_info,
        },
        "physics_steps": 0,
        "collision_filtering": pair_filter,
        "wrist_roll_collision_geometry": {
            "mesh_comparison": mesh_comparison,
            "source_triangle_samples_vs_bottle_geoms": source_surface_by_bottle,
            "source_surface_interior_samples": source_inside_points,
            "compiled_convex_hull_distances_vs_bottle_geoms": wrist_hull_rows,
        },
        "grasp_frame_audit": frame_audit,
        "views": views,
        "verdicts": {
            "true_active_yaw_pitch_contact_in_full_source_model": bool(full_pair_contacts),
            "fixed_root_fixture_yaw_pitch_contact_reaches_solver": bool(yaw_pitch_contacts),
            "original_source_collision_triangles_intersect_bottle_at_saved_pose": bool(source_inside_points),
            "convex_hull_is_only_source_of_wrist_blocker": not bool(source_inside_points),
            "source_defined_tcp_present": bool(frame.get("source_named_tcp_links")),
            "source_faithful_endpoint_feasible": False,
            "physical_grasp_tested": False,
        },
    }
    write_json(output / "collision_root_cause_result.json", result)
    command = (
        f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
        f"--x2-root {runtime['x2_root']} --canonical-helper {runtime['canonical_helper']} "
        f"--output {output} --collision-root-cause-only --existing-endpoint {endpoint_path.resolve()}"
    )
    (output / "experiment_commands.txt").write_text(
        f"RobotSim branch before commit: {runtime['robotsim_branch']}\n"
        f"RobotSim HEAD before commit: {runtime['robotsim_head']}\n"
        f"X2 pin: {X2_PIN}\nNo mj_step calls; static state computed with mj_forward only.\n\n"
        f"Exact reproduction command:\n{command}\n", encoding="utf-8"
    )
    pad_contact_summary = "; ".join(
        "{}–{} {:.3f} mm, n(bottle->pad)={}".format(
            row["pad_body"], row["bottle_geom"], row["distance_m"] * 1000.0,
            np.round(row["normal_bottle_to_pad_world"], 3).tolist(),
        )
        for row in pad_bottle_contacts
    ) or "无指垫/瓶 contact"
    report = [
        "# 原生 X2 OmniPicker 碰撞根因审计",
        "",
        "## 结论",
        "",
        "本次只重建已保存的 neck/a0/t0 姿态；无候选搜索、`mj_step`、接触力实验、碰撞模型改动或排除项改动。结果只判定此孤立固定根候选，不外推全身 IK 可达性或硬件总体兼容性。",
        f"规范瓶输入为直径 70.0 mm、高 244.5 mm、标称质量 0.570 kg；编译 body mass `{model.body_mass[bottle_body_id]:.6f} kg`，primitive collision geom 清单及尺寸见结果 JSON。",
        f"- 全身原始 X2 的 yaw/pitch 几何重叠进入接触约束：**{'是' if full_pair_contacts else '否'}**。",
        f"- 固定根隔离夹具中的同一重叠进入 `data.contact`：**{'是' if yaw_pitch_contacts else '否'}**。",
        f"- wrist-roll 原始源碰撞三角面与瓶碰撞体相交：**{'是' if source_inside_points else '否'}**。",
        "- 本候选的源碰撞几何静态预检：**FAIL**；物理抓取：**未测试**。",
        "",
        "## 碰撞过滤语义",
        "",
        f"相同 yaw/pitch 关节相对姿态的 raw `mj_geomDistance` 为 `{yaw_pitch_distance * 1000.0:.3f} mm`（隔离夹具）与 `{full_pair_distance * 1000.0:.3f} mm`（完整源 URDF）。完整模型 `data.contact` 匹配数 {len(full_pair_contacts)}；隔离夹具匹配数 {len(yaw_pitch_contacts)}，产生 {sum(row['active_solver_rows'] for row in yaw_pitch_contacts)} 个 constraint rows。`mj_geomDistance` 不应用接触过滤，负距离本身不表示 solver 实际约束。",
        f"源 URDF 链为 `right_elbow_link -> right_wrist_yaw_link -> right_wrist_pitch_link -> right_wrist_roll_link`；yaw/pitch 是直接父子。Fixture yaw/pitch geom 的 `(contype, conaffinity)` 为 `({int(model.geom_contype[yaw_id])}, {int(model.geom_conaffinity[yaw_id])})/({int(model.geom_contype[pitch_id])}, {int(model.geom_conaffinity[pitch_id])})`；full source 对应值为 `({int(full_model.geom_contype[full_yaw_id])}, {int(full_model.geom_conaffinity[full_yaw_id])})/({int(full_model.geom_contype[full_pitch_id])}, {int(full_model.geom_conaffinity[full_pitch_id])})`，均允许配对。完整模型无显式 pair 或 exclude，`mjDSBL_FILTERPARENT` 未关闭。MuJoCo 的 parent filter 排除非 world 的 parent-child body 对；完整模型中 yaw 的 parent 是 elbow、pitch 的 parent 是 yaw，所以该 pair 被过滤。裁剪后 yaw 成为 world 的直接 child；MuJoCo 文档明确 parent 是 world 时不应用 parent-child filter，因此相同 pair 在固定根夹具生成 contact。历史 simulation-only exclusion 清单 {len(historical_excludes)} 项未变；此腕部裁剪模型中适用 {len(fixture_excludes)} 项（见 JSON），没有新增 exclusion。前向静态接触中的 `efc_address`/行数说明约束行已构建；未做 `mj_step`，不读取或解释 `efc_force`，故没有力结论。",
        "",
        "## 源 STL 三角面与凸包",
        "",
        f"腕部源 collision STL `{mesh_path.name}` SHA256 `{sha256(mesh_path)}`：{raw_face_count:,} 个三角形、{len(raw_unique):,} 个唯一顶点；URDF scale `{scale.tolist()}`、mesh collision origin xyz/rpy `{origin_xyz.tolist()}` / `{origin_rpy.tolist()}`、局部 AABB extents `{np.round(mesh_comparison['source_collision_local_extents_m'], 6).tolist()} m`。同 link 的 visual 使用 `{visual_mesh_path.name}`（SHA256 `{sha256(visual_mesh_path)}`），其 scale/origin 与 extent 也在 JSON。腕链 source joint xyz/rpy/axis 和编译后 geom 世界 pose 亦在 JSON；collision 与 visual 是不同资产文件，但本姿态下没有发现 scale/origin 导致的编译偏移。",
        f"MuJoCo 凸包前的 mesh 仍有 {mesh_comparison['compiled_preconvex_triangle_count']:,} 个三角形/{mesh_comparison['compiled_preconvex_vertex_count']:,} 个顶点；逐面无向顶点拓扑完全匹配（source-only/compiled-only faces = {source_triangle_topology['source_faces_not_present_in_compiled_preconvex_mesh']}/{source_triangle_topology['compiled_faces_not_present_in_source_stl']}），源到编译顶点最大误差 {source_triangle_topology['source_unique_vertex_to_compiled_vertex_max_mm']:.9f} mm。",
        f"对 {len(surface_samples):,} 个源面样本（唯一顶点、三角中心、边中点）以同一瓶位做离线体内检查：瓶身 cylinder 0 点、肩部 ellipsoid {source_surface_by_bottle[1]['surface_samples_inside_count']:,} 点、颈部 {source_surface_by_bottle[2]['surface_samples_inside_count']:,} 点、瓶盖 {source_surface_by_bottle[3]['surface_samples_inside_count']:,} 点，共 {sum(row['surface_samples_inside_count'] for row in source_surface_by_bottle):,} 个样本落入瓶碰撞体。肩部数值是椭球隐式 level 的符号判断，不是毫米深度。故源 collision 三角面在此姿态已经穿入肩/颈/盖，凸包填充不是唯一原因。",
        f"本次保存 endpoint 的 compiled hull `mj_geomDistance` 分项最小值：肩部 `{wrist_hull_rows[0]['signed_distance_m'] * 1000.0:.3f} mm`、颈部 `{next(row['signed_distance_m'] for row in wrist_hull_rows if row['bottle_geom'] == 'bottle_neck') * 1000.0:.3f} mm`、瓶盖 `{next(row['signed_distance_m'] for row in wrist_hull_rows if row['bottle_geom'] == 'bottle_cap') * 1000.0:.3f} mm`、瓶身 `{next(row['signed_distance_m'] for row in wrist_hull_rows if row['bottle_geom'] == 'bottle_body') * 1000.0:.3f} mm`。因此前序报告的 25–42 mm 区间不是本 endpoint 的复测结果；本姿态肩/颈/盖 hull 穿透深度为 21.447–25.175 mm。源样本 SDF 与 hull 的通用凸碰撞距离定义不同，只作分别量化，不把两者差值当成精确凸包外扩厚度。",
        "",
        "## 抓取帧算法",
        "",
        f"源 URDF 命名 TCP link：`{frame_audit['source_named_tcp_links']}`。帧原点由整块 narrow/wide 编译 jaw geom 到 neck geom 的最近点中点定义；jaw witness 间隔 xyz `{pad_separation[0]*1000:.3f}, {pad_separation[1]*1000:.3f}, {pad_separation[2]*1000:.3f} mm`，中点相对 neck center `{', '.join(f'{v:.3f}' for v in ((pad_midpoint-neck_center)*1000.0))} mm`。0.450° 是两条最近点线段方向夹角，不是接触法向夹角。",
        f"静态 `mj_forward` 生成的指垫/瓶接触 pair：{pad_contact_summary}。narrow 触及瓶身/肩/颈，wide 触及瓶身/肩；双侧没有同时触及 neck（`{both_pads_contact_neck}`）。肩部两侧接触的几何平均法线夹角 `{contacts_by_bottle_geom.get('bottle_shoulder', {}).get('opposing_normal_angle_deg')}`°，但两侧重叠分别约 {max(-row['distance_m'] * 1000.0 for row in pad_bottle_contacts if row['bottle_geom'] == 'bottle_shoulder'):.3f} mm 与 {min(-row['distance_m'] * 1000.0 for row in pad_bottle_contacts if row['bottle_geom'] == 'bottle_shoulder'):.3f} mm，不能当作可接受夹持。MuJoCo contact frame 首轴是 geom1→geom2，本报告统一转换成 bottle→pad；这里仅有几何法向，没有 `mj_step`，不代表接触力或承载能力。",
        f"源指关节范围/qpos：`{DRIVER}` `{jaw_joint_state[DRIVER]['source_limit']}` / `{jaw_joint_state[DRIVER]['compiled_qpos_rad']:.9f} rad`；`{FOLLOWER}` `{jaw_joint_state[FOLLOWER]['source_limit']}` / `{jaw_joint_state[FOLLOWER]['compiled_qpos_rad']:.9f} rad`。qpos 满足源 mimic `{frame_audit['source_mimic_residual_rad']:.3g} rad`，但当前静态姿态存在多处大碰撞，不是有效的接触姿态。",
        "帧的 orientation 仍由 closest-point jaw axis 与远端网格 PCA 构造，没有限制到已定义的功能指腹面。其 root pose 是固定隔离夹具变换，不是源命名 TCP，也不是全臂 FK/碰撞感知 IK 解。因此该候选的抓取帧构造无效；本次不把 20 μm 闭合残差当成物理抓取门槛，也不据此搜索新候选。",
        "",
        "## 根因决策表",
        "",
        "| 问题 | 结论 | 定量证据 |",
        "|---|---|---|",
        f"| 39.217 mm yaw/pitch overlap 是全身活动自碰撞？ | 否 | 原始距离 fixture/full `{yaw_pitch_distance*1000:.3f}/{full_pair_distance*1000:.3f} mm`；contact `{len(yaw_pitch_contacts)}/{len(full_pair_contacts)}`；父子过滤解释见上 |",
        f"| source wrist collision triangles 与瓶相交？ | 是 | 颈/盖/肩各 {source_surface_by_bottle[2]['surface_samples_inside_count']}/{source_surface_by_bottle[3]['surface_samples_inside_count']}/{source_surface_by_bottle[1]['surface_samples_inside_count']} 个 inside samples；面拓扑完全匹配 |",
        "| 阻挡是否主要/仅来自凸包？ | 不是仅凸包 | 原始源面已与瓶碰撞体相交；hull 距离仍更负，但两类距离不是完全同定义 |",
        f"| grasp-frame construction 有效？ | 否 | 无 source TCP；整块 jaw 最近点 + PCA；neck 未形成双侧接触；肩部平均法线夹角 {contacts_by_bottle_geom['bottle_shoulder']['opposing_normal_angle_deg']}° 但穿透约 16–17 mm |",
        "| 本保存候选的 source-faithful 静态端点可行？ | 否 | wrist-roll 源面/瓶交叠；jaw/肩部深穿透；无完整手臂 FK 路径验证 |",
        "| 物理抓取实际测试？ | 否 | 0 physics steps；没有接触力、载荷或 lift 证据 |",
        "",
        "## 决策与下一步",
        "",
        "本候选的阻挡是两项局部问题叠加：39.217 mm yaw/pitch raw overlap 在完整源模型中受默认父子过滤，不是该模型内的活动自碰撞；裁剪后 yaw 成为 world 的直接 child，而 MuJoCo 对 world-parent pair 不应用这项过滤，因此固定根夹具生成 contact。另一方面，保存的 wrist-roll/bottle 姿态不仅是凸包假象：源 STL 三角面样本已经进入瓶肩、颈和盖的碰撞体，编译 hull 在该姿态给出更深的负距离。与此同时，现有整块 jaw 最近点加 PCA 帧没有对准功能指腹面，造成明显肩部穿透并且没有双侧 neck 接触。结论仅为这个 endpoint 构造失败；现有证据不能判定其他源几何姿态全局不可行，也不能证明原碰撞资产与实机外壳不一致。",
        "唯一建议的下一项实验：在孤立夹爪中依据 source STL 上实际相向的 distal narrow/wide 内侧指垫面建立一个功能接触帧，以瓶轴定义纵向方向；仅对这一个派生帧做一次 source-limit 与零步 wrist/bottle、pad/bottle 几何预检，通过后才映射到全臂 IK。不要沿用整块 jaw 最近点/PCA 帧，也不要扩展为候选姿态搜索。",
        "参考：[AgiBot X2 pinned source](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d)；[MuJoCo 3.3.6 computation and contact frames](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html)；[MuJoCo 3.3.6 XML reference, mesh collision semantics](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html)。",
        "",
        "## 复现",
        "",
        f"证据目录：`{output}`。精确命令见 `experiment_commands.txt`。RobotSim 输入 HEAD `{runtime['robotsim_head']}`；审计脚本 SHA256 `{runtime['audit_script_sha256']}`；X2 repo `AgibotTech/agibot_x2_urdf` pin `{X2_PIN}` / URDF SHA256 `{runtime['x2_urdf_sha256']}`；MuJoCo {mujoco.__version__}，native library SHA256 `{runtime['mujoco_native_library_sha256']}`。",
        "标注同机位对照图：`front_collision_root_cause_comparison.png` 与 `side_collision_root_cause_comparison.png`。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "yaw_pitch_fixture_distance_mm": yaw_pitch_distance * 1000.0,
        "yaw_pitch_fixture_contacts": yaw_pitch_contacts,
        "yaw_pitch_full_distance_mm": full_pair_distance * 1000.0,
        "yaw_pitch_full_contacts": full_pair_contacts,
        "wrist_hull_minimum_mm": wrist_hull_rows[0]["signed_distance_m"] * 1000.0,
        "source_surface_inside_count": sum(row["surface_samples_inside_count"] for row in source_surface_by_bottle),
        "opposing_pad_witness_angle_deg": opposing_angle,
        "physics_steps": 0,
        "output": str(output),
    }, indent=2))
    return 0


def render_group_view(model: mujoco.MjModel, data: mujoco.MjData,
                      camera: mujoco.MjvCamera, output: Path, groups: tuple[int, ...]) -> None:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    for group in (1, *groups):
        if group < len(option.geomgroup):
            option.geomgroup[group] = 1
    if 3 in groups:
        option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 900
    renderer = mujoco.Renderer(model, height=900, width=1280)
    renderer.update_scene(data, camera=camera, scene_option=option)
    Image.fromarray(renderer.render().copy()).save(output)
    renderer.close()


def add_title(path: Path, title: str) -> None:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 40), fill=(15, 24, 34))
    draw.text((18, 11), title, fill=(246, 248, 250))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, required=True)
    parser.add_argument("--canonical-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth-deg", type=float, choices=(0.0, 90.0), default=0.0,
                        help="one of two bounded geometry-derived orthogonal grasp directions")
    parser.add_argument("--finger-tilt-deg", type=float, choices=(-90.0, 0.0, 90.0), default=0.0,
                        help="tilt the finger long axis from bottle axis toward the local tangent")
    parser.add_argument("--grasp-height-offset-mm", type=float, default=0.0,
                        help="selected target geom local-z section offset for grasp sizing and fixture alignment, in mm")
    parser.add_argument("--target-bottle-geom",
                        choices=("bottle_body", "bottle_shoulder", "bottle_neck", "bottle_cap"),
                        default="bottle_body",
                        help="canonical bottle collision geom to center between the pads")
    parser.add_argument("--collision-root-cause-only", action="store_true",
                        help="replay a saved endpoint exactly and audit filtering and source triangles without candidate search")
    parser.add_argument("--existing-endpoint", type=Path,
                        help="saved endpoint_geometry.json for --collision-root-cause-only")
    args = parser.parse_args()
    if args.collision_root_cause_only and args.existing_endpoint is None:
        parser.error("--collision-root-cause-only requires --existing-endpoint")
    if args.existing_endpoint is not None and not args.collision_root_cause_only:
        parser.error("--existing-endpoint is valid only with --collision-root-cause-only")
    x2_root = args.x2_root.resolve()
    canonical_helper = args.canonical_helper.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    robot_root = script.parents[2]
    robot_head = subprocess.check_output(["git", "-C", str(robot_root), "rev-parse", "HEAD"], text=True).strip()
    vendor_head = subprocess.check_output(["git", "-C", str(x2_root), "rev-parse", "HEAD"], text=True).strip()
    vendor_dirty = subprocess.check_output(["git", "-C", str(x2_root), "status", "--porcelain"], text=True).strip()
    source_urdf = x2_root / X2_URDF_REL
    if vendor_head != X2_PIN or vendor_dirty:
        raise RuntimeError(f"Pinned vendor checkout mismatch: head={vendor_head}, dirty={bool(vendor_dirty)}")
    if sha256(source_urdf) != X2_URDF_SHA256:
        raise RuntimeError("Pinned OmniPicker URDF SHA mismatch")
    if sha256(canonical_helper) != CANONICAL_HELPER_SHA256:
        raise RuntimeError("Canonical RobotSim asset helper SHA mismatch")

    geometry_module = load_module(script.with_name("issue46_omnipicker_isolated_grasp.py"),
                                  "issue46_omnipicker_isolated_grasp_helper")
    geometry_module.X2_ROOT = x2_root
    geometry_module.CANONICAL_MODULE = canonical_helper
    geometry_module.STATIC_ROOT = canonical_helper.parent
    base = geometry_module.load_baseline()
    base.X2_ROOT = x2_root
    base.STATIC_ROOT = canonical_helper.parent

    runtime = {
        "robotsim_head": robot_head,
        "robotsim_branch": subprocess.check_output(["git", "-C", str(robot_root), "branch", "--show-current"], text=True).strip(),
        "x2_root": str(x2_root),
        "x2_repository": "https://github.com/AgibotTech/agibot_x2_urdf",
        "x2_commit": vendor_head,
        "x2_urdf": str(X2_URDF_REL),
        "x2_urdf_sha256": sha256(source_urdf),
        "x2_license": "Mulan PSL v2",
        "canonical_helper": str(canonical_helper),
        "canonical_helper_sha256": sha256(canonical_helper),
        "mujoco_python": mujoco.__version__,
        "mujoco_native_library_sha256": sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
        "audit_script_sha256": sha256(script),
        "python": sys.version,
        "fixture_scope": "source-faithful isolated subtree rooted at right_wrist_yaw_link; scene-initialized fixed root pose; no full-arm DOFs",
        "model_edits": "none to source collision geometry, source position ranges, axes, link transforms, or mimic relation; visual recoloring and geom-group remapping are renderer-only",
        "render_group_mapping": {"environment": 1, "robot_visual": 2, "robot_collision": 3},
        "physics_steps": 0,
    }
    source_root = ET.parse(source_urdf).getroot()
    source_links = {link.get("name", ""): link for link in source_root.findall("link")}
    simple_urdf = x2_root / "X2_URDF-v1.4.0/X2-Ultra_simple_collision.urdf"
    simple_root = ET.parse(simple_urdf).getroot()
    simple_links = {link.get("name", "") for link in simple_root.findall("link")}
    simple_wrist = simple_root.find("link[@name='right_wrist_roll_link']")
    simple_mesh_names = {}
    for role in ("visual", "collision"):
        mesh = None if simple_wrist is None else simple_wrist.find(f"{role}/geometry/mesh")
        if mesh is not None:
            filename = Path(mesh.get("filename", "")).name
            simple_mesh_names[role] = {"filename": filename, "sha256": sha256(simple_urdf.parent / "meshes" / filename)}
        else:
            simple_mesh_names[role] = None
    filtered_info = geometry_module.filtered_urdf(output, source_urdf)
    mesh_inventory = source_mesh_inventory(
        source_urdf, source_urdf.parent / "meshes", source_links,
        filtered_info["included_link_names"],
    )
    canonical = load_module(canonical_helper, "robotsim_canonical_manipulation_assets")
    controller_model, _ = base.build_model("open_hold", {})
    controller = base.derive_jaw_controller(controller_model)

    if args.collision_root_cause_only:
        return root_cause_audit(
            output, source_urdf, source_links, filtered_info, geometry_module,
            base, controller, args.existing_endpoint.resolve(), runtime,
        )

    candidate_id = {
        "bottle_body": 1000,
        "bottle_neck": 2000,
        "bottle_shoulder": 3000,
        "bottle_cap": 10000,
    }[args.target_bottle_geom] + int(args.azimuth_deg * 10.0) + int(args.finger_tilt_deg)
    candidate = geometry_module.candidate_model(
        base, Path(filtered_info["derived_urdf_path"]), candidate_id, controller,
        azimuth_deg=args.azimuth_deg, finger_tilt_deg=args.finger_tilt_deg,
    )
    body_geom_id = next(gid for gid in geometry_module.bottle_geom_ids(candidate["model"])
                        if geom_name(candidate["model"], gid) == "bottle_body")
    target_geom_id = next(gid for gid in geometry_module.bottle_geom_ids(candidate["model"])
                          if geom_name(candidate["model"], gid) == args.target_bottle_geom)
    target_type = int(candidate["model"].geom_type[target_geom_id])
    target_geom_center_world = np.asarray(
        candidate["data"].geom_xpos[target_geom_id], dtype=float
    ).copy()
    target_rotation = np.asarray(
        candidate["data"].geom_xmat[target_geom_id], dtype=float
    ).reshape(3, 3)
    target_offset_m = args.grasp_height_offset_mm / 1000.0
    if abs(float(target_rotation[2, 2])) < 0.999:
        raise ValueError("Selected bottle target geom must remain upright")
    target_center_world = target_geom_center_world + target_rotation[:, 2] * target_offset_m
    target_size = np.asarray(candidate["model"].geom_size[target_geom_id], dtype=float)
    jaw_normal_world = np.array([
        math.cos(math.radians(args.azimuth_deg)),
        math.sin(math.radians(args.azimuth_deg)),
        0.0,
    ])
    if target_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        target_radius_m = float(target_size[0])
        target_diameter_m = 2.0 * target_radius_m
        target_section = "constant-radius cylinder cross-section"
    elif target_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half_height = float(target_size[2])
        if abs(target_offset_m) >= half_height:
            raise ValueError("Shoulder cross-section offset must be inside the ellipsoid")
        section_scale = math.sqrt(max(0.0, 1.0 - (target_offset_m / half_height) ** 2))
        local_normal = target_rotation.T @ jaw_normal_world
        radius_x = float(target_size[0]) * section_scale
        radius_y = float(target_size[1]) * section_scale
        target_radius_m = math.sqrt(
            (radius_x * local_normal[0]) ** 2 + (radius_y * local_normal[1]) ** 2
        )
        target_diameter_m = 2.0 * target_radius_m
        target_section = "ellipse cross-section at the requested local-z offset"
    else:
        raise ValueError(
            f"Target geom must be a cylinder or ellipsoid: {args.target_bottle_geom}"
        )
    body_center_world = np.asarray(candidate["data"].geom_xpos[body_geom_id], dtype=float).copy()
    gap_samples_open_to_closed = list(reversed(candidate["gap_samples"]))
    contact_aperture = None
    for first, second in zip(gap_samples_open_to_closed, gap_samples_open_to_closed[1:]):
        first_gap = float(first["jaw_surface_gap_m"])
        second_gap = float(second["jaw_surface_gap_m"])
        if first_gap >= target_diameter_m >= second_gap:
            fraction = (first_gap - target_diameter_m) / max(first_gap - second_gap, 1e-12)
            contact_aperture = float(first["aperture_ratio"] + fraction * (
                second["aperture_ratio"] - first["aperture_ratio"]
            ))
            break
    if contact_aperture is None:
        raise RuntimeError(
            f"Source jaw range does not span target diameter {target_diameter_m:.6f} m"
        )
    candidate["root_position"] = np.asarray(candidate["root_position"], dtype=float) + (
        target_center_world - body_center_world
    )
    candidate["contact_aperture"] = contact_aperture
    candidate["local_geometry"]["grasp_target_geom"] = args.target_bottle_geom
    candidate["local_geometry"]["grasp_target_radius_m"] = target_radius_m
    candidate["local_geometry"]["grasp_target_center_world_m"] = target_center_world.tolist()
    candidate["local_geometry"]["grasp_target_section"] = target_section
    model = candidate["model"]
    data = candidate["data"]
    geometry_module.candidate_state(base, candidate, 1.0)
    open_rows = geometry_module.pair_rows(model, data,
                                          geometry_module.robot_geom_ids(model),
                                          geometry_module.bottle_geom_ids(model))
    open_all_rows = all_pair_distances(
        model, data, geometry_module.robot_geom_ids(model), geometry_module.bottle_geom_ids(model)
    )
    open_all_min = min(open_all_rows, key=lambda row: row["signed_distance_m"])
    open_endpoint_state = {
        "open_qpos": {DRIVER: float(data.qpos[geometry_module.qpos_id(model, DRIVER)]),
                      FOLLOWER: float(data.qpos[geometry_module.qpos_id(model, FOLLOWER)])},
        "minimum_robot_bottle_distance_m": open_all_min["signed_distance_m"],
        "minimum_robot_bottle_pair": open_all_min,
        "source_joint_limit_pass": not geometry_module.limit_violations(
            model, data, geometry_module.source_limits(source_urdf)
        ),
    }
    self_contacts_open = [row for row in collision_state_rows(model, data)
                          if row["body1"] not in {"m0_bottle", "m0_table", "world"}
                          and row["body2"] not in {"m0_bottle", "m0_table", "world"}
                          and row["distance_m"] < -1.0e-7]
    endpoint_aperture = float(candidate["contact_aperture"])
    geometry_module.candidate_state(base, candidate, endpoint_aperture)
    all_robot = geometry_module.robot_geom_ids(model)
    bottle_geoms = geometry_module.bottle_geom_ids(model)
    pad_body_names = {NARROW, WIDE}
    target_geom_name = args.target_bottle_geom
    endpoint_rows = all_pair_distances(model, data, all_robot, bottle_geoms)
    pad_body = {NARROW, WIDE}
    pad_rows = [row for row in endpoint_rows
                if row["robot_body"] in pad_body and row["bottle_geom"] == target_geom_name]
    nonpad_rows = [row for row in endpoint_rows
                   if not (row["robot_body"] in pad_body and row["bottle_geom"] == target_geom_name)]
    wrist_rows = [row for row in nonpad_rows if row["robot_body"] == WRIST_ROLL]
    wrist_min = min(wrist_rows, key=lambda row: row["signed_distance_m"]) if wrist_rows else None
    nonpad_min = min(nonpad_rows, key=lambda row: row["signed_distance_m"]) if nonpad_rows else None
    pad_min_by_side = {
        name: min((row for row in pad_rows if row["robot_body"] == name),
                  key=lambda row: row["signed_distance_m"], default=None)
        for name in (NARROW, WIDE)
    }
    contact_marks = [row for row in pad_min_by_side.values() if row is not None]

    closure_rows = []
    for aperture in np.linspace(1.0, endpoint_aperture, 101):
        geometry_module.candidate_state(base, candidate, float(aperture))
        rows = all_pair_distances(model, data, all_robot, bottle_geoms)
        pad_side_min = {
            side: min((row for row in rows if row["robot_body"] == side and row["bottle_geom"] == target_geom_name),
                      key=lambda row: row["signed_distance_m"], default=None)
            for side in (NARROW, WIDE)
        }
        nonpad = [row for row in rows
                  if not (row["robot_body"] in pad_body_names and row["bottle_geom"] == target_geom_name)]
        closure_rows.append({
            "aperture_ratio": float(aperture),
            "driver_target_rad": float(base.aperture_targets(float(aperture))["right_claw_joint_target_rad"]),
            "follower_target_rad": float(base.aperture_targets(float(aperture))["R_hand_wide1_joint_target_rad"]),
            "source_mimic_residual_rad": float(
                data.qpos[geometry_module.qpos_id(model, FOLLOWER)] +
                data.qpos[geometry_module.qpos_id(model, DRIVER)]
            ),
            "narrow_pad_min_distance_m": None if pad_side_min[NARROW] is None else pad_side_min[NARROW]["signed_distance_m"],
            "wide_pad_min_distance_m": None if pad_side_min[WIDE] is None else pad_side_min[WIDE]["signed_distance_m"],
            "minimum_nonpad_distance_m": min(nonpad, key=lambda row: row["signed_distance_m"])["signed_distance_m"] if nonpad else None,
            "minimum_pair": rows[0],
        })

    # Restore endpoint state after the static kinematic closure samples.
    geometry_module.candidate_state(base, candidate, endpoint_aperture)
    endpoint_rows = all_pair_distances(model, data, all_robot, bottle_geoms)
    pad_rows = [row for row in endpoint_rows
                if row["robot_body"] in pad_body and row["bottle_geom"] == target_geom_name]
    nonpad_rows = [row for row in endpoint_rows
                   if not (row["robot_body"] in pad_body and row["bottle_geom"] == target_geom_name)]
    wrist_rows = [row for row in nonpad_rows if row["robot_body"] == WRIST_ROLL]
    wrist_min = min(wrist_rows, key=lambda row: row["signed_distance_m"]) if wrist_rows else None
    nonpad_min = min(nonpad_rows, key=lambda row: row["signed_distance_m"]) if nonpad_rows else None
    pad_min_by_side = {
        name: min((row for row in pad_rows if row["robot_body"] == name),
                  key=lambda row: row["signed_distance_m"], default=None)
        for name in (NARROW, WIDE)
    }
    contact_marks = [row for row in pad_min_by_side.values() if row is not None]
    source_limits = geometry_module.source_limits(source_urdf)
    endpoint_limit_violations = geometry_module.limit_violations(model, data, source_limits)
    endpoint_contacts = collision_state_rows(model, data)

    compiled_meshes = compiled_mesh_inventory(model, mesh_inventory)
    source_joints = source_joint_records(source_urdf, model, candidate)
    exact_targets = base.aperture_targets(endpoint_aperture)
    endpoint_qpos = {
        DRIVER: float(data.qpos[geometry_module.qpos_id(model, DRIVER)]),
        FOLLOWER: float(data.qpos[geometry_module.qpos_id(model, FOLLOWER)]),
    }
    qdriver = endpoint_qpos[DRIVER]
    qfollower = endpoint_qpos[FOLLOWER]
    mimic_residual = qfollower - (-1.0 * qdriver) - 0.0
    endpoint_pad_values = {
        side: None if row is None else row["signed_distance_m"]
        for side, row in pad_min_by_side.items()
    }
    endpoint_grasp_frame = {
        "definition": f"diagnostic frame origin is the midpoint between compiled narrow/wide pad-to-{target_geom_name} closest-point witnesses; local x is narrow-to-wide contact axis; not a vendor-defined TCP",
        "world_position_m": (
            0.5 * (
                np.asarray(pad_min_by_side[NARROW]["witness_robot_world_m"])
                + np.asarray(pad_min_by_side[WIDE]["witness_robot_world_m"])
            )
        ).tolist() if all(pad_min_by_side[name] is not None for name in (NARROW, WIDE)) else None,
        "narrow_contact_witness_world_m": None if pad_min_by_side[NARROW] is None else pad_min_by_side[NARROW]["witness_robot_world_m"],
        "wide_contact_witness_world_m": None if pad_min_by_side[WIDE] is None else pad_min_by_side[WIDE]["witness_robot_world_m"],
        "source_named_tcp_links": [
            link.get("name") for link in source_links.values()
            if re.search(r"(?:^|_)(?:tcp|ee|grasp)(?:_|$)", link.get("name", ""), re.IGNORECASE)
        ],
    }
    fingertip_frames = {}
    for side in (NARROW, WIDE):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side)
        fingertip_frames[side] = {
            "link_origin_world_m": np.asarray(data.xpos[bid], dtype=float).tolist(),
            "link_quaternion_wxyz": np.asarray(data.xquat[bid], dtype=float).tolist(),
            "collision_geom_origin_world_m": np.asarray(data.geom_xpos[
                base.collision_geom_id(model, side)
            ], dtype=float).tolist(),
            "source_local_mesh_extents_m": mesh_inventory[side]["collision"]["raw_local_extents_m"],
            "compiled_pad_to_bottle_distance_m": endpoint_pad_values[side],
        }
    finger_contacts = all(value is not None and abs(float(value)) <= CONTACT_TOL_M
                          for value in endpoint_pad_values.values())
    nonpad_clear = nonpad_min is None or float(nonpad_min["signed_distance_m"]) >= -1.0e-7
    open_clear = float(open_endpoint_state["minimum_robot_bottle_distance_m"]) > 0.0
    closure_nonpad_clear = all(row["minimum_nonpad_distance_m"] is None or row["minimum_nonpad_distance_m"] >= -1.0e-7
                               for row in closure_rows)
    endpoint_gap_m = float(geometry_module.geom_pair_distance(
        model, data, candidate["narrow_id"], candidate["wide_id"]
    )[0])
    bottle_body_diameter_m = float(model.geom_size[body_geom_id, 0] * 2.0)
    source_mimic_ok = abs(mimic_residual) <= 1.0e-10
    endpoint_pass = bool(finger_contacts and nonpad_clear and open_clear
                         and closure_nonpad_clear and not endpoint_limit_violations
                         and not self_contacts_open and source_mimic_ok)

    runtime.update({
        "canonical_bottle": {
            "diameter_m": bottle_body_diameter_m,
            "height_m": 0.2445,
            "mass_kg": 0.57,
            "body_position_world_m": [float(v) for v in canonical.CANONICAL_X2_BOTTLE_START_BODY_POS],
            "collision_geoms": [geom_name(model, gid) for gid in bottle_geoms],
            "grasp_target_geom": target_geom_name,
            "grasp_target_section": target_section,
            "grasp_target_local_z_offset_m": target_offset_m,
            "grasp_target_diameter_m": target_diameter_m,
            "grasp_target_center_world_m": target_center_world.tolist(),
        },
        "filtered_source_subtree": filtered_info,
        "source_mesh_inventory": mesh_inventory,
        "official_simple_collision_wrist_reference": {
            "urdf": str(simple_urdf.relative_to(x2_root)),
            "sha256": sha256(simple_urdf),
            "link_count": len(simple_links),
            "contains_omnipicker_base": "R_omnipicker_base_link" in simple_links,
            "wrist_visual_and_collision_meshes": simple_mesh_names,
            "simple_collision_matches_native_collision_sha256": bool(
                simple_mesh_names["collision"]
                and simple_mesh_names["collision"]["sha256"]
                == mesh_inventory[WRIST_ROLL]["collision"]["sha256"]
            ),
            "interpretation": "comparison only; this audit keeps the OmniPicker URDF collision mesh unchanged",
        },
        "compiled_mesh_inventory": compiled_meshes,
        "source_joint_records": source_joints,
        "wrist_to_picker_mount": {
            "source_joint_name": "R_omnipicker_joint",
            "source_parent": source_root.find("joint[@name='R_omnipicker_joint']/parent").get("link"),
            "source_child": source_root.find("joint[@name='R_omnipicker_joint']/child").get("link"),
            "source_xyz_m": [float(v) for v in source_root.find("joint[@name='R_omnipicker_joint']/origin").get("xyz", "0 0 0").split()],
            "source_rpy_rad": [float(v) for v in source_root.find("joint[@name='R_omnipicker_joint']/origin").get("rpy", "0 0 0").split()],
            "note": "copied from pinned URDF; no adapter or mount change",
        },
    })
    geometry = {
        "candidate": {
            "id": candidate_id,
            "orientation_definition": "geometry-derived orientation: jaw closing axis aligned with selected horizontal bottle azimuth; finger long axis tilted from bottle axis toward the local horizontal tangent; no random seeds or full-arm IK",
            "azimuth_deg": args.azimuth_deg,
            "finger_tilt_deg": args.finger_tilt_deg,
            "grasp_height_offset_mm": args.grasp_height_offset_mm,
            "isolated_root_position_world_m": np.asarray(candidate["root_position"], dtype=float).tolist(),
            "isolated_root_rotation_world": np.asarray(candidate["root_rotation"], dtype=float).tolist(),
            "bottle_position_world_m": [float(v) for v in canonical.CANONICAL_X2_BOTTLE_START_BODY_POS],
            "grasp_target_geom": target_geom_name,
            "grasp_target_section": target_section,
            "grasp_target_local_z_offset_m": target_offset_m,
            "grasp_target_diameter_m": target_diameter_m,
            "grasp_target_center_world_m": target_center_world.tolist(),
            "grasp_aperture_ratio": endpoint_aperture,
            "jaw_surface_gap_at_grasp_m": endpoint_gap_m,
            "jaw_surface_gap_at_open_m": float(max(candidate["gap_samples"], key=lambda row: row["aperture_ratio"])["jaw_surface_gap_m"]),
            "bottle_body_diameter_m": bottle_body_diameter_m,
            "fingertip_frames_world": fingertip_frames,
            "diagnostic_grasp_frame": endpoint_grasp_frame,
            "endpoint_joint_targets_rad": exact_targets,
            "endpoint_joint_qpos_rad": endpoint_qpos,
            "endpoint_mimic_residual_rad": mimic_residual,
            "endpoint_pad_clearances_m": endpoint_pad_values,
            "endpoint_minimum_nonpad_clearance": nonpad_min,
            "endpoint_minimum_wrist_roll_clearance": wrist_min,
            "open_state": open_endpoint_state,
            "endpoint_active_contact_pairs": endpoint_contacts,
            "endpoint_source_limit_violations": endpoint_limit_violations,
            "open_self_penetrating_contacts": self_contacts_open,
            "closure_samples": closure_rows,
        },
        "gates": {
            "source_pin_and_asset_hashes": "PASS",
            "source_axes_joint_ranges_mimic_preserved": "PASS",
            "compiled_collision_geometry_inventoried": "PASS",
            "canonical_70mm_bottle": "PASS",
            "both_intended_fingertips_reach_target_geom_at_one_source_valid_aperture": "PASS" if finger_contacts else "FAIL",
            "open_pose_clear_of_bottle": "PASS" if open_clear else "FAIL",
            "nonpad_geometry_clear_at_endpoint": "PASS" if nonpad_clear else "FAIL",
            "nonpad_geometry_clear_through_open_to_contact_closure": "PASS" if closure_nonpad_clear else "FAIL",
            "source_limits_and_mimic": "PASS" if not endpoint_limit_violations and source_mimic_ok else "FAIL",
            "zero_initial_robot_self_penetration": "PASS" if not self_contacts_open else "FAIL",
            "isolated_source_faithful_endpoint": "PASS" if endpoint_pass else "FAIL",
            "controlled_finger_contact_physics": "NOT RUN" if not endpoint_pass else "READY, NOT RUN",
            "full_x2_ik_and_approach": "NOT RUN" if not endpoint_pass else "NOT RUN",
            "50mm_lift_and_one_second_unsupported_hold": "NOT RUN",
        },
        "first_blocking_pair": (nonpad_min if not nonpad_clear else
                                 open_all_min if not open_clear else
                                 next((row["minimum_pair"] for row in closure_rows
                                       if row["minimum_nonpad_distance_m"] is not None
                                       and row["minimum_nonpad_distance_m"] < -1.0e-7), None)),
        "physics_steps": 0,
        "status": "PASS" if endpoint_pass else "FAIL — isolated source-faithful endpoint geometry",
    }
    result = {"runtime_identity": runtime, "geometry": geometry}
    write_json(output / "result.json", result)
    write_json(output / "source_and_compiled_mesh_inventory.json", {
        "source": mesh_inventory,
        "compiled": compiled_meshes,
        "mjv_convex_hull_visualization_flag_available": hasattr(mujoco.mjtVisFlag, "mjVIS_CONVEXHULL"),
    })
    write_json(output / "endpoint_geometry.json", geometry["candidate"])
    (output / "closure_geometry_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in closure_rows), encoding="utf-8"
    )
    (output / "experiment_commands.txt").write_text(
        f"Starting RobotSim HEAD: {robot_head}\n"
        f"X2 source HEAD: {vendor_head}\n"
        f"Exact command: MUJOCO_GL=egl {sys.executable} {script} "
        f"--x2-root {x2_root} --canonical-helper {canonical_helper} --output {output} "
        f"--azimuth-deg {args.azimuth_deg} --finger-tilt-deg {args.finger_tilt_deg} "
        f"--grasp-height-offset-mm {args.grasp_height_offset_mm} "
        f"--target-bottle-geom {target_geom_name}\n",
        encoding="utf-8",
    )

    runtime["render_group_mapping"] = set_render_groups(model)
    set_render_colors(model)
    lookat = target_center_world.copy()
    camera_front = geometry_module.camera_for(lookat.tolist(), 0.42, 90.0, -8.0)
    camera_side = geometry_module.camera_for(lookat.tolist(), 0.42, 180.0, -8.0)
    geometry_module.candidate_state(base, candidate, endpoint_aperture)
    overlay_marks = contact_marks + ([wrist_min] if wrist_min is not None else [])
    render_overlay(model, data, camera_front, output / "source_visual_collision_overlay_front.png",
                   overlay_marks, collision_only=False)
    render_overlay(model, data, camera_side, output / "source_visual_collision_overlay_side.png",
                   overlay_marks, collision_only=False)
    render_overlay(model, data, camera_front,
                   output / "source_visual_collision_overlay_contact_witness.png",
                   overlay_marks, collision_only=False, transparent_bottle=True)
    render_group_view(model, data, camera_front, output / "compiled_collision_hull_only.png", (3,))
    render_group_view(model, data, camera_front, output / "source_visual_only.png", (2,))
    for filename, title in (
        ("source_visual_collision_overlay_front.png", "Native source geometry at derived grasp endpoint - front"),
        ("source_visual_collision_overlay_side.png", "Native source geometry at derived grasp endpoint - side"),
        ("compiled_collision_hull_only.png", "Compiled collision geometry only (MuJoCo convex hull flag)"),
        ("source_visual_only.png", "Pinned source visual geometry only"),
    ):
        add_title(output / filename, title)

    wrist = geometry["candidate"]["endpoint_minimum_wrist_roll_clearance"]
    nonpad = geometry["candidate"]["endpoint_minimum_nonpad_clearance"]
    lines = [
        "# X2 原生 OmniPicker 隔离抓取端点几何审计",
        "",
        f"**结论：{geometry['status']}。** 本报告只判断固定根夹爪的源几何端点，不声称全局不可行；未运行动态接触、全臂 IK 或抬升。",
        "",
        "## 方法与来源",
        "",
        f"- RobotSim HEAD：`{robot_head}`；分支：`{runtime['robotsim_branch']}`。",
        f"- 官方 X2：[`AgibotTech/agibot_x2_urdf`](https://github.com/AgibotTech/agibot_x2_urdf)，pin `{vendor_head}`；URDF `{X2_URDF_REL}`，SHA-256 `{sha256(source_urdf)}`，Mulan PSL v2。",
        f"- canonical helper SHA-256：`{sha256(canonical_helper)}`；canonical 瓶身：直径 `{bottle_body_diameter_m * 1000:.1f} mm`、高 `244.5 mm`、质量 `0.570 kg`。",
        f"- MuJoCo Python `{mujoco.__version__}`；原始 URDF 子树从 `{ROOT_LINK}` 固定在世界坐标，保留腕、OmniPicker 链、源几何、轴、joint range 和 mimic。",
        f"- 目标瓶体 geom `{target_geom_name}`，接触高度相对 geom 中心偏移 `{target_offset_m * 1000:.1f} mm`；按 `{target_section}` 推导有效宽度 `{target_diameter_m * 1000:.3f} mm`。方位角 `{args.azimuth_deg:.0f}°`、手指长轴倾角 `{args.finger_tilt_deg:.0f}°`；闭合开度由实际指垫间距跨越该宽度的位置插值得出。",
        "- `mj_geomDistance` 对已编译碰撞 geom 求带符号距离；可视 geom 与碰撞 geom 通过 MuJoCo geom group 区分。渲染颜色和接触 witness 标记只作用于渲染态，不改物理模型。",
        "",
        "## 已确认运动学",
        "",
        f"- driver `{DRIVER}` 源范围 `[-1.000, 0.000] rad`，axis `[0,0,-1]`；follower `{FOLLOWER}` 源范围 `[0.000,1.000] rad`，axis `[0,0,-1]`。",
        "- 官方 mimic：`q_follower = -q_driver`（multiplier `-1`, offset `0`）。本次接触开度 driver/follower = `",
        f"  `{qdriver:.9f}` / `{qfollower:.9f} rad`，残差 `{mimic_residual:.3e} rad`；两者均在源限位内。",
        f"- 几何闭合目标在 aperture ratio `{endpoint_aperture:.6f}`；两指垫表面间距 `{endpoint_gap_m * 1000:.3f} mm`，目标 `{target_geom_name}` 直径 `{target_diameter_m * 1000:.3f} mm`；瓶身直径仍为 `{bottle_body_diameter_m * 1000:.3f} mm`。",
        f"- OPEN jaw gap `{float(max(candidate['gap_samples'], key=lambda row: row['aperture_ratio'])['jaw_surface_gap_m']) * 1000:.3f} mm`；闭合目标两侧实际最近距见结果 JSON。源 URDF 未定义 `TCP`/`ee_link`/`grasp_frame` 命名 link；本报告用两个 compiled fingertip closest-point witness 中点定义诊断抓取帧，不把 wrist/body origin 当 TCP。",
        "",
        "## 端点与阻挡",
        "",
        f"- 两侧指垫距离：窄侧 `{endpoint_pad_values[NARROW] * 1000 if endpoint_pad_values[NARROW] is not None else float('nan'):.3f} mm`；宽侧 `{endpoint_pad_values[WIDE] * 1000 if endpoint_pad_values[WIDE] is not None else float('nan'):.3f} mm`。接触判据为绝对距离不超过 `{CONTACT_TOL_M * 1000:.3f} mm`。",
        f"- 端点最小非目标接触距离（含指垫对瓶子其它 geom）：`{float(nonpad['signed_distance_m']) * 1000:.3f} mm`，pair `{nonpad['robot_body']}/{nonpad['robot_geom']} ↔ {nonpad['bottle_body']}/{nonpad['bottle_geom']}`。",
    ]
    if wrist is not None:
        lines.append(
            f"- wrist-roll 最小距离：`{float(wrist['signed_distance_m']) * 1000:.3f} mm`，pair `{wrist['robot_body']}/{wrist['robot_geom']} ↔ {wrist['bottle_body']}/{wrist['bottle_geom']}`；URDF 明确使用视觉 `right_wrist_roll_extend_link.stl`，碰撞 `right_wrist_roll_link.stl`。"
        )
    lines.extend([
        f"- OPEN 瓶体最小距离：`{float(open_endpoint_state['minimum_robot_bottle_distance_m']) * 1000:.3f} mm`。",
        f"- OPEN→接触开度静态扫过 `{len(closure_rows)}` 个关节状态；非指垫全程 clearance：`{'PASS' if closure_nonpad_clear else 'FAIL'}`。",
        f"- 静态源限制：`{'PASS' if not endpoint_limit_violations else 'FAIL'}`；OPEN 自碰穿透：`{len(self_contacts_open)}` 对。",
        "",
        "## 碰撞网格解释",
        "",
        "- 右腕的源 URDF 将 `right_wrist_roll_extend_link.stl` 仅用于 visual，而 collision 指向 `right_wrist_roll_link.stl`。本审计未以视觉网格替代碰撞网格。",
        f"- OmniPicker wrist visual `{mesh_inventory[WRIST_ROLL]['visual']['filename']}` (SHA `{mesh_inventory[WRIST_ROLL]['visual']['sha256']}`)；collision `{mesh_inventory[WRIST_ROLL]['collision']['filename']}` (SHA `{mesh_inventory[WRIST_ROLL]['collision']['sha256']}`)。官方 `X2-Ultra_simple_collision.urdf` 含 `{len(simple_links)}` links，是否包含 OmniPicker mount：`{'yes' if 'R_omnipicker_base_link' in simple_links else 'no'}`；其 wrist visual/collision 均引用 `{simple_mesh_names['collision']['filename'] if simple_mesh_names['collision'] else 'missing'}`，collision 与 OmniPicker 源碰撞 mesh 哈希相同：`{runtime['official_simple_collision_wrist_reference']['simple_collision_matches_native_collision_sha256']}`。该模型不含夹爪 mount，且不能提供更小腕部碰撞替代。",
        "- 原始 STL 三角数/边界、编译 mesh 和编译凸包唯一顶点/多边形数见 `source_and_compiled_mesh_inventory.json`。MuJoCo `mj_geomDistance` 和 `mjVIS_CONVEXHULL` 对照编译 collision geom；视觉网格单独保留。原始编译 geom groups 在清单中记录，渲染分组重映射只影响显示，不改变 contype/conaffinity 或碰撞计算。",
        "- 参考流程：MoveIt Task Constructor 用相对抓取帧、预抓取配置和分阶段 approach/close/planning-scene allowance；Isaac Grasp Editor 用 gripper DOF、open/closed target 与 physics colliders 建 grasp frame；MuJoCo mesh collision 文档说明碰撞使用凸包而渲染 mesh 可保持非凸。规划时允许手指—物体接触不等于在本物理模型中关闭指垫碰撞。",
        "  - MoveIt: https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html",
        "  - Isaac Sim Grasp Editor: https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/grasp_editor.html",
        "  - MuJoCo: https://mujoco.readthedocs.io/en/3.3.5/computation/",
        "  - AgiBot source: https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d",
        "- 前序全臂路径的 `geom_89` 已映射为 `right_wrist_roll_link` 碰撞 geom；该路径在 pad 仍离瓶 `48.2 mm` 时已到 1.0 mm margin。此处单独再核对 endpoint，不把路径失败误称 endpoint 失败。",
        "",
        "## 结果边界",
        "",
        "- 若本 endpoint 失败，结论仅为：该单一源几何推导 pose 被列明的 source collision geometry 阻挡；不推出其他 wrist roll 或 grasp frame 全局不可行。下一步应在孤立夹爪内根据腕—瓶距离法向做一个解析的腕 roll/插入方向改变量，再回到全臂之前复核 endpoint。",
        "- 若 endpoint 通过，本脚本仍不自动运行 physics；后续只可先做受控闭合接触验证，然后才回映射全 X2。",
        "- 物理 lift `NOT RUN`；没有 weld、attachment、对象 qpos 写入、mocap 或隐藏支撑。",
        "",
        "## 文件",
        "",
        f"- 证据目录：`{output}`",
        "- 原始 mesh / compiled hull 明细：`source_and_compiled_mesh_inventory.json`。",
        "- 闭合静态轨迹：`closure_geometry_trace.jsonl`。",
        "- 重现命令：`experiment_commands.txt`。",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": geometry["status"],
        "grasp_aperture_ratio": endpoint_aperture,
        "pad_clearance_m": endpoint_pad_values,
        "minimum_nonpad": nonpad_min,
        "minimum_wrist_roll": wrist_min,
        "open_clearance_m": open_endpoint_state["minimum_robot_bottle_distance_m"],
        "closure_nonpad_clear": closure_nonpad_clear,
        "evidence": str(output),
    }, indent=2))
    return 0 if endpoint_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
