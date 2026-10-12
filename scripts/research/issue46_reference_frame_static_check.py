#!/usr/bin/env python3
"""One reference-derived, zero-step X2 OmniPicker grasp-frame check."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/"
    "issue46-x2-reference-physics-recovery-20261012/inputs/"
    "canonical_manipulation_assets.py"
)
DEFAULT_OUTPUT = Path(
    "/mnt/c/Users/HP/Desktop/RobotSim/reviews/"
    "issue46-x2-reference-physics-recovery-20261012/static_candidate"
)
APPROACH_STANDOFF_M = 0.120
APPROACH_STEP_M = 0.015
DIAMETER_TOLERANCE_M = 0.001


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def geom_ids(model: mujoco.MjModel, isolated, body: str | None = None) -> list[int]:
    if body is not None:
        return isolated.collision_geoms(model, body)
    return isolated.collision_geoms(model)


def set_aperture(model: mujoco.MjModel, data: mujoco.MjData, base, aperture: float) -> None:
    targets = base.aperture_targets(aperture)
    data.qpos[base.qpos_id(model, base.RIGHT_DRIVER)] = targets["right_claw_joint_target_rad"]
    data.qpos[base.qpos_id(model, base.RIGHT_FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def pair_distance(model: mujoco.MjModel, data: mujoco.MjData,
                  first: int, second: int, isolated) -> dict[str, Any]:
    segment = np.zeros(6, dtype=float)
    reported = float(mujoco.mj_geomDistance(model, data, first, second, 1.0, segment))
    witness_length = float(np.linalg.norm(segment[:3] - segment[3:]))
    active_distances = []
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair == {int(first), int(second)}:
            active_distances.append(float(contact.dist))
    if active_distances:
        distance = min(active_distances)
        source = "active_mujoco_contact_distance"
    elif reported >= 0.0 and abs(reported - witness_length) > 1e-5:
        distance = witness_length
        source = "witness_segment_norm_fallback_reported_scalar_inconsistent"
    else:
        distance = reported
        source = "mj_geomDistance"
    return {
        "geom1": isolated.geom_name(model, first),
        "body1": isolated.body_name(model, first),
        "geom2": isolated.geom_name(model, second),
        "body2": isolated.body_name(model, second),
        "signed_distance_m": distance,
        "mj_geomDistance_reported_m": reported,
        "witness_segment_length_m": witness_length,
        "distance_source": source,
        "penetration_m": max(0.0, -distance),
        "witness_segment_world_m": segment.tolist(),
    }


def min_pair(model: mujoco.MjModel, data: mujoco.MjData, firsts: list[int],
             seconds: list[int], isolated) -> dict[str, Any] | None:
    rows = [pair_distance(model, data, a, b, isolated) for a in firsts for b in seconds]
    return min(rows, key=lambda row: row["signed_distance_m"]) if rows else None


def active_contacts(model: mujoco.MjModel, data: mujoco.MjData, isolated) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        rows.append({
            "geom1": isolated.geom_name(model, first),
            "body1": isolated.body_name(model, first),
            "geom2": isolated.geom_name(model, second),
            "body2": isolated.body_name(model, second),
            "distance_m": float(contact.dist),
            "penetration_m": max(0.0, -float(contact.dist)),
        })
    return rows


def state_metrics(model: mujoco.MjModel, data: mujoco.MjData, isolated,
                  bottle_ids: list[int], table_ids: list[int], robot_ids: list[int],
                  bottle_body_ids: list[int], pad_body_names: tuple[str, str],
                  adjacent_link_pairs: set[frozenset[str]]) -> dict[str, Any]:
    pad = {name: geom_ids(model, isolated, name) for name in pad_body_names}
    nonpad = [gid for gid in robot_ids
              if isolated.body_name(model, gid) not in pad_body_names]
    pad_to_bottle = {
        name: min_pair(model, data, gids, bottle_ids, isolated)
        for name, gids in pad.items()
    }
    pad_to_body = {
        name: min_pair(model, data, gids, bottle_body_ids, isolated)
        for name, gids in pad.items()
    }
    nonpad_to_bottle = min_pair(model, data, nonpad, bottle_ids, isolated)
    robot_to_table = min_pair(model, data, robot_ids, table_ids, isolated)
    contacts = active_contacts(model, data, isolated)
    robot_bodies = {isolated.body_name(model, gid) for gid in robot_ids}
    self_contacts = []
    for row in contacts:
        if (row["body1"] in robot_bodies and row["body2"] in robot_bodies
                and row["body1"] != row["body2"]):
            pair = frozenset((row["body1"], row["body2"]))
            row["source_adjacent_parent_child"] = pair in adjacent_link_pairs
            row["full_model_contact_filter_expected"] = pair in adjacent_link_pairs
            self_contacts.append(row)
    external_unintended = [row for row in contacts
                           if ((row["body1"] in robot_bodies and row["body2"] == "m0_table")
                               or (row["body2"] in robot_bodies and row["body1"] == "m0_table")
                               or (row["body1"] in robot_bodies and row["body2"].startswith("m0_bottle")
                                   and row["body1"] not in pad_body_names)
                               or (row["body2"] in robot_bodies and row["body1"].startswith("m0_bottle")
                                   and row["body2"] not in pad_body_names))]
    return {
        "aperture_ratio": None,
        "pad_to_any_bottle_geom": pad_to_bottle,
        "pad_to_bottle_body_geom": pad_to_body,
        "minimum_nonpad_robot_to_bottle": nonpad_to_bottle,
        "minimum_robot_to_table": robot_to_table,
        "active_contacts": contacts,
        "active_robot_self_contacts": self_contacts,
        "active_unintended_external_contacts": external_unintended,
    }


def source_limits(urdf: Path) -> dict[str, list[float]]:
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf).getroot()
    out = {}
    for row in root.findall("joint"):
        limit = row.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            out[row.get("name", "")] = [float(limit.get("lower")), float(limit.get("upper"))]
    return out


def mesh_face_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) and int(faces.max()) >= len(vertices):
        faces = faces - int(faces.min())
    centroids = vertices[faces].mean(axis=1) if len(faces) else np.empty((0, 3))
    return np.vstack((vertices, centroids))


def bottle_contains_points(model: mujoco.MjModel, data: mujoco.MjData,
                           bottle_ids: list[int], points_world: np.ndarray) -> dict[str, Any]:
    inside_by_geom: dict[str, np.ndarray] = {}
    for gid in bottle_ids:
        rotation = data.geom_xmat[gid].reshape(3, 3)
        local = (points_world - data.geom_xpos[gid]) @ rotation
        size = model.geom_size[gid]
        kind = int(model.geom_type[gid])
        if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            inside = (local[:, 0] ** 2 + local[:, 1] ** 2 <= size[0] ** 2) & (np.abs(local[:, 2]) <= size[1])
        elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            inside = np.sum((local / size[:3]) ** 2, axis=1) <= 1.0
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            inside = np.sum(local ** 2, axis=1) <= size[0] ** 2
        else:
            continue
        inside_by_geom["m0_bottle/" + str(mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, gid))] = inside
    if not inside_by_geom:
        return {"sample_count": len(points_world), "inside_union_count": None, "by_geom_count": {}}
    union = np.logical_or.reduce(list(inside_by_geom.values()))
    return {
        "sample_count": int(len(points_world)),
        "inside_union_count": int(union.sum()),
        "inside_union_fraction": float(union.mean()),
        "by_geom_count": {name: int(mask.sum()) for name, mask in inside_by_geom.items()},
    }


def wrist_surface_comparison(model: mujoco.MjModel, data: mujoco.MjData,
                              isolated, source_urdf: Path,
                              bottle_ids: list[int]) -> dict[str, Any]:
    import xml.etree.ElementTree as ET
    import trimesh
    from scipy.spatial import cKDTree

    root = ET.parse(source_urdf).getroot()
    link = next(row for row in root.findall("link") if row.get("name") == "right_wrist_roll_link")
    collision = link.find("collision")
    mesh_element = collision.find(".//mesh") if collision is not None else None
    if mesh_element is None:
        raise RuntimeError("Pinned wrist-roll collision mesh is missing")
    mesh_path = source_urdf.parent / "meshes" / Path(mesh_element.get("filename", "")).name
    origin = collision.find("origin")
    xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r_origin = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
    source_mesh = trimesh.load_mesh(str(mesh_path), process=False)
    source_vertices = np.asarray(source_mesh.vertices, dtype=float) * scale
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    source_local_samples = mesh_face_samples(source_vertices, source_faces)
    wrist_body = isolated.body_id(model, "right_wrist_roll_link")
    body_rotation = data.xmat[wrist_body].reshape(3, 3)
    source_world = ((source_local_samples @ r_origin.T + xyz) @ body_rotation.T
                    + data.xpos[wrist_body])

    wrist_geom = next(gid for gid in isolated.collision_geoms(model, "right_wrist_roll_link")
                      if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH))
    mesh_id = int(model.geom_dataid[wrist_geom])
    vert_start, vert_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    hull_vertices = np.asarray(model.mesh_vert[vert_start:vert_start + vert_count], dtype=float)
    hull_faces = np.asarray(model.mesh_face[face_start:face_start + face_count], dtype=np.int64)
    if len(hull_faces) and int(hull_faces.max()) >= len(hull_vertices):
        hull_faces = hull_faces - vert_start
    hull_local_samples = mesh_face_samples(hull_vertices, hull_faces)
    geom_rotation = data.geom_xmat[wrist_geom].reshape(3, 3)
    hull_world = hull_local_samples @ geom_rotation.T + data.geom_xpos[wrist_geom]

    source_inside = bottle_contains_points(model, data, bottle_ids, source_world)
    hull_inside = bottle_contains_points(model, data, bottle_ids, hull_world)
    source_tree = cKDTree(source_world)
    hull_tree = cKDTree(hull_world)
    source_to_hull = hull_tree.query(source_world, k=1)[0]
    hull_to_source = source_tree.query(hull_world, k=1)[0]
    return {
        "classification": "OFFLINE_SURFACE_SAMPLE_DIAGNOSTIC_NOT_EXACT_BOOLEAN",
        "source_collision_mesh": str(mesh_path),
        "source_collision_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "source_collision_origin_xyz_rpy": {"xyz_m": xyz.tolist(), "rpy_rad": rpy.tolist(), "scale": scale.tolist()},
        "source_mesh_vertex_count": int(len(source_vertices)),
        "source_mesh_triangle_count": int(len(source_faces)),
        "source_surface_sampling": "all source vertices plus one centroid per STL triangle, transformed by source link pose and URDF collision origin",
        "compiled_hull_vertex_count": int(vert_count),
        "compiled_hull_face_count": int(face_count),
        "compiled_hull_surface_sampling": "all compiled hull vertices plus one centroid per compiled face, transformed with the static MuJoCo geom pose",
        "source_surface_samples_inside_canonical_bottle": source_inside,
        "compiled_hull_surface_samples_inside_canonical_bottle": hull_inside,
        "source_sample_to_compiled_hull_sample_distance_m": {
            "p50": float(np.percentile(source_to_hull, 50)),
            "p95": float(np.percentile(source_to_hull, 95)),
            "p99": float(np.percentile(source_to_hull, 99)),
            "max": float(source_to_hull.max()),
        },
        "compiled_hull_sample_to_source_sample_distance_m": {
            "p50": float(np.percentile(hull_to_source, 50)),
            "p95": float(np.percentile(hull_to_source, 95)),
            "p99": float(np.percentile(hull_to_source, 99)),
            "max": float(hull_to_source.max()),
        },
        "source_vs_hull_world_aabb_m": {
            "source_min": source_world.min(axis=0).tolist(),
            "source_max": source_world.max(axis=0).tolist(),
            "compiled_hull_min": hull_world.min(axis=0).tolist(),
            "compiled_hull_max": hull_world.max(axis=0).tolist(),
        },
    }


def render_collision_overlays(model: mujoco.MjModel, data: mujoco.MjData,
                              isolated, source_urdf: Path, bottle_ids: list[int],
                              output: Path, candidate_id: int) -> None:
    """Render diagnostic point overlays without changing model or physics."""
    import xml.etree.ElementTree as ET

    from PIL import Image, ImageDraw, ImageFont
    import trimesh

    def origin_transform(element):
        origin = element.find("origin")
        xyz = np.zeros(3) if origin is None else np.fromstring(
            origin.get("xyz", "0 0 0"), sep=" ")
        rpy = np.zeros(3) if origin is None else np.fromstring(
            origin.get("rpy", "0 0 0"), sep=" ")
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        return xyz, rotation

    def urdf_mesh_samples(link_name: str, kind: str) -> np.ndarray:
        root = ET.parse(source_urdf).getroot()
        link = next(row for row in root.findall("link") if row.get("name") == link_name)
        element = link.find(kind)
        mesh_element = element.find(".//mesh") if element is not None else None
        if mesh_element is None:
            return np.empty((0, 3))
        mesh_path = source_urdf.parent / "meshes" / Path(
            mesh_element.get("filename", "")).name
        scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        local = mesh_face_samples(
            np.asarray(mesh.vertices, dtype=float) * scale,
            np.asarray(mesh.faces, dtype=np.int64))
        xyz, rotation = origin_transform(element)
        body = isolated.body_id(model, link_name)
        return ((local @ rotation.T + xyz) @ data.xmat[body].reshape(3, 3).T
                + data.xpos[body])

    def compiled_geom_samples(geom_id: int) -> np.ndarray:
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        faces = np.asarray(model.mesh_face[face_start:face_start + face_count], dtype=np.int64)
        if len(faces) and int(faces.max()) >= len(vertices):
            faces = faces - start
        local = mesh_face_samples(vertices, faces)
        return local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]

    def bottle_surface_samples() -> np.ndarray:
        rows = []
        theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        for gid in bottle_ids:
            kind = int(model.geom_type[gid])
            size = model.geom_size[gid]
            if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                z = np.linspace(-size[1], size[1], 24)
                local = np.array([(size[0] * np.cos(t), size[0] * np.sin(t), h)
                                  for h in z for t in theta])
                cap = np.array([(size[0] * r * np.cos(t), size[0] * r * np.sin(t), h)
                                for h in (-size[1], size[1])
                                for r in np.linspace(0.0, 1.0, 8) for t in theta])
                local = np.vstack((local, cap))
            elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                lat = np.linspace(-np.pi / 2, np.pi / 2, 24)
                local = np.array([(size[0] * np.cos(phi) * np.cos(t),
                                   size[1] * np.cos(phi) * np.sin(t),
                                   size[2] * np.sin(phi)) for phi in lat for t in theta])
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                lat = np.linspace(-np.pi / 2, np.pi / 2, 24)
                local = np.array([(size[0] * np.cos(phi) * np.cos(t),
                                   size[0] * np.cos(phi) * np.sin(t),
                                   size[0] * np.sin(phi)) for phi in lat for t in theta])
            else:
                continue
            rows.append(local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
        return np.vstack(rows) if rows else np.empty((0, 3))

    wrist_body = isolated.body_id(model, "right_wrist_roll_link")
    visual = urdf_mesh_samples("right_wrist_roll_link", "visual")
    source_collision = urdf_mesh_samples("right_wrist_roll_link", "collision")
    wrist_geom = next(gid for gid in isolated.collision_geoms(
        model, "right_wrist_roll_link") if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH))
    compiled_hull = compiled_geom_samples(wrist_geom)
    pads = np.vstack([compiled_geom_samples(gid)
                      for name in isolated.TARGET_JAW_BODIES
                      for gid in isolated.collision_geoms(model, name)
                      if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)])
    loops = np.vstack([compiled_geom_samples(gid)
                       for name in ("R_hand_narrow_loop_Link", "R_hand_wide_loop_Link")
                       for gid in isolated.collision_geoms(model, name)
                       if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)])
    bottle = bottle_surface_samples()
    body_geom = next(gid for gid in bottle_ids if "bottle_body" in isolated.geom_name(model, gid))
    body_rotation = data.geom_xmat[body_geom].reshape(3, 3)
    body_local = (source_collision - data.geom_xpos[body_geom]) @ body_rotation
    body_size = model.geom_size[body_geom]
    inside = ((body_local[:, 0] ** 2 + body_local[:, 1] ** 2 <= body_size[0] ** 2)
              & (np.abs(body_local[:, 2]) <= body_size[1]))
    penetration_samples = source_collision[inside]

    palette = {
        "visual housing": (185, 190, 198),
        "source collision STL": (230, 130, 35),
        "MuJoCo compiled hull": (0, 190, 215),
        "OmniPicker distal pads": (35, 190, 90),
        "OmniPicker loops": (130, 95, 180),
        "canonical bottle": (24, 123, 213),
        "source collision samples inside bottle": (230, 35, 35),
    }
    series = [
        ("canonical bottle", bottle, palette["canonical bottle"], 2),
        ("visual housing", visual, palette["visual housing"], 2),
        ("OmniPicker loops", loops, palette["OmniPicker loops"], 1),
        ("source collision STL", source_collision, palette["source collision STL"], 1),
        ("MuJoCo compiled hull", compiled_hull, palette["MuJoCo compiled hull"], 1),
        ("OmniPicker distal pads", pads, palette["OmniPicker distal pads"], 2),
        ("source collision samples inside bottle", penetration_samples,
         palette["source collision samples inside bottle"], 3),
    ]
    font = ImageFont.load_default()
    for view_name, axes in (("front", (0, 2)), ("side", (1, 2))):
        projected = [points[:, axes] for _, points, _, _ in series if len(points)]
        joined = np.vstack(projected)
        lower, upper = joined.min(axis=0), joined.max(axis=0)
        padding = np.array([0.018, 0.018])
        lower -= padding
        upper += padding
        width, height, margin = 1400, 920, 110
        pixels_per_m = min((width - 2 * margin) / (upper[0] - lower[0]),
                           (height - 2 * margin) / (upper[1] - lower[1]))

        def project(points):
            if not len(points):
                return np.empty((0, 2), dtype=int)
            coords = points[:, axes]
            px = margin + (coords[:, 0] - lower[0]) * pixels_per_m
            py = height - margin - (coords[:, 1] - lower[1]) * pixels_per_m
            return np.column_stack((px, py)).astype(int)

        image = Image.new("RGB", (width, height), (250, 251, 252))
        draw = ImageDraw.Draw(image)
        draw.text((35, 22),
                  f"Candidate {candidate_id}: source-faithful collision overlay - {view_name}",
                  fill=(20, 25, 35), font=font)
        for value in np.arange(math.floor(lower[0] / 0.025) * 0.025, upper[0], 0.025):
            x = int(margin + (value - lower[0]) * pixels_per_m)
            draw.line((x, margin, x, height - margin), fill=(228, 231, 235), width=1)
            draw.text((x + 2, height - margin + 6), f"{value:.2f}", fill=(70, 75, 82), font=font)
        for value in np.arange(math.floor(lower[1] / 0.025) * 0.025, upper[1], 0.025):
            y = int(height - margin - (value - lower[1]) * pixels_per_m)
            draw.line((margin, y, width - margin, y), fill=(228, 231, 235), width=1)
            draw.text((8, y - 5), f"{value:.2f}", fill=(70, 75, 82), font=font)
        for _, points, color, radius in series:
            coords = project(points)
            stride = max(1, len(coords) // 12000)
            for px, py in coords[::stride]:
                if radius == 1:
                    draw.point((int(px), int(py)), fill=color)
                else:
                    draw.ellipse((int(px - radius), int(py - radius),
                                  int(px + radius), int(py + radius)), fill=color)
        title = ("X (m), Z (m)" if view_name == "front" else "Y (m), Z (m)")
        draw.text((width - 175, height - 32), title, fill=(35, 40, 45), font=font)
        legend_x, legend_y = 35, 55
        for index, (name, _, color, _) in enumerate(series):
            row = index % 4
            col = index // 4
            x, y = legend_x + col * 355, legend_y + row * 22
            draw.rectangle((x, y + 3, x + 12, y + 15), fill=color)
            draw.text((x + 18, y), name, fill=(35, 40, 45), font=font)
        draw.text((width - 490, 22),
                  "orange source STL / cyan compiled surface coincide; red = collision samples inside bottle",
                  fill=(120, 35, 25), font=font)
        image.save(output / f"annotated_collision_overlay_{view_name}.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2-root", type=Path, default=Path(
        "/tmp/robotsim-issue46-reference-study-20261012/x2source"))
    parser.add_argument("--canonical-helper", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-id", type=int, choices=(1, 2), default=1)
    parser.add_argument("--approach-azimuth-deg", type=float, choices=(0.0, -90.0), default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    baseline = load_module("issue46_m0_baseline", REPO / "scripts/research/issue46_omnipicker_1dof_m0.py")
    isolated = load_module("issue46_isolated_grasp", REPO / "scripts/research/issue46_omnipicker_isolated_grasp.py")
    baseline.X2_ROOT = args.x2_root
    baseline.STATIC_ROOT = args.canonical_helper.parent
    isolated.X2_ROOT = args.x2_root
    isolated.STATIC_ROOT = args.canonical_helper.parent
    isolated.CANONICAL_MODULE = args.canonical_helper
    source_urdf = args.x2_root / baseline.X2_URDF_REL

    identity = baseline.verify_inputs(args.output)
    isolated_info = isolated.filtered_urdf(args.output, source_urdf)
    filtered = Path(isolated_info["derived_urdf_path"])
    probe_spec = isolated.import_spec(
        baseline, filtered, np.zeros(3), np.eye(3), include_bottle=True, add_lift=False)
    probe_model = probe_spec.compile()
    probe_data = mujoco.MjData(probe_model)
    mujoco.mj_resetData(probe_model, probe_data)
    for name in isolated.WRIST_JOINTS:
        jid = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            probe_data.qpos[int(probe_model.jnt_qposadr[jid])] = 0.0
    aperture_targets = baseline.aperture_targets(1.0)
    probe_data.qpos[baseline.qpos_id(probe_model, baseline.RIGHT_DRIVER)] = aperture_targets["right_claw_joint_target_rad"]
    probe_data.qpos[baseline.qpos_id(probe_model, baseline.RIGHT_FOLLOWER)] = aperture_targets["R_hand_wide1_joint_target_rad"]
    mujoco.mj_forward(probe_model, probe_data)

    pad_names = isolated.TARGET_JAW_BODIES
    pad_ids = {name: geom_ids(probe_model, isolated, name) for name in pad_names}
    if any(len(ids) != 1 for ids in pad_ids.values()):
        raise RuntimeError(f"Expected one active collision geom per functional distal jaw: {pad_ids}")
    narrow_id, wide_id = pad_ids[pad_names[0]][0], pad_ids[pad_names[1]][0]
    bottle_ids = isolated.bottle_geom_ids(probe_model)
    body_geom = next(g for g in bottle_ids if "bottle_body" in isolated.geom_name(probe_model, g))
    bottle_body_ids = [body_geom]
    body_center = probe_data.geom_xpos[body_geom].copy()
    bottle_radius = float(probe_model.geom_size[body_geom][0])
    bottle_diameter = 2.0 * bottle_radius

    gap_samples = []
    for aperture in np.linspace(1.0, 0.0, 1001):
        set_aperture(probe_model, probe_data, baseline, float(aperture))
        row = pair_distance(probe_model, probe_data, narrow_id, wide_id, isolated)
        gap_samples.append({"aperture_ratio": float(aperture), "pad_gap_m": row["signed_distance_m"],
                            "witness_segment_world_m": row["witness_segment_world_m"]})
    crossing = next(((a, b) for a, b in zip(gap_samples, gap_samples[1:])
                     if a["pad_gap_m"] >= bottle_diameter >= b["pad_gap_m"]), None)
    if crossing is None:
        raise RuntimeError("Source-limited jaw sweep does not reach the canonical bottle diameter")
    a, b = crossing
    fraction = (a["pad_gap_m"] - bottle_diameter) / (a["pad_gap_m"] - b["pad_gap_m"])
    contact_aperture = a["aperture_ratio"] + fraction * (b["aperture_ratio"] - a["aperture_ratio"])
    set_aperture(probe_model, probe_data, baseline, 1.0)
    narrow_body_probe = isolated.body_id(probe_model, pad_names[0])
    wide_body_probe = isolated.body_id(probe_model, pad_names[1])
    relative_pad_separation_open = (
        probe_data.xpos[wide_body_probe] - probe_data.xpos[narrow_body_probe]).copy()
    set_aperture(probe_model, probe_data, baseline, contact_aperture)
    relative_pad_separation_contact = (
        probe_data.xpos[wide_body_probe] - probe_data.xpos[narrow_body_probe]).copy()
    gap_row = pair_distance(probe_model, probe_data, narrow_id, wide_id, isolated)
    gap_witness = np.asarray(gap_row["witness_segment_world_m"], dtype=float)
    midpoint_local = 0.5 * (gap_witness[:3] + gap_witness[3:])
    jaw_axis_local = relative_pad_separation_open - relative_pad_separation_contact
    jaw_axis_local /= np.linalg.norm(jaw_axis_local)
    tool_id = isolated.body_id(probe_model, "R_omnipicker_base_link")
    tool_z_local = probe_data.xmat[tool_id].reshape(3, 3)[:, 2].copy()
    jaw_axis_local -= np.dot(jaw_axis_local, tool_z_local) * tool_z_local
    jaw_axis_local /= np.linalg.norm(jaw_axis_local)
    local_x = np.cross(jaw_axis_local, tool_z_local)
    local_x /= np.linalg.norm(local_x)
    local_basis = np.column_stack((local_x, jaw_axis_local, tool_z_local))
    azimuth = math.radians(args.approach_azimuth_deg)
    target_z = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
    target_y = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    target_x = np.cross(target_y, target_z)
    target_basis = np.column_stack((target_x, target_y, target_z))
    rotation = target_basis @ local_basis.T
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("Tool-axis/jaw-motion basis produced an improper rotation")
    root_position = body_center - rotation @ midpoint_local

    controller_limits = {
        "per_joint": {
            baseline.RIGHT_DRIVER: {"effort_cap_nm": 1.0},
            baseline.RIGHT_FOLLOWER: {"effort_cap_nm": 1.0},
        }
    }
    model, build_info = isolated.build_model(
        baseline, filtered, root_position, rotation, include_bottle=True, add_lift=False)
    data = isolated.initialize(baseline, model, controller_limits)
    root_id = isolated.body_id(model, isolated.ROOT_LINK)
    bottle_ids = isolated.bottle_geom_ids(model)
    table_ids = isolated.table_geom_ids(model)
    robot_ids = isolated.robot_geom_ids(model)
    body_geom = next(g for g in bottle_ids if "bottle_body" in isolated.geom_name(model, g))
    body_center_world = data.geom_xpos[body_geom].copy()
    import xml.etree.ElementTree as ET
    source_root = ET.parse(source_urdf).getroot()
    adjacent_link_pairs = {
        frozenset((joint.find("parent").get("link"), joint.find("child").get("link")))
        for joint in source_root.findall("joint")
        if joint.find("parent") is not None and joint.find("child") is not None
    }
    approach_axis_world = rotation @ tool_z_local

    def evaluate(position: np.ndarray, aperture: float) -> dict[str, Any]:
        isolated.set_root_pose(model, root_id, data, position, isolated.quat_from_matrix(rotation))
        set_aperture(model, data, baseline, aperture)
        metrics = state_metrics(model, data, isolated, bottle_ids, table_ids, robot_ids,
                                [body_geom], pad_names, adjacent_link_pairs)
        metrics["aperture_ratio"] = aperture
        metrics["root_position_world_m"] = np.asarray(position).tolist()
        return metrics

    approach_count = int(round(APPROACH_STANDOFF_M / APPROACH_STEP_M))
    approach_rows = []
    for index in range(approach_count + 1):
        fraction_path = index / approach_count
        position = root_position - approach_axis_world * APPROACH_STANDOFF_M * (1.0 - fraction_path)
        approach_rows.append({"stage": "OPEN_APPROACH", "sample": index,
                              "path_fraction": fraction_path,
                              **evaluate(position, 1.0)})
    closure_rows = []
    for index, aperture in enumerate(np.linspace(1.0, contact_aperture, 101)):
        closure_rows.append({"stage": "STATIC_CLOSE", "sample": index,
                             **evaluate(root_position, float(aperture))})

    open_state = approach_rows[-1]
    contact_state = closure_rows[-1]
    nonpad_clearance = [row["minimum_nonpad_robot_to_bottle"] for row in approach_rows + closure_rows]
    table_clearance = [row["minimum_robot_to_table"] for row in approach_rows + closure_rows]
    self_contacts = [contact for row in approach_rows + closure_rows
                     for contact in row["active_robot_self_contacts"]]
    unexpected_self_contacts = [row for row in self_contacts
                                if not row["source_adjacent_parent_child"]]
    unintended = [contact for row in approach_rows + closure_rows
                  for contact in row["active_unintended_external_contacts"]]
    contact_distances = {name: row["signed_distance_m"] if row else None
                         for name, row in contact_state["pad_to_bottle_body_geom"].items()}
    open_pad_distances = {name: row["signed_distance_m"] if row else None
                          for name, row in open_state["pad_to_bottle_body_geom"].items()}
    source_limit_map = source_limits(source_urdf)
    selected_joint_names = (baseline.RIGHT_DRIVER, baseline.RIGHT_FOLLOWER,
                            "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
    joint_rows = {}
    for name in selected_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        q = float(data.qpos[int(model.jnt_qposadr[jid])])
        interval = source_limit_map.get(name)
        joint_rows[name] = {"qpos_rad": q, "source_range_rad": interval,
                            "inside_source_range": interval is None or interval[0] <= q <= interval[1]}

    jsonl = args.output / "static_geometry_trace.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for row in gap_samples:
            stream.write(json.dumps({"stage": "SOURCE_JAW_SWEEP", **row}, sort_keys=True) + "\n")
        for row in approach_rows + closure_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    pad_contact_ok = (len(contact_distances) == 2 and all(
        abs(float(value)) <= DIAMETER_TOLERANCE_M for value in contact_distances.values()))
    open_clear_ok = all(value is not None and float(value) > 0.0 for value in open_pad_distances.values())
    nonpad_ok = all(row is None or row["signed_distance_m"] > 0.0 for row in nonpad_clearance)
    table_ok = all(row is None or row["signed_distance_m"] > 0.0 for row in table_clearance)
    no_self_contact = not unexpected_self_contacts
    no_unintended_contact = not unintended
    limits_ok = all(row["inside_source_range"] for row in joint_rows.values())
    mesh_comparison = wrist_surface_comparison(model, data, isolated, source_urdf, bottle_ids)
    gates = {
        "source_joint_limits": limits_ok,
        "both_pads_clear_in_open": open_clear_ok,
        "both_pads_reach_bottle_body_at_source_limited_closure": pad_contact_ok,
        "nonpad_bottle_clearance_through_approach_and_closure": nonpad_ok,
        "table_clearance_through_approach_and_closure": table_ok,
        "no_unfiltered_active_robot_self_contacts": no_self_contact,
        "no_active_unintended_external_contacts": no_unintended_contact,
    }
    result = {
        "classification": "REFERENCE_DERIVED_STATIC_GEOMETRY_ONLY",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "physics_steps": 0,
        "object_qpos_writes": 0,
        "candidate_count": 1,
        "candidate_id": args.candidate_id,
        "gates": gates,
        "identity": {
            "robotsim_head_before_experiment": __import__("subprocess").check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "pinned_x2_commit": identity["commit"],
            "x2_urdf_sha256": identity["urdf_sha256"],
            "canonical_helper_sha256": identity["canonical_scene_helper_sha256"],
            "python": sys.version,
            "platform": __import__("platform").platform(),
            "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(),
            "mujoco_native_library_sha256": sha256(
                Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
            "filtered_subtree_sha256": isolated_info["derived_urdf_sha256"],
        },
        "model_scope": "Pinned official right_wrist_yaw_link descendant subtree, source meshes/joints/mimic retained; 11 previously accepted SIMULATION_ONLY exclusions retained; no new exclusions or geometry edits.",
        "frame_definition": {
            "upstream_demo_tool_axis": "local +Z of R_omnipicker_base_link aligned to the selected horizontal approach direction",
            "approach_azimuth_deg": args.approach_azimuth_deg,
            "roll_resolution": "actual compiled narrow3/wide3 witness separation at the source-limited aperture whose gap equals canonical bottle-body diameter; no PCA",
            "target_jaw_axis_world": target_y.tolist(),
            "tool_axis_local_root": tool_z_local.tolist(),
            "jaw_axis_local_root": jaw_axis_local.tolist(),
            "jaw_axis_derivation": "difference between wide-minus-narrow distal-link origins at OPEN and the 70 mm source-limited closure configuration",
            "relative_pad_separation_open_local_m": relative_pad_separation_open.tolist(),
            "relative_pad_separation_contact_local_m": relative_pad_separation_contact.tolist(),
            "tool_axis_jaw_axis_abs_dot": abs(float(np.dot(tool_z_local, jaw_axis_local))),
            "target_rotation_matrix": rotation.tolist(),
            "target_determinant": float(np.linalg.det(rotation)),
            "contact_aperture_ratio": contact_aperture,
            "source_joint_coordinates_at_contact": {
                name: float(data.qpos[baseline.qpos_id(model, name)])
                for name in (baseline.RIGHT_DRIVER, baseline.RIGHT_FOLLOWER)
            },
            "compiled_pad_gap_at_contact_m": gap_row["signed_distance_m"],
            "compiled_pad_gap_distance_source": gap_row["distance_source"],
            "compiled_pad_gap_witness_length_m": gap_row["witness_segment_length_m"],
            "bottle_body_diameter_m": bottle_diameter,
            "functional_pad_midpoint_local_m": midpoint_local.tolist(),
            "bottle_body_center_world_m": body_center_world.tolist(),
            "root_position_world_m": root_position.tolist(),
            "root_quaternion_wxyz": isolated.quat_from_matrix(rotation).tolist(),
            "approach_axis_world": approach_axis_world.tolist(),
            "standoff_m": APPROACH_STANDOFF_M,
            "approach_step_m": APPROACH_STEP_M,
        },
        "selected_joint_state": joint_rows,
        "open_pad_clearance_m": open_pad_distances,
        "closed_pad_to_bottle_distance_m": contact_distances,
        "minimum_nonpad_robot_to_bottle_m": min(
            (row["signed_distance_m"] for row in nonpad_clearance if row is not None), default=None),
        "minimum_robot_to_table_m": min(
            (row["signed_distance_m"] for row in table_clearance if row is not None), default=None),
        "active_robot_self_contacts": self_contacts,
        "unexpected_nonadjacent_robot_self_contacts": unexpected_self_contacts,
        "active_unintended_external_contacts": unintended,
        "wrist_source_triangles_vs_compiled_hull": mesh_comparison,
        "jaw_gap_samples": len(gap_samples),
        "approach_samples": len(approach_rows),
        "closure_samples": len(closure_rows),
        "raw_trace": str(jsonl),
        "build_info": build_info,
        "warning": "Static compiled-collision geometry does not establish contact-force capacity or physical grasp success.",
    }
    write_json(args.output / "result.json", result)

    lookat = body_center_world.tolist()
    isolated.set_root_pose(model, root_id, data, root_position, isolated.quat_from_matrix(rotation))
    set_aperture(model, data, baseline, 1.0)
    isolated.render_png(model, data, args.output, "overview_open.png", lookat, 0.85, 135.0, -18.0)
    set_aperture(model, data, baseline, contact_aperture)
    isolated.render_png(model, data, args.output, "contact_geometry_front.png", lookat, 0.55, 180.0, -8.0)
    isolated.render_png(model, data, args.output, "contact_geometry_side.png", lookat, 0.55, 90.0, -8.0)
    render_collision_overlays(model, data, isolated, source_urdf, bottle_ids,
                              args.output, args.candidate_id)
    (args.output / "experiment_command.txt").write_text(
        "MUJOCO_GL=egl /tmp/robotsim-issue46-coacd-venv/bin/python -B "
        f"{Path(__file__).resolve()} --x2-root {args.x2_root} "
        f"--canonical-helper {args.canonical_helper} --output {args.output} "
        f"--candidate-id {args.candidate_id} --approach-azimuth-deg {args.approach_azimuth_deg}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "gates": gates,
                      "root_position_world_m": root_position.tolist(),
                      "open_pad_clearance_m": open_pad_distances,
                      "contact_pad_distance_m": contact_distances,
                      "minimum_nonpad_robot_to_bottle_m": result["minimum_nonpad_robot_to_bottle_m"],
                      "minimum_robot_to_table_m": result["minimum_robot_to_table_m"],
                      "evidence": str(args.output)}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
