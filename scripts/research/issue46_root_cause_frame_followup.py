#!/usr/bin/env python3
"""Bounded source contact-frame and wrist-collision counterfactual for Issue #46."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d")
SOURCE_URDF = SOURCE_ROOT / "X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
ABC_PATH = REPO / "scripts/research/issue46_root_cause_abc.py"
AUDIT_PATH = REPO / "scripts/research/issue46_omnipicker_collision_audit_recovery.py"
BASELINE_PATH = REPO / "scripts/research/issue46_omnipicker_1dof_m0.py"
ROOT_LINK = "right_elbow_link"
WRIST_LINK = "right_wrist_roll_link"
VISUAL_WRIST_MESH = "right_wrist_roll_extend_link.stl"
COLLISION_WRIST_MESH = "right_wrist_roll_link.stl"
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def body_vertices_in_world(abc, model, data, link: str, mesh_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    urdf = ET.parse(SOURCE_URDF).getroot()
    link_node = next(row for row in urdf.findall("link") if row.get("name") == link)
    block = link_node.find("visual" if mesh_name == VISUAL_WRIST_MESH else "collision")
    mesh = block.find("geometry/mesh")
    path = SOURCE_URDF.parent / "meshes" / mesh_name
    triangles = abc.stl_triangles(path)
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    scale = np.asarray([float(value) for value in mesh.get("scale", "1 1 1").split()])
    vertices = vertices * scale
    origin = block.find("origin")
    xyz = np.asarray([float(value) for value in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
    rpy = [float(value) for value in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
    vertices = vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
    bid = abc.body_id(model, link)
    Rbody = data.xmat[bid].reshape(3, 3)
    world = vertices @ Rbody.T + data.xpos[bid]
    hull = ConvexHull(vertices)
    return world, {
        "mesh": mesh_name,
        "sha256": sha(path),
        "triangles": int(len(triangles)),
        "unique_vertices": int(len(vertices)),
        "source_extents_m": np.ptp(vertices, axis=0).tolist(),
        "convex_hull_volume_m3": float(hull.volume),
        "source_collision_origin_xyz_rpy": [xyz.tolist(), rpy],
    }


def mesh_to_bottle_metrics(abc, model, data, link: str, mesh_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    world_vertices, identity = body_vertices_in_world(abc, model, data, link, mesh_name)
    tri = abc.stl_triangles(SOURCE_URDF.parent / "meshes" / mesh_name)
    pts = np.concatenate((tri.reshape(-1, 3), tri.mean(axis=1)))
    urdf = ET.parse(SOURCE_URDF).getroot()
    link_node = next(row for row in urdf.findall("link") if row.get("name") == link)
    block = link_node.find("visual" if mesh_name == VISUAL_WRIST_MESH else "collision")
    mesh = block.find("geometry/mesh")
    scale = np.asarray([float(value) for value in mesh.get("scale", "1 1 1").split()])
    pts = pts * scale
    origin = block.find("origin")
    xyz = np.asarray([float(value) for value in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
    rpy = [float(value) for value in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
    pts = pts @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
    bid = abc.body_id(model, link)
    pts = pts @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
    bottle = abc.geom_id(model, "bottle_body")
    center = data.geom_xpos[bottle].copy()
    rotation = data.geom_xmat[bottle].reshape(3, 3).copy()
    sdf = abc.cylinder_sdf(pts, center, rotation, float(model.geom_size[bottle, 0]), float(model.geom_size[bottle, 1]))
    identity.update({
        "sampled_points": int(len(pts)),
        "min_signed_distance_to_bottle_body_cylinder_m": float(sdf.min()),
        "sampled_fraction_inside_bottle_body_cylinder": float(np.mean(sdf < 0)),
        "minimum_sample_world_m": pts[int(np.argmin(sdf))].tolist(),
    })
    return world_vertices, identity


def draw_mesh_comparison(path: Path, center: np.ndarray, radius: float, half_height: float,
                         visual: np.ndarray, collision: np.ndarray) -> None:
    width, height = 1400, 720
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    panels = [(80, 115, 665, 650, "Top view: X-Y"), (735, 115, 1320, 650, "Side view: X-Z")]
    colors = {"visual": (56, 102, 128), "collision": (213, 91, 58)}
    draw.text((80, 35), "Pinned right_wrist_roll_link source meshes at source-frame OPEN", fill=(30, 34, 37))
    draw.text((80, 68), "visual STL is blue; collision STL is red; bottle-body silhouette is black", fill=(55, 58, 60))
    draw.text((950, 68), "all distances in metres", fill=(55, 58, 60))
    for left, top, right, bottom, label in panels:
        draw.rectangle((left, top, right, bottom), outline=(90, 95, 98), width=2)
        draw.text((left + 8, top - 24), label, fill=(30, 34, 37))
        if label.startswith("Top"):
            x0, x1 = center[0] - .17, center[0] + .17
            y0, y1 = center[1] - .17, center[1] + .17
            def project(pt): return (left + (pt[0]-x0)/(x1-x0)*(right-left), bottom - (pt[1]-y0)/(y1-y0)*(bottom-top))
            a, b = project((center[0]-radius, center[1]-radius, 0)), project((center[0]+radius, center[1]+radius, 0))
            draw.ellipse((a[0], b[1], b[0], a[1]), outline=(20, 20, 20), width=4)
            ax1, ax2 = 0, 1
        else:
            x0, x1 = center[0] - .17, center[0] + .17
            z0, z1 = center[2] - .20, center[2] + .20
            def project(pt): return (left + (pt[0]-x0)/(x1-x0)*(right-left), bottom - (pt[2]-z0)/(z1-z0)*(bottom-top))
            a, b = project((center[0]-radius, 0, center[2]-half_height)), project((center[0]+radius, 0, center[2]+half_height))
            draw.rectangle((a[0], b[1], b[0], a[1]), outline=(20, 20, 20), width=4)
            ax1, ax2 = 0, 2
        draw.text((left + 10, bottom + 8), f"bottle body: dia={2*radius:.3f}, height={2*half_height:.3f}", fill=(40, 40, 40))
        for points, color in ((visual, colors["visual"]), (collision, colors["collision"])):
            stride = max(1, len(points)//9000)
            for point in points[::stride]:
                px, py = project(point)
                if left <= px <= right and top <= py <= bottom:
                    draw.ellipse((px-1, py-1, px+1, py+1), fill=color)
    image.save(path)


def set_aperture(abc, base, model, data, aperture: float) -> dict[str, Any]:
    targets = base.aperture_targets(float(aperture))
    data.qpos[abc.qpos_id(model, DRIVER)] = targets["right_claw_joint_target_rad"]
    data.qpos[abc.qpos_id(model, FOLLOWER)] = targets["R_hand_wide1_joint_target_rad"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return targets


def state_geometry(abc, base, model, data, aperture: float, label: str) -> dict[str, Any]:
    targets = set_aperture(abc, base, model, data, aperture)
    bottle_geoms = abc.bottle_geoms(model)
    bottle_set = set(bottle_geoms)
    robot_geoms = [g for g in abc.all_collision_geoms(model) if g not in bottle_set
                   and abc.name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) not in ("world", "m0_table")]
    table_geoms = [g for g in abc.all_collision_geoms(model)
                   if abc.name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) == "m0_table"]
    bdist = [abc.geom_distance(model, data, g, b) for g in robot_geoms for b in bottle_geoms]
    bdist = [row for row in bdist if row["query_consistent"]]
    tdist = [abc.geom_distance(model, data, g, t) for g in robot_geoms for t in table_geoms]
    tdist = [row for row in tdist if row["query_consistent"]]
    by_body = {}
    for row in bdist:
        by_body[row["body1"]] = min(float(row["signed_distance_m"]), by_body.get(row["body1"], math.inf))
    contacts = [row for row in abc.contact_rows(model, data, with_force=False)
                if row["body1"] == "m0_bottle" or row["body2"] == "m0_bottle"]
    active_tip_bodies = sorted({row["body1"] if row["body2"] == "m0_bottle" else row["body2"]
                                for row in contacts if (row["body1"] if row["body2"] == "m0_bottle" else row["body2"])
                                in ("R_hand_narrow3_Link", "R_hand_wide3_Link")})
    qdriver = float(data.qpos[abc.qpos_id(model, DRIVER)])
    qfollower = float(data.qpos[abc.qpos_id(model, FOLLOWER)])
    return {
        "label": label,
        "aperture_ratio": float(aperture),
        "driver_target_rad": float(targets["right_claw_joint_target_rad"]),
        "follower_target_rad": float(targets["R_hand_wide1_joint_target_rad"]),
        "driver_inside_source_range": -1.0 <= qdriver <= 0.0,
        "follower_inside_source_range": 0.0 <= qfollower <= 1.0,
        "minimum_robot_bottle_distance_m": min(bdist, key=lambda row: row["signed_distance_m"]) if bdist else None,
        "minimum_bottle_distance_by_robot_body_m": by_body,
        "minimum_robot_table_distance_m": min(tdist, key=lambda row: row["signed_distance_m"]) if tdist else None,
        "active_bottle_contacts": contacts,
        "active_distal_tip_contact_bodies": active_tip_bodies,
        "open_collision_gate_pass": (not contacts and min(row["signed_distance_m"] for row in bdist) > 0.0)
        if label == "OPEN" and bdist else (False if label == "OPEN" else None),
        "static_diagnostic_only": True,
    }


def collision_variant(source_urdf: Path, output: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    link = next(row for row in root.findall("link") if row.get("name") == WRIST_LINK)
    collision_mesh = link.find("collision/geometry/mesh")
    if collision_mesh is None:
        raise RuntimeError("Pinned wrist collision mesh not found")
    collision_mesh.set("filename", VISUAL_WRIST_MESH)
    ET.indent(root, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def connected_components(triangles: np.ndarray) -> list[np.ndarray]:
    parents = list(range(len(triangles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    owner: dict[tuple[float, float, float], int] = {}
    for triangle_id, triangle in enumerate(triangles):
        for vertex in triangle:
            key = tuple(float(value) for value in vertex)
            previous = owner.get(key)
            if previous is None:
                owner[key] = triangle_id
            else:
                union(triangle_id, previous)
    groups: dict[int, list[int]] = {}
    for triangle_id in range(len(triangles)):
        groups.setdefault(find(triangle_id), []).append(triangle_id)
    return [triangles[indices] for indices in sorted(groups.values(), key=len, reverse=True)]


def write_convex_obj(path: Path, points: np.ndarray) -> tuple[int, int, float]:
    vertices = np.unique(points, axis=0)
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    remap = {int(old): new + 1 for new, old in enumerate(hull.vertices)}
    with path.open("w", encoding="ascii") as stream:
        for vertex in hull_vertices:
            stream.write("v %.9g %.9g %.9g\n" % tuple(vertex))
        for face in hull.simplices:
            stream.write("f %d %d %d\n" % tuple(remap[int(index)] for index in face))
    return int(len(hull_vertices)), int(len(hull.simplices)), float(hull.volume)


def component_collision_variant(abc, source_urdf: Path, out: Path) -> tuple[Path, list[dict[str, Any]]]:
    meshdir = out / "connected_component_collision_meshes"
    meshdir.mkdir(parents=True)
    source_meshdir = SOURCE_URDF.parent / "meshes"
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        raise RuntimeError("Extracted X2 source URDF has no MuJoCo compiler element")
    compiler.set("meshdir", str(meshdir.resolve()))
    source_names = sorted({Path(mesh.get("filename", "")).name for mesh in root.findall(".//mesh")})
    for source_name in source_names:
        shutil.copy2(source_meshdir / source_name, meshdir / source_name)
    records: list[dict[str, Any]] = []
    for link in root.findall("link"):
        for collision_index, collision in enumerate(list(link.findall("collision"))):
            source_mesh = collision.find("geometry/mesh")
            if source_mesh is None:
                continue
            filename = Path(source_mesh.get("filename", "")).name
            source_path = source_meshdir / filename
            triangles = abc.stl_triangles(source_path)
            components = connected_components(triangles)
            component_records = []
            replacement: list[ET.Element] = []
            scale = source_mesh.get("scale", "1 1 1")
            for component_index, component in enumerate(components):
                points = np.unique(component.reshape(-1, 3), axis=0)
                if len(points) < 4:
                    raise RuntimeError(f"Degenerate source collision component {link.get('name')} {filename} #{component_index}")
                obj_name = f"{link.get('name')}_{collision_index}_component_{component_index}.obj"
                obj_path = meshdir / obj_name
                hull_vertices, hull_faces, volume = write_convex_obj(obj_path, points)
                clone = copy.deepcopy(collision)
                geometry = clone.find("geometry")
                geometry.clear()
                ET.SubElement(geometry, "mesh", {"filename": obj_name, "scale": scale})
                replacement.append(clone)
                component_records.append({
                    "component_index": component_index,
                    "source_triangle_count": int(len(component)),
                    "source_unique_vertex_count": int(len(points)),
                    "hull_vertex_count": hull_vertices,
                    "hull_face_count": hull_faces,
                    "hull_volume_m3": volume,
                    "obj_sha256": sha(obj_path),
                })
            parent = link
            position = list(parent).index(collision)
            parent.remove(collision)
            for offset, clone in enumerate(replacement):
                parent.insert(position + offset, clone)
            all_vertices = np.unique(triangles.reshape(-1, 3), axis=0)
            records.append({
                "link": link.get("name"),
                "source_mesh": filename,
                "source_mesh_sha256": sha(source_path),
                "source_triangle_count": int(len(triangles)),
                "source_unique_vertex_count": int(len(all_vertices)),
                "single_hull_volume_m3": float(ConvexHull(all_vertices).volume),
                "sum_connected_component_hull_volumes_m3": float(sum(row["hull_volume_m3"] for row in component_records)),
                "component_to_global_hull_volume_ratio": float(sum(row["hull_volume_m3"] for row in component_records)
                                                                / max(float(ConvexHull(all_vertices).volume), 1e-12)),
                "components": component_records,
            })
    ET.indent(root, space="  ")
    output = out / "isolated_omnipicker_connected_component_hulls_DIAGNOSTIC_ONLY.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, records


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)

    abc = load_module("issue46_abc_followup", ABC_PATH)
    audit, base = abc.modules()
    identity = abc.identity()
    identity["followup_runner_sha256"] = sha(Path(__file__).resolve())
    identity["contact_frame_audit_sha256"] = sha(AUDIT_PATH)
    identity["source_collision_wrist_sha256"] = sha(SOURCE_URDF.parent / "meshes" / COLLISION_WRIST_MESH)
    identity["source_visual_wrist_sha256"] = sha(SOURCE_URDF.parent / "meshes" / VISUAL_WRIST_MESH)
    identity["canonical_bottle_scene_sha256"] = sha(abc.CANONICAL)
    if identity["mujoco_python"] != "3.3.6" or identity["x2_commit"] != "575cc6b988f976c23550e0db85aa1e5475d3652d":
        raise RuntimeError("Runtime/model pin mismatch")

    source_dir = out / "source"
    source_dir.mkdir()
    extracted = audit.extract_subtree(source_dir, SOURCE_URDF, ROOT_LINK)
    frame_model, frame_data, compile_flags = audit.compile_isolated(
        base, Path(extracted["path"]), np.zeros(3), np.array([1., 0., 0., 0.]), True
    )
    frame = audit.contact_frame(base, frame_model, frame_data)
    center = np.asarray(frame["bottle_body_center_world_m"], dtype=float)
    local_midpoint = np.asarray(frame["local_frame"]["contact_midpoint_relative_to_root_m"], dtype=float)
    rotation = np.asarray(frame["local_frame"]["rotation_matrix_source_frame_to_horizontal_grasp"], dtype=float)
    candidate = {
        "name": "source_contact_frame_surface_centered",
        "root_position_world_m": (center - rotation @ local_midpoint).tolist(),
        "root_rotation_world": rotation.tolist(),
        "previous_variables": {"aperture": float(frame["contact_aperture_ratio"])},
        "source_frame_origin_rule": "Align the compiled narrow/wide jaw-to-jaw surface-witness midpoint to the canonical bottle-body center; source frame axes come from the compiled jaw witnesses and source fingertip-to-base link chain.",
    }
    candidate = abc.derive_tcp_centered_candidate(audit, base, Path(extracted["path"]), candidate)
    candidate["name"] = "source_contact_frame_and_cylinder_surface_centered"
    candidate["source_frame"] = {
        "aperture_ratio": float(frame["contact_aperture_ratio"]),
        "jaw_gap_m": float(frame["jaw_gap_at_contact_aperture_m"]),
        "jaw_normal_narrow_to_wide_root": frame["local_frame"]["normal_narrow_to_wide"],
        "finger_axis_tip_to_base_root": frame["local_frame"]["finger_axis_tip_to_base_from_compiled_link_chain"],
        "root_rotation_world": rotation.tolist(),
        "root_position_before_cylinder_surface_correction_m": (center - rotation @ local_midpoint).tolist(),
    }
    abc.write_json(out / "source_contact_frame.json", {"frame": frame, "candidate": candidate,
                                                         "frame_compile_flags": compile_flags})

    comparison = {}
    loaded_models = {}
    for label, urdf_path in (("source_collision", Path(extracted["path"])),):
        model, data, _ = abc.x2_modules_and_model(audit, base, urdf_path, candidate)
        states = []
        for aperture, state_name in ((1.0, "OPEN"), (float(candidate["previous_variables"]["aperture"]), "CONTACT_GEOMETRY"), (0.0, "CLOSED")):
            states.append(state_geometry(abc, base, model, data, aperture, state_name))
        comparison[label] = {"urdf_sha256": sha(urdf_path), "states": states,
                             "no_dynamic_steps_run": True, "active_rollout_qpos_writes": 0}
        loaded_models[label] = (model, data)

    variant_dir = out / "counterfactual"
    variant_dir.mkdir()
    variant = collision_variant(Path(extracted["path"]), variant_dir / "isolated_omnipicker_visual_mesh_as_wrist_collision.urdf")
    model, data, _ = abc.x2_modules_and_model(audit, base, variant, candidate)
    states = []
    for aperture, state_name in ((1.0, "OPEN"), (float(candidate["previous_variables"]["aperture"]), "CONTACT_GEOMETRY"), (0.0, "CLOSED")):
        states.append(state_geometry(abc, base, model, data, aperture, state_name))
    comparison["visual_mesh_as_wrist_collision_DIAGNOSTIC_ONLY"] = {
        "urdf_sha256": sha(variant), "states": states,
        "collision_change": f"Only {WRIST_LINK} collision mesh filename changes from {COLLISION_WRIST_MESH} to its distinct official visual mesh {VISUAL_WRIST_MESH}; origin, all other collision meshes, visuals, joints, source ranges, and bottle stay unchanged.",
        "not_source_faithful_for_physics": True,
        "no_dynamic_steps_run": True,
        "active_rollout_qpos_writes": 0,
    }

    component_urdf, component_records = component_collision_variant(abc, Path(extracted["path"]), variant_dir)
    component_model, component_data, _ = abc.x2_modules_and_model(audit, base, component_urdf, candidate)
    component_states = []
    for aperture, state_name in ((1.0, "OPEN"), (float(candidate["previous_variables"]["aperture"]), "CONTACT_GEOMETRY"), (0.0, "CLOSED")):
        component_states.append(state_geometry(abc, base, component_model, component_data, aperture, state_name))
    comparison["connected_source_component_hulls_DIAGNOSTIC_ONLY"] = {
        "urdf_sha256": sha(component_urdf),
        "collision_change": "Every source collision STL in the extracted right elbow/wrist/OmniPicker subtree is represented by one convex hull per exact vertex-connected triangle component. No source component is omitted; source triangles, link origins, mesh scales, joint tree, source limits, visuals, and bottle are retained. This is a source-derived collision approximation and diagnostic-only.",
        "states": component_states,
        "mesh_component_records": component_records,
        "not_source_faithful_for_physics": True,
        "no_dynamic_steps_run": True,
        "active_rollout_qpos_writes": 0,
    }

    original_model, original_data = loaded_models["source_collision"]
    set_aperture(abc, base, original_model, original_data, 1.0)
    open_state = comparison["source_collision"]["states"][0]
    visual_points, visual_stats = mesh_to_bottle_metrics(abc, original_model, original_data, WRIST_LINK, VISUAL_WRIST_MESH)
    collision_points, collision_stats = mesh_to_bottle_metrics(abc, original_model, original_data, WRIST_LINK, COLLISION_WRIST_MESH)
    visual_source = ConvexHull(abc.stl_triangles(SOURCE_URDF.parent / "meshes" / VISUAL_WRIST_MESH).reshape(-1, 3)).equations
    collision_source_vertices = np.unique(abc.stl_triangles(SOURCE_URDF.parent / "meshes" / COLLISION_WRIST_MESH).reshape(-1, 3), axis=0)
    visual_source_vertices = np.unique(abc.stl_triangles(SOURCE_URDF.parent / "meshes" / VISUAL_WRIST_MESH).reshape(-1, 3), axis=0)
    visual_inequalities = visual_source[:, :3] @ collision_source_vertices.T + visual_source[:, 3, None]
    collision_hull = ConvexHull(collision_source_vertices)
    collision_inequalities = collision_hull.equations[:, :3] @ visual_source_vertices.T + collision_hull.equations[:, 3, None]
    mesh_comparison = {
        "visual": visual_stats,
        "collision": collision_stats,
        "collision_to_visual_hull_volume_ratio": collision_stats["convex_hull_volume_m3"] / visual_stats["convex_hull_volume_m3"],
        "collision_vertices_outside_visual_convex_hull_fraction": float(np.mean(np.any(visual_inequalities > 1e-8, axis=0))),
        "visual_vertices_outside_collision_convex_hull_fraction": float(np.mean(np.any(collision_inequalities > 1e-8, axis=0))),
        "candidate_pose": candidate,
        "interpretation_limit": "Different visual and collision mesh names and envelopes are directly established. Whether the extra collision envelope models an intentionally hidden wrist extension or is an over-conservative collision asset is not established by the URDF alone.",
    }
    abc.write_json(out / "wrist_visual_collision_comparison.json", mesh_comparison)
    bottle_id = abc.geom_id(original_model, "bottle_body")
    draw_mesh_comparison(out / "wrist_collision_vs_visual_projection.png",
                         original_data.geom_xpos[bottle_id].copy(), float(original_model.geom_size[bottle_id, 0]),
                         float(original_model.geom_size[bottle_id, 1]), visual_points, collision_points)

    # Render the actual pinned visual model. The collision counterfactual intentionally has identical visuals.
    abc.render(original_model, original_data, out / "source_frame_open_overview.png", (.30, 0, .88), .76, 135, -16)
    state_geometry(abc, base, original_model, original_data, float(candidate["previous_variables"]["aperture"]), "CONTACT_GEOMETRY")
    abc.render(original_model, original_data, out / "source_frame_contact_closeup.png", (.30, 0, .88), .40, 130, -10)

    with (out / "static_state_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "state", "aperture_ratio", "driver_target_rad", "follower_target_rad",
                         "minimum_robot_bottle_distance_m", "minimum_body", "minimum_robot_table_distance_m",
                         "active_bottle_contact_count", "active_distal_tip_bodies", "open_gate_pass"])
        for model_name, record in comparison.items():
            for state in record["states"]:
                bottle_min = state["minimum_robot_bottle_distance_m"] or {}
                table_min = state["minimum_robot_table_distance_m"] or {}
                writer.writerow([model_name, state["label"], state["aperture_ratio"], state["driver_target_rad"],
                                 state["follower_target_rad"], bottle_min.get("signed_distance_m"), bottle_min.get("body1"),
                                 table_min.get("signed_distance_m"), len(state["active_bottle_contacts"]),
                                 json.dumps(state["active_distal_tip_contact_bodies"]), state["open_collision_gate_pass"]])

    result = {
        "runtime_identity": identity,
        "source_identity": extracted,
        "source_contact_frame": frame,
        "derived_candidate": candidate,
        "static_comparison": comparison,
        "wrist_visual_collision_comparison": mesh_comparison,
        "conclusions": {
            "source_contact_frame_reproduced": True,
            "distal_opposing_surfaces_centered_by_compiled_distance": candidate["tcp_correction"]["final_pair_distances_m"],
            "source_faithful_open_preflight_pass": open_state["open_collision_gate_pass"],
            "physical_dynamics_run": False,
            "physical_dynamics_stop_reason": "The source-faithful OPEN state has active bottle collision(s); the visual-mesh collision substitution remains diagnostic-only and is not an approved physical model.",
            "primary_observed_failure": "Source-faithful static OPEN preflight fails from active wrist, loop, and proximal jaw collision; no dynamic grasp or lift was run.",
            "global_source_gripper_bottle_incompatibility_proven": False,
        },
    }
    abc.write_json(out / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
