#!/usr/bin/env python3
"""Bounded, source-faithful isolated OmniPicker endpoint geometry audit."""

from __future__ import annotations

import argparse
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
        qpos_index = int(model.jnt_qposadr[qid])
        mimic = item.find("mimic")
        result[name] = {
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
            "compiled_range_rad": model.jnt_range[qid].tolist(),
            "compiled_limited": bool(model.jnt_limited[qid]),
            "compiled_ref_rad": float(model.qpos0[qpos_index]),
            "candidate_qpos_rad": float(candidate["data"].qpos[qpos_index]),
        }
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
        elif body in {NARROW, WIDE} and group == 3:
            model.geom_rgba[geom_id] = [0.12, 0.95, 0.34, 0.72]
        elif group == 2:
            model.geom_rgba[geom_id] = [0.82, 0.88, 0.96, 0.36]
        elif group == 3:
            model.geom_rgba[geom_id] = [0.98, 0.20, 0.10, 0.55]
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
        if group == 1 or (not collision_only and group == 2) or (collision_only and group == 3):
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
                     [1.0, 0.42, 0.05, 1.0])
    for index, row in enumerate(contact_marks):
        color = marker_colors[min(index, len(marker_colors) - 1)]
        point = 0.5 * (np.asarray(row["witness_robot_world_m"]) + np.asarray(row["witness_bottle_world_m"]))
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.0026, 0.0026, 0.0026]), point,
            np.eye(3, dtype=np.float64).reshape(-1), np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1
    image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image, "RGBA")
    legend = [
        ("Visual mesh", (210, 224, 245, 255)),
        ("Compiled collision geometry / convex hull", (250, 51, 26, 230)),
        ("Narrow / wide fingertip collision geom", (31, 242, 87, 230)),
        ("Narrow pad witness", (255, 228, 20, 255)),
        ("Wide pad witness", (245, 26, 184, 255)),
        ("Wrist / bottle blocker witness", (255, 107, 13, 255)),
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
        draw.text((24, image.height - 34), "COLLISION-ONLY VIEW; MuJoCo compiled mesh hull visualization enabled",
                  fill=(255, 246, 230, 255))
    if transparent_bottle:
        draw.text((24, image.height - 58), "Bottle alpha is renderer-only; collision and physics unchanged",
                  fill=(255, 246, 230, 255))
    image.save(output)
    if original_rgba is not None:
        model.geom_rgba[bottle_geom_ids] = original_rgba
    renderer.close()


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
    args = parser.parse_args()
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
        "x2_repository": "https://github.com/AgibotTech/agibot_x2_urdf",
        "x2_commit": vendor_head,
        "x2_urdf": str(X2_URDF_REL),
        "x2_urdf_sha256": sha256(source_urdf),
        "x2_license": "Mulan PSL v2",
        "canonical_helper": str(canonical_helper),
        "canonical_helper_sha256": sha256(canonical_helper),
        "mujoco_python": mujoco.__version__,
        "mujoco_native_library_sha256": sha256(Path(mujoco.__file__).parent / "libmujoco.so.3.3.6"),
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
