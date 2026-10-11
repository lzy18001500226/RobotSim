#!/usr/bin/env python3
"""Map the saved OmniPicker contact patches and measure fixed-object preload.

The fixed cylinder exists only in this SIMULATION_ONLY diagnostic. The source
meshes, joint ranges, collision settings, contact properties, controller gains,
and actuator effort caps are inherited unchanged from the isolated P0 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import issue46_omnipicker_p0_isolated as p0  # noqa: E402


BASELINE_EVIDENCE = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "omnipicker-p0-measured-preload-20261011-run05-safe"
)
PAD_NAMES = p0.PAD_BODIES
DRIVER, FOLLOWER = p0.accepted.DRIVER_FOLLOWER["right"][:2]
PRELOAD_RAMP_SAFETY_S = 9.0
PRELOAD_SETTLE_S = 1.0
FORCE_STABLE_WINDOW_S = 0.20
TRACE_INTERVAL_STEPS = 1


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    p0.write_json(path, value)


def source_triangle_mesh(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"STL is truncated: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * triangle_count:
        raise ValueError(f"Expected binary STL for source mesh: {path}")
    triangles = np.empty((triangle_count, 3, 3), dtype=np.float64)
    offset = 84
    for index in range(triangle_count):
        row = struct.unpack_from("<12fH", payload, offset)
        triangles[index] = np.asarray(row[3:12], dtype=np.float64).reshape(3, 3)
        offset += 50
    return triangles


def closest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, int]:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    closest = np.empty_like(a)
    assigned = np.zeros(len(triangles), dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    closest[mask] = a[mask]
    assigned |= mask

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    closest[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    t = d1[mask] / (d1[mask] - d3[mask])
    closest[mask] = a[mask] + t[:, None] * ab[mask]
    assigned |= mask

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    closest[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    t = d2[mask] / (d2[mask] - d6[mask])
    closest[mask] = a[mask] + t[:, None] * ac[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    t = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
    closest[mask] = b[mask] + t[:, None] * (c[mask] - b[mask])
    assigned |= mask

    mask = ~assigned
    denominator = va[mask] + vb[mask] + vc[mask]
    v = vb[mask] / denominator
    w = vc[mask] / denominator
    closest[mask] = a[mask] + ab[mask] * v[:, None] + ac[mask] * w[:, None]

    distance2 = np.einsum("ij,ij->i", closest - point, closest - point)
    index = int(np.argmin(distance2))
    return closest[index], index


def nearest_source_surface(point: np.ndarray, triangles: np.ndarray,
                           body_rotation: np.ndarray, body_position: np.ndarray) -> dict[str, Any]:
    world_triangles = triangles @ body_rotation.T + body_position
    lower = world_triangles.min(axis=1)
    upper = world_triangles.max(axis=1)
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    candidate_ids = np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= 0.012**2)
    if not len(candidate_ids):
        raise RuntimeError("Source mesh broad-phase returned no candidate triangles")
    nearest, candidate_index = closest_points_on_triangles(point, world_triangles[candidate_ids])
    triangle_id = int(candidate_ids[candidate_index])
    tri = world_triangles[triangle_id]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return {
        "point_world_m": nearest.tolist(),
        "distance_from_collision_witness_mm": float(np.linalg.norm(nearest - point) * 1000.0),
        "source_triangle_index": triangle_id,
        "source_triangle_normal_world_unoriented": normal.tolist(),
        "candidate_triangles_within_12mm_aabb": int(len(candidate_ids)),
    }


def baseline_free_state(model: mujoco.MjModel, data: mujoco.MjData,
                        row: dict[str, Any]) -> None:
    mujoco.mj_resetData(model, data)
    for name in ("p0_fixture_approach_slide", "p0_fixture_lift_slide"):
        data.qpos[p0.accepted.qpos_addr(model, name)] = row["fixture_controller"][name]["q_m"]
        data.qvel[p0.accepted.dof_addr(model, name)] = row["fixture_controller"][name]["qvel_m_s"]
    for name in (DRIVER, FOLLOWER):
        data.qpos[p0.accepted.qpos_addr(model, name)] = row["hand_joint_state"][name]["qpos_rad"]
        data.qvel[p0.accepted.dof_addr(model, name)] = row["hand_joint_state"][name]["qvel_rad_s"]
    qadr = p0.accepted.qpos_addr(model, "p0_canonical_test_cylinder_free")
    data.qpos[qadr:qadr + 7] = (row["bottle"]["position_world_m"]
                                 + row["bottle"]["quaternion_wxyz"])
    data.time = float(row["time_s"])
    mujoco.mj_forward(model, data)


def reconstruct_baseline_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                                  row: dict[str, Any]) -> dict[str, Any]:
    source_root = ET.parse(p0.SOURCE_URDF).getroot()
    mesh_dir = p0.SOURCE_ROOT / "X2_URDF-v1.4.0" / "meshes"
    mesh_names = {
        "R_hand_narrow3_Link": "narrow3_Link.stl",
        "R_hand_wide3_Link": "wide3_Link.stl",
    }
    mesh_triangles = {name: source_triangle_mesh(mesh_dir / filename)
                      for name, filename in mesh_names.items()}
    result_contacts = []
    pad_force_vectors = []
    force_normal_directions = []
    for contact in row["contacts"]:
        pad = contact["body2"]
        if pad not in mesh_names:
            continue
        link = next(link for link in source_root.findall("link") if link.get("name") == pad)
        collision = link.find("collision")
        origin = collision.find("origin") if collision is not None else None
        if origin is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            if np.linalg.norm(xyz) > 1e-12 or np.linalg.norm(rpy) > 1e-12:
                raise RuntimeError(f"Non-zero source collision origin needs explicit mesh-frame handling: {pad}")
        mesh_tag = collision.find("geometry/mesh") if collision is not None else None
        if mesh_tag is None or "scale" in mesh_tag.attrib:
            raise RuntimeError(f"Unexpected source collision mesh mapping for {pad}")

        midpoint = np.asarray(contact["contact_position_world_m"], dtype=float)
        normal_geom0_to_geom1 = np.asarray(contact["normal_world"], dtype=float)
        signed_distance = float(contact["signed_distance_m"])
        cylinder_witness = midpoint - 0.5 * signed_distance * normal_geom0_to_geom1
        finger_witness = midpoint + 0.5 * signed_distance * normal_geom0_to_geom1
        body_id = p0.accepted.body_id(model, pad)
        body_rotation = data.xmat[body_id].reshape(3, 3)
        body_position = data.xpos[body_id].copy()
        nearest_finger = nearest_source_surface(
            finger_witness, mesh_triangles[pad], body_rotation, body_position
        )

        cylinder_id = p0.accepted.body_id(model, "p0_canonical_test_cylinder")
        cylinder_rotation = data.xmat[cylinder_id].reshape(3, 3)
        cylinder_position = data.xpos[cylinder_id].copy()
        cylinder_local = cylinder_rotation.T @ (cylinder_witness - cylinder_position)
        radial = cylinder_local[:2]
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1e-12:
            raise RuntimeError("Cylinder contact witness lies on its local axis")
        cylinder_surface_local = np.array([
            radial[0] * p0.accepted.CYLINDER_RADIUS / radial_norm,
            radial[1] * p0.accepted.CYLINDER_RADIUS / radial_norm,
            float(np.clip(cylinder_local[2], -p0.accepted.CYLINDER_HALF_HEIGHT,
                          p0.accepted.CYLINDER_HALF_HEIGHT)),
        ])
        cylinder_surface_world = cylinder_position + cylinder_rotation @ cylinder_surface_local
        force_on_cylinder = np.asarray(contact["force_on_bottle_world_N"], dtype=float)
        normal_on_cylinder = -normal_geom0_to_geom1
        pad_force_vectors.append(force_on_cylinder)
        force_normal_directions.append(normal_on_cylinder)
        result_contacts.append({
            "geom0": contact["geom1"], "geom1": contact["geom2"],
            "cylinder_body": contact["body1"], "finger_body": pad,
            "contact_frame_normal_geom0_to_geom1": normal_geom0_to_geom1.tolist(),
            "normal_force_direction_on_cylinder": normal_on_cylinder.tolist(),
            "contact_position_midpoint_world_m": midpoint.tolist(),
            "signed_distance_m": signed_distance,
            "estimated_cylinder_surface_witness_world_m": cylinder_witness.tolist(),
            "projected_cylinder_sidewall_surface_world_m": cylinder_surface_world.tolist(),
            "projected_cylinder_radius_error_mm": float(abs(radial_norm - p0.accepted.CYLINDER_RADIUS) * 1000.0),
            "estimated_finger_surface_witness_world_m": finger_witness.tolist(),
            "nearest_original_finger_stl_surface": nearest_finger,
            "force_on_cylinder_world_N": force_on_cylinder.tolist(),
            "normal_force_N": float(contact["normal_force_N"]),
            "effective_friction": contact["effective_contact_friction"],
        })

    normals = np.asarray(force_normal_directions)
    angle = math.degrees(math.acos(float(np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0))))
    resultant = np.sum(pad_force_vectors, axis=0)
    return {
        "baseline_trace": str(BASELINE_EVIDENCE / "C_PRELOAD_TRACE.jsonl"),
        "baseline_step": row["step"], "baseline_time_s": row["time_s"],
        "bottle_pose_world": row["bottle"],
        "source_collision_meshes": {
            pad: {"path": str(mesh_dir / filename), "sha256": p0.sha256(mesh_dir / filename),
                  "triangle_count": int(len(mesh_triangles[pad]))}
            for pad, filename in mesh_names.items()
        },
        "contacts": result_contacts,
        "force_normal_angle_deg": angle,
        "resultant_pad_force_on_cylinder_world_N": resultant.tolist(),
        "resultant_pad_force_lateral_xy_N": float(np.linalg.norm(resultant[:2])),
        "resultant_pad_force_vertical_N": float(resultant[2]),
        "measured_effective_mu": min(float(row["contacts"][0]["effective_contact_friction"][0]),
                                      float(row["contacts"][1]["effective_contact_friction"][0])),
        "interpretation": "Forces are transformed to world and expressed on the cylinder. The 124-degree separation persists after the sign convention is applied.",
    }


def draw_contact_top_view(path: Path, geometry: dict[str, Any]) -> None:
    size = (1040, 760)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    center = np.asarray(geometry["bottle_pose_world"]["position_world_m"][:2], dtype=float)
    scale = 4300.0
    origin = (520, 390)

    def px(point: np.ndarray) -> tuple[int, int]:
        return (int(origin[0] + (point[0] - center[0]) * scale),
                int(origin[1] - (point[1] - center[1]) * scale))

    radius = p0.accepted.CYLINDER_RADIUS * scale
    draw.ellipse((origin[0] - radius, origin[1] - radius,
                  origin[0] + radius, origin[1] + radius),
                 fill=(211, 237, 250), outline=(24, 91, 132), width=4)
    draw.line((origin[0] - 260, origin[1], origin[0] + 260, origin[1]), fill=(210, 210, 210), width=1)
    draw.line((origin[0], origin[1] - 260, origin[0], origin[1] + 260), fill=(210, 210, 210), width=1)
    draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(30, 30, 30))
    colors = {"R_hand_narrow3_Link": (195, 62, 57), "R_hand_wide3_Link": (49, 115, 184)}
    for contact in geometry["contacts"]:
        color = colors[contact["finger_body"]]
        point = np.asarray(contact["projected_cylinder_sidewall_surface_world_m"][:2])
        location = px(point)
        draw.ellipse((location[0] - 8, location[1] - 8, location[0] + 8, location[1] + 8), fill=color)
        direction = np.asarray(contact["normal_force_direction_on_cylinder"][:2])
        endpoint = px(point + direction * 0.023)
        draw.line((location, endpoint), fill=color, width=5)
        vx, vy = endpoint[0] - location[0], endpoint[1] - location[1]
        norm = max(math.hypot(vx, vy), 1.0)
        for sign in (-1, 1):
            hx = endpoint[0] - 13 * vx / norm + sign * 7 * (-vy) / norm
            hy = endpoint[1] - 13 * vy / norm + sign * 7 * vx / norm
            draw.line((endpoint, (int(hx), int(hy))), fill=color, width=4)
        draw.text((location[0] + 12, location[1] - 18), contact["finger_body"].replace("R_hand_", ""), fill=color)
    draw.text((32, 28), "Saved free-cylinder contact geometry at the 5 mm slip stop", fill=(20, 20, 20))
    draw.text((32, 58), "Circle: canonical 70 mm cylinder sidewall; arrows: force direction on cylinder (not to scale)", fill=(50, 50, 50))
    draw.text((32, 88), f"Pad normal-force angle: {geometry['force_normal_angle_deg']:.2f} deg", fill=(50, 50, 50))
    draw.text((32, 118), f"Pad-force resultant: {geometry['resultant_pad_force_on_cylinder_world_N']}", fill=(50, 50, 50))
    draw.text((32, 148), "World XY projection; Z contact height is recorded in FREE_CONTACT_GEOMETRY.json", fill=(50, 50, 50))
    image.save(path)


def set_initial_fixed_body(spec: mujoco.MjSpec, row: dict[str, Any],
                           translation_world_m: np.ndarray) -> dict[str, Any]:
    body = spec.body("p0_canonical_test_cylinder")
    source_pos = np.asarray(body.pos, dtype=float).copy()
    source_quat = np.asarray(body.quat, dtype=float).copy()
    initial_position = np.asarray(row["bottle"]["position_world_m"], dtype=float)
    body.pos = (initial_position + translation_world_m).tolist()
    body.quat = row["bottle"]["quaternion_wxyz"]
    free_joint = next((joint for joint in spec.joints
                       if joint.name == "p0_canonical_test_cylinder_free"), None)
    if free_joint is None:
        raise RuntimeError("Expected canonical cylinder free joint is missing before fixed-object conversion")
    removed_joint_name = free_joint.name
    spec.delete(free_joint)
    spec.modelname = "SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC"
    return {"source_body_pos_before_m": source_pos.tolist(),
            "source_body_quat_before_wxyz": source_quat.tolist(),
            "baseline_free_run_position_world_m": initial_position.tolist(),
            "candidate_translation_world_m": translation_world_m.tolist(),
            "fixed_body_pos_world_m": body.pos.tolist(),
            "fixed_body_quat_wxyz": body.quat.tolist(),
            "removed_joint": removed_joint_name,
            "constraint_method": "static body with no joint; no equality, weld, mocap, attachment, or runtime object state"}


def candidate_static_preflight(model: mujoco.MjModel, bottle_row: dict[str, Any],
                               translation_world_m: np.ndarray,
                               output: Path) -> dict[str, Any]:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    bottle_qadr = p0.accepted.qpos_addr(model, "p0_canonical_test_cylinder_free")
    bottle_position = (np.asarray(bottle_row["bottle"]["position_world_m"], dtype=float)
                       + translation_world_m)
    bottle_quaternion = np.asarray(bottle_row["bottle"]["quaternion_wxyz"], dtype=float)
    data.qpos[bottle_qadr:bottle_qadr + 7] = np.concatenate((bottle_position, bottle_quaternion))
    cylinder_gid = p0.accepted.geom_id(model, "p0_canonical_test_cylinder_geom")
    cylinder_body = "p0_canonical_test_cylinder"
    pad_body_ids = {p0.accepted.body_id(model, name) for name in PAD_NAMES}
    hand_body_ids = p0.source_hand_body_ids(model)
    hand_body_names = {p0.accepted.body_name(model, body_id) for body_id in hand_body_ids}
    hand_geom_ids = [gid for gid in p0.accepted.collision_geoms(model)
                     if int(model.geom_bodyid[gid]) in hand_body_ids]
    table_ids = {gid for gid in p0.accepted.collision_geoms(model)
                 if p0.accepted.body_name(model, int(model.geom_bodyid[gid])) == "m0_table"}
    vertex_cache: dict[int, np.ndarray] = {}
    clearance = []

    def sample(phase: str, coordinate: float, aperture: float) -> dict[str, Any]:
        p0.set_fixture_state(model, data, coordinate, 0.0, aperture)
        data.qpos[bottle_qadr:bottle_qadr + 7] = np.concatenate((bottle_position, bottle_quaternion))
        mujoco.mj_forward(model, data)
        distances, min_all, min_nonpad = p0.all_geom_distances(
            model, data, cylinder_gid, pad_body_ids
        )
        pad_distances = {
            name: min((item["signed_distance_m"] for item in distances if item["body"] == name),
                      default=math.inf)
            for name in PAD_NAMES
        }
        contacts = p0.accepted.active_contact_rows(model, data)
        cylinder_contacts = [row for row in contacts
                             if cylinder_body in (row["body1"], row["body2"])]
        other_bodies = [body for row in cylinder_contacts
                        for body in (row["body1"], row["body2"]) if body != cylinder_body]
        hand_contacts = [row for row in cylinder_contacts
                         if any(body in hand_body_names for body in (row["body1"], row["body2"]))]
        table_support_contacts = [row for row in cylinder_contacts
                                  if "m0_table" in (row["body1"], row["body2"])]
        pad_contacts = {body for row in cylinder_contacts
                         for body in (row["body1"], row["body2"]) if body in PAD_NAMES}
        nonpad_contacts = [row for row in hand_contacts
                           if any(body not in PAD_NAMES and body != cylinder_body
                                  for body in (row["body1"], row["body2"]))]
        table_clearance = p0.compiled_mesh_table_clearance(
            model, data, hand_geom_ids, table_ids, vertex_cache
        )
        row = {
            "phase": phase, "coordinate": coordinate, "aperture": aperture,
            "minimum_all_bottle_distance_m": min_all,
            "minimum_nonpad_bottle_distance_m": min_nonpad,
            "pad_distances_m": pad_distances,
            "active_cylinder_contacts": cylinder_contacts,
            "active_cylinder_hand_contacts": hand_contacts,
            "expected_table_support_contacts": table_support_contacts,
            "other_cylinder_contact_bodies": sorted(set(other_bodies)),
            "active_pad_contact_bodies": sorted(pad_contacts),
            "nonpad_bottle_contacts": nonpad_contacts,
            "minimum_gripper_table_clearance_lower_bound_m": table_clearance["lower_bound_m"],
            "table_clearance_detail": table_clearance,
        }
        clearance.append(row)
        return row

    open_rows = [sample("OPEN_APPROACH", float(coordinate), 1.0)
                 for coordinate in np.linspace(p0.APPROACH_START, p0.APPROACH_END, 25)]
    approach_min_nonpad = min(row["minimum_nonpad_bottle_distance_m"] for row in open_rows)
    approach_min_table = min(row["minimum_gripper_table_clearance_lower_bound_m"] for row in open_rows)
    open_contacts = [row for row in open_rows if row["active_cylinder_hand_contacts"]]
    open_clear = (not open_contacts
                  and min(row["minimum_all_bottle_distance_m"] for row in open_rows)
                  > p0.STATIC_DISTANCE_TOLERANCE_M
                  and approach_min_nonpad > p0.STATIC_DISTANCE_TOLERANCE_M
                  and approach_min_table > p0.STATIC_DISTANCE_TOLERANCE_M)

    closure_rows = []
    bilateral_target = None
    nonpad_closure_clear = True
    table_closure_clear = True
    for aperture in np.linspace(1.0, 0.0, 1401):
        row = sample("STATIC_CLOSE", p0.APPROACH_END, float(aperture))
        closure_rows.append(row)
        nonpad_closure_clear &= (
            row["minimum_nonpad_bottle_distance_m"] > p0.STATIC_DISTANCE_TOLERANCE_M
            and not row["nonpad_bottle_contacts"]
        )
        table_closure_clear &= (
            row["minimum_gripper_table_clearance_lower_bound_m"]
            > p0.STATIC_DISTANCE_TOLERANCE_M
        )
        if bilateral_target is None and all(
                distance <= -0.0005 for distance in row["pad_distances_m"].values()
        ) and set(PAD_NAMES).issubset(row["active_pad_contact_bodies"]):
            bilateral_target = float(aperture)

    footprint_inside_table = bool(
        np.all(np.abs(bottle_position[:2] - p0.TABLE_CENTER[:2])
               + p0.accepted.CYLINDER_RADIUS <= p0.TABLE_HALF_EXTENTS)
    )
    passed = (open_clear and nonpad_closure_clear and table_closure_clear
              and bilateral_target is not None and footprint_inside_table)
    summary = {
        "classification": "zero-step source-collision-geometry preflight for one geometry-derived bottle translation",
        "bottle_position_world_m": bottle_position.tolist(),
        "bottle_translation_world_m": translation_world_m.tolist(),
        "table_footprint_inside": footprint_inside_table,
        "open_approach_clear": open_clear,
        "open_approach_minimum_nonpad_distance_m": approach_min_nonpad,
        "open_approach_minimum_table_clearance_m": approach_min_table,
        "open_approach_cylinder_contact_count": len(open_contacts),
        "nonpad_clear_through_closure": bool(nonpad_closure_clear),
        "table_clear_through_closure": bool(table_closure_clear),
        "bilateral_contact_target_aperture": bilateral_target,
        "closure_samples": len(closure_rows),
        "physics_steps": 0,
        "passed": bool(passed),
        "trace": "CANDIDATE_STATIC_PREFLIGHT_TRACE.jsonl",
    }
    write_json(output / "CANDIDATE_STATIC_PREFLIGHT.json", summary)
    p0.write_jsonl(output / "CANDIDATE_STATIC_PREFLIGHT_TRACE.jsonl", clearance)
    renderer = mujoco.Renderer(model, height=480, width=640)
    sample("RENDER_OPEN", p0.APPROACH_START, 1.0)
    Image.fromarray(p0.render(renderer, model, data, "overview")).save(
        output / "CANDIDATE_OPEN_OVERVIEW.png"
    )
    if bilateral_target is not None:
        sample("RENDER_BILATERAL", p0.APPROACH_END, bilateral_target)
        Image.fromarray(p0.render(renderer, model, data, "close")).save(
            output / "CANDIDATE_BILATERAL_CLOSEUP.png"
        )
    renderer.close()
    return summary


def bounded_trapezoid_reference(start_q: float, end_q: float, elapsed_s: float,
                                rate_limit: float, acceleration_limit: float) -> tuple[float, float, float]:
    distance = abs(end_q - start_q)
    if distance <= 1e-15:
        return end_q, 0.0, 0.0
    direction = math.copysign(1.0, end_q - start_q)
    peak_speed = min(rate_limit, math.sqrt(distance * acceleration_limit))
    accel_time = peak_speed / acceleration_limit
    accel_distance = 0.5 * acceleration_limit * accel_time**2
    cruise_time = max(0.0, (distance - 2.0 * accel_distance) / peak_speed)
    total_time = 2.0 * accel_time + cruise_time

    t = min(max(elapsed_s, 0.0), total_time)
    if t < accel_time:
        travelled = 0.5 * acceleration_limit * t**2
        speed = acceleration_limit * t
        acceleration = acceleration_limit
    elif t < accel_time + cruise_time:
        cruise_elapsed = t - accel_time
        travelled = accel_distance + peak_speed * cruise_elapsed
        speed = peak_speed
        acceleration = 0.0
    elif t < total_time:
        decel_elapsed = t - accel_time - cruise_time
        travelled = (accel_distance + peak_speed * cruise_time
                     + peak_speed * decel_elapsed
                     - 0.5 * acceleration_limit * decel_elapsed**2)
        speed = peak_speed - acceleration_limit * decel_elapsed
        acceleration = -acceleration_limit
    else:
        return end_q, 0.0, 0.0
    return (start_q + direction * travelled,
            direction * speed,
            direction * acceleration)


def all_relevant_robot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    hand_ids = p0.source_hand_body_ids(model)
    rows = p0.accepted.active_contact_rows(model, data)
    result = []
    for row in rows:
        body_ids = [p0.accepted.body_id(model, name) for name in (row["body1"], row["body2"])
                    if name != "world"]
        if any(body_id in hand_ids for body_id in body_ids):
            result.append(row)
    return result


def render_frame(renderer: mujoco.Renderer, model: mujoco.MjModel,
                 data: mujoco.MjData, mode: str) -> np.ndarray:
    return p0.render(renderer, model, data, mode)


def run_fixed_diagnostic(output: Path, candidate_translation_world_m: np.ndarray) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence directory: {output}")
    output.mkdir(parents=True)
    if not BASELINE_EVIDENCE.is_dir():
        raise FileNotFoundError(f"Required baseline evidence is missing: {BASELINE_EVIDENCE}")

    baseline_result = read_json(BASELINE_EVIDENCE / "C_RESULT.json")
    baseline_runtime = read_json(BASELINE_EVIDENCE / "runtime_identity.json")
    baseline_model = read_json(BASELINE_EVIDENCE / "fixture_model_inventory.json")
    baseline_geometry = read_json(BASELINE_EVIDENCE / "B2_STATIC_GEOMETRY_GATE.json")
    trace_path = BASELINE_EVIDENCE / "C_PRELOAD_TRACE.jsonl"
    trace_hash = p0.sha256(trace_path)
    rows = [json.loads(line) for line in trace_path.open(encoding="utf-8")]
    event_row = next(row for row in rows if row.get("step") == 4288)
    preload_start_row = next(row for row in rows if row.get("stage") == "PRELOAD")

    with tempfile.TemporaryDirectory(prefix="issue46-fixed-object-model-") as tmp:
        temp_root = Path(tmp)
        subtree_path = temp_root / "generated_right_wrist_omnipicker_subtree.urdf"
        subtree = p0.write_wrist_subtree_urdf(subtree_path)
        canonical = p0.accepted.load_canonical_module(p0.CANONICAL_HELPER)
        fixture_pose = p0.source_pose()
        spec, free_model, build = p0.build_model(subtree_path, fixture_pose, canonical)
        if (free_model.nq != baseline_model["nq"] or free_model.nv != baseline_model["nv"] or
                subtree["generated_urdf_sha256"] != baseline_runtime["generated_subtree_urdf_sha256"]):
            raise RuntimeError("Baseline model identity does not match the saved free-cylinder experiment")
        free_controller = p0.right_controller(free_model)
        current_runtime = p0.model_runtime_identity(
            free_model, Path(p0.__file__).resolve(), subtree_path
        )
        for key in ("mujoco_native", "source_commit", "source_urdf_sha256", "native_library_sha256"):
            if current_runtime.get(key) != baseline_runtime.get(key):
                raise RuntimeError(f"Baseline identity mismatch for {key}")

        free_data = mujoco.MjData(free_model)
        baseline_free_state(free_model, free_data, event_row)
        free_contact_geometry = reconstruct_baseline_contacts(free_model, free_data, event_row)
        write_json(output / "FREE_CONTACT_GEOMETRY.json", free_contact_geometry)
        draw_contact_top_view(output / "FREE_CONTACT_TOP_VIEW.png", free_contact_geometry)

        preflight = candidate_static_preflight(
            free_model, preload_start_row, candidate_translation_world_m, output
        )
        if not preflight["passed"]:
            raise RuntimeError("Geometry-derived candidate failed zero-step static preflight; no dynamics were run")
        fixed_body_record = set_initial_fixed_body(spec, preload_start_row, candidate_translation_world_m)
        fixed_model = spec.compile()
        if fixed_model.nq != free_model.nq - 7 or fixed_model.nv != free_model.nv - 6:
            raise RuntimeError("Fixed-object model did not remove exactly the cylinder free-joint coordinates")
        if fixed_model.neq != free_model.neq or fixed_model.neq != 0:
            raise RuntimeError("Unexpected equality/weld constraints in the fixed-object diagnostic")
        if p0.accepted.joint_names(fixed_model) != [
                "p0_fixture_approach_slide", "p0_fixture_lift_slide", DRIVER, FOLLOWER]:
            raise RuntimeError("Unexpected fixed-object diagnostic joint inventory")

        controller = p0.right_controller(fixed_model)
        baseline_controller = baseline_result["controller"]["per_side"]["right"]["joints"]
        controller_comparison = {}
        for name in (DRIVER, FOLLOWER):
            free_row = free_controller["per_side"]["right"]["joints"][name]
            baseline_row = baseline_controller[name]
            fixed_row = controller["per_side"]["right"]["joints"][name]
            compared = ("kp_nm_rad", "kd_nm_s_rad", "effort_cap_nm")
            for field in compared:
                if not math.isclose(free_row[field], baseline_row[field], rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"Reconstructed free-model controller differs from saved baseline: {name} {field}")
                if not math.isclose(fixed_row[field], baseline_row[field], rel_tol=0.0, abs_tol=1e-12):
                    raise RuntimeError(f"Fixed-object controller differs from saved baseline: {name} {field}")
            controller_comparison[name] = {
                field: {"saved_baseline": baseline_row[field], "free_rebuild": free_row[field],
                        "fixed_diagnostic": fixed_row[field]}
                for field in compared
            }
        data = mujoco.MjData(fixed_model)
        mujoco.mj_resetData(fixed_model, data)
        p0.set_fixture_state(fixed_model, data, p0.APPROACH_START, 0.0, 1.0)
        mujoco.mj_forward(fixed_model, data)
        slides = p0.slide_gains(fixed_model, data)
        source_joint_elements = p0._source_joint_elements(p0.SOURCE_URDF)
        # Preserve the exact source subtree input before recording its stable path/hash.
        (output / subtree_path.name).write_bytes(subtree_path.read_bytes())
        base_runtime = p0.model_runtime_identity(
            fixed_model, Path(__file__).resolve(), output / subtree_path.name
        )
        spec.to_file(str(output / "SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC.xml"))

        driver_profile = {
            "controller": controller,
            "saved_free_run_profile_exact_match": True,
            "profile_comparison": controller_comparison,
            "hand_actuators": {
                name: {
                    "actuator_name": f"diag_right_{role}_servo",
                    "actuator_id": p0.accepted.actuator_id(fixed_model, f"diag_right_{role}_servo"),
                    "ctrlrange": fixed_model.actuator_ctrlrange[
                        p0.accepted.actuator_id(fixed_model, f"diag_right_{role}_servo")
                    ].tolist(),
                    "forcerange": fixed_model.actuator_forcerange[
                        p0.accepted.actuator_id(fixed_model, f"diag_right_{role}_servo")
                    ].tolist(),
                    "gear": fixed_model.actuator_gear[
                        p0.accepted.actuator_id(fixed_model, f"diag_right_{role}_servo")
                    ].tolist(),
                }
                for role, name in (("driver", DRIVER), ("follower", FOLLOWER))
            },
            "operational_open_target_rad": p0.operational_aperture_reference(1.0, 0.0, 0.0),
            "baseline_bilateral_contact_aperture": baseline_geometry["closure"]["bilateral_contact_target_aperture"],
            "candidate_static_contact_aperture": preflight["bilateral_contact_target_aperture"],
            "source_ranges_rad": {
                name: fixed_model.jnt_range[p0.accepted.joint_id(fixed_model, name)].tolist()
                for name in (DRIVER, FOLLOWER)
            },
            "contact_parameters": {
                "cylinder_geom_friction": fixed_model.geom_friction[
                    p0.accepted.geom_id(fixed_model, "p0_canonical_test_cylinder_geom")
                ].tolist(),
                "cylinder_geom_condim": int(fixed_model.geom_condim[
                    p0.accepted.geom_id(fixed_model, "p0_canonical_test_cylinder_geom")
                ]),
                "timestep_s": float(fixed_model.opt.timestep),
                "gravity_m_s2": fixed_model.opt.gravity.tolist(),
            },
        }
        write_json(output / "CONTROLLER_AND_MODEL_IDENTITY.json", {
            "classification": "SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC",
            "baseline_source": str(BASELINE_EVIDENCE),
            "baseline_commit": baseline_runtime.get("robotsim_head"),
            "baseline_runner_sha256": baseline_runtime.get("runner_sha256"),
            "baseline_trace_sha256": trace_hash,
            "baseline_source_runtime_identity": baseline_runtime,
            "fixed_model_runtime_identity": base_runtime,
            "baseline_free_model_inventory": baseline_model,
            "fixed_model_inventory": {
                "nq": fixed_model.nq, "nv": fixed_model.nv, "nu": fixed_model.nu,
                "njnt": fixed_model.njnt, "nbody": fixed_model.nbody,
                "ngeom": fixed_model.ngeom, "nmesh": fixed_model.nmesh,
                "neq": fixed_model.neq,
                "joint_names": p0.accepted.joint_names(fixed_model),
                "source_joint_ranges_rad": driver_profile["source_ranges_rad"],
                "cylinder_has_free_joint": False,
                "canonical_free_control_cylinder_body_mass_kg": float(free_model.body_mass[
                    p0.accepted.body_id(free_model, "p0_canonical_test_cylinder")
                ]),
                "fixed_diagnostic_cylinder_dynamic_mass_kg": 0.0,
                "fixed_diagnostic_mass_note": "The same 0.570 kg canonical geom is held by a static body; its dynamic mass is intentionally absent only in this diagnostic.",
            },
            "fixed_body": fixed_body_record,
            "candidate_static_preflight": preflight,
            "driver_profile": driver_profile,
            "model_changes_from_free_control": [
                "Renamed model to SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC.",
                "Placed the static cylinder at the saved free-run PRELOAD-start pose before compilation.",
                "Removed only p0_canonical_test_cylinder_free; no equality or runtime attachment was added.",
            ],
            "prohibited_parameter_changes": [],
        })

        bilateral_aperture = float(preflight["bilateral_contact_target_aperture"])
        hold_ref = p0.operational_aperture_reference(bilateral_aperture, 0.0, 0.0)
        open_ref = p0.operational_aperture_reference(1.0, 0.0, 0.0)
        closed_ref = p0.operational_aperture_reference(0.0, 0.0, 0.0)
        close_delta = abs(hold_ref["driver_q"] - open_ref["driver_q"])
        close_peak_speed = 1.875 * close_delta / p0.accepted.MOTION_DURATION
        close_peak_acceleration = 5.773502691896258 * close_delta / p0.accepted.MOTION_DURATION**2
        ramp_rate_limit = p0.PRELOAD_REFERENCE_RATE_FRACTION * close_peak_speed
        ramp_acceleration_limit = p0.PRELOAD_REFERENCE_RATE_FRACTION * close_peak_acceleration
        ramp_distance = abs(closed_ref["driver_q"] - hold_ref["driver_q"])
        ramp_peak_speed = min(ramp_rate_limit, math.sqrt(ramp_distance * ramp_acceleration_limit))
        ramp_acceleration_time = ramp_peak_speed / ramp_acceleration_limit
        ramp_acceleration_distance = 0.5 * ramp_acceleration_limit * ramp_acceleration_time**2
        ramp_cruise_time = max(0.0, (ramp_distance - 2.0 * ramp_acceleration_distance)
                               / ramp_peak_speed)
        ramp_duration_budget = 2.0 * ramp_acceleration_time + ramp_cruise_time
        if ramp_duration_budget > PRELOAD_RAMP_SAFETY_S:
            raise RuntimeError(
                f"Bounded preload trajectory requires {ramp_duration_budget:.3f}s, "
                f"exceeding its {PRELOAD_RAMP_SAFETY_S:.3f}s safety budget"
            )
        total_preload_s = ramp_duration_budget + PRELOAD_SETTLE_S
        stages = [
            ("SETTLE_OPEN", 0.5, {"approach": (p0.APPROACH_START, p0.APPROACH_START),
                                  "aperture": (1.0, 1.0)}),
            ("APPROACH", p0.APPROACH_DURATION_S, {"approach": (p0.APPROACH_START, p0.APPROACH_END),
                                                   "aperture": (1.0, 1.0)}),
            ("CLOSE", p0.accepted.MOTION_DURATION, {"approach": (0.0, 0.0),
                                                    "aperture": (1.0, bilateral_aperture)}),
            ("HOLD", 1.0, {"approach": (0.0, 0.0),
                           "aperture": (bilateral_aperture, bilateral_aperture)}),
            ("PRELOAD_TO_OPERATIONAL_CLOSED", total_preload_s,
             {"approach": (0.0, 0.0), "aperture": (bilateral_aperture, 0.0)}),
        ]
        renderer = mujoco.Renderer(fixed_model, height=480, width=640)
        mp4_path = output / "FIXED_OBJECT_PRELOAD.mp4"
        trace_path = output / "FIXED_OBJECT_TRACE.jsonl"
        stage_summary: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        t_global = 0.0
        step = 0
        max_penetration = 0.0
        max_abs_hand_speed = 0.0
        source_limit_violations: list[dict[str, Any]] = []
        unexpected_contacts: set[str] = set()
        failure: str | None = None
        max_requested = {DRIVER: 0.0, FOLLOWER: 0.0}
        max_applied = {DRIVER: 0.0, FOLLOWER: 0.0}
        saturation_steps = {DRIVER: 0, FOLLOWER: 0}
        preload_force_rows: list[dict[str, float]] = []
        preload_bilateral_flags: list[bool] = []
        preload_penetrations: list[float] = []
        preload_closed_target_q = float(closed_ref["driver_q"])
        preload_target_q = float(hold_ref["driver_q"])
        preload_rate = 0.0
        required_stable_steps = round(FORCE_STABLE_WINDOW_S / p0.accepted.TIMESTEP)
        saturation_window_steps = round(p0.PRELOAD_SATURATION_WINDOW_S / p0.accepted.TIMESTEP)
        force_history_start: dict[str, float] | None = None
        saturation_plateau_steps = 0
        preload_start_global_step = None
        peak_stable_force: dict[str, float] = {name: 0.0 for name in PAD_NAMES}
        peak_stable_window_start_step: int | None = None
        peak_instantaneous_force: dict[str, float] = {name: 0.0 for name in PAD_NAMES}
        peak_force_event: dict[str, Any] | None = None
        fps_stride = 20
        first_contact_saved = False
        with imageio.get_writer(mp4_path, fps=25, codec="libx264", quality=7) as video:
            for stage, duration, targets in stages:
                local_steps = round(duration / p0.accepted.TIMESTEP)
                stage_bilateral = 0
                stage_force_samples = {name: [] for name in PAD_NAMES}
                stage_max_penetration = 0.0
                stage_start_time = t_global
                stage_rows = 0
                for local_step in range(local_steps):
                    local_t = local_step * p0.accepted.TIMESTEP
                    values: dict[str, tuple[float, float, float]] = {}
                    for key, (start, end) in targets.items():
                        if abs(end - start) > 1e-12:
                            values[key] = p0.accepted.minimum_jerk(local_t, duration, start, end)
                        else:
                            values[key] = (start, 0.0, 0.0)

                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED":
                        if preload_start_global_step is None:
                            preload_start_global_step = step
                        preload_target_q, preload_rate, target_acceleration = bounded_trapezoid_reference(
                            float(hold_ref["driver_q"]), preload_closed_target_q,
                            local_t, ramp_rate_limit, ramp_acceleration_limit,
                        )
                        span = 1.0 - p0.OPEN_OPERATIONAL_MARGIN_RAD - p0.CLOSED_OPERATIONAL_MARGIN_RAD
                        values["aperture"] = (
                            (-preload_target_q - p0.CLOSED_OPERATIONAL_MARGIN_RAD) / span,
                            -preload_rate / span,
                            -target_acceleration / span,
                        )
                        if local_t >= ramp_duration_budget:
                            preload_target_q = preload_closed_target_q
                            preload_rate = 0.0
                            target_acceleration = 0.0
                            values["aperture"] = (0.0, 0.0, 0.0)
                        if abs(preload_rate) > ramp_rate_limit + 1e-12:
                            raise RuntimeError("Preload reference exceeded its declared velocity bound")
                        if abs(target_acceleration) > ramp_acceleration_limit + 1e-12:
                            raise RuntimeError("Preload reference exceeded its declared acceleration bound")

                    mujoco.mj_forward(fixed_model, data)
                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED":
                        reference = p0.driver_coordinate_reference(
                            preload_target_q, preload_rate, target_acceleration
                        )
                    else:
                        reference = p0.operational_aperture_reference(*values["aperture"])
                    hand_controls = p0.accepted.controller_step(
                        fixed_model, data, controller, {"right": reference}
                    )
                    slide_controls = {}
                    for joint_name, actuator_name, key in (
                        ("p0_fixture_approach_slide", "p0_fixture_approach_motor", "approach"),
                        ("p0_fixture_lift_slide", "p0_fixture_lift_motor", None),
                    ):
                        qadr = p0.accepted.qpos_addr(fixed_model, joint_name)
                        dof = p0.accepted.dof_addr(fixed_model, joint_name)
                        q, velocity = float(data.qpos[qadr]), float(data.qvel[dof])
                        qref, vref, aref = values[key] if key else (0.0, 0.0, 0.0)
                        gains = slides[joint_name]
                        pd = gains["kp_N_m"] * (qref - q) + gains["kd_N_s_m"] * (vref - velocity)
                        bias = float(data.qfrc_bias[dof])
                        acceleration_force = float(p0.accepted.full_mass_matrix(fixed_model, data)[dof, dof]) * aref
                        requested = bias + pd + acceleration_force
                        applied = float(np.clip(requested, -gains["force_cap_N"], gains["force_cap_N"]))
                        data.ctrl[p0.accepted.actuator_id(fixed_model, actuator_name)] = applied
                        slide_controls[joint_name] = {
                            "q": q, "qvel": velocity, "target_q": qref, "target_qvel": vref,
                            "pd_force_N": pd, "bias_feedforward_N": bias,
                            "reference_acceleration_force_N": acceleration_force,
                            "requested_force_N": requested, "applied_force_N": applied,
                            "force_cap_N": gains["force_cap_N"],
                        }

                    mujoco.mj_forward(fixed_model, data)
                    for name, control in hand_controls.items():
                        aid = p0.accepted.actuator_id(
                            fixed_model, f"diag_right_{'driver' if name == DRIVER else 'follower'}_servo"
                        )
                        dof = p0.accepted.dof_addr(fixed_model, name)
                        control["actuation"] = {
                            "actuator_force": float(data.actuator_force[aid]),
                            "generalized_actuator_torque_nm": float(data.qfrc_actuator[dof]),
                            "qfrc_bias_nm": float(data.qfrc_bias[dof]),
                            "qacc_before_step_rad_s2": float(data.qacc[dof]),
                            "gear": fixed_model.actuator_gear[aid].tolist(),
                        }
                        max_requested[name] = max(max_requested[name], abs(control["requested_torque_nm"]))
                        max_applied[name] = max(max_applied[name], abs(control["applied_torque_nm"]))
                        saturation_steps[name] += int(bool(control["saturated"]))

                    mujoco.mj_step(fixed_model, data)
                    t_global += p0.accepted.TIMESTEP
                    step += 1
                    stage_rows += 1
                    contacts = p0.contact_forces(fixed_model, data)
                    measurements = p0.pad_contact_measurements(contacts)
                    force_by_pad = measurements["normal_force_by_pad_N"]
                    normal_vectors = [np.asarray(contact["force_on_bottle_world_N"], dtype=float)
                                      for contact in contacts if contact["other_body"] in PAD_NAMES]
                    resultant = (np.sum(normal_vectors, axis=0) if normal_vectors
                                 else np.zeros(3, dtype=float))
                    bottle_contacts = [contact for contact in contacts
                                       if contact["other_body"] in PAD_NAMES]
                    pen = max((contact["penetration_m"] for contact in bottle_contacts), default=0.0)
                    max_penetration = max(max_penetration, pen)
                    stage_max_penetration = max(stage_max_penetration, pen)
                    robot_contacts = all_relevant_robot_contacts(fixed_model, data)
                    for contact in robot_contacts:
                        bodies = {contact["body1"], contact["body2"]}
                        allowed = ("p0_canonical_test_cylinder" in bodies
                                   and bool(bodies & set(PAD_NAMES)))
                        if not allowed:
                            unexpected_contacts.add("|".join(sorted(bodies)))
                    if measurements["bilateral_contact"]:
                        stage_bilateral += 1

                    limits = p0.accepted.valid_joint_state(
                        fixed_model, data, source_joints=source_joint_elements
                    )
                    if limits:
                        source_limit_violations.extend({"step": step, **item} for item in limits)
                    finite = bool(np.isfinite(np.concatenate((data.qpos, data.qvel, data.qacc))).all())
                    max_abs_hand_speed = max(max_abs_hand_speed, *(
                        abs(float(data.qvel[p0.accepted.dof_addr(fixed_model, name)]))
                        for name in (DRIVER, FOLLOWER)
                    ))

                    for pad in PAD_NAMES:
                        stage_force_samples[pad].append(float(force_by_pad[pad]))
                        peak_instantaneous_force[pad] = max(peak_instantaneous_force[pad], float(force_by_pad[pad]))
                    penetration_over_limit = pen > p0.MAX_ALLOWED_CONTACT_PENETRATION_M
                    tracking_over_limit = any(
                        abs(float(hand_controls[name]["target_q_rad"]
                                  - data.qpos[p0.accepted.qpos_addr(fixed_model, name)]))
                        > p0.MAX_HOLD_JAW_TRACKING_ERROR_RAD
                        for name in (DRIVER, FOLLOWER)
                    )
                    speed_over_limit = any(
                        abs(float(data.qvel[p0.accepted.dof_addr(fixed_model, name)]))
                        > p0.PRELOAD_MAX_SPEED_MULTIPLE_OF_VALIDATED_CLOSE
                        * (1.875 * close_delta / p0.accepted.MOTION_DURATION)
                        for name in (DRIVER, FOLLOWER)
                    )
                    saturated = any(bool(hand_controls[name]["saturated"]) for name in (DRIVER, FOLLOWER))
                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED":
                        preload_force_rows.append({name: float(force_by_pad[name]) for name in PAD_NAMES})
                        preload_bilateral_flags.append(bool(measurements["bilateral_contact"]))
                        preload_penetrations.append(float(pen))
                        if saturated:
                            saturation_plateau_steps += 1
                        else:
                            saturation_plateau_steps = 0
                            force_history_start = {name: float(force_by_pad[name]) for name in PAD_NAMES}
                        if saturation_plateau_steps >= saturation_window_steps:
                            recent = preload_force_rows[-saturation_window_steps:]
                            progressed = max(
                                max((sample[name] for sample in recent), default=0.0)
                                - min((sample[name] for sample in recent), default=0.0)
                                for name in PAD_NAMES
                            )
                            if progressed < p0.PRELOAD_FORCE_PROGRESS_N:
                                failure = "ACTUATOR_SATURATION_PLATEAU"

                    row = {
                        "step": step, "time_s": float(data.time), "stage": stage,
                        "target_driver_q_rad": float(hand_controls[DRIVER]["target_q_rad"]),
                        "target_follower_q_rad": float(hand_controls[FOLLOWER]["target_q_rad"]),
                        "target_driver_qvel_rad_s": float(hand_controls[DRIVER]["target_qvel_rad_s"]),
                        "target_follower_qvel_rad_s": float(hand_controls[FOLLOWER]["target_qvel_rad_s"]),
                        "hand_joint_state": {
                            name: {"qpos_rad": float(data.qpos[p0.accepted.qpos_addr(fixed_model, name)]),
                                  "qvel_rad_s": float(data.qvel[p0.accepted.dof_addr(fixed_model, name)]),
                                  "qacc_rad_s2": float(data.qacc[p0.accepted.dof_addr(fixed_model, name)]),
                                  "position_error_rad": float(hand_controls[name]["target_q_rad"]
                                                               - data.qpos[p0.accepted.qpos_addr(fixed_model, name)]),
                                  "requested_torque_nm": float(hand_controls[name]["requested_torque_nm"]),
                                  "applied_torque_nm": float(hand_controls[name]["applied_torque_nm"]),
                                  "effort_cap_nm": float(hand_controls[name]["effort_cap_nm"]),
                                  "saturated": bool(hand_controls[name]["saturated"]),
                                  "qfrc_bias_nm": float(data.qfrc_bias[p0.accepted.dof_addr(fixed_model, name)]),
                                  "qfrc_actuator_nm": float(data.qfrc_actuator[p0.accepted.dof_addr(fixed_model, name)]),
                                  "qfrc_constraint_nm": float(data.qfrc_constraint[p0.accepted.dof_addr(fixed_model, name)])}
                            for name in (DRIVER, FOLLOWER)
                        },
                        "fixture_controller": slide_controls,
                        "contacts": contacts,
                        "pad_normal_force_N": {name: float(force_by_pad[name]) for name in PAD_NAMES},
                        "pad_force_resultant_world_N": resultant.tolist(),
                        "pad_force_resultant_lateral_xy_N": float(np.linalg.norm(resultant[:2])),
                        "pad_force_resultant_vertical_N": float(resultant[2]),
                        "bilateral_contact": bool(measurements["bilateral_contact"]),
                        "ideal_friction_capacity_upper_bound_N": float(
                            measurements["ideal_vertical_friction_capacity_upper_bound_N"]
                        ),
                        "maximum_pad_penetration_m": float(pen),
                        "source_limit_violations": limits,
                        "finite_state": finite,
                        "unexpected_contact_bodies": sorted(unexpected_contacts),
                        "fixed_object_has_freejoint": False,
                        "object_qpos_write_count": 0,
                    }
                    all_rows.append(row)

                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED":
                        if min(force_by_pad.values()) >= p0.accepted.CYLINDER_MASS * p0.accepted.GRAVITY / (
                                2.0 * p0.accepted.CONTACT_FRICTION_ASSUMPTION):
                            if peak_force_event is None:
                                peak_force_event = {"first_screening_force_step": step,
                                                    "first_screening_force_time_s": float(data.time)}

                    if not first_contact_saved and measurements["bilateral_contact"]:
                        Image.fromarray(render_frame(renderer, fixed_model, data, "close")).save(
                            output / "FIXED_OBJECT_FIRST_BILATERAL_CONTACT.png"
                        )
                        first_contact_saved = True
                    if stage == "HOLD" and local_step == local_steps - 1:
                        Image.fromarray(render_frame(renderer, fixed_model, data, "close")).save(
                            output / "FIXED_OBJECT_HOLD.png"
                        )
                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED" and local_step == local_steps - 1:
                        Image.fromarray(render_frame(renderer, fixed_model, data, "close")).save(
                            output / "FIXED_OBJECT_PRELOAD_END.png"
                        )
                    if step % fps_stride == 0 or (stage == "PRELOAD_TO_OPERATIONAL_CLOSED"
                                                  and local_step == local_steps - 1):
                        video.append_data(render_frame(renderer, fixed_model, data, "overview"))

                    if penetration_over_limit:
                        failure = "CONTACT_PENETRATION_LIMIT_2MM"
                    elif source_limit_violations:
                        failure = "SOURCE_JOINT_LIMIT"
                    elif not finite:
                        failure = "NONFINITE_STATE"
                    elif unexpected_contacts:
                        failure = "UNEXPECTED_ROBOT_CONTACT"
                    elif tracking_over_limit:
                        failure = "JAW_TRACKING_ERROR_20MM"
                    elif speed_over_limit:
                        failure = "JAW_SPEED_GUARD"
                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED" and failure is None:
                        # Count only windows whose endpoint passed every safety gate.
                        if len(preload_force_rows) >= required_stable_steps:
                            window = preload_force_rows[-required_stable_steps:]
                            flags = preload_bilateral_flags[-required_stable_steps:]
                            if all(flags):
                                window_min = {name: min(sample[name] for sample in window) for name in PAD_NAMES}
                                if min(window_min.values()) > min(peak_stable_force.values()):
                                    peak_stable_force = window_min
                                    peak_stable_window_start_step = step - required_stable_steps + 1
                    if stage == "PRELOAD_TO_OPERATIONAL_CLOSED" and failure:
                        break
                    if failure:
                        break
                stage_summary[stage] = {
                    "start_time_s": stage_start_time, "end_time_s": t_global,
                    "steps": stage_rows,
                    "bilateral_contact_fraction": stage_bilateral / max(stage_rows, 1),
                    "peak_normal_force_by_pad_N": {
                        pad: max(stage_force_samples[pad], default=0.0) for pad in PAD_NAMES
                    },
                    "mean_normal_force_by_pad_N": {
                        pad: float(np.mean(stage_force_samples[pad])) if stage_force_samples[pad] else 0.0
                        for pad in PAD_NAMES
                    },
                    "maximum_pad_penetration_m": stage_max_penetration,
                }
                if failure:
                    break

        with trace_path.open("w", encoding="utf-8") as trace:
            for row in all_rows:
                trace.write(json.dumps(row, separators=(",", ":"), default=p0.accepted.json_default) + "\n")
        renderer.close()

        stable_force_pass = min(peak_stable_force.values()) > 0.0
        target_screen_n = (p0.accepted.CYLINDER_MASS * p0.accepted.GRAVITY /
                           (2.0 * p0.accepted.CONTACT_FRICTION_ASSUMPTION))
        target_with_margin_n = p0.PRELOAD_CAPACITY_MARGIN * target_screen_n
        result = {
            "status": "PASS_DIAGNOSTIC_GATES" if failure is None else "FAIL_DIAGNOSTIC_GATE",
            "classification": "SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC; not free-object grasp or lift evidence",
            "first_failed_gate": failure,
            "fixed_object_state": fixed_body_record,
            "candidate_static_preflight": preflight,
            "candidate_translation_basis": "Y offset equals the mean Y of the two reconstructed cylinder-sidewall witnesses minus the saved bottle-center Y, centering their contact patches around the cylinder cross-section.",
            "baseline_free_contact_geometry": "FREE_CONTACT_GEOMETRY.json",
            "source_and_runtime_identity": "CONTROLLER_AND_MODEL_IDENTITY.json",
            "model_file": "SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC.xml",
            "trace": trace_path.name,
            "video": mp4_path.name,
            "source_ranges_unchanged": True,
            "contact_properties_unchanged": True,
            "controller_gains_and_effort_caps_unchanged": True,
            "new_exclusions": [], "equality_or_weld_constraints": 0,
            "cylinder_freejoint_present": False,
            "active_rollout_object_qpos_writes": 0,
            "preload_control": {
                "strategy": "fixed bounded slow closure ramp from measured bilateral HOLD aperture to the existing operational closed endpoint",
                "rate_fraction_of_validated_close": p0.PRELOAD_REFERENCE_RATE_FRACTION,
                "rate_limit_rad_s": ramp_rate_limit,
                "acceleration_limit_rad_s2": ramp_acceleration_limit,
            "peak_rate_rad_s": ramp_peak_speed,
            "profile": "bounded trapezoidal velocity with zero velocity at both endpoints",
                "ramp_budget_s": ramp_duration_budget,
                "post_ramp_hold_s": PRELOAD_SETTLE_S,
                "start_driver_target_rad": hold_ref["driver_q"],
                "end_driver_target_rad": closed_ref["driver_q"],
                "target_screen_per_pad_N": target_screen_n,
                "target_with_existing_margin_per_pad_N": target_with_margin_n,
            },
            "stage_summary": stage_summary,
            "maximum_instantaneous_normal_force_by_pad_N": peak_instantaneous_force,
            "maximum_0p20s_stable_normal_force_by_pad_N": peak_stable_force if stable_force_pass else None,
            "stable_force_window_start_step": peak_stable_window_start_step,
            "stable_force_window_duration_s": FORCE_STABLE_WINDOW_S if stable_force_pass else 0.0,
            "reached_ideal_two_pad_screening_force": bool(
                min(peak_stable_force.values()) >= target_screen_n if stable_force_pass else False
            ),
            "reached_existing_margin_force_target": bool(
                min(peak_stable_force.values()) >= target_with_margin_n if stable_force_pass else False
            ),
            "source_limit_violations": source_limit_violations,
            "unexpected_robot_contacts": sorted(unexpected_contacts),
            "maximum_pad_penetration_m": max_penetration,
            "maximum_abs_hand_joint_speed_rad_s": max_abs_hand_speed,
            "maximum_requested_torque_nm_by_joint": max_requested,
            "maximum_applied_torque_nm_by_joint": max_applied,
            "actuator_saturation_steps_by_joint": saturation_steps,
            "measured_threshold_interpretation": "Fixed object prevents lateral escape. Report force as force-generation evidence only; it is not applied free-object support.",
            "reproduction_command": " ".join(sys.argv),
        }
        write_json(output / "FIXED_OBJECT_RESULT.json", result)
        report = [
            "# OmniPicker fixed-object contact diagnostic",
            "",
            f"Status: **{result['status']}**",
            "",
            "This is a SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC. The cylinder was rigidly fixed by removing its free joint before compilation. It is not evidence of a free-object grasp, load transfer, or lift.",
            "",
            f"- First failed gate: `{failure or 'none'}`.",
            f"- Maximum instantaneous force by pad: `{peak_instantaneous_force}` N.",
            f"- Maximum 0.20 s stable force by pad: `{peak_stable_force if stable_force_pass else None}` N.",
            f"- Existing 2 mm penetration gate maximum: `{max_penetration * 1000:.3f}` mm.",
            f"- Source-limit violations: `{len(source_limit_violations)}`; unexpected robot contacts: `{sorted(unexpected_contacts)}`.",
            f"- Peak applied torque: `{max_applied}` Nm; caps remain unchanged.",
            "- See `FREE_CONTACT_GEOMETRY.json` for the saved free-cylinder force frame and original STL contact reconstruction.",
            "- See `FIXED_OBJECT_TRACE.jsonl`, `FIXED_OBJECT_RESULT.json`, and `FIXED_OBJECT_PRELOAD.mp4` for raw diagnostic evidence.",
            "",
            "Reproduction command:",
            "",
            "```bash",
            " ".join(sys.argv),
            "```",
            "",
        ]
        (output / "FIXED_OBJECT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-y-offset-mm", type=float, default=0.0)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    translation = np.array([0.0, args.candidate_y_offset_mm / 1000.0, 0.0])
    result = run_fixed_diagnostic(output, translation)
    print(json.dumps({
        "output": str(output), "status": result["status"],
        "first_failed_gate": result["first_failed_gate"],
        "maximum_instantaneous_normal_force_by_pad_N": result["maximum_instantaneous_normal_force_by_pad_N"],
        "maximum_0p20s_stable_normal_force_by_pad_N": result["maximum_0p20s_stable_normal_force_by_pad_N"],
        "maximum_pad_penetration_m": result["maximum_pad_penetration_m"],
        "reached_ideal_two_pad_screening_force": result["reached_ideal_two_pad_screening_force"],
        "reached_existing_margin_force_target": result["reached_existing_margin_force_target"],
    }, indent=2))
    return 0 if result["status"] == "PASS_DIAGNOSTIC_GATES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
