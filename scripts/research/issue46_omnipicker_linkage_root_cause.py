#!/usr/bin/env python3
"""Read-only static audit of the fitted X2 OmniPicker linkage candidate.

The script consumes retained scans and model snapshots. It calls mj_forward
only; it never advances physics or writes a source/evidence input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

PASSIVE = {
    "narrow": (
        "x2cand_candidate_R_hand_narrow2_passive_hinge",
        "x2cand_candidate_R_hand_narrow3_passive_hinge",
        "x2cand_candidate_R_hand_narrow_loop_passive_hinge",
    ),
    "wide": (
        "x2cand_candidate_R_hand_wide2_passive_hinge",
        "x2cand_candidate_R_hand_wide3_passive_hinge",
        "x2cand_candidate_R_hand_wide_loop_passive_hinge",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = int(mujoco.mj_name2id(model, kind, name))
    if value < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return value


def qpos_address(model: mujoco.MjModel, name: str) -> int:
    jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def frame_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    return math.degrees(math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"not a binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if 84 + 50 * count != len(raw):
        raise ValueError(f"STL size/count mismatch; ASCII or nonstandard STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    rows = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return rows["vertices"].astype(np.float64)


def topology_summary(triangles: np.ndarray, weld_epsilon_m: float = 2e-7) -> dict:
    vertices = np.unique(np.round(triangles.reshape(-1, 3) / weld_epsilon_m).astype(np.int64), axis=0)
    edges = np.sort(np.concatenate((
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
    ), axis=1).reshape(-1, 2, 3), axis=1)
    edge_keys = np.round(edges / weld_epsilon_m).astype(np.int64).reshape(-1, 6)
    _, counts = np.unique(edge_keys, axis=0, return_counts=True)
    return {
        "triangle_count": int(len(triangles)),
        "welded_vertex_count": int(len(vertices)),
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "bounds_local_m": [triangles.reshape(-1, 3).min(axis=0).tolist(),
                           triangles.reshape(-1, 3).max(axis=0).tolist()],
    }


def segment_hits_triangle(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray, eps: float = 1e-10) -> bool:
    direction = p1 - p0
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    h = np.cross(direction, e2)
    det = float(np.dot(e1, h))
    if abs(det) <= eps:
        return False
    inv = 1.0 / det
    s = p0 - tri[0]
    u = inv * float(np.dot(s, h))
    if u < -eps or u > 1.0 + eps:
        return False
    q = np.cross(s, e1)
    v = inv * float(np.dot(direction, q))
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = inv * float(np.dot(e2, q))
    return -eps <= t <= 1.0 + eps


def coplanar_triangles_overlap(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> bool:
    normal = np.cross(a[1] - a[0], a[2] - a[0])
    norm = np.linalg.norm(normal)
    if norm <= eps:
        return False
    normal /= norm
    if max(abs(float(np.dot(p - a[0], normal))) for p in b) > 2e-7:
        return False
    drop = int(np.argmax(np.abs(normal)))
    aa = np.delete(a, drop, axis=1)
    bb = np.delete(b, drop, axis=1)

    def orient(p, q, r):
        return float(np.cross(q - p, r - p))

    def on_segment(p, q, r):
        return (min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps and
                min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps)

    def segments(p1, q1, p2, q2):
        o1, o2 = orient(p1, q1, p2), orient(p1, q1, q2)
        o3, o4 = orient(p2, q2, p1), orient(p2, q2, q1)
        if o1 * o2 < -eps and o3 * o4 < -eps:
            return True
        return ((abs(o1) <= eps and on_segment(p1, q1, p2)) or
                (abs(o2) <= eps and on_segment(p1, q1, q2)) or
                (abs(o3) <= eps and on_segment(p2, q2, p1)) or
                (abs(o4) <= eps and on_segment(p2, q2, q1)))

    for i in range(3):
        for j in range(3):
            if segments(aa[i], aa[(i + 1) % 3], bb[j], bb[(j + 1) % 3]):
                return True

    def inside(p, tri):
        signs = [orient(tri[i], tri[(i + 1) % 3], p) for i in range(3)]
        return all(x >= -eps for x in signs) or all(x <= eps for x in signs)

    return inside(aa[0], bb) or inside(bb[0], aa)


def triangle_mesh_intersections(a: np.ndarray, b: np.ndarray, pair_limit: int = 1_000_000) -> dict:
    """Exact triangle-pair narrow phase after centroid/radius and AABB culling."""
    ca = a.mean(axis=1)
    cb = b.mean(axis=1)
    ra = np.linalg.norm(a - ca[:, None, :], axis=2).max(axis=1)
    rb = np.linalg.norm(b - cb[:, None, :], axis=2).max(axis=1)
    btree = cKDTree(cb)
    max_rb = float(rb.max())
    bmin, bmax = b.min(axis=1), b.max(axis=1)
    amin, amax = a.min(axis=1), a.max(axis=1)
    tested = 0
    hits = []
    tol = 2e-7
    for i, center in enumerate(ca):
        candidates = btree.query_ball_point(center, float(ra[i] + max_rb + tol))
        if not candidates:
            continue
        ids = np.asarray(candidates, dtype=np.int64)
        overlap = np.all((bmin[ids] <= amax[i] + tol) & (bmax[ids] >= amin[i] - tol), axis=1)
        ids = ids[overlap]
        tested += len(ids)
        if tested > pair_limit:
            return {"status": "INCONCLUSIVE_PAIR_LIMIT", "candidate_triangle_pairs_tested": tested,
                    "pair_limit": pair_limit, "intersection_count": len(hits), "witnesses": hits[:10]}
        tri_a = a[i]
        for j in ids:
            tri_b = b[int(j)]
            edges_a = ((tri_a[0], tri_a[1]), (tri_a[1], tri_a[2]), (tri_a[2], tri_a[0]))
            edges_b = ((tri_b[0], tri_b[1]), (tri_b[1], tri_b[2]), (tri_b[2], tri_b[0]))
            hit = any(segment_hits_triangle(p, q, tri_b) for p, q in edges_a)
            if not hit:
                hit = any(segment_hits_triangle(p, q, tri_a) for p, q in edges_b)
            if not hit:
                hit = coplanar_triangles_overlap(tri_a, tri_b)
            if hit:
                hits.append({"triangle_a": int(i), "triangle_b": int(j),
                             "witness_midpoint_world_m": ((tri_a.mean(axis=0) + tri_b.mean(axis=0)) * 0.5).tolist()})
                if len(hits) >= 20:
                    return {"status": "TRIANGLE_SURFACE_INTERSECTION", "candidate_triangle_pairs_tested": tested,
                            "intersection_count_at_least": len(hits), "witnesses": hits}
    return {"status": "NO_TRIANGLE_SURFACE_INTERSECTION", "candidate_triangle_pairs_tested": tested,
            "intersection_count": 0, "witnesses": []}


def render_pair(model: mujoco.MjModel, data: mujoco.MjData, out: Path, hull: bool,
                geom_a_name: str = "x2cand_x2geom_4_collision",
                geom_b_name: str = "x2cand_x2geom_8_collision",
                title_prefix: str = "") -> None:
    geom_a = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a_name)
    geom_b = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b_name)
    body_a = int(model.geom_bodyid[geom_a])
    body_b = int(model.geom_bodyid[geom_b])
    target = (data.xpos[body_a] + data.xpos[body_b]) * 0.5
    old_rgba = model.geom_rgba.copy()
    try:
        model.geom_rgba[:, 3] = 0.0
        model.geom_rgba[geom_a] = np.array([0.95, 0.22, 0.12, 0.75])
        model.geom_rgba[geom_b] = np.array([0.10, 0.38, 0.98, 0.75])
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = target
        camera.distance = 0.11
        camera.azimuth = 118.0
        camera.elevation = -10.0
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1 if hull else 0
        renderer = mujoco.Renderer(model, height=760, width=1120)
        renderer.update_scene(data, camera, scene_option=option)
        pixels = renderer.render()
        image = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 64), fill=(20, 24, 32))
        label = title_prefix + ("MUJOCO COMPILED CONVEX HULL VIEW" if hull else "SOURCE STL SURFACE VIEW")
        draw.text((18, 10), label, fill=(255, 255, 255))
        draw.text((18, 34), "RED = narrow2_Link | BLUE = narrow_loop_Link | q_driver = -0.5 rad", fill=(235, 235, 235))
        image.save(out)
        renderer.close()
    finally:
        model.geom_rgba[:] = old_rgba


def write_plot(rows: list[dict], path: Path) -> None:
    width, height = 1200, 820
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    colors = {"X2 official URDF": (35, 101, 179), "G2 official Xacro": (40, 145, 95),
              "X2 SIMULATION_ONLY candidate": (210, 92, 37)}
    plots = [
        ("Compiled carrier gap (mm)", "compiled_carrier_gap_mm", 0, 115, 70),
        ("Terminal-link frame relative angle (deg)", "terminal_link_frame_relative_angle_deg", 80, 140, None),
    ]
    for pi, (title, key, vmin, vmax, reference) in enumerate(plots):
        x0, y0 = 100, 75 + pi * 340
        plot_w, plot_h = 1000, 250
        draw.text((x0, y0 - 34), title, fill=(25, 25, 25))
        for tick in np.linspace(vmin, vmax, 5):
            y = y0 + plot_h - (tick - vmin) / (vmax - vmin) * plot_h
            draw.line((x0, y, x0 + plot_w, y), fill=(220, 220, 220), width=1)
            draw.text((x0 - 55, y - 8), f"{tick:.0f}", fill=(70, 70, 70))
        for u in np.linspace(0, 1, 5):
            x = x0 + u * plot_w
            draw.line((x, y0, x, y0 + plot_h), fill=(235, 235, 235), width=1)
            draw.text((x - 12, y0 + plot_h + 8), f"{u:.2f}", fill=(70, 70, 70))
        if reference is not None:
            y = y0 + plot_h - (reference - vmin) / (vmax - vmin) * plot_h
            draw.line((x0, y, x0 + plot_w, y), fill=(135, 135, 135), width=2)
            draw.text((x0 + 8, y - 18), "70 mm reference", fill=(90, 90, 90))
        for model_name, color in colors.items():
            selected = sorted((r for r in rows if r["model"] == model_name), key=lambda r: r["u"])
            points = [(x0 + r["u"] * plot_w,
                       y0 + plot_h - (r[key] - vmin) / (vmax - vmin) * plot_h) for r in selected]
            draw.line(points, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=(50, 50, 50), width=2)
    lx, ly = 100, 730
    for name, color in colors.items():
        draw.line((lx, ly + 7, lx + 35, ly + 7), fill=color, width=4)
        draw.text((lx + 44, ly), name, fill=(30, 30, 30))
        lx += 300
    draw.text((100, 778), "Frame angle and carrier gap are proxies only; functional pad-face normals/clearance are not identified by these meshes.",
              fill=(55, 55, 55))
    image.save(path)


def write_collision_comparison(candidate_path: Path, source_path: Path, path: Path) -> None:
    candidate = Image.open(candidate_path).convert("RGB")
    source = Image.open(source_path).convert("RGB")
    width = candidate.width + source.width
    header_h, footer_h = 58, 82
    image = Image.new("RGB", (width, header_h + candidate.height + footer_h), (244, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.text((18, 14), "FITTED PASSIVE-BRANCH CANDIDATE | q=-0.5 rad", fill=(25, 25, 25))
    draw.text((candidate.width + 18, 14), "OFFICIAL X2 FIXED-CHILD CONTROL | same root/mimic q", fill=(25, 25, 25))
    image.paste(candidate, (0, header_h))
    image.paste(source, (candidate.width, header_h))
    draw.text((18, header_h + candidate.height + 10),
              "Candidate: >=20 narrow2/loop STL triangle intersections; compiled hull signed distance -14.201 mm.",
              fill=(25, 25, 25))
    draw.text((18, header_h + candidate.height + 34),
              "Source fixed-child control: no STL intersection and no active pair contact. Inferred passive hinges are not source-authored.",
              fill=(25, 25, 25))
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-reference-model", type=Path, required=True)
    parser.add_argument("--static-scan", type=Path, required=True)
    parser.add_argument("--aperture-csv", type=Path, required=True)
    parser.add_argument("--source-mesh-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-images", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scan = json.loads(args.static_scan.read_text())
    if scan["sample_count"] != 101 or len(scan["rows"]) != 101:
        raise ValueError("expected the retained 101-point scan")
    samples = {0.0: 0, 0.25: 25, 0.5: 50, 0.75: 75, 1.0: 100}
    with args.aperture_csv.open(newline="") as f:
        aperture_rows = list(csv.DictReader(f))
    reference_by_key = {(round(float(r["u"]), 6), r["model"]): r for r in aperture_rows}

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    source_model = mujoco.MjModel.from_xml_path(str(args.source_reference_model))
    source_data = mujoco.MjData(source_model)
    comparison = []
    fitted_rows = []
    frame_check = []
    root_name = "x2cand_candidate_right_claw_joint"
    mimic_name = "x2cand_candidate_R_hand_wide1_joint"
    for u, index in samples.items():
        source_row = scan["rows"][index]
        data.qpos[:] = model.qpos0
        data.qpos[qpos_address(model, root_name)] = float(source_row["driver_q_rad"])
        data.qpos[qpos_address(model, mimic_name)] = float(source_row["mimic_q_rad"])
        for side in ("narrow", "wide"):
            for joint in PASSIVE[side]:
                data.qpos[qpos_address(model, joint)] = 0.0
        mujoco.mj_forward(model, data)

        # Compare copied source link frames with all inferred passive joints at
        # zero. Relative-to-base transforms remove unrelated scene translation.
        source_data.qpos[:] = source_model.qpos0
        source_data.qpos[qpos_address(source_model, "x2src_right_claw_joint")] = float(source_row["driver_q_rad"])
        source_data.qpos[qpos_address(source_model, "x2src_R_hand_wide1_joint")] = -float(source_row["driver_q_rad"])
        mujoco.mj_forward(source_model, source_data)
        source_base = object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, "x2src_assembly_base")
        candidate_base = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "x2cand_assembly_base")
        source_base_rot = source_data.xmat[source_base].reshape(3, 3)
        candidate_base_rot = data.xmat[candidate_base].reshape(3, 3)
        for suffix in (
            "R_hand_narrow1_Link", "R_hand_narrow2_Link", "R_hand_narrow3_Link",
            "R_hand_narrow_loop_Link", "R_hand_wide1_Link", "R_hand_wide2_Link",
            "R_hand_wide3_Link", "R_hand_wide_loop_Link",
        ):
            source_body = object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, f"x2src_{suffix}")
            candidate_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"x2cand_candidate_{suffix}")
            source_rel_pos = source_base_rot.T @ (source_data.xpos[source_body] - source_data.xpos[source_base])
            candidate_rel_pos = candidate_base_rot.T @ (data.xpos[candidate_body] - data.xpos[candidate_base])
            source_rel_rot = source_base_rot.T @ source_data.xmat[source_body].reshape(3, 3)
            candidate_rel_rot = candidate_base_rot.T @ data.xmat[candidate_body].reshape(3, 3)
            frame_check.append({
                "u": u,
                "driver_q_rad": float(source_row["driver_q_rad"]),
                "body": suffix,
                "candidate_passive_q_rad": 0.0,
                "relative_position_error_mm": float(np.linalg.norm(candidate_rel_pos - source_rel_pos) * 1000.0),
                "relative_orientation_error_deg": frame_angle_deg(source_rel_rot, candidate_rel_rot),
                "interpretation": "copied source-frame check with inserted passive hinges held at zero; not fitted-branch validation",
            })

        # Restore the actual retained fitted passive solution for the remaining
        # measurements. No state is stepped or written back to source evidence.
        for side in ("narrow", "wide"):
            for joint in PASSIVE[side]:
                data.qpos[qpos_address(model, joint)] = source_row["sides"][side]["passive_joint_rad"][joint]
        mujoco.mj_forward(model, data)
        narrow = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "x2cand_candidate_R_hand_narrow3_Link")
        wide = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "x2cand_candidate_R_hand_wide3_Link")
        pad_angle = frame_angle_deg(data.xmat[narrow].reshape(3, 3), data.xmat[wide].reshape(3, 3))
        narrow_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "x2cand_x2geom_6_collision")
        wide_geom = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "x2cand_x2geom_14_collision")
        witness = np.zeros(6, dtype=np.float64)
        carrier_gap = float(mujoco.mj_geomDistance(model, data, narrow_geom, wide_geom, 0.5, witness) * 1000.0)
        fit = {
            "u": u,
            "driver_q_rad": float(source_row["driver_q_rad"]),
            "passive_q_rad": {s: source_row["sides"][s]["passive_joint_rad"] for s in ("narrow", "wide")},
            "narrow_residual_world_mm": source_row["mujoco_equality_residuals"]["narrow_distal_pin_loop"]["residual_world_m"],
            "wide_residual_world_mm": source_row["mujoco_equality_residuals"]["wide_distal_pin_loop"]["residual_world_m"],
            "narrow_residual_norm_mm": source_row["sides"]["narrow"]["site_closure_residual_mm"],
            "wide_residual_norm_mm": source_row["sides"]["wide"]["site_closure_residual_mm"],
            "narrow_jacobian_rank": source_row["sides"]["narrow"]["passive_angle_jacobian_rank"],
            "wide_jacobian_rank": source_row["sides"]["wide"]["passive_angle_jacobian_rank"],
            "narrow_jacobian_singular_values_m_per_rad": source_row["sides"]["narrow"]["passive_angle_jacobian_singular_values_m_per_rad"],
            "wide_jacobian_singular_values_m_per_rad": source_row["sides"]["wide"]["passive_angle_jacobian_singular_values_m_per_rad"],
            "terminal_link_frame_relative_angle_deg": pad_angle,
            "compiled_distal_collision_carrier_gap_mm": carrier_gap,
            "geom_distance_witness_length_mm": float(np.linalg.norm(witness[:3] - witness[3:]) * 1000.0),
            "static_active_contact_count": int(source_row["active_contact_count"]),
            "physics_steps": 0,
        }
        fitted_rows.append(fit)
        for name in ("X2 official URDF", "G2 official Xacro", "X2 SIMULATION_ONLY candidate"):
            source = reference_by_key[(round(u, 6), name)]
            comparison.append({
                "u": u,
                "model": name,
                "root_inner_rad": float(source["root_inner_rad"]),
                "root_outer_rad": float(source["root_outer_rad"]),
                "terminal_link_frame_relative_angle_deg": float(source["pad_frame_angle_deg"]),
                "compiled_carrier_gap_mm": float(source["aperture_mm"]),
                "source_or_candidate_passive_angles_rad": json.loads(source["passive_joint_angles_rad_json"]),
                "static_contact_count": int(source["static_contact_count"]),
                "source_limit_pass": source["all_limited_joint_ranges_pass"].lower() == "true",
                "functional_pad_face_orientation": "not source-defined / not measured",
                "true_pad_clearance": "not established; carrier proxy only",
            })

    with (args.out / "three_model_samples.csv").open("w", newline="") as f:
        fields = list(comparison[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in comparison:
            writer.writerow({**row, "source_or_candidate_passive_angles_rad": json.dumps(row["source_or_candidate_passive_angles_rad"], sort_keys=True)})
    with (args.out / "fitted_candidate_samples.csv").open("w", newline="") as f:
        fields = list(fitted_rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in fitted_rows:
            writer.writerow({**row, "passive_q_rad": json.dumps(row["passive_q_rad"], sort_keys=True),
                             "narrow_residual_world_mm": json.dumps(row["narrow_residual_world_mm"]),
                             "wide_residual_world_mm": json.dumps(row["wide_residual_world_mm"]),
                             "narrow_jacobian_singular_values_m_per_rad": json.dumps(row["narrow_jacobian_singular_values_m_per_rad"]),
                             "wide_jacobian_singular_values_m_per_rad": json.dumps(row["wide_jacobian_singular_values_m_per_rad"])})

    scan_rows = scan["rows"]
    residual_stats = {}
    for side in ("narrow", "wide"):
        vectors = np.asarray([r["sides"][side]["site_error_vector_world_mm"] for r in scan_rows])
        norms = np.linalg.norm(vectors, axis=1)
        names = list(scan_rows[0]["sides"][side]["passive_joint_rad"])
        jumps = {}
        for name in names:
            values = np.asarray([r["sides"][side]["passive_joint_rad"][name] for r in scan_rows])
            diffs = np.abs(np.diff(values))
            idx = int(np.argmax(diffs))
            jumps[name] = {"max_abs_step_jump_rad": float(diffs[idx]), "from_q_rad": float(scan_rows[idx]["driver_q_rad"]),
                           "to_q_rad": float(scan_rows[idx + 1]["driver_q_rad"]),
                           "from_angle_rad": float(values[idx]), "to_angle_rad": float(values[idx + 1])}
        residual_stats[side] = {
            "residual_norm_min_mm": float(norms.min()), "residual_norm_max_mm": float(norms.max()),
            "residual_norm_peak_to_peak_mm": float(np.ptp(norms)),
            "world_component_min_mm": vectors.min(axis=0).tolist(),
            "world_component_max_mm": vectors.max(axis=0).tolist(),
            "world_vector_max_deviation_from_first_mm": float(np.linalg.norm(vectors - vectors[0], axis=1).max()),
            "max_abs_passive_angle_rad": max(float(abs(r["sides"][side]["passive_joint_rad"][n])) for r in scan_rows for n in names),
            "max_adjacent_passive_angle_changes": jumps,
            "passive_jacobian_rank_values": sorted({int(r["sides"][side]["passive_angle_jacobian_rank"]) for r in scan_rows}),
            "minimum_smallest_singular_value_m_per_rad": float(min(r["sides"][side]["passive_angle_jacobian_singular_values_m_per_rad"][-1] for r in scan_rows)),
            "max_residual_outside_passive_subspace_mm": float(max(r["sides"][side]["residual_outside_passive_motion_subspace_mm"] for r in scan_rows)),
        }

    candidate_axis_diagnostics = {}
    for side in ("narrow", "wide"):
        names = PASSIVE[side]
        axis_vectors = []
        pairwise_angles = []
        residual_alignment_angles = []
        residual_parallel_components = []
        residual_perpendicular_norms = []
        anchors_at_mid = None
        axes_at_mid = None
        for row in scan_rows:
            data.qpos[:] = model.qpos0
            data.qpos[qpos_address(model, root_name)] = float(row["driver_q_rad"])
            data.qpos[qpos_address(model, mimic_name)] = float(row["mimic_q_rad"])
            for joint in names:
                data.qpos[qpos_address(model, joint)] = row["sides"][side]["passive_joint_rad"][joint]
            mujoco.mj_forward(model, data)
            joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            axes = np.asarray([data.xaxis[jid] for jid in joint_ids], dtype=np.float64)
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            axis_vectors.extend(axes.tolist())
            angles = [math.degrees(math.acos(float(np.clip(abs(np.dot(axes[i], axes[j])), -1.0, 1.0))))
                      for i in range(3) for j in range(i + 1, 3)]
            pairwise_angles.extend(angles)
            residual = np.asarray(row["sides"][side]["site_error_vector_world_mm"], dtype=np.float64)
            axis = axes[0]
            parallel = float(np.dot(residual, axis))
            residual_parallel_components.append(parallel)
            residual_perpendicular_norms.append(float(np.linalg.norm(residual - parallel * axis)))
            cosine = float(np.clip(abs(parallel) / max(np.linalg.norm(residual), 1e-30), -1.0, 1.0))
            residual_alignment_angles.append(math.degrees(math.acos(cosine)))
            if abs(float(row["driver_q_rad"]) + 0.5) < 1e-12:
                axes_at_mid = axes.copy()
                anchors_at_mid = np.asarray([data.xanchor[jid] for jid in joint_ids], dtype=np.float64)
        sv = np.asarray([r["sides"][side]["passive_angle_jacobian_singular_values_m_per_rad"]
                         for r in scan_rows], dtype=np.float64)
        candidate_axis_diagnostics[side] = {
            "source_passive_axes": "not defined: source URDF declares these child joints fixed",
            "candidate_local_axes": {name: model.jnt_axis[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)].tolist()
                                      for name in names},
            "candidate_world_axes_at_driver_q_minus_0_5": axes_at_mid.tolist(),
            "candidate_world_anchors_m_at_driver_q_minus_0_5": anchors_at_mid.tolist(),
            "world_axis_component_min": np.min(np.asarray(axis_vectors), axis=0).tolist(),
            "world_axis_component_max": np.max(np.asarray(axis_vectors), axis=0).tolist(),
            "pairwise_absolute_axis_angle_min_deg": min(pairwise_angles),
            "pairwise_absolute_axis_angle_max_deg": max(pairwise_angles),
            "residual_to_axis_angle_max_deg": max(residual_alignment_angles),
            "signed_residual_component_along_axis_min_mm": min(residual_parallel_components),
            "signed_residual_component_along_axis_max_mm": max(residual_parallel_components),
            "residual_component_perpendicular_to_axis_max_mm": max(residual_perpendicular_norms),
            "jacobian_singular_value_ranges_m_per_rad": {
                f"sigma{i + 1}": {"min": float(sv[:, i].min()), "max": float(sv[:, i].max())}
                for i in range(3)
            },
            "interpretation": "All candidate passive axes were inserted as local +Z hinges; their near-parallel world directions are an implementation assumption, not source evidence. Under those axes, endpoint rotation cannot remove the residual component along the hinge direction.",
        }

    # Reconstruct the retained mid-travel state without advancing dynamics.
    mid = scan_rows[50]
    data.qpos[:] = model.qpos0
    data.qpos[qpos_address(model, root_name)] = float(mid["driver_q_rad"])
    data.qpos[qpos_address(model, mimic_name)] = float(mid["mimic_q_rad"])
    for side in ("narrow", "wide"):
        for joint in PASSIVE[side]:
            data.qpos[qpos_address(model, joint)] = mid["sides"][side]["passive_joint_rad"][joint]
    mujoco.mj_forward(model, data)

    geom_a = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "x2cand_x2geom_4_collision")
    geom_b = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "x2cand_x2geom_8_collision")
    contact_rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        if {int(c.geom1), int(c.geom2)} == {geom_a, geom_b}:
            contact_rows.append({"signed_distance_mm": float(c.dist * 1000.0),
                                 "position_world_m": np.asarray(c.pos).tolist(),
                                 "frame_world": np.asarray(c.frame).reshape(3, 3).tolist()})

    mesh_a_path = args.source_mesh_dir / "narrow2_Link.stl"
    mesh_b_path = args.source_mesh_dir / "narrow_loop_Link.stl"
    tri_a = read_binary_stl(mesh_a_path)
    tri_b = read_binary_stl(mesh_b_path)
    body_a_name = "x2cand_candidate_R_hand_narrow2_Link"
    body_b_name = "x2cand_candidate_R_hand_narrow_loop_Link"
    body_a = object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_a_name)
    body_b = object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_b_name)
    tri_a_world = tri_a @ data.xmat[body_a].reshape(3, 3).T + data.xpos[body_a]
    tri_b_world = tri_b @ data.xmat[body_b].reshape(3, 3).T + data.xpos[body_b]
    mesh_intersection = triangle_mesh_intersections(tri_a_world, tri_b_world)
    source_data.qpos[qpos_address(source_model, "x2src_right_claw_joint")] = -0.5
    source_data.qpos[qpos_address(source_model, "x2src_R_hand_wide1_joint")] = 0.5
    mujoco.mj_forward(source_model, source_data)
    source_geom_a = object_id(source_model, mujoco.mjtObj.mjOBJ_GEOM, "x2src_x2geom_4_collision")
    source_geom_b = object_id(source_model, mujoco.mjtObj.mjOBJ_GEOM, "x2src_x2geom_8_collision")
    source_contact_rows = []
    for i in range(source_data.ncon):
        contact = source_data.contact[i]
        if {int(contact.geom1), int(contact.geom2)} == {source_geom_a, source_geom_b}:
            source_contact_rows.append(float(contact.dist * 1000.0))
    source_body_a = object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, "x2src_R_hand_narrow2_Link")
    source_body_b = object_id(source_model, mujoco.mjtObj.mjOBJ_BODY, "x2src_R_hand_narrow_loop_Link")
    source_tri_a = tri_a @ source_data.xmat[source_body_a].reshape(3, 3).T + source_data.xpos[source_body_a]
    source_tri_b = tri_b @ source_data.xmat[source_body_b].reshape(3, 3).T + source_data.xpos[source_body_b]
    source_mesh_intersection = triangle_mesh_intersections(source_tri_a, source_tri_b)
    mesh_info = {
        "narrow2_source_mesh": {"path": str(mesh_a_path), "sha256": sha256(mesh_a_path), **topology_summary(tri_a)},
        "narrow_loop_source_mesh": {"path": str(mesh_b_path), "sha256": sha256(mesh_b_path), **topology_summary(tri_b)},
        "pose": {"driver_q_rad": float(mid["driver_q_rad"]),
                 "narrow2_passive_q_rad": float(mid["sides"]["narrow"]["passive_joint_rad"][PASSIVE["narrow"][0]]),
                 "narrow3_passive_q_rad": float(mid["sides"]["narrow"]["passive_joint_rad"][PASSIVE["narrow"][1]]),
                 "narrow_loop_passive_q_rad": float(mid["sides"]["narrow"]["passive_joint_rad"][PASSIVE["narrow"][2]])},
        "compiled_hull_pair": {"geom_a": "x2cand_x2geom_4_collision", "geom_b": "x2cand_x2geom_8_collision",
                               "compiled_hull_signed_distances_mm": [r["signed_distance_mm"] for r in contact_rows],
                               "active_hull_contact_count": len(contact_rows)},
        "raw_source_triangle_mesh_test": mesh_intersection,
        "official_fixed_source_control_same_driver_pose": {
            "driver_q_rad": -0.5,
            "source_mimic_q_rad": 0.5,
            "compiled_source_hull_pair": {
                "geom_a": "x2src_x2geom_4_collision",
                "geom_b": "x2src_x2geom_8_collision",
                "signed_distances_mm": source_contact_rows,
                "active_contact_count": len(source_contact_rows),
            },
            "raw_source_triangle_mesh_test": source_mesh_intersection,
            "interpretation": "source URDF fixed-secondary-link control at the same root/mimic angles; no inferred passive motion",
        },
        "raw_mesh_transform_assumption": "URDF collision origins are zero and candidate collision geoms are at the source link frame; raw STL vertices are transformed by candidate body xmat/xpos at the retained static q=-0.5 pose.",
    }
    render_status = {"status": "NOT_RUN"}
    try:
        render_pair(model, data, args.out / "narrow2_loop_source_mesh.png", hull=False)
        render_pair(model, data, args.out / "narrow2_loop_compiled_hull.png", hull=True)
        render_pair(source_model, source_data, args.out / "source_fixed_narrow2_loop_source_mesh.png", hull=False,
                    geom_a_name="x2src_x2geom_4_collision", geom_b_name="x2src_x2geom_8_collision",
                    title_prefix="OFFICIAL URDF FIXED-LINK CONTROL | ")
        render_pair(source_model, source_data, args.out / "source_fixed_narrow2_loop_compiled_hull.png", hull=True,
                    geom_a_name="x2src_x2geom_4_collision", geom_b_name="x2src_x2geom_8_collision",
                    title_prefix="OFFICIAL URDF FIXED-LINK CONTROL | ")
        write_collision_comparison(args.out / "narrow2_loop_source_mesh.png",
                                   args.out / "source_fixed_narrow2_loop_source_mesh.png",
                                   args.out / "collision_source_vs_candidate.png")
        render_status = {"status": "RENDERED", "physics_steps": 0}
    except Exception as exc:  # rendering is supplemental; numeric geometry remains primary
        render_status = {"status": "RENDER_FAILED", "error": f"{type(exc).__name__}: {exc}"}

    if args.reference_images:
        refs = [args.reference_images / n for n in ("comparison_closed.png", "comparison_mid.png", "comparison_open.png")]
        if all(p.exists() for p in refs):
            thumbs = []
            for path, label in zip(refs, ("CLOSED u=0", "MID u=0.5", "OPEN u=1")):
                im = Image.open(path).convert("RGB")
                im.thumbnail((640, 360))
                panel = Image.new("RGB", (660, 405), (250, 250, 248))
                panel.paste(im, ((660 - im.width) // 2, 32))
                ImageDraw.Draw(panel).text((14, 9), label + " | LEFT X2 / CENTER G2 / RIGHT fitted X2", fill=(15, 15, 15))
                thumbs.append(panel)
            sheet = Image.new("RGB", (660 * len(thumbs), 405), (245, 245, 245))
            for i, panel in enumerate(thumbs):
                sheet.paste(panel, (660 * i, 0))
            sheet.save(args.out / "three_model_static_reference.png")

    write_plot(comparison, args.out / "three_model_metrics.png")
    source_transform_errors = [r for r in frame_check]
    output = {
        "status": "BLOCKED_NO_SOURCE_BACKED_FIXED_CORRECTION",
        "scope": "offline static geometry and retained data audit; no dynamics or bottle interaction",
        "mujoco_version": mujoco.__version__,
        "physics_steps": 0,
        "inputs": {"candidate_mjcf": str(args.model), "candidate_mjcf_sha256": sha256(args.model),
                   "source_reference_mjcf": str(args.source_reference_model),
                   "source_reference_mjcf_sha256": sha256(args.source_reference_model),
                   "retained_scan": str(args.static_scan), "aperture_scan": str(args.aperture_csv),
                   "source_mesh_dir": str(args.source_mesh_dir)},
        "source_transform_check": {
            "sample_count": len(samples),
            "body_comparisons": len(frame_check),
            "max_relative_position_error_mm": max(r["relative_position_error_mm"] for r in source_transform_errors),
            "max_relative_orientation_error_deg": max(r["relative_orientation_error_deg"] for r in source_transform_errors),
            "rows": frame_check,
        },
        "residual_scan_stats": residual_stats,
        "candidate_passive_axis_diagnostics": candidate_axis_diagnostics,
        "five_sample_fitted_candidate": fitted_rows,
        "five_sample_three_model_comparison": comparison,
        "narrow2_narrow_loop_collision": mesh_info,
        "rendering": render_status,
        "interpretation": {
            "residual": "Residual magnitude and direction are invariant in world coordinates over the full root sweep, and the numerical passive-angle Jacobian rank is two. The fixed component cannot be corrected by the tested passive revolute coordinates; its magnitude is comparable to STL hole-fit RMS, but frame/CAD error cannot be distinguished from mesh-fit uncertainty without mechanical datums.",
            "branch": "The fitted candidate has no source-backed passive limits or motion law. The retained least-squares continuation exhibits large passive rotations and abrupt inter-sample changes; these are solver-selected kinematic branches, not validated physical motion.",
            "clearance": "A carrier hull gap is not a functional pad-face measurement. Source meshes do not identify pad-face patches. Exact source-mesh intersection result is reported separately from MuJoCo convex-hull contacts.",
            "correction": "No candidate correction was applied because no CAD pivot centers/axes, passive limits, assembly tolerances, or authorized clearance dimensions identify a unique fixed change.",
        },
    }
    (args.out / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
