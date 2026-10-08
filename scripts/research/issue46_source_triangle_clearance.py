#!/usr/bin/env python3
"""Check pinned source collision STL surfaces against the canonical bottle."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signed_distance_to_geom(model, data, geom_id: int, points_world: np.ndarray) -> np.ndarray:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local = (points_world - data.geom_xpos[geom_id]) @ rotation
    size = model.geom_size[geom_id]
    kind = int(model.geom_type[geom_id])

    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radial = np.linalg.norm(local[:, :2], axis=1) - float(size[0])
        axial = np.abs(local[:, 2]) - float(size[1])
        outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
        return outside + np.minimum(np.maximum(radial, axial), 0.0)
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        z = np.maximum(np.abs(local[:, 2]) - float(size[1]), 0.0)
        return np.linalg.norm(np.column_stack((local[:, :2], z)), axis=1) - float(size[0])
    if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radii = np.maximum(np.asarray(size[:3], dtype=float), 1.0e-15)
        k0 = np.linalg.norm(local / radii, axis=1)
        k1 = np.linalg.norm(local / (radii * radii), axis=1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1.0e-15)
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        q = np.abs(local) - np.asarray(size[:3], dtype=float)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        return outside + np.minimum(np.max(q, axis=1), 0.0)
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.linalg.norm(local, axis=1) - float(size[0])
    raise RuntimeError(f"Unsupported canonical bottle geom type {kind} for geom {geom_id}")


def bottle_bound_radius(model, geom_id: int) -> float:
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    kind = int(model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return math.hypot(float(size[0]), float(size[1]))
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(size[0] + size[1])
    if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(np.linalg.norm(size[:3]))
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.linalg.norm(size[:3]))
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    raise RuntimeError(f"Unsupported geom type {kind}")


def sampled_triangle_points(triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return np.concatenate((
        a, b, c,
        0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a),
        (a + b + c) / 3.0,
    ), axis=0)


def analyze_surface(model, data, mesh, bottle_geoms, chunk_triangles=5000):
    triangles = corridor.world_triangles(mesh, model, data)
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    rows = []
    for bottle_id in bottle_geoms:
        bottle_center = data.geom_xpos[bottle_id]
        search_radius = bottle_bound_radius(model, bottle_id) + 0.02
        selected = np.flatnonzero(np.linalg.norm(centers - bottle_center, axis=1) <= radii + search_radius)
        min_distance = math.inf
        min_point = None
        negative_count = 0
        sample_count = 0
        for start in range(0, len(selected), chunk_triangles):
            ids = selected[start:start + chunk_triangles]
            points = sampled_triangle_points(triangles[ids])
            distances = signed_distance_to_geom(model, data, bottle_id, points)
            sample_count += len(distances)
            negative_count += int(np.count_nonzero(distances < 0.0))
            local_min = int(np.argmin(distances))
            if float(distances[local_min]) < min_distance:
                min_distance = float(distances[local_min])
                min_point = points[local_min].tolist()
        rows.append({
            "source_link": mesh["body"], "source_collision_mesh": mesh["path"].name,
            "source_collision_mesh_sha256": mesh["sha256"],
            "source_triangle_count": int(len(triangles)),
            "bottle_part": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_id),
            "sample_scheme": "all triangle vertices, edge midpoints, and centroids",
            "sampled_surface_points_near_part": sample_count,
            "negative_sdf_sample_count": negative_count,
            "source_surface_overlap_confirmed_by_sample": negative_count > 0,
            "minimum_sampled_signed_distance_m": None if not math.isfinite(min_distance) else min_distance,
            "minimum_sample_point_world_m": min_point,
            "metric_limit": "Sampled point-to-analytic bottle SDF; a negative sample proves source-surface overlap. A nonnegative result is not an exact separation certificate.",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["runtime"]["upstream_commit"] != corridor.X2_PIN:
        raise RuntimeError("Result source commit does not match the pinned X2 source")

    abc = load_module("issue46_source_tri_abc", corridor.ABC_PATH)
    audit, base = abc.modules()
    source_urdf = corridor.X2_URDF
    subtree = Path(result["source_subtree"]["path"])
    all_rows = []
    pose_rows = []
    annotated_views = {}
    view_root = out / "annotated_collision_views"
    view_root.mkdir(parents=True, exist_ok=True)

    for item in result["candidates"]:
        candidate = item["candidate"]
        source_frame = item["source_frame"]
        candidate["tcp_target_world_m"] = source_frame["tcp_world_m"]
        candidate["_frame"] = {
            "jaw_normal_narrow_to_wide_world": source_frame["jaw_normal_narrow_to_wide_world"],
            "insertion_axis_distal_toward_wrist_world": source_frame["insertion_axis_distal_toward_wrist_world"],
        }
        seed = {
            "name": candidate["name"],
            "root_position_world_m": candidate["root_position_world_m"],
            "root_rotation_world": candidate["root_rotation_world"],
            "previous_variables": {"aperture": candidate["contact_aperture_ratio"]},
        }
        model, data, _ = abc.x2_modules_and_model(audit, base, subtree, seed)
        root_base = np.asarray(candidate["root_position_world_m"], dtype=float)
        bottle_body = corridor.body_id(model, "m0_bottle")
        bottle_geoms = [g for g in range(model.ngeom)
                        if int(model.geom_bodyid[g]) == bottle_body
                        and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
        meshes = corridor.stl_meshes(abc, source_urdf, model, data)
        robot_geoms, bottle_geoms_for_render, table_geoms = corridor.collision_sets(abc, model)

        for phase, aperture in (("OPEN", 1.0), ("CONTACT_GEOMETRY", candidate["contact_aperture_ratio"])):
            corridor.set_pose(abc, base, model, data, root_base, np.zeros(3),
                              candidate["wrist_pitch_target_rad"], aperture)
            root_id = corridor.body_id(model, corridor.ROOT_LINK)
            root_error = float(np.linalg.norm(data.xpos[root_id] - root_base))
            jaw_a = corridor.collision_geom_for_body(model, corridor.NARROW)
            jaw_b = corridor.collision_geom_for_body(model, corridor.WIDE)
            witness = np.zeros(6)
            jaw_gap = float(mujoco.mj_geomDistance(model, data, jaw_a, jaw_b, 1.0, witness))
            pose_rows.append({
                "candidate": candidate["name"], "phase": phase,
                "source_pitch_rad": candidate["wrist_pitch_target_rad"],
                "root_position_world_m": root_base.tolist(), "reconstructed_root_fk_error_m": root_error,
                "contact_aperture_ratio": float(aperture), "jaw_gap_m": jaw_gap,
                "jaw_witness_midpoint_world_m": (0.5 * (witness[:3] + witness[3:])).tolist(),
                "target_tcp_world_m": candidate["tcp_target_world_m"],
                "compiled_pose_reconstruction_within_2um": root_error <= 2.0e-6,
            })
            collision = corridor.collision_state(
                abc, model, data, robot_geoms, bottle_geoms_for_render, table_geoms)
            phase_dir = view_root / candidate["name"] / phase.lower()
            phase_dir.mkdir(parents=True, exist_ok=True)
            view_names = ("front", "side", "top", "closeup") if phase == "OPEN" else ("closeup",)
            for view in view_names:
                corridor.render_view(
                    phase_dir, candidate, model, data, meshes, view, collision, phase_label=phase)
            annotated_views[f"{candidate['name']}::{phase}"] = {
                view: str((phase_dir / f"{candidate['name']}_{view}.png").relative_to(out))
                for view in view_names
            }
            for mesh in meshes:
                all_rows.extend({"candidate": candidate["name"], "phase": phase, **row}
                                for row in analyze_surface(model, data, mesh, bottle_geoms))
        candidate.pop("_frame", None)

    csv_path = out / "source_triangle_clearance.csv"
    fields = list(all_rows[0]) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "task": "Original X2 collision STL surface samples against canonical bottle parts",
        "source_upstream_commit": result["runtime"]["upstream_commit"],
        "source_urdf_sha256": result["runtime"]["urdf_sha256"],
        "mujoco_version": result["runtime"]["mujoco_native"],
        "method": "Every original STL triangle vertex, three edge midpoints, and centroid was transformed by the pinned source collision scale/origin and compiled link FK. Samples were evaluated against the exact analytic MuJoCo bottle primitive SDF where available; ellipsoid distance magnitude uses a standard signed approximation. Negative samples confirm source-surface overlap. Nonnegative samples do not certify separation.",
        "view_rendering": "Supplemental views draw original pinned collision STL facets whose projected centroids intersect each orthographic viewport with a 20 mm margin; the bottle is drawn from canonical MuJoCo primitive collision geometry. Views label the pose phase and show compiled MuJoCo distance results. Rendering does not alter the distance or source-surface sample calculations.",
        "pose_reconstruction": pose_rows,
        "annotated_collision_views": annotated_views,
        "source_mesh_bottle_part_rows": all_rows,
        "any_source_surface_overlap_per_candidate_phase": {
            f"{candidate}::{phase}": any(
                row["source_surface_overlap_confirmed_by_sample"]
                for row in all_rows if row["candidate"] == candidate and row["phase"] == phase
            )
            for candidate in sorted({row["candidate"] for row in all_rows})
            for phase in ("OPEN", "CONTACT_GEOMETRY")
        },
        "csv": csv_path.name,
    }
    (out / "source_triangle_clearance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "REPRODUCE.md").write_text(
        "# Source Triangle Clearance Reproduction\n\n"
        "From the isolated research checkout, after the static corridor result exists:\n\n"
        "    cd /home/lzy18001500226/robotsim-issue46-omnipicker-collision-audit-20261008\n"
        "    MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \\\n"
        "      scripts/research/issue46_source_triangle_clearance.py \\\n"
        f"      --result {result_path} --output {out}\n\n"
        "This verifies only the two saved candidate poses. It does not search poses or call mj_step.\n",
        encoding="utf-8")
    return 0


corridor = load_module(
    "issue46_corridor_module",
    Path(__file__).resolve().with_name("issue46_source_faithful_grasp_corridor.py"),
)


if __name__ == "__main__":
    raise SystemExit(main())
