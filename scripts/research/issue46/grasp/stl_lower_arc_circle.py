"""Offline cross-section clearance circles; never modifies robot geometry.

The open groove has no unconstrained maximum circle. Search circle CENTRES
inside the common height of the approved arcs at gripper X=0. Radii are limited
by actual mesh cross-sections, not artificial end caps. This is a numerical
maximum in that explicitly bounded centre domain, not a global 3-D grasp proof.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def section_segments(triangles: np.ndarray, x: float = 0.0) -> np.ndarray:
    """Intersect original triangles with X=x, retaining exact YZ line segments."""
    a, b = triangles, triangles[:, [1, 2, 0]]
    da, db = a[:, :, 0] - x, b[:, :, 0] - x
    crossing = ((da <= 0) & (db > 0)) | ((da > 0) & (db <= 0))
    keep = crossing.sum(axis=1) == 2
    a, b, da, db, crossing = a[keep], b[keep], da[keep], db[keep], crossing[keep]
    fraction = np.divide(da, da - db, out=np.zeros_like(da), where=abs(da - db) > 1e-15)
    points = a + fraction[:, :, None] * (b - a)
    parts = points[:, :, 1:][crossing].reshape(-1, 2, 2)
    return parts[np.linalg.norm(parts[:, 1] - parts[:, 0], axis=1) > 1e-12]


def nearest(point: np.ndarray, segments: np.ndarray) -> tuple[float, np.ndarray, int]:
    a, direction = segments[:, 0], segments[:, 1] - segments[:, 0]
    t = np.clip(np.sum((point - a) * direction, axis=1) / np.sum(direction * direction, axis=1), 0, 1)
    points = a + t[:, None] * direction
    d2 = np.sum((point - points) ** 2, axis=1)
    index = int(np.argmin(d2))
    return float(np.sqrt(d2[index])), points[index], index


def inner_at_height(segments: np.ndarray, height: float, side: str) -> float | None:
    a, b = segments[:, 0], segments[:, 1]
    dz = b[:, 1] - a[:, 1]
    mask = (np.minimum(a[:, 1], b[:, 1]) - 1e-10 <= height) & (height <= np.maximum(a[:, 1], b[:, 1]) + 1e-10) & (abs(dz) > 1e-12)
    if not np.any(mask):
        return None
    ys = a[mask, 0] + (height - a[mask, 1]) / dz[mask] * (b[mask, 0] - a[mask, 0])
    return float(np.max(ys) if side == "narrow" else np.min(ys))


def solve_circle(arcs: list[np.ndarray], obstacles: np.ndarray, labels: list[str]) -> dict:
    low = max(float(a[:, :, 1].min()) for a in arcs) + 1e-8
    high = min(float(a[:, :, 1].max()) for a in arcs) - 1e-8
    if low >= high:
        return {"status": "NO_COMMON_SECTION"}
    _, groups = np.unique(labels, return_inverse=True)

    def inside_mesh_section(point):
        a, b = obstacles[:, 0], obstacles[:, 1]
        crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
        indices = np.flatnonzero(crossing)
        y = a[indices, 0] + (point[1] - a[indices, 1]) / (b[indices, 1] - a[indices, 1]) * (b[indices, 0] - a[indices, 0])
        counts = np.bincount(groups[indices[y > point[0]]], minlength=int(groups.max()) + 1)
        return bool(np.any(counts % 2))

    def position(parameters):
        t, fraction = np.clip(parameters, 0, 1)
        z = low + fraction * (high - low)
        left, right = inner_at_height(arcs[0], z, "narrow"), inner_at_height(arcs[1], z, "wide")
        if left is None or right is None or left >= right:
            return None
        return np.array([left + t * (right - left), z])

    def objective(parameters):
        point = position(parameters)
        if point is None:
            return 1.0
        distance = nearest(point, obstacles)[0]
        return distance if inside_mesh_section(point) else -distance

    seeds = [(objective((t, z)), t, z) for z in np.linspace(0, 1, 65) for t in np.linspace(0, 1, 17)]
    seeds.sort()
    candidates = [(value, np.array([t, z])) for value, t, z in seeds[:12]]
    for _, t, z in seeds[:6]:
        result = minimize(objective, [t, z], method="Nelder-Mead", bounds=[(0, 1), (0, 1)], options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 220})
        candidates.append((float(result.fun), result.x))
    value, parameters = min(candidates, key=lambda item: item[0])
    point = position(parameters)
    if point is None or value >= 0:
        return {"status": "NO_FREE_GAP"}
    radius, contact, index = nearest(point, obstacles)
    pad_contacts = [nearest(point, arc) for arc in arcs]
    bilateral = all(abs(distance - radius) <= 0.0001 for distance, _, _ in pad_contacts)
    touches_height_bound = bool(min(parameters[1], 1 - parameters[1]) < 0.001)
    return {
        "status": "BILATERAL_LOWER_ARC" if bilateral else "OTHER_GEOMETRY_OR_ONE_SIDED_LIMIT",
        "center_m": [0.0, *point.tolist()], "radius_m": radius, "diameter_mm": radius * 2000,
        "nearest_geometry": labels[index], "minimum_section_clearance_m": radius,
        "pad_distances_mm": [d * 1000 for d, _, _ in pad_contacts],
        "pad_contact_points_m": [[0.0, *p.tolist()] for _, p, _ in pad_contacts],
        "limiting_point_m": [0.0, *contact.tolist()],
        "bilateral_tolerance_mm": 0.1, "center_height_bounds_m": [low, high],
        "center_on_search_boundary": touches_height_bound,
        "grid_best_radius_m": -seeds[0][0], "method": "65x17_grid_plus_6_local_refinements",
        "global_maximum_certified": False, "finite_cylinder_collision_checked": False,
        "section_center_inside_mesh": inside_mesh_section(point),
    }


def scan_scene(scene: dict) -> dict:
    geoms = {str(g["id"]): g for g in scene["geoms"]}
    local = {}
    for gid, geom in geoms.items():
        mesh = scene["meshes"][str(geom["mesh"])]
        local[gid] = np.asarray(mesh["vertices"]).reshape(-1, 3)[np.asarray(mesh["faces"]).reshape(-1, 3)]
    output = []
    for opening, pose in enumerate(scene["poses"]):
        cuts, names, arcs = [], [], []
        sections = {}
        for gid, triangles in local.items():
            world = triangles @ np.asarray(pose["rotation"][gid]).reshape(3, 3).T + np.asarray(pose["position"][gid])
            sections[gid] = world
            cut = section_segments(world)
            cuts.extend(cut)
            names.extend([geoms[gid]["name"]] * len(cut))
        for pad in scene["pads"].values():
            arcs.append(section_segments(sections[str(pad["geom"])][pad["bottle_arc_review"]["source_face_indices"]]))
        result = solve_circle(arcs, np.asarray(cuts), names)
        result["opening_percent"] = opening
        result["arc_sections_m"] = [a.tolist() for a in arcs]
        output.append(result)
        if opening % 10 == 0:
            print(f"circle {opening}%: {result.get('diameter_mm', 0):.3f} mm {result['status']}", flush=True)
    return {"classification": "STATIC_CROSS_SECTION_NUMERICAL_CLEARANCE_REVIEW_ONLY", "axis": [1, 0, 0], "section_x_m": 0.0,
            "domain": "centres between approved lower-arc inner envelopes, within their shared section height; radius unconstrained by artificial height caps",
            "obstacles": "all displayed source visual triangle sections, including grey tips and supports; not MuJoCo convex hulls",
            "region_human_approved": True, "mj_step_calls": 0, "samples": output}
