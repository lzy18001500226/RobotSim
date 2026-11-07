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


def envelope_samples(segments: np.ndarray, side: str, count: int = 161) -> np.ndarray:
    """Uniform-height inner envelope, independent of STL triangle density.

    This samples the entire approved section; upper/lower thirds below are
    diagnostic scoring bands, not a change to the approved contact mask.
    """
    low, high = segments[:, :, 1].min(), segments[:, :, 1].max()
    points = []
    for z in np.linspace(low + 1e-8, high - 1e-8, count):
        y = inner_at_height(segments, z, side)
        if y is not None:
            points.append([y, z])
    return np.asarray(points)


def solve_support_circle(arcs: list[np.ndarray], obstacles: np.ndarray, labels: list[str], previous: dict | None = None) -> dict:
    """Variable-radius clearance circle prioritizing separated support regions.

    Radius is always the actual minimum distance to ALL section obstacles.
    Minimize worst nearest gap in four outer-third bands, plus 0.1 times RMS
    gap of the whole inner envelope. This objective is a disclosed geometric
    heuristic, not a force/area metric. No compliant penetration is permitted.
    """
    low = max(float(a[:, :, 1].min()) for a in arcs) + 1e-8
    high = min(float(a[:, :, 1].max()) for a in arcs) - 1e-8
    if low >= high:
        return {"status": "NO_COMMON_SECTION"}
    profiles = [envelope_samples(a, side) for a, side in zip(arcs, ("narrow", "wide"))]
    bands = []
    for profile in profiles:
        z0, z1 = profile[:, 1].min(), profile[:, 1].max()
        bands.extend([profile[profile[:, 1] <= z0 + (z1-z0)/3], profile[profile[:, 1] >= z0 + 2*(z1-z0)/3]])
    if any(len(b) == 0 for b in bands):
        return {"status": "MISSING_SUPPORT_BAND"}
    _, groups = np.unique(labels, return_inverse=True)
    oa, ob = obstacles[:, 0], obstacles[:, 1]

    def inside(point):
        crossing = (oa[:, 1] > point[1]) != (ob[:, 1] > point[1])
        idx = np.flatnonzero(crossing)
        y = oa[idx, 0] + (point[1]-oa[idx, 1]) / (ob[idx, 1]-oa[idx, 1]) * (ob[idx, 0]-oa[idx, 0])
        return bool(np.any(np.bincount(groups[idx[y > point[0]]], minlength=int(groups.max())+1) % 2))

    def position(parameters):
        t, fraction = np.clip(parameters, 0, 1)
        z = low + fraction * (high-low)
        left, right = inner_at_height(arcs[0], z, "narrow"), inner_at_height(arcs[1], z, "wide")
        if left is None or right is None or left >= right:
            return None
        return np.array([left+t*(right-left), z])

    def evaluate(parameters):
        point = position(parameters)
        if point is None or inside(point):
            return 1.0
        radius = nearest(point, obstacles)[0]
        gaps = [float(np.min(np.linalg.norm(b-point, axis=1))-radius) for b in bands]
        all_gaps = np.concatenate([np.linalg.norm(p-point, axis=1)-radius for p in profiles])
        return max(gaps) + 0.1*float(np.sqrt(np.mean(all_gaps**2)))

    seeds = sorted((evaluate([t, z]), t, z) for z in np.linspace(0, 1, 65) for t in np.linspace(0, 1, 17))
    candidates = [(value, np.array([t,z])) for value,t,z in seeds[:12]]
    starts = [[t,z] for _,t,z in seeds[:8]]
    if previous is not None and "normalized_center_parameters" in previous:
        starts.append(previous["normalized_center_parameters"])
    for start in starts:
        result = minimize(evaluate, start, method="Nelder-Mead", bounds=[(0,1),(0,1)], options={"xatol":1e-8,"fatol":1e-11,"maxiter":250})
        candidates.append((float(result.fun), result.x))
    # Only resolve numerically equivalent candidates using continuity; do not
    # smooth a circle into geometry or hide a genuine change of solution branch.
    best = min(value for value,_ in candidates)
    equivalent = [(v,p) for v,p in candidates if v <= best+1e-7]
    if previous and "normalized_center_parameters" in previous:
        value, parameters = min(equivalent,key=lambda item: np.linalg.norm(item[1]-previous["normalized_center_parameters"]))
    else:
        value, parameters = min(equivalent,key=lambda item: item[0])
    point = position(parameters)
    if point is None or value >= 1.0:
        return {"status":"NO_FREE_GAP"}
    radius, limiting, index = nearest(point, obstacles)
    pad_contacts = [nearest(point, a) for a in arcs]
    support = []
    for band, name in zip(bands, ("narrow_lower","narrow_upper","wide_lower","wide_upper")):
        distances = np.linalg.norm(band-point, axis=1)
        k = int(np.argmin(distances))
        support.append({"band":name,"point_m":[0.,*band[k].tolist()],"gap_mm":float((distances[k]-radius)*1000)})
    gaps = np.array([s["gap_mm"] for s in support])
    bilateral = all(abs(distance-radius)<=0.0001 for distance,_,_ in pad_contacts)
    result = {
        "status":"FOUR_BANDS_NEAR_0_5MM" if bilateral and gaps.max()<=0.5 else "BILATERAL_LIMITED_SUPPORT" if bilateral else "OTHER_GEOMETRY_OR_ONE_SIDED_LIMIT",
        "center_m":[0.,*point.tolist()],"radius_m":radius,"diameter_mm":radius*2000,
        "nearest_geometry":labels[index],"minimum_section_clearance_m":radius,
        "pad_distances_mm":[d*1000 for d,_,_ in pad_contacts],
        "pad_contact_points_m":[[0.,*p.tolist()] for _,p,_ in pad_contacts],"limiting_point_m":[0.,*limiting.tolist()],
        "center_height_bounds_m":[low,high],"center_on_search_boundary":bool(min(parameters[1],1-parameters[1])<0.001),
        "normalized_center_parameters":parameters.tolist(),"objective_m":value,"support_bands":support,
        "worst_support_band_gap_mm":float(gaps.max()),"support_band_nearness_tolerance_mm":0.5,
        "method":"minimax_four_outer_third_gaps_plus_0.1_full_envelope_RMS; variable_radius; 65x17_grid_9_refinements",
        "global_maximum_certified":False,"finite_cylinder_collision_checked":False,"section_center_inside_mesh":inside(point),
        "physical_contact_area_measured":False,"bilateral_tolerance_mm":0.1,
    }
    result["gap_profiles"] = []
    for profile in profiles:
        gap = np.linalg.norm(profile-point, axis=1)-radius
        ds = np.linalg.norm(np.diff(profile,axis=0),axis=1)
        near = gap <= 0.0005
        # Section-length estimate only. This is neither rubber compression nor
        # a 3-D contact-area measurement, and is sensitive to section sampling.
        result["gap_profiles"].append({"points_m":[[0.,*p.tolist()] for p in profile],"gap_mm":(gap*1000).tolist(),
            "near_0_5mm_section_length_mm":float(np.sum(ds[(near[:-1]&near[1:])])*1000),
            "rms_gap_mm":float(np.sqrt(np.mean(gap**2))*1000)})
    return result


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
        legacy = solve_circle(arcs, np.asarray(cuts), names)
        result = solve_support_circle(arcs, np.asarray(cuts), names, output[-1] if output else None)
        result["maximum_clearance_reference"] = legacy
        result["opening_percent"] = opening
        result["arc_sections_m"] = [a.tolist() for a in arcs]
        output.append(result)
        if opening % 10 == 0:
            print(f"circle {opening}%: {result.get('diameter_mm', 0):.3f} mm {result['status']}", flush=True)
    return {"classification": "STATIC_VARIABLE_DIAMETER_SUPPORT_CIRCLE_REVIEW_ONLY", "axis": [1, 0, 0], "section_x_m": 0.0,
            "domain": "centres between approved lower-arc inner envelopes, within their shared section height; radius unconstrained by artificial height caps",
            "obstacles": "all displayed source visual triangle sections, including grey tips and supports; not MuJoCo convex hulls",
            "support_scoring":"minimize worst lower/upper-third gap across both pads plus 0.1 full-profile RMS; 0.5mm is display proximity, not physical contact",
            "region_human_approved": True, "mj_step_calls": 0, "samples": output}
