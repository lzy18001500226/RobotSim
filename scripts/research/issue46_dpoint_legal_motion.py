#!/usr/bin/env python3
"""Apply the approved D-site map and audit the source-sampled legal interval."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import heapq
import json
import math
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


REPO = Path("/home/lzy18001500226/projects/RobotSim")
UPSTREAM = Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0")
SOURCE_MJCF = REPO / "reviews/issue-46/issue46-official-urdf-candidate-b-20261021T081308Z/candidate_B_official_meshes.xml"
SOURCE_URDF = UPSTREAM / "X2-Ultra_omnipicker.urdf"
MOTION_CSV = Path(__file__).resolve().parents[2] / "docs/research/issue46_candidate_b_motion_samples.csv"
SCHEMA = "robotsim.issue46.dpoint-legal-motion.v1"
STATIC_GAP_LIMIT_M = 0.001
GROSS_PENETRATION_LIMIT_M = 0.002
LEGAL_CAD_Q_MIN = -0.10
TRI_EPS_M = 1e-8

D_SITE_MAP = {
    "candidate_narrow_D_jaw": ("0 -0.0104 0", "0.000691289490 -0.010408336990 -0.000011495782"),
    "candidate_narrow_D_loop": ("0.0316000012456 -0.00972182496859 -2.81056021852e-05", "0.032861209960 -0.003021010867 -0.000039601384"),
    "candidate_wide_D_jaw": ("0 0.0104 0", "0.000691281259 0.010408390232 0.000011860848"),
    "candidate_wide_D_loop": ("0.0315999987547 0.00972180634187 2.79793618239e-05", "0.032861206432 0.003021046105 0.000039840210"),
}

STL_SHA256 = {
    "narrow1_Link": "82a707fe539d02f571e075a06c59b6846567ce61d741285b0f86f2d934cabb99",
    "narrow3_Link": "0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0",
    "narrow_loop_Link": "63b895ff646b799c303c41c924ea4c77cf00f9c5d357e2fb4f6761c649237b17",
    "wide1_Link": "c858476345282813a6311bd0501787f15431f58daf638ce4381d241250c98075",
    "wide3_Link": "a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb",
    "wide_loop_Link": "31aa42a993b3364e16efa657700ffe621eeaa5994d415bbe1b77f0687419b2b8",
}

COLLISION_PAIRS = {
    "narrow_root_loop": ("L_hand_narrow1_Link__collision", "L_hand_narrow_loop_Link__collision"),
    "narrow_distal_loop": ("L_hand_narrow3_Link__collision", "L_hand_narrow_loop_Link__collision"),
    "wide_root_loop": ("L_hand_wide1_Link__collision", "L_hand_wide_loop_Link__collision"),
    "wide_distal_loop": ("L_hand_wide3_Link__collision", "L_hand_wide_loop_Link__collision"),
}

JOINTS = {
    "narrow": {"root": "left_claw_joint", "B": "L_hand_narrow2_joint", "C": "L_hand_narrow3_joint", "E": "L_hand_narrow_loop_joint"},
    "wide": {"root": "L_hand_wide1_joint", "B": "L_hand_wide2_joint", "C": "L_hand_wide3_joint", "E": "L_hand_wide_loop_joint"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_corrected_xml(source: Path) -> tuple[bytes, str]:
    original = source.read_text(encoding="utf-8")
    root = ET.fromstring(original)
    tags = list(re.finditer(r"<site\b[^>]*>", original))
    patches: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for match in tags:
        element = ET.fromstring(match.group(0))
        name = element.get("name")
        if name not in D_SITE_MAP:
            continue
        if name in seen:
            raise ValueError(f"duplicate site in source MJCF: {name}")
        seen.add(name)
        old, new = D_SITE_MAP[name]
        if element.get("pos") != old:
            raise ValueError(f"source site does not match reviewed baseline: {name}")
        tag = match.group(0)
        old_attr = f'pos="{old}"'
        if tag.count(old_attr) != 1:
            raise ValueError(f"cannot safely patch source site position: {name}")
        patches.append((match.start(), match.end(), tag.replace(old_attr, f'pos="{new}"', 1)))
    if seen != set(D_SITE_MAP):
        raise ValueError(f"source MJCF is missing reviewed D sites: {sorted(set(D_SITE_MAP) - seen)}")
    updated = original
    for start, end, replacement in reversed(patches):
        updated = updated[:start] + replacement + updated[end:]
    check = ET.fromstring(updated)
    for name, (_, pos) in D_SITE_MAP.items():
        element = check.find(f".//site[@name='{name}']")
        if element is None or element.get("pos") != pos:
            raise AssertionError(f"corrected site did not round-trip: {name}")
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="candidate_B_official_meshes.xml (source)",
        tofile="corrected_candidate_B.xml (D sites only)",
    ))
    return updated.encode("utf-8"), patch


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"short STL file: {path.name}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"expected binary STL with exact record count: {path.name}")
    records = np.frombuffer(raw, dtype=np.dtype([
        ("normal", "<f4", (3,)), ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)), ("v2", "<f4", (3,)), ("attr", "<u2"),
    ]), offset=84)
    triangles = np.stack((records["v0"], records["v1"], records["v2"]), axis=1).astype(np.float64)
    unique, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    undirected = np.sort(directed, axis=1)
    edge_values, edge_counts = np.unique(undirected, axis=0, return_counts=True)
    directions = np.where(directed[:, 0] <= directed[:, 1], 1, -1)
    edge_sums = np.zeros(len(edge_values), dtype=np.int64)
    edge_index = np.searchsorted(edge_values[:, 0] * len(unique) + edge_values[:, 1], undirected[:, 0] * len(unique) + undirected[:, 1])
    np.add.at(edge_sums, edge_index, directions)
    source_volume = abs(float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0))
    hull_volume = float(ConvexHull(unique).volume)
    topology_ok = bool(np.all(edge_counts == 2) and np.all(edge_sums == 0))
    return triangles, unique, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "triangle_count": int(count),
        "unique_vertex_count": int(len(unique)),
        "bounds_m": np.ptp(unique, axis=0).tolist(),
        "source_signed_volume_abs_m3": source_volume,
        "convex_hull_volume_m3": hull_volume,
        "source_to_hull_volume_ratio": source_volume / hull_volume if hull_volume else None,
        "closed_oriented_two_manifold": topology_ok,
        "non_two_manifold_edge_count": int(np.count_nonzero(edge_counts != 2)),
        "inconsistently_oriented_edge_count": int(np.count_nonzero(edge_sums != 0)),
    }


def segment_triangle_hit(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray) -> bool:
    direction = p1 - p0
    edge1 = tri[1] - tri[0]
    edge2 = tri[2] - tri[0]
    h = np.cross(direction, edge2)
    det = float(np.dot(edge1, h))
    scale = max(float(np.linalg.norm(edge1) * np.linalg.norm(h)), 1e-30)
    if abs(det) <= 1e-12 * scale:
        return False
    inv = 1.0 / det
    s = p0 - tri[0]
    u = float(np.dot(s, h) * inv)
    if u < -1e-10 or u > 1.0 + 1e-10:
        return False
    q = np.cross(s, edge1)
    v = float(np.dot(direction, q) * inv)
    if v < -1e-10 or u + v > 1.0 + 1e-10:
        return False
    t = float(np.dot(edge2, q) * inv)
    return -1e-10 <= t <= 1.0 + 1e-10


def orient2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segment_segment_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float) -> bool:
    o1, o2, o3, o4 = orient2(a, b, c), orient2(a, b, d), orient2(c, d, a), orient2(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray, value: float) -> bool:
        return abs(value) <= eps and np.all(q >= np.minimum(p, r) - eps) and np.all(q <= np.maximum(p, r) + eps)
    return on_segment(a, c, b, o1) or on_segment(a, d, b, o2) or on_segment(c, a, d, o3) or on_segment(c, b, d, o4)


def point_in_triangle_2d(point: np.ndarray, tri: np.ndarray, eps: float) -> bool:
    a, b, c = tri
    area = orient2(a, b, c)
    if abs(area) <= eps:
        return False
    x1, x2, x3 = orient2(a, b, point), orient2(b, c, point), orient2(c, a, point)
    return (x1 >= -eps and x2 >= -eps and x3 >= -eps) or (x1 <= eps and x2 <= eps and x3 <= eps)


def triangles_intersect(a: np.ndarray, b: np.ndarray, eps: float = TRI_EPS_M) -> bool:
    if np.any(a.max(axis=0) < b.min(axis=0) - eps) or np.any(b.max(axis=0) < a.min(axis=0) - eps):
        return False
    for i in range(3):
        if segment_triangle_hit(a[i], a[(i + 1) % 3], b) or segment_triangle_hit(b[i], b[(i + 1) % 3], a):
            return True
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    la, lb = float(np.linalg.norm(na)), float(np.linalg.norm(nb))
    if la <= eps or lb <= eps:
        return False
    if np.linalg.norm(np.cross(na, nb)) > 1e-10 * la * lb or np.max(np.abs((b - a[0]) @ (na / la))) > eps:
        return False
    drop = int(np.argmax(np.abs(na)))
    aa = np.delete(a, drop, axis=1)
    bb = np.delete(b, drop, axis=1)
    for i in range(3):
        for j in range(3):
            if segment_segment_2d(aa[i], aa[(i + 1) % 3], bb[j], bb[(j + 1) % 3], eps):
                return True
    return point_in_triangle_2d(aa[0], bb, eps) or point_in_triangle_2d(bb[0], aa, eps)


@dataclass
class BVHNode:
    low: np.ndarray
    high: np.ndarray
    indices: np.ndarray | None
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None


class TriangleBVH:
    def __init__(self, triangles: np.ndarray, leaf_size: int = 12):
        self.triangles = triangles
        self.tri_low = triangles.min(axis=1)
        self.tri_high = triangles.max(axis=1)
        self.centers = (self.tri_low + self.tri_high) * 0.5
        self.root = self._build(np.arange(len(triangles), dtype=np.int32), leaf_size)

    def _build(self, indices: np.ndarray, leaf_size: int) -> BVHNode:
        low = self.tri_low[indices].min(axis=0)
        high = self.tri_high[indices].max(axis=0)
        if len(indices) <= leaf_size:
            return BVHNode(low, high, indices)
        axis = int(np.argmax(np.ptp(self.centers[indices], axis=0)))
        ordered = indices[np.argsort(self.centers[indices, axis], kind="mergesort")]
        middle = len(ordered) // 2
        return BVHNode(low, high, None, self._build(ordered[:middle], leaf_size), self._build(ordered[middle:], leaf_size))

    @staticmethod
    def _overlap(a: BVHNode, b: BVHNode, eps: float) -> bool:
        return bool(np.all(a.low <= b.high + eps) and np.all(b.low <= a.high + eps))

    def first_intersection(self, other: "TriangleBVH", *, eps: float = TRI_EPS_M) -> dict[str, object] | None:
        stack = [(self.root, other.root)]
        checked = 0
        while stack:
            a, b = stack.pop()
            if not self._overlap(a, b, eps):
                continue
            if a.indices is not None and b.indices is not None:
                for ia in a.indices:
                    candidate = b.indices[
                        np.all(self.tri_low[ia] <= other.tri_high[b.indices] + eps, axis=1)
                        & np.all(other.tri_low[b.indices] <= self.tri_high[ia] + eps, axis=1)
                    ]
                    for ib in candidate:
                        checked += 1
                        if triangles_intersect(self.triangles[ia], other.triangles[ib], eps):
                            return {"triangle_a": int(ia), "triangle_b": int(ib), "candidate_triangle_pairs_checked": checked}
                continue
            if b.indices is not None or (a.indices is None and np.prod(a.high - a.low) >= np.prod(b.high - b.low)):
                stack.append((a.left, b))
                stack.append((a.right, b))
            else:
                stack.append((a, b.left))
                stack.append((a, b.right))
        return {"triangle_a": None, "triangle_b": None, "candidate_triangle_pairs_checked": checked}


def point_inside_mesh(point: np.ndarray, triangles: np.ndarray) -> bool:
    directions = np.array([[1.0, 0.371390676, 0.129837], [0.213719, 1.0, 0.448197], [0.614321, 0.173911, 1.0]], dtype=np.float64)
    votes = []
    for direction in directions:
        direction /= np.linalg.norm(direction)
        e1 = triangles[:, 1] - triangles[:, 0]
        e2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(det) > 1e-14
        inv = np.zeros_like(det)
        inv[valid] = 1.0 / det[valid]
        s = point - triangles[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ direction)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits = valid & (t > 1e-9) & (u > 1e-10) & (v > 1e-10) & (u + v < 1.0 - 1e-10)
        votes.append(bool(np.count_nonzero(hits) % 2))
    return sum(votes) >= 2


def solid_overlap(
    a: TriangleBVH,
    b: TriangleBVH,
    *,
    closed_a: bool,
    closed_b: bool,
) -> dict[str, object]:
    intersection = a.first_intersection(b)
    if intersection is None:
        raise AssertionError("BVH intersection result must be explicit")
    if intersection["triangle_a"] is not None:
        return {"surface_intersection_or_contact": True, "containment": False, "containment_verified": True, **intersection}
    # Closed source meshes can overlap without crossing only if one contains the other.
    if not (closed_a and closed_b):
        return {
            "surface_intersection_or_contact": False,
            "containment": None,
            "containment_verified": False,
            "containment_note": "at least one source STL is not a closed consistently oriented two-manifold",
            "candidate_triangle_pairs_checked": intersection["candidate_triangle_pairs_checked"],
        }
    sample_a = a.triangles[::max(1, len(a.triangles) // 8), 0]
    sample_b = b.triangles[::max(1, len(b.triangles) // 8), 0]
    a_in_b = any(point_inside_mesh(point, b.triangles) for point in sample_a[:8])
    b_in_a = any(point_inside_mesh(point, a.triangles) for point in sample_b[:8])
    return {
        "surface_intersection_or_contact": False,
        "containment": bool(a_in_b or b_in_a),
        "containment_verified": True,
        "a_inside_b_sampled": bool(a_in_b),
        "b_inside_a_sampled": bool(b_in_a),
        "candidate_triangle_pairs_checked": intersection["candidate_triangle_pairs_checked"],
        "method": "BVH-pruned original STL triangles; odd/even ray test for containment",
    }


def self_test() -> None:
    overlap_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    crossing_b = np.array([[0.2, 0.2, -1], [0.2, 0.2, 1], [0.8, 0.2, 0]], dtype=float)
    separated_b = crossing_b + np.array([0.0, 2.0, 0.0])
    coplanar_b = np.array([[0.2, 0.2, 0], [0.8, 0.2, 0], [0.2, 0.8, 0]], dtype=float)
    disjoint_coplanar = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=float)
    assert triangles_intersect(overlap_a, crossing_b)
    assert not triangles_intersect(overlap_a, separated_b)
    assert triangles_intersect(overlap_a, coplanar_b)
    assert not triangles_intersect(overlap_a, disjoint_coplanar)
    a = TriangleBVH(np.stack([overlap_a]))
    b = TriangleBVH(np.stack([crossing_b]))
    assert a.first_intersection(b)["triangle_a"] == 0

    def box(low: np.ndarray, high: np.ndarray) -> np.ndarray:
        v = np.array([
            [low[0], low[1], low[2]], [high[0], low[1], low[2]],
            [high[0], high[1], low[2]], [low[0], high[1], low[2]],
            [low[0], low[1], high[2]], [high[0], low[1], high[2]],
            [high[0], high[1], high[2]], [low[0], high[1], high[2]],
        ])
        f = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ])
        return v[f]

    outer = TriangleBVH(box(np.array([0., 0., 0.]), np.array([2., 2., 2.])))
    inner = TriangleBVH(box(np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])))
    apart = TriangleBVH(box(np.array([3., 0., 0.]), np.array([4., 1., 1.])))
    contained = solid_overlap(inner, outer, closed_a=True, closed_b=True)
    separated = solid_overlap(outer, apart, closed_a=True, closed_b=True)
    open_mesh = solid_overlap(outer, apart, closed_a=False, closed_b=True)
    assert not contained["surface_intersection_or_contact"] and contained["containment"]
    assert not separated["surface_intersection_or_contact"] and not separated["containment"]
    assert not open_mesh["containment_verified"]
    print("mesh collision self-tests passed: crossing, separated, coplanar, containment, and open-mesh uncertainty")


def read_motion_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {"narrow": [], "wide": []}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["branch"] in result and abs(float(row["q_root_rel_rad"])) <= abs(LEGAL_CAD_Q_MIN) + 1e-10:
                result[row["branch"]].append(row)
    for side, rows in result.items():
        rows.sort(key=lambda row: abs(float(row["q_root_rel_rad"])))
        if len(rows) != 11 or not math.isclose(abs(float(rows[-1]["q_root_rel_rad"])), 0.10, abs_tol=1e-9):
            raise ValueError(f"expected 11 existing source samples from 0 to 0.10 rad for {side}, got {len(rows)}")
    return result


def joint_id(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"candidate joint missing: {name}")
    return jid, int(model.jnt_qposadr[jid])


def mesh_geom_triangles(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> tuple[np.ndarray, int]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"collision geom missing: {geom_name}")
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        raise ValueError(f"collision geom is not mesh-backed: {geom_name}")
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    local_vertices = model.mesh_vert[va:va + vn]
    faces = model.mesh_face[fa:fa + fn]
    triangles = local_vertices[faces].copy()
    rotation = data.geom_xmat[gid].reshape(3, 3)
    triangles = triangles @ rotation.T + data.geom_xpos[gid]
    return triangles, gid


def source_mesh_inventory() -> dict[str, dict[str, object]]:
    output = {}
    for name, expected in STL_SHA256.items():
        path = UPSTREAM / "meshes" / f"{name}.stl"
        if sha256(path) != expected:
            raise ValueError(f"official collision STL hash changed: {name}")
        _, _, meta = read_binary_stl(path)
        output[name] = meta
    return output


def static_preflight(corrected_model: Path, mesh_meta: dict[str, dict[str, object]]) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(corrected_model))
    data = mujoco.MjData(model)
    mesh_import_checks = {}
    for name, meta in mesh_meta.items():
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
        if mid < 0:
            raise ValueError(f"compiled model lacks source STL mesh: {name}")
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        face_count = int(model.mesh_facenum[mid])
        if face_count != meta["triangle_count"] or vn != meta["unique_vertex_count"]:
            raise ValueError(f"MuJoCo mesh topology differs from source STL: {name}")
        _, unique, _ = read_binary_stl(UPSTREAM / "meshes" / f"{name}.stl")
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.mesh_quat[mid])
        transformed = (unique - model.mesh_pos[mid]) @ rotation.reshape(3, 3)
        vertex_distance, _ = cKDTree(model.mesh_vert[va:va + vn]).query(transformed, k=1, workers=1)
        max_distance = float(vertex_distance.max()) if len(vertex_distance) else math.inf
        if max_distance > 1e-7:
            raise ValueError(f"source STL vertex frame does not match MuJoCo mesh import: {name}")
        mesh_import_checks[name] = {
            "source_triangles": int(meta["triangle_count"]),
            "compiled_triangles": face_count,
            "source_unique_vertices": int(meta["unique_vertex_count"]),
            "compiled_vertices": vn,
            "max_source_to_compiled_vertex_distance_m": max_distance,
        }
    rows = read_motion_rows(MOTION_CSV)
    root_ids = {side: joint_id(model, names["root"]) for side, names in JOINTS.items()}
    all_pair_rows = []
    sample_rows = []
    joint_limits_ok = True
    max_gap = 0.0
    max_rate = 0.0
    max_mimic = 0.0
    source_overlap = False
    containment_unknown = False
    hull_gross = False
    pair_geom_ids = {}
    for key, names in COLLISION_PAIRS.items():
        pair_geom_ids[key] = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)
        if min(pair_geom_ids[key]) < 0:
            raise ValueError(f"collision pair is missing from model: {key}")

    for index in range(11):
        data.qpos[:] = model.qpos0
        for side, side_rows in rows.items():
            row = side_rows[index]
            for coordinate in ("root", "B", "C", "E"):
                name = JOINTS[side][coordinate]
                jid, adr = joint_id(model, name)
                cad_value = float(row[{"root": "q_root_rel_rad", "B": "q_B_rel_rad", "C": "q_C_rel_rad", "E": "q_E_rel_rad"}[coordinate]])
                q_rel_urdf = -cad_value
                data.qpos[adr] = model.qpos0[adr] + q_rel_urdf
                if model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    joint_limits_ok = bool(joint_limits_ok and lo - 1e-10 <= data.qpos[adr] <= hi + 1e-10)
        mujoco.mj_forward(model, data)
        sample_q = -abs(float(rows["narrow"][index]["q_root_rel_rad"]))
        pair_site = {}
        for side in ("narrow", "wide"):
            j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"candidate_{side}_D_jaw")
            j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"candidate_{side}_D_loop")
            delta = data.site_xpos[j1] - data.site_xpos[j2]
            gap = float(np.linalg.norm(delta))
            max_gap = max(max_gap, gap)
            pair_site[side] = {"jaw_world_m": data.site_xpos[j1].tolist(), "loop_world_m": data.site_xpos[j2].tolist(), "delta_world_m": delta.tolist(), "gap_m": gap}
            rel = {}
            for coordinate in ("root", "B", "C", "E"):
                _, adr = joint_id(model, JOINTS[side][coordinate])
                rel[coordinate] = float(data.qpos[adr] - model.qpos0[adr])
            rate = rel["root"] + rel["B"] - rel["E"]
            max_rate = max(max_rate, abs(rate))
            pair_site[side].update({"q_urdf_relative_rad": rel, "q_root_plus_qB_minus_qE_rad": rate})
        qn, qw = root_ids["narrow"][1], root_ids["wide"][1]
        mimic = float(data.qpos[qn] + data.qpos[qw])
        max_mimic = max(max_mimic, abs(mimic))
        contacts = []
        for ci in range(data.ncon):
            contact = data.contact[ci]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            contacts.append({"geom1": n1, "geom2": n2, "dist_m": float(contact.dist), "penetration_m": max(0.0, -float(contact.dist))})
        contacts_by_geom = {(item["geom1"], item["geom2"]): item for item in contacts}
        contacts_by_geom.update({(item["geom2"], item["geom1"]): item for item in contacts})
        sample_rows.append({"sample": index, "q_narrow_cad_rad": sample_q, "narrow_wide_root_mimic_residual_rad": mimic, "branches": pair_site, "muJoCo_hull_contacts": contacts})

        for key, geom_names in COLLISION_PAIRS.items():
            a_tri, ga = mesh_geom_triangles(model, data, geom_names[0])
            b_tri, gb = mesh_geom_triangles(model, data, geom_names[1])
            mesh_a_id = int(model.geom_dataid[ga])
            mesh_b_id = int(model.geom_dataid[gb])
            mesh_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_a_id)
            mesh_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_b_id)
            source_result = solid_overlap(
                TriangleBVH(a_tri), TriangleBVH(b_tri),
                closed_a=bool(mesh_meta[mesh_a]["closed_oriented_two_manifold"]),
                closed_b=bool(mesh_meta[mesh_b]["closed_oriented_two_manifold"]),
            )
            source_overlap |= bool(source_result["surface_intersection_or_contact"] or source_result["containment"])
            containment_unknown |= not bool(source_result["containment_verified"])
            hull_contact = contacts_by_geom.get(geom_names)
            if hull_contact is None:
                dist = np.zeros(6, dtype=np.float64)
                distance = float(mujoco.mj_geomDistance(model, data, ga, gb, 1.0, dist))
            else:
                distance = float(hull_contact["dist_m"])
            penetration = max(0.0, -distance)
            hull_gross |= penetration > GROSS_PENETRATION_LIMIT_M
            all_pair_rows.append({
                "sample": index, "q_narrow_cad_rad": sample_q, "pair": key,
                "geom1": geom_names[0], "geom2": geom_names[1],
                "mujoco_convex_hull_signed_distance_m": distance,
                "mujoco_convex_hull_penetration_m": penetration,
                "source_stl_surface_intersection_or_contact": bool(source_result["surface_intersection_or_contact"]),
                "source_stl_containment": bool(source_result["containment"]),
                "source_stl_containment_verified": bool(source_result["containment_verified"]),
                "source_stl_containment_note": source_result.get("containment_note", ""),
                "source_triangle_hit": {k: source_result.get(k) for k in ("triangle_a", "triangle_b")},
                "triangle_pair_candidates_checked": source_result.get("candidate_triangle_pairs_checked"),
            })

    root_limit_result = bool(joint_limits_ok)
    constraints_static_ok = bool(max_gap <= STATIC_GAP_LIMIT_M and max_rate <= 1e-8 and max_mimic <= 1e-8)
    hull_rows_with_contacts = [row for row in all_pair_rows if row["mujoco_convex_hull_penetration_m"] > 0]
    proxy_artifact = any(
        row["mujoco_convex_hull_penetration_m"] > GROSS_PENETRATION_LIMIT_M
        and not row["source_stl_surface_intersection_or_contact"]
        and row["source_stl_containment_verified"]
        and not row["source_stl_containment"]
        for row in all_pair_rows
    )
    dynamics_ready = bool(root_limit_result and constraints_static_ok and not source_overlap and not containment_unknown and not hull_gross)
    return {
        "schema": SCHEMA,
        "status": "PASS_STATIC_READY_FOR_MOTION" if dynamics_ready else "BLOCKED_BEFORE_MOTION",
        "mujoco_version": mujoco.__version__,
        "source_candidate_sha256": sha256(SOURCE_MJCF),
        "corrected_candidate_sha256": sha256(corrected_model),
        "source_urdf_sha256": sha256(SOURCE_URDF),
        "source_urdf_pin": "AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d",
        "official_collision_stl": mesh_meta,
        "compiled_source_mesh_parity": mesh_import_checks,
        "joint_coordinate_map": "qpos_urdf = qpos0_urdf - q_candidate_CAD_rel; official axes and limits unchanged",
        "sampled_candidate_q_range_rad": [LEGAL_CAD_Q_MIN, 0.0],
        "sample_count_per_branch": 11,
        "source_samples": sample_rows,
        "collision_pairs": all_pair_rows,
        "collision_summary": {
            "source_mesh_interference_or_containment_detected": source_overlap,
            "source_mesh_containment_unverified_due_to_open_or_nonmanifold_stl": containment_unknown,
            "convex_hull_contact_present": bool(hull_rows_with_contacts),
            "gross_hull_penetration_over_2mm": hull_gross,
            "gross_hull_contact_not_present_in_source_triangles": proxy_artifact,
            "gross_hull_contact_count": sum(row["mujoco_convex_hull_penetration_m"] > GROSS_PENETRATION_LIMIT_M for row in all_pair_rows),
        },
        "gates": {
            "official_root_limits": root_limit_result,
            "all_corrected_D_gaps_at_or_below_1mm": max_gap <= STATIC_GAP_LIMIT_M,
            "candidate_rate_equalities_at_or_below_1e-8_rad": max_rate <= 1e-8,
            "official_root_mimic_at_or_below_1e-8_rad": max_mimic <= 1e-8,
            "no_source_mesh_intersection_or_containment": not source_overlap,
            "source_mesh_containment_check_verified": not containment_unknown,
            "no_gross_2mm_convex_hull_penetration": not hull_gross,
            "dynamics_ready": dynamics_ready,
        },
        "max_D_site_gap_m": max_gap,
        "max_candidate_rate_equality_residual_rad": max_rate,
        "max_official_root_mimic_residual_rad": max_mimic,
        "important_limit": "STATIC source-sample check only; no physical rollout in this command",
    }


def rows_to_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    from agent.path_governance import write_new_text
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_new_text(path, buffer.getvalue(), environment="wsl")


def run_preflight(run_dir: Path) -> dict[str, object]:
    from agent.path_governance import validate_output_path, write_new_bytes, write_new_text
    if not run_dir.exists():
        raise ValueError("pass the path helper-reserved --run-dir; this script does not invent output paths")
    location = validate_output_path(run_dir, require_exists=True, expected_kind="directory", environment="wsl")
    marker_path = run_dir / ".robotsim-output-reservation.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {"schema_version": "robotsim.output-reservation.v1", "path": location.wsl_path}:
        raise ValueError("run directory was not reserved by RobotSim path governance")
    unexpected = [p.name for p in run_dir.iterdir() if p.name not in {marker_path.name, "corrected_candidate_B.xml", "coordinate_overlay.diff"}]
    if unexpected:
        raise ValueError(f"refusing to write into run directory with later-stage artifacts: {unexpected}")
    if sha256(SOURCE_URDF) != "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d":
        raise ValueError("pinned official URDF source hash mismatch")
    source_bytes = SOURCE_MJCF.read_bytes()
    corrected_bytes, patch_text = create_corrected_xml(SOURCE_MJCF)
    model_path = run_dir / "corrected_candidate_B.xml"
    diff_path = run_dir / "coordinate_overlay.diff"
    if model_path.exists():
        if model_path.read_bytes() != corrected_bytes:
            raise ValueError("existing candidate copy differs from the approved D-site-only overlay")
    else:
        write_new_bytes(model_path, corrected_bytes, environment="wsl")
    if diff_path.exists():
        if diff_path.read_text(encoding="utf-8") != patch_text:
            raise ValueError("existing coordinate diff differs from the approved D-site-only overlay")
    else:
        write_new_text(diff_path, patch_text, environment="wsl")
    mesh_meta = source_mesh_inventory()
    result = static_preflight(model_path, mesh_meta)
    result["run_directory"] = location.as_dict()
    result["source_candidate_size_bytes"] = len(source_bytes)
    result["coordinate_overlay"] = D_SITE_MAP
    write_new_text(run_dir / "static_preflight.json", json.dumps(result, indent=2, sort_keys=True) + "\n", environment="wsl")
    collision_rows = result["collision_pairs"]
    rows_to_csv(run_dir / "legal_interval_collision.csv", collision_rows, [
        "sample", "q_narrow_cad_rad", "pair", "geom1", "geom2",
        "mujoco_convex_hull_signed_distance_m", "mujoco_convex_hull_penetration_m",
        "source_stl_surface_intersection_or_contact", "source_stl_containment",
        "source_stl_containment_verified", "source_stl_containment_note",
        "triangle_pair_candidates_checked",
    ])
    flat_samples = []
    for sample in result["source_samples"]:
        for side, branch in sample["branches"].items():
            flat_samples.append({
                "sample": sample["sample"], "q_narrow_cad_rad": sample["q_narrow_cad_rad"],
                "branch": side, "q_root_urdf_relative_rad": branch["q_urdf_relative_rad"]["root"],
                "q_B_urdf_relative_rad": branch["q_urdf_relative_rad"]["B"],
                "q_C_urdf_relative_rad": branch["q_urdf_relative_rad"]["C"],
                "q_E_urdf_relative_rad": branch["q_urdf_relative_rad"]["E"],
                "rate_residual_rad": branch["q_root_plus_qB_minus_qE_rad"],
                "D_gap_m": branch["gap_m"], "root_mimic_residual_rad": sample["narrow_wide_root_mimic_residual_rad"],
            })
    rows_to_csv(run_dir / "legal_interval_kinematics.csv", flat_samples, [
        "sample", "q_narrow_cad_rad", "branch", "q_root_urdf_relative_rad", "q_B_urdf_relative_rad",
        "q_C_urdf_relative_rad", "q_E_urdf_relative_rad", "rate_residual_rad", "D_gap_m", "root_mimic_residual_rad",
    ])
    return result


def kinematic_preview_html(result: dict[str, object]) -> str:
    samples = []
    for sample in result["source_samples"]:
        branches = {}
        for side, values in sample["branches"].items():
            branches[side] = {
                "root": values["q_urdf_relative_rad"]["root"],
                "B": values["q_urdf_relative_rad"]["B"],
                "C": values["q_urdf_relative_rad"]["C"],
                "E": values["q_urdf_relative_rad"]["E"],
                "D_gap_mm": values["gap_m"] * 1000,
            }
        samples.append({"q_cad_rad": sample["q_narrow_cad_rad"], "branches": branches})
    data = json.dumps(samples, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OmniPicker legal-range kinematic telemetry</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:24px;max-width:980px;color:#17212b;background:#f4f7f8}}
h1{{font-size:24px;margin:0 0 8px}}.badge{{color:#8b2b19;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin-top:16px;background:white}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d8e0e3}}
th{{background:#e6edef}}input{{width:100%}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{background:#e7ecee;padding:2px 4px}}
@media(max-width:650px){{.grid{{grid-template-columns:1fr}}body{{margin:14px}}}}
</style>
<body><div class="badge">NON-PHYSICAL KINEMATIC PREVIEW</div>
<h1>OmniPicker candidate B: source-sampled legal interval</h1>
<p>Static MuJoCo <code>mj_forward</code> samples only. This page does not show force-bearing motion, validate collision geometry, or certify a full opening/closing cycle.</p>
<label for="sample">Candidate CAD root q: <strong id="q"></strong></label><input id="sample" type="range" min="0" max="10" step="1" value="0">
<div class="grid"><section><h2>Narrow branch</h2><table><tbody id="narrow"></tbody></table></section>
<section><h2>Wide branch</h2><table><tbody id="wide"></tbody></table></section></div>
<script>
const samples={data};const slider=document.querySelector('#sample');
function render(){{const s=samples[Number(slider.value)];document.querySelector('#q').textContent=s.q_cad_rad.toFixed(3)+' rad';
for(const side of ['narrow','wide']){{const b=s.branches[side];document.querySelector('#'+side).innerHTML=
[['URDF root relative',b.root],['B passive',b.B],['C passive',b.C],['E passive',b.E],['D-site gap',b.D_gap_mm]].map(([k,v])=>'<tr><th>'+k+'</th><td>'+Number(v).toFixed(6)+(k==='D-site gap'?' mm':' rad')+'</td></tr>').join('')}}}}
slider.addEventListener('input',render);render();</script></body></html>
"""


def _joint_relative(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid, adr = joint_id(model, name)
    return float(data.qpos[adr] - model.qpos0[adr])


def _equality_values(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, dict[str, float]]:
    output = {}
    for name in ("candidate_narrow_D_closure", "candidate_wide_D_closure", "candidate_narrow_rate_eq", "candidate_wide_rate_eq", "source_wide_root_mimic"):
        eqid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        rows = np.flatnonzero(data.efc_id[:data.nefc] == eqid) if eqid >= 0 else np.array([], dtype=int)
        output[name] = {
            "max_abs_position_residual": float(np.max(np.abs(data.efc_pos[rows]))) if len(rows) else 0.0,
            "max_abs_constraint_force": float(np.max(np.abs(data.efc_force[rows]))) if len(rows) else 0.0,
            "rows": int(len(rows)),
        }
    return output


def _state_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    q = {side: {key: _joint_relative(model, data, name) for key, name in names.items()} for side, names in JOINTS.items()}
    sites = {}
    for side in ("narrow", "wide"):
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"candidate_{side}_D_jaw")
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"candidate_{side}_D_loop")
        sites[side] = float(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]))
    contacts = []
    for ci in range(data.ncon):
        contact = data.contact[ci]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        contacts.append({"geom1": geom1, "geom2": geom2, "dist_m": float(contact.dist), "penetration_m": max(0.0, -float(contact.dist))})
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)) and np.all(np.isfinite(data.qacc)))
    limits = True
    for side in ("narrow", "wide"):
        jid, adr = joint_id(model, JOINTS[side]["root"])
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            limits &= bool(lo - 1e-9 <= data.qpos[adr] <= hi + 1e-9)
    rate = {side: q[side]["root"] + q[side]["B"] - q[side]["E"] for side in ("narrow", "wide")}
    qn = model.jnt_qposadr[joint_id(model, JOINTS["narrow"]["root"])[0]]
    qw = model.jnt_qposadr[joint_id(model, JOINTS["wide"]["root"])[0]]
    mimic = float(data.qpos[qn] + data.qpos[qw])
    warnings = [int(warning.number) for warning in data.warning if int(warning.number) > 0]
    return {
        "finite_state": finite,
        "root_limits_ok": bool(limits),
        "joint_relative_rad": q,
        "candidate_rate_residual_rad": rate,
        "root_mimic_residual_rad": mimic,
        "D_site_gap_m": sites,
        "max_contact_penetration_m": max((item["penetration_m"] for item in contacts), default=0.0),
        "contacts": contacts,
        "equality_constraints": _equality_values(model, data),
        "max_abs_qvel_rad_s": float(np.max(np.abs(data.qvel))) if data.nv else 0.0,
        "warnings": warnings,
    }


def run_motion(run_dir: Path) -> dict[str, object]:
    preflight = json.loads((run_dir / "static_preflight.json").read_text(encoding="utf-8"))
    if not preflight.get("gates", {}).get("dynamics_ready"):
        raise ValueError("MuJoCo motion is prohibited because static preflight did not pass")
    model_path = run_dir / "corrected_candidate_B.xml"
    if sha256(model_path) != preflight["corrected_candidate_sha256"]:
        raise ValueError("corrected model changed after static preflight")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    root_id, root_adr = joint_id(model, JOINTS["narrow"]["root"])
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "candidate_root_servo_sim_only")
    if actuator < 0 or int(model.actuator_trnid[actuator, 0]) != root_id:
        raise ValueError("approved root servo is missing or targets another joint")
    # Match the source pose before the first integration step.
    data.ctrl[actuator] = float(model.qpos0[root_adr])
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    steps = int(round(1.0 / dt))
    ramp_steps = int(round(0.8 / dt))
    start = float(model.qpos0[root_adr])
    endpoint_cad = -0.10
    endpoint = start + abs(endpoint_cad)
    trace: list[dict[str, object]] = []
    frames = []
    renderer = None
    video_error = None
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception as exc:  # rendering is optional to physics validity
        video_error = f"renderer initialization failed: {type(exc).__name__}: {exc}"
    max_d_gap = 0.0
    max_penetration = 0.0
    max_rate = 0.0
    max_mimic = 0.0
    max_qvel = 0.0
    max_constraint_force = 0.0
    stop_reason = ""
    completed_steps = 0
    for step in range(steps):
        if step < ramp_steps:
            segment = min(9, step // max(1, ramp_steps // 10))
            local_start = segment / 10.0
            local_end = (segment + 1) / 10.0
            local_u = min(1.0, (step - segment * (ramp_steps // 10)) / max(1, ramp_steps // 10))
            smooth = local_u * local_u * (3.0 - 2.0 * local_u)
            fraction = local_start + (local_end - local_start) * smooth
        else:
            fraction = 1.0
        data.ctrl[actuator] = start + (endpoint - start) * fraction
        mujoco.mj_step(model, data)
        completed_steps = step + 1
        metrics = _state_metrics(model, data)
        max_d_gap = max(max_d_gap, *metrics["D_site_gap_m"].values())
        max_penetration = max(max_penetration, float(metrics["max_contact_penetration_m"]))
        max_rate = max(max_rate, *(abs(value) for value in metrics["candidate_rate_residual_rad"].values()))
        max_mimic = max(max_mimic, abs(float(metrics["root_mimic_residual_rad"])))
        max_qvel = max(max_qvel, float(metrics["max_abs_qvel_rad_s"]))
        max_constraint_force = max(max_constraint_force, *(
            float(row["max_abs_constraint_force"]) for row in metrics["equality_constraints"].values()
        ))
        if step % 10 == 0 or step == steps - 1:
            trace.append({"step": step + 1, "time_s": (step + 1) * dt, "commanded_root_qpos_rad": float(data.ctrl[actuator]), **metrics})
        if renderer is not None and step % 20 == 0:
            try:
                renderer.update_scene(data, camera="review_camera")
                frames.append(renderer.render().copy())
            except Exception as exc:
                video_error = f"renderer capture failed: {type(exc).__name__}: {exc}"
                renderer.close()
                renderer = None
        if not metrics["finite_state"]:
            stop_reason = "non-finite state"
        elif not metrics["root_limits_ok"]:
            stop_reason = "official root joint limit violation"
        elif metrics["warnings"]:
            stop_reason = f"MuJoCo warning counters: {metrics['warnings']}"
        elif float(metrics["max_contact_penetration_m"]) > GROSS_PENETRATION_LIMIT_M:
            stop_reason = "contact penetration exceeded the existing 2 mm stop gate"
        elif max(metrics["D_site_gap_m"].values()) > STATIC_GAP_LIMIT_M:
            stop_reason = "D-site closure gap exceeded 1 mm"
        elif max(abs(value) for value in metrics["candidate_rate_residual_rad"].values()) > 1e-4:
            stop_reason = "candidate rate constraint residual exceeded 1e-4 rad"
        elif abs(float(metrics["root_mimic_residual_rad"])) > 1e-8:
            stop_reason = "official root mimic residual exceeded 1e-8 rad"
        if stop_reason:
            break
    if renderer is not None:
        renderer.close()
    video_path = None
    if frames:
        try:
            import imageio.v2 as imageio
            from agent.path_governance import write_new_bytes
            with tempfile.TemporaryDirectory(prefix="robotsim-issue46-video-") as temp_dir:
                temporary_video = Path(temp_dir) / "candidate_B_legal_motion.mp4"
                writer = imageio.get_writer(str(temporary_video), fps=30, codec="libx264", macro_block_size=None)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                video_path = run_dir / "candidate_B_legal_motion.mp4"
                write_new_bytes(video_path, temporary_video.read_bytes(), environment="wsl")
        except Exception as exc:
            video_error = f"video encoding failed: {type(exc).__name__}: {exc}"
            video_path = None
    final_metrics = _state_metrics(model, data)
    completed_interval = completed_steps == steps and abs(float(data.qpos[root_adr]) - endpoint) < 0.01
    passed = bool(
        completed_interval and final_metrics["finite_state"] and final_metrics["root_limits_ok"]
        and not stop_reason and max_penetration <= GROSS_PENETRATION_LIMIT_M
        and max_d_gap <= STATIC_GAP_LIMIT_M and max_rate <= 1e-4 and max_mimic <= 1e-8
    )
    return {
        "status": "EXPERIMENTAL_MUJOCO_PASS" if passed else "EXPERIMENTAL_MUJOCO_FAIL",
        "mujoco_version": mujoco.__version__,
        "steps": completed_steps,
        "dt_s": dt,
        "simulated_time_s": completed_steps * dt,
        "command_start_root_qpos_rad": start,
        "command_endpoint_root_qpos_rad": endpoint,
        "commanded_candidate_cad_endpoint_rad": endpoint_cad,
        "actuator_name": "candidate_root_servo_sim_only",
        "actuator_kp": float(model.actuator_gainprm[actuator, 0]),
        "actuator_kv": -float(model.actuator_biasprm[actuator, 2]),
        "actuator_force_range": model.actuator_forcerange[actuator].tolist(),
        "max_D_gap_m": max_d_gap,
        "max_candidate_rate_residual_rad": max_rate,
        "max_root_mimic_residual_rad": max_mimic,
        "max_penetration_m": max_penetration,
        "max_abs_qvel_rad_s": max_qvel,
        "max_abs_equality_constraint_force": max_constraint_force,
        "final_state": final_metrics,
        "stop_reason": stop_reason,
        "video_file": video_path.name if video_path else None,
        "video_note": video_error,
        "qpos_writes_during_rollout": 0,
        "trace": trace,
    }


def render_report(result: dict[str, object], *, physical: dict[str, object] | None = None) -> str:
    lines = [
        "# Issue #46 — Corrected D-Site Legal-Range Validation",
        "",
        f"**Static result:** `{result['status']}`",
        f"**Physical MuJoCo motion:** `{physical['status'] if physical else 'NOT RUN'}`",
        "**Model class:** `UNVERIFIED_EXPERIMENTAL`; the candidate transmission remains a hypothesis.",
        "",
        "## Applied overlay",
        "",
        "The separate candidate copy changes only the four source-derived D-site positions from PR #84. The CAD-to-URDF coordinate map is applied in the analysis/command layer (`qpos_urdf = qpos0_urdf - q_candidate_CAD_rel`). Official URDF/STL files, root limits, joint axes, mimic, topology, actuator parameters, collision geoms, and equality relations remain unchanged.",
        "",
        "## Static interval",
        "",
        f"- Source samples: {result['sample_count_per_branch']} per branch, candidate q from {LEGAL_CAD_Q_MIN:.2f} to 0 rad.",
        f"- Maximum D-site gap from MuJoCo `mjData.site_xpos`: {result['max_D_site_gap_m'] * 1000:.6f} mm.",
        f"- Maximum candidate rate residual from actual `mjData.qpos - model.qpos0`: {result['max_candidate_rate_equality_residual_rad']:.3e} rad.",
        f"- Maximum official root mimic residual: {result['max_official_root_mimic_residual_rad']:.3e} rad.",
        f"- Official root limits across all sampled narrow/wide states: {result['gates']['official_root_limits']}.",
        "",
        "## Collision comparison",
        "",
        "The collision CSV compares MuJoCo convex-hull distances against intersection/containment checks on the original binary STL triangle surfaces. A hull contact is not classified as a real STL collision solely from its negative distance.",
        "",
    ]
    summary = result["collision_summary"]
    lines.extend([
        f"- Source STL surface intersection or containment in the four checked pairs: {summary['source_mesh_interference_or_containment_detected']}.",
        f"- Source STL containment check unavailable because some meshes are open/nonmanifold: {summary['source_mesh_containment_unverified_due_to_open_or_nonmanifold_stl']}.",
        f"- Gross convex-hull penetration over 2 mm: {summary['gross_hull_penetration_over_2mm']} ({summary['gross_hull_contact_count']} pair-samples).",
        f"- Gross hull contact absent from the corresponding original STL surfaces: {summary['gross_hull_contact_not_present_in_source_triangles']}.",
        "",
        "## Motion and limitations",
        "",
    ])
    if physical:
        lines.append(f"- MuJoCo 3.3.6 `mj_step` rollout: {physical['steps']} steps at {physical['dt_s']:.6f} s/step; status `{physical['status']}`.")
        lines.append(f"- Stop reason: {physical['stop_reason'] or 'bounded interval completed'}.")
        lines.append(f"- Maximum D gap: {physical['max_D_gap_m'] * 1000:.6f} mm; maximum penetration: {physical['max_penetration_m'] * 1000:.6f} mm.")
    else:
        lines.append("- No `mj_step` has run. Static qpos sampling uses `mj_forward` only; it does not establish force-bearing motion.")
    lines.extend([
        "- The interval ends at candidate q=-0.10 rad and is not a full opening/closing cycle. Full commercial-gripper travel and bottle-grasp readiness remain unverified.",
        "- Source STL hashes and the model/source hashes are in `static_preflight.json`; the exact four-site overlay is in `coordinate_overlay.diff`.",
        "",
        "Reproduction is recorded in the Issue #46 PR summary. This run's model and artifacts are isolated under the governed `reviews/issue-46/` directory.",
    ])
    return "\n".join(lines) + "\n"


def finalize_manifest(run_dir: Path, report: str) -> None:
    from agent.path_governance import write_new_text, write_artifact_manifest
    write_new_text(run_dir / "REPORT.md", report, environment="wsl")
    write_artifact_manifest(run_dir, environment="wsl")


def run_simulation_stage(run_dir: Path) -> dict[str, object]:
    from agent.path_governance import validate_output_path, write_new_text
    location = validate_output_path(run_dir, require_exists=True, expected_kind="directory", environment="wsl")
    marker = json.loads((run_dir / ".robotsim-output-reservation.json").read_text(encoding="utf-8"))
    if marker != {"schema_version": "robotsim.output-reservation.v1", "path": location.wsl_path}:
        raise ValueError("run directory reservation marker mismatch")
    result_path = run_dir / "static_preflight.json"
    if not result_path.is_file():
        raise ValueError("static preflight result is missing")
    preflight = json.loads(result_path.read_text(encoding="utf-8"))
    if not preflight.get("gates", {}).get("dynamics_ready"):
        raise ValueError("static preflight is not green; no mj_step run is permitted")
    if (run_dir / "artifact_paths.json").exists():
        raise ValueError("run already has a final artifact manifest")
    physical = run_motion(run_dir)
    write_new_text(run_dir / "physics_result.json", json.dumps({k: v for k, v in physical.items() if k != "trace"}, indent=2, sort_keys=True) + "\n", environment="wsl")
    write_new_text(run_dir / "physics_trace.json", json.dumps(physical["trace"], indent=2, sort_keys=True) + "\n", environment="wsl")
    if physical["status"] == "EXPERIMENTAL_MUJOCO_FAIL":
        kinematic_preview = kinematic_preview_html(preflight)
        write_new_text(run_dir / "kinematic_preview.html", kinematic_preview, environment="wsl")
    finalize_manifest(run_dir, render_report(preflight, physical=physical))
    return physical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--simulate", action="store_true", help="run one bounded mj_step rollout only after a passing static preflight")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.run_dir is None:
        parser.error("--run-dir must name a fresh path-governance-reserved reviews directory")
    if args.simulate:
        physical = run_simulation_stage(args.run_dir)
        print(json.dumps({k: v for k, v in physical.items() if k != "trace"}, indent=2))
        return 0 if physical["status"] == "EXPERIMENTAL_MUJOCO_PASS" else 1
    result = run_preflight(args.run_dir)
    print(json.dumps({"status": result["status"], "run_dir": result["run_directory"], "gates": result["gates"], "collision_summary": result["collision_summary"]}, indent=2))
    if not result["gates"]["dynamics_ready"]:
        from agent.path_governance import write_new_text
        write_new_text(args.run_dir / "kinematic_preview.html", kinematic_preview_html(result), environment="wsl")
        finalize_manifest(args.run_dir, render_report(result))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
