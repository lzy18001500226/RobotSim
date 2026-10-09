#!/usr/bin/env python3
"""Render source loop STLs against the convex hull used by a single MuJoCo mesh geom."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull


PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
MESHES = {
    "narrow_loop_Link": "63b895ff646b799c303c41c924ea4c77cf00f9c5d357e2fb4f6761c649237b17",
    "wide_loop_Link": "31aa42a993b3364e16efa657700ffe621eeaa5994d415bbe1b77f0687419b2b8",
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Not a binary STL: {path}")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * triangle_count:
        raise ValueError(f"Unexpected STL length: {path}")
    triangles = np.empty((triangle_count, 3, 3), dtype=np.float64)
    offset = 84
    for index in range(triangle_count):
        record = struct.unpack_from("<12fH", raw, offset)
        triangles[index] = np.asarray(record[3:12], dtype=np.float64).reshape(3, 3)
        offset += 50
    return triangles


def mesh_metrics(triangles: np.ndarray, path: Path) -> tuple[np.ndarray, dict]:
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    hull = ConvexHull(vertices)
    signed_volume = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ).sum() / 6.0
    edges: Counter[tuple[int, int]] = Counter()
    vertex_ids = {tuple(point): index for index, point in enumerate(vertices)}
    for face in triangles:
        ids = [vertex_ids[tuple(point)] for point in face]
        for start, end in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edges[tuple(sorted((start, end)))] += 1
    counts = Counter(edges.values())
    inside = hull.equations[:, :3] @ vertices.T + hull.equations[:, 3:]
    strict_inside = np.all(inside < -1e-9, axis=0)
    return vertices, {
        "mesh": path.name,
        "sha256": sha256(path),
        "triangle_count": int(len(triangles)),
        "unique_vertex_count": int(len(vertices)),
        "aabb_min_m": vertices.min(axis=0).tolist(),
        "aabb_max_m": vertices.max(axis=0).tolist(),
        "source_signed_volume_m3_unverified_if_nonmanifold": float(signed_volume),
        "convex_hull_volume_m3": float(hull.volume),
        "hull_to_source_signed_volume_ratio_unverified_if_nonmanifold": float(hull.volume / abs(signed_volume)) if signed_volume else None,
        "hull_strictly_inside_vertex_count": int(strict_inside.sum()),
        "hull_strictly_inside_vertex_fraction": float(strict_inside.mean()),
        "edge_incidence_counts": {str(key): value for key, value in sorted(counts.items())},
        "source_mesh_manifold_closed_by_edge_incidence": counts == Counter({2: len(edges)}),
        "hull_facet_count": int(len(hull.simplices)),
    },


def draw_projected_surface(
    draw: ImageDraw.ImageDraw,
    triangles: np.ndarray,
    vertices: np.ndarray,
    axis_pair: tuple[int, int],
    depth_axis: int,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    edge_color: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = rect
    points = vertices[:, axis_pair]
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1e-12)
    pad = 28
    scale = min((right - left - 2 * pad) / span[0], (bottom - top - 2 * pad) / span[1])
    offset = np.array([left + (right - left - scale * span[0]) / 2, top + (bottom - top - scale * span[1]) / 2])

    def pixel(points_2d: np.ndarray) -> list[tuple[int, int]]:
        mapped = (points_2d - low) * scale + offset
        mapped[:, 1] = top + bottom - mapped[:, 1]
        return [(int(round(x)), int(round(y))) for x, y in mapped]

    order = np.argsort(triangles[:, :, depth_axis].mean(axis=1))
    for index in order:
        polygon = pixel(triangles[index][:, axis_pair])
        draw.polygon(polygon, fill=color, outline=edge_color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    vendor = args.vendor_repo.resolve()
    output = args.output_dir.resolve()

    actual_pin = subprocess.check_output(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(vendor), "status", "--porcelain"], text=True
    ).strip()
    if actual_pin != PIN or dirty:
        raise SystemExit(f"Vendor checkout mismatch or dirty state: head={actual_pin}, dirty={bool(dirty)}")
    mesh_dir = vendor / "X2_URDF-v1.4.0" / "meshes"
    output.mkdir(parents=True, exist_ok=False)

    font = ImageFont.load_default()
    canvas = Image.new("RGBA", (1800, 1240), (250, 250, 249, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((30, 18), "Pinned X2 OmniPicker loop meshes vs single convex collision hull", fill=(25, 25, 25, 255), font=font)
    draw.text((30, 38), "Blue: original STL surface facets. Red: convex-hull surface facets. Orthographic projections only; not a physical-envelope validation.", fill=(55, 55, 55, 255), font=font)

    views = [("XY", (0, 1), 2), ("XZ", (0, 2), 1), ("YZ", (1, 2), 0)]
    results = {"vendor_commit": actual_pin, "vendor_dirty": bool(dirty), "meshes": {}}
    for row, (name, expected_hash) in enumerate(MESHES.items()):
        path = mesh_dir / f"{name}.stl"
        if sha256(path) != expected_hash:
            raise SystemExit(f"Mesh hash mismatch: {path}")
        triangles = read_binary_stl(path)
        vertices, metrics = mesh_metrics(triangles, path)
        hull = ConvexHull(vertices)
        hull_triangles = vertices[hull.simplices]
        results["meshes"][name] = metrics
        for col, (view, pair, depth) in enumerate(views):
            left = 20 + col * 590
            top = 75 + row * 570
            rect = (left + 12, top + 48, left + 568, top + 522)
            draw.rounded_rectangle(rect, radius=3, outline=(110, 110, 110, 255), width=1)
            draw.text((left + 15, top + 15), f"{name} | {view} projection", fill=(25, 25, 25, 255), font=font)
            draw.text((left + 15, top + 32), f"source facets={len(triangles)}   hull facets={len(hull.simplices)}", fill=(65, 65, 65, 255), font=font)
            draw_projected_surface(draw, hull_triangles, vertices, pair, depth, rect, (230, 85, 62, 52), (190, 45, 34, 150))
            draw_projected_surface(draw, triangles, vertices, pair, depth, rect, (35, 115, 190, 105), (20, 70, 130, 110))

    png_path = output / "loop_source_vs_convex_hull_projections.png"
    canvas.convert("RGB").save(png_path)
    (output / "loop_hull_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(png_path)
    print(output / "loop_hull_metrics.json")


if __name__ == "__main__":
    main()
