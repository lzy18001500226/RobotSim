#!/usr/bin/env python3
"""Create a read-only source-STL contact-envelope review for OmniPicker.

The pinned vendor STL files carry geometry but no material/part labels for the
black rubber in the product render.  This tool therefore does *not* claim to
recover an official rubber-material boundary.  It maps the two existing
inner-face planning references onto the exact source meshes, highlights small
geometric patches around them, and derives their midpoint as the proposed
upright-cylinder axis for human review.

No source model is edited and no ``mj_step`` call occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


MODEL_ID = "X2-OmniPicker-STL"
PATCH_RADIUS_M = 0.008
FRAME_CONTRACT_FILE = "grasp_frame_contract.json"
# These are the existing inner-face references used by centered_orientation_search.py.
# They are planning references, not vendor-labelled rubber material centroids.
INNER_FACE_REFERENCE_LOCAL_M = {
    "R_hand_narrow3_Link": np.array((0.0380, -0.0207, 0.0)),
    "R_hand_wide3_Link": np.array((0.0375, 0.0206, 0.0)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def closest_point_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on a triangle, Ericson's region test."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denominator = 1.0 / (va + vb + vc)
    return a + ab * (vb * denominator) + ac * (vc * denominator)


def mesh_arrays(model: mujoco.MjModel, mesh: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]],
        dtype=float,
    )
    faces = np.asarray(
        model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]],
        dtype=int,
    )
    return vertices, faces


def mesh_rotation(model: mujoco.MjModel, mesh: int) -> np.ndarray:
    value = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(value, model.mesh_quat[mesh])
    return value.reshape(3, 3)


def patch_for_reference(model: mujoco.MjModel, geom: int, reference_source_stl: np.ndarray) -> dict[str, object]:
    mesh = int(model.geom_dataid[geom])
    vertices, faces = mesh_arrays(model, mesh)
    # MuJoCo compiles STL assets into a centered/principal mesh frame.  Convert
    # the inherited source-STL reference into that frame before nearest-surface
    # measurement; it is not valid to compare source-STL coordinates directly
    # with ``model.mesh_vert``.
    rotation = mesh_rotation(model, mesh)
    reference = (reference_source_stl - model.mesh_pos[mesh]) @ rotation
    triangles = vertices[faces]
    centroids = triangles.mean(axis=1)
    candidates = np.argsort(np.linalg.norm(centroids - reference, axis=1))[:512]
    points = [closest_point_triangle(reference, *triangles[index]) for index in candidates]
    distances = [float(np.linalg.norm(point - reference)) for point in points]
    closest = points[int(np.argmin(distances))]
    selected = np.flatnonzero(np.linalg.norm(centroids - closest, axis=1) <= PATCH_RADIUS_M)
    if len(selected) == 0:
        raise RuntimeError("inner-face geometric patch is unexpectedly empty")
    return {
        "geom": geom,
        "mesh": mesh,
        "reference_source_stl_m": reference_source_stl.tolist(),
        "reference_mesh_local_m": reference.tolist(),
        "nearest_mesh_surface_local_m": closest.tolist(),
        "reference_to_surface_distance_mm": min(distances) * 1000.0,
        "patch_radius_mm": PATCH_RADIUS_M * 1000.0,
        "face_count": int(len(selected)),
        "faces": faces[selected].ravel().tolist(),
    }


def visual_geom(model: mujoco.MjModel, body_name: str) -> int:
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if body == body_name and "__visual_" in name:
            return geom
    raise RuntimeError(f"missing source visual mesh for {body_name}")


def selected_visual_geoms(model: mujoco.MjModel) -> list[int]:
    result = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if "__visual_" in name:
            result.append(geom)
    return result


def world_point(data: mujoco.MjData, geom: int, local: list[float]) -> np.ndarray:
    return data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ np.asarray(local, dtype=float)


def main() -> None:
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    frame_contract_path = loaded.package_path / FRAME_CONTRACT_FILE
    frame_contract = json.loads(frame_contract_path.read_text(encoding="utf-8"))
    if (
        frame_contract.get("model_id") != MODEL_ID
        or frame_contract.get("status") != "HUMAN_APPROVED_VISUAL_ORIENTATION_ONLY"
        or frame_contract.get("cylinder", {}).get("axis_unit_vector_in_gripper_frame") != [1.0, 0.0, 0.0]
        or frame_contract.get("jaw_closure", {}).get("direction_unit_vector_in_gripper_frame") != [0.0, 1.0, 0.0]
    ):
        raise RuntimeError("frozen STL grasp-frame orientation contract is missing or incompatible")
    trajectory = load_registered_trajectory(loaded.package_path)
    model = loaded.model
    geoms = selected_visual_geoms(model)
    body_ids = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in INNER_FACE_REFERENCE_LOCAL_M
    }
    if min(body_ids.values()) < 0:
        raise RuntimeError("selected STL model lacks a Link3 body")
    patches = {
        name: patch_for_reference(model, visual_geom(model, name), reference)
        for name, reference in INNER_FACE_REFERENCE_LOCAL_M.items()
    }

    meshes: dict[str, object] = {}
    geometry_records = []
    for geom in geoms:
        mesh = int(model.geom_dataid[geom])
        if str(mesh) not in meshes:
            vertices, faces = mesh_arrays(model, mesh)
            meshes[str(mesh)] = {
                "vertices": vertices.astype(np.float32).ravel().tolist(),
                "faces": faces.astype(np.int32).ravel().tolist(),
            }
        geometry_records.append(
            {
                "id": geom,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "unnamed",
                "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "world",
                "mesh": mesh,
                "rgba": [float(value) for value in model.geom_rgba[geom]],
            }
        )

    poses = []
    for command in range(101):
        data = mujoco.MjData(model)
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command))
        points = {
            name: world_point(data, int(patch["geom"]), patch["nearest_mesh_surface_local_m"])
            for name, patch in patches.items()
        }
        narrow = points["R_hand_narrow3_Link"]
        wide = points["R_hand_wide3_Link"]
        midpoint = (narrow + wide) * 0.5
        gap = float(np.linalg.norm(wide - narrow))
        poses.append(
            {
                "position": {str(geom): [float(value) for value in data.geom_xpos[geom]] for geom in geoms},
                "rotation": {str(geom): [float(value) for value in data.geom_xmat[geom]] for geom in geoms},
                "surface_points_world_m": {name: point.tolist() for name, point in points.items()},
                "midpoint_axis_world_m": midpoint.tolist(),
                "surface_reference_gap_mm": gap * 1000.0,
                "surface_height_delta_mm": float((wide[2] - narrow[2]) * 1000.0),
            }
        )

    source_assets = json.loads((loaded.package_path / "source_assets.json").read_text(encoding="utf-8"))
    data = {
        "schema": "robotsim.issue46.stl-contact-envelope-review.v1",
        "classification": "SOURCE_STL_GEOMETRY_REFERENCE_REVIEW_ONLY",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "grasp_frame_contract": {"path": str(frame_contract_path), "sha256": sha256(frame_contract_path), "value": frame_contract},
        "source_assets": source_assets["required_assets"],
        "commands": list(range(101)),
        "meshes": meshes,
        "geoms": geometry_records,
        "patches": patches,
        "poses": poses,
        "symmetric_midpoint_definition": {
            "name": "midpoint_between_source_surface_references",
            "formula": "0.5 * (narrow3 nearest source-surface reference + wide3 nearest source-surface reference)",
            "meaning": "A symmetric location between the two reviewed source-STL surface references; it does not choose an object axis or object orientation.",
        },
        "limitations": [
            "The binary source STL has no material, colour, or black-rubber submesh identifier.",
            "The highlighted patches are a reproducible 8 mm geometric neighbourhood of existing IK inner-face references, not a claim of vendor-labelled rubber boundaries.",
            "No circular-arc centre is asserted: local circle fits are not source-defined and are sensitive to the chosen patch extent.",
            "The midpoint is a symmetric geometric reference only. It does not choose an object axis, object orientation, contact force, or a physical grasp.",
        ],
    }
    template = Path(__file__).with_name("stl_contact_envelope_review.html")
    if not template.is_file():
        raise FileNotFoundError(f"missing review template: {template}")
    output = pg.create_unique_output_directory(EVIDENCE / "reviews/issue-46", prefix="stl-contact-envelope-web-review").native_path
    scene_json = json.dumps(data, separators=(",", ":"))
    page = template.read_text(encoding="utf-8")
    marker = "/*__SCENE_DATA__*/"
    if page.count(marker) != 1:
        raise RuntimeError("review template lacks the unique embedded-data marker")
    # A single-file review opens correctly from Codex's local-file preview as
    # well as from an HTTP server; do not require fetch('scene.json').
    page = page.replace(marker, "window.SCENE_DATA=" + scene_json + ";")
    pg.write_new_text(output / "scene.json", scene_json)
    pg.write_new_text(output / "index.html", page)
    review = {
        "schema": data["schema"],
        "classification": data["classification"],
        "mj_step_calls": 0,
        "model": data["model"],
        "grasp_frame_contract": data["grasp_frame_contract"],
        "patches": patches,
        "reference_gap_mm": {str(command): poses[command]["surface_reference_gap_mm"] for command in (0, 32, 38, 50, 100)},
        "limitations": data["limitations"],
    }
    pg.write_new_text(output / "review.json", json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "reference_gap_mm": review["reference_gap_mm"], "patch_face_counts": {name: patch["face_count"] for name, patch in patches.items()}, "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
