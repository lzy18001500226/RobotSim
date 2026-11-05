#!/usr/bin/env python3
"""Create a human selector for possible OmniPicker textured contact regions.

The official Link3 STL meshes have no material groups for the black rubber.
This page deliberately exposes several reproducible geometric candidates rather
than silently deciding which one is the contact surface.  It runs only
``mj_forward`` to animate the already frozen STL joint table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
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
SOURCE_MESH_ROOT = EVIDENCE / "reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/meshes"
FRAME_CONTRACT_FILE = "grasp_frame_contract.json"
RULES = {
    "R_hand_narrow3_Link": {"mesh_file": "narrow3_Link.stl", "threshold_m": -0.017, "sense": "lt"},
    "R_hand_wide3_Link": {"mesh_file": "wide3_Link.stl", "threshold_m": 0.017, "sense": "gt"},
}
DENSE_TRIANGLE_AREA_M2 = 0.1e-6
MIN_COMPONENT_TRIANGLES = 50
REVIEW_SEED_FACES = {"R_hand_narrow3_Link": 4406, "R_hand_wide3_Link": 16751}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_triangles(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise RuntimeError(f"unexpected binary STL size: {path}")
    record = np.frombuffer(raw, dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84, count=count)
    return np.asarray(record["vertices"], dtype=float)


def mesh_rotation(model: mujoco.MjModel, mesh: int) -> np.ndarray:
    values = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(values, model.mesh_quat[mesh])
    return values.reshape(3, 3)


def visual_geom(model: mujoco.MjModel, body: str) -> int:
    for geom in range(model.ngeom):
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if parent == body and "__visual_" in name:
            return geom
    raise RuntimeError(f"missing visual mesh for {body}")


def mesh_candidate(vertices: np.ndarray, mask: np.ndarray, label: str, description: str) -> dict[str, object]:
    selected = vertices[mask]
    if len(selected) == 0:
        raise RuntimeError(f"empty source geometry candidate: {label}")
    return {
        "label": label,
        "description": description,
        "triangle_count": int(len(selected)),
        "vertices": selected.reshape(-1, 3).astype(np.float32).ravel().tolist(),
        "faces": np.arange(len(selected) * 3, dtype=np.int32).reshape(-1, 3).ravel().tolist(),
        "mesh_local_bbox_mm": {"min": (selected.reshape(-1, 3).min(axis=0) * 1000.0).tolist(), "max": (selected.reshape(-1, 3).max(axis=0) * 1000.0).tolist()},
        # Kept temporarily for the browser template's explanatory field; both
        # entries describe the same MuJoCo mesh-local coordinates.
        "raw_bbox_mm": {"min": (selected.reshape(-1, 3).min(axis=0) * 1000.0).tolist(), "max": (selected.reshape(-1, 3).max(axis=0) * 1000.0).tolist()},
    }


def connected_components(triangles: np.ndarray) -> np.ndarray:
    """Label triangle components by exact shared source-STL vertices."""
    keys = np.round(triangles.reshape(-1, 3) * 1e7).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    parent = np.arange(len(faces), dtype=int)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    first_face: dict[int, int] = {}
    for face, vertices in enumerate(faces):
        for vertex in vertices:
            value = int(vertex)
            if value in first_face:
                union(face, first_face[value])
            else:
                first_face[value] = face
    roots = np.asarray([find(index) for index in range(len(faces))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def pad_candidates(model: mujoco.MjModel, body: str) -> dict[str, object]:
    rule = RULES[body]
    source_path = SOURCE_MESH_ROOT / str(rule["mesh_file"])
    raw = raw_triangles(source_path)
    centroid = raw.mean(axis=1)
    area = np.linalg.norm(np.cross(raw[:, 1] - raw[:, 0], raw[:, 2] - raw[:, 0]), axis=1) * 0.5
    inner = centroid[:, 1] < float(rule["threshold_m"]) if rule["sense"] == "lt" else centroid[:, 1] > float(rule["threshold_m"])
    inner &= centroid[:, 0] >= 0.018
    geom = visual_geom(model, body)
    mesh = int(model.geom_dataid[geom])
    converted = (raw - model.mesh_pos[mesh]) @ mesh_rotation(model, mesh)
    # A four-incident-face seam in this STL joins separately tessellated
    # surfaces.  Traverse only two-face edges to avoid crossing into the
    # supporting bracket, starting from the previously human-picked pad face.
    keys = np.round(raw.reshape(-1, 3) * 1e7).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges: dict[tuple[int, int], list[int]] = {}
    for face, vertices in enumerate(faces):
        for first, second in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
            edges.setdefault(tuple(sorted((int(first), int(second)))), []).append(face)
    parent = np.arange(len(raw), dtype=int)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for incident in edges.values():
        if len(incident) == 2:
            parent[root(incident[0])] = root(incident[1])
    shell_labels = np.asarray([root(index) for index in range(len(raw))])
    seed = REVIEW_SEED_FACES[body]
    indices = model.mesh_face[model.mesh_faceadr[mesh] + seed]
    compiled_seed = model.mesh_vert[model.mesh_vertadr[mesh] + indices]
    if not np.allclose(converted[seed], compiled_seed, atol=1e-8, rtol=0):
        raise RuntimeError(f"source/compiled seed triangle mismatch for {body}")
    shell = shell_labels == shell_labels[seed]
    shell_points = raw[shell].reshape(-1, 3)
    low, high = shell_points.min(axis=0) - 1e-6, shell_points.max(axis=0) + 1e-6
    # Include isolated texture and cap fragments contained within this seed
    # shell's envelope.  This selection changes display only, never geometry.
    complete = np.zeros(len(raw), dtype=bool)
    for label in np.unique(shell_labels):
        mask = shell_labels == label
        points = raw[mask].reshape(-1, 3)
        if np.all(points.min(axis=0) >= low) and np.all(points.max(axis=0) <= high):
            complete |= mask
    dense = inner & (area <= DENSE_TRIANGLE_AREA_M2)
    dense_indices = np.flatnonzero(dense)
    labels = connected_components(raw[dense])
    ordered = sorted(
        ((component, int(np.sum(labels == component))) for component in np.unique(labels)),
        key=lambda item: item[1],
        reverse=True,
    )
    candidates = {
        "rubber_shell_review": mesh_candidate(converted, complete, "完整包胶块高亮（待视觉审核）", "从人工种子沿双面边邻接扩展，以多面接缝分区；补入壳包围范围内的纹理/封口碎片。包含下部、侧面和封口候选面，尚未认定材质或闭合实体。"),
        "dense_all_fragments": mesh_candidate(converted, dense, "全部内侧纹理高密度面（含无关碎片）", "对照项：包含所有小三角形纹理碎片，通常过宽，不应默认采用。"),
        "inner_surface_all": mesh_candidate(converted, inner, "全部内侧几何面（含非包胶区域）", "对照项：包含所有满足内侧坐标阈值的面，通常过宽，不应默认采用。"),
    }
    major_masks = []
    for rank, (component, count) in enumerate(ordered, start=1):
        if count < MIN_COMPONENT_TRIANGLES:
            break
        mask = np.zeros(len(raw), dtype=bool)
        mask[dense_indices[labels == component]] = True
        major_masks.append(mask)
        candidates[f"dense_component_{rank}"] = mesh_candidate(
            converted,
            mask,
            f"连续纹理面组件 {rank}（{count} 个三角形）",
            "通过 STL 三角面共享顶点得到的单一连通纹理面；候选面不会跨越真实网格边界。",
        )
    if len(major_masks) >= 2:
        candidates["dense_component_1_2_union"] = mesh_candidate(
            converted,
            major_masks[0] | major_masks[1],
            "连续纹理面组件 1+2 联合",
            "仅在两个最大连通纹理组件视觉上属于同一包胶弧面时才选择。",
        )
    return {
        "body": body,
        "geom": geom,
        "source_mesh": str(source_path),
        "source_mesh_sha256": sha256(source_path),
        "inner_rule": {**rule, "min_x_m": 0.018, "dense_triangle_area_mm2_le": DENSE_TRIANGLE_AREA_M2 * 1e6, "component_min_triangles": MIN_COMPONENT_TRIANGLES},
        "shell_review": {"seed_source_face": seed, "primary_skin_triangles": int(np.sum(shell)), "complete_highlight_triangles": int(np.sum(complete)), "source_face_indices": np.flatnonzero(complete).tolist(), "status": "PENDING_HUMAN_VISUAL_REVIEW", "watertight_certified": False},
        "candidates": candidates,
    }


def all_visual_geoms(model: mujoco.MjModel) -> list[int]:
    return [geom for geom in range(model.ngeom) if "__visual_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")]


def main() -> None:
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    contract_path = loaded.package_path / FRAME_CONTRACT_FILE
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("cylinder", {}).get("axis_unit_vector_in_gripper_frame") != [1.0, 0.0, 0.0]:
        raise RuntimeError("frozen orientation contract is not +X")
    model = loaded.model
    trajectory = load_registered_trajectory(loaded.package_path)
    pads = {body: pad_candidates(model, body) for body in RULES}
    geoms = []
    meshes: dict[str, object] = {}
    visual = all_visual_geoms(model)
    for geom in visual:
        mesh = int(model.geom_dataid[geom])
        if str(mesh) not in meshes:
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            face_start, face_count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            meshes[str(mesh)] = {"vertices": model.mesh_vert[start:start + count].astype(np.float32).ravel().tolist(), "faces": model.mesh_face[face_start:face_start + face_count].astype(np.int32).ravel().tolist()}
        geoms.append({"id": geom, "mesh": mesh, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "unnamed", "rgba": [float(v) for v in model.geom_rgba[geom]]})
    poses = []
    for command in range(101):
        data = mujoco.MjData(model)
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command))
        poses.append({"position": {str(geom): [float(v) for v in data.geom_xpos[geom]] for geom in visual}, "rotation": {str(geom): [float(v) for v in data.geom_xmat[geom]] for geom in visual}})
    payload = {
        "schema": "robotsim.issue46.stl-pad-region-selector.v1",
        "classification": "HUMAN_PAD_REGION_SELECTION_ONLY",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "grasp_frame_contract": {"path": str(contract_path), "sha256": sha256(contract_path), "value": contract},
        "geoms": geoms,
        "meshes": meshes,
        "pads": pads,
        "poses": poses,
        "selection_instruction": "Review the automatically highlighted full pad candidates, including their lower, side and capped surfaces. No dropdown selection, circle fitting or grasp is required.",
    }
    template = Path(__file__).with_name("stl_pad_region_selector.html")
    output = pg.create_unique_output_directory(EVIDENCE / "reviews/issue-46", prefix="stl-pad-region-selector").native_path
    scene = json.dumps(payload, separators=(",", ":"))
    page = template.read_text(encoding="utf-8")
    if page.count("/*__SCENE_DATA__*/") != 1:
        raise RuntimeError("selector template lacks embedded scene marker")
    pg.write_new_text(output / "scene.json", scene)
    pg.write_new_text(output / "index.html", page.replace("/*__SCENE_DATA__*/", "window.SCENE_DATA=" + scene + ";"))
    result = {"schema": payload["schema"], "classification": payload["classification"], "mj_step_calls": 0, "model": payload["model"], "candidates": {body: {key: {"label": item["label"], "triangle_count": item["triangle_count"], "mesh_local_bbox_mm": item["mesh_local_bbox_mm"]} for key, item in pad["candidates"].items()} for body, pad in pads.items()}}
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "candidates": result["candidates"], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
