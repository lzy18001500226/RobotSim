#!/usr/bin/env python3
"""Refine a known upright-cylinder centre from RGB-D segmentation and a pose seed.

This is deliberately narrow: it assumes an upright cylinder of known height
and uses its visible top plane.  It is a geometric refinement after a 6-D
pose seed, not a replacement for general FoundationPose/AnyGrasp inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unproject_to_world(depth_m: np.ndarray, mask: np.ndarray, intrinsics: dict[str, float], t_world_optical: np.ndarray) -> np.ndarray:
    """Deproject masked metric depth into the known world frame."""
    if depth_m.ndim != 2 or mask.shape != depth_m.shape:
        raise ValueError("depth and mask dimensions must agree")
    if t_world_optical.shape != (4, 4) or not np.isfinite(t_world_optical).all():
        raise ValueError("T_world_optical is invalid")
    rows, columns = np.where(mask & np.isfinite(depth_m) & (depth_m > 0.0))
    if not len(rows):
        raise ValueError("masked depth has no valid points")
    for name in ("fx", "fy", "cx", "cy"):
        if name not in intrinsics or not math.isfinite(float(intrinsics[name])):
            raise ValueError("intrinsics are invalid")
    z = depth_m[rows, columns]
    optical = np.column_stack(
        (
            (columns - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
            (rows - float(intrinsics["cy"])) * z / float(intrinsics["fy"]),
            z,
        )
    )
    return optical @ t_world_optical[:3, :3].T + t_world_optical[:3, 3]


def refine_upright_cylinder(
    world_points: np.ndarray,
    seed_center_world: np.ndarray,
    *,
    half_height_m: float,
    top_band_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover centre from the horizontal top-plane support near a 6-D seed."""
    points = np.asarray(world_points, dtype=float)
    seed = np.asarray(seed_center_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or seed.shape != (3,):
        raise ValueError("points or seed have an invalid shape")
    if not np.isfinite(points).all() or not np.isfinite(seed).all() or half_height_m <= 0.0 or top_band_m <= 0.0:
        raise ValueError("points, seed, or cylinder dimensions are invalid")
    expected_top_z = seed[2] + half_height_m
    top = points[np.abs(points[:, 2] - expected_top_z) <= top_band_m]
    if len(top) < 20:
        raise ValueError("insufficient top-plane points near the pose seed")
    # The projection of a horizontal circular top is bounded by its left/right
    # and front/back rim.  Midpoints of those bounds give its XY centre even
    # though yaw remains unobservable for a cylinder.
    centre_xy = 0.5 * (top[:, :2].min(axis=0) + top[:, :2].max(axis=0))
    centre_z = float(np.median(top[:, 2]) - half_height_m)
    return np.array([centre_xy[0], centre_xy[1], centre_z]), top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--sam-mask", required=True, type=Path)
    parser.add_argument("--foundationpose-pose", required=True, type=Path)
    parser.add_argument("--half-height-m", required=True, type=float)
    parser.add_argument("--top-band-m", type=float, default=0.02)
    args = parser.parse_args()
    capture_dir = args.capture_dir.resolve()
    metadata_path = capture_dir / "metadata.json"
    depth_path = capture_dir / "depth_m.npy"
    if not all(path.is_file() for path in (metadata_path, depth_path, args.sam_mask, args.foundationpose_pose)):
        raise SystemExit("capture, SAM mask, or FoundationPose pose is missing")
    capture = json.loads(metadata_path.read_text(encoding="utf-8"))
    if capture.get("gates", {}).get("STATIC_RGBD_RENDER") != "PASS":
        raise SystemExit("capture has not passed the static RGB-D gate")
    camera = capture["camera"]
    if "T_world_optical_camera" not in camera:
        raise SystemExit("capture lacks an explicit optical-frame transform")
    depth_m = np.load(depth_path)
    mask = np.asarray(Image.open(args.sam_mask)) > 0
    pose_optical = np.loadtxt(args.foundationpose_pose).reshape(4, 4)
    t_world_optical = np.asarray(camera["T_world_optical_camera"], dtype=float)
    seed_world = (t_world_optical @ pose_optical)[:3, 3]
    points = unproject_to_world(depth_m, mask, camera["simulated_intrinsics_px"], t_world_optical)
    refined, top_points = refine_upright_cylinder(
        points,
        seed_world,
        half_height_m=args.half_height_m,
        top_band_m=args.top_band_m,
    )
    truth = np.asarray(capture["object_truth"]["world_position_m"], dtype=float)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-upright-cylinder-refine").native_path
    record = {
        "schema": "robotsim.issue46.upright-cylinder-refine.v1",
        "classification": "SIMULATION_ONLY_KNOWN_UPRIGHT_CYLINDER_GEOMETRIC_REFINEMENT",
        "mj_step_calls": 0,
        "inputs": {
            "capture_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "sam_mask": {"path": str(args.sam_mask.resolve()), "sha256": sha256(args.sam_mask)},
            "foundationpose_pose": {"path": str(args.foundationpose_pose.resolve()), "sha256": sha256(args.foundationpose_pose)},
        },
        "assumptions": {
            "object_geometry": "known upright cylinder",
            "half_height_m": args.half_height_m,
            "world_up_axis": "+Z",
            "top_band_m": args.top_band_m,
            "yaw_observable": False,
        },
        "foundationpose_seed_world_m": seed_world.tolist(),
        "refined_center_world_m": refined.tolist(),
        "top_plane_point_count": int(len(top_points)),
        "simulation_truth_center_world_m": truth.tolist(),
        "metrics": {
            "foundationpose_seed_translation_error_mm": float(np.linalg.norm(seed_world - truth) * 1000.0),
            "refined_translation_error_mm": float(np.linalg.norm(refined - truth) * 1000.0),
        },
        "gates": {
            "SAM_MASK_DEPTH_POINTS": "PASS",
            "FOUNDATIONPOSE_SEED": "PASS",
            "VISIBLE_UPRIGHT_TOP_PLANE": "PASS",
            "GEOMETRIC_CENTRE_REFINEMENT": "PASS",
            "GENERAL_OBJECT_POSE": "NOT_APPLICABLE",
            "YOLO_DETECTION": "NOT_RUN",
            "WHOLE_BODY_IK": "NOT_RUN",
        },
        "limitations": [
            "Only valid for a known upright cylinder with visible top-plane support.",
            "This uses a SIMULATION_ONLY truth-box-prompted SAM2 mask in the current fixture.",
            "The refined centre does not by itself authorize robot motion or physical grasping.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "metrics": record["metrics"], "gates": record["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
