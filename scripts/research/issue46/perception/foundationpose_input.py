#!/usr/bin/env python3
"""Build a verified FoundationPose input fixture from one static X2 RGB-D capture.

The fixture contains RGB, millimetre depth, a mask, intrinsics, an exact
cylinder observation mesh, and an independently recorded MuJoCo truth pose.
It does not invoke FoundationPose or command a robot.  The mask source remains
explicit so a SIMULATION_ONLY truth-prompted smoke test cannot become a claim
that YOLO detected the object.
"""
from __future__ import annotations

import argparse
import hashlib
import io
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


def write_png(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def cylinder_obj(radius_m: float, half_height_m: float, *, segments: int = 64) -> bytes:
    """Return a centered, Z-axis-aligned closed cylinder OBJ in metres."""
    if not (math.isfinite(radius_m) and math.isfinite(half_height_m) and radius_m > 0.0 and half_height_m > 0.0):
        raise ValueError("cylinder dimensions must be finite and positive")
    if segments < 8:
        raise ValueError("cylinder mesh requires at least eight segments")
    lines = ["# SIMULATION_ONLY exact MuJoCo cylinder observation mesh; units: metres"]
    for z in (-half_height_m, half_height_m):
        for index in range(segments):
            theta = 2.0 * math.pi * index / segments
            lines.append(f"v {radius_m * math.cos(theta):.12g} {radius_m * math.sin(theta):.12g} {z:.12g}")
    bottom_center = 2 * segments + 1
    top_center = bottom_center + 1
    lines.extend((f"v 0 0 {-half_height_m:.12g}", f"v 0 0 {half_height_m:.12g}"))
    for index in range(segments):
        following = (index + 1) % segments
        bottom = index + 1
        next_bottom = following + 1
        top = segments + index + 1
        next_top = segments + following + 1
        lines.append(f"f {bottom} {next_bottom} {next_top} {top}")
        lines.append(f"f {bottom_center} {next_bottom} {bottom}")
        lines.append(f"f {top_center} {top} {next_top}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def camera_to_object_truth(capture: dict[str, object]) -> np.ndarray:
    camera_record = capture["camera"]
    # New captures explicitly use REP-103/OpenCV optical coordinates.  The
    # fallback preserves compatibility with an old fixture but must not be
    # used for a new model-comparison claim.
    camera_key = "T_world_optical_camera" if "T_world_optical_camera" in camera_record else "T_world_camera"
    camera = np.asarray(camera_record[camera_key], dtype=float)
    object_pose = np.asarray(capture["object_truth"]["T_world_object"], dtype=float)
    if camera.shape != (4, 4) or object_pose.shape != (4, 4):
        raise ValueError("capture truth matrices are invalid")
    return np.linalg.inv(camera) @ object_pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--mask-source", required=True, choices=("SAM2_TRUTH_BOX_SMOKE_TEST", "EXTERNAL_DETECTOR", "EXTERNAL_MASK"))
    parser.add_argument("--radius-m", required=True, type=float)
    parser.add_argument("--half-height-m", required=True, type=float)
    parser.add_argument("--frame-id", default="000000")
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve()
    metadata_path = capture_dir / "metadata.json"
    rgb_path = capture_dir / "rgb.png"
    depth_path = capture_dir / "depth_m.npy"
    if not all(path.is_file() for path in (metadata_path, rgb_path, depth_path, args.mask)):
        raise SystemExit("capture metadata/RGB/depth or requested mask is missing")
    capture = json.loads(metadata_path.read_text(encoding="utf-8"))
    gates = capture.get("gates", {})
    required = ("SOURCE_SCENE_HASH", "STATIC_RGBD_RENDER", "HEAD_AIMED_SIMULATION_CAMERA")
    if any(gates.get(gate) != "PASS" for gate in required):
        raise SystemExit("capture does not satisfy required static RGB-D gates")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth_m = np.load(depth_path)
    mask = np.asarray(Image.open(args.mask)) > 0
    if depth_m.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
        raise SystemExit("RGB, depth, and mask dimensions must match")
    if not np.isfinite(depth_m).all() or not np.any(depth_m > 0.0) or not mask.any():
        raise SystemExit("depth or mask is empty/non-finite")
    intrinsics = capture["camera"]["simulated_intrinsics_px"]
    if any(not math.isfinite(float(intrinsics[key])) for key in ("fx", "fy", "cx", "cy")):
        raise SystemExit("capture intrinsics are invalid")
    depth_mm = np.rint(np.clip(depth_m, 0.0, np.iinfo(np.uint16).max / 1000.0) * 1000.0).astype(np.uint16)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-foundationpose-input").native_path
    scene = output / "scene"
    for directory in (scene / "rgb", scene / "depth", scene / "masks", scene / "mesh"):
        directory.mkdir(parents=True, exist_ok=False)
    name = f"{args.frame_id}.png"
    pg.write_new_bytes(scene / "rgb" / name, write_png(rgb))
    pg.write_new_bytes(scene / "depth" / name, write_png(depth_mm))
    pg.write_new_bytes(scene / "masks" / name, write_png(mask.astype(np.uint8) * 255))
    camera_matrix = np.array(
        [[float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])], [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])], [0.0, 0.0, 1.0]]
    )
    pg.write_new_text(scene / "cam_K.txt", "\n".join(" ".join(f"{value:.12g}" for value in row) for row in camera_matrix) + "\n")
    mesh_path = scene / "mesh" / "test_cylinder.obj"
    pg.write_new_bytes(mesh_path, cylinder_obj(args.radius_m, args.half_height_m))
    record = {
        "schema": "robotsim.issue46.foundationpose-input.v1",
        "classification": "SIMULATION_ONLY_STATIC_PERCEPTION_FIXTURE",
        "mj_step_calls": 0,
        "capture": {"path": str(capture_dir), "metadata_sha256": sha256(metadata_path)},
        "mask": {"path": str(args.mask.resolve()), "sha256": sha256(args.mask), "source": args.mask_source, "pixel_count": int(mask.sum())},
        "object_mesh": {"path": str(mesh_path), "sha256": sha256(mesh_path), "primitive": "cylinder", "radius_m": args.radius_m, "half_height_m": args.half_height_m},
        "camera_intrinsics": intrinsics,
        "T_camera_object_truth": camera_to_object_truth(capture).tolist(),
        "gates": {"CAPTURE_STATIC_RGBD": "PASS", "MASK_DIMENSIONS": "PASS", "TRUTH_PROMPT_IS_DETECTION": "NOT_RUN" if args.mask_source == "SAM2_TRUTH_BOX_SMOKE_TEST" else "NOT_APPLICABLE"},
        "limitations": [
            "This fixture is a static simulation input; no FoundationPose inference or robot command occurred here.",
            "A SAM2 truth-box smoke-test mask validates segmentation mechanics, not YOLO detection quality.",
        ],
    }
    pg.write_new_text(output / "input_record.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "scene_wsl": str(scene), "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": record["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
