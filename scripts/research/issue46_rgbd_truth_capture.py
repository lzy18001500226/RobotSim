#!/usr/bin/env python3
"""Capture RGB-D and object-pose truth from an X2 MuJoCo scene.

This source-preserving utility only adds a temporary rendering camera and
performs ``mj_forward``.  The official ``rgbd_head_front`` URDF joint gives a
mounting transform, but it does *not* by itself establish an optical frame.
The default mode is consequently an explicit ``SIMULATION_ONLY`` head-mounted
look-at camera for offline perception-contract tests.  The original mounting
transform remains available as a diagnostic mode and is never mislabelled as
vendor RGB-D calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


CAMERA_NAME = "rgbd_head_front_truth"
CAMERA_PARENT = "head_pitch_link"
# Official RAICOM x2_whole_body.urdf joint rgbd_head_front, expressed in the
# parent head_pitch_link frame.  The optical-axis convention is kept explicit
# in metadata and must be validated against the live vendor stream later.
CAMERA_POS = "0.05761 -0.011183 -0.04837"
CAMERA_EULER = "2.2689 0 1.5708"
# MuJoCo renderer camera: +X right, +Y up, -Z forward.
# REP-103/OpenCV optical: +X right, +Y down, +Z forward.
MUJOCO_CAMERA_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def normalized(vector: np.ndarray, *, label: str) -> np.ndarray:
    """Return a finite unit vector or reject a degenerate geometric request."""
    magnitude = float(np.linalg.norm(vector))
    if not math.isfinite(magnitude) or magnitude <= 1e-9:
        raise ValueError(f"{label} is degenerate")
    return np.asarray(vector, dtype=float) / magnitude


def look_at_rotation(camera_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera local-to-world rotation with local ``-Z`` at target.

    MuJoCo's rendering camera looks along local ``-Z``.  Its local ``+X`` is
    image-right and local ``+Y`` is image-up, so the returned three columns are
    right, up, and backwards respectively.
    """
    forward = normalized(target_position - camera_position, label="camera-to-target vector")
    backwards = -forward
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference_up, backwards))) > 0.98:
        reference_up = np.array([0.0, 1.0, 0.0])
    right = normalized(np.cross(reference_up, backwards), label="camera right axis")
    up = normalized(np.cross(backwards, right), label="camera up axis")
    rotation = np.column_stack((right, up, backwards))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("camera look-at rotation is not right-handed")
    return rotation


def quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a right-handed rotation matrix to MuJoCo's scalar-first quaternion."""
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("rotation must be right-handed")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        values = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            values = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            values = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            values = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalized(values, label="camera quaternion")


def pose_in_parent(
    parent_rotation_world: np.ndarray,
    parent_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    camera_position_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a desired world camera pose in the parent body's local frame."""
    parent_rotation_world = np.asarray(parent_rotation_world, dtype=float).reshape(3, 3)
    local_rotation = parent_rotation_world.T @ camera_rotation_world
    local_position = parent_rotation_world.T @ (camera_position_world - parent_position_world)
    return local_rotation, local_position


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"scene lacks body: {name}")


def rebind_missing_workspace_assets(
    root: ET.Element,
    *,
    canonical_reference: Path | None = None,
    canonical_root: Path | None = None,
) -> list[dict[str, str]]:
    """Rebind only missing immutable source/proxy paths in an evidence copy.

    Historical review XML files may retain an absolute path to a retired
    worktree.  The source/proxy identity is encoded after one of the two
    accepted roots below.  Replacing only that missing prefix with the
    canonical checkout is an execution-path repair, not a geometry edit.
    Any other missing path deliberately remains unresolved and fails closed.
    """
    # The active task worktree owns generated collision proxies; its reference
    # symlink points at the canonical immutable vendor checkout.  ``reviews``
    # owns evidence only and must not be used as an asset root.
    canonical_reference = canonical_reference or REPO / "reference"
    canonical_root = canonical_root or REPO
    if not canonical_reference.is_dir():
        raise RuntimeError(f"canonical source root is missing: {canonical_reference}")
    rebound: list[dict[str, str]] = []
    for element in root.iter():
        raw_path = element.get("file")
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.is_file():
            continue
        if not candidate.is_absolute():
            continue
        if "/reference/" in raw_path:
            replacement = canonical_reference / raw_path.split("/reference/", 1)[1]
        elif "/simulation/" in raw_path:
            replacement = canonical_root / "simulation" / raw_path.split("/simulation/", 1)[1]
        else:
            continue
        if not replacement.is_file():
            continue
        element.set("file", str(replacement))
        rebound.append({"element": element.tag, "from": raw_path, "to": str(replacement)})
    return rebound


def ensure_offscreen_framebuffer(root: ET.Element, *, width: int, height: int) -> dict[str, int]:
    """Raise only the overlay's offscreen framebuffer to the requested size."""
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    current_width = int(global_visual.get("offwidth", "0"))
    current_height = int(global_visual.get("offheight", "0"))
    result = {"offwidth": max(current_width, width), "offheight": max(current_height, height)}
    global_visual.set("offwidth", str(result["offwidth"]))
    global_visual.set("offheight", str(result["offheight"]))
    return result


def write_png(pixels: np.ndarray) -> bytes:
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def write_npy(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values)
    return buffer.getvalue()


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix.tolist()


def mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    """Return an inclusive image-space box for a non-empty boolean mask."""
    if mask.ndim != 2:
        raise ValueError("object mask must be two-dimensional")
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--object-body", default="test_cylinder")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fovy-deg", type=float, default=60.0)
    parser.add_argument(
        "--camera-mode",
        choices=("head_lookat_simulation_only", "official_mount_diagnostic"),
        default="head_lookat_simulation_only",
        help="Never treat the official mount origin as a confirmed optical frame.",
    )
    args = parser.parse_args()
    if not args.scene.is_file() or args.width <= 0 or args.height <= 0 or not 10.0 <= args.fovy_deg <= 160.0:
        raise SystemExit("scene/resolution or vertical field of view is invalid")
    source_hash = sha256(args.scene)
    source_tree = ET.parse(args.scene)
    source_root = source_tree.getroot()
    asset_rebindings = rebind_missing_workspace_assets(source_root)
    framebuffer = ensure_offscreen_framebuffer(source_root, width=args.width, height=args.height)
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-rgbd-truth-capture")
    output = location.native_path
    resolved_source_scene = output / "x2_rgbd_truth_resolved_source.xml"
    ET.indent(source_tree, space="  ")
    pg.write_new_bytes(resolved_source_scene, ET.tostring(source_root, encoding="utf-8", xml_declaration=True))
    parent = find_body(source_root, CAMERA_PARENT)
    if any(camera.get("name") == CAMERA_NAME for camera in source_root.findall(".//camera")):
        raise RuntimeError(f"source scene already contains {CAMERA_NAME}")
    source_model = mujoco.MjModel.from_xml_path(str(resolved_source_scene))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    parent_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_PARENT)
    source_object_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    if parent_id < 0 or source_object_id < 0:
        raise RuntimeError("source scene lacks the requested camera parent or target object")
    parent_rotation = source_data.xmat[parent_id].reshape(3, 3).copy()
    parent_position = source_data.xpos[parent_id].copy()
    mount_translation = np.fromstring(CAMERA_POS, sep=" ")
    camera_attributes = {"name": CAMERA_NAME, "fovy": f"{args.fovy_deg:.12g}"}
    if args.camera_mode == "head_lookat_simulation_only":
        camera_position = parent_position + parent_rotation @ mount_translation
        camera_rotation = look_at_rotation(camera_position, source_data.xpos[source_object_id].copy())
        local_rotation, local_position = pose_in_parent(
            parent_rotation,
            parent_position,
            camera_rotation,
            camera_position,
        )
        camera_attributes.update(
            {
                "pos": " ".join(f"{value:.12g}" for value in local_position),
                "quat": " ".join(f"{value:.12g}" for value in quaternion_wxyz(local_rotation)),
            }
        )
    else:
        camera_attributes.update({"pos": CAMERA_POS, "euler": CAMERA_EULER})
    ET.SubElement(
        parent,
        "camera",
        camera_attributes,
    )
    overlay_scene = output / "x2_rgbd_truth_sensor_overlay.xml"
    ET.indent(source_tree, space="  ")
    pg.write_new_bytes(overlay_scene, ET.tostring(source_root, encoding="utf-8", xml_declaration=True))
    model = mujoco.MjModel.from_xml_path(str(overlay_scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    if camera < 0 or object_body < 0:
        raise RuntimeError("compiled overlay lacks camera or target object")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=CAMERA_NAME)
        depth_m = renderer.render()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA_NAME)
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    finite_depth = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    camera_rotation = data.cam_xmat[camera].reshape(3, 3).copy()
    camera_position = data.cam_xpos[camera].copy()
    object_rotation = data.xmat[object_body].reshape(3, 3).copy()
    object_position = data.xpos[object_body].copy()
    object_geom_ids = np.flatnonzero(model.geom_bodyid == object_body)
    # MuJoCo Python's segmentation renderer returns [object_id, object_type].
    object_mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[..., 0], object_geom_ids)
    object_bbox = mask_bbox_xyxy(object_mask)
    simulated_intrinsic = 0.5 * args.height / math.tan(math.radians(args.fovy_deg) / 2.0)
    metadata = {
        "schema": "robotsim.issue46.rgbd-truth-capture.v1",
        "classification": (
            "SIMULATION_ONLY_HEAD_AIMED_SENSOR_OVERLAY_STATIC_TRUTH"
            if args.camera_mode == "head_lookat_simulation_only"
            else "OFFICIAL_MOUNT_DIAGNOSTIC_ONLY_STATIC_TRUTH"
        ),
        "mj_step_calls": 0,
        "source_scene": {"path": str(args.scene), "sha256": source_hash},
        "resolved_source_scene": {
            "path": str(resolved_source_scene),
            "sha256": sha256(resolved_source_scene),
            "asset_rebindings": asset_rebindings,
            "render_framebuffer_overlay_px": framebuffer,
        },
        "generated_scene": {"path": str(overlay_scene), "sha256": sha256(overlay_scene)},
        "camera": {
            "name": CAMERA_NAME,
            "parent_body": CAMERA_PARENT,
            "mode": args.camera_mode,
            "official_urdf_mount_origin": {"xyz": CAMERA_POS, "rpy": CAMERA_EULER},
            "fovy_deg": args.fovy_deg,
            "resolution_px": [args.width, args.height],
            "simulated_intrinsics_px": {
                "fx": simulated_intrinsic,
                "fy": simulated_intrinsic,
                "cx": (args.width - 1.0) / 2.0,
                "cy": (args.height - 1.0) / 2.0,
            },
            "world_position_m": camera_position.tolist(),
            "world_rotation_matrix_mujoco_camera": camera_rotation.tolist(),
            "T_world_mujoco_camera": homogeneous(camera_rotation, camera_position),
            "T_optical_from_mujoco_camera": homogeneous(MUJOCO_CAMERA_TO_OPTICAL, np.zeros(3)),
            "T_world_optical_camera": homogeneous(camera_rotation @ MUJOCO_CAMERA_TO_OPTICAL, camera_position),
            "optical_convention": (
                "SIMULATION_ONLY: local -Z is explicitly aimed at the target. "
                "This is not a live vendor RGB-D optical-frame calibration."
                if args.camera_mode == "head_lookat_simulation_only"
                else "UNVERIFIED: official mount origin/orientation is not established as a vendor RGB-D optical frame."
            ),
        },
        "object_truth": {
            "body": args.object_body,
            "geom_ids": [int(value) for value in object_geom_ids],
            "visible_pixel_count": int(object_mask.sum()),
            "visible_bbox_xyxy_px": object_bbox,
            "world_position_m": object_position.tolist(),
            "world_rotation_matrix": object_rotation.tolist(),
            "T_world_object": homogeneous(object_rotation, object_position),
        },
        "depth": {
            "units": "m",
            "valid_pixel_count": int(finite_depth.size),
            "min_m": None if finite_depth.size == 0 else float(finite_depth.min()),
            "max_m": None if finite_depth.size == 0 else float(finite_depth.max()),
        },
        "gates": {
            "SOURCE_SCENE_HASH": "PASS",
            "MISSING_WORKSPACE_ASSET_REBINDING": "PASS",
            "CAMERA_OVERLAY_ONLY": "PASS",
            "RENDER_FRAMEBUFFER_OVERLAY": "PASS",
            "STATIC_RGBD_RENDER": "PASS" if finite_depth.size else "FAIL",
            "STATIC_OBJECT_TRUTH_MASK": "PASS" if object_bbox is not None else "FAIL",
            "HEAD_AIMED_SIMULATION_CAMERA": "PASS" if args.camera_mode == "head_lookat_simulation_only" else "NOT_RUN",
            "LIVE_VENDOR_RGBD_FRAME_CONVENTION": "NOT_RUN",
        },
        "limitations": [
            "This is renderer truth from a source-preserving overlay, not a live RAICOM RGB-D topic capture.",
            "The resolved source copy may rebind missing historical worktree prefixes only under canonical reference/ or simulation/ roots.",
            "No perception model is invoked in this capture.",
        ],
    }
    pg.write_new_bytes(output / "rgb.png", write_png(rgb))
    pg.write_new_bytes(output / "depth_m.npy", write_npy(depth_m))
    pg.write_new_bytes(output / "object_truth_mask.png", write_png(object_mask.astype(np.uint8) * 255))
    pg.write_new_text(output / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": str(output),
                "artifact_manifest_wsl": manifest["manifest"]["WSL_PATH"],
                "gates": metadata["gates"],
                "depth": metadata["depth"],
                "mj_step_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
