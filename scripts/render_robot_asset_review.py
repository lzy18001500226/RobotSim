#!/usr/bin/env python3
"""Render static, source-identified review images without stepping physics."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "assets/robots/registry.json"
CHECKER_PATH = ROOT / "scripts/check_robot_asset_registry.py"
SPEC = importlib.util.spec_from_file_location("asset_registry_checker", CHECKER_PATH)
checker = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(checker)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def body_target(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    candidates = [
        "left_wrist_yaw_link" if side == "left" else "right_wrist_yaw_link",
        "left_wrist_roll_link" if side == "left" else "right_wrist_roll_link",
        "left_wrist_roll_rubber_hand" if side == "left" else "right_wrist_roll_rubber_hand",
        "L_hand_base_link" if side == "left" else "R_hand_base_link",
        "L_base_link" if side == "left" else "R_base_link",
        "L_palm" if side == "left" else "R_palm",
        "L_omnipicker_base_link" if side == "left" else "R_omnipicker_base_link",
        "left_dex1_base_link" if side == "left" else "right_dex1_base_link",
    ]
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i
        for i in range(model.nbody)
    }
    for name in candidates:
        if name in names:
            return data.xpos[names[name]].copy()
    fallback = [
        i for i in range(model.nbody)
        if (side == "left" and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower().startswith(("left_", "l_"))
        or (side == "right" and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower().startswith(("right_", "r_"))
    ]
    if fallback:
        return data.xpos[fallback[-1]].copy()
    return np.zeros(3)


def robot_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    keep = model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    if not np.any(keep):
        return np.zeros(3), 1.0
    positions = data.geom_xpos[keep]
    radii = model.geom_rbound[keep]
    lower = np.min(positions - radii[:, None], axis=0)
    upper = np.max(positions + radii[:, None], axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.linalg.norm(upper - lower)) * 0.52, 0.08)
    return center, radius


def new_camera(target: np.ndarray, radius: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = radius
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
           target: np.ndarray, distance: float, azimuth: float, elevation: float,
           groups: list[int] | None = None) -> np.ndarray:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[:] = False
    if groups is None:
        option.geomgroup[:] = True
    else:
        for group in groups:
            if 0 <= group < len(option.geomgroup):
                option.geomgroup[group] = True
    camera = new_camera(target, distance, azimuth, elevation)
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


def save_labeled(array: np.ndarray, path: Path, label: str) -> None:
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 42), fill=(15, 22, 28))
    draw.text((12, 7), label, fill=(250, 250, 250), font=font)
    image.save(path)


def set_source_joint_endpoint(model: mujoco.MjModel, data: mujoco.MjData, endpoint: int,
                              source_path: Path) -> tuple[bool, list[str]]:
    changed = False
    driver_tokens = ("claw_joint", "dex1", "finger_joint")
    joint_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id): joint_id
        for joint_id in range(model.njnt)
    }
    source_mimics: dict[str, tuple[str, float, float]] = {}
    drivers_changed = False
    if source_path.suffix.lower() == ".urdf":
        root = ET.parse(source_path).getroot()
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None and mimic.get("joint"):
                source_mimics[joint.get("name", "")] = (
                    mimic.get("joint", ""), float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0")),
                )
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not any(token in name.lower() for token in driver_tokens):
            continue
        if not model.jnt_limited[joint_id] or model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        ):
            continue
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address] = model.jnt_range[joint_id, endpoint]
        drivers_changed = True
    if not drivers_changed:
        return False, []
    applied_mimics: list[str] = []
    pending = dict(source_mimics)
    while pending:
        progressed = False
        for follower, (driver, multiplier, offset) in list(pending.items()):
            if driver not in joint_ids or follower not in joint_ids:
                pending.pop(follower)
                continue
            follower_id = joint_ids[follower]
            driver_id = joint_ids[driver]
            if not model.jnt_limited[driver_id] or model.jnt_type[driver_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
            ):
                pending.pop(follower)
                continue
            driver_qpos = float(data.qpos[int(model.jnt_qposadr[driver_id])])
            follower_value = multiplier * driver_qpos + offset
            if model.jnt_limited[follower_id]:
                lower, upper = model.jnt_range[follower_id]
                follower_value = float(np.clip(follower_value, lower, upper))
            data.qpos[int(model.jnt_qposadr[follower_id])] = follower_value
            applied_mimics.append(follower)
            pending.pop(follower)
            progressed = True
        if not progressed:
            break
    return True, applied_mimics


def make_sheet(paths: list[Path], target: Path, label: str) -> None:
    tiles = [Image.open(path).convert("RGB") for path in paths]
    if not tiles:
        return
    tile_w, tile_h = 480, 360
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + 32)), (238, 240, 241))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (path, tile) in enumerate(zip(paths, tiles)):
        tile.thumbnail((tile_w, tile_h))
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + 32)
        canvas.paste(tile, (x, y + 24))
        draw.text((x + 8, y + 5), path.stem, fill=(20, 25, 28), font=font)
    draw.rectangle((0, 0, canvas.width, 22), fill=(15, 22, 28))
    draw.text((8, 5), label, fill=(250, 250, 250), font=font)
    canvas.save(target)
    for image in tiles:
        image.close()


def render_asset(asset: dict[str, Any], source_profile: dict[str, Any], source_root: Path, output_root: Path,
                 width: int, height: int) -> dict[str, Any]:
    source_path = (source_root / asset["model_path"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    model_hash = sha256_file(source_path)
    model = mujoco.MjModel.from_xml_path(str(source_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center, radius = robot_bounds(model, data)
    directory = output_root / asset["asset_id"]
    directory.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    label = f"{asset['asset_id']} | model_sha256={model_hash[:16]} | static, physics_steps=0"
    generated: list[Path] = []

    views = [
        ("overview_front.png", center, radius * 2.15, 180, 0),
        ("overview_side.png", center, radius * 2.15, 90, 0),
        ("overview_rear.png", center, radius * 2.15, 0, 0),
    ]
    for filename, target, distance, azimuth, elevation in views:
        path = directory / filename
        save_labeled(render(renderer, model, data, target, distance, azimuth, elevation), path, label + " | all source geoms")
        generated.append(path)

    hand_targets = {}
    for side in ("left", "right"):
        target = body_target(model, data, side)
        hand_targets[side] = target.tolist()
        outward_azimuth = 90 if side == "left" else 270
        offset = np.array([0.035, 0.0, 0.0])
        path = directory / f"{side}_hand_closeup.png"
        save_labeled(render(renderer, model, data, target + offset, 0.44, outward_azimuth, -8), path, label + f" | {side} hand outboard view, static source pose")
        generated.append(path)
        front_path = directory / f"{side}_hand_front.png"
        save_labeled(render(renderer, model, data, target + offset, 0.38, 180, 0), front_path, label + f" | {side} hand frontal view, static source pose")
        generated.append(front_path)

    original_groups = model.geom_group.copy()
    original_rgba = model.geom_rgba.copy()
    has_contact = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_group[:] = np.where(has_contact, 4, 5)
    visual_path = directory / "visual_geometry_only.png"
    save_labeled(render(renderer, model, data, center, radius * 2.15, 180, 0, [5]), visual_path, label + " | contact-disabled visual geoms")
    generated.append(visual_path)
    collision_path = directory / "collision_geometry_only.png"
    save_labeled(render(renderer, model, data, center, radius * 2.15, 180, 0, [4]), collision_path, label + " | contact-enabled geoms")
    generated.append(collision_path)
    model.geom_rgba[has_contact, :3] = np.array([1.0, 0.12, 0.05])
    model.geom_rgba[has_contact, 3] = 0.55
    overlay_path = directory / "visual_collision_overlay.png"
    save_labeled(render(renderer, model, data, center, radius * 2.15, 180, 0), overlay_path, label + " | orange=contact-enabled geoms")
    generated.append(overlay_path)
    model.geom_group[:] = original_groups
    model.geom_rgba[:] = original_rgba

    static_states = []
    forward_calls = 1
    for endpoint, state_name in ((0, "driver_lower_limits_static.png"), (1, "driver_upper_limits_static.png")):
        data = mujoco.MjData(model)
        changed, mimic_followers = set_source_joint_endpoint(model, data, endpoint, source_path)
        if changed:
            mujoco.mj_forward(model, data)
            forward_calls += 1
            path = directory / state_name
            save_labeled(render(renderer, model, data, center, radius * 2.15, 180, 0), path,
                         label + f" | source driver range endpoint {endpoint}, not an actuator trajectory")
            generated.append(path)
            static_states.append({"image": path.name, "kind": "static_source_joint_range_endpoint",
                                  "mimic_followers_set_from_source_relation": mimic_followers, "physics_steps": 0})

    sheet_names = ["overview_front.png", "right_hand_closeup.png", "collision_geometry_only.png", "visual_collision_overlay.png"]
    sheet_path = directory / "contact_sheet.png"
    make_sheet([directory / name for name in sheet_names], sheet_path, label)
    generated.append(sheet_path)
    signature = checker.mujoco_snapshot(source_path)
    meta = {
        "asset_id": asset["asset_id"],
        "source_profile": asset["source_profile"],
        "upstream_repository": source_profile["repository"],
        "upstream_commit": source_profile["commit"],
        "source_model_path": asset["model_path"],
        "source_model_sha256": model_hash,
        "mujoco_version": mujoco.__version__,
        "compiled_signature": signature["signature"],
        "view_targets_world": {"left_hand": hand_targets["left"], "right_hand": hand_targets["right"]},
        "static_joint_range_states": static_states,
        "physics_steps": 0,
        "mj_step_calls": 0,
        "mj_forward_calls": forward_calls,
        "actuator_trajectory": False,
        "generated_images": [
            {"path": path.name, "sha256": sha256_file(path)} for path in generated
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    renderer.close()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-revision", help="required assertion for pinned raw source snapshots without Git metadata")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    asset = checker.find_asset(registry, args.asset_id)
    if asset["review_state"] == "REJECTED":
        raise SystemExit(f"refusing to generate new gallery for rejected asset: {args.asset_id}")
    hashes = json.loads(checker.HASHES_PATH.read_text(encoding="utf-8"))
    compiled = json.loads(checker.COMPILED_PATH.read_text(encoding="utf-8"))
    source_profile = registry["source_profiles"][asset["source_profile"]]
    source_root = Path(args.source_root).resolve()
    errors = checker.check_profile(asset["source_profile"], source_profile, hashes, source_root, args.source_revision)
    model_path = source_root / asset["model_path"]
    expected_hash = hashes["profiles"][asset["source_profile"]]["files"].get(asset["model_path"])
    if not model_path.is_file() or checker.sha256_file(model_path) != expected_hash:
        errors.append(f"source model SHA-256 mismatch: {asset['model_path']}")
    compiled_profile = compiled.get("profiles", {}).get(args.asset_id)
    if not compiled_profile:
        errors.append(f"compiled signature missing for {args.asset_id}")
    else:
        errors.extend(checker.compare_model(asset, checker.mujoco_snapshot(model_path), compiled_profile))
    if errors:
        raise SystemExit("\n".join(f"FAIL: {error}" for error in errors))
    output_root = Path(args.output_root).resolve()
    metadata = render_asset(asset, source_profile, source_root, output_root, args.width, args.height)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
