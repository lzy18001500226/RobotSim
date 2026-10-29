#!/usr/bin/env python3
"""Static source-chair LiDAR difference audit using MuJoCo ``mj_ray`` only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np

from issue80_lidar_obstacle_layer import GridSpec, RAICOM_CHEST_RAYCASTER
from issue80_raicom_chair_detour_static import (
    CHAIR_RELATIVE,
    ROBOT_ROOT_Z_M,
    WORKTREE,
    chair_model_xml,
    scene_with_extra_chair,
    static_mug_wrapper,
)
from issue80_local01_scene_route import (
    _resolve_raicom_reference_root,
    aligned_start_scene_tree,
    assert_native_robot_equivalence,
    _resolve_runtime_x2_path,
)


SCENE_SOURCE = WORKTREE / "simulation/mujoco/scenes/raicom2026/scene_workspace.xml"
CAMERA_NAME = "RayCasterLidar"
LIDAR_X2_OVERLAY = Path(
    "/home/lzy18001500226/projects/RobotSim/reviews/issue-80/"
    "raicom-raw-lidar-overlay-source-20261028T115855Z/lidar_overlay/x2_lidar_overlay.xml"
)


def stage_x2_camera_model(output: Path) -> Path:
    """Remove the runtime plugin only; retain its source-derived camera mount."""
    if LIDAR_X2_OVERLAY.is_symlink() or not LIDAR_X2_OVERLAY.is_file():
        raise ValueError("verified LiDAR X2 overlay is missing or unsafe")
    tree = ET.parse(LIDAR_X2_OVERLAY)
    root = tree.getroot()
    extension = root.find("./extension")
    if extension is not None:
        root.remove(extension)
    sensor = root.find("./sensor")
    if sensor is not None:
        for item in list(sensor):
            if item.tag == "plugin":
                sensor.remove(item)
    camera = [item for item in root.iter("camera") if item.get("name") == CAMERA_NAME]
    if len(camera) != 1:
        raise ValueError("LiDAR overlay has no unique source-derived camera")
    compiler = root.find("./compiler")
    if compiler is None:
        raise ValueError("X2 overlay has no compiler")
    runtime_x2 = _resolve_runtime_x2_path(WORKTREE)
    compiler.set("meshdir", str(runtime_x2.parent.parent / "meshes"))
    path = output / "x2_static_lidar_camera.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def stage_mug(scene: ET.Element, output: Path, reference_root: Path) -> None:
    mug = output / "mug_portable_static_runtime.xml"
    static_mug_wrapper(
        SCENE_SOURCE.parent / "mug_portable.xml",
        reference_root / "robot/lx2501_3_t2d5/model_info/object/model/mug",
        mug,
    )
    models = [item for item in scene.findall("./asset/model") if item.get("name") == "raicom_mug"]
    if len(models) != 1:
        raise ValueError("scene lacks unique mug model")
    models[0].set("file", str(mug))


def robot_root_body(model: mujoco.MjModel) -> int:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body < 0:
        raise ValueError("scene lacks pelvis")
    return body


def belongs_to_root(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return body == root


def scan(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    root = robot_root_body(model)
    if camera < 0:
        raise ValueError("scene lacks static audit camera")
    origin = np.array(data.cam_xpos[camera], dtype=float)
    rotation = np.array(data.cam_xmat[camera], dtype=float).reshape(3, 3)
    points = np.full((RAICOM_CHEST_RAYCASTER.ray_count, 3), np.nan, dtype=float)
    distances = np.full(RAICOM_CHEST_RAYCASTER.ray_count, np.nan, dtype=float)
    for index in range(RAICOM_CHEST_RAYCASTER.ray_count):
        row, column = divmod(index, RAICOM_CHEST_RAYCASTER.horizontal_rays)
        yaw = (0.5 - column / (RAICOM_CHEST_RAYCASTER.horizontal_rays - 1)) * RAICOM_CHEST_RAYCASTER.horizontal_fov_rad
        pitch = (0.5 - row / (RAICOM_CHEST_RAYCASTER.vertical_rays - 1)) * RAICOM_CHEST_RAYCASTER.vertical_fov_rad
        local = np.array([-math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])
        direction = rotation @ local
        shifted = origin.copy()
        travelled = 0.0
        for _ in range(64):
            geom = np.array([-1], dtype=np.int32)
            hit = float(mujoco.mj_ray(model, data, shifted, direction, None, 1, -1, geom))
            if hit < 0 or geom[0] < 0:
                break
            body = int(model.geom_bodyid[int(geom[0])])
            if not belongs_to_root(model, body, root):
                total = travelled + hit
                if RAICOM_CHEST_RAYCASTER.minimum_range_m <= total <= RAICOM_CHEST_RAYCASTER.maximum_range_m:
                    distances[index] = total
                    points[index] = origin + total * direction
                break
            advance = hit + 1e-3
            shifted += advance * direction
            travelled += advance
    return points, distances


def prepare_scene(output: Path, camera_model: Path, *, chair: bool) -> tuple[mujoco.MjModel, ET.Element]:
    reference = _resolve_raicom_reference_root(WORKTREE)
    scene = aligned_start_scene_tree(SCENE_SOURCE)
    stage_mug(scene, output, reference)
    if chair:
        mesh = reference / CHAIR_RELATIVE
        chair_model = output / "sim_only_dynamic_chair.xml"
        chair_tree = ET.ElementTree(chair_model_xml(mesh))
        ET.indent(chair_tree, space="  ")
        chair_tree.write(chair_model, encoding="utf-8", xml_declaration=True)
        scene = scene_with_extra_chair(scene_source=SCENE_SOURCE, chair_model_path=chair_model, chair_x_m=-0.25, chair_y_m=-1.50, chair_yaw_rad=0.0)
        # scene_with_extra_chair rebuilds from source; restore the static mug and camera below.
        frames = [item for item in scene.findall("./worldbody/frame") if item.get("name") == "x2_start_frame"]
        if len(frames) != 1:
            raise ValueError("source scene lacks unique x2_start_frame")
        frames[0].set("quat", "1 0 0 0")
        stage_mug(scene, output, reference)
    x2_models = [item for item in scene.findall("./asset/model") if item.get("name") == "x2_mc_runtime"]
    if len(x2_models) != 1:
        raise ValueError("scene lacks unique X2 model include")
    x2_models[0].set("file", str(camera_model))
    path = output / ("scene_with_chair_lidar_static.xml" if chair else "scene_baseline_lidar_static.xml")
    tree = ET.ElementTree(scene)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode")), scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=True)
    if output.is_symlink() or any(item.name != ".robotsim-output-reservation.json" for item in output.iterdir()):
        parser.error("output must be a new governed evidence directory")
    if mujoco.__version__ != "3.3.6":
        parser.error(f"requires MuJoCo 3.3.6, got {mujoco.__version__}")
    camera_model = stage_x2_camera_model(output)
    baseline, _ = prepare_scene(output, camera_model, chair=False)
    with_chair, _ = prepare_scene(output, camera_model, chair=True)
    baseline_data, chair_data = mujoco.MjData(baseline), mujoco.MjData(with_chair)
    for model, data in ((baseline, baseline_data), (with_chair, chair_data)):
        mujoco.mj_resetData(model, data)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        address = int(model.jnt_qposadr[root])
        data.qpos[address:address + 3] = [-1.5, -1.5, ROBOT_ROOT_Z_M]
        data.qpos[address + 3:address + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
    native = mujoco.MjModel.from_xml_path(str(_resolve_runtime_x2_path(WORKTREE).with_name("scene.xml")))
    equivalence = assert_native_robot_equivalence(native, with_chair)
    baseline_points, baseline_distances = scan(baseline, baseline_data)
    chair_points, chair_distances = scan(with_chair, chair_data)
    deltas = baseline_distances - chair_distances
    valid = np.isfinite(chair_distances) & ((~np.isfinite(baseline_distances)) | (deltas > 0.0))
    thresholds = (0.20, 0.25, 0.30, 0.40)
    grid = GridSpec(-2.0, -2.0, 0.10, 40, 40)
    result = {
        "schema": "robotsim.issue80.static-lidar-chair-audit.v1",
        "classification": "SIMULATION_ONLY_STATIC_MJ_RAY_SOURCE_CHAIR_DIFFERENCE",
        "status": "PASS_STATIC_CHAIR_RAY_DIFFERENCE",
        "mujoco_version": mujoco.__version__,
        "mj_step_calls": 0,
        "camera": {"name": CAMERA_NAME, "rays": RAICOM_CHEST_RAYCASTER.ray_count},
        "native_robot_equivalence": equivalence,
        "baseline_hit_count": int(np.count_nonzero(np.isfinite(baseline_distances))),
        "chair_hit_count": int(np.count_nonzero(np.isfinite(chair_distances))),
        "new_or_nearer_ray_count": int(np.count_nonzero(valid)),
        "ray_count_by_nearer_margin_m": {str(value): int(np.count_nonzero(np.isfinite(chair_distances) & ((~np.isfinite(baseline_distances)) | (deltas >= value)))) for value in thresholds},
        "chair_points_by_margin_m": {
            str(value): [chair_points[index].tolist() for index in np.flatnonzero(np.isfinite(chair_distances) & ((~np.isfinite(baseline_distances)) | (deltas >= value)))]
            for value in thresholds
        },
        "static_grid_cells_by_margin_m": {
            str(value): sorted({(grid.world_to_cell(*chair_points[index][:2]).x, grid.world_to_cell(*chair_points[index][:2]).y) for index in np.flatnonzero(np.isfinite(chair_distances) & ((~np.isfinite(baseline_distances)) | (deltas >= value))) if grid.contains(grid.world_to_cell(*chair_points[index][:2]))})
            for value in thresholds
        },
    }
    (output / "static_lidar_chair_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
