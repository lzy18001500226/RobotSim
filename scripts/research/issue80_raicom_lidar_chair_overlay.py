#!/usr/bin/env python3
"""Generate a source-mesh extra-chair wrapper around the verified LiDAR overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from issue80_raicom_chair_detour_static import CHAIR_RELATIVE, WORKTREE, chair_model_xml
from issue80_local01_scene_route import _resolve_raicom_reference_root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(source_root: Path, output_root: Path, *, chair_x_m: float, chair_y_m: float, chair_yaw_rad: float) -> dict:
    source = source_root / "lidar_overlay"
    if source_root.is_symlink() or source.is_symlink() or not source.is_dir():
        raise ValueError("source LiDAR overlay is missing or unsafe")
    if not output_root.is_dir() or output_root.is_symlink() or any(item.name != ".robotsim-output-reservation.json" for item in output_root.iterdir()):
        raise FileExistsError("output must be a new governed evidence directory")
    output = output_root / "lidar_overlay"
    shutil.copytree(source, output)
    plugin_source = source_root / "mujoco_plugin"
    if plugin_source.is_symlink() or not plugin_source.is_dir():
        raise ValueError("source LiDAR plugin directory is missing or unsafe")
    shutil.copytree(plugin_source, output_root / "mujoco_plugin")
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("copied LiDAR overlay unexpectedly contains a symlink")
    reference = _resolve_raicom_reference_root(WORKTREE)
    mesh = reference / CHAIR_RELATIVE
    if mesh.is_symlink() or not mesh.is_file():
        raise ValueError("source chair mesh is missing or unsafe")
    chair_model = output / "sim_only_dynamic_chair.xml"
    chair_tree = ET.ElementTree(chair_model_xml(mesh))
    compiler = chair_tree.getroot().find("./compiler")
    assert compiler is not None
    compiler.set("meshdir", "/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/object/mesh")
    ET.indent(chair_tree, space="  ")
    chair_tree.write(chair_model, encoding="utf-8", xml_declaration=True)

    scene_path = output / "scene_lidar_overlay.xml"
    scene = ET.parse(scene_path)
    root = scene.getroot()
    asset, worldbody = root.find("./asset"), root.find("./worldbody")
    if asset is None or worldbody is None:
        raise ValueError("LiDAR scene lacks asset/worldbody")
    ET.SubElement(asset, "model", {
        "name": "sim_only_dynamic_chair", "file": "/workspace/evidence/lidar_overlay/sim_only_dynamic_chair.xml",
    })
    import math
    frame = ET.SubElement(worldbody, "frame", {
        "name": "sim_only_dynamic_chair_frame",
        "pos": f"{chair_x_m:.9g} {chair_y_m:.9g} 0",
        "quat": f"{math.cos(chair_yaw_rad / 2.0):.17g} 0 0 {math.sin(chair_yaw_rad / 2.0):.17g}",
    })
    ET.SubElement(frame, "attach", {"model": "sim_only_dynamic_chair", "prefix": "simobs_"})
    ET.indent(scene, space="  ")
    scene.write(scene_path, encoding="utf-8", xml_declaration=True)

    config_path = output / "simulator_lidar_overlay.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    module = config.get("MujocoSimModule") if isinstance(config, dict) else None
    if not isinstance(module, dict):
        raise ValueError("LiDAR simulator config lacks MujocoSimModule")
    module["simulation_model"] = "/workspace/evidence/lidar_overlay/scene_lidar_overlay.xml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    unchanged = {
        name: sha256(output / name) == sha256(source / name)
        for name in ("x2_lidar_overlay.xml", "robot_lidar_overlay.yaml")
    }
    if not all(unchanged.values()):
        raise ValueError("chair overlay altered X2 or robot sensor configuration")
    manifest = {
        "schema": "robotsim.issue80.raicom-lidar-chair-overlay.v1",
        "classification": "SIMULATION_ONLY_SOURCE_CHAIR_ENVIRONMENT_OVERLAY",
        "source_overlay": str(source_root),
        "source_overlay_files_sha256": {name: sha256(source / name) for name in ("x2_lidar_overlay.xml", "robot_lidar_overlay.yaml", "scene_lidar_overlay.xml", "simulator_lidar_overlay.yaml")},
        "chair_mesh": str(mesh), "chair_mesh_sha256": sha256(mesh),
        "chair_pose": {"x_m": chair_x_m, "y_m": chair_y_m, "yaw_rad": chair_yaw_rad},
        "generated": {"overlay_dir": str(output), "chair_model": str(chair_model)},
        "plugin_copy_sha256": {
            item.name: sha256(item) for item in sorted((output_root / "mujoco_plugin").iterdir()) if item.is_file()
        },
        "unchanged_source_robot_files": unchanged,
        "allowed_scene_delta": ["one source chair mesh model", "one attached simobs_ chair frame", "simulator scene-model path"],
        "forbidden_delta": ["X2 body/joint/actuator/sensor", "MC binary/config", "collision filters", "controller gains"],
    }
    (output / "LIDAR_CHAIR_OVERLAY_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-overlay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chair-x-m", type=float, default=-0.25)
    parser.add_argument("--chair-y-m", type=float, default=-1.50)
    parser.add_argument("--chair-yaw-rad", type=float, default=0.0)
    args = parser.parse_args(argv)
    manifest = generate(args.source_overlay_root.resolve(strict=True), args.output_dir.resolve(strict=False), chair_x_m=args.chair_x_m, chair_y_m=args.chair_y_m, chair_yaw_rad=args.chair_yaw_rad)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
