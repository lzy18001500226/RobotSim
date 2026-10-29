#!/usr/bin/env python3
"""Package a random unknown-obstacle scene into a source-derived LiDAR overlay.

This creates a runtime-only scene wrapper.  It reuses the exact X2 LiDAR
overlay and plugin while setting only model-file paths and adding the
scenario's unknown static obstacle environment.  The known-map scene excludes
all generated obstacle bodies by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


class OverlayError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise OverlayError(f"{label} is missing or unsafe")


def rewrite_scene(scene_path: Path, output: Path, *, include_unknown: bool) -> None:
    _safe_file(scene_path, "scenario scene")
    tree = ET.parse(scene_path)
    root = tree.getroot()
    assets = root.find("./asset")
    if assets is None:
        raise OverlayError("scenario scene lacks asset section")
    required = {"x2_mc_runtime", "raicom_room", "raicom_mug"}
    includes = {item.get("name"): item for item in assets.findall("model")}
    if not required.issubset(includes):
        raise OverlayError("scenario scene lacks base model include")
    includes["x2_mc_runtime"].set("file", "/workspace/evidence/lidar_overlay/x2_lidar_overlay.xml")
    includes["raicom_room"].set(
        "file",
        "/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/scene/room/room_with_chair_review.xml",
    )
    includes["raicom_mug"].set("file", "/workspace/evidence/local01_scene/mug_portable_runtime.xml")
    unknown = assets.find("./model[@name='sim_only_unknown_chair']")
    if include_unknown:
        if unknown is None:
            raise OverlayError("runtime scenario has no unknown-chair source model")
        unknown.set("file", "/workspace/evidence/lidar_overlay/sim_only_unknown_chair.xml")
    elif unknown is not None:
        assets.remove(unknown)
    ET.indent(tree, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def generate(*, scenario_root: Path, base_overlay_root: Path, output_root: Path) -> dict:
    if output_root.is_symlink() or not output_root.is_dir():
        raise OverlayError("output root is unavailable")
    if any(item.name != ".robotsim-output-reservation.json" for item in output_root.iterdir()):
        raise OverlayError("output root must contain only its governed reservation")
    scenario = scenario_root / "scenario.json"
    static_scene = scenario_root / "scene_static_known_map.xml"
    runtime_scene = scenario_root / "scene_runtime_unknown_obstacles.xml"
    base_overlay = base_overlay_root / "lidar_overlay"
    base_plugins = base_overlay_root / "mujoco_plugin"
    for path, label in ((scenario, "scenario"), (static_scene, "static scene"), (runtime_scene, "runtime scene")):
        _safe_file(path, label)
    if base_overlay_root.is_symlink() or base_overlay.is_symlink() or base_plugins.is_symlink() or not base_overlay.is_dir() or not base_plugins.is_dir():
        raise OverlayError("base LiDAR overlay is missing or unsafe")
    scenario_data = json.loads(scenario.read_text(encoding="utf-8"))
    if scenario_data.get("status") != "PASS_SCENARIO_GENERATED":
        raise OverlayError("random scenario did not pass generation")
    overlay = output_root / "lidar_overlay"
    shutil.copytree(base_overlay, overlay)
    shutil.copytree(base_plugins, output_root / "mujoco_plugin")
    unknown_model = scenario_root / "sim_only_unknown_chair.xml"
    _safe_file(unknown_model, "unknown chair model")
    unknown_root = ET.parse(unknown_model).getroot()
    compiler = unknown_root.find("./compiler")
    if compiler is None:
        raise OverlayError("unknown chair model lacks compiler path")
    compiler.set("meshdir", "/workspace/raicom_scene_source/robot/lx2501_3_t2d5/model_info/object/mesh")
    ET.indent(ET.ElementTree(unknown_root), space="  ")
    ET.ElementTree(unknown_root).write(overlay / "sim_only_unknown_chair.xml", encoding="utf-8", xml_declaration=True)
    rewrite_scene(static_scene, overlay / "scene_static_known_map.xml", include_unknown=False)
    rewrite_scene(runtime_scene, overlay / "scene_lidar_overlay.xml", include_unknown=True)
    config_path = overlay / "simulator_lidar_overlay.yaml"
    _safe_file(config_path, "LiDAR simulator config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    module = config.get("MujocoSimModule") if isinstance(config, dict) else None
    if not isinstance(module, dict):
        raise OverlayError("LiDAR simulator config lacks MujocoSimModule")
    module["simulation_model"] = "/workspace/evidence/lidar_overlay/scene_lidar_overlay.xml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    unchanged = {
        name: sha256(overlay / name) == sha256(base_overlay / name)
        for name in ("x2_lidar_overlay.xml", "robot_lidar_overlay.yaml")
    }
    if not all(unchanged.values()):
        raise OverlayError("random overlay altered X2 or LiDAR robot configuration")
    result = {
        "schema": "robotsim.issue80.raicom-random-obstacle-lidar-overlay.v1",
        "classification": "SIMULATION_ONLY_KNOWN_MAP_UNKNOWN_RUNTIME_OBSTACLES_LIDAR_OVERLAY",
        "status": "PASS_OVERLAY_PACKAGED",
        "scenario_source": str(scenario_root),
        "scenario_sha256": sha256(scenario),
        "known_static_scene_sha256": sha256(static_scene),
        "runtime_unknown_scene_sha256": sha256(runtime_scene),
        "base_overlay_source": str(base_overlay_root),
        "base_overlay_files_sha256": {name: sha256(base_overlay / name) for name in ("x2_lidar_overlay.xml", "robot_lidar_overlay.yaml", "scene_lidar_overlay.xml", "simulator_lidar_overlay.yaml")},
        "generated": {
            "overlay_dir": str(overlay),
            "known_static_scene": str(overlay / "scene_static_known_map.xml"),
            "runtime_scene": str(overlay / "scene_lidar_overlay.xml"),
        },
        "unknown_obstacle_count": len(scenario_data["runtime_unknown_obstacles"]["chairs"]) + len(scenario_data["runtime_unknown_obstacles"]["barriers"]),
        "unchanged_source_robot_files": unchanged,
        "forbidden_delta": ["X2 body/joint/actuator/sensor", "MC binary/config", "collision filters", "controller gains"],
    }
    (overlay / "RANDOM_OBSTACLE_LIDAR_OVERLAY_PROVENANCE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--base-overlay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(
            scenario_root=args.scenario_root.resolve(strict=True),
            base_overlay_root=args.base_overlay_root.resolve(strict=True),
            output_root=args.output_dir.resolve(strict=True),
        )
    except (OSError, OverlayError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
