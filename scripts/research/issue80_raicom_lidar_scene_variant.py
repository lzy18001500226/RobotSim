#!/usr/bin/env python3
"""Create a scene-selected copy of one evidence-local LiDAR overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


class VariantError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(*, overlay_root: Path, output_root: Path, variant: str, native_x2_timestep: bool = False) -> dict:
    if variant not in {"known_static", "runtime_unknown"}:
        raise VariantError("variant must be known_static or runtime_unknown")
    source_overlay = overlay_root / "lidar_overlay"
    if overlay_root.is_symlink() or source_overlay.is_symlink() or not source_overlay.is_dir():
        raise VariantError("source overlay is missing or unsafe")
    if output_root.is_symlink() or not output_root.is_dir() or any(item.name != ".robotsim-output-reservation.json" for item in output_root.iterdir()):
        raise VariantError("output root must contain only its governed reservation")
    output_overlay = output_root / "lidar_overlay"
    shutil.copytree(source_overlay, output_overlay)
    plugins = overlay_root / "mujoco_plugin"
    if plugins.is_symlink() or not plugins.is_dir():
        raise VariantError("source plugin directory is missing or unsafe")
    shutil.copytree(plugins, output_root / "mujoco_plugin")
    selected = output_overlay / (
        "scene_static_known_map.xml" if variant == "known_static" else "scene_lidar_overlay.xml"
    )
    if selected.is_symlink() or not selected.is_file():
        raise VariantError("selected scenario scene is missing")
    active = output_overlay / "scene_lidar_overlay.xml"
    if variant == "known_static":
        active.write_bytes(selected.read_bytes())
    if native_x2_timestep:
        tree = ET.parse(active)
        root = tree.getroot()
        option = root.find("./option")
        if option is None:
            option = ET.Element("option")
            root.insert(0, option)
        option.set("timestep", "0.001")
        ET.indent(tree, space="  ")
        tree.write(active, encoding="utf-8", xml_declaration=True)
    return {
        "schema": "robotsim.issue80.raicom-lidar-scene-variant.v1",
        "classification": (
            "SIMULATION_ONLY_SCENE_SELECTION_NATIVE_X2_TIMESTEP_NO_ROBOT_MODEL_CHANGE"
            if native_x2_timestep else "SIMULATION_ONLY_SCENE_SELECTION_NO_ROBOT_MODEL_CHANGE"
        ),
        "status": "PASS_VARIANT_PACKAGED",
        "variant": variant,
        "source_overlay": str(overlay_root),
        "source_active_scene_sha256": sha256(source_overlay / "scene_lidar_overlay.xml"),
        "selected_scene_sha256": sha256(selected),
        "active_scene_sha256": sha256(active),
        "unchanged_robot_files": {
            name: sha256(source_overlay / name) == sha256(output_overlay / name)
            for name in ("x2_lidar_overlay.xml", "robot_lidar_overlay.yaml")
        },
        "native_x2_timestep": {
            "enabled": native_x2_timestep,
            "scene_option_timestep_s": 0.001 if native_x2_timestep else "SOURCE_DEFAULT",
            "robot_or_controller_parameters_changed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("known_static", "runtime_unknown"), required=True)
    parser.add_argument("--native-x2-timestep", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate(
            overlay_root=args.overlay_root.resolve(strict=True),
            output_root=args.output_dir.resolve(strict=True),
            variant=args.variant, native_x2_timestep=args.native_x2_timestep,
        )
    except (OSError, VariantError) as exc:
        parser.error(str(exc))
    (args.output_dir / "scene_variant.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
