#!/usr/bin/env python3
"""Generate an evidence-local, source-derived RAICOM chest-LiDAR overlay.

This tool leaves the delivered MC runtime and official X2 assets untouched.  It
adds only the vendor-provided ray-caster plugin sensor to an evidence-local
copy of the runtime MJCF/YAML configuration.  It is explicitly SIMULATION_ONLY
until the precompiled simulator is observed publishing valid point-cloud data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = REPO / "simulation/mujoco/raicom2026"
X2 = RUNTIME / "sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
ROBOT_YAML = RUNTIME / "sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml"
SIM_YAML = RUNTIME / "sim_mujoco/configuration/robot/lx2501_3_t2d5/simulator/default.yaml"
OFFICIAL_URDF = (
    REPO / "reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
)

X2_SHA256 = "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3"
ROBOT_YAML_SHA256 = "3abe18ecb23cddccf705cb0efe1fb7f589e54563729d2cf67abf847a2fdab4c8"
SIM_YAML_SHA256 = "c9af1d04a85baab88c54ca7fc31056734f6481a9f1d823d7bfe363c9421a1f02"
OFFICIAL_URDF_SHA256 = "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"

PLUGIN_NAME = "robotsim.issue80.ray_caster_lidar"
SENSOR_NAME = "raycaster_lidar"
SENSOR_INSTANCE = "issue80_raycaster_lidar_instance"
CAMERA_NAME = "RayCasterLidar"
SIM_POINTS_TOPIC = "/aima/sim/lidar/points"
MESH_DIR_CONTAINER = "/workspace/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5/meshes"

# Source: X2 v1.4.0 X2-Ultra_omnipicker.urdf, lidar_chest_front_joint.
CHEST_LIDAR_POS_M = (0.102632788961867, 0.0, 0.181583726384016)
# The source joint points the sensor toward torso +X.  These camera axes make
# MuJoCo camera -Z look along +X; the derivation is recorded in the manifest.
CHEST_LIDAR_XYAXES = "0 -1 0 0 0 1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_pinned(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required source is missing or unsafe: {path}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"source hash mismatch for {path.name}: expected {expected}, got {observed}")


def count_model_elements(root: ET.Element) -> dict[str, int]:
    return {tag: len(root.findall(f".//{tag}")) for tag in ("body", "joint", "geom", "actuator", "sensor", "camera")}


def one_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [element for element in root.iter(tag) if element.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {tag} named {name}, found {len(matches)}")
    return matches[0]


def generate_overlay(output_dir: Path, scene_source: Path) -> dict[str, Any]:
    """Write a generated overlay bundle and its exact provenance manifest."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite overlay output: {output_dir}")
    if scene_source.is_symlink() or not scene_source.is_file():
        raise ValueError("scene source must be a regular file")
    for path, expected in (
        (X2, X2_SHA256),
        (ROBOT_YAML, ROBOT_YAML_SHA256),
        (SIM_YAML, SIM_YAML_SHA256),
        (OFFICIAL_URDF, OFFICIAL_URDF_SHA256),
    ):
        require_pinned(path, expected)

    output_dir.mkdir(parents=True, exist_ok=False)
    x2_tree = ET.parse(X2)
    x2_root = x2_tree.getroot()
    before_counts = count_model_elements(x2_root)
    compiler = x2_root.find("compiler")
    sensor = x2_root.find("sensor")
    if compiler is None or sensor is None:
        raise ValueError("runtime MJCF lacks compiler or sensor section")
    if any(element.get("name") in {CAMERA_NAME, SENSOR_NAME} for element in x2_root.iter()):
        raise ValueError("runtime MJCF already contains the requested LiDAR names")

    compiler.set("meshdir", MESH_DIR_CONTAINER)
    extension = ET.Element("extension")
    extension_plugin = ET.SubElement(extension, "plugin", plugin=PLUGIN_NAME)
    instance = ET.SubElement(extension_plugin, "instance", name=SENSOR_INSTANCE)
    # The delivered 3.3.7 runtime requires an explicit instance for this
    # plugin to allocate sensor output dimensions.
    for key, value in (
        ("fov_h", "270"),
        ("fov_v", "60"),
        ("size", "90 5"),
        ("dis_range", "0.08 8"),
        ("n_step_update", "200"),
        ("sensor_data_types", "pos_b data"),
    ):
        ET.SubElement(instance, "config", key=key, value=value)
    x2_root.insert(0, extension)
    torso = one_named(x2_root, "body", "torso_link")
    ET.SubElement(
        torso,
        "camera",
        name=CAMERA_NAME,
        pos=" ".join(f"{value:.15g}" for value in CHEST_LIDAR_POS_M),
        xyaxes=CHEST_LIDAR_XYAXES,
        mode="fixed",
    )
    plugin = ET.SubElement(
        sensor,
        "plugin",
        name=SENSOR_NAME,
        instance=SENSOR_INSTANCE,
        objtype="camera",
        objname=CAMERA_NAME,
    )
    after_counts = count_model_elements(x2_root)
    if any(after_counts[key] != before_counts[key] for key in ("body", "joint", "geom", "actuator")):
        raise AssertionError("LiDAR overlay must not alter mechanical model element counts")
    if after_counts["camera"] != before_counts["camera"] + 1:
        raise AssertionError("LiDAR overlay must add exactly one camera")
    ET.indent(x2_tree, space="  ")
    x2_overlay = output_dir / "x2_lidar_overlay.xml"
    x2_tree.write(x2_overlay, encoding="utf-8", xml_declaration=True)

    scene_tree = ET.parse(scene_source)
    scene_root = scene_tree.getroot()
    model = one_named(scene_root, "model", "x2_mc_runtime")
    model.set("file", "/workspace/evidence/lidar_overlay/x2_lidar_overlay.xml")
    ET.indent(scene_tree, space="  ")
    scene_overlay = output_dir / "scene_lidar_overlay.xml"
    scene_tree.write(scene_overlay, encoding="utf-8", xml_declaration=True)

    robot_cfg = yaml.safe_load(ROBOT_YAML.read_text(encoding="utf-8"))
    sensor_cfg = robot_cfg.get("sensor")
    if not isinstance(sensor_cfg, dict):
        raise ValueError("runtime robot YAML lacks sensor configuration")
    frame_types = sensor_cfg.get("sensor_frame_type")
    sensor_names = sensor_cfg.get("sensor_name")
    if not isinstance(frame_types, list) or not isinstance(sensor_names, list):
        raise ValueError("runtime robot YAML sensor lists are malformed")
    frame_types.append(SENSOR_NAME)
    sensor_names.append([SENSOR_NAME])
    robot_overlay = output_dir / "robot_lidar_overlay.yaml"
    robot_overlay.write_text(yaml.safe_dump(robot_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    sim_cfg = yaml.safe_load(SIM_YAML.read_text(encoding="utf-8"))
    module = sim_cfg.get("MujocoSimModule")
    if not isinstance(module, dict):
        raise ValueError("runtime simulator YAML lacks MujocoSimModule")
    module["simulation_model"] = "/workspace/evidence/lidar_overlay/scene_lidar_overlay.xml"
    module["robot_model"] = "/workspace/evidence/lidar_overlay/robot_lidar_overlay.yaml"
    publishers = module.get("PublisherManager", {}).get("publisher_cfg_list")
    if not isinstance(publishers, list):
        raise ValueError("runtime simulator YAML lacks publisher configuration")
    if any(item.get("name") == "raycaster_lidar_ros" for item in publishers if isinstance(item, dict)):
        raise ValueError("runtime simulator YAML already contains a LiDAR publisher")
    publishers.append({
        "name": "raycaster_lidar_ros",
        "topic": SIM_POINTS_TOPIC,
        "enable": True,
        "executor": "data_state_publisher_thread",
        "frequency": 20,
        "print_interval": 0,
    })
    aimrt_pub = sim_cfg.get("aimrt", {}).get("channel", {}).get("pub_topics_options")
    if not isinstance(aimrt_pub, list):
        raise ValueError("runtime simulator YAML lacks ROS publisher topic configuration")
    aimrt_pub.append({"topic_name": SIM_POINTS_TOPIC, "enable_backends": ["ros2"]})
    sim_overlay = output_dir / "simulator_lidar_overlay.yaml"
    sim_overlay.write_text(yaml.safe_dump(sim_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema": "robotsim.issue80.raicom-lidar-overlay.v1",
        "classification": "SIMULATION_ONLY_SOURCE_DERIVED_CHEST_LIDAR_OVERLAY",
        "official_source": {
            "urdf": str(OFFICIAL_URDF),
            "sha256": OFFICIAL_URDF_SHA256,
            "joint": "lidar_chest_front_joint",
            "parent": "torso_link",
            "child": "lidar_chest_front_link",
            "xyz_m": list(CHEST_LIDAR_POS_M),
            "rpy_rad": [-math.pi / 2.0, 0.0, 0.0],
        },
        "runtime_sources": {
            "x2_xml": {"path": str(X2), "sha256": X2_SHA256},
            "robot_yaml": {"path": str(ROBOT_YAML), "sha256": ROBOT_YAML_SHA256},
            "sim_yaml": {"path": str(SIM_YAML), "sha256": SIM_YAML_SHA256},
            "scene_source": {"path": str(scene_source), "sha256": sha256(scene_source)},
        },
        "vendor_plugin": {
            "name": PLUGIN_NAME,
            "sensor_name": SENSOR_NAME,
            "sensor_instance": SENSOR_INSTANCE,
            "camera_name": CAMERA_NAME,
            "runtime_log_prior_state": "raycaster_lidar not found, disable raycaster publishing",
            "topic": SIM_POINTS_TOPIC,
            "topic_contract": "SIMULATION_ONLY raw simulator topic; not the official hardware PointCloud2 topic",
        },
        "mechanical_equivalence": {
            "before_counts": before_counts,
            "after_counts": after_counts,
            "unchanged": ["body", "joint", "geom", "actuator"],
            "added": ["one camera", "one plugin sensor", "one sensor mapping", "one publisher mapping"],
            "collision_geometry_changed": False,
            "inertia_changed": False,
            "joint_or_actuator_changed": False,
        },
        "generated": {
            "x2_xml": str(x2_overlay),
            "scene_xml": str(scene_overlay),
            "robot_yaml": str(robot_overlay),
            "sim_yaml": str(sim_overlay),
        },
        "limitations": [
            "The precompiled runtime's ROS message type for the raw simulator LiDAR topic is not yet observed.",
            "This overlay is not an assertion that the current delivery publishes the official hardware LiDAR topic.",
            "No MC, locomotion, model mechanics, collision geometry, or vendor source file is modified.",
        ],
    }
    manifest_path = output_dir / "LIDAR_OVERLAY_PROVENANCE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_overlay(args.output_dir, args.scene_source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
