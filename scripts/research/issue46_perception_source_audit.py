#!/usr/bin/env python3
"""Audit X2 RGB-D source contracts before perception-model integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


RAICOM_URDF = EVIDENCE_REPO / "simulation/mujoco/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/urdf/x2_whole_body.urdf"
DEMO_CONFIG = EVIDENCE_REPO / "reference/AgiBot-Community/agibot-X2-grasp-demo/src/x2_grasp/config/unified_grasp.yaml"
DEMO_LOCALIZER = EVIDENCE_REPO / "reference/AgiBot-Community/agibot-X2-grasp-demo/src/x2_grasp/src/localizer.cpp"
SIM_LOG = EVIDENCE_REPO / "simulation/mujoco/raicom2026/run-logs/mujoco-sim.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_joint(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") == "rgbd_head_front":
            origin = joint.find("origin")
            parent = joint.find("parent")
            child = joint.find("child")
            return {
                "joint": "rgbd_head_front",
                "parent": None if parent is None else parent.get("link"),
                "child": None if child is None else child.get("link"),
                "origin": {} if origin is None else dict(origin.attrib),
            }
    raise RuntimeError("RAICOM whole-body URDF lacks rgbd_head_front")


def yaml_topics(text: str) -> dict[str, str]:
    keys = ("rgb_image_topic", "depth_image_topic", "rgb_camera_info_topic", "depth_camera_info_topic", "target_frame")
    result = {}
    for key in keys:
        match = re.search(rf"^\s*{re.escape(key)}:\s*(\S.*?)\s*$", text, re.MULTILINE)
        if match:
            result[key] = match.group(1)
    return result


def main() -> None:
    required = (RAICOM_URDF, DEMO_CONFIG, DEMO_LOCALIZER, SIM_LOG)
    if any(not path.is_file() for path in required):
        raise SystemExit("required RAICOM or community-example source is missing")
    config_text = DEMO_CONFIG.read_text(encoding="utf-8")
    localizer_text = DEMO_LOCALIZER.read_text(encoding="utf-8")
    log_text = SIM_LOG.read_text(encoding="utf-8", errors="replace")
    topics = yaml_topics(config_text)
    optical_frames = sorted(set(re.findall(r'"(rgbd_head_front[^" ]*optical_frame)"', localizer_text)))
    ray_plugin_loaded = "mujoco.sensor.ray_caster_camera" in log_text
    ray_camera_active = bool(re.search(r"ray.?caster.?camera", log_text, re.IGNORECASE)) and "raycaster_lidar not found" not in log_text
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="x2-perception-source-audit")
    output = location.native_path
    report = {
        "schema": "robotsim.issue46.perception-source-audit.v1",
        "classification": "READ_ONLY_SOURCE_AND_RUNTIME_AUDIT",
        "mj_step_calls": 0,
        "sources": {
            "raicom_urdf": {"path": str(RAICOM_URDF), "sha256": sha256(RAICOM_URDF)},
            "community_demo_config": {"path": str(DEMO_CONFIG), "sha256": sha256(DEMO_CONFIG)},
            "community_demo_localizer": {"path": str(DEMO_LOCALIZER), "sha256": sha256(DEMO_LOCALIZER)},
            "raicom_log": {"path": str(SIM_LOG), "sha256": sha256(SIM_LOG)},
        },
        "official_sensor_link": camera_joint(RAICOM_URDF),
        "community_rgbd_contract": {
            "topics": topics,
            "optical_tf_candidates": optical_frames,
            "requires": ["synchronized RGB and depth", "RGB and depth CameraInfo", "timestamped camera optical TF to base_link"],
        },
        "observed_raicom_log": {
            "ray_camera_plugin_loaded": ray_plugin_loaded,
            "active_rgbd_topic_or_camera_info_observed": False,
            "note": "The archived log records the ray-caster plugin load but does not prove publication of the RGB-D topics consumed by the community localizer.",
        },
        "gates": {
            "SENSOR_LINK_SOURCE": "PASS",
            "COMMUNITY_RGBD_TOPIC_CONTRACT": "PASS",
            "LIVE_RGBD_TOPIC_DISCOVERY": "NOT_RUN",
            "CAMERA_INFO_AND_OPTICAL_TF": "NOT_RUN",
            "FOUNDATIONPOSE_INPUT_CONTRACT": "BLOCKED_BY_LIVE_RGBD",
            "ANYGRASP_POINT_CLOUD_CONTRACT": "BLOCKED_BY_LIVE_RGBD",
        },
        "next_safe_action": "Start the supported RAICOM simulator, read only the RGB/depth/CameraInfo/TF topics, and compare their frame_id and intrinsics to this contract before adding perception inference.",
    }
    pg.write_new_text(output / "report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": report["gates"], "mj_step_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
