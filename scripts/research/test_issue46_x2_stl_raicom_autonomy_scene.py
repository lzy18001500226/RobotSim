"""Focused static tests for the RAICOM-to-STL autonomy-scene boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "scripts/research/issue46_x2_stl_raicom_autonomy_scene.py"
SPEC = importlib.util.spec_from_file_location("issue46_x2_stl_raicom_autonomy_scene", TARGET)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RAICOMSTLAutonomySceneTests(unittest.TestCase):
    def test_generated_scene_selects_only_the_generated_stl_x2_overlay(self) -> None:
        payload = MODULE.scene_xml(
            x2_overlay=Path("/tmp/x2_stl_overlay.xml"),
            room=Path("/tmp/room.xml"),
            mug=Path("/tmp/mug.xml"),
            poses={
                "robot_start": {"xyz_m": [-1.5, -1.5, 0.0], "yaw_rad": 1.5707963267948966},
                "mug_source": {"xyz_m": [1.5, -1.4, 0.55], "yaw_rad": 0.0},
            },
        )
        root = ET.fromstring(payload)
        models = {item.get("name"): item.get("file") for item in root.findall("./asset/model")}
        self.assertEqual(models["x2_stl_runtime"], "/tmp/x2_stl_overlay.xml")
        self.assertNotIn("x2_runtime", models)
        attached = root.find("./worldbody/frame[@name='x2_start_frame']/attach")
        self.assertIsNotNone(attached)
        assert attached is not None
        self.assertEqual(attached.get("model"), "x2_stl_runtime")
        self.assertEqual(attached.get("prefix"), "x2_")

    def test_generated_scene_carries_only_the_existing_scoped_hand_exclusions(self) -> None:
        payload = MODULE.scene_xml(
            x2_overlay=Path("/tmp/x2_stl_overlay.xml"),
            room=Path("/tmp/room.xml"),
            mug=Path("/tmp/mug.xml"),
            poses={
                "robot_start": {"xyz_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0},
                "mug_source": {"xyz_m": [1.0, 2.0, 3.0], "yaw_rad": 0.0},
            },
        )
        root = ET.fromstring(payload)
        exclusions = {
            (item.get("body1"), item.get("body2"))
            for item in root.findall("./contact/exclude")
        }
        self.assertEqual(exclusions, set(MODULE.PREFIXED_INHERITED_EXCLUSIONS))
        self.assertEqual(len(exclusions), 16)


if __name__ == "__main__":
    unittest.main()
