"""Focused offline checks for the evidence-local RAICOM LiDAR overlay."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from issue80_raicom_lidar_overlay import (
    CAMERA_NAME,
    PLUGIN_NAME,
    SENSOR_NAME,
    SENSOR_INSTANCE,
    generate_overlay,
)


class RaicomLidarOverlayTests(unittest.TestCase):
    def test_generates_sensor_only_overlay_without_mechanical_delta(self) -> None:
        scene = (
            Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-80")
            / "raicom-vendor-table-heading-20261028T043433Z/local01_scene/scene_workspace_aligned_start_v1_native_x2_timestep_runtime.xml"
        )
        if not scene.is_file():
            self.skipTest("pinned RAICOM scene evidence unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lidar_overlay"
            manifest = generate_overlay(output, scene)
            x2 = ET.parse(output / "x2_lidar_overlay.xml").getroot()
            self.assertIsNotNone(next((item for item in x2.iter("camera") if item.get("name") == CAMERA_NAME), None))
            plugin = next((item for item in x2.iter("plugin") if item.get("name") == SENSOR_NAME), None)
            self.assertIsNotNone(plugin)
            self.assertEqual(plugin.get("instance"), SENSOR_INSTANCE)
            instance = next((item for item in x2.iter("instance") if item.get("name") == SENSOR_INSTANCE), None)
            self.assertIsNotNone(instance)
            self.assertEqual(
                manifest["mechanical_equivalence"]["before_counts"]["joint"],
                manifest["mechanical_equivalence"]["after_counts"]["joint"],
            )
            self.assertEqual(
                manifest["mechanical_equivalence"]["before_counts"]["geom"],
                manifest["mechanical_equivalence"]["after_counts"]["geom"],
            )
            stored = json.loads((output / "LIDAR_OVERLAY_PROVENANCE.json").read_text())
            self.assertEqual(stored["vendor_plugin"]["name"], PLUGIN_NAME)
            self.assertFalse(stored["mechanical_equivalence"]["collision_geometry_changed"])
            self.assertIn(
                'key="sensor_data_types" value="pos data"',
                (output / "x2_lidar_overlay.xml").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
