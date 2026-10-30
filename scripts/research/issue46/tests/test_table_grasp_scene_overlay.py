#!/usr/bin/env python3
"""Focused source-preserving composition checks for the standing table scene."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "native_runtime/table_grasp_scene_overlay.py"
SPEC = importlib.util.spec_from_file_location("table_grasp_scene_overlay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TableGraspSceneOverlayTests(unittest.TestCase):
    def test_cylinder_keeps_the_source_table_relative_transform_when_table_is_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            approved = root / "approved.xml"
            host = root / "host.xml"
            room = root / "room.xml"
            approved.write_text("<mujoco><worldbody><body name='pelvis' pos='1 2 3'/><body name='raicom_table' pos='4 5 0'/><body name='test_cylinder' pos='4.2 5.3 0.7'><freejoint name='test_cylinder_free'/></body></worldbody></mujoco>", encoding="utf-8")
            host.write_text("<mujoco><worldbody><body name='pelvis' pos='10 20 30'/></worldbody></mujoco>", encoding="utf-8")
            room.write_text("<mujoco><asset/><worldbody><body name='table' pos='1 1 0'/></worldbody></mujoco>", encoding="utf-8")
            frame, cylinder, details = MODULE.table_frame_and_cylinder(approved, host, room, table_world_override=np.array((7.0, 8.0, 0.0)))
            np.testing.assert_allclose(frame, (6.0, 7.0, 0.0))
            np.testing.assert_allclose(np.fromstring(cylinder.get("pos"), sep=" "), (7.2, 8.3, 0.7))
            np.testing.assert_allclose(details["cylinder_table_translation_xyz_m"], (-6.0, -15.0, -27.0))

    def test_source_floor_preserves_vendor_texture_and_does_not_add_a_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            floor_source = root / "floor.xml"
            floor_source.write_text("<mujoco><asset><texture name='groundplane' type='2d' builtin='checker'/><material name='groundplane' texture='groundplane'/></asset><worldbody><geom name='floor' type='plane' size='0 0 0.05' material='groundplane'/></worldbody></mujoco>", encoding="utf-8")
            scene = b"<mujoco><asset/><worldbody><body name='pelvis'/></worldbody></mujoco>"
            composed, details = MODULE.inline_source_floor(scene, floor_source)
            tree = ET.ElementTree(ET.fromstring(composed))
            self.assertEqual(tree.getroot().find(".//geom[@name='raicom_floor']").get("material"), "raicom_floor_groundplane")
            self.assertEqual(tree.getroot().find(".//material[@name='raicom_floor_groundplane']").get("texture"), "raicom_floor_groundplane")
            self.assertEqual(len(tree.getroot().findall(".//body")), 1)
            self.assertEqual(details["floor_geom"], "raicom_floor")


if __name__ == "__main__":
    unittest.main()
