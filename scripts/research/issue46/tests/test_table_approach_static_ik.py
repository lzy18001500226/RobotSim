#!/usr/bin/env python3
"""Unit tests for navigation-handoff static IK boundary parsing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "perception/table_approach_static_ik.py"
SPEC = importlib.util.spec_from_file_location("table_approach_static_ik", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TableApproachStaticIKTests(unittest.TestCase):
    def test_accepts_only_the_explicit_aligned_coordinate_contract(self) -> None:
        handoff = {"final odom": [0.85, -1.42, 0.65, 0.10]}
        route = {"runtime_coordinate_contract": {"numeric_coordinate_convention": "DERIVED_ALIGNED_SCENE_WORLD_RAW_ODOM_AFTER_LIVE_RESET_GATE"}}
        np.testing.assert_allclose(MODULE.handoff_pose(handoff, route), handoff["final odom"])
        with self.assertRaises(RuntimeError):
            MODULE.handoff_pose(handoff, {"runtime_coordinate_contract": {"numeric_coordinate_convention": "UNKNOWN"}})

    def test_pelvis_overlay_changes_only_pose_attributes(self) -> None:
        root = ET.fromstring("<mujoco><worldbody><body name='pelvis' pos='1 2 3' quat='1 0 0 0'/></worldbody></mujoco>")
        change = MODULE.set_pelvis_pose(root, np.array([0.8, -1.4, 0.65, 0.1]))
        pelvis = root.find(".//body")
        self.assertEqual(change["body"], "pelvis")
        self.assertEqual(pelvis.get("pos"), "0.8 -1.4 0.65")
        self.assertEqual(pelvis.get("euler"), "0 0 0.1")
        self.assertIsNone(pelvis.get("quat"))


if __name__ == "__main__":
    unittest.main()
