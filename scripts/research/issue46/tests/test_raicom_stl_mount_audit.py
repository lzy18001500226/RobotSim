#!/usr/bin/env python3
"""Focused source-frame tests for the RAICOM bilateral STL mount audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = Path(__file__).resolve().parents[1] / "scene/raicom_stl_mount_audit.py"
SPEC = importlib.util.spec_from_file_location("raicom_stl_mount_audit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VENDOR_URDF = Path("/home/lzy18001500226/projects/RobotSim/reference/AgibotTech/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")


class RAICOMSTLMountAuditTests(unittest.TestCase):
    def test_pinned_vendor_mounts_identify_wrist_parent_and_origins(self) -> None:
        mounts = MODULE.official_mounts(VENDOR_URDF)
        self.assertEqual(mounts["right"]["parent_link"], "right_wrist_roll_link")
        self.assertEqual(mounts["right"]["xyz"], "0.00196 0.00035 -0.062")
        self.assertEqual(mounts["left"]["parent_link"], "left_wrist_roll_link")
        self.assertEqual(mounts["left"]["rpy"], "3.14 0.0 3.14")

    def test_left_remap_changes_hand_identifiers_but_not_other_values(self) -> None:
        root = ET.fromstring(
            "<body name='R_hand_narrow1_Link'><joint name='right_claw_joint'/>"
            "<geom name='visual' mesh='mesh_base_link' group='0' contype='0' conaffinity='0'/>"
            "<geom name='collision' mesh='mesh_base_link' group='1' contype='1' conaffinity='1'/></body>"
        )
        MODULE.remap_hand(root, "left")
        self.assertEqual(root.get("name"), "L_hand_narrow1_Link")
        self.assertEqual(root.find("joint").get("name"), "left_claw_joint")
        self.assertEqual(root.findall("geom")[0].get("mesh"), "stl_L_mesh_base_link")
        self.assertEqual(root.findall("geom")[0].get("group"), "0")
        self.assertEqual(root.findall("geom")[1].get("group"), str(MODULE.STL_COLLISION_RENDER_GROUP))


if __name__ == "__main__":
    unittest.main()
