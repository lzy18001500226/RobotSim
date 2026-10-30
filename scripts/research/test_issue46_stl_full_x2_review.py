"""Regression coverage for complete hand-root replacement in the STL review."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = Path(__file__).with_name("issue46_stl_full_x2_review.py")
SPEC = importlib.util.spec_from_file_location("issue46_stl_full_x2_review", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullX2STLReviewTests(unittest.TestCase):
    def test_replacement_removes_old_root_geom_and_keeps_mount_transform(self) -> None:
        scene = ET.fromstring(
            """
            <mujoco><worldbody><body name='wrist'>
              <body name='R_omnipicker_base_link' pos='1 2 3' quat='1 0 0 0' childclass='mounted'>
                <inertial mass='9'/><geom name='old_base' mesh='old_mesh'/>
                <body name='old_link'><geom name='old_link_geom'/></body>
              </body>
            </body></worldbody><contact/></mujoco>
            """
        )
        source = ET.fromstring(
            """
            <mujoco><worldbody><body name='source_parent'>
              <body name='R_omnipicker_base_link'>
                <inertial mass='1'/><geom name='R_omnipicker_base_link__visual_0' mesh='mesh_base_link'/>
                <body name='R_hand_narrow1_Link'><joint name='right_claw_joint'/></body>
              </body>
            </body></worldbody></mujoco>
            """
        )
        MODULE.replace_hand(
            scene,
            source,
            target_name="R_omnipicker_base_link",
            bodies={"R_omnipicker_base_link", "R_hand_narrow1_Link"},
            joints={"right_claw_joint"},
            exclusions=(),
            left=False,
        )
        target = next(item for item in scene.findall(".//body") if item.get("name") == "R_omnipicker_base_link")
        self.assertEqual(target.attrib, {"name": "R_omnipicker_base_link", "pos": "1 2 3", "quat": "1 0 0 0", "childclass": "mounted"})
        self.assertEqual([item.get("name") for item in target.findall("geom")], ["R_omnipicker_base_link__visual_0"])
        self.assertIsNone(target.find("./body[@name='old_link']"))
        self.assertIsNotNone(target.find("./body[@name='R_hand_narrow1_Link']"))
        self.assertEqual(target.find("inertial").get("mass"), "1")
        audit = MODULE.assert_replaced_hand_payload(
            scene,
            target_name="R_omnipicker_base_link",
            bodies={"R_omnipicker_base_link", "R_hand_narrow1_Link"},
        )
        self.assertEqual(audit["direct_stl_geom"], "R_omnipicker_base_link__visual_0")
        self.assertEqual(audit["duplicate_or_missing_bodies"], {})


if __name__ == "__main__":
    unittest.main()
