"""Focused guards for composing the approved full X2 scene with the RAICOM room."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = Path(__file__).resolve().parents[1] / "scene/raicom_autonomy_scene_review.py"
SPEC = importlib.util.spec_from_file_location("issue46_x2_stl_raicom_autonomy_scene", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


ENVIRONMENT = {
    "remove_from_approved_scene": ["raicom_table"],
    "attach_model_name": "raicom_room",
    "attach_prefix": "raicom_map_",
    "retain_body": "test_cylinder",
    "table_body_after_attach": "raicom_map_table",
}
START_POSE = {
    "xyz_m": [-1.5, -1.5, 0.68],
    "yaw_rad": 1.5707963267948966,
    "source": "test",
}


class RAICOMSTLAutonomySceneTests(unittest.TestCase):
    def test_composition_retains_approved_robot_and_object_but_replaces_table(self) -> None:
        source = """<mujoco><asset/><worldbody>
          <body name='pelvis'/><body name='R_omnipicker_base_link'/><body name='L_omnipicker_base_link'/>
          <body name='raicom_table'/><body name='test_cylinder'/><geom name='floor' type='plane'/>
        </worldbody></mujoco>"""
        room = "<mujoco model='room'><worldbody><body name='table'/></worldbody></mujoco>"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, room_path = root / "approved.xml", root / "room.xml"
            source_path.write_text(source, encoding="utf-8")
            room_path.write_text(room, encoding="utf-8")
            payload, record = MODULE.compose_scene(source_path, room_path, ENVIRONMENT, START_POSE)
        scene = ET.fromstring(payload)
        names = [body.get("name") for body in scene.findall(".//body")]
        self.assertIn("pelvis", names)
        self.assertIn("R_omnipicker_base_link", names)
        self.assertIn("L_omnipicker_base_link", names)
        self.assertIn("test_cylinder", names)
        self.assertNotIn("raicom_table", names)
        attached = scene.find("./worldbody/frame[@name='raicom_competition_room']/attach")
        self.assertIsNotNone(attached)
        assert attached is not None
        self.assertEqual(attached.get("model"), "raicom_room")
        self.assertEqual(record["approved_scene_environment_bodies_removed"], ["raicom_table"])
        pelvis = scene.find("./worldbody/body[@name='pelvis']")
        self.assertIsNotNone(pelvis)
        assert pelvis is not None
        self.assertEqual(pelvis.get("pos"), "-1.5 -1.5 0.68")
        self.assertEqual(pelvis.get("euler"), "0 0 1.57079632679")

    def test_composition_rejects_missing_approved_hand(self) -> None:
        source = "<mujoco><asset/><worldbody><body name='pelvis'/><body name='test_cylinder'/></worldbody></mujoco>"
        room = "<mujoco model='room'><worldbody/></mujoco>"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, room_path = root / "approved.xml", root / "room.xml"
            source_path.write_text(source, encoding="utf-8")
            room_path.write_text(room, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete X2-STL"):
                MODULE.compose_scene(source_path, room_path, ENVIRONMENT, START_POSE)


if __name__ == "__main__":
    unittest.main()
