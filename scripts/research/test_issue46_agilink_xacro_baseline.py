"""Focused regressions for the source-derived AGILINK/X2 tree generator."""
from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT = Path(__file__).with_name("issue46_agilink_xacro_baseline.py")
sys.path.insert(0, str(SCRIPT.parent))
import issue46_agilink_xacro_baseline as subject  # noqa: E402


class AgilinkSourceTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = subject.load_sources()

    def test_xacro_declares_eight_revolute_two_fixed_joints_per_hand(self):
        counts = subject.Counter(j.get("type") for j in subject.xacro_joints(self.sources).values())
        self.assertEqual(counts, {"revolute": 8, "fixed": 2})

    def test_supplied_display_entrypoint_does_not_instantiate_macro(self):
        self.assertFalse(any(e.tag.endswith("robot_urdf") for e in self.sources.display.iter()))
        self.assertEqual(self.sources.macro.attrib.get("name"), "robot_urdf")

    def test_all_left_and_right_joint_frames_match_x2(self):
        rows = subject.candidate_delta_rows(self.sources)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["frame_match"] for row in rows))

    def test_only_six_internal_joint_types_change_per_side(self):
        rows = subject.candidate_delta_rows(self.sources)
        delta = [row for row in rows if row["candidate_delta"] == "type fixed -> revolute"]
        self.assertEqual(len(delta), 12)
        for row in delta:
            self.assertEqual(row["x2_type"], "fixed")
            self.assertEqual(row["candidate_type"], "revolute")
            self.assertIn(subject.suffix_from_xacro_name(row["xacro_joint"]), subject.INTERNAL_REVOLUTES)

    def test_root_limits_and_mimic_are_preserved_for_both_sides(self):
        for side in ("L", "R"):
            candidate = subject.build_candidate_urdf(self.sources, side)
            joints = {j.get("name"): j for j in candidate.findall("joint")}
            narrow = joints[subject.x2_joint_name(side, "narrow1")]
            wide = joints[subject.x2_joint_name(side, "wide1")]
            self.assertEqual(narrow.get("type"), "revolute")
            self.assertEqual(wide.get("type"), "revolute")
            self.assertEqual(subject.source_limits(narrow), {"lower": "-1.00", "upper": "0.0", "effort": "0", "velocity": "0"})
            self.assertEqual(subject.source_limits(wide), {"lower": "0.0", "upper": "1.0", "effort": "0", "velocity": "0"})
            self.assertEqual(wide.find("mimic").get("joint"), narrow.get("name"))
            self.assertEqual(float(wide.find("mimic").get("multiplier")), -1.0)

    def test_candidate_keeps_x2_link_and_collision_geometry_inventory(self):
        for side in ("L", "R"):
            candidate = subject.build_candidate_urdf(self.sources, side)
            names = {link.get("name") for link in candidate.findall("link")}
            self.assertEqual(len(names), 11)
            self.assertIn(subject.x2_link_name(side, "base_link"), names)
            for link in candidate.findall("link"):
                original = self.sources.x2_links[link.get("name")]
                self.assertEqual(len(link.findall("visual")), len(original.findall("visual")))
                self.assertEqual(len(link.findall("collision")), len(original.findall("collision")))
                self.assertTrue(subject.inertial_equal(link, original))
                for mesh in link.findall(".//mesh"):
                    self.assertTrue(mesh.get("filename", "").startswith("file:///"))

    def test_source_tree_has_only_official_root_mimic_not_candidate_laws(self):
        urdf = subject.build_candidate_urdf(self.sources, "L")
        mjcf = subject.build_mjcf(urdf, "L")
        equality = mjcf.find("equality")
        self.assertEqual([e.tag for e in equality], ["joint"])
        self.assertEqual(equality.find("joint").get("polycoef"), "0 -1 0 0 0")
        self.assertIsNone(mjcf.find("tendon"))
        self.assertIsNone(mjcf.find("actuator"))
        self.assertEqual(len(mjcf.findall(".//exclude")), 0)

    def test_d_site_overlay_is_separate_and_contains_only_two_mates(self):
        urdf = subject.build_candidate_urdf(self.sources, "L")
        overlay = subject.build_mjcf(urdf, "L", d_closure_overlay=True)
        equality = overlay.find("equality")
        self.assertEqual([e.tag for e in equality], ["joint", "connect", "connect"])
        self.assertEqual(len(overlay.findall(".//site")), 4)
        self.assertIsNone(overlay.find("tendon"))
        self.assertIsNone(overlay.find("actuator"))

    def test_xacro_and_x2_source_hashes_match_pins(self):
        for key, path in (("agilink_xacro_zip", subject.XACRO_ZIP), ("agilink_3d_zip", subject.MODEL_ZIP),
                          ("agilink_source_zip", subject.SOURCE_ZIP), ("x2_urdf", subject.X2_URDF), ("op1_step", subject.STEP)):
            self.assertEqual(subject.sha_file(path), subject.EXPECTED[key])


if __name__ == "__main__":
    unittest.main()
