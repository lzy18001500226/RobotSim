import unittest
import xml.etree.ElementTree as ET

import numpy as np

from simulation.mujoco.m0_pick_place import (
    OBJECT_COLLISION_GEOMS,
    GRASP_WELD_NAME,
    SOURCE_TABLE_HALF_EXTENTS,
    SOURCE_TABLE_XY,
    TABLE_GAP_M,
    TARGET_TABLE_HALF_EXTENTS,
    TARGET_TABLE_BODY_XY,
    TARGET_TABLE_BODY_Z,
    TARGET_TABLE_SITE_OFFSET_Y_M,
    TARGET_TABLE_TOP_Z,
    AcceptanceMonitor,
    make_bottle_scene_xml,
)


SCENE_XML = """<mujoco>
  <include file="models/g1.xml"/>
  <worldbody>
    <camera name="scene_camera"/>
    <body name="table"><geom type="box" size="0.2 0.2 0.4"/><geom name="place_marker"/></body>
    <body name="red_cube"><freejoint/></body>
    <body name="green_box" pos="0.3 0 0.875">
      <freejoint name="box_joint"/>
      <geom name="box_geom"/>
      <site name="box_site"/>
    </body>
    <body name="distractor_0"><geom/></body>
  </worldbody>
  <equality><weld name="unused_upstream_weld"/></equality>
</mujoco>"""


class SceneTests(unittest.TestCase):
    def test_builds_bottle_and_second_table_without_weld(self):
        root = ET.fromstring(make_bottle_scene_xml(SCENE_XML, "/tmp/g1.xml"))
        worldbody = root.find("worldbody")
        bottle = worldbody.find("body[@name='green_box']")
        self.assertEqual(root.find("include").attrib["file"], "/tmp/g1.xml")
        self.assertIsNotNone(bottle.find("freejoint[@name='box_joint']"))
        self.assertIsNotNone(bottle.find("site[@name='box_site']"))
        self.assertEqual(
            [geom.attrib["name"] for geom in bottle.findall("geom")],
            list(OBJECT_COLLISION_GEOMS),
        )
        self.assertIsNone(worldbody.find("body[@name='red_cube']"))
        self.assertIsNone(worldbody.find("body[@name='distractor_0']"))
        weld = root.find(f"equality/weld[@name='{GRASP_WELD_NAME}']")
        self.assertIsNotNone(weld)
        self.assertEqual(weld.attrib["body1"], "right_wrist_yaw_link")
        self.assertEqual(weld.attrib["body2"], "green_box")
        self.assertEqual(weld.attrib["active"], "false")
        self.assertIsNone(root.find("equality/weld[@name='unused_upstream_weld']"))
        target_table = worldbody.find("body[@name='m0_target_table']")
        target_top = target_table.find("geom[@name='m0_target_table_top']")
        target_site = target_table.find("site[@name='m0_target_site']")
        self.assertEqual(
            target_table.attrib["pos"],
            f"{TARGET_TABLE_BODY_XY[0]:.3f} {TARGET_TABLE_BODY_XY[1]:.3f} {TARGET_TABLE_BODY_Z:.3f}",
        )
        self.assertAlmostEqual(TARGET_TABLE_TOP_Z, TARGET_TABLE_BODY_Z + 0.8)
        self.assertEqual(
            target_top.attrib["size"],
            f"{TARGET_TABLE_HALF_EXTENTS[0]:.3f} {TARGET_TABLE_HALF_EXTENTS[1]:.3f} 0.4",
        )
        self.assertEqual(
            target_site.attrib["pos"],
            f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.805",
        )
        target_marker = target_table.find("geom[@name='m0_target_marker']")
        self.assertEqual(
            target_marker.attrib["pos"],
            f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.803",
        )
        source_table = worldbody.find("body[@name='table']")
        source_top = source_table.find("geom[@name='m0_source_table_top']")
        self.assertEqual(
            source_table.attrib["pos"],
            f"{SOURCE_TABLE_XY[0]:.3f} {SOURCE_TABLE_XY[1]:.3f} 0",
        )
        self.assertEqual(
            source_top.attrib["size"],
            f"{SOURCE_TABLE_HALF_EXTENTS[0]:.3f} {SOURCE_TABLE_HALF_EXTENTS[1]:.3f} 0.4",
        )
        source_y = float(source_table.attrib["pos"].split()[1])
        target_y = float(target_table.attrib["pos"].split()[1])
        self.assertAlmostEqual(
            (source_y - SOURCE_TABLE_HALF_EXTENTS[1])
            - (target_y + TARGET_TABLE_HALF_EXTENTS[1]),
            TABLE_GAP_M,
        )
        self.assertEqual(
            worldbody.find("body[@name='table']/geom[@name='m0_source_table_top']").attrib["type"],
            "box",
        )

    def test_rejects_missing_scene_parts(self):
        with self.assertRaisesRegex(ValueError, "no worldbody"):
            make_bottle_scene_xml("<mujoco/>", "/tmp/g1.xml")
        with self.assertRaisesRegex(ValueError, "no green_box"):
            make_bottle_scene_xml(
                "<mujoco><include/><worldbody/></mujoco>", "/tmp/g1.xml"
            )


class AcceptanceMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = AcceptanceMonitor(
            initial_position=np.array([0.0, 0.0, 0.8]),
            target_xy=np.array([0.5, 0.0]),
            target_half_extents=np.array([0.2, 0.2]),
            dt=0.032,
        )

    def feed(self, time_s, position, **overrides):
        sample = {
            "sim_time": time_s,
            "position": np.asarray(position, dtype=float),
            "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "upright": True,
            "left_contact": False,
            "right_contact": False,
            "grasp_attached": False,
            "left_force_n": 0.0,
            "right_force_n": 0.0,
            "target_contact": False,
            "linear_speed": 0.0,
            "angular_speed": 0.0,
            "minimum_contact_distance": float("inf"),
            "state_finite": True,
            "state_extent": 1.0,
        }
        sample.update(overrides)
        self.monitor.update(**sample)

    def test_requires_bilateral_grasp_lift_transfer_release_and_stable_place(self):
        for index in range(5):
            self.feed(
                index * 0.032,
                [0.0, 0.0, 0.8],
                left_contact=True,
                right_contact=True,
                left_force_n=4.0,
                right_force_n=4.0,
            )
        self.feed(0.16, [0.0, 0.0, 0.86], left_contact=True, right_contact=True,
                  left_force_n=4.0, right_force_n=4.0)
        self.feed(0.192, [0.1, 0.0, 0.86], left_contact=True, right_contact=True,
                  left_force_n=4.0, right_force_n=4.0)
        self.feed(0.224, [0.3, 0.0, 0.86], left_contact=True, right_contact=True,
                  left_force_n=4.0, right_force_n=4.0)
        self.feed(0.256, [0.5, 0.0, 0.86], left_contact=True, right_contact=True,
                  left_force_n=4.0, right_force_n=4.0)
        for index in range(5):
            self.feed(
                0.288 + index * 0.032,
                [0.5, 0.0, 0.86],
                grasp_attached=True,
            )
        self.assertIsNone(self.monitor.stages["RELEASE"])
        for index in range(5):
            self.feed(0.448 + index * 0.032, [0.5, 0.0, 0.86])
        self.feed(
            0.608,
            [0.5, 0.0, 0.86],
            linear_speed=0.5,
            angular_speed=0.3,
        )
        for index in range(self.monitor.settling_frames_required):
            self.feed(
                0.640 + index * 0.032,
                [0.64, 0.10, 0.8],
                target_contact=True,
                upright=False,
            )

        result = self.monitor.result(
            {
                "position": [0.64, 0.10, 0.8],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "sim_time_s": 1.696,
                "linear_speed_m_s": 0.0,
                "angular_speed_rad_s": 0.0,
                "upright": False,
                "target_contact": True,
                "left_contact": False,
                "right_contact": False,
                "grasp_attached": False,
            }
        )
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(stage["passed"] for stage in result["task_stages"].values()))
        self.assertEqual(result["metrics"]["settling_frames_required"], 33)
        self.assertAlmostEqual(result["metrics"]["settling_duration_required_s"], 1.024)
        self.assertGreaterEqual(result["metrics"]["max_post_release_linear_speed_m_s"], 0.5)
        self.assertGreaterEqual(result["metrics"]["max_post_release_angular_speed_rad_s"], 0.3)
        self.assertFalse(result["metrics"]["final_upright"])

    def test_rejects_single_hand_grasp_and_detects_safety_failures(self):
        for index in range(6):
            self.feed(
                index * 0.032,
                [0.0, 0.0, 0.8],
                left_contact=True,
                left_force_n=9.0,
            )
        self.feed(0.2, [0.3, 0.0, 0.8])
        self.feed(
            0.232,
            [0.3, 0.0, 0.4],
            minimum_contact_distance=-0.03,
            state_finite=False,
        )
        result = self.monitor.result(None)
        self.assertFalse(result["passed"])
        self.assertFalse(result["task_stages"]["GRASP"]["passed"])
        self.assertFalse(result["safety_checks"]["FINITE_STATE"])
        self.assertFalse(result["safety_checks"]["NO_TELEPORT"])
        self.assertFalse(result["safety_checks"]["NO_EXCESSIVE_PENETRATION"])
        self.assertFalse(result["safety_checks"]["NO_DROP"])


if __name__ == "__main__":
    unittest.main()
