import ast
from pathlib import Path
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
    OBJECT_GEOM_SPECS,
    AcceptanceMonitor,
    _projected_geom_half_extents_xy,
    check_projected_footprint,
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
        self.assertEqual(
            [geom.attrib["type"] for geom in bottle.findall("geom")],
            ["cylinder"] * len(OBJECT_GEOM_SPECS),
        )
        self.assertEqual(OBJECT_GEOM_SPECS[0]["type"], "cylinder")
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


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.target_xy = np.array([0.0, 0.0])
        self.target_half_extents = np.array([0.20, 0.18])
        self.margin = 0.03

    def test_cylinder_projection_tracks_orientation(self):
        upright = _projected_geom_half_extents_xy(
            "cylinder", np.eye(3), np.array([0.075, 0.10])
        )
        laid_over = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        horizontal = _projected_geom_half_extents_xy(
            "cylinder", laid_over, np.array([0.075, 0.10])
        )
        np.testing.assert_allclose(upright, [0.075, 0.075])
        np.testing.assert_allclose(horizontal, [0.10, 0.075])

    def test_fully_inside_bottle_footprint_passes(self):
        footprint = check_projected_footprint(
            [
                {"name": "body", "center_xy_m": [0.0, 0.0], "half_extents_xy_m": [0.10, 0.075]},
                {"name": "cap", "center_xy_m": [0.0, 0.0], "half_extents_xy_m": [0.03, 0.03]},
            ],
            self.target_xy,
            self.target_half_extents,
            self.margin,
        )
        self.assertTrue(footprint["passed"], footprint)

    def test_center_inside_but_overhanging_footprint_fails(self):
        center = np.array([0.15, 0.0])
        self.assertLessEqual(abs(center[0]), self.target_half_extents[0] - self.margin)
        footprint = check_projected_footprint(
            [{"name": "body", "center_xy_m": center, "half_extents_xy_m": [0.06, 0.05]}],
            self.target_xy,
            self.target_half_extents,
            self.margin,
        )
        self.assertFalse(footprint["passed"], footprint)
        self.assertFalse(footprint["geometries"][0]["inside_with_margin"])

    def test_unsupported_off_table_object_fails_placement(self):
        footprint = check_projected_footprint(
            [{"name": "body", "center_xy_m": [0.19, 0.0], "half_extents_xy_m": [0.02, 0.03]}],
            self.target_xy,
            self.target_half_extents,
            self.margin,
        )
        self.assertFalse(footprint["passed"], footprint)

        monitor = AcceptanceMonitor(
            initial_position=np.array([0.0, 0.0, 0.8]),
            target_xy=self.target_xy,
            target_half_extents=self.target_half_extents,
            dt=0.032,
        )
        monitor.stages["TRANSFER"] = {"sim_time_s": 1.0, "control_frame": 1}
        monitor.stages["RELEASE"] = {"sim_time_s": 1.1, "control_frame": 2}
        _feed_monitor(monitor, 1.2, [0.19, 0.0, 0.8], footprint, target_contact=False)
        result = monitor.result(
            {
                "position": [0.19, 0.0, 0.8],
                "target_contact": False,
            }
        )
        self.assertFalse(result["task_stages"]["PLACE"]["passed"])
        self.assertFalse(result["safety_checks"]["TARGET_TABLE_CONTACT"])
        self.assertFalse(result["safety_checks"]["WHOLE_OBJECT_FOOTPRINT_INSIDE_TARGET"])


def _feed_monitor(monitor, time_s, position, footprint_check, target_contact=True):
    monitor.update(
        sim_time=time_s,
        position=np.asarray(position, dtype=float),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        upright=False,
        left_contact=False,
        right_contact=False,
        grasp_attached=False,
        left_force_n=0.0,
        right_force_n=0.0,
        target_contact=target_contact,
        linear_speed=0.0,
        angular_speed=0.0,
        minimum_contact_distance=float("inf"),
        state_finite=True,
        state_extent=1.0,
        footprint_check=footprint_check,
    )


class IntegrityTests(unittest.TestCase):
    def test_physics_step_and_weld_event_jumps_are_measured(self):
        monitor = AcceptanceMonitor(
            initial_position=np.zeros(3),
            target_xy=np.zeros(2),
            target_half_extents=np.ones(2),
            dt=0.032,
        )
        monitor.record_physics_step(
            sim_time=0.002,
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            minimum_contact_distance=float("inf"),
        )
        angle = 0.2
        monitor.record_physics_step(
            sim_time=0.004,
            position=np.array([0.012, 0.0, 0.0]),
            quaternion_wxyz=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
            minimum_contact_distance=-0.03,
        )
        monitor.record_weld_event(
            {
                "event": "activation",
                "pose_jump_m": {"translation": 0.006, "rotation_rad": 0.1},
            }
        )
        result = monitor.result(None)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_translation_m"], 0.012)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_angular_jump_rad"], angle)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_penetration_m"], 0.03)
        self.assertFalse(result["safety_checks"]["NO_TELEPORT"])
        self.assertFalse(result["safety_checks"]["NO_WELD_EVENT_SNAP"])
        self.assertFalse(result["safety_checks"]["NO_EXCESSIVE_PENETRATION"])

    def test_rollout_source_has_no_bottle_or_full_qpos_writes(self):
        source_path = Path(__file__).parents[1] / "m0_pick_place.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        allowed_reset_functions = {
            "reset",
            "reset_state",
            "initialize_state",
            "_reset_object_state",
        }
        violations = []

        def is_allowed_qpos_write(function, index):
            return function in allowed_reset_functions or index in (
                "sim.left_arm_qpos_adr",
                "sim.right_arm_qpos_adr",
            )

        def enclosing_function(node):
            current = parents.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current.name
                current = parents.get(current)
            return None

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if not ast.unparse(target.value).endswith(".data.qpos"):
                    continue
                index = ast.unparse(target.slice)
                function = enclosing_function(node)
                if not is_allowed_qpos_write(function, index):
                    violations.append(f"{function}: {ast.unparse(target)}")
        self.assertEqual(violations, [])

        reset_fixture = ast.parse(
            "def reset_state(sim, adr, state):\n    sim.data.qpos[adr:adr + 7] = state\n"
        )
        fixture_parents = {
            child: parent
            for parent in ast.walk(reset_fixture)
            for child in ast.iter_child_nodes(parent)
        }
        fixture_target = next(
            node.targets[0]
            for node in ast.walk(reset_fixture)
            if isinstance(node, ast.Assign)
        )
        fixture_assignment = fixture_parents[fixture_target]
        fixture_function = fixture_parents[fixture_assignment].name
        self.assertTrue(is_allowed_qpos_write(fixture_function, "adr: adr + 7"))
        self.assertFalse(is_allowed_qpos_write("run_demo", "object_qpos_adr"))


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
        sample.setdefault("footprint_check", {"passed": True, "geometries": []})
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
