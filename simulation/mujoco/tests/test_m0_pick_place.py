import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from simulation.mujoco.m0_pick_place import (
    CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS,
    DEFAULT_REPRODUCTION_COMMAND,
    GRASP_PRELOAD_RAD,
    CANONICAL_TABLE_HALF_EXTENTS,
    CANONICAL_TABLE_XY,
    OBJECT_COLLISION_GEOMS,
    OBJECT_GEOM_SPECS,
    OBJECT_TOTAL_HEIGHT_M,
    RIGHT_HAND_CLOSED_POS,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_HAND_OPEN_POS,
    SOURCE_TABLE_TOP_Z,
    SOURCE_TABLE_XY,
    TARGET_TABLE_HALF_EXTENTS,
    TARGET_TABLE_SITE_OFFSET_Y_M,
    TARGET_TABLE_TOP_Z,
    AcceptanceMonitor,
    _projected_geom_half_extents_xy,
    _actuator_ordered_bias,
    _grasp_hold_target,
    check_projected_footprint,
    make_bottle_scene_xml,
    make_stock_hand_model_xml,
    main,
    run_demo,
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
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"/>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"/>
    <material name="groundplane" texture="groundplane"/>
  </asset>
  <equality><weld name="unused_upstream_weld"/></equality>
</mujoco>"""


class SceneTests(unittest.TestCase):
    def test_gravity_compensation_bias_is_mapped_by_actuator_joint(self):
        model = SimpleNamespace(
            actuator_trnid=np.array([[1, 0], [0, 0], [2, 0]], dtype=int),
            jnt_dofadr=np.array([2, 0, 1], dtype=int),
        )
        np.testing.assert_array_equal(
            _actuator_ordered_bias(model, np.array([11.0, 22.0, 33.0])),
            [11.0, 33.0, 22.0],
        )

    def test_grasp_hold_target_applies_bounded_preload_in_name_order(self):
        measured = {
            name: value
            for name, value in zip(
                RIGHT_HAND_JOINT_NAMES,
                (0.0, -0.4, -0.5, 0.6, 0.4, 0.7, 0.5),
            )
        }
        target = _grasp_hold_target(measured)
        measured_values = np.asarray([measured[name] for name in RIGHT_HAND_JOINT_NAMES])
        open_target = np.asarray(RIGHT_HAND_OPEN_POS)
        closed_target = np.asarray(RIGHT_HAND_CLOSED_POS)
        closure_direction = np.sign(closed_target - open_target)
        expected = np.clip(
            measured_values + closure_direction * GRASP_PRELOAD_RAD,
            np.minimum(open_target, closed_target),
            np.maximum(open_target, closed_target),
        )
        np.testing.assert_allclose(
            target,
            expected,
        )
        self.assertEqual(target.shape, (len(RIGHT_HAND_JOINT_NAMES),))

    def test_builds_normal_bottle_and_removes_all_equalities(self):
        root = ET.fromstring(make_bottle_scene_xml(SCENE_XML, "/tmp/g1.xml"))
        worldbody = root.find("worldbody")
        visual_global = root.find("visual/global")
        self.assertIsNotNone(visual_global)
        self.assertEqual(visual_global.attrib["offwidth"], "960")
        self.assertEqual(visual_global.attrib["offheight"], "720")
        bottle = worldbody.find("body[@name='green_box']")
        self.assertEqual(root.find("include").attrib["file"], "/tmp/g1.xml")
        self.assertIsNotNone(bottle.find("freejoint[@name='bottle_free']"))
        self.assertEqual(bottle.attrib["pos"], "0.300 -0.100 0.920")
        self.assertIsNotNone(bottle.find("site[@name='box_site']"))
        self.assertEqual(
            [geom.attrib["name"] for geom in bottle.findall("geom")],
            list(OBJECT_COLLISION_GEOMS),
        )
        self.assertEqual(
            [geom.attrib["type"] for geom in bottle.findall("geom")],
            ["cylinder", "ellipsoid", "cylinder", "cylinder"],
        )
        self.assertEqual(len(OBJECT_GEOM_SPECS), 4)
        self.assertAlmostEqual(sum(float(spec["mass"]) for spec in OBJECT_GEOM_SPECS), 0.57)
        bottle_com_z = sum(
            float(spec["mass"]) * float(spec["pos"].split()[2])
            for spec in OBJECT_GEOM_SPECS
        ) / sum(float(spec["mass"]) for spec in OBJECT_GEOM_SPECS)
        self.assertLessEqual(bottle_com_z, 0.01)
        self.assertAlmostEqual(2 * float(OBJECT_GEOM_SPECS[0]["size"].split()[0]), 0.070)
        self.assertAlmostEqual(OBJECT_TOTAL_HEIGHT_M, 0.2445)
        self.assertEqual(
            [float(spec["mass"]) for spec in OBJECT_GEOM_SPECS],
            [0.49, 0.05, 0.02, 0.01],
        )
        self.assertEqual(OBJECT_GEOM_SPECS[0]["rgba"], "0.12 0.52 0.82 1")
        self.assertEqual(OBJECT_GEOM_SPECS[0]["friction"], "1.4 0.02 0.001")
        self.assertEqual(OBJECT_GEOM_SPECS[0]["condim"], "4")
        self.assertEqual(OBJECT_GEOM_SPECS[-1]["rgba"], "0.10 0.16 0.21 1")
        self.assertIsNone(root.find("equality"))
        self.assertIsNone(worldbody.find("body[@name='red_cube']"))
        self.assertIsNone(worldbody.find("body[@name='distractor_0']"))

        table = worldbody.find("body[@name='m0_table']")
        table_top = table.find("geom[@name='m0_table_top']")
        target_site = table.find("site[@name='m0_target_site']")
        self.assertEqual(len(worldbody.findall("body")), 2)
        self.assertIsNone(worldbody.find("body[@name='m0_target_table']"))
        self.assertIsNone(worldbody.find(".//geom[@name='m0_target_marker']"))
        self.assertEqual(
            table.attrib["pos"],
            f"{CANONICAL_TABLE_XY[0]:.3f} {CANONICAL_TABLE_XY[1]:.3f} 0",
        )
        self.assertAlmostEqual(TARGET_TABLE_TOP_Z, 0.8)
        self.assertAlmostEqual(TARGET_TABLE_TOP_Z, SOURCE_TABLE_TOP_Z)
        self.assertEqual(
            table_top.attrib["size"],
            f"{CANONICAL_TABLE_HALF_EXTENTS[0]:.3f} {CANONICAL_TABLE_HALF_EXTENTS[1]:.3f} 0.025",
        )
        self.assertEqual(table_top.attrib["pos"], "0 0 0.775")
        self.assertEqual(len(table.findall("geom")), 5)
        self.assertEqual(
            {geom.attrib["name"] for geom in table.findall("geom")} - {"m0_table_top"},
            {
                "m0_table_leg_front_left",
                "m0_table_leg_front_right",
                "m0_table_leg_back_left",
                "m0_table_leg_back_right",
            },
        )
        self.assertEqual(target_site.attrib["pos"], f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.805")
        self.assertEqual(TARGET_TABLE_HALF_EXTENTS.tolist(), CANONICAL_TABLE_HALF_EXTENTS.tolist())
        self.assertIsNone(table.find("geom[@name='place_marker']"))
        self.assertEqual(table_top.attrib["rgba"], "0.6 0.4 0.2 1")
        skybox = root.find("asset/texture[@type='skybox']")
        self.assertEqual(skybox.attrib["rgb1"], "0.3 0.5 0.7")
        ground = root.find("asset/texture[@name='groundplane']")
        self.assertEqual(ground.attrib["rgb1"], "0.2 0.3 0.4")
        self.assertEqual(root.find("visual/headlight").attrib["diffuse"], "0.6 0.6 0.6")

    def test_rejects_missing_scene_parts(self):
        with self.assertRaisesRegex(ValueError, "no worldbody"):
            make_bottle_scene_xml("<mujoco/>", "/tmp/g1.xml")
        with self.assertRaisesRegex(ValueError, "no green_box"):
            make_bottle_scene_xml("<mujoco><include/><worldbody/></mujoco>", "/tmp/g1.xml")

    def test_restores_vendor_paired_stock_hands_without_scale_or_collision_fakes(self):
        candidate = (
            '<mujoco><asset>'
            '<mesh name="left_rubber_hand" file="left-old.stl"/>'
            '<mesh name="right_rubber_hand" file="right-old.stl"/>'
            '</asset><worldbody>'
            '<body name="left_wrist_yaw_link" pos="0.01 0 0">'
            '<geom mesh="left_rubber_hand"/><geom name="left_palm_pad"/>'
            '</body><body name="right_wrist_yaw_link" pos="0.01 0 0">'
            '<geom mesh="right_rubber_hand"/><geom name="right_palm_pad"/>'
            '<body name="right_hand_fake"><joint name="right_hand_fake_joint"/></body>'
            '</body></worldbody><actuator><motor name="body_0"/></actuator></mujoco>'
        )
        vendor = (
            '<mujoco><asset>'
            '<mesh name="left_rubber_hand" file="left_rubber_hand.STL"/>'
            '<mesh name="right_rubber_hand" file="right_rubber_hand.STL"/>'
            '</asset><worldbody>'
            '<body name="left_wrist_yaw_link" pos="0.046 0 0">'
            '<geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh" '
            'contype="0" conaffinity="0" mesh="left_rubber_hand"/></body>'
            '<body name="right_wrist_yaw_link" pos="0.046 0 0">'
            '<geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh" '
            'contype="0" conaffinity="0" mesh="right_rubber_hand"/></body>'
            '</worldbody></mujoco>'
        )
        with tempfile.TemporaryDirectory(prefix="robotsim-stock-g1-hands-") as temp:
            mesh_dir = Path(temp)
            (mesh_dir / "left_rubber_hand.STL").write_bytes(b"left mesh fixture")
            (mesh_dir / "right_rubber_hand.STL").write_bytes(b"right mesh fixture")
            root = ET.fromstring(make_stock_hand_model_xml(candidate, vendor, mesh_dir))

        for side, offset_y in (("left", 0.003), ("right", -0.003)):
            wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
            hand = wrist.find(f"geom[@mesh='{side}_rubber_hand']")
            self.assertEqual(wrist.attrib["pos"], "0.046 0 0")
            self.assertEqual(hand.attrib["pos"], f"0.0415 {offset_y} 0")
            self.assertEqual(hand.attrib["quat"], "1 0 0 0")
            self.assertEqual(hand.attrib["contype"], "0")
            self.assertEqual(hand.attrib["conaffinity"], "0")
            self.assertIsNone(wrist.find(f"geom[@name='{side}_palm_pad']"))
            self.assertFalse(any(body.attrib.get("name", "").startswith(f"{side}_hand_") for body in wrist.findall("body")))
            mesh = root.find(f"asset/mesh[@name='{side}_rubber_hand']")
            self.assertEqual(Path(mesh.attrib["file"]).name, f"{side}_rubber_hand.STL")
            self.assertNotIn("scale", mesh.attrib)
        self.assertEqual([actuator.attrib["name"] for actuator in root.findall("actuator/*")], ["body_0"])
        self.assertIsNone(root.find(".//joint[@name='right_hand_fake_joint']"))


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.target_xy = np.array([0.0, 0.0])
        self.target_half_extents = np.array([0.20, 0.18])
        self.margin = 0.03

    def test_cylinder_projection_tracks_orientation(self):
        upright = _projected_geom_half_extents_xy("cylinder", np.eye(3), np.array([0.075, 0.10]))
        laid_over = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        horizontal = _projected_geom_half_extents_xy(
            "cylinder", laid_over, np.array([0.075, 0.10])
        )
        np.testing.assert_allclose(upright, [0.075, 0.075])
        np.testing.assert_allclose(horizontal, [0.10, 0.075])

    def test_ellipsoid_projection_tracks_orientation(self):
        size = np.array([0.035, 0.035, 0.026])
        np.testing.assert_allclose(
            _projected_geom_half_extents_xy("ellipsoid", np.eye(3), size),
            [0.035, 0.035],
        )
        rotated = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            _projected_geom_half_extents_xy("ellipsoid", rotated, size),
            [0.026, 0.035],
        )

    def test_whole_bottle_inside_table_margin_passes(self):
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

    def test_center_inside_but_footprint_overhang_fails(self):
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

    def test_unsupported_or_off_table_bottle_fails(self):
        footprint = check_projected_footprint(
            [{"name": "body", "center_xy_m": [0.19, 0.0], "half_extents_xy_m": [0.02, 0.03]}],
            self.target_xy,
            self.target_half_extents,
            self.margin,
        )
        self.assertFalse(footprint["passed"], footprint)


def _sample(time_s, position, **overrides):
    sample = {
        "sim_time": time_s,
        "position": np.asarray(position, dtype=float),
        "quaternion_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "upright": True,
        "left_contact": False,
        "right_contact": False,
        "digit_contacts": {"thumb": False, "index": False, "middle": False},
        "digit_forces_n": {"thumb": 0.0, "index": 0.0, "middle": 0.0},
        "left_force_n": 0.0,
        "right_force_n": 0.0,
        "fingers_open": False,
        "target_contact": False,
        "linear_speed": 0.0,
        "angular_speed": 0.0,
        "minimum_contact_distance": 0.0,
        "state_finite": True,
        "state_extent": 1.0,
        "footprint_check": {"passed": True, "geometries": []},
    }
    sample.update(overrides)
    return sample


class AcceptanceMonitorTests(unittest.TestCase):
    def setUp(self):
        self.monitor = AcceptanceMonitor(
            initial_position=np.array([0.0, 0.0, 0.8]),
            target_xy=np.array([0.5, 0.0]),
            target_half_extents=np.array([0.2, 0.2]),
            dt=0.032,
        )

    def feed(self, time_s, position, **overrides):
        self.monitor.update(**_sample(time_s, position, **overrides))

    def test_ordered_single_hand_grasp_lift_transfer_open_release_and_settle(self):
        contact = {
            "right_contact": True,
            "digit_contacts": {"thumb": True, "index": True, "middle": False},
            "digit_forces_n": {"thumb": 1.2, "index": 1.4, "middle": 0.0},
            "right_force_n": 2.6,
        }
        for frame in range(5):
            self.feed(frame * 0.032, [0.0, 0.0, 0.8], **contact)
        self.assertIsNotNone(self.monitor.stages["GRASP"])

        self.monitor.carry_contact_required = True
        self.feed(0.16, [0.0, 0.0, 0.86], **contact)
        self.feed(0.192, [0.16, 0.0, 0.86], **contact)
        self.feed(0.224, [0.32, 0.0, 0.86], **contact)
        self.feed(0.256, [0.48, 0.0, 0.86], **contact)
        for step in range(64):
            self.monitor.record_physics_step(
                sim_time=0.162 + step * 0.002,
                position=np.array([0.0, 0.0, 0.86]),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                minimum_contact_distance=0.0,
                digit_contacts=contact["digit_contacts"],
                digit_forces_n=contact["digit_forces_n"],
            )
        self.assertIsNotNone(self.monitor.stages["LIFT"])
        self.assertIsNotNone(self.monitor.stages["TRANSFER"])

        self.monitor.carry_contact_required = False
        opened = {"fingers_open": True, "right_contact": False}
        for frame in range(5):
            self.feed(0.288 + frame * 0.032, [0.48, 0.0, 0.86], **opened)
        self.assertIsNotNone(self.monitor.stages["RELEASE"])

        self.feed(
            0.448,
            [0.48, 0.0, 0.86],
            fingers_open=True,
            target_contact=True,
        )
        self.assertIsNotNone(self.monitor.stages["PLACE"])
        self.assertIsNone(self.monitor.stages["SETTLE"])
        for frame in range(10):
            self.feed(
                0.480 + frame * 0.032,
                [0.48, 0.0, 0.86],
                fingers_open=True,
                target_contact=True,
            )
        self.feed(
            0.800,
            [0.48, 0.0, 0.86],
            fingers_open=True,
            target_contact=True,
            linear_speed=0.04,
        )
        self.assertEqual(self.monitor.settling_frames, 0)
        self.assertIsNone(self.monitor.stages["SETTLE"])
        for frame in range(self.monitor.settling_frames_required):
            self.feed(
                0.832 + frame * 0.032,
                [0.48, 0.0, 0.86],
                fingers_open=True,
                target_contact=True,
            )
        final_sample = {
            "position": [0.48, 0.0, 0.86],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "sim_time_s": 1.856,
            "linear_speed_m_s": 0.0,
            "angular_speed_rad_s": 0.0,
            "target_contact": True,
            "left_contact": False,
            "right_contact": False,
            "digit_contacts": {"thumb": False, "index": False, "middle": False},
            "fingers_open": True,
        }
        result = self.monitor.result(final_sample)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["reproduction_command"], DEFAULT_REPRODUCTION_COMMAND)
        self.assertEqual(
            [stage["name"] for stage in result["stage_order"]],
            ["GRASP", "LIFT", "TRANSFER", "RELEASE", "PLACE", "SETTLE"],
        )
        self.assertGreater(
            result["task_stages"]["SETTLE"]["sim_time_s"],
            result["task_stages"]["PLACE"]["sim_time_s"],
        )
        self.assertGreaterEqual(
            result["metrics"]["continuous_stable_duration_observed_max_s"],
            self.monitor.minimum_stable_duration_s,
        )
        self.assertTrue(result["safety_checks"]["RIGHT_HAND_MULTI_FINGER_CONTACT"])
        self.assertTrue(result["safety_checks"]["FREE_PHYSICS_SETTLE"])
        self.assertTrue(result["safety_checks"]["LEFT_ARM_CLEAR"])
        self.assertTrue(result["safety_checks"]["NO_RUNTIME_EQUALITY_CARRY"])
        self.assertTrue(result["safety_checks"]["TRUE_FINGER_OPEN_RELEASE"])
        self.assertTrue(result["safety_checks"]["WHOLE_OBJECT_FOOTPRINT_INSIDE_TARGET"])

    def test_thumb_alone_does_not_count_and_left_contact_fails(self):
        for frame in range(6):
            self.feed(
                frame * 0.032,
                [0.0, 0.0, 0.8],
                digit_contacts={"thumb": True, "index": False, "middle": False},
                digit_forces_n={"thumb": 9.0, "index": 0.0, "middle": 0.0},
            )
        self.assertIsNone(self.monitor.stages["GRASP"])
        self.feed(
            0.2,
            [0.0, 0.0, 0.8],
            left_contact=True,
            left_force_n=2.0,
        )
        result = self.monitor.result(None)
        self.assertFalse(result["passed"])
        self.assertFalse(result["safety_checks"]["LEFT_ARM_CLEAR"])

    def test_carry_allows_brief_solver_chatter_but_records_physics_step_contact_fraction(self):
        self.monitor.carry_contact_required = True
        for step in range(100):
            credible = step != 25
            self.monitor.record_physics_step(
                sim_time=(step + 1) * 0.002,
                position=np.array([0.0, 0.0, 0.8]),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                minimum_contact_distance=0.0,
                digit_contacts={"thumb": credible, "index": True, "middle": False},
                digit_forces_n={"thumb": 1.0 if credible else 0.0, "index": 2.0, "middle": 0.0},
            )
        result = self.monitor.result(None)
        self.assertTrue(result["safety_checks"]["RIGHT_HAND_CONTACT_DURING_CARRY"])
        self.assertEqual(result["metrics"]["max_carry_contact_gap_physics_steps"], 1)
        self.assertAlmostEqual(result["metrics"]["carry_contact_fraction"], 0.99)
        self.assertFalse(result["metrics"]["carry_contact_lost"])

    def test_carry_fails_after_sustained_physics_step_contact_loss(self):
        self.monitor.carry_contact_required = True
        for step in range(10 + CARRY_CONTACT_MAX_GAP_PHYSICS_STEPS + 1):
            credible = step < 10
            self.monitor.record_physics_step(
                sim_time=(step + 1) * 0.002,
                position=np.array([0.0, 0.0, 0.8]),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                minimum_contact_distance=0.0,
                digit_contacts={"thumb": credible, "index": True, "middle": True},
                digit_forces_n={"thumb": 1.0 if credible else 0.0, "index": 2.0, "middle": 2.0},
            )
        result = self.monitor.result(None)
        self.assertFalse(result["safety_checks"]["RIGHT_HAND_CONTACT_DURING_CARRY"])
        self.assertIn("right Dex3 multi-finger contact was lost during carry", result["metrics"]["failure_reasons"])

    def test_release_requires_open_fingers_and_no_hand_contact(self):
        self.monitor.stages["GRASP"] = {"sim_time_s": 0.1, "control_frame": 1}
        self.monitor.stages["LIFT"] = {"sim_time_s": 0.2, "control_frame": 2}
        self.monitor.stages["TRANSFER"] = {"sim_time_s": 0.3, "control_frame": 3}
        for frame in range(11):
            self.feed(
                0.4 + frame * 0.032,
                [0.5, 0.0, 0.86],
                fingers_open=frame >= 3,
                right_contact=frame < 6,
            )
        self.assertIsNotNone(self.monitor.stages["RELEASE"])
        self.assertAlmostEqual(self.monitor.stages["RELEASE"]["sim_time_s"], 0.4 + 10 * 0.032)


class IntegrityTests(unittest.TestCase):
    def test_physics_step_translation_rotation_and_penetration_are_measured(self):
        monitor = AcceptanceMonitor(
            initial_position=np.array([0.0, 0.0, 0.8]),
            target_xy=np.zeros(2),
            target_half_extents=np.ones(2),
            dt=0.032,
        )
        monitor.record_physics_step(
            sim_time=0.002,
            position=np.array([0.0, 0.0, 0.8]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            minimum_contact_distance=0.0,
        )
        angle = 0.2
        monitor.record_physics_step(
            sim_time=0.004,
            position=np.array([0.012, 0.0, 0.8]),
            quaternion_wxyz=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
            minimum_contact_distance=-0.03,
        )
        result = monitor.result(None)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_translation_m"], 0.012)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_angular_jump_rad"], angle)
        self.assertAlmostEqual(result["metrics"]["max_physics_step_penetration_m"], 0.03)
        self.assertFalse(result["safety_checks"]["NO_TELEPORT"])
        self.assertFalse(result["safety_checks"]["NO_EXCESSIVE_PENETRATION"])

    def test_active_rollout_has_no_direct_bottle_qpos_write(self):
        source_path = Path(__file__).parents[1] / "m0_pick_place.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        run_demo = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_demo"
        )
        writes = []
        for node in ast.walk(run_demo):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and ast.unparse(target.value).endswith(
                    ".data.qpos"
                ):
                    writes.append(ast.unparse(target))
        self.assertEqual(writes, [])
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("GRASP_WELD_NAME", source)
        self.assertNotIn("record_weld_event", source)


class ActiveEntryPointTests(unittest.TestCase):
    @patch("simulation.mujoco.m0_pick_place._run_paired_dex3_episode")
    @patch("simulation.mujoco.m0_pick_place._run_stock_hand_visual_audit")
    def test_demo_uses_stock_hand_audit_without_dex3_fallback(
        self, stock_audit, dex3_episode
    ):
        args = SimpleNamespace()
        stock_audit.return_value = {"state": "BLOCKED"}

        result = run_demo(args)

        self.assertEqual(result, {"state": "BLOCKED"})
        stock_audit.assert_called_once_with(args)
        dex3_episode.assert_not_called()

    @patch("simulation.mujoco.m0_pick_place._runtime_identity", return_value={})
    @patch("simulation.mujoco.m0_pick_place.parse_args")
    @patch("simulation.mujoco.m0_pick_place.run_demo", return_value={"state": "BLOCKED"})
    def test_blocked_result_records_exact_reproduction_command(
        self, run_demo_mock, parse_args_mock, runtime_identity_mock
    ):
        reproduction_command = (
            "ROBOTSIM_M0_OUTPUT_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/"
            "issue-43-g1-stock-hands-canonical ./scripts/run_m0_pick_place.sh"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "m0_result.json"
            parse_args_mock.return_value = SimpleNamespace(
                run_id="m0-test-blocked",
                output_json=result_path,
                reproduction_command=reproduction_command,
                hand_roll_deg=20.0,
                grasp_hold_frames=2,
                lift_frames=15,
                lift_height_m=0.16,
                transfer_frames=30,
                lower_frames=15,
            )

            exit_code = main([])
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(result["state"], "BLOCKED")
        self.assertEqual(result["reproduction_command"], reproduction_command)
        run_demo_mock.assert_called_once()
        runtime_identity_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
