#!/usr/bin/env python3
"""Focused tests for the source-preserving native STL hand runtime overlay."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = Path(__file__).with_name("issue46_raicom_stl_native_hand_overlay.py")
SPEC = importlib.util.spec_from_file_location("raicom_stl_native_hand_overlay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeHandOverlayTests(unittest.TestCase):
    def test_dynamic_layer_has_ten_followers_and_normalized_root_actuators(self) -> None:
        root = ET.fromstring("<mujoco><equality/><actuator/><sensor/><contact/></mujoco>")
        coefficients = {label: [0.0, 1.0, 0.0, 0.0, 0.0] for label in MODULE.FOLLOWER_COLUMNS.values()}
        record = MODULE.add_dynamic_hand_layer(root, coefficients)
        self.assertEqual(len(record["follower_equalities"]), 10)
        self.assertEqual(record["root_actuators"], ["motor_left_claw_joint", "motor_right_claw_joint"])
        actuators = {item.get("name"): item for item in root.findall("./actuator/*")}
        for name in record["root_actuators"]:
            self.assertEqual(actuators[name].tag, "position")
            self.assertEqual(actuators[name].get("gear"), "-1")
            self.assertEqual(actuators[name].get("ctrlrange"), "0 1")
        self.assertEqual(set(record["root_sensors"]), {"jointpos_left_claw_joint", "jointvel_left_claw_joint", "jointpos_right_claw_joint", "jointvel_right_claw_joint"})
        self.assertEqual(record["root_control"], {"kp_nm_per_rad": 0.5, "kv_nm_s_per_rad": 0.05, "force_limit_nm": 0.5})
        self.assertEqual(len(root.findall("./contact/exclude")), 2)

    def test_joint_named_actuator_mode_is_an_explicit_compatibility_variant(self) -> None:
        root = ET.fromstring("<mujoco><equality/><actuator/><sensor/><contact/></mujoco>")
        coefficients = {label: [0.0, 1.0, 0.0, 0.0, 0.0] for label in MODULE.FOLLOWER_COLUMNS.values()}
        record = MODULE.add_dynamic_hand_layer(root, coefficients, actuator_name_mode="joint")
        self.assertEqual(record["root_actuator_name_mode"], "joint")
        self.assertEqual(record["root_actuators"], ["left_claw_joint", "right_claw_joint"])

    def test_runtime_aliases_change_only_runtime_identifiers(self) -> None:
        root = ET.fromstring("<mujoco><actuator><position name='motor_left_claw_joint' joint='left_claw_joint'/></actuator><sensor><jointpos name='jointpos_left_claw_joint' joint='left_claw_joint'/></sensor><body name='L_hand_narrow1_Link'/></mujoco>")
        aliases = MODULE.apply_runtime_hand_aliases(root)
        actuator = root.find("./actuator/position")
        sensor = root.find("./sensor/jointpos")
        self.assertEqual(actuator.get("name"), "motor_L_claw_joint")
        self.assertEqual(actuator.get("joint"), "L_claw_joint")
        self.assertEqual(sensor.get("name"), "jointpos_L_claw_joint")
        self.assertEqual(sensor.get("joint"), "L_claw_joint")
        self.assertEqual(root.find("body").get("name"), "L_hand_narrow1_Link")
        self.assertEqual(aliases["right_claw_joint"], "R_claw_joint")

    def test_runtime_configs_add_only_one_hand_group(self) -> None:
        simulator = {
            "aimrt": {"channel": {"pub_topics_options": [], "sub_topics_options": []}},
            "MujocoSimModule": {"SubscriberManager": {"subscriber_cfg_list": []}, "PublisherManager": {"publisher_cfg_list": []}},
        }
        robot = {
            "actual": {"active_joint_frame_type": ["arm"], "active_joint_name": [["arm_joint"]], "active_joint_control_type": [["position"]]},
            "logical": {"nominal_configuration": [[0.0]]},
        }
        result = MODULE.enable_hand_channels(simulator, robot, scene_container_path="/workspace/evidence/scene.xml", robot_container_path="/workspace/evidence/robot.yaml")
        self.assertEqual(result["robot_model"]["actual"]["active_joint_frame_type"], ["arm", "hand"])
        self.assertEqual(result["robot_model"]["actual"]["active_joint_name"][-1], ["L_claw_joint", "R_claw_joint"])
        self.assertEqual(result["simulator"]["MujocoSimModule"]["simulation_model"], "/workspace/evidence/scene.xml")
        self.assertEqual(result["simulator"]["MujocoSimModule"]["SubscriberManager"]["subscriber_cfg_list"][0]["topic"], MODULE.HAND_COMMAND_TOPIC)
        with self.assertRaisesRegex(ValueError, "already contains topic"):
            MODULE.enable_hand_channels(result["simulator"], result["robot_model"], scene_container_path="x", robot_container_path="y")

    def test_container_path_rewrites_only_approved_mounts(self) -> None:
        self.assertEqual(MODULE.container_path(MODULE.RUNTIME_ROOT / "sim_mujoco"), "/workspace/raicom2026/sim_mujoco")
        self.assertEqual(MODULE.container_path(MODULE.REPO / "reference"), "/workspace/robotsim/reference")
        with self.assertRaisesRegex(ValueError, "outside approved"):
            MODULE.container_path(Path("/tmp/not-an-approved-asset"))


if __name__ == "__main__":
    unittest.main()
