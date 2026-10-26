#!/usr/bin/env python3
"""Focused safety-boundary tests for the full-X2 cylinder trial runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import mujoco


MODULE_PATH = Path(__file__).with_name("issue46_full_x2_horizontal_cylinder_trial.py")
SPEC = importlib.util.spec_from_file_location("full_x2_trial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PositionLimitTests(unittest.TestCase):
    def test_live_viewer_has_a_nonzero_observation_delay(self) -> None:
        self.assertGreater(MODULE.LIVE_VIEWER_START_DELAY_S, 0.0)

    def test_live_viewer_requires_real_dynamics(self) -> None:
        MODULE.validate_live_viewer_options(execute_dynamics=True, live_viewer=True)
        MODULE.validate_live_viewer_options(execute_dynamics=False, live_viewer=False)
        with self.assertRaisesRegex(ValueError, "--live-viewer requires --execute-dynamics"):
            MODULE.validate_live_viewer_options(execute_dynamics=False, live_viewer=True)

    def test_roundoff_at_closed_endpoint_is_not_a_physical_violation(self) -> None:
        self.assertFalse(MODULE.outside_position_range(6.1e-21, -2.356194, 0.0))

    def test_substantive_limit_violation_still_fails(self) -> None:
        self.assertTrue(MODULE.outside_position_range(2.0e-8, -2.356194, 0.0))
        self.assertTrue(MODULE.outside_position_range(-2.356194 - 2.0e-8, -2.356194, 0.0))

    def test_three_endpoint_joints_are_initialized_inside_unchanged_ranges(self) -> None:
        xml = """
        <mujoco><compiler angle="radian"/><worldbody>
          <body pos="0 0 1"><joint name="left_knee_joint" type="hinge" range="0 2.408554"/><geom type="sphere" size="0.01"/></body>
          <body pos="1 0 1"><joint name="right_knee_joint" type="hinge" range="0 2.408554"/><geom type="sphere" size="0.01"/></body>
          <body pos="2 0 1"><joint name="left_elbow_joint" type="hinge" range="-2.356194 0"/><geom type="sphere" size="0.01"/></body>
        </worldbody><actuator>
          <motor name="motor_left_knee_joint" joint="left_knee_joint"/>
          <motor name="motor_right_knee_joint" joint="right_knee_joint"/>
          <motor name="motor_left_elbow_joint" joint="left_elbow_joint"/>
        </actuator></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        margin = MODULE.ENDPOINT_INITIALIZATION_MARGIN_RAD
        targets = {
            "left_knee_joint": margin,
            "right_knee_joint": margin,
            "left_elbow_joint": -margin,
        }
        for joint_name, target in targets.items():
            joint = model.joint(joint_name).id
            data.qpos[model.jnt_qposadr[joint]] = target
        source = {
            "left_knee_joint": {"lower": 0.0, "upper": 2.408554},
            "right_knee_joint": {"lower": 0.0, "upper": 2.408554},
            "left_elbow_joint": {"lower": -2.356194, "upper": 0.0},
        }
        result = MODULE.verify_endpoint_initialization(model, data, targets, source)
        self.assertTrue(result["pass"])
        self.assertEqual({item["joint"] for item in result["joints"]}, set(targets))
        elbow = model.joint("left_elbow_joint").id
        data.qpos[model.jnt_qposadr[elbow]] = -0.01
        targets["left_elbow_joint"] = -0.01
        guarded = MODULE.verify_endpoint_initialization(model, data, targets, source, -0.01)
        self.assertTrue(guarded["pass"])


if __name__ == "__main__":
    unittest.main()
