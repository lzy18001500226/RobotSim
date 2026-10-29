#!/usr/bin/env python3
"""Unit tests for source-preserving FoundationPose fixture construction."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("issue46_foundationpose_input.py")
SPEC = importlib.util.spec_from_file_location("foundationpose_input", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FoundationPoseInputTests(unittest.TestCase):
    def test_cylinder_obj_has_expected_mesh_shape(self) -> None:
        text = MODULE.cylinder_obj(0.035, 0.15, segments=16).decode("utf-8")
        self.assertEqual(sum(line.startswith("v ") for line in text.splitlines()), 34)
        self.assertEqual(sum(line.startswith("f ") for line in text.splitlines()), 48)

    def test_camera_to_object_truth_identity_camera(self) -> None:
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.3, -0.2, 1.1]
        capture = {"camera": {"T_world_camera": np.eye(4).tolist()}, "object_truth": {"T_world_object": object_pose.tolist()}}
        np.testing.assert_allclose(MODULE.camera_to_object_truth(capture), object_pose)

    def test_camera_to_object_prefers_optical_camera_frame(self) -> None:
        camera_mujoco = np.eye(4)
        camera_optical = np.diag([1.0, -1.0, -1.0, 1.0])
        object_pose = np.eye(4)
        object_pose[2, 3] = -0.5
        capture = {
            "camera": {"T_world_camera": camera_mujoco.tolist(), "T_world_optical_camera": camera_optical.tolist()},
            "object_truth": {"T_world_object": object_pose.tolist()},
        }
        self.assertAlmostEqual(float(MODULE.camera_to_object_truth(capture)[2, 3]), 0.5)

    def test_cylinder_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.cylinder_obj(0.0, 0.15)


if __name__ == "__main__":
    unittest.main()
