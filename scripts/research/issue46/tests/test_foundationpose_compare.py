#!/usr/bin/env python3
"""Focused metric tests for FoundationPose static comparison."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "perception/foundationpose_compare.py"
SPEC = importlib.util.spec_from_file_location("foundationpose_compare", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FoundationPoseCompareTests(unittest.TestCase):
    def test_identical_pose_has_zero_errors(self) -> None:
        metrics = MODULE.pose_metrics(np.eye(4), np.eye(4))
        self.assertAlmostEqual(metrics["translation_error_m"], 0.0)
        self.assertAlmostEqual(metrics["rotation_error_rad"], 0.0)
        self.assertAlmostEqual(metrics["cylinder_axis_undirected_error_rad"], 0.0)

    def test_cylinder_half_turn_is_axis_equivalent_but_not_full_rotation_equivalent(self) -> None:
        truth = np.eye(4)
        prediction = np.diag([-1.0, -1.0, 1.0, 1.0])
        metrics = MODULE.pose_metrics(truth, prediction)
        self.assertAlmostEqual(metrics["rotation_error_rad"], np.pi)
        self.assertAlmostEqual(metrics["cylinder_axis_undirected_error_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
