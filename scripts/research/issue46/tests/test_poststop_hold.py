#!/usr/bin/env python3
"""Focused tests for the reviewed-trajectory dynamic constraint fit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "native_runtime/poststop_hold.py"
SPEC = importlib.util.spec_from_file_location("poststop_hold", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPO = Path(__file__).resolve().parents[4]
TRAJECTORY = REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic/trajectory.csv"


class PostStopHoldTests(unittest.TestCase):
    def test_follower_fit_covers_all_expected_table_columns(self) -> None:
        coefficients = MODULE.trajectory_coefficients(TRAJECTORY)
        self.assertEqual(set(coefficients), {"wide1", "narrow3", "narrow_loop", "wide3", "wide_loop"})
        self.assertLessEqual(MODULE.maximum_fit_residual(TRAJECTORY, coefficients), 0.006)

    def test_side_relationships_cover_one_root_and_five_followers(self) -> None:
        for side in ("left", "right"):
            self.assertIn(side, MODULE.ROOTS)
            self.assertEqual(set(MODULE.FOLLOWERS[side]), {"wide1", "narrow3", "narrow_loop", "wide3", "wide_loop"})


if __name__ == "__main__":
    unittest.main()
