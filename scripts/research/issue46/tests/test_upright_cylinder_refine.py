#!/usr/bin/env python3
"""Unit tests for known-upright-cylinder geometric refinement."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "perception/upright_cylinder_refine.py"
SPEC = importlib.util.spec_from_file_location("upright_cylinder_refine", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UprightCylinderRefineTests(unittest.TestCase):
    def test_refines_synthetic_visible_top_rim(self) -> None:
        centre = np.array([0.4, -0.2, 0.7])
        radius = 0.035
        half_height = 0.15
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        top = np.column_stack((centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles), np.full_like(angles, centre[2] + half_height)))
        refined, support = MODULE.refine_upright_cylinder(top, centre + np.array([0.01, -0.01, 0.005]), half_height_m=half_height, top_band_m=0.02)
        np.testing.assert_allclose(refined, centre, atol=1e-10)
        self.assertEqual(len(support), 64)

    def test_rejects_insufficient_top_support(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.refine_upright_cylinder(np.zeros((10, 3)), np.zeros(3), half_height_m=0.15, top_band_m=0.02)


if __name__ == "__main__":
    unittest.main()
