#!/usr/bin/env python3
"""Focused safety-boundary tests for the full-X2 cylinder trial runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("issue46_full_x2_horizontal_cylinder_trial.py")
SPEC = importlib.util.spec_from_file_location("full_x2_trial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PositionLimitTests(unittest.TestCase):
    def test_roundoff_at_closed_endpoint_is_not_a_physical_violation(self) -> None:
        self.assertFalse(MODULE.outside_position_range(6.1e-21, -2.356194, 0.0))

    def test_substantive_limit_violation_still_fails(self) -> None:
        self.assertTrue(MODULE.outside_position_range(2.0e-8, -2.356194, 0.0))
        self.assertTrue(MODULE.outside_position_range(-2.356194 - 2.0e-8, -2.356194, 0.0))


if __name__ == "__main__":
    unittest.main()
