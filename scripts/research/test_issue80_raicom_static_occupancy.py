from __future__ import annotations

import ast
import unittest
from pathlib import Path


class StaticOccupancyTests(unittest.TestCase):
    def test_builder_has_no_step_or_control_surface(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_static_occupancy.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertIn("mj_forward", calls)
        self.assertIn("measured_fullbody_radial_envelope_m", source)
        self.assertIn("Full-body static clearance", source)


if __name__ == "__main__":
    unittest.main()
