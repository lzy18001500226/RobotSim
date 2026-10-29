from __future__ import annotations

import ast
import unittest
from pathlib import Path


class WorldTraceOverlaySourceTests(unittest.TestCase):
    def test_overlay_is_evidence_local_and_does_not_change_robot_or_mc(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_world_overlay.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("SIMULATION_ONLY_WORLD_PROJECTED_RAY_TRACE_OVERLAY", source)
        self.assertIn("output directory must contain only its governed reservation", source)
        self.assertIn("libmujoco.so.3.3.7", source)
        self.assertIn("world_xyz is simulator geometry ground truth", source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)


if __name__ == "__main__":
    unittest.main()
