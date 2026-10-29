from __future__ import annotations

import ast
import unittest
from pathlib import Path


class RandomObstacleOverlayTests(unittest.TestCase):
    def test_keeps_robot_overlay_and_separates_static_runtime_scenes(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_random_obstacle_lidar_overlay.py").read_text(encoding="utf-8")
        self.assertIn("scene_static_known_map.xml", source)
        self.assertIn("scene_runtime_unknown_obstacles.xml", source)
        self.assertIn("sim_only_unknown_chair.xml", source)
        self.assertIn("assets.remove(unknown)", source)
        self.assertIn("x2_lidar_overlay.xml", source)
        self.assertIn("forbidden_delta", source)
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
