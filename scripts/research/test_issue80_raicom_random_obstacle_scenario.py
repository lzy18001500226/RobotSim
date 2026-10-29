from __future__ import annotations

import ast
import unittest
from pathlib import Path

from issue80_raicom_random_obstacle_scenario import sample_obstacles


class RandomScenarioTests(unittest.TestCase):
    def test_seed_is_reproducible_and_start_target_buffers_hold(self) -> None:
        first = sample_obstacles(seed=17, chair_count=2, barrier_count=2)
        second = sample_obstacles(seed=17, chair_count=2, barrier_count=2)
        self.assertEqual(first, second)
        for collection in first:
            for item in collection:
                self.assertGreaterEqual(((item["x_m"] + 1.5) ** 2 + (item["y_m"] + 1.5) ** 2) ** 0.5, 0.75)
                self.assertGreaterEqual(((item["x_m"] - 0.9) ** 2 + (item["y_m"] + 1.5) ** 2) ** 0.5, 0.75)

    def test_generator_has_no_control_or_step_surface(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_random_obstacle_scenario.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertIn("ET.ElementTree(tree).write", source)
