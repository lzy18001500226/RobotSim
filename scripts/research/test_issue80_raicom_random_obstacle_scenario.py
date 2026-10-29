from __future__ import annotations

import ast
import unittest
from pathlib import Path

from issue80_raicom_random_obstacle_scenario import (
    ENDPOINT_EXCLUSION_M,
    OBJECT_SEPARATION_M,
    sample_obstacles,
)


class RandomScenarioTests(unittest.TestCase):
    def test_seed_is_reproducible_and_start_target_buffers_hold(self) -> None:
        first = sample_obstacles(seed=8, chair_count=1, barrier_count=1)
        second = sample_obstacles(seed=8, chair_count=1, barrier_count=1)
        self.assertEqual(first, second)
        flattened = [item for collection in first for item in collection]
        for collection in first:
            for item in collection:
                self.assertGreaterEqual(((item["x_m"] + 1.5) ** 2 + (item["y_m"] + 1.5) ** 2) ** 0.5, ENDPOINT_EXCLUSION_M)
                self.assertGreaterEqual(((item["x_m"] - 0.9) ** 2 + (item["y_m"] + 1.5) ** 2) ** 0.5, ENDPOINT_EXCLUSION_M)
        self.assertGreaterEqual(
            ((flattened[0]["x_m"] - flattened[1]["x_m"]) ** 2 + (flattened[0]["y_m"] - flattened[1]["y_m"]) ** 2) ** 0.5,
            OBJECT_SEPARATION_M,
        )

    def test_generator_has_no_control_or_step_surface(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_random_obstacle_scenario.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertIn("ET.ElementTree(tree).write", source)
        self.assertIn("ALIGNED_START_QUAT_WXYZ", source)
