from __future__ import annotations

import ast
import unittest
from pathlib import Path


class SceneVariantTests(unittest.TestCase):
    def test_selects_static_or_runtime_scene_without_mutating_robot_overlay(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_lidar_scene_variant.py").read_text(encoding="utf-8")
        self.assertIn("scene_static_known_map.xml", source)
        self.assertIn("scene_lidar_overlay.xml", source)
        self.assertIn("unchanged_robot_files", source)
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("mj_step", calls)
        self.assertNotIn("create_publisher", calls)
