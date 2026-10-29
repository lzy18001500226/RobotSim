from __future__ import annotations

import ast
import unittest
from pathlib import Path


class DiagnosticSourceTests(unittest.TestCase):
    def test_diagnostic_is_subscription_only(self) -> None:
        source = Path(__file__).with_name("issue80_lidar_obstacle_diagnostic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("create_subscription", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertNotIn("publish", calls)
        self.assertIn("--background-capture", source)


if __name__ == "__main__":
    unittest.main()
