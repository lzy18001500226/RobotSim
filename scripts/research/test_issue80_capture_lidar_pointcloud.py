"""Focused static checks for the bounded read-only LiDAR capture tool."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


class LidarCaptureTests(unittest.TestCase):
    def test_capture_has_no_command_or_service_client_creation(self) -> None:
        path = Path(__file__).with_name("issue80_capture_lidar_pointcloud.py")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("create_subscription", calls)
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)
        self.assertIn("BLOCKED_EMPTY_OR_MISSING_POINTCLOUD", source)
        self.assertIn("PASS_NONEMPTY_POINTCLOUD", source)


if __name__ == "__main__":
    unittest.main()
