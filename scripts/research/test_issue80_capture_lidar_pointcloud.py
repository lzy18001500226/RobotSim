"""Focused static checks for the bounded read-only LiDAR capture tool."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from issue80_lidar_map_obstacle_field import classify_map_frame_capture


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
        self.assertIn("classify_map_frame_capture", source)
        self.assertIn("--expected-frame-id", source)
        self.assertIn("--topic", source)
        self.assertIn("--save-all-finite-points", source)
        self.assertIn("MAX_RETAINED_FINITE_POINTS", source)

    def test_capture_requires_requested_map_frame(self) -> None:
        self.assertEqual(
            classify_map_frame_capture(message_count=1, nonempty_message_count=1, last_frame_id="map", expected_frame_id="map"),
            "PASS_NONEMPTY_POINTCLOUD",
        )
        self.assertEqual(
            classify_map_frame_capture(message_count=1, nonempty_message_count=1, last_frame_id="lidar", expected_frame_id="map"),
            "BLOCKED_POINTCLOUD_FRAME_MISMATCH",
        )

    def test_rviz_review_configuration_has_no_command_topic(self) -> None:
        source = Path(__file__).with_name("issue80_lidar_map_review.rviz").read_text(encoding="utf-8")
        self.assertIn("Fixed Frame: map", source)
        self.assertIn("/aima/sim/lidar/points", source)
        self.assertIn("/map_tf_distribution/localization_pose", source)
        self.assertNotIn("locomotion/velocity", source)
        self.assertNotIn("cmd_vel", source)


if __name__ == "__main__":
    unittest.main()
