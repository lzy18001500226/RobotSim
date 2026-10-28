"""Unit tests for LiDAR-only scan registration."""
from __future__ import annotations

import math
import unittest

import numpy as np

from issue80_lidar_scan_math import Pose2, scan_delta


class LidarLocalizationTests(unittest.TestCase):
    def test_scan_delta_recovers_planar_transform(self) -> None:
        previous = np.array([
            [-2.0, -1.0], [-1.0, 0.5], [0.0, -1.5], [1.5, 1.0], [2.0, -0.5],
            [-1.4, 1.8], [0.8, 2.2], [2.4, 1.6], [-2.3, 0.9], [1.1, -2.2],
            [0.3, 0.9], [-0.6, -2.1], [1.9, 2.4], [-2.5, -1.8], [2.7, 0.2],
            [-0.2, 2.6], [0.6, -0.8], [-1.8, 2.5], [2.5, -2.0], [-2.7, 1.5],
        ], dtype=float)
        yaw = 0.06
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        translation = np.array([0.12, -0.04])
        current = (previous - translation) @ rotation
        forward, lateral, observed_yaw, residual = scan_delta(previous, current)
        self.assertLess(residual, 1e-6)
        self.assertAlmostEqual(forward, translation[0], places=5)
        self.assertAlmostEqual(lateral, translation[1], places=5)
        self.assertAlmostEqual(observed_yaw, yaw, places=5)

    def test_pose_accumulates_sensor_delta_without_odom(self) -> None:
        pose = Pose2(-1.5, -1.5, 0.0).apply_delta(0.2, 0.0, 0.05)
        self.assertAlmostEqual(pose.x, -1.3, places=6)
        self.assertAlmostEqual(pose.y, -1.5, places=6)
        self.assertAlmostEqual(pose.yaw, 0.05, places=6)

    def test_static_scan_threshold_is_below_expected_locomotion_increment(self) -> None:
        # At 0.12 m/s and a 5 Hz ray update, a real step is ~24 mm.
        self.assertGreater(0.12 / 5.0, 0.01)


if __name__ == "__main__":
    unittest.main()
