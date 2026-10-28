from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from issue80_lidar_map_localizer import (
    MAGIC,
    Pose2D,
    apply_map_to_odom,
    best_candidate,
    blend_map_to_odom,
    load_database,
    map_to_odom_from_correspondence,
    pose_residual,
    raw_ranges,
)
from issue80_lidar_raw_bridge import RAW_SCHEMA


class MapLocalizerTests(unittest.TestCase):
    def test_selects_static_signature_without_odom(self) -> None:
        ranges = np.array([[1.0, 2.0, np.nan, 4.0], [2.0, 3.0, np.nan, 5.0]], dtype=np.float32)
        candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1]], dtype=np.float32)
        index, score, matches = best_candidate(candidates, ranges, np.array([2.0, 3.0, np.nan, 5.0], dtype=np.float32), minimum_matches=2)
        self.assertEqual(index, 1)
        self.assertEqual(matches, 3)
        self.assertLess(score, 1e-6)

    def test_candidate_center_rejects_distant_signature_jump(self) -> None:
        ranges = np.array([[1.0] * 100, [1.0] * 100], dtype=np.float32)
        candidates = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        index, _, _ = best_candidate(candidates, ranges, np.array([1.0] * 100, dtype=np.float32), center=np.array([0.0, 0.0, 0.0]), max_translation_m=0.35, max_yaw_rad=0.25)
        self.assertEqual(index, 0)

    def test_raw_frame_preserves_ray_slots(self) -> None:
        frame = {"schema": RAW_SCHEMA, "xyz": [[3.0, 4.0, 0.0], None, ["nan", 0.0, 0.0]]}
        scan = raw_ranges(frame, 3)
        self.assertAlmostEqual(float(scan[0]), 5.0)
        self.assertTrue(np.isnan(scan[1]))
        self.assertTrue(np.isnan(scan[2]))

    def test_binary_map_header_loads_exact_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.bin"
            values = np.array([[1.0, 2.0, 0.0, 3.0, 4.0]], dtype="<f4")
            with path.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 2))
                stream.write(values.tobytes())
            poses, ranges = load_database(path)
            self.assertEqual(poses.tolist(), [[1.0, 2.0, 0.0]])
            self.assertEqual(ranges.tolist(), [[3.0, 4.0]])

    def test_map_to_odom_round_trip_preserves_correspondence(self) -> None:
        odom = Pose2D(2.0, -1.0, 0.2)
        map_pose = Pose2D(-1.5, -1.5, -0.1)
        transform = map_to_odom_from_correspondence(map_pose, odom)
        reconstructed = apply_map_to_odom(transform, odom)
        translation, yaw = pose_residual(map_pose, reconstructed)
        self.assertLess(translation, 1e-6)
        self.assertLess(abs(yaw), 1e-6)

    def test_bounded_lidar_blend_cannot_jump_one_grid_cell(self) -> None:
        current = Pose2D(0.0, 0.0, 0.0)
        observed = Pose2D(0.10, -0.10, 0.10)
        blended = blend_map_to_odom(current, observed, 0.15)
        self.assertAlmostEqual(blended.x_m, 0.015)
        self.assertAlmostEqual(blended.y_m, -0.015)
        self.assertAlmostEqual(blended.yaw_rad, 0.015)

    def test_lidar_residual_reports_translation_and_wrapped_yaw(self) -> None:
        distance, yaw = pose_residual(Pose2D(0.0, 0.0, 3.10), Pose2D(0.3, 0.4, -3.10))
        self.assertAlmostEqual(distance, 0.5)
        self.assertLess(abs(yaw), 0.1)


if __name__ == "__main__":
    unittest.main()
