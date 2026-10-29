from __future__ import annotations

import math
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from issue80_lidar_map_obstacle_field import (
    MapPoint,
    apply_planar_transform,
    classify_map_frame_capture,
    decode_map_xyz,
    novel_map_points,
    static_surface_cells_from_database,
    surface_neighborhood,
)
from issue80_lidar_obstacle_layer import GridCell, GridSpec, Pose2D, RayCasterProfile
from issue80_lidar_map_localizer import MAGIC
from issue80_lidar_static_surface_audit import audit, calibration_from_localization_trace


PROFILE = RayCasterProfile(
    horizontal_rays=3,
    vertical_rays=1,
    horizontal_fov_rad=math.pi / 2.0,
    vertical_fov_rad=0.0,
    minimum_range_m=0.08,
    maximum_range_m=8.0,
    chest_x_offset_m=0.10,
)


class MapObstacleFieldTests(unittest.TestCase):
    def test_static_surface_uses_camera_offset_and_explicit_transform(self) -> None:
        grid = GridSpec(-2.0, -2.0, 0.10, 50, 50)
        candidates = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        ranges = np.array([[np.nan, 1.0, np.nan]], dtype=np.float32)
        surface = static_surface_cells_from_database(
            candidates, ranges, static_to_map=Pose2D(1.0, -0.5, 0.0), grid=grid, profile=PROFILE,
        )
        # Center ray is forward: base x + camera x offset + range, then map shift.
        self.assertIn(grid.world_to_cell(2.10, -0.50), surface)

    def test_static_surface_transform_applies_yaw(self) -> None:
        x, y = apply_planar_transform(Pose2D(1.0, 2.0, math.pi / 2.0), 2.0, 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 4.0)

    def test_static_map_point_is_not_new_and_offset_point_is_new(self) -> None:
        grid = GridSpec(-1.0, -1.0, 0.10, 40, 40)
        surface = {grid.world_to_cell(0.5, 0.0)}
        observed = [MapPoint(0.52, 0.02, 0.7), MapPoint(1.3, 0.0, 0.7)]
        novel = novel_map_points(observed, static_surface=surface, grid=grid, static_match_distance_m=0.15)
        self.assertEqual(novel, (observed[1],))

    def test_surface_neighborhood_does_not_escape_grid(self) -> None:
        grid = GridSpec(0.0, 0.0, 0.10, 3, 3)
        neighborhood = surface_neighborhood({GridCell(0, 0)}, grid=grid, maximum_distance_m=0.25)
        self.assertTrue(neighborhood)
        self.assertTrue(all(grid.contains(cell) for cell in neighborhood))

    def test_audit_reads_calibration_and_builds_surface_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "map.bin"
            values = [0.0, 0.0, 0.0] + [1.0] * 450
            with database.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 450))
                stream.write(struct.pack("<" + "f" * len(values), *values))
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({"calibration": [0.2, -0.1, 0.0]}) + "\n", encoding="utf-8")
            result = audit(database, trace, grid_resolution_m=0.10)
            self.assertEqual(result["status"], "PASS_STATIC_SURFACE_FIELD_BUILT")
            self.assertEqual(result["database"]["ray_count"], 450)
            self.assertGreater(result["static_surface_cell_count"], 0)
            self.assertFalse(result["pointcloud_map_frame_verified"])

    def test_missing_calibration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text(json.dumps({"not_calibration": True}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no static-signature-to-map calibration"):
                calibration_from_localization_trace(trace)

    def test_map_pointcloud_decoder_requires_map_frame_and_complete_layout(self) -> None:
        payload = struct.pack("<fff", 1.0, -2.0, 0.5) + struct.pack("<fff", float("nan"), 0.0, 0.0)
        points = decode_map_xyz(
            frame_id="map", data=payload, width=2, height=1, point_step=12,
            field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(points, (MapPoint(1.0, -2.0, 0.5),))
        with self.assertRaisesRegex(Exception, "frame_id"):
            decode_map_xyz(
                frame_id="lidar_chest_front", data=payload, width=2, height=1, point_step=12,
                field_offsets={"x": 0, "y": 4, "z": 8},
            )

    def test_capture_classification_requires_expected_frame(self) -> None:
        self.assertEqual(
            classify_map_frame_capture(message_count=1, nonempty_message_count=1, last_frame_id="map", expected_frame_id="map"),
            "PASS_NONEMPTY_POINTCLOUD",
        )
        self.assertEqual(
            classify_map_frame_capture(message_count=1, nonempty_message_count=1, last_frame_id="lidar", expected_frame_id="map"),
            "BLOCKED_POINTCLOUD_FRAME_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
