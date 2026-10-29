from __future__ import annotations

import struct
import unittest
from pathlib import Path
import ast
import math

from issue80_lidar_map_obstacle_field import MapPoint
from issue80_lidar_obstacle_layer import GridCell, GridSpec
from issue80_lidar_obstacle_pipeline import (
    MapFrameObstaclePipeline,
    ObstaclePipelineConfig,
    PipelineStatus,
)


def cloud(points: list[tuple[float, float, float]]) -> bytes:
    return b"".join(struct.pack("<fff", *point) for point in points)


class ObstaclePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridSpec(-1.0, -1.0, 0.10, 21, 21)
        self.config = ObstaclePipelineConfig(
            grid=self.grid,
            static_surface=frozenset({GridCell(2, 2)}),
            static_route_blocked=frozenset(),
            start=GridCell(1, 10),
            goal=GridCell(19, 10),
            static_match_distance_m=0.15,
            obstacle_inflation_m=0.21,
            confirmation_frames=2,
            minimum_confirmed_cluster_cells=3,
        )

    def test_known_static_point_does_not_replan(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        result = pipeline.observe(
            sequence=1, frame_id="map", data=cloud([(-0.75, -0.75, 0.5)]),
            width=1, height=1, point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(result.status, PipelineStatus.CLEAR_DIRECT_PATH)

    def test_new_chair_requires_confirmation_then_generates_detour(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        kwargs = {
            "frame_id": "map", "data": cloud([(0.02, 0.02, 0.5), (0.02, 0.12, 0.5), (0.12, 0.12, 0.5)]),
            "width": 3, "height": 1, "point_step": 12, "field_offsets": {"x": 0, "y": 4, "z": 8},
        }
        first = pipeline.observe(sequence=1, **kwargs)
        self.assertEqual(first.status, PipelineStatus.WAITING_FOR_CONFIRMATION)
        second = pipeline.observe(sequence=2, **kwargs)
        self.assertEqual(second.status, PipelineStatus.DETOUR_READY)
        self.assertEqual(second.detour_cells[0], self.config.start)
        self.assertEqual(second.detour_cells[-1], self.config.goal)
        self.assertTrue(any(cell.y != self.config.start.y for cell in second.detour_cells))

    def test_confirmed_single_cell_noise_does_not_replan(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        kwargs = {
            "frame_id": "map", "data": cloud([(0.02, 0.02, 0.5)]),
            "width": 1, "height": 1, "point_step": 12, "field_offsets": {"x": 0, "y": 4, "z": 8},
        }
        pipeline.observe(sequence=1, **kwargs)
        result = pipeline.observe(sequence=2, **kwargs)
        self.assertEqual(result.status, PipelineStatus.WAITING_FOR_CONFIRMATION)

    def test_measured_map_background_rejects_static_repeat_and_confirms_cluster(self) -> None:
        config = ObstaclePipelineConfig(
            grid=self.grid,
            static_surface=frozenset(),
            static_route_blocked=frozenset(),
            start=GridCell(1, 10),
            goal=GridCell(19, 10),
            static_match_distance_m=0.02,
            obstacle_inflation_m=0.21,
            confirmation_frames=2,
            minimum_confirmed_cluster_cells=3,
            map_background_points=(MapPoint(-0.75, -0.75, 0.5),),
        )
        pipeline = MapFrameObstaclePipeline(config)
        static = dict(
            frame_id="map", data=cloud([(-0.75, -0.75, 0.5)]),
            width=1, height=1, point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(pipeline.observe(sequence=1, **static).status, PipelineStatus.CLEAR_DIRECT_PATH)
        chair = dict(
            frame_id="map", data=cloud([(0.02, 0.02, 0.5), (0.02, 0.12, 0.5), (0.12, 0.12, 0.5)]),
            width=3, height=1, point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(pipeline.observe(sequence=2, **chair).status, PipelineStatus.WAITING_FOR_CONFIRMATION)
        self.assertEqual(pipeline.observe(sequence=3, **chair).status, PipelineStatus.DETOUR_READY)

    def test_off_route_cluster_does_not_claim_a_detour(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        kwargs = {
            "frame_id": "map", "data": cloud([(-0.75, 0.75, 0.5), (-0.65, 0.75, 0.5), (-0.55, 0.75, 0.5)]),
            "width": 3, "height": 1, "point_step": 12, "field_offsets": {"x": 0, "y": 4, "z": 8},
        }
        pipeline.observe(sequence=1, **kwargs)
        result = pipeline.observe(sequence=2, **kwargs)
        self.assertEqual(result.status, PipelineStatus.OBSTACLE_CLEAR_OF_DIRECT_PATH)
        self.assertEqual(result.detour_map_waypoints, ())

    def test_known_static_occupancy_produces_a_static_detour(self) -> None:
        config = ObstaclePipelineConfig(
            grid=self.grid,
            static_surface=frozenset(),
            static_route_blocked=frozenset({GridCell(10, 10)}),
            start=GridCell(1, 10),
            goal=GridCell(19, 10),
            static_match_distance_m=0.15,
            obstacle_inflation_m=0.21,
            confirmation_frames=2,
            minimum_confirmed_cluster_cells=3,
        )
        result = MapFrameObstaclePipeline(config).observe(
            sequence=1, frame_id="map", data=cloud([]), width=0, height=1,
            point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(result.status, PipelineStatus.DETOUR_READY)
        self.assertEqual(result.detour_cells[0], config.start)
        self.assertEqual(result.detour_cells[-1], config.goal)

    def test_background_registration_rejects_reset_frame_drift_without_erasing_new_obstacle(self) -> None:
        background = tuple(
            MapPoint(-0.8 + 0.08 * column, -0.8 + 0.08 * row, 0.1)
            for row in range(10) for column in range(10)
        )
        config = ObstaclePipelineConfig(
            grid=GridSpec(-2.0, -2.0, 0.10, 41, 41),
            static_surface=frozenset(), static_route_blocked=frozenset(),
            start=GridCell(2, 20), goal=GridCell(38, 20),
            static_match_distance_m=0.02, obstacle_inflation_m=0.21,
            confirmation_frames=2, minimum_confirmed_cluster_cells=3,
            map_background_points=background,
        )
        # Simulate the measured small Reset-to-Reset rigid drift: the current
        # labelled map cloud is translated/rotated but has no novel geometry.
        angle, tx, ty = 0.03, -0.04, 0.04
        cosine, sine = math.cos(angle), math.sin(angle)
        drifted = [
            (tx + cosine * point.x_m - sine * point.y_m,
             ty + sine * point.x_m + cosine * point.y_m, point.z_m)
            for point in background
        ]
        pipeline = MapFrameObstaclePipeline(config)
        first = pipeline.observe(
            sequence=1, frame_id="map", data=cloud(drifted), width=len(drifted), height=1,
            point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(first.status, PipelineStatus.CLEAR_DIRECT_PATH)
        self.assertIsNotNone(first.background_registration)
        self.assertLess(float(first.background_registration["trimmed_xy_score_m"]), 0.03)

    def test_wrong_frame_fails_closed_without_route(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        result = pipeline.observe(
            sequence=1, frame_id="lidar_chest_front", data=cloud([(0.02, 0.02, 0.5)]),
            width=1, height=1, point_step=12, field_offsets={"x": 0, "y": 4, "z": 8},
        )
        self.assertEqual(result.status, PipelineStatus.BLOCKED_INVALID_OBSERVATION)
        self.assertEqual(result.detour_cells, ())

    def test_duplicate_scan_fails_closed(self) -> None:
        pipeline = MapFrameObstaclePipeline(self.config)
        kwargs = {
            "frame_id": "map", "data": cloud([(0.02, 0.02, 0.5)]),
            "width": 1, "height": 1, "point_step": 12, "field_offsets": {"x": 0, "y": 4, "z": 8},
        }
        pipeline.observe(sequence=1, **kwargs)
        result = pipeline.observe(sequence=1, **kwargs)
        self.assertEqual(result.status, PipelineStatus.BLOCKED_INVALID_OBSERVATION)

    def test_pipeline_has_no_runtime_control_surface(self) -> None:
        source = Path(__file__).with_name("issue80_lidar_obstacle_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(any(name.startswith(("rclpy", "mujoco")) for name in imports))
        self.assertNotIn("create_publisher", calls)
        self.assertNotIn("create_client", calls)


if __name__ == "__main__":
    unittest.main()
