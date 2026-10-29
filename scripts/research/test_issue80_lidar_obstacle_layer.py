from __future__ import annotations

import math
import json
import struct
import tempfile
import unittest
from pathlib import Path

from issue80_lidar_obstacle_layer import (
    GridCell,
    GridSpec,
    NovelObstacleTracker,
    NovelReturn,
    ObstacleLayerError,
    Pose2D,
    RayCasterProfile,
    direct_path_intersects_blocked,
    inflate_cells,
    novel_returns_from_ranges,
    path_world_points,
    plan_a_star,
    raw_xyz_ranges,
    connected_cell_components,
    simplify_grid_path,
)
from issue80_lidar_map_localizer import MAGIC
from issue80_lidar_obstacle_replay import grid_from_candidates, replay


PROFILE = RayCasterProfile(
    horizontal_rays=3,
    vertical_rays=1,
    horizontal_fov_rad=math.pi / 2.0,
    vertical_fov_rad=0.0,
    minimum_range_m=0.08,
    maximum_range_m=8.0,
    chest_x_offset_m=0.10,
)


class LidarObstacleLayerTests(unittest.TestCase):
    def test_raw_ranges_keep_invalid_slots(self) -> None:
        values = raw_xyz_ranges([[0.0, 0.0, -1.0], None, ["nan", 0.0, 0.0]], PROFILE)
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[1:], (None, None))

    def test_shorter_return_becomes_map_frame_novel_obstacle(self) -> None:
        returns = novel_returns_from_ranges(
            (None, 1.0, None), (None, 2.0, None),
            robot_pose=Pose2D(1.0, -2.0, 0.0), novelty_margin_m=0.20, profile=PROFILE,
        )
        self.assertEqual(len(returns), 1)
        item = returns[0]
        self.assertEqual(item.ray_index, 1)
        self.assertAlmostEqual(item.x_m, 2.10)
        self.assertAlmostEqual(item.y_m, -2.0)

    def test_static_or_small_range_residual_is_not_new_obstacle(self) -> None:
        self.assertEqual(
            novel_returns_from_ranges((None, 1.90, None), (None, 2.0, None), robot_pose=Pose2D(0, 0, 0), novelty_margin_m=0.20, profile=PROFILE),
            (),
        )
        self.assertEqual(
            novel_returns_from_ranges((None, 2.3, None), (None, 2.0, None), robot_pose=Pose2D(0, 0, 0), novelty_margin_m=0.20, profile=PROFILE),
            (),
        )

    def test_unknown_static_ray_can_be_new_obstacle(self) -> None:
        detected = novel_returns_from_ranges(
            (None, 1.0, None), (None, None, None),
            robot_pose=Pose2D(0, 0, 0), novelty_margin_m=0.20, profile=PROFILE,
        )
        self.assertEqual(len(detected), 1)
        self.assertIsNone(detected[0].expected_range_m)

    def test_tracker_requires_unique_frames_and_persistent_hit(self) -> None:
        grid = GridSpec(-1.0, -1.0, 0.10, 20, 20)
        tracker = NovelObstacleTracker(grid, confirmation_frames=2)
        point = NovelReturn(0, 0.02, 0.02, 1.0, 2.0)
        self.assertEqual(tracker.update(10, [point]), frozenset())
        self.assertEqual(tracker.update(11, [point]), frozenset({GridCell(10, 10)}))
        with self.assertRaisesRegex(ObstacleLayerError, "strictly advance"):
            tracker.update(11, [point])

    def test_tracker_drops_single_frame_stale_return(self) -> None:
        grid = GridSpec(-1.0, -1.0, 0.10, 20, 20)
        tracker = NovelObstacleTracker(grid, confirmation_frames=2)
        point = NovelReturn(0, 0.02, 0.02, 1.0, 2.0)
        self.assertEqual(tracker.update(1, [point]), frozenset())
        self.assertEqual(tracker.update(2, []), frozenset())
        self.assertEqual(tracker.update(3, [point]), frozenset())

    def test_inflated_chair_blocks_direct_path_and_astar_detours(self) -> None:
        grid = GridSpec(-1.0, -1.0, 0.10, 21, 21)
        start, goal = GridCell(1, 10), GridCell(19, 10)
        inflated = inflate_cells({GridCell(10, 10)}, grid=grid, radius_m=0.21)
        self.assertTrue(direct_path_intersects_blocked(start, goal, blocked=inflated))
        path = plan_a_star(start, goal, grid=grid, blocked=inflated)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertTrue(all(cell not in inflated for cell in path))
        self.assertTrue(any(cell.y != 10 for cell in path))
        self.assertEqual(len(path_world_points(path, grid)), len(path))
        simplified = simplify_grid_path(path, blocked=inflated)
        self.assertEqual(simplified[0], start)
        self.assertEqual(simplified[-1], goal)
        self.assertLess(len(simplified), len(path))
        self.assertTrue(all(
            not direct_path_intersects_blocked(a, b, blocked=inflated)
            for a, b in zip(simplified, simplified[1:])
        ))

    def test_diagonal_cannot_cut_touched_obstacle_corner(self) -> None:
        grid = GridSpec(0.0, 0.0, 0.10, 3, 3)
        with self.assertRaisesRegex(ObstacleLayerError, "no collision-free path"):
            plan_a_star(GridCell(0, 0), GridCell(1, 1), grid=grid, blocked={GridCell(1, 0), GridCell(0, 1)})

    def test_blocked_start_or_goal_fails_closed(self) -> None:
        grid = GridSpec(0.0, 0.0, 0.10, 3, 3)
        with self.assertRaisesRegex(ObstacleLayerError, "start or goal"):
            plan_a_star(GridCell(0, 0), GridCell(2, 2), grid=grid, blocked={GridCell(0, 0)})

    def test_components_separate_scattered_noise_from_chair_cluster(self) -> None:
        components = connected_cell_components({GridCell(1, 1), GridCell(1, 2), GridCell(2, 2), GridCell(9, 9)})
        self.assertEqual(sorted(len(component) for component in components), [1, 3])

    def test_static_replay_preserves_no_false_confirmed_obstacle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "map.bin"
            values = [0.0, 0.0, 0.0] + [2.0] * 450
            with database.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 450))
                stream.write(struct.pack("<" + "f" * len(values), *values))
            point = [0.0, 0.0, -2.0]
            trace = root / "raw.jsonl"
            trace.write_text("\n".join(json.dumps({"schema": "robotsim.issue80.lidar-raw.v1", "sequence": seq, "xyz": [point] * 450}) for seq in (0, 1)) + "\n", encoding="utf-8")
            result = replay(trace, database, novelty_margin_m=0.20, confirmation_frames=2, grid_resolution_m=0.10)
            self.assertEqual(result["status"], "PASS_STATIC_REPLAY_NO_CONFIRMED_NOVEL_OBSTACLE")
            self.assertEqual(result["confirmed_obstacle_cells"], [])

    def test_static_replay_confirms_persistent_new_occluder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "map.bin"
            values = [0.0, 0.0, 0.0] + [2.0] * 450
            with database.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 450))
                stream.write(struct.pack("<" + "f" * len(values), *values))
            scan = [[0.0, 0.0, -1.0] if ray == 225 else [0.0, 0.0, -2.0] for ray in range(450)]
            trace = root / "raw.jsonl"
            trace.write_text("\n".join(json.dumps({"schema": "robotsim.issue80.lidar-raw.v1", "sequence": seq, "xyz": scan}) for seq in (0, 1)) + "\n", encoding="utf-8")
            result = replay(trace, database, novelty_margin_m=0.20, confirmation_frames=2, grid_resolution_m=0.10)
            self.assertEqual(result["status"], "PASS_STATIC_REPLAY_CONFIRMED_NOVEL_OBSTACLES")
            self.assertTrue(result["confirmed_obstacle_cells"])

    def test_grid_derivation_includes_all_candidate_boundaries(self) -> None:
        import numpy as np
        grid = grid_from_candidates(np.array([[-1.5, -2.0, 0.0], [1.0, -0.8, 0.0]], dtype=np.float32), resolution_m=0.10)
        self.assertTrue(grid.contains(grid.world_to_cell(-1.5, -2.0)))
        self.assertTrue(grid.contains(grid.world_to_cell(1.0, -0.8)))

    def test_static_replay_fails_closed_when_static_signature_is_unqualified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "map.bin"
            values = [0.0, 0.0, 0.0] + [2.0] * 450
            with database.open("wb") as stream:
                stream.write(MAGIC)
                stream.write(struct.pack("<II", 1, 450))
                stream.write(struct.pack("<" + "f" * len(values), *values))
            trace = root / "raw.jsonl"
            trace.write_text(json.dumps({"schema": "robotsim.issue80.lidar-raw.v1", "sequence": 0, "xyz": [[0.0, 0.0, -1.0]] * 450}) + "\n", encoding="utf-8")
            result = replay(trace, database, novelty_margin_m=0.20, confirmation_frames=2, grid_resolution_m=0.10)
            self.assertEqual(result["status"], "BLOCKED_STATIC_SIGNATURE_UNQUALIFIED_FRAMES")
            self.assertEqual(result["candidate_qualified_frames"], 0)


if __name__ == "__main__":
    unittest.main()
