from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from issue80_lidar_obstacle_layer import GridCell, GridSpec
from issue80_raicom_chair_detour_static import (
    ChairDetourError,
    aabb_blocked_cells,
    chair_model_xml,
    heading_points,
    observed_detour_points,
)


class ChairDetourStaticTests(unittest.TestCase):
    def test_chair_model_is_static_mesh_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "chair.stl"
            mesh.write_bytes(b"placeholder")
            root = chair_model_xml(mesh)
            self.assertEqual(root.find("./worldbody/body").get("name"), "chair")
            self.assertEqual(root.find("./worldbody/body/geom").get("type"), "mesh")
            self.assertEqual(root.findall(".//joint"), [])
            self.assertEqual(root.findall(".//actuator"), [])

    def test_aabb_blocking_applies_explicit_inflation(self) -> None:
        grid = GridSpec(-1.0, -1.0, 0.10, 20, 20)
        blocked = aabb_blocked_cells(
            grid, np.array([[-0.1, -0.1, 0.0]]).reshape(3), np.array([[0.1, 0.1, 1.0]]).reshape(3), inflation_m=0.20,
        )
        self.assertIn(GridCell(10, 10), blocked)
        self.assertIn(GridCell(12, 10), blocked)
        self.assertNotIn(GridCell(14, 10), blocked)

    def test_detour_headings_follow_path_and_restore_final_yaw(self) -> None:
        poses = heading_points(((0.0, 0.0), (1.0, 1.0), (2.0, 1.0)), final_yaw_rad=0.0)
        self.assertAlmostEqual(poses[0][2], math.pi / 4.0)
        self.assertAlmostEqual(poses[1][2], 0.0)
        self.assertAlmostEqual(poses[2][2], 0.0)

    def test_static_audit_has_no_mj_step_call(self) -> None:
        source = Path(__file__).with_name("issue80_raicom_chair_detour_static.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertIn("mj_forward", calls)
        self.assertNotIn("mj_step", calls)


class ObservedDetourTests(unittest.TestCase):
    def test_requires_confirmed_background_replay_and_finite_waypoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            good = {
                "schema": "robotsim.issue80.lidar-map-background-replay.v1",
                "status": "PASS_CAPTURE_REPLAYED",
                "second_decision": "DETOUR_READY_STATIC_ONLY",
                "detour_waypoints": [{"x_m": -1.5, "y_m": -1.5}, {"x_m": 0.9, "y_m": -1.5}],
            }
            path.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(observed_detour_points(path), ((-1.5, -1.5), (0.9, -1.5)))
            good["second_decision"] = "CLEAR_DIRECT_PATH_STATIC_ONLY"
            path.write_text(json.dumps(good), encoding="utf-8")
            with self.assertRaisesRegex(ChairDetourError, "did not confirm"):
                observed_detour_points(path)


if __name__ == "__main__":
    unittest.main()
