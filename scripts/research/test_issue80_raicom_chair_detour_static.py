from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from issue80_lidar_obstacle_layer import GridCell, GridSpec
from issue80_raicom_chair_detour_static import (
    aabb_blocked_cells,
    chair_model_xml,
    heading_points,
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


if __name__ == "__main__":
    unittest.main()
