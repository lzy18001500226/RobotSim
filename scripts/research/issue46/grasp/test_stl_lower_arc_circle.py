"""Analytic fixtures for the offline review geometry, no MuJoCo stepping."""
import unittest
import numpy as np
from stl_lower_arc_circle import nearest, section_segments, solve_circle


def rectangle(y0, y1, z0, z1):
    p = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]])
    return np.stack([p, np.roll(p, -1, axis=0)], axis=1)


class CircleGeometryTests(unittest.TestCase):
    def test_exact_triangle_section(self):
        triangles = np.array([[[-1, 0, 0], [1, 2, 0], [1, 0, 2]]], dtype=float)
        np.testing.assert_allclose(section_segments(triangles), [[[1, 0], [0, 1]]])

    def test_no_intersection(self):
        self.assertEqual(len(section_segments(np.ones((1, 3, 3)))), 0)

    def test_distance_uses_edges_not_only_vertices(self):
        distance, point, _ = nearest(np.array([0., 0.]), np.array([[[-1., 2.], [1., 2.]]]))
        self.assertAlmostEqual(distance, 2.)
        np.testing.assert_allclose(point, [0, 2])

    def test_known_twenty_mm_bilateral_corridor(self):
        arcs = [np.array([[[-.01, -.02], [-.01, .02]]]), np.array([[[.01, -.02], [.01, .02]]])]
        obstacles = np.concatenate([rectangle(-.02, -.01, -.02, .02), rectangle(.01, .02, -.02, .02)])
        result = solve_circle(arcs, obstacles, ["left"] * 4 + ["right"] * 4)
        self.assertAlmostEqual(result["diameter_mm"], 20., places=5)
        self.assertEqual(result["status"], "BILATERAL_LOWER_ARC")
        self.assertFalse(result["section_center_inside_mesh"])
        self.assertFalse(result["finite_cylinder_collision_checked"])

    def test_support_obstacle_cannot_be_ignored(self):
        arcs = [np.array([[[-.01, -.004], [-.01, .004]]]), np.array([[[.01, -.004], [.01, .004]]])]
        obstacles = np.concatenate([rectangle(-.02, -.01, -.03, .03), rectangle(.01, .02, -.03, .03), rectangle(-.03, .03, -.03, -.005)])
        result = solve_circle(arcs, obstacles, ["left"] * 4 + ["right"] * 4 + ["support"] * 4)
        self.assertLess(result["diameter_mm"], 20.)
        self.assertEqual(result["nearest_geometry"], "support")
        self.assertEqual(result["status"], "OTHER_GEOMETRY_OR_ONE_SIDED_LIMIT")


if __name__ == "__main__":
    unittest.main()
