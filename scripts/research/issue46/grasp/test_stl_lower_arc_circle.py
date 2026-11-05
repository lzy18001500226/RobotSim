"""Analytic fixtures for the offline review geometry, no MuJoCo stepping."""
import unittest
import numpy as np
from stl_lower_arc_circle import nearest, section_segments, solve_circle, solve_support_circle


def rectangle(y0, y1, z0, z1):
    p = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]])
    return np.stack([p, np.roll(p, -1, axis=0)], axis=1)


class CircleGeometryTests(unittest.TestCase):
    def test_variable_radius_support_and_four_separated_bands(self):
        for radius in (.003, .035):
            z = np.linspace(-radius*.7, radius*.7, 101)
            y = np.sqrt(radius**2-z**2)
            arcs = []
            for sign in (-1,1):
                p = np.column_stack([sign*y,z])
                arcs.append(np.stack([p[:-1],p[1:]],axis=1))
            shells=[]
            for arc,sign in zip(arcs,(-1,1)):
                first,last=arc[0,0],arc[-1,1]
                outer_first=first+np.array([sign*radius,0]);outer_last=last+np.array([sign*radius,0])
                shells.append(np.concatenate([arc,np.array([[last,outer_last],[outer_last,outer_first],[outer_first,first]])]))
            obstacles = np.concatenate(shells)
            r = solve_support_circle(arcs, obstacles, ["left"]*103+["right"]*103)
            self.assertAlmostEqual(r["radius_m"],radius,delta=radius*0.001)
            self.assertLess(r["worst_support_band_gap_mm"],.01)
            self.assertEqual(len(r["support_bands"]),4)
            self.assertLess(r["support_bands"][0]["point_m"][2],r["support_bands"][1]["point_m"][2])
            self.assertFalse(r["physical_contact_area_measured"])

    def test_support_diameter_is_not_fixed(self):
        arcs=[np.array([[[-.006,-.02],[-.006,.02]]]),np.array([[[.006,-.02],[.006,.02]]])]
        obstacles=np.concatenate([rectangle(-.02,-.006,-.02,.02),rectangle(.006,.02,-.02,.02)])
        r=solve_support_circle(arcs,obstacles,["left"]*4+["right"]*4)
        self.assertAlmostEqual(r["diameter_mm"],12.,places=4)
        self.assertEqual(len(r["gap_profiles"]),2)

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
