import math
import unittest

import numpy as np

import issue46_candidate_b_small_cylinder_grasp as experiment


class SmallCylinderGraspMathTests(unittest.TestCase):
    def test_contact_hold_threshold_is_explicit_and_bounded(self):
        self.assertEqual(experiment.BILATERAL_HOLD_S, 0.5)
        self.assertGreater(experiment.CONTACT_FORCE_FLOOR_N, 0.0)

    def test_solid_cylinder_mass_and_inertia(self):
        props = experiment.cylinder_properties(0.025, 0.210, 1000.0)
        radius, height = 0.0125, 0.210
        expected_volume = math.pi * radius**2 * height
        expected_mass = expected_volume * 1000.0
        expected_axis = 0.5 * expected_mass * radius**2
        expected_transverse = expected_mass * (3*radius**2 + height**2) / 12.0
        self.assertAlmostEqual(props["volume_m3"], expected_volume, places=15)
        self.assertAlmostEqual(props["mass_kg"], expected_mass, places=15)
        np.testing.assert_allclose(props["inertia_kg_m2"],
                                   [expected_transverse, expected_transverse, expected_axis],
                                   rtol=1e-14, atol=1e-15)

    def test_contact_force_is_equal_and_opposite_by_geom_order(self):
        frame = np.eye(3)
        local = np.array([2.0, -0.5, 0.25, 0.0, 0.0, 0.0])
        on_geom2 = experiment.force_on_object_world(frame, local, True)
        on_geom1 = experiment.force_on_object_world(frame, local, False)
        np.testing.assert_allclose(on_geom2, [2.0, -0.5, 0.25])
        np.testing.assert_allclose(on_geom1, -on_geom2)
        np.testing.assert_allclose(experiment.object_normal_world(frame, True), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(experiment.object_normal_world(frame, False), [-1.0, 0.0, 0.0])

    def test_contact_frame_force_rotates_from_row_axis_frame(self):
        frame = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        world = experiment.force_on_geom2_world(frame, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(world, [0.0, 1.0, 0.0])

    def test_cylinder_surface_mesh_has_requested_full_height_and_radius(self):
        triangles = experiment.cylinder_surface_triangles(np.array([0.01, -0.02, 0.105]), 0.0125, 0.210)
        vertices = triangles.reshape(-1, 3)
        self.assertAlmostEqual(float(vertices[:, 2].min()), 0.0, places=12)
        self.assertAlmostEqual(float(vertices[:, 2].max()), 0.210, places=12)
        radial = np.linalg.norm(vertices[:, :2] - np.array([0.01, -0.02]), axis=1)
        self.assertAlmostEqual(float(radial.min()), 0.0, places=12)
        self.assertAlmostEqual(float(radial.max()), 0.0125, places=12)


if __name__ == "__main__":
    unittest.main()
