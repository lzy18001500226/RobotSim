from __future__ import annotations

import unittest

import numpy as np

from g1_dfq_static_wrench import (
    ContactWrench,
    candidate_rollout_allocation,
    pyramidal_wrench_rays,
    solve_static_wrench,
)


class StaticWrenchTests(unittest.TestCase):
    def test_condim4_pyramid_has_six_slide_and_spin_rays(self):
        contact = ContactWrench(
            position_world_m=np.zeros(3),
            frame_world_rows=np.eye(3),
            friction=np.array([1.4, 1.4, 0.02, 0.001, 0.001]),
            condim=4,
            force_on_bottle_sign=1.0,
        )
        rays = pyramidal_wrench_rays(contact, np.zeros(3))
        expected = np.array([
            [1, 1, 1, 1, 1, 1],
            [1.4, -1.4, 0, 0, 0, 0],
            [0, 0, 1.4, -1.4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)
        expected[3, 4] = 0.02
        expected[3, 5] = -0.02
        np.testing.assert_allclose(rays, expected)

    def test_geom_order_sign_and_contact_moment_arm(self):
        contact = ContactWrench(
            position_world_m=np.array([0.035, 0.0, 0.0]),
            frame_world_rows=np.eye(3),
            friction=np.array([1.4, 1.4, 0.02]),
            condim=4,
            force_on_bottle_sign=-1.0,
        )
        rays = pyramidal_wrench_rays(contact, np.zeros(3))
        np.testing.assert_allclose(rays[:3, 0], [-1.0, -1.4, 0.0])
        np.testing.assert_allclose(rays[3:, 0], [0.0, 0.0, -0.049])

    def test_opposed_body_contacts_can_balance_gravity_and_moment(self):
        contacts = [
            ContactWrench(
                position_world_m=np.array([0.035, 0.0, 0.0]),
                frame_world_rows=np.eye(3),
                friction=np.array([1.4, 1.4, 0.02]),
                condim=4,
                force_on_bottle_sign=1.0,
                name="right-side",
            ),
            ContactWrench(
                position_world_m=np.array([-0.035, 0.0, 0.0]),
                frame_world_rows=np.diag([-1.0, -1.0, 1.0]),
                friction=np.array([1.4, 1.4, 0.02]),
                condim=4,
                force_on_bottle_sign=1.0,
                name="left-side",
            ),
        ]
        result = solve_static_wrench(
            contacts,
            np.zeros(3),
            np.array([0.0, 0.0, 5.5917, 0.0, 0.0, 0.0]),
        )
        self.assertTrue(result["feasible"], result["solver_status"])
        np.testing.assert_allclose(result["equilibrium_residual_world"], np.zeros(6), atol=1e-8)

    def test_candidate_allocation_maps_six_rays_to_digit_normals(self):
        driver_joints = (
            "R_pinky_proximal_joint", "R_ring_proximal_joint",
            "R_middle_proximal_joint", "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
        )
        candidate = {
            "static_candidate_pass": True,
            "robust_perturbations_pass": True,
            "active_right_hand_bottle_contacts": [
                {"body": "R_thumb_distal", "bottle_geom": "bottle_body", "digit": "thumb"},
                {"body": "R_index_distal", "bottle_geom": "bottle_body", "digit": "index"},
            ],
            "effort_bounded_6d_equilibrium": {
                "feasible": True,
                "equilibrium_dimension": 6,
                "contact_names": [
                    "R_thumb_distal:bottle_body", "R_index_distal:bottle_body",
                ],
                "ray_coefficients_n": [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                "target_wrench_world": [0, 0, 5.5917, 0, 0, 0],
                "equilibrium_residual_world": [0, 0, 0, 0, 0, 0],
                "estimated_driver_effort_nm": {name: 0.1 for name in driver_joints},
            },
        }
        allocation = candidate_rollout_allocation(
            candidate, bottle_weight_n=5.5917, effort_cap_nm=0.531,
        )
        self.assertEqual(allocation["equilibrium_dimension"], 6)
        self.assertAlmostEqual(allocation["per_digit_normal_force_n"]["thumb"], 1.0)
        self.assertAlmostEqual(allocation["per_digit_normal_force_n"]["index"], 2.0)
        self.assertAlmostEqual(allocation["total_normal_force_n"], 3.0)
        self.assertLess(allocation["maximum_absolute_driver_effort_nm"], 0.531)

    def test_candidate_allocation_rejects_failed_robustness_or_effort_cap(self):
        candidate = {
            "static_candidate_pass": True,
            "robust_perturbations_pass": False,
            "effort_bounded_6d_equilibrium": {},
        }
        with self.assertRaisesRegex(ValueError, "perturbation"):
            candidate_rollout_allocation(candidate, bottle_weight_n=5.5917, effort_cap_nm=0.531)

        candidate["robust_perturbations_pass"] = True
        candidate["active_right_hand_bottle_contacts"] = [
            {"body": "R_thumb_distal", "bottle_geom": "bottle_body", "digit": "thumb"},
            {"body": "R_index_distal", "bottle_geom": "bottle_body", "digit": "index"},
        ]
        candidate["effort_bounded_6d_equilibrium"] = {
            "feasible": True,
            "equilibrium_dimension": 6,
            "contact_names": [
                "R_thumb_distal:bottle_body", "R_index_distal:bottle_body",
            ],
            "ray_coefficients_n": [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            "target_wrench_world": [0, 0, 5.5917, 0, 0, 0],
            "equilibrium_residual_world": [0, 0, 0, 0, 0, 0],
            "estimated_driver_effort_nm": {
                "R_pinky_proximal_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_index_proximal_joint": 0.0,
                "R_thumb_proximal_pitch_joint": 0.0,
                "R_thumb_proximal_yaw_joint": 0.6,
            },
        }
        with self.assertRaisesRegex(ValueError, "simulation cap"):
            candidate_rollout_allocation(candidate, bottle_weight_n=5.5917, effort_cap_nm=0.531)


if __name__ == "__main__":
    unittest.main()
