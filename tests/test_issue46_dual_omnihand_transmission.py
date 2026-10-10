import unittest

import numpy as np

from scripts.research.issue46_dual_omnihand_transmission_probe import (
    critical_damping,
    expand_transmission_targets,
    smoothstep5,
    transmission_column,
)


class ReducedTransmissionMathTests(unittest.TestCase):
    def test_quintic_motion_has_zero_endpoint_velocity(self):
        samples = np.array([smoothstep5(x) for x in np.linspace(0.0, 1.0, 101)])
        self.assertEqual(samples[0], 0.0)
        self.assertEqual(samples[-1], 1.0)
        self.assertTrue(np.all(np.diff(samples) >= 0.0))
        self.assertLess(samples[1] - samples[0], 0.001)
        self.assertLess(samples[-1] - samples[-2], 0.001)

    def test_critical_damping_is_derived_from_effective_inertia(self):
        inertia = 2.4e-4
        omega = 8.0
        kp, kd = critical_damping(inertia, omega)
        self.assertAlmostEqual(kp, inertia * omega**2)
        self.assertAlmostEqual(kd**2, 4.0 * inertia * kp)

    def test_mimic_column_preserves_sign_and_multiplier(self):
        column = transmission_column(8, 2, -1, [(5, 1.33, 1), (7, 1.30, -1)])
        np.testing.assert_array_equal(column, [0.0, 0.0, -1.0, 0.0, 0.0, 1.33, 0.0, -1.30])

    def test_follower_targets_follow_source_ratio_and_offset(self):
        targets = expand_transmission_targets(
            {"thumb_mcp": 0.2, "index_pip": 0.4},
            [
                {"driver_joint": "thumb_mcp", "follower_joint": "thumb_pip", "multiplier": 1.33, "offset": 0.0},
                {"driver_joint": "index_pip", "follower_joint": "index_dip", "multiplier": 1.097, "offset": 0.01},
            ],
        )
        self.assertAlmostEqual(targets["thumb_pip"], 0.266)
        self.assertAlmostEqual(targets["index_dip"], 0.4488)

    def test_invalid_inertia_is_rejected(self):
        with self.assertRaises(ValueError):
            critical_damping(0.0, 8.0)


if __name__ == "__main__":
    unittest.main()
