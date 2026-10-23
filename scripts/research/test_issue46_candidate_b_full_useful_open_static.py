import unittest

import numpy as np

import issue46_candidate_b_full_useful_open_static as full_open
import issue46_urdf_pipeline as pipeline


class CandidateBFullUsefulOpenStaticTests(unittest.TestCase):
    def test_normalized_endpoints_remain_inside_official_root_range(self):
        fraction0, q0, root0 = full_open.command_pose(0.0)
        fraction1, q1, root1 = full_open.command_pose(100.0)
        self.assertEqual(fraction0, 0.0)
        self.assertAlmostEqual(q0, pipeline.PHASE, places=12)
        self.assertAlmostEqual(root0, 0.0, places=12)
        self.assertEqual(fraction1, 1.0)
        self.assertAlmostEqual(q1, full_open.Q_USEFUL_OPEN, places=12)
        self.assertGreaterEqual(root1, -1.0)
        self.assertLessEqual(root1, 0.0)

    def test_command_mapping_is_finite_and_rejects_out_of_range(self):
        for percent in (0, 25, 50, 75, 100):
            fraction, q_cad, root = full_open.command_pose(float(percent))
            self.assertAlmostEqual(fraction, percent / 100.0, places=12)
            self.assertTrue(np.isfinite(q_cad))
            self.assertTrue(np.isfinite(root))
            self.assertGreaterEqual(root, -1.0)
            self.assertLessEqual(root, 0.0)
        for percent in (-1, 101, float("nan")):
            with self.assertRaises(ValueError):
                full_open.command_pose(percent)

    def test_wide_polynomial_is_the_mirrored_narrow_relation(self):
        narrow = full_open.CURRENT_COEFFICIENTS
        wide = full_open.mirrored_coefficients(narrow)
        for root in (-0.6841259890668666, -0.3420629945334333, 0.0):
            q2_narrow = np.polynomial.polynomial.polyval(root, narrow)
            q2_wide = np.polynomial.polynomial.polyval(-root, wide)
            self.assertAlmostEqual(q2_wide, -q2_narrow, places=11)

    def test_quartic_preserves_reviewed_static_anchors(self):
        coefficients = full_open.CURRENT_COEFFICIENTS
        anchors = (
            (full_open.Q_CLOSED, full_open.Q2_CLOSED_SAFE),
            (0.0, 0.0),
            (0.20, -0.3071444474513338),
            (full_open.Q_USEFUL_OPEN, -0.03865294431414085),
        )
        for q_cad, expected_q2 in anchors:
            root = pipeline.PHASE - q_cad
            actual = np.polynomial.polynomial.polyval(root, coefficients)
            self.assertAlmostEqual(actual, expected_q2, places=9)


if __name__ == "__main__":
    unittest.main()
