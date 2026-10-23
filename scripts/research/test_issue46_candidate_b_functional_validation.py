"""Pure geometry tests for the static cylindrical bottle-envelope check."""
import unittest

import numpy as np

import issue46_candidate_b_functional_validation as validation


def plane(y):
    return np.array([
        [[-.05, y, -.05], [.05, y, -.05], [.05, y, .05]],
        [[-.05, y, -.05], [.05, y, .05], [-.05, y, .05]],
    ])


class BottleBandFitTests(unittest.TestCase):
    def test_65_mm_envelope_fits_70_mm_parallel_gap(self):
        result = validation.cylinder_band_fit(plane(-.035), plane(.035), 65., 0.)
        self.assertEqual(result['status'], 'PASS')
        self.assertAlmostEqual(result['clearance_mm'], 5., places=5)

    def test_75_mm_envelope_fails_70_mm_parallel_gap(self):
        result = validation.cylinder_band_fit(plane(-.035), plane(.035), 75., 0.)
        self.assertEqual(result['status'], 'FAIL')
        self.assertAlmostEqual(result['clearance_mm'], -5., places=5)


if __name__ == '__main__':
    unittest.main()
