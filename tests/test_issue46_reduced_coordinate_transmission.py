import unittest

import numpy as np

from scripts.research.issue46_reduced_coordinate_transmission import reduced_column


class ReducedCoordinateTransmissionTests(unittest.TestCase):
    def test_source_mimic_ratios_define_one_velocity_basis(self):
        column = reduced_column(
            nv=7,
            driver_dof=2,
            driver_axis_sign=1,
            followers=[(4, 1.33, 1), (6, 1.30, -1)],
        )

        np.testing.assert_allclose(column, [0.0, 0.0, 1.0, 0.0, 1.33, 0.0, -1.30])

    def test_reduced_basis_excludes_unrelated_and_follower_independent_motion(self):
        column = reduced_column(
            nv=5,
            driver_dof=0,
            driver_axis_sign=-1,
            followers=[(1, 1.33, 1), (2, 1.30, 1)],
        )

        self.assertEqual(np.count_nonzero(column), 3)
        self.assertEqual(column[3], 0.0)
        self.assertEqual(column[4], 0.0)
        self.assertEqual(column[0], -1.0)
        self.assertEqual(column[1], 1.33)
        self.assertEqual(column[2], 1.30)


if __name__ == "__main__":
    unittest.main()
