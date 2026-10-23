"""Pure geometry checks for the offline passive-motion diagnosis."""
import unittest

import numpy as np

import issue46_passive_motion_diagnosis as d


class PassiveMotionDiagnosisTests(unittest.TestCase):
    def test_adaptive_multiray_finds_parallel_inner_faces(self):
        narrow = np.array([
            [[-1., -.015, -1.], [1., -.015, -1.], [1., -.015, 1.]],
            [[-1., -.015, -1.], [1., -.015, 1.], [-1., -.015, 1.]],
        ])
        wide = narrow.copy()
        wide[:, :, 1] = .015
        result = d.ray_grid_aperture(narrow, wide)
        self.assertEqual(result['valid_rays'], result['total_rays'])
        self.assertAlmostEqual(result['positive_width_mm']['median'], 30.)
        self.assertEqual(result['negative_width_rays'], 0)

    def test_disjoint_projected_jaws_are_reported_unmeasurable(self):
        narrow = np.array([
            [[-1., -.015, -1.], [1., -.015, -1.], [1., -.015, 1.]],
            [[-1., -.015, -1.], [1., -.015, 1.], [-1., -.015, 1.]],
        ])
        wide = narrow.copy()
        wide[:, :, 2] += 3.0
        result = d.ray_grid_aperture(narrow, wide)
        self.assertFalse(result['projected_bounds_overlap'])
        self.assertEqual(result['coverage'], 0.0)
        self.assertIsNone(result['positive_width_mm'])

    def test_reversed_inner_faces_report_negative_signed_gaps(self):
        narrow = np.array([
            [[-1., .015, -1.], [1., .015, -1.], [1., .015, 1.]],
            [[-1., .015, -1.], [1., .015, 1.], [-1., .015, 1.]],
        ])
        wide = narrow.copy()
        wide[:, :, 1] = -.015
        result = d.ray_grid_aperture(narrow, wide, x_count=2, z_count=2)
        self.assertEqual(result['valid_rays'], 4)
        self.assertEqual(result['negative_width_rays'], 4)
        self.assertAlmostEqual(result['most_negative_width_mm'], -30.)


if __name__ == '__main__':
    unittest.main()
