"""Offline geometry sampling checks; no physics integration."""
import math
import unittest

from issue80_raicom_table_plan import sample_segment


class TablePlanTests(unittest.TestCase):
    def test_translation_preserves_endpoints_and_spacing(self):
        a = (-1.5, -1.5, .68, 0.)
        b = (1., -1.5, .68, 0.)
        points = sample_segment(a, b)
        self.assertEqual(points[0], a)
        self.assertEqual(points[-1], b)
        self.assertTrue(all(math.dist(x[:3], y[:3]) <= .02500001
                            for x, y in zip(points, points[1:])))

    def test_heading_wrap_uses_short_branch(self):
        points = sample_segment((0, 0, .68, math.pi-.01),
                                (0, 0, .68, -math.pi+.01))
        self.assertAlmostEqual(points[-1][3]-points[0][3], .02)

    def test_initial_alignment_does_not_translate_base(self):
        points = sample_segment((-1.5, -1.5, .68, math.pi/2),
                                (-1.5, -1.5, .68, 0))
        self.assertGreater(len(points), 70)
        self.assertTrue(all(p[:3] == (-1.5, -1.5, .68) for p in points))


if __name__ == "__main__":
    unittest.main()
