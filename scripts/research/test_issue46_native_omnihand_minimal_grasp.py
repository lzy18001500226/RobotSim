import unittest

from issue46_native_omnihand_minimal_grasp import derived_targets, smoothstep


class NativeOmniHandControlMathTests(unittest.TestCase):
    def test_mimic_targets_preserve_drivers_and_held_joints(self):
        targets = {"thumb_mcp": 0.4, "middle_pip": 0.6, "held": -0.1}
        details = {
            "mimic_relations": {
                "thumb_pip": {"driver": "thumb_mcp", "multiplier": 1.33, "offset": 0.0},
                "thumb_dip": {"driver": "thumb_mcp", "multiplier": 1.3, "offset": 0.0},
                "middle_dip": {"driver": "middle_pip", "multiplier": 1.097, "offset": 0.0},
            }
        }

        resolved = derived_targets(targets, details)

        self.assertEqual(resolved["thumb_mcp"], 0.4)
        self.assertEqual(resolved["middle_pip"], 0.6)
        self.assertEqual(resolved["held"], -0.1)
        self.assertAlmostEqual(resolved["thumb_pip"], 0.532)
        self.assertAlmostEqual(resolved["thumb_dip"], 0.52)
        self.assertAlmostEqual(resolved["middle_dip"], 0.6582)

    def test_smoothstep_has_bounded_endpoints_and_midpoint(self):
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertAlmostEqual(smoothstep(0.5), 0.5)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(2.0), 1.0)


if __name__ == "__main__":
    unittest.main()
