"""Focused tests for the ideal-gripper diagnostic force accounting."""

import unittest

from scripts.research.issue46_ideal_parallel_gripper_baseline import signed_vertical_force_summary


class SignedVerticalForceSummaryTests(unittest.TestCase):
    def test_no_contacts_produce_zero_force_sums(self) -> None:
        result = signed_vertical_force_summary([])
        self.assertEqual(result["table"], {"net_N": 0.0, "upward_sum_N": 0.0, "downward_sum_N": 0.0})
        self.assertEqual(result["hand_pad"], {"net_N": 0.0, "upward_sum_N": 0.0, "downward_sum_N": 0.0})

    def test_opposing_vertical_contacts_are_not_clipped_before_net_sum(self) -> None:
        contacts = [
            {"support_class": "hand_pad", "force_on_bottle_world_N": [0.0, 0.0, 4.0]},
            {"support_class": "hand_pad", "force_on_bottle_world_N": [0.0, 0.0, -3.0]},
        ]
        result = signed_vertical_force_summary(contacts)["hand_pad"]
        self.assertEqual(result["net_N"], 1.0)
        self.assertEqual(result["upward_sum_N"], 4.0)
        self.assertEqual(result["downward_sum_N"], -3.0)

    def test_support_classes_are_aggregated_independently(self) -> None:
        contacts = [
            {"support_class": "table", "force_on_bottle_world_N": [0.0, 0.0, 5.5]},
            {"support_class": "hand_pad", "force_on_bottle_world_N": [0.0, 0.0, 0.1]},
            {"support_class": "other", "force_on_bottle_world_N": [0.0, 0.0, -50.0]},
        ]
        result = signed_vertical_force_summary(contacts)
        self.assertEqual(result["table"]["net_N"], 5.5)
        self.assertEqual(result["hand_pad"]["net_N"], 0.1)


if __name__ == "__main__":
    unittest.main()
