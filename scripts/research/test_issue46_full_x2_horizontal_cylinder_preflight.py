#!/usr/bin/env python3
"""Focused regressions for the full-X2 static-contact safety gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("issue46_full_x2_horizontal_cylinder_preflight.py")
SPEC = importlib.util.spec_from_file_location("full_x2_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullX2ContactGateTests(unittest.TestCase):
    def test_head_torso_proxy_is_not_an_allowed_internal_contact(self) -> None:
        records = [
            {
                "kind": "head_torso_proxy_contact",
                "bodies": ["head_pitch_link", "torso_link"],
                "geoms": [None, None],
                "distance_m": -0.007,
            },
            {
                "kind": "cylinder_table",
                "bodies": ["test_cylinder", "raicom_table"],
                "geoms": ["test_cylinder_collision", "raicom_table_contact"],
                "distance_m": -0.00001,
            },
            {
                "kind": "distal_pad",
                "bodies": ["test_cylinder", "R_hand_narrow3_Link"],
                "geoms": ["test_cylinder_collision", "narrow_pad"],
                "distance_m": -0.0002,
            },
        ]
        unexpected = MODULE.unexpected_scene_contacts(records)
        self.assertEqual(unexpected, [records[0]])

    def test_unknown_robot_self_contact_remains_fail_closed(self) -> None:
        record = {
            "kind": "robot_self",
            "bodies": ["right_wrist_roll_link", "torso_link"],
            "geoms": ["wrist", "torso"],
            "distance_m": -0.001,
        }
        self.assertEqual(MODULE.unexpected_scene_contacts([record]), [record])

    def test_summary_deduplicates_contexts_and_retains_deepest_contact(self) -> None:
        shallow = {
            "kind": "head_torso_proxy_contact",
            "bodies": ["head_pitch_link", "torso_link"],
            "geoms": ["head_proxy", "torso_proxy"],
            "distance_m": -0.001,
        }
        deep = {**shallow, "distance_m": -0.007}
        summary = MODULE.summarize_unexpected_contacts([
            {"context": "INITIAL_OPEN", "contact": shallow},
            {"context": "INITIAL_OPEN", "contact": shallow},
            {"context": "CLOSE_SCAN_75_PERCENT", "contact": deep},
        ])
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["observation_count"], 3)
        self.assertEqual(summary[0]["contexts"], ["CLOSE_SCAN_75_PERCENT", "INITIAL_OPEN"])
        self.assertEqual(summary[0]["deepest_distance_m"], -0.007)


if __name__ == "__main__":
    unittest.main()
