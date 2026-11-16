#!/usr/bin/env python3
"""Regression check for the human-review level side-pregrasp geometry."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO / "scripts/research/issue46/native_runtime/docked_level_pregrasp_review.py"
SPEC = importlib.util.spec_from_file_location("docked_level_pregrasp_review", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
TABLE = EVIDENCE / "raicom-stl-standing-table-overlay-20261031T140715Z/result.json"
PLACEMENT = EVIDENCE / "raicom-standing-table-placement-search-20261031T140816Z/result.json"
STAND = EVIDENCE / "raicom-stl-xvfb-native-reset-stand-20261031T124544Z/result.json"
SEED = EVIDENCE / "raicom-standing-table-ik-validation-20261031T140950Z/result.json"


class DockedLevelPregraspTests(unittest.TestCase):
    def test_saved_candidate_is_level_and_within_existing_joint_limits(self) -> None:
        table = json.loads(TABLE.read_text(encoding="utf-8"))
        placement = json.loads(PLACEMENT.read_text(encoding="utf-8"))
        stand = json.loads(STAND.read_text(encoding="utf-8"))
        seed = json.loads(SEED.read_text(encoding="utf-8"))
        candidate = placement["scan"]["best_candidates"][13]
        shift = np.asarray(table["transforms"]["table_world_override"], dtype=float) - np.asarray(candidate["table_world_xyz_m"], dtype=float)
        model = mujoco.MjModel.from_xml_path(table["generated"]["host_scene"]["path"])
        orientation = MODULE.load_module("stl_orientation_test", REPO / "scripts/research/issue46/grasp/centered_orientation_search.py")
        upper, metrics = MODULE.solve_level_pregrasp(
            model,
            stand["native_arm_state_capture"],
            shift,
            100,
            orientation,
            np.asarray(seed["ik"]["right_arm_q_rad"], dtype=float),
        )
        self.assertLessEqual(metrics["jaw_center_error_m"], 1.0e-4)
        self.assertLessEqual(abs(metrics["forearm_height_delta_m"]), 1.0e-4)
        self.assertLessEqual(abs(metrics["distal_height_delta_m"]), 1.0e-4)
        self.assertGreaterEqual(metrics["minimum_upper_joint_margin_rad"], 1.0e-3)
        data = mujoco.MjData(model)
        MODULE.apply_pose(model, data, stand["native_arm_state_capture"], upper, shift, 100)
        unexpected = [
            row
            for row in MODULE.contact_records(model, data)
            if not MODULE.is_expected_contact(tuple(row["body_names"]))
        ]
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
