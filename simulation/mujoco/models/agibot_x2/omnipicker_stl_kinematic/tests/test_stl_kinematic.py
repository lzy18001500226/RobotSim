from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import mujoco

from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (
    LEFT_JOINT_MAPPING,
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


ROOT = Path(__file__).resolve().parents[6]
PACKAGE = ROOT / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic"
MODEL_ID = "X2-OmniPicker-STL"
JOINT_COLUMNS = {
    "right_claw_joint": "root_q_rad",
    "R_hand_wide1_joint": "wide_root_q_rad",
    "R_hand_narrow3_joint": "narrow3_q_rad",
    "R_hand_narrow_loop_joint": "narrow_loop_q_rad",
    "R_hand_wide3_joint": "wide3_q_rad",
    "R_hand_wide_loop_joint": "wide_loop_q_rad",
}
EXPECTED_EXCLUDES = {
    frozenset(("R_hand_narrow1_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow2_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_narrow_loop_Link")),
    frozenset(("R_hand_wide1_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide2_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_wide3_Link", "R_hand_wide_loop_Link")),
    frozenset(("R_hand_narrow3_Link", "R_hand_wide3_Link")),
}


class STLTableKinematicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_x2_omnipicker(MODEL_ID, repo_root=ROOT)
        cls.model = cls.loaded.model
        cls.trajectory = load_registered_trajectory(cls.loaded.package_path)
        cls.registry = json.loads((PACKAGE.parent / "registry.json").read_text(encoding="utf-8"))
        cls.entry = cls.registry["models"][MODEL_ID]

    def test_registry_locks_the_model_and_source_table_hashes(self) -> None:
        self.assertEqual(self.loaded.classification, "KINEMATIC_SIMULATION_ONLY")
        self.assertEqual(self.loaded.model_sha256, self.entry["portable_model_sha256"])
        self.assertEqual(
            hashlib.sha256((PACKAGE / "trajectory.csv").read_bytes()).hexdigest(),
            self.entry["trajectory_sha256"],
        )
        self.assertEqual(len(self.trajectory.rows), 101)

    def test_compiled_model_has_only_the_six_table_driven_joints(self) -> None:
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu, self.model.neq), (6, 6, 0, 0))
        actual = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(self.model.njnt)
        }
        self.assertEqual(actual, set(JOINT_COLUMNS))
        for fixed in ("R_hand_narrow2_joint", "R_hand_wide2_joint"):
            self.assertEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, fixed), -1)

    def test_all_101_reviewed_samples_are_exact_table_poses_without_contacts(self) -> None:
        for percent in range(101):
            with self.subTest(percent=percent):
                data = mujoco.MjData(self.model)
                pose = apply_stl_kinematic_pose(mujoco, self.model, data, self.trajectory, percent)
                row = self.trajectory.rows[percent]
                self.assertEqual(pose.command_percent, float(percent))
                for joint, column in JOINT_COLUMNS.items():
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    self.assertGreaterEqual(joint_id, 0)
                    actual = float(data.qpos[self.model.jnt_qposadr[joint_id]])
                    self.assertAlmostEqual(actual, row[column], delta=1e-12)
                self.assertEqual(int(data.ncon), 0)

    def test_midpoint_is_explicit_linear_interpolation_and_uses_table_derivative(self) -> None:
        data = mujoco.MjData(self.model)
        pose = apply_stl_kinematic_pose(
            mujoco, self.model, data, self.trajectory, 47.5, command_rate_percent_s=2.0
        )
        for joint, column in JOINT_COLUMNS.items():
            expected = 0.5 * (self.trajectory.rows[47][column] + self.trajectory.rows[48][column])
            self.assertAlmostEqual(pose.positions_rad[joint], expected, delta=1e-12)
            expected_velocity = 2.0 * (self.trajectory.rows[48][column] - self.trajectory.rows[47][column])
            self.assertAlmostEqual(pose.velocities_rad_s[joint], expected_velocity, delta=1e-12)

    def test_official_frame_approved_left_mapping_applies_the_same_table_values(self) -> None:
        joints = "\n".join(
            f'<joint name="{name}" type="hinge" axis="0 0 1" range="-3.14 3.14" />'
            for name in LEFT_JOINT_MAPPING.values()
        )
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><body><inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />{joints}</body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        pose = apply_stl_kinematic_pose(
            mujoco, model, data, self.trajectory, 73.0, side="left",
        )
        row = self.trajectory.rows[73]
        for right, left in LEFT_JOINT_MAPPING.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left)
            self.assertGreaterEqual(joint_id, 0)
            self.assertAlmostEqual(
                float(data.qpos[model.jnt_qposadr[joint_id]]),
                row[JOINT_COLUMNS[right]],
                delta=1e-12,
            )
            self.assertIn(left, pose.positions_rad)

    def test_internal_collision_policy_is_exactly_the_reviewed_seven_pair_list(self) -> None:
        root = __import__("xml.etree.ElementTree", fromlist=["ElementTree"]).parse(PACKAGE / "model.xml").getroot()
        exclusions = {
            frozenset((node.get("body1"), node.get("body2")))
            for node in root.findall("./contact/exclude")
        }
        self.assertEqual(exclusions, EXPECTED_EXCLUDES)


if __name__ == "__main__":
    unittest.main()
