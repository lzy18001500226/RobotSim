"""Focused task-boundary checks; no physics, SDK, or gain tuning."""
import json
import unittest
from types import SimpleNamespace as NS

from scripts.research.issue46.native_runtime import native_task_grasp_region as task


class ClockRecoveryTests(unittest.TestCase):
    def node(self, *, failure=None, reset=False):
        calls = []
        node = NS(feedback_fault={"gate": "CLOCK_STEP_DURING_MOTION"},
                  moving=True, stop_started=False, latest_row={"generation": 1})
        def zero():
            calls.append("zero")
            node.moving = False
        def pump():
            calls.append("pump")
            if reset:
                node.latest_row["generation"] = 2
        def observe(seconds, **kwargs):
            calls.append(("observe", seconds, kwargs))
            if failure:
                raise RuntimeError(failure)
            return {"pass": True}
        node.deadline_zero, node.pump, node.observe = zero, pump, observe
        node.stand_once = lambda: calls.append("stand")
        return node, calls

    def test_zero_then_complete_stop_precedes_any_rearm(self):
        node, calls = self.node()
        result = task.recover_clock_at_stop(node, 1, 1, 0)
        self.assertEqual(calls[0], "zero")
        self.assertFalse(node.moving)
        self.assertTrue(result["stop"]["pass"])
        self.assertEqual(calls[-1], ("observe", 12.0, {"baseline": 1, "require_stop": True}))

    def test_stale_feedback_cannot_be_recovered_as_a_clock_step(self):
        node, calls = self.node(failure="CRITICAL_ROS_TELEMETRY_STALE:imu")
        with self.assertRaisesRegex(RuntimeError, "STALE"):
            task.recover_clock_at_stop(node, 1, 1, 0)
        self.assertFalse(node.moving)
        self.assertNotIn("stand", calls)

    def test_reset_is_not_a_wall_clock_recovery(self):
        node, calls = self.node(reset=True)
        with self.assertRaisesRegex(RuntimeError, "SIMULATION_RESET"):
            task.recover_clock_at_stop(node, 1, 1, 0)
        self.assertFalse(node.moving)

    def test_physical_fault_latch_is_not_cleared(self):
        node, calls = self.node()
        node.feedback_fault = {"gate": "SOURCE_JOINT_SPEED:right_knee_joint"}
        with self.assertRaisesRegex(RuntimeError, "NOT_A_CLOCK"):
            task.recover_clock_at_stop(node, 1, 1, 0)
        self.assertEqual(node.feedback_fault["gate"], "SOURCE_JOINT_SPEED:right_knee_joint")

    def test_a_second_clock_fault_exhausts_the_scoped_recovery(self):
        node, calls = self.node()
        with self.assertRaisesRegex(RuntimeError, "BUDGET"):
            task.recover_clock_at_stop(node, 1, 1, 1)

    def test_failed_stop_cannot_be_qualified(self):
        node, calls = self.node(failure="COMPOSITE_STOP_NOT_QUALIFIED")
        with self.assertRaisesRegex(RuntimeError, "STOP_NOT_QUALIFIED"):
            task.recover_clock_at_stop(node, 1, 1, 0)

    def test_estimator_return_has_a_fresh_control_envelope(self):
        estimated = {"center_xy_world_m": [1.4, -1.6], "top_world_z_m": .77,
                     "axis_world": [0., 0., 1.], "diameter_m": .07}
        native = {"host_monotonic_ns": 1_000_000_000, "generation": 1}
        value = task.control_observation(estimated, native, now_ns=1_100_000_000)
        self.assertIs(value["pass"], True)
        self.assertFalse(value["object_truth_used_by_estimator"])
        self.assertEqual(value["generation"], 1)
        with self.assertRaisesRegex(RuntimeError, "STALE_RGBD_SOURCE"):
            task.control_observation(estimated, native, now_ns=1_201_000_000)

    def test_torso_target_follows_observed_bottle_bearing(self):
        reference_root = [0., 0.]
        reference_object = [.5, 0.]
        left, a = task.torso_heading_for_target([.5, .08, .7], [0., 0.], reference_root, reference_object, .2)
        right, b = task.torso_heading_for_target([.5, -.08, .7], [0., 0.], reference_root, reference_object, .2)
        self.assertGreater(left, .2)
        self.assertLess(right, .2)
        self.assertAlmostEqual(a, -b)
        with self.assertRaisesRegex(ValueError, "bearing"):
            task.torso_heading_for_target([0., 0., .7], [0., 0.], reference_root, reference_object, .2)

    def test_short_route_preserves_control_and_physics_parameters(self):
        from pathlib import Path
        profile = json.loads((Path(task.__file__).resolve().parents[4]/"simulation/mujoco/scenes/raicom2026/config/reviewed_carry_step_h220_forward_no_roll.json").read_text())
        before = json.dumps(profile, sort_keys=True)
        result = task.task_route_profile(profile)
        self.assertEqual(before, json.dumps(profile, sort_keys=True))
        changed = {k for k in result if result[k] != profile.get(k)}
        self.assertEqual(changed, {"delivery_turn_direction", "turn_translation_strategy", "task_turn_selection", "stop_inside_functional_arrival_region"})
        self.assertIs(result["stop_inside_functional_arrival_region"], False)
        self.assertEqual(result["delivery_turn_direction"], "SHORTEST")

    def test_selected_posture_reaches_the_actual_consumer_before_any_motion(self):
        from pathlib import Path
        from unittest.mock import patch
        from scripts.research.issue46.native_runtime import standing_waist_preposition as waist
        class BeforeMotion(Exception):
            pass
        with patch.object(waist, "load_runtime_proposal", side_effect=BeforeMotion) as loader:
            with self.assertRaises(BeforeMotion):
                waist.execute_preposition(None, None, None, Path("/unused"), Path("/source"), runtime_proposal_path=Path("/selected"))
        loader.assert_called_once_with(Path("/unused"), Path("/selected"))

    def test_pose_override_preserves_enriched_safety_and_default_loading(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts.research.issue46.native_runtime import standing_waist_preposition as waist
        with TemporaryDirectory() as directory:
            output = Path(directory)
            base = {"schema": "robotsim.standing-grasp-posture.v1", "body_drop_m": .1,
                    "robot_bodies": ["pelvis", "head_pitch_link"],
                    "waist_target_rad": {"waist_yaw_joint": .18, "waist_pitch_joint": .26},
                    "torso_world_yaw_target_rad": .245}
            selected = {k:v for k,v in base.items() if k != "robot_bodies"}
            selected["torso_world_yaw_target_rad"] = .08
            (output/"standing_grasp_proposal.json").write_text(json.dumps(base))
            (output/"selected.json").write_text(json.dumps(selected))
            from unittest.mock import patch
            with patch.object(task.pg, "write_new_text", side_effect=lambda path, text: path.write_text(text)):
                effective = task.stage_runtime_posture(output, output/"selected.json")
            _, default = waist.load_runtime_proposal(output)
            _, actual = waist.load_runtime_proposal(output, effective)
            self.assertEqual(default["torso_world_yaw_target_rad"], .245)
            self.assertEqual(actual["torso_world_yaw_target_rad"], .08)
            self.assertEqual(actual["robot_bodies"], base["robot_bodies"])
            self.assertEqual(actual["body_drop_m"], base["body_drop_m"])

    def test_pose_selection_cannot_modify_body_drop_or_other_protected_fields(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"standing_grasp_proposal.json").write_text(json.dumps({"body_drop_m": .1}))
            (output/"selected.json").write_text(json.dumps({"body_drop_m": .2}))
            with self.assertRaisesRegex(ValueError, "protected field"):
                task.stage_runtime_posture(output, output/"selected.json")

    def test_forward_turn_preserves_checked_input_curvature_during_taper(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import planned_turn_forward
        cfg = {"angular_command_cap": .6, "delivery": {
            "turn_translation_strategy": "FORWARD_ARC", "delivery_forward_command_m_s": .2}}
        self.assertAlmostEqual(planned_turn_forward(cfg, .6, 0.), .2)
        self.assertAlmostEqual(planned_turn_forward(cfg, .3, 0.), .1)
        self.assertEqual(planned_turn_forward(cfg, 0., 0.), 0.)
        self.assertEqual(planned_turn_forward({"delivery": {"turn_translation_strategy": "PURE_YAW"}}, .6, .04), .04)
        with self.assertRaisesRegex(ValueError, "checked"):
            planned_turn_forward(cfg, .7, 0.)

    def test_generated_program_compiles_without_motion(self):
        code = task.build_motion_script([.5, -.5, 0.], {}, [], {}, "owned_test")
        compile(code, "task-region", "exec")
        self.assertIn("return run_region_navigation(n,contract)", code)
        self.assertIn('CLOCK_STEP_DURING_MOTION', code)
        self.assertIn('require_stop=True', code)


if __name__ == "__main__":
    unittest.main()
