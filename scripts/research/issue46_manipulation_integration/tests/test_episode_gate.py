from __future__ import annotations

import unittest

from scripts.research.issue46_manipulation_integration.episode_gate import (
    EpisodeCriteria,
    EpisodeFrame,
    ManipulationPhase,
    evaluate_episode_evidence,
)
from scripts.research.issue46_manipulation_integration.audit_historical_robotiq_episode import (
    infer_timestep_s,
    sample_step_index,
)


MODEL_SHA = "a" * 64


def frame(phase, step, time_s, z, roles=(), **overrides):
    values = {
        "phase": phase,
        "simulation_time_s": time_s,
        "physics_step_index": step,
        "model_id": "test-model-v1",
        "model_sha256": MODEL_SHA,
        "bottle_com_world_m": (0.3, 0.04, z),
        "bottle_quat_world_xyzw": (0.0, 0.0, 0.0, 1.0),
        "bottle_linear_speed_m_s": 0.001,
        "bottle_angular_speed_rad_s": 0.001,
        "target_bottle_com_world_m": (0.3, 0.04, 0.0),
        "force_bearing_contact_roles": frozenset(roles),
        "tool_object_contact_active": bool(roles),
        "left_hand_contact": False,
        "table_contact": True,
        "tool_upward_support_n": 0.0,
        "table_reaction_vertical_n": 5.4,
        "max_relative_slip_m": 0.001,
        "minimum_approach_clearance_m": 0.005,
        "tool_opening_m": 0.0934,
        "source_position_limits_pass": True,
        "source_velocity_limits_pass": True,
        "actuator_limits_pass": True,
        "follower_constraints_pass": True,
        "bottle_qpos_writes_after_start": 0,
        "bottle_attachment_active": False,
        "mocap_carry_active": False,
        "hidden_support_active": False,
    }
    values.update(overrides)
    return EpisodeFrame(**values)


def criteria(groups=(('left_pad',), ('right_pad',))):
    return EpisodeCriteria(
        required_contact_groups=groups,
        minimum_contact_fraction=0.9,
        minimum_tool_upward_support_n=3.0,
        minimum_table_unloading_n=3.0,
        maximum_relative_slip_m=0.005,
        place_target_xy_world_m=(0.3, 0.04),
    )


def valid_episode(groups=(('left_pad',), ('right_pad',))):
    # Select one valid representative from each required group.
    contacts = tuple(group[0] for group in groups)
    frames = [
        frame(ManipulationPhase.PREGRASP, 1, 0.01, 0.0, (), table_contact=True),
        frame(ManipulationPhase.APPROACH, 2, 0.02, 0.0, (), table_contact=True),
    ]
    for i in range(10):
        frames.append(frame(ManipulationPhase.CONTACT, 3+i, 0.03+0.01*i, 0.0, contacts))
    for i in range(10):
        frames.append(frame(
            ManipulationPhase.LOAD_TRANSFER, 13+i, 0.13+0.01*i, 0.0, contacts,
            tool_upward_support_n=4.2,
            table_reaction_vertical_n=1.2,
            table_contact=True,
        ))
    for i in range(10):
        frames.append(frame(
            ManipulationPhase.LIFT_50MM, 23+i, 0.23+0.01*i, 0.005*(i+1), contacts,
            tool_upward_support_n=5.4,
            table_reaction_vertical_n=0.0,
            table_contact=False,
        ))
    for i in range(121):
        frames.append(frame(
            ManipulationPhase.HOLD, 33+i, 0.33+0.01*i, 0.05, contacts,
            tool_upward_support_n=5.4,
            table_reaction_vertical_n=0.0,
            table_contact=False,
        ))
    for i in range(10):
        frames.append(frame(
            ManipulationPhase.TRANSFER, 154+i, 1.54+0.01*i, 0.05, contacts,
            tool_upward_support_n=5.4,
            table_reaction_vertical_n=0.0,
            table_contact=False,
            max_relative_slip_m=0.002,
        ))
    for i in range(10):
        frames.append(frame(
            ManipulationPhase.PLACE, 164+i, 1.64+0.01*i, 0.0, contacts,
            tool_upward_support_n=4.0,
            table_reaction_vertical_n=5.4,
            table_contact=True,
        ))
    for i in range(10):
        frames.append(frame(
            ManipulationPhase.RELEASE, 174+i, 1.74+0.01*i, 0.0, (),
            tool_object_contact_active=False,
            table_contact=True,
            tool_upward_support_n=0.0,
            table_reaction_vertical_n=5.4,
            tool_opening_m=0.0934,
        ))
    for i in range(31):
        frames.append(frame(
            ManipulationPhase.SETTLE, 184+i, 1.84+0.1*i, 0.0, (),
            tool_object_contact_active=False,
            table_contact=True,
            tool_upward_support_n=0.0,
            table_reaction_vertical_n=5.4,
            max_relative_slip_m=0.0,
        ))
    return tuple(frames)


class EpisodeGateTests(unittest.TestCase):
    def test_historical_truncating_step_index_is_reconstructed_from_monotonic_time(self):
        rows = [
            {"step": "1006", "time_s": "1.006"},
            {"step": "1006", "time_s": "1.007"},
        ]
        timestep = infer_timestep_s(rows)
        self.assertAlmostEqual(timestep, 0.001)
        self.assertEqual([sample_step_index(row, timestep) for row in rows], [1006, 1007])

    def test_measured_complete_episode_passes_with_role_groups(self):
        result = evaluate_episode_evidence(valid_episode(), criteria())
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.stage_status[ManipulationPhase.HOLD.value], "PASS")
        self.assertGreater(result.last_physics_step_index, 0)

    def test_omnihand_groups_are_configurable_without_robotiq_names(self):
        groups = (('thumb',), ('index', 'middle', 'ring', 'pinky'))
        result = evaluate_episode_evidence(valid_episode(groups), criteria(groups))
        self.assertEqual(result.status, "PASS")

    def test_missing_lift_and_hold_remain_not_run(self):
        frames = tuple(x for x in valid_episode() if x.phase in {
            ManipulationPhase.PREGRASP,
            ManipulationPhase.APPROACH,
            ManipulationPhase.CONTACT,
            ManipulationPhase.LOAD_TRANSFER,
        })
        result = evaluate_episode_evidence(frames, criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.stage_status[ManipulationPhase.LIFT_50MM.value], "NOT_RUN")
        self.assertEqual(result.stage_status[ManipulationPhase.HOLD.value], "NOT_RUN")

    def test_kinematic_only_rows_cannot_pass(self):
        rows = tuple(
            frame(x.phase, 0, x.simulation_time_s, x.bottle_com_world_m[2], x.force_bearing_contact_roles)
            for x in valid_episode()
        )
        result = evaluate_episode_evidence(rows, criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("POST_STEP_SAMPLE_REQUIRED", result.blockers)

    def test_bottle_state_forcing_or_attachment_blocks_episode(self):
        rows = list(valid_episode())
        rows[5] = frame(
            ManipulationPhase.CONTACT, 5, 0.05, 0.0, ('left_pad', 'right_pad'),
            bottle_qpos_writes_after_start=1,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("BOTTLE_QPOS_WRITE:5", result.blockers)

    def test_lift_with_table_contact_does_not_pass(self):
        rows = list(valid_episode())
        rows[25] = frame(
            ManipulationPhase.LIFT_50MM, 25, 0.25, 0.05, ('left_pad', 'right_pad'),
            tool_upward_support_n=5.4,
            table_reaction_vertical_n=0.0,
            table_contact=True,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("BOTTLE_STILL_TABLE_SUPPORTED_DURING_LIFT", result.blockers)

    def test_load_transfer_requires_measured_contact_and_bounded_slip(self):
        rows = list(valid_episode())
        rows[12] = frame(
            ManipulationPhase.LOAD_TRANSFER, 13, 0.13, 0.0, (),
            tool_upward_support_n=4.2,
            table_reaction_vertical_n=1.2,
        )
        rows[13] = frame(
            ManipulationPhase.LOAD_TRANSFER, 14, 0.14, 0.0, (),
            tool_upward_support_n=4.2,
            table_reaction_vertical_n=1.2,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("LOAD_TRANSFER_CONTACT_PERSISTENCE", result.blockers)

        rows = list(valid_episode())
        rows[13] = frame(
            ManipulationPhase.LOAD_TRANSFER, 14, 0.14, 0.0, ('left_pad', 'right_pad'),
            tool_upward_support_n=4.2,
            table_reaction_vertical_n=1.2,
            max_relative_slip_m=0.006,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("LOAD_TRANSFER_RELATIVE_SLIP", result.blockers)

    def test_invalid_bottle_orientation_blocks_episode(self):
        rows = list(valid_episode())
        rows[0] = frame(
            ManipulationPhase.PREGRASP, 1, 0.01, 0.0, (),
            bottle_quat_world_xyzw=(0.0, 0.0, 0.0, 0.0),
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("BOTTLE_ORIENTATION_MISSING_OR_INVALID:0", result.blockers)

    def test_lift_requires_measured_upward_support(self):
        rows = list(valid_episode())
        rows[22] = frame(
            ManipulationPhase.LIFT_50MM, 23, 0.23, 0.005, ('left_pad', 'right_pad'),
            tool_upward_support_n=2.9,
            table_reaction_vertical_n=0.0,
            table_contact=False,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("LIFT_MEASURED_UPWARD_TOOL_SUPPORT", result.blockers)

    def test_airborne_stage_requires_measured_table_reaction(self):
        rows = list(valid_episode())
        rows[33] = frame(
            ManipulationPhase.HOLD, 34, 0.34, 0.05, ('left_pad', 'right_pad'),
            tool_upward_support_n=5.4,
            table_reaction_vertical_n=None,
            table_contact=False,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("HOLD_TABLE_REACTION_EVIDENCE_MISSING", result.blockers)

    def test_release_requires_opening_and_contact_loss(self):
        rows = list(valid_episode())
        rows[173] = frame(
            ManipulationPhase.RELEASE, 174, 1.74, 0.0, ('left_pad', 'right_pad'),
            tool_upward_support_n=0.0,
            table_reaction_vertical_n=5.4,
            table_contact=True,
            tool_opening_m=0.0934,
        )
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("RELEASE_CONTACT_PERSISTED", result.blockers)

    def test_missing_source_limit_evidence_is_not_passed(self):
        rows = list(valid_episode())
        rows[0] = frame(ManipulationPhase.PREGRASP, 1, 0.01, 0.0, (), source_position_limits_pass=None)
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("SOURCE_POSITION_LIMIT_EVIDENCE_MISSING:0", result.blockers)

    def test_policy_cannot_weaken_required_lift_or_hold(self):
        with self.assertRaises(ValueError):
            EpisodeCriteria((('pad_a',), ('pad_b',)), 0.9, 1.0, 1.0, 0.005, minimum_lift_m=0.01)
        with self.assertRaises(ValueError):
            EpisodeCriteria((('pad_a',), ('pad_b',)), 0.9, 1.0, 1.0, 0.005, minimum_hold_duration_s=0.1)

    def test_post_step_sample_span_counts_first_physics_interval_for_hold_and_settle(self):
        rows = [row for row in valid_episode()
                if row.phase is not ManipulationPhase.HOLD and row.phase is not ManipulationPhase.SETTLE]
        rows.extend(
            frame(ManipulationPhase.HOLD, 1000 + i, 1.0 + i * 0.001, 0.05,
                  ('left_pad', 'right_pad'), tool_upward_support_n=5.4,
                  table_reaction_vertical_n=0.0, table_contact=False)
            for i in range(1000)
        )
        rows.extend(
            frame(ManipulationPhase.SETTLE, 2000 + i, 2.0 + i * 0.001, 0.0, (),
                  table_contact=True, tool_object_contact_active=False,
                  tool_upward_support_n=0.0, table_reaction_vertical_n=5.4,
                  max_relative_slip_m=0.0)
            for i in range(3000)
        )
        rows.sort(key=lambda row: row.physics_step_index)
        result = evaluate_episode_evidence(tuple(rows), criteria())
        self.assertEqual(result.stage_status[ManipulationPhase.HOLD.value], "PASS")
        self.assertEqual(result.stage_status[ManipulationPhase.SETTLE.value], "PASS")


if __name__ == "__main__":
    unittest.main()
