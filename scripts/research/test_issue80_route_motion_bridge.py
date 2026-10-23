#!/usr/bin/env python3
"""Offline tests for route-intent to bounded M1 proposal translation."""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME,
    COMMAND_PUBLISHER_MESSAGE_TYPE,
    COMMAND_PUBLISHER_TOPIC,
    SOURCE_PRIORITY,
    SOURCE_TIMEOUT_MS,
    VELOCITY_SOURCE_NAME,
)
from issue80_waypoint_route import DirectiveKind, PoseEstimate, Profile, RouteDirective
from issue80_route_motion_bridge import (
    MAX_INCREMENTAL_WINDOW_S,
    RouteMotionBridge,
    RouteMotionBridgeError,
    RuntimeReadiness,
    validate_physical_stop_evidence,
)


def make_pose(
    *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
    stamp: int = 1_000_000_000, received: float = 10.0,
    generation: str = "episode-1", frame: str = "odom",
) -> PoseEstimate:
    chain = ("odom->base_link",) if frame == "odom" else ("map->odom", "odom->base_link")
    return PoseEstimate(
        frame_id=frame, child_frame_id="base_link", x_m=x, y_m=y, yaw_rad=yaw,
        stamp_ns=stamp, received_steady_s=received, generation=generation,
        source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
        transform_chain=chain, transform_age_s=0.01,
    )


def readiness(**overrides) -> RuntimeReadiness:
    fields = dict(
        observed_steady_s=10.0,
        reset_gate_pass=True,
        standing_gate_pass=True,
        safety_gates_pass=True,
        source_registration_pass=True,
        action_source_setup_pass=True,
        source_mode_handshake_pass=False,
        telemetry_fresh=True,
        workspace_clearance_verified=True,
        corridor_clear=True,
        swept_path_clear=True,
        bounded_moving_turn_qualified=False,
        qualified_turn_directions=(),
        turn_stop_margin_m=None,
        turn_stop_margin_direction="",
        turn_stop_margin_evidence_id="",
        transport_isolated=True,
        publisher_identity_verified=True,
        mc_subscriber_identity_verified=True,
        qos_compatible=True,
        velocity_publisher_count=1,
        mc_velocity_subscriber_count=1,
        ros_domain_id=194,
        expected_ros_domain_id=194,
        mode_desc="STAND_DEFAULT",
        selected_source="",
    )
    fields.update(overrides)
    return RuntimeReadiness(**fields)


def follow_directive(**overrides) -> RouteDirective:
    fields = dict(
        kind=DirectiveKind.FOLLOW_WAYPOINT,
        route_id="test-route",
        segment_id="leg-1",
        waypoint_id="W1",
        profile=Profile.TRANSIT,
        goal_x_m=2.0,
        goal_y_m=0.0,
        position_tolerance_m=0.20,
        target_forward_m_s=0.14,
        max_forward_m_s=0.14,
        max_angular_rad_s=0.075,
        max_command_window_s=10.0,
    )
    fields.update(overrides)
    return RouteDirective(**fields)


def turn_directive(**overrides) -> RouteDirective:
    fields = dict(
        kind=DirectiveKind.BOUNDED_MOVING_TURN,
        route_id="test-route",
        segment_id="corner-1",
        waypoint_id="C1",
        profile=Profile.CORNER,
        goal_yaw_rad=-0.30,
        yaw_tolerance_rad=0.01,
        target_forward_m_s=0.12,
        max_forward_m_s=0.12,
        max_angular_rad_s=0.075,
        max_command_window_s=5.0,
        maximum_turn_displacement_m=0.25,
    )
    fields.update(overrides)
    return RouteDirective(**fields)


class RouteMotionBridgeTests(unittest.TestCase):
    def test_follow_maps_measured_bearing_to_bounded_forward_arc(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        plan = bridge.plan_next(
            follow_directive(goal_y_m=0.10),
            make_pose(),
            readiness(),
            now_steady_s=10.0,
        )
        self.assertEqual(plan.classification, "SIMULATION_ONLY_ROUTE_ACTION_PROPOSAL")
        self.assertEqual(plan.status, "PROPOSAL_ONLY")
        self.assertEqual(plan.source_name, VELOCITY_SOURCE_NAME)
        self.assertEqual(plan.source_priority, SOURCE_PRIORITY)
        self.assertEqual(plan.source_timeout_ms, SOURCE_TIMEOUT_MS)
        self.assertEqual(plan.action_source_name, ACTION_SOURCE_NAME)
        self.assertEqual(plan.topic, COMMAND_PUBLISHER_TOPIC)
        self.assertEqual(plan.message_type, COMMAND_PUBLISHER_MESSAGE_TYPE)
        self.assertAlmostEqual(plan.forward_m_s, 0.14)
        self.assertEqual(plan.lateral_m_s, 0.0)
        self.assertAlmostEqual(plan.angular_rad_s, 0.8 * math.atan2(0.10, 2.0))
        self.assertLessEqual(plan.maximum_duration_s, MAX_INCREMENTAL_WINDOW_S)
        self.assertTrue(plan.activation_handshake_required)
        self.assertFalse(plan.physical_stop_gate_required)

    def test_next_window_requires_fresh_same_sample_source_and_locomotion_handshake(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        first = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=0.5)
        stale_handshake = replace(
            readiness(observed_steady_s=10.1, mode_desc="LOCOMOTION_DEFAULT",
                      selected_source=VELOCITY_SOURCE_NAME),
            source_mode_handshake_pass=False,
        )
        second = bridge.plan_next(
            follow_directive(), make_pose(stamp=1_100_000_000, received=10.1),
            stale_handshake, now_steady_s=10.1,
        )
        self.assertEqual(second.status, "BLOCKED_ZERO_ONLY")
        self.assertEqual((second.forward_m_s, second.angular_rad_s), (0.0, 0.0))
        self.assertIn("handshake", second.reason)

    def test_confirmed_activation_allows_next_fresh_window(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        first = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=0.5)
        active = readiness(
            observed_steady_s=10.1, source_mode_handshake_pass=True,
            mode_desc="LOCOMOTION_DEFAULT", selected_source=VELOCITY_SOURCE_NAME,
        )
        second = bridge.plan_next(
            follow_directive(), make_pose(yaw=0.01, stamp=1_100_000_000, received=10.1),
            active, now_steady_s=10.1,
        )
        self.assertEqual(second.status, "PROPOSAL_ONLY")
        self.assertFalse(second.activation_handshake_required)

    def test_one_in_flight_window_prevents_concurrent_nonzero_proposals(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        plan = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        with self.assertRaisesRegex(RouteMotionBridgeError, "not been completed"):
            bridge.plan_next(
                follow_directive(), make_pose(stamp=1_100_000_000, received=10.1),
                readiness(observed_steady_s=10.1), now_steady_s=10.1,
            )
        bridge.complete_window(plan.window_id, actual_elapsed_s=0.4)

    def test_cancel_zero_invalidates_in_flight_window_without_claiming_stop(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        plan = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        zero = bridge.zero_window(
            RouteDirective(
                DirectiveKind.STOP_AND_ABORT, "test-route", "leg-1", "W1", Profile.TRANSIT,
                reason="cancelled",
            ),
            now_steady_s=10.01,
            reason="cancelled",
        )
        self.assertEqual((zero.forward_m_s, zero.angular_rad_s), (0.0, 0.0))
        self.assertEqual(zero.maximum_duration_s, 1.02)
        self.assertEqual(zero.zero_publish_minimum_messages, 50)
        self.assertTrue(zero.physical_stop_gate_required)
        self.assertNotEqual(zero.status, "PASS")
        with self.assertRaisesRegex(RouteMotionBridgeError, "does not match"):
            bridge.complete_window(plan.window_id, actual_elapsed_s=0.1)

    def test_next_route_activation_rearms_only_after_measured_physical_stop(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        first = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=0.3)
        stop_directive = RouteDirective(
            DirectiveKind.REQUEST_PHYSICAL_STOP, "test-route", "leg-1", "W1", Profile.TRANSIT,
        )
        bridge.plan_next(stop_directive, make_pose(stamp=1_300_000_000, received=10.3),
                         readiness(observed_steady_s=10.3), now_steady_s=10.3)

        next_pose = make_pose(stamp=1_400_000_000, received=10.4)
        blocked = bridge.plan_next(
            follow_directive(segment_id="leg-2"), next_pose,
            readiness(observed_steady_s=10.4), now_steady_s=10.4,
        )
        self.assertEqual(blocked.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("physical-stop evidence", blocked.reason)

        evidence = {
            "status": "PASS",
            "window_start_steady_s": 10.32,
            "window_end_steady_s": 12.32,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.20,
            "generation": "episode-1",
        }
        self.assertTrue(bridge.confirm_physical_stop(
            evidence, zero_request_started_s=10.31, now_steady_s=12.4,
            expected_generation="episode-1",
        ))
        rearmed = bridge.plan_next(
            follow_directive(segment_id="leg-2"), make_pose(stamp=1_500_000_000, received=12.4),
            readiness(observed_steady_s=12.4, mode_desc="STAND_DEFAULT", selected_source=""),
            now_steady_s=12.4,
        )
        self.assertEqual(rearmed.status, "PROPOSAL_ONLY")
        self.assertTrue(rearmed.activation_handshake_required)

    def test_route_stop_directive_generates_zero_and_keeps_physical_stop_gate(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        directive = RouteDirective(
            DirectiveKind.REQUEST_PHYSICAL_STOP, "r", "s", "s", Profile.CORNER,
            reason="corner reached",
        )
        plan = bridge.plan_next(directive, make_pose(), readiness(), now_steady_s=10.0)
        self.assertEqual(plan.status, "ZERO_ONLY_PROPOSAL")
        self.assertTrue(plan.physical_stop_gate_required)
        self.assertEqual(plan.forward_m_s, 0.0)
        self.assertEqual(plan.angular_rad_s, 0.0)

    def test_reset_stand_safety_source_graph_domain_and_qos_are_fail_closed(self) -> None:
        mutations = (
            {"reset_gate_pass": False},
            {"standing_gate_pass": False},
            {"safety_gates_pass": False},
            {"source_registration_pass": False},
            {"action_source_setup_pass": False},
            {"telemetry_fresh": False},
            {"transport_isolated": False},
            {"publisher_identity_verified": False},
            {"mc_subscriber_identity_verified": False},
            {"qos_compatible": False},
            {"swept_path_clear": False},
            {"velocity_publisher_count": 2},
            {"mc_velocity_subscriber_count": 2},
            {"ros_domain_id": 195},
            {"publisher_depth": 0},
            {"publisher_history": "keep_all"},
            {"selected_source": "another.source"},
            {"mode_desc": "UNKNOWN"},
            {"safety_gates_pass": 1},
            {"ros_domain_id": True},
            {"source_mode_handshake_pass": True},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bridge = RouteMotionBridge(route_frame_id="odom")
                plan = bridge.plan_next(
                    follow_directive(), make_pose(), readiness(**mutation), now_steady_s=10.0,
                )
                self.assertEqual(plan.status, "BLOCKED_ZERO_ONLY")
                self.assertEqual((plan.forward_m_s, plan.angular_rad_s), (0.0, 0.0))

    def test_stale_pose_generation_change_and_nonmonotonic_stamp_block_motion(self) -> None:
        stale_bridge = RouteMotionBridge(route_frame_id="odom")
        stale = stale_bridge.plan_next(
            follow_directive(), make_pose(received=9.7), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(stale.status, "BLOCKED_ZERO_ONLY")
        bridge = RouteMotionBridge(route_frame_id="odom")
        first = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=0.5)
        active = readiness(observed_steady_s=10.1, source_mode_handshake_pass=True,
                           mode_desc="LOCOMOTION_DEFAULT", selected_source=VELOCITY_SOURCE_NAME)
        changed_generation = bridge.plan_next(
            follow_directive(),
            make_pose(stamp=1_100_000_000, received=10.1, generation="episode-2"),
            active, now_steady_s=10.1,
        )
        self.assertEqual(changed_generation.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("generation", changed_generation.reason)

    def test_nonmonotonic_pose_stamp_blocks_next_window(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        first = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=0.5)
        active = readiness(
            observed_steady_s=10.1,
            source_mode_handshake_pass=True,
            mode_desc="LOCOMOTION_DEFAULT",
            selected_source=VELOCITY_SOURCE_NAME,
        )
        regressed = bridge.plan_next(
            follow_directive(),
            make_pose(stamp=1_000_000_000, received=10.1),
            active,
            now_steady_s=10.1,
        )
        self.assertEqual(regressed.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("stamp", regressed.reason)

    def test_map_frame_requires_complete_transform_chain(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="map")
        bad = make_pose(frame="odom")
        plan = bridge.plan_next(follow_directive(), bad, readiness(), now_steady_s=10.0)
        self.assertEqual(plan.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("frame", plan.reason)

    def test_turn_uses_measured_yaw_and_cannot_exceed_route_displacement_budget(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        turn_ready = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.005,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-trace",
        )
        first = bridge.plan_next(turn_directive(), make_pose(), turn_ready, now_steady_s=10.0)
        self.assertEqual(first.status, "PROPOSAL_ONLY")
        self.assertAlmostEqual(first.forward_m_s, 0.12)
        self.assertAlmostEqual(first.angular_rad_s, -0.075)
        self.assertTrue(first.measured_displacement_guard_required)
        self.assertLessEqual(first.maximum_duration_s, 1.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=1.0)
        active = readiness(
            observed_steady_s=11.0, bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.005,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-trace",
            source_mode_handshake_pass=True, mode_desc="LOCOMOTION_DEFAULT",
            selected_source=VELOCITY_SOURCE_NAME,
        )
        at_budget = bridge.plan_next(
            turn_directive(), make_pose(x=0.245, yaw=-0.15, stamp=2_000_000_000, received=11.0),
            active, now_steady_s=11.0,
        )
        self.assertEqual(at_budget.status, "TURN_DISPLACEMENT_BUDGET_REACHED_ZERO_ONLY")
        self.assertTrue(at_budget.physical_stop_gate_required)

    def test_turn_requires_evidence_backed_stopping_distance_reserve(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        unqualified_stop = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
        )
        plan = bridge.plan_next(turn_directive(), make_pose(), unqualified_stop, now_steady_s=10.0)
        self.assertEqual(plan.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("stopping-distance", plan.reason)

    def test_right_turn_evidence_does_not_qualify_left_turn(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        right_only = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.005,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-trace",
        )
        left = bridge.plan_next(
            turn_directive(goal_yaw_rad=0.30),
            make_pose(),
            right_only,
            now_steady_s=10.0,
        )
        self.assertEqual(left.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("left", left.reason)

    def test_nonfinite_turn_target_and_displacement_budget_fail_closed(self) -> None:
        base = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.005,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-trace",
        )
        for change in (
            {"goal_yaw_rad": float("nan")},
            {"maximum_turn_displacement_m": float("inf")},
        ):
            with self.subTest(change=change):
                bridge = RouteMotionBridge(route_frame_id="odom")
                plan = bridge.plan_next(
                    turn_directive(**change), make_pose(), base, now_steady_s=10.0,
                )
                self.assertEqual(plan.status, "BLOCKED_ZERO_ONLY")
                self.assertEqual((plan.forward_m_s, plan.angular_rad_s), (0.0, 0.0))

    def test_unqualified_turn_and_reached_turn_target_are_zero_only(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        unqualified = bridge.plan_next(turn_directive(), make_pose(), readiness(), now_steady_s=10.0)
        self.assertEqual(unqualified.status, "BLOCKED_ZERO_ONLY")
        reached_bridge = RouteMotionBridge(route_frame_id="odom")
        reached = reached_bridge.plan_next(
            turn_directive(goal_yaw_rad=0.005, yaw_tolerance_rad=0.01),
            make_pose(),
            readiness(
                bounded_moving_turn_qualified=True,
                qualified_turn_directions=("RIGHT",),
                turn_stop_margin_m=0.005,
                turn_stop_margin_direction="RIGHT",
                turn_stop_margin_evidence_id="run65-right-stop-trace",
            ),
            now_steady_s=10.0,
        )
        self.assertEqual(reached.status, "TURN_TARGET_REACHED_ZERO_ONLY")

    def test_speed_window_and_total_directive_budget_are_bounded(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        bad_speed = bridge.plan_next(
            follow_directive(target_forward_m_s=0.10, max_forward_m_s=0.14),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(bad_speed.status, "BLOCKED_ZERO_ONLY")
        malformed_speed = bridge.plan_next(
            follow_directive(target_forward_m_s=None),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(malformed_speed.status, "BLOCKED_ZERO_ONLY")
        malformed_window = bridge.plan_next(
            follow_directive(max_command_window_s=None),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(malformed_window.status, "BLOCKED_ZERO_ONLY")
        too_fast = bridge.plan_next(
            follow_directive(target_forward_m_s=0.15, max_forward_m_s=0.15),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(too_fast.status, "BLOCKED_ZERO_ONLY")

        budget_bridge = RouteMotionBridge(route_frame_id="odom")
        short = follow_directive(max_command_window_s=1.5)
        first = budget_bridge.plan_next(short, make_pose(), readiness(), now_steady_s=10.0)
        self.assertEqual(first.maximum_duration_s, 1.0)
        budget_bridge.complete_window(first.window_id, actual_elapsed_s=1.0)
        active = readiness(observed_steady_s=11.0, source_mode_handshake_pass=True,
                           mode_desc="LOCOMOTION_DEFAULT", selected_source=VELOCITY_SOURCE_NAME)
        second = budget_bridge.plan_next(
            short, make_pose(stamp=2_000_000_000, received=11.0), active, now_steady_s=11.0,
        )
        self.assertAlmostEqual(second.maximum_duration_s, 0.5)
        budget_bridge.complete_window(second.window_id, actual_elapsed_s=0.5)
        exhausted = budget_bridge.plan_next(
            short, make_pose(stamp=3_000_000_000, received=11.1),
            replace(active, observed_steady_s=11.1), now_steady_s=11.1,
        )
        self.assertEqual(exhausted.status, "WINDOW_BUDGET_EXHAUSTED_ZERO_ONLY")

    def test_window_completion_rejects_overrun_and_wrong_id(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        plan = bridge.plan_next(follow_directive(), make_pose(), readiness(), now_steady_s=10.0)
        with self.assertRaisesRegex(RouteMotionBridgeError, "does not match"):
            bridge.complete_window(plan.window_id + 1, actual_elapsed_s=0.5)
        with self.assertRaisesRegex(RouteMotionBridgeError, "exceeded its cap"):
            bridge.complete_window(plan.window_id, actual_elapsed_s=1.01)


class PhysicalStopEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = {
            "status": "PASS",
            "window_start_steady_s": 12.0,
            "window_end_steady_s": 14.0,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.20,
            "generation": "episode-1",
        }

    def test_fresh_two_second_same_generation_gate_passes(self) -> None:
        self.assertTrue(validate_physical_stop_evidence(
            self.evidence, zero_request_started_s=11.0, now_steady_s=14.1,
            expected_generation="episode-1",
        ))

    def test_zero_plan_or_invalid_stop_window_never_counts_as_physical_stop(self) -> None:
        cases = (
            {"status": "NOT_CHECKED"},
            {"window_start_steady_s": 10.0},
            {"window_end_steady_s": 13.0},
            {"sample_count": 19},
            {"maximum_sample_gap_s": 0.16},
            {"maximum_state_age_s": 0.51},
            {"generation": "episode-2"},
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                evidence = dict(self.evidence)
                evidence.update(mutation)
                self.assertFalse(validate_physical_stop_evidence(
                    evidence, zero_request_started_s=11.0, now_steady_s=14.1,
                    expected_generation="episode-1",
                ))


if __name__ == "__main__":
    unittest.main()
