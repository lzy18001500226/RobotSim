#!/usr/bin/env python3
"""Offline tests for route-intent to bounded M1 proposal translation."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import re
import unittest

from issue80_m1_first_motion import (
    ACTION_SOURCE_NAME,
    COMMAND_PUBLISHER_MESSAGE_TYPE,
    COMMAND_PUBLISHER_TOPIC,
    SOURCE_PRIORITY,
    SOURCE_TIMEOUT_MS,
    VELOCITY_SOURCE_NAME,
)
from issue80_waypoint_route import (
    DirectiveKind, MotionFeedback, MotionStatus, PhysicalStopEvidence,
    PhysicalStopStatus, PoseEstimate, Profile, RouteDirective, RouteState,
    SafetyObservation, WaypointRouteExecutor,
)
from issue80_route_motion_bridge import (
    MAX_INCREMENTAL_WINDOW_S,
    RouteMotionBridge,
    RouteMotionBridgeError,
    RuntimeReadiness,
    WAYPOINT_B_MAX_WINDOWS,
    WAYPOINT_B_POSITION_TOLERANCE_M,
    WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M,
    WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
    WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
    WAYPOINT_B_CORRIDOR_MAX_FORWARD_M,
    WAYPOINT_B_MAX_CROSS_TRACK_M,
    build_short_right_corner_route,
    measured_m1_nonzero_window_s,
    waypoint_b_arrival_passes,
    waypoint_b_arrival_after_verified_stop_passes,
    waypoint_b_arrival_trigger_radius_m,
    waypoint_b_forward_guard_reached,
    waypoint_b_cross_track_guard_reached,
    waypoint_bearing_angular_command,
    TransitSteeringGate,
    active_locomotion_source_readbacks_valid,
    validate_physical_stop_evidence,
    waypoint_b_window_budget_passes,
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
        measured_displacement_guard_verified=False,
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
    def test_steering_waits_for_warmup_and_fresh_mode_source_handshake(self):
        gate = TransitSteeringGate(warmup_s=0.5)
        self.assertFalse(gate.observe(10.0, False)["steering_enabled"])
        self.assertFalse(gate.observe(10.25, True)["steering_enabled"])
        self.assertTrue(gate.observe(10.5, True)["steering_enabled"])

    def test_established_handshake_tolerates_fresh_async_readback_timestamp_skew(self):
        gate = TransitSteeringGate(warmup_s=0.5)
        self.assertFalse(gate.observe(10.0, True, active_readbacks_valid=True)["steering_enabled"])
        intermediate = gate.observe(10.25, False, active_readbacks_valid=True)
        self.assertIsNone(intermediate["fault"])
        self.assertFalse(intermediate["steering_enabled"])
        self.assertTrue(gate.observe(10.5, False, active_readbacks_valid=True)["steering_enabled"])

    def test_active_locomotion_readbacks_require_fresh_consistent_values_without_skew_gate(self):
        snapshot = {
            "sample_monotonic_s": 10.213,
            "readback_monotonic_s": {"get_mode": 10.0, "get_source": 10.212},
            "state_age_s": {"common": 0.005},
            "get_mode_header_code": 0,
            "get_mode_status_value": 100,
            "get_mode_desc": "LOCOMOTION_DEFAULT",
            "common_mode_status_value": 100,
            "common_mode_desc": "LOCOMOTION_DEFAULT",
            "get_source_header_code": 0,
            "get_source_name": VELOCITY_SOURCE_NAME,
            "common_source_name": VELOCITY_SOURCE_NAME,
            "get_mode_readback_fault": None,
            "get_source_readback_fault": None,
        }
        self.assertTrue(active_locomotion_source_readbacks_valid(
            snapshot, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
        ))
        stale = dict(snapshot)
        stale["readback_monotonic_s"] = {"get_mode": 9.7, "get_source": 10.212}
        self.assertFalse(active_locomotion_source_readbacks_valid(
            stale, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
        ))
        conflict = dict(snapshot)
        conflict["common_source_name"] = "another.source"
        self.assertFalse(active_locomotion_source_readbacks_valid(
            conflict, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
        ))

    def test_steering_gate_fails_closed_when_handshake_is_missing_at_warmup_deadline(self):
        gate = TransitSteeringGate(warmup_s=0.5)
        gate.observe(10.0, False)
        state = gate.observe(10.5, False)
        self.assertFalse(state["steering_enabled"])
        self.assertEqual(
            state["fault"],
            "transit_locomotion_source_mode_handshake_not_confirmed_after_0p5s_straight_warmup",
        )

    def test_steering_gate_fails_closed_if_handshake_is_lost(self):
        gate = TransitSteeringGate(warmup_s=0.5)
        gate.observe(10.0, True)
        gate.observe(10.5, True)
        state = gate.observe(10.6, False)
        self.assertFalse(state["steering_enabled"])
        self.assertEqual(state["fault"], "transit_locomotion_source_mode_handshake_lost")

    def test_fresh_waypoint_bearing_feedback_changes_with_each_odom_pose(self):
        centered = waypoint_bearing_angular_command(
            current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
            target_x_m=0.40, target_y_m=0.0,
        )
        displaced_left = waypoint_bearing_angular_command(
            current_x_m=0.10, current_y_m=0.02, current_yaw_rad=0.0,
            target_x_m=0.40, target_y_m=0.0,
        )
        self.assertAlmostEqual(centered["angular_rad_s"], 0.0)
        self.assertLess(displaced_left["angular_rad_s"], 0.0)
        self.assertNotEqual(centered["angular_rad_s"], displaced_left["angular_rad_s"])
        self.assertAlmostEqual(
            displaced_left["angular_rad_s"], 0.8 * displaced_left["bearing_error_rad"]
        )

    def test_fresh_waypoint_bearing_feedback_wraps_and_respects_angular_cap(self):
        right_turn = waypoint_bearing_angular_command(
            current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
            target_x_m=0.0, target_y_m=1.0,
        )
        saturated = waypoint_bearing_angular_command(
            current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
            target_x_m=-1.0, target_y_m=0.0,
        )
        self.assertGreater(right_turn["angular_rad_s"], 0.0)
        self.assertTrue(saturated["saturated"])
        self.assertAlmostEqual(saturated["angular_rad_s"], 0.075)

    def test_fresh_waypoint_bearing_feedback_rejects_nonfinite_or_invalid_bounds(self):
        with self.assertRaisesRegex(RouteMotionBridgeError, "finite"):
            waypoint_bearing_angular_command(
                current_x_m=float("nan"), current_y_m=0.0, current_yaw_rad=0.0,
                target_x_m=1.0, target_y_m=0.0,
            )
        with self.assertRaisesRegex(RouteMotionBridgeError, "limits"):
            waypoint_bearing_angular_command(
                current_x_m=0.0, current_y_m=0.0, current_yaw_rad=0.0,
                target_x_m=1.0, target_y_m=0.0, maximum_angular_rad_s=0.0,
            )

    def test_route_window_diagnostics_have_no_stale_window_angular_reference(self):
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\bwindow_angular\b", source))

    def test_short_diagnostic_waypoint_is_inside_tolerance_of_measured_stop_pose(self) -> None:
        start = make_pose(
            x=0.018430078705869258,
            y=0.0034944242082286378,
            yaw=0.04873396980315637,
        )
        route = build_short_right_corner_route(
            route_id="short-waypoint-diagnostic",
            start_pose=start,
            robot_footprint_radius_m=0.41565748847422734,
            local_clear_radius_m=2.0,
            right_turn_qualified=False,
            waypoint_b_only=True,
            waypoint_b_distance_m=WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M,
        )
        measured_stop = make_pose(
            x=0.12341386802163214,
            y=0.0021726180301854946,
            yaw=-0.022873450552313584,
            stamp=2_000_000_000,
            received=30.0,
        )
        target = route.steps[0]
        dx = measured_stop.x_m - start.x_m
        dy = measured_stop.y_m - start.y_m
        forward = dx * math.cos(start.yaw_rad) + dy * math.sin(start.yaw_rad)
        cross_track = -dx * math.sin(start.yaw_rad) + dy * math.cos(start.yaw_rad)
        yaw_delta = math.atan2(
            math.sin(measured_stop.yaw_rad - start.yaw_rad),
            math.cos(measured_stop.yaw_rad - start.yaw_rad),
        )
        target_error = math.hypot(target.x_m - measured_stop.x_m, target.y_m - measured_stop.y_m)

        self.assertEqual(len(route.steps), 1)
        self.assertEqual(target.stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertAlmostEqual(target.position_tolerance_m, WAYPOINT_B_POSITION_TOLERANCE_M)
        self.assertAlmostEqual(math.hypot(target.x_m - start.x_m, target.y_m - start.y_m), 0.20)
        self.assertLess(target_error, WAYPOINT_B_POSITION_TOLERANCE_M)
        self.assertLess(abs(cross_track), 0.05)
        self.assertAlmostEqual(forward, 0.1047947543806907)
        self.assertGreaterEqual(
            forward,
            WAYPOINT_B_SHORT_DIAGNOSTIC_DISTANCE_M - WAYPOINT_B_POSITION_TOLERANCE_M,
        )
        self.assertAlmostEqual(yaw_delta, -0.071607420356472)

    def test_waypoint_b_window_budget_separates_short_diagnostic_from_original_acceptance(self):
        self.assertFalse(waypoint_b_window_budget_passes(1, short_diagnostic=False))
        self.assertTrue(waypoint_b_window_budget_passes(2, short_diagnostic=False))
        self.assertTrue(waypoint_b_window_budget_passes(1, short_diagnostic=True))
        self.assertTrue(waypoint_b_window_budget_passes(2, short_diagnostic=True))
        self.assertFalse(waypoint_b_window_budget_passes(3, short_diagnostic=True))
        self.assertFalse(waypoint_b_window_budget_passes(0, short_diagnostic=True))
        self.assertFalse(waypoint_b_window_budget_passes(True, short_diagnostic=True))
        self.assertEqual(WAYPOINT_B_MAX_WINDOWS, 2)

    def test_extended_b_budget_is_separate_and_capped_at_three_stopped_windows(self):
        self.assertTrue(waypoint_b_window_budget_passes(2, short_diagnostic=False))
        self.assertFalse(waypoint_b_window_budget_passes(3, short_diagnostic=False))
        self.assertTrue(waypoint_b_window_budget_passes(1, short_diagnostic=False, extended_diagnostic=True))
        self.assertTrue(waypoint_b_window_budget_passes(3, short_diagnostic=False, extended_diagnostic=True))
        self.assertFalse(waypoint_b_window_budget_passes(4, short_diagnostic=False, extended_diagnostic=True))
        self.assertFalse(waypoint_b_window_budget_passes(1, short_diagnostic=True, extended_diagnostic=True))
        self.assertEqual(WAYPOINT_B_EXTENDED_DIAGNOSTIC_MAX_WINDOWS, 3)
        self.assertEqual(WAYPOINT_B_EXTENDED_DIAGNOSTIC_AGGREGATE_TRANSIT_MAX_S, 15.0)
        self.assertEqual(WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M, 0.055)

    def test_extended_b_progress_guard_preserves_corridor_and_stop_margin(self):
        self.assertFalse(waypoint_b_forward_guard_reached(
            WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M - 1e-6,
        ))
        self.assertTrue(waypoint_b_forward_guard_reached(
            WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
        ))
        self.assertAlmostEqual(
            WAYPOINT_B_CORRIDOR_MAX_FORWARD_M - WAYPOINT_B_EXTENDED_DIAGNOSTIC_FORWARD_GUARD_M,
            0.15,
        )

    def test_waypoint_b_cross_track_guard_matches_declared_50mm_centerline_bound(self):
        self.assertFalse(waypoint_b_cross_track_guard_reached(WAYPOINT_B_MAX_CROSS_TRACK_M))
        self.assertFalse(waypoint_b_cross_track_guard_reached(-WAYPOINT_B_MAX_CROSS_TRACK_M))
        self.assertTrue(waypoint_b_cross_track_guard_reached(WAYPOINT_B_MAX_CROSS_TRACK_M + 1e-6))
        self.assertTrue(waypoint_b_cross_track_guard_reached(-WAYPOINT_B_MAX_CROSS_TRACK_M - 1e-6))

    def test_extended_b_stop_trigger_uses_measured_reserve_without_relaxing_arrival(self):
        measured_post_command_excursion_m = 0.048370466515190544
        trigger_radius_m = waypoint_b_arrival_trigger_radius_m(
            acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
            stop_reserve_m=WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M,
        )
        self.assertEqual(trigger_radius_m, 0.095)
        self.assertGreater(WAYPOINT_B_EXTENDED_DIAGNOSTIC_STOP_RESERVE_M, measured_post_command_excursion_m)
        self.assertEqual(WAYPOINT_B_POSITION_TOLERANCE_M, 0.15)

    def test_short_waypoint_trigger_reserves_measured_poststop_travel_without_changing_acceptance(self):
        measured_stop_tail_m = 0.03681105376544568
        trigger_radius_m = waypoint_b_arrival_trigger_radius_m(
            acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
            stop_reserve_m=WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M,
        )

        self.assertAlmostEqual(WAYPOINT_B_POSITION_TOLERANCE_M, 0.15)
        self.assertAlmostEqual(WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M, 0.045)
        self.assertAlmostEqual(trigger_radius_m, 0.105)
        self.assertGreater(WAYPOINT_B_SHORT_DIAGNOSTIC_STOP_RESERVE_M, measured_stop_tail_m)
        self.assertLessEqual(trigger_radius_m + measured_stop_tail_m, WAYPOINT_B_POSITION_TOLERANCE_M)
        self.assertAlmostEqual(
            waypoint_b_arrival_trigger_radius_m(
                acceptance_tolerance_m=WAYPOINT_B_POSITION_TOLERANCE_M,
                stop_reserve_m=0.0,
            ),
            WAYPOINT_B_POSITION_TOLERANCE_M,
        )

    def test_waypoint_arrival_requires_trigger_and_settled_pose_within_tolerance(self):
        self.assertTrue(waypoint_b_arrival_passes(0.10, 0.15, tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(0.10, 0.150001, tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(0.150001, 0.10, tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(None, 0.10, tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(float("nan"), 0.10, tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(0.10, float("inf"), tolerance_m=0.15))
        self.assertFalse(waypoint_b_arrival_passes(0.10, 0.10, tolerance_m=0.0))

    def test_poststop_arrival_reuses_only_a_previously_passed_composite_stop(self):
        self.assertTrue(waypoint_b_arrival_after_verified_stop_passes(
            0.112, 0.113, tolerance_m=0.15,
            composite_stop_gate_passed=True,
            bridge_rearmed=True,
            fresh_stand_readback_passed=True,
            source_readback_passed=True,
            safety_gates_passed=True,
        ))
        for gate in (
            {"composite_stop_gate_passed": False},
            {"bridge_rearmed": False},
            {"fresh_stand_readback_passed": False},
            {"source_readback_passed": False},
            {"safety_gates_passed": False},
        ):
            fields = {
                "composite_stop_gate_passed": True,
                "bridge_rearmed": True,
                "fresh_stand_readback_passed": True,
                "source_readback_passed": True,
                "safety_gates_passed": True,
            }
            fields.update(gate)
            self.assertFalse(waypoint_b_arrival_after_verified_stop_passes(
                0.112, 0.113, tolerance_m=0.15, **fields,
            ))
        self.assertFalse(waypoint_b_arrival_after_verified_stop_passes(
            0.112, 0.150001, tolerance_m=0.15,
            composite_stop_gate_passed=True,
            bridge_rearmed=True,
            fresh_stand_readback_passed=True,
            source_readback_passed=True,
            safety_gates_passed=True,
        ))

    def test_waypoint_b_only_route_reaches_b_after_measured_two_window_transit_and_stop(self) -> None:
        start = make_pose()
        route = build_short_right_corner_route(
            route_id="waypoint-b-only",
            start_pose=start,
            robot_footprint_radius_m=0.41565748847422734,
            local_clear_radius_m=2.0,
            right_turn_qualified=False,
            waypoint_b_only=True,
            waypoint_b_distance_m=0.40,
        )
        self.assertEqual(len(route.steps), 1)
        self.assertFalse(route.bounded_moving_turn_qualified)
        target = route.steps[0]
        self.assertEqual(target.waypoint_id, "TRANSIT-TO-CORNER")
        self.assertEqual(target.stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertAlmostEqual(target.position_tolerance_m, 0.15)
        self.assertAlmostEqual(math.hypot(target.x_m - start.x_m, target.y_m - start.y_m), 0.40)
        self.assertAlmostEqual(target.max_command_window_s, 10.0)

        class PoseStore:
            def __init__(self):
                self.pose = start

            def latest_pose(self, target_frame, now_steady_s):
                return self.pose if target_frame == "odom" else None

        class SafetyStore:
            def latest_observation(self, segment_id, frame_id, now_steady_s):
                return SafetyObservation(segment_id, frame_id, now_steady_s, True, True, 2.0)

        class MotionStore:
            def dispatch(self, directive):
                self.latest = directive

            def request_zero_and_stop(self, reason, requested_steady_s):
                return requested_steady_s

        pose_store = PoseStore()
        executor = WaypointRouteExecutor(route, pose_store, SafetyStore(), MotionStore())
        directive = executor.start(10.0, MotionFeedback(
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            initial_activation_authorized=True,
        ))
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0)
        first = bridge.plan_next(directive, pose_store.pose, readiness(), now_steady_s=10.0)
        self.assertEqual(first.maximum_duration_s, 5.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=4.98)

        first_stop = {
            "status": "PASS",
            "window_start_steady_s": 15.01,
            "window_end_steady_s": 16.92,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(directive, now_steady_s=15.0, reason="first B transit window")
        self.assertTrue(bridge.confirm_physical_stop(
            first_stop, zero_request_started_s=15.0, now_steady_s=16.95,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(x=0.127, stamp=2_000_000_000, received=16.95)
        follow = executor.tick(16.95, MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 15.01, 16.92, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(follow.kind, DirectiveKind.FOLLOW_WAYPOINT)
        second = bridge.plan_next(
            follow, pose_store.pose,
            readiness(observed_steady_s=16.95), now_steady_s=16.95,
        )
        self.assertEqual(second.maximum_duration_s, 5.0)
        bridge.complete_window(second.window_id, actual_elapsed_s=4.98)

        arrival_pose = make_pose(x=0.27, stamp=3_000_000_000, received=22.0)
        pose_store.pose = arrival_pose
        arrival = executor.tick(22.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(arrival.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertLessEqual(math.hypot(
            target.x_m - arrival_pose.x_m, target.y_m - arrival_pose.y_m,
        ), target.position_tolerance_m)

        b_stop = {
            "status": "PASS",
            "window_start_steady_s": 22.01,
            "window_end_steady_s": 23.92,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(arrival, now_steady_s=22.0, reason="waypoint B reached")
        self.assertTrue(bridge.confirm_physical_stop(
            b_stop, zero_request_started_s=22.0, now_steady_s=24.0,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(x=0.27, stamp=4_000_000_000, received=24.0)
        complete = executor.tick(24.0, MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 22.01, 23.92, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(executor.state, RouteState.COMPLETE)
        self.assertEqual(complete.kind, DirectiveKind.COMPLETE)

    def test_short_right_corner_geometry_derives_corridors_from_footprint_and_arc(self) -> None:
        start = make_pose(x=1.0, y=-0.5, yaw=0.4)
        footprint = 0.41565748847422734
        route = build_short_right_corner_route(
            route_id="geometry-fixture",
            start_pose=start,
            robot_footprint_radius_m=footprint,
            local_clear_radius_m=2.0,
            right_turn_qualified=True,
        )
        self.assertEqual([step.profile for step in route.steps], [
            Profile.TRANSIT, Profile.CORNER, Profile.TRANSIT,
        ])
        self.assertEqual(route.steps[-1].stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertTrue(route.bounded_moving_turn_qualified)
        self.assertFalse(route.zero_forward_turn_qualified)
        self.assertAlmostEqual(route.steps[0].position_tolerance_m, 0.15)
        self.assertAlmostEqual(route.steps[-1].position_tolerance_m, 0.15)
        self.assertAlmostEqual(
            math.hypot(route.steps[0].x_m - start.x_m,
                       route.steps[0].y_m - start.y_m),
            0.35,
            places=7,
        )
        self.assertAlmostEqual(
            math.hypot(route.steps[-1].x_m - route.steps[0].x_m,
                       route.steps[-1].y_m - route.steps[0].y_m),
            0.45,
            places=7,
        )
        self.assertGreater(
            math.hypot(route.steps[-1].x_m - route.steps[0].x_m,
                       route.steps[-1].y_m - route.steps[0].y_m)
            - route.steps[0].position_tolerance_m
            - route.steps[-1].position_tolerance_m,
            0.0,
        )
        self.assertAlmostEqual(route.steps[0].max_command_window_s, 10.0)
        self.assertAlmostEqual(route.steps[-1].max_command_window_s, 10.0)
        self.assertAlmostEqual(route.steps[1].max_command_window_s, 5.0)
        self.assertAlmostEqual(sum(step.max_command_window_s for step in route.steps), 25.0)
        self.assertAlmostEqual(
            math.hypot(route.steps[-1].x_m - route.steps[0].x_m,
                       route.steps[-1].y_m - route.steps[0].y_m)
            - route.steps[1].maximum_turn_displacement_m,
            0.20,
            places=7,
        )
        self.assertAlmostEqual(route.steps[1].maximum_turn_displacement_m, 0.25)
        corner_heading = math.atan2(
            route.steps[2].y_m - route.steps[1].y_m,
            route.steps[2].x_m - route.steps[1].x_m,
        )
        self.assertAlmostEqual(corner_heading, start.yaw_rad - 0.12, places=7)

        def lateral_span(polygon, center_x, center_y, heading):
            values = [
                -(x - center_x) * math.sin(heading) + (y - center_y) * math.cos(heading)
                for x, y in polygon
            ]
            return max(values) - min(values)

        straight_span = lateral_span(route.steps[0].corridor_polygon, start.x_m, start.y_m, start.yaw_rad)
        expected_straight_span = 2.0 * (footprint + 0.10 + 0.05)
        self.assertAlmostEqual(straight_span, expected_straight_span, places=7)
        turn_span = lateral_span(
            route.steps[1].turn_corridor_polygon,
            route.steps[1].x_m,
            route.steps[1].y_m,
            start.yaw_rad,
        )
        expected_lateral = (0.12 / 0.075) * (1.0 - math.cos(0.075 * 4.5))
        self.assertAlmostEqual(
            turn_span,
            2.0 * (footprint + 0.10 + expected_lateral + 0.05),
            places=7,
        )

    def test_short_right_corner_geometry_rejects_insufficient_clearance(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain the swept route corridor"):
            build_short_right_corner_route(
                route_id="too-small",
                start_pose=make_pose(),
                robot_footprint_radius_m=0.415,
                local_clear_radius_m=1.0,
                right_turn_qualified=True,
            )

    def test_route_manager_bridge_transit_stop_and_right_corner_lifecycle(self) -> None:
        route = build_short_right_corner_route(
            route_id="integrated-route-fixture",
            start_pose=make_pose(),
            robot_footprint_radius_m=0.41565748847422734,
            local_clear_radius_m=2.0,
            right_turn_qualified=True,
        )

        class PoseStore:
            def __init__(self):
                self.pose = make_pose()

            def latest_pose(self, target_frame, now_steady_s):
                return self.pose if target_frame == "odom" else None

        class SafetyStore:
            def latest_observation(self, segment_id, frame_id, now_steady_s):
                return SafetyObservation(segment_id, frame_id, now_steady_s, True, True, 2.0)

        class MotionStore:
            def __init__(self):
                self.latest = None

            def dispatch(self, directive):
                self.latest = directive

            def request_zero_and_stop(self, reason, requested_steady_s):
                return requested_steady_s

        pose_store = PoseStore()
        motion_store = MotionStore()
        executor = WaypointRouteExecutor(route, pose_store, SafetyStore(), motion_store)
        start_feedback = MotionFeedback(
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            initial_activation_authorized=True,
        )
        transit = executor.start(10.0, start_feedback)
        self.assertEqual(transit.kind, DirectiveKind.FOLLOW_WAYPOINT)

        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0)
        transit_plan = bridge.plan_next(transit, pose_store.pose, readiness(), now_steady_s=10.0)
        self.assertEqual((transit_plan.forward_m_s, transit_plan.angular_rad_s), (0.12, 0.0))
        bridge.complete_window(transit_plan.window_id, actual_elapsed_s=1.0)

        pose_store.pose = make_pose(x=0.40, stamp=2_000_000_000, received=11.0)
        arrived = executor.tick(11.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(arrived.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        stop = {
            "status": "PASS",
            "window_start_steady_s": 11.01,
            "window_end_steady_s": 12.91,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        zero_plan = bridge.zero_window(
            arrived, now_steady_s=11.0, reason="transit waypoint reached",
        )
        self.assertTrue(bridge.confirm_physical_stop(
            stop, zero_request_started_s=11.0, now_steady_s=13.0,
            expected_generation="episode-1",
        ))

        pose_store.pose = make_pose(x=0.40, stamp=3_000_000_000, received=13.0)
        next_follow = executor.tick(13.0, MotionFeedback(
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 11.01, 12.91, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(next_follow.kind, DirectiveKind.FOLLOW_WAYPOINT)
        pose_store.pose = make_pose(x=0.40, stamp=4_000_000_000, received=13.1)
        turn = executor.tick(13.1, MotionFeedback(
            status=MotionStatus.ARRIVED,
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 11.01, 12.91, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(turn.kind, DirectiveKind.BOUNDED_MOVING_TURN)

        turn_ready = readiness(
            observed_steady_s=13.1,
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.05,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="right-turn-attempt-a2",
            measured_displacement_guard_verified=True,
        )
        turn_plan = bridge.plan_next(turn, pose_store.pose, turn_ready, now_steady_s=13.1)
        self.assertEqual(turn_plan.status, "PROPOSAL_ONLY")
        self.assertEqual((turn_plan.forward_m_s, turn_plan.angular_rad_s), (0.12, -0.075))
        self.assertEqual(turn_plan.maximum_duration_s, 5.0)
        bridge.complete_window(turn_plan.window_id, actual_elapsed_s=0.9)

        pose_store.pose = make_pose(x=0.60, y=-0.02, yaw=-0.12, stamp=5_000_000_000, received=14.0)
        turn_stop = executor.tick(14.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(turn_stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        turn_stop_evidence = {
            "status": "PASS",
            "window_start_steady_s": 14.01,
            "window_end_steady_s": 15.91,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(turn_stop, now_steady_s=14.0, reason="corner yaw reached")
        self.assertTrue(bridge.confirm_physical_stop(
            turn_stop_evidence, zero_request_started_s=14.0, now_steady_s=16.0,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(x=0.60, y=-0.02, yaw=-0.12, stamp=6_000_000_000, received=16.0)
        continuation = executor.tick(16.0, MotionFeedback(
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 14.01, 15.91, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(continuation.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(executor.step_index, 2)
        self.assertEqual(executor.active_step.waypoint_id, "TRANSIT-AFTER-RIGHT-CORNER")

        continuation_plan = bridge.plan_next(
            continuation, pose_store.pose,
            readiness(observed_steady_s=16.0), now_steady_s=16.0,
        )
        self.assertEqual(continuation_plan.status, "PROPOSAL_ONLY")
        self.assertEqual(continuation_plan.forward_m_s, 0.12)
        self.assertLessEqual(abs(continuation_plan.angular_rad_s), 0.075)
        bridge.complete_window(continuation_plan.window_id, actual_elapsed_s=2.0)

        c_step = route.steps[2]
        pose_store.pose = make_pose(
            x=c_step.x_m - 0.08 * math.cos(-0.12),
            y=c_step.y_m - 0.08 * math.sin(-0.12),
            yaw=-0.12, stamp=7_000_000_000, received=17.0,
        )
        c_stop = executor.tick(17.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(c_stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        c_stop_evidence = {
            "status": "PASS",
            "window_start_steady_s": 17.01,
            "window_end_steady_s": 18.92,
            "sample_count": 40,
            "maximum_sample_gap_s": 0.08,
            "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(c_stop, now_steady_s=17.0, reason="waypoint C reached")
        self.assertTrue(bridge.confirm_physical_stop(
            c_stop_evidence, zero_request_started_s=17.0, now_steady_s=19.1,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(
            x=pose_store.pose.x_m, y=pose_store.pose.y_m, yaw=-0.12,
            stamp=8_000_000_000, received=19.1,
        )
        completed = executor.tick(19.1, MotionFeedback(
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 17.01, 18.92, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(executor.state, RouteState.COMPLETE)
        self.assertEqual(completed.kind, DirectiveKind.COMPLETE)

    def test_measured_m1_window_uses_last_nonzero_offset_not_post_deadline_loop_time(self) -> None:
        self.assertAlmostEqual(
            measured_m1_nonzero_window_s({
                "last_nonzero_command_offset_s": 4.98659384198254,
                "pulse_loop_elapsed_s": 5.000185087963473,
            }),
            4.98659384198254,
        )
        self.assertAlmostEqual(
            measured_m1_nonzero_window_s({"pulse_actual_duration_s": 0.98}),
            0.98,
        )
        with self.assertRaisesRegex(RouteMotionBridgeError, "lacks a positive measured"):
            measured_m1_nonzero_window_s({"pulse_loop_elapsed_s": 5.0})

    def test_transit_can_replan_after_a_verified_five_second_stop_window(self) -> None:
        route = build_short_right_corner_route(
            route_id="multi-window-transit",
            start_pose=make_pose(),
            robot_footprint_radius_m=0.41565748847422734,
            local_clear_radius_m=2.0,
            right_turn_qualified=True,
        )

        class PoseStore:
            def __init__(self):
                self.pose = make_pose()

            def latest_pose(self, target_frame, now_steady_s):
                return self.pose if target_frame == "odom" else None

        class SafetyStore:
            def latest_observation(self, segment_id, frame_id, now_steady_s):
                return SafetyObservation(segment_id, frame_id, now_steady_s, True, True, 2.0)

        class MotionStore:
            def dispatch(self, directive):
                self.latest = directive

            def request_zero_and_stop(self, reason, requested_steady_s):
                return requested_steady_s

        pose_store = PoseStore()
        motion_store = MotionStore()
        executor = WaypointRouteExecutor(route, pose_store, SafetyStore(), motion_store)
        directive = executor.start(10.0, MotionFeedback(
            reset_gate_pass=True, standing_gate_pass=True,
            source_mode_handshake_pass=False, safety_gates_pass=True,
            initial_activation_authorized=True,
        ))
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0)
        first = bridge.plan_next(directive, pose_store.pose, readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=4.98)

        first_stop = {
            "status": "PASS", "window_start_steady_s": 15.01,
            "window_end_steady_s": 16.92, "sample_count": 40,
            "maximum_sample_gap_s": 0.08, "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(directive, now_steady_s=15.0, reason="first bounded transit window ended")
        self.assertTrue(bridge.confirm_physical_stop(
            first_stop, zero_request_started_s=15.0, now_steady_s=16.95,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(x=0.127, stamp=2_000_000_000, received=16.95)
        first_feedback = MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 15.01, 16.92, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True, standing_gate_pass=True,
            source_mode_handshake_pass=False, safety_gates_pass=True,
            stand_readback_pass=True,
        )
        next_directive = executor.tick(16.95, first_feedback)
        self.assertEqual(next_directive.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(executor.step_index, 0)

        second = bridge.plan_next(
            next_directive, pose_store.pose,
            readiness(observed_steady_s=16.95), now_steady_s=16.95,
        )
        self.assertEqual(second.status, "PROPOSAL_ONLY")
        self.assertEqual(second.maximum_duration_s, 5.0)
        bridge.complete_window(second.window_id, actual_elapsed_s=4.98)

        pose_store.pose = make_pose(x=0.22, stamp=3_000_000_000, received=22.0)
        b_stop = executor.tick(22.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True, standing_gate_pass=True,
            source_mode_handshake_pass=True, safety_gates_pass=True,
        ))
        self.assertEqual(b_stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        b_evidence = {
            "status": "PASS", "window_start_steady_s": 22.01,
            "window_end_steady_s": 23.92, "sample_count": 40,
            "maximum_sample_gap_s": 0.08, "maximum_state_age_s": 0.2,
            "generation": "episode-1",
        }
        bridge.zero_window(b_stop, now_steady_s=22.0, reason="waypoint B reached")
        self.assertTrue(bridge.confirm_physical_stop(
            b_evidence, zero_request_started_s=22.0, now_steady_s=24.0,
            expected_generation="episode-1",
        ))
        pose_store.pose = make_pose(x=0.22, stamp=4_000_000_000, received=24.0)
        after_b_stop = executor.tick(24.0, MotionFeedback(
            physical_stop_evidence=PhysicalStopEvidence(
                PhysicalStopStatus.PASS, 22.01, 23.92, 40, 0.08, 0.2, "episode-1",
            ),
            reset_gate_pass=True, standing_gate_pass=True,
            source_mode_handshake_pass=False, safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(after_b_stop.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(executor.step_index, 1)

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
            measured_displacement_guard_verified=True,
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
            measured_displacement_guard_verified=True,
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

    def test_turn_needs_runtime_measured_displacement_guard_and_explicit_five_second_cap(self) -> None:
        qualified = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.05,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-envelope",
        )
        blocked = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0).plan_next(
            turn_directive(), make_pose(), qualified, now_steady_s=10.0,
        )
        self.assertEqual(blocked.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("measured-displacement", blocked.reason)

        guarded = replace(qualified, measured_displacement_guard_verified=True)
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0)
        plan = bridge.plan_next(turn_directive(), make_pose(), guarded, now_steady_s=10.0)
        self.assertEqual(plan.status, "PROPOSAL_ONLY")
        self.assertEqual(plan.maximum_duration_s, 5.0)
        self.assertTrue(plan.measured_displacement_guard_required)

    def test_route_action_window_remains_one_second_by_default(self) -> None:
        self.assertEqual(RouteMotionBridge(route_frame_id="odom").incremental_window_s, 1.0)
        with self.assertRaises(ValueError):
            RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.001)

    def test_right_turn_evidence_does_not_qualify_left_turn(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom")
        right_only = readiness(
            bounded_moving_turn_qualified=True,
            qualified_turn_directions=("RIGHT",),
            turn_stop_margin_m=0.005,
            turn_stop_margin_direction="RIGHT",
            turn_stop_margin_evidence_id="run65-right-stop-trace",
            measured_displacement_guard_verified=True,
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
            measured_displacement_guard_verified=True,
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
                measured_displacement_guard_verified=True,
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
