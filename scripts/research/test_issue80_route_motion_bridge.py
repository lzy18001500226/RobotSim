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
    CURVED_PATH_AGGREGATE_NONZERO_MAX_S,
    CURVED_PATH_FORWARD_M_S,
    CURVED_PATH_MAX_CROSS_TRACK_M,
    CURVED_PATH_MAX_WINDOW_S,
    CURVED_PATH_MAX_WINDOWS,
    CURVED_PATH_POSITION_TOLERANCE_M,
    CURVED_PATH_RADIUS_M,
    CURVED_PATH_STOP_RESERVE_M,
    CURVED_PATH_SWEEP_RAD,
    build_curved_waypoint_route,
    build_shallow_curved_path_reference,
    curved_path_metrics,
    classify_route_odom_stamp,
    RouteOdomFreshnessTracker,
    OdomSourceStampFence,
    route_feedback_sample_disposition,
    evaluate_route_odom_freshness_window,
    evaluate_route_window_rearm_gate,
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
    fresh_stand_mode_readback_valid,
    fresh_input_source_readback_valid,
    validate_physical_stop_evidence,
    waypoint_b_window_budget_passes,
    validate_curved_path_cli_profile,
    ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
    ZERO_FORWARD_HEADING_ACQUISITION_TRIGGER_RAD,
    zero_forward_heading_acquisition_plan,
    forward_gait_yaw_command,
    segment_lookahead_point,
    CONTINUOUS_TABLE_MAX_CROSS_TRACK_M,
    CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M,
    continuous_table_segment_cross_track_limit_m,
    FIXED_WINDOW_ARC_DEADBAND_RAD_S,
    FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD,
    fixed_window_arc_angular_command,
    fixed_window_heading_acquisition_required,
    route_replan_odom_freshness,
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
    def test_continuous_table_expanded_cross_track_profile_is_scoped_to_sixty_mm(self):
        self.assertEqual(CONTINUOUS_TABLE_MAX_CROSS_TRACK_M, 0.06)
        self.assertGreater(CONTINUOUS_TABLE_MAX_CROSS_TRACK_M, WAYPOINT_B_MAX_CROSS_TRACK_M)

    def test_continuous_table_open_transit_envelope_does_not_leak_to_table_approach(self):
        self.assertEqual(CONTINUOUS_TABLE_WAYPOINT_TRANSIT_MAX_CROSS_TRACK_M, 0.10)
        self.assertEqual(continuous_table_segment_cross_track_limit_m("WAYPOINT"), 0.10)
        self.assertEqual(continuous_table_segment_cross_track_limit_m("TABLE_APPROACH"), 0.06)
        with self.assertRaises(RouteMotionBridgeError):
            continuous_table_segment_cross_track_limit_m("TABLE_DOCKING")

    def test_fixed_window_arc_quantizes_only_fresh_prewindow_turn_requests(self):
        straight = fixed_window_arc_angular_command(
            FIXED_WINDOW_ARC_DEADBAND_RAD_S * 0.5
        )
        self.assertEqual(straight["mode"], "STRAIGHT")
        self.assertEqual(straight["angular_rad_s"], 0.0)
        right = fixed_window_arc_angular_command(-0.04)
        self.assertEqual(right["mode"], "FIXED_ARC")
        self.assertEqual(right["angular_rad_s"], -0.075)
        left = fixed_window_arc_angular_command(0.04)
        self.assertEqual(left["angular_rad_s"], 0.075)

    def test_fixed_window_arc_uses_current_pose_bearing_not_a_stale_prior_plan(self):
        fresh = waypoint_bearing_angular_command(
            current_x_m=-1.03,
            current_y_m=-1.526,
            current_yaw_rad=-0.028,
            target_x_m=-0.25,
            target_y_m=-1.5,
        )
        self.assertGreater(fresh["angular_rad_s"], 0.0)
        self.assertEqual(
            fixed_window_arc_angular_command(fresh["angular_rad_s"])["angular_rad_s"],
            0.075,
        )

    def test_fixed_window_yaw_acquisition_stays_in_open_waypoint_segment(self):
        self.assertAlmostEqual(FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE_RAD, .10)
        self.assertTrue(
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=True,
                segment_id="WAYPOINT",
                window_index=0,
            )
        )
        self.assertFalse(
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=True,
                segment_id="TABLE_APPROACH",
                window_index=0,
            )
        )
        self.assertFalse(
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=False,
                segment_id="WAYPOINT",
                window_index=0,
            )
        )
        self.assertFalse(
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=True,
                segment_id="WAYPOINT",
                window_index=1,
            )
        )
        with self.assertRaises(RouteMotionBridgeError):
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=True,
                segment_id=object(),  # type: ignore[arg-type]
                window_index=0,
            )
        with self.assertRaises(RouteMotionBridgeError):
            fixed_window_heading_acquisition_required(
                fixed_window_table_approach=True,
                segment_id="WAYPOINT",
                window_index=-1,
            )

    def test_route_replan_requires_new_fresh_odom_after_a_stop(self):
        snapshot = {
            "pose": {"stamp_sec": 12, "stamp_nanosec": 50},
            "state_age_s": {"odom": 0.01},
        }
        waiting = route_replan_odom_freshness(
            snapshot, previous_stamp_ns=12_000_000_050, maximum_age_s=0.5
        )
        self.assertEqual(waiting["status"], "WAITING_NO_ADVANCING_SAMPLE")
        fresh = route_replan_odom_freshness(
            snapshot, previous_stamp_ns=12_000_000_040, maximum_age_s=0.5
        )
        self.assertEqual(fresh["status"], "PASS")
        snapshot["state_age_s"]["odom"] = 0.6
        stale = route_replan_odom_freshness(
            snapshot, previous_stamp_ns=12_000_000_040, maximum_age_s=0.5
        )
        self.assertEqual(stale["status"], "WAITING_STALE")

    def test_segment_lookahead_advances_reference_without_changing_terminal_target(self):
        reference = segment_lookahead_point(
            segment_start_x_m=0.0,
            segment_start_y_m=0.0,
            segment_end_x_m=1.0,
            segment_end_y_m=0.0,
            current_x_m=0.30,
            current_y_m=-0.04,
            lookahead_m=0.20,
        )
        self.assertAlmostEqual(reference["x_m"], 0.50)
        self.assertAlmostEqual(reference["y_m"], 0.0)
        self.assertAlmostEqual(reference["along_track_progress_m"], 0.30)
        terminal = segment_lookahead_point(
            segment_start_x_m=0.0,
            segment_start_y_m=0.0,
            segment_end_x_m=1.0,
            segment_end_y_m=0.0,
            current_x_m=0.95,
            current_y_m=0.0,
            lookahead_m=0.20,
        )
        self.assertAlmostEqual(terminal["x_m"], 1.0)

    def test_zero_forward_heading_acquisition_uses_the_documented_start_band(self) -> None:
        right = zero_forward_heading_acquisition_plan(-0.04)
        self.assertTrue(right["required"])
        self.assertEqual(right["forward_velocity_m_s"], 0.0)
        self.assertEqual(
            right["angular_velocity_rad_s"],
            -ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S,
        )
        self.assertEqual(right["trigger_rad"], ZERO_FORWARD_HEADING_ACQUISITION_TRIGGER_RAD)

        left = zero_forward_heading_acquisition_plan(0.04)
        self.assertTrue(left["required"])
        self.assertEqual(left["angular_velocity_rad_s"], ZERO_FORWARD_HEADING_ACQUISITION_COMMAND_RAD_S)

        aligned = zero_forward_heading_acquisition_plan(0.02)
        self.assertFalse(aligned["required"])
        self.assertEqual(aligned["angular_velocity_rad_s"], 0.0)

    def test_forward_gait_yaw_mapping_keeps_stationary_and_gait_signs_separate(self) -> None:
        self.assertEqual(
            forward_gait_yaw_command(-0.075, invert_from_measured_response=False),
            -0.075,
        )
        self.assertEqual(
            forward_gait_yaw_command(-0.075, invert_from_measured_response=True),
            0.075,
        )
        with self.assertRaises(RouteMotionBridgeError):
            forward_gait_yaw_command(0.076, invert_from_measured_response=True)

    def test_verified_yaw_only_acquisition_still_requires_fresh_handshake(self) -> None:
        gate = TransitSteeringGate(warmup_s=0.5)
        gate.seed_verified_acquisition(started_steady_s=10.0, observed_steady_s=10.75)
        state = gate.observe(10.75, True, active_readbacks_valid=True)
        self.assertTrue(state["handshake_observed"])
        self.assertTrue(state["steering_enabled"])
        self.assertIsNone(state["fault"])

        lost = gate.observe(10.80, False, active_readbacks_valid=False)
        self.assertEqual(lost["fault"], "transit_locomotion_source_mode_handshake_lost")

    def test_route_adapter_skips_only_fresh_duplicate_odom_without_relaxing_core_stamps(self):
        self.assertEqual(
            classify_route_odom_stamp(
                previous_stamp_ns=10_000,
                current_stamp_ns=10_000,
                receive_age_s=0.0115,
                maximum_receive_age_s=0.20,
            ),
            "DUPLICATE",
        )
        self.assertEqual(
            classify_route_odom_stamp(
                previous_stamp_ns=10_000,
                current_stamp_ns=10_000,
                receive_age_s=0.201,
                maximum_receive_age_s=0.20,
            ),
            "STALE_OR_INVALID",
        )
        self.assertEqual(
            classify_route_odom_stamp(
                previous_stamp_ns=10_000,
                current_stamp_ns=10_001,
                receive_age_s=0.01,
                maximum_receive_age_s=0.20,
            ),
            "NEW",
        )
        self.assertEqual(
            classify_route_odom_stamp(
                previous_stamp_ns=10_000,
                current_stamp_ns=9_999,
                receive_age_s=0.01,
                maximum_receive_age_s=0.20,
            ),
            "REGRESSION",
        )

    def test_odometry_tracker_holds_one_fresh_duplicate_then_resumes_on_new_stamp(self):
        tracker = RouteOdomFreshnessTracker(10_000, 5.0, maximum_age_s=0.20)
        duplicate = tracker.observe(
            current_stamp_ns=10_000, sample_steady_s=5.03, received_steady_s=5.029,
        )
        self.assertEqual(duplicate["action"], "DUPLICATE")
        self.assertAlmostEqual(duplicate["age_since_last_unique_s"], 0.03)
        resumed = tracker.observe(
            current_stamp_ns=10_001, sample_steady_s=5.05, received_steady_s=5.049,
        )
        self.assertEqual(resumed["action"], "NEW")
        self.assertEqual(tracker.last_unique_stamp_ns, 10_001)
        self.assertAlmostEqual(tracker.last_unique_received_steady_s, 5.049)

    def test_consecutive_fresh_duplicate_deliveries_expire_from_last_unique_receipt(self):
        tracker = RouteOdomFreshnessTracker(20_000, 10.0, maximum_age_s=0.20)
        outcomes = [
            tracker.observe(
                current_stamp_ns=20_000,
                sample_steady_s=sample_time,
                received_steady_s=sample_time - 0.001,
            )
            for sample_time in (10.04, 10.10, 10.18)
        ]
        self.assertEqual([row["action"] for row in outcomes], ["DUPLICATE"] * 3)
        self.assertTrue(all(row["message_receive_age_s"] <= 0.20 for row in outcomes))
        self.assertAlmostEqual(outcomes[-1]["age_since_last_unique_s"], 0.18)
        expired = tracker.observe(
            current_stamp_ns=20_000,
            sample_steady_s=10.205,
            received_steady_s=10.204,
        )
        self.assertEqual(expired["action"], "STALE_OR_INVALID")
        self.assertEqual(expired["source_stamp_relation"], "DUPLICATE")
        self.assertAlmostEqual(expired["message_receive_age_s"], 0.001)
        self.assertAlmostEqual(expired["age_since_last_unique_s"], 0.205)
        self.assertEqual(tracker.last_unique_stamp_ns, 20_000)

    def test_odometry_stamp_regression_fails_closed_without_rewinding_unique_anchor(self):
        tracker = RouteOdomFreshnessTracker(30_000, 20.0, maximum_age_s=0.20)
        regression = tracker.observe(
            current_stamp_ns=29_999, sample_steady_s=20.02, received_steady_s=20.019,
        )
        self.assertEqual(regression["action"], "REGRESSION")
        self.assertEqual(tracker.last_unique_stamp_ns, 30_000)
        resumed = tracker.observe(
            current_stamp_ns=30_001, sample_steady_s=20.04, received_steady_s=20.039,
        )
        self.assertEqual(resumed["action"], "NEW")

    def test_odom_input_fence_discards_out_of_order_packet_without_refreshing_anchor(self):
        fence = OdomSourceStampFence()
        initial = fence.observe(current_stamp_ns=30_000, received_steady_s=20.0)
        newer = fence.observe(current_stamp_ns=30_010, received_steady_s=20.01)
        regression = fence.observe(current_stamp_ns=29_900, received_steady_s=20.02)
        resumed = fence.observe(current_stamp_ns=30_020, received_steady_s=20.03)

        self.assertEqual(initial["action"], "ACCEPT_NEW")
        self.assertEqual(newer["action"], "ACCEPT_NEW")
        self.assertEqual(regression["action"], "DROP_OUT_OF_ORDER")
        self.assertEqual(regression["source_stamp_relation"], "REGRESSION")
        self.assertEqual(regression["last_accepted_stamp_ns"], 30_010)
        self.assertAlmostEqual(float(regression["last_accepted_received_steady_s"]), 20.01)
        self.assertEqual(fence.dropped_regression_count, 1)
        self.assertEqual(resumed["action"], "ACCEPT_NEW")
        self.assertEqual(fence.last_accepted_stamp_ns, 30_020)

    def test_odom_input_fence_rejects_future_timestamp_leap_before_it_poisoning_anchor(self):
        fence = OdomSourceStampFence(maximum_forward_source_lead_s=0.20)
        self.assertEqual(
            fence.observe(
                current_stamp_ns=1_000_000_000,
                received_steady_s=10.0,
                received_wall_time_ns=1_000_000_000,
            )["action"],
            "ACCEPT_NEW",
        )
        future = fence.observe(
            current_stamp_ns=1_260_000_000,
            received_steady_s=10.01,
            received_wall_time_ns=1_010_000_000,
        )
        resumed = fence.observe(
            current_stamp_ns=1_012_000_000,
            received_steady_s=10.012,
            received_wall_time_ns=1_012_000_000,
        )

        self.assertEqual(future["action"], "DROP_FUTURE_LEAP")
        self.assertEqual(future["source_stamp_relation"], "FUTURE_WALL_CLOCK")
        self.assertAlmostEqual(float(future["source_wall_lead_s"]), 0.25)
        self.assertEqual(fence.last_accepted_stamp_ns, 1_012_000_000)
        self.assertEqual(resumed["action"], "ACCEPT_NEW")

    def test_odom_input_fence_uses_monotonic_continuity_after_host_wall_clock_rebase(self):
        fence = OdomSourceStampFence(maximum_forward_source_lead_s=0.20)
        self.assertEqual(
            fence.observe(
                current_stamp_ns=1_000_000_000,
                received_steady_s=10.0,
                received_wall_time_ns=1_000_000_000,
            )["action"],
            "ACCEPT_NEW",
        )
        rebased = fence.observe(
            current_stamp_ns=1_002_000_000,
            received_steady_s=10.002,
            received_wall_time_ns=400_000_000,
        )
        self.assertEqual(rebased["action"], "ACCEPT_NEW")
        self.assertTrue(rebased["wall_clock_rebased"])
        self.assertFalse(rebased["wall_clock_alignment_reliable"])
        continued = fence.observe(
            current_stamp_ns=1_004_000_000,
            received_steady_s=10.004,
            received_wall_time_ns=402_000_000,
        )
        self.assertEqual(continued["action"], "ACCEPT_NEW")
        self.assertEqual(fence.wall_clock_rebase_count, 1)

    def test_native_profile_can_reanchor_a_bounded_source_clock_rebase_only(self):
        fence = OdomSourceStampFence(
            allow_bounded_clock_rebase=True,
            maximum_clock_rebase_s=0.75,
        )
        fence.observe(current_stamp_ns=2_000_000_000, received_steady_s=10.0)
        accepted = fence.observe(
            current_stamp_ns=1_600_000_000, received_steady_s=10.01
        )
        self.assertEqual(accepted["action"], "ACCEPT_BOUNDED_CLOCK_REBASE")
        self.assertEqual(accepted["source_stamp_relation"], "BOUNDED_CLOCK_REBASE")
        self.assertEqual(fence.last_accepted_stamp_ns, 1_600_000_000)
        rejected = fence.observe(
            current_stamp_ns=200_000_000, received_steady_s=10.02
        )
        self.assertEqual(rejected["action"], "DROP_OUT_OF_ORDER")

    def test_native_profile_can_accept_forward_sim_clock_leap_but_default_cannot(self):
        generic = OdomSourceStampFence(maximum_forward_source_lead_s=0.20)
        generic.observe(current_stamp_ns=1_000_000_000, received_steady_s=10.0)
        self.assertEqual(
            generic.observe(
                current_stamp_ns=2_100_000_000, received_steady_s=10.002
            )["action"],
            "DROP_FUTURE_LEAP",
        )
        native = OdomSourceStampFence(
            maximum_forward_source_lead_s=0.20,
            allow_forward_sim_clock_leaps=True,
        )
        native.observe(current_stamp_ns=1_000_000_000, received_steady_s=10.0)
        accepted = native.observe(
            current_stamp_ns=2_100_000_000, received_steady_s=10.002
        )
        self.assertEqual(accepted["action"], "ACCEPT_FORWARD_SIM_CLOCK_LEAP")
        self.assertEqual(native.last_accepted_stamp_ns, 2_100_000_000)

    def test_actual_stale_odometry_delivery_is_rejected_even_with_advancing_stamp(self):
        tracker = RouteOdomFreshnessTracker(40_000, 30.0, maximum_age_s=0.20)
        stale = tracker.observe(
            current_stamp_ns=40_001, sample_steady_s=30.25, received_steady_s=30.049,
        )
        self.assertEqual(stale["action"], "STALE_OR_INVALID")
        self.assertEqual(stale["source_stamp_relation"], "NEW")
        self.assertEqual(tracker.last_unique_stamp_ns, 40_000)

    def test_duplicate_odom_does_not_skip_the_50mm_cross_track_guard(self):
        self.assertEqual(
            route_feedback_sample_disposition(
                odom_action="DUPLICATE",
                cross_track_error_m=0.050001,
                maximum_cross_track_m=0.05,
            ),
            "ABORT_CROSS_TRACK_GUARD",
        )
        self.assertEqual(
            route_feedback_sample_disposition(
                odom_action="DUPLICATE",
                cross_track_error_m=0.05,
                maximum_cross_track_m=0.05,
            ),
            "SKIP_DUPLICATE_ROUTE_TICK",
        )

    def test_duplicate_odometry_during_shutdown_cannot_pass_route_stop_freshness_gate(self):
        samples = []
        for index in range(61):
            sample_time = 3.0 + 0.05 * index
            samples.append({
                "sample_monotonic_s": sample_time,
                "phase": "post-stop",
                "pose": {"stamp_sec": 100, "stamp_nanosec": 0},
                "state_age_s": {"odom": 0.001},
            })
        gate = evaluate_route_odom_freshness_window(
            samples,
            window_start_steady_s=3.0,
            window_end_steady_s=6.0,
            maximum_age_s=0.20,
            minimum_window_s=1.90,
        )
        self.assertEqual(gate["status"], "FAIL")
        self.assertIn("NONUNIQUE_OR_STALE_ODOMETRY_SOURCE", gate["reason"])
        self.assertEqual(gate["unique_source_stamp_count"], 1)
        self.assertGreater(gate["maximum_age_since_last_unique_s"], 0.20)
        rearm = evaluate_route_window_rearm_gate(
            physical_stop_status="PASS",
            odometry_freshness_status=gate["status"],
        )
        self.assertFalse(rearm["rearm_allowed"])
        self.assertIn("UNIQUE_ODOMETRY_FRESHNESS_GATE_NOT_PASS", rearm["blockers"])

    def test_route_window_rearm_requires_composite_stop_and_unique_odometry(self):
        self.assertFalse(evaluate_route_window_rearm_gate(
            physical_stop_status="PASS", odometry_freshness_status="FAIL",
        )["rearm_allowed"])
        self.assertFalse(evaluate_route_window_rearm_gate(
            physical_stop_status="FAIL", odometry_freshness_status="PASS",
        )["rearm_allowed"])
        self.assertTrue(evaluate_route_window_rearm_gate(
            physical_stop_status="PASS", odometry_freshness_status="PASS",
        )["rearm_allowed"])

    def test_advancing_odometry_through_a_complete_stop_window_passes_unique_freshness(self):
        samples = []
        for index in range(61):
            sample_time = 3.0 + 0.05 * index
            samples.append({
                "sample_monotonic_s": sample_time,
                "phase": "post-stop",
                "pose": {"stamp_sec": 100 + index, "stamp_nanosec": 0},
                "state_age_s": {"odom": 0.001},
            })
        gate = evaluate_route_odom_freshness_window(
            samples,
            window_start_steady_s=3.0,
            window_end_steady_s=6.0,
            maximum_age_s=0.20,
            minimum_window_s=1.90,
        )
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["unique_source_stamp_count"], 61)

    def test_two_window_curved_route_rearms_only_after_both_stop_and_odometry_gates(self):
        start = make_pose(stamp=1_000_000_000, received=10.0)
        reference = build_shallow_curved_path_reference(start)
        route = build_curved_waypoint_route(
            route_id="two-window-curved-route",
            start_pose=start,
            reference=reference,
            robot_footprint_radius_m=0.25,
            local_clear_radius_m=2.0,
        )
        step = route.steps[0]
        directive = follow_directive(
            route_id=route.route_id,
            segment_id=step.waypoint_id,
            waypoint_id=step.waypoint_id,
            goal_x_m=step.x_m,
            goal_y_m=step.y_m,
            position_tolerance_m=step.position_tolerance_m,
            target_forward_m_s=step.target_forward_m_s,
            max_forward_m_s=step.max_forward_m_s,
            max_angular_rad_s=step.max_angular_rad_s,
            max_command_window_s=step.max_command_window_s,
        )
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=1.0)
        first = bridge.plan_next(directive, start, readiness(), now_steady_s=10.0)
        self.assertEqual(first.status, "PROPOSAL_ONLY")
        bridge.complete_window(first.window_id, actual_elapsed_s=1.0)
        zero = bridge.zero_window(directive, now_steady_s=11.0, reason="window-one stop")
        self.assertTrue(bridge.confirm_physical_stop(
            {"status": "PASS", "window_start_steady_s": 11.0,
             "window_end_steady_s": 12.95, "sample_count": 40,
             "maximum_sample_gap_s": 0.05, "maximum_state_age_s": 0.02,
             "generation": start.generation},
            zero_request_started_s=11.0,
            now_steady_s=13.0,
            expected_generation=start.generation,
        ))
        theta = 0.11
        origin = reference["start_pose"]
        resumed_pose = make_pose(
            x=float(origin["x_m"]) + CURVED_PATH_RADIUS_M * math.sin(theta),
            y=float(origin["y_m"]) - CURVED_PATH_RADIUS_M * (1.0 - math.cos(theta)),
            yaw=start.yaw_rad - theta,
            stamp=2_000_000_000,
            received=13.0,
        )
        second = bridge.plan_next(
            directive, resumed_pose, readiness(observed_steady_s=13.0), now_steady_s=13.0,
        )
        self.assertEqual(second.status, "PROPOSAL_ONLY")
        self.assertEqual(second.segment_id, first.segment_id)
        self.assertEqual(second.window_id, zero.window_id + 1)

    def test_curved_path_cli_profile_accepts_only_the_bounded_reviewed_envelope(self):
        values = {
            "route_integration": True,
            "curved_path": True,
            "waypoint_b_mode": False,
            "turn_evidence_present": False,
            "extra_stop_observation_s": 5.0,
            "pulse_duration_s": 1.0,
            "generic_forward_speed_m_s": 0.2,
            "ramp_duration_s": 0.0,
            "ramp_start_speed_m_s": None,
            "deceleration_duration_s": 0.0,
            "startup_speed_m_s": None,
            "startup_transition_displacement_m": None,
            "startup_transition_timeout_s": 0.0,
            "max_observed_planar_speed_m_s": 0.2,
            "zero_until_stand_timeout_s": 3.5,
            "zero_only_stop": True,
        }
        profile = validate_curved_path_cli_profile(**values)
        self.assertEqual(profile["profile_classification"], "SIMULATION_ONLY_CURVED_PATH_DIAGNOSTIC")
        self.assertEqual(profile["geometry"]["signed_sweep_rad"], -0.22)
        self.assertEqual(profile["maximum_windows"], 3)
        self.assertEqual(profile["aggregate_nonzero_limit_s"], 15.0)

        for mutation in (
            {"route_integration": False},
            {"waypoint_b_mode": True},
            {"turn_evidence_present": True},
            {"extra_stop_observation_s": 4.0},
            {"zero_only_stop": False},
            {"max_observed_planar_speed_m_s": 0.3},
        ):
            invalid = dict(values)
            invalid.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(RouteMotionBridgeError):
                validate_curved_path_cli_profile(**invalid)

    def test_shallow_curve_is_frozen_in_measured_odom_and_within_motion_envelope(self):
        start = make_pose(x=1.2, y=-0.4, yaw=0.31, received=10.0)
        reference = build_shallow_curved_path_reference(start)

        self.assertEqual(reference["frame_id"], "odom")
        self.assertEqual(reference["direction"], "RIGHT")
        self.assertAlmostEqual(reference["radius_m"], CURVED_PATH_RADIUS_M)
        self.assertAlmostEqual(reference["sweep_rad"], CURVED_PATH_SWEEP_RAD)
        self.assertAlmostEqual(reference["expected_path_length_m"], CURVED_PATH_RADIUS_M * CURVED_PATH_SWEEP_RAD, places=9)
        self.assertAlmostEqual(
            reference["nominal_angular_rate_rad_s"],
            -CURVED_PATH_FORWARD_M_S / CURVED_PATH_RADIUS_M,
            places=9,
        )
        self.assertLess(reference["nominal_angular_rate_rad_s"], reference["angular_command_limit_rad_s"])
        self.assertAlmostEqual(reference["position_tolerance_m"], CURVED_PATH_POSITION_TOLERANCE_M)
        self.assertAlmostEqual(reference["stop_reserve_m"], CURVED_PATH_STOP_RESERVE_M)
        self.assertAlmostEqual(reference["maximum_cross_track_m"], CURVED_PATH_MAX_CROSS_TRACK_M)
        points = reference["reference_points"]
        self.assertEqual(len(points), 21)
        self.assertAlmostEqual(points[0]["x_m"], start.x_m)
        self.assertAlmostEqual(points[0]["y_m"], start.y_m)
        self.assertAlmostEqual(points[-1]["x_m"], reference["target_x_m"])
        self.assertAlmostEqual(points[-1]["y_m"], reference["target_y_m"])

    def test_curve_projection_uses_curve_not_start_centerline(self):
        start = make_pose(x=-0.2, y=0.7, yaw=-0.4, received=10.0)
        reference = build_shallow_curved_path_reference(start)
        point = reference["reference_points"][10]
        on_curve = curved_path_metrics(reference, x_m=point["x_m"], y_m=point["y_m"])
        self.assertLess(on_curve["cross_track_error_m"], 1e-9)
        self.assertGreater(on_curve["path_progress_m"], 0.0)
        self.assertLess(on_curve["path_progress_m"], reference["expected_path_length_m"])
        self.assertLess(on_curve["path_tangent_yaw_rad"], start.yaw_rad)

        tangent = on_curve["path_tangent_yaw_rad"]
        offset = 0.051
        displaced = curved_path_metrics(
            reference,
            x_m=point["x_m"] - offset * math.sin(tangent),
            y_m=point["y_m"] + offset * math.cos(tangent),
        )
        self.assertGreater(displaced["cross_track_error_m"], CURVED_PATH_MAX_CROSS_TRACK_M)

    def test_curved_route_has_single_verified_stop_waypoint_and_bounded_windows(self):
        start = make_pose(x=0.5, y=0.1, yaw=0.2, received=10.0)
        reference = build_shallow_curved_path_reference(start)
        route = build_curved_waypoint_route(
            route_id="curved-test",
            start_pose=start,
            reference=reference,
            robot_footprint_radius_m=0.25,
            local_clear_radius_m=2.0,
        )
        self.assertEqual(route.frame_id, "odom")
        self.assertEqual(len(route.steps), 1)
        step = route.steps[0]
        self.assertEqual(step.waypoint_id, "CURVE-ENDPOINT")
        self.assertAlmostEqual(step.x_m, reference["target_x_m"])
        self.assertAlmostEqual(step.y_m, reference["target_y_m"])
        self.assertEqual(step.position_tolerance_m, CURVED_PATH_POSITION_TOLERANCE_M)
        self.assertEqual(step.target_forward_m_s, CURVED_PATH_FORWARD_M_S)
        self.assertEqual(step.max_forward_m_s, CURVED_PATH_FORWARD_M_S)
        self.assertEqual(step.max_angular_rad_s, 0.075)
        self.assertEqual(step.max_command_window_s, CURVED_PATH_MAX_WINDOW_S)
        self.assertEqual(step.stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertEqual(CURVED_PATH_MAX_WINDOWS, 3)
        self.assertEqual(CURVED_PATH_AGGREGATE_NONZERO_MAX_S, 15.0)

        with self.assertRaisesRegex(RouteMotionBridgeError, "clearance"):
            build_curved_waypoint_route(
                route_id="curved-test-too-tight",
                start_pose=start,
                reference=reference,
                robot_footprint_radius_m=0.25,
                local_clear_radius_m=1.0,
            )

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

    def test_poststop_stand_and_source_readbacks_must_be_fresh_and_consistent(self):
        stand = {
            "sample_monotonic_s": 20.0,
            "readback_monotonic_s": {"get_mode": 19.8, "get_source": 19.8},
            "state_age_s": {"common": 0.02},
            "get_mode_header_code": 0,
            "get_mode_status_value": 100,
            "get_mode_desc": "STAND_DEFAULT",
            "common_mode_status_value": 100,
            "common_mode_desc": "STAND_DEFAULT",
            "fsm_state": 4,
            "body_status": 1,
            "get_mode_readback_fault": None,
            "readback_fault": None,
        }
        self.assertTrue(fresh_stand_mode_readback_valid(stand, maximum_age_s=0.5))
        stale_stand = dict(stand)
        stale_stand["readback_monotonic_s"] = {"get_mode": 19.4, "get_source": 19.8}
        self.assertFalse(fresh_stand_mode_readback_valid(stale_stand, maximum_age_s=0.5))
        unsafe_stand = dict(stand)
        unsafe_stand["fsm_state"] = 2
        self.assertFalse(fresh_stand_mode_readback_valid(unsafe_stand, maximum_age_s=0.5))

        source = {
            "sample_monotonic_s": 20.0,
            "readback_monotonic_s": {"get_mode": 19.8, "get_source": 19.79},
            "state_age_s": {"common": 0.02},
            "get_source_header_code": 0,
            "get_source_name": VELOCITY_SOURCE_NAME,
            "common_source_name": VELOCITY_SOURCE_NAME,
            "get_source_readback_fault": None,
            "readback_fault": None,
        }
        self.assertTrue(fresh_input_source_readback_valid(
            source, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
        ))
        stale_source = dict(source)
        stale_source["readback_monotonic_s"] = {"get_mode": 19.8, "get_source": 19.4}
        self.assertFalse(fresh_input_source_readback_valid(
            stale_source, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
        ))
        conflict_source = dict(source)
        conflict_source["common_source_name"] = "another.source"
        self.assertFalse(fresh_input_source_readback_valid(
            conflict_source, expected_source=VELOCITY_SOURCE_NAME, maximum_age_s=0.5,
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
        # The live TABLE_APPROACH mission defines a five-second *per-window*
        # cap.  A verified stop must rearm a new cap without erasing the
        # runtime-owned aggregate route budget.
        directive = replace(directive, max_command_window_s=5.0)
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
        next_directive = replace(next_directive, max_command_window_s=5.0)

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

    def test_failed_physical_stop_keeps_the_next_five_second_window_blocked(self) -> None:
        """A per-window budget resets only after actual stop evidence passes."""
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=5.0)
        directive = follow_directive(max_command_window_s=5.0)
        first = bridge.plan_next(directive, make_pose(), readiness(), now_steady_s=10.0)
        bridge.complete_window(first.window_id, actual_elapsed_s=4.98)
        bridge.zero_window(
            directive,
            now_steady_s=15.0,
            reason="completed bounded transit window requires measured stop",
        )
        self.assertFalse(bridge.confirm_physical_stop(
            {
                "status": "FAIL",
                "window_start_steady_s": 15.01,
                "window_end_steady_s": 15.50,
                "sample_count": 12,
                "maximum_sample_gap_s": 0.08,
                "maximum_state_age_s": 0.2,
                "generation": "episode-1",
            },
            zero_request_started_s=15.0,
            now_steady_s=15.50,
            expected_generation="episode-1",
        ))
        blocked = bridge.plan_next(
            directive,
            make_pose(stamp=2_000_000_000, received=15.50),
            readiness(observed_steady_s=15.50),
            now_steady_s=15.50,
        )
        self.assertEqual(blocked.status, "BLOCKED_ZERO_ONLY")
        self.assertIn("physical-stop evidence passes", blocked.reason)

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
        self.assertEqual(RouteMotionBridge(route_frame_id="odom", incremental_window_s=10.0).incremental_window_s, 10.0)
        with self.assertRaises(ValueError):
            RouteMotionBridge(route_frame_id="odom", incremental_window_s=10.001)

    def test_explicit_ten_second_bridge_window_remains_under_the_m1_ceiling(self) -> None:
        bridge = RouteMotionBridge(route_frame_id="odom", incremental_window_s=10.0)
        plan = bridge.plan_next(
            follow_directive(max_command_window_s=10.0),
            make_pose(),
            readiness(),
            now_steady_s=10.0,
        )
        self.assertEqual(plan.status, "PROPOSAL_ONLY")
        self.assertEqual(plan.maximum_duration_s, 10.0)

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
        low_cruise = RouteMotionBridge(route_frame_id="odom").plan_next(
            follow_directive(target_forward_m_s=0.105, max_forward_m_s=0.105),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(low_cruise.status, "PROPOSAL_ONLY")
        self.assertAlmostEqual(low_cruise.forward_m_s, 0.105)
        below_validated_low_cruise = RouteMotionBridge(route_frame_id="odom").plan_next(
            follow_directive(target_forward_m_s=0.104, max_forward_m_s=0.104),
            make_pose(), readiness(), now_steady_s=10.0,
        )
        self.assertEqual(below_validated_low_cruise.status, "BLOCKED_ZERO_ONLY")
        low_speed_corner = RouteMotionBridge(route_frame_id="odom").plan_next(
            turn_directive(target_forward_m_s=0.105, max_forward_m_s=0.105),
            make_pose(), readiness(
                bounded_moving_turn_qualified=True,
                qualified_turn_directions=("RIGHT",),
                turn_stop_margin_m=0.005,
                turn_stop_margin_direction="RIGHT",
                turn_stop_margin_evidence_id="qualified-right-turn",
                measured_displacement_guard_verified=True,
            ), now_steady_s=10.0,
        )
        self.assertEqual(low_speed_corner.status, "BLOCKED_ZERO_ONLY")
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
