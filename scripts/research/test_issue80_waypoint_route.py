#!/usr/bin/env python3
"""Offline route-manager tests using synthetic measured-pose traces only."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from issue80_waypoint_route import (
    DirectiveKind,
    DockingReadiness,
    MotionFeedback,
    MotionStatus,
    PROPOSED_PROFILE_POLICIES,
    PhysicalStopEvidence,
    PhysicalStopStatus,
    PoseEstimate,
    Profile,
    RouteSpec,
    RouteState,
    SafetyObservation,
    WaypointRouteExecutor,
    YawPolicy,
)
from issue80_route_motion_bridge import (
    build_curved_waypoint_route,
    build_shallow_curved_path_reference,
)


ROOT = Path(__file__).parent
RECTANGLE = ROOT / "issue80_rectangle_return_route.template.json"


class FakePoseProvider:
    def __init__(self, pose: PoseEstimate):
        self.pose = pose

    def latest_pose(self, target_frame: str, now_steady_s: float) -> PoseEstimate:
        return self.pose

    def update(self, *, x: float, y: float, yaw: float, now: float, stamp: int) -> None:
        self.pose = PoseEstimate(
            frame_id="map", child_frame_id="base_link", x_m=x, y_m=y, yaw_rad=yaw,
            stamp_ns=stamp, received_steady_s=now, generation="epoch-1",
            source="LOCALIZATION", position_std_m=0.01, yaw_std_rad=0.02,
            transform_chain=("map->odom", "odom->base_link"), transform_age_s=0.05,
        )


class FakeSafetyProvider:
    def __init__(self):
        self.blocked = False
        self.robot_safe = True
        self.clearance_m = 2.0
        self.stale_by_s = 0.0

    def latest_observation(self, segment_id: str, frame_id: str, now_steady_s: float) -> SafetyObservation:
        return SafetyObservation(
            segment_id=segment_id, frame_id=frame_id, received_steady_s=now_steady_s - self.stale_by_s,
            clear=not self.blocked, robot_safe=self.robot_safe,
            minimum_obstacle_clearance_m=self.clearance_m,
            reason="synthetic obstacle marker" if self.blocked else "synthetic map clear",
        )


class FakeMotionAdapter:
    def __init__(self):
        self.directives = []
        self.stop_reasons = []
        self.zero_start_times = []

    def dispatch(self, directive) -> None:
        self.directives.append(directive)

    def request_zero_and_stop(self, reason: str, requested_steady_s: float) -> float:
        self.stop_reasons.append(reason)
        self.zero_start_times.append(requested_steady_s)
        return requested_steady_s


def startup_feedback(**overrides) -> MotionFeedback:
    values = dict(
        status=MotionStatus.ACTIVE,
        reset_gate_pass=True,
        standing_gate_pass=True,
        source_mode_handshake_pass=True,
        safety_gates_pass=True,
    )
    values.update(overrides)
    return MotionFeedback(**values)


def make_pose_provider(
    x=0.0, y=0.0, yaw=0.0, now=0.0, stamp=1_000_000_000,
    wall_time_ns=None, source="LOCALIZATION",
) -> FakePoseProvider:
    pose = PoseEstimate(
        frame_id="map", child_frame_id="base_link", x_m=x, y_m=y, yaw_rad=yaw,
        stamp_ns=stamp, received_steady_s=now, generation="epoch-1",
        source=source, position_std_m=0.01, yaw_std_rad=0.02,
        transform_chain=("map->odom", "odom->base_link"), transform_age_s=0.05,
        wall_time_ns=wall_time_ns,
    )
    return FakePoseProvider(pose)


def stop_evidence(start: float, generation: str = "epoch-1") -> PhysicalStopEvidence:
    return PhysicalStopEvidence(
        status=PhysicalStopStatus.PASS,
        window_start_steady_s=start,
        window_end_steady_s=start + 1.9,
        sample_count=40,
        maximum_sample_gap_s=0.08,
        maximum_state_age_s=0.2,
        generation=generation,
    )


def verified_rectangle_route() -> RouteSpec:
    raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
    raw["workspace_clearance_verified"] = True  # synthetic test fixture only
    raw["turn_capabilities"]["bounded_moving_turn_qualified"] = True  # synthetic backend only
    raw["route_id"] = "synthetic-rectangle-trace"
    return RouteSpec.from_mapping(raw)


def one_docking_route() -> RouteSpec:
    corridor = [[-2.0, -2.0], [4.0, -2.0], [4.0, 3.0], [-2.0, 3.0]]
    return RouteSpec.from_mapping({
        "schema_version": "robotsim.issue80.waypoint-route.v1",
        "route_id": "synthetic-docking-handoff",
        "frame_id": "map",
        "workspace_clearance_verified": True,
        "turn_capabilities": {"zero_forward_turn_qualified": False, "bounded_moving_turn_qualified": False,
                              "evidence_classification": "NO_CORNER_IN_DOCKING_FIXTURE", "evidence_attempt_id": "test"},
        "start_anchor": {"x_m": 0, "y_m": 0, "yaw_rad": 0,
                         "position_tolerance_m": 0.25, "yaw_tolerance_rad": 0.25},
        "localization": {"max_pose_age_s": 0.2, "max_transform_age_s": 0.5,
                         "max_position_std_m": 0.1, "max_yaw_std_rad": 0.2},
        "safety": {"robot_footprint_radius_m": 0.2, "boundary_margin_m": 0.1,
                   "max_observation_age_s": 0.2},
        "steps": [{
            "id": "PICKUP", "profile": "DOCKING",
            "pose": {"x_m": 1, "y_m": 0, "yaw_rad": 0.2},
            "acceptance": {"position_tolerance_m": 0.08, "yaw_policy": "EXPLICIT",
                           "yaw_tolerance_rad": 0.1, "max_position_std_m": 0.025,
                           "max_yaw_std_rad": 0.05},
            "stop_behavior": "VERIFY_PHYSICAL_STOP", "segment_timeout_s": 60,
            "recovery_limit": 0,
            "corridor": {"polygon": corridor, "minimum_obstacle_clearance_m": 0.3},
            "command_bounds": {"target_forward_m_s": 0.12, "max_forward_m_s": 0.12, "max_angular_rad_s": 0.075,
                               "max_command_window_s": 10}
        }]
    })


class RouteConfigTests(unittest.TestCase):
    def test_three_profile_accuracy_and_motion_envelopes_are_separate(self) -> None:
        transit = PROPOSED_PROFILE_POLICIES[Profile.TRANSIT]
        corner = PROPOSED_PROFILE_POLICIES[Profile.CORNER]
        docking = PROPOSED_PROFILE_POLICIES[Profile.DOCKING]
        self.assertEqual(transit.position_tolerance_m, 0.20)
        self.assertEqual(transit.yaw_policy, "IGNORE")
        self.assertEqual(transit.target_forward_m_s, 0.14)
        self.assertEqual(corner.position_tolerance_m, 0.25)
        self.assertEqual(corner.yaw_policy, "NEXT_SEGMENT")
        self.assertEqual(corner.target_forward_m_s, 0.12)
        self.assertEqual(corner.max_angular_rad_s, 0.075)
        self.assertTrue(corner.verify_physical_stop)
        self.assertEqual(docking.position_tolerance_m, 0.08)
        self.assertEqual(docking.yaw_tolerance_rad, 0.10)
        self.assertEqual(docking.max_corrections, 2)

    def test_configured_target_speed_cannot_fall_below_measured_deadzone(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["steps"][0]["command_bounds"]["target_forward_m_s"] = 0.10
        with self.assertRaisesRegex(ValueError, "below the measured effective command deadzone"):
            RouteSpec.from_mapping(raw)

    def test_rectangle_route_is_template_only_and_keeps_stage_b_gates_separate(self) -> None:
        route = RouteSpec.load(RECTANGLE)
        self.assertFalse(route.workspace_verified)
        self.assertEqual([step.waypoint_id for step in route.steps], ["B", "C", "D", "A_RETURN"])
        self.assertEqual([step.profile for step in route.steps], [Profile.CORNER] * 3 + [Profile.TRANSIT])
        self.assertEqual(route.steps[-1].position_tolerance_m, 0.25)
        self.assertEqual(route.steps[-1].yaw_rad, 0.0)

    def test_limits_reject_speed_angular_window_or_recovery_expansion(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["steps"][0]["command_bounds"]["max_forward_m_s"] = 0.15
        with self.assertRaisesRegex(ValueError, "0.14 m/s"):
            RouteSpec.from_mapping(raw)
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["steps"][0]["command_bounds"]["max_angular_rad_s"] = 0.08
        with self.assertRaisesRegex(ValueError, "0.075 rad/s"):
            RouteSpec.from_mapping(raw)

    def test_workspace_unverified_template_cannot_start_route(self) -> None:
        route = RouteSpec.load(RECTANGLE)
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        directive = runner.start(0.0, startup_feedback())
        self.assertEqual(runner.state, RouteState.ABORTED)
        self.assertEqual(directive.reason, "WORKSPACE_CLEARANCE_NOT_VERIFIED")
        self.assertFalse(any(item.kind is DirectiveKind.FOLLOW_WAYPOINT for item in motion.directives))

    def test_corner_route_refuses_unqualified_turn_capability(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["workspace_clearance_verified"] = True
        route = RouteSpec.from_mapping(raw)
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, make_pose_provider(), FakeSafetyProvider(), motion)
        directive = runner.start(0.0, startup_feedback())
        self.assertEqual(runner.state, RouteState.ABORTED)
        self.assertEqual(directive.reason, "BOUNDED_MOVING_TURN_CAPABILITY_NOT_QUALIFIED")

    def test_reset_standing_source_mode_and_safety_gates_are_required_before_route(self) -> None:
        route = verified_rectangle_route()
        cases = (
            {"reset_gate_pass": False},
            {"standing_gate_pass": False},
            {"source_mode_handshake_pass": False},
            {"safety_gates_pass": False},
        )
        for override in cases:
            motion = FakeMotionAdapter()
            runner = WaypointRouteExecutor(route, make_pose_provider(), FakeSafetyProvider(), motion)
            directive = runner.start(0.0, startup_feedback(**override))
            self.assertEqual(runner.state, RouteState.ABORTED)
            self.assertEqual(directive.reason, "RESET_STANDING_SOURCE_MODE_OR_SAFETY_GATE_NOT_READY")
            self.assertFalse(any(d.kind is DirectiveKind.FOLLOW_WAYPOINT for d in motion.directives))

    def test_first_route_action_allows_only_explicitly_authorized_activation_handshake(self) -> None:
        route = verified_rectangle_route()
        runner = WaypointRouteExecutor(
            route, make_pose_provider(), FakeSafetyProvider(), FakeMotionAdapter()
        )
        directive = runner.start(0.0, startup_feedback(
            source_mode_handshake_pass=False,
            initial_activation_authorized=True,
        ))
        self.assertEqual(runner.state, RouteState.RUNNING)
        self.assertEqual(directive.kind, DirectiveKind.FOLLOW_WAYPOINT)
        next_directive = runner.tick(0.1, startup_feedback(
            source_mode_handshake_pass=False,
            initial_activation_authorized=True,
        ))
        self.assertEqual(next_directive.kind, DirectiveKind.STOP_AND_ABORT)
        self.assertEqual(next_directive.reason, "SAFETY_OR_SOURCE_MODE_HANDSHAKE_LOST")


class RouteExecutionTests(unittest.TestCase):
    def test_shallow_curved_route_corridor_contains_endpoint_with_full_footprint_margin(self) -> None:
        start = PoseEstimate(
            frame_id="odom", child_frame_id="base_link", x_m=0.2, y_m=-0.1, yaw_rad=0.25,
            stamp_ns=1_000_000_000, received_steady_s=0.0, generation="epoch-1",
            source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
            transform_chain=("odom->base_link",), transform_age_s=0.01,
        )
        reference = build_shallow_curved_path_reference(start)
        route = build_curved_waypoint_route(
            route_id="curved-route-corridor-test",
            start_pose=start,
            reference=reference,
            robot_footprint_radius_m=0.41565748847422734,
            local_clear_radius_m=2.0,
        )
        pose = FakePoseProvider(start)
        safety = FakeSafetyProvider()
        runner = WaypointRouteExecutor(route, pose, safety, FakeMotionAdapter())
        first = runner.start(0.0, startup_feedback())
        self.assertEqual(first.kind, DirectiveKind.FOLLOW_WAYPOINT)
        pose.pose = PoseEstimate(**{
            **start.__dict__,
            "x_m": float(reference["target_x_m"]),
            "y_m": float(reference["target_y_m"]),
            "stamp_ns": 2_000_000_000,
            "received_steady_s": 0.1,
        })
        observation = runner.tick(0.1, startup_feedback(status=MotionStatus.ACTIVE))
        self.assertEqual(observation.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(runner.state, RouteState.RUNNING)

    def test_arrival_after_completed_stop_advances_without_duplicate_stop(self) -> None:
        route = verified_rectangle_route()
        first = replace(
            route.steps[0], profile=Profile.TRANSIT, yaw_policy=YawPolicy.IGNORE,
            yaw_rad=None, yaw_tolerance_rad=None, turn_corridor_polygon=None,
            turn_strategy=None, maximum_turn_displacement_m=None,
        )
        route = replace(route, steps=(first, *route.steps[1:]))
        route.validate()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        first = runner.start(0.0, startup_feedback())
        self.assertEqual(first.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(first.waypoint_id, "B")

        # Arrival during the M1 window requests one physical stop.
        pose.update(x=5.0, y=0.0, yaw=0.0, now=1.0, stamp=2_000_000_000)
        requested = runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(requested.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(motion.stop_reasons, ["WAYPOINT_REACHED:B"])

        # The M1 adapter completes its stop gate, then a fresh settled pose
        # confirms the waypoint. The executor advances to the next route step
        # using that same evidence and does not ask for another stop window.
        pose.update(x=5.0, y=0.0, yaw=0.0, now=3.05, stamp=3_000_000_000)
        next_step = runner.accept_arrival_after_verified_stop(
            3.05,
            startup_feedback(
                status=MotionStatus.ARRIVED,
                physical_stop_evidence=stop_evidence(1.05),
                stand_readback_pass=True,
                input_source_readback_pass=True,
            ),
            zero_request_started_s=1.05,
            bridge_rearmed=True,
        )
        self.assertEqual(next_step.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertNotEqual(next_step.waypoint_id, "B")
        self.assertEqual(runner.active_step.waypoint_id, next_step.waypoint_id)
        self.assertEqual(runner.phase.value, "FOLLOW")
        self.assertEqual(motion.stop_reasons, ["WAYPOINT_REACHED:B"])

    def test_last_waypoint_completes_after_prior_stop_without_second_stop(self) -> None:
        source_route = verified_rectangle_route()
        first = replace(
            source_route.steps[0], profile=Profile.TRANSIT, yaw_policy=YawPolicy.IGNORE,
            yaw_rad=None, yaw_tolerance_rad=None, turn_corridor_polygon=None,
            turn_strategy=None, maximum_turn_displacement_m=None,
        )
        route = replace(source_route, steps=(first,))
        route.validate()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=5.0, y=0.0, yaw=0.0, now=1.0, stamp=2_000_000_000)
        runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5.0, y=0.0, yaw=0.0, now=3.0, stamp=3_000_000_000)

        complete = runner.accept_arrival_after_verified_stop(
            3.0,
            startup_feedback(
                status=MotionStatus.ARRIVED,
                physical_stop_evidence=stop_evidence(1.0),
                stand_readback_pass=True,
                input_source_readback_pass=True,
            ),
            zero_request_started_s=1.0,
            bridge_rearmed=True,
        )

        self.assertEqual(complete.kind, DirectiveKind.COMPLETE)
        self.assertEqual(runner.state, RouteState.COMPLETE)
        self.assertEqual(motion.stop_reasons, ["WAYPOINT_REACHED:B"])

    def test_poststop_only_arrival_completes_from_external_m1_gate_without_route_stop(self) -> None:
        source_route = verified_rectangle_route()
        first = replace(
            source_route.steps[0], profile=Profile.TRANSIT, yaw_policy=YawPolicy.IGNORE,
            yaw_rad=None, yaw_tolerance_rad=None, turn_corridor_polygon=None,
            turn_strategy=None, maximum_turn_displacement_m=None,
        )
        route = replace(source_route, steps=(first,))
        route.validate()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        # No route-level arrival/stop transition occurred during motion. The
        # M1 adapter publishes zero and passes its composite stop gate first.
        pose.update(x=4.9, y=0.0, yaw=0.0, now=2.5, stamp=2_000_000_000)
        complete = runner.accept_arrival_after_verified_stop(
            2.5,
            startup_feedback(
                status=MotionStatus.ARRIVED,
                physical_stop_evidence=stop_evidence(0.5),
                stand_readback_pass=True,
                input_source_readback_pass=True,
            ),
            zero_request_started_s=0.5,
            bridge_rearmed=True,
        )
        self.assertEqual(complete.kind, DirectiveKind.COMPLETE)
        self.assertEqual(runner.state, RouteState.COMPLETE)
        self.assertEqual(motion.stop_reasons, [])

    def test_poststop_arrival_rejects_outside_tolerance_stale_readbacks_or_failed_stop(self) -> None:
        cases = (
            ("outside_tolerance", {"x": 4.7, "stand": True, "source": True, "stop": PhysicalStopStatus.PASS, "bridge": True},
             "POSTSTOP_ARRIVAL_POSE_OUTSIDE_WAYPOINT_TOLERANCE"),
            ("stale_stand", {"x": 5.0, "stand": False, "source": True, "stop": PhysicalStopStatus.PASS, "bridge": True},
             "POSTSTOP_ARRIVAL_REQUIRES_FRESH_STAND_AND_SOURCE_READBACKS"),
            ("stale_source", {"x": 5.0, "stand": True, "source": False, "stop": PhysicalStopStatus.PASS, "bridge": True},
             "POSTSTOP_ARRIVAL_REQUIRES_FRESH_STAND_AND_SOURCE_READBACKS"),
            ("stop_failed", {"x": 5.0, "stand": True, "source": True, "stop": PhysicalStopStatus.FAIL, "bridge": True},
             "POSTSTOP_ARRIVAL_REQUIRES_COMPOSITE_PHYSICAL_STOP_PASS"),
            ("bridge_not_rearmed", {"x": 5.0, "stand": True, "source": True, "stop": PhysicalStopStatus.PASS, "bridge": False},
             "POSTSTOP_ARRIVAL_REQUIRES_STOP_GATED_BRIDGE_REARM"),
            ("reset_gate_missing", {"x": 5.0, "stand": True, "source": True, "stop": PhysicalStopStatus.PASS, "bridge": True, "reset": False},
             "POSTSTOP_ARRIVAL_REQUIRES_RESET_AND_STANDING_GATES"),
            ("standing_gate_missing", {"x": 5.0, "stand": True, "source": True, "stop": PhysicalStopStatus.PASS, "bridge": True, "standing": False},
             "POSTSTOP_ARRIVAL_REQUIRES_RESET_AND_STANDING_GATES"),
        )
        for label, case, expected in cases:
            with self.subTest(label=label):
                source_route = verified_rectangle_route()
                first = replace(
                    source_route.steps[0], profile=Profile.TRANSIT, yaw_policy=YawPolicy.IGNORE,
                    yaw_rad=None, yaw_tolerance_rad=None, turn_corridor_polygon=None,
                    turn_strategy=None, maximum_turn_displacement_m=None,
                )
                route = replace(source_route, steps=(first, *source_route.steps[1:]))
                route.validate()
                pose = make_pose_provider()
                motion = FakeMotionAdapter()
                runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
                runner.start(0.0, startup_feedback())
                pose.update(x=5.0, y=0.0, yaw=0.0, now=1.0, stamp=2_000_000_000)
                runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
                pose.update(x=case["x"], y=0.0, yaw=0.0, now=3.0, stamp=3_000_000_000)
                evidence = replace(stop_evidence(1.0), status=case["stop"])
                result = runner.accept_arrival_after_verified_stop(
                    3.0,
                    startup_feedback(
                        status=MotionStatus.ARRIVED,
                        physical_stop_evidence=evidence,
                        reset_gate_pass=case.get("reset", True),
                        standing_gate_pass=case.get("standing", True),
                        stand_readback_pass=case["stand"],
                        input_source_readback_pass=case["source"],
                    ),
                    zero_request_started_s=1.0,
                    bridge_rearmed=case["bridge"],
                )
                self.assertEqual(result.kind, DirectiveKind.STOP_AND_ABORT)
                self.assertEqual(runner.result["failure_reason"], expected)
                self.assertEqual(len(motion.stop_reasons), 2)  # original stop plus fail-closed abort
                self.assertNotEqual(runner.state, RouteState.COMPLETE)

    def test_wall_aligned_realtime_step_rebases_odom_stamp_and_records_it(self) -> None:
        initial_time_ns = 1_700_000_000_000_000_000
        rebased_time_ns = 1_699_999_999_470_000_000
        route = verified_rectangle_route()
        pose = make_pose_provider(
            now=10.0,
            stamp=initial_time_ns,
            wall_time_ns=initial_time_ns,
            source="MUJOCO_GROUND_TRUTH",
        )
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(10.0, startup_feedback())
        pose.pose = PoseEstimate(**{
            **pose.pose.__dict__,
            "x_m": 0.1,
            "received_steady_s": 10.1,
            "stamp_ns": rebased_time_ns,
            "wall_time_ns": rebased_time_ns,
        })

        directive = runner.tick(10.1, startup_feedback())

        self.assertEqual(directive.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(runner.verified_wall_clock_step_rebases, 1)
        self.assertEqual(runner.state, RouteState.RUNNING)
        self.assertFalse(motion.stop_reasons)

    def test_unaligned_or_unfresh_stamp_regression_still_fails_closed(self) -> None:
        initial_time_ns = 1_700_000_000_000_000_000
        rebased_time_ns = 1_699_999_999_470_000_000
        invalid_updates = (
            ("wall_time_continued", {
                "stamp_ns": rebased_time_ns,
                "wall_time_ns": initial_time_ns + 100_000_000,
                "received_steady_s": 10.1,
            }),
            ("header_wall_mismatch", {
                "stamp_ns": rebased_time_ns,
                "wall_time_ns": 1_699_999_999_100_000_000,
                "received_steady_s": 10.1,
            }),
            ("same_receive_time", {
                "stamp_ns": rebased_time_ns,
                "wall_time_ns": rebased_time_ns,
                "received_steady_s": 10.0,
            }),
            ("wrong_source", {
                "stamp_ns": rebased_time_ns,
                "wall_time_ns": rebased_time_ns,
                "received_steady_s": 10.1,
                "source": "LOCALIZATION",
            }),
        )
        for label, update in invalid_updates:
            with self.subTest(label=label):
                route = verified_rectangle_route()
                pose = make_pose_provider(
                    now=10.0,
                    stamp=initial_time_ns,
                    wall_time_ns=initial_time_ns,
                    source="MUJOCO_GROUND_TRUTH",
                )
                motion = FakeMotionAdapter()
                runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
                runner.start(10.0, startup_feedback())
                pose.pose = PoseEstimate(**{**pose.pose.__dict__, **update})

                directive = runner.tick(10.1, startup_feedback())

                self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT)
                self.assertEqual(runner.verified_wall_clock_step_rebases, 0)
                self.assertEqual(runner._last_error, "POSE_STAMP_NOT_STRICTLY_INCREASING")

    def test_synthetic_rectangle_advances_only_on_pose_and_physical_stop_evidence(self) -> None:
        route = verified_rectangle_route()
        pose = make_pose_provider()
        safety = FakeSafetyProvider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, safety, motion)
        first = runner.start(0.0, startup_feedback())
        self.assertEqual(first.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(first.waypoint_id, "B")

        # Arrive at B, then turn to the measured bearing of B->C.
        pose.update(x=5, y=0, yaw=0, now=1.0, stamp=2_000_000_000)
        turn = runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(turn.kind, DirectiveKind.BOUNDED_MOVING_TURN)
        self.assertAlmostEqual(turn.goal_yaw_rad, math.pi / 2)
        self.assertEqual(turn.max_forward_m_s, 0.12)
        self.assertEqual(turn.target_forward_m_s, 0.12)
        self.assertEqual(turn.max_command_window_s, 5.0)

        # Turn completion is based on measured yaw and must then pass physical stop.
        pose.update(x=5.05, y=0.05, yaw=math.pi / 2, now=2.0, stamp=3_000_000_000)
        stop = runner.tick(2.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        pose.update(x=5.05, y=0.05, yaw=math.pi / 2, now=4.0, stamp=4_000_000_000)
        follow_c = runner.tick(4.0, startup_feedback(physical_stop_evidence=stop_evidence(2.0)))
        self.assertEqual(follow_c.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(follow_c.waypoint_id, "C")

        # Synthetic path visits C, D, and the return region A with measured pose updates.
        now = 5.0
        stamp = 5_000_000_000
        current = ("C", 5.0, 5.0, math.pi / 2, math.pi)
        for waypoint_id, x, y, incoming_yaw, target_yaw in [
            current,
            ("D", 0.0, 5.0, math.pi, -math.pi / 2),
        ]:
            pose.update(x=x, y=y, yaw=incoming_yaw, now=now, stamp=stamp)
            got_turn = runner.tick(now, startup_feedback(status=MotionStatus.ARRIVED))
            self.assertEqual(got_turn.kind, DirectiveKind.BOUNDED_MOVING_TURN)
            now += 1.0; stamp += 1_000_000_000
            pose.update(x=x, y=y, yaw=target_yaw, now=now, stamp=stamp)
            got_stop = runner.tick(now, startup_feedback(status=MotionStatus.ARRIVED))
            self.assertEqual(got_stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
            now += 2.0; stamp += 1_000_000_000
            pose.update(x=x, y=y, yaw=target_yaw, now=now, stamp=stamp)
            next_intent = runner.tick(now, startup_feedback(physical_stop_evidence=stop_evidence(now - 2.0)))
            self.assertEqual(next_intent.kind, DirectiveKind.FOLLOW_WAYPOINT)
            self.assertNotEqual(next_intent.waypoint_id, waypoint_id)
            now += 1.0; stamp += 1_000_000_000

        pose.update(x=0, y=0, yaw=0, now=now, stamp=stamp)
        final_stop = runner.tick(now, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(final_stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        now += 2.0; stamp += 1_000_000_000
        pose.update(x=0, y=0, yaw=0, now=now, stamp=stamp)
        complete = runner.tick(now, startup_feedback(physical_stop_evidence=stop_evidence(now - 2.0)))
        self.assertEqual(complete.kind, DirectiveKind.COMPLETE)
        self.assertEqual(runner.state, RouteState.COMPLETE)
        self.assertEqual(runner.result["return_position_error_m"], 0.0)
        self.assertEqual(runner.result["return_yaw_error_rad"], 0.0)

    def test_missing_stale_or_reset_generation_pose_fails_closed(self) -> None:
        for mode in ("missing", "stale", "generation"):
            route = verified_rectangle_route()
            pose = make_pose_provider()
            motion = FakeMotionAdapter()
            runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
            runner.start(0.0, startup_feedback())
            if mode == "missing":
                pose.pose = None
            elif mode == "stale":
                pose.pose = PoseEstimate(**{**pose.pose.__dict__, "received_steady_s": 0.5})
            else:
                pose.pose = PoseEstimate(**{**pose.pose.__dict__, "generation": "epoch-2", "stamp_ns": 2_000_000_000, "received_steady_s": 0.1})
            directive = runner.tick(0.1, startup_feedback())
            self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT, mode)
            self.assertTrue(motion.stop_reasons)

    def test_pose_stamp_frame_transform_and_confidence_errors_fail_closed(self) -> None:
        mutations = {
            "repeated_stamp": {"stamp_ns": 1_000_000_000},
            "frame": {"frame_id": "odom"},
            "transform_chain": {"transform_chain": ("odom->base_link",)},
            "confidence": {"position_std_m": 0.5},
            "transform_age": {"transform_age_s": 1.5},
        }
        for label, mutation in mutations.items():
            route = verified_rectangle_route()
            pose = make_pose_provider()
            motion = FakeMotionAdapter()
            runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
            runner.start(0.0, startup_feedback())
            pose.pose = PoseEstimate(**{**pose.pose.__dict__, "received_steady_s": 0.1, **mutation})
            directive = runner.tick(0.1, startup_feedback())
            self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT, label)

    def test_stale_obstacle_or_physical_safety_observation_fails_closed(self) -> None:
        for label in ("stale_observation", "unsafe_robot"):
            route = verified_rectangle_route()
            pose = make_pose_provider()
            safety = FakeSafetyProvider()
            motion = FakeMotionAdapter()
            runner = WaypointRouteExecutor(route, pose, safety, motion)
            runner.start(0.0, startup_feedback())
            pose.update(x=0.1, y=0, yaw=0, now=0.1, stamp=2_000_000_000)
            if label == "stale_observation":
                safety.stale_by_s = 0.3
            else:
                safety.robot_safe = False
            directive = runner.tick(0.1, startup_feedback())
            self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT, label)

    def test_blocked_corridor_robot_safety_and_polygon_violation_stop(self) -> None:
        for mode in ("obstructed", "robot", "polygon"):
            route = verified_rectangle_route()
            pose = make_pose_provider()
            safety = FakeSafetyProvider()
            motion = FakeMotionAdapter()
            runner = WaypointRouteExecutor(route, pose, safety, motion)
            runner.start(0.0, startup_feedback())
            pose.update(x=0.1, y=0, yaw=0, now=0.1, stamp=2_000_000_000)
            if mode == "obstructed": safety.blocked = True
            if mode == "robot": safety.robot_safe = False
            if mode == "polygon": pose.update(x=10, y=10, yaw=0, now=0.1, stamp=2_000_000_000)
            directive = runner.tick(0.1, startup_feedback())
            self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT, mode)
            self.assertTrue(motion.stop_reasons)

    def test_command_success_without_physical_stop_does_not_advance(self) -> None:
        route = verified_rectangle_route()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=5, y=0, yaw=0, now=1.0, stamp=2_000_000_000)
        runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5, y=0, yaw=math.pi / 2, now=2.0, stamp=3_000_000_000)
        runner.tick(2.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5, y=0, yaw=math.pi / 2, now=3.0, stamp=4_000_000_000)
        directive = runner.tick(3.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(directive.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(runner.active_step.waypoint_id, "B")
        self.assertEqual(runner.phase.value, "STOP")

    def test_physical_stop_and_stand_readback_allow_route_rearm_without_active_source(self) -> None:
        route = one_docking_route()
        pose = make_pose_provider()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), FakeMotionAdapter())
        runner.start(0.0, startup_feedback())
        pose.update(x=1.0, y=0.0, yaw=0.2, now=1.0, stamp=2_000_000_000)
        directive = runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(directive.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        pose.update(x=1.0, y=0.0, yaw=0.2, now=3.1, stamp=3_000_000_000)
        wait = runner.tick(3.1, MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=stop_evidence(1.0),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertNotEqual(runner.state, RouteState.ABORTED)
        self.assertEqual(wait.kind, DirectiveKind.WAIT)
        self.assertEqual(runner.state, RouteState.WAITING_FOR_DOCKING)

    def test_stop_evidence_must_be_new_fresh_and_cover_the_gate_window(self) -> None:
        route = verified_rectangle_route()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=5, y=0, yaw=0, now=1.0, stamp=2_000_000_000)
        runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5, y=0, yaw=math.pi / 2, now=2.0, stamp=3_000_000_000)
        runner.tick(2.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5, y=0, yaw=math.pi / 2, now=4.0, stamp=4_000_000_000)
        stale = PhysicalStopEvidence(
            PhysicalStopStatus.PASS, 0.0, 1.9, 40, 0.08, 0.2, "epoch-1"
        )
        runner.tick(4.0, startup_feedback(physical_stop_evidence=stale))
        self.assertEqual(runner.active_step.waypoint_id, "B")
        fresh = stop_evidence(2.0)
        pose.update(x=5, y=0, yaw=math.pi / 2, now=4.1, stamp=5_000_000_000)
        next_step = runner.tick(4.1, startup_feedback(physical_stop_evidence=fresh))
        self.assertEqual(next_step.waypoint_id, "C")

    def test_segment_timeout_stops_using_steady_time(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["workspace_clearance_verified"] = True
        raw["turn_capabilities"]["bounded_moving_turn_qualified"] = True
        raw["steps"][0]["segment_timeout_s"] = 0.2
        route = RouteSpec.from_mapping(raw)
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=0.1, y=0, yaw=0, now=0.3, stamp=2_000_000_000)
        directive = runner.tick(0.3, startup_feedback())
        self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT)
        self.assertIn("SEGMENT_TIMEOUT", directive.reason)

    def test_corner_displacement_budget_latches_stop(self) -> None:
        route = verified_rectangle_route()
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=5, y=0, yaw=0, now=1.0, stamp=2_000_000_000)
        runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=5.8, y=0.0, yaw=0.2, now=1.1, stamp=3_000_000_000)
        directive = runner.tick(1.1, startup_feedback())
        self.assertEqual(directive.kind, DirectiveKind.STOP_AND_ABORT)
        self.assertIn("CORNER_DISPLACEMENT_BUDGET_EXCEEDED", directive.reason)

    def test_retry_requires_physical_stop_and_respects_recovery_limit(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        raw["workspace_clearance_verified"] = True
        raw["turn_capabilities"]["bounded_moving_turn_qualified"] = True
        raw["steps"][0]["recovery_limit"] = 1
        route = RouteSpec.from_mapping(raw)
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=0.1, y=0, yaw=0, now=0.1, stamp=2_000_000_000)
        stop = runner.tick(0.1, startup_feedback(status=MotionStatus.RETRYABLE_FAILURE))
        self.assertEqual(stop.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        pose.update(x=0.1, y=0, yaw=0, now=2.2, stamp=3_000_000_000)
        retry = runner.tick(2.2, startup_feedback(physical_stop_evidence=stop_evidence(0.1)))
        self.assertEqual(retry.kind, DirectiveKind.FOLLOW_WAYPOINT)
        pose.update(x=0.2, y=0, yaw=0, now=2.3, stamp=4_000_000_000)
        exhausted = runner.tick(2.3, startup_feedback(status=MotionStatus.RETRYABLE_FAILURE))
        self.assertEqual(exhausted.kind, DirectiveKind.STOP_AND_ABORT)
        self.assertIn("RECOVERY_LIMIT_EXCEEDED", exhausted.reason)

    def test_docking_handoff_requires_fresh_object_pose_ik_stop_and_manipulator_ready(self) -> None:
        pose = make_pose_provider()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(one_docking_route(), pose, FakeSafetyProvider(), motion)
        runner.start(0.0, startup_feedback())
        pose.update(x=1, y=0, yaw=0.2, now=1.0, stamp=2_000_000_000)
        runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        pose.update(x=1, y=0, yaw=0.2, now=3.0, stamp=3_000_000_000)
        runner.tick(3.0, startup_feedback(physical_stop_evidence=stop_evidence(1.0)))
        incomplete = DockingReadiness("base_link", 0.8, 0.9, True, True)
        pose.update(x=1, y=0, yaw=0.2, now=3.1, stamp=4_000_000_000)
        wait = runner.tick(3.1, startup_feedback(physical_stop_evidence=stop_evidence(1.0)), docking_readiness=incomplete)
        self.assertEqual(wait.kind, DirectiveKind.WAIT)
        self.assertFalse(any(d.kind is DirectiveKind.DOCKING_HANDOFF for d in motion.directives))
        ready = DockingReadiness("base_link", 0.05, 0.95, True, True)
        pose.update(x=1, y=0, yaw=0.2, now=3.2, stamp=5_000_000_000)
        handoff = runner.tick(3.2, startup_feedback(physical_stop_evidence=stop_evidence(1.0)), docking_readiness=ready)
        self.assertEqual(handoff.kind, DirectiveKind.DOCKING_HANDOFF)
        before = len([d for d in motion.directives if d.kind is DirectiveKind.DOCKING_HANDOFF])
        pose.update(x=1, y=0, yaw=0.2, now=3.3, stamp=6_000_000_000)
        runner.tick(3.3, startup_feedback(physical_stop_evidence=stop_evidence(1.0)), docking_readiness=ready)
        after = len([d for d in motion.directives if d.kind is DirectiveKind.DOCKING_HANDOFF])
        self.assertEqual(before, after)
        pose.update(x=1, y=0, yaw=0.2, now=3.4, stamp=7_000_000_000)
        completed = runner.tick(3.4, startup_feedback(physical_stop_evidence=stop_evidence(1.0)), manipulation_complete=True)
        self.assertEqual(completed.kind, DirectiveKind.COMPLETE)

    def test_zero_forward_turn_is_not_selected_by_route_template(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        self.assertTrue(all(step.get("turn", {}).get("strategy") == "BOUNDED_MOVING_TURN" for step in raw["steps"][:3]))
        self.assertFalse(raw["workspace_clearance_verified"])


if __name__ == "__main__":
    unittest.main()
