#!/usr/bin/env python3
"""Offline route-manager tests using synthetic measured-pose traces only."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
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
    StopBehavior,
    WaypointRouteExecutor,
    YawPolicy,
    build_waypoint_docking_route_from_poses,
    load_waypoint_docking_pose_input,
    validate_waypoint_docking_scene_clearance,
    validate_waypoint_docking_route,
    validate_local01_short_scene_route_input,
    build_local01_short_scene_route,
    validate_local01_reset_odom_pose,
    build_runtime_raw_odom_short_scene_route,
    validate_runtime_raw_odom_start_pose,
)
from issue80_route_motion_bridge import (
    build_curved_waypoint_route,
    build_shallow_curved_path_reference,
)
from issue80_coordinated_pose_goal import (
    PoseGoalPhaseController,
    TERMINAL_POSE_CONTROL,
    ZERO_STOP,
    terminal_yaw_rate_command,
)
from issue80_local01_scene_route import (
    build_short_route_config,
    build_runtime_raw_odom_short_route_config,
    LOCAL01_COMMIT,
    LOCAL01_POSES_SHA256,
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
    @staticmethod
    def local01_scene_source_fixture():
        return {
            "source_commit": LOCAL01_COMMIT,
            "source_hashes": {"scene_world_poses": LOCAL01_POSES_SHA256},
            "scene_world_poses": {
                "poses": {
                    "START": {"xyz_m": [-1.5, -1.5, 0.68], "yaw_rad": math.pi / 2.0},
                    "TABLE_DOCKING": {"xyz_m": [1.0, -1.5, 0.68], "yaw_rad": 0.0},
                }
            },
            "odom_from_scene": {
                "parent_frame": "odom", "child_frame": "raicom_scene_world",
                "translation_xyz_m": [1.5, -1.5, 0.0], "yaw_rad": -math.pi / 2.0,
            },
            "reset_anchor": {"base_pose_in_odom_after_reset": {
                "xyz_m": [0.0, 0.0, 0.68], "yaw_rad": 0.0,
            }},
        }

    @classmethod
    def local01_short_route_input(cls):
        return build_short_route_config(cls.local01_scene_source_fixture())

    @classmethod
    def local01_short_route(cls):
        return build_local01_short_scene_route(
            cls.local01_short_route_input(),
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )

    def test_local01_short_route_is_transformed_bounded_and_not_mislabeled_as_docking(self) -> None:
        raw = self.local01_short_route_input()
        validate_local01_short_scene_route_input(raw)
        route = build_local01_short_scene_route(
            raw,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertEqual([step.waypoint_id for step in route.steps], ["WAYPOINT", "SHORT_ROUTE_END"])
        self.assertTrue(all(step.profile is Profile.TRANSIT for step in route.steps))
        self.assertTrue(all(step.stop_behavior is StopBehavior.VERIFY_PHYSICAL_STOP for step in route.steps))
        self.assertTrue(all(step.recovery_limit == 0 for step in route.steps))
        self.assertAlmostEqual(route.start_x_m, 0.0)
        self.assertAlmostEqual(route.steps[0].x_m, 0.2)
        self.assertAlmostEqual(route.steps[1].x_m, 0.4)
        self.assertFalse(any(step.waypoint_id == "TABLE_DOCKING" for step in route.steps))
        self.assertEqual(raw["safe_table_docking_reference"]["status"], "STATIC_BASELINE_ONLY_NOT_TARGETED_BY_SHORT_ROUTE")

    def test_runtime_raw_odom_short_route_uses_fresh_measured_heading(self) -> None:
        raw = build_runtime_raw_odom_short_route_config(self.local01_scene_source_fixture())
        observed = {"x_m": -1.504, "y_m": -1.481, "z_m": 0.653, "yaw_rad": 1.602}
        gate = validate_runtime_raw_odom_start_pose(raw, observed)
        self.assertEqual(gate["status"], "PASS")
        route, plan = build_runtime_raw_odom_short_scene_route(
            raw,
            observed_start_pose=observed,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertAlmostEqual(route.start_x_m, observed["x_m"], places=9)
        self.assertAlmostEqual(route.start_y_m, observed["y_m"], places=9)
        self.assertAlmostEqual(route.start_yaw_rad, observed["yaw_rad"], places=9)
        self.assertAlmostEqual(
            route.steps[0].y_m,
            observed["y_m"] + 0.20 * math.sin(observed["yaw_rad"]),
            places=9,
        )
        self.assertEqual(plan["segments_m"], [0.20, 0.20])
        self.assertEqual(plan["runtime_start_gate"]["status"], "PASS")
        self.assertAlmostEqual(plan["forward_m_s"], 0.12)
        self.assertTrue(all(step.target_forward_m_s == 0.12 for step in route.steps))
        blocked = validate_runtime_raw_odom_start_pose(
            raw, {**observed, "yaw_rad": observed["yaw_rad"] + 0.05}
        )
        self.assertEqual(blocked["status"], "FAIL")

    def test_runtime_raw_odom_short_route_propagates_the_explicit_low_cruise_profile(self) -> None:
        raw = build_runtime_raw_odom_short_route_config(self.local01_scene_source_fixture())
        observed = {"x_m": -1.504, "y_m": -1.481, "z_m": 0.653, "yaw_rad": 1.602}
        route, plan = build_runtime_raw_odom_short_scene_route(
            raw,
            observed_start_pose=observed,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
            forward_m_s=0.105,
        )
        self.assertEqual(plan["motion_profile"], "SIMULATION_ONLY_RAICOM_LOW_CRUISE_RECOVERY")
        self.assertAlmostEqual(plan["forward_m_s"], 0.105)
        self.assertTrue(all(step.target_forward_m_s == 0.105 for step in route.steps))
        self.assertTrue(all(step.max_forward_m_s == 0.105 for step in route.steps))
        with self.assertRaisesRegex(ValueError, "short route permits only"):
            build_runtime_raw_odom_short_scene_route(
                raw,
                observed_start_pose=observed,
                runtime_robot_footprint_radius_m=0.41565748847422734,
                verified_local_clear_radius_m=2.0,
                forward_m_s=0.104,
            )

    def test_local01_short_route_rejects_overbudget_or_nonforward_targets(self) -> None:
        raw = self.local01_short_route_input()
        raw["limits"]["max_aggregate_nonzero_s"] = 15.1
        with self.assertRaisesRegex(ValueError, "two 5 s windows"):
            validate_local01_short_scene_route_input(raw)
        raw = self.local01_short_route_input()
        raw["short_route_end"]["x_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "two 0.20 m segments"):
            validate_local01_short_scene_route_input(raw)

    def test_local01_reset_odom_gate_compares_actual_sample_and_fails_closed(self) -> None:
        expected = {"x_m": 0.0, "y_m": 0.0, "z_m": 0.68, "yaw_rad": 0.0}
        passed = validate_local01_reset_odom_pose(expected, dict(expected))
        self.assertEqual(passed["status"], "PASS")
        self.assertTrue(passed["permission_to_publish_motion"])
        mismatch = validate_local01_reset_odom_pose(
            expected, {"x_m": -1.5, "y_m": -1.5, "z_m": 0.68, "yaw_rad": 1.5707963267948966}
        )
        self.assertEqual(mismatch["status"], "FAIL")
        self.assertFalse(mismatch["permission_to_publish_motion"])

    def test_local01_source_pose_converts_to_two_short_targets_not_full_table_goal(self) -> None:
        route_input = self.local01_short_route_input()
        validate_local01_short_scene_route_input(route_input)
        self.assertAlmostEqual(route_input["start"]["x_m"], 0.0, places=9)
        self.assertAlmostEqual(route_input["start"]["y_m"], 0.0, places=9)
        self.assertAlmostEqual(route_input["waypoint"]["x_m"], 0.2, places=9)
        self.assertAlmostEqual(route_input["short_route_end"]["x_m"], 0.4, places=9)
        self.assertAlmostEqual(route_input["safe_table_docking_reference"]["distance_from_start_m"], 2.5)
        self.assertEqual(route_input["full_table_route_budget"]["status"], "CANNOT_REACH_ENDPOINT_WITH_EXISTING_BUDGET")

    @staticmethod
    def waypoint_docking_config():
        return {
            "schema_version": "robotsim.issue80.waypoint-docking-poses.v1",
            "route_id": "local01-waypoint-docking-fixture",
            "frame_id": "odom",
            "start": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            "waypoint": {"x_m": 0.25, "y_m": 0.0, "yaw_rad": 0.0},
            "table_docking": {"x_m": 0.6, "y_m": 0.1, "yaw_rad": 0.1},
        }

    @classmethod
    def waypoint_docking_route(cls):
        return build_waypoint_docking_route_from_poses(
            cls.waypoint_docking_config(),
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )

    @staticmethod
    def load_pose_input(raw):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "route.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return load_waypoint_docking_pose_input(path)

    def test_waypoint_docking_profile_accepts_two_bounded_stop_gated_steps(self) -> None:
        route = self.waypoint_docking_route()
        self.assertEqual([step.waypoint_id for step in route.steps], ["WAYPOINT", "TABLE_DOCKING"])
        self.assertEqual((route.start_position_tolerance_m, route.start_yaw_tolerance_rad), (0.02, 0.02))
        for step in route.steps:
            self.assertEqual(step.target_forward_m_s, 0.12)
            self.assertEqual(step.max_forward_m_s, 0.12)
            self.assertEqual(step.max_angular_rad_s, 0.075)
            self.assertEqual(step.max_command_window_s, 5.0)
            self.assertEqual(step.recovery_limit, 0)
            self.assertEqual(step.stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertEqual(route.steps[0].stop_behavior.value, "VERIFY_PHYSICAL_STOP")
        self.assertEqual(route.steps[1].profile, Profile.DOCKING)
        self.assertEqual(route.steps[1].yaw_policy, YawPolicy.EXPLICIT)
        self.assertEqual(route.steps[1].position_tolerance_m, 0.08)
        self.assertEqual(route.steps[1].yaw_tolerance_rad, 0.10)

    def test_waypoint_docking_profile_rejects_expanded_motion_or_retry_limits(self) -> None:
        route = self.waypoint_docking_route()
        for index, changed in (
            (0, replace(route.steps[0], max_forward_m_s=0.121)),
            (0, replace(route.steps[0], max_angular_rad_s=0.076)),
            (1, replace(route.steps[1], max_command_window_s=5.01)),
            (1, replace(route.steps[1], recovery_limit=1)),
            (1, replace(route.steps[1], stop_behavior=StopBehavior.ROLLING)),
        ):
            steps = list(route.steps)
            steps[index] = changed
            with self.subTest(step=changed.waypoint_id, field=changed):
                with self.assertRaises(ValueError):
                    validate_waypoint_docking_route(replace(route, steps=tuple(steps)))

    def test_waypoint_docking_pose_input_rejects_missing_frame_yaw_or_short_segment(self) -> None:
        raw = self.waypoint_docking_config()
        raw["frame_id"] = "map"
        with self.assertRaisesRegex(ValueError, "ground-truth odom"):
            self.load_pose_input(raw)

        raw = self.waypoint_docking_config()
        raw["table_docking"].pop("yaw_rad")
        with self.assertRaisesRegex(ValueError, "exactly x_m, y_m, yaw_rad"):
            self.load_pose_input(raw)

        raw = self.waypoint_docking_config()
        raw["waypoint"]["x_m"] = 0.03
        with self.assertRaisesRegex(ValueError, "0.05 m stop/noise envelope"):
            self.load_pose_input(raw)

        raw = self.waypoint_docking_config()
        raw["unreviewed_limit"] = 0.16
        with self.assertRaisesRegex(ValueError, "exactly schema_version"):
            self.load_pose_input(raw)

        for invalid in (True, "0.25"):
            raw = self.waypoint_docking_config()
            raw["waypoint"]["x_m"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "must be a JSON number"):
                self.load_pose_input(raw)

    def test_waypoint_docking_route_checks_the_inspected_scene_footprint_and_clear_radius(self) -> None:
        route = self.waypoint_docking_route()
        evidence = validate_waypoint_docking_scene_clearance(
            route,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertEqual(evidence["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "smaller than the inspected"):
            validate_waypoint_docking_scene_clearance(
                route,
                runtime_robot_footprint_radius_m=0.46,
                verified_local_clear_radius_m=2.0,
            )
        with self.assertRaisesRegex(ValueError, "clear radius"):
            validate_waypoint_docking_scene_clearance(
                route,
                runtime_robot_footprint_radius_m=0.41565748847422734,
                verified_local_clear_radius_m=0.45,
            )

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
        with self.assertRaisesRegex(ValueError, "below the validated command floor"):
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

    def test_configured_two_step_route_stops_once_at_each_pose_and_waits_at_docking(self) -> None:
        route = RouteConfigTests.waypoint_docking_route()

        class OdomStore:
            def __init__(self):
                self.pose = PoseEstimate(
                    frame_id="odom", child_frame_id="base_link", x_m=0.0, y_m=0.0, yaw_rad=0.0,
                    stamp_ns=1_000_000_000, received_steady_s=0.0, generation="epoch-1",
                    source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
                    transform_chain=("odom->base_link",), transform_age_s=0.01,
                )

            def latest_pose(self, target_frame, now_steady_s):
                return self.pose if target_frame == "odom" else None

            def set(self, x, y, yaw, now, stamp):
                self.pose = PoseEstimate(
                    frame_id="odom", child_frame_id="base_link", x_m=x, y_m=y, yaw_rad=yaw,
                    stamp_ns=stamp, received_steady_s=now, generation="epoch-1",
                    source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
                    transform_chain=("odom->base_link",), transform_age_s=0.01,
                )

        class OdomSafety:
            def latest_observation(self, segment_id, frame_id, now_steady_s):
                return SafetyObservation(segment_id, frame_id, now_steady_s, True, True, 2.0)

        pose = OdomStore()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, OdomSafety(), motion)
        start = runner.start(0.0, MotionFeedback(
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            initial_activation_authorized=True,
        ))
        self.assertEqual(start.kind, DirectiveKind.FOLLOW_WAYPOINT)

        waypoint = route.steps[0]
        pose.set(waypoint.x_m, waypoint.y_m, 0.0, 1.0, 2_000_000_000)
        stop_waypoint = runner.tick(1.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(stop_waypoint.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(len(motion.stop_reasons), 1)
        pose.set(waypoint.x_m, waypoint.y_m, 0.0, 3.0, 3_000_000_000)
        next_step = runner.tick(3.0, MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=stop_evidence(1.01),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(next_step.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(runner.active_step.waypoint_id, "TABLE_DOCKING")
        self.assertEqual(len(motion.stop_reasons), 1)

        docking = route.steps[1]
        pose.set(docking.x_m, docking.y_m, docking.yaw_rad, 4.0, 4_000_000_000)
        stop_docking = runner.tick(4.0, MotionFeedback(
            status=MotionStatus.ARRIVED,
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=True,
            safety_gates_pass=True,
        ))
        self.assertEqual(stop_docking.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(len(motion.stop_reasons), 2)
        pose.set(docking.x_m, docking.y_m, docking.yaw_rad, 6.0, 5_000_000_000)
        terminal = runner.tick(6.0, MotionFeedback(
            status=MotionStatus.ACTIVE,
            physical_stop_evidence=stop_evidence(4.01),
            reset_gate_pass=True,
            standing_gate_pass=True,
            source_mode_handshake_pass=False,
            safety_gates_pass=True,
            stand_readback_pass=True,
        ))
        self.assertEqual(terminal.kind, DirectiveKind.WAIT)
        self.assertEqual(runner.state, RouteState.WAITING_FOR_DOCKING)
        self.assertEqual(len(motion.stop_reasons), 2)
        self.assertFalse(any(d.kind is DirectiveKind.DOCKING_HANDOFF for d in motion.directives))

    def test_local01_short_route_uses_existing_stop_rearm_and_completes_without_docking_claim(self) -> None:
        route = RouteConfigTests.local01_short_route()

        class OdomStore:
            def __init__(self):
                self.pose = PoseEstimate(
                    frame_id="odom", child_frame_id="base_link", x_m=0.0, y_m=0.0, yaw_rad=0.0,
                    stamp_ns=1_000_000_000, received_steady_s=0.0, generation="epoch-1",
                    source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
                    transform_chain=("odom->base_link",), transform_age_s=0.01,
                )

            def latest_pose(self, target_frame, now_steady_s):
                return self.pose if target_frame == "odom" else None

            def set(self, x, y, now, stamp):
                self.pose = PoseEstimate(
                    frame_id="odom", child_frame_id="base_link", x_m=x, y_m=0.0, yaw_rad=0.0,
                    stamp_ns=stamp, received_steady_s=now, generation="epoch-1",
                    source="MUJOCO_GROUND_TRUTH", position_std_m=0.0, yaw_std_rad=0.0,
                    transform_chain=("odom->base_link",), transform_age_s=0.01,
                )

        pose = OdomStore()
        motion = FakeMotionAdapter()
        runner = WaypointRouteExecutor(route, pose, FakeSafetyProvider(), motion)
        first = runner.start(0.0, startup_feedback())
        self.assertEqual(first.waypoint_id, "WAYPOINT")

        pose.set(0.2, 0.0, 1.0, 2_000_000_000)
        stop_first = runner.tick(1.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(stop_first.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(len(motion.stop_reasons), 1)

        pose.set(0.2, 0.0, 3.0, 3_000_000_000)
        second = runner.tick(
            3.0,
            startup_feedback(
                physical_stop_evidence=stop_evidence(1.0),
                source_mode_handshake_pass=False,
                stand_readback_pass=True,
                input_source_readback_pass=True,
            ),
        )
        self.assertEqual(second.kind, DirectiveKind.FOLLOW_WAYPOINT)
        self.assertEqual(runner.active_step.waypoint_id, "SHORT_ROUTE_END")
        self.assertEqual(len(motion.stop_reasons), 1)

        pose.set(0.4, 0.0, 4.0, 4_000_000_000)
        stop_second = runner.tick(4.0, startup_feedback(status=MotionStatus.ARRIVED))
        self.assertEqual(stop_second.kind, DirectiveKind.REQUEST_PHYSICAL_STOP)
        self.assertEqual(len(motion.stop_reasons), 2)

        pose.set(0.4, 0.0, 6.0, 5_000_000_000)
        complete = runner.tick(
            6.0,
            startup_feedback(
                physical_stop_evidence=stop_evidence(4.0),
                source_mode_handshake_pass=False,
                stand_readback_pass=True,
                input_source_readback_pass=True,
            ),
        )
        self.assertEqual(complete.kind, DirectiveKind.COMPLETE)
        self.assertEqual(runner.state, RouteState.COMPLETE)
        self.assertEqual(len(motion.stop_reasons), 2)
        self.assertFalse(any(d.kind is DirectiveKind.DOCKING_HANDOFF for d in motion.directives))

    def test_docking_pose_goal_uses_bounded_terminal_yaw_then_zero_stop_phase(self) -> None:
        controller = PoseGoalPhaseController(
            goal_distance_m=0.6,
            final_position_tolerance_m=0.08,
            maximum_lateral_error_m=0.05,
            maximum_overshoot_m=0.05,
            final_yaw_tolerance_rad=0.10,
            terminal_yaw_setpoint_tolerance_rad=0.10,
        )
        entered = controller.update(
            monotonic_s=1.0,
            position_stop_reason="goal_approach_predicted_settle_within_tolerance",
            distance_to_goal_m=0.07,
            lateral_error_m=0.01,
            along_track_progress_m=0.55,
            yaw_error_rad=0.20,
            measured_yaw_rate_rad_s=0.0,
            terminal_setpoint_error_rad=0.20,
        )
        self.assertEqual(entered["phase"], TERMINAL_POSE_CONTROL)
        command = terminal_yaw_rate_command(
            current_yaw_rad=-0.10,
            target_yaw_rad=0.10,
            measured_yaw_rate_rad_s=0.0,
            maximum_angular_rad_s=0.075,
            final_yaw_tolerance_rad=0.10,
        )
        self.assertGreater(command["command_yaw_rate_rad_s"], 0.0)
        self.assertLessEqual(abs(command["command_yaw_rate_rad_s"]), 0.075)
        controller.update(
            monotonic_s=1.1,
            position_stop_reason="goal_approach_predicted_settle_within_tolerance",
            distance_to_goal_m=0.06,
            lateral_error_m=0.01,
            along_track_progress_m=0.56,
            yaw_error_rad=0.05,
            measured_yaw_rate_rad_s=0.001,
            terminal_setpoint_error_rad=0.05,
        )
        ready = controller.update(
            monotonic_s=1.51,
            position_stop_reason="goal_approach_predicted_settle_within_tolerance",
            distance_to_goal_m=0.06,
            lateral_error_m=0.01,
            along_track_progress_m=0.56,
            yaw_error_rad=0.05,
            measured_yaw_rate_rad_s=0.001,
            terminal_setpoint_error_rad=0.05,
        )
        self.assertEqual(ready["phase"], ZERO_STOP)
        self.assertTrue(ready["terminal_pose_ready"])

    def test_zero_forward_turn_is_not_selected_by_route_template(self) -> None:
        raw = json.loads(RECTANGLE.read_text(encoding="utf-8"))
        self.assertTrue(all(step.get("turn", {}).get("strategy") == "BOUNDED_MOVING_TURN" for step in raw["steps"][:3]))
        self.assertFalse(raw["workspace_clearance_verified"])


if __name__ == "__main__":
    unittest.main()
