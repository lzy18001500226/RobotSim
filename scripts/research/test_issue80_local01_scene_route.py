#!/usr/bin/env python3
"""Offline tests for the pinned Local01 scene transform and bounded route profile."""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path

from issue80_local01_scene_route import (
    LOCAL01_COMMIT,
    LOCAL01_POSES_SHA256,
    LOCAL01_RUNTIME_X2_SHA256,
    build_short_route_config,
    load_local01_scene_inputs,
    prepare_local01_scene_runtime_bundle,
    validate_short_route_static_scene,
)
from issue80_waypoint_route import (
    build_local01_short_scene_route,
    validate_local01_reset_odom_pose,
    validate_local01_short_scene_route_input,
)


LOCAL01_POSES_PATH = Path(os.environ.get(
    "ROBOTSIM_LOCAL01_SCENE_WORLD_POSES",
    "/home/lzy18001500226/robotsim-issue77-raicom-x2-reuse/simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json",
))
ROBOTSIM_ROOT = Path("/home/lzy18001500226/projects/RobotSim")


def local01_source_fixture():
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
            "parent_frame": "odom",
            "child_frame": "raicom_scene_world",
            "translation_xyz_m": [1.5, -1.5, 0.0],
            "yaw_rad": -math.pi / 2.0,
        },
        "reset_anchor": {
            "base_pose_in_odom_after_reset": {"xyz_m": [0.0, 0.0, 0.68], "yaw_rad": 0.0}
        },
    }


class Local01SceneRouteTests(unittest.TestCase):
    def test_short_route_uses_local01_scene_transform_and_retains_full_dock_reference(self):
        route_input = build_short_route_config(local01_source_fixture())
        validate_local01_short_scene_route_input(route_input)
        self.assertAlmostEqual(route_input["start"]["x_m"], 0.0, places=9)
        self.assertAlmostEqual(route_input["start"]["y_m"], 0.0, places=9)
        self.assertEqual(route_input["start"]["yaw_rad"], 0.0)
        self.assertAlmostEqual(route_input["waypoint"]["x_m"], 0.2, places=9)
        self.assertAlmostEqual(route_input["short_route_end"]["x_m"], 0.4, places=9)
        self.assertEqual(route_input["route_kind"], "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING")
        self.assertAlmostEqual(
            route_input["safe_table_docking_reference"]["distance_from_start_m"], 2.5
        )
        self.assertEqual(
            route_input["safe_table_docking_reference"]["status"],
            "STATIC_BASELINE_ONLY_NOT_TARGETED_BY_SHORT_ROUTE",
        )

    def test_full_table_route_is_outside_existing_motion_budget(self):
        route_input = build_short_route_config(local01_source_fixture())
        budget = route_input["full_table_route_budget"]
        self.assertAlmostEqual(budget["distance_m"], 2.5)
        self.assertAlmostEqual(budget["maximum_translation_distance_m"], 1.8)
        self.assertGreater(budget["minimum_translation_time_at_0p12_m_s"], 15.0)
        self.assertEqual(budget["status"], "CANNOT_REACH_ENDPOINT_WITH_EXISTING_BUDGET")

    def test_short_route_spec_reuses_existing_route_manager_and_stop_contract(self):
        route_input = build_short_route_config(local01_source_fixture())
        route = build_local01_short_scene_route(
            route_input,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertEqual([step.waypoint_id for step in route.steps], ["WAYPOINT", "SHORT_ROUTE_END"])
        self.assertEqual([step.position_tolerance_m for step in route.steps], [0.15, 0.15])
        self.assertEqual([step.position_tolerance_m for step in route.steps], [0.15, 0.15])
        self.assertEqual([step.profile.value for step in route.steps], ["TRANSIT", "TRANSIT"])
        self.assertTrue(all(step.stop_behavior.value == "VERIFY_PHYSICAL_STOP" for step in route.steps))
        self.assertTrue(all(step.recovery_limit == 0 for step in route.steps))
        self.assertTrue(all(step.max_command_window_s == 5.0 for step in route.steps))
        self.assertFalse(any(step.waypoint_id == "TABLE_DOCKING" for step in route.steps))

    def test_reset_odometry_gate_fails_closed_on_world_pose_instead_of_reset_anchor(self):
        route_input = build_short_route_config(local01_source_fixture())
        expected = route_input["reset_anchor_expected_odom"]
        self.assertEqual(
            validate_local01_reset_odom_pose(expected, expected)["status"], "PASS"
        )
        world_pose = {"x_m": -1.5, "y_m": -1.5, "z_m": 0.68, "yaw_rad": math.pi / 2.0}
        mismatch = validate_local01_reset_odom_pose(expected, world_pose)
        self.assertEqual(mismatch["status"], "FAIL")
        self.assertFalse(mismatch["permission_to_publish_motion"])

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_pinned_local01_scene_static_route_includes_table_and_is_forward_only(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        route_input = build_short_route_config(source)
        scene_root = Path(source["repository_root"]) / "simulation/mujoco/scenes/raicom2026"
        result = validate_short_route_static_scene(scene_root, route_input)
        self.assertEqual(result["status"], "PASS_STATIC_SCENE_CLEARANCE")
        self.assertEqual(result["mujoco_version"], "3.3.6")
        self.assertEqual(result["mj_step_calls"], 0)
        self.assertEqual(result["cross_track_guard_inflation_m"], 0.05)
        self.assertGreaterEqual(result["minimum_swept_aabb_clearance_lower_bound_m"], 0.05)
        self.assertEqual(result["active_robot_scene_contacts"], [])
        self.assertEqual(source["source_commit"], LOCAL01_COMMIT)
        self.assertEqual(source["source_hashes"]["scene_world_poses"], LOCAL01_POSES_SHA256)

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_local01_runtime_scene_bundle_changes_paths_only_and_keeps_x2_hash(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        runtime_root = ROBOTSIM_ROOT / "simulation/mujoco/raicom2026"
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            evidence.mkdir()
            bundle = prepare_local01_scene_runtime_bundle(source, evidence, runtime_root=runtime_root)
            local_scene = evidence / "local01_scene"
            self.assertEqual(bundle["classification"], "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER")
            self.assertEqual(bundle["scene_wrapper_change_classification"], "MODEL_FILE_PATHS_ONLY")
            self.assertEqual(bundle["mug_wrapper_change_classification"], "COMPILER_ASSET_SEARCH_PATHS_ONLY")
            self.assertEqual(bundle["runtime_x2_sha256"], "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3")
            self.assertTrue((local_scene / "scene_workspace_runtime.xml").is_file())
            self.assertTrue((local_scene / "mug_portable_runtime.xml").is_file())
            self.assertTrue((local_scene / "simulator_default_local01_scene.yaml").is_file())

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_local01_runtime_bundle_rejects_x2_model_different_from_static_preflight(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            runtime_x2 = (
                root / "runtime/sim_mujoco/configuration/robot/lx2501_3_t2d5/model_info/x2.xml"
            )
            runtime_x2.parent.mkdir(parents=True)
            runtime_x2.write_text("<mujoco model='different-x2'/>", encoding="utf-8")
            self.assertNotEqual(
                hashlib.sha256(runtime_x2.read_bytes()).hexdigest(),
                LOCAL01_RUNTIME_X2_SHA256,
            )
            with self.assertRaisesRegex(ValueError, "differs from the source model"):
                prepare_local01_scene_runtime_bundle(source, evidence, runtime_root=root / "runtime")
            self.assertFalse((evidence / "local01_scene").exists())


if __name__ == "__main__":
    unittest.main()
