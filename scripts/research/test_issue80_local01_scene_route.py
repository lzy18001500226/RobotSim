#!/usr/bin/env python3
"""Offline tests for the pinned Local01 scene transform and bounded route profile."""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from issue80_local01_scene_route import (
    LOCAL01_COMMIT,
    LOCAL01_POSES_SHA256,
    LOCAL01_RUNTIME_X2_SHA256,
    build_short_route_config,
    build_runtime_raw_odom_short_route_config,
    build_runtime_raw_odom_table_approach_mission,
    build_runtime_raw_odom_aligned_start_table_approach_mission,
    load_local01_scene_inputs,
    prepare_local01_scene_runtime_bundle,
    validate_runtime_raw_odom_table_approach_mission,
    validate_runtime_raw_odom_table_approach_start,
    validate_runtime_raw_odom_aligned_start_table_approach_mission,
    validate_runtime_raw_odom_aligned_start_table_approach_start,
    validate_short_route_static_scene,
    native_name_scene_tree,
    aligned_start_scene_tree,
    audit_runtime_name_contract,
    assert_scene_name_only_equivalence,
)
from issue80_waypoint_route import (
    build_table_approach_route_from_mission,
    build_local01_short_scene_route,
    distance_to_polygon_boundary,
    validate_local01_reset_odom_pose,
    validate_local01_short_scene_route_input,
)


LOCAL01_POSES_PATH = Path(os.environ.get(
    "ROBOTSIM_LOCAL01_SCENE_WORLD_POSES",
    "/home/lzy18001500226/robotsim-issue77-raicom-x2-reuse/simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json",
))
ROBOTSIM_ROOT = Path("/home/lzy18001500226/projects/RobotSim")
CURRENT_NAVIGATION_POSES_PATH = (
    ROBOTSIM_ROOT / "simulation/mujoco/scenes/raicom2026/config/scene_world_poses.json"
)


def local01_source_fixture():
    return {
        "source_commit": LOCAL01_COMMIT,
        "source_hashes": {"scene_world_poses": LOCAL01_POSES_SHA256},
        "scene_world_poses": {
            "poses": {
                "START": {"xyz_m": [-1.5, -1.5, 0.68], "yaw_rad": math.pi / 2.0},
                "START_ALIGN": {"xyz_m": [-1.5, -1.5, 0.68], "yaw_rad": 0.0},
                "WAYPOINT": {"xyz_m": [-0.25, -1.5, 0.68], "yaw_rad": 0.0},
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
    @unittest.skipUnless(
        CURRENT_NAVIGATION_POSES_PATH.is_file(),
        "materialized Local01 inputs are unavailable in the navigation worktree",
    )
    def test_materialized_pinned_inputs_are_accepted_on_a_newer_navigation_head(self):
        source = load_local01_scene_inputs(
            CURRENT_NAVIGATION_POSES_PATH,
            require_clean_commit=False,
        )
        self.assertEqual(source["source_commit"], LOCAL01_COMMIT)
        self.assertEqual(
            source["source_provenance"],
            "PINNED_LOCAL01_COMMIT_PATHS_MATERIALIZED_IN_NAVIGATION_WORKTREE",
        )
        self.assertEqual(source["source_hashes"]["scene_world_poses"], LOCAL01_POSES_SHA256)
        self.assertEqual(
            Path(source["reference_root"]),
            ROBOTSIM_ROOT / "reference/AgiBot/raicom2026-x2-mujoco-scene",
        )

    def test_aligned_start_mission_retains_original_reference_without_runtime_turn(self):
        mission = build_runtime_raw_odom_aligned_start_table_approach_mission(
            local01_source_fixture()
        )
        validate_runtime_raw_odom_aligned_start_table_approach_mission(mission)
        self.assertEqual(
            mission["classification"],
            "SIMULATION_ONLY_ALIGNED_START_TABLE_APPROACH_NOT_ORIGINAL_START_OR_TABLE_DOCKING_OR_MANIPULATION",
        )
        self.assertEqual(mission["execution_order"], ["START", "WAYPOINT", "TABLE_APPROACH"])
        self.assertAlmostEqual(mission["poses"]["START"]["yaw_rad"], 0.0)
        self.assertAlmostEqual(mission["original_raicom_start_reference"]["yaw_rad"], math.pi / 2.0)
        self.assertEqual(
            mission["initial_alignment"]["status"],
            "NOT_APPLICABLE_DERIVED_INITIAL_YAW_ZERO_BEFORE_PHYSICS",
        )
        self.assertFalse(mission["derived_scene_initialization"]["active_simulation_qpos_write"])

    def test_aligned_start_requires_fresh_post_sd_pose_inside_derived_envelope(self):
        mission = build_runtime_raw_odom_aligned_start_table_approach_mission(
            local01_source_fixture()
        )
        accepted = validate_runtime_raw_odom_aligned_start_table_approach_start(
            mission, {"x_m": -1.5, "y_m": -1.5, "z_m": 0.68, "yaw_rad": 0.0}
        )
        self.assertEqual(accepted["status"], "PASS")
        rejected = validate_runtime_raw_odom_aligned_start_table_approach_start(
            mission, {"x_m": -1.5, "y_m": -1.34, "z_m": 0.68, "yaw_rad": 0.0}
        )
        self.assertEqual(rejected["status"], "FAIL")
        self.assertFalse(rejected["permission_to_publish_first_nonzero"])

    def test_aligned_start_route_reuses_the_existing_route_manager(self):
        mission = build_runtime_raw_odom_aligned_start_table_approach_mission(
            local01_source_fixture()
        )
        route = build_table_approach_route_from_mission(
            mission,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertEqual([step.waypoint_id for step in route.steps], ["WAYPOINT", "TABLE_APPROACH"])
        self.assertAlmostEqual(route.start_x_m, -1.5)
        self.assertAlmostEqual(route.start_yaw_rad, 0.0)

    def test_table_route_corridors_match_the_segmented_static_envelope(self):
        mission = build_runtime_raw_odom_aligned_start_table_approach_mission(
            local01_source_fixture()
        )
        route = build_table_approach_route_from_mission(
            mission,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        required_margin = (
            route.robot_footprint_radius_m + route.corridor_boundary_margin_m
        )
        # Regression: the real fixed-window run stopped at y=-1.5504109 m.
        # This is 50.4109 mm from the centerline: inside the statically
        # certified 100 mm open segment and outside the old generic 50 mm
        # corridor by less than one millimetre.
        observed_open_clearance = distance_to_polygon_boundary(
            -1.2420052865823277, -1.5504109039747291,
            route.steps[0].corridor_polygon,
        )
        open_clearance = distance_to_polygon_boundary(
            -1.0, -1.599, route.steps[0].corridor_polygon
        )
        table_clearance = distance_to_polygon_boundary(
            0.5, -1.559, route.steps[1].corridor_polygon
        )
        self.assertGreaterEqual(observed_open_clearance, required_margin)
        self.assertGreaterEqual(open_clearance, required_margin)
        self.assertGreaterEqual(table_clearance, required_margin)
        self.assertLess(
            distance_to_polygon_boundary(-1.0, -1.601, route.steps[0].corridor_polygon),
            required_margin,
        )
        self.assertLess(
            distance_to_polygon_boundary(-1.0, -1.561, route.steps[1].corridor_polygon),
            required_margin,
        )

    def test_table_approach_mission_preserves_docking_and_declares_turn_gate(self):
        mission = build_runtime_raw_odom_table_approach_mission(local01_source_fixture())
        validate_runtime_raw_odom_table_approach_mission(mission)
        self.assertEqual(
            mission["classification"],
            "SIMULATION_ONLY_TABLE_APPROACH_NOT_TABLE_DOCKING_OR_MANIPULATION",
        )
        self.assertEqual(
            mission["execution_order"],
            ["START", "START_ALIGN", "WAYPOINT", "TABLE_APPROACH"],
        )
        self.assertEqual(mission["poses"]["TABLE_APPROACH"], {
            "x_m": 0.9, "y_m": -1.5, "z_m": 0.68, "yaw_rad": 0.0,
        })
        self.assertEqual(mission["poses"]["TABLE_DOCKING_REFERENCE_UNCHANGED"], {
            "x_m": 1.0, "y_m": -1.5, "z_m": 0.68, "yaw_rad": 0.0,
        })
        self.assertEqual(
            mission["initial_alignment"]["zero_forward_angular_mc_status"],
            "NOT_QUALIFIED_REQUIRES_BOUNDED_LIVE_EVIDENCE",
        )
        self.assertEqual(mission["limits"]["max_windows"], 15)
        self.assertEqual(mission["limits"]["max_aggregate_nonzero_s"], 75.0)

    def test_table_approach_mission_rejects_relabelled_docking_pose(self):
        mission = build_runtime_raw_odom_table_approach_mission(local01_source_fixture())
        mission["poses"]["TABLE_DOCKING_REFERENCE_UNCHANGED"]["x_m"] = 0.9
        with self.assertRaisesRegex(ValueError, "TABLE_DOCKING reference"):
            validate_runtime_raw_odom_table_approach_mission(mission)

    def test_table_approach_requires_fresh_post_sd_start_inside_static_envelope(self):
        mission = build_runtime_raw_odom_table_approach_mission(local01_source_fixture())
        valid = validate_runtime_raw_odom_table_approach_start(
            mission, {"x_m": -1.5, "y_m": -1.5, "z_m": 0.68, "yaw_rad": math.pi / 2.0},
        )
        self.assertEqual(valid["status"], "PASS")
        drifted = validate_runtime_raw_odom_table_approach_start(
            mission,
            {"x_m": -1.5038967, "y_m": -1.3409248, "z_m": 0.6527287, "yaw_rad": 1.3995752},
        )
        self.assertEqual(drifted["status"], "FAIL")
        self.assertFalse(drifted["permission_to_publish_first_nonzero"])
        self.assertGreater(drifted["planar_position_error_m"], 0.025)

    def test_table_approach_route_reuses_route_spec_without_manipulator_handoff(self):
        mission = build_runtime_raw_odom_table_approach_mission(local01_source_fixture())
        route = build_table_approach_route_from_mission(
            mission,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
        )
        self.assertEqual([step.waypoint_id for step in route.steps], ["WAYPOINT", "TABLE_APPROACH"])
        self.assertEqual([step.profile.value for step in route.steps], ["TRANSIT", "TRANSIT"])
        self.assertTrue(all(step.stop_behavior.value == "VERIFY_PHYSICAL_STOP" for step in route.steps))
        self.assertAlmostEqual(route.start_yaw_rad, 0.0)
        self.assertAlmostEqual(route.start_position_tolerance_m, 0.05)
        self.assertAlmostEqual(route.start_yaw_tolerance_rad, 0.075)
        self.assertTrue(all(step.segment_timeout_s == 60.0 for step in route.steps))

        bounded_wall_time = build_table_approach_route_from_mission(
            mission,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
            segment_timeout_s=300.0,
        )
        self.assertTrue(
            all(step.segment_timeout_s == 300.0 for step in bounded_wall_time.steps)
        )

        rolling = build_table_approach_route_from_mission(
            mission,
            runtime_robot_footprint_radius_m=0.41565748847422734,
            verified_local_clear_radius_m=2.0,
            rolling_intermediate_waypoint=True,
        )
        self.assertEqual(rolling.steps[0].stop_behavior.value, "ROLLING")
        self.assertEqual(rolling.steps[-1].stop_behavior.value, "VERIFY_PHYSICAL_STOP")

    def test_runtime_raw_odom_route_uses_scene_numeric_start_and_static_envelope(self):
        route_input = build_runtime_raw_odom_short_route_config(local01_source_fixture())
        validate_local01_short_scene_route_input(route_input)
        self.assertEqual(
            route_input["route_kind"],
            "SIMULATION_ONLY_SHORT_ROUTE_RAW_ODOM_SCENE_WORLD_DIAGNOSTIC_NOT_TABLE_DOCKING",
        )
        self.assertAlmostEqual(route_input["start"]["x_m"], -1.5, places=9)
        self.assertAlmostEqual(route_input["start"]["y_m"], -1.5, places=9)
        self.assertAlmostEqual(route_input["start"]["yaw_rad"], math.pi / 2.0, places=9)
        self.assertAlmostEqual(route_input["waypoint"]["x_m"], -1.5, places=9)
        self.assertAlmostEqual(route_input["waypoint"]["y_m"], -1.3, places=9)
        self.assertEqual(
            route_input["runtime_coordinate_contract"]["numeric_coordinate_convention"],
            "RAICOM_SCENE_WORLD_AFTER_LIVE_RESET_GATE",
        )
        self.assertAlmostEqual(
            route_input["runtime_coordinate_contract"]["maximum_start_yaw_deviation_rad"],
            0.075,
        )

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
        contract = result["native_runtime_name_contract"]
        self.assertEqual(contract["status"], "PASS_NATIVE_RUNTIME_NAMES")
        self.assertEqual(contract["native_freejoint_names"], ["floating_base_joint"])
        self.assertEqual({k: len(v) for k, v in contract["native_inventory"].items()},
                         {"body": 32, "joint": 32, "actuator": 31, "sensor": 67})
        self.assertTrue(all(not v for v in contract["missing_native_names"].values()))
        self.assertEqual(len(contract["preexisting_configured_sensors_absent_from_native_model"]), 5)
        self.assertTrue(all(result["original_prefixed_name_contract"]["missing_native_names"].values()))
        eq = result["compiled_name_only_equivalence"]
        self.assertEqual(eq["status"], "PASS_COMPILED_PHYSICS_EQUIVALENT")
        self.assertGreater(eq["numeric_array_count"], 350)
        self.assertEqual(eq["maximum_fk_residual"], 0.0)
        self.assertEqual(result["native_robot_equivalence"]["status"], "PASS_NATIVE_ROBOT_UNDER_SCENE_START_TRANSFORM")
        self.assertEqual((result["native_timestep_s"], result["combined_scene_timestep_s"]), (0.001, 0.002))
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
            self.assertEqual(bundle["scene_wrapper_change_classification"], "X2_ATTACH_PREFIX_ONLY_AND_MODEL_FILE_PATHS")
            self.assertEqual(bundle["x2_attach_prefix_change"], {"source": "x2_", "runtime": ""})
            self.assertEqual(bundle["mug_wrapper_change_classification"], "COMPILER_ASSET_SEARCH_PATHS_ONLY")
            self.assertEqual(bundle["runtime_x2_sha256"], "3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3")
            self.assertTrue((local_scene / "scene_workspace_runtime.xml").is_file())
            self.assertTrue((local_scene / "mug_portable_runtime.xml").is_file())
            self.assertTrue((local_scene / "simulator_default_local01_scene.yaml").is_file())
            self.assertEqual(bundle["generated_runtime_name_contract"]["status"], "PASS_NATIVE_RUNTIME_NAMES")
            self.assertEqual(bundle["generated_wrapper_physics_equivalence"]["status"], "PASS_COMPILED_PHYSICS_EQUIVALENT")
            runtime_tree = ET.parse(local_scene / "scene_workspace_runtime.xml").getroot()
            self.assertEqual(runtime_tree.find("./worldbody/frame/attach[@model='x2_mc_runtime']").get("prefix"), "")
            self.assertEqual(runtime_tree.find("./worldbody/frame/attach[@model='raicom_room']").get("prefix"), "raicom_")
            self.assertEqual(runtime_tree.find("./worldbody/frame/attach[@model='raicom_mug']").get("prefix"), "mug_")

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_aligned_start_bundle_is_separately_named_and_changes_only_prephysics_root_yaw(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        runtime_root = ROBOTSIM_ROOT / "simulation/mujoco/raicom2026"
        source_scene = Path(source["repository_root"]) / "simulation/mujoco/scenes/raicom2026/scene_workspace.xml"
        source_tree = ET.parse(source_scene).getroot()
        source_frame = source_tree.find("./worldbody/frame[@name='x2_start_frame']")
        self.assertEqual(source_frame.get("quat"), "0.7071067811865476 0 0 0.7071067811865476")
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            evidence.mkdir()
            bundle = prepare_local01_scene_runtime_bundle(
                source, evidence, runtime_root=runtime_root, aligned_start=True
            )
            self.assertEqual(bundle["classification"], "SIMULATION_ONLY_ALIGNED_START_SCENE_WRAPPER_V1")
            self.assertEqual(bundle["runtime_scene_filename"], "scene_workspace_aligned_start_v1_runtime.xml")
            self.assertEqual(bundle["runtime_simulator_config_filename"], "simulator_default_local01_aligned_start_v1.yaml")
            self.assertEqual(bundle["generated_wrapper_physics_equivalence"]["status"], "PASS_ALIGNED_START_PHYSICAL_INVARIANTS")
            self.assertEqual(bundle["generated_native_robot_equivalence"]["status"], "PASS_NATIVE_ROBOT_UNDER_SCENE_START_TRANSFORM")
            runtime_tree = ET.parse(evidence / "local01_scene" / bundle["runtime_scene_filename"]).getroot()
            frame = runtime_tree.find("./worldbody/frame[@name='x2_start_frame']")
            self.assertEqual(frame.get("pos"), "-1.5 -1.5 0")
            self.assertEqual(frame.get("quat"), "1 0 0 0")
            self.assertEqual(
                runtime_tree.find("./worldbody/frame/attach[@model='x2_mc_runtime']").get("prefix"), ""
            )
            self.assertEqual(ET.parse(source_scene).getroot().find("./worldbody/frame[@name='x2_start_frame']").get("quat"), source_frame.get("quat"))

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_aligned_start_native_x2_timestep_wrapper_changes_only_the_explicit_parent_option(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        runtime_root = ROBOTSIM_ROOT / "simulation/mujoco/raicom2026"
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            evidence.mkdir()
            bundle = prepare_local01_scene_runtime_bundle(
                source,
                evidence,
                runtime_root=runtime_root,
                aligned_start=True,
                native_x2_timestep=True,
            )
            self.assertEqual(
                bundle["classification"],
                "SIMULATION_ONLY_ALIGNED_START_NATIVE_X2_TIMESTEP_SCENE_WRAPPER_V1",
            )
            self.assertEqual(
                bundle["runtime_scene_filename"],
                "scene_workspace_aligned_start_v1_native_x2_timestep_runtime.xml",
            )
            self.assertEqual(
                bundle["runtime_simulator_config_filename"],
                "simulator_default_local01_aligned_start_v1_native_x2_timestep.yaml",
            )
            timing = bundle["native_x2_timestep"]
            self.assertTrue(timing["enabled"])
            self.assertEqual(timing["native_x2_source_timestep_s"], 0.001)
            equivalence = bundle["generated_wrapper_physics_equivalence"]
            self.assertEqual(
                equivalence["status"],
                "PASS_ALIGNED_START_NATIVE_TIMESTEP_PHYSICAL_INVARIANTS",
            )
            self.assertEqual((equivalence["timestep_before_s"], equivalence["timestep_after_s"]), (0.002, 0.001))
            runtime_tree = ET.parse(evidence / "local01_scene" / bundle["runtime_scene_filename"]).getroot()
            option = runtime_tree.find("./option")
            self.assertIsNotNone(option)
            self.assertEqual(option.get("timestep"), "0.001")

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_synchronized_nominal_keyframe_is_source_mc_derived_and_physics_invariant(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        runtime_root = ROBOTSIM_ROOT / "simulation/mujoco/raicom2026"
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            evidence.mkdir()
            bundle = prepare_local01_scene_runtime_bundle(
                source,
                evidence,
                runtime_root=runtime_root,
                aligned_start=True,
                synchronized_nominal_initialization=True,
            )
            self.assertEqual(
                bundle["classification"],
                "SIMULATION_ONLY_ALIGNED_START_KEYFRAME_SYNCHRONIZED_INITIALIZATION_V1",
            )
            init = bundle["synchronized_nominal_initialization"]
            self.assertTrue(init["enabled"])
            self.assertEqual(init["keyframe"]["key_name"], "issue80_nominal_stand_v1")
            self.assertAlmostEqual(init["keyframe"]["derived_root_qpos"][2], 0.6593205061260876)
            self.assertTrue(all(init["keyframe"]["selected_foot_tangent_contact"].values()))
            self.assertEqual(
                init["keyframe_equivalence"]["status"],
                "PASS_SYNCHRONIZED_KEYFRAME_PHYSICAL_INVARIANTS",
            )
            tree = ET.parse(evidence / "local01_scene" / bundle["runtime_scene_filename"]).getroot()
            key = tree.find("./keyframe/key[@name='issue80_nominal_stand_v1']")
            self.assertIsNotNone(key)
            self.assertTrue((evidence / "local01_scene/synchronized_nominal_keyframe.json").is_file())

    @unittest.skipUnless(LOCAL01_POSES_PATH.is_file(), "Local01 source worktree is unavailable")
    def test_static_nominal_start_moves_only_preload_frame_z_to_the_nominal_contact_height(self):
        source = load_local01_scene_inputs(LOCAL01_POSES_PATH, require_clean_commit=True)
        runtime_root = ROBOTSIM_ROOT / "simulation/mujoco/raicom2026"
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            evidence.mkdir()
            bundle = prepare_local01_scene_runtime_bundle(
                source,
                evidence,
                runtime_root=runtime_root,
                aligned_start=True,
                static_nominal_start_initialization=True,
            )
            self.assertEqual(
                bundle["classification"],
                "SIMULATION_ONLY_ALIGNED_START_STATIC_NOMINAL_INITIALIZATION_V1",
            )
            init = bundle["static_nominal_start_initialization"]
            self.assertTrue(init["enabled"])
            state = init["initialization"]
            self.assertAlmostEqual(state["derived_frame_z_offset_m"], -0.020679493873912384)
            self.assertAlmostEqual(state["derived_start_frame_pos_m"][2], -0.020679493873912384)
            self.assertEqual(
                state["equivalence"]["status"],
                "PASS_STATIC_NOMINAL_START_PHYSICAL_INVARIANTS",
            )
            self.assertLessEqual(
                abs(state["post_static_contact_validation"]["root_height_shift_m"]), 1e-8,
            )
            tree = ET.parse(evidence / "local01_scene" / bundle["runtime_scene_filename"]).getroot()
            self.assertIsNone(tree.find("./keyframe"))
            frame = tree.find("./worldbody/frame[@name='x2_start_frame']")
            self.assertEqual(frame.get("quat"), "1 0 0 0")
            self.assertAlmostEqual(float(frame.get("pos").split()[2]), -0.020679493873912384)
            self.assertTrue((evidence / "local01_scene/static_nominal_start.json").is_file())

    def test_native_name_wrapper_rejects_unexpected_source_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "scene.xml"
            path.write_text('<mujoco><worldbody><frame><attach model="x2_mc_runtime" prefix="other_"/></frame></worldbody></mujoco>')
            with self.assertRaisesRegex(ValueError, "exactly one source X2 attach"):
                native_name_scene_tree(path)

    def test_physics_equivalence_rejects_inertial_and_actuator_changes(self):
        import mujoco
        xml = '<mujoco><worldbody><body name="pelvis"><joint name="j"/><geom type="sphere" size=".1"/></body></worldbody><actuator><motor name="motor_j" joint="j"/></actuator></mujoco>'
        a = mujoco.MjModel.from_xml_string(xml)
        b = mujoco.MjModel.from_xml_string(xml)
        assert_scene_name_only_equivalence(a, b)
        b.body_mass[1] *= 2
        with self.assertRaisesRegex(ValueError, "changed compiled physical array"):
            assert_scene_name_only_equivalence(a, b)
        b.body_mass[:] = a.body_mass
        b.actuator_gear[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "actuator_gear"):
            assert_scene_name_only_equivalence(a, b)

    def test_runtime_contract_rejects_missing_joint_sensor_or_odom_body(self):
        import mujoco
        native = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><joint name="j"/><geom type="sphere" size=".1"/></body></worldbody><sensor><jointpos name="q" joint="j"/></sensor></mujoco>')
        broken = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="x2_pelvis"><joint name="x2_j"/><geom type="sphere" size=".1"/></body></worldbody><sensor><jointpos name="x2_q" joint="x2_j"/></sensor></mujoco>')
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "default.yaml"
            cfg.write_text('actual:\n  active_joint_name: [[j]]\nsensor:\n  sensor_frame_type: [imu, odom]\n  sensor_name: [[q], [pelvis]]\n')
            good = audit_runtime_name_contract(native, native, cfg)
            bad = audit_runtime_name_contract(native, broken, cfg)
            self.assertEqual(good["status"], "PASS_NATIVE_RUNTIME_NAMES")
            self.assertEqual(bad["status"], "FAIL_NATIVE_RUNTIME_NAMES")
            self.assertEqual(bad["missing_configured_native_lookups"], {
                "joint": ["j"], "odom_body": ["pelvis"], "sensor_present_in_native": ["q"],
            })

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
