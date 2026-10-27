"""Offline tests for exact, no-overwrite research-code staging."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import issue80_m1_auto_reset_run
from issue80_local01_scene_route import build_short_route_config
from issue80_m1_auto_reset_run import (
    BoundedRun, MC_LAUNCH, M1_FIRST_MOTION_LOGIC, M1_MODE_LOGIC, MC_RL_DEBUG_CAPTURE,
    POSE_GOAL_CONTROLLER,
    SET_ACTION_CAPTURE, SIM_LAUNCH,
    assess_container_absence, build_m1_child_command, classify_activation_probe_outcome,
    classify_first_motion_outcome, classify_run38_diagnostic_outcome,
    assess_activation_probe_prerequisite, decode_m1_child_result, inspect_route_workcell_geometry,
    validate_waypoint_docking_route_input,
    validate_configured_route_motion_profile,
    validate_stand_convergence_arguments,
    assess_static_nominal_start_sample,
    assess_prearmed_nominal_handoff_sample,
    stage_mc_generated_config_writeback,
    validate_static_nominal_start_table_approach_clearance,
    validate_fixed_window_open_segment_yaw_envelope,
)


class ResearchScriptStagingTests(unittest.TestCase):
    def test_fixed_window_static_plan_requires_open_only_yaw_envelope(self) -> None:
        plan = {
            "clearance_assumptions": {
                "yaw_envelope_by_segment_rad": {
                    "START->WAYPOINT": 0.10,
                    "WAYPOINT->TABLE_APPROACH": 0.075,
                }
            }
        }
        accepted = validate_fixed_window_open_segment_yaw_envelope(plan)
        self.assertEqual(accepted["status"], "PASS_FIXED_WINDOW_OPEN_SEGMENT_YAW_ENVELOPE")
        plan["clearance_assumptions"]["yaw_envelope_by_segment_rad"]["WAYPOINT->TABLE_APPROACH"] = 0.10
        with self.assertRaises(ValueError):
            validate_fixed_window_open_segment_yaw_envelope(plan)

    def test_local01_short_route_preserves_static_sample_count_for_runtime_evidence(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"static_sample_count": int(static_scene["static_sample_count"])',
            source,
        )
        self.assertIn('"mj_step_calls": int(static_scene["mj_step_calls"])', source)

    def test_prearmed_nominal_handoff_allows_only_the_early_bounded_contact_sample(self) -> None:
        expected_qpos = [0.0] * 45
        expected_qpos[7:19] = [-0.24, 0.0, 0.0, 0.45, -0.21, 0.0] * 2
        sample = {
            "qpos": list(expected_qpos),
            "captured_foot_floor_contact_count": 2,
            "pelvis": {"state": {
                "world_position_m": [0.0, 0.0, 0.6593],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "world_spatial_velocity_rot_lin": [0.0, 0.0, 0.0, 0.03, 0.0, -0.08],
            }},
            # Tangent contact may have no positive impulse before the control
            # handoff; the early eligibility path must not call it standing.
            "contacts": [
                {"involves_selected_left_foot": True, "normal_force_n": 0.0},
                {"involves_selected_right_foot": True, "normal_force_n": 0.0},
            ],
        }
        accepted = assess_prearmed_nominal_handoff_sample(sample, {"qpos": expected_qpos})
        self.assertTrue(accepted["eligible"])
        self.assertEqual(
            accepted["classification"],
            "PREARMED_NOMINAL_HANDOFF_ELIGIBILITY_NOT_STANDING",
        )
        sample["pelvis"]["state"]["world_spatial_velocity_rot_lin"][5] = -0.21
        self.assertFalse(
            assess_prearmed_nominal_handoff_sample(sample, {"qpos": expected_qpos})["eligible"]
        )

    def test_mc_prearm_stages_only_evidence_local_generated_config_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            staged_dump, manifest = stage_mc_generated_config_writeback(evidence)
            self.assertTrue(staged_dump.is_file())
            self.assertEqual(manifest["permitted_write_scope"], [
                issue80_m1_auto_reset_run.MC_SIM_CONFIG_DUMP_CONTAINER_PATH,
                issue80_m1_auto_reset_run.MC_SIM_GENERATED_DIRECTORY,
            ])
            self.assertTrue(manifest["runtime_readonly"])
            self.assertFalse(manifest["vendor_runtime_tree_modified"])
            self.assertEqual(
                set(manifest["seed_hashes"]),
                set(issue80_m1_auto_reset_run.MC_SIM_GENERATED_SEED_FILENAMES),
            )
            self.assertTrue((evidence / "mc_runtime_writeback_manifest.json").is_file())

    def test_mc_prearm_seed_uses_the_container_evidence_mount_not_host_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                evidence_dir=directory,
                container="prearm-seed-test",
                max_runtime_s=30.0,
                runtime_readonly=True,
                static_nominal_start_initialization=True,
                mc_prearm_before_simulator=True,
            )
            runner = BoundedRun(args)
            runner.prepare_mc_prearm_writeback()
            expected = dict(runner.mc_prearm_writeback_manifest["seed_hashes"])
            commands: list[str] = []

            def fake_docker_exec(name, command, **_kwargs):
                commands.append(command)
                if name.startswith("verify_mc_prearm_seed_"):
                    filename = name.removeprefix("verify_mc_prearm_seed_")
                    return SimpleNamespace(stdout=expected[filename] + " target\n")
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            runner.docker_exec = fake_docker_exec
            result = runner.seed_mc_prearm_generated_config()

        self.assertEqual(result["status"], "PASS_SEEDED_EPHEMERAL_GENERATED_CONFIG")
        self.assertIn(
            "/workspace/evidence/mc_runtime_writeback/sim_seed/action_setting.yaml",
            commands[0],
        )
        self.assertNotIn("/home/lzy18001500226", commands[0])

    def test_static_nominal_start_requires_vendor_nominal_legs_and_existing_safe_support(self) -> None:
        expected_qpos = [0.0] * 45
        expected_qpos[7:19] = [-0.24, 0.0, 0.0, 0.45, -0.21, 0.0] * 2
        sample = {
            "qpos": list(expected_qpos),
            "pelvis": {"state": {
                "world_position_m": [0.0, 0.0, 0.6593],
                "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "world_spatial_velocity_rot_lin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }},
            "contacts": [
                {"involves_selected_left_foot": True, "normal_force_n": 100.0},
                {"involves_selected_right_foot": True, "normal_force_n": 100.0},
            ],
        }
        accepted = assess_static_nominal_start_sample(sample, {"qpos": expected_qpos})
        self.assertTrue(accepted["eligible"])
        sample["qpos"][7] += 0.021
        self.assertFalse(assess_static_nominal_start_sample(sample, {"qpos": expected_qpos})["eligible"])

    def test_static_nominal_start_transfers_only_the_reviewed_table_approach_clearance(self) -> None:
        plan_path = (
            Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-80")
            / "raicom-table-plan-20261026T131502Z/table_plan.json"
        )
        if not plan_path.is_file():
            self.skipTest("pinned table plan unavailable")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        result = validate_static_nominal_start_table_approach_clearance(
            plan, root_z_offset_m=-0.020679493873912384,
        )
        self.assertEqual(result["status"], "PASS_STATIC_NOMINAL_START_TABLE_APPROACH_CLEARANCE")
        self.assertGreater(result["minimum_conservative_transferred_lower_bound_m"], 0.020)
        self.assertEqual({(row["from"], row["to"]) for row in result["segments"]}, {
            ("START", "WAYPOINT"), ("WAYPOINT", "TABLE_APPROACH"),
        })

    def test_passive_video_recorder_does_not_start_in_the_reset_to_jd_window(self) -> None:
        source = Path(__file__).with_name("issue80_m1_auto_reset_run.py").read_text(
            encoding="utf-8"
        )
        execute = source[source.index("    def execute("):]
        reset_dispatch = execute.index("self.dispatch_native_reset_button")
        warmup = execute.index("self.evaluate_standing_warmup")
        video = execute.index("self.start_verified_motion_video_capture")
        self.assertLess(reset_dispatch, warmup)
        self.assertLess(warmup, video)

    def test_configured_local01_route_can_require_existing_stand_warmup_gate(self) -> None:
        args = argparse.Namespace(
            require_stand_warmup_composite=True,
            maintainer_authorized_first_motion=True,
            run65_waypoint_b_extended_diagnostic=False,
            local01_scene_world_poses="/pinned/local01/scene_world_poses.json",
            route_config=None,
            post_locomotion_convergence_observation_s=None,
            extra_stop_observation=0.0,
        )
        profile = validate_stand_convergence_arguments(args)
        self.assertEqual(profile["status"], "SIMULATION_ONLY_STAND_WARMUP_COMPOSITE")
        self.assertTrue(profile["existing_composite_stop_gate_unchanged"])

        args.maintainer_authorized_first_motion = False
        with self.assertRaisesRegex(ValueError, "requires authorized extended-B or configured-route"):
            validate_stand_convergence_arguments(args)

    def test_extended_b_still_requires_its_explicit_post_locomotion_observation(self) -> None:
        args = argparse.Namespace(
            require_stand_warmup_composite=True,
            maintainer_authorized_first_motion=True,
            run65_waypoint_b_extended_diagnostic=True,
            route_config=None,
            local01_scene_world_poses=None,
            post_locomotion_convergence_observation_s=None,
            extra_stop_observation=0.0,
        )
        with self.assertRaisesRegex(ValueError, "explicit post-locomotion observation horizon"):
            validate_stand_convergence_arguments(args)

    def test_configured_route_profile_uses_route_caps_and_existing_stop_observation(self) -> None:
        args = argparse.Namespace(
            route_config=None,
            local01_scene_world_poses="/pinned/local01/scene_world_poses.json",
            forward_speed=0.12,
            pulse_duration=1.0,
            extra_stop_observation=7.0,
            ramp_duration_s=0.0,
            ramp_start_speed=None,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
            capture_mc_rl_debug=False,
            route_turn_evidence=None,
        )
        validate_configured_route_motion_profile(args)

        args.forward_speed = 0.13
        with self.assertRaisesRegex(ValueError, "configured route requires 0.12 m/s"):
            validate_configured_route_motion_profile(args)

        args.forward_speed = 0.105
        args.raicom_low_cruise_profile = True
        validate_configured_route_motion_profile(args)
        args.forward_speed = 0.104
        with self.assertRaisesRegex(ValueError, "requires 0.105 m/s"):
            validate_configured_route_motion_profile(args)

    def test_activation_probe_requires_safe_zero_preflight_and_prior_nonzero_source_evidence(self) -> None:
        fixture_path = Path(__file__).with_name("fixtures") / "issue80_curved_abort_transition_replay.json"
        prior_replay = json.loads(fixture_path.read_text(encoding="utf-8"))
        reset_run = {
            "attempt_id": "zero-only-attempt",
            "status": "BLOCKED",
            "reset_classification": "AUTO_RESET_PASS",
            "runtime_image_digest": "sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31",
        }
        observation = {
            "classification": "PENDING_STAND_REMAINS_ACTIVE",
            "get_mode_desc": "STAND_DEFAULT",
            "common_mode_desc": "STAND_DEFAULT",
            "get_source_name": "",
            "common_source_name": "",
            "safety_fault": None,
        }
        zero_preflight = {
            "attempt_id": "zero-only-attempt",
            "simulation_only": True,
            "installed_mc_sim_config_sha256": "999d75f0caed5f20479c670fb95b624910c66722b47d37b60217717a3fe34b60",
            "source_registration": {
                "success": True,
                "accepted_by_service": True,
                "task_completion_confirmed": False,
                "task_state_value": 0,
            },
            "set_mc_action_ld": {"accepted_by_service": True},
            "zero_only_activation_verification": {
                "status": "BLOCKED",
                "activation_confirmation_pass": False,
                "stand_and_composite_stop_pass": True,
                "late_locomotion_after_abort": False,
                "nonzero_publish_calls_returned": 0,
                "activation": {
                    "input_source_registration": {
                        "accepted_by_service": True,
                        "task_completion_confirmed": False,
                        "task_state_value": 0,
                        "task_id": 1,
                        "selected_action": "ADD",
                    },
                    "set_mc_action_ld": {"accepted_by_service": True},
                    "observations": [observation],
                },
            },
        }
        result = assess_activation_probe_prerequisite(reset_run, zero_preflight, prior_replay)
        self.assertEqual(result["status"], "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED")
        self.assertEqual(result["prior_nonzero_publish_count"], 14)
        self.assertEqual(result["activation_probe_forward_velocity_m_s"], 0.12)
        self.assertEqual(result["activation_probe_max_duration_s"], 1.8)

        unsafe = json.loads(json.dumps(zero_preflight))
        unsafe["zero_only_activation_verification"]["stand_and_composite_stop_pass"] = False
        with self.assertRaisesRegex(ValueError, "physical-stop PASS"):
            assess_activation_probe_prerequisite(reset_run, unsafe, prior_replay)

        no_source_evidence = json.loads(json.dumps(prior_replay))
        no_source_evidence["installed_runtime_nonzero_activation_evidence"]["selected_source_get_service"] = ""
        with self.assertRaisesRegex(ValueError, "preserved runtime trace"):
            assess_activation_probe_prerequisite(reset_run, zero_preflight, no_source_evidence)

    def test_activation_probe_result_separates_mode_confirmation_from_stop_failure(self) -> None:
        blocked = {
            "status": "BLOCKED",
            "mode_source_activation_pass": True,
            "safe_stand_and_physical_stop_pass": False,
            "late_locomotion_after_abort": False,
        }
        self.assertEqual(
            classify_activation_probe_outcome(blocked),
            "SIMULATION_ONLY_ACTIVATION_ABORT_PHYSICAL_STOP_FAIL",
        )
        passed = {
            "status": "PASS",
            "mode_source_activation_pass": True,
            "safe_stand_and_physical_stop_pass": True,
            "late_locomotion_after_abort": False,
        }
        self.assertEqual(
            classify_activation_probe_outcome(passed),
            "SIMULATION_ONLY_ACTIVATION_ABORT_AND_PHYSICAL_STOP_PASS",
        )
        passed_with_late_transition = dict(passed, late_locomotion_after_abort=True)
        self.assertEqual(
            classify_activation_probe_outcome(passed_with_late_transition),
            "SIMULATION_ONLY_ACTIVATION_ABORT_LATE_LOCOMOTION",
        )
        self.assertEqual(
            classify_activation_probe_outcome(None),
            "SIMULATION_ONLY_ACTIVATION_PROBE_EVIDENCE_MISSING",
        )

    def test_activation_probe_child_command_is_separate_and_uses_fixed_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.2,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="activation-probe-cli-test",
            maintainer_authorized_first_motion=False,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=True,
            activation_probe_prerequisite={
                "status": "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED",
                "zero_only_preflight_attempt_id": "zero-preflight-id",
            },
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "activation-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--maintainer-authorized-activation-probe", child)
        self.assertIn("--activation-probe-qualification", child)
        self.assertIn("--activation-probe-prerequisite-attempt-id", child)
        self.assertEqual(child[child.index("--activation-probe-prerequisite-attempt-id") + 1], "zero-preflight-id")
        self.assertIn("--activation-probe-prerequisite-path", child)
        self.assertEqual(
            child[child.index("--activation-probe-prerequisite-path") + 1],
            "/workspace/evidence/activation_probe_prerequisite.json",
        )
        self.assertEqual(child[child.index("--forward-speed") + 1], "0.12")
        self.assertEqual(child[child.index("--pulse-duration") + 1], "1.8")
        self.assertEqual(child[child.index("--extra-stop-observation") + 1], "5.0")
        self.assertEqual(child[child.index("--max-observed-planar-speed-m-s") + 1], "0.2")
        self.assertIn("--zero-only-stop", child)
        self.assertNotIn("--route-integration", child)
        self.assertNotIn("--maintainer-authorized", child)

    def test_zero_only_activation_command_has_separate_authority_and_no_motion_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.2,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="zero-only-activation-cli-test",
            maintainer_authorized_first_motion=False,
            maintainer_authorized_zero_only_activation=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "activation-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--maintainer-authorized-zero-only-activation", child)
        self.assertIn("--zero-only-activation-verification", child)
        self.assertIn("--reset-gate-verified", child)
        self.assertIn("--zero-only-stop", child)
        self.assertIn("--max-observed-planar-speed-m-s", child)
        self.assertEqual(child[child.index("--max-observed-planar-speed-m-s") + 1], "0.2")
        self.assertIn("--zero-until-stand-timeout-s", child)
        self.assertNotIn("--maintainer-authorized", child)
        self.assertNotIn("--route-curved-path-diagnostic", child)
        self.assertNotIn("--route-integration", child)

    def test_parent_child_curved_path_profile_is_explicit_and_keeps_stop_gates(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=5.0,
            pulse_duration=1.0,
            forward_speed=0.2,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="curved-path-cli-test",
            run65_ground_truth_point_goal=False,
            run65_curved_path_diagnostic=True,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "curve-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", child)
        self.assertIn("--route-curved-path-diagnostic", child)
        self.assertIn("--zero-only-stop", child)
        self.assertIn("--max-observed-planar-speed-m-s", child)
        self.assertIn("--zero-until-stand-timeout-s", child)
        self.assertNotIn("--route-waypoint-b-only", child)
        self.assertNotIn("--route-turn-qualification-evidence", child)
        self.assertEqual(child[child.index("--extra-stop-observation") + 1], "5.0")

    def test_parent_child_local01_short_scene_route_propagates_bounded_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="local01-short-route-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            route_config=None,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "local01-route-test")[-1].rsplit(" && ", 1)[1]
        )

        def option(name: str) -> str:
            return child[child.index(name) + 1]

        self.assertIn("--maintainer-authorized", child)
        self.assertIn("--route-integration", child)
        self.assertIn("--route-config", child)
        self.assertEqual(option("--route-config"), "/workspace/evidence/route_config.json")
        self.assertIn("--short-scene-route", child)
        self.assertIn("--route-config-post-stand-observation", child)
        self.assertEqual(option("--extra-stop-observation"), "7.0")
        self.assertEqual(option("--forward-speed"), "0.12")
        self.assertEqual(option("--pulse-duration"), "1.0")
        self.assertEqual(option("--max-observed-planar-speed-m-s"), "0.2")
        self.assertEqual(option("--zero-until-stand-timeout-s"), "3.5")
        self.assertIn("--zero-only-stop", child)
        self.assertNotIn("--route-curved-path-diagnostic", child)
        self.assertNotIn("--route-waypoint-b-only", child)

    def test_parent_child_low_cruise_short_route_propagates_explicit_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="local01-low-cruise-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            route_config=None,
            raicom_low_cruise_profile=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "local01-low-cruise-route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertEqual(child[child.index("--forward-speed") + 1], "0.105")
        self.assertIn("--short-scene-route", child)
        self.assertIn("--raicom-low-cruise-profile", child)

    def test_parent_child_table_approach_profile_never_requests_table_docking(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="table-approach-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/table-approach.json",
            route_config=None,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(build_m1_child_command(args, "table-approach-test")[-1].rsplit(" && ", 1)[1])
        self.assertIn("--route-integration", child)
        self.assertIn("--table-approach-mission", child)
        self.assertIn("--route-config", child)
        self.assertNotIn("--short-scene-route", child)
        self.assertNotIn("--route-waypoint-b-only", child)
        self.assertEqual(child[child.index("--route-config") + 1], "/workspace/evidence/route_config.json")
        self.assertEqual(child[child.index("--extra-stop-observation") + 1], "7.0")

    def test_parent_child_aligned_start_table_approach_propagates_explicit_derived_scene_flag(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-table-approach-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            route_config=None,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-table-approach-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", child)
        self.assertIn("--table-approach-mission", child)
        self.assertIn("--aligned-start-table-approach", child)
        self.assertIn("--zero-only-stop", child)
        self.assertEqual(child[child.index("--route-config") + 1], "/workspace/evidence/route_config.json")

    def test_parent_child_aligned_start_table_low_cruise_propagates_explicit_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-table-low-cruise-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            route_config=None,
            raicom_low_cruise_profile=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-table-low-cruise-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertEqual(child[child.index("--forward-speed") + 1], "0.105")
        self.assertIn("--table-approach-mission", child)
        self.assertIn("--aligned-start-table-approach", child)
        self.assertIn("--raicom-low-cruise-profile", child)

    def test_parent_child_native_x2_timestep_is_scoped_to_aligned_table_route(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-native-timestep-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-native-timestep-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--aligned-start-table-approach", child)
        self.assertIn("--native-x2-timestep", child)
        self.assertIn("--raicom-low-cruise-profile", child)

    def test_parent_child_zero_forward_heading_acquisition_is_explicit(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-heading-source-acquisition-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=True,
            raicom_zero_forward_heading_acquisition=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-heading-source-acquisition-test")[-1]
            .rsplit(" && ", 1)[1]
        )
        self.assertIn("--raicom-zero-forward-heading-acquisition", child)
        self.assertIn("--native-x2-timestep", child)
        self.assertIn("--raicom-low-cruise-profile", child)

    def test_parent_child_continuous_ten_second_transit_is_explicit(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-continuous-ten-second-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=True,
            raicom_zero_forward_heading_acquisition=True,
            raicom_continuous_ten_second_transit=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-continuous-ten-second-test")[-1]
            .rsplit(" && ", 1)[1]
        )
        self.assertIn("--raicom-continuous-ten-second-transit", child)
        self.assertIn("--raicom-zero-forward-heading-acquisition", child)

    def test_parent_child_forward_gait_yaw_inversion_is_explicit(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.105,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-forward-yaw-inversion-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=True,
            raicom_zero_forward_heading_acquisition=True,
            raicom_continuous_ten_second_transit=True,
            raicom_forward_gait_yaw_inversion=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-forward-yaw-inversion-test")[-1]
            .rsplit(" && ", 1)[1]
        )
        self.assertIn("--raicom-forward-gait-yaw-inversion", child)
        self.assertIn("--raicom-continuous-ten-second-transit", child)

    def test_parent_child_native_nominal_straight_gait_diagnostic_is_explicit(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-native-nominal-straight-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=False,
            raicom_zero_forward_heading_acquisition=False,
            raicom_continuous_ten_second_transit=False,
            raicom_forward_gait_yaw_inversion=False,
            raicom_native_nominal_straight_gait_diagnostic=True,
            raicom_native_mjdata_route_feedback=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-native-nominal-straight-test")[-1]
            .rsplit(" && ", 1)[1]
        )
        self.assertIn("--raicom-native-nominal-straight-gait-diagnostic", child)
        self.assertIn("--raicom-native-mjdata-route-feedback", child)
        self.assertEqual(child[child.index("--forward-speed") + 1], "0.12")
        self.assertNotIn("--raicom-low-cruise-profile", child)

    def test_parent_child_fixed_window_table_profile_is_explicit(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=0.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="aligned-start-fixed-window-table-cli-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/aligned-start-table-approach.json",
            aligned_start_table_approach=True,
            native_x2_timestep=True,
            route_config=None,
            raicom_low_cruise_profile=False,
            raicom_zero_forward_heading_acquisition=False,
            raicom_continuous_ten_second_transit=False,
            raicom_continuous_table_approach=False,
            raicom_forward_gait_yaw_inversion=False,
            raicom_native_nominal_straight_gait_diagnostic=False,
            raicom_native_mjdata_route_feedback=True,
            raicom_fixed_window_table_approach=True,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
        )
        child = shlex.split(
            build_m1_child_command(args, "aligned-start-fixed-window-table-test")[-1]
            .rsplit(" && ", 1)[1]
        )
        self.assertIn("--raicom-fixed-window-table-approach", child)
        self.assertIn("--raicom-native-mjdata-route-feedback", child)
        self.assertNotIn("--raicom-continuous-table-approach", child)

    def test_route_zero_stop_precedes_zero_forward_heading_acquisition(self) -> None:
        """A stop callback must never be reinterpreted as a yaw command."""
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        begin = source.index("def publish_window(forward_m_s: float) -> None:")
        end = source.index("last_result = run_first_motion_window(", begin)
        callback = source[begin:end]
        zero_guard = callback.index("if forward <= 0.0:")
        acquisition = callback.index("heading_acquisition_this_window")
        self.assertLess(zero_guard, acquisition)
        self.assertIn("self._publish(0.0, 0.0, 0.0)", callback[:acquisition])

    def test_fixed_window_yaw_acquisition_is_scoped_to_open_waypoint_segment(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("fixed_window_heading_acquisition_required(", source)
        self.assertIn("segment_id=segment_id", source)
        self.assertIn("segment_id=configured_step.waypoint_id", source)
        self.assertIn('segment_id="WAYPOINT"', source)
        self.assertIn('segment_id == "WAYPOINT"', source)
        self.assertIn("and window_index == 0", source)

    def test_transit_termination_reads_explicit_window_heading_state(self) -> None:
        """The route callback must not close over a per-window local name."""
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        start = source.index("            def transit_termination(")
        end = source.index("            def run_transit_windows(", start)
        termination = source[start:end]
        self.assertIn("heading_acquisition_enabled = bool(", termination)
        self.assertIn(
            'window_control.get("zero_forward_heading_acquisition_enabled", False)',
            termination,
        )
        self.assertNotIn("heading_acquisition_this_window", termination)
        route = source[start:]
        self.assertIn(
            'window_control["zero_forward_heading_acquisition_enabled"] = (',
            route,
        )

    def test_odom_source_stamp_fence_prevents_late_packet_pose_replacement(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        start = source.index("    def _store(self, key: str, msg: Any) -> None:")
        end = source.index("    def _store_mc_debug_f64", start)
        callback = source[start:end]
        self.assertIn("self.odom_source_stamp_fence.observe(", callback)
        self.assertIn('"DROP_OUT_OF_ORDER"', callback)
        self.assertLess(
            callback.index('"DROP_FUTURE_LEAP",'),
            callback.index("self.latest[key] = msg"),
        )

    def test_high_rate_raw_debug_subscription_requires_explicit_capture_request(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        enabled = source.index("self.mc_debug_f64_subscription_enabled = bool(")
        subscription = source.index("Float64MultiArray, MC_DEBUG_F64_TOPIC")
        self.assertLess(enabled, subscription)
        guarded = source[enabled:subscription]
        self.assertIn('getattr(args, "capture_mc_rl_debug", False)', guarded)
        self.assertIn("if self.mc_debug_f64_subscription_enabled:", source[enabled:subscription + 160])

    def test_high_rate_odom_subscription_uses_latest_only_compatible_qos(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        start = source.index("self.odom_state_qos = QoSProfile(")
        end = source.index("self.command_qos = QoSProfile(", start)
        odom_qos = source[start:end]
        self.assertIn("history=HistoryPolicy.KEEP_LAST, depth=1", odom_qos)
        self.assertIn("reliability=ReliabilityPolicy.BEST_EFFORT", odom_qos)
        subscription = source.index("Odometry, \"/aima/hal/odom/state\"")
        self.assertIn("self.odom_state_qos", source[subscription:subscription + 160])

    def test_native_mjdata_route_feedback_stays_read_only_and_replaces_only_route_pose(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("NativeMjDataLivePoseReader(", source)
        self.assertIn("def route_feedback_snapshot(", source)
        self.assertIn("sample=route_sample", source)
        self.assertIn('"native_capture_is_read_only": True', source)
        self.assertNotIn("mj_step(", source)
        self.assertNotIn("mjData.qpos", source)

    def test_continuous_table_route_uses_one_gait_stream_and_a_rolling_intermediate(self) -> None:
        probe = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        parent = Path(__file__).with_name("issue80_m1_auto_reset_run.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--raicom-continuous-table-approach", parent)
        self.assertIn("rolling_intermediate_waypoint=continuous_table_approach_mode", probe)
        self.assertIn("continuous_route_authorized=continuous_table_approach_mode", probe)
        self.assertIn("TABLE_APPROACH_CONTINUOUS_ROUTE_MAX_NONZERO_S = 75.0", probe)
        self.assertIn("ROLLING_WAYPOINT_PASSED", probe)

    def test_transit_baseline_precedes_all_route_mode_dispatch(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        baseline = source.index("baseline_displacements = [")
        shared_executor = source.index("def run_transit_windows(")
        straight_dispatch = source.index("if native_nominal_straight_gait_diagnostic_mode:")
        self.assertLess(baseline, shared_executor)
        self.assertLess(baseline, straight_dispatch)

    def test_synchronized_initialization_observer_requires_keyframe_and_release_barriers_not_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="sync-init-test-container",
                attempt_id="sync-init-test",
                max_runtime_s=30.0,
                synchronized_paused_initialization=True,
                native_mjdata_capture=True,
            ))
            calls: list[tuple[str, str]] = []

            def fake_docker_exec(name: str, command: str, **_kwargs):
                calls.append((name, command))
                if name == "start_prearmed_standing_takeover_observer":
                    (Path(temp_dir) / "reset_watch_ready.json").write_text(json.dumps({
                        "status": "READY",
                        "fresh_topics": ["odom", "imu", "common", "leg_state", "leg_command"],
                        "set_mc_action_service_ready": True,
                        "get_mc_action_service_ready": True,
                        "pre_reset_get_mc_action": {
                            "action_desc": "PASSIVE_DEFAULT", "action_status_value": 100,
                        },
                    }), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            runner.docker_exec = fake_docker_exec
            runner.start_observer()

        self.assertEqual(len(calls), 1)
        command = calls[0][1]
        self.assertIn("--synchronized-keyframe-event-file /workspace/evidence/keyframe_load_event.json", command)
        self.assertIn("--synchronized-keyframe-spec /workspace/evidence/local01_scene/synchronized_nominal_keyframe.json", command)
        self.assertIn("--physics-release-event-file /workspace/evidence/physics_release_event.json", command)
        self.assertIn("--synchronized-ready-file /workspace/evidence/synchronized_jd_ready.json", command)
        self.assertNotIn("--reset-event-file", command)

    def test_static_nominal_start_observer_uses_no_reset_or_native_generation_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=temp_dir,
                container="static-init-test-container",
                attempt_id="static-init-test",
                max_runtime_s=30.0,
                static_nominal_start_initialization=True,
                native_mjdata_capture=True,
            ))
            calls: list[tuple[str, str]] = []

            def fake_docker_exec(name: str, command: str, **_kwargs):
                calls.append((name, command))
                if name == "start_prearmed_standing_takeover_observer":
                    (Path(temp_dir) / "reset_watch_ready.json").write_text(json.dumps({
                        "status": "READY",
                        "fresh_topics": ["odom", "imu", "common", "leg_state", "leg_command"],
                        "set_mc_action_service_ready": True,
                        "get_mc_action_service_ready": True,
                        "pre_reset_get_mc_action": {
                            "action_desc": "PASSIVE_DEFAULT", "action_status_value": 100,
                        },
                    }), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            runner.docker_exec = fake_docker_exec
            runner.start_observer()

        command = calls[0][1]
        self.assertIn("--static-nominal-start-event-file /workspace/evidence/static_nominal_start_event.json", command)
        self.assertIn("--native-mjdata-trace /workspace/evidence/native_mjdata_trace.jsonl", command)
        self.assertNotIn("--require-native-reset-generation", command)
        self.assertNotIn("--reset-event-file", command)

    def test_static_nominal_start_motion_child_uses_generic_initialization_gate_not_reset_label(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=7.0,
            pulse_duration=1.0,
            forward_speed=0.12,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="static-nominal-motion-test",
            maintainer_authorized_first_motion=True,
            maintainer_authorized_zero_only_activation=False,
            maintainer_authorized_activation_probe=False,
            local01_scene_world_poses="/committed/local01/scene_world_poses.json",
            table_approach_mission="/review/static-nominal-table-approach.json",
            route_config=None,
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
            static_nominal_start_initialization=True,
        )
        child = shlex.split(
            build_m1_child_command(args, "static-nominal-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--initialization-gate-verified", child)
        self.assertNotIn("--reset-gate-verified", child)

    def test_route_windows_use_source_grounded_nonzero_activation_not_zero_only(self) -> None:
        probe = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text(
            encoding="utf-8"
        )
        route_start = probe.index("    def run_short_route_integration(")
        route_end = probe.index("    def execute(", route_start)
        route = probe[route_start:route_end]
        self.assertNotIn("begin_zero_only_locomotion_activation(", route)
        self.assertIn("begin_source_grounded_route_activation(", route)
        self.assertIn("source_grounded_route_activation=True", route)
        self.assertLess(
            route.index("route_activation_pending"),
            route.index("active_window_gate.observe("),
        )

    def test_parent_child_point_goal_invocation_propagates_runtime_profile(self) -> None:
        args = argparse.Namespace(
            extra_stop_observation=7.0,
            pulse_duration=10.0,
            forward_speed=0.14,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="offline-validation-attempt",
            run65_ground_truth_point_goal=True,
            heading_change_rad=-0.10,
            startup_speed=None,
            run41_min_run_diagnostic=False,
            run42_stop_confirmation_diagnostic=False,
            run43_zero_only_stop_diagnostic=False,
            run44_threshold_margin_diagnostic=False,
            run45_ten_second_gait_diagnostic=False,
            run46_startup_cruise_threshold_diagnostic=False,
            run48_low_cruise_baseline_diagnostic=False,
            run49_smoothstop_diagnostic=False,
            run50_threshold_respecting_smoothstop_diagnostic=False,
            run53_turning_diagnostic=False,
        )
        command = build_m1_child_command(args, "validation-container")
        self.assertEqual(command[:5], [
            "docker", "exec", "-e", "ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", "validation-container",
        ])
        shell = command[-1]
        child_argv = shlex.split(shell.rsplit(" && ", 1)[1])
        self.assertEqual(child_argv[0], "python3")
        self.assertEqual(child_argv[2], "first-motion")

        def option(name: str) -> str:
            return child_argv[child_argv.index(name) + 1]

        self.assertEqual(option("--extra-stop-observation"), "7.0")
        self.assertEqual(option("--heading-change-rad"), "-0.1")
        self.assertEqual(option("--final-yaw-tolerance-rad"), "0.17")
        self.assertEqual(option("--heading-control-policy"), "coordinated-pose")
        self.assertEqual(option("--heading-goal-stop-advance-m"), "0.025")
        self.assertEqual(option("--heading-goal-predicted-stop-tolerance-m"), "0.005")
        self.assertNotIn("--heading-yaw-rate-prediction-horizon-s", child_argv)
        self.assertIn("--point-goal-diagnostic", child_argv)
        self.assertIn("--zero-only-stop", child_argv)
        self.assertEqual(option("--max-observed-planar-speed-m-s"), "0.2")
        self.assertEqual(option("--zero-until-stand-timeout-s"), "3.5")

        args.heading_change_rad = 0.0
        args.extra_stop_observation = 5.0
        straight_shell = build_m1_child_command(args, "validation-container")[-1]
        straight_argv = shlex.split(straight_shell.rsplit(" && ", 1)[1])
        self.assertEqual(straight_argv[straight_argv.index("--heading-control-policy") + 1], "position-bearing")
        self.assertEqual(straight_argv[straight_argv.index("--extra-stop-observation") + 1], "5.0")
        self.assertNotIn("--heading-goal-stop-advance-m", straight_argv)

    def test_turning_probe_registers_one_turning_diagnostic_option(self):
        probe = Path(__file__).with_name("issue80_m1_locomotion_probe.py")
        tree = ast.parse(probe.read_text(encoding="utf-8"), filename=str(probe))
        registrations = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--turning-diagnostic"
        ]

        self.assertEqual(len(registrations), 1)

    def test_right_turn_and_route_modes_use_the_existing_m1_child_path(self):
        args = argparse.Namespace(
            extra_stop_observation=5.0,
            pulse_duration=1.0,
            forward_speed=0.2,
            ramp_start_speed=None,
            ramp_duration_s=0.0,
            attempt_id="offline-route-mode",
            run65_ground_truth_point_goal=False,
            heading_change_rad=0.0,
            startup_speed=None,
            startup_transition_displacement=None,
            startup_transition_timeout_s=0.0,
            run65_right_turn_qualification=True,
            run65_route_integration=False,
            run65_waypoint_b_only=False,
            run65_waypoint_b_short_diagnostic=False,
            run65_waypoint_b_extended_diagnostic=False,
        )
        turn_command = shlex.split(build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1])
        self.assertIn("--right-turn-qualification", turn_command)
        self.assertIn("--zero-only-stop", turn_command)
        self.assertIn("--max-observed-planar-speed-m-s", turn_command)
        self.assertNotIn("--point-goal-diagnostic", turn_command)

        args.run65_right_turn_qualification = False
        args.run65_route_integration = True
        route_command = shlex.split(build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1])
        self.assertIn("--route-integration", route_command)
        evidence_index = route_command.index("--route-turn-qualification-evidence")
        self.assertEqual(route_command[evidence_index + 1], "/workspace/evidence/route_turn_qualification_evidence.json")
        self.assertIn("--zero-only-stop", route_command)
        self.assertNotIn("--right-turn-qualification", route_command)

        args.run65_route_integration = False
        args.run65_waypoint_b_only = True
        b_only_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", b_only_command)
        self.assertIn("--route-waypoint-b-only", b_only_command)
        self.assertIn("--zero-only-stop", b_only_command)
        self.assertIn("--max-observed-planar-speed-m-s", b_only_command)
        self.assertNotIn("--route-turn-qualification-evidence", b_only_command)
        self.assertNotIn("--ramp-start-speed", b_only_command)
        self.assertNotIn("--ramp-duration-s", b_only_command)

        args.run65_waypoint_b_only = False
        args.run65_waypoint_b_short_diagnostic = True
        short_diagnostic_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", short_diagnostic_command)
        self.assertIn("--route-waypoint-b-only", short_diagnostic_command)
        self.assertIn("--route-waypoint-b-short-diagnostic", short_diagnostic_command)
        self.assertIn("--zero-only-stop", short_diagnostic_command)
        self.assertIn("--max-observed-planar-speed-m-s", short_diagnostic_command)
        self.assertNotIn("--route-turn-qualification-evidence", short_diagnostic_command)
        self.assertNotIn("--ramp-start-speed", short_diagnostic_command)
        self.assertNotIn("--ramp-duration-s", short_diagnostic_command)

        def option(command: list[str], name: str) -> str:
            return command[command.index(name) + 1]

        self.assertEqual(option(short_diagnostic_command, "--extra-stop-observation"), "5.0")
        self.assertEqual(option(short_diagnostic_command, "--pulse-duration"), "1.0")
        self.assertEqual(option(short_diagnostic_command, "--forward-speed"), "0.2")
        self.assertEqual(option(short_diagnostic_command, "--zero-until-stand-timeout-s"), "3.5")

        args.run65_waypoint_b_short_diagnostic = False
        args.run65_waypoint_b_extended_diagnostic = True
        extended_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", extended_command)
        self.assertIn("--route-waypoint-b-only", extended_command)
        self.assertIn("--route-waypoint-b-extended-diagnostic", extended_command)
        self.assertNotIn("--route-waypoint-b-short-diagnostic", extended_command)
        self.assertIn("--zero-only-stop", extended_command)
        self.assertEqual(option(extended_command, "--extra-stop-observation"), "5.0")
        self.assertEqual(option(extended_command, "--forward-speed"), "0.2")

        args.extra_stop_observation = 7.0
        args.post_locomotion_convergence_observation_s = 7.0
        convergence_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-waypoint-b-post-locomotion-convergence", convergence_command)
        self.assertEqual(option(convergence_command, "--extra-stop-observation"), "7.0")

        args.run65_waypoint_b_extended_diagnostic = False
        args.extra_stop_observation = 5.0
        args.post_locomotion_convergence_observation_s = None
        args.route_config = "/offline/route_config.json"
        configured_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", configured_command)
        self.assertIn("--route-config-post-stand-observation", configured_command)
        route_index = configured_command.index("--route-config")
        self.assertEqual(configured_command[route_index + 1], "/workspace/evidence/route_config.json")
        self.assertEqual(option(configured_command, "--extra-stop-observation"), "7.0")
        self.assertIn("--zero-only-stop", configured_command)

        args.route_config = None
        args.local01_scene_world_poses = "/pinned/local01/scene_world_poses.json"
        local01_command = shlex.split(
            build_m1_child_command(args, "route-test")[-1].rsplit(" && ", 1)[1]
        )
        self.assertIn("--route-integration", local01_command)
        self.assertIn("--short-scene-route", local01_command)
        self.assertIn("--route-config-post-stand-observation", local01_command)
        route_index = local01_command.index("--route-config")
        self.assertEqual(local01_command[route_index + 1], "/workspace/evidence/route_config.json")
        self.assertEqual(option(local01_command, "--extra-stop-observation"), "7.0")
        self.assertIn("--max-observed-planar-speed-m-s", local01_command)
        self.assertIn("--zero-until-stand-timeout-s", local01_command)
        self.assertIn("--zero-only-stop", local01_command)

    def test_bounded_run_stages_validated_route_config_immutably(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            source = root / "local01-route.json"
            route_config = {
                "schema_version": "robotsim.issue80.waypoint-docking-poses.v1",
                "route_id": "local01-waypoint-docking-fixture",
                "frame_id": "odom",
                "start": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                "waypoint": {"x_m": 0.25, "y_m": 0.0, "yaw_rad": 0.0},
                "table_docking": {"x_m": 0.6, "y_m": 0.1, "yaw_rad": 0.1},
            }
            payload = json.dumps(route_config, indent=2) + "\n"
            source.write_text(payload, encoding="utf-8")
            args = argparse.Namespace(
                evidence_dir=str(evidence),
                route_config=str(source),
                container="offline-route-test",
                max_runtime_s=180.0,
            )
            run = BoundedRun(args)
            staged = evidence / "route_config.json"
            manifest = evidence / "route_config_input.json"
            self.assertEqual(staged.read_text(encoding="utf-8"), payload)
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(record["classification"], "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE")
            self.assertEqual(record["source_sha256"], hashlib.sha256(payload.encode()).hexdigest())

    def test_waypoint_docking_route_is_checked_against_the_pinned_scene_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "route.json"
            source.write_text(
                json.dumps({
                    "schema_version": "robotsim.issue80.waypoint-docking-poses.v1",
                    "route_id": "local01-waypoint-docking-fixture",
                    "frame_id": "odom",
                    "start": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                    "waypoint": {"x_m": 0.25, "y_m": 0.0, "yaw_rad": 0.0},
                    "table_docking": {"x_m": 0.6, "y_m": 0.1, "yaw_rad": 0.1},
                }),
                encoding="utf-8",
            )
            expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            workcell = {
                "status": "PASS",
                "robot_collision_footprint_radial_upper_bound_m": 0.41565748847422734,
                "verified_local_clear_radius_m": 2.0,
            }
            with patch("issue80_m1_auto_reset_run.inspect_route_workcell_geometry", return_value=workcell):
                result = validate_waypoint_docking_route_input(source)
        self.assertEqual(result["route"].route_id, "local01-waypoint-docking-fixture")
        self.assertEqual(result["workcell"]["status"], "PASS")
        self.assertEqual(result["clearance"]["status"], "PASS")
        self.assertEqual(result["source_sha256"], expected_sha)

    def test_bounded_run_stages_local01_scene_and_short_route_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence"
            evidence.mkdir()
            pose_source = root / "scene_world_poses.json"
            pose_source.write_text("{}\n", encoding="utf-8")
            source_sha = hashlib.sha256(pose_source.read_bytes()).hexdigest()
            source = {
                "source_commit": "95d76d82ea7def0dd719da4ebe07f9654044415e",
                "source_hashes": {"scene_world_poses": source_sha},
            }
            route_input = build_short_route_config({
                **source,
                "scene_world_poses": {"poses": {
                    "START": {"xyz_m": [-1.5, -1.5, 0.68], "yaw_rad": 1.5707963267948966},
                    "TABLE_DOCKING": {"xyz_m": [1.0, -1.5, 0.68], "yaw_rad": 0.0},
                }},
                "odom_from_scene": {
                    "translation_xyz_m": [1.5, -1.5, 0.0], "yaw_rad": -1.5707963267948966,
                },
                "reset_anchor": {"base_pose_in_odom_after_reset": {
                    "xyz_m": [0.0, 0.0, 0.68], "yaw_rad": 0.0,
                }},
            })
            preflight = {
                "source": source,
                "route_input": route_input,
                "route": None,
                "source_sha256": source_sha,
                "static_scene_preflight": {"status": "PASS_STATIC_SCENE_CLEARANCE", "mj_step_calls": 0},
            }
            bundle = {
                "classification": "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER",
                "source_local01_commit": source["source_commit"],
            }
            args = argparse.Namespace(
                evidence_dir=str(evidence),
                route_config=None,
                local01_scene_world_poses=str(pose_source),
                configured_local01_scene_preflight=preflight,
                container="local01-route-test",
                max_runtime_s=180.0,
            )
            with patch("issue80_m1_auto_reset_run.load_local01_scene_inputs", return_value=source), patch(
                "issue80_m1_auto_reset_run.prepare_local01_scene_runtime_bundle", return_value=bundle
            ):
                run = BoundedRun(args)
            staged = evidence / "route_config.json"
            manifest = json.loads((evidence / "route_config_input.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(staged.read_text(encoding="utf-8"))["route_kind"],
                             "SIMULATION_ONLY_SHORT_ROUTE_DIAGNOSTIC_NOT_TABLE_DOCKING")
            self.assertEqual(manifest["source_local01_commit"], source["source_commit"])
            self.assertEqual(manifest["source_scene_world_poses_sha256"], source_sha)
            self.assertEqual(manifest["derived_route_sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())
            self.assertEqual(run.route_config_sha256, hashlib.sha256(staged.read_bytes()).hexdigest())

    def test_route_workcell_preflight_accepts_only_pinned_infinite_floor_scene(self):
        from issue80_m1_auto_reset_run import ROUTE_SCENE_RELATIVE, ROUTE_X2_RELATIVE

        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            scene = runtime / ROUTE_SCENE_RELATIVE
            x2 = runtime / ROUTE_X2_RELATIVE
            scene.parent.mkdir(parents=True)
            x2.write_text("<mujoco model='fixture'/>", encoding="utf-8")
            scene.write_text(
                "<mujoco model='fixture'><include file='x2.xml'/><worldbody>"
                "<geom name='floor' type='plane' size='0 0 0.05'/>"
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            with patch("issue80_m1_auto_reset_run.ROUTE_SCENE_SHA256", hashlib.sha256(scene.read_bytes()).hexdigest()), \
                 patch("issue80_m1_auto_reset_run.ROUTE_X2_SHA256", hashlib.sha256(x2.read_bytes()).hexdigest()):
                result = inspect_route_workcell_geometry(runtime)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["active_world_obstacles"], [])

            scene.write_text(
                "<mujoco model='fixture'><include file='x2.xml'/><worldbody>"
                "<geom name='floor' type='plane' size='0 0 0.05'/>"
                "<geom name='wall' type='box' size='1 1 1'/>"
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            with patch("issue80_m1_auto_reset_run.ROUTE_SCENE_SHA256", hashlib.sha256(scene.read_bytes()).hexdigest()), \
                 patch("issue80_m1_auto_reset_run.ROUTE_X2_SHA256", hashlib.sha256(x2.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "unreviewed world collision geom"):
                    inspect_route_workcell_geometry(runtime)

    def test_turning_result_stays_simulation_only_and_does_not_claim_manufacturer_acceptance(self):
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_DIAGNOSTIC_TURNING_PASS"),
            "SIMULATION_ONLY_DIAGNOSTIC_TURNING_PASS",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_DIAGNOSTIC_TURNING_BLOCKED"),
            "SIMULATION_ONLY_DIAGNOSTIC_TURNING_BLOCKED",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_PASS"),
            "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_PASS",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_BLOCKED"),
            "SIMULATION_ONLY_SHORT_WAYPOINT_DIAGNOSTIC_BLOCKED",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_PASS"),
            "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_PASS",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_BLOCKED"),
            "SIMULATION_ONLY_CONFIGURED_WAYPOINT_DOCKING_ROUTE_BLOCKED",
        )

    def make_runner(self, evidence: Path) -> BoundedRun:
        return BoundedRun(
            argparse.Namespace(
                evidence_dir=str(evidence),
                container="test-container",
                maintainer_authorized_first_motion=True,
                forward_speed=0.2, pulse_duration=1.0, extra_stop_observation=0.0,
                attempt_id="test-attempt",
                max_runtime_s=10.0,
            )
        )

    def make_debug_overlay_fixture(self, evidence: Path, runtime: Path) -> None:
        header_source = runtime / "aimdk_msgs_ws/src/aimdk_msgs/interface/robot/common/msg/MessageHeader.msg"
        header_source.parent.mkdir(parents=True)
        header_source.write_text("builtin_interfaces/Time stamp\nstring frame_id\nuint32 sequence\nbuiltin_interfaces/Time meas_stamp\n")
        overlay = evidence / "mc_debug_overlay/src/aimdk_msgs/msg"
        overlay.mkdir(parents=True)
        (overlay / "MessageHeader.msg").write_bytes(header_source.read_bytes())
        (overlay / "McRLDebug.msg").write_text(
            "MessageHeader header\nfloat32[] obs_vec\nfloat32[] action_vec\n"
        )
        (evidence / "mc_debug_overlay/install").mkdir(parents=True)
        (evidence / "mc_debug_overlay/install/setup.bash").write_text("# built overlay\n")
        mc_lib = runtime / "mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so"
        sim_lib = runtime / "sim_mujoco/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so"
        mc_lib.parent.mkdir(parents=True)
        sim_lib.parent.mkdir(parents=True)
        mc_lib.write_bytes(b"mc-typesupport")
        sim_lib.write_bytes(b"sim-typesupport")
        fields = [
            {"name": "header", "type": "aimdk_msgs/msg/MessageHeader"},
            {"name": "obs_vec", "type": "float32[]"},
            {"name": "action_vec", "type": "float32[]"},
        ]
        header_sha = hashlib.sha256(header_source.read_bytes()).hexdigest()
        mc_sha = hashlib.sha256(mc_lib.read_bytes()).hexdigest()
        sim_sha = hashlib.sha256(sim_lib.read_bytes()).hexdigest()
        provenance = {
            "classification": "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY",
            "type": "aimdk_msgs/msg/McRLDebug",
            "fields_in_order": fields,
            "message_header_source_sha256": header_sha,
            "mc_introspection_library_sha256": mc_sha,
            "sim_introspection_library_sha256": sim_sha,
            "field_metadata_matches_in_mc_and_sim": True,
        }
        (evidence / "MC_RL_DEBUG_SCHEMA_PROVENANCE.json").write_text(json.dumps(provenance))
        results = []
        for path, sha in ((mc_lib, mc_sha), (sim_lib, sim_sha)):
            results.append({
                "library": str(path), "sha256": sha, "namespace": "aimdk_msgs::msg",
                "message": "McRLDebug", "member_count": 3,
                "fields": [
                    {"name": "header", "type_id": 18, "is_array": False,
                     "nested": {"message": "MessageHeader"}},
                    {"name": "obs_vec", "type_id": 1, "is_array": True,
                     "array_size": 0, "is_upper_bound": False, "nested": None},
                    {"name": "action_vec", "type_id": 1, "is_array": True,
                     "array_size": 0, "is_upper_bound": False, "nested": None},
                ],
            })
        (evidence / "mc_rl_debug_schema.json").write_text(json.dumps({"results": results}))

    def test_mc_velocity_graph_discovery_is_bounded_and_polls_executor(self) -> None:
        source = Path(__file__).with_name("issue80_m1_locomotion_probe.py").read_text()
        self.assertIn("MC_VELOCITY_ENDPOINT_DISCOVERY_TIMEOUT_S = 5.0", source)
        self.assertIn("deadline = discovery_started + MC_VELOCITY_ENDPOINT_DISCOVERY_TIMEOUT_S", source)
        self.assertIn("rclpy.spin_once(self, timeout_sec=min(0.05", source)
        self.assertIn("expected_only_verified_mc_subscriber", Path(__file__).with_name("issue80_m1_first_motion.py").read_text())

    def test_parent_parser_supports_only_reviewed_ten_second_ceiling(self) -> None:
        source = Path(__file__).with_name("issue80_m1_auto_reset_run.py").read_text()
        self.assertIn('args.pulse_duration <= 10.0', source)
        self.assertIn('pulse duration must be positive and no greater than 10 seconds', source)

    def test_motion_is_reported_separately_from_unverified_stop(self) -> None:
        activation = {
            "nonzero_publish_calls_returned": 30,
            "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
            "status": "PHYSICAL_MOTION_CONFIRMED_POSTSTOP_UNVERIFIED",
            "post_stop_forward_tail": {"status": "ABOVE_BASELINE_ENVELOPE"},
            "legacy_baseline_stop_gate": {"status": "ABOVE_BASELINE_ENVELOPE"},
            "physical_stop_gate": {"status": "FAIL"},
        }
        self.assertEqual(
            classify_first_motion_outcome(activation),
            "FIRST_PHYSICAL_WALKING_MOTION_CONFIRMED_STOP_UNVERIFIED",
        )
        activation["physical_stop_gate"] = {"status": "PASS"}
        self.assertEqual(
            classify_first_motion_outcome(activation),
            "FIRST_PHYSICAL_WALKING_PASS",
            "the independent physics gate may pass while the preserved legacy baseline still fails",
        )
        self.assertEqual(activation["legacy_baseline_stop_gate"]["status"], "ABOVE_BASELINE_ENVELOPE")
        activation["physical_stop_gate"] = {"status": "FAIL"}
        activation["status"] = "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED"
        self.assertEqual(classify_first_motion_outcome(activation), "FIRST_PHYSICAL_WALKING_PASS")
        activation["nonzero_publish_calls_returned"] = 0
        self.assertEqual(classify_first_motion_outcome(activation), "VELOCITY_COMMAND_NOT_EXECUTED")

    def test_windows_activation_is_attempted_only_after_verified_x11_failure_sent_no_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            identity = {
                "window_id": 0x1234, "pid": 4321, "title": "MuJoCo : X2",
                "width": 800, "height": 600, "viewable": True,
            }
            first = {
                "window": identity,
                "event": {
                    "status": "RESET_NOT_DELIVERED",
                    "identity_before_dispatch": identity,
                    "xtest_key_events_sent": 0,
                },
            }
            calls = []
            runner.run = lambda name, command, **kwargs: (
                calls.append(name) or subprocess.CompletedProcess(command, 0, json.dumps(first), "")
            )
            focus_calls = []
            def windows_call(**kwargs):
                focus_calls.append(kwargs)
                if kwargs["mode"] == "Activate":
                    return {"status": "FOREGROUND_VERIFIED", "key_events_sent": 0}
                return {"status": "WIN32_KEY_EVENTS_INSERTED", "key_events_sent": 2}

            runner.activate_verified_windows_foreground = windows_call
            runner.inspect_x11_target_identity = lambda *args: {
                "status": "IDENTITY_VERIFIED", "window": identity,
            }

            runner.dispatch_reset(4321, "MuJoCo : X2", 0x1234)

            self.assertEqual(calls, ["x11_backspace_dispatch_attempt1"])
            self.assertEqual([call["mode"] for call in focus_calls], ["Activate", "SendReset"])
            self.assertEqual(runner.reset_dispatch_payload["event"]["status"], "KEY_DISPATCHED")
            self.assertEqual(runner.reset_dispatch_payload["event"]["xtest_key_events_sent"], 0)
            self.assertEqual(runner.reset_dispatch_payload["event"]["windows_key_events_inserted"], 2)
            self.assertEqual(runner.reset_dispatch_payload["reset_delivery_attempt_count"], 1)
            self.assertEqual(runner.reset_dispatch_payload["reset_input_path_attempt_count"], 2)
            self.assertEqual(json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())["event"]["status"], "KEY_DISPATCHED")

    def test_windows_focus_mapping_requires_observed_wslg_rail_identity(self):
        helper = Path(__file__).with_name("issue80_mujoco_windows_focus.ps1").read_text(encoding="utf-8")

        self.assertIn('$expectedWslgTitle = "[WARN:COPY MODE] $ExpectedTitle (Ubuntu-22.04)"', helper)
        self.assertIn('$_.class_name -ceq "RAIL_WINDOW"', helper)
        self.assertIn('$_.process_name -ieq "msrdc"', helper)
        self.assertIn("$windowMatches.Count -eq 1", helper)
        self.assertIn("$currentWindow.pid -eq $matchedWindow.pid", helper)
        self.assertIn("$currentChild.pid -eq $matchedWindow.pid", helper)
        self.assertIn("SetForegroundWindow", helper)
        self.assertIn('[ValidateSet("Inspect", "Activate", "SendReset")]', helper)
        self.assertIn("SendBackspaceToForeground", helper)
        self.assertIn("before != expectedForegroundHwnd", helper)
        self.assertIn("KEY_SEQUENCE_PARTIAL_OR_REJECTED", helper)

    def test_unverified_windows_foreground_result_sends_no_second_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            identity = {
                "window_id": 0x1234, "pid": 4321, "title": "MuJoCo : X2",
                "width": 800, "height": 600, "viewable": True,
            }
            first = {
                "window": identity,
                "event": {
                    "status": "RESET_NOT_DELIVERED",
                    "identity_before_dispatch": identity,
                    "xtest_key_events_sent": 0,
                },
            }
            calls = []
            runner.run = lambda name, command, **kwargs: (
                calls.append(name) or subprocess.CompletedProcess(command, 0, json.dumps(first), "")
            )
            runner.activate_verified_windows_foreground = lambda **kwargs: {
                "status": "WINDOW_MATCH_NOT_UNIQUE", "key_events_sent": 0,
            }
            runner.inspect_x11_target_identity = lambda *args: self.fail("must not recheck/send after unverified foreground")

            with self.assertRaises(RuntimeError):
                runner.dispatch_reset(4321, "MuJoCo : X2", 0x1234)

            self.assertEqual(calls, ["x11_backspace_dispatch_attempt1"])
            final = json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())
            self.assertEqual(final["event"]["xtest_key_events_sent"], 0)
            self.assertEqual(final["windows_foreground_activation"]["status"], "WINDOW_MATCH_NOT_UNIQUE")

    def test_windows_foreground_is_not_attempted_after_a_key_was_sent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            identity = {
                "window_id": 0x1234, "pid": 4321, "title": "MuJoCo : X2",
                "width": 800, "height": 600, "viewable": True,
            }
            first = {
                "window": identity,
                "event": {
                    "status": "RESET_NOT_DELIVERED",
                    "identity_before_dispatch": identity,
                    "xtest_key_events_sent": 2,
                },
            }
            calls = []
            runner.run = lambda name, command, **kwargs: (
                calls.append(name) or subprocess.CompletedProcess(command, 0, json.dumps(first), "")
            )
            runner.activate_verified_windows_foreground = lambda **kwargs: self.fail("must not re-focus after a key was sent")

            with self.assertRaises(RuntimeError):
                runner.dispatch_reset(4321, "MuJoCo : X2", 0x1234)

            self.assertEqual(calls, ["x11_backspace_dispatch_attempt1"])
            final = json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())
            self.assertEqual(final["event"]["xtest_key_events_sent"], 2)

    def test_windows_reset_dispatch_requires_fresh_exact_x11_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            identity = {
                "window_id": 0x1234, "pid": 4321, "title": "MuJoCo : X2",
                "width": 800, "height": 600, "viewable": True,
            }
            first = {
                "window": identity,
                "event": {
                    "status": "RESET_NOT_DELIVERED",
                    "identity_before_dispatch": identity,
                    "xtest_key_events_sent": 0,
                },
            }
            runner.run = lambda name, command, **kwargs: subprocess.CompletedProcess(
                command, 0, json.dumps(first), "",
            )
            modes = []
            def windows_call(**kwargs):
                modes.append(kwargs["mode"])
                return {
                    "status": "FOREGROUND_VERIFIED" if kwargs["mode"] == "Activate" else "WIN32_KEY_EVENTS_INSERTED",
                    "key_events_sent": 0 if kwargs["mode"] == "Activate" else 2,
                }

            runner.activate_verified_windows_foreground = windows_call
            runner.inspect_x11_target_identity = lambda *args: {
                "status": "IDENTITY_NOT_VERIFIED", "matching_windows": [],
            }

            with self.assertRaises(RuntimeError):
                runner.dispatch_reset(4321, "MuJoCo : X2", 0x1234)

            self.assertEqual(modes, ["Activate"])
            final = json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())
            self.assertEqual(final["event"]["xtest_key_events_sent"], 0)
            self.assertNotIn("windows_key_dispatch", final)

    def test_partial_windows_key_dispatch_is_recorded_without_retry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            identity = {
                "window_id": 0x1234, "pid": 4321, "title": "MuJoCo : X2",
                "width": 800, "height": 600, "viewable": True,
            }
            first = {
                "window": identity,
                "event": {
                    "status": "RESET_NOT_DELIVERED",
                    "identity_before_dispatch": identity,
                    "xtest_key_events_sent": 0,
                },
            }
            names = []
            runner.run = lambda name, command, **kwargs: (
                names.append(name) or subprocess.CompletedProcess(command, 0, json.dumps(first), "")
            )
            modes = []
            def windows_call(**kwargs):
                modes.append(kwargs["mode"])
                return {
                    "status": "FOREGROUND_VERIFIED" if kwargs["mode"] == "Activate" else "WIN32_KEY_DISPATCH_NOT_CONFIRMED",
                    "key_events_sent": 0 if kwargs["mode"] == "Activate" else 1,
                }

            runner.activate_verified_windows_foreground = windows_call
            runner.inspect_x11_target_identity = lambda *args: {
                "status": "IDENTITY_VERIFIED", "window": identity,
            }

            with self.assertRaises(RuntimeError):
                runner.dispatch_reset(4321, "MuJoCo : X2", 0x1234)

            self.assertEqual(names, ["x11_backspace_dispatch_attempt1"])
            self.assertEqual(modes, ["Activate", "SendReset"])
            final = json.loads((Path(temp_dir) / "reset_dispatch_event.json").read_text())
            self.assertEqual(final["event"]["xtest_key_events_sent"], 0)
            self.assertEqual(final["windows_key_dispatch"]["key_events_sent"], 1)
            self.assertEqual(final["classification"], "PARTIAL_WIN32_KEY_SEQUENCE_STOP_NO_RETRY")

    def test_source_joint_velocity_abort_overrides_unverified_stop_classification(self) -> None:
        activation = {
            "nonzero_publish_calls_returned": 54,
            "physical_motion_status": "PHYSICAL_MOTION_CONFIRMED",
            "status": "ABORTED",
            "fault": "source_joint_velocity_limit:left_knee_joint",
        }
        self.assertEqual(
            classify_first_motion_outcome(activation),
            "FIRST_PHYSICAL_WALKING_FAIL",
        )

    def test_stages_current_mode_logic_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / M1_MODE_LOGIC.name
            self.assertEqual(staged.read_bytes(), M1_MODE_LOGIC.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == M1_MODE_LOGIC.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())
            self.assertEqual(record["disposition"], "COPIED")

    def test_stages_single_action_capture_helper_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / SET_ACTION_CAPTURE.name
            self.assertEqual(staged.read_bytes(), SET_ACTION_CAPTURE.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == SET_ACTION_CAPTURE.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())

    def test_stages_first_motion_safety_logic_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / M1_FIRST_MOTION_LOGIC.name
            self.assertEqual(staged.read_bytes(), M1_FIRST_MOTION_LOGIC.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == M1_FIRST_MOTION_LOGIC.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())

    def test_stages_coordinated_pose_goal_controller_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / POSE_GOAL_CONTROLLER.name
            self.assertEqual(staged.read_bytes(), POSE_GOAL_CONTROLLER.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == POSE_GOAL_CONTROLLER.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())
            self.assertEqual(record["disposition"], "COPIED")

    def test_stages_mc_rl_debug_capture_sidecar_and_records_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            staged = evidence / MC_RL_DEBUG_CAPTURE.name
            self.assertEqual(staged.read_bytes(), MC_RL_DEBUG_CAPTURE.read_bytes())
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            record = next(row for row in manifest["scripts"] if row["name"] == MC_RL_DEBUG_CAPTURE.name)
            self.assertEqual(record["sha256"], hashlib.sha256(staged.read_bytes()).hexdigest())

    def test_mc_rl_debug_capture_is_read_only_and_starts_before_m1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir) / "evidence"
            runtime = Path(temp_dir) / "runtime"
            evidence.mkdir()
            self.make_debug_overlay_fixture(evidence, runtime)
            (evidence / "mc_rl_debug_ready.json").write_text(
                json.dumps({"status": "READY", "publisher_count": 1, "topic": "/aima/mc/rl/debug"})
            )
            runner = self.make_runner(evidence)
            calls = []

            def fake_docker_exec(name, shell, **kwargs):
                calls.append((name, shell, kwargs))
                return subprocess.CompletedProcess([], 0, "", "")

            runner.docker_exec = fake_docker_exec
            with patch("issue80_m1_auto_reset_run.RUNTIME", runtime):
                runner.start_mc_rl_debug_capture()

            self.assertEqual([call[0] for call in calls], [
                "start_mc_rl_debug_capture", "wait_mc_rl_debug_capture_ready"
            ])
            self.assertTrue(calls[0][2]["detached"])
            self.assertIn("issue80_mc_rl_debug_capture.py", calls[0][1])
            self.assertIn("mc_debug_overlay/install/setup.bash", calls[0][1])
            self.assertNotIn("publisher", calls[0][1].lower())
            self.assertEqual(runner.mc_rl_debug_capture["status"], "READY")

    def test_mc_rl_debug_discovery_method_does_not_shadow_rclpy_publishers_state(self) -> None:
        source = MC_RL_DEBUG_CAPTURE.read_text(encoding="utf-8")
        self.assertIn("def _debug_topic_publishers(", source)
        self.assertNotIn("def _publishers(", source)
        self.assertIn("records = self._debug_topic_publishers()", source)

    def test_requested_mc_rl_debug_observer_starts_before_the_single_m1_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.capture_mc_rl_debug = True
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            runner.remaining = lambda: 60.0
            calls = []
            runner.start_mc_rl_debug_capture = lambda: calls.append("debug_observer")

            def fake_run(name, command, **kwargs):
                calls.append(name)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()

            self.assertEqual(calls, ["debug_observer", "m1_first_motion_diagnostic"])
            self.assertEqual(runner.m1_result["status"], "NO_RESULT_FILE")

    def test_mc_rl_debug_overlay_hash_mismatch_blocks_before_observer_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir) / "evidence"
            runtime = Path(temp_dir) / "runtime"
            evidence.mkdir()
            self.make_debug_overlay_fixture(evidence, runtime)
            (runtime / "mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so").write_bytes(b"changed")
            runner = self.make_runner(evidence)
            calls = []
            runner.docker_exec = lambda *args, **kwargs: calls.append((args, kwargs))
            with patch("issue80_m1_auto_reset_run.RUNTIME", runtime):
                with self.assertRaisesRegex(RuntimeError, "mc McRLDebug introspection library hash"):
                    runner.start_mc_rl_debug_capture()
            self.assertEqual(calls, [])

    def test_stages_verified_runtime_launchers_under_names_used_by_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.stage_research_scripts()
            manifest = json.loads((evidence / "staged_research_scripts.json").read_text())
            by_name = {row["staged_name"]: row for row in manifest["scripts"]}
            for staged_name, source in (("launch_sim.sh", SIM_LAUNCH), ("launch_mc.sh", MC_LAUNCH)):
                staged = evidence / staged_name
                self.assertEqual(staged.read_bytes(), source.read_bytes())
                self.assertEqual(by_name[staged_name]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            sim_script = (evidence / "launch_sim.sh").read_text(encoding="utf-8")
            self.assertIn("ROBOTSIM_SIMULATOR_CFG_OVERRIDE", sim_script)
            self.assertIn("$SIM_ROBOT_PATH/simulator/default.yaml", sim_script)

    def test_refuses_to_overwrite_a_stale_staged_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            stale = evidence / M1_MODE_LOGIC.name
            stale.write_bytes(b"stale module")
            runner = self.make_runner(evidence)
            with self.assertRaisesRegex(RuntimeError, "differs from current source"):
                runner.stage_research_scripts()
            self.assertEqual(stale.read_bytes(), b"stale module")
            self.assertFalse((evidence / "staged_research_scripts.json").exists())

    def test_runner_does_not_start_first_motion_without_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.maintainer_authorized_first_motion = False
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(captured, [])
            self.assertEqual(runner.m1_result["status"], "NOT_AUTHORIZED")

    def test_runner_accepts_zero_only_authorization_without_motion_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.maintainer_authorized_first_motion = False
            runner.args.maintainer_authorized_zero_only_activation = True
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 2, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(len(captured), 1)
            child = shlex.split(captured[0][-1].rsplit(" && ", 1)[1])
            self.assertIn("--maintainer-authorized-zero-only-activation", child)
            self.assertIn("--zero-only-activation-verification", child)
            self.assertNotIn("--maintainer-authorized", child)
            self.assertNotIn("--route-integration", child)
            self.assertNotIn("--route-curved-path-diagnostic", child)

    def test_runner_accepts_only_prerequisite_validated_activation_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.maintainer_authorized_first_motion = False
            runner.args.maintainer_authorized_activation_probe = True
            runner.args.activation_probe_prerequisite = {
                "status": "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED",
                "zero_only_preflight_attempt_id": "zero-preflight-id",
            }
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 2, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(len(captured), 1)
            child = shlex.split(captured[0][-1].rsplit(" && ", 1)[1])
            self.assertIn("--maintainer-authorized-activation-probe", child)
            self.assertIn("--activation-probe-qualification", child)

            runner.args.activation_probe_prerequisite = {"status": "BLOCKED"}
            captured.clear()
            runner.run_m1()
            self.assertEqual(captured, [])
            self.assertEqual(runner.m1_result["status"], "BLOCKED")

    def test_activation_probe_cli_reaches_offline_runner_without_undefined_profile_fields(self) -> None:
        class OfflineRunner:
            def __init__(self, args: argparse.Namespace):
                self.args = args

            def execute(self) -> dict[str, str]:
                return {"status": "BLOCKED", "fault": "offline runner sentinel"}

        argv = [
            "issue80_m1_auto_reset_run.py",
            "--evidence-dir", "/tmp/reserved-evidence",
            "--container", "issue80-activation-probe-test",
            "--attempt-id", "activation-probe-cli-test",
            "--maintainer-authorized-activation-probe",
            "--activation-probe-prerequisite-evidence", "/tmp/prerequisite",
        ]
        prerequisite = {"status": "SOURCE_GROUNDED_ACTIVATION_PROBE_SUPPORTED"}
        with (
            patch.object(issue80_m1_auto_reset_run.sys, "argv", argv),
            patch.object(
                issue80_m1_auto_reset_run,
                "validate_activation_probe_prerequisite",
                return_value=prerequisite,
            ),
            patch.object(issue80_m1_auto_reset_run, "validate_reserved_evidence_directory"),
            patch.object(issue80_m1_auto_reset_run, "BoundedRun", OfflineRunner),
        ):
            self.assertEqual(issue80_m1_auto_reset_run.main(), 2)

    def test_parent_parser_accepts_local01_route_existing_seven_second_stop_observation(self) -> None:
        class OfflineRunner:
            def __init__(self, args: argparse.Namespace):
                self.args = args

            def execute(self) -> dict[str, str]:
                return {"status": "BLOCKED", "fault": "offline runner sentinel"}

        argv = [
            "issue80_m1_auto_reset_run.py",
            "--evidence-dir", "/tmp/reserved-evidence",
            "--container", "issue80-local01-route-test",
            "--attempt-id", "local01-route-cli-test",
            "--maintainer-authorized-first-motion",
            "--require-stand-warmup-composite",
            "--local01-scene-world-poses", "/tmp/scene_world_poses.json",
            "--forward-speed", "0.12",
            "--pulse-duration", "1.0",
            "--extra-stop-observation", "7.0",
        ]
        local01 = {
            "route": object(),
            "route_input": {"route_id": "fixture"},
            "source": {"source_commit": "fixture"},
            "static_scene_preflight": {"status": "PASS_STATIC_SCENE_CLEARANCE"},
            "source_sha256": "fixture",
        }
        with (
            patch.object(issue80_m1_auto_reset_run.sys, "argv", argv),
            patch.object(
                issue80_m1_auto_reset_run,
                "validate_local01_scene_route_input",
                return_value=local01,
            ),
            patch.object(issue80_m1_auto_reset_run, "validate_reserved_evidence_directory"),
            patch.object(issue80_m1_auto_reset_run, "BoundedRun", OfflineRunner),
        ):
            self.assertEqual(issue80_m1_auto_reset_run.main(), 2)

    def test_parent_does_not_start_container_without_first_motion_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.maintainer_authorized_first_motion = False
            runner.start_container = lambda: self.fail("unauthorized run must not start Docker")
            result = runner.execute()
            self.assertEqual(result["status"], "NOT_AUTHORIZED")
            self.assertFalse(runner.container_created)

    def test_runtime_container_must_be_bridge_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            inspect_payload = [{
                "HostConfig": {"NetworkMode": "host"},
                "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
            }]

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                if name == "container_inspect_start":
                    return subprocess.CompletedProcess(command, 0, json.dumps(inspect_payload), "")
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            with self.assertRaisesRegex(RuntimeError, "bridge-isolated"):
                runner.start_container()
            self.assertEqual(runner.network_mode, "host")

    def test_container_inspect_rejects_host_network_for_first_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            calls: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                payload = [{
                    "HostConfig": {"NetworkMode": "host"},
                    "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
                }]
                return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

            runner.run = fake_run
            with self.assertRaisesRegex(RuntimeError, "bridge-isolated"):
                runner.start_container()
            self.assertTrue(runner.container_created)
            self.assertIn("--network=bridge", calls[0])

    def test_local01_reference_scene_is_mounted_read_only_from_its_narrow_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence = root / "evidence"
            reference = root / "raicom2026-x2-mujoco-scene"
            evidence.mkdir()
            reference.mkdir()
            local01_scene = evidence / "local01_scene"
            local01_scene.mkdir()
            pose_input = local01_scene / "scene_world_poses.json"
            pose_input.write_text("{}\n", encoding="utf-8")
            runner = self.make_runner(evidence)
            runner.local01_scene_bundle = {"classification": "SIMULATION_ONLY_PATH_RESOLVED_SCENE_WRAPPER"}
            runner.local01_scene_source = {"reference_root": str(reference)}
            runner.local01_scene_readonly_mounts = [
                (pose_input, "/workspace/evidence/local01_scene/scene_world_poses.json")
            ]
            commands = []

            def fake_run(name, command, **kwargs):
                commands.append(command)
                if name == "docker_run":
                    return subprocess.CompletedProcess(command, 0, "container-id", "")
                mounts = [{
                    "Destination": "/workspace/raicom_scene_source",
                    "Source": str(reference),
                    "Type": "bind",
                    "RW": False,
                }, {
                    "Destination": "/workspace/evidence/local01_scene/scene_world_poses.json",
                    "Source": str(pose_input),
                    "Type": "bind",
                    "RW": False,
                }]
                return subprocess.CompletedProcess(command, 0, json.dumps([{
                    "Id": "a" * 64,
                    "HostConfig": {"NetworkMode": "bridge"},
                    "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
                    "Mounts": mounts,
                }]), "")

            runner.run = fake_run
            runner.start_container()
            self.assertTrue(runner.container_created)
            self.assertIn(
                "type=bind,source=" + str(reference) + ",target=/workspace/raicom_scene_source,readonly",
                commands[0],
            )
            self.assertIn(
                "type=bind,source=" + str(pose_input)
                + ",target=/workspace/evidence/local01_scene/scene_world_poses.json,readonly",
                commands[0],
            )
            self.assertEqual(runner.container_id, "a" * 64)

    def test_local01_scene_startup_uses_the_staged_simulator_config_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            local01 = evidence / "local01_scene"
            local01.mkdir()
            scene = local01 / "scene_workspace_runtime.xml"
            simulator_config = local01 / "simulator_default_local01_scene.yaml"
            scene.write_text("<mujoco model='test'/>", encoding="utf-8")
            simulator_config.write_text("simulation_model: scene.xml\n", encoding="utf-8")
            runner = self.make_runner(evidence)
            runner.local01_scene_bundle = {
                "runtime_files_sha256": {
                    "scene_workspace_runtime.xml": hashlib.sha256(scene.read_bytes()).hexdigest(),
                    "simulator_default_local01_scene.yaml": hashlib.sha256(simulator_config.read_bytes()).hexdigest(),
                }
            }
            calls = []

            def fake_exec(name, command, **kwargs):
                calls.append((name, command))
                if name in {"verify_local01_runtime_scene_wrapper_sha256", "verify_local01_simulator_config_sha256"}:
                    target = scene if name == "verify_local01_runtime_scene_wrapper_sha256" else simulator_config
                    return subprocess.CompletedProcess(command, 0, hashlib.sha256(target.read_bytes()).hexdigest() + "  staged\n", "")
                return subprocess.CompletedProcess(command, 0, "MujocoSimModule Started Succeeded\n", "")

            runner.docker_exec = fake_exec
            runner.start_sim_first()
        launch = calls[0][1]
        self.assertIn("ROBOTSIM_SIMULATOR_CFG_OVERRIDE=/workspace/evidence/local01_scene/simulator_default_local01_scene.yaml", launch)
        self.assertTrue(any(name == "verify_local01_runtime_scene_wrapper_sha256" for name, _ in calls))
        self.assertTrue(any(name == "verify_local01_simulator_config_sha256" for name, _ in calls))

    def test_cleanup_records_normal_teardown_and_process_crash_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.container_created = True
            runner.container_id = "a" * 64
            runner.run = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", "")
            docker_results = [
                subprocess.CompletedProcess(["docker", "inspect"], 0, json.dumps([{"Id": "a" * 64}]), ""),
                subprocess.CompletedProcess(["docker", "stop"], 0, "test-container", ""),
                subprocess.CompletedProcess(["docker", "rm"], 0, "test-container", ""),
                subprocess.CompletedProcess(
                    ["docker", "inspect"], 1, "", "daemon locale-specific missing-object text"
                ),
                subprocess.CompletedProcess(["docker", "ps"], 0, "", ""),
            ]
            with patch("issue80_m1_auto_reset_run.subprocess.run", side_effect=docker_results):
                runner.cleanup()
            self.assertTrue(runner.cleanup_evidence["container_absent_after_cleanup"])
            self.assertFalse(runner.cleanup_evidence["process_crash_or_sigkill_cleanup_guaranteed"])
            self.assertEqual(
                [row["operation"] for row in runner.cleanup_evidence["operations"]][-3:],
                ["docker_stop", "docker_rm", "verify_container_absent"],
            )

    def test_cleanup_verification_requires_positive_exact_name_and_id_absence(self) -> None:
        container_id = "a" * 64
        absent = assess_container_absence(
            inspect_returncode=1,
            list_returncode=0,
            list_stdout=f"{'b' * 64}\tother-container\n",
            container_name="test-container",
            container_id=container_id,
        )
        self.assertTrue(absent["container_absent"])
        still_present_by_exact_name = assess_container_absence(
            inspect_returncode=1,
            list_returncode=0,
            list_stdout=f"{'b' * 64}\ttest-container\n",
            container_name="test-container",
            container_id=container_id,
        )
        self.assertFalse(still_present_by_exact_name["container_absent"])
        still_present_by_exact_id = assess_container_absence(
            inspect_returncode=1,
            list_returncode=0,
            list_stdout=f"{container_id}\trenamed-container\n",
            container_name="test-container",
            container_id=container_id,
        )
        self.assertFalse(still_present_by_exact_id["container_absent"])
        self.assertFalse(assess_container_absence(
            inspect_returncode=1,
            list_returncode=1,
            list_stdout="",
            container_name="test-container",
            container_id=container_id,
        )["container_absent"])
        self.assertFalse(assess_container_absence(
            inspect_returncode=0,
            list_returncode=0,
            list_stdout="",
            container_name="test-container",
            container_id=container_id,
        )["container_absent"])
        self.assertFalse(assess_container_absence(
            inspect_returncode=1,
            list_returncode=0,
            list_stdout="unparseable list row\n",
            container_name="test-container",
            container_id=container_id,
        )["container_absent"])

    def test_child_result_decoder_accepts_mapping_and_one_string_wrapper(self) -> None:
        child = {"status": "BLOCKED", "fault": "transport identity mismatch"}
        encodings = (
            json.dumps(child),
            json.dumps(json.dumps(child)),
            json.dumps(str(child)),
        )
        for serialized in encodings:
            with self.subTest(serialized=serialized[:24]):
                self.assertEqual(decode_m1_child_result(serialized), child)

    def test_child_result_decoder_rejects_malformed_nonmapping_or_invalid_status(self) -> None:
        invalid = (
            "not-json",
            json.dumps(json.dumps(["not", "a", "mapping"])),
            json.dumps(json.dumps({"status": "FAILED_UNKNOWN"})),
            json.dumps(json.dumps(json.dumps({"status": "BLOCKED"}))),
        )
        for serialized in invalid:
            with self.subTest(serialized=serialized[:24]):
                with self.assertRaises(ValueError):
                    decode_m1_child_result(serialized)

    def test_run_m1_preserves_blocked_cause_from_string_wrapped_child_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.maintainer_authorized_first_motion = True
            runner.args.ramp_start_speed = 0.105
            runner.args.ramp_duration_s = 2.0
            runner.network_mode = "bridge"
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.remaining = lambda: 60.0
            child_result = {
                "status": "BLOCKED",
                "fault": "simulation-only transport isolation failed: expected verified MC endpoint",
                "velocity_command_trace": [],
            }

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                (evidence / "m1_first_motion_result.json").write_text(
                    json.dumps(str(child_result)), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 2, "", "")

            runner.run = fake_run
            runner.run_m1()
            self.assertEqual(runner.m1_result["status"], "BLOCKED")
            self.assertEqual(runner.m1_result["fault"], child_result["fault"])
            self.assertEqual(runner.m1_result["velocity_command_trace"], [])

    def test_parent_result_preserves_child_blocked_cause_without_attribute_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.args.maintainer_authorized_first_motion = True
            runner.remaining = lambda: 60.0
            runner.stage_research_scripts = lambda: None
            runner.start_container = lambda: None
            runner.start_sim_first = lambda: None
            runner.capture_service_topic_inventory = lambda: None
            runner.request_sd_and_verify = lambda: {
                "response_status_value": 0,
                "response_header_code": 0,
                "get_mc_action_readback": [{"stdout": "Mode: STAND_DEFAULT, Status: 100"}],
            }
            runner.wait_for_immediate_reset_pose = lambda: {"status": "POST_RESET_UPRIGHT_OBSERVED"}
            runner.inspect_window = lambda: (42, "MuJoCo", 7)
            runner.capture_reset_image = lambda *args: {"status": "NOT_CAPTURED"}
            runner.start_observer = lambda: None
            runner.dispatch_reset = lambda *args: None
            runner.dispatch_native_reset_button = lambda *args: None
            reset_result = {"classification": "AUTO_RESET_PASS"}

            def classify_reset(*args):
                runner.reset_classification = "AUTO_RESET_PASS"
                runner.reset_analysis = reset_result
                return reset_result

            runner.wait_and_classify_reset = classify_reset
            child_result = {"status": "BLOCKED", "fault": "verified MC identity mismatch"}
            runner.run_m1 = lambda: setattr(runner, "m1_result", child_result)

            result = runner.execute()

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["m1_status"], "BLOCKED")
        self.assertEqual(result["fault"], child_result["fault"])
        self.assertEqual(result["m1_result"]["fault"], child_result["fault"])

    def test_missing_container_teardown_blocks_run_closeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = self.make_runner(Path(temp_dir))
            runner.cleanup_evidence.update({
                "attempted": True,
                "container_absent_after_cleanup": False,
            })
            result = {"status": "PASS", "fault": None}
            runner.apply_container_teardown_gate(result)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("container teardown", result["fault"])

    def test_authorized_runner_uses_one_pinned_source_and_hard_limited_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            runner = self.make_runner(evidence)
            runner.args.maintainer_authorized_first_motion = True
            runner.args.ramp_start_speed = 0.105
            runner.args.ramp_duration_s = 2.0
            runner.args.startup_speed = 0.12
            runner.args.startup_transition_displacement = 0.02
            runner.args.startup_transition_timeout_s = 0.5
            runner.network_mode = "bridge"
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.remaining = lambda: 60.0
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                captured.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run
            runner.run_m1()
            shell_command = captured[0][-1]
            self.assertIn("first-motion --maintainer-authorized", shell_command)
            self.assertIn("--reset-gate-verified", shell_command)
            self.assertIn("--attempt-id test-attempt", shell_command)
            self.assertIn("--ramp-start-speed 0.105", shell_command)
            self.assertIn("--ramp-duration-s 2.0", shell_command)
            self.assertIn("--startup-speed 0.12", shell_command)
            self.assertIn("--startup-transition-displacement 0.02", shell_command)
            self.assertIn("--startup-transition-timeout-s 0.5", shell_command)
            self.assertIn("ROBOTSIM_VERIFIED_NETWORK_MODE=bridge", captured[0])
            self.assertIn("--robot-model-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/robot_model.yaml", shell_command)
            self.assertIn("--mc-sim-config-path /workspace/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml", shell_command)
            self.assertNotIn("--source ", shell_command)
            self.assertNotIn("--activation-pulse", shell_command)
            self.assertNotIn("--zero-source-arbitration", shell_command)


if __name__ == "__main__":
    unittest.main()
