"""Focused tests for the isolated Run38 FoundationWalk config diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from issue80_m1_auto_reset_run import (
    BoundedRun,
    FOUNDATIONWALK_CONFIG_CONTAINER_PATH,
    RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE,
    build_run38_foundationwalk_knee_scale_overlay,
    build_run41_foundationwalk_min_run_overlay,
    classify_run38_diagnostic_outcome,
    copy_verified_mc_rl_debug_overlay_bundle,
)


def make_config(*, left_scale: float = 0.25, right_scale: float = 0.25) -> bytes:
    action_seq = [f"joint_{index}" for index in range(29)]
    action_seq[9:11] = ["left_knee_joint", "right_knee_joint"]
    action_scale = [0.25] * 29
    action_scale[9:11] = [left_scale, right_scale]
    action_seq_text = ", ".join(f"\"{joint}\"" for joint in action_seq)
    scale_text = ", ".join(f"{scale:.2f}" for scale in action_scale)
    return (
        "BaseConfig:\n"
        f"  action_seq: [{action_seq_text}]\n"
        f"  action_scale: [{scale_text}]\n"
        "  dt: 0.02\n"
        "FoundationWalkConfig:\n"
        "  foundationwalk:\n"
        "    min_run_velx: 0.50\n"
        "    cmd_threshold: 0.1\n"
    ).encode("utf-8")


class FoundationWalkOverlayTests(unittest.TestCase):
    def test_overlay_changes_only_both_knee_scales(self) -> None:
        source = make_config()
        digest = hashlib.sha256(source).hexdigest()
        overlay, manifest = build_run38_foundationwalk_knee_scale_overlay(
            source, expected_source_sha256=digest
        )
        source_yaml = yaml.safe_load(source)
        overlay_yaml = yaml.safe_load(overlay)

        source_scales = source_yaml["BaseConfig"]["action_scale"]
        overlay_scales = overlay_yaml["BaseConfig"]["action_scale"]
        self.assertEqual(len(overlay_scales), 29)
        self.assertEqual(
            [index for index, pair in enumerate(zip(source_scales, overlay_scales)) if pair[0] != pair[1]],
            [9, 10],
        )
        self.assertEqual(overlay_scales[9:11], [RUN38_DIAGNOSTIC_KNEE_ACTION_SCALE] * 2)
        self.assertEqual(overlay_yaml["BaseConfig"]["action_seq"][9:11], ["left_knee_joint", "right_knee_joint"])
        self.assertEqual(overlay_yaml["BaseConfig"]["dt"], source_yaml["BaseConfig"]["dt"])
        self.assertTrue(manifest["all_other_yaml_values_unchanged"])
        self.assertFalse(manifest["trained_policy_weights_changed"])
        self.assertEqual(hashlib.sha256(overlay).hexdigest(), manifest["overlay_sha256"])

    def test_overlay_rejects_an_unpinned_source(self) -> None:
        source = make_config()
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            build_run38_foundationwalk_knee_scale_overlay(
                source, expected_source_sha256="0" * 64
            )

    def test_overlay_rejects_non_knee_or_nonbaseline_scales(self) -> None:
        source = make_config(left_scale=0.24)
        digest = hashlib.sha256(source).hexdigest()
        with self.assertRaisesRegex(ValueError, "not both 0.25"):
            build_run38_foundationwalk_knee_scale_overlay(
                source, expected_source_sha256=digest
            )

    def test_run41_overlay_changes_only_knees_and_measured_min_run_floor(self) -> None:
        source = make_config()
        digest = hashlib.sha256(source).hexdigest()
        overlay, manifest = build_run41_foundationwalk_min_run_overlay(
            source, expected_source_sha256=digest
        )
        source_yaml = yaml.safe_load(source)
        overlay_yaml = yaml.safe_load(overlay)
        changed_scale_indexes = [
            index for index, (old, new) in enumerate(zip(
                source_yaml["BaseConfig"]["action_scale"],
                overlay_yaml["BaseConfig"]["action_scale"],
            )) if old != new
        ]
        self.assertEqual(changed_scale_indexes, [9, 10])
        self.assertEqual(
            overlay_yaml["FoundationWalkConfig"]["foundationwalk"]["min_run_velx"],
            0.1,
        )
        self.assertEqual(
            overlay_yaml["FoundationWalkConfig"]["foundationwalk"]["cmd_threshold"],
            source_yaml["FoundationWalkConfig"]["foundationwalk"]["cmd_threshold"],
        )
        self.assertEqual(manifest["schema"], "robotsim.issue80.foundationwalk-knee-minrun-diagnostic.v1")
        self.assertEqual(manifest["changed_values"][-1]["config_path"], "FoundationWalkConfig.foundationwalk.min_run_velx")

    def test_overlay_rejects_duplicate_action_scale_lines(self) -> None:
        source = make_config() + b"  action_scale: [0.25]\n"
        digest = hashlib.sha256(source).hexdigest()
        with self.assertRaisesRegex(ValueError, "exactly one inline action_scale"):
            build_run38_foundationwalk_knee_scale_overlay(
                source, expected_source_sha256=digest
            )

    def test_diagnostic_classification_does_not_claim_original_acceptance(self) -> None:
        self.assertEqual(
            classify_run38_diagnostic_outcome("FIRST_PHYSICAL_WALKING_PASS"),
            "SIMULATION_ONLY_DIAGNOSTIC_MOTION_AND_STOP_OBSERVED",
        )
        self.assertEqual(
            classify_run38_diagnostic_outcome("FIRST_PHYSICAL_WALKING_FAIL"),
            "SIMULATION_ONLY_DIAGNOSTIC_SAFETY_FAILURE",
        )

    def test_container_mounts_exact_overlay_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            overlay = evidence / "derived.yaml"
            overlay.write_text("diagnostic: true\n", encoding="utf-8")
            args = argparse.Namespace(
                evidence_dir=str(evidence),
                container="issue80-test-overlay",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
            )
            runner = BoundedRun(args)
            runner.foundationwalk_config_overlay = overlay
            expected_id = "a" * 64
            captured: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0, check: bool = True):
                if name == "docker_run":
                    captured.append(command)
                    return subprocess.CompletedProcess(command, 0, "", "")
                if name == "container_inspect_start":
                    inspect = [{
                        "Id": expected_id,
                        "HostConfig": {"NetworkMode": "bridge"},
                        "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
                        "Mounts": [{
                            "Type": "bind",
                            "Source": str(overlay.resolve()),
                            "Destination": FOUNDATIONWALK_CONFIG_CONTAINER_PATH,
                            "RW": False,
                        }],
                    }]
                    return subprocess.CompletedProcess(command, 0, json.dumps(inspect), "")
                raise AssertionError(f"unexpected command {name}")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.start_container()
            docker_args = captured[0]
            mount_values = [
                docker_args[index + 1]
                for index, value in enumerate(docker_args[:-1])
                if value == "--mount"
            ]
            overlay_mounts = [
                value for value in mount_values
                if "target=" + FOUNDATIONWALK_CONFIG_CONTAINER_PATH in value
            ]
            self.assertEqual(len(overlay_mounts), 1)
            overlay_mount = overlay_mounts[0]
            self.assertIn("source=" + str(overlay.resolve()), overlay_mount)
            self.assertIn("target=" + FOUNDATIONWALK_CONFIG_CONTAINER_PATH, overlay_mount)
            self.assertTrue(overlay_mount.endswith(",readonly"))
            self.assertEqual(runner.container_id, expected_id)

    def test_overlay_hash_is_verified_after_sim_ready_and_before_mc_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            overlay = evidence / "derived.yaml"
            overlay.write_text("diagnostic: true\n", encoding="utf-8")
            digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
            runner = BoundedRun(argparse.Namespace(
                evidence_dir=str(evidence),
                container="issue80-test-order",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
            ))
            runner.foundationwalk_config_overlay = overlay
            runner.foundationwalk_config_overlay_manifest = {"overlay_sha256": digest}
            events: list[str] = []

            def fake_docker_exec(name: str, shell: str, *, timeout_s: float = 8.0,
                                 detached: bool = False, check: bool = True):
                events.append(name)
                if name == "verify_mounted_foundationwalk_config_overlay":
                    self.assertIn(FOUNDATIONWALK_CONFIG_CONTAINER_PATH, shell)
                    return subprocess.CompletedProcess(
                        ["docker", "exec"], 0, f"{digest}  {FOUNDATIONWALK_CONFIG_CONTAINER_PATH}\n", ""
                    )
                return subprocess.CompletedProcess(["docker", "exec"], 0, "", "")

            runner.docker_exec = fake_docker_exec  # type: ignore[method-assign]
            runner.start_sim_first()

            self.assertLess(
                events.index("verify_mounted_foundationwalk_config_overlay"),
                events.index("start_mc_after_sim_topics"),
            )
            verification = json.loads(
                (evidence / "foundationwalk_config_overlay_runtime_verification.json").read_text()
            )
            self.assertTrue(verification["matched_before_mc_launch"])

    def test_cli_fixes_diagnostic_speed_and_duration_before_any_runtime(self) -> None:
        script = Path(__file__).with_name("issue80_m1_auto_reset_run.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--evidence-dir", temp_dir,
                    "--container", "must-not-start",
                    "--attempt-id", "offline-rejection-test",
                    "--run38-knee-scale-diagnostic",
                    "--capture-mc-rl-debug",
                    "--maintainer-authorized-first-motion",
                    "--forward-speed", "0.11",
                    "--pulse-duration", "3.0",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("FoundationWalk diagnostic is fixed to 0.112 m/s, 3.0 s", result.stderr)

    def test_mc_rl_debug_bundle_copy_is_limited_to_governed_prior_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            issue_root = Path(temp_dir) / "reviews" / "issue-80"
            source = issue_root / "prior-run"
            evidence = issue_root / "next-run"
            (source / "mc_debug_overlay/src/aimdk_msgs/msg").mkdir(parents=True)
            (source / "mc_debug_overlay/install").mkdir(parents=True)
            evidence.mkdir(parents=True)
            (source / "mc_rl_debug_schema.json").write_text("{\"results\": []}\n", encoding="utf-8")
            (source / "MC_RL_DEBUG_SCHEMA_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
            (source / "build_mc_debug_overlay.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (source / "mc_debug_overlay/src/aimdk_msgs/msg/McRLDebug.msg").write_text(
                "MessageHeader header\nfloat32[] obs_vec\nfloat32[] action_vec\n", encoding="utf-8"
            )
            (source / "mc_debug_overlay/src/aimdk_msgs/msg/MessageHeader.msg").write_text(
                "uint64 sequence\n", encoding="utf-8"
            )
            (source / "mc_debug_overlay/install/setup.bash").write_text("# built overlay\n", encoding="utf-8")
            (source / "mc_debug_overlay/install/libaimdk_msgs.so").write_bytes(b"test-library")

            manifest = copy_verified_mc_rl_debug_overlay_bundle(
                source, evidence, approved_root=issue_root
            )

            self.assertEqual(manifest["classification"], "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY")
            self.assertFalse(manifest["source_artifact_manifest_present"])
            self.assertEqual(
                (evidence / "mc_debug_overlay/install/libaimdk_msgs.so").read_bytes(),
                b"test-library",
            )
            self.assertIn("mc_debug_overlay/install/setup.bash", manifest["copied_file_sha256"])

    def test_mc_rl_debug_bundle_copy_rejects_source_outside_approved_issue_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approved = root / "reviews" / "issue-80"
            approved.mkdir(parents=True)
            outside = root / "untrusted-source"
            outside.mkdir()
            evidence = approved / "next-run"
            evidence.mkdir()
            with self.assertRaisesRegex(ValueError, "beneath the approved Issue 80 evidence root"):
                copy_verified_mc_rl_debug_overlay_bundle(
                    outside, evidence, approved_root=approved
                )

    def test_container_mounts_verified_debug_install_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = Path(temp_dir)
            install = evidence / "mc_debug_overlay/install"
            install.mkdir(parents=True)
            (install / "setup.bash").write_text("# overlay\n", encoding="utf-8")
            args = argparse.Namespace(
                evidence_dir=str(evidence),
                container="issue80-test-debug-overlay",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
            )
            runner = BoundedRun(args)
            runner.mc_rl_debug_overlay_manifest = {"classification": "SIMULATION_ONLY_DIAGNOSTIC_MESSAGE_OVERLAY"}
            expected_id = "b" * 64

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0, check: bool = True):
                if name == "docker_run":
                    return subprocess.CompletedProcess(command, 0, "", "")
                if name == "container_inspect_start":
                    inspect = [{
                        "Id": expected_id,
                        "HostConfig": {"NetworkMode": "bridge"},
                        "Config": {"Env": ["ROS_DOMAIN_ID=20", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"]},
                        "Mounts": [{
                            "Type": "bind",
                            "Source": str(install.resolve()),
                            "Destination": "/workspace/evidence/mc_debug_overlay/install",
                            "RW": False,
                        }],
                    }]
                    return subprocess.CompletedProcess(command, 0, json.dumps(inspect), "")
                raise AssertionError(f"unexpected command {name}")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.start_container()
            inspect_output = json.loads(
                (evidence / "container-inspect-start.json").read_text(encoding="utf-8")
            )
            debug_mount = next(
                mount for mount in inspect_output[0]["Mounts"]
                if mount["Destination"] == "/workspace/evidence/mc_debug_overlay/install"
            )
            self.assertFalse(debug_mount["RW"])

    def test_run41_runner_passes_the_measured_speed_guard_to_the_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run41",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run41_min_run_diagnostic=True,
                run38_knee_scale_diagnostic=False,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=3.0,
                forward_speed=0.112,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            self.assertIn("--max-observed-planar-speed-m-s 0.2", commands[0][-1])

    def test_run42_runner_adds_zero_until_applied_stand_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run42",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run41_min_run_diagnostic=False,
                run42_stop_confirmation_diagnostic=True,
                run38_knee_scale_diagnostic=False,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=3.0,
                forward_speed=0.112,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            self.assertIn("--max-observed-planar-speed-m-s 0.2", commands[0][-1])
            self.assertIn("--zero-until-stand-timeout-s 3.5", commands[0][-1])

    def test_run43_runner_uses_zero_only_stop_without_stand_reassertion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run43",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run41_min_run_diagnostic=False,
                run42_stop_confirmation_diagnostic=False,
                run43_zero_only_stop_diagnostic=True,
                run38_knee_scale_diagnostic=False,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=3.0,
                forward_speed=0.112,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            child_command = commands[0][-1]
            self.assertIn("--max-observed-planar-speed-m-s 0.2", child_command)
            self.assertIn("--zero-until-stand-timeout-s 3.5", child_command)
            self.assertIn("--zero-only-stop", child_command)

    def test_run44_threshold_margin_is_fixed_at_014_and_keeps_root_speed_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run44",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run41_min_run_diagnostic=False,
                run42_stop_confirmation_diagnostic=False,
                run43_zero_only_stop_diagnostic=False,
                run44_threshold_margin_diagnostic=True,
                run38_knee_scale_diagnostic=False,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=3.0,
                forward_speed=0.14,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            child_command = commands[0][-1]
            self.assertIn("--forward-speed 0.14", child_command)
            self.assertIn("--max-observed-planar-speed-m-s 0.2", child_command)
            self.assertIn("--zero-only-stop", child_command)

    def test_run45_runner_uses_the_existing_ten_second_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run45",
                attempt_id="test-attempt",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run41_min_run_diagnostic=False,
                run42_stop_confirmation_diagnostic=False,
                run43_zero_only_stop_diagnostic=False,
                run44_threshold_margin_diagnostic=False,
                run45_ten_second_gait_diagnostic=True,
                run38_knee_scale_diagnostic=False,
                capture_mc_rl_debug=False,
                extra_stop_observation=5.0,
                pulse_duration=10.0,
                forward_speed=0.14,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=None,
                startup_transition_displacement=None,
                startup_transition_timeout_s=0.0,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            child_command = commands[0][-1]
            self.assertIn("--pulse-duration 10.0", child_command)
            self.assertIn("--forward-speed 0.14", child_command)
            self.assertIn("--max-observed-planar-speed-m-s 0.2", child_command)
            self.assertIn("--zero-only-stop", child_command)

    def test_run46_replays_the_threshold_drop_with_all_existing_safety_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                evidence_dir=temp_dir,
                container="issue80-test-run46",
                attempt_id="test-attempt-run46",
                max_runtime_s=180.0,
                maintainer_authorized_first_motion=True,
                maintainer_authorized_standing_takeover=False,
                run38_knee_scale_diagnostic=False,
                run41_min_run_diagnostic=False,
                run42_stop_confirmation_diagnostic=False,
                run43_zero_only_stop_diagnostic=False,
                run44_threshold_margin_diagnostic=False,
                run45_ten_second_gait_diagnostic=False,
                run46_startup_cruise_threshold_diagnostic=True,
                capture_mc_rl_debug=True,
                extra_stop_observation=5.0,
                pulse_duration=3.0,
                forward_speed=0.105,
                ramp_start_speed=None,
                ramp_duration_s=0.0,
                startup_speed=0.12,
                startup_transition_displacement=0.01,
                startup_transition_timeout_s=0.75,
            )
            runner = BoundedRun(args)
            runner.reset_classification = "AUTO_RESET_PASS"
            runner.network_mode = "bridge"
            runner.start_mc_rl_debug_capture = lambda: None  # type: ignore[method-assign]
            commands: list[list[str]] = []

            def fake_run(name: str, command: list[str], *, timeout_s: float = 8.0,
                         check: bool = True):
                self.assertEqual(name, "m1_first_motion_diagnostic")
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            runner.run = fake_run  # type: ignore[method-assign]
            runner.run_m1()
            child = commands[0][-1]
            self.assertIn("--forward-speed 0.105", child)
            self.assertIn("--pulse-duration 3.0", child)
            self.assertIn("--startup-speed 0.12", child)
            self.assertIn("--startup-transition-displacement 0.01", child)
            self.assertIn("--startup-transition-timeout-s 0.75", child)
            self.assertIn("--max-observed-planar-speed-m-s 0.2", child)
            self.assertIn("--zero-until-stand-timeout-s 3.5", child)
            self.assertIn("--zero-only-stop", child)

    def test_run46_cli_rejects_a_changed_cruise_target_before_runtime(self) -> None:
        script = Path(__file__).with_name("issue80_m1_auto_reset_run.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--evidence-dir", temp_dir,
                    "--container", "must-not-start",
                    "--attempt-id", "offline-run46-rejection",
                    "--run46-startup-cruise-threshold-diagnostic",
                    "--capture-mc-rl-debug",
                    "--maintainer-authorized-first-motion",
                    "--forward-speed", "0.106",
                    "--pulse-duration", "3.0",
                    "--startup-speed", "0.12",
                    "--startup-transition-displacement", "0.01",
                    "--startup-transition-timeout-s", "0.75",
                    "--extra-stop-observation", "5.0",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Run46 is fixed to 0.12 m/s startup, 0.105 m/s cruise", result.stderr)


if __name__ == "__main__":
    unittest.main()
