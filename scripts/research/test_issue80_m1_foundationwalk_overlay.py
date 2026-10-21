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
        self.assertIn("Run38 diagnostic is fixed to 0.112 m/s, 3.0 s", result.stderr)

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


if __name__ == "__main__":
    unittest.main()
