"""Offline tests for exact, no-overwrite research-code staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from issue80_m1_auto_reset_run import (
    BoundedRun, MC_LAUNCH, M1_FIRST_MOTION_LOGIC, M1_MODE_LOGIC, MC_RL_DEBUG_CAPTURE,
    SET_ACTION_CAPTURE, SIM_LAUNCH,
    assess_container_absence, classify_first_motion_outcome, decode_m1_child_result,
)


class ResearchScriptStagingTests(unittest.TestCase):
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
        }
        self.assertEqual(
            classify_first_motion_outcome(activation),
            "FIRST_PHYSICAL_WALKING_MOTION_CONFIRMED_STOP_UNVERIFIED",
        )
        activation["status"] = "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED"
        self.assertEqual(classify_first_motion_outcome(activation), "FIRST_PHYSICAL_WALKING_PASS")
        activation["nonzero_publish_calls_returned"] = 0
        self.assertEqual(classify_first_motion_outcome(activation), "VELOCITY_COMMAND_NOT_EXECUTED")

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
