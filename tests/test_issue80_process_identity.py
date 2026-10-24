from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "scripts" / "research"
sys.path.insert(0, str(RESEARCH))
import issue80_process_identity as identity  # noqa: E402
from issue80_n3_restore_m1_startup import (  # noqa: E402
    classify_mc_shutdown_log,
    extract_mc_service_identity,
)


class ProcessIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executable = "/workspace/raicom2026/mc/bin/aima-mc-app-main"
        self.digest = hashlib.sha256(b"pinned mc binary").hexdigest()
        self.record = {
            "pid": 284,
            "comm": "aima-mc-app-ma",  # Linux TASK_COMM_LEN truncates the 16-char name.
            "exe_path": self.executable,
            "exe_basename": "aima-mc-app-main",
            "exe_sha256": self.digest,
            "cwd": "/workspace/raicom2026/mc/bin",
            "argv": ["./aima-mc-app-main", "--cfg_file_path=/workspace/raicom2026/mc/config.yaml"],
            "argv0_path": self.executable,
            "start_ticks": 9001,
            "cmdline_identity_verified": True,
        }

    def test_relative_argv0_is_resolved_against_recorded_cwd(self) -> None:
        self.assertEqual(identity.resolve_argv0("./aima-mc-app-main", "/workspace/raicom2026/mc/bin"),
                         self.executable)

    def test_proc_stat_start_time_and_executable_hash_are_readable(self) -> None:
        current = identity._read_proc(os.getpid())
        self.assertIsNotNone(current)
        self.assertEqual(current["pid"], os.getpid())
        self.assertGreater(current["start_ticks"], 0)
        self.assertRegex(current["exe_sha256"], r"^[0-9a-f]{64}$")

    def test_exact_executable_and_cmdline_pass_even_when_comm_is_truncated(self) -> None:
        ok, reason = identity.verify_expected(self.record, self.executable, self.digest, 9001)
        self.assertTrue(ok)
        self.assertEqual(reason, "verified")

    def test_wrong_executable_path_hash_or_argv0_is_rejected(self) -> None:
        wrong_path = dict(self.record, exe_path="/tmp/aima-mc-app-main")
        self.assertEqual(identity.verify_expected(wrong_path, self.executable, self.digest)[1],
                         "executable_path_mismatch")
        wrong_hash = dict(self.record, exe_sha256="0" * 64)
        self.assertEqual(identity.verify_expected(wrong_hash, self.executable, self.digest)[1],
                         "executable_sha256_mismatch")
        wrong_argv = dict(self.record, argv0_path="/tmp/unrelated-wrapper", cmdline_identity_verified=False)
        self.assertEqual(identity.verify_expected(wrong_argv, self.executable, self.digest)[1],
                         "cmdline_argv0_does_not_resolve_to_executable")

    def test_reused_pid_is_not_accepted_as_the_recorded_process(self) -> None:
        ok, reason = identity.verify_expected(self.record, self.executable, self.digest, 9002)
        self.assertFalse(ok)
        self.assertEqual(reason, "pid_reused_or_process_restarted")

    def test_wait_requires_exactly_one_pinned_process(self) -> None:
        with patch.object(identity, "scan_processes", return_value=[self.record]):
            result = identity.wait_for_expected(self.executable, self.digest, timeout_s=0.01, poll_s=0.001)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["process"]["pid"], 284)
        with patch.object(identity, "scan_processes", return_value=[self.record, dict(self.record, pid=285)]):
            ambiguous = identity.wait_for_expected(self.executable, self.digest, timeout_s=0.01, poll_s=0.001)
        self.assertEqual(ambiguous["status"], "AMBIGUOUS_PROCESS_COUNT")

    def test_n3_lifecycle_definitions_are_unique_and_no_long_name_pgrep_pkills_remain(self) -> None:
        n3 = (RESEARCH / "issue80_n3_restore_m1_startup.py").read_text(encoding="utf-8")
        n2 = (RESEARCH / "issue80_n2_runtime_run.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^    def record_stage\(", n3, flags=re.MULTILINE)), 1)
        self.assertEqual(len(re.findall(r"^    def wait_after_reset_30s\(", n3, flags=re.MULTILINE)), 1)
        self.assertNotRegex(n3, r"(?m)^\s*(?:\[\"pgrep\"|\[\"pkill\"|pgrep -x aima-mc-app-main|pkill .*aima-mc-app-main)")
        self.assertNotRegex(n2, r"(?m)^\s*(?:\[\"pgrep\"|\[\"pkill\"|pgrep -x aima-mc-app-main|pkill .*aima-mc-app-main)")

    def test_mc_service_owner_is_extracted_from_runtime_registration_logs(self) -> None:
        console = (
            "[Info][270][core]Service 'ros2:/aimdk_msgs/srv/GetMcAction' is registered "
            "to ros2 rpc backend, module 'MotionControlModule', ros2 func name is '/aimdk_5Fmsgs/srv/GetMcAction'\n"
            "[Info][270][core]Service 'ros2:/aimdk_msgs/srv/GetCurrentInputSource' is registered "
            "to ros2 rpc backend, module 'MotionControlModule', ros2 func name is '/aimdk_5Fmsgs/srv/GetCurrentInputSource'\n"
        )
        inventory = "/mc_ros2_node270/describe_parameters [rcl_interfaces/srv/DescribeParameters]\n"
        process = {"pid": 270, "exe_path": self.executable, "exe_sha256": self.digest, "start_ticks": 9001}
        result = extract_mc_service_identity(console, inventory, process)
        self.assertEqual(result["status"], "MC_MODULE_SERVICE_REGISTRATIONS_OBSERVED")
        self.assertEqual(result["service_registrations"]["/aimdk_5Fmsgs/srv/GetMcAction"]["module"],
                         "MotionControlModule")
        self.assertEqual(result["observed_mc_ros2_graph_nodes"], ["/mc_ros2_node270"])

    def test_shutdown_crash_is_not_misreported_as_clean_exit(self) -> None:
        self.assertEqual(classify_mc_shutdown_log("MotionControlModule Shutdown Succeeded"),
                         "MC_MODULE_SHUTDOWN_MARKER_OBSERVED_NO_SEGFAULT_LINE")
        self.assertEqual(classify_mc_shutdown_log("MotionControlModule Shutdown Succeeded\nSegmentation fault"),
                         "MC_SHUTDOWN_SEGFAULT_OBSERVED")


if __name__ == "__main__":
    unittest.main()
