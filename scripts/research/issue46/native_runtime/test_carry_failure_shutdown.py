"""Offline boundary regressions: failed carry must not leave physics running."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
from scripts.research.issue46.native_runtime import native_grasp_audit as audit


class FailedCarryParentTests(unittest.TestCase):
    def run_parent(self, nav, stop_raises=False):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            now = time.monotonic_ns()
            ready = dict(schema="robotsim.x2.carry-ready.v1", pid=19, physical_lift_pass=True, post_lift_retract_pass=True,
                         generation=1, host_monotonic_ns=now)
            for name, value in {
                "carry_ready.json": ready,
                "carry_heartbeat.json": dict(pid=19, host_monotonic_ns=now),
                "native_tracking_carry_handoff.json": dict(
                    classification="BODY_RELATIVE_WORLD_UPRIGHT_IK_CARRY",
                    physical_hold_pass=True, generation=1),
            }.items():
                (out / name).write_text(json.dumps(value))
            (out / "native_micro_stop_contract.py").write_text("# fixture")
            args = NS(carry_goal_world_xy=[.015, .9], carry_timeout_s=5.,
                      carry_sdk_minimum_turn=False, carry_native_action_smoothing=False,
                      carry_native_step_mode=False)
            plan = {"lift_preflight": {"retract_preflight": dict(
                pass_result=True, robot_bodies=["pelvis"], source_motion_limits={}, carry_lease_s=30.)}}
            result = dict(gates={"OFFICIAL_CMD_VEL_BRIDGE": "PASS"}, commands_not_sent=[])
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if "stop" in command:
                    if stop_raises:
                        raise subprocess.TimeoutExpired(command, 7.)
                elif any("native_carry_route.py" in str(x) for x in command):
                    (out / "native_carry_route_result.json").write_text(json.dumps(nav))
                return subprocess.CompletedProcess(command, 0, "")
            def write(path, value):
                return path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
            with mock.patch.object(parent, "run", run), \
                 mock.patch.object(parent.pg, "write_new_text", side_effect=write), \
                 mock.patch.object(parent.pg, "write_new_bytes", side_effect=write), \
                 mock.patch.object(parent, "qualify_static_carry_hold", return_value={"pass": True}), \
                 mock.patch.object(audit, "require_fresh_sample", return_value={"generation": 1}), \
                 mock.patch.object(audit, "wait_for_native_trace", side_effect=AssertionError("slow audit reached before failed runtime stopped")):
                with self.assertRaisesRegex(RuntimeError, "CARRY_NAVIGATION_UNQUALIFIED_STOP"):
                    parent.execute_grasp_with_carry("owned-fixture-only", "", out, args, plan, result, 48)
            marker = json.loads((out / "native_carry_abort.json").read_text())
            self.assertFalse(marker["physical_stop_qualified"])
            self.assertEqual([c for c in calls if "stop" in c],
                             [["docker", "stop", "--time", "1", "owned-fixture-only"]])
            self.assertEqual(result["gates"]["PHYSICAL_TABLE_PLACEMENT"], "NOT_RUN")
            self.assertEqual(result["gates"]["TWO_TABLE_TRANSFER_EXECUTION"], "FAIL")
            self.assertFalse((out / "carry_release").exists())
            return result

    def test_clock_abort_stops_owned_runtime_before_evidence_drain(self):
        self.run_parent(dict(pass_result=False, failure="CLOCK_RECOVERY_EXHAUSTED",
                             abort_post_stop_composite={"pass": False}))

    def test_previous_successful_stop_cannot_qualify_later_fault(self):
        self.run_parent(dict(pass_result=False, failure="SOURCE_JOINT_SPEED:left_knee_joint",
                             post_stop_composite={"pass": True}))

    def test_stop_timeout_still_aborts_and_preserves_original_route_failure(self):
        result = self.run_parent(dict(pass_result=False, failure="CARRY_ROUTE_NO_RESULT"), True)
        self.assertEqual(result["failure"]["route_failure"], "CARRY_ROUTE_NO_RESULT")
        self.assertIn("error", result["carry_failure_shutdown"])


class QualifiedCarryTests(unittest.TestCase):
    def test_success_and_qualified_stops_keep_existing_audit_path(self):
        for nav in [dict(pass_result=True),
                    dict(pass_result=False, failure="CARRY_GOAL_OR_STOP_NOT_QUALIFIED",
                         post_stop_composite={"pass": True}),
                    dict(pass_result=False, failure="FINAL_DOCK_TURN_FOOTPRINT",
                         abort_post_stop_composite={"pass": True})]:
            with self.subTest(nav=nav), mock.patch.object(parent, "run") as run:
                parent.abort_unstopped_carry("unused", Path("/unused"), nav, {})
                run.assert_not_called()


class FailedCarryDisplayTests(unittest.TestCase):
    @staticmethod
    def status_function():
        # Load the pure status function without importing GLFW or opening a GUI.
        source = Path(parent.__file__).parents[1] / "grasp/native_live_review_windows.py"
        tree = ast.parse(source.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "review_status")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["review_status"]

    def test_failure_replaces_live_and_hold_labels_without_claiming_stop(self):
        status = self.status_function()
        title, phase = status(False, True, {"route_failure": "CLOCK_RECOVERY_EXHAUSTED"})
        self.assertEqual(title, "RUN FAILED - TEARDOWN REQUESTED")
        self.assertIn("CLOCK_RECOVERY_EXHAUSTED", phase)
        self.assertNotIn("CARRY_HOLD", phase)
        self.assertEqual(status(True, False, {"gate": "STOP_UNQUALIFIED"})[0],
                         "RUN FAILED - LAST ACTUAL FRAME")

    def test_normal_live_stale_and_completed_display_unchanged(self):
        status = self.status_function()
        self.assertEqual(status(False, True, None), ("LIVE WSL NATIVE PHYSICS", None))
        self.assertEqual(status(False, False, None)[0], "WAITING FOR FRESH NATIVE FRAME")
        self.assertEqual(status(True, False, None)[0], "RUN ENDED - LAST ACTUAL FRAME")


if __name__ == "__main__":
    unittest.main()
