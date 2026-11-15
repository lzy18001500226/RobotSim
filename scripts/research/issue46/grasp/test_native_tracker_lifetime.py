"""Tracker process-lifetime regressions; actual file loop, fake clock, no physics."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np
from scripts.research.issue46.grasp import stl_approved_circle_ik as tracker


class TrackerLifetimeTests(unittest.TestCase):
    def run_loop(self, phase_at, *, stop_after=90., stale_after=None,
                 generation_after=None, compute_delay=0., cycle_budget=0.):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "native_latest.json"
            output = root / "native_tracking_target.json"
            phase_path = root / "native_tracking_phase.json"
            reference = root / "reference.json"
            start = 1000.
            clock = types.SimpleNamespace(now=start)
            reference.write_text(json.dumps({
                "scene": {"path": "unused.xml"},
                "native_no_contact_gain_cycle_budget_s": cycle_budget,
                "initial_state": {"arm_positions_rad": dict.fromkeys(tracker.ik.ARM, 0.)},
            }))

            def ns(seconds):
                return round(seconds * 1e9)

            def inputs():
                elapsed = clock.now - start
                source = clock.now if stale_after is None else min(clock.now, start + stale_after)
                latest.write_text(json.dumps({
                    "host_monotonic_ns": ns(source),
                    "generation": 2 if generation_after is not None and elapsed >= generation_after else 1,
                }))
                phase = phase_at(elapsed)
                if phase is not None:
                    phase_path.write_text(json.dumps(phase))
                elif phase_path.exists():
                    phase_path.unlink()
                if elapsed >= stop_after:
                    output.with_suffix(".stop").write_text("test owner stopped")

            def sleep(seconds):
                clock.now += max(seconds, 1e-6)
                inputs()

            compute_calls=0
            def solve(model, row, reference, seed, phase):
                nonlocal compute_calls
                delay=compute_delay[min(compute_calls,len(compute_delay)-1)] if isinstance(compute_delay,list) else compute_delay
                compute_calls+=1;clock.now+=delay
                inputs()  # Native observer runs independently of this solver.
                return np.zeros(len(tracker.ik.ARM)), {}

            inputs()
            failure = None
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(tracker, "mujoco", types.SimpleNamespace(
                    MjModel=types.SimpleNamespace(from_xml_path=lambda path: object()))))
                stack.enter_context(mock.patch.object(tracker, "solve_tracking_endpoint", solve))
                stack.enter_context(mock.patch.object(time, "monotonic", lambda: clock.now))
                stack.enter_context(mock.patch.object(time, "monotonic_ns", lambda: ns(clock.now)))
                stack.enter_context(mock.patch.object(time, "sleep", sleep))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                try:
                    tracker.track_native(str(reference), str(latest), str(output))
                except RuntimeError as exc:
                    failure = str(exc)
            return {
                "failure": failure,
                "elapsed_s": clock.now - start,
                "target": json.loads(output.read_text()) if output.exists() else None,
                "trace": json.loads(output.with_suffix(".trace.json").read_text()),
                "recoveries": json.loads(output.with_suffix(".deadline-recoveries.json").read_text()),
            }

    @staticmethod
    def phase(stage, at, **extra):
        result = {"phase": stage, "created_host_ns": round((1000. + at) * 1e9)}
        result.update(extra)
        return result

    @classmethod
    def grasp_phases(cls, elapsed):
        if elapsed < 30.:
            return None
        if elapsed < 34.:
            return cls.phase("ARRIVAL_HOLD", 30.)
        if elapsed < 37.:
            return cls.phase("CLOSING", 34.)
        if elapsed < 41.:
            return cls.phase("LIFTING", 37.)
        if elapsed < 46.:
            return cls.phase("RETRACTING", 41., pre_retract_physical_lift_pass=True,
                             retract_from_hold_end_ns=round(1040.95 * 1e9))
        return cls.phase("CARRY_HOLD", 46., carry_lease_s=10.,
                         post_retract_physical_hold_pass=True,
                         carry_source_speed_rad_s=[1.] * len(tracker.ik.ARM))

    def test_valid_retract_and_carry_continue_past_old_45_second_exit(self):
        result = self.run_loop(self.grasp_phases, stop_after=47.)
        self.assertIsNone(result["failure"])
        self.assertGreaterEqual(result["elapsed_s"], 47.)
        self.assertTrue(any(row["phase"] == "RETRACTING" and
                            row["created_host_ns"] > 1045.1e9 for row in result["trace"]))
        self.assertTrue(any(row["phase"] == "CARRY_HOLD" for row in result["trace"]))

    def test_unphased_startup_expiry_is_explicit_failure_not_success_exit(self):
        result = self.run_loop(lambda elapsed: None)
        self.assertIn("NATIVE_TRACKING_LIFETIME_EXPIRED", result["failure"] or "")
        self.assertLess(result["elapsed_s"], 45.2)
        self.assertFalse(result["target"]["pass"])

    def test_precarry_total_cap_survives_repeated_phase_timestamp_renewal(self):
        result = self.run_loop(lambda elapsed: self.phase("CLOSING", elapsed))
        self.assertIn("NATIVE_TRACKING_LIFETIME_EXPIRED", result["failure"] or "")
        # Startup 45 + motion windows 12 + 6 + 9 + 6 + two existing 5s holds.
        self.assertGreaterEqual(result["elapsed_s"], 88.)
        self.assertLess(result["elapsed_s"], 88.2)

    def test_existing_six_second_closing_lease_is_not_extended(self):
        result = self.run_loop(lambda elapsed: self.phase("CLOSING", 0.))
        self.assertTrue(result["failure"].startswith("CLOSURE_PHASE_EXPIRED:"))
        self.assertLess(result["elapsed_s"], 6.2)

    def test_stale_source_after_old_cutoff_still_stops(self):
        result = self.run_loop(self.grasp_phases, stale_after=45.)
        self.assertEqual(result["failure"], "STALE_NATIVE_TRACKING_SOURCE")
        self.assertGreater(result["elapsed_s"], 45.2)
        self.assertLess(result["elapsed_s"], 45.4)

    def test_persistently_expired_computation_still_stops(self):
        result = self.run_loop(lambda elapsed: None, compute_delay=.201)
        self.assertEqual(result["failure"], "NATIVE_TRACKING_COMPUTE_DEADLINE")

    def test_one_expired_result_is_discarded_then_fresh_result_published(self):
        result=self.run_loop(lambda elapsed:None,compute_delay=[.205,.003],stop_after=.6)
        self.assertIsNone(result["failure"])
        self.assertEqual([x["event"] for x in result["recoveries"]],
                         ["EXPIRED_IK_RESULT_DISCARDED","FRESH_IK_RESULT_RECOVERED"])
        self.assertTrue(result["trace"])
        self.assertTrue(all(0<=r["created_host_ns"]-r["source_host_ns"]<=200000000 for r in result["trace"]))

    def test_carry_recovery_restarts_derivative_from_fresh_sample(self):
        phase=self.phase("CARRY_HOLD",0.,carry_lease_s=10.,
                         post_retract_physical_hold_pass=True,
                         carry_source_speed_rad_s=[1.]*len(tracker.ik.ARM))
        result=self.run_loop(lambda elapsed:phase,compute_delay=[0.,.205,.003],stop_after=.6)
        self.assertIsNone(result["failure"])
        self.assertEqual(result["recoveries"][-1]["event"],"FRESH_IK_RESULT_RECOVERED")
        self.assertTrue(all(0<=r["created_host_ns"]-r["source_host_ns"]<=200000000 for r in result["trace"]))

    def test_compute_recovery_cannot_hide_generation_change(self):
        result=self.run_loop(lambda elapsed:None,compute_delay=[.205,.003],generation_after=.2)
        self.assertEqual(result["failure"],"NATIVE_TRACKING_GENERATION_CHANGED")
        self.assertFalse(result["trace"])

    def test_compute_recovery_cannot_publish_after_50ms_budget(self):
        result=self.run_loop(lambda elapsed:None,compute_delay=[.205,.051],stop_after=1.)
        self.assertEqual(result["failure"],"NATIVE_TRACKING_COMPUTE_DEADLINE")
        self.assertFalse(result["trace"])

    def test_generation_change_still_stops(self):
        result = self.run_loop(lambda elapsed: None, generation_after=.5)
        self.assertEqual(result["failure"], "NATIVE_TRACKING_GENERATION_CHANGED")

    def test_expired_carry_lease_still_stops(self):
        result = self.run_loop(self.grasp_phases)
        self.assertTrue(result["failure"].startswith("CLOSURE_PHASE_EXPIRED:"))
        self.assertGreater(result["elapsed_s"], 56.)
        self.assertLess(result["elapsed_s"], 56.2)

    def test_retract_motion_and_existing_measured_hold_have_separate_budgets(self):
        # Real070857Z failure: 4s motion plus measured hold exceeded6s.
        phase=self.phase("RETRACTING",0.,pre_retract_physical_lift_pass=True,
                         retract_from_hold_end_ns=999950000000)
        result=self.run_loop(lambda elapsed:phase,stop_after=7.)
        self.assertIsNone(result["failure"])
        result=self.run_loop(lambda elapsed:phase,stop_after=12.)
        self.assertTrue(result["failure"].startswith("CLOSURE_PHASE_EXPIRED:"))
        self.assertGreater(result["elapsed_s"],11.)
        self.assertLess(result["elapsed_s"],11.2)
        self.assertIn('"phase": "RETRACTING"',result["failure"])

    def test_existing_hold_timeout_matches_tracker_accounting(self):
        import inspect
        from scripts.research.issue46.native_runtime.stl_hand_startup import hold_native_simulation
        hold_budget=inspect.signature(hold_native_simulation).parameters["wall_timeout_s"].default
        self.assertEqual(hold_budget,5.)
        self.assertEqual(tracker.tracking_phase_limit_s({"phase":"LIFTING"}),9.+hold_budget)
        self.assertEqual(tracker.tracking_phase_limit_s({"phase":"RETRACTING"}),6.+hold_budget)

    def test_existing_nine_second_cycle_budget_stays_scoped_to_startup(self):
        result = self.run_loop(lambda elapsed: None, cycle_budget=9.)
        self.assertIn("NATIVE_TRACKING_LIFETIME_EXPIRED", result["failure"] or "")
        self.assertGreater(result["elapsed_s"], 54.)
        self.assertLess(result["elapsed_s"], 54.2)

    def test_recorded_failure_is_still_inside_existing_retract_lease(self):
        phase = {
            "phase": "RETRACTING", "created_host_ns": 27313585829577,
            "retract_from_hold_end_ns": 27313578332715,
            "pre_retract_physical_lift_pass": True,
        }
        # Exact failing observation: source timestamp + reported source age.
        now_ns = 27319092289779 + 226740169
        old_deadline = 27274.062848381 + 45.
        self.assertGreater(now_ns * 1e-9, old_deadline)
        self.assertTrue(tracker.tracking_phase_is_fresh(phase, now_ns))
        self.assertLess(now_ns * 1e-9, tracker.tracking_lifetime_deadline(old_deadline, phase))

    @staticmethod
    def owner_target_reader():
        from scripts.research.issue46.native_runtime.test_native_grasp_launch import PostLiftRetractTests
        source = PostLiftRetractTests().generated_retract({"carry_lease_s": 115.})
        method = next(node for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.FunctionDef) and node.name == "tracking_target")
        namespace = {"json": json, "time": time}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "generated_owner", "exec"), namespace)
        return namespace["tracking_target"]

    def test_owner_receives_explicit_lifetime_failure_instead_of_stale_target(self):
        result = self.run_loop(lambda elapsed: None)
        reader = self.owner_target_reader()
        with mock.patch("builtins.open", return_value=io.StringIO(json.dumps(result["target"]))):
            with self.assertRaisesRegex(RuntimeError,
                                        "NATIVE_CIRCLE_TRACKER_FAILED:.*NATIVE_TRACKING_LIFETIME_EXPIRED"):
                reader(types.SimpleNamespace())

    def test_owner_200ms_target_freshness_guard_is_unchanged(self):
        reader = self.owner_target_reader()
        target = {"pass": True, "source_host_ns": 999799000000,
                  "created_host_ns": 1000000000000, "generation": 1}
        owner = types.SimpleNamespace(native_generation=1, native_receipt_ns=1000000000000)
        with (
            mock.patch("builtins.open", return_value=io.StringIO(json.dumps(target))),
            mock.patch.object(time, "monotonic_ns", return_value=1000000000000),
        ):
            with self.assertRaisesRegex(RuntimeError, "STALE_NATIVE_CIRCLE_TRACKING_TARGET"):
                reader(owner)

    def test_owner_stop_retains_trace_without_error(self):
        result = self.run_loop(lambda elapsed: None, stop_after=.5)
        self.assertIsNone(result["failure"])
        self.assertGreater(len(result["trace"]), 0)
        self.assertTrue(result["target"]["pass"])


if __name__ == "__main__":
    unittest.main()
