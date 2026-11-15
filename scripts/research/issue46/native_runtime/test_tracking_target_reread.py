"""Generated owner regression; no ROS graph, commands or physics."""
import ast
import io
import json
import math
import os
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from .test_pregrasp_feedback import generated


class TrackingTargetRereadTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(generated())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "tracking_target")
        self.clock = NS(ns=1_000_000_000)
        self.node = NS(native_generation=1, native_receipt_ns=990_000_000,
                       endpoint=None, tracking_metrics={})
        self.node.refresh_native = mock.Mock()
        self.output = []
        self.spin = mock.Mock(side_effect=self.tick)
        namespace = dict(time=NS(monotonic_ns=lambda: self.clock.ns),
                         os=os, json=json, math=math, RIGHT=("joint",),
                         rclpy=NS(spin_once=self.spin), print=lambda *a, **k: self.output.append(a))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "generated-target", "exec"),
             namespace)
        self.method = namespace["tracking_target"]

    def tick(self, *args, **kwargs):
        self.clock.ns += 2_000_000
        self.node.native_receipt_ns = self.clock.ns - 10_000_000

    def target(self, age_ns=205_000_000, **overrides):
        return dict(pass_=True) | {
            "pass": True, "source_host_ns": self.clock.ns-age_ns,
            "created_host_ns": self.clock.ns-134_000_000,
            "generation": 1, "right_target_rad": {"joint": .3}, "metrics": {}
        } | overrides

    def run_target(self, reader, retry=.05, stop=False):
        def read(*args, **kwargs):
            return io.StringIO(json.dumps(reader()))
        with mock.patch("builtins.open", read), mock.patch.object(os.path, "exists",
                                                                return_value=stop):
            return self.method(self.node, retry_s=retry)

    def test_fresh_target_has_no_wait(self):
        self.assertEqual(self.run_target(lambda: self.target(10_000_000)), {"joint": .3})
        self.spin.assert_not_called()

    def test_recorded_205ms_stale_read_recovers_after_16ms_without_old_target(self):
        def read():
            return self.target(10_000_000 if self.clock.ns >= 1_016_000_000 else 205_000_000)
        self.assertEqual(self.run_target(read), {"joint": .3})
        self.assertEqual(self.clock.ns, 1_016_000_000)
        self.assertEqual(self.node.refresh_native.call_count, 8)
        detail = json.loads(self.output[0][0])
        self.assertTrue(detail["fresh_tracking_target_recovered"])
        self.assertLessEqual(detail["accepted_source_age_s"], .2)

    def test_persistent_expiry_fails_at_50ms_without_setting_endpoint(self):
        with self.assertRaisesRegex(RuntimeError, "STALE_NATIVE_CIRCLE_TRACKING_TARGET"):
            self.run_target(lambda: self.target())
        self.assertEqual(self.clock.ns, 1_050_000_000)
        self.assertIsNone(self.node.endpoint)

    def test_default_no_retry_and_exact_200ms_acceptance(self):
        self.run_target(lambda: self.target(200_000_000), retry=0.)
        self.node.endpoint = None
        with self.assertRaisesRegex(RuntimeError, "STALE_NATIVE_CIRCLE_TRACKING_TARGET"):
            self.run_target(lambda: self.target(200_000_001), retry=0.)
        self.assertIsNone(self.node.endpoint)

    def test_wrong_generation_future_or_failed_tracker_never_wait(self):
        for row in (self.target(generation=2), self.target(-1), self.target(**{"pass": False})):
            with self.subTest(row=row):
                with self.assertRaises(RuntimeError):
                    self.run_target(lambda: row)
        self.spin.assert_not_called()

    def test_physical_fault_or_human_stop_interrupts_reread(self):
        self.node.refresh_native.side_effect = RuntimeError("PHYSICAL_FAULT")
        with self.assertRaisesRegex(RuntimeError, "PHYSICAL_FAULT"):
            self.run_target(lambda: self.target())
        self.node.refresh_native.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "HUMAN_REVIEW_STOP"):
            self.run_target(lambda: self.target(), stop=True)

    def test_recovery_after_budget_is_not_accepted(self):
        self.spin.side_effect = lambda *a, **k: setattr(self.clock, "ns", 1_051_000_000)
        def read():
            return self.target(10_000_000 if self.clock.ns > 1_000_000_000 else 205_000_000)
        with self.assertRaisesRegex(RuntimeError, "STALE_NATIVE_CIRCLE_TRACKING_TARGET"):
            self.run_target(read)
        self.assertIsNone(self.node.endpoint)

    def test_nonfinite_fresh_target_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, "NONFINITE_NATIVE_CIRCLE_TARGET"):
            self.run_target(lambda: self.target(10_000_000, right_target_rad={"joint": float("nan")}))


if __name__ == "__main__":
    unittest.main()
