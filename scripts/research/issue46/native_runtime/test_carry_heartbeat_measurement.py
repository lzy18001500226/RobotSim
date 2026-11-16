"""Measured heartbeat regression, no ROS, commands or physics stepping."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from .native_base_micro_probe import carry_owner_fault

HERE = Path(__file__).parent


def owner_block():
    tree = ast.parse((HERE / "stl_hand_startup.py").read_text())
    literal = next(n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and "heartbeat={'mode':'MEASURED_LOAD_BEARING_CARRY_HOLD'" in n.value)
    return compile(textwrap.dedent(literal[literal.index("   if stage=='CARRY_HOLD':"):]),
                   "generated-measured-heartbeat", "exec")


def reader_block():
    tree = ast.parse((HERE / "native_base_micro_probe.py").read_text())
    sample = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "sample")
    block = next(n for n in ast.walk(sample) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == "MICRO.get('carry')"
                 and "carry_heartbeat.json" in ast.unparse(n))
    return compile(ast.Module(body=[block], type_ignores=[]), "actual-heartbeat-guard", "exec")


class CarryHeartbeatMeasurementTests(unittest.TestCase):
    now = 1_000_000_000

    def measured_row(self):
        rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        return dict(contacts=[], bodies={
            "R_omnipicker_base_link": dict(position_m=[0., 0., .8], rotation=rotation),
            "test_cylinder": dict(position_m=[0., 0., .8], rotation=rotation)})

    def publish(self, failure=None, fresh=True):
        node = NS(native_row=None, native_receipt_ns=500_000_000,
                  native_generation=1, lift_phase=dict(initial_object_in_hand_m=[0., 0., 0.]),
                  get_name=lambda: "fixture", arm_fresh=lambda: fresh)
        def refresh():
            if failure:
                raise RuntimeError(failure)
            node.native_row = self.measured_row()
            node.native_receipt_ns = 990_000_000
        node.refresh_native = mock.Mock(side_effect=refresh)
        with tempfile.TemporaryDirectory() as folder:
            actual_open, actual_replace = open, os.replace
            def local(path, *args, **kwargs):
                return actual_open(Path(folder) / Path(path).name, *args, **kwargs)
            def replace(src, dst):
                actual_replace(Path(folder) / Path(src).name, Path(folder) / Path(dst).name)
            ns = dict(self=node, stage="CARRY_HOLD", math=math, json=json, os=os,
                      time=NS(monotonic_ns=lambda: self.now),
                      CIRCLE_CONTRACT=dict(height_above_center_m=0.),
                      LIFT_CONTRACT=dict(retract_preflight={}))
            with mock.patch("builtins.open", local), mock.patch.object(os, "replace", replace):
                exec(owner_block(), ns)
            value = json.loads((Path(folder) / "carry_heartbeat.json").read_text())
        node.refresh_native.assert_called_once()
        return value

    def check(self, health, row_stamp=995_000_000):
        ns = dict(carry_owner_fault=carry_owner_fault,json=json, MICRO=dict(carry=dict(pid=os.getpid())),
                  r=dict(generation=1, host_monotonic_ns=row_stamp, **self.measured_row()), math=math,
                  time=NS(monotonic_ns=lambda: self.now),
                  Path=lambda path: NS(read_text=lambda: json.dumps(health)))
        exec(reader_block(), ns)

    def test_remeasure_after_publication_does_not_restamp_old_state(self):
        health = self.publish()
        self.assertEqual(health["native_source_host_ns"], 990_000_000)
        self.assertEqual(health["host_monotonic_ns"], self.now)
        self.check(health)

    def test_failed_physical_refresh_is_not_published_as_healthy(self):
        with self.assertRaisesRegex(RuntimeError, "POST_LIFT_SOURCE_SAFETY"):
            self.publish(failure="POST_LIFT_SOURCE_SAFETY:fixture")

    def test_stale_refresh_cannot_claim_health(self):
        with self.assertRaisesRegex(RuntimeError, "STALE_CARRY_HEARTBEAT_MEASUREMENT"):
            self.publish(fresh=False)

    def test_two_hundred_ms_deadline_remains_exact(self):
        health = self.publish()
        health["native_source_host_ns"] = self.now - 200_000_000
        self.check(health)
        health["native_source_host_ns"] -= 1
        with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED") as result:
            self.check(health)
        detail = json.loads(str(result.exception).split(":", 1)[1])
        self.assertEqual(detail["failed_checks"], ["measurement_fresh_at_publication"])
        self.assertEqual(detail["checked_host_ns"], self.now)
        self.assertGreater(detail["physical_age_s"], .2)
        self.assertAlmostEqual(detail["latest_physics_age_s"], .005)

    def test_independent_fresh_streams_do_not_sum_transport_delays(self):
        health = self.publish()
        health["host_monotonic_ns"] = self.now - 134_425_928
        health["native_source_host_ns"] = self.now - 210_986_743
        self.check(health, row_stamp=self.now - 23_239_055)

    def test_current_physical_state_and_owner_must_both_remain_fresh(self):
        health = self.publish()
        with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED"):
            self.check(health, row_stamp=self.now - 200_000_001)
        health["host_monotonic_ns"] = self.now - 500_000_001
        health["native_source_host_ns"] = self.now - 510_000_000
        with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED"):
            self.check(health)

    def test_scheduling_jitter_does_not_expire_live_owner_with_fresh_physics(self):
        for age_ns in (201502700,210164530,228606798,400000000,500000000):
            health=self.publish()
            health.update(host_monotonic_ns=self.now-age_ns,
                          native_source_host_ns=self.now-age_ns-10000000)
            self.check(health)

    def test_owner_liveness_stops_at_half_second_without_restamping(self):
        health=self.publish()
        health.update(host_monotonic_ns=self.now-500000001,
                      native_source_host_ns=self.now-510000001)
        with self.assertRaisesRegex(RuntimeError,"HEARTBEAT_UNQUALIFIED") as exc:
            self.check(health)
        detail=json.loads(str(exc.exception).split(":",1)[1])
        self.assertEqual(detail["failed_checks"],["owner_liveness"])
        self.assertEqual(detail["owner_liveness_limit_s"],.5)
        self.assertEqual(detail["physical_freshness_limit_s"],.2)
        self.assertEqual(health["host_monotonic_ns"],499999999)

    def test_current_slip_or_external_support_fails_despite_healthy_owner(self):
        health = self.publish()
        current = self.measured_row()
        current["bodies"]["test_cylinder"]["position_m"][0] = .02
        original = self.measured_row
        self.measured_row = lambda: current
        with self.assertRaisesRegex(RuntimeError, "CURRENT_CARRY_OBJECT_POSE_OR_SLIP"):
            self.check(health)
        self.measured_row = original
        current = self.measured_row()
        current["contacts"] = [dict(body1="test_cylinder", body2="table",
                                    local_force_torque=[1., 0., 0., 0., 0., 0.])]
        self.measured_row = lambda: current
        with self.assertRaisesRegex(RuntimeError, "CURRENT_CARRY_OBJECT_EXTERNAL_SUPPORT"):
            self.check(health)

    def test_invalid_owner_generation_and_unhealthy_still_stop(self):
        good = self.publish()
        for key, value in [("pid", -1), ("generation", 2), ("healthy", False),
                           ("mode", "CARRY_ABORT_LAST_COMMAND_HOLD"),
                           ("host_monotonic_ns", self.now - 500_000_001),
                           ("native_source_host_ns", self.now + 1)]:
            with self.subTest(key=key):
                health = copy.deepcopy(good)
                health[key] = value
                with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED"):
                    self.check(health)






if __name__ == "__main__":
    unittest.main()
