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
        ns = dict(json=json, MICRO=dict(carry=dict(pid=os.getpid())),
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
        health["host_monotonic_ns"] = self.now - 200_000_001
        health["native_source_host_ns"] = self.now - 210_000_000
        with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED"):
            self.check(health)

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
                           ("host_monotonic_ns", self.now - 200_000_001),
                           ("native_source_host_ns", self.now + 1)]:
            with self.subTest(key=key):
                health = copy.deepcopy(good)
                health[key] = value
                with self.assertRaisesRegex(RuntimeError, "HEARTBEAT_UNQUALIFIED"):
                    self.check(health)


class CarryOwnerSnapshotRereadTests(unittest.TestCase):
    def run_reread(self, update_at=None, *, healthy=True, generation=1, native_age_s=.037):
        from scripts.research.issue46.native_runtime import native_base_micro_probe as m
        clock=NS(now=1.)
        def now_ns():return round(clock.now*1e9)
        old=dict(mode="MEASURED_LOAD_BEARING_CARRY_HOLD",healthy=healthy,
                 pid=19,generation=generation,host_monotonic_ns=789835470)
        def read():
            if update_at is not None and clock.now>=1.+update_at:
                return json.dumps(dict(old,host_monotonic_ns=now_ns()))
            return json.dumps(old)
        events=[]
        health_path=NS(read_text=read)
        native_path=NS(read_text=lambda:json.dumps(dict(
            generation=1,host_monotonic_ns=now_ns()-round(native_age_s*1e9))))
        def sleep(duration):clock.now+=duration
        with mock.patch.object(m,"time",NS(monotonic=lambda:clock.now,
                monotonic_ns=now_ns,sleep=sleep)):
            result=m.await_recent_carry_owner(health_path,native_path,19,1,events,1.)
        return result,events,clock.now-1.

    def test_recorded_210ms_snapshot_waits_for_next_publication(self):
        # 080004Z: at failure the next observed owner sample was 28.5ms away.
        result,events,elapsed=self.run_reread(.029)
        self.assertGreater(result["host_monotonic_ns"],1_000_000_000)
        self.assertGreaterEqual(elapsed,.029)
        self.assertLessEqual(elapsed,.05)
        self.assertLessEqual(events[0]["final_age_s"],.2)

    def test_missing_owner_update_never_refreshes_or_accepts_old_stamp(self):
        result,events,elapsed=self.run_reread()
        self.assertEqual(result["host_monotonic_ns"],789835470)
        self.assertAlmostEqual(elapsed,.05)
        self.assertGreater(events[0]["final_age_s"],.2)
        health=CarryHeartbeatMeasurementTests().publish()
        health.update(result,native_source_host_ns=780000000)
        with self.assertRaisesRegex(RuntimeError,"HEARTBEAT_UNQUALIFIED"):
            CarryHeartbeatMeasurementTests().check(health)

    def test_failed_owner_does_not_wait(self):
        result,events,elapsed=self.run_reread(.01,healthy=False)
        self.assertEqual(elapsed,0.)
        self.assertFalse(result["healthy"])
        self.assertEqual(events,[])

    def test_wrong_generation_does_not_wait(self):
        result,events,elapsed=self.run_reread(.01,generation=2)
        self.assertEqual(elapsed,0.)
        self.assertEqual(result["generation"],2)

    def test_stale_physics_is_not_masked_by_snapshot_wait(self):
        result,events,elapsed=self.run_reread(.01,native_age_s=.200000001)
        self.assertEqual(elapsed,0.)
        self.assertEqual(result["host_monotonic_ns"],789835470)


class CarryFirstPumpTests(unittest.TestCase):
    def test_actual_pump_reads_first_native_generation_before_owner_check(self):
        import threading
        from scripts.research.issue46.native_runtime import native_base_micro_probe as m
        tree=ast.parse((HERE/"native_base_micro_probe.py").read_text())
        method=next(n for n in ast.walk(tree)
                    if isinstance(n,ast.FunctionDef) and n.name=="pump")
        compiled=compile(ast.Module(body=[method],type_ignores=[]),"actual-first-pump","exec")
        stamp=1_000_000_000
        fresh=dict(generation=1,process_pid=101,host_monotonic_ns=stamp)
        health=dict(mode="MEASURED_LOAD_BEARING_CARRY_HOLD",healthy=True,
                    pid=19,generation=1,host_monotonic_ns=stamp)
        paths=lambda p:NS(read_text=lambda:json.dumps(health if "heartbeat" in p else fresh))
        node=NS(latest_row=None,refresh_native_pose=lambda:None,feedback_executor=None,
                telemetry_lock=threading.RLock(),pumped_epoch=0,source_epoch=0,
                pending={},readbacks={},events=[],moving=False,
                get_mode=NS(service_is_ready=lambda:False),
                current=NS(service_is_ready=lambda:False))
        actual_checked=[]
        def owner_check(*args):
            actual_checked.append(args[3])
            self.assertEqual(args[3],1)
            self.assertIsNone(node.latest_row)
        ns=dict(time=NS(monotonic=lambda:1.),MICRO=dict(carry=dict(pid=19)),
                drain_ready_callbacks=lambda *args:None,rclpy=None,
                GetMcAction=NS(Request=object),GetCurrentInputSource=NS(Request=object),
                Path=paths,json=json,await_recent_carry_owner=owner_check)
        exec(compiled,ns)
        ns["pump"](node)
        self.assertEqual(actual_checked,[1])
        self.assertEqual(node.latest_row,fresh)


if __name__ == "__main__":
    unittest.main()
