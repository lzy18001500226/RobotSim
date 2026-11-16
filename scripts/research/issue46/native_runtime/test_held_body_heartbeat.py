"""Exercise the actual generated held-body reader; no simulator or actuation."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from .native_base_micro_probe import carry_owner_fault
from . import native_body_height as body


class HeldBodyHeartbeatTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse(body.build_script({},[],label="body_restore_carry"))
        node=next(n for n in ast.walk(tree)
                  if isinstance(n,ast.FunctionDef) and n.name=="held_sample")
        self.row=dict(generation=1,host_monotonic_ns=965_000_000)
        self.health=dict(mode="MEASURED_LOAD_BEARING_CARRY_HOLD",healthy=True,
                         pid=12,generation=1,host_monotonic_ns=750_000_000,
                         native_source_host_ns=740_000_000)
        ns=dict(carry_owner_fault=carry_owner_fault,ready=dict(pid=12),
                n=NS(latest_row=self.row),original_sample=lambda *a,**k:"sample",
                time=NS(monotonic_ns=lambda:1_000_000_000),json=json,
                Path=lambda path:NS(read_text=lambda:json.dumps(self.health)))
        exec(compile(ast.Module(body=[node],type_ignores=[]),"held-body","exec"),ns)
        self.sample=ns["held_sample"]

    def test_owner_250ms_with_fresh_physics_is_not_a_sensor_expiry(self):
        self.assertEqual(self.sample(),"sample")

    def test_dead_owner_stale_physics_and_unmeasured_publication_remain_failures(self):
        for stream,key,value,reason in (
            (self.health,"host_monotonic_ns",499_999_999,"owner_liveness"),
            (self.row,"host_monotonic_ns",799_999_999,"physical_fresh"),
            (self.health,"native_source_host_ns",500_000_000,"measurement_fresh_at_publication"),
            (self.health,"pid",99,"owner_pid"),
            (self.health,"generation",2,"generation"),
            (self.health,"healthy",False,"healthy"),
            (self.health,"mode","CARRY_ABORT_LAST_COMMAND_HOLD","mode")):
            old=stream[key]
            with self.subTest(reason=reason):
                stream[key]=value
                with self.assertRaisesRegex(RuntimeError,reason):self.sample()
                stream[key]=old

    def test_original_owner_failure_is_preserved_in_body_error(self):
        self.health.update(healthy=False,failure="STALE_NATIVE_CIRCLE_TRACKING_TARGET")
        with self.assertRaisesRegex(RuntimeError,"STALE_NATIVE_CIRCLE_TRACKING_TARGET"):
            self.sample()

    def test_both_readers_reference_one_policy(self):
        import inspect
        from .native_base_micro_probe import run_probe
        self.assertIn("carry_owner_fault(health,r,",inspect.getsource(run_probe))
        self.assertIn("carry_owner_fault(health,n.latest_row,",inspect.getsource(body.run_height_probe))


if __name__=="__main__":
    unittest.main()
