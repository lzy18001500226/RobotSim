"""Snapshot ordering regression; no claim of DDS or physics validation."""
import contextlib,json,textwrap,types,unittest
from pathlib import Path
from unittest.mock import Mock

SOURCE=Path(__file__).with_name("native_base_micro_probe.py")

def run_pump(*,moving=False,producer_fresh=True):
    source=SOURCE.read_text();start=source.index("        def pump(self):\n",source.index("        def start_feedback"));end=source.index("\n        def sample(self, start=None):",start)
    clock=types.SimpleNamespace(now=10.)
    class SlowFuture:
        def done(self):return True
        def result(self):clock.now=10.3;return {"status":"received"}
    class Snapshot:
        def read_text(self):return json.dumps({"generation":1,"process_pid":99,"host_monotonic_ns":int((clock.now if producer_fresh else 10.)*1e9)})
    namespace={"time":types.SimpleNamespace(monotonic=lambda:clock.now),"json":json,"Path":lambda _:Snapshot(),"raw":lambda x:x,
               "GetMcAction":types.SimpleNamespace(Request=lambda:None),"GetCurrentInputSource":types.SimpleNamespace(Request=lambda:None)}
    exec(textwrap.dedent(source[start:end]),namespace)
    node=types.SimpleNamespace(feedback_executor=object(),feedback_executor_error=None,feedback_thread=types.SimpleNamespace(is_alive=lambda:True),
        refresh_native_pose=lambda:None,telemetry_lock=contextlib.nullcontext(),pumped_epoch=0,source_epoch=0,
        pending={"get_mode":(SlowFuture(),10.)},readbacks={},events=[],moving=moving,
        latest_row={"generation":1,"process_pid":99,"host_monotonic_ns":10000000000},
        get_mode=types.SimpleNamespace(service_is_ready=lambda:False),current=types.SimpleNamespace(service_is_ready=lambda:False))
    namespace["pump"](node)
    return node,clock

class SnapshotOrderTests(unittest.TestCase):
    def test_readonly_snapshot_acquired_after_service_delay(self):
        n,clock=run_pump();self.assertAlmostEqual(n.latest_row["host_monotonic_ns"]*1e-9,clock.now)
        self.assertAlmostEqual(n.readbacks["get_mode"][0],clock.now)
        self.assertEqual(n.events[-1]["event"],"PUMP_READBACK_DELAY")
    def test_moving_command_loop_cannot_hide_overrun(self):
        with self.assertRaisesRegex(RuntimeError,"MOTION_FEEDBACK_PUMP_OVERRUN"):run_pump(moving=True)
    def test_stale_producer_is_not_restamped(self):
        n,clock=run_pump(producer_fresh=False)
        self.assertEqual(n.latest_row["host_monotonic_ns"],10000000000)
        self.assertGreater(clock.now-n.latest_row["host_monotonic_ns"]*1e-9,.20)

if __name__=="__main__":unittest.main()
