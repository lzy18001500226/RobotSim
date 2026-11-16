import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from scripts.research.issue46.native_runtime.native_grasp_audit import wait_for_native_trace

class TraceDrainTests(unittest.TestCase):
    def test_waits_for_original_source_stamp_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trace=root/"native_physics.jsonl"
            (root/"native_latest.json").write_text(json.dumps({"timing":{"trace_io_mode":"BOUNDED_ASYNC_FIFO"}}))
            trace.write_text(json.dumps({"host_monotonic_ns":10})+"\n")
            def persist():
                time.sleep(.03)
                with trace.open("a") as f:f.write(json.dumps({"host_monotonic_ns":20})+"\n")
            worker=threading.Thread(target=persist);worker.start()
            try:result=wait_for_native_trace(trace,20,timeout_s=.3)
            finally:worker.join()
            self.assertEqual(result["last_source_ns"],20)
            self.assertEqual([json.loads(x)["host_monotonic_ns"] for x in trace.read_text().splitlines()],[10,20])

    def test_missing_evidence_times_out_instead_of_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);trace=root/"native_physics.jsonl";trace.write_text('{"host_monotonic_ns":10}\n')
            (root/"native_latest.json").write_text(json.dumps({"timing":{"trace_io_mode":"BOUNDED_ASYNC_FIFO"}}))
            with self.assertRaisesRegex(RuntimeError,"NOT_FLUSHED"):wait_for_native_trace(trace,20,timeout_s=.02)

    def test_legacy_recording_remains_immediate(self):
        with tempfile.TemporaryDirectory() as folder:
            trace=Path(folder)/"native_physics.jsonl";trace.write_text("")
            self.assertIn("LEGACY",wait_for_native_trace(trace,20)["mode"])
