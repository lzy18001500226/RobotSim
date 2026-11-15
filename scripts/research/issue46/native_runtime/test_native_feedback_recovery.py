import contextlib
import io
import json
import unittest
from unittest.mock import patch
from scripts.research.issue46.native_runtime.native_feedback_recovery import read_fresh_native_snapshot


class FreshSnapshotTests(unittest.TestCase):
    def run_read(self, *, recover=True, generation=1, engine=337, future=False):
        clock=[1.]
        def raw():
            source=clock[0] if recover and clock[0]>=1.02 else .780806116
            if future:source=clock[0]+.001
            return (json.dumps(dict(host_monotonic_ns=round(source*1e9),generation=generation,engine_version=engine))+"\n").encode()
        with patch("pathlib.Path.read_bytes",side_effect=raw), patch("time.monotonic_ns",side_effect=lambda:round(clock[0]*1e9)), patch("time.sleep",side_effect=lambda s:clock.__setitem__(0,clock[0]+s)),contextlib.redirect_stdout(io.StringIO()):
            return read_fresh_native_snapshot("unused",generation=1),clock[0]

    def test_219ms_snapshot_is_not_used_then_real_new_snapshot_is_accepted(self):
        row,t=self.run_read()
        self.assertGreaterEqual(t,1.02)
        self.assertLess(t,1.05)
        self.assertEqual(row["host_monotonic_ns"],round(t*1e9))

    def test_persistent_staleness_stops_with_original_age_and_wait(self):
        with self.assertRaisesRegex(RuntimeError,"NATIVE_SNAPSHOT_REACQUISITION_EXPIRED"):
            self.run_read(recover=False)

    def test_generation_changes_are_not_retried(self):
        with self.assertRaisesRegex(RuntimeError,"NATIVE_SNAPSHOT_GENERATION_CHANGED"):
            self.run_read(generation=2)

    def test_engine_changes_are_not_retried(self):
        with self.assertRaisesRegex(RuntimeError,"NATIVE_SNAPSHOT_ABI_CHANGED"):
            self.run_read(engine=336)

    def test_future_timestamps_are_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError,"NATIVE_SNAPSHOT_FROM_FUTURE"):
            self.run_read(future=True)
