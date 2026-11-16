"""Missing ROS discovery must not hide healthy OR unhealthy native physics."""
import math
import unittest
from .stl_hand_startup import native_stand_window,build_stand_client_script

class NativeStandWindowTests(unittest.TestCase):
    def rows(self):
        return [dict(generation=1,host_monotonic_ns=1_000_000_000+i*40_000_000,
                     sim_time_s=1.+i*.04,bodies=dict(pelvis=dict(position_m=[0.,0.,.65]),
                     torso_link=dict(rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.]))) for i in range(61)]
    def test_current_advancing_window_passes(self):
        self.assertTrue(native_stand_window(self.rows())["pass"])
    def test_missing_short_frozen_repeated_gap_and_generation_fail(self):
        self.assertFalse(native_stand_window([])["pass"])
        self.assertFalse(native_stand_window(self.rows()[:3])["pass"])
        for fault in ("freeze","repeat","gap","generation"):
            rows=self.rows()
            if fault=="freeze":rows[-1]["sim_time_s"]=rows[-2]["sim_time_s"]
            if fault=="repeat":rows[-1]=rows[-2]
            if fault=="gap":rows[-1]["host_monotonic_ns"]+=300_000_000
            if fault=="generation":rows[-1]["generation"]=2
            self.assertFalse(native_stand_window(rows)["pass"],fault)
    def test_original_drift_and_tilt_still_fail(self):
        rows=self.rows();rows[-1]["bodies"]["pelvis"]["position_m"][0]=.051
        self.assertFalse(native_stand_window(rows)["pass"])
        rows=self.rows();angle=math.radians(16.)
        rows[-1]["bodies"]["torso_link"]["rotation"]=[1.,0.,0.,0.,math.cos(angle),-math.sin(angle),0.,math.sin(angle),math.cos(angle)]
        self.assertFalse(native_stand_window(rows)["pass"])
    def test_fallback_is_read_only_and_does_not_replace_received_ros(self):
        code=build_stand_client_script(observe_only=True);compile(code,"stand_readonly","exec")
        self.assertIn("if len(odom_samples)<2 or not tilt_samples:",code)
        self.assertIn("if 0.<=age<=.20",code)
        self.assertNotIn("create_publisher",code)
        self.assertNotIn("SetMcAction.Request()",code)
        self.assertNotIn("native_stand_window",build_stand_client_script())
