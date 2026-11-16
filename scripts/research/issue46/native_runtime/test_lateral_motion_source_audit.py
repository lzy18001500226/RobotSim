import math,unittest
from scripts.research.issue46.native_runtime.lateral_motion_source_audit import project_displacement,message_summary,ideal_forward_s_curve
class RecordedMotionAuditTests(unittest.TestCase):
    def test_body_left_is_not_always_world_y(self):
        d=project_displacement({"x":0.,"y":0.,"yaw":math.pi/2},{"x":-.12,"y":0.,"yaw":math.pi/2})
        self.assertAlmostEqual(d["initial_body_left_m"],.12);self.assertAlmostEqual(d["initial_body_forward_m"],0.)
    def test_right_motion_stays_right(self):
        d=project_displacement({"x":0.,"y":0.,"yaw":0.},{"x":.014,"y":-.121,"yaw":0.})
        self.assertAlmostEqual(d["initial_body_left_m"],-.121)
    def test_zero_packet_between_motion_messages_is_visible(self):
        rows=[{"t":i,"raw":{"forward_velocity":v}} for i,v in enumerate((.12,0.,.12))]
        r=message_summary(rows,dds=True);self.assertEqual(r["zero_packets_inside_nonzero_span"],1);self.assertFalse(r["is_mc_receipt_ack"])
    def test_ideal_s_curve_has_large_forward_cost(self):
        d=ideal_forward_s_curve(.12,.4,math.acos(.8));self.assertAlmostEqual(d["left_m"],.12);self.assertAlmostEqual(d["forward_m"],.36)
    def test_missing_commands_are_not_fabricated(self):
        self.assertEqual(message_summary([])["nonzero_packets"],0)
if __name__=="__main__":unittest.main()
