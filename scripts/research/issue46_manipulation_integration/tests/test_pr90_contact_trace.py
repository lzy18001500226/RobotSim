import unittest
try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("NumPy is available in the pinned MuJoCo runtime only") from exc

from scripts.research.issue46_manipulation_integration.pr90_contact_trace import (
    contact_force_world,
    force_on_object,
    summarize_pr90_contact_window,
    tangential_velocity_components,
)

class ContactTraceMathTests(unittest.TestCase):
    def test_contact_frame_rows_transform_force_to_world(self):
        frame=np.asarray([[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]])
        got=contact_force_world((1.0,2.0,3.0,0.0,0.0,0.0),frame.reshape(-1))
        np.testing.assert_allclose(got,(-2.0,1.0,3.0))

    def test_equal_and_opposite_object_force_sign(self):
        geom2=force_on_object((0.0,0.0,5.0),True)
        geom1=force_on_object((0.0,0.0,5.0),False)
        np.testing.assert_allclose(geom2,(0.0,0.0,5.0))
        np.testing.assert_allclose(geom1,(0.0,0.0,-5.0))

    def test_slip_projection_uses_contact_tangent_axes(self):
        frame=np.eye(3)
        self.assertEqual(tangential_velocity_components((0.0,1.0,0.0),frame.reshape(-1)),(1.0,0.0))

    def test_window_summary_reports_contact_and_motion_without_acceptance_claim(self):
        frames=[
            {"contacts":[{"role":"narrow_pad","active":True,"normal_force_N":1.0,"relative_slip_speed_m_s":0.01},
                         {"role":"wide_pad","active":True,"normal_force_N":1.0,"relative_slip_speed_m_s":0.02}],
             "right_hand_vertical_support_N":1.5,"left_hand_contact":False,"table_reaction_vertical_N":4.0,
             "object_pose_world":{"position_m":[0.0,0.0,0.0]}},
            {"contacts":[{"role":"narrow_pad","active":True,"normal_force_N":0.5,"relative_slip_speed_m_s":0.03}],
             "right_hand_vertical_support_N":0.4,"left_hand_contact":True,"table_reaction_vertical_N":5.0,
             "object_pose_world":{"position_m":[0.004,0.0,0.005]}},
        ]
        out=summarize_pr90_contact_window(frames)
        self.assertEqual(out["frame_count"],2)
        self.assertEqual(out["bilateral_pad_contact_fraction"],0.5)
        self.assertEqual(out["left_hand_contact_frames"],1)
        self.assertAlmostEqual(out["max_object_horizontal_drift_m"],0.004)
        self.assertAlmostEqual(out["max_object_vertical_displacement_m"],0.005)
        self.assertEqual(out["claim_boundary"],"Measured trace summary only; acceptance thresholds are defined by the task contract.")

if __name__=="__main__":
    unittest.main()
