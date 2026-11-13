"""Focused acceptance checks for independent physical-grasp records."""
import copy,unittest
from scripts.research.issue46.grasp.table_edge_grasp_trials import summarize,REQUIRED
class GraspAcceptanceTests(unittest.TestCase):
    def sample(self):
        return {"gates":dict.fromkeys(REQUIRED,"PASS"),"physical_lift_evidence":{"pass":True,"minimum_lift_m":.059,"mean_pad_vertical_force_n":1.962,"max_table_force_n":0.,"minimum_pad_normal_force_n":{"a":14.,"b":14.},"terminal_hold_sim_s":1.02},"carry_static_hold":{"sim_span_s":8.02}}
    def test_physical_hold_pass_does_not_claim_transfer(self):
        r=summarize(self.sample());self.assertEqual(r["status"],"PASS");self.assertEqual(r["full_transfer"],"NOT_RUN")
    def test_missing_forces_do_not_become_zero_or_pass(self):
        d=self.sample();del d["physical_lift_evidence"]["mean_pad_vertical_force_n"];r=summarize(d);self.assertEqual(r["status"],"FAIL");self.assertIn("upward_hand_support_n",r["missing_evidence"])
    def test_runtime_failure_retains_measured_lift(self):
        d=self.sample();d["failure"]={"gate":"SOURCE_LIMIT"};r=summarize(d);self.assertEqual(r["status"],"FAIL");self.assertEqual(r["grasp_lift"],"PASS")
    def test_static_or_contact_only_is_not_physical_grasp(self):
        d=self.sample();d["physical_lift_evidence"]={};self.assertNotEqual(summarize(d)["status"],"PASS")
class WaistDurationTests(unittest.TestCase):
    def test_small_existing_posture_keeps_four_seconds(self):
        from scripts.research.issue46.native_runtime.standing_waist_preposition import waist_motion_duration,WAIST
        a=dict.fromkeys(WAIST,0.);b=dict(a);b[WAIST[0]]=.2
        self.assertEqual(waist_motion_duration(a,b),4.)
    def test_larger_yaw_preserves_peak_rate_without_relaxing_source_limits(self):
        from scripts.research.issue46.native_runtime.standing_waist_preposition import waist_motion_duration,WAIST
        a=dict.fromkeys(WAIST,0.);b=dict(a);b[WAIST[0]]=.6
        duration=waist_motion_duration(a,b)
        self.assertGreater(duration,4.);self.assertLessEqual(1.5*.6/duration,.1400000001)
    def test_unbounded_motion_is_rejected(self):
        from scripts.research.issue46.native_runtime.standing_waist_preposition import waist_motion_duration,WAIST
        a=dict.fromkeys(WAIST,0.);b=dict.fromkeys(WAIST,2.)
        with self.assertRaises(ValueError):waist_motion_duration(a,b)
class ArmSearchRangeTests(unittest.TestCase):
    def test_pre_raised_elbow_can_extend_through_original_range(self):
        import numpy as np
        from scripts.research.issue46.grasp.stl_approved_circle_ik import arm_search_bounds
        source=np.array([[-2.356194,0.]])
        original=source.copy()
        limited=arm_search_bounds(source,[-2.15]);full=arm_search_bounds(source,[-2.15],full_source=True)
        self.assertLess(limited[0,1],-.9);self.assertGreater(full[0,1],-.1)
        np.testing.assert_array_equal(source,original)
    def test_full_search_retains_inward_source_margin(self):
        from scripts.research.issue46.grasp.stl_approved_circle_ik import arm_search_bounds
        b=arm_search_bounds([[-2.,0.]],[-1.5],full_source=True,margin=.03)
        self.assertAlmostEqual(b[0,0],-1.97);self.assertAlmostEqual(b[0,1],-.03)
class ActivationVelocityTests(unittest.TestCase):
    def test_activation_floor_still_respects_command_cap(self):
        import math
        from scripts.research.issue46.native_runtime.native_task_grasp_region import body_translation_velocity
        pose={"x":0.,"y":0.,"yaw":0.}
        vx,vy,w=body_translation_velocity(pose,[.02,0.],minimum_speed=.12)
        self.assertAlmostEqual(vx,.12);self.assertEqual(vy,0.);self.assertEqual(w,0.)
        vx,vy,w=body_translation_velocity(pose,[10.,10.],minimum_speed=.12)
        self.assertLessEqual(math.hypot(vx,vy),.20000000001)
    def test_zero_position_error_remains_zero(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import body_translation_velocity
        self.assertEqual(body_translation_velocity({"x":0.,"y":0.,"yaw":0.},[0.,0.],minimum_speed=.12),(0.,0.,0.))
class CommonTransferAcceptanceTests(unittest.TestCase):
    def test_grasp_hold_alone_cannot_qualify_common_transfer(self):
        from scripts.research.issue46.grasp.table_edge_grasp_trials import summarize_transfer
        d=GraspAcceptanceTests().sample()
        r=summarize_transfer(d)
        self.assertNotEqual(r["full_transfer"],"PASS")
        self.assertIn("independent_table_support_s",r["missing_evidence"])
    def test_missing_place_measurements_fail_even_with_all_green_stage_labels(self):
        from scripts.research.issue46.grasp.table_edge_grasp_trials import summarize_transfer
        from scripts.research.issue46.grasp.table_edge_grid_trials import REQUIRED_GATES
        d=GraspAcceptanceTests().sample();d["gates"].update(dict.fromkeys(REQUIRED_GATES,"PASS"));d["table_placement"]={"pass_result":True}
        self.assertNotEqual(summarize_transfer(d)["full_transfer"],"PASS")
if __name__=="__main__":unittest.main()
