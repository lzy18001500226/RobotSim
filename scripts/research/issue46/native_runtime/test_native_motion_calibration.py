"""Pure measurement regressions; no SDK or physics execution."""
import math,unittest
from scripts.research.issue46.native_runtime import native_motion_calibration as c

class MotionMeasurementTests(unittest.TestCase):
    def test_body_axes_follow_actual_start_heading(self):
        d=c.signed_delta({'x':0.,'y':0.,'yaw':math.pi/2},{'x':-.03,'y':.05,'yaw':math.pi/2})
        self.assertAlmostEqual(d['forward_m'],.05);self.assertAlmostEqual(d['left_m'],.03)
    def test_yaw_crossing_pi_is_small_positive_rotation(self):
        d=c.signed_delta({'x':0.,'y':0.,'yaw':3.1},{'x':0.,'y':0.,'yaw':-3.1})
        self.assertAlmostEqual(d['yaw_rad'],2*math.pi-6.2)
    def test_gait_sway_is_not_a_completed_step(self):
        start={'x':0.,'y':0.,'yaw':0.}
        rows=[{'sim_time_s':i*.1,'pose':{'x':x,'y':0.,'yaw':0.,'vx':0.,'vy':0.}} for i,x in enumerate([0.,.05,0.,-.05,0.,.05,0.])]
        self.assertLess(c.projected_progress(rows,start,'forward'),.02)
        self.assertFalse(c.motion_metrics(start,start,rows,'forward',.05)['direction_observed'])
    def test_wrong_side_is_reported_as_unqualified(self):
        start={'x':0.,'y':0.,'yaw':0.};end={'x':0.,'y':.02,'yaw':0.}
        m=c.motion_metrics(start,end,[],'right',.05)
        self.assertFalse(m['direction_observed']);self.assertLess(m['signed_goal_progress_m_or_rad'],0.)
    def test_measured_response_does_not_claim_distance_accuracy(self):
        start={'x':0.,'y':0.,'yaw':0.};end={'x':.08,'y':.03,'yaw':.1}
        m=c.motion_metrics(start,end,[],'forward',.05)
        self.assertTrue(m['direction_observed']);self.assertAlmostEqual(m['goal_error_m_or_rad'],.03)
        self.assertTrue(m['not_a_distance_accuracy_PASS'])
    def test_active_clock_fault_cannot_use_logger_resume(self):
        with self.assertRaisesRegex(ValueError,'active clock faults'):
            c.logger_resume_context({'failure':'CLOCK_GENERATION_PENDING','abort_stop_composite':{'pass':True},'motions':[],'trace':[]},'/unused')
    def test_generated_program_compiles_with_its_dependencies(self):
        code=c.build_script({},[],{})
        compile(code,'calibration','exec')
        self.assertIn('return run_calibration(n,contract)',code)
        self.assertIn('CALIBRATION_MODE_ALLOWLIST',code)

if __name__=='__main__':unittest.main()
