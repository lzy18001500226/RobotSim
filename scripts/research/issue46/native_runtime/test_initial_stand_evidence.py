"""Missing ROS samples do not imply native physical instability."""
import unittest
from scripts.research.issue46.native_runtime.stl_hand_startup import qualify_initial_standing

class InitialStandTests(unittest.TestCase):
    def observation(self, count=0, passed=False):
        return dict(mode_name="STAND_DEFAULT",mode_status=100,
                    stability=dict(sample_count=count,pass_result=passed,
                                   planar_drift_m=None if count==0 else .001,
                                   **{"pass":passed}))
    def test_missing_ros_with_measured_native_support(self):
        r=qualify_initial_standing(self.observation(),dict(pass_result=True,max_planar_drift_m=.001))
        self.assertTrue(r["pass"]);self.assertIn("warning",r)
    def test_acknowledgement_or_missing_native_cannot_pass(self):
        for check in [None,dict(pass_result=True),dict(pass_result=False,max_planar_drift_m=0)]:
            self.assertFalse(qualify_initial_standing(self.observation(),check)["pass"])
    def test_actual_ros_instability_not_hidden_by_native_fallback(self):
        self.assertFalse(qualify_initial_standing(self.observation(count=10),dict(pass_result=True,max_planar_drift_m=0))["pass"])
    def test_native_drift_and_wrong_mode_still_fail(self):
        for drift in [.051,float("nan")]:
            self.assertFalse(qualify_initial_standing(self.observation(),dict(pass_result=True,max_planar_drift_m=drift))["pass"])
        r=self.observation();r["mode_status"]=200
        self.assertFalse(qualify_initial_standing(r,dict(pass_result=True,max_planar_drift_m=0))["pass"])
    def test_legacy_ros_pass(self):
        self.assertTrue(qualify_initial_standing(self.observation(count=100,passed=True))["pass"])

if __name__=="__main__":unittest.main()
