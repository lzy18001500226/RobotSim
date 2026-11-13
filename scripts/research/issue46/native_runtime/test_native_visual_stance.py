"""Offline behavior checks; these do not qualify physical navigation."""
import ast,math,unittest
from scripts.research.issue46.native_runtime import native_visual_stance as v

class VisualStanceTests(unittest.TestCase):
    def test_perceived_object_shift_causes_equal_base_shift(self):
        base={'rotation':[1,0,0,0,1,0,0,0,1]}
        ref={'object_offset_base_xy_m':[.45,-.12]}
        self.assertEqual(v.stance_goal([1.4,-1.3,.65],base,ref),[.95,-1.1800000000000002])
        a=v.stance_goal([1.4,-1.6,.65],base,ref)
        b=v.stance_goal([1.4,-1.3,.65],base,ref)
        self.assertAlmostEqual(b[1]-a[1],.3)
    def test_rotated_robot_calibration_and_body_direction(self):
        base={'rotation':[0,-1,0,1,0,0,0,0,1]}
        goal=v.stance_goal([1,-1,.65],base,{'object_offset_base_xy_m':[.5,-.1]})
        self.assertAlmostEqual(goal[0],.9);self.assertAlmostEqual(goal[1],-1.5)
        self.assertEqual(v.stance_axis({'x':0,'y':0,'yaw':math.pi/2},[-.2,.01])[0],'left')
        self.assertEqual(v.stance_axis({'x':0,'y':0,'yaw':0},[-.2,.01])[0],'backward')
    def test_truth_audit_fields_cannot_change_control_goal(self):
        base={'rotation':[1,0,0,0,1,0,0,0,1],'object_truth':[99,99,99]}
        ref={'object_offset_base_xy_m':[.45,-.12],'object_truth':[-99,-99,99]}
        a=v.stance_goal([1.4,-1.3,.65],base,ref)
        base['object_truth']=[0,0,0];ref['object_truth']=[5,5,5]
        self.assertEqual(a,v.stance_goal([1.4,-1.3,.65],base,ref))
    def test_bad_measurement_or_outside_workspace_rejected(self):
        b={'rotation':[1,0,0,0,1,0,0,0,1]};r={'object_offset_base_xy_m':[.45,-.12]}
        for point in ([float('nan'),0,0],[10,0,0],[0,0]):
            with self.assertRaises(ValueError):v.stance_goal(point,b,r)
    def test_initiated_lateral_motion_does_not_rearm_on_gait_sway(self):
        cfg={'startup_deadline_s':.5,'lateral_startup_speed':.6}
        command,started=v.stance_velocity('left',(0.,.12),.1,0.,False,cfg)
        self.assertEqual(command,(0.,.6));self.assertFalse(started)
        command,started=v.stance_velocity('left',(0.,.12),.427,.01543,started,cfg)
        self.assertEqual(command,(0.,.12));self.assertTrue(started)
        command,started=v.stance_velocity('left',(0.,.12),.8,.001,started,cfg)
        self.assertEqual(command,(0.,.12));self.assertTrue(started)
        with self.assertRaisesRegex(RuntimeError,'NO_MEASURED_PROGRESS'):
            v.stance_velocity('left',(0.,.12),.51,.001,False,cfg)
    def test_actual_generated_native_program_compiles(self):
        code=v.build_motion_script([.95,-1.2],{},[],'visual_stance_1')
        tree=ast.parse(code)
        self.assertEqual(sum(isinstance(n,ast.FunctionDef) and n.name=='stance_axis' for n in tree.body),1)
        self.assertIn('run_visual_stance(n,contract)',code)
        self.assertIn('"LOCOMOTION_STEP" if MICRO.get("pickup_stance_profile") else "LOCOMOTION_DEFAULT"',code)
        self.assertIn('native_foot_loads(native)',code)
        self.assertNotIn('mj_step(',code)
        self.assertNotIn('data.qpos[',code)
        with self.assertRaises(ValueError):v.build_motion_script([.95,-1.2],{},[],'bad/result')

if __name__=='__main__':unittest.main()
