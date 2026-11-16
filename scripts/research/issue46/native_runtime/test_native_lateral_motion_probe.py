import math,unittest
from scripts.research.issue46.native_runtime.native_lateral_motion_probe import phases,build_script
class LateralProbeTests(unittest.TestCase):
    def test_left_right_only_reverse_side_input(self):
        a,b=phases("left"),phases("right")
        for x,y in zip(a,b):
            self.assertEqual(x[:2],y[:2]);self.assertEqual(x[2][0],y[2][0]);self.assertEqual(x[2][1],-y[2][1]);self.assertEqual(x[2][2],-y[2][2]);self.assertLessEqual(abs(x[2][2]),.40)
    def test_combined_command_stays_below_existing_norm_cap(self):
        for _,_,(x,y,w) in phases("left"):
            self.assertLessEqual(math.hypot(x,y),.2);self.assertLessEqual(abs(w),.40)
    def test_existing_native_transport_can_compile_probe(self):
        code=build_script("left",{},[],{},dict(x=[0.,1.],y=[-2.,0.],yaw=[-.2,.2]))
        compile(code,"probe","exec");self.assertIn("return run_lateral_probe(n,contract)",code)
        self.assertNotIn("mj_step(",code)
    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):phases("diagonal")
class NativeStandingTests(unittest.TestCase):
    def sample(self,height):
        return dict(generation=1,host_monotonic_ns=1,sim_time_s=1.,bodies={"pelvis":{"position_m":[0.,0.,height],"rotation":[1.,0.,0.,0.,1.,0.,0.,0.,1.],"origin_velocity_world_angular_linear":[0.]*6}},contacts=[dict(efc_address=0,body1="world",body2=side+"_ankle_roll_link",local_force_torque=[100.,0.,0.,0.,0.,0.]) for side in ("left","right")])
    def test_stable_imu_does_not_make_a_collapsed_robot_ready(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import native_standing_snapshot
        r=native_standing_snapshot(self.sample(.065));self.assertFalse(r["ready"]);self.assertIn("HEIGHT",r["reasons"])
    def test_real_supported_standing_is_ready(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import native_standing_snapshot
        self.assertTrue(native_standing_snapshot(self.sample(.65))["ready"])
class SimulatorReceiptTests(unittest.TestCase):
    def test_creation_is_not_delivery(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import simulator_leg_receipt
        self.assertFalse(simulator_leg_receipt("Create subscriber: leg_joint_command succeeded"))
    def test_actual_leg_handle_is_delivery(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import simulator_leg_receipt
        self.assertTrue(simulator_leg_receipt("[module/subscriber/joint/leg_joint_command_subscriber.cc:37 @Handle]cmd size: 12"))
class SideStartupProfileTests(unittest.TestCase):
    def test_old_profile_cannot_send_high_side_input(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import validate_velocity
        with self.assertRaises(RuntimeError):validate_velocity((0.,.5,0.),"reference")
    def test_high_profile_is_short_pure_side_and_preserves_physical_bounds(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import phases,validate_velocity
        p=phases("left","side-start");self.assertEqual(p[0][1],.25);self.assertEqual(p[0][2],(0.,.5,0.))
        for _,_,v in p:validate_velocity(v,"side-start")
        with self.assertRaises(RuntimeError):validate_velocity((.1,.5,0.),"side-start")
    def test_gateway_cap_override_is_explicit_and_scoped(self):
        from scripts.research.issue46.native_runtime.native_lateral_motion_probe import side_start_bridge_command
        for old in ("0.2","0.20"):
            c=side_start_bridge_command("bridge --ros-args -p max_lateral_velocity:="+old+" > log")
            self.assertIn("max_lateral_velocity:=0.50",c)
        with self.assertRaises(ValueError):side_start_bridge_command("bridge --ros-args")
if __name__=="__main__":unittest.main()
