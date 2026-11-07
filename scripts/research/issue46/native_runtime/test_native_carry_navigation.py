"""Carry route adapter checks; no ROS or physical motion."""
import ast
from pathlib import Path
import math
import unittest
from scripts.research.issue46.native_runtime import native_base_micro_probe as m

class CarryNavigationTests(unittest.TestCase):
    def test_route_generation_uses_existing_controller_and_fixed_caps(self):
        code=m.build_carry_route_script([-.45,1.],{},["pelvis"],carry_pid=19,timeout_s=90.)
        compile(code,"carry_route","exec")
        tree=ast.parse(code)
        self.assertIn("run_carry_navigation", {n.name for n in tree.body if isinstance(n,ast.FunctionDef)})
        self.assertIn("PointGoalController",code)
        self.assertIn("max_forward_mps=.12",code)
        self.assertIn("max_angular_rps=.075",code)
        self.assertIn("CARRY_OWNER_OR_PHYSICAL_HEARTBEAT_UNQUALIFIED",code)
        self.assertTrue(code.endswith("run_probe()"+chr(10)))
    def test_invalid_goals_and_deadlines_are_not_executable(self):
        for goal,seconds in (([2.,0.],90.),([float("nan"),0.],90.),([0.,0.],91.)):
            with self.assertRaises(ValueError):m.build_carry_route_script(goal,{},[],carry_pid=19,timeout_s=seconds)
    def test_only_documented_clock_fault_can_enter_zero_stop_recovery(self):
        fault={"gate":"CLOCK_STEP_DURING_MOTION"}
        self.assertTrue(m.recoverable_carry_clock_fault(fault,RuntimeError("NATIVE_FEEDBACK_INVALID:clock"),0))
        for error in ("PLANAR_SPEED","SOURCE_JOINT_SPEED:knee","STALE_BILATERAL_ARM_FEEDBACK"):
            self.assertFalse(m.recoverable_carry_clock_fault(fault,RuntimeError(error),0))
        self.assertFalse(m.recoverable_carry_clock_fault({"gate":"POSTURE"},RuntimeError("NATIVE_FEEDBACK_INVALID:pose"),0))
        self.assertFalse(m.recoverable_carry_clock_fault(fault,RuntimeError("NATIVE_FEEDBACK_INVALID:clock"),4))
    def test_intended_pad_contact_is_the_only_new_collision_allowance(self):
        pose=dict(z=.65,roll=0.,pitch=0.,vx=0.,vy=0.)
        contact=dict(body1="R_hand_narrow3_Link",body2="test_cylinder",efc_address=0,distance_m=-.002,local_force_torque=[1.])
        row=dict(joints={},contacts=[contact],active_contact_count=1,contact_capture_scope="ALL_COMPILED_GEOM_CONTACTS")
        self.assertIsNone(m.physical_fault(row,pose,{},robot_bodies={"R_hand_narrow3_Link"},allowed_contact_pairs=[["R_hand_narrow3_Link","test_cylinder"]]))
        self.assertTrue(m.physical_fault(row,pose,{},robot_bodies={"R_hand_narrow3_Link"}).startswith("ROBOT_ENVIRONMENT_COLLISION"))
        with self.assertRaises(ValueError):m.physical_fault(row,pose,{},allowed_contact_pairs=[["pelvis","world"]])
        pose["vx"]=.201
        self.assertEqual(m.physical_fault(row,pose,{},robot_bodies={"R_hand_narrow3_Link"},allowed_contact_pairs=[["R_hand_narrow3_Link","test_cylinder"]]),"PLANAR_SPEED")
class CarryRouteControlFlowTests(unittest.TestCase):
    def run_route(self, fault=None, immediate_watchdog=False):
        import contextlib,io,json,sys,threading
        from types import SimpleNamespace
        from unittest import mock
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"ros2_ws/src/robotsim_navigation/robotsim_navigation"))
        clock=SimpleNamespace(t=10.)
        class Node:
            def __init__(self):
                self.lock=threading.Lock();self.moving=False;self.stop_started=False;self.feedback_fault=None
                self.events=[];self.command_log=[];self.bridge_commands=[];self.trace=[];self.latest_row={}
                self.messages={k:{} for k in ("common","imu","leg_state","leg_command")};self.readbacks={"get_mode":(10.,{})};self.x=0.;self.once=False;self.stands=0
            def start_feedback(self):pass
            def stop_feedback(self):pass
            def destroy_node(self):pass
            def pump(self):pass
            def graph(self):pass
            def await_graph(self):pass
            def publish(self,f=0.,l=0.,a=0.):
                if not self.moving:f=l=a=0.
                self.command_log.append(dict(t=clock.t,forward=f,lateral=l,angular=a))
                if f:self.x+=.03
            def deadline_zero(self):
                self.moving=False;self.publish();self.events.append(dict(event="ZERO",t=clock.t))
            def observe(self,seconds,**kwargs):
                self.events.append(dict(event="OBSERVE",seconds=seconds,require_stop=kwargs.get("require_stop",False),t=clock.t))
                clock.t+=.02
                return {"pass":True,"fixture_only":True}
            def stand_once(self):
                self.stands+=1;self.events.append(dict(event="STAND",t=clock.t))
            def sample(self):
                if self.moving and fault and not self.once:
                    self.once=True
                    if fault=="clock":
                        self.feedback_fault={"gate":"CLOCK_STEP_DURING_MOTION"};self.deadline_zero()
                        raise RuntimeError("NATIVE_FEEDBACK_INVALID:clock")
                    self.feedback_fault={"gate":fault};raise RuntimeError(fault)
                pose=dict(x=self.x,y=0.,yaw=0.,z=.65,vx=0.,vy=0.,vz=0.,roll=0.,pitch=0.)
                return dict(body_status=1,pose=pose,sample_monotonic_s=clock.t,common_mode_status_value=200 if self.moving else 100,common_mode_desc="LOCOMOTION_DEFAULT" if self.moving else "STAND_DEFAULT",get_mode_desc="LOCOMOTION_DEFAULT" if self.moving else "STAND_DEFAULT",get_mode_header_code=0,state_age_s={"get_mode":0.},common_source={"name":"robotsim_official_cmd_vel"},raw_common={"speed_status":{"forward_bounds":{"min_value":-1.,"max_value":1.},"angular_bounds":{"min_value":-1.,"max_value":1.}}})
        class Timer:
            def __init__(self,seconds,callback):self.callback=callback
            def start(self):
                if immediate_watchdog:self.callback()
            def cancel(self):pass
        n=Node();written=[]
        with mock.patch.object(m,"MICRO",{"carry":{"goal_xy_m":[.3,0.],"timeout_s":5.}},create=True),mock.patch.object(m,"rclpy",SimpleNamespace(shutdown=lambda:None),create=True),mock.patch("threading.Timer",Timer),mock.patch.object(m.time,"monotonic",lambda:clock.t),mock.patch.object(m.time,"sleep",lambda dt:setattr(clock,"t",clock.t+max(dt,.001))),mock.patch.object(Path,"write_text",lambda self,text:written.append(text)),contextlib.redirect_stdout(io.StringIO()):
            m.run_carry_navigation(n,SimpleNamespace())
        return json.loads(written[-1]),n
    def test_goal_stop_uses_zero_then_stand_then_independent_observation(self):
        result,n=self.run_route()
        self.assertTrue(result["pass_result"])
        self.assertTrue(result["goal_triggered_stop"])
        self.assertEqual(n.stands,1)
        first_stand=next(i for i,x in enumerate(n.events) if x["event"]=="STAND")
        self.assertTrue(any(x["event"]=="ZERO" for x in n.events[:first_stand]))
        self.assertTrue(any(x["event"]=="OBSERVE" and x.get("require_stop") for x in n.events[first_stand+1:]))
    def test_physical_fault_never_rearms(self):
        result,n=self.run_route("PLANAR_SPEED")
        self.assertFalse(result["pass_result"])
        self.assertEqual(result["failure"],"PLANAR_SPEED")
        self.assertEqual(result["clock_recoveries"],[])
        self.assertFalse(any(x["forward"] or x["angular"] for x in n.command_log))
    def test_clock_recovery_stops_and_qualifies_before_new_motion(self):
        result,n=self.run_route("clock")
        self.assertTrue(result["pass_result"])
        self.assertEqual(len(result["clock_recoveries"]),1)
        self.assertEqual(n.stands,2)
        self.assertTrue(result["clock_recoveries"][0]["stop"]["pass"])
    def test_deadline_watchdog_cannot_be_rearmed_by_loop(self):
        result,n=self.run_route(immediate_watchdog=True)
        self.assertFalse(result["pass_result"])
        self.assertFalse(any(x["forward"] or x["angular"] for x in n.command_log))

if __name__=="__main__":unittest.main()
