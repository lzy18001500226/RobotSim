"""Shorter task waiting must still measure physical time and live grip health."""
import json,tempfile,unittest
from pathlib import Path
from unittest import mock
from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as runtime,native_grasp_audit as audit

class FastTimingTests(unittest.TestCase):
    def exercise(self,required,*,bad_health=False):
        clock=[0.];calls=[0]
        def row(*args):
            calls[0]+=1
            return {"generation":1,"sim_time_s":(calls[0]-1)*.01,"host_monotonic_ns":int((10.+calls[0]*.02)*1e9)}
        def now():return 10.+calls[0]*.02
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            # Match live heartbeat timestamps without changing their age contract.
            original=Path.read_text
            def read(path,*a,**kw):
                if path.name=="carry_heartbeat.json":return json.dumps({"mode":"MEASURED_LOAD_BEARING_CARRY_HOLD","healthy":not bad_health,"generation":1,"pid":19,"host_monotonic_ns":int(now()*1e9),"native_source_host_ns":int(now()*1e9)})
                return original(path,*a,**kw)
            with mock.patch.object(audit,"require_fresh_sample",side_effect=row),mock.patch.object(runtime.time,"monotonic",side_effect=now),mock.patch.object(runtime.time,"monotonic_ns",side_effect=lambda:int(now()*1e9)),mock.patch.object(runtime.time,"sleep"),mock.patch.object(Path,"read_text",read):
                result=runtime.qualify_static_carry_hold(out,{"generation":1,"pid":19},minimum_sim_s=required)
            return result,calls[0]
    def test_short_hold_is_two_sim_seconds_not_sample_count_or_wall_seconds(self):
        result,count=self.exercise(2.)
        self.assertGreaterEqual(result["sim_span_s"],2.)
        self.assertGreater(count,200)
        self.assertEqual(result["required_sim_span_s"],2.)
        self.assertGreater(result["host_span_s"],3.9)
    def test_unhealthy_grip_stops_before_short_hold_acceptance(self):
        with self.assertRaisesRegex(RuntimeError,"CARRY_STATIC_HOLD_HEARTBEAT_UNQUALIFIED"):
            self.exercise(2.,bad_health=True)
    def test_arbitrary_wait_reduction_is_rejected(self):
        with self.assertRaises(ValueError):runtime.qualify_static_carry_hold(Path("unused"),{},minimum_sim_s=.5)
    def test_historical_default_stays_eight_seconds(self):
        self.assertEqual(runtime.qualify_static_carry_hold.__kwdefaults__["minimum_sim_s"],8.)
class AdaptiveCloseTests(unittest.TestCase):
    def test_short_close_waits_for_contact_without_reopening(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import await_prelift_contact
        class Node:
            closure_ramp_complete=False
            def __init__(self):self.calls=[]
            def send(self,target,duration,stage,hand):
                if not self.closure_ramp_complete:raise AssertionError("must not restart closure ramp")
                self.calls.append((target,duration,stage,hand))
        n=Node();count=[0]
        def reader(start,minimum_normal_force_n):
            self.assertEqual(start,10.);self.assertEqual(minimum_normal_force_n,1.5)
            count[0]+=1;return {"pass":count[0]>=3}
        result=await_prelift_contact(n,{},.5,10.,1.5,1.,reader)
        self.assertTrue(result["pass"]);self.assertEqual(len(n.calls),2)
    def test_old_zero_budget_never_adds_motion(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import await_prelift_contact
        class Node:
            def send(self,*a):raise AssertionError("historical timing must not add commands")
        result=await_prelift_contact(Node(),{},.5,10.,1.5,0.,lambda *a,**k:{"pass":False})
        self.assertFalse(result["pass"]);self.assertEqual(result["qualification_send_calls"],0)
class CarryStopTimingTests(unittest.TestCase):
    def test_fast_task_keeps_owned_source_active_through_zero_window(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import task_route_profile
        cfg={"schema":"robotsim.x2.two-table-delivery.v1","transit_root_cap_m_s":.60,"turn_translation_strategy":"PURE_YAW"}
        default=task_route_profile(cfg);fast=task_route_profile(cfg,fast_task_timing=True)
        self.assertNotIn("release_source_after_zero_window",default)
        self.assertTrue(fast["release_source_after_zero_window"])
        self.assertEqual(fast["delivery_turn_direction"],"CLOCKWISE")
        self.assertEqual(fast["turn_translation_strategy"],"PURE_YAW")
        self.assertTrue(fast["stop_inside_functional_arrival_region"])
        self.assertEqual(fast["approach_task_position_tolerance_m"],.15)
        self.assertEqual(fast["approach_task_yaw_tolerance_rad"],.30)
        self.assertEqual(fast["transit_root_cap_m_s"],cfg["transit_root_cap_m_s"])
        self.assertNotIn("release_source_after_zero_window",cfg)
class FunctionalArrivalTests(unittest.TestCase):
    def test_nominal_yaw_alone_is_not_required_when_actual_placement_will_be_checked(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import delivery_heading_acceptance
        self.assertFalse(delivery_heading_acceptance({},.481)["pass"])
        value=delivery_heading_acceptance({"delivery_pose_acceptance":"ACTUAL_PLACEMENT_IK","require_restored_route_preflight":True},.481)
        self.assertTrue(value["pass"]);self.assertTrue(value["requires_actual_placement_ik_before_arm_motion"])
    def test_missing_measured_preflight_cannot_enable_functional_arrival(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import delivery_heading_acceptance
        with self.assertRaises(ValueError):delivery_heading_acceptance({"delivery_pose_acceptance":"ACTUAL_PLACEMENT_IK"},.481)
class DeliverySteeringBandTests(unittest.TestCase):
    def test_saved31degree_bearing_error_can_use_existing_forward_steering(self):
        import sys,math
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"ros2_ws/src/robotsim_navigation"))
        from robotsim_navigation.point_goal_controller import PointGoalController,ControllerConfig,Goal2D,Pose2D,NavState
        p=Pose2D(0.,0.,-.5476095438300144,"native_scene_world",1.)
        g=Goal2D(1.,0.,0.,"native_scene_world")
        old=PointGoalController(ControllerConfig(align_tolerance_rad=.45,k_angular=2.4,max_angular_rps=.60)).step(p,g,now_steady_s=1.)
        new=PointGoalController(ControllerConfig(align_tolerance_rad=.75,k_angular=2.4,max_angular_rps=.60)).step(p,g,now_steady_s=1.)
        self.assertEqual(old.state,NavState.ALIGN);self.assertEqual(new.state,NavState.DRIVE)
        self.assertGreater(new.linear_x*math.cos(p.yaw),0.);self.assertLessEqual(abs(new.angular_z),.60)
        p=Pose2D(0.,0.,-1.,"native_scene_world",1.)
        self.assertEqual(PointGoalController(ControllerConfig(align_tolerance_rad=.75)).step(p,g,now_steady_s=1.).state,NavState.ALIGN)
if __name__=="__main__":unittest.main()
