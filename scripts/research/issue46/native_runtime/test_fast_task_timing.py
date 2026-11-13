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
    def test_preload_failure_cannot_advance_and_is_bounded_by_original_five_second_total(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import await_prelift_contact
        clock=[0.]
        class Node:
            def send(self,*a):clock[0]+=.04
        with mock.patch("time.monotonic",side_effect=lambda:clock[0]):
            result=await_prelift_contact(Node(),{},.5,10.,14.715,3.,lambda *a,**k:{"pass":False})
        self.assertFalse(result["pass"]);self.assertLess(result["additional_qualification_wall_s"],3.05)
    def test_old_zero_budget_never_adds_motion(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import await_prelift_contact
        class Node:
            def send(self,*a):raise AssertionError("historical timing must not add commands")
        result=await_prelift_contact(Node(),{},.5,10.,1.5,0.,lambda *a,**k:{"pass":False})
        self.assertFalse(result["pass"]);self.assertEqual(result["qualification_send_calls"],0)
class CarryStopTimingTests(unittest.TestCase):
    def test_fast_task_keeps_owned_source_active_through_zero_window(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import task_route_profile
        cfg={"schema":"robotsim.x2.two-table-delivery.v1","transit_root_cap_m_s":.60,"turn_translation_strategy":"PURE_YAW","base_goal_world_xy":[.015,.98]}
        default=task_route_profile(cfg);fast=task_route_profile(cfg,fast_task_timing=True)
        self.assertNotIn("release_source_after_zero_window",default)
        self.assertTrue(fast["release_source_after_zero_window"])
        self.assertEqual(fast["delivery_turn_direction"],"CLOCKWISE")
        self.assertEqual(fast["base_goal_world_xy"],[.015,.90])
        self.assertEqual(default["base_goal_world_xy"],[.015,.98])
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
class LoadedTerminalBrakeTests(unittest.TestCase):
    def test_saved_left_terminal_input_is_reduced_without_changing_safety_limits(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import loaded_delivery_terminal_command
        p={"x":.21225222683139955,"y":.7513642942241849};goal=[.015,.98]
        self.assertEqual(loaded_delivery_terminal_command({},p,goal,.20,.46133845196305034),(.20,.46133845196305034))
        self.assertEqual(loaded_delivery_terminal_command({"loaded_terminal_slowdown":True},p,goal,.20,.46133845196305034),(.12,.46133845196305034))
        self.assertEqual(loaded_delivery_terminal_command({"loaded_terminal_slowdown":True},{"x":.9,"y":-1.4},goal,.20,.5),(.20,.5))
        self.assertEqual(loaded_delivery_terminal_command({"loaded_terminal_slowdown":True},p,goal,.05,-.1),(.05,-.1))
class ActualTerminalCommandPipelineTests(unittest.TestCase):
    def test_frozen_ros_controller_output_is_not_mutated_or_later_overwritten(self):
        import ast,sys
        from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"ros2_ws/src/robotsim_navigation"))
        from robotsim_navigation.point_goal_controller import ControllerOutput,NavState
        output=ControllerOutput(NavState.DRIVE,.2,.46,.3,.1)
        script=micro.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        tree=ast.parse(script)
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(st,ast.Assign) and isinstance(st.value,ast.Call) and isinstance(st.value.func,ast.Name) and st.value.func.id=="loaded_delivery_terminal_command" for st in n.body))
        ns=dict(mission={"phase":"DELIVERY"},cfg={"delivery":{"loaded_terminal_slowdown":True}},p={"x":.212,"y":.751},goal=[.015,.98],forward=.2,angular=output.angular_z,loaded_delivery_terminal_command=micro.loaded_delivery_terminal_command)
        exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_generated_terminal_command","exec"),ns)
        self.assertEqual((ns["forward"],ns["angular"]),(.12,.46));self.assertEqual(output.angular_z,.46)
        self.assertNotIn("forward,output.angular_z=",script)
        self.assertGreater(script.index("forward,angular=loaded_delivery_terminal_command"),script.index('forward=float(cfg["delivery"].get("delivery_forward_command_m_s"'))
class TorsoRelativeCarryTests(unittest.TestCase):
    def test_same_relative_motion_rotates_with_body_and_never_adds_vertical_or_wrist_command(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from scripts.research.issue46.grasp.stl_approved_circle_ik import torso_horizontal_retract
        for yaw in (0.,.8,1.4):
            R=Rotation.from_euler("z",yaw).as_matrix();v=torso_horizontal_retract(R)
            np.testing.assert_allclose(R.T@v,[-.16,-.12,0.],atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(v),.20);self.assertEqual(v[2],0.)
class UniformReliablePreloadTests(unittest.TestCase):
    def test_current_complete_task_preserves_historical_five_second_preload(self):
        from scripts.research.issue46.grasp import stl_approved_circle_native_trial as runner
        import inspect
        code=inspect.getsource(runner)
        self.assertIn("argv[argv.index('--hal-hand-close-hold-s')+1]='5'",code)
        self.assertNotIn("argv[argv.index('--hal-hand-close-hold-s')+1]='2'",code)
class GeometricTableArrivalTests(unittest.TestCase):
    def test_nominal_xy_failure_only_can_request_original_physical_placement_preflight(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import geometric_delivery_handoff_eligible
        nav={"pass_result":False,"failure":"CARRY_GOAL_OR_STOP_NOT_QUALIFIED","goal_triggered_stop":True,"source_and_motion_mode_observed":True,"retreat_turn_prefix_pass":True,"away_mission":{"bilateral_support_alternation_observed":True},"measured_distance_m":2.}
        for name in ("post_stop_composite","retreat_stop_composite","turn_stop_composite"):nav[name]={"pass":True}
        cfg={"delivery_pose_acceptance":"ACTUAL_PLACEMENT_IK","require_restored_route_preflight":True}
        self.assertTrue(geometric_delivery_handoff_eligible(nav,{"pass":True},cfg))
        self.assertFalse(geometric_delivery_handoff_eligible(nav,{"pass":False},cfg))
        nav["failure"]="JOINT_LIMIT";self.assertFalse(geometric_delivery_handoff_eligible(nav,{"pass":True},cfg))
        nav["failure"]="CARRY_GOAL_OR_STOP_NOT_QUALIFIED";nav["post_stop_composite"]={"pass":False};self.assertFalse(geometric_delivery_handoff_eligible(nav,{"pass":True},cfg))
class EpochModeRestoreTests(unittest.TestCase):
    def test_clock_recovery_reselects_existing_default_mode_without_faking_readback(self):
        import ast
        from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
        script=micro.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        tree=ast.parse(script)
        guards=[n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(v,ast.Constant) and v.value=="prepare_default_after_pickup" for v in ast.walk(n.test)) and any(isinstance(st,ast.Expr) and isinstance(st.value,ast.Call) and isinstance(st.value.func,ast.Attribute) and st.value.func.attr=="stand_once" for st in n.body)]
        self.assertTrue(guards)
        from types import SimpleNamespace
        requests=[];ns=dict(cfg={"motion_mode":"LOCOMOTION_DEFAULT","delivery":{"prepare_default_after_pickup":True}},n=SimpleNamespace(stand_once=lambda x:requests.append(x)))
        exec(compile(ast.Module(body=[guards[-1]],type_ignores=[]),"actual_recovery_mode_selection","exec"),ns)
        self.assertEqual(requests,["LOCOMOTION_DEFAULT"])
class ConservativeTableStopTests(unittest.TestCase):
    def test_saved_maximum_coast_has_positive_remaining_distance(self):
        import ast
        from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
        script=micro.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        tree=ast.parse(script);node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(v,ast.Constant) and v.value=="conservative_terminal_stop" for v in ast.walk(n.test)))
        ns=dict(cfg={"delivery":{"conservative_terminal_stop":True}},trigger=.22699435678032145);exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_stop_reserve","exec"),ns)
        self.assertEqual(ns["trigger"],.45);self.assertGreater(ns["trigger"]-.38387407559438663,.06)
class PlacementRadialLimitTests(unittest.TestCase):
    def test_actual_saved_box_solution_is_outside_original_circle(self):
        import numpy as np
        hint=np.array([-.021424671576197978,1.4583865926539565]);selected=np.array([.12275726804475545,1.414193472354619]);delta=selected-hint
        self.assertGreater(np.linalg.norm(delta),.15)
        corrected=hint+delta*.149/np.linalg.norm(delta)
        self.assertLess(np.linalg.norm(corrected-hint),.15)
        np.testing.assert_allclose(corrected,[.12103368378638066,1.4147217671374237])
class PremotionNewEpochTests(unittest.TestCase):
    def test_missing_new_epoch_can_wait_only_without_any_prior_nonzero_command(self):
        import ast
        from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
        script=micro.build_script("forward",.6,{})
        tree=ast.parse(script);guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=="CRITICAL_NEW_CLOCK_GENERATION_TELEMETRY_MISSING" for x in ast.walk(n.test)) and any(isinstance(x,ast.Name) and x.id=="initial_zero_only" for x in ast.walk(n.test)))
        check=compile(ast.Expression(guard.test),"actual_initial_epoch_wait","eval")
        class Clock:
            def monotonic(self):return 10.
        ns=dict(initial_zero_only=True,exc=RuntimeError("CRITICAL_NEW_CLOCK_GENERATION_TELEMETRY_MISSING"),time=Clock(),hard_until=12.)
        self.assertTrue(eval(check,ns));ns["initial_zero_only"]=False;self.assertFalse(eval(check,ns))
        ns["initial_zero_only"]=True;ns["hard_until"]=9.;self.assertFalse(eval(check,ns))
        ns["hard_until"]=12.;ns["exc"]=RuntimeError("JOINT_LIMIT");self.assertFalse(eval(check,ns))
class BodyModeReadinessTests(unittest.TestCase):
    def test_empty_initial_mode_is_readiness_missing_not_accepted_stand(self):
        import ast
        from types import SimpleNamespace
        from scripts.research.issue46.native_runtime import native_body_height
        tree=ast.parse(native_body_height.build_script({},[],mode="observe"));outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_height_probe");fn=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=="sample")
        row={"get_mode_desc":None,"get_mode_header_code":None,"common_mode_desc":"STAND_DEFAULT"}
        ns=dict(n=SimpleNamespace(pump=lambda:None,sample=lambda:row),deadline=10.,time=SimpleNamespace(monotonic=lambda:1.),process=None,result={"commands":[]})
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual_body_mode_readiness","exec"),ns)
        with self.assertRaisesRegex(RuntimeError,"CRITICAL_ROS_TELEMETRY_STALE:get_mode"):ns["sample"]()
class TerminalReapproachContractTests(unittest.TestCase):
    def test_actual_generated_reapproach_is_bounded_and_requires_physical_stop(self):
        import ast
        from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
        script=micro.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        tree=ast.parse(script);guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=="bounded_terminal_forward_reapproach" for x in ast.walk(n.test)))
        code=ast.unparse(guard)
        self.assertIn('range(2)',code);self.assertIn("require_stop=True",code)
        self.assertIn('TERMINAL_REAPPROACH_STOP_OR_TIME_UNQUALIFIED',code)
        self.assertIn('TERMINAL_REAPPROACH_APPLIED_MODE_NOT_OBSERVED',code)
        self.assertIn('n.publish(0.12',code)
if __name__=="__main__":unittest.main()
