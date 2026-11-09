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
        self.assertIn('max_angular_rps=cfg.get("angular_command_cap",.075)',code)
        self.assertIn("CARRY_OWNER_OR_PHYSICAL_HEARTBEAT_UNQUALIFIED",code)
        self.assertTrue(code.endswith("run_probe()"+chr(10)))
    def test_sdk_turn_profile_is_explicit_and_retains_physical_speed_limit(self):
        code=m.build_carry_route_script([-.45,1.],{},['pelvis'],carry_pid=19,angular_command_cap=.1)
        self.assertIn("'angular_command_cap': 0.1",code)
        self.assertIn('planar_speed_limit=.20',code)
        with self.assertRaises(ValueError):m.build_carry_route_script([0.,0.],{},[],carry_pid=19,angular_command_cap=.3)
    def test_real_debug_command_duration_is_interpolated_not_literal_code(self):
        import shlex
        from types import SimpleNamespace
        parent=Path(__file__).with_name('xvfb_native_reset_stand.py')
        tree=ast.parse(parent.read_text())
        statement=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='debug_command' for t in n.targets))
        ns=dict(environment='source /opt/ros/humble/setup.bash; ',setup='/workspace/debug/setup.bash',shlex=shlex,expected_debug_node='mc_ros2_node158',args=SimpleNamespace(carry_timeout_s=90.,carry_goal_world_xy=[-.45,1.]))
        exec(compile(ast.Module(body=[statement],type_ignores=[]),'real_debug_command','exec'),ns)
        parts=shlex.split(ns['debug_command'])
        self.assertEqual(parts[parts.index('--max-duration-s')+1],'90')
        self.assertNotIn('str(args',ns['debug_command'])

    def test_pinned_debug_collectors_real_argument_validator_accepts_carry_capture(self):
        import argparse,contextlib,io,sys
        from unittest import mock
        source=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-80/foundationwalk-final-left-right-turns-run65-20261022T073901Z/issue80_mc_rl_debug_capture.py')
        tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        prefix=[]
        for node in main.body:
            if isinstance(node,ast.Expr) and ast.unparse(node)=='rclpy.init()':break
            prefix.append(node)
        self.assertTrue(prefix)
        args=['collector','--output','/unused','--ready-marker','/unused2','--summary','/unused3','--max-duration-s','90']
        with mock.patch.object(sys,'argv',args):
            ns=dict(argparse=argparse,__doc__='validation only')
            exec(compile(ast.Module(body=prefix,type_ignores=[]),'real_collector_argument_validation','exec'),ns)
            self.assertEqual(ns['args'].max_duration_s,90.)

        args[-1]='170'
        with mock.patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            exec(compile(ast.Module(body=prefix,type_ignores=[]),'real_collector_argument_validation','exec'),dict(argparse=argparse,__doc__='validation only'))

    def test_step_preparation_allows_zero_sd_but_requires_positive_speed_readback(self):
        state=dict(body_status=1,get_mode_header_code=0,state_age_s={'get_mode':.05},get_mode_desc='STAND_DEFAULT',common_mode_desc='STAND_DEFAULT',common_mode_status_value=100,raw_common={'speed_status':{'speed_mode':{'value':4}}})
        self.assertTrue(m.native_step_activation_ready(state,1))
        state['raw_common']['speed_status']['speed_mode']['value']=1
        self.assertFalse(m.native_step_activation_ready(state,1))
        state['raw_common']['speed_status']['speed_mode']['value']=4
        state['get_mode_desc']='LOCOMOTION_DEFAULT'
        self.assertFalse(m.native_step_activation_ready(state,1))
        state['get_mode_desc']='STAND_DEFAULT';state['state_age_s']['get_mode']=.21
        self.assertFalse(m.native_step_activation_ready(state,1))

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
    def run_route(self, fault=None, immediate_watchdog=False, pump_delay_s=0., native_step=False, transit_policy_050=False, graph_delay_s=0., check_cached_native_age=False, reverse_moving_arc=False):
        import contextlib,io,json,sys,threading
        from types import SimpleNamespace
        from unittest import mock
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"ros2_ws/src/robotsim_navigation/robotsim_navigation"))
        clock=SimpleNamespace(t=10.)
        class Node:
            def __init__(self):
                self.lock=threading.Lock();self.moving=False;self.stop_started=False;self.feedback_fault=None
                self.events=[];self.command_log=[];self.bridge_commands=[];self.trace=[];self.latest_row={"generation":1}
                self.messages={k:{} for k in ("common","imu","leg_state","leg_command")};self.readbacks={"get_mode":(10.,{})};self.x=0.;self.once=False;self.stands=0;self.current_mode="STAND_DEFAULT";self.native_sample_at=clock.t
            def start_feedback(self):pass
            def stop_feedback(self):pass
            def destroy_node(self):pass
            def pump(self):clock.t+=pump_delay_s;self.native_sample_at=clock.t
            def graph(self):clock.t+=graph_delay_s
            def await_graph(self):self.graph()
            def publish(self,f=0.,l=0.,a=0.):
                if not self.moving:f=l=a=0.
                self.command_log.append(dict(t=clock.t,forward=f,lateral=l,angular=a))
                if f:self.x+=.03 if f>0 else -.03
            def deadline_zero(self):
                self.moving=False;self.publish();self.events.append(dict(event="ZERO",t=clock.t))
            def observe(self,seconds,**kwargs):
                self.events.append(dict(event="OBSERVE",seconds=seconds,require_stop=kwargs.get("require_stop",False),t=clock.t))
                clock.t+=.02
                return {"pass":True,"fixture_only":True}
            def stand_once(self, requested="STAND_DEFAULT"):
                self.current_mode=requested
                self.stands+=int(requested=="STAND_DEFAULT");self.events.append(dict(event="STAND" if requested=="STAND_DEFAULT" else "STEP",t=clock.t))
            def sample(self):
                if check_cached_native_age and clock.t-self.native_sample_at>.20:raise RuntimeError("NATIVE_FEEDBACK_STALE_OR_INVALID")
                if self.moving and fault and not self.once:
                    self.once=True
                    if fault=="clock":
                        self.feedback_fault={"gate":"CLOCK_STEP_DURING_MOTION"};self.deadline_zero()
                        raise RuntimeError("NATIVE_FEEDBACK_INVALID:clock")
                    self.feedback_fault={"gate":fault};raise RuntimeError(fault)
                pose=dict(x=self.x,y=0.,yaw=0.,z=.65,vx=0.,vy=0.,vz=0.,roll=0.,pitch=0.)
                return dict(body_status=1,pose=pose,sim_time_s=clock.t,sample_monotonic_s=clock.t,common_mode_status_value=200 if self.moving else 100,common_mode_desc="LOCOMOTION_STEP" if self.current_mode=="LOCOMOTION_STEP" else "LOCOMOTION_DEFAULT" if self.moving else "STAND_DEFAULT",get_mode_desc="LOCOMOTION_STEP" if self.current_mode=="LOCOMOTION_STEP" else "LOCOMOTION_DEFAULT" if self.moving else "STAND_DEFAULT",get_mode_header_code=0,state_age_s={"get_mode":0.},common_source={"name":"robotsim_official_cmd_vel"},raw_common={"speed_status":{"forward_bounds":{"min_value":-1.,"max_value":1.},"lateral_bounds":{"min_value":-1.,"max_value":1.},"angular_bounds":{"min_value":-1.,"max_value":1.},"speed_mode":{"value":4 if native_step else 1}}})
        class Timer:
            def __init__(self,seconds,callback):self.callback=callback
            def start(self):
                if immediate_watchdog:self.callback()
            def cancel(self):pass
        n=Node();written=[]
        with mock.patch.object(m,"MICRO",{"carry":{"goal_xy_m":[-.3,0.] if reverse_moving_arc else [.3,0.],"timeout_s":5.,"pid":19,"transit_policy_050":transit_policy_050,"reverse_moving_arc":reverse_moving_arc,"motion_mode":"LOCOMOTION_STEP" if native_step else "LOCOMOTION_DEFAULT"}},create=True),mock.patch.object(m,"rclpy",SimpleNamespace(shutdown=lambda:None),create=True),mock.patch("threading.Timer",Timer),mock.patch.object(m.time,"monotonic",lambda:clock.t),mock.patch.object(m.time,"sleep",lambda dt:setattr(clock,"t",clock.t+max(dt,.001))),mock.patch.object(m,"write_durable_milestone",lambda path,value:written.append(json.dumps(value))),contextlib.redirect_stdout(io.StringIO()):
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
    def test_native_step_mode_is_requested_and_confirmed_before_nonzero(self):
        result,n=self.run_route(native_step=True)
        self.assertTrue(result['pass_result'])
        self.assertTrue(result['source_and_motion_mode_observed'])
        self.assertNotIn('source_and_ld_observed',result)
        self.assertEqual(n.stands,1)
        self.assertEqual(sum(e['event']=='STEP' for e in n.events),1)
        self.assertEqual(result['native_step_transition'][0]['common_mode_desc'],'LOCOMOTION_STEP')
    def test_startup_graph_must_refresh_native_sample_before_baseline(self):
        result,n=self.run_route(graph_delay_s=.25,check_cached_native_age=True)
        self.assertTrue(result["pass_result"])
        self.assertIn("pre_motion_composite",result)

    def test_pose_sample_newer_than_loop_start_is_still_fresh(self):
        result,n=self.run_route(pump_delay_s=.003)
        self.assertTrue(result["pass_result"])
        self.assertNotIn("controller_rejection",result)
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

class CarryParentIntegrationTests(unittest.TestCase):
    def test_complete_parent_handoff_branch_without_docker_or_physics(self):
        import json,tempfile,subprocess,time
        from types import SimpleNamespace as NS
        from unittest import mock
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent,native_grasp_audit as audit
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);now=time.monotonic_ns()
            ready=dict(schema="robotsim.x2.carry-ready.v1",pid=19,physical_lift_pass=True,post_lift_retract_pass=True,generation=1,host_monotonic_ns=now)
            native=dict(engine_version=337,host_monotonic_ns=now,generation=1,sim_time_s=80.,process_pid=7,bodies={n:{} for n in ("pelvis","R_omnipicker_base_link","test_cylinder")})
            (out/"native_latest.json").write_text(json.dumps(native)+chr(10))
            (out/"carry_ready.json").write_text(json.dumps(ready))
            (out/"carry_heartbeat.json").write_text(json.dumps(dict(pid=19,host_monotonic_ns=now)))
            (out/"native_tracking_carry_handoff.json").write_text(json.dumps(dict(classification="BODY_RELATIVE_WORLD_UPRIGHT_IK_CARRY",physical_hold_pass=True,generation=1)))
            (out/"native_micro_stop_contract.py").write_text("# fixture")
            args=NS(carry_goal_world_xy=[-.45,1.],carry_timeout_s=5.,carry_sdk_minimum_turn=False,carry_native_action_smoothing=False,carry_native_step_mode=False)
            plan={"lift_preflight":{"retract_preflight":{"pass_result":True,"robot_bodies":["pelvis"],"source_motion_limits":{},"carry_lease_s":30.}}}
            result={"gates":{"OFFICIAL_CMD_VEL_BRIDGE":"PASS"},"commands_not_sent":["locomotion velocity"]}
            calls=[]
            def run(cmd,**kwargs):
                calls.append(cmd)
                (out/"native_carry_owner.stdout").write_text('{"carry_execution":{"pass":true}}')
                if any("native_carry_route.py" in str(x) for x in cmd):
                    (out/"native_carry_route_result.json").write_text(json.dumps(dict(pass_result=True,command_log=[dict(forward=.12,lateral=0.,angular=0.)])))
                    (out/"native_carry_owner.exit").write_text("0")
                return subprocess.CompletedProcess(cmd,0,"")
            with mock.patch.object(parent.pg,"write_new_text",side_effect=lambda p,t:p.write_text(t)),mock.patch.object(parent.pg,"write_new_bytes",side_effect=lambda p,t:p.write_bytes(t)),mock.patch.object(parent,"qualify_static_carry_hold",return_value={"pass":True,"fixture_only":True}),mock.patch.object(parent,"run",run),mock.patch.object(parent.time,"monotonic_ns",return_value=now),mock.patch.object(audit,"iter_complete_native_rows",return_value=[]),mock.patch.object(audit,"evaluate_carry_motion",return_value={"pass":True,"fixture_only":True}):
                completed=parent.execute_grasp_with_carry("fixture", "",out,args,plan,result,48)
            self.assertEqual(completed.returncode,0)
            self.assertEqual(result["gates"]["CHAIR_AREA_ARRIVAL_AND_COMPOSITE_STOP"],"PASS")
            self.assertTrue((out/"carry_release").exists())
            self.assertTrue((out/"point_goal_controller.py").is_file())
            self.assertEqual(len(calls),2)
            self.assertNotIn("locomotion velocity",result["commands_not_sent"])

class DurableMilestoneTests(unittest.TestCase):
    def test_streamed_audit_keeps_late_regression_fail_closed(self):
        from unittest import mock
        from scripts.research.issue46.native_runtime import native_grasp_audit as audit
        def row(stamp):return dict(host_monotonic_ns=stamp,generation=1,engine_version=337,sim_time_s=1.,contacts=[],joints={})
        with mock.patch.object(m,'native_pose',return_value={'vx':0.,'vy':0.}),mock.patch.object(m,'physical_fault',return_value=None):
            passed=audit.evaluate_carry_motion(iter([row(1000000000),row(1020000000)]),{},[],start_ns=1000000000,end_ns=1100000000)
            failed=audit.evaluate_carry_motion(iter([row(1000000000),row(1020000000),row(2000000000),row(1010000000)]),{},[],start_ns=1000000000,end_ns=1100000000)
        self.assertTrue(passed['pass']);self.assertEqual(passed['sample_count'],2)
        self.assertFalse(failed['pass']);self.assertEqual(failed['failure'],'CARRY_NATIVE_GAP')

    def test_atomic_milestone_is_complete_and_refuses_rewrite(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ready.json'
            m.write_durable_milestone(path,{'ready':True})
            self.assertEqual(json.loads(path.read_text()),{'ready':True})
            self.assertFalse(path.with_name(path.name+'.committing').exists())
            with self.assertRaises(FileExistsError):m.write_durable_milestone(path,{'ready':False})
            self.assertEqual(json.loads(path.read_text()),{'ready':True})


class ApprovedLocalFrictionGuardTests(unittest.TestCase):
    def test_unapproved_solver_or_timestep_cannot_generate_a_model(self):
        import json
        from types import SimpleNamespace as NS
        from unittest import mock
        from scripts.research.issue46.native_runtime import claw_parameter_diagnostic as c
        record=dict(classification='SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY',source_pad_collision_comparison={'part_count':22},generated={k:{'path':'/unused','sha256':'same'} for k in ('host_scene','runtime_scene','simulator_config')})
        for options in (dict(npair=1,noslip=0,cone=1,dt=.001),dict(npair=0,noslip=2,cone=1,dt=.001),dict(npair=0,noslip=0,cone=0,dt=.001),dict(npair=0,noslip=0,cone=1,dt=.002)):
            model=NS(npair=options['npair'],opt=NS(noslip_iterations=options['noslip'],cone=options['cone'],impratio=10.,timestep=options['dt']))
            with mock.patch.object(Path,'read_text',return_value=json.dumps(record)),mock.patch.object(c,'digest',return_value='same'),mock.patch.object(c.mujoco.MjModel,'from_xml_path',return_value=model),mock.patch.object(c.pg,'create_unique_output_directory') as allocator:
                with self.assertRaisesRegex(ValueError,'native1ms'):c.add_approved_pad_friction_reference(Path('/unused'))
                allocator.assert_not_called()


class NativeActionSmoothingTests(unittest.TestCase):
    def test_only_two_native_action_pair_rules_are_added(self):
        import copy,yaml
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        base=Path('/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5')
        for path in (base/'action_setting.yaml',base/'sim/action_setting.yaml'):
            source=yaml.safe_load(path.read_text());original=copy.deepcopy(source)
            result=h.build_native_carry_transition_setting(source)
            self.assertEqual(source,original)
            self.assertEqual(result['smooth_transitions'][:2],source['smooth_transitions'])
            self.assertEqual(result['smooth_transitions'][2:],[dict(zip(('from','to','duration_ms'),('STAND_DEFAULT','LOCOMOTION_DEFAULT',500))),dict(zip(('from','to','duration_ms'),('LOCOMOTION_DEFAULT','STAND_DEFAULT',500)))])
            del result['smooth_transitions'];del original['smooth_transitions']
            self.assertEqual(result,original)
        step=h.build_native_carry_transition_setting(yaml.safe_load((base/'action_setting.yaml').read_text()),native_step=True)
        self.assertEqual(len(step['smooth_transitions']),6)
        self.assertEqual(step['smooth_transitions'][-2]['to'],'LOCOMOTION_STEP')
        self.assertEqual(step['smooth_transitions'][-1]['to'],'STAND_DEFAULT')
        with self.assertRaises(ValueError):h.build_native_carry_transition_setting({})


class DisabledPublisherIsolationTests(unittest.TestCase):
    def test_only_carry_isolates_two_disabled_native_endpoints(self):
        import copy
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        cfg={"MotionControlModule":{"PublisherManager":{"publisher_cfg_list":[
            {"name":"arm_joint_command","topic":"/aima/hal/joint/arm/command","enable":True,"frequency":500},
            {"name":"hand_joint_command","topic":"/aima/hal/joint/hand/command","enable":True,"frequency":50},
            {"name":"leg_joint_command","topic":"/aima/hal/joint/leg/command","enable":True,"frequency":500}]}}}
        cfg["aimrt"]={"channel":{"pub_topics_options":[{"topic_name":"/aima/hal/joint/arm/command"},{"topic_name":"/aima/hal/joint/hand/command"}]}}
        original=copy.deepcopy(cfg)
        default=h.build_arm_hand_split_mc_config(cfg)
        isolated=h.build_arm_hand_split_mc_config(cfg,isolate_disabled_topics=True)
        self.assertEqual(cfg,original)
        a=default["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"]
        b=isolated["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"]
        self.assertEqual(a[2],b[2]);self.assertEqual(b[2],original["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"][2])
        for old,new in zip(a[:2],b[:2]):
            self.assertFalse(new["enable"]);self.assertEqual(old["frequency"],new["frequency"])
            self.assertTrue(old["topic"].startswith("/aima/"))
            self.assertTrue(new["topic"].startswith("/robotsim/issue46/inactive_mc/"))

class StaticCarryHoldTests(unittest.TestCase):
    def test_prehold_waits_for_eight_sim_seconds_and_unique_samples(self):
        import json,tempfile
        from types import SimpleNamespace as NS
        from unittest import mock
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as p,native_grasp_audit as a
        clock=NS(t=10.,sim=-.25)
        def row(*args):
            clock.sim+=.25
            return dict(generation=1,sim_time_s=clock.sim,host_monotonic_ns=int(clock.t*1e9))
        def health(path):
            return json.dumps(dict(mode="MEASURED_LOAD_BEARING_CARRY_HOLD",pid=42,generation=1,healthy=True,host_monotonic_ns=int(clock.t*1e9),native_source_host_ns=int(clock.t*1e9)))
        with tempfile.TemporaryDirectory() as d,mock.patch.object(a,"require_fresh_sample",side_effect=row),mock.patch.object(Path,"read_text",health),mock.patch.object(p.time,"monotonic",lambda:clock.t),mock.patch.object(p.time,"monotonic_ns",lambda:int(clock.t*1e9)),mock.patch.object(p.time,"sleep",lambda t:setattr(clock,"t",clock.t+t)):
            result=p.qualify_static_carry_hold(Path(d),dict(pid=42,generation=1))
        self.assertTrue(result["pass"]);self.assertGreaterEqual(result["sim_span_s"],8.);self.assertGreaterEqual(result["unique_native_samples"],25)
    def test_stale_held_object_evidence_cannot_qualify(self):
        import json,tempfile
        from unittest import mock
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as p,native_grasp_audit as a
        with tempfile.TemporaryDirectory() as d,mock.patch.object(a,"require_fresh_sample",return_value=dict(generation=1,sim_time_s=0.)),mock.patch.object(Path,"read_text",return_value=json.dumps(dict(mode="MEASURED_LOAD_BEARING_CARRY_HOLD",pid=42,generation=1,healthy=True,host_monotonic_ns=0,native_source_host_ns=0))):
            with self.assertRaisesRegex(RuntimeError,"HEARTBEAT_UNQUALIFIED"):p.qualify_static_carry_hold(Path(d),dict(pid=42,generation=1))



class CarryStopEvidenceTests(unittest.TestCase):
    def test_zero_publication_does_not_prove_receipt_or_stop(self):
        result=dict(pass_result=False,failure="PLANAR_SPEED",
                    command_log=[dict(t=1.,forward=.12),dict(t=1.1,forward=0.),dict(t=2.15,forward=0.)])
        evidence=m.carry_stop_evidence(result)
        self.assertEqual(evidence["first_zero_after_nonzero_publish_s"],1.1)
        self.assertEqual(evidence["zero_receipt_ack"],"UNAVAILABLE")
        self.assertEqual(evidence["independent_composite_stop"],"NOT_RUN")
        self.assertEqual(evidence["stand_request_count"],0)
        self.assertFalse(evidence["rearm_authorized"])
        self.assertEqual(result["failure"],"PLANAR_SPEED")
    def test_service_acceptance_and_stale_stand_do_not_prove_physical_stop(self):
        row=dict(sample_monotonic_s=2.,get_mode_header_code=0,get_mode_desc="STAND_DEFAULT",
                 common_mode_desc="STAND_DEFAULT",get_mode_status_value=100,common_mode_status_value=100,
                 state_age_s={k:0. for k in ("odom","common","imu","leg_state","leg_command","get_mode")})
        row["state_age_s"]["leg_state"]=.21
        result=dict(events=[dict(event="STAND_REQUEST",t=1.2),
                            dict(event="STAND_RESPONSE",t=1.3,raw={"status":1})],
                    trace=[row],command_log=[dict(t=1.,angular=.1),dict(t=1.1,angular=0.)])
        evidence=m.carry_stop_evidence(result)
        self.assertEqual(evidence["stand_request_count"],1)
        self.assertEqual(evidence["fresh_agreeing_stand_after_zero_sample_count"],0)
        self.assertEqual(evidence["independent_composite_stop"],"NOT_RUN")
        self.assertFalse(evidence["rearm_authorized"])
    def test_complete_gate_is_independent_and_failure_still_prevents_rearm(self):
        result=dict(post_stop_composite={"pass":True},pass_result=True,failure=None)
        self.assertTrue(m.carry_stop_evidence(result)["route_acceptance_recorded"])
        self.assertFalse(m.carry_stop_evidence(result)["rearm_authorized"])
        result["failure"]="BOTTLE_TILTED"
        self.assertFalse(m.carry_stop_evidence(result)["rearm_authorized"])
        result["post_stop_composite"]={"pass":False}
        self.assertEqual(m.carry_stop_evidence(result)["independent_composite_stop"],"FAIL")
    def test_no_motion_cannot_select_prelude_zeros_as_a_motion_stop(self):
        result=dict(command_log=[dict(t=1.,forward=0.),dict(t=2.,forward=0.)])
        evidence=m.carry_stop_evidence(result)
        self.assertIsNone(evidence["first_zero_after_nonzero_publish_s"])
        self.assertEqual(evidence["zero_publish_count_after_nonzero"],0)
        self.assertFalse(evidence["rearm_authorized"])


    def test_duplicate_stand_samples_are_not_counted_as_unique_readbacks(self):
        row=dict(sample_monotonic_s=2.,get_mode_header_code=0,get_mode_desc="STAND_DEFAULT",
                 common_mode_desc="STAND_DEFAULT",get_mode_status_value=100,common_mode_status_value=100,
                 state_age_s={k:.01 for k in ("odom","common","imu","leg_state","leg_command","get_mode")})
        result=dict(command_log=[dict(t=1.,angular=.1),dict(t=1.1,angular=0.)],
                    trace=[row,row,dict(row,sample_monotonic_s=2.02)])
        evidence=m.carry_stop_evidence(result)
        self.assertEqual(evidence["fresh_agreeing_stand_after_zero_sample_count"],2)
        self.assertEqual(evidence["independent_composite_stop"],"NOT_RUN")
        self.assertFalse(evidence["rearm_authorized"])

    def test_summary_failure_preserves_physical_fault_and_zero_cleanup(self):
        from unittest import mock
        with mock.patch.object(m,"carry_stop_evidence",side_effect=ValueError("bad diagnostic")):
            result,n=CarryRouteControlFlowTests().run_route("PLANAR_SPEED")
        self.assertEqual(result["failure"],"PLANAR_SPEED")
        self.assertEqual(result["stop_evidence_error"],"bad diagnostic")
        self.assertFalse(result["pass_result"])
        self.assertFalse(any(x["forward"] or x["angular"] for x in n.command_log))




class CarryTransit050Tests(unittest.TestCase):
    def permit(self):
        return dict(schema="robotsim.carry-transit050.v1",owner_pid=42,route_pid=12,route_start_ticks=123,
                    generation=1,created_ns=10_000_000_000,command_deadline_ns=100_000_000_000,
                    absolute_deadline_ns=115_000_000_000,braking_ns=None,revoked_ns=None)
    def test_only_approved_phase_has_higher_speed_limit(self):
        p=self.permit()
        self.assertEqual(m.carry_transit_speed_limit(None,11_000_000_000,1,42),.20)
        for stamp,expected in ((9_999_999_999,.20),(11_000_000_000,.50),(100_000_000_001,.20)):
            self.assertEqual(m.carry_transit_speed_limit(p,stamp,1,42),expected)
        p["braking_ns"]=20_000_000_000
        self.assertEqual(m.carry_transit_speed_limit(p,34_999_999_999,1,42),.50)
        self.assertEqual(m.carry_transit_speed_limit(p,35_000_000_001,1,42),.20)
        p["revoked_ns"]=23_000_000_000
        self.assertEqual(m.carry_transit_speed_limit(p,24_000_000_000,1,42),.20)
        self.assertEqual(m.carry_transit_speed_limit(p,22_000_000_000,1,42),.50)
    def test_identity_generation_and_deadline_are_fail_closed(self):
        p=self.permit()
        for generation,pid in ((2,42),(1,99)):
            with self.assertRaises(ValueError):m.carry_transit_speed_limit(p,11_000_000_000,generation,pid)
        p["command_deadline_ns"]+=1;p["absolute_deadline_ns"]+=1
        with self.assertRaises(ValueError):m.carry_transit_speed_limit(p,11_000_000_000,1,42)
    def test_default_physical_and_all_other_safety_guards_remain(self):
        row=dict(joints={});pose=dict(z=.65,roll=0.,pitch=0.,vx=.3,vy=0.)
        self.assertEqual(m.physical_fault(row,pose,{}),"PLANAR_SPEED")
        self.assertIsNone(m.physical_fault(row,pose,{},planar_speed_limit=.50))
        pose["vx"]=.501
        self.assertEqual(m.physical_fault(row,pose,{},planar_speed_limit=.50),"PLANAR_SPEED")
        pose["vx"]=.3;pose["z"]=.4
        self.assertEqual(m.physical_fault(row,pose,{},planar_speed_limit=.50),"POSTURE")
        pose["z"]=.65;row["joints"]={"knee":dict(q=0.,qvel=2.,max_abs_qvel_since_previous_sample=2.)}
        self.assertEqual(m.physical_fault(row,pose,{"knee":(-1.,1.,1.)},planar_speed_limit=.50),"SOURCE_JOINT_SPEED:knee")
    def test_policy_flag_cannot_enable_other_motion_modes(self):
        ordinary=m.build_carry_route_script([-.45,1.],{},[],carry_pid=42,transit_policy_050=True)
        self.assertIn("'motion_mode': 'LOCOMOTION_DEFAULT'",ordinary)
        with self.assertRaises(ValueError):
            m.build_carry_route_script([-.45,1.],{},[],carry_pid=0,transit_policy_050=True)
        code=m.build_carry_route_script([-.45,1.],{},[],carry_pid=42,native_step=True,transit_policy_050=True)
        compile(code,"transit050","exec")
        self.assertIn("'transit_policy_050': True",code)
    def test_permit_is_one_shot_and_braking_cannot_extend_it(self):
        import tempfile,json
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(m.time,"monotonic_ns",return_value=10_000_000_000):
            cfg=dict(pid=42,timeout_s=90.)
            p=m.carry_transit_phase(None,"TRANSIT",cfg,{"generation":1},folder)
            with self.assertRaises(RuntimeError):m.carry_transit_phase(None,"TRANSIT",cfg,{"generation":1},folder)
            with mock.patch.object(m.time,"monotonic_ns",return_value=20_000_000_000):
                p=m.carry_transit_phase(p,"BRAKING",cfg,{},folder)
            with mock.patch.object(m.time,"monotonic_ns",return_value=25_000_000_000):
                p=m.carry_transit_phase(p,"BRAKING",cfg,{},folder)
                self.assertEqual(p["braking_ns"],20_000_000_000)
                p=m.carry_transit_phase(p,"REVOKED",cfg,{},folder)
            self.assertEqual(m.carry_transit_speed_limit(p,26_000_000_000,1,42),.20)
    def test_live_process_verification_rejects_non_route_python(self):
        import tempfile,json,os
        from unittest import mock
        p=self.permit();p.update(route_pid=os.getpid(),route_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19]))
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/"carry_transit_permit.json").write_text(json.dumps(p))
            with mock.patch.object(m.time,"monotonic_ns",return_value=11_000_000_000):
                with self.assertRaisesRegex(ValueError,"PROCESS_IDENTITY"):
                    m.current_carry_transit_speed_limit(folder,{"generation":1,"host_monotonic_ns":11_000_000_000},42)





    def test_whole_trace_audit_keeps_pre_and_post_transit_strict(self):
        from unittest import mock
        from scripts.research.issue46.native_runtime import native_grasp_audit as audit
        p=self.permit()
        def row(stamp,speed):
            return dict(host_monotonic_ns=stamp,generation=1,engine_version=337,sim_time_s=stamp/1e9,
                        contacts=[],joints={},speed=speed)
        with mock.patch.object(m,"native_pose",side_effect=lambda r:dict(vx=r["speed"],vy=0.,z=.65,roll=0.,pitch=0.)):
            a=audit.evaluate_carry_motion([row(11_000_000_000,.3),row(11_020_000_000,.3)],{},[],start_ns=0,end_ns=120_000_000_000,transit_policy=p,expected_owner_pid=42)
            self.assertTrue(a["pass"])
            b=audit.evaluate_carry_motion([row(9_900_000_000,.3)],{},[],start_ns=0,end_ns=120_000_000_000,transit_policy=p,expected_owner_pid=42)
            self.assertFalse(b["pass"]);self.assertEqual(b["failure"],"PLANAR_SPEED")
            p["revoked_ns"]=11_010_000_000
            c=audit.evaluate_carry_motion([row(11_000_000_000,.3),row(11_020_000_000,.3)],{},[],start_ns=0,end_ns=120_000_000_000,transit_policy=p,expected_owner_pid=42)
            self.assertFalse(c["pass"]);self.assertEqual(c["failure"],"PLANAR_SPEED")




    def test_actual_route_invocation_propagates_and_revokes_policy(self):
        from unittest import mock
        phases=[]
        def phase(record,name,cfg,row,evidence="/workspace/evidence"):
            phases.append(name)
            self.assertEqual(cfg["pid"],19)
            self.assertEqual(row["generation"],1)
            return {"fixture_phase":name}
        with mock.patch.object(m,"carry_transit_phase",side_effect=phase):
            result,n=CarryRouteControlFlowTests().run_route(native_step=True,transit_policy_050=True)
        self.assertTrue(result["pass_result"])
        self.assertEqual(phases,["TRANSIT","BRAKING","BRAKING","REVOKED"])
        self.assertEqual(result["transit_policy_record"],{"fixture_phase":"REVOKED"})
        self.assertEqual(n.stands,1)




    def test_real_outer_parser_forwarding_and_inner_parser_accept_new_option(self):
        import argparse,sys
        from unittest import mock
        outer=Path(__file__).parents[1]/"grasp/stl_approved_circle_native_trial.py"
        def parser_only(path,variable,arguments):
            tree=ast.parse(path.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
            body=[]
            for node in main.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in (variable,"args") for t in node.targets):
                    body.append(node)
                    if any(isinstance(t,ast.Name) and t.id=="args" for t in node.targets):break
                elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id==variable:body.append(node)
            ns=dict(argparse=argparse,Path=Path,__doc__="offline actual parser")
            with mock.patch.object(sys,"argv",["offline",*arguments]):
                exec(compile(ast.Module(body=body,type_ignores=[]),"actual_parser","exec"),ns)
            return ns["args"],main
        args,main=parser_only(outer,"p",["--baseline","/unused","--circle-review","/unused","--candidate","/unused",
            "--carry-goal-world-xy","-.45","1.","--carry-native-step-mode","--carry-native-action-smoothing",
            "--approved-pad-friction-reference","--carry-transit-speed-policy-050"])
        branch=next(n for n in main.body if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=="argv" and "carry_timeout_s" in ast.unparse(x) for x in ast.walk(n)))
        ns=dict(args=args,argv=[])
        exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_forwarding","exec"),ns)
        self.assertIn("--carry-transit-speed-policy-050",ns["argv"])
        inner=Path(__file__).with_name("xvfb_native_reset_stand.py")
        parsed,_=parser_only(inner,"parser",["--table-overlay-result","/unused","--table-overlay-result-sha256","unused",
            "--reset-helper","/unused","--reset-helper-sha256","unused","--claw-control-overlay-result","/unused","--claw-control-overlay-result-sha256","unused",*ns["argv"]])
        self.assertTrue(parsed.carry_transit_speed_policy_050)
        self.assertTrue(parsed.carry_native_step_mode)
        inner_main=next(n for n in ast.parse(inner.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="main")
        guard=next(n for n in inner_main.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith("args.carry_transit_speed_policy_050"))
        exec(compile(ast.Module(body=[guard],type_ignores=[]),"actual_native_guard","exec"),{"args":parsed})
        parsed.claw_control_overlay_result=None
        with self.assertRaises(ValueError):
            exec(compile(ast.Module(body=[guard],type_ignores=[]),"actual_native_guard","exec"),{"args":parsed})
        args.carry_native_step_mode=False;args.carry_reverse_moving_arc=True
        ns=dict(args=args,argv=[])
        exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_reverse_forwarding","exec"),ns)
        self.assertIn("--carry-reverse-moving-arc",ns["argv"])
        reverse,_=parser_only(inner,"parser",["--table-overlay-result","/unused","--table-overlay-result-sha256","unused",
            "--reset-helper","/unused","--reset-helper-sha256","unused","--claw-control-overlay-result","/unused",
            "--claw-control-overlay-result-sha256","unused",*ns["argv"]])
        reverse_guard=next(n for n in inner_main.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith("args.carry_reverse_moving_arc"))
        exec(compile(ast.Module(body=[reverse_guard],type_ignores=[]),"actual_reverse_guard","exec"),{"args":reverse})
        args.carry_reverse_moving_arc=False;args.carry_away_1m=True;args.carry_goal_world_xy=None;args.allow_carried_bottle_tilt=True
        ns=dict(args=args,argv=[])
        exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_away_forwarding","exec"),ns)
        self.assertIn("--carry-away-1m",ns["argv"])
        self.assertNotIn("--carry-goal-world-xy",ns["argv"])
        self.assertIn("--allow-carried-bottle-tilt",ns["argv"])
        away,_=parser_only(inner,"parser",["--table-overlay-result","/unused","--table-overlay-result-sha256","unused",
            "--reset-helper","/unused","--reset-helper-sha256","unused","--claw-control-overlay-result","/unused",
            "--claw-control-overlay-result-sha256","unused",*ns["argv"]])
        self.assertTrue(away.allow_carried_bottle_tilt)
        away_guard=next(n for n in inner_main.body if isinstance(n,ast.If) and ast.unparse(n.test).startswith("args.carry_away_1m"))
        exec(compile(ast.Module(body=[away_guard],type_ignores=[]),"actual_away_guard","exec"),{"args":away})






class CarrySchedulingRepairTests(unittest.TestCase):
    def test_graph_work_does_not_block_caller_and_deadline_stops(self):
        import threading
        from types import SimpleNamespace
        release=threading.Event();entered=threading.Event();zeros=[]
        def graph():entered.set();release.wait(1.)
        node=SimpleNamespace(graph=graph,deadline_zero=lambda:zeros.append(True),events=[])
        job=m.start_carry_graph_check(node)
        try:
            self.assertTrue(entered.wait(.1))
            self.assertIs(m.poll_carry_graph_check(node,job,job["started"]+.10),job)
            with self.assertRaisesRegex(RuntimeError,"GRAPH_CHECK_DEADLINE"):
                m.poll_carry_graph_check(node,job,job["started"]+.201)
            self.assertEqual(len(zeros),1)
        finally:release.set();job["thread"].join(.3)
    def test_graph_identity_failure_is_still_fail_closed(self):
        from types import SimpleNamespace
        zeros=[]
        def graph():raise RuntimeError("CARRY_PUBLISHER_NOT_UNIQUE")
        node=SimpleNamespace(graph=graph,deadline_zero=lambda:zeros.append(True),events=[])
        job=m.start_carry_graph_check(node);job["thread"].join(.1)
        with self.assertRaisesRegex(RuntimeError,"PUBLISHER_NOT_UNIQUE"):
            m.poll_carry_graph_check(node,job,job["started"]+.01)
        self.assertTrue(zeros)
    def test_left_bias_only_applies_in_existing_carry_stage_and_rejects_nan(self):
        from scripts.research.issue46.native_runtime.test_pregrasp_feedback import generated
        # Real generated owner code; extract the new stage branch only.
        code=generated()
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        import inspect
        source=inspect.getsource(h._with_carry_lease)
        tree=ast.parse(source)
        replacement=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                         and n.func.id=="replace" and isinstance(n.args[0],ast.Constant)
                         and "bias=({" in str(n.args[0].value))
        branch_code=replacement.args[1].value
        branch=next(n for n in ast.parse(inspect.cleandoc(branch_code)).body if isinstance(n,ast.If))
        from types import SimpleNamespace
        row=SimpleNamespace(native_row={"joints":{"left":{"bias_force":1.28}}})
        ns=dict(stage="CARRY_HOLD",USE_NATIVE_BIAS=True,USE_NATIVE_FEEDBACK=True,LEFT=["left"],self=row,bias={},math=math)
        exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_left_bias","exec"),ns)
        self.assertEqual(ns["bias"],{"left":1.28})
        ns.update(stage="LIFT_HOLD",bias={})
        exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_left_bias","exec"),ns)
        self.assertEqual(ns["bias"],{})
        ns["stage"]="CARRY_HOLD";row.native_row["joints"]["left"]["bias_force"]=float("nan")
        with self.assertRaisesRegex(RuntimeError,"NONFINITE_LEFT"):
            exec(compile(ast.Module(body=[branch],type_ignores=[]),"actual_left_bias","exec"),ns)





class CarryDebugTimingTests(unittest.TestCase):
    def test_carry_capture_starts_after_static_hold_and_before_motion(self):
        source=Path(__file__).with_name("xvfb_native_reset_stand.py").read_text()
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_grasp_with_carry")
        code=ast.unparse(fn)
        self.assertLess(code.index("qualify_static_carry_hold"),code.index("start_runtime_rl_debug_capture"))
        self.assertLess(code.index("start_runtime_rl_debug_capture"),code.index("route = run("))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        calls=[n for n in ast.walk(main) if isinstance(n,ast.If) and ast.unparse(n.test).startswith("args.capture_mc_rl_debug and") and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=="start_runtime_rl_debug_capture" for part in n.body for x in ast.walk(part))]
        self.assertEqual(len(calls),1)




class CarryTurnCapTests(unittest.TestCase):
    def test_away_turn_obeys_existing_explicit_angular_cap(self):
        from types import SimpleNamespace
        for cap in (.075,.10):
            code=m.build_carry_route_script(None,{},[],carry_pid=42,transit_policy_050=True,away_1m=True,angular_command_cap=cap)
            tree=ast.parse(code)
            assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='output' for t in n.targets) and isinstance(n.value,ast.Call) and 'turn_error' in ast.unparse(n.value))
            for error in (-3.,3.,-.01):
                ns={'cfg':{'angular_command_cap':cap},'turn_error':error,'SimpleNamespace':SimpleNamespace,'NavState':SimpleNamespace(ALIGN='align')}
                exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual_away_turn','exec'),ns)
                self.assertLessEqual(abs(ns['output'].angular_z),cap)
                self.assertEqual(ns['output'].linear_x,-.12)
            self.assertIn('planar_speed_limit=.20',code)

class MeasuredForceTargetTests(unittest.TestCase):
    def test_higher_control_target_does_not_raise_load_acceptance(self):
        from scripts.research.issue46.grasp.stl_approved_circle_ik import grip_force_budget
        minimum,target=grip_force_budget(.2*9.81,1.,6.)
        self.assertAlmostEqual(minimum,1.4715)
        self.assertAlmostEqual(target,5.886)
        self.assertEqual(grip_force_budget(.2*9.81,1.),(minimum,minimum))
        with self.assertRaises(ValueError):grip_force_budget(.2*9.81,1.,12.)




class ReverseMovingArcTests(unittest.TestCase):
    def test_reverse_route_stops_by_measured_distance_and_full_existing_stop(self):
        result,n=CarryRouteControlFlowTests().run_route(reverse_moving_arc=True)
        self.assertTrue(result["pass_result"])
        self.assertTrue(result["position_only_chair_area"])
        self.assertEqual(result["goal_yaw_requirement"],"NOT_REQUIRED_FOR_KNOWN_CHAIR_AREA")
        nonzero=[c for c in n.command_log if c["forward"] or c["angular"]]
        self.assertTrue(nonzero);self.assertTrue(all(c["forward"]==-.12 for c in nonzero))
        self.assertLessEqual(result["final_distance_m"],.15)
        self.assertTrue(result["post_stop_composite"]["pass"])
        self.assertEqual(n.stands,1)
    def test_reverse_profile_cannot_select_step_or_omit_phase_policy(self):
        for policy,step in ((False,False),(True,True)):
            with self.assertRaises(ValueError):
                m.build_carry_route_script([-.45,1.],{},[],carry_pid=42,transit_policy_050=policy,native_step=step,reverse_moving_arc=True)
        code=m.build_carry_route_script([-.45,1.],{},[],carry_pid=42,transit_policy_050=True,reverse_moving_arc=True)
        compile(code,"reverse_carry","exec")
        self.assertIn("'reverse_moving_arc': True",code)




class SourceBoundedCarryParkTests(unittest.TestCase):
    def test_non_grasping_arm_park_uses_source_margin_without_changing_limits(self):
        from scripts.research.issue46.grasp.stl_approved_circle_ik import carry_left_roll_park
        bounds=[-.061,2.993];park=carry_left_roll_park(-.027,bounds)
        self.assertAlmostEqual(park["target_rad"],.089)
        self.assertEqual(bounds,[-.061,2.993])
        self.assertLess(park["target_rad"]-park["start_rad"],.20)
        self.assertEqual(carry_left_roll_park(.2,bounds)["target_rad"],.2)
        with self.assertRaises(ValueError):carry_left_roll_park(-.2,bounds)
        with self.assertRaises(ValueError):carry_left_roll_park(-.027,[0.,-.1])




class RuntimePlanningBoundsTests(unittest.TestCase):
    def test_intersection_changes_search_constraints_without_editing_model(self):
        import mujoco,numpy as np
        from scripts.research.issue46.grasp.stl_approved_circle_ik import joint_planning_bounds
        model=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body><joint name="right_wrist_pitch_joint" range="-1 1"/><geom size=".01"/></body></worldbody></mujoco>')
        original=model.jnt_range.copy();name="right_wrist_pitch_joint"
        bounds=joint_planning_bounds(model,[name],{name:[-.523599,.523599,4.188]})
        np.testing.assert_allclose(bounds,[[-.523599,.523599]])
        np.testing.assert_array_equal(model.jnt_range,original)
        np.testing.assert_array_equal(joint_planning_bounds(model,[name],{name:[-2.,2.]}),original)
        with self.assertRaises(ValueError):joint_planning_bounds(model,[name],{})




class AwayOneMetreControlFlowTests(unittest.TestCase):
    def run_mission(self,timeout=90.,turn_stop_pass=True):
        import sys,threading,tempfile,json,contextlib,io
        from types import SimpleNamespace as NS
        from unittest import mock
        sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"ros2_ws/src/robotsim_navigation/robotsim_navigation"))
        clock=NS(t=100.)
        class Node:
            def __init__(self):
                self.lock=threading.Lock();self.moving=False;self.stop_started=False;self.feedback_fault=None
                self.events=[];self.command_log=[];self.bridge_commands=[];self.trace=[]
                self.messages={k:{} for k in ("common","imu","leg_state","leg_command")}
                self.readbacks={"get_mode":(100.,{})};self.latest_row={"generation":1}
                self.x=.85;self.y=-1.5;self.yaw=.1;self.last_t=100.;self.command=(0.,0.);self.stands=0
            def publish(self,f=0.,l=0.,a=0.):
                if not self.moving:f=a=0.
                self.command=(f,a);self.command_log.append(dict(t=clock.t,forward=f,lateral=0.,angular=a))
            def pump(self):
                dt=clock.t-self.last_t;f,a=self.command
                self.x+=f*math.cos(self.yaw)*dt;self.y+=f*math.sin(self.yaw)*dt
                self.yaw=(self.yaw+a*dt+math.pi)%(2*math.pi)-math.pi;self.last_t=clock.t
            def sample(self):
                self.pump();active=self.moving and any(self.command)
                return dict(body_status=1,sample_monotonic_s=clock.t,sim_time_s=clock.t,
                    pose=dict(x=self.x,y=self.y,yaw=self.yaw,z=.65,vx=0.,vy=0.,vz=0.,roll=0.,pitch=0.),
                    get_mode_header_code=0,get_mode_desc="LOCOMOTION_DEFAULT" if active else "STAND_DEFAULT",
                    common_mode_desc="LOCOMOTION_DEFAULT" if active else "STAND_DEFAULT",
                    common_mode_status_value=100,state_age_s={"get_mode":0.},
                    common_source={"name":"robotsim_official_cmd_vel"},
                    raw_common={"speed_status":{"forward_bounds":{"min_value":-1.,"max_value":1.},"lateral_bounds":{"min_value":-1.,"max_value":1.},"angular_bounds":{"min_value":-1.,"max_value":1.}}})
            def observe(self,seconds,**kw):
                self.publish();clock.t+=2.1 if kw.get("require_stop") else seconds;self.pump()
                passed=turn_stop_pass if self.stands==1 and kw.get("require_stop") else True
                result={"pass":passed,"fixture_only":True,"metrics":{"duration_s":2.,"sample_count":100,"window_end_monotonic_s":clock.t}}
                self.events.append(dict(event="OBSERVE_FIXTURE",t=clock.t,stands=self.stands,require_stop=kw.get("require_stop",False),pass_result=passed))
                return result
            def deadline_zero(self):
                self.moving=False;self.publish();self.events.append(dict(event="DEADLINE_ZERO_ATTEMPT",t=clock.t))
            def stand_once(self):
                self.stands+=1;self.events.append(dict(event="STAND_REQUEST",t=clock.t))
            def graph(self):pass
            def await_graph(self):self.graph()
            def start_feedback(self):pass
            def stop_feedback(self):pass
            def destroy_node(self):pass
        class Timer:
            durations=[]
            def __init__(self,seconds,callback):self.durations.append(seconds)
            def start(self):pass
            def cancel(self):pass
        n=Node();written=[]
        real_phase=m.carry_transit_phase
        with tempfile.TemporaryDirectory() as folder:
            def phase(record,name,cfg,row):return real_phase(record,name,cfg,row,folder)
            with mock.patch.object(m,"MICRO",{"carry":{"goal_xy_m":None,"timeout_s":timeout,"pid":19,"transit_policy_050":True,"away_1m":True,"angular_command_cap":.10,"motion_mode":"LOCOMOTION_DEFAULT"}},create=True),mock.patch.object(m,"rclpy",NS(shutdown=lambda:None),create=True),mock.patch("threading.Timer",Timer),mock.patch.object(m.time,"monotonic",lambda:clock.t),mock.patch.object(m.time,"monotonic_ns",lambda:int(clock.t*1e9)),mock.patch.object(m.time,"sleep",lambda dt:setattr(clock,"t",clock.t+max(.001,dt))),mock.patch.object(m,"carry_transit_phase",side_effect=phase),mock.patch.object(m,"start_carry_graph_check",return_value=None),mock.patch.object(m,"write_durable_milestone",lambda path,value:written.append(json.dumps(value))),contextlib.redirect_stdout(io.StringIO()):
                m.run_carry_navigation(n,NS())
        return json.loads(written[-1]),n,Timer.durations
    def test_full_task_uses_measured_turn_stop_then_a_separate_forward_metre(self):
        result,n,timers=self.run_mission()
        self.assertTrue(result["pass_result"],result.get("failure"))
        self.assertTrue(result["turn_stop_composite"]["pass"])
        mission=result["away_mission"]
        self.assertGreaterEqual(mission["final_forward_projection_m"],1.)
        self.assertLessEqual(result["final_distance_m"],.15)
        self.assertLessEqual(abs(mission["final_heading_error_rad"]),.12)
        self.assertEqual(n.stands,2);self.assertEqual(len(timers),1)
        positive=next(c["t"] for c in n.command_log if c["forward"]>0)
        self.assertGreaterEqual(positive,result["turn_stop_composite"]["metrics"]["window_end_monotonic_s"])
        records=result["transit_policy_records"]
        self.assertEqual(len(records),2)
        self.assertLessEqual(records[0]["revoked_ns"],records[1]["created_ns"])
        self.assertLessEqual(records[1]["command_deadline_ns"],records[0]["command_deadline_ns"])
    def test_failed_turn_stop_never_allows_forward_commands(self):
        result,n,_=self.run_mission(turn_stop_pass=False)
        self.assertFalse(result["pass_result"])
        self.assertEqual(result["failure"],"AWAY_TURN_STOP_NOT_QUALIFIED")
        self.assertFalse(any(c["forward"]>0 for c in n.command_log))
    def test_mid_stop_does_not_reset_total_command_budget(self):
        result,n,timers=self.run_mission(timeout=35.)
        self.assertFalse(result["pass_result"])
        self.assertEqual(result["failure"],"AWAY_FORWARD_AGGREGATE_DEADLINE_INSUFFICIENT")
        self.assertFalse(any(c["forward"]>0 for c in n.command_log))
        self.assertEqual(len(timers),1)




class SequentialCarryPermitTests(unittest.TestCase):
    def test_second_interval_requires_revoked_first_and_cannot_overlap(self):
        first=CarryTransit050Tests().permit();first["revoked_ns"]=20_000_000_000
        second=dict(first,created_ns=21_000_000_000,command_deadline_ns=100_000_000_000,
                    absolute_deadline_ns=115_000_000_000,revoked_ns=40_000_000_000)
        self.assertEqual(m.carry_policy_speed_limit([first,second],19_000_000_000,1,42),.50)
        self.assertEqual(m.carry_policy_speed_limit([first,second],20_500_000_000,1,42),.20)
        self.assertEqual(m.carry_policy_speed_limit([first,second],22_000_000_000,1,42),.50)
        first["revoked_ns"]=None
        with self.assertRaises(ValueError):m.carry_policy_speed_limit([first,second],22_000_000_000,1,42)
    def test_turn_stop_report_cannot_cover_later_forward_motion(self):
        gate={"pass":True,"metrics":{"window_end_monotonic_s":4.}}
        result=dict(turn_stop_composite=gate,command_log=[dict(t=1.,forward=-.12),dict(t=2.,forward=0.)])
        report=m.carry_stop_evidence(result)
        self.assertEqual(report["independent_composite_stop"],"PASS")
        self.assertEqual(report["composite_stop_scope"],"TURN_ONLY")
        result["command_log"].append(dict(t=5.,forward=.12))
        self.assertEqual(m.carry_stop_evidence(result)["independent_composite_stop"],"NOT_RUN")




class EmptyBottleOwnerGuardTests(unittest.TestCase):
    def test_actual_owner_guard_keeps_slip_and_drop_checks(self):
        import inspect
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        source=inspect.getsource(h._with_carry_lease)
        tree=ast.parse(source)
        value=next(n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="replace"
                   and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and "CARRY_OBJECT_POSE_OR_SLIP" in str(n.args[1].value))
        line=next(l.strip() for l in value.splitlines() if "CARRY_OBJECT_POSE_OR_SLIP:" in l and l.strip().startswith("if "))
        expr=line[len("if "):line.index(":raise")]
        code=compile(ast.parse(expr,mode="eval"),"actual_owner_guard","eval")
        ns=dict(slip=0.,B=[0.]*8+[math.cos(.4)],obj={"position_m":[0.,0.,.75]},math=math,
                LIFT_CONTRACT={"retract_preflight":{}})
        self.assertTrue(eval(code,ns))
        ns["LIFT_CONTRACT"]["retract_preflight"]["allow_carried_bottle_tilt"]=True
        self.assertFalse(eval(code,ns))
        ns["slip"]=.011;self.assertTrue(eval(code,ns))
        ns["LIFT_CONTRACT"]["retract_preflight"]["carry_retention_only"]=True
        self.assertFalse(eval(code,ns))
        ns["slip"]=0.;ns["obj"]["position_m"][2]=.2;self.assertTrue(eval(code,ns))

class InplaceCentreHoldTests(unittest.TestCase):
    def test020_publisher_rejects_rotation_or_other_phase_at_the_higher_limit(self):
        from types import SimpleNamespace as NS
        code=m.build_carry_route_script(None,{},[],carry_pid=42,away_1m=True,
          native_action_smoothing=True,transit_policy_050=True,angular_command_cap=.1,
          straight_retreat_turn=True,inplace_centre_hold=True,turn_recovery_speed=.20)
        tree=ast.parse(code)
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'RECOVERY020_OUTSIDE_POSITION_ONLY_PHASE' in ast.unparse(n) and any(isinstance(stmt,ast.Raise) and 'RECOVERY020_OUTSIDE_POSITION_ONLY_PHASE' in ast.unparse(stmt) for stmt in n.body))
        cfg={'active_centre_recovery':True,'active_mission_phase':'TURN','turn_recovery_speed':.20}
        ns={'MICRO':{'carry':cfg},'fwd':.2,'lat':0.,'ang':0.,'math':math}
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual020_scope','exec'),ns)
        ns['ang']=.1
        with self.assertRaisesRegex(RuntimeError,'POSITION_ONLY'):exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual020_scope','exec'),ns)
        ns['ang']=0.;cfg['active_centre_recovery']=False
        with self.assertRaisesRegex(RuntimeError,'POSITION_ONLY'):exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual020_scope','exec'),ns)

    def test_prepared020_recovery_is_opt_in_and_single_axis_only(self):
        origin={'x':0.,'y':0.}
        self.assertEqual(m.inplace_recovery_command({'x':-.03,'y':0.,'yaw':0.},origin,.20),(.20,0.))
        self.assertEqual(m.inplace_recovery_command({'x':0.,'y':-.03,'yaw':0.},origin,.20),(0.,.20))
        with self.assertRaises(ValueError):m.inplace_recovery_command({'x':0.,'y':0.,'yaw':0.},origin,.3)
        with self.assertRaises(ValueError):m.build_carry_route_script([0.,0.],{},[],carry_pid=42,turn_recovery_speed=.20)

    def test_recovery_uses_eligible_axis_commands_inside_the_same_total_cap(self):
        origin={'x':0.,'y':0.}
        self.assertEqual(m.inplace_recovery_command({'x':-.03,'y':0.,'yaw':0.},origin),(.12,0.))
        self.assertEqual(m.inplace_recovery_command({'x':0.,'y':-.03,'yaw':0.},origin),(0.,.05))
        for yaw in (0.,.7,1.5,3.):
            f,l=m.inplace_recovery_command({'x':-.03,'y':.02,'yaw':yaw},origin)
            self.assertLessEqual(math.hypot(f,l),.12)
            self.assertTrue((abs(f)==.12 and l==0.) or (f==0. and abs(l)==.05))

    def test_startup_is_bounded_monotonic_and_reaches_the_original_cap(self):
        values=[m.bounded_start_scale(t/100.) for t in range(51)]
        self.assertEqual(values[0],0.)
        self.assertEqual(values[-1],1.)
        self.assertEqual(values,sorted(values))
        self.assertTrue(all(0<=v<=1 for v in values))
        self.assertEqual(m.bounded_start_scale(1.),1.)
        with self.assertRaisesRegex(RuntimeError,'CLOCK'):m.bounded_start_scale(float('nan'))

    def test_world_error_is_rotated_to_body_and_total_speed_is_bounded(self):
        pose={'x':.1,'y':0.,'yaw':math.pi/2};origin={'x':0.,'y':0.}
        f,l=m.inplace_centre_command(pose,origin)
        self.assertAlmostEqual(f,0.,places=12);self.assertAlmostEqual(l,.12)
        for x,y,yaw in ((.03,.04,0.),(-.3,.4,1.2),(.005,0.,0.)):
            f,l=m.inplace_centre_command(dict(x=x,y=y,yaw=yaw),origin)
            self.assertLessEqual(math.hypot(f,l),.120000001)
        self.assertEqual(m.inplace_centre_command(dict(x=.005,y=0.,yaw=0.),origin),(0.,0.))
        with self.assertRaisesRegex(RuntimeError,'INVALID_TURN'):
            m.inplace_centre_command(dict(x=float('nan'),y=0.,yaw=0.),origin)

    def test_pose_hold_is_scoped_to_inplace_and_brake_predictor_is_bounded(self):
        code=m.build_carry_route_script(None,{},[],carry_pid=42,away_1m=True,
          native_action_smoothing=True,transit_policy_050=True,angular_command_cap=.1,
          straight_retreat_turn=True,inplace_centre_hold=True,turn_brake_delay_s=2.73)
        compile(code,'centre_hold','exec')
        self.assertIn('LATERAL_COMMAND_OUTSIDE_INPLACE_TURN',code)
        self.assertIn('OBSERVED_BRIDGE_COMMAND_BOUND',code)
        self.assertIn('reserve=min(.75',code)
        self.assertIn('bound=.20 if cfg.get("compact_turn") else .05',code)
        self.assertIn('if drift>bound:raise RuntimeError',code)
        with self.assertRaises(ValueError):
            m.build_carry_route_script([0.,0.],{},[],carry_pid=42,inplace_centre_hold=True)

class PreMotionReadinessTests(unittest.TestCase):
    def test_reacquire_never_applies_after_nonzero_or_to_physical_faults(self):
        code=m.build_carry_route_script(None,{},[],carry_pid=42,away_1m=True,
             native_action_smoothing=True,transit_policy_050=True,
             angular_command_cap=.10,straight_retreat_turn=True)
        tree=ast.parse(code)
        assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='initial_zero_only' for t in n.targets))
        from types import SimpleNamespace as NS
        ns=dict(MICRO={'carry':True},self=NS(moving=False,command_log=[dict(forward=0.,angular=0.)]))
        exec(compile(ast.Module(body=[assign],type_ignores=[]),'zero_only','exec'),ns)
        self.assertTrue(ns['initial_zero_only'])
        ns['self'].command_log.append(dict(forward=.12))
        exec(compile(ast.Module(body=[assign],type_ignores=[]),'zero_only','exec'),ns)
        self.assertFalse(ns['initial_zero_only'])
        self.assertIn('str(exc).startswith("CRITICAL_ROS_TELEMETRY_STALE:")',code)
        self.assertIn('min(3.,seconds)',code)
        self.assertIn('qualified=None',code)

class StraightRetreatInplaceTests(unittest.TestCase):
    def test_turn_has_zero_translation_and_rearm_is_after_measured_stop(self):
        code=m.build_carry_route_script(None,{},[],carry_pid=42,away_1m=True,
             native_step=True,native_action_smoothing=True,transit_policy_050=True,
             angular_command_cap=.10,straight_retreat_turn=True)
        tree=ast.parse(code)
        assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='forward' for t in n.targets) and
                    'straight_retreat_turn' in ast.unparse(n.value))
        from types import SimpleNamespace
        for phase,expected in [('RETREAT',-.12),('TURN',0.)]:
            ns=dict(cfg={'straight_retreat_turn':True},mission={'phase':phase},output=SimpleNamespace(linear_x=-.12))
            exec(compile(ast.Module(body=[assign],type_ignores=[]),'actual_command','exec'),ns)
            self.assertEqual(ns['forward'],expected)
        self.assertIn('RETREAT_CROSS_TRACK',code)
        self.assertIn('INPLACE_TURN_CENTER_TRANSLATION',code)
        self.assertIn('mission["previous_yaw"]=state["pose"]["yaw"]',code)
        self.assertIn('mission["target_unwrapped_yaw"]=state["pose"]["yaw"]-math.pi',code)
        self.assertIn('result["delivery_table_placement"]="NOT_RUN"',code)
        self.assertLess(code.index('result["retreat_stop_composite"]=n.observe'),code.index('n.stand_once("LOCOMOTION_STEP")',code.index('result["retreat_stop_composite"]=n.observe')))
        self.assertLess(code.index('result["retreat_stop_composite"]=n.observe'),code.index('mission["phase"]="TURN"'))
        compile(code,'retreat_inplace','exec')

if __name__=="__main__":unittest.main()

class EffectiveTransit020Tests(unittest.TestCase):
    def test_opt_in_is_scoped_and_keeps_defaults(self):
        kw=dict(carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,inplace_centre_hold=True,turn_recovery_speed=.20)
        code=m.build_carry_route_script(None,{},['pelvis'],transit_command_speed=.20,**kw)
        compile(code,'effective020','exec')
        self.assertIn("'transit_command_speed': 0.2",code)
        self.assertEqual(code.count('else MICRO["carry"].get("transit_command_speed",.12)'),2)
        self.assertIn('cfg.get("transit_command_speed",.12)==.12',code)
        self.assertIn('ACTIVE_ROBOT_SELF_COLLISION',code)
        default=m.build_carry_route_script([0.,0.],{},[],carry_pid=19)
        self.assertIn("'transit_command_speed': 0.12",default)
        for speed in (.21,float('nan')):
            with self.assertRaises(ValueError):m.build_carry_route_script(None,{},[],transit_command_speed=speed,**kw)
        with self.assertRaises(ValueError):m.build_carry_route_script([0.,0.],{},[],carry_pid=19,transit_command_speed=.20)

class GlobalImpedanceComparisonTests(unittest.TestCase):
    def test_only_global_impedance_changes_on_actual_compilation(self):
        import tempfile,json
        from unittest import mock
        from scripts.research.issue46.native_runtime import claw_parameter_diagnostic as c
        from types import SimpleNamespace as NS
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);out=d/"new";out.mkdir()
            xml='<mujoco><option cone="elliptic" impratio="10" timestep=".001"/><worldbody><geom type="sphere" size=".05"/></worldbody></mujoco>'
            rec={'approved_local_pad_friction_reference':{'pair_count':22},'generated':{},'diagnostic_parameter_delta':{'elliptic_solver':{'cone_to':1,'impratio_to':10.}}}
            for label in ('host_scene','runtime_scene','runtime_hostcheck_scene'):
                src=d/(label+'.xml');src.write_text(xml);rec['generated'][label]={'path':str(src),'sha256':c.digest(src)}
            cfg=d/'sim.yaml';cfg.write_text('MujocoSimModule: {simulation_model: unused}')
            rec['generated']['simulator_config']={'path':str(cfg),'sha256':c.digest(cfg)}
            parent=d/'result.json';parent.write_text(json.dumps(rec));before=parent.read_bytes()
            with mock.patch.object(c.pg,'create_unique_output_directory',return_value=NS(native_path=out)),mock.patch.object(c.pg,'write_artifact_manifest'),mock.patch.object(c.pg,'write_new_bytes',side_effect=lambda p,b:Path(p).write_bytes(b)),mock.patch.object(c.pg,'write_new_text',side_effect=lambda p,s:Path(p).write_text(s)):
                result=c.compare_global_impratio_one(parent)
            generated=json.loads(result.read_text())
            a=c.mujoco.MjModel.from_xml_path(generated['generated']['host_scene']['path'])
            self.assertEqual(a.opt.impratio,1.)
            self.assertEqual(int(a.opt.cone),1)
            self.assertEqual(parent.read_bytes(),before)
            self.assertTrue(generated['global_impratio_comparison']['physical_arrays_identical'])
            self.assertIn('approved_local_pad_friction_reference',generated)

class CompactTurnTests(unittest.TestCase):
    def test_compact_profile_keeps_actual_pure_yaw_and_bounded_measured_radius(self):
        code=m.build_carry_route_script(None,{},['pelvis'],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        compile(code,'compact_turn','exec')
        self.assertIn("'angular_command_cap': 0.3",code)
        self.assertIn('COMPACT_TURN_RADIUS_300MM',code)
        self.assertIn('bound=.20 if cfg.get("compact_turn") else .05',code)
        self.assertIn('0. if cfg.get("straight_retreat_turn") and mission["phase"]=="TURN"',code)
        self.assertIn('retreat_goal=.50 if cfg.get("compact_turn") else .30',code)
        with self.assertRaises(ValueError):m.build_carry_route_script([0.,0.],{},[],carry_pid=19,compact_turn=True)

class CompactClockRecoveryTests(unittest.TestCase):
    def test_pending_epoch_is_not_misclassified_but_physical_faults_never_rearm(self):
        self.assertTrue(m.recoverable_carry_clock_fault({'gate':'CLOCK_STEP_DURING_MOTION'},RuntimeError('CLOCK_GENERATION_PENDING'),0))
        self.assertFalse(m.recoverable_carry_clock_fault({'gate':'PLANAR_SPEED'},RuntimeError('CLOCK_GENERATION_PENDING'),0))
        self.assertFalse(m.recoverable_carry_clock_fault({'gate':'CLOCK_STEP_DURING_MOTION'},RuntimeError('SOURCE_JOINT_SPEED:knee'),0))
        self.assertFalse(m.recoverable_carry_clock_fault({'gate':'CLOCK_STEP_DURING_MOTION'},RuntimeError('CLOCK_GENERATION_PENDING'),4))
    def test_compact_generated_path_requires_stop_and_keeps_absolute_deadline(self):
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True)
        compile(code,'compact_clock','exec')
        self.assertIn('cfg["clock_stop_pass"]=bool(stopped["pass"])',code)
        self.assertIn('CLOCK_RECOVERY_TASK_DEADLINE_INSUFFICIENT',code)
        self.assertIn('CARRY_POLICY_DEADLINE_RESET',code)
        self.assertIn('CLOCK_RECOVERY_REARM_NOT_QUALIFIED',code)
        self.assertLess(code.index('stopped=n.observe(8.,baseline=baseline,require_stop=True)'),code.index('cfg["clock_stop_pass"]=bool(stopped["pass"])'))

class RealCompactBridgeArgumentTests(unittest.TestCase):
    def test_complete_actual_bridge_argument_path_does_not_overwrite_yaw_cap(self):
        import shlex
        from types import SimpleNamespace as NS
        tree=ast.parse(Path(__file__).with_name('xvfb_native_reset_stand.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_grasp_with_carry')
        bridge_if=next(n for n in func.body if isinstance(n,ast.If) and any(isinstance(c,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bridge_command' for t in c.targets) for c in n.body))
        statements=[n for n in bridge_if.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('bridge_linear','carry_angular','bridge_command') for t in n.targets) or isinstance(n,ast.If) and 'carry_compact_turn' in ast.unparse(n.test)]
        for compact,expected in ((True,.3),(True,.6),(False,.1)):
            args=NS(carry_compact_turn=compact,carry_compact_angular_cap=expected if compact else .3,carry_sdk_minimum_turn=True,carry_away_1m=True,carry_transit_command_speed=.12,carry_turn_recovery_speed=.12,carry_inplace_centre_hold=False)
            ns=dict(args=args,environment='')
            exec(compile(ast.Module(body=statements,type_ignores=[]),'actual_bridge_launch_arguments','exec'),ns)
            argv=shlex.split(ns['bridge_command'])
            self.assertIn('max_angular_velocity:='+str(expected),argv)
            self.assertEqual(ns['carry_angular'],expected)

class NativeGaitPreparationTests(unittest.TestCase):
    def test_gait_readiness_uses_actual_foot_contact_loads(self):
        def contact(side,force):return dict(efc_address=0,body1="world",body2=side+"_ankle_roll_link",local_force_torque=[force])
        self.assertEqual(m.native_foot_loads({'contacts':[contact('left',250),contact('right',20)]})[1],'left')
        self.assertEqual(m.native_foot_loads({'contacts':[contact('left',20),contact('right',250)]})[1],'right')
        self.assertIsNone(m.native_foot_loads({'contacts':[contact('left',130),contact('right',130)]})[1])
    def test_generated_prestep_is_bounded_and_has_zero_forward_input(self):
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True,gait_prepare=True)
        compile(code,'native_prestep','exec')
        self.assertIn('GAIT_PREPARATION_NOT_OBSERVED',code)
        self.assertIn('GAIT_PREPARATION_POSE_BOUND',code)
        self.assertIn('len(supports)>=required and acquired',code)
        self.assertIn('required==2 or knee_motion>.20',code)
        self.assertIn('mission["phase"]=="PRESTEP":forward=0.',code)

class SourceReleaseProtocolTests(unittest.TestCase):
    def test_real_source_method_uses_only_documented_custom_source_actions(self):
        from types import SimpleNamespace as NS
        from unittest import mock
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True,compact_angular_cap=.60,release_source_on_stop=True)
        method=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.FunctionDef) and n.name=='source_enabled')
        ns=dict(MICRO={'carry':{'release_source_on_stop':True}},SetMcInputSource=NS(Request=lambda:NS(request=NS(header=NS()),action=NS(),input_source=NS())),McInputAction=NS(INPUTACTION_ENABLE=2001,INPUTACTION_DISABLE=2002),time=__import__('time'),raw=lambda x:x if isinstance(x,dict) else vars(x))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'actual_source_admin','exec'),ns)
        requests=[]
        response={'response':{'header':{'code':0},'state':0,'task_id':1}}
        future=NS(done=lambda:True,result=lambda:response)
        node=NS(get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:None)),src=NS(service_is_ready=lambda:True,call_async=lambda q:(requests.append(q) or future)),events=[])
        ns['source_enabled'](node,False);ns['source_enabled'](node,True)
        self.assertEqual([q.action.value for q in requests],[2002,2001])
        self.assertEqual([q.input_source.name for q in requests],['robotsim_official_cmd_vel']*2)
        self.assertEqual(node.events[0]['meaning'],'SERVICE_ACCEPTANCE_NOT_TASK_COMPLETION_OR_ZERO_RECEIPT')
        self.assertIn('n.source_enabled(False) if cfg.get("release_source_on_stop") else None',code)
        self.assertIn('n.observe(1.05,baseline=baseline);n.stand_once()',code)

class CombinedCompactCentreTests(unittest.TestCase):
    def test_bounded_feedback_is_world_fixed_and_preserves_default(self):
        origin=dict(x=0.,y=0.)
        self.assertEqual(m.inplace_centre_command(dict(x=-.1,y=0.,yaw=0.),origin,.20),(.20,0.))
        f,l=m.inplace_centre_command(dict(x=-.1,y=0.,yaw=math.pi/2),origin,.20)
        self.assertAlmostEqual(f,0.);self.assertAlmostEqual(l,-.20)
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True,compact_centre_feedback=True,compact_angular_cap=.60)
        compile(code,'combined_compact','exec')
        self.assertIn('inplace_centre_command(p,mission["turn_origin_pose"],.20,forward_only=True)',code)
        self.assertEqual(m.inplace_centre_command(dict(x=-.1,y=-.1,yaw=0.),origin,.20,forward_only=True),(.20,0.))
        self.assertIn('UNQUALIFIED_LATERAL_COMPACT_COMMAND',code)
        self.assertIn('COMPACT_TURN_RADIUS_300MM',code)

class TwoTableDeliveryTests(unittest.TestCase):
    def test_actual_delivery_profile_uses_same_controller_and_separate_stop_permit(self):
        import json
        cfg=json.loads(Path("simulation/mujoco/scenes/raicom2026/config/two_table_delivery.json").read_text())
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,native_action_smoothing=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,compact_angular_cap=.60,release_source_on_stop=True,compact_centre_feedback=True,delivery=cfg)
        compile(code,"real_two_table_route","exec")
        self.assertIn('DELIVERY_REARM_NOT_QUALIFIED',code)
        self.assertIn('DELIVERY_GOAL_CROSSED_WITHOUT_ARRIVAL',code)
        self.assertIn('DELIVERY_ALIGNMENT_RADIUS_300MM',code)
        self.assertIn('mission["phase"]=="DELIVERY"',code)
        self.assertIn('turn_and_braking_peak_radius_m',code)
        with self.assertRaises(ValueError):m.build_carry_route_script([0.,0.],{},[],carry_pid=19,delivery=cfg)

    def test_three_segments_and_clock_recovery_never_reset_task_deadline(self):
        def rec(i,last=False):
            t=1000000000+i*3000000000
            return dict(schema="robotsim.carry-transit050.v1",owner_pid=19,generation=1,route_pid=6,route_start_ticks=1,created_ns=t,command_deadline_ns=90000000000,absolute_deadline_ns=105000000000,braking_ns=None if last else t+1000000000,revoked_ns=None if last else t+2000000000,task_interval_budget=7)
        xs=[rec(i,i==6) for i in range(7)]
        self.assertEqual(m.carry_policy_speed_limit(xs,20000000000,1,19),.50)
        with self.assertRaises(ValueError):m.carry_policy_speed_limit(xs+[rec(7,True)],23000000000,1,19)
        xs[-1]["command_deadline_ns"]+=1
        with self.assertRaisesRegex(ValueError,"DEADLINE"):m.carry_policy_speed_limit(xs,20000000000,1,19)

class VendorActionTableTests(unittest.TestCase):
    def test_original_transition_profile_does_not_masquerade_as_added_smoothing(self):
        import json
        cfg=json.loads(Path("simulation/mujoco/scenes/raicom2026/config/two_table_delivery.json").read_text())
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,release_source_on_stop=True,compact_centre_feedback=True,delivery=cfg,vendor_transition_table=True)
        compile(code,"vendor_action_table_carry","exec")
        self.assertIn("'native_action_smoothing': False",code)
        self.assertIn("'vendor_transition_table': True",code)
        with self.assertRaises(ValueError):m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,release_source_on_stop=True,compact_centre_feedback=True,delivery=cfg,vendor_transition_table=True,native_action_smoothing=True)

class DeliveryTurnStopGoTests(unittest.TestCase):
    def test_real_generated_handoff_never_rearms_without_complete_stop(self):
        import json,threading,time
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        config=json.loads(Path("simulation/mujoco/scenes/raicom2026/config/two_table_delivery.json").read_text())
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,release_source_on_stop=True,compact_centre_feedback=True,delivery=config,vendor_transition_table=True)
        node=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.If) and 'output.state == NavState.DRIVE and mission.get' in ast.unparse(n.test))
        wrapper=ast.parse("def actual_handoff():\n for _ in [0]:\n  pass\n")
        wrapper.body[0].body[0].body=[node];wrapper.body[0].body.insert(0,ast.Global(names=["transit_permit"]));ast.fix_missing_locations(wrapper)
        for passed in (False,True):
            events=[]
            n=NS(deadline_zero=lambda:events.append("zero"),source_enabled=lambda value:events.append("enable" if value else "disable"),
                 observe=Mock(side_effect=[None,dict(pass_result=passed,**{"pass":passed})]),stand_once=lambda:events.append("stand"),
                 await_graph=lambda:None,pump=lambda:None,sample=lambda:{"pose":{"x":0.,"y":0.,"yaw":0.}},
                 latest_row={},lock=threading.Lock())
            cfg=dict(release_source_on_stop=True,permit_filename="delivery.json")
            ns=dict(stand_transition=lambda:(n.observe(1.05,baseline={}),n.stand_once()),n=n,mission={"delivery_alignment_state":"ALIGN"},output=NS(state="DRIVE"),NavState=NS(DRIVE="DRIVE"),cfg=cfg,
                    baseline={},result={},transit_permit={},closed_permits=[],carry_transit_phase=lambda old,phase,c,row:{"phase":phase},
                    time=time,until=1e99,controller=NS(reset=lambda:events.append("controller_reset")))
            exec(compile(wrapper,"actual_delivery_handoff","exec"),ns)
            if passed:
                ns["actual_handoff"]();self.assertEqual(events,["zero","disable","stand","controller_reset","enable"])
                self.assertIsNone(ns["mission"]["delivery_alignment_state"])
            else:
                with self.assertRaisesRegex(RuntimeError,"DELIVERY_ALIGNMENT_STOP_NOT_QUALIFIED"):ns["actual_handoff"]()
                self.assertNotIn("enable",events)
            self.assertEqual(n.observe.call_args_list[0].args,(1.05,))
            self.assertTrue(n.observe.call_args_list[1].kwargs["require_stop"])

class TerminalPivotControlTests(unittest.TestCase):
    def test_actual_terminal_turn_removes_translation_without_changing_bounds(self):
        import json
        from types import SimpleNamespace as NS
        cfg=json.loads(Path("simulation/mujoco/scenes/raicom2026/config/two_table_delivery.json").read_text())
        cfg["turn_terminal_translation_free_angle_rad"]=1.
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,compact_centre_feedback=True,gait_prepare=True,release_source_on_stop=True,delivery=cfg,vendor_transition_table=True)
        node=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.If) and ast.unparse(n.test)=='cfg.get(\'compact_centre_feedback\')' and any('terminal_turn_translation_free_ticks' in ast.unparse(x) for x in n.body))
        for error,expect_zero in ((.5,True),(1.5,False)):
            out=NS(linear_x=9.,lateral_y=9.)
            ns=dict(cfg={"compact_centre_feedback":True,"delivery":cfg},turn_error=error,output=out,p=dict(x=-.10,y=0.,yaw=0.),
                    mission={"turn_origin_pose":dict(x=0.,y=0.)},inplace_centre_command=m.inplace_centre_command)
            exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_terminal_turn_control","exec"),ns)
            self.assertEqual(out.linear_x==0.,expect_zero)
            self.assertEqual(out.lateral_y,0.)
            self.assertLessEqual(abs(out.linear_x),.20)
        self.assertIn('COMPACT_TURN_RADIUS_300MM',code)
        self.assertEqual(cfg["turn_braking_footprint_m"],.35)

class ContinuousZeroStandStrategyTests(unittest.TestCase):
    def test_actual_stop_strategy_preserves_zero_duration_and_default_order(self):
        import json
        from types import SimpleNamespace as NS
        cfg=json.loads(Path("simulation/mujoco/scenes/raicom2026/config/two_table_delivery.json").read_text())
        code=m.build_carry_route_script(None,{},[],carry_pid=19,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,release_source_on_stop=True,compact_centre_feedback=True,delivery=cfg,vendor_transition_table=True)
        helper=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.FunctionDef) and n.name=="stand_transition")
        for strategy,expected in (("ZERO_WINDOW_THEN_SD",["zero_window","sd"]),("EARLY_SD_CONTINUOUS_ZERO_DIAGNOSTIC",["sd","zero_window"])):
            calls=[]
            n=NS(stand_once=lambda:calls.append("sd"),observe=lambda seconds,**kw:(self.assertGreaterEqual(seconds,1.05),calls.append("zero_window")))
            ns=dict(cfg={"delivery":{"stand_transition_strategy":strategy}},n=n,baseline={},result={})
            exec(compile(ast.Module(body=[helper],type_ignores=[]),"actual_stop_strategy","exec"),ns)
            ns["stand_transition"]();self.assertEqual(calls,expected)
            self.assertNotIn("pass_result",ns["result"])
        self.assertIn('require_stop=True',code)
        self.assertIn('DELIVERY_ALIGNMENT_STOP_NOT_QUALIFIED',code)

class TerminalAisleClearanceTests(unittest.TestCase):
    def test_extension_is_terminal_heading_and_one_sided_not_a_larger_circle(self):
        origin=dict(x=0.,y=0.)
        extension=dict(frame="native_scene_world",maximum_radius_m=.5,lateral_m=.2,terminal_yaw_error_rad=.4)
        pose=lambda x,y,yaw=math.pi:dict(x=x,y=y,yaw=yaw)
        self.assertTrue(m.table_terminal_braking_footprint([pose(.40,.10)],origin,math.pi,.35,extension)["pass_clearance"])
        self.assertFalse(m.table_terminal_braking_footprint([pose(.40,.10)],origin,math.pi,.35)["pass_clearance"])
        for unsafe in (pose(-.40,.10),pose(.40,-.25),pose(.51,0.),pose(.40,.10,math.pi/2)):
            self.assertFalse(m.table_terminal_braking_footprint([unsafe],origin,math.pi,.35,extension)["pass_clearance"])
        actual=m.table_terminal_braking_footprint([pose(.4,.1)],origin,math.pi,.35,extension)
        self.assertFalse(actual["original_radial_criterion_pass"])
        self.assertTrue(actual["used_aisle_reserve"])

class DeliveryAlignmentPermitTests(unittest.TestCase):
    def test_actual_writer_requires_stop_revocation_and_shared_deadline(self):
        import tempfile,time,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            deadline=time.monotonic_ns()+90000000000
            cfg=dict(pid=19,timeout_s=90.,compact_turn=True,delivery={"scope":"test"},absolute_command_deadline_ns=deadline)
            row=dict(generation=1)
            first=m.carry_transit_phase(None,"TRANSIT",cfg,row,evidence=folder)
            first=m.carry_transit_phase(first,"BRAKING",cfg,row,evidence=folder)
            first=m.carry_transit_phase(first,"REVOKED",cfg,row,evidence=folder)
            cfg.update(previous_permit_filename="carry_transit_permit.json",permit_filename="carry_transit_delivery_rearm_1.json")
            with self.assertRaisesRegex(RuntimeError,"DELIVERY_ALIGNMENT_REARM_NOT_QUALIFIED"):
                m.carry_transit_phase(None,"TRANSIT",cfg,row,evidence=folder)
            cfg["delivery_alignment_stop_pass"]=True
            record=m.carry_transit_phase(None,"TRANSIT",cfg,row,evidence=folder)
            self.assertEqual(record["command_deadline_ns"],first["command_deadline_ns"])
            self.assertEqual(record["task_interval_budget"],7)
            self.assertEqual(m.carry_policy_speed_limit([first,record],time.monotonic_ns(),1,19),.50)
            with self.assertRaisesRegex(RuntimeError,"CARRY_TRANSIT_REARM_FORBIDDEN"):
                m.carry_transit_phase(None,"TRANSIT",cfg,row,evidence=folder)
            cfg["permit_filename"]="carry_transit_delivery_rearm_99.json"
            with self.assertRaisesRegex(ValueError,"CARRY_PERMIT_PATH"):
                m.carry_transit_phase(None,"TRANSIT",cfg,row,evidence=folder)

class DeliveryCumulativeProgressTests(unittest.TestCase):
    def test_alignment_interleaving_does_not_reset_forward_failure(self):
        state=None
        for i in range(80):
            state=m.delivery_forward_progress(state,2.+i*.001,i*.1,i%4!=3,"same")
        self.assertTrue(state["no_progress"])
        self.assertGreater(state["active_without_progress_s"],3.)
        state=m.delivery_forward_progress(state,1.8,8.1,True,"same")
        self.assertFalse(state["no_progress"])
    def test_rearm_gap_does_not_count_as_command_time_or_clear_prior_failure_budget(self):
        state=m.delivery_forward_progress(None,2.,0.,True,"one")
        state=m.delivery_forward_progress(state,2.1,.1,True,"one")
        before=state["active_without_progress_s"]
        after=m.delivery_forward_progress(state,2.2,8.,True,"two")
        self.assertEqual(after["active_without_progress_s"],before)
        self.assertEqual(after["best_distance_m"],2.)
