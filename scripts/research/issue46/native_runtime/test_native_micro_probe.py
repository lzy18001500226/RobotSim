"""Offline micro-step feedback, deadline and unchanged stop-contract tests."""
import ast
import copy
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
from scripts.research.issue46.native_runtime import native_micro_stop_contract as stop


def rows():
    return [dict(sample_monotonic_s=10+i*.02,
        pose=dict(x=0.,y=0.,z=.65,roll=0.,pitch=0.,yaw=0.,vx=0.,vy=0.,vz=0.,
                  frame_id="native_scene_world",child_frame_id="pelvis_origin"),
        leg_states=[dict(name=n,position=0.,velocity=0.,error_code=0) for n in ("left","right")],
        mc_leg_command=dict(joints=[dict(name=n,position=0.) for n in ("left","right")]),
        state_age_s={k:.01 for k in ("odom","imu","common","leg_state","leg_command","get_mode")},
        get_mode_desc="STAND_DEFAULT",common_mode_desc="STAND_DEFAULT",
        get_mode_status_value=100,common_mode_status_value=100,
        get_mode_header_code=0,fsm_state=4,body_status=1) for i in range(101)]


class MicroTests(unittest.TestCase):
    def test_duplicates_never_refresh_last_unique_age(self):
        store={}
        self.assertTrue(micro.advancing_stamp(store,"odom",100,1.))
        for t in (1.1,1.19,1.4):
            self.assertFalse(micro.advancing_stamp(store,"odom",100,t))
        self.assertEqual(store["odom"],(100,1.))
        self.assertFalse(micro.advancing_stamp(store,"odom",99,1.5))
        self.assertTrue(micro.advancing_stamp(store,"odom",101,1.6))

    def test_ready_callbacks_drain_more_than_one_without_unbounded_loop(self):
        from unittest.mock import patch
        clock=NS(t=0.);calls=[]
        def spin(node,timeout_sec):
            calls.append(timeout_sec);clock.t+=.0001
        with patch.object(micro.time,"monotonic",lambda:clock.t):
            count=micro.drain_ready_callbacks(None,NS(spin_once=spin),budget_s=.004)
        self.assertGreater(count,30);self.assertLessEqual(clock.t,.0041)
        self.assertTrue(all(t==0. for t in calls))
        calls.clear()
        with patch.object(micro.time,"monotonic",lambda:0.):
            count=micro.drain_ready_callbacks(None,NS(spin_once=lambda *a,**k:calls.append(0)),max_callbacks=5)
        self.assertEqual(count,5)

    def test_native_velocity_is_explicit_body_origin_world(self):
        r=dict(bodies=dict(pelvis=dict(position_m=[0,0,.65],rotation=[1,0,0,0,1,0,0,0,1],
                    origin_velocity_world_angular_linear=[0,0,.1,.02,.03,0.])))
        p=micro.native_pose(r);self.assertEqual((p["vx"],p["vy"]),(.02,.03))
        del r["bodies"]["pelvis"]["origin_velocity_world_angular_linear"]
        with self.assertRaises(KeyError):micro.native_pose(r)

    def test_heading_feedback_opposes_measured_drift_and_preserves_caps(self):
        self.assertEqual(micro.heading_hold_command(-.12,0.),.075)
        self.assertEqual(micro.heading_hold_command(.12,0.),-.075)
        self.assertEqual(micro.heading_hold_command(.01,0.),0.)
        self.assertAlmostEqual(micro.heading_hold_command(math.pi-.03,-math.pi+.03),.048)
        for x in (float("nan"),float("inf")):
            with self.assertRaises(ValueError):micro.heading_hold_command(x,0.)
        code=micro.build_script("left",1.,{},command_speed=.12,heading_hold=True)
        compile(code,"heading-probe","exec")
        self.assertIn("'heading_hold': True",code)
        self.assertIn('n.publish(*command_velocity,angular)',code)

    def test_micro_directions_preserve_cross_track_guard(self):
        pose=dict(x=0.,y=.051,z=.65,roll=0.,pitch=0.,yaw=0.,vx=0.,vy=0.)
        root=dict(x=0.,y=0.,yaw=0.)
        self.assertEqual(micro.physical_fault(dict(joints={}),pose,{},root,"forward"),"MICRO_CROSS_TRACK")
        self.assertIsNone(micro.physical_fault(dict(joints={}),pose,{},root,"left"))
        pose["vx"]=.201
        self.assertEqual(micro.physical_fault(dict(joints={}),pose,{},root,"left"),"PLANAR_SPEED")

    def test_targeted_cycle_is_explicit_and_preserves_default_duration_limit(self):
        self.assertEqual(micro.micro_displacement(dict(x=0.,y=.03),dict(x=0.,y=0.,yaw=math.pi/2),"forward")[2],.03)
        code=micro.build_script("forward",2.,{},command_speed=.12,heading_hold=True,target_distance_m=.03)
        compile(code,"targeted","exec")
        self.assertIn("'target_distance_m': 0.03",code)
        self.assertIn('result["goal_triggered_stop"]=True',code)
        code14=micro.build_script("forward",2.,{},command_speed=.14,heading_hold=True,target_distance_m=.03)
        self.assertNotIn("command_velocity=(.12,0.)",code14)
        with self.assertRaises(ValueError):micro.build_script("left",2.,{},command_speed=.14,target_distance_m=.03)
        for kwargs in (dict(duration_s=2.01,target_distance_m=.03),dict(duration_s=2.,target_distance_m=.03,command_speed=.2),dict(duration_s=2.)):
            with self.assertRaises(ValueError):micro.build_script("forward",joint_limits={},**kwargs)

    def test_source_joint_bounds_are_not_raised(self):
        p=dict(z=.65,roll=0.,pitch=0.,vx=0.,vy=0.)
        r=dict(joints=dict(knee=dict(q=.1,qvel=1.01)))
        self.assertEqual(micro.physical_fault(r,p,dict(knee=(-1,1,1))),"SOURCE_JOINT_SPEED:knee")

    def test_inter_sample_joint_velocity_peak_is_not_hidden_by_current_low_speed(self):
        p=dict(z=.65,roll=0.,pitch=0.,vx=0.,vy=0.)
        r=dict(joints=dict(knee=dict(q=.1,qvel=.01,max_abs_qvel_since_previous_sample=1.01)))
        self.assertEqual(micro.physical_fault(r,p,dict(knee=(-1,1,1))),"SOURCE_JOINT_SPEED:knee")

    def test_unchanged_composite_contract(self):
        self.assertTrue(stop.evaluate_stop_window(rows(),baseline_body_status=1)["pass"])
        r=rows();r[-1]["mc_leg_command"]["joints"][1]["position"]=.02001
        out=stop.evaluate_stop_window(r,baseline_body_status=1)
        self.assertFalse(out["pass"]);self.assertIn("joint_target_not_settled",out["violations"])
        self.assertEqual(stop.PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD,.020)

    def test_transition_does_not_count_as_stand(self):
        r=rows()
        r[30]["common_mode_status_value"]=300
        self.assertIn("mode_status_not_idle_throughout_window",
                      stop.evaluate_stop_window(r,baseline_body_status=1)["violations"])

    def test_missing_stale_or_short_window_fails(self):
        for mutate in ("stale","missing","short"):
            r=rows()
            if mutate=="stale":r[-1]["state_age_s"]["get_mode"]=.51
            if mutate=="missing":r[-1]["state_age_s"].pop("leg_command")
            if mutate=="short":r=r[:70]
            self.assertFalse(stop.evaluate_stop_window(r,baseline_body_status=1)["pass"])

    def test_generated_code_compiles_and_never_ticks_navigation(self):
        code=micro.build_script("left",.6,{})
        compile(code,"micro","exec")
        self.assertNotIn("n.stand(); n.spin(1.5)",code)
        self.assertEqual(code.count("\nrun_probe()"),1)
        for bad in (0.,.39,1.01,float("nan")):
            with self.assertRaises(ValueError):micro.build_script("left",bad,{})
        with self.assertRaises(ValueError):micro.build_script("up",.6,{})

    def test_native_reset_requires_a_new_generation_and_same_actual_process(self):
        from scripts.research.issue46.native_runtime.native_grasp_audit import native_reset_generation_effect
        before=dict(engine_version=337,process_pid=30,host_monotonic_ns=10,generation=0,sim_time_s=10.)
        after=dict(before,host_monotonic_ns=20,generation=1,sim_time_s=.02)
        self.assertTrue(native_reset_generation_effect(before,after))
        for change in (dict(generation=0),dict(process_pid=31),dict(sim_time_s=11.),dict(host_monotonic_ns=10)):
            self.assertFalse(native_reset_generation_effect(before,dict(after,**change)))

    def test_real_parent_argument_path_allows_only_isolated_micro_exception(self):
        from unittest.mock import patch
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
        import sys
        args=["parent","--table-overlay-result","/nonexistent","--table-overlay-result-sha256","0"*64,
              "--reset-helper","/nonexistent","--reset-helper-sha256","0"*64,
              "--hal-arm-candidate","/nonexistent","--hal-arm-candidate-sha256","0"*64,
              "--disable-mc-arm-publisher","--disable-mc-hand-publisher",
              "--hal-right-hand-command","0","--hal-arm-plan-mode","pregrasp-hold",
              "--initial-hands-fully-open","--native-dock-truth-feedback","--use-official-cmd-vel-bridge"]
        with patch.object(sys,"argv",args+["--micro-base-probe","left"]),patch.object(parent,"require_hashed",side_effect=RuntimeError("REACHED_INPUT_HASH_GATE")):
            with self.assertRaisesRegex(RuntimeError,"REACHED_INPUT_HASH_GATE"):parent.main()
        with patch.object(sys,"argv",args):
            with self.assertRaisesRegex(ValueError,"requires a measured native table-docking"):parent.main()
        with patch.object(sys,"argv",args+["--micro-base-probe","forward","--native-arm-policy-relay"]),patch.object(parent,"require_hashed",side_effect=RuntimeError("REACHED_INPUT_HASH_GATE")):
            with self.assertRaisesRegex(RuntimeError,"REACHED_INPUT_HASH_GATE"):parent.main()
        heading=["--micro-base-probe","left","--micro-command-speed",".12",
                 "--native-reset-input","button","--micro-heading-hold",
                 "--dock-angular-speed-cap",".075"]
        with patch.object(sys,"argv",args+heading),patch.object(parent,"require_hashed",side_effect=RuntimeError("REACHED_INPUT_HASH_GATE")):
            with self.assertRaisesRegex(RuntimeError,"REACHED_INPUT_HASH_GATE"):parent.main()
        targeted=heading+["--reviewed-foundationwalk-profile","--micro-target-distance-m",".03","--micro-probe-duration-s","2.0"]
        with patch.object(sys,"argv",args+targeted),patch.object(parent,"require_hashed",side_effect=RuntimeError("REACHED_INPUT_HASH_GATE")):
            with self.assertRaisesRegex(RuntimeError,"REACHED_INPUT_HASH_GATE"):parent.main()

    def test_low_input_profile_changes_only_bridge_command_amplitude(self):
        code=micro.build_script("left",.6,{},command_speed=.12)
        self.assertIn("'command_speed': 0.12",code)
        self.assertIn('bounds=n.messages["common"]["speed_status"]',code)
        for value in (.119,.201,float("nan")):
            with self.assertRaises(ValueError):micro.build_script("left",.6,{},command_speed=value)
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
        import yaml
        data=yaml.safe_load((parent.RUNTIME/"mc/bin/mc_param/src/robot/lx2501_3_t2d5/robot_model.yaml").read_text())
        limits={item["name"]:item["limits"] for group in data["joints"] for item in group[1]}
        self.assertEqual(limits["right_knee_joint"]["max_velocity"],11.936)

    def test_native_arm_policy_checks_all_29_existing_source_coordinates(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract
        limits,_=native_leg_safety_contract(include_upper_body=True)
        self.assertEqual(len(limits),29)
        self.assertEqual(limits["right_wrist_pitch_joint"][2],4.188)
        self.assertLessEqual(limits["right_knee_joint"][2],11.936)

    def runtime_methods(self):
        tree=ast.parse(Path(micro.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
        cls=next(n for n in run.body if isinstance(n,ast.ClassDef))
        cls.body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in ("publish","deadline_zero","observe","await_graph")]
        sent=[]
        class Base:
            def publish(self,*v):sent.append(v)
        clock=NS(t=1.)
        ns=dict(Dock=Base,time=NS(monotonic=lambda:clock.t,sleep=lambda d:setattr(clock,"t",clock.t+d)),
                threading=threading,rclpy=NS(ok=lambda:True),contract=stop,json=json)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),"micro","exec"),ns)
        n=ns["Micro"]();n.lock=threading.Lock();n.moving=True;n.events=[];n.command_log=[];n.trace=[]
        return n,sent,clock

    def test_deadline_zero_latches_before_stalled_sampling(self):
        n,sent,clock=self.runtime_methods()
        n.publish(.2,0.,0.);n.deadline_zero()
        n.publish(.2,0.,0.)  # late loop callback cannot re-enable motion
        self.assertEqual(sent,[(.2,0.,0.),(0.,0.,0.),(0.,0.,0.)])
        self.assertFalse(n.moving)

    def test_discovery_wait_requires_positive_graph_and_real_bridge_output(self):
        n,sent,clock=self.runtime_methods();n.moving=False;n.bridge_commands=[]
        n.pump=lambda:None
        def graph():
            if clock.t<1.05:raise RuntimeError("GRAPH_DISCOVERY_PENDING")
            n.bridge_commands.append(dict(t=clock.t))
        n.graph=graph;n.await_graph()
        self.assertGreaterEqual(clock.t,1.05)
        self.assertTrue(all(v==(0.,0.,0.) for v in sent))
        n.bridge_commands=[];n.graph=lambda:None
        with self.assertRaisesRegex(RuntimeError,"POSITIVE_BRIDGE_GRAPH"):n.await_graph()
        def conflicting():raise RuntimeError("CONFLICTING_VELOCITY_GRAPH")
        n.graph=conflicting;before=clock.t
        with self.assertRaisesRegex(RuntimeError,"CONFLICTING"):n.await_graph()
        self.assertEqual(clock.t,before)

    def test_zero_observer_keeps_publishing_while_readback_not_ready(self):
        n,sent,clock=self.runtime_methods();n.moving=False
        fixture=rows()
        def sample(start):
            r=copy.deepcopy(fixture[0]);r["sample_monotonic_s"]=clock.t
            r["state_age_s"]["get_mode"]=.8  # no valid service readback
            return r
        n.sample=sample;n.pump=lambda:None
        with self.assertRaisesRegex(RuntimeError,"COMPOSITE_STOP_NOT_QUALIFIED"):
            n.observe(2.15,baseline=1,require_stop=True)
        self.assertGreater(len(sent),100)
        self.assertTrue(all(v==(0.,0.,0.) for v in sent))

    def test_observer_uses_unique_native_samples(self):
        n,sent,clock=self.runtime_methods();n.moving=False
        n.pump=lambda:None;n.sample=lambda start:copy.deepcopy(rows()[0])
        with self.assertRaisesRegex(RuntimeError,"COMPOSITE_STOP_NOT_QUALIFIED"):
            n.observe(2.15,baseline=1,require_stop=True)
        self.assertEqual(len(n.trace),1)


class ContinuousFeedbackTests(unittest.TestCase):
    def methods(self, names, **values):
        tree=ast.parse(Path(micro.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
        cls=next(n for n in run.body if isinstance(n,ast.ClassDef))
        cls.bases=[];cls.body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in names]
        ns=dict(threading=threading,time=__import__("time"),raw=lambda x:x,
                advancing_stamp=micro.advancing_stamp,observed_clock_step=micro.observed_clock_step,json=json,**values)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),"micro-executor","exec"),ns)
        node=ns["Micro"]()
        node.telemetry_lock=threading.RLock();node.clock_offset_ns=None
        node.source_epoch=0;node.pumped_epoch=0;node.epoch_changed_at=None
        node.unique={};node.messages={};node.receipt_diagnostics={};node.events=[]
        node.moving=False
        return node

    def test_feedback_is_processed_during_foreground_wait_and_stops_cleanly(self):
        from unittest.mock import patch
        import sys,types,time
        stopped=threading.Event();calls=[]
        class Executor:
            def add_node(self,n):calls.append("add")
            def spin(self):
                while not stopped.wait(.001):calls.append("callback")
            def shutdown(self,timeout_sec):calls.append("shutdown");stopped.set()
        module=types.ModuleType("rclpy.executors");module.SingleThreadedExecutor=Executor
        n=self.methods(("start_feedback","stop_feedback"))
        n.feedback_executor=None;n.feedback_thread=None;n.feedback_executor_error=None
        with patch.dict(sys.modules,{"rclpy.executors":module}):
            n.start_feedback();time.sleep(.04);n.stop_feedback()
        self.assertGreater(calls.count("callback"),10)
        self.assertEqual(calls.count("add"),1);self.assertEqual(calls.count("shutdown"),1)
        self.assertFalse(n.feedback_thread.is_alive())

    def test_callback_exception_is_reported_and_never_hidden(self):
        from unittest.mock import patch
        import sys,types,time
        class Executor:
            def add_node(self,n):pass
            def spin(self):raise RuntimeError("callback failed")
            def shutdown(self,timeout_sec):pass
        module=types.ModuleType("rclpy.executors");module.SingleThreadedExecutor=Executor
        n=self.methods(("start_feedback","stop_feedback"))
        n.feedback_executor=None;n.feedback_thread=None;n.feedback_executor_error=None
        with patch.dict(sys.modules,{"rclpy.executors":module}):
            n.start_feedback();n.feedback_thread.join(1.);n.stop_feedback()
        self.assertIn("callback failed",n.feedback_executor_error)

    def test_duplicate_and_regression_callbacks_are_counted_but_do_not_refresh(self):
        n=self.methods(("store",))
        n.unique={};n.messages={};n.receipt_diagnostics={}
        def msg(stamp):return NS(header=NS(stamp=NS(sec=1,nanosec=stamp)))
        from unittest.mock import patch
        with patch("time.time_ns",return_value=1000000000):
            n.store("common",msg(10));before=n.unique["common"]
            n.store("common",msg(10));n.store("common",msg(9))
        self.assertEqual(n.unique["common"],before)
        d=n.receipt_diagnostics["common"]
        self.assertEqual((d["callbacks"],d["duplicates"],d["regressions"]),(3,1,1))
        with patch("time.time_ns",return_value=1000000000):n.store("common",msg(11))
        self.assertGreater(n.unique["common"][0],before[0])

    def test_stale_native_feedback_still_fails_with_continuous_executor(self):
        n=self.methods(("sample",))
        n.latest_row=dict(host_monotonic_ns=1)
        n.feedback_fault=None
        with self.assertRaisesRegex(RuntimeError,"NATIVE_FEEDBACK_STALE"):
            n.sample()


class ClockGenerationTests(unittest.TestCase):
    methods=ContinuousFeedbackTests.methods
    def test_clock_step_requires_an_independent_well_bracketed_clock_read(self):
        offset,step=micro.observed_clock_step(None,100,1100,100)
        self.assertEqual((offset,step),(1000,None))
        self.assertEqual(micro.observed_clock_step(offset,200,1200,200),(1000,None))
        self.assertEqual(micro.observed_clock_step(offset,300,1300,2000000),(1000,None))
        off,step=micro.observed_clock_step(offset,400,400-1000000000,400)
        self.assertLess(step,-.9e9)

    def clock_node(self,moving=False):
        n=self.methods(("store",))
        n.feedback_fault=None;n.moving=moving
        n.deadline_zero=lambda:(setattr(n,"moving",False),n.events.append(dict(event="ZERO")))
        return n

    def message(self,ns):
        return NS(header=NS(stamp=NS(sec=ns//1000000000,nanosec=ns%1000000000)))

    def test_stopped_clock_generation_drops_old_queued_messages(self):
        from unittest.mock import patch
        n=self.clock_node()
        with patch("time.monotonic_ns",return_value=10000000000),patch("time.time_ns",return_value=2000000000):
            n.store("common",self.message(2000000000))
        n.messages["imu"]="old"
        with patch("time.monotonic_ns",return_value=10020000000),patch("time.time_ns",return_value=1020000000):
            n.store("common",self.message(2000000000))
            self.assertEqual(n.messages,{})
            n.store("common",self.message(1020000000))
        self.assertEqual(n.source_epoch,1);self.assertNotIn("imu",n.messages)
        self.assertEqual(n.unique["common"][0],1020000000)
        self.assertEqual(n.receipt_diagnostics["common"]["rejected_wrong_epoch"],1)
        self.assertIsNone(n.feedback_fault)

    def test_clock_step_during_motion_latches_zero_and_a_fault(self):
        from unittest.mock import patch
        n=self.clock_node(moving=True)
        with patch("time.monotonic_ns",return_value=10000000000),patch("time.time_ns",return_value=2000000000):
            n.store("common",self.message(2000000000))
        with patch("time.monotonic_ns",return_value=10020000000),patch("time.time_ns",return_value=1020000000):
            n.store("common",self.message(1020000000))
        self.assertFalse(n.moving);self.assertEqual(n.feedback_fault["gate"],"CLOCK_STEP_DURING_MOTION")
        self.assertEqual(sum(e["event"]=="ZERO" for e in n.events),1)

    def test_stop_observer_never_combines_different_clock_generations(self):
        n,sent,clock=MicroTests().runtime_methods();n.moving=False;n.pump=lambda:None
        def sample(start):
            r=copy.deepcopy(rows()[0]);r["sample_monotonic_s"]=clock.t
            r["source_clock_generation"]=0 if clock.t<2.1 else 1
            return r
        n.sample=sample
        with self.assertRaisesRegex(RuntimeError,"COMPOSITE_STOP_NOT_QUALIFIED"):
            n.observe(3.,baseline=1,require_stop=True)
        self.assertTrue(all(v==(0.,0.,0.) for v in sent))


class SteadyBridgeTests(unittest.TestCase):
    def source(self):
        path=Path("/home/lzy18001500226/projects/RobotSim/reference/AgiBot-Community/raicom2026-agibot-X2/src/race_task/race_task/cmd_vel_bridge.py")
        return micro.build_steady_bridge_source(path.read_text())

    def test_only_timer_and_input_age_clock_change(self):
        source=self.source()
        compile(source,"steady-bridge","exec")
        self.assertIn("clock=_robotsim_Clock(clock_type=_robotsim_ClockType.STEADY_TIME)",source)
        self.assertIn("msg.header.stamp = now.to_msg()",source)
        self.assertIn("age = _robotsim_steady_time.monotonic() - self.last_cmd_time",source)
        with self.assertRaises(ValueError):micro.build_steady_bridge_source("# altered vendor source")

    def test_backward_wall_time_cannot_extend_nonzero_input_timeout(self):
        source=ast.parse(self.source())
        cls=next(x for x in source.body if isinstance(x,ast.ClassDef))
        cls.bases=[];cls.body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=="publish_velocity"]
        clock=NS(t=10.05);sent=[]
        ns=dict(_robotsim_steady_time=NS(monotonic=lambda:clock.t),
                MessageHeader=lambda:NS(),McLocomotionVelocity=lambda:NS())
        exec(compile(ast.Module(body=[cls],type_ignores=[]),"steady","exec"),ns)
        n=ns[cls.name]();n.last_cmd_time=10.;n.cmd_timeout_sec=.20
        n.get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:"REAL_BACKWARD_WALL_STAMP"))
        n.forward_velocity=.12;n.lateral_velocity=0.;n.angular_velocity=0.
        n.publisher=NS(publish=sent.append);n.source="robotsim_official_cmd_vel"
        n.publish_velocity();clock.t=10.201;n.publish_velocity()
        self.assertEqual(sent[0].forward_velocity,.12)
        self.assertEqual(sent[1].forward_velocity,0.)
        self.assertEqual(sent[1].header.stamp,"REAL_BACKWARD_WALL_STAMP")


class ReviewedFoundationWalkProfileTests(unittest.TestCase):
    def test_exact_prior_overlay_and_runtime_identity_are_reused(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import reviewed_foundationwalk_profile
        runtime=Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026")
        path,record=reviewed_foundationwalk_profile(runtime)
        self.assertEqual(record["overlay_sha256"],"6eb712872248c1c377b27c1bb58fdd5b44bdd4262e0271e779d0040ee82964e2")
        self.assertEqual(len(record["changed_values"]),3)
        self.assertFalse(record["fresh_parameter_tuning"])
        self.assertTrue(record["mc_binary_onnx_gains_and_joint_limits_unchanged"])
        self.assertTrue(path.is_file())

    def test_wrong_runtime_never_selects_a_different_profile(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import reviewed_foundationwalk_profile
        with self.assertRaises(FileNotFoundError):
            reviewed_foundationwalk_profile(Path("/nonexistent/runtime"))


class McDebugReuseTests(unittest.TestCase):
    def test_collector_reuse_has_no_motion_publishers_or_action_clients(self):
        import tempfile
        from scripts.research.issue46.native_runtime.native_base_micro_probe import prepare_reviewed_rl_debug_capture
        runtime=Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026")
        from unittest.mock import patch
        from scripts.agent import path_governance as pg
        with tempfile.TemporaryDirectory() as folder,patch.object(pg,"write_new_bytes",side_effect=lambda p,b:Path(p).write_bytes(b)),patch.object(pg,"write_new_text",side_effect=lambda p,t:Path(p).write_text(t)):
            p=Path(folder);record=prepare_reviewed_rl_debug_capture(p,runtime)
            self.assertEqual(record["mc_introspection_sha256"],"935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931")
            capture=ast.parse((p/"issue80_mc_rl_debug_capture.py").read_text())
            calls=[n.func.attr for n in ast.walk(capture) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertNotIn("create_publisher",calls);self.assertNotIn("create_client",calls)
            helper=ast.parse((p/"issue80_m1_first_motion.py").read_text())
            self.assertEqual([x.name for x in helper.body if isinstance(x,ast.FunctionDef)],["mc_rl_debug_snapshot"])
            compile(capture,"collector","exec");compile(helper,"snapshot","exec")

    def sample_node(self):
        n=ContinuousFeedbackTests().methods(("sample",),math=math,native_pose=micro.native_pose,
                physical_fault=micro.physical_fault,MICRO=dict(joint_limits={},direction="forward"),
                contract=stop,LEG=("left","right"))
        n.latest_row=dict(host_monotonic_ns=10000000000,sim_time_s=1.,generation=1,
                bodies=dict(pelvis=dict(position_m=[0.,0.,.65],rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.],
                             origin_velocity_world_angular_linear=[0.]*6)),joints={})
        n.feedback_fault=None
        n.unique={name:(1,9.99) for name in ("common","imu","leg_state","leg_command")}
        n.messages=dict(imu=dict(orientation=dict(w=1.,x=0.,y=0.,z=0.)),
                common=dict(action_info=dict(action_desc="STAND_DEFAULT",status=dict(value=100)),
                            fsm_state=dict(current_state=4),body_status=dict(value=1),input_source=dict(name="")),
                leg_state=dict(joints=[dict(name=x,error_code=0) for x in ("left","right")]),
                leg_command=dict(header=dict(stamp=dict(sec=1,nanosec=0)),joints=[dict(name=x,position=0.) for x in ("left","right")]),
                ros_odom=dict(header=dict(frame_id="odom")))
        n.readbacks=dict(get_mode=(9.99,dict(info=dict(action_desc="STAND_DEFAULT",status=dict(value=100)),header=dict(code=0))))
        return n

    def test_raw_imu_odom_and_command_source_header_are_preserved(self):
        from unittest.mock import patch
        n=self.sample_node()
        with patch("time.monotonic",return_value=10.):r=n.sample()
        self.assertEqual(r["raw_imu"]["orientation"]["w"],1.)
        self.assertEqual(r["raw_ros_odom"]["header"]["frame_id"],"odom")
        self.assertEqual(r["mc_leg_command"]["header"]["stamp"]["sec"],1)
        self.assertEqual(r["imu_quaternion_norm_squared"],1.)

    def test_zero_or_invalid_imu_quaternion_fails_before_motion(self):
        from unittest.mock import patch
        for value in (0.,float("nan"),3.):
            n=self.sample_node();n.messages["imu"]["orientation"]["w"]=value
            with patch("time.monotonic",return_value=10.):
                with self.assertRaisesRegex(RuntimeError,"INVALID_IMU_QUATERNION"):n.sample()


if __name__=="__main__":unittest.main()
