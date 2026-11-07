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

    def test_micro_directions_preserve_cross_track_guard(self):
        pose=dict(x=0.,y=.051,z=.65,roll=0.,pitch=0.,yaw=0.,vx=0.,vy=0.)
        root=dict(x=0.,y=0.,yaw=0.)
        self.assertEqual(micro.physical_fault(dict(joints={}),pose,{},root,"forward"),"MICRO_CROSS_TRACK")
        self.assertIsNone(micro.physical_fault(dict(joints={}),pose,{},root,"left"))
        pose["vx"]=.201
        self.assertEqual(micro.physical_fault(dict(joints={}),pose,{},root,"left"),"PLANAR_SPEED")

    def test_source_joint_bounds_are_not_raised(self):
        p=dict(z=.65,roll=0.,pitch=0.,vx=0.,vy=0.)
        r=dict(joints=dict(knee=dict(q=.1,qvel=1.01)))
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

    def runtime_methods(self):
        tree=ast.parse(Path(micro.__file__).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
        cls=next(n for n in run.body if isinstance(n,ast.ClassDef))
        cls.body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in ("publish","deadline_zero","observe")]
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


if __name__=="__main__":unittest.main()
