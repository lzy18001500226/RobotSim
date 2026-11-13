import copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.research.issue46.native_runtime import standing_waist_preposition as w
from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as native
from scripts.research.issue46.native_runtime import native_body_height as body
class TorsoTurnTest(unittest.TestCase):
    def proposal(self):
        return dict(schema="robotsim.standing-grasp-posture.v1",roll_mode="RETAIN_MEASURED",waist_target_rad={"waist_yaw_joint":.18,"waist_pitch_joint":.25},body_drop_m=.1,grasp_height_above_center_m=.01,grasp_height_from_bottom_m=.12)
    def test_clearance_posture_is_invariant_to_actual_base_translation_and_yaw(self):
        I=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        row=dict(bodies={"pelvis":dict(position_m=[0.,0.,0.],rotation=I),"R_omnipicker_base_link":dict(position_m=[.3,0.,.5],rotation=I)})
        self.assertAlmostEqual(w.clearance_posture_error(row,[.1,0.,0.],[.4,0.,.5])[2],0.)
        R=[0.,-1.,0.,1.,0.,0.,0.,0.,1.]
        row["bodies"]["pelvis"]=dict(position_m=[.017,-.03,.02],rotation=R)
        row["bodies"]["R_omnipicker_base_link"]=dict(position_m=[.017,.27,.52],rotation=R)
        self.assertAlmostEqual(w.clearance_posture_error(row,[.1,0.,0.],[.4,0.,.5])[2],0.)
        row["bodies"]["R_omnipicker_base_link"]["position_m"][0]+=.003
        self.assertAlmostEqual(w.clearance_posture_error(row,[.1,0.,0.],[.4,0.,.5])[2],.003)
    def test_postured_startup_reuses_already_active_sd_without_reissuing(self):
        script=native.stl_hand_startup.build_stand_client_script(observe_only=True)
        self.assertNotIn("create_client(SetMcAction",script)
        self.assertIn("time.monotonic() + 2.5",script)
        self.assertIn("math.radians(15.0)",script)
        self.assertIn("planar_drift <= 0.05",script)
    def test_live_ik_reserves_both_anatomical_and_compensated_command_limits(self):
        import numpy as np
        from scripts.research.issue46.grasp.stl_approved_circle_ik import position_mode_tracking_bounds
        source=np.array([[-.523599,.523599],[-.724,1.50971]])
        bias=np.array([-.7,.1]);kp=np.array([30.,30.])
        bounded=position_mode_tracking_bounds(source,bias,kp)
        self.assertGreater(bounded[0,0],-.4984746128846975)
        for target in (bounded[:,0],bounded[:,1]):
            command=target+bias/kp
            self.assertTrue(np.all(command>=source[:,0]+.01-1e-12))
            self.assertTrue(np.all(command<=source[:,1]-.01+1e-12))
        with self.assertRaises(ValueError):position_mode_tracking_bounds(source,bias,[0.,30.])
    def test_frozen_gains_are_loaded_once_not_per_feedback_tick(self):
        from scripts.research.issue46.grasp import stl_approved_circle_ik as ik
        reference={};gain={name:(30.,2.) for name in ik.ik.ARM}
        with patch("scripts.research.issue46.native_runtime.stl_hand_startup.load_animation_player_arm_pd_parameters",return_value=gain) as loader:
            for _ in range(100):
                self.assertEqual(ik.source_position_mode_kp(reference),(30.,)*7)
            loader.assert_called_once()
    def test_position_mode_compensation_preserves_source_gains(self):
        desired=.175;actual=.175;bias=-8.2;kp=40.;kd=5.
        command=w.position_mode_setpoint(desired,actual,0.,bias,kp,kd,(-.314,.314,13.088),36.)
        self.assertAlmostEqual(command,-.03)
        self.assertAlmostEqual(kp*(command-actual),bias)
        with self.assertRaises(RuntimeError):w.position_mode_setpoint(.31,-.30,10.,-30.,kp,kd,(-.314,.314,13.088),36.)
    def test_proposal_allows_turn_and_forward_bend_only(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"p.json";value=self.proposal();p.write_text(json.dumps(value))
            limits={n:(-1,1,2) for n in w.WAIST}
            self.assertEqual(w.load_proposal(p,limits)["roll_mode"],"RETAIN_MEASURED")
            value["waist_target_rad"]["waist_roll_joint"]=.33;p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):w.load_proposal(p,limits)
    def test_native_freshness_verdict_is_not_the_joint_payload(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"native_latest.json";stamp=time.monotonic_ns()
            row=dict(generation=1,process_pid=8,engine_version=337,host_monotonic_ns=stamp,joints={"waist_pitch_joint":{"q":.2}})
            p.write_text(json.dumps(row))
            with patch("scripts.research.issue46.native_runtime.native_grasp_audit.require_fresh_sample",return_value={k:row[k] for k in ("generation","process_pid")}):
                self.assertEqual(w.read_native(p)["joints"]["waist_pitch_joint"]["q"],.2)
    def test_wrong_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"native_latest.json";p.write_text(json.dumps(dict(generation=2,process_pid=8,engine_version=337,host_monotonic_ns=time.monotonic_ns())))
            with patch("scripts.research.issue46.native_runtime.native_grasp_audit.require_fresh_sample",return_value=dict(generation=1,process_pid=8)):
                with self.assertRaises(RuntimeError):w.read_native(p)
    def test_clearance_path_is_open_and_measured_before_original_approach(self):
        import ast
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        gains=native.stl_hand_startup.load_animation_player_arm_pd_parameters()
        script=h.build_hal_bilateral_arm_plan_script(gains,[0.]*7,[0.]*7,duration_s=6.,hold_s=1.,restore=False,use_native_joint_feedback=True)
        ast.parse(script)
        self.assertIn('CLEARANCE_WAYPOINT_NOT_REACHED',script)
        self.assertLess(script.index('measured_clearance_waypoint'),script.index('n.approach_start=dict(start)'))
        self.assertIn("'PREGRASP_CLEARANCE',HAND_OPEN",script)
        self.assertIn("preclose_waist_error<=.05",script)
        self.assertIn("source_effort_limits_nm",script)
        self.assertIn("'desired_target_rad':dict(target)",script)
        self.assertIn("'published_target_rad':published_positions",script)
    def test_gain_cycle_reuse_rejects_unproven_model_change(self):
        from scripts.research.issue46.grasp import stl_approved_circle_native_trial as trial
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"current.json";p.write_text(json.dumps(dict(generated=dict(host_scene=dict(sha256="changed")))))
            prior=Path(t)/"prior.json";prior.write_text(json.dumps(dict(generated=dict(host_scene=dict(sha256="original")))))
            (Path(t)/"result.json").write_text(json.dumps(dict(claw_control_overlay=dict(path=str(prior)))))
            with patch.object(trial,"verified_grasp_reuse",return_value=dict(scene_sha256="original")):
                with self.assertRaisesRegex(ValueError,"bottle-only"):trial.verified_claw_cycle_reuse(Path(t),p)
    def test_gain_cycle_reuse_requires_exact_fixture_parent(self):
        from scripts.research.issue46.grasp import stl_approved_circle_native_trial as trial
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);scene=root/"source.xml";scene.write_text("original robot model")
            parent=root/"parent.json";parent.write_text(json.dumps(dict(generated=dict(host_scene=dict(path=str(scene),sha256=trial.native.sha256(scene))))))
            fixture=dict(robot_and_environment_unchanged=True,mj_step_calls=0,source_height_m=.30,height_m=.22,diameter_m=.07,mass_kg=.2,parent=dict(path=str(parent),sha256=trial.native.sha256(parent)))
            current=root/"current.json";current.write_text(json.dumps(dict(generated=dict(host_scene=dict(sha256="changed")),fixture_height_delta=fixture)))
            prior=root/"prior.json";prior.write_text(json.dumps(dict(generated=dict(host_scene=dict(path=str(scene),sha256=trial.native.sha256(scene))))))
            (root/"result.json").write_text(json.dumps(dict(claw_control_overlay=dict(path=str(prior)))))
            with patch.object(trial,"verified_grasp_reuse",return_value=dict(scene_sha256=trial.native.sha256(scene))):
                result=trial.verified_claw_cycle_reuse(root,current)
                self.assertIn("NOT_IK",result["scope"])
                parent.write_text("{}")
                with self.assertRaisesRegex(ValueError,"parent changed"):trial.verified_claw_cycle_reuse(root,current)
    def test_cycle_cannot_be_reused_across_different_contact_algorithms(self):
        from scripts.research.issue46.grasp import stl_approved_circle_native_trial as trial
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prior=root/"prior.json";prior.write_text(json.dumps(dict(generated=dict(host_scene=dict(sha256="original")))))
            current=root/"current.json";current.write_text(json.dumps(dict(numeric_contact_manifold=dict(new_enableflags=16))))
            (root/"result.json").write_text(json.dumps(dict(claw_control_overlay=dict(path=str(prior)))))
            with patch.object(trial,"verified_grasp_reuse",return_value=dict(scene_sha256="original")):
                with self.assertRaisesRegex(ValueError,"changed contact algorithm"):trial.verified_claw_cycle_reuse(root,current)
    def test_generated_relay_retains_single_owner(self):
        gains=native.stl_hand_startup.load_animation_player_arm_pd_parameters()
        script=w.extend_bootstrap(native.arm_bootstrap_hold_script(gains,relay_waist=True))
        self.assertEqual(script.count("self.waist_pub.publish(m)"),1)
        self.assertIn("emit_manual_waist(n)",script)
        self.assertIn("STALE_MC_WAIST_RELAY",script)
        compile(script,"waist_owner","exec")
    def test_stale_native_sample_never_publishes(self):
        owner=SimpleNamespace(waist_request=dict(schema="robotsim.waist-preposition.v1",duration_s=4.,created_host_ns=time.monotonic_ns(),generation=2),waist_read_at=time.monotonic(),waist_native=dict(generation=2,host_monotonic_ns=time.monotonic_ns()-500000000))
        with self.assertRaisesRegex(RuntimeError,"NOT_FRESH"):w.override_waist(owner,SimpleNamespace(joints=[]))
    def test_posture_observation_has_no_body_or_velocity_publisher(self):
        script=body.build_script({},[],.10,mode="observe",label="body_posture_hold")
        self.assertIn('if mode=="observe":',script)
        self.assertIn("self.pub=None",script)
        self.assertIn("BODY_OBSERVER_VELOCITY_FORBIDDEN",script)
        self.assertLess(script.index('if mode=="observe":'),script.index('if mode!="restore":request_source(1001)'))
        compile(script,"observer","exec")
if __name__=="__main__":unittest.main()


class WaistCallbackMemoryBoundaryTests(unittest.TestCase):
    def state(self, source_ns):
        names=w.WAIST
        request=dict(schema='robotsim.waist-preposition.v1',duration_s=4.,lease_s=300.,generation=1,created_host_ns=9000000000,start_sim_time_s=20.,source_limits={n:(-.3,.3,3.) for n in names},target_rad={n:.1 for n in names},start_rad={n:0. for n in names},effort_caps_nm={n:36. for n in names})
        owner=SimpleNamespace(waist_request=request,waist_report_at=0.,waist_native=dict(generation=1,host_monotonic_ns=source_ns,sim_time_s=21.,joints={n:dict(q=0.,qvel=0.,bias_force=0.) for n in names}),get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:SimpleNamespace(sec=10,nanosec=0))))
        msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10,nanosec=0)),joints=[SimpleNamespace(name=n,position=0.,velocity=0.,effort=0.,stiffness=40.,damping=5.) for n in names])
        return owner,msg
    def test_callback_operates_with_filesystem_unavailable(self):
        owner,msg=self.state(10000000000)
        with patch('builtins.open',side_effect=AssertionError('no disk I/O in MC callback')),patch('os.path.exists',side_effect=AssertionError('no stat in MC callback')),patch('time.monotonic',return_value=10.),patch('time.monotonic_ns',return_value=10000000000):
            self.assertIs(w.override_waist(owner,msg),msg)
        self.assertAlmostEqual(msg.joints[0].position,.015625)
        self.assertEqual(owner.waist_pending_feedback['generation'],1)
    def test_cached_source_freshness_limit_is_unchanged(self):
        owner,msg=self.state(9790000000)
        with patch('time.monotonic',return_value=10.),patch('time.monotonic_ns',return_value=10000000000):
            with self.assertRaisesRegex(RuntimeError,'NATIVE_STATE_NOT_FRESH'):w.override_waist(owner,msg)


class DiagnosticLoggingOverlayTests(unittest.TestCase):
    def test_logging_overlay_leaves_all_control_fields_intact(self):
        source={'aimrt':{'log':{'core_lvl':'Info','default_module_lvl':'Info'},'channel':{'topics':['a','b']}},'controller':{'gain':40.,'range':[-.314,.314]}}
        original=copy.deepcopy(source);out=w.bounded_logging_overlay(source)
        self.assertEqual(source,original)
        self.assertEqual(out['controller'],source['controller'])
        self.assertEqual(out['aimrt']['channel'],source['aimrt']['channel'])
        self.assertEqual(out['aimrt']['log'],{'core_lvl':'Info','default_module_lvl':'Warn'})


class CompiledTorqueBoundedCarryTests(unittest.TestCase):
    def test_frozen_wrist_cap_bounds_command_without_qpos_clamp(self):
        desired=-.171;actual=-.138;velocity=2.035;bias=-.761;kp=30.;kd=1.;limits=(-.523599,.523599,10.472)
        command=w.position_mode_setpoint(desired,actual,velocity,bias,kp,kd,limits,2.2,bound_effort=True)
        self.assertLessEqual(abs(kp*(command-actual)-kd*velocity),2.2)
        self.assertGreater(command,limits[0]);self.assertLess(command,limits[1])
        self.assertEqual(actual,-.138)
        with self.assertRaises(RuntimeError):w.position_mode_setpoint(.31,-.30,10.,-30.,40.,5.,(-.314,.314,13.088),36.)
    def test_infeasible_effort_and_source_position_intersection_stops(self):
        with self.assertRaises(RuntimeError):w.position_mode_setpoint(.5,.52,10.,0.,30.,1.,(-.523599,.523599,10.472),2.2,bound_effort=True)


class ManualWaistCycleTests(unittest.TestCase):
    def test_manual_cycle_uses_captured_gains_and_fresh_actual_state(self):
        owner,msg=WaistCallbackMemoryBoundaryTests().state(10000000000)
        owner.waist_manual_template=copy.deepcopy(msg);owner.waist_last_receipt=0.
        sent=[];owner.waist_pub=SimpleNamespace(publish=sent.append)
        with patch('time.monotonic',return_value=10.),patch('time.monotonic_ns',return_value=10000000000):w.emit_manual_waist(owner)
        self.assertEqual(len(sent),1);self.assertEqual(sent[0].joints[0].stiffness,40.)
        self.assertAlmostEqual(sent[0].joints[0].position,.015625)
        self.assertEqual(owner.waist_manual_template.joints[0].position,0.)
        owner.waist_request=None;w.emit_manual_waist(owner);self.assertEqual(len(sent),1)

class UnownedWaistInputPauseTests(unittest.TestCase):
    def test_old_input_is_not_ready_but_fresh_input_resumes_before_deadline(self):
        n=SimpleNamespace(waist_request=None,waist_last_receipt=0.)
        with patch('time.monotonic',return_value=.21):self.assertFalse(w.bootstrap_unowned_waist_ready(n))
        self.assertTrue(n.waist_relay_paused)
        n.waist_last_receipt=.20
        with patch('time.monotonic',return_value=.22):self.assertTrue(w.bootstrap_unowned_waist_ready(n))
        self.assertFalse(n.waist_relay_paused)
    def test_persistent_gap_or_active_forearm_without_manual_owner_still_fails(self):
        for age,forearm in [(1.01,None),(.21,{})]:
            n=SimpleNamespace(waist_request=None,waist_last_receipt=0.,forearm_request=forearm)
            with patch('time.monotonic',return_value=age):
                with self.assertRaisesRegex(RuntimeError,'STALE_MC_WAIST_RELAY'):w.bootstrap_unowned_waist_ready(n)
        n=SimpleNamespace(waist_request={'captured':True},waist_last_receipt=0.)
        with patch('time.monotonic',return_value=2.):self.assertTrue(w.bootstrap_unowned_waist_ready(n))


class WaistSimulationClockTests(unittest.TestCase):
    def test_slow_physics_does_not_rush_target_on_host_clock(self):
        request=dict(start_sim_time_s=20.,duration_s=4.)
        self.assertAlmostEqual(w.waist_trajectory_fraction(request,dict(sim_time_s=21.)),.15625)
        # Host time/age is intentionally not a motion-progress input.
        owner,msg=WaistCallbackMemoryBoundaryTests().state(14000000000)
        with patch('time.monotonic',return_value=14.),patch('time.monotonic_ns',return_value=14000000000):
            w.override_waist(owner,msg)
        self.assertAlmostEqual(msg.joints[0].position,.015625)
    def test_sim_reset_and_nonfinite_clock_fail_closed(self):
        for current in (19.,float('nan'),float('inf')):
            with self.assertRaisesRegex(RuntimeError,'SIM_CLOCK_INVALID'):
                w.waist_trajectory_fraction(dict(start_sim_time_s=20.,duration_s=4.),dict(sim_time_s=current))
    def test_missing_clock_cannot_fall_back_to_host_time(self):
        with self.assertRaises(KeyError):w.waist_trajectory_fraction(dict(duration_s=4.),dict(sim_time_s=21.))
    def test_completion_requires_all_physical_seconds(self):
        self.assertEqual(w.waist_trajectory_fraction(dict(start_sim_time_s=20.,duration_s=4.),dict(sim_time_s=24.)),1.)
