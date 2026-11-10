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
            with patch.object(trial,"verified_grasp_reuse",return_value=dict(scene_sha256="original")):
                with self.assertRaisesRegex(ValueError,"bottle-only"):trial.verified_claw_cycle_reuse(Path(t),p)
    def test_gain_cycle_reuse_requires_exact_fixture_parent(self):
        from scripts.research.issue46.grasp import stl_approved_circle_native_trial as trial
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);scene=root/"source.xml";scene.write_text("original robot model")
            parent=root/"parent.json";parent.write_text(json.dumps(dict(generated=dict(host_scene=dict(path=str(scene),sha256=trial.native.sha256(scene))))))
            fixture=dict(robot_and_environment_unchanged=True,mj_step_calls=0,source_height_m=.30,height_m=.22,diameter_m=.07,mass_kg=.2,parent=dict(path=str(parent),sha256=trial.native.sha256(parent)))
            current=root/"current.json";current.write_text(json.dumps(dict(generated=dict(host_scene=dict(sha256="changed")),fixture_height_delta=fixture)))
            with patch.object(trial,"verified_grasp_reuse",return_value=dict(scene_sha256=trial.native.sha256(scene))):
                result=trial.verified_claw_cycle_reuse(root,current)
                self.assertIn("NOT_IK",result["scope"])
                parent.write_text("{}")
                with self.assertRaisesRegex(ValueError,"parent changed"):trial.verified_claw_cycle_reuse(root,current)
    def test_generated_relay_retains_single_owner(self):
        gains=native.stl_hand_startup.load_animation_player_arm_pd_parameters()
        script=w.extend_bootstrap(native.arm_bootstrap_hold_script(gains,relay_waist=True))
        self.assertEqual(script.count("self.waist_pub.publish(m)"),1)
        self.assertIn("m=override_waist(self,m)",script)
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
