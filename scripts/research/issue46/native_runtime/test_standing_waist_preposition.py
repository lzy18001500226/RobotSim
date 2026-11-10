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
