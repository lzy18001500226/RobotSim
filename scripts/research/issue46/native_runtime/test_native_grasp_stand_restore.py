"""Offline ownership and source-limit checks; no ROS or physics claims."""
import copy
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from scripts.research.issue46.native_runtime import native_grasp_stand_restore as restore


class WaistHandoffTests(unittest.TestCase):
    def setUp(self):
        self.sent=[]
        self.owner=S(mc_waist_blend_started=10.,mc_waist_blend_from={n:.2 for n in restore.WAIST},
                     mc_waist_report_at=100.,waist_pub=S(publish=lambda m:self.sent.append(copy.deepcopy(m))))
        self.message=S(header=S(stamp=S(sec=1,nanosec=0)),joints=[S(name=n,position=0.,velocity=0.,effort=.1,stiffness=20.,damping=2.) for n in restore.WAIST])
        self.owner.get_clock=lambda:S(now=lambda:S(nanoseconds=1100000000))
        self.contract=patch.object(restore,'LIFT_CONTRACT',{'retract_preflight':{'source_motion_limits':{n:(-1.,1.,5.) for n in restore.WAIST}}},create=True)
        self.contract.start();self.addCleanup(self.contract.stop)

    def test_existing_owner_keeps_publication_until_explicit_handoff(self):
        self.owner.mc_waist_blend_started=None
        with patch('time.monotonic',return_value=12.):restore.receive_native_waist(self.owner,self.message)
        self.assertEqual(self.sent,[])
        self.assertEqual(self.owner.mc_waist_message.joints[0].position,0.)

    def test_blend_is_continuous_and_preserves_native_gains(self):
        for now,expected in [(10.,.2),(12.,.1),(14.,0.)]:
            for joint in self.message.joints:joint.position=0.
            self.message.header.stamp.nanosec+=1
            with patch('time.monotonic',return_value=now):restore.receive_native_waist(self.owner,self.message)
            self.assertAlmostEqual(self.sent[-1].joints[0].position,expected)
            self.assertEqual(self.sent[-1].joints[0].stiffness,20.)
            self.assertEqual(self.sent[-1].joints[0].damping,2.)
            self.assertEqual(self.message.joints[0].position,expected)
            # The real subscriber receives a new message for every callback.
            self.message.joints[0].position=0.

    def test_source_range_violation_stops_without_publishing(self):
        self.message.joints[0].position=1.1
        with patch('time.monotonic',return_value=14.):
            with self.assertRaisesRegex(RuntimeError,'MC_WAIST_PUBLISHED_TARGET_RANGE'):restore.receive_native_waist(self.owner,self.message)
        self.assertEqual(self.sent,[])

    def test_unchanged_mc_source_endpoints_are_legal_targets(self):
        for k,endpoint in enumerate((-1.,1.)):
            self.message.joints[0].position=endpoint
            self.message.header.stamp.nanosec=k+1
            with patch('time.monotonic',return_value=14.):restore.receive_native_waist(self.owner,self.message)
            self.assertEqual(self.sent[-1].joints[0].position,endpoint)
        self.assertEqual(len(self.sent),2)

    def test_actual_pitch_limit_endpoint_is_forwarded_exactly_after_handoff(self):
        # Exact failed arithmetic from the physical carry attempt.
        self.owner.mc_waist_blend_from["waist_pitch_joint"]=0.02030692677861748
        self.assertLess(0.02030692677861748+(-.314-0.02030692677861748),-.314)
        restore.LIFT_CONTRACT["retract_preflight"]["source_motion_limits"]["waist_pitch_joint"]=(-.314,.314,13.088)
        pitch=next(j for j in self.message.joints if j.name=="waist_pitch_joint")
        pitch.position=-.314
        with patch('time.monotonic',return_value=14.):restore.receive_native_waist(self.owner,self.message)
        actual=next(j.position for j in self.sent[-1].joints if j.name=="waist_pitch_joint")
        self.assertEqual(actual,-.314)
        self.assertEqual(actual.hex(),(-.314).hex())
    def test_wall_clock_rollback_accepts_fresh_new_epoch_packet(self):
        walls=iter((1100000000,981000000))
        self.owner.get_clock=lambda:S(now=lambda:S(nanoseconds=next(walls)))
        with patch('time.monotonic_ns',side_effect=[9999999000,10000000000,10000001000,10000002000,10000003000,10000004000,10009999000,10010000000,10010001000,10010002000,10010003000,10010004000]),patch('time.monotonic',return_value=14.):
            restore.receive_native_waist(self.owner,self.message)
            self.message.header.stamp.sec=0;self.message.header.stamp.nanosec=980000000
            restore.receive_native_waist(self.owner,self.message)
        self.assertEqual(len(self.sent),2)
        self.assertEqual(self.owner.mc_waist_source_clock_generation,1)
        self.assertEqual(self.owner.mc_waist_source_stamp_ns,980000000)

    def test_old_source_message_does_not_become_fresh_on_receipt(self):
        self.owner.mc_waist_blend_started=None
        self.owner.get_clock=lambda:S(now=lambda:S(nanoseconds=2000000000))
        restore.receive_native_waist(self.owner,self.message)
        self.assertFalse(hasattr(self.owner,'mc_waist_receipt'))
        self.assertEqual(self.sent,[])

    def test_repeated_source_stamp_does_not_refresh_receipt(self):
        self.owner.mc_waist_blend_started=None
        with patch('time.monotonic',return_value=12.):restore.receive_native_waist(self.owner,self.message)
        with patch('time.monotonic',return_value=15.):restore.receive_native_waist(self.owner,self.message)
        self.assertEqual(self.owner.mc_waist_receipt,12.)


class NativeHoldClockTests(unittest.TestCase):
    def run_hold(self, speed, generation_change=False):
        from scripts.research.issue46.native_runtime.stl_hand_startup import hold_native_simulation
        clock=[0.]
        n=S(native_row={'sim_time_s':0.,'generation':1},refresh_native=lambda:None)
        def send(target,seconds,stage,hand):
            clock[0]+=seconds
            n.native_row['sim_time_s']+=seconds*speed
            if generation_change:n.native_row['generation']=2
        n.send=send
        with patch('time.monotonic',side_effect=lambda:clock[0]):hold_native_simulation(n,{},'RETRACT_HOLD',.5)
        return clock[0],n.native_row['sim_time_s']

    def test_slow_physics_is_held_for_actual_simulated_duration(self):
        wall,sim=self.run_hold(.5)
        self.assertGreaterEqual(sim,1.5)
        self.assertGreaterEqual(wall,3.)
        self.assertLess(wall,5.)

    def test_stalled_physics_stops_at_wall_watchdog(self):
        with self.assertRaisesRegex(RuntimeError,'HOLD_SIMULATION_TIME_INCOMPLETE'):self.run_hold(0.)

    def test_clock_generation_change_stops(self):
        with self.assertRaisesRegex(RuntimeError,'HOLD_SIMULATION_CLOCK_CHANGED'):self.run_hold(1.,True)


class DeliveryCommandBoundTests(unittest.TestCase):
    def test_later_delivery_never_overrides_reduced_angular_cap(self):
        import ast
        from scripts.research.issue46.native_runtime.native_base_micro_probe import build_carry_route_script
        import json
        from pathlib import Path
        profile=json.loads((Path(__file__).resolve().parents[4]/'simulation/mujoco/scenes/raicom2026/config/reviewed_carry_step_forward.json').read_text())
        code=build_carry_route_script(None,{},[],carry_pid=42,away_1m=True,straight_retreat_turn=True,transit_policy_050=True,compact_turn=True,gait_prepare=True,compact_centre_feedback=True,release_source_on_stop=True,compact_angular_cap=.3,delivery=profile,vendor_transition_table=True)
        updates=[n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='update' and isinstance(n.func.value,ast.Name) and n.func.value.id=='MICRO']
        effective=ast.literal_eval(updates[-1].args[0])['carry']
        self.assertEqual(effective['angular_command_cap'],.3)
        self.assertEqual(effective['delivery']['delivery_tracking_angular_cap_rad_s'],.3)
        self.assertEqual(profile['delivery_tracking_angular_cap_rad_s'],.6)


class CruiseCalibrationTests(unittest.TestCase):
    def trajectory(self,accelerating):
        return [dict(sample_monotonic_s=i*.05,sim_time_s=i*.05,native_generation=1,source_clock_generation=1,get_mode_desc='LOCOMOTION_STEP',common_mode_desc='LOCOMOTION_STEP',pose=dict(x=(.1*(i*.05)**2 if accelerating else .3*i*.05),y=0.)) for i in range(61)]
    def test_acceleration_is_not_cruise_calibration(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import step_cruise_reference
        self.assertIsNone(step_cruise_reference(self.trajectory(True),dict(x=.9,y=0.),[3.,0.],3.))
    def test_constant_motion_qualifies_and_reset_epoch_does_not(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import step_cruise_reference
        trace=self.trajectory(False)
        self.assertAlmostEqual(step_cruise_reference(trace,dict(x=.9,y=0.),[3.,0.],3.),.3)
        trace[-1]['source_clock_generation']=2
        self.assertIsNone(step_cruise_reference(trace,dict(x=.9,y=0.),[3.,0.],3.))


if __name__=='__main__':unittest.main()


class BootstrapFeedbackOwnershipTests(unittest.TestCase):
    def test_bootstrap_uses_one_exclusive_callback_executor(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import arm_bootstrap_hold_script
        from scripts.research.issue46.native_runtime.standing_waist_preposition import extend_bootstrap
        code=extend_bootstrap(arm_bootstrap_hold_script({},relay_waist=True,initial_hand_opening=1.,initial_arm_target={},max_hold_s=300.))
        compile(code,'generated_bootstrap','exec')
        self.assertNotIn('rclpy.spin_once(n',code)
        self.assertEqual(code.count('feedback_executor.spin()'),1)
        self.assertIn('feedback_executor.shutdown(timeout_sec=1.)',code)
        self.assertIn("time.monotonic()-self.waist_last_receipt>.20",code)


class CoordinatedTransportOwnershipTests(unittest.TestCase):
    def test_disabled_mc_arm_and_hand_have_no_public_command_endpoints(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import build_coordinated_waist_relay_config
        source={'MotionControlModule':{'PublisherManager':{'publisher_cfg_list':[dict(name=group+'_joint_command',topic='/aima/hal/joint/'+group+'/command',enable=True,frequency=500) for group in ('arm','hand','waist','leg')]}},'aimrt':{'channel':{'pub_topics_options':[dict(topic_name='/aima/hal/joint/'+group+'/command',enable_backends=['ros2']) for group in ('arm','hand','waist','leg')]}}}
        original=copy.deepcopy(source)
        cfg=build_coordinated_waist_relay_config(source,isolate_disabled_topics=True)
        pub={p['name']:p for p in cfg['MotionControlModule']['PublisherManager']['publisher_cfg_list']}
        for group in ('arm','hand'):
            self.assertFalse(pub[group+'_joint_command']['enable'])
            self.assertEqual(pub[group+'_joint_command']['topic'],'/robotsim/issue46/inactive_mc/'+group+'/command')
        self.assertEqual(pub['waist_joint_command']['topic'],'/robotsim/issue46/mc/waist/command')
        self.assertTrue(pub['waist_joint_command']['enable'])
        self.assertEqual(pub['leg_joint_command'],original['MotionControlModule']['PublisherManager']['publisher_cfg_list'][3])
        topics={p['topic_name'] for p in cfg['aimrt']['channel']['pub_topics_options']}
        self.assertNotIn('/aima/hal/joint/arm/command',topics)
        self.assertNotIn('/aima/hal/joint/hand/command',topics)
        self.assertEqual(source,original)


class CarryTurnRampTests(unittest.TestCase):
    def test_longer_start_ramp_preserves_end_command_cap(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import bounded_start_scale
        self.assertEqual(bounded_start_scale(0.,2.),0.)
        self.assertLess(bounded_start_scale(.3,2.),bounded_start_scale(.3))
        self.assertEqual(bounded_start_scale(2.,2.),1.)
        self.assertEqual(bounded_start_scale(4.,2.),1.)
        with self.assertRaises(RuntimeError):bounded_start_scale(0.,20.)


class PregraspSegmentTests(unittest.TestCase):
    def test_checked_staged_path_keeps_original_per_segment_jump_bound(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import validate_pregrasp_segments
        start={'r':0.,'l':0.};goal={'r':1.29,'l':.05}
        with self.assertRaises(RuntimeError):validate_pregrasp_segments(start,goal,None)
        values=validate_pregrasp_segments(start,goal,{'pass_result':True,'right_arm_waypoints_rad':[{'r':.7}]})
        self.assertLess(max(values),1.1)
        with self.assertRaises(RuntimeError):validate_pregrasp_segments(start,{'r':2.5,'l':0.},{'pass_result':True,'right_arm_waypoints_rad':[{'r':.7}]})


class CarryFrameReexpressionTests(unittest.TestCase):
    def test_reference_switch_keeps_identical_measured_world_pose(self):
        from scripts.research.issue46.native_runtime.native_grasp_stand_restore import torso_carry_reference
        I=[1.,0.,0.,0.,1.,0.,0.,0.,1.];B=[0.,-1.,0.,1.,0.,0.,0.,0.,1.]
        row={'bodies':{'R_omnipicker_base_link':{'position_m':[1.,2.,.8],'rotation':I},'torso_link':{'position_m':[.5,1.8,.7],'rotation':B}}}
        ref=torso_carry_reference(row,[.1,0.,0.],[0.,0.,1.]);delta=ref['carry_rigid_offset_base_m'];axis=ref['carry_rigid_axis_base']
        actual=[row['bodies']['torso_link']['position_m'][i]+sum(B[3*i+j]*delta[j] for j in range(3)) for i in range(3)]
        self.assertEqual(ref['carry_pose_frame'],'TORSO_RIGID')
        for a,b in zip(actual,[1.1,2.,.8]):self.assertAlmostEqual(a,b)
        for i,b in enumerate([0.,0.,1.]):self.assertAlmostEqual(sum(B[3*i+j]*axis[j] for j in range(3)),b)


class ReviewSourceReadinessTests(unittest.TestCase):
    def test_viewer_wait_budget_starts_after_verified_generation(self):
        import json,tempfile
        from pathlib import Path
        from scripts.research.issue46.grasp.live_retreat_turn_review import review_source_ready
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.assertFalse(review_source_ready(root))
            for row,expected in [(None,False),({'generation':0},False),({'generation':1},True)]:
                (root/'live_view_latest.json').write_text(json.dumps(row));self.assertEqual(review_source_ready(root),expected)


class QualifiedCarryFrameLeaseTests(unittest.TestCase):
    def test_torso_frame_requires_same_generation_standing_handoff(self):
        from scripts.research.issue46.native_runtime.native_grasp_audit import carry_lease_limit
        phase={'postured_native_stand_restore':True,'carry_pose_frame':'TORSO_RIGID','generation':1}
        with self.assertRaises(ValueError):carry_lease_limit(phase)
        phase['postured_restore_torso_reference']={'schema':'robotsim.postured-carry-frame.v1','standing_restore_pass':True,'owner_pid':619,'generation':1}
        self.assertEqual(carry_lease_limit(phase),300.)
        phase['postured_restore_torso_reference']['generation']=2
        with self.assertRaises(ValueError):carry_lease_limit(phase)


class MeasuredTurnRefinementTests(unittest.TestCase):
    def test_saved_stop_rebound_is_compensated_without_new_budget(self):
        from scripts.research.issue46.native_runtime.native_base_micro_probe import measured_turn_refinement
        mission={'previous_yaw':2.119681795644191,'unwrapped_yaw':-4.163503511535396}
        stopped={'pose':{'yaw':2.258107682130605},'sample_monotonic_s':10.}
        out=measured_turn_refinement(mission,stopped,1.913119320969793,{'pass':True})
        raw=out['target_unwrapped_yaw']-out['unwrapped_yaw'];command_error=raw-out['turn_observed_stop_delta_rad']
        final=stopped['pose']['yaw']+command_error+out['turn_observed_stop_delta_rad']
        self.assertAlmostEqual(final,1.913119320969793)
        with self.assertRaises(RuntimeError):measured_turn_refinement(out,stopped,1.9,{'pass':True})
        with self.assertRaises(RuntimeError):measured_turn_refinement(mission,stopped,1.9,{'pass':False})


class HeldWaistReturnTests(unittest.TestCase):
    def rows(self):
        return [dict(sim_time_s=10.+i*.02,host_monotonic_ns=10000000000+i*20000000,generation=1,
            joints={name:dict(q=.02,qvel=.01) for name in restore.WAIST}) for i in range(13)]

    def test_loaded_neutral_waist_requires_continuous_measured_window(self):
        rows=self.rows();baseline={name:0. for name in restore.WAIST}
        self.assertTrue(restore.held_waist_return_ready(rows,baseline))
        self.assertFalse(restore.held_waist_return_ready(rows[:5],baseline))
        for q,velocity in ((.307,.01),(.02,.051),(float('nan'),.0)):
            bad=copy.deepcopy(rows);bad[5]['joints']['waist_pitch_joint'].update(q=q,qvel=velocity)
            self.assertFalse(restore.held_waist_return_ready(bad,baseline))

    def test_stale_duplicates_gap_and_generation_switch_cannot_qualify(self):
        baseline={name:0. for name in restore.WAIST}
        for change in ('duplicate','generation','gap'):
            rows=self.rows()
            if change=='duplicate':rows[5]['host_monotonic_ns']=rows[4]['host_monotonic_ns']
            if change=='generation':rows[5]['generation']=2
            if change=='gap':rows=rows[:4]+rows[8:]
            self.assertFalse(restore.held_waist_return_ready(rows,baseline))
