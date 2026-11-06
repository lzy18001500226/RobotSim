"""Scoped regressions for required native observation and truth feedback."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import TestCase, main, mock
import math
import tempfile
from scripts.research.issue46.native_runtime import native_grasp_audit as audit
from scripts.research.issue46.native_runtime.test_pregrasp_feedback import generated

class NativeLaunchTests(TestCase):
    def test_required_parent_launch_and_single_observer_owner(self):
        parent=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        adapter=Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_native_trial.py').read_text()
        self.assertIn('native_grasp_audit.build(output',parent)
        self.assertLess(parent.index('native_grasp_audit.build(output'),parent.index('        run(command, timeout_s=12.0)'))
        self.assertIn('native_grasp_audit.require_fresh_sample',parent)
        self.assertNotIn('LD_PRELOAD=',adapter)
        self.assertIn('LD_PRELOAD=',audit.launch_prefix())

    def test_observer_fresh_complete_sample_required(self):
        row={'host_monotonic_ns':1000000000,'engine_version':337,'sim_time_s':1.,'generation':0,'bodies':{n:{} for n in ('pelvis','R_omnipicker_base_link','test_cylinder')}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.jsonl';p.write_text(json.dumps(row)+'\n'+ '{"partial":')
            self.assertTrue(audit.require_fresh_sample(p,now_ns=1100000000)['pass'])
            with self.assertRaisesRegex(RuntimeError,'STALE'):audit.require_fresh_sample(p,now_ns=1300000000)
            row['bodies']['pelvis']=None;p.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(RuntimeError,'BODIES'):audit.require_fresh_sample(p,now_ns=1100000000)

    def test_open_gate_precedes_approach_and_does_not_accept_command_only(self):
        code=generated()
        self.assertLess(code.index('HAND_OPEN_NOT_REACHED'),code.index('steps=max(1,int(DURATION'))
        tree=ast.parse(code)
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'abs(actual_open + HAND_OPEN)' in ast.unparse(n.test))
        check=compile(ast.Expression(guard.test),'opening','eval')
        self.assertTrue(eval(check,{'actual_open':-.097,'HAND_OPEN':.5}))
        self.assertFalse(eval(check,{'actual_open':-.5,'HAND_OPEN':.5}))


    def test_planning_snapshot_replaces_every_group_without_mutating_capture(self):
        groups=('arm_positions_rad','leg_positions_rad','waist_positions_rad')
        old={g:{g:0.} for g in groups}
        row={'bodies':{'pelvis':{'position_m':[1.,2.,.65],'rotation':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}},'joints':{g:{'q':.12} for g in groups},'host_monotonic_ns':10,'sim_time_s':2.,'generation':1}
        new=audit.planning_state_from_native(row,old)
        for g in groups:
            self.assertEqual(new[g][g],.12);self.assertEqual(old[g][g],0.)
        self.assertEqual(new['base_pose_world']['quaternion_wxyz'],[1.,0.,0.,0.])
        del row['joints'][groups[0]]
        with self.assertRaises(KeyError):audit.planning_state_from_native(row,old)


    def test_coupled_budget_respects_negative_transmission_and_command_bound(self):
        import mujoco
        xml='<mujoco><worldbody><body><joint name="root" frictionloss="0.3"/><geom size=".01"/><body><joint name="follower" frictionloss="0.3"/><geom size=".01"/></body></body></worldbody><equality><joint joint1="follower" joint2="root" polycoef="0 -2 0 0 0"/></equality><actuator><position joint="root" gear="-1" kp="2" ctrllimited="true" ctrlrange="0 1"/></actuator></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml)
        budget=audit.coupled_opening_budget(m,0,0,-.6)
        self.assertAlmostEqual(budget['estimated_coupled_coulomb_nm'],.9)
        self.assertAlmostEqual(budget['opening_command_to_balance_coulomb_estimate'],1.05)
        self.assertFalse(budget['estimate_within_command_range'])


    def test_startup_open_is_explicit_and_walking_requires_measured_opening(self):
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='arm_bootstrap_hold_script')
        ns={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),'bootstrap','exec'),ns)
        default=ns[fn.name]({});opened=ns[fn.name]({},initial_hand_opening=1.)
        self.assertIn('INITIAL_HAND_OPENING=None',default)
        self.assertIn('INITIAL_HAND_OPENING=1.0',opened)
        self.assertIn('and n.pub.get_subscription_count()==1:',opened)
        compile(opened,'open_bootstrap','exec')
        with self.assertRaises(ValueError):ns[fn.name]({},initial_hand_opening=.5)
        self.assertLess(source.index('INITIAL_HANDS_NOT_FULLY_OPEN'),source.index('dock = run('))
        self.assertIn('abs(q+1.0)<=.03',source)
        code=generated();tree=ast.parse(code)
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='not startup_open')
        self.assertIn('HAND_ACTIVATION',ast.unparse(guard.body))
        self.assertNotIn('n.send',ast.unparse(guard.orelse))


    def test_friction_diagnostic_changes_only_six_named_joint_attributes(self):
        import xml.etree.ElementTree as ET
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import change_friction, JOINTS
        xml='<mujoco><worldbody><body>'+''.join('<joint name="'+n+'" axis="0 0 1"/>' for n in JOINTS+('left_claw_joint',))+'</body></worldbody></mujoco>'
        before=ET.fromstring(xml);after=ET.fromstring(change_friction(xml))
        for a,b in zip(before.iter(),after.iter()):
            expected=dict(a.attrib)
            if a.tag=='joint' and a.get('name') in JOINTS:expected['frictionloss']='0'
            self.assertEqual(b.attrib,expected)
        with self.assertRaises(ValueError):change_friction('<mujoco/>')

    def test_open_diagnostic_returns_before_locomotion(self):
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text();tree=ast.parse(source)
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='args.startup_opening_diagnostic')
        self.assertTrue(any(isinstance(n,ast.Return) for n in branch.body))
        self.assertNotIn('SetMcInputSource',ast.unparse(branch))
        self.assertNotIn('native_locomotion_dock.py',ast.unparse(branch))


    def test_both_friction_scope_still_preserves_all_other_attributes(self):
        import xml.etree.ElementTree as ET
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import change_friction, JOINTS
        names=JOINTS+tuple('L_'+n[2:] for n in JOINTS)
        xml='<mujoco><worldbody><body>'+''.join('<joint name="'+n+'" axis="0 0 1"/>' for n in names)+'</body></worldbody></mujoco>'
        result=ET.fromstring(change_friction(xml,both_hands=True))
        self.assertEqual(len(result.findall('.//joint[@frictionloss="0"]')),12)

    def test_native_truth_dock_source_is_explicit_and_absolute(self):
        from scripts.research.issue46.native_runtime.test_dock_feedback import script
        source=script()
        self.assertIn("PLAN.get('native_pose_feedback')",source)
        self.assertIn('return world_xy[0],world_xy[1],wrap(world_yaw)',source)
        self.assertIn("NATIVE_DOCK_RESET_GENERATION_CHANGED",source)
        self.assertIn("STALE_NATIVE_DOCK_POSE",source)
        compile(source,'native_dock.py','exec')


    def test_latest_snapshot_and_live_tracking_preserve_deadlines(self):
        code=generated()
        self.assertIn('/workspace/evidence/native_latest.json',code)
        self.assertIn('/workspace/evidence/native_tracking_target.json',code)
        self.assertIn('BOTTLE_MOVED_OR_TILTED_BEFORE_CLOSURE',code)
        self.assertIn('STALE_NATIVE_CIRCLE_TRACKING_TARGET',code)
        self.assertIn('self.approach_fraction',code)
        self.assertIn('n.closing_started=True',code)
        compile(code,'tracked_plan.py','exec')


    def test_native_contact_does_not_promote_disconnected_samples(self):
        tree=ast.parse(generated());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_bilateral_contact')
        ns={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),'contact','exec'),ns)
        contacts=[{'body1':'test_cylinder','body2':n,'local_force_torque':[1.,0.,0.,0.,0.,0.]} for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')]
        rows=[{'host_monotonic_ns':int(t*1e9),'sim_time_s':t,'contacts':contacts} for t in (1.,1.02,1.04,1.5,1.52)]
        with mock.patch('builtins.open',return_value=io.BytesIO(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())):
            self.assertFalse(ns['native_bilateral_contact'](.5)['pass'])


    def test_tracking_readiness_does_not_relax_active_freshness(self):
        code=Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_ik.py').read_text()
        self.assertIn('readiness_deadline=time.monotonic()+3.',code)
        self.assertIn('0.<=initial_age<=.10',code)
        self.assertIn('0<=age<=.20',code)
        self.assertIn('NATIVE_TRACKING_COMPUTE_DEADLINE',code)
        compile(code,'circle.py','exec')


    def test_tracking_uses_kinematics_not_full_dynamics_for_fast_updates(self):
        tree=ast.parse(Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_ik.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='solve_tracking_endpoint')
        code=ast.unparse(fn)
        self.assertIn('mj_kinematics',code)
        self.assertIn('mj_comPos',code)
        self.assertIn('mj_jac',code)
        self.assertNotIn('mj_forward',code)
        self.assertIn('LIVE_CIRCLE_IK_NOT_CONVERGED',code)
        self.assertIn('BOTTLE_MOVED_BEFORE_CLOSURE',code)


    def test_native_feedback_cannot_refresh_from_stale_or_reset_generation(self):
        code=generated();tree=ast.parse(code)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Plan')
        cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('refresh_native','arm_fresh')]
        clock=NS(ns=1100000000)
        ns={'USE_NATIVE_FEEDBACK':True,'CONTROL':('left','right'),'TRACK_NATIVE_CIRCLE':False,'time':NS(monotonic_ns=lambda:clock.ns),'json':json,'math':math}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'feedback','exec'),ns)
        n=ns['Plan']();n.arm={};n.native_generation=None;n.native_receipt_ns=None;n.native_sim_time=None
        row={'host_monotonic_ns':1000000000,'engine_version':337,'generation':1,'sim_time_s':3.,'joints':{x:{'q':.2,'qvel':0.} for x in ('left','right')}}
        def refresh():
            with mock.patch('builtins.open',return_value=io.BytesIO((json.dumps(row)+'\n').encode())):n.refresh_native()
        refresh();self.assertTrue(n.arm_fresh());self.assertEqual(n.arm['left'],.2)
        clock.ns=1300000000
        with self.assertRaisesRegex(RuntimeError,'STALE'):refresh()
        self.assertFalse(n.arm_fresh())
        clock.ns=1100000000;row['generation']=2
        with self.assertRaisesRegex(RuntimeError,'RESET'):refresh()
        self.assertEqual(n.native_generation,1)

if __name__=='__main__':main()
