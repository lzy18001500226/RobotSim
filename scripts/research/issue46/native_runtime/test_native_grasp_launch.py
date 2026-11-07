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
    def test_native_graphics_identity_requires_the_actual_physics_pid_and_selected_adapter(self):
        import shlex
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gl.json';record={'classification':'READ_ONLY_ACTUAL_NATIVE_GL_CONTEXT','pid':9,'renderer':'D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)'};path.write_text(json.dumps(record))
            self.assertEqual(audit.require_graphics_adapter(path,'NVIDIA',expected_pid=9)['pid'],9)
            with self.assertRaisesRegex(RuntimeError,'NOT_CONFIRMED'):audit.require_graphics_adapter(path,'NVIDIA',expected_pid=10)
            with self.assertRaisesRegex(RuntimeError,'NOT_CONFIRMED'):audit.require_graphics_adapter(path,'Intel',expected_pid=9)
        value="GPU name; echo injected"
        self.assertIn('MESA_D3D12_DEFAULT_ADAPTER_NAME='+value,shlex.split(audit.launch_prefix(value)))
        self.assertIn('LIBGL_ALWAYS_SOFTWARE=0',shlex.split(audit.launch_prefix('NVIDIA')))
        self.assertNotIn('MESA_LOADER_DRIVER_OVERRIDE=d3d12',shlex.split(audit.launch_prefix()))
        self.assertEqual(audit.graphics_mount_options(None,None),[])
        with self.assertRaises(ValueError):audit.graphics_mount_options('NVIDIA',None)
        with mock.patch.object(audit.Path,'exists',return_value=True):
            options=audit.graphics_mount_options('NVIDIA',':0')
        self.assertIn('/dev/dxg',options);self.assertIn('type=bind,source=/usr/lib/wsl,target=/usr/lib/wsl,readonly',options)
        self.assertNotIn('--privileged',options)

    def test_manual_terminal_review_is_opt_in_and_after_lift_evaluation(self):
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        branch=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.If) and 'args.pause_on_claw_failure and' in ast.unparse(n.test) and 'result' in ast.unparse(n.test))
        condition=compile(ast.Expression(branch.test),'review','eval')
        result={'gates':{'HAL_RIGHT_HAND_CLOSE':'PASS','HAL_PHYSICAL_LIFT':'PASS'}}
        self.assertFalse(eval(condition,{'args':NS(pause_on_claw_failure=False,physical_lift_test=True),'result':result}))
        self.assertTrue(eval(condition,{'args':NS(pause_on_claw_failure=True,physical_lift_test=True),'result':result}))
        self.assertFalse(eval(condition,{'args':NS(pause_on_claw_failure=True,physical_lift_test=False),'result':result}))
        result['gates']['HAL_RIGHT_HAND_CLOSE']='FAIL'
        self.assertTrue(eval(condition,{'args':NS(pause_on_claw_failure=True,physical_lift_test=False),'result':result}))
        self.assertLess(source.index('metrics=native_grasp_audit.evaluate_unsupported_lift'),source.index('before_pause=json.loads'))
        self.assertIn('PHYSICAL_LIFT_PASS_REVIEW',source)

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

    def test_damping_diagnostic_changes_only_right_velocity_bias_and_selected_loss(self):
        import mujoco
        import numpy as np
        from scripts.research.issue46.native_runtime.claw_parameter_diagnostic import change_friction,JOINTS
        xml='<mujoco><worldbody><body>'+''.join('<joint name="'+n+'" axis="0 0 1" frictionloss=".3"/>' for n in JOINTS)+'<geom size=".01"/></body></worldbody><actuator><position name="motor_R_claw_joint" joint="R_claw_joint" kp="2" kv=".05" forcelimited="true" forcerange="-2 2"/></actuator></mujoco>'
        a=mujoco.MjModel.from_xml_string(xml);b=mujoco.MjModel.from_xml_string(change_friction(xml,right_damping=1.27,noslip_iterations=2,elliptic_grip_solver=True).decode())
        self.assertAlmostEqual(-b.actuator_biasprm[0,2],1.27)
        self.assertEqual(b.opt.noslip_iterations,2)
        self.assertEqual(a.opt.noslip_iterations,0)
        self.assertEqual(int(b.opt.cone),1);self.assertEqual(b.opt.impratio,10.)
        for name in dir(a):
            x=getattr(a,name);y=getattr(b,name)
            if not isinstance(x,np.ndarray):continue
            expected=x.copy()
            if name=='actuator_biasprm':expected[0,2]=-1.27
            if name=='dof_frictionloss':expected[:]=0.
            np.testing.assert_array_equal(expected,y,err_msg=name)
        with self.assertRaises(ValueError):change_friction(xml,right_damping=2.01)
        with self.assertRaises(ValueError):change_friction(xml,noslip_iterations=3)

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
        rows.extend({'host_monotonic_ns':int(t*1e9),'sim_time_s':t,'contacts':contacts} for t in (1.54,1.58,1.62,1.66,1.7,1.74))
        with mock.patch('builtins.open',return_value=io.BytesIO(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())):
            self.assertTrue(ns['native_bilateral_contact'](.5)['pass'])

    def test_lift_stage_never_runs_without_static_and_measured_contact_gates(self):
        code=generated()
        self.assertIn('LIFT_CONTRACT=None',code)
        self.assertLess(code.index('LIFT_GATED_NO_SUSTAINED_BILATERAL_CONTACT'),code.index("n.send(goal,3.,'LIFT'"))
        self.assertIn('BILATERAL_CONTACT_LOST_DURING_LIFT',code)
        self.assertIn("'bottle_center_world_m':grasped_position",code)
        self.assertIn("'preclose_bottle_center_world_m':phase['bottle_center_world_m']",code)
        self.assertIn("n.send(goal,1.5,'LIFT_HOLD'",code)
        self.assertIn('probe_deadline=time.monotonic()+2.',code)
        self.assertIn('if probe_rise>=.005 and table_force<=.1:break',code)
        self.assertIn("if probe_rise<.005 or table_force>.1:raise RuntimeError('LOAD_TRANSFER_PROBE_FAILED')",code)
        self.assertIn('max(-.002,min(.002,.05*correction))',code)
        self.assertNotIn('qpos[',code)
        self.assertIn('self.grip_command=HAND_OPEN+u*(probe-HAND_OPEN)',code)
        self.assertIn("self.grip_command=force_guided_close_step",code)
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        self.assertIn('0.0 if args.physical_lift_test else',source)
        self.assertIn('and abs(raw_angle-expected_raw)<=.03',code)

    def test_force_guided_closure_is_bounded_and_rejects_invalid_feedback(self):
        tree=ast.parse(generated());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='force_guided_close_step')
        ns={'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),'force','exec'),ns);step=ns[fn.name]
        self.assertAlmostEqual(step(.39,0.,.39,{'left':.4,'right':.4},1.4715,.02),.3892)
        self.assertAlmostEqual(step(.3,0.,.39,{'left':1.6,'right':1.6},1.4715,.02),.3)
        self.assertGreater(step(.3,0.,.39,{'left':5.,'right':5.},1.4715,.02),.3)
        self.assertEqual(step(0.,0.,.39,{'left':0.,'right':0.},1.4715,3.),0.)
        with self.assertRaisesRegex(RuntimeError,'INVALID'):step(.3,0.,.39,{'left':float('nan'),'right':1.},1.4715,.02)

    def test_reused_source_proxy_is_hash_locked_and_limited_to_two_pad_bodies(self):
        source=Path(__file__).with_name('claw_parameter_diagnostic.py').read_text();tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reuse_source_pad_proxy')
        code=ast.unparse(fn)
        self.assertIn("('narrow3_Link', 'wide3_Link')",code)
        self.assertIn('7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a',code)
        self.assertIn("digest(path) != part['sha256']",code)
        self.assertIn('proxy comparison changed physical array',code)
        self.assertIn('visual geometry or exclusions',code)
        self.assertNotIn('run_coacd',code)

    def test_proxy_scene_identity_requires_exact_ancestry_and_unchanged_targets(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import candidate_scene_matches_overlay,sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=root/'proxy.xml';scene.write_text('<mujoco/>');control=root/'overlay.json';control.write_text('{}')
            original={'scene':{'path':'original.xml','sha256':'original-scene'},'target':[1.,2.,3.],'gates':{'static':'PASS'}}
            source=root/'original.json';source.write_text(json.dumps(original));world={'generated':{'host_scene':{'sha256':'original-scene'}}}
            generated={'path':str(scene),'sha256':sha256(scene)}
            overlay={'generated':{'host_scene':generated},'source_pad_collision_comparison':{'manifest_sha256':'7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a','scope':['R_hand_narrow3_Link','R_hand_wide3_Link']}}
            candidate={**original,'scene':generated,'source_pad_collision_comparison':{'source_candidate':str(source),'source_candidate_sha256':sha256(source),'overlay':str(control),'overlay_sha256':sha256(control)}}
            self.assertTrue(candidate_scene_matches_overlay(candidate,world,overlay,control))
            candidate['target']=[9.,9.,9.]
            self.assertFalse(candidate_scene_matches_overlay(candidate,world,overlay,control))
            candidate['target']=original['target'];overlay['source_pad_collision_comparison']['scope'].append('right_wrist_roll_link')
            self.assertFalse(candidate_scene_matches_overlay(candidate,world,overlay,control))


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
        self.assertIn('tracking_task(row, reference, phase)',code)
        task=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tracking_task')
        self.assertIn('BOTTLE_MOVED_BEFORE_CLOSURE',ast.unparse(task))
        self.assertIn('BOTTLE_MOVED_DURING_CLOSURE',ast.unparse(task))


    def test_recorded_review_cannot_invoke_runtime_physics(self):
        tree=ast.parse(Path(__file__).with_name('native_grasp_audit.py').read_text())
        for name in ('apply_recorded_native_frame','review_native_capture'):
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            code=ast.unparse(fn)
            self.assertNotIn('mj_step(',code)
            self.assertNotIn('subprocess.',code)
        adapter=ast.parse(Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_native_trial.py').read_text())
        fn=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        review=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.review_captured_run')
        self.assertIsInstance(review.body[-1],ast.Return)


    def test_visible_wslg_x11_has_no_headless_tmp_dependency(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import headless_x11_mounts
        self.assertEqual(headless_x11_mounts(None,None,None,interactive=True),[])
        with self.assertRaises(ValueError):headless_x11_mounts(None,None,None,interactive=False)
        adapter=Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_native_trial.py').read_text()
        self.assertNotIn('/tmp/robotsim-xvfb-pkg',adapter)


    def test_execution_summary_survives_trailing_exception_context(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import decode_arm_execution_output
        rows=[{'native_tcp_arrival':{'pass':True}},{'motion_stability':{'pass':True},'physical_bilateral_contact':{'pass':True},'closure_issued':True},{'execution_exception_context':True,'closure_issued':True,'close_publish_call_count':100}]
        result=decode_arm_execution_output(0,'\n'.join(json.dumps(r) for r in rows))
        self.assertTrue(result['native_tcp_arrival']['pass'])
        self.assertTrue(result['physical_bilateral_contact']['pass'])
        self.assertEqual(result['child_returncode'],0)
        failure=decode_arm_execution_output(1,json.dumps(rows[-1])+'\nTraceback: interrupted')
        self.assertTrue(failure['closure_issued']);self.assertEqual(failure['child_returncode'],1)
        self.assertNotIn('physical_bilateral_contact',failure)


    def test_fixed_checkpoint_does_not_qualify_navigation(self):
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        self.assertIn('args.use_official_cmd_vel_bridge and not args.fixed_reviewed_checkpoint',source)
        self.assertIn("'navigation':'NOT_RUN'",source)
        self.assertIn('NATIVE_KEYFRAME_INITIAL_POSE_UNCONFIRMED',source)
        self.assertIn('NATIVE_FIXED_STANCE_STATE_HANDOFF',source)
        compile(source,'native.py','exec')


    def test_model_loading_budget_is_separate_from_active_feedback_age(self):
        source=Path(__file__).parents[1].joinpath('grasp/stl_approved_circle_native_trial.py').read_text()
        self.assertIn('readiness_started+25.',source)
        self.assertIn('if 0.<=age<=.20:ready=True',source)
        self.assertIn('UNSAFE_POSTURE_DURING_TRACKER_INITIALIZATION',source)
        self.assertIn('require_fresh_sample',source)


    def test_keyframe_selection_never_clicks_without_verified_focus(self):
        source=ast.parse(Path(__file__).with_name('native_mujoco_reset.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='X11')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='select_native_keyframe_zero_once')
        fn.returns=None
        for arg in fn.args.args:arg.annotation=None
        scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'selection','exec'),scope)
        node=NS(_target_identity=lambda *a:{'width':1706,'height':1066},activate_window=lambda *a,**k:{'status':'FOCUS_NOT_CONFIRMED'},focus_window_direct=lambda w:{'status':'FOCUS_NOT_CONFIRMED'})
        result=scope[fn.name](node,123,28,'MuJoCo : x2t2.5')
        self.assertEqual(result['mouse_button_events_sent'],0)
        self.assertEqual(result['status'],'KEYFRAME_INDEX_INPUT_NOT_DISPATCHED')

    def test_key_slider_consumption_precedes_any_second_input(self):
        source=Path(__file__).with_name('xvfb_native_reset_stand.py').read_text()
        block=source[source.index('        if args.fixed_reviewed_checkpoint:\n            identity='):source.index('        else:\n            reset_before=')]
        self.assertIn('NATIVE_KEY_SLIDER_AUTO_LOAD',block)
        self.assertNotIn('click-native-load-key',block)
        self.assertIn("observed['generation']==before_key['generation']+1 and boundary['pass']",block)

    def test_reset_boundary_is_distinct_from_later_pose_and_never_accepts_unchanged_generation(self):
        wanted={'position_m':[.85,-1.49,.65],'quaternion_wxyz':[1.,0.,0.,0.]}
        rows=[{'host_monotonic_ns':1000000000,'sim_time_s':.001,'generation':1,'engine_version':337,'bodies':{'pelvis':wanted}},{'host_monotonic_ns':2000000000,'sim_time_s':1.,'generation':1,'engine_version':337,'bodies':{'pelvis':{'position_m':[.88,-1.52,.65],'quaternion_wxyz':[1.,0.,0.,0.]}}}]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            result=audit.keyframe_initialization_evidence(p,wanted,generation=1,after_ns=900000000)
            self.assertTrue(result['pass']);self.assertEqual(result['source_sim_time_s'],.001)
            self.assertFalse(audit.keyframe_initialization_evidence(p,wanted,generation=2,after_ns=900000000)['pass'])
            self.assertFalse(audit.keyframe_initialization_evidence(p,wanted,generation=1,after_ns=1100000000)['pass'])

    def test_final_feedback_refinement_is_bounded_and_cannot_close_on_failed_arrival(self):
        code=generated()
        self.assertIn('for refinement in range(8):',code)
        self.assertIn('max(-.03,min(.03,correction))',code)
        self.assertIn('max(limits[0]+.01,min(limits[1]-.01,command))',code)
        self.assertLess(code.index('PREGRASP_FEEDBACK_REFINEMENT'),code.index('n.closing_started=True'))
        self.assertIn('preclose_error<=.08 and preclose_waist_error<=.05 and arrival["pass"]',code)

    def test_fixed_stance_initialization_precedes_approach(self):
        code=Path(__file__).with_name('claw_parameter_diagnostic.py').read_text()
        self.assertIn("r['stage']=='HAND_OPEN_PREGRASP'",code)
        self.assertNotIn("abs(r['sim_time_s']-91.021)",code)

    def test_measured_object_pose_is_in_the_same_snapshot_as_robot(self):
        row={'bodies':{'pelvis':{'position_m':[0.,0.,.65],'rotation':[1.,0.,0.,0.,1.,0.,0.,0.,1.]},'test_cylinder':{'position_m':[1.2,-1.4,.7],'quaternion_wxyz':[1.,0.,0.,0.]}},'joints':{},'host_monotonic_ns':10,'sim_time_s':2.,'generation':1}
        state=audit.planning_state_from_native(row,{g:{} for g in ('arm_positions_rad','leg_positions_rad','waist_positions_rad')})
        self.assertEqual(state['object_pose_world']['xyz_m'],[1.2,-1.4,.7])
        row['bodies']['test_cylinder']['position_m'][0]=9.
        self.assertEqual(state['object_pose_world']['xyz_m'][0],1.2)

    def test_lift_acceptance_requires_actual_continuous_unsupported_hold(self):
        contacts=[{'body1':'test_cylinder','body2':n,'local_force_torque':[4.,0.,0.,0.,0.,0.],'signed_bottle_world_z_force_n':3.} for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')]
        rows=[{'host_monotonic_ns':1000000000+i*20000000,'sim_time_s':1.+i*.02,'generation':1,'bodies':{'test_cylinder':{'position_m':[1.,2.,.76]}},'contacts':contacts} for i in range(76)]
        def evaluate(r):return audit.evaluate_unsupported_lift(r,initial_bottle_z_m=.7,hold_start_ns=1000000000,hold_end_ns=2500000000)
        self.assertTrue(evaluate(rows)['pass'])
        self.assertFalse(evaluate(rows[:20])['pass'])
        import copy
        supported=copy.deepcopy(rows)
        supported[-1]['contacts'].append({'body1':'test_cylinder','body2':'raicom_table_table','local_force_torque':[1.,0.,0.,0.,0.,0.]})
        self.assertFalse(evaluate(supported)['pass'])
        one_sided=copy.deepcopy(rows)
        for r in one_sided:r['contacts']=r['contacts'][:1]
        self.assertFalse(evaluate(one_sided)['pass'])
        dropped=copy.deepcopy(rows);dropped[-1]['bodies']['test_cylinder']['position_m'][2]=.7
        self.assertFalse(evaluate(dropped)['pass'])


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

class NativeStartupReadinessTests(TestCase):
    def test_stale_discovery_needs_two_later_advancing_fresh_samples(self):
        clock=NS(t=0.)
        row=dict(pass_result=True,age_s=.01,engine_version=337,sim_time_s=1.,generation=0,process_pid=30)
        with mock.patch.object(audit,"require_fresh_sample",side_effect=[RuntimeError("NATIVE_OBSERVER_STALE_SAMPLE"),row,dict(row,sim_time_s=1.02)]),mock.patch("time.monotonic",lambda:clock.t),mock.patch("time.sleep",lambda dt:setattr(clock,"t",clock.t+dt)):
            result=audit.await_native_startup_ready(Path("/input"),timeout_s=.1)
        self.assertEqual(result["advancing_fresh_samples"],2)
        self.assertAlmostEqual(result["startup_wait_s"],.04)

    def test_missing_bodies_never_turn_into_a_longer_startup_wait(self):
        with mock.patch.object(audit,"require_fresh_sample",side_effect=RuntimeError("NATIVE_OBSERVER_MISSING_REQUIRED_BODIES")) as check:
            with self.assertRaisesRegex(RuntimeError,"MISSING_REQUIRED_BODIES"):
                audit.await_native_startup_ready(Path("/input"))
        self.assertEqual(check.call_count,1)

    def test_stale_or_frozen_startup_cannot_pass(self):
        clock=NS(t=0.)
        row=dict(age_s=.01,engine_version=337,sim_time_s=1.,generation=0,process_pid=30)
        for response in (row,RuntimeError("NATIVE_OBSERVER_STALE_SAMPLE")):
            clock.t=0.
            with mock.patch.object(audit,"require_fresh_sample",side_effect=(response if isinstance(response,Exception) else None),return_value=row),mock.patch("time.monotonic",lambda:clock.t),mock.patch("time.sleep",lambda dt:setattr(clock,"t",clock.t+dt)):
                with self.assertRaisesRegex(RuntimeError,"STARTUP_NOT_READY"):
                    audit.await_native_startup_ready(Path("/input"),timeout_s=.1)

    def test_generation_or_pid_change_invalidates_startup(self):
        row=dict(age_s=.01,engine_version=337,sim_time_s=1.,generation=0,process_pid=30)
        for changed in (dict(row,sim_time_s=1.02,generation=1),dict(row,sim_time_s=1.02,process_pid=31)):
            with mock.patch.object(audit,"require_fresh_sample",side_effect=[row,changed]):
                with self.assertRaisesRegex(RuntimeError,"IDENTITY_OR_GENERATION_CHANGED"):
                    audit.await_native_startup_ready(Path("/input"))


class NativeArmRelayTests(TestCase):
    def test_config_only_restores_and_redirects_the_original_arm_transport(self):
        import copy
        from scripts.research.issue46.native_runtime.stl_hand_startup import build_native_arm_relay_config
        source={"MotionControlModule":{"PublisherManager":{"publisher_cfg_list":[
            {"name":"arm_joint_command","enable":True,"topic":"/aima/hal/joint/arm/command"},
            {"name":"hand_joint_command","enable":True,"topic":"/aima/hal/joint/hand/command"},
            {"name":"waist_joint_command","enable":True,"topic":"/aima/hal/joint/waist/command"},
            {"name":"leg_joint_command","enable":True,"topic":"/aima/hal/joint/leg/command"}]}},
            "aimrt":{"channel":{"pub_topics_options":[{"topic_name":"/aima/hal/joint/arm/command","enable_backends":["ros2"]}]}},
            "frozen_policy_and_gains":{"action_nums":29,"gain":50.}}
        original=copy.deepcopy(source);expected=copy.deepcopy(source)
        expected["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"][0]["topic"]="/robotsim/issue46/mc/arm/command"
        expected["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"][1]["enable"]=False
        expected["aimrt"]["channel"]["pub_topics_options"][0]["topic_name"]="/robotsim/issue46/mc/arm/command"
        self.assertEqual(build_native_arm_relay_config(source),expected)
        self.assertEqual(source,original)

    def relay_method(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import arm_bootstrap_hold_script
        from scripts.research.issue46.native_runtime.native_base_micro_probe import observed_clock_step
        source=ast.parse(arm_bootstrap_hold_script(__import__("scripts.research.issue46.native_runtime.stl_hand_startup",fromlist=["load_animation_player_arm_pd_parameters"]).load_animation_player_arm_pd_parameters(),relay_arms=True))
        hold=next(n for n in source.body if isinstance(n,ast.ClassDef))
        method=next(n for n in hold.body if isinstance(n,ast.FunctionDef) and n.name=="on_mc_arm")
        hold.bases=[];hold.body=[method]
        arm=tuple("joint"+str(i) for i in range(14));clock=NS(now=10.);sent=[]
        ns=dict(ARM=arm,ARM_LIMITS={n:(-1.,1.,2.) for n in arm},math=math,
                time=NS(monotonic=lambda:clock.now,monotonic_ns=lambda:int(clock.now*1e9),time_ns=lambda:int((clock.now-9.)*1e9)),
                observed_clock_step=observed_clock_step)
        exec(compile(ast.Module(body=[hold],type_ignores=[]),"relay","exec"),ns)
        n=ns[hold.name]();n.clock_offset=None;n.arm_source_stamp=0;n.arm_receipt=None
        n.arm_relay_count=0;n.positive_count=0;n.owner_verified=True;n.verify_owner=lambda:None
        n.last_gain_positive=False;n.state={name:.25 for name in arm};n.hold_target={name:.2 for name in arm}
        n.pub=NS(publish=sent.append)
        msg=NS(header=NS(stamp=NS(sec=1,nanosec=5)),joints=[NS(name=name,position=.2,velocity=.1,effort=.3,stiffness=50.,damping=3.) for name in arm])
        return n,msg,sent,clock

    def test_native_source_preserves_parent_ready_result_contract(self):
        from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import arm_bootstrap_hold_script
        code=arm_bootstrap_hold_script(__import__("scripts.research.issue46.native_runtime.stl_hand_startup",fromlist=["load_animation_player_arm_pd_parameters"]).load_animation_player_arm_pd_parameters(),relay_arms=True)
        tree=ast.parse(code)
        dictionaries=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="dict" and any(k.arg=="ready" for k in n.keywords)]
        self.assertEqual(len(dictionaries),1)
        fields={k.arg for k in dictionaries[0].keywords}
        self.assertTrue({"ready","subscription_count","arm_positions_rad","transport_only"}<=fields)
        self.assertNotIn("TARGET=(.4",code)

    def test_active_relay_forwards_the_original_message_and_rejects_duplicates(self):
        n,msg,sent,clock=self.relay_method()
        n.on_mc_arm(msg)
        self.assertIs(sent[0],msg);self.assertEqual(n.arm_relay_count,1)
        clock.now=10.1;n.on_mc_arm(msg)
        self.assertEqual(n.arm_receipt,10.);self.assertEqual(n.arm_relay_count,1)

    def test_inactive_stand_commands_are_valid_but_never_count_as_active_gains(self):
        n,msg,sent,clock=self.relay_method()
        for j in msg.joints:j.stiffness=j.damping=j.effort=0.
        n.on_mc_arm(msg)
        self.assertEqual(sent,[]);self.assertEqual(n.positive_count,0);self.assertFalse(n.last_gain_positive)
        self.assertEqual(n.arm_receipt,10.)

    def test_unknown_owner_and_invalid_source_targets_cannot_actuate(self):
        n,msg,sent,clock=self.relay_method();n.owner_verified=False;n.on_mc_arm(msg)
        self.assertEqual(sent,[])
        n,msg,sent,clock=self.relay_method();msg.joints[0].position=2.
        with self.assertRaisesRegex(RuntimeError,"SOURCE_LIMIT"):n.on_mc_arm(msg)
        self.assertEqual(sent,[])



class VerifiedPointerControlsTests(TestCase):
    def node(self, *, pixels_good=True, ancestry=True, mask=16, moved=False):
        import ctypes
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        n=ui.X11.__new__(ui.X11);n.display=1;n.root=1107
        identity=dict(window_id=123,pid=28,title="MuJoCo : x2t2.5",x=38,y=59,width=1706,height=1074,viewable=True)
        n._target_identity=lambda *a:identity.copy()
        n._input_focus=lambda:0
        n.activate_window=lambda *a,**k:(_ for _ in ()).throw(AssertionError("keyboard activation must not be used"))
        n.focus_window_direct=n.activate_window
        n.native_button_pixels=lambda window,points:({k:(76,51,76) if k.startswith("key") else (153,102,102) if k.endswith("border") else (26,26,26) for k in points} if pixels_good else {k:(0,0,0) for k in points})
        n.pointer_root_child_contains_target=lambda window:dict(verified=ancestry)
        calls=[];translation=[0]
        def translate(display,window,root,x,y,rx,ry,child):
            translation[0]+=1
            rx._obj.value=x+100+(10 if moved and translation[0]>1 else 0);ry._obj.value=y+200
            return 1
        def pointer(display,window,root,child,rx,ry,wx,wy,bits):
            wx._obj.value=145;wy._obj.value=517;rx._obj.value=245;ry._obj.value=717;bits._obj.value=mask
            return 1
        n.lib=NS(XTranslateCoordinates=translate,XQueryPointer=pointer,XFlush=lambda *a:None,XSync=lambda *a:None)
        n.xtst=NS(XTestFakeMotionEvent=lambda *a:1,XTestFakeButtonEvent=lambda d,b,pressed,t:(calls.append((b,pressed)) or 1))
        return n,calls

    def test_no_keyboard_focus_can_use_only_the_strict_mouse_contract(self):
        n,calls=self.node()
        event=n.select_native_keyframe_zero_once(123,28,"MuJoCo : x2t2.5",verified_pointer_only=True)
        self.assertEqual(event["status"],"KEYFRAME_INDEX_SELECTION_INPUT_DISPATCHED")
        self.assertEqual(event["keyboard_focus_observed"],0)
        self.assertEqual(event["keyboard_events_sent"],0)
        self.assertEqual(calls,[(1,1),(1,0)])
        self.assertFalse(event["selection_authoritatively_observed"])

    def test_mouse_path_rejects_wrong_pixels_occlusion_held_button_and_motion(self):
        for settings in (dict(pixels_good=False),dict(ancestry=False),dict(mask=256),dict(moved=True)):
            n,calls=self.node(**settings)
            event=n.select_native_keyframe_zero_once(123,28,"MuJoCo : x2t2.5",verified_pointer_only=True)
            self.assertEqual(calls,[])
            self.assertEqual(event["mouse_button_events_sent"],0)

    def test_mouse_path_never_clicks_a_lost_identity(self):
        n,calls=self.node();n._target_identity=lambda *a:None
        event=n.select_native_keyframe_zero_once(123,28,"MuJoCo : x2t2.5",verified_pointer_only=True)
        self.assertEqual(calls,[]);self.assertEqual(event["mouse_button_events_sent"],0)

    def test_archived_live_ui_signatures_and_negative_controls(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        key={k:(76,51,76) if k.startswith("key") else (153,102,102) if k.endswith("border") else (25,25,25) for k in ui.NATIVE_KEY_SLIDER_PIXEL_POINTS}
        self.assertTrue(ui.native_key_slider_signature_matches(key))
        self.assertFalse(ui.native_key_slider_signature_matches({k:(25,25,25) for k in key}))
        run=dict(left_interior=(25,25,25),right_interior=(102,153,102),border_top=(102,153,102),border_left=(102,153,102))
        paused=dict(run,left_interior=(102,153,102),right_interior=(25,25,25))
        self.assertTrue(ui.native_pause_signature_matches(run));self.assertTrue(ui.native_pause_signature_matches(paused))
        self.assertFalse(ui.native_pause_signature_matches(dict(run,left_interior=(102,153,102))))


class NativeUiReceiptTests(TestCase):
    node = VerifiedPointerControlsTests.node
    def test_release_waits_for_the_actual_key_press_without_a_second_click(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        n, calls = self.node()
        def acknowledged(*args, **kwargs):
            self.assertEqual(calls, [(1, 1)])
            self.assertEqual(kwargs["expected_pid"], 28)
            self.assertEqual(kwargs["xy"], (145, 557))
            self.assertEqual(kwargs["item_name"], "Key")
            return {"status": "NATIVE_UI_PRESS_CONFIRMED"}
        with mock.patch.object(ui, "wait_for_native_ui_receipt", side_effect=acknowledged):
            event = n.select_native_keyframe_zero_once(123, 28, "MuJoCo : x2t2.5",
                verified_pointer_only=True, native_ui_events=Path("/verified/receipt"))
        self.assertEqual(calls, [(1, 1), (1, 0)])
        self.assertEqual(event["native_ui_receipt"]["status"], "NATIVE_UI_PRESS_CONFIRMED")
        self.assertFalse(event["selection_authoritatively_observed"])

    def test_reset_uses_the_same_actual_native_receipt_contract(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        n,calls=self.node()
        n.native_reset_button_pixels=lambda w:{k:((26,26,26) if k=="center" else (153,102,102)) for k in ui.NATIVE_RESET_BUTTON_PIXEL_POINTS}
        def pointer(display,window,root,child,rx,ry,wx,wy,bits):
            wx._obj.value=68;wy._obj.value=456;rx._obj.value=168;ry._obj.value=656;bits._obj.value=16
            return 1
        n.lib.XQueryPointer=pointer
        def acknowledge(*args,**kw):
            self.assertEqual(calls,[(1,1)])
            self.assertEqual((kw["item_name"],kw["section_id"],kw["item_id"],kw["xy"]),("Reset",2,1,(68,618)))
            return {"status":"NATIVE_UI_PRESS_CONFIRMED"}
        with mock.patch.object(ui,"wait_for_native_ui_receipt",side_effect=acknowledge):
            event=n.click_native_reset_button_once(123,28,"MuJoCo : x2t2.5",
                require_pointer_ancestry=True,native_ui_events=Path("/verified/ui"),expected_ui_item=("Reset",2,1))
        self.assertEqual(event["status"],"RESET_BUTTON_CLICKED")
        self.assertEqual(calls,[(1,1),(1,0)])

    def test_receipt_timeout_releases_and_fails_closed_without_retry(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        n, calls = self.node()
        with mock.patch.object(ui, "wait_for_native_ui_receipt",
                return_value={"status": "NATIVE_UI_PRESS_UNCONFIRMED"}):
            event = n.select_native_keyframe_zero_once(123, 28, "MuJoCo : x2t2.5",
                verified_pointer_only=True, native_ui_events=Path("/verified/receipt"))
        self.assertEqual(calls, [(1, 1), (1, 0)])
        self.assertEqual(event["status"], "NATIVE_UI_PRESS_UNCONFIRMED")

    def test_exception_still_releases_mouse(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        n, calls = self.node()
        with mock.patch.object(ui, "wait_for_native_ui_receipt", side_effect=RuntimeError("reader")):
            with self.assertRaisesRegex(RuntimeError, "reader"):
                n.select_native_keyframe_zero_once(123, 28, "MuJoCo : x2t2.5",
                    verified_pointer_only=True, native_ui_events=Path("/verified/receipt"))
        self.assertEqual(calls, [(1, 1), (1, 0)])

    def test_native_receipt_requires_pid_time_control_and_real_coordinates(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        row=dict(kind="native_mjui_event",pid=28,host_ns=101,event_type=2,
            returned_item="Key",section_id=2,item_id=5,xy=[145,549])
        kw=dict(expected_pid=28,after_ns=100,item_name="Key",section_id=2,item_id=5,xy=(145,549))
        self.assertTrue(ui.native_ui_receipt_matches(row, **kw))
        for change in (dict(pid=29),dict(host_ns=100),dict(event_type=3),
                dict(returned_item="Reset"),dict(section_id=1),dict(item_id=6),
                dict(xy=[3358,713]),dict(xy=[145,517])):
            self.assertFalse(ui.native_ui_receipt_matches(dict(row,**change),**kw))
        self.assertFalse(ui.native_ui_receipt_matches([],**kw))

    def test_missing_and_partial_receipts_never_count_as_application_receipt(self):
        from scripts.research.issue46.native_runtime import native_mujoco_reset as ui
        kw=dict(expected_pid=28,after_ns=100,item_name="Key",section_id=2,item_id=5,xy=(145,549),timeout_s=0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"events"
            self.assertEqual(ui.wait_for_native_ui_receipt(path,**kw)["status"],"NATIVE_UI_PRESS_UNCONFIRMED")
            path.write_text('{"kind":')
            self.assertEqual(ui.wait_for_native_ui_receipt(path,**kw)["status"],"NATIVE_UI_PRESS_UNCONFIRMED")
            path.write_text(json.dumps(dict(kind="native_mjui_event",pid=28,host_ns=101,event_type=2,
                returned_item="Key",section_id=2,item_id=5,xy=[145,549]))+'\n')
            self.assertEqual(ui.wait_for_native_ui_receipt(path,**kw)["status"],"NATIVE_UI_PRESS_CONFIRMED")


class WslgForegroundTests(TestCase):
    def test_foreground_is_verified_and_never_submits_keyboard_or_mouse(self):
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
        calls=[]
        def fake_run(command,**kwargs):
            calls.append(command)
            return NS(returncode=0,stdout="C:\\verified-helper.ps1\n" if command[0]=="wslpath" else json.dumps(dict(status="FOREGROUND_VERIFIED",keyboardEvents=0,mouseEvents=0)))
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(Path,"is_file",return_value=True),mock.patch.object(parent,"run",side_effect=fake_run),mock.patch.object(parent.pg,"write_new_text"):
            result=parent.activate_wslg_native_window({"title":"MuJoCo : x2t2.5"},Path(folder),label="test")
        self.assertEqual(result["status"],"FOREGROUND_VERIFIED")
        self.assertEqual(calls[1][-2:],["-ExpectedTitle","MuJoCo : x2t2.5"])
        self.assertIn("Hidden",calls[1])
        self.assertNotIn("SendKeys"," ".join(calls[1]))

    def test_pointer_preflight_distinguishes_stale_state_from_user_holding(self):
        from scripts.research.issue46.native_runtime.wslg_input_preflight import classify
        up=dict(LeftDown=False,MiddleDown=False,RightDown=False)
        self.assertEqual(classify(up,16),"PASS")
        self.assertEqual(classify(up,1040),"STALE_X11_POINTER_BUTTON_STATE")
        self.assertEqual(classify(dict(up,RightDown=True),1040),"RELEASE_PHYSICAL_MOUSE_BUTTONS_BEFORE_AUTOMATION")
        self.assertEqual(classify({},0),"WINDOWS_BUTTON_STATE_UNVERIFIED")

    def test_false_activation_stops_before_any_x11_input(self):
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
        responses=[NS(returncode=0,stdout="C:\\helper.ps1"),NS(returncode=2,stdout='{"status":"FOREGROUND_NOT_CONFIRMED"}')]
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(Path,"is_file",return_value=True),mock.patch.object(parent,"run",side_effect=responses),mock.patch.object(parent.pg,"write_new_text"):
            with self.assertRaisesRegex(RuntimeError,"WSLG_WINDOWS_FOREGROUND_NOT_VERIFIED"):
                parent.activate_wslg_native_window({"title":"MuJoCo : x2t2.5"},Path(folder),label="test")


class GainCaptureReadinessTests(TestCase):
    def replay(self, delay, *, freeze_stamps=False):
        import ast,json,math,io,contextlib
        from types import SimpleNamespace as NS
        from scripts.research.issue46.native_runtime.stl_hand_startup import build_stand_command_capture_script
        tree=ast.parse(build_stand_command_capture_script())
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        clock=NS(t=0.)
        class Node:
            def __init__(self,*args):pass
            def create_subscription(self,*args):return object()
            def destroy_node(self):pass
        ns=dict(Node=Node,json=json,math=math,time=NS(monotonic=lambda:clock.t),
                QoSProfile=lambda **k:None,ReliabilityPolicy=NS(BEST_EFFORT=2),JointCommandArray=object)
        def spin(n,timeout_sec):
            clock.t+=.01
            if clock.t<delay:return
            stamp=int((1.+(delay if freeze_stamps else clock.t))*1e9)
            def packet(names):
                return NS(header=NS(stamp=NS(sec=stamp//1000000000,nanosec=stamp%1000000000)),
                          joints=[NS(name=name,position=0.,velocity=0.,effort=0.,stiffness=10.,damping=1.) for name in names])
            n.store("leg",packet(ns["LEG_ORDER"]));n.store("waist",packet(ns["WAIST_ORDER"]))
        ns["rclpy"]=NS(init=lambda:None,ok=lambda:True,spin_once=spin,shutdown=lambda:None)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):exec(compile(tree,"capture","exec"),ns)
        return json.loads(output.getvalue())

    def test_complete_measurement_starts_after_fresh_readiness(self):
        r=self.replay(1.80)
        self.assertGreaterEqual(r["readiness_delay_s"],1.79)
        self.assertGreaterEqual(r["capture_duration_s"],2.)
        self.assertGreater(r["window_counts"]["leg"],100)
        self.assertGreaterEqual(r["source_span_s"]["waist"],1.90)

    def test_missing_discovery_fails_closed_without_capture(self):
        with self.assertRaisesRegex(RuntimeError,"discovery/readiness"):
            self.replay(2.1)

    def test_duplicate_command_stamps_cannot_fake_gain_freshness(self):
        with self.assertRaisesRegex(RuntimeError,"stale or incomplete"):
            self.replay(.1,freeze_stamps=True)



class ResetDispatchHistoryTests(TestCase):
    def evaluate(self, rows, *, event_change=None, partial=False):
        import tempfile,json
        event={"status":"RESET_BUTTON_CLICKED","mouse_button_events_sent":2,
               "dispatch_monotonic_ns":1000000000,
               "native_ui_receipt":{"status":"NATIVE_UI_PRESS_CONFIRMED","event":{
                   "pid":7,"returned_item":"Reset","event_type":2,"host_ns":1050000000}}}
        if event_change:event.update(event_change)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"trace.jsonl"
            p.write_text("".join(json.dumps(r)+"\n" for r in rows)+('{"incomplete":' if partial else ""))
            return audit.native_reset_dispatch_evidence(p,{"event":event},expected_pid=7)
    @staticmethod
    def row(t,g,sim,pid=7):
        return dict(host_monotonic_ns=t,process_pid=pid,engine_version=337,generation=g,sim_time_s=sim)
    def test_short_prereset_clock_and_later_readback_are_valid(self):
        rows=[self.row(990000000,1,.001),self.row(1060000000,2,.001),self.row(1150000000,2,.1)]
        self.assertTrue(self.evaluate(rows,partial=True)["pass"])
    def test_duplicate_time_without_generation_reset_is_not_reset(self):
        self.assertFalse(self.evaluate([self.row(990000000,1,.001),self.row(1060000000,1,.001)])["pass"])
    def test_predispatch_reset_does_not_count(self):
        self.assertFalse(self.evaluate([self.row(900000000,0,5.),self.row(990000000,1,.001),self.row(1060000000,1,.06)])["pass"])
    def test_multiple_resets_fail_closed(self):
        r=self.evaluate([self.row(990000000,1,.1),self.row(1060000000,2,.001),self.row(1100000000,3,.001)])
        self.assertEqual(r["reason"],"UNEXPECTED_MULTIPLE_RESET_GENERATIONS")
    def test_stale_or_wrong_identity_baseline_fails(self):
        for rows in ([self.row(700000000,1,2.),self.row(1060000000,2,.001)],[self.row(990000000,1,.1,pid=9),self.row(1060000000,2,.001)]):
            self.assertFalse(self.evaluate(rows)["pass"])
    def test_unconfirmed_press_fails(self):
        self.assertFalse(self.evaluate([self.row(990000000,1,.1),self.row(1060000000,2,.001)],event_change={"mouse_button_events_sent":0})["pass"])

class PostLiftRetractTests(TestCase):
    def generated_retract(self, change=None):
        from scripts.research.issue46.native_runtime import stl_hand_startup as h
        names=[side+"_"+suffix+"_joint" for side in ("left","right") for suffix in ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_yaw","wrist_pitch","wrist_roll")]
        leg=[side+"_"+suffix+"_joint" for side in ("left","right") for suffix in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")]
        limits={n:[-2.,2.,5.] for n in names+leg+["waist_yaw_joint","waist_pitch_joint","waist_roll_joint"]}
        retract=dict(pass_result=True,offset_world_m=[-.15,-.15,0.],duration_s=4.,hold_s=1.5,source_motion_limits=limits)
        if change:retract.update(change)
        lift={"pass":True,"height_m":.06,"duration_s":3.,"hold_s":1.5,"retract_preflight":retract}
        return h.build_hal_bilateral_arm_plan_script({n:(30.,1.) for n in names},[0.]*7,[0.]*7,duration_s=6.,hold_s=1.,restore=False,right_hand_command=0.,hand_hold_s=2.,circle_contract={"dummy":True},track_native_circle=True,use_native_joint_feedback=True,lift_contract=lift)
    def test_actual_generated_retract_compiles_and_retains_force_and_source_guards(self):
        code=self.generated_retract();compile(code,"generated_retract","exec")
        tree=ast.parse(code)
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertIn("evaluate_unsupported_lift",functions)
        self.assertIn("physical_fault",functions)
        self.assertIn("source_motion_limits",code)
        self.assertLess(code.index("RETRACT_GATED_UNQUALIFIED_PHYSICAL_LIFT"),code.index("n.lift_phase['phase']='RETRACTING'"))
        self.assertIn("'RETRACT','RETRACT_HOLD'",code)
        self.assertIn("RETRACT_PHYSICAL_HOLD_UNQUALIFIED",code)
    def test_bad_retract_cannot_generate_runtime(self):
        for change in (dict(pass_result=False),dict(offset_world_m=[-.3,0.,0.]),dict(duration_s=8.),dict(source_motion_limits={})):
            with self.assertRaises(ValueError):self.generated_retract(change)
    def test_carry_lease_uses_joint_hold_and_keeps_velocity_out_of_grip_owner(self):
        code=self.generated_retract(dict(carry_lease_s=115.))
        compile(code,"carry_lease","exec")
        self.assertIn("MEASURED_LOAD_BEARING_CARRY_HOLD",code)
        self.assertIn("STANDING_BEFORE_CARRY",code)
        self.assertIn("carry_ready.json",code)
        self.assertIn("CARRY_LEASE_EXPIRED",code)
        self.assertNotIn("create_publisher(Twist",code)
        self.assertNotIn("McLocomotionVelocity",code)
        for bad in (0.,201.,float("inf")):
            with self.assertRaises(ValueError):self.generated_retract(dict(carry_lease_s=bad))
    def test_default_grasp_does_not_enable_retract(self):
        self.assertNotIn("RETRACT_GATED",generated())

class CompleteNativeTraceTests(TestCase):
    def test_partial_final_writer_record_is_not_parsed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"trace.jsonl";p.write_bytes(b'{"a":1}'+bytes([10])+b'{"b":')
            self.assertEqual(audit.read_complete_native_rows(p),[{"a":1}])
            p.write_bytes(b'{"a":1}'+bytes([10])+b'BROKEN'+bytes([10]))
            with self.assertRaises(ValueError):audit.read_complete_native_rows(p)

if __name__=='__main__':main()
