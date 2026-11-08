"""Frame/axis tests of the actual planning residual; no MuJoCo physics."""
import ast
from pathlib import Path
import unittest
import numpy as np


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse(Path(__file__).with_name('stl_approved_circle_ik.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('axis_grasp_error','circle_target','tracking_task','tracking_phase_is_fresh','bounded_tracking_velocity','qualify_grasp_motion_branches','grip_force_budget')]
        cls.ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'planner','exec'),cls.ns)

    def test_native_gain_cycle_has_scoped_startup_budget_without_freshness_relaxation(self):
        source=Path(__file__).with_name('stl_approved_circle_ik.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='track_native')
        assign=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cycle_budget' for t in n.targets))
        guard=fn.body[fn.body.index(assign)+1]
        code=compile(ast.Module(body=[assign,guard],type_ignores=[]),'cycle_budget','exec')
        for value in (0.,9.):
            ns={'reference':{'native_no_contact_gain_cycle_budget_s':value}}
            exec(code,ns);self.assertEqual(ns['cycle_budget'],value)
        for value in (1.,float('nan'),100.):
            with self.assertRaises(ValueError):exec(code,{'reference':{'native_no_contact_gain_cycle_budget_s':value}})
        self.assertIn("deadline=time.monotonic()+45.+cycle_budget",source)
        self.assertIn("if not 0<=age<=.20:raise RuntimeError('STALE_NATIVE_TRACKING_SOURCE')",source)
        self.assertIn("if (now-row['host_monotonic_ns'])*1e-9>.20",source)

    def test_carry_target_follows_base_xy_yaw_but_not_pelvis_sway_or_object_truth(self):
        from scipy.spatial.transform import Rotation
        reference=dict(grasp_height_above_center_m=.1)
        phase=dict(phase='CARRY_HOLD',native_arrival_pass=True,post_retract_physical_hold_pass=True,generation=1,carry_heading_xy_m=[.3,-.2],carry_world_z_m=.85,initial_object_in_hand_m=[0.,0.,.1],slip_reference='GRASP_SECTION',created_host_ns=100,carry_lease_s=100.)
        row=dict(generation=1,bodies={
            'pelvis':dict(position_m=[2.,3.,.65],rotation=Rotation.from_euler('xyz',[.07,-.06,.5]).as_matrix().ravel().tolist()),
            'R_omnipicker_base_link':dict(position_m=[1.,1.,.75],rotation=np.eye(3).ravel().tolist()),
            'test_cylinder':dict(position_m=[1.,1.,.75],rotation=np.eye(3).ravel().tolist())})
        point,axis=self.ns['tracking_task'](row,reference,phase)
        expected=np.array([2.,3.])+np.array([[np.cos(.5),-np.sin(.5)],[np.sin(.5),np.cos(.5)]])@np.array([.3,-.2])
        np.testing.assert_allclose(point,np.r_[expected,.85],atol=1e-12)
        np.testing.assert_array_equal(axis,[0.,0.,1.])
        # Change the independently checked object pose while preserving hand
        # relation: the target does not chase the object.
        row['bodies']['test_cylinder']['position_m'][0]+=.004
        np.testing.assert_allclose(self.ns['tracking_task'](row,reference,phase)[0],point)
        self.assertTrue(self.ns['tracking_phase_is_fresh'](phase,100+99000000000))
        self.assertFalse(self.ns['tracking_phase_is_fresh'](phase,100+101000000000))
        row['bodies']['test_cylinder']['position_m'][0]+=.007
        with self.assertRaisesRegex(RuntimeError,'SLIPPED'):self.ns['tracking_task'](row,reference,phase)

    def test_empty_bottle_may_tilt_but_cannot_leave_the_grasp_point(self):
        from scipy.spatial.transform import Rotation
        reference=dict(grasp_height_above_center_m=.1)
        phase=dict(phase="CARRY_HOLD",native_arrival_pass=True,post_retract_physical_hold_pass=True,
                   generation=1,carry_heading_xy_m=[.3,-.2],carry_world_z_m=.85,
                   initial_object_in_hand_m=[0.,0.,.1],slip_reference="GRASP_SECTION")
        B=Rotation.from_euler("y",.4).as_matrix()
        com=np.array([1.,1.,.85])-.1*B[:,2]
        row=dict(generation=1,bodies={
            "pelvis":dict(position_m=[2.,3.,.65],rotation=np.eye(3).ravel().tolist()),
            "R_omnipicker_base_link":dict(position_m=[1.,1.,.75],rotation=np.eye(3).ravel().tolist()),
            "test_cylinder":dict(position_m=com.tolist(),rotation=B.ravel().tolist())})
        with self.assertRaisesRegex(RuntimeError,"TILTED"):self.ns["tracking_task"](row,reference,phase)
        phase["allow_carried_bottle_tilt"]=True
        point,axis=self.ns["tracking_task"](row,reference,phase)
        np.testing.assert_allclose(point,[2.3,2.8,.85])
        np.testing.assert_array_equal(axis,[0.,0.,1.])
        row["bodies"]["test_cylinder"]["position_m"][0]+=.011
        with self.assertRaisesRegex(RuntimeError,"SLIPPED"):self.ns["tracking_task"](row,reference,phase)

    def test_carry_velocity_feedforward_uses_unique_source_time_and_source_limits(self):
        calculate=self.ns['bounded_tracking_velocity']
        np.testing.assert_allclose(calculate([0.,0.],[.02,-.1],.02,[2.,3.]),[1.,-3.])
        for dt in (0.,-.02,.21,float('nan')):
            with self.assertRaisesRegex(RuntimeError,'CLOCK'):calculate([0.],[.01],dt,[2.])
    def test_whole_task_rejects_lift_infeasible_grasp_without_discarding_other_branch(self):
        from types import SimpleNamespace as NS
        from unittest import mock
        rows=[dict(pass_result=True,pass_=True,q_rad=dict(q=.6),gates={}),dict(pass_result=True,pass_=True,q_rad=dict(q=.2),gates={}),dict(pass_result=True,pass_=True,q_rad=dict(q=.1),gates={})]
        for row in rows:row['pass']=row.pop('pass_')
        def lift(model,state,q,*args):return dict(pass_=float(q[0])<.3)
        def checked(model,state,q,*args,**kwargs):
            result=lift(model,state,q,*args);result['pass']=result.pop('pass_');return result
        with mock.patch.dict(self.ns,dict(ik=NS(ACTIVE=['q']),screen_lift=checked)):
            self.ns['qualify_grasp_motion_branches'](rows,None,{},39,np.zeros(3),np.zeros(3),np.array([1.,0.,0.]))
        self.assertTrue(all(r['grasp_only_pass'] for r in rows))
        self.assertFalse(rows[0]['pass']);self.assertEqual(rows[0]['gates']['LIFT_SWEEP'],'FAIL')
        self.assertTrue(rows[1]['pass']);self.assertEqual(rows[1]['gates']['LIFT_SWEEP'],'PASS')
        self.assertFalse(rows[2]['pass']);self.assertEqual(rows[2]['motion_qualification'],'NOT_EVALUATED')

    def test_25percent_actual_grip_force_increment_keeps_minimum_gate(self):
        minimum,old=self.ns['grip_force_budget'](.2*9.81,1.,6.)
        minimum2,new=self.ns['grip_force_budget'](.2*9.81,1.,7.5)
        self.assertEqual(minimum,minimum2)
        self.assertAlmostEqual(old,5.886)
        self.assertAlmostEqual(new,7.3575)
        self.assertAlmostEqual(new/old,1.25)
        with self.assertRaises(ValueError):self.ns['grip_force_budget'](.2*9.81,1.,8.)

    def test_grip_force_setpoint_is_not_the_minimum_load_acceptance(self):
        minimum,target=self.ns['grip_force_budget'](.2*9.81,1.,3.)
        self.assertAlmostEqual(minimum,1.4715);self.assertAlmostEqual(target,2.943)
        self.assertGreater(1.93,minimum);self.assertLess(1.93,target)
        self.assertEqual(self.ns['grip_force_budget'](.2*9.81,1.,1.5),(minimum,minimum))

    def test_full_source_posture_cannot_be_executed_or_silently_enabled(self):
        import subprocess,sys
        planner=Path(__file__).with_name("stl_approved_circle_ik.py")
        required=["--circle-review","/nonexistent","--candidate","/nonexistent","--standing-state","/nonexistent"]
        for extra in ([],["--optimize-waist-posture","--runtime-handoff"]):
            r=subprocess.run([sys.executable,str(planner),*required,"--full-source-posture-diagnostic",*extra],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertNotEqual(r.returncode,0)
            self.assertIn("full-source diagnostic is offline only",r.stdout)
            self.assertNotIn("FileNotFoundError",r.stdout)

    def test_source_waist_roll_diagnostic_has_no_dynamic_execution_path(self):
        import subprocess,sys
        tool=Path(__file__).with_name("approved_circle_stance_region.py")
        tree=ast.parse(tool.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="source_roll_diagnostic")
        calls=[n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn("mj_step",calls)
        self.assertNotIn("create_publisher",calls)
        r=subprocess.run([sys.executable,str(tool),"--source-roll-diagnostic","/nonexistent","--execute-simulation"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.assertEqual(r.returncode,2)
        self.assertIn("unrecognized arguments",r.stdout)
        self.assertNotIn("FileNotFoundError",r.stdout)

    def test_axial_roll_is_free_but_centre_offset_rotates(self):
        c=np.array([0.,.002,.132]);p=np.array([1.3,-1.65,.8]);axis=np.array([0.,0.,1.])
        for theta in (0.,.8,-.8,2.):
            x=axis;y=np.array([np.cos(theta),np.sin(theta),0.]);r=np.column_stack([x,y,np.cross(x,y)])
            hand=self.ns['circle_target'](p,r,c)
            np.testing.assert_allclose(self.ns['axis_grasp_error'](hand,r,c,p,axis),0,atol=1e-12)

    def test_inverted_axis_does_not_falsely_pass(self):
        r=np.diag([-1.,1.,-1.]);e=self.ns['axis_grasp_error'](np.zeros(3),r,np.zeros(3),np.zeros(3),np.array([1.,0.,0.]))
        self.assertAlmostEqual(np.linalg.norm(e[3:]),.2)

    def test_explicit_opposite_cylinder_axis_is_a_separate_valid_pose_branch(self):
        rotation=np.diag([-1.,1.,-1.]);center=np.array([0.,.002,.132]);target=np.array([1.,2.,.7])
        hand=target-rotation@center
        e=self.ns['axis_grasp_error'](hand,rotation,center,target,np.array([1.,0.,0.]),np.array([-1.,0.,0.]))
        np.testing.assert_allclose(e,0.,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(self.ns['axis_grasp_error'](hand,rotation,center,target,np.array([1.,0.,0.]))[3:]),.2)

    def test_ten_mm_tracking_error_is_not_hidden(self):
        e=self.ns['axis_grasp_error'](np.array([.01,0,0]),np.eye(3),np.zeros(3),np.zeros(3),np.array([1.,0,0]))
        self.assertAlmostEqual(np.linalg.norm(e[:3]),.01)

    def test_small_task_frame_tilt_does_not_move_the_circle_center(self):
        theta=.026;c=np.cos(theta);s=np.sin(theta)
        local=np.array([c,0.,s])
        # Build any proper rotation with R0*X along the bottle axis.
        r0=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        r=r0@np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
        circle=np.array([0.,0.,.132]);target=np.array([1.,2.,.8]);p=target-r@circle
        np.testing.assert_allclose(self.ns['axis_grasp_error'](p,r,circle,target,np.array([0.,0.,1.]),local),0.,atol=1e-12)

    def test_closure_freezes_anchor_without_chasing_object(self):
        row={'generation':1,'bodies':{'test_cylinder':{'position_m':[1.,2.,.7],'rotation':np.eye(3).reshape(-1).tolist()}}}
        reference={'grasp_height_above_center_m':.1,'grasp_section_center_world_m':[1.,2.,.8]}
        phase={'phase':'CLOSING','native_arrival_pass':True,'generation':1,'target_world_m':[1.,2.,.8],'bottle_center_world_m':[1.,2.,.7],'bottle_axis_world':[0.,0.,1.]}
        row['bodies']['test_cylinder']['position_m'][0]+=.005
        measured,_=self.ns['tracking_task'](row,reference)
        frozen,_=self.ns['tracking_task'](row,reference,phase)
        self.assertAlmostEqual(measured[0],1.005)
        np.testing.assert_allclose(frozen,[1.,2.,.8])
        row['bodies']['test_cylinder']['position_m'][0]=1.011
        with self.assertRaisesRegex(RuntimeError,'BEFORE_CLOSURE'):self.ns['tracking_task'](row,reference)
        with self.assertRaisesRegex(RuntimeError,'DURING_CLOSURE'):self.ns['tracking_task'](row,reference,phase)

    def test_closure_requires_real_arrival_and_same_generation(self):
        row={'generation':1,'bodies':{'test_cylinder':{'position_m':[0.,0.,0.],'rotation':np.eye(3).reshape(-1).tolist()}}}
        reference={'grasp_height_above_center_m':.1,'grasp_section_center_world_m':[0.,0.,.1]}
        phase={'phase':'CLOSING','native_arrival_pass':False,'generation':1,'target_world_m':[0.,0.,.1],'bottle_center_world_m':[0.,0.,0.],'bottle_axis_world':[0.,0.,1.]}
        with self.assertRaisesRegex(RuntimeError,'INVALID_CLOSURE'):self.ns['tracking_task'](row,reference,phase)
        phase['native_arrival_pass']=True;phase['generation']=2
        with self.assertRaisesRegex(RuntimeError,'GENERATION'):self.ns['tracking_task'](row,reference,phase)

    def test_arrival_hold_freezes_verified_vision_target_without_authorizing_closure(self):
        row={'generation':1,'bodies':{'test_cylinder':{'position_m':[.002,0.,.7],'rotation':np.eye(3).reshape(-1).tolist()}}}
        reference={'grasp_height_above_center_m':.1}
        phase={'phase':'ARRIVAL_HOLD','native_arrival_pass':True,'generation':1,'target_world_m':[0.,0.,.8],'bottle_center_world_m':[0.,0.,.7],'bottle_axis_world':[0.,0.,1.]}
        target,axis=self.ns['tracking_task'](row,reference,phase)
        np.testing.assert_allclose(target,[0.,0.,.8])
        self.assertNotIn('bilateral_contact_pass',phase)
        row['bodies']['test_cylinder']['position_m'][0]=.02
        with self.assertRaisesRegex(RuntimeError,'MOVED_DURING_CLOSURE'):self.ns['tracking_task'](row,reference,phase)

    def test_lift_requires_contact_and_tracks_only_bounded_planned_vertical_motion(self):
        row={'generation':1,'bodies':{'test_cylinder':{'position_m':[0.,0.,.73],'rotation':np.eye(3).reshape(-1).tolist()}}}
        phase={'phase':'LIFTING','native_arrival_pass':True,'generation':1,'target_world_m':[0.,0.,.8],'bottle_center_world_m':[0.,0.,.7],'bottle_axis_world':[0.,0.,1.],'lift_offset_m':.03,'bilateral_contact_pass':True}
        row['bodies']['R_omnipicker_base_link']={'position_m':[0.,0.,.03],'rotation':np.eye(3).reshape(-1).tolist()}
        phase['initial_object_in_hand_m']=[0.,0.,.8];phase['slip_reference']='GRASP_SECTION'
        target,axis=self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)
        np.testing.assert_allclose(target,[0.,0.,.83])
        phase['bilateral_contact_pass']=False
        with self.assertRaisesRegex(RuntimeError,'UNQUALIFIED'):self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)
        phase['bilateral_contact_pass']=True;phase['lift_offset_m']=.061
        with self.assertRaisesRegex(RuntimeError,'UNBOUNDED'):self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)
        phase['lift_offset_m']=.03;row['bodies']['test_cylinder']['position_m'][2]=.7
        with self.assertRaisesRegex(RuntimeError,'NOT_FOLLOWING'):self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)

    def test_rotation_about_grasp_section_is_not_com_slip_but_tilt_remains_guarded(self):
        phase={'phase':'LIFTING','native_arrival_pass':True,'generation':1,'target_world_m':[0.,0.,.8],'bottle_center_world_m':[0.,0.,.7],'bottle_axis_world':[0.,0.,1.],'lift_offset_m':.03,'bilateral_contact_pass':True,'initial_object_in_hand_m':[0.,0.,.8],'slip_reference':'GRASP_SECTION'}
        for theta in (.06,.11):
            c=np.cos(theta);s=np.sin(theta);R=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
            row={'generation':1,'bodies':{'test_cylinder':{'position_m':(np.array([0.,0.,.83])-.1*R[:,2]).tolist(),'rotation':R.reshape(-1).tolist()},'R_omnipicker_base_link':{'position_m':[0.,0.,.03],'rotation':np.eye(3).reshape(-1).tolist()}}}
            if theta<.1:self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)
            else:
                with self.assertRaisesRegex(RuntimeError,'TILTED'):self.ns['tracking_task'](row,{'grasp_height_above_center_m':.1},phase)



    def test_retract_target_preserves_grasp_and_does_not_chase_object(self):
        R=np.eye(3).reshape(-1).tolist()
        row={"generation":1,"bodies":{"test_cylinder":{"position_m":[-.1,-.1,.76],"rotation":R},
             "R_omnipicker_base_link":{"position_m":[-.1,-.1,.86],"rotation":R}}}
        phase={"phase":"RETRACTING","native_arrival_pass":True,"generation":1,"pre_retract_physical_lift_pass":True,
               "target_world_m":[0.,0.,.8],"bottle_center_world_m":[0.,0.,.7],
               "bottle_axis_world":[0.,0.,1.],"lift_offset_m":.06,
               "bilateral_contact_pass":True,"slip_reference":"GRASP_SECTION",
               "initial_object_in_hand_m":[0.,0.,0.],"retract_offset_world_m":[-.1,-.1,0.]}
        target,axis=self.ns["tracking_task"](row,{"grasp_height_above_center_m":.1},phase)
        np.testing.assert_allclose(target,[-.1,-.1,.86])
        np.testing.assert_allclose(axis,[0.,0.,1.])
        row["bodies"]["test_cylinder"]["position_m"][0]+=.02
        with self.assertRaisesRegex(RuntimeError,"NOT_FOLLOWING"):
            self.ns["tracking_task"](row,{"grasp_height_above_center_m":.1},phase)

    def test_retract_rejects_out_of_bounds_translation(self):
        R=np.eye(3).reshape(-1).tolist()
        row={"generation":1,"bodies":{"test_cylinder":{"position_m":[0.,0.,.76],"rotation":R},
             "R_omnipicker_base_link":{"position_m":[0.,0.,.86],"rotation":R}}}
        phase={"phase":"RETRACTING","native_arrival_pass":True,"generation":1,"pre_retract_physical_lift_pass":True,
               "target_world_m":[0.,0.,.8],"bottle_center_world_m":[0.,0.,.7],
               "bottle_axis_world":[0.,0.,1.],"lift_offset_m":.06,
               "bilateral_contact_pass":True,"slip_reference":"GRASP_SECTION",
               "initial_object_in_hand_m":[0.,0.,0.]}
        for offset in ([.251,0.,0.],[0.,0.,.051],[float("nan"),0.,0.]):
            phase["retract_offset_world_m"]=offset
            with self.assertRaisesRegex(RuntimeError,"UNQUALIFIED_RETRACT"):
                self.ns["tracking_task"](row,{"grasp_height_above_center_m":.1},phase)

    def test_retract_phase_has_own_bounded_clock_after_fresh_physical_lift(self):
        now=20000000000
        phase={"phase":"RETRACTING","created_host_ns":14000000000,"pre_retract_physical_lift_pass":True,"retract_from_hold_end_ns":13990000000}
        self.assertTrue(self.ns["tracking_phase_is_fresh"](phase,now))
        self.assertFalse(self.ns["tracking_phase_is_fresh"](phase,now+1))
        phase["created_host_ns"]=now;phase["retract_from_hold_end_ns"]=now-10000000
        self.assertTrue(self.ns["tracking_phase_is_fresh"](phase,now+5500000000))
        phase["pre_retract_physical_lift_pass"]=False
        self.assertFalse(self.ns["tracking_phase_is_fresh"](phase,now))
        phase["pre_retract_physical_lift_pass"]=True;phase["retract_from_hold_end_ns"]=now-201000000
        self.assertFalse(self.ns["tracking_phase_is_fresh"](phase,now))

if __name__=='__main__':unittest.main()
