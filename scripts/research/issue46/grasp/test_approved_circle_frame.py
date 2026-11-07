"""Frame/axis tests of the actual planning residual; no MuJoCo physics."""
import ast
from pathlib import Path
import unittest
import numpy as np


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse(Path(__file__).with_name('stl_approved_circle_ik.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('axis_grasp_error','circle_target','tracking_task')]
        cls.ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'planner','exec'),cls.ns)

    def test_full_source_posture_cannot_be_executed_or_silently_enabled(self):
        import subprocess,sys
        planner=Path(__file__).with_name("stl_approved_circle_ik.py")
        required=["--circle-review","/nonexistent","--candidate","/nonexistent","--standing-state","/nonexistent"]
        for extra in ([],["--optimize-waist-posture","--runtime-handoff"]):
            r=subprocess.run([sys.executable,str(planner),*required,"--full-source-posture-diagnostic",*extra],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertNotEqual(r.returncode,0)
            self.assertIn("full-source diagnostic is offline only",r.stdout)
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



if __name__=='__main__':unittest.main()
