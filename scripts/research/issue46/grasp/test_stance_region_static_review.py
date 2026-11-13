"""Pelvis navigation goals must survive a change of posture template."""
import unittest
import numpy as np
import mujoco
from scripts.research.issue46.grasp.stance_region_static_review import Study

class RuntimeRootPlacementTests(unittest.TestCase):
    def study(self):
        s=Study.__new__(Study)
        s.model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="pelvis"><freejoint/><geom size=".1" mass="1"/>
           <body name="left_ankle_roll_link" pos=".066 .1 -1"/>
           <body name="right_ankle_roll_link" pos=".066 -.1 -1"/>
          </body><body name="object"><freejoint/><geom size=".01" mass=".1"/></body>
          </worldbody></mujoco>''')
        s.data=mujoco.MjData(s.model);s.root=0;s.obj=7
        s.ids={name:mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_BODY,name) for name in ('left_ankle_roll_link','right_ankle_roll_link')}
        q=s.model.qpos0.copy();q[:3]=[.5,-1.5,1.]
        s.fixed_object=q[7:14].copy();s.templates={'prepared':q};s.qadr=np.zeros(9,dtype=int);s.source_bounds=np.tile([-3.,3.],(9,1));s.grasp_seed=np.zeros(9);s.nominal_yaw=0.;s.section=np.zeros(3)
        # Stop immediately after candidate placement, before any IK or contacts.
        class Captured(Exception):pass
        def solve(base,*args,**kwargs):
            s.captured=base.copy();raise Captured
        s.solve=solve
        return s,Captured

    def test_runtime_root_goal_does_not_inherit_the_66mm_ankle_offset(self):
        s,stop=self.study();goal=np.array([.94,-1.27])
        with self.assertRaises(stop):s.candidate(goal,.8,'prepared',root_xy=goal)
        np.testing.assert_allclose(s.captured[:2],goal,atol=1e-12)
        self.assertAlmostEqual(s.captured[2],1.)
        # Historical ankle-anchored study semantics stay available unchanged.
        with self.assertRaises(stop):s.candidate(goal,.8,'prepared')
        self.assertGreater(np.linalg.norm(s.captured[:2]-goal),.065)

    def test_nonfinite_runtime_goal_is_rejected_before_solving(self):
        s,_=self.study()
        with self.assertRaisesRegex(ValueError,'pelvis'):s.candidate([0.,0.],0.,'prepared',root_xy=[float('nan'),0.])

if __name__=='__main__':unittest.main()
