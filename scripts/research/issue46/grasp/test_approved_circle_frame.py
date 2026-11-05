"""Frame/axis tests of the actual planning residual; no MuJoCo physics."""
import ast
from pathlib import Path
import unittest
import numpy as np


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse(Path(__file__).with_name('stl_approved_circle_ik.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('axis_grasp_error','circle_target')]
        cls.ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'planner','exec'),cls.ns)

    def test_axial_roll_is_free_but_centre_offset_rotates(self):
        c=np.array([0.,.002,.132]);p=np.array([1.3,-1.65,.8]);axis=np.array([0.,0.,1.])
        for theta in (0.,.8,-.8,2.):
            x=axis;y=np.array([np.cos(theta),np.sin(theta),0.]);r=np.column_stack([x,y,np.cross(x,y)])
            hand=self.ns['circle_target'](p,r,c)
            np.testing.assert_allclose(self.ns['axis_grasp_error'](hand,r,c,p,axis),0,atol=1e-12)

    def test_inverted_axis_does_not_falsely_pass(self):
        r=np.diag([-1.,1.,-1.]);e=self.ns['axis_grasp_error'](np.zeros(3),r,np.zeros(3),np.zeros(3),np.array([1.,0.,0.]))
        self.assertAlmostEqual(np.linalg.norm(e[3:]),.2)

    def test_ten_mm_tracking_error_is_not_hidden(self):
        e=self.ns['axis_grasp_error'](np.array([.01,0,0]),np.eye(3),np.zeros(3),np.zeros(3),np.array([1.,0,0]))
        self.assertAlmostEqual(np.linalg.norm(e[:3]),.01)


if __name__=='__main__':unittest.main()
