"""Exercise generated runtime feedback code without ROS, Docker or physics."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


def generated():
    tree=ast.parse(Path(__file__).with_name('stl_hand_startup.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_hal_bilateral_arm_plan_script')
    ns={'json':json,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'builder','exec'),ns)
    right=tuple('right_'+n for n in ('shoulder_pitch_joint','shoulder_roll_joint','shoulder_yaw_joint','elbow_joint','wrist_yaw_joint','wrist_pitch_joint','wrist_roll_joint'))
    names=right+tuple(n.replace('right_','left_') for n in right)
    return ns[node.name]({n:(30.,1.) for n in names},[0.]*7,[0.]*7,duration_s=6.,hold_s=1.,restore=False,right_hand_command=0.,hand_hold_s=2.)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.code=generated();self.tree=ast.parse(self.code)
        cls=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name=='Plan')
        cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('on_arm','arm_fresh')]
        self.clock=NS(now=1.)
        self.ns={'time':NS(monotonic=lambda:self.clock.now),'math':math,'LEFT':('left',),'RIGHT':('right',)}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'runtime','exec'),self.ns)
        self.n=self.ns['Plan']();self.n.arm={};self.n.arm_stamp={};self.n.arm_receipt={};self.n.arm_callbacks=0

    def message(self,stamp,names=('left','right'),position=.1):
        return NS(header=NS(stamp=NS(sec=1,nanosec=stamp)),joints=[NS(name=n,position=position) for n in names])

    def test_partial_messages_preserve_each_arm(self):
        self.n.on_arm(self.message(1,('left',)));self.assertFalse(self.n.arm_fresh())
        self.n.on_arm(self.message(1,('right',)));self.assertTrue(self.n.arm_fresh())

    def test_duplicate_messages_never_refresh_unique_receipt(self):
        self.n.on_arm(self.message(1));self.clock.now+=.19;self.n.on_arm(self.message(1))
        self.assertTrue(self.n.arm_fresh());self.clock.now+=.02;self.n.on_arm(self.message(1));self.assertFalse(self.n.arm_fresh())

    def test_regression_and_nonfinite_fail_closed(self):
        self.n.on_arm(self.message(2))
        with self.assertRaisesRegex(RuntimeError,'REGRESSING'):self.n.on_arm(self.message(1))
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):self.n.on_arm(self.message(3,position=float('nan')))

    def test_closure_needs_both_freshness_and_bilateral_tracking(self):
        guard=next(n for n in ast.walk(self.tree) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and 'preclose_error' in ast.unparse(n.test))
        test=compile(ast.Expression(guard.test),'closure','eval')
        for fresh,error,expected in [(False,0.,False),(True,.435,False),(True,.079,True)]:
            self.assertEqual(eval(test,{'HAND_RIGHT':0.,'preclose_error':error,'n':NS(arm_fresh=lambda:fresh)}),expected)

    def test_full_run_trace_and_callback_drain(self):
        self.assertIn('self.spin(max(0,min(next_tick,end)-time.monotonic()))',self.code)
        self.assertIn('self.last_trace_at-self.run_started',self.code)
        self.assertIn('arm_source_stamp_ns',self.code)
        self.assertNotIn('time.sleep(max(0,next_tick-time.monotonic()))',self.code)
        compile(self.code,'generated_pregrasp.py','exec')

    def test_optional_fallback_generates_executable_python(self):
        ns={}
        constants=[n for n in self.tree.body if isinstance(n,ast.Assign) and not any(isinstance(x,ast.Call) for x in ast.walk(n))]
        exec(compile(ast.Module(body=constants,type_ignores=[]),'constants','exec'),ns)
        self.assertEqual(ns['INITIAL_FALLBACK'],{})
        self.assertNotIn('INITIAL_FALLBACK[n_]',self.code)


if __name__=='__main__':unittest.main()
