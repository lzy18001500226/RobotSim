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


    def test_native_feedback_cannot_refresh_from_stale_or_reset_generation(self):
        code=generated();tree=ast.parse(code)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Plan')
        cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('refresh_native','arm_fresh')]
        clock=NS(ns=1100000000)
        ns={'USE_NATIVE_FEEDBACK':True,'CONTROL':('left','right'),'time':NS(monotonic_ns=lambda:clock.ns),'json':json,'math':math}
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
