"""Generated waist/arm ownership and native-TCP arrival regressions; no physics."""
import ast
import builtins
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from scripts.research.issue46.native_runtime.test_pregrasp_feedback import generated


def builder():
    tree=ast.parse(Path(__file__).with_name('stl_hand_startup.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_hal_bilateral_arm_plan_script')
    ns={'math':math,'json':json}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'builder','exec'),ns)
    return ns[fn.name]


class CoordinatedTests(unittest.TestCase):
    def test_waist_relay_preserves_mc_policy_and_every_other_config_field(self):
        import copy
        tree=ast.parse(Path(__file__).with_name('stl_hand_startup.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_arm_hand_split_mc_config','build_coordinated_waist_relay_config')]
        ns={};exec(compile(ast.Module(body=funcs,type_ignores=[]),'relay_config','exec'),ns)
        names=('arm_joint_command','waist_joint_command','hand_joint_command','leg_joint_command')
        source={'MotionControlModule':{'PublisherManager':{'publisher_cfg_list':[{'name':n,'enable':True,'topic':'/aima/hal/joint/'+n.split('_')[0]+'/command','frequency':500} for n in names]}},'aimrt':{'channel':{'pub_topics_options':[{'topic_name':'/aima/hal/joint/waist/command','backend':'ros2'}]}},'policy':{'gains':'UNCHANGED'}}
        expected=copy.deepcopy(source)
        for item in expected['MotionControlModule']['PublisherManager']['publisher_cfg_list']:
            if item['name'] in ('arm_joint_command','hand_joint_command'):item['enable']=False
            if item['name']=='waist_joint_command':item['topic']='/robotsim/issue46/mc/waist/command'
        expected['aimrt']['channel']['pub_topics_options'][0]['topic_name']='/robotsim/issue46/mc/waist/command'
        actual=ns['build_coordinated_waist_relay_config'](source)
        self.assertEqual(actual,expected)
        self.assertTrue(source['MotionControlModule']['PublisherManager']['publisher_cfg_list'][0]['enable'])

    def test_bootstrap_relays_exact_command_not_pre_reset_angle(self):
        path=Path(__file__).with_name('xvfb_native_reset_stand.py')
        fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='arm_bootstrap_hold_script')
        ns={'json':json};exec(compile(ast.Module(body=[fn],type_ignores=[]),'bootstrap','exec'),ns)
        code=ns[fn.name]({},relay_waist=True);compile(code,'bootstrap','exec')
        self.assertIn('self.waist_pub.publish(m)',code)
        self.assertNotIn('waist_initial',code)
        self.assertIn("STALE_MC_WAIST_RELAY",code)
        self.assertIn('while rclpy.ok() and time.monotonic()<next_tick',code)

    def test_native_observer_exports_every_arm_bias_joint(self):
        source=Path(__file__).with_name('native_grasp_audit.c').read_text()
        self.assertIn('j<m->njnt',source)
        self.assertIn('mjJNT_FREE',source)
        self.assertIn('mjJNT_BALL',source)
        self.assertIn('qfrc_bias[m->jnt_dofadr[j]]',source)

    def test_source_gains_and_all_controlled_axes(self):
        constants={}
        exec(compile(ast.Module(body=[n for n in ast.parse(generated()).body if isinstance(n,ast.Assign) and not any(isinstance(x,ast.Call) for x in ast.walk(n))],type_ignores=[]),'constants','exec'),constants)
        names=constants['LEFT']+constants['RIGHT'];waist=('waist_yaw_joint','waist_pitch_joint','waist_roll_joint')
        gains={n:(30.,1.) for n in names};gains.update({n:(150.,3.) for n in waist})
        code=builder()(gains,[0.]*7,[0.]*7,duration_s=6.,hold_s=1.,waist_target=[0.,.2,0.],hand_open_command=.5)
        compile(code,'coordinated','exec')
        ns={};exec(compile(ast.Module(body=[n for n in ast.parse(code).body if isinstance(n,ast.Assign) and not any(isinstance(x,ast.Call) for x in ast.walk(n))],type_ignores=[]),'constants','exec'),ns)
        self.assertEqual(len(ns['CONTROL']),17);self.assertEqual(ns['GAINS'],{k:list(v) for k,v in gains.items()})
        self.assertEqual(ns['HAND_OPEN'],.5)
        self.assertIn("self.waist_pub.publish(w)",code)
        self.assertIn("x.effort=0.",code)
        self.assertIn("NATIVE_BIAS_FILE_MISSING",code)
        self.assertIn("INCOMPLETE_NATIVE_BIAS_SAMPLE",code)
        self.assertIn("preclose_waist_error<=.05",code)
        self.assertIn("arrival[\"pass\"]",code)
        with self.assertRaises(ValueError):builder()({n:(30.,1.) for n in names},[0.]*7,[0.]*7,duration_s=6.,hold_s=1.,waist_target=[0.,.2,0.])

    def native(self,row):
        fn=next(n for n in ast.parse(generated()).body if isinstance(n,ast.FunctionDef) and n.name=='native_circle_arrival')
        ns={'math':math,'json':json,'time':NS(monotonic_ns=lambda:1_000_000_000),'open':lambda *a,**k:io.BytesIO((json.dumps(row)+'\n').encode())}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'native_tcp','exec'),ns)
        return ns['native_circle_arrival']({'circle_center_local_m':[0.,0.,0.],'height_above_center_m':0.})

    def row(self):
        # Proper rotation maps hand +X to bottle +Z.
        return {'host_monotonic_ns':950_000_000,'sim_time_s':1.,'bodies':{'R_omnipicker_base_link':{'position_m':[0.,0.,0.],'rotation':[0.,0.,1.,0.,1.,0.,1.,0.,0.]},'test_cylinder':{'position_m':[0.,0.,0.],'rotation':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}}}

    def test_native_pose_proves_arrival_not_joint_timer(self):
        row=self.row();self.assertTrue(self.native(row)['pass'])
        row['bodies']['R_omnipicker_base_link']['position_m'][0]=.0518
        self.assertFalse(self.native(row)['pass'])

    def test_axis_inversion_is_not_arrival(self):
        row=self.row();row['bodies']['R_omnipicker_base_link']['rotation'][6]=-1.
        self.assertFalse(self.native(row)['pass'])

    def test_odom_late_sample_does_not_refresh_unique_freshness(self):
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as native
        src=Path(native.__file__).read_text()
        self.assertIn("late_sample_discarded",src)
        self.assertIn("odom_last_unique_receipt",src)
        self.assertIn("odom_regressions",src)

    def test_stale_native_pose_cannot_authorize_close(self):
        row=self.row();row['host_monotonic_ns']=700_000_000
        with self.assertRaisesRegex(RuntimeError,'STALE_NATIVE'):self.native(row)


if __name__=='__main__':unittest.main()
