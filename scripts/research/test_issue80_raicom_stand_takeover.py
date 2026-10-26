"""Offline policy and fake-ROS tests; real ROS construction is a separate smoke."""
import ast
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from issue80_raicom_stand_takeover import StandingRun, OBSERVER


def pure(name):
    tree=ast.parse(OBSERVER.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'math':math,'MIN_ROOT_Z_M':.52,'MAX_ABS_TILT_RAD':math.radians(15),
         'MAX_PLANAR_SPEED_M_S':.2}
    if name=='request_allowed':env['safe_posture']=pure('safe_posture')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(OBSERVER),'exec'),env)
    return env[name]


class StandingTests(unittest.TestCase):
    def test_safe_request_requires_reset_effect_fresh_advancing_one_shot(self):
        args=dict(reset_verified=True,reset_effect=True,fresh=True,
                  pose={'z_m':.68,'roll_rad':0,'pitch_rad':0,'linear_velocity_m_s':[0,0,0]},
                  advancing=True,count=0)
        allowed=pure('request_allowed')
        self.assertTrue(allowed(**args))
        for key in ('reset_verified','reset_effect','fresh','advancing'):
            self.assertFalse(allowed(**dict(args,**{key:False})))
        self.assertFalse(allowed(**dict(args,count=1)))
        for key,value in (('z_m',.1),('pitch_rad',.3),('roll_rad',float('nan'))):
            self.assertFalse(allowed(**dict(args,pose=dict(args['pose'],**{key:value}))))

    def test_request_path_is_sd_only_no_publishers_and_no_retry(self):
        tree=ast.parse(OBSERVER.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        clients=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='create_client']
        self.assertEqual(len(clients),2)
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='create_publisher' for n in calls))
        source=OBSERVER.read_text()
        self.assertEqual(source.count('self.sd_client.call_async(request)'),1)
        self.assertIn("request.command.action_desc = 'STAND_DEFAULT'",source)
        self.assertNotIn('LOCOMOTION_DEFAULT',source)
        self.assertNotIn('SetMcInputSource',source)
        self.assertIn('self.joint_fault()',source)
        self.assertIn('UNIQUE_ODOM_STALE',source)

    def test_takeover_reuses_single_reset_parent_and_unchanged_watchdog(self):
        from issue80_raicom_reset_only import ResetOnlyRun,ACTIVE_CONTAINER_WATCHDOG_S,MAX_RUNTIME_S
        self.assertTrue(issubclass(StandingRun,ResetOnlyRun))
        self.assertEqual(ACTIVE_CONTAINER_WATCHDOG_S,26)
        self.assertEqual(MAX_RUNTIME_S,30)
        self.assertEqual(StandingRun.expected_observer_clients,2)
        self.assertIn('hold_samples>=25',OBSERVER.read_text())
        self.assertIn('STABLE_HOLD_S = 2.5',OBSERVER.read_text())

    def test_accepted_service_alone_cannot_pass_active_hold(self):
        import copy
        common={'action_info':{'action_desc':'STAND_DEFAULT','status':{'value':100}},
                'body_status':{'value':1},'fsm_state':{'current_state':4}}
        readback={'header':{'code':0},'info':{'action_desc':'STAND_DEFAULT','status':{'value':100}}}
        qualifies=pure('active_mode_agrees')
        self.assertTrue(qualifies(common,readback))
        self.assertFalse(qualifies({},{}))
        for key,value in (('action_desc','PASSIVE_DEFAULT'),('status',{'value':300})):
            bad=copy.deepcopy(common);bad['action_info'][key]=value
            self.assertFalse(qualifies(bad,readback))
            bad=copy.deepcopy(readback);bad['info'][key]=value
            self.assertFalse(qualifies(common,bad))
        for key,value in (('body_status',{'value':0}),('fsm_state',{'current_state':6})):
            bad=copy.deepcopy(common);bad[key]=value
            self.assertFalse(qualifies(bad,readback))
        self.assertFalse(qualifies(common,dict(readback,header={'code':1})))


if __name__=='__main__':
    unittest.main()
