"""Offline policy and fake-ROS tests; real ROS construction is a separate smoke."""
import ast
import math
import unittest

from issue80_raicom_stand_takeover import StandingRun, OBSERVER


def pure(name):
    tree=ast.parse(OBSERVER.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'math':math,'MIN_ROOT_Z_M':.52,'MAX_ABS_TILT_RAD':math.radians(15),
         'MAX_PLANAR_SPEED_M_S':.2,'JD_READY_TIMEOUT_S':1.,'SD_READY_TIMEOUT_S':2.}
    if name=='request_allowed':env['safe_posture']=pure('safe_posture')
    if name=='native_transition_state':env['active_mode_agrees']=pure('active_mode_agrees')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(OBSERVER),'exec'),env)
    return env[name]


class StandingTests(unittest.TestCase):
    def test_safe_request_requires_reset_effect_fresh_advancing_one_shot(self):
        args=dict(reset_verified=True,reset_effect=True,fresh=True,
                  pose={'x_m':-1.5,'y_m':-1.5,'yaw_rad':math.pi/2,
                        'z_m':.68,'roll_rad':0,'pitch_rad':0,'linear_velocity_m_s':[0,0,0]},
                  advancing=True,count=0)
        allowed=pure('request_allowed')
        self.assertTrue(allowed(**args))
        for key in ('reset_verified','reset_effect','fresh','advancing'):
            self.assertFalse(allowed(**dict(args,**{key:False})))
        self.assertFalse(allowed(**dict(args,count=1)))
        for key,value in (('z_m',.1),('pitch_rad',.3),('roll_rad',float('nan')),('yaw_rad',float('nan'))):
            self.assertFalse(allowed(**dict(args,pose=dict(args['pose'],**{key:value}))))

    def test_request_path_is_sd_only_no_publishers_and_no_retry(self):
        tree=ast.parse(OBSERVER.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        clients=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='create_client']
        self.assertEqual(len(clients),2)
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='create_publisher' for n in calls))
        source=OBSERVER.read_text()
        self.assertEqual(source.count('self.sd_client.call_async(request)'),1)
        self.assertIn("request.command.action_desc = action",source)
        self.assertIn("action='JOINT_DEFAULT'",source)
        self.assertIn("elif not self.sd_count and qualified_jd:",source)
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

    def test_saved_5d005147_conflicting_transition_cannot_qualify_standing(self):
        # Exact reduced observations from the one authorized live episode.
        common={'action_info':{'action_desc':'STAND_DEFAULT','status':{'value':100}},
                'body_status':{'value':0},'fsm_state':{'current_state':6}}
        get={'header':{'code':0},'info':{'action_desc':'JOINT_DEFAULT','status':{'value':200}}}
        self.assertFalse(pure('active_mode_agrees')(common,get))
        self.assertIn('SD_READBACK_CONFLICT_RAW_FSM6',OBSERVER.read_text())

    @staticmethod
    def observations(mode,fsm,body=1,status=100,get_mode=None,get_status=None):
        return ({'action_info':{'action_desc':mode,'status':{'value':status}},
                 'fsm_state':{'current_state':fsm},'body_status':{'value':body}},
                {'header':{'code':0},'info':{'action_desc':get_mode or mode,
                 'status':{'value':status if get_status is None else get_status}}})

    def test_actual_run65_initial_fsm6_then_jd1_then_sd1_then_sd4(self):
        classify=pure('native_transition_state')
        for fsm,elapsed,expected in ((6,.035,'TRANSITION_PENDING'),(6,.136,'TRANSITION_PENDING'),
                                    (1,.23,'JD_READY')):
            c,g=self.observations('JOINT_DEFAULT',fsm)
            self.assertEqual(classify('JD',c,g,elapsed_s=elapsed,jd_accepted=True,sd_accepted=False),expected)
        c,g=self.observations('STAND_DEFAULT',1)
        self.assertEqual(classify('SD',c,g,elapsed_s=.097,jd_accepted=True,sd_accepted=True),'TRANSITION_PENDING')
        c,g=self.observations('STAND_DEFAULT',4)
        self.assertEqual(classify('SD',c,g,elapsed_s=.187,jd_accepted=True,sd_accepted=True),'SD_READY')
        self.assertEqual(classify('HOLD',c,g,elapsed_s=3.,jd_accepted=True,sd_accepted=True),'SD_READY')

    def test_jd_acceptance_and_both_ready_readbacks_required(self):
        c,g=self.observations('JOINT_DEFAULT',1)
        classify=pure('native_transition_state')
        self.assertEqual(classify('JD',c,g,elapsed_s=.2,jd_accepted=False,sd_accepted=False),'TRANSITION_PENDING')
        c,g=self.observations('JOINT_DEFAULT',1,get_status=200)
        self.assertEqual(classify('JD',c,g,elapsed_s=.2,jd_accepted=True,sd_accepted=False),'TRANSITION_PENDING')

    def test_new_protection_ambiguity_and_timeouts_fail_closed(self):
        classify=pure('native_transition_state')
        c,g=self.observations('JOINT_DEFAULT',6)
        self.assertEqual(classify('JD',c,g,elapsed_s=.3,jd_accepted=True,sd_accepted=False,
                                  left_initial_fsm6=True),'PROTECTION_CONFIRMED')
        self.assertEqual(classify('JD',c,g,elapsed_s=1.,jd_accepted=True,sd_accepted=False),'TRANSITION_TIMEOUT')
        c,g=self.observations('STAND_DEFAULT',6,body=0,get_mode='JOINT_DEFAULT',get_status=200)
        self.assertEqual(classify('SD',c,g,elapsed_s=.056,jd_accepted=True,sd_accepted=True),'PROTECTION_CONFIRMED')
        c,g=self.observations('STAND_DEFAULT',1)
        self.assertEqual(classify('SD',c,g,elapsed_s=2.,jd_accepted=True,sd_accepted=True),'TRANSITION_TIMEOUT')
        c,g=self.observations('STAND_DEFAULT',8)
        self.assertEqual(classify('SD',c,g,elapsed_s=.1,jd_accepted=True,sd_accepted=True),'PROTECTION_AMBIGUOUS')

    def test_duplicate_or_regressing_stamps_never_refresh_state(self):
        progress=pure('source_progress')
        self.assertEqual(progress(None,10),'ADVANCE')
        self.assertEqual(progress(10,11),'ADVANCE')
        self.assertEqual(progress(10,10),'DUPLICATE')
        self.assertEqual(progress(10,9),'REGRESSION')
        self.assertEqual(progress(10,None),'MISSING')

    def test_protocol_propagates_to_observer_without_new_publisher(self):
        import argparse
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            run=StandingRun(argparse.Namespace(attempt_id='offline',protocol='jd-sd'),Path(directory),'no-runtime')
            self.assertEqual(run.observer_cli_suffix,'--protocol jd-sd')


if __name__=='__main__':
    unittest.main()
