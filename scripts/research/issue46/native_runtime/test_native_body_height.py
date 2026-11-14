"""No-physics checks of the installed-MC body-height compatibility probe."""
import ast,builtins,dis,inspect,types,unittest
from scripts.research.issue46.native_runtime import native_body_height as body
class BodyHeightTests(unittest.TestCase):
    def test_probe_closures_have_no_missing_runtime_globals(self):
        pending=[body.run_height_probe.__code__];names=set()
        while pending:
            code=pending.pop();names.update(i.argval for i in dis.get_instructions(code) if i.opname=="LOAD_GLOBAL")
            pending.extend(x for x in code.co_consts if isinstance(x,types.CodeType))
        self.assertFalse(names-set(dir(builtins))-{"MICRO","SOURCE","write_durable_milestone","same_episode_registered_body_source","body_pause_zero_required","standing_height_reference_ready","standing_height_upper_limit_reached"})
    def test_generated_observer_cannot_construct_velocity_publisher(self):
        code=body.build_script({},[])
        tree=ast.parse(code);dock=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Dock")
        init=next(n for n in dock.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="create_publisher" for n in ast.walk(init)))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
        cls=next(n for n in ast.walk(run) if isinstance(n,ast.ClassDef) and n.name=="Micro")
        pub=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="publish")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="publish" for n in ast.walk(pub)))
        self.assertIn("BODY_OBSERVER_VELOCITY_FORBIDDEN",ast.unparse(pub))
    def test_probe_never_requests_action_switch_or_joint_commands(self):
        code=inspect.getsource(body.run_height_probe)
        self.assertNotIn("stand_once",code);self.assertNotIn("SetMcAction",code)
        self.assertNotIn("JointCommand",code);self.assertNotIn("mj_step",code)
        with self.assertRaises(ValueError):body.build_script({},[],.20)


class BodyClockPauseTests(unittest.TestCase):
    def exercise_pause(self, failure="CLOCK_GENERATION_PENDING"):
        from types import SimpleNamespace as S
        from unittest.mock import patch
        import time
        tree=ast.parse(inspect.getsource(body.run_height_probe))
        sample=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='sample')
        fresh_mode=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='fresh_mode_sample')
        calls=[];result={};identity={'generation':1,'process_pid':42,'contacts':[{'efc_address':1,'body1':side+'_ankle_roll_link','body2':'world','local_force_torque':[100.]} for side in ('left','right')]}
        row=dict(get_mode_desc='STAND_DEFAULT',common_mode_desc='STAND_DEFAULT',get_mode_status_value=100,common_mode_status_value=100,get_mode_header_code=0,fsm_state=7,body_status=1,sample_monotonic_s=1.)
        replies=iter([dict(row,get_mode_desc=None,get_mode_header_code=None,get_mode_status_value=None) if failure=='MISSING_MODE_READBACK' else RuntimeError(failure),row,row])
        def read():
            value=next(replies);calls.append('read')
            if isinstance(value,Exception):raise value
            return value
        n=S(pump=lambda:None,sample=read,latest_row=identity,trace=[])
        def stop(*args):calls.append('stop');return {'pass':True}
        env=dict(time=time,deadline=time.monotonic()+10.,n=n,process=S(poll=lambda:None),body_pause_zero_required=body.body_pause_zero_required,result=result,command=lambda v:calls.append(('command',v)),settle_native_body=stop,contract=S(MC_FSM_ACTIVE_STATES=(7,)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fresh_mode,sample],type_ignores=[])),'body_clock_sample','exec'),env)
        self.assertEqual(env['sample'](),row)
        self.assertEqual(calls,['read',('command',0.),'read','stop','read'])
        self.assertTrue(result['clock_pauses'][0]['zero_before_reacquire'])
        n.sample=lambda:(_ for _ in ()).throw(RuntimeError('CLOCK_GENERATION_PENDING'))
        with self.assertRaisesRegex(RuntimeError,'CLOCK_GENERATION_PENDING'):env['sample']()

    def test_clock_pause_zeros_before_readback_and_requires_stop(self):
        self.exercise_pause()

    def test_clock_invalidated_get_mode_after_publisher_start_is_zero_only(self):
        self.exercise_pause('MISSING_MODE_READBACK')

    def test_confirmed_nonstand_is_not_missing_telemetry(self):
        row=dict(get_mode_desc='WALK',get_mode_header_code=0,get_mode_status_value=100)
        from types import SimpleNamespace as S
        tree=ast.parse(inspect.getsource(body.run_height_probe))
        fn=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='fresh_mode_sample')
        scope=dict(n=S(sample=lambda:row))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'fresh_mode_read','exec'),scope)
        self.assertEqual(scope['fresh_mode_sample'](),row)
        # Existing semantic gate remains authoritative after readback.
        self.assertIn('row["get_mode_desc"]!="STAND_DEFAULT"',inspect.getsource(body.run_height_probe))

    def test_common_state_gap_also_stops_and_requires_new_fresh_state(self):
        self.exercise_pause("CRITICAL_ROS_TELEMETRY_STALE:{'common':0.449}")

    def test_substantial_source_failure_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError,'SOURCE_JOINT_SPEED'):
            self.exercise_pause('SOURCE_JOINT_SPEED:left_knee_joint')

class BodySourceReuseTests(unittest.TestCase):
    def test_only_successful_same_native_episode_registration_can_be_reused(self):
        import copy
        row={'generation':1,'process_pid':46}
        proof={'pass_result':True,'mode':'lower_hold','generation':1,'native_process_pid':46,'source_responses':[{'action':1001,'raw':{'response':{'header':{'code':0}}}}]}
        self.assertTrue(body.same_episode_registered_body_source(proof,row))
        for key,value in [('pass_result',False),('generation',2),('native_process_pid',47),('mode','restore'),('source_responses',[])]:
            bad=copy.deepcopy(proof);bad[key]=value;self.assertFalse(body.same_episode_registered_body_source(bad,row))
        proof['source_responses'][0]['raw']['response']['header']['code']=1
        self.assertFalse(body.same_episode_registered_body_source(proof,row))
    def test_task_lowering_margin_does_not_change_default_or_restore(self):
        text=body.build_script({},[],.10,mode='lower_hold',label='body_crouch_pickup',task_shortfall_tolerance_m=.015)
        compile(text,'body_task_lowering','exec')
        self.assertTrue(any(isinstance(n,ast.FunctionDef) and n.name=='same_episode_registered_body_source' for n in ast.parse(text).body))
        self.assertIn("'task_shortfall_tolerance_m': 0.015",text)
        for value in [.009,.016]:
            with self.assertRaises(ValueError):body.build_script({},[],.10,task_shortfall_tolerance_m=value)
        with self.assertRaises(ValueError):body.build_script({},[],.10,mode='restore',reference_result='proof',task_shortfall_tolerance_m=.015)

class CompletedStopClockEpochTests(unittest.TestCase):
    def test_clock_pending_after_passed_stop_only_rechecks_while_zero(self):
        from types import SimpleNamespace as NS
        tree=ast.parse(inspect.getsource(body.run_height_probe));fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='sample_after_qualified_stop')
        calls=[];states=iter([RuntimeError('CLOCK_GENERATION_PENDING'),{'pose':{'z':.65}}])
        def sample():
            x=next(states)
            if isinstance(x,Exception):raise x
            return x
        scope=dict(process=NS(poll=lambda:None),body_pause_zero_required=body.body_pause_zero_required,sample=sample,n=NS(latest_row={'generation':1,'process_pid':42}),result={'restored_stop':{'pass':True}},command=lambda v:calls.append(v),settle_native_body=lambda *a:({'pass':True}),deadline=100.)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_stop_epoch_read','exec'),scope)
        self.assertEqual(scope['sample_after_qualified_stop']('restored_stop')['pose']['z'],.65)
        self.assertEqual(calls,[0.]);self.assertFalse(scope['result']['stationary_clock_rechecks'][0]['motion_commands_issued'])

class ReadOnlyNativeStartupReadinessTests(unittest.TestCase):
    def test_initial_observer_waits_for_first_fresh_native_state_without_motion(self):
        from types import SimpleNamespace as NS
        tree=ast.parse(inspect.getsource(body.run_height_probe))
        loop=next(x for x in ast.walk(tree) if isinstance(x,ast.While) and any(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='first' for t in v.targets) for v in ast.walk(x)))
        for error,recover in [('NATIVE_FEEDBACK_STALE_OR_INVALID',True),('CRITICAL_ROS_TELEMETRY_STALE:common',True),('NATIVE_FEEDBACK_INVALID:ABI',False)]:
            clock=[0.];calls=[]
            def sample():
                calls.append('read')
                if len(calls)==1:raise RuntimeError(error)
                return {'fresh':True}
            scope=dict(n=NS(pump=lambda:None,messages=dict.fromkeys(['common','imu','leg_state','leg_command']),readbacks={'get_mode':1}),time=NS(monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt)),until=5.,sample=sample,result={},first=None)
            code=compile(ast.Module(body=[loop],type_ignores=[]),'actual_first_native_readiness','exec')
            if recover:
                exec(code,scope);self.assertEqual(scope['first'],{'fresh':True});self.assertEqual(len(calls),2);self.assertFalse(scope['result']['initial_readiness_waits'][0]['motion_publishers_started'])
            else:
                with self.assertRaisesRegex(RuntimeError,'ABI'):exec(code,scope)


class HeightZeroStopAcquisitionTests(unittest.TestCase):
    def helper(self, scope):
        tree=ast.parse(inspect.getsource(body.run_height_probe));fn=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='settle_native_body')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'height_zero_stop','exec'),scope)
        return scope['settle_native_body']

    def test_stopping_height_allows_fresh_window_reacquisition_and_restores_scope(self):
        import time
        from types import SimpleNamespace as S
        config={};events=[];scope=dict(MICRO=config);helper=self.helper(scope)
        def observe(*args,**kwargs):
            events.append(config.get('body_readonly_zero_phase'))
            return {'pass':True}
        result=helper(S(observe=observe),1,time.monotonic()+5.,{'commands':[{'stand_height':0.}]},'low_stop')
        self.assertTrue(result['pass']);self.assertEqual(events,[True]);self.assertEqual(config,{})

    def test_nonzero_height_command_cannot_be_reclassified_as_readonly(self):
        import time
        from types import SimpleNamespace as S
        config={};helper=self.helper(dict(MICRO=config))
        with self.assertRaisesRegex(RuntimeError,'BODY_STOP_REACQUIRE_REQUIRES_ZERO_HEIGHT_COMMAND'):
            helper(S(),1,time.monotonic()+5.,{'commands':[{'stand_height':-.03}]},'low_stop')
        self.assertEqual(config,{})


class StandingRecoveryHandoffTests(unittest.TestCase):
    def row(self, offset=0., enabled=False, limit=2):
        return {"raw_common": {"servo_status": dict(squat_offset=offset,
                                                    squat_enable=enabled,
                                                    squat_limit=limit)}}

    def test_recorded_near_zero_but_recovering_state_cannot_handoff(self):
        self.assertFalse(body.standing_height_reference_ready(
            self.row(-6.3106417869107645e-9, True, 0)))
        self.assertFalse(body.standing_height_reference_ready(self.row(0., True, 2)))
        self.assertTrue(body.standing_height_reference_ready(self.row()))

    def test_missing_or_invalid_state_is_not_recovery_completion(self):
        for row in [{}, self.row(float("nan")), self.row(float("inf")),
                    self.row(-.004), self.row(0., False, 0)]:
            self.assertFalse(body.standing_height_reference_ready(row))

    def test_actual_restore_loop_keeps_source_command_until_mc_clears(self):
        from types import SimpleNamespace as S
        tree = ast.parse(inspect.getsource(body.run_height_probe))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
                    and ast.unparse(n.test).startswith("time.monotonic() < raise_deadline"))
        calls = []
        rows = iter([self.row(-.002, True, 0),
                     self.row(-6.3106417869107645e-9, True, 0), self.row(0., True, 2)])
        def sample():
            row = next(rows)
            row.update(pose={"z": .653}, sample_monotonic_s=1.,
                       state_age_s={"common": .01})
            return row
        scope = dict(time=S(monotonic=lambda: 1., sleep=lambda t: None),
                     raise_deadline=2., result={}, sample=sample, height=.653,
                     initial_reference=0., command=calls.append, rate=.03,
                     standing_height_upper_limit_reached=body.standing_height_upper_limit_reached)
        exec(compile(ast.Module(body=[loop], type_ignores=[]), "restore_loop", "exec"), scope)
        self.assertEqual(calls, [.03, .03])
        self.assertEqual(len(scope["result"]["restore_reference_samples"]), 3)
        self.assertFalse(body.standing_height_reference_ready(scope["row"]))

    def test_generated_runtime_includes_the_mc_completion_predicate(self):
        code = body.build_script({}, [], .10, mode="restore",
                                 reference_result="same_episode.json", label="body_restore_carry")
        tree = ast.parse(code)
        self.assertTrue(any(isinstance(n, ast.FunctionDef)
                            and n.name == "standing_height_reference_ready" for n in tree.body))
        compile(code, "generated_body_restore", "exec")
