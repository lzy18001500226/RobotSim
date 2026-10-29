"""Offline policy and fake-ROS tests; real ROS construction is a separate smoke."""
import ast
import math
import unittest
from pathlib import Path

from issue80_raicom_stand_takeover import StandingRun, OBSERVER, summarize_raw_lidar_trace


def pure(name):
    tree=ast.parse(OBSERVER.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'math':math,'MIN_ROOT_Z_M':.52,'MAX_ABS_TILT_RAD':math.radians(15),
         'MAX_PLANAR_SPEED_M_S':.2,'JD_READY_TIMEOUT_S':1.,'SD_READY_TIMEOUT_S':2.}
    if name=='request_allowed':env['safe_posture']=pure('safe_posture')
    if name=='safe_posture':env['posture_safety_reason']=pure('posture_safety_reason')
    if name=='native_transition_state':env['active_mode_agrees']=pure('active_mode_agrees')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(OBSERVER),'exec'),env)
    return env[name]


class StandingTests(unittest.TestCase):
    def test_source_local_raw_trace_requires_complete_fixed_ray_rows(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            trace=Path(directory)/'lidar_raw_frames.jsonl'
            self.assertEqual(summarize_raw_lidar_trace(trace)['status'],'BLOCKED_RAW_TRACE_MISSING')
            trace.write_text('',encoding='utf-8')
            self.assertEqual(summarize_raw_lidar_trace(trace)['status'],'BLOCKED_RAW_TRACE_EMPTY')
            trace.write_text(json.dumps({'schema':'robotsim.issue80.lidar-raw.v1','horizontal':2,'vertical':1,'xyz':[[0,0,0],None]})+'\n',encoding='utf-8')
            summary=summarize_raw_lidar_trace(trace)
            self.assertEqual(summary['status'],'PASS_SOURCE_LOCAL_RAY_TRACE')
            self.assertEqual(summary['row_count'],1)
            self.assertEqual(summary['ray_count'],2)
            trace.write_text(json.dumps({'schema':'robotsim.issue80.lidar-raw.v1','horizontal':2,'vertical':1,'xyz':[[0,0,0]]})+'\n',encoding='utf-8')
            self.assertEqual(summarize_raw_lidar_trace(trace)['status'],'BLOCKED_RAW_TRACE_RAY_COUNT')

    def test_raw_trace_enable_is_after_qualified_hold_and_never_a_control_path(self):
        source=Path(__file__).with_name('issue80_raicom_stand_takeover.py').read_text(encoding='utf-8')
        method=source[source.index('    def after_observer_completion'):source.index('    def run_once')]
        self.assertIn("enable_source_local_lidar_trace_after_sd_stable_hold",method)
        self.assertIn("': > /workspace/evidence/lidar_raw_enable'",method)
        self.assertGreater(method.index('enable_source_local_lidar_trace_after_sd_stable_hold'),method.index("summary=summaries.get('stand_takeover_summary.json')"))
        self.assertNotIn('create_publisher(',method)
        self.assertNotIn('SetMcInputSource',method)

    def test_background_diagnostic_is_read_only_and_optional(self):
        source=Path(__file__).with_name('issue80_raicom_stand_takeover.py').read_text(encoding='utf-8')
        method=source[source.index('    def after_observer_completion'):source.index('    def run_once')]
        self.assertIn('start_readonly_lidar_background_obstacle_diagnostic',method)
        self.assertIn('--background-capture /workspace/evidence/lidar_map_background_capture.json',method)
        self.assertIn('--static-match-distance-m 0.02',method)
        self.assertNotIn('create_publisher(',method)
        self.assertNotIn('SetMcInputSource',method)
        self.assertIn('LIDAR_RAW_BRIDGE',source)
        self.assertIn("overlay_target=self.evidence/'lidar_overlay'",source)
        self.assertIn('staged_input=self.evidence/name',source)
        self.assertIn("parser.add_argument('--lidar-background-capture'",source)
        self.assertIn("parser.add_argument('--lidar-static-map-source'",source)
        self.assertIn("parser.add_argument('--lidar-world-projection'",source)
        self.assertIn('start_simulation_only_world_projected_lidar_bridge',method)
        self.assertIn('--coordinate-field world_xyz',method)
        self.assertIn('--topic /robotsim/lidar/map_points --frame map',method)

    def test_background_staging_does_not_replace_the_overlay_directory(self):
        import argparse
        import json
        import tempfile
        from pathlib import Path
        class Runtime:
            lidar_overlay_source = None
            lidar_overlay_dir = None
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            overlay=root/'overlay-source'
            lidar=overlay/'lidar_overlay'
            plugins=overlay/'mujoco_plugin'
            lidar.mkdir(parents=True)
            plugins.mkdir()
            for name in ('x2_lidar_overlay.xml','robot_lidar_overlay.yaml','scene_lidar_overlay.xml','simulator_lidar_overlay.yaml'):
                (lidar/name).write_text(name,encoding='utf-8')
            (plugins/'libissue80_ray.so').write_bytes(b'plugin')
            background=root/'background.json'
            background.write_text(json.dumps({'status':'PASS_NONEMPTY_POINTCLOUD'}),encoding='utf-8')
            static=root/'static-map'
            static.mkdir()
            (static/'lidar_static_map.bin').write_bytes(b'map')
            (static/'lidar_map_localization_trace.jsonl').write_text('{}\n',encoding='utf-8')
            evidence=root/'evidence'
            evidence.mkdir()
            args=argparse.Namespace(
                attempt_id='offline',protocol='jd-sd',lidar_overlay_source=overlay,
                lidar_background_capture=background,lidar_static_map_source=static,
            )
            run=StandingRun(args,evidence,'none')
            runtime=Runtime()
            run.configure_runtime_overlay(runtime,{})
            self.assertEqual(runtime.lidar_overlay_dir,evidence/'lidar_overlay')
            self.assertEqual(run.lidar_static_map_inputs['lidar_map_localization_trace.jsonl'],evidence/'lidar_map_localization_trace.jsonl')
            self.assertTrue((runtime.lidar_overlay_dir/'x2_lidar_overlay.xml').is_file())

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

    def test_posture_failure_reason_preserves_each_existing_gate(self):
        reason=pure('posture_safety_reason')
        pose={'x_m':-1.5,'y_m':-1.5,'yaw_rad':0.,'z_m':.68,
              'roll_rad':0.,'pitch_rad':0.,'linear_velocity_m_s':[0.,0.,0.]}
        self.assertIsNone(reason(pose))
        self.assertEqual(reason(dict(pose,z_m=.519)),'ROOT_HEIGHT_BELOW_MINIMUM')
        self.assertEqual(reason(dict(pose,pitch_rad=math.radians(15))),'ROLL_OR_PITCH_LIMIT')
        self.assertEqual(reason(dict(pose,linear_velocity_m_s=[.201,0.,0.])),
                         'PLANAR_SPEED_LIMIT')
        self.assertEqual(reason(dict(pose,yaw_rad=float('nan'))),'NONFINITE_POSTURE_OR_TWIST')

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

    def test_existing_node_graph_rejects_any_velocity_publisher(self):
        from types import SimpleNamespace
        class Graph:
            def __init__(self,endpoints):self.endpoints=endpoints
            def get_publishers_info_by_topic(self,topic):
                self.topic=topic;return self.endpoints
        snapshot=pure('velocity_graph_snapshot')
        self.assertEqual(snapshot(Graph([]))['status'],'PASS_ZERO_VELOCITY_PUBLISHERS')
        graph=Graph([SimpleNamespace(node_name='unexpected',node_namespace='/')])
        self.assertEqual(snapshot(graph)['status'],'BLOCKED_VELOCITY_PUBLISHER')
        self.assertEqual(graph.topic,'/aima/mc/locomotion/velocity')

    def test_graph_errors_fail_closed_and_post_reset_readbacks_are_required(self):
        source=OBSERVER.read_text()
        self.assertIn('VELOCITY_GRAPH_INSPECTION_ERROR:',source)
        self.assertIn("reason='READY'",source)
        self.assertIn("reason='ACTION_GATE_' + action",source)
        self.assertIn('self.get_result_request_ns >= self.reset_ns',source)
        self.assertIn("self.received_ns.get('common', 0) >= self.reset_ns",source)
        self.assertIn('GET_MC_ACTION_TIMEOUT',source)
        self.assertNotIn('ros2 topic info --verbose',source)

    def test_jd_protocol_cannot_reset_without_observer_zero_publisher_proof(self):
        import argparse,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            run=StandingRun(argparse.Namespace(attempt_id='offline',protocol='jd-sd'),Path(directory),'none')
            self.assertEqual(run.inspect_velocity_endpoint()['status'],'PENDING_OBSERVER_GRAPH_REQUIRED_BEFORE_RESET')
            for invalid in ({},{'velocity_graph':{'publisher_count':0}},
                            {'velocity_graph':{'status':'PASS_ZERO_VELOCITY_PUBLISHERS','publisher_count':1,
                                               'method':'rclpy_existing_observer_graph'}}):
                with self.assertRaises(RuntimeError):run.validate_observer_ready(invalid)
            run.validate_observer_ready({'velocity_graph':{'status':'PASS_ZERO_VELOCITY_PUBLISHERS',
                'publisher_count':0,'method':'rclpy_existing_observer_graph'}})
            self.assertEqual(run.reset_dispatch_count,0)

    def test_pre_observer_failure_proves_zero_actions_without_inventing_post_launch_counts(self):
        import argparse,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from issue80_raicom_reset_only import ResetOnlyRun
        def blocked():return {'status':'BLOCKED','failure':'graph CLI timeout','runtime_non_motion_invariants':{}}
        with tempfile.TemporaryDirectory() as directory:
            run=StandingRun(argparse.Namespace(attempt_id='offline',protocol='jd-sd'),Path(directory),'none')
            with patch.object(ResetOnlyRun,'run_once',return_value=blocked()):
                self.assertEqual(run.run_once({})['mc_action_request_count'],0)
            run.observer_launch_started=True
            with patch.object(ResetOnlyRun,'run_once',return_value=blocked()):
                self.assertIsNone(run.run_once({})['mc_action_request_count'])

    def test_step_log_retains_monotonic_deadline_evidence(self):
        import json,tempfile,subprocess
        from pathlib import Path
        from issue80_m1_auto_reset_run import BoundedRun
        with tempfile.TemporaryDirectory() as directory:
            run=object.__new__(BoundedRun)
            run.evidence=Path(directory);run.step=0;run.steps=[]
            run.save_command('offline_timeout',['no-process'],subprocess.CompletedProcess([],124,'',''),
                             100.,103.003,'subprocess timeout')
            record=json.loads((run.evidence/'execution.jsonl').read_text())
            self.assertEqual(record['started_monotonic_s'],100.)
            self.assertEqual(record['ended_monotonic_s'],103.003)
            self.assertAlmostEqual(record['duration_s'],3.003)


if __name__=='__main__':
    unittest.main()
