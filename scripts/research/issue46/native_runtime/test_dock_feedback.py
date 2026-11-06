"""Behavioural tests of generated dock feedback; no ROS or simulation."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


def script():
    path=Path(__file__).with_name('xvfb_native_reset_stand.py')
    tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='locomotion_dock_script')
    ns={'json':json};exec(compile(ast.Module(body=[f],type_ignores=[]),'builder','exec'),ns)
    return ns[f.name](world_start_xy=[-1.5,-1.5],world_start_yaw_rad=0.,dock_world_xy=[.74,-1.44],dock_world_yaw_rad=0.,dock_position_only=True,refine_world_xy=None,refine_world_yaw_rad=None,refine_position_only=False,dock_position_tolerance_m=.18,refine_position_tolerance_m=.15,timeout_s=30.,use_cmd_vel_bridge=True)


class DockFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.code=script();tree=ast.parse(self.code)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dock')
        cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('on_odom','on_arm','publish','zero_hold')]
        self.clock=NS(now=1.)
        self.ns={'math':math,'time':NS(monotonic=lambda:self.clock.now),'yaw':lambda q:0.,'rclpy':NS(ok=lambda:True),'Twist':lambda:NS(linear=NS(x=0.,y=0.),angular=NS(z=0.)),'PLAN':{'use_cmd_vel_bridge':True}}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'node','exec'),self.ns)
        self.n=self.ns['Dock']();self.n.pose=None;self.n.quat=None;self.n.odom_stamp=None;self.n.odom_callbacks=0;self.n.odom_duplicates=0;self.n.odom_regressions=0;self.n.odom_last_unique_receipt=None;self.n.feedback_fault=None;self.n.command_publish_count=0;self.n.arm={}

    def message(self,stamp,position=1.):
        return NS(header=NS(stamp=NS(sec=1,nanosec=stamp)),pose=NS(pose=NS(position=NS(x=position,y=0.,z=.65),orientation=NS(w=1.,x=0.,y=0.,z=0.))))

    def test_duplicate_does_not_refresh_age_or_pose(self):
        self.n.on_odom(self.message(1));self.clock.now+=.3;self.n.on_odom(self.message(1,2.))
        self.assertEqual(self.n.pose[0],1.);self.assertEqual(self.n.pose[4],1.);self.assertEqual(self.n.odom_duplicates,1)

    def test_warmup_history_discards_but_active_regression_latches_stop(self):
        self.n.on_odom(self.message(2));self.n.on_odom(self.message(1));self.assertIsNone(self.n.feedback_fault)
        self.n.command_publish_count=1;self.clock.now+=.21;self.n.on_odom(self.message(1));self.assertEqual(self.n.feedback_fault['gate'],'ODOMETRY_SOURCE_REGRESSION')

    def test_nonfinite_pose_is_invalid(self):
        self.n.on_odom(self.message(2,float('nan')));self.assertEqual(self.n.feedback_fault['gate'],'INVALID_ODOMETRY')

    def test_fault_forces_zero_even_if_nonzero_requested(self):
        sent=[];self.n.pub=NS(publish=sent.append);self.n.feedback_fault={'gate':'bad'}
        self.n.publish(.2,.1,.075);self.assertEqual((sent[0].linear.x,sent[0].linear.y,sent[0].angular.z),(0.,0.,0.))

    def test_stop_window_processes_callbacks_without_gaps(self):
        ticks=[];self.n.publish=lambda:ticks.append(self.clock.now)
        self.n.spin=lambda duration:setattr(self.clock,'now',self.clock.now+duration)
        self.n.zero_hold(.101)
        self.assertGreaterEqual(len(ticks),5);self.assertLessEqual(max(b-a for a,b in zip(ticks,ticks[1:])),.0200001)

    def test_partial_arm_groups_merge(self):
        self.n.on_arm(NS(joints=[NS(name='left',position=.1)]));self.n.on_arm(NS(joints=[NS(name='right',position=.2)]))
        self.assertEqual(self.n.arm,{'left':.1,'right':.2})

    def test_contract_still_uses_original_position_and_freshness_limits(self):
        compile(self.code,'dock','exec')
        self.assertIn('>.20',self.code);self.assertIn('DOCK_POST_ZERO_OUTSIDE_TOLERANCE',self.code)
        self.assertIn('FINAL_ODOMETRY_FRESHNESS',self.code);self.assertIn('n.spin(.019)',self.code)
        self.assertNotIn('time.sleep(.019)',self.code)

    def test_measured_stop_predictor_uses_velocity_and_stops_at_crossing(self):
        tree=ast.parse(self.code);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measured_stop_prediction')
        ns={'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),'stop','exec'),ns);predict=ns[fn.name]
        history=[(i*.02,.50+.5*i*.02,0.) for i in range(18)]
        result=predict(history,(.85,0.),(0.,0.),.43)
        self.assertTrue(result['stop']);self.assertAlmostEqual(result['measured_velocity_wall_m_s'][0],.5)
        self.assertIsNone(predict(history[:2],(.85,0.),(0.,0.),.43))
        crossed=[(t,x+.30,.08) for t,x,y in history]
        self.assertTrue(predict(crossed,(.85,0.),(0.,0.),.43)['crossed'])

    def test_postdock_observer_has_no_action_or_velocity_command_path(self):
        from scripts.research.issue46.native_runtime.stl_hand_startup import build_stand_client_script
        code=build_stand_client_script(observe_only=True);compile(code,'readonly','exec')
        self.assertNotIn('node.create_client(SetMcAction',code)
        self.assertNotIn('SetMcAction.Request()',code)
        self.assertNotIn('create_publisher',code)
        self.assertIn('node.create_client(GetMcAction',code)
        self.assertIn('READ_ONLY_NO_ACTION_REQUEST',code)
        self.assertIn('time.monotonic() + 2.5',code)


if __name__=='__main__':unittest.main()
