"""Explicit simulation speed diagnostics; physical/source faults still stop."""
import ast
import copy
import math
import textwrap
from pathlib import Path
import unittest
from scripts.research.issue46.native_runtime import native_base_micro_probe as m
from scripts.research.issue46.native_runtime import native_grasp_audit as audit

class SpeedPolicyTests(unittest.TestCase):
    def pose(self):
        return dict(x=0.,y=0.,z=.64,roll=0.,pitch=0.,yaw=0.,vx=.611,vy=0.)
    def test_legacy_still_stops_and_optin_preserves_measurement(self):
        p=self.pose()
        self.assertEqual(m.physical_fault(dict(joints={}),p,{},planar_speed_limit=.60),"PLANAR_SPEED")
        self.assertIsNone(m.physical_fault(dict(joints={}),p,{},planar_speed_limit=.60,planar_speed_warning_only=True))
        self.assertEqual(p["planar_speed_warning"]["measured_m_s"],.611)
    def test_warning_scope_cannot_relax_legacy_micro_or_expired_permit(self):
        for cap in [.20,.50]:
            with self.assertRaises(ValueError):
                m.physical_fault(dict(joints={}),self.pose(),{},planar_speed_limit=cap,planar_speed_warning_only=True)
    def test_warning_does_not_hide_posture_or_source_joint_faults(self):
        p=self.pose();p["z"]=.4
        self.assertEqual(m.physical_fault(dict(joints={}),p,{},planar_speed_limit=.60,planar_speed_warning_only=True),"POSTURE")
        row=dict(joints=dict(knee=dict(q=0.,qvel=2.)))
        self.assertEqual(m.physical_fault(row,self.pose(),dict(knee=(-1.,1.,1.)),planar_speed_limit=.60,planar_speed_warning_only=True),"SOURCE_JOINT_SPEED:knee")
        row["joints"]["knee"]["q"]=2.
        self.assertEqual(m.physical_fault(row,self.pose(),dict(knee=(-1.,1.,1.)),planar_speed_limit=.60,planar_speed_warning_only=True),"JOINT_LIMIT:knee")
    def test_table_collision_still_stops(self):
        row=dict(joints={},contacts=[dict(body1="pelvis",body2="raicom_delivery_table",efc_address=0,distance_m=-.002,local_force_torque=[10.])],active_contact_count=1,contact_capture_scope="ALL_COMPILED_GEOM_CONTACTS")
        self.assertEqual(m.physical_fault(row,self.pose(),{},robot_bodies={"pelvis"},planar_speed_limit=.60,planar_speed_warning_only=True),"ROBOT_ENVIRONMENT_COLLISION:pelvis:raicom_delivery_table")
    def test_nan_is_not_a_warning(self):
        p=self.pose();p["vx"]=float("nan")
        self.assertEqual(m.physical_fault(dict(joints={}),p,{},planar_speed_limit=.60,planar_speed_warning_only=True),"NONFINITE_BASE_STATE")
    def row(self,stamp):
        return dict(host_monotonic_ns=stamp,sim_time_s=stamp*1e-9,generation=1,engine_version=337,joints={},contacts=[],bodies=dict(pelvis=dict(position_m=[0.,0.,.64],rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.],origin_velocity_world_angular_linear=[0.,0.,0.,.611,0.,0.])))
    def test_independent_audit_keeps_legacy_failure_and_reports_warning(self):
        permit=dict(schema="robotsim.carry-transit.v2",policy="SIMULATION_ONLY_TABLE_TRANSFER_ROOT060",root_speed_limit_m_s=.60,owner_pid=42,generation=1,created_ns=10_000_000_000,command_deadline_ns=20_000_000_000,absolute_deadline_ns=35_000_000_000,route_pid=99,route_start_ticks=1,braking_ns=None,revoked_ns=None)
        rows=[self.row(11_000_000_000),self.row(11_020_000_000)]
        kw=dict(start_ns=0,end_ns=30_000_000_000,transit_policy=permit,expected_owner_pid=42)
        self.assertEqual(audit.evaluate_carry_motion(copy.deepcopy(rows),{},[],**kw)["failure"],"PLANAR_SPEED")
        result=audit.evaluate_carry_motion(copy.deepcopy(rows),{},[],**kw,planar_speed_warning_only=True)
        self.assertTrue(result["pass"]);self.assertFalse(result["legacy_planar_speed_gate_pass"])
        self.assertEqual(result["planar_speed_warnings"]["sample_count"],2)
        self.assertEqual(result["max_recorded_root_speed_m_s"],.611)
    def test_actual_navigation_call_requires_explicit_flag_and_active_root060(self):
        tree=ast.parse(Path(m.__file__).read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="physical_fault" and any(k.arg=="planar_speed_warning_only" for k in n.keywords))
        expr=next(k.value for k in call.keywords if k.arg=="planar_speed_warning_only")
        for limit,flag,expected in [(.6,True,True),(.2,True,False),(.6,False,False)]:
            result=eval(compile(ast.Expression(expr),"actual-speed-policy","eval"),dict(limit=limit,MICRO=dict(carry=dict(delivery=dict(planar_speed_warning_only=flag)))))
            self.assertEqual(result,expected)
    def test_actual_hand_owner_uses_same_explicit_scope(self):
        from scripts.research.issue46.native_runtime import stl_hand_startup
        tree=ast.parse(Path(stl_hand_startup.__file__).read_text())
        text=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "POST_LIFT_SOURCE_SAFETY" in n.value and "planar_speed_warning_only" in n.value)
        expr=ast.parse(textwrap.dedent(text)).body[1].value
        kw=next(k.value for k in expr.keywords if k.arg=="planar_speed_warning_only")
        for limit,flag,expected in [(.6,True,True),(.2,True,False),(.6,False,False)]:
            result=eval(compile(ast.Expression(kw),"actual-owner-policy","eval"),dict(limit=limit,LIFT_CONTRACT=dict(retract_preflight=dict(planar_speed_warning_only=flag))))
            self.assertEqual(result,expected)

if __name__=="__main__":unittest.main()
