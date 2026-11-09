"""Focused table-support and existing-owner extension checks; no physics."""
import unittest
from scripts.research.issue46.native_runtime import native_carry_placement as p

class PlacementTests(unittest.TestCase):
    def request(self):
        return dict(generation=1,bottle_height_m=.30,bottle_radius_m=.035,bottle_mass_kg=.2,
                    table_bounds_min_world_m=[-.45089,1.32174,-.011],table_bounds_max_world_m=[.45089,1.87825,.54683],tabletop_world_z_m=.54683)
    def row(self,i,pad=0.,side="raicom_delivery_table"):
        return dict(host_monotonic_ns=1000000000+i*40000000,sim_time_s=10.+i*.04,generation=1,
                    bodies={"test_cylinder":dict(position_m=[.15,1.52,.69683],rotation=[1.,0.,0.,0.,1.,0.,0.,0.,1.],origin_velocity_world_angular_linear=[0.]*6)},
                    contacts=[dict(efc_address=0,body1=side,body2="test_cylinder",local_force_torque=[1.962,0.,0.,0.,0.,0.],signed_bottle_world_z_force_n=1.962),
                              dict(efc_address=1,body1="R_hand_narrow3_Link",body2="test_cylinder",local_force_torque=[pad,0.,0.,0.,0.,0.],signed_bottle_world_z_force_n=0.)])
    def test_actual_released_support_contract_is_separate_from_grip(self):
        result=p.evaluate_placed_hold([self.row(i) for i in range(52)],self.request())
        self.assertTrue(result["pass_result"]);self.assertGreaterEqual(result["measured_hold_sim_s"],2.)
        for rows in ([self.row(i,pad=2.) for i in range(52)],[self.row(i,side="world") for i in range(52)],[self.row(i) for i in range(20)]):
            self.assertFalse(p.evaluate_placed_hold(rows,self.request())["pass_result"])
    def test_stale_duplicate_or_off_table_data_cannot_pass(self):
        rows=[self.row(i) for i in range(52)];rows[-1]=rows[-2]
        self.assertFalse(p.evaluate_placed_hold(rows,self.request())["pass_result"])
        rows=[self.row(i) for i in range(52)]
        for r in rows:r["bodies"]["test_cylinder"]["position_m"][1]=1.30
        self.assertFalse(p.evaluate_placed_hold(rows,self.request())["pass_result"])
    def test_opt_in_preserves_the_existing_owner_and_has_no_motion_or_pose_write_path(self):
        from scripts.research.issue46.native_runtime.test_native_grasp_launch import PostLiftRetractTests
        normal=PostLiftRetractTests().generated_retract(dict(carry_lease_s=140.))
        code=PostLiftRetractTests().generated_retract(dict(carry_lease_s=140.,delivery_table_placement=True,robot_bodies=["pelvis"]))
        compile(code,"actual_placement_owner","exec")
        self.assertNotIn("perform_native_placement",normal)
        self.assertIn("perform_native_placement(n,request)",code)
        self.assertIn("PLACEMENT_PHYSICAL_SAFETY",code)
        self.assertIn("TABLE_SUPPORT_LOST_BEFORE_CLOSEOUT",code)
        self.assertNotIn("create_publisher(Twist",code)
        self.assertNotIn("mj_step(",code)
        self.assertNotIn(".qpos[",code)

    def test_final_mode_observer_has_no_control_or_velocity_interface(self):
        import ast
        code=p.placement_mode_observer_script();compile(code,"placement_observe_only","exec")
        imports=[n.module for n in ast.walk(ast.parse(code)) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn("SetMcAction",code)
        self.assertNotIn("create_publisher",code)
        self.assertNotIn("SetMcInputSource",code)
        self.assertIn("GetMcAction",code)
        self.assertIn("McCommonState",code)

    def test_neutral_stow_and_placement_share_existing_owner_and_release_contract(self):
        from scripts.research.issue46.native_runtime.test_native_grasp_launch import PostLiftRetractTests
        from scripts.research.issue46.native_runtime.native_carry_posture import native_stand_arm_reference
        settings=native_stand_arm_reference()
        self.assertEqual(len(settings["arm_target_rad"]),14)
        self.assertEqual(settings["duration_s"],6.)
        code=PostLiftRetractTests().generated_retract(dict(carry_lease_s=160.,delivery_table_placement=True,
             carry_native_neutral_stow=settings,robot_bodies=["pelvis"]))
        compile(code,"stow_and_placement_owner","exec")
        self.assertEqual(code.count("carry_target=perform_native_neutral_stow(n,NATIVE_STOW_SETTINGS)"),1)
        force_line=next(l for l in code.splitlines() if "if (sum(forces.values()) if stage" in l)
        self.assertIn("placement_release",force_line)
        self.assertIn("STOW_PHYSICAL_SAFETY",code)
        self.assertIn("STOW_TRACKER_REVISION_UNCONFIRMED",code)
        self.assertIn("TABLE_SUPPORT_LOST_BEFORE_CLOSEOUT",code)
        self.assertNotIn(".qpos[",code)
        self.assertNotIn("mj_step(",code)

    def test_neutral_stow_validates_all_arm_limits_before_first_publication(self):
        import ast,math
        from pathlib import Path
        from scripts.research.issue46.native_runtime.native_carry_posture import native_stand_arm_reference,perform_native_neutral_stow
        source=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46/raicom-stl-xvfb-native-reset-stand-20261108T215555Z/native_hal_bilateral_arm_pregrasp.py")
        values={}
        for node in ast.parse(source.read_text()).body:
            if isinstance(node,ast.Assign):
                for name in node.targets:
                    if isinstance(name,ast.Name) and name.id in ("LIFT_CONTRACT","LEFT","RIGHT"):
                        values[name.id]=ast.literal_eval(node.value)
        settings=native_stand_arm_reference()
        class BeforePublish(Exception):pass
        class Node:
            def refresh_native(self):raise BeforePublish()
        scope={**values,"math":math}
        import inspect
        exec(inspect.getsource(perform_native_neutral_stow),scope)
        with self.assertRaises(BeforePublish):scope["perform_native_neutral_stow"](Node(),settings)

    def test_pelvis_carry_reference_does_not_follow_independent_waist_sway(self):
        import numpy as np
        from scripts.research.issue46.grasp.stl_approved_circle_ik import tracking_task
        identity=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        row=dict(generation=1,bodies={"pelvis":dict(position_m=[1.,2.,.6],rotation=identity),
             "torso_link":dict(position_m=[1.10,2.,.8],rotation=identity),
             "R_omnipicker_base_link":dict(position_m=[1.2,2.,.8],rotation=identity),
             "test_cylinder":dict(position_m=[1.2,2.,.7],rotation=identity)})
        phase=dict(phase="CARRY_HOLD",native_arrival_pass=True,post_retract_physical_hold_pass=True,generation=1,
             carry_heading_xy_m=[.2,0.],carry_world_z_m=.8,initial_object_in_hand_m=[0.,0.,0.],slip_reference="GRASP_SECTION",
             carry_retention_only=True,allow_carried_bottle_tilt=True,carry_pose_frame="PELVIS_RIGID",
             carry_rigid_offset_base_m=[.2,0.,.2],carry_rigid_axis_base=[0.,0.,1.])
        target,axis=tracking_task(row,{"grasp_height_above_center_m":.1},phase)
        np.testing.assert_allclose(target,[1.2,2.,.8])
        row["bodies"]["torso_link"]["position_m"]=[1.2,2.,.55]
        again,_=tracking_task(row,{"grasp_height_above_center_m":.1},phase)
        np.testing.assert_allclose(again,target)
        phase["carry_pose_frame"]="TORSO_RIGID"
        torso,_=tracking_task(row,{"grasp_height_above_center_m":.1},phase)
        self.assertGreater(np.linalg.norm(torso-target),.1)
