"""Regress task-motion behavior without pretending to validate MC physics."""
import unittest
from scripts.research.issue46.native_runtime.far_observe_approach import forward_turn_command,stopping_distance,stance_proposals
from scripts.research.issue46.native_runtime.native_task_grasp_region import build_motion_script,reference_contract

class FarApproachTests(unittest.TestCase):
    def test_align_keeps_explicit_forward_demand(self):
        p={"x":.5275,"y":-1.413,"yaw":.2844,"vx":.2,"vy":0.}
        v=forward_turn_command(p,[.907,-1.387,0.],0.,-.1955)
        self.assertGreaterEqual(v[0],.12);self.assertLess(v[2],0.);self.assertEqual(v[1],0.)
    def test_rightward_approach_uses_existing_turn_envelope(self):
        p={"x":.55,"y":-1.41,"yaw":.25,"vx":0.,"vy":0.}
        v=forward_turn_command(p,[.967,-1.588,-.4],0.,-.4)
        self.assertGreaterEqual(v[2],-.60);self.assertLess(v[2],-.4)
        mid=forward_turn_command(p,[.907,-1.387,0.],0.,-.2)
        self.assertGreater(mid[2],-.40)
    def test_predictive_stop_uses_actual_velocity(self):
        p={"x":0.,"y":0.,"yaw":0.,"vx":.2,"vy":0.}
        self.assertAlmostEqual(stopping_distance(p,[1.,0.,0.]),.22)
        p["vx"]=-.2;self.assertEqual(stopping_distance(p,[1.,0.,0.]),.06)
    def test_left_endpoint_acquires_coarse_heading_before_forward_space(self):
        p={"x":.55,"y":-1.53,"yaw":.18,"vx":0.,"vy":0.}
        v=forward_turn_command(p,[.947,-.995,.93],0.,.4)
        self.assertEqual(v[0],0.);self.assertEqual(v[1],0.)
        self.assertGreater(v[2],.4);self.assertLessEqual(v[2],.60)
        p["yaw"]=.70
        self.assertGreaterEqual(forward_turn_command(p,[.947,-.995,.93],.1,.2)[0],.12)
    def test_input_depends_on_observed_position_not_grid_id(self):
        a=stance_proposals([.5,-1.4],[.52,-1.4],0.,[1.426,-1.39,.657])
        b=stance_proposals([.5,-1.4],[.52,-1.4],0.,[1.426,-1.31,.657])
        self.assertEqual(a[0][1],"CURRENT_STANCE");self.assertNotEqual(next(x[0].tolist() for x in a if x[1]=="FAR_DIAGONAL_0"),next(x[0].tolist() for x in b if x[1]=="FAR_DIAGONAL_0"))
    def test_right_station_does_not_disappear_when_approaching(self):
        import numpy as np
        center=[1.427,-1.787,.657]
        far=stance_proposals([.52,-1.51],[.52,-1.51],0.,center)
        near=stance_proposals([.889,-1.602],[.889,-1.602],0.,center)
        a=next(p[0] for p in far if p[1]=="FAR_DIAGONAL_0")
        b=next(p[0] for p in near if p[1]=="FAR_DIAGONAL_0")
        np.testing.assert_allclose(a,b,atol=1e-12)
    def test_candidate_final_yaw_is_not_forced_to_instantaneous_travel_bearing(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        anchor=np.array([.86,-1.47]);root=np.array([.892,-1.454]);yaw=.597
        candidates=stance_proposals(anchor,root,yaw,[1.427,-.995,.657])
        old=next(x for x in candidates if x[1]=="FAR_DIAGONAL_1")
        new=next(x for x in candidates if x[1]=="FAR_DIAGONAL_1_CURRENT_HEADING")
        target_old=old[0]-Rotation.from_euler("z",old[2]-yaw).as_matrix()[:2,:2]@(anchor-root)
        target_new=new[0]-(anchor-root)
        np.testing.assert_allclose(target_old,target_new,atol=1e-12)
        self.assertEqual(new[2],yaw);self.assertGreater(abs(old[2]-yaw),.5)
    def test_mode_ack_cannot_replace_applied_mode_and_source(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import locomotion_source_observed
        row={"common_source":{"name":"robotsim_official_cmd_vel"},"common_mode_desc":"LOCOMOTION_STEP","get_mode_desc":"STAND_DEFAULT","common_mode_status_value":100,"get_mode_status_value":100}
        self.assertFalse(locomotion_source_observed(row,"LOCOMOTION_STEP"))
        row["get_mode_desc"]="LOCOMOTION_STEP"
        self.assertTrue(locomotion_source_observed(row,"LOCOMOTION_STEP"))
        row["common_source"]["name"]="other_owner"
        self.assertFalse(locomotion_source_observed(row,"LOCOMOTION_STEP"))
    def test_nonzero_triggered_mode_can_be_verified_after_short_command(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import locomotion_cycle_proof
        sd={"sample_monotonic_s":10.,"common_source":{"name":"robotsim_official_cmd_vel"},"common_mode_desc":"STAND_DEFAULT","get_mode_desc":"STAND_DEFAULT","common_mode_status_value":100,"get_mode_status_value":100}
        step=dict(sd,sample_monotonic_s=10.4,common_mode_desc="LOCOMOTION_STEP",get_mode_desc="LOCOMOTION_STEP")
        self.assertIsNone(locomotion_cycle_proof([sd],10.,"LOCOMOTION_STEP"))
        self.assertIsNotNone(locomotion_cycle_proof([sd,step],10.,"LOCOMOTION_STEP"))
        self.assertIsNone(locomotion_cycle_proof([step],11.,"LOCOMOTION_STEP"))
    def test_only_hanging_hand_travel_collision_triggers_forearm_recovery(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import forearm_clearance_recovery_needed
        def candidate(a,b):return {"stage":"NAVIGATION_PATH_COLLISION","navigation_preflight":{"screen":{"violations":[{"bodies":[a,b]}]}}}
        self.assertTrue(forearm_clearance_recovery_needed([candidate("R_hand_narrow3_Link","raicom_table_table")]))
        self.assertTrue(forearm_clearance_recovery_needed([candidate("test_cylinder","R_hand_wide3_Link")]))
        self.assertFalse(forearm_clearance_recovery_needed([candidate("right_ankle_roll_link","raicom_table_table")]))
        self.assertFalse(forearm_clearance_recovery_needed([{"stage":"IK_UNRESOLVED"}]))
    def test_lateral_error_cannot_drive_past_forward_table_station(self):
        from scripts.research.issue46.native_runtime.far_observe_approach import forward_table_plane_reached
        pose={"x":.80,"vx":.20}
        self.assertTrue(forward_table_plane_reached(pose,[.947,-.995],[.52,-1.54]))
        self.assertFalse(forward_table_plane_reached({"x":.52,"vx":0.},[.947,-.995],[.52,-1.54]))
        self.assertFalse(forward_table_plane_reached(pose,[.947,-1.53],[.52,-1.54]))
    def test_compact_recovery_covers_preparation_sweep_but_not_hip_collision(self):
        from scripts.research.issue46.native_runtime.native_task_grasp_region import forearm_clearance_recovery_needed
        hand={"stage":"ACTUAL_PREPARATION_COLLISION","actual_preparation_preflight":{"stage":"waist","pair":["test_cylinder","R_hand_wide3_Link"]}}
        hip={"stage":"ACTUAL_PREPARATION_COLLISION","actual_preparation_preflight":{"stage":"lower","pair":["right_hip_yaw_link","raicom_table_table"]}}
        self.assertTrue(forearm_clearance_recovery_needed([hand]))
        self.assertFalse(forearm_clearance_recovery_needed([hip]))
    def test_tracking_reserve_stays_inside_source_limits(self):
        import numpy as np
        from scripts.research.issue46.grasp.stl_approved_circle_ik import position_mode_tracking_bounds
        source=np.array([[-2.0,.061]])
        old=position_mode_tracking_bounds(source,[-1.97],[100.],margin=.01)
        new=position_mode_tracking_bounds(source,[-1.97],[100.],margin=.02)
        self.assertAlmostEqual(old[0,1],.051)
        self.assertAlmostEqual(new[0,1],.041)
        transient=.06429233142650555-.04983438135009828
        self.assertGreater(old[0,1]+transient,source[0,1])
        self.assertLess(new[0,1]+transient,source[0,1])
        np.testing.assert_array_equal(source,[[-2.,.061]])
    def test_intermediate_transit_does_not_require_full_grasp_at_waypoint(self):
        from scripts.research.issue46.native_runtime.far_observe_approach import intermediate_forward_waypoint
        import numpy as np
        np.testing.assert_allclose(intermediate_forward_waypoint([.85044,-1.48275],[.97044,-1.48275]),[.89044,-1.48275])
        self.assertIsNone(intermediate_forward_waypoint([.5,-1.5],[.947,-1.0]))
        self.assertIsNone(intermediate_forward_waypoint([.9,-1.5],[.94,-1.5]))
        self.assertIsNone(intermediate_forward_waypoint([.9,-1.5],[.85,-1.5]))
        with self.assertRaises(ValueError):intermediate_forward_waypoint([float("nan"),0.],[0.,0.])
    def test_new_contract_and_generated_script_remain_separate(self):
        self.assertNotEqual(reference_contract(),reference_contract(far_observe=True))
        s=build_motion_script([.9,-1.3,0.],{},[],{},"test",point_goal=True,locomotion_action="LOCOMOTION_STEP")
        compile(s,"generated","exec");self.assertIn("forward_turn_command(pose",s)
        self.assertIn("'locomotion_action': 'LOCOMOTION_STEP'",s)

if __name__=="__main__":unittest.main()
