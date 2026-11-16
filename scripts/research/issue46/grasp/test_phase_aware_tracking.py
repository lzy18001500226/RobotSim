"""Phase-specific task accuracy and released-hand regression; no physics."""
import unittest
import numpy as np
from scripts.research.issue46.grasp import stl_approved_circle_ik as ik


class PhaseAwareTrackingTests(unittest.TestCase):
    def test_approved_staged_axis_accepts_recorded_noise_but_not_large_error(self):
        position,axis=ik.tracking_task_limits({"approach_staging":{"pass_result":True}},None)
        self.assertEqual(position,.001)
        self.assertLess(.0010173472336489188,axis)
        self.assertGreater(.2*np.sin(np.deg2rad(2.)/2.),axis)
        self.assertAlmostEqual(2.*np.arcsin(axis/.2),np.deg2rad(1.))

    def test_unstaged_and_placement_contracts_remain_separate(self):
        self.assertEqual(ik.tracking_task_limits({},None),(.001,.001))
        phase={"placement_active":True,"placement_request":{
            "placement_position_tolerance_m":.005,"placement_axis_tolerance_rad":np.deg2rad(1.)}}
        self.assertEqual(ik.tracking_task_limits({},phase)[0],.005)
        phase["placement_request"]["placement_position_tolerance_m"]=.006
        with self.assertRaises(RuntimeError):ik.tracking_task_limits({},phase)

    def proof(self):
        return dict(placement_active=True,
            placement_execution_timing={"stage":"OPEN_CLAW_WITHDRAWAL"},
            placement_withdrawal_support=dict(pass_result=True,generation=3,
                measured_hold_sim_s=2.04,
                classification="RELEASED_BOTTLE_SUPPORTED_BY_DELIVERY_TABLE_NOT_GRIP"))

    def row(self):
        return dict(generation=3,joints={"R_claw_joint":{"q":-.99}})

    def test_axis_free_only_after_independent_release_and_real_open(self):
        self.assertTrue(ik.released_withdrawal_axis_free(self.row(),self.proof()))
        self.assertFalse(ik.released_withdrawal_axis_free(self.row(),{}))
        for field,value in [("pass_result",False),("generation",2),
                            ("measured_hold_sim_s",1.99),("measured_hold_sim_s",float("nan")),("classification","GRIP")]:
            phase=self.proof();phase["placement_withdrawal_support"][field]=value
            with self.assertRaises(RuntimeError):ik.released_withdrawal_axis_free(self.row(),phase)
        row=self.row();row["joints"]["R_claw_joint"]["q"]=-.5
        with self.assertRaises(RuntimeError):ik.released_withdrawal_axis_free(row,self.proof())
        phase=self.proof();phase["placement_execution_timing"]["stage"]="LOWER_UNTIL_TABLE_SUPPORT"
        with self.assertRaises(RuntimeError):ik.released_withdrawal_axis_free(self.row(),phase)




class LowReleaseRegionTests(unittest.TestCase):
    def phase(self):
        return dict(placement_active=True,
            placement_execution_timing=dict(stage="LOWER_TO_TOOL_ESTIMATED_RELEASE_BAND"),
            placement_request=dict(placement_release_policy="TOOL_ESTIMATED_LOW_DROP",
                placement_release_height_m=.02,generation=1,
                grasp_height_above_center_m=.01,bottle_height_m=.22,bottle_radius_m=.035,
                tabletop_world_z_m=.547,table_bounds_min_world_m=[-.45,1.32,0.],
                table_bounds_max_world_m=[.45,1.88,.547],maximum_planned_bottle_tilt_rad=np.deg2rad(16.5)))

    def test_release_region_accepts_multiple_positions_and_tilts_without_point_residual(self):
        from scripts.research.issue46.native_runtime.native_carry_placement import low_release_center_height
        phase=self.phase();cfg=phase["placement_request"]
        for gap in (.01,.02,.03):
            for x in (-.3,0.,.3):
                for tilt in (0.,10.,16.):
                    axis=np.array([0.,np.sin(np.deg2rad(tilt)),np.cos(np.deg2rad(tilt))])
                    z=low_release_center_height({**cfg,"placement_release_height_m":gap},axis)+.01*axis[2]
                    # Interior heights avoid binary-rounding ambiguity at endpoints.
                    if gap==.01:z+=1e-9
                    if gap==.03:z-=1e-9
                    result=ik.placement_release_region([x,1.6,z],axis,phase)
                    self.assertIsNotNone(result)
                    self.assertAlmostEqual(result["bottom_gap_m"],gap,places=7)

    def test_wrong_stage_legacy_policy_off_table_or_bad_height_is_not_region_acceptance(self):
        phase=self.phase();point=[0.,1.6,.687];axis=[0.,0.,1.]
        self.assertIsNotNone(ik.placement_release_region(point,axis,phase))
        for stage in ("RAISE_TO_CLEARANCE","TRANSLATE_ABOVE_TABLE","MEASURED_APPROACH","OPEN_CLAW_WITHDRAWAL"):
            phase["placement_execution_timing"]["stage"]=stage
            self.assertIsNone(ik.placement_release_region(point,axis,phase))
        phase=self.phase()
        for point in ([0.,1.30,.687],[.44,1.6,.687],[0.,1.6,.65],[0.,1.6,.75]):
            self.assertIsNone(ik.placement_release_region(point,axis,phase))
        phase["placement_request"]["placement_release_policy"]="MEASURED_TABLE_SUPPORT"
        self.assertIsNone(ik.placement_release_region([0.,1.6,.687],axis,phase))
        self.assertIsNone(ik.placement_release_region([0.,1.6,.687],axis,{}))

if __name__=="__main__":unittest.main()
