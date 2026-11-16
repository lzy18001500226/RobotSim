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


if __name__=="__main__":unittest.main()
