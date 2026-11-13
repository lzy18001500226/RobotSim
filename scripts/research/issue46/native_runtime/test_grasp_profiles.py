import unittest
import numpy as np
from scripts.research.issue46.native_runtime.grasp_profiles import load_profiles,select_profile,stance_from_profile

class GraspProfileTests(unittest.TestCase):
    def test_all11_references_have_original_evidence_not_current_pass_claim(self):
        profiles=load_profiles();self.assertEqual(len(profiles),11)
        self.assertEqual(len({p['profile'] for p in profiles}),11)
        for p in profiles:
            self.assertEqual(p['provenance']['scope'],'HISTORICAL_SINGLE_GRASP_PASS_NOT_CURRENT_VERSION_TRANSFER_PASS')
            self.assertEqual(len(p['provenance']['state_sha256']),64)
    def test_selection_depends_on_observation_not_test_case_id(self):
        for p in load_profiles():
            selected=select_profile([1.43,p['reference_object_y_world_m'],.66])
            self.assertEqual(selected['profile'],p['profile'])
    def test_foot_anchor_transform_reconstructs_requested_actual_root(self):
        profile=load_profiles()[10];center=np.array([1.43,-1.,.66]);root=np.array([.9,-1.4]);foot=root+[-.02,.01];yaw=.2
        anchor,heading=stance_from_profile(profile,center,root,foot,yaw)
        c,s=np.cos(heading-yaw),np.sin(heading-yaw)
        reconstructed=anchor+np.array([[c,-s],[s,c]])@(root-foot)
        np.testing.assert_allclose(reconstructed,center[:2]+profile['root_offset_from_object_xy_m'])
    def test_reference_torso_heading_survives_base_arrival_yaw_error(self):
        import math
        from scripts.research.issue46.native_runtime.grasp_profiles import torso_world_heading
        profile=load_profiles()[10];heading=torso_world_heading(profile)
        for actual_base_yaw in (.4955949,.6522245,profile['base_yaw_rad']):
            waist=math.atan2(math.sin(heading-actual_base_yaw),math.cos(heading-actual_base_yaw))
            self.assertAlmostEqual(actual_base_yaw+waist,heading)
        self.assertAlmostEqual(heading,.9514027919586301)

    def test_invalid_observation_is_rejected(self):
        for value in ([0.,0.],[0.,float('nan'),1.]):
            with self.assertRaises(ValueError):select_profile(value)

if __name__=='__main__':unittest.main()
