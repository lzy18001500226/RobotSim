import unittest

from issue46_native_omnihand_model_fidelity_audit import VISUAL_CLOSE_POSE, geom_role


class NativeOmniHandModelFidelityAuditTests(unittest.TestCase):
    def test_only_zero_collision_masks_classify_a_geom_as_visual(self):
        self.assertEqual(geom_role(0, 0), "visual")
        self.assertEqual(geom_role(1, 0), "collision")
        self.assertEqual(geom_role(0, 1), "collision")
        self.assertEqual(geom_role(1, 1), "collision")

    def test_visual_close_pose_includes_all_five_finger_families(self):
        targets = VISUAL_CLOSE_POSE
        self.assertIn("R_thumb_mcp_joint", targets)
        self.assertIn("R_index_pip_joint", targets)
        self.assertIn("R_middle_pip_joint", targets)
        self.assertIn("R_ring_pip_joint", targets)
        self.assertIn("R_pinky_pip_joint", targets)


if __name__ == "__main__":
    unittest.main()
