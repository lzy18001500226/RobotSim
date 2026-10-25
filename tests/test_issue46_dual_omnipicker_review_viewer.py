import unittest

import mujoco

from scripts.research.issue46_dual_omnipicker_review_viewer import (
    DISPLAY_COLLISION,
    DISPLAY_ENVIRONMENT,
    DISPLAY_LEFT_GRIPPER,
    DISPLAY_RIGHT_GRIPPER,
    DISPLAY_VISUAL,
    POSES,
    Pose,
    physical_fingerprint,
    select_display_groups,
    set_review_display_groups,
    static_joint_values,
)


class ReviewViewerTests(unittest.TestCase):
    def test_supported_pose_map_is_mirrored_from_saved_sample(self):
        self.assertEqual(POSES, (2.0, 50.0, 100.0))
        pose = Pose(50.0, -0.342, -0.317, 0.680, -0.527, 61.4, 0.00001)
        values = static_joint_values(pose)
        self.assertEqual(values["left_claw_joint"], pose.root_q_rad)
        self.assertEqual(values["right_claw_joint"], pose.root_q_rad)
        self.assertEqual(values["L_hand_wide1_joint"], -pose.root_q_rad)
        self.assertEqual(values["R_hand_wide1_joint"], -pose.root_q_rad)
        self.assertEqual(values["L_hand_narrow3_joint"], pose.narrow3_rad)
        self.assertEqual(values["R_hand_wide_loop_joint"], -pose.narrow_loop_rad)
        self.assertNotIn(0.0, POSES)

    def test_render_group_assignment_preserves_physical_arrays(self):
        xml = """<mujoco>
          <compiler inertiafromgeom="true"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" contype="1" conaffinity="1"/>
            <geom name="body_visual" type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
            <body name="L_hand_test">
              <joint name="left_joint" type="hinge" axis="0 0 1"/>
              <geom name="left_visual" type="sphere" size="0.05" contype="0" conaffinity="0"/>
              <geom name="left_collision" type="sphere" pos="0.1 0 0" size="0.05" contype="1" conaffinity="1"/>
            </body>
            <body name="R_hand_test">
              <joint name="right_joint" type="hinge" axis="0 0 1"/>
              <geom name="right_visual" type="sphere" size="0.05" contype="0" conaffinity="0"/>
            </body>
            <body name="m0_table"><geom name="table" type="box" size="0.2 0.2 0.02" contype="1" conaffinity="1"/></body>
            <body name="canonical_bottle"><geom name="bottle" type="cylinder" size="0.03 0.1" contype="1" conaffinity="1"/></body>
          </worldbody>
        </mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        before = physical_fingerprint(model)
        info = set_review_display_groups(model)
        self.assertEqual(before, physical_fingerprint(model))
        self.assertEqual(info["physical_fingerprint_before"], info["physical_fingerprint_after"])
        roles = {}
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            roles[name] = int(model.geom_group[geom_id])
        self.assertEqual(roles["body_visual"], DISPLAY_VISUAL)
        self.assertEqual(roles["left_visual"], DISPLAY_LEFT_GRIPPER)
        self.assertEqual(roles["right_visual"], DISPLAY_RIGHT_GRIPPER)
        self.assertEqual(roles["left_collision"], DISPLAY_COLLISION)
        self.assertEqual(roles["floor"], DISPLAY_ENVIRONMENT)
        self.assertEqual(roles["table"], DISPLAY_ENVIRONMENT)
        self.assertEqual(roles["bottle"], DISPLAY_ENVIRONMENT)

    def test_render_modes_change_view_option_not_model(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
        )
        before = physical_fingerprint(model)
        opt = mujoco.MjvOption()
        for mode in ("VISUAL_ONLY", "COLLISION_ONLY", "OVERLAY"):
            select_display_groups(opt, mode)
            self.assertEqual(before, physical_fingerprint(model))
        self.assertTrue(opt.geomgroup[DISPLAY_VISUAL])
        self.assertTrue(opt.geomgroup[DISPLAY_COLLISION])
        self.assertTrue(opt.geomgroup[DISPLAY_ENVIRONMENT])


if __name__ == "__main__":
    unittest.main()
