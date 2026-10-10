from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scripts.research.issue46_mjstep_thumb_transmission import (
    DEFAULT_VENDOR_ROOT,
    DRIVER,
    FOLLOWERS,
    build_model,
    controller_torque,
    source_records,
    verify_source,
)


class MjStepThumbTransmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_urdf = verify_source(DEFAULT_VENDOR_ROOT)

    def test_minimal_model_keeps_official_thumb_and_only_driver_actuated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model, info = build_model(self.source_urdf, Path(temp))
        self.assertEqual((model.nq, model.nv, model.nu, model.neq), (5, 5, 1, 4))
        self.assertGreaterEqual(model.nmesh, 5)
        self.assertEqual(info["actuator_joint_names"], [DRIVER])
        self.assertEqual({r["follower"] for r in info["mimic_relations"]}, set(FOLLOWERS))
        self.assertEqual(info["fixture_contact_exclusions"], [["R_palm", "R_thumb_mcp"]])
        driver_aid = info["driver_actuator_id"]
        driver_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DRIVER)
        self.assertEqual(int(model.actuator_trnid[driver_aid, 0]), driver_jid)
        np.testing.assert_allclose(model.actuator_gear[driver_aid], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(model.actuator_forcerange[driver_aid], [-0.08, 0.08])
        np.testing.assert_allclose(model.eq_solref, np.tile([-10000.0, -200.0], (4, 1)))
        for name in (DRIVER, *FOLLOWERS, "R_thumb_roll_joint", "R_thumb_abad_joint"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            np.testing.assert_allclose(model.jnt_solref[jid], [-10000.0, -200.0])
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 0)

    def test_controller_writes_only_control_and_mj_step_advances_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model, info = build_model(self.source_urdf, Path(temp))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        qpos_before = data.qpos.copy()
        for _ in range(30):
            mujoco.mj_forward(model, data)
            state_qpos, state_qvel = data.qpos.copy(), data.qvel.copy()
            controller_torque(model, data, info, 0.2)
            np.testing.assert_array_equal(data.qpos, state_qpos)
            np.testing.assert_array_equal(data.qvel, state_qvel)
            self.assertTrue(np.isfinite(data.ctrl).all())
            self.assertTrue(np.all(data.eq_active))
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
        self.assertFalse(np.allclose(data.qpos, qpos_before))
        source_joints, relations = source_records(ET.parse(self.source_urdf).getroot())
        maps = info["source_joint_mappings"]
        q_driver = maps[DRIVER]["source_ref"] + maps[DRIVER]["sign"] * (data.qpos[maps[DRIVER]["qadr"]] - maps[DRIVER]["qpos0"])
        for follower in FOLLOWERS:
            mapping = maps[follower]
            q_follower = mapping["source_ref"] + mapping["sign"] * (data.qpos[mapping["qadr"]] - mapping["qpos0"])
            relation = relations[follower]
            residual = q_follower - relation["offset"] - relation["multiplier"] * q_driver
            self.assertTrue(np.isfinite(residual))
            self.assertTrue(np.isfinite(residual))
            self.assertIsNotNone(source_joints[follower])

    def test_unexcluded_reference_pose_reports_the_palm_mcp_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model, info = build_model(
                self.source_urdf,
                Path(temp),
                exclude_fixture_root_contact=False,
            )
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        body_pairs = {
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[data.contact[i].geom1])),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[data.contact[i].geom2])),
            )
            for i in range(data.ncon)
        }
        self.assertEqual(body_pairs, {("R_palm", "R_thumb_mcp")})
        self.assertAlmostEqual(float(data.contact[0].dist), -0.001308099962, places=9)
        mcp = info["source_joint_mappings"][DRIVER]
        self.assertAlmostEqual(float(data.qpos[mcp["qadr"]]), mcp["source_range"][0], places=12)
        self.assertFalse(any(int(kind) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT) for kind in data.efc_type))


if __name__ == "__main__":
    unittest.main()
