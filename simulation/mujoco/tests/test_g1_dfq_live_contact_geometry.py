from pathlib import Path
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

# This test exercises contact-geometry math only; rendering is not part of the
# headless smoke dependency set.
cv2_stub = ModuleType("cv2")
cv2_stub.__version__ = "unused-by-contact-geometry-test"
sys.modules.setdefault("cv2", cv2_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_dfq_grasp_core as core


class LiveContactGeometryTests(unittest.TestCase):
    def test_contact_sequence_waits_for_stable_opposition_before_releasing_thumb(self):
        self.assertEqual(core.contact_sequence_phase({"thumb"}), "OPPOSING_APPROACH")
        self.assertTrue(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "OPPOSING_APPROACH", "thumb", True
        ))
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "OPPOSING_APPROACH", "thumb", False
        ))

    def test_contact_sequence_holds_opposition_until_thumb_is_confirmed(self):
        self.assertEqual(core.contact_sequence_phase({"index", "ring"}), "THUMB_APPROACH")
        self.assertTrue(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "THUMB_APPROACH", "index", True
        ))
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "THUMB_APPROACH", "thumb", False
        ))
        self.assertEqual(
            core.contact_sequence_phase({"thumb", "index"}), "BILATERAL_CLOSE"
        )
        self.assertFalse(core.contact_sequence_holds_target(
            "support-before-thumb-advance", "BILATERAL_CLOSE", "thumb", True
        ))

    def test_simultaneous_close_profile_does_not_apply_sequence_holds(self):
        self.assertFalse(core.contact_sequence_holds_target(
            "simultaneous", "OPPOSING_APPROACH", "thumb", True
        ))

    def test_candidate_force_requirements_allow_digits_without_allocated_contacts(self):
        thumb, opposing = core.candidate_digit_normal_requirements({
            "thumb": {"normal_force_n": 2.1},
            "index": {"normal_force_n": 2.3},
            "ring": {"normal_force_n": 0.01},
        })
        self.assertAlmostEqual(thumb, 2.1)
        self.assertAlmostEqual(opposing, 2.31)

    def test_world_pitch_reduces_measured_thumb_index_height_offset(self):
        base = np.array([
            [np.cos(np.deg2rad(60.0)), -np.sin(np.deg2rad(60.0)), 0.0],
            [np.sin(np.deg2rad(60.0)), np.cos(np.deg2rad(60.0)), 0.0],
            [0.0, 0.0, 1.0],
        ])
        candidate = core.world_pitch_test_rotation(base)
        measured_contact_delta = np.array([0.0678, -0.0126, 0.0237])
        transformed_delta = candidate @ base.T @ measured_contact_delta
        expected_z = (
            -np.sin(np.deg2rad(15.0)) * measured_contact_delta[0]
            + np.cos(np.deg2rad(15.0)) * measured_contact_delta[2]
        )
        self.assertAlmostEqual(transformed_delta[2], expected_z, places=12)
        self.assertLess(abs(expected_z), abs(measured_contact_delta[2]))

    def test_cached_snapshot_does_not_read_stale_mujoco_contact_index(self):
        model = SimpleNamespace(nv=1, geom_bodyid=np.asarray([0, 2]))
        item = {
            "side": "right_hand",
            "digit": "index",
            "distance_m": -0.00001,
            "true_normal_force_n": 0.05,
            "contact_geom1_id": 0,
            "contact_geom2_id": 1,
            "contact_frame_world": np.eye(3).tolist(),
            "contact_position_world_m": [0.1, 0.2, 0.3],
            "contact_friction_axes": [1.4, 1.4],
            "contact_observed_time_s": 4.0,
            "mujoco_contact_index": 999,
            "other_geom": "index_geom",
            "other_body": "R_index_proximal",
            "bottle_geom": "bottle_body",
        }

        def jacobian(_model, _data, jacp, _jacr, _position, _body_id):
            jacp[:] = 0.0

        with patch.object(
            core.mujoco,
            "mj_id2name",
            side_effect=lambda _model, _kind, geom_id: (
                "bottle_body" if geom_id == 0 else "index_geom"
            ),
        ), patch.object(core.mujoco, "mj_jac", side_effect=jacobian):
            geometry = core.live_contact_allocation_geometry(
                model, object(), [item], {}, {}
            )

        self.assertTrue(geometry["success"])
        self.assertEqual(geometry["contacts"][0]["distance_m"], -0.00001)
        self.assertEqual(
            geometry["contacts"][0]["contact_position_world_m"], [0.1, 0.2, 0.3]
        )
        self.assertEqual(geometry["contacts"][0]["source_contact_time_s"], 4.0)


class MuJoCoContactForceConventionTests(unittest.TestCase):
    @staticmethod
    def make_contact_fixture(bottle_first=True):
        bottle_body = """
            <body name="bottle_body">
              <freejoint name="bottle_free"/>
              <geom name="bottle_body" type="sphere" size="0.05" mass="0.57"
                    condim="3"/>
            </body>
        """
        ring_body = """
            <body name="R_ring_proximal" pos="0.09 0 0">
              <joint name="ring_joint" type="slide" axis="1 0 0" damping="0"/>
              <geom name="ring_geom" type="sphere" size="0.05" mass="0.1"
                    condim="3"/>
            </body>
        """
        first, second = (bottle_body, ring_body) if bottle_first else (ring_body, bottle_body)
        xml = f"""
        <mujoco model="contact_force_convention_test">
          <option timestep="0.00025" gravity="0 0 0" cone="elliptic"/>
          <worldbody>
            {first}
            {second}
          </worldbody>
        </mujoco>
        """
        model = core.mujoco.MjModel.from_xml_string(xml)
        data = core.mujoco.MjData(model)
        ring_joint_id = core.mujoco.mj_name2id(
            model, core.mujoco.mjtObj.mjOBJ_JOINT, "ring_joint"
        )
        ring_dof = int(model.jnt_dofadr[ring_joint_id])
        data.qvel[ring_dof] = -1.0
        core.mujoco.mj_forward(model, data)
        return model, data, ring_joint_id, ring_dof

    def test_bottle_contact_force_sign_matches_free_body_acceleration(self):
        bottle_geom_orders = set()
        for bottle_first in (True, False):
            with self.subTest(bottle_first=bottle_first):
                model, data, _, _ = self.make_contact_fixture(bottle_first)
                self.assertGreater(data.ncon, 0)

                observed = core.bottle_contacts(model, data)
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0]["side"], "right_hand")
                self.assertEqual(observed[0]["digit"], "ring")

                contact_id = int(observed[0]["mujoco_contact_index"])
                contact = data.contact[contact_id]
                bottle_geom_id = core.mujoco.mj_name2id(
                    model, core.mujoco.mjtObj.mjOBJ_GEOM, "bottle_body"
                )
                bottle_geom_orders.add(int(contact.geom1) == bottle_geom_id)
                wrench_contact = np.zeros(6)
                core.mujoco.mj_contactForce(model, data, contact_id, wrench_contact)
                geom1_is_bottle = int(contact.geom1) == bottle_geom_id
                geom2_force_world = contact.frame.reshape(3, 3).T @ wrench_contact[:3]
                expected_bottle_force = (
                    -geom2_force_world if geom1_is_bottle else geom2_force_world
                )
                np.testing.assert_allclose(
                    observed[0]["force_on_bottle_world_n"], expected_bottle_force,
                    rtol=0.0, atol=1e-10,
                )
                self.assertGreater(float(np.linalg.norm(expected_bottle_force)), 0.0)

                bottle_id = core.mujoco.mj_name2id(
                    model, core.mujoco.mjtObj.mjOBJ_BODY, "bottle_body"
                )
                bottle_joint_id = core.mujoco.mj_name2id(
                    model, core.mujoco.mjtObj.mjOBJ_JOINT, "bottle_free"
                )
                bottle_dof = int(model.jnt_dofadr[bottle_joint_id])
                bottle_mass = float(model.body_mass[bottle_id])
                np.testing.assert_allclose(
                    bottle_mass * data.qacc[bottle_dof:bottle_dof + 3],
                    expected_bottle_force, rtol=2e-7, atol=2e-7,
                )
        self.assertEqual(bottle_geom_orders, {True, False})

    def test_live_torque_map_matches_finite_difference_and_virtual_work(self):
        model, data, ring_joint_id, ring_dof = self.make_contact_fixture()
        body_id = core.mujoco.mj_name2id(
            model, core.mujoco.mjtObj.mjOBJ_BODY, "R_ring_proximal"
        )
        qpos_adr = int(model.jnt_qposadr[ring_joint_id])

        # Check the active slide DOF and its world-axis direction independently.
        base_qpos = float(data.qpos[qpos_adr])
        epsilon = 1e-6
        origin_positions = []
        for offset in (-epsilon, epsilon):
            data.qpos[qpos_adr] = base_qpos + offset
            core.mujoco.mj_forward(model, data)
            origin_positions.append(np.asarray(data.xpos[body_id], dtype=float).copy())
        data.qpos[qpos_adr] = base_qpos
        core.mujoco.mj_forward(model, data)
        finite_difference_origin_velocity = (
            origin_positions[1] - origin_positions[0]
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            finite_difference_origin_velocity, [1.0, 0.0, 0.0],
            rtol=0.0, atol=1e-8,
        )

        data.qvel[ring_dof] = -1.0
        core.mujoco.mj_forward(model, data)
        contacts = core.bottle_contacts(model, data)
        geometry = core.live_contact_allocation_geometry(
            model, data, contacts, {"ring_proximal": "ring_joint"}, {}
        )
        self.assertTrue(geometry["success"])
        self.assertEqual(geometry["channels"], ["ring_proximal"])
        self.assertEqual(len(geometry["contact_driver_torque_maps"]), 1)

        contact = data.contact[0]
        bottle_geom_id = core.mujoco.mj_name2id(
            model, core.mujoco.mjtObj.mjOBJ_GEOM, "bottle_body"
        )
        basis = (-1.0 if int(contact.geom1) == bottle_geom_id else 1.0) \
            * contact.frame.reshape(3, 3).T
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        core.mujoco.mj_jac(
            model, data, jacp, jacr, np.asarray(contact.pos, dtype=float), body_id
        )
        contact_velocity_per_unit_qdot = jacp[:, ring_dof]
        torque_map = np.asarray(geometry["contact_driver_torque_maps"][0])
        expected_map = contact_velocity_per_unit_qdot.reshape(1, 3) @ basis
        np.testing.assert_allclose(torque_map, expected_map, rtol=0.0, atol=1e-12)

        force_on_bottle_local = np.asarray([0.4, 0.1, -0.2])
        force_on_bottle_world = basis @ force_on_bottle_local
        qdot = 0.37
        generalized_effort = float(torque_map[0] @ force_on_bottle_local)
        self.assertAlmostEqual(
            generalized_effort,
            float(np.dot(force_on_bottle_world, contact_velocity_per_unit_qdot)),
            places=12,
        )
        self.assertAlmostEqual(
            generalized_effort * qdot,
            float(np.dot(force_on_bottle_world, contact_velocity_per_unit_qdot * qdot)),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
