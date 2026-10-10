import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.research import issue46_dual_omnihand_transmission_probe as probe
from scripts.research import issue46_mimic_constraint_isolation as isolation


class MimicConstraintIsolationTests(unittest.TestCase):
    def test_self_contact_filter_rejects_only_hand_to_hand_pairs(self):
        model = SimpleNamespace(geom_bodyid=np.array([11, 22, 33]))
        body_names = {11: "L_thumb_pip_link", 22: "R_middle_pip_link", 33: "m0_table_top"}

        def body_name(_model, _obj, body_id):
            return body_names[body_id]

        with patch.object(isolation.mujoco, "mj_id2name", side_effect=body_name):
            self.assertEqual(isolation.disable_hand_self_contact(model, None, 0, 1), 1)
            self.assertEqual(isolation.disable_hand_self_contact(model, None, 0, 2), 0)
            self.assertEqual(isolation.disable_hand_self_contact(model, None, 2, 2), 0)

    def test_constraint_correction_scales_only_compiled_mimic_solref(self):
        model = SimpleNamespace(
            neq=12,
            eq_solref=np.tile([-10000.0, -200.0], (12, 1)),
        )
        result = probe.apply_sim_only_mimic_constraint_correction(model)
        self.assertEqual(result["classification"], "SIMULATION_ONLY constraint regularization; not hardware-equivalent")
        self.assertEqual(result["scale"], 2.0)
        np.testing.assert_array_equal(model.eq_solref, np.tile([-20000.0, -400.0], (12, 1)))

    def test_gate_and_diagnostic_window_remain_fixed(self):
        self.assertEqual(probe.MIMIC_LIMIT_RAD, 0.003)
        self.assertEqual(isolation.STEPS, 5)
        self.assertEqual(isolation.STEPS * probe.DT, 0.010)
        self.assertEqual(
            isolation.CONFIGURATIONS,
            ("baseline", "gravity_disabled", "hand_self_contact_disabled", "driver_actuation_disabled"),
        )


if __name__ == "__main__":
    unittest.main()
