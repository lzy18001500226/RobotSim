from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker
from simulation.mujoco.models.agibot_x2.omnipicker_sim_functional_v0_1.tests.model_signature import (
    compiled_model_signature,
)
from simulation.mujoco.models.agibot_x2.omnipicker_sim_functional_v0_1.tests.source_surfaces import (
    OriginalSTLSurfaces,
)


ROOT = Path(__file__).resolve().parents[6]
PACKAGE = ROOT / "simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1"
MODEL_ID = "X2-OmniPicker-CAD"


def _joint_id(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise AssertionError(f"missing registered joint: {name}")
    return jid


def _set_sample(model, data, row):
    root = float(row["right_claw_joint_rad"])
    narrow2 = float(row["narrow2_rad"])
    narrow3 = float(row["narrow3_rad"])
    loop = float(row["narrow_loop_rad"])
    positions = {
        "right_claw_joint": root,
        "R_hand_wide1_joint": -root,
        "R_hand_narrow2_joint": narrow2,
        "R_hand_wide2_joint": -narrow2,
        "R_hand_narrow3_joint": narrow3,
        "R_hand_wide3_joint": -narrow3,
        "R_hand_narrow_loop_joint": loop,
        "R_hand_wide_loop_joint": -loop,
    }
    for name, value in positions.items():
        jid = _joint_id(model, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return positions


def _site_gap_mm(model, data, a, b):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, b)
    if aid < 0 or bid < 0:
        raise AssertionError("missing registered D-point site")
    return float(np.linalg.norm(data.site_xpos[aid] - data.site_xpos[bid]) * 1000.0)


class StaticEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = load_x2_omnipicker(MODEL_ID, repo_root=ROOT)
        cls.model = cls.loaded.model
        cls.registry = json.loads((PACKAGE.parent / "registry.json").read_text(encoding="utf-8"))
        cls.entry = cls.registry["models"][MODEL_ID]
        cls.source_root = ROOT / "reference/AgibotTech/agibot_x2_urdf"
        cls.surfaces = OriginalSTLSurfaces(cls.model, cls.source_root)
        with (PACKAGE / "tests/fixtures/pr90_static_samples.csv").open(newline="") as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.signature_fixture = json.loads((PACKAGE / "tests/fixtures/pr90_model_signature.json").read_text(encoding="utf-8"))

    def test_portable_compiled_model_matches_frozen_pr90_signature(self):
        model_path = PACKAGE / self.entry["model_file"]
        actual = compiled_model_signature(self.model, model_path)
        expected = self.signature_fixture["signature"]
        self.assertEqual(actual, expected)
        self.assertEqual(self.signature_fixture["reference_model_sha256"], self.entry["frozen_pr90_reference_model_sha256"])
        self.assertEqual(self.signature_fixture["mujoco_version"], self.entry["mujoco_version"])

    def test_static_reference_fixtures_match_portability_hashes(self):
        portability = json.loads((PACKAGE / "portability.json").read_text(encoding="utf-8"))
        samples = PACKAGE / "tests/fixtures/pr90_static_samples.csv"
        signature = PACKAGE / "tests/fixtures/pr90_model_signature.json"
        self.assertEqual(
            hashlib.sha256(samples.read_bytes()).hexdigest(),
            portability["static_sample_fixture_sha256"],
        )
        self.assertEqual(
            samples.stat().st_size,
            portability["static_sample_fixture_bytes"],
        )
        self.assertEqual(
            hashlib.sha256(signature.read_bytes()).hexdigest(),
            portability["static_signature_fixture_sha256"],
        )

    def test_101_static_configurations_match_source_stl_aperture_and_constraints(self):
        self.assertEqual(len(self.rows), 101)
        self.assertEqual([int(float(row["command_percent"])) for row in self.rows], list(range(101)))
        coeff_n = self.entry["quartic_equalities"]["narrow"]["coefficients_ascending"]
        coeff_w = self.entry["quartic_equalities"]["wide"]["coefficients_ascending"]
        for row in self.rows:
            with self.subTest(command_percent=row["command_percent"]):
                data = mujoco.MjData(self.model)
                q = _set_sample(self.model, data, row)
                root = q["right_claw_joint"]
                wide = q["R_hand_wide1_joint"]
                residual_n = q["R_hand_narrow2_joint"] - float(np.polynomial.polynomial.polyval(root, coeff_n))
                residual_w = q["R_hand_wide2_joint"] - float(np.polynomial.polynomial.polyval(wide, coeff_w))
                self.assertAlmostEqual(root + wide, 0.0, delta=1e-9)
                self.assertAlmostEqual(residual_n, 0.0, delta=1e-8)
                self.assertAlmostEqual(residual_w, 0.0, delta=1e-8)
                d_gaps = [
                    _site_gap_mm(self.model, data, "narrow_D_jaw", "narrow_D_loop"),
                    _site_gap_mm(self.model, data, "wide_D_jaw", "wide_D_loop"),
                ]
                self.assertLessEqual(max(d_gaps), 0.01)
                source_widths = self.surfaces.five_section_widths_mm(data)
                self.assertAlmostEqual(min(source_widths), float(row["min_five_section_mm"]), delta=0.02)
                penetrations = [max(0.0, -float(c.dist) * 1000.0) for c in data.contact if c.dist < -1e-7]
                self.assertEqual(penetrations, [])
                self.assertAlmostEqual(max(d_gaps), float(row["D_gap_mm"]), delta=0.02)


if __name__ == "__main__":
    unittest.main()
