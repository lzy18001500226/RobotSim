"""Focused optional-dependency research checks; no physical success claims."""
import copy
import json
import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import issue46_urdf_pipeline as p


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.law = p.CandidateLaw()
        cls.raw = (p.UPSTREAM/'X2-Ultra_omnipicker.urdf').read_bytes()
        cls.alignment = json.loads((p.REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json').read_text())
        cls.derived, _ = p.derive_urdf(cls.raw, cls.alignment)

    def test_source_pin(self):
        self.assertEqual(p.sha(p.UPSTREAM/'X2-Ultra_omnipicker.urdf'), p.URDF_HASH)

    def test_same_previously_reviewed_candidate_samples(self):
        for r in self.law.rows:
            q = float(r['q_root_rel_rad'])
            if abs(q) > .13:
                continue
            actual, residual = self.law.branch(r['branch'], q)
            expected = -np.array([float(r[k]) for k in ('q_root_rel_rad', 'q_B_rel_rad', 'q_C_rel_rad', 'q_E_rel_rad')])
            np.testing.assert_allclose(actual, expected, atol=1e-8)
            self.assertLess(residual, 1e-8)

    def test_analytic_endpoint_has_no_linear_extrapolation(self):
        expected, residual = self.law.branch('narrow', p.Q_CLOSED)
        target = self.law.targets(0)
        self.assertAlmostEqual(target['right_claw_joint'], 0., places=12)
        self.assertAlmostEqual(target['R_hand_narrow3_joint'], expected[2], places=12)
        self.assertLess(residual, 1e-8)
        a = self.law.at_q(-.1)['R_hand_narrow3_joint']
        b = self.law.at_q(-.095)['R_hand_narrow3_joint']
        linear = a+(a-b)/(-.1+.095)*(p.Q_CLOSED+.1)
        self.assertGreater(abs(linear-expected[2]), 1e-4)

    def test_limits_mimic_and_continuity_both_directions(self):
        values = [self.law.targets(float(r)) for r in np.linspace(0, 1, 1001)]
        for r, v in zip(np.linspace(0, 1, 1001), values):
            self.assertGreaterEqual(v['right_claw_joint'], -1.)
            self.assertLessEqual(v['right_claw_joint'], 0.)
            self.assertAlmostEqual(v['R_hand_wide1_joint'], -v['right_claw_joint'], places=12)
            self.assertEqual(v, self.law.targets(float(r)))
        self.assertLess(max(abs(a[k]-b[k]) for a, b in zip(values, values[1:]) for k in a), .01)

    def test_invalid_commands_fail_closed(self):
        for r in (float('nan'), float('inf'), -.01, 1.01):
            with self.assertRaises(ValueError):
                self.law.targets(r)

    def test_no_change_to_left_hand_mount_or_root_contract(self):
        original = ET.fromstring(self.raw)
        for a in original.findall('joint'):
            name = a.get('name')
            if name not in p.FOLLOWERS and name not in {f'R_hand_{s}4_joint' for s in p.GROUPS}:
                b = self.derived.find(f'joint[@name="{name}"]')
                self.assertEqual(ET.tostring(a).strip(), ET.tostring(b).strip())

    def test_all_original_inertias_and_mesh_identity_retained(self):
        original = ET.fromstring(self.raw)
        for a in original.findall('link'):
            b = self.derived.find(f'link[@name="{a.get("name")}"]')
            ia, ib = a.find('inertial'), b.find('inertial')
            if ia is not None:
                self.assertEqual(ET.tostring(ia).strip(), ET.tostring(ib).strip())
            else:
                self.assertIsNone(ib)
            for ma, mb in zip(a.findall('.//mesh'), b.findall('.//mesh')):
                self.assertEqual(ma.get('filename').split('/')[-1], mb.get('filename').split('/')[-1])

    def test_only_six_fixed_joints_articulated(self):
        original = ET.fromstring(self.raw)
        changes = [a.get('name') for a in original.findall('joint') if a.get('type') != self.derived.find(f'joint[@name="{a.get("name")}"]').get('type')]
        self.assertEqual(set(changes), set(p.FOLLOWERS))

    def test_review_input_hashes_are_pinned(self):
        self.assertEqual(p.sha(p.SAMPLES), p.SAMPLES_HASH)
        self.assertEqual(p.sha(p.REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json'), p.ALIGNMENT_HASH)
        self.assertEqual(p.sha(p.REPO/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json'), p.DECOMPOSITION_HASH)

    def test_minimal_serialization_retains_the_derived_model(self):
        text = p.serialize_derived(self.raw, self.derived)
        reparsed = ET.fromstring(text)
        for q in (0., -.1, .2, p.Q_CLOSED):
            targets = self.law.at_q(q)
            before, after = p.fk(self.derived, targets), p.fk(reparsed, targets)
            self.assertEqual(set(before), set(after))
            for name in before:
                np.testing.assert_allclose(before[name][0], after[name][0], atol=1e-15)
                np.testing.assert_allclose(before[name][1], after[name][1], atol=1e-15)


if __name__ == '__main__':
    unittest.main()
