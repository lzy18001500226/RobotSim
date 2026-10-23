"""Local research regressions: phase coordinates, passive actuation, surfaces."""
import json
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import issue46_urdf_pipeline as p
import issue46_single_motor_travel as t


class SingleMotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=(p.UPSTREAM/'X2-Ultra_omnipicker.urdf').read_bytes()
        alignment=json.loads((p.REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json').read_text())
        cls.derived,_=p.derive_urdf(raw,alignment)
        decomp=json.loads((p.REPO/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json').read_text())
        cls.xml=p.convert(cls.derived,decomp,actuation='single-motor-experimental')
        cls.model=mujoco.MjModel.from_xml_string(ET.tostring(cls.xml,encoding='unicode'))
        cls.law=p.CandidateLaw()

    def test_exactly_one_motor_and_six_unactuated_followers(self):
        self.assertEqual(self.model.nu,1)
        joint=int(self.model.actuator_trnid[0,0])
        self.assertEqual(mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint),'right_claw_joint')
        self.assertEqual(len(self.xml.findall('./tendon/fixed')),2)
        self.assertEqual(len(self.xml.findall('./contact/exclude')),4)

    def test_phase_correct_in_direct_urdf_coordinates(self):
        for q in (-.075,0.,.1,.2,.235):
            d=mujoco.MjData(self.model)
            t.set_static(self.model,d,self.law,q)
            self.assertLess(max(abs(v)for v in t.coupled_residuals(self.model,d).values()),1e-10)
            self.assertLess(max(abs(d.efc_pos)),1e-5)
            self.assertEqual(t.equality_jacobian(self.model,d)['remaining_dofs'],1)

    def test_no_zero_offset_tendon_regression(self):
        for s in p.GROUPS:
            e=self.xml.find(f'./equality/tendon[@name="candidate_{s}_rate_eq"]')
            self.assertAlmostEqual(float(e.get('polycoef').split()[0]),p.PHASE if s=='narrow'else-p.PHASE)
        self.assertTrue(np.all(self.model.qpos0==0))

    def test_official_ranges_and_axes_unchanged(self):
        for name,limits in (('right_claw_joint',[-1,0]),('R_hand_wide1_joint',[0,1])):
            i=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            np.testing.assert_allclose(self.model.jnt_range[i],limits)
            np.testing.assert_allclose(self.model.jnt_axis[i],[0,0,-1])
        for name in p.FOLLOWERS:
            j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            self.assertFalse(self.model.jnt_limited[j])

    def test_official_limit_is_not_candidate_reachability(self):
        target,_=t.targets(self.law,.235)
        self.assertGreater(target['right_claw_joint'],-1)
        with self.assertRaises(ValueError):
            t.targets(self.law,.24)
        for q in (float('nan'),1.,-.2):
            with self.assertRaises(ValueError):t.targets(self.law,q)

    def test_ray_measures_actual_surface_not_vertices(self):
        # The ray hits the interior of each large triangle, far from vertices.
        a=np.array([[[-1.,-.01,-1.],[1.,-.01,-1.],[0.,-.01,1.]]])
        b=a.copy();b[:,:,1]=.02
        left=t.ray_surface(a,0.,0.,'narrow');right=t.ray_surface(b,0.,0.,'wide')
        self.assertAlmostEqual((right['y_m']-left['y_m'])*1000,30.)
        self.assertIsNone(t.ray_surface(a,10.,0.,'narrow'))

    def test_saved_candidate_is_not_positive_cad_measurement(self):
        values=[float(r['q_root_rel_rad'])for r in self.law.rows if r['branch']=='narrow']
        self.assertEqual((min(values),max(values)),(-.32,0.))

    def test_raw_stl_and_compiled_mesh_frames_agree(self):
        from scipy.spatial import cKDTree
        d=mujoco.MjData(self.model)
        t.set_static(self.model,d,self.law,.1)
        source=t.SourceSurfaces(self.model,self.derived)
        for side in p.GROUPS:
            gid=source.gid(side)
            raw=source.triangles(d,gid).reshape(-1,3)
            mesh=self.model.geom_dataid[gid]
            va,n=self.model.mesh_vertadr[mesh],self.model.mesh_vertnum[mesh]
            compiled=self.model.mesh_vert[va:va+n]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
            distances=cKDTree(raw).query(compiled[::max(1,n//50)])[0]
            self.assertLess(float(np.max(distances)),1e-7)

    def test_root_bias_is_independent_virtual_work_not_passive_servo(self):
        d=mujoco.MjData(self.model)
        t.set_static(self.model,d,self.law,.1)
        force,tangent=t.root_bias_compensation(self.model,d)
        energy=[]
        for sign in (-1,1):
            probe=mujoco.MjData(self.model)
            probe.qpos[:]=d.qpos+sign*1e-6*tangent
            mujoco.mj_forward(self.model,probe)
            mujoco.mj_energyPos(self.model,probe)
            energy.append(float(probe.energy[0]))
        self.assertAlmostEqual((energy[1]-energy[0])/2e-6,force,places=7)
        self.assertEqual(self.model.nu,1)

    def test_measure_stage_cannot_finalize_or_run_physics(self):
        with tempfile.TemporaryDirectory()as tmp:
            location=SimpleNamespace(native_path=Path(tmp))
            with mock.patch('sys.argv',['travel','--stage','measure','--run-dir',tmp]), \
                 mock.patch.object(p.pg,'validate_output_path',return_value=location), \
                 mock.patch.object(p.pg,'_require_reserved_directory'), \
                 mock.patch.object(t,'measure_saved_physics',return_value={})as measure, \
                 mock.patch.object(p.pg,'write_artifact_manifest')as finalize, \
                 mock.patch.object(t,'physics')as physics:
                t.main()
                measure.assert_called_once_with(Path(tmp))
                finalize.assert_not_called()
                physics.assert_not_called()

    def test_finalize_refuses_incomplete_review_package(self):
        with tempfile.TemporaryDirectory()as tmp:
            location=SimpleNamespace(native_path=Path(tmp))
            with mock.patch('sys.argv',['travel','--stage','finalize','--run-dir',tmp]), \
                 mock.patch.object(p.pg,'validate_output_path',return_value=location), \
                 mock.patch.object(p.pg,'_require_reserved_directory'), \
                 mock.patch.object(p.pg,'write_artifact_manifest')as finalize:
                with self.assertRaisesRegex(ValueError,'incomplete human review package'):t.main()
                finalize.assert_not_called()


if __name__=='__main__':unittest.main()
