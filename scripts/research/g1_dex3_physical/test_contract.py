"""Focused checks for source mapping, force signs and no object-state control."""
import ast, tempfile, unittest
from baseline import *
from episode import (SOURCE_LIMITS, bottle_contact, close_targets, evaluate_readiness_gates,
                     minimum_penetration_contact, prepare, preload_target, validate_preload_advance,
                     grasp_height_description, shifted_grasp_wrist_target)
from analyze_contact_mechanics import summarize_contact_pairs

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=OUT,prefix='.test-contract-');cls.m=build(Path(cls.tmp.name)/'model.xml')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_official_hand_mapping(self):
        m=self.m
        for side in ['left','right']:
            js=[j for j in range(m.njnt) if name(m,mj.mjtObj.mjOBJ_JOINT,j).startswith(side+'_hand_')]
            self.assertEqual(len(js),7)
            for j in js:
                n=name(m,mj.mjtObj.mjOBJ_JOINT,j);src=SOURCE_LIMITS[n]
                expected=[src['lower'],src['upper']]
                # Upstream MJCF and URDF disagree for thumb_1; document rather than silently patch vendor files.
                if n=='left_hand_thumb_1_joint':expected=[-.724312,1.0472]
                if n=='right_hand_thumb_1_joint':expected=[-1.0472,.724312]
                np.testing.assert_allclose(m.jnt_range[j],expected,atol=5e-6)
                np.testing.assert_allclose(m.jnt_actfrcrange[j],[-src['effort'],src['effort']],atol=1e-8)
                self.assertEqual(sum(m.actuator_trnid[:,0]==j),1)
        self.assertEqual(m.neq,0);self.assertEqual(m.nmocap,0)
        self.assertEqual(m.jnt_type[jid(m,'bottle_free')],mj.mjtJoint.mjJNT_FREE)
        self.assertAlmostEqual(m.body_mass[bid(m,'green_box')],.57)
    def test_preload_correction_is_directional_bounded_and_source_valid(self):
        base=np.array([-.65448295,-.18924012,-.92799210,.70098552,.70741650,.70220062,.70445550])
        corrected=close_targets(base,.05)
        np.testing.assert_allclose(corrected-base,[0,-.05,-.05,.05,.05,.05,.05],atol=1e-12)
        for n,q in zip(HAND,corrected):
            self.assertGreaterEqual(q,SOURCE_LIMITS[n]['lower']+.02)
            self.assertLessEqual(q,SOURCE_LIMITS[n]['upper']-.02)
        with self.assertRaises(ValueError):close_targets(base,.050001)
    def test_only_reviewed_shoulder_height_candidate_is_admitted(self):
        original=np.array([.246,-.282,.882])
        shifted=shifted_grasp_wrist_target(original,.070)
        np.testing.assert_allclose(shifted-original,[0.,0.,.070],atol=1e-12)
        midpoint=shifted_grasp_wrist_target(original,.060)
        np.testing.assert_allclose(midpoint-original,[0.,0.,.060],atol=1e-12)
        np.testing.assert_allclose(original,[.246,-.282,.882],atol=0.)
        np.testing.assert_allclose(shifted_grasp_wrist_target(original,0.),original,atol=0.)
        for unsupported in [.055,.071,np.nan]:
            with self.assertRaises(ValueError):shifted_grasp_wrist_target(original,unsupported)

    def test_grasp_height_metadata_distinguishes_reviewed_candidates(self):
        self.assertEqual(grasp_height_description(0.),'baseline grasp height (0 mm offset)')
        self.assertIn('C3',grasp_height_description(.060))
        self.assertIn('C2',grasp_height_description(.070))
        with self.assertRaises(ValueError):grasp_height_description(.055)
    def test_preload_advance_is_one_bounded_source_valid_target_ramp(self):
        base=np.array([-.65448295,-.18924012,-.92799210,.70098552,.70741650,.70220062,.70445550])
        closed=close_targets(base,.05)
        advanced=preload_target(closed,.05)
        np.testing.assert_allclose(advanced-closed,[0.,-.05,-.05,.05,.05,.05,.05],atol=1e-12)
        for n,q in zip(HAND,advanced):
            self.assertGreaterEqual(q,SOURCE_LIMITS[n]['lower']+.02)
            self.assertLessEqual(q,SOURCE_LIMITS[n]['upper']-.02)
        np.testing.assert_allclose(preload_target(closed,0.),closed,atol=0.)
        with self.assertRaises(ValueError):preload_target(closed,.025)
        self.assertEqual(validate_preload_advance(.05),.05)
        with self.assertRaises(ValueError):validate_preload_advance(.025)
    def test_diagnostic_hand_stiffness_is_bounded_and_preserves_default(self):
        self.assertEqual(HAND_KP_DEFAULT,1.5)
        self.assertEqual(HAND_KD_DEFAULT,.1)
        self.assertEqual(HAND_KP_DIAGNOSTIC_MAX,3.0)
        diagnostic=build(Path(self.tmp.name)/'kp3.xml',hand_kp=3.0)
        a=next(i for i,j in enumerate(diagnostic.actuator_trnid[:,0]) if j==jid(diagnostic,HAND[1]))
        self.assertAlmostEqual(diagnostic.actuator_gainprm[a,0],3.0)
        self.assertAlmostEqual(diagnostic.actuator_biasprm[a,1],-3.0)
        self.assertAlmostEqual(diagnostic.actuator_biasprm[a,2],-.1)
        np.testing.assert_allclose(diagnostic.jnt_actfrcrange, self.m.jnt_actfrcrange)
        with self.assertRaises(ValueError):build(Path(self.tmp.name)/'invalid-kp.xml',hand_kp=3.001)
        with self.assertRaises(ValueError):build(Path(self.tmp.name)/'zero-kp.xml',hand_kp=0.)
    def test_readiness_gates_require_preload_stability_and_both_load_reactions(self):
        hold={'simultaneous_thumb_opposing_fraction':1.,
              'ideal_vertical_friction_capacity_screen_N':{'p05':7.},
              'max_bottle_horizontal_drift_mm':.5,'max_contact_migration_mm':2.,
              'max_contact_migration_limit_mm':3.5}
        load={'hand_support_N':{'p05':4.6},'table_support_N':{'p95':1.},
              'simultaneous_thumb_opposing_fraction':1.,'max_contact_migration_mm':2.,
              'max_contact_migration_limit_mm':3.5}
        gates=evaluate_readiness_gates(hold,load)
        self.assertEqual(set(gates.values()),{'PASS'})
        load['table_support_N']['p95']=2.
        self.assertEqual(evaluate_readiness_gates(hold,load)['LOAD_READY'],'FAIL')
        hold['max_bottle_horizontal_drift_mm']=1.1
        self.assertEqual(evaluate_readiness_gates(hold,load)['GRASP_STABLE'],'FAIL')
    def test_wrist_mount_and_mesh_provenance(self):
        root=ET.parse(SOURCE).getroot();r=ET.parse(Path(self.tmp.name)/'model.xml').getroot()
        for n in ['right_wrist_yaw_link','left_wrist_yaw_link','right_hand_thumb_0_link','left_hand_thumb_0_link']:
            for attr in ['pos','quat']:
                self.assertEqual(root.find(f".//body[@name='{n}']").get(attr),r.find(f".//body[@name='{n}']").get(attr))
        for mesh in root.findall('./asset/mesh'):
            other=r.find(f"./asset/mesh[@name='{mesh.get('name')}']")
            self.assertEqual(mesh.attrib,other.attrib)
    def test_jacobian_frame_and_virtual_work(self):
        m=self.m;d=init(m);qa,va=indices(m,HAND);d.qpos[qa]=[-.5,-.2,-.7,.5,.5,.5,.5];mj.mj_forward(m,d)
        b=bid(m,'right_hand_index_1_link');point=d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([.025,0,0])
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,point,b)
        local=d.xmat[b].reshape(3,3).T@(point-d.xpos[b]);q0=d.qpos.copy();eps=1e-6;F=np.array([1.3,-.8,.6])
        for q,v in zip(qa,va):
            d.qpos[:]=q0;d.qpos[q]+=eps;mj.mj_forward(m,d);plus=d.xpos[b]+d.xmat[b].reshape(3,3)@local
            d.qpos[:]=q0;d.qpos[q]-=eps;mj.mj_forward(m,d);minus=d.xpos[b]+d.xmat[b].reshape(3,3)@local
            fd=(plus-minus)/(2*eps);np.testing.assert_allclose(fd,jp[:,v],atol=1e-7)
            self.assertAlmostEqual(float(F@fd),float((jp.T@F)[v]),places=6)
    def test_contact_force_on_bottle_sign(self):
        m=self.m;d=init(m);dynamics_hold(m,d,.7)
        cs=bottle_contact(m,d);b=bid(m,'green_box');v=m.jnt_dofadr[jid(m,'bottle_free')]
        self.assertGreater(cs['table_z'],5.)
        measured=cs['table_z']+cs['hand_z']
        self.assertAlmostEqual(measured,float(d.qfrc_constraint[v+2]),places=5)
        # Independent Newton balance: summed force equals mass * world COM acceleration.
        self.assertLess(abs(measured-.57*9.81-.57*d.qacc[v+2]),.15)
    def test_contact_migration_does_not_mix_distinct_geom_pairs(self):
        rows = [
            {'time': 0., 'contacts': [
                {'other': 'finger', 'geoms': ['bottle_body', 'finger_a'],
                 'point_bottle_local_m': [0., 0., 0.], 'normal': [1., 0., 0.],
                 'normal_force_N': 1., 'tangential_force_N': .1,
                 'friction_pyramid_tangent_utilization': .2,
                 'world_force_on_bottle': [0., 0., 1.]},
                {'other': 'finger', 'geoms': ['bottle_shoulder', 'finger_b'],
                 'point_bottle_local_m': [.04, 0., 0.], 'normal': [0., 1., 0.],
                 'normal_force_N': 2., 'tangential_force_N': .2,
                 'friction_pyramid_tangent_utilization': .3,
                 'world_force_on_bottle': [0., 0., -1.]},]},
            {'time': 1., 'contacts': [
                {'other': 'finger', 'geoms': ['bottle_body', 'finger_a'],
                 'point_bottle_local_m': [.001, 0., 0.], 'normal': [1., 0., 0.],
                 'normal_force_N': 1., 'tangential_force_N': .1,
                 'friction_pyramid_tangent_utilization': .2,
                 'world_force_on_bottle': [0., 0., 1.]},
                {'other': 'finger', 'geoms': ['bottle_shoulder', 'finger_b'],
                 'point_bottle_local_m': [.041, 0., 0.], 'normal': [0., 1., 0.],
                 'normal_force_N': 2., 'tangential_force_N': .2,
                 'friction_pyramid_tangent_utilization': .3,
                 'world_force_on_bottle': [0., 0., -1.]},]},
        ]
        pairs = summarize_contact_pairs(rows, 'finger')
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(abs(p['max_bottle_frame_centroid_migration_mm'] - 1.) < 1e-9
                            for p in pairs))
        self.assertTrue(all(p['max_contact_normal_change_deg'] < 1e-6 for p in pairs))
    def test_minimum_penetration_reports_exact_geom_and_body_pair(self):
        xml='''<mujoco><worldbody>
          <body name="body_a"><freejoint name="free_a"/><geom name="geom_a" type="sphere" size="0.10"/></body>
          <body name="body_b" pos="0.15 0 0"><freejoint name="free_b"/><geom name="geom_b" type="sphere" size="0.10"/></body>
        </worldbody></mujoco>'''
        m=mj.MjModel.from_xml_string(xml);d=mj.MjData(m);mj.mj_forward(m,d)
        distance,pair=minimum_penetration_contact(m,d)
        self.assertLess(distance,-.049)
        self.assertEqual(pair['geom_names'],['geom_a','geom_b'])
        self.assertEqual(pair['body_names'],['body_a','body_b'])
        self.assertAlmostEqual(pair['distance_m'],distance)
    def test_active_loop_has_no_qpos_or_external_force_writes(self):
        p=Path(__file__).with_name('episode.py');tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for node in ast.walk(run):
            if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for t in targets:
                    text=ast.unparse(t)
                    self.assertNotIn('qpos',text);self.assertNotIn('xfrc_applied',text);self.assertNotIn('qfrc_applied',text)
        self.assertNotIn('eq_active',ast.unparse(run))

if __name__=='__main__':unittest.main(verbosity=2)
