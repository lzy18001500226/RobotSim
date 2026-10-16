"""Focused checks for source mapping, force signs and no object-state control."""
import ast, tempfile, unittest
from baseline import *
from episode import SOURCE_LIMITS, bottle_contact, prepare

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.m=build(Path(cls.tmp.name)/'model.xml')
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
