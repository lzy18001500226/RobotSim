#!/usr/bin/env python3
"""Isolated official-mesh OmniPicker with coordinated candidate kinematic targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import path_governance as pg
from research.issue46_dpoint_legal_motion import read_binary_stl

ROOT = Path('/home/lzy18001500226/projects/RobotSim')
SOURCE = Path('/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0')
CORRECTED = ROOT/'reviews/issue-46/dpoint-legal-motion-20261021T113855Z/corrected_candidate_B.xml'
STEP_SOURCE=Path('/home/lzy18001500226/.cache/robotsim/upstreams/agibot_op1_cad/OP1-官网模型.STEP')
DECOMP = ROOT/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json'
SAMPLES = Path(__file__).resolve().parents[2]/'docs/research/issue46_candidate_b_motion_samples.csv'
# Phase-A rollout stays inside the source-sampled, official-limit-valid interval.
Q_OPEN, Q_CLOSED = 0.0, -.10
PHASE = -.13412598906686654
EQ_SOLREF = '0.002 1'
JOINTS = {'narrow':['right_claw_joint','R_hand_narrow2_joint','R_hand_narrow3_joint','R_hand_narrow_loop_joint'],
          'wide':['R_hand_wide1_joint','R_hand_wide2_joint','R_hand_wide3_joint','R_hand_wide_loop_joint']}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(run, name, value):
    if isinstance(value, (dict,list)):
        value=json.dumps(value,indent=2,allow_nan=False)+'\n'
    pg.write_new_text(run/name,value,environment='wsl')


def fmt(a):
    return ' '.join(f'{float(v):.12g}' for v in a)


def quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return fmt(q[[3,0,1,2]])


class Linkage:
    """Analytic continuation of candidate B's existing theta2-theta4 law."""
    def __init__(self):
        rows=list(csv.DictReader(SAMPLES.open()))
        self.source={s:next(r for r in rows if r['branch']==s and int(r['sample'])==0) for s in JOINTS}
        self.rows=rows

    def solve(self,side,q):
        r=self.source[side]
        A,B,C,D,E=[np.array([float(r[k+'_x_mm']),float(r[k+'_z_mm'])]) for k in 'ABCDE']
        angle=lambda v:math.atan2(v[1],v[0])
        t1,t2,t3,t4=[angle(b-a) for a,b in [(A,B),(B,C),(C,D),(E,D)]]
        l1,l2,l3,l4=[float(np.linalg.norm(b-a)) for a,b in [(A,B),(B,C),(C,D),(E,D)]]
        phase=t2-t4
        V=l4*np.array([math.cos(-phase),math.sin(-phase)])-np.array([l2,0.])
        B=A+l1*np.array([math.cos(t1+q),math.sin(t1+q)])
        P=E-B
        cosine=(l3*l3-P@P-V@V)/(2*np.linalg.norm(P)*np.linalg.norm(V))
        if abs(cosine)>1+1e-10:
            raise ValueError(f'no real candidate closure: {side} q={q}, cos={cosine}')
        candidates=[angle(P)-angle(V)+sign*math.acos(np.clip(cosine,-1,1)) for sign in [-1,1]]
        wrap=lambda a:math.atan2(math.sin(a),math.cos(a))
        t2now=min(candidates,key=lambda a:abs(wrap(a-t2)))
        t2now=t2+wrap(t2now-t2)
        t4now=t2now-phase
        C=B+l2*np.array([math.cos(t2now),math.sin(t2now)])
        D=E+l4*np.array([math.cos(t4now),math.sin(t4now)])
        t3now=t3+wrap(angle(D-C)-t3)
        cad=np.array([q,t2now-t2-q,t3now-t3-(t2now-t2),t4now-t4])
        return -cad,float(abs(np.linalg.norm(D-C)-l3))

    def targets(self,q):
        values={};res=0.
        for side in JOINTS:
            v,r=self.solve(side,q if side=='narrow' else -q);res=max(res,r)
            for i,name in enumerate(JOINTS[side]):
                values[name]=float(v[i]+(PHASE if side=='narrow' else -PHASE) if i==0 else v[i])
        if not -1<=values['right_claw_joint']<=0 or not 0<=values['R_hand_wide1_joint']<=1:
            raise ValueError('official root limit would be violated')
        return values,res

    def test(self):
        error=0.
        for row in self.rows:
            if abs(float(row['q_root_rel_rad']))>.13:continue
            got,_=self.solve(row['branch'],float(row['q_root_rel_rad']))
            expect=-np.array([float(row[k]) for k in ['q_root_rel_rad','q_B_rel_rad','q_C_rel_rad','q_E_rel_rad']])
            error=max(error,float(np.max(np.abs(got-expect))))
        assert error<1e-8,error
        forward=[self.targets(q)[0] for q in np.linspace(Q_CLOSED,Q_OPEN,101)]
        reverse=[self.targets(q)[0] for q in np.linspace(Q_OPEN,Q_CLOSED,101)][::-1]
        assert max(abs(a[k]-b[k]) for a,b in zip(forward,reverse) for k in a)<1e-10
        motor={}
        for side in ['left','right']:
            closed=normalized_gripper_target(side,0.);opened=normalized_gripper_target(side,1.)
            assert closed['active_motor_joint']==('left_claw_joint' if side=='left' else 'right_claw_joint')
            assert closed['passive_joint_targets']==opened['passive_joint_targets']=={}
            assert abs(closed['mimic_target_rad']+closed['active_motor_target_rad'])<1e-12
            assert abs(opened['mimic_target_rad']+opened['active_motor_target_rad'])<1e-12
            motor[side]={'closed':closed,'open':opened}
        return {'source_sample_max_angle_error_rad':error,'reverse_target_error_rad':0.,'candidate_relation':'(q_root - qpos0_root) + q_B - q_E = 0 per branch; UNVERIFIED_EXPERIMENTAL','motor_api_mapping':motor}


def jid(model,name):
    return mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)


def set_static(model,data,targets):
    for name,value in targets.items():data.qpos[model.jnt_qposadr[jid(model,name)]]=value
    mujoco.mj_forward(model,data)


def verify_sources():
    repo=SOURCE.parent
    head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if head!='575cc6b988f976c23550e0db85aa1e5475d3652d':raise ValueError(f'official URDF pin mismatch: {head}')
    assert digest(SOURCE/'X2-Ultra_omnipicker.urdf')=='35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d'
    if digest(STEP_SOURCE)!='5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b':raise ValueError('STEP source hash mismatch')
    if digest(CORRECTED)!='e935c38d2869760de2ec1da6a689639a90ba143922aae0b48f8dbb7a98be1ba2':raise ValueError('approved D-site corrected candidate hash mismatch')
    provenance=json.load(DECOMP.open())
    if provenance['source_repo_commit']!='575cc6b988f976c23550e0db85aa1e5475d3652d' or provenance['source_urdf_sha256']!='35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d':raise ValueError('collision decomposition provenance does not match official pin')
    root=ET.parse(SOURCE/'X2-Ultra_omnipicker.urdf').getroot()
    joints={e.get('name'):e for e in root.findall('joint')}
    for name in ['left_claw_joint','right_claw_joint']:
        j=joints[name];axis=np.array([float(v) for v in j.find('axis').get('xyz').split()]);lim=j.find('limit')
        if not np.allclose(axis,[0,0,-1]) or (float(lim.get('lower')),float(lim.get('upper')))!=(-1.,0.):raise ValueError(f'official root contract mismatch: {name}')
    for name in ['L_hand_wide1_joint','R_hand_wide1_joint']:
        j=joints[name];axis=np.array([float(v) for v in j.find('axis').get('xyz').split()]);lim=j.find('limit')
        if not np.allclose(axis,[0,0,-1]) or (float(lim.get('lower')),float(lim.get('upper')))!=(0.,1.):raise ValueError(f'official wide-root contract mismatch: {name}')
        mimic=joints[name].find('mimic')
        if mimic is None or float(mimic.get('multiplier','nan'))!=-1. or mimic.get('joint') not in ('left_claw_joint','right_claw_joint'):
            raise ValueError(f'official opposite-root mimic mismatch: {name}')
    links={e.get('name'):e for e in root.findall('link')}
    # The right/left isolated gripper links use identical official geometry/inertias.
    for name in [e.get('name') for e in root.findall('link') if e.get('name','').startswith('L_hand_') or e.get('name')=='L_omnipicker_base_link']:
        a=ET.tostring(links[name]);b=ET.tostring(links[name.replace('L_','R_')])
        assert a.replace(b'L_',b'R_')==b,name
        for element in links[name].iter():
            mesh=element.find('geometry/mesh')
            if mesh is None:continue
            path=SOURCE/mesh.get('filename').removeprefix('./')
            if not path.is_file():raise FileNotFoundError(path)
    return root


def normalized_gripper_target(side,opening):
    if side not in ('left','right'):raise ValueError('side must be left or right')
    opening=float(opening)
    if not 0.<=opening<=1.:raise ValueError('opening must be normalized to [0,1]')
    q=Q_CLOSED+opening*(Q_OPEN-Q_CLOSED)
    root='left_claw_joint' if side=='left' else 'right_claw_joint'
    wide='L_hand_wide1_joint' if side=='left' else 'R_hand_wide1_joint'
    root_target=PHASE-q
    if not -1.<=root_target<=0.:raise ValueError('mapped root command exceeds unchanged official joint limits')
    return {'side':side,'opening':opening,'candidate_q_CAD':q,'active_motor_joint':root,'active_motor_target_rad':root_target,'mimic_joint':wide,'mimic_target_rad':-root_target,'passive_joint_targets':{}}


def set_gripper(side,opening,model,data,actuator_map):
    """Motor-only simulation API. Passive hinges receive no actuator command."""
    command=normalized_gripper_target(side,opening)
    name=command['active_motor_joint']+'_servo'
    if name not in actuator_map:raise ValueError(f'isolated model does not include {side} hand root actuator')
    data.ctrl[actuator_map[name]]=command['active_motor_target_rad']
    return command


def build(run,kind='preview',placement=None):
    source=verify_sources()
    xml=CORRECTED.read_text().replace('L_','R_').replace('left_claw_joint','right_claw_joint')
    root=ET.fromstring(xml);root.set('model','SIMULATION_ONLY_CAD_DRIVEN_OMNIPICKER')
    eq=root.find('equality');removed_equalities=[]
    for e in list(eq):
        if e.tag not in ('joint','connect','tendon'):
            removed_equalities.append({'type':e.tag,'name':e.get('name')})
            eq.remove(e)
    tendon=root.find('tendon')
    required_rate_equalities={f'candidate_{side}_rate_eq' for side in ['narrow','wide']}
    present={e.get('name') for e in eq if e.tag=='tendon'}
    if present!=required_rate_equalities or tendon is None:
        raise ValueError('Candidate B must provide exactly the two per-side rate equalities and fixed tendons')
    for equality in eq:
        equality.set('solref',EQ_SOLREF)
    act=root.find('actuator');act.clear()
    ET.SubElement(act,'position',name='right_claw_joint_servo',joint='right_claw_joint',kp='0.5',kv='0.05',ctrllimited='true',ctrlrange='-1 0',forcelimited='true',forcerange='-0.5 0.5')
    root.find('option').set('iterations','100')
    root.find('option').set('cone','elliptic')
    world=root.find('worldbody')
    for cam in list(world.findall('camera')):world.remove(cam)
    ET.SubElement(world,'light',pos='0 -0.5 0.8',dir='0 1 -1',diffuse='0.8 0.8 0.8')
    base=world.find('body')
    colors={'narrow1':'0.9 0.58 0.16 1','wide1':'0.9 0.58 0.16 1','narrow2':'0.15 0.65 0.9 1','wide2':'0.15 0.65 0.9 1','narrow3':'0.2 0.8 0.5 1','wide3':'0.85 0.4 0.25 1','loop':'0.55 0.4 0.8 1'}
    for g in base.iter('geom'):
        if g.get('name','').endswith('__visual'):
            for key,color in colors.items():
                if key in g.get('name',''):g.set('rgba',color)
    provenance=json.load(DECOMP.open())
    asset=root.find('asset');meshes={Path(m['source_mesh']).stem:m for m in provenance['meshes']}
    coverage=[]
    for body in base.iter('body'):
        for geom in list(body.findall('geom')):
            if not geom.get('name','').endswith('__collision'):continue
            src_name=geom.get('mesh');src=meshes[src_name]
            assert digest(SOURCE/'meshes'/Path(src['source_mesh']).name)==src['sha256']
            for piece in src['collision_pieces']:
                path=Path(piece['file']);assert digest(path)==piece['sha256']
                ET.SubElement(asset,'mesh',name=piece['name'],file=str(path))
                attrs=dict(geom.attrib);attrs.update(name=body.get('name')+'__'+piece['name'],mesh=piece['name'],friction='1.4 0.02 0.001',condim='4',solref='0.01 1')
                ET.SubElement(body,'geom',attrs)
            body.remove(geom)
            coverage.append({'body':body.get('name'),'source_mesh':src['source_mesh'],'source_sha256':src['sha256'],'pieces':src['parts']})
    contact=ET.SubElement(root,'contact');exclusions=[]
    for side in ['narrow','wide']:
        for part in ['1','3']:
            a=f'R_hand_{side}{part}_Link';b=f'R_hand_{side}_loop_Link'
            ET.SubElement(contact,'exclude',body1=a,body2=b)
            exclusions.append([a,b])
    if kind=='contact':
        R=np.array([[0,0,-1],[0,1,0],[1,0,0.]])
        center=np.array(placement['source_neck_center_m']);neck_world=np.array([0.,0.,.1175+.0855])
        base.set('quat',quat(R));base.set('pos',fmt(neck_world-R@center))
        ET.SubElement(base,'joint',name='fixture_lift',type='slide',axis='1 0 0',range='0 0.12',limited='true')
        ET.SubElement(act,'position',name='fixture_lift_servo',joint='fixture_lift',kp='2000',kv='90',forcelimited='true',forcerange='-50 50')
        ET.SubElement(world,'geom',name='table',type='plane',size='0 0 0.02',rgba='0.5 0.5 0.5 1',friction='1.0 0.01 0.001',contype='1',conaffinity='1')
        bottle=ET.SubElement(world,'body',name='bottle',pos='0 0 0.1175')
        ET.SubElement(bottle,'freejoint',name='bottle_free')
        for name,t,p,s,mass in [('body','cylinder','0 0 -0.04','0.035 0.0775',.49),('shoulder','ellipsoid','0 0 0.0535','0.035 0.035 0.026',.05),('neck','cylinder','0 0 0.0855','0.018 0.020',.02),('cap','cylinder','0 0 0.1185','0.020 0.0085',.01)]:
            ET.SubElement(bottle,'geom',name='bottle_'+name,type=t,pos=p,size=s,mass=str(mass),rgba='0.15 0.45 0.8 1',friction='1.4 0.02 0.001',condim='4')
    elif kind=='test-cylinder':
        base.insert(0,ET.Element('joint',name='fixture_lift',type='slide',axis='0 0 1',range='0 0.08',limited='true'))
        ET.SubElement(act,'position',name='fixture_lift_servo',joint='fixture_lift',kp='2000',kv='90',forcelimited='true',forcerange='-50 50')
        ET.SubElement(world,'geom',name='test_shelf',type='box',pos='0 0 0.113',size='0.006 0.006 0.002',rgba='0.35 0.38 0.42 1',friction='1.0 0.01 0.001',contype='1',conaffinity='1')
        obj=ET.SubElement(world,'body',name='test_cylinder',pos='0 0 0.125')
        ET.SubElement(obj,'inertial',pos='0 0 0',mass='0.03',diaginertia='0.0000016075 0.0000016075 0.000001215')
        ET.SubElement(obj,'freejoint',name='test_cylinder_free')
        ET.SubElement(obj,'geom',name='test_cylinder_geom',type='cylinder',size='0.009 0.010',mass='0.03',rgba='0.15 0.45 0.8 1',friction='1.4 0.02 0.001',condim='4')
    out=run/(kind+'.xml');ET.indent(root);write(run,out.name,ET.tostring(root,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(out))
    actuator_joints=['right_claw_joint']+(['fixture_lift'] if kind in ('contact','test-cylinder') else [])
    return model,{'internal_pair_exclusions':exclusions,'external_collision_bodies':coverage,'collision_provenance_sha256':digest(DECOMP),'collision_caveat':provenance['caveat'],'wrist_in_physical_rig':False,'fixture':'isolated gripper mount; not full X2 arm','actuator_joints':actuator_joints,'removed_unverified_equalities':removed_equalities,'retained_equalities':[{'type':e.tag,'name':e.get('name')} for e in root.find('equality')],'candidate_equality_solref':EQ_SOLREF,'test_object':{'type':'free cylinder on a small static shelf; diameter 18 mm, height 20 mm, mass 0.03 kg'} if kind=='test-cylinder' else None,'passive_joint_interpretation':'B/C/E and loop hinges are unactuated; one Candidate B fixed-tendon relation per side reduces each D-closed branch to one local DOF; both relations remain UNVERIFIED_EXPERIMENTAL','candidate_coupling':'(root - qpos0_root) + B - E = 0 per branch; SIMULATION_ONLY','springs':'not modeled'}


def world_mesh(model,data,gid):
    mi=model.geom_dataid[gid];va=model.mesh_vertadr[mi];vn=model.mesh_vertnum[mi];fa=model.mesh_faceadr[mi];fn=model.mesh_facenum[mi]
    vertices=model.mesh_vert[va:va+vn].copy();faces=model.mesh_face[fa:fa+fn].copy()
    return vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid],faces


def jaw(model,data,side):
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'R_hand_{side}3_Link__visual')
    return world_mesh(model,data,gid)


def ray_inner(tri,x,z,side):
    origin=np.array([x,-.5,z]);direction=np.array([0.,1.,0.]);e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,h)
    valid=np.abs(det)>1e-14;inv=np.zeros_like(det);inv[valid]=1/det[valid];s=origin-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);v=inv*(q@direction);t=inv*np.einsum('ij,ij->i',e2,q)
    hits=origin[1]+t[valid&(u>=0)&(v>=0)&(u+v<=1)&(t>0)]
    if not len(hits):return None
    return float(hits.max() if side=='narrow' else hits.min())


def camera(lookat,distance=.36):
    c=mujoco.MjvCamera();c.lookat[:]=lookat;c.distance=distance;c.azimuth=135;c.elevation=-17
    return c


def prepare(run):
    kin=Linkage();tests=kin.test();model,overlay=build(run);data=mujoco.MjData(model)
    samples=[];scene={'meshes':[],'frames':[],'label':'UNVERIFIED_EXPERIMENTAL Candidate B coupling; source-range kinematics; ZERO PHYSICS STEPS'};max_coupling=0.;max_D_gap=0.
    selected=[i for i in range(model.ngeom) if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i).endswith('__visual')]
    for gid in selected:
        mi=model.geom_dataid[gid];va=model.mesh_vertadr[mi];vn=model.mesh_vertnum[mi];fa=model.mesh_faceadr[mi];fn=model.mesh_facenum[mi]
        scene['meshes'].append({'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid),'vertices':model.mesh_vert[va:va+vn].astype(float).reshape(-1).tolist(),'faces':model.mesh_face[fa:fa+fn].reshape(-1).tolist(),'color':model.geom_rgba[gid,:3].tolist()})
    renderer=mujoco.Renderer(model,height=480,width=640);cam=camera([0,0,.085]);tempdir=tempfile.TemporaryDirectory(prefix='issue46-kinematic-video-');video_tmp=Path(tempdir.name)/'kinematic_open_close.mp4';writer=imageio.get_writer(str(video_tmp),fps=30,codec='libx264')
    for i,q in enumerate(np.r_[np.linspace(Q_OPEN,Q_CLOSED,61),np.linspace(Q_CLOSED,Q_OPEN,61)[1:]]):
        targets,res=kin.targets(float(q));set_static(model,data,targets)
        a,af=jaw(model,data,'narrow');b,bf=jaw(model,data,'wide');dist=float(cKDTree(a).query(b,k=1)[0].min())
        gaps={s:float(np.linalg.norm(data.site(f'candidate_{s}_D_jaw').xpos-data.site(f'candidate_{s}_D_loop').xpos)) for s in JOINTS}
        coupling=candidate_B_residuals(model,data);max_coupling=max(max_coupling,max(abs(v) for v in coupling.values()));max_D_gap=max(max_D_gap,max(gaps.values()))
        samples.append({'frame':i+1,'q_CAD':float(q),'nearest_jaw_vertices_m':dist,'D_gap_m':gaps,'candidate_B_coupling_residual_rad':coupling,'closure_length_residual_mm':res,'targets':targets})
        scene['frames'].append({'q':float(q),'transforms':[np.block([[data.geom_xmat[g].reshape(3,3),data.geom_xpos[g,None].T],[np.zeros((1,3)),np.ones((1,1))]]).T.reshape(-1).tolist() for g in selected]})
        renderer.update_scene(data,camera=cam);frame=renderer.render();writer.append_data(frame)
        if i in [0,30,60]:
            image_tmp=Path(tempdir.name)/f'kinematic_frame_{i:03d}.png';imageio.imwrite(image_tmp,frame);pg.write_new_bytes(run/image_tmp.name,image_tmp.read_bytes(),environment='wsl')
    writer.close();renderer.close();pg.write_new_bytes(run/'kinematic_open_close.mp4',video_tmp.read_bytes(),environment='wsl');tempdir.cleanup();write(run,'kinematic_samples.json',samples);write(run,'official_mesh_scene.json',scene)
    # Surface-based neck placement candidates at a central x cross-section.
    sections=[]
    for z in np.linspace(.085,.155,71):
        widths=[];centers=[]
        for q in [Q_OPEN,(Q_OPEN+Q_CLOSED)/2,Q_CLOSED]:
            set_static(model,data,kin.targets(q)[0]);a,af=jaw(model,data,'narrow');b,bf=jaw(model,data,'wide')
            lo=ray_inner(a[af],0.,float(z),'narrow');hi=ray_inner(b[bf],0.,float(z),'wide')
            widths.append(None if lo is None or hi is None else hi-lo);centers.append(None if lo is None or hi is None else (hi+lo)/2)
        if all(v is not None for v in widths) and widths[0]>.040 and 0<widths[1]<.040 and widths[2]<.020:sections.append({'z':float(z),'widths':widths,'centers':centers})
    placement=None
    if sections:
        best=max(sections,key=lambda r:r['widths'][0]);placement={'source_neck_center_m':[0.,best['centers'][0],best['z']],'section':best,'bottle':'canonical 70 mm body, 36 mm neck, 0.570 kg; neck grasp'}
    write(run,'placement.json',placement or {'status':'NO_SURFACE_SECTION_FOR_CANONICAL_NECK'})
    set_static(model,data,kin.targets(Q_OPEN)[0])
    mobility=mobility_audit(model,data);initial_contacts=all_contacts(model,data)
    static_ok=max_coupling<1e-8 and max_D_gap<1e-3 and mobility['unconstrained_velocity_dofs_at_pose']==1 and all(v['remaining_local_dofs']==1 for v in mobility['branch_mobility'].values())
    validation={'status':'STATIC_CANDIDATE_B_PASS' if static_ok else 'STATIC_CANDIDATE_B_FAIL','tests':tests,'q_window':[Q_CLOSED,Q_OPEN],'source_supported_q_window':[Q_CLOSED,Q_OPEN],'reverse_same_branch':True,'physics_steps':0,'max_D_gap_m':max_D_gap,'max_candidate_B_coupling_residual_rad':max_coupling,'initial_equality_mobility':mobility,'initial_contacts_after_scoped_exclusions':initial_contacts,'surface_sections':sections,'canonical_neck_placement_found':placement is not None}
    write(run,'overlay.json',overlay);write(run,'kinematic_validation.json',validation)
    write(run,'viewer.html',VIEWER)
    print(json.dumps({'kinematics':tests,'static_candidate_B':validation,'placement':placement,'frames':len(samples),'run':str(run)},indent=2))
    if not static_ok:raise ValueError('Candidate B source-range static preflight failed; refusing to start mj_step')


def contacts(model,data,object_name='bottle'):
    out=[]
    for i in range(data.ncon):
        c=data.contact[i];b1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[c.geom1]));b2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[c.geom2]))
        if object_name not in [b1,b2]:continue
        f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f);world=c.frame.reshape(3,3).T@f[:3];world*=1 if b2==object_name else -1
        other=b1 if b2==object_name else b2;bg=c.geom2 if b2==object_name else c.geom1
        side='narrow' if other.startswith('R_hand_narrow') and '_loop_' not in other else 'wide' if other.startswith('R_hand_wide') and '_loop_' not in other else 'table' if other=='world' else 'other'
        out.append({'side':side,'other':other,'object_geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(bg)),'normal_N':float(f[0]),'world_force_N':world.tolist(),'position_m':c.pos.tolist(),'penetration_m':max(0.,-float(c.dist))})
    return out


def mobility_audit(model,data):
    eqnames=['candidate_narrow_D_closure','candidate_wide_D_closure','source_wide_root_mimic','candidate_narrow_rate_eq','candidate_wide_rate_eq']
    all_ids=np.asarray(data.efc_id[:data.nefc])
    selected_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,name) for name in eqnames]
    equality_mask=np.isin(all_ids,selected_ids)
    raw=np.asarray(data.efc_J)
    if raw.size==data.nefc*model.nv:
        jac=raw.reshape(data.nefc,model.nv)
    else:
        jac=np.zeros((data.nefc,model.nv))
        for row in range(data.nefc):
            start=int(data.efc_J_rowadr[row]);count=int(data.efc_J_rownnz[row])
            columns=np.asarray(data.efc_J_colind[start:start+count],dtype=int)
            jac[row,columns]=raw[start:start+count]
    rows={};branch_joints={s:[jid(model,n) for n in names] for s,names in JOINTS.items()}
    for name in eqnames:
        eid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,name)
        idx=np.flatnonzero(np.asarray(data.efc_id[:data.nefc])==eid) if eid>=0 else np.array([],dtype=int)
        block=jac[idx]
        rows[name]={'rows':int(len(idx)),'rank':int(np.linalg.matrix_rank(block,tol=1e-9)) if len(idx) else 0,'max_abs_efc_pos':float(np.max(np.abs(data.efc_pos[idx]))) if len(idx) else 0.0}
    branch_mobility={}
    for side in ['narrow','wide']:
        names=[f'candidate_{side}_D_closure',f'candidate_{side}_rate_eq']
        eids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,name) for name in names]
        idx=np.flatnonzero(np.isin(np.asarray(data.efc_id[:data.nefc]),eids))
        cols=[int(model.jnt_dofadr[j]) for j in branch_joints[side]]
        block=jac[np.ix_(idx,cols)]
        rank=int(np.linalg.matrix_rank(block,tol=1e-9)) if len(idx) else 0
        closure_id=eids[0]
        closure_idx=np.flatnonzero(np.asarray(data.efc_id[:data.nefc])==closure_id) if closure_id>=0 else np.array([],dtype=int)
        closure_block=jac[np.ix_(closure_idx,cols)]
        closure_rank=int(np.linalg.matrix_rank(closure_block,tol=1e-9)) if len(closure_idx) else 0
        branch_mobility[side]={'hinge_dofs':len(cols),'D_closure_rows':int(len(closure_idx)),'D_closure_rank_on_branch':closure_rank,'candidate_coupling_rows':int(np.count_nonzero(np.asarray(data.efc_id[:data.nefc])==eids[1])),'combined_D_and_coupling_rank_on_branch':rank,'remaining_local_dofs':int(len(cols)-rank)}
    eq_jac=jac[equality_mask]
    rank=int(np.linalg.matrix_rank(eq_jac,tol=1e-9)) if len(eq_jac) else 0
    return {'nv':int(model.nv),'constraint_rows':int(len(eq_jac)),'total_efc_rows':int(data.nefc),'non_equality_efc_rows':int(data.nefc-len(eq_jac)),'constraint_jacobian_rank':rank,'unconstrained_velocity_dofs_at_pose':int(model.nv-rank),'equality_rows':rows,'branch_mobility':branch_mobility,'note':'DOF rank uses only the five named equalities, not incidental contact rows; includes official root mimic, both D closures, and the two UNVERIFIED_EXPERIMENTAL Candidate B tendon couplings'}


def candidate_B_residuals(model,data):
    result={}
    for side,names in JOINTS.items():
        values=[]
        for name in [names[0],names[1],names[3]]:
            j=jid(model,name);adr=model.jnt_qposadr[j]
            values.append(float(data.qpos[adr]-model.qpos0[adr]))
        result[side]=values[0]+values[1]-values[2]
    return result


def equality_force_magnitudes(model,data):
    names=['candidate_narrow_D_closure','candidate_wide_D_closure','source_wide_root_mimic','candidate_narrow_rate_eq','candidate_wide_rate_eq']
    eqids={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,name) for name in names}
    ids=np.asarray(data.efc_id[:data.nefc]);forces=np.asarray(data.efc_force[:data.nefc])
    return {name:float(np.max(np.abs(forces[ids==eid]))) if eid>=0 and np.any(ids==eid) else 0.0 for name,eid in eqids.items()}


def all_contacts(model,data):
    out=[]
    for i in range(data.ncon):
        c=data.contact[i];g1=int(c.geom1);g2=int(c.geom2)
        b1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g1])) or 'world'
        b2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g2])) or 'world'
        out.append({'body1':b1,'body2':b2,'geom1':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g1),'geom2':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g2),'dist_m':float(c.dist),'penetration_m':max(0.,-float(c.dist))})
    return out


def qpos_initial(model,data,targets):
    for side,names in JOINTS.items():
        for name in names:data.qpos[model.jnt_qposadr[jid(model,name)]]=targets[name]
    mujoco.mj_forward(model,data)


def physics_open_close(run):
    kin=Linkage();model,overlay=build(run,'physics');data=mujoco.MjData(model)
    targets_open=kin.targets(Q_OPEN)[0];qpos_initial(model,data,targets_open)
    amap={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a):a for a in range(model.nu)};act=amap['right_claw_joint_servo']
    api_open=normalized_gripper_target('right',1.);data.ctrl[act]=api_open['active_motor_target_rad'];start=mobility_audit(model,data);initial_contacts=all_contacts(model,data)
    baseline_xml=ET.fromstring(CORRECTED.read_text().replace('L_','R_').replace('left_claw_joint','right_claw_joint'))
    for equality in list(baseline_xml.find('equality')):
        if equality.tag not in ('joint','connect'):baseline_xml.find('equality').remove(equality)
    baseline_xml.remove(baseline_xml.find('tendon'))
    baseline_model=mujoco.MjModel.from_xml_string(ET.tostring(baseline_xml,encoding='unicode'))
    baseline_data=mujoco.MjData(baseline_model);qpos_initial(baseline_model,baseline_data,targets_open)
    baseline_contacts=all_contacts(baseline_model,baseline_data)
    dt=float(model.opt.timestep);total_steps=int(4.6/dt);trace=[];max_gap=0.;max_pen=0.;max_qvel=0.;max_D=0.;max_rate=0.;stop='';passive_min={n:float(data.qpos[model.jnt_qposadr[jid(model,n)]]) for side,names in JOINTS.items() for n in names[1:]};passive_max=dict(passive_min);max_constraint_forces={name:0.0 for name in equality_force_magnitudes(model,data)}
    render_tmp=Path(tempfile.gettempdir())/'issue46_single_motor_open_close.mp4';writer=imageio.get_writer(str(render_tmp),fps=30,codec='libx264');renderer=mujoco.Renderer(model,height=480,width=640);cam=camera([0,0,.085])
    record_steps=set(np.linspace(0,total_steps-1,101).round().astype(int).tolist())
    for step in range(total_steps):
        t=step*dt
        if t<.5:q=Q_OPEN
        elif t<1.9:q=Q_OPEN+(Q_CLOSED-Q_OPEN)*(t-.5)/1.4
        elif t<2.3:q=Q_CLOSED
        elif t<3.7:q=Q_CLOSED+(Q_OPEN-Q_CLOSED)*(t-2.3)/1.4
        else:q=Q_OPEN
        opening=float(np.clip((q-Q_CLOSED)/(Q_OPEN-Q_CLOSED),0.,1.));api_command=set_gripper('right',opening,model,data,amap)
        mujoco.mj_step(model,data)
        force_magnitudes=equality_force_magnitudes(model,data)
        for name,value in force_magnitudes.items():max_constraint_forces[name]=max(max_constraint_forces[name],value)
        current=all_contacts(model,data);gap=max(float(np.linalg.norm(data.site(f'candidate_{s}_D_jaw').xpos-data.site(f'candidate_{s}_D_loop').xpos)) for s in JOINTS);max_gap=max(max_gap,gap)
        penetration=max((c['penetration_m'] for c in current),default=0.);max_pen=max(max_pen,penetration);max_qvel=max(max_qvel,float(np.max(np.abs(data.qvel))))
        D=max(float(np.linalg.norm(data.site(f'candidate_{s}_D_jaw').xpos-data.site(f'candidate_{s}_D_loop').xpos)) for s in JOINTS);max_D=max(max_D,D)
        qactual=float(PHASE-data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]])
        law={s:float(sum(data.qpos[model.jnt_qposadr[jid(model,n)]]-model.qpos0[model.jnt_qposadr[jid(model,n)]] for n in [names[0],names[1]])-(data.qpos[model.jnt_qposadr[jid(model,names[3])]]-model.qpos0[model.jnt_qposadr[jid(model,names[3])]])) for s,names in JOINTS.items()}
        max_rate=max(max_rate,max(abs(x) for x in law.values()))
        for name in passive_min:
            value=float(data.qpos[model.jnt_qposadr[jid(model,name)]]);passive_min[name]=min(passive_min[name],value);passive_max[name]=max(passive_max[name],value)
        if step in record_steps or step%10==0:
            state={'time_s':float(data.time),'command_opening':opening,'command_CAD':float(q),'actual_CAD':qactual,'right_root_q':float(data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]]),'wide_root_q':float(data.qpos[model.jnt_qposadr[jid(model,'R_hand_wide1_joint')]]),'passive_q':{name:float(data.qpos[model.jnt_qposadr[jid(model,name)]]) for side,names in JOINTS.items() for name in names[1:]},'candidate_B_coupling_residual_rad':law,'D_gap_m':gap,'qvel_max_rad_s':float(np.max(np.abs(data.qvel))),'root_actuator_force_Nm':float(data.actuator_force[act]),'constraint_generalized_force_abs':force_magnitudes,'internal_contacts':current,'motor_api':api_command}
            trace.append(state)
        if step%20==0:
            renderer.update_scene(data,camera=cam);writer.append_data(renderer.render())
        rootq=float(data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]])
        wideq=float(data.qpos[model.jnt_qposadr[jid(model,'R_hand_wide1_joint')]])
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):stop='nonfinite state'
        elif not -1.00001<=rootq<=.00001:stop='official right root limit violation'
        elif not -.00001<=wideq<=1.00001:stop='official wide root limit violation'
        elif gap>.005:stop='D-site closure error exceeded 5 mm'
        elif max_rate>.01:stop='Candidate B coupling residual exceeded 0.01 rad'
        elif max_qvel>25:stop='runaway joint velocity above 25 rad/s'
        elif penetration>.005:stop='remaining internal penetration exceeded 5 mm after scoped exclusions'
        if stop:break
    writer.close();renderer.close()
    if render_tmp.is_file():pg.write_new_bytes(run/'single_motor_open_close.mp4',render_tmp.read_bytes(),environment='wsl')
    root_values=[x['actual_CAD'] for x in trace];passive_ranges={n:{'min_rad':passive_min[n],'max_rad':passive_max[n],'range_rad':passive_max[n]-passive_min[n]} for n in passive_min}
    passive_moved=bool(passive_ranges) and all(v['range_rad']>.01 for v in passive_ranges.values())
    completed=step+1==total_steps and not stop and max(root_values)-min(root_values)>.08 and abs(trace[-1]['actual_CAD']-Q_OPEN)<.01 and passive_moved and max_rate<=.01 and start['unconstrained_velocity_dofs_at_pose']==1
    result={'status':'SINGLE_MOTOR_PHYSICS_PASS' if completed else 'SINGLE_MOTOR_PHYSICS_FAIL','mujoco_version':mujoco.__version__,'steps':step+1,'dt_s':dt,'simulated_time_s':float(data.time),'stop_reason':stop,'initial_mobility':start,'max_D_gap_m':max_D,'max_candidate_B_coupling_residual_rad':max_rate,'max_internal_contact_penetration_m':max_pen,'max_qvel_rad_s':max_qvel,'original_collision_baseline_initial_contacts':baseline_contacts,'diagnostic_model_initial_contacts':initial_contacts,'final_CAD_q':trace[-1]['actual_CAD'],'CAD_root_stroke_rad':max(root_values)-min(root_values),'passive_joint_ranges':passive_ranges,'active_gripper_actuator':'right_claw_joint_servo','passive_actuator_count':0,'passive_qpos_writes_after_start':0,'collision_pair_exclusions':overlay['internal_pair_exclusions'],'trajectory_phases_s':{'open_hold':[0.0,0.5],'closing':[0.5,1.9],'closed_hold':[1.9,2.3],'reopening':[2.3,3.7],'final_open_hold':[3.7,4.6]},'source_supported_candidate_q_window':[Q_CLOSED,Q_OPEN],'trace_samples':len(trace)}
    result['max_constraint_generalized_force_abs']=max_constraint_forces
    result['constraint_force_units_note']='MuJoCo efc_force magnitudes are generalized forces; translational and rotational equality rows use different units.'
    result['candidate_equality_solref']=EQ_SOLREF
    write(run,'single_motor_physics_result.json',result);write(run,'single_motor_physics_trace.json',trace);return result


def rollout(run):
    placement=json.load((run/'placement.json').open())
    if 'source_neck_center_m' not in placement:raise ValueError('milestone A has no suitable neck aperture')
    kin=Linkage();model,overlay=build(run,'contact',placement);data=mujoco.MjData(model)
    open_targets=kin.targets(Q_OPEN)[0]
    for side,names in JOINTS.items():
        for name in names:data.qpos[model.jnt_qposadr[jid(model,name)]]=open_targets[name]
    mujoco.mj_forward(model,data)
    amap={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a):a for a in range(model.nu)}
    data.ctrl[amap['right_claw_joint_servo']]=open_targets['right_claw_joint']
    liftact=amap['fixture_lift_servo'];bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bottle')
    initial_z=float(data.xpos[bid,2]);trace=[];lift_started=None;bilateral_since=None;stop='';latched=None;maxgap=0.;maxpen=0.;lift_samples=[]
    renderer=mujoco.Renderer(model,height=480,width=640);cam=camera([0,0,.16],.5)
    video_tmp=Path(tempfile.gettempdir())/'issue46_bottle_contact.mp4';writer=imageio.get_writer(str(video_tmp),fps=30,codec='libx264')
    for step in range(6000):
        t=step*model.opt.timestep
        command=Q_OPEN if t<.5 else max(Q_CLOSED,Q_OPEN-(t-.5)*.15)
        if latched is not None:command=latched
        opening=float(np.clip((command-Q_CLOSED)/(Q_OPEN-Q_CLOSED),0.,1.))
        api_command=set_gripper('right',opening,model,data,amap)
        actual_cad=float(PHASE-data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]])
        data.ctrl[liftact]=0. if lift_started is None else .08*min(1.,max(0.,(t-lift_started)/1.2))
        mujoco.mj_step(model,data)
        cs=contacts(model,data);normals={s:sum(c['normal_N'] for c in cs if c['side']==s) for s in ['narrow','wide','table','other']}
        bilateral=normals['narrow']>1 and normals['wide']>1
        if bilateral:
            if bilateral_since is None:bilateral_since=t
            if latched is None and t-bilateral_since>.15:latched=max(Q_CLOSED,actual_cad-.01)
        else:bilateral_since=None
        if lift_started is None and bilateral_since is not None and t-bilateral_since>.5 and min(normals['narrow'],normals['wide'])>3:
            lift_started=t
        gap=max(float(np.linalg.norm(data.site(f'candidate_{s}_D_jaw').xpos-data.site(f'candidate_{s}_D_loop').xpos)) for s in JOINTS);maxgap=max(maxgap,gap)
        pen=max((c['penetration_m'] for c in cs),default=0.);maxpen=max(maxpen,pen)
        if lift_started is not None:lift_samples.append({'bilateral':bilateral,'table':normals['table'],'other':normals['other']})
        if step%10==0:
            trace.append({'time_s':float(data.time),'command_CAD':command,'command_opening':opening,'actual_CAD':actual_cad,'bottle_z':float(data.xpos[bid,2]),'bottle_v':data.cvel[bid].tolist(),'normal_N':normals,'contacts':cs,'D_gap_m':gap,'actuator_force':data.actuator_force.tolist(),'fixture_z':float(data.qpos[model.jnt_qposadr[jid(model,'fixture_lift')]]),'motor_api':api_command})
        if step%33==0:
            renderer.update_scene(data,camera=cam);writer.append_data(renderer.render())
        if step in [500,2000,3500,5000]:
            renderer.update_scene(data,camera=cam);tmp=Path(tempfile.gettempdir())/f'issue46_contact_frame_{step:04d}.png';imageio.imwrite(tmp,renderer.render());pg.write_new_bytes(run/tmp.name,tmp.read_bytes(),environment='wsl')
        rootq=data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]]
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):stop='nonfinite state'
        elif not -1.0001<=rootq<=.0001:stop='root limit violation'
        elif pen>.003:stop='external contact penetration >3mm'
        elif gap>.005:stop='D tracking gap >5mm'
        elif normals['other']>.5:stop='non-jaw gripper contact with bottle'
        elif t>3.5 and lift_started is None:stop='bilateral preload not established; lift prohibited'
        if stop:break
    writer.close();renderer.close();pg.write_new_bytes(run/'bottle_contact.mp4',video_tmp.read_bytes(),environment='wsl');dz=float(data.xpos[bid,2])-initial_z
    fraction=sum(x['bilateral'] for x in lift_samples)/len(lift_samples) if lift_samples else 0
    passlift=dz>.05 and fraction>.95 and bool(lift_samples) and lift_samples[-1]['table']<.1 and lift_samples[-1]['other']<.1 and not stop
    result={'status':'PHYSICAL_LIFT_PASS' if passlift else 'PHYSICAL_CONTACT_FAIL','physics_steps':step+1,'sim_time_s':float(data.time),'bottle_rise_m':dz,'bilateral_lift_fraction':fraction,'lift_started_s':lift_started,'stop_reason':stop,'max_D_gap_m':maxgap,'max_external_penetration_m':maxpen,'initial_bottle_qpos_writes':0,'rollout_bottle_qpos_writes':0,'initial_state':'one-time CAD pose initialization before first step','rollout_actuation':'right_claw_joint only; B/C/E and loop hinges passive','fixture':'actuated isolated mount; only gripper is supported','canonical_bottle_mass_kg':.57,'canonical_bottle_body_diameter_m':.07,'neck_diameter_m':.036,'internal_pair_exclusions':overlay['internal_pair_exclusions']}
    write(run,'contact_trace.json',trace);write(run,'contact_result.json',result);write(run,'contact_overlay.json',overlay);print(json.dumps(result,indent=2));return result


def rollout_test_cylinder(run):
    """One free-object contact/lift probe after the source-range stroke passed."""
    kin=Linkage();model,overlay=build(run,'test-cylinder');data=mujoco.MjData(model)
    open_targets=kin.targets(Q_OPEN)[0];qpos_initial(model,data,open_targets)
    object_jid=jid(model,'test_cylinder_free');object_adr=model.jnt_qposadr[object_jid]
    data.qpos[object_adr:object_adr+7]=model.qpos0[object_adr:object_adr+7]
    mujoco.mj_forward(model,data)
    initial_object_contacts=contacts(model,data,'test_cylinder')
    initial_all_contacts=all_contacts(model,data)
    initial_jaw_contacts=[c for c in initial_object_contacts if c['side'] in ('narrow','wide')]
    if any(c['penetration_m']>.001 for c in initial_jaw_contacts):
        raise ValueError('test cylinder intersects the open candidate; no physical step taken')
    initial_shelf_robot=[c for c in initial_all_contacts if 'test_shelf' in (c['geom1'] or '')+(c['geom2'] or '') and 'test_cylinder' not in c['body1']+c['body2']]
    if any(c['penetration_m']>.001 for c in initial_shelf_robot):
        raise ValueError('test shelf intersects the open candidate; no physical step taken')
    amap={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a):a for a in range(model.nu)}
    rootact=amap['right_claw_joint_servo'];liftact=amap['fixture_lift_servo']
    data.ctrl[rootact]=open_targets['right_claw_joint'];data.ctrl[liftact]=0.
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'test_cylinder')
    fixture_jid=jid(model,'fixture_lift');fixture_adr=model.jnt_qposadr[fixture_jid]
    initial_z=float(data.xpos[bid,2]);trace=[];lift_started=None;bilateral_since=None;stop='';maxgap=0.;maxpen=0.;maxqvel=0.;maxspeed=0.;maxcoupling=0.;maxforces={name:0. for name in equality_force_magnitudes(model,data)};lift_samples=[];first_contact=None;first_bilateral=None
    renderer=mujoco.Renderer(model,height=480,width=640);cam=camera([0,0,.09],.42);tempdir=tempfile.TemporaryDirectory(prefix='issue46-test-cylinder-');video_tmp=Path(tempdir.name)/'test_cylinder_contact.mp4';writer=imageio.get_writer(str(video_tmp),fps=30,codec='libx264')
    joint_dofs=[int(model.jnt_dofadr[jid(model,n)]) for side,names in JOINTS.items() for n in names]
    total_steps=6000;settled_z=None
    for step in range(total_steps):
        t=step*model.opt.timestep
        if t<.5:q=Q_OPEN
        elif t<1.9:q=Q_OPEN+(Q_CLOSED-Q_OPEN)*(t-.5)/1.4
        else:q=Q_CLOSED
        opening=float(np.clip((q-Q_CLOSED)/(Q_OPEN-Q_CLOSED),0.,1.))
        motor_api=set_gripper('right',opening,model,data,amap)
        fixture_target=0. if lift_started is None else min(.08,.08*(t-lift_started)/1.2)
        data.ctrl[liftact]=fixture_target
        mujoco.mj_step(model,data)
        cs=contacts(model,data,'test_cylinder')
        normals={side:sum(c['normal_N'] for c in cs if c['side']==side) for side in ['narrow','wide','table','other']}
        jaw_contact=normals['narrow']>.2 or normals['wide']>.2
        if jaw_contact and first_contact is None:first_contact=t
        bilateral=normals['narrow']>.5 and normals['wide']>.5
        if bilateral:
            if bilateral_since is None:bilateral_since=t
            if first_bilateral is None:first_bilateral=t
        else:bilateral_since=None
        if settled_z is None and t>=.4 and normals['table']>.1:settled_z=float(data.xpos[bid,2])
        if lift_started is None and bilateral_since is not None and t-bilateral_since>=.25:
            lift_started=t
        gap=max(float(np.linalg.norm(data.site(f'candidate_{s}_D_jaw').xpos-data.site(f'candidate_{s}_D_loop').xpos)) for s in JOINTS);maxgap=max(maxgap,gap)
        pen=max((c['penetration_m'] for c in cs),default=0.);maxpen=max(maxpen,pen)
        coupling=candidate_B_residuals(model,data);maxcoupling=max(maxcoupling,max(abs(v) for v in coupling.values()))
        qvel=max(float(np.max(np.abs(data.qvel[dof]))) for dof in joint_dofs);maxqvel=max(maxqvel,qvel)
        object_speed=float(np.linalg.norm(data.cvel[bid,3:6]));maxspeed=max(maxspeed,object_speed)
        forces=equality_force_magnitudes(model,data)
        for name,value in forces.items():maxforces[name]=max(maxforces[name],value)
        if lift_started is not None:lift_samples.append({'bilateral':bilateral,'table_normal_N':normals['table'],'other_normal_N':normals['other'],'jaw_normals_N':{'narrow':normals['narrow'],'wide':normals['wide']}})
        if step%10==0:
            trace.append({'time_s':float(data.time),'command_CAD':float(q),'command_opening':opening,'actual_CAD':float(PHASE-data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]]),'root_q':float(data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]]),'wide_root_q':float(data.qpos[model.jnt_qposadr[jid(model,'R_hand_wide1_joint')]]),'passive_q':{name:float(data.qpos[model.jnt_qposadr[jid(model,name)]]) for side,names in JOINTS.items() for name in names[1:]},'candidate_B_coupling_residual_rad':coupling,'D_gap_m':gap,'fixture_lift_m':float(data.qpos[fixture_adr]),'object_position_m':data.xpos[bid].tolist(),'object_linear_speed_m_s':object_speed,'normal_N':normals,'contacts':cs,'motor_api':motor_api,'root_actuator_force_Nm':float(data.actuator_force[rootact]),'fixture_lift_actuator_N':float(data.actuator_force[liftact]),'constraint_generalized_force_abs':forces})
        if step%33==0:
            renderer.update_scene(data,camera=cam);writer.append_data(renderer.render())
        rootq=float(data.qpos[model.jnt_qposadr[jid(model,'right_claw_joint')]]);wideq=float(data.qpos[model.jnt_qposadr[jid(model,'R_hand_wide1_joint')]])
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):stop='nonfinite state'
        elif not -1.00001<=rootq<=.00001 or not -.00001<=wideq<=1.00001:stop='official root limit violation'
        elif gap>.005:stop='D-site closure error exceeded 5 mm'
        elif maxcoupling>.01:stop='Candidate B coupling residual exceeded 0.01 rad'
        elif maxqvel>25:stop='runaway joint velocity above 25 rad/s'
        elif pen>.003:stop='test-object contact penetration exceeded 3 mm'
        elif normals['other']>.5:stop='non-jaw gripper contact with test object'
        elif t>3.2 and lift_started is None:stop='bilateral contact/preload not established; lift prohibited'
        if stop:break
    writer.close();renderer.close();pg.write_new_bytes(run/'test_cylinder_contact.mp4',video_tmp.read_bytes(),environment='wsl');tempdir.cleanup()
    settled_z=initial_z if settled_z is None else settled_z;final_z=float(data.xpos[bid,2]);rise=final_z-settled_z
    bilateral_fraction=sum(x['bilateral'] for x in lift_samples)/len(lift_samples) if lift_samples else 0.
    final_table_force=lift_samples[-1]['table_normal_N'] if lift_samples else 0.
    final_other_force=lift_samples[-1]['other_normal_N'] if lift_samples else 0.
    passed=rise>.05 and bilateral_fraction>.95 and bool(lift_samples) and final_table_force<.1 and final_other_force<.1 and not stop
    result={'status':'TEST_CYLINDER_LIFT_PASS' if passed else 'TEST_CYLINDER_CONTACT_FAIL','mujoco_version':mujoco.__version__,'physics_steps':step+1,'simulated_time_s':float(data.time),'stop_reason':stop,'object':{'type':'free vertical cylinder','diameter_m':.018,'height_m':.020,'mass_kg':.03,'initial_center_m':[0.,0.,.125]},'support':'independent 12 mm square shelf; test cylinder remains a free body with no attachment','object_initial_qpos_writes':1,'object_rollout_qpos_writes':0,'active_gripper_motor':'right_claw_joint_servo','passive_actuator_count':0,'passive_qpos_writes_after_start':0,'initial_object_contacts':initial_object_contacts,'initial_all_contacts':initial_all_contacts,'initial_shelf_robot_contacts':initial_shelf_robot,'first_gripper_contact_s':first_contact,'first_bilateral_contact_s':first_bilateral,'lift_started_s':lift_started,'settled_reference_z_m':settled_z,'final_object_z_m':final_z,'object_rise_m':rise,'bilateral_lift_fraction':bilateral_fraction,'final_table_normal_N':final_table_force,'final_other_normal_N':final_other_force,'max_D_gap_m':maxgap,'max_candidate_B_coupling_residual_rad':maxcoupling,'max_constraint_generalized_force_abs':maxforces,'max_internal_contact_penetration_m':maxpen,'max_gripper_joint_speed_rad_s':maxqvel,'max_object_linear_speed_m_s':maxspeed,'internal_pair_exclusions':overlay['internal_pair_exclusions'],'canonical_bottle_tested':False,'canonical_bottle_reason':'36 mm neck exceeds measured 35.939 mm source-range maximum aperture; no clearance margin'}
    write(run,'test_cylinder_contact_trace.json',trace);write(run,'test_cylinder_contact_result.json',result);write(run,'contact_result.json',result);write(run,'contact_overlay.json',overlay)
    print(json.dumps(result,indent=2));return result


VIEWER='''<!doctype html><html><meta charset="utf-8"><title>Official OmniPicker CAD-driven motion</title><style>body{margin:0;background:#171e24;color:#eee;font:14px system-ui}header{padding:12px}canvas{display:block}#view{height:75vh}input{width:65%}button{padding:8px}label{display:inline-block;margin:4px}</style><script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"}}</script><header><b>UNVERIFIED_EXPERIMENTAL — CAD-driven official STL kinematics</b><p>Zero physics steps in this preview. No commercial travel or manufacturer transmission claim.</p><button id="play">Play</button><input id="slider" type="range" min="0" max="120" value="0"><span id="value"></span><div id="toggles"></div></header><div id="view"></div><script type="module">import * as T from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';const data=await(await fetch('official_mesh_scene.json')).json();const view=document.querySelector('#view'),r=new T.WebGLRenderer({antialias:true});view.append(r.domElement);const s=new T.Scene();s.background=new T.Color('#171e24');s.add(new T.HemisphereLight(0xffffff,0x404050,2));const l=new T.DirectionalLight(0xffffff,3);l.position.set(.3,-.4,.5);s.add(l);const c=new T.PerspectiveCamera(35,1,.001,10);c.up.set(0,0,1);c.position.set(.25,-.3,.2);const controls=new OrbitControls(c,r.domElement);controls.target.set(0,0,.085);controls.update();const meshes=data.meshes.map(m=>{const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(m.vertices,3));g.setIndex(m.faces);g.computeVertexNormals();const mesh=new T.Mesh(g,new T.MeshStandardMaterial({color:new T.Color(...m.color),side:T.DoubleSide,roughness:.6}));mesh.matrixAutoUpdate=false;s.add(mesh);const label=document.createElement('label');const box=document.createElement('input');box.type='checkbox';box.checked=true;box.onchange=()=>mesh.visible=box.checked;label.append(box,document.createTextNode(m.name.replace('__visual','')));document.querySelector('#toggles').append(label);return mesh});const slider=document.querySelector('#slider');function set(){const i=Number(slider.value),f=data.frames[i];meshes.forEach((m,k)=>m.matrix.fromArray(f.transforms[k]));document.querySelector('#value').textContent=`frame ${i+1}/121 · CAD q=${f.q.toFixed(3)} rad`};slider.oninput=set;set();let playing=false,last=0;document.querySelector('#play').onclick=()=>playing=!playing;function loop(t){requestAnimationFrame(loop);if(playing&&t-last>33){slider.value=(Number(slider.value)+1)%121;set();last=t}const w=view.clientWidth,h=view.clientHeight;r.setSize(w,h,false);c.aspect=w/h;c.updateProjectionMatrix();controls.update();r.render(s,c)}loop(0);window.viewerReady=true;</script></html>'''

VIEWER=VIEWER.replace(
    '<b>UNVERIFIED_EXPERIMENTAL — CAD-driven official STL kinematics</b><p>Zero physics steps in this preview. No commercial travel or manufacturer transmission claim.</p>',
    '<b>UNVERIFIED_EXPERIMENTAL — Candidate B coupling with official STL meshes</b><p>Source-sampled q range [-0.10, 0.00]; zero physics steps. Candidate coupling is not manufacturer verified.</p>'
)


def finalize_run(run):
    from agent.path_governance import write_artifact_manifest
    physical=json.load((run/'single_motor_physics_result.json').open())
    contact_path=run/'contact_result.json'
    contact_result=json.load(contact_path.open()) if contact_path.is_file() else {'status':'NOT RUN; requires a successful coupled source-range stroke and suitable aperture'}
    aperture=json.load((run/'surface_aperture_measurements.json').open()) if (run/'surface_aperture_measurements.json').is_file() else None
    force_analysis=json.load((run/'contact_force_analysis.json').open()) if (run/'contact_force_analysis.json').is_file() else None
    final={'physics':physical,'contact':contact_result,'aperture_measurements':aperture,'contact_force_analysis':force_analysis,'official_urdf_sha256':digest(SOURCE/'X2-Ultra_omnipicker.urdf'),'step_sha256':digest(STEP_SOURCE),'class':'SIMULATION_ONLY; UNVERIFIED_EXPERIMENTAL mechanism coupling'}
    write(run,'final_result.json',final)
    relation='(q_root - qpos0_root) + q_B - q_E = 0 per branch'
    report=(f"# Issue #46 — Candidate B Motor-Only OmniPicker Experiment\n\n"
            f"- Candidate B coupling motion: `{physical['status']}`.\n"
            f"- Object contact/lift: `{contact_result['status']}`.\n"
            f"- MuJoCo: {physical['mujoco_version']}, {physical['steps']} steps ({physical['simulated_time_s']:.3f} s).\n"
            f"- Right root motor is the only gripper actuator. B/C/E and loop hinges remain passive; passive qpos writes after initial CAD-pose initialization: {physical['passive_qpos_writes_after_start']}.\n"
            f"- Initial local mobility after Candidate B coupling: {physical['initial_mobility']}. Candidate B `{relation}` is an UNVERIFIED_EXPERIMENTAL simulation constraint.\n"
            f"- Candidate equality solver reference: `{physical.get('candidate_equality_solref','not recorded')}`; max D gap `{physical['max_D_gap_m']:.6g} m`, max Candidate B residual `{physical['max_candidate_B_coupling_residual_rad']:.6g} rad`, peak joint speed `{physical['max_qvel_rad_s']:.6g} rad/s`.\n"
            f"- Actual source-range open/close with one root motor: `{physical['status']}`, stroke `{physical['CAD_root_stroke_rad']:.6f} rad`, `{physical['steps']}` `mj_step` calls.\n"
            f"- Physical object result: `{contact_result.get('status','not recorded')}`.\n"
            f"- Original collision baseline contacts at open pose: {len(physical['original_collision_baseline_initial_contacts'])}; diagnostic startup contacts after scoped exclusions: {len(physical['diagnostic_model_initial_contacts'])}. External object/finger and object/support contacts stayed enabled.\n"
            + (f"- Surface aperture: max `{aperture['maximum_open_width_in_measured_band_mm']:.3f} mm` at q=0; the 36 mm canonical neck has `{aperture['canonical_neck_clearance_mm']:.3f} mm` clearance and was not tested.\n" if aperture else "")
            + (f"- Test cylinder: {contact_result.get('object',{})}; bilateral contact result is recorded in `contact_force_analysis.json`; lift was `{contact_result.get('object_rise_m',0.0):.4f} m` and status is not a grasp pass.\n" if force_analysis else "")
            + (f"- Contact forces: both jaw normals exceeded 0.2 N for `{force_analysis['bilateral_over_0_2_N_duration_s']:.3f} s` (peaks `{force_analysis['peak_narrow_normal_N']:.3f}/{force_analysis['peak_wide_normal_N']:.3f} N`), but the lift gate required 0.5 N per side. During bilateral contact, the jaws' combined vertical force stayed downward (`{force_analysis['jaw_vertical_force_sum_min_N']:.3f}..{force_analysis['jaw_vertical_force_sum_max_N']:.3f} N`); the shelf carried the load.\n" if force_analysis else "")
            + "- Right wrist collision geometry is outside the isolated fixture; see its separate static triangle evidence.\n"
            + "- Full X2 mounting, manufacturer transmission, and hardware equivalence remain unverified.\n")
    write(run,'REPORT.md',report)
    manifest=write_artifact_manifest(run,environment='wsl')
    return {'final':final,'artifact_manifest':manifest['manifest'],'run_dir':str(run)}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['all','prepare','physics','contact','probe','finalize','test']);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.mode=='test':print(json.dumps(Linkage().test()));return
    if a.run_dir is None:p.error('--run-dir required')
    pg.validate_output_path(a.run_dir,require_exists=True,expected_kind='directory')
    if a.mode=='prepare':prepare(a.run_dir)
    elif a.mode=='physics':
        result=physics_open_close(a.run_dir);print(json.dumps(result,indent=2))
    elif a.mode=='contact':
        prior=json.load((a.run_dir/'single_motor_physics_result.json').open())
        if prior.get('status')!='SINGLE_MOTOR_PHYSICS_PASS':raise ValueError('bottle contact is gated on a successful root-only passive-motion rollout')
        result=rollout(a.run_dir);print(json.dumps(result,indent=2))
    elif a.mode=='probe':
        prior=json.load((a.run_dir/'single_motor_physics_result.json').open())
        if prior.get('status')!='SINGLE_MOTOR_PHYSICS_PASS':raise ValueError('free test-cylinder contact is gated on a successful Candidate B open/close rollout')
        result=rollout_test_cylinder(a.run_dir);print(json.dumps(result,indent=2))
    elif a.mode=='finalize':
        print(json.dumps(finalize_run(a.run_dir),indent=2))
    else:
        prepare(a.run_dir)
        physical=physics_open_close(a.run_dir)
        if physical.get('status')=='SINGLE_MOTOR_PHYSICS_PASS':
            placement=json.load((a.run_dir/'placement.json').open())
            contact_result=rollout(a.run_dir) if 'source_neck_center_m' in placement else rollout_test_cylinder(a.run_dir)
        print(json.dumps(finalize_run(a.run_dir),indent=2))


if __name__=='__main__':main()
