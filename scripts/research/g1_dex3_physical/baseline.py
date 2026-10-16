"""Isolated research: official G1/Dex3 torque-controlled physical baseline."""
from pathlib import Path
import hashlib, json, subprocess, xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

SOURCE_REV='5994d4faef0a9cadd3287f8de0199a67eeb2a259'
UPSTREAM=Path('/home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/upstream/unitree_ros')
SOURCE=UPSTREAM/'robots/g1_description/g1_29dof_with_hand_rev_1_0.xml'
REVIEW=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue43-g1-dex3-visual-review-20261015/g1_29dof_with_hand_rev_1_0_review.xml')
OUT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-43/dex3-physical-20261016')
ARM=['right_'+s+'_joint' for s in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
HAND=['right_hand_'+s+'_joint' for s in ['thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1']]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def name(m,obj,i):return mj.mj_id2name(m,obj,int(i)) or f'unnamed_{i}'
def jid(m,n):return mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)
def bid(m,n):return mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)
def gid(m,n):return mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n)
def indices(m,names):
    js=np.array([jid(m,n) for n in names]);return m.jnt_qposadr[js],m.jnt_dofadr[js]

def build(out):
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==SOURCE_REV
    assert not subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain'],text=True).strip()
    assert sha(SOURCE)=='77c98e7d34e428c6cdbd1e0d665e989cbc43bf17abb8a7c048494b4bc51ead88'
    assert sha(REVIEW)=='dec332c6152d27a5a39bc4b4f521edbaa4cb97e6436a858b7a0aa406211f052e'
    root=ET.parse(REVIEW).getroot()
    pelvis=root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@name='floating_base_joint']"))
    opt=root.find('option')
    if opt is None:opt=ET.SubElement(root,'option')
    opt.set('timestep','0.0005');opt.set('integrator','implicitfast')
    table=root.find(".//body[@name='m0_table']");table.set('pos','0.55 -0.30 0')
    bottle=root.find(".//body[@name='green_box']");bottle.set('pos','0.38 -0.22 0.9175')
    for i,g in enumerate(root.findall('.//geom')):
        if 'name' not in g.attrib:g.set('name',f'official_geom_{i}')
    for motor in root.findall('./actuator/motor'):
        hand='hand' in motor.get('joint');kp,kd=(1.5,.1) if hand else (120.,5.)
        joint=root.find(".//joint[@name='%s']"%motor.get('joint'))
        motor.tag='general';motor.set('gaintype','fixed');motor.set('biastype','affine')
        motor.set('gainprm',str(kp));motor.set('biasprm',f'0 {-kp} {-kd}')
        motor.set('forcelimited','true');motor.set('forcerange',joint.get('actuatorfrcrange'))
    out.parent.mkdir(parents=True,exist_ok=True);ET.ElementTree(root).write(out)
    return mj.MjModel.from_xml_path(str(out))

def init(m):
    d=mj.MjData(m)
    # Joint-limit interior, near-open hand; no active rollout state writes.
    for j in range(m.njnt):
        if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE:
            lo,hi=m.jnt_range[j];d.qpos[m.jnt_qposadr[j]]=np.clip(0,lo+0.015,hi-0.015)
    mj.mj_forward(m,d)
    return d

def contacts(m,d):
    result=[]
    for i in range(d.ncon):
        c=d.contact[i];f=np.zeros(6);mj.mj_contactForce(m,d,i,f)
        result.append({'geoms':[name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in c.geom],
           'bodies':[name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) for g in c.geom],
           'distance':float(c.dist),'force':f.tolist(),'active':int(c.efc_address)>=0})
    return result

def dynamics_hold(m,d,duration=1):
    js=m.actuator_trnid[:,0];qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js]
    target=d.qpos[qa].copy();hand=np.array(['hand' in name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in js])
    kp=np.where(hand,1.5,120.);kd=np.where(hand,.1,5.)
    cap=np.maximum(abs(m.jnt_actfrcrange[js,0]),abs(m.jnt_actfrcrange[js,1]))
    rows=[];maxvel=0.;minlim=1.;worstcontacts=[]
    for step in range(round(duration/m.opt.timestep)):
        d.ctrl[:]=target+np.where(hand,0,d.qfrc_bias[va]/kp)
        mj.mj_step(m,d)
        maxvel=max(maxvel,float(abs(d.qvel[va]).max()))
        if step%20==0:
            cs=contacts(m,d);rows.append({'time':d.time,'q':d.qpos[qa].tolist(),'v':d.qvel[va].tolist(),'effort':d.actuator_force.tolist(),'contacts':cs})
            worstcontacts+= [c for c in cs if c['distance']<-.001]
        if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite')
    return {'maxvel':maxvel,'final_contacts':contacts(m,d),'penetrations':worstcontacts[:10],'rows':rows}

if __name__=='__main__':
    m=build(OUT/'fixed_base_model.xml');d=init(m)
    audit={'model_sha256':sha(OUT/'fixed_base_model.xml'),'source_sha':SOURCE_REV,'initial_contacts':contacts(m,d),
       'joints':[{'name':name(m,mj.mjtObj.mjOBJ_JOINT,j),'range':m.jnt_range[j].tolist(),'effort':m.jnt_actfrcrange[j].tolist()} for j in range(m.njnt)],
       'wrist_position':d.xpos[bid(m,'right_wrist_yaw_link')].tolist(),'wrist_rotation':d.xmat[bid(m,'right_wrist_yaw_link')].reshape(3,3).tolist()}
    (OUT/'initial_audit.json').write_text(json.dumps(audit,indent=2))
    h=dynamics_hold(m,d);(OUT/'hold_audit.json').write_text(json.dumps(h,indent=2))
    print(json.dumps({k:v for k,v in h.items() if k!='rows'},indent=2))
