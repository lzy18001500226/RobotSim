"""Bounded seven-pose, two-height screen using the existing circle IK.

All qpos changes here belong to offline planning MjData, never the live robot.
Passing points are geometric samples, not a guaranteed continuous or balanced
region. Physical validation is recorded separately by the native runner.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg

ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bottle_height_variant(source, state, height_m):
    """Only the separately requested bottle height, centre and inertia change."""
    if height_m not in (.15,.30):raise ValueError('only the two authorized bottle heights')
    root=ET.fromstring(source.read_bytes())
    geom=root.find('.//geom[@name="test_cylinder_collision"]')
    body=root.find('.//body[@name="test_cylinder"]')
    if geom is None or body is None:raise ValueError('canonical bottle missing')
    radius,half=np.fromstring(geom.get('size'),sep=' ')
    if abs(radius-.035)>1e-9 or abs(half-.15)>1e-9 or geom.get('mass')!='0.2':raise ValueError('unexpected canonical bottle')
    position=np.fromstring(body.get('pos'),sep=' ');position[2]+=height_m/2-half
    body.set('pos',' '.join(map(str,position)));geom.set('size',f'{radius} {height_m/2}')
    modified=copy.deepcopy(state)
    q=np.asarray(modified['object_pose_world']['quaternion_wxyz'])
    axis=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()[:,2]
    modified['object_pose_world']['xyz_m']=(np.asarray(modified['object_pose_world']['xyz_m'])+(height_m/2-half)*axis).tolist()
    return ET.tostring(root,encoding='utf-8',xml_declaration=True),modified

def source_roll_diagnostic(record_path):
    """Three source-bound seeds; add only the existing waist-roll coordinate.

    No mechanical model edit or motion command. A passing endpoint still needs
    approach/lift/balance and supported physical waist handoff qualification.
    """
    import mujoco
    from scipy.optimize import least_squares
    from scripts.research.issue46.grasp import stl_approved_circle_ik as circle_ik
    ik=circle_ik.ik
    record_path=Path(record_path).resolve();record=json.loads(record_path.read_text())
    scene=Path(record['scene']['path'])
    if digest(scene)!=record['scene']['sha256']:raise ValueError('scene source changed')
    state_path=Path(record['standing_state']);state=json.loads(state_path.read_text())
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    names=('waist_yaw_joint','waist_pitch_joint','waist_roll_joint',*ik.ARM)
    bounds=np.array([model.jnt_range[ik.native.jid(model,n)] for n in names])
    dofs=np.array([model.jnt_dofadr[ik.native.jid(model,n)] for n in names])
    start=np.array([state['waist_positions_rad'][n] if n.startswith('waist') else state['arm_positions_rad'][n] for n in names])
    best=record['rows'][0]['q_rad']
    seed=np.array([best.get(n,start[i]) for i,n in enumerate(names)])
    target=np.array(record['grasp_section_center_world_m'])
    R=Rotation.from_quat(np.array(state['object_pose_world']['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
    axis=R[:,2];center=np.array(record['circle']['center_m'])
    local_axis=np.array(record['local_bottle_axis_in_hand']);hand=ik.native.bid(model,'R_omnipicker_base_link')
    local=copy.deepcopy(state)
    def setpose(q,opening=40):
        local['waist_positions_rad']['waist_roll_joint']=float(q[2])
        ik.set_pose(model,data,local,np.r_[q[:2],q[3:]],np.zeros(3),0.,opening,np.zeros(3))
    def residual(q):
        setpose(q)
        return np.r_[circle_ik.axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,target,axis,local_axis),1e-6*(q-start)]
    def jac(q):
        residual(q)
        return np.vstack([circle_ik.circle_task_jacobian(model,data,hand,center,local_axis,dofs),1e-6*np.eye(len(q))])
    from scipy.spatial import ConvexHull
    setpose(start,100)
    floor_points=[]
    for contact in data.contact:
        names_at_contact=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in (contact.geom1,contact.geom2)]
        if any(model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in (contact.geom1,contact.geom2)) and any(n in ('left_ankle_roll_link','right_ankle_roll_link') for n in names_at_contact):
            floor_points.append(contact.pos[:2].copy())
    support=ConvexHull(np.unique(np.array(floor_points),axis=0))
    robot_root=model.body_rootid[ik.native.bid(model,'pelvis')]
    robot=model.body_rootid==robot_root
    def self_penetrations():
        pairs={}
        for co in data.contact:
            if co.efc_address<0 or co.dist>=-.001:continue
            b1,b2=int(model.geom_bodyid[co.geom1]),int(model.geom_bodyid[co.geom2])
            if model.body_rootid[b1]!=robot_root or model.body_rootid[b2]!=robot_root:continue
            key=tuple(sorted((int(co.geom1),int(co.geom2))))
            pairs[key]=min(pairs.get(key,0.),float(co.dist))
        return pairs
    baseline_self={}
    for opening in (100,40):
        setpose(start,opening);baseline_self[opening]=self_penetrations()
    def new_self_penetrations(opening):
        bad=[]
        for pair,depth in self_penetrations().items():
            if depth<baseline_self[opening].get(pair,-.001)-.0005:
                bad.append({'geoms':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in pair],'depth_m':depth,'baseline_m':baseline_self[opening].get(pair)})
        return bad
    def support_margin():
        com=(data.xipos[robot]*model.body_mass[robot,None]).sum(axis=0)/model.body_mass[robot].sum()
        margin=float(-np.max(support.equations[:,:2]@com[:2]+support.equations[:,2]))
        return margin,com.tolist()
    original_target=target.copy()
    original_object=np.array(local['object_pose_world']['xyz_m'])
    rows=[]
    for roll_seed in (0.,-.2,.2):
        q=seed.copy();q[2]=roll_seed
        result=least_squares(residual,np.clip(q,bounds[:,0]+.01101,bounds[:,1]-.01101),jac=jac,bounds=(bounds[:,0]+.011,bounds[:,1]-.011),max_nfev=220,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        error=residual(result.x)[:6]
        angle=float(2*np.arcsin(np.clip(np.linalg.norm(error[3:])/.2,0,1)))
        item={'roll_seed_rad':roll_seed,'q_rad':dict(zip(names,map(float,result.x))),'position_error_m':float(np.linalg.norm(error[:3])),'axis_error_rad':angle,'evaluations':result.nfev,'source_margin_rad':float(np.min(np.minimum(result.x-bounds[:,0],bounds[:,1]-result.x)))}
        contacts={}
        for opening in (100,40,39):
            setpose(result.x,opening)
            pads,bad,table,robot_table=ik.contacts(model,data,ik.native.bid(model,'test_cylinder'))
            contacts[opening]={'pads':pads,'forbidden_object_bodies':bad,'robot_table_bodies':robot_table,'table_support':table}
        item['endpoint_contacts']=contacts
        item['endpoint_geometry_pass']=item['position_error_m']<=1e-4 and angle<=1e-3 and all(not v['forbidden_object_bodies'] and not v['robot_table_bodies'] for v in contacts.values()) and not contacts[100]['pads'] and len(contacts[40]['pads'])==2
        approach=[];margins=[]
        for u in np.linspace(0.,1.,21):
            blend=3*u*u-2*u*u*u;setpose(start+blend*(result.x-start),100)
            pads,bad,table,robot_table=ik.contacts(model,data,ik.native.bid(model,'test_cylinder'))
            margin,com=support_margin();margins.append(margin);self_bad=new_self_penetrations(100)
            approach.append({'fraction':float(u),'pass':not pads and not bad and not robot_table and not self_bad and margin>=.005,'new_active_self_penetrations':self_bad,'pad_contact':pads,'nonpad':bad,'robot_table':robot_table,'static_com_margin_m':margin,'com_world_m':com})
        lift=[];q_lift=result.x.copy()
        for h in np.linspace(0.,.06,13):
            target[:]=original_target+[0.,0.,h]
            local['object_pose_world']['xyz_m']=(original_object+[0.,0.,h]).tolist()
            refined=least_squares(residual,q_lift,jac=jac,bounds=(bounds[:,0]+.011,bounds[:,1]-.011),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            q_lift=refined.x;err=residual(q_lift)[:6]
            angle_lift=float(2*np.arcsin(np.clip(np.linalg.norm(err[3:])/.2,0,1)))
            pads,bad,table,robot_table=ik.contacts(model,data,ik.native.bid(model,'test_cylinder'))
            margin,com=support_margin();margins.append(margin);self_bad=new_self_penetrations(40)
            lift.append({'lift_m':float(h),'pass':np.linalg.norm(err[:3])<=1e-4 and angle_lift<=1e-3 and len(pads)==2 and not bad and not robot_table and not self_bad and margin>=.005,'new_active_self_penetrations':self_bad,'position_error_m':float(np.linalg.norm(err[:3])),'axis_error_rad':angle_lift,'pads':pads,'nonpad':bad,'robot_table':robot_table,'q_rad':dict(zip(names,map(float,q_lift))),'static_com_margin_m':margin,'com_world_m':com})
        target[:]=original_target;local['object_pose_world']['xyz_m']=original_object.tolist()
        item.update(approach_sweep=approach,lift_sweep=lift,minimum_static_com_margin_m=min(margins),static_sweeps_pass=bool(item['endpoint_geometry_pass'] and all(v['pass'] for v in approach) and all(v['pass'] for v in lift)))
        rows.append(item)
    out=pg.create_unique_output_directory(ROOT,prefix='150mm-existing-waist-roll-diagnostic').native_path
    record={'classification':'ZERO_STEP_EXISTING_SOURCE_WAIST_ROLL_COORDINATE_DIAGNOSTIC','input':str(record_path),'input_sha256':digest(record_path),'scene':str(scene),'scene_sha256':digest(scene),'standing_state':str(state_path),'standing_state_sha256':digest(state_path),'joint_names':names,'source_bounds_rad':bounds.tolist(),'root_and_legs_frozen':True,'geometry_joints_or_source_limits_modified':False,'rows':rows,'mj_step_calls':0,'dynamics':'NOT_RUN','not_yet_qualified':['dynamic balance (static COM is only a screen)','MC waist handoff','physical grasp']}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    print(json.dumps({'output':str(out),'endpoint_passes':sum(r['endpoint_geometry_pass'] for r in rows),'rows':[{k:r[k] for k in ('roll_seed_rad','position_error_m','axis_error_rad','endpoint_geometry_pass','static_sweeps_pass','minimum_static_com_margin_m')} for r in rows]}))
    return out/'result.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--capture',type=Path);g.add_argument('--source-roll-diagnostic',type=Path)
    a=p.parse_args()
    if a.source_roll_diagnostic:source_roll_diagnostic(a.source_roll_diagnostic);return
    capture=json.loads((a.capture/'result.json').read_text())
    if capture.get('gates',{}).get('HAL_PHYSICAL_LIFT')!='PASS':raise ValueError('start from an actual qualified lift')
    replan=Path(capture['hal_arm_runtime_replan']['path'])
    if digest(replan)!=capture['hal_arm_runtime_replan']['sha256']:raise ValueError('replan hash mismatch')
    plan=json.loads(replan.read_text());state_path=Path(plan['standing_state']);state=json.loads(state_path.read_text())
    source=Path(plan['scene']['path'])
    if digest(source)!=plan['scene']['sha256']:raise ValueError('scene hash mismatch')
    # Reuse the actual runtime's candidate ancestry and existing solver seeds.
    original=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46/stl-standing-side-grasp-candidate-20261104T021318Z/result.json')
    candidate=json.loads(original.read_text());candidate['scene']=plan['scene']
    out=pg.create_unique_output_directory(ROOT,prefix='approved-circle-bounded-stance-region').native_path
    offsets=[('center',0.,0.,0.),('near',.02,0.,0.),('far',-.02,0.,0.),('left',0.,.02,0.),('right',0.,-.02,0.),('yaw_left',0.,0.,.03),('yaw_right',0.,0.,-.03)]
    rows=[]
    for height in (.30,.15):
        xml,base_state=bottle_height_variant(source,state,height)
        scene=out/f'bottle_{round(height*1000)}mm.xml';pg.write_new_bytes(scene,xml)
        local=copy.deepcopy(candidate);local['scene']={'path':str(scene),'sha256':digest(scene)}
        cp=out/f'candidate_{round(height*1000)}mm.json';pg.write_new_text(cp,json.dumps(local)+'\n')
        for label,dx,dy,dyaw in offsets:
            current=copy.deepcopy(base_state);base=current['base_pose_world'];base['xyz_m'][0]+=dx;base['xyz_m'][1]+=dy
            quat=np.array(base['quaternion_wxyz']);r=Rotation.from_rotvec([0,0,dyaw])*Rotation.from_quat(quat[[1,2,3,0]])
            base['quaternion_wxyz']=r.as_quat()[[3,0,1,2]].tolist()
            sp=out/f'state_{round(height*1000)}_{label}.json';pg.write_new_text(sp,json.dumps(current)+'\n')
            command=[sys.executable,str(Path(__file__).with_name('stl_approved_circle_ik.py')),'--circle-review',plan['circle_review'],'--candidate',str(cp),'--standing-state',str(sp),'--target-source','truth-diagnostic','--free-bottle-axis-roll','--physical-lift-test','--grasp-height-above-center-m',str(.10 if height==.30 else .05)]
            run=subprocess.run(command,text=True,capture_output=True,timeout=60,cwd=REPO)
            log=out/f'ik_{round(height*1000)}_{label}.log';pg.write_new_text(log,run.stdout+run.stderr)
            if run.returncode:row={'gate':'ERROR','returncode':run.returncode,'log':str(log)}
            else:
                result=json.loads(run.stdout.strip().splitlines()[-1]);detail=json.loads((Path(result['output'])/'result.json').read_text());best=detail['rows'][0]
                row={'gate':detail['gate'],'evidence':result['output'],'gates':best['gates'],'metrics':best['metrics'],'lift_sweep':detail.get('lift_preflight',{}).get('pass')}
            row.update({'bottle_height_m':height,'offset_xy_yaw':[dx,dy,dyaw],'pose_name':label,'base_pose_world':base,'dynamics':'NOT_RUN'})
            rows.append(row);print(json.dumps({k:row[k] for k in ('bottle_height_m','pose_name','gate')}),flush=True)
    record={'classification':'SIMULATION_ONLY_GEOMETRIC_STANCE_SAMPLES_NOT_CONTINUOUS_OR_DYNAMIC_QUALIFICATION','mj_step_calls':0,'source_capture':str(a.capture),'source_capture_sha256':digest(a.capture/'result.json'),'robot_geometry_changed':False,'bottle_variants':{'diameter_m':.07,'mass_kg':.2,'heights_m':[.30,.15],'bottom_on_table_preserved':True},'rows':rows,'limits':'Existing source limits, collision checks, opening and lift sweep unchanged. Interpolation, balance and physical grasp need independent dynamics.'}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    print(json.dumps({'output':str(out),'passing_samples':sum(r['gate']=='PASS' for r in rows),'sample_count':len(rows)}))

if __name__=='__main__':main()
