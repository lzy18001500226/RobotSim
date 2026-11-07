"""Source-preserving approved support-circle adapter for existing standing IK.

No model edits or dynamics. Frozen standing root/legs/waist are retained.
The resulting gate must pass before handing any target to live arm control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime import standing_side_grasp_static_search as ik


def circle_target(object_center, orientation, circle_center):
    return np.asarray(object_center) - np.asarray(orientation) @ np.asarray(circle_center)


def circle_task_jacobian(model,data,hand,center,local_axis,dofs):
    rotation=data.xmat[hand].reshape(3,3);point=data.xpos[hand]+rotation@center
    linear=np.zeros((3,model.nv));angular=np.zeros((3,model.nv))
    mujoco.mj_jac(model,data,linear,angular,point,hand)
    return np.vstack([linear[:,dofs],.1*np.cross(angular[:,dofs].T,rotation@local_axis).T])

def solve_bounded_upper_posture(model,state,start,center,section,axis,local_axis,opening,old_seed, *, full_source_diagnostic=False):
    """Coupled existing waist-yaw/pitch + arm IK, frozen root/legs/waist-roll.

    Pose bounds are tighter than source limits. The arm excursion and all
    final collision/lift gates are retained; this does not command the robot.
    """
    source=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ACTIVE])
    excursion=np.array([.25,.18]+[1.1]*7)
    lower=np.maximum(source[:,0]+.011,start-excursion);upper=np.minimum(source[:,1]-.011,start+excursion)
    if full_source_diagnostic:lower=source[:,0]+.011;upper=source[:,1]-.011
    hand=ik.native.bid(model,'R_omnipicker_base_link');data=mujoco.MjData(model)
    dofs=np.array([model.jnt_dofadr[ik.native.jid(model,n)] for n in ik.ACTIVE]);weight=np.array([1e-5,1e-5]+[1e-6]*7)
    def residual(q):
        ik.set_pose(model,data,state,q,np.zeros(3),0.,opening,np.zeros(3))
        error=axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,section,axis,local_axis)
        return np.r_[error,weight*(q-start)]
    def jacobian(q):
        residual(q)
        return np.vstack([circle_task_jacobian(model,data,hand,center,local_axis,dofs),np.diag(weight)])
    biased=old_seed.copy();biased[:2]=start[:2]+[.20,.12]
    seeds=[start,biased]
    if full_source_diagnostic:
        for fraction in (.15,.85):
            seed=biased.copy()
            for name in ('waist_yaw_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_wrist_roll_joint'):
                i=ik.ACTIVE.index(name);seed[i]=lower[i]+fraction*(upper[i]-lower[i])
            seeds.append(seed)
    rows=[]
    for seed in seeds:
        result=least_squares(residual,np.clip(seed,lower+1e-5,upper-1e-5),jac=jacobian,bounds=(lower,upper),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        error=residual(result.x)[:6];angle=float(2*np.arcsin(np.clip(np.linalg.norm(error[3:])/.2,0,1)))
        rows.append({'q':result.x.tolist(),'position_error_m':float(np.linalg.norm(error[:3])),'axis_error_rad':angle,'evaluations':result.nfev,'source_margin_rad':float(np.min(np.minimum(result.x-source[:,0],source[:,1]-result.x)))})
    def preference(row):
        passed=row['position_error_m']<=1e-4 and row['axis_error_rad']<=1e-3
        movement=float(np.linalg.norm((np.asarray(row['q'])-start)/(source[:,1]-source[:,0])))
        return (not passed,movement if passed else row['position_error_m']+.1*row['axis_error_rad'])
    rows.sort(key=preference)
    return np.array(rows[0]['q']),{'classification':'FULL_SOURCE_STATIC_DIAGNOSTIC_NOT_EXECUTABLE' if full_source_diagnostic else 'BOUNDED_SOURCE_UPPER_POSTURE_IK_ONLY','root_and_legs_frozen':True,'waist_roll_frozen':True,'waist_yaw_excursion_rad':None if full_source_diagnostic else .25,'waist_pitch_excursion_rad':None if full_source_diagnostic else .18,'arm_excursion_rad':None if full_source_diagnostic else 1.1,'minimum_planned_source_margin_rad':.011,'source_bounds_rad':source.tolist(),'rows':rows,'dynamics':'NOT_RUN'}


def calibrate_contact_frame(capture_path, circle_review_path):
    """Bounded zero-step contact-centre solve on unchanged compiled geometry.

    Match both mesh/cylinder surfaces and place gravity's line in the plane of
    the two contact points. This tests a grasp pose, not a mechanical change.
    The original human-approved circle remains separately recorded.
    """
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
    from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory
    capture=Path(capture_path);record=json.loads((capture/'result.json').read_text())
    overlay_path=Path(record['claw_control_overlay']['path'])
    if hashlib.sha256(overlay_path.read_bytes()).hexdigest()!=record['claw_control_overlay']['sha256']:raise ValueError('calibration overlay hash mismatch')
    scene=json.loads(overlay_path.read_text())['generated']['host_scene'];model=mujoco.MjModel.from_xml_path(scene['path'])
    trace=json.loads((capture/'native_hal_pregrasp_trace.json').read_text())['trace']
    close_start=min(x['host_monotonic_s'] for x in trace if x['stage']=='HAND_CLOSE_HOLD')
    rows=[json.loads(x) for x in (capture/'native_physics.jsonl').read_text().splitlines()]
    row=min(rows,key=lambda r:abs(r['host_monotonic_ns']*1e-9-(close_start-.05)))
    data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row);data.qvel[:]=0.
    hand=ik.native.bid(model,'R_omnipicker_base_link');bottle=ik.native.bid(model,'test_cylinder')
    hp=data.xpos[hand].copy();R=data.xmat[hand].reshape(3,3).copy();axis=data.xmat[bottle].reshape(3,3)[:,2].copy()
    review=json.loads(Path(circle_review_path).read_text());approved=review['circle_review']['samples'][40]
    bounds=[r['center_height_bounds_m'] for r in review['circle_review']['samples'][30:61]]
    # A fixed-radius object's centre need not lie between the patch endpoints.
    # That bound made the *variable-radius* review problem finite; it is not a
    # physical collision rule. Expanded candidates still need a surface audit.
    lo=np.array([.20,-.008,min(x[0] for x in bounds)-.02]);hi=np.array([.70,.008,max(x[1] for x in bounds)+.02])
    trajectory=load_registered_trajectory(REPO/'simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic')
    gids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('R_hand_narrow3_Link__collision_0','R_hand_wide3_Link__collision_0','test_cylinder_collision')]
    adr=int(model.jnt_qposadr[ik.native.jid(model,'test_cylinder_free')]);height=.1
    def residual(x):
        pose=trajectory.pose(float(x[0])*100.)
        for n,q in pose.positions_rad.items():
            n='R_claw_joint' if n=='right_claw_joint' else n
            j=ik.native.jid(model,n);data.qpos[model.jnt_qposadr[j]]=q
        data.qpos[adr:adr+3]=hp+R@np.array([0.,x[1],x[2]])-height*axis
        mujoco.mj_kinematics(model,data)
        witnesses=[np.zeros(6),np.zeros(6)]
        distances=[mujoco.mj_geomDistance(model,data,g,gids[2],.1,w) for g,w in zip(gids[:2],witnesses)]
        a,b=[w[3:] for w in witnesses];line=b-a;length=np.linalg.norm(line)
        imbalance=float(np.dot(line/length,np.cross(data.qpos[adr:adr+3]-a,[0.,0.,1.]))) if length>1e-9 else .1
        return np.array([distances[0]+.0001,distances[1]+.0001,imbalance])
    def jac(x):
        columns=[]
        for i,h in enumerate((1e-4,1e-5,1e-5)):
            a=x.copy();b=x.copy();a[i]+=h;b[i]-=h
            columns.append((residual(a)-residual(b))/(2*h))
        return np.array(columns).T
    candidates=[]
    for seed in ([.4,0.,approved['center_m'][2]],[.5,0.,.125]):
        result=least_squares(residual,np.clip(seed,lo+1e-6,hi-1e-6),jac=jac,bounds=(lo,hi),x_scale=[.1,.005,.01],max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        error=residual(result.x);mujoco.mj_forward(model,data)
        pads,bad,table,robot_table=ik.contacts(model,data,bottle)
        geometric=max(abs(error[:2]))<=.0002 and abs(error[2])<=1e-5 and not bad and not robot_table and set(pads)==ik.DISTALS
        zlo,zhi=review['circle_review']['samples'][round(float(result.x[0])*100)]['center_height_bounds_m']
        inside=zlo<=result.x[2]<=zhi
        candidates.append({'opening_percent':float(result.x[0]*100),'center_m':[0.,float(result.x[1]),float(result.x[2])],'residual_m':error.tolist(),'evaluations':result.nfev,'pad_bodies':pads,'forbidden_object_bodies':bad,'robot_table_bodies':robot_table,'geometry_and_wrench_pass':bool(geometric),'within_original_fitting_center_domain':bool(inside),'surface_audit':'NOT_RUN','pass':False})
    out=pg.create_unique_output_directory(Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46'),prefix='compiled-contact-frame-calibration').native_path
    result={'classification':'SIMULATION_ONLY_OFFLINE_CONTACT_FRAME_CALIBRATION_NOT_GRASP_PASS','mj_step_calls':0,'scene':scene,'capture':str(capture),'circle_review':str(circle_review_path),'approved_nominal_circle':{k:approved[k] for k in ('center_m','radius_m','diameter_mm')},'parameter_bounds':{'lower':lo.tolist(),'upper':hi.tolist()},'candidates':candidates,'model_geometry_modified':False}
    pg.write_new_text(out/'result.json',json.dumps(result,indent=2)+'\n');pg.write_artifact_manifest(out)
    print(json.dumps({'output':str(out),'candidates':candidates}),flush=True)


def axis_grasp_error(hand_position, hand_rotation, local_circle_center, section_center, bottle_axis, local_bottle_axis=None):
    """Five independent task constraints: centre XYZ and directed axis tilt.

    Roll around the bottle axis is free. Use directed vector subtraction,
    rather than a cross product that also vanishes for an inverted axis.
    """
    local=np.asarray([1.,0.,0.] if local_bottle_axis is None else local_bottle_axis)
    return np.r_[np.asarray(hand_position)+np.asarray(hand_rotation)@np.asarray(local_circle_center)-section_center,
                 .1*(np.asarray(hand_rotation)@local-np.asarray(bottle_axis))]



def tracking_phase_is_fresh(phase, now_ns):
    limits={"ARRIVAL_HOLD":12.,"CLOSING":6.,"LIFTING":9.,"RETRACTING":6.}
    if phase.get("phase")=="CARRY_HOLD":
        lease=phase.get("carry_lease_s")
        if not isinstance(lease,(int,float)) or not np.isfinite(lease) or not 5.<=lease<=200. or phase.get("post_retract_physical_hold_pass") is not True:return False
        limits["CARRY_HOLD"]=float(lease)
    stage=phase.get("phase");created=phase.get("created_host_ns")
    if stage not in limits or not isinstance(created,int) or not 0.<=(now_ns-created)*1e-9<=limits[stage]:return False
    if stage=="RETRACTING":
        end=phase.get("retract_from_hold_end_ns")
        if phase.get("pre_retract_physical_lift_pass") is not True or not isinstance(end,int) or not 0<=created-end<=200000000:return False
    return True


def tracking_task(row, reference, phase=None):
    """Phase-aware fixed grasp target, without promoting motion to success."""
    bottle=row['bodies']['test_cylinder'];B=np.array(bottle['rotation']).reshape(3,3)
    centre=np.array(bottle['position_m']);measured=centre+reference['grasp_height_above_center_m']*B[:,2]
    if phase is None:
        if np.linalg.norm(measured-np.array(reference['grasp_section_center_world_m']))>.01:raise RuntimeError('BOTTLE_MOVED_BEFORE_CLOSURE')
        return measured,B[:,2]
    if phase.get('phase')=='CARRY_HOLD':
        if phase.get('native_arrival_pass') is not True or phase.get('post_retract_physical_hold_pass') is not True or phase.get('generation')!=row['generation']:
            raise RuntimeError('UNQUALIFIED_BODY_RELATIVE_CARRY')
        offset=np.asarray(phase.get('carry_heading_xy_m'),dtype=float);height=float(phase.get('carry_world_z_m',float('nan')))
        if offset.shape!=(2,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.8 or not np.isfinite(height) or not .25<=height<=1.4:
            raise RuntimeError('INVALID_CARRY_TASK_FRAME')
        hand=row['bodies']['R_omnipicker_base_link'];H=np.asarray(hand['rotation']).reshape(3,3)
        baseline=np.asarray(phase['initial_object_in_hand_m'],dtype=float)
        if phase.get('slip_reference')!='GRASP_SECTION' or baseline.shape!=(3,) or not np.isfinite(baseline).all():raise RuntimeError('INVALID_CARRY_SLIP_REFERENCE')
        if np.linalg.norm(H.T@(measured-np.asarray(hand['position_m']))-baseline)>.01:raise RuntimeError('BOTTLE_SLIPPED_DURING_CARRY')
        if B[2,2]<np.cos(.10):raise RuntimeError('BOTTLE_TILTED_DURING_CARRY')
        base=row['bodies']['pelvis'];R=np.asarray(base['rotation']).reshape(3,3);yaw=np.arctan2(R[1,0],R[0,0])
        c,s=np.cos(yaw),np.sin(yaw)
        xy=np.asarray(base['position_m'][:2])+np.array([[c,-s],[s,c]])@offset
        # Follow translation/yaw, reject pelvis roll/pitch sway. Object truth
        # only checks slip/tilt; it never defines the commanded carry target.
        return np.r_[xy,height],np.array([0.,0.,1.])
    if phase.get('phase') not in ('ARRIVAL_HOLD','CLOSING','LIFTING','RETRACTING') or phase.get('native_arrival_pass') is not True:raise RuntimeError('INVALID_CLOSURE_PHASE_HANDOFF')
    if phase.get('generation')!=row['generation']:raise RuntimeError('CLOSURE_PHASE_GENERATION_MISMATCH')
    anchor=np.asarray(phase['target_world_m'],dtype=float);axis=np.asarray(phase['bottle_axis_world'],dtype=float);initial=np.asarray(phase['bottle_center_world_m'],dtype=float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (anchor,axis,initial)) or abs(np.linalg.norm(axis)-1.)>.001:raise RuntimeError('INVALID_CLOSURE_ANCHOR')
    offset=np.zeros(3)
    if phase['phase'] in ('LIFTING','RETRACTING'):
        lift=float(phase.get('lift_offset_m',float('nan')))
        if phase.get('bilateral_contact_pass') is not True or not np.isfinite(lift) or not 0.<=lift<=.060000001:raise RuntimeError('UNQUALIFIED_OR_UNBOUNDED_LIFT_PHASE')
        offset[2]=lift
        if phase['phase']=='RETRACTING':
            if phase.get('pre_retract_physical_lift_pass') is not True:raise RuntimeError('RETRACT_WITHOUT_PHYSICAL_LIFT_GATE')
            retreat=np.asarray(phase.get('retract_offset_world_m'),dtype=float)
            if retreat.shape!=(3,) or not np.isfinite(retreat).all() or np.linalg.norm(retreat)>.250000001 or not -.02<=retreat[2]<=.05:
                raise RuntimeError('UNQUALIFIED_RETRACT_PHASE')
            offset+=retreat
        hand=row['bodies']['R_omnipicker_base_link'];H=np.asarray(hand['rotation']).reshape(3,3)
        if phase.get('slip_reference')!='GRASP_SECTION':raise RuntimeError('INVALID_LIFT_SLIP_REFERENCE')
        section=centre+reference['grasp_height_above_center_m']*B[:,2]
        local=H.T@(section-np.asarray(hand['position_m']))
        baseline=np.asarray(phase['initial_object_in_hand_m'],dtype=float)
        if baseline.shape!=(3,) or not np.isfinite(baseline).all():raise RuntimeError('INVALID_LIFT_SLIP_BASELINE')
        # Real slip is measured relative to the actual hand, not its command.
        # Servo tracking lag must remain distinguishable from object slippage.
        if np.linalg.norm(local-baseline)>.01:raise RuntimeError('BOTTLE_NOT_FOLLOWING_BOUNDED_LIFT')
    elif np.linalg.norm(centre-initial)>.01:raise RuntimeError('BOTTLE_MOVED_DURING_CLOSURE')
    if np.dot(B[:,2],axis)<np.cos(.10):raise RuntimeError('BOTTLE_TILTED_DURING_CLOSURE')
    # Hold the attained world target; never chase an object pushed by one pad.
    return anchor+offset,axis


def screen_lift(model, state, seed, opening, center, section, local_bottle_axis=None):
    """Thirteen zero-step, source-limited lift samples; no runtime pose writes."""
    import copy
    bounds=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ARM])
    hand=ik.native.bid(model,'R_omnipicker_base_link');bottle=ik.native.bid(model,'test_cylinder')
    free=ik.native.jid(model,'test_cylinder_free')
    attachments=[i for i in range(model.neq) if model.eq_type[i] in (mujoco.mjtEq.mjEQ_CONNECT,mujoco.mjtEq.mjEQ_WELD) and bottle in (model.eq_obj1id[i],model.eq_obj2id[i])]
    driven=[i for i in range(model.nu) if model.actuator_trntype[i] in (mujoco.mjtTrn.mjTRN_JOINT,mujoco.mjtTrn.mjTRN_JOINTINPARENT) and model.actuator_trnid[i,0]==free]
    if model.jnt_type[free]!=mujoco.mjtJoint.mjJNT_FREE or model.body_gravcomp[bottle]!=0 or np.linalg.norm(model.opt.gravity)==0 or attachments or driven:raise ValueError('bottle is not an unassisted free body under gravity')
    arm=np.array(seed[2:]);samples=[]
    for dz in np.linspace(0.,.06,13):
        frozen=copy.deepcopy(state);frozen['object_pose_world']['xyz_m'][2]+=float(dz)
        data=mujoco.MjData(model);target=np.asarray(section)+[0.,0.,dz]
        ik.set_pose(model,data,frozen,np.r_[seed[:2],arm],np.zeros(3),0.,opening,np.zeros(3))
        qa=np.array([model.jnt_qposadr[ik.native.jid(model,n)] for n in ik.ARM])
        def residual(q):
            data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,target,data.xmat[bottle].reshape(3,3)[:,2],local_bottle_axis)
        dofs=np.array([model.jnt_dofadr[ik.native.jid(model,n)] for n in ik.ARM])
        def jacobian(q):
            residual(q)
            return circle_task_jacobian(model,data,hand,np.asarray(center),np.asarray([1.,0.,0.] if local_bottle_axis is None else local_bottle_axis),dofs)
        solved=least_squares(residual,np.clip(arm,bounds[:,0]+.001,bounds[:,1]-.001),jac=jacobian,bounds=(bounds[:,0]+.001,bounds[:,1]-.001),max_nfev=70,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        arm=solved.x;e=residual(arm);mujoco.mj_forward(model,data);pads,bad,table,robot_table=ik.contacts(model,data,bottle)
        passed=np.linalg.norm(e[:3])<=.001 and np.linalg.norm(e[3:])<=.001 and not bad and not robot_table
        samples.append({'rise_m':float(dz),'position_error_m':float(np.linalg.norm(e[:3])),'axis_residual':float(np.linalg.norm(e[3:])),'forbidden_object_bodies':bad,'robot_table_bodies':robot_table,'right_arm_rad':arm.tolist(),'pass':bool(passed)})
    mu=min(float(model.geom_friction[g,0]) for g in range(model.ngeom) if int(model.geom_bodyid[g]) in (bottle,ik.native.bid(model,'R_hand_narrow3_Link'),ik.native.bid(model,'R_hand_wide3_Link')))
    mass=float(model.body_mass[bottle]);weight=mass*float(np.linalg.norm(model.opt.gravity))
    if not mu>0 or not mass>0:raise ValueError('invalid known payload/friction for diagnostic force budget')
    return {'pass':all(s['pass'] for s in samples),'samples':samples,'mj_step_calls':0,'height_m':.06,'duration_s':3.,'hold_s':1.5,'contact_command':opening/100.,'mass_kg':mass,'sliding_friction':mu,'required_normal_force_n':1.5*weight/(2*mu),'force_budget_is_simulation_assumption':True,'free_body_preflight':{'pass':True,'joint':'test_cylinder_free','gravity_m_s2':model.opt.gravity.tolist(),'gravcomp':0.,'object_actuator_count':len(driven),'attachment_count':len(attachments)}}



def screen_retract(model, state, seed, opening, center, section, offset_world, local_bottle_axis=None):
    """Reuse the source arm IK for a <=250mm post-lift retreat, zero physics steps."""
    import copy
    offset=np.asarray(offset_world,dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.250000001 or not -.02<=offset[2]<=.05:
        raise ValueError("post-lift retract outside bounded translation")
    bounds=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ARM])
    hand=ik.native.bid(model,"R_omnipicker_base_link");bottle=ik.native.bid(model,"test_cylinder")
    axis=Rotation.from_quat(np.asarray(state["object_pose_world"]["quaternion_wxyz"])[[1,2,3,0]]).as_matrix()[:,2]
    arm=np.array(seed[2:]);samples=[]
    dofs=np.array([model.jnt_dofadr[ik.native.jid(model,n)] for n in ik.ARM])
    local=np.asarray([1.,0.,0.] if local_bottle_axis is None else local_bottle_axis)
    for fraction in np.linspace(0.,1.,13):
        delta=float(fraction)*offset
        frozen=copy.deepcopy(state);frozen["object_pose_world"]["xyz_m"]=(np.asarray(state["object_pose_world"]["xyz_m"])+delta).tolist()
        data=mujoco.MjData(model);target=np.asarray(section)+delta
        ik.set_pose(model,data,frozen,np.r_[seed[:2],arm],np.zeros(3),0.,opening,np.zeros(3))
        qa=np.array([model.jnt_qposadr[ik.native.jid(model,n)] for n in ik.ARM])
        def residual(q):
            data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,target,axis,local)
        def jacobian(q):
            residual(q)
            return circle_task_jacobian(model,data,hand,np.asarray(center),local,dofs)
        solved=least_squares(residual,np.clip(arm,bounds[:,0]+.001,bounds[:,1]-.001),jac=jacobian,bounds=(bounds[:,0]+.001,bounds[:,1]-.001),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        arm=solved.x;e=residual(arm);mujoco.mj_forward(model,data);pads,bad,table,robot_table=ik.contacts(model,data,bottle)
        passed=np.linalg.norm(e[:3])<=.001 and np.linalg.norm(e[3:])<=.001 and not bad and not robot_table
        samples.append(dict(fraction=float(fraction),position_error_m=float(np.linalg.norm(e[:3])),axis_residual=float(np.linalg.norm(e[3:])),forbidden_object_bodies=bad,robot_table_bodies=robot_table,right_arm_rad=arm.tolist(),solver_evaluations=solved.nfev,solver_status=solved.status,solver_optimality=float(solved.optimality),pass_result=bool(passed)))
    root=ik.native.bid(model,"pelvis")
    robot_bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in range(model.nbody) if model.body_rootid[b]==root]
    return dict(robot_bodies=robot_bodies,pass_result=all(x["pass_result"] for x in samples),classification="SOURCE_LIMITED_STATIC_RETRACT_NOT_PHYSICS_PASS",mj_step_calls=0,offset_world_m=offset.tolist(),duration_s=4.,hold_s=1.5,samples=samples)


def solve_tracking_endpoint(model, row, reference, seed, phase=None):
    """Re-solve existing circle task using one actual native snapshot (offline MjData)."""
    data=mujoco.MjData(model)
    base=row['bodies']['pelvis'];root=ik.native.jid(model,'floating_base_joint');adr=int(model.jnt_qposadr[root])
    data.qpos[adr:adr+3]=base['position_m'];data.qpos[adr+3:adr+7]=base['quaternion_wxyz']
    for name,item in row['joints'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if j>=0:data.qpos[model.jnt_qposadr[j]]=float(item['q'])
    b=row['bodies']['test_cylinder'];bottle=ik.native.bid(model,'test_cylinder');adr=int(model.jnt_qposadr[ik.native.jid(model,'test_cylinder_free')]);data.qpos[adr:adr+3]=b['position_m'];data.qpos[adr+3:adr+7]=b['quaternion_wxyz']
    centre=np.array(reference.get('execution_grasp_frame',{}).get('center_m',reference['circle']['center_m']));target,bottle_axis=tracking_task(row,reference,phase);local_axis=np.asarray(reference.get('local_bottle_axis_in_hand',[1.,0.,0.]))
    ids=[ik.native.jid(model,n) for n in ik.ARM];qa=np.array([model.jnt_qposadr[j] for j in ids]);bounds=np.array([model.jnt_range[j] for j in ids]);hand=ik.native.bid(model,'R_omnipicker_base_link')
    def residual(q):
        data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),centre,target,bottle_axis,local_axis)
    dofs=np.array([model.jnt_dofadr[j] for j in ids])
    def jacobian(q):
        data.qpos[qa]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        R=data.xmat[hand].reshape(3,3);point=data.xpos[hand]+R@centre
        linear=np.zeros((3,model.nv));angular=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,linear,angular,point,hand)
        return np.vstack([linear[:,dofs],.1*np.cross(angular[:,dofs].T,R@local_axis).T])
    q=np.clip(seed,bounds[:,0]+.001,bounds[:,1]-.001)
    for iteration in range(20):
        error=residual(q)
        if np.linalg.norm(error[:3])<1e-4 and np.linalg.norm(error[3:])<1e-4:break
        J=jacobian(q)
        dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),-error)
        q=np.clip(q+np.clip(dq,-.03,.03),bounds[:,0]+.001,bounds[:,1]-.001)
    error=residual(q)
    if np.linalg.norm(error[:3])>.001 or np.linalg.norm(error[3:])>.001:
        # Bound-aware recovery can redistribute axial-roll redundancy instead
        # of repeatedly clipping the same wrist joint at its source limit.
        opt=least_squares(residual,q,jac=jacobian,bounds=(bounds[:,0]+.001,bounds[:,1]-.001),max_nfev=70,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q=opt.x;error=residual(q)
    if np.linalg.norm(error[:3])>.001 or np.linalg.norm(error[3:])>.001:raise RuntimeError('LIVE_CIRCLE_IK_NOT_CONVERGED:'+json.dumps({'position_m':float(np.linalg.norm(error[:3])),'axis_residual':float(np.linalg.norm(error[3:])),'iterations':iteration+1,'q':q.tolist()}))
    actual=np.array([row['joints'][n]['q'] for n in ik.ARM])
    actual_error=residual(actual);J=jacobian(actual)
    correction=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),-actual_error)
    return q,{'position_error_m':float(np.linalg.norm(error[:3])),'axis_residual':float(np.linalg.norm(error[3:])),'source_host_ns':row['host_monotonic_ns'],'actual_position_error_m':float(np.linalg.norm(actual_error[:3])),'actual_axis_residual':float(np.linalg.norm(actual_error[3:])),'cartesian_correction_rad':dict(zip(ik.ARM,correction.tolist()))}


def bounded_tracking_velocity(previous, target, dt, speed_limits):
    if not np.isfinite(dt) or not .001<=dt<=.20:raise RuntimeError('CARRY_TARGET_VELOCITY_CLOCK')
    velocity=(np.asarray(target)-np.asarray(previous))/dt
    bounds=np.asarray(speed_limits)
    if velocity.shape!=bounds.shape or not np.isfinite(velocity).all() or not np.isfinite(bounds).all() or np.any(bounds<=0):raise RuntimeError('CARRY_TARGET_VELOCITY_INVALID')
    return np.clip(velocity,-bounds,bounds)


def track_native(reference_path, latest_path, output_path):
    import os,time
    print(json.dumps({'tracker_initialization':'MODEL_LOADING','host_monotonic_ns':time.monotonic_ns()}),flush=True)
    reference=json.loads(Path(reference_path).read_text())
    if reference.get('host_model_cache'):
        from scripts.research.issue46.perception.native_rgbd_grasp import load_cached_model
        cache=reference['host_model_cache']
        if hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()!=cache['sha256']:raise ValueError('tracking cache reference changed')
        model=load_cached_model(reference['scene']['path'],cache['path'])
    else:model=mujoco.MjModel.from_xml_path(reference['scene']['path'])
    print(json.dumps({'tracker_initialization':'MODEL_READY','host_monotonic_ns':time.monotonic_ns()}),flush=True)
    seed=np.array([reference['initial_state']['arm_positions_rad'][n] for n in ik.ARM]);deadline=time.monotonic()+45.;generation=None;trace=[];previous_carry=None;carry_velocity=np.zeros(7)
    output_path=Path(output_path);stop=output_path.with_suffix('.stop')
    # Loading the offline model can temporarily contend with the native runtime.
    # Before producing any executable target, await a genuinely fresh sample.
    # Once tracking starts the unchanged 0.20s deadline remains fail-closed.
    readiness_deadline=time.monotonic()+3.
    while True:
        initial=json.loads(Path(latest_path).read_text())
        initial_age=(time.monotonic_ns()-initial['host_monotonic_ns'])*1e-9
        if 0.<=initial_age<=.10:break
        if time.monotonic()>=readiness_deadline:raise RuntimeError('NATIVE_TRACKING_INITIAL_FRESHNESS_NOT_READY')
        time.sleep(.02)
    try:
        while time.monotonic()<deadline and not stop.exists():
            tick=time.monotonic();row=json.loads(Path(latest_path).read_text());age=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9
            if not 0<=age<=.20:raise RuntimeError('STALE_NATIVE_TRACKING_SOURCE')
            if generation is not None and row['generation']!=generation:raise RuntimeError('NATIVE_TRACKING_GENERATION_CHANGED')
            phase_path=output_path.with_name('native_tracking_phase.json')
            phase=json.loads(phase_path.read_text()) if phase_path.exists() else None
            if reference.get('target_source')=='fresh-rgbd-snapshot' and phase is None:
                from scripts.research.issue46.perception.native_rgbd_grasp import qualify_stationary_snapshot
                snapshot_ref=reference['perception']['snapshot'];snapshot_path=Path(snapshot_ref['path'])
                if hashlib.sha256(snapshot_path.read_bytes()).hexdigest()!=snapshot_ref['sha256']:raise RuntimeError('RGBD_SNAPSHOT_REFERENCE_CHANGED')
                row['bodies']['test_cylinder']=qualify_stationary_snapshot(json.loads(snapshot_path.read_text()),generation=row['generation'])
            if reference.get('target_source')=='fresh-rgbd' and phase is None:
                from scripts.research.issue46.perception.native_rgbd_grasp import qualify_estimate
                if not trace:
                    from scripts.research.issue46.perception.native_rgbd_grasp import await_initial_estimate
                    estimate,_=await_initial_estimate(reference['perception']['latest_path'],generation=row['generation'])
                    row=json.loads(Path(latest_path).read_text())
                    if not 0.<=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('STALE_NATIVE_AFTER_INITIAL_IMAGE_READINESS')
                else:estimate=json.loads(Path(reference['perception']['latest_path']).read_text())
                row['bodies']['test_cylinder']=qualify_estimate(estimate,generation=row['generation'])
            if phase is not None and not tracking_phase_is_fresh(phase,time.monotonic_ns()):raise RuntimeError('CLOSURE_PHASE_EXPIRED')
            generation=row['generation'];seed,metrics=solve_tracking_endpoint(model,row,reference,seed,phase)
            if phase is not None and phase.get('phase')=='CARRY_HOLD':
                if previous_carry is not None and row['host_monotonic_ns']>previous_carry[0]:
                    carry_velocity=bounded_tracking_velocity(previous_carry[1],seed,(row['host_monotonic_ns']-previous_carry[0])*1e-9,phase['carry_source_speed_rad_s'])
                previous_carry=(row['host_monotonic_ns'],seed.copy())
            else:previous_carry=None;carry_velocity=np.zeros(7)
            if phase is not None and phase.get('phase')=='CARRY_HOLD':
                if not output_path.with_name('native_tracking_carry_handoff.json').exists():
                    handoff={'classification':'BODY_RELATIVE_WORLD_UPRIGHT_IK_CARRY','generation':row['generation'],'source_host_ns':row['host_monotonic_ns'],'physical_hold_pass':True}
                    output_path.with_name('native_tracking_carry_handoff.json').write_text(json.dumps(handoff))
                deadline=phase['created_host_ns']*1e-9+float(phase['carry_lease_s'])
            now=time.monotonic_ns()
            if (now-row['host_monotonic_ns'])*1e-9>.20:raise RuntimeError('NATIVE_TRACKING_COMPUTE_DEADLINE')
            record={'pass':True,'created_host_ns':now,'source_host_ns':row['host_monotonic_ns'],'generation':generation,'right_target_rad':dict(zip(ik.ARM,seed.tolist())),'phase':phase.get('phase') if phase else None,'right_velocity_rad_s':dict(zip(ik.ARM,carry_velocity.tolist())),'metrics':metrics}
            temp=output_path.with_suffix('.tmp');temp.write_text(json.dumps(record));os.replace(temp,output_path);trace.append(record)
            period=.02 if phase is not None and phase.get('phase')=='CARRY_HOLD' else .1
            time.sleep(max(0,period-(time.monotonic()-tick)))
    except Exception as exc:
        output_path.with_suffix('.failure-input.json').write_text(json.dumps(row))
        record={'pass':False,'failure':str(exc),'created_host_ns':time.monotonic_ns()};temp=output_path.with_suffix('.tmp');temp.write_text(json.dumps(record));os.replace(temp,output_path)
        raise
    finally:
        output_path.with_suffix('.trace.json').write_text(json.dumps(trace))


def qualify_grasp_motion_branches(rows,model,state,opening,center,object_center,local_axis,retract_offset=None):
    import copy
    for row in rows:
        row['grasp_only_pass']=row['pass'];row['pass']=False
        row['motion_qualification']='NOT_EVALUATED'
    for candidate_row in rows:
        if not candidate_row['grasp_only_pass']:continue
        lift_seed=np.array([candidate_row['q_rad'][n] for n in ik.ACTIVE])
        task_lift=screen_lift(model,state,lift_seed,opening,center,object_center,local_axis)
        candidate_row['lift_preflight']=task_lift
        candidate_row['gates']['LIFT_SWEEP']='PASS' if task_lift['pass'] else 'FAIL'
        candidate_row['pass']=bool(task_lift['pass'])
        if retract_offset is not None and task_lift['pass']:
            lifted=copy.deepcopy(state)
            lifted['object_pose_world']['xyz_m']=(np.asarray(state['object_pose_world']['xyz_m'])+[0.,0.,.06]).tolist()
            retract_seed=np.r_[lift_seed[:2],task_lift['samples'][-1]['right_arm_rad']]
            retreat=screen_retract(model,lifted,retract_seed,opening,center,object_center+[0.,0.,.06],retract_offset,local_axis)
            task_lift['retract_preflight']=retreat
            candidate_row['gates']['POST_LIFT_RETRACT_SWEEP']='PASS' if retreat['pass_result'] else 'FAIL'
            candidate_row['pass']=bool(retreat['pass_result'])

        candidate_row['motion_qualification']='PASS' if candidate_row['pass'] else 'FAIL'
        if candidate_row['pass']:break



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circle-review", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--standing-state", type=Path, required=True)
    parser.add_argument("--host-model-cache",type=Path)
    parser.add_argument("--grasp-height-above-center-m",type=float,default=0.0)
    parser.add_argument("--grasp-axis-tilt-rad",type=float,default=0.,help="measured functional-axis correction; 0.03rad orientation residual remains relative to the planned grasp frame")
    parser.add_argument("--inverted-bottle-axis",action="store_true",help="explicit opposite axial branch of the same cylindrical grasp, unchanged source geometry and limits")
    parser.add_argument("--grasp-axis-calibration",type=Path,help="measured load-axis hypothesis required for a nontrivial task-frame correction")
    parser.add_argument("--use-measured-load-frame",action="store_true",help="apply measured <=5mm TCP correction without altering the approved nominal circle")
    parser.add_argument("--runtime-handoff",action="store_true")
    parser.add_argument("--physical-lift-test",action="store_true")
    parser.add_argument("--post-lift-retract-world-m",type=float,nargs=3)
    parser.add_argument("--approach-opening-percent",type=int,default=100)
    parser.add_argument("--free-bottle-axis-roll",action="store_true",help="constrain reviewed bottle axis and circle centre, letting IK choose axial roll")
    parser.add_argument("--target-source",choices=("historical-rgbd-replay","truth-diagnostic","fresh-rgbd","fresh-rgbd-snapshot"),default="historical-rgbd-replay",help="explicit target provenance and continuous/snapshot temporal contract")
    parser.add_argument("--base-offset-xyz-m",type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument("--waist-pitch-offset-rad",type=float,default=0.,help="offline source-limited forward-bend candidate; nonzero requires a separately supported waist control handoff")
    parser.add_argument("--waist-yaw-offset-rad",type=float,default=0.,help="bounded <=0.25rad torso-side posture; source limits and collision sweep remain mandatory")
    parser.add_argument("--optimize-waist-posture",action="store_true",help="coupled source-bounded waist yaw/pitch and arm solve; requires coordinated physical handoff if executed")
    parser.add_argument("--full-source-posture-diagnostic",action="store_true",help="offline only: distinguish the solver trust region from actual source joint limits; no runtime handoff")
    args = parser.parse_args()
    if args.full_source_posture_diagnostic and (not args.optimize_waist_posture or args.runtime_handoff):raise ValueError('full-source diagnostic is offline only and requires the coupled posture solver')
    scene = json.loads(args.circle_review.read_text())
    candidate = json.loads(args.candidate.read_text())
    state = json.loads(args.standing_state.read_text())
    base_offset=np.asarray(args.base_offset_xyz_m,dtype=float)
    if base_offset.shape!=(3,) or not np.isfinite(base_offset).all(): raise ValueError("base offset must be finite")
    state=json.loads(json.dumps(state))
    state["base_pose_world"]["xyz_m"]=(np.asarray(state["base_pose_world"]["xyz_m"])+base_offset).tolist()
    samples = scene["circle_review"]["samples"]
    if scene["circle_review"]["classification"] != "STATIC_VARIABLE_DIAMETER_SUPPORT_CIRCLE_REVIEW_ONLY":
        raise ValueError("expected approved variable-diameter support-circle record")
    source = Path(candidate["scene"]["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != candidate["scene"]["sha256"]:
        raise ValueError("source scene hash mismatch")
    if args.host_model_cache:
        from scripts.research.issue46.perception.native_rgbd_grasp import load_cached_model
        model=load_cached_model(source,args.host_model_cache)
    else:model = mujoco.MjModel.from_xml_path(str(source))
    q = np.array([state["waist_positions_rad"][n] for n in ("waist_yaw_joint","waist_pitch_joint")]+[state["arm_positions_rad"][n] for n in ik.ARM])
    approach_start=q.copy()
    if not np.isfinite(args.waist_pitch_offset_rad):raise ValueError("waist offset must be finite")
    if not np.isfinite(args.waist_yaw_offset_rad) or abs(args.waist_yaw_offset_rad)>.25:raise ValueError('waist yaw outside bounded posture diagnostic')
    q[0]+=args.waist_yaw_offset_rad
    q[1]+=args.waist_pitch_offset_rad
    for index,name in enumerate(('waist_yaw_joint','waist_pitch_joint')):
        limits=model.jnt_range[ik.native.jid(model,name)]
        if not limits[0]+1e-3<=q[index]<=limits[1]-1e-3:raise ValueError('planned waist pose violates unchanged source joint limits:'+name)
    waist_changed=bool(args.waist_pitch_offset_rad or args.waist_yaw_offset_rad or args.optimize_waist_posture)
    perception = candidate["perception"]
    object_center = np.array(perception["estimated_center_world_m"])
    if args.target_source in ("truth-diagnostic","fresh-rgbd","fresh-rgbd-snapshot"):
        if args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot'):
            from scripts.research.issue46.perception.native_rgbd_grasp import await_initial_estimate
            latest=Path(state['perception']['latest_path']);estimated,pose=await_initial_estimate(latest,generation=state['native_snapshot']['generation'])
            state['object_pose_world']={'xyz_m':pose['position_m'],'quaternion_wxyz':pose['quaternion_wxyz']}
        source_data=mujoco.MjData(model)
        if 'object_pose_world' in state:
            measured=state['object_pose_world'];j=ik.native.jid(model,'test_cylinder_free');adr=int(model.jnt_qposadr[j])
            source_data.qpos[adr:adr+3]=measured['xyz_m'];source_data.qpos[adr+3:adr+7]=measured['quaternion_wxyz']
        elif args.runtime_handoff:
            raise ValueError('runtime truth requires the same native object snapshot as robot state')
        mujoco.mj_forward(model,source_data)
        object_center=source_data.xpos[ik.native.bid(model,"test_cylinder")].copy()
        bottle_axis=source_data.xmat[ik.native.bid(model,'test_cylinder')].reshape(3,3)[:,2].copy()
        perception=({**state['perception'],'observation_at_plan':estimated,'object_truth_used':False,'temporal_contract':'SNAPSHOT_REVERIFY_BEFORE_EXECUTION' if args.target_source=='fresh-rgbd-snapshot' else 'CONTINUOUS_UNTIL_VERIFIED_ARRIVAL'} if args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot') else {"source":"NATIVE_EPISODE_TRUTH_DIAGNOSTIC_ONLY" if 'object_pose_world' in state else 'XML_INITIAL_TRUTH_DIAGNOSTIC_ONLY',"object_center_world_m":object_center.tolist(),"scene_sha256":candidate["scene"]["sha256"],"native_snapshot":state.get('native_snapshot')})
    else:
        perception={**perception,"source":"HISTORICAL_RGBD_REPLAY_NOT_CURRENT_EPISODE","fresh_capture":False}
    # Grasp height is a task pose, not a physical/joint limit. Retain >=10mm
    # centre-to-top margin; any pad overhang still needs the actual full 3-D
    # contact, collision and lift screens below. No bottle geometry is edited.
    cylinder_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'test_cylinder_collision')
    half_height=float(model.geom_size[cylinder_geom,1])
    if not 0<=args.grasp_height_above_center_m<=half_height-.01:
        raise ValueError("grasp section height must be within the canonical bottle body")
    theta=args.grasp_axis_tilt_rad;axis_calibration=None
    if args.inverted_bottle_axis and (theta!=0. or args.grasp_axis_calibration or args.use_measured_load_frame):raise ValueError('opposite axial branch must remain separate from measured tilt/TCP hypotheses')
    if not np.isfinite(theta) or abs(theta)>.18:raise ValueError('task-frame tilt outside bounded measured calibration envelope')
    if abs(theta)>.028:
        if args.grasp_axis_calibration is None:raise ValueError('nontrivial grasp-frame correction requires independent measured load evidence')
        axis_calibration=json.loads(args.grasp_axis_calibration.read_text())
        if axis_calibration.get('classification')!='SIMULATION_ONLY_MEASURED_LOAD_AXIS_HYPOTHESIS_NOT_GRASP_PASS' or abs(theta-axis_calibration['tilt_rad'])>.001:raise ValueError('task-frame correction does not match measured load hypothesis')
        observations=axis_calibration.get('observations',[])
        if len(observations)<2:raise ValueError('independent load-axis source evidence missing')
        for observation in observations:
            source=Path(observation['capture'])/'native_tracking_target.failure-input.json'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=observation['source_sha256']:raise ValueError('load-axis source evidence hash mismatch')
    local_axis=np.array([-1.,0.,0.] if args.inverted_bottle_axis else [np.cos(theta),0.,np.sin(theta)])
    if axis_calibration is not None:local_axis=np.asarray(axis_calibration.get('local_bottle_axis_in_hand',local_axis),dtype=float)
    if args.target_source=='historical-rgbd-replay':bottle_axis=np.array([0.,0.,1.])
    object_center+=args.grasp_height_above_center_m*bottle_axis
    # The canonical existing fixture is 70mm diameter. Match its circumference
    # with the nearest approved opening, then inspect adjacent closure commands.
    requested_diameter_mm=float(estimated['diameter_m']*1000) if args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot') else 70.
    match = min(samples,key=lambda s:abs(s["diameter_mm"]-requested_diameter_mm))
    command = match["opening_percent"]
    open_command=args.approach_opening_percent
    if not command+5<=open_command<=100:raise ValueError("approach opening must exceed bottle-match opening by at least 5 percent")
    if match["status"] != "FOUR_BANDS_NEAR_0_5MM":
        raise ValueError("canonical diameter has no qualified support distribution")
    center = np.array(match["center_m"]);nominal_center=center.copy()
    if args.use_measured_load_frame:
        if axis_calibration is None:raise ValueError('measured TCP requires the verified load-axis hypothesis')
        center=np.asarray(axis_calibration['section_center_hand_m'])
        if center.shape!=(3,) or not np.isfinite(center).all() or np.linalg.norm(center-nominal_center)>.005:raise ValueError('TCP correction exceeds original arrival tolerance')
    # Bottle frame +Z is its upright axis. Set gripper +X exactly to it,
    # preserving the old reference roll as closely as possible.
    x = bottle_axis.copy()
    y = ik.REFERENCE_HAND_ROTATION[:,1].copy(); y-=x*np.dot(x,y); y/=np.linalg.norm(y)
    rotation = np.column_stack([x,y,np.cross(x,y)])
    rotation=rotation@Rotation.align_vectors([[1.,0.,0.]],[local_axis])[0].as_matrix()
    ik.REFERENCE_HAND_ROTATION = rotation
    ik.REFERENCE_HAND_POSITION = -rotation @ center
    rows=[]
    # Reuse the previously converged arm seed, not only the neutral standing
    # arm. Test three rolls about the bottle axis without moving the base.
    old_q=np.array([candidate["search"]["candidate"]["upper_q_rad"][n] for n in ik.ACTIVE])
    old_q[:2]=q[:2]
    upper_posture=None
    if args.optimize_waist_posture:
        if args.waist_yaw_offset_rad or args.waist_pitch_offset_rad or not args.free_bottle_axis_roll:raise ValueError('coupled posture solve requires free axial roll and no separate waist offsets')
        q,upper_posture=solve_bounded_upper_posture(model,state,approach_start,center,object_center,bottle_axis,local_axis,command,old_q,full_source_diagnostic=args.full_source_posture_diagnostic);old_q=q.copy()
    azimuths=(float(candidate["search"]["candidate"]["grasp_azimuth_rad"]),) if args.optimize_waist_posture else (float(candidate["search"]["candidate"]["grasp_azimuth_rad"]),float(candidate["search"]["candidate"]["grasp_azimuth_rad"])-0.8,float(candidate["search"]["candidate"]["grasp_azimuth_rad"])+0.8)
    for azimuth in azimuths:
        attempts=[ik.solve(model,state,np.zeros(3),0.,command,seed.copy(),np.zeros(3),azimuth,np.zeros(3),0.,0.,None,object_center) for seed in (q,old_q)]
        solved,metrics=min(attempts,key=lambda item:item[1]["hand_position_error_m"]+item[1]["hand_orientation_error_rad"]*.1)
        if upper_posture is not None:
            # A full-rotation seed solve must not discard a better directed-axis
            # solution of the actual five-constraint task.
            check=mujoco.MjData(model);ik.set_pose(model,check,state,q,np.zeros(3),0.,command,np.zeros(3))
            bid=ik.native.bid(model,'R_omnipicker_base_link');error=axis_grasp_error(check.xpos[bid],check.xmat[bid].reshape(3,3),center,object_center,bottle_axis,local_axis)
            angle=float(2*np.arcsin(np.clip(np.linalg.norm(error[3:])/.2,0,1)))
            if np.linalg.norm(error[:3])+.1*angle < metrics['hand_position_error_m']+.1*metrics['hand_orientation_error_rad']:
                solved=q.copy();bounds_source=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ARM])
                metrics={**metrics,'solver':'coupled_source_upper_directed_axis','hand_position_error_m':float(np.linalg.norm(error[:3])),'hand_orientation_error_rad':angle,'minimum_joint_margin_rad':float(np.min(np.minimum(q[2:]-bounds_source[:,0],bounds_source[:,1]-q[2:])))}
        # Independently verify a stuck Jacobian solution using bounded nonlinear
        # least squares and the full SO(3) rotation error (no 180deg ambiguity).
        data=mujoco.MjData(model)
        hand=ik.native.bid(model,"R_omnipicker_base_link")
        cylinder=ik.native.bid(model,"test_cylinder")
        source_bounds=np.array([model.jnt_range[ik.native.jid(model,n)] for n in ik.ARM])
        bounds=np.column_stack((np.maximum(source_bounds[:,0],approach_start[2:]-1.1),np.minimum(source_bounds[:,1],approach_start[2:]+1.1)))
        def residual(arm):
            trial=np.r_[q[:2],arm]
            ik.set_pose(model,data,state,trial,np.zeros(3),0.,command,np.zeros(3),0.,None)
            c,s=np.cos(azimuth),np.sin(azimuth)
            around=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            desired=data.xmat[cylinder].reshape(3,3)@around@rotation
            target=circle_target(object_center,desired,center)
            if args.free_bottle_axis_roll:
                return axis_grasp_error(data.xpos[hand],data.xmat[hand].reshape(3,3),center,object_center,data.xmat[cylinder].reshape(3,3)[:,2],local_axis)
            return np.r_[data.xpos[hand]-target,.1*Rotation.from_matrix(desired@data.xmat[hand].reshape(3,3).T).as_rotvec()]
        for seed in (solved,old_q):
            opt=least_squares(residual,np.clip(seed[2:],bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10,diff_step=1e-5)
            e=residual(opt.x)
            if np.linalg.norm(e[:3])+.1*np.linalg.norm(e[3:]) < metrics["hand_position_error_m"]+.01*metrics["hand_orientation_error_rad"]:
                solved=np.r_[q[:2],opt.x]
                angle=float(2*np.arcsin(np.clip(np.linalg.norm(e[3:])/.2,0,1))) if args.free_bottle_axis_roll else float(np.linalg.norm(e[3:])/.1)
                metrics={**metrics,"solver":"bounded_least_squares_directed_axis" if args.free_bottle_axis_roll else "bounded_least_squares_SO3","hand_position_error_m":float(np.linalg.norm(e[:3])),"hand_orientation_error_rad":angle,"orientation_constraint":"directed_bottle_axis_free_axial_roll" if args.free_bottle_axis_roll else "full_rotation","minimum_joint_margin_rad":float(np.min(np.minimum(opt.x-source_bounds[:,0],source_bounds[:,1]-opt.x)))}
        checked={}
        for opening in (open_command,command,max(command-1,0)):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,solved,np.zeros(3),0.,opening,np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            checked[str(opening)]={"pad_bodies":pads,"forbidden_object_bodies":bad,"robot_table_bodies":robot_table,"table_support":table}
        final_check=checked[str(command)]
        metrics.update({"pad_bodies":final_check["pad_bodies"],"nonpad_cylinder_bodies":final_check["forbidden_object_bodies"],"robot_table_bodies":final_check["robot_table_bodies"],"table_support":final_check["table_support"]})
        metrics["mc_owned_waist_fixed"]=not waist_changed
        position_ok=metrics["hand_position_error_m"]<=1e-4 and metrics["hand_orientation_error_rad"]<=1e-3 and metrics["minimum_joint_margin_rad"]>=1e-3
        clear=all(not c["forbidden_object_bodies"] and not c["robot_table_bodies"] for c in checked.values())
        open_clear=not checked[str(open_command)]["pad_bodies"]
        bilateral=any(set(c["pad_bodies"])==ik.DISTALS for k,c in checked.items() if int(k)<=command)
        sweep=[]
        for fraction in np.linspace(0,1,31):
            d=mujoco.MjData(model)
            ik.set_pose(model,d,state,approach_start+(solved-approach_start)*fraction,np.zeros(3),0.,open_command,np.zeros(3),0.,None)
            pads,bad,table,robot_table=ik.contacts(model,d,ik.native.bid(model,"test_cylinder"))
            if pads or bad or robot_table:sweep.append({"fraction":float(fraction),"pads":pads,"forbidden_object":bad,"robot_table":robot_table})
        row_gates={"RUNTIME_STATIC_IK":"PASS" if position_ok else "FAIL","LEFT_ARM_PARK_CLEAR":"PASS" if all(not any(n.startswith('L_') or n.startswith('left_') for n in c["robot_table_bodies"]+c["forbidden_object_bodies"]) for c in checked.values()) else "FAIL","OPEN_PREGRASP_CLEAR":"PASS" if open_clear and not checked[str(open_command)]["forbidden_object_bodies"] and not checked[str(open_command)]["robot_table_bodies"] else "FAIL","CLOSED_CONTACT_CLEAR":"PASS" if bilateral and clear else "FAIL","APPROACH_SWEEP":"PASS" if not sweep else "FAIL"}
        rows.append({"azimuth_rad":azimuth,"metrics":metrics,"contacts":checked,"gates":row_gates,"approach_sweep_violations":sweep,"q_rad":dict(zip(ik.ACTIVE,solved.tolist())),"pass":bool(position_ok and clear and open_clear and bilateral and not sweep)})
    # Arrival alone is insufficient: qualify the whole already requested
    # lift/retract on each existing grasp branch before selecting its winner.
    if args.physical_lift_test:
        qualify_grasp_motion_branches(rows,model,state,max(command-1,0),center,object_center,local_axis,args.post_lift_retract_world_m)
    # Prefer joint margin only among complete task-feasible candidates.
    rows.sort(key=lambda r:(not r["pass"],-r["metrics"]["minimum_joint_margin_rad"] if r["pass"] else r["metrics"]["hand_position_error_m"],r["metrics"]["hand_position_error_m"]))
    out=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="approved-circle-standing-ik").native_path
    record={"classification":"SIMULATION_ONLY_APPROVED_CIRCLE_STANDING_IK","mj_step_calls":0,"grasp_section_center_world_m":object_center.tolist(),"grasp_height_above_center_m":args.grasp_height_above_center_m,"circle_review":str(args.circle_review),"circle_review_sha256":hashlib.sha256(args.circle_review.read_bytes()).hexdigest(),"circle":match,"standing_state":str(args.standing_state),"scene":candidate["scene"],"perception":perception,"root_legs_waist_frozen":True,"commands":{"open":open_command,"match":command,"closure_probe":max(command-1,0)},"rows":rows,"gate":"PASS" if rows[0]["pass"] else "FAIL","dynamics":"NOT_RUN"}
    record["source_joint_limits_rad"]={name:model.jnt_range[ik.native.jid(model,name)].tolist() for name in ik.ACTIVE}
    record["target_source"]=args.target_source
    record["free_bottle_axis_roll"]=args.free_bottle_axis_roll
    record["waist_pitch_offset_rad"]=args.waist_pitch_offset_rad
    record["waist_yaw_offset_rad"]=args.waist_yaw_offset_rad
    record["waist_actuation_required"]=waist_changed
    if upper_posture is not None:record['bounded_upper_posture_solve']=upper_posture
    record["root_legs_waist_frozen"]=not waist_changed
    record["root_and_legs_frozen"]=True
    record["fresh_runtime_rgbd_verified"]=args.target_source in ('fresh-rgbd','fresh-rgbd-snapshot')
    if args.host_model_cache:record['host_model_cache']={'path':str(args.host_model_cache),'sha256':hashlib.sha256(args.host_model_cache.read_bytes()).hexdigest()}
    record['local_bottle_axis_in_hand']=local_axis.tolist();record['grasp_axis_tilt_rad']=theta
    record['explicit_axial_branch']='HAND_MINUS_X' if args.inverted_bottle_axis else 'HAND_PLUS_X'
    if args.use_measured_load_frame:record['execution_grasp_frame']={'center_m':center.tolist(),'nominal_circle_center_m':nominal_center.tolist(),'local_bottle_axis_in_hand':local_axis.tolist(),'classification':'MEASURED_LOAD_FRAME_HYPOTHESIS_NOT_MODEL_CHANGE'}
    if axis_calibration is not None:record['axis_calibration']={'path':str(args.grasp_axis_calibration),'sha256':hashlib.sha256(args.grasp_axis_calibration.read_bytes()).hexdigest(),'record':axis_calibration}
    if args.physical_lift_test and rows[0].get('lift_preflight'):
        record['lift_preflight']=rows[0]['lift_preflight']

    if args.runtime_handoff:
        best=rows[0];target_state=json.loads(json.dumps(state))
        target_state["arm_positions_rad"].update({n:best["q_rad"][n] for n in ik.ARM})
        target_state["waist_positions_rad"].update({n:best["q_rad"][n] for n in ik.ACTIVE[:2]})
        record.update({"classification":"SIMULATION_ONLY_RUNTIME_CAPTURED_STATIC_REPLAN","initial_state":target_state,"gates":{"DOCK_CAPTURE":"PASS" if state.get("motion_stability",{}).get("pass") is True else "FAIL",**best["gates"]}})
        if args.physical_lift_test:record['gates']['LIFT_SWEEP']='PASS' if record.get('lift_preflight',{}).get('pass') else 'FAIL'
        if args.post_lift_retract_world_m is not None:record['gates']['POST_LIFT_RETRACT_SWEEP']='PASS' if record.get('lift_preflight',{}).get('retract_preflight',{}).get('pass_result') else 'FAIL'
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n")
    pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes())
    pg.write_artifact_manifest(out)
    print(json.dumps({"output":str(out),"output_wsl":str(out),"gate":record["gate"],"rows":rows}))


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='--track-native':
        track_native(*sys.argv[2:])
    else:main()
