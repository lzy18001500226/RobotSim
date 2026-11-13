"""Bounded H220 stance study. Offline MjData only; never starts physics/MC.

A feasible point is a witness for the checked posture/path family, not a proof
of a continuous region, locomotion reachability, or physical grasp stability.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import sys
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.grasp import stl_approved_circle_ik as grasp
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory

ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
EPISODE=ROOT/'raicom-stl-xvfb-native-reset-stand-20261111T004845Z'
CONTRACT=ROOT/'approved-circle-cached-runtime-contract-20261111T005041Z/result.json'
ACTIVE=tuple(grasp.ik.ACTIVE)
PADS={'R_hand_narrow3_Link','R_hand_wide3_Link'}
FEET={'left_ankle_roll_link','left_ankle_pitch_link','right_ankle_roll_link','right_ankle_pitch_link'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):pg.write_new_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def angle_error(error):return float(2*np.arcsin(np.clip(np.linalg.norm(error[3:])/.2,0,1)))
def yaw_of(quat):return float(Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_euler('xyz')[2])
def compose_yaw(quat,delta):
    return (Rotation.from_euler('z',delta)*Rotation.from_quat(np.asarray(quat)[[1,2,3,0]])).as_quat()[[3,0,1,2]]


class Study:
    def __init__(self):
        self.reference=json.loads(CONTRACT.read_text());r=self.reference
        if json.loads((EPISODE/'result.json').read_text())['gates']['TWO_TABLE_TRANSFER_EXECUTION']!='PASS':raise ValueError('accepted source episode required')
        self.scene=Path(r['scene']['path'])
        if sha(self.scene)!=r['scene']['sha256']:raise ValueError('frozen model changed')
        self.model=mujoco.MjModel.from_xml_path(str(self.scene));m=self.model
        self.data=mujoco.MjData(m)
        self.ids={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('pelvis','test_cylinder','R_omnipicker_base_link','left_ankle_roll_link','right_ankle_roll_link')}
        if min(self.ids.values())<0:raise ValueError('full frozen scene required')
        self.robot={i for i in range(m.nbody) if m.body_rootid[i]==self.ids['pelvis']}
        self.names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or 'world' for i in range(m.nbody)]
        self.qadr=np.array([m.jnt_qposadr[grasp.ik.native.jid(m,n)] for n in ACTIVE]);self.dofs=np.array([m.jnt_dofadr[grasp.ik.native.jid(m,n)] for n in ACTIVE])
        self.root=int(m.jnt_qposadr[grasp.ik.native.jid(m,'floating_base_joint')]);self.obj=int(m.jnt_qposadr[grasp.ik.native.jid(m,'test_cylinder_free')])
        self.source_bounds=grasp.joint_planning_bounds(m,ACTIVE,r['source_joint_limits_rad'])
        self.center=np.array(r['circle']['center_m']);self.local_axis=np.array(r['local_bottle_axis_in_hand']);self.hand=self.ids['R_omnipicker_base_link']
        self.trajectory=load_registered_trajectory(REPO/'simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic')
        self.opening={}
        for side in ('R','L'):
            for name,value in self.trajectory.pose(100).positions_rad.items():
                n=('right_claw_joint' if side=='R' else 'left_claw_joint') if name=='right_claw_joint' else name if side=='R' else 'L_'+name[2:]
                j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
                if j<0 and name=='right_claw_joint':j=grasp.ik.native.jid(m,side+'_claw_joint')
                self.opening[int(m.jnt_qposadr[j])]=float(value)
        trace=json.loads((EPISODE/'native_hal_pregrasp_trace.json').read_text())['trace']
        stamps={'prepared':trace[0]['host_monotonic_s']*1e9,'grasp':next(x['host_monotonic_s']*1e9 for x in trace if x['stage']=='HAND_CLOSE_HOLD')-100000000,'standing':json.loads((EPISODE/'standing_waist_initial_hold.json').read_text())['request']['created_host_ns']}
        best={k:(math.inf,None) for k in stamps};preparation=[]
        for line in (EPISODE/'native_physics.jsonl').open():
            row=json.loads(line)
            if row['generation']!=1:continue
            for key,stamp in stamps.items():
                if abs(row['host_monotonic_ns']-stamp)<best[key][0]:best[key]=(abs(row['host_monotonic_ns']-stamp),row)
            if stamps['standing']<=row['host_monotonic_ns']<=stamps['prepared'] and (not preparation or row['host_monotonic_ns']-preparation[-1]['host_monotonic_ns']>=350000000):preparation.append(row)
            if row['host_monotonic_ns']>max(stamps.values())+1000000000:break
        self.frames={k:v[1] for k,v in best.items()};self.templates={}
        for label in ('prepared','standing'):
            self.templates[label]=self.frame(self.frames[label])
        self.fixed_object=self.templates['prepared'][self.obj:self.obj+7].copy()
        self.grasp_seed=self.frame(self.frames['grasp'])[self.qadr]
        self.preparation=[self.frame(x) for x in preparation[::max(1,len(preparation)//24)]]+[self.templates['prepared']]
        self.apply(self.templates['prepared']);self.anchor=self.foot_anchor();self.nominal_yaw=yaw_of(self.data.qpos[self.root+3:self.root+7])
        self.object_center=self.data.xpos[self.ids['test_cylinder']].copy();self.axis=self.data.xmat[self.ids['test_cylinder']].reshape(3,3)[:,2].copy();self.section=self.object_center+.01*self.axis
        explicit={frozenset((int(a),int(b))) for a,b in zip(m.pair_geom1,m.pair_geom2)}
        excluded=set(map(int,m.exclude_signature));self.external=[];self.pad_pairs=[]
        for a in range(m.ngeom):
            for b in range(a+1,m.ngeom):
                ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b]);na,nb=self.names[ba],self.names[bb]
                if (ba in self.robot)==(bb in self.robot):continue
                if not ((m.geom_contype[a]&m.geom_conaffinity[b]) or (m.geom_contype[b]&m.geom_conaffinity[a]) or frozenset((a,b)) in explicit):continue
                if ((min(ba,bb)<<16)+max(ba,bb)) in excluded:continue
                if (na=='world' and nb in FEET) or (nb=='world' and na in FEET):continue
                intended=('test_cylinder' in (na,nb) and bool(PADS.intersection((na,nb))))
                self.external.append((a,b,intended))
                if intended:self.pad_pairs.append((a,b,next(n for n in (na,nb) if n in PADS)))
        self.model_identity=sha(self.scene)
        self.model_arrays={n:hashlib.sha256(getattr(m,n).tobytes()).hexdigest() for n in dir(m) if isinstance(getattr(m,n),np.ndarray)}
        self.table_bounds={}
        for body in ('raicom_table_table','delivery_table'):
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body)
            if bid<0:continue
            self.table_bounds[body]=self.body_bounds(bid)

    def frame(self,row):
        apply_recorded_native_frame(self.model,self.data,row)
        q=self.data.qpos.copy()
        for qa,value in self.opening.items():q[qa]=value
        return self.followers(q)

    def apply(self,q,forward=True):
        self.data.qpos[:]=q;self.data.qvel[:]=0.
        if forward:mujoco.mj_forward(self.model,self.data)
        else:mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)

    def foot_anchor(self):return self.data.xpos[[self.ids['left_ankle_roll_link'],self.ids['right_ankle_roll_link']],:2].mean(axis=0)

    def body_bounds(self,bid):
        points=[];m,d=self.model,self.data
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=bid:continue
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                j=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[j]:m.mesh_vertadr[j]+m.mesh_vertnum[j]]
                points.extend(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
        v=np.array(points)
        return {'min':v.min(axis=0).tolist(),'max':v.max(axis=0).tolist()} if len(v) else None

    def relocate(self,q,xy,yaw):
        self.apply(q,False);anchor=self.foot_anchor();delta=yaw-yaw_of(q[self.root+3:self.root+7]);rot=Rotation.from_euler('z',delta).as_matrix()
        result=q.copy();result[self.root:self.root+3]=np.r_[xy,0.]+rot@(q[self.root:self.root+3]-np.r_[anchor,0.])
        result[self.root+3:self.root+7]=compose_yaw(q[self.root+3:self.root+7],delta);result[self.obj:self.obj+7]=self.fixed_object
        return result

    def followers(self,q):
        # Evaluate the frozen MJCF equalities; never refit/change coefficients.
        result=q.copy();m=self.model
        for i in range(m.neq):
            if m.eq_type[i]!=mujoco.mjtEq.mjEQ_JOINT:continue
            j1,j2=int(m.eq_obj1id[i]),int(m.eq_obj2id[i])
            if j2<0:raise ValueError('unexpected frozen joint equality')
            a,b=int(m.jnt_qposadr[j1]),int(m.jnt_qposadr[j2]);x=result[b]-m.qpos0[b]
            result[a]=m.qpos0[a]+sum(m.eq_data[i,k]*x**k for k in range(5))
        return result

    def command(self,q,percent):
        result=q.copy()
        for name,value in self.trajectory.pose(percent).positions_rad.items():
            n='right_claw_joint' if name=='right_claw_joint' else name;j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)
            if j<0 and name=='right_claw_joint':j=grasp.ik.native.jid(self.model,'R_claw_joint')
            result[self.model.jnt_qposadr[j]]=value
        return self.followers(result)

    def solve(self,base,target,seeds,bounds=None,max_nfev=65,*,fixed_waist=False):
        bounds=self.source_bounds if bounds is None else bounds
        free=np.arange(2 if fixed_waist else 0,len(ACTIVE))
        lo,hi=bounds[free,0]+.011,bounds[free,1]-.011
        best=None
        for seed in seeds:
            q=base.copy();full=np.asarray(seed,dtype=float).copy()
            if fixed_waist:full[:2]=base[self.qadr[:2]]
            def fun(x):
                full[free]=x;q[self.qadr]=full;self.apply(q,False)
                return np.r_[grasp.axis_grasp_error(self.data.xpos[self.hand],self.data.xmat[self.hand].reshape(3,3),self.center,target,self.axis,self.local_axis),1e-6*(x-np.asarray(seed)[free])]
            def jac(x):
                fun(x)
                return np.vstack([grasp.circle_task_jacobian(self.model,self.data,self.hand,self.center,self.local_axis,self.dofs[free]),np.eye(len(free))*1e-6])
            result=least_squares(fun,np.clip(full[free],lo+1e-6,hi-1e-6),jac=jac,bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            e=fun(result.x)[:6];entry={'q':q.copy(),'position_error_m':float(np.linalg.norm(e[:3])),'axis_error_rad':angle_error(e),'nfev':result.nfev}
            entry['pass']=entry['position_error_m']<=(.0001 if fixed_waist else .001) and entry['axis_error_rad']<=(.001 if fixed_waist else math.radians(1))
            if best is None or entry['position_error_m']+.1*entry['axis_error_rad']<best['position_error_m']+.1*best['axis_error_rad']:best=entry
            if entry['pass']:return entry
        return best

    def screen(self,q,allow_pads=False):
        self.apply(q);m,d=self.model,self.data;bad=[]
        for c in d.contact:
            a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);na,nb=self.names[a],self.names[b]
            if not (a in self.robot or b in self.robot):continue
            if (na=='world' and nb in FEET) or (nb=='world' and na in FEET):continue
            if allow_pads and 'test_cylinder' in (na,nb) and PADS.intersection((na,nb)) and c.dist>=-.001:continue
            if c.dist < -.00001:bad.append({'bodies':[na,nb],'geoms':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)],'distance_m':float(c.dist)})
        minimum=.05;pair=None
        for a,b,intended in self.external:
            if allow_pads and intended:continue
            lower=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])-m.geom_rbound[a]-m.geom_rbound[b]
            if m.geom_type[a]!=mujoco.mjtGeom.mjGEOM_PLANE and m.geom_type[b]!=mujoco.mjtGeom.mjGEOM_PLANE and lower>=minimum:continue
            dist=float(mujoco.mj_geomDistance(m,d,a,b,minimum,None))
            if dist<minimum:minimum=dist;pair=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (a,b)]
        return {'pass':not bad,'minimum_external_clearance_m_capped_50mm':minimum,'limiting_pair':pair,'violations':sorted(bad,key=lambda x:x['distance_m'])[:4]}

    def candidate(self,xy,yaw,label,*,source_arm_range=False,runtime_arm_only=False,arm_target_inset_rad=0.,root_xy=None):
        began=time.monotonic();base=self.relocate(self.templates[label],xy,yaw)
        if root_xy is not None:
            requested=np.asarray(root_xy,dtype=float)
            if requested.shape!=(2,) or not np.isfinite(requested).all():raise ValueError("finite runtime pelvis XY required")
            # Runtime navigation controls the pelvis, not the ankle midpoint.
            # Crouched and standing templates have different pelvis/ankle offsets.
            # Use one rigid translation for the entire hypothetical preparation.
            xy=np.asarray(xy)+requested-base[self.root:self.root+2]
            base=self.relocate(self.templates[label],xy,yaw)
        start=base[self.qadr].copy()
        bounds=self.source_bounds.copy();exc=np.array([.25,.18]+[np.inf if source_arm_range else 1.1]*7);bounds[:,0]=np.maximum(bounds[:,0],start-exc);bounds[:,1]=np.minimum(bounds[:,1],start+exc)
        if not 0.<=arm_target_inset_rad<=.03:raise ValueError("task arm target inset")
        bounds[2:,0]+=arm_target_inset_rad;bounds[2:,1]-=arm_target_inset_rad
        seed=self.grasp_seed.copy();seed[0]-=yaw-self.nominal_yaw
        endpoint=self.solve(base,self.section,[seed,start],bounds,fixed_waist=runtime_arm_only)
        row={'posture':label,'endpoint_ik':{k:v for k,v in endpoint.items() if k!='q'},'stage':'IK_UNRESOLVED','path':[],'minimum_external_clearance_m_capped_50mm':.05}
        if not endpoint['pass']:return row
        def check(q,phase,allow=False):
            screen=self.screen(q,allow);row['minimum_external_clearance_m_capped_50mm']=min(row['minimum_external_clearance_m_capped_50mm'],screen['minimum_external_clearance_m_capped_50mm']);row['path'].append({'phase':phase,'qpos':q.tolist(),'screen':screen})
            if not screen['pass']:row.update(stage=phase+'_COLLISION',first_blocker=screen['violations'][0]);return False
            return True
        # Replay the existing physical preparation geometry at the candidate
        # support anchor. This is NOT proof that MC can walk to this placement.
        preparation=self.preparation if label=='prepared' else [self.templates['standing']]
        for q in preparation:
            if not check(self.relocate(q,xy,yaw),'PREPARATION'):return row
        final=endpoint['q'];self.apply(final,False);R=self.data.xmat[self.hand].reshape(3,3);direction=R@self.center;direction[2]=0.;direction/=np.linalg.norm(direction)
        pretarget=self.section-.12*direction+np.array([0.,0.,.06])
        pre=self.solve(base,pretarget,[final[self.qadr],start],bounds,fixed_waist=runtime_arm_only)
        if not pre['pass']:row['stage']='PREGRASP_IK_UNRESOLVED';return row
        count=max(2,int(math.ceil(np.max(np.abs(pre['q'][self.qadr]-start))/.06)))
        for u in np.linspace(0.,1.,count+1):
            q=base.copy();q[self.qadr]=start+u*(pre['q'][self.qadr]-start)
            if not check(q,'REACH'):return row
        seed=pre['q'][self.qadr]
        for u in np.linspace(0.,1.,17)[1:]:
            target=(1-u)*pretarget+u*self.section;step=self.solve(base,target,[seed],bounds,max_nfev=45,fixed_waist=runtime_arm_only)
            if not step['pass']:row['stage']='APPROACH_IK_UNRESOLVED';return row
            seed=step['q'][self.qadr]
            if not check(step['q'],'APPROACH'):return row
        final=step['q'];pad_distances={}
        for opening in np.linspace(100,self.reference['commands']['match'],13):
            q=self.command(final,float(opening))
            if not check(q,'CLOSURE',True):return row
        self.apply(q)
        for a,b,pad in self.pad_pairs:
            dist=float(mujoco.mj_geomDistance(self.model,self.data,a,b,.05,None));pad_distances[pad]=min(pad_distances.get(pad,.05),dist)
        row['pad_distances_m']=pad_distances
        if set(pad_distances)!=PADS or any(not -.001<=v<=.0006 for v in pad_distances.values()):row['stage']='PAD_CONTACT_NOT_ESTABLISHED';return row
        closed=q.copy();seed=closed[self.qadr]
        for dz in np.linspace(.005,.06,12):
            shifted=closed.copy();shifted[self.obj+2]+=dz;step=self.solve(shifted,self.section+[0.,0.,dz],[seed],bounds,max_nfev=45,fixed_waist=runtime_arm_only)
            if not step['pass']:row['stage']='LIFT_IK_UNRESOLVED';return row
            seed=step['q'][self.qadr]
            if not check(step['q'],'LIFT',True):return row
        row.update(stage='STATIC_WITNESS',joint_margin_rad=float(np.min(np.minimum(seed-self.source_bounds[:,0],self.source_bounds[:,1]-seed))),elapsed_s=time.monotonic()-began)
        return row


def run(args):
    # Deliberately make accidental physics execution impossible in this process.
    def forbidden(*a,**k):raise RuntimeError('STATIC_STUDY_MJ_STEP_FORBIDDEN')
    mujoco.mj_step=forbidden;mujoco.mj_step1=forbidden;mujoco.mj_step2=forbidden
    out=args.output_dir;out.mkdir(parents=False,exist_ok=True);study=Study();start=time.monotonic()
    dx=np.arange(-2,3)*.08;dy=np.arange(-2,5)*.08;yaws=[0.,-15.,15.,-30.,30.]
    points=[];plan=[]
    for angle in yaws:
        for iy,y in enumerate(dy):
            for ix,x in enumerate(dx):plan.append((angle,iy,ix,x,y))
    plan.sort(key=lambda x:(yaws.index(x[0]),x[3]**2+x[4]**2))
    if args.limit:plan=plan[:args.limit]
    dump(out/'inputs.json',{'scene':str(study.scene),'scene_sha256':study.model_identity,'source_episode':str(EPISODE),'contract':str(CONTRACT),'contract_sha256':sha(CONTRACT),'model_nbody':study.model.nbody,'model_ngeom':study.model.ngeom,'model_npair':study.model.npair,'model_nexclude':study.model.nexclude,'static_follower_states':'EXACT_FROZEN_MJCF_JOINT_POLYNOMIALS_NOT_CSV_INTERPOLATION','anchor_definition':'mean left/right ankle_roll origins XY; not pelvis XY','reference_anchor_world_xy_m':study.anchor.tolist(),'reference_pelvis_yaw_rad':study.nominal_yaw,'fixed_object_center_m':study.object_center.tolist(),'fixed_object_qpos':study.fixed_object.tolist(),'grasp_section_m':study.section.tolist(),'tables':study.table_bounds,'grid_dx_m':dx.tolist(),'grid_dy_m':dy.tolist(),'yaw_offsets_deg':yaws,'active_joints':ACTIVE,'source_bounds_rad':study.source_bounds.tolist(),'solver_excursion_rad':[.25,.18]+[1.1]*7,'classification':'SAMPLED_STATIC_CANDIDATES_NOT_CONTINUOUS_OR_DYNAMICS_PROOF','posture_source':'actual standing and pre-raised/crouched frames; same episode','path_resolution':{'reach_max_joint_step_rad':.06,'approach_samples':16,'closure_samples':13,'lift_samples':12},'criteria':{'ik_position_m':.001,'ik_axis_rad':math.radians(1),'numerical_penetration_m':.00001,'bilateral_proximity_m':.0006,'margin_label_m':.002},'mj_step_calls':0,'scope_limits':['No walking-to-stance proof','No source-mesh continuous swept certificate','No force/support feasibility from static samples','No complete-space infeasibility proof','Static object movement only during hypothetical lift path']})
    for angle,iy,ix,x,y in plan:
        xy=(study.anchor+[x,y]).tolist();yaw=study.nominal_yaw+math.radians(angle);attempts=[]
        for label in ('prepared','standing'):
            item=study.candidate(xy,yaw,label);attempts.append(item)
            if item['stage']=='STATIC_WITNESS':break
        successful=next((a for a in attempts if a['stage']=='STATIC_WITNESS'),None)
        ik_ok=any(a['endpoint_ik']['pass'] for a in attempts)
        chosen=successful or next((a for a in attempts if a['endpoint_ik']['pass']),attempts[0])
        point={'id':f'y{int(angle):+d}_r{iy}_c{ix}','ix':ix,'iy':iy,'yaw_offset_deg':angle,'yaw_world_rad':yaw,'anchor_xy_m':xy,'status':'STATIC_WITNESS' if successful else 'PATH_BLOCKED' if ik_ok and any('COLLISION' in a['stage'] or a['stage']=='PAD_CONTACT_NOT_ESTABLISHED' for a in attempts) else 'UNRESOLVED','endpoint_ik_found':ik_ok,'stage':chosen['stage'],'posture':chosen['posture'],'minimum_external_clearance_m_capped_50mm':chosen['minimum_external_clearance_m_capped_50mm'],'first_blocker':chosen.get('first_blocker'),'attempts':[{'posture':a['posture'],'stage':a['stage'],'endpoint_ik':a['endpoint_ik']} for a in attempts]}
        dump(out/(point['id']+'.json'),{**point,'witness':chosen,'alternative_attempts':attempts})
        points.append(point)
        (out/'progress.json').write_text(json.dumps({'completed':len(points),'total':len(plan),'wall_s':time.monotonic()-start,'counts':{s:sum(p['status']==s for p in points) for s in ('STATIC_WITNESS','PATH_BLOCKED','UNRESOLVED')},'last':point['id']}))
        print(json.dumps({'point':point['id'],'status':point['status'],'stage':point['stage'],'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
    same=all(hashlib.sha256(getattr(study.model,n).tobytes()).hexdigest()==h for n,h in study.model_arrays.items())
    if not same or sha(study.scene)!=study.model_identity:raise RuntimeError('frozen model mutated')
    dump(out/'result.json',{'classification':'BOUNDED_H220_STATIC_STANCE_STUDY','points':points,'wall_s':time.monotonic()-start,'frozen_model_unchanged':same,'mj_step_calls':0,'dynamics':'NOT_RUN','walking':'NOT_RUN','continuous_region_certification':'NOT_RUN','script_sha256':sha(__file__),'command':sys.argv})
    # Finalize the run manifest after render + independent readback verification.


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args();run(args)
