from baseline import *
from plan import render, dist
import argparse,shlex,sys,shutil

# Official URDF range/effort/velocity, not the unrelated DFQ limits.
URDF=UPSTREAM/'robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
SOURCE_LIMITS={j.get('name'):{k:float(v) for k,v in j.find('limit').attrib.items()} for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None}

def verified_plan_dir(plan_dir):
    plan_root=Path(path_governance.normalize_output_path(plan_dir))
    path_governance.validate_output_path(plan_root,require_exists=True,expected_kind='directory')
    manifest_path=plan_root/'artifact_paths.json'
    path_governance.verify_artifacts([path_governance.describe_output_path(manifest_path)])
    plan_path=plan_root/'hand_geometry_plan.json'
    path_governance.validate_output_path(plan_path,require_exists=True,expected_kind='file')
    return plan_root,plan_path

def ik(m,d,pos,R,seed):
    qa,va=indices(m,ARM);w=bid(m,'right_wrist_yaw_link');js=[jid(m,n) for n in ARM]
    lo=np.array([SOURCE_LIMITS[n]['lower']+.015 for n in ARM]);hi=np.array([SOURCE_LIMITS[n]['upper']-.015 for n in ARM])
    def residual(q):
        d.qpos[qa]=q;mj.mj_forward(m,d)
        return np.r_[(d.xpos[w]-pos)*50,Rotation.from_matrix(R@d.xmat[w].reshape(3,3).T).as_rotvec()*5,.005*(q-seed)]
    sol=least_squares(residual,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=200,diff_step=1e-4,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    residual(sol.x)
    err=np.linalg.norm(d.xpos[w]-pos);angle=np.linalg.norm(Rotation.from_matrix(R@d.xmat[w].reshape(3,3).T).as_rotvec())
    if err>.001 or angle>.01:raise RuntimeError(f'IK infeasible {err} {angle}')
    return sol.x,{'position_error_m':err,'rotation_error_rad':angle,'q':sol.x.tolist(),'target':pos.tolist(),'R':R.tolist()}

def bottle_contact(m,d):
    bottle=bid(m,'green_box');forces={};details=[];table=0.;left=False;min_dist=0.
    for i in range(d.ncon):
        c=d.contact[i];a,b=map(int,c.geom);ba,bb=m.geom_bodyid[[a,b]]
        min_dist=min(min_dist,float(c.dist))
        if bottle not in (ba,bb):continue
        other=bb if ba==bottle else ba;oname=name(m,mj.mjtObj.mjOBJ_BODY,other)
        f=np.zeros(6);mj.mj_contactForce(m,d,i,f)
        world=c.frame.reshape(3,3).T@f[:3]*(1 if bb==bottle else -1)
        if oname=='m0_table':table+=world[2]
        if 'left_' in oname:left=True
        if 'right_' in oname:forces[oname]=forces.get(oname,np.zeros(3))+world
        details.append({'geoms':[name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in c.geom], 'other':oname,'world_force_on_bottle':world.tolist(),'contact_force':f.tolist(),'point':c.pos.tolist(),'normal':c.frame[:3].tolist(),'distance':c.dist})
    thumb=any('thumb' in k and np.linalg.norm(v)>.05 for k,v in forces.items())
    opposing=any(('index' in k or 'middle' in k) and np.linalg.norm(v)>.05 for k,v in forces.items())
    return {'thumb':thumb,'opposing':opposing,'left':left,'table_z':table,'hand_z':float(sum(v[2] for v in forces.values())),
      'per_link_forces':{k:v.tolist() for k,v in forces.items()},'contacts':details,'min_distance':min_dist}

def prepare(m, attempt=1, plan_dir=None):
    if plan_dir is None:raise ValueError('an explicit path-governed --plan-dir is required')
    plan_root,plan_path=verified_plan_dir(plan_dir)
    d=init(m);config=json.loads(plan_path.read_text());closed=np.array([config['q'][n] for n in HAND])
    opened=closed.copy();opened[2]=-.03;opened[3:]=.03
    hqa,_=indices(m,HAND);aqa,_=indices(m,ARM)
    d.qpos[hqa]=opened
    # Ground-truth desired wrist, original hand mounting unchanged.
    R=np.eye(3);bottle=np.array([.38,-.22,.8775]);local=np.array(config['bottle_center_local_body']);wpos=bottle-R@local
    seed=np.array([-.5,-.4,.1,1.2,.3,.2,.1])
    grasp,ga=ik(m,d,wpos,R,seed)
    pre,pa=ik(m,d,wpos+np.array([-.060,0,.025]),R,grasp)
    # Lift and transfer use arm joints, never a hidden wrist carriage.
    unload,ua=ik(m,d,wpos+[0,0,(.0005 if attempt==1 else .005)],R,grasp)
    lift,la=ik(m,d,wpos+[0,0,.06],R,grasp)
    transfer,ta=ik(m,d,wpos+[.035,.060,.06],R,lift)
    place,pla=ik(m,d,wpos+[.035,.060,0],R,transfer)
    d.qpos[aqa]=pre;mj.mj_forward(m,d)
    return d,{'open':opened,'closed':closed,'pre':pre,'grasp':grasp,'unload':unload,'lift':lift,'transfer':transfer,'place':place}, {'pre':pa,'grasp':ga,'lift':la,'transfer':ta,'place':pla}

def run(run_id,attempt=1,plan_dir=None):
    assert mj.__version__=='3.3.6'
    if plan_dir is None:raise ValueError('an explicit path-governed --plan-dir is required')
    plan_root,plan_path=verified_plan_dir(plan_dir)
    out=Path(path_governance.create_unique_output_directory(OUT,prefix=run_id).wsl_path)
    episode_copy=out/'executed_episode.py';baseline_copy=out/'executed_baseline.py'
    path_governance.validate_output_path(episode_copy,require_new=True)
    path_governance.validate_output_path(baseline_copy,require_new=True)
    shutil.copy2(__file__,episode_copy);shutil.copy2(Path(__file__).with_name('baseline.py'),baseline_copy)
    m=build(out/'model.xml');d,poses,plans=prepare(m,attempt,plan_root)
    plan_record={'poses':{k:v.tolist() for k,v in poses.items()},'ik':plans,
       'evidence_directory_paths':path_governance.describe_output_path(out),
       'artifact_manifest':path_governance.describe_output_path(out/'artifact_paths.json')}
    path_governance.write_new_text(out/'plan.json',json.dumps(plan_record,indent=2)+'\n')
    js=m.actuator_trnid[:,0];qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js];w=bid(m,'right_wrist_yaw_link');b=bid(m,'green_box')
    target=d.qpos[qa].copy();hqa,hva=indices(m,HAND);aqa,ava=indices(m,ARM)
    hi=np.array([list(qa).index(q) for q in hqa]);ai=np.array([list(qa).index(q) for q in aqa]);hand=np.array(['hand' in name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in js]);kp=m.actuator_gainprm[:,0]
    cap=np.array([SOURCE_LIMITS[name(m,mj.mjtObj.mjOBJ_JOINT,j)]['effort'] for j in js]);vlim=np.array([SOURCE_LIMITS[name(m,mj.mjtObj.mjOBJ_JOINT,j)]['velocity'] for j in js]);lo=np.array([SOURCE_LIMITS[name(m,mj.mjtObj.mjOBJ_JOINT,j)]['lower'] for j in js]);up=np.array([SOURCE_LIMITS[name(m,mj.mjtObj.mjOBJ_JOINT,j)]['upper'] for j in js])
    initial_contacts=contacts(m,d);initial_bad=[c for c in initial_contacts if c['active'] and c['distance']<-.0005]
    result={'run_id':run_id,'passed':False,'status':'RUNNING','robot':'G1-29DoF Rev1.0 + bilateral official Dex3-1; right-only manipulation',
       'mujoco':mj.__version__,'source_revision':SOURCE_REV,'source_model_sha256':sha(SOURCE),'runtime_model_sha256':sha(out/'model.xml'),
       'runner_sha256':sha(__file__),'RobotSim_SHA':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'dirty':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
       'no_object_qpos_writes_after_initialization':True,'object_equality_count':m.neq,'mocap_count':m.nmocap,'seed':42,'attempt':attempt,
       'initial_contacts':initial_contacts,'thresholds':{'velocity':'URDF per-joint','penetration_m':.003,'hold_opposing_fraction':.8,'lift_m':.05,'load_support_N':.8*.57*9.81},'stages':[],
       'evidence_directory':str(out),'evidence_directory_paths':path_governance.describe_output_path(out),
       'artifact_manifest':path_governance.describe_output_path(out/'artifact_paths.json')}
    result_path=out/'result.json'
    path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n')
    log_path=out/'run.log';trace_path=out/'trace.jsonl';video_path=out/'episode.mp4'
    for output_path in (log_path,trace_path,video_path):
        path_governance.validate_output_path(output_path,require_new=True)
    log=log_path.open('x');trace=trace_path.open('x');
    def emit(s):print(s,flush=True);log.write(s+'\n');log.flush()
    command=f'MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python scripts/research/g1_dex3_physical/episode.py --run-id {shlex.quote(run_id)} --attempt {attempt} --plan-dir {shlex.quote(str(plan_root))}'
    result['reproduction_command']=command;emit(command)
    if initial_bad:raise RuntimeError('invalid initial contacts '+str(initial_bad))
    import cv2
    video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
    if not video.isOpened():raise RuntimeError(f'failed to open video output: {video_path}')
    renderer=mj.Renderer(m,height=720,width=960);cam=mj.MjvCamera();cam.lookat[:]=[.35,-.26,.92];cam.distance=1.25;cam.azimuth=125;cam.elevation=-22
    opt=mj.MjvOption();opt.geomgroup[:]=1
    peakv=0.;peakpen=0.;peakforce=0.;holdstats=[];ready=[];b0=None;prevpos=d.xpos[b].copy();maxjump=0.;lastvideo=-1.;fail=None
    phases=[('SETTLE',1.5,'pre','open'),('PREGRASP',1.,'pre','open'),('APPROACH',2.,'grasp','open'),('GRASP',4.,'grasp','closed'),('GRIP_HOLD',1.5,'grasp','closed')]
    # Same predeclared contact target in both attempts; only the unload distance differs.
    closed=poses['closed'].copy()
    closed+=np.array([0,0,-.18,.18,.18,.18,.18])
    for k,n in enumerate(HAND):closed[k]=np.clip(closed[k],SOURCE_LIMITS[n]['lower']+.02,SOURCE_LIMITS[n]['upper']-.02)
    poses['closed']=closed
    result['executed_hand_targets']={key:dict(zip(HAND,poses[key].tolist())) for key in ['open','closed']}
    result['load_transfer_wrist_command_m']=.0005 if attempt==1 else .005
    path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True)
    try:
        stageidx=0
        while stageidx<len(phases):
            phase,duration,armkey,handkey=phases[stageidx];start=target.copy();end=target.copy();end[ai]=poses[armkey];end[hi]=poses[handkey]
            emit(f'{phase} t={d.time:.6f}');beg=d.time;phase_contacts=[];phase_rows=[]
            for k in range(round(duration/m.opt.timestep)):
                u=(k+1)/(duration/m.opt.timestep);s=u*u*u*(10+u*(-15+6*u));target=start+(end-start)*s
                # SDK position PD contract; implicit damping for numerical stability. Body gravity compensation only.
                d.ctrl[:]=target+np.where(hand,0,d.qfrc_bias[va]/kp)
                mj.mj_step(m,d)
                if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite state')
                v=abs(d.qvel[va]);peakv=max(peakv,float(v.max()))
                if (v>vlim+1e-4).any():raise RuntimeError('source velocity '+name(m,mj.mjtObj.mjOBJ_JOINT,js[np.argmax(v/vlim)])+f' {v.max()}')
                if (d.qpos[qa]<lo-.002).any() or (d.qpos[qa]>up+.002).any():raise RuntimeError('source position limit')
                if (abs(d.actuator_force)>cap+1e-6).any():raise RuntimeError('source effort')
                cs=bottle_contact(m,d);peakpen=max(peakpen,-cs['min_distance']);phase_contacts.append(cs)
                maxjump=max(maxjump,float(np.linalg.norm(d.xpos[b]-prevpos)));prevpos=d.xpos[b].copy()
                if cs['left']:raise RuntimeError('left hand assistance')
                if cs['min_distance']<-.003:raise RuntimeError('penetration exceeds 3 mm')
                peakforce=max(peakforce,max([np.linalg.norm(v) for v in cs['per_link_forces'].values()] or [0]))
                # Any unexpected active robot/environment or self contact is captured and checked.
                unsafe=[c for c in contacts(m,d) if c['active'] and 'green_box' not in c['bodies'] and 'world' not in c['bodies'] and c['distance']<-.001]
                if unsafe:raise RuntimeError('unexpected collision '+str(unsafe[0]))
                if k%4==0:
                    row={'time':d.time,'stage':phase,'target':target.tolist(),'joint_names':[name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in js],
                       'q':d.qpos[qa].tolist(),'v':d.qvel[va].tolist(),'effort':d.actuator_force.tolist(),'bottle_pos':d.xpos[b].tolist(),'bottle_quat':d.xquat[b].tolist(),
                       'wrist_pos':d.xpos[w].tolist(),'wrist_R':d.xmat[w].tolist(),**cs}
                    trace.write(json.dumps(row)+'\n');phase_rows.append(row)
                if d.time-lastvideo>=1/30-1e-8:
                    renderer.update_scene(d,camera=cam,scene_option=opt);rgb=renderer.render();bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    cv2.putText(bgr,f'{phase}  t={d.time:.2f}s  handZ={cs["hand_z"]:.2f}N tableZ={cs["table_z"]:.2f}N',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2)
                    video.write(bgr);lastvideo=d.time
            render(m,d,out/f'{stageidx:02d}_{phase}.png',np.array([.4,-.3,.93]),.65,115,-25)
            summary={'phase':phase,'end_time':d.time,'opposing_fraction':float(np.mean([c['thumb'] and c['opposing'] for c in phase_contacts])),
               'hand_support_mean_N':float(np.mean([c['hand_z'] for c in phase_contacts])),'table_support_mean_N':float(np.mean([c['table_z'] for c in phase_contacts])),
               'bottle_pos':d.xpos[b].tolist(),'wrist_pos':d.xpos[w].tolist()}
            result['stages'].append(summary);emit(json.dumps(summary))
            if phase=='GRIP_HOLD':
                b0=d.xpos[b].copy();holdstats=phase_contacts
                if summary['opposing_fraction']<.8:raise RuntimeError('GRASP_READY thumb + opposing persistence below 80%')
                phases += [('LOAD_TRANSFER',2.,'unload','closed')]
            if phase=='LOAD_TRANSFER':
                # Probe the declared 0.5/5 mm arm elevation. Only measured forces decide readiness.
                if summary['hand_support_mean_N']<.8*.57*9.81:
                    raise RuntimeError('LOAD_READY measured upward support below 80% bottle weight')
                phases += [('LIFT',3.,'lift','closed'),('HOLD',1.5,'lift','closed'),('TRANSFER',3.,'transfer','closed'),('PLACE',3.,'place','closed'),('RELEASE',3.,'place','open'),('SETTLE_FINAL',2.,'place','open')]
            if phase=='LIFT' and (d.xpos[b][2]-b0[2]<.05 or cs['table_z']>.1 or not(cs['thumb'] and cs['opposing'])):raise RuntimeError('physical 50mm lift gate')
            if phase=='SETTLE_FINAL':
                if cs['thumb'] or cs['opposing'] or cs['table_z']<4.:raise RuntimeError('release/table support gate')
                result['passed']=True
            path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True);stageidx+=1
        result['status']='PASS' if result['passed'] else 'FAIL'
    except Exception as e:
        fail=str(e);emit('FAIL '+fail);result['status']='FAIL';result['first_failure']=fail
        render(m,d,out/'failure.png',np.array([.4,-.3,.93]),.65,115,-25)
    finally:
        result.update({'peak_joint_speed_rad_s':peakv,'max_penetration_m':peakpen,'peak_link_contact_force_N':peakforce,'max_physics_step_bottle_jump_m':maxjump,'physics_steps':round(d.time/m.opt.timestep),'final_bottle_pos':d.xpos[b].tolist()})
        trace.close();log.close();video.release();renderer.close()
        path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True)
    manifest=path_governance.write_artifact_manifest(out)
    result['artifact_manifest']=manifest['manifest']
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--attempt',type=int,default=1);p.add_argument('--plan-dir',required=True);a=p.parse_args();sys.exit(0 if run(a.run_id,a.attempt,a.plan_dir)['passed'] else 2)
