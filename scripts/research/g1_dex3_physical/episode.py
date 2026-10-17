from baseline import *
from plan import render, dist
import argparse,shlex,sys,shutil

# Official URDF range/effort/velocity, not the unrelated DFQ limits.
URDF=UPSTREAM/'robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
SOURCE_LIMITS={j.get('name'):{k:float(v) for k,v in j.find('limit').attrib.items()} for j in ET.parse(URDF).getroot().findall('joint') if j.find('limit') is not None}
BOTTLE_MASS_KG=.570
GRAVITY_M_S2=9.81
CONTACT_FORCE_FLOOR_N=.05
CONTACT_PERSISTENCE_FRACTION=.80
LOAD_READY_SUPPORT_N=.80*BOTTLE_MASS_KG*GRAVITY_M_S2
GRIP_PRELOAD_RESERVE_N=1.20*BOTTLE_MASS_KG*GRAVITY_M_S2
HOLD_BOTTLE_DRIFT_LIMIT_M=.001
CONTACT_MIGRATION_LIMIT_FRACTION=.05
EXTRA_CLOSURE_DIRECTION=np.array([0.,-1.,-1.,1.,1.,1.,1.])
ALLOWED_GRASP_HEIGHT_OFFSETS_M=(0.,.060,.070)
ALLOWED_PRELOAD_ADVANCES_RAD=(0.,.050)

def grasp_height_description(offset_m):
    offset=float(offset_m)
    if abs(offset) <= 1e-12:
        return 'baseline grasp height (0 mm offset)'
    if abs(offset-.060) <= 1e-12:
        return 'C3 near-midpoint grasp-height candidate (+60 mm wrist offset)'
    if abs(offset-.070) <= 1e-12:
        return 'C2 bottle-shoulder underface candidate (+70 mm wrist offset)'
    raise ValueError(f'unsupported grasp-height offset: {offset} m')

def shifted_grasp_wrist_target(position, offset_m=0.):
    offset=float(offset_m)
    if not np.isfinite(offset) or not any(abs(offset-candidate)<=1e-12 for candidate in ALLOWED_GRASP_HEIGHT_OFFSETS_M):
        raise ValueError(f'grasp-height offset must be one of {ALLOWED_GRASP_HEIGHT_OFFSETS_M} m')
    target=np.asarray(position,dtype=float).copy()
    if target.shape!=(3,) or not np.isfinite(target).all():
        raise ValueError('grasp wrist target must be a finite 3D position')
    target[2]+=offset
    return target

def close_targets(closed, extra_closure_rad=0.):
    if not 0. <= extra_closure_rad <= .05:
        raise ValueError('extra closure must be between 0 and 0.05 rad')
    target=np.asarray(closed,dtype=float)+EXTRA_CLOSURE_DIRECTION*extra_closure_rad
    for k,n in enumerate(HAND):
        lo=SOURCE_LIMITS[n]['lower']+.02;hi=SOURCE_LIMITS[n]['upper']-.02
        if target[k] < lo or target[k] > hi:
            raise ValueError(f'extra closure would exceed source joint range: {n}={target[k]}')
    return target

def validate_preload_advance(advance_rad):
    advance=float(advance_rad)
    if not np.isfinite(advance) or not any(abs(advance-value)<=1e-12 for value in ALLOWED_PRELOAD_ADVANCES_RAD):
        raise ValueError(f'preload advancement must be one of {ALLOWED_PRELOAD_ADVANCES_RAD} rad')
    return advance

def preload_target(closed, advance_rad=0.):
    advance=validate_preload_advance(advance_rad)
    return close_targets(closed,advance)

def summarize_window(samples,bottle_positions,bottle_diameter_m):
    if not samples:
        raise ValueError('cannot summarize an empty physical phase')
    support=np.asarray([s['hand_z'] for s in samples]);table=np.asarray([s['table_z'] for s in samples])
    capacity=np.asarray([sum(c['vertical_friction_capacity_N'] for c in s['contacts'] if c['other'].startswith('right_hand_')) for s in samples])
    first_points={};max_migration=0.;normal_by_link={};tangent_by_link={};ratio_by_link={}
    for s in samples:
        for c in s['contacts']:
            if not c['other'].startswith('right_hand_'):continue
            key=(c['other'],tuple(c['geoms']))
            local=np.asarray(c['point_bottle_local_m'])
            first_points.setdefault(key,local.copy())
            max_migration=max(max_migration,float(np.linalg.norm(local-first_points[key])))
            normal_by_link.setdefault(c['other'],[]).append(c['normal_force_N'])
            tangent_by_link.setdefault(c['other'],[]).append(c['tangential_force_N'])
            ratio_by_link.setdefault(c['other'],[]).append(c['friction_pyramid_tangent_utilization'])
    p=np.asarray(bottle_positions);start=p[0]
    xy_drift=float(np.max(np.linalg.norm(p[:,:2]-start[:2],axis=1)))
    return {
        'simultaneous_thumb_opposing_fraction':float(np.mean([s['thumb'] and s['opposing'] for s in samples])),
        'hand_support_N':{'mean':float(np.mean(support)),'p05':float(np.percentile(support,5)),'p95':float(np.percentile(support,95)),'min':float(np.min(support)),'max':float(np.max(support))},
        'table_support_N':{'mean':float(np.mean(table)),'p05':float(np.percentile(table,5)),'p95':float(np.percentile(table,95)),'min':float(np.min(table)),'max':float(np.max(table))},
        'ideal_vertical_friction_capacity_screen_N':{'p05':float(np.percentile(capacity,5)),'mean':float(np.mean(capacity)),'max':float(np.max(capacity))},
        'max_contact_migration_mm':1000.*max_migration,
        'max_contact_migration_limit_mm':1000.*CONTACT_MIGRATION_LIMIT_FRACTION*bottle_diameter_m,
        'max_bottle_horizontal_drift_mm':1000.*xy_drift,
        'per_link_normal_force_mean_N':{k:float(np.mean(v)) for k,v in normal_by_link.items()},
        'per_link_tangential_force_mean_N':{k:float(np.mean(v)) for k,v in tangent_by_link.items()},
        'per_link_friction_pyramid_utilization_p95':{k:float(np.percentile(v,95)) for k,v in ratio_by_link.items()},
    }

def evaluate_readiness_gates(hold,load=None):
    contact=hold['simultaneous_thumb_opposing_fraction']>=CONTACT_PERSISTENCE_FRACTION
    preload=contact and hold['ideal_vertical_friction_capacity_screen_N']['p05']>=GRIP_PRELOAD_RESERVE_N
    stable=contact and hold['max_bottle_horizontal_drift_mm']<=1000.*HOLD_BOTTLE_DRIFT_LIMIT_M and hold['max_contact_migration_mm']<=hold['max_contact_migration_limit_mm']
    gates={'CONTACT_ESTABLISHED':'PASS' if contact else 'FAIL',
           'GRIP_PRELOAD_ESTABLISHED':'PASS' if preload else 'FAIL',
           'GRASP_STABLE':'PASS' if stable else 'FAIL',
           'LOAD_READY':'NOT_REACHED'}
    if load is not None:
        load_ready=(stable and
            load['hand_support_N']['p05']>=LOAD_READY_SUPPORT_N and
            load['table_support_N']['p95']<=BOTTLE_MASS_KG*GRAVITY_M_S2-LOAD_READY_SUPPORT_N and
            load['simultaneous_thumb_opposing_fraction']>=CONTACT_PERSISTENCE_FRACTION and
            load['max_contact_migration_mm']<=load['max_contact_migration_limit_mm'])
        gates['LOAD_READY']='PASS' if load_ready else 'FAIL'
    return gates

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

def minimum_penetration_contact(m,d):
    min_dist=0.;min_contact=None
    for i in range(d.ncon):
        c=d.contact[i];a,b=map(int,c.geom);ba,bb=m.geom_bodyid[[a,b]]
        if float(c.dist)<min_dist:
            min_dist=float(c.dist)
            min_contact={'distance_m':float(c.dist),'geom_ids':[a,b],
                'geom_names':[name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in (a,b)],
                'body_ids':[int(ba),int(bb)],
                'body_names':[name(m,mj.mjtObj.mjOBJ_BODY,body) for body in (ba,bb)],
                'contact_position_world_m':c.pos.tolist(),
                'geom_positions_world_m':[d.geom_xpos[g].tolist() for g in (a,b)]}
    return min_dist,min_contact

def bottle_contact(m,d):
    bottle=bid(m,'green_box');forces={};details=[];table=0.;left=False;hand_wrench=np.zeros(6)
    min_dist,min_contact=minimum_penetration_contact(m,d)
    bottle_xpos=d.xpos[bottle].copy();bottle_R=d.xmat[bottle].reshape(3,3)
    for i in range(d.ncon):
        c=d.contact[i];a,b=map(int,c.geom);ba,bb=m.geom_bodyid[[a,b]]
        if bottle not in (ba,bb):continue
        other=bb if ba==bottle else ba;oname=name(m,mj.mjtObj.mjOBJ_BODY,other)
        f=np.zeros(6);mj.mj_contactForce(m,d,i,f)
        world=c.frame.reshape(3,3).T@f[:3]*(1 if bb==bottle else -1)
        if oname=='m0_table':table+=world[2]
        if 'left_' in oname:left=True
        geoms=[name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in c.geom]
        fn=max(float(f[0]),0.);mu=np.asarray(c.friction);mu1=max(float(mu[0]),1e-12);mu2=max(float(mu[1]),1e-12)
        ratio=(abs(float(f[1]))/mu1+abs(float(f[2]))/mu2)/max(fn,1e-12)
        frame=c.frame.reshape(3,3);up_local=frame@np.array([0.,0.,1.]);vcap=0.
        if oname.startswith('right_hand_') and abs(up_local[0])<.01:
            denom=abs(up_local[1])/mu1+abs(up_local[2])/mu2
            if denom>1e-12:vcap=fn/denom
        point_local=bottle_R.T@(c.pos-bottle_xpos)
        if oname.startswith('right_hand_'):
            forces[oname]=forces.get(oname,np.zeros(3))+world
            hand_wrench[:3]+=world;hand_wrench[3:]+=np.cross(c.pos-d.xipos[bottle],world)
        gm=[]
        for g in c.geom:
            g=int(g);gm.append({'name':name(m,mj.mjtObj.mjOBJ_GEOM,g),'friction':m.geom_friction[g].tolist(),
                'condim':int(m.geom_condim[g]),'solref':m.geom_solref[g].tolist(),'solimp':m.geom_solimp[g].tolist(),
                'margin':float(m.geom_margin[g]),'gap':float(m.geom_gap[g])})
        details.append({'geoms':geoms,'other':oname,'world_force_on_bottle':world.tolist(),'contact_force':f.tolist(),
            'normal_force_N':fn,'tangential_force_N':float(np.linalg.norm(f[1:3])),'friction':mu.tolist(),'condim':int(c.dim),
            'solref':np.asarray(c.solref).tolist(),'solimp':np.asarray(c.solimp).tolist(),'includemargin':float(c.includemargin),
            'friction_pyramid_tangent_utilization':ratio,'vertical_friction_capacity_N':vcap,
            'contact_up_in_frame':up_local.tolist(),'geom_contact_parameters':gm,
            'point':c.pos.tolist(),'point_bottle_local_m':point_local.tolist(),
            'normal':c.frame[:3].tolist(),'distance':float(c.dist)})
    thumb=any('thumb' in k and np.linalg.norm(v)>CONTACT_FORCE_FLOOR_N for k,v in forces.items())
    opposing=any(('index' in k or 'middle' in k) and np.linalg.norm(v)>CONTACT_FORCE_FLOOR_N for k,v in forces.items())
    return {'thumb':thumb,'opposing':opposing,'left':left,'table_z':table,'hand_z':float(sum(v[2] for v in forces.values())),
      'hand_wrench_on_bottle':hand_wrench.tolist(),'per_link_forces':{k:v.tolist() for k,v in forces.items()},
      'contacts':details,'min_distance':min_dist,'min_distance_contact':min_contact}

def prepare(m, attempt=1, plan_dir=None, grasp_height_offset_m=0.):
    if plan_dir is None:raise ValueError('an explicit path-governed --plan-dir is required')
    plan_root,plan_path=verified_plan_dir(plan_dir)
    d=init(m);config=json.loads(plan_path.read_text());closed=np.array([config['q'][n] for n in HAND])
    opened=closed.copy();opened[2]=-.03;opened[3:]=.03
    hqa,_=indices(m,HAND);aqa,_=indices(m,ARM)
    d.qpos[hqa]=opened
    # Ground-truth desired wrist, original hand mounting unchanged.
    R=np.eye(3);bottle=np.array([.38,-.22,.8775]);local=np.array(config['bottle_center_local_body'])
    wpos=shifted_grasp_wrist_target(bottle-R@local,grasp_height_offset_m)
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

def run(run_id,attempt=1,plan_dir=None,extra_closure_rad=0.,hand_kp=HAND_KP_DEFAULT,grasp_height_offset_m=0.,preload_advance_rad=0.):
    assert mj.__version__=='3.3.6'
    if plan_dir is None:raise ValueError('an explicit path-governed --plan-dir is required')
    if not 0. <= extra_closure_rad <= .05:raise ValueError('extra closure must be between 0 and 0.05 rad')
    if not 0. < hand_kp <= HAND_KP_DIAGNOSTIC_MAX:raise ValueError(f'hand Kp must be in (0, {HAND_KP_DIAGNOSTIC_MAX}]')
    grasp_height_offset_m=float(shifted_grasp_wrist_target(np.zeros(3),grasp_height_offset_m)[2])
    preload_advance_rad=validate_preload_advance(preload_advance_rad)
    plan_root,plan_path=verified_plan_dir(plan_dir)
    out=reserve_output_run(run_id)
    episode_copy=out/'executed_episode.py';baseline_copy=out/'executed_baseline.py'
    path_governance.validate_output_path(episode_copy,require_new=True)
    path_governance.validate_output_path(baseline_copy,require_new=True)
    shutil.copy2(__file__,episode_copy);shutil.copy2(Path(__file__).with_name('baseline.py'),baseline_copy)
    m=build(out/'model.xml',hand_kp=hand_kp);d,poses,plans=prepare(m,attempt,plan_root,grasp_height_offset_m)
    plan_record={'poses':{k:v.tolist() for k,v in poses.items()},'ik':plans,
       'grasp_height_offset_m':grasp_height_offset_m,
       'grasp_geometry_correction':grasp_height_description(grasp_height_offset_m),
       'preload_advance_rad':preload_advance_rad,
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
       'initial_contacts':initial_contacts,'hand_control':{'kp':hand_kp,'kd':HAND_KD_DEFAULT,'effort_caps':'unchanged source URDF/MJCF'},
       'grasp_height_offset_m':grasp_height_offset_m,
       'grasp_geometry_correction':grasp_height_description(grasp_height_offset_m),
       'preload_advance_rad':preload_advance_rad,
       'preload_controller':'slow position-target ramp after contact closure' if preload_advance_rad else 'none',
       'thresholds':{'velocity':'unchanged source per-joint','penetration_m':.003,'contact_force_floor_N':CONTACT_FORCE_FLOOR_N,
           'contact_persistence':CONTACT_PERSISTENCE_FRACTION,'preload_screen_reserve_N':GRIP_PRELOAD_RESERVE_N,
           'hold_bottle_drift_m':HOLD_BOTTLE_DRIFT_LIMIT_M,'contact_migration_fraction_of_bottle_diameter':CONTACT_MIGRATION_LIMIT_FRACTION,
           'load_support_p05_N':LOAD_READY_SUPPORT_N,'table_support_p95_max_N':BOTTLE_MASS_KG*GRAVITY_M_S2-LOAD_READY_SUPPORT_N,'lift_m':.05},
       'extra_closure_rad':extra_closure_rad,'gates':{'CONTACT_ESTABLISHED':'NOT_REACHED','GRIP_PRELOAD_ESTABLISHED':'NOT_REACHED','GRASP_STABLE':'NOT_REACHED','LOAD_READY':'NOT_REACHED'},
       'gate_measurements':{},'stages':[],
       'evidence_directory':str(out),'evidence_directory_paths':path_governance.describe_output_path(out),
       'artifact_manifest':path_governance.describe_output_path(out/'artifact_paths.json')}
    result_path=out/'result.json'
    path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n')
    log_path=out/'run.log';trace_path=out/'trace.jsonl';video_path=out/'episode.mp4'
    for output_path in (log_path,trace_path,video_path):
        path_governance.validate_output_path(output_path,require_new=True)
    log=log_path.open('x');trace=trace_path.open('x');
    def emit(s):print(s,flush=True);log.write(s+'\n');log.flush()
    command=f'MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python scripts/research/g1_dex3_physical/episode.py --run-id {shlex.quote(run_id)} --attempt {attempt} --extra-closure-rad {extra_closure_rad:.3f} --hand-kp {hand_kp:.3f} --grasp-height-offset-m {grasp_height_offset_m:.3f} --preload-advance-rad {preload_advance_rad:.3f} --plan-dir {shlex.quote(str(plan_root))}'
    result['reproduction_command']=command;emit(command)
    if initial_bad:raise RuntimeError('invalid initial contacts '+str(initial_bad))
    import cv2
    video=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,720))
    if not video.isOpened():raise RuntimeError(f'failed to open video output: {video_path}')
    renderer=mj.Renderer(m,height=720,width=960);cam=mj.MjvCamera();cam.lookat[:]=[.35,-.26,.92];cam.distance=1.25;cam.azimuth=125;cam.elevation=-22
    opt=mj.MjvOption();opt.geomgroup[:]=1
    peakv=0.;peakpen=0.;peakforce=0.;holdstats=[];ready=[];b0=None;prevpos=d.xpos[b].copy();maxjump=0.;lastvideo=-1.;fail=None;penetration_event=None
    phases=[('SETTLE',1.5,'pre','open'),('PREGRASP',1.,'pre','open'),('APPROACH',2.,'grasp','open'),('GRASP',4.,'grasp','closed')]
    # Same predeclared contact target in both attempts; only the unload distance differs.
    closed=poses['closed'].copy()
    closed+=np.array([0,0,-.18,.18,.18,.18,.18])
    closed=close_targets(closed,extra_closure_rad)
    poses['closed']=closed
    poses['preload']=preload_target(closed,preload_advance_rad)
    if preload_advance_rad:phases.append(('GRIP_PRELOAD',1.,'grasp','preload'))
    phases.append(('GRIP_HOLD',1.5,'grasp','preload' if preload_advance_rad else 'closed'))
    result['executed_hand_targets']={key:dict(zip(HAND,poses[key].tolist())) for key in ['open','closed']}
    if preload_advance_rad:result['executed_hand_targets']['preload']=dict(zip(HAND,poses['preload'].tolist()))
    result['load_transfer_wrist_command_m']=.0005 if attempt==1 else .005
    path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True)
    try:
        stageidx=0
        while stageidx<len(phases):
            phase,duration,armkey,handkey=phases[stageidx];start=target.copy();end=target.copy();end[ai]=poses[armkey];end[hi]=poses[handkey]
            emit(f'{phase} t={d.time:.6f}');beg=d.time;phase_contacts=[];phase_rows=[];phase_bottle_positions=[]
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
                cs=bottle_contact(m,d);peakpen=max(peakpen,-cs['min_distance']);phase_contacts.append(cs);phase_bottle_positions.append(d.xpos[b].copy())
                maxjump=max(maxjump,float(np.linalg.norm(d.xpos[b]-prevpos)));prevpos=d.xpos[b].copy()
                if cs['left']:raise RuntimeError('left hand assistance')
                if cs['min_distance']<-.003:
                    penetration_event={'time_s':float(d.time),'threshold_m':-.003,
                        'minimum_distance_m':cs['min_distance'],
                        'contact_pair':cs['min_distance_contact']}
                    result['penetration_gate_event']=penetration_event
                    path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True)
                    emit('PENETRATION_GATE '+json.dumps(penetration_event))
                    raise RuntimeError('penetration exceeds 3 mm '+json.dumps(penetration_event))
                peakforce=max(peakforce,max([np.linalg.norm(v) for v in cs['per_link_forces'].values()] or [0]))
                # Any unexpected active robot/environment or self contact is captured and checked.
                unsafe=[c for c in contacts(m,d) if c['active'] and 'green_box' not in c['bodies'] and 'world' not in c['bodies'] and c['distance']<-.001]
                if unsafe:raise RuntimeError('unexpected collision '+str(unsafe[0]))
                if k%4==0:
                    row={'time':d.time,'stage':phase,'target':target.tolist(),'q_error':(target-d.qpos[qa]).tolist(),
                       'effort_utilization':(abs(d.actuator_force)/cap).tolist(),'velocity_utilization':(abs(d.qvel[va])/vlim).tolist(),
                       'control':d.ctrl.tolist(),'joint_names':[name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in js],
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
            body_geom=gid(m,'bottle_body');diameter=2*float(m.geom_size[body_geom,0])
            summary['contact_and_load_metrics']=summarize_window(phase_contacts,phase_bottle_positions,diameter)
            result['stages'].append(summary);emit(json.dumps(summary))
            if phase=='GRASP' and preload_advance_rad:
                tail=phase_contacts[-max(1,round(.25/m.opt.timestep)):]
                contact_fraction=float(np.mean([c['thumb'] and c['opposing'] for c in tail]))
                result['gates']['CONTACT_ESTABLISHED']='PASS' if contact_fraction>=CONTACT_PERSISTENCE_FRACTION else 'FAIL'
                result['gate_measurements']['GRASP_PRELOAD_ENTRY']={'simultaneous_thumb_opposing_fraction_last_0_25_s':contact_fraction}
                if contact_fraction<CONTACT_PERSISTENCE_FRACTION:
                    raise RuntimeError('GRIP_PRELOAD requires stable thumb plus opposing contact before advancing targets')
            if phase=='GRIP_HOLD':
                b0=d.xpos[b].copy();holdstats=phase_contacts
                gm=summary['contact_and_load_metrics']
                result['gates']=evaluate_readiness_gates(gm)
                result['gate_measurements']['GRIP_HOLD']=gm
                phases += [('LOAD_TRANSFER',2.,'unload','closed')]
            if phase=='LOAD_TRANSFER':
                gm=summary['contact_and_load_metrics'];result['gate_measurements']['LOAD_TRANSFER']=gm
                result['gates']=evaluate_readiness_gates(result['gate_measurements']['GRIP_HOLD'],gm)
                if result['gates']['LOAD_READY']!='PASS':
                    raise RuntimeError('LOAD_READY requires >=80% hand support (p05), <=20% table reaction (p95), stable contact and bounded migration')
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
        if penetration_event is not None:result['penetration_gate_event']=penetration_event
        trace.close();log.close();video.release();renderer.close()
        path_governance.write_reserved_text(out,result_path,json.dumps(result,indent=2)+'\n',overwrite_existing=True)
    manifest=path_governance.write_artifact_manifest(out)
    result['artifact_manifest']=manifest['manifest']
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--attempt',type=int,default=1);p.add_argument('--extra-closure-rad',type=float,default=0.);p.add_argument('--hand-kp',type=float,default=HAND_KP_DEFAULT);p.add_argument('--grasp-height-offset-m',type=float,default=0.);p.add_argument('--preload-advance-rad',type=float,default=0.);p.add_argument('--plan-dir',required=True);a=p.parse_args();sys.exit(0 if run(a.run_id,a.attempt,a.plan_dir,a.extra_closure_rad,a.hand_kp,a.grasp_height_offset_m,a.preload_advance_rad)['passed'] else 2)
