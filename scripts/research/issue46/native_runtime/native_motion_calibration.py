"""Bounded unloaded native X2 motion measurement; no controller retuning.

Reuse the existing native bootstrap/Micro transport and strict stop contract.
Diagnostic prototype: the smoothed/gait-phase stop trigger is NOT a qualified
relative-position controller or a fair minimum-resolution benchmark.
Only initial placement/posture, bridge input bounds, and measured motion goals
are task configuration. No active rollout qpos writes or artificial support.
"""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,math,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco,numpy as np
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
SOURCE=ROOT/'raicom-stl-xvfb-native-reset-stand-20261111T004845Z'
PARENT=ROOT/'h220-scene-identity-await-restored-route-20261111T004844Z/result.json'
RESUME_CONTEXT={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):pg.write_new_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def signed_delta(start,end,yaw_trace=None):
    c,s=math.cos(start['yaw']),math.sin(start['yaw']);dx=end['x']-start['x'];dy=end['y']-start['y']
    headings=yaw_trace or [start['yaw'],end['yaw']];angle=sum(math.atan2(math.sin(b-a),math.cos(b-a)) for a,b in zip(headings,headings[1:]))
    return {'forward_m':c*dx+s*dy,'left_m':-s*dx+c*dy,'yaw_rad':angle,'world_dx_m':dx,'world_dy_m':dy}


def projected_progress(rows,start,kind):
    if not rows:return 0.
    end=rows[-1]['sim_time_s'];tail=[x for x in rows if 0.<=end-x['sim_time_s']<=.60]
    if not tail:return 0.
    values=[signed_delta(start,x['pose']) for x in tail]
    if kind.startswith('turn_'):
        sign=1. if kind=='turn_left' else -1.
        return sign*sum(v['yaw_rad'] for v in values)/len(values)
    axis='forward_m' if kind in ('forward','backward') else 'left_m';sign=-1. if kind in ('backward','right') else 1.
    return sign*sum(v[axis] for v in values)/len(values)


def motion_metrics(start,end,rows,kind,target):
    delta=signed_delta(start,end,[start['yaw']]+[r['pose']['yaw'] for r in rows]+[end['yaw']])
    sign=-1. if kind in ('backward','right','turn_right') else 1.
    main=delta['yaw_rad']*sign if kind.startswith('turn_') else delta['forward_m' if kind in ('forward','backward') else 'left_m']*sign
    noise_floor=math.radians(2.) if kind.startswith('turn_') else .005
    return {'actual':delta,'requested_m_or_rad':target,'signed_goal_progress_m_or_rad':main,'goal_error_m_or_rad':main-target,'direction_observed':main>=noise_floor,'peak_planar_speed_m_s':max((math.hypot(r['pose']['vx'],r['pose']['vy']) for r in rows),default=0.),'maximum_planar_excursion_m':max((math.hypot(r['pose']['x']-start['x'],r['pose']['y']-start['y']) for r in rows),default=0.),'not_a_distance_accuracy_PASS':True}


def logger_resume_context(prior,run_path):
    if (prior.get('failure')!='refusing existing milestone:/workspace/evidence/motion_calibration_progress.json'
            or not prior.get('abort_stop_composite',{}).get('pass') or not prior.get('motions') or not prior.get('trace')
            or any(not m.get('post_stop_composite',{}).get('pass') for m in prior['motions'])):
        raise ValueError('only a verified logger-only interruption may resume; active clock faults are locked')
    rows=prior['trace']
    return {'prior_run':str(run_path),'prior_result_sha256':sha(Path(run_path)/'native_micro_probe_result.json'),'consumed_wall_s':rows[-1]['sample_monotonic_s']-rows[0]['sample_monotonic_s'],'consumed_sim_s':rows[-1]['sim_time_s']-rows[0]['sim_time_s'],'prior_motions':[{k:m[k] for k in ('kind','metrics','status')} for m in prior['motions']]}

def repeat_eligible(kind,metrics):
    value=metrics['signed_goal_progress_m_or_rad'];target=metrics['requested_m_or_rad']
    if not metrics['direction_observed'] or not .5*target<=value<=2.*target:return False
    actual=metrics['actual']
    if kind.startswith('turn_'):return math.hypot(actual['world_dx_m'],actual['world_dy_m'])<=.10
    cross=actual['left_m'] if kind in ('forward','backward') else actual['forward_m']
    return abs(cross)<=.05

def run_calibration(n,contract):
    import threading
    cfg=MICRO['motion_calibration'];result={'schema':'robotsim.native-motion-calibration.v1','classification':'SIMULATION_ONLY_UNLOADED_FULL_X2_MEASURED_MOTION_NOT_GRASP','pass_result':False,'nonzero_started':False,'motions':[],'all_source_joint_limits_unchanged':True,'root_speed_limit_m_s':.60,'legacy_micro_root020':'SEPARATE_DIAGNOSTIC_NOT_ACCEPTANCE','precision_gates_used_as_stops':False,'failure':None};baseline=None;watchdog=None;origin=None;source_begin=None
    original=n.sample
    def sample(*a,**k):
        row=original(*a,**k);raw=n.latest_row
        if origin is not None and (math.hypot(row['pose']['x']-origin['x'],row['pose']['y']-origin['y'])>.50 or max(abs(row['pose']['x']),abs(row['pose']['y']))>.50):raise RuntimeError('CALIBRATION_GLOBAL_WORKSPACE')
        if source_begin is not None and row['sim_time_s']-source_begin+cfg.get('resume',{}).get('consumed_sim_s',0.)>160.:raise RuntimeError('CALIBRATION_SIM_TIME_BUDGET')
        for name,cap in cfg['actuator_caps_nm'].items():
            value=raw['actuators'].get(name)
            if value is None:raise RuntimeError('ACTUATOR_TELEMETRY_MISSING:'+name)
            if not math.isfinite(value['force']) or abs(value['force'])>cap+1e-5:raise RuntimeError('SOURCE_ACTUATOR_EFFORT:'+name)
        return row
    n.sample=sample
    try:
        n.start_feedback();until=time.monotonic()+6.
        while time.monotonic()<until:
            n.publish();n.pump()
            if all(k in n.messages for k in ('common','imu','leg_state','leg_command')) and 'get_mode' in n.readbacks:
                try:initial=n.sample();break
                except RuntimeError as exc:
                    if not (str(exc) in ('CLOCK_GENERATION_PENDING','NATIVE_FEEDBACK_STALE_OR_INVALID') or str(exc).startswith('CRITICAL_ROS_TELEMETRY_STALE:')):raise
            time.sleep(.01)
        else:raise RuntimeError('CALIBRATION_READINESS_TIMEOUT')
        n.await_graph();baseline=initial['body_status'];result['pre_motion_composite']=n.observe(8.,baseline=baseline,require_stop=True)
        origin=n.sample()['pose'];source_begin=n.sample()['sim_time_s'];resume=cfg.get('resume',{});campaign_end=time.monotonic()+140.-resume.get('consumed_wall_s',0.);eligible=[m['kind'] for m in resume.get('prior_motions',[]) if repeat_eligible(m['kind'],m['metrics'])];done={m['kind'] for m in resume.get('prior_motions',[])}
        sequence=[(name,0) for name in ('forward','backward','turn_left','turn_right','left','right') if name not in done];initial_count=len(sequence);result['resume']=resume
        index=0
        while index<len(sequence):
            kind,repeat=sequence[index];index+=1
            if time.monotonic()>campaign_end:raise RuntimeError('CALIBRATION_TOTAL_WALL_BUDGET')
            if index>initial_count and repeat==1 and kind not in eligible:continue
            mode='LOCOMOTION_DEFAULT' if kind.startswith('turn_') else 'LOCOMOTION_STEP';cfg['active_mode']=mode
            start=n.sample()['pose'];begin_index=len(n.trace);cmd_index=len(n.command_log);target=math.radians(15.) if kind.startswith('turn_') else .05
            item={'kind':kind,'repeat':repeat,'requested_m_or_rad':target,'start_pose':start,'start_host_ns':time.monotonic_ns(),'mode_requested':mode,'command_limits':{'forward_m_s':.20,'lateral_m_s':.60,'angular_rad_s':.60},'root_speed_cap_m_s':.60,'status':'RUNNING'}
            result['motions'].append(item);n.stand_once(mode);n.observe(.2,baseline=baseline)
            with n.lock:n.moving=True
            n.stop_started=False;began=time.monotonic();first_sim=n.latest_row['sim_time_s'];local=[];acquired=False;support=set()
            maximum=3.5 if kind.startswith('turn_') else 2.5
            watchdog=threading.Timer(maximum,n.deadline_zero);watchdog.daemon=True;watchdog.start()
            while n.moving and time.monotonic()-began<maximum:
                tick=time.monotonic();n.pump();row=n.sample();n.trace.append(row);local.append(row)
                if row['body_status']!=baseline:raise RuntimeError('BODY_STATUS_CHANGED')
                # Per-action drift is measured; the preflighted global area remains hard-bounded.
                seen=row['common_source']['name']=='robotsim_official_cmd_vel' and row['common_mode_desc']==mode and row['get_mode_desc']==mode
                acquired=acquired or seen
                if not acquired and tick-began>1.2:raise RuntimeError('CALIBRATION_SOURCE_MODE_NOT_OBSERVED')
                loads,side=native_foot_loads(n.latest_row)
                if side:support.add(side)
                if acquired and row['sim_time_s']-first_sim>=1. and support=={'left','right'} and projected_progress(local,start,kind)>=target:
                    item['target_trigger']={'wall_s':tick-began,'sim_elapsed_s':row['sim_time_s']-first_sim,'filtered_progress':projected_progress(local,start,kind)};break
                scale=bounded_start_scale(tick-began,2. if kind.startswith('turn_') else .5)
                vector={'forward':(.20,0.,0.),'backward':(-.20,0.,0.),'left':(0.,.60,0.),'right':(0.,-.60,0.),'turn_left':(0.,0.,.60),'turn_right':(0.,0.,-.60)}[kind]
                vector=tuple(v*scale for v in vector);bounds=row['raw_common']['speed_status']
                if any(not bounds[k+'_bounds']['min_value']<=v<=bounds[k+'_bounds']['max_value'] for k,v in zip(('forward','lateral','angular'),vector)):raise RuntimeError('MC_REPORTED_COMMAND_BOUNDS')
                n.publish(*vector)
                if any(abs(v)>0. for v in vector):result['nonzero_started']=True
                time.sleep(max(0.,tick+.02-time.monotonic()))
            n.deadline_zero();n.stop_started=True;watchdog.cancel();watchdog=None;item['zero_host_ns']=time.monotonic_ns()
            n.observe(1.05,baseline=baseline);n.stand_once();stop=n.observe(12.,baseline=baseline,require_stop=True);end=n.sample()['pose'];rows=n.trace[begin_index:];metrics=motion_metrics(start,end,rows,kind,target)
            item.update(end_pose=end,end_host_ns=time.monotonic_ns(),post_stop_composite=stop,metrics=metrics,source_and_mode_observed=acquired,alternating_support_observed=support=={'left','right'},trace_indices=[begin_index,len(n.trace)],command_indices=[cmd_index,len(n.command_log)],status='MEASURED_RESPONSE' if metrics['direction_observed'] and acquired else 'DIRECTION_OR_RESPONSE_NOT_QUALIFIED')
            item['repeat_eligible']=repeat_eligible(kind,metrics)
            if repeat==0 and item['status']=='MEASURED_RESPONSE' and item['repeat_eligible']:eligible.append(kind)
            result['post_stop_composite']=stop
            write_durable_milestone('/workspace/evidence/motion_calibration_'+str(len(result['motions'])).zfill(2)+'_'+kind+'_r'+str(repeat)+'.json',{'completed':len(result['motions']),'last':item,'failure':None})
            if index==initial_count:
                sequence.extend([(name,r) for r in (1,2) for name in eligible])
        result['post_stop_composite']=stop;result['measurement_completed']=True;result['pass_result']=all(x['status']=='MEASURED_RESPONSE' for x in result['motions']);result['meaning_of_pass']='ALL_TESTED_DIRECTION_RESPONSES_OBSERVED_NOT_DISTANCE_ACCURACY'
        if not result['pass_result']:result['failure']='ONE_OR_MORE_DIRECTIONS_NOT_QUALIFIED'
    except Exception as exc:
        n.deadline_zero();n.stop_started=True;result['failure']=str(exc)
        if result['motions'] and result['motions'][-1]['status']=='RUNNING':result['motions'][-1].update(status='ABORTED',failure=str(exc),end_host_ns=time.monotonic_ns())
        try:
            n.stand_once();result['abort_stop_composite']=n.observe(12.,baseline=baseline,require_stop=True)
        except Exception as stop_exc:result['abort_stop_failure']=str(stop_exc)
    finally:
        n.deadline_zero()
        if watchdog:watchdog.cancel()
        end=time.monotonic()+1.05
        while time.monotonic()<end:
            n.publish()
            try:n.pump()
            except Exception:pass
            time.sleep(.015)
        n.stop_feedback();result.update(trace=n.trace,command_log=n.command_log,bridge_commands=n.bridge_commands,events=n.events)
        write_durable_milestone('/workspace/evidence/native_micro_probe_result.json',result)
        n.destroy_node();rclpy.shutdown();print(json.dumps({k:v for k,v in result.items() if k not in ('trace','command_log','bridge_commands','events')}),flush=True)


def build_script(limits,bodies,caps,resume=None):
    from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
    code=micro.build_script('forward',.6,limits,command_speed=.20,robot_bodies=bodies).rsplit('\nrun_probe()',1)[0]
    marker='    if MICRO.get("carry"):\n        return run_carry_navigation(n,contract)'
    if code.count(marker)!=1:raise ValueError('Micro dispatch changed')
    code=code.replace(marker,'    if MICRO.get("motion_calibration"):\n        return run_calibration(n,contract)\n'+marker,1)
    code=code.replace('initial_zero_only=bool(', 'initial_zero_only=bool(MICRO.get("motion_calibration") or ')
    code=code.replace('                fault=physical_fault(', '                if MICRO.get("motion_calibration"):limit=.60\n                fault=physical_fault(',1)
    mode_line=next(l for l in code.splitlines() if l.strip().startswith('if mode not in '));mode_part,rest=mode_line.split(' or status not in ',1);old_modes=mode_part.split('if mode not in ',1)[1]
    code=code.replace(mode_line, '                if mode not in (("STAND_DEFAULT",MICRO["motion_calibration"].get("active_mode","LOCOMOTION_STEP")) if MICRO.get("motion_calibration") else '+old_modes+') or status not in '+rest,1)
    action_line=next(l for l in code.splitlines() if l.strip().startswith('if requested!="STAND_DEFAULT"'))
    code=code.replace(action_line, '            if MICRO.get("motion_calibration") and requested not in ("STAND_DEFAULT","LOCOMOTION_STEP","LOCOMOTION_DEFAULT"):raise RuntimeError("CALIBRATION_MODE_ALLOWLIST")\n'+action_line.replace('if requested!=', 'if not MICRO.get("motion_calibration") and requested!='),1)
    code+='\n'+inspect.getsource(signed_delta)+'\n'+inspect.getsource(projected_progress)+'\n'+inspect.getsource(motion_metrics)+'\n'+inspect.getsource(repeat_eligible)+'\n'+inspect.getsource(run_calibration)+'\n'
    code+='MICRO.update('+repr({'graph_startup_timeout_s':5.,'motion_calibration':{'actuator_caps_nm':caps,'resume':resume or {}}})+')\nrun_probe()\n'
    compile(code,'native_motion_calibration','exec');return code


def prepare(out):
    from scripts.research.issue46.native_runtime.xvfb_native_reset_stand import native_leg_safety_contract
    record=copy.deepcopy(json.loads(PARENT.read_text()));model=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path']);root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');qa=int(model.jnt_qposadr[root]);left=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'left_shoulder_roll_joint');la=int(model.jnt_qposadr[left]);limits,provenance=native_leg_safety_contract(include_upper_body=True)
    # Existing source-range carry park avoids the known unloaded left-roll edge.
    park=float(limits['left_shoulder_roll_joint'][0]+.15)
    if abs(park-record['reviewed_checkpoint']['initial_arm_command_rad']['left_shoulder_roll_joint'])>.20:raise ValueError('park exceeds existing posture change bound')
    for label in ('host_scene','runtime_scene','runtime_hostcheck_scene'):
        if label not in record['generated']:continue
        item=record['generated'][label];path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('frozen scene changed')
        xml=ET.fromstring(path.read_bytes())
        body=xml.find(".//body[@name='pelvis']");position=np.array(list(map(float,body.get('pos').split())));position[:2]=0.;body.set('pos',' '.join(map(str,position)))
        for key in xml.findall('keyframe/key'):
            q=np.array(list(map(float,key.get('qpos').split())));q[qa:qa+2]=[0.,0.];q[la]=park;key.set('qpos',' '.join(f'{v:.17g}' for v in q))
        target=out/path.name;content=ET.tostring(xml,encoding='utf-8',xml_declaration=True)
        if target.exists():
            if target.read_bytes()!=content:raise ValueError('existing initialization snapshot differs')
        else:pg.write_new_bytes(target,content)
        record['generated'][label]={'path':str(target),'sha256':sha(target)}
    cp=record['reviewed_checkpoint'];old=np.array(cp['initial_pelvis_pose']['position_m']);delta=-old[:2]
    for k in ('position_m','com_m'):
        if k in cp['initial_pelvis_pose']:cp['initial_pelvis_pose'][k][:2]=(np.array(cp['initial_pelvis_pose'][k][:2])+delta).tolist()
    cp['initial_arm_command_rad']['left_shoulder_roll_joint']=park;cp['classification']='SIMULATION_ONLY_CALIBRATION_EPISODE_INITIAL_PLACEMENT_ACTUAL_STABILITY_REQUIRED'
    record['calibration_initialization']={'parent':str(PARENT),'parent_sha256':sha(PARENT),'only_initial_root_xy_and_keyframe_existing_left_park_changed':True,'left_roll_rad':park,'no_active_rollout_qpos_writes':True}
    current=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path']);checked=[];roundoff={}
    for name in dir(model):
        value=getattr(model,name)
        if not isinstance(value,np.ndarray):continue
        expected=value.copy()
        if name=='key_qpos':expected[:,qa:qa+2]=0.;expected[:,la]=park
        elif name in ('qpos0','qpos_spring'):expected[qa:qa+2]=0.
        elif name=='body_pos':expected[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis'),:2]=0.
        if not np.array_equal(expected,getattr(current,name),equal_nan=True):
            if name in ('actuator_acc0','body_invweight0','dof_M0','dof_invweight0','light_poscom0') and np.allclose(expected,getattr(current,name),rtol=0.,atol=1e-12):roundoff[name]=float(np.max(np.abs(expected-getattr(current,name))))
            else:raise ValueError('non-initialization model change:'+name)
        checked.append(name)
    data=mujoco.MjData(current);mujoco.mj_resetDataKeyframe(current,data,0);mujoco.mj_forward(current,data);robot=mujoco.mj_name2id(current,mujoco.mjtObj.mjOBJ_BODY,'pelvis');bodies=[mujoco.mj_id2name(current,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(current.nbody) if current.body_rootid[i]==robot]
    violations=[]
    for x,y,yaw in [(x,y,a) for x in (-.5,0.,.5) for y in (-.5,0.,.5) for a in np.linspace(-np.pi,np.pi,13)]:
        mujoco.mj_resetDataKeyframe(current,data,0);data.qpos[qa:qa+2]=[x,y];q=np.empty(4);mujoco.mju_axisAngle2Quat(q,np.array([0.,0.,1.]),yaw);oldq=data.qpos[qa+3:qa+7].copy();mujoco.mju_mulQuat(data.qpos[qa+3:qa+7],q,oldq);mujoco.mj_forward(current,data)
        for c in data.contact:
            a,b=[mujoco.mj_id2name(current,mujoco.mjtObj.mjOBJ_BODY,int(current.geom_bodyid[g])) for g in (c.geom1,c.geom2)]
            if (a in bodies or b in bodies) and 'world' not in (a,b) and c.dist<-.001:violations.append({'xy':[x,y],'yaw':float(yaw),'bodies':[a,b],'distance_m':float(c.dist)})
    if violations:raise ValueError('calibration-area static obstacle:'+str(violations[:2]))
    urdf={j.attrib['name']:j for j in ET.parse(provenance['urdf']).getroot().findall('joint')};caps={}
    for i in range(current.nu):
        if current.actuator_trntype[i]!=mujoco.mjtTrn.mjTRN_JOINT:continue
        j=int(current.actuator_trnid[i,0]);name=mujoco.mj_id2name(current,mujoco.mjtObj.mjOBJ_JOINT,j);value=None
        if current.actuator_forcelimited[i]:value=float(np.max(np.abs(current.actuator_forcerange[i])))
        if name in urdf and urdf[name].find('limit') is not None:
            effort=float(urdf[name].find('limit').get('effort','inf'));value=min(effort,value) if value is not None else effort
        if value is not None and math.isfinite(value):caps[mujoco.mj_id2name(current,mujoco.mjtObj.mjOBJ_ACTUATOR,i)]=value
    write(out/'calibration_overlay.json',record);candidate=json.loads((SOURCE/'result.json').read_text())['hal_arm_candidate'];candidate=copy.deepcopy(json.loads(Path(candidate['path']).read_text()));candidate['scene']=record['generated']['host_scene'];write(out/'candidate.json',candidate)
    code=build_script(limits,bodies,caps,RESUME_CONTEXT);pg.write_new_text(out/'generated_calibration.py',code)
    write(out/'preflight.json',{'classification':'INITIAL_PLACEMENT_AND_SAMPLED_STATIC_AREA_ONLY_NOT_DYNAMICS','checked_model_arrays':checked,'translation_only_compiler_roundoff':roundoff,'scene':record['generated']['host_scene'],'xy_bounds_m':[-.5,.5,-.5,.5],'yaw_samples':13,'position_samples':9,'violations':violations,'fixed_left_roll_rad':park,'root_speed_cap_m_s':.60,'source_motion_limits':limits,'source_provenance':provenance,'actuator_caps_nm':caps,'physics_calls':0})
    return out/'calibration_overlay.json',limits,bodies,caps


def execute(out,overlay,limits,bodies,caps):
    from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as parent
    if subprocess.check_output(['docker','ps','-q'],text=True).strip():raise RuntimeError('another container is active; calibration not launched')
    record=json.loads(overlay.read_text());cp=record['reviewed_checkpoint']
    config={'schema':'robotsim.unloaded-motion-calibration-input.v1','scene':record['generated']['host_scene'],'fixed_arm_targets_rad':cp['initial_arm_command_rad'],'fixed_arm_effort_nm':cp['initial_arm_effort_nm'],'actuator_caps_nm':caps,'resume':RESUME_CONTEXT,'source_motion_limits':limits,'maximum_actions':18,'source_sim_budget_s':160,'motion_wall_budget_s':140}
    expected=json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if (out/'motion_input.json').exists():
        if (out/'motion_input.json').read_text()!=expected:raise ValueError('prepared motion input differs')
    else:write(out/'motion_input.json',config)
    baseline=json.loads((SOURCE/'invocation.json').read_text())['argv'];opts={};key=None
    for token in baseline[1:]:
        if token.startswith('--'):key=token;opts[key]=[]
        else:opts[key].append(token)
    argv=['native-motion-calibration']
    for key in ['--table-overlay-result','--table-overlay-result-sha256','--reset-helper','--reset-helper-sha256','--xvfb-root','--xvfb-libs','--xkb-root']:argv.extend([key,*opts[key]])
    argv.extend(['--claw-control-overlay-result',str(overlay),'--claw-control-overlay-result-sha256',sha(overlay),'--motion-calibration-input',str(out/'motion_input.json'),'--motion-calibration-input-sha256',sha(out/'motion_input.json'),'--disable-mc-arm-publisher','--disable-mc-hand-publisher','--use-official-cmd-vel-bridge','--reviewed-foundationwalk-profile','--native-dock-truth-feedback','--native-reset-input','backspace','--initial-hands-fully-open','--micro-base-probe','forward','--micro-command-speed','0.20'])

    sys.argv=argv;original_allocator=pg.create_unique_output_directory
    class ParentConfigValidated(Exception):pass
    def preflight_allocator(*a,**kwargs):
        if kwargs.get('prefix')=='raicom-stl-xvfb-native-reset-stand':raise ParentConfigValidated()
        return original_allocator(*a,**kwargs)
    pg.create_unique_output_directory=preflight_allocator
    try:parent.main()
    except ParentConfigValidated:write(out/'parent_config_validation.json',{'passed':True,'stopped_before_runtime_allocation':True,'physics_started':False})
    finally:pg.create_unique_output_directory=original_allocator
    write(out/'validated_invocation.json',{'command':sys.argv,'parent_argv':argv,'host_script_sha256':sha(__file__),'parent_sha256':sha(parent.__file__),'scope':'ISOLATED_EMPTY_HAND_NO_GRASP_NO_ACTIVE_STATE_WRITES','maximum_motion_actions':18})
    write(out/'runtime_started.json',{'runtime_may_start':True,'time_monotonic_ns':time.monotonic_ns()})
    parent.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--resume-after-log-error',type=Path);args=parser.parse_args()
    if args.resume_after_log_error:
        prior=json.loads((args.resume_after_log_error/'native_micro_probe_result.json').read_text())
        RESUME_CONTEXT=logger_resume_context(prior,args.resume_after_log_error)
    if (args.output_dir/'runtime_started.json').exists():raise RuntimeError('this immutable experiment already executed; no physics retry')
    if (args.output_dir/'preflight.json').exists():
        pre=json.loads((args.output_dir/'preflight.json').read_text());overlay=args.output_dir/'calibration_overlay.json'
        if sha(pre['scene']['path'])!=pre['scene']['sha256'] or pre['violations'] or pre['physics_calls']!=0:raise ValueError('prepared preflight changed')
        model=mujoco.MjModel.from_xml_path(pre['scene']['path']);root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody) if model.body_rootid[i]==root];limits=pre['source_motion_limits'];caps=pre['actuator_caps_nm']
        expected=build_script(limits,bodies,caps,RESUME_CONTEXT)
        if (args.output_dir/'generated_calibration.py').read_text()!=expected:raise ValueError('prepared calibration program changed')
    else:overlay,limits,bodies,caps=prepare(args.output_dir)
    print(json.dumps({'prepared':str(args.output_dir),'execute':args.execute}),flush=True)
    if args.execute:execute(args.output_dir,overlay,limits,bodies,caps)
