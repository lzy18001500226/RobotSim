"""RGB-D-directed unloaded stance correction, using existing native MC/bridge."""
from pathlib import Path
import copy,hashlib,inspect,json,math,time
import numpy as np
from scripts.agent import path_governance as pg
ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')

def reference_contract():
    source=ROOT/'raicom-stl-xvfb-native-reset-stand-20261111T004845Z'
    result=json.loads((source/'result.json').read_text())
    if result['gates']['TWO_TABLE_TRANSFER_EXECUTION']!='PASS':raise ValueError('accepted physical stance required')
    snapshot=json.loads((source/'rgbd_grasp_snapshot.json').read_text());stamp=json.loads((source/'standing_waist_initial_hold.json').read_text())['request']['created_host_ns']
    for line in (source/'native_physics.jsonl').open():
        row=json.loads(line)
        if row['host_monotonic_ns']>=stamp:break
    b=row['bodies']['pelvis'];yaw=math.atan2(b['rotation'][3],b['rotation'][0]);dx,dy=np.asarray(snapshot['estimated_center_world_m'][:2])-b['position_m'][:2]
    return dict(schema='robotsim.rgbd-stance-reference.v1',source=str(source),source_result_sha256=hashlib.sha256((source/'result.json').read_bytes()).hexdigest(),perception_snapshot_sha256=hashlib.sha256((source/'rgbd_grasp_snapshot.json').read_bytes()).hexdigest(),object_offset_base_xy_m=[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy],control_object_source='CURRENT_RGBD_ONLY',root_speed_profile='EXISTING_PICKUP_NAVIGATION_ROOT060',task_tolerance_m=.06,max_corrections=3,max_single_excursion_m=.45)

def stance_goal(estimated_center,base,reference):
    p=np.asarray(estimated_center,dtype=float);r=np.asarray(reference['object_offset_base_xy_m'],dtype=float)
    if p.shape!=(3,) or r.shape!=(2,) or not np.isfinite(np.r_[p,r]).all():raise ValueError('finite RGBD pose and calibrated stance required')
    yaw=math.atan2(base['rotation'][3],base['rotation'][0]);c,s=math.cos(yaw),math.sin(yaw)
    goal=p[:2]-np.array([c*r[0]-s*r[1],s*r[0]+c*r[1]])
    if not -1.65<=goal[0]<=1.65 or not -1.75<=goal[1]<=1.75:raise ValueError('stance outside existing task workspace')
    return goal.tolist()

def stance_axis(pose,goal):
    dx,dy=goal[0]-pose['x'],goal[1]-pose['y'];c,s=math.cos(pose['yaw']),math.sin(pose['yaw']);f,l=c*dx+s*dy,-s*dx+c*dy
    return ('forward' if f>=0 else 'backward',abs(f)) if abs(f)>=abs(l) else ('left' if l>=0 else 'right',abs(l))

def stance_velocity(direction,nominal,elapsed,progress,started,cfg):
    """Latch observed initiation across the natural alternating-step sway."""
    if direction not in ('left','right'):return nominal,started
    started=started or progress>=.015
    if not started and elapsed>=cfg['startup_deadline_s']:raise RuntimeError('LATERAL_STARTUP_NO_MEASURED_PROGRESS')
    return ((0.,math.copysign(cfg['lateral_startup_speed'],nominal[1])) if not started else nominal),started

def build_motion_script(goal,limits,bodies,label):
    from scripts.research.issue46.native_runtime import native_base_micro_probe as micro
    if not label.isidentifier():raise ValueError('stance result label')
    code=micro.build_script('left',.6,limits,command_speed=.12,robot_bodies=bodies).rsplit('\nrun_probe()',1)[0]
    route='    if MICRO.get("carry"):\n        return run_carry_navigation(n,contract)'
    if code.count(route)!=1:raise ValueError('existing micro dispatch changed')
    code=code.replace(route,'    if MICRO.get("visual_stance"):\n        return run_visual_stance(n,contract)\n'+route,1)
    code+='\n'+inspect.getsource(stance_axis)+'\n'+inspect.getsource(stance_velocity)+'\n'+inspect.getsource(run_visual_stance)+'\n'
    code+='MICRO.update('+repr({'visual_stance':{'goal_xy_m':list(goal),'label':label,'max_motion_s':3.,'command_speed':.12,'lateral_startup_speed':.60,'startup_deadline_s':.50,'max_excursion_m':.45},'pickup_stance_profile':True,'graph_startup_timeout_s':5.})+')\nrun_probe()\n'
    compile(code,'native_rgbd_stance','exec');return code

def run_visual_stance(n,contract):
    import threading,os
    if os.path.exists('/workspace/evidence/carry_ready.json'):raise RuntimeError('STANCE_MUST_PRECEDE_GRASP')
    cfg=MICRO['visual_stance'];result={'classification':'SIMULATION_ONLY_RGBD_DIRECTED_UNLOADED_NATIVE_STEP','pass_result':False,'goal_xy_m':cfg['goal_xy_m'],'root_speed_limit_m_s':.60,'legacy_root020_pass':'NOT_EVALUATED','commands_are_not_receipt':True};supports=[];origin=None;watchdog=None;baseline=None;last_support_ns=None
    original_sample=n.sample
    def sampled(*a,**k):
        nonlocal last_support_ns
        row=original_sample(*a,**k);native=n.latest_row
        if origin is not None and math.hypot(row['pose']['x']-origin['x'],row['pose']['y']-origin['y'])>cfg['max_excursion_m']:raise RuntimeError('STANCE_EXCURSION_BOUND')
        if native['host_monotonic_ns']!=last_support_ns:
            loads,side=native_foot_loads(native);supports.append({'source_host_ns':native['host_monotonic_ns'],'sim_time_s':native['sim_time_s'],'loads_n':loads,'side':side});last_support_ns=native['host_monotonic_ns']
        return row
    n.sample=sampled
    try:
        n.start_feedback();ready_until=time.monotonic()+5.
        while time.monotonic()<ready_until:
            n.publish();n.pump()
            if not all(k in n.messages for k in ('common','imu','leg_state','leg_command')) or 'get_mode' not in n.readbacks:
                time.sleep(.01);continue
            try:state=n.sample();break
            except RuntimeError as exc:
                if str(exc)!='CLOCK_GENERATION_PENDING' and str(exc)!='NATIVE_FEEDBACK_STALE_OR_INVALID' and not str(exc).startswith('CRITICAL_ROS_TELEMETRY_STALE:'):raise
            time.sleep(.01)
        else:raise RuntimeError('STANCE_FRESH_READINESS_TIMEOUT')
        n.await_graph();baseline=state['body_status'];result['pre_motion_composite']=n.observe(8.,baseline=baseline,require_stop=True);origin=n.sample()['pose'];result['start_pose']=origin
        direction,distance=stance_axis(origin,cfg['goal_xy_m']);result['direction']=direction;result['requested_axis_distance_m']=distance;result['lateral_input_policy']='PERSIST_060_UNTIL_POSITION_BRAKE_OR_3S; ACTUAL_ROOT_CAP_060_UNCHANGED'
        n.stand_once('LOCOMOTION_STEP');n.observe(.2,baseline=baseline)
        with n.lock:n.moving=True
        n.stop_started=False;began=time.monotonic();source_begin=n.latest_row['host_monotonic_ns'];acquired=False;lateral_started=False
        watchdog=threading.Timer(cfg['max_motion_s'],n.deadline_zero);watchdog.daemon=True;watchdog.start()
        velocity={'forward':(cfg['command_speed'],0.),'backward':(-cfg['command_speed'],0.),'left':(0.,cfg['lateral_startup_speed']),'right':(0.,-cfg['lateral_startup_speed'])}[direction]
        while time.monotonic()-began<cfg['max_motion_s'] and n.moving:
            tick=time.monotonic();n.pump();row=n.sample();n.trace.append(row)
            if row['body_status']!=baseline:raise RuntimeError('STANCE_BODY_STATUS_CHANGED')
            seen=row['common_source']['name']=='robotsim_official_cmd_vel' and row['common_mode_desc']=='LOCOMOTION_STEP' and row['get_mode_desc']=='LOCOMOTION_STEP'
            acquired=acquired or seen
            if not acquired and tick-began>.75:raise RuntimeError('STANCE_SOURCE_AND_STEP_NOT_OBSERVED')
            _,_,progress,_=micro_displacement(row['pose'],origin,direction)
            along=max(0.,task_along_speed(n.trace,row['pose'],cfg['goal_xy_m'],row['sample_monotonic_s']))
            if acquired and progress>=.035 and distance-progress<=max(.04,along*.8+.03):result['predicted_stop_trigger']={'progress_m':progress,'measured_speed_sim_m_s':along};break
            command,lateral_started=stance_velocity(direction,velocity,tick-began,progress,lateral_started,cfg)
            if lateral_started and 'lateral_initiation' not in result:result['lateral_initiation']={'progress_m':progress,'elapsed_wall_s':tick-began}
            n.publish(*command,0.);time.sleep(max(0.,tick+.02-time.monotonic()))
        n.deadline_zero();n.stop_started=True
        if watchdog:watchdog.cancel()
        n.observe(1.05,baseline=baseline);n.stand_once();result['post_stop_composite']=n.observe(12.,baseline=baseline,require_stop=True);end=n.sample()['pose'];result['end_pose']=end
        forward,lateral,progress,cross=micro_displacement(end,origin,direction);result.update(actual_axis_progress_m=progress,cross_axis_m=cross,final_goal_error_m=math.hypot(end['x']-cfg['goal_xy_m'][0],end['y']-cfg['goal_xy_m'][1]),source_and_step_observed=acquired)
        sequence=[]
        for item in supports:
            if item['source_host_ns']>=source_begin and item['side'] and (not sequence or sequence[-1]['side']!=item['side']):sequence.append(item)
        alternated={x['side'] for x in sequence}=={'left','right'};result['support_changes']=sequence;result['alternating_support_observed']=alternated
        result['legacy_root020_pass']=max((math.hypot(x['pose']['vx'],x['pose']['vy']) for x in n.trace),default=0.)<=.20
        result['pass_result']=bool(acquired and progress>.005 and alternated and result['post_stop_composite']['pass']);result['failure']=None if result['pass_result'] else 'STANCE_MOTION_OR_SUPPORT_NOT_QUALIFIED'
    except Exception as exc:
        n.deadline_zero();n.stop_started=True;result['failure']=str(exc)
        try:
            n.observe(1.05,baseline=baseline);n.stand_once()
            result['abort_stop_composite']=n.observe(12.,baseline=baseline,require_stop=True)
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
        n.stop_feedback();result.update(trace=n.trace,commands=n.command_log,bridge_commands=n.bridge_commands,support_samples=supports,events=n.events)
        write_durable_milestone('/workspace/evidence/'+cfg['label']+'_result.json',result);n.destroy_node();rclpy.shutdown();print(json.dumps({k:v for k,v in result.items() if k not in ('trace','commands','support_samples','bridge_commands','events')}),flush=True)

def execute(run,container,environment,output,scene,cache,reference,limits,bodies):
    """Host sensor/planner; existing native MC owns every physical base step."""
    import mujoco
    from PIL import Image
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    from scripts.research.issue46.native_runtime.native_grasp_audit import read_fresh_native_row
    if (reference.get('schema')!='robotsim.rgbd-stance-reference.v1' or reference.get('control_object_source')!='CURRENT_RGBD_ONLY'
            or reference.get('root_speed_profile')!='EXISTING_PICKUP_NAVIGATION_ROOT060' or reference.get('max_corrections')!=3
            or reference.get('task_tolerance_m')!=.06 or reference.get('max_single_excursion_m')!=.45):raise ValueError('unreviewed stance contract')
    model,data,renderer,camera=vision.build_sensor(scene,cache);events=[];began=time.monotonic()
    pg.write_new_text(output/'visual_stance_reference.json',json.dumps(reference,indent=2)+'\n')
    def capture(label,after_ns=0):
        row=read_fresh_native_row(output/'native_latest.json')
        if row['host_monotonic_ns']<=after_ns:raise RuntimeError('STANCE_REOBSERVATION_NOT_AFTER_STOP')
        rgb,depth,K,T=vision.render(model,data,renderer,camera,row);est,mask=vision.estimate(rgb,depth,K,T)
        center=[*est['center_xy_world_m'],est['top_world_z_m']-.11];obs=dict(pass_result=True,**{'pass':True},estimated_center_world_m=center,axis_world=est['axis_world'],diameter_m=est['diameter_m'],source_host_ns=row['host_monotonic_ns'],created_host_ns=time.monotonic_ns(),generation=row['generation'],object_truth_used_by_estimator=False)
        vision.qualify_estimate(obs,generation=row['generation']);Image.fromarray(rgb).save(output/(label+'.png'));pg.write_new_text(output/(label+'.json'),json.dumps(obs,indent=2)+'\n');return row,obs
    try:
        # The cold shader/mesh/estimator work is discarded, never timestamped as
        # a control observation. Acquire a NEW source row after warming up.
        warm_began=time.monotonic();warm_row=read_fresh_native_row(output/'native_latest.json')
        rgb,depth,K,T=vision.render(model,data,renderer,camera,warm_row);vision.estimate(rgb,depth,K,T)
        pg.write_new_text(output/'stance_sensor_warmup.json',json.dumps({'discarded':True,'source_host_ns':warm_row['host_monotonic_ns'],'wall_s':time.monotonic()-warm_began,'control_observation':False})+'\n')
        after_ns=warm_row['host_monotonic_ns']
        for index in range(reference['max_corrections']+1):
            row,first=capture('stance_observation_'+str(index)+'a',after_ns);time.sleep(.03);row,second=capture('stance_observation_'+str(index)+'b',first['source_host_ns']);snapshot=vision.make_stationary_snapshot(first,second,generation=row['generation']);goal=stance_goal(snapshot['estimated_center_world_m'],row['bodies']['pelvis'],reference);base=row['bodies']['pelvis']['position_m']
            error=math.hypot(goal[0]-base[0],goal[1]-base[1]);events.append({'index':index,'source_host_ns':row['host_monotonic_ns'],'estimated_object_world_m':snapshot['estimated_center_world_m'],'stance_goal_world_xy_m':goal,'stance_error_m':error})
            if error<=reference['task_tolerance_m']:
                result={'pass_result':True,'classification':'RGBD_BASE_STANCE_STOP_REOBSERVATION_NOT_GRASP_ACCEPTANCE','corrections':index,'events':events,'elapsed_wall_s':time.monotonic()-began,'control_object_source':'RGBD_ONLY','final_observation_after_last_motion':True};pg.write_new_text(output/'visual_stance_result.json',json.dumps(result,indent=2)+'\n');return result
            if index==reference['max_corrections']:raise RuntimeError('RGBD_STANCE_CORRECTION_BUDGET_EXHAUSTED')
            label='visual_stance_'+str(index+1)
            delta=np.asarray(goal)-np.asarray(base[:2]);motion_goal=(np.asarray(base[:2])+delta*min(1.,.30/max(error,1e-9))).tolist()
            from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
            apply_recorded_native_frame(model,data,row);root_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint');qa=int(model.jnt_qposadr[root_joint]);robot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');robots={i for i in range(model.nbody) if model.body_rootid[i]==robot};bad=[]
            for fraction in np.linspace(0.,1.,15):
                data.qpos[qa:qa+2]=np.asarray(base[:2])+fraction*(np.asarray(motion_goal)-base[:2]);mujoco.mj_forward(model,data)
                for c in data.contact:
                    a,b=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                    if c.dist<-.001 and (a in robots)!=(b in robots) and 0 not in (a,b):bad.append({'fraction':float(fraction),'pair':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,k) for k in (a,b)],'distance_m':float(c.dist)})
            pg.write_new_text(output/(label+'_static.json'),json.dumps({'classification':'SAMPLED_STATIC_STANCE_TRANSLATION_NOT_GAIT_OR_CONTINUOUS_PROOF','mj_step_calls':0,'goal_xy_m':motion_goal,'violations':bad},indent=2)+'\n')
            if bad:raise RuntimeError('STANCE_ROUTE_COLLISION:'+str(bad[:2]))
            code=build_motion_script(motion_goal,limits,bodies,label);pg.write_new_text(output/(label+'.py'),code)
            step=run(['docker','exec',container,'bash','-lc',environment+'timeout 55s python3 /workspace/evidence/'+label+'.py'],timeout_s=58.,check=False);pg.write_new_text(output/(label+'.stdout'),step.stdout)
            result=json.loads((output/(label+'_result.json')).read_text())
            if step.returncode or not result.get('pass_result'):raise RuntimeError('RGBD_STANCE_STEP_FAILED:'+str(result.get('failure')))
            events[-1]['motion_result']=label+'_result.json';after_ns=read_fresh_native_row(output/'native_latest.json')['host_monotonic_ns']
    except Exception as exc:
        pg.write_new_text(output/'visual_stance_result.json',json.dumps({'pass_result':False,'failure':str(exc),'events':events,'classification':'RGBD_STANCE_STOPPED_BEFORE_GRASP'},indent=2)+'\n')
        raise
    finally:renderer.close()
