"""Build a read-only native step observer using exact-version MuJoCo headers."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, runtime_library):
    from scripts.agent import path_governance as pg
    output=Path(output);library=Path(runtime_library)
    api=ctypes.CDLL(str(library));api.mj_version.restype=ctypes.c_int
    version=api.mj_version()
    if version!=337:raise RuntimeError('native observer currently qualified only for engine ABI 337')
    # Header discovery is not a dependency install and never falls back to a
    # different ABI. Source package versions/hashes are persisted below.
    matches=[]
    for header in (Path.home()/'.cache/uv/archive-v0').glob('**/mujoco/include/mujoco/mujoco.h'):
        if '#define mjVERSION_HEADER 337' in header.read_text():matches.append(header)
    if not matches:raise RuntimeError('exact MuJoCo 3.3.7 headers absent; no native model will be launched')
    header=sorted(matches)[0];include=header.parents[1]
    source=Path(__file__).with_suffix('.c');target=output/'native_grasp_audit.so'
    if target.exists():raise FileExistsError(target)
    command=['gcc','-O2','-fPIC','-shared','-Wall','-Wextra','-Werror','-I',str(include),str(source),'-o',str(target),'-ldl','-pthread']
    r=subprocess.run(command,text=True,capture_output=True,timeout=20)
    pg.write_new_text(output/'native_audit_build.stdout',r.stdout)
    pg.write_new_text(output/'native_audit_build.stderr',r.stderr)
    if r.returncode:raise RuntimeError('native observer build failed: '+r.stderr)
    pg.write_new_bytes(output/source.name,source.read_bytes())
    headers={p.name:digest(p) for p in header.parent.glob('*.h')}
    record={'classification':'READ_ONLY_NATIVE_MJ_STEP_OBSERVER','engine_api_version':version,'runtime_library':str(library),'runtime_library_sha256':digest(library),'header_path':str(header),'header_sha256':headers,'source_sha256':digest(source),'binary_sha256':digest(target),'command':command,'writes_model_or_state':False,'scope':'post-mj_step snapshot; no extra mj_forward; timestamps and performance recorded; not grasp acceptance'}
    pg.write_new_text(output/'native_audit_provenance.json',json.dumps(record,indent=2)+'\n')
    return target


def launch_prefix():
    """Only the simulator process receives the read-only interposer."""
    return "env LD_PRELOAD=/workspace/evidence/native_grasp_audit.so ROBOTSIM_NATIVE_AUDIT_OUT=/workspace/evidence/native_physics.jsonl "


def require_fresh_sample(path, *, now_ns=None):
    import time
    import math
    path=Path(path)
    latest=path.with_name("native_latest.json")
    raw = (latest if latest.exists() else path).read_bytes()[-1048576:]
    lines = raw.splitlines()
    if not raw.endswith(b"\n"):
        lines = lines[:-1]
    for line in reversed(lines):
        try:
            row = json.loads(line)
            break
        except (ValueError, UnicodeDecodeError):
            continue
    else:
        raise RuntimeError("NATIVE_OBSERVER_NO_COMPLETE_SAMPLE")
    age = ((time.monotonic_ns() if now_ns is None else now_ns) - row["host_monotonic_ns"]) * 1e-9
    if not math.isfinite(age) or not 0.0 <= age <= 0.20:
        raise RuntimeError("NATIVE_OBSERVER_STALE_SAMPLE")
    if row.get("engine_version") != 337 or any(row.get("bodies", {}).get(n) is None for n in ("pelvis", "R_omnipicker_base_link", "test_cylinder")):
        raise RuntimeError("NATIVE_OBSERVER_MISSING_REQUIRED_BODIES")
    return {"pass": True, "age_s": age, "engine_version": row["engine_version"], "sim_time_s": row["sim_time_s"], "generation": row["generation"]}


def planning_state_from_native(row, previous):
    """One coherent actual-engine sample for offline planning; never writes runtime."""
    import copy
    import math
    from scipy.spatial.transform import Rotation
    state = copy.deepcopy(previous)
    body = row["bodies"]["pelvis"]
    q = Rotation.from_matrix([body["rotation"][i:i+3] for i in (0,3,6)]).as_quat()
    state["base_pose_world"] = {"xyz_m": list(body["position_m"]), "quaternion_wxyz": [float(q[3]), *map(float,q[:3])]}
    for group in ("arm_positions_rad", "leg_positions_rad", "waist_positions_rad"):
        for name in state[group]:
            value = float(row["joints"][name]["q"])
            if not math.isfinite(value):
                raise RuntimeError("NONFINITE_NATIVE_PLANNING_STATE:" + name)
            state[group][name] = value
    state["native_snapshot"] = {k:row[k] for k in ("host_monotonic_ns", "sim_time_s", "generation")}
    state["state_source"] = "SYNCHRONIZED_NATIVE_MJDATA_NOT_ROS_CACHE"
    return state


def coupled_opening_budget(model, root_joint, actuator, q):
    """Quasi-static estimate, not proof of solver or hardware feasibility."""
    import mujoco
    import math
    if not math.isfinite(q):
        raise ValueError("nonfinite root position")
    terms=[{"joint": mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,root_joint), "derivative": 1., "frictionloss_nm": float(model.dof_frictionloss[model.jnt_dofadr[root_joint]])}]
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[i] != root_joint:
            continue
        j=int(model.eq_obj1id[i])
        derivative=sum(k*float(model.eq_data[i,k])*q**(k-1) for k in range(1,5))
        terms.append({"joint":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),"derivative":derivative,"frictionloss_nm":float(model.dof_frictionloss[model.jnt_dofadr[j]])})
    resistance=sum(abs(t["derivative"])*t["frictionloss_nm"] for t in terms)
    kp=float(model.actuator_gainprm[actuator,0]);gear=float(model.actuator_gear[actuator,0])
    if kp<=0 or gear!=-1 or abs(float(model.actuator_biasprm[actuator,1])+kp)>1e-9:
        raise ValueError("budget requires the recorded gear=-1 position servo")
    command=-q+resistance/kp
    return {"q_rad":q,"estimated_coupled_coulomb_nm":resistance,"terms":terms,"kp":kp,"opening_command_to_balance_coulomb_estimate":command,"command_upper_bound":float(model.actuator_ctrlrange[actuator,1]),"estimate_within_command_range":command<=float(model.actuator_ctrlrange[actuator,1]),"classification":"QUASISTATIC_ESTIMATE_NOT_PHYSICS_ACCEPTANCE"}


def apply_recorded_native_frame(model, data, row):
    """Review data only: restore captured configuration, never step physics."""
    import mujoco
    import numpy as np
    for name, item in row['joints'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if j<0:raise ValueError('recorded joint absent:'+name)
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE,mujoco.mjtJoint.mjJNT_BALL):raise ValueError('unexpected nonscalar snapshot joint')
        data.qpos[model.jnt_qposadr[j]]=float(item['q'])
        data.qvel[model.jnt_dofadr[j]]=float(item['qvel'])
    for name in ('pelvis','test_cylinder'):
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError('recorded body absent:'+name)
        j=int(model.body_jntadr[body])
        if j<0 or model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('recorded body lacks free joint:'+name)
        pose=row['bodies'][name];qadr=int(model.jnt_qposadr[j])
        data.qpos[qadr:qadr+7]=[*pose['position_m'],*pose['quaternion_wxyz']]
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise ValueError('nonfinite captured frame')
    data.time=float(row['sim_time_s'])
    mujoco.mj_forward(model,data)


def review_native_capture(run, *, dry_run=False):
    import math,time
    import mujoco
    import numpy as np
    run=Path(run);record=json.loads((run/'result.json').read_text())
    overlay_ref=record['claw_control_overlay'];overlay_path=Path(overlay_ref['path'])
    if digest(overlay_path)!=overlay_ref['sha256']:raise ValueError('capture overlay hash mismatch')
    overlay=json.loads(overlay_path.read_text());scene=overlay['generated']['host_scene']
    if digest(scene['path'])!=scene['sha256']:raise ValueError('captured scene hash mismatch')
    model=mujoco.MjModel.from_xml_path(scene['path']);data=mujoco.MjData(model)
    with (run/'native_physics.jsonl').open() as stream:rows=[json.loads(line) for line in stream if line.strip()]
    generation=max(r['generation'] for r in rows);rows=[r for r in rows if r['generation']==generation]
    if len(rows)<2:raise ValueError('capture lacks post-reset frames')
    if any(b['host_monotonic_ns']<=a['host_monotonic_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('nonmonotonic capture')
    replan_ref=record['hal_arm_runtime_replan'];replan_path=Path(replan_ref['path'])
    if digest(replan_path)!=replan_ref['sha256']:raise ValueError('capture replan hash mismatch')
    plan=json.loads(replan_path.read_text());centre=np.asarray(plan['circle']['center_m']);radius=float(plan['circle']['radius_m']);height=float(plan['grasp_height_above_center_m'])
    trace=json.loads((run/'native_hal_pregrasp_trace.json').read_text())['trace']
    close=[r['host_monotonic_s'] for r in trace if r['stage']=='HAND_CLOSE_HOLD']
    if not close:raise ValueError('capture has no executed close phase')
    close_ns=min(close)*1e9
    stages=[min(range(len(rows)),key=lambda i:abs(rows[i]['host_monotonic_ns']-t)) for t in (close_ns-5e7,close_ns+5e8,rows[-1]['host_monotonic_ns'])]
    pose_errors=[]
    for i in stages:
        apply_recorded_native_frame(model,data,rows[i])
        for body in ('pelvis','R_omnipicker_base_link','test_cylinder'):
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
            pose_errors.append(float(np.linalg.norm(data.xpos[bid]-np.asarray(rows[i]['bodies'][body]['position_m']))))
    if max(pose_errors)>.005:raise ValueError('recorded/rendered body pose mismatch exceeds 5mm')
    summary={'classification':'RECORDED_NATIVE_PHYSICS_VISUAL_REVIEW_ONLY','mj_step_calls':0,'docker_or_ros_commands':0,'frames':len(rows),'max_recorded_render_pose_error_m':max(pose_errors),'stage_sim_times':[rows[i]['sim_time_s'] for i in stages],'source_engine_version':rows[-1]['engine_version'],'review_engine_version':mujoco.mj_version(),'scene_sha256':scene['sha256'],'controls':'1=arrival; 2=closing; 3=final; arrows=frame; space=recorded playback; mouse=rotate/pan/zoom','limitation':'Recorded actual poses; displayed forces are not recomputed physical evidence; no new rollout.'}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if dry_run:return summary
    from mujoco import viewer
    state={'index':stages[0],'play':False}
    def key(code):
        if code in (49,50,51):state['index']=stages[code-49];state['play']=False
        elif code==262:state['index']=min(len(rows)-1,state['index']+1);state['play']=False
        elif code==263:state['index']=max(0,state['index']-1);state['play']=False
        elif code==32:state['play']=not state['play']
    with viewer.launch_passive(model,data,key_callback=key,show_left_ui=True,show_right_ui=True) as window:
        window.cam.lookat[:]=[1.05,-1.50,.85];window.cam.distance=1.65;window.cam.azimuth=135.;window.cam.elevation=-20.
        window.opt.geomgroup[3:]=0
        while window.is_running():
            row=rows[state['index']];apply_recorded_native_frame(model,data,row)
            h=row['bodies']['R_omnipicker_base_link'];R=np.array(h['rotation']).reshape(3,3);actual=np.array(h['position_m'])+R@centre
            bottle=row['bodies']['test_cylinder'];B=np.array(bottle['rotation']).reshape(3,3);target=np.array(bottle['position_m'])+height*B[:,2]
            with window.lock():
                window.user_scn.ngeom=0
                for point,color in ((actual,[1.,.8,0.,1.]),(target,[1.,0.,.6,1.])):
                    geom=window.user_scn.geoms[window.user_scn.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004,.004,.004]),point,np.eye(3).reshape(-1),np.array(color,dtype=np.float32));window.user_scn.ngeom+=1
                for j in range(64):
                    points=[]
                    for angle in (2*math.pi*j/64,2*math.pi*(j+1)/64):points.append(actual+R@np.array([0.,radius*math.cos(angle),radius*math.sin(angle)]))
                    geom=window.user_scn.geoms[window.user_scn.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).reshape(-1),np.array([1.,.8,0.,1.],dtype=np.float32));mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0006,points[0],points[1]);window.user_scn.ngeom+=1
            window.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,'RECORDED REVIEW / NO PHYSICS','1 arrival | 2 closing | 3 final | arrows frames | space replay'),(None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,'YELLOW circle / MAGENTA bottle section',f"sim={row['sim_time_s']:.3f}s  centre_error={np.linalg.norm(actual-target)*1000:.2f}mm")])
            window.sync();time.sleep(.02)
            if state['play']:state['index']=min(len(rows)-1,state['index']+1)
    return summary
