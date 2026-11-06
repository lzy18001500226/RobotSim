"""Derived, hash-recorded right-claw joint-resistance diagnostic. No vendor edits."""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import yaml
from scripts.agent import path_governance as pg

JOINTS=('R_claw_joint','R_hand_wide1_joint','R_hand_narrow3_joint','R_hand_narrow_loop_joint','R_hand_wide3_joint','R_hand_wide_loop_joint')
ROOT=Path('/home/lzy18001500226/projects/RobotSim')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def change_friction(xml, *, both_hands=False, right_damping=None, noslip_iterations=0, elliptic_grip_solver=False):
    root=ET.fromstring(xml)
    names=JOINTS+tuple("L_"+n[2:] for n in JOINTS) if both_hands else JOINTS
    for name in names:
        nodes=root.findall('.//body/joint[@name="'+name+'"]')
        if len(nodes)!=1:raise ValueError('expected exactly one right-claw joint:'+name)
        nodes[0].set('frictionloss','0')
    if right_damping is not None:
        if not np.isfinite(right_damping) or not 0.<right_damping<=2.:raise ValueError('right actuator damping outside bounded diagnostic range')
        nodes=root.findall('./actuator/position[@name="motor_R_claw_joint"]')
        if len(nodes)!=1:raise ValueError('right position actuator not unique')
        nodes[0].set('kv',str(float(right_damping)))
    if noslip_iterations:
        if noslip_iterations!=2:raise ValueError('only the bounded two-iteration numerical diagnostic is supported')
        option=root.find('option')
        if option is None:option=ET.SubElement(root,'option')
        option.set('noslip_iterations','2')
    if elliptic_grip_solver:
        option=root.find('option')
        if option is None:option=ET.SubElement(root,'option')
        option.set('cone','elliptic');option.set('impratio','10')
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)

def build(parent, *, both_hands=False, right_damping=None, noslip_iterations=0, elliptic_grip_solver=False):
    parent=Path(parent);record=json.loads(parent.read_text());generated=record['generated']
    if record['classification']!='SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY':raise ValueError('unexpected parent')
    for item in generated.values():
        if isinstance(item,dict) and 'path' in item and digest(item['path'])!=item['sha256']:raise ValueError('parent hash mismatch')
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='claw-zero-internal-friction').native_path
    for name in ('host_scene','runtime_scene'):
        src=Path(generated[name]['path']);dst=out/src.name
        pg.write_new_bytes(dst,change_friction(src.read_bytes(),both_hands=both_hands,right_damping=right_damping,noslip_iterations=noslip_iterations,elliptic_grip_solver=elliptic_grip_solver))
        generated[name]={'path':str(dst),'sha256':digest(dst)}
    original=mujoco.MjModel.from_xml_path(json.loads(parent.read_text())['generated']['host_scene']['path'])
    derived=mujoco.MjModel.from_xml_path(generated['host_scene']['path'])
    allowed={int(derived.jnt_dofadr[mujoco.mj_name2id(derived,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in (JOINTS+tuple("L_"+n[2:] for n in JOINTS) if both_hands else JOINTS)}
    checked=[]
    # Every public numeric model array must be equal except the six loss entries.
    for name in dir(original):
        a=getattr(original,name);b=getattr(derived,name)
        if not isinstance(a,np.ndarray):continue
        if name=='dof_frictionloss':
            expected=a.copy()
            for i in allowed:expected[i]=0.
        elif name=='actuator_biasprm' and right_damping is not None:
            expected=a.copy();act=mujoco.mj_name2id(derived,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_R_claw_joint');expected[act,2]=-right_damping
        else:expected=a
        if not np.array_equal(expected,b,equal_nan=True):raise ValueError('unexpected compiled model delta:'+name)
        checked.append(name)
    config=yaml.safe_load(Path(json.loads(parent.read_text())['generated']['simulator_config']['path']).read_text())
    config['MujocoSimModule']['simulation_model']='/workspace/evidence/'+Path(generated['runtime_scene']['path']).name
    dst=out/'simulator_stl_native_hand_table_control.yaml';pg.write_new_text(dst,yaml.safe_dump(config,sort_keys=False));generated['simulator_config']={'path':str(dst),'sha256':digest(dst)}
    runtime_source=Path(generated['runtime_scene']['path'])
    projection=runtime_source.read_bytes().replace(b'/workspace/raicom2026',str(ROOT/'simulation/mujoco/raicom2026').encode()).replace(b'/workspace/robotsim',str(ROOT).encode())
    projected=out/'runtime_compile_check.xml';pg.write_new_bytes(projected,projection)
    runtime_model=mujoco.MjModel.from_xml_path(str(projected))
    for name in checked:
        if not np.array_equal(getattr(derived,name),getattr(runtime_model,name),equal_nan=True):raise ValueError('host/runtime compiled delta:'+name)
    generated['runtime_hostcheck_scene']={'path':str(projected),'sha256':digest(projected)}
    record['diagnostic_parameter_delta']={'classification':'SIMULATION_ONLY_NOT_HARDWARE_PARAMETERS','parent':str(parent),'parent_sha256':digest(parent),'joint_frictionloss':{'names':list(JOINTS+tuple('L_'+n[2:] for n in JOINTS) if both_hands else JOINTS),'from':.3,'to':0.},'compiled_arrays_checked':checked,'other_model_arrays_identical':True,'force_limits_unchanged':True}
    if right_damping is not None:
        act=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_R_claw_joint')
        record['diagnostic_parameter_delta']['right_actuator_kv']={'from':float(-original.actuator_biasprm[act,2]),'to':float(right_damping),'kp_unchanged':float(original.actuator_gainprm[act,0]),'purpose':'damp measured 171640Z impact/rebound; local coupled inertia estimate ~0.2027 kg m2 with native kp=2 gives critical kv ~1.2735; not hardware identification'}
    if noslip_iterations:
        if derived.opt.noslip_iterations!=2 or runtime_model.opt.noslip_iterations!=2:raise ValueError('numerical solver option mismatch')
        record['diagnostic_parameter_delta']['numerical_solver']={'noslip_iterations_from':int(original.opt.noslip_iterations),'noslip_iterations_to':2,'purpose':'official MuJoCo soft-friction drift diagnostic after 180821Z','official_source':'https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip','geometry_condim_and_contact_friction_unchanged':True,'inverse_dynamics_limitation':'Noslip postprocessing makes inverse dynamics ill-defined; not a vendor-default or hardware qualification'}
    if elliptic_grip_solver:
        if int(derived.opt.cone)!=1 or derived.opt.impratio!=10 or int(runtime_model.opt.cone)!=1 or runtime_model.opt.impratio!=10:raise ValueError('elliptic numerical profile mismatch')
        record['diagnostic_parameter_delta']['elliptic_solver']={'cone_from':int(original.opt.cone),'cone_to':1,'impratio_from':float(original.opt.impratio),'impratio_to':10.,'source':'https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/2f85.xml','physical_friction_condim_force_limits_and_geometry_unchanged':True,'not_vendor_default':True}
    record['generated']=generated
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    return out/'result.json'


def add_reviewed_checkpoint(parent, capture):
    """Add only a native episode-boundary keyframe from the reviewed real pose."""
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
    parent=Path(parent);capture=Path(capture);record=json.loads(parent.read_text())
    rows=[json.loads(line) for line in (capture/'native_physics.jsonl').read_text().splitlines()]
    trace=json.loads((capture/'native_hal_pregrasp_trace.json').read_text())['trace']
    # Begin before arm approach, with the bottle supported and the hand away.
    # Loading the attained, almost-touching pose while MC reinitializes caused
    # a real bottle knock-over in 165524Z. Never hide that failed trial.
    opening=next(r for r in trace if r['stage']=='HAND_OPEN_PREGRASP')
    row=min((r for r in rows if r['generation']==max(x['generation'] for x in rows)),key=lambda r:abs(r['host_monotonic_ns']*1e-9-opening['host_monotonic_s']))
    model=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path']);data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row)
    command=opening['published_target_rad']
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='reviewed-native-checkpoint').native_path
    for key in ('host_scene','runtime_scene'):
        source=Path(record['generated'][key]['path']);xml=ET.fromstring(source.read_bytes())
        if xml.find('keyframe') is not None:raise ValueError('refusing to overwrite existing source keyframes')
        frames=ET.SubElement(xml,'keyframe');ET.SubElement(frames,'key',name='reviewed_ungrasped_arrival',qpos=' '.join(f'{float(v):.17g}' for v in data.qpos),qvel=' '.join('0' for _ in range(model.nv)))
        # Two identical initialization keys keep the native integer slider
        # range nondegenerate; either selection restores the same actual pose.
        duplicate=copy.deepcopy(frames[0]);duplicate.set('name','reviewed_ungrasped_arrival_alias');frames.append(duplicate)
        dest=out/source.name;pg.write_new_bytes(dest,ET.tostring(xml,encoding='utf-8',xml_declaration=True));record['generated'][key]={'path':str(dest),'sha256':digest(dest)}
    derived=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path'])
    checked=[]
    for name in dir(model):
        a=getattr(model,name);b=getattr(derived,name)
        if not isinstance(a,np.ndarray) or name.startswith(('key_','names')) or name in ('_sizes','name_keyadr'):continue
        if not np.array_equal(a,b,equal_nan=True):raise ValueError('checkpoint unexpectedly changed model array:'+name)
        checked.append(name)
    cfg=yaml.safe_load(Path(json.loads(parent.read_text())['generated']['simulator_config']['path']).read_text());cfg['MujocoSimModule']['simulation_model']='/workspace/evidence/'+Path(record['generated']['runtime_scene']['path']).name
    dest=out/'simulator_stl_native_hand_table_control.yaml';pg.write_new_text(dest,yaml.safe_dump(cfg,sort_keys=False));record['generated']['simulator_config']={'path':str(dest),'sha256':digest(dest)}
    record['reviewed_checkpoint']={'classification':'SIMULATION_ONLY_FIXED_STANCE_NO_NAVIGATION','source_run':str(capture),'source_sim_time_s':row['sim_time_s'],'source_generation':row['generation'],'native_capture_sha256':digest(capture/'native_physics.jsonl'),'initial_arm_command_rad':command,'initial_arm_effort_nm':{n:float(row['joints'][n]['bias_force']) if n.startswith('right_') else 0. for n in command},'initial_pelvis_pose':row['bodies']['pelvis'],'initial_bottle_pose':row['bodies']['test_cylinder'],'keyframe_index':0,'equivalent_keyframe_indices':[0,1],'zero_episode_initial_velocities':True,'physical_arrays_unchanged':checked,'initial_state_is_not_grasp_success':True}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);return out/'result.json'
