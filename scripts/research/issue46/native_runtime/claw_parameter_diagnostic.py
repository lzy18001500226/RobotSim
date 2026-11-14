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


def reuse_source_pad_proxy(parent):
    """Reuse existing hash-locked source-STL CoACD parts, two pad bodies only.

    This is the previously requested, separate source-derived collision
    comparison. It never regenerates meshes or promotes the variant. Physical
    mass/joint/actuator/equality arrays and every visual/exclusion are checked.
    """
    parent=Path(parent);record=json.loads(parent.read_text());original_record=copy.deepcopy(record)
    manifest=ROOT/'simulation/mujoco/models/agibot_x2/omnipicker_sim_functional_v0_1/collision_proxy_manifest.json'
    if digest(manifest)!='7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a':raise ValueError('locked source proxy manifest differs')
    spec=json.loads(manifest.read_text());parts_root=ROOT/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/collision_parts'
    selected={Path(x['source_mesh']).stem:x for x in spec['mesh_decompositions'] if Path(x['source_mesh']).stem in ('narrow3_Link','wide3_Link')}
    if len(selected)!=2 or spec['source_repo_commit']!='575cc6b988f976c23550e0db85aa1e5475d3652d':raise ValueError('proxy source pin mismatch')
    files=[]
    for item in selected.values():
        source=ROOT/'reference/AgibotTech/agibot_x2_urdf'/item['source_mesh']
        if digest(source)!=item['source_sha256']:raise ValueError('original source STL differs')
        for part in item['parts']:
            path=parts_root/part['filename']
            if digest(path)!=part['sha256']:raise ValueError('source proxy part bytes differ:'+str(path))
            files.append({'path':str(path),'sha256':part['sha256']})
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='native-source-pad-coacd-comparison').native_path
    def fingerprint(root):
        return (sorted((x.get('name') or '',tuple(sorted(x.attrib.items()))) for x in root.findall('.//geom') if x.get('contype')=='0' and x.get('conaffinity')=='0'),ET.tostring(root.find('contact')))
    for label in ('host_scene','runtime_scene'):
        source=Path(record['generated'][label]['path']);root=ET.fromstring(source.read_bytes());before=fingerprint(root);asset=root.find('asset')
        for stem,item in selected.items():
            body_name='R_hand_'+stem;body=root.find('.//body[@name="'+body_name+'"]')
            geom=body.find('./geom[@name="'+body_name+'__collision_0"]') if body is not None else None
            if geom is None:raise ValueError('raw source pad collider not unique:'+body_name)
            index=list(body).index(geom);body.remove(geom)
            for number,part in enumerate(item['parts']):
                mesh_name='source_pad_coacd_'+part['name'];path=parts_root/part['filename']
                if label=='runtime_scene':path=Path('/workspace/robotsim')/path.relative_to(ROOT)
                ET.SubElement(asset,'mesh',name=mesh_name,file=str(path))
                replacement=copy.deepcopy(geom);replacement.set('name',body_name+'__collision_coacd_'+str(number));replacement.set('mesh',mesh_name);body.insert(index+number,replacement)
        if fingerprint(root)!=before:raise ValueError('source proxy comparison changed visual geometry or exclusions')
        destination=out/source.name;pg.write_new_bytes(destination,ET.tostring(root,encoding='utf-8',xml_declaration=True));record['generated'][label]={'path':str(destination),'sha256':digest(destination)}
    original=mujoco.MjModel.from_xml_path(original_record['generated']['host_scene']['path']);derived=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path'])
    physical=('body_parentid','body_pos','body_quat','body_ipos','body_iquat','body_mass','body_inertia','body_gravcomp','body_jntadr','body_jntnum','jnt_type','jnt_qposadr','jnt_dofadr','jnt_bodyid','jnt_limited','jnt_range','jnt_pos','jnt_axis','jnt_stiffness','dof_bodyid','dof_jntid','dof_parentid','dof_Madr','dof_solref','dof_solimp','dof_frictionloss','dof_armature','dof_damping','eq_type','eq_obj1id','eq_obj2id','eq_data','eq_solref','eq_solimp','eq_active0','actuator_trntype','actuator_trnid','actuator_gear','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_dynprm','qpos0')
    for field in physical:
        if not np.array_equal(getattr(original,field),getattr(derived,field),equal_nan=True):raise ValueError('proxy comparison changed physical array:'+field)
    cfg=yaml.safe_load(Path(original_record['generated']['simulator_config']['path']).read_text());cfg['MujocoSimModule']['simulation_model']='/workspace/evidence/'+Path(record['generated']['runtime_scene']['path']).name
    destination=out/'simulator_stl_native_hand_table_control.yaml';pg.write_new_text(destination,yaml.safe_dump(cfg,sort_keys=False));record['generated']['simulator_config']={'path':str(destination),'sha256':digest(destination)}
    record['source_pad_collision_comparison']={'classification':'SIMULATION_ONLY_SOURCE_STL_COACD_COMPARISON_NOT_VENDOR_COLLISION','manifest':str(manifest),'manifest_sha256':digest(manifest),'parent':str(parent),'parent_sha256':digest(parent),'scope':['R_hand_narrow3_Link','R_hand_wide3_Link'],'part_count':len(files),'locked_parts':files,'physical_arrays_unchanged':list(physical),'visuals_and_contact_exclusions_unchanged':True,'original_condim_friction_and_force_caps_unchanged':True,'historical_trials_are_not_current_grasp_acceptance':True}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);return out/'result.json'
def change_friction(xml, *, both_hands=False, right_damping=None, right_kp=None, noslip_iterations=0, elliptic_grip_solver=False):
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
    if right_kp is not None:
        if not np.isfinite(right_kp) or float(right_kp) not in (2.,4.,6.):raise ValueError('only source2 or explicit SIMULATION_ONLY4/6 right claw kp is supported')
        nodes=root.findall('./actuator/position[@name="motor_R_claw_joint"]')
        if len(nodes)!=1:raise ValueError('right position actuator not unique')
        nodes[0].set('kp',str(float(right_kp)))
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

def build(parent, *, both_hands=False, right_damping=None, right_kp=None, noslip_iterations=0, elliptic_grip_solver=False):
    parent=Path(parent);record=json.loads(parent.read_text());generated=record['generated']
    if record['classification']!='SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY':raise ValueError('unexpected parent')
    for item in generated.values():
        if isinstance(item,dict) and 'path' in item and digest(item['path'])!=item['sha256']:raise ValueError('parent hash mismatch')
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='claw-zero-internal-friction').native_path
    for name in ('host_scene','runtime_scene'):
        src=Path(generated[name]['path']);dst=out/src.name
        pg.write_new_bytes(dst,change_friction(src.read_bytes(),both_hands=both_hands,right_damping=right_damping,right_kp=right_kp,noslip_iterations=noslip_iterations,elliptic_grip_solver=elliptic_grip_solver))
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
        elif name in ('actuator_gainprm','actuator_biasprm') and (right_damping is not None or right_kp is not None):
            expected=a.copy();act=mujoco.mj_name2id(derived,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_R_claw_joint')
            if right_kp is not None:expected[act,0 if name=='actuator_gainprm' else 1]=right_kp if name=='actuator_gainprm' else -right_kp
            if right_damping is not None and name=='actuator_biasprm':expected[act,2]=-right_damping
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
    if right_kp is not None:
        act=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_R_claw_joint')
        if 'right_actuator_kv' in record['diagnostic_parameter_delta']:
            record['diagnostic_parameter_delta']['right_actuator_kv'].pop('kp_unchanged',None)
            record['diagnostic_parameter_delta']['right_actuator_kv']['kp_after_explicit_delta']=float(right_kp)
        record['diagnostic_parameter_delta']['right_actuator_kp']={'from':float(original.actuator_gainprm[act,0]),'to':float(right_kp),'force_cap_nm':derived.actuator_forcerange[act].tolist(),'kv_unchanged_by_kp':float(-derived.actuator_biasprm[act,2]),'requires_native_no_contact_cycle':True,'purpose':'092646Z ctrl0 saturated at ~.76Nm with retained2Nm cap; compare proportional authority, not limit or contact modification'}
    if noslip_iterations:
        if derived.opt.noslip_iterations!=2 or runtime_model.opt.noslip_iterations!=2:raise ValueError('numerical solver option mismatch')
        record['diagnostic_parameter_delta']['numerical_solver']={'noslip_iterations_from':int(original.opt.noslip_iterations),'noslip_iterations_to':2,'purpose':'official MuJoCo soft-friction drift diagnostic after 180821Z','official_source':'https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip','geometry_condim_and_contact_friction_unchanged':True,'inverse_dynamics_limitation':'Noslip postprocessing makes inverse dynamics ill-defined; not a vendor-default or hardware qualification'}
    if elliptic_grip_solver:
        if int(derived.opt.cone)!=1 or derived.opt.impratio!=10 or int(runtime_model.opt.cone)!=1 or runtime_model.opt.impratio!=10:raise ValueError('elliptic numerical profile mismatch')
        record['diagnostic_parameter_delta']['elliptic_solver']={'cone_from':int(original.opt.cone),'cone_to':1,'impratio_from':float(original.opt.impratio),'impratio_to':10.,'source':'https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/2f85.xml','physical_friction_condim_force_limits_and_geometry_unchanged':True,'not_vendor_default':True}
    record['generated']=generated
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    return out/'result.json'



def add_approved_pad_friction_reference(parent):
    """2026-11-08 case-approved local numerical comparison; no source edits."""
    parent=Path(parent);record=json.loads(parent.read_text());before=copy.deepcopy(record)
    if record.get('classification')!='SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY' or record.get('source_pad_collision_comparison',{}).get('part_count')!=22:raise ValueError('approved source22 parent not established')
    for label in ('host_scene','runtime_scene','simulator_config'):
        item=record['generated'][label]
        if digest(item['path'])!=item['sha256']:raise ValueError('local friction parent hash mismatch')
    model=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path'])
    if model.npair or model.opt.noslip_iterations!=0 or int(model.opt.cone)!=1 or model.opt.impratio!=10. or abs(model.opt.timestep-.001)>1e-12:
        raise ValueError('approved local friction comparison requires no preexisting pairs, native1ms, NoSlip0, elliptic10')
    bottle=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'test_cylinder_collision')
    if bottle<0:raise ValueError('bottle collider missing')
    pairs=[]
    for body,count in (('R_hand_narrow3_Link',12),('R_hand_wide3_Link',10)):
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
        selected=[g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and model.geom_contype[g]]
        if len(selected)!=count:raise ValueError('approved22 source pad pieces not present')
        for g in selected:
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
            if not name.startswith(body+'__collision_coacd_'):raise ValueError('unexpected pad collider identity')
            if not ((model.geom_contype[g]&model.geom_conaffinity[bottle]) or (model.geom_contype[bottle]&model.geom_conaffinity[g])):raise ValueError('pair was not already collision-enabled')
            if model.geom_priority[g]!=model.geom_priority[bottle] or any(not np.array_equal(getattr(model,k)[g],getattr(model,k)[bottle]) for k in ('geom_solref','geom_solimp','geom_friction')):raise ValueError('explicit normal/friction mixing would differ')
            friction=model.geom_friction[g]
            pairs.append({'name':'diagnostic_pad_bottle_'+str(len(pairs)),'geom1':name,'geom2':'test_cylinder_collision','condim':int(max(model.geom_condim[g],model.geom_condim[bottle])),'friction':[float(friction[0]),float(friction[0]),float(friction[1]),float(friction[2]),float(friction[2])],'solref':model.geom_solref[g].tolist(),'solimp':model.geom_solimp[g].tolist(),'margin':float(max(model.geom_margin[g],model.geom_margin[bottle])),'gap':float(max(model.geom_gap[g],model.geom_gap[bottle])),'solreffriction':[.002,1.]})
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='approved-pad-tangent-reference').native_path
    for label in ('host_scene','runtime_scene'):
        source=Path(record['generated'][label]['path']);xml=ET.fromstring(source.read_bytes())
        contact=xml.find('contact')
        if contact is None:contact=ET.SubElement(xml,'contact')
        if contact.findall('pair'):raise ValueError('refusing duplicate explicit contact pairs')
        for pair in pairs:
            attributes={k:(' '.join(format(x,'.17g') for x in v) if isinstance(v,list) else str(v)) for k,v in pair.items()}
            ET.SubElement(contact,'pair',**attributes)
        dest=out/source.name;pg.write_new_bytes(dest,ET.tostring(xml,encoding='utf-8',xml_declaration=True));record['generated'][label]={'path':str(dest),'sha256':digest(dest)}
    derived=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path'])
    # Appending pair names changes packed-name offsets/hash slots. Validate
    # every existing object's semantic name/ID, rather than hiding name drift.
    named_types=((mujoco.mjtObj.mjOBJ_BODY,model.nbody),(mujoco.mjtObj.mjOBJ_JOINT,model.njnt),(mujoco.mjtObj.mjOBJ_GEOM,model.ngeom),(mujoco.mjtObj.mjOBJ_ACTUATOR,model.nu),(mujoco.mjtObj.mjOBJ_SENSOR,model.nsensor),(mujoco.mjtObj.mjOBJ_SITE,model.nsite),(mujoco.mjtObj.mjOBJ_MESH,model.nmesh),(mujoco.mjtObj.mjOBJ_EQUALITY,model.neq))
    for kind,count in named_types:
        for i in range(count):
            if mujoco.mj_id2name(model,kind,i)!=mujoco.mj_id2name(derived,kind,i):raise ValueError('existing object name/ID changed')
    checked=[]
    for name in dir(model):
        a=getattr(model,name)
        if not isinstance(a,np.ndarray) or name=='_sizes' or name.startswith(('pair_','name_','names')):continue
        if not np.array_equal(a,getattr(derived,name),equal_nan=True):raise ValueError('local friction unexpected physical array delta:'+name)
        checked.append(name)
    for dimension in ('nq','nv','nu','njnt','nbody','ngeom','nmesh','neq','nsensor','nsite','ncam','nkey','ntendon'):
        if getattr(model,dimension)!=getattr(derived,dimension):raise ValueError('non-pair model dimensions changed:'+dimension)
    if derived.npair!=22 or derived.opt.noslip_iterations!=0 or derived.opt.timestep!=model.opt.timestep:raise ValueError('local friction scope/solver changed')
    for pair in pairs:
        i=mujoco.mj_name2id(derived,mujoco.mjtObj.mjOBJ_PAIR,pair['name'])
        for field,key in (('pair_friction','friction'),('pair_solref','solref'),('pair_solimp','solimp'),('pair_solreffriction','solreffriction')):
            if not np.array_equal(getattr(derived,field)[i],pair[key]):raise ValueError('compiled explicit pair differs:'+field)
        if derived.pair_dim[i]!=pair['condim'] or derived.pair_margin[i]!=pair['margin'] or derived.pair_gap[i]!=pair['gap']:raise ValueError('pair contact dimensions/margin changed')
    cfg=yaml.safe_load(Path(before['generated']['simulator_config']['path']).read_text())
    cfg['MujocoSimModule']['simulation_model']='/workspace/evidence/'+Path(record['generated']['runtime_scene']['path']).name
    cfg_path=out/'simulator_stl_native_hand_table_control.yaml';pg.write_new_text(cfg_path,yaml.safe_dump(cfg,sort_keys=False));record['generated']['simulator_config']={'path':str(cfg_path),'sha256':digest(cfg_path)}
    runtime_source=Path(record['generated']['runtime_scene']['path'])
    projection=runtime_source.read_bytes().replace(b'/workspace/raicom2026',str(ROOT/'simulation/mujoco/raicom2026').encode()).replace(b'/workspace/robotsim',str(ROOT).encode())
    projected=out/'runtime_compile_check.xml';pg.write_new_bytes(projected,projection)
    runtime=mujoco.MjModel.from_xml_path(str(projected))
    for name in checked+['pair_dim','pair_friction','pair_solref','pair_solimp','pair_solreffriction','pair_geom1','pair_geom2','pair_margin','pair_gap']:
        if not np.array_equal(getattr(derived,name),getattr(runtime,name),equal_nan=True):raise ValueError('local friction host/runtime mismatch:'+name)
    record['generated']['runtime_hostcheck_scene']={'path':str(projected),'sha256':digest(projected)}
    record['approved_local_pad_friction_reference']={'classification':'SIMULATION_ONLY_CASE_APPROVED_NUMERICAL_CONTACT_COMPARISON','approval':'direct maintainer reply 2026-11-08: approve exact22 existing pairs, .02s to .002s, NoSlip0; no other mechanical/limit changes','parent':str(parent),'parent_sha256':digest(parent),'pairs':pairs,'pair_count':22,'normal_friction_geometry_limits_unchanged':True,'global_noslip_iterations':0,'other_compiled_arrays_unchanged':checked,'existing_named_object_ids_unchanged':True,'mj_step_calls':0,'official_source':'https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#contact-pair-solreffriction'}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    return out/'result.json'


def add_reviewed_checkpoint(parent, capture, *, base_offset_xy=(0.,0.)):
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
    offset=np.asarray(base_offset_xy,dtype=float)
    if offset.shape!=(2,) or not np.isfinite(offset).all() or np.max(np.abs(offset))>.02000001:raise ValueError('checkpoint offset outside screened 20mm neighbourhood')
    row=copy.deepcopy(row)
    row['bodies']['pelvis']['position_m'][:2]=(np.asarray(row['bodies']['pelvis']['position_m'][:2])+offset).tolist()
    model=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path']);data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row)
    if np.any(offset):
        from scripts.research.issue46.native_runtime.standing_side_grasp_static_search import contacts,native
        pads,bad,table,robot_table=contacts(model,data,native.bid(model,'test_cylinder'))
        if pads or bad or robot_table or not table:raise ValueError('offset episode initialization contact preflight failed')
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
    record['reviewed_checkpoint']['episode_boundary_base_offset_xy_m']=offset.tolist()
    record['reviewed_checkpoint']['offset_is_not_navigation_or_region_qualification']=True
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);return out/'result.json'


def compare_global_impratio_one(parent):
    """Restore native global friction/normal impedance ratio; retain elliptic cone.

    The .002s per-pad tangent references require elliptic contacts. This is
    therefore a one-condition numerical comparison, not vendor-default physics.
    """
    parent=Path(parent);record=json.loads(parent.read_text())
    if not record.get("approved_local_pad_friction_reference"):raise ValueError("missing qualified local22 parent")
    for label in ("host_scene","runtime_scene","simulator_config"):
        item=record["generated"][label]
        if digest(item["path"])!=item["sha256"]:raise ValueError("solver comparison parent changed")
    original=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    if int(original.opt.cone)!=1 or original.opt.impratio!=10. or original.opt.noslip_iterations:
        raise ValueError("expected existing elliptic10/NoSlip0 comparison")
    out=pg.create_unique_output_directory(ROOT/"reviews/issue-46",prefix="global-impratio-one-comparison").native_path
    for label in ("host_scene","runtime_scene","runtime_hostcheck_scene"):
        source=Path(record["generated"][label]["path"]);xml=ET.fromstring(source.read_bytes())
        xml.find("option").set("impratio","1")
        dst=out/source.name;pg.write_new_bytes(dst,ET.tostring(xml,encoding="utf-8",xml_declaration=True))
        record["generated"][label]={"path":str(dst),"sha256":digest(dst)}
    derived=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    for name in dir(original):
        a=getattr(original,name);b=getattr(derived,name)
        if isinstance(a,np.ndarray) and not np.array_equal(a,b,equal_nan=True):raise ValueError("unexpected physical delta:"+name)
    if int(derived.opt.cone)!=1 or derived.opt.impratio!=1. or derived.opt.timestep!=original.opt.timestep:raise ValueError("solver comparison scope")
    cfg=yaml.safe_load(Path(record["generated"]["simulator_config"]["path"]).read_text())
    cfg["MujocoSimModule"]["simulation_model"]="/workspace/evidence/"+Path(record["generated"]["runtime_scene"]["path"]).name
    dst=out/"simulator_stl_native_hand_table_control.yaml";pg.write_new_text(dst,yaml.safe_dump(cfg,sort_keys=False))
    record["generated"]["simulator_config"]={"path":str(dst),"sha256":digest(dst)}
    record["diagnostic_parameter_delta"]["elliptic_solver"]["impratio_to"]=1.
    record["global_impratio_comparison"]={"from":10.,"to":1.,"cone":"elliptic","native_cone":"pyramidal","not_vendor_default":True,"purpose":"isolate global contact impedance effect on loaded gait; no friction,geometry,policy or gains changed","parent":str(parent),"parent_sha256":digest(parent),"physical_arrays_identical":True}
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(out)
    return out/"result.json"
