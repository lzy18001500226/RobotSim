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
def change_friction(xml, *, both_hands=False):
    root=ET.fromstring(xml)
    names=JOINTS+tuple("L_"+n[2:] for n in JOINTS) if both_hands else JOINTS
    for name in names:
        nodes=root.findall('.//body/joint[@name="'+name+'"]')
        if len(nodes)!=1:raise ValueError('expected exactly one right-claw joint:'+name)
        nodes[0].set('frictionloss','0')
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)

def build(parent, *, both_hands=False):
    parent=Path(parent);record=json.loads(parent.read_text());generated=record['generated']
    if record['classification']!='SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY':raise ValueError('unexpected parent')
    for item in generated.values():
        if isinstance(item,dict) and 'path' in item and digest(item['path'])!=item['sha256']:raise ValueError('parent hash mismatch')
    out=pg.create_unique_output_directory(ROOT/'reviews/issue-46',prefix='claw-zero-internal-friction').native_path
    for name in ('host_scene','runtime_scene'):
        src=Path(generated[name]['path']);dst=out/src.name
        pg.write_new_bytes(dst,change_friction(src.read_bytes(),both_hands=both_hands))
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
    record['generated']=generated
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    return out/'result.json'
