"""Initial scenario placement only; no active-rollout object state manipulation."""
from pathlib import Path
import copy,hashlib,json,math
import xml.etree.ElementTree as ET
import mujoco,numpy as np
from scripts.agent import path_governance as pg

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build(parent, offset_xy):
    offset=np.asarray(offset_xy,dtype=float)
    if offset.shape!=(2,) or not np.isfinite(offset).all():raise ValueError('finite initial tabletop scenario offset; supported inset checked below')
    parent=Path(parent);record=copy.deepcopy(json.loads(parent.read_text()))
    model=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path'])
    b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'test_cylinder');j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'test_cylinder_free')
    if min(b,j)<0 or model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('independent canonical object free joint required')
    adr=int(model.jnt_qposadr[j])
    out=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='unknown-bottle-initial-scenario').native_path
    for label in ('host_scene','runtime_scene','runtime_hostcheck_scene'):
        if label not in record['generated']:continue
        item=record['generated'][label];src=Path(item['path'])
        if digest(src)!=item['sha256']:raise ValueError('initial-scenario parent changed')
        xml=ET.fromstring(src.read_bytes());body=xml.find(".//body[@name='test_cylinder']");pos=np.array(list(map(float,body.get('pos').split())));pos[:2]+=offset;body.set('pos',' '.join(map(str,pos)))
        for key in xml.findall('keyframe/key'):
            q=np.array(list(map(float,key.get('qpos').split())));q[adr:adr+2]+=offset;key.set('qpos',' '.join(f'{v:.17g}' for v in q))
        dest=out/src.name;pg.write_new_bytes(dest,ET.tostring(xml,encoding='utf-8',xml_declaration=True));record['generated'][label]={'path':str(dest),'sha256':digest(dest)}
    current=mujoco.MjModel.from_xml_path(record['generated']['host_scene']['path']);checked=[]
    for name in dir(model):
        old=getattr(model,name)
        if not isinstance(old,np.ndarray):continue
        expected=old.copy()
        if name=='body_pos':expected[b,:2]+=offset
        elif name in ('qpos0','qpos_spring'):expected[adr:adr+2]+=offset
        elif name=='key_qpos':expected[:,adr:adr+2]+=offset
        if not np.array_equal(expected,getattr(current,name),equal_nan=True):raise ValueError('non-initial-pose model delta:'+name)
        checked.append(name)
    data=mujoco.MjData(current);mujoco.mj_resetDataKeyframe(current,data,0);mujoco.mj_forward(current,data)
    table=mujoco.mj_name2id(current,mujoco.mjtObj.mjOBJ_BODY,'raicom_table_table');points=[]
    for g in range(current.ngeom):
        if current.geom_bodyid[g]!=table or current.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=current.geom_dataid[g];v=current.mesh_vert[current.mesh_vertadr[mesh]:current.mesh_vertadr[mesh]+current.mesh_vertnum[mesh]];points.extend(v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
    points=np.asarray(points)
    if not len(points) or not np.all(data.xpos[b,:2]>=points.min(axis=0)[:2]+.055) or not np.all(data.xpos[b,:2]<=points.max(axis=0)[:2]-.055):raise ValueError('initial object lacks supported table inset')
    robot=mujoco.mj_name2id(current,mujoco.mjtObj.mjOBJ_BODY,'pelvis');robots={i for i in range(current.nbody) if current.body_rootid[i]==robot}
    bad=[float(c.dist) for c in data.contact if c.dist<-.001 and b in (current.geom_bodyid[c.geom1],current.geom_bodyid[c.geom2]) and any(current.geom_bodyid[g] in robots for g in (c.geom1,c.geom2))]
    if bad:raise ValueError('initial object/robot intersection')
    cp=record.get('reviewed_checkpoint',{})
    for name in ('position_m','com_m'):
        if name in cp.get('initial_bottle_pose',{}):cp['initial_bottle_pose'][name][:2]=(np.asarray(cp['initial_bottle_pose'][name][:2])+offset).tolist()
    cp.pop('physical_arrays_unchanged',None)
    record['initial_object_placement']={'classification':'INITIALIZATION_ONLY_UNKNOWN_TO_CONTROLLER','offset_xy_m':offset.tolist(),'position_world_m':data.xpos[b].tolist(),'controller_knows_object_pose':False,'physical_properties_and_robot_unchanged':True,'checked_arrays':checked,'mj_step_calls':0,'parent':{'path':str(parent),'sha256':digest(parent)}}
    record.pop('delivery_static_plan',None);pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);return out/'result.json'
