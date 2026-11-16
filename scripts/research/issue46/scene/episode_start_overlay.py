"""Episode-boundary aligned START keyframe; physical models stay unchanged."""
from pathlib import Path
import copy,hashlib,json
import xml.etree.ElementTree as ET
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scripts.agent import path_governance as pg
ROOT=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(parent, poses):
    parent=Path(parent);poses=Path(poses);record=copy.deepcopy(json.loads(parent.read_text()))
    checkpoint=record["reviewed_checkpoint"]
    start=json.loads(poses.read_text())["poses"]["START_ALIGN"]
    xy=np.asarray(start["xyz_m"][:2]);yaw=float(start["yaw_rad"])
    if not np.array_equal(xy,[-1.5,-1.5]) or yaw!=0.:raise ValueError("unreviewed aligned START")
    model=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"floating_base_joint");adr=int(model.jnt_qposadr[j])
    if model.nkey!=2 or checkpoint["zero_episode_initial_velocities"] is not True:raise ValueError("episode keyframe contract")
    source_pose=copy.deepcopy(checkpoint["initial_pelvis_pose"]);oldq=model.key_qpos[0,adr+3:adr+7].copy()
    oldR=Rotation.from_quat(oldq[[1,2,3,0]]);oldyaw=np.arctan2(oldR.as_matrix()[1,0],oldR.as_matrix()[0,0])
    rotation=Rotation.from_euler("z",yaw-oldyaw)*oldR;q=rotation.as_quat()[[3,0,1,2]]
    out=pg.create_unique_output_directory(ROOT,prefix="aligned-start-keyframe").native_path
    for label in ("host_scene","runtime_scene","runtime_hostcheck_scene"):
        if label not in record["generated"]:continue
        item=record["generated"][label];source=Path(item["path"])
        if sha(source)!=item["sha256"]:raise ValueError("episode source changed")
        xml=ET.fromstring(source.read_bytes())
        for key in xml.findall("keyframe/key"):
            values=np.asarray(list(map(float,key.get("qpos").split())))
            values[adr:adr+2]=xy;values[adr+3:adr+7]=q
            key.set("qpos"," ".join(f"{v:.17g}" for v in values))
        target=out/source.name;pg.write_new_bytes(target,ET.tostring(xml,encoding="utf-8",xml_declaration=True))
        record["generated"][label]=dict(path=str(target),sha256=sha(target))
    current=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"]);checked=[]
    for name in dir(model):
        a=getattr(model,name)
        if not isinstance(a,np.ndarray):continue
        expected=a.copy()
        if name=="key_qpos":expected[:,adr:adr+2]=xy;expected[:,adr+3:adr+7]=q
        if not np.array_equal(expected,getattr(current,name),equal_nan=True):raise ValueError("non-start model delta:"+name)
        checked.append(name)
    data=mujoco.MjData(current);mujoco.mj_resetDataKeyframe(current,data,0);mujoco.mj_forward(current,data)
    root=mujoco.mj_name2id(current,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    robot={i for i in range(current.nbody) if current.body_rootid[i]==root}
    bad=[]
    for c in data.contact:
        a,b=[int(current.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
        if c.dist<-.001 and (a in robot)!=(b in robot) and 0 not in (a,b):bad.append(float(c.dist))
    if bad:raise ValueError("START environment collision:"+str(bad))
    pose=checkpoint["initial_pelvis_pose"];oldpos=np.asarray(source_pose["position_m"])
    pose["position_m"][:2]=xy.tolist()
    if "com_m" in pose:pose["com_m"]=(np.asarray(pose["position_m"])+(rotation*oldR.inv()).apply(np.asarray(source_pose["com_m"])-oldpos)).tolist()
    pose["quaternion_wxyz"]=q.tolist();pose["rotation"]=rotation.as_matrix().reshape(-1).tolist()
    checkpoint["classification"]="SIMULATION_ONLY_ALIGNED_START_NATIVE_EPISODE_INITIALIZATION"
    checkpoint["physical_arrays_unchanged"]=checked
    record["episode_start_navigation"]=dict(parent_overlay=dict(path=str(parent),sha256=sha(parent)),start_xy_m=xy.tolist(),start_yaw_rad=yaw,
        pickup_goal_world_xyz_m=source_pose["position_m"],source_poses=dict(path=str(poses),sha256=sha(poses)),
        keyframe_only_before_stepping=True,physical_arrays_unchanged=True,mj_step_calls=0)
    record.pop("delivery_static_plan",None)
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(out)
    return out/"result.json"
