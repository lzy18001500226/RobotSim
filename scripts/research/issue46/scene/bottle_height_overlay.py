"""User-authorized H300 -> H220 task fixture; robot/vendor assets stay unchanged."""
from pathlib import Path
import copy,hashlib,json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scripts.agent import path_governance as pg

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(parent, height_m=.22):
    if height_m!=.22:raise ValueError("only the explicitly requested H220 fixture is supported")
    parent=Path(parent).resolve();record=copy.deepcopy(json.loads(parent.read_text()))
    original=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    bottle=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder")
    geom=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_GEOM,"test_cylinder_collision")
    joint=mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_JOINT,"test_cylinder_free")
    if min(bottle,geom,joint)<0:raise ValueError("canonical fixture names missing")
    if not np.allclose(original.geom_size[geom,:2],[.035,.15]) or abs(original.body_mass[bottle]-.2)>1e-10:
        raise ValueError("H220 requires the established H300/D70/200g source")
    adr=int(original.jnt_qposadr[joint]);dz=height_m/2.-.15
    out=pg.create_unique_output_directory(Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46"),prefix="h220-task-fixture").native_path
    for label in ("host_scene","runtime_scene","runtime_hostcheck_scene"):
        if label not in record["generated"]:continue
        item=record["generated"][label];src=Path(item["path"])
        if digest(src)!=item["sha256"]:raise ValueError("fixture parent source changed:"+label)
        root=ET.fromstring(src.read_bytes())
        body=root.find(".//body[@name='test_cylinder']")
        cylinder=body.find("geom[@name='test_cylinder_collision']")
        cylinder.set("size",f".035 {height_m/2.:.17g}")
        position=list(map(float,body.get("pos").split()));position[2]+=dz
        body.set("pos"," ".join(f"{v:.17g}" for v in position))
        for key in root.findall("keyframe/key"):
            if key.get("qpos"):
                q=list(map(float,key.get("qpos").split()));q[adr+2]+=dz
                key.set("qpos"," ".join(f"{v:.17g}" for v in q))
        dest=out/src.name;pg.write_new_bytes(dest,ET.tostring(root,encoding="utf-8",xml_declaration=True))
        record["generated"][label]=dict(path=str(dest),sha256=digest(dest))
    derived=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    if (original.nq,original.nv,original.nu,original.neq)!=(derived.nq,derived.nv,derived.nu,derived.neq):
        raise ValueError("fixture edit changed robot topology")
    checked=[]
    for name in ("body_parentid","body_quat","body_gravcomp","jnt_type","jnt_range","jnt_pos","jnt_axis",
                 "dof_frictionloss","dof_damping","geom_type","geom_pos","geom_quat","geom_friction",
                 "geom_contype","geom_conaffinity","eq_type","eq_obj1id","eq_obj2id","eq_data",
                 "actuator_trnid","actuator_ctrlrange","actuator_forcerange","actuator_gainprm","actuator_biasprm"):
        if not np.array_equal(getattr(original,name),getattr(derived,name),equal_nan=True):
            raise ValueError("non-fixture physical delta:"+name)
        checked.append(name)
    for name in ("body_pos","body_mass","body_inertia","body_ipos","body_iquat"):
        a=getattr(original,name).copy();b=getattr(derived,name).copy();a[bottle]=b[bottle]
        if not np.array_equal(a,b,equal_nan=True):raise ValueError("robot/environment body delta:"+name)
    a=original.geom_size.copy();a[geom]=derived.geom_size[geom]
    if not np.array_equal(a,derived.geom_size):raise ValueError("non-fixture geometry delta")
    keep=np.ones(original.nq,dtype=bool);keep[adr+2]=False
    if not np.array_equal(original.qpos0[keep],derived.qpos0[keep]) or not np.array_equal(original.key_qpos[:,keep],derived.key_qpos[:,keep]):
        raise ValueError("robot initialization changed")
    if not np.allclose(derived.geom_size[geom,:2],[.035,.11]) or abs(derived.body_mass[bottle]-.2)>1e-10:
        raise ValueError("H220 physical dimensions/mass mismatch")
    if record.get("reviewed_checkpoint"):
        checkpoint=record["reviewed_checkpoint"]
        checkpoint.pop("physical_arrays_unchanged",None)
        for field in ("position_m","com_m"):checkpoint["initial_bottle_pose"][field][2]+=dz
        checkpoint["fixture_height_updated_after_source_checkpoint"]=True
    record["fixture_classification"]="SIMULATION_ONLY_USER_AUTHORIZED_H220_TASK_FIXTURE_NO_ROBOT_CHANGE"
    record["fixture_height_delta"]=dict(source_height_m=.30,height_m=.22,diameter_m=.07,mass_kg=.2,
        grasp_height_from_bottom_m=.12,grasp_height_above_center_m=.01,initial_object_z_delta_m=dz,
        inertia_rule="MuJoCo cylinder inertia inferred from unchanged 0.2kg geom mass",
        robot_and_environment_unchanged=True,checked_arrays=checked,mj_step_calls=0,parent=dict(path=str(parent),sha256=digest(parent)))
    pg.write_new_text(out/"result.json",json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(out)
    return out/"result.json"
