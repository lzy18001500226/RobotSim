"""Opt-in numerical manifold comparison; no geometry, material or controller edits."""
from pathlib import Path
import copy,hashlib,json,xml.etree.ElementTree as ET
import yaml
import mujoco,numpy as np
from scripts.agent import path_governance as pg

def build(parent_path):
    parent_path=Path(parent_path);parent=json.loads(parent_path.read_text());record=copy.deepcopy(parent)
    root=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
    output=pg.create_unique_output_directory(root,prefix="native-multiccd-solver-overlay").native_path
    for name in ("host_scene","runtime_scene","runtime_hostcheck_scene"):
        source=Path(parent["generated"][name]["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=parent["generated"][name]["sha256"]:raise ValueError("solver parent scene changed")
        tree=ET.parse(source);option=tree.getroot().find("option")
        if option is None:option=ET.SubElement(tree.getroot(),"option")
        flag=option.find("flag")
        if flag is None:flag=ET.SubElement(option,"flag")
        if flag.get("multiccd","disable")!="disable":raise ValueError("comparison parent must have multiccd disabled")
        flag.set("multiccd","enable")
        target=output/(name+".xml");tree.write(target,encoding="utf-8",xml_declaration=True)
        record["generated"][name]=dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    config_source=Path(parent["generated"]["simulator_config"]["path"])
    if hashlib.sha256(config_source.read_bytes()).hexdigest()!=parent["generated"]["simulator_config"]["sha256"]:raise ValueError("solver parent config changed")
    config=yaml.safe_load(config_source.read_text())
    config["MujocoSimModule"]["simulation_model"]="/workspace/evidence/runtime_scene.xml"
    config_target=output/config_source.name
    pg.write_new_text(config_target,yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    record["generated"]["simulator_config"]=dict(path=str(config_target),sha256=hashlib.sha256(config_target.read_bytes()).hexdigest())
    a=mujoco.MjModel.from_xml_path(parent["generated"]["host_scene"]["path"]);b=mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    checked=[]
    for name in dir(a):
        x=getattr(a,name)
        if isinstance(x,np.ndarray):
            if not np.array_equal(x,getattr(b,name),equal_nan=True):raise ValueError("non-solver compiled-array delta:"+name)
            checked.append(name)
    runtime=mujoco.MjModel.from_xml_path(record["generated"]["runtime_hostcheck_scene"]["path"])
    if runtime.nbody!=b.nbody or runtime.nq!=b.nq or int(runtime.opt.enableflags)!=16:raise ValueError("runtime hostcheck/model flags differ")
    if int(a.opt.enableflags)!=0 or int(b.opt.enableflags)!=int(mujoco.mjtEnableBit.mjENBL_MULTICCD):raise ValueError("only MULTICCD enable bit allowed")
    for name in dir(a.opt):
        if name.startswith("_") or name=="enableflags":continue
        x=getattr(a.opt,name);y=getattr(b.opt,name)
        if isinstance(x,np.ndarray):
            if not np.array_equal(x,y,equal_nan=True):raise ValueError("other solver array changed:"+name)
        elif isinstance(x,(int,float)):
            if x!=y:raise ValueError("other solver field changed:"+name)
    record["numeric_contact_manifold"]=dict(classification="SIMULATION_ONLY_NATIVE337_MULTICCD_COMPARISON_NOT_VENDOR_DEFAULT",parent=dict(path=str(parent_path),sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest()),old_enableflags=0,new_enableflags=16,source_doc="https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option-flag-multiccd",geometry_material_mass_gain_limit_arrays_unchanged=checked,mj_step_calls=0)
    result=output/"result.json";pg.write_new_text(result,json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(output);return result
