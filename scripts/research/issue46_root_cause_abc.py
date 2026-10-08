#!/usr/bin/env python3
"""Root-cause A/B/C benchmark for canonical bottle and X2 OmniPicker."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
X2_ROOT = Path("/tmp/robotsim-issue46-agibot-x2-575cc6b988f976c23550e0db85aa1e5475d3652d")
X2_PIN = "575cc6b988f976c23550e0db85aa1e5475d3652d"
X2_REL = Path("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf")
MENAGERIE = Path("/tmp/robotsim-issue46-mujoco-menagerie-0059d4335f8156206f63a35662313385f7ad6d74")
MENAGERIE_PIN = "0059d4335f8156206f63a35662313385f7ad6d74"
CANONICAL = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-static/20261007-ready/canonical_manipulation_assets.py")
PRIOR = Path("/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-grasp-topology-redesign/run-20261008-3/candidate_results.json")
BASELINE_PATH = REPO / "scripts/research/issue46_omnipicker_1dof_m0.py"
AUDIT_PATH = REPO / "scripts/research/issue46_omnipicker_collision_audit_recovery.py"
ROOT_LINK = "right_elbow_link"
DRIVER = "right_claw_joint"
FOLLOWER = "R_hand_wide1_joint"
JAW_LINKS = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
CRITICAL_LINKS = ("R_hand_narrow_loop_Link", "R_hand_wide_loop_Link", "right_wrist_roll_link")
TIMESTEP = 0.002
GRAVITY = 9.81


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    return mujoco.mj_id2name(model, kind, int(ident)) or f"{kind.name.lower()}_{ident}"


def body_id(model: mujoco.MjModel, body: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if value < 0:
        raise RuntimeError(f"Missing body {body}")
    return int(value)


def geom_id(model: mujoco.MjModel, geom: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if value < 0:
        raise RuntimeError(f"Missing geom {geom}")
    return int(value)


def qpos_id(model: mujoco.MjModel, joint: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if value < 0:
        raise RuntimeError(f"Missing joint {joint}")
    return int(model.jnt_qposadr[value])


def identity() -> dict[str, Any]:
    x2urdf = X2_ROOT / X2_REL
    menxml = MENAGERIE / "robotiq_2f85/2f85.xml"
    scene = MENAGERIE / "robotiq_2f85/scene.xml"
    for root, expected in ((X2_ROOT, X2_PIN), (MENAGERIE, MENAGERIE_PIN)):
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        if head != expected or dirty:
            raise RuntimeError(f"Vendor checkout is not the exact clean pin: {root} {head} dirty={bool(dirty)}")
    native = Path(mujoco.__file__).with_name(f"libmujoco.so.{mujoco.__version__}")
    urdf = ET.parse(x2urdf).getroot()
    meshes = sorted({Path(m.get("filename", "")).name for m in urdf.findall(".//mesh")})
    assets = sorted((MENAGERIE / "robotiq_2f85/assets").glob("*"))
    return {"python": sys.version.split()[0], "mujoco_python": mujoco.__version__,
            "mujoco_native": mujoco.mj_versionString(), "native_library": str(native), "native_sha256": sha(native),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "x2_repo": "https://github.com/AgibotTech/agibot_x2_urdf",
            "x2_commit": X2_PIN, "x2_urdf": str(X2_REL), "x2_urdf_sha256": sha(x2urdf),
            "x2_license": "Mulan PSL v2", "x2_mesh_sha256": {n: sha(X2_ROOT/X2_REL.parent/"meshes"/n) for n in meshes},
            "menagerie_repo": "https://github.com/google-deepmind/mujoco_menagerie", "menagerie_commit": MENAGERIE_PIN,
            "robotiq_2f85_xml_sha256": sha(menxml), "menagerie_scene_xml_sha256": sha(scene),
            "menagerie_asset_sha256": {p.name: sha(p) for p in assets if p.is_file()},
            "menagerie_license": "BSD-2-Clause", "canonical_helper_sha256": sha(CANONICAL),
            "baseline_controller_sha256": sha(BASELINE_PATH), "conversion_audit_script_sha256": sha(AUDIT_PATH),
            "benchmark_runner_sha256": sha(Path(__file__).resolve())}


def add_geom(parent: ET.Element, name: str, typ: str, pos: tuple[float, ...], size: tuple[float, ...],
             rgba: tuple[float, ...], friction: tuple[float, ...], mass: float | None = None) -> ET.Element:
    a = {"name": name, "type": typ, "pos": " ".join(map(str, pos)), "size": " ".join(map(str, size)),
         "rgba": " ".join(map(str, rgba)), "friction": " ".join(map(str, friction)), "condim": "4", "group": "1"}
    if mass is not None:
        a["mass"] = str(mass)
    return ET.SubElement(parent, "geom", a)


def add_canonical_xml(root: ET.Element, canonical) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "abc_checker", "type": "2d", "builtin": "checker", "mark": "edge",
                                      "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
                                      "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "abc_floor", "texture": "abc_checker", "texuniform": "true",
                                      "texrepeat": "5 5", "reflectance": "0.2"})
    world = root.find("worldbody")
    ET.SubElement(world, "light", {"pos": "0.3 0 1.8", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(world, "light", {"pos": "-0.4 0.8 1.4", "dir": "0.4 -0.6 -1", "directional": "true"})
    floor = add_geom(world, "floor", "plane", (0,0,0), (0,0,0.05), (0.25,0.27,0.30,1), (1.0,0.01,0.001))
    floor.set("material", "abc_floor")
    cx, cy = canonical.G1_CANONICAL_TABLE_CENTER_XY
    table = ET.SubElement(world, "body", {"name": "m0_table", "pos": f"{cx} {cy} 0"})
    add_geom(table,"m0_table_top","box",(0,0,0.775),(0.2,0.2,0.025),canonical.G1_CANONICAL_TABLE_RGBA,(1.2,0.01,0.001))
    for n,x,y in (("m0_table_leg_front_left",-.175,.175),("m0_table_leg_front_right",.175,.175),
                  ("m0_table_leg_back_left",-.175,-.175),("m0_table_leg_back_right",.175,-.175)):
        add_geom(table,n,"box",(x,y,.375),(.025,.025,.375),canonical.G1_CANONICAL_TABLE_RGBA,(.9,.01,.001))
    bottle = ET.SubElement(world,"body",{"name":"m0_bottle","pos":" ".join(map(str,canonical.CANONICAL_X2_BOTTLE_START_BODY_POS))})
    ET.SubElement(bottle,"freejoint",{"name":"m0_bottle_free"})
    for g in canonical.CANONICAL_X2_BOTTLE_GEOMS:
        add_geom(bottle,g["name"],g["type"],tuple(g["pos"]),tuple(g["size"]),tuple(g["rgba"]),(1.4,.02,.001),float(g["mass"]))


def reference_root_pose(source_xml: Path, canonical) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(source_xml))
    root = body_id(model,"base_mount")
    # Side-entry orientation keeps the housing behind the bottle neck while
    # preserving the model's local-X opposing jaw closure axis.
    root_quat = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    model.body_quat[root] = root_quat
    data = mujoco.MjData(model); mujoco.mj_resetData(model,data); mujoco.mj_forward(model,data)
    pads = [geom_id(model,n) for n in ("right_pad1","right_pad2","left_pad1","left_pad2")]
    pad_center = np.mean(data.geom_xpos[pads],axis=0)
    mount_origin = data.xpos[root]
    bottle_center = np.array(canonical.CANONICAL_X2_BOTTLE_START_BODY_POS,dtype=float)+np.array([0,0,-.04])
    position = bottle_center-(pad_center-mount_origin)
    return {"root_position_world_m":position.tolist(),"root_quaternion_wxyz":root_quat,
            "open_pad_center_relative_to_mount_m":(pad_center-mount_origin).tolist(),
            "target_contact_center_world_m":bottle_center.tolist()}


def build_reference_scene(out: Path, canonical) -> tuple[mujoco.MjModel, dict[str,Any]]:
    src=MENAGERIE/"robotiq_2f85/2f85.xml"; tree=ET.parse(src); root=tree.getroot()
    root.find("compiler").set("meshdir",str((MENAGERIE/"robotiq_2f85/assets").resolve()))
    opt=root.find("option"); opt.set("timestep",str(TIMESTEP)); opt.set("gravity",f"0 0 {-GRAVITY}"); opt.set("integrator","implicitfast")
    pose=reference_root_pose(src,canonical)
    mount=root.find("worldbody/body[@name='base_mount']")
    mount.set("pos"," ".join(f"{v:.12g}" for v in pose["root_position_world_m"]))
    mount.set("quat"," ".join(map(str,pose["root_quaternion_wxyz"])))
    # The local +Y axis maps to world +Z for this side-entry root orientation.
    mount.insert(0,ET.Element("joint",{"name":"m0_gripper_lift_joint","type":"slide","axis":"0 1 0",
                                        "range":"0 0.03","limited":"true","damping":"1","armature":"0.01"}))
    add_canonical_xml(root,canonical)
    ET.SubElement(root.find("actuator"),"position",{"name":"m0_gripper_lift_motor","joint":"m0_gripper_lift_joint",
        "kp":"10000","kv":"200","ctrlrange":"0 0.035","forcelimited":"true","forcerange":"-50 50"})
    vis=root.find("visual")
    if vis is None: vis=ET.SubElement(root,"visual")
    ET.SubElement(vis,"headlight",{"diffuse":"0.75 0.75 0.75","ambient":"0.35 0.35 0.35","specular":"0.15 0.15 0.15"})
    path=out/"reference_2f85_scene.xml"; ET.indent(root,space="  "); tree.write(path,encoding="utf-8",xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path))
    return model,{"scene_xml":str(path),"scene_xml_sha256":sha(path),"root_pose":pose,
                  "lift_force_bound_n":[-50,50],"lift_kp":10000,"lift_kv":200,
                  "lift_target_m":0.035,"minimum_bottle_lift_m":0.03,
                  "gripper_subtree_mass_kg":1.0526083388427392,"gripper_gravity_load_n":10.326087804047274}


def contact_rows(model: mujoco.MjModel,data: mujoco.MjData,with_force: bool=False)->list[dict[str,Any]]:
    out=[]
    for i in range(data.ncon):
        c=data.contact[i]; g1,g2=int(c.geom1),int(c.geom2)
        b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
        force=np.zeros(6)
        if with_force and int(c.efc_address)>=0: mujoco.mj_contactForce(model,data,i,force)
        out.append({"geom1":name(model,mujoco.mjtObj.mjOBJ_GEOM,g1),"body1":name(model,mujoco.mjtObj.mjOBJ_BODY,b1),
                    "geom2":name(model,mujoco.mjtObj.mjOBJ_GEOM,g2),"body2":name(model,mujoco.mjtObj.mjOBJ_BODY,b2),
                    "distance_m":float(c.dist),"efc_address":int(c.efc_address),"excluded":int(c.exclude),
                    "force_local_n":force.tolist() if with_force and c.efc_address>=0 else None,
                    "normal_force_n":float(force[0]) if with_force and c.efc_address>=0 else None})
    return out


def render(model: mujoco.MjModel,data: mujoco.MjData,path: Path,lookat,distance,azimuth,elevation)->np.ndarray:
    renderer=mujoco.Renderer(model,height=480,width=640)
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.lookat[:]=lookat
    cam.distance=distance; cam.azimuth=azimuth; cam.elevation=elevation
    renderer.update_scene(data,camera=cam); frame=renderer.render(); Image.fromarray(frame).save(path); renderer.close()
    return frame


def reference_views(model: mujoco.MjModel,data: mujoco.MjData,out: Path,label: str)->None:
    render(model,data,out/f"reference_{label}_overview.png",(.30,0,.87),.65,135,-18)
    render(model,data,out/f"reference_{label}_side.png",(.30,0,.87),.55,90,-8)
    render(model,data,out/f"reference_{label}_closeup.png",(.30,0,.88),.34,130,-12)


def run_reference(model: mujoco.MjModel,out: Path,meta: dict[str,Any])->dict[str,Any]:
    data=mujoco.MjData(model); mujoco.mj_resetData(model,data); mujoco.mj_forward(model,data)
    grip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"fingers_actuator")
    lift=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"m0_gripper_lift_motor")
    bottle=body_id(model,"m0_bottle"); bj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"m0_bottle_free")
    bvel=int(model.jnt_dofadr[bj]); start=data.xpos[bottle].copy(); initial_contacts=contact_rows(model,data,True)
    robot_bottle=[c for c in initial_contacts if (c["body1"]=="m0_bottle" and c["body2"] not in {"world","m0_table"}) or
                  (c["body2"]=="m0_bottle" and c["body1"] not in {"world","m0_table"})]
    robot_table=[c for c in initial_contacts if (c["body1"].startswith("m0_table") and c["body2"] not in {"world","m0_bottle"}) or
                 (c["body2"].startswith("m0_table") and c["body1"] not in {"world","m0_bottle"})]
    if robot_bottle or robot_table:
        result={"model":"official MuJoCo Menagerie Robotiq 2F-85","status":"INVALID_INITIAL_GEOMETRY",
                "physics_steps":0,"initial_robot_bottle_contacts":robot_bottle,"initial_robot_table_contacts":robot_table,
                "initial_contacts":initial_contacts,"bottle_qpos_writes_after_initialization":0,"scene":meta}
        write_json(out/"reference_preflight.json",result)
        return result
    reference_views(model,data,out,"pregrasp")
    renderer=mujoco.Renderer(model,height=480,width=640); cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:]=[.30,0,.87]; cam.distance=.55; cam.azimuth=135; cam.elevation=-18
    frames=[]; rows=[]; t_first_any=None; t_first_bilateral=None; max_force={}; max_lift=-1.0; max_lift_frame=None
    dt=float(model.opt.timestep); step=0
    for phase,count in (("open_hold",250),("close",250),("hold",500),("lift",250),("post_lift_hold",500)):
        for k in range(1,count+1):
            if phase=="close":
                s=min(1.,k/250); smooth=10*s**3-15*s**4+6*s**5; data.ctrl[grip]=255*smooth
            else: data.ctrl[grip]=255. if phase in {"hold","lift","post_lift_hold"} else 0.
            data.ctrl[lift]=.035 if phase in {"lift","post_lift_hold"} else 0.
            mujoco.mj_step(model,data); step+=1
            contacts=contact_rows(model,data,True); bc=[c for c in contacts if c["body1"]=="m0_bottle" or c["body2"]=="m0_bottle"]
            right=[c for c in bc if "right_pad" in c["geom1"]+c["geom2"]]
            left=[c for c in bc if "left_pad" in c["geom1"]+c["geom2"]]
            if t_first_any is None and (right or left): t_first_any=float(data.time)
            if t_first_bilateral is None and right and left: t_first_bilateral=float(data.time)
            for c in bc:
                if c["normal_force_n"] is not None:
                    pads=[g for g in (c["geom1"],c["geom2"]) if g.startswith(("left_pad", "right_pad"))]
                    for pad in pads:
                        max_force[pad]=max(max_force.get(pad,0.),c["normal_force_n"])
            p=data.xpos[bottle].copy(); q=data.qpos[qpos_id(model,"m0_gripper_lift_joint")]
            table_contacts=sum(1 for c in bc if c["body1"].startswith("m0_table") or c["body2"].startswith("m0_table"))
            row={"step":step,"time_s":float(data.time),"phase":phase,"gripper_ctrl":float(data.ctrl[grip]),
                 "lift_target_m":float(data.ctrl[lift]),"lift_qpos_m":float(q),"driver_qpos_rad":float(data.qpos[qpos_id(model,"right_driver_joint")]),
                 "bottle_x_m":float(p[0]),"bottle_y_m":float(p[1]),"bottle_z_m":float(p[2]),"bottle_lift_m":float(p[2]-start[2]),
                 "bottle_linear_speed_m_s":float(np.linalg.norm(data.qvel[bvel:bvel+3])),"bottle_angular_speed_rad_s":float(np.linalg.norm(data.qvel[bvel+3:bvel+6])),
                 "right_pad_contacts":json.dumps(right,separators=(",",":")),"left_pad_contacts":json.dumps(left,separators=(",",":")),
                 "bottle_table_contacts":table_contacts,"bottle_qpos_writes_after_start":0,
                 "active_bottle_contacts":json.dumps(bc,separators=(",",":"))}; rows.append(row)
            if step%10==0:
                renderer.update_scene(data,camera=cam); frames.append(renderer.render())
            if phase=="hold" and k==count: reference_views(model,data,out,"hold")
            if phase=="lift" and k==count:
                reference_views(model,data,out,"maximum_lift"); max_lift_frame=step
            if float(p[2]-start[2])>max_lift: max_lift=float(p[2]-start[2]); max_lift_frame=step
    renderer.close(); imageio.mimsave(out/"reference_physical_attempt.mp4",frames,fps=25)
    with (out/"reference_physics_trace.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hold=[r for r in rows if r["phase"]=="hold"]
    bilateral=sum(bool(json.loads(r["right_pad_contacts"])) and bool(json.loads(r["left_pad_contacts"])) for r in hold)/max(1,len(hold))
    liftrows=[r for r in rows if r["phase"] in {"lift","post_lift_hold"}]
    lost_support=bool(liftrows) and all(r["bottle_table_contacts"]==0 for r in liftrows[-100:])
    lift_value=max((r["bottle_lift_m"] for r in rows),default=0.)
    result={"model":"official MuJoCo Menagerie Robotiq 2F-85", "protocol":"OPEN -> CLOSE -> bilateral contact -> HOLD >=1s -> actuated vertical carriage 30mm -> HOLD >=1s",
            "status":"PASS" if bilateral>=.8 and lift_value>=.03 and lost_support else "FAIL",
            "first_any_pad_contact_s":t_first_any,"first_bilateral_pad_contact_s":t_first_bilateral,
            "bilateral_contact_fraction_during_hold":bilateral,"maximum_bottle_lift_m":lift_value,
            "table_contact_absent_in_final_100_lift_steps":lost_support,"maximum_pad_normal_force_n":max_force,
            "final_bottle_position_world_m":[rows[-1]["bottle_x_m"],rows[-1]["bottle_y_m"],rows[-1]["bottle_z_m"]],
            "bottle_qpos_writes_after_initialization":0,"bottle_freejoint":True,"bottle_mocap_id":int(model.body_mocapid[bottle]),
            "bottle_related_equalities":[name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i) for i in range(model.neq) if "bottle" in (name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i).lower())],
            "initial_contacts":initial_contacts,"max_lift_sample_step":max_lift_frame,
            "evidence":["reference_physics_trace.csv","reference_physical_attempt.mp4","reference_pregrasp_overview.png","reference_hold_overview.png","reference_maximum_lift_overview.png"],"scene":meta}
    write_json(out/"reference_result.json",result); return result


def modules():
    base=load_module("issue46_omnipicker_1dof_m0_abc",BASELINE_PATH); base.X2_ROOT=X2_ROOT; base.X2_URDF_REL=X2_REL
    audit=load_module("issue46_omnipicker_collision_audit_abc",AUDIT_PATH); audit.X2_ROOT=X2_ROOT; audit.X2_URDF=X2_ROOT/X2_REL
    audit.ROOT_LINK=ROOT_LINK
    return audit,base


def candidate_data()->dict[str,Any]:
    d=json.loads(PRIOR.read_text(encoding="utf-8")); x=next(r for r in d["results"] if r["candidate"]=="witness_derived_wrist_relief_tilt")
    return {"name":x["candidate"],"root_position_world_m":x["root_position_world_m"],"root_rotation_world":x["root_rotation_world"],
            "source_targets":x["source_targets"],"previous_variables":x["variables"],
            "previous_jaw_distances_m":[r["signed_distance_m"] for r in x["closed"]["jaw_distance_rows"]],
            "previous_wrist_loop_clearance_m":x["closed"]["minimum_critical_robot_bottle"]["signed_distance_m"]}


def bottle_geoms(model):
    bid=body_id(model,"m0_bottle"); return [g for g in range(model.ngeom) if int(model.geom_bodyid[g])==bid]


def all_collision_geoms(model):
    return [g for g in range(model.ngeom) if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])]


def geom_distance(model,data,g1,g2):
    seg=np.zeros(6); d=float(mujoco.mj_geomDistance(model,data,g1,g2,1.,seg)); l=float(np.linalg.norm(seg[3:]-seg[:3]))
    return {"geom1":name(model,mujoco.mjtObj.mjOBJ_GEOM,g1),"body1":name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g1]),
            "geom2":name(model,mujoco.mjtObj.mjOBJ_GEOM,g2),"body2":name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g2]),
            "signed_distance_m":d,"witness_length_m":l,"query_consistent":math.isfinite(d) and abs(abs(d)-l)<=1e-7+1e-6*l,
            "witness_segment_world_m":seg.tolist()}


def static_profile(base,model,data,candidate,label,out):
    bottle=bottle_geoms(model); bset=set(bottle)
    robots=[g for g in all_collision_geoms(model) if g not in bset and name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g])!="world"]
    jawset={name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) for g in robots if name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) in JAW_LINKS}
    jawids=[g for g in robots if name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) in JAW_LINKS]
    source_meshes=prepare_source_surface_samples((*CRITICAL_LINKS,*JAW_LINKS))
    rows=[]
    for a in np.linspace(1.,0.,101):
        target=base.aperture_targets(float(a)); data.qpos[qpos_id(model,DRIVER)]=target["right_claw_joint_target_rad"]; data.qpos[qpos_id(model,FOLLOWER)]=target["R_hand_wide1_joint_target_rad"]
        data.qvel[:]=0.; mujoco.mj_forward(model,data)
        jaw=[geom_distance(model,data,g,b) for g in jawids for b in bottle]
        allpairs=[geom_distance(model,data,g,b) for g in robots for b in bottle]
        source_minima=source_surface_minima(model,data,source_meshes)
        valid=[x for x in allpairs if x["query_consistent"]]
        contacts=[c for c in contact_rows(model,data) if c["body1"]=="m0_bottle" or c["body2"]=="m0_bottle"]
        by_jaw={link:min((x["signed_distance_m"] for x in jaw if x["body1"]==link),default=None) for link in JAW_LINKS}
        rows.append({"aperture_ratio":float(a),"driver_target_rad":target["right_claw_joint_target_rad"],"follower_target_rad":target["R_hand_wide1_joint_target_rad"],
                     "jaw_distance_rows":jaw,"min_jaw_by_link_m":by_jaw,"min_nonjaw_distance_m":min((x["signed_distance_m"] for x in valid if x["body1"] not in jawset),default=None),
                     "source_sampled_surface_min_sdf_by_link_m":source_minima,
                     "active_bottle_contacts":contacts,"invalid_distance_queries":sum(not x["query_consistent"] for x in allpairs)})
    with (out/f"{label}_closure_profile.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v,separators=(",",":")) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    a0=float(candidate["previous_variables"]["aperture"]); nearest=min(rows,key=lambda r:abs(r["aperture_ratio"]-a0))
    target=base.aperture_targets(a0); data.qpos[qpos_id(model,DRIVER)]=target["right_claw_joint_target_rad"]
    data.qpos[qpos_id(model,FOLLOWER)]=target["R_hand_wide1_joint_target_rad"]; data.qvel[:]=0.; mujoco.mj_forward(model,data)
    jaw=[geom_distance(model,data,g,b) for g in jawids for b in bottle]
    allpairs=[geom_distance(model,data,g,b) for g in robots for b in bottle]; valid=[x for x in allpairs if x["query_consistent"]]
    source_minima=source_surface_minima(model,data,source_meshes)
    exact={"aperture_ratio":a0,"driver_target_rad":target["right_claw_joint_target_rad"],"follower_target_rad":target["R_hand_wide1_joint_target_rad"],
           "jaw_distance_rows":jaw,"min_jaw_by_link_m":{link:min((x["signed_distance_m"] for x in jaw if x["body1"]==link),default=None) for link in JAW_LINKS},
           "min_nonjaw_distance_m":min((x["signed_distance_m"] for x in valid if x["body1"] not in jawset),default=None),
           "source_sampled_surface_min_sdf_by_link_m":source_minima,
           "active_bottle_contacts":[c for c in contact_rows(model,data) if c["body1"]=="m0_bottle" or c["body2"]=="m0_bottle"],
           "invalid_distance_queries":sum(not x["query_consistent"] for x in allpairs)}
    active_jaw_bodies={c["body1"] if c["body2"]=="m0_bottle" else c["body2"]
                        for c in exact["active_bottle_contacts"]}
    active_bilateral=all(link in active_jaw_bodies for link in JAW_LINKS)
    surfaces_within_tolerance=all(v is not None and abs(v)<=1e-5 for v in exact["min_jaw_by_link_m"].values())
    return {"label":label,"candidate_pose":candidate,"selected_aperture":a0,"selected_row":exact,"nearest_grid_row":nearest,"open_row":rows[0],"fully_closed_row":rows[-1],
            "minimum_nonjaw_distance_over_profile_m":min((r["min_nonjaw_distance_m"] for r in rows if r["min_nonjaw_distance_m"] is not None),default=None),
            "opposing_surfaces_within_10um_at_selected_aperture":surfaces_within_tolerance,
            "active_bilateral_jaw_contacts_at_selected_aperture":active_bilateral,
            "active_jaw_contact_bodies_at_selected_aperture":sorted(active_jaw_bodies.intersection(JAW_LINKS)),
            "physics_steps":0,"kinematic_qpos_assignments_for_profile":204,"active_rollout_qpos_writes":0}


def stl_triangles(path:Path)->np.ndarray:
    import struct
    b=path.read_bytes(); n=struct.unpack_from("<I",b,80)[0]
    if len(b)!=84+50*n: raise ValueError(f"Expected binary STL: {path}")
    out=np.empty((n,3,3),float)
    for i in range(n): out[i]=np.asarray(struct.unpack_from("<12fH",b,84+i*50)[3:12]).reshape(3,3)
    return out


def prepare_source_surface_samples(links):
    urdf=ET.parse(X2_ROOT/X2_REL).getroot(); linkmap={x.get("name"):x for x in urdf.findall("link")}; prepared={}
    for linkname in links:
        col=linkmap[linkname].find("collision"); mesh=col.find("geometry/mesh"); path=X2_ROOT/X2_REL.parent/"meshes"/Path(mesh.get("filename")).name
        tri=stl_triangles(path)*np.asarray([float(v) for v in mesh.get("scale","1 1 1").split()])[None,None,:]
        pts=np.concatenate((tri.reshape(-1,3),tri.mean(1),.5*(tri[:,0]+tri[:,1]),.5*(tri[:,1]+tri[:,2]),.5*(tri[:,2]+tri[:,0])))
        origin=col.find("origin"); xyz=np.asarray([float(v) for v in (origin.get("xyz","0 0 0") if origin is not None else "0 0 0").split()])
        rpy=[float(v) for v in (origin.get("rpy","0 0 0") if origin is not None else "0 0 0").split()]
        prepared[linkname]=(pts,Rotation.from_euler("xyz",rpy).as_matrix(),xyz)
    return prepared


def source_surface_minima(model,data,prepared):
    bottle=geom_id(model,"bottle_body"); center=data.geom_xpos[bottle].copy(); rot=data.geom_xmat[bottle].reshape(3,3).copy()
    rad=float(model.geom_size[bottle,0]); half=float(model.geom_size[bottle,1]); result={}
    for linkname,(pts,Rcollision,xyz) in prepared.items():
        bid=body_id(model,linkname); Rbody=data.xmat[bid].reshape(3,3)
        world=(pts@Rcollision.T+xyz)@Rbody.T+data.xpos[bid]
        result[linkname]=float(np.min(cylinder_sdf(world,center,rot,rad,half)))
    return result


def cylinder_sdf(points,center,rot,radius,half):
    p=(points-center)@rot; r=np.linalg.norm(p[:,:2],axis=1); q0=r-radius; q1=np.abs(p[:,2])-half
    outside=np.linalg.norm(np.column_stack((np.maximum(q0,0),np.maximum(q1,0))),axis=1)
    return outside+np.minimum(np.maximum(q0,q1),0)


def source_surface_rows(model,data,links):
    urdf=ET.parse(X2_ROOT/X2_REL).getroot(); linkmap={x.get("name"):x for x in urdf.findall("link")}; bottle=geom_id(model,"bottle_body")
    center=data.geom_xpos[bottle].copy(); rot=data.geom_xmat[bottle].reshape(3,3).copy(); rad=float(model.geom_size[bottle,0]); half=float(model.geom_size[bottle,1]); rows=[]
    for linkname in links:
        col=linkmap[linkname].find("collision"); mesh=col.find("geometry/mesh"); path=X2_ROOT/X2_REL.parent/"meshes"/Path(mesh.get("filename")).name
        tri=stl_triangles(path)*np.asarray([float(v) for v in mesh.get("scale","1 1 1").split()])[None,None,:]
        pts=np.concatenate((tri.reshape(-1,3),tri.mean(1),.5*(tri[:,0]+tri[:,1]),.5*(tri[:,1]+tri[:,2]),.5*(tri[:,2]+tri[:,0])))
        origin=col.find("origin"); xyz=np.asarray([float(v) for v in (origin.get("xyz","0 0 0") if origin is not None else "0 0 0").split()])
        rpy=[float(v) for v in (origin.get("rpy","0 0 0") if origin is not None else "0 0 0").split()]
        Rcollision=Rotation.from_euler("xyz",rpy).as_matrix(); bid=body_id(model,linkname); Rbody=data.xmat[bid].reshape(3,3)
        # MuJoCo recenters and rotates mesh assets at compile time; source STL
        # points must be transformed from the URDF collision frame, not geom_xpos.
        world=(pts@Rcollision.T+xyz)@Rbody.T+data.xpos[bid]; sdf=cylinder_sdf(world,center,rot,rad,half); ix=int(np.argmin(sdf))
        gids=[g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and model.geom_contype[g]]
        compiled=[geom_distance(model,data,g,bottle) for g in gids]
        hull=min(compiled,key=lambda x:x["signed_distance_m"])
        rows.append({"link":linkname,"source_mesh":path.name,"source_mesh_sha256":sha(path),"triangles":len(tri),"sample_count":len(pts),
                     "collision_origin_xyz_rpy":[xyz.tolist(),rpy],
                     "sampled_surface_min_sdf_m":float(sdf[ix]),"source_surface_intersects_bottle_cylinder":bool(sdf[ix]<0),
                     "closest_sample_world_m":world[ix].tolist(),"compiled_mesh_signed_distance_m":hull["signed_distance_m"],
                     "compiled_witness_consistent":hull["query_consistent"],"compiled_collision_geoms":compiled})
    return rows


def x2_modules_and_model(audit,base,urdf,candidate,with_lift=False):
    spec=mujoco.MjSpec.from_file(str(urdf)); spec.compiler.fusestatic=False; spec.option.timestep=TIMESTEP; spec.option.gravity=[0,0,-GRAVITY]
    root=next(b for b in spec.bodies if b.name==ROOT_LINK); root.pos=candidate["root_position_world_m"]
    R=np.asarray(candidate["root_rotation_world"],float); root.quat=Rotation.from_matrix(R).as_quat(scalar_first=True).tolist()
    if with_lift:
        axis=R.T@np.array([0.,0.,1.])
        root.add_joint(name="m0_gripper_lift_joint",type=mujoco.mjtJoint.mjJNT_SLIDE,axis=axis.tolist(),limited=True,range=[0.,.03],damping=1.,armature=.01)
    joints={str(j.name):j for j in spec.joints if j.name}
    for n in ("right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"):
        if n in joints: spec.add_equality(name=f"m0_rigid_mount_{n}",type=mujoco.mjtEq.mjEQ_JOINT,name1=n,data=[0.]*11,solref=[.005,1.])
    for n,j in (("m0_right_driver_servo",DRIVER),("m0_right_follower_servo",FOLLOWER)):
        spec.add_actuator(name=n,target=j,trntype=mujoco.mjtTrn.mjTRN_JOINT,dyntype=mujoco.mjtDyn.mjDYN_NONE,gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_NONE,gainprm=[1.]+[0.]*9,ctrllimited=True,ctrlrange=[-1.,1.],forcelimited=True,forcerange=[-1.,1.])
    if with_lift:
        spec.add_actuator(name="m0_gripper_lift_motor",target="m0_gripper_lift_joint",trntype=mujoco.mjtTrn.mjTRN_JOINT,dyntype=mujoco.mjtDyn.mjDYN_NONE,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED,biastype=mujoco.mjtBias.mjBIAS_AFFINE,gainprm=[1000.]+[0.]*9,
                          biasprm=[0.,-1000.,-100.]+[0.]*7,ctrllimited=True,ctrlrange=[0.,.03],forcelimited=True,forcerange=[-50.,50.])
    for i,(a,b) in enumerate(base.STATIC_COLLISION_EXCLUSIONS):
        present={str(x.name) for x in spec.bodies}
        if a in present and b in present: spec.add_exclude(name=f"m0_static_exclusion_{i}",bodyname1=a,bodyname2=b)
    base.add_canonical_scene(spec,bottle=True); model=spec.compile(); data=mujoco.MjData(model); mujoco.mj_resetData(model,data)
    t=base.aperture_targets(1.); data.qpos[qpos_id(model,DRIVER)]=t["right_claw_joint_target_rad"]; data.qpos[qpos_id(model,FOLLOWER)]=t["R_hand_wide1_joint_target_rad"]
    data.qvel[:]=0.; mujoco.mj_forward(model,data)
    return model,data,None


def source_fk_report(model,data,urdf):
    root=ET.parse(urdf).getroot(); joints=root.findall("joint"); rootid=body_id(model,ROOT_LINK); T=np.eye(4); T[:3,:3]=data.xmat[rootid].reshape(3,3); T[:3,3]=data.xpos[rootid]
    trans={ROOT_LINK:T}; rem=list(joints)
    while rem:
        changed=False
        for j in rem[:]:
            p,c=j.find("parent"),j.find("child")
            if p is None or c is None or p.get("link") not in trans: continue
            o=j.find("origin"); xyz=np.array([float(x) for x in (o.get("xyz","0 0 0") if o is not None else "0 0 0").split()]); rpy=[float(x) for x in (o.get("rpy","0 0 0") if o is not None else "0 0 0").split()]
            To=np.eye(4); To[:3,:3]=Rotation.from_euler("xyz",rpy).as_matrix(); To[:3,3]=xyz; motion=np.eye(4)
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j.get("name",""))
            if jid>=0 and j.get("type") in {"revolute","continuous","prismatic"}:
                q=float(data.qpos[model.jnt_qposadr[jid]]); axis=np.array([float(x) for x in j.find("axis").get("xyz","0 0 1").split()])
                if j.get("type")=="prismatic": motion[:3,3]=axis*q
                else: motion[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix()
            trans[c.get("link")]=trans[p.get("link")]@To@motion; rem.remove(j); changed=True
        if not changed: break
    errs=[]
    for link,T in trans.items():
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,link)
        if bid<0: continue
        pe=float(np.linalg.norm(T[:3,3]-data.xpos[bid])); Rm=data.xmat[bid].reshape(3,3); ca=np.clip((np.trace(T[:3,:3].T@Rm)-1)/2,-1,1)
        errs.append({"link":link,"position_error_m":pe,"orientation_error_rad":float(math.acos(ca))})
    return {"source_joint_count":len(joints),"matched_body_count":len(errs),"max_position_error_m":max((r["position_error_m"] for r in errs),default=None),
            "max_orientation_error_rad":max((r["orientation_error_rad"] for r in errs),default=None),"per_link":errs}


def x2_static(audit,base,candidate,urdf,out,label):
    model,data,control=x2_modules_and_model(audit,base,urdf,candidate); profile=static_profile(base,model,data,candidate,label,out)
    target=base.aperture_targets(float(candidate["previous_variables"]["aperture"])); data.qpos[qpos_id(model,DRIVER)]=target["right_claw_joint_target_rad"]; data.qpos[qpos_id(model,FOLLOWER)]=target["R_hand_wide1_joint_target_rad"]
    data.qvel[:]=0.; mujoco.mj_forward(model,data); fk=source_fk_report(model,data,urdf); surfaces=source_surface_rows(model,data,list(CRITICAL_LINKS)+list(JAW_LINKS))
    render(model,data,out/f"{label}_overview.png",(.30,0,.88),.65,135,-16); render(model,data,out/f"{label}_side.png",(.30,0,.88),.52,90,-8); render(model,data,out/f"{label}_closeup.png",(.30,0,.90),.34,130,-10)
    equalities=[{"name":name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i),"type":int(model.eq_type[i]),"active_at_reset":bool(model.eq_active0[i]),"obj1":int(model.eq_obj1id[i]),"obj2":int(model.eq_obj2id[i]),"data":model.eq_data[i].tolist()} for i in range(model.neq)]
    return {"profile":profile,"source_fk_parity":fk,"source_surface_samples":surfaces,"equalities":equalities,
            "actuators":[{"name":name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i),"trnid":model.actuator_trnid[i].tolist(),"gear":model.actuator_gear[i].tolist(),
                           "ctrlrange":model.actuator_ctrlrange[i].tolist(),"forcerange":model.actuator_forcerange[i].tolist()} for i in range(model.nu)],
            "model_counts":{"nq":int(model.nq),"nv":int(model.nv),"nu":int(model.nu),"nbody":int(model.nbody),"ngeom":int(model.ngeom),"neq":int(model.neq),"nexclude":int(model.nexclude),
                            "parent_filter_enabled":not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))}}


def derive_tcp_centered_candidate(audit,base,urdf,candidate)->dict[str,Any]:
    initial_model,initial_data,_=x2_modules_and_model(audit,base,urdf,candidate)
    def body_collision(model,body):
        bid=body_id(model,body); matches=[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==bid and int(model.geom_contype[g])]
        if len(matches)!=1: raise RuntimeError(f"Expected one collision geom for {body}, found {matches}")
        return matches[0]
    narrow=body_collision(initial_model,"R_hand_narrow3_Link"); wide=body_collision(initial_model,"R_hand_wide3_Link")
    bottle=geom_id(initial_model,"bottle_body"); aperture=float(candidate["previous_variables"]["aperture"])
    def set_aperture(model,data,value):
        target=base.aperture_targets(float(value)); data.qpos[qpos_id(model,DRIVER)]=target["right_claw_joint_target_rad"]
        data.qpos[qpos_id(model,FOLLOWER)]=target["R_hand_wide1_joint_target_rad"]; data.qvel[:]=0.; mujoco.mj_forward(model,data)
        return (geom_distance(model,data,narrow,bottle)["signed_distance_m"],
                geom_distance(model,data,wide,bottle)["signed_distance_m"],target)
    dn,dw,_=set_aperture(initial_model,initial_data,aperture)
    nb=body_id(initial_model,"R_hand_narrow3_Link"); wb=body_id(initial_model,"R_hand_wide3_Link")
    jaw_axis=initial_data.xpos[wb]-initial_data.xpos[nb]; jaw_axis[2]=0.; jaw_axis/=np.linalg.norm(jaw_axis)
    original_pos=np.asarray(candidate["root_position_world_m"],dtype=float); root_offset=0.; iterations=[]
    solved_aperture=aperture; solved_distances=(dn,dw)
    for iteration in range(5):
        trial=copy.deepcopy(candidate); trial["root_position_world_m"]=(original_pos+root_offset*jaw_axis).tolist()
        model,data,_=x2_modules_and_model(audit,base,urdf,trial); bottle_id=geom_id(model,"bottle_body")
        narrow_id=body_collision(model,"R_hand_narrow3_Link"); wide_id=body_collision(model,"R_hand_wide3_Link")
        def pair_at(value):
            target=base.aperture_targets(float(value)); data.qpos[qpos_id(model,DRIVER)]=target["right_claw_joint_target_rad"]
            data.qpos[qpos_id(model,FOLLOWER)]=target["R_hand_wide1_joint_target_rad"]; data.qvel[:]=0.; mujoco.mj_forward(model,data)
            return (geom_distance(model,data,narrow_id,bottle_id)["signed_distance_m"],
                    geom_distance(model,data,wide_id,bottle_id)["signed_distance_m"],target)
        lo,hi=0.,1.; f_lo=sum(pair_at(lo)[:2]); f_hi=sum(pair_at(hi)[:2])
        if f_lo>0 or f_hi<0: raise RuntimeError(f"No source-limited aperture brackets simultaneous opposing contact: {f_lo=} {f_hi=}")
        for _ in range(56):
            mid=(lo+hi)/2.; f_mid=sum(pair_at(mid)[:2])
            if f_mid>0: hi=mid
            else: lo=mid
        solved_aperture=(lo+hi)/2.; solved_distances=pair_at(solved_aperture)[:2]
        imbalance=(solved_distances[0]-solved_distances[1])/2.
        iterations.append({"iteration":iteration,"root_lateral_offset_m":root_offset,"aperture":solved_aperture,
                           "narrow_distance_m":solved_distances[0],"wide_distance_m":solved_distances[1],
                           "center_error_m":imbalance,"jaw_axis_world":jaw_axis.tolist()})
        if abs(imbalance)<1e-5: break
        root_offset+=imbalance
    result=copy.deepcopy(candidate); result["name"]="derived_tcp_centered_opposing_contact"
    result["root_position_world_m"]=(original_pos+root_offset*jaw_axis).tolist()
    result["previous_variables"]["aperture"]=float(solved_aperture)
    target=base.aperture_targets(float(solved_aperture))
    result["source_targets"]=[{"joint":DRIVER,"target_rad":target["right_claw_joint_target_rad"],"source_range_rad":[-1.,0.]},
                              {"joint":FOLLOWER,"target_rad":target["R_hand_wide1_joint_target_rad"],"source_range_rad":[0.,1.]}]
    result["tcp_correction"]={"method":"solve lateral jaw-center error and aperture gap from opposing compiled contact distances; no pose sweep",
                               "source_pose_pair_distances_m":[dn,dw],"derived_lateral_translation_m":root_offset,
                               "jaw_axis_world":jaw_axis.tolist(),"source_position_limits_unchanged":True,
                               "solver_iterations":iterations,"final_pair_distances_m":list(solved_distances)}
    return result


def make_decomposition(urdf_path:Path,out:Path)->dict[str,Any]:
    meshdir=out/"x2_collision_decomp_meshes"; meshdir.mkdir(parents=True,exist_ok=True); tree=ET.parse(urdf_path); root=tree.getroot(); compiler=root.find("mujoco/compiler")
    if compiler is None: raise RuntimeError("Expected compiler meshdir in extracted source URDF")
    source_dir=X2_ROOT/X2_REL.parent/"meshes"; files=sorted({Path(m.get("filename","")).name for m in root.findall(".//mesh")})
    for f in files: shutil.copy2(source_dir/f,meshdir/f)
    compiler.set("meshdir",str(meshdir.resolve())); records=[]
    for body in CRITICAL_LINKS:
        link=next(x for x in root.findall("link") if x.get("name")==body); col=link.find("collision"); mesh=col.find("geometry/mesh"); filename=Path(mesh.get("filename")).name
        tri=stl_triangles(source_dir/filename); verts=np.unique(tri.reshape(-1,3),axis=0); _,_,vt=np.linalg.svd(verts-verts.mean(axis=0),full_matrices=False); axis=vt[0]
        proj=tri.mean(axis=1)@axis; cuts=np.linspace(float(proj.min()),float(proj.max()),7); whole=ConvexHull(verts); pieces=[]; clones=[]
        for i in range(6):
            mask=(proj>=cuts[i])&(proj<cuts[i+1] if i<5 else proj<=cuts[i+1]); pts=np.unique(tri[mask].reshape(-1,3),axis=0)
            if len(pts)<4: continue
            hull=ConvexHull(pts); hv=pts[hull.vertices]; remap={int(old):idx+1 for idx,old in enumerate(hull.vertices)}; obj=meshdir/f"{body}_piece_{i}.obj"
            with obj.open("w",encoding="ascii") as f:
                for p in hv: f.write("v %.9g %.9g %.9g\n"%tuple(p))
                for face in hull.simplices:
                    if all(int(k) in remap for k in face): f.write("f %d %d %d\n"%tuple(remap[int(k)] for k in face))
            clone=copy.deepcopy(col); g=clone.find("geometry"); g.clear(); ET.SubElement(g,"mesh",{"filename":obj.name,"scale":"1 1 1"}); clones.append(clone)
            pieces.append({"piece":i,"triangle_count":int(mask.sum()),"source_unique_vertices":int(len(pts)),"hull_vertices":int(len(hv)),"hull_faces":int(len(hull.simplices)),"hull_volume_m3":float(hull.volume),"file":obj.name,"sha256":sha(obj)})
        idx=list(link).index(col); link.remove(col)
        for k,clone in enumerate(clones): link.insert(idx+k,clone)
        records.append({"link":body,"source_mesh":filename,"source_mesh_sha256":sha(source_dir/filename),"source_triangles":len(tri),"source_unique_vertices":len(verts),
                        "principal_axis_local":axis.tolist(),"single_convex_hull_volume_m3":float(whole.volume),"sum_local_piece_volumes_m3":sum(p["hull_volume_m3"] for p in pieces),
                        "local_to_single_volume_ratio":sum(p["hull_volume_m3"] for p in pieces)/max(float(whole.volume),1e-12),"pieces":pieces})
    path=out/"isolated_omnipicker_collision_decomp.urdf"; ET.indent(root,space="  "); tree.write(path,encoding="utf-8",xml_declaration=True)
    return {"urdf":str(path),"urdf_sha256":sha(path),"meshdir":str(meshdir),
            "method":"Six convex hulls per critical source STL, formed from original source triangles binned along that STL's principal axis. Source visual meshes, kinematic joints, mount, other collision meshes, and bottle are retained. This is a collision-only diagnostic representation.","modified_links":records}


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--phase",choices=("all","reference","x2"),default="all"); args=ap.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite nonempty evidence: {out}")
    out.mkdir(parents=True,exist_ok=True); canonical=load_module("canonical_abc",CANONICAL); audit,base=modules(); ident=identity(); write_json(out/"runtime_identity.json",ident)
    result={"identity":ident,"evidence_directory":str(out)}
    if args.phase in {"all","reference"}:
        model,meta=build_reference_scene(out,canonical); result["experiment_A"]=run_reference(model,out,meta)
    if args.phase in {"all","x2"}:
        x2dir=out/"x2_source"; x2dir.mkdir(); extracted=audit.extract_subtree(x2dir,X2_ROOT/X2_REL,ROOT_LINK); candidate=candidate_data()
        result["experiment_B"]=x2_static(audit,base,candidate,Path(extracted["path"]),out,"x2_source")
        corrected=derive_tcp_centered_candidate(audit,base,Path(extracted["path"]),candidate)
        result["experiment_B_tcp_centered"]=x2_static(audit,base,corrected,Path(extracted["path"]),out,"x2_source_tcp_centered")
        decomp=make_decomposition(Path(extracted["path"]),out); result["experiment_C_collision_decomposition"]=decomp
        result["experiment_C"]=x2_static(audit,base,candidate,Path(decomp["urdf"]),out,"x2_decomposed")
        result["experiment_C_tcp_centered"]=x2_static(audit,base,corrected,Path(decomp["urdf"]),out,"x2_decomposed_tcp_centered")
    write_json(out/"result.json",result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
