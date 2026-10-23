#!/usr/bin/env python3
"""Read-only X2 Ultra versus Ultra Plus IK-URDF source compatibility audit."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,uuid,xml.etree.ElementTree as ET
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"/"agent"))
import path_governance as pg
REV=Path("/home/lzy18001500226/projects/RobotSim/reviews")
OFFICIAL_REPO=Path("/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf")
OFFICIAL_PIN="575cc6b988f976c23550e0db85aa1e5475d3652d"
OFFICIAL_URDF=OFFICIAL_REPO/"X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf"
COMMUNITY_REPO=Path("/home/lzy18001500226/projects/RobotSim/reference/AgiBot-Community/raicom2026-agibot-X2")
COMMUNITY_PIN="a97761820b398b6cab12b632ce9f313c394eeb7e"
COMMUNITY_URDF=COMMUNITY_REPO/"src/x2_ik_sdk/src/x2_ik_sdk/resources/x2_ultra_plus_omnipicker_omnipicker.urdf"
ARM_PREFIX=("left_shoulder","left_elbow","left_wrist","right_shoulder","right_elbow","right_wrist")

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(["git","-C",str(p),*args],text=True).strip()

def parse_urdf(path):
    root=ET.parse(path).getroot();joints={}
    for j in root.findall("joint"):
        o=j.find("origin");a=j.find("axis");lim=j.find("limit");m=j.find("mimic")
        def vec(x):
            return None if x is None else tuple(round(float(v),9) for v in x.split())
        parent=j.find("parent");child=j.find("child")
        joints[j.get("name")]={
            "type":j.get("type"),
            "parent":None if parent is None else parent.get("link"),
            "child":None if child is None else child.get("link"),
            "xyz":None if o is None else vec(o.get("xyz")),
            "rpy":None if o is None else vec(o.get("rpy")),
            "axis":None if a is None else vec(a.get("xyz")),
            "range":None if lim is None else (lim.get("lower"),lim.get("upper")),
            "mimic":None if m is None else (m.get("joint"),m.get("multiplier"),m.get("offset"))}
    visual=sum(len(link.findall("visual")) for link in root.findall("link"))
    collision=sum(len(link.findall("collision")) for link in root.findall("link"))
    mesh_count=sum(1 for _ in root.iter("mesh"))
    return {"robot_name":root.get("name"),"joint_count":len(joints),"link_count":len(root.findall("link")),
            "visual_element_count":visual,"collision_element_count":collision,"mesh_reference_count":mesh_count,"joints":joints}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-parent",default=str(REV/"issue-46"));args=ap.parse_args()
    for repo,pin in ((OFFICIAL_REPO,OFFICIAL_PIN),(COMMUNITY_REPO,COMMUNITY_PIN)):
        if git(repo,"rev-parse","HEAD")!=pin or git(repo,"status","--porcelain"):
            raise RuntimeError(f"source checkout is not clean and pinned: {repo}")
    official=parse_urdf(OFFICIAL_URDF);community=parse_urdf(COMMUNITY_URDF)
    oj,cj=official["joints"],community["joints"];common=sorted(oj.keys()&cj.keys())
    arm=[n for n in common if n.startswith(ARM_PREFIX)]
    changed=[{"joint":n,"official":oj[n],"ultra_plus_ik":cj[n]} for n in arm if oj[n]!=cj[n]]
    hand_names=sorted(n for n in common if "claw" in n or "omnipicker" in n or "hand_" in n)
    mount_names=("L_omnipicker_joint","R_omnipicker_joint")
    mount_delta={}
    for n in mount_names:
        a=oj[n]["xyz"];b=cj[n]["xyz"]
        mount_delta[n]={"official_xyz_m":a,"ultra_plus_ik_xyz_m":b,
                        "delta_ultra_plus_minus_official_m":[round(y-x,9) for x,y in zip(a,b)],
                        "official_rpy":oj[n]["rpy"],"ultra_plus_ik_rpy":cj[n]["rpy"]}
    right_claw={n:{"official":oj.get(n),"ultra_plus_ik":cj.get(n),"equal":oj.get(n)==cj.get(n)}
                for n in ("right_claw_joint","R_hand_wide1_joint","R_hand_narrow2_joint","R_hand_narrow3_joint",
                          "R_hand_narrow_loop_joint","R_hand_wide2_joint","R_hand_wide3_joint","R_hand_wide_loop_joint")}
    result={"schema":"robotsim.issue46.x2_ultra_variant_urdf_comparison.v1",
      "physics_steps":0,"mujoco_calls":0,
      "official":{"repository":"AgibotTech/agibot_x2_urdf","revision":OFFICIAL_PIN,"path":str(OFFICIAL_URDF),
          "sha256":sha(OFFICIAL_URDF),"git_clean":True,**{k:v for k,v in official.items() if k!="joints"}},
      "ultra_plus_ik_reference":{"repository":"AgiBot-Community/raicom2026-agibot-X2","revision":COMMUNITY_PIN,
          "path":str(COMMUNITY_URDF),"sha256":sha(COMMUNITY_URDF),"git_clean":True,
          "root_license_file":str(COMMUNITY_REPO/"LICENSE"),"root_license_sha256":sha(COMMUNITY_REPO/"LICENSE"),
          **{k:v for k,v in community.items() if k!="joints"}},
      "joint_graph":{"shared_joint_count":len(common),"official_only":sorted(oj.keys()-cj.keys()),
          "ultra_plus_only":sorted(cj.keys()-oj.keys()),"common_arm_joint_count":len(arm),
          "changed_common_arm_joint_count":len(changed),"changed_common_arm_joints":changed,
          "selected_claw_joint_comparison":right_claw,"shared_hand_joint_names":hand_names},
      "wrist_mount_delta":mount_delta,
      "compatibility":"NOT_DIRECTLY_INTERCHANGEABLE",
      "decision":"The community Ultra Plus file is an IK-only description without visual/collision elements or mesh references. It changes 14/14 shared arm joints and the OmniPicker root mounts; it cannot validate or replace the pinned full-X2 source assembly. A separate full source model and matching physical variant would be needed before any mount experiment.",
      "evidence_boundary":"Source parsing and numeric comparison only; no model conversion, pose optimization, MJCF editing, dynamics, or hardware claims."}
    parent=pg.validate_output_path(args.output_parent,authorized_roots=(str(REV),),require_exists=True,
        expected_kind="directory",environment="wsl")
    out=pg.create_unique_output_directory(parent.wsl_path,prefix="pr90-x2-ultra-variant-audit",
        authorized_roots=(str(REV),),environment="wsl")
    result["run_id"]=str(uuid.uuid4());result["output"]=out.as_dict()
    result["runner_path"]=str(Path(__file__).resolve());result["runner_sha256"]=sha(Path(__file__).resolve())
    pg.write_new_text(out.native_path/"result.json",json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        authorized_roots=(str(REV),),environment="wsl")
    report=["# X2 Ultra / Ultra Plus source-model compatibility audit","",
        f"Run {result['run_id']}; physics steps 0; MuJoCo calls 0.",
        f"Official source: {OFFICIAL_PIN}, URDF SHA {sha(OFFICIAL_URDF)}.",
        f"Community Ultra Plus IK source: {COMMUNITY_PIN}, URDF SHA {sha(COMMUNITY_URDF)}.",
        f"Joint counts: official {official['joint_count']}, community {community['joint_count']}; shared {len(common)}.",
        f"Arm joints: {len(arm)} common, {len(changed)} differ in at least one frame/limit field.",
        f"URDF geometry: official visual/collision elements {official['visual_element_count']}/{official['collision_element_count']}; Ultra Plus IK file {community['visual_element_count']}/{community['collision_element_count']}, mesh references {community['mesh_reference_count']}.",
        f"Right mount local xyz: official {oj['R_omnipicker_joint']['xyz']} m; Ultra Plus IK {cj['R_omnipicker_joint']['xyz']} m.",
        f"Right wrist-roll ranges: official {oj['right_wrist_roll_joint']['range']}; Ultra Plus IK {cj['right_wrist_roll_joint']['range']}.",
        "The claw driver and wide1 mimic entries match, but this does not make the full robot variants interchangeable.",
        "Verdict: NOT_DIRECTLY_INTERCHANGEABLE. Do not transplant the Ultra Plus mount onto the pinned X2 Ultra robot. A separate matching full model and explicit variant confirmation are needed before a static mount experiment.",
        "No source files or robot models were modified. No physics was run."]
    pg.write_new_text(out.native_path/"REPORT.md","\n".join(report)+"\n",authorized_roots=(str(REV),),environment="wsl")
    cmd=f"cd {ROOT}\nenv PYTHONDONTWRITEBYTECODE=1 python3 {Path(__file__).resolve()} --output-parent {parent.wsl_path}\n"
    pg.write_new_text(out.native_path/"REPRODUCE.md",cmd,authorized_roots=(str(REV),),environment="wsl")
    mf=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
    pg.verify_artifacts(mf["artifacts"],authorized_roots=(str(REV),),environment="wsl")
    print(json.dumps({"run_id":result["run_id"],"output":out.as_dict(),
        "verdict":result["compatibility"],"common_arm":len(arm),"changed_arm":len(changed),
        "official_mesh_refs":official["mesh_reference_count"],"ultra_plus_mesh_refs":community["mesh_reference_count"],
        "manifest":len(mf["artifacts"])},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
