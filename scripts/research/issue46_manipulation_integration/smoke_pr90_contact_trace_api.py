#!/usr/bin/env python3
"""Zero-step API smoke for PR90 contact tracing on the exact compiled full-X2 model."""
from __future__ import annotations
import hashlib,json,sys,uuid,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"/"research"))
sys.path.insert(0,str(ROOT/"scripts"/"agent"))
import path_governance as pg
import issue46_manipulation_integration.audit_pr90_full_x2_static as base
from issue46_manipulation_integration.pr90_contact_trace import collect_pr90_contact_frame
REV=base.REV
COARSE=REV/"issue-46/pr90-full-x2-bounded-24-static-20261023T180937Z"
REFINE=REV/"issue-46/pr90-full-x2-lateral-7-refine-static-20261023T181534Z"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    cf=json.loads((COARSE/"result.json").read_text());rf=json.loads((REFINE/"result.json").read_text())
    for p in (COARSE,REFINE):
        mf=json.loads((p/"artifact_paths.json").read_text())
        pg.verify_artifacts(mf["artifacts"],authorized_roots=(str(REV),),environment="wsl")
    cand=next(x for x in rf["candidates"] if x["candidate"]["id"]=="SIDE_Y6_X+0")
    model_path=COARSE/"full_x2_pr90_bounded_static.xml"
    if sha(model_path)!=cf["generated_full_x2_model_sha256"]:raise RuntimeError("saved compiled-source MJCF SHA mismatch")
    m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
    if mujoco.__version__!="3.3.6":raise RuntimeError("runtime version mismatch")
    obj=base.gid(m,base.OBJ);ob=int(m.geom_bodyid[obj]);oq=int(m.jnt_qposadr[base.jid(m,base.OJ)])
    aq=tuple(int(m.jnt_qposadr[base.jid(m,n)]) for n in base.ARM)
    fr=m.geom_friction[obj].copy();open_hand=base.profile()[-1]["q"]
    original_forward=mujoco.mj_forward;original_steps=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2)
    fw=0;steps=0
    def forward(*a,**k):
        nonlocal fw;fw+=1;return original_forward(*a,**k)
    def forbidden(*a,**k):
        nonlocal steps;steps+=1;raise RuntimeError("physics stepping prohibited in contact-trace API smoke")
    mujoco.mj_forward=forward;mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
    try:
        # One static ground-truth setup only; no active rollout follows.
        base.state(m,d,cand["q_approach_arm"],open_hand,oq,aq,obj,ob,fr)
        trace=collect_pr90_contact_frame(m,d,base.OBJ)
        required={"contacts","right_hand_support_world_N","table_reaction_on_object_world_N","object_pose_world"}
        if not required.issubset(trace):raise RuntimeError("contact collector schema missing required fields")
        for c in trace["contacts"]:
            if not {"geom0","geom1","body0","body1","active","force_on_geom2_contact_frame_N",
                    "force_on_object_world_N","relative_slip_speed_m_s","point_object_local_m"}.issubset(c):
                raise RuntimeError("contact row schema incomplete")
            vals=np.asarray(c["force_on_object_world_N"]+c["point_object_local_m"],dtype=float)
            if not np.all(np.isfinite(vals)):raise RuntimeError("nonfinite collector output")
        if steps!=0:raise RuntimeError("unexpected physics step")
        pairs=[{"geom0":x["geom0"],"geom1":x["geom1"],"body0":x["body0"],"body1":x["body1"],
                "role":x["role"],"active_contact":x["active"],"distance_mm":x["distance_m"]*1000.0}
               for x in trace["contacts"]]
        summary={"schema":"robotsim.issue46.pr90_contact_trace_api_smoke.v1","run_id":str(uuid.uuid4()),
            "classification":"STATIC_API_SMOKE_ONLY_NOT_FORCE_OR_GRASP_EVIDENCE",
            "mujoco":"3.3.6","model_sha256":sha(model_path),"source_static_run_id":cf["run_id"],
            "source_refinement_run_id":rf["run_id"],"candidate_id":"SIDE_Y6_X+0",
            "physics_steps":steps,"mj_forward_calls":fw,"contact_count":trace["contact_count"],
            "active_contact_count":sum(x["active_contact"] for x in pairs),"contact_pair_inventory":pairs,
            "collector_fields_tested":["force_on_geom2_contact_frame_N","force_on_object_world_N",
                 "normal_force_N","tangential_force_on_object_world_N","relative_slip_speed_m_s",
                 "right_hand_support_world_N","table_reaction_on_object_world_N","object_pose_world"],
            "static_force_warning":"mj_forward output here is not an equilibrium/load-transfer measurement. Contact identities and API field plumbing only.",
            "limitations":"The tested state is already statically blocked and penetrated. No controller, rollout, mj_step, or physical force acceptance was performed."}
    finally:
        mujoco.mj_forward=original_forward
        mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original_steps
    out=pg.create_unique_output_directory(str(REV/"issue-46"),prefix="pr90-contact-trace-api-smoke",
        authorized_roots=(str(REV),),environment="wsl")
    summary["output"]=out.as_dict();summary["runner_path"]=str(Path(__file__).resolve());summary["runner_sha256"]=sha(Path(__file__).resolve())
    pg.write_new_text(out.native_path/"result.json",json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        authorized_roots=(str(REV),),environment="wsl")
    report=["# PR90 contact-trace API smoke","",f"Run {summary['run_id']}; output {out.wsl_path}; UNC {out.windows_unc}.",
        "This was one static qpos setup and mj_forward only: physics_steps=0.",
        f"Compiled full-X2 model SHA: {summary['model_sha256']}; MuJoCo {summary['mujoco']}.",
        f"The collector returned {summary['contact_count']} contact records, of which {summary['active_contact_count']} were active in the static setup.",
        "The report keeps only contact identities, active flags, and distances. Solver force values from this static overlap are not saved or interpreted as physical support.",
        "No grasp, actuator motion, object lift, or load-transfer result is claimed."]
    pg.write_new_text(out.native_path/"REPORT.md","\n".join(report)+"\n",authorized_roots=(str(REV),),environment="wsl")
    command=f"cd {ROOT}\nenv PYTHONDONTWRITEBYTECODE=1 {Path(sys.executable)} {Path(__file__).resolve()}\n"
    pg.write_new_text(out.native_path/"REPRODUCE.md",command,authorized_roots=(str(REV),),environment="wsl")
    mf=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
    pg.verify_artifacts(mf["artifacts"],authorized_roots=(str(REV),),environment="wsl")
    print(json.dumps({"run_id":summary["run_id"],"output":out.as_dict(),"steps":steps,"forwards":fw,
        "contacts":summary["contact_count"],"active":summary["active_contact_count"],"manifest":len(mf["artifacts"])},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
