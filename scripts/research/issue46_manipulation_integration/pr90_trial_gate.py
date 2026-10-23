#!/usr/bin/env python3
"""Fail-closed authorization check before any full-X2 PR90 dynamic trial."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"/"agent"))
import path_governance as pg

REV=Path("/home/lzy18001500226/projects/RobotSim/reviews")
PR90_HEAD="facdaa0d6e5f8152168d2e812d864b660eac9f96"
PR90_MODEL_SHA="32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45"
URDF_PIN="575cc6b988f976c23550e0db85aa1e5475d3652d"
URDF_SHA="35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d"
EXPECTED_OBJECT={"diameter_m":0.070,"height_m":0.230,"mass_kg":0.550}
CLEARANCE_MM=2.0

def evaluate_trial_authorization(results)->dict:
    if isinstance(results,dict):results=[results]
    global_errors=[];rows=[]
    for result in results:
        run_id=result.get("run_id","UNKNOWN")
        if result.get("classification") not in ("SIMULATION_ONLY_STATIC_GEOMETRY_AND_IK","SIMULATION_ONLY_STATIC_GEOMETRY"):
            global_errors.append(f"{run_id}:INPUT_NOT_STATIC_GEOMETRY_SCREEN")
        if result.get("physics_steps")!=0:global_errors.append(f"{run_id}:STATIC_SCREEN_HAS_NONZERO_PHYSICS_STEPS")
        version=result.get("mujoco",result.get("Mujoco"))
        if version!="3.3.6":global_errors.append(f"{run_id}:MUJOCO_VERSION_MISMATCH")
        if result.get("PR90_head")!=PR90_HEAD or result.get("PR90_model_sha256")!=PR90_MODEL_SHA:
            global_errors.append(f"{run_id}:PR90_MODEL_IDENTITY_MISMATCH")
        if result.get("official_URDF_sha256")!=URDF_SHA:global_errors.append(f"{run_id}:OFFICIAL_X2_URDF_SHA_MISMATCH")
        if result.get("official_URDF_revision") not in (None,URDF_PIN):
            global_errors.append(f"{run_id}:OFFICIAL_X2_URDF_REVISION_MISMATCH")
        obj=result.get("object") or {}
        aliases={"diameter_m":("diameter_m","D_m"),"height_m":("height_m","H_m"),"mass_kg":("mass_kg",)}
        for key,names in aliases.items():
            try:value=next(float(obj[n]) for n in names if n in obj)
            except (StopIteration,TypeError,ValueError):value=None
            if value is None or abs(value-EXPECTED_OBJECT[key])>1e-12:
                global_errors.append(f"{run_id}:OBJECT_{key.upper()}_MISMATCH")
        refined={x.get("candidate_id"):x for x in result.get("refined_closures",[])}
        for entry in result.get("candidates",[]):
            c=entry.get("candidate") or {};cid=c.get("id","UNKNOWN");block=[]
            endpoint=entry.get("endpoint_open_clear")
            if endpoint is None:endpoint=(entry.get("endpoint_open") or {}).get("pass")
            if entry.get("ik_status")!="PASS_STATIC_IK":block.append("SOURCE_LIMITED_IK_NOT_PASS")
            if entry.get("arm_limits_pass") is not True:block.append("ARM_LIMIT_MARGIN_NOT_PASS")
            if entry.get("path_status")!="PASS_STATIC_SWEPT":block.append("SWEPT_APPROACH_NOT_PROVED_CLEAR")
            try:
                if float(entry.get("minimum_path_clearance_mm"))<CLEARANCE_MM:block.append("PATH_CLEARANCE_BELOW_2MM")
            except (TypeError,ValueError):block.append("PATH_CLEARANCE_MISSING")
            if endpoint is not True:block.append("OPEN_ENDPOINT_NOT_CLEAR")
            close=refined.get(cid) or entry.get("closure")
            if close is None:block.append("CLOSURE_STATIC_GATE_NOT_RUN")
            else:
                bilateral=close.get("first_bilateral_in_contact_band",close.get("first_bilateral"))
                padfirst=close.get("pad_first_topology",close.get("pad_first"))
                status=close.get("closure_status")
                if status is None:status="PASS_STATIC_PAD_FIRST" if padfirst else "BLOCKED_STATIC_CLOSURE"
                if status!="PASS_STATIC_PAD_FIRST":block.append("PAD_FIRST_CLOSURE_NOT_PASS")
                if padfirst is not True:block.append("BILATERAL_PAD_FIRST_TOPOLOGY_NOT_PASS")
                if bilateral is None:block.append("NO_BILATERAL_CONTACT_IN_BAND")
                if close.get("first_nonpad_object_contact",close.get("first_nonpad")) is not None:
                    block.append("NONPAD_CONTACT_PRECEDES_BILATERAL")
                if close.get("first_internal_self_contact",close.get("first_internal_contact")) is not None:
                    block.append("INTERNAL_SELF_CONTACT_BEFORE_BILATERAL")
                meshes=(close.get("source_meshes_at_first_bilateral") or
                        close.get("source_meshes_at_bilateral") or
                        close.get("source_meshes_open") or close.get("source_open") or [])
                if bilateral is not None and not (close.get("source_meshes_at_first_bilateral") or close.get("source_meshes_at_bilateral")):
                    block.append("SOURCE_STL_AT_CONTACT_NOT_CHECKED")
                if meshes and any(m.get("source_surface",{}).get("intersects_cylinder_volume") for m in meshes):
                    block.append("ORIGINAL_SOURCE_STL_INTERSECTS_OBJECT")
                if not meshes:block.append("SOURCE_STL_GEOMETRY_NOT_CHECKED")
            rows.append({"source_run_id":run_id,"candidate_id":cid,"authorized":not block,"blockers":block})
    if not rows:global_errors.append("NO_CANDIDATES")
    allowed=not global_errors and any(x["authorized"] for x in rows)
    return {"schema":"robotsim.issue46.pr90_dynamic_trial_authorization.v1",
        "dynamic_trial_authorized":allowed,"status":"STATIC_GATES_PASS" if allowed else "BLOCKED_STATIC_GATES",
        "global_blockers":global_errors,"candidate_decisions":rows,
        "limits":{"mujoco":"3.3.6","max_simulated_seconds":30.0,"max_physics_steps":30000,"clearance_mm":CLEARANCE_MM},
        "safe_termination_conditions":["nonfinite state or solver warning",
          "source joint position, velocity, or actuator safety gate violation",
          "loss of bilateral pad contact during hold or lift",
          "unexpected nonpad contact or bottle slip outside measured tolerance",
          "bottle table reaction does not unload before lift",
          "30 simulated seconds or 30000 physics steps, whichever occurs first"],
        "claim_boundary":"Authorization preflight only. It performs no model stepping and proves no physical grasp or lift."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--static-result",action="append",required=True)
    ap.add_argument("--output-parent",default=str(REV/"issue-46"))
    args=ap.parse_args()
    sources=[];source_records=[]
    for value in args.static_result:
        src=pg.validate_output_path(value,authorized_roots=(str(REV),),require_exists=True,
            expected_kind="file",environment="wsl")
        manifest_path=src.native_path.parent/"artifact_paths.json"
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text())
            pg.verify_artifacts(manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
        sources.append(json.loads(src.native_path.read_text()))
        source_records.append({"path":src.wsl_path,"sha256":hashlib.sha256(src.native_path.read_bytes()).hexdigest()})
    gate=evaluate_trial_authorization(sources)
    parent=pg.validate_output_path(args.output_parent,authorized_roots=(str(REV),),require_exists=True,
        expected_kind="directory",environment="wsl")
    out=pg.create_unique_output_directory(parent.wsl_path,prefix="pr90-dynamic-trial-gate",
        authorized_roots=(str(REV),),environment="wsl")
    gate.update({"run_id":str(uuid.uuid4()),"static_results":source_records,"output":out.as_dict(),
        "runner_path":str(Path(__file__).resolve()),
        "runner_sha256":hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()})
    pg.write_new_text(out.native_path/"result.json",json.dumps(gate,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        authorized_roots=(str(REV),),environment="wsl")
    lines=["# PR90 dynamic trial authorization gate","",
        f"Run ID: {gate['run_id']}",f"Static inputs: {source_records}",
        f"Decision: {gate['status']}",f"Dynamic trial authorized: {gate['dynamic_trial_authorized']}","",
        "This preflight does not load/step MuJoCo. A dynamic bottle trial is only allowed when one candidate passes all source, IK, swept approach, open-state, source-mesh, and bilateral pad-first closure gates."]
    for c in gate["candidate_decisions"]:
        lines.append(f"- {c['candidate_id']}: {'AUTHORIZED' if c['authorized'] else 'BLOCKED'}; blockers={c['blockers']}")
    pg.write_new_text(out.native_path/"REPORT.md","\n".join(lines)+"\n",authorized_roots=(str(REV),),environment="wsl")
    args=" ".join(f"--static-result {x['path']}" for x in source_records)
    command=f"cd {ROOT}\nenv PYTHONDONTWRITEBYTECODE=1 python3 scripts/research/issue46_manipulation_integration/pr90_trial_gate.py {args}\n"
    pg.write_new_text(out.native_path/"REPRODUCE.md",command,authorized_roots=(str(REV),),environment="wsl")
    manifest=pg.write_artifact_manifest(out.native_path,authorized_roots=(str(REV),),environment="wsl")
    pg.verify_artifacts(manifest["artifacts"],authorized_roots=(str(REV),),environment="wsl")
    print(json.dumps({"status":gate["status"],"dynamic_trial_authorized":gate["dynamic_trial_authorized"],
        "run_id":gate["run_id"],"output":out.as_dict(),"manifest_count":len(manifest["artifacts"])},indent=2))
    return 0 if not gate["dynamic_trial_authorized"] else 0

if __name__=="__main__":raise SystemExit(main())
