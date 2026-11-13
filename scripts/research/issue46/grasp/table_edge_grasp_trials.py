"""Independent RGB-D grasp/lift grid trials; reuse the existing physical runner.

Research disposition: RETAIN_RESEARCH_DIAGNOSTIC. No second physics backend.
"""
from __future__ import annotations
import argparse,json,math,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.grasp import table_edge_grid_trials as base
REQUIRED=("NATIVE_FIXED_STANCE_STATE_HANDOFF","RGBD_TASK_GRASP_REGION",
          "POSTURED_PICKUP_READINESS","HAL_RIGHT_HAND_CLOSE","HAL_PHYSICAL_LIFT",
          "POST_LIFT_RETRACT","GRASP_ONLY_STATIC_HOLD")

def summarize(result, *,from_start=False):
    required=tuple("NATIVE_TABLE_DOCK_STATE_HANDOFF" if k=="NATIVE_FIXED_STANCE_STATE_HANDOFF" and from_start else k for k in REQUIRED)
    if from_start:required+=("NATIVE_EPISODE_KEYFRAME_INIT","NATIVE_TABLE_DOCK_ARRIVAL","NATIVE_TABLE_DOCK_STOP_HOLD","INITIAL_SDK_LEG_COMMAND_RECEIPT","INITIAL_NATIVE_PHYSICAL_STANDING")
    gates=result.get("gates",{});lift=result.get("physical_lift_evidence",{})
    hold=result.get("carry_static_hold",{});missing=[]
    values={"minimum_lift_mm":None if lift.get("minimum_lift_m") is None else 1000*lift["minimum_lift_m"],
            "upward_hand_support_n":lift.get("mean_pad_vertical_force_n"),
            "table_support_n":lift.get("max_table_force_n"),
            "pad_normal_force_n":lift.get("minimum_pad_normal_force_n"),
            "terminal_lift_hold_s":lift.get("terminal_hold_sim_s"),
            "static_hold_s":hold.get("sim_span_s")}
    for key,value in values.items():
        entries=value.values() if isinstance(value,dict) else [value]
        if value is None or isinstance(value,dict) and not value or any(not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) for v in entries):missing.append(key)
    physical_pass=lift.get("pass") is True and gates.get("HAL_RIGHT_HAND_CLOSE")=="PASS"
    passed=physical_pass and all(gates.get(k)=="PASS" for k in required) and not missing and not result.get("failure")
    return {"status":"PASS" if passed else "FAIL","grasp_lift":"PASS" if physical_pass else gates.get("HAL_PHYSICAL_LIFT","NOT_RUN"),
            "stages":{k:gates.get(k,"NOT_RUN") for k in required},"measurements":values,
            "missing_evidence":missing,"failure":result.get("failure"),
            "full_transfer":"NOT_RUN","hardware":"NOT_RUN"}

def run_case(output,index,visible=False,far_observe=False):
    output=output.resolve(strict=True)
    if not output.is_relative_to(base.REVIEW_ROOT.resolve(strict=True)) or not (output/"task.json").is_file():raise ValueError("owned governed task root required")
    manifest=json.loads((output/"grid_inputs.json").read_text());point=manifest["positions"][index]
    if base.digest(manifest["scene"]["path"])!=manifest["scene"]["sha256"]:raise ValueError("frozen geometry changed")
    case=pg.create_unique_output_directory(output,prefix=f"point{index:02}").native_path
    pg.write_new_bytes(case/"grid_inputs.json",(output/"grid_inputs.json").read_bytes())
    argv=[sys.executable,"-B",str(REPO/"scripts/research/issue46/grasp/live_retreat_turn_review.py"),
          "--to-delivery-table","--bottle-height-m",".22","--multiccd","--task-grasp-region","--grasp-only",
          "--task-bottle-offset-xy-m",*map(str,point["offset_xy_m"])]
    if far_observe:argv.extend(["--from-start","--far-observe-approach"])
    if not visible:argv.append("--no-viewer")
    identity=base.source_identity()
    import tarfile
    with tarfile.open(case/"executed_source.tar.gz","x:gz") as archive:
        for rel in identity["files"]:archive.add(REPO/rel,arcname=rel)
    pg.write_new_text(case/"command.json",json.dumps({"argv":argv,"cwd":str(REPO),"source":identity,"initialization":point,"scope":"FULL_START_FAR_RGBD_DIAGONAL_GRASP_ONLY" if far_observe else "GRASP_ONLY_NO_START_NAVIGATION_OR_TRANSFER"},indent=2)+"\n")
    began=time.monotonic()
    with (case/"wrapper.stdout").open("x") as stream:
        process=subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    try:
        wrapper,episode=base.owned_episode((case/"wrapper.stdout").read_text())
        result=json.loads((episode/"result.json").read_text());overlay=json.loads(Path(result["claw_control_overlay"]["path"]).read_text())
        if math.dist(overlay["initial_object_placement"]["offset_xy_m"],point["offset_xy_m"])>1e-10:raise ValueError("wrong grid initialization")
        init=overlay["episode_start_navigation"]
        if far_observe:
            if init.get("observation_only") or math.dist(init["start_xy_m"],[-1.5,-1.5])>1e-10:raise ValueError("full START initialization absent")
            base.verify_task_entrypoint(json.loads((episode/"invocation.json").read_text())["argv"],manifest["observation"])
        elif init.get("observation_only") is not True or math.dist(init["start_xy_m"],manifest["observation"]["xy_m"])>1e-10:raise ValueError("wrong table-center initialization")
        record=summarize(result,from_start=far_observe);record.update(episode=str(episode),wrapper=str(wrapper),result_sha256=base.digest(episode/"result.json"))
    except (OSError,ValueError,KeyError) as exc:
        record={"status":"BLOCKED","failure":str(exc),"grasp_lift":"NOT_RUN"}
    after=base.source_identity()
    changed=[p for p,h in identity["files"].items() if after["files"].get(p)!=h]
    extra=set(after["files"])-set(identity["files"])
    if after["branch"]!=identity["branch"] or after["head"]!=identity["head"] or changed or any(not Path(p).name.startswith("test_") for p in extra):
        record.update(status="BLOCKED",failure="SOURCE_CHANGED_DURING_RUN")
    record["source_verification"]={"changed_original_files":changed,"extra_offline_tests":sorted(extra)}
    if process.returncode and record["status"]=="PASS":record.update(status="BLOCKED",failure="RUNNER_EXIT_NONZERO")
    record.update(id=index,returncode=process.returncode,wall_s=time.monotonic()-began,case=str(case))
    pg.write_new_text(case/"result.json",json.dumps(record,indent=2)+"\n");pg.write_artifact_manifest(case)
    matrix_path=output/"matrix.json"
    matrix=json.loads(matrix_path.read_text()) if matrix_path.exists() else {"scope":"11 independent 200g physical grasps, not complete transfers","cases":[{"id":i,"status":"NOT_RUN"} for i in range(11)],"attempts":[]}
    matrix["attempts"].append(record);matrix["cases"][index]=record
    pg.write_reserved_text(output,matrix_path,json.dumps(matrix,indent=2)+"\n",overwrite_existing=True)
    print(json.dumps(record),flush=True)
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--case",type=int,choices=range(11),required=True);p.add_argument("--visible",action="store_true");p.add_argument("--far-observe",action="store_true");a=p.parse_args()
    r=run_case(a.output_dir,a.case,a.visible,a.far_observe)
    if r["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
