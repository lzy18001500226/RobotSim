"""Fail-fast 11-position task batch using the existing full-X2 physics runner.

No motion during prepare/report. Execute never retries a failed point and never
changes the model, physical gates, MC or source controllers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg

REVIEW_ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
BRANCH = "codex/x2-autonomy-task-20261031"
ORDER = (5, 0, 10, 1, 2, 3, 4, 6, 7, 8, 9)
REQUIRED_GATES = ("NATIVE_EPISODE_KEYFRAME_INIT", "NATIVE_STATE_OBSERVER",
    "INITIAL_HANDS_ACTUALLY_OPEN", "NATIVE_TABLE_DOCK_ARRIVAL", "NATIVE_TABLE_DOCK_STOP_HOLD",
    "RGBD_TASK_GRASP_REGION", "POSTURED_PICKUP_READINESS", "HAL_BILATERAL_ARM_PREGRASP",
    "HAL_RIGHT_HAND_CLOSE", "HAL_PHYSICAL_LIFT", "POST_LIFT_RETRACT",
    "HELD_NATIVE_STAND_RESTORE", "LEAVE_TABLE_AND_TURN",
    "SECOND_TABLE_BASE_ARRIVAL_AND_STOP", "FINAL_COMPOSITE_STOP",
    "PHYSICAL_TABLE_PLACEMENT", "TWO_TABLE_TRANSFER_EXECUTION")
SOURCES = (
    "scripts/research/issue46/grasp/live_retreat_turn_review.py",
    "scripts/research/issue46/grasp/stl_approved_circle_native_trial.py",
    "scripts/research/issue46/grasp/table_edge_grid_trials.py",
    "scripts/research/issue46/native_runtime/native_task_grasp_region.py",
    "scripts/research/issue46/native_runtime/native_base_micro_probe.py",
    "scripts/research/issue46/native_runtime/xvfb_native_reset_stand.py",
    "scripts/research/issue46/native_runtime/native_grasp_stand_restore.py",
    "scripts/research/issue46/native_runtime/native_carry_placement.py",
    "scripts/research/issue46/native_runtime/standing_waist_preposition.py",
    "scripts/research/issue46/scene/table_edge_grid.py",
    "scripts/research/issue46/scene/bottle_initial_pose_overlay.py",
    "scripts/research/issue46/scene/episode_start_overlay.py",
    "scripts/research/issue46/perception/native_rgbd_grasp.py",
    "scripts/research/issue46/perception/native_rgb_video_stream.py",
)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

def source_identity():
    # Freeze the bounded active research consumer and approved gripper package,
    # including untracked source dependencies; never hash unrelated docs/data.
    paths=set(SOURCES)
    for root in (REPO/"scripts/research/issue46", REPO/"simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic"):
        paths.update(str(p.relative_to(REPO)) for p in root.rglob("*")
                     if p.is_file() and p.suffix in (".py",".c",".h",".json",".xml") and "__pycache__" not in p.parts)
    paths.add("ros2_ws/src/robotsim_navigation/robotsim_navigation/point_goal_controller.py")
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "files": {p: digest(REPO/p) for p in sorted(paths)}}

def scene_assets(scene):
    import xml.etree.ElementTree as ET
    root=ET.fromstring(scene.read_bytes())
    if root.findall("include"):
        raise ValueError("batch requires existing flattened approved scene")
    compiler=root.find("compiler")
    compiler={} if compiler is None else compiler.attrib
    result={}
    for kind,folder in (("mesh","meshdir"),("texture","texturedir")):
        for item in root.findall("asset/"+kind):
            file=item.get("file")
            if file is None:continue
            path=Path(file)
            if not path.is_absolute():path=scene.parent/compiler.get(folder,"")/path
            path=path.resolve(strict=True)
            result[str(path)]=digest(path)
    return result

def command_for(point, *, visible=False):
    argv = [sys.executable, "-B", str(REPO/"scripts/research/issue46/grasp/live_retreat_turn_review.py"),
            "--to-delivery-table", "--from-start", "--bottle-height-m", ".22", "--multiccd",
            "--task-grasp-region", "--task-bottle-offset-xy-m", *map(str,point["offset_xy_m"])]
    if not visible:
        argv.append("--no-viewer")
    return argv

def summarize_case(result, region=None):
    """Aggregate existing physical gates; absent measurements never become zero."""
    gates = result.get("gates", {})
    lift, placement = result.get("physical_lift_evidence", {}), result.get("table_placement", {})
    region = region or result.get("task_grasp_region", {})
    phases = {name:gates.get(name,"NOT_RUN") for name in REQUIRED_GATES}
    raw_lift=lift.get("minimum_lift_m")
    lift_mm=1000*raw_lift if isinstance(raw_lift,(int,float)) and not isinstance(raw_lift,bool) else None
    measures = {
        "minimum_lift_mm": lift_mm,
        "mean_upward_hand_support_n": lift.get("mean_pad_vertical_force_n"),
        "maximum_table_support_during_lift_n": lift.get("max_table_force_n"),
        "minimum_pad_normal_force_n": lift.get("minimum_pad_normal_force_n"),
        "independent_table_support_s": placement.get("measured_hold_sim_s"),
        "minimum_final_table_support_n": placement.get("minimum_table_vertical_force_n"),
        "maximum_final_pad_contact_n": placement.get("maximum_pad_normal_force_n"),
        "stance_adjustments": region.get("adjustments"),
        "stance_selection_wall_s": region.get("elapsed_wall_s"),
        "placement_wall_s": placement.get("elapsed_wall_s"),
    }
    def finite_tree(value):
        if isinstance(value,dict):return bool(value) and all(finite_tree(v) for v in value.values())
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    missing = [k for k,v in measures.items() if not finite_tree(v)]
    stage_pass = all(v=="PASS" for v in phases.values())
    evidence_pass = lift.get("pass") is True and placement.get("pass_result") is True
    success = stage_pass and evidence_pass and not missing and not result.get("failure")
    return {"status": "PASS" if success else "FAIL" if result.get("failure") or "FAIL" in phases.values() else "BLOCKED",
            "phases": phases, "measurements": measures, "missing_evidence": missing,
            "failure":result.get("failure"), "source_gate_results_preserved":True,
            "new_acceptance_thresholds":False}

def verify_task_entrypoint(argv, observation):
    """Old successful fixed-dock episodes cannot qualify this centered task."""
    if not all(flag in argv for flag in ("--execute-pickup-navigation","--task-grasp-region")):
        raise ValueError("new START/RGBD task entrypoint not used")
    try:
        i=argv.index("--dock-world-xy");xy=list(map(float,argv[i+1:i+3]))
        yaw=float(argv[argv.index("--dock-yaw-rad")+1])
    except (ValueError,IndexError,TypeError) as exc:
        raise ValueError("actual native dock arguments missing") from exc
    expected=observation["xy_m"]
    if len(xy)!=2 or not all(math.isfinite(x) for x in xy+[yaw]):
        raise ValueError("finite actual observation goal required")
    if math.dist(xy,expected)>1e-10 or abs(yaw-observation["yaw_rad"])>1e-10:
        raise ValueError("actual initial dock is not the frozen table-center observation")
    return {"pass":True,"actual_dock_xy_m":xy,"actual_dock_yaw_rad":yaw,
            "classification":"EXECUTED_ARGV_MATCH_NOT_ACTUAL_ARRIVAL_PROOF"}

def owned_episode(wrapper_stdout):
    paths = [Path(line.removeprefix("EVIDENCE ").strip()) for line in wrapper_stdout.splitlines()
             if line.startswith("EVIDENCE ")]
    if len(paths)!=1 or not paths[0].resolve(strict=True).is_relative_to(REVIEW_ROOT.resolve(strict=True)):
        raise ValueError("one exact governed wrapper evidence directory required")
    execution = json.loads((paths[0]/"execution.json").read_text())
    episodes = execution.get("episodes", [])
    if len(episodes)!=1:
        raise ValueError("one actual owned native episode required")
    episode=Path(episodes[0]).resolve(strict=True)
    if not episode.is_relative_to(REVIEW_ROOT.resolve(strict=True)):
        raise ValueError("episode outside canonical reviews")
    return paths[0],episode

def prepare(goal_output, overlay):
    import mujoco
    import numpy as np
    from scripts.research.issue46.scene.table_edge_grid import table_bounds,initial_positions,observation_goal
    if pg.detect_environment()!="wsl" or git("branch","--show-current")!=BRANCH:
        raise RuntimeError("exact WSL task branch required")
    goal_output=goal_output.resolve(strict=True)
    if not goal_output.is_relative_to(REVIEW_ROOT.resolve(strict=True)):
        raise ValueError("existing governed goal output required")
    record=json.loads(overlay.read_text())
    scene=Path(record["generated"]["host_scene"]["path"])
    if digest(scene)!=record["generated"]["host_scene"]["sha256"] or record.get("initial_object_placement"):
        raise ValueError("unchanged neutral source overlay required")
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"test_cylinder")
    if bid<0 or abs(float(m.body_mass[bid])-.2)>1e-10:
        raise ValueError("200g canonical task object required")
    cylinder=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER]
    if not cylinder or any(not np.allclose(m.geom_size[g,:2],[.035,.11],rtol=0,atol=1e-12) for g in cylinder):
        raise ValueError("D70/H220 original cylinder required")
    bounds=table_bounds(m,d);points=initial_positions(bounds,d.xpos[bid,:2])
    if len(points)!=11:
        raise ValueError("reviewed eleven-point table interval changed")
    out=pg.create_unique_output_directory(goal_output,prefix="physical-grid-batch").native_path
    manifest={"schema":"robotsim.table-edge-physical-batch.v1",
              "classification":"PREPARED_NOT_EXECUTED","source":source_identity(),
              "neutral_overlay":{"path":str(overlay),"sha256":digest(overlay)},
              "scene":{"path":str(scene),"sha256":digest(scene)},
              "referenced_asset_hashes":scene_assets(scene),
              "object":{"diameter_m":.07,"height_m":.22,"mass_kg":.2},
              "observation":observation_goal(bounds),"positions":points,
              "order":list(ORDER),"automatic_retries":0,"new_mj_step_calls":0}
    pg.write_new_text(out/"manifest.json",json.dumps(manifest,indent=2)+"\n")
    matrix={"classification":"PHYSICAL_RESULTS_ONLY_STATIC_WITNESSES_ARE_NOT_PASS",
            "cases":[{"id":p["id"],"status":"NOT_RUN"} for p in points]}
    pg.write_new_text(out/"matrix.json",json.dumps(matrix,indent=2)+"\n")
    return out

def execute(batch, *, visible_first=False):
    if pg.detect_environment()!="wsl":
        raise RuntimeError("WSL native process required")
    batch=batch.resolve(strict=True)
    if not batch.is_relative_to(REVIEW_ROOT.resolve(strict=True)):
        raise ValueError("governed batch required")
    manifest=json.loads((batch/"manifest.json").read_text())
    if source_identity()!=manifest["source"]:
        raise RuntimeError("FROZEN_BATCH_SOURCE_CHANGED; prepare a new version, do not silently mix cases")
    for key in ("scene","neutral_overlay"):
        item=manifest[key]
        if digest(item["path"])!=item["sha256"]:
            raise RuntimeError("FROZEN_BATCH_INPUT_CHANGED:"+key)
    if scene_assets(Path(manifest["scene"]["path"]))!=manifest["referenced_asset_hashes"]:
        raise RuntimeError("FROZEN_VENDOR_ASSET_CHANGED")
    matrix=json.loads((batch/"matrix.json").read_text())
    if any(p["status"]!="NOT_RUN" for p in matrix["cases"]):
        raise RuntimeError("single-execution batch; no automatic retry or skip of failed points")
    # Read-only readiness before creating any case process or requesting motion.
    state=subprocess.run(["docker","info","--format","{{.ServerVersion}}"],capture_output=True,text=True,timeout=8)
    if state.returncode:
        raise RuntimeError("DOCKER_NOT_READY:"+state.stderr[-500:])
    if subprocess.check_output(["docker","ps","-q"],text=True).strip():
        raise RuntimeError("ACTIVE_CONTAINER_PRESENT; no parallel simulator")
    for ordinal,index in enumerate(manifest["order"]):
        if source_identity()!=manifest["source"]:
            raise RuntimeError("FROZEN_BATCH_SOURCE_CHANGED_BETWEEN_CASES")
        point=manifest["positions"][index];case=batch/f"case_{index:02}";case.mkdir()
        command=command_for(point,visible=visible_first and ordinal==0)
        pg.write_new_text(case/"command.json",json.dumps({"argv":command,"purpose":"ACTUAL_NEW_NATIVE_EPISODE"},indent=2)+"\n")
        began=time.monotonic()
        with (case/"wrapper.stdout").open("x") as log:
            process=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        try:
            wrapper,episode=owned_episode((case/"wrapper.stdout").read_text())
            result=json.loads((episode/"result.json").read_text())
            invocation=json.loads((episode/"invocation.json").read_text())
            entrypoint=verify_task_entrypoint(invocation["argv"],manifest["observation"])
            overlay=json.loads(Path(result["claw_control_overlay"]["path"]).read_text())
            actual_offset=overlay["initial_object_placement"]["offset_xy_m"]
            if math.dist(actual_offset,point["offset_xy_m"])>1e-10:
                raise ValueError("executed initial scenario does not match this grid point")
            summary=summarize_case(result)
            summary["executed_task_entrypoint"]=entrypoint
            summary.update(wrapper=str(wrapper),episode=str(episode),result_sha256=digest(episode/"result.json"))
        except (OSError,ValueError,KeyError) as exc:
            summary={"status":"BLOCKED","failure":"OWNED_PHYSICS_EVIDENCE_MISSING:"+str(exc)}
        summary.update(id=index,returncode=process.returncode,wall_s=time.monotonic()-began)
        if process.returncode and summary["status"]=="PASS":
            summary.update(status="BLOCKED",failure="RUNNER_NONZERO_DESPITE_PASS_FIELDS")
        pg.write_new_text(case/"result.json",json.dumps(summary,indent=2)+"\n")
        matrix["cases"][index]=summary
        temporary=batch/"matrix.tmp";temporary.write_text(json.dumps(matrix,indent=2)+"\n")
        temporary.replace(batch/"matrix.json")
        print(json.dumps({"id":index,"status":summary["status"],"episode":summary.get("episode")}),flush=True)
        if summary["status"]!="PASS":
            raise RuntimeError("STOP_AT_FIRST_CASE_FAILURE:"+str(index))
    return matrix

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("prepare");p.add_argument("--goal-output",type=Path,required=True);p.add_argument("--neutral-overlay",type=Path,required=True)
    p=sub.add_parser("execute");p.add_argument("--batch",type=Path,required=True);p.add_argument("--visible-first",action="store_true")
    args=parser.parse_args()
    if args.mode=="prepare":
        print(prepare(args.goal_output,args.neutral_overlay),flush=True)
    else:
        execute(args.batch,visible_first=args.visible_first)

if __name__=="__main__":main()
