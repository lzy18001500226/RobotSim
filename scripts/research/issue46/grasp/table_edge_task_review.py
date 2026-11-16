"""Review saved table-edge experiments. Never starts MC or calls mj_step."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-46")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def summarize(run):
    path = run/"result.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    native = json.loads((run/"native_latest.json").read_text()) if (run/"native_latest.json").exists() else {}
    lift = result.get("physical_lift_evidence", {})
    gates = result.get("gates", {})
    region = json.loads((run/"visual_stance_result.json").read_text()) if (run/"visual_stance_result.json").exists() else {}
    offset = None
    overlay = result.get("claw_control_overlay", {}).get("path")
    if overlay:
        record = json.loads(Path(overlay).read_text())
        offset = record.get("initial_object_placement", {}).get("offset_xy_m", [0., 0.])
    failure = result.get("failure")
    message = None
    if isinstance(failure, dict):
        message = failure.get("message") or failure.get("gate")
        detail = failure.get("detail")
        if isinstance(detail, dict):
            message = str(message)+": "+str(detail.get("recorded_native_failure") or detail.get("failure"))
    phase = {k: gates.get(k, "NOT_RUN") for k in (
        "HAL_PHYSICAL_LIFT", "POST_LIFT_RETRACT", "LEAVE_TABLE_AND_TURN",
        "SECOND_TABLE_BASE_ARRIVAL_AND_STOP", "PHYSICAL_TABLE_PLACEMENT")}
    if phase["LEAVE_TABLE_AND_TURN"] != "PASS":
        phase["SECOND_TABLE_BASE_ARRIVAL_AND_STOP"] = "NOT_RUN"
    images = []
    for p in sorted(ROOT.glob("native-run-end-capture-*/result.json"), reverse=True):
        try:
            capture = json.loads(p.read_text())
            if capture.get("run") == str(run):
                images = ["../"+str(Path(x).relative_to(ROOT)) for x in capture["images"]]
                break
        except (OSError, ValueError, KeyError):
            continue
    trail = []
    max_speed = 0.
    source = run/"native_physics.jsonl"
    if source.exists():
        for i, line in enumerate(source.open()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row["generation"] != 1:
                continue
            body = row["bodies"]["pelvis"]
            velocity = body["origin_velocity_world_angular_linear"]
            max_speed = max(max_speed, math.hypot(velocity[3], velocity[4]))
            if i % 25 == 0:
                trail.append(body["position_m"][:2])
    success = gates.get("TWO_TABLE_TRANSFER_EXECUTION") == "PASS"
    reused = run.name.endswith("20261111T004845Z")
    return {"run": run.name, "reused_baseline": reused, "offset_xy_m": offset,
            "status": "PASS" if success else "FAIL" if failure else "RUNNING",
            "phases": phase, "failure": message, "sim_end_s": native.get("sim_time_s"),
            "min_lift_mm": None if not lift else 1000*lift.get("minimum_lift_m", 0.),
            "pad_normals_n": lift.get("minimum_pad_normal_force_n"),
            "upward_support_n": lift.get("mean_pad_vertical_force_n"),
            "table_support_n": lift.get("max_table_force_n"),
            "stance_adjustments": region.get("adjustments"),
            "selected_stance": next((e.get("chosen") for e in region.get("events", []) if e.get("chosen")), None),
            "torso_target_rad": json.loads((run/"task_rgbd_posture.json").read_text()).get("torso_world_yaw_target_rad") if (run/"task_rgbd_posture.json").exists() else None,
            "maximum_recorded_root_speed_m_s": max_speed, "images": images,
            "trail_xy_m": trail, "raw_result": "../"+run.name+"/result.json" if path.exists() else None,
            "source_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}


def make_video(run, output):
    import numpy as np
    import mujoco
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame

    result = json.loads((run/"result.json").read_text())
    overlay = json.loads(Path(result["claw_control_overlay"]["path"]).read_text())
    scene = overlay["generated"]["host_scene"]["path"]
    argv = json.loads((run/"invocation.json").read_text())["argv"]
    cache = argv[argv.index("--visual-stance-host-cache")+1]
    model = vision.load_cached_model(scene, cache)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135.
    camera.elevation = -25.
    writer = imageio_ffmpeg.write_frames(str(output), (640, 480), fps=10, codec="libx264", quality=7)
    writer.send(None)
    last = -math.inf
    count = 0
    try:
        for line in (run/"native_physics.jsonl").open():
            row = json.loads(line)
            if row["generation"] != 1 or row["sim_time_s"]-last < .19999:
                continue
            last = row["sim_time_s"]
            apply_recorded_native_frame(model, data, row, visual_kinematics_only=True)
            camera.lookat[:] = [*row["bodies"]["pelvis"]["position_m"][:2], .75]
            renderer.update_scene(data, camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 640, 54), fill=(8, 16, 25))
            draw.text((8, 6), "ACTUAL NATIVE STATES / RECORDED REVIEW / 2x SIM TIME", fill="white")
            status = result["gates"].get("TWO_TABLE_TRANSFER_EXECUTION", "NOT_RUN")
            draw.text((8, 25), f"sim={last:.2f}s | lift={result['gates'].get('HAL_PHYSICAL_LIFT')} | task={status}", fill=(255, 215, 130))
            writer.send(np.asarray(picture))
            count += 1
    finally:
        writer.close()
        renderer.close()
    return {"frames": count, "fps": 10, "duration_s": count/10,
            "classification": "SAVED_REAL_NATIVE_PHYSICS_STATES_NOT_NEW_PHYSICS_OR_LIVE_CAPTURE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--video-run", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not (out/".robotsim-output-reservation.json").exists() or out.parent != ROOT:
        raise ValueError("existing governed output directory required")
    records = [summarize(p.resolve()) for p in args.runs]
    if args.video_run:
        video_name = "actual_task_"+args.video_run.name[-16:]+".mp4"
        video = make_video(args.video_run, out/video_name)
        (out/(video_name+".json")).write_text(json.dumps(video, indent=2)+"\n")
    payload = {"schema": "robotsim.table-edge-task-review.v1", "cases": records,
               "mj_step_calls": 0, "scope": "D70/H220/200g; object variation along table edge only",
               "not_claimed": ["570g bottle", "hardware", "arbitrary table/bottle", "statistical repeatability"]}
    (out/"cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    videos = [p.name for p in sorted(out.glob("actual_task_*.mp4"))]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X2 · 沿桌边换瓶位</title><style>
:root{color-scheme:dark;font-family:system-ui,'Microsoft YaHei',sans-serif;background:#0c1420;color:#e6eef8}body{margin:0}main{max-width:1120px;margin:auto;padding:24px}p{line-height:1.8;color:#bed0e1}a{color:#82c7ff}section{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}.card{background:#172638;padding:18px;border-radius:12px}input,img,video,svg{width:100%}input{accent-color:#78c6e9}.status{font-size:25px;font-weight:700}td,th{padding:10px;border-bottom:1px solid #38516b;text-align:left}table{width:100%;border-collapse:collapse}.scroll{overflow:auto}button{padding:9px;background:#28445e;color:white;border:0;border-radius:6px}img{border-radius:8px}.warn{border-left:4px solid #e6ae5f;padding:12px;background:#2c2a26}@media(max-width:750px){section{grid-template-columns:1fr}main{padding:12px}}</style>
<main><h1>X2：瓶子沿桌边换位置，还能抓放吗？</h1>
<p>沿用同一完整X2、STL OmniPicker、两桌和D70/H220mm/200g圆柱。位置真值只用于初始化与评估，控制使用RGB-D。中点标明历史基线复用；没有重新运行就不计为新试验。</p>
<p class="warn">下面区分静态可达、实际抓取和完整搬运放置。失败记录保留，未执行阶段为NOT RUN。视频是已执行原生物理状态的2倍速复核渲染，不是新物理，也不以画面替代承重证据。</p>
<section><div class="card"><label>选择实际记录 <output id="caseLabel"></output><input id="case" type="range" min="0" max="__MAX__" value="0" step="1"></label><p id="status" class="status"></p><p id="metrics"></p><p id="failure"></p><a id="raw">原始结果JSON</a><svg id="plot" viewBox="0 0 480 360"></svg></div><div class="card"><img id="overview"><p id="imageNote"></p></div></section>
<div class="scroll"><table><thead><tr><th>来源</th><th>沿桌边位移</th><th>抬升/承重</th><th>退开转向</th><th>第二桌放置</th><th>完整任务</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<section><div class="card"><h2>视野问题与修正</h2><p>固定世界躯干朝向使新瓶位在弯腰后被画面边缘裁切。改为按RGB-D瓶子方位选择现有腰部目标；相机光学、识别阈值和夹爪不变。右图为离线姿态提案，实际执行结果以对应记录为准。</p><img src="left_posture_camera_clipping.png"><img src="left_posture_yaw_correction_proposal.png"></div><div class="card"><h2>实际状态视频</h2>__VIDEOS__<p><a href="cases.json">测量数据</a> · <a href="representative_preflight.json">三点静态前检</a> · <a href="carry_turn_direction_static.json">实际持瓶姿态转向对比</a></p></div></section></main>
<script>const D=__DATA__,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';function add(tag,a){let n=document.createElementNS(NS,tag);for(let k in a)n.setAttribute(k,a[k]);$('plot').append(n);return n}function show(i){let c=D.cases[i];$('caseLabel').textContent=c.run.slice(-16)+(c.reused_baseline?' · 历史基线复用':'');$('status').textContent=c.status;$('metrics').textContent='最小抬升 '+(c.min_lift_mm===null?'NOT RUN':c.min_lift_mm.toFixed(2)+'mm')+'；向上支持 '+(c.upward_support_n==null?'NOT RUN':c.upward_support_n.toFixed(4)+'N')+'；站位调整 '+(c.stance_adjustments==null?'未到该阶段':c.stance_adjustments);$('failure').textContent=c.failure||'';$('raw').href=c.raw_result||'#';$('overview').src=c.images[0]||'';$('overview').style.display=c.images.length?'block':'none';$('imageNote').textContent=c.images.length?'最后记录的实际状态后台渲染，非新物理':'尚无终帧';$('plot').replaceChildren();add('rect',{x:20,y:15,width:420,height:325,fill:'#102033',stroke:'#4e6b88'});let points=c.trail_xy_m.map(p=>[230+100*p[0],180-75*p[1]].join(',')).join(' ');add('polyline',{points,stroke:'#67d7c6','stroke-width':2,fill:'none'});for(let p of c.trail_xy_m.slice(0,1))add('circle',{cx:230+100*p[0],cy:180-75*p[1],r:5,fill:'#f0c26c'});let t=add('text',{x:30,y:35,fill:'#c7d9eb','font-size':13});t.textContent='实测骨盆XY轨迹；起点黄色；未重放物理'}$('case').oninput=e=>show(+e.target.value);show(0);</script></html>"""
    import html
    table = []
    for c in records:
        value = "待记录" if c["offset_xy_m"] is None else f'{c["offset_xy_m"][1]*100:+.1f}cm'
        phase = c["phases"]
        table.append("<tr><td>"+c["run"][-16:]+("（复用）" if c["reused_baseline"] else "")+
                     "</td><td>"+value+"</td><td>"+phase["HAL_PHYSICAL_LIFT"]+
                     "</td><td>"+phase["LEAVE_TABLE_AND_TURN"]+
                     "</td><td>"+phase["PHYSICAL_TABLE_PLACEMENT"]+
                     "</td><td>"+c["status"]+"</td></tr>")
    video_html = "".join('<video controls preload="metadata" src="'+html.escape(v)+'"></video>' for v in videos) or "<p>视频待当前有界运行结束后生成。</p>"
    page = page.replace("__TABLE__", "".join(table)).replace("__VIDEOS__", video_html)
    page = page.replace("__MAX__", str(len(records)-1)).replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (out/"index.html").write_text(page)
    print(json.dumps({"review": str(out/"index.html"), "cases": [
        {"run": c["run"], "status": c["status"], "lift_mm": c["min_lift_mm"]} for c in records]}))


if __name__ == "__main__":
    main()
