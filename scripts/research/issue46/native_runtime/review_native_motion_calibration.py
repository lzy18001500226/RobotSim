"""Measured native-motion report and trace-rendered video. No new physics."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time,html
import numpy as np,mujoco
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.research.issue46.native_runtime.native_motion_calibration import signed_delta,repeat_eligible
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
from scripts.agent import path_governance as pg

NAMES={'forward':'前进','backward':'后退','left':'左移','right':'右移','turn_left':'左转','turn_right':'右转'}


def main(out,runs,video=True):
    started=time.monotonic();rows=[];sessions=[];models=[];segments=[];evidence=[]
    for run in runs:
        result=json.loads((run/'result.json').read_text());raw=json.loads((run/'native_micro_probe_result.json').read_text());overlay=Path(result['claw_control_overlay']['path']);info=json.loads(overlay.read_text());scene=info['generated']['host_scene'];models.append(scene)
        if hashlib.sha256(Path(scene['path']).read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('source model changed')
        native=[]
        for line in (run/'native_physics.jsonl').open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r['generation']==1:native.append(r)
        model=mujoco.MjModel.from_xml_path(scene['path']);data=mujoco.MjData(model);feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
        def at(stamp):return min(native,key=lambda r:abs(r['host_monotonic_ns']-stamp))
        def foot(row):
            apply_recorded_native_frame(model,data,row);return data.xpos[feet,:2].mean(axis=0).copy()
        action_rows=[]
        for item in raw['motions']:
            begin=item['start_host_ns'];end=item.get('end_host_ns',native[-1]['host_monotonic_ns']);samples=[r for r in native if begin<=r['host_monotonic_ns']<=end]
            entry={'kind':item['kind'],'label':NAMES[item['kind']],'repeat':item['repeat'],'mode':item['mode_requested'],'status':item['status'],'source_run':str(run),'start_host_ns':begin,'end_host_ns':end,'requested':item['requested_m_or_rad'],'complete_stop_qualified':item.get('post_stop_composite',{}).get('pass',False),'metrics':item.get('metrics'),'source_and_mode_observed':item.get('source_and_mode_observed',False),'trace_local_xy_m':[],'trace_t_yaw_deg':[]}
            start=item['start_pose'];c,s=math.cos(start['yaw']),math.sin(start['yaw'])
            for r in samples[::max(1,len(samples)//150)]:
                x,y=r['bodies']['pelvis']['position_m'][:2];dx,dy=x-start['x'],y-start['y'];entry['trace_local_xy_m'].append([c*dx+s*dy,-s*dx+c*dy]);R=r['bodies']['pelvis']['rotation'];yaw=math.atan2(R[3],R[0]);entry['trace_t_yaw_deg'].append([r['sim_time_s']-samples[0]['sim_time_s'],math.degrees(math.atan2(math.sin(yaw-start['yaw']),math.cos(yaw-start['yaw'])))])
            if item.get('metrics'):
                a,b=foot(at(begin)),foot(at(end));fake_start={'x':float(a[0]),'y':float(a[1]),'yaw':start['yaw']};fake_end={'x':float(b[0]),'y':float(b[1]),'yaw':start['yaw']};entry['settled_ankle_midpoint_delta']={k:v for k,v in signed_delta(fake_start,fake_end).items() if k!='yaw_rad'};entry['coarse_repeat_eligible']=repeat_eligible(item['kind'],item['metrics'])
            else:entry['precision_assessment']='NOT_QUALIFIED_ABORTED_NOT_A_SETTLED_ENDPOINT'
            if item.get('failure'):entry['failure']=item['failure']
            rows.append(entry);action_rows.append(entry)
        clock=[e for e in raw['events'] if e.get('event')=='HOST_CLOCK_GENERATION_CHANGE'];fault=[e for e in clock if e.get('while_moving')]
        zero_after=None
        if fault:
            stamp=fault[0]['t'];zeros=[x for x in raw['bridge_commands'] if x['t']>=stamp and all(abs(x['raw'][k])<1e-12 for k in ('forward_velocity','lateral_velocity','angular_velocity'))]
            if zeros:zero_after=zeros[0]['t']-stamp
        sessions.append({'run':str(run),'model_sha256':scene['sha256'],'failure':raw.get('failure'),'abort_stop_qualified':raw.get('abort_stop_composite',{}).get('pass',False),'abort_stop_failure':raw.get('abort_stop_failure'),'clock_events':clock,'active_clock_events':fault,'first_observed_zero_after_active_fault_s':zero_after,'native_time_monotonic':all(a['host_monotonic_ns']<b['host_monotonic_ns'] and a['sim_time_s']<b['sim_time_s'] for a,b in zip(native,native[1:])),'cleanup_confirmed':result.get('cleanup',{}).get('confirmed_absent',False)})
        lo=min(x['start_host_ns'] for x in action_rows)-1000000000;hi=max(x['end_host_ns'] for x in action_rows)+1000000000;selected=[r for r in native if lo<=r['host_monotonic_ns']<=hi];segments.append((run,selected,action_rows,model))
        for f in ['native_physics.jsonl','native_micro_probe_result.json','result.json','invocation.json']:
            path=run/f;evidence.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    attempted={r['kind'] for r in rows}
    for kind in NAMES:
        if kind not in attempted:rows.append({'kind':kind,'label':NAMES[kind],'status':'NOT_RUN','requested':math.radians(15) if kind.startswith('turn_') else .05,'metrics':None})
    payload={'schema':'robotsim.motion-capability-review.v1','classification':'PARTIAL_MEASUREMENTS_NO_MINIMUM_PRECISION_QUALIFICATION','motions':rows,'sessions':sessions,'all_physics_from_recorded_native_runs':True,'new_physics_calls':0,'repetitions':'NOT_RUN: no completed initial response met this batch coarse repeat screen','coarse_repeat_screen':{'linear_signed_progress_m':[.025,.10],'linear_cross_drift_max_m':.05,'turn_signed_progress_rad':[math.radians(7.5),math.radians(30)],'turn_translation_max_m':.10,'purpose':'repeat selection only, not physical safety or universal task tolerance'},'source_models':models,'inputs':evidence,'limitations':['Only this simulation/model/mode/command-and-stop procedure was tested','No physical hardware precision claim','No statistically qualified repeatability','Aborted lateral motion has no qualified settled endpoint','Angle and position filter/stop latency affects results; not a hardware resolution limit']}
    (out/'measurements.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    if video:
        import imageio_ffmpeg
        stream=imageio_ffmpeg.write_frames(str(out/'actual_motion_trace_review.mp4'),(640,480),fps=5,codec='libx264',quality=7,pix_fmt_in='rgb24');stream.send(None)
        for run,frames,actions,model in segments:
            renderer=mujoco.Renderer(model,height=480,width=640);data=mujoco.MjData(model);cam=mujoco.MjvCamera();cam.lookat[:]=[.1,-.05,.7];cam.distance=2.8;cam.azimuth=135.;cam.elevation=-20.;last=-math.inf
            for row in frames:
                if row['sim_time_s']-last<.199999:continue
                last=row['sim_time_s'];apply_recorded_native_frame(model,data,row);renderer.update_scene(data,camera=cam);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0;renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0;im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,65),fill=(4,14,24));action=next((a for a in actions if a['start_host_ns']<=row['host_monotonic_ns']<=a['end_host_ns']),None);text=(action['kind']+' | '+action['status']) if action else 'independent episode / initialization or settling'
                draw.text((9,6),'RECORDED NATIVE MuJoCo/MC STATES | NO NEW PHYSICS',fill='white');draw.text((9,25),run.name[-16:]+' | sim '+f"{row['sim_time_s']:.3f}s",fill=(170,210,235));draw.text((9,44),text,fill=(255,210,135));stream.send(np.asarray(im))
            renderer.close()
        stream.close()
    table=[]
    for i,r in enumerate(rows):
        m=r.get('metrics');turn=r['kind'].startswith('turn_');target='15°' if turn else '5 cm'
        if m:
            a=m['actual'];actual=f"{a['yaw_rad']*180/math.pi:+.2f}°" if turn else f"{m['signed_goal_progress_m_or_rad']*100:.2f} cm"
            cross=f"{math.hypot(a['world_dx_m'],a['world_dy_m'])*100:.2f} cm 漂移" if turn else f"{a['left_m' if r['kind'] in ('forward','backward') else 'forward_m']*100:+.2f} cm 串扰"
        else:actual='未执行' if r['status']=='NOT_RUN' else '中断，不作终值精度判断';cross='—'
        table.append(f'<tr data-i="{i}"><td>{r["label"]}</td><td>{html.escape(r.get("mode","—"))}</td><td>{target}</td><td>{actual}</td><td>{cross}</td><td>{"通过" if r.get("complete_stop_qualified") else "未取得资格" if r["status"]!="NOT_RUN" else "NOT RUN"}</td></tr>')
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X2运动能力 · 实际测量</title><style>:root{color-scheme:dark;font-family:system-ui,'Microsoft YaHei',sans-serif;background:#0a1220;color:#e6edf5}body{margin:0}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px}p{color:#b5c8dc;line-height:1.8}.note{border-left:3px solid #e9b15d;padding:12px;background:#1a2635}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #33475d;text-align:left}tr[data-i]{cursor:pointer}tr.selected{background:#1d4564}section{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}svg,video{width:100%;background:#102030;border-radius:8px}a{color:#8ac8ff}input{width:100%;accent-color:#77bfff}code{overflow-wrap:anywhere}.table{overflow:auto}@media(max-width:800px){section{grid-template-columns:1fr}main{padding:14px}}</style><main><h1>5cm / 15° 请求，机器人实际移动了多少？</h1><p class="note">本轮标定未完成：已取得四项停稳响应；左移在运动中发生时钟跳变而中断，右移未执行。不再启动后续运动，不放宽检查。这里的结果属于当前模式、命令和停止流程，不能称为硬件最小精度。</p><p class="note">方法限制：这是本轮新标定器的首次结果，不是现成SDK的最小分辨率。进度触发使用0.6秒窗口、至少1秒运动及双侧支撑条件，发零之后还要恢复站立。前进发零时已约10.7cm，随后又增加约18.8cm；转向判定也有明显滞后。不能把这些结果当成“机器人只能这么粗”，更不能据此盲调增益。</p><div class="table"><table><thead><tr><th>动作</th><th>模式</th><th>目标</th><th>实际主方向（骨盆）</th><th>串扰 / 漂移</th><th>复合停稳</th></tr></thead><tbody>__ROWS__</tbody></table></div><section><div><label>查看动作 <output id="label"></output><input id="trial" type="range" min="0" max="__MAX__" step="1" value="0"></label><svg id="plot" viewBox="0 0 520 420" role="img" aria-label="实际身体轨迹，单位米"></svg><p id="detail"></p></div><div><video controls preload="metadata" src="actual_motion_trace_review.mp4"></video><p>实际原生物理状态的复核渲染，非新物理或现场屏幕录制。不同会话之间是正常重新初始化，不声称连续完成同一任务。</p></div></section><details><summary>时钟故障与证据边界</summary><p>左移已发出非零命令。运行记录 CLOCK_STEP_DURING_MOTION，随后发零并请求站立；原复合停止资格未取得。原生物理时间保持单调，不将此误诊为物理时间冻结或摩擦问题。</p><p>前四项也只是单次响应，没有达到本轮粗筛重复条件，因此没有反复调参或制造“三次通过”。</p></details><p><a href="measurements.json">完整数据与原始文件哈希</a></p><script>const D=__DATA__,S='http://www.w3.org/2000/svg',$=id=>document.getElementById(id);function el(t,a,txt){const n=document.createElementNS(S,t);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));if(txt)n.textContent=txt;return n}function show(i){$('trial').value=i;const m=D.motions[i];$('label').textContent=m.label+' · '+m.status;document.querySelectorAll('tr[data-i]').forEach((r,k)=>r.classList.toggle('selected',k===i));const p=$('plot');p.replaceChildren();if(m.kind.startsWith('turn_')){const a=m.trace_t_yaw_deg||[],tmax=Math.max(1,...a.map(v=>v[0])),vmax=Math.max(30,...a.map(v=>Math.abs(v[1]))),X=t=>55+t/tmax*420,Y=v=>210-v/vmax*160;const target=m.kind==='turn_left'?15:-15;for(const v of [-vmax,0,vmax]){p.append(el('line',{x1:55,y1:Y(v),x2:475,y2:Y(v),stroke:'#2b425b'}));p.append(el('text',{x:10,y:Y(v)+4,fill:'#a7c5de','font-size':12},v.toFixed(0)+'°'));}p.append(el('line',{x1:55,y1:Y(target),x2:475,y2:Y(target),stroke:'#76aaff','stroke-dasharray':'6 4'}));p.append(el('polyline',{points:a.map(v=>X(v[0])+','+Y(v[1])).join(' '),fill:'none',stroke:'#66d9a5','stroke-width':3}));p.append(el('text',{x:15,y:25,fill:'#b9d2ec','font-size':14},'实际转角 / 仿真时间；蓝虚线为15°目标'));}else{const X=x=>260+x*500,Y=y=>210-y*500;for(let v=-.4;v<=.401;v+=.1){p.append(el('line',{x1:X(v),y1:10,x2:X(v),y2:410,stroke:'#2b425b'}));p.append(el('line',{x1:10,y1:Y(v),x2:510,y2:Y(v),stroke:'#2b425b'}));}const a=m.trace_local_xy_m||[];p.append(el('polyline',{points:a.map(v=>X(v[0])+','+Y(v[1])).join(' '),fill:'none',stroke:'#66d9a5','stroke-width':3}));p.append(el('circle',{cx:X(0),cy:Y(0),r:5,fill:'#fff'}));if(a.length)p.append(el('circle',{cx:X(a.at(-1)[0]),cy:Y(a.at(-1)[1]),r:6,fill:'#ffcc7a'}));let tx=m.kind==='forward'?.05:m.kind==='backward'?-.05:0,ty=m.kind==='left'?.05:m.kind==='right'?-.05:0;p.append(el('circle',{cx:X(tx),cy:Y(ty),r:7,fill:'none',stroke:'#76aaff','stroke-width':2}));p.append(el('text',{x:15,y:25,fill:'#b9d2ec','font-size':14},'X前 / Y左；网格0.1m；蓝圈目标，绿线实际'));}$('detail').textContent=m.metrics?'实际数值来自停稳后骨盆姿态；完整JSON另含踝中点平移，与站位研究坐标相对应。方向有响应不代表达到目标精度。':m.status==='NOT_RUN'?'该项未执行，不能给出精度。':'中断轨迹不作为合格停稳终值。';}document.querySelectorAll('tr[data-i]').forEach(r=>r.onclick=()=>show(+r.dataset.i));$('trial').oninput=e=>show(+e.target.value);show(0);window.MOTION_REVIEW={count:D.motions.length};</script></main></html>"""
    page=page.replace('__ROWS__',''.join(table)).replace('__MAX__',str(len(rows)-1)).replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'));(out/'index.html').write_text(page)
    (out/'input_hashes.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'output':str(out),'motions':len(rows),'video':video,'wall_s':time.monotonic()-started}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--no-video',action='store_true');args=p.parse_args();main(args.output_dir,args.runs,not args.no_video)
