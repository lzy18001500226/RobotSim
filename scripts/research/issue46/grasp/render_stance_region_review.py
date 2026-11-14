"""Static-study web review: exact sampled poses, no physics or robot commands."""
from pathlib import Path
import argparse,copy,json,math,sys,time,hashlib
import numpy as np
import mujoco
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.research.issue46.grasp.stance_region_static_review import Study,sha
from scripts.agent import path_governance as pg

HTML=r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X2 · 22cm 瓶子站位范围审核</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#e6edf5;background:#0a1220;color-scheme:dark}*{box-sizing:border-box}body{margin:0}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:26px;margin:6px 0}h2{font-size:17px;margin:0 0 10px}.sub{color:#a8bbd0;line-height:1.7}header{margin-bottom:22px}.tag{display:inline-block;color:#ffd595;border:1px solid #6b5735;padding:5px 9px;border-radius:6px;font-size:12px}.controls,.legend{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}button,select,a.link{font:inherit;border:1px solid #42546d;border-radius:7px;padding:8px 12px;background:#152339;color:#e6edf5;cursor:pointer}button[aria-pressed=true]{background:#235e8f;border-color:#70bdff}.layout{display:grid;grid-template-columns:1fr 1.1fr;gap:22px}.panel{min-width:0}.stats{font-size:14px;color:#b9cce1}svg{display:block;width:100%;height:auto;background:#0f1b2b;border:1px solid #273c53;border-radius:10px}.dot{cursor:pointer;stroke:#101f30;stroke-width:1.5}.dot.selected{stroke:#fff;stroke-width:3}.legend span{font-size:13px}.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}#pose{display:block;width:100%;border-radius:10px;background:#15212c}#detail{line-height:1.8;margin-top:12px;font-size:14px;min-height:120px}.green{color:#70e2ad}.yellow{color:#ffd081}.muted{color:#a8bbd0}.stage{margin:10px 0}.sliders{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:18px 0}.sliders label,.stage label{display:grid;gap:8px;font-size:14px}.sliders output,.stage output{font-variant-numeric:tabular-nums;color:#8dccff}input[type=range]{width:100%;min-width:0;accent-color:#6bc0ff;cursor:pointer}.stage input[type=range]{max-width:100%}a{color:#8dccff}details{margin:20px 0;border-top:1px solid #304259;padding-top:12px}summary{cursor:pointer}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #2c3d52;text-align:left}footer{margin-top:24px;font-size:12px;color:#94a9bf;overflow-wrap:anywhere}.note{padding:10px 12px;border-left:3px solid #e7b965;background:#1d2430;font-size:13px;line-height:1.7}#phaseNote{font-size:13px;color:#a8bbd0;min-height:36px}.coords{font-variant-numeric:tabular-nums}.controls label{max-width:100%;display:flex;flex-wrap:wrap;gap:6px}select{max-width:100%;min-width:0}button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #f8d77e;outline-offset:2px}@media(max-width:850px){.layout,.sliders{grid-template-columns:1fr}main{padding:14px}h1{font-size:22px}}
</style><main><header><span class="tag">静态 FK / IK / 碰撞筛选 · 未运行动力学</span><h1>同一只瓶子，机器人可以站在哪里？</h1><div class="sub">完整 X2 · 冻结 OmniPicker · D70 / H220 mm / 200 g。桌子、瓶位、模型和物理参数不变。每个点是一组足底参考站位与本体朝向。</div></header>
<div class="note">绿色表示已保存指定路径族的静态见证，不能据此认定能够走到该点或实际抓稳。机械连杆与瓶身的固有小间隙不等于要求底座达到同样的定位精度。网格间未测试，不把色点连成“保证可用”的连续区域。灰色表示求解未决，不能据此断言不可达。</div>
<section class="sliders" aria-label="已计算站位滑块">
<label>本体朝向偏移 <output id="yawValue"></output><input id="yawSlider" type="range" min="-30" max="30" step="15" value="0"></label>
<label>站位 X <output id="xValue"></output><input id="xSlider" type="range" min="0" max="4" step="1" value="2"></label>
<label>站位 Y <output id="yValue"></output><input id="ySlider" type="range" min="0" max="6" step="1" value="2"></label>
</section><div class="sub" style="font-size:12px">滑块连续拖动，结果吸附到已计算采样：位置间隔8cm，朝向间隔15°。不插值制造未验证姿态。</div>
<div class="controls"><button id="full" aria-pressed="true">完整路径结果</button><button id="ik" aria-pressed="false">只看终点 IK</button></div>
<div class="layout"><section class="panel"><h2>桌前局部站位图</h2><div id="stats" class="stats" aria-live="polite"></div><svg id="map" viewBox="0 0 650 545" role="img" aria-label="站位采样、桌子和固定瓶位"></svg><div class="legend"><span><i class="sw" style="background:#49c98e"></i>静态路径候选</span><span><i class="sw" style="background:#efa85e"></i>有终点解，路径/接触受阻</span><span><i class="sw" style="background:#718397"></i>求解未决</span><span>★ 原成功站位参考</span></div><div class="sub" style="font-size:12px">坐标为世界系。站位点使用左右 ankle_roll 参考点的 XY 中点，非不同蹲姿下会变化的骨盆位置。箭头显示本体前方。</div></section>
<section class="panel"><h2 id="selectedTitle">选中候选</h2><img id="pose" alt="完整机器人与桌子的选中静态姿态"><div class="stage"><label>动作阶段 <output id="phaseValue"></output><input id="phaseSlider" type="range" min="0" max="4" step="1" value="3"></label></div><div id="phaseNote"></div><div id="detail" aria-live="polite"></div><div class="controls"><a class="link" id="witness">查看该点完整见证 JSON</a><a class="link" id="allResults" href="result.json">全部结果</a></div></section></div>
<details><summary>如何判断“容错范围”，以及本轮没有证明什么</summary><p class="sub">优先看绿色点如何随朝向变化，而不是追逐一个精确站位。选中点的“邻点通过数”只反映已计算的邻居，不是误差球认证。最小外部间隙不含脚底支持接触和闭合/抬升时的有意夹持接触；大于50mm的距离截断显示。自碰撞通过启用的完整模型接触逐帧检查，未额外关闭任何碰撞。</p><table><tr><th>已做</th><th>尚未做</th></tr><tr><td>两种实际源姿态、有限多初值 IK；准备动作重放几何、伸手、接近、闭合、假设60mm抬升的离散检查</td><td>行走到站位、动态平衡、实际抓力/滑移/离桌；连续扫掠认证；所有可能姿态与求解初值的穷尽证明</td></tr><tr><td>模型数组与源 XML 哈希不变；原有碰撞 pair / exclude 保留</td><td>厂商等效、真实光学噪声容错、换桌子后无需重算的通用区域</td></tr></table><p class="sub">IK 使用原有源范围和有限运动信赖域。受阻点只否定本次检查的姿态/路径族，可能存在其他尚未搜索的路径。</p></details>
<footer id="provenance"></footer></main><script>
const DATA=__DATA__;const SVG='http://www.w3.org/2000/svg';const $=id=>document.getElementById(id);let angle=0,mode='full',selected=null;
const labels={STATIC_WITNESS:'静态路径候选',PATH_BLOCKED:'有终点解，路径或接触受阻',UNRESOLVED:'求解未决'};const stageNames={PREPARATION:'已有准备动作',REACH:'到预抓取位',APPROACH:'开放夹爪接近',CLOSURE:'闭合几何',LIFT:'假设抬升60mm',INITIAL:'准备姿态（未取得合格路径）'};
function node(tag,attrs,text){const el=document.createElementNS(SVG,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el}
function visible(){return DATA.points.filter(p=>p.yaw_offset_deg===angle)}
function color(p){return mode==='ik'?(p.endpoint_ik_found?'#49c98e':'#718397'):p.status==='STATIC_WITNESS'?'#49c98e':p.status==='PATH_BLOCKED'?'#efa85e':'#718397'}
function draw(){const svg=$('map');svg.replaceChildren();const [xmin,xmax,ymin,ymax]=DATA.view;const X=x=>55+(x-xmin)/(xmax-xmin)*560,Y=y=>485-(y-ymin)/(ymax-ymin)*420;
 for(let x=Math.ceil(xmin*10)/10;x<=xmax;x+=.1){svg.append(node('line',{x1:X(x),y1:55,x2:X(x),y2:485,stroke:'#253a50'}));svg.append(node('text',{x:X(x),y:509,fill:'#a8bbd0','font-size':12,'text-anchor':'middle'},x.toFixed(1)))}
 for(let y=Math.ceil(ymin*10)/10;y<=ymax;y+=.1){svg.append(node('line',{x1:55,y1:Y(y),x2:615,y2:Y(y),stroke:'#253a50'}));svg.append(node('text',{x:48,y:Y(y)+4,fill:'#a8bbd0','font-size':12,'text-anchor':'end'},y.toFixed(1)))}
 Object.entries(DATA.inputs.tables).forEach(([name,b])=>{if(!b)return;const [a,z]=[b.min,b.max];if(z[0]<xmin||a[0]>xmax||z[1]<ymin||a[1]>ymax)return;svg.append(node('rect',{x:X(a[0]),y:Y(z[1]),width:X(z[0])-X(a[0]),height:Y(a[1])-Y(z[1]),fill:'#51667b','fill-opacity':.32,stroke:'#a9bdcf','stroke-width':2}));svg.append(node('text',{x:Math.min(560,X((a[0]+z[0])/2)),y:70,fill:'#e6edf5','font-size':14,'text-anchor':'middle'},'原取瓶桌轮廓'))});
 const bottle=DATA.inputs.fixed_object_center_m;svg.append(node('circle',{cx:X(bottle[0]),cy:Y(bottle[1]),r:Math.abs(X(bottle[0]+.035)-X(bottle[0])),fill:'#3288ef',stroke:'#b5daff','stroke-width':2}));svg.append(node('text',{x:X(bottle[0]),y:Y(bottle[1])-28,fill:'#aad2ff','font-size':14,'text-anchor':'middle'},'固定瓶位'));
 visible().forEach(p=>{const el=node('circle',{cx:X(p.anchor_xy_m[0]),cy:Y(p.anchor_xy_m[1]),r:p.id===selected?.id?10:8,fill:color(p),class:'dot'+(p.id===selected?.id?' selected':'')});el.append(node('title',{},`${p.id} ${labels[p.status]}`));el.addEventListener('click',()=>selectPoint(p.id));svg.append(el)});
 const ref=DATA.inputs.reference_anchor_world_xy_m;svg.append(node('text',{x:X(ref[0])+12,y:Y(ref[1])-14,fill:'#f9dc7a','font-size':21},'★'));
 if(selected){const [x,y]=selected.anchor_xy_m,a=selected.yaw_world_rad,x1=X(x),y1=Y(y),x2=X(x+.10*Math.cos(a)),y2=Y(y+.10*Math.sin(a));svg.append(node('line',{x1,y1,x2,y2,stroke:'#fff','stroke-width':3}));svg.append(node('circle',{cx:x2,cy:y2,r:3,fill:'#fff'}))}
 svg.append(node('text',{x:335,y:537,fill:'#b4c9dd','font-size':13,'text-anchor':'middle'},'世界 X / m →'));svg.append(node('text',{x:20,y:30,fill:'#b4c9dd','font-size':13},'Y / m ↑'));
 const v=visible(),good=v.filter(p=>p.status==='STATIC_WITNESS').length,ik=v.filter(p=>p.endpoint_ik_found).length;$('stats').textContent=`本朝向 ${v.length} 点 · 静态路径候选 ${good} · 终点 IK 有解 ${ik} · 参考朝向偏移 ${angle}°`;
}
function selectPoint(id){selected=DATA.points.find(p=>p.id===id);if(!selected)return;angle=selected.yaw_offset_deg;
 $('yawSlider').value=angle;$('xSlider').value=selected.ix;$('ySlider').value=selected.iy;
 const ref=DATA.inputs.reference_anchor_world_xy_m,fmt=(v,k)=>`${v.toFixed(3)} m（参考${v-ref[k]>=0?'+':''}${Math.round((v-ref[k])*100)} cm）`;
 $('yawValue').textContent=`${angle>0?'+':''}${angle}°（世界 ${(selected.yaw_world_rad*180/Math.PI).toFixed(1)}°）`;$('xValue').textContent=fmt(selected.anchor_xy_m[0],0);$('yValue').textContent=fmt(selected.anchor_xy_m[1],1);
 $('selectedTitle').textContent=`${id} · ${labels[selected.status]}`;
 const phase=$('phaseSlider');phase.max=Math.max(0,selected.images.length-1);phase.disabled=selected.images.length===1;const prefer=selected.images.findIndex(x=>x.phase==='CLOSURE');phase.value=prefer>=0?prefer:selected.images.length-1;showImage(+phase.value);
 const n=DATA.points.filter(p=>p.yaw_offset_deg===angle&&p.status==='STATIC_WITNESS'&&Math.abs(p.ix-selected.ix)<=1&&Math.abs(p.iy-selected.iy)<=1&&p.id!==id).length;
 const gap=selected.minimum_external_clearance_m_capped_50mm,block=selected.first_blocker,attempts=selected.attempts.map(a=>`${a.posture==='prepared'?'预抬肘＋下蹲':'站立'}：${a.stage}`).join('；');
 $('detail').replaceChildren();[['站位 / 朝向',`${selected.anchor_xy_m.map(x=>x.toFixed(3)).join(', ')} m；${(selected.yaw_world_rad*180/Math.PI).toFixed(1)}°`],['检查结果',labels[selected.status]],['限制阶段',selected.stage],['最小非接触外部间隙',`${gap>=.049999?'≥':''}${(gap*1000).toFixed(2)} mm（自碰撞另查）`],['最近外部几何对',`${(selected.limiting_external_pair||[]).join(' ↔ ')} / ${selected.limiting_external_phase||'未检查'}`],['附近已通过采样点',`${n} 个；不代表连续区域保证`],['尝试姿态',attempts],...(block?[['首个阻碍',`${block.bodies.join(' ↔ ')}；有符号距离 ${(block.distance_m*1000).toFixed(3)} mm`]]:[])].forEach(([a,b])=>{const d=document.createElement('div'),strong=document.createElement('strong');strong.textContent=a+'：';d.append(strong,document.createTextNode(b));$('detail').append(d)});
 $('witness').href=(DATA.witness_base||'')+encodeURIComponent(selected.id)+'.json';draw();}
function showImage(i){const item=selected.images[i];if(!item)return;$('pose').src=item.file;$('phaseSlider').value=i;$('phaseValue').textContent=`${i+1}/${selected.images.length} · ${stageNames[item.phase]||item.phase}`;$('phaseNote').textContent=item.phase==='LIFT'?'假设抬升的静态见证：对象位姿仅用于离线几何检查，没有物理抬升。':item.phase==='INITIAL'?'该点未取得合格路径；图中显示检查的准备姿态。':'选中离散路径姿态：仅 mj_forward，无动力学、无控制命令。'}
function fromSliders(){const a=+$('yawSlider').value,x=+$('xSlider').value,y=+$('ySlider').value;const p=DATA.points.find(p=>p.yaw_offset_deg===a&&p.ix===x&&p.iy===y);if(p)selectPoint(p.id)}
['yawSlider','xSlider','ySlider'].forEach(id=>$(id).addEventListener('input',fromSliders));$('phaseSlider').addEventListener('input',()=>showImage(+$('phaseSlider').value));
$('full').onclick=()=>{mode='full';$('full').setAttribute('aria-pressed','true');$('ik').setAttribute('aria-pressed','false');draw()};$('ik').onclick=()=>{mode='ik';$('ik').setAttribute('aria-pressed','true');$('full').setAttribute('aria-pressed','false');draw()};
$('allResults').href=(DATA.witness_base||'')+'result.json';$('provenance').textContent=`模型 SHA-256 ${DATA.inputs.scene_sha256} · ${DATA.points.length} 个离散组合 · ${DATA.render.engine} 渲染 · mj_step = 0 · 物理抓取/行走 NOT RUN。`;
window.STANCE_REVIEW={pointCount:DATA.points.length,getSelection:()=>({angle,id:selected?.id,mode,ix:selected?.ix,iy:selected?.iy,phase:+$('phaseSlider').value}),setAngle:a=>{const p=DATA.points.find(p=>p.yaw_offset_deg===a&&p.ix===selected.ix&&p.iy===selected.iy);if(p)selectPoint(p.id)}};selectPoint('y+0_r2_c2');
</script></html>"""


def main(source,page_only=False,output_dir=None,html_name="index.html"):
    import os
    out=output_dir or source
    if output_dir is not None and not page_only:raise ValueError("separate presentation reuses verified images only")
    study=Study();inputs=json.loads((source/'inputs.json').read_text());result=json.loads((source/'result.json').read_text())
    if study.model_identity!=inputs['scene_sha256'] or not result['frozen_model_unchanged']:raise ValueError('review model mismatch')
    images=(source if page_only else out)/'images';images.mkdir(exist_ok=True);renderer=mujoco.Renderer(study.model,height=480,width=640);began=time.monotonic();count=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[1.05,-1.5,.65];camera.distance=2.1;camera.azimuth=135.;camera.elevation=-23.
    for point in result['points']:
        point_source=json.loads((source/(point['id']+'.json')).read_text());path=point_source['witness']['path'];frames=[]
        if path:
            minimum=min(path,key=lambda x:x['screen']['minimum_external_clearance_m_capped_50mm']);point['limiting_external_pair']=minimum['screen']['limiting_pair'];point['limiting_external_phase']=minimum['phase']
        if point['status']=='STATIC_WITNESS':
            for phase in ('PREPARATION','REACH','APPROACH','CLOSURE','LIFT'):
                found=[p for p in path if p['phase']==phase]
                if found:frames.append((phase,np.array(found[-1]['qpos'])))
        elif path:frames=[(path[-1]['phase'],np.array(path[-1]['qpos']))]
        else:frames=[('INITIAL',study.relocate(study.templates[point['posture']],point['anchor_xy_m'],point['yaw_world_rad']))]
        point['images']=[]
        for phase,q in frames:
            if not page_only:
                study.apply(q);renderer.update_scene(study.data,camera=camera);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0;renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                block=set((point.get('first_blocker') or {}).get('bodies',[]))
                for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
                        body=study.names[int(study.model.geom_bodyid[g.objid])]
                        if body in block:g.rgba[:]=[1.,.22,.12,.85]
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,48),fill=(5,15,25));draw.text((12,8),'STATIC FK ONLY - NO PHYSICS | '+point['id']+' | '+phase,fill='white');draw.text((12,27),'Full X2 / original table / fixed bottle / sampled path geometry',fill=(170,200,220))
            f=images/(point['id']+'_'+phase+'.png');
            if not page_only:image.save(f)
            elif not f.is_file():raise FileNotFoundError(f)
            point['images'].append({'phase':phase,'file':Path(os.path.relpath(f,out)).as_posix()});count+=1
        if count%35<5:print(json.dumps({'images':count,'point':point['id']}),flush=True)
    renderer.close()
    for p in result['points']:
        neighbors=[q for q in result['points'] if q['yaw_offset_deg']==p['yaw_offset_deg'] and abs(q['ix']-p['ix'])<=1 and abs(q['iy']-p['iy'])<=1]
        p['tested_neighbor_margin_label']=len(neighbors)==9 and all(q['status']=='STATIC_WITNESS' and q['minimum_external_clearance_m_capped_50mm']>=.002 for q in neighbors)
    ref=inputs['reference_anchor_world_xy_m'];b=inputs['fixed_object_center_m'];v=[ref[0]-.26,max(ref[0]+.28,b[0]+.14),min(ref[1]-.25,b[1]-.20),max(ref[1]+.39,b[1]+.25)]
    if any(hashlib.sha256(getattr(study.model,k).tobytes()).hexdigest()!=h for k,h in study.model_arrays.items()):raise RuntimeError('review renderer mutated frozen model')
    payload={'witness_base':Path(os.path.relpath(source,out)).as_posix().rstrip('/')+'/' if source!=out else '', 'inputs':inputs,'points':result['points'],'view':v,'render':{'engine':mujoco.__version__,'images':count,'wall_s':time.monotonic()-began,'new_physics_calls':0}}
    (out/'review_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    (out/html_name).write_text(HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
    # Numerical verifier writes the final no-overwrite manifest on a fresh run.
    print(json.dumps({'html':str(out/html_name),'images':count,'wall_s':time.monotonic()-began}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--page-only',action='store_true');parser.add_argument('--output-dir',type=Path);parser.add_argument('--html-name',default='index.html');args=parser.parse_args();main(args.input_dir,args.page_only,args.output_dir,args.html_name)
