"""Process presentation links immutable historical pages; no copied meshes."""
from pathlib import Path
import argparse,hashlib,html,json,os,sys
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.grasp.render_stance_region_review import main as render
ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
STUDY=ROOT/'h220-stance-region-static-exact-coupling-20261111T084002Z'
ENTRIES=[
 ('机构对照','原始、CAD与历史拟合模型','omnipicker-legacy-cad-slider-20261026T074114Z/viewer.html','历史静态比较：原始fixed、G2失败转换、CAD分支与当时的模拟候选。保留失败，不把滑块动画当作机构物理通过。'),
 ('STL接触面','STL夹爪接触区域与开闭','stl-contact-definition-review-20261105T034620Z/index.html','查看源STL、候选接触面和0–100%开闭下的圆拟合；这里展示的是几何审核。'),
 ('早期拟合','全包胶弧面圆拟合','stl-full-pad-circle-fit-20261104T160730Z/index.html','保留早期整个包胶面的拟合方法和局限，展示接触几何研究过程。'),
 ('接触区域细化','下方凹弧容纳圆','stl-lower-arc-circle-review-20261105T074043Z/index.html','针对下方凹弧区域的静态容纳圆审阅；几何容纳不等于实际承载。'),
 ('多尺寸研究','多尺寸包持圆','stl-variable-support-circle-review-20261105T080615Z/index.html','通过开闭滑块观察接触带和不同尺寸的包持关系；不宣称穷尽或通用抓取通过。'),
 ('整机集成','STL完整机器人姿态','stl-posture-human-review-20261101T031629Z/index.html','历史完整X2静态姿态审核，保留手臂、躯干与场景关系。'),
 ('物理结果','22cm完整搬运视频','native-transfer-recorded-review-20261111T005652Z/full_x2_actual_transfer_review.mp4','D70/H220mm/200g圆柱：已执行原生物理轨迹的复核渲染。与静态网页、570g瓶验收及真机结果分开。')]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(output=None):
    out=output or pg.create_unique_output_directory(ROOT,prefix='h220-stance-slider-process-review').native_path
    if not out.is_dir():raise ValueError('use an existing governed reserved output directory')
    records=[];cards=[]
    for tag,title,relative,description in ENTRIES:
        p=ROOT/relative
        if not p.is_file():raise FileNotFoundError(p)
        href=Path(os.path.relpath(p,out)).as_posix()
        records.append({'title':title,'tag':tag,'source_path':str(p),'windows_unc':'\\\\wsl.localhost\\Ubuntu-22.04'+str(p).replace('/','\\'),'sha256':digest(p),'description':description,'relative_href':href})
        cards.append(f'<article><div class="tag">{html.escape(tag)}</div><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p><a href="{html.escape(href)}">打开原始页面 ↗</a><details><summary>原文件位置</summary><code>{html.escape(str(p))}</code></details></article>')
    render(STUDY,page_only=True,output_dir=out,html_name='stance.html')
    records.insert(0,{'title':'固定瓶位站位研究 · 滑块版','source_path':str(out/'stance.html'),'dataset_path':str(STUDY),'dataset_result_sha256':digest(STUDY/'result.json'),'status':'STATIC_ONLY','reuses_images_without_copy':True})
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X2抓取研究 · 过程展示入口</title><style>
:root{color-scheme:dark;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#0a1220;color:#e6edf5}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:30px 24px}h1{font-size:29px;margin:8px 0 12px}p{color:#b6c8dc;line-height:1.75}a{color:#8dccff}header{margin-bottom:26px}.hero{padding:22px;border:1px solid #397eab;border-radius:12px;background:#12253a;display:grid;grid-template-columns:1fr 360px;gap:22px}.hero img{width:100%;border-radius:8px}.hero a{display:inline-block;background:#245c8a;color:#fff;padding:12px 18px;border-radius:8px;text-decoration:none}.tag{font-size:12px;color:#f6cd8c}h2{font-size:19px;margin:9px 0}section{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:22px}article{padding:19px;border:1px solid #30445e;border-radius:10px;background:#101d2e}article p{font-size:14px;min-height:100px}details{margin-top:16px;font-size:12px;color:#9cb1ca}summary{cursor:pointer}code{display:block;overflow-wrap:anywhere;line-height:1.6;padding-top:8px}footer{margin-top:24px;font-size:13px;color:#9cb1ca}a:focus-visible,summary:focus-visible{outline:3px solid #f5d784;outline-offset:4px}@media(max-width:900px){.hero{grid-template-columns:1fr 260px}section{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){main{padding:18px}.hero,section{grid-template-columns:1fr}h1{font-size:24px}article p{min-height:0}}
</style><main><header><div class="tag">RobotSim · 本地研究过程展示</div><h1>从夹爪几何，到完整X2抓放</h1><p>按研究过程展示源模型、拟合方法、接触几何、整机集成与物理结果。每一页保留自己的日期和验证范围；历史失败与静态候选不升级为物理成功。</p></header><div class="hero"><div><div class="tag">当前审核 · 固定22cm瓶位</div><h2>站位范围：改用滑块直接探索</h2><p>拖动站位X、站位Y、本体朝向与动作阶段。175个组合中53个有完整静态路径见证；只切换已计算采样，不插值伪造新结果。</p><a href="stance.html">打开滑块站位页 →</a><details><summary>当前页面保存位置</summary><code>__PATH__</code></details></div><img src="__IMAGE__" alt="当前完整X2、桌子和固定瓶位的静态几何"></div><section>__CARDS__</section><footer>所有旧文件保持原位置；总入口只链接已有页面与图像，没有迁移或复制厂商网格。部分历史3D网页依赖Three.js CDN，建议展示前检查网络。静态几何 ≠ 动态承载 ≠ 真机通过。</footer></main></html>"""
    page=page.replace('__PATH__',html.escape(str(out))).replace('__CARDS__',''.join(cards)).replace('__IMAGE__',Path(os.path.relpath(STUDY/'images/y+0_r2_c2_CLOSURE.png',out)).as_posix())
    pg.write_new_text(out/'index.html',page);pg.write_new_text(out/'presentation_index.json',json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    pg.write_new_text(out/'README.md',f"""# X2 process presentation

Review hub: {out}/index.html
Slider view: {out}/stance.html

Serve the existing issue-46 evidence root read-only so relative historical links work:

```bash
python3 -m http.server 8792 --bind 127.0.0.1 --directory {ROOT}
```

Open http://127.0.0.1:8792/{out.name}/index.html .

Original numerical study and historical pages are unchanged. Position sliders select8cm grid samples; yaw selects15degree samples; stage selects saved static images. No new physics or geometry calculation is represented.
""")
    print(json.dumps({'output':str(out),'hub':f'http://127.0.0.1:8792/{out.name}/index.html','slider':f'http://127.0.0.1:8792/{out.name}/stance.html','original_pages':len(ENTRIES)},ensure_ascii=False),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);main(p.parse_args().output_dir)
