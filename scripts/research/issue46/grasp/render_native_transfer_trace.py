"""Render measured native MC/MuJoCo states; never a new physics rollout."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import mujoco
import imageio_ffmpeg
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True);p.add_argument('--fps',type=int,default=10);p.add_argument('--start-sim-time',type=float);p.add_argument('--end-sim-time',type=float);a=p.parse_args()
    if not 5<=a.fps<=20:raise ValueError('bounded review frame rate')
    if a.start_sim_time is not None and a.start_sim_time<0. or a.end_sim_time is not None and a.end_sim_time<0. or a.start_sim_time is not None and a.end_sim_time is not None and a.start_sim_time>=a.end_sim_time:raise ValueError('review source time interval')
    record=json.loads((a.run/'result.json').read_text());cache=json.loads(a.model_manifest.read_text());scene=json.loads((a.run/'live_view_scene.json').read_text())
    if scene['scene']['sha256']!=cache['source_scene']['sha256'] or digest(cache['path'])!=cache['sha256']:raise ValueError('exact scene/cache identity failed')
    model=mujoco.MjModel.from_binary_path(cache['path']);data=mujoco.MjData(model)
    out=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='native-transfer-recorded-review').native_path
    video=out/'full_x2_actual_transfer_review.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1280,520),fps=a.fps,codec='libx264',quality=7,ffmpeg_log_level='error');writer.send(None)
    renderer=mujoco.Renderer(model,height=480,width=640);opt=mujoco.MjvOption();opt.geomgroup[3:]=0
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.distance=2.8;camera.azimuth=130.;camera.elevation=-22.
    close=mujoco.MjvCamera();mujoco.mjv_defaultCamera(close);close.distance=.65;close.azimuth=130.;close.elevation=-18.
    generation=None;last=None;first=None;count=0;previous_host=None;source_hash=hashlib.sha256()
    try:
        with (a.run/'native_physics.jsonl').open('rb') as stream:
            for raw in stream:
                source_hash.update(raw)
                row=json.loads(raw)
                if row['generation']<1:continue
                if generation is None:generation=row['generation']
                if row['generation']!=generation:raise ValueError('unexpected reset inside review')
                if previous_host is not None and row['host_monotonic_ns']<=previous_host:raise ValueError('nonmonotonic source record')
                previous_host=row['host_monotonic_ns'];t=float(row['sim_time_s'])
                if a.start_sim_time is not None and t<a.start_sim_time or a.end_sim_time is not None and t>a.end_sim_time:continue
                if last is not None and t-last<1./a.fps-1e-5:continue
                apply_recorded_native_frame(model,data,row,visual_kinematics_only=True)
                camera.lookat[:]=np.asarray(row['bodies']['pelvis']['position_m'])+[0.,0.,.20]
                close.lookat[:]=row['bodies']['R_omnipicker_base_link']['position_m']
                renderer.update_scene(data,camera=camera,scene_option=opt);full=renderer.render().copy()
                renderer.update_scene(data,camera=close,scene_option=opt);hand=renderer.render().copy()
                frame=Image.new('RGB',(1280,520),(20,27,35));frame.paste(Image.fromarray(full),(0,40));frame.paste(Image.fromarray(hand),(640,40));draw=ImageDraw.Draw(frame)
                draw.text((12,7),(('PARTIAL / SOURCE FAILED | ' if record.get('failure') else '')+'RECORDED ACTUAL MC / MuJoCo PHYSICS | sim %.2fs')%t,fill='white')
                draw.text((12,23),'Review rendering only: no mj_step, no commands. Source episode: '+a.run.name[-16:],fill=(158,210,223))
                if count==0:frame.save(out/'preview.png')
                writer.send(np.asarray(frame).tobytes());first=t if first is None else first;last=t;count+=1
                if count%300==0:print(json.dumps({'rendered_frames':count,'source_sim_time_s':t}),flush=True)
    finally:writer.close();renderer.close()
    summary={'classification':'RENDERING_OF_RECORDED_ACTUAL_NATIVE_PHYSICS_NOT_NEW_DYNAMICS','source_run':str(a.run),'source_result_sha256':digest(a.run/'result.json'),'source_trace_sha256':source_hash.hexdigest(),'review_model_manifest':str(a.model_manifest),'review_model_manifest_sha256':digest(a.model_manifest),'frames':count,'fps':a.fps,'requested_source_interval_s':[a.start_sim_time,a.end_sim_time],'executed_argv':sys.argv,'source_sim_start_s':first,'source_sim_end_s':last,'mj_step_calls':0,'runtime_commands_sent':0,'video':str(video),'video_sha256':digest(video),'source_episode_failure':record.get('failure')}
    pg.write_new_text(out/'result.json',json.dumps(summary,indent=2)+'\n');pg.write_artifact_manifest(out);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
