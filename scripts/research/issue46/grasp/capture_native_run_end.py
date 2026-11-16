"""Background rendering of the last measured native state; no desktop input."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame,digest


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);args=a.parse_args();run=args.run
    if not (run/'result.json').is_file() or not (run/'native_latest.json').is_file():raise RuntimeError('no completed native state/result to capture')
    result=json.loads((run/'result.json').read_text());row=json.loads((run/'native_latest.json').read_text());manifest=json.loads((run/'live_view_scene.json').read_text())
    out=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='native-run-end-capture').native_path
    cachepath=run/'live_review_model_manifest.json'
    if not cachepath.is_file():
        from scripts.research.issue46.grasp.live_retreat_turn_review import matching_review_model
        cachepath=matching_review_model(manifest,out)
    cache=json.loads(cachepath.read_text())
    if cache['source_scene']['sha256']!=manifest['scene']['sha256'] or digest(cache['path'])!=cache['sha256']:raise RuntimeError('end capture scene/cache identity mismatch')
    model=mujoco.MjModel.from_binary_path(cache['path']);data=mujoco.MjData(model);apply_recorded_native_frame(model,data,row,visual_kinematics_only=True)
    renderer=mujoco.Renderer(model,height=480,width=640);opt=mujoco.MjvOption();opt.geomgroup[3:]=0
    failure=result.get('failure');reason=(failure.get('message') or failure.get('gate') or str(failure)) if isinstance(failure,dict) else str(failure)
    contacts=[c for c in row.get('contacts',[]) if 'test_cylinder' in (c['body1'],c['body2']) and any(n in (c['body1'],c['body2']) for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')) and c['local_force_torque'][0]>.1]
    views=[]
    for name,body,distance in [('overview','pelvis',2.4),('hand','R_omnipicker_base_link',.65)]:
        camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.lookat[:]=np.asarray(row['bodies'][body]['position_m'])+([0,0,.2] if body=='pelvis' else [0,0,0]);camera.distance=distance;camera.azimuth=135.;camera.elevation=-20.
        renderer.update_scene(data,camera=camera,scene_option=opt);pixels=renderer.render().copy();image=Image.new('RGB',(640,528),(15,23,30));image.paste(Image.fromarray(pixels),(0,48));draw=ImageDraw.Draw(image)
        draw.text((8,4),'LAST ACTUAL NATIVE STATE - background rendering, no physics',fill='white');draw.text((8,18),'sim %.3fs | bottle-pad contacts %d | %s'%(row['sim_time_s'],len(contacts),name),fill='white');draw.text((8,32),'STOP: '+reason[:90],fill=(255,190,120))
        path=out/(name+'.png');image.save(path);views.append(str(path))
    renderer.close();record=dict(classification='BACKGROUND_RENDER_OF_LAST_RECORDED_ACTUAL_NATIVE_STATE',run=str(run),source_result_sha256=digest(run/'result.json'),source_state_sha256=digest(run/'native_latest.json'),source_scene_sha256=manifest['scene']['sha256'],source_pid=row['process_pid'],generation=row['generation'],sim_time_s=row['sim_time_s'],failure=failure,bottle_pad_contact_count=len(contacts),images=views,mj_step_calls=0,desktop_input_events=0)
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
