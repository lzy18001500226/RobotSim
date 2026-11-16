"""Offline camera/waist-frame comparison using recorded native state."""
from pathlib import Path
import argparse,copy,json,math,sys
import numpy as np,mujoco
from PIL import Image
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from scripts.research.issue46.perception import native_rgbd_grasp as sensor
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame


def main():
    a=argparse.ArgumentParser();a.add_argument('--reference-run',type=Path,required=True);a.add_argument('--failure-run',type=Path,required=True);a.add_argument('--left-roll-park',type=float,choices=(.15,));args=a.parse_args()
    stamp=json.loads((args.reference_run/'native_synchronized_dock_state.json').read_text())['native_snapshot']['host_monotonic_ns'];best=None
    for line in (args.reference_run/'native_physics.jsonl').open():
        row=json.loads(line)
        if best is None or abs(row['host_monotonic_ns']-stamp)<abs(best['host_monotonic_ns']-stamp):best=row
    R=best['bodies']['torso_link']['rotation'];heading=math.atan2(R[3],R[0]);raw=json.loads((args.failure_run/'fresh_rgbd/failure_sensor_input.json').read_text())['row'];B=raw['bodies']['pelvis']['rotation'];baseyaw=math.atan2(B[3],B[0]);target=(heading-baseyaw+math.pi)%(2*math.pi)-math.pi
    pred=copy.deepcopy(raw);pred['joints']['waist_yaw_joint']['q']=target
    if args.left_roll_park is not None:pred['joints']['left_shoulder_roll_joint']['q']=args.left_roll_park
    cfg=json.loads((args.failure_run/'fresh_rgbd/sensor_configuration.json').read_text());cmd=json.loads((args.failure_run/'fresh_rgbd_command.json').read_text());model,data,renderer,camera=sensor.build_sensor(Path(cfg['scene']),cmd[cmd.index('--host-model-cache')+1])
    try:
        apply_recorded_native_frame(model,data,pred,visual_kinematics_only=True)
        for name in ['head_pitch_link','torso_link']:
            b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name);pred['bodies'][name]['position_m']=data.xpos[b].tolist();pred['bodies'][name]['rotation']=data.xmat[b].tolist()
        out=pg.create_unique_output_directory('/home/lzy18001500226/projects/RobotSim/reviews/issue-46',prefix='h220-world-torso-frame-camera-static').native_path;rgb,depth,K,T=sensor.render(model,data,renderer,camera,pred);Image.fromarray(rgb).save(out/'predicted_rgb.png')
        try:
            estimate,_=sensor.estimate(rgb,depth,K,T);passed=True;error=None
            estimate={k:v for k,v in estimate.items() if not isinstance(v,np.ndarray) or v.size<=9}
        except Exception as e:estimate=None;passed=False;error=str(e)
        request=json.loads((args.failure_run/'standing_waist_request.json').read_text());ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in request['start_rad']};root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis');robots={i for i in range(model.nbody) if model.body_rootid[i]==root};bad=[]
        for f in np.linspace(0,1,21):
            apply_recorded_native_frame(model,data,raw)
            if args.left_roll_park is not None:
                j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'left_shoulder_roll_joint');data.qpos[model.jnt_qposadr[j]]=args.left_roll_park
            for n,j in ids.items():data.qpos[model.jnt_qposadr[j]]=request['start_rad'][n]+f*((target if n=='waist_yaw_joint' else request['target_rad'][n])-request['start_rad'][n])
            mujoco.mj_forward(model,data)
            for c in data.contact:
                if c.dist>=-.001:continue
                x,y=[int(model.geom_bodyid[g]) for g in [c.geom1,c.geom2]];names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,j) for j in [x,y]]
                if not (x in robots or y in robots) or 'world' in names:continue
                if set(names)=={'pelvis','left_hip_roll_link'} and c.dist>=-.005:continue
                bad.append(dict(fraction=float(f),pair=names,distance_m=float(c.dist)))
        record=dict(classification='STATIC_WORLD_TORSO_FRAME_PREDICTION_NOT_DYNAMICS',mj_step_calls=0,source_run=str(args.failure_run),reference_run=str(args.reference_run),reference_native_stamp_ns=best['host_monotonic_ns'],reference_torso_world_yaw_rad=heading,measured_pelvis_yaw_rad=baseyaw,corrected_waist_yaw_rad=target,camera_fit_pass=passed,estimate=estimate,failure=error,waist_sweep_violations=bad,camera_mount_fov_resolution_unchanged=True)
        pg.write_new_text(out/'result.json',json.dumps(record,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n');pg.write_artifact_manifest(out);print(json.dumps(dict(output=str(out),camera_fit_pass=passed,waist_sweep_clear=not bad,yaw_target_rad=target,failure=error)),flush=True)
    finally:renderer.close()


if __name__=='__main__':main()
