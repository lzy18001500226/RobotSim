#!/usr/bin/env python3
"""Bounded full-contact static stand using delivered FoundationWalk PD config."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import mujoco, numpy as np, yaml
REPO=Path(__file__).resolve().parents[4]; EVIDENCE_REPO=Path('/home/lzy18001500226/projects/RobotSim'); sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg

CONFIG=EVIDENCE_REPO/'simulation/mujoco/raicom2026/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/rl/foundationwalk_config.yaml'
ORDER=('left_hip_pitch_joint','right_hip_pitch_joint','waist_yaw_joint','left_hip_roll_joint','right_hip_roll_joint','waist_pitch_joint','left_hip_yaw_joint','right_hip_yaw_joint','waist_roll_joint','left_knee_joint','right_knee_joint','left_shoulder_pitch_joint','right_shoulder_pitch_joint','left_ankle_pitch_joint','right_ankle_pitch_joint','left_shoulder_roll_joint','right_shoulder_roll_joint','left_ankle_roll_joint','right_ankle_roll_joint','left_shoulder_yaw_joint','right_shoulder_yaw_joint','left_elbow_joint','right_elbow_joint','left_wrist_yaw_joint','right_wrist_yaw_joint','left_wrist_pitch_joint','right_wrist_pitch_joint','left_wrist_roll_joint','right_wrist_roll_joint')
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--scene-result',type=Path,required=True); ap.add_argument('--scene-result-sha256',required=True); args=ap.parse_args()
 if sha(args.scene_result)!=args.scene_result_sha256: raise SystemExit('scene result hash mismatch')
 r=json.loads(args.scene_result.read_text()); item=r['generated']['host_scene']; scene=Path(item['path'])
 if sha(scene)!=item['sha256']: raise SystemExit('scene hash mismatch')
 cfg=yaml.safe_load(CONFIG.read_text()); ac=cfg['ACConfig']; q=np.array(ac['robot']['default_dof_pos']); kp=np.array(ac['robot']['kps']); kd=np.array(ac['robot']['kds'])
 if len(q)!=29 or len(kp)!=29 or len(kd)!=29: raise SystemExit('foundationwalk config shape invalid')
 out=pg.create_unique_output_directory(EVIDENCE_REPO/'reviews/issue-46',prefix='foundationwalk-static-stand').native_path; m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m)
 for name,value in zip(ORDER,q):
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name); d.qpos[m.jnt_qposadr[j]]=value
 # exact reviewed 99% STL approach pose, initialized once before stepping
 vals=(-.99,.99,.5444466735264069,-1.5124137100694064,-.5444489967884211,1.5124144523980185)
 for pre in ('L','R'):
  for suffix,value in zip(('_claw_joint','_hand_wide1_joint','_hand_narrow3_joint','_hand_narrow_loop_joint','_hand_wide3_joint','_hand_wide_loop_joint'),vals):
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,pre+suffix); d.qpos[m.jnt_qposadr[j]]=value
  a=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_'+pre+'_claw_joint'); d.ctrl[a]=.99
 mujoco.mj_forward(m,d); pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis'); start=d.xpos[pelvis].copy(); trace=[]; fail=None
 for step in range(2000):
  for name,target,p,k in zip(ORDER,q,kp,kd):
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name); a=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_'+name); dof=m.jnt_dofadr[j]; d.ctrl[a]=np.clip(p*(target-d.qpos[m.jnt_qposadr[j]])-k*d.qvel[dof]+d.qfrc_bias[dof],*m.actuator_ctrlrange[a])
  mujoco.mj_step(m,d); drift=float(np.linalg.norm(d.xpos[pelvis,:2]-start[:2])); height=float(d.xpos[pelvis,2]);
  if step%10==0: trace.append({'step':step+1,'time_s':float(d.time),'drift_m':drift,'height_m':height})
  if not np.isfinite(d.qpos).all() or drift>.1 or not .2<=height<=1.2: fail={'step':step+1,'drift_m':drift,'height_m':height}; break
 res={'classification':'SIMULATION_ONLY_FOUNDATIONWALK_PD_STATIC_STAND','mj_step_calls':step+1,'scene':{'path':str(scene),'sha256':sha(scene)},'foundationwalk_config':{'path':str(CONFIG),'sha256':sha(CONFIG)},'failure':fail,'final':trace[-1],'gates':{'SOURCE_FOUNDATIONWALK_PD_CONFIG':'PASS','FULL_CONTACT_STAND':'PASS' if fail is None else 'FAIL','GRASP':'NOT_RUN'}}
 pg.write_new_text(out/'result.json',json.dumps(res,indent=2)+'\n'); pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes()); pg.write_artifact_manifest(out); print(json.dumps({'output_wsl':str(out),'gates':res['gates'],'failure':fail},indent=2))
if __name__=='__main__': main()
