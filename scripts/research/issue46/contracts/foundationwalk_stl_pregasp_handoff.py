#!/usr/bin/env python3
"""Classify FoundationWalk stop samples for frozen STL pregrasp handoff.

This is an explicit static contract: it replays recorded full-body native
MuJoCo qpos samples only through ``mj_forward`` and reports source-table
clearance plus wrist-to-cylinder reach.  It never calls ``mj_step`` or writes
runtime qpos.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
from collections import deque
import mujoco, numpy as np
REPO=Path(__file__).resolve().parents[4]; EVIDENCE=Path('/home/lzy18001500226/projects/RobotSim'); sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 a=argparse.ArgumentParser(); a.add_argument('--scene',type=Path,required=True); a.add_argument('--scene-sha256',required=True); a.add_argument('--trace',type=Path,required=True); a.add_argument('--trace-sha256',required=True); a.add_argument('--metadata',type=Path,required=True); a.add_argument('--metadata-sha256',required=True); a.add_argument('--object-body',default='test_cylinder'); args=a.parse_args()
 if sha(args.scene)!=args.scene_sha256 or sha(args.trace)!=args.trace_sha256 or sha(args.metadata)!=args.metadata_sha256: raise SystemExit('source hash mismatch')
 m=mujoco.MjModel.from_xml_path(str(args.scene)); wrist=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_roll_link'); cyl=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,args.object_body)
 if wrist < 0 or cyl < 0: raise SystemExit(f'missing wrist or object body: wrist={wrist}, object={args.object_body!r}/{cyl}')
 meta=json.loads(args.metadata.read_text()); source_joints=meta['model_metadata']['joint_names']
 rows=[]; candidates=deque(maxlen=3)
 with args.trace.open(encoding='utf-8',errors='replace') as stream:
  for line in stream:
   try: r=json.loads(line)
   except json.JSONDecodeError: continue
   if r.get('event')!='post_step' or not isinstance(r.get('qpos'),list): continue
   candidates.append(r)
 for r in candidates:
  d=mujoco.MjData(m)
  for item in source_joints:
   name=item.get('name')
   if not name:
    continue
   sid=int(item['qpos_adr']); did=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
   if did>=0:
    width=7 if int(item['type'])==int(mujoco.mjtJoint.mjJNT_FREE) else 1; d.qpos[m.jnt_qposadr[did]:m.jnt_qposadr[did]+width]=r['qpos'][sid:sid+width]
  for prefix in ('L','R'):
   for suffix,value in zip(('_claw_joint','_hand_wide1_joint','_hand_narrow3_joint','_hand_narrow_loop_joint','_hand_wide3_joint','_hand_wide_loop_joint'),(-.99,.99,.5444466735264069,-1.5124137100694064,-.5444489967884211,1.5124144523980185)):
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,prefix+suffix); d.qpos[m.jnt_qposadr[j]]=value
  mujoco.mj_forward(m,d); contacts=[]; unexpected=[]
  for c in d.contact:
   b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom1])) or ''; b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom2])) or ''
   if 'table' in b1 or 'table' in b2:
    pair=[b1,b2]; contacts.append(pair)
    if 'test_cylinder' not in pair:
     unexpected.append(pair)
  rows.append({'time_s':r.get('sim_time_s'),'pelvis_xyz_m':d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')].tolist(),'wrist_to_cylinder_m':float(np.linalg.norm(d.xpos[wrist]-d.xpos[cyl])),'table_contacts':contacts,'unexpected_table_contacts':unexpected,'valid':not unexpected})
 out=pg.create_unique_output_directory(EVIDENCE/'reviews/issue-46',prefix='foundationwalk-stl-pregrasp-handoff').native_path
 valid=[r for r in rows if r['valid']]; result={'classification':'STATIC_FOUNDATIONWALK_STL_PREGRASP_HANDOFF','mj_step_calls':0,'scene':{'path':str(args.scene),'sha256':sha(args.scene)},'trace':{'path':str(args.trace),'sha256':sha(args.trace)},'object_body':args.object_body,'samples':rows,'closest_clear_sample':min(valid,key=lambda x:x['wrist_to_cylinder_m']) if valid else None,'gates':{'SOURCE_SAMPLES_REPLAYED':'PASS' if rows else 'FAIL','TABLE_CLEARANCE':'PASS' if valid else 'FAIL','IK':'NOT_RUN','GRASP':'NOT_RUN'}}
 pg.write_new_text(out/'result.json',json.dumps(result,indent=2)+'\n'); pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes()); pg.write_artifact_manifest(out); print(json.dumps({'output_wsl':str(out),'gates':result['gates'],'closest':result['closest_clear_sample']},indent=2))
if __name__=='__main__': main()
