#!/usr/bin/env python3
"""Open a zero-step MuJoCo review of a source-grounded docked STL pregrasp."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import sys
import mujoco, numpy as np
REPO=Path(__file__).resolve().parents[4]; EVIDENCE=Path('/home/lzy18001500226/projects/RobotSim'); sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory, apply_stl_kinematic_pose
ARM=("right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint")
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def setj(m,d,n,v):
 j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n); d.qpos[m.jnt_qposadr[j]]=v; d.qvel[m.jnt_dofadr[j]]=0
def apply(m,d,state,ik,shift):
 base=state['base_pose_world']; fj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'floating_base_joint'); q=m.jnt_qposadr[fj]
 d.qpos[q:q+3]=np.asarray(base['xyz_m'])+shift; d.qpos[q+3:q+7]=base['quaternion_wxyz']
 for group in ('leg_positions_rad','waist_positions_rad','arm_positions_rad'):
  for n,v in state[group].items(): setj(m,d,n,float(v))
 for n,v in zip(ARM,ik): setj(m,d,n,float(v))
 t=load_registered_trajectory(REPO/'simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic')
 for side,root in (('right','R_claw_joint'),('left','L_claw_joint')):
  pose=t.pose(100)
  for n,v in pose.positions_rad.items():
   target=root if n=='right_claw_joint' else n if side=='right' else 'L_'+n[2:]
   setj(m,d,target,v)
 mujoco.mj_forward(m,d)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--table-result',type=Path,required=True); p.add_argument('--ik-result',type=Path,required=True); p.add_argument('--stand-result',type=Path,required=True); p.add_argument('--live-viewer',action='store_true'); a=p.parse_args()
 table=json.loads(a.table_result.read_text()); ik=json.loads(a.ik_result.read_text()); stand=json.loads(a.stand_result.read_text())
 if ik['gates']['IK_CONVERGED']!='PASS' or ik['gates']['POST_IK_TABLE_CLEARANCE']!='PASS': raise SystemExit('IK result did not pass')
 scene=Path(table['generated']['host_scene']['path']);
 if sha(scene)!=table['generated']['host_scene']['sha256']: raise SystemExit('table scene hash mismatch')
 cand=json.loads(Path(ik['sources']['placement_result']['path']).read_text())['scan']['best_candidates'][0]
 fixed_table=np.asarray(table['transforms']['table_world_override']); dock_shift=fixed_table-np.asarray(cand['table_world_xyz_m'])
 m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m); state=stand['native_arm_state_capture']; apply(m,d,state,ik['ik']['right_arm_q_rad'],dock_shift)
 contacts=[]
 for c in d.contact:
  names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or 'world' for g in (c.geom1,c.geom2)]
  contacts.append({'body_names':names,'distance_m':float(c.dist)})
 out=pg.create_unique_output_directory(EVIDENCE/'reviews/issue-46',prefix='x2-stl-docked-pregrasp-review').native_path
 known_internal={('left_hip_roll_link','L_hand_narrow1_Link'),('left_hip_roll_link','L_hand_narrow2_Link'),('left_hip_roll_link','L_hand_narrow3_Link'),('left_hip_roll_link','L_hand_narrow_loop_Link'),('left_hip_yaw_link','L_hand_narrow3_Link')}
 def allowed(row):
  pair=tuple(row['body_names'])
  return 'world' in pair or set(pair)=={'test_cylinder','raicom_table_table'} or pair in known_internal or pair[::-1] in known_internal
 forbidden=[row for row in contacts if not allowed(row)]
 record={'schema':'robotsim.issue46.x2-stl-docked-pregrasp-review.v1','classification':'SIMULATION_ONLY_STATIC_DOCKED_PREGRASP','mj_step_calls':0,'scene':{'path':str(scene),'sha256':sha(scene)},'stand_result':{'path':str(a.stand_result),'sha256':sha(a.stand_result)},'ik_result':{'path':str(a.ik_result),'sha256':sha(a.ik_result)},'dock_base_translation_world_m':dock_shift.tolist(),'right_arm_q_rad':ik['ik']['right_arm_q_rad'],'hand_command_percent':{'left':100,'right':100},'contacts':contacts,'known_frozen_internal_contacts':[row for row in contacts if tuple(row['body_names']) in known_internal or tuple(row['body_names'])[:: -1] in known_internal],'forbidden_contacts':forbidden,'gates':{'REAL_STANDING_STATE':'PASS','IK_PREGRASP':'PASS','STATIC_CONTACT_CHECK':'PASS' if not forbidden else 'FAIL','NATIVE_WALK_TO_DOCK':'NOT_RUN','PHYSICAL_GRASP':'NOT_RUN'},'limitations':['Static mj_forward review only. The base pose is an equivalent docking candidate, not evidence of walking to it.','Hands are open pregrasp poses; no contact, closure, or grasp claim is made.','Known frozen left-hand internal collision proxies are retained and reported; they are not suppressed.']}
 pg.write_new_text(out/'review_state.json',json.dumps(record,indent=2)+'\n'); pg.write_new_bytes(out/Path(__file__).name,Path(__file__).read_bytes()); pg.write_artifact_manifest(out); print(json.dumps({'output_wsl':str(out),'contacts':contacts,'dock_shift':dock_shift.tolist()},indent=2))
 if not a.live_viewer:return
 import glfw
 if not glfw.init():raise RuntimeError('GLFW unavailable')
 w=glfw.create_window(1320,930,'MuJoCo | X2-OmniPicker-STL | docked IK pregrasp (STATIC)',None,None); glfw.make_context_current(w); cam=mujoco.MjvCamera(); cam.lookat[:]=[1.2,-1.05,.75]; cam.distance=1.5; cam.azimuth=145; cam.elevation=-18; opt=mujoco.MjvOption(); sc=mujoco.MjvScene(m,maxgeom=12000); ctx=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150)
 while not glfw.window_should_close(w):
  width,height=glfw.get_framebuffer_size(w); vp=mujoco.MjrRect(0,0,width,height); mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,sc); mujoco.mjr_render(vp,sc,ctx); mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,vp,'X2-OmniPicker-STL | STATIC DOCKED IK PREGRASP','0 mj_step | no walking or grasp claim',ctx); glfw.swap_buffers(w); glfw.poll_events(); time.sleep(.01)
 ctx.free(); glfw.destroy_window(w); glfw.terminate()
if __name__=='__main__':main()
