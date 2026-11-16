"""Audit held native contact forces against the human-approved source pad faces."""
from pathlib import Path
import argparse,hashlib,sys
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
import json
import numpy as np
import mujoco,trimesh
from scipy.spatial import cKDTree
from scripts.agent import path_governance as pg
from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory
def require(condition,message):
 if not condition:raise ValueError(message)

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True);args=parser.parse_args()
root=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46');capture=args.capture;record=json.loads((capture/'result.json').read_text());require(record['gates']['HAL_PHYSICAL_LIFT']=='PASS','capture has no accepted physical lift')
overlay_path=Path(record['claw_control_overlay']['path']);require(hashlib.sha256(overlay_path.read_bytes()).hexdigest()==record['claw_control_overlay']['sha256'],'overlay hash mismatch')
overlay=json.loads(overlay_path.read_text());scene=overlay['generated']['host_scene'];require(hashlib.sha256(Path(scene['path']).read_bytes()).hexdigest()==scene['sha256'],'scene hash mismatch')
model=mujoco.MjModel.from_xml_path(overlay['generated']['host_scene']['path'])
plan_path=Path(record['hal_arm_runtime_replan']['path']);require(hashlib.sha256(plan_path.read_bytes()).hexdigest()==record['hal_arm_runtime_replan']['sha256'],'replan hash mismatch');plan=json.loads(plan_path.read_text());review_path=Path(plan['circle_review']);require(hashlib.sha256(review_path.read_bytes()).hexdigest()==plan['circle_review_sha256'],'review hash mismatch')
review=json.loads(review_path.read_text());rows=[json.loads(l) for l in (capture/'native_physics.jsonl').read_text().splitlines()]
last=record['physical_lift_evidence']['tail'][-1]['sim_time_s'];generation=record['physical_lift_evidence']['tail'][-1]['generation'];held=[r for r in rows if r['generation']==generation and last-1.-1e-8<=r['sim_time_s']<=last+1e-8]
require(len(held)>=25 and held[0]['engine_version']==mujoco.mj_version(),'held window missing or wrong audit engine')
hand=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'R_omnipicker_base_link');data=mujoco.MjData(model);apply_recorded_native_frame(model,data,held[0]);trajectory=load_registered_trajectory(REPO/'simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic')
for n,v in trajectory.pose(40).positions_rad.items():
 n='R_claw_joint' if n=='right_claw_joint' else n;j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n);data.qpos[model.jnt_qposadr[j]]=v
mujoco.mj_kinematics(model,data);H=data.xmat[hand].reshape(3,3);hp=data.xpos[hand];prepared={}
for body,pad in review['pads'].items():
 spec=next(x for x in review['geoms'] if x['id']==pad['geom']);ref=review['meshes'][str(spec['mesh'])];vertices=np.array(ref['vertices']).reshape(-1,3);faces=np.array(ref['faces']).reshape(-1,3);g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,body+'__visual_0');mid=model.geom_dataid[g]
 N=H.T@data.geom_xmat[g].reshape(3,3);p=H.T@(data.geom_xpos[g]-hp);R=np.array(review['poses'][40]['rotation'][str(spec['id'])]).reshape(3,3);q=np.array(review['poses'][40]['position'][str(spec['id'])]);mapped=(vertices@R.T+q-p)@N
 native=np.array(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]);error=max(cKDTree(mapped).query(native)[0].max(),cKDTree(native).query(mapped)[0].max());require(error<1e-5,'source/native mesh mapping failed')
 prepared[body]=(g,trimesh.Trimesh(mapped,faces,process=False),set(pad['bottle_arc_review']['source_face_indices']),float(error))
observations=[];min_approved={b:float('inf') for b in prepared};max_outside=0.;max_distance=0.;counts=[]
for row in held:
 apply_recorded_native_frame(model,data,row);approved={b:0. for b in prepared};outside=0.;count=0
 for body,(g,mesh,selected,error) in prepared.items():
  contacts=[c for c in row['contacts'] if 'test_cylinder' in (c['body1'],c['body2']) and body in (c['body1'],c['body2']) and c['local_force_torque'][0]>1e-4]
  if not contacts:continue
  points=np.array([c['position_m'] for c in contacts]);local=(points-data.geom_xpos[g])@data.geom_xmat[g].reshape(3,3)
  closest,distance,ids=trimesh.proximity.closest_point_naive(mesh,local)
  for c,dist,face in zip(contacts,distance,ids):
   force=c['local_force_torque'][0];inside=int(face) in selected;count+=1;max_distance=max(max_distance,float(dist))
   if inside:approved[body]+=force
   else:outside+=force
   observations.append({'sim_time_s':row['sim_time_s'],'body':body,'native_geom':c['geom2'] if c['body2']==body else c['geom1'],'normal_n':force,'nearest_source_face':int(face),'approved_lower_arc':inside,'proxy_contact_to_source_surface_m':float(dist)})
 for b in approved:min_approved[b]=min(min_approved[b],approved[b])
 max_outside=max(max_outside,outside);counts.append(count)
passed=bool(len(held)>=25 and min(min_approved.values())>=.1 and max_outside<=.1)
result={'classification':'POSTHOC_RECORDED_FORCE_CONTACT_SOURCE_REGION_AUDIT','pass':passed,'capture':str(capture),'mj_step_calls':0,'held_frames':len(held),'minimum_contacts_per_frame':min(counts),'maximum_contacts_per_frame':max(counts),'minimum_approved_region_normal_force_n':min_approved,'maximum_off_region_normal_force_per_frame_n':max_outside,'maximum_proxy_contact_to_source_surface_mm':max_distance*1000,'mesh_mapping_error_m':{b:v[3] for b,v in prepared.items()},'observations':observations,'scene_sha256':scene['sha256'],'review_sha256':plan['circle_review_sha256'],'native_trace_sha256':hashlib.sha256((capture/'native_physics.jsonl').read_bytes()).hexdigest(),'capture_result_sha256':hashlib.sha256((capture/'result.json').read_bytes()).hexdigest(),'source_engine':held[0]['engine_version'],'audit_engine':mujoco.mj_version()}
out=pg.create_unique_output_directory(root,prefix='native-lift-pad-region-audit').native_path;pg.write_new_text(out/'result.json',json.dumps(result,indent=2)+'\n');pg.write_artifact_manifest(out);print(out);print(json.dumps({k:v for k,v in result.items() if k!='observations'}))
