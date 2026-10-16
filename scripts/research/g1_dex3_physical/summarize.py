from baseline import *
from episode import SOURCE_LIMITS
from collections import defaultdict
m=build(OUT/'model_inventory.xml')
diffs=[]
for j in range(m.njnt):
 n=name(m,mj.mjtObj.mjOBJ_JOINT,j)
 if n not in SOURCE_LIMITS:continue
 lim=SOURCE_LIMITS[n]
 if max(abs(m.jnt_range[j]-[lim['lower'],lim['upper']]))>5e-6:diffs.append({'joint':n,'MJCF_range':m.jnt_range[j].tolist(),'URDF_range':[lim['lower'],lim['upper']]})
print(json.dumps(diffs,indent=2));(OUT/'source_range_discrepancies.json').write_text(json.dumps(diffs,indent=2))
for rid in ['dex3-contact-attempt01c','dex3-contact-attempt02']:
 rr=[json.loads(s) for s in (OUT/rid/'trace.jsonl').read_text().splitlines()];load=[r for r in rr if r['stage']=='LOAD_TRANSFER'];hold=[r for r in rr if r['stage']=='GRIP_HOLD']
 arrays={}
 for n in HAND:
  i=rr[-1]['joint_names'].index(n);arrays[n]={'peak_qvel':max(abs(r['v'][i]) for r in rr),'peak_effort':max(abs(r['effort'][i]) for r in rr),'range':[min(r['q'][i] for r in rr),max(r['q'][i] for r in rr)]}
 slip={}
 for link in ['right_hand_thumb_2_link','right_hand_index_1_link','right_hand_middle_1_link']:
  pts=[];normals=[];nf=[];tf=[]
  for r in load:
   for c in r['contacts']:
    if c['other']!=link:continue
    R=Rotation.from_quat(np.array(r['bottle_quat'])[[1,2,3,0]]).as_matrix()
    pts.append(R.T@(np.array(c['point'])-r['bottle_pos']));normals.append(R.T@np.array(c['normal']));nf.append(c['contact_force'][0]);tf.append(np.linalg.norm(c['contact_force'][1:3]))
  if pts:
   pts=np.array(pts);slip[link]={'contact_migration_max_m':float(np.linalg.norm(pts-pts[0],axis=1).max()),'normal_rotation_max_deg':float(np.degrees(np.arccos(np.clip(np.array(normals)@normals[0],-1,1))).max()),'normal_force_mean_N':float(np.mean(nf)),'tangent_force_mean_N':float(np.mean(tf))}
 summary={'joint_metrics':arrays,'load':{'wrist_delta_m':(np.array(load[-1]['wrist_pos'])-load[0]['wrist_pos']).tolist(),'bottle_delta_m':(np.array(load[-1]['bottle_pos'])-load[0]['bottle_pos']).tolist(),'hand_force_N':{'mean':float(np.mean([r['hand_z'] for r in load])),'p05':float(np.percentile([r['hand_z'] for r in load],5)),'p95':float(np.percentile([r['hand_z'] for r in load],95)),'peak':max(r['hand_z'] for r in load)},'table_N_mean':float(np.mean([r['table_z'] for r in load])),'contact_migration':slip},'hold_contact_fraction':float(np.mean([r['thumb'] and r['opposing'] for r in hold]))}
 (OUT/rid/'measured_summary.json').write_text(json.dumps(summary,indent=2));print(rid,json.dumps(summary['load'],indent=2))
