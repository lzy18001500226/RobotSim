"""Replay saved state with mj_forward only: independent slip/clearance audit."""
from episode import *
from collections import defaultdict

for rid in ['dex3-contact-attempt01c','dex3-contact-attempt02']:
    out=OUT/rid;m=mj.MjModel.from_xml_path(str(out/'model.xml'));d=mj.MjData(m)
    rr=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
    names=rr[0]['joint_names'];qa,va=indices(m,names);bj=jid(m,'bottle_free');bqa=m.jnt_qposadr[bj];bva=m.jnt_dofadr[bj]
    rotor=[];forces=[];limits=[];slips=defaultdict(list);clearances=defaultdict(list)
    physical=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    station=[g for g in physical if name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])=='m0_table']
    robot=[g for g in physical if 'world'!=name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) and m.geom_bodyid[g]!=bid(m,'green_box') and g not in station]
    velocity_report={};position_report={};effort_report={}
    for k,r in enumerate(rr):
        if k%5 and r['stage']!='LOAD_TRANSFER':continue
        # Offline reconstruction of recorded states, never active-rollout command/state mutation.
        d.qpos[qa]=r['q'];d.qvel[va]=r['v'];d.qpos[bqa:bqa+3]=r['bottle_pos'];d.qpos[bqa+3:bqa+7]=r['bottle_quat'];mj.mj_forward(m,d)
        if r['stage'] in ('SETTLE','PREGRASP','APPROACH'):
            for g in robot:
                for t in station+[gid(m,'bottle_body'),gid(m,'bottle_shoulder'),gid(m,'bottle_neck'),gid(m,'bottle_cap')]:
                    seg=np.zeros(6);distance=mj.mj_geomDistance(m,d,g,t,1.,seg)
                    # MuJoCo 3.3.6 legacy distance query can return zero for far mesh/box pairs.
                    # A positive world-AABB gap is an independent conservative separation proof.
                    boxes=[]
                    for geom in [g,t]:
                        R=d.geom_xmat[geom].reshape(3,3);center=d.geom_xpos[geom]+R@m.geom_aabb[geom,:3];half=abs(R)@m.geom_aabb[geom,3:]
                        boxes.append((center-half,center+half))
                    gap=np.maximum(0,np.maximum(boxes[0][0]-boxes[1][1],boxes[1][0]-boxes[0][1]))
                    aabb_lower_bound=float(np.linalg.norm(gap))
                    clearances[r['stage']].append(max(float(distance),aabb_lower_bound))
        if r['stage']=='LOAD_TRANSFER':
            prev=rr[max(0,k-1)];nxt=rr[min(len(rr)-1,k+1)];dt=nxt['time']-prev['time'];linear=(np.array(nxt['bottle_pos'])-prev['bottle_pos'])/dt
            Ra=Rotation.from_quat(np.array(prev['bottle_quat'])[[1,2,3,0]]);Rb=Rotation.from_quat(np.array(nxt['bottle_quat'])[[1,2,3,0]]);angular=(Rb*Ra.inv()).as_rotvec()/dt
            for c in r['contacts']:
                if 'right_hand' not in c['other']:continue
                body=bid(m,c['other']);p=np.array(c['point']);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,p,body)
                hand_velocity=jp@d.qvel;bottle_velocity=linear+np.cross(angular,p-r['bottle_pos']);relative=hand_velocity-bottle_velocity;n=np.array(c['normal']);tangent=relative-n*(n@relative)
                slips[c['other']].append(float(np.linalg.norm(tangent)))
        for i,n in enumerate(names):
            velocity_report[n]=max(velocity_report.get(n,0),abs(r['v'][i]));effort_report[n]=max(effort_report.get(n,0),abs(r['effort'][i]))
            position_report[n]=max(position_report.get(n,0),SOURCE_LIMITS[n]['lower']-r['q'][i],r['q'][i]-SOURCE_LIMITS[n]['upper'])
    audit={'physics_steps':0,'method':'offline recorded-q reconstruction, source FK Jacobians; bottle twist central finite difference at 500Hz. Not a new rollout.',
      'robot_station_bottle_min_clearance_m':{k:min(v) for k,v in clearances.items()},
      'load_tangent_relative_velocity_m_s':{n:{'mean':float(np.mean(v)),'p95':float(np.percentile(v,95)),'max':max(v)} for n,v in slips.items()},
      'sampled_peak_joint_velocity':velocity_report,'sampled_peak_effort':effort_report,'sampled_URDF_range_violation_rad':position_report}
    (out/'offline_audit.json').write_text(json.dumps(audit,indent=2));print(rid,json.dumps({k:v for k,v in audit.items() if k not in ['sampled_peak_joint_velocity','sampled_peak_effort','sampled_URDF_range_violation_rad']},indent=2))
