"""Read back every positive static witness; never executes physics."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np
import mujoco
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO))
from scripts.research.issue46.grasp.stance_region_static_review import Study,PADS
from scripts.agent import path_governance as pg


def main(out):
    began=time.monotonic();study=Study();m,d=study.model,study.data;result=json.loads((out/'result.json').read_text());inputs=json.loads((out/'inputs.json').read_text())
    def forbidden(*a,**k):raise AssertionError('physics forbidden')
    mujoco.mj_step=forbidden;mujoco.mj_step1=forbidden;mujoco.mj_step2=forbidden
    failures=[];frames=0;max_eq=0.;max_anchor=0.;max_bound=0.;max_object=0.;worst_pad=0.;tested=set()
    for point in result['points']:
        if point['id'] in tested:failures.append({'point':point['id'],'reason':'duplicate'})
        tested.add(point['id'])
        for label in study.templates:
            q=study.relocate(study.templates[label],point['anchor_xy_m'],point['yaw_world_rad']);study.apply(q)
            if np.linalg.norm(study.foot_anchor()-point['anchor_xy_m'])>1e-8:failures.append({'point':point['id'],'reason':'anchor transform'})
        if point['status']!='STATIC_WITNESS':continue
        witness=json.loads((out/(point['id']+'.json')).read_text())['witness']
        if not all(any(s['phase']==phase for s in witness['path']) for phase in ('PREPARATION','REACH','APPROACH','CLOSURE','LIFT')):failures.append({'point':point['id'],'reason':'missing phase'})
        for item in witness['path']:
            q=np.array(item['qpos']);study.apply(q);frames+=1
            anchor=float(np.linalg.norm(study.foot_anchor()-point['anchor_xy_m']));max_anchor=max(max_anchor,anchor)
            expected=study.fixed_object.copy()
            if item['phase']=='LIFT':expected[2]=q[study.obj+2]
            err=float(np.max(np.abs(q[study.obj:study.obj+7]-expected)));max_object=max(max_object,err)
            if anchor>1e-8 or err>1e-12:failures.append({'point':point['id'],'reason':'anchor/object pose invariant'})
            eq=np.array(d.efc_pos)[np.array(d.efc_type)==mujoco.mjtConstraint.mjCNSTR_EQUALITY];value=float(np.max(np.abs(eq))) if len(eq) else 0.;max_eq=max(max_eq,value)
            if value>1e-10:failures.append({'point':point['id'],'reason':'frozen polynomial residual','rad':value})
            for j in range(m.njnt):
                if not m.jnt_limited[j]:continue
                value=q[m.jnt_qposadr[j]];lo,hi=m.jnt_range[j];violation=max(0.,lo-value,value-hi);max_bound=max(max_bound,float(violation))
                if violation>1e-6:failures.append({'point':point['id'],'reason':'source limit','joint':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j),'violation_rad':float(violation)})
            for c in d.contact:
                a,b=[int(m.geom_bodyid[g]) for g in (c.geom1,c.geom2)];na,nb=study.names[a],study.names[b]
                intended=item['phase'] in ('CLOSURE','LIFT') and 'test_cylinder' in (na,nb) and PADS.intersection((na,nb))
                if intended:worst_pad=min(worst_pad,float(c.dist))
                support=('world' in (na,nb) and any(n in ('left_ankle_roll_link','left_ankle_pitch_link','right_ankle_roll_link','right_ankle_pitch_link') for n in (na,nb)))
                if (a in study.robot or b in study.robot) and not support and c.dist<(-.001 if intended else -.00001):failures.append({'point':point['id'],'reason':'contact','phase':item['phase'],'bodies':[na,nb],'distance_m':float(c.dist)})
    arrays_same=all(hashlib.sha256(getattr(m,k).tobytes()).hexdigest()==h for k,h in study.model_arrays.items())
    record={'pass':not failures and arrays_same,'physics_calls':0,'checked_points':len(tested),'positive_witness_frames':frames,'max_polynomial_residual_rad':max_eq,'max_ankle_anchor_error_m':max_anchor,'max_fixed_object_error_m_excluding_hypothetical_lift_z':max_object,'maximum_all_limited_joint_violation_rad':max_bound,'worst_intended_pad_penetration_m':worst_pad,'frozen_model_arrays_unchanged':arrays_same,'source_scene_sha256':study.model_identity,'wall_s':time.monotonic()-began,'failures':failures[:40],'failure_count':len(failures),'not_checked':['continuous swept path','dynamics and force','walking to stance','perception visibility']}
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));assert record['pass'],record
    if not (out/'artifact_paths.json').exists():pg.write_artifact_manifest(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);main(parser.parse_args().input_dir)
