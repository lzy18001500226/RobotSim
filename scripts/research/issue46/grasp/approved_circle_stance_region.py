"""Bounded seven-pose, two-height screen using the existing circle IK.

All qpos changes here belong to offline planning MjData, never the live robot.
Passing points are geometric samples, not a guaranteed continuous or balanced
region. Physical validation is recorded separately by the native runner.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
from scripts.agent import path_governance as pg

ROOT=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bottle_height_variant(source, state, height_m):
    """Only the separately requested bottle height, centre and inertia change."""
    if height_m not in (.15,.30):raise ValueError('only the two authorized bottle heights')
    root=ET.fromstring(source.read_bytes())
    geom=root.find('.//geom[@name="test_cylinder_collision"]')
    body=root.find('.//body[@name="test_cylinder"]')
    if geom is None or body is None:raise ValueError('canonical bottle missing')
    radius,half=np.fromstring(geom.get('size'),sep=' ')
    if abs(radius-.035)>1e-9 or abs(half-.15)>1e-9 or geom.get('mass')!='0.2':raise ValueError('unexpected canonical bottle')
    position=np.fromstring(body.get('pos'),sep=' ');position[2]+=height_m/2-half
    body.set('pos',' '.join(map(str,position)));geom.set('size',f'{radius} {height_m/2}')
    modified=copy.deepcopy(state)
    q=np.asarray(modified['object_pose_world']['quaternion_wxyz'])
    axis=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()[:,2]
    modified['object_pose_world']['xyz_m']=(np.asarray(modified['object_pose_world']['xyz_m'])+(height_m/2-half)*axis).tolist()
    return ET.tostring(root,encoding='utf-8',xml_declaration=True),modified

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    a=p.parse_args();capture=json.loads((a.capture/'result.json').read_text())
    if capture.get('gates',{}).get('HAL_PHYSICAL_LIFT')!='PASS':raise ValueError('start from an actual qualified lift')
    replan=Path(capture['hal_arm_runtime_replan']['path'])
    if digest(replan)!=capture['hal_arm_runtime_replan']['sha256']:raise ValueError('replan hash mismatch')
    plan=json.loads(replan.read_text());state_path=Path(plan['standing_state']);state=json.loads(state_path.read_text())
    source=Path(plan['scene']['path'])
    if digest(source)!=plan['scene']['sha256']:raise ValueError('scene hash mismatch')
    # Reuse the actual runtime's candidate ancestry and existing solver seeds.
    original=Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46/stl-standing-side-grasp-candidate-20261104T021318Z/result.json')
    candidate=json.loads(original.read_text());candidate['scene']=plan['scene']
    out=pg.create_unique_output_directory(ROOT,prefix='approved-circle-bounded-stance-region').native_path
    offsets=[('center',0.,0.,0.),('near',.02,0.,0.),('far',-.02,0.,0.),('left',0.,.02,0.),('right',0.,-.02,0.),('yaw_left',0.,0.,.03),('yaw_right',0.,0.,-.03)]
    rows=[]
    for height in (.30,.15):
        xml,base_state=bottle_height_variant(source,state,height)
        scene=out/f'bottle_{round(height*1000)}mm.xml';pg.write_new_bytes(scene,xml)
        local=copy.deepcopy(candidate);local['scene']={'path':str(scene),'sha256':digest(scene)}
        cp=out/f'candidate_{round(height*1000)}mm.json';pg.write_new_text(cp,json.dumps(local)+'\n')
        for label,dx,dy,dyaw in offsets:
            current=copy.deepcopy(base_state);base=current['base_pose_world'];base['xyz_m'][0]+=dx;base['xyz_m'][1]+=dy
            quat=np.array(base['quaternion_wxyz']);r=Rotation.from_rotvec([0,0,dyaw])*Rotation.from_quat(quat[[1,2,3,0]])
            base['quaternion_wxyz']=r.as_quat()[[3,0,1,2]].tolist()
            sp=out/f'state_{round(height*1000)}_{label}.json';pg.write_new_text(sp,json.dumps(current)+'\n')
            command=[sys.executable,str(Path(__file__).with_name('stl_approved_circle_ik.py')),'--circle-review',plan['circle_review'],'--candidate',str(cp),'--standing-state',str(sp),'--target-source','truth-diagnostic','--free-bottle-axis-roll','--physical-lift-test','--grasp-height-above-center-m',str(.10 if height==.30 else .05)]
            run=subprocess.run(command,text=True,capture_output=True,timeout=60,cwd=REPO)
            log=out/f'ik_{round(height*1000)}_{label}.log';pg.write_new_text(log,run.stdout+run.stderr)
            if run.returncode:row={'gate':'ERROR','returncode':run.returncode,'log':str(log)}
            else:
                result=json.loads(run.stdout.strip().splitlines()[-1]);detail=json.loads((Path(result['output'])/'result.json').read_text());best=detail['rows'][0]
                row={'gate':detail['gate'],'evidence':result['output'],'gates':best['gates'],'metrics':best['metrics'],'lift_sweep':detail.get('lift_preflight',{}).get('pass')}
            row.update({'bottle_height_m':height,'offset_xy_yaw':[dx,dy,dyaw],'pose_name':label,'base_pose_world':base,'dynamics':'NOT_RUN'})
            rows.append(row);print(json.dumps({k:row[k] for k in ('bottle_height_m','pose_name','gate')}),flush=True)
    record={'classification':'SIMULATION_ONLY_GEOMETRIC_STANCE_SAMPLES_NOT_CONTINUOUS_OR_DYNAMIC_QUALIFICATION','mj_step_calls':0,'source_capture':str(a.capture),'source_capture_sha256':digest(a.capture/'result.json'),'robot_geometry_changed':False,'bottle_variants':{'diameter_m':.07,'mass_kg':.2,'heights_m':[.30,.15],'bottom_on_table_preserved':True},'rows':rows,'limits':'Existing source limits, collision checks, opening and lift sweep unchanged. Interpolation, balance and physical grasp need independent dynamics.'}
    pg.write_new_text(out/'result.json',json.dumps(record,indent=2)+'\n');pg.write_artifact_manifest(out)
    print(json.dumps({'output':str(out),'passing_samples':sum(r['gate']=='PASS' for r in rows),'sample_count':len(rows)}))

if __name__=='__main__':main()
