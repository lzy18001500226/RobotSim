#!/usr/bin/env python3
"""Bounded Candidate B travel analysis and one-motor diagnostic. No objects.

Official assets are immutable. Positive CAD angles are analytic continuation,
not STEP-measured motion. Existing Candidate B transmission remains unverified.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import issue46_urdf_pipeline as p
from issue46_dpoint_legal_motion import TriangleBVH

GRID = .005
APERTURE_X, APERTURE_Z = 0., .115


def targets(law, q):
    if not math.isfinite(q) or not p.PHASE-1e-12 <= q <= p.PHASE+1+1e-12:
        raise ValueError('outside unchanged official root limits')
    out, closure = {}, 0.
    for side, names in p.GROUPS.items():
        v, residual = law.branch(side, q if side == 'narrow' else -q)
        closure = max(closure, residual)
        for i, n in enumerate(names):
            out[n] = float(v[i]+((p.PHASE if side == 'narrow' else -p.PHASE) if i == 0 else 0.))
    return out, closure


def set_static(model, data, law, q):
    values, closure = targets(law, q)
    for n, v in values.items():
        data.qpos[p.addresses(model)[n]] = v
    mujoco.mj_forward(model, data)
    return values, closure


def ray_surface(triangles, x, z, side):
    """Facing triangle surface on one specified +Y ray, not a tip distance."""
    o = np.array([x, -.5, z])
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    h = np.cross([0., 1., 0.], e2)
    det = np.einsum('ij,ij->i', e1, h)
    inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det) > 1e-14)
    s = o-triangles[:, 0]
    u = inv*np.einsum('ij,ij->i', s, h)
    cross = np.cross(s, e1)
    v, t = inv*cross[:, 1], inv*np.einsum('ij,ij->i', e2, cross)
    ids = np.flatnonzero((np.abs(det) > 1e-14)&(u >= -1e-9)&(v >= -1e-9)&(u+v <= 1+1e-9)&(t > 0))
    if not len(ids):
        return None
    ys = -.5+t[ids]
    index = int(ids[np.argmax(ys) if side == 'narrow' else np.argmin(ys)])
    normal = np.cross(e1[index], e2[index])
    normal /= np.linalg.norm(normal)
    return {'y_m': float(-.5+t[index]), 'triangle': index, 'normal_world': normal.tolist()}


class SourceSurfaces:
    def __init__(self, model, derived):
        self.model, self.local, self.frames = model, {}, {}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if '__visual__' not in name:
                continue
            stem = name.split('__visual__')[1]
            self.local[g] = p.stl_vertices(p.UPSTREAM/'meshes'/f'{stem}.stl').reshape(-1, 3, 3)
            body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g])
            self.frames[g]=p.origin(derived.find(f'link[@name="{body_name}"]').find('visual'))

    def triangles(self, data, gid):
        # Original STL is in the URDF mesh frame. geom_xmat/xpos instead map
        # MuJoCo's recentered/principal-axis compiled vertices; using those for
        # raw STL would apply mesh preprocessing twice and fabricate overlaps.
        pos,rotation=self.frames[gid]
        body=self.model.geom_bodyid[gid]
        return (self.local[gid]@rotation.T+pos)@data.xmat[body].reshape(3,3).T+data.xpos[body]

    def gid(self, side):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'R_hand_{side}3_Link__visual__{side}3_Link')

    def aperture(self, data):
        tri = {s: self.triangles(data, self.gid(s)) for s in p.GROUPS}
        sections = []
        for z in (.110, .1125, APERTURE_Z, .1175, .120):
            hits = {s: ray_surface(tri[s], APERTURE_X, z, s) for s in p.GROUPS}
            a, b = hits['narrow'], hits['wide']
            sections.append({'x_m': APERTURE_X, 'z_m': z, 'width_mm': None if a is None or b is None else (b['y_m']-a['y_m'])*1000,
                             'surfaces': hits})
        central = sections[2]
        if central['width_mm'] is None:
            width = None
        else:
            width = central['width_mm']
        return width, sections

    def interference(self, data):
        """Exact original STL triangle crossing/contact; not a volume claim."""
        meshes = {g: TriangleBVH(self.triangles(data, g)) for g in self.local}
        out = []
        for ga, gb in itertools.combinations(meshes, 2):
            a, b = meshes[ga], meshes[gb]
            if not a._overlap(a.root, b.root, 1e-8):
                continue
            hit = a.first_intersection(b)
            if hit['triangle_a'] is None:
                continue
            ba, bb = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) for g in (ga, gb)]
            out.append({'body1': ba, 'body2': bb, 'source_triangle_a': hit['triangle_a'], 'source_triangle_b': hit['triangle_b'],
                        'triangle_a_world_m': a.triangles[hit['triangle_a']].tolist(), 'triangle_b_world_m': b.triangles[hit['triangle_b']].tolist(),
                        'inherited_exclusion': any(set(pair) == {ba, bb} for pair in p.INTERNAL_EXCLUSIONS),
                        'classification': 'UNRESOLVED_SOURCE_MESH_CONTACT_OR_INTERSECTION; not automatically volume penetration or intentional fit'})
        return out


def equality_jacobian(model, data):
    if data.efc_J.size != data.nefc*model.nv:
        raise ValueError('expected the small model dense constraint Jacobian')
    rows = np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    J = data.efc_J.reshape(data.nefc, model.nv)[rows]
    singular = np.linalg.svd(J, compute_uv=False)
    return {'rank': int(np.linalg.matrix_rank(J, tol=1e-9)), 'remaining_dofs': int(model.nv-np.linalg.matrix_rank(J, tol=1e-9)),
            'singular_values': singular.tolist()}


def proxy_contacts(model, data):
    out = []
    for c in data.contact:
        if c.dist >= -1e-7:
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (c.geom1, c.geom2)]
        out.append({'geoms': names, 'penetration_mm': -float(c.dist)*1000})
    return out


def state_static(model, data, surfaces, law, q):
    values, residual = set_static(model, data, law, q)
    width, sections = surfaces.aperture(data)
    axis_error = 0.
    for s in p.GROUPS:
        matrices = [data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'R_hand_{s}{suffix}')].reshape(3, 3) for suffix in ('3_Link', '_loop_Link')]
        axis_error = max(axis_error, math.degrees(math.acos(float(np.clip(abs(matrices[0][:,2]@matrices[1][:,2]), -1, 1)))))
    return {'q_CAD': float(q), 'q_URDF': values['right_claw_joint'], 'targets': values, 'closure_length_residual_mm': residual,
            'D_gap_mm': [v*1000 for v in p.gaps(model, data)], 'axis_deviation_deg': axis_error,
            'reference_surface_aperture_mm': width, 'surface_sections': sections, 'proxy_contacts': proxy_contacts(model, data),
            'mobility': equality_jacobian(model, data),
            'pose_provenance': 'ANALYTIC_CONTINUATION_NOT_STEP_MEASURED' if q > 1e-12 else 'SAVED_COMPUTED_CANDIDATE_RANGE' if q >= -.32 else 'OUTSIDE_SAVED_CANDIDATE_RANGE'}


def wrist_check(derived, model, data, surfaces):
    mount = derived.find('joint[@name="R_omnipicker_joint"]')
    t, R = p.origin(mount)
    wrist = derived.find('link[@name="right_wrist_roll_link"]')
    result = {'mount_xyz_m': t.tolist(), 'mount_rpy_rad': mount.find('origin').get('rpy'), 'modified': False, 'checks': []}
    for kind in ('visual', 'collision'):
        element = wrist.find(kind)
        pos, rotation = p.origin(element)
        file = Path(element.find('geometry/mesh').get('filename'))
        local = p.stl_vertices(file).reshape(-1, 3, 3)
        triangles = ((local@rotation.T+pos)-t)@R
        tree = TriangleBVH(triangles)
        pairs = []
        for g in surfaces.local:
            other = TriangleBVH(surfaces.triangles(data, g))
            if not tree._overlap(tree.root, other.root, 1e-8):
                continue
            hit = tree.first_intersection(other)
            if hit['triangle_a'] is not None:
                pairs.append({'gripper_geom': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), **hit})
        result['checks'].append({'kind': kind, 'source_mesh': file.name, 'source_sha256': p.sha(file), 'intersections_at_source_pose': pairs})
    result['status'] = 'BLOCKED_UNCLASSIFIED_BASELINE_MOUNT_INTERSECTIONS' if any(r['intersections_at_source_pose'] for r in result['checks']) else 'SOURCE_POSE_CLEAR_ONLY'
    return result


def analyze(run, derived, model, law):
    data = mujoco.MjData(model)
    surfaces = SourceSurfaces(model,derived)
    source = state_static(model, data, surfaces, law, 0.)
    wrist = wrist_check(derived, model, data, surfaces)
    rows, boundaries = [source], {}
    for direction in (-1, 1):
        last = 0.
        boundary = None
        limit = p.PHASE if direction < 0 else p.PHASE+1.
        q = direction*GRID
        while direction*q <= direction*limit+1e-12:
            try:
                row = state_static(model, data, surfaces, law, q)
                rows.append(row)
                if row['proxy_contacts']:
                    boundary = {'first_failing_q': q, 'reason': 'COLLISION_PROXY_CONTACT_NOT_CLEARED', 'contacts': row['proxy_contacts']}
                elif max(row['D_gap_mm']) > .01:
                    boundary = {'first_failing_q': q, 'reason': 'D_MAPPING_GAP_OVER_0P01MM'}
                elif row['mobility']['remaining_dofs'] != 1:
                    boundary = {'first_failing_q': q, 'reason': 'MOBILITY_OR_SINGULARITY'}
                elif row['reference_surface_aperture_mm'] is None or row['reference_surface_aperture_mm'] < 0:
                    boundary = {'first_failing_q': q, 'reason': 'REFERENCE_GRIPPING_SURFACE_NOT_VALID'}
            except ValueError as e:
                boundary = {'first_failing_q': q, 'reason': 'NO_REAL_CANDIDATE_CLOSURE', 'detail': str(e)}
            if boundary:
                break
            last = q
            q += direction*GRID
        if boundary is None:
            boundary = {'first_failing_q': None, 'reason': 'OFFICIAL_ROOT_LIMIT_REACHED'}
        boundary['last_grid_valid_q'] = last
        if boundary['reason'] == 'NO_REAL_CANDIDATE_CLOSURE':
            low, high = last, boundary['first_failing_q']
            for _ in range(35):
                mid = (low+high)/2
                try:
                    targets(law, mid)
                    low = mid
                except ValueError:
                    high = mid
            boundary['analytic_reachability_boundary_q_bracket'] = [low, high]
        boundaries['closing' if direction < 0 else 'opening'] = boundary
    exact = []
    for q in (0., boundaries['closing']['last_grid_valid_q'], boundaries['opening']['last_grid_valid_q']):
        set_static(model, data, law, q)
        exact.append({'q_CAD': q, 'pairs': surfaces.interference(data)})
    low, high = boundaries['closing']['last_grid_valid_q'], boundaries['opening']['last_grid_valid_q']
    selected = sorted((r for r in rows if low-1e-12 <= r['q_CAD'] <= high+1e-12), key=lambda r:r['q_CAD'])
    widths = [r['reference_surface_aperture_mm'] for r in selected]
    candidate = {'q_CAD_min': low, 'q_CAD_max': high, 'q_root_min': p.PHASE-high, 'q_root_max': p.PHASE-low,
                 'reference_aperture_min_mm': widths[0], 'reference_aperture_max_mm': widths[-1],
                 'surface_aperture_monotonic': all(b >= a-1e-7 for a, b in zip(widths, widths[1:])),
                 'qualification': 'EXPERIMENTAL_ALGEBRAIC_AND_PROXY_GRID_ONLY; exact-source baseline overlaps unresolved'}
    coverage = {s: {'count': sum(r['branch']==s for r in law.rows), 'range': [min(float(r['q_root_rel_rad']) for r in law.rows if r['branch']==s), max(float(r['q_root_rel_rad']) for r in law.rows if r['branch']==s)]} for s in p.GROUPS}
    result = {'saved_computed_candidate_coverage': coverage, 'STEP_measured_dynamic_pose_coverage': 'NONE; q_CAD=0 is registered source assembly, other poses are computed',
              'official_root_limits': [-1, 0], 'mapping': 'q_URDF=PHASE-q_CAD; PHASE=-0.13412598906686654; q_wide=-q_root',
              'previous_Q_OPEN_0P2': 'manually selected analytic demonstration bound, not a manufacturer stop or measured STEP pose',
              'grid_rad': GRID, 'boundaries': boundaries, 'candidate_grid_interval': candidate,
              'physical_aperture_definition': {'method': 'facing original STL triangle surfaces intersected by +Y ray', 'x_m': APERTURE_X, 'z_m': APERTURE_Z,
                                               'qualification': 'section-specific clearance, not largest graspable cylinder or commercial jaw specification'},
              'wrist': wrist, 'exact_source_triangle_checks': exact,
              'maximum_manufacturer_verified_range': 'NOT_ESTABLISHED: passive transmission, complete contact/mate classification and full-travel reference absent',
              'passive_joint_limits': 'continuous retained; no verified individual hard stops, so no invented revolute bounds',
              'rows': rows}
    p.put(run, 'travel_analysis.json', result)
    with (run/'aperture_vs_root.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['q_CAD','q_URDF','reference_surface_aperture_mm','max_D_gap_mm','axis_deviation_deg','equality_rank','max_proxy_penetration_mm','provenance'])
        writer.writeheader()
        for r in sorted(rows, key=lambda v:v['q_CAD']):
            writer.writerow({'q_CAD':r['q_CAD'],'q_URDF':r['q_URDF'],'reference_surface_aperture_mm':r['reference_surface_aperture_mm'],
                             'max_D_gap_mm':max(r['D_gap_mm']),'axis_deviation_deg':r['axis_deviation_deg'],'equality_rank':r['mobility']['rank'],
                             'max_proxy_penetration_mm':max([v['penetration_mm'] for v in r['proxy_contacts']] or [0]),'provenance':r['pose_provenance']})
    return result


def coupled_residuals(model, data):
    adr = p.addresses(model)
    return {s: float(data.qpos[adr[names[0]]]+data.qpos[adr[names[1]]]-data.qpos[adr[names[3]]]
                     -(p.PHASE if s=='narrow' else -p.PHASE)) for s,names in p.GROUPS.items()}


def root_bias_compensation(model,data):
    """Project model gravity/Coriolis bias onto the single root DOF.

    Virtual work through the equality-Jacobian nullspace. This changes only the
    root motor command; no passive target, state write, or extra actuator exists.
    It is a simulation controller, not manufacturer calibration.
    """
    mask=np.asarray(data.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    J=data.efc_J.reshape(data.nefc,model.nv)[mask]
    _,_,V=np.linalg.svd(J,full_matrices=True)
    root_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'right_claw_joint')
    dof=int(model.jnt_dofadr[root_joint])
    tangent=V[-1].copy()
    if abs(tangent[dof])<1e-8:raise ValueError('root tangent singularity')
    tangent/=tangent[dof]
    return float(tangent@data.qfrc_bias),tangent


def physics(run, model, law, analysis, motion='extended', bias_compensation=False):
    import imageio.v2 as imageio
    from PIL import Image
    if model.nu != 1 or int(model.actuator_trnid[0,0]) != mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'right_claw_joint'):
        raise ValueError('single motor invariant failed')
    interval = analysis['candidate_grid_interval']
    # Controller targets stay two grid increments inside the observed boundaries.
    # These are diagnostic safety margins, never new mechanical joint limits.
    lo, hi = interval['q_CAD_min']+2*GRID, interval['q_CAD_max']-2*GRID
    schedule = [(0., .5)]
    if motion in ('extended','accepted-window'):
        if motion=='accepted-window':hi=.2  # Last completed stable extension stage; not a new mechanical limit.
        schedule += [(q, 1.75) for q in (.05,.10,.15,.20,hi) if q <= hi+1e-10]
        schedule += [(lo, 3.5), (hi, 3.5), (0., 3.5)]
    else:
        hi=0.
        schedule += [(lo,2.),(0.,2.),(lo,2.),(0.,2.)]
    data = mujoco.MjData(model)
    set_static(model,data,law,0.)
    initial_mobility = equality_jacobian(model,data)
    if initial_mobility['remaining_dofs'] != 1 or proxy_contacts(model,data):
        raise ValueError('initial single-motor static gate failed')
    aid, adr = 0,p.addresses(model)
    renderer=mujoco.Renderer(model,height=480,width=640)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:],camera.distance,camera.azimuth,camera.elevation=[0,0,.12],.34,0,-12
    options=mujoco.MjvOption();options.geomgroup[1]=0
    writer=imageio.get_writer(str(run/'single_motor_open_close.mp4'),fps=25,codec='libx264')
    rows,phase_records,stop=[],[],None
    maxgap=maxpen=maxspeed=maxcoupling=maxlimit=maxforce=0.
    previous=0.;steps=0;max_actual=-float('inf');min_actual=float('inf')
    for phase,(goal,duration) in enumerate(schedule):
        start_actual=p.PHASE-float(data.qpos[adr['right_claw_joint']])
        for i in range(round(duration/.001)):
            blend=.5-.5*math.cos(math.pi*min(i*.001/max(duration-.5,.001),1.))
            q=previous+(goal-previous)*blend
            root_target=p.PHASE-q
            feedforward,tangent=root_bias_compensation(model,data) if bias_compensation else (0.,None)
            raw_control=root_target+feedforward/float(model.actuator_gainprm[aid,0])
            data.ctrl[aid]=float(np.clip(raw_control,*model.actuator_ctrlrange[aid]))
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data);steps+=1
            actual=p.PHASE-float(data.qpos[adr['right_claw_joint']])
            max_actual,min_actual=max(max_actual,actual),min(min_actual,actual)
            gap=max(p.gaps(model,data));pen=max([max(0.,-c.dist) for c in data.contact] or [0.])
            speed=float(np.max(np.abs(data.qvel)));coupling=max(abs(v) for v in coupled_residuals(model,data).values())
            qn,qw=float(data.qpos[adr['right_claw_joint']]),float(data.qpos[adr['R_hand_wide1_joint']])
            limit=max(0.,qn,-1-qn,-qw,qw-1)
            forces={}
            for eid in range(model.neq):
                mask=(np.asarray(data.efc_type)==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(np.asarray(data.efc_id)==eid)
                forces[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_EQUALITY,eid)]=float(np.max(np.abs(data.efc_force[mask]))) if np.any(mask) else 0.
            maxgap,maxpen,maxspeed,maxcoupling,maxlimit,maxforce=max(maxgap,gap),max(maxpen,pen),max(maxspeed,speed),max(maxcoupling,coupling),max(maxlimit,limit),max(maxforce,max(forces.values(),default=0.))
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                stop='NONFINITE'
            elif limit>1e-5:
                stop='OFFICIAL_ROOT_LIMIT_OVERSHOOT_OVER_1E_MINUS5'
            elif actual<interval['q_CAD_min']-1e-5 or actual>interval['q_CAD_max']+1e-5:
                stop='LEFT_STATIC_TESTED_INTERVAL'
            elif gap>.001 or coupling>.01 or speed>25 or pen>.002:
                stop='PHYSICS_SAFETY_GATE'
            elif any(w.number for w in data.warning):
                stop='MUJOCO_WARNING'
            if steps%10==0 or stop:
                rows.append({'step':steps,'time_s':float(data.time),'phase':phase,'command_CAD':q,'actual_CAD':actual,'root_target':root_target,
                             'root_servo_control':float(data.ctrl[aid]),'root_feedforward_Nm':feedforward,
                             'D_gap_mm':gap*1000,'penetration_mm':pen*1000,'coupling_residual_rad':coupling,'root_limit_error_rad':limit,
                             'motor_force_Nm':float(data.actuator_force[0]),'max_qvel':speed,'constraint_force_magnitudes':forces,
                             'joint_positions':{n:float(data.qpos[a])for n,a in adr.items()},'contacts':proxy_contacts(model,data)})
            if steps%40==0 or stop:
                renderer.update_scene(data,camera=camera,scene_option=options);frame=renderer.render();writer.append_data(frame)
                if steps%1000==0 or stop:
                    Image.fromarray(frame).save(run/f'physics_{steps:05d}.png')
            if stop:break
        end_actual=p.PHASE-float(data.qpos[adr['right_claw_joint']])
        phase_records.append({'phase':phase,'command_start':previous,'command_end':goal,'actual_start':start_actual,'actual_end':end_actual,'completed':stop is None})
        if stop:break
        if abs(end_actual-goal)>.02:
            phase_records[-1]['completed']=False
            stop='PHASE_TARGET_NOT_REACHED_WITHIN_0P02RAD'
            break
        previous=goal
    writer.close();renderer.close()
    p.put(run,'single_motor_trace.json',rows)
    with (run/'single_motor_trace.csv').open('x',newline='')as f:
        columns=['step','time_s','phase','command_CAD','actual_CAD','D_gap_mm','penetration_mm','coupling_residual_rad','root_limit_error_rad','motor_force_Nm','max_qvel']+list(adr)
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({**{k:r[k]for k in columns if k in r},**r['joint_positions']})
    result={'status':'SINGLE_MOTOR_DIAGNOSTIC_PHYSICS_PASS'if stop is None else'SINGLE_MOTOR_DIAGNOSTIC_PHYSICS_FAIL','stop_reason':stop,
            'steps':steps,'simulation_time_s':float(data.time),'mujoco_version':mujoco.__version__,'gravity':model.opt.gravity.tolist(),
            'active_actuator_count':model.nu,'active_joint':'right_claw_joint','passive_actuator_count':0,'passive_qpos_writes_after_start':0,
            'command_interval_CAD':[lo,hi],'actual_interval_CAD':[min_actual,max_actual],'max_D_gap_mm':maxgap*1000,
            'max_coupling_residual_rad':maxcoupling,'max_penetration_mm':maxpen*1000,'max_speed_rad_s':maxspeed,'max_root_limit_error_rad':maxlimit,
            'max_constraint_force_mixed_units':maxforce,'initial_mobility':initial_mobility,'phases':phase_records,
            'motion':motion,'root_bias_compensation':bias_compensation,
            'controller':'same root kp=.5 kv=.05 force cap +/- .5Nm; optional model-bias virtual-work compensation on root only',
            'mechanical_validation':'NOT_ESTABLISHED: unclassified source mesh mating/intersections and wrist clearance remain unresolved; physics numerical PASS is not mechanical full-travel PASS',
            'inherited_exclusions':p.INTERNAL_EXCLUSIONS,'object_contact':'NOT RUN; not authorized'}
    p.put(run,'single_motor_result.json',result)
    return result


def measure_saved_physics(run):
    """Offline geometry queries on recorded physics states, never a rollout."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    model=mujoco.MjModel.from_xml_path(str(run/'single_motor.xml'))
    derived=ET.parse(run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf').getroot()
    surfaces=SourceSurfaces(model,derived);data=mujoco.MjData(model);law=p.CandidateLaw()
    rows=json.loads((run/'single_motor_trace.json').read_text());adr=p.addresses(model)
    selected=set(np.linspace(0,len(rows)-1,101).round().astype(int))
    selected.update([min(range(len(rows)),key=lambda i:rows[i]['actual_CAD']),max(range(len(rows)),key=lambda i:rows[i]['actual_CAD'])])
    measures=[]
    for i in sorted(selected):
        row=rows[i]
        for n,v in row['joint_positions'].items():data.qpos[adr[n]]=v
        mujoco.mj_forward(model,data)
        width,sections=surfaces.aperture(data)
        expected=targets(law,row['actual_CAD'])[0]
        measures.append({'time_s':row['time_s'],'actual_CAD':row['actual_CAD'],'q_root':row['joint_positions']['right_claw_joint'],
                         'surface_aperture_mm':width,'surface_sections':sections,
                         'root_mimic_error_rad':abs(row['joint_positions']['right_claw_joint']+row['joint_positions']['R_hand_wide1_joint']),
                         'max_passive_error_rad':max(abs(expected[n]-row['joint_positions'][n])for n in p.FOLLOWERS)})
    p.put(run,'measured_surface_aperture.json',{'method':'original source STL triangle ray intersections at recorded actual physics states; offline mj_forward, zero new physics steps',
                                             'reference_x_m':APERTURE_X,'reference_z_m':APERTURE_Z,'rows':measures})
    with(run/'measured_aperture_vs_root.csv').open('x',newline='')as f:
        keys=['time_s','actual_CAD','q_root','surface_aperture_mm','root_mimic_error_rad','max_passive_error_rad']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k]for k in keys}for r in measures)
    diagnostics=[]
    for q,label in((-.09,'first_proxy_contact'),(p.PHASE,'old_root_zero_reference_not_approved_travel')):
        set_static(model,data,law,q)
        width,sections=surfaces.aperture(data)
        diagnostics.append({'q_CAD':q,'label':label,'q_root':p.PHASE-q,'surface_aperture_mm':width,
                            'proxy_contacts':proxy_contacts(model,data),'original_surface_pairs':surfaces.interference(data)})
    p.put(run,'endpoint_diagnostics.json',diagnostics)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    reader=imageio.get_reader(str(run/'single_motor_open_close.mp4'))
    writer=imageio.get_writer(str(run/'single_motor_review.mp4'),fps=25,codec='libx264')
    for i,frame in enumerate(reader):
        row=rows[min(range(len(rows)),key=lambda k:abs(rows[k]['step']-(i+1)*40))]
        image=Image.fromarray(frame);draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,640,64),fill=(12,20,28))
        draw.text((8,3),'ONE ROOT MOTOR / PASSIVE LINKAGE / EXPERIMENTAL COUPLING',font=font,fill=(255,207,112))
        draw.text((8,23),f"t={row['time_s']:.2f}s  CAD actual={row['actual_CAD']:+.4f} rad  gravity ON",font=font,fill='white')
        draw.text((8,43),f"D gap={row['D_gap_mm']:.4f} mm  passive servos=0  phase={row['phase']}",font=font,fill='white')
        writer.append_data(np.asarray(image))
        if i in(0,200,290,380):image.save(run/f'review_{i:04d}.png')
    reader.close();writer.close()
    result={'measured_aperture_min_mm':min(r['surface_aperture_mm']for r in measures if r['surface_aperture_mm']is not None),
            'measured_aperture_max_mm':max(r['surface_aperture_mm']for r in measures if r['surface_aperture_mm']is not None),
            'max_passive_error_rad':max(r['max_passive_error_rad']for r in measures),
            'max_root_mimic_error_rad':max(r['root_mimic_error_rad']for r in measures),
            'reference_section_only':True,'new_mj_steps':0}
    p.put(run,'measured_summary.json',result)
    return result


def grid_surface_check(run):
    analysis=json.loads((run/'travel_analysis.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(run/'single_motor.xml'));data=mujoco.MjData(model)
    derived=ET.parse(run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf').getroot()
    surfaces=SourceSurfaces(model,derived);law=p.CandidateLaw()
    source=next(r for r in analysis['exact_source_triangle_checks']if r['q_CAD']==0.)
    baseline={tuple(sorted((r['body1'],r['body2'])))for r in source['pairs']}
    result=[];stop=None
    # Reuse the already bounded grid. Do not extend past its first stop.
    for direction in (-1,1):
        rows=sorted([r for r in analysis['rows']if r['q_CAD']*direction>=0],key=lambda r:abs(r['q_CAD']))
        previous=None
        for row in rows:
            q=row['q_CAD'];set_static(model,data,law,q)
            pairs=surfaces.interference(data)
            new=[]
            for pair in pairs:
                key=tuple(sorted((pair['body1'],pair['body2'])))
                if key in baseline:continue
                ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)for n in key]
                adjacent=int(model.body_parentid[ids[0]])==ids[1]or int(model.body_parentid[ids[1]])==ids[0]
                new.append({**pair,'adjacent_URDF_parent_child':adjacent})
            unexpected=[r for r in new if not r['inherited_exclusion']and not r['adjacent_URDF_parent_child']]
            step=0. if previous is None else max(abs(row['targets'][n]-previous['targets'][n])for n in row['targets'])
            result.append({'q_CAD':q,'surface_pair_count':len(pairs),'new_pairs_vs_source_pose':new,'max_joint_step_rad':step})
            if unexpected:
                stop={'q_CAD':q,'reason':'NEW_NONADJACENT_ORIGINAL_SURFACE_CROSSING_OR_CONTACT','pairs':unexpected}
                break
            previous=row
        if stop:break
    out={'status':'NO_NEW_NONADJACENT_CROSSING_ON_SAMPLED_GRID'if stop is None else'STOPPED_AT_NEW_SOURCE_SURFACE_CONTACT',
         'stop':stop,'grid_rad':GRID,'rows':result,'baseline_unclassified_mating_pairs':[list(k)for k in sorted(baseline)],
         'qualification':'relative to source-pose triangle contact baseline; no assertion of zero volume penetration or vendor-approved mates; wrist already blocked at its source pose',
         'new_physics_steps':0}
    p.put(run,'incremental_source_surfaces.json',out)
    return {'status':out['status'],'samples_checked':len(result),'stop':stop}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path)
    ap.add_argument('--stage',choices=('analyze','physics','finalize','confirm','measure','review-export','surface-grid'),default='analyze')
    ap.add_argument('--motion',choices=('extended','source-window','accepted-window'),default='extended')
    ap.add_argument('--from-run',type=Path)
    ap.add_argument('--root-bias-compensation',action='store_true')
    args=ap.parse_args()
    if args.run_dir:
        location=p.pg.validate_output_path(args.run_dir,require_exists=True,expected_kind='directory',environment='wsl')
        p.pg._require_reserved_directory(location)
    else:
        if args.stage not in ('analyze','confirm','review-export'):raise ValueError('existing analysis run is required')
        location=p.pg.create_unique_output_directory(p.REPO/'reviews/issue-46',prefix='single-motor-full-travel',environment='wsl')
    run=location.native_path
    if (run/'artifact_paths.json').exists() and args.stage!='finalize':
        raise ValueError('sealed evidence is immutable; use review-export --from-run for new measurements')
    if args.stage=='review-export':
        if args.from_run is None:raise ValueError('recorded physical run is required')
        source_run=p.pg.validate_output_path(args.from_run,require_exists=True,expected_kind='directory',environment='wsl').native_path
        manifest=json.loads((source_run/'artifact_paths.json').read_text())
        p.pg.verify_artifacts(manifest['artifacts'],environment='wsl')
        copied={}
        for old in sorted(source_run.iterdir()):
            if old.name in ('artifact_paths.json','.robotsim-output-reservation.json'):continue
            if not old.is_file()or old.is_symlink():raise ValueError('unexpected evidence member')
            p.pg.write_new_bytes(run/old.name,old.read_bytes(),environment='wsl')
            if p.sha(run/old.name)!=p.sha(old):raise AssertionError('evidence copy changed bytes')
            copied[old.name]=p.sha(old)
        p.put(run,'recorded_physics_source.json',{'source_run':str(source_run),'copied_file_hashes':copied,'new_physics_steps':0})
        print(json.dumps({'run':str(run),'measured':measure_saved_physics(run)},indent=2))
    elif args.stage=='confirm':
        if args.from_run is None:raise ValueError('existing extended-run evidence required')
        source_run=p.pg.validate_output_path(args.from_run,require_exists=True,expected_kind='directory',environment='wsl').native_path
        old=json.loads((source_run/'single_motor_result.json').read_text())
        phases=[phase for phase in old['phases'] if phase['completed']]
        if old['stop_reason']!='LEFT_STATIC_TESTED_INTERVAL' or max(r['command_end']for r in phases)!=.2:
            raise ValueError('confirmation is limited to the already completed .2 stage')
        prov=json.loads((source_run/'provenance.json').read_text())
        if p.sha(source_run/'single_motor.xml')!=prov['mjcf_sha256']:raise ValueError('source model hash mismatch')
        for filename in ('single_motor.xml','X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf','official_to_derived.diff','travel_analysis.json','aperture_vs_root.csv','provenance.json'):
            p.put(run,filename,(source_run/filename).read_text())
        p.put(run,'confirmation_source.json',{'source_run':str(source_run),'source_result_sha256':p.sha(source_run/'single_motor_result.json'),'maximum_confirmed_command_q':.2,
                                             'purpose':'open-close cycle inside last completed extension stage, not another boundary search'})
        model=mujoco.MjModel.from_xml_path(str(run/'single_motor.xml'))
        result=physics(run,model,p.CandidateLaw(),json.loads((run/'travel_analysis.json').read_text()),'accepted-window',True)
        print(json.dumps({'run':str(run),'physics':result},indent=2))
    elif args.stage=='analyze':
        source=p.UPSTREAM/'X2-Ultra_omnipicker.urdf'
        align=p.REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json'
        decomp=p.REPO/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json'
        for path,expected in((source,p.URDF_HASH),(align,p.ALIGNMENT_HASH),(decomp,p.DECOMPOSITION_HASH),(p.SAMPLES,p.SAMPLES_HASH)):
            if p.sha(path)!=expected:raise ValueError('reviewed input hash mismatch')
        raw=source.read_bytes();derived,changes=p.derive_urdf(raw,json.loads(align.read_text()))
        p.put(run,'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf',p.serialize_derived(raw,derived))
        import difflib
        p.put(run,'official_to_derived.diff',''.join(difflib.unified_diff(raw.decode().splitlines(True),(run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf').read_text().splitlines(True),fromfile='official.urdf',tofile='derived.urdf')))
        xml=p.convert(derived,json.loads(decomp.read_text()),actuation='single-motor-experimental');ET.indent(xml,space='  ')
        p.put(run,'single_motor.xml',ET.tostring(xml,encoding='unicode')+'\n')
        model=mujoco.MjModel.from_xml_path(str(run/'single_motor.xml'))
        result=analyze(run,derived,model,p.CandidateLaw())
        p.put(run,'provenance.json',{'classification':'UNVERIFIED_EXPERIMENTAL','source_commit':p.PIN,'official_urdf_sha256':p.URDF_HASH,
                                    'derived_urdf_sha256':p.sha(run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf'),'mjcf_sha256':p.sha(run/'single_motor.xml'),
                                    'input_hashes':{str(f):p.sha(f)for f in(source,align,decomp,p.SAMPLES)},'changes':changes,
                                    'root_limits_unchanged':[-1,0],'passive_limits':'continuous; no invented hard stops','actuator_count':model.nu,
                                    'coupling':'q_root + q_B - q_E = PHASE per side (wide sign reversed), same unverified Candidate B relation as PR84',
                                    'model_mode':'single-motor-experimental','fixture':'isolated right gripper; wrist checked statically, not silently approved',
                                    'inherited_exclusions':p.INTERNAL_EXCLUSIONS})
        print(json.dumps({'run':str(run),'candidate_interval':result['candidate_grid_interval'],'boundaries':result['boundaries'],'wrist_status':result['wrist']['status']},indent=2))
    elif args.stage=='physics':
        prov=json.loads((run/'provenance.json').read_text())
        if p.sha(run/'single_motor.xml')!=prov['mjcf_sha256']:raise ValueError('model changed after kinematic audit')
        model=mujoco.MjModel.from_xml_path(str(run/'single_motor.xml'))
        result=physics(run,model,p.CandidateLaw(),json.loads((run/'travel_analysis.json').read_text()),args.motion,args.root_bias_compensation)
        print(json.dumps({'run':str(run),'physics':result},indent=2))
    elif args.stage=='measure':
        print(json.dumps(measure_saved_physics(run),indent=2))
    elif args.stage=='surface-grid':
        print(json.dumps(grid_surface_check(run),indent=2))
    else:
        required=('single_motor.xml','X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf','official_to_derived.diff','travel_analysis.json',
                  'single_motor_result.json','single_motor_trace.json','single_motor_review.mp4','measured_summary.json',
                  'measured_aperture_vs_root.csv','incremental_source_surfaces.json','REPORT.md')
        missing=[name for name in required if not(run/name).is_file()]
        if missing:raise ValueError('incomplete human review package: '+', '.join(missing))
        p.pg.write_artifact_manifest(run,environment='wsl')
        print(json.dumps({'run':str(run),'status':'AWAITING_HUMAN_FULL_TRAVEL_REVIEW'}))


if __name__=='__main__':main()
