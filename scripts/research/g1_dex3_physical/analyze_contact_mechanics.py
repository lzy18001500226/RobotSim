"""Offline contact kinematics and static wrench analysis for saved Dex3 runs.

This tool reconstructs recorded poses with mj_forward only. It never calls
mj_step and does not write results into the source experiment directories.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from baseline import OUT, UPSTREAM, path_governance

BOTTLE_BODY = 'green_box'
BOTTLE_GEOM = 'bottle_body'
BOTTLE_GEOMS = {'bottle_body', 'bottle_shoulder', 'bottle_neck', 'bottle_cap'}
FINGER_BODIES = ['right_hand_thumb_2_link', 'right_hand_index_1_link',
                 'right_hand_middle_1_link']
FINGER_SHORT = {'right_hand_thumb_2_link': 'thumb',
                'right_hand_index_1_link': 'index',
                'right_hand_middle_1_link': 'middle'}
WEIGHT_N = .570 * 9.81


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_run(run_dir):
    root = Path(path_governance.validate_output_path(
        run_dir, require_exists=True, expected_kind='directory').wsl_path)
    manifest = root / 'artifact_paths.json'
    path_governance.verify_artifacts([path_governance.describe_output_path(manifest)])
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (root / 'trace.jsonl').open(encoding='utf-8')]
    model = mj.MjModel.from_xml_path(str(root / 'model.xml'))
    return root, result, rows, model


def set_recorded_pose(model, data, row):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    names = row['joint_names']
    for name, value in zip(names, row['q']):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f'trace joint missing from compiled model: {name}')
        if model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE:
            raise ValueError(f'free joint unexpectedly present in trace list: {name}')
        data.qpos[model.jnt_qposadr[jid]] = value
    bottle_free = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'bottle_free')
    if bottle_free < 0 or model.jnt_type[bottle_free] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError('compiled bottle_free joint not found')
    adr = model.jnt_qposadr[bottle_free]
    data.qpos[adr:adr + 3] = row['bottle_pos']
    data.qpos[adr + 3:adr + 7] = row['bottle_quat']
    mj.mj_forward(model, data)


def reconstruct_body_poses(model, rows):
    data = mj.MjData(model)
    bodies = set(FINGER_BODIES + [BOTTLE_BODY, 'right_wrist_yaw_link'])
    ids = {name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in bodies}
    if any(value < 0 for value in ids.values()):
        raise ValueError(f'body missing from compiled model: {ids}')
    poses = []
    max_wrist_error = 0.
    max_bottle_error = 0.
    for row in rows:
        set_recorded_pose(model, data, row)
        pose = {name: (data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy())
                for name, bid in ids.items()}
        poses.append(pose)
        max_wrist_error = max(max_wrist_error, float(np.linalg.norm(
            pose['right_wrist_yaw_link'][0] - np.asarray(row['wrist_pos']))))
        max_bottle_error = max(max_bottle_error, float(np.linalg.norm(
            pose[BOTTLE_BODY][0] - np.asarray(row['bottle_pos']))))
    return poses, {'max_wrist_position_reconstruction_error_m': max_wrist_error,
                   'max_bottle_position_reconstruction_error_m': max_bottle_error,
                   'mj_step_calls': 0, 'mj_forward_calls': len(rows)}


def point_velocity(poses, body, index, point, times):
    left, right = poses[index - 1][body], poses[index + 1][body]
    current = poses[index][body]
    dt = float(times[index + 1] - times[index - 1])
    linear = (right[0] - left[0]) / dt
    rot_left = Rotation.from_matrix(left[1])
    rot_right = Rotation.from_matrix(right[1])
    omega_world = (rot_right * rot_left.inv()).as_rotvec() / dt
    return linear + np.cross(omega_world, np.asarray(point) - current[0])


def summarize_contact_pairs(selected_rows, finger):
    """Summarize contact motion within each finger/geom pair, never across pairs."""
    grouped = {}
    for row in selected_rows:
        contacts_by_pair = {}
        for contact in row['contacts']:
            if contact.get('other') != finger:
                continue
            pair = tuple(sorted(contact['geoms']))
            contacts_by_pair.setdefault(pair, []).append(contact)
        for pair, contacts in contacts_by_pair.items():
            points = np.asarray([c['point_bottle_local_m'] for c in contacts], dtype=float)
            normals = np.asarray([c['normal'] for c in contacts], dtype=float)
            mean_normal = normals.mean(axis=0)
            mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
            grouped.setdefault(pair, []).append({
                'time_s': float(row['time']),
                'centroid_bottle_local_m': points.mean(axis=0),
                'normal': mean_normal,
                'normal_force_N': float(sum(c['normal_force_N'] for c in contacts)),
                'tangential_force_N': float(sum(c['tangential_force_N'] for c in contacts)),
                'friction_utilization': [float(c['friction_pyramid_tangent_utilization'])
                                         for c in contacts],
                'force_on_bottle_N': np.sum([
                    np.asarray(c['world_force_on_bottle'], dtype=float) for c in contacts
                ], axis=0),
            })

    summaries = []
    for pair, samples in sorted(grouped.items()):
        points = np.asarray([s['centroid_bottle_local_m'] for s in samples])
        normals = np.asarray([s['normal'] for s in samples])
        angles = np.rad2deg(np.arccos(np.clip(normals @ normals[0], -1., 1.)))
        active = np.zeros(len(selected_rows), dtype=bool)
        row_times = {float(row['time']): i for i, row in enumerate(selected_rows)}
        for sample in samples:
            active[row_times[sample['time_s']]] = True
        transitions = []
        for i in range(1, len(active)):
            if active[i] != active[i - 1]:
                transitions.append({
                    'time_s': float(selected_rows[i]['time']),
                    'event': 'created' if active[i] else 'lost'})
        force_z = [float(s['force_on_bottle_N'][2]) for s in samples]
        utilizations = [u for s in samples for u in s['friction_utilization']]
        summaries.append({
            'geom_pair': list(pair),
            'sample_fraction': float(np.mean(active)) if len(active) else 0.,
            'contact_transitions': transitions,
            'max_bottle_frame_centroid_migration_mm': float(
                np.max(np.linalg.norm(points - points[0], axis=1)) * 1000),
            'max_contact_normal_change_deg': float(np.max(angles)),
            'normal_force_N': {
                'mean': float(np.mean([s['normal_force_N'] for s in samples])),
                'p95': float(np.percentile([s['normal_force_N'] for s in samples], 95))},
            'tangential_force_N': {
                'mean': float(np.mean([s['tangential_force_N'] for s in samples])),
                'p95': float(np.percentile([s['tangential_force_N'] for s in samples], 95))},
            'force_on_bottle_vertical_N': {
                'mean': float(np.mean(force_z)),
                'p05': float(np.percentile(force_z, 5)),
                'p95': float(np.percentile(force_z, 95))},
            'friction_pyramid_utilization_p95': (
                float(np.percentile(utilizations, 95)) if utilizations else None),
        })
    return summaries


def contact_mechanics(rows, poses):
    times = np.asarray([r['time'] for r in rows], dtype=float)
    records = []
    phase_values = {}
    for i in range(1, len(rows) - 1):
        row = rows[i]
        for contact in row['contacts']:
            finger = contact.get('other')
            if finger not in FINGER_BODIES:
                continue
            p = np.asarray(contact['point'], dtype=float)
            vf = point_velocity(poses, finger, i, p, times)
            vb = point_velocity(poses, BOTTLE_BODY, i, p, times)
            relative = vf - vb
            normal = np.asarray(contact['normal'], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            tangent = relative - normal * float(np.dot(relative, normal))
            force_on_bottle = np.asarray(contact['world_force_on_bottle'], dtype=float)
            force_tangent = force_on_bottle - normal * float(np.dot(force_on_bottle, normal))
            tangent_speed = float(np.linalg.norm(tangent))
            force_norm = float(np.linalg.norm(force_tangent))
            along_motion = (float(np.dot(force_tangent, tangent / tangent_speed))
                            if tangent_speed > 1e-9 else 0.)
            item = {'time_s': float(row['time']), 'stage': row['stage'],
                    'finger_body': finger, 'geom_pair': contact['geoms'],
                    'contact_point_world_m': contact['point'],
                    'contact_point_bottle_local_m': contact['point_bottle_local_m'],
                    'normal_world': normal.tolist(),
                    'finger_surface_velocity_world_m_s': vf.tolist(),
                    'bottle_surface_velocity_world_m_s': vb.tolist(),
                    'relative_tangent_velocity_world_m_s': tangent.tolist(),
                    'relative_tangent_speed_m_s': tangent_speed,
                    'normal_force_N': float(contact['normal_force_N']),
                    'tangential_force_N': float(contact['tangential_force_N']),
                    'tangential_force_world_on_bottle_N': force_tangent.tolist(),
                    'tangential_force_along_finger_relative_motion_N': along_motion,
                    'contact_pair_power_W': -float(np.dot(force_tangent, tangent)),
                    'force_velocity_alignment_cosine': (
                        float(np.dot(force_tangent, tangent) / (force_norm * tangent_speed))
                        if force_norm > 1e-9 and tangent_speed > 1e-9 else None),
                    'pyramid_utilization': float(contact['friction_pyramid_tangent_utilization'])}
            records.append(item)
            phase_values.setdefault((row['stage'], finger), []).append(item)

    stages = {}
    for stage in ['GRASP', 'GRIP_HOLD', 'LOAD_TRANSFER']:
        selected_rows = [r for r in rows if r['stage'] == stage]
        if not selected_rows:
            continue
        t0 = selected_rows[0]['time']
        t1 = selected_rows[-1]['time']
        stage_report = {'start_time_s': float(t0), 'end_time_s': float(t1), 'digits': {}}
        start_pos = np.asarray(selected_rows[0]['bottle_pos'], dtype=float)
        start_quat = Rotation.from_quat(np.asarray(selected_rows[0]['bottle_quat'])[[1, 2, 3, 0]])
        end_pos = np.asarray(selected_rows[-1]['bottle_pos'], dtype=float)
        end_quat = Rotation.from_quat(np.asarray(selected_rows[-1]['bottle_quat'])[[1, 2, 3, 0]])
        stage_report['bottle_horizontal_drift_from_stage_start_mm'] = float(
            1000 * max(np.linalg.norm(np.asarray(r['bottle_pos'])[:2] - start_pos[:2])
                       for r in selected_rows))
        stage_report['bottle_vertical_change_from_stage_start_mm'] = float(
            1000 * (end_pos[2] - start_pos[2]))
        stage_report['bottle_rotation_change_from_stage_start_deg'] = float(
            np.rad2deg((end_quat * start_quat.inv()).magnitude()))
        active_frames = 0
        for finger in FINGER_BODIES:
            by_row = []
            for r in selected_rows:
                cs = [c for c in r['contacts'] if c.get('other') == finger]
                by_row.append(cs)
            active = np.asarray([bool(c) for c in by_row])
            active_frames += int(np.sum(active))
            events = []
            for idx in range(1, len(active)):
                if active[idx] != active[idx - 1]:
                    events.append({'time_s': float(selected_rows[idx]['time']),
                                   'event': 'created' if active[idx] else 'lost'})
            all_c = [c for cs in by_row for c in cs]
            slip_rows = [v for (s, f), values in phase_values.items()
                         if s == stage and f == finger for v in values]
            pair_summaries = summarize_contact_pairs(selected_rows, finger)
            migration = [p['max_bottle_frame_centroid_migration_mm'] / 1000
                         for p in pair_summaries]
            normal_angles = [p['max_contact_normal_change_deg'] for p in pair_summaries]
            speeds = np.asarray([c['relative_tangent_speed_m_s'] for c in slip_rows], dtype=float)
            stage_report['digits'][FINGER_SHORT[finger]] = {
                'contact_sample_fraction': float(np.mean(active)) if len(active) else 0.,
                'contact_transition_events': events,
                'first_contact_time_s': next((float(r['time']) for r, on in zip(selected_rows, active) if on), None),
                'last_contact_time_s': next((float(r['time']) for r, on in reversed(list(zip(selected_rows, active))) if on), None),
                'max_bottle_frame_contact_point_migration_mm': float(np.max(migration) * 1000) if migration else 0.,
                'max_contact_normal_change_deg': float(np.max(normal_angles)) if normal_angles else 0.,
                'contact_pairs': pair_summaries,
                'relative_tangent_speed_m_s': ({'median': float(np.median(speeds)),
                                                'p95': float(np.percentile(speeds, 95)),
                                                'max': float(np.max(speeds)),
                                                'fraction_over_0_005_m_s': float(np.mean(speeds > .005))}
                                               if len(speeds) else None),
                'normal_force_N': ({'mean': float(np.mean([c['normal_force_N'] for c in all_c])),
                                    'p95': float(np.percentile([c['normal_force_N'] for c in all_c], 95))}
                                   if all_c else None),
                'tangential_force_N': ({'mean': float(np.mean([c['tangential_force_N'] for c in all_c])),
                                        'p95': float(np.percentile([c['tangential_force_N'] for c in all_c], 95)),
                                        'p95_pyramid_utilization': float(np.percentile(
                                            [c['friction_pyramid_tangent_utilization'] for c in all_c], 95))}
                                       if all_c else None)}
        stage_report['thumb_opposing_simultaneous_sample_fraction'] = float(np.mean([
            any(c.get('other') == FINGER_BODIES[0] for c in r['contacts']) and
            any(c.get('other') in FINGER_BODIES[1:] for c in r['contacts'])
            for r in selected_rows]))
        stages[stage] = stage_report
    return {'contact_samples': records, 'stages': stages}


def solve_static_wrench(model, rows, stage, target_time_s):
    candidates = [r for r in rows if r['stage'] == stage and
                  all(any(c.get('other') == f for c in r['contacts']) for f in FINGER_BODIES)]
    if not candidates:
        return {'status': 'NOT_RUN_NO_THREE_DIGIT_CONTACT_SAMPLE'}
    row = min(candidates, key=lambda r: abs(r['time'] - target_time_s))
    data = mj.MjData(model)
    set_recorded_pose(model, data, row)
    bottle_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BOTTLE_BODY)
    bottle_com = data.xipos[bottle_bid].copy()
    source = UPSTREAM / 'robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
    import xml.etree.ElementTree as ET
    urdf = ET.parse(source).getroot()
    limits = {j.get('name'): float(j.find('limit').get('effort'))
              for j in urdf.findall('joint') if j.find('limit') is not None and
              j.find('limit').get('effort') is not None}
    joint_names = row['joint_names']
    trace_names = set(joint_names)
    contacts = []
    for saved in row['contacts']:
        finger = saved.get('other')
        if finger not in FINGER_BODIES or not (BOTTLE_GEOMS & set(saved['geoms'])):
            continue
        g_saved = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in saved['geoms']]
        matches = []
        for ci in range(data.ncon):
            con = data.contact[ci]
            if set(map(int, con.geom)) != set(g_saved):
                continue
            matches.append(con)
        if not matches:
            continue
        con = min(matches, key=lambda c: float(np.linalg.norm(c.pos - np.asarray(saved['point']))))
        frame = np.asarray(con.frame).reshape(3, 3)
        sign = 1. if int(model.geom_bodyid[int(con.geom[1])]) == bottle_bid else -1.
        basis = [sign * frame[0], sign * frame[1], sign * frame[2]]
        torsion_axis = sign * frame[0]
        body_id = int(model.geom_bodyid[int(con.geom[0])])
        if body_id == bottle_bid:
            body_id = int(model.geom_bodyid[int(con.geom[1])])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jac(model, data, jacp, jacr, con.pos, body_id)
        dof_coeff = {}
        for name in joint_names:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            dof = int(model.jnt_dofadr[jid])
            dof_coeff[name] = [float(np.dot(jacp[:, dof], v)) for v in basis] + [
                float(np.dot(jacr[:, dof], torsion_axis))]
        contacts.append({'finger': finger, 'geom_pair': [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(g)) for g in con.geom],
                         'point_world_m': con.pos.tolist(),
                         'point_bottle_local_m': saved['point_bottle_local_m'],
                         'normal_on_bottle_world': basis[0].tolist(),
                         'tangent1_world': basis[1].tolist(), 'tangent2_world': basis[2].tolist(),
                         'friction': np.asarray(con.friction).tolist(), 'dim': int(con.dim),
                         'jacobian_torque_coefficients_by_joint': dof_coeff,
                         'basis': [v.tolist() for v in basis],
                         'torsion_axis': torsion_axis.tolist(), 'body_id': body_id,
                         'jacp': jacp, 'jacr': jacr, 'raw_contact': con})
    if {contact['finger'] for contact in contacts} != set(FINGER_BODIES):
        return {'status': 'UNVERIFIABLE_CONTACT_RECONSTRUCTION', 'time_s': float(row['time']),
                'matched_hand_contacts': len(contacts), 'required_fingers': FINGER_BODIES,
                'available_saved_contacts': [c.get('other') for c in row['contacts']]}

    n = len(contacts)
    nvar = 4 * n
    aeq = np.zeros((6, nvar))
    beq = np.array([0., 0., WEIGHT_N, 0., 0., 0.])
    aub, bub = [], []
    bounds = []
    for i, contact in enumerate(contacts):
        base = 4 * i
        r = np.asarray(contact['point_world_m']) - bottle_com
        axes = contact['basis']
        mu1, mu2, mu_torsion = contact['friction'][:3]
        for j, direction in enumerate(axes):
            aeq[:3, base + j] = direction
            aeq[3:, base + j] = np.cross(r, direction)
        aeq[3:, base + 3] = np.asarray(contact['torsion_axis'])
        bounds.extend([(0., None), (None, None), (None, None), (None, None)])
        for s1 in (-1., 1.):
            for s2 in (-1., 1.):
                line = np.zeros(nvar)
                line[base] = -1.
                line[base + 1] = s1 / max(mu1, 1e-12)
                line[base + 2] = s2 / max(mu2, 1e-12)
                aub.append(line); bub.append(0.)
        for s in (-1., 1.):
            line = np.zeros(nvar)
            line[base + 3] = s
            line[base] = -mu_torsion
            aub.append(line); bub.append(0.)

    effort_names = [name for name in joint_names if name in limits]
    effort_rows = {}
    for name in effort_names:
        line = np.zeros(nvar)
        for i, contact in enumerate(contacts):
            coeff = contact['jacobian_torque_coefficients_by_joint'][name]
            line[4 * i:4 * i + 4] = coeff
        effort_rows[name] = line
        for sign in (-1., 1.):
            aub.append(sign * line)
            bub.append(limits[name])

    objective = np.zeros(nvar)
    objective[::4] = 1.
    cone_only = linprog(objective, A_ub=np.asarray(aub[:6 * n]), b_ub=np.asarray(bub[:6 * n]),
                        A_eq=aeq, b_eq=beq, bounds=bounds, method='highs')
    with_effort = linprog(objective, A_ub=np.asarray(aub), b_ub=np.asarray(bub),
                          A_eq=aeq, b_eq=beq, bounds=bounds, method='highs')

    def solution_report(result):
        if not result.success:
            return {'feasible': False, 'solver_status': int(result.status),
                    'solver_message': result.message}
        x = result.x
        wrench = aeq @ x
        effort = {name: float(np.dot(line, x)) for name, line in effort_rows.items()}
        utilization = {name: abs(value) / limits[name] for name, value in effort.items()}
        return {'feasible': True, 'solver_status': int(result.status),
                'normal_force_sum_N': float(sum(x[::4])),
                'contact_forces_local_by_finger': {
                    c['finger']: {'normal_N': float(x[4 * i]), 'tangent1_N': float(x[4 * i + 1]),
                                  'tangent2_N': float(x[4 * i + 2]), 'torsional_moment_Nm': float(x[4 * i + 3])}
                    for i, c in enumerate(contacts)},
                'wrench_residual_N_Nm': (wrench - beq).tolist(),
                'required_generalized_effort_Nm': effort,
                'source_effort_limit_utilization': utilization,
                'peak_source_effort_limit_utilization': max(utilization.values(), default=0.)}

    return {'status': 'SOLVED', 'stage': stage, 'sample_time_s': float(row['time']),
            'bottle_com_position_m': bottle_com.tolist(), 'bottle_weight_N': WEIGHT_N,
            'model_cone': int(model.opt.cone), 'interpretation':
                'local static contact-wrench screen using the reconstructed saved pose; it is not dynamic load-transfer evidence',
            'contacts': [{k: v for k, v in c.items() if k not in ('jacp', 'jacr', 'raw_contact')}
                         for c in contacts],
            'contact_point_match_error_m': [float(np.linalg.norm(
                c['point_world_m'] - np.asarray(next(
                    saved['point'] for saved in row['contacts']
                    if saved.get('other') == c['finger'] and
                    set(saved['geoms']) == set(c['geom_pair'])))))
                for c in contacts],
            'friction_cone_only': solution_report(cone_only),
            'with_source_joint_effort_caps': solution_report(with_effort)}


def draw_contact_plot(mechanics, output):
    samples = mechanics['contact_samples']
    canvas = np.full((1050, 1600, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Dex3 saved-trace contact mechanics: measured relative surface slip',
                (28, 38), cv2.FONT_HERSHEY_SIMPLEX, .76, (20, 20, 20), 2, cv2.LINE_AA)
    colors = {'thumb': (30, 75, 220), 'index': (30, 150, 30), 'middle': (220, 90, 30)}
    panels = [('GRIP_HOLD', 90, 350), ('LOAD_TRANSFER', 540, 350)]
    for stage, top, height in panels:
        stage_rows = [s for s in samples if s['stage'] == stage]
        cv2.putText(canvas, f'{stage}: relative tangential surface speed (mm/s)',
                    (30, top - 18), cv2.FONT_HERSHEY_SIMPLEX, .63, (20, 20, 20), 2, cv2.LINE_AA)
        x0, x1, y0, y1 = 110, 1540, top + 20, top + height - 25
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1)
        max_speed = max([s['relative_tangent_speed_m_s'] * 1000 for s in stage_rows] + [1.])
        ymax = max(10., float(np.ceil(max_speed / 10.) * 10.))
        for tick in np.linspace(0, ymax, 5):
            y = int(y1 - tick / ymax * (y1 - y0))
            cv2.line(canvas, (x0, y), (x1, y), (225, 225, 225), 1)
            cv2.putText(canvas, f'{tick:.0f}', (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, (60, 60, 60), 1, cv2.LINE_AA)
        if not stage_rows:
            cv2.putText(canvas, 'no reconstructed contact-velocity samples', (x0 + 20, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1, cv2.LINE_AA)
            continue
        tmin = min(s['time_s'] for s in stage_rows)
        tmax = max(s['time_s'] for s in stage_rows)
        for finger in FINGER_SHORT.values():
            seq = [s for s in stage_rows if FINGER_SHORT[s['finger_body']] == finger]
            if len(seq) < 2:
                continue
            pts = []
            for s in seq:
                x = int(x0 + (s['time_s'] - tmin) / max(tmax - tmin, 1e-12) * (x1 - x0))
                speed = s['relative_tangent_speed_m_s'] * 1000
                y = int(y1 - speed / ymax * (y1 - y0))
                pts.append((x, y))
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)],
                          False, colors[finger], 2, cv2.LINE_AA)
        for i, finger in enumerate(FINGER_SHORT.values()):
            lx = x0 + 20 + i * 220
            cv2.line(canvas, (lx, y1 + 20), (lx + 28, y1 + 20), colors[finger], 3)
            cv2.putText(canvas, finger, (lx + 36, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .46, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Velocity is reconstructed from finite differences of mj_forward body poses at saved trace times; no stepping.',
                (30, 1015), cv2.FONT_HERSHEY_SIMPLEX, .43, (70, 70, 70), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f'failed to write {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root, result, rows, model = load_run(args.run_dir)
    poses, reconstruction = reconstruct_body_poses(model, rows)
    mechanics = contact_mechanics(rows, poses)
    wrench = [solve_static_wrench(model, rows, 'GRIP_HOLD', 9.2),
              solve_static_wrench(model, rows, 'LOAD_TRANSFER', 11.9)]
    out = Path(path_governance.create_unique_output_directory(
        OUT, prefix='contact-mechanics-audit').wsl_path)
    draw_contact_plot(mechanics, out / 'relative_contact_slip.png')
    report = {'schema_version': 'robotsim.dex3.contact-mechanics.v1',
              'run_id': result['run_id'], 'run_directory': path_governance.describe_output_path(root),
              'result_sha256': sha256_file(root / 'result.json'),
              'trace_sha256': sha256_file(root / 'trace.jsonl'),
              'runtime_model_sha256': result['runtime_model_sha256'],
              'mujoco_version': mj.__version__, 'physics_steps_during_analysis': 0,
              'reconstruction': reconstruction, 'contact_mechanics': mechanics,
              'static_wrench_screens': wrench,
              'plots': {'relative_contact_slip': path_governance.describe_output_path(out / 'relative_contact_slip.png')},
              'interpretation_limit': 'relative tangential speed uses central finite differences at the 125 Hz saved trace cadence; it distinguishes sustained surface slip from bottle-frame contact migration but can miss sub-sample transients'}
    for stage, stage_data in mechanics['stages'].items():
        for digit, digit_data in stage_data['digits'].items():
            samples = [sample for sample in mechanics['contact_samples']
                       if sample['stage'] == stage and
                       FINGER_SHORT[sample['finger_body']] == digit and
                       sample['relative_tangent_speed_m_s'] > 1e-6]
            if samples:
                digit_data['tangential_force_velocity_check'] = {
                    'sample_count': len(samples),
                    'fraction_force_on_bottle_along_finger_motion': float(np.mean([
                        sample['tangential_force_along_finger_relative_motion_N'] > 0
                        for sample in samples])),
                    'median_force_along_motion_N': float(np.median([
                        sample['tangential_force_along_finger_relative_motion_N']
                        for sample in samples])),
                    'median_pair_power_W': float(np.median([
                        sample['contact_pair_power_W'] for sample in samples])),
                    'note': 'positive force along finger-relative motion corresponds to dissipative contact-pair power under the saved force-on-bottle convention'}
    path_governance.write_new_text(out / 'contact_mechanics.json', json.dumps(report, indent=2) + '\n')
    report_path = out / 'REPORT.md'
    status = ', '.join(f"{x['stage'] if 'stage' in x else 'sample'}={x.get('with_source_joint_effort_caps', {}).get('feasible', False)}" for x in wrench)
    text = f'''# G1 Dex3 Offline Contact Mechanics Audit

Run: `{result['run_id']}`
MuJoCo: `{mj.__version__}`
Analysis physics steps: `0` (`mj_forward` pose reconstruction only)

## Static wrench screen

Samples: {status}. Full contact locations, compiled friction values, cone solver outputs, and source-joint effort checks are in `contact_mechanics.json`. This is a local static feasibility screen at saved contact poses, not evidence of dynamic load transfer.

## Relative contact motion and force direction

Contact surface relative velocities were reconstructed from central differences of the finger and bottle body poses at the saved trace cadence. Contact point migration in bottle coordinates is reported separately. Thus nonzero migration is not automatically labeled slip. The cadence is 125 Hz and can miss shorter transients.

For reconstructed tangential motion, the saved world force on the bottle is projected onto the contact tangent plane. A positive force component along finger-relative motion means the equal-and-opposite contact force opposes finger motion and dissipates pair energy. Per-digit direction/power summaries and per-sample vectors are recorded in the JSON. Contact reconstruction point offsets are also reported for each static wrench sample.

## Provenance

- Run result SHA-256: `{report['result_sha256']}`
- Run trace SHA-256: `{report['trace_sha256']}`
- Compiled runtime model SHA-256: `{report['runtime_model_sha256']}`
- Analysis tool SHA-256: `{sha256_file(Path(__file__))}`
- Saved-run evidence manifest was verified before analysis.
- Output directory: `{path_governance.describe_output_path(out)['WINDOWS_UNC']}`
'''
    path_governance.write_new_text(report_path, text)
    manifest = path_governance.write_artifact_manifest(out)
    print(json.dumps({'output': path_governance.describe_output_path(out),
                      'manifest': manifest['manifest'], 'wrench_status': status,
                      'reconstruction': reconstruction}, indent=2))


if __name__ == '__main__':
    main()
