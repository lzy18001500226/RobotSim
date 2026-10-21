#!/usr/bin/env python3
"""Reproduce Issue 46 candidate-B D-anchor transforms and source-sample checks.

The core calculation uses only the compact CSV beside the report and Python's
standard library. Optional source-mesh analysis uses NumPy/SciPy and a local
copy of the exact hashed scene_data_single_stroke.json; it never steps physics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / 'docs/research/issue46_candidate_b_motion_samples.csv'

URDF_REPO = 'AgibotTech/agibot_x2_urdf'
URDF_COMMIT = '575cc6b988f976c23550e0db85aa1e5475d3652d'
URDF_SHA256 = '35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d'
STEP_SHA256 = '5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b'
FCSTD_SHA256 = 'c5a1bca530eb52649544d0eaa87b1bf8881858040002e9631352e18b0f775c1f'
SCENE_SHA256 = '07ceab9431c97da7e7b53fbce90697100deea621518db12c6acfd7114f48a92d'
MOTION_SHA256 = '893604ac0a366d6e3997e661e7e83cd708549ca66c22e1cab890425d6d7809ca'
MOTION_EXTRACT_SHA256 = '7b544d5d5ebeb18e32a75ab6b9cd31ac24e8f63d03a83fc238988b941f4684f3'
FRAME_REG_SHA256 = 'dcaed08ba724b64cfa5a85f0967bd1609c83d41b6175d682ecd4605e4ff25f3e'
JOINT_MAP_SHA256 = '3802fd6e325c1abbd24eef6073799e39d0e109736eb90448975d66f27826cbb9'
COMPONENT_MAP_SHA256 = '0d5ba60e9e56d658f384a6f200a2cd0ea09c12530d023b97c43fd808b154fba7'

CAD_TO_BASE_R = ((0.0, -1.0, 0.0),
                 (0.9816018541834217, 0.0, 0.19093925699988815),
                 (-0.19093925699988812, 0.0, 0.9816018541834215))
CAD_TO_BASE_T_MM = (0.0, -24.160159242836876, -132.5856581622842)
ROOT_PHASE = {'narrow': -0.13412598906686654, 'wide': 0.13412598906686654}
INDEPENDENT_WIDE_PHASE = 0.134124879357947

ORIGINS = {
    'narrow': {
        'root': ((0.0, -19.5, 56.5), (-2.9951, -1.5708, -0.15964)),
        'B': ((30.852, 18.551, 0.0), (0.0, 0.0, 0.0)),
        'C': ((18.118, -15.74, 0.0), (0.0, 0.0, 0.0)),
        'E': ((0.0, -21.633, 73.87), (-2.9951, -1.5708, -0.15964)),
    },
    'wide': {
        'root': ((0.0, 19.5, 56.5), (-2.9951, -1.5708, -0.15964)),
        'B': ((30.852, -18.551, 0.0), (0.0, 0.0, 0.0)),
        'C': ((18.118, 15.74, 0.0), (0.0, 0.0, 0.0)),
        'E': ((0.0, 21.633, 73.87), (-2.9951, -1.5708, -0.15964)),
    },
}
AXIS = (0.0, 0.0, -1.0)
OLD_SITES_MM = {
    'narrow': {'jaw': (0.0, -10.4, 0.0),
               'loop': (31.6000012456, -9.72182496859, -0.0281056021852)},
    'wide': {'jaw': (0.0, 10.4, 0.0),
             'loop': (31.5999987547, 9.72180634187, 0.0279793618239)},
}
MOTION_LIMIT_RAD = 0.32


def vadd(a, b): return tuple(x + y for x, y in zip(a, b))
def vsub(a, b): return tuple(x - y for x, y in zip(a, b))
def vscale(a, s): return tuple(x * s for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
def norm(a): return math.sqrt(dot(a, a))


def matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def matvec(A, v):
    return tuple(sum(A[i][j] * v[j] for j in range(3)) for i in range(3))


def transpose(A):
    return tuple(tuple(A[j][i] for j in range(3)) for i in range(3))


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = ((1.0,0.0,0.0),(0.0,cr,-sr),(0.0,sr,cr))
    Ry = ((cp,0.0,sp),(0.0,1.0,0.0),(-sp,0.0,cp))
    Rz = ((cy,-sy,0.0),(sy,cy,0.0),(0.0,0.0,1.0))
    return matmul(matmul(Rz, Ry), Rx)


def axis_rotation(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return ((c+x*x*C, x*y*C-z*s, x*z*C+y*s),
            (y*x*C+z*s, c+y*y*C, y*z*C-x*s),
            (z*x*C-y*s, z*y*C+x*s, c+z*z*C))


def pose(origin, angle=0.0):
    xyz_mm, rpy = origin
    return matmul(rpy_matrix(rpy), axis_rotation(AXIS, angle)), xyz_mm


def compose(A, B):
    R1, t1 = A
    R2, t2 = B
    return matmul(R1, R2), vadd(t1, matvec(R1, t2))


def point(T, local):
    R, t = T
    return vadd(t, matvec(R, local))


def inverse_point(T, world):
    R, t = T
    return matvec(transpose(R), vsub(world, t))


def cad_point_to_base_mm(x_mm, z_mm):
    return vadd(matvec(CAD_TO_BASE_R, (x_mm, 0.0, z_mm)), CAD_TO_BASE_T_MM)


def frames(side, q_root_rel=0.0, q_b=0.0, q_c=0.0, q_e=0.0):
    origin = ORIGINS[side]
    # Candidate CAD joint coordinates use the opposite sign to the pinned URDF axes.
    root = pose(origin['root'], ROOT_PHASE[side] - q_root_rel)
    body_b = compose(root, pose(origin['B'], -q_b))
    body_c = compose(body_b, pose(origin['C'], -q_c))
    body_e = pose(origin['E'], -q_e)
    return {'root': root, 'B': body_b, 'C': body_c, 'E': body_e}


def site_metrics(jaw, loop):
    delta = vsub(jaw, loop)
    return {'delta_jaw_minus_loop_mm': list(delta),
            'euclidean_separation_mm': norm(delta),
            'connect_max_abs_component_mm': max(abs(x) for x in delta)}


def local_phase_fit(E, loop_point, target, axis):
    v, w = vsub(loop_point, E), vsub(target, E)
    a = vscale(axis, 1.0 / norm(axis))
    # A rotation about E can only align the components in the plane normal to its axis.
    vp = vsub(v, vscale(a, dot(a, v)))
    wp = vsub(w, vscale(a, dot(a, w)))
    theta = math.atan2(dot(a, cross(vp, wp)), dot(vp, wp))
    c, s = math.cos(theta), math.sin(theta)
    rotated = vadd(vadd(vscale(v, c), vscale(cross(a, v), s)),
                   vscale(a, dot(a, v) * (1.0-c)))
    return {'angle_rad': theta, 'gap_after_rotation_mm': norm(vsub(vadd(E, rotated), target)),
            'parallel_component_mismatch_mm': abs(dot(a, v) - dot(a, w)),
            'radial_mismatch_mm': abs(norm(vp) - norm(wp))}


def source_joint_angle(v_reference, v_target, axis):
    a = vscale(axis, 1.0 / norm(axis))
    vr = vsub(v_reference, vscale(a, dot(a, v_reference)))
    vt = vsub(v_target, vscale(a, dot(a, v_target)))
    angle = math.atan2(dot(a, cross(vr, vt)), dot(vr, vt))
    c, s = math.cos(angle), math.sin(angle)
    rotated = vadd(vadd(vscale(v_reference, c), vscale(cross(a, v_reference), s)),
                   vscale(a, dot(a, v_reference) * (1.0-c)))
    return angle, norm(vsub(rotated, v_target))


def read_motion_rows():
    digest = hashlib.sha256(SAMPLES.read_bytes()).hexdigest()
    if digest != MOTION_EXTRACT_SHA256:
        raise ValueError(f'curated motion sample SHA mismatch: {digest}')
    with SAMPLES.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 66 or {r['branch'] for r in rows} != {'narrow', 'wide'}:
        raise ValueError('expected the 33 + 33 source candidate samples in the approved q window')
    return rows


def core_analysis():
    samples = read_motion_rows()
    result = {'provenance': {
        'x2_urdf': {'repository': URDF_REPO, 'commit': URDF_COMMIT, 'sha256': URDF_SHA256},
        'step_sha256': STEP_SHA256, 'source_fcstd_sha256': FCSTD_SHA256,
        'scene_data_sha256': SCENE_SHA256, 'source_candidate_motion_sha256': MOTION_SHA256,
        'frame_registration_sha256': FRAME_REG_SHA256, 'joint_correspondence_sha256': JOINT_MAP_SHA256,
        'component_motion_map_sha256': COMPONENT_MAP_SHA256},
        'units': 'all reported point coordinates and lengths are millimetres; angles are radians',
        'cad_to_urdf_base': {'R': [list(r) for r in CAD_TO_BASE_R], 't_mm': list(CAD_TO_BASE_T_MM),
                             'support_fit_residuals_mm': [0.0002211732701,0.0003619316283,0.0002211733037,0.0003619313046]},
        'root_phases': {'narrow_official_candidate_rad': ROOT_PHASE['narrow'],
                        'wide_mimic_preserving_candidate_rad': ROOT_PHASE['wide'],
                        'independent_wide_fit_rad': INDEPENDENT_WIDE_PHASE,
                        'independent_wide_vs_mimic_delta_rad': ROOT_PHASE['wide']-INDEPENDENT_WIDE_PHASE},
        'branches': {}, 'motion_samples': {}}

    for side in ('narrow', 'wide'):
        rows = [r for r in samples if r['branch'] == side]
        source = rows[0]
        cad_d = cad_point_to_base_mm(float(source['D_x_mm']), float(source['D_z_mm']))
        f0 = frames(side)
        jaw_old = point(f0['C'], OLD_SITES_MM[side]['jaw'])
        loop_old = point(f0['E'], OLD_SITES_MM[side]['loop'])
        old_sep = site_metrics(jaw_old, loop_old)
        jaw_source_fit = norm(vsub(cad_d, jaw_old))
        loop_source_fit = norm(vsub(cad_d, loop_old))
        jaw_local = inverse_point(f0['C'], cad_d)
        loop_local = inverse_point(f0['E'], cad_d)
        jaw_corrected = point(f0['C'], jaw_local)
        loop_corrected = point(f0['E'], loop_local)
        phase_to_official_jaw = local_phase_fit(f0['E'][1], loop_old,
                                                 jaw_old, matvec(f0['E'][0], AXIS))
        phase_to_source_cad = local_phase_fit(f0['E'][1], loop_old,
                                              cad_d, matvec(f0['E'][0], AXIS))
        result['branches'][side] = {
            'source_cad_D_local_mm': [float(source['D_x_mm']),0.0,float(source['D_z_mm'])],
            'source_cad_D_in_urdf_base_mm': list(cad_d),
            'source_cad_E_in_urdf_base_mm': list(cad_point_to_base_mm(float(source['E_x_mm']), float(source['E_z_mm']))),
            'official_root_to_Djaw_local_existing_mm': list(OLD_SITES_MM[side]['jaw']),
            'official_Ebody_to_Dloop_local_existing_mm': list(OLD_SITES_MM[side]['loop']),
            'official_Djaw_world_at_source_mm': list(jaw_old), 'official_Dloop_world_at_source_mm': list(loop_old),
            'old_D_separation': old_sep,
            'mujoco_preflight_recorded_connect_max_abs_component_mm': 6.616061899382433 if side=='narrow' else 6.616062570877254,
            'cad_to_existing_official_Djaw_fit_residual_mm': jaw_source_fit,
            'cad_to_existing_official_Dloop_fit_residual_mm': loop_source_fit,
            'source_mapped_Djaw_local_mm': list(jaw_local),
            'source_mapped_Dloop_local_mm': list(loop_local),
            'source_mapped_site_roundtrip_gap_mm': norm(vsub(jaw_corrected, loop_corrected)),
            'proposed_local_site_offset_from_approved_mm': {
                'jaw': list(vsub(jaw_local, OLD_SITES_MM[side]['jaw'])),
                'loop': list(vsub(loop_local, OLD_SITES_MM[side]['loop']))},
            'phase_fit_to_existing_official_jaw_target': phase_to_official_jaw,
            'phase_fit_to_transformed_source_cad_D': phase_to_source_cad,
        }

        max_rate = 0.0
        max_centerline = 0.0
        original_gaps = []
        corrected_gaps = []
        cad_target_residuals = []
        pivot_state = []
        limit_ok = True
        for row in rows:
            qr = float(row['q_root_rel_rad'])
            qb, qc, qe = (float(row[k]) for k in ('q_B_rel_rad','q_C_rel_rad','q_E_rel_rad'))
            max_rate = max(max_rate, abs(qr + qb - qe - float(row['gear_relation_residual_rad'])))
            max_centerline = max(max_centerline, float(row['centerline_residual_mm']))
            qabs = ROOT_PHASE[side] - qr
            limit_ok &= (-1.0-1e-9 <= qabs <= 0.0+1e-9) if side=='narrow' else (0.0-1e-9 <= qabs <= 1.0+1e-9)
            current = frames(side, qr, qb, qc, qe)
            cad_now = cad_point_to_base_mm(float(row['D_x_mm']), float(row['D_z_mm']))
            pjo = point(current['C'], OLD_SITES_MM[side]['jaw'])
            plo = point(current['E'], OLD_SITES_MM[side]['loop'])
            pjn = point(current['C'], jaw_local)
            pln = point(current['E'], loop_local)
            original_gaps.append(norm(vsub(pjo, plo)))
            corrected_gaps.append(norm(vsub(pjn, pln)))
            cad_target_residuals.append(max(norm(vsub(pjn,cad_now)), norm(vsub(pln,cad_now))))
            # Invert the source pivot chain at this sample without stepping a simulator.
            cad_pivots = {name: cad_point_to_base_mm(float(row[f'{name}_x_mm']), float(row[f'{name}_z_mm']))
                          for name in ('A','B','C','D','E')}
            root_plus = pose(ORIGINS[side]['root'], ROOT_PHASE[side] + qr)
            pB_plus = point(root_plus, ORIGINS[side]['B'][0])
            root_minus = pose(ORIGINS[side]['root'], ROOT_PHASE[side] - qr)
            pB_minus = point(root_minus, ORIGINS[side]['B'][0])
            root = pose(ORIGINS[side]['root'], ROOT_PHASE[side] - qr)
            Bframe0 = compose(root, pose(ORIGINS[side]['B']))
            Baxis = matvec(Bframe0[0], AXIS)
            C_origin_vec = ORIGINS[side]['C'][0]
            pB_model = Bframe0[1]
            qB_geom, qB_fit = source_joint_angle(matvec(Bframe0[0], C_origin_vec),
                                                  vsub(cad_pivots['C'], pB_model), Baxis)
            Bbody = (matmul(Bframe0[0], axis_rotation(AXIS, qB_geom)), pB_model)
            Cframe0 = compose(Bbody, pose(ORIGINS[side]['C']))
            Caxis = matvec(Cframe0[0], AXIS)
            qC_geom, qC_fit = source_joint_angle(matvec(Cframe0[0], jaw_local),
                                                  vsub(cad_pivots['D'], Cframe0[1]), Caxis)
            Eframe0 = pose(ORIGINS[side]['E'])
            Eaxis = matvec(Eframe0[0], AXIS)
            qE_geom, qE_fit = source_joint_angle(matvec(Eframe0[0], loop_local),
                                                  vsub(cad_pivots['D'], Eframe0[1]), Eaxis)
            pD_jaw_geom = point((matmul(Cframe0[0], axis_rotation(AXIS, qC_geom)), Cframe0[1]), jaw_local)
            pD_loop_geom = point((matmul(Eframe0[0], axis_rotation(AXIS, qE_geom)), Eframe0[1]), loop_local)
            pivot_state.append({'sample': int(row['sample']), 'q_root_rel_rad': qr,
                                'official_root_q_if_direct_plus_rad': ROOT_PHASE[side] + qr,
                                'official_root_q_from_CAD_A_B_vector_rad': ROOT_PHASE[side] - qr,
                                'root_B_point_error_direct_plus_mm': norm(vsub(pB_plus,cad_pivots['B'])),
                                'root_B_point_error_sign_converted_mm': norm(vsub(pB_minus,cad_pivots['B'])),
                                'q_B_candidate_CAD_rad': qb, 'q_C_candidate_CAD_rad': qc,
                                'q_E_candidate_CAD_rad': qe,
                                'q_B_from_pivots_rad': qB_geom, 'q_C_from_pivots_rad': qC_geom,
                                'q_E_from_pivots_rad': qE_geom,
                                'q_urdf_relative_plus_q_B_urdf_minus_q_E_urdf_rad': -qr+qB_geom-qE_geom,
                                'qB_sign_mapping_residual_rad': qB_geom+qb,
                                'qC_sign_mapping_residual_rad': qC_geom+qc,
                                'qE_sign_mapping_residual_rad': qE_geom+qe,
                                'B_vector_fit_residual_mm': qB_fit, 'C_D_vector_fit_residual_mm': qC_fit,
                                'E_D_vector_fit_residual_mm': qE_fit,
                                'source_C_vs_official_BC_frame_residual_mm': norm(vsub(Cframe0[1], cad_pivots['C'])),
                                'source_Djaw_point_residual_mm': norm(vsub(pD_jaw_geom,cad_pivots['D'])),
                                'source_Dloop_point_residual_mm': norm(vsub(pD_loop_geom,cad_pivots['D']))})
        result['motion_samples'][side] = {
            'count': len(rows), 'root_relative_range_rad': [min(float(r['q_root_rel_rad']) for r in rows),max(float(r['q_root_rel_rad']) for r in rows)],
            'official_absolute_root_range_rad_after_sign_conversion': [min(ROOT_PHASE[side]-float(r['q_root_rel_rad']) for r in rows),max(ROOT_PHASE[side]-float(r['q_root_rel_rad']) for r in rows)],
            'max_root_B_point_error_if_direct_plus_mm': max(p['root_B_point_error_direct_plus_mm'] for p in pivot_state),
            'max_root_B_point_error_if_sign_converted_mm': max(p['root_B_point_error_sign_converted_mm'] for p in pivot_state),
            'official_root_limits_pass': limit_ok,
            'max_candidate_CAD_rate_residual_rad': max_rate,
            'pivot_inverted_URDF_rate_residual_max_abs_rad': max(abs(p['q_urdf_relative_plus_q_B_urdf_minus_q_E_urdf_rad']) for p in pivot_state),
            'max_passive_joint_sign_map_residual_rad': {
                key: max(abs(p[key]) for p in pivot_state)
                for key in ('qB_sign_mapping_residual_rad','qC_sign_mapping_residual_rad','qE_sign_mapping_residual_rad')},
            'max_source_candidate_centerline_residual_mm': max_centerline,
            'pivot_inversion_max_fit_residual_mm': max(max(p['B_vector_fit_residual_mm'],p['C_D_vector_fit_residual_mm'],p['E_D_vector_fit_residual_mm']) for p in pivot_state),
            'source_point_forward_residual_max_mm': max(max(p['source_C_vs_official_BC_frame_residual_mm'],p['source_Djaw_point_residual_mm'],p['source_Dloop_point_residual_mm']) for p in pivot_state),
            'pivot_inversion_samples': pivot_state,
            'old_official_D_gap_range_mm': [min(original_gaps), max(original_gaps)],
            'source_mapped_site_D_gap_range_mm': [min(corrected_gaps), max(corrected_gaps)],
            'source_mapped_sites_to_candidate_CAD_D_max_mm': max(cad_target_residuals),
            'mechanism_claim': 'source-candidate kinematic samples only; no physics integration or mechanical validation'}

    narrow_motion = result['motion_samples']['narrow']
    wide_motion = result['motion_samples']['wide']
    result['coordinate_sign_audit'] = {
        'source_to_urdf_joint_coordinate_map': 'q_urdf_abs = q_source_phase - q_candidate_CAD_rel',
        'joint_axis_change': 'none; official joint axes remain (0,0,-1)',
        'passive_coordinate_map': 'q_urdf_B/C/E = -q_candidate_CAD_B/C/E for the measured source frames',
        'approved_transmission_expression_after_sign_map': 'q_urdf_root_rel + q_urdf_B - q_urdf_E = 0 (same zero equation after multiplying the CAD relation by -1)',
        'root_B_point_error_max_if_old_plus_map_mm': max(narrow_motion['max_root_B_point_error_if_direct_plus_mm'],
                                                          wide_motion['max_root_B_point_error_if_direct_plus_mm']),
        'root_B_point_error_max_if_source_sign_map_mm': max(narrow_motion['max_root_B_point_error_if_sign_converted_mm'],
                                                             wide_motion['max_root_B_point_error_if_sign_converted_mm']),
        'official_root_limits_pass_full_candidate_window': narrow_motion['official_root_limits_pass'] and wide_motion['official_root_limits_pass'],
        'candidate_narrow_CAD_range_allowed_by_official_root_limit_rad': [-abs(ROOT_PHASE['narrow']), 0.0],
        'candidate_wide_CAD_range_allowed_by_official_root_limit_rad': [0.0, abs(ROOT_PHASE['wide'])],
    }
    for side in ('narrow','wide'):
        branch = result['branches'][side]
        if abs(branch['old_D_separation']['euclidean_separation_mm'] -
               math.sqrt(sum(x*x for x in branch['old_D_separation']['delta_jaw_minus_loop_mm']))) > 1e-9:
            raise AssertionError(f'Euclidean D residual is not reproducible for {side}')
        if branch['source_mapped_site_roundtrip_gap_mm'] > 1e-9:
            raise AssertionError(f'source-mapped D sites do not coincide at source pose for {side}')
    if result['coordinate_sign_audit']['root_B_point_error_max_if_source_sign_map_mm'] >= result['coordinate_sign_audit']['root_B_point_error_max_if_old_plus_map_mm']:
        raise AssertionError('source root-coordinate sign mapping did not improve the measured A-B transform')

    return result


def scene_mesh_analysis(scene_path: Path):
    digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if digest != SCENE_SHA256:
        raise ValueError(f'scene data SHA mismatch: {digest}')
    import numpy as np
    from scipy.spatial import cKDTree
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    if scene.get('coordinate_mapping') != 'FreeCAD world (x,y,z) mm to Three.js (x,z,-y) mm; this is a rigid coordinate rotation':
        raise ValueError('unexpected source-scene coordinate mapping')

    def points(group):
        target_parts = {'jaw_narrow': {'Part__Feature061','Part__Feature062'},
                        'jaw_wide': {'Part__Feature054','Part__Feature056'}}[group]
        result = []
        owner = []
        for part in scene['parts']:
            if part['name'] not in target_parts:
                continue
            three = np.asarray(part['positions_xyz_mm_three'],dtype=float).reshape(-1,3)
            # Invert FreeCAD (x,y,z) -> Three.js (x,z,-y).
            cad = np.column_stack((three[:,0],-three[:,2],three[:,1]))
            result.append(cad)
            owner.extend([part['name']]*len(cad))
        return np.vstack(result), np.asarray(owner,dtype=object)

    def move_rigid(vertices, source, target):
        C0=np.array([float(source['C_x_mm']),float(source['C_z_mm'])])
        D0=np.array([float(source['D_x_mm']),float(source['D_z_mm'])])
        C1=np.array([float(target['C_x_mm']),float(target['C_z_mm'])])
        D1=np.array([float(target['D_x_mm']),float(target['D_z_mm'])])
        a,b=D0-C0,D1-C1
        angle=math.atan2(float(a[0]*b[1]-a[1]*b[0]),float(a@b))
        out=vertices.copy(); dx=vertices[:,0]-C0[0]; dz=vertices[:,2]-C0[1]
        out[:,0]=C1[0]+math.cos(angle)*dx-math.sin(angle)*dz
        out[:,2]=C1[1]+math.sin(angle)*dx+math.cos(angle)*dz
        return out

    clouds={s:points('jaw_'+s) for s in ('narrow','wide')}
    rows={s:scene['candidate_motion'][s] for s in ('narrow','wide')}
    output=[]
    for i in range(33):
        nr,wr=rows['narrow'][i],rows['wide'][i]
        nverts,nowner=clouds['narrow']; wverts,wowner=clouds['wide']
        n=move_rigid(nverts,rows['narrow'][0],nr)
        w=move_rigid(wverts,rows['wide'][0],wr)
        tree=cKDTree(w); dist,idx=tree.query(n,k=1,workers=1); k=int(np.argmin(dist)); j=int(idx[k])
        output.append({'q_narrow_rel_rad':float(nr['q_narrow_rad']),
                       'nearest_vertex_distance_mm':float(dist[k]),
                       'narrow_part':str(nowner[k]),'wide_part':str(wowner[j]),
                       'narrow_vertex_world_mm':n[k].tolist(),'wide_vertex_world_mm':w[j].tolist()})
    interference=scene['single_stroke_interference']
    frame1=interference['frame_results']['1']
    return {'scene_sha256':digest,'mesh_note':'0.5 mm display triangulation; nearest vertices across main narrow/wide jaw bodies, not exact face distance or a tolerance measurement',
            'jaw_surface_samples':output,
            'first_single_stroke_exact_BRep_interference':frame1['positive_pairs'],
            'collision_scope_note':interference['limitations']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-json',type=Path,help='optional local scene_data_single_stroke.json with the pinned hash')
    args=parser.parse_args()
    result=core_analysis()
    if args.scene_json:
        result['source_mesh_direction_and_interference']=scene_mesh_analysis(args.scene_json)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
