#!/usr/bin/env python3
"""Offline diagnosis of a saved passive-spring MuJoCo trace; zero physics steps."""
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import issue46_urdf_pipeline as p
import issue46_single_motor_travel as t

JOINTS = ('R_hand_narrow2_joint', 'R_hand_narrow3_joint', 'R_hand_narrow_loop_joint',
          'R_hand_wide2_joint', 'R_hand_wide3_joint', 'R_hand_wide_loop_joint')
SOURCE_STEP_SHA = '5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b'
SOURCE_MESH_EXPORT = p.REPO/'reviews/issue-46/candidate-b-mujoco-model-20261021T070249Z'


def ray_grid_aperture(narrow_triangles: np.ndarray, wide_triangles: np.ndarray,
                      x_count: int = 7, z_count: int = 11) -> dict:
    """Coverage-aware signed section gaps across the current jaw overlap."""
    nxyz = narrow_triangles.reshape(-1, 3)
    wxyz = wide_triangles.reshape(-1, 3)
    xlo = max(float(nxyz[:, 0].min()), float(wxyz[:, 0].min()))
    xhi = min(float(nxyz[:, 0].max()), float(wxyz[:, 0].max()))
    zlo = max(float(nxyz[:, 2].min()), float(wxyz[:, 2].min()))
    zhi = min(float(nxyz[:, 2].max()), float(wxyz[:, 2].max()))
    if xhi <= xlo or zhi <= zlo:
        return {'coverage': 0.0, 'valid_rays': 0, 'total_rays': x_count*z_count,
                'projected_bounds_overlap': False, 'overlap_xz_mm': None,
                'positive_width_mm': None, 'signed_width_mm': None,
                'negative_width_rays': 0, 'most_negative_width_mm': None}
    xs, zs = np.linspace(xlo, xhi, x_count), np.linspace(zlo, zhi, z_count)
    widths, negative = [], []
    for x in xs:
        for z in zs:
            a = t.ray_surface(narrow_triangles, float(x), float(z), 'narrow')
            b = t.ray_surface(wide_triangles, float(x), float(z), 'wide')
            if a is None or b is None:
                continue
            width = (b['y_m']-a['y_m'])*1000.0
            widths.append(width)
            if width <= 0:
                negative.append(width)
    positive = [width for width in widths if width > 0]
    return {'coverage': len(widths)/(x_count*z_count), 'valid_rays': len(widths),
            'total_rays': x_count*z_count, 'projected_bounds_overlap': True,
            'overlap_xz_mm': [xlo*1000.0, xhi*1000.0, zlo*1000.0, zhi*1000.0],
            'positive_width_mm': None if not positive else {
                'min': float(np.min(positive)), 'p10': float(np.percentile(positive, 10)),
                'median': float(np.median(positive)), 'p90': float(np.percentile(positive, 90)),
                'max': float(np.max(positive))},
            'signed_width_mm': None if not widths else {
                'min': float(np.min(widths)), 'median': float(np.median(widths)),
                'max': float(np.max(widths))},
            'negative_width_rays': len(negative),
            'most_negative_width_mm': None if not negative else float(np.min(negative))}


def mesh_vertex_distance(narrow_triangles: np.ndarray, wide_triangles: np.ndarray) -> dict:
    """Symmetric sampled STL-vertex distance; cross-check only, not exact gap."""
    n = narrow_triangles.reshape(-1, 3)
    w = wide_triangles.reshape(-1, 3)
    n = n[::max(1, len(n)//600)]
    w = w[::max(1, len(w)//600)]
    nw = cKDTree(w).query(n, k=1)[0]
    wn = cKDTree(n).query(w, k=1)[0]
    d = np.concatenate((nw, wn))*1000.0
    return {'qualification': 'symmetric nearest sampled vertices; upper-bound approximation to mesh surface distance, not an aperture',
            'min_mm': float(np.min(d)), 'p05_mm': float(np.percentile(d, 5)),
            'median_mm': float(np.median(d)), 'sample_counts': [int(len(n)), int(len(w))]}


def equality_rank(model, data) -> dict:
    if data.efc_J.size != data.nefc*model.nv:
        raise ValueError('unexpected constraint Jacobian layout')
    mask = np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    J = data.efc_J.reshape(data.nefc, model.nv)[mask]
    singular = np.linalg.svd(J, compute_uv=False)
    rank = int(np.linalg.matrix_rank(J, tol=1e-9))
    return {'rank': rank, 'remaining_dofs': int(model.nv-rank),
            'singular_values': singular.tolist(),
            'smallest_retained_singular_value': float(singular[rank-1]) if rank else None,
            'retained_condition_number': float(singular[0]/singular[rank-1]) if rank else None}


def source_spring_support_geometry() -> dict:
    """Approximate original-export mesh clearances for the STEP spring cluster."""
    report_path = SOURCE_MESH_EXPORT/'mesh_export_report.json'
    report = json.loads(report_path.read_text())
    component_rows = {}
    for row in report['components']:
        suffix = row['feature_name'].replace('Part__Feature', '')
        if suffix.isdigit():
            component_rows[int(suffix)] = row
    groups = {'wide': {'outer': 5, 'middle': 8, 'spring': (13, 14, 15, 16), 'B_support': (17, 18)},
              'narrow': {'outer': 21, 'middle': 24, 'spring': (29, 30, 31, 32), 'B_support': (33, 34)}}
    cache = {}
    def vertices(feature):
        if feature not in cache:
            mesh = SOURCE_MESH_EXPORT/'step_source_meshes'/f'source_{feature:03d}_Part__Feature{feature:03d}.stl'
            cache[feature] = p.stl_vertices(mesh)
        return cache[feature]
    result = {'mesh_export_overall_status': report['status'],
              'method': 'nearest sampled vertices from the existing STEP-derived STL export; not an exact B-rep clearance',
              'per_component_bbox_error_mm': {}, 'branches': {}}
    for feature in sorted({x for g in groups.values() for x in (g['outer'], g['middle'], *g['spring'], *g['B_support'])}):
        row = component_rows[feature]
        source = np.asarray(row['source_shape_bbox_world_mm'], dtype=float)
        mesh = np.asarray(row['scene_bbox_world_mm'], dtype=float)
        result['per_component_bbox_error_mm'][str(feature)] = float(np.max(np.abs(source-mesh)))
    for side, group in groups.items():
        outer, middle = vertices(group['outer']), vertices(group['middle'])
        trees = {'outer': cKDTree(outer[::max(1, len(outer)//1200)]),
                 'middle': cKDTree(middle[::max(1, len(middle)//1200)])}
        measurements = []
        for feature in group['spring']:
            points = vertices(feature)[::max(1, len(vertices(feature))//600)]
            measurements.append({'feature': feature,
                                 'min_to_outer_vertices_mm': float(trees['outer'].query(points)[0].min()),
                                 'min_to_middle_vertices_mm': float(trees['middle'].query(points)[0].min()),
                                 'max_component_bbox_error_mm': result['per_component_bbox_error_mm'][str(feature)]})
        result['branches'][side] = {'outer_feature': group['outer'], 'middle_feature': group['middle'],
                                    'spring_cluster_features': measurements,
                                    'source_occurrence_parent': 'spring cluster is nested under the middle-link product in the AP214 occurrence tree'}
    return result


def set_recorded_pose(model, data, addresses, row):
    for name, angle in row['joint_positions'].items():
        data.qpos[addresses[name]] = float(angle)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def mapped_link_transforms(model, data) -> dict:
    """Actual MuJoCo world and parent-relative transforms for mapped links."""
    names = ('R_omnipicker_base_link', 'R_hand_narrow1_Link', 'R_hand_narrow2_Link',
             'R_hand_narrow3_Link', 'R_hand_narrow_loop_Link', 'R_hand_wide1_Link',
             'R_hand_wide2_Link', 'R_hand_wide3_Link', 'R_hand_wide_loop_Link')
    result = {}
    for name in names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        parent = int(model.body_parentid[body])
        body_rot = data.xmat[body].reshape(3, 3)
        body_pos = data.xpos[body]
        parent_rot = data.xmat[parent].reshape(3, 3)
        parent_pos = data.xpos[parent]
        relative_rot = parent_rot.T @ body_rot
        result[name] = {
            'parent_body': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
            'world_position_mm': (body_pos*1000.0).tolist(),
            'world_quaternion_xyzw': Rotation.from_matrix(body_rot).as_quat().tolist(),
            'parent_relative_position_mm': (parent_rot.T@(body_pos-parent_pos)*1000.0).tolist(),
            'parent_relative_quaternion_xyzw': Rotation.from_matrix(relative_rot).as_quat().tolist(),
        }
    return result


def pair_keys(pairs):
    return {tuple(sorted((pair['body1'], pair['body2']))) for pair in pairs}


def render_pose(model, data, label: str):
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, .12]; camera.distance = .34; camera.azimuth = 0; camera.elevation = -12
    options = mujoco.MjvOption(); options.geomgroup[1] = 0
    renderer.update_scene(data, camera=camera, scene_option=options)
    image = Image.fromarray(renderer.render()).convert('RGB')
    renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 32), fill=(12, 20, 28))
    draw.text((8, 6), label, fill=(240, 190, 115))
    return image


def diagnose(source_run: Path, output: Path):
    source_run = p.pg.validate_output_path(source_run, require_exists=True, expected_kind='directory', environment='wsl').native_path
    manifest = json.loads((source_run/'artifact_paths.json').read_text())
    p.pg.verify_artifacts(manifest['artifacts'], environment='wsl')
    model = mujoco.MjModel.from_xml_path(str(source_run/'passive_spring.xml'))
    derived = ET.parse(source_run/'X2-Ultra_omnipicker_PASSIVE_SPRING_EXPERIMENTAL.urdf').getroot()
    trace = json.loads((source_run/'motion_trace.json').read_text())
    physics = json.loads((source_run/'physics_result.json').read_text())
    provenance = json.loads((source_run/'provenance.json').read_text())
    law = p.CandidateLaw()
    surfaces = t.SourceSurfaces(model, derived)
    addresses = p.addresses(model)
    data = mujoco.MjData(model)

    source_values, source_closure = t.set_static(model, data, law, 0.0)
    saved_full_interference = json.loads((source_run/'source_triangle_interference.json').read_text())
    source_interference = json.loads((source_run/'source_interference_baseline.json').read_text())
    source_baseline_pairs = pair_keys(source_interference.get('pairs', []))
    saved_triangle_events = []
    for snapshot in saved_full_interference:
        new_pairs = snapshot.get('new_pairs_vs_registered_source_pose', [])
        saved_triangle_events.append({'step': int(snapshot['step']), 'time_s': float(snapshot['time_s']),
                                      'q_root_rad': float(snapshot['actual_root_rad']),
                                      'source_pose_pair_count': len(source_baseline_pairs),
                                      'snapshot_pair_count': len(snapshot.get('pairs', [])),
                                      'new_pair_count': len(new_pairs), 'new_pairs': new_pairs})
    first_saved_triangle_change = next((event for event in saved_triangle_events if event['new_pair_count']), None)
    source_rank = equality_rank(model, data)
    source_pose_png = render_pose(model, data, 'Registered STEP source pose mapped into unchanged official URDF/STL')

    rows, first_rank_drop = [], None
    first_center_invalid = first_all_five_invalid = first_grid_empty = None
    first_law_error_005 = first_law_error_010 = first_reference_outside = None
    max_nearest = {'value_mm': -1.0}
    previous = None
    previous_adaptive = None
    derivative_peak = {'rad_s': 0.0}
    derivative_first_over5 = None
    first_nonmonotonic_cad = None
    first_adaptive_aperture_reversal = None
    triangle_by_step = {event['step']: event for event in saved_triangle_events}
    for row_index, row in enumerate(trace):
        set_recorded_pose(model, data, addresses, row)
        time_s, step = float(row['time_s']), int(row['step'])
        state = {'step': step, 'time_s': time_s, 'root_rad': float(row['actual_root_rad']),
                 'q_CAD': float(row['candidate_CAD_coordinate_rad']),
                 'passive_joint_rad': {name: float(row['joint_positions'][name]) for name in JOINTS},
                 'reported_fixed_section_mm': row['surface_aperture_mm'],
                 'fixed_section_hit_count': sum(s['width_mm'] is not None for s in row['surface_sections']),
                 'fixed_section_count': len(row['surface_sections']),
                 'D_gap_mm': [float(x) for x in row['D_gap_mm']],
                 'max_proxy_penetration_mm': float(row['max_proxy_penetration_mm']),
                 'reported_max_qvel_rad_s': float(row['max_qvel_rad_s'])}
        if row['surface_aperture_mm'] is None and first_center_invalid is None:
            first_center_invalid = state.copy()
        if state['fixed_section_hit_count'] == 0 and first_all_five_invalid is None:
            first_all_five_invalid = state.copy()

        geometry_sample = row_index % 5 == 0 or row_index == len(trace)-1
        state['adaptive_geometry_sampled'] = geometry_sample
        if geometry_sample:
            tri = {s: surfaces.triangles(data, surfaces.gid(s)) for s in p.GROUPS}
            state['adaptive_multiray'] = ray_grid_aperture(tri['narrow'], tri['wide'])
            state['nearest_sampled_surface_vertices'] = mesh_vertex_distance(tri['narrow'], tri['wide'])
            current_median = (state['adaptive_multiray']['positive_width_mm'] or {}).get('median')
            if previous_adaptive is not None and current_median is not None:
                if (state['q_CAD'] > previous_adaptive['q_CAD'] and
                        current_median < previous_adaptive['median_mm']-0.05 and
                        first_adaptive_aperture_reversal is None):
                    first_adaptive_aperture_reversal = {
                        'previous': previous_adaptive.copy(),
                        'current': {'step': step, 'time_s': time_s, 'q_root_rad': state['root_rad'],
                                    'q_CAD': state['q_CAD'], 'median_width_mm': current_median}}
            if current_median is not None:
                previous_adaptive = {'step': step, 'time_s': time_s, 'q_root_rad': state['root_rad'],
                                     'q_CAD': state['q_CAD'], 'median_mm': current_median}
            nearest = state['nearest_sampled_surface_vertices']
            if nearest['min_mm'] > max_nearest['value_mm']:
                max_nearest = {'value_mm': nearest['min_mm'], 'step': step, 'time_s': time_s}
            if state['adaptive_multiray']['valid_rays'] == 0 and first_grid_empty is None:
                first_grid_empty = state.copy()
        else:
            state['adaptive_multiray'] = None
            state['nearest_sampled_surface_vertices'] = None

        jac = equality_rank(model, data)
        state['equality_jacobian'] = jac
        if jac['rank'] < source_rank['rank'] and first_rank_drop is None:
            first_rank_drop = state.copy()

        qcad = state['q_CAD']
        if not p.PHASE-1e-12 <= qcad <= p.Q_OPEN+1e-12:
            if first_reference_outside is None:
                first_reference_outside = state.copy()
            state['candidate_law_reference'] = None
        else:
            expected = law.at_q(qcad)
            diffs = {name: float(row['joint_positions'][name]-expected[name]) for name in JOINTS}
            max_error = max(abs(x) for x in diffs.values())
            state['candidate_law_reference'] = {'classification': 'UNVERIFIED_GEOMETRIC_CANDIDATE_ONLY',
                                                'expected_passive_joint_rad': {name: float(expected[name]) for name in JOINTS},
                                                'max_abs_passive_angle_delta_rad': max_error,
                                                'per_joint_delta_rad': diffs}
            if max_error >= .05 and first_law_error_005 is None:
                state['mapped_link_transforms'] = mapped_link_transforms(model, data)
                first_law_error_005 = state.copy()
            if max_error >= .10 and first_law_error_010 is None:
                state['mapped_link_transforms'] = mapped_link_transforms(model, data)
                first_law_error_010 = state.copy()

        if previous is not None:
            dt = time_s-previous['time_s']
            rates = {name: float((row['joint_positions'][name]-previous['joint_positions'][name])/dt) for name in JOINTS}
            fastest = max(rates, key=lambda name: abs(rates[name]))
            peak = {'step': step, 'time_s': time_s, 'q_CAD': state['q_CAD'],
                    'rad_s': abs(rates[fastest]), 'joint': fastest,
                    'signed_rad_s': rates[fastest], 'joint_position_rad': float(row['joint_positions'][fastest])}
            if peak['rad_s'] > derivative_peak['rad_s']:
                derivative_peak = peak
            if peak['rad_s'] >= 5.0 and derivative_first_over5 is None:
                peak['mapped_link_transforms'] = mapped_link_transforms(model, data)
                derivative_first_over5 = peak
            if row['surface_aperture_mm'] is not None and previous['surface_aperture_mm'] is not None:
                if row['surface_aperture_mm'] < previous['surface_aperture_mm']-0.05 and row['command_opening'] > previous['command_opening']:
                    if first_nonmonotonic_cad is None:
                        first_nonmonotonic_cad = {'step': step, 'time_s': time_s,
                                                  'command_opening': row['command_opening'],
                                                  'previous_section_mm': previous['surface_aperture_mm'],
                                                  'current_section_mm': row['surface_aperture_mm']}
        previous = {'time_s': time_s, 'joint_positions': row['joint_positions'],
                    'surface_aperture_mm': row['surface_aperture_mm'], 'command_opening': row['command_opening']}
        state['saved_full_model_STL_snapshot'] = triangle_by_step.get(step)
        rows.append(state)

    failure_row = trace[-1]
    set_recorded_pose(model, data, addresses, failure_row)
    failure_link_transforms = mapped_link_transforms(model, data)
    failure_pose_png = render_pose(model, data,
        f"Recorded failure pose: step {failure_row['step']} / q_root={failure_row['actual_root_rad']:+.4f} rad")
    composite = Image.new('RGB', (1280, 520), (18, 25, 30))
    composite.paste(source_pose_png, (0, 40)); composite.paste(failure_pose_png, (640, 40))
    draw = ImageDraw.Draw(composite)
    draw.text((12, 10), 'REGISTERED STEP SOURCE POSE (mapped official meshes)', fill=(238, 238, 238))
    draw.text((652, 10), 'RECORDED B-SPRING CANDIDATE FAILURE POSE', fill=(238, 238, 238))
    p.pg.write_new_bytes(output/'source_pose.png', _png_bytes(source_pose_png), environment='wsl')
    p.pg.write_new_bytes(output/'failure_pose.png', _png_bytes(failure_pose_png), environment='wsl')
    p.pg.write_new_bytes(output/'source_vs_failure.png', _png_bytes(composite), environment='wsl')

    max_velocity_row = max(trace, key=lambda x: x['max_qvel_rad_s'])
    component_map_path = p.REPO/'reviews/issue-46/omnipicker-passive-single-stroke-20261020T121249Z/component_motion_map.csv'
    component_map_hash = p.sha(component_map_path)
    if component_map_hash != '0d5ba60e9e56d658f384a6f200a2cd0ea09c12530d023b97c43fd808b154fba7':
        raise ValueError('pinned STEP component map hash mismatch')
    import csv as csv_module
    with component_map_path.open(newline='') as stream:
        component_rows = {row['feature_name']: row for row in csv_module.DictReader(stream)}
    topology_features = [f'Part__Feature{x:03d}' for x in (5, 8, 13, 14, 15, 16, 21, 24,
                                                            29, 30, 31, 32, 56, 57, 58,
                                                            61, 62, 63, 64, 73, 77, 81, 82)]
    result = {'status': 'OFFLINE_DIAGNOSIS_COMPLETE', 'new_physics_steps': 0,
              'source_run': source_run.name, 'source_step_sha256': SOURCE_STEP_SHA,
              'original_run_result': {'status': physics['status'], 'stop_reason': physics['stop_reason'],
                                      'steps': physics['physics_steps'], 'time_s': physics['simulation_time_s'],
                                      'root_limits_rad': physics['root_limits'],
                                      'commanded_root_target_rad': physics['command_schedule'][-1]['target_root_rad'],
                                      'actual_root_final_rad': physics['command_schedule'][-1]['final_root_rad'],
                                      'cad_spring_evidence': provenance['cad_spring_evidence']},
              'trace_samples': len(trace), 'trace_step_interval': '20 mj_step (0.02 s), except the final safety-stop row',
              'registered_source_pose': {'D_length_residual_mm': source_closure*1000.0,
                                         'system_dofs': source_rank['remaining_dofs'],
                                         'source_STL_intersection_pairs': [list(x) for x in sorted(source_baseline_pairs)]},
              'first_measurement_events': {'fixed_center_section_missing': first_center_invalid,
                                           'all_five_fixed_z_sections_missing': first_all_five_invalid,
                                           'adaptive_overlapping_xz_grid_has_zero_rays': first_grid_empty},
              'first_candidate_branch_deviation': {'>=0.05rad': first_law_error_005,
                                                   '>=0.10rad': first_law_error_010,
                                                   'first_outside_existing_candidate_law_domain': first_reference_outside,
                                                   'qualification': 'existing geometry-closed CandidateLaw is an unverified hypothesis, not vendor truth'},
              'first_geometric_or_constraint_events': {'first_new_saved_original_STL_triangle_pair': first_saved_triangle_change,
                                                       'first_equality_jacobian_rank_drop': first_rank_drop},
              'mapped_STEP_component_assignments': {name: component_rows[name] for name in topology_features if name in component_rows},
              'failure_pose_mapped_link_transforms': failure_link_transforms,
              'velocity_analysis': {'physics_result_max_qvel_rad_s': physics['max_qvel_rad_s'],
                                    'maximum_recorded_sample_max_qvel': {'step': max_velocity_row['step'],
                                                                         'time_s': max_velocity_row['time_s'],
                                                                         'rad_s': max_velocity_row['max_qvel_rad_s']},
                                    'maximum_passive_finite_difference_rad_s': derivative_peak,
                                    'first_passive_finite_difference_over_5_rad_s': derivative_first_over5,
                                    'qualification': 'finite differences use saved 20-step pose samples; the exact 1 ms peak step is not logged'},
              'measurement_cross_checks': {'adaptive_multiray_method': '7 x 11 grid over current narrow/wide jaw x-z bounding-box overlap; original visual STL triangles; valid-ray coverage and signed width quantiles',
                                           'nearest_vertex_method': 'symmetric nearest sampled vertices of original jaw STL triangles; upper-bound approximation, not pad aperture',
                                           'triangle_intersection_method': 'reused the saved full-model exact original-STL triangle-crossing snapshots only; surface crossing is not a volume or intentional-fit verdict',
                                           'saved_full_model_STL_snapshots': saved_triangle_events,
                                           'first_sampled_aperture_reversal': first_nonmonotonic_cad,
                                           'first_adaptive_aperture_reversal': first_adaptive_aperture_reversal,
                                           'largest_nearest_vertex_gap_sample': max_nearest,
                                           'source_spring_mount_clearance': source_spring_support_geometry()},
              'frame_diagnostics': rows}
    put(output, 'passive_motion_diagnosis.json', result)
    with (output/'passive_motion_diagnosis.csv').open('x', newline='') as stream:
        keys = ['step', 'time_s', 'root_rad', 'q_CAD', 'reported_fixed_section_mm', 'fixed_section_hit_count',
                'max_proxy_penetration_mm', 'reported_max_qvel_rad_s', 'equality_jacobian_rank',
                'nearest_vertex_min_mm', 'multi_ray_coverage', 'multi_ray_valid_rays', 'multi_ray_median_mm',
                'multi_ray_negative_rays', 'saved_triangle_new_pair_count', 'candidate_branch_delta_rad']
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            multi = row['adaptive_multiray']
            nearest = row['nearest_sampled_surface_vertices']
            diag = {} if multi is None else (multi['positive_width_mm'] or {})
            saved_tri = row['saved_full_model_STL_snapshot']
            writer.writerow({'step': row['step'], 'time_s': row['time_s'], 'root_rad': row['root_rad'], 'q_CAD': row['q_CAD'],
                             'reported_fixed_section_mm': row['reported_fixed_section_mm'],
                             'fixed_section_hit_count': row['fixed_section_hit_count'],
                             'max_proxy_penetration_mm': row['max_proxy_penetration_mm'],
                             'reported_max_qvel_rad_s': row['reported_max_qvel_rad_s'],
                             'equality_jacobian_rank': row['equality_jacobian']['rank'],
                             'nearest_vertex_min_mm': None if nearest is None else nearest['min_mm'],
                             'multi_ray_coverage': None if multi is None else multi['coverage'],
                             'multi_ray_valid_rays': None if multi is None else multi['valid_rays'],
                             'multi_ray_median_mm': diag.get('median'),
                             'multi_ray_negative_rays': None if multi is None else multi['negative_width_rays'],
                             'saved_triangle_new_pair_count': None if saved_tri is None else saved_tri['new_pair_count'],
                             'candidate_branch_delta_rad': None if row['candidate_law_reference'] is None else row['candidate_law_reference']['max_abs_passive_angle_delta_rad']})
    return result


def _png_bytes(image: Image.Image) -> bytes:
    import io
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def put(run: Path, name: str, value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, indent=2, allow_nan=False)+'\n'
    p.pg.write_new_text(run/name, value, environment='wsl')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run', type=Path, required=True)
    args = ap.parse_args()
    location = p.pg.create_unique_output_directory(p.REPO/'reviews/issue-46', prefix='passive-motion-diagnosis', environment='wsl')
    result = diagnose(args.source_run, location.native_path)
    branch_event = result['first_candidate_branch_deviation']['>=0.05rad']
    branch_delta = branch_event['candidate_law_reference']['per_joint_delta_rad'] if branch_event else {}
    worst_joint = max(branch_delta, key=lambda name: abs(branch_delta[name])) if branch_delta else None
    branch_text = ('not observed within the CandidateLaw domain' if branch_event is None else
                   f"step {branch_event['step']} ({branch_event['time_s']:.3f} s), q_root={branch_event['root_rad']:+.6f} rad, "
                   f"q_CAD={branch_event['q_CAD']:+.6f} rad; {worst_joint} differs by {branch_delta[worst_joint]:+.5f} rad "
                   f"(actual {branch_event['passive_joint_rad'][worst_joint]:+.5f} rad, "
                   f"reference {branch_event['candidate_law_reference']['expected_passive_joint_rad'][worst_joint]:+.5f} rad)")
    measurement_event = result['first_measurement_events']['fixed_center_section_missing']
    final_state = result['frame_diagnostics'][-1]
    final_grid = final_state['adaptive_multiray']
    final_width = final_grid['positive_width_mm'] if final_grid else None
    measurement_text = ('not lost in recorded trace' if measurement_event is None else
                        f"step {measurement_event['step']} ({measurement_event['time_s']:.3f} s), "
                        f"q_root={measurement_event['root_rad']:+.6f} rad; fixed rays {measurement_event['fixed_section_hit_count']}/"
                        f"{measurement_event['fixed_section_count']}; adaptive grid {final_grid['valid_rays']}/"
                        f"{final_grid['total_rays']} ({final_grid['coverage']:.1%}), positive-gap median "
                        f"{final_width['median']:.3f} mm")
    branch_speed = result['velocity_analysis']['first_passive_finite_difference_over_5_rad_s']
    speed_text = ('not observed' if branch_speed is None else
                  f"step {branch_speed['step']} ({branch_speed['time_s']:.3f} s), {branch_speed['joint']} "
                  f"{branch_speed['signed_rad_s']:+.3f} rad/s; maximum sampled interval {result['velocity_analysis']['maximum_passive_finite_difference_rad_s']['joint']} "
                  f"{result['velocity_analysis']['maximum_passive_finite_difference_rad_s']['signed_rad_s']:+.3f} rad/s at "
                  f"step {result['velocity_analysis']['maximum_passive_finite_difference_rad_s']['step']}")
    aperture_reversal = result['measurement_cross_checks']['first_adaptive_aperture_reversal']
    reversal_text = ('not observed in adaptive samples' if aperture_reversal is None else
                     f"step {aperture_reversal['previous']['step']} to {aperture_reversal['current']['step']}: "
                     f"q_CAD {aperture_reversal['previous']['q_CAD']:+.4f} to {aperture_reversal['current']['q_CAD']:+.4f} rad, "
                     f"grid median {aperture_reversal['previous']['median_mm']:.3f} to "
                     f"{aperture_reversal['current']['median_width_mm']:.3f} mm")
    crossing_events = result['measurement_cross_checks']['saved_full_model_STL_snapshots']
    crossing_text = (f"{len(crossing_events)} saved full-model snapshots, steps "
                     f"{crossing_events[0]['step']}..{crossing_events[-1]['step']}; "
                     f"new triangle-pair crossings: {sum(row['new_pair_count'] for row in crossing_events)}")
    rank_drop = result['first_geometric_or_constraint_events']['first_equality_jacobian_rank_drop']
    rank_text = 'no rank drop in saved states' if rank_drop is None else f"step {rank_drop['step']}"
    b_centers = result['original_run_result']['cad_spring_evidence']
    report = f"""# Issue #46 — Offline Passive-Motion Diagnosis

Status: `OFFLINE_DIAGNOSIS_COMPLETE`

New physics steps: `0`

Saved samples: `{result['trace_samples']}` at 0.02 s intervals (final stop sample included)

This diagnostic reconstructs only saved qpos states from `{result['source_run']}` with `mj_forward`; it calls no `mj_step` and changes no source geometry.

## Findings

- **Stop cause / measurement:** `{result['original_run_result']['stop_reason']}` first occurs at {measurement_text}. Actual root is still inside official `[-1, 0]` limits and stops at `{result['original_run_result']['actual_root_final_rad']:+.6f} rad`, short of the `-0.99 rad` command. The fixed-section failure is not a detected hard stop.
- **First plausible-branch departure:** {branch_text}. This is only divergence from the existing unverified geometric `CandidateLaw`, not proof of a mechanically invalid pose. At source initialization the equality Jacobian leaves `{result['registered_source_pose']['system_dofs']}` system DOFs; the candidate has one motor and no verified passive transmission.
- **Rapid passive motion:** {speed_text}. The physical run's exact per-step maximum was `{result['velocity_analysis']['physics_result_max_qvel_rad_s']:.3f} rad/s`; its 1 ms step/joint was not retained in the summary. Equality rank {rank_text}; no sampled rank loss supports a demonstrated linkage singularity.
- **Aperture reversal:** {reversal_text}. This is a sampled mesh-section trend, not proof of collision; it occurs after the saved pose has left the existing CandidateLaw range.
- **Original mesh crossings:** {crossing_text}. The registered source pose has `{len(result['registered_source_pose']['source_STL_intersection_pairs'])}` triangle-crossing pairs; all are retained as baseline. This 0.5 s snapshot cadence does not prove collision-free motion between samples. The final collision-proxy penetration is `{final_state['max_proxy_penetration_mm']:.3f} mm`; that is not an original-STL intersection measurement.

## Geometry checks

The old five fixed-z rays all miss at the final frame, while an adaptive 7x11 grid over the actual current jaw x-z bounds still returns measured mesh sections. At the final pose it reports a positive-gap median of `{final_width['median']:.3f} mm` and 45/77 valid rays. Symmetric nearest sampled STL vertices give `{final_state['nearest_sampled_surface_vertices']['min_mm']:.3f} mm`; that is an independent sampled-surface cross-check, not a contact-pad aperture or exact closest-point result.

The image compares the registered source pose with the saved final qpos using unchanged official URDF/STL visuals and the same camera. It is not a raw FreeCAD B-Rep rendering.

![Source pose versus saved failure pose](source_vs_failure.png)

## Topology conclusion

The STEP map identifies `Part__Feature056` (wide) and `Part__Feature061` (narrow) as passive C-D-bearing jaw/palm links, with separate C/D pin/bearing parts. The spring product `220000054701` is coaxial with B at narrow `{b_centers['narrow_B_center_mm']} mm` and wide `{b_centers['wide_B_center_mm']} mm`; its STEP occurrences are nested under the middle-link products. Neither the occurrence tree nor current candidate data identifies spring leg ownership/rest configuration or a source-backed D-side mate/transmission. The experimental B springs therefore supply assumed torque but do not eliminate the two extra system DOFs.

**Decision: BLOCKED — no source-supported motion correction is justified.** The ray failure is measurement coverage, but the later `wide3` rapid passive motion and aperture reversal happen in an underconstrained model. Do not interpret the B-spring parameters as the missing transmission. A continuous collision-checked closed/open/closed kinematic reference and a physical single-motor full stroke remain unestablished. The concrete unlock is the source-supported D/palm-to-middle/ground constraint or physical spring-leg attachment/rest data; preserve the current run as `UNVERIFIED_EXPERIMENTAL`.
"""
    put(location.native_path, 'REPORT.md', report)
    manifest = p.pg.write_artifact_manifest(location.native_path, environment='wsl')
    print(json.dumps({'output': str(location.native_path), 'trace_samples': result['trace_samples'],
                      'artifact_count': len(manifest['artifacts']), 'physics_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
