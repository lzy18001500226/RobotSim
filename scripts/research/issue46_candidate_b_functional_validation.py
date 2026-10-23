#!/usr/bin/env python3
"""Reproduce the frozen Issue 46 Candidate B gripper and test static clearance."""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import issue46_urdf_pipeline as p
import issue46_single_motor_travel as travel
import issue46_passive_motion_diagnosis as diagnosis
from agent import path_governance as pg

SOURCE_RUN = p.REPO / 'reviews/issue-46/single-motor-full-travel-20261022T084103Z'
SOURCE_FILES = ('single_motor.xml', 'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf',
                'travel_analysis.json', 'provenance.json', 'source_integrity.json')
HOLD_SECONDS = .5
ROOT_TOLERANCE_RAD = .02
MODEL_MAX_PENETRATION_MM = 2.0
MODEL_MAX_D_GAP_MM = 1.0
MODEL_MAX_COUPLING_RESIDUAL_RAD = .01
GRIP_BAND_HEIGHT_MM = 10.0
GRIP_BAND_Z_MM = (110.0, 120.0)


def copy_frozen_candidate(output: Path) -> dict:
    source = pg.validate_output_path(SOURCE_RUN, require_exists=True,
                                     expected_kind='directory', environment='wsl').native_path
    manifest = json.loads((source / 'artifact_paths.json').read_text())
    pg.verify_artifacts(manifest['artifacts'], environment='wsl')
    provenance = json.loads((source / 'provenance.json').read_text())
    integrity = json.loads((source / 'source_integrity.json').read_text())
    if integrity.get('status') != 'UNCHANGED':
        raise ValueError('pinned official URDF/STL source integrity did not pass')
    official_urdf = p.UPSTREAM/'X2-Ultra_omnipicker.urdf'
    if p.sha(official_urdf) != integrity['URDF_sha256']:
        raise ValueError('pinned official URDF hash differs from source-integrity record')
    for filename, expected in integrity['STL_sha256'].items():
        if p.sha(filename) != expected:
            raise ValueError(f'official visual STL changed: {Path(filename).name}')
    for filename, expected in (
        ('single_motor.xml', provenance['mjcf_sha256']),
        ('X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf', provenance['derived_urdf_sha256']),
    ):
        if p.sha(source / filename) != expected:
            raise ValueError(f'frozen Candidate B {filename} hash mismatch')
    sample_entries = [(name, digest) for name, digest in provenance['input_hashes'].items()
                      if name.endswith('/docs/research/issue46_candidate_b_motion_samples.csv')]
    if len(sample_entries) != 1 or p.sha(p.SAMPLES) != sample_entries[0][1]:
        raise ValueError('Candidate B CAD motion sample input hash mismatch')
    for filename in SOURCE_FILES:
        pg.write_new_bytes(output / filename, (source / filename).read_bytes(), environment='wsl')
    model = mujoco.MjModel.from_xml_path(str(output / 'single_motor.xml'))
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_claw_joint')
    if model.nu != 1 or int(model.actuator_trnid[0, 0]) != root_id:
        raise ValueError('frozen Candidate B is not the reviewed one-root-motor model')
    if not np.allclose(model.jnt_range[root_id], [-1.0, 0.0], rtol=0.0, atol=1e-12):
        raise ValueError('official root limits changed in the frozen Candidate B model')
    if 'UNVERIFIED_EXPERIMENTAL' not in provenance.get('classification', ''):
        raise ValueError('candidate coupling classification was not preserved')
    if p.sha(output / 'single_motor.xml') != provenance['mjcf_sha256']:
        raise ValueError('copied MJCF changed during validation setup')
    equality_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)]
    return {'source_run': source.name, 'source_manifest_verified_files': len(manifest['artifacts']),
            'mujoco_version': mujoco.__version__, 'mjcf_sha256': p.sha(output / 'single_motor.xml'),
            'derived_urdf_sha256': p.sha(output / 'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf'),
            'official_commit': provenance['source_commit'], 'official_urdf_sha256': provenance['official_urdf_sha256'],
            'source_integrity': integrity['status'], 'actuators': model.nu, 'nv': model.nv, 'nq': model.nq,
            'neq': model.neq, 'equality_names': equality_names, 'coupling': provenance['coupling'],
            'inherited_exclusions': provenance['inherited_exclusions']}


def _hold_metrics(rows: list[dict], phase: int, target: float, joint_names: list[str]) -> dict:
    phase_rows = [row for row in rows if row['phase'] == phase]
    if not phase_rows:
        raise ValueError(f'missing recorded hold phase {phase}')
    end_time = float(phase_rows[-1]['time_s'])
    selected = [row for row in phase_rows
                if row['time_s'] >= end_time-HOLD_SECONDS-1e-9
                and abs(float(row['command_CAD'])-target) <= 1e-6]
    if len(selected) < 20:
        raise ValueError(f'phase {phase} has insufficient recorded endpoint hold samples')
    first, last = selected[0], selected[-1]
    joint_drift = {name: float(last['joint_positions'][name]-first['joint_positions'][name]) for name in joint_names}
    actual = [float(row['actual_CAD']) for row in selected]
    return {'phase': phase, 'target_CAD': target, 'duration_s': float(last['time_s']-first['time_s']),
            'samples': len(selected), 'actual_CAD_min': min(actual), 'actual_CAD_max': max(actual),
            'actual_CAD_drift_rad': actual[-1]-actual[0],
            'max_sampled_qvel_rad_s': max(float(row['max_qvel']) for row in selected),
            'max_D_gap_mm': max(float(row['D_gap_mm']) for row in selected),
            'max_coupling_residual_rad': max(float(row['coupling_residual_rad']) for row in selected),
            'max_proxy_penetration_mm': max(float(row['penetration_mm']) for row in selected),
            'root_limit_error_rad': max(float(row['root_limit_error_rad']) for row in selected),
            'penetrating_contact_samples': sum(bool(row['contacts']) for row in selected),
            'joint_drift_rad': joint_drift,
            'stability_status': 'PASS' if max(actual)-min(actual) <= ROOT_TOLERANCE_RAD else 'FAIL'}


def _set_saved_pose(model, data, row):
    addresses = p.addresses(model)
    for name, value in row['joint_positions'].items():
        data.qpos[addresses[name]] = float(value)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _surface_metrics(model, derived, data) -> tuple[dict, dict[str, np.ndarray]]:
    surfaces = travel.SourceSurfaces(model, derived)
    triangles = {side: surfaces.triangles(data, surfaces.gid(side)) for side in p.GROUPS}
    grid = diagnosis.ray_grid_aperture(triangles['narrow'], triangles['wide'], x_count=11, z_count=21)
    vertices = diagnosis.mesh_vertex_distance(triangles['narrow'], triangles['wide'])
    return {'multi_ray_grid': grid, 'nearest_sampled_vertices': vertices,
            'qualification': 'original visual STL surfaces; section widths sampled across current x-z overlap, not a manufacturer pad specification'}, triangles


def cylinder_band_fit(narrow_triangles: np.ndarray, wide_triangles: np.ndarray, diameter_mm: float,
                      z_center_m: float, *, cx_m: float = 0.0, height_mm: float = GRIP_BAND_HEIGHT_MM,
                      x_samples: int = 25, z_samples: int = 7) -> dict:
    """Find a feasible y-center interval for a vertical cylindrical slice."""
    radius = diameter_mm / 2000.0
    z_values = np.linspace(z_center_m-height_mm/2000.0, z_center_m+height_mm/2000.0, z_samples)
    xlo = max(float(narrow_triangles[:, :, 0].min()), float(wide_triangles[:, :, 0].min()), cx_m-radius)
    xhi = min(float(narrow_triangles[:, :, 0].max()), float(wide_triangles[:, :, 0].max()), cx_m+radius)
    if xhi <= xlo:
        return {'diameter_mm': diameter_mm, 'z_center_mm': z_center_m*1000, 'surface_samples': 0,
                'feasible_center_y_interval_mm': None, 'clearance_mm': None, 'status': 'INDETERMINATE_NO_FACE_OVERLAP'}
    lower, upper, narrow_hits, wide_hits = -math.inf, math.inf, 0, 0
    xs = np.linspace(xlo, xhi, x_samples)
    for x in xs:
        chord = math.sqrt(max(0.0, radius*radius-(x-cx_m)**2))
        for z in z_values:
            narrow = travel.ray_surface(narrow_triangles, float(x), float(z), 'narrow')
            wide = travel.ray_surface(wide_triangles, float(x), float(z), 'wide')
            if narrow is not None:
                lower = max(lower, narrow['y_m']+chord)
                narrow_hits += 1
            if wide is not None:
                upper = min(upper, wide['y_m']-chord)
                wide_hits += 1
    if not narrow_hits or not wide_hits:
        return {'diameter_mm': diameter_mm, 'z_center_mm': z_center_m*1000, 'surface_samples': min(narrow_hits, wide_hits),
                'narrow_hits': narrow_hits, 'wide_hits': wide_hits, 'feasible_center_y_interval_mm': None,
                'clearance_mm': None, 'status': 'INDETERMINATE_INCOMPLETE_FACE_COVERAGE'}
    clearance = (upper-lower)*1000.0
    return {'diameter_mm': diameter_mm, 'z_center_mm': z_center_m*1000, 'height_mm': height_mm,
            'center_x_mm': cx_m*1000, 'narrow_hits': narrow_hits, 'wide_hits': wide_hits,
            'feasible_center_y_interval_mm': [lower*1000.0, upper*1000.0],
            'best_center_y_mm': (lower+upper)*500.0, 'clearance_mm': clearance,
            'status': 'PASS' if clearance >= 0 else 'FAIL'}


def _static_collision_probe(output: Path, source_xml: Path, model, row: dict,
                            diameter_mm: float, z_center_mm: float, x_center_mm: float,
                            center_y_mm: float) -> dict:
    tree = ET.parse(source_xml)
    worldbody = tree.getroot().find('worldbody')
    if worldbody is None:
        raise ValueError('MJCF lacks worldbody for static clearance probe')
    name = f'bottle_probe_{int(diameter_mm)}mm'
    ET.SubElement(worldbody, 'geom', {
        'name': name, 'type': 'cylinder',
        'size': f'{diameter_mm/2000.0:.9g} {GRIP_BAND_HEIGHT_MM/2000.0:.9g}',
        'pos': f'{x_center_mm/1000.0:.9g} {center_y_mm/1000.0:.9g} {z_center_mm/1000.0:.9g}',
        'contype': '1', 'conaffinity': '1', 'density': '0', 'condim': '4',
    })
    probe_xml = output / f'{name}.xml'
    pg.write_new_text(probe_xml, ET.tostring(tree.getroot(), encoding='unicode')+'\n', environment='wsl')
    probe_model = mujoco.MjModel.from_xml_path(str(probe_xml))
    if (probe_model.nq, probe_model.nv, probe_model.nu, probe_model.neq) != (model.nq, model.nv, model.nu, model.neq):
        raise ValueError('static bottle probe altered gripper generalized coordinates or constraints')
    data = mujoco.MjData(probe_model)
    _set_saved_pose(probe_model, data, row)
    probe_id = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    contacts = []
    for contact in data.contact:
        if probe_id not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2) if int(contact.geom1) == probe_id else int(contact.geom1)
        contacts.append({'other_geom': mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_GEOM, other),
                         'distance_mm': float(contact.dist)*1000.0})
    return {'diameter_mm': diameter_mm, 'height_mm': GRIP_BAND_HEIGHT_MM,
            'pose_mm': [x_center_mm, center_y_mm, z_center_mm], 'physics_steps': 0,
            'contact_count': len(contacts), 'penetrating_contact_count': sum(item['distance_mm'] < -1e-4 for item in contacts),
            'min_contact_distance_mm': None if not contacts else min(item['distance_mm'] for item in contacts),
            'contacts': contacts, 'classification': 'STATIC_COLLISION_PROXY_PROBE; no bottle dynamics or model promotion'}


def run_validation() -> dict:
    source = pg.validate_output_path(SOURCE_RUN, require_exists=True, expected_kind='directory', environment='wsl').native_path
    provenance = json.loads((source/'provenance.json').read_text())
    analysis = json.loads((source/'travel_analysis.json').read_text())
    source_result = json.loads((source/'single_motor_result.json').read_text())
    prior_measured = json.loads((source/'measured_summary.json').read_text())
    source_integrity = json.loads((source/'source_integrity.json').read_text())
    prior_surface = json.loads((source/'incremental_source_surfaces.json').read_text())
    if source_result['steps'] != 19750 or source_result['status'] != 'SINGLE_MOTOR_DIAGNOSTIC_PHYSICS_PASS':
        raise ValueError('designated Candidate B source run is not the expected 19,750-step PASS')
    if source_integrity.get('status') != 'UNCHANGED':
        raise ValueError('official source-integrity record is not UNCHANGED')
    out = pg.create_unique_output_directory(p.REPO/'reviews/issue-46', prefix='candidate-b-functional-validation', environment='wsl').native_path
    candidate = copy_frozen_candidate(out)
    if p.sha(out/'single_motor.xml') != provenance['mjcf_sha256']:
        raise ValueError('frozen Candidate B MJCF SHA mismatch')
    if p.sha(out/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf') != provenance['derived_urdf_sha256']:
        raise ValueError('frozen Candidate B derived URDF SHA mismatch')
    model = mujoco.MjModel.from_xml_path(str(out/'single_motor.xml'))
    result = travel.physics(out, model, p.CandidateLaw(), analysis, 'accepted-window', True)
    if p.sha(out/'single_motor.xml') != provenance['mjcf_sha256']:
        raise ValueError('frozen Candidate B MJCF changed during physics validation')
    for filename, expected in source_integrity['STL_sha256'].items():
        if p.sha(filename) != expected:
            raise ValueError(f'official visual STL changed during run: {Path(filename).name}')
    if p.sha(p.UPSTREAM/'X2-Ultra_omnipicker.urdf') != source_integrity['URDF_sha256']:
        raise ValueError('official URDF changed during physics validation')
    if p.sha(out/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf') != provenance['derived_urdf_sha256']:
        raise ValueError('frozen derived URDF changed during physics validation')
    trace = json.loads((out/'single_motor_trace.json').read_text())
    by_phase = {}
    for row in trace:
        by_phase.setdefault(int(row['phase']), []).append(row)
    joint_names = list(p.addresses(model))
    closed_hold = _hold_metrics(trace, phase=6, target=-.075, joint_names=joint_names)
    open_hold_primary = _hold_metrics(trace, phase=5, target=.2, joint_names=joint_names)
    open_hold_return = _hold_metrics(trace, phase=7, target=.2, joint_names=joint_names)
    hold_rows = {
        'closed_endpoint_q_CAD_-0.075': by_phase[6][-1],
        'open_endpoint_q_CAD_0.2': by_phase[5][-1],
        'midstroke_q_CAD_0.1': min(by_phase[2], key=lambda row: abs(float(row['actual_CAD'])-.1)),
    }
    derived = ET.parse(out/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf').getroot()
    surfaces = travel.SourceSurfaces(model, derived)
    data = mujoco.MjData(model)
    geometry = {}
    contact_sample_summary = {}
    for key, row in hold_rows.items():
        _set_saved_pose(model, data, row)
        mesh, triangles = _surface_metrics(model, derived, data)
        geometry[key] = {'step': row['step'], 'time_s': row['time_s'], 'actual_CAD': row['actual_CAD'],
                         'D_gap_per_side_mm': [float(value)*1000.0 for value in p.gaps(model, data)], **mesh}
        contacts = []
        for contact in data.contact:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (contact.geom1, contact.geom2)]
            contacts.append({'geoms': names, 'distance_mm': float(contact.dist)*1000.0})
        contact_sample_summary[key] = {'contact_count': len(contacts),
                                       'penetrating_count': sum(item['distance_mm'] < -1e-4 for item in contacts),
                                       'contacts': contacts}
        if key == 'open_endpoint_q_CAD_0.2':
            zlo, zhi = mesh['multi_ray_grid']['overlap_xz_mm'][2:]
            half = GRIP_BAND_HEIGHT_MM/2000.0
            z_centers = np.unique(np.append(np.linspace(zlo/1000.0+half, zhi/1000.0-half, 7),
                                            sum(GRIP_BAND_Z_MM)/2000.0))
            bottle_results = []
            for diameter in (65.0, 70.0, 75.0):
                bands = [cylinder_band_fit(triangles['narrow'], triangles['wide'], diameter, float(zc)) for zc in z_centers]
                best = max(bands, key=lambda item: -math.inf if item['clearance_mm'] is None else item['clearance_mm'])
                fixed = min(bands, key=lambda item: abs(item['z_center_mm']-115.0))
                if best['clearance_mm'] is None:
                    probe = {'diameter_mm': diameter, 'status': 'INDETERMINATE', 'reason': 'inner-face ray coverage is incomplete'}
                else:
                    probe = _static_collision_probe(out, out/'single_motor.xml', model, row, diameter,
                                                    best['z_center_mm'], 0.0, best['best_center_y_mm'])
                bottle_results.append({'diameter_mm': diameter, 'best_10mm_band': best,
                                       'reference_110_120mm_band': fixed, 'static_collision_probe': probe})
            geometry[key]['bottle_cylinder_fit'] = {
                'axis': 'world Z; centered on gripper X=0; cylinder Y center selected from sampled jaw bounds',
                'band_height_mm': GRIP_BAND_HEIGHT_MM, 'tested_z_centers_mm': [float(v*1000) for v in z_centers],
                'diameters': bottle_results,
                'qualification': 'sampled fit across original visual STL facing surfaces; static 10 mm collision cylinder uses the unchanged MuJoCo collision proxies; no approach sweep or bottle dynamics'}

    passive_writes = result['passive_qpos_writes_after_start']
    one_motor = result['active_actuator_count'] == 1 and result['passive_actuator_count'] == 0
    complete_phases = all(phase['completed'] for phase in result['phases'])
    open_range = any(row['command_CAD'] >= .199 for row in trace)
    close_range = any(row['command_CAD'] <= -.074 for row in trace)
    if result['status'] == 'SINGLE_MOTOR_DIAGNOSTIC_PHYSICS_PASS' and one_motor and passive_writes == 0 and complete_phases and open_range and close_range:
        physical_open_close = 'PASS'
    else:
        physical_open_close = 'FAIL'
    hold_pass = lambda hold: hold['stability_status'] == 'PASS' and hold['max_D_gap_mm'] <= MODEL_MAX_D_GAP_MM and hold['max_proxy_penetration_mm'] <= MODEL_MAX_PENETRATION_MM
    metrics = {
        'schema': 'robotsim.issue46.candidate-b-functional-validation.v1',
        'classification': 'UNVERIFIED_EXPERIMENTAL_CANDIDATE_B; not manufacturer-verified mechanics',
        'new_mj_steps': result['steps'], 'new_physics_time_s': result['simulation_time_s'],
        'frozen_model': {**candidate, 'official_root_limits': provenance['root_limits_unchanged'],
                         'actuators': result['active_actuator_count'], 'passive_actuators': result['passive_actuator_count'],
                         'passive_qpos_writes_after_start': passive_writes,
                         'model_sha_after_run': p.sha(out/'single_motor.xml')},
        'new_physics_result': result,
        'existing_evidence': {'source_physics_status': source_result['status'], 'source_physics_steps': source_result['steps'],
                              'source_physics_aperture_range_mm': [prior_measured['measured_aperture_min_mm'],
                                                                   prior_measured['measured_aperture_max_mm']],
                              'prior_original_surface_grid': {'status': prior_surface['status'],
                                                              'samples_checked': len(prior_surface['rows']),
                                                              'qualification': prior_surface['qualification']}},
        'acceptance': {
            'PHYSICAL_OPEN_CLOSE': {'status': physical_open_close, 'steps': result['steps'],
                                    'actual_CAD_range': result['actual_interval_CAD'], 'all_schedule_phases_completed': complete_phases,
                                    'one_root_motor': one_motor, 'passive_qpos_writes': passive_writes},
            'OPEN_POSITION_HOLD': {'status': 'PASS' if hold_pass(open_hold_primary) and hold_pass(open_hold_return) else 'FAIL',
                                   'primary': open_hold_primary, 'return': open_hold_return},
            'CLOSED_POSITION_HOLD': {'status': 'PASS' if hold_pass(closed_hold) else 'FAIL',
                                     'tested_endpoint_CAD': -.075, 'qualification': 'bounded closed-side endpoint; not a full-closure claim',
                                     'measurement': closed_hold},
            'D_POINT_CLOSURE': {'status': 'PASS' if result['max_D_gap_mm'] <= MODEL_MAX_D_GAP_MM and result['max_coupling_residual_rad'] <= MODEL_MAX_COUPLING_RESIDUAL_RAD else 'FAIL',
                                'max_D_gap_mm': result['max_D_gap_mm'], 'gate_mm': MODEL_MAX_D_GAP_MM,
                                'max_coupling_residual_rad': result['max_coupling_residual_rad'],
                                'hold_pose_gaps_by_side_mm': {key: value['D_gap_per_side_mm'] for key, value in geometry.items()}},
            'COLLISION_AND_LIMIT_SAFETY': {'status': 'PASS' if result['max_penetration_mm'] <= MODEL_MAX_PENETRATION_MM and result['max_root_limit_error_rad'] <= 1e-5 and not result['stop_reason'] else 'FAIL',
                                           'max_collision_proxy_penetration_mm': result['max_penetration_mm'],
                                           'gate_penetration_mm': MODEL_MAX_PENETRATION_MM,
                                           'max_root_limit_error_rad': result['max_root_limit_error_rad'],
                                           'hold_contact_samples': contact_sample_summary,
                                           'qualification': 'dynamic contacts/penetration are from the unchanged CoACD proxy collision model; exact source-mesh mating intersections remain a separate limitation'},
            'BOTTLE_GEOMETRIC_CLEARANCE': {'status': 'PENDING_GEOMETRY_RESULT',
                                          'geometry': geometry.get('open_endpoint_q_CAD_0.2', {}).get('bottle_cylinder_fit')},
        },
        'three_dimensional_gripping_region': geometry,
        'limits_and_qualifications': ['No bottle dynamics, grasp-force claim, lifting, or contact-load test.',
                                      'The coupling equality remains UNVERIFIED_EXPERIMENTAL and unchanged.',
                                      'Bottle probe is a static 10 mm high vertical cylinder envelope; no approach sweep or full bottle height was specified.',
                                      'The official URDF/STL source files were verified unchanged before the run.'],
    }
    # Bottle status requires both sampled original-surface fit and proxy contact clearance.
    bottle_rows = metrics['acceptance']['BOTTLE_GEOMETRIC_CLEARANCE']['geometry']['diameters']
    clear_diameters = [item['diameter_mm'] for item in bottle_rows
                       if item['best_10mm_band']['status'] == 'PASS' and
                       item['static_collision_probe'].get('penetrating_contact_count', 0) == 0]
    metrics['acceptance']['BOTTLE_GEOMETRIC_CLEARANCE']['status'] = (
        'PASS' if len(clear_diameters) == len(bottle_rows) else
        'PARTIAL' if clear_diameters else 'FAIL')
    metrics['acceptance']['BOTTLE_GEOMETRIC_CLEARANCE']['clear_diameters_mm'] = clear_diameters
    p.put(out, 'functional_validation.json', metrics)
    report = f"""# Issue #46 — Frozen Candidate B Functional Validation

Status: `SIMULATION_ONLY`; Candidate B coupling remains `UNVERIFIED_EXPERIMENTAL`.
New physical steps: `{result['steps']}` at `{result['simulation_time_s']:.3f} s` in MuJoCo `{result['mujoco_version']}`.

The existing CAD-derived single-root model was hash-verified before use and byte-identically copied to this run. No joint, equality, mesh, collision exclusion, limit, or actuator was changed. The new reproduction uses the existing accepted-window command schedule and controller.

## Acceptance

| Check | Result | Measurement |
|---|---|---|
| PHYSICAL_OPEN_CLOSE | `{physical_open_close}` | one motor, zero passive qpos writes, actual CAD interval `{result['actual_interval_CAD']}` |
| OPEN_POSITION_HOLD | `{metrics['acceptance']['OPEN_POSITION_HOLD']['status']}` | 0.5 s stable windows at q=0.2, see JSON for drift/rates |
| CLOSED_POSITION_HOLD | `{metrics['acceptance']['CLOSED_POSITION_HOLD']['status']}` | 0.5 s at tested q=-0.075; not fully closed |
| D_POINT_CLOSURE | `{metrics['acceptance']['D_POINT_CLOSURE']['status']}` | max gap `{result['max_D_gap_mm']:.6f} mm`; coupling residual `{result['max_coupling_residual_rad']:.6f} rad` |
| COLLISION_AND_LIMIT_SAFETY | `{metrics['acceptance']['COLLISION_AND_LIMIT_SAFETY']['status']}` | proxy penetration `{result['max_penetration_mm']:.6f} mm`; root limit error `{result['max_root_limit_error_rad']:.8f} rad` |
| BOTTLE_GEOMETRIC_CLEARANCE | `{metrics['acceptance']['BOTTLE_GEOMETRIC_CLEARANCE']['status']}` | static 65/70/75 mm vertical-cylinder probes in the open hold pose; no object dynamics |

## Three-dimensional gripping region

At the closed-side hold, intermediate q=0.1, and open hold, the original visual jaw STLs were sampled over an 11x21 grid spanning their current overlapping x-z faces. The 65/70/75 mm cylinder-envelope test checks a 10 mm high vertical slice across multiple x positions and z bands, with the candidate y-center chosen from the measured facing surfaces. An additional static cylinder is queried against the unchanged collision proxies. This is a geometric clearance test, not bottle insertion trajectory, full-height bottle fit, contact dynamics, or grasp evidence.

`functional_validation.json` contains the per-pose width distributions, hold drift, sampled passive-joint motion, per-side D gaps, contact probes, and bottle results. `single_motor.xml` has SHA-256 `{provenance['mjcf_sha256']}`; the official source integrity record is `{source_integrity['status']}`. The derived model still carries the prior candidate's four internal collision exclusions; no new exclusions were added.

## Limits

The validated command window is only `[-0.075, 0.2]` CAD coordinates. It is not full manufacturer travel or a full-closure claim. The isolated gripper does not validate wrist clearance. Bottle dynamics and lifting were not run.
"""
    p.put(out, 'REPORT.md', report)
    artifact_manifest = pg.write_artifact_manifest(out, environment='wsl')
    print(json.dumps({'output': str(out), 'new_mj_steps': result['steps'],
                      'artifact_count': len(artifact_manifest['artifacts']),
                      'acceptance': {key: value['status'] for key, value in metrics['acceptance'].items()}}, indent=2))
    return metrics


def main():
    run_validation()


if __name__ == '__main__':
    main()
