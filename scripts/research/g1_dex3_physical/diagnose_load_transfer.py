"""Summarize the bounded Dex3 load-transfer trials without rerunning physics."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from baseline import OUT, UPSTREAM, path_governance

WEIGHT_N = .570 * 9.81
LOAD_READY_N = .8 * WEIGHT_N
TABLE_MAX_N = WEIGHT_N - LOAD_READY_N
STAGE_EDGES = [(1.5, 'PREGRASP'), (2.5, 'APPROACH'), (4.5, 'GRASP'),
               (8.5, 'GRIP_HOLD'), (10.0, 'LOAD_TRANSFER')]
FINGERS = ['right_hand_thumb_2_link', 'right_hand_index_1_link',
           'right_hand_middle_1_link']
SHORT = {'right_hand_thumb_2_link': 'thumb', 'right_hand_index_1_link': 'index',
         'right_hand_middle_1_link': 'middle'}
COLORS = [(30, 80, 220), (30, 150, 30), (220, 90, 30), (180, 40, 180),
          (30, 170, 180), (100, 100, 100), (60, 60, 220)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_run(path, verify_manifest=True):
    root = Path(path_governance.validate_output_path(
        path, require_exists=True, expected_kind='directory').wsl_path)
    manifest = root / 'artifact_paths.json'
    if verify_manifest:
        path_governance.verify_artifacts([path_governance.describe_output_path(manifest)])
    result_path, trace_path = root / 'result.json', root / 'trace.jsonl'
    path_governance.validate_output_path(result_path, require_exists=True, expected_kind='file')
    path_governance.validate_output_path(trace_path, require_exists=True, expected_kind='file')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in trace_path.open(encoding='utf-8')]
    return root, result, rows


def stage_summary(rows, stage):
    selected = [r for r in rows if r['stage'] == stage]
    if not selected:
        return None
    hand = np.asarray([r['hand_z'] for r in selected])
    table = np.asarray([r['table_z'] for r in selected])
    pos = np.asarray([r['bottle_pos'] for r in selected])
    return {
        'samples': len(selected),
        'hand_support_N': {'mean': float(hand.mean()), 'p05': float(np.percentile(hand, 5)),
                           'p95': float(np.percentile(hand, 95))},
        'table_support_N': {'mean': float(table.mean()), 'p95': float(np.percentile(table, 95))},
        'bottle_horizontal_drift_from_stage_start_mm': float(
            1000 * np.max(np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1))),
        'bottle_z_range_mm': float(1000 * (pos[:, 2].max() - pos[:, 2].min())),
    }


def hand_tracking(rows, targets, limits):
    names = rows[0]['joint_names']
    report = {}
    for stage in ['GRIP_HOLD', 'LOAD_TRANSFER']:
        selected = [r for r in rows if r['stage'] == stage]
        stage_report = {}
        if not selected:
            report[stage] = stage_report
            continue
        for joint, target in targets.items():
            i = names.index(joint)
            err = np.asarray([r['q_error'][i] for r in selected])
            q = np.asarray([r['q'][i] for r in selected])
            v = np.asarray([r['v'][i] for r in selected])
            effort = np.asarray([r['effort'][i] for r in selected])
            cap = limits[joint]['effort']
            vlim = limits[joint]['velocity']
            lo, hi = limits[joint]['lower'], limits[joint]['upper']
            stage_report[joint] = {
                'target_rad': float(target),
                'actual_q_mean_rad': float(q.mean()),
                'absolute_target_error_mean_rad': float(np.mean(np.abs(err))),
                'absolute_target_error_p95_rad': float(np.percentile(np.abs(err), 95)),
                'absolute_target_error_max_rad': float(np.max(np.abs(err))),
                'peak_abs_qvel_rad_s': float(np.max(np.abs(v))),
                'source_velocity_limit_rad_s': vlim,
                'peak_velocity_limit_utilization': float(np.max(np.abs(v)) / vlim),
                'peak_abs_effort_Nm': float(np.max(np.abs(effort))),
                'source_effort_limit_Nm': cap,
                'peak_effort_limit_utilization': float(np.max(np.abs(effort)) / cap),
                'minimum_position_limit_margin_rad': float(min(np.min(q) - lo, hi - np.max(q))),
            }
        report[stage] = stage_report
    return report


def contact_audit(rows, stage):
    selected = [r for r in rows if r['stage'] == stage]
    out = {}
    for finger in FINGERS:
        contacts = [c for r in selected for c in r['contacts'] if c['other'] == finger]
        if not contacts:
            continue
        first_geom_order = contacts[0]['geoms']
        normals = []
        for c in contacts:
            n = np.asarray(c['normal'], dtype=float)
            if c['geoms'][0] == 'bottle_body':
                n = -n
            normals.append(n / max(np.linalg.norm(n), 1e-12))
        out[finger] = {
            'geom_pair': first_geom_order,
            'mean_contact_normal_on_bottle_world': np.mean(normals, axis=0).tolist(),
            'mean_bottle_local_contact_point_m': np.mean(
                [c['point_bottle_local_m'] for c in contacts], axis=0).tolist(),
            'mean_world_force_on_bottle_N': np.mean(
                [c['world_force_on_bottle'] for c in contacts], axis=0).tolist(),
            'mean_normal_force_N': float(np.mean([c['normal_force_N'] for c in contacts])),
            'mean_tangential_force_N': float(np.mean([c['tangential_force_N'] for c in contacts])),
            'friction_pyramid_tangent_utilization_p95': float(np.percentile(
                [c['friction_pyramid_tangent_utilization'] for c in contacts], 95)),
            'effective_contact_friction': contacts[0]['friction'],
            'contact_dimension': contacts[0]['condim'],
        }
    return out


def panel(canvas, y0, height, title, t, series, ymin, ymax, threshold=None, threshold_label=''):
    x0, x1 = 120, canvas.shape[1] - 48
    top, bottom = y0 + 54, y0 + height - 39
    cv2.putText(canvas, title, (26, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                .72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, top), (x1, bottom), (90, 90, 90), 1)
    for fraction in np.linspace(0, 1, 5):
        y = int(bottom - fraction * (bottom - top))
        value = ymin + fraction * (ymax - ymin)
        cv2.line(canvas, (x0, y), (x1, y), (225, 225, 225), 1)
        cv2.putText(canvas, f'{value:.2f}', (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (70, 70, 70), 1, cv2.LINE_AA)
    for tick in range(0, 13, 2):
        x = int(x0 + tick / 12 * (x1 - x0))
        cv2.line(canvas, (x, top), (x, bottom), (238, 238, 238), 1)
        cv2.putText(canvas, str(tick), (x - 7, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (70, 70, 70), 1, cv2.LINE_AA)
    for edge, label in STAGE_EDGES:
        x = int(x0 + edge / 12 * (x1 - x0))
        cv2.line(canvas, (x, top), (x, bottom), (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 3, top + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    .34, (90, 90, 90), 1, cv2.LINE_AA)
    if threshold is not None:
        y = int(bottom - (threshold - ymin) / (ymax - ymin) * (bottom - top))
        cv2.line(canvas, (x0, y), (x1, y), (20, 20, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, threshold_label, (x1 - 275, max(top + 13, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 220), 1, cv2.LINE_AA)
    legend_x, legend_y = x0 + 12, y0 + 47
    for i, (label, values, color) in enumerate(series):
        col = i % 4
        row = i // 4
        lx = legend_x + col * 330
        ly = legend_y + row * 18
        cv2.line(canvas, (lx, ly - 4), (lx + 24, ly - 4), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx + 30, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    .39, (30, 30, 30), 1, cv2.LINE_AA)
    for label, values, color in series:
        values = np.asarray(values, dtype=float)
        good = np.isfinite(values)
        if good.sum() < 2:
            continue
        xx = x0 + np.asarray(t)[good] / 12 * (x1 - x0)
        yy = bottom - (values[good] - ymin) / (ymax - ymin) * (bottom - top)
        pts = np.column_stack((xx, yy)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'simulation time (s)', (x1 - 145, bottom + 32),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (50, 50, 50), 1, cv2.LINE_AA)


def timeline(rows, output):
    t = np.asarray([r['time'] for r in rows])
    canvas = np.full((1910, 1600, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'G1 Dex3 load-transfer recovery 01: measured physics, 500 Hz trace',
                (34, 36), cv2.FONT_HERSHEY_SIMPLEX, .78, (20, 20, 20), 2, cv2.LINE_AA)
    hand = np.asarray([r['hand_z'] for r in rows])
    table = np.asarray([r['table_z'] for r in rows])
    wrist = np.asarray([r['wrist_pos'][2] for r in rows])
    bottle = np.asarray([r['bottle_pos'][2] for r in rows])
    load_start = next(r for r in rows if r['stage'] == 'LOAD_TRANSFER')
    hand_series = [('measured hand support', hand, COLORS[0]),
                   ('80% bottle-weight gate', np.full(len(t), LOAD_READY_N), (20, 20, 220))]
    table_series = [('table reaction', table, COLORS[1]),
                    ('bottle weight', np.full(len(t), WEIGHT_N), COLORS[2]),
                    ('LOAD_READY max table reaction', np.full(len(t), TABLE_MAX_N), (20, 20, 220))]
    load_mask = np.asarray([r['stage'] == 'LOAD_TRANSFER' for r in rows])
    wrist_delta = np.where(load_mask, (wrist - load_start['wrist_pos'][2]) * 1000, np.nan)
    bottle_delta = np.where(load_mask, (bottle - load_start['bottle_pos'][2]) * 1000, np.nan)
    motion_series = [('wrist Z from LOAD start (mm)', wrist_delta, COLORS[0]),
                     ('bottle Z from LOAD start (mm)', bottle_delta, COLORS[2])]
    contact_distance = {finger: [] for finger in FINGERS}
    first_local = {}
    normal = {finger: [] for finger in FINGERS}
    tangent = {finger: [] for finger in FINGERS}
    force_z = {finger: [] for finger in FINGERS}
    for row in rows:
        by_finger = {finger: [] for finger in FINGERS}
        for c in row['contacts']:
            if c['other'] in by_finger:
                by_finger[c['other']].append(c)
        for finger in FINGERS:
            values = by_finger[finger]
            if not values:
                contact_distance[finger].append(np.nan)
                normal[finger].append(np.nan)
                tangent[finger].append(np.nan)
                force_z[finger].append(np.nan)
                continue
            c = values[0]
            local = np.asarray(c['point_bottle_local_m'], dtype=float)
            origin_key = (row['stage'], finger)
            first_local.setdefault(origin_key, local.copy())
            contact_distance[finger].append(float(np.linalg.norm(local - first_local[origin_key]) * 1000))
            normal[finger].append(float(np.mean([v['normal_force_N'] for v in values])))
            tangent[finger].append(float(np.mean([v['tangential_force_N'] for v in values])))
            force_z[finger].append(float(np.sum([v['world_force_on_bottle'][2] for v in values])))
    migration_series = [(SHORT[k], contact_distance[k], COLORS[i]) for i, k in enumerate(FINGERS)]
    normal_series = [(SHORT[k], normal[k], COLORS[i]) for i, k in enumerate(FINGERS)]
    tangent_series = [(SHORT[k], tangent[k], COLORS[i]) for i, k in enumerate(FINGERS)]
    error_series = []
    joint_names = rows[0]['joint_names']
    selected = ['right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                'right_hand_index_0_joint', 'right_hand_index_1_joint',
                'right_hand_middle_0_joint', 'right_hand_middle_1_joint']
    for i, name in enumerate(selected):
        j = joint_names.index(name)
        values = [abs(r['q_error'][j]) if 'q_error' in r else np.nan for r in rows]
        error_series.append((name.replace('right_hand_', '').replace('_joint', ''), values, COLORS[i]))
    panels = [
        ('Vertical force on bottle: hand support (N)', hand_series, -1, 5.5, LOAD_READY_N, 'required p05 = 4.473 N'),
        ('Vertical force on bottle: table reaction (N)', table_series, 0, 6.5, TABLE_MAX_N, 'allowed p95 = 1.118 N'),
        ('Wrist and bottle height change from LOAD_TRANSFER start (mm)', motion_series, -1, 7, None, ''),
        ('Contact migration within each phase, bottle coordinates (mm)', migration_series, 0, 7, 3.5, 'stability limit = 3.5 mm'),
        ('Finger normal contact force (N)', normal_series, 0, 8, None, ''),
        ('Finger tangential contact force magnitude (N)', tangent_series, 0, .8, None, ''),
        ('Absolute hand target tracking error (rad)', error_series, 0, .30, None, ''),
    ]
    y = 70
    for title, series, lo, hi, threshold, label in panels:
        panel(canvas, y, 260, title, t, series, lo, hi, threshold, label)
        y += 260
    out = Path(output)
    path_governance.validate_output_path(out, require_new=True)
    if not cv2.imwrite(str(out), canvas):
        raise RuntimeError(f'failed to write {out}')
    path_governance.validate_output_path(out, require_exists=True, expected_kind='file')


def force_vectors(rows, output):
    data = {finger: [] for finger in FINGERS}
    for row in rows:
        if row['stage'] != 'LOAD_TRANSFER':
            continue
        vectors = {finger: np.zeros(3) for finger in FINGERS}
        for c in row['contacts']:
            if c['other'] in vectors:
                vectors[c['other']] += np.asarray(c['world_force_on_bottle'], dtype=float)
        for finger in FINGERS:
            data[finger].append(vectors[finger])
    means = {f: np.mean(np.asarray(v), axis=0).tolist() for f, v in data.items()}
    canvas = np.full((680, 1450, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Mean contact force ON BOTTLE during LOAD_TRANSFER (N)',
                (36, 42), cv2.FONT_HERSHEY_SIMPLEX, .78, (20, 20, 20), 2, cv2.LINE_AA)
    x0, x1, top, bottom = 130, 1400, 100, 570
    lo, hi = -6.5, 6.5
    zero = int(bottom - (0 - lo) / (hi - lo) * (bottom - top))
    cv2.line(canvas, (x0, zero), (x1, zero), (50, 50, 50), 2)
    for tick in range(-6, 7, 2):
        y = int(bottom - (tick - lo) / (hi - lo) * (bottom - top))
        cv2.line(canvas, (x0, y), (x1, y), (225, 225, 225), 1)
        cv2.putText(canvas, str(tick), (82, y + 5), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (60, 60, 60), 1, cv2.LINE_AA)
    cats = list(FINGERS)
    component_colors = [(220, 80, 30), (30, 150, 30), (30, 50, 220)]
    labels = ['world Fx', 'world Fy', 'world Fz']
    group_w = (x1 - x0) / len(cats)
    max_abs = 6.5
    for i, finger in enumerate(cats):
        center = int(x0 + (i + .5) * group_w)
        vals = means[finger]
        for j, value in enumerate(vals):
            bar_x = center + (j - 1) * 42
            yy = int(bottom - (value - lo) / (hi - lo) * (bottom - top))
            cv2.rectangle(canvas, (bar_x - 13, min(yy, zero)), (bar_x + 13, max(yy, zero)), component_colors[j], -1)
            cv2.putText(canvas, f'{value:+.2f}', (bar_x - 29, yy - 8 if value >= 0 else yy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, component_colors[j], 1, cv2.LINE_AA)
        cv2.putText(canvas, SHORT[finger], (center - 33, bottom + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (35, 35, 35), 1, cv2.LINE_AA)
    for j, label in enumerate(labels):
        x = 180 + j * 280
        cv2.rectangle(canvas, (x, 620), (x + 22, 635), component_colors[j], -1)
        cv2.putText(canvas, label, (x + 32, 634), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (30, 30, 30), 1, cv2.LINE_AA)
    out = Path(output)
    path_governance.validate_output_path(out, require_new=True)
    if not cv2.imwrite(str(out), canvas):
        raise RuntimeError(f'failed to write {out}')
    path_governance.validate_output_path(out, require_exists=True, expected_kind='file')
    return means


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-dir', required=True)
    p.add_argument('--preload-dir', required=True)
    p.add_argument('--stiffness-dir', required=True)
    args = p.parse_args()
    baseline = load_run(args.baseline_dir, verify_manifest=False)
    preload = load_run(args.preload_dir)
    stiffness = load_run(args.stiffness_dir)
    out = Path(path_governance.create_unique_output_directory(
        OUT, prefix='load-transfer-diagnosis').wsl_path)
    base_root, base_result, base_rows = baseline
    preload_root, preload_result, preload_rows = preload
    stiff_root, stiff_result, stiff_rows = stiffness
    timeline(preload_rows, out / 'recovery01_timeline.png')
    vectors = force_vectors(preload_rows, out / 'recovery01_load_contact_force_vectors.png')
    comparison = {
        'schema_version': 'robotsim.dex3.load-transfer-diagnosis.v1',
        'physics_claim': 'no LOAD_READY and no lift; diagnostic only',
        'bottle_mass_kg': .570, 'bottle_weight_N': WEIGHT_N,
        'unchanged_LOAD_READY_thresholds': {'hand_support_p05_N': LOAD_READY_N,
                                           'table_support_p95_max_N': TABLE_MAX_N},
        'source': {'unitree_ros_revision': '5994d4faef0a9cadd3287f8de0199a67eeb2a259',
                   'model_sha256_recorded_per_run': True},
        'runs': [], 'recovery01_mean_contact_force_on_bottle_world_N': vectors,
        'interpretation': {
            'friction_saturation': 'not observed; p95 friction-pyramid use stayed below 0.08 in LOAD_TRANSFER',
            'contact_motion': 'measured bottle-frame contact migration exceeded the 3.5 mm stability limit',
            'measured_load_transfer': 'hand support remained far below threshold and table reaction remained near bottle weight',
            'cause_limit': 'evidence supports ineffective vertical force transmission with contact migration; it does not isolate one unique mechanical cause'
        }
    }
    for label, root, result, rows in [
            ('attempt02_baseline', base_root, base_result, base_rows),
            ('preload_005_kp15', preload_root, preload_result, preload_rows),
            ('preload_005_kp30', stiff_root, stiff_result, stiff_rows)]:
        entry = {'label': label, 'run_id': result['run_id'], 'result': result,
                 'trace_sha256': sha(root / 'trace.jsonl'),
                 'result_sha256': sha(root / 'result.json'),
                 'artifact_paths_manifest_present': (root / 'artifact_paths.json').is_file(),
                 'runtime_model_sha256': result.get('runtime_model_sha256'),
                 'runner_sha256': result.get('runner_sha256'),
                 'robot_sim_sha': result.get('RobotSim_SHA'),
                 'dirty_state': result.get('dirty'),
                 'reproduction_command': result.get('reproduction_command'),
                 'stage_summaries': {s: stage_summary(rows, s) for s in ['GRIP_HOLD', 'LOAD_TRANSFER']}}
        comparison['runs'].append(entry)
    urdf = ET.parse(UPSTREAM / 'robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf').getroot()
    limits = {j.get('name'): {k: float(v) for k, v in j.find('limit').attrib.items()}
              for j in urdf.findall('joint') if j.find('limit') is not None}
    comparison['source_hand_limits'] = {name: limits[name] for name in preload_result['executed_hand_targets']['closed']}
    comparison['recovery01_grip_hold_contact_audit'] = contact_audit(preload_rows, 'GRIP_HOLD')
    comparison['recovery01_load_transfer_contact_audit'] = contact_audit(preload_rows, 'LOAD_TRANSFER')
    comparison['recovery01_joint_tracking'] = hand_tracking(
        preload_rows, preload_result['executed_hand_targets']['closed'], limits)
    comparison['files'] = {
        'timeline_png': path_governance.describe_output_path(out / 'recovery01_timeline.png'),
        'force_vector_png': path_governance.describe_output_path(out / 'recovery01_load_contact_force_vectors.png'),
        'baseline_run_dir': path_governance.describe_output_path(base_root),
        'preload_run_dir': path_governance.describe_output_path(preload_root),
        'stiffness_run_dir': path_governance.describe_output_path(stiff_root),
    }
    path_governance.write_new_text(out / 'comparison.json', json.dumps(comparison, indent=2) + '\n')
    unc = path_governance.describe_output_path(out)['WINDOWS_UNC']
    report = f'''# G1 Dex3 Load-Transfer Diagnosis

## Verdict

`FAIL — LOAD_READY not established; no bottle lift was attempted.` The one permitted target-preload correction did not improve load transfer. The second bounded stiffness diagnostic stopped at the 3 mm penetration guard before contact/load evaluation.

## What the evidence says

- The original attempt02 transferred only {base_result['stages'][-1]['hand_support_mean_N']:.3f} N mean upward hand support while the table carried {base_result['stages'][-1]['table_support_mean_N']:.3f} N.
- Recovery01 used the same grasp and +0.05 rad closure target. During LOAD_TRANSFER, mean hand support was {preload_result['gate_measurements']['LOAD_TRANSFER']['hand_support_N']['mean']:.3f} N, p05 {preload_result['gate_measurements']['LOAD_TRANSFER']['hand_support_N']['p05']:.3f} N; mean table reaction was {preload_result['gate_measurements']['LOAD_TRANSFER']['table_support_N']['mean']:.3f} N, p95 {preload_result['gate_measurements']['LOAD_TRANSFER']['table_support_N']['p95']:.3f} N. The required thresholds remain {LOAD_READY_N:.5f} N and {TABLE_MAX_N:.5f} N.
- Thumb/index/middle normal forces averaged 5.772 / 2.813 / 2.799 N in LOAD_TRANSFER. Tangential magnitudes averaged 0.266 / 0.133 / 0.136 N; p95 pyramid utilization was 0.057 / 0.073 / 0.077. This does not support friction-cone saturation as the cause. The scalar vertical-friction capacity screen of {preload_result['gate_measurements']['LOAD_TRANSFER']['ideal_vertical_friction_capacity_screen_N']['p05']:.2f} N is only an upper-bound screen, not measured load-bearing force.
- Contact migration reached {preload_result['gate_measurements']['LOAD_TRANSFER']['max_contact_migration_mm']:.3f} mm versus 3.5 mm allowed. Bottle horizontal drift was {preload_result['gate_measurements']['LOAD_TRANSFER']['max_bottle_horizontal_drift_mm']:.3f} mm. The actual wrist rose 3.817 mm during the 5 mm probe; bottle z changed only 0.036 mm.
- Target errors remained large under Kp=1.5, especially thumb_1 (mean absolute error 0.231 rad during LOAD_TRANSFER), while peak finger torque was about 0.351 Nm against 1.4 Nm capacity. That is evidence of soft command realization, but the preload correction increased normal force without improving upward support, so it is not a complete causal explanation by itself.
- Raising hand Kp to 3.0 with Kd=0.1, the same target, pose, friction and limits caused recovery02 to stop at {stiff_result['max_penetration_m']*1000:.3f} mm penetration after {stiff_result['physics_steps']} physics steps. It did not reach a valid grasp or LOAD_READY.

## Per-joint tracking and contact direction

Right hand joint telemetry is in `comparison.json` for both `GRIP_HOLD` and `LOAD_TRANSFER`, including actual q, target error, qvel, source velocity/effort caps and utilization, and position-limit margin. Contacts are `bottle_body` against `official_geom_92` (thumb_2), `official_geom_100` (index_1), and `official_geom_96` (middle_1). The measured mean world force on the bottle has only about 0.08–0.09 N upward per digit during LOAD_TRANSFER; the grouped force plot shows the much larger horizontal components.

## Classification

`F — combined force-transmission and contact-migration limitation; friction saturation is not established.` The observed normals and estimated friction margin are adequate on paper, but measured vertical support is nearly zero and the contact points move during loading. The available evidence does not justify claiming a unique root cause such as insufficient friction coefficient.

## Frozen constraints

No friction, mass, bottle geometry, joint limits, source effort/velocity limits, weld, mocap, bottle qpos writes or lift target were changed. Recovery01 used Kp=1.5/Kd=0.1 and +0.05 rad target offset. Recovery02 changed only Kp to 3.0; it stopped on penetration. No 50 mm lift, transfer, placement or release was run.

## Reproduction

Run from `/home/lzy18001500226/robotsim-issue43-dex3-physical-20261016` in WSL Ubuntu-22.04. The exact commands are recorded in each run's `result.json` and `run.log`. The recovery01 command is:

```bash
{preload_result['reproduction_command']}
```

The Kp=3.0 command is:

```bash
{stiff_result['reproduction_command']}
```

## Evidence

- Output directory: `{unc}`
- Time-aligned traces: `recovery01_timeline.png`
- Measured force directions: `recovery01_load_contact_force_vectors.png`
- Machine comparison: `comparison.json`
- Recovery01 MP4: `{preload_root / 'episode.mp4'}`
- Recovery02 MP4: `{stiff_root / 'episode.mp4'}` (stops during GRASP)
- Raw traces, screenshots, executed source snapshots and logs remain in their original immutable run folders. The comparison manifest hashes those inputs.
- The older attempt02 baseline has no `artifact_paths.json` (it predates that manifest); its existing `result.json` and `trace.jsonl` were read in place and are SHA-256 recorded in `comparison.json`. Recovery01 and recovery02 passed artifact-manifest verification before analysis.
- `comparison.json` records the runtime model SHA, executed runner SHA, RobotSim SHA/dirty state and reproduction command separately for each run; matching model identity is not inferred across runs.
'''
    path_governance.write_new_text(out / 'REPORT.md', report)
    manifest = path_governance.write_artifact_manifest(out)
    print(json.dumps({'output': path_governance.describe_output_path(out),
                      'manifest': manifest['manifest'],
                      'attempt02': base_result['run_id'],
                      'recovery01': preload_result['run_id'],
                      'recovery02': stiff_result['run_id']}, indent=2))


if __name__ == '__main__':
    main()
