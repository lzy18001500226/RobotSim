#!/usr/bin/env python3
"""Isolated one-motor OmniPicker candidate with CAD-located B-axis springs.

The STEP supports torsion-spring occurrences coaxial with both B pivots, but
does not provide spring-end ownership or calibration. All spring parameters and
the resulting mechanics are therefore UNVERIFIED_EXPERIMENTAL. This runner
never writes passive qpos after source-pose initialization.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import issue46_urdf_pipeline as p
import issue46_single_motor_travel as t

ALIGNMENT = p.REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json'
DECOMPOSITION = p.REPO/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json'
SPRING_STIFFNESS = (0.01, 0.05, 0.20)  # Nm/rad; bounded diagnostic values, not vendor data.
SPRING_REFERENCE = (-0.05, 0.0, 0.05)  # rad about the registered STEP pose.
SPRING_DAMPING = 0.002  # Nm*s/rad; experimental numerical/physical sensitivity value.
ROOT_SERVO_KP = 2.0  # Experimental root tracking correction; not a vendor gain.
ROOT_SERVO_KV = 0.1
ROOT_SERVO_FORCE_LIMIT_NM = 0.5  # Preserve the previously used torque cap.
ROOT_CLOSE = -0.01  # Numerical command margin only; official limits remain [-1, 0].
ROOT_OPEN = -0.99
ROOT_PHASES = (('close', ROOT_CLOSE, 2.0), ('open', ROOT_OPEN, 3.0), ('close', ROOT_CLOSE, 3.0),
               ('open', ROOT_OPEN, 3.0), ('close', ROOT_CLOSE, 3.0))


def opening_from_root(root: float) -> float:
    return (ROOT_CLOSE-float(root))/(ROOT_CLOSE-ROOT_OPEN)


def put(run: Path, name: str, value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, indent=2, allow_nan=False)+'\n'
    p.pg.write_new_text(run/name, value, environment='wsl')


def make_model(run: Path, stiffness: float, springref: float, damping: float):
    source = p.UPSTREAM/'X2-Ultra_omnipicker.urdf'
    for path, expected in ((source, p.URDF_HASH), (ALIGNMENT, p.ALIGNMENT_HASH), (DECOMPOSITION, p.DECOMPOSITION_HASH)):
        if p.sha(path) != expected:
            raise ValueError(f'reviewed input hash mismatch: {path.name}')
    raw = source.read_bytes()
    alignment = json.loads(ALIGNMENT.read_text())
    decomposition = json.loads(DECOMPOSITION.read_text())
    derived, changes = p.derive_urdf(raw, alignment)
    derived_text = p.serialize_derived(raw, derived)
    put(run, 'X2-Ultra_omnipicker_PASSIVE_SPRING_EXPERIMENTAL.urdf', derived_text)
    put(run, 'official_to_derived.diff', ''.join(difflib.unified_diff(raw.decode().splitlines(True), derived_text.splitlines(True),
                                                                      fromfile='official.urdf', tofile='derived.urdf')))
    xml = p.convert(derived, decomposition, actuation='single-motor-passive-spring-experimental',
                    spring_stiffness_nm_rad=stiffness, spring_reference_rad=springref,
                    spring_damping_nm_s_rad=damping, root_servo_kp=ROOT_SERVO_KP,
                    root_servo_kv=ROOT_SERVO_KV, root_servo_force_limit_nm=ROOT_SERVO_FORCE_LIMIT_NM)
    ET.indent(xml, space='  ')
    put(run, 'passive_spring.xml', ET.tostring(xml, encoding='unicode')+'\n')
    model = mujoco.MjModel.from_xml_path(str(run/'passive_spring.xml'))
    if model.nu != 1 or model.neq != 3 or model.ntendon != 0:
        raise ValueError('candidate must have one root actuator, two D closures, mimic, and no tendon')
    provenance = {
        'classification': 'UNVERIFIED_EXPERIMENTAL',
        'source_commit': p.PIN,
        'official_urdf_sha256': p.URDF_HASH,
        'derived_urdf_sha256': p.sha(run/'X2-Ultra_omnipicker_PASSIVE_SPRING_EXPERIMENTAL.urdf'),
        'mjcf_sha256': p.sha(run/'passive_spring.xml'),
        'step_sha256': '5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b',
        'cad_spring_evidence': {
            'component_product': '220000054701',
            'narrow_B_center_mm': [-58.990803883, 209.775099922],
            'wide_B_center_mm': [23.486014595, 225.818328011],
            'verified': 'torsion-spring occurrences are coaxial with the measured B pivots in both branches',
            'unknown': ['spring-end attachment/ownership', 'handedness', 'stiffness', 'preload', 'damping']},
        'experimental_assumptions': {
            'spring_joints': list(p.TORSION_SPRING_JOINTS),
            'spring_stiffness_Nm_per_rad': stiffness,
            'spring_reference_rad_from_registered_source_pose': springref,
            'spring_damping_Nm_s_per_rad': damping,
            'root_servo': {'kp': ROOT_SERVO_KP, 'kv': ROOT_SERVO_KV,
                           'force_limit_Nm': ROOT_SERVO_FORCE_LIMIT_NM,
                           'classification': 'UNVERIFIED_SIMULATION_CONTROL; chosen from measured gravity-bias margin, not vendor data'},
            'mounting_hypothesis': 'torsion spring acts between outer and middle links at B; CAD centerline supports location only',
            'normalized_test_command': 'opening in [0,1] maps linearly across tested interior root q=-0.01..-0.99 rad; not a manufacturer endpoint map',
            'unmodeled': ['gear/slot transmission', 'unverified spring end geometry', 'hard stops beyond official root limits']},
        'model_invariants': {'root_limits_unchanged': [-1.0, 0.0], 'official_mimic_retained': True,
                             'D_point_equalities': 2, 'candidate_B_rate_equalities': 0, 'tendons': 0,
                             'actuator_count': model.nu, 'active_joint': 'right_claw_joint',
                             'passive_actuators': model.nu-1, 'passive_qpos_writes_after_initialization': 0,
                             'inherited_internal_exclusions': [list(x) for x in p.INTERNAL_EXCLUSIONS]},
        'changes': changes,
        'input_hashes': {str(f): p.sha(f) for f in (source, ALIGNMENT, DECOMPOSITION)}}
    put(run, 'provenance.json', provenance)
    return model, derived, provenance


def initial_source_state(model, data, law):
    values, closure = t.set_static(model, data, law, 0.0)
    if abs(closure) > 1e-8 or max(t.p.gaps(model, data)) > 1e-5:
        raise ValueError('registered source-pose D closure preflight failed')
    if t.proxy_contacts(model, data):
        raise ValueError('registered source-pose has unresolved collision-proxy penetration')
    return values


def spring_sensitivity(run: Path, derived, law):
    rows = []
    for k in SPRING_STIFFNESS:
        for ref in SPRING_REFERENCE:
            xml = p.convert(derived, json.loads(DECOMPOSITION.read_text()),
                            actuation='single-motor-passive-spring-experimental',
                            spring_stiffness_nm_rad=k, spring_reference_rad=ref,
                            spring_damping_nm_s_rad=SPRING_DAMPING, root_servo_kp=ROOT_SERVO_KP,
                            root_servo_kv=ROOT_SERVO_KV, root_servo_force_limit_nm=ROOT_SERVO_FORCE_LIMIT_NM)
            model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
            data = mujoco.MjData(model)
            initial_source_state(model, data, law)
            surfaces = t.SourceSurfaces(model, derived)
            initial_aperture, _ = surfaces.aperture(data)
            maxspeed = maxgap = maxpen = maxlimit = 0.0
            stop = None
            initial_root = float(data.qpos[t.p.addresses(model)['right_claw_joint']])
            for step in range(500):
                # Bounded source-pose-to-close-side step; no state write after initialization.
                blend = 0.5-0.5*math.cos(math.pi*min(step/249.0, 1.0))
                data.ctrl[0] = initial_root*(1.0-blend)+ROOT_CLOSE*blend
                mujoco.mj_step(model, data)
                root = float(data.qpos[t.p.addresses(model)['right_claw_joint']])
                gap = max(t.p.gaps(model, data))*1000.0
                pen = max([c['penetration_mm'] for c in t.proxy_contacts(model, data)] or [0.0])
                speed = float(np.max(np.abs(data.qvel)))
                limit = max(0.0, root, -1.0-root)
                maxgap, maxpen, maxspeed, maxlimit = max(maxgap, gap), max(maxpen, pen), max(maxspeed, speed), max(maxlimit, limit)
                if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                    stop = 'NONFINITE'
                elif limit > 1e-5:
                    stop = 'OFFICIAL_ROOT_LIMIT_OVERSHOOT'
                elif gap > 1.0:
                    stop = 'D_POINT_GAP_OVER_1MM'
                elif pen > 2.0:
                    stop = 'PROXY_PENETRATION_OVER_2MM'
                elif speed > 25.0:
                    stop = 'SPEED_OVER_25_RAD_S'
                elif any(w.number for w in data.warning):
                    stop = 'MUJOCO_WARNING'
                if stop:
                    break
            rows.append({'stiffness_Nm_per_rad': k, 'springref_rad': ref,
                         'damping_Nm_s_per_rad': SPRING_DAMPING, 'steps': step+1,
                         'actual_root_rad': float(data.qpos[t.p.addresses(model)['right_claw_joint']]),
                         'passive_angles_rad': {name: float(data.qpos[adr]) for name, adr in t.p.addresses(model).items()
                                                if name not in ('right_claw_joint', 'R_hand_wide1_joint')},
                         'initial_aperture_mm': initial_aperture,
                         'final_aperture_mm': surfaces.aperture(data)[0],
                         'max_D_gap_mm': maxgap, 'max_proxy_penetration_mm': maxpen,
                         'max_root_limit_overshoot_rad': maxlimit, 'max_speed_rad_s': maxspeed,
                         'status': 'PASS' if stop is None else 'FAIL', 'stop_reason': stop})
    put(run, 'spring_sensitivity.json', {'purpose': 'bounded parameter sensitivity, not parameter fitting',
                                         'grid_stiffness_Nm_per_rad': list(SPRING_STIFFNESS),
                                         'grid_reference_rad': list(SPRING_REFERENCE),
                                         'root_servo': {'kp': ROOT_SERVO_KP, 'kv': ROOT_SERVO_KV,
                                                        'force_limit_Nm': ROOT_SERVO_FORCE_LIMIT_NM},
                                         'rows': rows})
    return rows


def run_cycle(run: Path, model, derived, law):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    data = mujoco.MjData(model)
    initial_source_state(model, data, law)
    surfaces = t.SourceSurfaces(model, derived)
    source_intersections = surfaces.interference(data)
    baseline_pairs = {tuple(sorted((x['body1'], x['body2']))) for x in source_intersections}
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, .12]; camera.distance = .34; camera.azimuth = 0; camera.elevation = -12
    options = mujoco.MjvOption(); options.geomgroup[1] = 0
    writer = imageio.get_writer(str(run/'passive_spring_open_close.mp4'), fps=25, codec='libx264')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    rows, interference_rows, phases = [], [], []
    stop = None
    maxgap = maxpen = maxspeed = maxmimic = 0.0
    widths = []
    rootadr = t.p.addresses(model)['right_claw_joint']
    steps = 0
    for phase, (label, target, duration) in enumerate(ROOT_PHASES):
        start = float(data.qpos[rootadr])
        for index in range(round(duration/float(model.opt.timestep))):
            blend = 0.5-0.5*math.cos(math.pi*min(index*float(model.opt.timestep)/max(duration-.5, .001), 1.0))
            data.ctrl[0] = start+(target-start)*blend
            mujoco.mj_step(model, data)
            steps += 1
            mujoco.mj_forward(model, data)
            root = float(data.qpos[rootadr])
            speed = float(np.max(np.abs(data.qvel)))
            gap = max(t.p.gaps(model, data))*1000.0
            proxy = t.proxy_contacts(model, data)
            pen = max([x['penetration_mm'] for x in proxy] or [0.0])
            wideadr = t.p.addresses(model)['R_hand_wide1_joint']
            mimic = abs(root+float(data.qpos[wideadr]))
            maxgap, maxpen, maxspeed, maxmimic = max(maxgap, gap), max(maxpen, pen), max(maxspeed, speed), max(maxmimic, mimic)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                stop = 'NONFINITE_STATE'
            elif root < -1.0-1e-5 or root > 1e-5:
                stop = 'OFFICIAL_ROOT_LIMIT_OVERSHOOT'
            elif gap > 1.0:
                stop = 'D_POINT_GAP_OVER_1MM'
            elif pen > 2.0:
                stop = 'PROXY_PENETRATION_OVER_2MM'
            elif speed > 25.0:
                stop = 'SPEED_OVER_25_RAD_S'
            elif any(w.number for w in data.warning):
                stop = 'MUJOCO_WARNING'
            sampled = steps % 20 == 0 or stop is not None
            if sampled:
                aperture, sections = surfaces.aperture(data)
                if aperture is not None:
                    widths.append({'phase': phase, 'time_s': float(data.time), 'command_opening': opening_from_root(data.ctrl[0]),
                                   'actual_root_rad': root, 'aperture_mm': aperture})
                elif stop is None:
                    stop = 'GRIPPING_SURFACE_SECTION_NOT_MEASURABLE'
                spring_torque = {}
                for name in p.TORSION_SPRING_JOINTS:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    dof = int(model.jnt_dofadr[jid])
                    spring_torque[name] = float(data.qfrc_passive[dof])
                if steps % 250 == 0 or stop:
                    pairs = surfaces.interference(data)
                    now = {tuple(sorted((x['body1'], x['body2']))) for x in pairs}
                    interference_rows.append({'step': steps, 'time_s': float(data.time), 'actual_root_rad': root,
                                              'aperture_mm': aperture, 'pairs': pairs,
                                              'new_pairs_vs_registered_source_pose': [list(x) for x in sorted(now-baseline_pairs)]})
                    if now-baseline_pairs and stop is None:
                        stop = 'NEW_ORIGINAL_STL_TRIANGLE_INTERSECTION'
                rows.append({'step': steps, 'time_s': float(data.time), 'phase': phase, 'phase_name': label,
                             'command_opening': opening_from_root(data.ctrl[0]), 'command_root_rad': float(data.ctrl[0]),
                             'actual_root_rad': root, 'candidate_CAD_coordinate_rad': float(p.PHASE-root),
                             'D_gap_mm': [float(v*1000) for v in p.gaps(model, data)],
                             'mimic_error_rad': mimic, 'max_proxy_penetration_mm': pen, 'max_qvel_rad_s': speed,
                             'surface_aperture_mm': aperture, 'surface_sections': sections,
                             'joint_positions': {name: float(data.qpos[adr]) for name, adr in t.p.addresses(model).items()},
                             'B_spring_passive_torque_Nm': spring_torque,
                             'constraint_residuals': data.efc_pos.tolist(), 'proxy_contacts': proxy})
            if steps % 40 == 0 or stop:
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 67), fill=(12, 20, 28))
                draw.text((8, 4), 'UNVERIFIED EXPERIMENTAL B-SPRING MODEL / ONE ROOT MOTOR', font=font, fill=(255, 207, 112))
                draw.text((8, 24), f"t={data.time:.2f}s opening={opening_from_root(data.ctrl[0])*100:.1f}% root={root:+.4f} rad", font=font, fill='white')
                aperture_text = 'n/a' if aperture is None else f'{aperture:.2f}'
                draw.text((8, 44), f"D gap={gap:.3f} mm aperture={aperture_text} mm", font=font, fill='white')
                writer.append_data(np.asarray(frame))
                if steps in (40, 1000, 3000, 6000, 9000, 12000, 16000):
                    frame.save(run/f'frame_{steps:05d}.png')
            if stop:
                break
        phases.append({'phase': phase, 'name': label, 'target_opening': opening_from_root(target),
                       'target_root_rad': target, 'initial_root_rad': start,
                       'final_root_rad': float(data.qpos[rootadr]), 'complete': stop is None})
        if stop is None and abs(float(data.qpos[rootadr])-target) > .02:
            stop = 'ROOT_TARGET_NOT_REACHED_WITHIN_0P02RAD'
            phases[-1]['complete'] = False
        if stop:
            break
    writer.close(); renderer.close()
    put(run, 'source_interference_baseline.json', {'pairs': source_intersections,
                                                    'qualification': 'source-pose triangle intersections are retained as baseline; not assumed harmless'})
    put(run, 'motion_trace.json', rows)
    put(run, 'source_triangle_interference.json', interference_rows)
    with (run/'motion_trace.csv').open('x', newline='') as stream:
        keys = ['step', 'time_s', 'phase', 'phase_name', 'command_opening', 'command_root_rad', 'actual_root_rad',
                'candidate_CAD_coordinate_rad', 'D_gap_mm', 'mimic_error_rad', 'max_proxy_penetration_mm', 'max_qvel_rad_s', 'surface_aperture_mm']
        out = csv.DictWriter(stream, fieldnames=keys); out.writeheader()
        for row in rows:
            out.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in keys})
    with (run/'measured_aperture.csv').open('x', newline='') as stream:
        out = csv.DictWriter(stream, fieldnames=['phase', 'time_s', 'command_opening', 'actual_root_rad', 'aperture_mm']); out.writeheader(); out.writerows(widths)
    aperture_monotonic_by_phase = {}
    for phase in range(len(phases)):
        segment = [x for x in widths if x['phase'] == phase]
        if segment:
            direction = phases[phase]['name']
            delta = [b['aperture_mm']-a['aperture_mm'] for a, b in zip(segment, segment[1:])]
            aperture_monotonic_by_phase[str(phase)] = {'commanded_direction': direction,
                                                       'increases_on_open': all(x >= -0.05 for x in delta) if direction == 'open' else None,
                                                       'decreases_on_close': all(x <= 0.05 for x in delta) if direction == 'close' else None,
                                                       'min_aperture_mm': min(x['aperture_mm'] for x in segment),
                                                       'max_aperture_mm': max(x['aperture_mm'] for x in segment)}
    nonmonotonic = [phase for phase, item in aperture_monotonic_by_phase.items()
                    if (item['commanded_direction'] == 'open' and not item['increases_on_open'])
                    or (item['commanded_direction'] == 'close' and not item['decreases_on_close'])]
    if stop is None and nonmonotonic:
        stop = 'NONMONOTONIC_MEASURED_APERTURE'
    result = {'status': 'EXPERIMENTAL_MUJOCO_PASS' if stop is None and len(phases) == len(ROOT_PHASES) else 'EXPERIMENTAL_MUJOCO_FAIL',
              'stop_reason': stop, 'mujoco_version': mujoco.__version__, 'physics_steps': steps,
              'simulation_time_s': float(data.time), 'gravity_m_s2': model.opt.gravity.tolist(),
              'active_actuators': model.nu, 'active_joint': 'right_claw_joint', 'passive_actuators': 0,
              'passive_qpos_writes_after_initialization': 0, 'system_remaining_dofs_at_source': t.equality_jacobian(model, initial_data(model, law))['remaining_dofs'],
              'root_limits': [-1.0, 0.0], 'commanded_test_interval_rad': [ROOT_OPEN, ROOT_CLOSE],
              'endpoint_margin_rad': abs(ROOT_CLOSE), 'command_schedule': phases, 'max_D_gap_mm': maxgap,
              'max_mimic_error_rad': maxmimic, 'max_proxy_penetration_mm': maxpen,
              'max_qvel_rad_s': maxspeed, 'root_servo': {'kp': ROOT_SERVO_KP, 'kv': ROOT_SERVO_KV,
                                                         'force_limit_Nm': ROOT_SERVO_FORCE_LIMIT_NM},
              'aperture_monotonicity': aperture_monotonic_by_phase,
              'triangle_interference_samples': len(interference_rows),
              'triangle_interference_new_pair_count': sum(len(x['new_pairs_vs_registered_source_pose']) for x in interference_rows),
              'mechanics': 'D-point closures plus one experimental B-axis torsion spring per side; no Candidate B tendon equalities',
              'classification': 'UNVERIFIED_EXPERIMENTAL; not a manufacturer-equivalent mechanism'}
    put(run, 'physics_result.json', result)
    return result


def initial_data(model, law):
    data = mujoco.MjData(model)
    initial_source_state(model, data, law)
    return data


def export_review(source_run: Path):
    source_run = p.pg.validate_output_path(source_run, require_exists=True, expected_kind='directory', environment='wsl').native_path
    source_manifest = json.loads((source_run/'artifact_paths.json').read_text())
    p.pg.verify_artifacts(source_manifest['artifacts'], environment='wsl')
    result = json.loads((source_run/'physics_result.json').read_text())
    location = p.pg.create_unique_output_directory(p.REPO/'reviews/issue-46', prefix='passive-spring-review', environment='wsl')
    out = location.native_path
    files = ('passive_spring_open_close.mp4', 'motion_trace.json', 'measured_aperture.csv',
             'physics_result.json', 'source_triangle_interference.json', 'spring_sensitivity.json',
             'frame_00040.png', 'frame_01000.png', 'frame_03000.png')
    copied = {}
    for filename in files:
        source = source_run/filename
        if not source.is_file() or source.is_symlink():
            continue
        p.pg.write_new_bytes(out/filename, source.read_bytes(), environment='wsl')
        copied[filename] = p.sha(out/filename)
        if copied[filename] != p.sha(source):
            raise AssertionError(f'review copy hash mismatch: {filename}')
    summary = {'status': result['status'], 'stop_reason': result['stop_reason'],
               'source_run': str(source_run), 'root_limits': result['root_limits'],
               'commanded_test_interval_rad': result['commanded_test_interval_rad'],
               'actual_root_rad': result['command_schedule'][-1]['final_root_rad'],
               'physics_steps': result['physics_steps'], 'simulation_time_s': result['simulation_time_s'],
               'measured_limits': {key: result[key] for key in ('max_D_gap_mm', 'max_mimic_error_rad',
                                      'max_proxy_penetration_mm', 'max_qvel_rad_s', 'triangle_interference_new_pair_count')},
               'assumptions': ['B-axis torsion springs are located from STEP; spring ends/calibration are experimental',
                               'No Candidate B tendon equality; root motor only; D closures and official root mimic retained',
                               'The page scrubs a recorded physics run; it does not send commands or run physics']}
    put(out, 'review_summary.json', summary)
    page = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OmniPicker MuJoCo Trace Review</title>
<style>
:root{color-scheme:dark;--bg:#11181d;--panel:#1a252c;--line:#34444d;--text:#e9f0f2;--muted:#a5b3b8;--orange:#f0ad4e;--red:#e56f64;--green:#55c6ad;--blue:#7bc4e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,"Segoe UI",sans-serif}header{padding:16px 22px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0;font-weight:650}.status{font-size:12px;padding:4px 8px;border:1px solid var(--red);color:#ffd4ce;border-radius:4px}main{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:16px;padding:16px;max-width:1500px;margin:auto}.view{min-width:0}.video{width:100%;aspect-ratio:4/3;background:#05090b;display:block;border:1px solid var(--line);border-radius:4px}section{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px;margin-bottom:12px}h2{font-size:14px;margin:0 0 10px;color:#dce6e8}.controls{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center}input[type=range]{width:100%;accent-color:var(--green)}button{background:#26353d;color:var(--text);border:1px solid #52646c;border-radius:4px;padding:7px 10px;cursor:pointer}button:hover{background:#30444c}.clock{font:13px ui-monospace,monospace;color:var(--muted);margin-top:8px}.warning{border-left:3px solid var(--orange);padding-left:10px;color:#f4d4a2}.grid{display:grid;grid-template-columns:1fr auto;gap:5px 10px;font-size:13px}.grid .label{color:var(--muted)}.grid .value{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}.joint-list{display:grid;grid-template-columns:1fr auto;gap:4px 8px;font:12px ui-monospace,monospace}.joint-list span:nth-child(odd){color:var(--muted)}.small{font-size:12px;color:var(--muted)}.range{color:var(--blue)}a{color:var(--blue)}footer{padding:0 20px 18px;max-width:1500px;margin:auto;color:var(--muted);font-size:12px}@media(max-width:900px){main{grid-template-columns:1fr}.video{aspect-ratio:16/10}}
</style></head><body>
<header><h1>OmniPicker · recorded MuJoCo motion</h1><span class="status" id="runStatus">EXPERIMENTAL_MUJOCO_FAIL</span><span class="small">One root motor · B-axis spring hypothesis · gravity on</span></header>
<main><div class="view"><video id="movie" class="video" controls preload="metadata" src="passive_spring_open_close.mp4"></video>
<section><h2>Captured attempt timeline</h2><div class="controls"><button id="back" title="Previous sample">−</button><input id="scrub" type="range" min="0" max="1000" value="0" step="1"><button id="forward" title="Next sample">+</button></div><div class="clock" id="clock">Loading trace…</div></section>
<section class="warning" id="stopText">This page replays recorded MuJoCo physics states. It does not run MuJoCo or issue motor commands.</section></div>
<aside><section><h2>State at selected frame</h2><div class="grid" id="metrics"></div></section><section><h2>Passive joint angles</h2><div class="joint-list" id="joints"></div></section><section><h2>Model and outcome</h2><div class="grid" id="summary"></div></section><section><h2>Evidence</h2><p class="small"><a href="physics_result.json">Physics result</a> · <a href="motion_trace.json">Full trace</a> · <a href="measured_aperture.csv">Surface measurements</a></p><p class="small">The 0–100% readout is the commanded coordinate inside the tested interior root range. It is not a verified manufacturer closed/open range.</p></section></aside></main>
<footer>Official X2 v1.4.0 meshes and inertias are preserved. B spring placement is CAD-located at both B axes; spring-end ownership, preload, stiffness and damping are not vendor-verified. No bottle, grasp, contact-load or lift test was run.</footer>
<script>
const movie=document.querySelector('#movie'),scrub=document.querySelector('#scrub'),clock=document.querySelector('#clock');let trace=[],result={},updating=false;
const fmt=(v,n=3)=>v===null||v===undefined?'NOT MEASURABLE':Number(v).toFixed(n);
function selectTime(t){if(!trace.length)return;const i=trace.reduce((best,r,k)=>Math.abs(r.time_s-t)<Math.abs(trace[best].time_s-t)?k:best,0);const r=trace[i];if(!updating&&Number.isFinite(movie.duration))movie.currentTime=Math.min(t,movie.duration);scrub.value=Math.round(1000*r.time_s/Math.max(result.simulation_time_s,1e-9));clock.textContent=`${r.time_s.toFixed(3)} s / ${result.simulation_time_s.toFixed(3)} s · command ${(r.command_opening*100).toFixed(1)}% · step ${r.step}`;
document.querySelector('#metrics').innerHTML=[['Root angle',fmt(r.actual_root_rad,5)+' rad'],['CAD coordinate',fmt(r.candidate_CAD_coordinate_rad,5)+' rad'],['Inner-face section',fmt(r.surface_aperture_mm,2)+' mm'],['D gaps',r.D_gap_mm.map(x=>fmt(x,4)).join(' / ')+' mm'],['Mimic error',fmt(r.mimic_error_rad,6)+' rad'],['Proxy penetration',fmt(r.max_proxy_penetration_mm,4)+' mm'],['Max joint speed',fmt(r.max_qvel_rad_s,3)+' rad/s']].map(x=>`<span class="label">${x[0]}</span><span class="value">${x[1]}</span>`).join('');
const names=['right_claw_joint','R_hand_narrow2_joint','R_hand_narrow3_joint','R_hand_narrow_loop_joint','R_hand_wide1_joint','R_hand_wide2_joint','R_hand_wide3_joint','R_hand_wide_loop_joint'];document.querySelector('#joints').innerHTML=names.map(n=>`<span>${n}</span><span>${fmt(r.joint_positions[n],5)} rad</span>`).join('');}
Promise.all([fetch('motion_trace.json').then(r=>r.json()),fetch('physics_result.json').then(r=>r.json())]).then(([a,b])=>{trace=a;result=b;document.querySelector('#runStatus').textContent=result.status;document.querySelector('#stopText').textContent=`Stopped at ${result.stop_reason}. The configured inner-face ray became unmeasurable before the requested root endpoint; the full open/close cycle did not pass.`;document.querySelector('#summary').innerHTML=[['Steps',result.physics_steps],['Simulation time',fmt(result.simulation_time_s,3)+' s'],['Remaining system DOF',result.system_remaining_dofs_at_source],['Root command range',result.commanded_test_interval_rad.join(' … ')+' rad'],['New source-triangle pairs',result.triangle_interference_new_pair_count],['Requested endpoint reached',Math.abs(result.command_schedule.at(-1).target_root_rad-result.command_schedule.at(-1).final_root_rad)<0.02?'yes':'no']].map(x=>`<span class="label">${x[0]}</span><span class="value">${x[1]}</span>`).join('');selectTime(0);}).catch(e=>{clock.textContent='Trace load failed: '+e});
scrub.oninput=()=>{if(!trace.length)return;selectTime(Number(scrub.value)/1000*result.simulation_time_s)};movie.addEventListener('timeupdate',()=>{if(!trace.length||updating)return;updating=true;selectTime(movie.currentTime);updating=false});document.querySelector('#back').onclick=()=>{movie.currentTime=Math.max(0,movie.currentTime-.1)};document.querySelector('#forward').onclick=()=>{movie.currentTime=Math.min(movie.duration,movie.currentTime+.1)};
</script></body></html>'''
    put(out, 'review.html', page)
    p.pg.write_artifact_manifest(out, environment='wsl')
    return {'review_directory': str(out), 'source_run': str(source_run), 'status': result['status'],
            'stop_reason': result['stop_reason'], 'copied_files_sha256': copied}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path)
    ap.add_argument('--stage', choices=('run', 'review-export'), default='run')
    ap.add_argument('--source-run', type=Path)
    ap.add_argument('--spring-stiffness', type=float, default=.05)
    ap.add_argument('--spring-reference', type=float, default=0.0)
    ap.add_argument('--spring-damping', type=float, default=SPRING_DAMPING)
    args = ap.parse_args()
    if args.stage == 'review-export':
        if args.source_run is None:
            raise ValueError('--source-run is required for review-export')
        print(json.dumps(export_review(args.source_run), indent=2))
        return
    if args.run_dir:
        location = p.pg.validate_output_path(args.run_dir, require_exists=True, expected_kind='directory', environment='wsl')
        p.pg._require_reserved_directory(location)
    else:
        location = p.pg.create_unique_output_directory(p.REPO/'reviews/issue-46', prefix='passive-spring-single-motor', environment='wsl')
    run = location.native_path
    model, derived, provenance = make_model(run, args.spring_stiffness, args.spring_reference, args.spring_damping)
    law = p.CandidateLaw()
    sensitivity = spring_sensitivity(run, derived, law)
    put(run, 'sensitivity_nominal_not_fitted.json', {'nominal': {'stiffness_Nm_per_rad': args.spring_stiffness,
                                                                   'spring_reference_rad': args.spring_reference,
                                                                   'damping_Nm_s_per_rad': args.spring_damping},
                                                         'parameter_grid': [{'k': r['stiffness_Nm_per_rad'], 'ref': r['springref_rad'], 'status': r['status'], 'stop_reason': r['stop_reason']}
                                                                            for r in sensitivity]})
    result = run_cycle(run, model, derived, law)
    report = f"""# Issue #46 — Passive B-Spring Single-Motor Candidate

Result: `{result['status']}`

Classification: `UNVERIFIED_EXPERIMENTAL`

MuJoCo: `{result['mujoco_version']}`

Run directory: `{run}`

## Model

This candidate retains the pinned official X2 v1.4.0 URDF meshes/inertias, reviewed measured pivot and D-point frames, two D-point connects, official opposite-root mimic, and unchanged root limits. It has one active root motor, zero follower actuators, and no Candidate B fixed-tendon equalities. Two experimental torsion springs act at the CAD-located narrow/wide B axes. The CAD occurrence tree locates spring product `220000054701` coaxially at both B pivots. It does not provide spring-end ownership, handedness, stiffness, preload, damping, or a transmission law; all spring values in this candidate are diagnostic assumptions.

The single input maps normalized `opening=0..1` linearly to the interior command range `root=-0.01..-0.99 rad`. This leaves a 0.01 rad numerical margin inside the unchanged official limits `[-1,0]`; it is a controller-safe test interval, not a new mechanical limit or certified closed/open travel. The run starts at the registered STEP source pose. It does not initialize at a claimed mechanically closed pose.

## Actual physics result

Steps: `{result['physics_steps']}` over `{result['simulation_time_s']:.3f} s`; gravity was on. The run stopped with `{result['stop_reason'] or 'no safety stop'}`. Max D gap `{result['max_D_gap_mm']:.6f} mm`; max root mimic error `{result['max_mimic_error_rad']:.6g} rad`; max collision-proxy penetration `{result['max_proxy_penetration_mm']:.6f} mm`; max joint speed `{result['max_qvel_rad_s']:.6f} rad/s`. New original-triangle intersection pairs: `{result['triangle_interference_new_pair_count']}` across `{result['triangle_interference_samples']}` sampled states.

The sequence and actual jaw-surface section widths are recorded in `motion_trace.json` and `measured_aperture.csv`. The aperture samples are original visual-STL inner faces at a fixed section, not a certified gripper diameter. Source-pose intersection pairs remain recorded and were not silently waived.

## Parameter sensitivity

`spring_sensitivity.json` tests stiffness `{list(SPRING_STIFFNESS)} Nm/rad` and reference-angle offsets `{list(SPRING_REFERENCE)} rad`, with damping `{SPRING_DAMPING} Nm*s/rad`. The grid is not fitted to make the cycle pass; the main candidate uses the grid-center values unless overridden. The root servo uses `kp={ROOT_SERVO_KP}`, `kv={ROOT_SERVO_KV}`, and the unchanged `±{ROOT_SERVO_FORCE_LIMIT_NM} Nm` cap. The gain adjustment follows measured source-pose root gravity bias (`0.003224 Nm`) versus the old proportional torque (`0.005 Nm` at 0.01 rad error); it remains a simulation control assumption, not vendor calibration.

## Evidence

- `passive_spring.xml`: derived MuJoCo model, no fixed-tendon relation.
- `X2-Ultra_omnipicker_PASSIVE_SPRING_EXPERIMENTAL.urdf`, `official_to_derived.diff`: source overlay and exact diff.
- `passive_spring_open_close.mp4`: rendered actual physics states.
- `motion_trace.csv/json`, `measured_aperture.csv`, `source_triangle_interference.json`, `source_interference_baseline.json`, `spring_sensitivity.json`, `physics_result.json`.
- `provenance.json`, `artifact_paths.json`: pinned input and verified evidence hashes.

## Limitations

`EXPERIMENTAL_MUJOCO_PASS` would mean only that this explicitly experimental candidate completed the tested command cycle under its assumptions. It would not prove the source mechanism, vendor full travel, or hardware behavior. A failure identifies the first observed physical/safety gate. No bottle, grasp, contact-load, or lift task is included.
"""
    put(run, 'REPORT.md', report)
    manifest = p.pg.write_artifact_manifest(run, environment='wsl')
    print(json.dumps({'run': str(run), 'physics': result,
                      'artifact_count': len(manifest['artifacts']), 'sensitivity_cases': len(sensitivity)}, indent=2))


if __name__ == '__main__':
    main()
