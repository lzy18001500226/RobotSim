#!/usr/bin/env python3
"""Static X2/G2/experimental OmniPicker aperture comparison and viewer.

All poses are assigned to qpos and evaluated with mj_forward(). No mj_step().
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

X2_ROOT = Path('/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf')
G2_ROOT = Path('/home/lzy18001500226/.cache/robotsim/upstreams/genie_sim_robot_model')
X2_URDF = X2_ROOT / 'X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf'
G2_XACRO = G2_ROOT / 'robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro'
COMPARISON_MODEL = Path('/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-g2-omnipicker-compatibility-20261013/x2_g2_omnipicker_review.xml')
PRIOR_METRICS = Path('/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-variant-b-full-x2-geometry-feasibility-20261012/sourcefaithful/linkage_comparison/comparison_static_metrics.json')
OUT_DEFAULT = Path('/home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-joint-aperture-20261013-run01')

MODELS = ('X2 official URDF', 'G2 official Xacro', 'X2 SIMULATION_ONLY candidate')
ROOTS = {
    MODELS[0]: ('x2src_right_claw_joint', 'x2src_R_hand_wide1_joint'),
    MODELS[1]: ('g2_inner_joint1', 'g2_outer_joint1'),
    MODELS[2]: ('x2cand_candidate_right_claw_joint', 'x2cand_candidate_R_hand_wide1_joint'),
}
BASES = {MODELS[0]: 'x2src_assembly_base', MODELS[1]: 'g2_base', MODELS[2]: 'x2cand_assembly_base'}
CONTACT_PREFIX = {MODELS[0]: 'x2src_', MODELS[1]: 'g2_', MODELS[2]: 'x2cand_'}
PADS = {
    MODELS[0]: ('x2src_R_hand_narrow3_Link', 'x2src_R_hand_wide3_Link'),
    MODELS[1]: ('g2_inner_link4', 'g2_outer_link4'),
    MODELS[2]: ('x2cand_candidate_R_hand_narrow3_Link', 'x2cand_candidate_R_hand_wide3_Link'),
}
G2_FOLLOWERS = {
    'g2_inner_joint3': .1, 'g2_inner_joint4': .25, 'g2_inner_joint0': -.7,
    'g2_outer_joint3': .1, 'g2_outer_joint4': -.25, 'g2_outer_joint0': .7,
}
CAND_FOLLOWERS = {
    'inner': ('x2cand_candidate_R_hand_narrow2_passive_hinge',
              'x2cand_candidate_R_hand_narrow3_passive_hinge',
              'x2cand_candidate_R_hand_narrow_loop_passive_hinge'),
    'outer': ('x2cand_candidate_R_hand_wide2_passive_hinge',
              'x2cand_candidate_R_hand_wide3_passive_hinge',
              'x2cand_candidate_R_hand_wide_loop_passive_hinge'),
}
CAND_PINS = {
    'inner': ('x2cand_candidate_R_hand_narrow3_Link', (-.0000657836, -.0104436701, -.0001030215),
              'x2cand_candidate_R_hand_narrow_loop_Link', (.0315228467, -.0097562996, -.0002187793)),
    'outer': ('x2cand_candidate_R_hand_wide3_Link', (-.0001688067, .0103357106, .0007564969),
              'x2cand_candidate_R_hand_wide_loop_Link', (.0315133385, .0096536578, .0009045486)),
}
G2_LOOPS = {
    'inner': ('g2_base', (0., -.012, .105), 'g2_inner_link2'),
    'outer': ('g2_base', (0., .012, .105), 'g2_outer_link2'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def oid(m, kind, name):
    i = int(mujoco.mj_name2id(m, kind, name))
    if i < 0:
        raise KeyError(f'Model object missing: {name}')
    return i


def qadr(m, joint):
    return int(m.jnt_qposadr[oid(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def setq(m, d, joint, angle):
    adr = qadr(m, joint)
    d.qpos[adr] = float(m.qpos0[adr] + angle)


def point(m, d, body, local):
    i = oid(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return d.xpos[i] + d.xmat[i].reshape(3, 3) @ np.asarray(local, dtype=float)


def collision_geoms(m, body):
    bid = oid(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return [int(g) for g in np.flatnonzero(m.geom_bodyid == bid)
            if m.geom_contype[g] or m.geom_conaffinity[g]]


def geom_dist(m, d, ga, gb):
    pts = np.zeros(6)
    dist = float(mujoco.mj_geomDistance(m, d, ga, gb, 1., pts))
    witness_length = float(np.linalg.norm(pts[:3] - pts[3:]))
    return {'gap_mm': dist * 1000., 'points': [pts[:3].tolist(), pts[3:].tolist()],
            'witness_segment_mm': witness_length * 1000.,
            'witness_distance_discrepancy_mm': abs(witness_length - abs(dist)) * 1000.,
            'geoms': [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, ga),
                      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gb)],
            'geom_ids': [ga, gb]}


def pad_metrics(m, d, name):
    ba, bb = PADS[name]
    ag, bg = collision_geoms(m, ba), collision_geoms(m, bb)
    if not ag or not bg:
        raise RuntimeError(f'Missing terminal collision geoms: {ba}, {bb}')
    pair_rows = [geom_dist(m, d, a, b) for a in ag for b in bg]
    nearest = min(pair_rows, key=lambda r: r['gap_mm'])
    g0, g1 = nearest['geom_ids']
    p0, p1 = (np.asarray(p) for p in nearest['points'])
    line = p1 - p0
    norm = np.linalg.norm(line)
    if norm < 1e-9:
        line = d.geom_xpos[g1] - d.geom_xpos[g0]
        norm = np.linalg.norm(line)
    line = line / norm if norm > 1e-12 else np.zeros(3)
    r0, r1 = d.geom_xmat[g0].reshape(3, 3), d.geom_xmat[g1].reshape(3, 3)
    relative = r0.T @ r1
    frame_angle = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    base = oid(m, mujoco.mjtObj.mjOBJ_BODY, BASES[name])
    rb = d.xmat[base].reshape(3, 3)
    pinner = rb.T @ (d.geom_xpos[g0] - d.xpos[base])
    pouter = rb.T @ (d.geom_xpos[g1] - d.xpos[base])
    reflection = np.linalg.norm(pinner - [pouter[0], -pouter[1], pouter[2]]) * 1000
    ji, jo = (oid(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ROOTS[name])
    return {
        'aperture_mm': nearest['gap_mm'],
        'clearance_over_70mm_mm': nearest['gap_mm'] - 70.,
        'surface_points_world_m': nearest['points'],
        'witness_segment_mm': nearest['witness_segment_mm'],
        'witness_distance_discrepancy_mm': nearest['witness_distance_discrepancy_mm'],
        'distance_witness_consistent': nearest['witness_distance_discrepancy_mm'] <= 1.0,
        'nearest_collision_geom_pair': nearest['geoms'],
        'nearest_geom_collision_masks': [
            {'contype': int(m.geom_contype[g]), 'conaffinity': int(m.geom_conaffinity[g])}
            for g in nearest['geom_ids']],
        'geom_pair_count_checked': len(pair_rows),
        'nearest_surface_line_world': line.tolist(),
        'terminal_collision_geom_frame_angle_deg': frame_angle,
        'root_axes_world_dot': float(np.clip(np.dot(d.xaxis[ji], d.xaxis[jo]), -1, 1)),
        'carrier_reflection_error_base_frame_mm': float(reflection),
        'definition': 'Minimum signed distance over all collision-enabled geoms on the two terminal carrier bodies; compiled MuJoCo mesh hulls, not a labeled pad-face datum.',
    }


def collision_contacts(m, d):
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        ga, gb = int(c.geom1), int(c.geom2)
        ba, bb = int(m.geom_bodyid[ga]), int(m.geom_bodyid[gb])
        rows.append({'geom1': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, ga),
                     'geom2': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gb),
                     'body1': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ba),
                     'body2': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb),
                     'distance_mm': float(c.dist * 1000.)})
    return rows


def contacts_for_model(row, name):
    prefix = CONTACT_PREFIX[name]
    return [c for c in row['static_contact_candidates']
            if c['body1'].startswith(prefix) and c['body2'].startswith(prefix)]


def apply_pose(m, d, name, u, previous):
    u = float(np.clip(u, 0., 1.))
    qx = -u
    if name == MODELS[1]:
        driver = qx * math.pi / 4
        roots = [driver, -driver]
        setq(m, d, ROOTS[name][0], roots[0])
        setq(m, d, ROOTS[name][1], roots[1])
        followers = {joint: scale * driver for joint, scale in G2_FOLLOWERS.items()}
        for joint, angle in followers.items():
            setq(m, d, joint, angle)
        mujoco.mj_forward(m, d)
        loop_residual = {}
        for side, (base, anchor, terminal) in G2_LOOPS.items():
            loop_residual[side] = float(np.linalg.norm(point(m, d, base, anchor) - point(m, d, terminal, (0, 0, 0))) * 1000)
        return {'roots_rad': roots, 'passive_rad': followers,
                'loop_residual_mm': loop_residual, 'max_loop_residual_mm': max(loop_residual.values()),
                'mapping': 'q_inner=-u*pi/4; outer=-inner; exact pinned Xacro mimic ratios'}
    roots = [qx, -qx]
    for joint, angle in zip(ROOTS[name], roots):
        setq(m, d, joint, angle)
    mujoco.mj_forward(m, d)
    if name == MODELS[0]:
        return {'roots_rad': roots, 'passive_rad': {}, 'mapping': 'q_narrow=-u; q_wide=+u; source mimic -1; secondary links fixed by URDF'}
    passive = {}
    closures = {}
    for side, joints in CAND_FOLLOWERS.items():
        body_a, local_a, body_b, local_b = CAND_PINS[side]
        start = previous[side].copy()

        def residual(angles):
            for joint, angle in zip(joints, angles):
                setq(m, d, joint, float(angle))
            mujoco.mj_forward(m, d)
            delta_mm = (point(m, d, body_a, local_a) - point(m, d, body_b, local_b)) * 1000
            return np.r_[delta_mm, (angles - start) * 1e-3]

        sol = least_squares(residual, start, bounds=(-2, 2), max_nfev=120,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        for joint, angle in zip(joints, sol.x):
            setq(m, d, joint, float(angle))
            passive[joint] = float(angle)
        mujoco.mj_forward(m, d)
        error = float(np.linalg.norm((point(m, d, body_a, local_a) - point(m, d, body_b, local_b)) * 1000))
        closures[side] = {'pin_center_separation_mm': error, 'optimizer_success': bool(sol.success),
                          'status': int(sol.status), 'evaluations': int(sol.nfev)}
        previous[side] = sol.x.copy()
    return {'roots_rad': roots, 'passive_rad': passive, 'closure_by_side': closures,
            'max_closure_mm': max(v['pin_center_separation_mm'] for v in closures.values()),
            'mapping': 'same root relation as X2; six passive hinges solved from prior fitted pin centers with continuation'}


def joint_state(m, d, name):
    if name == MODELS[0]:
        names = list(ROOTS[name])
    elif name == MODELS[1]:
        names = list(ROOTS[name]) + list(G2_FOLLOWERS)
    else:
        names = list(ROOTS[name]) + [j for side in ('inner', 'outer') for j in CAND_FOLLOWERS[side]]
    result = []
    for joint in names:
        jid = oid(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        adr = int(m.jnt_qposadr[jid])
        raw = float(d.qpos[adr])
        limited = bool(m.jnt_limited[jid])
        lo, hi = map(float, m.jnt_range[jid]) if limited else (None, None)
        result.append({'name': joint, 'raw_qpos_rad': raw, 'compiled_qpos0_rad': float(m.qpos0[adr]),
                       'reference_coordinate_note': 'MuJoCo 3.3.6 compiled MjModel exposes qpos0; joint ref is folded into this compiled initial coordinate.',
                       'limited': limited,
                       'range_rad': [lo, hi] if limited else None,
                       'within_compiled_range': (not limited) or lo - 1e-10 <= raw <= hi + 1e-10,
                       'limit_margin_rad': min(raw - lo, hi - raw) if limited else None})
    unbounded = [x['name'] for x in result if not x['limited']]
    return {'joints': result,
            'all_within_compiled_ranges': all(x['within_compiled_range'] for x in result),
            'all_limited_joint_ranges_pass': all(x['within_compiled_range'] for x in result if x['limited']),
            'unbounded_joint_names': unbounded}


def loop_summary(name, pose):
    if name == MODELS[0]:
        return {'status': 'NO_ARTICULATED_LOOP_IN_PINNED_URDF', 'max_residual_mm': None}
    if name == MODELS[1]:
        return {'status': 'SOURCE_XACRO_SPHERICAL_ANCHOR_COINCIDENCE_DIAGNOSTIC',
                'max_residual_mm': pose['max_loop_residual_mm'], 'by_side_mm': pose['loop_residual_mm'],
                'caveat': 'Custom loop tags, not standard URDF joints; no source tolerance or physical loop implementation specified.'}
    return {'status': 'SIMULATION_ONLY_MESH_FITTED_PIN_CLOSURE',
            'max_residual_mm': pose['max_closure_mm'],
            'by_side_mm': {k: v['pin_center_separation_mm'] for k, v in pose['closure_by_side'].items()},
            'caveat': 'STL-fit pin centers are estimates, not CAD datums; candidate solver is a view branch, not a physical constraint simulation.'}


def loop_summary_current(m, d, name):
    if name == MODELS[0]:
        return {'status': 'NO_ARTICULATED_LOOP_IN_PINNED_URDF', 'max_residual_mm': None}
    if name == MODELS[1]:
        values = {}
        for side, (base, anchor, terminal) in G2_LOOPS.items():
            values[side] = float(np.linalg.norm(point(m, d, base, anchor) - point(m, d, terminal, (0, 0, 0))) * 1000)
        return {'status': 'SOURCE_XACRO_SPHERICAL_ANCHOR_COINCIDENCE_DIAGNOSTIC',
                'max_residual_mm': max(values.values()), 'by_side_mm': values}
    values = {}
    for side, (body_a, local_a, body_b, local_b) in CAND_PINS.items():
        values[side] = float(np.linalg.norm(point(m, d, body_a, local_a) - point(m, d, body_b, local_b)) * 1000)
    return {'status': 'SIMULATION_ONLY_MESH_FITTED_PIN_CLOSURE',
            'max_residual_mm': max(values.values()), 'by_side_mm': values}


def scan(m, values):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    previous = {'inner': np.zeros(3), 'outer': np.zeros(3)}
    rows = []
    for u in values:
        poses = {name: apply_pose(m, d, name, float(u), previous) for name in MODELS}
        mujoco.mj_forward(m, d)
        rows.append({'u': float(u), 'poses': poses,
                     'apertures': {name: pad_metrics(m, d, name) for name in MODELS},
                     'joint_state': {name: joint_state(m, d, name) for name in MODELS},
                     'static_contact_candidates': collision_contacts(m, d), 'ncon': int(d.ncon),
                     'physics_steps': 0})
    return rows


def add_slider_xml(src, dest):
    tree = ET.parse(src)
    root = tree.getroot()
    old = root.find('actuator')
    if old is not None:
        root.remove(old)
    wb = root.find('worldbody')
    wb.append(ET.fromstring(
        '<body name="ui_open_fraction_body" pos="0 0 -10">'
        '<joint name="ui_open_fraction_joint" type="slide" axis="0 0 1" range="0 1" limited="true" damping="1"/>'
        '<inertial pos="0 0 0" mass="0.001" diaginertia="0.000001 0.000001 0.000001"/>'
        '</body>'))
    act = ET.SubElement(root, 'actuator')
    ET.SubElement(act, 'position', {'name': 'Opening fraction (0 closed, 1 open)',
                                    'joint': 'ui_open_fraction_joint', 'kp': '1',
                                    'ctrllimited': 'true', 'ctrlrange': '0 1'})
    tree.write(dest, encoding='utf-8', xml_declaration=True)


def plot(rows, dest):
    width, height = 1280, 760
    left, top, right, bottom = 112, 112, 1210, 650
    colors = {MODELS[0]: (36, 104, 166), MODELS[1]: (212, 122, 31), MODELS[2]: (22, 139, 114)}
    values = [r['apertures'][name]['aperture_mm'] for r in rows for name in MODELS
              if r['apertures'][name]['distance_witness_consistent']]
    ymin = 0
    ymax = max(math.ceil(max(max(values), 70) / 10) * 10, ymin + 10)
    im = Image.new('RGB', (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(im)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    bold_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    regular = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    bold = ImageFont.truetype(str(bold_path), 22) if bold_path.exists() else regular
    draw.text((left, 18), 'OmniPicker compiled collision-carrier gap', fill=(25, 36, 48), font=bold)
    draw.text((left, 52), 'Samples with >1 mm distance/witness mismatch are omitted; this is not a labeled pad-face measurement.',
              fill=(75, 83, 91), font=regular)
    def X(u):
        return left + u * (right - left)
    def Y(gap):
        return bottom - (gap - ymin) / (ymax - ymin) * (bottom - top)
    for tick in range(int(ymin), int(ymax) + 1, 10):
        y = Y(tick)
        draw.line((left, y, right, y), fill=(222, 227, 231), width=1)
        draw.text((54, y - 9), str(tick), fill=(65, 76, 87), font=regular)
    for i in range(11):
        u = i / 10
        x = X(u)
        draw.line((x, top, x, bottom), fill=(234, 237, 240), width=1)
        draw.text((x - 12, bottom + 10), f'{u:.1f}', fill=(65, 76, 87), font=regular)
    draw.rectangle((left, top, right, bottom), outline=(60, 69, 78), width=2)
    for threshold, color in ((70, (70, 70, 70)),):
        if ymin <= threshold <= ymax:
            y = Y(threshold)
            for x in range(left, right, 16):
                draw.line((x, y, min(x + 9, right), y), fill=color, width=2)
    for name in MODELS:
        segment = []
        for row in rows:
            sample = row['apertures'][name]
            if sample['distance_witness_consistent']:
                segment.append((X(row['u']), Y(sample['aperture_mm'])))
            else:
                if len(segment) >= 2:
                    draw.line(segment, fill=colors[name], width=4)
                segment = []
        if len(segment) >= 2:
            draw.line(segment, fill=colors[name], width=4)
    draw.text((435, 704), 'Normalized opening fraction u (0 closed, 1 open)', fill=(35, 45, 55), font=regular)
    label = Image.new('RGBA', (height, 34), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((0, 0), 'Terminal collision-carrier gap (mm)',
                               fill=(35, 45, 55), font=regular)
    label = label.rotate(90, expand=True)
    im.paste(label, (0, (height - label.height) // 2), label)
    legend = [('Official X2', colors[MODELS[0]]), ('Official G2', colors[MODELS[1]]),
              ('X2 candidate (SIM ONLY)', colors[MODELS[2]]), ('70 mm bottle', (70, 70, 70))]
    positions = (125, 300, 475, 800)
    for x, (text, color) in zip(positions, legend):
        draw.line((x, 91, x + 28, 91), fill=color, width=4)
        draw.text((x + 35, 81), text, fill=(40, 48, 57), font=regular)
    im.save(dest)


def output_csv(rows, dest):
    fields = ['u', 'model', 'root_inner_rad', 'root_outer_rad', 'aperture_mm',
              'clearance_over_70mm_mm', 'distance_witness_consistent',
              'witness_distance_discrepancy_mm', 'pad_frame_angle_deg', 'mirror_error_mm',
              'passive_joint_angles_rad_json', 'unbounded_joint_names_json',
              'max_linkage_residual_mm', 'static_contact_count',
              'all_limited_joint_ranges_pass']
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for name in MODELS:
                pose, pad = row['poses'][name], row['apertures'][name]
                loop = loop_summary(name, pose)
                writer.writerow({'u': row['u'], 'model': name,
                                 'root_inner_rad': pose['roots_rad'][0], 'root_outer_rad': pose['roots_rad'][1],
                                 'aperture_mm': pad['aperture_mm'],
                                 'clearance_over_70mm_mm': pad['clearance_over_70mm_mm'],
                                 'distance_witness_consistent': pad['distance_witness_consistent'],
                                 'witness_distance_discrepancy_mm': pad['witness_distance_discrepancy_mm'],
                                 'pad_frame_angle_deg': pad['terminal_collision_geom_frame_angle_deg'],
                                 'mirror_error_mm': pad['carrier_reflection_error_base_frame_mm'],
                                 'passive_joint_angles_rad_json': json.dumps(
                                     pose['passive_rad'], sort_keys=True, separators=(',', ':')),
                                 'unbounded_joint_names_json': json.dumps(
                                     row['joint_state'][name]['unbounded_joint_names'], separators=(',', ':')),
                                 'max_linkage_residual_mm': loop['max_residual_mm'],
                                 'static_contact_count': row['ncon'],
                                 'all_limited_joint_ranges_pass': row['joint_state'][name]['all_limited_joint_ranges_pass']})


def render_stills(m, rows, out):
    paths = []
    for index, label in ((0, 'closed'), (50, 'mid'), (100, 'open')):
        row = rows[index]
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for name in MODELS:
            for joint in row['joint_state'][name]['joints']:
                d.qpos[qadr(m, joint['name'])] = joint['raw_qpos_rad']
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, .15]
        cam.distance, cam.azimuth, cam.elevation = .66, 75, 12
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[0] = 1
        opt.geomgroup[1] = 1
        opt.sitegroup[2] = 1
        renderer = mujoco.Renderer(m, height=900, width=2100)
        renderer.update_scene(d, cam, scene_option=opt)
        image = Image.fromarray(renderer.render())
        renderer.close()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 2100, 56), fill=(23, 31, 39))
        draw.text((18, 8), 'LEFT: official X2 URDF | CENTER: official G2 Xacro mimic | RIGHT: prior X2 SIMULATION_ONLY candidate', fill='white')
        draw.text((18, 32), f'u={row["u"]:.2f} (0 closed / 1 open) | mj_forward only | no physics steps', fill=(190, 220, 228))
        path = out / f'comparison_{label}.png'
        image.save(path)
        paths.append(str(path))
    return paths


def interpolate_sample(rows, u):
    pos = float(np.clip(u, 0, 1)) * (len(rows) - 1)
    i0, i1 = int(math.floor(pos)), int(math.ceil(pos))
    alpha = pos - i0
    q = {}
    for name in MODELS:
        for joint in rows[i0]['joint_state'][name]['joints']:
            n = joint['name']
            q[n] = joint['raw_qpos_rad'] * (1 - alpha) + rows[i1]['joint_state'][name]['joints'][
                next(k for k, item in enumerate(rows[i1]['joint_state'][name]['joints']) if item['name'] == n)]['raw_qpos_rad'] * alpha
    return q


def interactive(m, rows, viewer_xml):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    actuator = oid(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'Opening fraction (0 closed, 1 open)')
    ui_joint = oid(m, mujoco.mjtObj.mjOBJ_JOINT, 'ui_open_fraction_joint')
    ui_adr = int(m.jnt_qposadr[ui_joint])
    d.ctrl[actuator] = 1.0
    print(f'VIEWER_READY model={viewer_xml} sha256={sha256(viewer_xml)} MuJoCo={mujoco.__version__} physics_steps=0 mj_step_calls=0', flush=True)
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=True) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0, 0, .15]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = .66, 75, 12
        last_u = None
        while v.is_running():
            with v.lock():
                u = float(np.clip(d.ctrl[actuator], 0, 1))
                if last_u is None or abs(last_u - u) > 1e-5:
                    values = interpolate_sample(rows, u)
                    for joint, value in values.items():
                        d.qpos[qadr(m, joint)] = value
                    d.qpos[ui_adr] = u
                    mujoco.mj_forward(m, d)
                    last_u = u
                    texts = []
                    positions = (mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                 mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)
                    headers = (
                        'LEFT | official pinned X2 URDF',
                        'CENTER | official G2 Xacro + source mimic',
                        'RIGHT | prior X2 SIMULATION_ONLY candidate',
                    )
                    for name, gridpos, header in zip(MODELS, positions, headers):
                        roots = [float(d.qpos[qadr(m, j)]) for j in ROOTS[name]]
                        if name == MODELS[0]:
                            extra_joints = []
                            followers = 'fixed in source URDF'
                        elif name == MODELS[1]:
                            extra_joints = list(G2_FOLLOWERS)
                            followers = ', '.join(
                                f'{j.replace("g2_", "")}={float(d.qpos[qadr(m, j)]):+.3f}'
                                for j in extra_joints)
                        else:
                            extra_joints = [j for side in ('inner', 'outer') for j in CAND_FOLLOWERS[side]]
                            followers = ', '.join(
                                f'{j.split("candidate_")[-1]}={float(d.qpos[qadr(m, j)]):+.3f}'
                                for j in extra_joints)
                        pad = pad_metrics(m, d, name)
                        loop = loop_summary_current(m, d, name)
                        closure = loop['max_residual_mm']
                        states = joint_state(m, d, name)['joints']
                        limit_pass = all(j['within_compiled_range'] for j in states if j['limited'])
                        unbounded_count = sum(not j['limited'] for j in states)
                        if closure is None:
                            closure_text = 'loop: N/A (no articulated loop)'
                        elif name == MODELS[1]:
                            closure_text = f'Xacro anchor residual: {closure:.3f} mm (no tolerance)'
                        else:
                            closure_text = f'fitted pin residual: {closure:.3f} mm (SIM_ONLY)'
                        if pad['distance_witness_consistent']:
                            gap_text = (f'pad-carrier gap={pad["aperture_mm"]:.2f} mm | '
                                        f'margin vs 70 mm={pad["clearance_over_70mm_mm"]:+.2f} mm')
                            witness = 'witness consistent'
                        else:
                            gap_text = (f'gap uncertain: mjGeomDistance={pad["aperture_mm"]:.2f} mm, '
                                        f'witness={pad["witness_segment_mm"]:.2f} mm')
                            witness = 'distance/witness mismatch'
                        first = f'{header}\nu={u:.3f} (0 closed / 1 open) | root q={roots[0]:+.3f}, {roots[1]:+.3f} rad'
                        second = (
                            f'{gap_text}\n'
                            f'passive q: {followers}\n{closure_text}\n'
                            f'pad-frame delta={pad["terminal_collision_geom_frame_angle_deg"]:.1f} deg; '
                            f'pair-center mirror error={pad["carrier_reflection_error_base_frame_mm"]:.2f} mm\n'
                            f'limited joints: {"PASS" if limit_pass else "FAIL"}; '
                            f'unbounded coords: {unbounded_count} | {witness}'
                        )
                        texts.append((mujoco.mjtFontScale.mjFONTSCALE_100, gridpos, first, second))
                    v.set_texts(texts)
                v.sync()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=COMPARISON_MODEL)
    ap.add_argument('--evidence-dir', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--interactive', action='store_true')
    args = ap.parse_args()
    for path in (args.model, X2_URDF, G2_XACRO, PRIOR_METRICS):
        if not path.is_file():
            raise SystemExit(f'Required pinned input absent; refusing substitution: {path}')
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    viewer_xml = args.evidence_dir / 'omnipicker_aperture_viewer.xml'
    add_slider_xml(args.model, viewer_xml)
    m = mujoco.MjModel.from_xml_path(str(viewer_xml))
    rows = scan(m, np.linspace(0, 1, 101))
    pngs = render_stills(m, rows, args.evidence_dir)
    plot_path = args.evidence_dir / 'aperture_vs_opening_fraction.png'
    plot(rows, plot_path)
    csv_path = args.evidence_dir / 'aperture_scan.csv'
    output_csv(rows, csv_path)
    summary = {}
    for name in MODELS:
        gaps = [r['apertures'][name]['aperture_mm'] for r in rows]
        reliable_gaps = [r['apertures'][name]['aperture_mm'] for r in rows
                         if r['apertures'][name]['distance_witness_consistent']]
        closures = [loop_summary(name, r['poses'][name])['max_residual_mm'] for r in rows]
        closures = [x for x in closures if x is not None]
        contact_rows = [c for r in rows for c in contacts_for_model(r, name)]
        pair_counts = {}
        for contact in contact_rows:
            pair = ' <-> '.join(sorted((contact['geom1'], contact['geom2'])))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        worst_contact = min(contact_rows, key=lambda c: c['distance_mm']) if contact_rows else None
        witness_anomalies = [r for r in rows
                             if r['apertures'][name]['witness_distance_discrepancy_mm'] > 1.0]
        nonmonotonic_steps = sum(gaps[i] < gaps[i - 1] - 0.05 for i in range(1, len(gaps)))
        reliable_adjacent = [i for i in range(1, len(rows))
                             if rows[i]['apertures'][name]['distance_witness_consistent']
                             and rows[i - 1]['apertures'][name]['distance_witness_consistent']]
        reliable_nonmonotonic_steps = sum(
            rows[i]['apertures'][name]['aperture_mm'] < rows[i - 1]['apertures'][name]['aperture_mm'] - .05
            for i in reliable_adjacent)
        joint_names = [j['name'] for j in rows[0]['joint_state'][name]['joints']]
        joint_ranges = {
            joint: [min(r['joint_state'][name]['joints'][k]['raw_qpos_rad']
                        for r in rows
                        for k, item in enumerate(r['joint_state'][name]['joints'])
                        if item['name'] == joint),
                    max(r['joint_state'][name]['joints'][k]['raw_qpos_rad']
                        for r in rows
                        for k, item in enumerate(r['joint_state'][name]['joints'])
                        if item['name'] == joint)]
            for joint in joint_names
        }
        optimizer_failures = {}
        if name == MODELS[2]:
            for side in ('inner', 'outer'):
                optimizer_failures[side] = sum(
                    not r['poses'][name]['closure_by_side'][side]['optimizer_success']
                    for r in rows)
        summary[name] = {
            'closed_mm': gaps[0], 'half_open_mm': gaps[50], 'open_mm': gaps[-1],
            'at_q_minus_0_07_mm': gaps[7],
            'closed_root_rad': rows[0]['poses'][name]['roots_rad'],
            'open_root_rad': rows[-1]['poses'][name]['roots_rad'],
            'open_clearance_over_70mm_mm': gaps[-1] - 70,
            'min_gap_mm_witness_consistent_samples': min(reliable_gaps),
            'max_gap_mm_witness_consistent_samples': max(reliable_gaps),
            'min_mj_geomDistance_reported_mm_including_flagged': min(gaps),
            'reliable_aperture_sample_count': len(reliable_gaps),
            'max_linkage_closure_residual_mm': max(closures) if closures else None,
            'measured_joint_qpos_min_max_rad_over_scan': joint_ranges,
            'candidate_fit_optimizer_failures_by_side': optimizer_failures or None,
            'static_contact_candidates_max_count_for_model': max(len(contacts_for_model(r, name)) for r in rows),
            'static_contact_candidates_observed_pair_count': len(pair_counts),
            'static_contact_candidates_most_frequent_pairs': sorted(pair_counts.items(), key=lambda x: -x[1])[:8],
            'static_contact_candidates_worst_penetration_mm': max(0., -worst_contact['distance_mm']) if worst_contact else 0.,
            'static_contact_candidates_worst_pair': [worst_contact['geom1'], worst_contact['geom2']] if worst_contact else None,
            'mj_geomDistance_witness_anomalies_over_1mm_count': len(witness_anomalies),
            'mj_geomDistance_witness_anomaly_u': [r['u'] for r in witness_anomalies],
            'nonmonotonic_raw_reported_steps_over_0_05mm': nonmonotonic_steps,
            'nonmonotonic_witness_consistent_adjacent_steps_over_0_05mm': reliable_nonmonotonic_steps,
            'max_reflection_error_mm': max(r['apertures'][name]['carrier_reflection_error_base_frame_mm'] for r in rows),
            'pad_frame_angle_closed_mid_open_deg': [rows[i]['apertures'][name]['terminal_collision_geom_frame_angle_deg'] for i in (0, 50, 100)],
            'root_axis_world_dot_closed_mid_open': [rows[i]['apertures'][name]['root_axes_world_dot'] for i in (0, 50, 100)],
            'all_samples_limited_joint_ranges_pass': all(
                r['joint_state'][name]['all_limited_joint_ranges_pass'] for r in rows),
            'unbounded_joint_names': rows[0]['joint_state'][name]['unbounded_joint_names'],
        }
    result = {
        'scope': 'static kinematic and compiled geometry inspection; no bottle, no physics rollout, no mj_step calls',
        'runtime': {'mujoco': mujoco.__version__, 'nq': m.nq, 'nv': m.nv, 'ngeom': m.ngeom},
        'source_identity': {
            'x2': {'repository': 'AgibotTech/agibot_x2_urdf', 'commit': '575cc6b988f976c23550e0db85aa1e5475d3652d', 'file': str(X2_URDF), 'sha256': sha256(X2_URDF)},
            'g2': {'repository': 'AgibotTech/genie_sim_robot_model', 'commit': '6d5b2159806ae0e64b432b638e9ee288d8bddc8f', 'file': str(G2_XACRO), 'sha256': sha256(G2_XACRO)},
            'previous_x2_candidate': {'model': str(args.model), 'sha256': sha256(args.model), 'metrics': str(PRIOR_METRICS), 'metrics_sha256': sha256(PRIOR_METRICS)},
        },
        'mapping': {'u': '0=closed endpoint, 1=open endpoint', 'x2_q': '-u', 'x2_wide_mimic': '+u',
                    'g2_inner': '-u*pi/4', 'g2_outer': '+u*pi/4',
                    'g2_followers': G2_FOLLOWERS, 'candidate': 'previous branch rule, not G2 mimic'},
        'measurement_limits': {
            'aperture': 'minimum signed distance over collision-enabled geoms on terminal narrow/wide carrier bodies via mj_geomDistance on compiled MuJoCo model',
            'not_a_pad_datum': 'Source does not designate a functional pad face. Mesh geom collision uses compiled convex hulls; values are collision-carrier aperture proxies, not CAD metrology.',
            '70mm_margin': 'aperture - 70 mm; positive only states scalar clearance, not a grasp pass.',
            'orientation': 'relative terminal collision-geom frame angle and nearest surface line; no source pad-face normal exists.',
            'static_contacts': 'mj_forward contact candidates only; not forces, solver contact, or proof of physical self-collision.',
        },
        'loop_definition_audit': {
            'g2': 'Measure source Xacro custom loop_joint type=spherical anchors: base xyz (0,+/-0.012,0.105)m and return-link origin. This is 3D anchor coincidence; tags are nonstandard URDF extensions and give no tolerance.',
            'x2': 'Secondary and return links are fixed in pinned URDF; no articulated loop constraint is declared.',
            'candidate': 'Two source-mesh-fitted pin-center closure targets from prior candidate evidence; fitted STL features are not CAD datums.',
        },
        'summary': summary, 'samples': rows,
        'outputs': {'viewer_xml': str(viewer_xml), 'csv': str(csv_path), 'plot': str(plot_path), 'pngs': pngs},
        'physics_steps': 0, 'mj_step_calls': 0,
    }
    result_path = args.evidence_dir / 'aperture_results.json'
    result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'result': str(result_path), 'viewer_xml': str(viewer_xml), 'plot': str(plot_path),
                      'sample_count': len(rows), 'physics_steps': 0, 'mj_step_calls': 0,
                      'summary': summary}, indent=2), flush=True)
    if args.interactive:
        interactive(m, rows, viewer_xml)


if __name__ == '__main__':
    main()
