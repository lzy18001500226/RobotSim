#!/usr/bin/env python3
"""Pinned official URDF -> reviewed experimental URDF -> isolated MJCF.

SIMULATION_ONLY: geometry measurements do not verify the candidate transmission.
Reuses Candidate B's existing analytic relation; never fits a new mechanism.
"""
from __future__ import annotations

import argparse
import copy
import csv
import difflib
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import path_governance as pg

REPO = Path('/home/lzy18001500226/projects/RobotSim')
UPSTREAM = Path('/home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0')
PIN = '575cc6b988f976c23550e0db85aa1e5475d3652d'
URDF_HASH = '35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d'
ALIGNMENT_HASH = 'e6fa5d449572f4bc42ded8147a56dcdada5287203d5175cebd717d42f9871454'
DECOMPOSITION_HASH = '9582b787684ad41e9f722380b510ee9ba95ba66b1f0775425c55f6c3a8917ff3'
SAMPLES_HASH = '7b544d5d5ebeb18e32a75ab6b9cd31ac24e8f63d03a83fc238988b941f4684f3'
SAMPLES = Path(__file__).resolve().parents[2] / 'docs/research/issue46_candidate_b_motion_samples.csv'
PHASE = -0.13412598906686654
Q_CLOSED, Q_OPEN = PHASE, 0.2
GROUPS = {
    'narrow': ['right_claw_joint', 'R_hand_narrow2_joint', 'R_hand_narrow3_joint', 'R_hand_narrow_loop_joint'],
    'wide': ['R_hand_wide1_joint', 'R_hand_wide2_joint', 'R_hand_wide3_joint', 'R_hand_wide_loop_joint'],
}
FOLLOWERS = [n for group in GROUPS.values() for n in group[1:]]
ACTUATED = [GROUPS['narrow'][0], *FOLLOWERS]
D_POINTS = {
    'narrow': {'jaw': [0.000691289490, -0.010408336990, -0.000011495782],
               'loop': [0.032861209960, -0.003021010867, -0.000039601384]},
    'wide': {'jaw': [0.000691281259, 0.010408390232, 0.000011860848],
             'loop': [0.032861206432, 0.003021046105, 0.000039840210]},
}
INTERNAL_EXCLUSIONS = [(f'R_hand_{s}{i}_Link', f'R_hand_{s}_loop_Link')
                       for s in GROUPS for i in (1, 3)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value):
    return ' '.join(f'{float(v):.16g}' for v in value)


def vec(value, default=(0., 0., 0.)):
    return np.array([float(v) for v in value.split()] if value else default)


def origin(node):
    o = node.find('origin')
    return (vec(o.get('xyz')), Rotation.from_euler('xyz', vec(o.get('rpy'))).as_matrix()) if o is not None else (np.zeros(3), np.eye(3))


def quat(matrix):
    return fmt(Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]])


def put(run, name, value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, indent=2, allow_nan=False) + '\n'
    pg.write_new_text(run / name, value, environment='wsl')


class CandidateLaw:
    """Same theta2-theta4 closure as issue46_cad_driven_gripper.Linkage.

    Uses measured A/B/C/D/E and the existing candidate source phase, not a
    newly inferred vendor transmission. Analytic evaluation replaces endpoint
    linear extrapolation; source-loop topology and root sign stay unchanged.
    """
    def __init__(self, samples=SAMPLES):
        with Path(samples).open() as stream:
            self.rows = list(csv.DictReader(stream))
        self.zero = {s: next(r for r in self.rows if r['branch'] == s and int(r['sample']) == 0) for s in GROUPS}

    def branch(self, side, q):
        r = self.zero[side]
        A, B, C, D, E = [np.array([float(r[k + '_x_mm']), float(r[k + '_z_mm'])]) for k in 'ABCDE']
        angle = lambda v: math.atan2(v[1], v[0])
        pairs = [(A, B), (B, C), (C, D), (E, D)]
        t1, t2, t3, t4 = [angle(b-a) for a, b in pairs]
        l1, l2, l3, l4 = [np.linalg.norm(b-a) for a, b in pairs]
        phase = t2-t4
        V = l4*np.array([math.cos(-phase), math.sin(-phase)])-np.array([l2, 0.])
        B = A+l1*np.array([math.cos(t1+q), math.sin(t1+q)])
        P = E-B
        cosine = (l3*l3-P@P-V@V)/(2*np.linalg.norm(P)*np.linalg.norm(V))
        if abs(cosine) > 1+1e-10:
            raise ValueError('existing candidate has no real closure')
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        choices = [angle(P)-angle(V)+sign*math.acos(np.clip(cosine, -1, 1)) for sign in (-1, 1)]
        t2n = t2+wrap(min(choices, key=lambda a: abs(wrap(a-t2)))-t2)
        t4n = t2n-phase
        C = B+l2*np.array([math.cos(t2n), math.sin(t2n)])
        D = E+l4*np.array([math.cos(t4n), math.sin(t4n)])
        t3n = t3+wrap(angle(D-C)-t3)
        return -np.array([q, t2n-t2-q, t3n-t3-(t2n-t2), t4n-t4]), float(abs(np.linalg.norm(D-C)-l3))

    def at_q(self, q):
        if not math.isfinite(q) or not Q_CLOSED-1e-12 <= q <= Q_OPEN+1e-12:
            raise ValueError('outside reviewed experimental command interval')
        out = {}
        for side, names in GROUPS.items():
            angles, _ = self.branch(side, q if side == 'narrow' else -q)
            for i, name in enumerate(names):
                out[name] = float(angles[i] + ((PHASE if side == 'narrow' else -PHASE) if i == 0 else 0))
        if not -1-1e-12 <= out['right_claw_joint'] <= 1e-12 or not -1e-12 <= out['R_hand_wide1_joint'] <= 1+1e-12:
            raise ValueError('official root limits violated')
        return out

    def targets(self, opening):
        if not math.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError('opening must be finite and in [0,1]')
        return self.at_q(Q_CLOSED+(Q_OPEN-Q_CLOSED)*opening)


def derive_urdf(raw, alignment, upstream=UPSTREAM):
    root = ET.fromstring(raw)
    joints = {j.get('name'): j for j in root.findall('joint')}
    links = {l.get('name'): l for l in root.findall('link')}
    changes = []
    for name in FOLLOWERS:
        j = joints[name]
        if j.get('type') != 'fixed':
            raise ValueError(f'not the reviewed official fixed joint: {name}')
        j.set('type', 'continuous')
        # No invented position limit: these were fixed-joint placeholder values.
        # Effort/velocity are simulator overlay values, never vendor measurements.
        j.find('limit').attrib = {'effort': '0.5', 'velocity': '25'}
        changes.append({'joint': name, 'delta': 'fixed -> continuous', 'origin_and_axis': 'unchanged official; CAD compared', 'classification': 'UNVERIFIED_EXPERIMENTAL'})
    for name, item in alignment['actual_bore_registration'].items():
        for kind in ('visual', 'collision'):
            element = links[name].find(kind)
            p, R = origin(element)
            # Same reviewed registration as the accepted HTML, in mesh-local coordinates.
            corrected = Rotation.from_euler('z', item['visual_frame_rotation_deg'], degrees=True).as_matrix() @ R
            o = element.find('origin')
            if o is None:
                o = ET.SubElement(element, 'origin')
            o.set('xyz', fmt(p))
            o.set('rpy', fmt(Rotation.from_matrix(corrected).as_euler('xyz')))
        changes.append({'link': name, 'visual_collision_local_rotation_deg': item['visual_frame_rotation_deg'], 'source': 'reviewed measured STL bore registration'})
    for side in GROUPS:
        j = joints[f'R_hand_{side}4_joint']
        if j.get('type') != 'fixed':
            raise ValueError('D frame unexpectedly articulated')
        j.find('origin').set('xyz', fmt(D_POINTS[side]['jaw']))
        changes.append({'joint': j.get('name'), 'delta': 'meshless D reference frame coordinate re-expression', 'xyz': D_POINTS[side]['jaw']})
    for mesh in root.findall('.//mesh'):
        f = mesh.get('filename')
        if not f.startswith('./meshes/'):
            raise ValueError('unexpected upstream mesh URI')
        mesh.set('filename', str(upstream / f[2:]))
    # Check exact scope structurally, without mistaking XML reformatting for changes.
    original = ET.fromstring(raw)
    for a in original.findall('link'):
        b = links[a.get('name')]
        ia, ib = a.find('inertial'), b.find('inertial')
        if (None if ia is None else ET.tostring(ia)) != (None if ib is None else ET.tostring(ib)):
            raise AssertionError('source inertia changed')
    for a in original.findall('joint'):
        b = joints[a.get('name')]
        if a.get('name') not in FOLLOWERS and not a.get('name') in {f'R_hand_{s}4_joint' for s in GROUPS}:
            if ET.tostring(a) != ET.tostring(b):
                raise AssertionError(f'unapproved joint edit: {a.get("name")}')
    ET.indent(root, space='  ')
    return root, changes


def serialize_derived(raw, derived):
    """Preserve unchanged upstream text; replace only changed XML blocks."""
    text = raw.decode().replace('filename="./meshes/', f'filename="{UPSTREAM}/meshes/')
    rebased = ET.fromstring(text)
    def signature(e):
        return e.tag, e.attrib, [signature(c) for c in e]
    replacements = []
    for a in rebased:
        name = a.get('name')
        if a.tag not in ('link', 'joint') or name is None:
            continue
        b = derived.find(f'{a.tag}[@name="{name}"]')
        if signature(a) == signature(b):
            continue
        pattern = rf'<{a.tag}\b[^>]*\bname="{re.escape(name)}"[^>]*>.*?</{a.tag}>'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if len(matches) != 1:
            raise ValueError('ambiguous source block')
        element = copy.deepcopy(b)
        element.tail = None
        ET.indent(element, space='  ', level=1)
        replacements.append((matches[0].start(), matches[0].end(), ET.tostring(element, encoding='unicode')))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start]+replacement+text[end:]
    if signature(ET.fromstring(text)) != signature(derived):
        raise AssertionError('minimal URDF serialization changed the model')
    return text


def convert(urdf, decomposition, *, actuation='seven-servo-reference'):
    """Read the derived URDF itself; no body placements copied from an old MJCF."""
    links = {l.get('name'): l for l in urdf.findall('link')}
    joints = urdf.findall('joint')
    root = ET.Element('mujoco', model='EXPERIMENTAL_OFFICIAL_URDF_CAD_OVERLAY')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false', fusestatic='false')
    ET.SubElement(root, 'option', timestep='0.001', gravity='0 0 -9.81', integrator='implicitfast', iterations='100', cone='elliptic')
    ET.SubElement(ET.SubElement(root, 'visual'), 'global', offwidth='800', offheight='640')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -0.4 0.5', dir='0 1 -1')
    ET.SubElement(world, 'camera', name='review', pos='0 -0.38 0.12', xyaxes='1 0 0 0 0 1')
    parts = {Path(p['source_mesh']).stem: p for p in decomposition['meshes']}
    assets = set()
    base = ET.SubElement(world, 'body', name='R_omnipicker_base_link')

    def add_link(body, name):
        link = links[name]
        inertial = link.find('inertial')
        if inertial is not None:
            p, R = origin(inertial)
            m = inertial.find('mass').get('value')
            a = inertial.find('inertia').attrib
            I = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])], [float(a['ixy']), float(a['iyy']), float(a['iyz'])], [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
            I = R@I@R.T
            ET.SubElement(body, 'inertial', pos=fmt(p), mass=m, fullinertia=fmt([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]]))
        for kind in ('visual', 'collision'):
            for element in link.findall(kind):
                mesh = element.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('unexpected non-mesh official geometry')
                path = Path(mesh.get('filename'))
                if mesh.get('scale') is not None and not np.allclose(vec(mesh.get('scale')), 1):
                    raise ValueError('unreviewed mesh scale')
                p, R = origin(element)
                stem = path.stem
                if kind == 'visual' or stem == 'base_link':
                    objects = [{'name': stem, 'file': str(path)}]
                else:
                    dec = parts[stem]
                    if sha(path) != dec['sha256']:
                        raise ValueError('collision source hash mismatch')
                    objects = dec['collision_pieces']
                for obj in objects:
                    meshname = obj['name']
                    if meshname not in assets:
                        if 'sha256' in obj and sha(obj['file']) != obj['sha256']:
                            raise ValueError('collision piece hash mismatch')
                        ET.SubElement(asset, 'mesh', name=meshname, file=obj['file'])
                        assets.add(meshname)
                    visual = kind == 'visual'
                    color = '0.75 0.78 0.82 1' if 'base' in name else ('0.3 0.75 0.55 1' if 'narrow3' in name else '0.75 0.5 0.3 1')
                    ET.SubElement(body, 'geom', name=f'{name}__{kind}__{meshname}', type='mesh', mesh=meshname,
                                  pos=fmt(p), quat=quat(R), contype='0' if visual else '1', conaffinity='0' if visual else '1',
                                  group='0' if visual else '1', rgba=color if visual else '0.1 0.8 0.8 0.2', density='0',
                                  friction='1.4 0.02 0.001', condim='4', solref='0.01 1')

    def walk(parent, body):
        add_link(body, parent)
        for j in joints:
            if j.find('parent').get('link') != parent:
                continue
            child = j.find('child').get('link')
            p, R = origin(j)
            b = ET.SubElement(body, 'body', name=child, pos=fmt(p), quat=quat(R))
            if j.get('type') != 'fixed':
                attrs = {'name': j.get('name'), 'type': 'hinge', 'axis': j.find('axis').get('xyz')}
                lim = j.find('limit')
                if j.get('type') == 'revolute':
                    attrs.update(limited='true', range=f'{lim.get("lower")} {lim.get("upper")}')
                else:
                    attrs['limited'] = 'false'
                ET.SubElement(b, 'joint', attrs)
            walk(child, b)
    walk('R_omnipicker_base_link', base)
    for side in GROUPS:
        jaw = root.find(f'.//body[@name="R_hand_{side}4_Link"]')
        loop = root.find(f'.//body[@name="R_hand_{side}_loop_Link"]')
        ET.SubElement(jaw, 'site', name=f'{side}_D_jaw', pos='0 0 0', size='0.0004', rgba='1 0 0 1')
        ET.SubElement(loop, 'site', name=f'{side}_D_loop', pos=fmt(D_POINTS[side]['loop']), size='0.0004', rgba='0 1 1 1')
    eq = ET.SubElement(root, 'equality')
    mimic = next(j for j in joints if j.get('name') == 'R_hand_wide1_joint').find('mimic')
    ET.SubElement(eq, 'joint', name='official_root_mimic', joint1='R_hand_wide1_joint', joint2=mimic.get('joint'),
                  polycoef=f'{mimic.get("offset", "0")} {mimic.get("multiplier")} 0 0 0', solref='0.002 1')
    for side in GROUPS:
        ET.SubElement(eq, 'connect', name=f'candidate_{side}_D', site1=f'{side}_D_jaw', site2=f'{side}_D_loop', solref='0.002 1')
    contact = ET.SubElement(root, 'contact')
    for a, b in INTERNAL_EXCLUSIONS:
        ET.SubElement(contact, 'exclude', body1=a, body2=b)
    if actuation not in ('seven-servo-reference', 'single-motor-experimental'):
        raise ValueError('unknown experimental actuation mode')
    if actuation == 'single-motor-experimental':
        # Same PR #84 Candidate B relation, re-expressed for URDF coordinates.
        # This converter has root ref=0, unlike the old ref=+/-PHASE model.
        # Fixed-tendon length therefore needs the source phase as its offset.
        tendons = ET.SubElement(root, 'tendon')
        for side, names in GROUPS.items():
            tendon = ET.SubElement(tendons, 'fixed', name=f'candidate_{side}_rate')
            for joint, coefficient in ((names[0], '1'), (names[1], '1'), (names[3], '-1')):
                ET.SubElement(tendon, 'joint', joint=joint, coef=coefficient)
            phase = PHASE if side == 'narrow' else -PHASE
            ET.SubElement(eq, 'tendon', name=f'candidate_{side}_rate_eq', tendon1=f'candidate_{side}_rate',
                          polycoef=f'{phase:.16g} 1 0 0 0', solref='0.002 1')
    root.set('model', 'UNVERIFIED_EXPERIMENTAL_SINGLE_MOTOR' if actuation == 'single-motor-experimental' else 'SEVEN_SERVO_KINEMATIC_REFERENCE')
    actuators = ET.SubElement(root, 'actuator')
    driven = ['right_claw_joint'] if actuation == 'single-motor-experimental' else ACTUATED
    for name in driven:
        attrs = {'name': f'servo_{name}', 'joint': name, 'kp': '0.5', 'kv': '0.05', 'forcelimited': 'true', 'forcerange': '-0.5 0.5'}
        if name == 'right_claw_joint':
            attrs.update(ctrllimited='true', ctrlrange='-1 0')
        ET.SubElement(actuators, 'position', attrs)
    return root


def addresses(model):
    return {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for group in GROUPS.values() for n in group}


def gaps(model, data):
    out = []
    for s in GROUPS:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'{s}_D_{k}') for k in ('jaw', 'loop')]
        out.append(float(np.linalg.norm(data.site_xpos[ids[0]]-data.site_xpos[ids[1]])))
    return out


def fk(urdf, targets):
    out = {'R_omnipicker_base_link': (np.zeros(3), np.eye(3))}
    todo = list(urdf.findall('joint'))
    while todo:
        progress = False
        for j in todo[:]:
            parent = j.find('parent').get('link')
            if parent not in out:
                continue
            child = j.find('child').get('link')
            p, R = origin(j)
            if j.get('type') != 'fixed':
                axis = vec(j.find('axis').get('xyz'))
                R = R@Rotation.from_rotvec(axis*targets[j.get('name')]).as_matrix()
            pp, Rp = out[parent]
            out[child] = (pp+Rp@p, Rp@R)
            todo.remove(j)
            progress = True
        if not progress:
            break  # Other robot links are deliberately outside this isolated subtree.
    return out


def stl_vertices(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84+50*n:
        raise ValueError('expected exact binary STL')
    return np.ndarray((n, 3, 3), dtype='<f4', buffer=raw, offset=96, strides=(50, 12, 4)).reshape(-1, 3).astype(float)


def static_check(urdf, model, law):
    data = mujoco.MjData(model)
    addrs = addresses(model)
    surface = {}
    for side in GROUPS:
        l = urdf.find(f'link[@name="R_hand_{side}3_Link"]').find('visual')
        p, R = origin(l)
        surface[side] = stl_vertices(l.find('geometry/mesh').get('filename'))@R.T+p
    rows = []
    for ratio in np.linspace(0, 1, 21):
        targets = law.targets(float(ratio))
        for n, v in targets.items():
            data.qpos[addrs[n]] = v
        mujoco.mj_forward(model, data)
        poses = fk(urdf, targets)
        poserr, roterr = 0., 0.
        for name, (p, R) in poses.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            poserr = max(poserr, float(np.linalg.norm(p-data.xpos[bid])))
            roterr = max(roterr, float(np.max(np.abs(R-data.xmat[bid].reshape(3, 3)))))
        clouds = []
        for s in GROUPS:
            p, R = poses[f'R_hand_{s}3_Link']
            clouds.append(surface[s]@R.T+p)
        distance, _ = cKDTree(clouds[1]).query(clouds[0])
        contact_pairs = [{'geom1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                          'geom2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                          'penetration_mm': max(0., -float(c.dist))*1000} for c in data.contact]
        rows.append({'opening': float(ratio), 'q_root': targets['right_claw_joint'], 'D_gap_mm': [v*1000 for v in gaps(model, data)],
                     'urdf_mjcf_position_error_m': poserr, 'urdf_mjcf_rotation_matrix_error': roterr,
                     'jaw_vertex_distance_mm_NOT_exact_aperture': float(np.min(distance)*1000),
                     'max_penetration_mm': max([max(0., -c.dist)*1000 for c in data.contact] or [0.]),
                     'contact_pairs': contact_pairs})
    if max(r['urdf_mjcf_position_error_m'] for r in rows) > 1e-9 or max(r['urdf_mjcf_rotation_matrix_error'] for r in rows) > 1e-9:
        raise AssertionError('derived URDF and generated MJCF disagree')
    return rows


def rollout(model, law, run, motion='source-window'):
    """Historical seven-servo experiment helper; not the target physics path.

    The public exporter no longer invokes this. Use the single-motor travel
    entrypoint for new physical validation; retained only for old evidence.
    """
    data = mujoco.MjData(model)
    addrs = addresses(model)
    actuators = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'servo_{n}') for n in ACTUATED}
    source_opening = (0.-Q_CLOSED)/(Q_OPEN-Q_CLOSED)
    min_opening = (-.1-Q_CLOSED)/(Q_OPEN-Q_CLOSED) if motion == 'source-window' else 0.
    initial = law.targets(source_opening)
    for n, v in initial.items():
        data.qpos[addrs[n]] = v
    mujoco.mj_forward(model, data)
    rows, stop, frames = [], None, []
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, .12]
    camera.distance, camera.azimuth, camera.elevation = .34, 0, -12
    renderer = mujoco.Renderer(model, height=640, width=800)
    visual_options = mujoco.MjvOption()
    visual_options.geomgroup[1] = 0  # Hide collision proxies for rendering only.
    from PIL import Image
    maxspeed, maxgap, maxpen, maxerror = 0., 0., 0., 0.
    # Mapped STEP source -> open -> close endpoint. qpos is set ONLY above.
    for step in range(6000):
        t = step*.001
        if t < 3:
            blend = .5-.5*math.cos(math.pi*min(t/2.5, 1))
            ratio = source_opening+(1-source_opening)*blend
        else:
            blend = .5-.5*math.cos(math.pi*min((t-3)/2.5, 1))
            ratio = 1-(1-min_opening)*blend
        targets = law.targets(ratio)
        for n, a in actuators.items():
            data.ctrl[a] = targets[n]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)  # Sample transforms/forces at the new qpos.
        speed = float(np.max(np.abs(data.qvel)))
        gap = max(gaps(model, data))
        pen = max([max(0., -c.dist) for c in data.contact] or [0.])
        error = max(abs(float(data.qpos[addrs[n]])-targets[n]) for n in ACTUATED)
        maxspeed, maxgap, maxpen, maxerror = max(maxspeed, speed), max(maxgap, gap), max(maxpen, pen), max(maxerror, error)
        limit_error = max(0., -1-float(data.qpos[addrs['right_claw_joint']]), float(data.qpos[addrs['right_claw_joint']]), -float(data.qpos[addrs['R_hand_wide1_joint']]), float(data.qpos[addrs['R_hand_wide1_joint']])-1)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            stop = 'NONFINITE'
        elif speed > 25:
            stop = 'VELOCITY_LIMIT'
        elif gap > .001:
            stop = 'D_GAP_OVER_1MM'
        elif pen > .002:
            stop = 'PENETRATION_OVER_2MM'
        elif limit_error > .001:
            stop = 'ROOT_LIMIT_OVER_1MRAD'
        elif any(int(w.number) for w in data.warning):
            stop = 'MUJOCO_WARNING'
        if step % 10 == 0 or stop:
            rows.append({'step': step+1, 'time_s': float(data.time), 'opening_command': ratio, 'D_gap_mm': gap*1000,
                         'max_penetration_mm': pen*1000, 'max_qvel': speed, 'target_error_rad': error,
                         'root_limit_error_rad': limit_error, **{n: float(data.qpos[a]) for n, a in addrs.items()},
                         'max_constraint_force': float(np.max(np.abs(data.efc_force))) if data.nefc else 0.,
                         'max_actuator_force_Nm': float(np.max(np.abs(data.actuator_force))),
                         **{n+'_target': v for n, v in targets.items()}})
        if step % 50 == 0 or stop:
            renderer.update_scene(data, camera=camera, scene_option=visual_options)
            frame = Image.fromarray(renderer.render())
            frames.append(frame.copy())
            if step in (0, 2500, 5500) or stop:
                frame.save(run/f'physics_{step+1:05d}.png')
        if stop:
            break
    renderer.close()
    frames[0].save(run/'physics_open_close.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
    with (run/'physics_trace.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return {'status': 'PASS' if stop is None else 'FAIL', 'stop_reason': stop, 'steps': step+1, 'simulation_time_s': float(data.time),
            'gravity': model.opt.gravity.tolist(), 'qpos_writes_after_first_step': 0, 'max_speed_rad_s': maxspeed,
            'max_D_gap_mm': maxgap*1000, 'max_penetration_mm': maxpen*1000, 'max_target_error_rad': maxerror,
            'last_root_rad': float(data.qpos[addrs['right_claw_joint']]), 'last_opening_command': ratio,
            'motion': motion, 'CAD_interval': [-.1 if motion == 'source-window' else Q_CLOSED, Q_OPEN],
            'scope': 'isolated fixed-base free-space open-close, no object, no bottle or lift validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, default=REPO/'reviews/issue-46')
    parser.add_argument('--alignment', type=Path, default=REPO/'reviews/issue-46/omnipicker-bore-alignment-fix-20261022T040538Z/result.json')
    parser.add_argument('--decomposition', type=Path, default=REPO/'reviews/issue-46/omnipicker-collision-decomp-20261020T061939Z/decomposition.json')
    parser.add_argument('--motion', choices=('source-window', 'root-endpoint'), default='source-window')
    args = parser.parse_args()
    source = UPSTREAM/'X2-Ultra_omnipicker.urdf'
    raw = source.read_bytes()
    if sha(source) != URDF_HASH:
        raise ValueError('pinned official URDF hash mismatch')
    for path, expected in ((args.alignment, ALIGNMENT_HASH), (args.decomposition, DECOMPOSITION_HASH), (SAMPLES, SAMPLES_HASH)):
        if sha(path) != expected:
            raise ValueError(f'reviewed input changed: {path.name}')
    alignment = json.loads(args.alignment.read_text())
    dec = json.loads(args.decomposition.read_text())
    if dec['source_repo_commit'] != PIN or dec['source_urdf_sha256'] != URDF_HASH:
        raise ValueError('collision provenance mismatch')
    urdf, changes = derive_urdf(raw, alignment)
    xml = convert(urdf, dec)
    law = CandidateLaw()
    location = pg.create_unique_output_directory(args.reviews, prefix='seven-servo-reference', environment='wsl')
    run = location.native_path
    urdf_path = run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf'
    put(run, urdf_path.name, serialize_derived(raw, urdf))
    put(run, 'official_to_derived.diff', ''.join(difflib.unified_diff(raw.decode().splitlines(True), urdf_path.read_text().splitlines(True), fromfile='official/X2-Ultra_omnipicker.urdf', tofile='derived/X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf')))
    ET.indent(xml, space='  ')
    put(run, 'gripper.xml', ET.tostring(xml, encoding='unicode')+'\n')
    model = mujoco.MjModel.from_xml_path(str(run/'gripper.xml'))
    statics = static_check(ET.parse(urdf_path).getroot(), model, law)
    put(run, 'static_validation.json', statics)
    physics = {'status':'REFERENCE_ONLY_NOT_RUN','steps':0,'simulation_time_s':0.,'stop_reason':None,
               'scope':'seven-servo static/visual reference only; use issue46_single_motor_travel.py for physical validation'}
    put(run, 'physics_result.json', physics)
    provenance = {'classification': 'UNVERIFIED_EXPERIMENTAL_MODEL', 'official_repo': 'AgibotTech/agibot_x2_urdf', 'official_commit': PIN,
                  'official_urdf_sha256': URDF_HASH, 'derived_urdf_sha256': sha(urdf_path), 'mjcf_sha256': sha(run/'gripper.xml'),
                  'input_hashes': {str(p): sha(p) for p in (SAMPLES, args.alignment, args.decomposition)}, 'changes': changes,
                  'not_cad_measured': ['candidate passive law theta2-theta4 source-phase relation', 'servo kp=0.5 kv=0.05 force cap 0.5 Nm', 'four inherited internal contact exclusions', 'CoACD collision approximation'],
                  'internal_exclusions_unchanged': INTERNAL_EXCLUSIONS, 'gripper_fixture': 'right hand fixed base; official wrist mount untouched in full URDF',
                  'source_unchanged_after_run': sha(source) == URDF_HASH,
                  'root_mapping': 'q_URDF = -0.13412598906686654 - q_CAD; official axis/limits/mimic unchanged',
                  'endpoint_evaluation': 'analytic existing candidate closure; no linear angle extrapolation',
                  'aperture_measurement': 'minimum vertex distance only; not exact gripping-surface aperture',
                  'runtime': mujoco.__version__, 'physics': physics}
    put(run, 'result.json', provenance)
    put(run, 'REPORT.md', f'''# Official URDF / CAD overlay pipeline\n\nUNVERIFIED_EXPERIMENTAL, SIMULATION_ONLY.\n\n- Full official URDF -> derived URDF -> parsed right-hand subtree -> generated MJCF.\n- Official meshes, inertias, root axes/limits/mimic and wrist mount retained.\n- Six right downstream joints articulated; four reviewed local mesh registrations and measured D reference frame re-expression applied. Left hand untouched.\n- Runtime uses one opening command, analytic Candidate B follower targets, seven simulated servos; no new gear equality. The candidate relationship is not vendor verified.\n- Same four inherited internal exclusions and hashed source-STL convex decompositions; no new filtering or mesh generation.\n- Closed endpoint at official root zero still has {statics[0]['jaw_vertex_distance_mm_NOT_exact_aperture']:.4f} mm minimum jaw vertex distance. This is not exact contact-face clearance or proof of manufacturer travel.\n- URDF/MJCF world transforms agree over 21 poses; max position error {max(r['urdf_mjcf_position_error_m'] for r in statics):.3g} m.\n- Physics {physics['status']}, {physics['steps']} mj_step calls, {physics['simulation_time_s']:.3f} s, stop={physics['stop_reason']}. Gravity enabled throughout.\n- No bottle/lift success claim. Do not promote this overlay into canonical X2.\n\nSee result.json, static_validation.json, physics_trace.csv and physics_open_close.gif.\n''')
    pg.write_artifact_manifest(run, environment='wsl')
    print(json.dumps({'run': str(run), 'physics': physics}, indent=2))


if __name__ == '__main__':
    main()
