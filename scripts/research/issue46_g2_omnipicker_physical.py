#!/usr/bin/env python3
"""Build and validate the pinned Genie G2 OmniPicker in MuJoCo 3.3.6.

This converter handles only the pinned source macro used here; it is not a
general Xacro implementation. Source mesh and joint data are read at runtime.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SOURCE_ROOT = Path('/home/lzy18001500226/.cache/robotsim/upstreams/genie_sim_robot_model')
XACRO_REL = Path('robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro')
MESH_REL = Path('robots/genie/g2/meshes/gripper/omnipicker')
PIN = '6d5b2159806ae0e64b432b638e9ee288d8bddc8f'
XACRO_SHA = 'b20d8691b03a256ad3481380b71fd1e1eaa11495673f4bf7fd7377f67238d183'
NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
PASSIVE = {'inner': ('inner_joint3', 'inner_joint4', 'inner_joint0'),
           'outer': ('outer_joint3', 'outer_joint4', 'outer_joint0')}
MIMIC = {'inner_joint3': .1, 'inner_joint4': .25, 'inner_joint0': -.7,
         'outer_joint3': .1, 'outer_joint4': -.25, 'outer_joint0': .7}
LINKS = ('base_link', 'inner_link1', 'inner_link3', 'inner_link4', 'inner_link2',
         'outer_link1', 'outer_link3', 'outer_link4', 'outer_link2', 'camera_link')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    import subprocess
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def safe_eval(expr: str) -> float:
    expr = expr.strip().replace('PI_2', '(pi/2)').replace('PI', 'pi')
    tree = ast.parse(expr, mode='eval')
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        if isinstance(n, ast.Name) and n.id == 'pi': return math.pi
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -visit(n.operand) if isinstance(n.op, ast.USub) else visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if isinstance(n.op, ast.Div): return a / b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'radians':
            return math.radians(visit(n.args[0]))
        raise ValueError(f'Unsupported source expression {expr!r}')
    return float(visit(tree))


def expand(value: str, side='l') -> str:
    base = 30 if side == 'l' else 70
    def repl(m):
        expr = m.group(1).strip()
        if expr == 'side': return side
        if expr == 'parent_link': return 'parent_link'
        expr = expr.replace('_b', str(base))
        return f'{safe_eval(expr):.17g}'
    return re.sub(r'\$\{([^}]+)\}', repl, value)


def vec(value, width=3, default=None):
    if value is None: return np.asarray(default if default is not None else [0.] * width, float)
    value = expand(value)
    return np.asarray([safe_eval(v) for v in value.split()], float)

def number(value):
    return safe_eval(expand(value))


def quat(rpy):
    q = Rotation.from_euler('xyz', rpy).as_quat()
    return [q[3], q[0], q[1], q[2]]


def parse_source(path: Path):
    root = ET.parse(path).getroot()
    links = {}
    for e in root.findall('.//link'):
        name = expand(e.get('name', ''))
        ine = e.find('inertial'); org = ine.find('origin'); it = ine.find('inertia')
        links[name] = {'name': name, 'mass': float(ine.find('mass').get('value')),
            'pos': vec(org.get('xyz') if org is not None else None).tolist(),
            'rpy': vec(org.get('rpy') if org is not None else None).tolist(),
            'inertia': {k: float(it.get(k, '0')) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}}

    joints = {}
    def walk(node, active=True):
        tag = node.tag.split('}')[-1]
        if tag == 'if':
            condition = node.get('value', '')
            if "side == 'l'" in condition or 'side == "l"' in condition: active &= True
            elif "side == 'r'" in condition or 'side == "r"' in condition: active = False
        if tag == 'joint' and active:
            name = expand(node.get('name', ''))
            parent = expand(node.find('parent').get('link', ''))
            child = expand(node.find('child').get('link', ''))
            if child in ('gripper_l_camera_link', 'gripper_l_base_link'):
                for c in list(node): walk(c, active)
                return
            org = node.find('origin'); lim = node.find('limit'); mim = node.find('mimic'); drv = node.find('drive')
            joints[name] = {'name': name, 'type': node.get('type'), 'parent': parent, 'child': child,
                'xyz': vec(org.get('xyz') if org is not None else None).tolist(),
                'rpy': vec(org.get('rpy') if org is not None else None).tolist(),
                'axis': vec(node.find('axis').get('xyz') if node.find('axis') is not None else None, default=[0, 0, 1]).tolist(),
                'limit': ({k: number(lim.get(k)) for k in ('lower', 'upper', 'effort', 'velocity')} if lim is not None else None),
                'mimic': ({'joint': expand(mim.get('joint')), 'multiplier': number(mim.get('multiplier', '1')),
                           'offset': number(mim.get('offset', '0'))} if mim is not None else None),
                'drive': ({k: number(drv.get(k)) for k in ('stiffness', 'damping')} if drv is not None else None)}
        for c in list(node): walk(c, active)
    walk(root)
    loops = []
    for e in root.findall('.//loop_joint'):
        a, b = e.find('link1'), e.find('link2')
        loops.append({'name': expand(e.get('name')), 'type': e.get('type'),
            'link1': expand(a.get('link')), 'xyz1': vec(a.get('xyz')).tolist(), 'rpy1': vec(a.get('rpy')).tolist(),
            'link2': expand(b.get('link')), 'xyz2': vec(b.get('xyz')).tolist(), 'rpy2': vec(b.get('rpy')).tolist()})
    if len(links) != 10 or len(joints) != 8 or len(loops) != 2:
        raise ValueError(f'Pinned macro structure mismatch: links={len(links)}, joints={len(joints)}, loops={len(loops)}')
    return links, joints, loops


def dae_to_obj(src: Path, dst: Path):
    root = ET.parse(src).getroot()
    unit = root.find('c:asset/c:unit', NS); meter = float(unit.get('meter', '1')) if unit is not None else 1.
    up = root.findtext('c:asset/c:up_axis', default='Y_UP', namespaces=NS)
    sources = {}
    for s in root.findall('.//c:source', NS):
        arr, acc = s.find('c:float_array', NS), s.find('.//c:accessor', NS)
        if arr is None or acc is None: continue
        stride = int(acc.get('stride', '1')); raw = np.fromstring(arr.text or '', sep=' '); count = int(acc.get('count', len(raw)//stride))
        sources[s.get('id')] = raw[:count*stride].reshape(count, stride)
    vm = {}
    for v in root.findall('.//c:vertices', NS):
        p = next((i for i in v.findall('c:input', NS) if i.get('semantic') == 'POSITION'), None)
        if p is not None: vm['#' + v.get('id', '')] = p.get('source')
    geom = root.find('.//c:geometry', NS); positions = None; faces = []
    for prim in geom.findall('.//c:triangles', NS) + geom.findall('.//c:polylist', NS):
        ins = prim.findall('c:input', NS); stride = max(int(i.get('offset', '0')) for i in ins) + 1
        vi = next((i for i in ins if i.get('semantic') == 'VERTEX'), None)
        if vi is None: vi = next(i for i in ins if i.get('semantic') == 'POSITION')
        ref = vm.get(vi.get('source'), vi.get('source')).lstrip('#'); positions = sources[ref][:, :3]
        stream = np.fromstring(prim.findtext('c:p', default='', namespaces=NS), sep=' ', dtype=np.int64).reshape(-1, stride)[:, int(vi.get('offset', '0'))]
        if prim.tag.endswith('triangles'):
            faces.extend(tuple(map(int, stream[i:i+3])) for i in range(0, len(stream), 3))
        else:
            counts = np.fromstring(prim.findtext('c:vcount', default='', namespaces=NS), sep=' ', dtype=np.int64); cur = 0
            for n in counts:
                poly = stream[cur:cur+int(n)]; cur += int(n)
                faces.extend((int(poly[0]), int(poly[j]), int(poly[j+1])) for j in range(1, len(poly)-1))
    mat = np.eye(4); nodes = root.findall('.//c:visual_scene//c:node[c:instance_geometry]', NS)
    if nodes and nodes[0].find('c:matrix', NS) is not None:
        mat = np.fromstring(nodes[0].findtext('c:matrix', default='', namespaces=NS), sep=' ').reshape(4, 4)
    vertices = (positions @ mat[:3, :3].T + mat[:3, 3]) * meter
    if up == 'Y_UP': vertices = vertices @ np.array([[1,0,0],[0,0,-1],[0,1,0]], float).T
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='ascii', newline='\n') as f:
        f.write(f'# Rendering conversion from pinned source DAE {src.name}; unit and scene matrix baked.\n')
        for p in vertices: f.write('v %.10g %.10g %.10g\n' % tuple(p))
        for face in faces: f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    return {'source_dae_sha256': sha256(src), 'derived_obj_sha256': sha256(dst), 'unit_meter': meter,
        'up_axis': up, 'scene_matrix': mat.tolist(), 'vertex_count': int(len(vertices)),
        'triangle_count': int(len(faces)), 'bounds_m': [vertices.min(0).tolist(), vertices.max(0).tolist()]}


def build_model(out: Path, links: dict, joints: dict, loops: list):
    root = ET.Element('mujoco', {'model': 'Pinned G2 OmniPicker | constrained dynamics research'})
    ET.SubElement(root, 'compiler', {'angle': 'radian'})
    ET.SubElement(root, 'option', {'timestep': '.001', 'gravity': '0 0 -9.81', 'integrator': 'implicitfast',
        'solver': 'Newton', 'iterations': '100', 'tolerance': '1e-10'})
    ET.SubElement(root, 'size', {'njmax': '400', 'nconmax': '200'})
    vis = ET.SubElement(root, 'visual'); ET.SubElement(vis, 'global', {'offwidth': '1280', 'offheight': '960'})
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', {'name': 'sky', 'type': 'skybox', 'builtin': 'gradient', 'rgb1': '.85 .91 .97', 'rgb2': '.38 .51 .62', 'width': '512', 'height': '512'})
    ET.SubElement(asset, 'texture', {'name': 'grid', 'type': '2d', 'builtin': 'checker', 'rgb1': '.88 .90 .92', 'rgb2': '.71 .76 .80', 'width': '512', 'height': '512'})
    ET.SubElement(asset, 'material', {'name': 'floor_mat', 'texture': 'grid', 'texrepeat': '8 8', 'rgba': '1 1 1 1'})
    meshdir = SOURCE_ROOT / MESH_REL; mesh_info = {}
    for stem in LINKS:
        dae = meshdir / f'{stem}.dae'; obj = out / 'meshes' / f'{stem}.obj'
        mi = dae_to_obj(dae, obj); stl = meshdir / 'convex' / f'{stem}.STL'
        mi.update({'collision_stl_sha256': sha256(stl), 'collision_stl_bytes': stl.stat().st_size})
        mesh_info[stem] = mi
        ET.SubElement(asset, 'mesh', {'name': f'visual_{stem}', 'file': str(obj)})
        ET.SubElement(asset, 'mesh', {'name': f'collision_{stem}', 'file': str(stl)})
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', {'name': 'key', 'pos': '0 -.25 .35', 'dir': '0 0 -1', 'diffuse': '.9 .9 .9'})
    ET.SubElement(world, 'geom', {'name': 'floor', 'type': 'plane', 'pos': '0 0 -.005', 'size': '1 1 .1', 'material': 'floor_mat', 'contype': '0', 'conaffinity': '0'})
    parent_joints = {}
    for j in joints.values(): parent_joints.setdefault(j['parent'], []).append(j)
    stem = {n: n.removeprefix('gripper_l_') for n in links}

    def add_geom(body, src_link):
        suffix = stem[src_link]
        ET.SubElement(body, 'geom', {'name': f'{suffix}_visual', 'type': 'mesh', 'mesh': f'visual_{suffix}',
            'group': '0', 'contype': '0', 'conaffinity': '0', 'rgba': '.86 .89 .91 1', 'mass': '0'})
        ET.SubElement(body, 'geom', {'name': f'{suffix}_collision', 'type': 'mesh', 'mesh': f'collision_{suffix}',
            'group': '1', 'contype': '1', 'conaffinity': '1', 'rgba': '.95 .48 .12 .35', 'mass': '0'})

    def add_link(parent_xml, name, j=None):
        attrs = {'name': stem[name]}
        if j is not None:
            attrs.update({'pos': ' '.join(f'{v:.12g}' for v in j['xyz']), 'quat': ' '.join(f'{v:.12g}' for v in quat(j['rpy']))})
        body = ET.SubElement(parent_xml, 'body', attrs)
        if j is not None and j['type'] == 'revolute':
            lim = j['limit']; jname = j['name'].split('gripper_l_', 1)[-1]
            ET.SubElement(body, 'joint', {'name': jname, 'type': 'hinge', 'axis': ' '.join(f'{v:.12g}' for v in j['axis']),
                'range': f"{lim['lower']:.12g} {lim['upper']:.12g}", 'limited': 'true',
                'actuatorfrclimited': 'true', 'actuatorfrcrange': f"{-lim['effort']:.12g} {lim['effort']:.12g}"})
        ld = links[name]; inertia = ld['inertia']
        ET.SubElement(body, 'inertial', {'pos': ' '.join(f'{v:.12g}' for v in ld['pos']),
            'quat': ' '.join(f'{v:.12g}' for v in quat(ld['rpy'])), 'mass': f"{ld['mass']:.12g}",
            'diaginertia': ' '.join(f'{inertia[k]:.12g}' for k in ('ixx', 'iyy', 'izz'))})
        add_geom(body, name)
        if name == 'gripper_l_base_link':
            for lp in loops:
                side = 'inner' if 'inner' in lp['name'] else 'outer'
                ET.SubElement(body, 'site', {'name': f'loop_base_{side}', 'pos': ' '.join(map(str, lp['xyz1'])),
                    'quat': ' '.join(map(str, quat(lp['rpy1']))), 'size': '.002', 'rgba': '1 .2 .12 1'})
        if name in ('gripper_l_inner_link2', 'gripper_l_outer_link2'):
            side = 'inner' if 'inner' in name else 'outer'
            ET.SubElement(body, 'site', {'name': f'loop_end_{side}', 'pos': '0 0 0', 'size': '.002', 'rgba': '.1 .9 .3 1'})
        for child_j in parent_joints.get(name, []):
            if child_j['child'] in links: add_link(body, child_j['child'], child_j)
        return body

    base = add_link(world, 'gripper_l_base_link')
    # In the official left wrapper, camera and base share the same parent-frame
    # RPY; camera is at z=0, base at z=+5.1 mm. Thus camera relative to base is -5.1 mm.
    camera = links['gripper_l_camera_link']; ci = camera['inertia']
    cb = ET.SubElement(base, 'body', {'name': 'camera_link', 'pos': '0 0 -.0051'})
    ET.SubElement(cb, 'inertial', {'pos': ' '.join(map(str, camera['pos'])), 'quat': ' '.join(map(str, quat(camera['rpy']))),
        'mass': str(camera['mass']), 'diaginertia': ' '.join(str(ci[k]) for k in ('ixx','iyy','izz'))})
    add_geom(cb, 'gripper_l_camera_link')

    eq = ET.SubElement(root, 'equality')
    ET.SubElement(eq, 'joint', {'name': 'source_outer_root_mimic', 'joint1': 'outer_joint1', 'joint2': 'inner_joint1', 'polycoef': '0 -1 0 0 0'})
    for lp in loops:
        side = 'inner' if 'inner' in lp['name'] else 'outer'
        ET.SubElement(eq, 'connect', {'name': f'{side}_spherical_loop', 'site1': f'loop_base_{side}', 'site2': f'loop_end_{side}'})
    act = ET.SubElement(root, 'actuator')
    drive = joints[next(k for k in joints if k.endswith('inner_joint1'))]['drive']
    lim = joints[next(k for k in joints if k.endswith('inner_joint1'))]['limit']
    ET.SubElement(act, 'position', {'name': 'inner_root_position', 'joint': 'inner_joint1',
        'kp': str(drive['stiffness']), 'kv': str(drive['damping']), 'ctrllimited': 'true',
        'ctrlrange': f"{lim['lower']:.12g} {lim['upper']:.12g}", 'forcelimited': 'true',
        'forcerange': f"{-lim['effort']:.12g} {lim['effort']:.12g}"})
    ET.indent(root, space='  ')
    path = out / 'g2_omnipicker_constrained.xml'
    path.write_text(ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n', encoding='utf-8')
    return path, mesh_info


def oid(m, kind, name):
    i = int(mujoco.mj_name2id(m, kind, name))
    if i < 0: raise KeyError(f'MuJoCo object missing: {name}')
    return i

def qadr(m, name): return int(m.jnt_qposadr[oid(m, mujoco.mjtObj.mjOBJ_JOINT, name)])
def vadr(m, name): return int(m.jnt_dofadr[oid(m, mujoco.mjtObj.mjOBJ_JOINT, name)])
def endpoint(m, d, side):
    a = d.site_xpos[oid(m, mujoco.mjtObj.mjOBJ_SITE, f'loop_base_{side}')]
    b = d.site_xpos[oid(m, mujoco.mjtObj.mjOBJ_SITE, f'loop_end_{side}')]
    return np.asarray(b - a)

def loop_measurement(m, d):
    vectors = {s: endpoint(m, d, s) for s in ('inner', 'outer')}
    return {
        'residual_m': {s: v.tolist() for s, v in vectors.items()},
        'residual_mm': {s: float(np.linalg.norm(v) * 1000) for s, v in vectors.items()},
        'max_residual_mm': max(float(np.linalg.norm(v) * 1000) for v in vectors.values()),
    }

def static_solve(m, d, q, seed):
    d.qpos[:] = 0.; d.qpos[qadr(m, 'inner_joint1')] = q; d.qpos[qadr(m, 'outer_joint1')] = -q
    names = [*PASSIVE['inner'], *PASSIVE['outer']]
    def setp(x):
        for n, v in zip(names, x): d.qpos[qadr(m, n)] = v

    # Source mimic expansion is a kinematic reference only. It is measured
    # separately from the passive coordinates solved by the spherical loops.
    source_mimic_q = {n: float(q * ratio) for n, ratio in MIMIC.items()}
    setp([source_mimic_q[n] for n in names]); mujoco.mj_forward(m, d)
    source_mimic_loops = loop_measurement(m, d)

    def residual(x):
        setp(x); mujoco.mj_forward(m, d)
        return np.concatenate([endpoint(m, d, s) for s in ('inner','outer')])
    sol = least_squares(residual, np.asarray(seed), bounds=(-2.,2.), xtol=1e-13, ftol=1e-13,
                        gtol=1e-13, max_nfev=300, diff_step=1e-5, x_scale='jac')
    setp(sol.x); mujoco.mj_forward(m, d)
    loops = loop_measurement(m, d)
    passive = {n: float(d.qpos[qadr(m,n)]) for n in names}
    limits = {}
    for n in ('inner_joint1','outer_joint1',*names):
        j = oid(m,mujoco.mjtObj.mjOBJ_JOINT,n); val = float(d.qpos[qadr(m,n)])
        limits[n] = {'q_rad':val,'lower_margin_rad':val-float(m.jnt_range[j,0]),'upper_margin_rad':float(m.jnt_range[j,1])-val}
    return {'q_driver_rad':float(q),'solver_success':bool(sol.success),'nfev':int(sol.nfev),
        'loop_residual_mm':loops['residual_mm'],
        'loop_residual_m':loops['residual_m'],
        'max_loop_residual_m':loops['max_residual_mm'] / 1000,
        'source_mimic_loop_residual_mm':source_mimic_loops['residual_mm'],
        'source_mimic_loop_residual_m':source_mimic_loops['residual_m'],
        'source_mimic_max_loop_residual_mm':source_mimic_loops['max_residual_mm'],
        'passive_q_rad':passive,'source_mimic_targets_rad':source_mimic_q,
        'passive_minus_source_mimic_rad':{n:passive[n]-source_mimic_q[n] for n in names},
        'joint_limits':limits,'solver_message':str(sol.message)}

def contacts(m,d):
    out=[]
    for i in range(d.ncon):
        c=d.contact[i]
        out.append({'geom1':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),
            'geom2':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)),
            'distance_mm':float(c.dist*1000)})
    return out

def run_static(m,out,samples):
    d=mujoco.MjData(m);mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
    qvalues=np.linspace(0.,-math.pi/4,samples);seed=np.zeros(6);rows=[];pairs={}
    for q in qvalues:
        row=static_solve(m,d,float(q),seed)
        seed=[row['passive_q_rad'][n] for n in (*PASSIVE['inner'],*PASSIVE['outer'])]
        row['mj_forward_collision_pairs']=contacts(m,d)
        for c in row['mj_forward_collision_pairs']:
            key=tuple(sorted((c['geom1'],c['geom2'])))
            pairs[key]=min(pairs.get(key,1e9),c['distance_mm'])
        rows.append(row)
    with (out/'static_scan.csv').open('w',newline='') as f:
        names=['inner_joint3','inner_joint4','inner_joint0','outer_joint3','outer_joint4','outer_joint0']
        w=csv.DictWriter(f,fieldnames=['q_driver_rad','inner_loop_mm','outer_loop_mm','max_loop_m','solver_success',*names]);w.writeheader()
        for r in rows:w.writerow({'q_driver_rad':r['q_driver_rad'],'inner_loop_mm':r['loop_residual_mm']['inner'],'outer_loop_mm':r['loop_residual_mm']['outer'],'max_loop_m':r['max_loop_residual_m'],'solver_success':r['solver_success'],**r['passive_q_rad']})
    (out/'static_scan.json').write_text(json.dumps(rows,indent=2)+'\n')
    tol=1e-6;failed=[r for r in rows if not r['solver_success'] or r['max_loop_residual_m']>tol]
    summary={'status':'PASS_STATIC' if not failed else 'BLOCKED_STATIC_CLOSURE_NOT_ESTABLISHED','sample_count':len(rows),
        'root_range_rad':[float(qvalues[0]),float(qvalues[-1]),],'numerical_threshold_m':tol,
        'threshold_note':'Local optimizer-residual screen only; not an AgiBot manufacturing or operating tolerance. A small residual failure at the closed endpoint is recorded, not treated alone as proof of mechanical infeasibility.',
        'max_loop_residual_mm':max(r['max_loop_residual_m'] for r in rows)*1000,
        'small_closed_endpoint_residual_mm':rows[0]['max_loop_residual_m']*1000,
        'samples_with_mm_scale_residual':sum(r['max_loop_residual_m'] >= .001 for r in rows),
        'first_sample_with_mm_scale_residual':next(({'q_driver_rad':r['q_driver_rad'],'max_loop_residual_mm':r['max_loop_residual_m']*1000} for r in rows if r['max_loop_residual_m'] >= .001),None),
        'numerical_screen_failed_samples':[{'q_driver_rad':r['q_driver_rad'],'max_loop_residual_mm':r['max_loop_residual_m']*1000} for r in failed[:20]],
        'max_passive_abs_angle_rad':max(abs(v) for r in rows for v in r['passive_q_rad'].values()),
        'max_source_mimic_difference_rad':max(abs(v) for r in rows for v in r['passive_minus_source_mimic_rad'].values()),
        'collision_pair_min_distance_mm':[{'geom1':a,'geom2':b,'min_dist_mm':v} for (a,b),v in sorted(pairs.items())],
        'explicit_collision_exclusions_added':0,'closed':rows[0],'open':rows[-1]}
    (out/'static_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,rows

def run_dynamics(m,out,rows,cycles):
    d=mujoco.MjData(m);mujoco.mj_resetData(m,d)
    closed=static_solve(m,d,0.,[rows[0]['passive_q_rad'][n] for n in (*PASSIVE['inner'],*PASSIVE['outer'])])
    if closed['max_loop_residual_m']>1e-6: raise RuntimeError('closed seed does not satisfy site constraints')
    mujoco.mj_forward(m,d);initial=d.qpos.copy()
    dt=float(m.opt.timestep);leg=1.25;steps=cycles*2*int(round(leg/dt));trace=[]
    maxres=maxvel=maxforce=maxpenetration=0.;names=('inner_joint1','outer_joint1',*PASSIVE['inner'],*PASSIVE['outer'])
    for i in range(steps):
        phase=(i*dt%(2*leg))/leg
        u=phase if phase<=1 else phase-1
        smooth=3*u*u-2*u*u*u
        target=(-math.pi/4*smooth) if phase<=1 else (-math.pi/4*(1-smooth))
        d.ctrl[0]=target
        mujoco.mj_step(m,d)
        loop={s:float(np.linalg.norm(endpoint(m,d,s))*1000) for s in ('inner','outer')}
        q={n:float(d.qpos[qadr(m,n)]) for n in names};qd={n:float(d.qvel[vadr(m,n)]) for n in names};qacc={n:float(d.qacc[vadr(m,n)]) for n in names}
        cont=contacts(m,d);pen=max([max(0.,-c['distance_mm']) for c in cont] or [0.])
        force=float(d.actuator_force[0]);iter_count=int(d.solver_niter[0]) if d.solver_niter.size else None
        maxres=max(maxres,*loop.values());maxvel=max(maxvel,*map(abs,qd.values()));maxforce=max(maxforce,abs(force));maxpenetration=max(maxpenetration,pen)
        trace.append({'step':i+1,'time_s':float(d.time),'target_inner_rad':target,'q_rad':q,'qvel_rad_s':qd,'qacc_rad_s2':qacc,
            'actuator_force_Nm':force,'actuator_saturated':abs(force)>=9.999,'loop_residual_mm':loop,
            'active_contacts':cont,'solver_iterations':iter_count})
    with (out/'dynamics_trace.jsonl').open('w') as f:
        for r in trace:f.write(json.dumps(r,separators=(',',':'))+'\n')
    fields=['step','time_s','target_inner_rad','actuator_force_Nm','actuator_saturated','solver_iterations','inner_loop_mm','outer_loop_mm']+[f'q_{n}' for n in names]+[f'qd_{n}' for n in names]+[f'qacc_{n}' for n in names]
    with (out/'dynamics_trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in trace:
            v={'step':r['step'],'time_s':r['time_s'],'target_inner_rad':r['target_inner_rad'],'actuator_force_Nm':r['actuator_force_Nm'],'actuator_saturated':r['actuator_saturated'],'solver_iterations':r['solver_iterations'],'inner_loop_mm':r['loop_residual_mm']['inner'],'outer_loop_mm':r['loop_residual_mm']['outer']}
            v.update({f'q_{k}':x for k,x in r['q_rad'].items()});v.update({f'qd_{k}':x for k,x in r['qvel_rad_s'].items()});v.update({f'qacc_{k}':x for k,x in r['qacc_rad_s2'].items()});w.writerow(v)
    travel={n:max(r['q_rad'][n] for r in trace)-min(r['q_rad'][n] for r in trace) for n in names}
    limit_violations=[]
    for r in trace:
        for n,x in r['q_rad'].items():
            j=oid(m,mujoco.mjtObj.mjOBJ_JOINT,n);lo,hi=m.jnt_range[j]
            if x<lo-1e-8 or x>hi+1e-8:limit_violations.append({'step':r['step'],'joint':n,'q':x,'range':[float(lo),float(hi)]})
    status='PASS_DYNAMICS' if maxres<1.0 and maxvel<10.0 and not limit_violations and all(travel[n]>.005 for n in names) and np.isfinite(d.qpos).all() and maxpenetration<2.0 else 'FAIL_DYNAMICS'
    result={'status':status,'mj_step_calls':steps,'cycles':cycles,'duration_s':float(d.time),'timestep_s':dt,
        'integrator':'implicitfast','solver':'Newton','solver_iterations_setting':int(m.opt.iterations),
        'max_loop_residual_mm':maxres,'max_joint_speed_rad_s':maxvel,'max_actuator_effort_Nm':maxforce,
        'max_convex_collision_penetration_proxy_mm':maxpenetration,'joint_travel_rad':travel,
        'source_joint_limit_violations':limit_violations,'passive_qpos_writes_after_initialization':0,
        'actuator':{'joint':'inner_joint1','kp_source':1000.,'kv_source':10.,'effort_cap_source_Nm':10.,'velocity_limit_source_rad_s':10.,'velocity_limit_monitor_only':True},
        'pass_criteria_note':'Local diagnostic criteria: full source root travel, all source ranges, finite state, <1 mm loop residual, <10 rad/s source velocity, <2 mm convex-proxy penetration; none are vendor-published performance tolerances.'}
    (out/'dynamics_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,d,trace

def render_states(m,rows,out):
    renderer=mujoco.Renderer(m,height=720,width=960); images=[]
    for idx,label in ((0,'closed'),(-1,'open')):
        d=mujoco.MjData(m);mujoco.mj_resetData(m,d);r=rows[idx];d.qpos[qadr(m,'inner_joint1')]=r['q_driver_rad'];d.qpos[qadr(m,'outer_joint1')]=-r['q_driver_rad']
        for n,v in r['passive_q_rad'].items():d.qpos[qadr(m,n)]=v
        mujoco.mj_forward(m,d);renderer.update_scene(d);im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((16,16),f'G2 OmniPicker - {label}',fill=(20,20,20));im.save(out/f'{label}.png');images.append(im.resize((480,360)))
    sheet=Image.new('RGB',(960,360),'white');sheet.paste(images[0],(0,0));sheet.paste(images[1],(480,0));sheet.save(out/'open_closed.png');renderer.close()

def run_viewer(m,out):
    import mujoco.viewer
    d=mujoco.MjData(m);mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d);target={'q':0.}
    def opening(_):target['q']=-math.pi/4
    def closing(_):target['q']=0.
    def toggle(_):target['q']=0. if abs(target['q'])>1e-8 else -math.pi/4
    with mujoco.viewer.launch_passive(m,d) as v:
        v.cam.lookat[:]=[0.,0.,.07];v.cam.distance=.30;v.cam.azimuth=135;v.cam.elevation=-20
        v.add_key_callback(ord('O'),opening);v.add_key_callback(ord('C'),closing);v.add_key_callback(ord(' '),toggle)
        last=0.
        while v.is_running():
            d.ctrl[0]=target['q'];mujoco.mj_step(m,d)
            if d.time-last>.25:
                residual={s:float(np.linalg.norm(endpoint(m,d,s))*1000) for s in ('inner','outer')}
                joints={n:float(d.qpos[qadr(m,n)]) for n in ('inner_joint1','outer_joint1',*PASSIVE['inner'],*PASSIVE['outer'])}
                print(f't={d.time:.2f}s target={target["q"]:.3f} q={joints} loop_mm={residual} contacts={d.ncon}',flush=True);last=d.time
            v.sync();time.sleep(m.opt.timestep*.5)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--samples',type=int,default=101);p.add_argument('--dynamics',action='store_true');p.add_argument('--cycles',type=int,default=3);p.add_argument('--viewer',action='store_true');a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);source=SOURCE_ROOT/XACRO_REL
    head=git(SOURCE_ROOT,'rev-parse','HEAD');status=git(SOURCE_ROOT,'status','--porcelain')
    if head!=PIN or status or sha256(source)!=XACRO_SHA:raise RuntimeError('Pinned source SHA/revision/clean-tree check failed')
    if mujoco.__version__!='3.3.6':raise RuntimeError(f'Expected MuJoCo 3.3.6, got {mujoco.__version__}')
    links,joints,loops=parse_source(source);xml,mesh_manifest=build_model(out,links,joints,loops);model=mujoco.MjModel.from_xml_path(str(xml));mujoco.mj_forward(model,mujoco.MjData(model))
    result={'attempt_id':'e2956566-cb0c-405e-8bc3-b11f631e677b','model_classification':'pinned G2 source geometry/topology; MuJoCo dynamics with source-derived drive, simulation-derived integration settings',
        'upstream_repository':'https://github.com/AgibotTech/genie_sim_robot_model','upstream_commit':head,'license':'Apache-2.0',
        'xacro_path':str(source),'xacro_sha256':sha256(source),'mujoco_version':mujoco.__version__,'python':sys.version,'numpy':np.__version__,
        'model_path':str(xml),'model_sha256':sha256(xml),'model_dimensions':{'nq':int(model.nq),'nv':int(model.nv),'njnt':int(model.njnt),'neq':int(model.neq),'nu':int(model.nu),'ngeom':int(model.ngeom)},
        'source_mapping':{'links':links,'joints':joints,'loop_joints':loops,
            'custom_spherical_loop_mapping':'connect constraints coincide the declared link1 and link2 attachment points; relative rotation remains free; source rpy preserved in the site frame but does not add orientation constraints',
            'outer_root_mimic':'outer_joint1 = -inner_joint1, one joint equality, no outer actuator',
            'six_passive_mimics':'reference values only; passive qpos are determined by the two loop constraints and are not independently actuated/equality locked'},
        'mesh_manifest':mesh_manifest,'simulation_derived':{'timestep_s':float(model.opt.timestep),'integrator':'implicitfast','solver':'Newton','iterations':int(model.opt.iterations),'explicit_exclusions_added':0}}
    static,rows=run_static(model,out,max(101,a.samples));result['static']=static
    render_states(model,rows,out)
    if a.dynamics and static['status']=='PASS_STATIC':
        dyn,d,tr=run_dynamics(model,out,rows,a.cycles);result['dynamics']=dyn
        if a.viewer and dyn['status']=='PASS_DYNAMICS':run_viewer(model,out)
    elif a.dynamics:result['dynamics']={'status':'NOT_RUN_STATIC_GATE_NOT_ESTABLISHED','mj_step_calls':0}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'static':static['status'],'max_loop_residual_mm':static['max_loop_residual_mm'],'model':str(xml),'dimensions':result['model_dimensions'],'dynamics':result.get('dynamics',{}).get('status','NOT_REQUESTED')},indent=2))

if __name__=='__main__':main()
