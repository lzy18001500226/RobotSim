#!/usr/bin/env python3
"""One bounded contact probe using the generated URDF-derived gripper MJCF.

Reuses the existing Issue 46 canonical bottle/table and fixture parameters.
Not a new gripper model. No object attachment, qpos replay, or collision edits.
"""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import issue46_urdf_pipeline as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-run', type=Path, required=True)
    args = ap.parse_args()
    source = args.model_run/'gripper.xml'
    provenance = json.loads((args.model_run/'result.json').read_text())
    if p.sha(source) != provenance['mjcf_sha256'] or provenance['physics']['status'] != 'PASS':
        raise ValueError('requires the unchanged, motion-tested URDF-derived model')
    tree = ET.parse(source)
    root = tree.getroot()
    # Only the fixture bottle has no explicit inertial. Auto leaves every
    # explicit official gripper inertial unchanged and computes the bottle's
    # compound inertia from the four existing canonical mass/shape definitions.
    root.find('compiler').set('inertiafromgeom', 'auto')
    world = root.find('worldbody')
    base = world.find('body')
    # Same prior right-gripper neck fixture: source neck point (0,0,.132)
    # at world (0,0,.203); local +X fixture lift is world +Z.
    R = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    base.set('quat', p.quat(R))
    base.set('pos', p.fmt(np.array([0., 0., .203])-R@np.array([0., 0., .132])))
    ET.SubElement(base, 'joint', name='fixture_lift', type='slide', axis='1 0 0', limited='true', range='0 .12')
    ET.SubElement(root.find('actuator'), 'position', name='fixture_lift_servo', joint='fixture_lift', kp='2000', kv='90', forcelimited='true', forcerange='-50 50')
    ET.SubElement(world, 'geom', name='table', type='plane', size='0 0 .02', rgba='.5 .5 .5 1', friction='1 .01 .001')
    bottle = ET.SubElement(world, 'body', name='canonical_bottle', pos='0 0 .1175')
    ET.SubElement(bottle, 'freejoint', name='bottle_free')
    # Unchanged canonical geometry from issue46_cad_driven_gripper.build.
    for name, kind, pos, size, mass in [
        ('body', 'cylinder', '0 0 -.04', '.035 .0775', .49),
        ('shoulder', 'ellipsoid', '0 0 .0535', '.035 .035 .026', .05),
        ('neck', 'cylinder', '0 0 .0855', '.018 .020', .02),
        ('cap', 'cylinder', '0 0 .1185', '.020 .0085', .01),
    ]:
        ET.SubElement(bottle, 'geom', name='bottle_'+name, type=kind, pos=pos, size=size,
                      mass=str(mass), friction='1.4 .02 .001', condim='4', rgba='.2 .55 .8 1')
    run = p.pg.create_unique_output_directory(p.REPO/'reviews/issue-46', prefix='urdf-derived-canonical-bottle', environment='wsl').native_path
    p.put(run, 'bottle_probe.xml', ET.tostring(root, encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(run/'bottle_probe.xml'))
    source_model = mujoco.MjModel.from_xml_path(str(source))
    for bid in range(1, source_model.nbody):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        derived_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for attr in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
            if not np.allclose(getattr(source_model, attr)[bid], getattr(model, attr)[derived_id], rtol=0, atol=1e-12):
                raise AssertionError('canonical gripper inertial changed in the bottle fixture')
    data = mujoco.MjData(model)
    addrs = p.addresses(model)
    law = p.CandidateLaw()
    initial = law.targets(1.)
    for n, v in initial.items():
        data.qpos[addrs[n]] = v
    aids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'servo_'+n) for n in p.ACTUATED}
    lift = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fixture_lift_servo')
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'canonical_bottle')
    mujoco.mj_forward(model, data)
    z0 = float(data.xpos[bottle_id, 2])
    initial_contacts = [{'geom1': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                         'geom2': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                         'penetration_mm': max(0., -float(c.dist))*1000} for c in data.contact]
    if max([c['penetration_mm'] for c in initial_contacts] or [0.]) > 2.:
        result = {'status': 'PREFLIGHT_BLOCKED', 'steps': 0, 'simulation_time_s': 0.,
                  'stop_reason': 'INITIAL_CONTACT_PENETRATION_OVER_2MM', 'initial_contacts': initial_contacts,
                  'source_model': str(source), 'source_model_sha256': p.sha(source),
                  'bottle_qpos_writes_after_first_step': 0, 'fixture_lift_started': False,
                  'interpretation': 'old neck-grasp fixture is not cleared for this corrected model; no dynamics or grasp attempted'}
        p.put(run, 'result.json', result)
        p.pg.write_artifact_manifest(run, environment='wsl')
        print(json.dumps({'run': str(run), 'result': result}, indent=2))
        return
    rows, frames, stop = [], [], None
    stable_contact_s = 0.
    lift_started, lift_start_time = False, None
    renderer = mujoco.Renderer(model, height=640, width=800)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:], camera.distance, camera.azimuth, camera.elevation = [0, 0, .16], .55, 140, -18
    maxpen, maxgap, maxspeed = 0., 0., 0.
    for step in range(7500):
        t = step*.001
        ratio = 1. if t < .5 else .5+.5*np.cos(np.pi*min((t-.5)/2.5, 1.))
        target = law.targets(float(ratio))
        for n, aid in aids.items():
            data.ctrl[aid] = target[n]
        data.ctrl[lift] = .03*min(max((t-lift_start_time)/1.5, 0), 1.) if lift_started else 0.
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        forces = {'narrow': 0., 'wide': 0., 'table': 0.}
        contact_count = 0
        for cidx, c in enumerate(data.contact):
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if bottle_id not in (b1, b2):
                continue
            other = c.geom2 if b1 == bottle_id else c.geom1
            bodyname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other]) or ''
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, cidx, force)
            for s in p.GROUPS:
                if bodyname == f'R_hand_{s}3_Link':
                    forces[s] += max(0., float(force[0]))
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) == 'table':
                forces['table'] += max(0., float(force[0]))
            contact_count += 1
        bilateral = min(forces['narrow'], forces['wide']) >= .5
        stable_contact_s = stable_contact_s+.001 if bilateral else 0.
        if not lift_started and t >= 3.5 and stable_contact_s >= .5:
            lift_started, lift_start_time = True, t
        pen = max([max(0., -c.dist) for c in data.contact] or [0.])
        gap = max(p.gaps(model, data))
        speed = float(np.max(np.abs(data.qvel)))
        maxpen, maxgap, maxspeed = max(maxpen, pen), max(maxgap, gap), max(maxspeed, speed)
        qn, qw = float(data.qpos[addrs['right_claw_joint']]), float(data.qpos[addrs['R_hand_wide1_joint']])
        limerr = max(0., qn, -1-qn, -qw, qw-1)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            stop = 'NONFINITE'
        elif speed > 25:
            stop = 'VELOCITY_LIMIT'
        elif gap > .001:
            stop = 'D_GAP_OVER_1MM'
        elif pen > .002:
            stop = 'PENETRATION_OVER_2MM'
        elif limerr > .001:
            stop = 'ROOT_LIMIT_OVER_1MRAD'
        elif any(int(w.number) for w in data.warning):
            stop = 'MUJOCO_WARNING'
        elif t >= 5 and not lift_started:
            stop = 'BILATERAL_0P5N_FOR_0P5S_NOT_REACHED'
        if step % 10 == 0 or stop:
            rows.append({'step': step+1, 'time_s': float(data.time), 'opening_command': float(ratio), 'bottle_z': float(data.xpos[bottle_id, 2]),
                         'bottle_rise_m': float(data.xpos[bottle_id, 2])-z0, 'narrow_normal_N': forces['narrow'], 'wide_normal_N': forces['wide'],
                         'table_normal_N': forces['table'], 'bilateral_duration_s': stable_contact_s, 'lift_command_m': float(data.ctrl[lift]),
                         'D_gap_mm': gap*1000, 'max_penetration_mm': pen*1000, 'root_limit_error_rad': limerr,
                         **{n: float(data.qpos[a]) for n, a in addrs.items()}})
        if step % 100 == 0 or stop:
            renderer.update_scene(data, camera=camera, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            frames.append(frame.copy())
            if step in (0, 3000) or stop:
                frame.save(run/f'contact_{step+1:05d}.png')
        if stop:
            break
    renderer.close()
    frames[0].save(run/'bottle_contact.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    with (run/'contact_trace.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    passed = stop is None and lift_started and rows[-1]['bottle_rise_m'] >= .025 and rows[-1]['table_normal_N'] < .1 and min(rows[-1]['narrow_normal_N'], rows[-1]['wide_normal_N']) >= .5
    result = {'status': 'PASS' if passed else 'FAIL', 'stop_reason': stop, 'steps': step+1, 'simulation_time_s': float(data.time),
              'source_model': str(source), 'source_model_sha256': p.sha(source), 'bottle_mass_kg': .57,
              'bottle_qpos_writes_after_first_step': 0, 'bottle_attachment': False, 'gravity': model.opt.gravity.tolist(),
              'collision_exclusions_added': [], 'fixture_lift_started': lift_started, 'max_penetration_mm': maxpen*1000,
              'max_D_gap_mm': maxgap*1000, 'max_speed': maxspeed, 'final': rows[-1],
              'interpretation': 'one bounded canonical bottle attempt; no controller/friction/geometry tuning or retry'}
    p.put(run, 'result.json', result)
    p.pg.write_artifact_manifest(run, environment='wsl')
    print(json.dumps({'run': str(run), 'result': result}, indent=2))


if __name__ == '__main__':
    main()
