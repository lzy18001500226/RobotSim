"""Evaluate one shoulder-height grasp pose without advancing physics."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

import mujoco as mj
import numpy as np

from baseline import (ARM, OUT, SOURCE, SOURCE_REV, UPSTREAM, bid, indices, name,
                      path_governance, sha)
from analyze_contact_mechanics import (BOTTLE_BODY, BOTTLE_GEOMS, load_run,
                                      set_recorded_pose, sha256_file,
                                      solve_static_wrench)
from episode import SOURCE_LIMITS, ik
from plan import render


def inspect_pose(model, data):
    bottle_bid = bid(model, BOTTLE_BODY)
    bottle_R = data.xmat[bottle_bid].reshape(3, 3)
    bottle_pos = data.xpos[bottle_bid].copy()
    fingers = {'right_hand_thumb_2_link', 'right_hand_index_1_link',
               'right_hand_middle_1_link'}
    contacts = []
    global_min = None
    for i in range(data.ncon):
        contact = data.contact[i]
        g0, g1 = map(int, contact.geom)
        b0, b1 = map(int, model.geom_bodyid[[g0, g1]])
        geom_names = [name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in (g0, g1)]
        body_names = [name(model, mj.mjtObj.mjOBJ_BODY, b) for b in (b0, b1)]
        record = {'distance_m': float(contact.dist), 'geom_names': geom_names,
                  'body_names': body_names}
        if global_min is None or contact.dist < global_min['distance_m']:
            global_min = record
        if BOTTLE_BODY not in body_names or not (BOTTLE_GEOMS & set(geom_names)):
            continue
        finger_body = next((body for body in body_names if body in fingers), None)
        frame = np.asarray(contact.frame).reshape(3, 3)
        bottle_is_geom1 = b1 == bottle_bid
        normal_bottle = frame[0] * (1. if bottle_is_geom1 else -1.)
        contacts.append({**record, 'finger_body': finger_body,
            'point_world_m': contact.pos.tolist(),
            'point_bottle_local_m': (bottle_R.T @ (contact.pos - bottle_pos)).tolist(),
            'normal_on_bottle_world': normal_bottle.tolist(),
            'normal_vertical_component': float(normal_bottle[2])})
    return contacts, global_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--delta-z-m', type=float, default=.045)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not any(abs(args.delta_z_m - candidate) <= 1e-12 for candidate in (.045, .060, .070)):
        raise ValueError('only evidence-derived +0.045 m, +0.060 m and +0.070 m candidate heights are permitted')
    run_dir, run_result, rows, model = load_run(args.run_dir)
    samples = [row for row in rows if row['stage'] == 'GRIP_HOLD']
    if not samples:
        raise ValueError('run contains no GRIP_HOLD sample')
    row = samples[-1]
    data = mj.MjData(model)
    set_recorded_pose(model, data, row)
    arm_qpos, _ = indices(model, ARM)
    wrist = bid(model, 'right_wrist_yaw_link')
    before_wrist = data.xpos[wrist].copy()
    wrist_rotation = data.xmat[wrist].reshape(3, 3).copy()
    target_wrist = before_wrist + np.array([0., 0., args.delta_z_m])
    seed = data.qpos[arm_qpos].copy()
    arm_target, ik_result = ik(model, data, target_wrist, wrist_rotation, seed)
    data.qpos[arm_qpos] = arm_target
    mj.mj_forward(model, data)
    contacts, min_contact = inspect_pose(model, data)
    finger_contacts = {finger: [c for c in contacts if c['finger_body'] == finger]
                       for finger in ['right_hand_thumb_2_link',
                                      'right_hand_index_1_link',
                                      'right_hand_middle_1_link']}
    candidate_row = {
        'stage': 'SHOULDER_HEIGHT_CANDIDATE', 'time': row['time'],
        'joint_names': row['joint_names'], 'q': data.qpos[
            model.jnt_qposadr[model.actuator_trnid[:, 0]]].tolist(),
        'bottle_pos': data.xpos[bid(model, BOTTLE_BODY)].tolist(),
        'bottle_quat': data.xquat[bid(model, BOTTLE_BODY)].tolist(),
        'contacts': [{'other': c['finger_body'], 'geoms': c['geom_names'],
                      'point': c['point_world_m'],
                      'point_bottle_local_m': c['point_bottle_local_m']}
                     for c in contacts if c['finger_body'] is not None]}
    wrench = solve_static_wrench(model, [candidate_row],
                                 'SHOULDER_HEIGHT_CANDIDATE', row['time'])
    out = Path(path_governance.create_unique_output_directory(
        OUT, prefix=args.run_id).wsl_path)
    renderer_path = out / 'shoulder_height_candidate.png'
    render(model, data, renderer_path, np.array([.38, -.22, .95]), .72, 120, -23)
    qa, _ = indices(model, ARM)
    arm_joints = []
    for joint, value in zip(ARM, arm_target):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = model.jnt_range[jid]
        source = SOURCE_LIMITS[joint]
        arm_joints.append({'name': joint, 'q_rad': float(value),
                           'range_rad': [float(lo), float(hi)],
                           'within_compiled_limit': bool(lo <= value <= hi),
                           'source_range_rad': [source['lower'], source['upper']],
                           'within_source_limit': bool(source['lower'] <= value <= source['upper'])})
    thumb = bool(finger_contacts['right_hand_thumb_2_link'])
    opposing = bool(finger_contacts['right_hand_index_1_link'] or
                    finger_contacts['right_hand_middle_1_link'])
    no_deep_penetration = not any(c['distance_m'] < -.003 for c in contacts)
    result = {
        'schema_version': 'robotsim.dex3.grasp-height-static-candidate.v1',
        'run_id': args.run_id, 'source_run_id': run_result['run_id'],
        'source_run_result_sha256': sha256_file(run_dir / 'result.json'),
        'source_run_trace_sha256': sha256_file(run_dir / 'trace.jsonl'),
        'source_runtime_model_sha256': run_result['runtime_model_sha256'],
        'RobotSim_SHA': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        'source_revision': SOURCE_REV, 'source_model_sha256': sha(SOURCE),
        'mujoco_version': mj.__version__, 'physics_steps': 0,
        'reproduction_command': (
            f'MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python '
            f'scripts/research/g1_dex3_physical/evaluate_grasp_height_candidate.py '
            f'--run-id {shlex.quote(args.run_id)} --delta-z-m {args.delta_z_m:.3f} --run-dir {shlex.quote(str(run_dir))}'),
        'candidate': {'delta_wrist_z_m': args.delta_z_m,
            'target_wrist_position_world_m': target_wrist.tolist(),
            'achieved_wrist_position_world_m': data.xpos[wrist].tolist(),
            'ik': ik_result, 'arm_joints': arm_joints,
            'contact_counts_by_finger': {k: len(v) for k, v in finger_contacts.items()},
            'contacts': contacts, 'minimum_scene_contact': min_contact,
            'static_wrench_screen': wrench,
            'gates': {'thumb_contact': thumb,
                'opposing_index_or_middle_contact': opposing,
                'all_arm_joint_limits': all(j['within_compiled_limit'] and
                                            j['within_source_limit'] for j in arm_joints),
                'no_hand_bottle_penetration_beyond_3mm': no_deep_penetration,
                'static_wrench_with_source_effort_caps': wrench.get(
                    'with_source_joint_effort_caps', {}).get('feasible', False)}},
        'output_directory': path_governance.describe_output_path(out),
        'render': path_governance.describe_output_path(renderer_path)}
    path_governance.write_new_text(out / 'result.json', json.dumps(result, indent=2) + '\n')
    manifest = path_governance.write_artifact_manifest(out)
    print(json.dumps({'output_directory': result['output_directory'],
                      'manifest': manifest['manifest'],
                      'gates': result['candidate']['gates'],
                      'contacts': contacts,
                      'wrench_status': wrench.get('status'),
                      'physics_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
