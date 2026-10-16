"""Render full-body and hand-closeup videos from a saved Dex3 trace.

The renderer restores recorded qpos and calls mj_forward only; it never steps
physics or writes a state back to a physical episode.
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

from baseline import OUT, path_governance
from analyze_contact_mechanics import load_run, set_recorded_pose, sha256_file


VIEWS = {
    'full_body': {'lookat': [0.12, -0.10, 0.76], 'distance': 2.7,
                  'azimuth': 220., 'elevation': -15.},
    'hand_closeup': {'lookat': [0.37, -0.22, 0.93], 'distance': 0.68,
                     'azimuth': 120., 'elevation': -22.},
}


def render_view(model, rows, output_dir, view_name, camera_args):
    times = np.asarray([row['time'] for row in rows], dtype=float)
    if len(times) < 2:
        raise ValueError('saved trace needs at least two rows')
    fps = 30
    requested = np.arange(times[0], times[-1] + 1e-9, 1. / fps)
    indices = np.unique(np.clip(np.searchsorted(times, requested), 0, len(times) - 1))
    if indices[-1] != len(rows) - 1:
        indices = np.r_[indices, len(rows) - 1]

    video_path = output_dir / f'{view_name}.mp4'
    preview_path = output_dir / f'{view_name}_preview.png'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (960, 720))
    if not writer.isOpened():
        raise RuntimeError(f'could not open video writer: {video_path}')
    renderer = mj.Renderer(model, height=720, width=960)
    camera = mj.MjvCamera()
    camera.lookat[:] = camera_args['lookat']
    camera.distance = camera_args['distance']
    camera.azimuth = camera_args['azimuth']
    camera.elevation = camera_args['elevation']
    option = mj.MjvOption()
    option.geomgroup[:] = 1
    data = mj.MjData(model)
    preview_written = False
    try:
        for index in indices:
            row = rows[int(index)]
            set_recorded_pose(model, data, row)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"{view_name.replace('_', ' ')} | {row['stage']} | t={row['time']:.2f}s",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .58, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.putText(frame, f"hand support {row['hand_z']:.2f} N | table {row['table_z']:.2f} N",
                        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, .54, (15, 15, 15), 2, cv2.LINE_AA)
            writer.write(frame)
            if not preview_written or int(index) == len(rows) - 1:
                name = preview_path if not preview_written else output_dir / f'{view_name}_final.png'
                if not cv2.imwrite(str(name), frame):
                    raise RuntimeError(f'could not write preview: {name}')
                preview_written = True
    finally:
        writer.release()
        renderer.close()
    return {'video': video_path, 'preview': preview_path,
            'frames': int(len(indices)), 'duration_s': float(times[indices[-1]] - times[indices[0]])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir, result, rows, model = load_run(args.run_dir)
    out = Path(path_governance.create_unique_output_directory(
        OUT, prefix=f"saved-views-{result['run_id']}").wsl_path)
    generated = {}
    for view_name, camera in VIEWS.items():
        paths = render_view(model, rows, out, view_name, camera)
        generated[view_name] = {
            **{k: str(v) for k, v in paths.items()},
            'video_sha256': sha256_file(paths['video']),
            'preview_sha256': sha256_file(paths['preview']),
            'frames': paths['frames'], 'duration_s': paths['duration_s'],
            'camera': camera,
        }
    reproduction = (
        'MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue43-g1-physical-grasp-20261009/venv/bin/python '
        'scripts/research/g1_dex3_physical/render_saved_views.py '
        f'--run-dir {run_dir}')
    record = {
        'schema_version': 'robotsim.dex3.saved-view-render.v1',
        'source_run_id': result['run_id'],
        'source_result_sha256': sha256_file(run_dir / 'result.json'),
        'source_trace_sha256': sha256_file(run_dir / 'trace.jsonl'),
        'source_model_sha256': sha256_file(run_dir / 'model.xml'),
        'robot_model_sha256': result['source_model_sha256'],
        'robot_sim_sha': result['RobotSim_SHA'],
        'mujoco_version': mj.__version__,
        'mj_step_calls': 0,
        'mj_forward_calls': int(sum(view['frames'] for view in generated.values())),
        'reproduction_command': reproduction,
        'views': generated,
        'output_directory': path_governance.describe_output_path(out),
    }
    path_governance.write_new_text(out / 'result.json', json.dumps(record, indent=2) + '\n')
    manifest = path_governance.write_artifact_manifest(out)
    print(json.dumps({'output': record['output_directory'], 'views': generated,
                      'mj_step_calls': 0, 'manifest': manifest['manifest']}, indent=2))


if __name__ == '__main__':
    main()
