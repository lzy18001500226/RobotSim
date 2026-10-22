#!/usr/bin/env python3
"""Live native MuJoCo review of the exact tested URDF-generated model."""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
import mujoco.viewer
import numpy as np

import issue46_urdf_pipeline as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-run', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8781)
    ap.add_argument('--single-motor', action='store_true', help='review only the tested one-root model, not the seven-servo reference')
    args = ap.parse_args()
    path = args.model_run/('single_motor.xml' if args.single_motor else 'gripper.xml')
    evidence = json.loads((args.model_run/('provenance.json' if args.single_motor else 'result.json')).read_text())
    physical = json.loads((args.model_run/'single_motor_result.json').read_text()) if args.single_motor else evidence['physics']
    expected_status = 'SINGLE_MOTOR_DIAGNOSTIC_PHYSICS_PASS' if args.single_motor else 'PASS'
    if p.sha(path) != evidence['mjcf_sha256'] or physical['status'] != expected_status:
        raise ValueError('model must match the recorded physical-motion PASS')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    law = p.CandidateLaw()
    addrs = p.addresses(model)
    # Begin at the mapped STEP source, not a penetrating hard-limit pose.
    qlo,qhi=physical['command_interval_CAD'] if args.single_motor else (p.Q_CLOSED,p.Q_OPEN)
    source_ratio = -qlo/(qhi-qlo)
    if args.single_motor:
        import issue46_single_motor_travel as travel
        if model.nu!=1:raise ValueError('single motor model unexpectedly has extra actuators')
        initial=travel.targets(law,0.)[0]
    else:
        initial=law.targets(source_ratio)
    for name, value in initial.items():
        data.qpos[addrs[name]] = value
    mujoco.mj_forward(model, data)
    driven=['right_claw_joint'] if args.single_motor else p.ACTUATED
    aids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'servo_'+n) for n in driven}
    lock = threading.Lock()
    state = {'opening': source_ratio, 'command': source_ratio, 'model': str(path), 'model_sha256': p.sha(path),
             'derived_urdf': str(args.model_run/'X2-Ultra_omnipicker_CAD_EXPERIMENTAL.urdf'),
             'gravity': model.opt.gravity.tolist(), 'safety_stop': None,'actuator_count':model.nu,
             'mode':'ONE_ROOT_PASSIVE_EXPERIMENTAL' if args.single_motor else 'SEVEN_SERVO_REFERENCE_ONLY',
             'q_CAD_command_range':[qlo,qhi],'root_bias_compensation':bool(physical.get('root_bias_compensation',False))}
    page = '''<!doctype html><meta charset="utf-8"><title>URDF → MuJoCo review</title>
<style>body{font:18px system-ui;background:#18222b;color:#e9f0f5;max-width:850px;margin:40px auto}input{width:100%}pre{white-space:pre-wrap;font:14px monospace}strong{color:#ffcb68}</style>
<h1>官方 URDF + CAD 修正 → MuJoCo</h1>
<p><strong>SIMULATION_ONLY / UNVERIFIED_EXPERIMENTAL</strong></p>
<p>在独立 MuJoCo 3D 窗口中查看。重力开启，原网格和官方根关节限位保留。</p>
<p>0 = 根关节限位端（仍有间隙，不能称为完全闭合）；1 = 当前候选最大张开。</p>
<input id="opening" type="range" min="0" max="1" step="0.001"><p id="value"></p>
<p>缓慢跟随滑块；D 间隙 >1mm、穿透 >2mm、速度 >25 或根限位误差 >0.001rad 时停止。</p><pre id="status"></pre>
<script>let initialized=false;let s=document.querySelector('#opening');
s.oninput=()=>{document.querySelector('#value').textContent=s.value;fetch('/set?opening='+s.value)};
setInterval(async()=>{let r=await fetch('/status');let v=await r.json();if(!initialized){s.value=v.command;initialized=true}document.querySelector('#status').textContent=JSON.stringify(v,null,2)},300);</script>'''
    if args.single_motor:
        page=page.replace('官方 URDF + CAD 修正 → MuJoCo','单根电机 + 被动连杆 → MuJoCo')
        page=page.replace('0 = 根关节限位端（仍有间隙，不能称为完全闭合）；1 = 当前候选最大张开。',
                          '0 = 本次已验证闭端，1 = 已验证开端。参考夹面截面开度约 19.5–61.1 mm，并非完整闭合或厂商全行程。仅根电机接受控制，六个下游关节无伺服。')
        page=page.replace('根限位误差 >0.001rad','根限位误差 >0.00001rad')
        page=page.replace('<input id="opening"', '<p><a style="color:#85c9ff" href="/video" target="_blank">播放真实单电机物理开合录像</a></p><input id="opening"')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            code = 200
            if parsed.path == '/':
                payload, kind = page.encode(), 'text/html; charset=utf-8'
            elif parsed.path == '/status':
                with lock:
                    payload = json.dumps(state).encode()
                kind = 'application/json'
            elif parsed.path == '/video' and args.single_motor and (args.model_run/'single_motor_review.mp4').is_file():
                payload=(args.model_run/'single_motor_review.mp4').read_bytes()
                kind='video/mp4'
            elif parsed.path == '/set':
                try:
                    value = float(parse_qs(parsed.query)['opening'][0])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError()
                    with lock:
                        state['command'] = value
                    payload = b'{"ok":true}'
                except (ValueError, KeyError, IndexError):
                    code, payload = 400, b'{"error":"invalid opening"}'
                kind = 'application/json'
            else:
                code, payload, kind = 404, b'not found', 'text/plain'
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = [0, 0, .12], .34, 0, -12
            viewer.opt.geomgroup[1] = 0  # Display only; collision physics remains on.
            ratio = source_ratio
            while viewer.is_running():
                with lock:
                    command, stopped = state['command'], state['safety_stop']
                if stopped is None:
                    ratio += float(np.clip(command-ratio, -.002, .002))
                    qcommand=qlo+(qhi-qlo)*ratio
                    if not args.single_motor:
                        target = law.targets(ratio)
                        for n, a in aids.items():data.ctrl[a] = target[n]
                    for _ in range(5):
                        if args.single_motor:
                            force,_=travel.root_bias_compensation(model,data)
                            control=p.PHASE-qcommand+force/float(model.actuator_gainprm[0,0])
                            data.ctrl[0]=float(np.clip(control,*model.actuator_ctrlrange[0]))
                        mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        gap = max(p.gaps(model, data))*1000
                        penetration = max([max(0., -c.dist)*1000 for c in data.contact] or [0.])
                        qn, qw = float(data.qpos[addrs['right_claw_joint']]), float(data.qpos[addrs['R_hand_wide1_joint']])
                        limit_error = max(0., qn, -1-qn, -qw, qw-1)
                        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                            stopped = 'NONFINITE'
                        elif gap > 1 or penetration > 2 or limit_error > (1e-5 if args.single_motor else .001) or np.max(np.abs(data.qvel)) > 25 or any(w.number for w in data.warning):
                            stopped = 'RUNTIME_SAFETY_GATE'
                        if args.single_motor:
                            actual=p.PHASE-qn
                            if not -.085-1e-5<=actual<=.235+1e-5 or max(abs(v)for v in travel.coupled_residuals(model,data).values())>.01:
                                stopped='PASSIVE_COUPLING_OR_STATIC_INTERVAL_GATE'
                        with lock:
                            state.update(opening=ratio, root_rad=qn, root_limit_error_rad=limit_error,
                                         D_gap_mm=gap, penetration_mm=penetration, safety_stop=stopped, simulation_time_s=float(data.time))
                            if args.single_motor:
                                state.update(q_CAD_actual=p.PHASE-qn,q_CAD_command=qcommand,
                                             root_feedforward_Nm=force,motor_force_Nm=float(data.actuator_force[0]),
                                             root_servo_control=float(data.ctrl[0]),passive_actuator_count=0)
                        if stopped:
                            break
                viewer.sync()
                time.sleep(.005)
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
