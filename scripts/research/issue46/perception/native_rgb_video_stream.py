"""Read-only first-person H.264 observer for the existing native MuJoCo run.

Private rendering data only: no mj_step, ROS publisher, actuator or MC access.
Slow viewers/encoders never backpressure the physical controller or sensor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>X2 第一视角 · RGB 视频流</title><style>
body{background:#101a25;color:#e8f2ff;font:17px system-ui;margin:24px}
video{width:min(100%,1100px);height:auto;background:#000;image-rendering:auto}
#state{font-size:22px;padding:12px;background:#263544}small{color:#adc0d3}
</style><h1>X2 第一视角 RGB 视频流</h1><div id="state">连接中</div>
<video id="camera" src="/stream.mp4" autoplay muted controls playsinline></video>
<p><small>同一仿真实际状态的只读相机渲染。此页面不推进物理、不发送机器人指令。相机为现有虚拟标定，未验证真机标定。</small></p>
<script>
const v=document.querySelector('video'),s=document.querySelector('#state');
setInterval(async()=>{try{let r=await fetch('/status',{cache:'no-store'}),x=await r.json();
s.textContent=x.state+' | sim='+Number(x.sim_time_s||0).toFixed(3)+'s | RGB '+x.frames+' 帧 | '+(x.failure||'');
s.style.color=x.state==='LIVE'?'#7effb3':x.state==='REPLAY / NO PHYSICS'?'#ffdd84':'#ffb18e';
}catch(e){s.textContent='DISCONNECTED / 非实时';s.style.color='#ffb18e'}},250);
</script></html>"""

def source_state(mode, stamp, now, ended=False, failure=None):
    if failure:
        return "OBSERVER ERROR"
    if mode == "replay":
        return "REPLAY ENDED / NO PHYSICS" if ended else "REPLAY / NO PHYSICS"
    if ended:
        return "RUN ENDED / LAST FRAME"
    if stamp is None:
        return "WAITING FOR NATIVE STATE"
    age = (now-stamp)*1e-9
    return "LIVE" if 0 <= age <= .20 else "STALE / NOT LIVE"

class Frames:
    """One immutable latest RGB buffer; no growing video queue."""
    def __init__(self, mode):
        self.mode, self.lock = mode, threading.Lock()
        self.rgb, self.stamp, self.sim_time = None, None, None
        self.frames, self.ended, self.failure = 0, False, None
        self.rejected_frames = 0

    def put(self, rgb, row):
        with self.lock:
            self.rgb = rgb
            self.stamp = row["host_monotonic_ns"]
            self.sim_time = row["sim_time_s"]
            self.frames += 1
            self.failure = None

    def snapshot(self):
        with self.lock:
            return self.rgb, {"state": source_state(self.mode, self.stamp, time.monotonic_ns(),
                         self.ended, self.failure), "frames": self.frames,
                    "sim_time_s": self.sim_time, "source_host_ns": self.stamp,
                    "failure": self.failure, "mode": self.mode,
                    "head_transform_rejected_frames": self.rejected_frames,
                    "observer_only": True, "new_mj_step_calls": 0}

def replay_rows(path, fps, rate, stop):
    """Pace saved actual frames; never relabel replay as fresh measurements."""
    wall_start, sim_start, previous = time.monotonic(), None, -float("inf")
    for line in path.open():
        if stop.is_set():
            return
        row = json.loads(line)
        if sim_start is None or row["sim_time_s"] < previous:
            wall_start, sim_start, previous = time.monotonic(), row["sim_time_s"], -float("inf")
        if row["sim_time_s"]-previous < rate/fps:
            continue
        delay = wall_start+(row["sim_time_s"]-sim_start)/rate-time.monotonic()
        if delay > 0 and stop.wait(delay):
            return
        previous = row["sim_time_s"]
        yield row

def capture(args, frames, stop):
    import mujoco
    from scripts.research.issue46.perception import native_rgbd_grasp as vision
    from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
    renderer = None
    try:
        m, d, renderer, camera = vision.build_sensor(args.scene, args.cache)
        rows = replay_rows(args.replay, args.fps, args.replay_rate, stop) if args.replay else None
        previous = None
        while not stop.is_set():
            tick = time.monotonic()
            if rows is not None:
                try:
                    row = next(rows)
                except StopIteration:
                    frames.ended = True
                    return
            else:
                if (args.run/"live_review_finished.json").exists():
                    frames.ended = True
                try:
                    row = json.loads((args.run/"native_latest.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    stop.wait(.05)
                    continue
                key = (row["generation"], row["host_monotonic_ns"])
                if key == previous:
                    stop.wait(.02)
                    continue
                previous = key
            apply_recorded_native_frame(m, d, row, visual_kinematics_only=True)
            if row["bodies"].get("head_pitch_link"):
                import numpy as np
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")
                if np.linalg.norm(d.xpos[bid]-row["bodies"]["head_pitch_link"]["position_m"]) > .001:
                    # Preserve the same 1mm transform check as control RGB-D.
                    # A mismatched render frame is rejected, never published as
                    # LIVE; a later conforming source frame can resume display.
                    frames.failure = "CAMERA_HEAD_NATIVE_FK_MISMATCH"
                    frames.rejected_frames += 1
                    continue
            renderer.update_scene(d, camera=vision.CAMERA)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            frames.put(renderer.render().tobytes(), row)
            if rows is None:
                stop.wait(max(0., 1/args.fps-(time.monotonic()-tick)))
    except Exception as exc:
        frames.failure = str(exc)
    finally:
        if renderer is not None:
            renderer.close()

def encoder_command(executable, fps):
    return [executable, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x240",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-threads", "1", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps))),
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000", "-flush_packets", "1", "-f", "mp4", "pipe:1"]

def handler_class(frames, stop, args, ffmpeg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path in ("/", "/status"):
                content = PAGE.encode() if self.path == "/" else json.dumps(frames.snapshot()[1]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if self.path == "/" else "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            if self.path != "/stream.mp4":
                self.send_error(404)
                return
            process = subprocess.Popen(encoder_command(ffmpeg, args.fps), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            disconnected = threading.Event()
            def feed():
                try:
                    while not stop.is_set() and not disconnected.is_set():
                        tick = time.monotonic()
                        rgb, _ = frames.snapshot()
                        if rgb is not None:
                            process.stdin.write(rgb)
                            process.stdin.flush()
                        disconnected.wait(max(0., 1/args.fps-(time.monotonic()-tick)))
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try: process.stdin.close()
                    except OSError: pass
            writer = threading.Thread(target=feed, daemon=True)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                writer.start()
                while not stop.is_set():
                    chunk = process.stdout.read(16384)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                disconnected.set()
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                process.stdout.close()
                writer.join(timeout=2)
    return Handler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--fps", type=int, default=10, choices=range(1,16))
    parser.add_argument("--replay-rate", type=float, default=1.)
    args = parser.parse_args()
    if not (args.port == 0 or 1024 <= args.port <= 65535) or not 0 < args.replay_rate <= 10:
        parser.error("bounded local stream parameters required")
    from scripts.agent import path_governance as pg
    import imageio_ffmpeg
    # Existing governed evidence directory; no implicit location substitution.
    root = Path("/home/lzy18001500226/projects/RobotSim/reviews").resolve()
    out = args.output_dir.resolve(strict=True)
    if not out.is_relative_to(root):
        raise ValueError("stream evidence must remain under canonical reviews")
    mode = "replay" if args.replay else "live"
    frames, stop = Frames(mode), threading.Event()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    config = {"classification": "READ_ONLY_FIRST_PERSON_H264_VIDEO_STREAM",
              "mode": mode, "scene": str(args.scene), "cache": str(args.cache),
              "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ffmpeg": ffmpeg, "ffmpeg_sha256": hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest(),
              "resolution": [320,240], "fps": args.fps, "url": f"http://localhost:{args.port}/",
              "new_mj_step_calls": 0, "no_control_publishers": True,
              "queue_policy": "LATEST_FRAME_ONLY_INDEPENDENT_OBSERVER"}
    worker = threading.Thread(target=capture, args=(args,frames,stop), daemon=True)
    server = ThreadingHTTPServer(("127.0.0.1",args.port), handler_class(frames,stop,args,ffmpeg))
    config["url"] = f"http://localhost:{server.server_address[1]}/"
    config["bound_port"] = server.server_address[1]
    pg.write_new_text(out/"first_person_stream_configuration.json", json.dumps(config,indent=2)+"\n")
    server.timeout = .25
    worker.start()
    print(json.dumps(config),flush=True)
    began, ended_at = time.monotonic(), None
    try:
        while not (out/"first_person_stream_stop").exists():
            server.handle_request()
            if frames.ended and ended_at is None:
                ended_at = time.monotonic()
            if time.monotonic()-began > 1800 or (ended_at and time.monotonic()-ended_at > 120):
                break
    finally:
        stop.set();server.server_close();worker.join(timeout=3)
        (out/"first_person_stream_final.json").write_text(json.dumps(frames.snapshot()[1],indent=2)+"\n")

if __name__ == "__main__":
    main()
