#!/usr/bin/env python3
"""Run the verified native X11 Reset -> STAND_DEFAULT initialization headlessly.

This is research tooling for the frozen X2-OmniPicker-STL scene.  It creates
an isolated Docker runtime with a temporary Xvfb server *inside* the
container, starts the delivered simulator and MC, pre-arms ``STAND_DEFAULT``,
then sends exactly one process-verified MuJoCo Backspace Reset through XTest.
No qpos is written, and no walking, arm, or hand command is sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
IMAGE = "robotsim:humble-raicom-runtime-20261019"
MC_DUMP = "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump"
MC_GENERATED = "mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim"
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from scripts.research.issue46.native_runtime import stl_hand_startup  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, timeout_s: float, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    if check and result.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(command) + "\n" + result.stdout[-4000:])
    return result


def wait_for(path: Path, marker: str, *, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if marker in text:
                return text
        time.sleep(0.15)
    raise RuntimeError(f"timed out waiting for {marker!r}: {text[-1200:]}")


def require_hashed(path: Path, expected: str, *, label: str) -> Path:
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"{label} is missing, unsafe, or hash-mismatched")
    return path.resolve()


def prearm_script() -> str:
    return """#!/usr/bin/env python3
import json
import time
import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import CommonRequest, McActionCommand, RequestHeader
from aimdk_msgs.srv import GetMcAction, SetMcAction

def call(node, client, request):
    if not client.wait_for_service(timeout_sec=5.0): raise RuntimeError('service unavailable')
    future=client.call_async(request); rclpy.spin_until_future_complete(node,future,timeout_sec=5.0)
    if not future.done() or future.result() is None: raise RuntimeError('service call timed out')
    return future.result()
rclpy.init(); node=Node('robotsim_xvfb_native_prearm')
try:
    set_client=node.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction')
    request=SetMcAction.Request(); request.header=RequestHeader(); request.header.stamp=node.get_clock().now().to_msg(); request.source='node'; request.command=McActionCommand(); request.command.action_desc='STAND_DEFAULT'
    set_result=call(node,set_client,request)
    get_client=node.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction'); samples=[]
    for _ in range(10):
        get_request=GetMcAction.Request(); get_request.request=CommonRequest(); get_request.request.header.stamp=node.get_clock().now().to_msg(); get_result=call(node,get_client,get_request)
        sample={'mode_name':get_result.info.action_desc,'mode_status':int(get_result.info.status.value)}; samples.append(sample)
        if sample['mode_name']=='STAND_DEFAULT': break
        time.sleep(0.3)
    print(json.dumps({'set_status':int(set_result.response.status.value),'set_message':set_result.response.message,'mode_name':samples[-1]['mode_name'],'mode_status':samples[-1]['mode_status'],'mode_samples':samples}))
finally:
    node.destroy_node(); rclpy.shutdown()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-overlay-result", required=True, type=Path)
    parser.add_argument("--table-overlay-result-sha256", required=True)
    parser.add_argument("--reset-helper", required=True, type=Path)
    parser.add_argument("--reset-helper-sha256", required=True)
    parser.add_argument("--xvfb-root", required=True, type=Path)
    parser.add_argument("--xvfb-libs", required=True, type=Path)
    parser.add_argument("--xkb-root", required=True, type=Path)
    args = parser.parse_args()

    table_result = require_hashed(args.table_overlay_result, args.table_overlay_result_sha256, label="table overlay result")
    reset_helper = require_hashed(args.reset_helper, args.reset_helper_sha256, label="native reset helper")
    for label, path in (("xvfb root", args.xvfb_root), ("xvfb libraries", args.xvfb_libs), ("XKB root", args.xkb_root)):
        if not path.is_absolute() or not path.is_dir() or path.is_symlink():
            raise ValueError(f"{label} is missing or unsafe")
    overlay = json.loads(table_result.read_text(encoding="utf-8"))
    required = ("runtime_scene", "simulator_config", "robot_config")
    generated = overlay.get("generated", {})
    if overlay.get("classification") != "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED":
        raise ValueError("table overlay classification is unexpected")
    if overlay.get("gates", {}).get("TABLE_SCENE_STATIC_COMPILE") != "PASS":
        raise ValueError("table overlay static check did not pass")
    inputs = {
        name: require_hashed(Path(str(generated.get(name, {}).get("path", ""))), str(generated.get(name, {}).get("sha256", "")), label=name)
        for name in required
    }
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-xvfb-native-reset-stand").native_path
    container = "robotsim-i46-xvfb-" + output.name[-15:].lower().replace("_", "-")
    result: dict[str, object] = {
        "schema": "robotsim.issue46.xvfb-native-reset-stand.v1",
        "classification": "SIMULATION_ONLY_NATIVE_RESET_AND_STAND_WITH_FROZEN_STL_HANDS",
        "mj_step_calls": "NOT_MEASURED_BY_STARTUP_EVIDENCE",
        "gates": {"SIMULATOR": "NOT_RUN", "MC": "NOT_RUN", "PREARM": "NOT_RUN", "NATIVE_RESET": "NOT_RUN", "POST_RESET_LEG_STATE": "NOT_RUN", "POST_RESET_STAND_STABILITY": "NOT_RUN"},
        "commands_not_sent": ["locomotion velocity", "arm trajectory", "hand command", "grasp"],
    }
    try:
        for source in inputs.values():
            pg.write_new_bytes(output / source.name, source.read_bytes())
        staged_dump = output / "mc_sim.yaml.dump"
        pg.write_new_bytes(staged_dump, (RUNTIME / MC_DUMP).read_bytes())
        staged_seed = output / "sim_seed"
        shutil.copytree(RUNTIME / MC_GENERATED, staged_seed)
        pg.write_new_bytes(output / "native_mujoco_reset.py", reset_helper.read_bytes())
        pg.write_new_text(output / "native_prearm.py", prearm_script())
        pg.write_new_text(output / "native_stand_sample.py", stl_hand_startup.build_stand_client_script())
        pg.write_new_text(output / "native_arm_state_capture.py", stl_hand_startup.build_arm_state_capture_script())
        for script in ("native_mujoco_reset.py", "native_prearm.py", "native_stand_sample.py", "native_arm_state_capture.py"):
            (output / script).chmod(0o755)
        pg.write_new_text(output / "launch_sim.sh", stl_hand_startup.build_launch_script(inputs["simulator_config"].name))
        pg.write_new_text(output / "launch_mc.sh", stl_hand_startup.build_mc_launch_script())
        (output / "launch_sim.sh").chmod(0o755); (output / "launch_mc.sh").chmod(0o755)
        command = [
            "docker", "run", "-d", "--pull=never", "--name", container, "--network=bridge", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770",
            "--tmpfs", "/workspace/raicom2026/" + MC_GENERATED + ":rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0755,size=8m",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026,readonly",
            "--mount", f"type=bind,source={EVIDENCE_REPO},target=/workspace/robotsim,readonly",
            "--mount", f"type=bind,source={output},target=/workspace/evidence",
            "--mount", f"type=bind,source={args.xvfb_root},target=/opt/xvfb,readonly",
            "--mount", f"type=bind,source={args.xvfb_libs},target=/tmp/xvfb-libs,readonly",
            "--mount", f"type=bind,source={args.xvfb_root / 'usr/bin/xkbcomp'},target=/usr/bin/xkbcomp,readonly",
            "--mount", f"type=bind,source={args.xkb_root},target=/usr/share/X11/xkb,readonly",
            "--mount", f"type=bind,source={staged_dump},target=/workspace/raicom2026/{MC_DUMP}",
            "-e", "DISPLAY=:99", "-e", "LD_LIBRARY_PATH=/tmp/xvfb-libs", "-e", "LIBGL_ALWAYS_SOFTWARE=1", "-e", "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe",
            "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20", "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        pg.write_new_text(output / "docker_command.json", json.dumps(command, indent=2) + "\n")
        run(command, timeout_s=12.0)
        run(["docker", "exec", "-d", container, "bash", "-lc", "/opt/xvfb/usr/bin/Xvfb :99 -screen 0 1706x1066x24 -ac -nolisten tcp > /workspace/evidence/xvfb.log 2>&1"], timeout_s=5.0)
        run(["docker", "exec", container, "bash", "-lc", "sleep 0.4; test -S /tmp/.X11-unix/X99; cp /workspace/evidence/sim_seed/action_setting.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_setting.yaml; cp /workspace/evidence/sim_seed/action_ruler.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_ruler.yaml; cp /workspace/evidence/sim_seed/action_state.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_state.yaml"], timeout_s=5.0)
        run(["docker", "exec", "-d", container, "bash", "-lc", "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"], timeout_s=5.0)
        wait_for(output / "mujoco-console.log", "MujocoSimModule Started Succeeded", timeout_s=25.0)
        result["gates"]["SIMULATOR"] = "PASS"
        run(["docker", "exec", "-d", container, "bash", "-lc", "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1"], timeout_s=5.0)
        wait_for(output / "mc-console.log", "MotionControlModule Started Succeeded", timeout_s=25.0)
        result["gates"]["MC"] = "PASS"
        environment = "source /opt/ros/humble/setup.bash; source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash; "
        prearm = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_prearm.py"], timeout_s=12.0, check=False)
        pg.write_new_text(output / "prearm.stdout", prearm.stdout)
        prearm_data = json.loads(prearm.stdout.strip().splitlines()[-1]) if prearm.returncode == 0 else {"returncode": prearm.returncode}
        result["prearm"] = prearm_data
        if prearm.returncode != 0 or prearm_data.get("mode_name") != "STAND_DEFAULT":
            raise RuntimeError(f"STAND_DEFAULT prearm failed: {prearm.stdout[-1200:]}")
        result["gates"]["PREARM"] = "PASS"
        pid = int(run(["docker", "exec", container, "bash", "-lc", "pgrep -x soc_sim"], timeout_s=4.0).stdout.strip().splitlines()[0])
        inspect = run(["docker", "exec", container, "bash", "-lc", f"python3 /workspace/evidence/native_mujoco_reset.py inspect --expected-pid {pid}"], timeout_s=8.0)
        inspected = json.loads(inspect.stdout); windows = inspected.get("windows", [])
        if len(windows) != 1:
            raise RuntimeError(f"expected one verified MuJoCo window: {inspect.stdout}")
        window = windows[0]
        reset = run(["docker", "exec", container, "bash", "-lc", "python3 /workspace/evidence/native_mujoco_reset.py send-reset --expected-pid " + str(pid) + " --expected-title " + repr(str(window["title"])) + " --window-id " + str(window["window_id"]) + " --before-png /workspace/evidence/mujoco_before_reset.png --after-png /workspace/evidence/mujoco_after_reset.png --dispatch-event-json /workspace/evidence/reset_dispatch_event.json"], timeout_s=12.0)
        pg.write_new_text(output / "reset_dispatch.stdout", reset.stdout)
        reset_data = json.loads(reset.stdout)
        result["reset"] = reset_data
        if reset_data.get("event", {}).get("status") != "KEY_DISPATCHED":
            raise RuntimeError("native Reset was not dispatched")
        result["gates"]["NATIVE_RESET"] = "PASS"
        state = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 6s ros2 topic echo --once /aima/hal/joint/leg/state"], timeout_s=8.0)
        pg.write_new_text(output / "post_reset_leg_state.yaml", state.stdout)
        count = state.stdout.count("- name:")
        if state.returncode != 0 or count != 12:
            raise RuntimeError(f"post-reset leg state is not complete: count={count}")
        result["post_reset_leg_joint_count"] = count
        result["gates"]["POST_RESET_LEG_STATE"] = "PASS"
        sample = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_stand_sample.py"], timeout_s=14.0, check=False)
        pg.write_new_text(output / "post_reset_stand_sample.stdout", sample.stdout)
        sample_data = json.loads(sample.stdout.strip().splitlines()[-1]) if sample.returncode == 0 else {"returncode": sample.returncode}
        result["post_reset_stand"] = sample_data
        if sample.returncode != 0 or not sample_data.get("stability", {}).get("pass"):
            raise RuntimeError(f"post-reset stand is unstable: {sample.stdout[-1200:]}")
        result["gates"]["POST_RESET_STAND_STABILITY"] = "PASS"
        capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_arm_state_capture.py"], timeout_s=10.0, check=False)
        pg.write_new_text(output / "native_arm_state_capture.stdout", capture.stdout)
        result["native_arm_state_capture"] = json.loads(capture.stdout.strip().splitlines()[-1]) if capture.returncode == 0 else {"returncode": capture.returncode}
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        subprocess.run(["docker", "stop", "--time", "5", container], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        subprocess.run(["docker", "rm", container], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
        pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": result["gates"], "failure": result.get("failure")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
