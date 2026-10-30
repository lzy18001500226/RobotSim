#!/usr/bin/env python3
"""One bounded startup check for the native RAICOM simulator STL-hand overlay.

This intentionally does not publish a motion, upper-body, or hand command.
It proves only that the delivered simulator accepts the separately generated
STL-hand scene and enables its compiled hand command/state components.  A
later bounded command readback is required before any balance or grasp claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
IMAGE = "robotsim:humble-raicom-runtime-20261019"
IMAGE_ID = "fed10384dea8"
MC_DUMP_RELATIVE = Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump")
SIM_READY_MARKER = "MujocoSimModule Started Succeeded"
HAND_MARKERS = ("hand_joint_command", "hand_joint_state")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


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


def exact_generated_inputs(overlay_result: Path) -> dict[str, Path]:
    result = json.loads(overlay_result.read_text(encoding="utf-8"))
    if result.get("classification") != "SIMULATION_ONLY_NATIVE_MC_HAND_BRIDGE_PREPARED":
        raise ValueError("not a native STL hand bridge preparation result")
    gates = result.get("gates", {})
    required = ("FROZEN_STL_MOUNT_INPUT", "HOST_STATIC_MODEL_COMPILE", "NATIVE_HAND_CHANNEL_CONFIGURATION")
    if any(gates.get(key) != "PASS" for key in required):
        raise ValueError("native STL hand overlay preflight did not pass")
    generated = result.get("generated")
    if not isinstance(generated, dict):
        raise ValueError("overlay result has no generated input manifest")
    resolved: dict[str, Path] = {}
    for label in ("runtime_scene", "simulator_config", "robot_config"):
        item = generated.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"generated overlay lacks {label}")
        path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"generated {label} is missing or hash-mismatched")
        resolved[label] = path
    return resolved


def build_launch_script(config_name: str) -> str:
    return f"""#!/bin/bash
# ROS setup scripts legitimately reference optional unset variables, so retain
# error/pipe checking but do not impose nounset on vendor setup code.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash
current_dir=/workspace/raicom2026/sim_mujoco/bin
install_dir=\"$current_dir/..\"
export LD_LIBRARY_PATH=\"$current_dir:$install_dir/bin:$install_dir/lib:${{LD_LIBRARY_PATH:-}}\"
export EM_APP_NAME=sim
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export AGIBOT_ENABLE_HDS_COMPONENT=false
export AGIBOT_ENABLE_EVENT_COMPONENT=false
export AGIBOT_ENABLE_MONITOR_COMPONENT=false
export AGIBOT_ENABLE_MONITOR=true
export AGIBOT_ENABLE_AUDIT_COMPONENT=false
export AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US=300
export SIM_ROBOT_PATH=/workspace/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5
export SIM_RESOURCE_MODEL_PATH=\"$install_dir/resource/model\"
mkdir -p /workspace/evidence/log /workspace/evidence/cfg/tmp
cd /workspace/evidence
exec \"$current_dir/aima-sim-app\" --cfg_file_path=/workspace/evidence/{config_name}
"""


def build_hand_probe_script(command_position: float = 0.6, hand_type: int = 2) -> str:
    """Return a self-contained ROS probe; it never writes MuJoCo qpos."""
    return """#!/usr/bin/env python3
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import HandCommand, HandCommandArray, HandType, HandStateArray


class Probe(Node):
    def __init__(self):
        super().__init__('robotsim_stl_hand_probe')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(HandCommandArray, '/aima/hal/joint/hand/command', qos)
        self.latest = {}
        self.hand_types = {}
        self.create_subscription(HandStateArray, '/aima/hal/joint/hand/state', self.on_state, qos)

    def on_state(self, message):
        self.hand_types = {'left': int(message.left_hand_type.value), 'right': int(message.right_hand_type.value)}
        for command in list(message.left_hands) + list(message.right_hands):
            canonical = {'L_claw_joint': 'left_claw_joint', 'R_claw_joint': 'right_claw_joint'}.get(command.name, command.name)
            self.latest[canonical] = float(command.position)

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def send_both(self, position, seconds=0.8):
        message = HandCommandArray()
        message.left_hand_type = HandType(value=__HAND_TYPE__)
        message.right_hand_type = HandType(value=__HAND_TYPE__)
        for name, side in (('left_claw_joint', 'left_hands'), ('right_claw_joint', 'right_hands')):
            command = HandCommand()
            command.name = name
            command.position = float(position)
            command.velocity = 1.0
            command.acceleration = 1.0
            command.deceleration = 1.0
            command.effort = 1.0
            getattr(message, side).append(command)
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            self.publisher.publish(message)
            self.spin_for(0.02)
        return dict(self.latest)


rclpy.init()
node = Probe()
try:
    node.spin_for(1.0)
    initial = dict(node.latest)
    closed = node.send_both(0.0)
    opened = node.send_both(__COMMAND_POSITION__)
    restored = node.send_both(0.0)
    required = ('left_claw_joint', 'right_claw_joint')
    target = __COMMAND_POSITION__
    response = all(name in opened and (opened[name] < -0.10 if target > 0.0 else abs(opened[name] - target) < 0.12) for name in required)
    restored_ok = all(name in restored and abs(restored[name]) < 0.08 for name in required)
    print(json.dumps({'initial': initial, 'closed': closed, 'opened': opened, 'restored': restored, 'state_hand_types': node.hand_types, 'publisher_subscription_count': node.publisher.get_subscription_count(), 'response_pass': response, 'restore_pass': restored_ok}, sort_keys=True))
finally:
    node.destroy_node()
    rclpy.shutdown()
""".replace("__COMMAND_POSITION__", f"{command_position:.17g}").replace("__HAND_TYPE__", str(hand_type))


def docker_container_name(output: Path) -> str:
    suffix = output.name.rsplit("-", 1)[-1].lower()
    return "robotsim-i46-stl-hand-" + suffix


def safe_cleanup(container: str) -> str:
    result = run(["docker", "rm", "-f", container], timeout_s=15.0, check=False)
    return result.stdout


def wait_for_log(path: Path, needles: tuple[str, ...], *, timeout_s: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if all(needle in text for needle in needles):
            return True, text
        time.sleep(0.2)
    return False, path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def host_preflight(container: str) -> dict[str, object]:
    if shutil.which("docker") is None:
        raise RuntimeError("docker executable is unavailable")
    active = run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"], timeout_s=5.0).stdout.splitlines()
    conflicts = [line for line in active if "aima" in line.lower() or "raicom" in line.lower() or "robotsim-i46" in line.lower()]
    if conflicts:
        raise RuntimeError("conflicting RAICOM/MuJoCo container is already active: " + "; ".join(conflicts))
    image = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], timeout_s=5.0).stdout.strip()
    if IMAGE_ID not in image:
        raise RuntimeError("the pinned RAICOM image is absent or has a different image id")
    if run(["docker", "ps", "-a", "--filter", f"name=^{container}$", "--format", "{{.Names}}"], timeout_s=5.0).stdout.strip():
        raise RuntimeError("the uniquely named output container already exists")
    if not Path("/tmp/.X11-unix").is_dir():
        raise RuntimeError("WSLg/X11 unix socket directory is unavailable")
    runtime_dump = RUNTIME / MC_DUMP_RELATIVE
    if not runtime_dump.is_file():
        raise RuntimeError("pinned MC runtime dump template is missing")
    return {"image": image, "active_container_conflicts": conflicts, "runtime_dump": str(runtime_dump), "runtime_dump_sha256": sha256(runtime_dump)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-result", required=True, type=Path)
    parser.add_argument("--startup-timeout-s", type=float, default=25.0)
    parser.add_argument("--run-hand-probe", action="store_true")
    parser.add_argument("--probe-command-position", type=float, default=0.6, help="diagnostic raw HandCommand.position; ordinary public command mapping is 0..1")
    parser.add_argument("--probe-hand-type", type=int, default=2, choices=(1, 2), help="diagnostic HandType; 2 is the public claw mode")
    args = parser.parse_args()
    if not args.overlay_result.is_file() or not 2.0 <= args.startup_timeout_s <= 35.0 or not -1.0 <= args.probe_command_position <= 1.0:
        raise SystemExit("overlay result and a startup timeout in [2,35] seconds are required")
    inputs = exact_generated_inputs(args.overlay_result)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-native-hand-startup").native_path
    container = docker_container_name(output)
    result: dict[str, object] = {
        "schema": "robotsim.issue46.raicom-stl-native-hand-startup.v1",
        "classification": "SIMULATION_ONLY_NATIVE_HAND_RUNTIME_STARTUP",
        "mj_step_calls": "NOT_MEASURED_BY_STARTUP_CHECK",
        "overlay_result": {"path": str(args.overlay_result), "sha256": sha256(args.overlay_result)},
        "generated_inputs": {label: {"path": str(path), "sha256": sha256(path)} for label, path in inputs.items()},
        "container": container,
        "gates": {"HOST_PREFLIGHT": "NOT_RUN", "NATIVE_SIMULATOR_MODEL_LOAD": "NOT_RUN", "HAND_COMMAND_SUBSCRIBER": "NOT_RUN", "HAND_STATE_PUBLISHER": "NOT_RUN", "HAL_HAND_COMMAND_ROUNDTRIP": "NOT_RUN", "MC": "NOT_RUN", "DYNAMICS_QUALIFICATION": "NOT_RUN"},
        "limitations": ["No arm, action, locomotion, or MC command is published by this check.", "No balance, physical contact, or grasp conclusion is permitted from a simulator hand readback."],
    }
    try:
        host = host_preflight(container)
        result["host_preflight"] = host
        result["gates"]["HOST_PREFLIGHT"] = "PASS"
        for label, source in inputs.items():
            destination = output / source.name
            pg.write_new_bytes(destination, source.read_bytes())
        runtime_dump = Path(str(host["runtime_dump"]))
        staged_dump = output / "mc_sim.yaml.dump"
        pg.write_new_bytes(staged_dump, runtime_dump.read_bytes())
        launch = output / "launch_sim.sh"
        pg.write_new_text(launch, build_launch_script(inputs["simulator_config"].name))
        launch.chmod(0o755)
        if args.run_hand_probe:
            probe = output / "native_hand_probe.py"
            pg.write_new_text(probe, build_hand_probe_script(args.probe_command_position, args.probe_hand_type))
        command = [
            "docker", "run", "-d", "--pull=never", "--name", container,
            "--network=bridge", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770,size=64m",
            "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026,readonly",
            "--mount", f"type=bind,source={EVIDENCE_REPO},target=/workspace/robotsim,readonly",
            "--mount", f"type=bind,source={output},target=/workspace/evidence",
            "--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly",
            "--mount", "type=bind,source=" + str(staged_dump) + ",target=/workspace/raicom2026/" + str(MC_DUMP_RELATIVE),
            "-e", "DISPLAY=:0", "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20", "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        started = run(command, timeout_s=10.0)
        pg.write_new_text(output / "container_id.txt", started.stdout)
        run(["docker", "exec", "-d", container, "bash", "-lc", "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"], timeout_s=5.0)
        ready, console = wait_for_log(output / "mujoco-console.log", (SIM_READY_MARKER, *HAND_MARKERS), timeout_s=args.startup_timeout_s)
        pg.write_new_text(output / "mujoco-console.tail.txt", console[-12000:])
        result["gates"]["NATIVE_SIMULATOR_MODEL_LOAD"] = "PASS" if ready else "FAIL"
        result["gates"]["HAND_COMMAND_SUBSCRIBER"] = "PASS" if ready else "FAIL"
        result["gates"]["HAND_STATE_PUBLISHER"] = "PASS" if ready else "FAIL"
        result["simulator_log_markers"] = {marker: marker in console for marker in (SIM_READY_MARKER, *HAND_MARKERS)}
        if ready and args.run_hand_probe:
            probe = run([
                "docker", "exec", container, "bash", "-lc",
                "source /opt/ros/humble/setup.bash; source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash; timeout 12s python3 /workspace/evidence/native_hand_probe.py",
            ], timeout_s=15.0, check=False)
            pg.write_new_text(output / "native_hand_probe.stdout", probe.stdout)
            try:
                hand_probe = json.loads(probe.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                hand_probe = {"parse_error": True, "returncode": probe.returncode}
            result["hand_probe"] = hand_probe
            result["hand_probe_command_position"] = args.probe_command_position
            result["hand_probe_hand_type"] = args.probe_hand_type
            result["gates"]["HAL_HAND_COMMAND_ROUNDTRIP"] = "PASS" if probe.returncode == 0 and hand_probe.get("response_pass") and hand_probe.get("restore_pass") else "FAIL"
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["container_cleanup"] = safe_cleanup(container)
        pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
        pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": result["gates"], "failure": result.get("failure")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
