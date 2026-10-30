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
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
IMAGE = "robotsim:humble-raicom-runtime-20261019"
IMAGE_ID = "fed10384dea8"
MC_DUMP_RELATIVE = Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump")
MC_GENERATED_RELATIVE = Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim")
MC_SIM_CONFIG_RELATIVE = Path("mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml")
SIM_READY_MARKER = "MujocoSimModule Started Succeeded"
HAND_MARKERS = ("hand_joint_command", "hand_joint_state")
MC_READY_MARKER = "MotionControlModule Started Succeeded"
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
    classification = result.get("classification")
    if classification not in {
        "SIMULATION_ONLY_NATIVE_MC_HAND_BRIDGE_PREPARED",
        "SIMULATION_ONLY_NATIVE_MC_HAND_TABLE_BRIDGE_PREPARED",
    }:
        raise ValueError("not a native STL hand bridge preparation result")
    gates = result.get("gates", {})
    required = ("FROZEN_STL_MOUNT_INPUT", "HOST_STATIC_MODEL_COMPILE", "NATIVE_HAND_CHANNEL_CONFIGURATION")
    if any(gates.get(key) != "PASS" for key in required):
        raise ValueError("native STL hand overlay preflight did not pass")
    if classification == "SIMULATION_ONLY_NATIVE_MC_HAND_TABLE_BRIDGE_PREPARED" and gates.get("TABLE_SCENE_STATIC_COMPILE") != "PASS":
        raise ValueError("native STL table overlay static preflight did not pass")
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


def build_mc_launch_script() -> str:
    return """#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash
export LOG_PATH=/workspace/evidence/mc_logs
mkdir -p "$LOG_PATH"
cd /workspace/raicom2026/mc/bin
exec ./em_run_wsl2.sh
"""


def build_arm_split_mc_config(source: dict[str, object]) -> dict[str, object]:
    """Disable only MC's arm HAL publisher in an evidence-local config copy."""
    import copy
    result = copy.deepcopy(source)
    module = result.get("MotionControlModule")
    if not isinstance(module, dict):
        raise ValueError("MC config lacks MotionControlModule")
    manager = module.get("PublisherManager")
    if not isinstance(manager, dict) or not isinstance(manager.get("publisher_cfg_list"), list):
        raise ValueError("MC config lacks publisher_cfg_list")
    matches = [item for item in manager["publisher_cfg_list"] if isinstance(item, dict) and item.get("name") == "arm_joint_command"]
    if len(matches) != 1 or matches[0].get("enable") is not True:
        raise ValueError("MC config does not expose exactly one enabled arm_joint_command publisher")
    matches[0]["enable"] = False
    return result


def build_stand_client_script() -> str:
    """Minimal equivalent of the delivered ``set_mc_action SD`` example."""
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import CommonRequest, McActionCommand, RequestHeader
from aimdk_msgs.srv import GetMcAction, SetMcAction
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def call(node, client, request):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError('service unavailable')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    if not future.done() or future.result() is None:
        raise RuntimeError('service call timed out')
    return future.result()


rclpy.init()
node = Node('robotsim_stl_native_stand_request')
odom_samples = []
tilt_samples = []
def on_odom(message):
    odom_samples.append((float(message.pose.pose.position.x), float(message.pose.pose.position.y), float(message.pose.pose.position.z)))
def on_imu(message):
    q = message.orientation
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    tilt_samples.append(max(abs(roll), abs(pitch)))
state_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
node.create_subscription(Odometry, '/aima/hal/odom/state', on_odom, state_qos)
node.create_subscription(Imu, '/aima/hal/imu/torso/state', on_imu, state_qos)
try:
    set_client = node.create_client(SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')
    request = SetMcAction.Request()
    request.header = RequestHeader()
    request.header.stamp = node.get_clock().now().to_msg()
    request.source = 'node'
    request.command = McActionCommand()
    request.command.action_desc = 'STAND_DEFAULT'
    set_result = call(node, set_client, request)
    get_client = node.create_client(GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction')
    samples = []
    for _ in range(10):
        get_request = GetMcAction.Request()
        get_request.request = CommonRequest()
        get_request.request.header.stamp = node.get_clock().now().to_msg()
        get_result = call(node, get_client, get_request)
        sample = {'mode_name': get_result.info.action_desc, 'mode_status': int(get_result.info.status.value)}
        samples.append(sample)
        if sample['mode_name'] == 'STAND_DEFAULT':
            break
        time.sleep(0.3)
    deadline = time.monotonic() + 2.0
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
    if len(odom_samples) >= 2:
        first, last = odom_samples[0], odom_samples[-1]
        planar_drift = math.hypot(last[0] - first[0], last[1] - first[1])
    else:
        planar_drift = None
    max_tilt = max(tilt_samples) if tilt_samples else None
    stability = {'sample_count': len(odom_samples), 'planar_drift_m': planar_drift, 'max_tilt_rad': max_tilt, 'pass': samples[-1]['mode_name'] == 'STAND_DEFAULT' and planar_drift is not None and planar_drift <= 0.05 and max_tilt is not None and max_tilt <= math.radians(15.0)}
    print(json.dumps({'set_status': int(set_result.response.status.value), 'set_message': set_result.response.message, 'mode_name': samples[-1]['mode_name'], 'mode_status': samples[-1]['mode_status'], 'mode_samples': samples, 'stability': stability}))
finally:
    node.destroy_node()
    rclpy.shutdown()
"""


def build_upperbody_split_client_script() -> str:
    """Same bounded service/readback pattern for the delivered US mode."""
    return build_stand_client_script().replace(
        "robotsim_stl_native_stand_request", "robotsim_stl_native_upperbody_split_request"
    ).replace("STAND_DEFAULT", "UPPERBODY_REMOTE_SPLIT")


def build_hand_probe_script(command_position: float = 0.6, hand_type: int = 2) -> str:
    """Return a self-contained ROS probe; it never writes MuJoCo qpos."""
    return """#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import HandCommand, HandCommandArray, HandType, HandStateArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class Probe(Node):
    def __init__(self):
        super().__init__('robotsim_stl_hand_probe')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(HandCommandArray, '/aima/hal/joint/hand/command', qos)
        self.latest = {}
        self.hand_types = {}
        self.odom_samples = []
        self.tilt_samples = []
        self.create_subscription(HandStateArray, '/aima/hal/joint/hand/state', self.on_state, qos)
        state_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, state_qos)
        self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, state_qos)

    def on_state(self, message):
        self.hand_types = {'left': int(message.left_hand_type.value), 'right': int(message.right_hand_type.value)}
        for command in list(message.left_hands) + list(message.right_hands):
            canonical = {'L_claw_joint': 'left_claw_joint', 'R_claw_joint': 'right_claw_joint'}.get(command.name, command.name)
            self.latest[canonical] = float(command.position)

    def on_odom(self, message):
        self.odom_samples.append((float(message.pose.pose.position.x), float(message.pose.pose.position.y)))

    def on_imu(self, message):
        q = message.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        self.tilt_samples.append(max(abs(roll), abs(pitch)))

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
    if len(node.odom_samples) >= 2:
        first, last = node.odom_samples[0], node.odom_samples[-1]
        drift = math.hypot(last[0] - first[0], last[1] - first[1])
    else:
        drift = None
    motion_stability = {'sample_count': len(node.odom_samples), 'planar_drift_m': drift, 'max_tilt_rad': max(node.tilt_samples) if node.tilt_samples else None}
    motion_stability['pass'] = motion_stability['sample_count'] >= 100 and motion_stability['planar_drift_m'] is not None and motion_stability['planar_drift_m'] <= 0.05 and motion_stability['max_tilt_rad'] is not None and motion_stability['max_tilt_rad'] <= math.radians(15.0)
    print(json.dumps({'initial': initial, 'closed': closed, 'opened': opened, 'restored': restored, 'state_hand_types': node.hand_types, 'publisher_subscription_count': node.publisher.get_subscription_count(), 'response_pass': response, 'restore_pass': restored_ok, 'motion_stability': motion_stability}, sort_keys=True))
finally:
    node.destroy_node()
    rclpy.shutdown()
""".replace("__COMMAND_POSITION__", f"{command_position:.17g}").replace("__HAND_TYPE__", str(hand_type))


def build_upper_body_probe_script() -> str:
    """Small source-compatible arm command/readback probe during native stand."""
    return """#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointStateArray, UpperBodyCommandArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

ARM_ORDER = ('left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint')

class Probe(Node):
    def __init__(self):
        super().__init__('robotsim_stl_upper_body_probe')
        self.arm = {}
        self.odom = []
        self.tilt = []
        state_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointStateArray, '/aima/hal/joint/arm/state', self.on_arm, state_qos)
        self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, state_qos)
        self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, state_qos)
        self.publisher = self.create_publisher(UpperBodyCommandArray, '/mc/upper_body_command', 10)

    def on_arm(self, message):
        for joint in message.joints:
            self.arm[joint.name] = float(joint.position)
    def on_odom(self, message):
        self.odom.append((float(message.pose.pose.position.x), float(message.pose.pose.position.y)))
    def on_imu(self, message):
        q = message.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        self.tilt.append(max(abs(roll), abs(pitch)))
    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
    def publish_for(self, arm_pos, seconds):
        message = UpperBodyCommandArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'mc_upper_body'
        message.source = 'robotsim_upper_body_probe'
        # Do not command either hand during an arm compatibility probe.
        message.hand_sub_mode = 0
        message.head_pos = [0.0, 0.0]
        message.arm_pos = [float(value) for value in arm_pos]
        message.hand_pos = []
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            message.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(message)
            self.spin_for(0.02)

rclpy.init()
node = Probe()
try:
    node.spin_for(1.0)
    if any(name not in node.arm for name in ARM_ORDER):
        raise RuntimeError('missing named arm state')
    start = [node.arm[name] for name in ARM_ORDER]
    goal = list(start)
    goal[7] -= 0.03
    subscriptions = node.publisher.get_subscription_count()
    node.publish_for(goal, 2.0)
    moved = [node.arm.get(name) for name in ARM_ORDER]
    node.publish_for(start, 2.0)
    restored = [node.arm.get(name) for name in ARM_ORDER]
    observed_delta = None if moved[7] is None else moved[7] - start[7]
    restore_error = None if restored[7] is None else abs(restored[7] - start[7])
    if len(node.odom) >= 2:
        drift = math.hypot(node.odom[-1][0] - node.odom[0][0], node.odom[-1][1] - node.odom[0][1])
    else:
        drift = None
    stability = {'sample_count': len(node.odom), 'planar_drift_m': drift, 'max_tilt_rad': max(node.tilt) if node.tilt else None}
    stability['pass'] = stability['sample_count'] >= 100 and stability['planar_drift_m'] is not None and stability['planar_drift_m'] <= 0.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad'] <= math.radians(15.0)
    response = subscriptions >= 1 and observed_delta is not None and observed_delta <= -0.01 and restore_error is not None and restore_error <= 0.03
    print(json.dumps({'publisher_subscription_count': subscriptions, 'start_right_shoulder_pitch': start[7], 'goal_right_shoulder_pitch': goal[7], 'moved_right_shoulder_pitch': moved[7], 'restored_right_shoulder_pitch': restored[7], 'observed_delta_rad': observed_delta, 'restore_error_rad': restore_error, 'response_pass': response, 'motion_stability': stability}))
finally:
    node.destroy_node()
    rclpy.shutdown()
"""


def build_hal_arm_probe_script() -> str:
    """Bounded direct HAL wrist probe using the MC's current command fields."""
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

TARGET = 'right_wrist_roll_joint'

class Probe(Node):
    def __init__(self):
        super().__init__('robotsim_stl_hal_arm_probe')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.states, self.commands, self.odom, self.tilt = {}, {}, [], []
        self.create_subscription(JointStateArray, '/aima/hal/joint/arm/state', self.on_state, qos)
        self.create_subscription(JointCommandArray, '/aima/hal/joint/arm/command', self.on_command, qos)
        self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, qos)
        self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)
        self.publisher = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', qos)
    def on_state(self, msg):
        self.states = {item.name: float(item.position) for item in msg.joints}
    def on_command(self, msg):
        self.commands = {item.name: item for item in msg.joints}
    def on_odom(self, msg):
        self.odom.append((float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)))
    def on_imu(self, msg):
        q=msg.orientation; pitch=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x)))); roll=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(roll),abs(pitch)))
    def spin(self, seconds):
        end=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<end: rclpy.spin_once(self,timeout_sec=0.001)
    def publish(self, command_map, seconds):
        end=time.monotonic()+seconds; seq=0
        while rclpy.ok() and time.monotonic()<end:
            msg=JointCommandArray(); msg.header=MessageHeader(); now=self.get_clock().now(); msg.header.stamp=now.to_msg(); msg.header.frame_id='hal_arm_probe'; msg.header.sequence=seq
            for name, source in command_map.items():
                out=JointCommand(); out.name=name; out.position=float(source.position); out.velocity=float(source.velocity); out.effort=float(source.effort); out.stiffness=float(source.stiffness); out.damping=float(source.damping); msg.joints.append(out)
            self.publisher.publish(msg); rclpy.spin_once(self,timeout_sec=0.001); seq+=1; time.sleep(0.001)

rclpy.init(); node=Probe()
try:
    node.spin(1.0)
    if TARGET not in node.states or len(node.states) < 14: raise RuntimeError('missing complete arm state')
    start=float(node.states[TARGET]); observed_source=node.commands.get(TARGET); source_fields=None if observed_source is None else {'position':float(observed_source.position),'velocity':float(observed_source.velocity),'effort':float(observed_source.effort),'stiffness':float(observed_source.stiffness),'damping':float(observed_source.damping)}; command={}
    for name, position in node.states.items():
        item=JointCommand(); item.name=name; item.position=float(position); item.velocity=0.0; item.effort=0.0; item.stiffness=0.0; item.damping=0.0; command[name]=item
    baseline={name:item for name,item in command.items()}; target=JointCommand(); source=command[TARGET]; target.name=source.name; target.position=float(source.position)+0.03; target.velocity=0.0; target.effort=0.6; target.stiffness=0.0; target.damping=0.0; command[TARGET]=target
    node.publish(command,2.0); moved=float(node.states.get(TARGET,float('nan'))); node.publish(baseline,2.0); restored=float(node.states.get(TARGET,float('nan')))
    delta=moved-start; restore_error=abs(restored-start)
    drift=math.hypot(node.odom[-1][0]-node.odom[0][0],node.odom[-1][1]-node.odom[0][1]) if len(node.odom)>=2 else None
    stability={'sample_count':len(node.odom),'planar_drift_m':drift,'max_tilt_rad':max(node.tilt) if node.tilt else None}; stability['pass']=stability['sample_count']>=100 and drift is not None and drift<=0.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad']<=math.radians(15)
    print(json.dumps({'subscription_count':node.publisher.get_subscription_count(),'source_command_fields':source_fields,'start_rad':start,'target_rad':target.position,'command_effort_nm':target.effort,'moved_rad':moved,'restored_rad':restored,'delta_rad':delta,'restore_error_rad':restore_error,'response_pass':delta>=0.005 and delta<=0.08 and restore_error<=0.04,'motion_stability':stability}))
finally:
    node.destroy_node(); rclpy.shutdown()
"""


def load_source_arm_pd_parameters() -> dict[str, tuple[float, float]]:
    """Read the delivered remote-controller arm gains without inventing values."""
    config = RUNTIME / "mc/bin/mc_param/src/robot/lx2501_3_t2d5/remote_controller_config/model_config.yaml"
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    joints = payload.get("enabledJoints") if isinstance(payload, dict) else None
    if not isinstance(joints, list):
        raise RuntimeError("delivered remote-controller model config lacks enabledJoints")
    values: dict[str, tuple[float, float]] = {}
    for item in joints:
        if not isinstance(item, dict):
            continue
        name, stiffness, damping = item.get("name"), item.get("stiffness"), item.get("damping")
        if isinstance(name, str) and name.endswith(("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint", "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint")):
            try:
                gains = float(stiffness), float(damping)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid delivered arm gains for {name}") from exc
            if not all(math.isfinite(value) and value > 0.0 for value in gains):
                raise RuntimeError(f"non-positive delivered arm gains for {name}")
            values[name] = gains
    if len(values) != 14:
        raise RuntimeError(f"delivered remote-controller model config provides {len(values)} arm gain records, expected 14")
    return values


def build_hal_arm_pd_probe_script(gains: dict[str, tuple[float, float]]) -> str:
    """Bounded direct position/stiffness arm probe using delivered gains."""
    serialized = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in gains.items()}, sort_keys=True)
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

TARGET = 'right_shoulder_pitch_joint'
GAINS = __GAINS__

class Probe(Node):
    def __init__(self):
        super().__init__('robotsim_stl_hal_arm_pd_probe')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.states, self.odom, self.tilt = {}, [], []
        self.create_subscription(JointStateArray, '/aima/hal/joint/arm/state', self.on_state, qos)
        self.create_subscription(Odometry, '/aima/hal/odom/state', self.on_odom, qos)
        self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)
        self.publisher = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', qos)
    def on_state(self, msg): self.states = {item.name: float(item.position) for item in msg.joints}
    def on_odom(self, msg): self.odom.append((float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)))
    def on_imu(self, msg):
        q=msg.orientation; pitch=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x)))); roll=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(roll),abs(pitch)))
    def spin(self, seconds):
        end=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<end: rclpy.spin_once(self, timeout_sec=0.002)
    def publish(self, targets, seconds):
        end=time.monotonic()+seconds; seq=0
        while rclpy.ok() and time.monotonic()<end:
            msg=JointCommandArray(); msg.header=MessageHeader(); msg.header.stamp=self.get_clock().now().to_msg(); msg.header.frame_id='hal_arm_pd_probe'; msg.header.sequence=seq
            for name in sorted(targets):
                item=JointCommand(); item.name=name; item.position=float(targets[name]); item.velocity=0.0; item.effort=0.0; item.stiffness=float(GAINS[name][0]); item.damping=float(GAINS[name][1]); msg.joints.append(item)
            self.publisher.publish(msg); rclpy.spin_once(self, timeout_sec=0.002); seq+=1; time.sleep(0.01)

rclpy.init(); node=Probe()
try:
    node.spin(1.0)
    if set(GAINS)-set(node.states): raise RuntimeError('missing complete arm state')
    start={name:float(node.states[name]) for name in sorted(GAINS)}
    goal=dict(start); goal[TARGET]=start[TARGET]-0.03
    subscriptions=node.publisher.get_subscription_count()
    node.publish(goal,2.0); moved=float(node.states.get(TARGET,float('nan')))
    node.publish(start,2.0); restored=float(node.states.get(TARGET,float('nan')))
    # Explicitly hold the restored safe pose during the final observation.
    node.publish(start,0.5)
    delta=moved-start[TARGET]; restore_error=abs(restored-start[TARGET])
    drift=math.hypot(node.odom[-1][0]-node.odom[0][0],node.odom[-1][1]-node.odom[0][1]) if len(node.odom)>=2 else None
    stability={'sample_count':len(node.odom),'planar_drift_m':drift,'max_tilt_rad':max(node.tilt) if node.tilt else None}; stability['pass']=stability['sample_count']>=100 and drift is not None and drift<=0.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad']<=math.radians(15)
    response=subscriptions>=1 and delta<=-0.005 and delta>=-0.08 and restore_error<=0.03
    print(json.dumps({'subscription_count':subscriptions,'source_gains':GAINS[TARGET],'start_rad':start[TARGET],'goal_rad':goal[TARGET],'moved_rad':moved,'restored_rad':restored,'delta_rad':delta,'restore_error_rad':restore_error,'response_pass':response,'motion_stability':stability}))
finally:
    node.destroy_node(); rclpy.shutdown()
""".replace("__GAINS__", serialized)


def build_arm_state_capture_script() -> str:
    """Read complete arm state and standing telemetry without publishing a command."""
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointStateArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

ARM_ORDER = ('left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint')
LEG_ORDER = ('left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint')
WAIST_ORDER = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
class Capture(Node):
    def __init__(self):
        super().__init__('robotsim_stl_arm_state_capture')
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.arm={}; self.leg={}; self.waist={}; self.odom=[]; self.tilt=[]
        self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,qos)
        self.create_subscription(JointStateArray,'/aima/hal/joint/leg/state',self.on_leg,qos)
        self.create_subscription(JointStateArray,'/aima/hal/joint/waist/state',self.on_waist,qos)
        self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,qos)
        self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,qos)
    def on_arm(self,msg): self.arm={item.name:float(item.position) for item in msg.joints}
    def on_leg(self,msg): self.leg={item.name:float(item.position) for item in msg.joints}
    def on_waist(self,msg): self.waist={item.name:float(item.position) for item in msg.joints}
    def on_odom(self,msg):
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        self.odom.append((float(p.x),float(p.y),float(p.z),float(q.w),float(q.x),float(q.y),float(q.z)))
    def on_imu(self,msg):
        q=msg.orientation; pitch=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x)))); roll=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(roll),abs(pitch)))
rclpy.init(); node=Capture()
try:
    end=time.monotonic()+2.0
    while rclpy.ok() and time.monotonic()<end: rclpy.spin_once(node,timeout_sec=0.02)
    if any(name not in node.arm for name in ARM_ORDER): raise RuntimeError('missing complete arm state')
    if any(name not in node.leg for name in LEG_ORDER): raise RuntimeError('missing complete leg state')
    if any(name not in node.waist for name in WAIST_ORDER): raise RuntimeError('missing complete waist state')
    drift=math.hypot(node.odom[-1][0]-node.odom[0][0],node.odom[-1][1]-node.odom[0][1]) if len(node.odom)>=2 else None
    stability={'sample_count':len(node.odom),'planar_drift_m':drift,'max_tilt_rad':max(node.tilt) if node.tilt else None}; stability['pass']=stability['sample_count']>=100 and drift is not None and drift<=0.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad']<=math.radians(15)
    print(json.dumps({'arm_positions_rad':{name:node.arm[name] for name in ARM_ORDER},'leg_positions_rad':{name:node.leg[name] for name in LEG_ORDER},'waist_positions_rad':{name:node.waist[name] for name in WAIST_ORDER},'base_pose_world':{'xyz_m':list(node.odom[-1][:3]),'quaternion_wxyz':list(node.odom[-1][3:])},'motion_stability':stability}))
finally:
    node.destroy_node(); rclpy.shutdown()
"""


def build_stand_command_capture_script() -> str:
    """Record delivered MC position/stiffness commands without publishing."""
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommandArray

ARM_ORDER = ('left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint')
LEG_ORDER = ('left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint')
WAIST_ORDER = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
class Capture(Node):
    def __init__(self):
        super().__init__('robotsim_stl_stand_command_capture')
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.samples={'arm':0,'leg':0,'waist':0}; self.commands={'arm':{},'leg':{},'waist':{}}
        self.create_subscription(JointCommandArray,'/aima/hal/joint/arm/command',lambda msg:self.store('arm',msg),qos)
        self.create_subscription(JointCommandArray,'/aima/hal/joint/leg/command',lambda msg:self.store('leg',msg),qos)
        self.create_subscription(JointCommandArray,'/aima/hal/joint/waist/command',lambda msg:self.store('waist',msg),qos)
    def store(self,area,msg):
        self.samples[area]+=1
        self.commands[area]={item.name:{'position':float(item.position),'velocity':float(item.velocity),'effort':float(item.effort),'stiffness':float(item.stiffness),'damping':float(item.damping)} for item in msg.joints}
rclpy.init(); node=Capture()
try:
    end=time.monotonic()+2.0
    while rclpy.ok() and time.monotonic()<end: rclpy.spin_once(node,timeout_sec=0.02)
    expected={'arm':ARM_ORDER,'leg':LEG_ORDER,'waist':WAIST_ORDER}
    for area,names in expected.items():
        if node.samples[area] < 10 or any(name not in node.commands[area] for name in names): raise RuntimeError('missing complete '+area+' command stream')
        for name in names:
            if not all(math.isfinite(node.commands[area][name][field]) for field in ('position','velocity','effort','stiffness','damping')): raise RuntimeError('non-finite '+area+' command')
    print(json.dumps({'sample_counts':node.samples,'arm_commands':{name:node.commands['arm'][name] for name in ARM_ORDER},'leg_commands':{name:node.commands['leg'][name] for name in LEG_ORDER},'waist_commands':{name:node.commands['waist'][name] for name in WAIST_ORDER}}))
finally:
    node.destroy_node(); rclpy.shutdown()
"""


def build_hal_arm_plan_script(gains: dict[str, tuple[float, float]], right_target: list[float], *, duration_s: float, hold_s: float) -> str:
    """Build a bounded native standing arm plan with an explicit safe return."""
    if len(right_target) != 7 or not all(math.isfinite(value) for value in right_target):
        raise ValueError("right arm plan target must contain seven finite values")
    if not 4.0 <= duration_s <= 12.0 or not 0.5 <= hold_s <= 2.0:
        raise ValueError("arm plan duration/hold are outside bounded range")
    serialized_gains = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in gains.items()}, sort_keys=True)
    serialized_target = json.dumps(right_target)
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

ARM_ORDER = ('left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint')
RIGHT_ORDER = ARM_ORDER[7:]
GAINS = __GAINS__
RIGHT_TARGET = __TARGET__
DURATION = __DURATION__
HOLD = __HOLD__
class Plan(Node):
    def __init__(self):
        super().__init__('robotsim_stl_standing_arm_plan')
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.states={}; self.odom=[]; self.tilt=[]
        self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_state,qos)
        self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,qos)
        self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,qos)
        self.publisher=self.create_publisher(JointCommandArray,'/aima/hal/joint/arm/command',qos)
    def on_state(self,msg): self.states={item.name:float(item.position) for item in msg.joints}
    def on_odom(self,msg): self.odom.append((float(msg.pose.pose.position.x),float(msg.pose.pose.position.y)))
    def on_imu(self,msg):
        q=msg.orientation; pitch=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x)))); roll=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(roll),abs(pitch)))
    def spin(self,seconds):
        end=time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<end: rclpy.spin_once(self,timeout_sec=0.002)
    def publish(self,targets,seconds):
        end=time.monotonic()+seconds; seq=0
        while rclpy.ok() and time.monotonic()<end:
            msg=JointCommandArray(); msg.header=MessageHeader(); msg.header.stamp=self.get_clock().now().to_msg(); msg.header.frame_id='standing_arm_plan'; msg.header.sequence=seq
            for name in ARM_ORDER:
                item=JointCommand(); item.name=name; item.position=float(targets[name]); item.velocity=0.0; item.effort=0.0; item.stiffness=float(GAINS[name][0]); item.damping=float(GAINS[name][1]); msg.joints.append(item)
            self.publisher.publish(msg); rclpy.spin_once(self,timeout_sec=0.002); seq+=1; time.sleep(0.01)
    def ramp(self,start,end,duration):
        steps=max(1,int(round(duration/0.01)))
        for step in range(steps):
            phase=(step+1)/steps; blend=3*phase*phase-2*phase*phase*phase
            self.publish({name:start[name]+blend*(end[name]-start[name]) for name in ARM_ORDER},0.01)
rclpy.init(); node=Plan()
try:
    node.spin(1.0)
    if any(name not in node.states for name in ARM_ORDER): raise RuntimeError('missing complete arm state')
    start={name:float(node.states[name]) for name in ARM_ORDER}
    target=dict(start); target.update(dict(zip(RIGHT_ORDER,RIGHT_TARGET)))
    max_delta=max(abs(target[name]-start[name]) for name in RIGHT_ORDER)
    if max_delta>1.1: raise RuntimeError('arm plan exceeds 1.1 rad bounded delta')
    subscriptions=node.publisher.get_subscription_count()
    node.ramp(start,target,DURATION); node.publish(target,HOLD)
    reached={name:float(node.states.get(name,float('nan'))) for name in ARM_ORDER}
    node.ramp(target,start,DURATION); node.publish(start,0.5)
    restored={name:float(node.states.get(name,float('nan'))) for name in ARM_ORDER}
    target_error=max(abs(reached[name]-target[name]) for name in RIGHT_ORDER)
    restore_error=max(abs(restored[name]-start[name]) for name in RIGHT_ORDER)
    drift=math.hypot(node.odom[-1][0]-node.odom[0][0],node.odom[-1][1]-node.odom[0][1]) if len(node.odom)>=2 else None
    stability={'sample_count':len(node.odom),'planar_drift_m':drift,'max_tilt_rad':max(node.tilt) if node.tilt else None}; stability['pass']=stability['sample_count']>=100 and drift is not None and drift<=0.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad']<=math.radians(15)
    response=subscriptions>=1 and target_error<=0.08 and restore_error<=0.04
    print(json.dumps({'subscription_count':subscriptions,'start_right_arm_rad':[start[name] for name in RIGHT_ORDER],'target_right_arm_rad':[target[name] for name in RIGHT_ORDER],'reached_right_arm_rad':[reached[name] for name in RIGHT_ORDER],'restored_right_arm_rad':[restored[name] for name in RIGHT_ORDER],'max_target_error_rad':target_error,'max_restore_error_rad':restore_error,'max_target_delta_rad':max_delta,'response_pass':response,'motion_stability':stability}))
finally:
    node.destroy_node(); rclpy.shutdown()
""".replace("__GAINS__", serialized_gains).replace("__TARGET__", serialized_target).replace("__DURATION__", f"{duration_s:.17g}").replace("__HOLD__", f"{hold_s:.17g}")


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
    generated_directory = RUNTIME / MC_GENERATED_RELATIVE
    if not runtime_dump.is_file() or not generated_directory.is_dir():
        raise RuntimeError("pinned MC runtime generated-config inputs are missing")
    entries = list(generated_directory.rglob("*"))
    for entry in entries:
        if not entry.is_symlink():
            continue
        try:
            entry.resolve(strict=True).relative_to(RUNTIME.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RuntimeError("pinned MC generated-config directory has an escaping or broken link") from exc
    source_files = sorted(path for path in entries if path.is_file())
    if not source_files:
        raise RuntimeError("pinned MC generated-config directory is empty")
    return {"image": image, "active_container_conflicts": conflicts, "runtime_dump": str(runtime_dump), "runtime_dump_sha256": sha256(runtime_dump), "generated_directory": str(generated_directory), "generated_file_count": len(source_files), "generated_symlink_count": sum(path.is_symlink() for path in entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-result", required=True, type=Path)
    parser.add_argument("--startup-timeout-s", type=float, default=25.0)
    parser.add_argument("--run-hand-probe", action="store_true")
    parser.add_argument("--probe-command-position", type=float, default=0.6, help="diagnostic raw HandCommand.position; ordinary public command mapping is 0..1")
    parser.add_argument("--probe-hand-type", type=int, default=2, choices=(1, 2), help="diagnostic HandType; 2 is the public claw mode")
    parser.add_argument("--start-mc", action="store_true", help="start native MC for initialization only; no action or motion command is issued")
    parser.add_argument("--request-stand", action="store_true", help="issue the vendor SD standing request after MC startup; does not command arms, hands, or locomotion")
    parser.add_argument("--run-upper-body-probe", action="store_true", help="small 0.05 rad right-shoulder probe during native stand, then restore")
    parser.add_argument("--run-upper-body-probe-in-stand", action="store_true", help="test the documented upper-body topic while remaining in STAND_DEFAULT, then restore")
    parser.add_argument("--request-upperbody-split", action="store_true", help="switch from SD to the delivered US upper-body remote split mode before any arm probe")
    parser.add_argument("--run-hal-arm-probe", action="store_true", help="direct 0.03 rad wrist probe on the delivered HAL arm command channel during SD")
    parser.add_argument("--run-hal-arm-pd-probe", action="store_true", help="direct 0.03 rad shoulder position/stiffness probe using the delivered remote-controller gains")
    parser.add_argument("--run-hal-arm-plan", action="store_true", help="execute one bounded right-arm pregrasp plan from a static IK validation result")
    parser.add_argument("--ik-validation-result", type=Path, help="hash-validated static standing IK result required by --run-hal-arm-plan")
    parser.add_argument("--ik-validation-result-sha256", help="SHA-256 paired with --ik-validation-result")
    parser.add_argument("--arm-plan-duration-s", type=float, default=8.0)
    parser.add_argument("--arm-plan-hold-s", type=float, default=1.0)
    parser.add_argument("--capture-arm-state", action="store_true", help="record complete standing arm state and stability without publishing a command")
    parser.add_argument("--capture-stand-commands", action="store_true", help="record the delivered MC standing command streams without publishing a command")
    parser.add_argument("--disable-mc-arm-publisher", action="store_true", help="SIMULATION_ONLY: evidence-local MC config disables only arm HAL publishing for the direct arm probe")
    args = parser.parse_args()
    plan_inputs = (args.ik_validation_result, args.ik_validation_result_sha256)
    if not args.overlay_result.is_file() or not 2.0 <= args.startup_timeout_s <= 35.0 or not -1.0 <= args.probe_command_position <= 1.0 or (args.request_stand and not args.start_mc) or (args.request_upperbody_split and not (args.start_mc and args.request_stand)) or (args.run_upper_body_probe and not (args.start_mc and args.request_stand and args.request_upperbody_split)) or (args.run_upper_body_probe_in_stand and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_probe and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_pd_probe and not (args.start_mc and args.request_stand and args.disable_mc_arm_publisher)) or (args.capture_arm_state and not (args.start_mc and args.request_stand)) or (args.capture_stand_commands and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_plan and not (args.start_mc and args.request_stand and args.disable_mc_arm_publisher and all(plan_inputs))) or ((not args.run_hal_arm_plan) and any(plan_inputs)) or (args.disable_mc_arm_publisher and not (args.start_mc and (args.run_hal_arm_probe or args.run_hal_arm_pd_probe or args.run_hal_arm_plan))):
        raise SystemExit("overlay result and a startup timeout in [2,35] seconds are required")
    if args.run_hal_arm_plan and (not 4.0 <= args.arm_plan_duration_s <= 12.0 or not 0.5 <= args.arm_plan_hold_s <= 2.0):
        raise SystemExit("arm plan duration/hold are outside bounded range")
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
        "gates": {"HOST_PREFLIGHT": "NOT_RUN", "NATIVE_SIMULATOR_MODEL_LOAD": "NOT_RUN", "HAND_COMMAND_SUBSCRIBER": "NOT_RUN", "HAND_STATE_PUBLISHER": "NOT_RUN", "HAL_HAND_COMMAND_ROUNDTRIP": "NOT_RUN", "STL_HAND_COMMAND_STAND_STABILITY": "NOT_RUN", "MC_INITIALIZATION": "NOT_RUN", "NATIVE_STAND_REQUEST": "NOT_RUN", "NATIVE_STAND_READBACK": "NOT_RUN", "NATIVE_STAND_STABILITY": "NOT_RUN", "MC_STAND_COMMAND_CAPTURE": "NOT_RUN", "HAL_ARM_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_ARM_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PD_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_ARM_PD_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PLAN_ROUNDTRIP": "NOT_RUN", "HAL_ARM_PLAN_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PLAN_RESTORE": "NOT_RUN", "UPPERBODY_SPLIT_REQUEST": "NOT_RUN", "UPPERBODY_SPLIT_READBACK": "NOT_RUN", "UPPER_BODY_COMMAND_ROUNDTRIP": "NOT_RUN", "UPPER_BODY_COMMAND_STAND_STABILITY": "NOT_RUN", "UPPER_BODY_STAND_COMMAND_ROUNDTRIP": "NOT_RUN", "UPPER_BODY_STAND_COMMAND_STABILITY": "NOT_RUN", "DYNAMICS_QUALIFICATION": "NOT_RUN"},
        "limitations": ["No arm, upper-body, or locomotion command is published by this check.", "A successful SD request does not establish balance, physical contact, or grasp."],
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
        generated_source = Path(str(host["generated_directory"]))
        generated_target = output / "mc_sim_generated"
        shutil.copytree(generated_source, generated_target)
        arm_split_config = None
        if args.disable_mc_arm_publisher:
            mc_source = RUNTIME / MC_SIM_CONFIG_RELATIVE
            arm_split_config = output / "mc_sim_arm_split.yaml"
            pg.write_new_text(arm_split_config, yaml.safe_dump(build_arm_split_mc_config(yaml.safe_load(mc_source.read_text(encoding="utf-8"))), allow_unicode=True, sort_keys=False))
        launch = output / "launch_sim.sh"
        pg.write_new_text(launch, build_launch_script(inputs["simulator_config"].name))
        launch.chmod(0o755)
        if args.start_mc:
            mc_launch = output / "launch_mc.sh"
            pg.write_new_text(mc_launch, build_mc_launch_script())
            mc_launch.chmod(0o755)
        if args.request_stand:
            pg.write_new_text(output / "native_stand_client.py", build_stand_client_script())
        if args.request_upperbody_split:
            pg.write_new_text(output / "native_upperbody_split_client.py", build_upperbody_split_client_script())
        if args.run_upper_body_probe or args.run_upper_body_probe_in_stand:
            pg.write_new_text(output / "native_upper_body_probe.py", build_upper_body_probe_script())
        if args.run_hal_arm_probe:
            pg.write_new_text(output / "native_hal_arm_probe.py", build_hal_arm_probe_script())
        if args.run_hal_arm_pd_probe:
            source_arm_gains = load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_arm_pd_probe.py", build_hal_arm_pd_probe_script(source_arm_gains))
            result["source_arm_pd_gains"] = {name: {"stiffness": stiffness, "damping": damping} for name, (stiffness, damping) in source_arm_gains.items()}
        if args.run_hal_arm_plan:
            assert args.ik_validation_result is not None and args.ik_validation_result_sha256 is not None
            if not args.ik_validation_result.is_file() or sha256(args.ik_validation_result) != args.ik_validation_result_sha256:
                raise ValueError("IK validation result is missing or hash-mismatched")
            ik_payload = json.loads(args.ik_validation_result.read_text(encoding="utf-8"))
            if ik_payload.get("classification") != "SIMULATION_ONLY_STATIC_EXTERNAL_IK_REFERENCE" or ik_payload.get("gates", {}).get("IK_CONVERGED") != "PASS" or ik_payload.get("gates", {}).get("POST_IK_TABLE_CLEARANCE") != "PASS":
                raise ValueError("IK validation does not pass standing/table preflight")
            right_target = ik_payload.get("ik", {}).get("right_arm_q_rad")
            if not isinstance(right_target, list):
                raise ValueError("IK validation lacks right arm target")
            source_arm_gains = load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_arm_plan.py", build_hal_arm_plan_script(source_arm_gains, [float(value) for value in right_target], duration_s=args.arm_plan_duration_s, hold_s=args.arm_plan_hold_s))
            result["standing_arm_plan"] = {"ik_validation_result": {"path": str(args.ik_validation_result), "sha256": sha256(args.ik_validation_result)}, "right_arm_target_rad": right_target, "duration_s": args.arm_plan_duration_s, "hold_s": args.arm_plan_hold_s, "source_gains": {name: {"stiffness": stiffness, "damping": damping} for name, (stiffness, damping) in source_arm_gains.items()}}
        if args.capture_arm_state:
            pg.write_new_text(output / "native_arm_state_capture.py", build_arm_state_capture_script())
        if args.capture_stand_commands:
            pg.write_new_text(output / "native_stand_command_capture.py", build_stand_command_capture_script())
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
            "--mount", "type=bind,source=" + str(generated_target) + ",target=/workspace/raicom2026/" + str(MC_GENERATED_RELATIVE),
            "-e", "DISPLAY=:0", "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20", "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        if arm_split_config is not None:
            command[command.index(IMAGE):command.index(IMAGE)] = ["--mount", "type=bind,source=" + str(arm_split_config) + ",target=/workspace/raicom2026/" + str(MC_SIM_CONFIG_RELATIVE) + ",readonly"]
            result["mc_arm_split_overlay"] = {"path": str(arm_split_config), "sha256": sha256(arm_split_config), "disabled_publisher": "arm_joint_command"}
        started = run(command, timeout_s=10.0)
        pg.write_new_text(output / "container_id.txt", started.stdout)
        run(["docker", "exec", "-d", container, "bash", "-lc", "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"], timeout_s=5.0)
        ready, console = wait_for_log(output / "mujoco-console.log", (SIM_READY_MARKER, *HAND_MARKERS), timeout_s=args.startup_timeout_s)
        pg.write_new_text(output / "mujoco-console.tail.txt", console[-12000:])
        result["gates"]["NATIVE_SIMULATOR_MODEL_LOAD"] = "PASS" if ready else "FAIL"
        result["gates"]["HAND_COMMAND_SUBSCRIBER"] = "PASS" if ready else "FAIL"
        result["gates"]["HAND_STATE_PUBLISHER"] = "PASS" if ready else "FAIL"
        result["simulator_log_markers"] = {marker: marker in console for marker in (SIM_READY_MARKER, *HAND_MARKERS)}
        if ready and args.start_mc:
            run(["docker", "exec", "-d", container, "bash", "-lc", "bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1"], timeout_s=5.0)
            mc_ready, mc_log = wait_for_log(output / "mc_logs/mc.log", (MC_READY_MARKER,), timeout_s=args.startup_timeout_s)
            pg.write_new_text(output / "mc-console.tail.txt", (output / "mc-console.log").read_text(encoding="utf-8", errors="replace")[-12000:] if (output / "mc-console.log").is_file() else "")
            pg.write_new_text(output / "mc-log.tail.txt", mc_log[-12000:])
            result["gates"]["MC_INITIALIZATION"] = "PASS" if mc_ready else "FAIL"
            if mc_ready:
                environment = "source /opt/ros/humble/setup.bash; source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash; "
                topics = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s ros2 topic list -t"], timeout_s=10.0, check=False)
                pg.write_new_text(output / "native_ros_topics.txt", topics.stdout)
                result["native_ros_topics"] = {"returncode": topics.returncode, "upper_body_topic_present": "/mc/upper_body_command" in topics.stdout, "arm_hal_topic_present": "/aima/hal/joint/arm/command" in topics.stdout}
            if mc_ready and args.request_stand:
                stand = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_stand_client.py"], timeout_s=12.0, check=False)
                pg.write_new_text(output / "native_stand_request.stdout", stand.stdout)
                result["gates"]["NATIVE_STAND_REQUEST"] = "PASS" if stand.returncode == 0 else "FAIL"
                try:
                    stand_result = json.loads(stand.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    stand_result = {"parse_error": True}
                result["native_stand"] = {"returncode": stand.returncode, "result": stand_result, "stdout": stand.stdout}
                result["gates"]["NATIVE_STAND_READBACK"] = "PASS" if stand.returncode == 0 and stand_result.get("mode_name") == "STAND_DEFAULT" else "FAIL"
                result["gates"]["NATIVE_STAND_STABILITY"] = "PASS" if stand.returncode == 0 and stand_result.get("stability", {}).get("pass") else "FAIL"
                if args.run_hal_arm_probe and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    hal = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_hal_arm_probe.py"], timeout_s=14.0, check=False)
                    pg.write_new_text(output / "native_hal_arm_probe.stdout", hal.stdout)
                    try:
                        hal_result = json.loads(hal.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        hal_result = {"parse_error": True}
                    result["hal_arm_probe"] = {"returncode": hal.returncode, "result": hal_result, "stdout": hal.stdout}
                    result["gates"]["HAL_ARM_COMMAND_ROUNDTRIP"] = "PASS" if hal.returncode == 0 and hal_result.get("response_pass") else "FAIL"
                    result["gates"]["HAL_ARM_COMMAND_STAND_STABILITY"] = "PASS" if hal.returncode == 0 and hal_result.get("motion_stability", {}).get("pass") else "FAIL"
                if args.run_hal_arm_pd_probe and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    pd_probe = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 14s python3 /workspace/evidence/native_hal_arm_pd_probe.py"], timeout_s=16.0, check=False)
                    pg.write_new_text(output / "native_hal_arm_pd_probe.stdout", pd_probe.stdout)
                    try:
                        pd_result = json.loads(pd_probe.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        pd_result = {"parse_error": True}
                    result["hal_arm_pd_probe"] = {"returncode": pd_probe.returncode, "result": pd_result, "stdout": pd_probe.stdout}
                    result["gates"]["HAL_ARM_PD_COMMAND_ROUNDTRIP"] = "PASS" if pd_probe.returncode == 0 and pd_result.get("response_pass") else "FAIL"
                    result["gates"]["HAL_ARM_PD_COMMAND_STAND_STABILITY"] = "PASS" if pd_probe.returncode == 0 and pd_result.get("motion_stability", {}).get("pass") else "FAIL"
                if args.run_hal_arm_plan and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    plan = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 28s python3 /workspace/evidence/native_hal_arm_plan.py"], timeout_s=30.0, check=False)
                    pg.write_new_text(output / "native_hal_arm_plan.stdout", plan.stdout)
                    try:
                        plan_result = json.loads(plan.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        plan_result = {"parse_error": True}
                    result["standing_arm_plan_result"] = {"returncode": plan.returncode, "result": plan_result, "stdout": plan.stdout}
                    result["gates"]["HAL_ARM_PLAN_ROUNDTRIP"] = "PASS" if plan.returncode == 0 and plan_result.get("response_pass") else "FAIL"
                    result["gates"]["HAL_ARM_PLAN_STAND_STABILITY"] = "PASS" if plan.returncode == 0 and plan_result.get("motion_stability", {}).get("pass") else "FAIL"
                    result["gates"]["HAL_ARM_PLAN_RESTORE"] = "PASS" if plan.returncode == 0 and plan_result.get("max_restore_error_rad", float("inf")) <= 0.04 else "FAIL"
                if args.capture_arm_state and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    state_capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_arm_state_capture.py"], timeout_s=10.0, check=False)
                    pg.write_new_text(output / "native_arm_state_capture.stdout", state_capture.stdout)
                    try:
                        state_result = json.loads(state_capture.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        state_result = {"parse_error": True}
                    result["native_arm_state_capture"] = {"returncode": state_capture.returncode, "result": state_result, "stdout": state_capture.stdout}
                if args.capture_stand_commands and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    command_capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_stand_command_capture.py"], timeout_s=10.0, check=False)
                    pg.write_new_text(output / "native_stand_command_capture.stdout", command_capture.stdout)
                    try:
                        command_result = json.loads(command_capture.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        command_result = {"parse_error": True}
                    result["native_stand_command_capture"] = {"returncode": command_capture.returncode, "result": command_result, "stdout": command_capture.stdout}
                    result["gates"]["MC_STAND_COMMAND_CAPTURE"] = "PASS" if command_capture.returncode == 0 else "FAIL"
                if args.run_upper_body_probe_in_stand and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    upper_stand = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_upper_body_probe.py"], timeout_s=14.0, check=False)
                    pg.write_new_text(output / "native_upper_body_stand_probe.stdout", upper_stand.stdout)
                    try:
                        upper_stand_result = json.loads(upper_stand.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        upper_stand_result = {"parse_error": True}
                    result["upper_body_stand_probe"] = {"returncode": upper_stand.returncode, "result": upper_stand_result, "stdout": upper_stand.stdout}
                    result["gates"]["UPPER_BODY_STAND_COMMAND_ROUNDTRIP"] = "PASS" if upper_stand.returncode == 0 and upper_stand_result.get("response_pass") else "FAIL"
                    result["gates"]["UPPER_BODY_STAND_COMMAND_STABILITY"] = "PASS" if upper_stand.returncode == 0 and upper_stand_result.get("motion_stability", {}).get("pass") else "FAIL"
                if args.request_upperbody_split and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    split = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_upperbody_split_client.py"], timeout_s=14.0, check=False)
                    pg.write_new_text(output / "native_upperbody_split.stdout", split.stdout)
                    try:
                        split_result = json.loads(split.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        split_result = {"parse_error": True}
                    result["native_upperbody_split"] = {"returncode": split.returncode, "result": split_result, "stdout": split.stdout}
                    result["gates"]["UPPERBODY_SPLIT_REQUEST"] = "PASS" if split.returncode == 0 else "FAIL"
                    result["gates"]["UPPERBODY_SPLIT_READBACK"] = "PASS" if split.returncode == 0 and split_result.get("mode_name") == "UPPERBODY_REMOTE_SPLIT" else "FAIL"
                    topics_after = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s ros2 topic list -t"], timeout_s=10.0, check=False)
                    pg.write_new_text(output / "native_ros_topics_after_upperbody_split.txt", topics_after.stdout)
                    result["native_ros_topics_after_upperbody_split"] = {"returncode": topics_after.returncode, "upper_body_topic_present": "/mc/upper_body_command" in topics_after.stdout}
                if args.run_upper_body_probe and result["gates"]["UPPERBODY_SPLIT_READBACK"] == "PASS":
                    upper = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_upper_body_probe.py"], timeout_s=14.0, check=False)
                    pg.write_new_text(output / "native_upper_body_probe.stdout", upper.stdout)
                    try:
                        upper_result = json.loads(upper.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        upper_result = {"parse_error": True}
                    result["upper_body_probe"] = {"returncode": upper.returncode, "result": upper_result, "stdout": upper.stdout}
                    result["gates"]["UPPER_BODY_COMMAND_ROUNDTRIP"] = "PASS" if upper.returncode == 0 and upper_result.get("response_pass") else "FAIL"
                    result["gates"]["UPPER_BODY_COMMAND_STAND_STABILITY"] = "PASS" if upper.returncode == 0 and upper_result.get("motion_stability", {}).get("pass") else "FAIL"
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
            if args.start_mc and args.request_stand:
                result["gates"]["STL_HAND_COMMAND_STAND_STABILITY"] = "PASS" if hand_probe.get("motion_stability", {}).get("pass") else "FAIL"
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
