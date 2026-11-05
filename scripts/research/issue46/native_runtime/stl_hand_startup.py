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


RIGHT_ARM_ORDER = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
)


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
        "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED",
    }:
        raise ValueError("not a native STL hand bridge preparation result")
    gates = result.get("gates", {})
    required = ("FROZEN_STL_MOUNT_INPUT", "HOST_STATIC_MODEL_COMPILE", "NATIVE_HAND_CHANNEL_CONFIGURATION")
    if any(gates.get(key) != "PASS" for key in required):
        raise ValueError("native STL hand overlay preflight did not pass")
    if classification in {
        "SIMULATION_ONLY_NATIVE_MC_HAND_TABLE_BRIDGE_PREPARED",
        "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED",
    } and gates.get("TABLE_SCENE_STATIC_COMPILE") != "PASS":
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


def build_arm_hand_split_mc_config(source: dict[str, object]) -> dict[str, object]:
    """Disable only MC's competing arm and hand publishers in an overlay.

    Legs and waist remain owned by the delivered standing controller.  The
    hand-topic ownership is needed only while the SIMULATION_ONLY external
    sidecar sends the exact static-candidate claw command; no model parameter
    or collision setting is changed.
    """
    import copy

    result = copy.deepcopy(source)
    module = result.get("MotionControlModule")
    if not isinstance(module, dict):
        raise ValueError("MC config lacks MotionControlModule")
    manager = module.get("PublisherManager")
    if not isinstance(manager, dict) or not isinstance(manager.get("publisher_cfg_list"), list):
        raise ValueError("MC config lacks publisher_cfg_list")
    for expected in ("arm_joint_command", "hand_joint_command"):
        matches = [item for item in manager["publisher_cfg_list"] if isinstance(item, dict) and item.get("name") == expected]
        if len(matches) != 1 or matches[0].get("enable") is not True:
            raise ValueError(f"MC config does not expose exactly one enabled {expected} publisher")
        matches[0]["enable"] = False
    return result


def build_waist_split_mc_config(source: dict[str, object]) -> dict[str, object]:
    """Disable only MC's waist HAL publisher in an evidence-local config copy."""
    import copy
    result = copy.deepcopy(source)
    module = result.get("MotionControlModule")
    if not isinstance(module, dict):
        raise ValueError("MC config lacks MotionControlModule")
    manager = module.get("PublisherManager")
    if not isinstance(manager, dict) or not isinstance(manager.get("publisher_cfg_list"), list):
        raise ValueError("MC config lacks publisher_cfg_list")
    matches = [item for item in manager["publisher_cfg_list"] if isinstance(item, dict) and item.get("name") == "waist_joint_command"]
    if len(matches) != 1 or matches[0].get("enable") is not True:
        raise ValueError("MC config does not expose exactly one enabled waist_joint_command publisher")
    matches[0]["enable"] = False
    return result


def build_arm_waist_split_mc_config(source: dict[str, object]) -> dict[str, object]:
    """Disable only the competing MC arm/waist publishers in an evidence overlay.

    The delivered direct HAL example owns these two topics while legs remain
    under the native standing controller.  This is a controller-source split,
    not a mechanical-model or collision modification.
    """
    import copy

    result = copy.deepcopy(source)
    module = result.get("MotionControlModule")
    if not isinstance(module, dict):
        raise ValueError("MC config lacks MotionControlModule")
    manager = module.get("PublisherManager")
    if not isinstance(manager, dict) or not isinstance(manager.get("publisher_cfg_list"), list):
        raise ValueError("MC config lacks publisher_cfg_list")
    for expected in ("arm_joint_command", "waist_joint_command"):
        matches = [item for item in manager["publisher_cfg_list"] if isinstance(item, dict) and item.get("name") == expected]
        if len(matches) != 1 or matches[0].get("enable") is not True:
            raise ValueError(f"MC config does not expose exactly one enabled {expected}")
        matches[0]["enable"] = False
    return result



def build_coordinated_waist_relay_config(source: dict[str, object]) -> dict[str, object]:
    """Redirect only MC waist output through the pregrasp handoff relay.

    The MC computes the original waist commands throughout standing/navigation;
    this overlay changes only transport ownership, never gains or policy.
    """
    result=build_arm_hand_split_mc_config(source)
    original='/aima/hal/joint/waist/command'
    private='/robotsim/issue46/mc/waist/command'
    entries=result["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"]
    publishers=[x for x in entries if x.get("name")=="waist_joint_command"]
    channels=[x for x in result["aimrt"]["channel"]["pub_topics_options"] if x.get("topic_name")==original]
    if len(publishers)!=1 or publishers[0].get("enable") is not True or publishers[0].get("topic")!=original or len(channels)!=1:
        raise ValueError("expected exact enabled MC waist publisher and ROS channel")
    publishers[0]["topic"]=private
    channels[0]["topic_name"]=private
    return result


def build_stand_client_script() -> str:
    """Minimal equivalent of the delivered ``set_mc_action SD`` example."""
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import CommonRequest, McActionCommand, RequestHeader
from aimdk_msgs.srv import GetMcAction, SetMcAction
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def call(node, client, request):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError('service unavailable')
    # The delivered MC service can be registered before its remote executor is
    # ready immediately after a GUI Reset.  Follow the vendor examples: make
    # a bounded sequence of short attempts rather than treating one missed
    # response as a controller failure.
    for _ in range(8):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.25)
        if future.done() and future.result() is not None:
            return future.result()
        time.sleep(0.05)
    raise RuntimeError('service call timed out after 8 bounded retries')


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
    node.spin(2.0)
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
        if isinstance(name, str) and (name in {"waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"} or name.endswith(("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint", "wrist_yaw_joint", "wrist_pitch_joint", "wrist_roll_joint"))):
            try:
                gains = float(stiffness), float(damping)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid delivered arm gains for {name}") from exc
            if not all(math.isfinite(value) and value > 0.0 for value in gains):
                raise RuntimeError(f"non-positive delivered arm gains for {name}")
            values[name] = gains
    if len(values) != 17:
        raise RuntimeError(f"delivered remote-controller model config provides {len(values)} arm/waist gain records, expected 17")
    return values


def load_animation_player_arm_pd_parameters() -> dict[str, tuple[float, float]]:
    """Read the delivered AnimationPlayer PD values for the fourteen arms.

    These are not invented gains: the old MC's own AnimationPlayer uses them
    for its packaged upper-body CSV resources.  The hybrid controller may use
    them only in an explicitly isolated simulation overlay.
    """
    path = RUNTIME / "mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation_player.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("arm_joints"), list):
        raise RuntimeError("delivered AnimationPlayer config lacks arm_joints")
    values: dict[str, tuple[float, float]] = {}
    for item in payload["arm_joints"]:
        if not isinstance(item, dict):
            continue
        name, stiffness, damping = item.get("name"), item.get("kp"), item.get("kd")
        if not isinstance(name, str) or not name.startswith(("left_", "right_")):
            continue
        try:
            gains = float(stiffness), float(damping)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid delivered AnimationPlayer gain for {name}") from exc
        if not all(math.isfinite(value) and value > 0.0 for value in gains):
            raise RuntimeError(f"non-positive delivered AnimationPlayer gain for {name}")
        values[name] = gains
    if len(values) != 14:
        raise RuntimeError(f"delivered AnimationPlayer provides {len(values)} arm gains, expected 14")
    return values


def build_hal_arm_pd_probe_script(gains: dict[str, tuple[float, float]]) -> str:
    """Bounded direct position/stiffness arm probe using delivered gains."""
    arm_gains = {name: values for name, values in gains.items() if name.startswith(("left_", "right_"))}
    if len(arm_gains) != 14:
        raise ValueError("delivered source gains do not contain exactly 14 arm joints")
    serialized = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in arm_gains.items()}, sort_keys=True)
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
    def publish(self, targets, seconds, inject_effort=False):
        end=time.monotonic()+seconds; seq=0
        while rclpy.ok() and time.monotonic()<end:
            msg=JointCommandArray(); msg.header=MessageHeader(); msg.header.stamp=self.get_clock().now().to_msg(); msg.header.frame_id='hal_arm_pd_probe'; msg.header.sequence=seq
            for name in sorted(targets):
                item=JointCommand(); item.name=name; item.position=float(targets[name]); item.velocity=0.0; item.effort=0.6 if inject_effort and name == TARGET else 0.0; item.stiffness=float(GAINS[name][0]); item.damping=float(GAINS[name][1]); msg.joints.append(item)
            self.publisher.publish(msg); rclpy.spin_once(self, timeout_sec=0.002); seq+=1; time.sleep(0.01)

rclpy.init(); node=Probe()
try:
    node.spin(2.0)
    if set(GAINS)-set(node.states): raise RuntimeError('missing complete arm state')
    start={name:float(node.states[name]) for name in sorted(GAINS)}
    goal=dict(start); goal[TARGET]=start[TARGET]-0.03
    subscriptions=node.publisher.get_subscription_count()
    # This validates position/stiffness tracking alone.  An additive effort
    # pulse turns a 0.03-rad identification move into a force experiment and
    # previously caused a large overshoot under the arm-split overlay.
    node.publish(goal,2.0,inject_effort=False); moved=float(node.states.get(TARGET,float('nan')))
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


def build_hal_bilateral_arm_plan_script(
    gains: dict[str, tuple[float, float]],
    right_target: list[float],
    left_target: list[float],
    *,
    duration_s: float,
    hold_s: float,
    restore: bool = True,
    right_hand_command: float | None = None,
    hand_hold_s: float = 0.0,
    initial_arm_state: dict[str, float] | None = None,
    waist_target: list[float] | None = None,
    hand_open_command: float = .97,
    circle_contract: dict | None = None,
    use_native_bias_effort: bool = False,
    source_joint_limits: dict[str, list[float]] | None = None,
    enable_measured_correction: bool = False,
) -> str:
    """Build a bounded direct-HAL bilateral pregrasp -> restore trajectory.

    This uses the delivered AnimationPlayer position gains and the existing
    arm HAL topic.  It deliberately sends no additive effort: position and
    stiffness tracking is the bounded validation here, while a hard-coded
    effort pulse would be a different force experiment.  Legs, waist, source
    geometry, the bottle and fingers remain under their normal controllers.
    """
    if not math.isfinite(hand_open_command) or not 0. <= hand_open_command <= 1.:raise ValueError("opening outside source range")
    arm_gains = {name: values for name, values in gains.items() if name.startswith(("left_", "right_"))}
    waist_names = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
    if waist_target is not None:
        if len(waist_target) != 3 or any(not math.isfinite(float(v)) for v in waist_target) or any(name not in gains for name in waist_names):
            raise ValueError("coordinated waist plan requires three finite targets and source waist gains")
        arm_gains.update({name: gains[name] for name in waist_names})
    if len(arm_gains) != (17 if waist_target is not None else 14) or len(right_target) != 7 or len(left_target) != 7:
        raise ValueError("bilateral arm plan requires fourteen source gains and two seven-joint targets")
    if right_hand_command is not None and not 0.0 <= right_hand_command <= 1.0:
        raise ValueError("right hand command must use the source 0..1 range")
    if right_hand_command is None and hand_hold_s != 0.0:
        raise ValueError("hand hold requires an explicit right hand command")
    if right_hand_command is not None and not 0.5 <= hand_hold_s <= 5.0:
        raise ValueError("hand-close hold must be bounded to 0.5..5 seconds")
    if initial_arm_state is not None and any(name not in initial_arm_state or not math.isfinite(float(initial_arm_state[name])) for name in arm_gains):
        raise ValueError("initial arm fallback must contain every finite bilateral arm joint")
    serialized_gains = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in arm_gains.items()}, sort_keys=True)
    return """#!/usr/bin/env python3
import json, math, os, time, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import HandCommand, HandCommandArray, HandStateArray, HandType, JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

LEFT=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint')
RIGHT=('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
CIRCLE_CONTRACT=__CIRCLE_CONTRACT__; USE_NATIVE_BIAS=__USE_NATIVE_BIAS__; SOURCE_LIMITS=__SOURCE_LIMITS__; ENABLE_CORRECTION=__ENABLE_CORRECTION__; HAND_OPEN=__HAND_OPEN__; WAIST=__WAIST_NAMES__; TARGET_WAIST=__WAIST_TARGET__; CONTROL=LEFT+RIGHT+WAIST
TARGET_LEFT=__LEFT_TARGET__; TARGET_RIGHT=__RIGHT_TARGET__; GAINS=__GAINS__; DURATION=__DURATION__; HOLD=__HOLD__; RESTORE=__RESTORE__; HAND_RIGHT=__HAND_RIGHT__; HAND_HOLD=__HAND_HOLD__; INITIAL_FALLBACK=__INITIAL_FALLBACK__
def native_circle_arrival(contract):
 if contract is None:return {'pass':True,'classification':'LEGACY_NO_NATIVE_TCP_GATE'}
 path='/workspace/evidence/native_physics.jsonl'
 with open(path,'rb') as stream:
  stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-131072));raw=stream.read()
 lines=raw.splitlines()
 if not raw.endswith(bytes([10])):lines=lines[:-1]
 records=[]
 for line in reversed(lines):
  try:records=[json.loads(line)];break
  except (ValueError,UnicodeError):continue
 if not records:raise RuntimeError('NO_NATIVE_PHYSICS_FOR_TCP')
 row=records[0];age=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9
 if not 0.<=age<=.20:raise RuntimeError('STALE_NATIVE_TCP_POSE')
 hand=row['bodies']['R_omnipicker_base_link'];bottle=row['bodies']['test_cylinder']
 R=hand['rotation'];B=bottle['rotation'];c=contract['circle_center_local_m'];height=contract['height_above_center_m']
 tcp=[hand['position_m'][i]+sum(R[3*i+j]*c[j] for j in range(3)) for i in range(3)]
 target=[bottle['position_m'][i]+height*B[3*i+2] for i in range(3)]
 error=math.sqrt(sum((tcp[i]-target[i])**2 for i in range(3)))
 dot=max(-1.,min(1.,sum(R[3*i]*B[3*i+2] for i in range(3))));angle=math.acos(dot)
 return {'pass':error<=.005 and angle<=.03,'position_error_m':error,'axis_error_rad':angle,'tcp_world_m':tcp,'target_world_m':target,'native_pose_age_s':age,'sim_time_s':row['sim_time_s'],'classification':'SIMULATION_ONLY_NATIVE_TCP_ARRIVAL'}
def native_bilateral_contact(close_start):
 path='/workspace/evidence/native_physics.jsonl'
 with open(path,'rb') as stream:
  stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-1048576));raw=stream.read()
 lines=raw.splitlines()
 if not raw.endswith(bytes([10])):lines=lines[:-1]
 bilateral=[]
 for line in lines:
  try:row=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if int(row['host_monotonic_ns'])*1e-9 < close_start:continue
  forces={'R_hand_narrow3_Link':0.,'R_hand_wide3_Link':0.}
  for c in row.get('contacts',[]):
   pair={c.get('body1'),c.get('body2')}
   if 'test_cylinder' not in pair:continue
   for name in forces:
    if name in pair:forces[name]+=max(0.,float(c['local_force_torque'][0]))
  if all(v>=.1 for v in forces.values()):bilateral.append((int(row['host_monotonic_ns']),forces,row['sim_time_s']))
 if len(bilateral)<5 or (bilateral[-1][0]-bilateral[0][0])*1e-9<.20:return {'pass':False,'sample_count':len(bilateral),'minimum_force_n':{n:min((x[1][n] for x in bilateral),default=0.) for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')},'classification':'NO_SUSTAINED_BILATERAL_PHYSICAL_CONTACT'}
 return {'pass':True,'sample_count':len(bilateral),'duration_s':(bilateral[-1][0]-bilateral[0][0])*1e-9,'minimum_force_n':{n:min(x[1][n] for x in bilateral) for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')},'sim_time_range_s':[bilateral[0][2],bilateral[-1][2]],'classification':'SUSTAINED_BILATERAL_PHYSICAL_CONTACT'}
class Plan(Node):
 def __init__(self):
  super().__init__('robotsim_hal_bilateral_pregrasp');q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT);hand_q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.TRANSIENT_LOCAL);self.arm={};self.hand={};self.odom=[];self.tilt=[];self.seq=0;self.trace=[]
  # The simulator may publish the two arm groups in separate messages.  Merge
  # named samples so a later partial packet cannot discard the already-read
  # opposite-arm state before the bounded plan begins.
  self.odom_trace=[];self.arm_receipt={};self.arm_stamp={};self.arm_regressions=0;self.arm_callbacks=0;self.run_started=time.monotonic();self.last_trace_at=None
  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,q)
  self.waist_pub=None
  if WAIST:
   self.create_subscription(JointStateArray,'/aima/hal/joint/waist/state',self.on_arm,q)
   self.waist_pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/waist/command',q)
  self.create_subscription(HandStateArray,'/aima/hal/joint/hand/state',self.on_hand,hand_q)
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,q)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.imu,q);self.pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/arm/command',q);self.hand_pub=self.create_publisher(HandCommandArray,'/aima/hal/joint/hand/command',hand_q)
 def on_odom(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation;v=m.twist.twist.linear
  self.odom.append((float(p.x),float(p.y)))
  self.odom_trace.append({'host_monotonic_s':time.monotonic(),'source_stamp_ns':int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec),'xyz_m':[p.x,p.y,p.z],'quaternion_wxyz':[q.w,q.x,q.y,q.z],'linear_velocity':[v.x,v.y,v.z]})
 def on_arm(self,m):
  now=time.monotonic();stamp=int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
  self.arm_callbacks+=1
  for item in m.joints:
   if not math.isfinite(float(item.position)):raise RuntimeError('NONFINITE_ARM_STATE:'+item.name)
   old=self.arm_stamp.get(item.name)
   if old is not None and stamp<old:
    fault={'gate':'REGRESSING_ARM_STATE','joint':item.name,'previous_source_stamp_ns':old,'received_source_stamp_ns':stamp,'regression_ns':old-stamp,'host_receipt_monotonic_s':now,'last_unique_receipt_monotonic_s':self.arm_receipt.get(item.name),'arm_callback_count':self.arm_callbacks,'control_sequence':self.seq}
    self.arm_regressions+=1
    print(json.dumps({'arm_feedback_late_sample_discarded':fault,'regression_count':self.arm_regressions}),flush=True)
    # DDS can deliver an older source-stamped frame after a newer one. Never
    # apply or refresh from it. Keep consuming only strictly increasing frames;
    # if those stop, arm_fresh() fails against the last unique receipt age.
    if now-self.arm_receipt.get(item.name,0.)>.20:
     raise RuntimeError('STALE_ARM_FEEDBACK_AFTER_SOURCE_REGRESSION:'+json.dumps(fault))
    continue
   if stamp>0 and (old is None or stamp>old):
    self.arm[item.name]=float(item.position);self.arm_receipt[item.name]=now;self.arm_stamp[item.name]=stamp
 def arm_fresh(self,max_age=.20):
  now=time.monotonic()
  return all(name in self.arm_receipt and now-self.arm_receipt[name]<=max_age for name in CONTROL)
 def on_hand(self,m):
  for item in list(m.left_hands)+list(m.right_hands):self.hand[{'L_claw_joint':'left_claw_joint','R_claw_joint':'right_claw_joint'}.get(item.name,item.name)]=float(item.position)
 def imu(self,m):
  q=m.orientation;p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))));r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y));self.tilt.append(max(abs(r),abs(p)))
 def spin(self,s):
  end=time.monotonic()+s
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.002)
 def native_bias(self):
  if not USE_NATIVE_BIAS:return {}
  path='/workspace/evidence/native_physics.jsonl'
  try:
   with open(path,'rb') as f:
    f.seek(0,2);f.seek(max(0,f.tell()-262144));lines=f.read().splitlines()
   for raw in reversed(lines):
    try:row=json.loads(raw);break
    except (ValueError,UnicodeDecodeError):continue
   else:raise RuntimeError('NO_NATIVE_BIAS_SAMPLE')
   if time.monotonic_ns()/1e9-float(row['host_monotonic_ns'])/1e9>.20:raise RuntimeError('STALE_NATIVE_BIAS')
   return {n:float(row['joints'][n]['bias_force']) for n in RIGHT if n in row.get('joints',{})}
  except FileNotFoundError as exc:raise RuntimeError('NATIVE_BIAS_FILE_MISSING') from exc
 def send(self,target,seconds,stage,hand_right=None):
  end=time.monotonic()+seconds;next_tick=time.monotonic();start=time.monotonic()
  while rclpy.ok() and time.monotonic()<end:
   if self.tilt and max(self.tilt)>math.radians(15):raise RuntimeError('PREGRASP_TILT_LIMIT')
   if len(self.odom)>1 and math.hypot(self.odom[-1][0]-self.odom[0][0],self.odom[-1][1]-self.odom[0][1])>.05:raise RuntimeError('PREGRASP_BASE_DRIFT_LIMIT:'+json.dumps({'stage':stage,'start_xy':self.odom[0],'current_xy':self.odom[-1],'measured_drift_m':math.hypot(self.odom[-1][0]-self.odom[0][0],self.odom[-1][1]-self.odom[0][1])}))
   if not self.arm_fresh():raise RuntimeError('STALE_BILATERAL_ARM_FEEDBACK_DURING:'+stage)
   bias=self.native_bias()
   if USE_NATIVE_BIAS and any(name not in bias for name in RIGHT):raise RuntimeError('INCOMPLETE_NATIVE_BIAS_SAMPLE')
   m=JointCommandArray();m.header=MessageHeader();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='hal_bilateral_pregrasp';m.header.sequence=self.seq;self.seq+=1
   for name in LEFT+RIGHT:
    x=JointCommand();x.name=name;x.position=float(target[name]);x.velocity=0.;x.effort=float(bias.get(name,0.));x.stiffness=float(GAINS[name][0]);x.damping=float(GAINS[name][1]);m.joints.append(x)
   self.pub.publish(m)
   if WAIST:
    w=JointCommandArray();w.header=m.header
    for name in WAIST:
     x=JointCommand();x.name=name;x.position=float(target[name]);x.velocity=0.;x.effort=0.;x.stiffness=float(GAINS[name][0]);x.damping=float(GAINS[name][1]);w.joints.append(x)
    self.waist_pub.publish(w)
   if hand_right is not None:
    # Match the pinned vendor example: activate only the right claw and leave
    # the parked left hand in NONE mode.
    h=HandCommandArray();h.left_hand_type=HandType(value=0);h.right_hand_type=HandType(value=2)
    x=HandCommand();x.name='right_claw_joint';x.position=float(hand_right);x.velocity=1.;x.acceleration=1.;x.deceleration=1.;x.effort=1.;h.right_hands.append(x)
    self.hand_pub.publish(h)
   rclpy.spin_once(self,timeout_sec=.001)
   if self.last_trace_at is None or time.monotonic()-self.last_trace_at>=.05:
    err=max(abs(float(self.arm.get(name,float('nan')))-target[name]) for name in LEFT+RIGHT) if all(name in self.arm for name in LEFT+RIGHT) else float('inf')
    self.last_trace_at=time.monotonic()
    self.trace.append({'time_s':self.last_trace_at-self.run_started,'stage_elapsed_s':self.last_trace_at-start,'host_monotonic_s':self.last_trace_at,'stage':stage,'arm_fresh':self.arm_fresh(),'arm_source_stamp_ns':dict(self.arm_stamp),'arm_unique_receipt_s':dict(self.arm_receipt),'arm_callback_count':self.arm_callbacks,'arm_regression_discard_count':self.arm_regressions,'published_target_rad':dict(target),'max_arm_error_rad':err,'waist':{n:self.arm.get(n) for n in WAIST},'right':{n:self.arm.get(n) for n in RIGHT},'left':{n:self.arm.get(n) for n in LEFT}})
   # Drain feedback during the remainder of the 50Hz period. Sleeping here
   # starved callbacks from 1000Hz arm + 500Hz IMU/odom subscriptions.
   next_tick+=.02
   self.spin(max(0,min(next_tick,end)-time.monotonic()))
rclpy.init();n=Plan()
try:
 n.spin(2.0)
 if not n.arm_fresh():raise RuntimeError('STALE_OR_MISSING_BILATERAL_ARM_STATE')
 if WAIST and n.waist_pub.get_subscription_count()!=1:raise RuntimeError('WAIST_SUBSCRIBER_NOT_UNIQUE')
 if any(name not in n.arm and name not in INITIAL_FALLBACK for name in LEFT+RIGHT):raise RuntimeError('missing complete arm state and dock-state fallback')
 start={n_:float(n.arm[n_]) for n_ in CONTROL};fallback_used=[]; goal={**dict(zip(LEFT,TARGET_LEFT)),**dict(zip(RIGHT,TARGET_RIGHT)),**dict(zip(WAIST,TARGET_WAIST))}
 if max(abs(goal[n_]-start[n_]) for n_ in CONTROL)>1.1:raise RuntimeError('pregrasp target delta exceeds 1.1 rad')
 # The delivered claw component is initialized by the same 0-command
 # handshake used in the previously passing native hand round-trip before a
 # nonzero target is sent.
 if HAND_RIGHT is not None:
  n.send(start,.8,'HAND_ACTIVATION',0.0)
  # Open while the arms are still at the measured dock-safe pose.  Starting
  # the arm approach with closed fingers can lodge a pad against the object
  # before the root position actuator has clearance to move.
  n.send(start,2.0,'HAND_OPEN_PREGRASP',HAND_OPEN)
 steps=max(1,int(DURATION/.02))
 for i in range(steps):
  u=(i+1)/steps;u=3*u*u-2*u*u*u;n.send({name:start[name]+u*(goal[name]-start[name]) for name in CONTROL},.02,'APPROACH',HAND_OPEN)
 n.send(goal,HOLD,'PREGRASP_HOLD',HAND_OPEN);reached={n_:float(n.arm[n_]) for n_ in CONTROL}
 preclose_error=max(abs(reached[name]-goal[name]) for name in LEFT+RIGHT)
 preclose_waist_error=max((abs(reached[name]-goal[name]) for name in WAIST),default=0.)
 arrival=native_circle_arrival(CIRCLE_CONTRACT)
 print(json.dumps({'native_tcp_arrival':arrival}),flush=True)
 correction_target=dict(goal); correction_issued=False; correction_error=None; correction_iterations=0
 closure_issued=False
 # Retain the existing bilateral 0.08 rad execution threshold. Failed arm
 # tracking must never trigger closure just because the trajectory timer ended.
 if HAND_RIGHT is not None and n.arm_fresh() and preclose_error<=.08 and preclose_waist_error<=.05 and arrival["pass"]:
  n.send(goal,HAND_HOLD,'HAND_CLOSE_HOLD',HAND_RIGHT);closure_issued=True
 if RESTORE:
  for i in range(steps):
   u=(i+1)/steps;u=3*u*u-2*u*u*u;n.send({name:goal[name]+u*(start[name]-goal[name]) for name in CONTROL},.02,'RESTORE',HAND_OPEN)
  # Keep the source-gain controller at the restored pose long enough for the
  # existing 0.04-rad recovery gate to measure a settled state.
  n.send(start,1.0,'RESTORE_HOLD',HAND_OPEN);restored={n_:float(n.arm[n_]) for n_ in CONTROL};max_restore=max(abs(restored[n_]-start[n_]) for n_ in CONTROL)
 else:
  n.send(goal,1.0,'PREGRASP_STABLE_HOLD',HAND_RIGHT if closure_issued else HAND_OPEN);restored={n_:float(n.arm[n_]) for n_ in CONTROL};max_restore=None
 max_goal=max(abs(reached[n_]-goal[n_]) for n_ in CONTROL)
 drift=math.hypot(n.odom[-1][0]-n.odom[0][0],n.odom[-1][1]-n.odom[0][1]) if len(n.odom)>=2 else None; stable=len(n.odom)>=100 and drift is not None and drift<=.05 and n.tilt and max(n.tilt)<=math.radians(15)
 # No installed native-MjData/HandState mapping is calibrated for this overlay.
 # Do not promote a normalized zero command, absent feedback, or q~=0 into a
 # physical-grasp acknowledgement. Preserve observations pending calibration.
 close_rows=[x for x in n.trace if x['stage']=='HAND_CLOSE_HOLD']
 contact_result=native_bilateral_contact(min((x['host_monotonic_s'] for x in close_rows),default=float('inf'))) if HAND_RIGHT is not None else {'pass':True,'classification':'NO_CLOSE_REQUESTED'}
 hand_pass=HAND_RIGHT is None or contact_result['pass']
 print(json.dumps({'native_tcp_arrival':arrival,'correction_issued':correction_issued,'correction_error_rad':correction_error,'correction_iterations':correction_iterations,'preclose_waist_error_rad':preclose_waist_error,'preclose_arm_error_rad':preclose_error,'closure_issued':closure_issued,'arm_feedback_fresh':n.arm_fresh(),'arm_callback_count':n.arm_callbacks,'arm_late_sample_discard_count':n.arm_regressions,'hand_mapping_status':'COMMAND_SENT; native qpos/actuator mapping not yet calibrated','physical_bilateral_contact':contact_result}),flush=True)
 print(json.dumps({'plan_mode':'ROUND_TRIP' if RESTORE else ('PREGRASP_CLOSE_HOLD' if HAND_RIGHT is not None else 'PREGRASP_HOLD'),'subscription_count':n.pub.get_subscription_count(),'hand_subscription_count':n.hand_pub.get_subscription_count(),'start_arm_rad':start,'state_fallback_joint_names':fallback_used,'goal_arm_rad':goal,'reached_arm_rad':reached,'restored_arm_rad':restored,'right_hand_command':HAND_RIGHT,'physical_bilateral_contact':contact_result,'final_hand_state_rad':n.hand,'hand_response_pass':hand_pass,'max_goal_error_rad':max_goal,'max_restore_error_rad':max_restore,'response_pass':max_goal<=.08 and preclose_waist_error<=.05 and hand_pass and ((not RESTORE) or max_restore<=.04),'motion_stability':{'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None,'pass':stable},'trace':n.trace}))
finally:
 with open('/workspace/evidence/native_hal_pregrasp_trace.json','w') as stream:json.dump({'trace':n.trace,'odom_trace':n.odom_trace,'last_controlled_state_rad':n.arm,'last_unique_source_stamps':n.arm_stamp},stream)
 n.destroy_node();rclpy.shutdown()
""".replace("__LEFT_TARGET__", json.dumps(left_target)).replace("__RIGHT_TARGET__", json.dumps(right_target)).replace("__GAINS__", serialized_gains).replace("__DURATION__", f"{duration_s:.17g}").replace("__HOLD__", f"{hold_s:.17g}").replace("__RESTORE__", "True" if restore else "False").replace("__HAND_RIGHT__", "None" if right_hand_command is None else f"{right_hand_command:.17g}").replace("__HAND_HOLD__", f"{hand_hold_s:.17g}").replace("__INITIAL_FALLBACK__", json.dumps(initial_arm_state or {}, sort_keys=True, allow_nan=False)).replace("__WAIST_NAMES__", repr(waist_names if waist_target is not None else ())).replace("__WAIST_TARGET__", repr(waist_target or [])).replace("__HAND_OPEN__", repr(hand_open_command)).replace("__CIRCLE_CONTRACT__", repr(circle_contract)).replace("__USE_NATIVE_BIAS__", repr(use_native_bias_effort)).replace("__SOURCE_LIMITS__", repr(source_joint_limits or {})).replace("__ENABLE_CORRECTION__", repr(enable_measured_correction))


def build_hal_bilateral_upperbody_plan_script(
    gains: dict[str, tuple[float, float]],
    right_target: list[float],
    waist_target: list[float],
    *,
    duration_s: float,
    hold_s: float,
    start_marker: str | None = None,
) -> str:
    """Build a bounded arm-plus-waist HAL pregrasp and safe restore.

    Legs remain owned by the delivered standing controller.  This function is
    intentionally separate from the arm-only probe because the static visual
    candidate proved that the measured standing waist must participate in the
    reachable double-pad posture.
    """
    arm_gains = {name: values for name, values in gains.items() if name.startswith(("left_", "right_"))}
    waist_names = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
    waist_gains = {name: gains[name] for name in waist_names}
    if len(arm_gains) != 14 or len(right_target) != 7 or len(waist_target) != 3:
        raise ValueError("upper-body plan requires fourteen arm gains, three waist gains, and complete targets")
    if not all(math.isfinite(value) for value in right_target + waist_target):
        raise ValueError("upper-body targets must be finite")
    if not 4.0 <= duration_s <= 12.0 or not 0.5 <= hold_s <= 2.0:
        raise ValueError("upper-body plan duration/hold are outside bounded range")
    serialized_arm_gains = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in arm_gains.items()}, sort_keys=True)
    serialized_waist_gains = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in waist_gains.items()}, sort_keys=True)
    marker_literal = "None" if start_marker is None else json.dumps(start_marker)
    return """#!/usr/bin/env python3
import json, math, os, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

LEFT=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint')
RIGHT=('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
WAIST=('waist_yaw_joint','waist_pitch_joint','waist_roll_joint')
TARGET_RIGHT=__RIGHT_TARGET__; TARGET_WAIST=__WAIST_TARGET__; ARM_GAINS=__ARM_GAINS__; WAIST_GAINS=__WAIST_GAINS__; DURATION=__DURATION__; HOLD=__HOLD__; START_MARKER=__START_MARKER__
class Plan(Node):
 def __init__(self):
  super().__init__('robotsim_hal_bilateral_upperbody_pregrasp');q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT);self.arm={};self.waist={};self.odom=[];self.tilt=[];self.seq=0;self.trace=[]
  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',lambda m:self.arm.update({j.name:float(j.position) for j in m.joints}),q)
  self.create_subscription(JointStateArray,'/aima/hal/joint/waist/state',lambda m:self.waist.update({j.name:float(j.position) for j in m.joints}),q)
  self.create_subscription(Odometry,'/aima/hal/odom/state',lambda m:self.odom.append((float(m.pose.pose.position.x),float(m.pose.pose.position.y))),q)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.imu,q);self.arm_pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/arm/command',q);self.waist_pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/waist/command',q)
 def imu(self,m):
  q=m.orientation;p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))));r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y));self.tilt.append(max(abs(r),abs(p)))
 def spin(self,s):
  end=time.monotonic()+s
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.002)
 def send(self,arm_target,waist_target,seconds,stage):
  end=time.monotonic()+seconds;next_tick=time.monotonic();started=time.monotonic()
  while rclpy.ok() and time.monotonic()<end:
   arm_msg=JointCommandArray();arm_msg.header=MessageHeader();arm_msg.header.stamp=self.get_clock().now().to_msg();arm_msg.header.frame_id='hal_upperbody_pregrasp';arm_msg.header.sequence=self.seq
   for name in LEFT+RIGHT:
    x=JointCommand();x.name=name;x.position=float(arm_target[name]);x.velocity=0.;x.effort=.6;x.stiffness=float(ARM_GAINS[name][0]);x.damping=float(ARM_GAINS[name][1]);arm_msg.joints.append(x)
   waist_msg=JointCommandArray();waist_msg.header=MessageHeader();waist_msg.header.stamp=arm_msg.header.stamp;waist_msg.header.frame_id='hal_upperbody_pregrasp';waist_msg.header.sequence=self.seq;self.seq+=1
   for name in WAIST:
    x=JointCommand();x.name=name;x.position=float(waist_target[name]);x.velocity=0.;x.effort=0.;x.stiffness=float(WAIST_GAINS[name][0]);x.damping=float(WAIST_GAINS[name][1]);waist_msg.joints.append(x)
   self.arm_pub.publish(arm_msg);self.waist_pub.publish(waist_msg);rclpy.spin_once(self,timeout_sec=.001)
   if not self.trace or time.monotonic()-self.trace[-1]['time_s']>=.05:
    arm_err=max(abs(float(self.arm.get(n,float('nan')))-arm_target[n]) for n in LEFT+RIGHT) if all(n in self.arm for n in LEFT+RIGHT) else float('inf')
    waist_err=max(abs(float(self.waist.get(n,float('nan')))-waist_target[n]) for n in WAIST) if all(n in self.waist for n in WAIST) else float('inf')
    self.trace.append({'time_s':time.monotonic()-started,'stage':stage,'max_arm_error_rad':arm_err,'max_waist_error_rad':waist_err,'right':{n:self.arm.get(n) for n in RIGHT},'waist':{n:self.waist.get(n) for n in WAIST}})
   next_tick+=.02;time.sleep(max(0,next_tick-time.monotonic()))
rclpy.init();n=Plan()
try:
 n.spin(2.0)
 if any(name not in n.arm for name in LEFT+RIGHT) or any(name not in n.waist for name in WAIST):raise RuntimeError('missing complete upper-body state')
 start_arm={name:float(n.arm[name]) for name in LEFT+RIGHT}; start_waist={name:float(n.waist[name]) for name in WAIST}; goal_arm=dict(start_arm);goal_arm.update(dict(zip(RIGHT,TARGET_RIGHT)));goal_waist=dict(zip(WAIST,TARGET_WAIST))
 if max(abs(goal_arm[n]-start_arm[n]) for n in LEFT+RIGHT)>1.1 or max(abs(goal_waist[n]-start_waist[n]) for n in WAIST)>.45:raise RuntimeError('upper-body pregrasp target exceeds bounded delta')
 if START_MARKER is not None:
  deadline=time.monotonic()+55.0
  while time.monotonic()<deadline and not os.path.exists(START_MARKER):
   n.send(start_arm,start_waist,.02,'DOCK_HOLD')
  if not os.path.exists(START_MARKER):raise RuntimeError('upper-body pregrasp start marker timed out')
 steps=max(1,int(DURATION/.02))
 for i in range(steps):
  u=(i+1)/steps;u=3*u*u-2*u*u*u;n.send({n_:start_arm[n_]+u*(goal_arm[n_]-start_arm[n_]) for n_ in LEFT+RIGHT},{n_:start_waist[n_]+u*(goal_waist[n_]-start_waist[n_]) for n_ in WAIST},.02,'APPROACH')
 n.send(goal_arm,goal_waist,HOLD,'PREGRASP_HOLD');reached_arm={n_:float(n.arm[n_]) for n_ in LEFT+RIGHT};reached_waist={n_:float(n.waist[n_]) for n_ in WAIST}
 for i in range(steps):
  u=(i+1)/steps;u=3*u*u-2*u*u*u;n.send({n_:goal_arm[n_]+u*(start_arm[n_]-goal_arm[n_]) for n_ in LEFT+RIGHT},{n_:goal_waist[n_]+u*(start_waist[n_]-goal_waist[n_]) for n_ in WAIST},.02,'RESTORE')
 n.send(start_arm,start_waist,.5,'RESTORE_HOLD');restored_arm={n_:float(n.arm[n_]) for n_ in LEFT+RIGHT};restored_waist={n_:float(n.waist[n_]) for n_ in WAIST}
 max_goal_arm=max(abs(reached_arm[n_]-goal_arm[n_]) for n_ in LEFT+RIGHT);max_goal_waist=max(abs(reached_waist[n_]-goal_waist[n_]) for n_ in WAIST);max_restore=max([abs(restored_arm[n_]-start_arm[n_]) for n_ in LEFT+RIGHT]+[abs(restored_waist[n_]-start_waist[n_]) for n_ in WAIST])
 drift=math.hypot(n.odom[-1][0]-n.odom[0][0],n.odom[-1][1]-n.odom[0][1]) if len(n.odom)>=2 else None;stable=len(n.odom)>=100 and drift is not None and drift<=.05 and n.tilt and max(n.tilt)<=math.radians(15)
 print(json.dumps({'arm_subscription_count':n.arm_pub.get_subscription_count(),'waist_subscription_count':n.waist_pub.get_subscription_count(),'start_arm_rad':start_arm,'start_waist_rad':start_waist,'goal_arm_rad':goal_arm,'goal_waist_rad':goal_waist,'reached_arm_rad':reached_arm,'reached_waist_rad':reached_waist,'restored_arm_rad':restored_arm,'restored_waist_rad':restored_waist,'max_goal_arm_error_rad':max_goal_arm,'max_goal_waist_error_rad':max_goal_waist,'max_restore_error_rad':max_restore,'response_pass':max_goal_arm<=.08 and max_goal_waist<=.05 and max_restore<=.04,'motion_stability':{'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None,'pass':stable},'trace':n.trace}))
finally:n.destroy_node();rclpy.shutdown()
""".replace("__RIGHT_TARGET__", json.dumps(right_target)).replace("__WAIST_TARGET__", json.dumps(waist_target)).replace("__ARM_GAINS__", serialized_arm_gains).replace("__WAIST_GAINS__", serialized_waist_gains).replace("__DURATION__", f"{duration_s:.17g}").replace("__HOLD__", f"{hold_s:.17g}").replace("__START_MARKER__", marker_literal)


def build_hal_waist_pd_probe_script(gains: dict[str, tuple[float, float]]) -> str:
    """Probe the delivered waist HAL channel with its own source gains.

    This is deliberately a 0.03-rad round trip.  It establishes whether the
    simulator's documented waist command subscriber can coexist with native
    STAND_DEFAULT before any grasp posture is attempted.
    """
    selected = {name: gains[name] for name in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")}
    serialized = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in selected.items()}, sort_keys=True)
    return """#!/usr/bin/env python3
import json, math, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
ORDER=('waist_yaw_joint','waist_pitch_joint','waist_roll_joint'); TARGET='waist_roll_joint'; GAINS=__GAINS__
class Probe(Node):
 def __init__(self):
  super().__init__('robotsim_stl_hal_waist_pd_probe'); q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT); self.state={}; self.odom=[]; self.tilt=[]
  self.create_subscription(JointStateArray,'/aima/hal/joint/waist/state',lambda m:setattr(self,'state',{x.name:float(x.position) for x in m.joints}),q)
  self.create_subscription(Odometry,'/aima/hal/odom/state',lambda m:self.odom.append((float(m.pose.pose.position.x),float(m.pose.pose.position.y))),q)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.imu,q); self.pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/waist/command',q)
 def imu(self,m):
  q=m.orientation; p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x)))); r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(r),abs(p)))
 def spin(self,s):
  end=time.monotonic()+s
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.002)
 def send(self,target,s):
  end=time.monotonic()+s; seq=0
  while rclpy.ok() and time.monotonic()<end:
   m=JointCommandArray();m.header=MessageHeader();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='hal_waist_probe';m.header.sequence=seq
   for n in ORDER:
    x=JointCommand();x.name=n;x.position=float(target[n]);x.velocity=0.;x.effort=0.;x.stiffness=float(GAINS[n][0]);x.damping=float(GAINS[n][1]);m.joints.append(x)
   self.pub.publish(m);rclpy.spin_once(self,timeout_sec=.002);seq+=1;time.sleep(.01)
rclpy.init();n=Probe()
try:
 n.spin(2.0);
 if any(k not in n.state for k in ORDER):raise RuntimeError('missing complete waist state')
 start={k:float(n.state[k]) for k in ORDER}; target=dict(start);target[TARGET]+=0.03; subs=n.pub.get_subscription_count();n.send(target,2.);moved=float(n.state.get(TARGET,float('nan')));n.send(start,2.);restored=float(n.state.get(TARGET,float('nan')))
 drift=math.hypot(n.odom[-1][0]-n.odom[0][0],n.odom[-1][1]-n.odom[0][1]) if len(n.odom)>=2 else None; stability={'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None};stability['pass']=stability['sample_count']>=100 and drift is not None and drift<=.05 and stability['max_tilt_rad'] is not None and stability['max_tilt_rad']<=math.radians(15); delta=moved-start[TARGET]; err=abs(restored-start[TARGET]);print(json.dumps({'subscription_count':subs,'source_gains':GAINS[TARGET],'start_rad':start[TARGET],'goal_rad':target[TARGET],'moved_rad':moved,'restored_rad':restored,'delta_rad':delta,'restore_error_rad':err,'response_pass':subs>=1 and .005<=delta<=.08 and err<=.04,'motion_stability':stability}))
finally:n.destroy_node();rclpy.shutdown()
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
    # The delivered HAL can publish a transient partial JointStateArray when
    # STAND_DEFAULT switches its internal runners.  Retain the latest value
    # per named joint until a complete sampled state is available instead of
    # mistaking that transient packet for an erased robot state.
    def on_arm(self,msg): self.arm.update({item.name:float(item.position) for item in msg.joints})
    def on_leg(self,msg): self.leg.update({item.name:float(item.position) for item in msg.joints})
    def on_waist(self,msg): self.waist.update({item.name:float(item.position) for item in msg.joints})
    def on_odom(self,msg):
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        self.odom.append((float(p.x),float(p.y),float(p.z),float(q.w),float(q.x),float(q.y),float(q.z)))
    def on_imu(self,msg):
        q=msg.orientation; pitch=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x)))); roll=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y)); self.tilt.append(max(abs(roll),abs(pitch)))
rclpy.init(); node=Capture()
try:
    end=time.monotonic()+4.0
    while rclpy.ok() and time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=0.02)
        if all(name in node.arm for name in ARM_ORDER) and all(name in node.leg for name in LEG_ORDER) and all(name in node.waist for name in WAIST_ORDER) and len(node.odom)>=100:
            break
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
    expected={'leg':LEG_ORDER,'waist':WAIST_ORDER}
    for area,names in expected.items():
        if node.samples[area] < 10 or any(name not in node.commands[area] for name in names): raise RuntimeError('missing complete '+area+' command stream')
        for name in names:
            if not all(math.isfinite(node.commands[area][name][field]) for field in ('position','velocity','effort','stiffness','damping')): raise RuntimeError('non-finite '+area+' command')
    arm_complete=all(name in node.commands['arm'] for name in ARM_ORDER)
    print(json.dumps({'sample_counts':node.samples,'arm_command_stream_complete':arm_complete,'arm_commands':{name:node.commands['arm'][name] for name in ARM_ORDER} if arm_complete else {},'leg_commands':{name:node.commands['leg'][name] for name in LEG_ORDER},'waist_commands':{name:node.commands['waist'][name] for name in WAIST_ORDER}}))
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
    parser.add_argument("--run-hal-waist-pd-probe", action="store_true", help="direct 0.03 rad waist-roll probe using delivered waist gains during native standing")
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
    if not args.overlay_result.is_file() or not 2.0 <= args.startup_timeout_s <= 35.0 or not -1.0 <= args.probe_command_position <= 1.0 or (args.request_stand and not args.start_mc) or (args.request_upperbody_split and not (args.start_mc and args.request_stand)) or (args.run_upper_body_probe and not (args.start_mc and args.request_stand and args.request_upperbody_split)) or (args.run_upper_body_probe_in_stand and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_probe and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_pd_probe and not (args.start_mc and args.request_stand and args.disable_mc_arm_publisher)) or (args.run_hal_waist_pd_probe and not (args.start_mc and args.request_stand)) or (args.capture_arm_state and not (args.start_mc and args.request_stand)) or (args.capture_stand_commands and not (args.start_mc and args.request_stand)) or (args.run_hal_arm_plan and not (args.start_mc and args.request_stand and all(plan_inputs))) or ((not args.run_hal_arm_plan) and any(plan_inputs)) or (args.disable_mc_arm_publisher and not (args.start_mc and (args.run_hal_arm_probe or args.run_hal_arm_pd_probe or args.run_hal_arm_plan))):
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
        "gates": {"HOST_PREFLIGHT": "NOT_RUN", "NATIVE_SIMULATOR_MODEL_LOAD": "NOT_RUN", "HAND_COMMAND_SUBSCRIBER": "NOT_RUN", "HAND_STATE_PUBLISHER": "NOT_RUN", "HAL_HAND_COMMAND_ROUNDTRIP": "NOT_RUN", "STL_HAND_COMMAND_STAND_STABILITY": "NOT_RUN", "MC_INITIALIZATION": "NOT_RUN", "NATIVE_STAND_REQUEST": "NOT_RUN", "NATIVE_STAND_READBACK": "NOT_RUN", "NATIVE_STAND_STABILITY": "NOT_RUN", "MC_STAND_COMMAND_CAPTURE": "NOT_RUN", "HAL_ARM_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_ARM_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PD_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_ARM_PD_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_WAIST_PD_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_WAIST_PD_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PLAN_ROUNDTRIP": "NOT_RUN", "HAL_ARM_PLAN_STAND_STABILITY": "NOT_RUN", "HAL_ARM_PLAN_RESTORE": "NOT_RUN", "UPPERBODY_SPLIT_REQUEST": "NOT_RUN", "UPPERBODY_SPLIT_READBACK": "NOT_RUN", "UPPER_BODY_COMMAND_ROUNDTRIP": "NOT_RUN", "UPPER_BODY_COMMAND_STAND_STABILITY": "NOT_RUN", "UPPER_BODY_STAND_COMMAND_ROUNDTRIP": "NOT_RUN", "UPPER_BODY_STAND_COMMAND_STABILITY": "NOT_RUN", "DYNAMICS_QUALIFICATION": "NOT_RUN"},
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
        if args.run_hal_waist_pd_probe:
            source_gains = load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_waist_pd_probe.py", build_hal_waist_pd_probe_script(source_gains))
            result["source_waist_pd_gains"] = {name: {"stiffness": source_gains[name][0], "damping": source_gains[name][1]} for name in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")}
        if args.run_hal_arm_plan:
            assert args.ik_validation_result is not None and args.ik_validation_result_sha256 is not None
            if not args.ik_validation_result.is_file() or sha256(args.ik_validation_result) != args.ik_validation_result_sha256:
                raise ValueError("IK validation result is missing or hash-mismatched")
            ik_payload = json.loads(args.ik_validation_result.read_text(encoding="utf-8"))
            external_ik = (
                ik_payload.get("classification") == "SIMULATION_ONLY_STATIC_EXTERNAL_IK_REFERENCE"
                and ik_payload.get("gates", {}).get("IK_CONVERGED") == "PASS"
                and ik_payload.get("gates", {}).get("POST_IK_TABLE_CLEARANCE") == "PASS"
            )
            native_truth_ik = (
                ik_payload.get("classification") == "SIMULATION_ONLY_STATIC_NATIVE_STAND_TRUTH_IK"
                and all(ik_payload.get("gates", {}).get(name) == "PASS" for name in ("TRUTH_IK_ALIGNMENT", "JOINT_LIMIT_MARGIN", "TABLE_CLEARANCE", "OPEN_PREGRASP_NONPAD_CLEAR"))
            )
            if not (external_ik or native_truth_ik):
                raise ValueError("IK validation does not pass standing/table preflight")
            right_target = ik_payload.get("ik", {}).get("right_arm_q_rad")
            if native_truth_ik:
                named_target = ik_payload.get("right_arm_q_rad")
                right_target = [named_target.get(name) for name in RIGHT_ARM_ORDER] if isinstance(named_target, dict) else None
            if not isinstance(right_target, list) or len(right_target) != len(RIGHT_ARM_ORDER):
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
                if args.run_hal_waist_pd_probe and result["gates"]["NATIVE_STAND_STABILITY"] == "PASS":
                    waist_probe = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 14s python3 /workspace/evidence/native_hal_waist_pd_probe.py"], timeout_s=16.0, check=False)
                    pg.write_new_text(output / "native_hal_waist_pd_probe.stdout", waist_probe.stdout)
                    try:
                        waist_result = json.loads(waist_probe.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        waist_result = {"parse_error": True}
                    result["hal_waist_pd_probe"] = {"returncode": waist_probe.returncode, "result": waist_result, "stdout": waist_probe.stdout}
                    result["gates"]["HAL_WAIST_PD_COMMAND_ROUNDTRIP"] = "PASS" if waist_probe.returncode == 0 and waist_result.get("response_pass") else "FAIL"
                    result["gates"]["HAL_WAIST_PD_COMMAND_STAND_STABILITY"] = "PASS" if waist_probe.returncode == 0 and waist_result.get("motion_stability", {}).get("pass") else "FAIL"
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
