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


def build_native_carry_transition_setting(source, *, native_step=False):
    """Use the installed ActionSelector's existing 500ms smoothing schema.

    Only these two action-pair rules change. Controller runners, gains, policy
    files, constraints and source assets remain untouched.
    """
    import copy
    expected=[dict(from_= 'STAND_DEFAULT',to='WHOLE_BODY_TELEOP',duration_ms=500),
              dict(from_='WHOLE_BODY_TELEOP',to='STAND_DEFAULT',duration_ms=500)]
    expected=[{('from' if k=='from_' else k):v for k,v in row.items()} for row in expected]
    if source.get('smooth_transitions')!=expected or not {'STAND_DEFAULT','LOCOMOTION_DEFAULT'}.issubset(source.get('transition_actions',{})):
        raise ValueError('unsupported native transition-setting schema')
    result=copy.deepcopy(source)
    result['smooth_transitions']+= [
        {'from':'STAND_DEFAULT','to':'LOCOMOTION_DEFAULT','duration_ms':500},
        {'from':'LOCOMOTION_DEFAULT','to':'STAND_DEFAULT','duration_ms':500}]
    if native_step:
        if 'LOCOMOTION_STEP' not in source.get('transition_actions',{}):raise ValueError('native step action missing')
        result['smooth_transitions'] += [{'from':'STAND_DEFAULT','to':'LOCOMOTION_STEP','duration_ms':500},{'from':'LOCOMOTION_STEP','to':'STAND_DEFAULT','duration_ms':500}]
    return result


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


def build_arm_hand_split_mc_config(source: dict[str, object], *, isolate_disabled_topics=False) -> dict[str, object]:
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
        if isolate_disabled_topics:
            expected_topic="/aima/hal/joint/"+("arm" if expected=="arm_joint_command" else "hand")+"/command"
            if matches[0].get("topic")!=expected_topic:raise ValueError("unexpected native disabled publisher topic")
            matches[0]["topic"]="/robotsim/issue46/inactive_mc/"+("arm" if expected=="arm_joint_command" else "hand")+"/command"
            channels=[c for c in result.get("aimrt",{}).get("channel",{}).get("pub_topics_options",[]) if c.get("topic_name")==expected_topic]
            if len(channels)!=1:raise ValueError("expected exact native arm/hand ROS publisher channel")
            channels[0]["topic_name"]=matches[0]["topic"]
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


def build_native_arm_relay_config(source: dict[str, object], *, relay_waist=False) -> dict[str, object]:
    """Restore the vendor 29-DOF policy output; change transport ownership only."""
    result=build_coordinated_waist_relay_config(source) if relay_waist else build_arm_hand_split_mc_config(source)
    original="/aima/hal/joint/arm/command";private="/robotsim/issue46/mc/arm/command"
    entries=result["MotionControlModule"]["PublisherManager"]["publisher_cfg_list"]
    pubs=[x for x in entries if x.get("name")=="arm_joint_command"]
    channels=[x for x in result["aimrt"]["channel"]["pub_topics_options"] if x.get("topic_name")==original]
    if len(pubs)!=1 or pubs[0].get("enable") is not False or pubs[0].get("topic")!=original or len(channels)!=1:
        raise ValueError("expected exact source MC arm publisher and channel")
    pubs[0]["enable"]=True;pubs[0]["topic"]=private;channels[0]["topic_name"]=private
    return result


def build_stand_client_script(*, observe_only: bool = False) -> str:
    """Minimal equivalent of the delivered ``set_mc_action SD`` example."""
    code = """#!/usr/bin/env python3
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
        if sample['mode_name'] == 'STAND_DEFAULT' and sample['mode_status'] == 100:
            break
        time.sleep(0.3)
    deadline = time.monotonic() + 2.5
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
    if observe_only:
        start=code.index('    set_client = node.create_client(')
        end=code.index('    get_client = node.create_client(')
        code=code[:start]+code[end:]
        code=code.replace('int(set_result.response.status.value)','None').replace('set_result.response.message',"'READ_ONLY_NO_ACTION_REQUEST'")
    return code


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
    use_native_joint_feedback: bool = False,
    track_native_circle: bool = False,
    source_joint_limits: dict[str, list[float]] | None = None,
    enable_measured_correction: bool = False,
    enable_single_residual_feedforward: bool = False,
    lift_contract: dict | None = None,
) -> str:
    """Build a bounded direct-HAL bilateral pregrasp -> restore trajectory.

    This uses the delivered AnimationPlayer position gains and the existing
    arm HAL topic. Optional additive effort comes only from fresh native
    qfrc_bias, never an arbitrary force pulse. Native joint feedback is an
    explicitly labeled simulation-only alternative to ROS source timestamps.
    Source geometry, gains, limits and leg control remain unchanged.
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
    if lift_contract is not None:
        if restore or not track_native_circle or not use_native_joint_feedback or circle_contract is None or right_hand_command is None:raise ValueError('lift requires exclusive native tracked close-hold path')
        if lift_contract.get('pass') is not True or (lift_contract.get('height_m'),lift_contract.get('duration_s'),lift_contract.get('hold_s'))!=(.06,3.,1.5):raise ValueError('lift preflight or bounded profile invalid')
    serialized_gains = json.dumps({name: [stiffness, damping] for name, (stiffness, damping) in arm_gains.items()}, sort_keys=True)
    script = """#!/usr/bin/env python3
import json, math, os, time, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import HandCommand, HandCommandArray, HandStateArray, HandType, JointCommand, JointCommandArray, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

LEFT=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint')
RIGHT=('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
CIRCLE_CONTRACT=__CIRCLE_CONTRACT__; TRACK_NATIVE_CIRCLE=__TRACK_NATIVE_CIRCLE__; USE_NATIVE_FEEDBACK=__USE_NATIVE_FEEDBACK__; USE_NATIVE_BIAS=__USE_NATIVE_BIAS__; SOURCE_LIMITS=__SOURCE_LIMITS__; ENABLE_CORRECTION=__ENABLE_CORRECTION__; ENABLE_RESIDUAL_FF=__ENABLE_RESIDUAL_FF__; HAND_OPEN=__HAND_OPEN__; WAIST=__WAIST_NAMES__; TARGET_WAIST=__WAIST_TARGET__; CONTROL=LEFT+RIGHT+WAIST
TARGET_LEFT=__LEFT_TARGET__; TARGET_RIGHT=__RIGHT_TARGET__; GAINS=__GAINS__; DURATION=__DURATION__; HOLD=__HOLD__; RESTORE=__RESTORE__; HAND_RIGHT=__HAND_RIGHT__; HAND_HOLD=__HAND_HOLD__; INITIAL_FALLBACK=__INITIAL_FALLBACK__; LIFT_CONTRACT=__LIFT_CONTRACT__
def force_guided_close_step(command,lower,probe,normal,required,dt):
 if not all(math.isfinite(v) for v in [command,lower,probe,required,dt]+list(normal.values())) or required<=0 or not 0<=lower<=probe<=1:raise RuntimeError('INVALID_GRIP_FORCE_FEEDBACK')
 dt=min(.04,max(0.,dt))
 if min(normal.values())<required:command-=.04*dt
 elif min(normal.values())>1.5*required:command+=.02*dt
 return max(lower,min(probe,command))
def native_circle_arrival(contract):
 if contract is None:return {'pass':True,'classification':'LEGACY_NO_NATIVE_TCP_GATE'}
 path='/workspace/evidence/native_latest.json'
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
 target_source='NATIVE_TRUTH_DIAGNOSTIC';image_source_ns=None
 if contract.get('perception_snapshot'):
  with open(contract['perception_snapshot']) as f:snapshot=json.load(f)
  frozen=snapshot['frozen_at_host_ns'];original=snapshot['original_observation'];verification=snapshot['pre_execution_verification']
  if snapshot.get('schema')!='robotsim.issue46.stationary-rgbd-grasp-snapshot.v1' or snapshot.get('generation')!=row['generation'] or snapshot.get('valid_for_s')!=45. or snapshot.get('expected_object_stationary') is not True or not 0.<=(time.monotonic_ns()-frozen)*1e-9<=45.:raise RuntimeError('INVALID_OR_EXPIRED_RGBD_SNAPSHOT')
  if any(v.get('pass') is not True or v.get('generation')!=row['generation'] or v.get('object_truth_used_by_estimator') is not False for v in (original,verification)) or not 0.<=(frozen-verification['source_host_ns'])*1e-9<=.20 or verification['source_host_ns']<=original['source_host_ns']:raise RuntimeError('RGBD_SNAPSHOT_ACQUISITION_UNQUALIFIED')
  position=snapshot['estimated_center_world_m'];axis=snapshot['axis_world'];image_source_ns=original['source_host_ns']
  if position!=original['estimated_center_world_m'] or axis!=[0.,0.,1.] or not all(math.isfinite(v) for v in position) or math.sqrt(sum((a-b)**2 for a,b in zip(position,verification['estimated_center_world_m'])))>.002 or abs(original['diameter_m']-verification['diameter_m'])>.002:raise RuntimeError('RGBD_SNAPSHOT_TARGET_CHANGED')
  bottle={'position_m':position,'rotation':[1.,0.,0.,0.,1.,0.,0.,0.,1.]};target_source='RGBD_BOUNDED_STATIONARY_OBJECT_SNAPSHOT'
 if contract.get('perception_latest'):
  lock=None
  phase_path='/workspace/evidence/native_tracking_phase.json'
  if os.path.exists(phase_path):
   with open(phase_path) as f:lock=json.load(f)
  if lock and lock.get('phase')=='ARRIVAL_HOLD':
   age_lock=(time.monotonic_ns()-lock['created_host_ns'])*1e-9
   image_source_ns=lock['image_source_host_ns']
   if lock.get('native_arrival_pass') is not True or lock.get('target_source')!='RGBD_FROZEN_AT_VERIFIED_ARRIVAL' or lock.get('generation')!=row['generation'] or not 0.<=age_lock<=12. or not 0.<=(lock['created_host_ns']-image_source_ns)*1e-9<=.20:raise RuntimeError('INVALID_OR_EXPIRED_RGBD_ARRIVAL_LOCK')
   position=lock['bottle_center_world_m'];axis=lock['bottle_axis_world'];target_source='RGBD_FROZEN_AT_VERIFIED_ARRIVAL'
  else:
   with open(contract['perception_latest']) as f:observed=json.load(f)
   sensor_age=(time.monotonic_ns()-observed['source_host_ns'])*1e-9
   if observed.get('pass') is not True or observed.get('generation')!=row['generation'] or not 0.<=sensor_age<=.20 or observed.get('object_truth_used_by_estimator') is not False:raise RuntimeError('RGBD_ARRIVAL_UNQUALIFIED')
   position=observed['estimated_center_world_m'];axis=observed['axis_world'];image_source_ns=observed['source_host_ns'];target_source='FRESH_SIMULATED_HEAD_RGBD'
  if len(position)!=3 or not all(math.isfinite(v) for v in position) or axis!=[0.,0.,1.]:raise RuntimeError('INVALID_RGBD_ARRIVAL_POSE')
  bottle={'position_m':position,'rotation':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}
 R=hand['rotation'];B=bottle['rotation'];c=contract['circle_center_local_m'];height=contract['height_above_center_m']
 tcp=[hand['position_m'][i]+sum(R[3*i+j]*c[j] for j in range(3)) for i in range(3)]
 target=[bottle['position_m'][i]+height*B[3*i+2] for i in range(3)]
 error=math.sqrt(sum((tcp[i]-target[i])**2 for i in range(3)))
 nominal=contract.get('nominal_circle_center_local_m',c)
 nominal_tcp=[hand['position_m'][i]+sum(R[3*i+j]*nominal[j] for j in range(3)) for i in range(3)]
 nominal_error=math.sqrt(sum((nominal_tcp[i]-target[i])**2 for i in range(3)))
 dot=max(-1.,min(1.,sum(R[3*i]*B[3*i+2] for i in range(3))));raw_angle=math.acos(dot)
 local_axis=contract.get('local_bottle_axis_in_hand',[1.,0.,0.]);world_axis=[sum(R[3*i+j]*local_axis[j] for j in range(3)) for i in range(3)]
 angle=math.acos(max(-1.,min(1.,sum(world_axis[i]*B[3*i+2] for i in range(3)))))
 expected_raw=math.acos(max(-1.,min(1.,local_axis[0])))
 return {'pass':error<=.005 and nominal_error<=.005 and angle<=.03 and abs(raw_angle-expected_raw)<=.03,'position_error_m':error,'nominal_circle_error_m':nominal_error,'axis_error_rad':angle,'raw_hand_axis_to_bottle_rad':raw_angle,'planned_mesh_axis_offset_rad':expected_raw,'tcp_world_m':tcp,'target_world_m':target,'bottle_center_world_m':bottle['position_m'],'bottle_axis_world':[B[2],B[5],B[8]],'target_source':target_source,'image_source_host_ns':image_source_ns,'native_pose_age_s':age,'sim_time_s':row['sim_time_s'],'classification':'SIMULATION_ONLY_NATIVE_TCP_ARRIVAL'}
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
  if all(v>=.1 for v in forces.values()):
   if bilateral and (int(row['host_monotonic_ns'])-bilateral[-1][0])*1e-9>.10:bilateral=[]
   bilateral.append((int(row['host_monotonic_ns']),forces,row['sim_time_s']))
  else:bilateral=[]
 if len(bilateral)<5 or (bilateral[-1][0]-bilateral[0][0])*1e-9<.20 or any((b[0]-a[0])*1e-9>.10 for a,b in zip(bilateral,bilateral[1:])):return {'pass':False,'sample_count':len(bilateral),'minimum_force_n':{n:min((x[1][n] for x in bilateral),default=0.) for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')},'classification':'NO_SUSTAINED_BILATERAL_PHYSICAL_CONTACT'}
 return {'pass':True,'sample_count':len(bilateral),'duration_s':(bilateral[-1][0]-bilateral[0][0])*1e-9,'minimum_force_n':{n:min(x[1][n] for x in bilateral) for n in ('R_hand_narrow3_Link','R_hand_wide3_Link')},'sim_time_range_s':[bilateral[0][2],bilateral[-1][2]],'classification':'SUSTAINED_BILATERAL_PHYSICAL_CONTACT'}
class Plan(Node):
 def __init__(self):
  super().__init__('robotsim_hal_bilateral_pregrasp');q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT);hand_q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.TRANSIENT_LOCAL);self.arm={};self.hand={};self.odom=[];self.tilt=[];self.seq=0;self.trace=[]
  # The simulator may publish the two arm groups in separate messages.  Merge
  # named samples so a later partial packet cannot discard the already-read
  # opposite-arm state before the bounded plan begins.
  self.odom_trace=[];self.arm_receipt={};self.arm_stamp={};self.arm_regressions=0;self.arm_callbacks=0;self.run_started=time.monotonic();self.last_trace_at=None;self.native_row=None;self.native_receipt_ns=None;self.native_generation=None;self.native_sim_time=None;self.ros_arm={};self.ff_offset={n:0. for n in RIGHT};self.approach_start=None;self.approach_fraction=0.;self.endpoint=None;self.initial_bottle_position=None;self.closing_started=False;self.close_publish_count=0
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
    if self.arm_regressions<=2 or self.arm_regressions%500==0:print(json.dumps({'arm_feedback_late_sample_discarded':fault,'regression_count':self.arm_regressions}),flush=True)
    # DDS can deliver an older source-stamped frame after a newer one. Never
    # apply or refresh from it. Keep consuming only strictly increasing frames;
    # if those stop, arm_fresh() fails against the last unique receipt age.
    if not USE_NATIVE_FEEDBACK and now-self.arm_receipt.get(item.name,0.)>.20:
     raise RuntimeError('STALE_ARM_FEEDBACK_AFTER_SOURCE_REGRESSION:'+json.dumps(fault))
    continue
   if stamp>0 and (old is None or stamp>old):
    self.ros_arm[item.name]=float(item.position)
    if not USE_NATIVE_FEEDBACK:self.arm[item.name]=float(item.position)
    self.arm_receipt[item.name]=now;self.arm_stamp[item.name]=stamp
 def refresh_native(self):
  if not USE_NATIVE_FEEDBACK:return
  path='/workspace/evidence/native_latest.json'
  with open(path,'rb') as f:
   f.seek(0,2);f.seek(max(0,f.tell()-262144));raw=f.read()
  lines=raw.splitlines()
  if not raw.endswith(bytes([10])):lines=lines[:-1]
  for line in reversed(lines):
   try:row=json.loads(line);break
   except (ValueError,UnicodeDecodeError):continue
  else:raise RuntimeError('NO_COMPLETE_NATIVE_ARM_SAMPLE')
  stamp=int(row['host_monotonic_ns']);age=(time.monotonic_ns()-stamp)*1e-9
  if row.get('engine_version')!=337 or not 0.<=age<=.20:raise RuntimeError('STALE_OR_WRONG_ABI_NATIVE_ARM_SAMPLE:'+json.dumps({'age_s':age,'engine':row.get('engine_version'),'source_ns':stamp,'host_ns':time.monotonic_ns()}))
  if self.native_generation is not None and row['generation']!=self.native_generation:raise RuntimeError('UNEXPECTED_NATIVE_RESET_DURING_ARM_CONTROL')
  if self.native_receipt_ns is not None and (stamp<self.native_receipt_ns or row['sim_time_s']<self.native_sim_time):raise RuntimeError('REGRESSING_NATIVE_ARM_SAMPLE')
  values={}
  for name in CONTROL:
   j=row.get('joints',{}).get(name)
   if j is None or not math.isfinite(float(j['q'])) or not math.isfinite(float(j['qvel'])):raise RuntimeError('MISSING_OR_INVALID_NATIVE_ARM_JOINT:'+name)
   values[name]=float(j['q'])
  if TRACK_NATIVE_CIRCLE and not self.closing_started:
   bottle=row['bodies']['test_cylinder'];position=bottle['position_m']
   if self.initial_bottle_position is None:self.initial_bottle_position=list(position)
   if math.sqrt(sum((a-b)**2 for a,b in zip(position,self.initial_bottle_position)))>.01 or bottle['rotation'][8]<math.cos(.10):raise RuntimeError('BOTTLE_MOVED_OR_TILTED_BEFORE_CLOSURE')
  for contact in row.get('contacts',[]):
   pair={contact.get('body1'),contact.get('body2')}
   if 'test_cylinder' in pair and not pair.issubset({'test_cylinder','raicom_table_table','R_hand_narrow3_Link','R_hand_wide3_Link'}) and float(contact['local_force_torque'][0])>.1:
    raise RuntimeError('FORBIDDEN_NATIVE_BOTTLE_CONTACT:'+json.dumps(contact))
  self.arm.update(values);self.native_row=row;self.native_receipt_ns=stamp;self.native_generation=row['generation'];self.native_sim_time=row['sim_time_s']
 def arm_fresh(self,max_age=.20):
  if USE_NATIVE_FEEDBACK:return self.native_receipt_ns is not None and 0.<=(time.monotonic_ns()-self.native_receipt_ns)*1e-9<=max_age and all(n in self.arm for n in CONTROL)
  now=time.monotonic()
  return all(name in self.arm_receipt and now-self.arm_receipt[name]<=max_age for name in CONTROL)
 def on_hand(self,m):
  for item in list(m.left_hands)+list(m.right_hands):self.hand[{'L_claw_joint':'left_claw_joint','R_claw_joint':'right_claw_joint'}.get(item.name,item.name)]=float(item.position)
 def imu(self,m):
  q=m.orientation;p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))));r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y));self.tilt.append(max(abs(r),abs(p)))
 def spin(self,s):
  end=time.monotonic()+s
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.002)
 def tracking_target(self):
  with open('/workspace/evidence/native_tracking_target.json') as f:r=json.load(f)
  if not r.get('pass'):raise RuntimeError('NATIVE_CIRCLE_TRACKER_FAILED:'+str(r))
  age=(time.monotonic_ns()-int(r['source_host_ns']))*1e-9
  if not 0<=age<=.20 or r['generation']!=self.native_generation:raise RuntimeError('STALE_NATIVE_CIRCLE_TRACKING_TARGET:'+json.dumps({'source_age_s':age,'target_created_age_s':(time.monotonic_ns()-int(r['created_host_ns']))*1e-9,'target_generation':r['generation'],'native_generation':self.native_generation,'native_sample_age_s':(time.monotonic_ns()-self.native_receipt_ns)*1e-9}))
  goal={n:float(r['right_target_rad'][n]) for n in RIGHT}
  if any(not math.isfinite(goal[n]) for n in RIGHT):raise RuntimeError('NONFINITE_NATIVE_CIRCLE_TARGET')
  self.endpoint=goal
  self.tracking_metrics=r.get('metrics',{})
  return goal
 def write_phase(self,phase):
  with open('/workspace/evidence/native_tracking_phase.tmp','w') as stream:json.dump(phase,stream)
  os.replace('/workspace/evidence/native_tracking_phase.tmp','/workspace/evidence/native_tracking_phase.json')
 def native_bias(self):
  if not USE_NATIVE_BIAS:return {}
  path='/workspace/evidence/native_latest.json'
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
   self.refresh_native()
   if CIRCLE_CONTRACT and CIRCLE_CONTRACT.get('perception_latest') and not self.closing_started and self.approach_fraction>=1. and not os.path.exists('/workspace/evidence/native_tracking_phase.json'):
    observed_arrival=native_circle_arrival(CIRCLE_CONTRACT)
    if observed_arrival['pass']:
     lock={'phase':'ARRIVAL_HOLD','native_arrival_pass':True,'target_world_m':observed_arrival['target_world_m'],'bottle_center_world_m':observed_arrival['bottle_center_world_m'],'bottle_axis_world':observed_arrival['bottle_axis_world'],'target_source':'RGBD_FROZEN_AT_VERIFIED_ARRIVAL','image_source_host_ns':observed_arrival['image_source_host_ns'],'generation':self.native_generation,'created_host_ns':time.monotonic_ns(),'first_measured_arrival':observed_arrival}
     if not 0.<=(lock['created_host_ns']-lock['image_source_host_ns'])*1e-9<=.20:raise RuntimeError('RGBD_LOCK_SOURCE_EXPIRED_BEFORE_HANDOFF')
     self.write_phase(lock);print(json.dumps({'rgbd_target_locked_after_actual_arrival':lock}),flush=True)
   if stage in ('LIFT_PROBE','LIFT'):
    u=min(1.,(time.monotonic()-start)/seconds);smooth=3*u*u-2*u*u*u
    self.lift_phase['lift_offset_m']=.008*smooth if stage=='LIFT_PROBE' else .008+.052*smooth
    self.write_phase(self.lift_phase)
   if stage in ('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD'):
    forces={name:sum(max(0.,float(c['local_force_torque'][0])) for c in self.native_row.get('contacts',[]) if 'test_cylinder' in (c.get('body1'),c.get('body2')) and name in (c.get('body1'),c.get('body2'))) for name in ('R_hand_narrow3_Link','R_hand_wide3_Link')}
    if min(forces.values())<.1:
     self.lift_contact_lost_since=self.lift_contact_lost_since or time.monotonic()
     if time.monotonic()-self.lift_contact_lost_since>.10:raise RuntimeError('BILATERAL_CONTACT_LOST_DURING_LIFT')
    else:self.lift_contact_lost_since=None
   if TRACK_NATIVE_CIRCLE and stage not in ('HAND_ACTIVATION','HAND_OPEN_PREGRASP'):
    endpoint=self.tracking_target();target=dict(target)
    if stage in ('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD'):
     for name in RIGHT:
      correction=float(self.tracking_metrics['cartesian_correction_rad'][name])
      if not math.isfinite(correction):raise RuntimeError('NONFINITE_LIFT_TRACKING_CORRECTION')
      limits=SOURCE_LIMITS[name]
      command=endpoint[name]+self.ff_offset[name]+max(-.002,min(.002,.05*correction))
      self.ff_offset[name]=max(limits[0]+.01,min(limits[1]-.01,command))-endpoint[name]
    for name in RIGHT:
     desired=endpoint[name]+self.ff_offset[name]
     if stage=='APPROACH':desired=self.approach_start[name]+self.approach_fraction*(desired-self.approach_start[name])
     limits=SOURCE_LIMITS.get(name)
     if limits is None or not limits[0]+.001<=desired<=limits[1]-.001:raise RuntimeError('LIVE_TARGET_SOURCE_LIMIT:'+name)
     target[name]=desired
   if not self.arm_fresh():raise RuntimeError('STALE_BILATERAL_ARM_FEEDBACK_DURING:'+stage)
   bias=({name:float(self.native_row['joints'][name]['bias_force']) for name in RIGHT} if USE_NATIVE_BIAS and USE_NATIVE_FEEDBACK else self.native_bias())
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
    commanded_hand=float(hand_right)
    if LIFT_CONTRACT is not None and stage in ('HAND_CLOSE_HOLD','LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD'):
     now=time.monotonic();probe=float(LIFT_CONTRACT['contact_command'])
     if stage=='HAND_CLOSE_HOLD' and now-start<=1.5:
      u=min(1.,(now-start)/1.5);u=3*u*u-2*u*u*u
      self.grip_command=HAND_OPEN+u*(probe-HAND_OPEN)
     else:
      normal={name:sum(max(0.,float(c['local_force_torque'][0])) for c in self.native_row.get('contacts',[]) if 'test_cylinder' in (c.get('body1'),c.get('body2')) and name in (c.get('body1'),c.get('body2'))) for name in ('R_hand_narrow3_Link','R_hand_wide3_Link')}
      self.grip_command=force_guided_close_step(self.grip_command,float(hand_right),probe,normal,float(LIFT_CONTRACT['required_normal_force_n']),now-self.grip_tick)
     commanded_hand=self.grip_command;self.grip_tick=now
    x=HandCommand();x.name='right_claw_joint';x.position=commanded_hand;x.velocity=1.;x.acceleration=1.;x.deceleration=1.;x.effort=1.;h.right_hands.append(x)
    self.hand_pub.publish(h)
    if stage=='HAND_CLOSE_HOLD':self.close_publish_count+=1
   rclpy.spin_once(self,timeout_sec=.001)
   if self.last_trace_at is None or time.monotonic()-self.last_trace_at>=.05:
    err=max(abs(float(self.arm.get(name,float('nan')))-target[name]) for name in LEFT+RIGHT) if all(name in self.arm for name in LEFT+RIGHT) else float('inf')
    self.last_trace_at=time.monotonic()
    self.trace.append({'time_s':self.last_trace_at-self.run_started,'stage_elapsed_s':self.last_trace_at-start,'host_monotonic_s':self.last_trace_at,'stage':stage,'published_right_hand_command':commanded_hand if hand_right is not None else None,'arm_fresh':self.arm_fresh(),'feedback_source':'NATIVE_MJDATA' if USE_NATIVE_FEEDBACK else 'ROS_ARM_STATE','native_feedback_host_ns':self.native_receipt_ns,'ros_arm_state_rad':dict(self.ros_arm),'arm_source_stamp_ns':dict(self.arm_stamp),'arm_unique_receipt_s':dict(self.arm_receipt),'arm_callback_count':self.arm_callbacks,'arm_regression_discard_count':self.arm_regressions,'published_target_rad':dict(target),'max_arm_error_rad':err,'waist':{n:self.arm.get(n) for n in WAIST},'right':{n:self.arm.get(n) for n in RIGHT},'left':{n:self.arm.get(n) for n in LEFT}})
   # Drain feedback during the remainder of the 50Hz period. Sleeping here
   # starved callbacks from 1000Hz arm + 500Hz IMU/odom subscriptions.
   next_tick+=.02
   self.spin(max(0,min(next_tick,end)-time.monotonic()))
rclpy.init();n=Plan()
try:
 n.spin(.05 if USE_NATIVE_FEEDBACK else 2.0)
 n.refresh_native()
 if not n.arm_fresh():raise RuntimeError('STALE_OR_MISSING_BILATERAL_ARM_STATE')
 if WAIST and n.waist_pub.get_subscription_count()!=1:raise RuntimeError('WAIST_SUBSCRIBER_NOT_UNIQUE')
 if any(name not in n.arm and name not in INITIAL_FALLBACK for name in LEFT+RIGHT):raise RuntimeError('missing complete arm state and dock-state fallback')
 start={n_:float(n.arm[n_]) for n_ in CONTROL};fallback_used=[]; goal={**dict(zip(LEFT,TARGET_LEFT)),**dict(zip(RIGHT,TARGET_RIGHT)),**dict(zip(WAIST,TARGET_WAIST))}
 if max(abs(goal[n_]-start[n_]) for n_ in CONTROL)>1.1:raise RuntimeError('pregrasp target delta exceeds 1.1 rad')
 # The delivered claw component is initialized by the same 0-command
 # handshake used in the previously passing native hand round-trip before a
 # nonzero target is sent.
 if HAND_RIGHT is not None:
  # An already active/open hand must not close again at docking handoff.
  startup_open=USE_NATIVE_FEEDBACK and float(n.native_row['joints']['R_claw_joint']['q'])<-.03
  if not startup_open:n.send(start,.8,'HAND_ACTIVATION',0.0)
  else:print(json.dumps({'hand_activation':'PRESERVED_ALREADY_OPEN_NATIVE_STATE'}),flush=True)
  # Open while the arms are still at the measured dock-safe pose.  Starting
  # the arm approach with closed fingers can lodge a pad against the object
  # before the root position actuator has clearance to move.
  n.send(start,2.0,'HAND_OPEN_PREGRASP',HAND_OPEN)
  n.refresh_native()
  if USE_NATIVE_FEEDBACK:
   actual_open=float(n.native_row['joints']['R_claw_joint']['q'])
   print(json.dumps({'native_hand_open':{'command':HAND_OPEN,'expected_joint_rad':-HAND_OPEN,'actual_joint_rad':actual_open,'error_rad':abs(actual_open+HAND_OPEN)}}),flush=True)
   if abs(actual_open+HAND_OPEN)>.03:raise RuntimeError('HAND_OPEN_NOT_REACHED; approach and closure NOT_RUN')
 n.approach_start=dict(start)
 steps=max(1,int(DURATION/.02))
 for i in range(steps):
  u=(i+1)/steps;u=3*u*u-2*u*u*u;n.approach_fraction=u;n.send({name:start[name]+u*(goal[name]-start[name]) for name in CONTROL},.02,'APPROACH',HAND_OPEN)
 n.send(goal,HOLD,'PREGRASP_HOLD',HAND_OPEN);n.refresh_native()
 if TRACK_NATIVE_CIRCLE:goal.update(n.endpoint)
 reached={n_:float(n.arm[n_]) for n_ in CONTROL}
 preclose_error=max(abs(reached[name]-goal[name]) for name in LEFT+RIGHT)
 preclose_waist_error=max((abs(reached[name]-goal[name]) for name in WAIST),default=0.)
 arrival=native_circle_arrival(CIRCLE_CONTRACT)
 print(json.dumps({'native_tcp_arrival':arrival}),flush=True)
 correction_target=dict(goal); correction_issued=False; correction_error=None; correction_iterations=0
 residual_ff_issued=False
 if ENABLE_RESIDUAL_FF and not TRACK_NATIVE_CIRCLE and CIRCLE_CONTRACT is not None and not arrival['pass'] and preclose_error<=.15:
  ff_target=dict(goal)
  for name in RIGHT:
   candidate=goal[name]+(goal[name]-reached[name])
   limits=SOURCE_LIMITS.get(name)
   if limits is None or not limits[0]+.01<=candidate<=limits[1]-.01:raise RuntimeError('RESIDUAL_FF_SOURCE_LIMIT:'+name)
   ff_target[name]=candidate
   if TRACK_NATIVE_CIRCLE:n.ff_offset[name]=goal[name]-reached[name]
  residual_ff_issued=True
  n.send(ff_target,HOLD,'PREGRASP_RESIDUAL_FEEDFORWARD',HAND_OPEN)
  goal=ff_target
  reached={n_:float(n.arm[n_]) for n_ in CONTROL}
  if TRACK_NATIVE_CIRCLE:goal.update({name:n.endpoint[name]+n.ff_offset[name] for name in RIGHT})
  preclose_error=max(abs(reached[name]-goal[name]) for name in LEFT+RIGHT)
  arrival=native_circle_arrival(CIRCLE_CONTRACT)
  print(json.dumps({'native_tcp_arrival_after_residual_feedforward':arrival,'residual_ff_issued':True}),flush=True)
 # Correct measured task-space error, not all seven joint errors (including
 # unconstrained axial-roll redundancy). The latter demanded an unnecessary
 # shoulder-roll limit crossing in 171134Z. Keep actual arrival fail-closed.
 if TRACK_NATIVE_CIRCLE and ENABLE_RESIDUAL_FF and not arrival['pass'] and preclose_error<=.15:
  for refinement in range(8):
   n.refresh_native();endpoint=n.tracking_target()
   for name in RIGHT:
    correction=float(n.tracking_metrics['cartesian_correction_rad'][name])
    if not math.isfinite(correction):raise RuntimeError('NONFINITE_CARTESIAN_CORRECTION')
    limits=SOURCE_LIMITS[name]
    command=endpoint[name]+n.ff_offset[name]+max(-.03,min(.03,correction))
    command=max(limits[0]+.01,min(limits[1]-.01,command))
    n.ff_offset[name]=command-endpoint[name]
   residual_ff_issued=True
   n.send(goal,.6,'PREGRASP_FEEDBACK_REFINEMENT',HAND_OPEN)
   goal.update({name:n.endpoint[name]+n.ff_offset[name] for name in RIGHT})
   reached={n_:float(n.arm[n_]) for n_ in CONTROL}
   preclose_error=max(abs(reached[name]-goal[name]) for name in LEFT+RIGHT)
   arrival=native_circle_arrival(CIRCLE_CONTRACT)
   print(json.dumps({'native_tcp_arrival_after_refinement':arrival,'refinement':refinement+1}),flush=True)
   if arrival['pass'] and arrival['position_error_m']<=.001 and arrival['axis_error_rad']<=.01:break
 closure_issued=False;lift_execution={'requested':LIFT_CONTRACT is not None,'executed':False}
 # Retain the existing bilateral 0.08 rad execution threshold. Failed arm
 # tracking must never trigger closure just because the trajectory timer ended.
 if HAND_RIGHT is not None and n.arm_fresh() and preclose_error<=.08 and preclose_waist_error<=.05 and arrival["pass"]:
  if TRACK_NATIVE_CIRCLE:
   n.refresh_native();bottle=n.native_row['bodies']['test_cylinder'];rotation=bottle['rotation']
   phase={'phase':'CLOSING','native_arrival_pass':bool(arrival['pass']),'target_world_m':arrival['target_world_m'],'bottle_center_world_m':arrival.get('bottle_center_world_m',bottle['position_m']),'bottle_axis_world':arrival.get('bottle_axis_world',[rotation[2],rotation[5],rotation[8]]),'target_source':arrival.get('target_source'),'generation':n.native_generation,'created_host_ns':time.monotonic_ns()}
   with open('/workspace/evidence/native_tracking_phase.tmp','w') as stream:json.dump(phase,stream)
   os.replace('/workspace/evidence/native_tracking_phase.tmp','/workspace/evidence/native_tracking_phase.json')
  n.closing_started=True
  n.grip_command=HAND_OPEN;n.grip_tick=time.monotonic()
  n.send(goal,HAND_HOLD,'HAND_CLOSE_HOLD',HAND_RIGHT);closure_issued=True
  if LIFT_CONTRACT is not None:
   close_start=min(x['host_monotonic_s'] for x in n.trace if x['stage']=='HAND_CLOSE_HOLD')
   contact=native_bilateral_contact(close_start)
   print(json.dumps({'prelift_bilateral_contact':contact}),flush=True)
   if not contact['pass']:raise RuntimeError('LIFT_GATED_NO_SUSTAINED_BILATERAL_CONTACT')
   n.refresh_native();grasped_position=list(n.native_row['bodies']['test_cylinder']['position_m'])
   # Slip is relative to the actual grasped pose at this stage boundary, not
   # the ungrasped pre-close pose. The closure's independent 10mm gate already
   # passed; retain its motion separately and keep the fixed hand anchor.
   n.lift_phase={**phase,'phase':'LIFTING','bilateral_contact_pass':True,'created_host_ns':time.monotonic_ns(),'lift_offset_m':0.,'bottle_center_world_m':grasped_position,'preclose_bottle_center_world_m':phase['bottle_center_world_m']}
   hand=n.native_row['bodies']['R_omnipicker_base_link'];H=hand['rotation'];B=n.native_row['bodies']['test_cylinder']['rotation'];height=CIRCLE_CONTRACT['height_above_center_m'];delta=[grasped_position[i]+height*B[3*i+2]-hand['position_m'][i] for i in range(3)]
   n.lift_phase['initial_object_in_hand_m']=[sum(H[3*i+j]*delta[i] for i in range(3)) for j in range(3)]
   n.lift_phase['slip_reference']='GRASP_SECTION'
   n.lift_contact_lost_since=None;n.write_phase(n.lift_phase)
   lift_execution={'requested':True,'executed':True,'initial_bottle_z_m':n.native_row['bodies']['test_cylinder']['position_m'][2],'lift_start_host_ns':time.monotonic_ns()}
   print(json.dumps({'lift_execution':lift_execution}),flush=True)
   n.send(goal,1.,'LIFT_PROBE',HAND_RIGHT);n.lift_phase['lift_offset_m']=.008;n.write_phase(n.lift_phase)
   # Complete the physical probe through feedback rather than assuming the
   # inner servo reached 8mm when the one-second ramp timer elapsed.
   probe_deadline=time.monotonic()+2.
   while time.monotonic()<probe_deadline:
    n.send(goal,.04,'LIFT_PROBE_HOLD',HAND_RIGHT);n.refresh_native()
    probe_rise=n.native_row['bodies']['test_cylinder']['position_m'][2]-lift_execution['initial_bottle_z_m']
    table_force=sum(max(0.,float(c['local_force_torque'][0])) for c in n.native_row.get('contacts',[]) if 'test_cylinder' in (c.get('body1'),c.get('body2')) and 'raicom_table_table' in (c.get('body1'),c.get('body2')))
    if probe_rise>=.005 and table_force<=.1:break
   n.refresh_native()
   probe_rise=n.native_row['bodies']['test_cylinder']['position_m'][2]-lift_execution['initial_bottle_z_m']
   table_force=sum(max(0.,float(c['local_force_torque'][0])) for c in n.native_row.get('contacts',[]) if 'test_cylinder' in (c.get('body1'),c.get('body2')) and 'raicom_table_table' in (c.get('body1'),c.get('body2')))
   print(json.dumps({'lift_probe':{'rise_m':probe_rise,'table_force_n':table_force}}),flush=True)
   if probe_rise<.005 or table_force>.1:raise RuntimeError('LOAD_TRANSFER_PROBE_FAILED')
   n.send(goal,3.,'LIFT',HAND_RIGHT)
   n.lift_phase['lift_offset_m']=.06;n.write_phase(n.lift_phase)
   lift_execution['hold_start_ns']=time.monotonic_ns()
   n.send(goal,1.5,'LIFT_HOLD',HAND_RIGHT)
   lift_execution['hold_end_ns']=time.monotonic_ns()
   print(json.dumps({'lift_execution':lift_execution}),flush=True)
 if RESTORE:
  for i in range(steps):
   u=(i+1)/steps;u=3*u*u-2*u*u*u;n.send({name:goal[name]+u*(start[name]-goal[name]) for name in CONTROL},.02,'RESTORE',HAND_OPEN)
  # Keep the source-gain controller at the restored pose long enough for the
  # existing 0.04-rad recovery gate to measure a settled state.
  n.send(start,1.0,'RESTORE_HOLD',HAND_OPEN);restored={n_:float(n.arm[n_]) for n_ in CONTROL};max_restore=max(abs(restored[n_]-start[n_]) for n_ in CONTROL)
 else:
  if not lift_execution['executed']:n.send(goal,1.0,'PREGRASP_STABLE_HOLD',HAND_RIGHT if closure_issued else HAND_OPEN)
  restored={n_:float(n.arm[n_]) for n_ in CONTROL};max_restore=None
 max_goal=max(abs(reached[n_]-goal[n_]) for n_ in CONTROL)
 drift=math.hypot(n.odom[-1][0]-n.odom[0][0],n.odom[-1][1]-n.odom[0][1]) if len(n.odom)>=2 else None; stable=len(n.odom)>=100 and drift is not None and drift<=.05 and n.tilt and max(n.tilt)<=math.radians(15)
 # No installed native-MjData/HandState mapping is calibrated for this overlay.
 # Do not promote a normalized zero command, absent feedback, or q~=0 into a
 # physical-grasp acknowledgement. Preserve observations pending calibration.
 close_rows=[x for x in n.trace if x['stage']=='HAND_CLOSE_HOLD']
 contact_result=native_bilateral_contact(min((x['host_monotonic_s'] for x in close_rows),default=float('inf'))) if HAND_RIGHT is not None else {'pass':True,'classification':'NO_CLOSE_REQUESTED'}
 hand_pass=HAND_RIGHT is None or contact_result['pass']
 print(json.dumps({'native_tcp_arrival':arrival,'correction_issued':correction_issued,'correction_error_rad':correction_error,'correction_iterations':correction_iterations,'residual_ff_issued':residual_ff_issued,'preclose_waist_error_rad':preclose_waist_error,'preclose_arm_error_rad':preclose_error,'closure_issued':closure_issued,'arm_feedback_fresh':n.arm_fresh(),'arm_callback_count':n.arm_callbacks,'arm_late_sample_discard_count':n.arm_regressions,'hand_mapping_status':'COMMAND_SENT; native qpos/actuator mapping not yet calibrated','physical_bilateral_contact':contact_result}),flush=True)
 print(json.dumps({'plan_mode':'ROUND_TRIP' if RESTORE else ('PREGRASP_CLOSE_HOLD' if HAND_RIGHT is not None else 'PREGRASP_HOLD'),'subscription_count':n.pub.get_subscription_count(),'hand_subscription_count':n.hand_pub.get_subscription_count(),'start_arm_rad':start,'state_fallback_joint_names':fallback_used,'goal_arm_rad':goal,'reached_arm_rad':reached,'restored_arm_rad':restored,'right_hand_command':HAND_RIGHT,'physical_bilateral_contact':contact_result,'final_hand_state_rad':n.hand,'hand_response_pass':hand_pass,'max_goal_error_rad':max_goal,'max_restore_error_rad':max_restore,'response_pass':max_goal<=.08 and preclose_waist_error<=.05 and hand_pass and ((not RESTORE) or max_restore<=.04),'motion_stability':{'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None,'pass':stable},'trace':n.trace}))
finally:
 print(json.dumps({'execution_exception_context':True,'closure_started':n.closing_started,'closure_issued':n.close_publish_count>0,'close_publish_call_count':n.close_publish_count,'publication_is_not_mc_acknowledgement':True}),flush=True)
 with open('/workspace/evidence/native_hal_pregrasp_trace.json','w') as stream:json.dump({'trace':n.trace,'odom_trace':n.odom_trace,'last_controlled_state_rad':n.arm,'last_unique_source_stamps':n.arm_stamp},stream)
 n.destroy_node();rclpy.shutdown()
""".replace("__LIFT_CONTRACT__", repr(lift_contract)).replace("__LEFT_TARGET__", json.dumps(left_target)).replace("__RIGHT_TARGET__", json.dumps(right_target)).replace("__GAINS__", serialized_gains).replace("__DURATION__", f"{duration_s:.17g}").replace("__HOLD__", f"{hold_s:.17g}").replace("__RESTORE__", "True" if restore else "False").replace("__HAND_RIGHT__", "None" if right_hand_command is None else f"{right_hand_command:.17g}").replace("__HAND_HOLD__", f"{hand_hold_s:.17g}").replace("__INITIAL_FALLBACK__", json.dumps(initial_arm_state or {}, sort_keys=True, allow_nan=False)).replace("__WAIST_NAMES__", repr(waist_names if waist_target is not None else ())).replace("__WAIST_TARGET__", repr(waist_target or [])).replace("__HAND_OPEN__", repr(hand_open_command)).replace("__CIRCLE_CONTRACT__", repr(circle_contract)).replace("__TRACK_NATIVE_CIRCLE__", repr(track_native_circle)).replace("__USE_NATIVE_BIAS__", repr(use_native_bias_effort)).replace("__USE_NATIVE_FEEDBACK__", repr(use_native_joint_feedback)).replace("__SOURCE_LIMITS__", repr(source_joint_limits or {})).replace("__ENABLE_CORRECTION__", repr(enable_measured_correction)).replace("__ENABLE_RESIDUAL_FF__", repr(enable_single_residual_feedforward))
    if lift_contract and lift_contract.get('retract_preflight'):
        return _with_postlift_retract(script, lift_contract)
    return script


def _with_postlift_retract(script, lift_contract):
    """Extend only an explicitly screened lift; the default script is unchanged."""
    if not lift_contract or not lift_contract.get("retract_preflight"):
        return script
    import inspect
    from scripts.research.issue46.native_runtime.native_grasp_audit import evaluate_unsupported_lift
    from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose,physical_fault
    retract=lift_contract["retract_preflight"];v=retract.get("offset_world_m",[])
    if retract.get("pass_result") is not True or len(v)!=3 or any(not math.isfinite(float(x)) for x in v) or math.sqrt(sum(float(x)**2 for x in v))>.25 or not -.02<=v[2]<=.05 or (retract.get("duration_s"),retract.get("hold_s"))!=(4.,1.5):
        raise ValueError("retract preflight or bounded profile invalid")
    limits=retract.get('source_motion_limits',{})
    if len(limits)!=29 or any(len(v)!=3 or not all(math.isfinite(float(x)) for x in v) or not v[0]<v[1] or v[2]<=0 for v in limits.values()):raise ValueError('retract needs all29 pinned source motion limits')
    def replace(old,new,count=1):
        nonlocal script
        if script.count(old)!=count:raise ValueError("retract script extension anchor changed:"+old[:50])
        script=script.replace(old,new)
    replace("class Plan(Node):",inspect.getsource(evaluate_unsupported_lift)+"\n"+inspect.getsource(native_pose)+"\n"+inspect.getsource(physical_fault)+"\nclass Plan(Node):")
    replace("self.closing_started=False;self.close_publish_count=0","self.closing_started=False;self.close_publish_count=0;self.native_lift_rows=[]")
    replace("  values={}\n","  fault=physical_fault(row,native_pose(row),LIFT_CONTRACT['retract_preflight']['source_motion_limits'])\n  if fault:raise RuntimeError('POST_LIFT_SOURCE_SAFETY:'+fault)\n  values={}\n")
    replace("  self.arm.update(values);self.native_row=row;","  if not self.native_lift_rows or stamp>self.native_lift_rows[-1]['host_monotonic_ns']:self.native_lift_rows=(self.native_lift_rows+[row])[-140:]\n  self.arm.update(values);self.native_row=row;")
    replace("   if stage in ('LIFT_PROBE','LIFT'):\n",
"""   if stage=='RETRACT':
    u=min(1.,(time.monotonic()-start)/seconds);smooth=3*u*u-2*u*u*u
    self.lift_phase['retract_offset_world_m']=[smooth*float(x) for x in LIFT_CONTRACT['retract_preflight']['offset_world_m']]
    self.write_phase(self.lift_phase)
   if stage in ('LIFT_PROBE','LIFT'):
""")
    replace("('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD')","('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD')",count=2)
    replace("('HAND_CLOSE_HOLD','LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD')","('HAND_CLOSE_HOLD','LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD')")
    replace("   print(json.dumps({'lift_execution':lift_execution}),flush=True)\n if RESTORE:",
"""   print(json.dumps({'lift_execution':lift_execution}),flush=True)
   accepted=evaluate_unsupported_lift(n.native_lift_rows,initial_bottle_z_m=lift_execution['initial_bottle_z_m'],hold_start_ns=lift_execution['hold_start_ns'],hold_end_ns=lift_execution['hold_end_ns'])
   print(json.dumps({'pre_retract_physical_lift':accepted}),flush=True)
   if not accepted['pass']:raise RuntimeError('RETRACT_GATED_UNQUALIFIED_PHYSICAL_LIFT')
   retract_execution={'requested':True,'executed':True,'start_host_ns':time.monotonic_ns()}
   n.lift_phase['phase']='RETRACTING';n.lift_phase['created_host_ns']=time.monotonic_ns();n.lift_phase['retract_from_hold_end_ns']=lift_execution['hold_end_ns'];n.lift_phase['pre_retract_physical_lift_pass']=True;n.lift_phase['retract_offset_world_m']=[0.,0.,0.];n.write_phase(n.lift_phase)
   n.send(goal,4.,'RETRACT',HAND_RIGHT)
   n.lift_phase['retract_offset_world_m']=LIFT_CONTRACT['retract_preflight']['offset_world_m'];n.write_phase(n.lift_phase)
   retract_execution['hold_start_ns']=time.monotonic_ns()
   n.send(goal,1.5,'RETRACT_HOLD',HAND_RIGHT)
   retract_execution['hold_end_ns']=time.monotonic_ns()
   accepted=evaluate_unsupported_lift(n.native_lift_rows,initial_bottle_z_m=lift_execution['initial_bottle_z_m'],hold_start_ns=retract_execution['hold_start_ns'],hold_end_ns=retract_execution['hold_end_ns'])
   retract_execution['physical_hold']=accepted;retract_execution['pass']=accepted['pass']
   print(json.dumps({'post_lift_retract':retract_execution}),flush=True)
   if not accepted['pass']:raise RuntimeError('RETRACT_PHYSICAL_HOLD_UNQUALIFIED')
 if RESTORE:""")
    if retract.get("carry_lease_s") is not None:script=_with_carry_lease(script,retract)
    compile(script,"post_lift_retract_generated","exec")
    return script



def carry_abort_last_command_hold(n, carry_error):
 heartbeat={'mode':'CARRY_ABORT_LAST_COMMAND_HOLD','healthy':False,'failure':str(carry_error),'pid':os.getpid(),'host_monotonic_ns':time.monotonic_ns()}
 with open('/workspace/evidence/carry_heartbeat.tmp','w') as f:json.dump(heartbeat,f)
 os.replace('/workspace/evidence/carry_heartbeat.tmp','/workspace/evidence/carry_heartbeat.json')
 print(json.dumps({'carry_abort_hold':heartbeat,'maximum_hold_s':5.}),flush=True)
 abort_until=time.monotonic()+5.
 while rclpy.ok() and time.monotonic()<abort_until and not os.path.exists('/workspace/evidence/carry_release'):
  try:
   with open('/workspace/evidence/native_latest.json') as f:sample=json.load(f)
   if sample['generation']!=n.native_generation or not 0.<=(time.monotonic_ns()-sample['host_monotonic_ns'])*1e-9<=.20:break
   if not hasattr(n,'last_arm_message') or not hasattr(n,'last_hand_message'):break
   n.last_arm_message.header.stamp=n.get_clock().now().to_msg();n.last_arm_message.header.sequence=n.seq;n.seq+=1
   n.pub.publish(n.last_arm_message);n.hand_pub.publish(n.last_hand_message)
   n.spin(.02)
  except Exception:break

def _with_carry_lease(script, retract):
    """Keep the proven HAL grip/arm owner alive; it never publishes velocity."""
    lease=float(retract["carry_lease_s"])
    if not math.isfinite(lease) or not 5.<=lease<=200.:raise ValueError("carry lease outside bounded 5..200s")
    def replace(old,new,count=1):
        nonlocal script
        if script.count(old)!=count:raise ValueError("carry extension anchor changed:"+old[:50])
        script=script.replace(old,new)
    import inspect
    replace("class Plan(Node):",inspect.getsource(carry_abort_last_command_hold)+"\nclass Plan(Node):")
    replace("   if len(self.odom)>1 and math.hypot(","   if stage!='CARRY_HOLD' and len(self.odom)>1 and math.hypot(")
    replace("stage not in ('HAND_ACTIVATION','HAND_OPEN_PREGRASP')","stage not in ('HAND_ACTIVATION','HAND_OPEN_PREGRASP','CARRY_ABORT_HOLD')")
    replace("('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD')","('LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD','CARRY_HOLD')",count=2)
    replace("('HAND_CLOSE_HOLD','LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD')","('HAND_CLOSE_HOLD','LIFT_PROBE','LIFT_PROBE_HOLD','LIFT','LIFT_HOLD','RETRACT','RETRACT_HOLD','CARRY_HOLD')")
    replace("  self.tracking_metrics=r.get('metrics',{})","  self.tracking_metrics=r.get('metrics',{});self.tracking_velocity=r.get('right_velocity_rad_s',{})")
    replace("x.velocity=0.;x.effort=float(bias.get(name,0.))","x.velocity=float(self.tracking_velocity.get(name,0.)) if stage=='CARRY_HOLD' and name in RIGHT else 0.;x.effort=float(bias.get(name,0.))")
    replace("   self.pub.publish(m)\n","   self.pub.publish(m);self.last_arm_message=m;self.last_published_arm_target={name:float(target[name]) for name in LEFT+RIGHT}\n")
    replace("    self.hand_pub.publish(h)\n    if stage==\'HAND_CLOSE_HOLD\':self.close_publish_count+=1\n",
"""    self.hand_pub.publish(h);self.last_hand_message=h
    if stage==\'HAND_CLOSE_HOLD\':self.close_publish_count+=1
   if stage=='CARRY_HOLD':
    for c in self.native_row.get('contacts',[]):
     pair={c.get('body1'),c.get('body2')}
     if 'test_cylinder' in pair and not pair.intersection({'R_hand_narrow3_Link','R_hand_wide3_Link'}) and float(c['local_force_torque'][0])>.1:raise RuntimeError('CARRY_OBJECT_EXTERNAL_SUPPORT')
    hand=self.native_row['bodies']['R_omnipicker_base_link'];obj=self.native_row['bodies']['test_cylinder'];H=hand['rotation'];B=obj['rotation']
    delta=[obj['position_m'][i]+CIRCLE_CONTRACT['height_above_center_m']*B[3*i+2]-hand['position_m'][i] for i in range(3)]
    local=[sum(H[3*i+j]*delta[i] for i in range(3)) for j in range(3)]
    slip=math.sqrt(sum((a-b)**2 for a,b in zip(local,self.lift_phase['initial_object_in_hand_m'])))
    if slip>.01 or B[8]<math.cos(.10) or obj['position_m'][2]<.25:raise RuntimeError('CARRY_OBJECT_POSE_OR_SLIP:'+json.dumps({'slip_m':slip,'vertical_axis_dot':B[8],'bottle_z_m':obj['position_m'][2],'source_host_ns':self.native_receipt_ns}))
    heartbeat={'mode':'MEASURED_LOAD_BEARING_CARRY_HOLD','healthy':True,'pid':os.getpid(),'node_name':self.get_name(),'host_monotonic_ns':time.monotonic_ns(),'native_source_host_ns':self.native_receipt_ns,'generation':self.native_generation,'object_in_hand_error_m':slip,'bottle_world_xyz_m':obj['position_m'],'publication_is_not_mc_acknowledgement':True}
    with open('/workspace/evidence/carry_heartbeat.tmp','w') as f:json.dump(heartbeat,f)
    os.replace('/workspace/evidence/carry_heartbeat.tmp','/workspace/evidence/carry_heartbeat.json')
""")
    needle="   if not accepted['pass']:raise RuntimeError('RETRACT_PHYSICAL_HOLD_UNQUALIFIED')\n if RESTORE:"
    replace(needle,
"""   if not accepted['pass']:raise RuntimeError('RETRACT_PHYSICAL_HOLD_UNQUALIFIED')
   carry_target=dict(n.last_published_arm_target)
   hand=n.native_row['bodies']['R_omnipicker_base_link'];base=n.native_row['bodies']['pelvis'];R=hand['rotation'];c=CIRCLE_CONTRACT['circle_center_local_m']
   tcp=[hand['position_m'][i]+sum(R[3*i+j]*c[j] for j in range(3)) for i in range(3)]
   yaw=math.atan2(base['rotation'][3],base['rotation'][0]);dx=tcp[0]-base['position_m'][0];dy=tcp[1]-base['position_m'][1]
   n.lift_phase.update(phase='CARRY_HOLD',created_host_ns=time.monotonic_ns(),post_retract_physical_hold_pass=True,carry_lease_s=LIFT_CONTRACT['retract_preflight']['carry_lease_s'],carry_heading_xy_m=[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy],carry_world_z_m=tcp[2],carry_source_speed_rad_s=[LIFT_CONTRACT['retract_preflight']['source_motion_limits'][name][2] for name in RIGHT])
   n.write_phase(n.lift_phase)
   try:
    n.send(carry_target,.05,'CARRY_HOLD',HAND_RIGHT)
    n.pre_carry_odom=list(n.odom);n.pre_carry_tilt=list(n.tilt)
    ready={'schema':'robotsim.x2.carry-ready.v1','pid':os.getpid(),'node_name':n.get_name(),'generation':n.native_generation,'host_monotonic_ns':time.monotonic_ns(),'physical_lift_pass':True,'post_lift_retract_pass':True,'arm_target_rad':carry_target,'lease_s':LIFT_CONTRACT['retract_preflight']['carry_lease_s'],'start_base_world_xyz_m':n.native_row['bodies']['pelvis']['position_m'],'start_bottle_world_xyz_m':n.native_row['bodies']['test_cylinder']['position_m']}
    with open('/workspace/evidence/carry_ready.tmp','w') as f:json.dump(ready,f)
    os.replace('/workspace/evidence/carry_ready.tmp','/workspace/evidence/carry_ready.json')
    carry_until=time.monotonic()+float(ready['lease_s'])
    while not os.path.exists('/workspace/evidence/carry_release') and time.monotonic()<carry_until:
     n.send(carry_target,.05,'CARRY_HOLD',HAND_RIGHT)
    if not os.path.exists('/workspace/evidence/carry_release'):raise RuntimeError('CARRY_LEASE_EXPIRED')
    n.send(carry_target,1.5,'CARRY_HOLD',HAND_RIGHT)
    print(json.dumps({'carry_execution':{'pass':True,'classification':'GRIP_HOLD_AND_RELEASE_NOT_ROUTE_ACCEPTANCE','released_by_parent':True,'generation':n.native_generation,'final_base_world_xyz_m':n.native_row['bodies']['pelvis']['position_m'],'final_bottle_world_xyz_m':n.native_row['bodies']['test_cylinder']['position_m'],'measured_base_displacement_m':math.dist(n.native_row['bodies']['pelvis']['position_m'],ready['start_base_world_xyz_m']),'measured_bottle_displacement_m':math.dist(n.native_row['bodies']['test_cylinder']['position_m'],ready['start_bottle_world_xyz_m'])}}),flush=True)
   except Exception as carry_error:
    carry_abort_last_command_hold(n,carry_error)
    raise
 if RESTORE:""")
    final_line=next(line for line in script.splitlines() if line.startswith(" drift=math.hypot("))
    replace(final_line," standing_odom=getattr(n,'pre_carry_odom',n.odom);standing_tilt=getattr(n,'pre_carry_tilt',n.tilt)\n"+final_line.replace("n.odom","standing_odom").replace("n.tilt","standing_tilt"))
    replace("'motion_stability':{'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None,'pass':stable}","'motion_stability':{'scope':'STANDING_BEFORE_CARRY','sample_count':len(standing_odom),'planar_drift_m':drift,'max_tilt_rad':max(standing_tilt) if standing_tilt else None,'pass':stable}")
    compile(script,"carry_lease_generated","exec")
    return script


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
        self.samples={'arm':0,'leg':0,'waist':0}; self.commands={'arm':{},'leg':{},'waist':{}}; self.stamps={}; self.received={}
        self.create_subscription(JointCommandArray,'/aima/hal/joint/arm/command',lambda msg:self.store('arm',msg),qos)
        self.create_subscription(JointCommandArray,'/aima/hal/joint/leg/command',lambda msg:self.store('leg',msg),qos)
        self.create_subscription(JointCommandArray,'/aima/hal/joint/waist/command',lambda msg:self.store('waist',msg),qos)
    def store(self,area,msg):
        stamp=int(msg.header.stamp.sec)*1000000000+int(msg.header.stamp.nanosec)
        if stamp<=self.stamps.get(area,0):return
        self.stamps[area]=stamp;self.received[area]=time.monotonic()
        self.samples[area]+=1
        self.commands[area]={item.name:{'position':float(item.position),'velocity':float(item.velocity),'effort':float(item.effort),'stiffness':float(item.stiffness),'damping':float(item.damping)} for item in msg.joints}
rclpy.init(); node=Capture()
try:
    began=time.monotonic();ready=None;begin_counts=None;begin_stamps=None
    expected={'leg':LEG_ORDER,'waist':WAIST_ORDER}
    while rclpy.ok() and time.monotonic()-began<4.:
        rclpy.spin_once(node,timeout_sec=.002);now=time.monotonic()
        complete=all(all(name in node.commands[area] for name in names) and now-node.received.get(area,0.)<=.20 for area,names in expected.items())
        if ready is None:
            if complete:ready=now;begin_counts=dict(node.samples);begin_stamps=dict(node.stamps)
            elif now-began>=2.:raise RuntimeError('command discovery/readiness unconfirmed within 2s')
        else:
            if not complete:raise RuntimeError('captured command stream became stale or incomplete')
            if now-ready>=2.:break
    if ready is None or time.monotonic()-ready<2.:raise RuntimeError('complete post-readiness capture window missing')
    window_counts={area:node.samples[area]-begin_counts[area] for area in expected}
    window_source_span={area:(node.stamps[area]-begin_stamps[area])*1e-9 for area in expected}
    for area,names in expected.items():
        if window_counts[area] < 10 or window_source_span[area]<1.90 or any(name not in node.commands[area] for name in names): raise RuntimeError('missing complete '+area+' command stream')
        for name in names:
            if not all(math.isfinite(node.commands[area][name][field]) for field in ('position','velocity','effort','stiffness','damping')): raise RuntimeError('non-finite '+area+' command')
    arm_complete=all(name in node.commands['arm'] for name in ARM_ORDER)
    print(json.dumps({'sample_counts':node.samples,'window_counts':window_counts,'source_span_s':window_source_span,'readiness_delay_s':ready-began,'capture_duration_s':time.monotonic()-ready,'arm_command_stream_complete':arm_complete,'arm_commands':{name:node.commands['arm'][name] for name in ARM_ORDER} if arm_complete else {},'leg_commands':{name:node.commands['leg'][name] for name in LEG_ORDER},'waist_commands':{name:node.commands['waist'][name] for name in WAIST_ORDER}}))
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


def reviewed_foundationwalk_profile(runtime):
    """Select the already reviewed Issue80 overlay; never tune or edit a source."""
    import hashlib,json,copy,yaml
    from pathlib import Path
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    base=Path("/home/lzy18001500226/projects/RobotSim/reviews/issue-80/foundationwalk-final-left-right-turns-run65-20261022T073901Z")
    manifest=base/"foundationwalk_config_overlay_manifest.json"
    if sha(manifest)!="ae6e6fe56905efddf9dd5796d805f01bec6b15f5831549c14fadd4302789b1c1":
        raise ValueError("reviewed FoundationWalk manifest differs")
    record=json.loads(manifest.read_text())
    original=Path(runtime)/"mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl/foundationwalk_config.yaml"
    overlay=base/"foundationwalk_knee019_minrun010.yaml"
    if sha(original)!=record["source_sha256"] or sha(overlay)!=record["overlay_sha256"]:
        raise ValueError("frozen FoundationWalk source/overlay hash mismatch")
    source=yaml.safe_load(original.read_text());selected=yaml.safe_load(overlay.read_text())
    expected=copy.deepcopy(source)
    for i,name in ((9,"left_knee_joint"),(10,"right_knee_joint")):
        if source["BaseConfig"]["action_seq"][i]!=name or source["BaseConfig"]["action_scale"][i]!=.25:
            raise ValueError("FoundationWalk source action mapping mismatch")
        expected["BaseConfig"]["action_scale"][i]=.19
    if source["FoundationWalkConfig"]["foundationwalk"]["min_run_velx"]!=.5:
        raise ValueError("FoundationWalk source floor mismatch")
    expected["FoundationWalkConfig"]["foundationwalk"]["min_run_velx"]=.1
    if selected!=expected:raise ValueError("unexpected reviewed FoundationWalk delta")
    paths=dict(
        foundationwalk_onnx="mc/bin/mc_param/src/robot/lx2501_3_t2d5/rl_models/foundationwalk_v1.onnx",
        foundationwalk_runner="mc/lib/libmc_runner_rl_foundationwalk.so",
        robot_model="mc/bin/mc_param/src/robot/lx2501_3_t2d5/robot_model.yaml")
    verified={}
    for key,rel in paths.items():
        actual=sha(Path(runtime)/rel)
        if actual!=record["verified_runtime_input_sha256"][key]:
            raise ValueError("reviewed FoundationWalk runtime identity differs:"+key)
        verified[key]=actual
    return overlay,dict(classification="SIMULATION_ONLY_REUSED_ISSUE80_DIAGNOSTIC_NOT_VENDOR_DEFAULT",
        reviewed_manifest=str(manifest),reviewed_manifest_sha256=sha(manifest),
        source_path=str(original),source_sha256=sha(original),overlay_sha256=sha(overlay),
        changed_values=record["changed_values"],verified_unchanged_runtime=verified,
        fresh_parameter_tuning=False,mc_binary_onnx_gains_and_joint_limits_unchanged=True,
        historical_success_is_not_current_robot_qualification=True)
