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
import math
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import time

import yaml


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
RUNTIME = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
IMAGE = "robotsim:humble-raicom-runtime-20261019"
MC_DUMP = "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml.dump"
MC_GENERATED = "mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim"
OFFICIAL_CMD_VEL_BRIDGE = EVIDENCE_REPO / "reference/AgiBot-Community/raicom2026-agibot-X2/src/race_task/race_task/cmd_vel_bridge.py"
VR_DATA_PROBE_SOURCE = EVIDENCE_REPO / "reference/Limao0521/SENIA-SABANAHACK-2026/robot/official_sim/vr_data_probe.c"
VR_DATA_PROBE_SOURCE_SHA256 = "62eb495ddac4359c4871daa8164af897e827227f8d35993baa426a5e75771d03"
VR_DATA_MC_GENERATOR_SHA256 = "c9d31ff619d47bb7f5d49e022579a4d5ae7b3fe5324e906162384e0586c83237"
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


def candidate_scene_matches_overlay(candidate, table_overlay, control_overlay, control_path=None):
    """Accept only the original world or its exact locked pad-only derivative."""
    world=table_overlay.get('generated',{}).get('host_scene',{})
    scene=candidate.get('scene',{})
    if scene.get('sha256') and scene.get('sha256')==world.get('sha256'):return True
    provenance=candidate.get('source_pad_collision_comparison',{})
    derivative=(control_overlay or {}).get('source_pad_collision_comparison',{})
    if not provenance or derivative.get('manifest_sha256')!='7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a' or set(derivative.get('scope',[]))!={'R_hand_narrow3_Link','R_hand_wide3_Link'}:return False
    if control_path is None or provenance.get('overlay')!=str(control_path) or provenance.get('overlay_sha256')!=sha256(control_path):return False
    if scene!=(control_overlay or {}).get('generated',{}).get('host_scene'):return False
    require_hashed(Path(scene.get('path','')),scene.get('sha256',''),label='exact selected pad proxy scene')
    source_path=require_hashed(Path(provenance.get('source_candidate','')),provenance.get('source_candidate_sha256',''),label='original table candidate')
    original=json.loads(source_path.read_text())
    unchanged={k:v for k,v in candidate.items() if k not in ('scene','source_pad_collision_comparison')}
    expected={k:v for k,v in original.items() if k!='scene'}
    return original.get('scene',{}).get('sha256')==world.get('sha256') and unchanged==expected


def quaternion_yaw_wxyz(values: list[float]) -> float:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("expected a finite wxyz quaternion")
    w, x, y, z = values
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrapped_angle_error(actual: float, expected: float) -> float:
    return math.atan2(math.sin(actual - expected), math.cos(actual - expected))


def prearm_script() -> str:
    return """#!/usr/bin/env python3
import json
import time
import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import CommonRequest, McActionCommand, RequestHeader
from aimdk_msgs.srv import GetMcAction, SetMcAction

def call(node, client, request, attempts=8):
    if not client.wait_for_service(timeout_sec=5.0): raise RuntimeError('service unavailable')
    # The delivered MC endpoint can advertise before its executor completes
    # post-start initialization.  Use bounded short retries, matching the
    # existing stand client instead of treating a first missed readback as a
    # robot-control failure.
    for _ in range(attempts):
        future=client.call_async(request); rclpy.spin_until_future_complete(node,future,timeout_sec=.25)
        if future.done() and future.result() is not None: return future.result()
        time.sleep(.05)
    raise RuntimeError('service call timed out after 8 bounded retries')
rclpy.init(); node=Node('robotsim_xvfb_native_prearm')
try:
    # A registered RPC endpoint preceded the installed ActionManager's initial
    # PASSIVE_DEFAULT transition in the 175044Z trace. Observe initialized mode
    # before sending the one SD request; never retry a rejected action.
    get_client=node.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction')
    readiness=[]
    for _ in range(10):
        r=GetMcAction.Request();r.request=CommonRequest();r.request.header.stamp=node.get_clock().now().to_msg()
        info=call(node,get_client,r).info
        readiness.append({'mode_name':info.action_desc,'mode_status':int(info.status.value)})
        if info.action_desc in ('PASSIVE_DEFAULT','STAND_DEFAULT') and int(info.status.value)!=0:break
        time.sleep(.3)
    else:raise RuntimeError('MC action manager not initialized; no SD request sent:'+json.dumps(readiness))
    set_client=node.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction')
    request=SetMcAction.Request(); request.header=RequestHeader(); request.header.stamp=node.get_clock().now().to_msg(); request.source='node'; request.command=McActionCommand(); request.command.action_desc='STAND_DEFAULT'
    set_result=call(node,set_client,request,attempts=1)
    get_client=node.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction'); samples=[]
    for _ in range(10):
        get_request=GetMcAction.Request(); get_request.request=CommonRequest(); get_request.request.header.stamp=node.get_clock().now().to_msg(); get_result=call(node,get_client,get_request)
        sample={'mode_name':get_result.info.action_desc,'mode_status':int(get_result.info.status.value)}; samples.append(sample)
        if sample['mode_name']=='STAND_DEFAULT': break
        time.sleep(0.3)
    print(json.dumps({'initialization_readbacks':readiness,'set_header_code':int(set_result.response.header.code),'set_status':int(set_result.response.status.value),'set_message':set_result.response.message,'mode_name':samples[-1]['mode_name'],'mode_status':samples[-1]['mode_status'],'mode_samples':samples}))
finally:
    node.destroy_node(); rclpy.shutdown()
"""


def preset_motion_probe_script(motion_id: int, area_id: int) -> str:
    """Build a bounded probe for the delivered animation-player service.

    This uses a packaged preset motion first.  It deliberately does not
    provide a custom animation path, touch the gripper, or publish directly to
    simulator joints.
    """
    return f"""#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import CommonState, JointStateArray, McControlArea, McPresetMotion, RequestHeader
from aimdk_msgs.srv import SetMcPresetMotion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

QOS=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
RIGHT=('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
def yaw(q): return math.atan2(2.0*(q.w*q.z+q.x*q.y),1.0-2.0*(q.y*q.y+q.z*q.z))
def tilt(q):
 p=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x))))
 r=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y))
 return max(abs(p),abs(r))
class Probe(Node):
 def __init__(self):
  super().__init__('robotsim_preset_motion_probe')
  self.client=self.create_client(SetMcPresetMotion,'/aimdk_5Fmsgs/srv/SetMcPresetMotion')
  self.arm={{}};self.initial=None;self.final=None;self.max_arm_delta=0.0;self.start_odom=None;self.odom=None;self.max_tilt=0.0
  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,QOS)
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,QOS)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,QOS)
 def on_arm(self,m):
  self.arm.update({{x.name:float(x.position) for x in m.joints}})
  if self.initial is not None and all(name in self.arm for name in RIGHT):self.max_arm_delta=max(self.max_arm_delta,max(abs(self.arm[name]-self.initial[name]) for name in RIGHT))
 def on_odom(self,m):
  p=m.pose.pose.position;self.odom=(float(p.x),float(p.y),float(p.z));self.start_odom=self.start_odom or self.odom
 def on_imu(self,m): self.max_tilt=max(self.max_tilt,tilt(m.orientation))
 def spin(self,duration):
  end=time.monotonic()+duration
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.02)
rclpy.init();n=Probe()
try:
 n.spin(1.0)
 if not n.client.wait_for_service(timeout_sec=5.0):raise RuntimeError('SetMcPresetMotion unavailable')
 if any(name not in n.arm for name in RIGHT) or n.start_odom is None:raise RuntimeError('incomplete native state before preset probe')
 n.initial={{name:n.arm[name] for name in RIGHT}}
 # Match the delivered AgiBot preset_motion_client.py: this ROS service may
 # not answer the first request while its remote peer is waking up.
 response=None
 for _ in range(8):
  req=SetMcPresetMotion.Request();req.header=RequestHeader();req.header.stamp=n.get_clock().now().to_msg();req.area.value={area_id};req.motion.value={motion_id};req.interrupt=False;req.ani_path='';req.play_timestamp=0
  future=n.client.call_async(req);rclpy.spin_until_future_complete(n,future,timeout_sec=.25)
  if future.done() and future.result() is not None:
   response=future.result().response;break
 if response is None:raise RuntimeError('SetMcPresetMotion timed out after 8 official-style retries')
 n.spin(4.5)
 if any(name not in n.arm for name in RIGHT) or n.odom is None:raise RuntimeError('incomplete native state after preset probe')
 n.final={{name:n.arm[name] for name in RIGHT}}
 final_delta=max(abs(n.final[name]-n.initial[name]) for name in RIGHT)
 drift=math.hypot(n.odom[0]-n.start_odom[0],n.odom[1]-n.start_odom[1])
 response_code=int(response.header.code);accepted=response_code==0 and int(response.state.value) in (CommonState.SUCCESS,CommonState.RUNNING,CommonState.CREATED,CommonState.PENDING)
 print(json.dumps({{'motion_id':{motion_id},'area_id':{area_id},'request_area_id':int(req.area.value),'request_motion_id':int(req.motion.value),'request_play_timestamp':int(req.play_timestamp),'response_code':response_code,'response_state':int(response.state.value),'response_task_id':int(response.task_id),'max_right_arm_delta_rad':n.max_arm_delta,'final_right_arm_delta_rad':final_delta,'planar_drift_m':drift,'max_tilt_rad':n.max_tilt,'pass':accepted and n.max_arm_delta>=0.03 and drift<=0.05 and n.max_tilt<=math.radians(15)}}))
finally:
 n.destroy_node();rclpy.shutdown()
"""


def native_leg_safety_contract(*, include_upper_body=False):
    """Read-only intersection of pinned leg URDF and installed MC model limits."""
    import xml.etree.ElementTree as ET
    source=REPO/"reference/AgibotTech/agibot_x2_urdf"
    revision=run(["git","-C",str(source),"rev-parse","HEAD"],timeout_s=5.).stdout.strip()
    if revision!="575cc6b988f976c23550e0db85aa1e5475d3652d":raise ValueError("pinned X2 source revision")
    urdfs=list(source.rglob("X2-Ultra_omnipicker.urdf"))
    if len(urdfs)!=1:raise ValueError("unique official OmniPicker source required")
    joints={j.get("name"):j.find("limit") for j in ET.parse(urdfs[0]).getroot().findall("joint") if j.find("limit") is not None}
    native_path=RUNTIME/"mc/bin/mc_param/src/robot/lx2501_3_t2d5/robot_model.yaml"
    installed={item["name"]:item["limits"] for group in yaml.safe_load(native_path.read_text())["joints"] for item in group[1]}
    limits={}
    names=[side+"_"+suffix+"_joint" for side in ("left","right") for suffix in ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")]
    if include_upper_body:
        names += ["waist_yaw_joint","waist_pitch_joint","waist_roll_joint"]
        names += [side+"_"+suffix+"_joint" for side in ("left","right") for suffix in ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_yaw","wrist_pitch","wrist_roll")]
    for name in names:
        a,b=joints[name],installed[name]
        limits[name]=[max(float(a.get("lower")),b["lower_position"]),min(float(a.get("upper")),b["upper_position"]),min(float(a.get("velocity")),b["max_velocity"])]
    return limits,{"urdf":str(urdfs[0]),"sha256":sha256(urdfs[0]),"revision":revision,"installed_mc_robot_model":str(native_path),"installed_mc_robot_model_sha256":sha256(native_path),"bounds":limits,"policy":"intersection, never raise any pinned or native bound"}

def locomotion_dock_script(
    *,
    world_start_xy: list[float],
    world_start_yaw_rad: float,
    dock_world_xy: list[float],
    dock_world_yaw_rad: float,
    dock_position_only: bool,
    refine_world_xy: list[float] | None,
    refine_world_yaw_rad: float | None,
    refine_position_only: bool,
    dock_position_tolerance_m: float,
    refine_position_tolerance_m: float,
    timeout_s: float,
    use_cmd_vel_bridge: bool = False,
    native_pose_feedback: bool = False,
    braking_delay_s: float | None = None,
    native_leg_limits: dict | None = None,
    command_speed_cap: float | None = None,
    angular_speed_cap: float | None = None,
    native_arm_relay: bool = False,
) -> str:
    """Build one bounded, odometry-closed official velocity-control route.

    The generated client deliberately follows the official velocity interface:
    it registers one named source, publishes ``McLocomotionVelocity`` at 50 Hz,
    and ends with a sustained stream of zero commands before unregistering.
    It never addresses MuJoCo qpos/qvel or a simulator-only control topic.
    """
    payload = json.dumps(
        {
            "world_start_xy": world_start_xy,
            "world_start_yaw_rad": world_start_yaw_rad,
            "dock_world_xy": dock_world_xy,
            "dock_world_yaw_rad": dock_world_yaw_rad,
            "dock_position_only": dock_position_only,
            "refine_world_xy": refine_world_xy,
            "refine_world_yaw_rad": refine_world_yaw_rad,
            "refine_position_only": refine_position_only,
            "dock_position_tolerance_m": dock_position_tolerance_m,
            "refine_position_tolerance_m": refine_position_tolerance_m,
            "timeout_s": timeout_s,
            "use_cmd_vel_bridge": use_cmd_vel_bridge,
            "native_pose_feedback": native_pose_feedback,
            "braking_delay_s": braking_delay_s,
            "native_leg_limits":native_leg_limits,
            "command_speed_cap":command_speed_cap,
            "angular_speed_cap":angular_speed_cap,
            "native_arm_relay":native_arm_relay,
        }
    )
    import inspect
    from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose,physical_fault
    safety_source=inspect.getsource(native_pose)+"\n"+inspect.getsource(physical_fault)
    return """#!/usr/bin/env python3
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader, McInputAction, RequestHeader, JointStateArray
from aimdk_msgs.srv import GetCurrentInputSource, SetMcAction, SetMcInputSource
from aimdk_msgs.msg import McActionCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

PLAN = json.loads(r'''__PLAN__''')
SOURCE = 'robotsim_table_dock'
QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
def measured_stop_prediction(history,goal,start,delay):
 if delay is None or len(history)<3:return None
 end=history[-1][0];window=[r for r in history if end-r[0]<=.35]
 if len(window)<3 or window[-1][0]-window[0][0]<.20:return None
 mt=sum(r[0] for r in window)/len(window);mx=sum(r[1] for r in window)/len(window);my=sum(r[2] for r in window)/len(window)
 denom=sum((r[0]-mt)**2 for r in window)
 if denom<=0:return None
 vx=sum((r[0]-mt)*(r[1]-mx) for r in window)/denom;vy=sum((r[0]-mt)*(r[2]-my) for r in window)/denom
 x,y=window[-1][1:];px,py=x+vx*delay,y+vy*delay;dx,dy=goal[0]-start[0],goal[1]-start[1];length=math.hypot(dx,dy)
 if length<1e-6:return {'predicted_xy_m':[px,py],'stop':True,'crossed':False,'remaining_along_m':0.,'predicted_cross_track_m':0.,'delay_s':delay}
 ux,uy=dx/length,dy/length;remaining=(goal[0]-px)*ux+(goal[1]-py)*uy;cross=(px-start[0])*uy-(py-start[1])*ux
 return {'predicted_xy_m':[px,py],'measured_velocity_wall_m_s':[vx,vy],'remaining_along_m':remaining,'predicted_cross_track_m':cross,'stop':(math.hypot(goal[0]-px,goal[1]-py)<=.03 or remaining<=0.),'crossed':remaining<=0.,'delay_s':delay,'history_span_s':window[-1][0]-window[0][0]}
ARM=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
LEG=('left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint')
WAIST=('waist_yaw_joint','waist_pitch_joint','waist_roll_joint')

def wrap(a): return (a + math.pi) % (2.0 * math.pi) - math.pi
def yaw(q): return math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
def tilt(q):
 p=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x))))
 r=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y))
 return max(abs(p),abs(r))
def call(n,c,req):
 if not c.wait_for_service(timeout_sec=5.0): raise RuntimeError('service unavailable')
 f=c.call_async(req); end=time.monotonic()+5.0
 while rclpy.ok() and not f.done() and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
 if not f.done() or f.result() is None:raise RuntimeError('service timeout')
 return f.result()

__SAFETY_SOURCE__

class Dock(Node):
 def __init__(self):
  super().__init__('robotsim_native_table_dock')
  self.pub=self.create_publisher(Twist,'/cmd_vel',10) if PLAN['use_cmd_vel_bridge'] else self.create_publisher(McLocomotionVelocity,'/aima/mc/locomotion/velocity',QOS)
  self.src=self.create_client(SetMcInputSource,'/aimdk_5Fmsgs/srv/SetMcInputSource')
  self.mode=self.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction')
  self.current=self.create_client(GetCurrentInputSource,'/aimdk_5Fmsgs/srv/GetCurrentInputSource')
  self.pose=None;self.quat=None;self.max_tilt=0.;self.samples=[];self.seq=0;self.arm={};self.leg={};self.waist={}
  self.odom_stamp=None;self.odom_callbacks=0;self.odom_duplicates=0;self.odom_regressions=0;self.odom_last_unique_receipt=None;self.feedback_fault=None;self.command_publish_count=0
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,QOS)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,QOS)
  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,QOS)
  self.create_subscription(JointStateArray,'/aima/hal/joint/leg/state',self.on_leg,QOS)
  self.create_subscription(JointStateArray,'/aima/hal/joint/waist/state',self.on_waist,QOS)
 def on_odom(self,m):
  if PLAN.get('native_pose_feedback'):
   self.odom_callbacks+=1;return
  p,q=m.pose.pose.position,m.pose.pose.orientation
  stamp=int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec);self.odom_callbacks+=1
  if not all(math.isfinite(v) for v in (p.x,p.y,p.z,q.w,q.x,q.y,q.z)) or q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z<1e-12:
   self.feedback_fault={'gate':'INVALID_ODOMETRY'};return
  if stamp<=0:return
  now=time.monotonic()
  if self.odom_stamp is not None and stamp<self.odom_stamp:
   self.odom_regressions+=1
   # The installed simulator can deliver a late DDS sample with a source
   # stamp a few microseconds behind the latest sample. It is stale data, not
   # a new pose: discard it without refreshing freshness or changing state.
   self.feedback_fault={'gate':'ODOMETRY_SOURCE_REGRESSION','previous_stamp_ns':self.odom_stamp,'received_stamp_ns':stamp,'regression_count':self.odom_regressions,'late_sample_discarded':True} if now-(self.odom_last_unique_receipt or now)>.20 else None
   return
  if stamp==self.odom_stamp:self.odom_duplicates+=1;return
  self.odom_stamp=stamp
  self.odom_last_unique_receipt=now
  self.pose=(float(p.x),float(p.y),float(p.z),yaw(q),now);self.quat=[float(q.w),float(q.x),float(q.y),float(q.z)]
 def on_imu(self,m):self.max_tilt=max(self.max_tilt,tilt(m.orientation))
 def on_arm(self,m):self.arm.update({j.name:float(j.position) for j in m.joints})
 def on_leg(self,m):self.leg.update({j.name:float(j.position) for j in m.joints})
 def on_waist(self,m):self.waist.update({j.name:float(j.position) for j in m.joints})
 def state_frame(self):
  if self.pose is None or self.quat is None or any(j not in self.arm for j in ARM) or any(j not in self.leg for j in LEG) or any(j not in self.waist for j in WAIST):return None
  x,y,z,heading,_=self.pose
  return {'base_pose_world':{'xyz_m':[x,y,z],'quaternion_wxyz':self.quat},'leg_positions_rad':{j:self.leg[j] for j in LEG},'waist_positions_rad':{j:self.waist[j] for j in WAIST},'arm_positions_rad':{j:self.arm[j] for j in ARM}}
 def publish(self,fwd=0.,lat=0.,ang=0.):
  if self.feedback_fault is not None:fwd=lat=ang=0.
  self.command_publish_count+=1
  if PLAN['use_cmd_vel_bridge']:
   m=Twist();m.linear.x=float(fwd);m.linear.y=float(lat);m.angular.z=float(ang);self.pub.publish(m);return
  m=McLocomotionVelocity();m.header=MessageHeader();m.header.stamp=self.get_clock().now().to_msg();m.header.sequence=self.seq;self.seq+=1;m.source=SOURCE;m.forward_velocity=float(fwd);m.lateral_velocity=float(lat);m.angular_velocity=float(ang);self.pub.publish(m)
 def register(self):
  if PLAN['use_cmd_vel_bridge']:return {'transport':'cmd_vel_bridge','source':'robotsim_official_cmd_vel'}
  r=SetMcInputSource.Request();r.action=McInputAction();r.action.value=1001;r.input_source.name=SOURCE;r.input_source.priority=40;r.input_source.timeout=1000
  z=call(self,self.src,r)
  if int(z.response.header.code)!=0:raise RuntimeError('input source registration failed')
  return {'code':int(z.response.header.code),'state':int(z.response.state.value),'task_id':int(z.response.task_id)}
 def current_source(self):
  r=GetCurrentInputSource.Request();r.request.header.stamp=self.get_clock().now().to_msg();z=call(self,self.current,r)
  return str(z.input_source.name)
 def disable(self):
  if PLAN['use_cmd_vel_bridge']:return
  try:
   r=SetMcInputSource.Request();r.action=McInputAction();r.action.value=1003;r.input_source.name=SOURCE;r.input_source.priority=40;r.input_source.timeout=1000;call(self,self.src,r)
  except Exception:pass
 def stand(self):
  r=SetMcAction.Request();r.header=RequestHeader();r.header.stamp=self.get_clock().now().to_msg();r.source=SOURCE;r.command=McActionCommand();r.command.action_desc='STAND_DEFAULT';z=call(self,self.mode,r)
  if int(z.response.status.value)!=1:raise RuntimeError('STAND_DEFAULT request rejected')
 def refresh_native_pose(self):
  if not PLAN.get('native_pose_feedback'):return
  with open('/workspace/evidence/native_latest.json','rb') as f:
   f.seek(0,2);f.seek(max(0,f.tell()-262144));raw=f.read()
  lines=raw.splitlines()
  if not raw.endswith(bytes([10])):lines=lines[:-1]
  for line in reversed(lines):
   try:row=json.loads(line);break
   except (ValueError,UnicodeDecodeError):continue
  else:self.feedback_fault={'gate':'NO_NATIVE_DOCK_POSE'};return
  stamp=int(row['host_monotonic_ns']);now=time.monotonic();age=now-stamp*1e-9
  if PLAN.get('native_arm_relay'):
   try:
    with open('/workspace/evidence/native_arm_relay_heartbeat.json') as f:health=json.load(f)
    if not 0.<=now-health['host_monotonic_ns']*1e-9<=.20 or not 0.<=now-health['last_unique_source_receipt_s']<=.20:raise ValueError('stale heartbeat/source')
   except (OSError,ValueError,KeyError) as exc:self.feedback_fault=self.feedback_fault or {'gate':'ARM_POLICY_RELAY_UNAVAILABLE','detail':str(exc)};return
  if not 0.<=age<=.20 or row.get('engine_version')!=337:self.feedback_fault={'gate':'STALE_NATIVE_DOCK_POSE'};return
  if self.odom_stamp is not None and stamp<=self.odom_stamp:return
  if hasattr(self,'native_generation') and row['generation']!=self.native_generation:self.feedback_fault={'gate':'NATIVE_DOCK_RESET_GENERATION_CHANGED'};return
  p=row['bodies']['pelvis'];q=p['quaternion_wxyz'];R=p['rotation'];x,y,z=p['position_m']
  if PLAN.get('native_leg_limits'):
   fault=physical_fault(row,native_pose(row),PLAN['native_leg_limits'])
   if fault:self.feedback_fault=self.feedback_fault or {'gate':fault,'native_row_source_ns':stamp};return
  if not all(math.isfinite(v) for v in [x,y,z,*q]):self.feedback_fault={'gate':'NONFINITE_NATIVE_DOCK_POSE'};return
  self.native_generation=row['generation'];self.odom_stamp=stamp;self.odom_last_unique_receipt=stamp*1e-9
  self.pose=(x,y,z,math.atan2(R[3],R[0]),stamp*1e-9);self.quat=q
  if not hasattr(self,'native_pose_history'):self.native_pose_history=[]
  self.native_pose_history.append((stamp*1e-9,x,y));self.native_pose_history=[r for r in self.native_pose_history if stamp*1e-9-r[0]<=.40]
  self.max_tilt=max(self.max_tilt,abs(math.asin(max(-1.,min(1.,-R[6])))),abs(math.atan2(R[7],R[8])))
  for names,target in ((ARM,self.arm),(LEG,self.leg),(WAIST,self.waist)):
   for name in names:target[name]=float(row['joints'][name]['q'])
 def spin(self,seconds):
  end=time.monotonic()+seconds
  while rclpy.ok() and time.monotonic()<end:
   rclpy.spin_once(self,timeout_sec=.005);self.refresh_native_pose()
 def zero_hold(self,seconds):
  end=time.monotonic()+seconds
  while rclpy.ok() and time.monotonic()<end:
   tick=time.monotonic();self.publish();self.spin(max(0,min(end,tick+.02)-time.monotonic()))

rclpy.init();n=Dock();registered=False
try:
 n.stand(); n.spin(1.5)
 if n.pose is None:raise RuntimeError('no odometry after stable stand')
 sx,sy,sz,syaw,_=n.pose
 # The source scene defines both start and dock in its world frame.  Odom is
 # translated at startup but preserves the planar world axes in this delivery,
 # so apply the measured start offset and preserve the requested yaw delta.
 def odom_goal(world_xy,world_yaw):
  if PLAN.get('native_pose_feedback'):return world_xy[0],world_xy[1],wrap(world_yaw)
  dx=world_xy[0]-PLAN['world_start_xy'][0];dy=world_xy[1]-PLAN['world_start_xy'][1]
  # Position has an initial odom translation, but the delivered odometry yaw
  # is already expressed in the source-map world convention.  Do not add the
  # instantaneous post-stand yaw offset again, or each local dock target
  # acquires that offset twice.
  return sx+dx,sy+dy,wrap(world_yaw)
 goals=[('TABLE_DOCK',*odom_goal(PLAN['dock_world_xy'],PLAN['dock_world_yaw_rad']),PLAN['dock_position_only'])]
 if PLAN['refine_world_xy'] is not None:goals.append(('LOCAL_REFINE',*odom_goal(PLAN['refine_world_xy'],PLAN['refine_world_yaw_rad']),PLAN['refine_position_only']))
 goal_index=0;goal_name,gx,gy,gyaw,position_only=goals[goal_index];milestones=[];arrival_latched=False
 source=n.register();registered=True
 started=time.monotonic(); stable_since=None; phase='ALIGN'; failure=None; last_sample_at=None
 while time.monotonic()-started < PLAN['timeout_s']:
  rclpy.spin_once(n,timeout_sec=.001);n.refresh_native_pose()
  if n.feedback_fault is not None or n.pose is None or time.monotonic()-n.pose[4]>.20:
   n.publish(); failure=n.feedback_fault or {'gate':'ODOMETRY_FRESHNESS'};break
  x,y,z,heading,_=n.pose;dist=math.hypot(gx-x,gy-y);bearing=math.atan2(gy-y,gx-x);berr=wrap(bearing-heading);yerr=wrap(gyaw-heading);lat=0.
  prediction=measured_stop_prediction(getattr(n,'native_pose_history',[]),(gx,gy),(sx,sy),PLAN.get('braking_delay_s'))
  if z < .45 or n.max_tilt > math.radians(15):
   n.publish();failure={'gate':'BALANCE_ENVELOPE','z_m':z,'max_tilt_rad':n.max_tilt};break
  position_tolerance=PLAN['refine_position_tolerance_m'] if goal_name=='LOCAL_REFINE' else PLAN['dock_position_tolerance_m']
  # Once a position-only goal enters its accepted radius, latch the hold phase.
  # Re-entering LOCAL_TURN after stand-transition drift caused large yaw changes
  # and walked the base back out of the docking area.
  predicted_stop=bool(prediction and prediction['stop'])
  if position_only and not arrival_latched and (dist<=position_tolerance if PLAN.get('braking_delay_s') is None else predicted_stop):
   arrival_latched=True;stable_since=None
   print(json.dumps({'dock_stop_trigger':{'actual_xy_m':[x,y],'actual_error_m':dist,'prediction':prediction,'acceptance_not_yet_passed':True}}),flush=True)
  if arrival_latched:
   phase='HOLD';fwd=lat=ang=0.
   drift_limit=max(.35,position_tolerance+.15)
   if dist>drift_limit:
    failure={'gate':'DOCK_HOLD_DRIFT','distance_m':dist,'limit_m':drift_limit};break
   if stable_since is None:stable_since=time.monotonic()
   if time.monotonic()-stable_since>=1.5:
    if dist>position_tolerance:
     failure={'gate':'DOCK_POST_ZERO_OUTSIDE_TOLERANCE','distance_m':dist,'tolerance_m':position_tolerance};break
    milestones.append({'goal':goal_name,'position_only':position_only,'odom_xy_m':[x,y],'odom_yaw_rad':heading,'distance_m':dist,'yaw_error_rad':yerr,'arrival_latched':True})
    if goal_index+1<len(goals):
     goal_index+=1;goal_name,gx,gy,gyaw,position_only=goals[goal_index];stable_since=None;arrival_latched=False;phase='REFINE_ALIGN';continue
    break
  elif position_only and (dist>position_tolerance or PLAN.get('braking_delay_s') is not None and not arrival_latched):
   # Follow the delivered/community move_toward controller: turn toward the
   # local target before moving, then reduce forward speed with distance.
   # Earlier direct strafe and forced forward-arc variants both failed this
   # exact local pose loop, so they are deliberately not used here.
   if abs(berr)>.22:
    phase='LOCAL_TURN';fwd=0.;ang=max(-.40,min(.40,1.4*berr))
   else:
    phase='LOCAL_DRIVE';floor=.12 if PLAN.get('braking_delay_s') is not None else .06;fwd=max(floor,min(.20,.55*dist))*max(.25,math.cos(berr));ang=max(-.40,min(.40,1.4*berr))
  elif dist>PLAN['dock_position_tolerance_m']:
   if abs(berr)>.22:
    # The delivered controller's observed, source-managed turn primitive is
    # a forward arc.  A zero-forward yaw request stays almost stationary in
    # this version, whereas a bounded forward-plus-yaw command is the actual
    # locomotion behavior.  The pose loop still owns the stop condition.
    phase='ARC_ALIGN';fwd=.20;ang=max(-.25,min(.25,.8*berr))
   else:
    phase='DRIVE';fwd=max(.20,min(.30,.65*dist));ang=max(-.25,min(.25,.8*berr))
  # Aim inside the final acceptance boundary.  STAND_DEFAULT recovery after
  # the zero-velocity hold can perturb heading slightly, so stopping exactly
  # at the 0.12-rad acceptance edge is not a stable terminal condition.
  elif not position_only and abs(yerr)>.08:
   phase='FINAL_ALIGN';fwd=0.;ang=max(-.25,min(.25,1.0*yerr))
  else:
   phase='HOLD';fwd=ang=0.
   if stable_since is None: stable_since=time.monotonic()
   if time.monotonic()-stable_since>=1.5:
    milestones.append({'goal':goal_name,'position_only':position_only,'odom_xy_m':[x,y],'odom_yaw_rad':heading,'distance_m':dist,'yaw_error_rad':yerr})
    if goal_index+1 < len(goals):
     goal_index+=1;goal_name,gx,gy,gyaw,position_only=goals[goal_index];stable_since=None;phase='REFINE_ALIGN';continue
    break
  if phase!='HOLD':stable_since=None
  if PLAN.get('command_speed_cap') is not None:fwd=max(-PLAN['command_speed_cap'],min(PLAN['command_speed_cap'],fwd))
  if PLAN.get('angular_speed_cap') is not None:ang=max(-PLAN['angular_speed_cap'],min(PLAN['angular_speed_cap'],ang))
  n.publish(fwd,lat,ang)
  now=time.monotonic()
  if last_sample_at is None or now-last_sample_at>=.10:
   entry={'wall_s':now-started,'goal':goal_name,'position_only':position_only,'phase':phase,'odom_xyz_m':[x,y,z],'yaw_rad':heading,'goal_distance_m':dist,'goal_yaw_error_rad':yerr,'forward_mps':fwd,'lateral_mps':lat,'angular_rps':ang,'feedback_source':'NATIVE_MJDATA_TRUTH' if PLAN.get('native_pose_feedback') else 'ROS_ODOM','odom_source_stamp_ns':n.odom_stamp,'odom_unique_receipt_monotonic_s':n.pose[4],'odom_feedback_age_s':now-n.pose[4],'odom_callbacks':n.odom_callbacks,'odom_duplicates':n.odom_duplicates,'odom_regressions':n.odom_regressions}
   state=n.state_frame()
   if state is not None:entry['joint_state']=state
   n.samples.append(entry)
   last_sample_at=now
  n.spin(.019)
 n.zero_hold(1.5)
 # The delivered velocity interface returns to stable standing after its
 # timeout.  Request and observe that transition explicitly before exposing a
 # route terminal state to manipulation, so arm swing states are not mistaken
 # for a docked manipulation posture.
 n.stand(); n.spin(2.2)
 if n.pose is None:raise RuntimeError('odometry lost during final hold')
 if n.feedback_fault is not None or time.monotonic()-n.pose[4]>.20:
  failure=n.feedback_fault or {'gate':'FINAL_ODOMETRY_FRESHNESS'}
 x,y,z,heading,_=n.pose;dist=math.hypot(gx-x,gy-y);yerr=wrap(gyaw-heading)
 if failure is None and not (stable_since is not None and time.monotonic()-stable_since>=1.5):failure={'gate':'TABLE_DOCK_ARRIVAL','distance_m':dist,'yaw_error_rad':yerr}
 frame_count=sum(1 for row in n.samples if 'joint_state' in row)
 print(json.dumps({'source':source,'active_source_before_stop':n.current_source(),'initial_odom_xyz_m':[sx,sy,sz],'initial_yaw_rad':syaw,'goals_odom':[{'name':name,'xy_m':[x,y],'yaw_rad':yaw,'position_only':position_only} for name,x,y,yaw,position_only in goals],'milestones':milestones,'final_odom_xyz_m':[x,y,z],'final_yaw_rad':heading,'final_distance_m':dist,'final_yaw_error_rad':yerr,'max_tilt_rad':n.max_tilt,'joint_state_frame_count':frame_count,'trace':n.samples,'failure':failure,'pass':failure is None and len(milestones)==len(goals) and frame_count>=2 and dist<= (PLAN['refine_position_tolerance_m'] if position_only else PLAN['dock_position_tolerance_m']) and (position_only or abs(yerr)<=.12) and n.max_tilt<=math.radians(15)}))
finally:
 n.zero_hold(.5)
 if registered:n.disable()
 n.destroy_node();rclpy.shutdown()
""".replace("__PLAN__", payload).replace("__SAFETY_SOURCE__",safety_source)


def lateral_probe_script(lateral_mps: float, duration_s: float) -> str:
    """Build one bounded official ``cmd_vel`` lateral-motion probe.

    This deliberately exercises only the native locomotion stack.  It does
    not alter the model, hand, arm, object, or scene and it returns to
    ``STAND_DEFAULT`` before the result is exposed to a later manipulation
    stage.  The source-map/world convention is x-forward, y-left, z-up.
    """
    plan = json.dumps({"lateral_mps": lateral_mps, "duration_s": duration_s})
    return """#!/usr/bin/env python3
import json, math, time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from aimdk_msgs.msg import McActionCommand, RequestHeader
from aimdk_msgs.srv import SetMcAction

PLAN=json.loads(r'''__PLAN__''')
QOS=QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
def yaw(q): return math.atan2(2.0*(q.w*q.z+q.x*q.y),1.0-2.0*(q.y*q.y+q.z*q.z))
def tilt(q):
 p=math.asin(max(-1.0,min(1.0,2.0*(q.w*q.y-q.z*q.x))))
 r=math.atan2(2.0*(q.w*q.x+q.y*q.z),1.0-2.0*(q.x*q.x+q.y*q.y))
 return max(abs(p),abs(r))
class Probe(Node):
 def __init__(self):
  super().__init__('robotsim_native_lateral_probe')
  self.pub=self.create_publisher(Twist,'/cmd_vel',10);self.mode=self.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction')
  self.pose=None;self.max_tilt=0.0;self.samples=[]
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,QOS)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,QOS)
 def on_odom(self,m):
  p=m.pose.pose.position;self.pose=(float(p.x),float(p.y),float(p.z),yaw(m.pose.pose.orientation))
 def on_imu(self,m):self.max_tilt=max(self.max_tilt,tilt(m.orientation))
 def spin(self,seconds,lat=0.0):
  end=time.monotonic()+seconds;next_tick=time.monotonic()
  while rclpy.ok() and time.monotonic()<end:
   msg=Twist();msg.linear.y=float(lat);self.pub.publish(msg)
   rclpy.spin_once(self,timeout_sec=.001)
   if self.pose is not None:self.samples.append({'wall_time_s':time.monotonic(),'odom_xyz_m':list(self.pose[:3]),'yaw_rad':self.pose[3],'command_lateral_mps':float(lat),'max_tilt_rad':self.max_tilt})
   next_tick+=.02;time.sleep(max(0.0,next_tick-time.monotonic()))
 def stand(self):
  if not self.mode.wait_for_service(timeout_sec=5.0):raise RuntimeError('SetMcAction unavailable')
  q=SetMcAction.Request();q.header=RequestHeader();q.header.stamp=self.get_clock().now().to_msg();q.source='robotsim_native_lateral_probe';q.command=McActionCommand();q.command.action_desc='STAND_DEFAULT'
  f=self.mode.call_async(q);rclpy.spin_until_future_complete(self,f,timeout_sec=5.0)
  if not f.done() or f.result() is None or int(f.result().response.status.value)!=1:raise RuntimeError('STAND_DEFAULT request rejected')
rclpy.init();n=Probe()
try:
 n.stand();n.spin(1.5)
 if n.pose is None:raise RuntimeError('no odometry after stable stand')
 start=n.pose;n.spin(PLAN['duration_s'],PLAN['lateral_mps'])
 if n.pose is None:raise RuntimeError('odometry lost during lateral motion')
 motion_end=n.pose;n.spin(.8,0.0);stop_start=n.pose;n.stand();n.spin(2.2,0.0)
 if n.pose is None:raise RuntimeError('odometry lost during final stand hold')
 final=n.pose
 dx=motion_end[0]-start[0];dy=motion_end[1]-start[1];heading=start[3]
 forward=math.cos(heading)*dx+math.sin(heading)*dy
 lateral=-math.sin(heading)*dx+math.cos(heading)*dy
 stop_drift=math.hypot(final[0]-stop_start[0],final[1]-stop_start[1])
 same_direction=lateral*PLAN['lateral_mps']>0.0
 passed=(same_direction and abs(lateral)>=.02 and abs(forward)<=.12 and min(start[2],motion_end[2],final[2])>=.45 and n.max_tilt<=math.radians(15) and stop_drift<=.05)
 print(json.dumps({'command_lateral_mps':PLAN['lateral_mps'],'command_duration_s':PLAN['duration_s'],'initial_odom_xyz_m':list(start[:3]),'motion_end_odom_xyz_m':list(motion_end[:3]),'final_odom_xyz_m':list(final[:3]),'start_yaw_rad':heading,'body_frame_displacement_m':{'forward':forward,'lateral':lateral},'post_stop_drift_m':stop_drift,'max_tilt_rad':n.max_tilt,'sample_count':len(n.samples),'trace':n.samples,'pass':passed}))
finally:
 try:n.spin(.3,0.0)
 except Exception:pass
 n.destroy_node();rclpy.shutdown()
""".replace("__PLAN__", plan)


def retarget_mode_script() -> str:
    """Switch only to the installed retargeting action and report its state."""
    return """#!/usr/bin/env python3
import json,time
import rclpy
from rclpy.node import Node
from aimdk_msgs.msg import CommonRequest,McActionCommand,RequestHeader
from aimdk_msgs.srv import GetMcAction,SetMcAction
def call(n,c,q):
 if not c.wait_for_service(timeout_sec=5):raise RuntimeError('service unavailable')
 f=c.call_async(q);rclpy.spin_until_future_complete(n,f,timeout_sec=5)
 if not f.done() or f.result() is None:raise RuntimeError('service timeout')
 return f.result()
rclpy.init();n=Node('robotsim_retarget_mode_probe')
try:
 setc=n.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction');q=SetMcAction.Request();q.header=RequestHeader();q.header.stamp=n.get_clock().now().to_msg();q.source='robotsim_retarget_mode_probe';q.command=McActionCommand();q.command.action_desc='WHOLE_BODY_TELEOP';reply=call(n,setc,q)
 getc=n.create_client(GetMcAction,'/aimdk_5Fmsgs/srv/GetMcAction');end=time.monotonic()+1.5;latest=None
 while time.monotonic()<end:
  r=GetMcAction.Request();r.request=CommonRequest();r.request.header.stamp=n.get_clock().now().to_msg();latest=call(n,getc,r);rclpy.spin_once(n,timeout_sec=.02)
 print(json.dumps({'set_status':int(reply.response.status.value),'set_message':reply.response.message,'mode_name':latest.info.action_desc,'mode_status':int(latest.info.status.value),'pass':int(reply.response.status.value)==1 and latest.info.action_desc=='WHOLE_BODY_TELEOP'}))
finally:
 n.destroy_node();rclpy.shutdown()
"""


def vr_mode_script() -> str:
    """Switch only to the delivered VR action and confirm the controller state."""
    return retarget_mode_script().replace("WHOLE_BODY_TELEOP", "VR_REMOTE_CONTROLLER").replace("robotsim_retarget_mode_probe", "robotsim_vr_mode_probe")


def vr_compile_script() -> str:
    """Compile the pinned C probe using only the container's installed headers."""
    return """#!/usr/bin/env bash
set -euo pipefail
include_dirs=()
for directory in /opt/ros/humble/include/*; do
  [[ -d \"${directory}\" ]] || continue
  case \"${directory##*/}\" in dds|idl) continue ;; esac
  include_dirs+=(\"-I${directory}\")
done
gcc -std=c11 -O2 -Wall -Wextra -D_DEFAULT_SOURCE \"${include_dirs[@]}\" \\
  /workspace/evidence/native_vr_data_probe.c \\
  -L/workspace/raicom2026/mc/lib -L/opt/ros/humble/lib \\
  -Wl,-rpath,/workspace/raicom2026/mc/lib -Wl,-rpath,/opt/ros/humble/lib \\
  -laimdk_msgs__rosidl_generator_c -laimdk_msgs__rosidl_typesupport_c \\
  -lrosidl_runtime_c -lrcl -lrcutils -lm -o /workspace/evidence/native_vr_zero_probe
"""


def arm_bootstrap_hold_script(gains: dict[str, tuple[float, float]], *, relay_waist: bool = False, relay_arms: bool = False, initial_hand_opening: float | None = None, open_hand_transit_park: bool = False, initial_arm_target=None, initial_arm_effort=None) -> str:
    """Hold arms at the installed MC's neutral target before MC owns the legs.

    This is an explicitly simulation-only bootstrap for an arm-split MC
    overlay.  It does not publish leg, waist, hand, or locomotion commands.
    A marker gives the caller an orderly hand-off to the bounded arm probe.
    """
    if initial_hand_opening is not None and initial_hand_opening != 1.0:
        raise ValueError("startup hand opening must be exactly 1.0 or disabled")
    encoded = json.dumps({name: [float(kp), float(kd)] for name, (kp, kd) in gains.items()}, sort_keys=True)
    encoded_target=json.dumps(initial_arm_target) if initial_arm_target is not None else "None"
    encoded_effort=json.dumps(initial_arm_effort or {})
    return """#!/usr/bin/env python3
import json, math, os, time, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, MessageHeader, HandCommand, HandCommandArray, HandType, McCommonState
ARM=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
TARGET=(.4,0.,0.,-1.2,0.,0.,0.,.4,0.,0.,-1.2,0.,0.,0.)
RELAY_ARMS=__RELAY_ARMS__; RELAY_WAIST=__RELAY_WAIST__; INITIAL_HAND_OPENING=__INITIAL_HAND_OPENING__; OPEN_HAND_TRANSIT_PARK=__OPEN_HAND_TRANSIT_PARK__
INITIAL_ARM_TARGET=__INITIAL_ARM_TARGET__; INITIAL_ARM_EFFORT=__INITIAL_ARM_EFFORT__
GAINS=__GAINS__; STOP='/workspace/evidence/stop_arm_bootstrap'
class Hold(Node):
 def __init__(self):
  super().__init__('robotsim_arm_split_bootstrap');q=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT);self.state={};self.pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/arm/command',q);self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',lambda m:self.state.update({x.name:float(x.position) for x in m.joints}),q);self.seq=0;self.ready=False;self.waist_last_receipt=None;self.waist_relay_count=0
  self.started=time.monotonic();self.hand_pub=None
  if INITIAL_HAND_OPENING is not None:
   self.hand_pub=self.create_publisher(HandCommandArray,'/aima/hal/joint/hand/command',QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.TRANSIENT_LOCAL))
  self.arm_msg=None;self.arm_source_stamp=0;self.arm_receipt=None;self.arm_relay_count=0;self.arm_relay_started=None;self.arm_handoff_start=None;self.arm_last_metadata_time=0.
  self.common_value=None;self.common_stamp=0;self.common_receipt=None
  if RELAY_ARMS:self.create_subscription(McCommonState,'/aima/mc/common/state',self.on_common,q)
  if RELAY_ARMS:self.create_subscription(JointCommandArray,'/robotsim/issue46/mc/arm/command',self.on_mc_arm,q)
  self.waist_pub=None
  if RELAY_WAIST:
   self.waist_pub=self.create_publisher(JointCommandArray,'/aima/hal/joint/waist/command',q)
   self.create_subscription(JointCommandArray,'/robotsim/issue46/mc/waist/command',self.on_waist,q)
 def on_common(self,m):
  stamp=int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
  if stamp<=self.common_stamp:return
  self.common_stamp=stamp;self.common_receipt=time.monotonic()
  self.common_value=(str(m.action_info.action_desc),int(m.action_info.status.value),int(m.fsm_state.current_state),int(m.body_status.value))
 def on_mc_arm(self,m):
  if len(m.joints)!=14 or {j.name for j in m.joints}!=set(ARM):raise RuntimeError('INVALID_MC_ARM_GROUP')
  if any(not all(math.isfinite(float(getattr(j,f))) for f in ('position','velocity','effort','stiffness','damping')) for j in m.joints):raise RuntimeError('NONFINITE_MC_ARM_COMMAND')
  stamp=int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
  if stamp<=self.arm_source_stamp:return
  self.arm_source_stamp=stamp;self.arm_receipt=time.monotonic();self.arm_msg=m
  first='/workspace/evidence/native_arm_relay_initial.json'
  if not os.path.exists(first):
   with open(first,'x') as f:json.dump({'source_stamp_ns':stamp,'joints':{j.name:{k:float(getattr(j,k)) for k in ('position','velocity','effort','stiffness','damping')} for j in m.joints}},f)
  if all(j.stiffness>0. and j.damping>0. for j in m.joints) and time.monotonic()-self.arm_last_metadata_time>=.02:
   path='/workspace/evidence/native_arm_relay_source_latest.json'
   with open(path+'.tmp','w') as f:json.dump({'source_stamp_ns':stamp,'receipt_host_ns':time.monotonic_ns(),'joints':{j.name:{k:float(getattr(j,k)) for k in ('position','velocity','effort','stiffness','damping')} for j in m.joints}},f)
   os.replace(path+'.tmp',path);self.arm_last_metadata_time=time.monotonic()
  if self.arm_relay_started is not None and time.monotonic()-self.arm_relay_started>=2.:
   self.pub.publish(m);self.arm_relay_count+=1
 def on_waist(self,m):
  if len(m.joints)!=3 or {j.name for j in m.joints}!={'waist_yaw_joint','waist_pitch_joint','waist_roll_joint'}:raise RuntimeError('INVALID_MC_WAIST_GROUP')
  if any(not all(math.isfinite(float(getattr(j,f))) for f in ('position','velocity','effort','stiffness','damping')) for j in m.joints):raise RuntimeError('NONFINITE_MC_WAIST_COMMAND')
  # Forward unchanged MC fields while walking. Do not replace its balancing
  # targets with pre-Reset measured joint angles or remote-controller gains.
  self.waist_pub.publish(m);self.waist_last_receipt=time.monotonic();self.waist_relay_count+=1
 def send(self):
  m=JointCommandArray();m.header=MessageHeader();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='hybrid_arm_bootstrap';m.header.sequence=self.seq;self.seq+=1
  for name,target in zip(ARM,TARGET):
   if INITIAL_ARM_TARGET is not None:target=float(INITIAL_ARM_TARGET[name])
   if OPEN_HAND_TRANSIT_PARK and name=='right_shoulder_pitch_joint':
    u=max(0.,min(1.,(time.monotonic()-self.started)/2.));target=.4+.4*(3*u*u-2*u*u*u)
   x=JointCommand();x.name=name;x.position=target;x.velocity=0.;x.effort=float(INITIAL_ARM_EFFORT.get(name,0.));x.stiffness=GAINS[name][0];x.damping=GAINS[name][1];m.joints.append(x)
  if RELAY_ARMS and os.path.exists('/workspace/evidence/enable_native_arm_relay'):
   if self.arm_relay_started is None or time.monotonic()-self.arm_relay_started<2.:
    if self.common_receipt is None or time.monotonic()-self.common_receipt>.20 or self.common_value!=('STAND_DEFAULT',100,4,1):raise RuntimeError('UNQUALIFIED_MC_STAND_ARM_HANDOFF')
   if self.arm_msg is None or time.monotonic()-self.arm_receipt>.20:raise RuntimeError('STALE_MC_ARM_RELAY')
   if self.arm_relay_started is None:
    marker=json.load(open('/workspace/evidence/enable_native_arm_relay'))
    pubs=self.get_publishers_info_by_topic('/robotsim/issue46/mc/arm/command')
    public=self.get_publishers_info_by_topic('/aima/hal/joint/arm/command')
    identity=(len(pubs)==1 and pubs[0].node_name==marker['expected_mc_node'] and pubs[0].node_namespace=='/' and len(public)==1 and public[0].node_name==self.get_name())
    if not identity:raise RuntimeError('NATIVE_ARM_RELAY_OWNER_NOT_VERIFIED')
    if any(j.stiffness<=0. or j.damping<=0. for j in self.arm_msg.joints):raise RuntimeError('INACTIVE_MC_ARM_GAINS')
    row=json.load(open('/workspace/evidence/native_latest.json'))
    if row['generation']!=marker['native_generation'] or row['engine_version']!=337 or not 0.<=(time.monotonic_ns()-row['host_monotonic_ns'])*1e-9<=.20:raise RuntimeError('STALE_NATIVE_ARM_HANDOFF')
    self.arm_handoff_start={name:float(row['joints'][name]['q']) for name in ARM};self.arm_relay_started=time.monotonic()
    if max(abs(j.position-self.arm_handoff_start[j.name]) for j in self.arm_msg.joints)>1.1:raise RuntimeError('SOURCE_ARM_HANDOFF_EXCURSION')
   elapsed=time.monotonic()-self.arm_relay_started
   if elapsed<2.:
    u=max(0.,min(1.,elapsed/2.));u=3*u*u-2*u*u*u
    # Bumpless position handoff to the actual unmodified MC gain/effort fields.
    for dst,src in zip(m.joints,self.arm_msg.joints):
     if dst.name!=src.name:
      src=next(j for j in self.arm_msg.joints if j.name==dst.name)
     dst.position=self.arm_handoff_start[dst.name]+u*(src.position-self.arm_handoff_start[dst.name])
     dst.velocity=src.velocity;dst.effort=src.effort;dst.stiffness=src.stiffness;dst.damping=src.damping
    self.pub.publish(m)
   elif self.arm_relay_count>0 and not os.path.exists('/workspace/evidence/native_arm_relay_ready.json'):
    with open('/workspace/evidence/native_arm_relay_ready.json','x') as f:json.dump({'ready':True,'source_stamp_ns':self.arm_source_stamp,'relay_count':self.arm_relay_count,'source_fields_unchanged_after_handoff':True},f)
  else:self.pub.publish(m)
  if self.hand_pub is not None:
   # Retain the delivered initialization handshake, then hold fully open.
   command=0.0 if time.monotonic()-self.started<.8 else INITIAL_HAND_OPENING
   h=HandCommandArray();h.left_hand_type=HandType(value=2);h.right_hand_type=HandType(value=2)
   for side,items in (('left',h.left_hands),('right',h.right_hands)):
    x=HandCommand();x.name=side+'_claw_joint';x.position=command;x.velocity=1.;x.acceleration=1.;x.deceleration=1.;x.effort=1.;items.append(x)
   self.hand_pub.publish(h)
  if RELAY_ARMS and self.arm_relay_started is not None:
   path='/workspace/evidence/native_arm_relay_heartbeat.json'
   with open(path+'.tmp','w') as f:json.dump({'host_monotonic_ns':time.monotonic_ns(),'last_unique_source_receipt_s':self.arm_receipt,'source_stamp_ns':self.arm_source_stamp,'relay_count':self.arm_relay_count},f)
   os.replace(path+'.tmp',path)
  if self.waist_last_receipt is not None and time.monotonic()-self.waist_last_receipt>.20:raise RuntimeError('STALE_MC_WAIST_RELAY')
rclpy.init();n=Hold()
try:
 # Native docking plus a host-side zero-step IK replan can take longer than
 # the original short probe window.  This remains a bounded neutral arm hold
 # and exits immediately at the explicit hand-off marker.
 end=time.monotonic()+90.;next_tick=time.monotonic()
 while rclpy.ok() and time.monotonic()<end and not os.path.exists(STOP):
  n.send();rclpy.spin_once(n,timeout_sec=.001)
  if not n.ready and all(name in n.state for name in ARM) and n.pub.get_subscription_count()==1:
   n.ready=True;print(json.dumps({'ready':True,'subscription_count':n.pub.get_subscription_count(),'arm_positions_rad':{name:n.state[name] for name in ARM}}),flush=True)
  next_tick+=.01
  while rclpy.ok() and time.monotonic()<next_tick:rclpy.spin_once(n,timeout_sec=.001)
 print(json.dumps({'stopped':os.path.exists(STOP),'ready':n.ready,'waist_relay_count':n.waist_relay_count,'arm_relay_count':n.arm_relay_count,'subscription_count':n.pub.get_subscription_count()}),flush=True)
finally:n.destroy_node();rclpy.shutdown()
""".replace("__GAINS__", encoded).replace("__RELAY_WAIST__", repr(relay_waist)).replace("__RELAY_ARMS__",repr(relay_arms)).replace("__INITIAL_HAND_OPENING__", repr(initial_hand_opening)).replace("__OPEN_HAND_TRANSIT_PARK__", repr(open_hand_transit_park)).replace("__INITIAL_ARM_TARGET__", encoded_target).replace("__INITIAL_ARM_EFFORT__", encoded_effort)


def retarget_neutral_script() -> str:
    """Publish a hold-only McRetargetFrame through the installed MC boundary."""
    return """#!/usr/bin/env python3
import argparse,json,math,time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy,QoSProfile,ReliabilityPolicy
from aimdk_msgs.msg import McRetargetFrame
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
QOS=QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
def tilt(q):
 p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))));r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y));return max(abs(p),abs(r))
class Probe(Node):
 def __init__(self,target,duration,capture_only):
  super().__init__('robotsim_retarget_neutral_probe');self.target=target;self.duration=duration;self.capture_only=capture_only;self.pose=None;self.start_pose=None;self.latest=None;self.pub=self.create_publisher(McRetargetFrame,'/aima/mc/joint/retargeting',QOS);self.max_tilt=0.;self.seq=0;self.frames=0
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,QOS);self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,QOS);self.create_subscription(McRetargetFrame,'/aima/mc/retarget_frame',self.on_retarget,QOS)
 def on_odom(self,m):
  p=m.pose.pose.position;self.pose=(float(p.x),float(p.y),float(p.z));self.start_pose=self.start_pose or self.pose
 def on_imu(self,m):self.max_tilt=max(self.max_tilt,tilt(m.orientation))
 def on_retarget(self,m):self.latest={'joint_state':[float(v) for v in m.joint_state],'root_position':[float(v) for v in m.root_position],'root_quarteniond':[float(v) for v in m.root_quarteniond],'left_hand_type':int(m.left_hand_type.value),'right_hand_type':int(m.right_hand_type.value)}
 def run(self):
  end=time.monotonic()+self.duration;next_tick=time.monotonic()
  while rclpy.ok() and time.monotonic()<end:
   if not self.capture_only:
    m=McRetargetFrame();now=self.get_clock().now().to_msg();m.header.stamp=now;m.header.meas_stamp=now;m.header.frame_id='mc_retargeting';m.header.sequence=self.seq;self.seq+=1
    m.joint_state=[float(v) for v in self.target['joint_state']];m.root_position=[float(v) for v in self.target['root_position']];m.root_quarteniond=[float(v) for v in self.target['root_quarteniond']];m.left_hand_type.value=int(self.target['left_hand_type']);m.right_hand_type.value=int(self.target['right_hand_type']);self.pub.publish(m);self.frames+=1
   rclpy.spin_once(self,timeout_sec=.001);next_tick+=.02;time.sleep(max(0.,next_tick-time.monotonic()))
parser=argparse.ArgumentParser();parser.add_argument('--target');parser.add_argument('--duration-s',type=float,default=1.5);parser.add_argument('--capture-only',action='store_true');args=parser.parse_args();
if not args.capture_only and not args.target:parser.error('--target is required unless --capture-only is used')
target={} if args.target is None else json.load(open(args.target));rclpy.init();n=Probe(target,args.duration_s,args.capture_only)
try:
 n.run();end=time.monotonic()+1.0
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.02)
 drift=None if n.pose is None or n.start_pose is None else math.hypot(n.pose[0]-n.start_pose[0],n.pose[1]-n.start_pose[1])
 payload={'frames':n.frames,'subscription_count':n.pub.get_subscription_count(),'initial_odom_xyz_m':None if n.start_pose is None else list(n.start_pose),'final_odom_xyz_m':None if n.pose is None else list(n.pose),'planar_drift_m':drift,'max_tilt_rad':n.max_tilt,'captured_frame':n.latest}
 payload['pass']=bool(n.latest) and (args.capture_only or (n.frames>=max(10,int(args.duration_s*40)) and n.pub.get_subscription_count()>=1 and n.pose is not None and n.pose[2]>=.45 and n.max_tilt<=math.radians(15) and drift is not None and drift<=.05))
 print(json.dumps(payload))
finally:n.destroy_node();rclpy.shutdown()
"""


def retarget_overlay_files() -> dict[str, str]:
    """Exact interface shape read from the installed introspection library."""
    return {
        "CMakeLists.txt": """cmake_minimum_required(VERSION 3.8)
project(aimdk_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  \"msg/MessageHeader.msg\" \"msg/HandType.msg\" \"msg/McRetargetFrame.msg\"
  DEPENDENCIES builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
""",
        "package.xml": """<?xml version=\"1.0\"?>
<package format=\"3\"><name>aimdk_msgs</name><version>0.0.0</version><description>Isolated binary-derived retargeting interface probe</description><maintainer email=\"noreply@example.invalid\">RobotSim</maintainer><license>MIT</license><buildtool_depend>ament_cmake</buildtool_depend><buildtool_depend>rosidl_default_generators</buildtool_depend><depend>builtin_interfaces</depend><exec_depend>rosidl_default_runtime</exec_depend><member_of_group>rosidl_interface_packages</member_of_group></package>
""",
        "msg/MessageHeader.msg": """builtin_interfaces/Time stamp
string frame_id
uint32 sequence
builtin_interfaces/Time meas_stamp
""",
        "msg/HandType.msg": """uint8 value
""",
        "msg/McRetargetFrame.msg": """MessageHeader header
float32[] joint_state
float32[] root_position
float32[] root_quarteniond
HandType left_hand_type
HandType right_hand_type
""",
    }


def urs_plan_script(right_target: list[float], duration_s: float, hold_s: float, left_target: list[float] | None = None, action_name: str = "WHOLE_BODY_TELEOP") -> str:
    """Build a bounded URS arm plan using the documented input-source path."""
    serialized = json.dumps(right_target)
    serialized_left = "None" if left_target is None else json.dumps(left_target)
    serialized_action = json.dumps(action_name)
    return """#!/usr/bin/env python3
import json, math, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import (CommonState, McActionCommand, McInputAction, MessageHeader, RequestHeader, UpperBodyCommandArray)
from aimdk_msgs.srv import SetMcAction, SetMcInputSource, GetCurrentInputSource
from aimdk_msgs.msg import JointStateArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

ARM=('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
RIGHT=ARM[7:]
TARGET=__TARGET__
LEFT_TARGET=__LEFT_TARGET__
DURATION=__DURATION__
HOLD=__HOLD__
ACTION=__ACTION__
SOURCE='upper_body_example'; PRIORITY=40; TIMEOUT_MS=1000
class Plan(Node):
 def __init__(self):
  super().__init__('robotsim_urs_truth_pregrasp'); q=QoSProfile(depth=50,reliability=ReliabilityPolicy.BEST_EFFORT)
  self.arm={};self.head=[0.0,0.0];self.odom=[];self.tilt=[];self.seq=0
  self.pub=self.create_publisher(UpperBodyCommandArray,'/mc/upper_body_command',q)
  self.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',self.on_arm,q)
  self.create_subscription(JointStateArray,'/aima/hal/joint/head/state',self.on_head,q)
  self.create_subscription(Odometry,'/aima/hal/odom/state',self.on_odom,q)
  self.create_subscription(Imu,'/aima/hal/imu/torso/state',self.on_imu,q)
  self.mode=self.create_client(SetMcAction,'/aimdk_5Fmsgs/srv/SetMcAction');self.setsrc=self.create_client(SetMcInputSource,'/aimdk_5Fmsgs/srv/SetMcInputSource');self.getsrc=self.create_client(GetCurrentInputSource,'/aimdk_5Fmsgs/srv/GetCurrentInputSource')
 def on_arm(self,m): self.arm={x.name:float(x.position) for x in m.joints}
 def on_head(self,m):
  vals={x.name:float(x.position) for x in m.joints};self.head=[vals.get('head_yaw_joint',0.0),vals.get('head_pitch_joint',0.0)]
 def on_odom(self,m):self.odom.append((float(m.pose.pose.position.x),float(m.pose.pose.position.y)))
 def on_imu(self,m):
  q=m.orientation;p=math.asin(max(-1.,min(1.,2*(q.w*q.y-q.z*q.x))));r=math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y));self.tilt.append(max(abs(p),abs(r)))
 def spin(self,s):
  end=time.monotonic()+s
  while rclpy.ok() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.01)
 def publish(self,arm,s):
  end=time.monotonic()+s;period=.02;next_tick=time.monotonic()
  while rclpy.ok() and time.monotonic()<end:
   m=UpperBodyCommandArray();m.header=MessageHeader();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='mc_upper_body';m.header.sequence=self.seq;self.seq+=1;m.source=SOURCE;m.hand_sub_mode=0;m.hand_pos=[];m.head_pos=list(self.head);m.arm_pos=[float(v) for v in arm];self.pub.publish(m);rclpy.spin_once(self,timeout_sec=.001);next_tick+=period;time.sleep(max(0,next_tick-time.monotonic()))
 def call(self,c,rq):
  if not c.wait_for_service(timeout_sec=5):raise RuntimeError('service unavailable')
  f=c.call_async(rq);end=time.monotonic()+5
  while rclpy.ok() and not f.done() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=.01)
  if not f.done() or f.result() is None:raise RuntimeError('service timeout')
  return f.result()
 def source_action(self,a):
  rq=SetMcInputSource.Request();rq.request.header.stamp=self.get_clock().now().to_msg();rq.action=McInputAction();rq.action.value=a;rq.input_source.name=SOURCE;rq.input_source.priority=PRIORITY;rq.input_source.timeout=TIMEOUT_MS;return self.call(self.setsrc,rq)
 def set_mode(self,name):
  rq=SetMcAction.Request();rq.header=RequestHeader();rq.header.stamp=self.get_clock().now().to_msg();rq.source='robotsim_urs_truth_pregrasp';rq.command=McActionCommand();rq.command.action_desc=name;return self.call(self.mode,rq)
 def current_source(self):
  rq=GetCurrentInputSource.Request();rq.request.header.stamp=self.get_clock().now().to_msg();return self.call(self.getsrc,rq)
rclpy.init();n=Plan();registered=False
try:
 n.spin(2.0)
 if any(x not in n.arm for x in ARM):raise RuntimeError('missing arm state')
 start=[n.arm[x] for x in ARM];target=list(start);target[7:]=TARGET
 if LEFT_TARGET is not None:target[:7]=LEFT_TARGET
 if max(abs(a-b) for a,b in zip(start,target))>1.1:raise RuntimeError('target delta exceeds 1.1 rad')
 n.publish(start,1.0)
 rsp=n.source_action(McInputAction.INPUTACTION_ADD)
 if int(rsp.response.header.code)!=0:
  rsp=n.source_action(McInputAction.INPUTACTION_MODIFY)
  if int(rsp.response.header.code)!=0:raise RuntimeError('input source add/modify failed')
  rsp=n.source_action(McInputAction.INPUTACTION_ENABLE)
  if int(rsp.response.header.code)!=0:raise RuntimeError('input source enable failed')
 registered=True;n.publish(start,1.0)
 n.set_mode(ACTION);n.publish(start,1.0)
 if n.pub.get_subscription_count()<1:raise RuntimeError('upper body topic has no subscriber after action switch:'+ACTION)
 src=n.current_source();source_name=str(src.input_source.name)
 if source_name!=SOURCE:raise RuntimeError('URS source not selected:'+source_name)
 for phase in range(max(1,int(DURATION/.02))):
  u=(phase+1)/max(1,int(DURATION/.02));u=3*u*u-2*u*u*u;n.publish([a+u*(b-a) for a,b in zip(start,target)],.02)
 n.publish(target,HOLD);reached=[n.arm[x] for x in RIGHT];reached_left=[n.arm[x] for x in ARM[:7]]
 for phase in range(max(1,int(DURATION/.02))):
  u=(phase+1)/max(1,int(DURATION/.02));u=3*u*u-2*u*u*u;n.publish([a+u*(b-a) for a,b in zip(target,start)],.02)
 n.publish(start,.5);restored=[n.arm[x] for x in RIGHT]
 n.set_mode('STAND_DEFAULT');n.publish(start,.5)
 try:n.source_action(McInputAction.INPUTACTION_DISABLE)
 except Exception:pass
 registered=False
 drift=math.hypot(n.odom[-1][0]-n.odom[0][0],n.odom[-1][1]-n.odom[0][1]) if len(n.odom)>=2 else None
 stable=len(n.odom)>=100 and drift is not None and drift<=.05 and n.tilt and max(n.tilt)<=math.radians(15)
 terr=max(abs(a-b) for a,b in zip(reached,TARGET));lerr=0.0 if LEFT_TARGET is None else max(abs(a-b) for a,b in zip(reached_left,LEFT_TARGET));rerr=max(abs(a-b) for a,b in zip(restored,start[7:]))
 print(json.dumps({'source_name':source_name,'subscriber_count':n.pub.get_subscription_count(),'start_right_arm_rad':start[7:],'target_right_arm_rad':TARGET,'target_left_arm_rad':LEFT_TARGET,'reached_right_arm_rad':reached,'reached_left_arm_rad':reached_left,'restored_right_arm_rad':restored,'max_target_error_rad':terr,'max_left_target_error_rad':lerr,'max_restore_error_rad':rerr,'response_pass':terr<=.08 and lerr<=.08 and rerr<=.04,'motion_stability':{'sample_count':len(n.odom),'planar_drift_m':drift,'max_tilt_rad':max(n.tilt) if n.tilt else None,'pass':stable}}))
finally:
 if registered:
  try:n.source_action(McInputAction.INPUTACTION_DISABLE)
  except Exception:pass
 n.destroy_node();rclpy.shutdown()
""".replace("__TARGET__", serialized).replace("__LEFT_TARGET__", serialized_left).replace("__DURATION__", f"{duration_s:.17g}").replace("__HOLD__", f"{hold_s:.17g}").replace("__ACTION__", serialized_action)


def headless_x11_mounts(root, libs, xkb, *, interactive):
    """A visible WSLg session must not depend on disposable headless files."""
    if interactive:return []
    for label,path in (("xvfb root",root),("xvfb libraries",libs),("XKB root",xkb)):
        if path is None or not path.is_absolute() or not path.is_dir() or path.is_symlink():raise ValueError(label+" is missing or unsafe")
    return ["--mount",f"type=bind,source={root},target=/opt/xvfb,readonly",
            "--mount",f"type=bind,source={libs},target=/tmp/xvfb-libs,readonly",
            "--mount",f"type=bind,source={root / 'usr/bin/xkbcomp'},target=/usr/bin/xkbcomp,readonly",
            "--mount",f"type=bind,source={xkb},target=/usr/share/X11/xkb,readonly"]


def activate_wslg_native_window(window, output, *, label):
    """Narrow verified Windows activation; X11 identity/focus gates still apply."""
    powershell=Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    helper=Path(__file__).with_name("wslg_mujoco_foreground.ps1")
    if not powershell.is_file() or not helper.is_file():raise RuntimeError("WSLg foreground helper unavailable")
    winpath=run(["wslpath","-w",str(helper)],timeout_s=3.).stdout.strip()
    attempt=run([str(powershell),"-NoProfile","-ExecutionPolicy","Bypass","-WindowStyle","Hidden","-File",winpath,"-ExpectedTitle",window["title"]],timeout_s=6.,check=False)
    pg.write_new_text(output/(label+".stdout"),attempt.stdout)
    if attempt.returncode:raise RuntimeError("WSLG_WINDOWS_FOREGROUND_NOT_VERIFIED:"+attempt.stdout[-1200:])
    result=json.loads(attempt.stdout.strip().splitlines()[-1])
    if result.get("status")!="FOREGROUND_VERIFIED" or result.get("keyboardEvents")!=0 or result.get("mouseEvents")!=0:raise RuntimeError("WSLg foreground evidence invalid")
    return result


def decode_arm_execution_output(returncode, stdout):
    """Preserve execution summary when an exception-context JSON follows it."""
    decoded={}
    for line in stdout.splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if not isinstance(row,dict):continue
        decoded.update(row)
        if 'native_tcp_arrival_after_residual_feedforward' in row:
            decoded['native_tcp_arrival']=row['native_tcp_arrival_after_residual_feedforward']
        if 'native_tcp_arrival_after_refinement' in row:
            decoded['native_tcp_arrival']=row['native_tcp_arrival_after_refinement']
        if 'prelift_bilateral_contact' in row:decoded['physical_bilateral_contact']=row['prelift_bilateral_contact']
    decoded['child_returncode']=returncode
    if returncode:decoded['stdout_tail']=stdout[-1600:]
    return decoded


class MicroProbeComplete(Exception):
    """End only the isolated probe; mandatory finally still tears down."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-overlay-result", required=True, type=Path)
    parser.add_argument("--table-overlay-result-sha256", required=True)
    parser.add_argument("--claw-control-overlay-result", type=Path, help="hash-pinned SIMULATION_ONLY claw actuator overlay derived from this table scene")
    parser.add_argument("--claw-control-overlay-result-sha256")
    parser.add_argument("--reset-helper", required=True, type=Path)
    parser.add_argument("--reset-helper-sha256", required=True)
    parser.add_argument("--xvfb-root", type=Path)
    parser.add_argument("--xvfb-libs", type=Path)
    parser.add_argument("--xkb-root", type=Path)
    parser.add_argument("--interactive-x11-display", help="use a user-visible WSLg X11 display and pause after failed claw response; requires the host X socket mounted explicitly")
    parser.add_argument("--verified-native-mouse-controls",action="store_true",help="explicit pixel/ancestry-verified mouse-only native Key/Pause controls; no keyboard focus assumption")
    parser.add_argument("--native-reset-input",choices=("backspace","button"),default="backspace",help="one verified native Reset input, without fallback/retry")
    parser.add_argument("--native-render-adapter",help="process-scoped WSLg GPU selection; verified against the actual native GL context before MC startup")
    parser.add_argument("--native-arm-policy-relay",action="store_true",help="preserve actual MC arm commands during standing/navigation; exclusive bounded grasp handoff afterward")
    parser.add_argument("--open-hand-transit-park",action="store_true",help="source-limited right shoulder parked at 0.8rad to keep open hand away from bottle during walking")
    parser.add_argument("--native-circle-tracking",action="store_true",help="consume fresh source-limited native circle IK updates during arm execution")
    parser.add_argument("--physical-lift-test",action="store_true",help="60mm/3s lift and 1.5s observation only after measured bilateral contact and static lift sweep")
    parser.add_argument("--native-dock-truth-feedback",action="store_true",help="SIMULATION_ONLY exact native pose feedback; not estimator/navigation qualification")
    parser.add_argument("--dock-angular-speed-cap",type=float,help="explicit <=0.075 rad/s external steering cap; bridge promotion also bounded")
    parser.add_argument("--dock-command-speed-cap",type=float,help="explicit smaller external bridge speed cap; MC/policy/model unchanged")
    parser.add_argument("--dock-braking-delay-s",type=float,help="explicit measured stop-tail predictor; final actual position/stop criteria remain unchanged")
    parser.add_argument("--fixed-reviewed-checkpoint",action="store_true",help="native keyframe episode initialization; no navigation qualification")
    parser.add_argument("--startup-opening-diagnostic", action="store_true", help="8s no-locomotion native opening observation, then mandatory teardown")
    parser.add_argument("--initial-hands-fully-open", action="store_true", help="hold both hands open from bootstrap; require actual native opening before walking")
    parser.add_argument("--pause-on-claw-failure", action="store_true", help="retain the simulator/container at the failed closure stage for manual inspection")
    parser.add_argument("--arm-ik-result", type=Path)
    parser.add_argument("--arm-ik-result-sha256")
    parser.add_argument("--arm-plan-duration-s", type=float, default=6.0)
    parser.add_argument("--arm-plan-hold-s", type=float, default=1.0)
    parser.add_argument("--urs-arm-ik-result", type=Path)
    parser.add_argument("--urs-arm-ik-result-sha256")
    parser.add_argument("--urs-grasp-candidate", type=Path, help="passing static STL candidate whose arm targets are executed through the documented upper-body remote source")
    parser.add_argument("--urs-grasp-candidate-sha256")
    parser.add_argument("--urs-action-name", default="UPPERBODY_REMOTE_SPLIT", choices=("UPPERBODY_REMOTE_SPLIT", "WHOLE_BODY_TELEOP", "VR_REMOTE_CONTROLLER"), help="official upper-body action used for the bounded arm command validation")
    parser.add_argument("--run-hal-arm-pd-probe", action="store_true", help="run one source-gain 0.03-rad direct HAL arm round trip after native docking")
    parser.add_argument("--hal-grasp-candidate", type=Path, help="passing open-gripper static candidate executed through the verified direct HAL arm channel")
    parser.add_argument("--hal-grasp-candidate-sha256")
    parser.add_argument("--hal-arm-candidate", type=Path, help="passing static candidate executed through the exclusive SIMULATION_ONLY arm HAL overlay while MC retains waist and leg ownership")
    parser.add_argument("--hal-arm-candidate-sha256")
    parser.add_argument("--hal-arm-plan-mode", choices=("round-trip", "pregrasp-hold"), default="round-trip", help="whether the bounded arm-only plan restores neutral or holds the verified pregrasp for the next contact stage")
    parser.add_argument("--hal-right-hand-command", type=float, help="optional normalized 0..1 right STL hand command sent only while the arm plan holds its verified pregrasp")
    parser.add_argument("--hal-hand-close-hold-s", type=float, default=2.0, help="bounded right-hand close hold when --hal-right-hand-command is used")
    parser.add_argument("--allow-concurrent-hal-upperbody-probe", action="store_true", help="SIMULATION_ONLY: retain MC arm/waist publishers and measure whether one bounded direct HAL upper-body plan is accepted or overridden")
    parser.add_argument("--run-hal-waist-pd-probe", action="store_true", help="after verified Reset→Stand, perform one 0.03-rad source-gain waist-roll round trip and restore")
    parser.add_argument("--disable-mc-arm-publisher", action="store_true")
    parser.add_argument("--disable-mc-hand-publisher", action="store_true", help="SIMULATION_ONLY: let the bounded external STL hand command own only hand_joint_command while MC retains legs and waist")
    parser.add_argument("--coordinated-circle-waist", action="store_true", help="SIMULATION_ONLY measured waist hold and source-gain waist/arm pregrasp; requires exclusive ownership")
    parser.add_argument("--disable-mc-waist-publisher", action="store_true", help="SIMULATION_ONLY: let the direct waist HAL probe own only waist_joint_command; leg standing remains MC-controlled")
    parser.add_argument("--dock-world-xy", type=float, nargs=2, metavar=("X", "Y"), help="run one bounded official velocity-control route to this source-map table-docking point")
    parser.add_argument("--dock-yaw-rad", type=float, help="required final source-map yaw for --dock-world-xy")
    parser.add_argument("--dock-position-only", action="store_true", help="accept a table-dock position with the measured terminal yaw; required when downstream IK is re-solved from that measured pose")
    parser.add_argument("--refine-world-xy", type=float, nargs=2, metavar=("X", "Y"), help="optional measured local base-refinement point after table docking")
    parser.add_argument("--refine-yaw-rad", type=float, help="required final source-map yaw for --refine-world-xy")
    parser.add_argument("--refine-position-only", action="store_true", help="for a local refinement, accept measured lateral/forward placement without a forced in-place yaw correction")
    parser.add_argument("--refine-position-tolerance-m", type=float, default=0.15, help="explicit local base-position acceptance radius for position-only refinement")
    parser.add_argument("--dock-position-tolerance-m", type=float, default=0.18, help="primary table-dock acceptance radius; use a smaller value only for manipulation alignment")
    parser.add_argument("--dock-timeout-s", type=float, default=30.0, help="bounded route deadline; used only with --dock-world-xy")
    parser.add_argument("--use-official-cmd-vel-bridge", action="store_true", help="route through the pinned RAICOM cmd_vel_bridge instead of publishing the MC velocity type directly")
    parser.add_argument("--micro-base-probe", choices=("forward","backward","left","right"), help="one native body-frame micro pulse and strict composite stop; no arm approach")
    parser.add_argument("--micro-probe-duration-s",type=float,default=.6)
    parser.add_argument("--micro-command-speed",type=float,default=.2)
    parser.add_argument("--lateral-probe-mps", type=float, help="one bounded lateral cmd_vel probe in body-frame m/s; requires the official bridge")
    parser.add_argument("--lateral-probe-duration-s", type=float, default=0.60, help="duration for --lateral-probe-mps; no hand or arm command is sent")
    parser.add_argument("--inspect-vr-topic", action="store_true", help="read-only discovery of the delivered VR teleoperation topic type; does not change the MC action")
    parser.add_argument("--inspect-retargeting-topic", action="store_true", help="read-only discovery of the delivered MC joint-retargeting topic type; does not change the MC action")
    parser.add_argument("--run-retargeting-neutral-probe", action="store_true", help="SIMULATION_ONLY: switch to installed WHOLE_BODY_TELEOP and publish only the measured standing joint targets through McRetargetFrame")
    parser.add_argument("--run-vr-zero-probe", action="store_true", help="SIMULATION_ONLY: network-isolated official VRData ABI probe with all axes zero; no arm, hand, or object command")
    parser.add_argument("--run-vr-right-controller-x-probe", action="store_true", help="SIMULATION_ONLY: one bounded +10 mm right VR-controller x probe; no joystick, hand, or object command")
    parser.add_argument("--preset-motion-id", type=int, help="one built-in animation-player motion ID used only to verify the delivered native upper-body service")
    parser.add_argument("--preset-motion-area", type=int, default=2, help="official preset-motion control-area bitmask; the default is the right hand")
    args = parser.parse_args()

    if args.verified_native_mouse_controls and not (args.fixed_reviewed_checkpoint and args.interactive_x11_display and args.pause_on_claw_failure):
        raise ValueError("verified mouse controls are scoped to the visible fixed-stance human-review entry")
    if args.native_arm_policy_relay:
        from scripts.research.issue46.native_runtime.native_micro_stop_contract import verify_mc_fsm_runtime_contract
        verify_mc_fsm_runtime_contract(RUNTIME/"mc/lib")
    if args.native_arm_policy_relay and (not args.disable_mc_arm_publisher or not args.disable_mc_hand_publisher or args.hal_arm_candidate is None or args.fixed_reviewed_checkpoint or args.open_hand_transit_park or args.coordinated_circle_waist):
        raise ValueError("native arm relay requires the unparked arm/hand split path; no fixed checkpoint or waist posture candidate")
    if args.dock_angular_speed_cap is not None and (args.dock_command_speed_cap is None or not 0.<args.dock_angular_speed_cap<=.075):
        raise ValueError("angular cap requires bounded dock speed and <=.075 rad/s")
    if args.dock_command_speed_cap is not None and (args.dock_world_xy is None or not args.native_dock_truth_feedback or not .12<=args.dock_command_speed_cap<=.20):
        raise ValueError("bounded dock cap requires native feedback and .12..0.20 m/s")

    if args.micro_base_probe:
        if args.dock_world_xy is not None or args.fixed_reviewed_checkpoint or args.lateral_probe_mps is not None or args.physical_lift_test or args.pause_on_claw_failure:
            raise ValueError("micro capability probe is isolated from dock, checkpoint and grasp")
        if not args.use_official_cmd_vel_bridge or not args.native_dock_truth_feedback or not args.initial_hands_fully_open or not .4<=args.micro_probe_duration_s<=1. or not .12<=args.micro_command_speed<=.20:
            raise ValueError("micro probe requires official bridge, native feedback, open hands and .4..1s")
        from scripts.research.issue46.native_runtime.native_micro_stop_contract import verify_mc_fsm_runtime_contract
        verify_mc_fsm_runtime_contract(RUNTIME/"mc/lib")

    if args.dock_braking_delay_s is not None and (not 0.<args.dock_braking_delay_s<=1. or not args.native_dock_truth_feedback or not args.dock_position_only):
        raise ValueError('measured braking requires a bounded native position-only diagnostic')

    if (args.claw_control_overlay_result is None) != (args.claw_control_overlay_result_sha256 is None):
        raise ValueError("claw control overlay result and SHA-256 must be supplied together")

    if (args.arm_ik_result is None) != (args.arm_ik_result_sha256 is None):
        raise ValueError("arm IK result and its SHA-256 must be supplied together")
    if args.arm_ik_result is not None and (not 4.0 <= args.arm_plan_duration_s <= 12.0 or not 0.5 <= args.arm_plan_hold_s <= 2.0):
        raise ValueError("arm plan duration/hold is outside the bounded range")
    if (args.urs_arm_ik_result is None) != (args.urs_arm_ik_result_sha256 is None):
        raise ValueError("URS arm IK result and its SHA-256 must be supplied together")
    if (args.urs_grasp_candidate is None) != (args.urs_grasp_candidate_sha256 is None):
        raise ValueError("URS grasp candidate and its SHA-256 must be supplied together")
    if args.urs_arm_ik_result is not None and args.urs_grasp_candidate is not None:
        raise ValueError("choose either a native IK result or a static grasp candidate for the URS arm plan")
    if (args.hal_grasp_candidate is None) != (args.hal_grasp_candidate_sha256 is None):
        raise ValueError("HAL grasp candidate and its SHA-256 must be supplied together")
    if (args.hal_arm_candidate is None) != (args.hal_arm_candidate_sha256 is None):
        raise ValueError("HAL arm candidate and its SHA-256 must be supplied together")
    if args.hal_grasp_candidate is not None and args.hal_arm_candidate is not None:
        raise ValueError("choose either the arm-plus-waist HAL candidate or the arm-only HAL candidate")
    if args.hal_grasp_candidate is not None and args.run_hal_arm_pd_probe:
        raise ValueError("choose either the HAL arm probe or a HAL pregrasp candidate")
    if args.hal_arm_candidate is not None and args.run_hal_arm_pd_probe:
        raise ValueError("choose either the HAL arm probe or the arm-only HAL pregrasp candidate")
    if args.hal_arm_candidate is not None and not args.disable_mc_arm_publisher:
        raise ValueError("the arm-only HAL candidate requires exclusive arm publisher ownership")
    if args.coordinated_circle_waist and not (args.hal_arm_candidate is not None and args.disable_mc_arm_publisher and args.disable_mc_waist_publisher and args.disable_mc_hand_publisher):
        raise ValueError("coordinated circle waist requires exclusive arm/waist/hand publishers")
    if args.hal_arm_candidate is not None and args.disable_mc_waist_publisher and not args.coordinated_circle_waist:
        raise ValueError("the arm-only HAL candidate leaves waist ownership with MC")
    if args.hal_arm_candidate is not None and args.dock_world_xy is None and not args.micro_base_probe:
        raise ValueError("the arm-only HAL candidate requires a measured native table-docking route")
    if args.hal_right_hand_command is not None and args.hal_arm_candidate is None:
        raise ValueError("a right-hand command requires the bounded arm-only candidate path")
    if args.hal_right_hand_command is not None and not args.disable_mc_hand_publisher:
        raise ValueError("a bounded external hand command requires exclusive hand publisher ownership")
    if args.disable_mc_hand_publisher and args.hal_right_hand_command is None:
        raise ValueError("disabling MC hand publisher requires one bounded external hand command")
    if args.hal_right_hand_command is not None and args.hal_arm_plan_mode != "pregrasp-hold":
        raise ValueError("a right-hand command requires the continuous pregrasp-hold mode")
    if args.hal_right_hand_command is not None and not 0.0 <= args.hal_right_hand_command <= 1.0:
        raise ValueError("right hand command must use the source 0..1 range")
    if args.pause_on_claw_failure and (args.interactive_x11_display is None or args.hal_right_hand_command is None):
        raise ValueError("pause-on-claw-failure requires a visible X11 display and a bounded hand command")
    if args.hal_right_hand_command is not None and not 0.5 <= args.hal_hand_close_hold_s <= 5.0:
        raise ValueError("right-hand close hold must lie in 0.5..5 seconds")
    if args.disable_mc_arm_publisher and args.arm_ik_result is None and args.hal_grasp_candidate is None and args.hal_arm_candidate is None and not args.run_hal_arm_pd_probe:
        raise ValueError("disabling the MC arm publisher requires a bounded HAL arm plan or probe")
    if args.disable_mc_arm_publisher and args.disable_mc_waist_publisher and args.hal_grasp_candidate is None and not args.coordinated_circle_waist:
        raise ValueError("the combined arm/waist split requires a bounded HAL grasp candidate")
    if args.disable_mc_waist_publisher and args.hal_grasp_candidate is None and not args.run_hal_waist_pd_probe and not args.coordinated_circle_waist:
        raise ValueError("disabling the MC waist publisher requires a bounded waist plan or probe")
    if args.hal_grasp_candidate is not None and not (args.disable_mc_arm_publisher and args.disable_mc_waist_publisher) and not args.allow_concurrent_hal_upperbody_probe:
        raise ValueError("a HAL grasp candidate requires explicit arm and waist publisher ownership")
    if args.allow_concurrent_hal_upperbody_probe and args.hal_grasp_candidate is None:
        raise ValueError("concurrent HAL upper-body probe requires a grasp candidate")
    if args.urs_arm_ik_result is not None and (not 4.0 <= args.arm_plan_duration_s <= 12.0 or not 0.5 <= args.arm_plan_hold_s <= 2.0):
        raise ValueError("URS arm plan duration/hold is outside the bounded range")
    if args.urs_arm_ik_result is not None or args.urs_grasp_candidate is not None:
        installed_actions = yaml.safe_load(
            (RUNTIME / "mc/bin/mc_param/src/robot/lx2501_3_t2d5/action_setting.yaml").read_text(encoding="utf-8")
        )
        available_actions = installed_actions.get("transition_actions", {}) if isinstance(installed_actions, dict) else {}
        if args.urs_action_name not in available_actions:
            raise ValueError(
                f"installed MC does not provide requested upper-body action: {args.urs_action_name}"
            )
    if (args.dock_world_xy is None) != (args.dock_yaw_rad is None):
        raise ValueError("--dock-world-xy and --dock-yaw-rad must be supplied together")
    if (args.refine_world_xy is None) != (args.refine_yaw_rad is None):
        raise ValueError("--refine-world-xy and --refine-yaw-rad must be supplied together")
    if args.refine_world_xy is not None and args.dock_world_xy is None:
        raise ValueError("local refinement requires an initial table docking target")
    if args.refine_position_only and args.refine_world_xy is None:
        raise ValueError("position-only refinement requires --refine-world-xy")
    if not 0.05 <= args.refine_position_tolerance_m <= 0.20:
        raise ValueError("local refinement position tolerance must lie in 0.05..0.20 m")
    if not 0.05 <= args.dock_position_tolerance_m <= 0.18:
        raise ValueError("dock position tolerance must lie in 0.05..0.18 m")
    if args.dock_world_xy is not None and (not 8.0 <= args.dock_timeout_s <= 60.0):
        raise ValueError("dock timeout must lie in 8..60 seconds")
    if args.lateral_probe_mps is not None:
        if not 0.20 <= abs(args.lateral_probe_mps) <= 0.30:
            raise ValueError("lateral probe magnitude must be within the delivered 0.20..0.30 m/s command range")
        if not 0.40 <= args.lateral_probe_duration_s <= 1.0:
            raise ValueError("lateral probe duration must lie in 0.40..1.0 seconds")
        if args.dock_world_xy is not None:
            raise ValueError("run either one table-docking route or one isolated lateral probe, not both")
    if args.use_official_cmd_vel_bridge and args.dock_world_xy is None and args.lateral_probe_mps is None and not args.micro_base_probe:
        raise ValueError("the official cmd_vel bridge requires a bounded table docking route or lateral probe")
    if args.lateral_probe_mps is not None and not args.use_official_cmd_vel_bridge:
        raise ValueError("the lateral probe requires the pinned official cmd_vel bridge")
    if args.run_retargeting_neutral_probe and (args.dock_world_xy is not None or args.lateral_probe_mps is not None or args.urs_arm_ik_result is not None or args.urs_grasp_candidate is not None or args.hal_grasp_candidate is not None or args.hal_arm_candidate is not None or args.arm_ik_result is not None):
        raise ValueError("the retargeting neutral probe is isolated from docking and all arm/grasp commands")
    if args.run_vr_zero_probe and (args.run_retargeting_neutral_probe or args.dock_world_xy is not None or args.lateral_probe_mps is not None or args.urs_arm_ik_result is not None or args.urs_grasp_candidate is not None or args.hal_grasp_candidate is not None or args.hal_arm_candidate is not None or args.arm_ik_result is not None):
        raise ValueError("the VR zero probe is isolated from docking and all arm/grasp commands")
    if args.run_vr_right_controller_x_probe and (args.run_vr_zero_probe or args.run_retargeting_neutral_probe or args.dock_world_xy is not None or args.lateral_probe_mps is not None or args.urs_arm_ik_result is not None or args.urs_grasp_candidate is not None or args.hal_grasp_candidate is not None or args.hal_arm_candidate is not None or args.arm_ik_result is not None):
        raise ValueError("the right-controller probe is isolated from docking and all arm/grasp commands")
    if args.preset_motion_id is not None and args.preset_motion_id <= 0:
        raise ValueError("preset motion ID must be positive")
    if args.preset_motion_area not in (0, 1, 2, 3, 4, 8, 11):
        raise ValueError("preset motion area must be an official control-area value")

    if args.native_dock_truth_feedback and args.hal_arm_candidate is None:raise ValueError("native truth docking requires the observed arm-candidate path")
    if args.startup_opening_diagnostic and not args.initial_hands_fully_open:
        raise ValueError("opening diagnostic requires initial hands open")
    if args.initial_hands_fully_open and not (args.hal_arm_candidate is not None and args.disable_mc_arm_publisher and args.disable_mc_hand_publisher):
        raise ValueError("startup hand command requires exclusive arm/hand candidate path")
    table_result = require_hashed(args.table_overlay_result, args.table_overlay_result_sha256, label="table overlay result")
    reset_helper = require_hashed(args.reset_helper, args.reset_helper_sha256, label="native reset helper")
    xvfb_mount_options=headless_x11_mounts(args.xvfb_root,args.xvfb_libs,args.xkb_root,interactive=bool(args.interactive_x11_display))
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
    claw_control_overlay = None
    if args.claw_control_overlay_result is not None:
        control_path = require_hashed(args.claw_control_overlay_result, args.claw_control_overlay_result_sha256 or "", label="claw control overlay")
        claw_control_overlay = json.loads(control_path.read_text(encoding="utf-8"))
        if claw_control_overlay.get("classification") != "SIMULATION_ONLY_STL_CLAW_CONTROL_OVERLAY" or any(claw_control_overlay.get("gates", {}).get(name) != "PASS" for name in ("PARENT_HASHES", "ACTUATOR_SCOPE", "HOST_STATIC_COMPILE", "RUNTIME_STATIC_COMPILE")):
            raise ValueError("claw control overlay does not pass static preflight")
        if claw_control_overlay.get("parent_table_overlay", {}).get("sha256") != sha256(table_result):
            raise ValueError("claw control overlay is not derived from this exact table overlay")
        control_generated = claw_control_overlay.get("generated", {})
        for name in ("runtime_scene", "simulator_config"):
            inputs[name] = require_hashed(Path(str(control_generated.get(name, {}).get("path", ""))), str(control_generated.get(name, {}).get("sha256", "")), label=f"claw control {name}")
    checkpoint=(claw_control_overlay or {}).get("reviewed_checkpoint") if args.fixed_reviewed_checkpoint else None
    if args.fixed_reviewed_checkpoint and not isinstance(checkpoint,dict):raise ValueError("missing reviewed checkpoint provenance")
    bridge_source: Path | None = None
    bridge_revision: str | None = None
    if args.use_official_cmd_vel_bridge:
        if not OFFICIAL_CMD_VEL_BRIDGE.is_file() or OFFICIAL_CMD_VEL_BRIDGE.is_symlink():
            raise ValueError("pinned official cmd_vel bridge is unavailable")
        bridge_source = OFFICIAL_CMD_VEL_BRIDGE.resolve()
        bridge_revision = run(["git", "-C", str(bridge_source.parents[3]), "rev-parse", "HEAD"], timeout_s=5.0).stdout.strip()
        if args.micro_base_probe and (bridge_revision!="a97761820b398b6cab12b632ce9f313c394eeb7e" or sha256(bridge_source)!="9996c19790ef3de720c4d0f9fa2a8a75949f95a2d5800db50cfd3c7587562cc5"):
            raise ValueError("micro bridge provenance changed")

    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-xvfb-native-reset-stand").native_path
    import uuid
    pg.write_new_text(output/"invocation.json",json.dumps({"attempt_id":str(uuid.uuid4()),"worktree":str(REPO.resolve()),"head":run(["git","rev-parse","HEAD"],timeout_s=3.).stdout.strip(),"argv":sys.argv,"task_owned_worktree_may_include_reviewed_uncommitted_changes":True},indent=2)+"\n")
    # Native arrival and gravity feedforward both require live exact-engine data.
    # Build before container creation; every arm-candidate entry point owns this
    # dependency, including direct human-review invocations.
    native_observer = None
    if args.hal_arm_candidate is not None:
        from scripts.research.issue46.native_runtime import native_grasp_audit
        native_observer = native_grasp_audit.build(output, RUNTIME / "sim_mujoco/lib/libmujoco.so.3.3.7")
    container = "robotsim-i46-xvfb-" + output.name[-15:].lower().replace("_", "-")
    result: dict[str, object] = {
        "schema": "robotsim.issue46.xvfb-native-reset-stand.v1",
        "classification": "SIMULATION_ONLY_NATIVE_RESET_AND_STAND_WITH_FROZEN_STL_HANDS",
        "mj_step_calls": "NOT_MEASURED_BY_STARTUP_EVIDENCE",
        "table_overlay": {"path": str(args.table_overlay_result), "sha256": sha256(args.table_overlay_result)},
        "gates": {"SIMULATOR": "NOT_RUN", "HYBRID_ARM_BOOTSTRAP": "NOT_RUN", "MC": "NOT_RUN", "PREARM": "NOT_RUN", "NATIVE_RESET": "NOT_RUN", "POST_RESET_LEG_STATE": "NOT_RUN", "POST_RESET_STAND_STABILITY": "NOT_RUN", "MC_STAND_COMMAND_CAPTURE": "NOT_RUN", "VR_TELEOP_TOPIC_DISCOVERY": "NOT_RUN", "VR_ZERO_MODE": "NOT_RUN", "VR_ZERO_DELIVERY": "NOT_RUN", "VR_ZERO_RESTORE": "NOT_RUN", "VR_RIGHT_X_MODE": "NOT_RUN", "VR_RIGHT_X_DELIVERY": "NOT_RUN", "VR_RIGHT_X_RESPONSE": "NOT_RUN", "VR_RIGHT_X_RESTORE": "NOT_RUN", "RETARGETING_TOPIC_DISCOVERY": "NOT_RUN", "RETARGET_OVERLAY_BUILD": "NOT_RUN", "RETARGET_NEUTRAL_SEED": "NOT_RUN", "RETARGET_ACTION_SWITCH": "NOT_RUN", "RETARGET_NEUTRAL_DELIVERY": "NOT_RUN", "RETARGET_NEUTRAL_STABILITY": "NOT_RUN", "RETARGET_NEUTRAL_RESTORE": "NOT_RUN", "OFFICIAL_CMD_VEL_BRIDGE": "NOT_RUN", "NATIVE_LATERAL_SOURCE_REGISTRATION": "NOT_RUN", "NATIVE_LATERAL_DISPLACEMENT": "NOT_RUN", "NATIVE_LATERAL_STOP_HOLD": "NOT_RUN", "NATIVE_TABLE_DOCK_SOURCE_REGISTRATION": "NOT_RUN", "NATIVE_TABLE_DOCK_ARRIVAL": "NOT_RUN", "NATIVE_TABLE_DOCK_STOP_HOLD": "NOT_RUN", "NATIVE_TABLE_DOCK_STATE_HANDOFF": "NOT_RUN", "HAL_ARM_PD_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_ARM_PD_COMMAND_STAND_STABILITY": "NOT_RUN", "HAL_BILATERAL_PREGRASP": "NOT_RUN", "HAL_ARM_CANDIDATE_STATE_MATCH": "NOT_APPLICABLE", "HAL_ARM_RUNTIME_REPLAN": "NOT_RUN", "HAL_BILATERAL_ARM_PREGRASP": "NOT_RUN", "HAL_RIGHT_HAND_CLOSE": "NOT_RUN", "HAL_WAIST_PD_COMMAND_ROUNDTRIP": "NOT_RUN", "HAL_WAIST_PD_COMMAND_STAND_STABILITY": "NOT_RUN", "ARM_PLAN_ROUNDTRIP": "NOT_RUN", "ARM_PLAN_STAND_STABILITY": "NOT_RUN", "ARM_PLAN_RESTORE": "NOT_RUN", "URS_INPUT_SOURCE": "NOT_RUN", "URS_ARM_PLAN_ROUNDTRIP": "NOT_RUN", "URS_ARM_PLAN_STAND_STABILITY": "NOT_RUN", "URS_ARM_PLAN_RESTORE": "NOT_RUN"},
        "commands_not_sent": ["locomotion velocity", "arm trajectory", "hand command", "grasp"],
        "native_preset_motion": {"requested": args.preset_motion_id, "area": args.preset_motion_area, "built_in_only": True},
    }
    if bridge_source is not None:
        result["official_cmd_vel_bridge"] = {"path": str(bridge_source), "sha256": sha256(bridge_source), "revision": bridge_revision, "source": "AgiBot-Community/raicom2026-agibot-X2"}
    if claw_control_overlay is not None:
        result["claw_control_overlay"] = {"path": str(args.claw_control_overlay_result), "sha256": sha256(args.claw_control_overlay_result), "control_change": claw_control_overlay.get("control_change")}
        result['contact_representation']=claw_control_overlay.get('source_pad_collision_comparison',{'classification':'RAW_WHOLE_STL_CONVEX_COLLISION_BASELINE'})
    try:
        for source in inputs.values():
            pg.write_new_bytes(output / source.name, source.read_bytes())
        staged_dump = output / "mc_sim.yaml.dump"
        pg.write_new_bytes(staged_dump, (RUNTIME / MC_DUMP).read_bytes())
        arm_split_config = None
        arm_bootstrap_stdout = output / "native_arm_bootstrap_hold.stdout"
        arm_bootstrap_stop = output / "stop_arm_bootstrap"
        arm_bootstrap_enabled = False
        if args.disable_mc_arm_publisher and args.disable_mc_waist_publisher:
            mc_source = RUNTIME / stl_hand_startup.MC_SIM_CONFIG_RELATIVE
            arm_split_config = output / "mc_sim_arm_waist_split.yaml"
            split = stl_hand_startup.build_arm_waist_split_mc_config(yaml.safe_load(mc_source.read_text(encoding="utf-8")))
            if args.coordinated_circle_waist:
                split=stl_hand_startup.build_coordinated_waist_relay_config(yaml.safe_load(mc_source.read_text(encoding="utf-8")))
                bootstrap_gains=stl_hand_startup.load_animation_player_arm_pd_parameters()
                pg.write_new_text(output/"native_arm_bootstrap_hold.py",arm_bootstrap_hold_script(bootstrap_gains,relay_waist=True,initial_hand_opening=1.0 if args.initial_hands_fully_open else None,open_hand_transit_park=args.open_hand_transit_park,initial_arm_target=None if checkpoint is None else checkpoint["initial_arm_command_rad"],initial_arm_effort=None if checkpoint is None else checkpoint["initial_arm_effort_nm"]))
                arm_bootstrap_enabled=True
            pg.write_new_text(arm_split_config,yaml.safe_dump(split,allow_unicode=True,sort_keys=False))
        elif args.disable_mc_arm_publisher and args.disable_mc_hand_publisher:
            mc_source = RUNTIME / stl_hand_startup.MC_SIM_CONFIG_RELATIVE
            arm_split_config = output / "mc_sim_arm_hand_split.yaml"
            pg.write_new_text(arm_split_config, yaml.safe_dump((stl_hand_startup.build_native_arm_relay_config(yaml.safe_load(mc_source.read_text(encoding="utf-8"))) if args.native_arm_policy_relay else stl_hand_startup.build_arm_hand_split_mc_config(yaml.safe_load(mc_source.read_text(encoding="utf-8")))), allow_unicode=True, sort_keys=False))
            bootstrap_gains = stl_hand_startup.load_animation_player_arm_pd_parameters()
            pg.write_new_text(output / "native_arm_bootstrap_hold.py", arm_bootstrap_hold_script(bootstrap_gains,relay_arms=args.native_arm_policy_relay,initial_hand_opening=1.0 if args.initial_hands_fully_open else None,open_hand_transit_park=args.open_hand_transit_park,initial_arm_target=None if checkpoint is None else checkpoint["initial_arm_command_rad"],initial_arm_effort=None if checkpoint is None else checkpoint["initial_arm_effort_nm"]))
            arm_bootstrap_enabled = True
        elif args.disable_mc_arm_publisher:
            mc_source = RUNTIME / stl_hand_startup.MC_SIM_CONFIG_RELATIVE
            arm_split_config = output / "mc_sim_arm_split.yaml"
            pg.write_new_text(arm_split_config, yaml.safe_dump(stl_hand_startup.build_arm_split_mc_config(yaml.safe_load(mc_source.read_text(encoding="utf-8"))), allow_unicode=True, sort_keys=False))
            bootstrap_gains = stl_hand_startup.load_animation_player_arm_pd_parameters()
            pg.write_new_text(output / "native_arm_bootstrap_hold.py", arm_bootstrap_hold_script(bootstrap_gains,initial_hand_opening=1.0 if args.initial_hands_fully_open else None,open_hand_transit_park=args.open_hand_transit_park,initial_arm_target=None if checkpoint is None else checkpoint["initial_arm_command_rad"],initial_arm_effort=None if checkpoint is None else checkpoint["initial_arm_effort_nm"]))
            arm_bootstrap_enabled = True
        elif args.disable_mc_waist_publisher:
            mc_source = RUNTIME / stl_hand_startup.MC_SIM_CONFIG_RELATIVE
            arm_split_config = output / "mc_sim_waist_split.yaml"
            pg.write_new_text(arm_split_config, yaml.safe_dump(stl_hand_startup.build_waist_split_mc_config(yaml.safe_load(mc_source.read_text(encoding="utf-8"))), allow_unicode=True, sort_keys=False))
        staged_seed = output / "sim_seed"
        shutil.copytree(RUNTIME / MC_GENERATED, staged_seed)
        pg.write_new_bytes(output / "native_mujoco_reset.py", reset_helper.read_bytes())
        pg.write_new_text(output / "native_prearm.py", prearm_script())
        pg.write_new_text(output / "native_stand_sample.py", stl_hand_startup.build_stand_client_script())
        pg.write_new_text(output / "native_postdock_stand_observer.py", stl_hand_startup.build_stand_client_script(observe_only=True))
        pg.write_new_text(output / "native_arm_state_capture.py", stl_hand_startup.build_arm_state_capture_script())
        pg.write_new_text(output / "native_stand_command_capture.py", stl_hand_startup.build_stand_command_capture_script())
        if args.run_vr_zero_probe or args.run_vr_right_controller_x_probe:
            if not VR_DATA_PROBE_SOURCE.is_file() or VR_DATA_PROBE_SOURCE.is_symlink() or sha256(VR_DATA_PROBE_SOURCE) != VR_DATA_PROBE_SOURCE_SHA256:
                raise ValueError("pinned VRData reference probe is unavailable or hash-mismatched")
            mc_generator = RUNTIME / "mc/lib/libaimdk_msgs__rosidl_generator_c.so"
            if sha256(mc_generator) != VR_DATA_MC_GENERATOR_SHA256:
                raise ValueError("installed MC VRData ABI library does not match the pinned zero-input probe")
            vr_source = VR_DATA_PROBE_SOURCE.read_text(encoding="utf-8")
            if args.run_vr_right_controller_x_probe:
                marker = "    s->orientation.w = 1.0;"
                if vr_source.count(marker) != 1:
                    raise RuntimeError("pinned VRData probe source no longer has the verified controller initializer")
                vr_source = vr_source.replace(marker, marker + "\n    if (i == 1) s->position.x = 0.01;")
            pg.write_new_text(output / "native_vr_data_probe.c", vr_source)
            pg.write_new_text(output / "native_vr_mode.py", vr_mode_script())
            pg.write_new_text(output / "native_vr_build.sh", vr_compile_script())
        if args.run_retargeting_neutral_probe:
            retarget_overlay = output / "retarget_overlay" / "src" / "aimdk_msgs"
            retarget_overlay.mkdir(parents=True, exist_ok=False)
            (retarget_overlay / "msg").mkdir(exist_ok=False)
            for relative, text in retarget_overlay_files().items():
                pg.write_new_text(retarget_overlay / relative, text)
            pg.write_new_text(output / "native_retarget_mode.py", retarget_mode_script())
            pg.write_new_text(output / "native_retarget_neutral.py", retarget_neutral_script())
        if args.preset_motion_id is not None:
            pg.write_new_text(output / "native_preset_motion_probe.py", preset_motion_probe_script(args.preset_motion_id, args.preset_motion_area))
        if args.dock_world_xy is not None:
            robot_start = overlay.get("robot_start", {})
            start_xyz = robot_start.get("xyz_m")
            start_yaw = robot_start.get("yaw_rad")
            if not isinstance(start_xyz, list) or len(start_xyz) != 3 or not isinstance(start_yaw, (int, float)):
                raise ValueError("table overlay lacks a finite source-map robot start pose")
            pg.write_new_text(
                output / "native_locomotion_dock.py",
                locomotion_dock_script(
                    world_start_xy=[float(start_xyz[0]), float(start_xyz[1])],
                    world_start_yaw_rad=float(start_yaw),
                    dock_world_xy=[float(args.dock_world_xy[0]), float(args.dock_world_xy[1])],
                    dock_world_yaw_rad=float(args.dock_yaw_rad),
                    dock_position_only=bool(args.dock_position_only),
                    refine_world_xy=None if args.refine_world_xy is None else [float(args.refine_world_xy[0]), float(args.refine_world_xy[1])],
                    refine_world_yaw_rad=None if args.refine_yaw_rad is None else float(args.refine_yaw_rad),
                    refine_position_only=bool(args.refine_position_only),
                    dock_position_tolerance_m=float(args.dock_position_tolerance_m),
                    refine_position_tolerance_m=float(args.refine_position_tolerance_m),
                    timeout_s=float(args.dock_timeout_s),
                    use_cmd_vel_bridge=bool(args.use_official_cmd_vel_bridge),
                    native_pose_feedback=bool(args.native_dock_truth_feedback),
                    braking_delay_s=args.dock_braking_delay_s,
                    native_leg_limits=native_leg_safety_contract(include_upper_body=args.native_arm_policy_relay)[0] if args.native_dock_truth_feedback else None,
                    command_speed_cap=args.dock_command_speed_cap,
                    angular_speed_cap=args.dock_angular_speed_cap,
                    native_arm_relay=args.native_arm_policy_relay,
                ),
            )
        if args.micro_base_probe:
            from scripts.research.issue46.native_runtime.native_base_micro_probe import build_script
            from scripts.research.issue46.native_runtime import native_micro_stop_contract
            limits,provenance=native_leg_safety_contract(include_upper_body=args.native_arm_policy_relay)
            pg.write_new_text(output/"native_micro_probe.py",build_script(args.micro_base_probe,args.micro_probe_duration_s,limits,command_speed=args.micro_command_speed,native_arm_relay=args.native_arm_policy_relay))
            pg.write_new_bytes(output/"native_micro_stop_contract.py",Path(native_micro_stop_contract.__file__).read_bytes())
            result["micro_source_joint_limits"]=provenance
        if args.lateral_probe_mps is not None:
            pg.write_new_text(
                output / "native_lateral_probe.py",
                lateral_probe_script(float(args.lateral_probe_mps), float(args.lateral_probe_duration_s)),
            )
        if args.run_hal_waist_pd_probe:
            waist_gains = stl_hand_startup.load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_waist_pd_probe.py", stl_hand_startup.build_hal_waist_pd_probe_script(waist_gains))
            result["source_waist_pd_gains"] = {name: {"stiffness": waist_gains[name][0], "damping": waist_gains[name][1]} for name in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")}
        if args.run_hal_arm_pd_probe:
            arm_gains = stl_hand_startup.load_animation_player_arm_pd_parameters() if args.disable_mc_arm_publisher and not args.disable_mc_waist_publisher else stl_hand_startup.load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_arm_pd_probe.py", stl_hand_startup.build_hal_arm_pd_probe_script(arm_gains))
            result["source_arm_pd_gains"] = {name: {"stiffness": arm_gains[name][0], "damping": arm_gains[name][1]} for name in tuple(arm_gains) if name.startswith("right_")}
            result["arm_controller_profile"] = "ANIMATION_PLAYER_SOURCE_PD" if args.disable_mc_arm_publisher and not args.disable_mc_waist_publisher else "REMOTE_CONTROLLER_SOURCE_PD"
        hal_target_right = None
        hal_target_left = None
        hal_target_waist = None
        hal_pregrasp_marker = output / "hal_pregrasp_start.json"
        hal_pregrasp_stdout = output / "native_hal_bilateral_pregrasp.stdout"
        hal_pregrasp_background = False
        if args.hal_grasp_candidate is not None:
            candidate_path = require_hashed(args.hal_grasp_candidate, args.hal_grasp_candidate_sha256 or "", label="HAL grasp candidate")
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            required_candidate = ("SEARCH_CANDIDATE", "STANDING_WAIST_ROLL_RETAINED", "OPEN_PREGRASP_CLEAR")
            if candidate.get("classification") not in {"SIMULATION_ONLY_STATIC_TABLE_PREPLACED_TRUTH_GRASP_CANDIDATE", "SIMULATION_ONLY_STATIC_TABLE_PREPLACED_VISUAL_GRASP_CANDIDATE"} or any(candidate.get("gates", {}).get(name) != "PASS" for name in required_candidate):
                raise ValueError("HAL grasp candidate does not pass its static preflight")
            if candidate.get("scene", {}).get("sha256") != overlay.get("generated", {}).get("host_scene", {}).get("sha256"):
                raise ValueError("HAL grasp candidate is not for this table scene")
            state = candidate.get("initial_state")
            named = state.get("arm_positions_rad") if isinstance(state, dict) else None
            waist = state.get("waist_positions_rad") if isinstance(state, dict) else None
            order = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
            left_order = tuple("left_" + name[len("right_"):] for name in order)
            waist_order = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
            if not isinstance(named, dict) or not isinstance(waist, dict) or any(name not in named for name in order + left_order) or any(name not in waist for name in waist_order):
                raise ValueError("HAL grasp candidate lacks complete arm/waist target names")
            hal_target_right = [float(named[name]) for name in order]
            hal_target_left = [float(named[name]) for name in left_order]
            hal_target_waist = [float(waist[name]) for name in waist_order]
            arm_gains = stl_hand_startup.load_source_arm_pd_parameters()
            marker = "/workspace/evidence/hal_pregrasp_start.json" if args.disable_mc_arm_publisher and args.disable_mc_waist_publisher else None
            pg.write_new_text(output / "native_hal_bilateral_pregrasp.py", stl_hand_startup.build_hal_bilateral_upperbody_plan_script(arm_gains, hal_target_right, hal_target_waist, duration_s=args.arm_plan_duration_s, hold_s=args.arm_plan_hold_s, start_marker=marker))
            result["hal_grasp_candidate"] = {"path": str(args.hal_grasp_candidate), "sha256": sha256(args.hal_grasp_candidate)}
            result["hal_upperbody_command_ownership"] = "EXCLUSIVE_OVERLAY" if marker is not None else "CONCURRENT_PROBE_MC_RETAINS_OWNERSHIP"
        hal_arm_target_right = None
        hal_arm_target_left = None
        hal_arm_candidate_path = None
        if args.hal_arm_candidate is not None:
            candidate_path = require_hashed(args.hal_arm_candidate, args.hal_arm_candidate_sha256 or "", label="HAL arm candidate")
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            required_candidate = ("SEARCH_CANDIDATE", "STANDING_WAIST_ROLL_RETAINED", "OPEN_PREGRASP_CLEAR", "CLOSED_CONTACT_CLEAR")
            legacy_ok = candidate.get("classification") in {"SIMULATION_ONLY_STATIC_TABLE_PREPLACED_TRUTH_GRASP_CANDIDATE", "SIMULATION_ONLY_STATIC_TABLE_PREPLACED_VISUAL_GRASP_CANDIDATE"} and all(candidate.get("gates", {}).get(name) == "PASS" for name in required_candidate)
            approved_circle_ok = candidate.get("classification") == "SIMULATION_ONLY_APPROVED_CIRCLE_STANDING_IK" and candidate.get("gate") == "PASS"
            if not (legacy_ok or approved_circle_ok):
                raise ValueError("arm-only HAL candidate does not pass its static preflight")
            if not candidate_scene_matches_overlay(candidate,overlay,claw_control_overlay,args.claw_control_overlay_result):
                raise ValueError("arm-only HAL candidate is not for this table scene")
            hal_arm_candidate_path = candidate_path
            result["hal_arm_candidate"] = {"path": str(candidate_path), "sha256": sha256(candidate_path)}
            result["hal_arm_command_ownership"] = "MC_WAIST_RELAY_THEN_CIRCLE_OWNER_LEGS_REMAIN_MC" if args.coordinated_circle_waist else "EXCLUSIVE_ARM_OVERLAY_MC_RETAINS_WAIST_AND_LEGS"
        arm_target = None
        if args.arm_ik_result is not None:
            ik_path = require_hashed(args.arm_ik_result, args.arm_ik_result_sha256 or "", label="arm IK result")
            ik = json.loads(ik_path.read_text(encoding="utf-8"))
            if ik.get("classification") != "SIMULATION_ONLY_STATIC_NATIVE_STAND_TRUTH_IK" or any(ik.get("gates", {}).get(name) != "PASS" for name in ("TRUTH_IK_ALIGNMENT", "JOINT_LIMIT_MARGIN", "TABLE_CLEARANCE", "OPEN_PREGRASP_NONPAD_CLEAR")):
                raise ValueError("arm IK result does not pass the native truth preflight")
            named = ik.get("right_arm_q_rad")
            order = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
            if not isinstance(named, dict):
                raise ValueError("arm IK result lacks named right-arm target")
            arm_target = [float(named[name]) for name in order]
            gains = stl_hand_startup.load_source_arm_pd_parameters()
            pg.write_new_text(output / "native_hal_arm_plan.py", stl_hand_startup.build_hal_arm_plan_script(gains, arm_target, duration_s=args.arm_plan_duration_s, hold_s=args.arm_plan_hold_s))
        urs_target = None
        urs_left_target = None
        if args.urs_arm_ik_result is not None:
            urs_path = require_hashed(args.urs_arm_ik_result, args.urs_arm_ik_result_sha256 or "", label="URS arm IK result")
            urs = json.loads(urs_path.read_text(encoding="utf-8"))
            if urs.get("classification") != "SIMULATION_ONLY_STATIC_NATIVE_STAND_TRUTH_IK" or any(urs.get("gates", {}).get(name) != "PASS" for name in ("TRUTH_IK_ALIGNMENT", "JOINT_LIMIT_MARGIN", "TABLE_CLEARANCE", "OPEN_PREGRASP_NONPAD_CLEAR")):
                raise ValueError("URS arm IK result does not pass native truth preflight")
            named = urs.get("right_arm_q_rad")
            order = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
            if not isinstance(named, dict):
                raise ValueError("URS arm IK result lacks named right-arm target")
            urs_target = [float(named[name]) for name in order]
        if args.urs_grasp_candidate is not None:
            candidate_path = require_hashed(args.urs_grasp_candidate, args.urs_grasp_candidate_sha256 or "", label="URS grasp candidate")
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            required_candidate = ("SEARCH_CANDIDATE", "STANDING_WAIST_ROLL_RETAINED", "OPEN_PREGRASP_CLEAR")
            if candidate.get("classification") not in {"SIMULATION_ONLY_STATIC_TABLE_PREPLACED_TRUTH_GRASP_CANDIDATE", "SIMULATION_ONLY_STATIC_TABLE_PREPLACED_VISUAL_GRASP_CANDIDATE"} or any(candidate.get("gates", {}).get(name) != "PASS" for name in required_candidate):
                raise ValueError("URS grasp candidate does not pass its static preflight")
            if candidate.get("scene", {}).get("sha256") != overlay.get("generated", {}).get("host_scene", {}).get("sha256"):
                raise ValueError("URS grasp candidate is not for this table scene")
            state = candidate.get("initial_state")
            if not isinstance(state, dict) or not isinstance(state.get("arm_positions_rad"), dict):
                raise ValueError("URS grasp candidate lacks complete arm targets")
            named = state["arm_positions_rad"]
            order = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
            left_order = tuple("left_" + name[len("right_"):] for name in order)
            if any(name not in named for name in order + left_order):
                raise ValueError("URS grasp candidate lacks all arm target names")
            urs_target = [float(named[name]) for name in order]
            urs_left_target = [float(named[name]) for name in left_order]
            result["urs_grasp_candidate"] = {"path": str(args.urs_grasp_candidate), "sha256": sha256(args.urs_grasp_candidate)}
        if urs_target is not None:
            pg.write_new_text(output / "native_urs_arm_plan.py", urs_plan_script(urs_target, args.arm_plan_duration_s, args.arm_plan_hold_s, urs_left_target, args.urs_action_name))
        for script in ("native_mujoco_reset.py", "native_prearm.py", "native_stand_sample.py", "native_postdock_stand_observer.py", "native_arm_state_capture.py", "native_stand_command_capture.py", "native_locomotion_dock.py", "native_lateral_probe.py", "native_preset_motion_probe.py", "native_retarget_mode.py", "native_retarget_neutral.py", "native_vr_mode.py", "native_vr_build.sh", "native_arm_bootstrap_hold.py", "native_hal_bilateral_arm_pregrasp.py"):
            if not (output / script).is_file():
                continue
            (output / script).chmod(0o755)
        pg.write_new_text(output / "launch_sim.sh", stl_hand_startup.build_launch_script(inputs["simulator_config"].name))
        pg.write_new_text(output / "launch_mc.sh", stl_hand_startup.build_mc_launch_script())
        (output / "launch_sim.sh").chmod(0o755); (output / "launch_mc.sh").chmod(0o755)
        command = [
            "docker", "run", "-d", "--pull=never", "--name", container, "--network=none" if (args.run_vr_zero_probe or args.run_vr_right_controller_x_probe) else "--network=bridge", "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--tmpfs", "/agibot/data/var:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0770",
            "--tmpfs", "/workspace/raicom2026/" + MC_GENERATED + ":rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0755,size=8m",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=1000:1000",
            "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026,readonly",
            "--mount", f"type=bind,source={EVIDENCE_REPO},target=/workspace/robotsim,readonly",
            "--mount", f"type=bind,source={output},target=/workspace/evidence",
            *xvfb_mount_options,
            "--mount", f"type=bind,source={staged_dump},target=/workspace/raicom2026/{MC_DUMP}",
            "-e", "DISPLAY=" + (args.interactive_x11_display or ":99"), "-e", "LD_LIBRARY_PATH=" + ("" if args.interactive_x11_display else "/tmp/xvfb-libs"), "-e", "LIBGL_ALWAYS_SOFTWARE=1", "-e", "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe",
            "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=77" if (args.run_vr_zero_probe or args.run_vr_right_controller_x_probe) else "ROS_DOMAIN_ID=20", "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "sleep", "infinity",
        ]
        if args.interactive_x11_display:
            x_socket = Path("/tmp/.X11-unix")
            if not x_socket.is_dir() or not any(x_socket.glob("X*")):
                raise RuntimeError("visible X11 socket is unavailable; refusing hidden Xvfb fallback")
            command[command.index(IMAGE):command.index(IMAGE)] = ["--mount", f"type=bind,source={x_socket},target=/tmp/.X11-unix"]
        graphics_options=native_grasp_audit.graphics_mount_options(args.native_render_adapter,args.interactive_x11_display)
        if graphics_options:
            command[command.index(IMAGE):command.index(IMAGE)]=graphics_options
            result['native_graphics_runtime']={'requested_adapter':args.native_render_adapter,'device':'/dev/dxg','driver_libraries':'/usr/lib/wsl (read-only)','native_process_only_environment':True,'headless_software_default_unchanged':True}
        if args.run_vr_zero_probe or args.run_vr_right_controller_x_probe:
            command[command.index(IMAGE):command.index(IMAGE)] = ["-e", "X2_SIM_ONLY=1", "-e", "ROS_LOCALHOST_ONLY=1"]
        if arm_split_config is not None:
            command[command.index(IMAGE):command.index(IMAGE)] = ["--mount", f"type=bind,source={arm_split_config},target=/workspace/raicom2026/{stl_hand_startup.MC_SIM_CONFIG_RELATIVE},readonly"]
            result["mc_command_split_overlay"] = {"path": str(arm_split_config), "sha256": sha256(arm_split_config), "disabled_publishers": (["arm_joint_command", "hand_joint_command"] if args.coordinated_circle_waist else ["arm_joint_command", "waist_joint_command"] if args.disable_mc_arm_publisher and args.disable_mc_waist_publisher else ["arm_joint_command", "hand_joint_command"] if args.disable_mc_arm_publisher and args.disable_mc_hand_publisher else ["arm_joint_command" if args.disable_mc_arm_publisher else "waist_joint_command"])}
            if args.native_arm_policy_relay:
                result["mc_command_split_overlay"]["disabled_publishers"]=["hand_joint_command"]
                result["mc_command_split_overlay"]["redirected_publishers"]={"arm_joint_command":"/robotsim/issue46/mc/arm/command"}

        pg.write_new_text(output / "docker_command.json", json.dumps(command, indent=2) + "\n")
        run(command, timeout_s=12.0)
        if not args.interactive_x11_display:
            run(["docker", "exec", "-d", container, "bash", "-lc", "/opt/xvfb/usr/bin/Xvfb :99 -screen 0 1706x1066x24 -ac -nolisten tcp > /workspace/evidence/xvfb.log 2>&1"], timeout_s=5.0)
        display_number=(args.interactive_x11_display or ":99").split(":")[-1].split(".")[0]
        if not display_number.isdigit():raise ValueError("expected a local numeric X11 display")
        run(["docker", "exec", container, "bash", "-lc", "test -S /tmp/.X11-unix/X" + display_number + " && cp /workspace/evidence/sim_seed/action_setting.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_setting.yaml; cp /workspace/evidence/sim_seed/action_ruler.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_ruler.yaml; cp /workspace/evidence/sim_seed/action_state.yaml /workspace/raicom2026/" + MC_GENERATED + "/action_state.yaml"], timeout_s=5.0)
        sim_launch = "bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"
        if native_observer is not None:
            sim_launch = native_grasp_audit.launch_prefix(args.native_render_adapter) + sim_launch
        sim_command = ["docker", "exec", "-d", container, "bash", "-lc", sim_launch]
        pg.write_new_text(output / "native_sim_launch.json", json.dumps({"command": sim_command, "native_observer_required": native_observer is not None}, indent=2) + "\n")
        run(sim_command, timeout_s=5.0)
        wait_for(output / "mujoco-console.log", "MujocoSimModule Started Succeeded", timeout_s=25.0)
        result["gates"]["SIMULATOR"] = "PASS"
        if native_observer is not None:
            wait_for(output / "native_physics.jsonl", '"host_monotonic_ns"', timeout_s=5.0)
            result["native_observer_readiness"] = native_grasp_audit.require_fresh_sample(output / "native_physics.jsonl")
            current=json.loads((output/'native_latest.json').read_text());actual_solver=current.get('solver_options',{})
            delta=(claw_control_overlay or {}).get('diagnostic_parameter_delta',{})
            if delta.get('numerical_solver') and actual_solver.get('noslip_iterations')!=2:raise RuntimeError('NATIVE_NOSLIP_OPTION_NOT_APPLIED')
            if delta.get('elliptic_solver') and (actual_solver.get('cone')!=1 or actual_solver.get('impratio')!=10.):raise RuntimeError('NATIVE_ELLIPTIC_OPTION_NOT_APPLIED')
            result['native_solver_options']=actual_solver
            result["gates"]["NATIVE_STATE_OBSERVER"] = "PASS"
            if args.native_render_adapter:
                wait_for(output/'native_graphics.json','"renderer"',timeout_s=3.)
                result['native_graphics']=native_grasp_audit.require_graphics_adapter(output/'native_graphics.json',args.native_render_adapter,expected_pid=current.get('process_pid'))
                result['gates']['NATIVE_RENDER_ADAPTER']='PASS'
        bootstrap_environment = "source /opt/ros/humble/setup.bash; source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash; "
        if arm_bootstrap_enabled:
            run(["docker", "exec", "-d", container, "bash", "-lc", bootstrap_environment + "python3 /workspace/evidence/native_arm_bootstrap_hold.py > /workspace/evidence/native_arm_bootstrap_hold.stdout 2>&1"], timeout_s=5.0)
            bootstrap_text = wait_for(arm_bootstrap_stdout, '"ready": true', timeout_s=12.0)
            bootstrap_line = bootstrap_text.strip().splitlines()[0]
            result["hybrid_arm_bootstrap"] = json.loads(bootstrap_line)
            if not result["hybrid_arm_bootstrap"].get("ready") or int(result["hybrid_arm_bootstrap"].get("subscription_count", 0)) < 1:
                raise RuntimeError("hybrid arm bootstrap did not acquire the simulator arm command subscriber")
            result["gates"]["HYBRID_ARM_BOOTSTRAP"] = "PASS"
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
        if args.interactive_x11_display and args.pause_on_claw_failure:
            result["windows_foreground_before_episode"]=activate_wslg_native_window(window,output,label="windows_foreground_before_episode")
        if args.fixed_reviewed_checkpoint:
            identity=" --expected-pid "+str(pid)+" --expected-title "+repr(str(window['title']))+" --window-id "+str(window['window_id'])
            capture_code="import sys;from pathlib import Path;sys.path.insert(0,'/workspace/evidence');import native_mujoco_reset as m;x=m.X11();assert x._target_identity("+str(pid)+","+repr(window['title'])+","+str(window['window_id'])+") is not None;x.capture_png("+str(window['window_id'])+",Path('/workspace/evidence/checkpoint_before_key.png'))"
            run(["docker","exec",container,"bash","-lc","python3 -c "+shlex.quote(capture_code)],timeout_s=8.)
            before_key=json.loads((output/'native_latest.json').read_text())
            selection=run(["docker","exec",container,"bash","-lc","python3 /workspace/evidence/native_mujoco_reset.py select-key-zero"+identity+(" --verified-pointer-only --native-ui-events /workspace/evidence/native_ui_events.jsonl" if args.verified_native_mouse_controls else "")+" --dispatch-event-json /workspace/evidence/keyframe_selection.json"],timeout_s=10.)
            pg.write_new_text(output/'keyframe_selection.stdout',selection.stdout)
            # MuJoCo 3.3.7 Simulation/Key already requests load_key when its
            # slider changes (simulate.cc, SECT_SIMULATION case 5). Wait for
            # that native effect before relocating the pointer or issuing a
            # second input. XSync acknowledges X11, not renderer consumption.
            result['reset']={'input':'NATIVE_KEY_SLIDER_AUTO_LOAD','dispatch':json.loads(selection.stdout),'load_key_button_sent':False}
            init_deadline=time.monotonic()+2.;init_match=False
            while time.monotonic()<init_deadline:
                observed=json.loads((output/'native_latest.json').read_text());pelvis=observed['bodies']['pelvis'];wanted=checkpoint['initial_pelvis_pose']
                error=math.dist(pelvis['position_m'],wanted['position_m']);dot=abs(sum(a*b for a,b in zip(pelvis['quaternion_wxyz'],wanted['quaternion_wxyz'])))
                angle=2*math.acos(max(-1.,min(1.,dot)))
                event=json.loads(selection.stdout)['event']
                boundary=native_grasp_audit.keyframe_initialization_evidence(output/'native_physics.jsonl',wanted,generation=before_key['generation']+1,after_ns=event['dispatch_monotonic_ns'])
                if observed['generation']==before_key['generation']+1 and boundary['pass']:
                    native_grasp_audit.require_fresh_sample(output/'native_latest.json')
                    init_match=True;break
                time.sleep(.02)
            if not init_match:
                run(["docker","exec",container,"bash","-lc","python3 -c "+shlex.quote(capture_code.replace('checkpoint_before_key.png','checkpoint_after_key.png'))],timeout_s=8.,check=False)
                result['checkpoint_initialization_failure']={'actual_pelvis_pose':pelvis,'wanted':wanted,'position_error_m':error,'angle_error_rad':angle,'generation':observed['generation'],'native_keyframe_count':observed.get('model_keyframe_count')}
                raise RuntimeError('NATIVE_KEYFRAME_INITIAL_POSE_UNCONFIRMED')
            result['reviewed_checkpoint']=checkpoint;result['native_checkpoint_observation']={**boundary,'current_position_error_m':error,'current_angle_error_rad':angle};result['gates']['NATIVE_EPISODE_KEYFRAME_INIT']='PASS'
        else:
            reset_before=json.loads((output/"native_latest.json").read_text()) if native_observer else None
            reset_command="click-native-reset" if args.native_reset_input=="button" else "send-reset"
            captures="" if args.native_reset_input=="button" else " --before-png /workspace/evidence/mujoco_before_reset.png --after-png /workspace/evidence/mujoco_after_reset.png"
            reset = run(["docker", "exec", container, "bash", "-lc", "python3 /workspace/evidence/native_mujoco_reset.py "+reset_command+" --expected-pid " + str(pid) + " --expected-title " + repr(str(window["title"])) + " --window-id " + str(window["window_id"]) + captures+" --dispatch-event-json /workspace/evidence/reset_dispatch_event.json"], timeout_s=12.0)
            pg.write_new_text(output / "reset_dispatch.stdout", reset.stdout)
            reset_data = json.loads(reset.stdout)
            result["reset"] = reset_data
            if reset_data.get("event", {}).get("status") != ("RESET_BUTTON_CLICKED" if args.native_reset_input=="button" else "KEY_DISPATCHED"):
                raise RuntimeError("native Reset was not dispatched")
            if native_observer:
                reset_deadline=time.monotonic()+1.
                while time.monotonic()<reset_deadline:
                    reset_after=json.loads((output/"native_latest.json").read_text())
                    if native_grasp_audit.native_reset_generation_effect(reset_before,reset_after):
                        result["native_reset_effect"]={"before_sim_time_s":reset_before["sim_time_s"],"after_sim_time_s":reset_after["sim_time_s"],"before_generation":reset_before["generation"],"after_generation":reset_after["generation"]};break
                    time.sleep(.01)
                else:raise RuntimeError("NATIVE_RESET_EFFECT_UNCONFIRMED")
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
        if args.startup_opening_diagnostic:
            # No route/source/arm plan may execute after this diagnostic branch.
            observed=[];deadline=time.monotonic()+8.;initial_xy=None
            while time.monotonic()<deadline:
                sample=native_grasp_audit.require_fresh_sample(output/'native_physics.jsonl')
                with (output/'native_physics.jsonl').open('rb') as f:
                    f.seek(0,2);f.seek(max(0,f.tell()-1048576));raw=f.read()
                lines=raw.splitlines()
                if not raw.endswith(bytes([10])):lines=lines[:-1]
                for line in reversed(lines):
                    try:row=json.loads(line);break
                    except (ValueError,UnicodeDecodeError):continue
                pelvis=row['bodies']['pelvis'];R=pelvis['rotation'];xyz=pelvis['position_m']
                initial_xy=initial_xy or xyz[:2]
                roll=math.atan2(R[7],R[8]);pitch=math.asin(max(-1.,min(1.,-R[6])))
                if not .50<=xyz[2]<=.82 or max(abs(roll),abs(pitch))>math.radians(15) or math.hypot(xyz[0]-initial_xy[0],xyz[1]-initial_xy[1])>.05:
                    raise RuntimeError('OPENING_DIAGNOSTIC_POSTURE')
                observed.append({'sim_time_s':row['sim_time_s'],'host_monotonic_ns':row['host_monotonic_ns'],'pelvis':pelvis,'joints':row['joints'],'actuators':row['claw_actuators']})
                time.sleep(.05)
            end=observed[-1]
            passed=all(abs(end['joints'][n]['q']+1.)<=.03 and abs(end['joints'][n]['qvel'])<=.05 for n in ('R_claw_joint',))
            result['startup_opening_diagnostic']={'classification':'RIGHT_HAND_CAUSAL_COMPARISON_LEFT_REFERENCE','duration_s':8.,'right_pass':passed,'samples':observed}
            result['gates']['STARTUP_RIGHT_OPENING_DIAGNOSTIC']='PASS' if passed else 'FAIL'
            return
        if args.initial_hands_fully_open:
            opening_deadline=time.monotonic()+8.;qualified_since=None
            while True:
                native_grasp_audit.require_fresh_sample(output / "native_physics.jsonl")
                data=(output / "native_physics.jsonl").read_bytes()[-1048576:]
                lines=data.splitlines()
                if not data.endswith(b"\n"):lines=lines[:-1]
                for line in reversed(lines):
                    try:hand_row=json.loads(line);break
                    except (ValueError,UnicodeDecodeError):continue
                opening={name:float(hand_row["joints"][name]["q"]) for name in ("L_claw_joint","R_claw_joint")}
                hand_ok=all(abs(q+1.0)<=.03 and abs(hand_row['joints'][name]['qvel'])<=.05 for name,q in opening.items())
                pelvis=hand_row['bodies']['pelvis'];R=pelvis['rotation']
                if not .5<=pelvis['position_m'][2]<=.82 or abs(math.asin(max(-1.,min(1.,-R[6]))))>math.radians(15) or abs(math.atan2(R[7],R[8]))>math.radians(15):raise RuntimeError('UNSAFE_POSTURE_DURING_STARTUP_OPENING')
                if hand_ok:
                    qualified_since=qualified_since or time.monotonic()
                    if time.monotonic()-qualified_since>=.3:break
                else:qualified_since=None
                if time.monotonic()>=opening_deadline:break
                time.sleep(.05)
            result["initial_hand_opening"]={"command":1.0,"actual_root_rad":opening,"pass":hand_ok and qualified_since is not None and time.monotonic()-qualified_since>=.3,"native_host_ns":hand_row["host_monotonic_ns"]}
            result["gates"]["INITIAL_HANDS_ACTUALLY_OPEN"]="PASS" if result['initial_hand_opening']['pass'] else "FAIL"
            if result['gates']['INITIAL_HANDS_ACTUALLY_OPEN']!='PASS':raise RuntimeError("INITIAL_HANDS_NOT_FULLY_OPEN; walking and grasp NOT_RUN")
        capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_arm_state_capture.py"], timeout_s=10.0, check=False)
        pg.write_new_text(output / "native_arm_state_capture.stdout", capture.stdout)
        result["native_arm_state_capture"] = json.loads(capture.stdout.strip().splitlines()[-1]) if capture.returncode == 0 else {"returncode": capture.returncode}
        command_capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_stand_command_capture.py"], timeout_s=10.0, check=False)
        pg.write_new_text(output / "native_stand_command_capture.stdout", command_capture.stdout)
        result["native_stand_command_capture"] = json.loads(command_capture.stdout.strip().splitlines()[-1]) if command_capture.returncode == 0 else {"returncode": command_capture.returncode}
        # Some delivered MC builds stop publishing the position-command topics
        # once STAND_DEFAULT has settled, while state/odometry continue.  A
        # missing post-settle command stream does not invalidate the directly
        # observed native standing gate; preserve it as an explicit evidence
        # gap for any later shadow-controller comparison.
        result["gates"]["MC_STAND_COMMAND_CAPTURE"] = "PASS" if command_capture.returncode == 0 else "NOT_AVAILABLE_AFTER_SETTLE"
        if args.run_vr_zero_probe or args.run_vr_right_controller_x_probe:
            compile_vr = run(["docker", "exec", container, "bash", "/workspace/evidence/native_vr_build.sh"], timeout_s=20.0, check=False)
            pg.write_new_text(output / "native_vr_compile.stdout", compile_vr.stdout)
            if compile_vr.returncode != 0:
                raise RuntimeError("pinned VRData probe did not compile: " + compile_vr.stdout[-1600:])
            vr_mode = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_vr_mode.py"], timeout_s=12.0, check=False)
            pg.write_new_text(output / "native_vr_mode.stdout", vr_mode.stdout)
            vr_mode_result = json.loads(vr_mode.stdout.strip().splitlines()[-1]) if vr_mode.returncode == 0 else {"returncode": vr_mode.returncode, "stdout_tail": vr_mode.stdout[-1200:]}
            result["native_vr_mode"] = {"returncode": vr_mode.returncode, "result": vr_mode_result}
            if vr_mode.returncode != 0 or not vr_mode_result.get("pass"):
                raise RuntimeError("VR_REMOTE_CONTROLLER was not accepted by the installed MC")
            result["gates"]["VR_ZERO_MODE" if args.run_vr_zero_probe else "VR_RIGHT_X_MODE"] = "PASS"
            vr_zero = run(["docker", "exec", container, "bash", "-lc", "env X2_SIM_ONLY=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77 LD_LIBRARY_PATH=/workspace/raicom2026/mc/lib:/opt/ros/humble/lib /workspace/evidence/native_vr_zero_probe --forward 0 --lateral 0 --yaw 0 --seconds 0.5 --lead 0"], timeout_s=8.0, check=False)
            pg.write_new_text(output / "native_vr_zero.stdout", vr_zero.stdout)
            result["native_vr_zero" if args.run_vr_zero_probe else "native_vr_right_x"] = {"reference_source": str(VR_DATA_PROBE_SOURCE), "reference_source_sha256": VR_DATA_PROBE_SOURCE_SHA256, "mc_generator_sha256": VR_DATA_MC_GENERATOR_SHA256, "right_controller_x_m": 0.0 if args.run_vr_zero_probe else 0.01, "returncode": vr_zero.returncode, "stdout": vr_zero.stdout}
            if vr_zero.returncode != 0:
                raise RuntimeError("VRData delivery failed: " + vr_zero.stdout[-1600:])
            result["gates"]["VR_ZERO_DELIVERY" if args.run_vr_zero_probe else "VR_RIGHT_X_DELIVERY"] = "PASS"
            probe_capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_arm_state_capture.py"], timeout_s=10.0, check=False)
            pg.write_new_text(output / "native_vr_probe_arm_capture.stdout", probe_capture.stdout)
            probe_state = json.loads(probe_capture.stdout.strip().splitlines()[-1]) if probe_capture.returncode == 0 else {"returncode": probe_capture.returncode, "stdout_tail": probe_capture.stdout[-1200:]}
            result["native_vr_probe_arm_capture"] = probe_state
            if args.run_vr_right_controller_x_probe:
                before_arm = result.get("native_arm_state_capture", {}).get("arm_positions_rad", {})
                after_arm = probe_state.get("arm_positions_rad", {}) if isinstance(probe_state, dict) else {}
                if not isinstance(before_arm, dict) or not isinstance(after_arm, dict) or any(name not in after_arm for name in before_arm):
                    raise RuntimeError("VR right-controller probe lacks complete arm-state readback")
                deltas = {name: abs(float(after_arm[name]) - float(value)) for name, value in before_arm.items()}
                max_delta = max(deltas.values(), default=float("inf"))
                result["native_vr_right_x_response"] = {"arm_delta_rad": deltas, "max_arm_delta_rad": max_delta}
                result["gates"]["VR_RIGHT_X_RESPONSE"] = "PASS" if 1e-4 <= max_delta <= 0.25 and probe_state.get("motion_stability", {}).get("pass") else "FAIL"
            restore = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_prearm.py"], timeout_s=12.0, check=False)
            pg.write_new_text(output / "native_vr_restore.stdout", restore.stdout)
            restore_result = json.loads(restore.stdout.strip().splitlines()[-1]) if restore.returncode == 0 else {"returncode": restore.returncode, "stdout_tail": restore.stdout[-1200:]}
            result["native_vr_restore"] = {"returncode": restore.returncode, "result": restore_result}
            if restore.returncode != 0 or restore_result.get("mode_name") != "STAND_DEFAULT":
                raise RuntimeError("VR zero probe did not restore STAND_DEFAULT")
            post_vr = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_stand_sample.py"], timeout_s=14.0, check=False)
            pg.write_new_text(output / "post_vr_stand_sample.stdout", post_vr.stdout)
            post_vr_result = json.loads(post_vr.stdout.strip().splitlines()[-1]) if post_vr.returncode == 0 else {"returncode": post_vr.returncode, "stdout_tail": post_vr.stdout[-1200:]}
            result["post_vr_stand"] = post_vr_result
            result["gates"]["VR_ZERO_RESTORE" if args.run_vr_zero_probe else "VR_RIGHT_X_RESTORE"] = "PASS" if post_vr.returncode == 0 and post_vr_result.get("stability", {}).get("pass") else "FAIL"
        if args.run_retargeting_neutral_probe:
            captured = result["native_arm_state_capture"]
            leg = captured.get("leg_positions_rad") if isinstance(captured, dict) else None
            waist = captured.get("waist_positions_rad") if isinstance(captured, dict) else None
            arm = captured.get("arm_positions_rad") if isinstance(captured, dict) else None
            joint_order = (
                "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
                "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
                "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
                "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
                "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                "left_wrist_roll_joint", "right_wrist_roll_joint",
            )
            groups = {**(leg or {}), **(waist or {}), **(arm or {})}
            if not isinstance(leg, dict) or not isinstance(waist, dict) or not isinstance(arm, dict) or any(name not in groups for name in joint_order):
                raise RuntimeError("native standing capture lacks the complete retargeting joint order")
            build = run(["docker", "exec", container, "bash", "-lc", "source /opt/ros/humble/setup.bash; cd /workspace/evidence/retarget_overlay; colcon --log-base /tmp/retarget-colcon-log build --merge-install --packages-select aimdk_msgs"], timeout_s=60.0, check=False)
            pg.write_new_text(output / "retarget_overlay_build.stdout", build.stdout)
            result["retarget_overlay"] = {"source": "installed sim_mujoco introspection", "build_returncode": build.returncode}
            if build.returncode != 0:
                raise RuntimeError("retargeting overlay build failed: " + build.stdout[-1600:])
            result["gates"]["RETARGET_OVERLAY_BUILD"] = "PASS"
            capture_frame = run(["docker", "exec", container, "bash", "-lc", "source /opt/ros/humble/setup.bash; source /workspace/evidence/retarget_overlay/install/setup.bash; timeout 5s python3 /workspace/evidence/native_retarget_neutral.py --capture-only --duration-s 0.60"], timeout_s=7.0, check=False)
            pg.write_new_text(output / "native_retarget_capture.stdout", capture_frame.stdout)
            captured_result = json.loads(capture_frame.stdout.strip().splitlines()[-1]) if capture_frame.returncode == 0 else {"returncode": capture_frame.returncode, "stdout_tail": capture_frame.stdout[-1200:]}
            captured_frame = captured_result.get("captured_frame") if isinstance(captured_result, dict) else None
            if capture_frame.returncode != 0 or not captured_result.get("pass") or not isinstance(captured_frame, dict):
                raise RuntimeError("could not capture one installed MC retarget frame")
            required_frame = ("joint_state", "root_position", "root_quarteniond", "left_hand_type", "right_hand_type")
            if any(name not in captured_frame for name in required_frame) or not all(isinstance(captured_frame[name], list) for name in required_frame[:3]):
                raise RuntimeError("captured retarget frame is incomplete")
            retarget_target = output / "retarget_neutral_target.json"
            pg.write_new_text(retarget_target, json.dumps(captured_frame, indent=2) + "\n")
            result["retarget_overlay"]["target"] = str(retarget_target)
            result["retarget_capture"] = {"returncode": capture_frame.returncode, "result": captured_result}
            seed = run(["docker", "exec", container, "bash", "-lc", "source /opt/ros/humble/setup.bash; source /workspace/evidence/retarget_overlay/install/setup.bash; timeout 5s python3 /workspace/evidence/native_retarget_neutral.py --target /workspace/evidence/retarget_neutral_target.json --duration-s 0.60"], timeout_s=7.0, check=False)
            pg.write_new_text(output / "native_retarget_seed.stdout", seed.stdout)
            seed_result = json.loads(seed.stdout.strip().splitlines()[-1]) if seed.returncode == 0 else {"returncode": seed.returncode, "stdout_tail": seed.stdout[-1200:]}
            result["native_retarget_seed"] = {"returncode": seed.returncode, "result": seed_result}
            if seed.returncode != 0 or not seed_result.get("pass"):
                raise RuntimeError("retargeting seed frame was not accepted while standing")
            result["gates"]["RETARGET_NEUTRAL_SEED"] = "PASS"
            mode = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_retarget_mode.py"], timeout_s=12.0, check=False)
            pg.write_new_text(output / "native_retarget_mode.stdout", mode.stdout)
            mode_result = json.loads(mode.stdout.strip().splitlines()[-1]) if mode.returncode == 0 else {"returncode": mode.returncode, "stdout_tail": mode.stdout[-1200:]}
            result["native_retarget_mode"] = {"returncode": mode.returncode, "result": mode_result}
            if mode.returncode != 0 or not mode_result.get("pass"):
                raise RuntimeError("WHOLE_BODY_TELEOP was not accepted by the installed MC")
            result["gates"]["RETARGET_ACTION_SWITCH"] = "PASS"
            neutral = run(["docker", "exec", container, "bash", "-lc", "source /opt/ros/humble/setup.bash; source /workspace/evidence/retarget_overlay/install/setup.bash; timeout 8s python3 /workspace/evidence/native_retarget_neutral.py --target /workspace/evidence/retarget_neutral_target.json --duration-s 1.50"], timeout_s=10.0, check=False)
            pg.write_new_text(output / "native_retarget_neutral.stdout", neutral.stdout)
            neutral_result = json.loads(neutral.stdout.strip().splitlines()[-1]) if neutral.returncode == 0 else {"returncode": neutral.returncode, "stdout_tail": neutral.stdout[-1200:]}
            result["native_retarget_neutral"] = {"returncode": neutral.returncode, "result": neutral_result}
            neutral_pass = neutral.returncode == 0 and neutral_result.get("pass") is True
            result["gates"]["RETARGET_NEUTRAL_DELIVERY"] = "PASS" if neutral_pass else "FAIL"
            result["gates"]["RETARGET_NEUTRAL_STABILITY"] = "PASS" if neutral_pass else "FAIL"
            restore = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 10s python3 /workspace/evidence/native_prearm.py"], timeout_s=12.0, check=False)
            pg.write_new_text(output / "native_retarget_restore.stdout", restore.stdout)
            restore_result = json.loads(restore.stdout.strip().splitlines()[-1]) if restore.returncode == 0 else {"returncode": restore.returncode, "stdout_tail": restore.stdout[-1200:]}
            result["native_retarget_restore"] = {"returncode": restore.returncode, "result": restore_result}
            result["gates"]["RETARGET_NEUTRAL_RESTORE"] = "PASS" if restore.returncode == 0 and restore_result.get("mode_name") == "STAND_DEFAULT" else "FAIL"
        if args.native_arm_policy_relay:
            arm_relay_source=output/"native_arm_relay_source_latest.json"
            wait_for(arm_relay_source,'"source_stamp_ns"',timeout_s=3.)
            from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
            from scripts.research.issue46.native_runtime.native_base_micro_probe import native_pose,physical_fault
            import mujoco,numpy as np,re
            scene=Path((claw_control_overlay or overlay)["generated"]["host_scene"]["path"])
            model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
            source_packet=json.loads(arm_relay_source.read_text());native_frame=json.loads((output/"native_latest.json").read_text())
            if not 0.<=(time.monotonic_ns()-source_packet["receipt_host_ns"])*1e-9<=.20 or not 0.<=(time.monotonic_ns()-native_frame["host_monotonic_ns"])*1e-9<=.20:raise RuntimeError("NATIVE_ARM_HANDOFF_INPUT_STALE")
            apply_recorded_native_frame(model,data,native_frame)
            start_qpos=data.qpos.copy();targets=source_packet["joints"]
            if len(targets)!=14:raise RuntimeError("MC_ARM_HANDOFF_INCOMPLETE")
            goal=start_qpos.copy()
            for name,item in targets.items():
                joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                if joint<0 or not model.jnt_limited[joint] or not model.jnt_range[joint,0]<=item["position"]<=model.jnt_range[joint,1] or item["stiffness"]<=0 or item["damping"]<=0:raise RuntimeError("MC_ARM_TARGET_OUTSIDE_SOURCE:"+name)
                addr=model.jnt_qposadr[joint];goal[addr]=item["position"]
                if abs(goal[addr]-start_qpos[addr])>1.1:raise RuntimeError("MC_ARM_HANDOFF_EXCURSION:"+name)
            from scripts.research.issue46.native_runtime.standing_side_grasp_static_search import contacts,native as static_native
            from scipy.spatial import ConvexHull
            feet=[]
            for co in data.contact:
                bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) for g in (co.geom1,co.geom2)]
                if any(model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in (co.geom1,co.geom2)) and any(n in ("left_ankle_roll_link","right_ankle_roll_link") for n in bodies):feet.append(co.pos[:2].copy())
            hull=ConvexHull(np.unique(np.asarray(feet),axis=0));root=model.body_rootid[static_native.bid(model,"pelvis")];robot=model.body_rootid==root
            def active_self_pairs():
                pairs={}
                for co in data.contact:
                    b1,b2=int(model.geom_bodyid[co.geom1]),int(model.geom_bodyid[co.geom2])
                    if co.efc_address<0 or co.dist>=-.001 or model.body_rootid[b1]!=root or model.body_rootid[b2]!=root:continue
                    pair=tuple(sorted((int(co.geom1),int(co.geom2))));pairs[pair]=min(float(co.dist),pairs.get(pair,0.))
                return pairs
            baseline_self=active_self_pairs()
            clear=[];com_margins=[]
            for u in np.linspace(0,1,21):
                data.qpos[:]=start_qpos+float(u)*(goal-start_qpos);mujoco.mj_forward(model,data)
                pads,bad,table,touch=contacts(model,data,static_native.bid(model,"test_cylinder"))
                if pads or bad or touch:raise RuntimeError("MC_ARM_HANDOFF_COLLISION:"+str((float(u),pads,bad,touch)))
                for pair,depth in active_self_pairs().items():
                    if depth<baseline_self.get(pair,-.001)-.0005:raise RuntimeError("MC_ARM_HANDOFF_NEW_SELF_PENETRATION:"+str((pair,depth)))
                com=(data.xipos[robot]*model.body_mass[robot,None]).sum(axis=0)/model.body_mass[robot].sum()
                margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
                if margin<0.:raise RuntimeError("MC_ARM_HANDOFF_STATIC_COM_OUTSIDE_SUPPORT")
                com_margins.append(margin);clear.append(float(u))
            log=(output/"mc-console.log").read_text(errors="replace")
            pids=set(re.findall(r"\|[ \t]*Process PID[ \t]*\|[ \t]*(\d+)[ \t]*\|",log))
            if len(pids)!=1:raise RuntimeError("MC_ARM_RELAY_PROCESS_IDENTITY")
            marker={"expected_mc_node":"mc_ros2_node"+pids.pop(),"native_generation":native_frame["generation"],"target_source_stamp_ns":source_packet["source_stamp_ns"],"source_fields":targets,"collision_clear_fractions":clear,"minimum_static_com_margin_m":min(com_margins),"physical_model_changed":False}
            pg.write_new_text(output/"native_arm_handoff_preflight.json",json.dumps(marker,indent=2)+"\n")
            pg.write_new_text(output/"enable_native_arm_relay",json.dumps(marker)+"\n")
            end=time.monotonic()+4.;limits,_=native_leg_safety_contract(include_upper_body=True);ready=False
            while time.monotonic()<end:
                bootstrap_tail=arm_bootstrap_stdout.read_text()[-1800:]
                if "Traceback" in bootstrap_tail:raise RuntimeError("NATIVE_ARM_RELAY_CHILD_FAILURE:"+bootstrap_tail)
                frame=json.loads((output/"native_latest.json").read_text())
                if not 0.<=(time.monotonic_ns()-frame["host_monotonic_ns"])*1e-9<=.20:raise RuntimeError("STALE_NATIVE_DURING_ARM_HANDOFF")
                fault=physical_fault(frame,native_pose(frame),limits)
                if fault:raise RuntimeError("ARM_POLICY_HANDOFF:"+fault)
                ready_path=output/"native_arm_relay_ready.json"
                if ready_path.exists() and json.loads(ready_path.read_text()).get("ready"):
                    current_source=json.loads(arm_relay_source.read_text())
                    if not 0.<=(time.monotonic_ns()-current_source["receipt_host_ns"])*1e-9<=.20:raise RuntimeError("STALE_MC_ARM_READINESS")
                    error=max(abs(frame["joints"][name]["q"]-item["position"]) for name,item in current_source["joints"].items())
                    if error<=.04:ready=True;break
                time.sleep(.02)
            if not ready:raise RuntimeError("NATIVE_ARM_POLICY_RELAY_NOT_READY")
            result["native_arm_policy_relay"]={"handoff_preflight":str(output/"native_arm_handoff_preflight.json"),"ready":json.loads(ready_path.read_text()),"gain_or_policy_file_changes":False}
            result["gates"]["NATIVE_ARM_POLICY_RELAY"]="PASS"

        if args.inspect_vr_topic:
            vr_topic = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 5s ros2 topic info /aima/teleop_bridge/vr_data --verbose"], timeout_s=7.0, check=False)
            pg.write_new_text(output / "vr_teleop_topic_info.stdout", vr_topic.stdout)
            result["vr_teleop_topic"] = {"topic": "/aima/teleop_bridge/vr_data", "returncode": vr_topic.returncode, "raw": vr_topic.stdout}
            result["gates"]["VR_TELEOP_TOPIC_DISCOVERY"] = "PASS" if vr_topic.returncode == 0 and "Type:" in vr_topic.stdout else "FAIL"
        if args.inspect_retargeting_topic:
            retargeting_topic = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 5s ros2 topic info /aima/mc/joint/retargeting --verbose"], timeout_s=7.0, check=False)
            pg.write_new_text(output / "retargeting_topic_info.stdout", retargeting_topic.stdout)
            result["retargeting_topic"] = {"topic": "/aima/mc/joint/retargeting", "returncode": retargeting_topic.returncode, "raw": retargeting_topic.stdout}
            result["gates"]["RETARGETING_TOPIC_DISCOVERY"] = "PASS" if retargeting_topic.returncode == 0 and "Type:" in retargeting_topic.stdout else "FAIL"
        if args.use_official_cmd_vel_bridge and not args.fixed_reviewed_checkpoint:
            bridge_log = output / "official_cmd_vel_bridge.stdout"
            lateral_parameter = (" -p max_lateral_velocity:=0.20 -p cmd_timeout_sec:=0.20 -p min_forward_velocity:="+str(args.micro_command_speed)+" -p min_lateral_velocity:="+str(args.micro_command_speed) if args.micro_base_probe else " -p min_forward_velocity:="+str(args.dock_command_speed_cap) if args.dock_command_speed_cap is not None else "" if args.lateral_probe_mps is None else " -p max_lateral_velocity:=0.30")
            if args.dock_angular_speed_cap is not None:lateral_parameter+=" -p min_angular_velocity:="+str(args.dock_angular_speed_cap)+" -p max_angular_velocity:="+str(args.dock_angular_speed_cap)
            bridge_command = (
                environment
                + "exec python3 /workspace/robotsim/reference/AgiBot-Community/raicom2026-agibot-X2/"
                "src/race_task/race_task/cmd_vel_bridge.py --ros-args "
                "-p source:=robotsim_official_cmd_vel -p publish_hz:=50.0 "
                "-p input_source_priority:=40 -p input_source_timeout:=1000 "
                + lateral_parameter
                + "> /workspace/evidence/official_cmd_vel_bridge.stdout 2>&1"
            )
            run(["docker", "exec", "-d", container, "bash", "-lc", bridge_command], timeout_s=5.0)
            bridge_text = wait_for(bridge_log, "Input source registration response", timeout_s=10.0)
            if "state=0" not in bridge_text:
                raise RuntimeError("official cmd_vel bridge did not register its MC source")
            result["gates"]["OFFICIAL_CMD_VEL_BRIDGE"] = "PASS"
        if args.micro_base_probe:
            micro=run(["docker","exec",container,"bash","-lc",environment+"timeout 30s python3 /workspace/evidence/native_micro_probe.py"],timeout_s=33.,check=False)
            pg.write_new_text(output/"native_micro_probe.stdout",micro.stdout)
            path=output/"native_micro_probe_result.json"
            micro_result=json.loads(path.read_text()) if path.is_file() else {"failure":"CHILD_NO_RESULT","stdout_tail":micro.stdout[-1600:]}
            result["native_micro_probe"]={"returncode":micro.returncode,"result":{k:v for k,v in micro_result.items() if k not in ("trace","command_log","bridge_commands","abort_native_sample")}}
            result["gates"]["NATIVE_MICRO_PRESTOP"]="PASS" if micro_result.get("pre_motion_composite",{}).get("pass") else "FAIL"
            result["gates"]["NATIVE_MICRO_SIGNED_DISPLACEMENT"]="NOT_RUN" if not micro_result.get("nonzero_started") else "PASS" if micro.returncode==0 and micro_result.get("pass_result") else "FAIL"
            result["gates"]["NATIVE_MICRO_COMPOSITE_STOP"]="NOT_RUN" if not micro_result.get("nonzero_started") else "PASS" if micro_result.get("post_stop_composite",{}).get("pass") else "FAIL"
            if micro_result.get("nonzero_started") and "locomotion velocity" in result["commands_not_sent"]:result["commands_not_sent"].remove("locomotion velocity")
            if micro.returncode or not micro_result.get("pass_result"):result["failure"]={"gate":"NATIVE_MICRO_PROBE","detail":micro_result.get("failure")}
            raise MicroProbeComplete()
        if args.lateral_probe_mps is not None:
            lateral = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 12s python3 /workspace/evidence/native_lateral_probe.py"], timeout_s=14.0, check=False)
            pg.write_new_text(output / "native_lateral_probe.stdout", lateral.stdout)
            lateral_result = json.loads(lateral.stdout.strip().splitlines()[-1]) if lateral.returncode == 0 else {"returncode": lateral.returncode, "stdout_tail": lateral.stdout[-1200:]}
            result["native_lateral_probe"] = {"transport": "official_cmd_vel_bridge", "command_lateral_mps": float(args.lateral_probe_mps), "command_duration_s": float(args.lateral_probe_duration_s), "returncode": lateral.returncode, "result": lateral_result}
            result["commands_not_sent"].remove("locomotion velocity")
            result["gates"]["NATIVE_LATERAL_SOURCE_REGISTRATION"] = "PASS" if lateral.returncode == 0 else "FAIL"
            lateral_pass = lateral.returncode == 0 and lateral_result.get("pass") is True
            result["gates"]["NATIVE_LATERAL_DISPLACEMENT"] = "PASS" if lateral_pass else "FAIL"
            result["gates"]["NATIVE_LATERAL_STOP_HOLD"] = "PASS" if lateral_pass else "FAIL"
        if hal_target_right is not None:
            # The evidence-local MC overlay disabled only the competing arm
            # and waist publishers.  Start an explicit hold immediately so
            # native locomotion never runs with those groups uncommanded;
            # the worker blocks on a marker until table docking has passed.
            run(["docker", "exec", "-d", container, "bash", "-lc", environment + "python3 /workspace/evidence/native_hal_bilateral_pregrasp.py > /workspace/evidence/native_hal_bilateral_pregrasp.stdout 2>&1"], timeout_s=5.0)
            hal_pregrasp_background = True
        if args.preset_motion_id is not None:
            preset = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 14s python3 /workspace/evidence/native_preset_motion_probe.py"], timeout_s=16.0, check=False)
            pg.write_new_text(output / "native_preset_motion_probe.stdout", preset.stdout)
            preset_result = json.loads(preset.stdout.strip().splitlines()[-1]) if preset.returncode == 0 else {"returncode": preset.returncode, "stdout_tail": preset.stdout[-1200:]}
            result["native_preset_motion"] = {"requested": args.preset_motion_id, "area": args.preset_motion_area, "result": preset_result}
            result["gates"]["NATIVE_PRESET_MOTION_SERVICE"] = "PASS" if preset.returncode == 0 and preset_result.get("pass") else "FAIL"
        if args.dock_world_xy is not None or args.fixed_reviewed_checkpoint:
            if args.fixed_reviewed_checkpoint:
                dock_pass=True
                result['fixed_stance_diagnostic']={'navigation':'NOT_RUN','source_registration':'NOT_RUN','velocity_publishers_created':0}
            else:
                dock = run(["docker", "exec", container, "bash", "-lc", environment + "timeout " + str(int(math.ceil(args.dock_timeout_s + 8.0))) + "s python3 /workspace/evidence/native_locomotion_dock.py"], timeout_s=float(args.dock_timeout_s + 12.0), check=False)
                pg.write_new_text(output / "native_locomotion_dock.stdout", dock.stdout)
                dock_result = json.loads(dock.stdout.strip().splitlines()[-1]) if dock.returncode == 0 else {"returncode": dock.returncode, "stdout_tail": dock.stdout[-1200:]}
                result["native_locomotion_dock"] = {"transport": "official_cmd_vel_bridge" if args.use_official_cmd_vel_bridge else "direct_mc_velocity_legacy", "world_goal_xy_m": [float(args.dock_world_xy[0]), float(args.dock_world_xy[1])], "world_goal_yaw_rad": float(args.dock_yaw_rad), "dock_position_only": bool(args.dock_position_only), "refine_world_xy_m": None if args.refine_world_xy is None else [float(args.refine_world_xy[0]), float(args.refine_world_xy[1])], "refine_world_yaw_rad": args.refine_yaw_rad, "refine_position_only": bool(args.refine_position_only), "dock_position_tolerance_m": float(args.dock_position_tolerance_m), "refine_position_tolerance_m": float(args.refine_position_tolerance_m), "returncode": dock.returncode, "result": dock_result}
                result["commands_not_sent"].remove("locomotion velocity")
                result["gates"]["NATIVE_TABLE_DOCK_SOURCE_REGISTRATION"] = "PASS" if dock.returncode == 0 and isinstance(dock_result.get("source"), dict) else "FAIL"
                dock_pass = dock.returncode == 0 and dock_result.get("pass") is True
                result["gates"]["NATIVE_TABLE_DOCK_ARRIVAL"] = "PASS" if dock_pass else "FAIL"
                result["gates"]["NATIVE_TABLE_DOCK_STOP_HOLD"] = "PASS" if dock_pass else "FAIL"
                if not dock_pass:result["failure"]={"gate":"NATIVE_TABLE_DOCK_ARRIVAL_OR_STOP","detail":dock_result.get("failure"),"child_returncode":dock.returncode}
            if dock_pass:
                if not args.fixed_reviewed_checkpoint:
                    observed=run(["docker","exec",container,"bash","-lc",environment+"timeout 12s python3 /workspace/evidence/native_postdock_stand_observer.py"],timeout_s=14.,check=False)
                    pg.write_new_text(output/'postdock_stand_observer.stdout',observed.stdout)
                    state=json.loads(observed.stdout.strip().splitlines()[-1]) if observed.returncode==0 else {'stdout_tail':observed.stdout[-1200:]}
                    result['postdock_stand_observation']=state
                    ready=observed.returncode==0 and state.get('set_status') is None and state.get('mode_name')=='STAND_DEFAULT' and state.get('mode_status')==100 and state.get('stability',{}).get('pass') is True
                    result['gates']['POSTDOCK_READONLY_STAND']='PASS' if ready else 'FAIL'
                    if not ready:raise RuntimeError('postdock read-only standing qualification failed')
                dock_capture = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 8s python3 /workspace/evidence/native_arm_state_capture.py"], timeout_s=10.0, check=False)
                pg.write_new_text(output / "native_dock_state_capture.stdout", dock_capture.stdout)
                dock_state = json.loads(dock_capture.stdout.strip().splitlines()[-1]) if dock_capture.returncode == 0 else {"returncode": dock_capture.returncode, "stdout_tail": dock_capture.stdout[-1200:]}
                result["native_dock_state_capture"] = dock_state
                required_groups = ("base_pose_world", "leg_positions_rad", "waist_positions_rad", "arm_positions_rad")
                state_ok = dock_capture.returncode == 0 and all(isinstance(dock_state.get(group), dict) for group in required_groups) and dock_state.get("motion_stability", {}).get("pass") is True
                result["gates"]["NATIVE_FIXED_STANCE_STATE_HANDOFF" if args.fixed_reviewed_checkpoint else "NATIVE_TABLE_DOCK_STATE_HANDOFF"] = "PASS" if state_ok else "FAIL"
                if hal_arm_candidate_path is not None:
                    pg.write_new_text(output / "native_dock_state.json", json.dumps(dock_state, ensure_ascii=False, indent=2) + "\n")
                    replan = run([
                        "/home/lzy18001500226/.local/bin/uv", "run", "--offline", "--with", "mujoco==3.3.6", "--with", "numpy", "python3",
                        str(REPO / "scripts/research/issue46/native_runtime/runtime_actual_grasp_replan.py"),
                        "--static-candidate", str(hal_arm_candidate_path),
                        "--static-candidate-sha256", sha256(hal_arm_candidate_path),
                        "--dock-state", str(output / "native_dock_state.json"),
                        "--table", str(args.table_overlay_result),
                    ], timeout_s=30.0, check=False)
                    pg.write_new_text(output / "runtime_actual_grasp_replan.stdout", replan.stdout)
                    replan_summary = json.loads(replan.stdout.strip().splitlines()[-1]) if replan.returncode == 0 else {"returncode": replan.returncode, "stdout_tail": replan.stdout[-1200:]}
                    replan_path = Path(str(replan_summary.get("output_wsl", "")))
                    replan_result_path = replan_path / "result.json"
                    if replan.returncode != 0 or not replan_result_path.is_file():
                        result["hal_arm_runtime_replan"] = {"returncode": replan.returncode, "result": replan_summary}
                        result["gates"]["HAL_ARM_RUNTIME_REPLAN"] = "FAIL"
                        raise RuntimeError("runtime static replan did not produce a governed evidence record")
                    else:
                        replan_record = json.loads(replan_result_path.read_text(encoding="utf-8"))
                        required_replan = ("DOCK_CAPTURE", "RUNTIME_STATIC_IK", "LEFT_ARM_PARK_CLEAR", "OPEN_PREGRASP_CLEAR", "CLOSED_CONTACT_CLEAR")
                        if "circle" in replan_record:required_replan+=("APPROACH_SWEEP",)
                        if args.physical_lift_test:required_replan+=("LIFT_SWEEP",)
                        replan_ok = replan_record.get("classification") == "SIMULATION_ONLY_RUNTIME_CAPTURED_STATIC_REPLAN" and all(replan_record.get("gates", {}).get(name) == "PASS" for name in required_replan)
                        result["hal_arm_runtime_replan"] = {"path": str(replan_result_path), "sha256": sha256(replan_result_path), "returncode": replan.returncode, "gates": replan_record.get("gates")}
                        result["gates"]["HAL_ARM_RUNTIME_REPLAN"] = "PASS" if replan_ok else "FAIL"
                        if replan_ok:
                            runtime_state = replan_record.get("initial_state")
                            runtime_arm = runtime_state.get("arm_positions_rad") if isinstance(runtime_state, dict) else None
                            order = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
                            left_order = tuple("left_" + name[len("right_"):] for name in order)
                            if not isinstance(runtime_arm, dict) or any(name not in runtime_arm for name in order + left_order):
                                raise RuntimeError("runtime static replan lacks complete bilateral arm targets")
                            hal_arm_target_right = [float(runtime_arm[name]) for name in order]
                            hal_arm_target_left = [float(runtime_arm[name]) for name in left_order]
                            arm_gains = stl_hand_startup.load_animation_player_arm_pd_parameters()
                            waist_target=None
                            if args.coordinated_circle_waist:
                                runtime_waist=runtime_state.get("waist_positions_rad",{})
                                waist_order=("waist_yaw_joint","waist_pitch_joint","waist_roll_joint")
                                if any(n not in runtime_waist for n in waist_order):raise RuntimeError("replan lacks coordinated waist target")
                                waist_target=[float(runtime_waist[n]) for n in waist_order]
                                waist_commands=json.loads((output/"native_stand_command_capture.stdout").read_text().strip().splitlines()[-1])["waist_commands"]
                                for n in waist_order:
                                    c=waist_commands[n];kp=float(c["stiffness"]);kd=float(c["damping"])
                                    if not math.isfinite(kp) or not math.isfinite(kd) or kp<=0 or kd<=0:raise RuntimeError("invalid captured active MC waist gains")
                                    arm_gains[n]=(kp,kd)
                                result["coordinated_waist_gains_provenance"]={"source":"actual unchanged MC STAND command capture","values":{n:arm_gains[n] for n in waist_order},"capture_sha256":sha256(output/"native_stand_command_capture.stdout")}
                            elif replan_record.get("waist_actuation_required"):
                                raise RuntimeError("waist target cannot execute on arm-only controller")
                            dock_arm_state = {**dock_state.get("arm_positions_rad",{}),**dock_state.get("waist_positions_rad",{})}
                            if not isinstance(dock_arm_state, dict):
                                raise RuntimeError("native docking capture lacks a complete arm fallback state")
                            pg.write_new_text(output / "native_hal_bilateral_arm_pregrasp.py", stl_hand_startup.build_hal_bilateral_arm_plan_script(arm_gains, hal_arm_target_right, hal_arm_target_left, duration_s=args.arm_plan_duration_s, hold_s=args.arm_plan_hold_s, restore=args.hal_arm_plan_mode == "round-trip", right_hand_command=((0.0 if args.physical_lift_test else float(replan_record.get("commands",{}).get("closure_probe",0))/100.) if args.hal_right_hand_command is not None else None), hand_hold_s=args.hal_hand_close_hold_s if args.hal_right_hand_command is not None else 0.0, initial_arm_state={name: float(dock_arm_state[name]) for name in tuple(arm_gains)},circle_contract=({"circle_center_local_m":replan_record.get("execution_grasp_frame",{}).get("center_m",replan_record["circle"]["center_m"]),"nominal_circle_center_local_m":replan_record["circle"]["center_m"],"height_above_center_m":replan_record["grasp_height_above_center_m"],"local_bottle_axis_in_hand":replan_record.get("local_bottle_axis_in_hand",[1.,0.,0.]),"perception_latest":"/workspace/evidence/fresh_rgbd/latest.json" if replan_record.get("target_source")=="fresh-rgbd" else None,"perception_snapshot":"/workspace/evidence/rgbd_grasp_snapshot.json" if replan_record.get("target_source")=="fresh-rgbd-snapshot" else None} if "circle" in replan_record else None),use_native_bias_effort=True,use_native_joint_feedback=not args.coordinated_circle_waist,track_native_circle=args.native_circle_tracking,enable_single_residual_feedforward=True,source_joint_limits=replan_record.get("source_joint_limits_rad",{}),waist_target=waist_target,hand_open_command=float(replan_record.get("commands",{}).get("open",97))/100.,lift_contract=replan_record.get("lift_preflight") if args.physical_lift_test else None))
                        else:
                            raise RuntimeError("runtime static replan did not retain the required pad, table-clearance, and joint-margin gates")
                if hal_pregrasp_background:
                    pg.write_new_text(hal_pregrasp_marker, json.dumps({"after_native_table_dock": True}) + "\n")
                    try:
                        pregrasp_text = wait_for(hal_pregrasp_stdout, "response_pass", timeout_s=35.0)
                        pregrasp_result = json.loads(pregrasp_text.strip().splitlines()[-1])
                        pregrasp_returncode = 0
                    except Exception as exc:
                        pregrasp_result = {"error": str(exc)}
                        pregrasp_returncode = 1
                    result["native_hal_bilateral_pregrasp"] = {"candidate": result.get("hal_grasp_candidate"), "target_right_arm_rad": hal_target_right, "target_left_arm_rad": hal_target_left, "target_waist_rad": hal_target_waist, "returncode": pregrasp_returncode, "result": pregrasp_result, "hold_during_dock": True}
                    result["gates"]["HAL_BILATERAL_PREGRASP"] = "PASS" if pregrasp_returncode == 0 and pregrasp_result.get("response_pass") and pregrasp_result.get("motion_stability", {}).get("pass") else "FAIL"
                    if "arm trajectory" in result["commands_not_sent"]:
                        result["commands_not_sent"].remove("arm trajectory")
        if args.run_hal_waist_pd_probe:
            waist_probe = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 14s python3 /workspace/evidence/native_hal_waist_pd_probe.py"], timeout_s=16.0, check=False)
            pg.write_new_text(output / "native_hal_waist_pd_probe.stdout", waist_probe.stdout)
            waist_result = json.loads(waist_probe.stdout.strip().splitlines()[-1]) if waist_probe.returncode == 0 else {"returncode": waist_probe.returncode, "stdout_tail": waist_probe.stdout[-1200:]}
            result["native_hal_waist_pd_probe"] = {"returncode": waist_probe.returncode, "result": waist_result}
            result["gates"]["HAL_WAIST_PD_COMMAND_ROUNDTRIP"] = "PASS" if waist_probe.returncode == 0 and waist_result.get("response_pass") else "FAIL"
            result["gates"]["HAL_WAIST_PD_COMMAND_STAND_STABILITY"] = "PASS" if waist_probe.returncode == 0 and waist_result.get("motion_stability", {}).get("pass") else "FAIL"
        if args.run_hal_arm_pd_probe:
            if arm_bootstrap_enabled:
                pg.write_new_text(arm_bootstrap_stop, json.dumps({"handoff": "native_hal_arm_pd_probe"}) + "\n")
                bootstrap_stop_text = wait_for(arm_bootstrap_stdout, '"stopped": true', timeout_s=8.0)
                result["hybrid_arm_bootstrap_handoff"] = json.loads(bootstrap_stop_text.strip().splitlines()[-1])
                if not result["hybrid_arm_bootstrap_handoff"].get("stopped"):
                    raise RuntimeError("hybrid arm bootstrap did not stop before the bounded arm probe")
            arm_probe = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 14s python3 /workspace/evidence/native_hal_arm_pd_probe.py"], timeout_s=16.0, check=False)
            pg.write_new_text(output / "native_hal_arm_pd_probe.stdout", arm_probe.stdout)
            arm_probe_result = json.loads(arm_probe.stdout.strip().splitlines()[-1]) if arm_probe.returncode == 0 else {"returncode": arm_probe.returncode, "stdout_tail": arm_probe.stdout[-1200:]}
            result["native_hal_arm_pd_probe"] = {"returncode": arm_probe.returncode, "result": arm_probe_result}
            result["gates"]["HAL_ARM_PD_COMMAND_ROUNDTRIP"] = "PASS" if arm_probe.returncode == 0 and arm_probe_result.get("response_pass") else "FAIL"
            result["gates"]["HAL_ARM_PD_COMMAND_STAND_STABILITY"] = "PASS" if arm_probe.returncode == 0 and arm_probe_result.get("motion_stability", {}).get("pass") else "FAIL"
        if hal_arm_target_right is not None:
            handoff_pass=(result["gates"].get("NATIVE_FIXED_STANCE_STATE_HANDOFF")=="PASS" if args.fixed_reviewed_checkpoint else result["gates"]["NATIVE_TABLE_DOCK_ARRIVAL"]=="PASS" and result["gates"]["NATIVE_TABLE_DOCK_STATE_HANDOFF"]=="PASS")
            if not handoff_pass or result["gates"]["HAL_ARM_RUNTIME_REPLAN"] != "PASS":
                raise RuntimeError("arm-only HAL pregrasp is gated on successful native docking, state handoff, and runtime static replan")
            if arm_bootstrap_enabled:
                pg.write_new_text(arm_bootstrap_stop, json.dumps({"handoff": "native_hal_bilateral_arm_pregrasp"}) + "\n")
                bootstrap_stop_text = wait_for(arm_bootstrap_stdout, '"stopped": true', timeout_s=8.0)
                result["hybrid_arm_bootstrap_handoff"] = json.loads(bootstrap_stop_text.strip().splitlines()[-1])
                if not result["hybrid_arm_bootstrap_handoff"].get("stopped"):
                    raise RuntimeError("hybrid arm bootstrap did not stop before the arm-only pregrasp")
            child_budget=42 if args.physical_lift_test else 35
            arm_pregrasp = run(["docker", "exec", container, "bash", "-lc", environment + f"timeout {child_budget}s python3 /workspace/evidence/native_hal_bilateral_arm_pregrasp.py"], timeout_s=child_budget+2., check=False)
            pg.write_new_text(output / "native_hal_bilateral_arm_pregrasp.stdout", arm_pregrasp.stdout)
            arm_pregrasp_result = decode_arm_execution_output(arm_pregrasp.returncode,arm_pregrasp.stdout)
            result["native_hal_bilateral_arm_pregrasp"] = {"candidate": result.get("hal_arm_candidate"), "plan_mode": args.hal_arm_plan_mode, "target_right_arm_rad": hal_arm_target_right, "target_left_arm_rad": hal_arm_target_left, "returncode": arm_pregrasp.returncode, "result": arm_pregrasp_result}
            result["gates"]["HAL_BILATERAL_ARM_PREGRASP"] = "PASS" if arm_pregrasp.returncode == 0 and arm_pregrasp_result.get("native_tcp_arrival", {}).get("pass") and arm_pregrasp_result.get("motion_stability", {}).get("pass") else "FAIL"
            if args.hal_right_hand_command is not None:
                closure_issued = arm_pregrasp_result.get("closure_issued")
                if closure_issued is None:
                    for line in arm_pregrasp.stdout.splitlines():
                        try:
                            observed = json.loads(line)
                        except ValueError:
                            continue
                        if "closure_issued" in observed:
                            closure_issued = observed["closure_issued"]
                result["gates"]["HAL_RIGHT_HAND_CLOSE"] = ("NOT_RUN" if not closure_issued else "PASS" if arm_pregrasp_result.get("physical_bilateral_contact", {}).get("pass") else "FAIL")
                result['gates']['HAL_PHYSICAL_LIFT']='NOT_RUN'
                if args.physical_lift_test:
                    lift=arm_pregrasp_result.get('lift_execution',{})
                    if lift.get('executed'):
                        result['gates']['HAL_PHYSICAL_LIFT']='FAIL'
                        if 'hold_end_ns' in lift:
                            rows=[json.loads(line) for line in (output/'native_physics.jsonl').read_text().splitlines()]
                            metrics=native_grasp_audit.evaluate_unsupported_lift(rows,initial_bottle_z_m=lift['initial_bottle_z_m'],hold_start_ns=lift['hold_start_ns'],hold_end_ns=lift['hold_end_ns'])
                            result['physical_lift_evidence']=metrics
                            result['gates']['HAL_PHYSICAL_LIFT']='PASS' if arm_pregrasp.returncode==0 and metrics['pass'] else 'FAIL'
                if args.physical_lift_test and lift.get('executed') and result['gates']['HAL_PHYSICAL_LIFT']!='PASS':
                    result['failure']={'gate':'HAL_PHYSICAL_LIFT','detail':result.get('physical_lift_evidence',lift),'execution_tail':arm_pregrasp.stdout[-1600:]}
                elif result["gates"]["HAL_BILATERAL_ARM_PREGRASP"] != "PASS":
                    result["failure"] = {"gate": "HAL_BILATERAL_ARM_PREGRASP", "child_returncode": arm_pregrasp.returncode, "closure_issued": bool(closure_issued), "detail": arm_pregrasp.stdout[-1200:] if arm_pregrasp.returncode else "Actual TCP arrival or standing stability did not qualify"}
                elif result['gates']['HAL_RIGHT_HAND_CLOSE']=='FAIL':
                    result['failure']={'gate':'HAL_RIGHT_HAND_CLOSE','detail':arm_pregrasp_result.get('physical_bilateral_contact')}
                elif args.physical_lift_test and result['gates']['HAL_PHYSICAL_LIFT']!='PASS':
                    result['failure']={'gate':'HAL_PHYSICAL_LIFT','detail':result.get('physical_lift_evidence',lift)}
                if "hand command" in result["commands_not_sent"]:
                    result["commands_not_sent"].remove("hand command")
                if args.pause_on_claw_failure and (result["gates"]["HAL_RIGHT_HAND_CLOSE"] != "PASS" or args.physical_lift_test):
                    if args.native_circle_tracking and not (output/'native_tracking_target.stop').exists():pg.write_new_text(output/'native_tracking_target.stop','controlled execution ended before review-only physics pause\n')
                    # Freeze physical stepping through the verified native UI,
                    # keeping the renderer interactive for multi-angle review.
                    result["windows_foreground_before_review_pause"]=activate_wslg_native_window(window,output,label="windows_foreground_before_review_pause")
                    before_pause=json.loads((output/'native_latest.json').read_text())
                    paused=run(["docker","exec",container,"bash","-lc","python3 /workspace/evidence/native_mujoco_reset.py "+("click-native-pause --native-ui-events /workspace/evidence/native_ui_events.jsonl" if args.verified_native_mouse_controls else "send-pause")+" --expected-pid "+str(pid)+" --expected-title "+repr(str(window['title']))+" --window-id "+str(window['window_id'])+" --dispatch-event-json /workspace/evidence/manual_pause_dispatch.json"],timeout_s=8.,check=False)
                    pg.write_new_text(output/'manual_pause.stdout',paused.stdout)
                    pause_deadline=time.monotonic()+3.;pause_stable_since=None;last_sim=None
                    while time.monotonic()<pause_deadline:
                        paused_second=json.loads((output/'native_latest.json').read_text())
                        if paused_second['sim_time_s']==last_sim:
                            pause_stable_since=pause_stable_since or time.monotonic()
                            if time.monotonic()-pause_stable_since>=.5:break
                        else:pause_stable_since=None
                        last_sim=paused_second['sim_time_s'];time.sleep(.05)
                    if paused.returncode or pause_stable_since is None or time.monotonic()-pause_stable_since<.5:raise RuntimeError('MANUAL_REVIEW_NATIVE_PAUSE_UNCONFIRMED')
                    result['manual_review_physics_pause']={'confirmed':True,'before_sim_time':before_pause['sim_time_s'],'paused_sim_time':paused_second['sim_time_s'],'do_not_resume_physics':True}
                    pg.write_new_text(output/'manual_review_partial_result.json',json.dumps(result,indent=2)+'\n')
                    release = output / "continue_after_manual_review"
                    pause_reason = ("PHYSICAL_LIFT_PASS_REVIEW" if result['gates'].get('HAL_PHYSICAL_LIFT')=='PASS' else "PHYSICAL_LIFT_UNQUALIFIED_REVIEW" if args.physical_lift_test else "ARM_CHILD_EXECUTION_ERROR" if arm_pregrasp.returncode != 0 else "HAND_MAPPING_OR_RESPONSE_UNQUALIFIED")
                    result["manual_review_pause"] = {"reason": pause_reason, "child_returncode": arm_pregrasp.returncode, "child_output_tail": arm_pregrasp.stdout[-2000:], "container": container, "display": args.interactive_x11_display, "release_file": str(release), "instructions": "Physics is paused after recording the acceptance result. Rotate the visible MuJoCo window; create the release file after review. Review time does not count toward the unsupported hold."}
                    pg.write_new_text(output / "manual_review_pause.json", json.dumps(result["manual_review_pause"], ensure_ascii=False, indent=2) + "\n")
                    print(pause_reason + ". MuJoCo remains open. Create continue_after_manual_review to clean up.", flush=True)
                    deadline = time.monotonic() + 900.0
                    review_window_closed=False
                    while time.monotonic() < deadline and not release.exists():
                        alive=run(["docker","exec",container,"bash","-c","kill -0 "+str(pid)+" 2>/dev/null"],timeout_s=3.,check=False)
                        if alive.returncode:
                            review_window_closed=True;result["manual_review_window_closed"]=True;break
                        time.sleep(0.5)
                    if not release.exists() and not review_window_closed:
                        raise TimeoutError("manual review pause expired without release file")
            if "arm trajectory" in result["commands_not_sent"]:
                result["commands_not_sent"].remove("arm trajectory")
        if hal_target_right is not None and not hal_pregrasp_background:
            pregrasp = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 35s python3 /workspace/evidence/native_hal_bilateral_pregrasp.py"], timeout_s=37.0, check=False)
            pg.write_new_text(output / "native_hal_bilateral_pregrasp.stdout", pregrasp.stdout)
            pregrasp_result = json.loads(pregrasp.stdout.strip().splitlines()[-1]) if pregrasp.returncode == 0 else {"returncode": pregrasp.returncode, "stdout_tail": pregrasp.stdout[-1200:]}
            result["native_hal_bilateral_pregrasp"] = {"candidate": result.get("hal_grasp_candidate"), "target_right_arm_rad": hal_target_right, "target_left_arm_rad": hal_target_left, "target_waist_rad": hal_target_waist, "returncode": pregrasp.returncode, "result": pregrasp_result}
            result["gates"]["HAL_BILATERAL_PREGRASP"] = "PASS" if pregrasp.returncode == 0 and pregrasp_result.get("response_pass") and pregrasp_result.get("motion_stability", {}).get("pass") else "FAIL"
            if "arm trajectory" in result["commands_not_sent"]:
                result["commands_not_sent"].remove("arm trajectory")
        if arm_target is not None:
            arm_plan = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 30s python3 /workspace/evidence/native_hal_arm_plan.py"], timeout_s=32.0, check=False)
            pg.write_new_text(output / "native_hal_arm_plan.stdout", arm_plan.stdout)
            arm_result = json.loads(arm_plan.stdout.strip().splitlines()[-1]) if arm_plan.returncode == 0 else {"returncode": arm_plan.returncode}
            result["native_arm_plan"] = {"ik_result": {"path": str(args.arm_ik_result), "sha256": sha256(args.arm_ik_result)}, "target_right_arm_rad": arm_target, "returncode": arm_plan.returncode, "result": arm_result}
            result["gates"]["ARM_PLAN_ROUNDTRIP"] = "PASS" if arm_plan.returncode == 0 and arm_result.get("response_pass") else "FAIL"
            result["gates"]["ARM_PLAN_STAND_STABILITY"] = "PASS" if arm_plan.returncode == 0 and arm_result.get("motion_stability", {}).get("pass") else "FAIL"
            result["gates"]["ARM_PLAN_RESTORE"] = "PASS" if arm_plan.returncode == 0 and arm_result.get("max_restore_error_rad", float("inf")) <= 0.04 else "FAIL"
        if urs_target is not None:
            urs_plan = run(["docker", "exec", container, "bash", "-lc", environment + "timeout 35s python3 /workspace/evidence/native_urs_arm_plan.py"], timeout_s=37.0, check=False)
            pg.write_new_text(output / "native_urs_arm_plan.stdout", urs_plan.stdout)
            urs_result = json.loads(urs_plan.stdout.strip().splitlines()[-1]) if urs_plan.returncode == 0 else {"returncode": urs_plan.returncode, "stdout_tail": urs_plan.stdout[-1200:]}
            result["native_urs_arm_plan"] = {"action_name": args.urs_action_name, "ik_result": None if args.urs_arm_ik_result is None else {"path": str(args.urs_arm_ik_result), "sha256": sha256(args.urs_arm_ik_result)}, "grasp_candidate": None if args.urs_grasp_candidate is None else {"path": str(args.urs_grasp_candidate), "sha256": sha256(args.urs_grasp_candidate)}, "target_right_arm_rad": urs_target, "target_left_arm_rad": urs_left_target, "returncode": urs_plan.returncode, "result": urs_result}
            result["gates"]["URS_INPUT_SOURCE"] = "PASS" if urs_plan.returncode == 0 and urs_result.get("source_name") == "upper_body_example" else "FAIL"
            result["gates"]["URS_ARM_PLAN_ROUNDTRIP"] = "PASS" if urs_plan.returncode == 0 and urs_result.get("response_pass") else "FAIL"
            result["gates"]["URS_ARM_PLAN_STAND_STABILITY"] = "PASS" if urs_plan.returncode == 0 and urs_result.get("motion_stability", {}).get("pass") else "FAIL"
            result["gates"]["URS_ARM_PLAN_RESTORE"] = "PASS" if urs_plan.returncode == 0 and urs_result.get("max_restore_error_rad", float("inf")) <= 0.04 else "FAIL"
            if "arm trajectory" in result["commands_not_sent"]:
                result["commands_not_sent"].remove("arm trajectory")
    except MicroProbeComplete:
        pass
    except Exception as exc:
        observed_failure={"type": type(exc).__name__, "message": str(exc)}
        if result.get("failure") is not None:result.setdefault("secondary_failures",[]).append(observed_failure)
        else:result["failure"]=observed_failure
    finally:
        if args.native_circle_tracking:
            marker=output/'native_tracking_target.stop'
            if not marker.exists():pg.write_new_text(marker,'stop before native runtime teardown\n')
        if (output/'rgbd_grasp_snapshot.json').exists():result['rgbd_stationary_target_snapshot']={'path':str(output/'rgbd_grasp_snapshot.json'),'sha256':sha256(output/'rgbd_grasp_snapshot.json'),'contract':'FRESH_ACQUISITION_PRE_EXECUTION_REOBSERVATION_BOUNDED_STATIC_TARGET'}
        if (output/'fresh_rgbd').exists() and not (output/'fresh_rgbd'/'stop').exists():pg.write_new_text(output/'fresh_rgbd'/'stop','stop before runtime teardown\n')
        subprocess.run(["docker", "stop", "--time", "5", container], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        subprocess.run(["docker", "rm", container], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        absent=subprocess.run(["docker","inspect",container],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=5.,check=False)
        result['cleanup']={'container':container,'inspect_returncode':absent.returncode,'confirmed_absent':absent.returncode==1 and 'no such object' in absent.stdout.lower() and container.lower() in absent.stdout.lower(),'inspect_output':absent.stdout.strip()}
        pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
        pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": result["gates"], "failure": result.get("failure")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
