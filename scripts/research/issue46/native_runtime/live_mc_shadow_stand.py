#!/usr/bin/env python3
"""Run a bounded live-MC-command full-contact shadow standing check.

The delivered simulator cannot publish joint state when its scene has any
additional environment geom.  This research bridge therefore leaves native
``sim_mujoco + MC`` on the immutable source scene and feeds their *live*
leg/waist command stream into a separately loaded full-contact MuJoCo model.
The shadow is physically stepped and never receives a qpos/qvel write after
its initialization.  It is deliberately not a claim of vendor-equivalent
closed-loop state feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from scripts.research.issue46.native_runtime import stl_hand_startup as startup  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_result(path: Path, expected: str, *, label: str) -> dict[str, object]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha256(path) != expected:
        raise ValueError(f"{label} is unavailable or hash-mismatched")
    return json.loads(path.read_text(encoding="utf-8"))


def full_scene(result: dict[str, object]) -> Path:
    if result.get("classification") != "SIMULATION_ONLY_FULL_CONTACT_TABLE_SCENE_PREPARED":
        raise ValueError("not a full-contact table scene")
    gates = result.get("gates", {})
    if not isinstance(gates, dict) or any(gates.get(name) != "PASS" for name in ("HOST_STATIC_MODEL_COMPILE", "TABLE_SCENE_STATIC_COMPILE", "TRUE_POSE_CYLINDER_PRESENT")):
        raise ValueError("full-contact scene has not passed static preflight")
    generated = result.get("generated")
    item = generated.get("runtime_scene") if isinstance(generated, dict) else None
    if not isinstance(item, dict):
        raise ValueError("full-contact result lacks runtime scene")
    path = Path(str(item.get("path", "")))
    if not path.is_file() or path.is_symlink() or sha256(path) != str(item.get("sha256", "")):
        raise ValueError("runtime full-contact scene is unavailable or hash-mismatched")
    return path


def build_shadow_script(state: dict[str, object]) -> str:
    """Create a self-contained ROS/MuJoCo bridge script for one container."""
    payload = json.dumps(state, sort_keys=True)
    return f'''#!/usr/bin/env python3
import json, math, os, time
import mujoco, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from aimdk_msgs.msg import HandState, HandStateArray, HandType, JointCommandArray, JointState, JointStateArray, MessageHeader
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

STATE = json.loads({payload!r})
LEG = ('left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint')
WAIST = ('waist_yaw_joint','waist_pitch_joint','waist_roll_joint')
ARM = ('left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_yaw_joint','left_wrist_pitch_joint','left_wrist_roll_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_yaw_joint','right_wrist_pitch_joint','right_wrist_roll_joint')
HEAD = ('head_yaw_joint','head_pitch_joint')
READY = (-.35,.45,0.,-1.,0.,.15,0.,-.35,-.45,0.,-1.,0.,.15,0.)
ARM_GAINS = {{'shoulder_pitch':(3.,.1),'shoulder_roll':(3.,.1),'shoulder_yaw':(2.,.05),'elbow':(2.,.05),'wrist_yaw':(2.,.05),'wrist_pitch':(1.,.02),'wrist_roll':(1.,.02)}}

def jid(model,name):
    value=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    if value<0: raise RuntimeError('missing joint '+name)
    return value
def setq(model,data,name,value): data.qpos[model.jnt_qposadr[jid(model,name)]]=value
def qv(model,data,name):
    j=jid(model,name); return float(data.qpos[model.jnt_qposadr[j]]),float(data.qvel[model.jnt_dofadr[j]])
def aid(model,name):
    value=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)
    if value<0: raise RuntimeError('missing actuator '+name)
    return value
def arm_gain(name):
    for key,value in ARM_GAINS.items():
        if key in name: return value
    raise RuntimeError('no arm gain '+name)
def freebase(model,data):
    pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    free=next(i for i in range(model.njnt) if model.jnt_bodyid[i]==pelvis and model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
    return pelvis,free
def tilt(quat):
    w,x,y,z=map(float,quat); return max(abs(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),abs(math.asin(max(-1,min(1,2*(w*y-z*x))))))

class Bridge(Node):
    def __init__(self):
        super().__init__('robotsim_live_mc_shadow_stand')
        q=QoSProfile(depth=20,reliability=ReliabilityPolicy.BEST_EFFORT); self.cmd={{'leg':{{}},'waist':{{}}}}; self.count={{'leg':0,'waist':0}}
        self.create_subscription(JointCommandArray,'/aima/hal/joint/leg/command',lambda m:self.store('leg',m),q)
        self.create_subscription(JointCommandArray,'/aima/hal/joint/waist/command',lambda m:self.store('waist',m),q)
        self.leg_pub=self.create_publisher(JointStateArray,'/aima/hal/joint/leg/state',q); self.waist_pub=self.create_publisher(JointStateArray,'/aima/hal/joint/waist/state',q); self.arm_pub=self.create_publisher(JointStateArray,'/aima/hal/joint/arm/state',q); self.head_pub=self.create_publisher(JointStateArray,'/aima/hal/joint/head/state',q); self.hand_pub=self.create_publisher(HandStateArray,'/aima/hal/joint/hand/state',q); self.imu_pub=self.create_publisher(Imu,'/aima/hal/imu/torso/state',q); self.chest_imu_pub=self.create_publisher(Imu,'/aima/hal/imu/chest/state',q); self.odom_pub=self.create_publisher(Odometry,'/aima/hal/odom/state',q)
    def store(self,area,msg):
        self.count[area]+=1; self.cmd[area]={{x.name:(float(x.position),float(x.velocity),float(x.effort),float(x.stiffness),float(x.damping)) for x in msg.joints}}
    def state(self,model,data,names,pub):
        msg=JointStateArray(); msg.header=MessageHeader(); msg.header.stamp=self.get_clock().now().to_msg(); msg.header.meas_stamp=msg.header.stamp
        for name in names:
            q,v=qv(model,data,name); item=JointState(); item.name=name; item.position=q; item.velocity=v; item.effort=float(data.qfrc_actuator[model.jnt_dofadr[jid(model,name)]]); msg.joints.append(item)
        pub.publish(msg)
    def publish_state(self,model,data,pelvis,free):
        self.state(model,data,LEG,self.leg_pub); self.state(model,data,WAIST,self.waist_pub); self.state(model,data,ARM,self.arm_pub); self.state(model,data,HEAD,self.head_pub)
        hands=HandStateArray(); hands.header=MessageHeader(); hands.header.stamp=self.get_clock().now().to_msg(); hands.header.meas_stamp=hands.header.stamp; hands.left_hand_type=HandType(value=2); hands.right_hand_type=HandType(value=2)
        for name,collection in (('left_claw_joint',hands.left_hands),('right_claw_joint',hands.right_hands)):
            item=HandState(); item.name=name; item.position=.99; item.velocity=0.; item.effort=0.; collection.append(item)
        self.hand_pub.publish(hands)
        site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu_0'); sensor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'body-angular-velocity'); q=np.empty(4); mujoco.mju_mat2Quat(q,data.site_xmat[site]); sadr=model.sensor_adr[sensor]; ang=data.sensordata[sadr:sadr+3]; lin=data.qvel[model.jnt_dofadr[free]:model.jnt_dofadr[free]+3]
        imu=Imu(); imu.orientation.w=float(q[0]); imu.orientation.x=float(q[1]); imu.orientation.y=float(q[2]); imu.orientation.z=float(q[3]); imu.angular_velocity.x=float(ang[0]); imu.angular_velocity.y=float(ang[1]); imu.angular_velocity.z=float(ang[2]); self.imu_pub.publish(imu); self.chest_imu_pub.publish(imu)
        odom=Odometry(); odom.pose.pose.position.x=float(data.xpos[pelvis,0]); odom.pose.pose.position.y=float(data.xpos[pelvis,1]); odom.pose.pose.position.z=float(data.xpos[pelvis,2]); odom.pose.pose.orientation.w=float(q[0]); odom.pose.pose.orientation.x=float(q[1]); odom.pose.pose.orientation.y=float(q[2]); odom.pose.pose.orientation.z=float(q[3]); odom.twist.twist.linear.x=float(lin[0]); odom.twist.twist.linear.y=float(lin[1]); odom.twist.twist.linear.z=float(lin[2]); self.odom_pub.publish(odom)

rclpy.init(); node=Bridge()
try:
    model=mujoco.MjModel.from_xml_path('/workspace/evidence/shadow_full_contact.xml'); data=mujoco.MjData(model)
    pelvis,free=freebase(model,data); qadr=int(model.jnt_qposadr[free]); base=STATE['base_pose_world']; data.qpos[qadr:qadr+2]=base['xyz_m'][:2]; data.qpos[qadr+3:qadr+7]=np.asarray(base['quaternion_wxyz'],float)/np.linalg.norm(base['quaternion_wxyz'])
    for name in LEG+WAIST: setq(model,data,name,STATE['joint_positions'][name])
    for name,value in zip(ARM,READY): setq(model,data,name,value)
    # Reviewed source-STL open approach pose; all follower qpos are initialized
    # once before stepping, then driven only by equality/position actuator.
    hand=STATE['hand_positions']
    for name,value in hand.items(): setq(model,data,name,value)
    data.ctrl[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_L_claw_joint')]=.99; data.ctrl[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_R_claw_joint')]=.99
    mujoco.mj_forward(model,data); start=data.xpos[pelvis].copy()
    deadline=time.monotonic()+6.;
    while time.monotonic()<deadline and (node.count['leg']<20 or node.count['waist']<20): node.publish_state(model,data,pelvis,free); rclpy.spin_once(node,timeout_sec=.002); time.sleep(.002)
    if node.count['leg']<20 or node.count['waist']<20 or any(n not in node.cmd['leg'] for n in LEG) or any(n not in node.cmd['waist'] for n in WAIST): raise RuntimeError('missing live MC leg/waist stream')
    trace=[]; failure=None; steps=2000
    for step in range(steps):
        node.publish_state(model,data,pelvis,free); rclpy.spin_once(node,timeout_sec=0.0)
        allcmd={{**node.cmd['leg'],**node.cmd['waist']}}
        for name in LEG+WAIST:
            target,vel,eff,kp,kv=allcmd[name]; qpos,qvel=qv(model,data,name); torque=eff+kp*(target-qpos)+kv*(vel-qvel)+float(data.qfrc_bias[model.jnt_dofadr[jid(model,name)]])
            a=aid(model,name); data.ctrl[a]=np.clip(torque,*model.actuator_ctrlrange[a])
        for name,target in zip(ARM,READY):
            qpos,qvel=qv(model,data,name); kp,kv=arm_gain(name); a=aid(model,name); data.ctrl[a]=np.clip(kp*(target-qpos)-kv*qvel+float(data.qfrc_bias[model.jnt_dofadr[jid(model,name)]]),*model.actuator_ctrlrange[a])
        mujoco.mj_step(model,data)
        drift=float(np.linalg.norm(data.xpos[pelvis,:2]-start[:2])); h=float(data.xpos[pelvis,2]); t=tilt(data.xquat[pelvis]); limits=[]
        for j in range(model.njnt):
            if model.jnt_limited[j] and model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
                q=float(data.qpos[model.jnt_qposadr[j]]); lo,hi=model.jnt_range[j]
                if q<lo-1e-5 or q>hi+1e-5: limits.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j))
        if step%10==0 or step==steps-1: trace.append({{'step':step+1,'time_s':float(data.time),'drift_m':drift,'height_m':h,'tilt_rad':t,'leg_messages':node.count['leg'],'waist_messages':node.count['waist'],'limits':limits}})
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all(): failure={{'gate':'NONFINITE','step':step+1}}; break
        if drift>.10: failure={{'gate':'DRIFT','step':step+1,'drift_m':drift}}; break
        if not .2<=h<=1.2: failure={{'gate':'HEIGHT','step':step+1,'height_m':h}}; break
        if t>math.radians(15): failure={{'gate':'TILT','step':step+1,'tilt_rad':t}}; break
        if limits: failure={{'gate':'JOINT_LIMIT','step':step+1,'joints':limits}}; break
    print(json.dumps({{'live_command_counts':node.count,'steps':step+1,'final':trace[-1] if trace else None,'failure':failure,'pass':failure is None,'trace':trace}}))
finally:
    node.destroy_node(); rclpy.shutdown()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-overlay-result", required=True, type=Path)
    parser.add_argument("--full-contact-result", required=True, type=Path)
    parser.add_argument("--full-contact-result-sha256", required=True)
    parser.add_argument("--native-stand-result", required=True, type=Path)
    parser.add_argument("--native-stand-result-sha256", required=True)
    args = parser.parse_args()
    native_inputs = startup.exact_generated_inputs(args.native_overlay_result)
    full = require_result(args.full_contact_result, args.full_contact_result_sha256, label="full-contact result")
    shadow_scene = full_scene(full)
    state = require_result(args.native_stand_result, args.native_stand_result_sha256, label="native standing result")
    gates = state.get("gates", {})
    capture = state.get("native_arm_state_capture", {}).get("result") if isinstance(state.get("native_arm_state_capture"), dict) else None
    if not isinstance(gates, dict) or any(gates.get(gate) != "PASS" for gate in ("NATIVE_STAND_STABILITY", "MC_STAND_COMMAND_CAPTURE")) or not isinstance(capture, dict):
        raise SystemExit("native result lacks a passing standing capture")
    # Hand qpos are the exact reviewed 99% source-STL trajectory row captured
    # from the compiled full-contact model at initialization.
    hand_positions = {"L_claw_joint": -0.99, "L_hand_wide1_joint": 0.99, "L_hand_narrow3_joint": 0.5444466735264069, "L_hand_narrow_loop_joint": -1.5124137100694064, "L_hand_wide3_joint": -0.5444489967884211, "L_hand_wide_loop_joint": 1.5124144523980185, "R_claw_joint": -0.99, "R_hand_wide1_joint": 0.99, "R_hand_narrow3_joint": 0.5444466735264069, "R_hand_narrow_loop_joint": -1.5124137100694064, "R_hand_wide3_joint": -0.5444489967884211, "R_hand_wide_loop_joint": 1.5124144523980185}
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-live-mc-shadow-stand").native_path
    container = startup.docker_container_name(output)
    stage = "created_output"
    result = {"schema":"robotsim.issue46.live-mc-shadow-stand.v1","classification":"SIMULATION_ONLY_LIVE_MC_COMMAND_FULL_CONTACT_SHADOW","mj_step_calls":"NOT_RUN","gates":{"HOST_PREFLIGHT":"NOT_RUN","NATIVE_MC_STAND":"NOT_RUN","LIVE_MC_COMMAND_STREAM":"NOT_RUN","FULL_CONTACT_SHADOW_STAND":"NOT_RUN","GRASP":"NOT_RUN"}}
    try:
        result["host_preflight"] = startup.host_preflight(container); result["gates"]["HOST_PREFLIGHT"]="PASS"
        stage = "copy_native_inputs"
        for source in native_inputs.values(): pg.write_new_bytes(output/source.name,source.read_bytes())
        simulator_payload = yaml.safe_load((output / native_inputs["simulator_config"].name).read_text(encoding="utf-8"))
        publisher_list = simulator_payload["MujocoSimModule"]["PublisherManager"]["publisher_cfg_list"]
        for publisher in publisher_list:
            if publisher.get("name") in {"imu_state_ros", "imu_chest_state_ros", "arm_joint_state", "leg_joint_state", "waist_joint_state", "head_joint_state", "odom_state_ros"}:
                publisher["enable"] = False
        stage = "rewrite_simulator_config"
        pg.write_reserved_text(output, output / native_inputs["simulator_config"].name, yaml.safe_dump(simulator_payload, allow_unicode=True, sort_keys=False), overwrite_existing=True)
        stage = "copy_full_contact_scene"
        pg.write_new_bytes(output/"shadow_full_contact.xml",shadow_scene.read_bytes())
        stage = "install_mujoco_site"
        mujoco_site = output / "mujoco_python_site"
        startup.run(["/home/lzy18001500226/.local/bin/uv", "pip", "install", "--offline", "--target", str(mujoco_site), "mujoco==3.3.6"], timeout_s=30.0)
        result["mujoco_python_site"] = {"path": str(mujoco_site), "package": "mujoco==3.3.6"}
        pg.write_new_text(output/"shadow_bridge.py",build_shadow_script({"base_pose_world":capture["base_pose_world"],"joint_positions":{**capture["leg_positions_rad"],**capture["waist_positions_rad"]},"hand_positions":hand_positions}))
        pg.write_new_text(output/"native_stand_client.py",startup.build_stand_client_script())
        pg.write_new_text(output/"launch_sim.sh",startup.build_launch_script(native_inputs["simulator_config"].name)); (output/"launch_sim.sh").chmod(0o755)
        pg.write_new_text(output/"launch_mc.sh",startup.build_mc_launch_script()); (output/"launch_mc.sh").chmod(0o755)
        runtime_dump=Path(str(result["host_preflight"]["runtime_dump"])); pg.write_new_bytes(output/"mc_sim.yaml.dump",runtime_dump.read_bytes()); generated=Path(str(result["host_preflight"]["generated_directory"])); import shutil; shutil.copytree(generated,output/"mc_sim_generated")
        command=["docker","run","-d","--pull=never","--name",container,"--network=bridge","--read-only","--cap-drop=ALL","--security-opt=no-new-privileges","--user=1000:1000","--tmpfs","/tmp:rw,nosuid,nodev,size=1g","--tmpfs","/agibot/data/var:rw,nosuid,nodev,uid=1000,gid=1000,mode=0770,size=64m","--mount",f"type=bind,source={startup.RUNTIME},target=/workspace/raicom2026,readonly","--mount",f"type=bind,source={EVIDENCE_REPO},target=/workspace/robotsim,readonly","--mount",f"type=bind,source={output},target=/workspace/evidence","--mount","type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly","--mount",f"type=bind,source={output/'mc_sim.yaml.dump'},target=/workspace/raicom2026/{startup.MC_DUMP_RELATIVE}","--mount",f"type=bind,source={output/'mc_sim_generated'},target=/workspace/raicom2026/{startup.MC_GENERATED_RELATIVE}","-e","DISPLAY=:0","-e","HOME=/tmp","-e","ROS_DOMAIN_ID=20","-e","RMW_IMPLEMENTATION=rmw_fastrtps_cpp",startup.IMAGE,"sleep","infinity"]
        startup.run(command,timeout_s=10.0); startup.run(["docker","exec","-d",container,"bash","-lc","bash /workspace/evidence/launch_sim.sh > /workspace/evidence/mujoco-console.log 2>&1"],timeout_s=5.0)
        ready,_=startup.wait_for_log(output/"mujoco-console.log",(startup.SIM_READY_MARKER,),timeout_s=35.0)
        if not ready: raise RuntimeError("native simulator did not become ready")
        startup.run(["docker","exec","-d",container,"bash","-lc","bash /workspace/evidence/launch_mc.sh > /workspace/evidence/mc-console.log 2>&1"],timeout_s=5.0)
        mc_ready,_=startup.wait_for_log(output/"mc_logs/mc.log",(startup.MC_READY_MARKER,),timeout_s=35.0)
        if not mc_ready: raise RuntimeError("MC did not become ready")
        env="source /opt/ros/humble/setup.bash; source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash; "
        startup.run(["docker","exec","-d",container,"bash","-lc",env+"PYTHONPATH=/workspace/evidence/mujoco_python_site:$PYTHONPATH python3 /workspace/evidence/shadow_bridge.py > /workspace/evidence/shadow_bridge.stdout 2>&1"],timeout_s=5.0)
        time.sleep(3.0)
        stand=startup.run(["docker","exec",container,"bash","-lc",env+"timeout 10s python3 /workspace/evidence/native_stand_client.py"],timeout_s=12.0,check=False)
        try:
            stand_payload=json.loads(stand.stdout.strip().splitlines()[-1])
        except (IndexError,json.JSONDecodeError):
            stand_payload={"parse_error":True}
        stand_pass=stand.returncode==0 and stand_payload.get("set_status")==1 and stand_payload.get("mode_name")=="STAND_DEFAULT" and stand_payload.get("stability", {}).get("pass")
        result["native_stand"]={"returncode":stand.returncode,"result":stand_payload,"stdout":stand.stdout}; result["gates"]["NATIVE_MC_STAND"]="PASS" if stand_pass else "FAIL"
        shadow_stdout = output / "shadow_bridge.stdout"
        deadline = time.monotonic() + 22.0
        while time.monotonic() < deadline:
            if shadow_stdout.is_file() and "live_command_counts" in shadow_stdout.read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(0.2)
        bridge_stdout = shadow_stdout.read_text(encoding="utf-8", errors="replace") if shadow_stdout.is_file() else ""
        try: bridge_result=json.loads(bridge_stdout.strip().splitlines()[-1])
        except (IndexError,json.JSONDecodeError): bridge_result={"parse_error":True}
        result["bridge"]={"returncode":0 if bridge_stdout else 1,"result":bridge_result,"stdout":bridge_stdout}; result["mj_step_calls"]=bridge_result.get("steps",0); result["gates"]["LIVE_MC_COMMAND_STREAM"]="PASS" if bridge_stdout and not bridge_result.get("parse_error") else "FAIL"; result["gates"]["FULL_CONTACT_SHADOW_STAND"]="PASS" if bridge_stdout and bridge_result.get("pass") else "FAIL"
    except Exception as exc:
        result["failure"]={"type":type(exc).__name__,"message":str(exc),"stage":stage}
    finally:
        result["container_cleanup"]=startup.safe_cleanup(container); pg.write_new_text(output/"result.json",json.dumps(result,ensure_ascii=False,indent=2)+"\n"); pg.write_new_bytes(output/Path(__file__).name,Path(__file__).read_bytes()); pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl":str(output),"gates":result["gates"],"failure":result.get("failure")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
