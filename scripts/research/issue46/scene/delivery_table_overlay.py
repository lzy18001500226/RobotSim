"""Derived room layout: replace only the chair with the existing pickup table.

Native MuJoCo layout review only; no MC, ROS publishers or mj_step.
"""
import argparse
import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent import path_governance as pg

ROOT = Path("/home/lzy18001500226/projects/RobotSim")
CHAIR = "raicom_table_chair"
PICKUP = "raicom_table_table"
DELIVERY = "raicom_delivery_table"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_chair(root, delivery_yaw_deg=90., backboard_offset_m=0.):
    root = copy.deepcopy(root)
    parents = {child: parent for parent in root.iter() for child in parent}
    chair = root.find('.//body[@name="' + CHAIR + '"]')
    table = root.find('.//body[@name="' + PICKUP + '"]')
    if chair is None or table is None:
        raise ValueError("expected exact existing chair and table bodies")
    if any(x.tag in ("joint", "freejoint") for x in chair.iter()) or any(x.tag in ("joint", "freejoint") for x in table.iter()):
        raise ValueError("furniture replacement must not alter generalized coordinates")
    names = {x.get("name") for x in root.iter("body")}
    if DELIVERY in names:
        raise ValueError("delivery table already present")
    replacement = copy.deepcopy(table)
    replacement.set("name", DELIVERY)
    if not np.isfinite(backboard_offset_m) or not 0. <= backboard_offset_m <= .10:
        raise ValueError("backboard offset outside the user-approved0..10cm change")
    position = np.fromstring(chair.get("pos", "0 0 0"), sep=" ")
    if position.shape != (3,):
        raise ValueError("invalid chair floor anchor")
    position[1] += backboard_offset_m
    replacement.set("pos", " ".join(format(float(x), ".17g") for x in position))
    # User-requested vertical rotation at the former chair anchor plus explicit offset.
    if delivery_yaw_deg not in (0., 90.):
        raise ValueError("only reviewed table orientations0/90deg are supported")
    angle = np.deg2rad(delivery_yaw_deg) * .5
    qz = np.array([np.cos(angle), 0., 0., np.sin(angle)])
    original_q = np.fromstring(table.get("quat", "1 0 0 0"), sep=" ")
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qz, original_q)
    replacement.set("quat", " ".join(format(float(x), ".17g") for x in q))
    for i, geom in enumerate(replacement.iter("geom")):
        geom.set("name", DELIVERY + "_geom_" + str(i))
    for body in list(replacement.iter("body"))[1:]:
        body.set("name", DELIVERY + "__" + body.get("name", "child"))
    parent = parents[chair]
    if parents[table] is not parent:
        raise ValueError("expected furniture in one existing room frame")
    idx = list(parent).index(chair)
    parent.remove(chair)
    parent.insert(idx, replacement)
    return root



def move_pickup(root, delta_world, model, initial_data):
    """Scene-initialization translation only; robot configuration is untouched."""
    root = copy.deepcopy(root)
    for name in (PICKUP, "test_cylinder"):
        body = root.find('.//body[@name="' + name + '"]')
        if body is None:
            raise ValueError("missing pickup furniture/object:" + name)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        parent = int(model.body_parentid[bid])
        local_delta = initial_data.xmat[parent].reshape(3, 3).T @ delta_world
        pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ") + local_delta
        body.set("pos", " ".join(format(float(v), ".17g") for v in pos))
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    adr = int(model.jnt_qposadr[joint])
    for key in root.findall("./keyframe/key"):
        q = np.fromstring(key.get("qpos", ""), sep=" ")
        if len(q) != model.nq:
            raise ValueError("initial keyframe vector differs")
        q[adr:adr+3] += delta_world
        key.set("qpos", " ".join(format(float(v), ".17g") for v in q))
    return root

def furniture_bounds(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    points = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("expected original source table mesh")
        mesh = int(model.geom_dataid[g])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start+count]
        points.append(vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    vertices = np.concatenate(points)
    return dict(body_origin_world_m=data.xpos[bid].tolist(),
                minimum_world_m=vertices.min(axis=0).tolist(),
                maximum_world_m=vertices.max(axis=0).tolist(),
                top_height_m=float(vertices[:, 2].max()))


def build(source_result, delivery_yaw_deg=90., backboard_offset_m=0., pickup_forward_m=0., robot_stance_forward_m=0.):
    source_result = Path(source_result).resolve()
    record = json.loads(source_result.read_text())
    out = pg.create_unique_output_directory(ROOT / "reviews/issue-46", prefix="delivery-table-layout-review").native_path
    if not np.isfinite(pickup_forward_m) or not 0. <= pickup_forward_m <= .10:
        raise ValueError("pickup translation outside requested0..10cm")
    original = mujoco.MjModel.from_xml_path(record["generated"]["host_scene"]["path"])
    initial_data = mujoco.MjData(original)
    mujoco.mj_resetDataKeyframe(original, initial_data, 0)
    mujoco.mj_forward(original, initial_data)
    pelvis = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    R = initial_data.xmat[pelvis].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    pickup_delta = np.array([np.cos(yaw), np.sin(yaw), 0.]) * pickup_forward_m
    bottle_joint = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    bottle_adr = int(original.jnt_qposadr[bottle_joint])
    if robot_stance_forward_m not in (0., .10):
        raise ValueError("only explicit original or10cm fixed-stance target is authorized")
    stance_delta = np.array([np.cos(yaw), np.sin(yaw), 0.]) * robot_stance_forward_m
    root_adr = int(original.jnt_qposadr[mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")])
    generated = {}
    for label in ("host_scene", "runtime_scene"):
        item = record["generated"][label]
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("source scene hash differs:" + label)
        before = ET.parse(path).getroot()
        after = replace_chair(before, delivery_yaw_deg, backboard_offset_m)
        if pickup_forward_m:
            after = move_pickup(after, pickup_delta, original, initial_data)
        if robot_stance_forward_m:
            for key in after.findall("./keyframe/key"):
                q = np.fromstring(key.get("qpos", ""), sep=" ")
                q[root_adr:root_adr+3] += stance_delta
                key.set("qpos", " ".join(format(float(v), ".17g") for v in q))
        for name in ("pelvis", PICKUP, "test_cylinder"):
            query = './/body[@name="' + name + '"]'
            previous, current = copy.deepcopy(before.find(query)), copy.deepcopy(after.find(query))
            if pickup_forward_m and name in (PICKUP, "test_cylinder"):
                current.set("pos", previous.get("pos"))
            if ET.tostring(previous) != ET.tostring(current):
                raise ValueError("unrelated subtree changed:" + name)
        # Only host review gets a distinct window title; native runtime name stays unchanged.
        if label == "host_scene":
            after.set("model", "X2-TWO-TABLES-LAYOUT-REVIEW-" + str(int(delivery_yaw_deg)) + "DEG-Y" + str(int(round((1.5+backboard_offset_m)*1000))) + "MM-PICKUP" + str(int(round(pickup_forward_m*1000))) + "MM")
        dest = out / (label + ".xml")
        pg.write_new_bytes(dest, ET.tostring(after, encoding="utf-8", xml_declaration=True))
        generated[label] = dict(path=str(dest), sha256=sha(dest))
    derived = mujoco.MjModel.from_xml_path(generated["host_scene"]["path"])
    for field in ("nq", "nv", "nu", "njnt", "neq", "nsensor", "nkey"):
        if getattr(original, field) != getattr(derived, field):
            raise ValueError("non-furniture dimension changed:" + field)
    checked = ["jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range",
               "qpos0", "key_qpos", "key_qvel", "actuator_trnid", "actuator_gear",
               "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
               "eq_type", "eq_data", "opt.gravity", "opt.timestep"]
    for field in checked:
        a = getattr(original.opt, field[4:]) if field.startswith("opt.") else getattr(original, field)
        z = getattr(derived.opt, field[4:]) if field.startswith("opt.") else getattr(derived, field)
        expected = np.array(a, copy=True)
        if pickup_forward_m and field in ("qpos0", "key_qpos"):
            expected[..., bottle_adr:bottle_adr+3] += pickup_delta
        if robot_stance_forward_m and field == "key_qpos":
            expected[..., root_adr:root_adr+3] += stance_delta
        if not np.allclose(expected, z, rtol=0., atol=1e-14):
            raise ValueError("robot/control/keyframe changed:" + field)
    if mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_BODY, CHAIR) != -1:
        raise ValueError("chair is still active")
    a, z = mujoco.MjData(original), mujoco.MjData(derived)
    mujoco.mj_resetDataKeyframe(original, a, 0)
    mujoco.mj_resetDataKeyframe(derived, z, 0)
    mujoco.mj_forward(original, a)
    mujoco.mj_forward(derived, z)
    pickup = furniture_bounds(derived, z, PICKUP)
    delivery = furniture_bounds(derived, z, DELIVERY)
    expected_pickup_minimum = np.asarray(furniture_bounds(original, a, PICKUP)["minimum_world_m"]) + pickup_delta
    if not np.allclose(pickup["minimum_world_m"], expected_pickup_minimum, rtol=0., atol=1e-12):
        raise ValueError("pickup table translation differs")
    old_bottle = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    new_bottle = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if not np.allclose(z.xpos[new_bottle] - a.xpos[old_bottle], pickup_delta, atol=1e-12):
        raise ValueError("bottle did not move with tabletop")
    if abs(pickup["top_height_m"] - delivery["top_height_m"]) > 1e-9:
        raise ValueError("table heights differ")
    result = dict(
        classification="SIMULATION_ONLY_ENVIRONMENT_LAYOUT_REVIEW_NO_DYNAMICS",
        source_result=dict(path=str(source_result), sha256=sha(source_result)),
        generated=generated,
        delta=dict(remove_body=CHAIR, add_body=DELIVERY, copy_geometry_from=PICKUP,
                   position_rule="former chair floor anchor; original table source mesh, explicitly rotated about vertical axis",
                   delivery_yaw_offset_deg=delivery_yaw_deg, backboard_offset_world_m=[0., backboard_offset_m, 0.],
                   no_robot_or_vendor_source_edits=True, pickup_forward_m=pickup_forward_m,
                   pickup_world_delta_m=pickup_delta.tolist(), source_robot_heading_rad=yaw,
                   bottle_translated_with_pickup=True, fixed_stance_world_delta_m=stance_delta.tolist()),
        pickup_table=pickup, delivery_table=delivery,
        robot_control_and_keyframe_checks=checked, robot_subtree_unchanged=True,
        furniture_and_bottle_geometry_unchanged=True, only_scene_initial_translations_updated_in_keyframes=True,
        mj_step_calls=0, mc_started=False, new_ros_publishers=0,
        layout_review="PENDING_HUMAN", placement_or_route_validation="NOT_RUN",
        user_authorization="2026-11-08 explicit request: replace chair by a table at the chair location and open MuJoCo for review",
    )
    # Preserve the existing native-overlay interface; only the approved scene
    # initialization and furniture transforms differ from the parent.
    import yaml
    native_record = copy.deepcopy(record)
    native_record["generated"].update(generated)
    cfg = yaml.safe_load(Path(record["generated"]["simulator_config"]["path"]).read_text())
    cfg["MujocoSimModule"]["simulation_model"] = "/workspace/evidence/" + Path(generated["runtime_scene"]["path"]).name
    cfg_path = out / "simulator_two_table_layout.yaml"
    pg.write_new_text(cfg_path, yaml.safe_dump(cfg, sort_keys=False))
    native_record["generated"]["simulator_config"] = dict(path=str(cfg_path), sha256=sha(cfg_path))
    cp = native_record.get("reviewed_checkpoint")
    if cp:
        for name, delta in (("initial_pelvis_pose", stance_delta), ("initial_bottle_pose", pickup_delta)):
            cp["source_" + name] = copy.deepcopy(cp[name])
            cp[name]["position_m"] = (np.asarray(cp[name]["position_m"]) + delta).tolist()
        cp["expected_pose_provenance"] = "derived user-approved episode initialization; not re-measured runtime"
    runtime_xml = ET.parse(generated["runtime_scene"]["path"]).getroot()
    for node in runtime_xml.iter():
        value = node.get("file")
        if value and value.startswith("/workspace/robotsim/"):
            node.set("file", str(ROOT / value[len("/workspace/robotsim/"):]))
        elif value and value.startswith("/workspace/raicom2026/"):
            node.set("file", str(ROOT / "simulation/mujoco/raicom2026" / value[len("/workspace/raicom2026/"):]))
    check_path = out / "runtime_hostcheck_scene.xml"
    pg.write_new_bytes(check_path, ET.tostring(runtime_xml, encoding="utf-8", xml_declaration=True))
    runtime_model = mujoco.MjModel.from_xml_path(str(check_path))
    for key in ("geom_pos","geom_quat","geom_dataid","body_pos","body_quat","key_qpos","jnt_range","actuator_gainprm","actuator_forcerange"):
        if not np.array_equal(getattr(derived,key),getattr(runtime_model,key)):
            raise ValueError("runtime/host scene mismatch:"+key)
    native_record["generated"]["runtime_hostcheck_scene"] = dict(path=str(check_path),sha256=sha(check_path))
    native_record["approved_two_table_layout"] = result["delta"]
    pg.write_new_text(out / "native_overlay_result.json", json.dumps(native_record, indent=2) + "\n")
    result["native_overlay_result"] = dict(path=str(out / "native_overlay_result.json"), sha256=sha(out / "native_overlay_result.json"))
    pg.write_new_text(out / "result.json", json.dumps(result, indent=2) + "\n")
    pg.write_artifact_manifest(out)
    return out / "result.json"


def load_review(result_path):
    result_path = Path(result_path).resolve()
    record = json.loads(result_path.read_text())
    item = record["generated"]["host_scene"]
    if sha(item["path"]) != item["sha256"]:
        raise ValueError("review scene differs")
    model = mujoco.MjModel.from_xml_path(item["path"])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return record, model, data


def configure_camera(camera, preset, delivery_position=(0., 1.5, 0.), pickup_position=(1.5, -1.4, 0.), hip_position=(.95,-1.43,.61)):
    target, distance = {
        "overview": ([0.0, 0.0, .55], 5.7),
        "hip": (hip_position, .85),
        "delivery": ([delivery_position[0], delivery_position[1], .45], 2.2),
        "pickup": ([pickup_position[0]-.3, pickup_position[1], .7], 2.2),
    }[preset]
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = 135.
    camera.elevation = -42.


def view(result_path, initial_preset="overview", recorded_frame=None):
    import mujoco.viewer
    frame = json.loads(Path(recorded_frame).read_text()) if recorded_frame else None
    record, model, data = load_review(result_path)
    contacts = [] if frame is None else [c for c in frame["contacts"] if {c["body1"],c["body2"]}=={"pelvis","left_hip_roll_link"}]
    if frame is not None:
        from scripts.research.issue46.native_runtime.native_grasp_audit import apply_recorded_native_frame
        apply_recorded_native_frame(model,data,frame)
    hip = contacts[0]["position_m"] if contacts else [.95,-1.43,.61]
    state = {"preset": initial_preset, "colliders": False}
    def key(code):
        if code in (49, 50, 51, 52):
            state["preset"] = {49:"overview",50:"delivery",51:"pickup",52:"hip"}[code]
        if code==67:state["colliders"]=not state["colliders"]
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as window:
        window.opt.geomgroup[:] = [1, 1, 0, 0, 1, 0]
        while window.is_running():
            if state["preset"]:
                configure_camera(window.cam, state["preset"], record["delivery_table"]["body_origin_world_m"], record["pickup_table"]["body_origin_world_m"],hip)
                state["preset"] = None
            window.opt.geomgroup[:] = [1,0,0,1,0,0] if state["colliders"] else [1,1,0,0,1,0]
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = state["colliders"]
            with window.lock():
                window.user_scn.ngeom=0
                for contact in contacts:
                    g=window.user_scn.geoms[window.user_scn.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.006]*3),np.asarray(contact["position_m"]),np.eye(3).ravel(),np.array([1.,.15,.12,1.],dtype=np.float32))
                    window.user_scn.ngeom+=1
            title="RECORDED HIP CONTACT / NO PHYSICS" if frame else "TWO-TABLE LAYOUT REVIEW / NO PHYSICS"
            detail="1 overview | 2 delivery | 3 pickup | 4 HIP | C colliders/visual | mouse free view"
            window.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,title,detail),
                (None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,"RED = recorded contact, not a new simulation",
                 ("pelvis / left_hip_roll: %.3fmm, %.1fN" % (contacts[0]["distance_m"]*1000.,contacts[0]["local_force_torque"][0])) if contacts else "")])
            window.sync()
            time.sleep(.02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-result", type=Path)
    group.add_argument("--review-result", type=Path)
    parser.add_argument("--delivery-yaw-deg", type=float, choices=(0., 90.), default=90.)
    parser.add_argument("--toward-backboard-m", type=float, default=0.)
    parser.add_argument("--pickup-forward-m", type=float, default=0.)
    parser.add_argument("--camera", choices=("overview", "delivery", "pickup","hip"), default="overview")
    parser.add_argument("--robot-stance-forward-m", type=float, choices=(0., .1), default=0.)
    parser.add_argument("--recorded-native-frame",type=Path)
    args = parser.parse_args()
    if args.source_result:
        print(build(args.source_result, args.delivery_yaw_deg, args.toward_backboard_m, args.pickup_forward_m, args.robot_stance_forward_m))
    else:
        view(args.review_result, args.camera,args.recorded_native_frame)
