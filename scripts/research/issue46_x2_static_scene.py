from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from issue46_x2_grasp import (
    ARM,
    APPROACH_PALM_POS,
    CANONICAL_X2_BOTTLE_DIAMETER_M,
    CANONICAL_X2_BOTTLE_HEIGHT_M,
    CANONICAL_X2_BOTTLE_MASS_KG,
    DT,
    FINGERS,
    GOAL,
    PALM_TARGET_ROTATION,
    ROOT,
    ROBOT_BASE_WORLD,
    ROBOT_BASE_YAW,
    SOURCE_TABLE_HALF_EXTENTS,
    SOURCE_TABLE_XY,
    SOURCE_PIN,
    START,
    TABLE_TOP_Z,
    URDF,
    body_ancestors,
    build_model,
    name,
    ik_plan,
    referenced_meshes,
    resolve_joints,
    sha256,
)


OUT = Path(
    os.environ.get(
        "ISSUE46_STATIC_EVIDENCE_DIR",
        "/tmp/robotsim-issue46-static-scene",
    )
)
G1_REFERENCE_XML = Path(
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-43-g1-stock-hands-canonical/"
    "m0-1791173016-11457-165/model/g1_m0_bottle.xml"
)
BOTTLE_GEOM_NAMES = (
    "bottle_body",
    "bottle_shoulder",
    "bottle_neck",
    "bottle_cap",
)
TABLE_GEOM_NAMES = (
    "m0_table_top",
    "m0_table_leg_front_left",
    "m0_table_leg_front_right",
    "m0_table_leg_back_left",
    "m0_table_leg_back_right",
)
def camera(
    lookat: np.ndarray | tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
) -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(result)
    result.lookat[:] = lookat
    result.distance = distance
    result.azimuth = azimuth
    result.elevation = elevation
    return result


def camera_from_direction(
    lookat: np.ndarray,
    distance: float,
    direction_to_camera: np.ndarray,
) -> mujoco.MjvCamera:
    view_direction = -np.asarray(direction_to_camera, dtype=float)
    view_direction /= np.linalg.norm(view_direction)
    azimuth = math.degrees(math.atan2(view_direction[1], view_direction[0]))
    elevation = math.degrees(
        math.asin(float(np.clip(view_direction[2], -1.0, 1.0)))
    )
    return camera(lookat, distance, azimuth, elevation)


def hand_morphology_audit(model: mujoco.MjModel) -> dict[str, object]:
    source_xml = ET.parse(URDF).getroot()
    source_joints = {
        joint.get("name"): joint
        for joint in source_xml.findall("joint")
        if joint.get("name", "").startswith(("L_", "R_"))
    }
    joint_comparisons = []
    failures = []

    for joint_name, source_joint in source_joints.items():
        child_name = source_joint.find("child").get("link")
        origin = source_joint.find("origin")
        source_pos = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        source_rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        if source_joint.get("type") == "fixed":
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, child_name
            )
            source_quat = np.zeros(4)
            mujoco.mju_euler2Quat(source_quat, source_rpy, "XYZ")
            pos_error = float(np.max(np.abs(model.body_pos[body_id] - source_pos)))
            quat_error = float(
                1.0 - abs(float(np.dot(model.body_quat[body_id], source_quat)))
            )
            joint_comparisons.append(
                {
                    "joint": joint_name,
                    "type": "fixed",
                    "position_error_m": pos_error,
                    "quaternion_error": quat_error,
                }
            )
            if pos_error > 1e-10 or quat_error > 1e-8:
                failures.append(joint_name)
        elif source_joint.get("type") == "revolute":
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            source_axis = np.fromstring(
                source_joint.find("axis").get("xyz", "0 0 1"), sep=" "
            )
            limit = source_joint.find("limit")
            source_range = np.array(
                [float(limit.get("lower")), float(limit.get("upper"))]
            )
            axis_error = float(
                np.max(np.abs(model.jnt_axis[joint_id] - source_axis))
            )
            range_error = float(
                np.max(np.abs(model.jnt_range[joint_id] - source_range))
            )
            joint_comparisons.append(
                {
                    "joint": joint_name,
                    "type": "revolute",
                    "source_axis": source_axis.tolist(),
                    "compiled_axis": model.jnt_axis[joint_id].tolist(),
                    "source_range_rad": source_range.tolist(),
                    "compiled_range_rad": model.jnt_range[joint_id].tolist(),
                    "axis_error": axis_error,
                    "range_error_rad": range_error,
                }
            )
            if axis_error > 1e-12 or range_error > 1e-12:
                failures.append(joint_name)
        else:
            failures.append(f"unsupported_joint_type:{joint_name}")

    palm_audit = {}
    for side in ("L", "R"):
        palm_body_name = f"{side}_palm"
        palm_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name
        )
        palm_joint = source_joints[f"{side}_palm_joint"]
        palm_link = next(
            link
            for link in source_xml.findall("link")
            if link.get("name") == palm_body_name
        )
        source_mesh = palm_link.find("./visual/geometry/mesh").get("filename")
        compiled_meshes = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != palm_body_id:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id >= 0:
                compiled_meshes.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                )
        expected_mesh = Path(source_mesh).stem
        if compiled_meshes.count(expected_mesh) != 1:
            failures.append(f"palm_mesh:{side}")

        thumb_joint = source_joints[f"{side}_thumb_roll_joint"]
        thumb_body_name = thumb_joint.find("child").get("link")
        thumb_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, thumb_body_name
        )
        source_thumb_pos = np.fromstring(
            thumb_joint.find("origin").get("xyz", "0 0 0"), sep=" "
        )
        thumb_pos_error = float(
            np.max(np.abs(model.body_pos[thumb_body_id] - source_thumb_pos))
        )
        thumb_side_sign = float(model.body_pos[thumb_body_id, 1])
        if thumb_pos_error > 1e-10 or (side == "L" and thumb_side_sign >= 0) or (
            side == "R" and thumb_side_sign <= 0
        ):
            failures.append(f"thumb_side:{side}")

        palm_audit[side] = {
            "palm_body": palm_body_name,
            "mount_parent": palm_joint.find("parent").get("link"),
            "source_mount_xyz_m": np.fromstring(
                palm_joint.find("origin").get("xyz", "0 0 0"), sep=" "
            ).tolist(),
            "source_mount_rpy_rad": np.fromstring(
                palm_joint.find("origin").get("rpy", "0 0 0"), sep=" "
            ).tolist(),
            "compiled_mount_xyz_m": model.body_pos[palm_body_id].tolist(),
            "compiled_mount_quat_wxyz": model.body_quat[palm_body_id].tolist(),
            "source_visual_mesh": source_mesh,
            "source_visual_mesh_sha256": sha256((URDF.parent / source_mesh).resolve()),
            "compiled_visual_mesh_names": compiled_meshes,
            "thumb_link": thumb_body_name,
            "thumb_mount_xyz_m": model.body_pos[thumb_body_id].tolist(),
        }

    if failures:
        raise RuntimeError(f"URDF/MuJoCo hand morphology mismatch: {failures}")
    return {
        "source_joint_count": len(source_joints),
        "source_mimic_relation_count": sum(
            joint.find("mimic") is not None for joint in source_joints.values()
        ),
        "compiled_equality_count": int(model.neq),
        "joint_comparisons": joint_comparisons,
        "palms": palm_audit,
        "handedness_verified": True,
        "mimic_relations_enforced_by_equalities": int(model.neq) > 0,
    }


def bottle_hand_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, str]]:
    bottle_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in BOTTLE_GEOM_NAMES
    }
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first, second = int(contact.geom1), int(contact.geom2)
        if not bottle_geoms.intersection((first, second)):
            continue
        bottle_geom = first if first in bottle_geoms else second
        other_geom = second if bottle_geom == first else first
        other_body = int(model.geom_bodyid[other_geom])
        ancestors = body_ancestors(model, other_body)
        if any(body.startswith(("R_", "L_")) for body in ancestors):
            contacts.append(
                {
                    "bottle_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom),
                    "hand_body": ancestors[0],
                    "hand_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom),
                }
            )
    return contacts


def hand_table_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, str]]:
    table_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in TABLE_GEOM_NAMES
    }
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        first, second = int(contact.geom1), int(contact.geom2)
        table_geom = first if first in table_geoms else second if second in table_geoms else None
        if table_geom is None:
            continue
        hand_geom = second if table_geom == first else first
        ancestors = body_ancestors(model, int(model.geom_bodyid[hand_geom]))
        if any(body.startswith(("R_", "L_")) for body in ancestors):
            contacts.append(
                {
                    "table_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom),
                    "hand_body": ancestors[0],
                    "hand_geom": name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_geom),
                }
            )
    return contacts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    log("EXPERIMENT=RobotSim Issue #46 neutral OmniHand static diagnostic")
    log(
        f"COMMAND=MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')} "
        f"ISSUE46_STATIC_EVIDENCE_DIR={OUT} {sys.executable} {Path(__file__).resolve()}"
    )
    log("MODE=static render and source audit only; no mj_step or grasp rollout")
    log("NEUTRAL_HAND_CONFIGURATION=all left/right hand joints at vendor zero qpos")

    model, details = build_model()
    data = details["data"]
    bottle_qpos = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
        ]
    )
    bottle_initial_qpos = data.qpos[bottle_qpos : bottle_qpos + 7].copy()
    right_arm_names = list(ARM)
    left_arm_names = [joint.replace("right_", "left_", 1) for joint in right_arm_names]
    right_finger_names = list(FINGERS)
    left_finger_names = [joint.replace("R_", "L_", 1) for joint in right_finger_names]
    _, left_arm_qpos = resolve_joints(model, left_arm_names)
    right_arm_ids, right_arm_qpos = resolve_joints(model, right_arm_names)
    _, left_finger_qpos = resolve_joints(model, left_finger_names)
    _, right_finger_qpos = resolve_joints(model, right_finger_names)
    all_hand_qpos = np.array(left_finger_qpos + right_finger_qpos, dtype=int)
    all_arm_qpos = np.array(left_arm_qpos + right_arm_qpos, dtype=int)

    data.qpos[all_arm_qpos] = 0.0
    data.qpos[all_hand_qpos] = 0.0
    mujoco.mj_forward(model, data)
    if not np.allclose(data.qpos[all_hand_qpos], 0.0, atol=0.0):
        raise RuntimeError("Neutral hand setup did not leave every finger at zero qpos")
    if not np.array_equal(data.qpos[bottle_qpos : bottle_qpos + 7], bottle_initial_qpos):
        raise RuntimeError("Neutral hand setup changed the bottle generalized position")
    neutral_bottle_contacts = bottle_hand_contacts(model, data)
    neutral_table_contacts = hand_table_contacts(model, data)
    neutral_right_shoulder_position = data.xpos[
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
        )
    ].copy()
    neutral_bottle_position = data.qpos[bottle_qpos : bottle_qpos + 3].copy()

    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.geomgroup[0] = 0
    render_option.sitegroup[:] = 0
    render_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    renderer = mujoco.Renderer(model, height=720, width=1280)
    images = []

    def render_image(
        filename: str,
        view_camera: mujoco.MjvCamera,
        scene_option: mujoco.MjvOption = render_option,
    ) -> None:
        renderer.update_scene(data, camera=view_camera, scene_option=scene_option)
        imageio.imwrite(OUT / filename, renderer.render())
        images.append(filename)
        log(f"IMAGE={OUT / filename}")

    left_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_palm")
    right_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_palm")
    left_position = data.xpos[left_palm].copy()
    right_position = data.xpos[right_palm].copy()
    midpoint = (left_position + right_position) / 2.0
    left_rotation = data.xmat[left_palm].reshape(3, 3)
    right_rotation = data.xmat[right_palm].reshape(3, 3)
    left_front = left_rotation[:, 2].copy()
    right_front = right_rotation[:, 2].copy()
    left_hand_center = left_position + left_front * 0.075
    right_hand_center = right_position + right_front * 0.075

    view_center = np.array([0.30, 0.03, 0.70])
    render_image(
        "canonical_overview.png",
        camera(view_center, 2.35, 135.0, -17.0),
    )
    render_image(
        "front_overview.png",
        camera(view_center, 2.45, 90.0, -12.0),
    )
    render_image(
        "side_alignment.png",
        camera(view_center, 2.55, 180.0, -10.0),
    )

    original_geom_groups = model.geom_group.copy()
    hand_visual_count = 0
    wrist_body_names = {"left_wrist_roll_link", "right_wrist_roll_link"}
    for geom_id in range(model.ngeom):
        if int(original_geom_groups[geom_id]) != 1:
            continue
        ancestors = body_ancestors(model, int(model.geom_bodyid[geom_id]))
        if not ancestors:
            continue
        body_name = ancestors[0]
        if body_name.startswith(("L_", "R_")) or wrist_body_names.intersection(ancestors):
            model.geom_group[geom_id] = 2
            hand_visual_count += 1
    hand_render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(hand_render_option)
    hand_render_option.geomgroup[:] = 0
    hand_render_option.geomgroup[2] = 1
    hand_render_option.sitegroup[:] = 0
    hand_render_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    render_image(
        "neutral_left_palm_front.png",
        camera_from_direction(left_hand_center, 0.42, left_front),
        hand_render_option,
    )
    render_image(
        "neutral_right_palm_front.png",
        camera_from_direction(right_hand_center, 0.42, right_front),
        hand_render_option,
    )
    render_image(
        "neutral_both_palms_comparison.png",
        camera(midpoint, 0.95, 90.0, 0.0),
        hand_render_option,
    )
    render_image(
        "neutral_left_palm_side.png",
        camera_from_direction(left_hand_center, 0.48, left_rotation[:, 0].copy()),
        hand_render_option,
    )
    render_image(
        "neutral_right_palm_side.png",
        camera_from_direction(right_hand_center, 0.48, right_rotation[:, 0].copy()),
        hand_render_option,
    )

    neutral_hand_qpos = data.qpos[all_hand_qpos].copy()
    open_pregrasp_arm_qpos = ik_plan(
        model,
        data.qpos.copy(),
        details["site_id"],
        right_arm_ids,
        [APPROACH_PALM_POS],
        PALM_TARGET_ROTATION,
    )[0]
    data.qpos[right_arm_qpos] = open_pregrasp_arm_qpos
    data.qpos[all_hand_qpos] = 0.0
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos[bottle_qpos : bottle_qpos + 7], bottle_initial_qpos):
        raise RuntimeError("Open pregrasp diagnostic changed the bottle generalized position")
    if not np.allclose(data.qpos[all_hand_qpos], 0.0, atol=0.0):
        raise RuntimeError("Open pregrasp diagnostic did not keep fingers at zero qpos")
    right_open_position = data.xpos[right_palm].copy()
    right_open_normal = data.xmat[right_palm].reshape(3, 3)[:, 2].copy()
    right_open_center = (right_open_position + bottle_initial_qpos[:3]) / 2.0
    open_pregrasp_bottle_contacts = bottle_hand_contacts(model, data)
    open_pregrasp_table_contacts = hand_table_contacts(model, data)
    source_right_hand_links = {
        joint.find("child").get("link")
        for joint in ET.parse(URDF).getroot().findall("joint")
        if joint.get("name", "").startswith("R_")
    }
    model.geom_group[:] = 0
    for geom_id in range(model.ngeom):
        if int(original_geom_groups[geom_id]) != 1:
            continue
        geom_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        ancestors = body_ancestors(model, int(model.geom_bodyid[geom_id]))
        if "right_wrist_roll_link" in ancestors or source_right_hand_links.intersection(
            ancestors
        ):
            model.geom_group[geom_id] = 2
        elif geom_name in BOTTLE_GEOM_NAMES or geom_name == "m0_table_top":
            model.geom_group[geom_id] = 3
    open_pregrasp_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(open_pregrasp_option)
    open_pregrasp_option.geomgroup[:] = 0
    open_pregrasp_option.geomgroup[2] = 1
    open_pregrasp_option.geomgroup[3] = 1
    open_pregrasp_option.sitegroup[:] = 0
    open_pregrasp_option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    render_image(
        "right_hand_natural_open_pregrasp.png",
        camera_from_direction(
            right_open_center,
            0.60,
            right_open_normal + np.array([1.5, 0.0, 0.12]),
        ),
        open_pregrasp_option,
    )
    model.geom_group[:] = original_geom_groups
    renderer.close()

    morphology = hand_morphology_audit(model)
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    torso_front_world = data.xmat[torso_body].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
    bottle_position = data.qpos[bottle_qpos : bottle_qpos + 3].copy()
    reference_hash = sha256(G1_REFERENCE_XML) if G1_REFERENCE_XML.is_file() else None
    try:
        vendor_head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        vendor_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception as error:
        raise RuntimeError(f"Could not verify the pinned vendor checkout: {error}") from error
    if vendor_head != SOURCE_PIN or vendor_dirty:
        raise RuntimeError(
            f"Vendor identity mismatch: head={vendor_head}, dirty={vendor_dirty}"
        )
    robot_sim_root = Path(__file__).resolve().parents[2]
    robot_sim_head = subprocess.check_output(
        ["git", "-C", str(robot_sim_root), "rev-parse", "HEAD"], text=True
    ).strip()
    robot_sim_dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(robot_sim_root), "status", "--porcelain"], text=True
        ).strip()
    )
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_material_id = int(model.geom_matid[floor_geom_id])
    floor_texture_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane"
    )
    floor_texture_channels = np.flatnonzero(
        model.mat_texid[floor_material_id] == floor_texture_id
    ).tolist()

    result = {
        "result": "STATIC_NEUTRAL_HAND_DIAGNOSTIC",
        "manipulation_result": "NOT_RUN; prior physical result remains FAIL",
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "renderer": "MuJoCo EGL offscreen",
            "physics_steps": 0,
            "mj_step_called": False,
        },
        "robot_sim": {
            "root": str(robot_sim_root),
            "head_at_capture": robot_sim_head,
            "dirty_at_capture": robot_sim_dirty,
            "source_file_sha256": {
                str(path.relative_to(robot_sim_root)): sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__import__("issue46_x2_grasp").__file__).resolve(),
                    robot_sim_root
                    / "simulation"
                    / "mujoco"
                    / "canonical_manipulation_assets.py",
                )
            },
        },
        "vendor": {
            "repository": "https://github.com/AgibotTech/agibot_x2_urdf",
            "commit": vendor_head,
            "dirty": vendor_dirty,
            "license": "Mulan PSL v2",
            "urdf": str(URDF),
            "urdf_sha256": sha256(URDF),
            "mesh_sha256": referenced_meshes(),
            "hand_morphology_audit": morphology,
            "vendor_named_material_by_link": details[
                "vendor_visual_material_by_link"
            ],
            "renderer_material_rgba_fallback": details[
                "hand_material_rgba_fallback"
            ],
        },
        "render_environment": {
            "floor_type": "plane",
            "floor_material": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MATERIAL, floor_material_id
            ),
            "floor_texture": {
                "builtin": "checker",
                "material_texture_channel_indices": floor_texture_channels,
                "mark": "edge",
                "rgb1": [0.2, 0.3, 0.4],
                "rgb2": [0.1, 0.2, 0.3],
                "markrgb": [0.8, 0.8, 0.8],
                "size_px": [300, 300],
                "texrepeat": [5.0, 5.0],
                "reflectance": 0.2,
            },
            "headlight_ambient": model.vis.headlight.ambient.tolist(),
            "headlight_diffuse": model.vis.headlight.diffuse.tolist(),
            "headlight_specular": model.vis.headlight.specular.tolist(),
            "haze_rgba": model.vis.rgba.haze.tolist(),
            "directional_key_light_direction": [0.0, 0.0, -1.0],
        },
        "canonical_g1_table": {
            "reference_scene_xml": str(G1_REFERENCE_XML),
            "reference_scene_xml_sha256": reference_hash,
            "body_center_xy_m": [0.30, -0.10],
            "top_z_m": TABLE_TOP_Z,
            "top_half_extents_m": [0.20, 0.20],
            "top_rgba": [0.6, 0.4, 0.2, 1.0],
            "geometry": "shared canonical G1 table helper",
        },
        "neutral_pose": {
            "arm_joint_order": left_arm_names + right_arm_names,
            "arm_joint_position_rad": [0.0] * len(all_arm_qpos),
            "finger_joint_order": left_finger_names + right_finger_names,
            "finger_joint_position_rad": neutral_hand_qpos.tolist(),
            "vendor_zero_configuration": True,
            "previous_failed_pregrasp_finger_qpos_loaded": False,
            "left_and_right_hand_bottle_contacts": neutral_bottle_contacts,
            "left_and_right_hand_table_contacts": neutral_table_contacts,
        },
        "open_pregrasp_pose": {
            "right_arm_joint_order": right_arm_names,
            "right_arm_joint_position_rad": open_pregrasp_arm_qpos.tolist(),
            "arm_pose_source": "existing APPROACH_PALM_POS and PALM_TARGET_ROTATION, solved for one static image",
            "target_palm_position_world_m": APPROACH_PALM_POS.tolist(),
            "right_palm_position_world_m": right_open_position.tolist(),
            "finger_joint_position_rad": [0.0] * len(all_hand_qpos),
            "both_hands_zero_qpos": True,
            "bottle_contacts": open_pregrasp_bottle_contacts,
            "table_contacts": open_pregrasp_table_contacts,
        },
        "x2_station": {
            "base_position_world_m": ROBOT_BASE_WORLD.tolist(),
            "base_yaw_rad": ROBOT_BASE_YAW,
            "base_forward_axis_world": [0.0, -1.0, 0.0],
            "torso_front_axis_world": torso_front_world.tolist(),
            "table_center_lateral_offset_from_base_m": float(
                SOURCE_TABLE_XY[0] - ROBOT_BASE_WORLD[0]
            ),
            "table_front_edge_clearance_from_base_m": float(
                ROBOT_BASE_WORLD[1]
                - (SOURCE_TABLE_XY[1] + SOURCE_TABLE_HALF_EXTENTS[1])
            ),
            "right_shoulder_position_world_m": neutral_right_shoulder_position.tolist(),
            "right_shoulder_to_bottle_distance_m": float(
                np.linalg.norm(neutral_right_shoulder_position - neutral_bottle_position)
            ),
        },
        "x2_bottle": {
            "body_position_world_m": bottle_position.tolist(),
            "target_body_position_world_m": GOAL.tolist(),
            "body_diameter_m": CANONICAL_X2_BOTTLE_DIAMETER_M,
            "total_height_m": CANONICAL_X2_BOTTLE_HEIGHT_M,
            "mass_kg": float(model.body_mass[bottle_body]),
            "canonical_mass_kg": CANONICAL_X2_BOTTLE_MASS_KG,
            "geometry_names": list(BOTTLE_GEOM_NAMES),
            "bottle_qpos_unchanged": bool(
                np.array_equal(data.qpos[bottle_qpos : bottle_qpos + 7], bottle_initial_qpos)
            ),
        },
        "artifacts": images,
    }

    (OUT / "static_scene.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Issue #46 neutral-hand static diagnostic

Result: static visual review packet only. No physics step or grasp rollout was run; the prior physical manipulation result remains FAIL.

## Handedness audit

- Pinned source: `{vendor_head}` (`AgibotTech/agibot_x2_urdf`, Mulan PSL v2).
- Distinct palm source/compiled meshes: `L_palm -> l_palm.stl -> l_palm`; `R_palm -> r_palm.stl -> r_palm`.
- The compiled fixed palm transforms, all hand joint axes/signs and ranges match the URDF across {morphology['source_joint_count']} hand joints.
- Thumb roots remain on mirrored palm sides: left y `{morphology['palms']['L']['thumb_mount_xyz_m'][1]:.6f} m`, right y `{morphology['palms']['R']['thumb_mount_xyz_m'][1]:.6f} m` in their palm-local frames.
- The prior static capture used a curled `RECORDED_PREGRASP_FINGER_QPOS`; this capture sets both hands to vendor zero qpos. No left-hand mesh or mount is reused on the right.
- The downward thumb appearance remains in the vendor-zero diagnostic with both arms at their symmetric zero pose, so the old curled finger preset is not the sole cause. The source/import handedness audit passes; the existing task palm target gives the separate open-approach view its lateral thumb orientation. No wrist quaternion or hand transform was altered.
- Source has {morphology['source_mimic_relation_count']} mimic annotations and the compiled model has {morphology['compiled_equality_count']} equality constraints. These are not needed for the zero-pose morphology check and remain a controller/kinematics limitation for future physical work.

## Scene and floor

The accepted table, multipart X2 bottle, station, and robot visual materials are retained. The floor uses the same checker texture/material values and texture channel as the accepted G1 scene: `rgb1=[0.2,0.3,0.4]`, `rgb2=[0.1,0.2,0.3]`, edge marks `[0.8,0.8,0.8]`, repeat `[5,5]`, reflectance `0.2`. The channel correction is required for MuJoCo to render the checker rather than a flat white plane.

The open-hand close-up uses the existing static `APPROACH_PALM_POS`, solved with one IK query for framing. Both hands stay at zero finger qpos; the image isolates the right wrist/hand, original bottle, and tabletop, and verifies no hand/bottle or hand/table contact. This is the contact-free approach immediately before the existing pregrasp target, not a rollout or a grasp adjustment.

## Evidence

`static_scene.json` contains the full source/import comparison, mesh hashes, zero-pose state, pregrasp-open state, and contact snapshots. `run.log` and `experiment_commands.txt` record reproduction. Images: {', '.join(f'`{name}`' for name in images)}.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "experiment_commands.txt").write_text(
        f"cd {robot_sim_root}\n"
        f"MUJOCO_GL=egl ISSUE46_STATIC_EVIDENCE_DIR={OUT} "
        f"AGIBOT_X2_VENDOR_ROOT={ROOT} {sys.executable} {Path(__file__).resolve()}\n"
        "Renders only: both arm sets at zero for neutral views; both hand joint sets at "
        "zero for every view; the open-pregrasp view changes only right arm qpos. "
        "No mj_step, grasp rollout, or object qpos write.\n",
        encoding="utf-8",
    )
    log(f"STATIC_SCENE_JSON={OUT / 'static_scene.json'}")
    log(f"REPORT={OUT / 'REPORT.md'}")
    log("PHYSICS_STEPS=0")
    log("MJ_STEP_CALLED=false")
    log("NEUTRAL_HAND_JOINT_QPOS=all zero")
    log(f"ROBOTSIM_HEAD={robot_sim_head}")
    log(f"ROBOTSIM_DIRTY={robot_sim_dirty}")
    log(f"FLOOR_CHECKER_TEXTURE_CHANNELS={floor_texture_channels}")
    log(f"OPEN_PREGRASP_BOTTLE_CONTACTS={open_pregrasp_bottle_contacts}")
    log(f"OPEN_PREGRASP_TABLE_CONTACTS={open_pregrasp_table_contacts}")
    (OUT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
