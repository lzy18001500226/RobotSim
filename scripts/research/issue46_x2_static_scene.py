from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from issue46_x2_grasp import (
    ARM,
    CANONICAL_X2_BOTTLE_DIAMETER_M,
    CANONICAL_X2_BOTTLE_HEIGHT_M,
    CANONICAL_X2_BOTTLE_MASS_KG,
    DT,
    FINGERS,
    GOAL,
    PALM_TARGET_ROTATION,
    PREGRASP_PALM_POS,
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
RECORDED_PREGRASP_ARM_QPOS = np.array(
    [
        0.550040025346196,
        0.05276293126895596,
        0.8097892019415919,
        -2.150936646865968,
        0.2736775191639196,
        0.10030415147233902,
        -0.6420269029202836,
    ]
)
RECORDED_PREGRASP_FINGER_QPOS = np.array(
    [
        0.7415676248750857,
        -1.1537244409957665,
        0.3203594617927349,
        0.4434472037472062,
        0.3909104916999252,
        -0.1483972443985791,
        0.44958153886365904,
        0.5082140331328054,
        0.44957345599527765,
        0.5082104906646138,
        0.13075661244048462,
        0.44956772851595134,
        0.5082073949631831,
        0.14644678774669564,
        0.44956449966135636,
        0.5082070804762859,
    ]
)
RECORDED_PREGRASP_SOURCE = (
    "/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/"
    "canonical-scene/20261005-final-visual/frame_diagnostics.json"
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

    log("EXPERIMENT=RobotSim Issue #46 static X2 manipulation scene review")
    log(
        f"COMMAND=MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')} "
        f"ISSUE46_STATIC_EVIDENCE_DIR={OUT} {sys.executable} {Path(__file__).resolve()}"
    )
    log("MODE=static render only; no mj_step, no grasp rollout, no trajectory tuning")
    log("STATIC_JOINT_POSE=recorded collision-free pregrasp arm and finger state; no solver or stepping")

    model, details = build_model()
    data = details["data"]
    bottle_qpos = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free")
        ]
    )
    bottle_initial_qpos = data.qpos[bottle_qpos : bottle_qpos + 7].copy()
    _, arm_qpos = resolve_joints(model, ARM)
    _, finger_qpos = resolve_joints(model, list(FINGERS))
    static_arm_pose = RECORDED_PREGRASP_ARM_QPOS.copy()
    data.qpos[arm_qpos] = static_arm_pose
    data.qpos[finger_qpos] = RECORDED_PREGRASP_FINGER_QPOS.copy()
    mujoco.mj_forward(model, data)
    if not np.array_equal(data.qpos[bottle_qpos : bottle_qpos + 7], bottle_initial_qpos):
        raise RuntimeError("Static arm-pose setup changed the bottle generalized position")

    render_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(render_option)
    render_option.geomgroup[0] = 0
    render_option.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=720, width=1280)
    view_center = np.array([0.30, 0.03, 0.70])
    views = {
        "canonical_overview.png": camera(view_center, 2.35, 135.0, -17.0),
        "front_overview.png": camera(view_center, 2.45, 90.0, -12.0),
        "side_alignment.png": camera(view_center, 2.55, 180.0, -10.0),
        "hand_material_closeup.png": camera(
            (0.30, 0.10, float(START[2])), 0.72, 135.0, -10.0
        ),
    }
    for filename, view_camera in views.items():
        renderer.update_scene(data, camera=view_camera, scene_option=render_option)
        imageio.imwrite(OUT / filename, renderer.render())
        log(f"IMAGE={OUT / filename}")
    renderer.close()

    bottle_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    right_shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
    )
    torso_front_world = (
        data.xmat[torso_body].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
    )
    bottle_position = data.qpos[bottle_qpos : bottle_qpos + 3].copy()
    palm_position = data.site_xpos[details["site_id"]].copy()
    palm_rotation = data.site_xmat[details["site_id"]].reshape(3, 3)
    relative_rotation = PALM_TARGET_ROTATION.T @ palm_rotation
    palm_orientation_error = float(
        np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
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

    result = {
        "result": "STATIC_SCENE_ONLY",
        "manipulation_result": "FAIL_REMAINS_FROM_PRIOR_RUN; no physical rollout rerun",
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "timestep_s": DT,
            "renderer": "MuJoCo EGL offscreen",
            "physics_steps": 0,
        },
        "robot_sim": {
            "root": str(robot_sim_root),
            "head_before_commit": robot_sim_head,
            "dirty_during_capture": robot_sim_dirty,
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
            "vendor_scene_xml": str(URDF.parents[0] / "scene.xml"),
            "vendor_robot_mjcf": str(URDF.parents[0] / "X2-Ultra.xml"),
            "vendor_visual_rgba_values": details["vendor_visual_color_values"],
            "vendor_named_material_by_link": details[
                "vendor_visual_material_by_link"
            ],
            "vendor_named_material_numeric_definitions": False,
            "renderer_material_rgba_fallback": details[
                "hand_material_rgba_fallback"
            ],
            "mapped_hand_visual_geom_count": len(
                details["hand_material_geom_assignments"]
            ),
            "unmatched_hand_material_links": details[
                "unmatched_hand_material_links"
            ],
        },
        "render_environment": {
            "floor_geom_rgba": model.geom_rgba[floor_geom_id].tolist(),
            "floor_geom_material_id": int(model.geom_matid[floor_geom_id]),
            "headlight_ambient": model.vis.headlight.ambient.tolist(),
            "headlight_diffuse": model.vis.headlight.diffuse.tolist(),
            "headlight_specular": model.vis.headlight.specular.tolist(),
            "haze_rgba": model.vis.rgba.haze.tolist(),
            "skybox_gradient_rgb1": [0.30, 0.50, 0.70],
            "skybox_gradient_rgb2": [0.0, 0.0, 0.0],
            "directional_key_light_direction": [0.0, 0.0, -1.0],
        },
        "canonical_g1_table": {
            "reference_scene_xml": str(G1_REFERENCE_XML),
            "reference_scene_xml_sha256": reference_hash,
            "body_center_xy_m": [0.30, -0.10],
            "top_z_m": TABLE_TOP_Z,
            "top_half_extents_m": [0.20, 0.20],
            "top_rgba": [0.6, 0.4, 0.2, 1.0],
            "geometry": "one tabletop plus four 25 mm square legs from shared canonical_manipulation_assets.py",
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
            "torso_front_dot_table_edge_axis": float(np.dot(torso_front_world, [1, 0, 0])),
            "right_shoulder_position_world_m": data.xpos[right_shoulder].tolist(),
            "right_shoulder_to_bottle_distance_m": float(
                np.linalg.norm(data.xpos[right_shoulder] - bottle_position)
            ),
            "static_palm_target_position_world_m": PREGRASP_PALM_POS.tolist(),
            "static_palm_position_world_m": palm_position.tolist(),
            "static_palm_position_error_m": float(
                np.linalg.norm(palm_position - PREGRASP_PALM_POS)
            ),
            "static_palm_orientation_error_rad": palm_orientation_error,
            "static_joint_pose_source": RECORDED_PREGRASP_SOURCE,
            "static_arm_joint_order": ARM,
            "static_arm_joint_position_rad": static_arm_pose.tolist(),
            "static_finger_joint_order": list(FINGERS),
            "static_finger_joint_position_rad": RECORDED_PREGRASP_FINGER_QPOS.tolist(),
            "static_hand_bottle_contacts": bottle_hand_contacts(model, data),
            "static_hand_table_contacts": hand_table_contacts(model, data),
        },
        "x2_bottle": {
            "body_position_world_m": bottle_position.tolist(),
            "target_body_position_world_m": GOAL.tolist(),
            "body_diameter_m": CANONICAL_X2_BOTTLE_DIAMETER_M,
            "total_height_m": CANONICAL_X2_BOTTLE_HEIGHT_M,
            "mass_kg": float(model.body_mass[bottle_body]),
            "canonical_mass_kg": CANONICAL_X2_BOTTLE_MASS_KG,
            "geometry_names": list(BOTTLE_GEOM_NAMES),
            "bottle_qpos_unchanged_by_static_joint_pose": True,
        },
        "artifacts": list(views),
    }
    (OUT / "static_scene.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "experiment_commands.txt").write_text(
        f"cd {robot_sim_root}\n"
        "MUJOCO_GL=egl ISSUE46_STATIC_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/"
        "reviews/issue-46-x2-single-hand/static-scene/20261005-final-review/ "
        f"{sys.executable} {Path(__file__).resolve()}\n"
        "Recorded pregrasp arm/finger joint state only; mj_forward for rendering; "
        "no mj_step, grasp rollout, or trajectory tuning.\n",
        encoding="utf-8",
    )
    log(f"STATIC_SCENE_JSON={OUT / 'static_scene.json'}")
    log("PHYSICS_STEPS=0")
    log("MJ_STEP_CALLED=false")
    log(f"STATIC_HAND_BOTTLE_CONTACTS={result['x2_station']['static_hand_bottle_contacts']}")
    log(f"STATIC_HAND_TABLE_CONTACTS={result['x2_station']['static_hand_table_contacts']}")
    (OUT / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
