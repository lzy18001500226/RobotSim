#!/usr/bin/env python3
"""Render a zero-step, level side-pregrasp of the frozen STL OmniPicker.

This is a human visual-review aid.  It starts from a recorded native standing
state, places the full robot at one already collision-free static table-docking
candidate, and solves only existing waist/right-arm joints.  The solve enforces
the actual STL jaw centre, a level forearm, and two distal links at equal
height.  It does not command a simulator, step physics, or claim a grasp.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import sys

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    load_registered_trajectory,
)


WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
UPPER = WAIST + ARM
DISTAL = ("R_hand_narrow3_Link", "R_hand_wide3_Link")
KNOWN_FROZEN_INTERNAL = {
    frozenset(pair)
    for pair in (
        ("left_hip_roll_link", "L_hand_narrow1_Link"),
        ("left_hip_roll_link", "L_hand_narrow2_Link"),
        ("left_hip_roll_link", "L_hand_narrow3_Link"),
        ("left_hip_roll_link", "L_hand_narrow_loop_Link"),
        ("left_hip_yaw_link", "L_hand_narrow3_Link"),
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def joint_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if value < 0:
        raise ValueError(f"scene lacks joint: {name}")
    return value


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise ValueError(f"scene lacks body: {name}")
    return value


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = joint_id(model, name)
    data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[model.jnt_dofadr[joint]] = 0.0


def trajectory_joint_name(side: str, source_name: str) -> str:
    if source_name == "right_claw_joint":
        return "R_claw_joint" if side == "right" else "L_claw_joint"
    if side == "left" and source_name.startswith("R_"):
        return "L_" + source_name[2:]
    return source_name


def apply_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    standing: dict[str, object],
    upper: np.ndarray,
    dock_shift: np.ndarray,
    command_percent: int,
) -> None:
    base = standing["base_pose_world"]
    if not isinstance(base, dict):
        raise ValueError("standing state lacks base pose")
    free = joint_id(model, "floating_base_joint")
    address = int(model.jnt_qposadr[free])
    data.qpos[address:address + 3] = np.asarray(base["xyz_m"], dtype=float) + dock_shift
    data.qpos[address + 3:address + 7] = np.asarray(base["quaternion_wxyz"], dtype=float)
    for group_name in ("leg_positions_rad", "waist_positions_rad", "arm_positions_rad"):
        group = standing.get(group_name)
        if not isinstance(group, dict):
            raise ValueError(f"standing state lacks {group_name}")
        for name, value in group.items():
            set_joint(model, data, str(name), float(value))
    for name, value in zip(UPPER, upper):
        set_joint(model, data, name, float(value))
    trajectory = load_registered_trajectory(REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic")
    pose = trajectory.pose(command_percent)
    for side in ("right", "left"):
        for name, value in pose.positions_rad.items():
            set_joint(model, data, trajectory_joint_name(side, name), float(value))
    mujoco.mj_forward(model, data)


def point_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body: int, dofs: np.ndarray) -> np.ndarray:
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, linear, angular, body)
    return linear[:, dofs]


def solve_level_pregrasp(
    model: mujoco.MjModel,
    standing: dict[str, object],
    dock_shift: np.ndarray,
    command_percent: int,
    orientation,
    seed_right_arm: np.ndarray,
    *,
    max_iterations: int = 300,
) -> tuple[np.ndarray, dict[str, float]]:
    dofs = np.asarray([int(model.jnt_dofadr[joint_id(model, name)]) for name in UPPER], dtype=int)
    limits = np.asarray([model.jnt_range[joint_id(model, name)] for name in UPPER], dtype=float)
    arm = standing["arm_positions_rad"]
    waist = standing["waist_positions_rad"]
    assert isinstance(arm, dict) and isinstance(waist, dict)
    if seed_right_arm.shape != (len(ARM),) or not np.isfinite(seed_right_arm).all():
        raise ValueError("right-arm seed must contain seven finite joint values")
    seed = np.asarray([float(waist[name]) for name in WAIST] + seed_right_arm.tolist(), dtype=float)
    q = seed.copy()
    elbow, wrist, narrow, wide, cylinder = (body_id(model, name) for name in ("right_elbow_link", "right_wrist_roll_link", *DISTAL, "test_cylinder"))
    metrics: dict[str, float] = {}
    for iteration in range(1, max_iterations + 1):
        data = mujoco.MjData(model)
        apply_pose(model, data, standing, q, dock_shift, command_percent)
        centre_error = data.xpos[cylinder] - orientation.jaw_center(model, data)
        forearm_delta = float(data.xpos[wrist, 2] - data.xpos[elbow, 2])
        distal_delta = float(data.xpos[narrow, 2] - data.xpos[wide, 2])
        metrics = {
            "jaw_center_error_m": float(np.linalg.norm(centre_error)),
            "forearm_height_delta_m": forearm_delta,
            "distal_height_delta_m": distal_delta,
            "minimum_upper_joint_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
        }
        if (
            metrics["jaw_center_error_m"] <= 1.0e-4
            and abs(forearm_delta) <= 1.0e-4
            and abs(distal_delta) <= 1.0e-4
            and metrics["minimum_upper_joint_margin_rad"] >= 1.0e-3
        ):
            break
        elbow_jac = point_jacobian(model, data, elbow, dofs)
        wrist_jac = point_jacobian(model, data, wrist, dofs)
        narrow_jac = point_jacobian(model, data, narrow, dofs)
        wide_jac = point_jacobian(model, data, wide, dofs)
        error = np.concatenate((centre_error, (-forearm_delta,), (-distal_delta,)))
        jacobian = np.vstack((
            orientation.jaw_jacobian(model, data)[:, dofs],
            wrist_jac[2] - elbow_jac[2],
            narrow_jac[2] - wide_jac[2],
        ))
        update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1.0e-5 * np.eye(5), error)
        update -= 2.0e-4 * (q - seed)
        q += np.clip(update, -0.025, 0.025)
        q = np.clip(q, limits[:, 0] + 1.0e-5, limits[:, 1] - 1.0e-5)
    data = mujoco.MjData(model)
    apply_pose(model, data, standing, q, dock_shift, command_percent)
    metrics = {
        "iterations": float(iteration),
        "jaw_center_error_m": float(np.linalg.norm(data.xpos[cylinder] - orientation.jaw_center(model, data))),
        "forearm_height_delta_m": float(data.xpos[wrist, 2] - data.xpos[elbow, 2]),
        "distal_height_delta_m": float(data.xpos[narrow, 2] - data.xpos[wide, 2]),
        "minimum_upper_joint_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
    }
    return q, metrics


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    rows = []
    for contact in data.contact:
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "world"
            for geom in (contact.geom1, contact.geom2)
        )
        rows.append({"body_names": names, "distance_m": float(contact.dist)})
    return rows


def is_expected_contact(pair: tuple[str, str]) -> bool:
    values = frozenset(pair)
    return "world" in values or values == frozenset(("test_cylinder", "raicom_table_table")) or values in KNOWN_FROZEN_INTERNAL


def configure_camera_callbacks(window, camera, glfw) -> None:
    button_left = button_middle = button_right = False
    last_x = last_y = 0.0

    def mouse_button(_window, button, action, _mods):
        nonlocal button_left, button_middle, button_right, last_x, last_y
        button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        last_x, last_y = glfw.get_cursor_pos(window)

    def mouse_move(_window, x, y):
        nonlocal last_x, last_y
        dx, dy = x - last_x, y - last_y
        last_x, last_y = x, y
        if not (button_left or button_middle or button_right):
            return
        width, height = glfw.get_window_size(window)
        if height <= 0:
            return
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        if button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(None, action, dx / height, dy / height, None, camera)

    def scroll(_window, _xoffset, yoffset):
        mujoco.mjv_moveCamera(None, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, None, camera)

    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-result", required=True, type=Path)
    parser.add_argument("--placement-result", required=True, type=Path)
    parser.add_argument("--stand-result", required=True, type=Path)
    parser.add_argument("--seed-ik-result", required=True, type=Path, help="saved position-only IK result used solely as a continuous arm seed")
    parser.add_argument("--candidate-index", type=int, default=6)
    parser.add_argument("--command-percent", type=int, default=100)
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.command_percent <= 100:
        raise SystemExit("command percent must be in 0..100")
    table = json.loads(args.table_result.read_text(encoding="utf-8"))
    placement = json.loads(args.placement_result.read_text(encoding="utf-8"))
    stand = json.loads(args.stand_result.read_text(encoding="utf-8"))
    seed_result = json.loads(args.seed_ik_result.read_text(encoding="utf-8"))
    candidates = placement.get("scan", {}).get("best_candidates", [])
    if not isinstance(candidates, list) or not 0 <= args.candidate_index < len(candidates):
        raise SystemExit("candidate index is outside the saved collision-free table placements")
    if stand.get("gates", {}).get("POST_RESET_STAND_STABILITY") != "PASS":
        raise SystemExit("native standing evidence did not pass")
    standing = stand.get("native_arm_state_capture")
    if not isinstance(standing, dict):
        raise SystemExit("native standing result lacks the arm-state capture")
    seed_right_arm = np.asarray(seed_result.get("ik", {}).get("right_arm_q_rad", []), dtype=float)
    if seed_result.get("gates", {}).get("IK_CONVERGED") != "PASS" or seed_right_arm.shape != (len(ARM),) or not np.isfinite(seed_right_arm).all():
        raise SystemExit("seed result does not contain a passing seven-joint right-arm position IK reference")
    scene = Path(str(table.get("generated", {}).get("host_scene", {}).get("path", "")))
    expected_scene_hash = str(table.get("generated", {}).get("host_scene", {}).get("sha256", ""))
    if not scene.is_file() or sha256(scene) != expected_scene_hash:
        raise SystemExit("table scene is missing or differs from its evidence hash")
    candidate = candidates[args.candidate_index]
    fixed_table = np.asarray(table["transforms"]["table_world_override"], dtype=float)
    dock_shift = fixed_table - np.asarray(candidate["table_world_xyz_m"], dtype=float)
    orientation = load_module("stl_orientation", REPO / "scripts/research/issue46/grasp/centered_orientation_search.py")
    model = mujoco.MjModel.from_xml_path(str(scene))
    upper, metrics = solve_level_pregrasp(model, standing, dock_shift, args.command_percent, orientation, seed_right_arm)
    data = mujoco.MjData(model)
    apply_pose(model, data, standing, upper, dock_shift, args.command_percent)
    contacts = contact_records(model, data)
    forbidden = [row for row in contacts if not is_expected_contact(tuple(row["body_names"]))]
    output = pg.create_unique_output_directory(EVIDENCE / "reviews/issue-46", prefix="x2-stl-docked-level-pregrasp-review").native_path
    gates = {
        "NATIVE_STANDING_STATE": "PASS",
        "JAW_CENTER_ALIGNMENT": "PASS" if metrics["jaw_center_error_m"] <= 1.0e-4 else "FAIL",
        "FOREARM_LEVEL": "PASS" if abs(metrics["forearm_height_delta_m"]) <= 1.0e-4 else "FAIL",
        "DISTAL_LINK_LEVEL": "PASS" if abs(metrics["distal_height_delta_m"]) <= 1.0e-4 else "FAIL",
        "UPPER_JOINT_LIMITS": "PASS" if metrics["minimum_upper_joint_margin_rad"] >= 1.0e-3 else "FAIL",
        "STATIC_CONTACT_CHECK": "PASS" if not forbidden else "FAIL",
        "MUJOCO_DYNAMICS": "NOT_RUN",
        "PHYSICAL_GRASP": "NOT_RUN",
    }
    record = {
        "schema": "robotsim.issue46.x2-stl-docked-level-pregrasp-review.v1",
        "classification": "SIMULATION_ONLY_STATIC_LEVEL_SIDE_PREGRASP",
        "mj_step_calls": 0,
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "standing_result": {"path": str(args.stand_result), "sha256": sha256(args.stand_result)},
        "position_ik_seed_result": {"path": str(args.seed_ik_result), "sha256": sha256(args.seed_ik_result), "use": "continuous starting point only; final orientation comes from MuJoCo joint geometry"},
        "placement_result": {"path": str(args.placement_result), "sha256": sha256(args.placement_result), "candidate_index": args.candidate_index, "candidate": candidate},
        "dock_base_translation_world_m": dock_shift.tolist(),
        "upper_joint_q_rad": {name: float(value) for name, value in zip(UPPER, upper)},
        "right_hand_command_percent": args.command_percent,
        "left_hand_command_percent": args.command_percent,
        "metrics": metrics,
        "contacts": contacts,
        "known_frozen_internal_contacts": [row for row in contacts if frozenset(tuple(row["body_names"])) in KNOWN_FROZEN_INTERNAL],
        "forbidden_contacts": forbidden,
        "gates": gates,
        "limitations": [
            "Zero-step mj_forward visual review only; this is not evidence of walking, balance with arm motion, closure, contact forces, or a grasp.",
            "The dock translation is a static equivalent of a saved collision-free placement, not a measured locomotion endpoint.",
            "The frozen left-hand internal collision proxies remain visible and reported; no collision pair is disabled by this review.",
        ],
    }
    pg.write_new_text(output / "review_state.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "gates": gates, "metrics": metrics, "upper_joint_q_rad": record["upper_joint_q_rad"]}, ensure_ascii=False, indent=2))
    if not args.live_viewer:
        return
    import glfw
    if not glfw.init():
        raise RuntimeError("GLFW is unavailable")
    window = glfw.create_window(1400, 980, "MuJoCo | X2-OmniPicker-STL | level side-pregrasp (STATIC)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("cannot create a MuJoCo review window")
    glfw.make_context_current(window)
    camera = mujoco.MjvCamera()
    cylinder = body_id(model, "test_cylinder")
    camera.lookat[:] = (data.xpos[cylinder] + np.array((-0.15, 0.0, 0.10))).tolist()
    camera.distance, camera.azimuth, camera.elevation = 1.65, 142.0, -14.0
    option = mujoco.MjvOption()
    scene_view = mujoco.MjvScene(model, maxgeom=16000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    configure_camera_callbacks(window, camera, glfw)
    while not glfw.window_should_close(window):
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene_view)
        mujoco.mjr_render(viewport, scene_view, context)
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "X2-OmniPicker-STL | LEVEL SIDE PREGRASP | STATIC",
            "0 mj_step | left drag rotate | right drag pan | wheel zoom | no grasp claim",
            context,
        )
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)
    context.free()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
