#!/usr/bin/env python3
"""Open a zero-step MuJoCo visual review of the jaw-centered STL grasp scan.

The slider selects only measured static samples.  At every command it applies
the arm IK result that centers the measured narrow3/wide3 inner-face midpoint
on the vertical cylinder; it is not a physics rollout and never calls mj_step.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    apply_stl_kinematic_pose,
    load_registered_trajectory,
)


MODEL_ID = "X2-OmniPicker-STL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_static_module():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("stl_static", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load STL static helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_contacts(row: dict[str, object]) -> list[str]:
    names: set[str] = set()
    for contact in row["contacts"]:
        for name in contact["body_names"]:
            if name.startswith("R_hand_"):
                names.add(name)
    return sorted(names)


def build_review(source_scene: Path, alignment_result: Path, prior_config: Path) -> Path:
    result = json.loads(alignment_result.read_text(encoding="utf-8"))
    if sha256(source_scene) != result["source_scene"]["sha256"]:
        raise RuntimeError("source scene hash differs from jaw-center evidence")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if result["model"]["sha256"] != loaded.model_sha256:
        raise RuntimeError("frozen STL model hash differs from jaw-center evidence")
    samples = {}
    for row in result["rows"]:
        command = int(row["command_percent"])
        samples[str(command)] = {
            "arm_q_rad": row["ik"]["arm_q_rad"],
            "jaw_center_error_m": row["jaw_center_error_m"],
            "contact_bodies": summarize_contacts(row),
        }
    commands = sorted(int(key) for key in samples)
    if commands != list(range(commands[0], commands[-1] + 1)):
        raise RuntimeError("jaw-center evidence must contain consecutive integer slider samples")
    static = load_static_module()
    state = static.state_by_phase(json.loads(prior_config.read_text(encoding="utf-8")), "HORIZONTAL_APPROACH")
    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-centered-grasp-visual-review")
    output = location.native_path
    review = {
        "schema": "robotsim.issue46.stl.centered-grasp-visual-review.v1",
        "mode": "STATIC_MJ_FORWARD_ONLY",
        "mj_step_calls": 0,
        "scene": {"path": str(source_scene), "sha256": sha256(source_scene)},
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "alignment_result": {"path": str(alignment_result), "sha256": sha256(alignment_result)},
        "prior_config": {"path": str(prior_config), "sha256": sha256(prior_config), "phase": "HORIZONTAL_APPROACH"},
        "state": state,
        "samples": samples,
        "command_range_percent": [commands[0], commands[-1]],
        "recommended_samples": {"first_dual_contact": 40, "maximum_without_loop_contact": 35, "first_loop_contact": 34},
        "limitations": [
            "Static geometry review only; no mj_step, contact force, load transfer, or grasp claim.",
            "Visual STL and collision-proxy groups can be toggled; this does not alter physics masks or model parameters.",
        ],
    }
    pg.write_new_text(output / "review_state.json", json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "review_state": str(output / "review_state.json"), "mj_step_calls": 0}, ensure_ascii=False, indent=2))
    return output / "review_state.json"


def open_review(review_state: Path, *, dry_run: bool, live_viewer: bool) -> None:
    review = json.loads(review_state.read_text(encoding="utf-8"))
    scene_path = Path(review["scene"]["path"])
    if sha256(scene_path) != review["scene"]["sha256"]:
        raise RuntimeError("review scene hash differs from saved review state")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if loaded.model_sha256 != review["model"]["sha256"]:
        raise RuntimeError("frozen STL model hash differs from saved review state")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    trajectory = load_registered_trajectory(loaded.package_path)
    static = load_static_module()
    commands = sorted(int(key) for key in review["samples"])

    def apply_command(data: mujoco.MjData, command: int) -> list[str]:
        sample = review["samples"][str(command)]
        arm = [float(sample["arm_q_rad"][name]) for name in static.ARM_JOINTS]
        motor_override = review.get("motor_override_rad")
        if motor_override is not None and not isinstance(motor_override, dict):
            raise RuntimeError("motor_override_rad must be a joint-to-angle mapping")
        static.apply_prior_state(model, data, review["state"], arm, motor_override)
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command))
        mujoco.mj_forward(model, data)
        contacts = static.classify_contacts(model, data)
        names = set(contacts["pad_bodies"])
        for item in contacts["nonpad_right_contacts"]:
            for name in item["body_names"]:
                if name.startswith("R_hand_"):
                    names.add(name)
        return sorted(names)

    if dry_run:
        data = mujoco.MjData(model)
        initial_command = int(review.get("initial_command_percent", 35))
        contacts = apply_command(data, initial_command)
        print(json.dumps({"model": MODEL_ID, "command_percent": initial_command, "contact_bodies": contacts, "mj_step_calls": 0, "nq": model.nq}, ensure_ascii=False, indent=2))
        return
    if not live_viewer:
        return
    import glfw

    if not glfw.init():
        raise RuntimeError("could not initialize GLFW")
    window = glfw.create_window(1320, 930, str(review.get("viewer_title", "MuJoCo | X2-OmniPicker-STL | centered static grasp review")), None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create MuJoCo GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=12000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera_config = review.get("camera", {})
    if not isinstance(camera_config, dict):
        raise RuntimeError("camera must be an object when supplied")
    camera.lookat[:] = camera_config.get("lookat", [1.43, -1.14, 0.76])
    camera.distance = float(camera_config.get("distance", 1.25))
    camera.azimuth = float(camera_config.get("azimuth", 160.0))
    camera.elevation = float(camera_config.get("elevation", -18.0))
    ui = {"command": int(review.get("initial_command_percent", 35)), "previous": None, "left": False, "right": False, "middle": False, "slider": False, "lastx": 0.0, "lasty": 0.0, "render_mode": 0, "show_contacts": True}

    def mode_name() -> str:
        option.geomgroup[2] = 1 if ui["render_mode"] in (0, 2) else 0
        option.geomgroup[3] = 1 if ui["render_mode"] in (1, 2) else 0
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(ui["show_contacts"])
        return ("VISUAL_STL", "COLLISION_PROXY", "OVERLAY")[ui["render_mode"]]

    def slider_value(x: float) -> None:
        width, _ = glfw.get_window_size(window)
        low, high = review["command_range_percent"]
        fraction = min(1.0, max(0.0, (x - 45.0) / max(1.0, width - 90.0)))
        ui["command"] = int(round(low + fraction * (high - low)))

    def cursor_callback(_window, x, y) -> None:
        if ui["slider"]:
            slider_value(x)
        elif ui["left"] or ui["right"] or ui["middle"]:
            _, height = glfw.get_window_size(window)
            dx, dy = float(x - ui["lastx"]) / max(1, height), float(y - ui["lasty"]) / max(1, height)
            shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if ui["right"] and shift else mujoco.mjtMouse.mjMOUSE_MOVE_V if ui["right"] else mujoco.mjtMouse.mjMOUSE_ROTATE_H if ui["left"] and shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V if ui["left"] else mujoco.mjtMouse.mjMOUSE_ZOOM
            mujoco.mjv_moveCamera(model, action, dx, dy, scene, camera)
        ui["lastx"], ui["lasty"] = x, y

    def mouse_callback(_window, button, action, _mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        _, height = glfw.get_window_size(window)
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and height - 63 <= y <= height - 35:
            ui["slider"] = True
            slider_value(x)
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            ui["left"] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            ui["right"] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            ui["middle"] = action == glfw.PRESS
        if action == glfw.RELEASE:
            ui["slider"] = False
        ui["lastx"], ui["lasty"] = x, y

    def scroll_callback(_window, _xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * float(yoffset), scene, camera)

    def key_callback(_window, key, _scancode, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_LEFT:
            ui["command"] = max(commands[0], ui["command"] - 1)
        elif key == glfw.KEY_RIGHT:
            ui["command"] = min(commands[-1], ui["command"] + 1)
        elif key == glfw.KEY_C:
            ui["render_mode"] = (ui["render_mode"] + 1) % 3
        elif key == glfw.KEY_V:
            ui["show_contacts"] = not ui["show_contacts"]

    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    try:
        while not glfw.window_should_close(window):
            if ui["command"] != ui["previous"]:
                contacts = apply_command(data, ui["command"])
                ui["previous"] = ui["command"]
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            low, high = review["command_range_percent"]
            fraction = (ui["command"] - low) / max(1, high - low)
            track = mujoco.MjrRect(45, 44, max(1, width - 90), 14)
            mujoco.mjr_rectangle(track, 0.13, 0.17, 0.22, 0.95)
            mujoco.mjr_rectangle(mujoco.MjrRect(45, 44, int(track.width * fraction), 14), 0.20, 0.90, 0.55, 1.0)
            marker = int(45 + track.width * fraction)
            mujoco.mjr_rectangle(mujoco.MjrRect(marker - 4, 38, 8, 26), 0.97, 0.70, 0.18, 1.0)
            primary = str(review.get("overlay_title", "X2-OmniPicker-STL | centered static review")) + " | 0 mj_step | " + mode_name()
            metric_text = str(review.get("metric_text", ""))
            secondary = "command %d%% | contact: %s%s" % (
                ui["command"],
                ", ".join(contacts) if contacts else "none",
                " | " + metric_text if metric_text else "",
            )
            footer = str(review.get("footer", "30 closed / 45 open: drag slider or Left/Right | C visual/collision/overlay | V contact markers | left rotate, right pan, wheel zoom"))
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, primary, secondary, context)
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, footer, "", context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(0.005)
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path)
    parser.add_argument("--alignment-result", type=Path)
    parser.add_argument("--prior-config", type=Path)
    parser.add_argument("--review-state", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()
    if args.review_state is not None:
        open_review(args.review_state, dry_run=args.dry_run, live_viewer=args.live_viewer)
        return
    if args.source_scene is None or args.alignment_result is None or args.prior_config is None:
        parser.error("provide --review-state, or source scene, alignment result, and prior config")
    review_state = build_review(args.source_scene, args.alignment_result, args.prior_config)
    if args.dry_run or args.live_viewer:
        open_review(review_state, dry_run=args.dry_run, live_viewer=args.live_viewer)


if __name__ == "__main__":
    main()
