#!/usr/bin/env python3
"""Build and inspect a full-X2 static STL OmniPicker review scene.

This research tool is intentionally static: it applies a recorded kinematic
hand state and calls mj_forward.  It never calls mj_step and never makes a
claim about physical grasp dynamics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import xml.etree.ElementTree as ET
import zlib

import mujoco
import numpy as np


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
REVIEW_LOOKAT = (1.42, -0.85, 0.98)
REVIEW_DISTANCE = 2.4
REVIEW_AZIMUTH = 135.0
REVIEW_ELEVATION = -16.0
RIGHT_HAND_BODIES = {
    "R_omnipicker_base_link",
    "R_hand_narrow1_Link",
    "R_hand_narrow2_Link",
    "R_hand_narrow3_Link",
    "R_hand_narrow4_Link",
    "R_hand_narrow_loop_Link",
    "R_hand_wide1_Link",
    "R_hand_wide2_Link",
    "R_hand_wide3_Link",
    "R_hand_wide4_Link",
    "R_hand_wide_loop_Link",
}
RIGHT_HAND_JOINTS = {
    "right_claw_joint",
    "R_hand_narrow2_joint",
    "R_hand_narrow3_joint",
    "R_hand_narrow_loop_joint",
    "R_hand_wide1_joint",
    "R_hand_wide2_joint",
    "R_hand_wide3_joint",
    "R_hand_wide_loop_joint",
}
STL_EXCLUSIONS = (
    ("R_hand_narrow1_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow2_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_narrow_loop_Link"),
    ("R_hand_wide1_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide2_Link", "R_hand_wide_loop_Link"),
    ("R_hand_wide3_Link", "R_hand_wide_loop_Link"),
    ("R_hand_narrow3_Link", "R_hand_wide3_Link"),
)
MESH_RENAMES = {
    "mesh_base_link": "base_link",
    "mesh_narrow1_Link": "narrow1_Link",
    "mesh_narrow2_Link": "narrow2_Link",
    "mesh_narrow3_Link": "narrow3_Link",
    "mesh_narrow_loop_Link": "narrow_loop_Link",
    "mesh_wide1_Link": "wide1_Link",
    "mesh_wide2_Link": "wide2_Link",
    "mesh_wide3_Link": "wide3_Link",
    "mesh_wide_loop_Link": "wide_loop_Link",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_with_parent(root: ET.Element, name: str) -> tuple[ET.Element, ET.Element]:
    for parent in root.iter():
        for child in parent.findall("body"):
            if child.get("name") == name:
                return parent, child
    raise RuntimeError(f"missing body in review source: {name}")


def remap_meshes(element: ET.Element) -> None:
    for geom in element.iter("geom"):
        mesh = geom.get("mesh")
        if mesh in MESH_RENAMES:
            geom.set("mesh", MESH_RENAMES[mesh])
        name = geom.get("name", "")
        if "__visual" in name:
            geom.set("group", "2")
        elif "__collision" in name:
            # Group changes rendering only; contype/conaffinity remain source STL collision values.
            geom.set("group", "3")


def references_right_hand(element: ET.Element) -> bool:
    return any(value in RIGHT_HAND_JOINTS or value in RIGHT_HAND_BODIES for value in element.attrib.values())


def compose_scene(base_scene: Path, stl_model: Path, output_scene: Path) -> dict[str, object]:
    tree = ET.parse(base_scene)
    root = tree.getroot()
    stl_root = ET.parse(stl_model).getroot()
    _, target = find_with_parent(root, "R_omnipicker_base_link")
    _, source = find_with_parent(stl_root, "R_omnipicker_base_link")
    original_pose = {key: target.get(key) for key in ("name", "pos", "quat", "childclass")}
    for child in list(target):
        target.remove(child)
    for child in list(source):
        cloned = copy.deepcopy(child)
        remap_meshes(cloned)
        target.append(cloned)
    for key, value in original_pose.items():
        if value is not None:
            target.set(key, value)

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for child in list(contact):
        if child.tag == "exclude" and child.get("body1") in RIGHT_HAND_BODIES and child.get("body2") in RIGHT_HAND_BODIES:
            contact.remove(child)
    for body1, body2 in STL_EXCLUSIONS:
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    # The source review scene used CAD equality and actuator overlays.  Neither
    # is part of the table-driven STL right hand and neither is needed for a
    # zero-step static review.
    equality = root.find("equality")
    if equality is not None:
        # The saved scene's equality set belongs to the historical CAD overlay.
        # Static STL review applies direct table poses and therefore retains none.
        root.remove(equality)
    for container_name in ("actuator", "sensor", "tendon"):
        container = root.find(container_name)
        if container is not None:
            for child in list(container):
                if references_right_hand(child):
                    container.remove(child)
            if not list(container):
                root.remove(container)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    ET.indent(tree, space="  ")
    pg.write_new_bytes(output_scene, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {
        "base_right_mount": original_pose,
        "right_hand_replacement": "X2-OmniPicker-STL only",
        "left_hand": "unchanged source-scene hand; no unreviewed STL mirror mapping was inferred",
        "right_internal_collision_exclusions": [list(pair) for pair in STL_EXCLUSIONS],
    }


def configure_viewer_groups(viewer, mode: str) -> None:
    viewer.opt.geomgroup[:] = 1
    if mode == "visual":
        viewer.opt.geomgroup[3] = 0
    elif mode == "collision":
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[1] = 1  # table/floor physical context
        viewer.opt.geomgroup[3] = 1


def encode_png(pixels: np.ndarray) -> bytes:
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("expected HxWxRGB uint8 pixels")
    rgb = pixels[:, :, :3].tobytes()
    height, width = pixels.shape[:2]
    rows = b"".join(b"\x00" + rgb[row * width * 3:(row + 1) * width * 3] for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    return payload + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


def render_static_png(model, data, *, closeup: bool = False) -> tuple[bytes | None, str | None]:
    try:
        renderer = mujoco.Renderer(model, width=640, height=480)
        camera = mujoco.MjvCamera()
        if closeup:
            hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
            camera.lookat[:] = data.xpos[hand_id]
            camera.distance = 0.42
            camera.azimuth = 58.0
            camera.elevation = -12.0
        else:
            camera.lookat[:] = REVIEW_LOOKAT
            camera.distance = REVIEW_DISTANCE
            camera.azimuth = REVIEW_AZIMUTH
            camera.elevation = REVIEW_ELEVATION
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1
        option.geomgroup[3] = 0
        renderer.update_scene(data, camera=camera, scene_option=option)
        payload = encode_png(renderer.render())
        renderer.close()
        return payload, None
    except Exception as exc:  # Rendering availability is environment-specific; model evidence remains valid.
        return None, f"{type(exc).__name__}: {exc}"


def check_scene(model, trajectory, command_percent: int) -> dict[str, object]:
    data = mujoco.MjData(model)
    pose = apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command_percent))
    joint_values = {}
    for name, value in pose.positions_rad.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_values[name] = float(data.qpos[model.jnt_qposadr[joint_id]])
    contacts = []
    right_hand_contacts = []
    for contact in data.contact:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        record = {"geom1": geom1, "geom2": geom2, "distance_m": float(contact.dist)}
        contacts.append(record)
        if (geom1 or "").startswith("R_hand") or (geom2 or "").startswith("R_hand"):
            right_hand_contacts.append(record)
    return {
        "command_percent": command_percent,
        "right_stl_joint_qpos_rad": joint_values,
        "static_contact_count": int(data.ncon),
        "right_stl_hand_contact_count": len(right_hand_contacts),
        "right_stl_hand_contacts": right_hand_contacts,
        "static_contacts": contacts,
        "finite_qpos": bool(np.isfinite(data.qpos).all()),
        "finite_qvel": bool(np.isfinite(data.qvel).all()),
    }


def run_live_viewer(model, trajectory) -> None:
    """Open a native MuJoCo-rendered GLFW window with a direct 0–100 slider.

    The render loop never calls mj_step.  Slider changes only apply the
    registered table pose and mj_forward, so this remains a static kinematic
    review rather than a physics rollout.
    """
    import glfw

    if not glfw.init():
        raise RuntimeError("could not initialize GLFW for the MuJoCo review window")
    window = glfw.create_window(1280, 900, "MuJoCo | X2-OmniPicker-STL static review", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create the MuJoCo review window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera.lookat[:] = REVIEW_LOOKAT
    camera.distance = REVIEW_DISTANCE
    camera.azimuth = REVIEW_AZIMUTH
    camera.elevation = REVIEW_ELEVATION
    state = {
        "command": 0.0,
        "mode": "visual",
        "previous": None,
        "left": False,
        "right": False,
        "middle": False,
        "slider": False,
        "lastx": 0.0,
        "lasty": 0.0,
    }

    def set_command_from_x(x: float) -> None:
        width, _ = glfw.get_window_size(window)
        state["command"] = float(min(100, max(0, round((x - 40.0) * 100.0 / max(1.0, width - 80.0)))))

    def cursor_callback(_window, x, y) -> None:
        if state["slider"]:
            set_command_from_x(x)
        elif state["left"] or state["right"] or state["middle"]:
            _, height = glfw.get_window_size(window)
            dx = float(x - state["lastx"]) / max(1, height)
            dy = float(y - state["lasty"]) / max(1, height)
            shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            if state["right"]:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif state["left"]:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            mujoco.mjv_moveCamera(model, action, dx, dy, scene, camera)
        state["lastx"], state["lasty"] = x, y

    def mouse_button_callback(_window, button, action, _mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        _, height = glfw.get_window_size(window)
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and y >= height - 100:
            state["slider"] = True
            set_command_from_x(x)
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            state["left"] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            state["right"] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            state["middle"] = action == glfw.PRESS
        if action == glfw.RELEASE:
            state["slider"] = False
        state["lastx"], state["lasty"] = x, y

    def scroll_callback(_window, _xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * float(yoffset), scene, camera)

    def key_callback(_window, key, _scancode, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_LEFT:
            state["command"] = max(0.0, state["command"] - 1.0)
        elif key == glfw.KEY_RIGHT:
            state["command"] = min(100.0, state["command"] + 1.0)
        elif key == glfw.KEY_V:
            state["mode"] = "visual"
        elif key == glfw.KEY_C:
            state["mode"] = "collision"
        elif key == glfw.KEY_O:
            state["mode"] = "overlay"

    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    try:
        while not glfw.window_should_close(window):
            state_key = (state["command"], state["mode"])
            if state_key != state["previous"]:
                apply_stl_kinematic_pose(mujoco, model, data, trajectory, state["command"])
                state["previous"] = state_key
            configure_viewer_groups(type("ViewerOption", (), {"opt": option})(), state["mode"])
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            slider_y = 30
            track = mujoco.MjrRect(40, slider_y, max(1, width - 80), 12)
            mujoco.mjr_rectangle(track, 0.14, 0.18, 0.22, 0.92)
            fill = mujoco.MjrRect(40, slider_y, int(track.width * state["command"] / 100.0), 12)
            mujoco.mjr_rectangle(fill, 0.18, 0.75, 0.60, 1.0)
            marker_x = int(40 + track.width * state["command"] / 100.0)
            mujoco.mjr_rectangle(mujoco.MjrRect(marker_x - 4, slider_y - 6, 8, 24), 0.97, 0.70, 0.18, 1.0)
            left = "X2-OmniPicker-STL | KINEMATIC_SIMULATION_ONLY | 0 mj_step"
            right = f"command {int(state['command'])}%  |  V visual, C collision, O overlay"
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, left, right, context)
            footer = "drag slider or Left/Right | left-drag rotate | right-drag pan | wheel zoom | Esc close"
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, footer, "right hand STL; left remains source-scene state", context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(0.005)
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--open-scene", type=Path, help="open an already generated static review scene")
    parser.add_argument("--result", type=Path, help="result.json paired with --open-scene")
    parser.add_argument("--live-viewer", action="store_true")
    args = parser.parse_args()
    if args.open_scene is None and args.base_scene is None:
        parser.error("--base-scene is required when --open-scene is not provided")
    if args.open_scene is not None and args.base_scene is not None:
        parser.error("choose either --base-scene or --open-scene")
    if args.open_scene is not None and not args.live_viewer:
        parser.error("--open-scene is only for --live-viewer")
    if args.base_scene is not None and not args.base_scene.is_file():
        raise SystemExit(f"missing source full-X2 scene: {args.base_scene}")

    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    if args.open_scene is not None:
        if not args.open_scene.is_file():
            raise SystemExit(f"missing static review scene: {args.open_scene}")
        if args.result is not None:
            result = json.loads(args.result.read_text(encoding="utf-8"))
            if result.get("generated_scene", {}).get("sha256") != sha256(args.open_scene):
                raise SystemExit("static review scene hash differs from its paired result.json")
            if result.get("model_selection", {}).get("model_sha256") != loaded.model_sha256:
                raise SystemExit("selected STL model hash differs from its paired result.json")
        run_live_viewer(mujoco.MjModel.from_xml_path(str(args.open_scene)), trajectory)
        return
    output_location = (
        pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
        if args.output_dir is not None
        else pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-human-review")
    )
    output = output_location.native_path
    if any(path.name != ".robotsim-output-reservation.json" for path in output.iterdir()):
        raise SystemExit("review output directory is not fresh")

    scene = output / "full_x2_stl_static_review.xml"
    composition = compose_scene(args.base_scene, loaded.model_path, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R_hand_narrow2_joint") >= 0:
        raise SystemExit("right STL review scene unexpectedly retains narrow2 joint")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "R_hand_wide2_joint") >= 0:
        raise SystemExit("right STL review scene unexpectedly retains wide2 joint")
    states = [check_scene(model, trajectory, percent) for percent in (0, 50, 100)]
    render_errors = {}
    for state in states:
        command_percent = int(state["command_percent"])
        render_data = mujoco.MjData(model)
        apply_stl_kinematic_pose(mujoco, model, render_data, trajectory, command_percent)
        payload, error = render_static_png(model, render_data)
        if payload is not None:
            pg.write_new_bytes(output / f"full_x2_stl_{command_percent:03d}.png", payload)
        if error is not None:
            render_errors[str(command_percent)] = error
        closeup_payload, closeup_error = render_static_png(model, render_data, closeup=True)
        if closeup_payload is not None:
            pg.write_new_bytes(output / f"right_hand_stl_{command_percent:03d}.png", closeup_payload)
        if closeup_error is not None:
            render_errors[f"right_hand_{command_percent}"] = closeup_error
    result = {
        "schema": "robotsim.issue46.full-x2-stl-static-review.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "model_selection": {
            "model_id": loaded.model_id,
            "model_sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
            "registry_default_model_id": "X2-OmniPicker-STL",
            "selection_status": "PENDING_HUMAN_FULL_X2_REVIEW",
        },
        "source_scene": {"path": str(args.base_scene), "sha256": sha256(args.base_scene)},
        "generated_scene": {"path": str(scene), "sha256": sha256(scene)},
        "composition": composition,
        "static_states": states,
        "mujoco_version": mujoco.__version__,
        "mj_step_calls": 0,
        "render_errors": render_errors,
        "qualification": {
            "right_stl_hand_kinematic_slider": "READY_FOR_HUMAN_REVIEW",
            "full_x2_visual_assembly": "READY_FOR_HUMAN_REVIEW",
            "left_stl_mirror_mapping": "NOT_IMPLEMENTED",
            "dynamic_physics": "NOT_RUN",
            "external_object_contact": "NOT_RUN",
            "grasp": "NOT_RUN",
            "manufacturer_equivalence": "NOT_QUALIFIED",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_text(output / "README.md", """# Full X2 STL OmniPicker human review

This is a **zero-step** MuJoCo review scene. The complete X2, floor, RAICOM table,
and existing cylinder remain from the recorded source scene. Only the right
OmniPicker hand has been replaced with `X2-OmniPicker-STL`; the left hand stays
unchanged because no left-side STL mirror table has been inferred.

The native MuJoCo GLFW window has a direct bottom slider for the 101 reviewed
source-STL command rows at integer commands 0–100. Each change calls
`mj_forward`; it does not run physics.

Classification: `KINEMATIC_SIMULATION_ONLY`. No object contact, load transfer,
grasp, full-X2 dynamics, or manufacturer transmission claim is made.
""")
    launch = (
        "# Open the native MuJoCo window; no mj_step is called.\n"
        "$wt = \"/home/lzy18001500226/robotsim-issue46-stl-mainline-audit\"\n"
        f"$scene = \"{scene}\"\n"
        f"$result = \"{output / 'result.json'}\"\n"
        "wsl.exe -d Ubuntu-22.04 --cd $wt --exec env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=glfw "
        "/home/lzy18001500226/.local/bin/uv run --offline --with mujoco==3.3.6 --with numpy "
        "python3 scripts/research/issue46_stl_full_x2_review.py --open-scene $scene --result $result --live-viewer\n"
    )
    pg.write_new_text(output / "OPEN_MUJOCO_REVIEW.ps1", launch)
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": output_location.wsl_path, "output_unc": output_location.windows_unc, "scene": str(scene), "result": result["qualification"]}, ensure_ascii=False, indent=2))
    if args.live_viewer:
        run_live_viewer(model, trajectory)


if __name__ == "__main__":
    main()
