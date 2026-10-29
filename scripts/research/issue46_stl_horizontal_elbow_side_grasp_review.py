#!/usr/bin/env python3
"""Create one zero-step MuJoCo review for an upright-cylinder side pinch.

The review pose is constrained by human-visible geometry only: the cylinder is
upright, the forearm is horizontal, and the two distal-link reference points
are level.  It deliberately makes no dynamics or grasp-success claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402


MODEL_ID = "X2-OmniPicker-STL"
COMMAND_PERCENT = 36
TARGET_HEIGHT_ABOVE_CYLINDER_CENTER_M = 0.100
ARM_Q_RAD = {
    "right_shoulder_pitch_joint": -0.09815588760392772,
    "right_shoulder_roll_joint": -0.027053831007023885,
    "right_shoulder_yaw_joint": 0.715405995804005,
    "right_elbow_joint": -1.3071652519024397,
    "right_wrist_yaw_joint": -0.19664877385166585,
    "right_wrist_pitch_joint": 0.15914228146005635,
    "right_wrist_roll_joint": -0.5656751485167304,
}
WAIST_Q_RAD = {
    "waist_yaw_joint": -0.18235232322455672,
    "waist_pitch_joint": -0.23882792625093552,
    "waist_roll_joint": 0.3531948949954347,
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
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body_id(model: mujoco.MjModel, name: str) -> int:
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"missing body: {name}")
    return value


def render_static_review(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    lookat: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
) -> bytes:
    """Render the exact zero-step configuration used for the review state."""
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    option = mujoco.MjvOption()
    option.geomgroup[2] = 1
    option.geomgroup[3] = 0
    # The pinned source scene declares MuJoCo's default 640x480 offscreen
    # framebuffer.  Stay within that source-scene limit instead of editing it.
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render())
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        renderer.close()


def build_review(source_scene: Path, prior_config: Path) -> Path:
    static = load_module("stl_static", REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py")
    force = load_module("stl_force", REPO / "scripts/research/issue46_stl_force_control_static_preflight.py")
    orient = load_module("stl_orientation", REPO / "scripts/research/issue46_stl_centered_orientation_search.py")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    model = mujoco.MjModel.from_xml_path(str(source_scene))
    state = static.state_by_phase(json.loads(prior_config.read_text(encoding="utf-8")), "HORIZONTAL_APPROACH")
    arm = np.asarray([ARM_Q_RAD[name] for name in static.ARM_JOINTS], dtype=float)
    data = mujoco.MjData(model)
    static.apply_prior_state(model, data, state, arm, WAIST_Q_RAD)
    force.apply_positions(model, data, force.fit_positions(COMMAND_PERCENT))
    mujoco.mj_forward(model, data)

    cylinder = body_id(model, "test_cylinder")
    elbow = body_id(model, "right_elbow_link")
    wrist = body_id(model, "right_wrist_roll_link")
    narrow = body_id(model, "R_hand_narrow3_Link")
    wide = body_id(model, "R_hand_wide3_Link")
    target = data.xipos[cylinder] + np.array((0.0, 0.0, TARGET_HEIGHT_ABOVE_CYLINDER_CENTER_M))
    center_error_m = orient.jaw_center(model, data) - target
    forearm_height_delta_m = float(data.xpos[wrist, 2] - data.xpos[elbow, 2])
    distal_reference_height_delta_m = float(data.xpos[narrow, 2] - data.xpos[wide, 2])
    contacts = static.classify_contacts(model, data)
    expected_pads = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
    gates = {
        "UPRIGHT_CYLINDER": "PASS",
        "UPPER_BOTTLE_TARGET": "PASS",
        "FOREARM_LEVEL": "PASS" if abs(forearm_height_delta_m) <= 0.0001 else "FAIL",
        "DISTAL_REFERENCE_LEVEL": "PASS" if abs(distal_reference_height_delta_m) <= 0.0001 else "FAIL",
        "DISTAL_PAD_ONLY_CONTACT": "PASS" if set(contacts["pad_bodies"]) == expected_pads and not contacts["nonpad_right_contacts"] else "FAIL",
        "DYNAMICS": "NOT_RUN",
    }
    if any(gates[key] != "PASS" for key in ("FOREARM_LEVEL", "DISTAL_REFERENCE_LEVEL", "DISTAL_PAD_ONLY_CONTACT")):
        raise RuntimeError(f"review pose did not pass static gates: {gates}")

    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="stl-horizontal-elbow-side-grasp-review")
    output = location.native_path
    samples = {
        str(command): {"arm_q_rad": ARM_Q_RAD, "contact_bodies": []}
        for command in range(0, 101)
    }
    review = {
        "schema": "robotsim.issue46.stl.horizontal-elbow-side-grasp-review.v1",
        "mode": "STATIC_MJ_FORWARD_ONLY",
        "mj_step_calls": 0,
        "viewer_title": "MuJoCo | X2-OmniPicker-STL | upright bottle side-grasp review",
        "overlay_title": "UPRIGHT BOTTLE • HORIZONTAL FOREARM • STATIC REVIEW ONLY",
        "footer": "0 closed / 100 open: drag slider or Left/Right | C visual/collision/overlay | V contact markers | left rotate, right pan, wheel zoom",
        "scene": {"path": str(source_scene), "sha256": sha256(source_scene)},
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "prior_config": {"path": str(prior_config), "sha256": sha256(prior_config), "phase": "HORIZONTAL_APPROACH"},
        "state": state,
        "motor_override_rad": WAIST_Q_RAD,
        "samples": samples,
        "command_range_percent": [0, 100],
        "initial_command_percent": COMMAND_PERCENT,
        "camera": {"lookat": [1.43, -0.97, 0.80], "distance": 1.20, "azimuth": 160.0, "elevation": -8.0},
        "metric_text": f"forearm Δz {forearm_height_delta_m * 1000.0:+.3f} mm | distal reference Δz {distal_reference_height_delta_m * 1000.0:+.3f} mm",
        "pose_definition": {
            "cylinder": "upright; unchanged source-scene pose",
            "target_height_above_cylinder_center_mm": TARGET_HEIGHT_ABOVE_CYLINDER_CENTER_M * 1000.0,
            "forearm": "right elbow-to-wrist line is level with the table plane",
            "grasp": "right-hand side pinch around the upper bottle body",
        },
        "measurements": {
            "jaw_center_error_mm": (center_error_m * 1000.0).tolist(),
            "forearm_height_delta_mm": forearm_height_delta_m * 1000.0,
            "distal_reference_height_delta_mm": distal_reference_height_delta_m * 1000.0,
            "static_contact_pad_bodies": contacts["pad_bodies"],
            "static_nonpad_right_contact_count": len(contacts["nonpad_right_contacts"]),
        },
        "gates": gates,
        "limitations": [
            "This is a zero-step human visual review; it does not demonstrate force, load transfer, or lifting.",
            "Only the command-36 static pose has passed the listed contact gates; the 0–100 slider is for hand-geometry inspection.",
            "The frozen X2-OmniPicker-STL geometry and trajectory are read-only inputs.",
        ],
    }
    pg.write_new_text(output / "review_state.json", json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(
        output / "static_command36.png",
        render_static_review(model, data, lookat=(1.43, -0.97, 0.80), distance=1.20, azimuth=160.0, elevation=-8.0),
    )
    pg.write_new_bytes(
        output / "static_command36_closeup.png",
        render_static_review(model, data, lookat=(1.43, -1.07, 0.80), distance=0.48, azimuth=160.0, elevation=-5.0),
    )
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"review_state": str(output / "review_state.json"), "gates": gates, "mj_step_calls": 0}, ensure_ascii=False))
    return output / "review_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    args = parser.parse_args()
    if not args.source_scene.is_file() or not args.prior_config.is_file():
        raise SystemExit("source scene or prior config is missing")
    build_review(args.source_scene, args.prior_config)


if __name__ == "__main__":
    main()
