#!/usr/bin/env python3
"""Open the exact observed TABLE_APPROACH post-stop state with bilateral STL hands."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", required=True, type=Path)
    parser.add_argument("--left-command-percent", type=float, default=0.0)
    parser.add_argument("--right-command-percent", type=float, default=0.0)
    parser.add_argument("--camera-distance-m", type=float, default=1.8)
    parser.add_argument("--camera-azimuth-deg", type=float, default=135.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=-12.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        not args.state_audit.is_file()
        or not all(0.0 <= value <= 100.0 for value in (args.left_command_percent, args.right_command_percent))
        or not 0.5 <= args.camera_distance_m <= 10.0
    ):
        raise SystemExit("state audit or hand command is invalid")
    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    required = ("SOURCE_POSTSTOP_STATE", "NAMED_ROBOT_STATE_MAPPING", "BILATERAL_STL_POSE", "STATIC_MJ_FORWARD")
    if any(audit.get("gates", {}).get(gate) != "PASS" for gate in required):
        raise SystemExit("post-stop audit has not passed state-transfer gates")
    state_helper = load_module("poststop_state", REPO / "scripts/research/issue46_raicom_stl_poststop_state_audit.py")
    handoff = json.loads(Path(audit["balance_handoff"]["path"]).read_text(encoding="utf-8"))
    summary = json.loads(Path(audit["source_summary"]["path"]).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(audit["mount_overlay"]["path"])
    data = mujoco.MjData(model)
    state_helper.apply_named_state(model, data, handoff["native_post_stop_sample"], summary["model_metadata"])
    loaded = load_x2_omnipicker("X2-OmniPicker-STL", repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.left_command_percent, side="left", forward=True)
    apply_stl_kinematic_pose(mujoco, model, data, trajectory, args.right_command_percent, side="right", forward=True)
    mujoco.mj_forward(model, data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise RuntimeError("post-stop overlay lacks pelvis")
    camera_lookat = data.xpos[pelvis] + np.array([0.0, 0.0, 0.5])
    print(json.dumps({"viewer": "MuJoCo native post-stop static review; 0 mj_step", "final_pose_odom": handoff["final_pose_odom"], "left_command_percent": args.left_command_percent, "right_command_percent": args.right_command_percent, "camera": {"lookat_world_m": camera_lookat.tolist(), "distance_m": args.camera_distance_m, "azimuth_deg": args.camera_azimuth_deg, "elevation_deg": args.camera_elevation_deg}}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = camera_lookat
        viewer.cam.distance = args.camera_distance_m
        viewer.cam.azimuth = args.camera_azimuth_deg
        viewer.cam.elevation = args.camera_elevation_deg
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
