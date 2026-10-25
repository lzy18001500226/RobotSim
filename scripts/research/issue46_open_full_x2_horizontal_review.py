#!/usr/bin/env python3
"""Open the horizontal cylinder candidate in a MuJoCo passive review window."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


RIGHT_HAND_NAMES = [
    "right_claw_joint", "R_hand_wide1_joint", "R_hand_narrow2_joint",
    "R_hand_wide2_joint", "R_hand_narrow3_joint", "R_hand_wide3_joint",
    "R_hand_narrow_loop_joint", "R_hand_wide_loop_joint",
]
LEFT_FROM_RIGHT = {
    "right_claw_joint": "left_claw_joint",
    "R_hand_wide1_joint": "L_hand_wide1_joint",
    "R_hand_narrow2_joint": "L_hand_narrow2_joint",
    "R_hand_wide2_joint": "L_hand_wide2_joint",
    "R_hand_narrow3_joint": "L_hand_narrow3_joint",
    "R_hand_wide3_joint": "L_hand_wide3_joint",
    "R_hand_narrow_loop_joint": "L_hand_narrow_loop_joint",
    "R_hand_wide_loop_joint": "L_hand_wide_loop_joint",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = model.joint(name).id
    data.qpos[model.jnt_qposadr[joint]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("full_x2_raicom_horizontal_cylinder.xml"))
    parser.add_argument("--result", type=Path, default=Path(__file__).with_name("result.json"))
    parser.add_argument("--mode", choices=["visual", "collision", "overlay"], default="overlay")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if sha256(args.scene) != result["scene_sha256"]:
        raise SystemExit("scene SHA-256 does not match static preflight evidence")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    for name, value in zip(
        [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ],
        result["waypoints"]["HORIZONTAL_CONTACT"]["arm_q_rad"],
    ):
        set_joint(model, data, name, float(value))
    # Reviewed static contact state: right hand at the 75% source fixture;
    # left hand at its source-consistent 90% idle fixture.
    fixture = result["gripper"]["right_contact_hand_q_rad"]
    left = result["gripper"]["left_idle_hand_q_rad"]
    for name in RIGHT_HAND_NAMES:
        set_joint(model, data, name, float(fixture[name]))
        set_joint(model, data, LEFT_FROM_RIGHT[name], float(left[name]))
    mujoco.mj_forward(model, data)
    print("Static horizontal full-X2 MuJoCo review; mj_step is never called.")
    print(
        f"Preflight status: {result['status']}; root gate: "
        f"{result['gates']['INITIAL_RIGHT_ROOT_TRACKING_AND_LINKAGE']}; "
        f"arm/hand hold: {result['gates']['RIGHT_ARM_AND_HAND_STATIC_HOLD']}."
    )
    print("Mouse: left-drag orbit, wheel zoom, right-drag pan. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [1.43, -0.91, 0.81]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 175
        viewer.cam.elevation = -12
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1
        if args.mode in ("visual", "overlay"):
            viewer.opt.geomgroup[1] = 1
        if args.mode in ("collision", "overlay"):
            viewer.opt.geomgroup[3] = 1
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
