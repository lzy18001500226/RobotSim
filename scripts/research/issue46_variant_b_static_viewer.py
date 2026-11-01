#!/usr/bin/env python3
"""Read-only interactive viewer for the persistent Variant B model snapshot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "variant_b_local03_first_blocker.xml"
TARGET_BODIES = {"right_wrist_roll_link", "R_omnipicker_base_link"}


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""


def prepare(model: mujoco.MjModel, view: str) -> tuple[str, set[int]]:
    if view == "visual":
        groups = {1}
        for gid in range(model.ngeom):
            if body_name(model, model.geom_bodyid[gid]) not in TARGET_BODIES and not body_name(
                model, model.geom_bodyid[gid]
            ).startswith("R_hand_"):
                model.geom_rgba[gid, 3] = 0.0
        title = "Variant B | local03_first_blocker | VISUAL ONLY"
    elif view == "collision":
        groups = {0}
        for gid in range(model.ngeom):
            body = body_name(model, model.geom_bodyid[gid])
            if body not in TARGET_BODIES and not body.startswith("R_hand_"):
                model.geom_rgba[gid, 3] = 0.0
        title = "Variant B | local03_first_blocker | ACTIVE COLLISION ONLY"
    elif view.startswith("group"):
        groups = {int(view[-1])}
        title = f"Variant B | local03_first_blocker | GEOM GROUP {view[-1]} ONLY"
    else:
        groups = set(range(6))
        title = "Variant B | local03_first_blocker | ALL GEOM GROUPS"

    return title, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=("visual", "collision", "all", "group0", "group1", "group2"),
                        default="visual")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    model_path = args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for joint, value in (("right_claw_joint", -0.98), ("R_hand_wide1_joint", 0.98)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0:
            raise RuntimeError(f"Variant B model is missing open-pose joint {joint}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)
    title, groups = prepare(model, args.view)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.30, -0.11, 0.92]
        viewer.cam.distance = 0.38
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -14.0
        viewer.opt.geomgroup[:] = 0
        for group in groups:
            viewer.opt.geomgroup[group] = 1
        viewer.set_texts((None, None, title, f"MuJoCo {mujoco.__version__} | STATIC | mj_step=0"))
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.025)


if __name__ == "__main__":
    main()
