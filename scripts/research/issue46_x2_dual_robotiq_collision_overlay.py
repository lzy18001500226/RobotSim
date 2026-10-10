#!/usr/bin/env python3
"""Render the retained worst-cycle wrist/coupler collision geometry state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ARM = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)


def objid(model, kind, label):
    result = int(mujoco.mj_name2id(model, kind, label))
    if result < 0:
        raise RuntimeError(f"Missing {kind.name}: {label}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-xml", required=True, type=Path)
    parser.add_argument("--reference-left-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    trace = [json.loads(line) for line in args.reference_left_trace.open(encoding="utf-8")]
    row = next(item for item in trace if int(item["step"]) == 959)
    model = mujoco.MjModel.from_xml_path(str(args.baseline_xml.resolve()))
    data = mujoco.MjData(model)
    arm_q = [-0.55, -0.35, 0.0, -0.80, 0.0, 0.0, 0.0]
    for joint, value in zip(ARM, arm_q):
        jid = objid(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[int(model.jnt_qposadr[jid])] = value
    for joint, value in row["qpos_rad"].items():
        if joint.startswith("rq_left_"):
            right = "rq_right_" + joint[len("rq_left_"):]
            jid = objid(model, mujoco.mjtObj.mjOBJ_JOINT, right)
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    mujoco.mj_forward(model, data)

    model.vis.headlight.ambient[:] = [0.38, 0.38, 0.38]
    model.vis.headlight.diffuse[:] = [0.78, 0.78, 0.78]
    model.vis.headlight.specular[:] = [0.12, 0.12, 0.12]
    for geom in range(model.ngeom):
        if int(model.geom_group[geom]) != 3:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[geom])) or ""
        if body == "right_wrist_roll_link":
            model.geom_rgba[geom] = [0.90, 0.12, 0.12, 0.78]
        elif body in {"rq_right_right_coupler", "rq_right_right_follower"}:
            model.geom_rgba[geom] = [1.0, 0.44, 0.04, 0.82]
        elif body.startswith("rq_right_"):
            model.geom_rgba[geom] = [0.22, 0.26, 0.30, 0.65]
        else:
            model.geom_rgba[geom] = [0.68, 0.72, 0.76, 0.23]

    wrist = objid(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.xpos[wrist]
    camera.distance = 0.42
    camera.azimuth = 135.0
    camera.elevation = -7.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[2] = 0
    option.geomgroup[3] = 1
    renderer = mujoco.Renderer(model, height=720, width=1200)
    renderer.update_scene(data, camera, scene_option=option)
    image = Image.fromarray(renderer.render()).convert("RGB")
    renderer.close()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 62), fill=(20, 26, 32))
    draw.text((18, 13), "COLLISION GEOMS | FORWARD/OUTBOARD POSE | REOPEN STEP 959", fill="white")
    draw.text((18, 37), "red: X2 wrist collision mesh   orange: Robotiq coupler/follower collision meshes   distance: -28.949 mm",
              fill=(255, 218, 170))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
