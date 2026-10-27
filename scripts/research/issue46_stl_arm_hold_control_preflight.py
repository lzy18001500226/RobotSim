#!/usr/bin/env python3
"""Preflight one evidence-based arm-hold PD overlay for the horizontal grasp.

The overlay exists only in the research runner: it scales arm position/damping
PD gains while retaining model actuator ranges, source force limits, geometry,
collision setup, gripper table, object, and all stop gates.  No model XML is
edited and no ``mj_step`` call is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


ARM_TOKENS = ("shoulder", "elbow", "wrist")
BASE_ARM_KP = 30.0
BASE_ARM_KD = 4.0
TRACKING_GATE_RAD = 0.08


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-result", required=True, type=Path)
    parser.add_argument("--prior-dynamic-result", required=True, type=Path)
    parser.add_argument("--gain-scale", type=float, default=2.0)
    args = parser.parse_args()
    if args.gain_scale <= 1.0:
        raise SystemExit("this preflight requires a strictly larger gain scale")
    if not args.static_result.is_file() or not args.prior_dynamic_result.is_file():
        raise SystemExit("static or prior dynamic result is missing")
    static = json.loads(args.static_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_dynamic_result.read_text(encoding="utf-8"))
    required = (
        "COACD_OVERLAY_INTEGRITY",
        "HORIZONTAL_CYLINDER_TABLE_SUPPORTED",
        "OPEN_APPROACH_CLEAR",
        "BILATERAL_PAD_ONLY_STATIC_CANDIDATE",
        "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY",
        "SIMULATION_ONLY_MU3_CONTACT_OVERLAY",
    )
    if any(static["gates"].get(gate) != "PASS" for gate in required):
        raise SystemExit("horizontal static preflight is incomplete")
    if prior.get("status") != "FAIL" or prior["metrics"].get("max_arm_tracking_error_rad", 0.0) <= 0.0:
        raise SystemExit("prior dynamic trace does not establish an arm-tracking error")
    scene = Path(static["generated_scene"]["path"])
    if not scene.is_file() or sha256(scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("static scene missing or hash changed")
    model = mujoco.MjModel.from_xml_path(str(scene))
    prior_max = prior["metrics"]["max_requested_ctrl"]
    predicted = []
    max_extra = BASE_ARM_KP * (args.gain_scale - 1.0) * TRACKING_GATE_RAD
    for actuator in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or ""
        joint = int(model.actuator_trnid[actuator, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if not name.startswith("motor_") or not any(token in joint_name for token in ARM_TOKENS):
            continue
        low, high = (float(value) for value in model.actuator_ctrlrange[actuator])
        prior_abs = float(prior_max.get(name, 0.0))
        conservative_abs = prior_abs + max_extra
        margin = min(abs(low) - conservative_abs, abs(high) - conservative_abs)
        predicted.append(
            {
                "actuator": name,
                "joint": joint_name,
                "ctrlrange": [low, high],
                "prior_max_requested_abs_ctrl": prior_abs,
                "conservative_additional_pd_ctrl_at_gate": max_extra,
                "conservative_max_abs_ctrl": conservative_abs,
                "conservative_margin": margin,
            }
        )
    if not predicted or min(item["conservative_margin"] for item in predicted) <= 0.0:
        raise SystemExit("2x controller overlay could exceed an existing actuator range")
    location = pg.create_unique_output_directory(
        EVIDENCE_REPO / "reviews/issue-46", prefix="stl-coacd-horizontal-arm-hold-preflight"
    )
    output = location.native_path
    result = {
        "schema": "robotsim.issue46.stl-coacd-horizontal-arm-hold-preflight.v1",
        "classification": "SIMULATION_ONLY_CONTROL_OVERLAY",
        "mj_step_calls": 0,
        "static_result": {"path": str(args.static_result), "sha256": sha256(args.static_result)},
        "prior_dynamic_result": {"path": str(args.prior_dynamic_result), "sha256": sha256(args.prior_dynamic_result)},
        "scene": {"path": str(scene), "sha256": sha256(scene)},
        "evidence": {
            "prior_max_arm_tracking_error_rad": prior["metrics"]["max_arm_tracking_error_rad"],
            "prior_max_gripper_base_position_error_m": prior["metrics"]["max_gripper_base_position_error_m"],
            "prior_max_pad_force_N": prior["metrics"]["max_pad_force_norm_N"],
            "prior_max_cylinder_lift_mm": prior["metrics"]["max_cylinder_com_rise_m"] * 1000.0,
        },
        "controller_overlay": {
            "id": "STL_HORIZONTAL_ARM_HOLD_PD_SIM_V1",
            "only_changed_runtime_parameter": "arm_gain_scale",
            "prior_scale": 1.0,
            "requested_scale": args.gain_scale,
            "prior_arm_kp_kd": [BASE_ARM_KP, BASE_ARM_KD],
            "derived_arm_kp_kd": [BASE_ARM_KP * args.gain_scale, BASE_ARM_KD * args.gain_scale],
            "actuator_ranges_unchanged": True,
            "model_xml_unchanged": True,
            "safety_gates_unchanged": True,
            "predicted_ranges": predicted,
        },
        "gates": {
            "STATIC_WORKCELL": "PASS",
            "CONTROL_RANGE_PREFLIGHT": "PASS",
            "DYNAMICS": "NOT_RUN",
        },
        "limitations": [
            "The conservative control estimate is not a proof of dynamic contact stability.",
            "The overlay does not change actuator XML gains, force ranges, joint limits, or gripper kinematics.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(
        json.dumps(
            {
                "output_wsl": location.wsl_path,
                "gain_scale": args.gain_scale,
                "minimum_conservative_margin": min(item["conservative_margin"] for item in predicted),
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
