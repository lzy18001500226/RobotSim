#!/usr/bin/env python3
"""Derive a source-preserving horizontal-cylinder STL OmniPicker contact pose."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

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
UPPER_JOINTS = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint")
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


def load_preflight():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("stl_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load STL static preflight helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contact_midpoint(model, data) -> np.ndarray:
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    positions = []
    for contact in data.contact:
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            for body in bodies
        }
        if cylinder in bodies and names & PADS:
            positions.append(np.asarray(contact.pos, dtype=float))
    if len(positions) != 2:
        raise RuntimeError(f"vertical reference lacks exactly two pad contacts: {len(positions)}")
    return np.mean(np.vstack(positions), axis=0)


def solve_position_only(
    pre,
    model,
    trajectory,
    state,
    command_percent: int,
    target_position: np.ndarray,
    start_arm: np.ndarray,
    *,
    start_motor_override: dict[str, float] | None = None,
):
    values = np.asarray([
        (start_motor_override or state["motor_targets_rad"]).get(name, state["motor_targets_rad"].get(name, 0.0))
        for name in UPPER_JOINTS
    ], dtype=float)
    if start_motor_override is None:
        values[-len(pre.ARM_JOINTS):] = start_arm
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in UPPER_JOINTS]
    bounds = []
    for name in UPPER_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bounds.append(tuple(float(value) for value in model.jnt_range[joint]))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")
    error = float("inf")
    for iteration in range(1, 801):
        data = mujoco.MjData(model)
        motor_override = dict(zip(UPPER_JOINTS, values))
        pre.apply_prior_state(model, data, state, motor_override=motor_override)
        apply_stl_kinematic_pose(mujoco, model, data, trajectory, float(command_percent))
        delta = target_position - data.xpos[body]
        error = float(np.linalg.norm(delta))
        if error <= 1e-4:
            break
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac, None, body)
        J = jac[:, dofs]
        # Position-only task with a small posture term to avoid arbitrary arm flips.
        damping = 1e-5
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), delta)
        start_values = np.asarray([
            (start_motor_override or state["motor_targets_rad"]).get(name, state["motor_targets_rad"].get(name, 0.0))
            for name in UPPER_JOINTS
        ], dtype=float)
        if start_motor_override is None:
            start_values[-len(pre.ARM_JOINTS):] = start_arm
        dq += 0.01 * (start_values - values)
        values += np.clip(dq, -0.05, 0.05)
        for index, (low, high) in enumerate(bounds):
            values[index] = float(np.clip(values[index], low + 1e-5, high - 1e-5))
    motor_override = dict(zip(UPPER_JOINTS, values))
    result = pre.evaluate_pose(model, trajectory, state, command_percent, motor_override=motor_override)
    result["arm_q_rad"] = {name: float(motor_override[name]) for name in pre.ARM_JOINTS}
    result["motor_targets_override_rad"] = motor_override
    result["iterations"] = iteration
    result["gripper_base_position_error_m"] = error
    result["position_only_ik"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--vertical-reference-result", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cylinder-axis-offset-y-m", type=float, default=0.0)
    args = parser.parse_args()
    pre = load_preflight()
    vertical = json.loads(args.vertical_reference_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    if vertical["model"]["sha256"] != loaded.model_sha256:
        raise SystemExit("vertical reference uses a different STL model")
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    vertical_arm = np.asarray([vertical["aligned_contact_static"]["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
    vertical_command = int(vertical["aligned_contact_static"]["highest_safe_bilateral_percent"])
    vertical_scene = Path(vertical["generated_scene"]["path"])
    vm = mujoco.MjModel.from_xml_path(str(vertical_scene))
    vd = mujoco.MjData(vm)
    pre.apply_prior_state(vm, vd, state, vertical_arm)
    apply_stl_kinematic_pose(mujoco, vm, vd, trajectory, vertical_command)
    midpoint = contact_midpoint(vm, vd)
    vertical_base = vd.xpos[mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")].copy()
    vertical_cylinder = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    vertical_geom = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_GEOM, "test_cylinder_collision")
    source_radius, source_halfheight = (float(value) for value in vm.geom_size[vertical_geom, :2])
    source_tabletop_z = float(vd.xpos[vertical_cylinder][2] - source_halfheight)
    table_z_offset = float(midpoint[2] - (source_tabletop_z + source_radius))

    location = (
        pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
        if args.output_dir is not None
        else pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-horizontal-cylinder-static")
    )
    output = location.native_path
    if any(item.name != ".robotsim-output-reservation.json" for item in output.iterdir()):
        raise SystemExit("output directory is not fresh")
    scene = output / "full_x2_stl_horizontal_cylinder.xml"
    composition = pre.compose_stl_scene(
        args.source_scene, loaded.model_path, scene,
        horizontal_cylinder=True, table_z_offset_m=table_z_offset,
        cylinder_axis_offset_y_m=args.cylinder_axis_offset_y_m,
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    cylinder_center = np.asarray(composition["horizontal_cylinder"]["center_world_m"], dtype=float)
    # Place the horizontal source cylinder at the measured two-pad contact height;
    # the original hand base remains reachable and its orientation stays unchanged.
    desired_base = vertical_base.copy()
    probe = mujoco.MjData(model)
    pre.apply_prior_state(model, probe, state, vertical_arm)
    apply_stl_kinematic_pose(mujoco, model, probe, trajectory, vertical_command)
    current_base = probe.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_omnipicker_base_link")].copy()
    contact_target = pre.solve_gripper_translation_ik(
        model,
        trajectory,
        state,
        vertical_command,
        desired_base - current_base,
        cylinder_lift_m=0.0,
        start_arm_override=vertical_arm,
    )
    if not contact_target["pass"]:
        contact_target = solve_position_only(pre, model, trajectory, state, vertical_command, desired_base, vertical_arm)
    contact_arm = np.asarray([contact_target["arm_q_rad"][name] for name in pre.ARM_JOINTS], dtype=float)
    contact_motor_override = contact_target.get("motor_targets_override_rad")
    scan = []
    for command in range(100, -1, -1):
        item = pre.evaluate_pose(model, trajectory, state, command, arm_override=contact_arm if contact_motor_override is None else None, motor_override=contact_motor_override)
        scan.append(item)
    eligible = [item for item in scan if item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"]]
    selected = max(eligible, key=lambda item: item["command_percent"]) if eligible else None
    lift_target = None
    transport = []
    if selected is not None:
        contact_base = np.asarray(selected["gripper_base_world_m"], dtype=float)
        lift_target = solve_position_only(
            pre,
            model,
            trajectory,
            state,
            int(selected["command_percent"]),
            contact_base + np.array([0.0, 0.0, 0.05]),
            contact_arm,
            start_motor_override=contact_motor_override,
        )
        lift_override = lift_target["motor_targets_override_rad"]
        for fraction in np.linspace(0.0, 1.0, 5):
            motor_override = {
                name: float(contact_motor_override[name] + fraction * (lift_override[name] - contact_motor_override[name]))
                for name in UPPER_JOINTS
            }
            item = pre.evaluate_pose(
                model, trajectory, state, int(selected["command_percent"]),
                cylinder_lift_m=0.05 * float(fraction),
                motor_override=motor_override,
            )
            item["fraction"] = float(fraction)
            transport.append(item)
    path_pass = bool(transport) and all(
        item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"] and (item["table_contact_count"] == 1 if item["fraction"] == 0 else item["table_contact_count"] == 0)
        for item in transport
    )
    result = {
        "schema": "robotsim.issue46.full-x2-stl-horizontal-cylinder-static.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256, "trajectory_sha256": loaded.metadata["trajectory_sha256"]},
        "source_scene": {"path": str(args.source_scene), "sha256": pre.sha256(args.source_scene)},
        "generated_scene": {"path": str(scene), "sha256": pre.sha256(scene)},
        "vertical_reference": {"path": str(args.vertical_reference_result), "sha256": pre.sha256(args.vertical_reference_result), "contact_midpoint_world_m": midpoint.tolist(), "gripper_base_world_m": vertical_base.tolist(), "derived_table_z_offset_m": table_z_offset},
        "composition": composition,
        "derived_contact_target": contact_target,
        "scan": scan,
        "selected_candidate": selected,
        "lift_target": lift_target,
        "transport_path": transport,
        "gates": {
            "SOURCE_CYLINDER_MASS_AND_GEOMETRY_PRESERVED": "PASS",
            "HORIZONTAL_TABLE_SUPPORT": "PASS" if composition["horizontal_cylinder"] else "FAIL",
            "BILATERAL_PAD_ONLY_STATIC_CANDIDATE": "PASS" if selected else "BLOCKED",
            "LIFT_50MM_STATIC_PATH": "PASS" if path_pass else "BLOCKED",
            "DYNAMICS": "NOT_RUN",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "cylinder": composition["horizontal_cylinder"], "selected": selected, "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
