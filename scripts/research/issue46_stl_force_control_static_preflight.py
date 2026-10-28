#!/usr/bin/env python3
"""Build and statically validate a one-root force-control mode for X2 OmniPicker STL.

The reviewed STL visual model and 101-pose table remain the source of all
geometry and pose values.  This research tool generates a run-local full-X2
scene that adds only five right-hand kinematic relations and one right-root
motor.  It does not call mj_step.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import (  # noqa: E402
    load_registered_trajectory,
)


MODEL_ID = "X2-OmniPicker-STL"
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
RIGHT_BODIES = (
    "R_hand_narrow1_Link",
    "R_hand_narrow3_Link",
    "R_hand_narrow_loop_Link",
    "R_hand_wide1_Link",
    "R_hand_wide3_Link",
    "R_hand_wide_loop_Link",
)
FIT_POLYNOMIALS = {
    "R_hand_wide1_joint": (0.0, -1.0, 0.0, 0.0, 0.0),
    "R_hand_narrow3_joint": (
        0.0029657100375548285,
        0.8093117399818315,
        1.8195527752874592,
        0.9325443703657845,
        0.47991225814436084,
    ),
    "R_hand_narrow_loop_joint": (
        -0.00020970305921240518,
        1.5443519370240675,
        -0.09427652364284096,
        -0.3856444789280108,
        -0.2753031441123662,
    ),
    "R_hand_wide3_joint": (
        -0.0029650176565236006,
        -0.8093096546344786,
        -1.8195575233843015,
        -0.9325544826462018,
        -0.4799185855150382,
    ),
    "R_hand_wide_loop_joint": (
        0.00021025715965882334,
        -1.5443519033164943,
        0.0942764680521156,
        0.38564528125718045,
        0.2753042294336954,
    ),
}
RIGHT_EQUALITY_NAMES = {
    "R_stl_review_wide_root_mimic",
    "R_stl_review_narrow3_follow",
    "R_stl_review_narrow_loop_follow",
    "R_stl_review_wide3_follow",
    "R_stl_review_wide_loop_follow",
}
MAX_ANGLE_DRIFT_RAD = math.radians(0.5)
MAX_BODY_DRIFT_M = 0.0005
FOLLOWER_SOLREF = "0.001 1"
FOLLOWER_SOLIMP = "0.99 0.999 0.001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_fingerprint(root: ET.Element) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    records: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    def visit(node: ET.Element, body_name: str = "") -> None:
        name = node.get("name", body_name) if node.tag == "body" else body_name
        for child in list(node):
            if child.tag == "geom" and child.get("contype") == "0" and child.get("conaffinity") == "0":
                records.append((name, tuple(sorted(child.attrib.items()))))
            visit(child, name)

    visit(root)
    return sorted(records)


def contact_exclusion_fingerprint(root: ET.Element) -> list[tuple[tuple[str, str], ...]]:
    contact = root.find("contact")
    if contact is None:
        return []
    return sorted(
        tuple(sorted(node.attrib.items()))
        for node in contact.findall("exclude")
    )


def right_proxy_fingerprint(root: ET.Element) -> list[tuple[tuple[str, str], ...]]:
    return sorted(
        tuple(sorted(node.attrib.items()))
        for node in root.findall(".//geom")
        if node.get("name", "").startswith("R_hand_")
        and "__collision_0__" in node.get("name", "")
    )


def poly(value: float, coefficients: tuple[float, float, float, float, float]) -> float:
    return float(sum(coefficient * value**power for power, coefficient in enumerate(coefficients)))


def fit_positions(command_percent: float) -> dict[str, float]:
    root = -0.01 * float(command_percent)
    return {
        "right_claw_joint": root,
        **{joint: poly(root, coefficients) for joint, coefficients in FIT_POLYNOMIALS.items()},
    }


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise RuntimeError(f"missing joint: {name}")
    data.qpos[model.jnt_qposadr[joint]] = float(value)
    data.qvel[model.jnt_dofadr[joint]] = 0.0


def state_by_phase(config: dict[str, object], phase: str) -> dict[str, object]:
    for state in config.get("static_states", []):
        if state.get("phase") == phase:
            return state
    raise RuntimeError(f"prior config lacks phase: {phase}")


def apply_prior_state(model: mujoco.MjModel, data: mujoco.MjData, state: dict[str, object]) -> None:
    for name, value in dict(state["motor_targets_rad"]).items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, str(name), float(value))
    for right, value in dict(state["left_hand_qpos_rad"]).items():
        name = "left_claw_joint" if right == "right_claw_joint" else str(right).replace("R_", "L_", 1)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, name, float(value))


def apply_contact_arm(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, object]) -> None:
    for name, value in values.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
            set_joint(model, data, str(name), float(value))


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom{geom_id}"


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    if cylinder < 0:
        raise RuntimeError("scene lacks test_cylinder")
    pad_bodies: set[str] = set()
    nonpad: list[dict[str, object]] = []
    table: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        bodies = (body_name(model, body1), body_name(model, body2))
        item = {
            "geom_names": (geom_name(model, geom1), geom_name(model, geom2)),
            "body_names": bodies,
            "distance_m": float(contact.dist),
        }
        records.append(item)
        if cylinder not in (body1, body2):
            continue
        other = bodies[1] if body1 == cylinder else bodies[0]
        if other in PADS:
            pad_bodies.add(other)
        elif other.startswith("R_hand_"):
            nonpad.append(item)
        elif other == "raicom_table":
            table.append(item)
    return {
        "bilateral_pad_contact": pad_bodies == PADS,
        "pad_bodies": sorted(pad_bodies),
        "nonpad_right_contacts": nonpad,
        "table_contacts": table,
        "all_contact_count": len(records),
    }


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.reshape(3, 3) @ second.reshape(3, 3).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def source_positions(trajectory, command_percent: int) -> dict[str, float]:
    return trajectory.pose(float(command_percent)).positions_rad


def apply_positions(model: mujoco.MjModel, data: mujoco.MjData, values: dict[str, float]) -> None:
    for name, value in values.items():
        set_joint(model, data, name, value)


def build_force_control_scene(source_scene: Path, destination: Path, torque_cap_nm: float) -> dict[str, object]:
    tree = ET.parse(source_scene)
    root = tree.getroot()
    visual_before = visual_fingerprint(root)
    excludes_before = contact_exclusion_fingerprint(root)
    proxies_before = right_proxy_fingerprint(root)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing_names = {node.get("name") for node in equality}
    if existing_names.intersection(RIGHT_EQUALITY_NAMES):
        raise RuntimeError("source scene already contains a right STL force-control equality")
    equality_specs = (
        ("R_stl_review_wide_root_mimic", "R_hand_wide1_joint", "right_claw_joint", FIT_POLYNOMIALS["R_hand_wide1_joint"]),
        ("R_stl_review_narrow3_follow", "R_hand_narrow3_joint", "right_claw_joint", FIT_POLYNOMIALS["R_hand_narrow3_joint"]),
        ("R_stl_review_narrow_loop_follow", "R_hand_narrow_loop_joint", "right_claw_joint", FIT_POLYNOMIALS["R_hand_narrow_loop_joint"]),
        ("R_stl_review_wide3_follow", "R_hand_wide3_joint", "right_claw_joint", FIT_POLYNOMIALS["R_hand_wide3_joint"]),
        ("R_stl_review_wide_loop_follow", "R_hand_wide_loop_joint", "right_claw_joint", FIT_POLYNOMIALS["R_hand_wide_loop_joint"]),
    )
    for name, joint1, joint2, coefficients in equality_specs:
        ET.SubElement(
            equality,
            "joint",
            name=name,
            joint1=joint1,
            joint2=joint2,
            polycoef=" ".join(f"{value:.17g}" for value in coefficients),
            solref=FOLLOWER_SOLREF,
            solimp=FOLLOWER_SOLIMP,
        )
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    if any(node.get("name") == "motor_right_claw_stl_force" for node in actuator):
        raise RuntimeError("source scene already contains motor_right_claw_stl_force")
    ET.SubElement(
        actuator,
        "motor",
        name="motor_right_claw_stl_force",
        joint="right_claw_joint",
        ctrllimited="true",
        ctrlrange=f"{-torque_cap_nm:.17g} {torque_cap_nm:.17g}",
    )
    if visual_fingerprint(root) != visual_before:
        raise RuntimeError("visual geometry changed while adding force-control relations")
    if contact_exclusion_fingerprint(root) != excludes_before:
        raise RuntimeError("collision exclusions changed while adding force-control relations")
    if right_proxy_fingerprint(root) != proxies_before:
        raise RuntimeError("right collision proxy geometry changed while adding force-control relations")
    ET.indent(tree, space="  ")
    pg.write_new_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return {
        "added_equalities": [name for name, _, _, _ in equality_specs],
        "added_actuator": "motor_right_claw_stl_force",
        "root_torque_cap_nm": torque_cap_nm,
        "follower_solref": FOLLOWER_SOLREF,
        "follower_solimp": FOLLOWER_SOLIMP,
        "visual_fingerprint_unchanged": True,
        "contact_exclusions_unchanged": True,
        "right_proxy_fingerprint_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--source-static-result", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root-torque-cap-nm", type=float, default=0.5)
    args = parser.parse_args()
    if not math.isfinite(args.root_torque_cap_nm) or args.root_torque_cap_nm <= 0.0:
        raise SystemExit("--root-torque-cap-nm must be positive and finite")
    if not args.source_scene.is_file() or not args.source_static_result.is_file() or not args.prior_config.is_file():
        raise SystemExit("source scene, static result, or prior config is missing")
    static = json.loads(args.source_static_result.read_text(encoding="utf-8"))
    if Path(static["generated_scene"]["path"]) != args.source_scene:
        raise SystemExit("source static result does not identify --source-scene")
    if sha256(args.source_scene) != static["generated_scene"]["sha256"]:
        raise SystemExit("source scene hash differs from its static result")
    required = (
        "BILATERAL_PAD_ONLY_STATIC_CANDIDATE",
        "OPPOSING_PAD_CONTACT_STATIC_CANDIDATE",
        "LIFT_50MM_PAD_ONLY_STATIC_GEOMETRY",
    )
    if any(static.get("gates", {}).get(gate) != "PASS" for gate in required):
        raise SystemExit("source static result has not passed the required pad-only geometry gates")
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    if static["model"]["sha256"] != loaded.model_sha256:
        raise SystemExit("registered STL model hash differs from source static result")
    trajectory = load_registered_trajectory(loaded.package_path)
    contact_arm = static.get("aligned_contact_static", {}).get("arm_q_rad", {})
    if not isinstance(contact_arm, dict) or not contact_arm:
        raise SystemExit("source static result lacks an aligned contact arm pose")
    location = (
        pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory")
        if args.output_dir is not None
        else pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-force-control-static")
    )
    output = location.native_path
    if any(path.name != ".robotsim-output-reservation.json" for path in output.iterdir()):
        raise SystemExit("output directory is not fresh")
    scene = output / "full_x2_stl_force_control.xml"
    overlay = build_force_control_scene(args.source_scene, scene, float(args.root_torque_cap_nm))
    model = mujoco.MjModel.from_xml_path(str(scene))
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_claw_stl_force")
    if actuator < 0:
        raise RuntimeError("compiled scene lacks the right root force motor")
    if tuple(float(value) for value in model.actuator_ctrlrange[actuator]) != (-float(args.root_torque_cap_nm), float(args.root_torque_cap_nm)):
        raise RuntimeError("compiled root torque range differs from requested range")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))
    approach = state_by_phase(config, "HORIZONTAL_APPROACH")
    body_errors: dict[str, dict[str, float]] = {
        name: {"max_position_m": 0.0, "max_rotation_rad": 0.0} for name in RIGHT_BODIES
    }
    max_angle_error = 0.0
    max_joint_error: dict[str, float] = {name: 0.0 for name in FIT_POLYNOMIALS}
    for command in range(101):
        source = mujoco.MjData(model)
        fitted = mujoco.MjData(model)
        apply_prior_state(model, source, approach)
        apply_prior_state(model, fitted, approach)
        apply_contact_arm(model, source, contact_arm)
        apply_contact_arm(model, fitted, contact_arm)
        exact = source_positions(trajectory, command)
        approximate = fit_positions(command)
        apply_positions(model, source, exact)
        apply_positions(model, fitted, approximate)
        mujoco.mj_forward(model, source)
        mujoco.mj_forward(model, fitted)
        for joint in FIT_POLYNOMIALS:
            error = abs(exact[joint] - approximate[joint])
            max_joint_error[joint] = max(max_joint_error[joint], error)
            max_angle_error = max(max_angle_error, error)
        for name in RIGHT_BODIES:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            position_error = float(np.linalg.norm(source.xpos[body] - fitted.xpos[body]))
            angle_error = rotation_error(source.xmat[body], fitted.xmat[body])
            body_errors[name]["max_position_m"] = max(body_errors[name]["max_position_m"], position_error)
            body_errors[name]["max_rotation_rad"] = max(body_errors[name]["max_rotation_rad"], angle_error)
    selected_command = static.get("selected_command_percent")
    if selected_command is None:
        selected_command = static.get("aligned_contact_static", {}).get("command_percent")
    if not isinstance(selected_command, (int, float)) or not math.isfinite(float(selected_command)):
        raise SystemExit("source static result lacks a finite selected contact command")

    contact_data = mujoco.MjData(model)
    apply_prior_state(model, contact_data, approach)
    apply_contact_arm(model, contact_data, contact_arm)
    apply_positions(model, contact_data, fit_positions(float(selected_command)))
    contact_data.ctrl[actuator] = float(args.root_torque_cap_nm)
    mujoco.mj_forward(model, contact_data)
    contacts = contact_summary(model, contact_data)
    max_body_drift = max(value["max_position_m"] for value in body_errors.values())
    max_body_rotation = max(value["max_rotation_rad"] for value in body_errors.values())
    gates = {
        "SOURCE_SCENE_HASH": "PASS",
        "SOURCE_STATIC_GATES": "PASS",
        "FORCE_CONTROL_SCENE_COMPILED": "PASS",
        "VISUAL_GEOMETRY_UNCHANGED": "PASS",
        "COLLISION_EXCLUSIONS_UNCHANGED": "PASS",
        "COACD_PROXY_GEOMETRY_UNCHANGED": "PASS",
        "SINGLE_RIGHT_ROOT_MOTOR": "PASS",
        "FIVE_REVIEWED_FOLLOWER_RELATIONS": "PASS",
        "FIT_ANGLE_DRIFT_LE_0_5_DEG": "PASS" if max_angle_error <= MAX_ANGLE_DRIFT_RAD else "FAIL",
        "FIT_BODY_DRIFT_LE_0_5_MM": "PASS" if max_body_drift <= MAX_BODY_DRIFT_M else "FAIL",
        "P37_BILATERAL_PAD_ONLY_CONTACT": "PASS" if contacts["bilateral_pad_contact"] and not contacts["nonpad_right_contacts"] else "FAIL",
        "DYNAMICS": "NOT_RUN",
    }
    result = {
        "schema": "robotsim.issue46.stl.force-control-static.v1",
        "classification": "SIMULATION_ONLY_STL_FORCE_CONTROL_PROPOSAL",
        "mj_step_calls": 0,
        "model": {
            "id": loaded.model_id,
            "sha256": loaded.model_sha256,
            "trajectory_sha256": loaded.metadata["trajectory_sha256"],
        },
        "source_scene": {"path": str(args.source_scene), "sha256": sha256(args.source_scene)},
        "source_static_result": {"path": str(args.source_static_result), "sha256": sha256(args.source_static_result)},
        "source_contact_arm_q_rad": {name: float(value) for name, value in contact_arm.items()},
        "prior_config": {"path": str(args.prior_config), "sha256": sha256(args.prior_config)},
        "generated_scene": {"path": str(scene), "sha256": sha256(scene)},
        "overlay": overlay,
        "fit": {
            "coefficients_constant_to_quartic": {name: list(values) for name, values in FIT_POLYNOMIALS.items()},
            "max_joint_angle_error_rad": max_angle_error,
            "max_joint_angle_error_deg": math.degrees(max_angle_error),
            "max_body_position_drift_m": max_body_drift,
            "max_body_rotation_drift_rad": max_body_rotation,
            "max_body_rotation_drift_deg": math.degrees(max_body_rotation),
            "joint_errors_rad": max_joint_error,
            "body_errors": body_errors,
        },
        "p37_static_force_path": {
            "command_percent": float(selected_command),
            "right_root_motor_ctrl_nm": float(contact_data.ctrl[actuator]),
            "right_root_motor_force_nm": float(contact_data.actuator_force[actuator]),
            "contacts": contacts,
        },
        "gates": gates,
        "limitations": [
            "The five follower relationships are fits to reviewed STL poses, not verified manufacturer transmission equations.",
            "The 0.5 Nm cap is retained from the pre-existing single-root simulation cap for the first physical trial; it is not a vendor torque calibration.",
            "The 1 ms follower solref is a simulation constraint-enforcement setting; it does not change a mesh, joint axis, or fitted follower relation.",
            "The follower solimp raises only numerical constraint impedance so the reviewed one-root relation remains tight under holding torque.",
            "No dynamics were run by this static preflight.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "gates": gates, "max_fit_angle_deg": result["fit"]["max_joint_angle_error_deg"], "max_fit_body_drift_mm": max_body_drift * 1000.0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
