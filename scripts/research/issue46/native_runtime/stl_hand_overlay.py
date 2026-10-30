#!/usr/bin/env python3
"""Prepare a source-preserving RAICOM runtime overlay for the reviewed STL hands.

The delivered RAICOM simulator contains hand command/state handlers, but its
default robot configuration neither enables those channels nor describes the
two OmniPicker roots.  This tool creates a *separate* SIMULATION_ONLY overlay:

* it starts with the already-audited bilateral STL wrist mount;
* it adds only the reviewed STL follower equalities and one position actuator
  per claw root; and
* it enables the simulator's documented hand command/state channel in a copied
  simulator and robot-model configuration.

No vendor file, frozen STL package, source mesh, or MC configuration is
modified.  The command convention is deliberately explicit: a HAL hand
command of 0..1 maps to the reviewed root coordinate 0..-1 rad through a
``gear=-1`` position actuator.  This is a runtime control mapping, not a
claim about the manufacturer's internal transmission.
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
import yaml


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
# The RAICOM delivery is shared, immutable reference runtime material rooted in
# the canonical checkout.  This task worktree contains the tool source only.
RUNTIME_ROOT = EVIDENCE_REPO / "simulation/mujoco/raicom2026"
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402


FOLLOWER_COLUMNS = {
    "wide_root_q_rad": "wide1",
    "narrow3_q_rad": "narrow3",
    "narrow_loop_q_rad": "narrow_loop",
    "wide3_q_rad": "wide3",
    "wide_loop_q_rad": "wide_loop",
}
ROOTS = {"right": "right_claw_joint", "left": "left_claw_joint"}
FOLLOWERS = {
    "right": {
        "wide1": "R_hand_wide1_joint",
        "narrow3": "R_hand_narrow3_joint",
        "narrow_loop": "R_hand_narrow_loop_joint",
        "wide3": "R_hand_wide3_joint",
        "wide_loop": "R_hand_wide_loop_joint",
    },
    "left": {
        "wide1": "L_hand_wide1_joint",
        "narrow3": "L_hand_narrow3_joint",
        "narrow_loop": "L_hand_narrow_loop_joint",
        "wide3": "L_hand_wide3_joint",
        "wide_loop": "L_hand_wide_loop_joint",
    },
}
INTERNAL_WRIST_HAND_EXCLUSIONS = (
    ("left_wrist_roll_link", "L_hand_narrow3_Link"),
    ("right_wrist_roll_link", "R_hand_narrow3_Link"),
)
HAND_COMMAND_TOPIC = "/aima/hal/joint/hand/command"
HAND_STATE_TOPIC = "/aima/hal/joint/hand/state"
RUNTIME_HAND_ALIASES = {
    "left_claw_joint": "L_claw_joint",
    "right_claw_joint": "R_claw_joint",
    "motor_left_claw_joint": "motor_L_claw_joint",
    "motor_right_claw_joint": "motor_R_claw_joint",
    "jointpos_left_claw_joint": "jointpos_L_claw_joint",
    "jointpos_right_claw_joint": "jointpos_R_claw_joint",
    "jointvel_left_claw_joint": "jointvel_L_claw_joint",
    "jointvel_right_claw_joint": "jointvel_R_claw_joint",
}
# Each generated artifact directory is mounted directly at this container path.
# Keeping the output name out of the container path makes the launch contract
# reproducible while still allowing every run to use its own governed host dir.
CONTAINER_EVIDENCE = "/workspace/evidence"
CONTAINER_REWRITES = (
    (RUNTIME_ROOT, "/workspace/raicom2026"),
    (EVIDENCE_REPO, "/workspace/robotsim"),
    (REPO, "/workspace/robotsim"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_coefficients(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 101:
        raise ValueError("reviewed STL trajectory must contain exactly 101 rows")
    root = np.asarray([float(row["root_q_rad"]) for row in rows])
    result: dict[str, list[float]] = {}
    for column, label in FOLLOWER_COLUMNS.items():
        follower = np.asarray([float(row[column]) for row in rows])
        coefficients = np.polynomial.polynomial.polyfit(root, follower, 4)
        if not np.isfinite(coefficients).all():
            raise ValueError(f"non-finite STL fit for {label}")
        result[label] = [float(value) for value in coefficients]
    return result


def maximum_fit_residual(path: Path, coefficients: dict[str, list[float]]) -> float:
    maximum = 0.0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            root = float(row["root_q_rad"])
            for column, label in FOLLOWER_COLUMNS.items():
                expected = float(np.polynomial.polynomial.polyval(root, coefficients[label]))
                maximum = max(maximum, abs(float(row[column]) - expected))
    return maximum


def add_dynamic_hand_layer(root: ET.Element, coefficients: dict[str, list[float]], *, root_kp: float = 0.5, root_kv: float = 0.05, root_force_limit_nm: float = 0.5, actuator_name_mode: str = "motor") -> dict[str, object]:
    """Add only the reviewed follower constraints and normalized root inputs."""
    equality = root.find("equality")
    actuator = root.find("actuator")
    contact = root.find("contact")
    sensor = root.find("sensor")
    if actuator is None or contact is None or sensor is None:
        raise ValueError("source STL overlay lacks actuator, contact, or sensor sections")
    if actuator_name_mode not in {"motor", "joint"}:
        raise ValueError("root actuator name mode must be motor or joint")
    if equality is None:
        # The audited static scene has no equality section.  Insert one before
        # actuators so the generated document retains the normal MJCF section
        # ordering instead of relying on XML parser permissiveness.
        equality = ET.Element("equality")
        root.insert(list(root).index(actuator), equality)
    equalities: list[str] = []
    root_actuators: list[str] = []
    root_sensors: list[str] = []
    for side in ("left", "right"):
        for label, follower in FOLLOWERS[side].items():
            name = f"{side}_stl_native_{label}_follow"
            if equality.find(f"joint[@name='{name}']") is not None:
                raise ValueError(f"refusing duplicate follower equality: {name}")
            ET.SubElement(
                equality,
                "joint",
                {
                    "name": name,
                    "joint1": follower,
                    "joint2": ROOTS[side],
                    "polycoef": " ".join(f"{value:.17g}" for value in coefficients[label]),
                    "solref": "0.001 1",
                    "solimp": "0.99 0.999 0.001",
                },
            )
            equalities.append(name)
        actuator_name = f"motor_{ROOTS[side]}" if actuator_name_mode == "motor" else ROOTS[side]
        if actuator.find(f"*[@name='{actuator_name}']") is not None:
            raise ValueError(f"refusing duplicate root actuator: {actuator_name}")
        # MuJoCo's position-actuator length is gear*q.  gear=-1 therefore
        # gives the reviewed q_root=-command mapping for the public 0..1 API.
        ET.SubElement(
            actuator,
            "position",
            {
                "name": actuator_name,
                "joint": ROOTS[side],
                "gear": "-1",
                "kp": f"{root_kp:.17g}",
                "kv": f"{root_kv:.17g}",
                "ctrllimited": "true",
                "ctrlrange": "0 1",
                "forcelimited": "true",
                "forcerange": f"{-root_force_limit_nm:.17g} {root_force_limit_nm:.17g}",
            },
        )
        root_actuators.append(actuator_name)
        for prefix, tag in (("jointpos", "jointpos"), ("jointvel", "jointvel")):
            sensor_name = f"{prefix}_{ROOTS[side]}"
            if sensor.find(f"{tag}[@name='{sensor_name}']") is not None:
                raise ValueError(f"refusing duplicate root sensor: {sensor_name}")
            ET.SubElement(sensor, tag, {"name": sensor_name, "joint": ROOTS[side], "noise": "0"})
            root_sensors.append(sensor_name)
    existing = {(item.get("body1"), item.get("body2")) for item in contact.findall("exclude")}
    exclusions: list[list[str]] = []
    for first, second in INTERNAL_WRIST_HAND_EXCLUSIONS:
        if (first, second) in existing or (second, first) in existing:
            raise ValueError(f"source overlay already declares an internal exclusion for {first}/{second}")
        ET.SubElement(contact, "exclude", body1=first, body2=second)
        exclusions.append([first, second])
    return {
        "follower_equalities": equalities,
        "root_actuators": root_actuators,
        "root_sensors": root_sensors,
        "root_control": {"kp_nm_per_rad": root_kp, "kv_nm_s_per_rad": root_kv, "force_limit_nm": root_force_limit_nm},
        "root_actuator_name_mode": actuator_name_mode,
        "normalized_command_mapping": "HAL position in [0,1] -> reviewed root q in [0,-1] rad via gear=-1",
        "scoped_internal_collision_exclusions": exclusions,
    }


def _append_unique_topic(items: list[object], topic: str) -> None:
    if any(isinstance(item, dict) and item.get("topic_name") == topic for item in items):
        raise ValueError(f"runtime config already contains topic: {topic}")
    items.append({"topic_name": topic, "enable_backends": ["ros2"]})


def _append_unique_component(items: list[object], name: str, record: dict[str, object]) -> None:
    if any(isinstance(item, dict) and item.get("name") == name for item in items):
        raise ValueError(f"runtime config already contains component: {name}")
    items.append(record)


def enable_hand_channels(simulator: dict[str, object], robot_model: dict[str, object], *, scene_container_path: str, robot_container_path: str, hand_joint_names: tuple[str, str] = ("L_claw_joint", "R_claw_joint")) -> dict[str, object]:
    """Return copied configs with the delivered hand handler enabled once."""
    simulator = copy.deepcopy(simulator)
    robot_model = copy.deepcopy(robot_model)
    aimrt = simulator.get("aimrt")
    module = simulator.get("MujocoSimModule")
    if not isinstance(aimrt, dict) or not isinstance(module, dict):
        raise ValueError("simulator config lacks aimrt or MujocoSimModule")
    channel = aimrt.get("channel")
    if not isinstance(channel, dict):
        raise ValueError("simulator config lacks channel configuration")
    pub_topics = channel.get("pub_topics_options")
    sub_topics = channel.get("sub_topics_options")
    if not isinstance(pub_topics, list) or not isinstance(sub_topics, list):
        raise ValueError("simulator topic lists are malformed")
    _append_unique_topic(pub_topics, HAND_STATE_TOPIC)
    _append_unique_topic(sub_topics, HAND_COMMAND_TOPIC)
    subscriber = module.get("SubscriberManager")
    publisher = module.get("PublisherManager")
    if not isinstance(subscriber, dict) or not isinstance(publisher, dict):
        raise ValueError("simulator config lacks subscriber or publisher manager")
    subscribers = subscriber.get("subscriber_cfg_list")
    publishers = publisher.get("publisher_cfg_list")
    if not isinstance(subscribers, list) or not isinstance(publishers, list):
        raise ValueError("simulator manager lists are malformed")
    _append_unique_component(
        subscribers,
        "hand_joint_command",
        {
            "name": "hand_joint_command",
            "topic": HAND_COMMAND_TOPIC,
            "enable": True,
            "executor": "joint_command_subscriber_thread",
            "wait_ready": False,
            "print_interval": 0,
        },
    )
    _append_unique_component(
        publishers,
        "hand_joint_state",
        {
            "name": "hand_joint_state",
            "topic": HAND_STATE_TOPIC,
            "enable": True,
            "executor": "common_joint_state_publisher_thread",
            "frequency": 100,
            "print_interval": 0,
        },
    )
    module["simulation_model"] = scene_container_path
    module["robot_model"] = robot_container_path

    actual = robot_model.get("actual")
    logical = robot_model.get("logical")
    if not isinstance(actual, dict) or not isinstance(logical, dict):
        raise ValueError("robot model config lacks actual or logical section")
    frame_types = actual.get("active_joint_frame_type")
    joint_names = actual.get("active_joint_name")
    control_types = actual.get("active_joint_control_type")
    nominal = logical.get("nominal_configuration")
    if not all(isinstance(value, list) for value in (frame_types, joint_names, control_types, nominal)):
        raise ValueError("robot model active joint configuration is malformed")
    if "hand" in frame_types:
        raise ValueError("robot model already has a hand joint frame")
    frame_types.append("hand")
    joint_names.append(list(hand_joint_names))
    control_types.append(["position", "position"])
    nominal.append([0.0, 0.0])
    return {"simulator": simulator, "robot_model": robot_model}


def apply_runtime_hand_aliases(root: ET.Element) -> dict[str, str]:
    """Match the pinned simulator HandCommand subscriber's L_/R_ convention.

    The public ROS message remains ``left/right_claw_joint``.  The delivered
    subscriber translates it to ``L_/R_claw_joint`` before model lookup, as
    observed in its own command log.  This renames only the generated runtime
    identifiers and does not affect body names, geometry, axes, or poses.
    """
    for element in root.iter():
        for attribute, value in tuple(element.attrib.items()):
            if value in RUNTIME_HAND_ALIASES:
                element.set(attribute, RUNTIME_HAND_ALIASES[value])
    return dict(RUNTIME_HAND_ALIASES)


def container_path(path: Path) -> str:
    path = path.resolve()
    for host_root, runtime_root in CONTAINER_REWRITES:
        try:
            relative = path.relative_to(host_root)
        except ValueError:
            continue
        return runtime_root if relative == Path(".") else runtime_root + "/" + relative.as_posix()
    raise ValueError(f"asset path is outside approved read-only mounts: {path}")


def remap_mesh_paths_for_runtime(root: ET.Element) -> list[dict[str, str]]:
    remapped: list[dict[str, str]] = []
    for mesh in root.findall("./asset/mesh"):
        current = mesh.get("file")
        if not current or not Path(current).is_absolute():
            raise ValueError(f"expected an absolute audited mesh path, got: {current!r}")
        source = Path(current)
        if not source.is_file():
            raise ValueError(f"audited mesh input is missing: {source}")
        target = container_path(source)
        mesh.set("file", target)
        remapped.append({"source": str(source), "container": target})
    if not remapped:
        raise ValueError("STL overlay contains no remappable mesh assets")
    return remapped


def compile_and_validate(host_scene: Path, *, actuator_name_mode: str, hand_joint_names: tuple[str, str]) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(host_scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("native hand overlay initializes with non-finite state")
    details: dict[str, object] = {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "neq": int(model.neq)}
    for joint_name in hand_joint_names:
        actuator_name = f"motor_{joint_name}" if actuator_name_mode == "motor" else joint_name
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if actuator < 0 or joint < 0:
            raise ValueError(f"missing native hand root mapping for {joint_name}")
        if int(model.actuator_trnid[actuator, 0]) != joint:
            raise ValueError(f"native hand actuator is not mapped to {joint_name}")
        if not np.allclose(model.actuator_ctrlrange[actuator], (0.0, 1.0)):
            raise ValueError(f"native hand command range is not [0,1] for {joint_name}")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount-audit", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--simulator-config", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--root-kp", type=float, default=0.5, help="SIMULATION_ONLY root position gain in N*m/rad")
    parser.add_argument("--root-kv", type=float, default=0.05, help="SIMULATION_ONLY root velocity gain in N*m*s/rad")
    parser.add_argument("--root-force-limit-nm", type=float, default=0.5, help="SIMULATION_ONLY symmetric root torque cap")
    parser.add_argument("--root-actuator-name-mode", choices=("motor", "joint"), default="motor", help="SIMULATION_ONLY name compatibility variant for the delivered hand subscriber")
    args = parser.parse_args()
    if not all(path.is_file() for path in (args.mount_audit, args.trajectory, args.simulator_config, args.robot_config)):
        raise SystemExit("mount audit, trajectory, simulator config, and robot config must be regular files")
    if not all(math.isfinite(value) and value > 0.0 for value in (args.root_kp, args.root_kv, args.root_force_limit_nm)):
        raise SystemExit("root control parameters must be finite and strictly positive")
    mount_audit = json.loads(args.mount_audit.read_text(encoding="utf-8"))
    if mount_audit.get("gates", {}).get("BILATERAL_STL_STATIC_MJ_FORWARD") != "PASS":
        raise SystemExit("the source STL mount audit is not a static pass")
    source_scene = Path(str(mount_audit.get("generated_overlay", {}).get("path", "")))
    source_hash = str(mount_audit.get("generated_overlay", {}).get("sha256", ""))
    if not source_scene.is_file() or sha256(source_scene) != source_hash:
        raise SystemExit("audited STL mount scene is missing or hash-mismatched")
    if not args.simulator_config.resolve().is_relative_to(RUNTIME_ROOT.resolve()) or not args.robot_config.resolve().is_relative_to(RUNTIME_ROOT.resolve()):
        raise SystemExit("simulator and robot configs must remain inside the pinned RAICOM runtime")

    coefficients = trajectory_coefficients(args.trajectory)
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-stl-native-hand-overlay").native_path
    host_tree = ET.parse(source_scene)
    dynamic = add_dynamic_hand_layer(host_tree.getroot(), coefficients, root_kp=args.root_kp, root_kv=args.root_kv, root_force_limit_nm=args.root_force_limit_nm, actuator_name_mode=args.root_actuator_name_mode)
    aliases = apply_runtime_hand_aliases(host_tree.getroot())
    ET.indent(host_tree, space="  ")
    host_scene = output / "scene_stl_native_hand_host.xml"
    pg.write_new_bytes(host_scene, ET.tostring(host_tree.getroot(), encoding="utf-8", xml_declaration=True))
    runtime_hand_joint_names = ("L_claw_joint", "R_claw_joint")
    model_summary = compile_and_validate(host_scene, actuator_name_mode=args.root_actuator_name_mode, hand_joint_names=runtime_hand_joint_names)

    runtime_tree = ET.parse(host_scene)
    remapped_meshes = remap_mesh_paths_for_runtime(runtime_tree.getroot())
    ET.indent(runtime_tree, space="  ")
    runtime_scene = output / "scene_stl_native_hand_runtime.xml"
    pg.write_new_bytes(runtime_scene, ET.tostring(runtime_tree.getroot(), encoding="utf-8", xml_declaration=True))

    simulator_source = yaml.safe_load(args.simulator_config.read_text(encoding="utf-8"))
    robot_source = yaml.safe_load(args.robot_config.read_text(encoding="utf-8"))
    configured = enable_hand_channels(
        simulator_source,
        robot_source,
        scene_container_path=f"{CONTAINER_EVIDENCE}/{runtime_scene.name}",
        robot_container_path=f"{CONTAINER_EVIDENCE}/robot_stl_native_hand.yaml",
        hand_joint_names=runtime_hand_joint_names,
    )
    simulator_output = output / "simulator_stl_native_hand.yaml"
    robot_output = output / "robot_stl_native_hand.yaml"
    pg.write_new_text(simulator_output, yaml.safe_dump(configured["simulator"], allow_unicode=True, sort_keys=False))
    pg.write_new_text(robot_output, yaml.safe_dump(configured["robot_model"], allow_unicode=True, sort_keys=False))

    result = {
        "schema": "robotsim.issue46.raicom-stl-native-hand-overlay.v1",
        "classification": "SIMULATION_ONLY_NATIVE_MC_HAND_BRIDGE_PREPARED",
        "mj_step_calls": 0,
        "source_mount_audit": {"path": str(args.mount_audit), "sha256": sha256(args.mount_audit)},
        "source_scene": {"path": str(source_scene), "sha256": source_hash},
        "trajectory": {"path": str(args.trajectory), "sha256": sha256(args.trajectory), "fit_max_residual_rad": maximum_fit_residual(args.trajectory, coefficients)},
        "runtime_inputs": {
            "simulator_config": {"path": str(args.simulator_config), "sha256": sha256(args.simulator_config)},
            "robot_config": {"path": str(args.robot_config), "sha256": sha256(args.robot_config)},
        },
        "generated": {
            "host_scene": {"path": str(host_scene), "sha256": sha256(host_scene)},
            "runtime_scene": {"path": str(runtime_scene), "sha256": sha256(runtime_scene)},
            "simulator_config": {"path": str(simulator_output), "sha256": sha256(simulator_output)},
            "robot_config": {"path": str(robot_output), "sha256": sha256(robot_output)},
        },
        "model": model_summary,
        "dynamic_hand_layer": dynamic,
        "runtime_hand_aliases": aliases,
        "mesh_runtime_rewrites": remapped_meshes,
        "gates": {
            "FROZEN_STL_MOUNT_INPUT": "PASS",
            "HOST_STATIC_MODEL_COMPILE": "PASS",
            "NATIVE_HAND_CHANNEL_CONFIGURATION": "PASS",
            "NATIVE_MC_DYNAMIC_HAND_COMMAND": "NOT_RUN",
            "BALANCED_ARM_APPROACH": "NOT_RUN",
            "CONTACT_DRIVEN_LIFT": "NOT_RUN",
        },
        "limitations": [
            "This generates a separate SIMULATION_ONLY runtime overlay and never changes the RAICOM runtime, MC configuration, vendor meshes, or frozen STL package.",
            "The follower equalities are a polynomial representation of the reviewed STL pose table, not manufacturer transmission evidence.",
            "The two collision exclusions are limited to the pre-existing wrist-roll/narrow3 pairs; external hand/object collisions remain enabled.",
            "A simulator/MC startup and live hand-command readback are still required before this overlay can claim native runtime integration.",
        ],
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "gates": result["gates"], "model": model_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
