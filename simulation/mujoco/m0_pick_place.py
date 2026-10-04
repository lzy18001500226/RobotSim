#!/usr/bin/env python3
"""Run the RobotSim M0 bottle task with a pinned G1 MuJoCo controller."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


MUJOCO_VERSION = "3.2.6"
UPSTREAM_COMMIT = "3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
UNITREE_COMMIT = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
SOURCE_TABLE_TOP_Z = 0.8
SOURCE_TABLE_XY = np.array([0.3, 0.0], dtype=float)
SOURCE_TABLE_HALF_EXTENTS = np.array([0.2, 0.12], dtype=float)
TARGET_TABLE_HALF_EXTENTS = np.array([0.20, 0.18], dtype=float)
TABLE_GAP_M = 0.10
TARGET_TABLE_BODY_XY = np.array(
    [
        SOURCE_TABLE_XY[0],
        SOURCE_TABLE_XY[1]
        - SOURCE_TABLE_HALF_EXTENTS[1]
        - TARGET_TABLE_HALF_EXTENTS[1]
        - TABLE_GAP_M,
    ],
    dtype=float,
)
TARGET_TABLE_SITE_OFFSET_Y_M = 0.06
TARGET_TABLE_XY = TARGET_TABLE_BODY_XY + np.array(
    [0.0, TARGET_TABLE_SITE_OFFSET_Y_M], dtype=float
)
TARGET_TABLE_BODY_Z = 0.20
TARGET_TABLE_TOP_Z = TARGET_TABLE_BODY_Z + 0.8
OBJECT_BODY_NAME = "green_box"
OBJECT_JOINT_NAME = "box_joint"
OBJECT_MAIN_GEOM = "box_geom"
GRASP_WELD_NAME = "m0_grasp_weld"
OBJECT_COLLISION_GEOMS = (
    OBJECT_MAIN_GEOM,
    "m0_bottle_shoulder",
    "m0_bottle_neck",
    "m0_bottle_cap",
)
OBJECT_GEOM_SPECS = (
    {
        "name": OBJECT_MAIN_GEOM,
        "type": "box",
        "pos": "0 0 0",
        "size": "0.100 0.075 0.100",
        "mass": "0.28",
        "rgba": "0.10 0.52 0.72 1",
    },
    {
        "name": "m0_bottle_shoulder",
        "type": "box",
        "pos": "0 0 0.125",
        "size": "0.075 0.056 0.035",
        "mass": "0.020",
        "rgba": "0.10 0.52 0.72 1",
    },
    {
        "name": "m0_bottle_neck",
        "type": "cylinder",
        "pos": "0 0 0.170",
        "size": "0.025 0.035",
        "mass": "0.010",
        "rgba": "0.10 0.52 0.72 1",
    },
    {
        "name": "m0_bottle_cap",
        "type": "cylinder",
        "pos": "0 0 0.213",
        "size": "0.028 0.008",
        "mass": "0.005",
        "rgba": "0.10 0.20 0.28 1",
    },
)


def make_bottle_scene_xml(scene_xml: str, robot_model_path: str) -> str:
    """Adapt the candidate scene while preserving its G1 model and freejoint."""
    root = ET.fromstring(scene_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Candidate scene has no worldbody")

    include = root.find("include")
    if include is None:
        raise ValueError("Candidate scene has no G1 model include")
    include.set("file", str(Path(robot_model_path).resolve()))

    bottle = worldbody.find(f"body[@name='{OBJECT_BODY_NAME}']")
    if bottle is None:
        raise ValueError(f"Candidate scene has no {OBJECT_BODY_NAME} body")
    freejoint = bottle.find(f"freejoint[@name='{OBJECT_JOINT_NAME}']")
    if freejoint is None:
        raise ValueError(f"{OBJECT_BODY_NAME} must retain its freejoint")
    site = bottle.find("site[@name='box_site']")
    bottle.set("pos", "0.3 0 0.9")

    for child in list(bottle):
        if child is not freejoint and child is not site:
            bottle.remove(child)
    for spec in OBJECT_GEOM_SPECS:
        ET.SubElement(
            bottle,
            "geom",
            {
                **spec,
                "friction": "1.5 0.01 0.001",
                "condim": "4",
                "solimp": "0.95 0.95 0.001",
                "solref": "0.01 1",
            },
        )

    for body in list(worldbody.findall("body")):
        name = body.attrib.get("name", "")
        if name == "red_cube" or name.startswith("distractor_"):
            worldbody.remove(body)

    source_table = worldbody.find("body[@name='table']")
    if source_table is None:
        raise ValueError("Candidate scene has no source table")
    source_top = next(
        (geom for geom in source_table.findall("geom") if geom.attrib.get("type") == "box"),
        None,
    )
    if source_top is None:
        raise ValueError("Candidate source table has no box top")
    source_top.set("name", "m0_source_table_top")
    source_table.set("pos", f"{SOURCE_TABLE_XY[0]:.3f} {SOURCE_TABLE_XY[1]:.3f} 0")
    source_top.set(
        "size",
        f"{SOURCE_TABLE_HALF_EXTENTS[0]:.3f} {SOURCE_TABLE_HALF_EXTENTS[1]:.3f} 0.4",
    )
    for geom in list(source_table.findall("geom")):
        if geom.attrib.get("name") == "place_marker":
            source_table.remove(geom)

    target_table = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "m0_target_table",
            "pos": (
                f"{TARGET_TABLE_BODY_XY[0]:.3f} {TARGET_TABLE_BODY_XY[1]:.3f} "
                f"{TARGET_TABLE_BODY_Z:.3f}"
            ),
        },
    )
    ET.SubElement(
        target_table,
        "geom",
        {
            "name": "m0_target_table_top",
            "type": "box",
            "pos": "0 0 0.4",
            "size": (
                f"{TARGET_TABLE_HALF_EXTENTS[0]:.3f} "
                f"{TARGET_TABLE_HALF_EXTENTS[1]:.3f} 0.4"
            ),
            "rgba": "0.24 0.34 0.42 1",
            "friction": "1.2 0.01 0.001",
        },
    )
    ET.SubElement(
        target_table,
        "geom",
        {
            "name": "m0_target_marker",
            "type": "cylinder",
            "pos": f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.803",
            "size": "0.055 0.002",
            "rgba": "0.90 0.66 0.16 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        target_table,
        "site",
        {
            "name": "m0_target_site",
            "pos": f"0 {TARGET_TABLE_SITE_OFFSET_Y_M:.3f} 0.805",
            "size": "0.005",
        },
    )

    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": GRASP_WELD_NAME,
            "body1": "right_wrist_yaw_link",
            "body2": OBJECT_BODY_NAME,
            "active": "false",
        },
    )

    camera = worldbody.find("camera[@name='scene_camera']")
    if camera is not None:
        camera.set("pos", "0.4 -1.1 1.7")
        camera.set("xyaxes", "1 0 0 0 0.55 1")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def prepare_scene(candidate_root: Path, mesh_dir: Path, output_dir: Path) -> Path:
    source_model_path = candidate_root / "sim" / "models" / "g1_29dof.xml"
    source_scene_path = candidate_root / "sim" / "g1_with_camera.xml"
    if not source_model_path.is_file() or not source_scene_path.is_file():
        raise FileNotFoundError("Pinned candidate is missing its G1 scene assets")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Pinned Unitree G1 meshes are missing: {mesh_dir}")

    model_copy_path = output_dir / "g1_29dof.xml"
    scene_path = output_dir / "g1_m0_bottle.xml"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_root = ET.parse(source_model_path).getroot()
    compiler = model_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        model_root.insert(0, compiler)
    compiler.set("meshdir", str(mesh_dir.resolve()))
    ET.indent(model_root, space="  ")
    ET.ElementTree(model_root).write(model_copy_path, encoding="unicode", xml_declaration=True)

    scene_xml = source_scene_path.read_text(encoding="utf-8")
    adapted = make_bottle_scene_xml(scene_xml, str(model_copy_path))
    scene_path.write_text(adapted + "\n", encoding="utf-8")
    return scene_path


@dataclass
class AcceptanceMonitor:
    initial_position: np.ndarray
    target_xy: np.ndarray
    target_half_extents: np.ndarray
    dt: float
    lift_height: float = 0.05
    target_margin: float = 0.03
    minimum_grasp_force_n: float = 2.0
    grasp_frames_required: int = 5
    release_frames_required: int = 5
    minimum_stable_duration_s: float = 1.0
    settling_frames_required: int = field(init=False)
    linear_speed_limit: float = 0.03
    angular_speed_limit: float = 0.20
    stable_position_radius: float = 0.02
    teleport_step_limit: float = 0.20
    penetration_limit: float = 0.025
    drop_z_limit: float = 0.50

    def __post_init__(self) -> None:
        self.initial_position = np.asarray(self.initial_position, dtype=float).copy()
        self.target_xy = np.asarray(self.target_xy, dtype=float).copy()
        self.target_half_extents = np.asarray(self.target_half_extents, dtype=float).copy()
        self.settling_frames_required = math.ceil(
            self.minimum_stable_duration_s / self.dt
        ) + 1
        self.stages: dict[str, dict[str, float | int] | None] = {
            name: None
            for name in ("GRASP", "LIFT", "TRANSFER", "RELEASE", "PLACE")
        }
        self.failures: list[str] = []
        self.frame_count = 0
        self.bilateral_contact_frames = 0
        self.max_bilateral_contact_frames = 0
        self.release_frames = 0
        self.settling_frames = 0
        self.max_settling_frames = 0
        self.max_left_force_n = 0.0
        self.max_right_force_n = 0.0
        self.max_post_release_linear_speed = 0.0
        self.max_post_release_angular_speed = 0.0
        self.max_object_step = 0.0
        self.max_penetration = 0.0
        self.last_position: np.ndarray | None = None
        self.stable_anchor: np.ndarray | None = None
        self.finite_state = True
        self.bounded_state = True
        self.dropped = False

    def _stage(self, name: str, sim_time: float) -> None:
        if self.stages[name] is None:
            self.stages[name] = {"sim_time_s": float(sim_time), "control_frame": self.frame_count}

    def update(
        self,
        *,
        sim_time: float,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        upright: bool,
        left_contact: bool,
        right_contact: bool,
        grasp_attached: bool,
        left_force_n: float,
        right_force_n: float,
        target_contact: bool,
        linear_speed: float,
        angular_speed: float,
        minimum_contact_distance: float,
        state_finite: bool,
        state_extent: float,
    ) -> None:
        position = np.asarray(position, dtype=float)
        self.frame_count += 1
        self.finite_state &= bool(
            state_finite
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(quaternion_wxyz))
            and math.isfinite(sim_time)
        )
        self.bounded_state &= bool(math.isfinite(state_extent) and state_extent <= 1000.0)
        self.max_left_force_n = max(self.max_left_force_n, float(left_force_n))
        self.max_right_force_n = max(self.max_right_force_n, float(right_force_n))

        if self.last_position is not None and np.all(np.isfinite(position)):
            step_distance = float(np.linalg.norm(position - self.last_position))
            self.max_object_step = max(self.max_object_step, step_distance)
            if step_distance > self.teleport_step_limit:
                self._fail("object teleport exceeded the per-control-step limit")
        if np.all(np.isfinite(position)):
            self.last_position = position.copy()
            if position[2] < self.drop_z_limit:
                self.dropped = True
                self._fail("object fell below the configured drop-height limit")

        if math.isfinite(minimum_contact_distance):
            self.max_penetration = max(self.max_penetration, max(0.0, -minimum_contact_distance))
            if minimum_contact_distance < -self.penetration_limit:
                self._fail("object penetration exceeded the configured limit")

        bilateral = bool(
            left_contact
            and right_contact
            and left_force_n >= self.minimum_grasp_force_n
            and right_force_n >= self.minimum_grasp_force_n
        )
        self.bilateral_contact_frames = self.bilateral_contact_frames + 1 if bilateral else 0
        self.max_bilateral_contact_frames = max(
            self.max_bilateral_contact_frames, self.bilateral_contact_frames
        )
        if self.bilateral_contact_frames >= self.grasp_frames_required:
            self._stage("GRASP", sim_time)

        lifted = position[2] >= self.initial_position[2] + self.lift_height
        if self.stages["GRASP"] is not None and lifted:
            self._stage("LIFT", sim_time)

        target_inner_extents = self.target_half_extents - self.target_margin
        inside_target = bool(
            np.all(target_inner_extents > 0)
            and np.all(np.abs(position[:2] - self.target_xy) <= target_inner_extents)
        )
        if self.stages["LIFT"] is not None and inside_target:
            self._stage("TRANSFER", sim_time)

        any_hand_contact = left_contact or right_contact
        if (
            self.stages["TRANSFER"] is not None
            and not any_hand_contact
            and not grasp_attached
        ):
            self.release_frames += 1
            if self.release_frames >= self.release_frames_required:
                self._stage("RELEASE", sim_time)
        else:
            self.release_frames = 0

        if self.stages["RELEASE"] is not None:
            self.max_post_release_linear_speed = max(
                self.max_post_release_linear_speed, float(linear_speed)
            )
            self.max_post_release_angular_speed = max(
                self.max_post_release_angular_speed, float(angular_speed)
            )

        stable = bool(
            self.stages["RELEASE"] is not None
            and target_contact
            and inside_target
            and linear_speed <= self.linear_speed_limit
            and angular_speed <= self.angular_speed_limit
        )
        if stable:
            if (
                self.stable_anchor is None
                or np.linalg.norm(position - self.stable_anchor) > self.stable_position_radius
            ):
                self.stable_anchor = position.copy()
                self.settling_frames = 1
            else:
                self.settling_frames += 1
            self.max_settling_frames = max(self.max_settling_frames, self.settling_frames)
            if self.settling_frames >= self.settling_frames_required:
                self._stage("PLACE", sim_time)
        else:
            self.stable_anchor = None
            self.settling_frames = 0

    def _fail(self, reason: str) -> None:
        if reason not in self.failures:
            self.failures.append(reason)

    def result(self, final_sample: dict | None) -> dict:
        task_stages = {name: sample is not None for name, sample in self.stages.items()}
        safety = {
            "FINITE_STATE": self.finite_state,
            "ROBOT_STATE_BOUNDED": self.bounded_state,
            "NO_TELEPORT": self.max_object_step <= self.teleport_step_limit,
            "NO_EXCESSIVE_PENETRATION": self.max_penetration <= self.penetration_limit,
            "NO_DROP": not self.dropped,
        }
        final = final_sample or {}
        final_position = final.get("position")
        return {
            "passed": bool(all(task_stages.values()) and all(safety.values())),
            "task_stages": {
                name: {"passed": task_stages[name], **(sample or {})}
                for name, sample in self.stages.items()
            },
            "safety_checks": safety,
            "metrics": {
                "control_frames": self.frame_count,
                "bilateral_grasp_frames_max": self.max_bilateral_contact_frames,
                "grasp_contact_frames_required": self.grasp_frames_required,
                "minimum_grasp_force_n_per_palm": self.minimum_grasp_force_n,
                "max_left_palm_force_n": self.max_left_force_n,
                "max_right_palm_force_n": self.max_right_force_n,
                "lift_height_required_m": self.lift_height,
                "minimum_stable_duration_s": self.minimum_stable_duration_s,
                "settling_frames_required": self.settling_frames_required,
                "settling_duration_required_s": (
                    self.settling_frames_required - 1
                ) * self.dt,
                "settling_frames_observed_max": self.max_settling_frames,
                "max_object_step_m": self.max_object_step,
                "teleport_step_limit_m": self.teleport_step_limit,
                "max_penetration_m": self.max_penetration,
                "penetration_limit_m": self.penetration_limit,
                "max_post_release_linear_speed_m_s": self.max_post_release_linear_speed,
                "max_post_release_angular_speed_rad_s": self.max_post_release_angular_speed,
                "final_position_m": None
                if final_position is None
                else np.asarray(final_position, dtype=float).tolist(),
                "final_quaternion_wxyz": None
                if final.get("quaternion_wxyz") is None
                else np.asarray(final["quaternion_wxyz"], dtype=float).tolist(),
                "final_target_xy_error_m": None
                if final_position is None
                else float(np.linalg.norm(np.asarray(final_position)[:2] - self.target_xy)),
                "final_sim_time_s": final.get("sim_time_s"),
                "final_linear_speed_m_s": final.get("linear_speed_m_s"),
                "final_angular_speed_rad_s": final.get("angular_speed_rad_s"),
                "final_upright": final.get("upright"),
                "final_target_contact": final.get("target_contact"),
                "final_left_palm_contact": final.get("left_contact"),
                "final_right_palm_contact": final.get("right_contact"),
                "final_grasp_attached": final.get("grasp_attached"),
                "failure_reasons": list(self.failures),
            },
        }


def _name_id(mujoco, model, kind, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, kind, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo model has no named object {name!r}")
    return object_id


def load_upstream(candidate_root: Path):
    if not (candidate_root / ".git").exists():
        raise FileNotFoundError(f"Pinned Humanoid VLA checkout not found: {candidate_root}")
    import mujoco

    if mujoco.__version__ != MUJOCO_VERSION or mujoco.mj_versionString() != MUJOCO_VERSION:
        raise RuntimeError(
            f"Expected MuJoCo {MUJOCO_VERSION}; found Python {mujoco.__version__} "
            f"and native {mujoco.mj_versionString()}"
        )
    head = subprocess.run(
        ["git", "-C", str(candidate_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected Humanoid VLA {UPSTREAM_COMMIT}; found {head}")

    sys.path.insert(0, str(candidate_root / "scripts"))
    from physics_sim import LEFT_ARM_CTRL, RIGHT_ARM_CTRL, PhysicsSim
    from generate_bimanual_demos import plan_bimanual_trajectory

    return mujoco, PhysicsSim, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, plan_bimanual_trajectory


def _contact_sample(mujoco, sim, geometry: dict) -> dict:
    data = sim.data
    object_position = np.asarray(data.xpos[geometry["object_body"]], dtype=float).copy()
    qvel = np.asarray(data.qvel[geometry["object_qvel_adr"] : geometry["object_qvel_adr"] + 6])
    qpos_start = geometry["object_qpos_adr"]
    quaternion = np.asarray(data.qpos[qpos_start + 3 : qpos_start + 7], dtype=float).copy()
    object_rotation = np.asarray(data.xmat[geometry["object_body"]]).reshape(3, 3)
    left_contact = False
    right_contact = False
    left_force = 0.0
    right_force = 0.0
    target_contact = False
    contact_distances = []

    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geometry["object_geoms"] or second in geometry["object_geoms"]:
            contact_distances.append(float(contact.dist))
            other = second if first in geometry["object_geoms"] else first
            if other == geometry["target_geom"]:
                target_contact = True
            if other in (geometry["left_palm_geom"], geometry["right_palm_geom"]):
                force = np.zeros(6)
                mujoco.mj_contactForce(sim.model, data, index, force)
                force_n = float(np.linalg.norm(force[:3]))
                if other == geometry["left_palm_geom"]:
                    left_contact = True
                    left_force += force_n
                else:
                    right_contact = True
                    right_force += force_n

    state = np.concatenate((data.qpos, data.qvel, data.qacc, object_position))
    state_extent = float(np.max(np.abs(state))) if state.size else 0.0
    hand_distance = min(
        float(np.linalg.norm(sim.left_hand_pos - object_position)),
        float(np.linalg.norm(sim.right_hand_pos - object_position)),
    )
    grasp_attached = bool(data.eq_active[geometry["grasp_weld"]])
    weld_position_error_m, weld_rotation_error_rad = (
        _grasp_weld_pose_error(mujoco, sim, geometry)
        if grasp_attached
        else (None, None)
    )
    return {
        "sim_time_s": float(data.time),
        "position": object_position,
        "quaternion_wxyz": quaternion,
        "upright": bool(float(object_rotation[2, 2]) >= 0.95),
        "linear_speed_m_s": float(np.linalg.norm(qvel[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
        "left_contact": left_contact,
        "right_contact": right_contact,
        "grasp_attached": grasp_attached,
        "grasp_weld_position_error_m": weld_position_error_m,
        "grasp_weld_rotation_error_rad": weld_rotation_error_rad,
        "left_force_n": left_force,
        "right_force_n": right_force,
        "hand_distance_m": hand_distance,
        "target_contact": target_contact,
        "minimum_contact_distance_m": min(contact_distances) if contact_distances else math.inf,
        "state_finite": bool(np.all(np.isfinite(state))),
        "state_extent": state_extent,
    }


def _geometry(mujoco, sim) -> dict:
    model = sim.model
    object_body = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    joint = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT_NAME)
    target_geom = _name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "m0_target_table_top"
    )
    target_site = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "m0_target_site")
    left_palm = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "left_palm_pad")
    right_palm = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "right_palm_pad")
    grasp_weld = _name_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD_NAME
    )
    object_geoms = {
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in OBJECT_COLLISION_GEOMS
    }
    return {
        "object_body": object_body,
        "object_qpos_adr": int(model.jnt_qposadr[joint]),
        "object_qvel_adr": int(model.jnt_dofadr[joint]),
        "object_geoms": object_geoms,
        "target_geom": target_geom,
        "target_center": np.asarray(sim.data.site_xpos[target_site][:2], dtype=float).copy(),
        "target_half_extents": np.asarray(model.geom_size[target_geom][:2], dtype=float).copy(),
        "left_palm_geom": left_palm,
        "right_palm_geom": right_palm,
        "right_palm_body": int(model.geom_bodyid[right_palm]),
        "grasp_weld": grasp_weld,
    }


def _plan_hand_targets(sim, target_object_position, left_offset, right_offset):
    import mujoco

    saved_qpos = sim.data.qpos.copy()
    if not sim.solve_ik_left(np.asarray(target_object_position) + left_offset):
        sim.data.qpos[:] = saved_qpos
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            f"Left arm IK cannot reach object waypoint {target_object_position}; "
            f"current hand site is {sim.left_hand_pos.tolist()}"
        )
    left_target = sim.left_arm_q
    if not sim.solve_ik_right(np.asarray(target_object_position) + right_offset):
        sim.data.qpos[:] = saved_qpos
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            f"Right arm IK cannot reach object waypoint {target_object_position}; "
            f"current hand site is {sim.right_hand_pos.tolist()}"
        )
    right_target = sim.right_arm_q
    sim.data.qpos[:] = saved_qpos
    mujoco.mj_forward(sim.model, sim.data)
    return left_target, right_target


def _engage_grasp_weld(mujoco, sim, geometry: dict) -> None:
    """Attach at the measured palm/object pose, without changing object qpos."""
    model, data = sim.model, sim.data
    palm_id = geometry["right_palm_body"]
    object_id = geometry["object_body"]
    palm_quat_inv = np.empty(4, dtype=float)
    delta = np.empty(3, dtype=float)
    relative_position = np.empty(3, dtype=float)
    relative_quaternion = np.empty(4, dtype=float)

    mujoco.mju_negQuat(palm_quat_inv, data.xquat[palm_id])
    mujoco.mju_sub3(delta, data.xpos[object_id], data.xpos[palm_id])
    mujoco.mju_rotVecQuat(relative_position, delta, palm_quat_inv)
    mujoco.mju_mulQuat(relative_quaternion, palm_quat_inv, data.xquat[object_id])

    weld_id = geometry["grasp_weld"]
    weld_data = model.eq_data[weld_id]
    weld_data[:] = 0.0
    weld_data[3:6] = relative_position
    weld_data[6:10] = relative_quaternion
    weld_data[10] = 1.0
    data.eq_active[weld_id] = 1
    mujoco.mj_forward(model, data)


def _grasp_weld_pose_error(mujoco, sim, geometry: dict) -> tuple[float, float]:
    data, model = sim.data, sim.model
    palm_id = geometry["right_palm_body"]
    object_id = geometry["object_body"]
    palm_rotation = np.asarray(data.xmat[palm_id], dtype=float).reshape(3, 3)
    actual_position = palm_rotation.T @ (data.xpos[object_id] - data.xpos[palm_id])
    expected_position = model.eq_data[geometry["grasp_weld"], 3:6]

    palm_quaternion_inverse = np.empty(4, dtype=float)
    actual_quaternion = np.empty(4, dtype=float)
    mujoco.mju_negQuat(palm_quaternion_inverse, data.xquat[palm_id])
    mujoco.mju_mulQuat(actual_quaternion, palm_quaternion_inverse, data.xquat[object_id])
    expected_quaternion = model.eq_data[geometry["grasp_weld"], 6:10]
    dot = float(np.clip(abs(np.dot(actual_quaternion, expected_quaternion)), 0.0, 1.0))
    return (
        float(np.linalg.norm(actual_position - expected_position)),
        float(2.0 * math.acos(dot)),
    )


def _upright_grasp_transform(sim, geometry: dict) -> dict:
    """Keep the captured weld transform while making the bottle world-upright."""
    data = sim.data
    palm_id = geometry["right_palm_body"]
    object_id = geometry["object_body"]
    site_id = sim.right_hand_site_id
    palm_rotation = np.asarray(data.xmat[palm_id], dtype=float).reshape(3, 3)
    palm_position = np.asarray(data.xpos[palm_id], dtype=float)
    object_rotation = np.asarray(data.xmat[object_id], dtype=float).reshape(3, 3)
    object_position = np.asarray(data.xpos[object_id], dtype=float)
    site_rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    site_position = np.asarray(data.site_xpos[site_id], dtype=float)

    body_to_object_position = palm_rotation.T @ (object_position - palm_position)
    body_to_object_rotation = palm_rotation.T @ object_rotation
    body_to_site_position = palm_rotation.T @ (site_position - palm_position)
    body_to_site_rotation = palm_rotation.T @ site_rotation
    target_body_rotation = body_to_object_rotation.T

    return {
        "body_to_object_position": body_to_object_position,
        "body_to_site_position": body_to_site_position,
        "target_site_rotation": target_body_rotation @ body_to_site_rotation,
        "target_body_rotation": target_body_rotation,
    }


def _hand_position_for_object(transform: dict, object_position: np.ndarray) -> np.ndarray:
    target_body_rotation = transform["target_body_rotation"]
    target_body_position = (
        np.asarray(object_position, dtype=float)
        - target_body_rotation @ transform["body_to_object_position"]
    )
    return target_body_position + target_body_rotation @ transform["body_to_site_position"]


def _plan_right_hand_target(
    sim,
    target_hand_position: np.ndarray,
    target_hand_rotation: np.ndarray,
) -> np.ndarray:
    import mujoco

    saved_qpos = sim.data.qpos.copy()
    target_position = np.asarray(target_hand_position, dtype=float)
    target_rotation = np.asarray(target_hand_rotation, dtype=float).reshape(3, 3)
    joint_ids = [
        int(np.flatnonzero(sim.model.jnt_qposadr == address)[0])
        for address in sim.right_arm_qpos_adr
    ]
    dof_ids = np.asarray(sim.model.jnt_dofadr[joint_ids], dtype=int)
    site_id = sim.right_hand_site_id
    orientation_weight = 0.25
    position_error = np.zeros(3)
    rotation_error = np.zeros(3)

    for _ in range(700):
        mujoco.mj_forward(sim.model, sim.data)
        position_error = target_position - sim.data.site_xpos[site_id]
        current_rotation = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(current_rotation[:, axis], target_rotation[:, axis])
            for axis in range(3)
        )
        if np.linalg.norm(position_error) <= 0.02 and np.linalg.norm(rotation_error) <= 0.06:
            target = sim.right_arm_q
            break

        jacobian_position = np.zeros((3, sim.model.nv))
        jacobian_rotation = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(
            sim.model, sim.data, jacobian_position, jacobian_rotation, site_id
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, dof_ids],
                jacobian_rotation[:, dof_ids] * orientation_weight,
            )
        )
        error = np.concatenate((position_error, rotation_error * orientation_weight))
        damping = 0.05
        delta_q = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        delta_norm = float(np.linalg.norm(delta_q))
        if delta_norm > 0.02:
            delta_q *= 0.02 / delta_norm
        sim.data.qpos[sim.right_arm_qpos_adr] = np.clip(
            sim.data.qpos[sim.right_arm_qpos_adr] + delta_q,
            sim.right_arm_lo,
            sim.right_arm_hi,
        )
    else:
        sim.data.qpos[:] = saved_qpos
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            "Right arm IK cannot reach pose "
            f"{target_position.tolist()} (position error {np.linalg.norm(position_error):.3f} m, "
            f"orientation error {np.linalg.norm(rotation_error):.3f} rad)"
        )

    sim.data.qpos[:] = saved_qpos
    mujoco.mj_forward(sim.model, sim.data)
    return target


def _plan_left_hand_target(sim, target_hand_position: np.ndarray) -> np.ndarray:
    import mujoco

    saved_qpos = sim.data.qpos.copy()
    if not sim.solve_ik_left(np.asarray(target_hand_position, dtype=float)):
        hand_position = sim.left_hand_pos.tolist()
        sim.data.qpos[:] = saved_qpos
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            f"Left arm IK cannot reach hand waypoint {target_hand_position}; "
            f"current hand site is {hand_position}"
        )
    target = sim.left_arm_q
    sim.data.qpos[:] = saved_qpos
    mujoco.mj_forward(sim.model, sim.data)
    return target


def _plan_right_hand_position(sim, target_hand_position: np.ndarray) -> np.ndarray:
    """Use upstream position-only IK after the grasp has been released."""
    import mujoco

    saved_qpos = sim.data.qpos.copy()
    target_position = np.asarray(target_hand_position, dtype=float)
    if not sim.solve_ik_right(target_position):
        hand_position = sim.right_hand_pos.tolist()
        sim.data.qpos[:] = saved_qpos
        mujoco.mj_forward(sim.model, sim.data)
        raise RuntimeError(
            f"Right arm IK cannot reach retreat waypoint {target_position}; "
            f"current hand site is {hand_position}"
        )

    target = sim.right_arm_q
    sim.data.qpos[:] = saved_qpos
    mujoco.mj_forward(sim.model, sim.data)
    return target


def _interpolate(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    return np.linspace(np.asarray(start), np.asarray(end), frames + 1)[1:]


def run_demo(args: argparse.Namespace) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2

    mujoco, PhysicsSim, left_ctrl, right_ctrl, plan_bimanual = load_upstream(args.candidate_root)
    scene_path = prepare_scene(args.candidate_root, args.mesh_dir, args.output_dir / "model")
    sim = PhysicsSim(model_path=str(scene_path))
    geometry = _geometry(mujoco, sim)
    frame_dt = float(sim.model.opt.timestep * (500 // 30))
    rng = np.random.default_rng(args.seed)

    sim.reset_with_noise(rng, noise_x=0.0, noise_y=0.0)
    plan = plan_bimanual(sim, sim.box_pos.copy(), rng)
    if plan is None:
        raise RuntimeError("Upstream bimanual IK could not plan the source pick and lift")
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)

    initial_position = np.asarray(sim.box_pos, dtype=float).copy()
    initial_qpos_start = geometry["object_qpos_adr"]
    initial_quaternion = np.asarray(
        sim.data.qpos[initial_qpos_start + 3 : initial_qpos_start + 7], dtype=float
    ).copy()
    monitor = AcceptanceMonitor(
        initial_position=initial_position,
        target_xy=geometry["target_center"],
        target_half_extents=geometry["target_half_extents"],
        dt=frame_dt,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(args.video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / frame_dt,
        (640, 480),
    )
    if not video.isOpened():
        raise RuntimeError(f"Could not create demo video: {args.video}")

    controller_stages: dict[str, dict[str, float | int]] = {}
    post_release_trace: list[dict[str, object]] = []
    waypoint_checks: dict[str, dict[str, object]] = {}
    trace_start_frame: int | None = None
    final_sample = None
    frame_index = 0

    def record_post_release_sample(sample: dict) -> None:
        if trace_start_frame is None or frame_index - trace_start_frame > 60:
            return
        post_release_trace.append(
            {
                "control_frame": frame_index,
                "sim_time_s": sample["sim_time_s"],
                "position_m": sample["position"].tolist(),
                "quaternion_wxyz": sample["quaternion_wxyz"].tolist(),
                "linear_speed_m_s": sample["linear_speed_m_s"],
                "angular_speed_rad_s": sample["angular_speed_rad_s"],
                "upright": sample["upright"],
                "target_contact": sample["target_contact"],
                "left_contact": sample["left_contact"],
                "right_contact": sample["right_contact"],
                "hand_distance_m": sample["hand_distance_m"],
            }
        )

    def observe(label: str | None = None) -> dict:
        nonlocal frame_index, final_sample
        sample = _contact_sample(mujoco, sim, geometry)
        frame_index += 1
        record_post_release_sample(sample)
        monitor.update(
            sim_time=sample["sim_time_s"],
            position=sample["position"],
            quaternion_wxyz=sample["quaternion_wxyz"],
            upright=sample["upright"],
            left_contact=sample["left_contact"],
            right_contact=sample["right_contact"],
            grasp_attached=sample["grasp_attached"],
            left_force_n=sample["left_force_n"],
            right_force_n=sample["right_force_n"],
            target_contact=sample["target_contact"],
            linear_speed=sample["linear_speed_m_s"],
            angular_speed=sample["angular_speed_rad_s"],
            minimum_contact_distance=sample["minimum_contact_distance_m"],
            state_finite=sample["state_finite"],
            state_extent=sample["state_extent"],
        )
        sim.renderer.update_scene(sim.data, camera="scene_camera")
        image_rgb = sim.renderer.render().copy()
        video.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        final_sample = sample
        if label:
            controller_stages[label] = {
                "sim_time_s": sample["sim_time_s"],
                "control_frame": frame_index,
            }
            print(
                f"{label}: sim_time={sample['sim_time_s']:.3f}s frame={frame_index} "
                f"bottle_xyz={np.round(sample['position'], 3).tolist()}",
                flush=True,
            )
        for name, stage in monitor.stages.items():
            if stage is not None and name not in getattr(observe, "announced", set()):
                announced = getattr(observe, "announced", set())
                announced.add(name)
                setattr(observe, "announced", announced)
                print(
                    f"ACCEPT_{name}: sim_time={stage['sim_time_s']:.3f}s "
                    f"frame={stage['control_frame']}",
                    flush=True,
                )
        if frame_index % 50 == 0:
            print(
                f"frame={frame_index} sim_time={sample['sim_time_s']:.3f}s "
                f"left={sample['left_contact']}:{sample['left_force_n']:.1f}N "
                f"right={sample['right_contact']}:{sample['right_force_n']:.1f}N "
                f"target={sample['target_contact']}",
                flush=True,
            )
        if frame_index % 25 == 0:
            checkpoint = monitor.result(sample)
            checkpoint["partial_run"] = True
            checkpoint["controller_stages"] = controller_stages
            checkpoint["last_sample"] = {
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else None
                    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
                    else value
                )
                for key, value in sample.items()
            }
            args.output_json.write_text(
                json.dumps(checkpoint, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        return sample

    def move_right_arm(right_targets: np.ndarray) -> None:
        for right_q in right_targets:
            sim.target_pos[right_ctrl] = right_q
            sim.step_frame()
            observe()

    def move_left_arm(left_targets: np.ndarray) -> None:
        for left_q in left_targets:
            sim.target_pos[left_ctrl] = left_q
            sim.step_frame()
            observe()

    def hold_right_arm(right_target: np.ndarray, frames: int) -> None:
        for _ in range(frames):
            sim.target_pos[right_ctrl] = right_target
            sim.step_frame()
            observe()

    start_wall = time.monotonic()
    try:
        observe("START")
        phase_names = (
            "PREGRASP",
            "APPROACH",
            "SQUEEZE",
            "GRASP_HOLD",
            "LIFT",
            "LIFT_HOLD",
        )
        phase_ends = tuple(int(value) for value in plan["phase_ends"])
        if len(phase_ends) != len(phase_names):
            raise RuntimeError("upstream grasp plan has an unexpected phase layout")
        for index in range(plan["n_frames"]):
            sim.target_pos[left_ctrl] = plan["left_traj"][index]
            sim.target_pos[right_ctrl] = plan["right_traj"][index]
            sim.step_frame()
            sample = observe()
            for phase_index, phase_end in enumerate(phase_ends):
                if index + 1 == phase_end:
                    name = phase_names[phase_index]
                    controller_stages[name] = {
                        "sim_time_s": sample["sim_time_s"],
                        "control_frame": frame_index,
                    }
                    print(f"{name}: sim_time={sample['sim_time_s']:.3f}s frame={frame_index}", flush=True)

        grasp_sample = _contact_sample(mujoco, sim, geometry)
        if monitor.stages["GRASP"] is None:
            raise RuntimeError("bilateral palm contact was not sustained long enough to grasp")
        if (
            not grasp_sample["right_contact"]
            or grasp_sample["right_force_n"] < monitor.minimum_grasp_force_n
        ):
            raise RuntimeError("right palm lost measured bottle contact before grasp attachment")
        _engage_grasp_weld(mujoco, sim, geometry)
        observe("GRASP_WELD")

        left_retreat_target = _plan_left_hand_target(
            sim, sim.left_hand_pos + np.array([0.0, 0.0, 0.20])
        )
        move_left_arm(
            _interpolate(sim.left_arm_q, left_retreat_target, args.release_frames),
        )
        controller_stages["LEFT_HAND_CLEAR"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }

        object_at_lift = np.asarray(sim.box_pos, dtype=float).copy()
        grasp_transform = _upright_grasp_transform(sim, geometry)
        clearance_z = max(object_at_lift[2] + 0.12, SOURCE_TABLE_TOP_Z + 0.30)
        clearance_position = np.array(
            [object_at_lift[0], object_at_lift[1], clearance_z], dtype=float
        )
        right_clearance = _plan_right_hand_target(
            sim,
            _hand_position_for_object(grasp_transform, clearance_position),
            grasp_transform["target_site_rotation"],
        )
        move_right_arm(
            _interpolate(sim.right_arm_q, right_clearance, max(25, args.transfer_frames // 3)),
        )
        clearance_sample = _contact_sample(mujoco, sim, geometry)
        if not clearance_sample["upright"]:
            raise RuntimeError("bottle did not reach an upright pose before lateral transfer")
        controller_stages["UPRIGHT_CLEARANCE"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }

        target_lift = np.array(
            [
                geometry["target_center"][0],
                geometry["target_center"][1],
                clearance_z,
            ],
            dtype=float,
        )
        transfer_hand_target = _hand_position_for_object(grasp_transform, target_lift)
        right_transfer = _plan_right_hand_target(
            sim, transfer_hand_target, grasp_transform["target_site_rotation"]
        )
        move_right_arm(
            _interpolate(sim.right_arm_q, right_transfer, args.transfer_frames),
        )
        right_transfer = _plan_right_hand_target(
            sim, transfer_hand_target, grasp_transform["target_site_rotation"]
        )
        move_right_arm(
            _interpolate(sim.right_arm_q, right_transfer, max(60, args.release_frames * 2)),
        )
        hold_right_arm(right_transfer, 60)
        transfer_sample = _contact_sample(mujoco, sim, geometry)
        transfer_weld_error = _grasp_weld_pose_error(mujoco, sim, geometry)
        waypoint_checks["TRANSFER_END"] = {
            "object_position_m": transfer_sample["position"].tolist(),
            "object_quaternion_wxyz": transfer_sample["quaternion_wxyz"].tolist(),
            "object_upright": transfer_sample["upright"],
            "right_arm_joint_error_rad": float(np.linalg.norm(sim.right_arm_q - right_transfer)),
            "grasp_weld_position_error_m": transfer_weld_error[0],
            "grasp_weld_rotation_error_rad": transfer_weld_error[1],
        }
        print(
            "TRANSFER_END: "
            f"object_xyz={np.round(sim.box_pos, 3).tolist()} "
            f"target_xyz={np.round(target_lift, 3).tolist()} "
            f"right_site_xyz={np.round(sim.right_hand_pos, 3).tolist()} "
            f"upright={transfer_sample['upright']} "
            f"weld_rotation_error_rad={transfer_weld_error[1]:.3f} "
            f"site_error_m={np.linalg.norm(sim.right_hand_pos - transfer_hand_target):.3f} "
            f"joint_error_rad={np.linalg.norm(sim.right_arm_q - right_transfer):.3f}",
            flush=True,
        )
        if monitor.stages["TRANSFER"] is None:
            raise RuntimeError("right-hand transfer did not carry the bottle into the target footprint")
        controller_stages["TRANSFER"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        print(f"TRANSFER: sim_time={sim.data.time:.3f}s frame={frame_index}", flush=True)

        lower_z = TARGET_TABLE_TOP_Z + 0.100 + 0.007
        target_place = np.array(
            [geometry["target_center"][0], geometry["target_center"][1], lower_z],
            dtype=float,
        )
        right_lower = _plan_right_hand_target(
            sim,
            _hand_position_for_object(grasp_transform, target_place),
            grasp_transform["target_site_rotation"],
        )
        move_right_arm(
            _interpolate(sim.right_arm_q, right_lower, args.lower_frames),
        )
        right_lower = _plan_right_hand_target(
            sim,
            _hand_position_for_object(grasp_transform, target_place),
            grasp_transform["target_site_rotation"],
        )
        move_right_arm(
            _interpolate(sim.right_arm_q, right_lower, max(40, args.release_frames)),
        )
        hold_right_arm(right_lower, 40)
        lower_sample = _contact_sample(mujoco, sim, geometry)
        lower_weld_error = _grasp_weld_pose_error(mujoco, sim, geometry)
        waypoint_checks["LOWER_END"] = {
            "object_position_m": lower_sample["position"].tolist(),
            "object_quaternion_wxyz": lower_sample["quaternion_wxyz"].tolist(),
            "object_upright": lower_sample["upright"],
            "target_contact": lower_sample["target_contact"],
            "right_arm_joint_error_rad": float(np.linalg.norm(sim.right_arm_q - right_lower)),
            "grasp_weld_position_error_m": lower_weld_error[0],
            "grasp_weld_rotation_error_rad": lower_weld_error[1],
        }
        print(
            "LOWER_SETTLED: "
            f"object_xyz={np.round(lower_sample['position'], 3).tolist()} "
            f"quaternion_wxyz={np.round(lower_sample['quaternion_wxyz'], 3).tolist()} "
            f"upright={lower_sample['upright']} "
            f"target_contact={lower_sample['target_contact']} "
            f"weld_rotation_error_rad={lower_weld_error[1]:.3f} "
            f"joint_error_rad={np.linalg.norm(sim.right_arm_q - right_lower):.3f}",
            flush=True,
        )
        controller_stages["LOWER"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        print(f"LOWER: sim_time={sim.data.time:.3f}s frame={frame_index}", flush=True)

        sim.data.eq_active[geometry["grasp_weld"]] = 0
        mujoco.mj_forward(sim.model, sim.data)
        trace_start_frame = frame_index
        controller_stages["DETACH"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        record_post_release_sample(_contact_sample(mujoco, sim, geometry))
        right_retreat_target = sim.right_hand_pos + np.array([-0.04, -0.04, 0.16])
        right_release = _plan_right_hand_position(sim, right_retreat_target)
        move_right_arm(
            _interpolate(sim.right_arm_q, right_release, max(40, args.release_frames)),
        )
        controller_stages["RELEASE"] = {
            "sim_time_s": float(sim.data.time),
            "control_frame": frame_index,
        }
        print(f"RELEASE: sim_time={sim.data.time:.3f}s frame={frame_index}", flush=True)

        for _ in range(args.settle_frames):
            sim.target_pos[right_ctrl] = right_release
            sim.step_frame()
            observe()
            if monitor.stages["PLACE"] is not None:
                break
    finally:
        video.release()
        if final_sample is not None:
            sim.renderer.update_scene(sim.data, camera="scene_camera")
            final_rgb = sim.renderer.render().copy()
            cv2.imwrite(str(args.screenshot), cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        sim.renderer.close()

    result = monitor.result(final_sample)
    result.update(
        {
            "issue": 43,
            "demo": "M0 G1 bottle pick-and-place",
            "candidate": {
                "repository": "https://github.com/ozkannceylan/humanoid_vla",
                "commit": UPSTREAM_COMMIT,
                "license": "MIT",
                "unitree_mesh_repository": "https://github.com/unitreerobotics/unitree_mujoco",
                "unitree_mesh_commit": UNITREE_COMMIT,
                "mujoco_python_version": "3.2.6",
                "controller": "upstream bimanual Jacobian IK plus PD torque control",
                "grasp_reference_repository": "https://github.com/maxwellrobotics/g1-ros2",
                "grasp_reference_commit": "ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff",
                "grasp_reference_license": "BSD-3-Clause",
                "physics": "MuJoCo mj_step, 500 Hz physics substeps, pinned base",
                "grasp": "measured bilateral palm contact gates a runtime right-palm weld; detached before physics-driven release and settling",
            },
            "run": {
                "seed": args.seed,
                "simulation_step_s": float(sim.model.opt.timestep),
                "control_substeps": 500 // 30,
                "control_step_s": frame_dt,
                "wall_duration_s": time.monotonic() - start_wall,
                "initial_position_m": initial_position.tolist(),
                "initial_quaternion_wxyz": initial_quaternion.tolist(),
                "controller_stages": controller_stages,
                "post_release_trace": post_release_trace,
                "waypoint_checks": waypoint_checks,
                "source_table_top_z_m": SOURCE_TABLE_TOP_Z,
                "target_table_top_z_m": TARGET_TABLE_TOP_Z,
                "target_xy_m": geometry["target_center"].tolist(),
                "target_table_body_xy_m": TARGET_TABLE_BODY_XY.tolist(),
                "target_table_gap_m": TABLE_GAP_M,
                "target_half_extents_m": geometry["target_half_extents"].tolist(),
                "bottle_collision_geoms": list(OBJECT_COLLISION_GEOMS),
                "bottle_total_mass_kg": 0.315,
                "bottle_body_half_width_y_m": 0.075,
                "bottle_main_half_height_m": 0.100,
            },
            "artifacts": {
                "scene_xml": str(scene_path),
                "video": str(args.video),
                "screenshot": str(args.screenshot),
            },
        }
    )
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transfer-frames", type=int, default=250)
    parser.add_argument("--lower-frames", type=int, default=35)
    parser.add_argument("--release-frames", type=int, default=25)
    parser.add_argument("--settle-frames", type=int, default=100)
    args = parser.parse_args(argv)
    if min(args.transfer_frames, args.lower_frames, args.release_frames, args.settle_frames) <= 0:
        parser.error("all phase frame counts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"issue": 43, "passed": False, "state": "RUNNING"}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        result = run_demo(args)
    except Exception as exc:
        try:
            failure = json.loads(args.output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failure = {}
        failure.update(
            {
                "issue": 43,
                "demo": "M0 G1 bottle pick-and-place",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        args.output_json.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {failure['error']}", file=sys.stderr, flush=True)
        return 1
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.output_json}", flush=True)
    print(json.dumps(result["task_stages"], indent=2), flush=True)
    print(json.dumps(result["safety_checks"], indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
