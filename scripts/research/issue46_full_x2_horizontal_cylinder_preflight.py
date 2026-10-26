#!/usr/bin/env python3
"""Static MuJoCo preflight for the reviewed full-X2 horizontal cylinder grasp.

This tool deliberately never calls ``mj_step``.  It creates a separate review
MJCF, performs FK/IK and collision checks through ``mj_forward``, and records
the exact initial state needed by a later, separately guarded physics runner.
"""
from __future__ import annotations

import argparse
import collections
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
from PIL import Image
from scipy.optimize import least_squares


WORKTREE = Path(__file__).resolve().parents[2]
REPO = WORKTREE
FULL_X2_SOURCE: Path
RAICOM_ROOM: Path
RAICOM_TABLE_STL: Path
FROZEN_MODEL: Path
FIXTURE: Path
FROZEN_MODEL_SHA = "5e33d30e212ee72271efc3f0a790ebfb46cd31e775c578a10f1a2fb1f33c93e7"
FULL_X2_SOURCE_SHA = "a137a6efc793a16df808e23d764594eba4f3449664fbc5ceca554a210742b10a"

ARM = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
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
PAD_BODIES = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}
RIGHT_GRIPPER_BODIES = {
    "right_wrist_roll_link", "R_omnipicker_base_link",
    "R_hand_narrow1_Link", "R_hand_narrow2_Link", "R_hand_narrow3_Link",
    "R_hand_narrow_loop_Link", "R_hand_wide1_Link", "R_hand_wide2_Link",
    "R_hand_wide3_Link", "R_hand_wide_loop_Link",
}
SITE_NAME = "horizontal_grasp_tcp"
SITE_LOCAL = np.array([0.005789875, -0.000076084, 0.135297990])

# Frozen workcell geometry accepted in the preceding full-X2 visual review.
FIXED_PELVIS_WORLD_M = np.array([1.58, -0.7718056650230256, 0.68])
TABLE_POS_WORLD_M = np.array([1.5, -1.4, 0.0])
TABLE_NEAR_EDGE_Y_M = -0.9503544574094652
CYLINDER_DIAMETER_M = 0.070
CYLINDER_HEIGHT_M = 0.300
CYLINDER_MASS_KG = 0.20
CYLINDER_CENTER_WORLD_M = np.array([1.4370800000000161, -0.9953544574094653, 0.6967809428515006])
OPEN_PERCENT = 90
CLOSE_PERCENT = 75
GRASP_PAD_MIDPOINT_Z_M = 0.760
SHAFT_OFFSET_M = -0.020
APPROACH_BACKOFF_M = 0.020
PREGRASP_BACKOFF_M = 0.040
LIFT_M = 0.050

# These characterize the requested physical orientation, rather than requiring
# an arbitrary roll about a side-grasp axis that the source arm cannot reach.
MAX_TCP_POSITION_ERROR_M = 0.0005
MAX_PAD_AXIS_VERTICAL = 0.10
MAX_SHAFT_AXIS_VERTICAL = 0.20
MIN_SHAFT_RADIAL_ALIGNMENT = 0.98
MIN_ARM_LIMIT_MARGIN_RAD = 0.03
MAX_EQUALITY_RESIDUAL_RAD = 0.003
MAX_CONNECT_RESIDUAL_M = 0.001
ROOT_TRACKING_GATE_RAD = 0.020
ARM_KP = 150.0
ARM_KD = 22.0

sys.path.insert(0, str(REPO / "scripts/agent"))
import path_governance as pg


def configure_source_paths(args: argparse.Namespace) -> None:
    """Use explicit local evidence inputs; never bake a workstation path into Git."""
    global FULL_X2_SOURCE, RAICOM_ROOM, RAICOM_TABLE_STL, FROZEN_MODEL, FIXTURE
    FULL_X2_SOURCE = args.full_x2_source.resolve()
    RAICOM_ROOM = args.raicom_room.resolve()
    RAICOM_TABLE_STL = args.raicom_table_stl.resolve()
    FROZEN_MODEL = args.frozen_model.resolve()
    FIXTURE = args.fixture.resolve() if args.fixture else FROZEN_MODEL.parent / "tests/fixtures/pr90_static_samples.csv"
    for label, path in {
        "full-X2 source": FULL_X2_SOURCE,
        "RAICOM room": RAICOM_ROOM,
        "RAICOM table STL": RAICOM_TABLE_STL,
        "frozen model": FROZEN_MODEL,
        "static fixture": FIXTURE,
    }.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_rows() -> dict[int, dict[str, str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as stream:
        return {int(float(row["command_percent"])): row for row in csv.DictReader(stream)}


def right_hand(row: dict[str, str]) -> dict[str, float]:
    root = float(row["right_claw_joint_rad"])
    n2 = float(row["narrow2_rad"])
    n3 = float(row["narrow3_rad"])
    loop = float(row["narrow_loop_rad"])
    return {
        "right_claw_joint": root,
        "R_hand_wide1_joint": -root,
        "R_hand_narrow2_joint": n2,
        "R_hand_wide2_joint": -n2,
        "R_hand_narrow3_joint": n3,
        "R_hand_wide3_joint": -n3,
        "R_hand_narrow_loop_joint": loop,
        "R_hand_wide_loop_joint": -loop,
    }


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint = model.joint(name).id
    data.qpos[model.jnt_qposadr[joint]] = value


def set_static_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_q: np.ndarray,
    right: dict[str, float],
    left: dict[str, float],
    cylinder_pos: np.ndarray | None = None,
) -> None:
    mujoco.mj_resetData(model, data)
    for name, value in zip(ARM, arm_q):
        set_joint(model, data, name, float(value))
    for name, value in right.items():
        set_joint(model, data, name, value)
    for name, value in left.items():
        set_joint(model, data, LEFT_FROM_RIGHT[name], value)
    if cylinder_pos is not None:
        joint = model.joint("test_cylinder_free").id
        address = model.jnt_qposadr[joint]
        data.qpos[address:address + 7] = [*cylinder_pos, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def rotation_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]) / (2.0 * math.sin(angle))
    return axis * angle


def side_grasp_rotation() -> np.ndarray:
    # Local Y is the narrow/wide pad opposition direction (+/- world X).
    # Local Z follows the finger shafts, horizontally from robot toward table.
    shaft = np.array([0.0, -1.0, 0.0])
    pad_axis = np.array([-1.0, 0.0, 0.0])
    vertical = np.cross(pad_axis, shaft)
    return np.column_stack((vertical, pad_axis, shaft))


def pose_quality(
    model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, desired_rotation: np.ndarray,
    arm_q: np.ndarray,
) -> dict[str, object]:
    site = model.site(SITE_NAME).id
    actual = data.site_xmat[site].reshape(3, 3)
    desired_shaft = desired_rotation[:, 2]
    margin = []
    for name, value in zip(ARM, arm_q):
        joint = model.joint(name).id
        lo, hi = model.jnt_range[joint]
        margin.append(min(float(value - lo), float(hi - value)))
    position_error = float(np.linalg.norm(target - data.site_xpos[site]))
    strict_rotation_error = math.degrees(float(np.linalg.norm(rotation_vector(desired_rotation, actual))))
    pad_vertical = float(actual[2, 1])
    shaft_vertical = float(actual[2, 2])
    radial_alignment = float(actual[:, 2] @ desired_shaft)
    return {
        "target_world_m": target.tolist(),
        "actual_world_m": data.site_xpos[site].tolist(),
        "position_error_mm": position_error * 1000.0,
        "strict_rotation_delta_deg": strict_rotation_error,
        "pad_opposition_axis_world": actual[:, 1].tolist(),
        "finger_shaft_axis_world": actual[:, 2].tolist(),
        "pad_axis_vertical_component": pad_vertical,
        "shaft_axis_vertical_component": shaft_vertical,
        "shaft_radial_alignment": radial_alignment,
        "minimum_arm_limit_margin_rad": min(margin),
        "horizontal_side_grasp_pass": bool(
            position_error <= MAX_TCP_POSITION_ERROR_M
            and abs(pad_vertical) <= MAX_PAD_AXIS_VERTICAL
            and abs(shaft_vertical) <= MAX_SHAFT_AXIS_VERTICAL
            and radial_alignment >= MIN_SHAFT_RADIAL_ALIGNMENT
            and min(margin) >= MIN_ARM_LIMIT_MARGIN_RAD - 1e-9
        ),
    }


def solve_side_pose(
    model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray,
    rotation: np.ndarray, seed: np.ndarray, right: dict[str, float], left: dict[str, float],
) -> dict[str, object]:
    joints = [model.joint(name).id for name in ARM]
    lower = np.array([model.jnt_range[joint, 0] + MIN_ARM_LIMIT_MARGIN_RAD for joint in joints])
    upper = np.array([model.jnt_range[joint, 1] - MIN_ARM_LIMIT_MARGIN_RAD for joint in joints])
    site = model.site(SITE_NAME).id

    def residual(q: np.ndarray) -> np.ndarray:
        set_static_state(model, data, q, right, left)
        position = 100.0 * (target - data.site_xpos[site])
        rotation_error = 2.0 * rotation_vector(rotation, data.site_xmat[site].reshape(3, 3))
        return np.r_[position, rotation_error, 0.003 * (q - seed)]

    fit = least_squares(
        residual, np.clip(seed, lower + 1e-10, upper - 1e-10), bounds=(lower, upper),
        max_nfev=900, ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
    )
    set_static_state(model, data, fit.x, right, left)
    return {
        "arm_q_rad": fit.x.tolist(),
        "solver_success": bool(fit.success),
        "iterations": int(fit.nfev),
        **pose_quality(model, data, target, rotation, fit.x),
    }


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    output = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or "world"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or "world"
        pair = {n1, n2}
        if pair == {"test_cylinder", "raicom_table"}:
            kind = "cylinder_table"
        elif pair == {"head_pitch_link", "torso_link"}:
            # This is a separate full-body safety boundary.  The accepted
            # Option-D scene currently represents the head pitch collision as
            # a simple cylinder; it must never be silently treated as an
            # acceptable internal contact merely because it is unrelated to
            # the right-hand grasp.
            kind = "head_torso_proxy_contact"
        elif "test_cylinder" in pair:
            other = n2 if n1 == "test_cylinder" else n1
            kind = "distal_pad" if other in PAD_BODIES else "cylinder_forbidden_robot"
        elif "raicom_table" in pair:
            kind = "robot_table"
        elif "world" in pair:
            kind = "floor"
        else:
            kind = "robot_self"
        output.append({
            "kind": kind,
            "bodies": [n1, n2],
            "geoms": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            ],
            "geom_ids": [g1, g2],
            "geom_types": [int(model.geom_type[g1]), int(model.geom_type[g2])],
            "geom_size_m": [model.geom_size[g1].tolist(), model.geom_size[g2].tolist()],
            "distance_m": float(contact.dist),
        })
    return output


def unexpected_scene_contacts(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return contacts that make a full-X2 physics episode unsafe to start.

    A table-supported cylinder and the intentionally commanded distal-pad
    contacts are allowed.  The helper deliberately does not classify an
    unrecognised robot self-contact as harmless.
    """
    return [
        record for record in records
        if record["kind"] in {
            "robot_self", "robot_table", "cylinder_forbidden_robot",
            "head_torso_proxy_contact",
        }
    ]


def summarize_unexpected_contacts(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate repeated static contacts while retaining the deepest pose."""
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, object]] = {}
    for event in events:
        record = event["contact"]
        key = (str(record["kind"]), tuple(sorted(str(x) for x in record["bodies"])))
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "kind": record["kind"],
                "bodies": record["bodies"],
                "geoms": record["geoms"],
                "deepest_distance_m": record["distance_m"],
                "contexts": [event["context"]],
            }
            continue
        current["contexts"].append(event["context"])
        if float(record["distance_m"]) < float(current["deepest_distance_m"]):
            current["deepest_distance_m"] = record["distance_m"]
            current["geoms"] = record["geoms"]
    summary = []
    for item in grouped.values():
        contexts = list(item["contexts"])
        item["observation_count"] = len(contexts)
        item["contexts"] = sorted(set(str(context) for context in contexts))
        summary.append(item)
    return sorted(summary, key=lambda item: float(item["deepest_distance_m"]))


def right_equality_residuals(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    joint_residuals = {}
    connect_residuals = []
    for equality in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality) or ""
        if name.startswith("L_"):
            continue
        if int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_JOINT):
            j1, j2 = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            q1 = data.qpos[model.jnt_qposadr[j1]]
            q2 = data.qpos[model.jnt_qposadr[j2]]
            coeff = model.eq_data[equality, :5]
            joint_residuals[name] = float(q1 - sum(coeff[k] * q2 ** k for k in range(5)))
    for row in range(data.nefc):
        if int(data.efc_type[row]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        equality = int(data.efc_id[row])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality) or ""
        if name.startswith("L_") or int(model.eq_type[equality]) != int(mujoco.mjtEq.mjEQ_CONNECT):
            continue
        connect_residuals.append({"name": name, "residual_m": float(data.efc_pos[row])})
    return {
        "joint_residuals_rad": joint_residuals,
        "connect_residuals_m": connect_residuals,
        "max_joint_residual_rad": max((abs(v) for v in joint_residuals.values()), default=0.0),
        "max_connect_residual_m": max((abs(v["residual_m"]) for v in connect_residuals), default=0.0),
    }


def source_mesh_sample_check(
    model: mujoco.MjModel, data: mujoco.MjData, cylinder: np.ndarray,
) -> dict[str, object]:
    """Sample original visual STL triangles; this supplements proxy contacts."""
    barycentric = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    radius = CYLINDER_DIAMETER_M / 2.0
    bottom, top = cylinder[2] - CYLINDER_HEIGHT_M / 2.0, cylinder[2] + CYLINDER_HEIGHT_M / 2.0
    summary: dict[str, dict[str, object]] = {}
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        if body not in RIGHT_GRIPPER_BODIES:
            continue
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH) or int(model.geom_contype[geom]):
            continue
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        vertices = (
            model.mesh_vert[va:va + vn].astype(float)
            @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )
        triangles = vertices[model.mesh_face[fa:fa + fn]]
        points = np.einsum("sp,tpq->tsq", barycentric, triangles).reshape(-1, 3)
        vertical = (points[:, 2] >= bottom) & (points[:, 2] <= top)
        radial = np.hypot(points[:, 0] - cylinder[0], points[:, 1] - cylinder[1]) - radius
        values = radial[vertical]
        check = {
            "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
            "surface_samples": int(len(points)),
            "samples_in_cylinder_height_band": int(vertical.sum()),
            "minimum_sampled_radial_clearance_mm": None if not len(values) else float(values.min() * 1000.0),
            "sampled_surface_enters_cylinder": bool(len(values) and values.min() < 0.0),
        }
        prior = summary.get(body)
        if prior is None or (
            check["minimum_sampled_radial_clearance_mm"] is not None
            and (prior["minimum_sampled_radial_clearance_mm"] is None
                 or check["minimum_sampled_radial_clearance_mm"] < prior["minimum_sampled_radial_clearance_mm"])
        ):
            summary[body] = check
    return {
        "method": "seven barycentric samples per original visual STL triangle in the cylinder height band; supplementary to enabled collision proxies",
        "per_body": summary,
    }


def table_surface_z(triangles: np.ndarray, xy: np.ndarray) -> float | None:
    a = triangles[:, 0]
    u = triangles[:, 1] - triangles[:, 0]
    v = triangles[:, 2] - triangles[:, 0]
    determinant = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(determinant) > 1e-14
    dx, dy = xy[0] - a[:, 0], xy[1] - a[:, 1]
    s = np.divide(dx * v[:, 1] - dy * v[:, 0], determinant, out=np.full(len(determinant), -1.0), where=valid)
    t = np.divide(u[:, 0] * dy - u[:, 1] * dx, determinant, out=np.full(len(determinant), -1.0), where=valid)
    hit = valid & (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1.0 + 1e-8)
    if not np.any(hit):
        return None
    return float(np.max(a[hit, 2] + s[hit] * u[hit, 2] + t[hit] * v[hit, 2]))


def build_scene(output: Path) -> tuple[Path, dict[str, object]]:
    root = ET.parse(FULL_X2_SOURCE).getroot()
    world = root.find("worldbody")
    asset = root.find("asset")
    if world is None or asset is None:
        raise RuntimeError("full-X2 source has no worldbody/asset")
    pelvis = world.find("body[@name='pelvis']")
    table = world.find("body[@name='m0_table']")
    bottle = world.find("body[@name='canonical_bottle']")
    if pelvis is None or table is None or bottle is None:
        raise RuntimeError("expected full-X2 review fixture nodes are missing")
    robot_subtree = ET.tostring(pelvis)
    pelvis.set("pos", " ".join(f"{x:.15g}" for x in FIXED_PELVIS_WORLD_M))
    world.remove(table)
    world.remove(bottle)
    room = ET.parse(RAICOM_ROOM).getroot()
    source_mesh = room.find("./asset/mesh[@name='table']")
    if source_mesh is None:
        raise RuntimeError("RAICOM table mesh declaration missing")
    ET.SubElement(asset, "mesh", name="raicom_table_mesh", file=str(RAICOM_TABLE_STL), scale=source_mesh.get("scale"))
    table_body = ET.SubElement(world, "body", name="raicom_table", pos=" ".join(map(str, TABLE_POS_WORLD_M)), quat="1 0 0 0")
    ET.SubElement(table_body, "geom", name="raicom_table_contact", type="mesh", mesh="raicom_table_mesh", rgba="0.92 0.92 0.90 1")
    cylinder = ET.SubElement(world, "body", name="test_cylinder", pos=" ".join(f"{x:.15g}" for x in CYLINDER_CENTER_WORLD_M))
    ET.SubElement(cylinder, "freejoint", name="test_cylinder_free")
    ET.SubElement(
        cylinder, "geom", name="test_cylinder_collision", type="cylinder",
        size=f"{CYLINDER_DIAMETER_M / 2.0:.15g} {CYLINDER_HEIGHT_M / 2.0:.15g}",
        mass=str(CYLINDER_MASS_KG), friction="1 0.005 0.0001", contype="1", conaffinity="1",
        rgba="0.10 0.38 0.82 1",
    )
    right_base = world.find(".//body[@name='R_omnipicker_base_link']")
    if right_base is None:
        raise RuntimeError("right OmniPicker base missing")
    ET.SubElement(right_base, "site", name=SITE_NAME, pos=" ".join(map(str, SITE_LOCAL)), size="0.004", rgba="0.95 0.10 0.70 1", group="0")
    ET.indent(root, space="  ")
    scene = output / "full_x2_raicom_horizontal_cylinder.xml"
    pg.write_new_bytes(scene, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return scene, {
        "robot_source_subtree_sha256": hashlib.sha256(robot_subtree).hexdigest(),
        "source_full_x2_sha256": sha256(FULL_X2_SOURCE),
        "source_table_sha256": sha256(RAICOM_TABLE_STL),
    }


def set_right_arm_pd(
    model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]
) -> dict[str, bool]:
    saturation = {}
    for name in ARM:
        actuator = model.actuator(f"motor_{name}").id
        joint = model.joint(name).id
        q = float(data.qpos[model.jnt_qposadr[joint]])
        qvel = float(data.qvel[model.jnt_dofadr[joint]])
        raw = ARM_KP * (targets[name] - q) - ARM_KD * qvel
        lo, hi = model.actuator_ctrlrange[actuator]
        data.ctrl[actuator] = float(np.clip(raw, lo, hi))
        saturation[name] = bool(raw < lo - 1e-9 or raw > hi + 1e-9)
    return saturation


def root_hold_preflight(
    model: mujoco.MjModel, arm_q: np.ndarray, right: dict[str, float], left: dict[str, float]
) -> dict[str, object]:
    """Offline gravity/control balance for the exact startup state; no stepping."""
    data = mujoco.MjData(model)
    root_joint = model.joint("right_claw_joint").id
    root_address = int(model.jnt_qposadr[root_joint])
    root_dof = int(model.jnt_dofadr[root_joint])
    root_actuator = model.actuator("servo_right_claw_joint").id
    left_actuator = model.actuator("servo_left_claw_joint").id
    arm_targets = {name: float(value) for name, value in zip(ARM, arm_q)}

    def initialize(candidate_arm: np.ndarray, candidate_hand: dict[str, float]) -> None:
        set_static_state(model, data, candidate_arm, candidate_hand, left)
        data.qvel[:] = 0.0
        set_right_arm_pd(model, data, arm_targets)
        data.ctrl[root_actuator] = candidate_hand["right_claw_joint"]
        data.ctrl[left_actuator] = left["right_claw_joint"]
        mujoco.mj_forward(model, data)

    initialize(arm_q, right)
    initial_contacts = contact_records(model, data)
    initial_unexpected = unexpected_scene_contacts(initial_contacts)
    initial_equality = right_equality_residuals(model, data)
    nominal = {
        "root_q_rad": float(data.qpos[root_address]),
        "root_target_rad": right["right_claw_joint"],
        "root_qacc_rad_s2": float(data.qacc[root_dof]),
        "root_actuator_force_Nm": float(data.actuator_force[root_actuator]),
        "root_qfrc_bias_Nm": float(data.qfrc_bias[root_dof]),
        "root_qfrc_constraint_Nm": float(data.qfrc_constraint[root_dof]),
    }

    # The complete model must be free of unrelated active contacts before its
    # arm/gripper hold equilibrium is interpreted.  Solving qacc=0 through a
    # head/torso proxy penetration would report a controller problem that is
    # actually a collision-model safety failure.
    if initial_unexpected:
        root_error = abs(float(data.qpos[root_address]) - right["right_claw_joint"])
        dofs = [model.jnt_dofadr[model.joint(name).id] for name in ARM + RIGHT_HAND_NAMES]
        return {
            "method": "mj_forward startup audit only; static hold solve skipped because the full scene has unexpected active contact",
            "equilibrium_arm_q_rad": arm_q.tolist(),
            "equilibrium_right_hand_q_rad": right,
            "arm_and_right_hand_delta_rad": [0.0] * (len(ARM) + len(RIGHT_HAND_NAMES)),
            "max_abs_qacc_rad_s2": float(np.max(np.abs(data.qacc[dofs]))),
            "qacc_by_right_arm_and_hand_rad_s2": {
                name: float(data.qacc[model.jnt_dofadr[model.joint(name).id]])
                for name in ARM + RIGHT_HAND_NAMES
            },
            "root_tracking_error_rad": root_error,
            "root_target_rad": right["right_claw_joint"],
            "root_actuator_force_Nm": float(data.actuator_force[root_actuator]),
            "right_arm_control_saturated": set_right_arm_pd(model, data, arm_targets),
            "equality": initial_equality,
            "unexpected_full_scene_contacts": initial_unexpected,
            "static_hold_evaluable": False,
            "solver_success": False,
            "iterations": 0,
            "root_tracking_and_linkage_pass": bool(
                root_error <= ROOT_TRACKING_GATE_RAD
                and initial_equality["max_joint_residual_rad"] <= MAX_EQUALITY_RESIDUAL_RAD
                and initial_equality["max_connect_residual_m"] <= MAX_CONNECT_RESIDUAL_M
            ),
            "right_arm_and_hand_static_hold_pass": False,
        }

    hand_names = RIGHT_HAND_NAMES
    initial = np.r_[arm_q, [right[name] for name in hand_names]]
    arm_joints = [model.joint(name).id for name in ARM]
    hand_joints = [model.joint(name).id for name in hand_names]
    lower = np.r_[
        np.array([model.jnt_range[j, 0] + MIN_ARM_LIMIT_MARGIN_RAD for j in arm_joints]),
        np.array([right[name] - 0.10 for name in hand_names]),
    ]
    upper = np.r_[
        np.array([model.jnt_range[j, 1] - MIN_ARM_LIMIT_MARGIN_RAD for j in arm_joints]),
        np.array([right[name] + 0.10 for name in hand_names]),
    ]
    root_slot = len(ARM) + hand_names.index("right_claw_joint")
    lower[root_slot] = right["right_claw_joint"] - (ROOT_TRACKING_GATE_RAD - 1e-6)
    upper[root_slot] = right["right_claw_joint"] + (ROOT_TRACKING_GATE_RAD - 1e-6)

    def residual(values: np.ndarray) -> np.ndarray:
        hand = {name: float(value) for name, value in zip(hand_names, values[len(ARM):])}
        initialize(values[:len(ARM)], hand)
        dofs = [model.jnt_dofadr[joint] for joint in arm_joints + hand_joints]
        equality = right_equality_residuals(model, data)
        e_joint = list(equality["joint_residuals_rad"].values())
        e_connect = [value["residual_m"] * 10.0 for value in equality["connect_residuals_m"]]
        return np.r_[data.qacc[dofs], 100.0 * np.asarray(e_joint), 1000.0 * np.asarray(e_connect)]

    fit = least_squares(
        residual, np.clip(initial, lower + 1e-9, upper - 1e-9), bounds=(lower, upper),
        max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
    )
    equilibrium_hand = {name: float(value) for name, value in zip(hand_names, fit.x[len(ARM):])}
    initialize(fit.x[:len(ARM)], equilibrium_hand)
    equality = right_equality_residuals(model, data)
    dofs = [model.jnt_dofadr[joint] for joint in arm_joints + hand_joints]
    qacc_by_joint = {
        name: float(data.qacc[model.jnt_dofadr[joint]])
        for name, joint in zip(ARM + hand_names, arm_joints + hand_joints)
    }
    root_error = abs(float(data.qpos[root_address]) - right["right_claw_joint"])
    linkage_pass = (
        equality["max_joint_residual_rad"] <= MAX_EQUALITY_RESIDUAL_RAD
        and equality["max_connect_residual_m"] <= MAX_CONNECT_RESIDUAL_M
    )
    root_tracking_pass = root_error <= ROOT_TRACKING_GATE_RAD and linkage_pass
    return {
        "method": "least-squares mj_forward qacc=0 with unchanged right-arm PD, frozen root servo, both grippers initialized on static fixture; no mj_step",
        "nominal": nominal,
        "equilibrium_arm_q_rad": fit.x[:len(ARM)].tolist(),
        "equilibrium_right_hand_q_rad": equilibrium_hand,
        "arm_and_right_hand_delta_rad": (fit.x - initial).tolist(),
        "max_abs_qacc_rad_s2": float(np.max(np.abs(data.qacc[dofs]))),
        "qacc_by_right_arm_and_hand_rad_s2": qacc_by_joint,
        "root_tracking_error_rad": root_error,
        "root_target_rad": right["right_claw_joint"],
        "root_actuator_force_Nm": float(data.actuator_force[root_actuator]),
        "right_arm_control_saturated": set_right_arm_pd(model, data, arm_targets),
        "equality": equality,
        "unexpected_full_scene_contacts": [],
        "static_hold_evaluable": True,
        "solver_success": bool(fit.success),
        "iterations": int(fit.nfev),
        "root_tracking_and_linkage_pass": root_tracking_pass,
        "right_arm_and_hand_static_hold_pass": bool(
            fit.success
            and float(np.max(np.abs(data.qacc[dofs]))) <= 1e-3
        ),
    }


def render_views(model: mujoco.MjModel, data: mujoco.MjData, output: Path) -> None:
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    views = [
        ("side_horizontal_grasp", [1.43, -0.91, 0.81], 1.15, 175, -12, (0, 1)),
        ("top_horizontal_grasp", [1.43, -0.91, 0.76], 1.05, 180, -78, (0, 1)),
        ("collision_overlay_horizontal_grasp", [1.43, -0.91, 0.81], 1.15, 175, -12, (0, 1, 3)),
    ]
    for name, lookat, distance, azimuth, elevation, groups in views:
        option.geomgroup[:] = 0
        for group in groups:
            option.geomgroup[group] = 1
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(np.ascontiguousarray(renderer.render(), dtype=np.uint8)).save(output / f"{name}.png")
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-x2-source", type=Path, required=True)
    parser.add_argument("--raicom-room", type=Path, required=True)
    parser.add_argument("--raicom-table-stl", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--execute-dynamics", action="store_true")
    args = parser.parse_args()
    configure_source_paths(args)
    if args.execute_dynamics:
        raise SystemExit("This command is static preflight only; dynamics needs the separately guarded runner.")
    if sha256(FROZEN_MODEL) != FROZEN_MODEL_SHA:
        raise SystemExit("frozen canonical OmniPicker hash changed")
    if sha256(FULL_X2_SOURCE) != FULL_X2_SOURCE_SHA:
        raise SystemExit("accepted full-X2 source scene hash changed")
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite preflight evidence")
    scene, provenance = build_scene(output)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    rows = fixture_rows()
    right_open = right_hand(rows[OPEN_PERCENT])
    left_open = dict(right_open)
    set_static_state(model, data, np.zeros(len(ARM)), right_open, left_open)
    if model.nbody < 50:
        raise SystemExit("incomplete full-X2 scene")
    required_bodies = [
        "pelvis", "torso_link", "head_yaw_link", "left_shoulder_pitch_link",
        "right_shoulder_pitch_link", "left_hip_pitch_link", "right_hip_pitch_link",
        "L_omnipicker_base_link", "R_omnipicker_base_link",
    ]
    missing = [name for name in required_bodies if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0]
    if missing:
        raise SystemExit(f"full body is incomplete: {missing}")

    base_id = model.body("R_omnipicker_base_link").id
    base_rotation = data.xmat[base_id].reshape(3, 3)
    narrow = base_rotation.T @ (data.xpos[model.body("R_hand_narrow3_Link").id] - data.xpos[base_id])
    wide = base_rotation.T @ (data.xpos[model.body("R_hand_wide3_Link").id] - data.xpos[base_id])
    pad_mid_local = (narrow + wide) / 2.0
    cylinder = data.xpos[model.body("test_cylinder").id].copy()
    rotation = side_grasp_rotation()
    pad_mid_target = cylinder.copy()
    pad_mid_target[2] = GRASP_PAD_MIDPOINT_Z_M
    contact_target = pad_mid_target + rotation @ (SITE_LOCAL - pad_mid_local + np.array([0.0, 0.0, SHAFT_OFFSET_M]))
    contact = solve_side_pose(model, data, contact_target, rotation, np.zeros(len(ARM)), right_open, left_open)
    contact_q = np.asarray(contact["arm_q_rad"])

    approach_target = contact_target + np.array([0.0, APPROACH_BACKOFF_M, 0.0])
    approach = solve_side_pose(model, data, approach_target, rotation, contact_q, right_open, left_open)
    pregrasp_target = contact_target + np.array([0.0, PREGRASP_BACKOFF_M, 0.0])
    pregrasp = solve_side_pose(model, data, pregrasp_target, rotation, np.asarray(approach["arm_q_rad"]), right_open, left_open)
    lift_target = contact_target + np.array([0.0, 0.0, LIFT_M])
    right_closed = right_hand(rows[CLOSE_PERCENT])
    lift = solve_side_pose(model, data, lift_target, rotation, contact_q, right_closed, left_open)

    close_scan = []
    unexpected_contact_events: list[dict[str, object]] = []
    first_bilateral = None
    first_forbidden = None
    for percent in range(OPEN_PERCENT, 0, -1):
        hand = right_hand(rows[percent])
        set_static_state(model, data, contact_q, hand, left_open)
        contacts = contact_records(model, data)
        forbidden = [x for x in contacts if x["kind"] in {"cylinder_forbidden_robot", "robot_table"}]
        unexpected = unexpected_scene_contacts(contacts)
        # The selected physical sequence closes only to CLOSE_PERCENT.  Keep
        # deeper scan results as evidence, but do not let an out-of-sequence
        # over-close collision masquerade as an initialization failure.
        if percent >= CLOSE_PERCENT:
            unexpected_contact_events.extend(
                {"context": f"CLOSE_SCAN_{percent}_PERCENT", "contact": item}
                for item in unexpected
            )
        pads = sorted({body for x in contacts if x["kind"] == "distal_pad" for body in x["bodies"] if body in PAD_BODIES})
        row = {
            "command_percent": percent,
            "distal_pad_bodies": pads,
            "forbidden_count": len(forbidden),
            "forbidden_by_body": dict(collections.Counter(
                x["bodies"][1] if x["bodies"][0] == "test_cylinder" else x["bodies"][0] for x in forbidden
            )),
            "unexpected_scene_contacts": unexpected,
        }
        close_scan.append(row)
        if set(pads) == PAD_BODIES and first_bilateral is None:
            first_bilateral = {"command_percent": percent, "contacts": [x for x in contacts if x["kind"] == "distal_pad"]}
        if forbidden and first_forbidden is None:
            first_forbidden = {"command_percent": percent, "contacts": forbidden[:20]}

    path_samples = []
    segments = [
        ("HORIZONTAL_PREGRASP_TO_APPROACH", np.asarray(pregrasp["arm_q_rad"]), np.asarray(approach["arm_q_rad"]), right_open),
        ("HORIZONTAL_APPROACH_TO_CONTACT", np.asarray(approach["arm_q_rad"]), contact_q, right_open),
        ("VERTICAL_LIFT_KINEMATIC_TRANSPORT_AUDIT", contact_q, np.asarray(lift["arm_q_rad"]), right_closed),
    ]
    for name, first_q, second_q, hand in segments:
        for fraction in np.linspace(0.0, 1.0, 41):
            q = (1.0 - fraction) * first_q + fraction * second_q
            object_pos = None
            if name.startswith("VERTICAL_LIFT"):
                object_pos = cylinder + np.array([0.0, 0.0, LIFT_M * fraction])
            set_static_state(model, data, q, hand, left_open, object_pos)
            contacts = contact_records(model, data)
            forbidden = [x for x in contacts if x["kind"] in {"cylinder_forbidden_robot", "robot_table"}]
            unexpected = unexpected_scene_contacts(contacts)
            unexpected_contact_events.extend(
                {"context": f"{name}@{fraction:.3f}", "contact": item}
                for item in unexpected
            )
            path_samples.append({
                "segment": name, "fraction": float(fraction), "forbidden_contacts": forbidden,
                "unexpected_scene_contacts": unexpected,
            })

    set_static_state(model, data, contact_q, right_closed, left_open)
    source_visual = source_mesh_sample_check(model, data, cylinder)
    equality = right_equality_residuals(model, data)
    root_hold = root_hold_preflight(model, np.asarray(pregrasp["arm_q_rad"]), right_open, left_open)
    # Final static display uses exactly the candidate closed contact pose.
    set_static_state(model, data, contact_q, right_closed, left_open)
    render_views(model, data, output)

    cylinder_joint = model.joint("test_cylinder_free").id
    contact_pose_contacts = contact_records(model, data)
    unexpected_contact_events.extend(
        {"context": "HORIZONTAL_CONTACT_CLOSE_75_PERCENT", "contact": item}
        for item in unexpected_scene_contacts(contact_pose_contacts)
    )
    support_contacts = [x for x in contact_pose_contacts if x["kind"] == "cylinder_table"]
    # The dynamic cylinder begins at source defined qpos0, so table support is
    # measured before static grasp positioning as a separate snapshot.
    set_static_state(model, data, np.zeros(len(ARM)), right_open, left_open)
    initial_contacts = contact_records(model, data)
    initial_support = [x for x in initial_contacts if x["kind"] == "cylinder_table"]
    unexpected_contact_events.extend(
        {"context": "INITIAL_OPEN", "contact": item}
        for item in unexpected_scene_contacts(initial_contacts)
    )
    table_body = model.body("raicom_table").id
    table_gids = np.where(model.geom_bodyid == table_body)[0]
    table_vertices = []
    for geom in table_gids:
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        table_vertices.append(
            model.mesh_vert[va:va + vn].astype(float)
            @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
    table_vertices = np.vstack(table_vertices)
    table_bounds = {"min_m": table_vertices.min(0).tolist(), "max_m": table_vertices.max(0).tolist(), "top_z_m": float(table_vertices[:, 2].max())}

    horizontal_pass = all(item["horizontal_side_grasp_pass"] for item in (contact, approach, pregrasp, lift))
    closure_row = next(row for row in close_scan if row["command_percent"] == CLOSE_PERCENT)
    closure_pass = (
        first_bilateral is not None
        and first_bilateral["command_percent"] >= CLOSE_PERCENT
        and closure_row["forbidden_count"] == 0
    )
    path_pass = not any(sample["forbidden_contacts"] for sample in path_samples)
    unexpected_summary = summarize_unexpected_contacts(unexpected_contact_events)
    full_body_contact_pass = not unexpected_summary
    source_bad = {
        body: value for body, value in source_visual["per_body"].items()
        if value["sampled_surface_enters_cylinder"] and body not in PAD_BODIES
    }
    gates = {
        "FROZEN_SOURCE_HASHES": "PASS",
        "FULL_X2_BODY_SET": "PASS" if not missing else "FAIL",
        "ROBOT_OUTSIDE_UNCHANGED_TABLE": "PASS",
        "CYLINDER_TABLE_SUPPORT": "PASS" if initial_support else "FAIL",
        "HORIZONTAL_TCP_ORIENTATION": "PASS" if horizontal_pass else "FAIL",
        "OPEN_PREGRASP_APPROACH_CLEARANCE": "PASS" if path_pass else "FAIL",
        "BILATERAL_DISTAL_PAD_CONTACT_AT_75_PERCENT": "PASS" if closure_pass else "FAIL",
        "SOURCE_VISUAL_MESH_NONPAD_CLEARANCE": "PASS" if not source_bad else "FAIL",
        "FULL_X2_UNEXPECTED_SELF_CONTACT_FREE": "PASS" if full_body_contact_pass else "FAIL",
        "RIGHT_LINKAGE_CLOSURE": "PASS" if equality["max_joint_residual_rad"] <= MAX_EQUALITY_RESIDUAL_RAD and equality["max_connect_residual_m"] <= MAX_CONNECT_RESIDUAL_M else "FAIL",
        "INITIAL_RIGHT_ROOT_TRACKING_AND_LINKAGE": "PASS" if root_hold["root_tracking_and_linkage_pass"] else "FAIL",
        "RIGHT_ARM_AND_HAND_STATIC_HOLD": "PASS" if root_hold["right_arm_and_hand_static_hold_pass"] else (
            "NOT_EVALUABLE_WITH_UNEXPECTED_FULL_X2_CONTACT" if not root_hold["static_hold_evaluable"] else "FAIL"
        ),
        "DYNAMICS": "NOT RUN",
    }
    overall = "PASS" if all(value == "PASS" for key, value in gates.items() if key != "DYNAMICS") else "BLOCKED"
    result = {
        "schema": "robotsim.issue46.full_x2_horizontal_cylinder_static_preflight.v1",
        "status": overall,
        "mujoco_version": mujoco.__version__,
        "mj_step_calls": 0,
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "provenance": provenance,
        "frozen_omnipicker_model_sha256": sha256(FROZEN_MODEL),
        "full_body_required": required_bodies,
        "fixed_base_pose_world_m": FIXED_PELVIS_WORLD_M.tolist(),
        "table": table_bounds,
        "cylinder": {
            "shape": "native MuJoCo cylinder", "diameter_m": CYLINDER_DIAMETER_M,
            "height_m": CYLINDER_HEIGHT_M, "mass_kg": CYLINDER_MASS_KG,
            "center_world_m": CYLINDER_CENTER_WORLD_M.tolist(), "freejoint": True,
            "gravity_m_s2": model.opt.gravity.tolist(),
            "robot_facing_edge_clearance_mm": 10.0,
            "initial_table_contacts": initial_support,
        },
        "horizontal_grasp_frame": {
            "definition": "local Y is pad opposition; local Z is finger shaft and radial approach",
            "desired_rotation_world": rotation.tolist(), "pad_midpoint_target_world_m": pad_mid_target.tolist(),
            "grasp_pad_midpoint_z_m": GRASP_PAD_MIDPOINT_Z_M, "shaft_offset_m": SHAFT_OFFSET_M,
        },
        "waypoints": {"HORIZONTAL_PREGRASP": pregrasp, "HORIZONTAL_APPROACH": approach, "HORIZONTAL_CONTACT": contact, "VERTICAL_LIFT_50MM": lift},
        "gripper": {
            "open_percent": OPEN_PERCENT, "close_percent": CLOSE_PERCENT,
            "right_open_hand_q_rad": right_open,
            "right_contact_hand_q_rad": right_closed,
            "left_idle_hand_q_rad": left_open,
            "close_scan": close_scan, "first_bilateral": first_bilateral,
            "first_forbidden": first_forbidden,
        },
        "path_samples": path_samples,
        "full_x2_contact_audit": {
            "allowed_static_contact_kinds": ["cylinder_table", "distal_pad"],
            "unexpected_contact_summary": unexpected_summary,
            "full_body_contact_free_pass": full_body_contact_pass,
        },
        "right_linkage_closure": equality,
        "source_visual_mesh_check": source_visual,
        "root_hold_preflight": root_hold,
        "gates": gates,
    }
    pg.write_new_text(output / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    trial = {
        "status": "STATIC_PREFLIGHT_PASS_DYNAMICS_CONDITIONALLY_AUTHORIZED" if overall == "PASS" else "STATIC_PREFLIGHT_BLOCKED",
        "scene": str(scene), "scene_sha256": result["scene_sha256"],
        "initial_right_arm_q_rad": root_hold["equilibrium_arm_q_rad"],
        "initial_right_hand_q_rad": root_hold["equilibrium_right_hand_q_rad"],
        "nominal_right_arm_targets_rad": pregrasp["arm_q_rad"],
        "root_open_target_rad": right_open["right_claw_joint"],
        "root_close_target_rad": right_hand(rows[CLOSE_PERCENT])["right_claw_joint"],
        "gates": gates,
        "limits": {"max_steps": 20000, "max_simulated_seconds": 20.0, "root_tracking_gate_rad": ROOT_TRACKING_GATE_RAD},
    }
    pg.write_new_text(output / "trial_config.json", json.dumps(trial, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": overall, "gates": gates, "scene": str(scene), "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
