#!/usr/bin/env python3
"""Zero-step full-X2 dual OmniPicker human review viewer and evidence capture.

This tool uses the pinned canonical gripper pose samples in a read-only Option D
full-X2 scene. It calls mj_forward for static poses and never calls mj_step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent import path_governance as pg  # noqa: E402


EXPECTED = {
    "source_commit": "575cc6b988f976c23550e0db85aa1e5475d3652d",
    "source_urdf_sha256": "35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d",
    "scene_sha256": "a137a6efc793a16df808e23d764594eba4f3449664fbc5ceca554a210742b10a",
    "canonical_model_sha256": "5e33d30e212ee72271efc3f0a790ebfb46cd31e775c578a10f1a2fb1f33c93e7",
    "visual_frames_sha256": "c53724fe5bfead6f3a11624b18f84c6b8d03c0c7bfbab7f80b1cd6a9b37acb19",
    "source_manifest_sha256": "982d01c39f97d16df7d20c78901f170ce448ff90325dd01428dd2fee6e1a6c17",
    "collision_manifest_sha256": "7f4c5fe6db54a21aacddec58681d5207dfbe5e638ff1884bf1ee9c6ad5b31a3a",
    "portability_sha256": "eb6ddaeef5334d28b71307a2634d4b9db3e6c543cece358edc8d0c1ff9baa9c4",
    "fixture_sha256": "4ce15e41dcfd3088f377d6790ed2c2ee75a9cba3dc8bed11a528dd88589b1e43",
    "left_extended_wrist_sha256": "f379d912d341e6224ec16a62ec9a3c21621e43b6aae1bec0ba93e9a947d7507f",
    "right_extended_wrist_sha256": "94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de",
}
CANONICAL_EVIDENCE_PARENT = Path(
    "/home/lzy18001500226/projects/RobotSim/reviews/issue-46"
)

POSES = (2.0, 50.0, 100.0)
DISPLAY_VISUAL = 0
DISPLAY_COLLISION = 1
DISPLAY_ENVIRONMENT = 2
DISPLAY_LEFT_GRIPPER = 3
DISPLAY_RIGHT_GRIPPER = 4
KNOWN_HEAD_TORSO_BODIES = {"head_pitch_link", "torso_link"}


class ReviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    scene: Path
    fixture: Path
    source_checkout: Path
    canonical_package: Path
    provenance_result: Path
    provenance_invariants: Path


@dataclass(frozen=True)
class Pose:
    command_percent: float
    root_q_rad: float
    narrow2_rad: float
    narrow3_rad: float
    narrow_loop_rad: float
    minimum_five_section_mm: float
    d_gap_mm: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ReviewError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ReviewError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _git_value(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise ReviewError(f"git {' '.join(args)} failed for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_inputs(args: argparse.Namespace) -> Inputs:
    values = Inputs(
        scene=args.scene.resolve(strict=True),
        fixture=args.fixture.resolve(strict=True),
        source_checkout=args.source_checkout.resolve(strict=True),
        canonical_package=args.canonical_package.resolve(strict=True),
        provenance_result=args.provenance_result.resolve(strict=True),
        provenance_invariants=args.provenance_invariants.resolve(strict=True),
    )
    _expect_sha(values.scene, EXPECTED["scene_sha256"], "Option D scene")
    _expect_sha(values.canonical_package / "model.xml", EXPECTED["canonical_model_sha256"], "canonical model")
    _expect_sha(values.canonical_package / "visual_frames.json", EXPECTED["visual_frames_sha256"], "visual frame manifest")
    _expect_sha(values.canonical_package / "source_assets.json", EXPECTED["source_manifest_sha256"], "source asset manifest")
    _expect_sha(values.canonical_package / "collision_proxy_manifest.json", EXPECTED["collision_manifest_sha256"], "collision proxy manifest")
    _expect_sha(values.canonical_package / "portability.json", EXPECTED["portability_sha256"], "portable model manifest")
    _expect_sha(values.fixture, EXPECTED["fixture_sha256"], "canonical static pose fixture")

    source_manifest = json.loads((values.canonical_package / "source_assets.json").read_text(encoding="utf-8"))
    if source_manifest.get("commit") != EXPECTED["source_commit"]:
        raise ReviewError("canonical source manifest does not pin the approved AgiBot commit")
    if source_manifest["required_assets"]["urdf"]["sha256"] != EXPECTED["source_urdf_sha256"]:
        raise ReviewError("canonical source manifest URDF identity changed")
    actual_commit = _git_value(values.source_checkout, "rev-parse", "HEAD")
    if actual_commit != EXPECTED["source_commit"]:
        raise ReviewError(f"source checkout HEAD mismatch: {actual_commit}")
    if _git_value(values.source_checkout, "status", "--porcelain"):
        raise ReviewError("pinned vendor source checkout is dirty")

    required = [source_manifest["required_assets"]["urdf"], *source_manifest["required_assets"]["meshes"]]
    for asset in required:
        source_path = values.source_checkout / asset["path"]
        _expect_sha(source_path, asset["sha256"], f"vendor asset {asset['path']}")
    mesh_root = values.source_checkout / "X2_URDF-v1.4.0" / "meshes"
    _expect_sha(mesh_root / "left_wrist_roll_extend_link.stl", EXPECTED["left_extended_wrist_sha256"], "left extended wrist STL")
    _expect_sha(mesh_root / "right_wrist_roll_extend_link.stl", EXPECTED["right_extended_wrist_sha256"], "right extended wrist STL")

    collision_manifest = json.loads((values.canonical_package / "collision_proxy_manifest.json").read_text(encoding="utf-8"))
    proxy_hashes = {
        part["filename"]: part["sha256"]
        for decomposition in collision_manifest["mesh_decompositions"]
        for part in decomposition["parts"]
    }
    for filename, expected_hash in proxy_hashes.items():
        _expect_sha(values.canonical_package / "collision_generated" / filename, expected_hash, f"generated collision proxy {filename}")
    scene_xml = ET.parse(values.scene).getroot()
    source_paths: dict[str, str] = {}
    proxy_paths: dict[str, str] = {}
    source_root = values.source_checkout.resolve(strict=True)
    proxy_root = (values.canonical_package / "collision_generated").resolve(strict=True)
    for mesh in scene_xml.findall("./asset/mesh"):
        mesh_file = mesh.get("file")
        mesh_name = mesh.get("name") or "unnamed_mesh"
        if not mesh_file:
            raise ReviewError(f"scene mesh has no file path: {mesh_name}")
        candidate = Path(mesh_file)
        if not candidate.is_absolute():
            candidate = values.scene.parent / candidate
        candidate = candidate.resolve(strict=True)
        if candidate.is_relative_to(source_root):
            relative = candidate.relative_to(source_root).as_posix()
            if not relative.startswith("X2_URDF-v1.4.0/meshes/"):
                raise ReviewError(f"scene references an unexpected source path: {relative}")
            if _git_value(values.source_checkout, "ls-files", "--error-unmatch", relative) != relative:
                raise ReviewError(f"scene source mesh is not a tracked asset at the pinned commit: {relative}")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            source_paths[relative] = digest
        elif candidate.is_relative_to(proxy_root):
            filename = candidate.name
            expected_hash = proxy_hashes.get(filename)
            if expected_hash is None:
                raise ReviewError(f"scene collision proxy is absent from the pinned manifest: {filename}")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ReviewError(f"collision proxy hash mismatch: {filename}")
            proxy_paths[filename] = digest
        else:
            raise ReviewError(f"scene mesh is outside the pinned source/proxy roots: {candidate}")
    if not source_paths:
        raise ReviewError("scene has no source mesh assets")

    result = json.loads(values.provenance_result.read_text(encoding="utf-8"))
    source = result.get("source", {})
    status = result.get("status", {})
    if source.get("scene_d_sha256") != EXPECTED["scene_sha256"]:
        raise ReviewError("Option D provenance report does not identify the selected scene")
    if source.get("commit") != EXPECTED["source_commit"]:
        raise ReviewError("Option D provenance report does not identify the pinned source")
    if source.get("gripper_sha256") != EXPECTED["canonical_model_sha256"]:
        raise ReviewError("Option D provenance report does not identify the canonical v0.1 gripper")
    if status.get("D_wrist_gripper") != "PASS_SAMPLED_99_STATES":
        raise ReviewError("Option D wrist/gripper source audit is not the expected qualified baseline")
    if status.get("D_full_scene") != "BLOCKED_PREEXISTING_HEAD_TORSO_CONTACT":
        raise ReviewError("Option D full-scene limitation changed; review the new provenance before use")
    invariants = json.loads(values.provenance_invariants.read_text(encoding="utf-8"))
    variant = invariants.get("variants", {}).get("D_RIGHT_EXT_COLLISION_DIAGNOSTIC", {})
    if variant.get("scene_sha256") != EXPECTED["scene_sha256"]:
        raise ReviewError("Option D scene invariants do not match the selected scene")
    if not variant.get("gripper_child_subtrees_preserved") or not variant.get("all_original_asset_definitions_unchanged"):
        raise ReviewError("Option D invariants do not preserve the original gripper/asset definitions")
    return values


def pose_from_row(row: dict[str, str]) -> Pose:
    return Pose(
        command_percent=float(row["command_percent"]),
        root_q_rad=float(row["right_claw_joint_rad"]),
        narrow2_rad=float(row["narrow2_rad"]),
        narrow3_rad=float(row["narrow3_rad"]),
        narrow_loop_rad=float(row["narrow_loop_rad"]),
        minimum_five_section_mm=float(row["min_five_section_mm"]),
        d_gap_mm=float(row["D_gap_mm"]),
    )


def load_pose_rows(fixture: Path) -> dict[float, Pose]:
    with fixture.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[float, Pose] = {}
    for target in POSES:
        row = next((item for item in rows if abs(float(item["command_percent"]) - target) < 1e-9), None)
        if row is None:
            raise ReviewError(f"canonical pose fixture is missing {target:g}%")
        result[target] = pose_from_row(row)
    return result


def static_joint_values(pose: Pose, prefix: str | None = None) -> dict[str, float]:
    """Map one saved canonical sample to both mirrored X2 hand branches."""
    sides = (("right", "R"), ("left", "L")) if prefix is None else (("right" if prefix == "R" else "left", prefix),)
    values: dict[str, float] = {}
    for root_name, hand_prefix in sides:
        values.update(
            {
                f"{root_name}_claw_joint": pose.root_q_rad,
                f"{hand_prefix}_hand_wide1_joint": -pose.root_q_rad,
                f"{hand_prefix}_hand_narrow2_joint": pose.narrow2_rad,
                f"{hand_prefix}_hand_wide2_joint": -pose.narrow2_rad,
                f"{hand_prefix}_hand_narrow3_joint": pose.narrow3_rad,
                f"{hand_prefix}_hand_wide3_joint": -pose.narrow3_rad,
                f"{hand_prefix}_hand_narrow_loop_joint": pose.narrow_loop_rad,
                f"{hand_prefix}_hand_wide_loop_joint": -pose.narrow_loop_rad,
            }
        )
    return values


def _set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ReviewError(f"required joint is absent from the Option D model: {name}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ReviewError(f"review pose expects a hinge joint: {name}")
    low, high = model.jnt_range[joint_id]
    if model.jnt_limited[joint_id] and not (low - 1e-10 <= value <= high + 1e-10):
        raise ReviewError(f"{name}={value:.9f} violates the compiled source range [{low}, {high}]")
    data.qpos[int(model.jnt_qposadr[joint_id])] = value


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: Pose) -> None:
    values = static_joint_values(pose)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    for name, value in values.items():
        _set_joint(model, data, name, value)
    mujoco.mj_forward(model, data)
    if data.time != 0.0:
        raise ReviewError("static viewer changed simulation time; no dynamic stepping is permitted")


def physical_fingerprint(model: mujoco.MjModel) -> str:
    """Hash physical arrays only; display grouping and RGBA are deliberately excluded."""
    digest = hashlib.sha256()
    names = (
        "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
        "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_dataid", "geom_contype", "geom_conaffinity",
        "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range",
        "qpos0", "eq_type", "eq_obj1id", "eq_obj2id", "eq_data", "actuator_trnid", "actuator_ctrlrange",
        "actuator_forcerange", "opt.gravity", "opt.timestep",
    )
    for name in names:
        value: Any = model
        for part in name.split("."):
            value = getattr(value, part)
        array = np.asarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, ident: int) -> str:
    value = mujoco.mj_id2name(model, kind, int(ident))
    if value:
        return value
    suffix = {
        mujoco.mjtObj.mjOBJ_BODY: "body",
        mujoco.mjtObj.mjOBJ_GEOM: "geom",
        mujoco.mjtObj.mjOBJ_MESH: "mesh",
        mujoco.mjtObj.mjOBJ_JOINT: "joint",
    }.get(kind, "object")
    return f"unnamed_{suffix}_id_{ident}"


def _body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))


def set_review_display_groups(model: mujoco.MjModel) -> dict[str, Any]:
    """Assign render-only groups and colors; do not touch collision masks or geometry."""
    before = physical_fingerprint(model)
    counts = {"visual_body": 0, "collision": 0, "environment": 0, "left_visual": 0, "right_visual": 0}
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = _body_name(model, geom_id)
        collision_enabled = bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        if body_name in {"m0_table", "canonical_bottle"} or geom_name in {"floor", "ground"}:
            group = DISPLAY_ENVIRONMENT
            counts["environment"] += 1
        elif collision_enabled:
            group = DISPLAY_COLLISION
            counts["collision"] += 1
        elif body_name.startswith("L_hand_"):
            group = DISPLAY_LEFT_GRIPPER
            counts["left_visual"] += 1
        elif body_name.startswith("R_hand_"):
            group = DISPLAY_RIGHT_GRIPPER
            counts["right_visual"] += 1
        else:
            group = DISPLAY_VISUAL
            counts["visual_body"] += 1
        model.geom_group[geom_id] = group

        # These RGBA values are presentation-only; source mesh vertices and all
        # physical collision fields remain untouched.
        if group == DISPLAY_COLLISION:
            if body_name in {"left_wrist_roll_link", "right_wrist_roll_link"}:
                model.geom_rgba[geom_id] = (1.0, 0.36, 0.06, 0.56)
            elif body_name in KNOWN_HEAD_TORSO_BODIES:
                model.geom_rgba[geom_id] = (1.0, 0.12, 0.16, 0.36)
            else:
                model.geom_rgba[geom_id] = (0.02, 0.82, 0.92, 0.38)
        elif group == DISPLAY_LEFT_GRIPPER:
            model.geom_rgba[geom_id] = (0.12, 0.68, 0.62, 1.0)
        elif group == DISPLAY_RIGHT_GRIPPER:
            model.geom_rgba[geom_id] = (0.94, 0.43, 0.18, 1.0)

    after = physical_fingerprint(model)
    if before != after:
        raise ReviewError("display setup changed a physical model field")
    if not all(counts[k] > 0 for k in ("collision", "left_visual", "right_visual", "environment")):
        raise ReviewError(f"compiled scene is missing display categories: {counts}")
    return {"display_group_counts": counts, "physical_fingerprint_before": before, "physical_fingerprint_after": after}


def select_display_groups(opt: mujoco.MjvOption, mode: str, focus: str | None = None) -> None:
    opt.geomgroup[:] = 0
    # Environment visuals remain visible in every mode so table and bottle stay
    # in context. They are original geoms from the chosen scene.
    opt.geomgroup[DISPLAY_ENVIRONMENT] = 1
    if mode == "VISUAL_ONLY":
        opt.geomgroup[DISPLAY_VISUAL] = 1
        if focus in (None, "LEFT"):
            opt.geomgroup[DISPLAY_LEFT_GRIPPER] = 1
        if focus in (None, "RIGHT"):
            opt.geomgroup[DISPLAY_RIGHT_GRIPPER] = 1
    elif mode == "COLLISION_ONLY":
        opt.geomgroup[DISPLAY_COLLISION] = 1
    elif mode == "OVERLAY":
        opt.geomgroup[DISPLAY_VISUAL] = 1
        opt.geomgroup[DISPLAY_COLLISION] = 1
        if focus in (None, "LEFT"):
            opt.geomgroup[DISPLAY_LEFT_GRIPPER] = 1
        if focus in (None, "RIGHT"):
            opt.geomgroup[DISPLAY_RIGHT_GRIPPER] = 1
    else:
        raise ReviewError(f"unknown display mode: {mode}")


def state_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = _body_name(model, geom1)
        body2 = _body_name(model, geom2)
        rows.append(
            {
                "body1": body1,
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "body2": body2,
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "distance_m": float(contact.dist),
                "known_preexisting_head_torso_pair": {body1, body2} == KNOWN_HEAD_TORSO_BODIES,
            }
        )
    return rows


def _relative_body_pose(model: mujoco.MjModel, data: mujoco.MjData, base_name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if base_id < 0 or body_id < 0:
        raise ReviewError(f"canonical body mapping is missing: {base_name} / {body_name}")
    base_rotation = data.xmat[base_id].reshape(3, 3)
    position = base_rotation.T @ (data.xpos[body_id] - data.xpos[base_id])
    rotation = base_rotation.T @ data.xmat[body_id].reshape(3, 3)
    return position, rotation


def verify_follower_kinematics(inputs: Inputs, poses: dict[float, Pose]) -> dict[str, Any]:
    full_model = mujoco.MjModel.from_xml_path(str(inputs.scene))
    full_data = mujoco.MjData(full_model)
    canonical_model = mujoco.MjModel.from_xml_path(str(inputs.canonical_package / "model.xml"))
    canonical_data = mujoco.MjData(canonical_model)
    names = (
        "R_hand_narrow1_Link", "R_hand_narrow2_Link", "R_hand_narrow3_Link", "R_hand_narrow_loop_Link",
        "R_hand_wide1_Link", "R_hand_wide2_Link", "R_hand_wide3_Link", "R_hand_wide_loop_Link",
    )
    samples: dict[str, Any] = {}
    for percent, pose in poses.items():
        full_data.qpos[:] = full_model.qpos0
        canonical_data.qpos[:] = canonical_model.qpos0
        full_data.qvel[:] = 0.0
        canonical_data.qvel[:] = 0.0
        for joint, value in static_joint_values(pose).items():
            _set_joint(full_model, full_data, joint, value)
        for joint, value in static_joint_values(pose, "R").items():
            _set_joint(canonical_model, canonical_data, joint, value)
        mujoco.mj_forward(full_model, full_data)
        mujoco.mj_forward(canonical_model, canonical_data)
        per_side: dict[str, Any] = {}
        for side in ("L", "R"):
            base = f"{side}_omnipicker_base_link"
            errors = []
            for canonical_body in names:
                full_body = canonical_body if side == "R" else "L" + canonical_body[1:]
                fp, fr = _relative_body_pose(full_model, full_data, base, full_body)
                cp, cr = _relative_body_pose(canonical_model, canonical_data, "R_omnipicker_base_link", canonical_body)
                errors.append(
                    {
                        "body": full_body,
                        "position_error_m": float(np.linalg.norm(fp - cp)),
                        "rotation_matrix_max_error": float(np.max(np.abs(fr - cr))),
                    }
                )
            per_side[side] = {
                "max_position_error_m": max(item["position_error_m"] for item in errors),
                "max_rotation_matrix_error": max(item["rotation_matrix_max_error"] for item in errors),
                "bodies": errors,
            }
        samples[f"{percent:g}%"] = per_side
    tolerance = 1e-8
    passed = all(
        side["max_position_error_m"] <= tolerance and side["max_rotation_matrix_error"] <= tolerance
        for pose_row in samples.values()
        for side in pose_row.values()
    )
    if not passed:
        raise ReviewError("full-X2 follower body transforms differ from the selected canonical gripper model")
    return {"status": "PASS_STATIC_TRANSFORM_PARITY", "tolerance": tolerance, "samples": samples}


def verify_scene(inputs: Inputs, poses: dict[float, Pose]) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(inputs.scene))
    data = mujoco.MjData(model)
    required_bodies = (
        "pelvis", "torso_link", "head_pitch_link", "left_wrist_roll_link", "right_wrist_roll_link",
        "L_omnipicker_base_link", "R_omnipicker_base_link", "m0_table", "canonical_bottle",
    )
    body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in required_bodies}
    missing = [name for name, ident in body_ids.items() if ident < 0]
    if missing:
        raise ReviewError(f"Option D scene is not the required complete X2 assembly: missing {missing}")
    mesh_names = [_name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)]
    geom_names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    if any("head_pitch" in name.lower() and "coacd" in name.lower() for name in mesh_names + geom_names):
        raise ReviewError("failed head/torso CoACD candidate detected; it is forbidden as the review baseline")
    if any("torso" in name.lower() and "coacd" in name.lower() for name in mesh_names + geom_names):
        raise ReviewError("failed head/torso CoACD candidate detected; it is forbidden as the review baseline")
    for body_name, expected_mesh in (
        ("left_wrist_roll_link", "left_wrist_roll_extend__variant"),
        ("right_wrist_roll_link", "right_wrist_roll_extend__variant"),
    ):
        body_id = body_ids[body_name]
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, expected_mesh)
        attached = [
            i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) == body_id
            and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(model.geom_dataid[i]) == mesh_id
        ]
        has_visual = any(not (model.geom_contype[i] or model.geom_conaffinity[i]) for i in attached)
        has_enabled_collision = any(bool(model.geom_contype[i] or model.geom_conaffinity[i]) for i in attached)
        if mesh_id < 0 or not has_visual or not has_enabled_collision:
            raise ReviewError(f"{body_name} does not use the source extended wrist mesh for both visual and enabled collision geoms")

    display_info = set_review_display_groups(model)
    physical_hash = physical_fingerprint(model)
    pose_report: dict[str, Any] = {}
    for percent, pose in poses.items():
        apply_pose(model, data, pose)
        pose_report[f"{percent:g}%"] = {
            "command_percent": percent,
            "right_root_q_rad": pose.root_q_rad,
            "left_root_q_rad": pose.root_q_rad,
            "narrow2_rad": pose.narrow2_rad,
            "narrow3_rad": pose.narrow3_rad,
            "narrow_loop_rad": pose.narrow_loop_rad,
            "fixture_minimum_five_section_mm": pose.minimum_five_section_mm,
            "fixture_d_gap_mm": pose.d_gap_mm,
            "physics_steps": 0,
            "static_contacts": state_contacts(model, data),
        }
    if physical_fingerprint(model) != physical_hash:
        raise ReviewError("scene validation changed a physical model field")
    kinematics = verify_follower_kinematics(inputs, poses)
    source_manifest = json.loads((inputs.canonical_package / "source_assets.json").read_text(encoding="utf-8"))
    collision_manifest = json.loads((inputs.canonical_package / "collision_proxy_manifest.json").read_text(encoding="utf-8"))
    portability = json.loads((inputs.canonical_package / "portability.json").read_text(encoding="utf-8"))
    provenance_result = json.loads(inputs.provenance_result.read_text(encoding="utf-8"))
    provenance_invariants = json.loads(inputs.provenance_invariants.read_text(encoding="utf-8"))
    scene_meshes = ET.parse(inputs.scene).getroot().findall("./asset/mesh")
    scene_source_meshes: dict[str, str] = {}
    scene_proxy_meshes: dict[str, str] = {}
    source_root = inputs.source_checkout.resolve(strict=True)
    proxy_root = (inputs.canonical_package / "collision_generated").resolve(strict=True)
    for mesh in scene_meshes:
        candidate = Path(mesh.attrib["file"])
        if not candidate.is_absolute():
            candidate = inputs.scene.parent / candidate
        candidate = candidate.resolve(strict=True)
        if candidate.is_relative_to(source_root):
            scene_source_meshes[candidate.relative_to(source_root).as_posix()] = sha256_file(candidate)
        elif candidate.is_relative_to(proxy_root):
            scene_proxy_meshes[candidate.name] = sha256_file(candidate)
    manifest_proxy_count = sum(len(item["parts"]) for item in collision_manifest["mesh_decompositions"])
    if portability.get("collision_proxy_count") != len(scene_proxy_meshes):
        raise ReviewError("portable model proxy count does not match the collision meshes referenced by the Option D scene")
    summary = {
        "schema": "robotsim.issue46.dual_omnipicker_visual_review.v1",
        "scene_sha256": sha256_file(inputs.scene),
        "canonical_model_sha256": sha256_file(inputs.canonical_package / "model.xml"),
        "source_commit": EXPECTED["source_commit"],
        "source_identities": {
            "repository": source_manifest["repository"],
            "commit": source_manifest["commit"],
            "urdf_sha256": source_manifest["required_assets"]["urdf"]["sha256"],
            "source_manifest_sha256": sha256_file(inputs.canonical_package / "source_assets.json"),
            "visual_frames_sha256": sha256_file(inputs.canonical_package / "visual_frames.json"),
            "collision_proxy_manifest_sha256": sha256_file(inputs.canonical_package / "collision_proxy_manifest.json"),
            "portability_manifest_sha256": sha256_file(inputs.canonical_package / "portability.json"),
            "collision_proxy_manifest_generated_parts": manifest_proxy_count,
            "collision_proxy_parts_referenced_by_model": len(scene_proxy_meshes),
            "collision_proxy_parts_not_referenced_by_model": manifest_proxy_count - len(scene_proxy_meshes),
            "pose_fixture_sha256": sha256_file(inputs.fixture),
            "option_d_result_sha256": sha256_file(inputs.provenance_result),
            "option_d_invariants_sha256": sha256_file(inputs.provenance_invariants),
            "left_extended_wrist_sha256": EXPECTED["left_extended_wrist_sha256"],
            "right_extended_wrist_sha256": EXPECTED["right_extended_wrist_sha256"],
            "scene_source_meshes": scene_source_meshes,
            "scene_collision_proxies": scene_proxy_meshes,
            "scene_mesh_asset_entries": len(scene_meshes),
        },
        "mujoco_version": mujoco.__version__,
        "physics_steps": 0,
        "static_mj_forward_only": True,
        "model_counts": {"bodies": model.nbody, "geoms": model.ngeom, "meshes": model.nmesh, "joints": model.njnt, "equalities": model.neq},
        "active_collision_geom_count": int(sum(bool(model.geom_contype[i] or model.geom_conaffinity[i]) for i in range(model.ngeom))),
        "display_only_groups": display_info,
        "physical_model_fingerprint": physical_hash,
        "pose_samples": pose_report,
        "follower_kinematics_parity": kinematics,
        "known_baseline_collision": {
            "bodies": ["head_pitch_link", "torso_link"],
            "status": "KNOWN_PREEXISTING_STATIC_CONTACT",
            "detail": "Option D full-scene audit reports this pair in 99/99 sampled poses, deepest penetration -7.112 mm; this viewer does not resolve or mask it.",
        },
        "model_boundary": {
            "classification": "SIMULATION_ONLY_DIAGNOSTIC",
            "right_wrist_collision": "official right_wrist_roll_extend_link.stl substituted for the official URDF right collision mesh; visual and source gripper assets remain unchanged",
            "bottle": "scene canonical_bottle primitive collision/visual assembly, mass 0.570 kg; static display only",
            "full_x2_dynamics": "NOT_RUN",
            "bottle_grasp": "NOT_RUN",
        },
        "wrist_geom_mapping": {
            "left_wrist_roll_link": {"visual_mesh": "left_wrist_roll_extend__variant", "collision_mesh": "left_wrist_roll_extend__variant"},
            "right_wrist_roll_link": {"visual_mesh": "right_wrist_roll_extend__variant", "collision_mesh": "right_wrist_roll_extend__variant"},
        },
    }
    return model, data, summary


def _camera(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


CAMERA_PRESETS = {
    "front": _camera((0.32, 0.20, 0.86), 2.45, 90.0, -12.0),
    "rear": _camera((0.32, 0.20, 0.86), 2.45, 270.0, -12.0),
    "side": _camera((0.32, 0.20, 0.86), 2.45, 0.0, -10.0),
    "three_quarter": _camera((0.32, 0.20, 0.86), 2.45, 135.0, -16.0),
    "both_grippers": _camera((0.38, 0.515, 0.62), 1.35, 135.0, -10.0),
    "left_wrist": _camera((0.67, 0.515, 0.585), 0.78, 105.0, -12.0),
    "right_wrist": _camera((0.10, 0.515, 0.585), 0.78, 345.0, -12.0),
}


def _scene_option(mode: str, focus: str | None, contacts: bool) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    select_display_groups(option, mode, focus)
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(contacts)
    return option


def _banner(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    bar_h = 66
    canvas = Image.new("RGB", (image.width, image.height + bar_h), (20, 28, 37))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((14, 9), title, fill=(245, 248, 250), font=font)
    draw.text((14, 35), subtitle, fill=(247, 178, 72), font=font)
    return canvas


def _render_one(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, pose: Pose, preset: str, mode: str, focus: str | None = None) -> Image.Image:
    apply_pose(model, data, pose)
    option = _scene_option(mode, focus, contacts=(mode == "OVERLAY" and "wrist" in preset))
    renderer.update_scene(data, camera=CAMERA_PRESETS[preset], scene_option=option)
    frame = Image.fromarray(renderer.render())
    title = f"X2 DUAL OMNIPICKER | {pose.command_percent:g}% STATIC POSE | {preset.upper()} | {mode}"
    subtitle = "ZERO mj_step | OPTION D SIMULATION_ONLY | RIGHT WRIST COLLISION USES OFFICIAL EXTENDED STL DIAGNOSTIC"
    return _banner(frame, title, subtitle)


def _capture(inputs: Inputs, poses: dict[float, Pose], output_parent: Path) -> Path:
    output = pg.create_unique_output_directory(
        str(output_parent),
        prefix="dual-omnipicker-human-review-20261025",
        environment="wsl",
    )
    model, data, summary = verify_scene(inputs, poses)
    run = output.native_path
    # Keep within the reviewed source scene's declared offscreen buffer (640x480).
    renderer = mujoco.Renderer(model, height=480, width=640)
    captures: dict[str, Image.Image] = {}
    try:
        for key, preset in (
            ("full_x2_front_02_visual.png", "front"),
            ("full_x2_rear_02_visual.png", "rear"),
            ("full_x2_side_02_visual.png", "side"),
            ("full_x2_three_quarter_02_visual.png", "three_quarter"),
            ("left_wrist_02_visual.png", "left_wrist"),
            ("right_wrist_02_visual.png", "right_wrist"),
            ("left_wrist_02_collision_overlay.png", "left_wrist"),
            ("right_wrist_02_collision_overlay.png", "right_wrist"),
            ("full_x2_02_collision_only.png", "front"),
        ):
            mode = "COLLISION_ONLY" if "collision_only" in key else "OVERLAY" if "collision_overlay" in key else "VISUAL_ONLY"
            captures[key] = _render_one(renderer, model, data, poses[2.0], preset, mode)
        for percent in (2.0, 50.0, 100.0):
            filename = f"full_x2_front_{int(percent):03d}_visual.png"
            captures[filename] = _render_one(renderer, model, data, poses[percent], "front", "VISUAL_ONLY")
            captures[f"both_grippers_{int(percent):03d}_visual.png"] = _render_one(
                renderer, model, data, poses[percent], "both_grippers", "VISUAL_ONLY"
            )
        for percent in (50.0, 100.0):
            for side in ("left", "right"):
                captures[f"{side}_wrist_{int(percent):03d}_visual.png"] = _render_one(
                    renderer, model, data, poses[percent], f"{side}_wrist", "VISUAL_ONLY"
                )
                captures[f"{side}_wrist_{int(percent):03d}_collision_overlay.png"] = _render_one(
                    renderer, model, data, poses[percent], f"{side}_wrist", "OVERLAY"
                )
        for filename, image in captures.items():
            pg.write_new_bytes(run / filename, _image_bytes(image))
    finally:
        renderer.close()

    comparison = _montage(
        [captures[f"both_grippers_{int(percent):03d}_visual.png"] for percent in (2.0, 50.0, 100.0)],
        ["2% OPENING", "50% OPENING", "100% OPENING"],
    )
    pg.write_new_bytes(run / "opening_comparison_02_50_100.png", _image_bytes(comparison))
    summary["rendered_screenshots"] = sorted([*captures.keys(), "opening_comparison_02_50_100.png"])
    summary["artifact_status"] = "STATIC_REVIEW_CAPTURE_COMPLETE"
    output_paths = pg.describe_output_path(
        str(run), environment="wsl"
    )
    summary["output_paths"] = output_paths
    pg.write_new_text(run / "static_review_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pg.write_new_text(run / "REPORT.md", _report_text(inputs, summary, output_paths))
    pg.write_new_text(run / "REPRODUCE.md", _reproduce_text(inputs, run, output_paths))
    pg.write_artifact_manifest(run)
    return run


def _image_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _montage(images: list[Image.Image], labels: list[str]) -> Image.Image:
    thumb_w = 560
    gap = 12
    header_h = 34
    thumbs = []
    for image, label in zip(images, labels):
        thumb_h = round(image.height * thumb_w / image.width)
        thumbs.append((image.resize((thumb_w, thumb_h)), label))
    height = header_h + max(img.height for img, _ in thumbs) + gap * 2
    result = Image.new("RGB", (thumb_w * len(thumbs) + gap * (len(thumbs) + 1), height), (18, 26, 35))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(thumbs):
        x = gap + index * (thumb_w + gap)
        draw.text((x, 10), label, fill=(242, 244, 246), font=font)
        result.paste(image, (x, header_h))
    return result


def _report_text(inputs: Inputs, summary: dict[str, Any], output_paths: dict[str, str]) -> str:
    s = summary
    identity = s["source_identities"]
    parity_lines = []
    for percent, sides in s["follower_kinematics_parity"]["samples"].items():
        parity_lines.append(
            f"| {percent} | {max(v['max_position_error_m'] for v in sides.values()):.3e} | "
            f"{max(v['max_rotation_matrix_error'] for v in sides.values()):.3e} |"
        )
    return "\n".join(
        [
            "# Full-X2 Dual OmniPicker Static Visual Review",
            "",
            "## Qualification",
            "",
            "- VISUAL_ASSEMBLY_READY: PASS for this static Option D scene and the listed source identities.",
            "- LEFT_WRIST_VISUAL: PASS; official extended left-wrist source mesh is rendered.",
            "- RIGHT_WRIST_VISUAL: PASS; official extended right-wrist source mesh is rendered.",
            "- LEFT_GRIPPER_KINEMATICS: PASS_STATIC_TRANSFORM_PARITY at 2%, 50%, and 100%.",
            "- RIGHT_GRIPPER_KINEMATICS: PASS_STATIC_TRANSFORM_PARITY at 2%, 50%, and 100%.",
            "- COLLISION_OVERLAY_AVAILABLE: PASS; actual enabled collision geoms from the compiled Option D model are displayed.",
            "- FULL_X2_DYNAMIC_PHYSICS: NOT_RUN.",
            "- BOTTLE_GRASP: NOT_RUN.",
            "- HUMAN_VISUAL_REVIEW: PENDING maintainer inspection.",
            "",
            "## Scene and limits",
            "",
            f"- MuJoCo: `{s['mujoco_version']}`; model counts: `{s['model_counts']}`; physics steps: `{s['physics_steps']}`.",
            f"- Option D scene SHA-256: `{s['scene_sha256']}`.",
            f"- Canonical model SHA-256: `{s['canonical_model_sha256']}`; source commit: `{s['source_commit']}`.",
            f"- Full scene mesh audit: `{identity['scene_mesh_asset_entries']}` assets resolved under the pinned source checkout or collision proxy root; `{len(identity['scene_source_meshes'])}` source mesh files and `{len(identity['scene_collision_proxies'])}` proxies hash-checked.",
            f"- Official OmniPicker URDF SHA-256: `{identity['urdf_sha256']}`; source manifest: `{identity['source_manifest_sha256']}`; collision proxy manifest: `{identity['collision_proxy_manifest_sha256']}`.",
            f"- Proxy inventory: `{identity['collision_proxy_manifest_generated_parts']}` generated manifest pieces, `{identity['collision_proxy_parts_referenced_by_model']}` unique meshes referenced by this model, `{identity['collision_proxy_parts_not_referenced_by_model']}` generated pieces not referenced by this compiled scene; portability manifest SHA-256 `{identity['portability_manifest_sha256']}`.",
            f"- Extended wrists: left STL `{identity['left_extended_wrist_sha256']}`; right STL `{identity['right_extended_wrist_sha256']}`.",
            f"- Option D provenance result SHA-256: `{identity['option_d_result_sha256']}`; scene invariants SHA-256: `{identity['option_d_invariants_sha256']}`.",
            "- The displayed command positions are saved canonical candidate samples at 2%, 50%, and 100%; 0% is not exposed.",
            "- The scene's `canonical_bottle` is the existing 0.570 kg primitive bottle assembly; this is static geometry review, not a bottle contact or grasp test.",
            "- Option D is `SIMULATION_ONLY_DIAGNOSTIC`: it uses the official extended right-wrist STL as the right wrist collision mesh. No official source or canonical model was modified.",
            "- The collision overlay uses the model's compiled geoms whose `contype` or `conaffinity` is nonzero. This viewer assigns render-only geom groups and RGBA colors once at startup; switching modes changes only `MjvOption.geomgroup`. Physical masks, joint frames, sizes, equalities, actuators, and inertias remain unchanged.",
            "- The known pre-existing `head_pitch_link` / `torso_link` static contact is retained and marked red; this viewer does not mask or resolve it. The provenance audit records 99/99 samples and a deepest reported penetration of -7.112 mm.",
            "- Wrist source pass does not imply full-scene collision pass; the full-scene audit remains blocked by that head/torso baseline contact.",
            "",
            "## Canonical follower transform parity",
            "",
            "Full-X2 local body transforms relative to each gripper base were compared against the canonical v0.1 standalone model after `mj_forward` (no `mj_step`).",
            "",
            "| Command sample | Maximum position error (m) | Maximum rotation-matrix element error |",
            "|---|---:|---:|",
            *parity_lines,
            "",
            "## Screenshots",
            "",
            "See the PNGs in this directory. `opening_comparison_02_50_100.png` compares both grippers in identical close-up framing; separate full-X2 front/rear/side/three-quarter views are also included.",
            "",
            f"Scene input: `{inputs.scene}`",
            f"Canonical fixture: `{inputs.fixture}`",
            f"WSL_PATH: `{output_paths['WSL_PATH']}`",
            f"WINDOWS_UNC: `{output_paths['WINDOWS_UNC']}`",
            f"OPEN_COMMAND: `{output_paths['OPEN_COMMAND']}`",
        ]
    ) + "\n"


def _reproduce_text(inputs: Inputs, output: Path, output_paths: dict[str, str]) -> str:
    script = REPO_ROOT / "scripts/research/issue46_dual_omnipicker_review_viewer.py"
    command = (
        f"wsl.exe --distribution Ubuntu-22.04 --exec /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python "
        f"{script} --interactive --scene {inputs.scene} --fixture {inputs.fixture} "
        f"--source-checkout {inputs.source_checkout} --canonical-package {inputs.canonical_package} "
        f"--provenance-result {inputs.provenance_result} --provenance-invariants {inputs.provenance_invariants}"
    )
    return "\n".join(
        [
            "# Launch and controls",
            "",
            "Open Windows PowerShell and run this exact command:",
            "",
            "```powershell",
            "& " + command,
            "```",
            "",
            "Mouse: left-drag orbits, right-drag pans, wheel zooms. The viewer is passive and static; the Python loop only calls `mj_forward`.",
            "",
            "| Key | Action |",
            "|---|---|",
            "| `2` | 2% canonical static opening |",
            "| `5` | 50% canonical static opening |",
            "| `1` | 100% canonical static opening |",
            "| `V` | VISUAL_ONLY |",
            "| `C` | COLLISION_ONLY |",
            "| `O` | OVERLAY |",
            "| `L` | left-gripper visual focus |",
            "| `R` | right-gripper visual focus |",
            "| `B` | both grippers |",
            "| `F` / `E` | front / rear full-X2 camera |",
            "| `S` / `T` | side / three-quarter full-X2 camera |",
            "| `G` | paired left/right gripper close-up camera |",
            "| `[` / `]` | left / right wrist close-up camera |",
            "| `P` | toggle static contact-point markers |",
            "",
            "`L` and `R` only filter the visual hand meshes; the mode remains static and contact masks are not changed. The right wrist collision mesh is the disclosed Option D substitution. Do not interpret this review as dynamic physics or grasp evidence.",
            "",
            f"Screenshots and report: `{output}`",
            f"Windows UNC review directory: `{output_paths['WINDOWS_UNC']}`",
            f"Open this folder from PowerShell: `{output_paths['OPEN_COMMAND']}`",
        ]
    ) + "\n"


def _apply_camera(viewer: mujoco.viewer.Handle, preset: str) -> None:
    camera = CAMERA_PRESETS[preset]
    viewer.cam.type = camera.type
    viewer.cam.lookat[:] = camera.lookat
    viewer.cam.distance = camera.distance
    viewer.cam.azimuth = camera.azimuth
    viewer.cam.elevation = camera.elevation


def _set_status_overlay(viewer: mujoco.viewer.Handle, state: dict[str, Any]) -> None:
    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "STATIC KINEMATIC REVIEW - 0 PHYSICS STEPS",
                f"{state['percent']:g}% | {state['mode']} | Option D SIMULATION_ONLY",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "2/5/1: pose   V/C/O: mode   L/R/B: focus   F/E/S/T: camera   G: both wrists   [ ]: wrist   P: contacts",
                "Known unchanged baseline contact: head_pitch_link geom#387 / torso_link geom#62",
            ),
        ]
    )


def _interactive(inputs: Inputs, poses: dict[float, Pose], initial_percent: float = 2.0) -> None:
    model = mujoco.MjModel.from_xml_path(str(inputs.scene))
    data = mujoco.MjData(model)
    display = set_review_display_groups(model)
    physical_hash = display["physical_fingerprint_after"]
    if initial_percent not in POSES:
        raise ReviewError("interactive viewer only permits 2%, 50%, and 100% pose keys")
    state: dict[str, Any] = {"percent": initial_percent, "mode": "OVERLAY", "focus": None, "preset": "three_quarter", "contacts": False}
    commands: queue.SimpleQueue[int] = queue.SimpleQueue()
    apply_pose(model, data, poses[initial_percent])

    def key_callback(keycode: int) -> None:
        commands.put(int(keycode))

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        _apply_camera(viewer, state["preset"])
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        _set_status_overlay(viewer, state)
        while viewer.is_running():
            changed = False
            try:
                while True:
                    key = commands.get_nowait()
                    with viewer.lock():
                        if key == ord("2"):
                            state["percent"] = 2.0
                        elif key == ord("5"):
                            state["percent"] = 50.0
                        elif key == ord("1"):
                            state["percent"] = 100.0
                        elif key in (ord("v"), ord("V")):
                            state["mode"], state["focus"] = "VISUAL_ONLY", None
                        elif key in (ord("c"), ord("C")):
                            state["mode"], state["focus"] = "COLLISION_ONLY", None
                        elif key in (ord("o"), ord("O")):
                            state["mode"], state["focus"] = "OVERLAY", None
                        elif key in (ord("l"), ord("L")):
                            state["mode"], state["focus"] = "VISUAL_ONLY", "LEFT"
                            state["preset"] = "left_wrist"
                        elif key in (ord("r"), ord("R")):
                            state["mode"], state["focus"] = "VISUAL_ONLY", "RIGHT"
                            state["preset"] = "right_wrist"
                        elif key in (ord("b"), ord("B")):
                            state["focus"] = None
                            state["mode"] = "OVERLAY"
                            state["preset"] = "three_quarter"
                        elif key in (ord("f"), ord("F")):
                            state["preset"] = "front"
                        elif key in (ord("e"), ord("E")):
                            state["preset"] = "rear"
                        elif key in (ord("s"), ord("S")):
                            state["preset"] = "side"
                        elif key in (ord("t"), ord("T")):
                            state["preset"] = "three_quarter"
                        elif key in (ord("g"), ord("G")):
                            state["preset"] = "both_grippers"
                        elif key == ord("["):
                            state["preset"] = "left_wrist"
                        elif key == ord("]"):
                            state["preset"] = "right_wrist"
                        elif key in (ord("p"), ord("P")):
                            state["contacts"] = not state["contacts"]
                        else:
                            continue
                        pose = poses[state["percent"]]
                        apply_pose(model, data, pose)
                        select_display_groups(viewer.opt, state["mode"], state["focus"])
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(state["contacts"])
                        _apply_camera(viewer, state["preset"])
                        if physical_fingerprint(model) != physical_hash:
                            raise ReviewError("an interactive render control modified a physical model field")
                        changed = True
            except queue.Empty:
                pass
            if changed:
                _set_status_overlay(viewer, state)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="read-only Option D full-X2 XML")
    parser.add_argument("--fixture", type=Path, required=True, help="canonical PR #92 static pose CSV")
    parser.add_argument("--source-checkout", type=Path, required=True, help="clean pinned AgiBot source checkout")
    parser.add_argument("--canonical-package", type=Path, required=True, help="read-only PR #92 canonical v0.1 package")
    parser.add_argument("--provenance-result", type=Path, required=True, help="Option D audit result.json")
    parser.add_argument("--provenance-invariants", type=Path, required=True, help="Option D scene_variant_invariants.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--capture", action="store_true", help="create governed PNG/report evidence and exit")
    action.add_argument("--preflight-only", action="store_true", help="validate hashes and static poses without opening GUI")
    action.add_argument("--interactive", action="store_true", help="open the passive keyboard-controlled MuJoCo review window")
    parser.add_argument("--output-parent", type=Path, default=CANONICAL_EVIDENCE_PARENT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = load_inputs(args)
    poses = load_pose_rows(inputs.fixture)
    model, data, summary = verify_scene(inputs, poses)
    print(
        f"Option D static review preflight: PASS | MuJoCo {mujoco.__version__} | "
        f"scene={summary['scene_sha256']} | physics_steps=0"
    )
    print(f"Canonical follower transform parity: {summary['follower_kinematics_parity']['status']}")
    print(f"Known baseline static contact: head_pitch_link geom#387 / torso_link geom#62; retained")
    if args.preflight_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.capture:
        run = _capture(inputs, poses, args.output_parent.resolve())
        print(f"STATIC_REVIEW_ARTIFACTS={run}")
        return 0
    try:
        _interactive(inputs, poses)
    except KeyboardInterrupt:
        print("STATIC_REVIEW_VIEWER_CLOSED | physics_steps=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReviewError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"REVIEW_VIEWER_BLOCKED: {exc}") from exc
