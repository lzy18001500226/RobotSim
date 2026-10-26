#!/usr/bin/env python3
"""Trace the accepted full-X2 head collision proxy to its vendor URDF source.

The audit is intentionally static.  It distinguishes a source-variant choice
from an XML conversion error before any collision-policy change is proposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "scripts"))
from agent import path_governance as pg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(text: str | None, count: int, fallback: list[float]) -> np.ndarray:
    if text is None:
        return np.asarray(fallback, dtype=float)
    values = [float(value) for value in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {text!r}")
    return np.asarray(values, dtype=float)


def rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(first, second)), -1.0, 1.0)))))


def urdf_collision(urdf: Path, link_name: str) -> dict[str, object]:
    root = ET.parse(urdf).getroot()
    link = next((item for item in root.findall("link") if item.get("name") == link_name), None)
    if link is None:
        raise RuntimeError(f"missing {link_name} in {urdf}")
    collision = link.find("collision")
    if collision is None:
        raise RuntimeError(f"missing collision for {link_name} in {urdf}")
    origin = collision.find("origin")
    geometry = collision.find("geometry")
    if geometry is None:
        raise RuntimeError("missing collision geometry")
    primitive = next(iter(geometry), None)
    if primitive is None:
        raise RuntimeError("empty collision geometry")
    return {
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
        "link": link_name,
        "origin_xyz_m": floats(None if origin is None else origin.get("xyz"), 3, [0.0, 0.0, 0.0]).tolist(),
        "origin_rpy_rad": floats(None if origin is None else origin.get("rpy"), 3, [0.0, 0.0, 0.0]).tolist(),
        "geometry_tag": primitive.tag,
        "geometry_attributes": dict(primitive.attrib),
    }


def accepted_proxy(scene: Path) -> dict[str, object]:
    xml = ET.parse(scene).getroot()
    body = next((item for item in xml.findall(".//body") if item.get("name") == "head_pitch_link"), None)
    if body is None:
        raise RuntimeError("accepted scene has no head_pitch_link")
    candidates = [
        geom for geom in body.findall("geom")
        if geom.get("type") == "cylinder" and geom.get("contype", "1") != "0"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one active head cylinder, found {len(candidates)}")
    geom = candidates[0]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body("head_pitch_link").id
    active = [
        index for index in np.where(model.geom_bodyid == body_id)[0]
        if int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        and int(model.geom_contype[index]) != 0
    ]
    if len(active) != 1:
        raise RuntimeError(f"compiled scene has {len(active)} active head cylinders")
    index = active[0]
    return {
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "xml": {
            "pos_m": floats(geom.get("pos"), 3, [0.0, 0.0, 0.0]).tolist(),
            "quat_wxyz": floats(geom.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]).tolist(),
            "size": floats(geom.get("size"), 2, []).tolist(),
            "class": geom.get("class"),
            "contype": geom.get("contype"),
            "conaffinity": geom.get("conaffinity"),
        },
        "compiled": {
            "geom_id": int(index),
            "type": int(model.geom_type[index]),
            "size_m": model.geom_size[index].tolist(),
            "pos_m": model.geom_pos[index].tolist(),
            "quat_wxyz": model.geom_quat[index].tolist(),
            "contype": int(model.geom_contype[index]),
            "conaffinity": int(model.geom_conaffinity[index]),
            "mj_step_calls": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accepted-scene", type=Path, required=True)
    parser.add_argument("--full-urdf", type=Path, required=True)
    parser.add_argument("--simple-collision-urdf", type=Path, required=True)
    args = parser.parse_args()
    output = pg.validate_output_path(args.output_dir, require_exists=True, expected_kind="directory", environment="wsl").native_path
    if (output / "result.json").exists():
        raise SystemExit("refusing to overwrite evidence")
    accepted_scene = args.accepted_scene.resolve()
    full_urdf = args.full_urdf.resolve()
    simple_urdf = args.simple_collision_urdf.resolve()
    if any(not path.is_file() for path in [accepted_scene, full_urdf, simple_urdf]):
        raise SystemExit("all source paths must exist")

    full = urdf_collision(full_urdf, "head_pitch_link")
    simple = urdf_collision(simple_urdf, "head_pitch_link")
    accepted = accepted_proxy(accepted_scene)
    simple_xyz = np.asarray(simple["origin_xyz_m"], dtype=float)
    simple_quat = rpy_to_quat_wxyz(np.asarray(simple["origin_rpy_rad"], dtype=float))
    simple_size = np.asarray([
        float(simple["geometry_attributes"]["radius"]),
        float(simple["geometry_attributes"]["length"]) / 2.0,
    ])
    accepted_xml = accepted["xml"]
    equivalent = bool(
        simple["geometry_tag"] == "cylinder"
        and np.allclose(simple_xyz, accepted_xml["pos_m"], atol=1e-9)
        and np.allclose(simple_size, accepted_xml["size"], atol=1e-9)
        and quat_delta_deg(simple_quat, np.asarray(accepted_xml["quat_wxyz"], dtype=float)) <= 0.05
    )
    result = {
        "schema": "robotsim.issue46.head_collision_proxy_provenance.v1",
        "mj_step_calls": 0,
        "accepted_head_proxy": accepted,
        "v1_4_full_urdf_head_collision": full,
        "v1_4_simple_collision_urdf_head_collision": simple,
        "simple_collision_to_accepted_equivalence": {
            "origin_delta_mm": float(1000.0 * np.linalg.norm(simple_xyz - np.asarray(accepted_xml["pos_m"], dtype=float))),
            "size_delta_mm": float(1000.0 * np.linalg.norm(simple_size - np.asarray(accepted_xml["size"], dtype=float))),
            "orientation_delta_deg": quat_delta_deg(simple_quat, np.asarray(accepted_xml["quat_wxyz"], dtype=float)),
            "pass": equivalent,
        },
        "decision": {
            "status": "SOURCE_VARIANT_CONFIRMED_NOT_CONVERSION_TYPO" if equivalent else "UNRESOLVED",
            "meaning": "The accepted active cylinder matches the pinned v1.4 simple-collision URDF definition; changing or excluding it is a collision-model/policy decision, not a controller fix.",
            "full_urdf_difference": "The pinned full OmniPicker URDF uses head_pitch_link.stl as its collision geometry rather than this cylinder.",
        },
    }
    pg.write_new_text(output / "result.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    pg.write_artifact_manifest(output)
    print(json.dumps({"status": result["decision"]["status"], "evidence_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
