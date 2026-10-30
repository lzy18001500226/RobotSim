#!/usr/bin/env python3
"""Static table-placement search from the captured physical-standing posture.

This uses the source table and cylinder as one rigid world placement. It never
steps physics or changes source geometry; it rejects table positions that
already overlap the captured X2 standing envelope before IK is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))

from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import load_registered_trajectory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_result(path: Path, expected_sha256: str, *, label: str) -> dict[str, object]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha256(path) != expected_sha256:
        raise ValueError(f"{label} is missing, unsafe, or hash-mismatched")
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(stem: str):
    path = Path(__file__).with_name(stem + ".py")
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_captured_standing_state(model, data, captured_state, captured_commands, shadow) -> None:
    shadow.set_free_pelvis_pose(model, data, dict(captured_state.get("base_pose_world", {})))
    for group, names in (("leg_positions_rad", shadow.LEG_ORDER), ("waist_positions_rad", shadow.WAIST_ORDER)):
        values = captured_state.get(group)
        if not isinstance(values, dict):
            raise ValueError(f"standing capture lacks {group}")
        for name in names:
            shadow.set_initial_joint_inside_limits(model, data, name, float(values[name]))
    arms = captured_commands.get("arm_commands")
    if not isinstance(arms, dict):
        raise ValueError("standing capture lacks arm commands")
    for name in shadow.ARM_ORDER:
        shadow.set_initial_joint_inside_limits(model, data, name, float(arms[name]["position"]))
    trajectory = load_registered_trajectory(REPO / "simulation/mujoco/models/agibot_x2/omnipicker_stl_kinematic")
    for side in ("left", "right"):
        shadow.apply_native_stl_pose(model, data, trajectory, side=side, command_percent=shadow.INITIAL_STL_COMMAND_PERCENT)
    mujoco.mj_forward(model, data)


def candidate_values(low: float, high: float, step: float) -> np.ndarray:
    if not low <= high or not 0.05 <= step <= 0.25:
        raise ValueError("invalid placement grid")
    return np.arange(low, high + step * 0.5, step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-overlay-result", required=True, type=Path)
    parser.add_argument("--approved-object-scene", required=True, type=Path)
    parser.add_argument("--approved-object-scene-sha256", required=True)
    parser.add_argument("--room-source", required=True, type=Path)
    parser.add_argument("--room-source-sha256", required=True)
    parser.add_argument("--floor-source", required=True, type=Path)
    parser.add_argument("--floor-source-sha256", required=True)
    parser.add_argument("--native-stand-result", required=True, type=Path)
    parser.add_argument("--native-stand-result-sha256", required=True)
    parser.add_argument("--x-range", type=float, nargs=2, default=(-0.7, 0.7))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-1.8, -0.5))
    parser.add_argument("--step-m", type=float, default=0.1)
    args = parser.parse_args()
    helper, shadow = load_module("table_grasp_scene_overlay"), load_module("fullbody_shadow_stand_replay")
    inputs = helper.exact_hand_overlay_inputs(args.hand_overlay_result)
    hand_host = inputs["host_scene"]
    approved = helper.require_hashed_file(args.approved_object_scene, args.approved_object_scene_sha256, label="approved object scene")
    room = helper.require_hashed_file(args.room_source, args.room_source_sha256, label="RAICOM room source")
    floor = helper.require_hashed_file(args.floor_source, args.floor_source_sha256, label="RAICOM floor source")
    native = require_result(args.native_stand_result, args.native_stand_result_sha256, label="native standing capture")
    gates = native.get("gates", {})
    captured_state = native.get("native_arm_state_capture", {}).get("result") if isinstance(native.get("native_arm_state_capture"), dict) else None
    captured_commands = native.get("native_stand_command_capture", {}).get("result") if isinstance(native.get("native_stand_command_capture"), dict) else None
    if not isinstance(gates, dict) or any(gates.get(name) != "PASS" for name in ("NATIVE_STAND_STABILITY", "MC_STAND_COMMAND_CAPTURE")) or not isinstance(captured_state, dict) or not isinstance(captured_commands, dict):
        raise SystemExit("native standing capture does not pass required gates")
    room_frame, cylinder, source_geometry = helper.table_frame_and_cylinder(approved, hand_host, room)
    scan_payload, _ = helper.inline_room(hand_host, room, room_frame, cylinder, runtime=False)
    scan_payload, _ = helper.inline_source_floor(scan_payload, floor)
    scan_root = ET.fromstring(scan_payload)
    scan_frame = scan_root.find(".//body[@name='raicom_table_room_frame']")
    if scan_frame is None:
        raise SystemExit("inlined source room has no movable scan frame")
    scan_frame.set("mocap", "true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(scan_root, encoding="unicode"))
    room_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "raicom_table_room_frame")
    room_mocap = int(model.body_mocapid[room_id])
    cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "test_cylinder_free")
    cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    if min(room_id, room_mocap, cylinder_joint, cylinder_id, wrist_id) < 0:
        raise SystemExit("compiled scan scene lacks required room/cylinder/wrist identifiers")
    table_z = float(source_geometry["source_aligned_table_xyz_m"][2])
    source_table = np.asarray(source_geometry["source_aligned_table_xyz_m"], dtype=float)
    source_cylinder = np.asarray(source_geometry["source_aligned_cylinder_xyz_m"], dtype=float)
    source_room_frame = np.asarray(source_geometry["room_frame_xyz_m"], dtype=float)
    candidates: list[dict[str, object]] = []
    for x in candidate_values(*args.x_range, args.step_m):
        for y in candidate_values(*args.y_range, args.step_m):
            requested_table = np.array((x, y, table_z), dtype=float)
            room_translation = requested_table - source_table
            cylinder_world = source_cylinder + room_translation
            data = mujoco.MjData(model)
            apply_captured_standing_state(model, data, captured_state, captured_commands, shadow)
            data.mocap_pos[room_mocap] = source_room_frame + room_translation
            qadr = int(model.jnt_qposadr[cylinder_joint])
            data.qpos[qadr:qadr + 3] = cylinder_world
            data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(model, data)
            candidates.append({
                "table_world_xyz_m": requested_table.tolist(),
                "room_frame_xyz_m": (source_room_frame + room_translation).tolist(),
                "cylinder_world_xyz_m": cylinder_world.tolist(),
                "cylinder_table_translation_xyz_m": room_translation.tolist(),
                "right_wrist_to_cylinder_m": float(np.linalg.norm(data.xpos[wrist_id] - data.xpos[cylinder_id])),
                "table_robot_contacts": shadow.table_robot_contacts(model, data),
                "joint_limit_violations": shadow.limited_joint_violations(model, data),
            })
    valid = [candidate for candidate in candidates if not candidate["table_robot_contacts"] and not candidate["joint_limit_violations"]]
    valid.sort(key=lambda item: float(item["right_wrist_to_cylinder_m"]))
    output = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="raicom-standing-table-placement-search").native_path
    best_scene = None
    if valid:
        room_frame, cylinder, _ = helper.table_frame_and_cylinder(approved, hand_host, room, table_world_override=np.asarray(valid[0]["table_world_xyz_m"], dtype=float))
        payload, _ = helper.inline_room(hand_host, room, room_frame, cylinder, runtime=False)
        payload, _ = helper.inline_source_floor(payload, floor)
        best_scene = output / "best_standing_table_scene.xml"
        pg.write_new_bytes(best_scene, payload)
    result = {
        "schema": "robotsim.issue46.raicom-standing-table-placement-search.v2",
        "classification": "SIMULATION_ONLY_STATIC_STANDING_TABLE_PLACEMENT_SEARCH",
        "mj_step_calls": 0,
        "sources": {"hand_overlay_result": {"path": str(args.hand_overlay_result), "sha256": sha256(args.hand_overlay_result)}, "approved_object_scene": {"path": str(approved), "sha256": sha256(approved)}, "room": {"path": str(room), "sha256": sha256(room)}, "floor": {"path": str(floor), "sha256": sha256(floor)}, "native_stand_capture": {"path": str(args.native_stand_result), "sha256": sha256(args.native_stand_result)}},
        "scan": {"x_range_m": args.x_range, "y_range_m": args.y_range, "step_m": args.step_m, "candidate_count": len(candidates), "collision_free_count": len(valid), "best_candidates": valid[:20]},
        "best_scene": None if best_scene is None else {"path": str(best_scene), "sha256": sha256(best_scene)},
        "gates": {"STATIC_SCAN_COMPILED": "PASS", "STANDING_STATE_APPLIED": "PASS", "COLLISION_FREE_TABLE_STANCE": "PASS" if valid else "FAIL", "IK": "NOT_RUN", "NATIVE_STANDING_GRASP": "NOT_RUN"},
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    manifest = pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": str(output), "manifest_wsl": manifest["manifest"]["WSL_PATH"], "best": valid[:3], "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
