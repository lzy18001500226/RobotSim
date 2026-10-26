#!/usr/bin/env python3
"""Bounded source-STL horizontal-cylinder workcell placement calibration."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_REPO = Path("/home/lzy18001500226/projects/RobotSim")
sys.path.insert(0, str(REPO))
from scripts.agent import path_governance as pg  # noqa: E402
from simulation.mujoco.models.agibot_x2 import load_x2_omnipicker  # noqa: E402
from simulation.mujoco.models.agibot_x2.omnipicker_stl_kinematic import apply_stl_kinematic_pose, load_registered_trajectory  # noqa: E402

MODEL_ID = "X2-OmniPicker-STL"
PADS = {"R_hand_narrow3_Link", "R_hand_wide3_Link"}


def load_pre():
    path = REPO / "scripts/research/issue46_stl_cylinder_static_preflight.py"
    spec = importlib.util.spec_from_file_location("pre", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load static preflight")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_xy(root: ET.Element, x_shift: float, y_shift: float) -> None:
    for name in ("raicom_table", "test_cylinder"):
        body = next(item for item in root.findall(".//body") if item.get("name") == name)
        position = [float(value) for value in body.get("pos", "").split()]
        position[0] += x_shift
        position[1] += y_shift
        body.set("pos", " ".join(f"{value:.12g}" for value in position))


def robot_table_contacts(model, data) -> list[dict[str, object]]:
    result = []
    cylinder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    for contact in data.contact:
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, value) or "" for value in bodies)
        if "raicom_table" in names and cylinder not in bodies:
            result.append({"body_names": names, "distance_m": float(contact.dist)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--prior-config", required=True, type=Path)
    parser.add_argument("--vertical-reference-result", required=True, type=Path)
    args = parser.parse_args()
    pre = load_pre()
    vertical = json.loads(args.vertical_reference_result.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_config.read_text(encoding="utf-8"))
    loaded = load_x2_omnipicker(MODEL_ID, repo_root=REPO)
    trajectory = load_registered_trajectory(loaded.package_path)
    state = pre.state_by_phase(prior, "HORIZONTAL_APPROACH")
    arm = np.asarray([vertical["aligned_contact_static"]["arm_q_rad"][name] for name in pre.ARM_JOINTS])
    command_reference = int(vertical["aligned_contact_static"]["highest_safe_bilateral_percent"])
    # Contact midpoint determines the source-preserving table-height correction for a horizontal cylinder.
    vm = mujoco.MjModel.from_xml_path(str(vertical["generated_scene"]["path"]))
    vd = mujoco.MjData(vm)
    pre.apply_prior_state(vm, vd, state, arm)
    apply_stl_kinematic_pose(mujoco, vm, vd, trajectory, float(command_reference))
    cylinder_id = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, "test_cylinder")
    geom_id = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_GEOM, "test_cylinder_collision")
    radius, halfheight = (float(value) for value in vm.geom_size[geom_id, :2])
    pad_positions = [np.asarray(contact.pos) for contact in vd.contact if cylinder_id in (int(vm.geom_bodyid[contact.geom1]), int(vm.geom_bodyid[contact.geom2])) and any((mujoco.mj_id2name(vm,mujoco.mjtObj.mjOBJ_BODY,int(vm.geom_bodyid[g])) or "") in PADS for g in (contact.geom1,contact.geom2))]
    if len(pad_positions) != 2:
        raise RuntimeError("vertical source reference lacks exactly two pad contacts")
    midpoint = np.mean(np.vstack(pad_positions), axis=0)
    source_tabletop = float(vd.xpos[cylinder_id][2] - halfheight)
    table_z_offset = float(midpoint[2] - (source_tabletop + radius))

    location = pg.create_unique_output_directory(EVIDENCE_REPO / "reviews/issue-46", prefix="full-x2-stl-horizontal-workcell-search")
    output = location.native_path
    template = output / "horizontal_template.xml"
    composition = pre.compose_stl_scene(args.source_scene, loaded.model_path, template, horizontal_cylinder=True, table_z_offset_m=table_z_offset)
    root_template = ET.parse(template).getroot()
    scan_results = []
    x_offsets = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)
    y_offsets = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)
    with tempfile.TemporaryDirectory(prefix="issue46-horizontal-search-") as temp:
        scratch = Path(temp) / "candidate.xml"
        for dx in x_offsets:
            for dy in y_offsets:
                root = ET.fromstring(ET.tostring(root_template, encoding="utf-8"))
                set_xy(root, dx, dy)
                ET.ElementTree(root).write(scratch, encoding="utf-8", xml_declaration=True)
                model = mujoco.MjModel.from_xml_path(str(scratch))
                candidate = None
                for command in range(100, -1, -1):
                    item = pre.evaluate_pose(model, trajectory, state, command, arm_override=arm)
                    table_bad = robot_table_contacts(model, mujoco.MjData(model))
                    if item["bilateral_pad_contact"] and not item["nonpad_right_contact_count"] and not table_bad:
                        candidate = item
                        break
                scan_results.append({"table_xy_offset_m": [dx, dy], "candidate": candidate})
    viable = [item for item in scan_results if item["candidate"] is not None]
    best = max(viable, key=lambda item: item["candidate"]["command_percent"]) if viable else None
    result = {
        "schema": "robotsim.issue46.horizontal-cylinder-workcell-placement-search.v1",
        "classification": "KINEMATIC_SIMULATION_ONLY",
        "mj_step_calls": 0,
        "model": {"id": loaded.model_id, "sha256": loaded.model_sha256},
        "horizontal_table_z_offset_m": table_z_offset,
        "horizontal_cylinder_axis": "-Y",
        "composition": composition,
        "grid_offsets_m": {"x": x_offsets, "y": y_offsets},
        "results": scan_results,
        "best": best,
        "gates": {"SOURCE_STL_DUAL_PAD_WORKCELL_PLACEMENT": "PASS" if best else "BLOCKED", "DYNAMICS": "NOT_RUN"},
    }
    pg.write_new_text(output / "result.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pg.write_new_bytes(output / Path(__file__).name, Path(__file__).read_bytes())
    pg.write_artifact_manifest(output)
    print(json.dumps({"output_wsl": location.wsl_path, "best": best, "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
